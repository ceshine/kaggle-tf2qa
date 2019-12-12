import glob
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit
from transformers import BertTokenizer
from pytorch_helper_bot import (
    MovingAverageStatsTrackerCallback, LearningRateSchedulerCallback,
    MultiStageScheduler, LinearLR, CheckpointCallback,
    AdamW, TelegramCallback, StepwiseLinearPropertySchedulerCallback
)

from .bot import BasicQABot
from .loss import BasicQALoss
from .models import BasicBert
from .dataset import QADataset, collate_example_for_training

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

CACHE_DIR = Path("cache/")
NO_DECAY = [
    'LayerNorm.weight', 'LayerNorm.bias'
]
MODEL_NAME = "bert-base-cased"

torch.multiprocessing.set_sharing_strategy('file_system')


def count_parameters(parameters):
    return int(np.sum(list(p.numel() for p in parameters)))


def get_optimizer(model, lr):
    # print([name for name, _ in model.named_parameters()])
    params = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in NO_DECAY)],
            'weight_decay': 0.1
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in NO_DECAY)],
            'weight_decay': 0
        }
    ]
    for i, group in enumerate(params):
        print(
            f"# of parameters in group {i}: {count_parameters(group['params']):,d}")
    return AdamW(params, lr=lr)


def get_data(tokenizer, pattern, max_q_len, max_ex_len, batch_size, sample_negatives):
    file_paths = np.asarray([x for x in glob.glob(pattern)])
    ss = ShuffleSplit(n_splits=1, test_size=1)
    train_index, valid_index = next(ss.split(file_paths))
    train_paths, test_paths = file_paths[train_index], file_paths[valid_index]

    train_ds = QADataset(
        train_paths, tokenizer, seed=42, sample_negatives=sample_negatives,
        max_question_length=max_q_len, max_example_length=max_ex_len
    )
    train_loader: DataLoader = DataLoader(
        train_ds, collate_fn=collate_example_for_training,
        batch_size=batch_size, num_workers=2
    )
    valid_ds = QADataset(
        test_paths, tokenizer, seed=42, is_test=True,
        max_question_length=max_q_len, max_example_length=max_ex_len,
        sample_negatives=sample_negatives
    )
    valid_loader: DataLoader = DataLoader(
        valid_ds, collate_fn=collate_example_for_training,
        batch_size=batch_size, num_workers=1
    )
    return train_ds, train_loader, valid_ds, valid_loader


class Trainer:

    def _setup(
        self, pattern: str, max_q_len: int, max_ex_len: int, batch_size: int,
        lr: float = 3e-4, sample_negatives: float = 1.0, use_amp: bool = False
    ):
        assert use_amp is False or APEX_AVAILABLE is True
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        train_ds, train_loader, valid_ds, valid_loader = get_data(
            tokenizer, pattern, max_q_len, max_ex_len,
            batch_size, sample_negatives
        )

        model = BasicBert(MODEL_NAME).cuda()
        optimizer = get_optimizer(model, lr)
        if use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, "O1"
            )
        return (
            train_ds, train_loader, valid_ds, valid_loader,
            model, optimizer
        )

    def resume(
        self, checkpoint_path: str,
        pattern: str = "cache/train/train_*.jl", max_q_len: int = 128,
        max_ex_len: int = 350, batch_size: int = 4,
        lr: float = 3e-4, sample_negatives: float = 1.0,
        checkpoint_interval: int = 3000, use_amp: bool = False
    ):
        (
            train_ds, train_loader, _, valid_loader,
            model, optimizer
        ) = self._setup(
            pattern, max_q_len, max_ex_len, batch_size, lr, sample_negatives, use_amp
        )
        bot = BasicQABot.load_checkpoint(
            checkpoint_path, train_loader, valid_loader,
            model, optimizer
        )
        assert bot.use_amp == use_amp
        checkpoints: Optional[CheckpointCallback] = None
        for callback in bot.callbacks:
            if isinstance(callback, CheckpointCallback):
                checkpoints = callback
                break
        for callback in bot.callbacks:
            if isinstance(callback, StepwiseLinearPropertySchedulerCallback):
                callback.target_obj = train_ds
                break
        if checkpoints:
            # We can reset the checkpoints
            checkpoints.reset(ignore_previous=True)
        bot.train(checkpoint_interval=checkpoint_interval)
        if checkpoints:
            bot.load_model(checkpoints.best_performers[0][1])
            checkpoints.remove_checkpoints(keep=0)
        bot.model.save(CACHE_DIR / "final/")

    def train(
        self, pattern: str = "cache/train/train_*.jl", max_q_len: int = 128,
        max_ex_len: int = 350, batch_size: int = 4, n_steps: int = 20000,
        lr: float = 3e-4, grad_accu: int = 1, sample_negatives: float = 1.0,
        log_freq: int = 200, checkpoint_interval: int = 3000, use_amp: bool = False
    ):
        (
            train_ds, train_loader, _, valid_loader,
            model, optimizer
        ) = self._setup(
            pattern, max_q_len, max_ex_len, batch_size, lr, sample_negatives, use_amp
        )
        checkpoints = CheckpointCallback(
            keep_n_checkpoints=1,
            checkpoint_dir=CACHE_DIR / "model_cache/",
            monitor_metric="loss"
        )
        lr_durations = [
            int(n_steps*0.2),
            int(np.ceil(n_steps*0.8))
        ]
        break_points = [0] + list(np.cumsum(lr_durations))[:-1]
        callbacks = [
            MovingAverageStatsTrackerCallback(
                avg_window=log_freq * 2,
                log_interval=log_freq
            ),
            LearningRateSchedulerCallback(
                MultiStageScheduler(
                    [
                        LinearLR(optimizer, 0.01, lr_durations[0]),
                        LinearLR(optimizer, 0.001,
                                 lr_durations[1], upward=False)
                        # CosineAnnealingLR(optimizer, lr_durations[1])
                    ],
                    start_at_epochs=break_points
                )
            ),
            checkpoints,
            TelegramCallback(
                token="559760930:AAGOgPA0OlqlFB7DrX0lyRc4Di3xeixdNO8",
                chat_id="213781869", name="QABot",
                report_evals=True
            ),
            StepwiseLinearPropertySchedulerCallback(
                train_ds, "sample_negatives", 0.02, 0.5,
                5000, int(n_steps * 0.95), log_freq=log_freq*4
            )
        ]
        bot = BasicQABot(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            log_dir=CACHE_DIR / "logs/",
            clip_grad=10.,
            optimizer=optimizer,
            echo=True,
            criterion=BasicQALoss(
                0.5,
                log_freq=log_freq*4,
                alpha=0.01
            ),
            callbacks=callbacks,
            pbar=True,
            use_tensorboard=False,
            gradient_accumulation_steps=grad_accu,
            metrics=(),
            use_amp=use_amp
        )
        bot.logger.info("train batch size: %d", train_loader.batch_size)
        bot.train(
            total_steps=n_steps,
            checkpoint_interval=checkpoint_interval
        )
        bot.load_model(checkpoints.best_performers[0][1])
        checkpoints.remove_checkpoints(keep=0)
        bot.model.save(CACHE_DIR / "final/")


if __name__ == '__main__':
    fire.Fire(Trainer)
