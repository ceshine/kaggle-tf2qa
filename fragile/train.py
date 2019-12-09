import glob
from pathlib import Path

import fire
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit
from transformers import BertTokenizer
from pytorch_helper_bot import (
    MovingAverageStatsTrackerCallback, LearningRateSchedulerCallback,
    MultiStageScheduler, LinearLR, CheckpointCallback,
    AdamW
)

from .bot import BasicQABot
from .loss import BasicQALoss
from .models import BasicBert
from .dataset import QADataset, collate_example_for_training

CACHE_DIR = Path("cache/")
NO_DECAY = [
    'LayerNorm.weight', 'LayerNorm.bias'
]
MODEL_NAME = "bert-base-cased"


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
        sample_negatives=sample_negatives,
    )
    valid_loader: DataLoader = DataLoader(
        valid_ds, collate_fn=collate_example_for_training,
        batch_size=batch_size, num_workers=0
    )
    return train_loader, valid_loader


def main(
    pattern: str = "cache/train/train_*.jl", max_q_len: int = 128,
    max_ex_len: int = 350, batch_size: int = 4, n_steps: int = 20000,
    lr: float = 3e-4, grad_accu: int = 1, sample_negatives: float = 1.0
):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_loader, valid_loader = get_data(
        tokenizer, pattern, max_q_len, max_ex_len,
        batch_size, sample_negatives
    )

    model = BasicBert(MODEL_NAME).cuda()
    optimizer = get_optimizer(model, lr)

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
            avg_window=400,
            log_interval=200
        ),
        LearningRateSchedulerCallback(
            MultiStageScheduler(
                [
                    LinearLR(optimizer, 0.01, lr_durations[0]),
                    LinearLR(optimizer, 0.001, lr_durations[1], upward=False)
                    # CosineAnnealingLR(optimizer, lr_durations[1])
                ],
                start_at_epochs=break_points
            )
        ),
        checkpoints
    ]
    bot = BasicQABot(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        log_dir=CACHE_DIR / "logs/",
        clip_grad=10.,
        optimizer=optimizer,
        echo=True,
        criterion=BasicQALoss(5),
        callbacks=callbacks,
        pbar=True,
        use_tensorboard=False,
        use_amp=False,
        gradient_accumulation_steps=grad_accu,
        metrics=()
    )
    bot.logger.info("train batch size: %d", train_loader.batch_size)
    bot.train(
        total_steps=n_steps,
        checkpoint_interval=1000
    )
    bot.load_model(checkpoints.best_performers[0][1])
    checkpoints.remove_checkpoints(keep=0)
    bot.model.save(CACHE_DIR / "final/")


if __name__ == '__main__':
    fire.Fire(main)
