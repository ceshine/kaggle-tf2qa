import glob
from pathlib import Path

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from pytorch_helper_bot.bot import batch_to_device

from .bot import BasicQABot, ShortAnswerAccuracy
from .loss import BasicQALoss
from .models import BasicBert
from .train import get_data, MODEL_NAME, MovingAverageStatsTrackerCallback

CACHE_DIR = Path("cache/")


def main(
    pattern: str = "cache/train/train_*.jl",
    model_path: str = "cache/first_model/",
    max_q_len: int = 128,
    max_ex_len: int = 350,
    batch_size: int = 4,
    sample_negatives: float = 1.0
):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BasicBert.load(model_path).cuda()
    _, train_loader, valid_ds, valid_loader = get_data(
        tokenizer, pattern, max_q_len, max_ex_len,
        batch_size, sample_negatives
    )

    print("Training set:")
    print("_" * 20)
    cnt = 0
    for batch, targets in train_loader:
        with torch.no_grad():
            output = model(batch_to_device([batch], "cuda")[0])
        sa_pred = torch.argmax(
            output["logit_sa"].cpu(), dim=1)
        print(targets[targets[:, 0] == 1, 1])
        print(sa_pred[targets[:, 0] == 1, 0])
        print("-" * 20)
        print(targets[targets[:, 0] == 1, 2])
        print(sa_pred[targets[:, 0] == 1, 1])
        print("=" * 20)
        cnt += 1
        if cnt == 5:
            break

    print("Validation set:")
    print("_" * 20)
    cnt = 0
    for batch, targets in valid_loader:
        with torch.no_grad():
            output = model(batch_to_device([batch], "cuda")[0])
        sa_pred = torch.argmax(
            output["logit_sa"].cpu(), dim=1)
        print(targets[targets[:, 0] == 1, 1])
        print(sa_pred[targets[:, 0] == 1, 0])
        print("-" * 20)
        print(targets[targets[:, 0] == 1, 2])
        print(sa_pred[targets[:, 0] == 1, 1])
        print("=" * 20)
        print(ShortAnswerAccuracy()(targets, {"sa": sa_pred})[-1])
        cnt += 1
        if cnt == 5:
            break

    valid_ds.sample_negatives = 0.5
    bot = BasicQABot(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        log_dir=CACHE_DIR / "logs/",
        clip_grad=10.,
        optimizer=None,
        echo=True,
        criterion=BasicQALoss(0.5),
        callbacks=[
            MovingAverageStatsTrackerCallback(
                avg_window=400,
                log_interval=200
            )
        ],
        pbar=True,
        use_tensorboard=False,
        use_amp=False,
        gradient_accumulation_steps=1,
        metrics=()
    )
    metrics = bot.eval(valid_loader)
    bot.run_eval_ends_callbacks(metrics)


if __name__ == '__main__':
    fire.Fire(main)
