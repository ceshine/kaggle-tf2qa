import glob

import fire
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from .dataset import QADataset, collate_example_for_inference
from .preprocessing import preprocess
from .train import MODEL_NAME, CACHE_DIR
from .bot import BasicQABot
from .models import BasicBert


def main(
    model_path: str = "cache/first_model/",
    run_preprocess: bool = False,
    file_pattern: str = '/tmp/test/test_%d.jl',
    max_q_len: int = 128,
    max_ex_len: int = 350,
    batch_size: int = 4,
    no_answer_threshold: float = 0.5
):
    if run_preprocess is True:
        print("Prepropcessing..")
        preprocess(
            'data/simplified-nq-test.jsonl',
            output_pattern=file_pattern, tokenizer_model=model_path,
            has_annotations=False, chunk_size=500, write_per_chunk=2
        )
    tokenizer = BertTokenizer.from_pretrained(model_path)
    file_paths = np.asarray(
        [x for x in glob.glob(file_pattern.replace("%d", "*"))])
    test_ds = QADataset(
        file_paths, tokenizer, seed=42, is_test=True,
        max_question_length=max_q_len, max_example_length=max_ex_len,
        sample_negatives=1.
    )
    test_loader: DataLoader = DataLoader(
        test_ds, collate_fn=collate_example_for_inference,
        batch_size=batch_size, num_workers=0
    )
    model = BasicBert.load(model_path).cuda()
    bot = BasicQABot(
        model=model,
        train_loader=None,
        valid_loader=None,
        log_dir=CACHE_DIR / "logs/",
        clip_grad=10.,
        optimizer=None,
        echo=True,
        criterion=None,
        callbacks=[],
        pbar=True,
        use_tensorboard=False,
        use_amp=False
    )
    df = bot.predict(test_loader)
    df.sort_values("type_0_prob", ascending=True, inplace=True)
    df_top = df.groupby("example_id").head(1).copy()
    df_top["example_id"] = df_top["example_id"].astype("str")
    results = []
    for _, row in df_top.iterrows():
        if row["type_0_prob"] > no_answer_threshold:
            results.append((
                f"{row['example_id']}_long", ""
            ))
            results.append((
                f"{row['example_id']}_short", ""
            ))
            continue
        results.append((
            f"{row['example_id']}_long", f"{row['long_start']}:{row['long_end']}"
        ))
        if row["type_1_prob"] > row["type_2_prob"] and row["type_1_prob"] > row["type_3_prob"]:
            if row["short_start_hat"] == 0 or row["short_end_hat"] == 0:
                results.append((
                    f"{row['example_id']}_short", ""
                ))
            else:
                results.append((
                    f"{row['example_id']}_short",
                    f'{row["short_start_hat"] + row["long_start"]}:{row["short_end_hat"] + row["long_start"]}'
                ))
        elif row["type_2_prob"] > row["type_1_prob"] and row["type_2_prob"] > row["type_3_prob"]:
            results.append((
                f"{row['example_id']}_short", "NO"
            ))
        else:
            results.append((
                f"{row['example_id']}_short", "YES"
            ))
    df_sub = pd.DataFrame(results, columns=["example_id", "PredictionString"])
    df_sub.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    fire.Fire(main)
