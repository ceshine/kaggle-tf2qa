from transformers import BertTokenizer
from torch.utils.data import DataLoader

from fragile.dataset import (
    QADataset, collate_example_for_training,
    collate_example_for_inference
)


def inspect_short_answers(samples=10):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    ds = QADataset(["cache/train_0.jl"], tokenizer)
    cnt = 0
    for i, example in enumerate(ds):
        if example.short_answer_start > 0:
            print(
                i, cnt,
                tokenizer.decode(example.input_ids[
                    example.short_answer_start:example.short_answer_end
                ].numpy())
            )
            cnt += 1
            if cnt == samples:
                break


def inspect_collate_for_training(samples=2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    ds = QADataset(["cache/train_0.jl"], tokenizer)
    loader = DataLoader(
        ds, collate_fn=collate_example_for_training, batch_size=5)
    cnt = 0
    for i, (batch, labels) in enumerate(loader):
        if labels.sum(dim=0)[1] > 0:
            print("=" * 20)
            print(i, batch.keys())
            print(batch["input_ids"].size())
            print((batch["input_ids"] > 0).sum(dim=1))
            print("-" * 20)
            print(batch["input_mask"].size())
            print(batch["input_mask"].sum(dim=1))
            print("-" * 20)
            print(batch["token_type_ids"].size())
            print((batch["token_type_ids"] == 0).sum(dim=1))
            print("-" * 20)
            print(batch["sa_mask"].size())
            print((batch["sa_mask"] == 0).sum(dim=1))
            print("-" * 20)
            print(labels.size())
            print(labels.numpy())
            if cnt == samples:
                break
            cnt += 1


def inspect_collate_for_inference(samples=2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    ds = QADataset(["cache/train_0.jl"], tokenizer, is_test=True)
    loader = DataLoader(
        ds, collate_fn=collate_example_for_inference, batch_size=5)
    cnt = 0
    previous_id = None
    for i, (batch, metas) in enumerate(loader):
        if metas["example_id"][0] != previous_id:
            print("=" * 20)
            print(i, batch.keys())
            print(batch["input_ids"].size())
            print((batch["input_ids"] > 0).sum(dim=1))
            print("-" * 20)
            print(batch["input_mask"].size())
            print(batch["input_mask"].sum(dim=1))
            print("-" * 20)
            print(batch["token_type_ids"].size())
            print((batch["token_type_ids"] == 0).sum(dim=1))
            print("-" * 20)
            print(batch["sa_mask"].size())
            print((batch["sa_mask"] == 0).sum(dim=1))
            print("-" * 20)
            print(metas["example_id"])
            print(metas["tok_to_orig"][0][:10])
            print(metas["starts_at"])
            print(metas["ends_at"])
            print(metas["offset"])
            if cnt == samples:
                break
            cnt += 1
            previous_id = metas["example_id"][0]


if __name__ == "__main__":
    # inspect_short_answers()
    # inspect_collate_for_training()
    inspect_collate_for_inference()
