from transformers import BertTokenizer
from torch.utils.data import DataLoader

from fragile.dataset import QADataset, collate_example_for_training


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
    for i, (batch, labels) in enumerate(loader):
        print("=" * 20)
        print(batch.keys())
        print(batch["input_ids"].size())
        print("-" * 20)
        print(batch["input_mask"].size())
        print(batch["input_mask"].sum(dim=1))
        print("-" * 20)
        print(batch["token_type_ids"].size())
        print(batch["token_type_ids"].sum(dim=1))
        print("-" * 20)
        print(batch["sa_mask"].size())
        print(batch["sa_mask"].sum(dim=1))
        print("-" * 20)
        print(labels.size())
        print(labels.numpy())
        if i + 1 == samples:
            break


if __name__ == "__main__":
    # inspect_short_answers()
    inspect_collate_for_training()
