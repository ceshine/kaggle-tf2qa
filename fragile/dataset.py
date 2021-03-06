import random
from typing import Sequence, Dict, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .data_structures import TokenizedCandidate, Example


class QADataset(IterableDataset):
    def __init__(
            self, file_paths, tokenizer,
            seed=939, is_test: bool = False,
            max_question_length=128,
            max_example_length=400,
            sample_negatives=1.,
            debug=False
    ):
        super().__init__()
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.seed = seed
        self.debug = debug
        self.max_question_length = max_question_length
        self.max_example_length = max_example_length
        self.is_test = is_test
        self.sample_negatives = sample_negatives
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.sep_token])[0]
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.pad_token])[0]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            seed = self.seed
        else:  # in a worker process
               # split workload
            if worker_info.num_workers > 1 and self.is_test:
                raise ValueError("Validation cannot have num_workers > 1!")
            seed = self.seed + worker_info.id * 10
        if self.is_test is False:
            # traininig mode
            np.random.seed(seed)
            np.random.shuffle(self.file_paths)
        return self.generator(seed)

    def generator(self, seed):
        cnt, file_cnt = 0, 0
        while True:
            for chunk_path in self.file_paths:
                file_cnt += 1
                print(
                    f"loading {chunk_path}... (loaded {file_cnt} files) is_test: {self.is_test}"
                )
                chunk = joblib.load(chunk_path)
                if self.is_test is False:
                    # shuffle only in training
                    np.random.shuffle(chunk)
                for eid, qtokens, candidates in chunk:
                    for example in self.prepare_one_example(
                            eid, qtokens, candidates):
                        yield example
                    cnt += 1
                    # if self.is_test is True and cnt == 600:
                    #     # to make validation faster
                    #     break
            if self.is_test is True:
                break
            np.random.shuffle(self.file_paths)

    def prepare_one_example(self, eid: int, qtokens: Sequence[int], candidates: Sequence[TokenizedCandidate]):
        # truncate qtokens
        qtokens = list(qtokens[:self.max_question_length])
        np.random.shuffle(candidates)
        for cand in candidates:
            if cand.answer_type == 0 and self.sample_negatives < 1 and self.sample_negatives > 0:
                if random.random() > self.sample_negatives:
                    continue
            index_tokens = self.tokenizer.encode(
                str(cand.filtered_index), add_special_tokens=False)
            # 1 [CLS], 2 [SEP]'s
            max_cand_len = (
                self.max_example_length -
                len(qtokens) - 3 - len(index_tokens)
            )
            cand_offset = len(qtokens) + 2 + len(index_tokens)
            cand_tokens = list(cand.token_ids[:max_cand_len])
            overall_tokens = torch.tensor(
                [self.cls_token_id] + qtokens + [self.sep_token_id] +
                index_tokens + cand_tokens + [self.sep_token_id],
                dtype=torch.long
            )
            input_ids = torch.zeros(
                self.max_example_length, dtype=torch.long) + self.pad_token_id
            input_ids[:overall_tokens.size(0)] = overall_tokens
            input_mask = torch.zeros(self.max_example_length, dtype=torch.long)
            input_mask[:overall_tokens.size(0)] = 1
            token_type_ids = torch.zeros(
                self.max_example_length, dtype=torch.long)
            token_type_ids[cand_offset - len(index_tokens):] = 1
            # 1 indicates not part of the short answer:
            sa_mask = torch.ones(self.max_example_length, dtype=torch.bool)
            sa_mask[cand_offset:(overall_tokens).size(0)-1] = False
            sa_mask[0] = 0  # point to [CLS] when no answer exists
            assert (
                (overall_tokens).size(0) - 1 - cand_offset
                == len(cand_tokens)
            )

            # building short answer
            short_answer = self._filter_short_answers(cand, len(cand_tokens))
            if short_answer:
                # pick the first one
                short_answer = sorted(short_answer, key=lambda x: x[0])[0]

            yield Example(
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                sa_mask=sa_mask,
                answer_type=cand.answer_type,
                short_answer_start=(
                    short_answer[0] + cand_offset if short_answer else 0
                ),
                short_answer_end=(
                    short_answer[1] + cand_offset if short_answer else 0
                ),
                example_id=eid,
                tok_to_orig=cand.tok_to_orig,
                starts_at=cand.start_token,
                ends_at=cand.end_token,
                offset=cand_offset
            )

    def _filter_short_answers(self, candidate: TokenizedCandidate, max_answer_length: int):
        results = []
        for sans in candidate.short_answers:
            if sans[1] < max_answer_length:
                results.append(sans)
        return results


def collate_example_for_training(
    batch: Sequence[Example]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    targets = torch.stack([
        torch.tensor([
            x.answer_type for x in batch
        ], dtype=torch.long),
        torch.tensor([
            x.short_answer_start for x in batch
        ], dtype=torch.long),
        torch.tensor([
            x.short_answer_end for x in batch
        ], dtype=torch.long)
    ], dim=1)
    return (
        {
            "input_ids": torch.stack([
                x.input_ids for x in batch
            ], dim=0),
            "input_mask": torch.stack([
                x.input_mask for x in batch
            ], dim=0),
            "token_type_ids": torch.stack([
                x.token_type_ids for x in batch
            ], dim=0),
            "sa_mask": torch.stack([
                x.sa_mask for x in batch
            ], dim=0)
        },
        targets
    )


def collate_example_for_inference(
    batch: Sequence[Example]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Sequence]]:
    return (
        {
            "input_ids": torch.stack([
                x.input_ids for x in batch
            ], dim=0),
            "input_mask": torch.stack([
                x.input_mask for x in batch
            ], dim=0),
            "token_type_ids": torch.stack([
                x.token_type_ids for x in batch
            ], dim=0),
            "sa_mask": torch.stack([
                x.sa_mask for x in batch
            ], dim=0)
        },
        {
            "example_id": [x.example_id for x in batch],
            "tok_to_orig": [x.tok_to_orig for x in batch],
            "starts_at": [x.starts_at for x in batch],
            "ends_at": [x.ends_at for x in batch],
            "offset": [x.offset for x in batch],
        }
    )
