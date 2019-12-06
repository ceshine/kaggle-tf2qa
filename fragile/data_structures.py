from dataclasses import dataclass
from typing import Tuple, Sequence

import torch


@dataclass
class RawCandidate:
    original_index: int
    filtered_index: int
    tokens: Sequence[str]
    start_token: int
    end_token: int
    answer_type: int = 0
    short_answers: Sequence[Tuple[int, int]] = ()


@dataclass
class TokenizedCandidate:
    original_index: int
    filtered_index: int
    token_ids: Sequence[str]
    start_token: int
    end_token: int
    tok_to_orig: Sequence[int]
    # orig_to_tok: Sequence[int]
    answer_type: int
    short_answers: Sequence[Tuple[int, int]]


@dataclass
class Example:
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    token_type_ids: torch.Tensor
    sa_mask: torch.Tensor
    starts_at: int = 0
    ends_at: int = 0
    offset: int = 0
    tok_to_orig: Sequence[int] = ()
    answer_type: int = 0
    short_answer_start: int = 0
    short_answer_end: int = 0
    example_id: int = 0
