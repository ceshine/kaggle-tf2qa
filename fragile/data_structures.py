from dataclasses import dataclass
from typing import Tuple, Sequence


@dataclass
class Candidate:
    original_index: int
    filtered_index: int
    tokens: Sequence[str]
    start_token: int
    answer_type: int = -1
    short_answers: Sequence[Tuple[int, int, str]] = ()
