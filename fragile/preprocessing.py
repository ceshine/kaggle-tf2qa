import json
import itertools
from pathlib import Path

import fire
import joblib
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from pandas.io.json._json import JsonReader
from tqdm import tqdm
from transformers import BertTokenizer

from .data_structures import RawCandidate, TokenizedCandidate

DEBUG = 0


class JsonChunkReader(JsonReader):
    """JsonReader provides an interface for reading in a JSON file.

    Source: https://www.kaggle.com/c/tensorflow2-question-answering/discussion/116268
    """

    @classmethod
    def chunk_reader(cls, filepath_or_buffer, chunksize):
        return cls(
            filepath_or_buffer,
            orient=None,
            typ='frame',
            dtype=None,
            convert_axes=None,
            convert_dates=True,
            keep_default_dates=True,
            numpy=False,
            precise_float=False,
            date_unit=None,
            encoding=None,
            lines=True,
            chunksize=chunksize,
            compression=None,
        )

    def __next__(self):
        lines = list(itertools.islice(self.data, self.chunksize))
        if lines:
            obj = pd.DataFrame([
                json.loads(line)
                for line in lines
            ])
            return obj

        self.close()
        raise StopIteration


def parse_annotations(row, candidates):
    success = False
    for entry in row["annotations"]:
        long_answer = entry['long_answer']
        if long_answer['candidate_index'] == -1:
            success = True
            continue
        if long_answer['candidate_index'] not in candidates:
            # print(candidates.keys(), long_answer['candidate_index'])
            # Bad entry
            continue
        reference = candidates[long_answer['candidate_index']]
        assert reference.start_token == long_answer['start_token']
        short_answers = []
        for sans in entry["short_answers"]:
            start = sans['start_token'] - long_answer['start_token']
            end = sans['end_token'] - long_answer['start_token']
            short_answers.append((start, end))
            if DEBUG:
                print(
                    " ".join(reference.tokens[start:end]) + f" {start} {end}")
        reference.answer_type = ANSWER_TYPES[entry["yes_no_answer"]]
        reference.short_answers = short_answers
        success = True
    return success


def tokenize_candidate(candidate, tokenizer):
    orig_to_tok = []
    tok_to_orig = []
    token_ids = []
    previous_spaces = 0
    for token in candidate.tokens:
        sub_tokens = tokenizer.encode(token, add_special_tokens=False)
        orig_to_tok.append(len(token_ids))
        if not sub_tokens:
            previous_spaces += 1
            continue
        # assert sub_tokens, f"{ord(token[0])}, {len(token)}, {token}"
        tok_to_orig.extend(
            [len(orig_to_tok) - previous_spaces - 1] * len(sub_tokens))
        token_ids.extend(sub_tokens)
        previous_spaces = 0
    short_answers = []
    for start, end in candidate.short_answers:
        # point to the first non-white token
        while start > 1 and orig_to_tok[start] == orig_to_tok[start-1]:
            start = start - 1
        while end > 1 and orig_to_tok[end] == orig_to_tok[end-1]:
            end = end - 1
        short_answers.append((orig_to_tok[start], orig_to_tok[end]))
        if DEBUG:
            print(
                start, end, orig_to_tok[start], orig_to_tok[end],
                tok_to_orig[orig_to_tok[start]], tok_to_orig[orig_to_tok[end]])
        assert tok_to_orig[orig_to_tok[start]] == start
        assert tok_to_orig[orig_to_tok[end]
                           ] == end, f"{start}, {end}, {tok_to_orig[orig_to_tok[start]]}, {tok_to_orig[orig_to_tok[end]]}"
    return TokenizedCandidate(
        original_index=candidate.original_index,
        filtered_index=candidate.filtered_index,
        token_ids=token_ids,
        start_token=candidate.start_token,
        end_token=candidate.end_token,
        tok_to_orig=tok_to_orig,
        answer_type=candidate.answer_type,
        short_answers=short_answers
    )


def parse_long_candidates(row, has_annotations, tokenizer):
    text = row["document_text"]
    tokens = text.split(" ")
    candidates = {}
    for i, entry in enumerate(row["long_answer_candidates"]):
        if not entry["top_level"]:
            continue
        candidates[i] = RawCandidate(
            i,
            len(candidates),
            tokens[entry["start_token"]:entry["end_token"]],
            entry["start_token"],
            entry["end_token"]
        )
        assert ">" == candidates[i].tokens[-1][-1]
    if has_annotations:
        success = parse_annotations(row, candidates)
        if not success:
            return None
    return [tokenize_candidate(x, tokenizer) for x in candidates.values()]


ANSWER_TYPES = {
    "NONE": 1,
    "NO": 2,
    "YES": 3
}


def preprocess(
    filepath: str = 'data/simplified-nq-train.jsonl',
    output_pattern: str = 'cache/train/train_%d.jl',
    has_annotations: bool = False,
    chunk_size=500,
    write_per_chunk=2,
    skip_writes=0,
    stop_at=-1,
    tokenizer_model: str = "bert-base-cased"
):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
    Path(output_pattern).parent.mkdir(exist_ok=True, parents=True)
    reader = JsonChunkReader.chunk_reader(
        filepath, chunksize=chunk_size,
    )
    buffer, n_written = [], skip_writes
    na_count = 1
    with Parallel(n_jobs=2, prefer="threads") as parallel:
        for i, df in tqdm(enumerate(reader)):
            if i < skip_writes * write_per_chunk:
                continue
            if has_annotations:
                df.drop(["document_url"], axis=1, inplace=True)
            # candidates = np.asarray(parallel(
            #     delayed(parse_long_candidates)(
            #         row, has_annotations=has_annotations
            #     )
            #     for _, row in df.iterrows()
            # ))
            candidates = [
                parse_long_candidates(
                    row, has_annotations=has_annotations, tokenizer=tokenizer)
                for _, row in df.iterrows()
            ]
            question_tokens = parallel(
                delayed(tokenizer.encode)(
                    row["question_text"], add_special_tokens=False
                )
                for _, row in df.iterrows()
            )
            # question_tokens = [
            #     TOKENIZER.encode(row["question_text"], add_special_tokens=False)
            #     for _, row in df.iterrows()]
            ids = df.example_id.values
            # print(ids.dtype)
            # df.drop(
            #     ["document_text", "long_answer_candidates", "question_text"],
            #     axis=1, inplace=True)
            # if has_annotations:
            #     df.drop(["annotations"], axis=1, inplace=True)
            na_count += np.sum([x is None for x in candidates])
            # not_none = ~np.equal(candidates, None)
            # buffer.append(np.stack([
            #     ids[not_none], question_tokens[not_none], candidates[not_none]
            # ], axis=1))
            for eid, qtokens, cands in zip(ids, question_tokens, candidates):
                if cands is None:
                    continue
                buffer.append((eid, qtokens, cands))
            # print(len(buffer))
            # print(df.columns)
            # raise ValueError()
            # for i, row in df.iterrows():
            #     print(i, [
            #         (x.short_answers, x.answer_type)
            #         for x in row["candidates"].values() if x.answer_type > 0])
            if (i + 1) % write_per_chunk == 0:
                print(
                    f"Writing... NA ratio: {na_count/chunk_size/write_per_chunk}")
                joblib.dump(buffer, output_pattern % n_written)
                n_written += 1
                buffer = []
                na_count = 0
            if stop_at > 0 and i + 1 == stop_at * write_per_chunk:
                break
    if buffer:
        joblib.dump(buffer, output_pattern % n_written)


if __name__ == '__main__':
    fire.Fire(preprocess)
