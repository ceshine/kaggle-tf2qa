import json
import itertools

import fire
import joblib
from joblib import Parallel, delayed
import pandas as pd
from pandas.io.json._json import JsonReader
from tqdm import tqdm

from.data_structures import Candidate


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
            short_answers.append(
                (start, end, " ".join(reference.tokens[start:end])))
        reference.answer_type = ANSWER_TYPES[entry["yes_no_answer"]]
        reference.short_answers = short_answers
        success = True
    return success


def parse_long_candidates(row, has_annotations=False):
    text = row["document_text"]
    tokens = text.split(" ")
    results = {}
    for i, entry in enumerate(row["long_answer_candidates"]):
        if not entry["top_level"]:
            continue
        results[i] = Candidate(
            i,
            len(results),
            tokens[entry["start_token"]:entry["end_token"]],
            entry["start_token"]
        )
    if has_annotations:
        success = parse_annotations(row, results)
        if not success:
            return None
    return results


ANSWER_TYPES = {
    "NONE": 1,
    "NO": 2,
    "YES": 3
}


def preprocess(
    filepath: str = 'data/simplified-nq-train.jsonl',
    output_pattern: str = 'cache/train_%d.jl',
    has_annotations: bool = False,
    read_chunk_size=2000,
    write_per_chunk=5
):
    reader = JsonChunkReader.chunk_reader(
        filepath, chunksize=read_chunk_size
    )
    buffer, n_written = [], 0
    na_count = 0
    with Parallel(n_jobs=4) as parallel:
        for df in tqdm(reader):
            df.drop(["document_url"], axis=1, inplace=True)
            df["candidates"] = parallel(
                delayed(parse_long_candidates)(
                    row, has_annotations=has_annotations)
                for _, row in df.iterrows()
            )
            df.drop(
                ["document_text", "long_answer_candidates"],
                axis=1, inplace=True)
            if has_annotations:
                df.drop(["annotations"], axis=1, inplace=True)
            na_count += df.candidates.isnull().sum()
            df = df[~df.candidates.isnull()].copy()
            buffer.append(df)
            # for i, row in df.iterrows():
            #     print(i, [
            #         (x.short_answers, x.answer_type)
            #         for x in row["candidates"].values() if x.answer_type > 0])
            if len(buffer) == write_per_chunk:
                print(
                    f"Writing... NA ratio: {na_count/read_chunk_size/write_per_chunk}")
                df_final = pd.concat(buffer)
                joblib.dump(df_final, output_pattern % n_written)
                n_written += 1
                buffer = []
                na_count = 0
                del df_final
    if buffer:
        df_final = pd.concat(buffer)
        joblib.dump(df_final, output_pattern % n_written)


if __name__ == '__main__':
    fire.Fire(preprocess)
