import fire
from tqdm import tqdm

from fragile.preprocessing import JsonChunkReader


def preprocess(
    example_id: str,
    filepath: str = 'data/simplified-nq-train.jsonl'
):
    reader = JsonChunkReader.chunk_reader(
        filepath, chunksize=100
    )
    example_id = int(example_id)
    for _, df in tqdm(enumerate(reader)):
        for _, row in df.iterrows():
            # print(str(row['example_id']))
            if row['example_id'] == example_id:
                for annot in row["annotations"]:
                    print(annot)
                return


if __name__ == '__main__':
    fire.Fire(preprocess)
