# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import os
import json
import sys
from pathlib import Path
from functools import partial

import tqdm
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

ON_KAGGLE = Path("/kaggle").exists()


if ON_KAGGLE:
    # Import stuffs from dataset
    DATASET_PATH = Path('../input/bert-joint-baseline/')
    TEST_FILE = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl'
else:
    DATASET_PATH = Path('cache/bert-joint-baseline/')
    TEST_FILE = 'data/simplified-nq-test.jsonl'
PUBLIC_DATASET = os.path.getsize(TEST_FILE) < 20_000_000

if True:
    sys.path.extend([str(DATASET_PATH)])
    import tokenization
    import bert_utils
    import modeling


class TDense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                 **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `TDense` layer with "
                            "non-floating point (and non-complex) "
                            "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to "
                             "`TDense` should be defined. "
                             "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=3, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.output_size, last_dim],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        super(TDense, self).build(input_shape)

    def call(self, x):
        return tf.matmul(x, self.kernel, transpose_b=True)+self.bias


def mk_model(config):
    seq_len = config['max_position_embeddings']
    unique_id = tf.keras.Input(shape=(1,), dtype=tf.int64, name='unique_id')
    input_ids = tf.keras.Input(
        shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.Input(
        shape=(seq_len,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.Input(
        shape=(seq_len,), dtype=tf.int32, name='segment_ids')
    BERT = modeling.BertModel(config=config, name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)

    logits = TDense(2, name='logits')(sequence_output)
    start_logits, end_logits = tf.split(
        logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = tf.squeeze(start_logits, axis=-1, name='start_squeeze')
    end_logits = tf.squeeze(end_logits,  axis=-1, name='end_squeeze')

    ans_type = TDense(5, name='ans_type')(pooled_output)
    return tf.keras.Model([input_ for input_ in [unique_id, input_ids, input_mask, segment_ids]
                           if input_ is not None],
                          [unique_id, start_logits, end_logits, ans_type],
                          name='bert-baseline')


def load_model():
    with open(DATASET_PATH / 'bert_config.json', 'r') as f:
        config = json.load(f)
    print(json.dumps(config, indent=4))
    model = mk_model(config)
    cpkt = tf.train.Checkpoint(model=model)
    cpkt.restore(str(DATASET_PATH / 'model_cpkt-1')).assert_consumed()
    return model


def write_eval_records(filepath: Path):
    eval_writer = bert_utils.FeatureWriter(
        filename=str(filepath),
        is_training=False)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=str(DATASET_PATH / 'vocab-nq.txt'),
        do_lower_case=True)
    features = []
    convert = bert_utils.ConvertExamples2Features(
        tokenizer=tokenizer,
        is_training=False,
        output_fn=eval_writer.process_feature,
        collect_stat=False)
    n_examples = 0
    for examples in bert_utils.nq_examples_iter(
            input_file=TEST_FILE,
            is_training=False,
            tqdm=tqdm.tqdm):
        for example in examples:
            n_examples += convert(example)
    eval_writer.close()
    print('number of test examples: %d, written to file: %d' %
          (n_examples, eval_writer.num_features))


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(
        serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if name != 'unique_id':  # t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = t

    return example


def _decode_tokens(record, seq_length):
    return tf.io.parse_single_example(
        serialized=record,
        features={
            "unique_id": tf.io.FixedLenFeature([], tf.int64),
            "token_map":  tf.io.FixedLenFeature([seq_length], tf.int64)
        })


def main():
    model = load_model()
    eval_records = DATASET_PATH / "nq-test.tfrecords"
    if ON_KAGGLE and not PUBLIC_DATASET:
        eval_records = Path('nq-test.tfrecords')
    if not eval_records.exists():
        write_eval_records(eval_records)

    seq_length = bert_utils.FLAGS.max_seq_length
    name_to_features = {
        "unique_id": tf.io.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
    }

    raw_ds = tf.data.TFRecordDataset(str(eval_records))
    token_map_ds = raw_ds.map(partial(_decode_tokens, seq_length=seq_length))
    decoded_ds = raw_ds.map(
        partial(_decode_record, name_to_features=name_to_features))
    ds = decoded_ds.batch(batch_size=16, drop_remainder=False)

    result = model.predict_generator(
        ds, verbose=1 if not ON_KAGGLE else 0)

    np.savez_compressed(
        'bert-joint-baseline-output.npz',
        **dict(zip(['uniqe_id', 'start_logits', 'end_logits', 'answer_type_logits'],
                   result)))

    all_results = [bert_utils.RawResult(*x) for x in zip(*result)]

    print("Going to candidates file")

    candidates_dict = bert_utils.read_candidates(TEST_FILE)

    print("setting up eval features")

    eval_features = list(token_map_ds)

    print("compute_pred_dict")

    nq_pred_dict = bert_utils.compute_pred_dict(
        candidates_dict,
        eval_features,
        all_results,
        tqdm=tqdm.tqdm)

    predictions_json = {"predictions": list(nq_pred_dict.values())}

    print("writing json")

    with tf.io.gfile.GFile('predictions.json', "w") as f:
        json.dump(predictions_json, f, indent=4)

    test_answers_df = pd.read_json("predictions.json")
    for var_name in ['long_answer_score', 'short_answer_score', 'answer_type']:
        test_answers_df[var_name] = test_answers_df['predictions'].apply(
            lambda q: q[var_name])
    test_answers_df["long_answer"] = test_answers_df["predictions"].apply(
        create_long_answer)
    test_answers_df["short_answer"] = test_answers_df["predictions"].apply(
        create_short_answer)
    test_answers_df["example_id"] = test_answers_df["predictions"].apply(
        lambda q: str(q["example_id"]))

    long_answers = dict(
        zip(test_answers_df["example_id"], test_answers_df["long_answer"]))
    short_answers = dict(
        zip(test_answers_df["example_id"], test_answers_df["short_answer"]))

    sample_submission = pd.read_csv(
        Path(TEST_FILE).parent / 'sample_submission.csv')

    long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains(
        "_long")].apply(lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)
    short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains(
        "_short")].apply(lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)

    sample_submission.loc[sample_submission["example_id"].str.contains(
        "_long"), "PredictionString"] = long_prediction_strings
    sample_submission.loc[sample_submission["example_id"].str.contains(
        "_short"), "PredictionString"] = short_prediction_strings
    sample_submission.to_csv("submission.csv", index=False)


def create_short_answer(entry):
    # if entry["short_answer_score"] < 1.5:
    #     return ""

    answer = []
    for short_answer in entry["short_answers"]:
        if short_answer["start_token"] > -1:
            answer.append(
                str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))
    if entry["yes_no_answer"] != "NONE":
        answer.append(entry["yes_no_answer"])
    return " ".join(answer)


def create_long_answer(entry):
   # if entry["long_answer_score"] < 1.5:
   # return ""

    answer = []
    if entry["long_answer"]["start_token"] > -1:
        answer.append(str(entry["long_answer"]["start_token"]) +
                      ":" + str(entry["long_answer"]["end_token"]))
    return " ".join(answer)


if __name__ == "__main__":
    main()
