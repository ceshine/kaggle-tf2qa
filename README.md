# PyTorch baseline to TensorFlow 2.0 Question Answering competition on Kaggle

This model gives roughly same performance as the [Tensorflow baseline](https://github.com/tensorflow/models/tree/master/official/nlp) (0.57~0.58 public score).

This is intended to serve as a reference implementation for more complicated QA tasks.

[Example inference notebook on Kaggle](https://www.kaggle.com/ceshine/tfqa-inference/data).

The GCP pre-emptible GPUs became too hard to use (i.e., got pre-empted too often) in mid-December 2019, so I dropped this PyTorch version and started to work on the Tensorflow version using TPU.

## Requirements

1. [pytorch_helper_bot==0.4.0](https://github.com/ceshine/pytorch-helper-bot/tree/0.4.0)
2. transformer==2.2.1
3. [fire](https://github.com/google/python-fire)

## Notes

- The `master` branch uses BERT pretrained models. The `albert` branch uses ALBERT pretrained models.
- don't worry about the Telegram bot token in train.py. I've already expired the token.)
