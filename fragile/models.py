from pathlib import Path
from typing import Dict, Union

import torch
from torch import nn
from transformers import BertModel, BertConfig


class BasicBert(nn.Module):
    """BERT model to generate token embeddings.

    Each token is mapped to an output vector from BERT.
    """

    def __init__(self, model_name_or_path: str):
        super().__init__()
        config = BertConfig.from_pretrained(model_name_or_path)
        config.output_hidden_states = False
        self.bert = BertModel.from_pretrained(
            model_name_or_path, config=config)
        self.answer_type_classifier = nn.Linear(
            self.get_hidden_dimension(), 4)
        self.short_answer_classifier = nn.Linear(
            self.get_hidden_dimension(), 2)

    def forward(self, features) -> Dict:
        """Returns token_embeddings, cls_token"""
        last_hidden_states, pooler_output = self.bert(
            input_ids=features['input_ids'],
            token_type_ids=features.get('token_type_ids', None),
            attention_mask=features['input_mask']
        )
        # shape: (batch, 4)
        logit_type = self.answer_type_classifier(pooler_output)
        # shape: (batch, seq_len, 2)
        logit_sa = self.short_answer_classifier(last_hidden_states)
        # make irrelevant positions nearly impossible
        logit_sa[features['sa_mask'][:, :, None].expand(-1, -1, 2)] = -1e20
        return {
            "logit_sa": logit_sa,
            "logit_type": logit_type
        }

    def get_hidden_dimension(self) -> int:
        return self.bert.config.hidden_size

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.bert.save_pretrained(str(output_path))
        torch.save({
            "type_cls": self.answer_type_classifier.state_dict(),
            "sa_cls": self.short_answer_classifier.state_dict()
        }, output_path / "classifiers.pth")

    @staticmethod
    def load(input_path: Union[str, Path]):
        input_path = Path(input_path)
        model = BasicBert(model_name_or_path=str(input_path))
        states = torch.load(input_path / "classifiers.pth")
        model.answer_type_classifier.load_state_dict(states["type_cls"])
        model.short_answer_classifier.load_state_dict(states["sa_cls"])
        return model
