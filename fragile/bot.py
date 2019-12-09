from typing import Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from pytorch_helper_bot.bot import BaseBot, batch_to_device


class ShortAnswerAccuracy:
    name = "sa_accuracy"

    def get_stats(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]):
        eligible = (truth[:, 0] == 1)
        correct_start = torch.sum(
            (truth[:, 1] == pred["sa"][:, 0]) * eligible).item()
        correct_end = torch.sum(
            (truth[:, 2] == pred["sa"][:, 1]) * eligible).item()
        n_eligible = eligible.sum()
        return correct_start, correct_end, n_eligible

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        correct_start, correct_end, n_eligible = self.get_stats(truth, pred)
        if n_eligible == 0:
            return 0, f"0%"
        accuracy = float(correct_start + correct_end) / n_eligible / 2
        return accuracy * -1, f"{accuracy * 100:.2f}%"


class ShortAnswerStrictAccuracy:
    name = "sa_strict_accuracy"

    def get_stats(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]):
        eligible = (truth[:, 0] == 1)
        correct = torch.sum(
            (truth[:, 2] == pred["sa"][:, 1]) *
            (truth[:, 1] == pred["sa"][:, 0]) *
            eligible
        ).item()
        n_eligible = eligible.sum()
        return correct, n_eligible

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        correct, n_eligible = self.get_stats(truth, pred)
        if n_eligible == 0:
            return 0, f"0%"
        accuracy = float(correct) / n_eligible
        return accuracy * -1, f"{accuracy * 100:.2f}%"


class AnswerTypeOneAccuracy:
    name = "type_1_accuracy"

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        correct = torch.sum(
            (pred["type"] == truth[:, 0]) * (truth[:, 0] == 1)
        ).item()
        total = (truth[:, 0] == 1).sum().item()
        accuracy = float(correct) / total
        return accuracy * -1, f"{accuracy * 100:.2f}%"


class AnswerTypeAccuracy:
    name = "type_accuracy"

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        correct = torch.sum(
            (pred["type"] == truth[:, 0])
        ).item()
        total = truth.size(0)
        accuracy = float(correct) / total
        print(confusion_matrix(truth[:, 0].numpy(), pred["type"].numpy()))
        return accuracy * -1, f"{accuracy * 100:.2f}%"


class BasicQABot(BaseBot):
    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"
        self.metrics = (
            AnswerTypeAccuracy(), ShortAnswerAccuracy(),
            ShortAnswerStrictAccuracy(), AnswerTypeOneAccuracy()
        )

    def extract_prediction(self, x):
        return x

    def eval(self, loader):
        """Warning: Only support datasets whose predictions and labels together fit in memory."""
        self.model.eval()
        preds_sa, preds_type, ys = [], [], []
        losses, weights = [], []
        self.logger.debug("Evaluating...")
        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader, disable=not self.pbar):
                input_tensors = batch_to_device(input_tensors, self.device)
                output = self.extract_prediction(self.model(*input_tensors))
                batch_loss = self.criterion(
                    output, y_local.to(self.device))
                losses.append(batch_loss.data.cpu().item())
                weights.append(y_local.size(self.batch_dim))
                # Save batch labels and predictions
                # shape (batch, 2)
                preds_sa.append(
                    torch.argmax(
                        output["logit_sa"], dim=1
                    ).cpu())
                # shape (batch,)
                preds_type.append(
                    torch.argmax(
                        output["logit_type"], dim=1
                    ).cpu())
                ys.append(y_local.cpu())
        loss = np.average(losses, weights=weights)
        metrics = {"loss": (loss, self.loss_format % loss)}
        global_ys, global_preds_sa, global_preds_type = (
            torch.cat(ys), torch.cat(preds_sa), torch.cat(preds_type)
        )
        for metric in self.metrics:
            metric_loss, metric_string = metric(
                global_ys, {"sa": global_preds_sa, "type": global_preds_type})
            metrics[metric.name] = (metric_loss, metric_string)
        return metrics
