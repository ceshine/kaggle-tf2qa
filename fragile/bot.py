from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from pytorch_helper_bot.bot import BaseBot, batch_to_device


class ShortAnswerAccuracy:
    name = "sa_accuracy"

    def get_stats(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]):
        eligible = (truth[:, 0] == 1)
        correct_start = torch.sum(
            (truth[:, 1] == pred["sa"][:, 0]) * eligible).item()
        correct_end = torch.sum(
            (truth[:, 2] == pred["sa"][:, 1]) * eligible).item()
        n_eligible = eligible.sum().item()
        print(
            "Average start deviance: %.2f" %
            ((torch.abs(truth[:, 1] - pred["sa"][:, 0]).float() *
              eligible * (truth[:, 1] != 0)).sum()
                / n_eligible).item())
        print(
            "Average end deviance: %.2f" %
            ((torch.abs(truth[:, 2] - pred["sa"][:, 1]).float() *
              eligible * (truth[:, 1] != 0)).sum()
                / n_eligible).item())
        return correct_start, correct_end, n_eligible

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        correct_start, correct_end, n_eligible = self.get_stats(truth, pred)
        # print(correct_start, correct_end, n_eligible)
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
        n_eligible = eligible.sum().item()
        return correct, n_eligible

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        correct, n_eligible = self.get_stats(truth, pred)
        if n_eligible == 0:
            return 0, f"0%"
        accuracy = float(correct) / n_eligible
        return accuracy * -1, f"{accuracy * 100:.2f}%"


class AnswerTypeOneRecall:
    name = "type_1_recall"

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        correct = torch.sum(
            (pred["type"] == truth[:, 0]) * (truth[:, 0] == 1)
        ).item()
        total = (truth[:, 0] == 1).sum().item()
        accuracy = float(correct) / total
        return accuracy * -1, f"{accuracy * 100:.2f}%"


class AnswerTypeOnePrecision:
    name = "type_1_precision"

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        correct = torch.sum(
            (pred["type"] == truth[:, 0]) * (pred["type"] == 1)
        ).item()
        total = (pred["type"] == 1).sum().item()
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


class AnswerTypeAUC:
    name = "type_AUC"

    def __call__(self, truth: torch.Tensor, pred: Dict[str, torch.Tensor]) -> Tuple[float, str]:
        auc = roc_auc_score(
            (truth[:, 0] == 0).numpy(), pred["type0_prob"].numpy()
        )
        return auc * -1, f"{auc * 100:.2f}%"


class BasicQABot(BaseBot):
    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"
        self.metrics = (
            AnswerTypeAccuracy(), ShortAnswerAccuracy(),
            ShortAnswerStrictAccuracy(), AnswerTypeOneRecall(),
            AnswerTypeOnePrecision(), AnswerTypeAUC()
        )

    def extract_prediction(self, x):
        return x

    def eval(self, loader):
        """Warning: Only support datasets whose predictions and labels together fit in memory."""
        self.model.eval()
        preds_sa, preds_type, preds_type0_prob, ys = [], [], [], []
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
                # print((output["logit_sa"][0, y_local[0, 1], 0]))
                # print((output["logit_sa"][0, y_local[0, 2], 1]))
                assert(output["logit_sa"][0, y_local[0, 1], 0] > -1e3)
                assert(output["logit_sa"][0, y_local[0, 2], 1] > -1e3)
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
                # shape (batch,)
                preds_type0_prob.append(
                    F.softmax(output["logit_type"], dim=1)[:, 0].cpu()
                )
                ys.append(y_local.cpu())
        loss = np.average(losses, weights=weights)
        metrics = {"loss": (loss, self.loss_format % loss)}
        global_ys, global_preds_sa, global_preds_type, global_type0_prob = (
            torch.cat(ys), torch.cat(preds_sa), torch.cat(preds_type),
            torch.cat(preds_type0_prob)
        )
        for metric in self.metrics:
            metric_loss, metric_string = metric(
                global_ys,
                {
                    "sa": global_preds_sa,
                    "type": global_preds_type,
                    "type0_prob": global_type0_prob
                }
            )
            metrics[metric.name] = (metric_loss, metric_string)
        return metrics

    def predict(self, loader):
        self.model.eval()
        ids, sas, types, lstarts, lends = [], [], [], [], []
        with torch.no_grad():
            for *input_tensors, batch_metas in tqdm(loader, disable=not self.pbar):
                input_tensors = batch_to_device(input_tensors, self.device)
                pred_dict = self.predict_batch(input_tensors)
                # shape: (batch, 2)
                pred_sa = torch.argmax(
                    pred_dict["logit_sa"].cpu(), dim=1).numpy()
                for i, sa in enumerate(pred_sa):
                    start, end = sa[0], sa[1]
                    if start > 0:
                        start = batch_metas['tok_to_orig'][i][
                            sa[0] - batch_metas["offset"][i]]
                    if end > 0:
                        end = batch_metas['tok_to_orig'][i][
                            sa[1] - batch_metas["offset"][i]]
                    sas.append((start, end))
                # shape: (batch, 4)
                types.append(
                    F.softmax(pred_dict["logit_type"], dim=1).cpu().numpy())
                ids.append(np.asarray(batch_metas["example_id"]))
                lstarts.append(np.asarray(batch_metas["starts_at"]))
                lends.append(np.asarray(batch_metas["ends_at"]))
        sas = np.asarray(sas)
        types = np.concatenate(types)
        df = pd.DataFrame({
            "example_id": np.concatenate(ids),
            "long_start": np.concatenate(lstarts),
            "long_end": np.concatenate(lends),
            "short_start_hat": sas[:, 0],
            "short_end_hat": sas[:, 1],
            "type_0_prob": types[:, 0],
            "type_1_prob": types[:, 1],
            "type_2_prob": types[:, 2],
            "type_3_prob": types[:, 3],
        })
        print(df.dtypes)
        return df
