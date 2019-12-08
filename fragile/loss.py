import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicQALoss(nn.Module):
    def __init__(self, sa_weight: float):
        super().__init__()
        self.sa_weight = sa_weight

    def forward(self, preds, targets):
        logit_sa = preds["logit_sa"]
        logit_type = preds["logit_type"]
        n_sa = (targets[:, 0] == 1).sum()
        type_loss = F.cross_entropy(
            logit_type, targets[:, 0]
        )
        if n_sa > 0:
            sa_start_loss = (F.cross_entropy(
                logit_sa[:, :, 0], targets[:, 1],
                reduction="none"
            ) * (targets[:, 0] == 1)).sum() / n_sa
            sa_end_loss = (F.cross_entropy(
                logit_sa[:, :, 1], targets[:, 2],
                reduction="none"
            ) * (targets[:, 0] == 1)).sum() / n_sa
            return (
                type_loss + self.sa_weight * (sa_start_loss + sa_end_loss)
            ) / (1 + self.sa_weight)
        else:
            return type_loss
