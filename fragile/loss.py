import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicQALoss(nn.Module):
    def __init__(self, sa_weight: float, alpha: float = 0.02):
        super().__init__()
        self.sa_weight = sa_weight
        self.alpha = alpha
        self.type_loss = 1.
        self.sa_loss = 1.
        self.steps = 0

    def forward(self, preds, targets):
        logit_sa = preds["logit_sa"]
        logit_type = preds["logit_type"]
        n_sa = (targets[:, 0] == 1).sum()
        type_loss = F.cross_entropy(
            logit_type, targets[:, 0],
            weight=torch.tensor([.5, 1., 2., 2.]).cuda()
        )
        if logit_sa.requires_grad:
            self.type_loss = (
                (1 - self.alpha) * self.type_loss +
                self.alpha * type_loss.item()
            )
        result = type_loss
        if n_sa > 0:
            sa_start_loss = (F.cross_entropy(
                logit_sa[:, :, 0], targets[:, 1],
                reduction="none"
            ) * (targets[:, 0] == 1)).sum() / n_sa
            sa_end_loss = (F.cross_entropy(
                logit_sa[:, :, 1], targets[:, 2],
                reduction="none"
            ) * (targets[:, 0] == 1)).sum() / n_sa
            if logit_sa.requires_grad:
                self.sa_loss = (
                    (1 - self.alpha) * self.sa_loss +
                    self.alpha * self.sa_weight *
                    (sa_start_loss + sa_end_loss).item()
                )
            result = (
                type_loss + self.sa_weight * (sa_start_loss + sa_end_loss)
            ) / (1 + self.sa_weight)
        if logit_sa.requires_grad:
            self.steps += 1
            if self.steps % 500 == 0:
                print(f"{self.type_loss:.4f} {self.sa_loss:.4f}")
        return result
