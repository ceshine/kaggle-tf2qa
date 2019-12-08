import torch
import numpy as np
from tqdm import tqdm
from pytorch_helper_bot.bot import BaseBot, batch_to_device


class BasicQABot(BaseBot):
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
                preds_sa.append(output["logit_sa"].cpu())
                preds_type.append(output["logit_type"].cpu())
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
