from .task import Task
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def image_to_256(x):
    # Rescale to [0,1]
    xmin, xmax = x.min(), x.max()
    x = x-xmin
    if xmin < xmax:
        x = x/(xmax-xmin)
    # Now to 0-255
    return (x*255).long()


class ImageDensityEstimationTask(Task):
    """Handles sentence by sentence language modeling"""

    def nll(self, model, batch, reduction="mean"):
        """Compute the NLL loss for this task"""
        device = list(model.parameters())[0].device
        batch = batch.to(device)
        inputs = batch.inputs
        # Extract features with the model
        return self.nll_on_features(model(*inputs), batch, reduction)

    def nll_on_features(self, h, batch, reduction="mean"):
        """Compute the NLL loss given features h and targets y
        This assumes that the features have already be computed by the model"""
        batch = batch.to(h.device)
        # The targets are the inputs, rescaled to 256
        y = image_to_256(batch.inputs[0])
        log_probs = F.log_softmax(h, dim=-1)
        nll_loss = F.nll_loss(
            log_probs.view(y.numel(), -1).contiguous(),
            y.contiguous().view(-1),
            reduction=reduction,
        )
        if reduction == "none":
            nll_loss = nll_loss.view(y.size())
        return nll_loss

    def predict(self, model, batch):
        """Predict label on this batch"""
        device = list(model.parameters())[0].device
        batch = batch.to(device)
        inputs = batch.inputs
        # Extract features with the model
        h = model(*inputs)
        # predictions
        return self.predict_on_features(h)

    def predict_on_features(self, h):
        """Predict label on this batch"""
        logits = self.head(h.view(h.size(0), -1))
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits.argmax(dim=-1)

    @property
    def n_classes(self):
        """Number of classes for this task"""
        return 256

    @property
    def input_size(self):
        """Shape of the input for this task"""
        return None

    def create_compatible_head(self, n_features, device=None):
        """For language modeling the head is part of the model
        (because the vocabulary is a model design choice)"""
        return nn.Identity()

    def eval_model(self, model, batch_size=32, max_tokens=2000, data="test"):
        """Evaluate a model on a dataset (using perplexity)"""
        # Set model to test mode
        mode = model.training
        model.train(mode=False)
        # Select dataset for evaluation
        dataset = data
        if data == "test":
            dataset = self.test_data
        elif data == "valid" or data == "dev":
            dataset = self.valid_data
        else:
            if not isinstance(dataset, th.utils.data.Dataset):
                raise ValueError(
                    "`data` must be a pytorch dataset or one of 'dev'/'valid'"
                    f"/'test', got {dataset.__class__.__name__} instead"
                )
        # Dataloader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        total_nll = 0
        n_pixels = 0
        for batch in data_loader:
            # Get model predictions
            with th.no_grad():
                nll = self.nll(model, batch, reduction="sum")
                # Track predictions and reference
                total_nll += nll.item()
                n_pixels += batch.inputs[0].numel()
        # Normalize NLL
        bpp = total_nll/n_pixels/np.log(2)
        # Reset model to the original mode
        model.train(mode=mode)

        return bpp

    @classmethod
    def from_image_task(cls, task):
        # Create new task
        de_task = cls()
        de_task._train_data = task.train_data
        de_task._valid_data = task.valid_data
        de_task._test_data = task.test_data

        return de_task
