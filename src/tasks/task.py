#!/usr/bin/env python3
from copy import deepcopy
import torch as th
from torch import nn
import torch.nn.functional as F
from typing import Union, Callable, Optional
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from ..data.minibatch import TupleMiniBatch
from ..utils import xavier_initialize


class Identity(nn.Module):
    """Identity function"""

    def forward(self, x):
        return x


class Task(nn.Module):
    """A Task encapuslates a variety of loss functions and datasets, as well as
    a task `head` optionally (for multitask learning.)"""

    def __init__(self):
        super(Task, self).__init__()
        self._name = "task"
        self._train_data = None
        self._valid_data = None
        self._test_data = None
        self.head = Identity()

    def nll(
        self,
        model: nn.Module,
        batch: TupleMiniBatch,
        reduction: str = "mean",
        predict: bool = False,
    ):
        """Compute the NLL loss for this task

        Args:
            model: Model
            batch: Batch of data
            reduction: Either sum, mean or none. Defaults to "mean".
            predict: Return predictions as well. Defaults to False.

        Returns:
            NLL and optionlly model predictions too
        """
        device = list(model.parameters())[0].device
        batch = batch.to(device)
        inputs = batch.inputs
        # Extract features with the model
        features = model(*inputs)
        nlls = self.nll_on_features(features, batch, reduction)
        if predict:
            predictions = self.predict_on_features(features)
            return (nlls,) + predictions
        else:
            return nlls

    def logits(self, model, batch):
        """Compute the logits for this task"""
        device = list(model.parameters())[0].device
        batch = batch.to(device)
        inputs = batch.inputs
        # Extract features with the model
        features = model(*inputs)
        logits = self.logits_on_features(features, batch)
        return logits

    def logits_on_features(self, h, batch):
        """Computes logits based on features from the model"""
        batch = batch.to(h.device)
        # Extract features with the model
        features = h.view(batch.size, -1)
        # Log loss
        logits = self.head(features)
        return logits

    def nll_on_features(self, h, batch, reduction="mean"):
        """Compute the NLL loss given features h and targets y

        This assumes that the features have already be computed with the model
        """
        batch = batch.to(h.device)
        y = batch.outputs
        # Extract features with the model
        features = h.view(batch.size, -1)
        # Log loss
        logits = self.head(features)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, y, reduction=reduction)
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
        """Predict label given features from the model"""
        logits = self.head(h.view(h.size(0), -1))
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits.argmax(dim=-1)

    def score(self, y_hat, y):
        """Score a collection of labels and predictions.

        Usually this is accuracy, but it depends on the task.
        """
        return (y_hat == y.to(y_hat.device)).float().mean().item()

    def create_compatible_head(
        self,
        n_features: int,
        device: Optional[str] = None,
    ):
        """Return a head compatible with this task"""
        head = nn.Linear(n_features, self.n_classes)
        xavier_initialize(head)
        if device is not None:
            head = head.to(device)
        return head

    def build_head(self, n_features, device=None):
        """Build this task's classification head."""
        # By default this is a linear layer
        self.head = self.create_compatible_head(n_features, device)

    def eval_model(
        self,
        model: nn.Module,
        batch_size: int = 32,
        data: Union[str, th.utils.data.Dataset] = "test",
        collate_fn: Optional[Callable] = None,
        by_example: bool = False,
        label_map: Optional[Callable] = None,
        nll: bool = False,
    ):
        """Evaluate a model on a dataset

        Args:
            model: Model to evaluate
            batch_size: Batch size. Defaults to 32.
            data: Dataset to evaluate on. Can be "train", "valid" or "test" or
                a custom torch dataset. Defaults to "test".
            collate_fn: Function to use to create batches. Defaults to None.
            by_example: Return loss/score for every example.
            label_map: Function to transform the output label. This can be
                useful as a hack for single-head multitask learning. Defaults
                to None.
            nll: Return NLL as well. Defaults to False.


        Returns:
            Score on dataset (and optionally the NLL as well)
        """
        # Set model to test mode
        mode = model.training
        model.train(mode=False)
        # Select dataset for evaluation
        dataset = data
        if isinstance(data, str):
            dataset = self.get_split(data)
        elif not isinstance(dataset, th.utils.data.Dataset):
            raise ValueError(
                "`data` must be a pytorch dataset or one of 'dev'/'valid'"
                f"/'test/'train', got {dataset.__class__.__name__} instead"
            )
        # Dataloader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn if collate_fn is None else collate_fn,
        )
        y, y_hat, all_nlls = [], [], []
        for batch in data_loader:
            # Get model predictions
            with th.no_grad():
                nlls, _, predicted = self.nll(
                    model,
                    batch,
                    reduction="none",
                    predict=True,
                )
            # Track predictions and reference
            y.append(batch[-1])
            y_hat.append(predicted)
            all_nlls.append(nlls)
        # Concatenate
        y = th.cat(y, dim=0).cpu()
        y_hat = th.cat(y_hat, dim=0).cpu()
        all_nlls = th.cat(all_nlls, dim=0).cpu()
        # Map predictions to labels (this is useful for single
        # head model evaluated on multiple tasks)
        if label_map:
            y_hat = th.tensor([label_map(y_hat_i.item()) for y_hat_i in y_hat])
        # Task specific score
        if by_example:
            score = (y == y_hat).float()
        else:
            score = self.score(y_hat, y)
            nlls = nlls.mean()
        # Reset model to the original mode
        model.train(mode=mode)

        result = score
        if nll:
            result = (score, all_nlls)
        return result

    def predict_dataset(
        self,
        model: nn.Module,
        batch_size: int = 32,
        data: Union[str, th.utils.data.Dataset] = "test",
        collate_fn: Optional[Callable] = None,
    ):
        """Make predictions on a dataset"""
        # Set model to test mode
        mode = model.training
        model.train(mode=False)
        # Select dataset for evaluation
        dataset = data
        if isinstance(data, str):
            dataset = self.get_split(data)
        elif not isinstance(dataset, th.utils.data.Dataset):
            raise ValueError(
                "`data` must be a pytorch dataset or one of 'dev'/'valid'"
                f"/'test/'train', got {dataset.__class__.__name__} instead"
            )
        # Dataloader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn if collate_fn is None else collate_fn,
        )
        log_ps, y_hats = [], []
        for batch in data_loader:
            # Get model predictions
            with th.no_grad():
                log_p, y_hat = self.predict(model, batch)
            # Track predictions and log probabilities
            log_ps.append(log_p)
            y_hats.append(y_hat)
        # Concatenate
        log_ps = th.cat(log_ps, dim=0).cpu()
        y_hats = th.cat(y_hats, dim=0).cpu()
        # Reset model to the original mode
        model.train(mode=mode)
        return log_ps, y_hats

    @property
    def n_classes(self):
        """Number of classes for this task"""
        raise NotImplementedError()

    @property
    def input_size(self):
        """Shape of the input for this task"""
        raise NotImplementedError()

    @property
    def train_data(self):
        """Training data for this task"""
        return self._train_data

    @property
    def valid_data(self):
        """Validation data for this task"""
        return self._valid_data

    @property
    def test_data(self):
        """Test data for this task"""
        return self._test_data

    def collate_fn(self, *args):
        """Collater to make batches"""
        return TupleMiniBatch(default_collate(*args))

    def shatter_batch(self, batch):
        """This is the reverse of `self.collate_fn`"""
        return [tuple([elem[i] for elem in batch])
                for i in range(batch.size)]

    def _subsample_training_set(self, k):
        if not hasattr(self, "_full_train_data"):
            self._full_train_data = self._train_data
        N = len(self._full_train_data)
        self._train_data, _ = random_split(self._full_train_data, [k, N-k])

    def subsample_training_set(self, k, seed=None):
        """Subsample the training data (for low resource experiments)"""
        if seed is not None:
            rng_state = th.random.get_rng_state()
            with th.random.fork_rng():
                th.manual_seed(seed)
                self._subsample_training_set(k)
            if any(th.random.get_rng_state() != rng_state):
                raise ValueError("Bad RNG state")
        else:
            self._subsample_training_set(k)

    def dataloader(self):
        """Dataloader type for this task"""
        return DataLoader

    @property
    def name(self):
        """Name of this task"""
        return self._name

    def get_split(self, split):
        if split == "test":
            return self.test_data
        elif split == "valid" or split == "dev":
            return self.valid_data
        elif split == "train":
            return self.train_data
        else:
            raise ValueError(
                "`split` must be  one of 'dev'/'valid'"
                f"/'test/'train', got {split} instead"
            )


def concatenate_tasks(
    tasks,
    concat_train=True,
    concat_valid=True,
    concat_test=True,
):
    """Concatenate two task's datasets"""
    new_task = deepcopy(tasks[0])
    new_task._name = "+".join(task.name for task in tasks)
    if concat_train:
        new_task._train_data = ConcatDataset(
            [task.train_data for task in tasks])
    if concat_valid:
        new_task._valid_data = ConcatDataset(
            [task.valid_data for task in tasks])
    if concat_test:
        new_task._test_data = ConcatDataset([task.test_data for task in tasks])
