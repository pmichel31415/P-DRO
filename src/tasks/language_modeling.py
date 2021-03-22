#!/usr/bin/env python3
import os.path
from .task import Task
from .text_classification import TextClassificationTask
from ..data.language_modeling import (
    LanguageModelingDataset,
    WikiTextProcessor,
)

import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from ..data import tokenizers, ByTokensSampler


class LanguageModelingTask(Task):
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
        # The targets are
        y = batch.outputs
        # Retrieve attention mask
        loss_mask = batch.inputs[1].float().view(y.size(0), y.size(1), 1)
        log_probs = F.log_softmax(h, dim=-1) * loss_mask
        nll_loss = F.nll_loss(
            log_probs.view(y.numel(), -1).contiguous(),
            y.contiguous().view(-1),
            reduction=reduction,
        )
        if reduction == "none":
            nll_loss = nll_loss.view(y.size(0), y.size(1))
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

    def collate_fn(self, *args):
        """Collater to make batches"""
        return self.train_data.collate_fn(*args)

    @property
    def n_classes(self):
        """Number of classes for this task"""
        return None

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
        # Batch sampler
        sampler = ByTokensSampler(
            dataset,
            max_samples=batch_size,
            max_tokens=max_tokens,
            shuffle=False,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
        )
        total_nll = 0

        for batch in data_loader:
            with th.no_grad():
                nll = self.nll(model, batch, reduction="sum")
                # Track predictions and reference
                total_nll += nll.item()
        # Normalize NLL
        n_tokens = dataset.canonical_n_tokens
        ppl = np.exp(total_nll/n_tokens)
        # Reset model to the original mode
        model.train(mode=mode)

        return ppl

    @classmethod
    def from_text_task(cls, task):
        if not isinstance(task, TextClassificationTask):
            raise ValueError(
                f"Expected TextClassificationTask, "
                f"got {task.__class__.__name__}"
            )
        # Convert datasets
        train_data = LanguageModelingDataset.from_text_dataset(task.train_data)
        valid_data = LanguageModelingDataset.from_text_dataset(task.valid_data)
        test_data = LanguageModelingDataset.from_text_dataset(task.test_data)
        # Create new task
        lm_task = cls()
        lm_task._train_data = train_data
        lm_task._valid_data = valid_data
        lm_task._test_data = test_data

        return lm_task


class CCLanguageModelingTask(Task):
    """Handles sentence by sentence class conditional language modeling"""

    def __init__(self, n_classes, generative=False):
        super(CCLanguageModelingTask, self).__init__()
        self._n_classes = n_classes
        self.generative = generative

    def nll(self, model, batch, reduction="mean"):
        """Compute the NLL loss for this task"""
        device = list(model.parameters())[0].device
        batch = batch.to(device)
        inputs = batch.inputs
        labels = batch.outputs
        # Extract features with the model
        return self.nll_on_features(model(labels, *inputs), batch, reduction)

    def nll_on_features(self, h, batch, reduction="mean"):
        """Compute the NLL loss given features h and targets y
        This assumes that the features have already be computed by the model"""
        batch = batch.to(h[0].device)
        # The targets are
        outputs = batch.inputs[0][:, 1:].contiguous()
        labels = batch.outputs
        # Retrieve attention mask
        mask = batch.inputs[1]
        loss_mask = mask[:, :-1].float().unsqueeze(-1)
        # Retrieve the logits for words and labels
        word_logits = h[0]
        word_log_probs = F.log_softmax(word_logits, dim=-1) * loss_mask
        word_loss = F.nll_loss(
            word_log_probs.view(outputs.numel(), -1).contiguous(),
            outputs.contiguous().view(-1),
            reduction=reduction,
        )
        if self.generative:
            # Now we need to predict the label as well
            label_logits = h[1]
            label_log_probs = F.log_softmax(label_logits, dim=-1)
            label_loss = F.nll_loss(
                label_log_probs,
                labels,
                reduction=reduction,
            )
        # Add the term for the labels
        if reduction == "none":
            word_loss = word_loss.view(outputs.size(0), outputs.size(1))
            if self.generative:
                nll_loss = th.cat([label_loss.view(-1, 1), word_loss], dim=1)
            else:
                nll_loss = word_loss
        elif self.generative:
            nll_loss = word_loss + label_loss
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

    def collate_fn(self, *args):
        """Collater to make batches"""
        return self.train_data.collate_fn(*args)

    @property
    def n_classes(self):
        """Number of classes for this task"""
        return self._n_classes

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
        # Batch sampler
        sampler = ByTokensSampler(
            dataset,
            max_samples=batch_size,
            max_tokens=max_tokens,
            shuffle=False,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
        )
        total_nll = 0
        total_n_tokens = 0
        for batch in data_loader:
            # x = batch.inputs[0][0]
            # y = batch.outputs[0]
            # tok = self.tokenizer._tokenizer
            # print(" ".join(tok.convert_ids_to_tokens(x.cpu().numpy())))
            # print(" ".join(tok.convert_ids_to_tokens(y.cpu().numpy())))
            # Get model predictions
            with th.no_grad():
                nll = self.nll(model, batch, reduction="sum")
                # Track predictions and reference
                total_nll += nll.item()
                # Denominator
                total_n_tokens += batch.inputs[1].float().sum().item()
                # add the lables
                total_n_tokens += batch.size
        # Normalize NLL
        ppl = np.exp(total_nll/total_n_tokens)
        # Reset model to the original mode
        model.train(mode=mode)

        return ppl

    @classmethod
    def from_text_task(cls, task, generative=False):
        if not isinstance(task, TextClassificationTask):
            raise ValueError(
                f"Expected TextClassificationTask, "
                f"got {task.__class__.__name__}"
            )
        # Create new task
        lm_task = cls(n_classes=task.n_classes, generative=generative)
        lm_task._train_data = task.train_data
        lm_task._valid_data = task.valid_data
        lm_task._test_data = task.test_data
        lm_task._name = f"{task._name}_gen_LM"

        return lm_task


class WikiText(LanguageModelingTask):
    """Handles sentence by sentence language modeling"""

    def __init__(self, path, version="2", model_name="gpt2", tokenizer=None):
        self.version = version
        self._name = f"wikitext{version}"
        self.path = os.path.join(path, f"wikitext-{version}-raw")
        self.model_name = model_name
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = tokenizers.PretrainedTokenizer(model_name)
        self.pad_token = self.tokenizer.pad_token_id
        self.load_data()

    def load_data(self):
        processor = WikiTextProcessor(line_by_line=True)
        # Train data
        self._train_data = LanguageModelingDataset.from_examples(
            lambda: processor.get_train_examples(self.path),
            self.tokenizer,
            # pad on the left for xlnet
            pad_on_left=bool(self.model_name in ["xlnet"]),
            pad_token=self.pad_token,
            max_length=512,
            cached_filename=os.path.join(
                self.path,
                f"cached_train_{self.model_name}_wikitext{self.version}",
            )
        )
        # Valid data
        self._valid_data = LanguageModelingDataset.from_examples(
            lambda: processor.get_dev_examples(self.path),
            self.tokenizer,
            # pad on the left for xlnet
            pad_on_left=bool(self.model_name in ["xlnet"]),
            pad_token=self.pad_token,
            max_length=1024,
            cached_filename=os.path.join(
                self.path,
                f"cached_valid_{self.model_name}_wikitext{self.version}",
            )
        )
        # Test data
        self._test_data = LanguageModelingDataset.from_examples(
            lambda: processor.get_test_examples(self.path),
            self.tokenizer,
            # pad on the left for xlnet
            pad_on_left=bool(self.model_name in ["xlnet"]),
            pad_token=self.pad_token,
            max_length=1024,
            cached_filename=os.path.join(
                self.path,
                f"cached_test_{self.model_name}_wikitext{self.version}",
            )
        )
