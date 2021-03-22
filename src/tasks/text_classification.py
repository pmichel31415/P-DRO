#!/usr/bin/env python3
import os.path
from copy import deepcopy
import numpy as np
import csv
import torch.nn.functional as F
from transformers import glue_processors
from ..data import tokenizers, scoring, tcd
from ..data.glue import MultiNLIProcessor  # noqa
from ..data.cached_dataset import load_and_cache_examples
from ..data.text_dataset import TextDataset, get_test_text_dataset
from ..data.biased_text_datasets import make_biased_sst
from ..data.amazon_multi_domain import AmazonByDomainProcessor

from .task import Task

scorers = {
    "sst-2": scoring.Accuracy(),
    "mrpc": scoring.F1(),
    "cola": scoring.Matthews(),
    "mnli": scoring.Accuracy(),
    "qnli": scoring.Accuracy(),
    "rte": scoring.Accuracy(),
    "qqp": scoring.F1(),
}


class TextClassificationTask(Task):
    """Generic class for text classification datasets"""

    def __init__(
        self,
        path,
        task_name,
        processor,
        max_seq_length=128,
        model_name="bert-base-uncased",
        tokenizer=None,
    ):
        super(TextClassificationTask, self).__init__()
        self.task_name = task_name.lower()
        self._name = f"{self.task_name}"
        # Data path
        self.path = path
        self.processor = processor
        self.scorer = scoring.Accuracy()
        if self._name in scorers:
            self.scorer = scorers[self._name]
        # Sequence length
        self.max_seq_length = max_seq_length
        # Tokenizer and model
        self.model_name = model_name
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = tokenizers.PretrainedTokenizer(model_name)
        # Labels
        self.label_list = self.processor.get_labels()
        # load data
        self.load_data()

    def load_data(self):
        self._train_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="train",
            overwrite_cache=False
        )
        self._valid_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="dev",
            overwrite_cache=False
        )
        self._test_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="test",
            overwrite_cache=False
        )
        print(len(self.train_data), len(self.valid_data), len(self.test_data))

    def nll_on_features(self, h, batch, reduction="mean"):
        """Compute the NLL loss given features h and targets y
        This assumes that the features have already be computed by the model"""
        batch = batch.to(h.device)
        y = batch.outputs
        # Extract features with the model
        features = h[:, 0].view(batch.size, -1)
        # Log loss
        logits = self.head(features)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, y, reduction=reduction)
        return nll_loss

    def logits_on_features(self, h, batch):
        batch = batch.to(h.device)
        # Extract features with the model
        features = h[:, 0].view(h.size(0), -1)
        # Log loss
        logits = self.head(features)
        return logits

    def predict_on_features(self, h):
        """Predict label on this batch"""
        logits = self.head(h[:, 0].view(h.size(0), -1))
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits.argmax(dim=-1)

    def score(self, y_hat, y):
        """Score a collection of labels"""
        return self.scorer(y_hat.cpu().numpy(), y.cpu().numpy())

    @property
    def n_classes(self):
        return len(self.label_list)

    @property
    def input_size(self):
        return None

    def collate_fn(self, *args):
        """Collater to make batches"""
        return self.train_data.collate_fn(*args)

    def shatter_batch(self, batch):
        """This is the reverse of `self.collate_fn`"""
        return self.train_data.shatter_batch(batch)


class TestTextClassificationTask(TextClassificationTask):

    def __init__(
        self,
        model_name="bert-base-uncased",
        tokenizer=None,
    ):
        Task.__init__(self)
        self._name = f"test-{model_name}"
        # Data path
        self.scorer = scoring.Accuracy()
        # Sequence length
        self.max_seq_length = 50
        # Tokenizer and model
        self.model_name = model_name
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = tokenizers.PretrainedTokenizer(model_name)
        # Labels
        self.label_list = ["Yes", "No", "Maybe"]
        # load data
        self.load_data()

    def load_data(self):
        data = get_test_text_dataset(200, self.tokenizer)
        self._train_data = data.subset(range(0, 10))
        self._valid_data = data.subset(range(100, 150))
        self._test_data = data.subset(range(150, 200))
        print(len(self.train_data), len(self.valid_data), len(self.test_data))


class GlueTask(TextClassificationTask):
    """
    GLUE task adapted from Huggingface transformers

    For use with BERT and other big pretrained models
    """

    def __init__(
        self,
        path,
        task_name,
        max_seq_length=128,
        model_name="bert-base-uncased",
        tokenizer=None,
        n_valid=0,
    ):
        # Make biased sst
        self.n_valid = n_valid
        # Call constructor
        super(GlueTask, self).__init__(
            os.path.join(path, "glue_data", task_name),
            task_name,
            glue_processors[task_name.lower()](),
            max_seq_length=max_seq_length,
            model_name=model_name,
            tokenizer=tokenizer,
        )
        # GLUE name
        self._name = f"glue-{self.task_name}"

    def load_data(self):
        self._train_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="train",
            overwrite_cache=False
        )
        self._valid_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="dev",
            overwrite_cache=False
        )
        self._test_data = self._valid_data
        self._make_validation_set()
        print(len(self.train_data), len(self.valid_data), len(self.test_data))

    def _make_validation_set(self):
        # Build validation data out of the training set
        if self.n_valid > 0:
            # Fixed rng
            seed = (hash(self._name)+154) % (2**32)
            rng = np.random.RandomState(seed=seed)
            order = rng.permutation(len(self._train_data))
            valid_idxs = order[:self.n_valid]
            self._valid_data = self._train_data.subset(valid_idxs)
            train_idxs = order[self.n_valid:]
            self._train_data.inplace_subset(train_idxs)


_CACHED_MNLI_TRAIN = None
_CACHED_MNLI_DEV = None


class MultiNLI(GlueTask):
    """MultiNLItask with additional options to filter by genre
    or presence/absence of negation
    """

    _matched_genres = ["fiction", "government", "slate",
                       "telephone", "travel"]
    _mismatched_genres = ["facetoface", "letters", "nineeleven",
                          "oup", "verbatim"]

    def __init__(
        self,
        path,
        max_seq_length=128,
        genres=None,
        negations=None,
        labels=None,
        model_name="bert-base-uncased",
        tokenizer=None,
        cached_train=None,
    ):
        self.genres = genres
        self.negations = negations
        self.labels = labels
        # Custom suffix for this task
        self.suffix = ""
        if self.genres is not None:
            self.suffix += "-".join(genre for genre in self.genres)
        if self.negations is not None:
            self.suffix += "negation" if self.negations else "no_negation"
        if self.labels is not None:
            self.suffix += "-".join(str(label) for label in self.labels)
        if self.suffix == "":
            self.suffix = None
        # This is to only load the training data once
        self.cached_train = cached_train

        super(MultiNLI, self).__init__(
            path,
            "MNLI",
            max_seq_length,
            model_name,
            tokenizer,
        )
        self._name = f"mnli"
        if self.suffix is not None:
            self._name = f"{self._name}_{self.suffix}"

    def filter_example(self, features):
        """Function used to filter examples by genre/label/negation"""
        include_example = True
        genre = features.attributes["genre"]
        has_negation = features.attributes["has_negation"]
        if self.genres is not None:
            include_example &= genre in self.genres
        if self.negations is not None:
            include_example &= has_negation == self.negations
        if self.labels is not None:
            include_example &= features.label.item() in self.labels
        return include_example

    def load_data(self):
        # Training data
        global _CACHED_MNLI_TRAIN
        if self.cached_train and _CACHED_MNLI_TRAIN is not None:
            self._train_data = _CACHED_MNLI_TRAIN
        else:
            self._train_data = load_and_cache_examples(
                self.path,
                self.task_name,
                self.tokenizer._tokenizer,
                self.processor,
                self.max_seq_length,
                model_type=self.model_name,
                split="train",
                overwrite_cache=False
            )
            if self.cached_train:
                _CACHED_MNLI_TRAIN = deepcopy(self._train_data)
        if self.suffix is not None:
            self._train_data = self._train_data.filtered(self.filter_example)
        # Validation data
        self._valid_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="dev" if self.genres is None else "dev_full",
            overwrite_cache=False
        )
        if self.suffix is not None:
            self._valid_data.filter(self.filter_example)
        self._test_data = self._valid_data
        self._make_validation_set()


class BiasedSST(TextClassificationTask):
    """SST with a bias."""

    def __init__(
        self,
        path,
        max_seq_length=128,
        bias_percent=90,
        label=None,
        biased=None,
        model_name="bert-base-uncased",
        tokenizer=None,
    ):
        self.bias_percent = bias_percent
        self.orig_data_dir = os.path.join(path, "glue_data", "SST-2")
        self.label = label
        self.biased = biased
        # Call constructor
        super(BiasedSST, self).__init__(
            os.path.join(path, "glue_data",
                         f"SST-2-biased-{self.bias_percent}"),
            "SST-2-biased",
            glue_processors["sst-2"](),
            max_seq_length=max_seq_length,
            model_name=model_name,
            tokenizer=tokenizer,
        )
        if self.label is not None:
            self._name = f"{self._name}_{self.label}"
        if self.biased is not None:
            is_biased = 'biased' if self.biased else 'not_biased'
            self._name = f"{self._name}_{is_biased}"
        print(self.name)

    def filter_example(self, features):
        """Function used to filter examples by genre/label/negation"""
        include_example = True
        biased = features.attributes["biased"]
        if self.biased is not None:
            include_example &= (biased == self.biased)
        if self.label is not None:
            include_example &= int(features.label.item()) == self.label
        return include_example

    def load_data(self):
        # Create the biased SST dataset
        train, valid, test = make_biased_sst(
            self.orig_data_dir,
            percent_biased=self.bias_percent,
            save_dir=self.path,
        )
        # Number of biased samples
        print(len([x for x in test if x.biased]))

        def prepare_split(split, split_name):
            split_cached_filename = os.path.join(
                self.path,
                (f"cached_{split_name}_{self.model_name}_"
                 f"sst-2-biased_{self.bias_percent}"),
            )
            # Prepare dataset
            dataset = TextDataset.from_examples(
                split,
                self.tokenizer,
                # pad on the left for xlnet
                pad_on_left=bool(self.model_name in ["xlnet"]),
                label_list=self.processor.get_labels(),
                max_length=self.max_seq_length,
                cached_filename=split_cached_filename
            )
            # Filter by attribute
            if self.biased is not None or self.label is not None:
                dataset = dataset.filtered(self.filter_example)
            return dataset

        # Train data
        self._train_data = prepare_split(train, "train")
        # Valid data
        self._valid_data = prepare_split(valid, "valid")
        # Test data
        self._test_data = prepare_split(test, "test")
        # Print data size
        print(len(self.train_data), len(self.valid_data), len(self.test_data))


_CACHED_MNLI_SAGAWA = None


class MultiNLISagawa(MultiNLI):
    """MultiNLI with the custom train/dev split from Sagawa et al. 2020."""

    def __init__(
        self,
        path,
        max_seq_length=128,
        genres=None,
        negations=None,
        labels=None,
        model_name="bert-base-uncased",
        tokenizer=None,
        cached_train=None,
        split_type="random",
    ):
        self.split_type = split_type
        super(MultiNLISagawa, self).__init__(
            path,
            max_seq_length=max_seq_length,
            genres=genres,
            negations=negations,
            labels=labels,
            model_name=model_name,
            tokenizer=tokenizer,
            cached_train=cached_train,
        )
        self._name = f"{self._name}_sagawa"

    def load_data(self):
        # Training data
        global _CACHED_MNLI_SAGAWA
        if self.cached_train and _CACHED_MNLI_SAGAWA is not None:
            full_data = _CACHED_MNLI_SAGAWA
        else:
            # Train
            full_data = load_and_cache_examples(
                self.path,
                self.task_name,
                self.tokenizer._tokenizer,
                self.processor,
                self.max_seq_length,
                model_type=self.model_name,
                split="train",
                overwrite_cache=False
            )
            # + dev
            full_data.concatenate_(load_and_cache_examples(
                self.path,
                self.task_name,
                self.tokenizer._tokenizer,
                self.processor,
                self.max_seq_length,
                model_type=self.model_name,
                split="dev",
                overwrite_cache=False
            ))
            # + dev mismatched
            full_data.concatenate_(load_and_cache_examples(
                self.path,
                self.task_name,
                self.tokenizer._tokenizer,
                self.processor,
                self.max_seq_length,
                model_type=self.model_name,
                split="dev_mismatched",
                overwrite_cache=False
            ))

            if self.cached_train:
                _CACHED_MNLI_SAGAWA = deepcopy(full_data)
        # Load the split
        split_file = os.path.join(self.path, f"metadata_{self.split_type}.csv")
        with open(split_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            # skip header
            next(reader)
            # Read split values
            split_idxs = []
            for row in reader:
                split_idxs.append(int(row[-1]))
        # Check that we have one split for every example
        if len(split_idxs) != len(full_data):
            raise ValueError(
                f"Mismatch between number of examples ({len(full_data)})"
                f" and length of metadata ({len(split_idxs)})"
            )
        # Retrieve indices for each split
        train_idxs = [idx for idx, split_idx in enumerate(split_idxs)
                      if split_idx == 0]
        dev_idxs = [idx for idx, split_idx in enumerate(split_idxs)
                    if split_idx == 1]
        test_idxs = [idx for idx, split_idx in enumerate(split_idxs)
                     if split_idx == 2]
        # Subsample
        self._train_data = full_data.subset(train_idxs)
        self._valid_data = full_data.subset(dev_idxs)
        self._test_data = full_data.subset(test_idxs)
        # Filter by genre, etc...
        if self.suffix is not None:
            self._train_data = self._train_data.filtered(self.filter_example)
            self._valid_data = self._valid_data.filtered(self.filter_example)
            self._test_data = self._test_data.filtered(self.filter_example)


class TCDTask(TextClassificationTask):
    """
    Text classification dataset task adapted from Huggingface transformers

    For use with BERT and other big pretrained models
    """

    def __init__(
        self,
        path,
        task_name,
        max_seq_length=128,
        model_name="bert-base-uncased",
        tokenizer=None,
    ):

        # Check GLUE task
        if task_name.lower() not in tcd.tcd_processors:
            raise ValueError(f"TCD task not found: {task_name.lower()}")
        # Call constructor
        super(TCDTask, self).__init__(
            os.path.join(
                path,
                "TextClassificationDatasets-120k",
                f"{task_name}_csv",
            ),
            task_name,
            tcd.tcd_processors[task_name.lower()](),
            max_seq_length=max_seq_length,
            model_name=model_name,
            tokenizer=tokenizer,
        )
        # GLUE name
        self._name = f"tcd-{self.task_name}"


_CACHED_AMAZON_TRAIN = None


class AmazonMultiDomainTask(TextClassificationTask):
    """
    Text classification dataset task adapted from Huggingface transformers

    For use with BERT and other big pretrained models
    """

    def __init__(
        self,
        path,
        max_seq_length=128,
        model_name="bert-base-uncased",
        tokenizer=None,
        dev_set="imbalanced_dev",
        test_set="test",
        cached_train=False,
        domain=None,
    ):
        self.dev_set = dev_set
        self.cached_train = cached_train
        self.domain = domain
        task_name = "amazon_multi_domain"
        # Call constructor
        super(AmazonMultiDomainTask, self).__init__(
            os.path.join(path, "amazon-multi-domain"),
            task_name,
            AmazonByDomainProcessor(),
            max_seq_length=max_seq_length,
            model_name=model_name,
            tokenizer=tokenizer,
        )
        if self.domain is not None:
            self._name = task_name + "_" + domain

    def filter_example(self, features):
        """Function used to filter examples by domain"""
        include_example = True
        domain = features.attributes["domain"]
        if self.domain is not None:
            include_example &= domain == self.domain
        return include_example

    def load_data(self):
        global _CACHED_AMAZON_TRAIN
        if self.cached_train and _CACHED_AMAZON_TRAIN is not None:
            self._train_data = _CACHED_AMAZON_TRAIN
        else:
            self._train_data = load_and_cache_examples(
                self.path,
                self.task_name,
                self.tokenizer._tokenizer,
                self.processor,
                self.max_seq_length,
                model_type=self.model_name,
                split="train",
                overwrite_cache=False
            )
            if self.cached_train:
                _CACHED_AMAZON_TRAIN = deepcopy(self._train_data)
        if self.domain is not None:
            self._train_data = self._train_data.filtered(self.filter_example)
        self._valid_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split=self.dev_set,
            overwrite_cache=False
        )
        if self.domain is not None:
            self._valid_data = self._valid_data.filtered(self.filter_example)
        self._test_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="test",
            overwrite_cache=False
        )
        if self.domain is not None:
            self._test_data = self._test_data.filtered(self.filter_example)
