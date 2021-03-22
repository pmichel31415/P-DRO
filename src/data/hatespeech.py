#!/usr/bin/env python3
"""Adapted from
https://github.com/huggingface/pytorch-pretrained-BERT

Classes to load Hatespeech datasets
"""
import os

from transformers import DataProcessor

from .text_dataset import ClassificationExample


class HatespeechExample(ClassificationExample):
    def __init__(self, guid, text_a, label, dialect):
        super(HatespeechExample, self).__init__(guid, text_a, None, label)
        self.dialect = dialect

    @property
    def attributes(self):
        return {"dialect": self.dialect}


class HateSpeechProcessor(DataProcessor):
    """Processor for hate speech classification"""

    def get_train_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        return self._create_examples(tsv[1:], "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "valid.tsv"))
        return self._create_examples(tsv[1:], "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        return self._create_examples(tsv[1:], "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (idx, line) in enumerate(lines):
            examples.append(
                HatespeechExample(
                    guid=f"{set_type}-{idx}",
                    text_a=line[2],
                    label=line[0],
                    dialect=line[1],
                )
            )
        return examples


class FountaProcessor(HateSpeechProcessor):
    """A processor for the hate speech detection dataset from Founta et al."""

    def get_labels(self):
        """See base class."""
        return ["normal", "abusive", "hateful", "spam"]


class DavidsonProcessor(HateSpeechProcessor):
    """A procssor for the hate speech detection dataset from Davidson et al."""

    def get_labels(self):
        """See base class."""
        return ["neither", "hate", "offensive"]


hatespeech_processors = {
    "founta": FountaProcessor,
    "davidson": DavidsonProcessor,
}
