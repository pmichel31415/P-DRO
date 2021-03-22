#!/usr/bin/env python3
"""Adapted from
https://github.com/huggingface/pytorch-pretrained-BERT

Classes to load TextClassificationDatasets
"""
import os


from .text_dataset import InputExample

from transformers.data.processors.utils import DataProcessor


class TCDProcessor(DataProcessor):
    """Processor for the MRPC data set(GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        return self._create_examples(tsv, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "valid.tsv"))
        return self._create_examples(tsv, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        return self._create_examples(tsv, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


class AgNewsProcessor(TCDProcessor):

    def get_labels(self):
        """See base class."""
        return [f"{i+1}" for i in range(4)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = (line[1] + ". " + line[2]).rstrip()
            text_b = None
            label = line[0]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples


class AmazonFullProcessor(TCDProcessor):

    def get_labels(self):
        """See base class."""
        return [f"{i+1}" for i in range(5)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = (line[1] + ". " + line[2]).rstrip()
            text_b = None
            label = line[0]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples


class AmazonPolarityProcessor(TCDProcessor):

    def get_labels(self):
        """See base class."""
        return [f"{i+1}" for i in range(2)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = (line[1] + ". " + line[2]).rstrip()
            text_b = None
            label = line[0]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples


class DBPediaProcessor(TCDProcessor):

    def get_labels(self):
        """See base class."""
        return [f"{i+1}" for i in range(14)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = (line[1] + ". " + line[2]).rstrip()
            text_b = None
            label = line[0]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples


class YahooAnswersProcessor(TCDProcessor):

    def get_labels(self):
        """See base class."""
        return [f"{i+1}" for i in range(10)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = (line[1] + ". " + line[2]).rstrip()
            text_b = line[3]
            label = line[0]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples


class YelpFullProcessor(TCDProcessor):

    def get_labels(self):
        """See base class."""
        return [f"{i+1}" for i in range(5)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[1]
            text_b = None
            label = line[0]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples


class YelpPolarityProcessor(TCDProcessor):

    def get_labels(self):
        """See base class."""
        return [f"{i+1}" for i in range(2)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[1]
            text_b = None
            label = line[0]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples


tcd_processors = {
    "ag_news": AgNewsProcessor,
    "amazon_review_full": AmazonFullProcessor,
    "amazon_review_polarity": AmazonPolarityProcessor,
    "dbpedia": DBPediaProcessor,
    "yahoo_answers": YahooAnswersProcessor,
    "yelp_review_full": YelpFullProcessor,
    "yelp_review_polarity": YelpPolarityProcessor,
}
