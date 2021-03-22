#!/usr/bin/env python3
"""Adapted from
https://github.com/huggingface/pytorch-pretrained-BERT

Classes to load GLUE classes
"""
import os

from transformers import DataProcessor, glue_processors
from transformers.data.processors.glue import Sst2Processor

from .text_dataset import ClassificationExample


class NLIDiagnosticExample(ClassificationExample):

    def __init__(
        self,
        guid,
        text_a,
        text_b=None,
        label=None,
        lex_sem=None,
        pred_arg_struct=None,
        logic=None,
        knowledge=None,
        domain=None,
    ):
        super(NLIDiagnosticExample, self).__init__(guid, text_a, text_b, label)
        self.lex_sem = lex_sem
        self.pred_arg_struct = pred_arg_struct
        self.logic = logic
        self.knowledge = knowledge
        self.domain = domain


class MnliExample(ClassificationExample):
    """A MultiNLI example contains information on genre
    and the presence of negation"""

    def __init__(self, guid, text_a, text_b, label, genre, has_negation):
        super(MnliExample, self).__init__(guid, text_a, text_b, label)
        # Mnli contains sentence (pairs) from different genres
        self.genre = genre
        # Negation can be a confounding factor due to annotation
        # Annotators often craft contradictions with negations
        # Cf Gururangan et al., 2018 https://arxiv.org/abs/1803.02324
        self.has_negation = has_negation

    @property
    def attributes(self):
        return {"genre": self.genre, "has_negation": self.has_negation}


class MySst2Processor(Sst2Processor):
    """A procssor for SSt2 but with the real references for the test set"""

    def get_test_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "test_withref.tsv"))
        return self._create_examples(tsv, "test_withref")


class MultiNLIProcessor(DataProcessor):
    """Processor for MultiNLI. The difference with the
    transformers one is that we are tracking attributes of the data"""

    def get_train_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        return self._create_examples(tsv, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "dev_matched.tsv"))
        return self._create_examples(tsv, "dev")

    def get_dev_mismatched_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv"))
        return self._create_examples(tsv, "dev")

    def get_dev_full_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "dev_matched.tsv"))
        tsv2 = self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv"))
        tsv.extend(tsv2)
        return self._create_examples(tsv, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        return self._create_examples(tsv, "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (idx, line) in enumerate(lines):
            # Skip first line
            if line[-1] == "gold_label":
                continue
            # Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
            negation_words = {"nobody", "no", "never", "nothing"}
            hyp_words = set(line[5].split())
            has_negation = len(negation_words & hyp_words) > 0
            examples.append(
                MnliExample(
                    guid=f"{set_type}-{idx}",
                    text_a=line[8],
                    text_b=line[9],
                    label=line[-1],
                    genre=line[3],
                    has_negation=has_negation,
                )
            )
        return examples


# Overwrite mnli processor
glue_processors["mnli"] = MultiNLIProcessor
glue_processors["sst-2"] = MySst2Processor
