import os.path
import torch as th
from sacremoses import MosesDetokenizer
from transformers import DataProcessor, InputFeatures, InputExample

from .text_dataset import (
    encode_sentences,
    TextDataset,
    pad_sequences,
    attention_masks,
    InputFeaturesWithAttributes,
)
from .minibatch import TupleMiniBatch
from .utils import as_tensor


class LanguageModelingExample(InputExample):
    """A single training/test example for simple sequenec classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, canonical_n_tokens=None):
        self.guid = guid
        self.text_a = text_a
        # This is to compute perplexity
        if canonical_n_tokens is None:
            self.canonical_n_tokens = len(text_a.split())
        else:
            self.canonical_n_tokens = canonical_n_tokens

    def to_features(
        self,
        tokenizer,
        max_length,
        label_map,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True
    ):
        # Get input feature
        input_ids, token_type_ids = encode_sentences(
            self.text_a,
            None,
            tokenizer,
            max_length,
            pad_token_segment_id=pad_token_segment_id,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            mask_padding_with_zero=mask_padding_with_zero,
        )

        return InputFeaturesWithAttributes(input_ids, token_type_ids=token_type_ids)

    @property
    def attributes(self):
        return {"canonical_n_tokens": self.canonical_n_tokens}


class WikiTextProcessor(DataProcessor):
    """Loads the WikiText dataset line by line"""

    def __init__(self, line_by_line=True):
        self.line_by_line = line_by_line

    def get_train_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "wiki.train.raw"))
        return self._create_examples(tsv, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "wiki.valid.raw"))
        return self._create_examples(tsv, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "wiki.test.raw"))
        return self._create_examples(tsv, "test")

    def get_labels(self):
        """See base class."""
        return None

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        detokenizer = MosesDetokenizer("en")
        for (idx, line) in enumerate(lines):
            # Skip empty lines if we are doing line by line
            if (len(line[0]) == 0 or line[0].isspace()) and self.line_by_line:
                continue
            # Detokenize
            text = detokenizer.detokenize(line[0].split())
            examples.append(
                LanguageModelingExample(
                    guid=f"{set_type}-{idx}",
                    text_a=text,
                    canonical_n_tokens=len(line[0].split()),
                )
            )
        # If we are only using one contiguous stream of text, aggregate
        # everything in one example
        if not self.line_by_line:
            full_text = "\n".join(ex.text_a for ex in examples)
            full_token_count = sum(ex.canonical_n_tokens for ex in examples)
            examples = [LanguageModelingExample(
                set_type,
                text_a=full_text,
                canonical_n_tokens=full_token_count,
            )]
        return examples


class LanguageModelingDataset(TextDataset):
    """A dataset for language modeling"""

    def __init__(
        self,
        features,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        max_length=256,
        attributes=None,
    ):
        self.pad_token_segment_id = pad_token_segment_id
        self.pad_on_left = pad_on_left
        self.mask_padding_with_zero = mask_padding_with_zero
        self.max_length = max_length
        self.pad_token = pad_token
        self.attributes = attributes
        # Features to tensor
        self.features = [
            InputFeaturesWithAttributes(
                input_ids=as_tensor(f.input_ids, th.long),
                token_type_ids=as_tensor(f.token_type_ids, th.long),
            ) for f in features
        ]
        # Number of tokens for evaluation
        self.canonical_n_tokens = sum(
            attr["canonical_n_tokens"]
            for attr in attributes
        )

    @classmethod
    def from_text_dataset(cls, dataset):
        """Convert any text classification dataset into a language modeling one
        (basically just ignore the labels)"""
        # Check type
        if not isinstance(dataset, TextDataset):
            raise ValueError(
                f"Can only create a language modeling dataset from a "
                f"TextDataset object, got {dataset.__class__.__name__} instead"
            )
        # We modify attributes to add the number of tokens
        attributes = [{"canonical_n_tokens": len(f.input_ids)}
                      for f in dataset.features]
        if dataset.attributes is not None:
            for idx in range(len(dataset)):
                attributes[idx].update(dataset.attributes[idx])
        # Return a language modeling dataset
        return cls(
            features=dataset.features,
            pad_token_segment_id=dataset.pad_token_segment_id,
            pad_on_left=dataset.pad_on_left,
            pad_token=dataset.pad_token,
            mask_padding_with_zero=dataset.mask_padding_with_zero,
            max_length=dataset.max_length,
            attributes=attributes,
        )

    def collate_fn(self, features):
        """This creates a batch"""
        return TupleMiniBatch(
            [
                pad_sequences(
                    [f.input_ids[:-1] for f in features],
                    self.pad_token,
                    self.pad_on_left,
                ),
                attention_masks(
                    [len(f.input_ids)-1 for f in features],
                    self.mask_padding_with_zero,
                    self.pad_on_left,
                ),
                pad_sequences(
                    [f.token_type_ids[:-1] for f in features],
                    self.pad_token_segment_id,
                    self.pad_on_left,
                ),
                # The target is actually the input but shifted
                pad_sequences(
                    [f.input_ids[1:] for f in features],
                    self.pad_token,
                    self.pad_on_left,
                ),
            ],
            attributes={
                k: [f.attributes[k] for f in features]
                for k in features[0].attributes
            }
        )


def to_lm_batch(batch):
    """Converts a minibatch for text classification to a minibatch for LM"""
    return TupleMiniBatch(
        (
            batch[0][:, :-1],
            batch[1][:, 1:],
            batch[2][:, 1:],
            batch[0][:, 1:],
        ),
        attributes=batch.attributes
    )
