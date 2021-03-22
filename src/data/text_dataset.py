#!/usr/bin/env python3
"""This code bracnhed off of the former pytorch-pretrained-bert repo
(now pytorch transformers: https://github.com/huggingface/transformers)"""
from collections import defaultdict
import torch as th
import os
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, InputExample, InputFeatures

from typing import List

from .minibatch import TupleMiniBatch
from .utils import as_tensor


class InputFeaturesWithAttributes(InputFeatures):
    """An input feature object with an additional attributes field which
    we can use to cram additional information about the sample
    """

    def add_attribute(self, name, value):
        if not hasattr(self, "attributes"):
            self.attributes = {}
        self.attributes[name] = value


def encode_sentences(
    text_a,
    text_b,
    tokenizer,
    max_length,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True
):
    """Converts a sentence or sentence pair to a list of integers

    Args:
        text_a (str): First sentence
        text_b (str): Second sentence (can be None)
        tokenizer (Tokenizer): Tokenizer implementing the `encode_plus` method
        max_length (int): Length cap
        pad_token_segment_id (int, optional): Segment id for padding tokens.
            Defaults to 0.
        pad_on_left (bool, optional): Align sentences to the right.
            Defaults to False.
        pad_token (int, optional): Padding token value. Defaults to 0.
        mask_padding_with_zero (bool, optional): Mask padding tokens with 0.
            Defaults to True.


    Returns:
        tuple: input_ids, token_type_ids
    """
    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
    )

    return inputs["input_ids"], inputs["token_type_ids"]


class ClassificationExample(InputExample):
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
            self.text_b,
            tokenizer,
            max_length,
            pad_token_segment_id=pad_token_segment_id,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            mask_padding_with_zero=mask_padding_with_zero,
        )
        # TODO: fix glue and refactor other examples
        label = label_map[self.label]

        return InputFeaturesWithAttributes(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            label=label
        )


class RegressionExample(InputExample):

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
        # Get sentences together
        input_ids, token_type_ids = encode_sentences(
            self.text_a,
            self.text_b,
            tokenizer,
            max_length,
            pad_token_segment_id=pad_token_segment_id,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            mask_padding_with_zero=mask_padding_with_zero,
        )
        label = float(self.label)

        return InputFeaturesWithAttributes(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            label=label
        )


def pad_sequences(seqs, pad_token, pad_to_left=False):
    """Pad a batch of sequences to the same length"""
    bsz = len(seqs)
    L = max(len(x) for x in seqs)
    device = seqs[0].device
    output = th.full((bsz, L), fill_value=pad_token, device=device)
    for idx, seq in enumerate(seqs):
        if pad_to_left:
            output[idx, -len(seq):] = seq
        else:
            output[idx, :len(seq)] = seq
    return output.long()


def attention_masks(lengths, mask_with_zeros, pad_to_left=False):
    """Return attention masks"""
    bsz = len(lengths)
    L = max(lengths)
    mask_val = th.full((bsz, L), 0 if mask_with_zeros else 1)
    positions = th.arange(L)
    if pad_to_left:
        positions = positions.flip((0,))
    positions = positions.view(1, -1).repeat(bsz, 1)
    lengths = th.tensor(lengths).view(-1, 1)
    return th.where(positions >= lengths, mask_val, 1-mask_val).long()


class TextDataset(Dataset):
    """A class for handling text datasets

    Each example consists of a tuple:

    (
        input_ids,      # Ids for the input tokens
        attention_mask, # Mask for attention (because of batching)
        token_type_ids, # Ids for multiple sentences
        label,          # Classification label
    )
    """

    def __init__(
        self,
        features,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        max_length=256,
        cached_filename=None,
        attributes=None,
    ):
        self.pad_token_segment_id = pad_token_segment_id
        self.pad_on_left = pad_on_left
        self.mask_padding_with_zero = mask_padding_with_zero
        self.max_length = max_length
        self.pad_token = pad_token
        # Features to tensor
        self.features = [
            InputFeaturesWithAttributes(
                input_ids=as_tensor(f.input_ids, th.long),
                token_type_ids=as_tensor(f.token_type_ids, th.long),
                label=as_tensor(
                    f.label,
                    th.float if isinstance(f.label, float) else th.long
                ),
            ) for f in features
        ]
        # Attributes
        self.attributes = attributes
        if attributes is None:
            self.attributes = [{} for f in features]
        # Add label to attributes
        for idx in range(len(self.features)):
            self.attributes[idx]["label"] = self.features[idx].label.item()

    @classmethod
    def from_examples(
        cls,
        examples_or_fn,
        tokenizer,
        pad_token=0,
        pad_token_segment_id=0,
        pad_on_left=False,
        mask_padding_with_zero=True,
        max_length=256,
        cached_filename=None,
        label_list=None,
        verbose=True,
        skip_long_examples=True,
    ):
        """Create a dataset from example and a tokenizer

        Args:
            examples_or_fn (list, function): List of InputExamples or function
                to call to load the examples
            tokenizer (Tokenizer): Must have an .encode method that takes
                in a string and returns a list of indices
            pad_token (int, optional): [description]. Defaults to 0.
            pad_token_segment_id (int, optional): Padding token segment id
                (for multiple sentences). Defaults to 0.
            pad_on_left (bool, optional): Pad to the left (why though?).
                Defaults to False.
            mask_padding_with_zero (bool, optional): Mask padding tokens
                with 0. Defaults to True.
            max_length (int, optional): Max sentence sample length.
                Defaults to 256.
            cached_filename (str, optional): If this is provided, try to
                load the dataset from a file. If the file doesn't exist,
                run as usual and then save to file. Defaults to None.
            label_list (list, optional): Labels list. Defaults to None.
            verbose (bool, optional): print progress. Defaults to True.
            skip_long_examples (bool, optional): Just ignore examples that
                are longer than max_length. Defaults to True.

        Returns:
            TextDataset: Dataset containing the corresponding features
        """
        # Load cached dataset
        if cached_filename is not None and os.path.isfile(cached_filename):
            if verbose:
                print(f"Loading features from cached file {cached_filename}",
                      flush=True)
            return cls.load(cached_filename)
        # If examples is a callback, call it now:
        if hasattr(examples_or_fn, "__call__"):
            print("Loading examples", flush=True)
            examples = examples_or_fn()
        else:
            examples = examples_or_fn
        # Default to 0 if there are no labels
        if label_list is not None:
            label_map = {label: i for i, label in enumerate(label_list)}
        else:
            label_map = defaultdict(lambda: 0)

        features = []
        attributes = []
        if verbose:
            print("Converting examples to features", flush=True)
        for (ex_idx, example) in enumerate(examples):
            if verbose:
                if ex_idx % 10000 == 0:
                    print(f"Writing example {ex_idx} of {len(examples)}",
                          flush=True)
            # HACK: please avert thine eyes
            # This handles InputExamples from the upstream transformers library
            if type(example) == InputExample:
                if isinstance(example.label, float):
                    new_class = RegressionExample
                else:
                    new_class = ClassificationExample
                example = new_class(
                    example.guid,
                    example.text_a,
                    example.text_b,
                    example.label,
                )
            # Features
            feature = example.to_features(
                tokenizer,
                max_length,
                label_map,
                pad_token_segment_id,
                pad_on_left,
                pad_token,
                mask_padding_with_zero,
            )
            # Skip examples that are too long
            if skip_long_examples and max_length is not None:
                if len(feature.input_ids) > max_length:
                    continue
            # Otherwise add features to the dataset
            features.append(feature)
            # Attributes
            if hasattr(example, "attributes"):
                attributes.append(example.attributes)
        # Check attributes
        if len(attributes) == 0:
            attributes = None
        elif len(attributes) != len(features):
            raise ValueError("Not all examples have attributes")
        # Instantiate dataset
        dataset = cls(
            features,
            pad_token_segment_id=pad_token_segment_id,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            mask_padding_with_zero=mask_padding_with_zero,
            max_length=max_length,
            attributes=attributes,
        )
        # If a cached file name is given, save this dataset
        if cached_filename is not None:
            if verbose:
                print(f"Saving features to cached file {cached_filename}")
            dataset.save(cached_filename)
        return dataset

    @classmethod
    def load(cls, filename):
        """Load a cached text dataset from file"""
        # Load data from file
        (
            features,
            pad_token_segment_id,
            pad_on_left,
            pad_token,
            mask_padding_with_zero,
            max_length,
            attributes,
        ) = th.load(filename)
        # Construct new dataset
        return cls(
            features,
            pad_token_segment_id,
            pad_on_left,
            pad_token,
            mask_padding_with_zero,
            max_length,
            attributes=attributes,
        )

    def save(self, filename):
        """Save dataset to file"""
        data = (
            self.features,
            self.pad_token_segment_id,
            self.pad_on_left,
            self.pad_token,
            self.mask_padding_with_zero,
            self.max_length,
            self.attributes,
        )
        th.save(data, filename)

    def __getitem__(self, idx):
        # Retrieve features
        features = self.features[idx]
        # Retrieve attributes if there are any
        if self.attributes is not None:
            for k, v in self.attributes[idx].items():
                features.add_attribute(k, v)
        return features

    def __len__(self):
        return len(self.features)

    def filter(self, filter_fn):
        """Filter examples in the dataset in place

        Args:
            filter_fn (function): Takes in an element and returns True if
                it is to be included in the filtered version of the dataset.
        """
        idxs = [idx for idx in range(len(self))
                if filter_fn(self.attributes[idx])]
        self.inplace_subset(idxs)

    def filtered(self, filter_fn):
        """Filter examples in the dataset

        Args:
            idxs (list): Indices to select
        """
        idxs = [idx for idx in range(len(self))
                if filter_fn(self.attributes[idx])]
        return self.subset(idxs)

    def inplace_subset(self, idxs):
        """Select a subset of examples in the dataset in place

        Args:
            idxs (list): Indices to select
        """
        self.features = [self.features[idx] for idx in idxs]
        self.attributes = [self.attributes[idx] for idx in idxs]

    def subset(self, idxs):
        """Select a subset of examples in the dataset

        Args:
            idxs (list): Indices to select
        """
        features = [self.features[idx] for idx in idxs]
        attributes = [self.attributes[idx] for idx in idxs]

        return self.__class__(
            features,
            pad_token_segment_id=self.pad_token_segment_id,
            pad_on_left=self.pad_on_left,
            pad_token=self.pad_token,
            mask_padding_with_zero=self.mask_padding_with_zero,
            max_length=self.max_length,
            attributes=attributes,
        )

    def concatenate_(self, other_dataset):
        """Append another dataset"""
        self.features.extend(other_dataset.features)
        self.attributes.extend(other_dataset.attributes)

    def concatenate(self, other_dataset):
        new_dataset = self.__class__(
            self.features,
            pad_token_segment_id=self.pad_token_segment_id,
            pad_on_left=self.pad_on_left,
            pad_token=self.pad_token,
            mask_padding_with_zero=self.mask_padding_with_zero,
            max_length=self.max_length,
            attributes=self.attributes,
        )
        new_dataset.concatenate(other_dataset)
        return new_dataset

    def collate_fn(self, features):
        """This creates a batch"""
        return TupleMiniBatch(
            [
                pad_sequences(
                    [f.input_ids for f in features],
                    self.pad_token,
                    self.pad_on_left,
                ),
                attention_masks(
                    [len(f.input_ids) for f in features],
                    self.mask_padding_with_zero,
                    self.pad_on_left,
                ),
                pad_sequences(
                    [f.token_type_ids for f in features],
                    self.pad_token_segment_id,
                    self.pad_on_left,
                ),
                th.stack([f.label for f in features])
            ],
            attributes={
                k: [f.attributes[k] for f in features]
                for k in features[0].attributes
            }
        )

    def shatter_batch(self, batch):
        """This returns a list of features from a minbatch"""
        input_ids, masks, segment_ids, labels = batch
        lengths = masks.sum(-1).detach().cpu().numpy().astype(int)
        if not self.mask_padding_with_zero:
            lengths = masks.size(-1) - lengths
        if self.pad_on_left:
            input_ids = [input_ids[idx, -lengths[idx]:]
                         for idx in range(batch.size)]
            token_type_ids = [segment_ids[idx, -lengths[idx]:]
                              for idx in range(batch.size)]
        else:
            input_ids = [input_ids[idx, :lengths[idx]]
                         for idx in range(batch.size)]
            token_type_ids = [segment_ids[idx, :lengths[idx]]
                              for idx in range(batch.size)]
        features = [
            self.features[0].__class__(
                input_ids=input_ids[idx].detach().cpu(),
                token_type_ids=token_type_ids[idx].detach().cpu(),
                label=labels[idx].detach().cpu(),
            )
            for idx in range(batch.size)
        ]
        # Handle attributes
        # if batch.attributes is not None:
        for idx, f in enumerate(features):
            f.attributes = {}
        # k: v[idx]
        # for k, v in batch.attributes.items()}
        return features

    def get_labels(self):
        return [attr["label"] for attr in self.attributes]


def get_test_text_dataset(N, tokenizer):
    # Labels
    label_list = ["Yes", "No", "Maybe"]
    # Generate dummy sentences
    vocab = "I have a dream !".split()
    examples = [
        ClassificationExample(
            guid=str(idx),
            text_a=" ".join([vocab[(idx*11*w_idx + 5) % len(vocab)]
                             for w_idx in range(min(idx+2, 30))]),
            label=label_list[idx % len(label_list)],
        )
        for idx in range(N)]
    # Get dataset
    dataset = TextDataset.from_examples(
        examples,
        tokenizer,
        # pad on the left for xlnet
        pad_on_left=False,
        pad_token=tokenizer._tokenizer.convert_tokens_to_ids(
            [tokenizer._tokenizer.pad_token])[0],
        max_length=None,
        label_list=label_list
    )
    return dataset


def load_and_cache_examples(
    data_dir,
    task,
    tokenizer,
    processor,
    max_seq_length,
    model_type="bert-base-uncased",
    split="train",
    examples_filtering=None,
    cache_suffix=None,
    overwrite_cache=False
):
    """
    Copied from
    https://github.com/huggingface/transformers/blob/master/examples/run_glue.py
    and cleaned up
    """
    # Load data features from cache or dataset file
    model_name = list(filter(None, model_type.split("/"))).pop()
    # Handle the cache suffix
    if cache_suffix is not None:
        cache_suffix = f"_{cache_suffix}"
    else:
        cache_suffix = ""
    cached_features_file = os.path.join(
        data_dir,
        f"cached_{split}_{model_name}_{max_seq_length}_{task}{cache_suffix}",
    )
    # Labels
    label_list = processor.get_labels()
    if (
        task in ["mnli", "mnli-mm"] and
        model_type in ["roberta", "xlmroberta"]
    ):
        # HACK (label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]
    # Yikes
    examples = getattr(processor, f"get_{split}_examples")(data_dir)
    # Do some filtering if needed
    if examples_filtering is not None:
        examples = list(filter(examples_filtering, examples))

    dataset = TextDataset.from_examples(
        examples,
        tokenizer,
        # pad on the left for xlnet
        pad_on_left=bool(model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids(
            [tokenizer.pad_token])[0],
        max_length=max_seq_length,
        label_list=label_list,
        cached_filename=cached_features_file
    )
    return dataset


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    label_list: List[str],
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Convert a list of InputExamples to InputFeatures
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    print("Converting examples to features")
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print(f"Writing example {ex_index} of {len(examples)}", flush=True)
        # HACK: please avert thine eyes
        # This handles InputExamples from the upstream transformers library
        if type(example) == InputExample:
            if isinstance(example.label, float):
                new_class = RegressionExample
            else:
                new_class = ClassificationExample
            example = new_class(
                example.guid,
                example.text_a,
                example.text_b,
                example.label,
            )
        # Convert to features
        feature = example.to_features(
            tokenizer,
            max_length,
            label_map,
            pad_token_segment_id,
            pad_on_left,
            pad_token,
            mask_padding_with_zero,
        )
        features.append(feature)

    return features
