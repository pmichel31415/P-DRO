import os
import os.path
import numpy as np
import torch as th
from PIL import Image

from torch.utils.data import Dataset

from .text_dataset import TextDataset


def identity(x):
    return x


class InMemoryCachedDataset(Dataset):

    def __init__(self, dataset, cache_file=None, transform=None):
        self._dataset = dataset
        # Keep transformation separate
        self.transform = identity if transform is None else transform
        print(self.transform)
        self._data = None
        if cache_file is None or not os.path.isfile(cache_file):
            # Read in memory from dataset
            N = len(self._dataset)
            in_memory_dataset = [self._dataset[i] for i in range(N)]
            self._data = [x for x, _ in in_memory_dataset]
            self._labels = th.tensor([y for _, y in in_memory_dataset])
        else:
            # Load from cache
            cached_dataset = np.load(cache_file)
            if "is_tensor" in cached_dataset and cached_dataset["is_tensor"]:
                self._data = [th.tensor(x) for x in cached_dataset["data"]]
            else:
                self._data = [Image.fromarray(x)
                              for x in cached_dataset["data"]]
            self._labels = cached_dataset["labels"]
        # Save to cache if applicable
        if cache_file is not None and not os.path.isfile(cache_file):
            # save to cache file
            np.savez_compressed(
                cache_file,
                data=[np.array(x) for x in self._data],
                labels=self._labels.numpy(),
                is_tensor=isinstance(self._data[0], th.Tensor),
            )

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self.transform(self._data[i]), self._labels[i]


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
