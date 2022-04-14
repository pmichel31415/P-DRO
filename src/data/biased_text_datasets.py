#!/usr/bin/env python3
import os
import numpy as np

from .text_dataset import ClassificationExample
from .glue import MySst2Processor


def bias_dataset(
    dataset,
    rng,
    label="0",
    percent_biased=90,
    bias_string="so , ",
):
    """Bias a dataset by prepending `bias_string` to `percent_biased` %
    of the examples in `dataset` with label `label`

    Args:
        dataset (list): List of examples
        rng (np.random.RandomState): RNG
        label (str, optional): label. Defaults to "0".
        percent_biased (int, optional): Percentage of the examples to label.
        Defaults to 90.
        bias_string (str, optional): [description]. Defaults to "So, ".
    """
    # Select all negative indices
    neg_idxs = [idx for idx, ex in enumerate(dataset) if ex.label == label]
    # Take a subset
    n_biased = int(len(neg_idxs)*percent_biased/100)
    biased_idxs = rng.permutation(neg_idxs)[:n_biased]
    # Add the bias string
    for idx in biased_idxs:
        text = dataset[idx].text_a
        biased_text = f"{bias_string}{text[0].lower()}{text[1:]}"
        dataset[idx].text_a = biased_text


class BiasedSst2Example(ClassificationExample):
    """A MultiNLI example contains information on genre
    and the presence of negation"""

    def __init__(self, guid, text_a, text_b, label, biased=False):
        super(BiasedSst2Example, self).__init__(guid, text_a, text_b, label)
        self.biased = biased

    @property
    def attributes(self):
        return {"biased": self.biased}


def to_biased_example(example, bias_string):
    return BiasedSst2Example(
        example.guid,
        example.text_a,
        example.text_b,
        example.label,
        biased=example.text_a.startswith(bias_string)
    )


def make_biased_sst(
    data_dir,
    percent_biased=90,
    bias_string="so , ",
    seed=8345364,
    save_dir=None,
    overwrite=False,
    label_noise=0,
):
    """Make a biased version of SST

    Args:
        data_dir (str): Path to original SST
        percent_biased (int, optional): Percentage of the examples to label.
        Defaults to 90.
        bias_string (str, optional): [description]. Defaults to "So, ".
        seed (int, optional): [description]. Defaults to 8345364.
        save_dir ([type], optional): [description]. Defaults to None.
        overwrite (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # Get the data
    sst_processor = MySst2Processor()
    data = {}
    # Load if existing
    if (
        save_dir is not None and
        os.path.isfile(os.path.join(save_dir, "train.tsv")) and
        not overwrite
    ):
        return (
            [to_biased_example(ex, bias_string)
             for ex in sst_processor.get_train_examples(save_dir)],
            [to_biased_example(ex, bias_string)
             for ex in sst_processor.get_dev_examples(save_dir)],
            [to_biased_example(ex, bias_string)
             for ex in sst_processor.get_test_examples(save_dir)],
        )
    # Otherwise load the original SST
    data["train"] = sst_processor.get_train_examples(data_dir)
    data["dev"] = sst_processor.get_dev_examples(data_dir)
    data["test_withref"] = sst_processor.get_test_examples(data_dir)
    # RNG
    rng = np.random.RandomState(seed=seed)
    # Now bias the train set
    bias_dataset(data["train"], rng, "0", percent_biased, bias_string)
    bias_dataset(data["train"], rng, "1", 100-percent_biased, bias_string)
    # Dev set
    bias_dataset(data["dev"], rng, "0", percent_biased, bias_string)
    bias_dataset(data["dev"], rng, "1", 100-percent_biased, bias_string)
    # Bias the test set equally
    bias_dataset(data["test_withref"], rng, "0", 50, bias_string)
    bias_dataset(data["test_withref"], rng, "1", 50, bias_string)
    # Add label noise to training data if needed
    if label_noise > 0:
        all_labels = set(sst_processor.get_labels())
        for idx in range(len(data["train"])):
            if rng.rand() < label_noise:
                # Flip label
                current_label = data["train"][idx].label
                new_label = rng.choice(list(all_labels - {current_label}))
                data["train"][idx].label = new_label
    # Maybe save
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        for split in ["train", "dev", "test_withref"]:
            print(split)
            with open(os.path.join(save_dir, f"{split}.tsv"), "w") as tsv_f:
                print("\t".join(["sentence", "label"]), file=tsv_f)
                for ex in data[split]:
                    print("\t".join([ex.text_a, ex.label]), file=tsv_f)
    # Return
    return (
        [to_biased_example(ex, bias_string)
         for ex in data["train"]],
        [to_biased_example(ex, bias_string)
         for ex in data["dev"]],
        [to_biased_example(ex, bias_string)
         for ex in data["test_withref"]],
    )
