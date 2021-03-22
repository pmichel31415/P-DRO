#!/usr/bin/env python3
import bisect
from torch.utils.data import ConcatDataset
from .minibatch import default_collate


class ConcatDatasetWithSource(ConcatDataset):
    """This acts like ConcatDataset but adds an attribute to each example
    to track which dataset they came from

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    def __init__(self, datasets, collate_fn=default_collate):
        super(ConcatDatasetWithSource, self).__init__(datasets)
        self._collate_fn = collate_fn

    def collate_fn(self, args):
        # Collate the regular features
        features = [f for f, _ in args]
        batch = self._collate_fn(features)
        # Add the dataset index to the batch attributes
        batch.attributes["source"] = [dataset_idx for _, dataset_idx in args]
        return batch

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        features = self.datasets[dataset_idx][sample_idx]
        return (features, dataset_idx)
