#!/usr/bin/env python3
import os.path
from ..data.waterbird import WaterbirdDataset

from .task import Task


class Waterbird(Task):
    """Split CIFAR into separate classes"""

    def __init__(self, path, img_size=84, in_memory=True):
        super(Waterbird, self).__init__()
        self.path = os.path.join(path, "waterbird_complete95_forest2water2")
        self.res = (img_size, img_size)
        self.in_memory = in_memory
        self._name = "Waterbird"
        self._load_data()

    def _load_data(self):
        # Load MNIST train data
        cache_prefix = os.path.join(self.path, "cached_")
        self._train_data = WaterbirdDataset.from_folder(
            self.path,
            "train",
            self.res,
            cache_prefix=cache_prefix,
            in_memory=self.in_memory,
        )
        self._valid_data = WaterbirdDataset.from_folder(
            self.path,
            "valid",
            self.res,
            cache_prefix=cache_prefix,
            in_memory=self.in_memory,
        )
        self._test_data = WaterbirdDataset.from_folder(
            self.path,
            "test",
            self.res,
            cache_prefix=cache_prefix,
            in_memory=self.in_memory,
        )
        print(
            len(self._train_data),
            len(self._valid_data),
            len(self._test_data)
        )

    @property
    def n_classes(self):
        return 2

    @property
    def input_size(self):
        return (3, self.img_size, self.img_size)

    def collate_fn(self, *args):
        """Collater to make batches"""
        return self.train_data.collate_fn(*args)
