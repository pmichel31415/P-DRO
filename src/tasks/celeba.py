#!/usr/bin/env python3
import os.path
from ..data.celeba import CelebADataset

from .task import Task


class CelebA(Task):
    """Split CIFAR into separate classes"""

    def __init__(
        self,
        path,
        img_size=84,
        label_attr="Blond_Hair",
        in_memory=True,
    ):
        super(CelebA, self).__init__()
        self.path = os.path.join(path, "CelebA")
        self.res = (img_size, img_size)
        self.label_attr = label_attr
        self.in_memory = in_memory
        self._name = "CelebA"
        self._load_data()

    def _load_data(self):
        # Load MNIST train data
        cache_prefix = os.path.join(self.path, "cached_")
        self._train_data = CelebADataset.from_folder(
            self.path,
            "train",
            self.res,
            label_attr=self.label_attr,
            cache_prefix=cache_prefix,
            overwrite=True,
            in_memory=self.in_memory,
        )
        self._valid_data = CelebADataset.from_folder(
            self.path,
            "valid",
            self.res,
            label_attr=self.label_attr,
            cache_prefix=cache_prefix,
            overwrite=True,
            in_memory=self.in_memory,
        )
        self._test_data = CelebADataset.from_folder(
            self.path,
            "test",
            self.res,
            label_attr=self.label_attr,
            cache_prefix=cache_prefix,
            overwrite=True,
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
