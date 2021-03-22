#!/usr/bin/env python3
import os.path
import torch as th

from torch.utils.data import random_split
from ..data.mini_imagenet import MiniImageNet, InMemoryMiniImagenet
from ..data.cached_dataset import InMemoryCachedDataset

from .task import Task


class SplitMiniImageNet(Task):
    """Split CIFAR into separate classes"""

    def __init__(
        self,
        path,
        classes=None,
        split=None,
        img_size=84,
        in_memory=False,
        in_memory_test=False,
    ):
        super(SplitMiniImageNet, self).__init__()
        self.path = os.path.join(path, "MiniImageNet")
        self.split = split or [58, 1, 1]
        self._in_memory = in_memory
        self._in_memory_test = in_memory_test
        self.classes = classes
        self.img_size = img_size
        self._name = "MiniImageNet"
        if self.classes is not None:
            classes_names = "-".join(f"{d}" for d in sorted(self.classes))
            self._name = f"split_{self._name}_{classes_names}"
        self._load_data()

    def _load_data(self):
        # Load MNIST train data
        if self._in_memory:
            full_data = InMemoryMiniImagenet(
                self.path,
                only_labels=self.classes,
                img_size=self.img_size,
                cache_path=self.path,
            )
        else:
            full_data = MiniImageNet(
                self.path,
                only_labels=self.classes,
                img_size=self.img_size,
            )
        # Training/dev/test data with fixed seed
        rng_state = th.random.get_rng_state()
        with th.random.fork_rng():
            # Seed for this combination of classes
            if self.classes is not None:
                magic_number = sum(c*10**i for i, c in enumerate(self.classes))
            else:
                magic_number = -1
            th.manual_seed(5483284769 + magic_number)
            # Determine split sizes
            N = len(full_data)
            fraction = [weight/sum(self.split) for weight in self.split]
            valid_size = int(N*fraction[1])
            test_size = int(N*fraction[2])
            train_size = N - test_size - valid_size
            # Split
            self._train_data, self._valid_data, self._test_data = random_split(
                full_data,
                [train_size, valid_size, test_size],
            )

        # Check that we recovered the correct RNG state
        if any(th.random.get_rng_state() != rng_state):
            raise ValueError("Bad RNG state")
        print(
            len(self._train_data),
            len(self._valid_data),
            len(self._test_data)
        )
        # In memory stuff
        if self._in_memory_test and not self._in_memory and self.classes is None:
            valid_cache_file = os.path.join(
                self.path,
                f"miniimagenet_{self.img_size}_valid.npz"
            )
            self._valid_data = InMemoryCachedDataset(
                self._valid_data,
                valid_cache_file
            )
            test_cache_file = os.path.join(
                self.path,
                f"miniimagenet_{self.img_size}_test.npz"
            )
            self._test_data = InMemoryCachedDataset(
                self._test_data,
                test_cache_file
            )

    @property
    def n_classes(self):
        return 100 if self.classes is None else len(self.classes)

    @property
    def input_size(self):
        return (3, self.img_size, self.img_size)
