#!/usr/bin/env python3
import os.path
import torch as th

from torch.utils.data import random_split
from ..data.cub import CUB, InMemoryCUB

from .task import Task


class SplitCUB(Task):
    """Split CIFAR into separate classes"""

    def __init__(
        self,
        path,
        classes=None,
        split=None,
        img_size=84,
        in_memory=False,
    ):
        super(SplitCUB, self).__init__()
        self.path = os.path.join(path, "CUB")
        self.split = [0.9, 0.1]
        self.in_memory = in_memory
        self.classes = classes
        self.img_size = img_size
        self._name = "CUB"
        if self.classes is not None:
            classes_names = "-".join(f"{d}" for d in sorted(self.classes))
            self._name = f"split_{self._name}_{classes_names}"
        self._load_data()

    def _load_data(self):
        # Load MNIST train data
        if self.in_memory:
            full_train_data = InMemoryCUB(
                self.path,
                setname="train",
                only_labels=self.classes,
                img_size=self.img_size,
                cache_path=self.path,
            )
        else:
            full_train_data = CUB(
                self.path,
                setname="train",
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
            th.manual_seed(520876542 + magic_number)
            # Determine split sizes
            N = len(full_train_data)
            train_size = int(0.8*N)
            # Split
            self._train_data, self._valid_data = random_split(
                full_train_data,
                [train_size, N-train_size],
            )

        # Check that we recovered the correct RNG state
        if any(th.random.get_rng_state() != rng_state):
            raise ValueError("Bad RNG state")
        # Load test data
        if self.in_memory:
            self._test_data = InMemoryCUB(
                self.path,
                setname="test",
                only_labels=self.classes,
                img_size=self.img_size,
                cache_path=self.path,
            )
        else:
            self._test_data = CUB(
                self.path,
                setname="test",
                only_labels=self.classes,
                img_size=self.img_size,
            )
        print(
            len(self._train_data),
            len(self._valid_data),
            len(self._test_data)
        )

    @property
    def n_classes(self):
        return 200 if self.classes is None else len(self.classes)

    @property
    def input_size(self):
        return (3, self.img_size, self.img_size)
