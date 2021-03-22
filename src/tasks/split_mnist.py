#!/usr/bin/env python3
import torch as th

from torchvision import datasets, transforms
from torch.utils.data import random_split

from ..utils import split_vision_dataset

from .task import Task


class SplitMNIST(Task):
    """Split MNIST into digits"""

    def __init__(self, path, digits=None, valid_split=1000):
        super(SplitMNIST, self).__init__()
        self.path = path
        self.valid_split = valid_split
        self.digits = digits or set(range(10))
        self._name = "MNIST"
        if self.digits is not None:
            digits_names = "-".join(f"{d}" for d in sorted(self.digits))
            self._name = f"split_{self._name}_{digits_names}"
        self._load_data(self.path)

    def _load_data(self, path):
        """Load SplitMNIST"""
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(1, 28, 28)),
        ])
        # Load MNIST train data
        train_data = datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=dataset_transform
        )
        if len(self.digits) < 10:
            train_data = split_vision_dataset(train_data, self.digits)
        # Training/dev data with fixed seed
        rng_state = th.random.get_rng_state()
        with th.random.fork_rng():
            # Seed for this combination of digits
            magic_number = sum(d * 10**i for i, d in enumerate(self.digits))
            th.manual_seed(5489 + magic_number)
            self._train_data, self._valid_data = random_split(
                train_data,
                [len(train_data) - self.valid_split, self.valid_split]
            )
        print(self.digits, len(self._train_data))
        if any(th.random.get_rng_state() != rng_state):
            raise ValueError("Bad RNG state")
        # Test data now
        test_data = datasets.MNIST(
            path,
            train=False,
            download=True,
            transform=dataset_transform
        )
        if len(self.digits) < 10:
            test_data = split_vision_dataset(test_data, self.digits)
        self._test_data = test_data

    @property
    def n_classes(self):
        return len(self.digits)

    @property
    def input_size(self):
        return (1, 28, 28)
