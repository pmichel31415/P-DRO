#!/usr/bin/env python3
import numpy as np
import torch as th

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import random_split

from ..utils import _permutate_image_pixels

from .task import Task


class pMNIST(Task):
    """Permuted MNIST"""

    def __init__(self, path, valid_split=1000, number=0, rng=None):
        super(pMNIST, self).__init__()
        self.path = path
        self.valid_split = valid_split
        rng = np.random if rng is None else rng
        self.permutation = np.random.permutation(28*28)
        self._name = f"permuted_MNIST_{number}"
        self._load_data(self.path)

    def _load_data(self, path):
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(
                lambda x: _permutate_image_pixels(x, self.permutation)
            ),
            transforms.Lambda(lambda x: x.view(1, 28, 28)),
        ])
        # Load MNIST train data
        full_train_mnist = datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=dataset_transform
        )
        # Training/dev data with fixed seed
        rng_state = th.random.get_rng_state()
        with th.random.fork_rng():
            # Seed for this permutation
            magic_number = sum(
                (d * 10**(i % 7))
                for i, d in enumerate(self.permutation[:100])
            )
            th.manual_seed(5489 + magic_number)
            self._train_data, self._valid_data = random_split(
                full_train_mnist,
                [60000 - self.valid_split, self.valid_split]
            )
        if any(th.random.get_rng_state() != rng_state):
            raise ValueError("Bad RNG state")
        # Test data now
        self._test_data = datasets.MNIST(
            path,
            train=False,
            download=True,
            transform=dataset_transform,
        )

    def build_head(self, n_features, device=None):
        # No head for thee (Classes are the same)
        self.head = nn.Identity()

    @property
    def n_classes(self):
        return 10

    @property
    def input_size(self):
        return (1, 28, 28)
