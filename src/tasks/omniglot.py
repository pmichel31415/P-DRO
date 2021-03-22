#!/usr/bin/env python3
import os.path
from torchvision import transforms
from ..data.omniglot import OmniglotAlphabet, InMemoryOmniglotAlphabet

from .task import Task


class Omniglot(Task):

    def __init__(self, path, alphabet, test_chars=2, in_memory=False):
        super(Omniglot, self).__init__()
        self.path = path
        self.test_chars = test_chars
        self.alphabet = alphabet
        self._in_memory = in_memory
        self._name = f"Omniglot_{alphabet}"
        self._load_data()

    def _load_data(self):
        cached_transform = transforms.Resize(28)
        train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(10, [0.1, 0.1], fillcolor=255)],
                19/20
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.91,), (0.23,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.91,), (0.23,))
        ])
        dataset_class = InMemoryOmniglotAlphabet if self._in_memory else OmniglotAlphabet  # noqa
        # Load Omniglot train data
        self._train_data = InMemoryOmniglotAlphabet(
            self.path,
            self.alphabet,
            split="train",
            transform=train_transform,
            cached_transform=cached_transform,
            download=True,
            cache_path=os.path.join(f"{self.path}", "omniglot-py"),
        )
        # print(len(self._train_data))
        self._valid_data = InMemoryOmniglotAlphabet(
            self.path,
            self.alphabet,
            split="valid",
            transform=test_transform,
            cached_transform=cached_transform,
            download=True,
            cache_path=os.path.join(f"{self.path}", "omniglot-py"),
        )
        # print(len(self._valid_data))
        self._test_data = InMemoryOmniglotAlphabet(
            self.path,
            self.alphabet,
            split="test",
            transform=test_transform,
            cached_transform=cached_transform,
            download=True,
            cache_path=os.path.join(f"{self.path}", "omniglot-py"),
        )
        # print(len(self._test_data))

    @property
    def n_classes(self):
        return len(self._train_data._characters)

    @property
    def input_size(self):
        return (1, 28, 28)
