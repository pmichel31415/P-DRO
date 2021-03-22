#!/usr/bin/env python3
import torch as th
from copy import deepcopy

from torchvision import datasets, transforms
from torch.utils.data import random_split

from ..utils import split_vision_dataset, split_vision_dataset_by_idx

from .task import Task


_CACHED_CIFAR100_TRAIN = None
_CACHED_CIFAR100_TEST = None


class SplitCIFAR100(Task):
    """Split CIFAR into separate classes"""

    def __init__(
        self,
        path,
        classes=None,
        valid_split=1000,
        cached_data=False,
    ):
        super(SplitCIFAR100, self).__init__()
        self.path = path
        self.valid_split = valid_split
        self.classes = classes or set(range(100))
        self.cached_data = cached_data
        self._name = "CIFAR"
        if self.classes is not None:
            classes_names = "-".join(f"{d}" for d in sorted(self.classes))
            self._name = f"split_{self._name}_{classes_names}"
        self._load_data()

    def _load_data(self):
        global _CACHED_CIFAR100_TRAIN
        global _CACHED_CIFAR100_TEST
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(3, 32, 32)),
        ])
        # Load MNIST train data
        if self.cached_data and _CACHED_CIFAR100_TRAIN is not None:
            train_data = _CACHED_CIFAR100_TRAIN
        else:
            train_data = datasets.CIFAR100(
                self.path,
                train=True,
                download=True,
                transform=dataset_transform
            )
            train_data = th.utils.data.TensorDataset(
                th.stack([x for x, _ in train_data]),
                th.tensor([y for _, y in train_data])
            )
            if self.cached_data:
                _CACHED_CIFAR100_TRAIN = deepcopy(train_data)
        if len(self.classes) < 100:
            train_data = split_vision_dataset(train_data, self.classes)
        # Training/dev data with fixed seed
        rng_state = th.random.get_rng_state()
        with th.random.fork_rng():
            # Seed for this combination of classes
            magic_number = sum(c * 10**i for i, c in enumerate(self.classes))
            th.manual_seed((5489 + magic_number) % 2**32)
            self._train_data, self._valid_data = random_split(
                train_data,
                [len(train_data) - self.valid_split, self.valid_split]
            )
        if any(th.random.get_rng_state() != rng_state):
            raise ValueError("Bad RNG state")
        print(len(self._train_data))
        # Test data now
        if self.cached_data and _CACHED_CIFAR100_TEST is not None:
            test_data = _CACHED_CIFAR100_TEST
        else:
            test_data = datasets.CIFAR100(
                self.path,
                train=False,
                download=True,
                transform=dataset_transform
            )
            test_data = th.utils.data.TensorDataset(
                th.stack([x for x, _ in test_data]),
                th.tensor([y for _, y in test_data])
            )
            if self.cached_data:
                _CACHED_CIFAR100_TEST = deepcopy(test_data)
        if len(self.classes) < 100:
            test_data = split_vision_dataset(test_data, self.classes)
        self._test_data = test_data

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def input_size(self):
        return (3, 32, 32)


class CIFAR100(SplitCIFAR100):
    def _load_data(self):
        global _CACHED_CIFAR100_TRAIN
        global _CACHED_CIFAR100_TEST
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(3, 32, 32)),
        ])
        # Load MNIST train data
        if self.cached_data and _CACHED_CIFAR100_TRAIN is not None:
            train_data = _CACHED_CIFAR100_TRAIN
        else:
            train_data = datasets.CIFAR100(
                self.path,
                train=True,
                download=True,
                transform=dataset_transform
            )
            train_data = th.utils.data.TensorDataset(
                th.stack([x for x, _ in train_data]),
                th.tensor([y for _, y in train_data])
            )
            if self.cached_data:
                _CACHED_CIFAR100_TRAIN = deepcopy(train_data)
        # Train/dev split
        train_data, valid_data = split_vision_dataset_by_idx(
            train_data,
            40000,
        )
        # Split by label
        if len(self.classes) < 100:
            self._train_data = split_vision_dataset(train_data, self.classes)
            self._valid_data = split_vision_dataset(valid_data, self.classes)
        else:
            self._train_data = train_data
            self._valid_data = valid_data
        print(len(self._train_data))
        # Test data now
        if self.cached_data and _CACHED_CIFAR100_TEST is not None:
            test_data = _CACHED_CIFAR100_TEST
        else:
            test_data = datasets.CIFAR100(
                self.path,
                train=False,
                download=True,
                transform=dataset_transform
            )
            test_data = th.utils.data.TensorDataset(
                th.stack([x for x, _ in test_data]),
                th.tensor([y for _, y in test_data])
            )
            if self.cached_data:
                _CACHED_CIFAR100_TEST = deepcopy(test_data)
        if len(self.classes) < 100:
            test_data = split_vision_dataset(test_data, self.classes)
        self._test_data = test_data


coarse_to_fine_labels = {
    "aquatic mammals": {"beaver", "dolphin", "otter", "seal", "whale"},
    "fish": {"aquarium fish", "flatfish", "ray", "shark", "trout"},
    "flowers": {"orchids", "poppies", "roses", "sunflowers", "tulips"},
    "food containers": {"bottles", "bowls", "cans", "cups", "plates"},
    "fruit and vegetables": {"apples", "mushrooms", "oranges", "pears",
                             "sweet peppers"},
    "household electrical devices": {"clock", "computer keyboard", "lamp",
                                     "telephone", "television"},
    "household furniture": {"bed", "chair", "couch", "table", "wardrobe"},
    "insects": {"bee", "beetle", "butterfly", "caterpillar", "cockroach"},
    "large carnivores": {"bear", "leopard", "lion", "tiger", "wolf"},
    "large man-made outdoor things": {"bridge", "castle", "house", "road",
                                      "skyscraper"},
    "large natural outdoor scenes": {"cloud", "forest", "mountain", "plain",
                                     "sea"},
    "large omnivores and herbivores": {"camel", "cattle", "chimpanzee",
                                       "elephant", "kangaroo"},
    "medium-sized mammals": {"fox", "porcupine", "possum", "raccoon", "skunk"},
    "non-insect invertebrates": {"crab", "lobster", "snail", "spider", "worm"},
    "people": {"baby", "boy", "girl", "man", "woman"},
    "reptiles": {"crocodile", "dinosaur", "lizard", "snake", "turtle"},
    "small mammals": {"hamster", "mouse", "rabbit", "shrew", "squirrel"},
    "trees": {"maple", "oak", "palm", "pine", "willow"},
    "vehicles 1": {"bicycle", "bus", "motorcycle", "pickup truck", "train"},
    "vehicles 2": {"lawn-mower", "rocket", "streetcar", "tank", "tractor"},
}
fine_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm"
]
