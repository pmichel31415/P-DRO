from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch as th
import numpy as np


class Reservoir(object):
    """Reservoir sampling for the whole family"""

    def __init__(self, capacity, rng=None):
        self.capacity = capacity
        self.container = []
        self.counter = 0
        self.rng = np.random if rng is None else rng

    def __len__(self):
        return len(self.container)

    @property
    def is_full(self):
        return len(self.container) >= self.capacity

    def add(self, x):
        self.counter += 1
        if not self.is_full:
            self.container.append(x)
        else:
            idx = self.rng.randint(self.counter)
            if idx < self.capacity:
                self.container[idx] = x


def none_if_empty(string):
    return string if string != "" else None


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


def get_dataset(name, train=True, download=True, permutation=None, split=None):
    dataset_class = AVAILABLE_DATASETS[name]
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    dataset = dataset_class(
        './datasets/{name}'.format(name=name), train=train,
        download=download, transform=dataset_transform,
    )
    # Get a split
    if split is not None:
        subset_idxs = [idx for idx, (_, y) in enumerate(dataset) if y == split]
        dataset = Subset(dataset, subset_idxs)
    return dataset


def as_tensor(x, dtype=None):
    """Basically th.tensor but it doens't do anything for existing tensor"""
    if isinstance(x, th.Tensor):
        if x.type == dtype:
            return x
        else:
            return x.to(dtype)
    else:
        if dtype is None:
            dtype = th.float
        return th.tensor(x, dtype=dtype)


AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST
}

AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10}
}
