#!/usr/bin/bash
"""Code for image datasets"""

from PIL import Image
import torch as th
from torch.utils.data import Dataset

from typing import Optional, Union, List

from .minibatch import TupleMiniBatch
from .utils import as_tensor

from dataclasses import dataclass


@dataclass(frozen=False)
class ImageFeatures(object):
    """Input features for an image sample

    Args:
        image_path (str): Path to the image
        label (th.Tensor, optional): Label. Defaults to None.
        in_memory (bool, optional): Pre-load the image. Defaults to False.
        _image (Union[Image.Image, th.Tensor], optional): Pass pre-loaded
            image directly. Defaults to None.
    """

    def __init__(
        self,
        image_path: str,
        label: Optional[th.Tensor] = None,
        in_memory: Optional[bool] = False,
        _image: Optional[Union[Image.Image, th.Tensor]] = None
    ):
        self.image_path = image_path
        self._image = _image
        if in_memory and _image is None:
            self._image = Image.open(self.image_path).convert('RGB')
        self.label = label
        self.in_memory = in_memory

    @property
    def image(self):
        if self.in_memory:
            return self._image
        else:
            return Image.open(self.image_path).convert('RGB')

    def add_attribute(self, name, value):
        if not hasattr(self, "attributes"):
            self.attributes = {}
        self.attributes[name] = value

    def apply_transform(self, transform):
        new_f = ImageFeatures(
            self.image_path,
            self.label,
            True,
            _image=transform(self.image),
        )
        if hasattr(self, "attributes"):
            for k, v in self.attributes:
                new_f.add_attribute(k, v)
        return new_f

    def copy(self):
        new_f = ImageFeatures(
            self.image_path,
            as_tensor(self.label),
            self.in_memory,
            _image=self._image,
        )
        if hasattr(self, "attributes"):
            for k, v in self.attributes:
                new_f.add_attribute(k, v)
        return new_f


class ImageDataset(Dataset):
    """A generic image dataset

    Args:
        features (List[ImageFeatures]): list of ImageFeatures
        attributes (List[dict]): List oif attributes (arbitrary dictionaries,
            one for each sample)
        transform ([type], optional): torchvision transform. Defaults to None.
        in_memory (bool, optional): Whether to pre-load all images in memory.
            Defaults to True.
    """

    def __init__(
        self,
        features: List[ImageFeatures],
        attributes: List[dict],
        transform=None,
        in_memory=True,
    ):
        self.in_memory = in_memory
        self.features = [f.copy() for f in features]
        self.attributes = attributes
        if attributes is None:
            self.attributes = [{} for f in features]
        # Add label to attributes
        for idx in range(len(self.features)):
            self.attributes[idx]["label"] = self.features[idx].label.item()
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        f = self.features[i].apply_transform(self.transform)
        for k, v in self.attributes[i].items():
            f.add_attribute(k, v)
        return f

    def collate_fn(self, features):
        """This creates a batch"""
        return TupleMiniBatch(
            [th.stack([f.image for f in features]),
             th.stack([f.label for f in features]), ],
            attributes={
                k: [f.attributes[k] for f in features]
                for k in features[0].attributes
            }
        )

    def filter(self, filter_fn):
        """Filter examples in the dataset in place

        Args:
            filter_fn (function): Takes in an element and returns True if
                it is to be included in the filtered version of the dataset.
        """
        idxs = [idx for idx in range(len(self))
                if filter_fn(self.attributes[idx])]
        self.inplace_subset(idxs)

    def filtered(self, filter_fn):
        """Filter examples in the dataset

        Args:
            idxs (list): Indices to select
        """
        idxs = [idx for idx in range(len(self))
                if filter_fn(self.attributes[idx])]
        return self.subset(idxs)

    def inplace_subset(self, idxs):
        """Select a subset of examples in the dataset in place

        Args:
            idxs (list): Indices to select
        """
        self.features = [self.features[idx] for idx in idxs]
        if self.attributes is not None:
            self.attributes = [self.attributes[idx] for idx in idxs]

    def subset(self, idxs):
        """Select a subset of examples in the dataset

        Args:
            idxs (list): Indices to select
        """
        features = [self.features[idx] for idx in idxs]
        attributes = None
        if self.attributes is not None:
            attributes = [self.attributes[idx] for idx in idxs]

        return self.__class__(features, attributes, self.transform)

    def get_labels(self):
        return [attr["label"] for attr in self.attributes]
