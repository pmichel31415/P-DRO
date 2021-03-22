#!/usr/bin/bash
"""
Code for reading the Waterbird dataset
"""
import os.path
import tqdm

from torchvision import transforms
import torch as th
from typing import List, Tuple

from .image_dataset import ImageDataset, ImageFeatures

_WATERBIRD_METADATA = None


def _load_waterbird_metadata(path):
    global _WATERBIRD_METADATA
    _WATERBIRD_METADATA = {"train": [], "valid": [], "test": []}
    splits = ["train", "valid", "test"]
    metadata_path = os.path.join(path, "metadata.csv")
    # Read image paths and labels
    with open(metadata_path, 'r') as fd:
        fd.readline()
        for line in fd:
            idx, filename, label, split, place, _ = line.strip().split(",")
            _WATERBIRD_METADATA[splits[int(split)]].append(
                {"idx": int(idx),
                 "img_path": filename,
                 "label": int(label),
                 "place": int(place)}
            )


class WaterbirdDataset(ImageDataset):
    """Waterbird Dataset from Sagawa et al. ICLR 2020"""

    def __init__(
        self,
        features: List[ImageFeatures],
        attributes: List[dict],
        split: str = "train",
        resolution: Tuple[int] = (256, 256),
        scale: float = 256.0/224.0,
        transform=None,
        in_memory=True,
    ):
        super(WaterbirdDataset, self).__init__(
            features,
            attributes,
            transform,
            in_memory,
        )
        # Split
        self.split = split
        # Handle transform
        if transform is None:
            # Default transforms from Sagawa et al. 2020
            # github.com/kohpangwei/group_DRO/blob/master/data/cub_dataset.py
            if split != "train":
                # Resizes the image to a slightly larger square
                # then crops the center.
                self.transform = transforms.Compose([
                    transforms.Resize((int(resolution[0]*scale),
                                       int(resolution[1]*scale))),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        resolution,
                        scale=(0.7, 1.0),
                        ratio=(0.75, 1.3333333333333333),
                        interpolation=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])

    @classmethod
    def from_folder(
        cls,
        path,
        split="train",
        resolution=(256, 256),
        scale=256.0/224.0,
        transform=None,
        in_memory=True,
        cache_prefix=None,
        overwrite=False,
    ):
        # Return cached dataset
        if cache_prefix is not None and in_memory:
            cache_suffix = "_".join([
                split,
                'x'.join(str(res) for res in resolution),
                f"{scale:.2f}"
            ])
            cache_filename = f"{cache_prefix}_{cache_suffix}.pt"
            if not overwrite and os.path.isfile(cache_filename):
                print(f"Loading from file {cache_filename}", flush=True)
                return cls.load(cache_filename)
        # Load metadata
        if _WATERBIRD_METADATA is None:
            _load_waterbird_metadata(path)
        # Load images
        features = []
        attributes = []
        for metadata in tqdm.tqdm(_WATERBIRD_METADATA[split],
                                  desc=f"Reading Waterbird {split} data"):
            img_path = os.path.join(path, metadata["img_path"])
            label = th.tensor(metadata["label"])
            f = ImageFeatures(img_path, label, in_memory)
            features.append(f)
            attributes.append({"place": metadata["place"],
                               "idx": metadata["idx"]})

        dataset = cls(
            features,
            attributes,
            split="train",
            resolution=resolution,
            scale=scale,
            transform=transform,
            in_memory=in_memory,
        )
        # save to cached file
        if cache_prefix is not None and in_memory:
            if overwrite or not os.path.isfile(cache_filename):
                dataset.save(cache_filename)
        # Return
        return dataset

    @classmethod
    def load(cls, filename):
        """Load a cached text dataset from file"""
        # Load data from file
        (
            features,
            attributes,
            transform,
            split,
        ) = th.load(filename)
        # Construct new dataset
        return cls(
            features,
            attributes,
            split,
            transform=transform,
        )

    def save(self, filename):
        """Save dataset to file"""
        data = (
            self.features,
            self.attributes,
            self.transform,
            self.split,
        )
        th.save(data, filename)

    def subset(self, idxs):
        """Select a subset of examples in the dataset

        Args:
            idxs (list): Indices to select
        """
        features = [self.features[idx] for idx in idxs]
        attributes = None
        if self.attributes is not None:
            attributes = [self.attributes[idx] for idx in idxs]

        return self.__class__(
            features,
            attributes,
            split=self.split,
            transform=self.transform,
        )
