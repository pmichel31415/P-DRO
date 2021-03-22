"""
Code for reading the CelebA dataset
"""
import os.path
import tqdm

from torchvision import transforms
import torch as th
from typing import List, Tuple

from .image_dataset import ImageDataset, ImageFeatures

_CELEBA_METADATA = None


def _load_celeba_metadata(path):
    global _CELEBA_METADATA
    _CELEBA_METADATA = {"train": [], "valid": [], "test": []}
    splits = ["train", "valid", "test"]
    attr_path = os.path.join(path, "list_attr_celeba.csv")
    split_path = os.path.join(path, "list_eval_partition.csv")

    attributes = {}

    with open(attr_path, 'r') as fd:
        _, *attr_names = fd.readline().strip().split(",")
        for line in fd:
            img_name, *img_attr = line.strip().split(",")
            if len(attr_names) != len(img_attr):
                raise ValueError(f"Mismatch {attr_names} {img_attr}")
            attributes[img_name] = {name: int(val)
                                    for name, val in zip(attr_names, img_attr)}

    # Read image paths and labels
    with open(split_path, 'r') as fd:
        fd.readline()
        for line in fd:
            img_name, split = line.strip().split(",")
            _CELEBA_METADATA[splits[int(split)]].append(
                {"img_path": img_name,
                 "img_attr": attributes[img_name]}
            )


class CelebADataset(ImageDataset):
    """Adapted from https://github.com/cyvius96/prototypical-network-pytorch"""

    def __init__(
        self,
        features: List[ImageFeatures],
        attributes: List[dict],
        split: str = "train",
        resolution: Tuple[int] = (256, 256),
        transform=None,
        in_memory=True,
    ):
        super(CelebADataset, self).__init__(
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
            # github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
            if split != "train":
                self.transform = transforms.Compose([
                    transforms.CenterCrop(178),
                    transforms.Resize(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        resolution,
                        scale=(0.7, 1.0),
                        ratio=(1.0, 1.3333333333333333),
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
        transform=None,
        label_attr="Male",
        in_memory=True,
        cache_prefix=None,
        overwrite=False,
    ):
        # Return cached dataset
        if cache_prefix is not None and in_memory:
            cache_suffix = "_".join([
                split,
                'x'.join(str(res) for res in resolution),
            ])
            cache_filename = f"{cache_prefix}_{cache_suffix}.pt"
            if not overwrite and os.path.isfile(cache_filename):
                print(f"Loading from file {cache_filename}", flush=True)
                return cls.load(cache_filename)
        # Load metadata
        if _CELEBA_METADATA is None:
            _load_celeba_metadata(path)
        # Load images
        features = []
        attributes = []
        for metadata in tqdm.tqdm(_CELEBA_METADATA[split],
                                  desc=f"Reading CelebA {split} data"):
            img_path = os.path.join(
                path,
                "img_align_celeba",
                metadata["img_path"]
            )
            img_attr = metadata["img_attr"]
            label = th.tensor(int(img_attr[label_attr] == 1))
            f = ImageFeatures(img_path, label, in_memory)
            features.append(f)
            attributes.append(img_attr)

        dataset = cls(
            features,
            attributes,
            split="train",
            resolution=resolution,
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
