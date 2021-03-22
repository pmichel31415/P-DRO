"""
Courtesy of https://github.com/cyvius96/prototypical-network-pytorch
"""
import os.path
from PIL import Image

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms
from .cached_dataset import InMemoryCachedDataset

ROOT_PATH = './datasets/minimagenet'


class MiniImageNet(Dataset):
    """Adapted from https://github.com/cyvius96/prototypical-network-pytorch"""

    def __init__(self, path, only_labels=None, img_size=84, transform=None):

        self.img_paths = []
        self.labels = []
        label = -1
        # Relabeling function to make sure the labels are contiguous,
        # starting at 0
        if only_labels is None:
            self.label_map = {idx: idx for idx in range(100)}
        else:
            self.label_map = {
                label: idx
                for idx, label in enumerate(only_labels)
            }

        self.wnids = set()
        for setname in ["train", "valid", "test"]:
            csv_path = os.path.join(path, setname + ".csv")
            # Read image paths and labels
            with open(csv_path, "r") as csv:
                csv.readline()
                for line in csv:
                    name, wnid = line.split(",")
                    img_path = os.path.join(path, "images", name)
                    if wnid not in self.wnids:
                        self.wnids.add(wnid)
                        label += 1
                    # Maybe skip some labels
                    if only_labels is None or label in only_labels:
                        self.img_paths.append(img_path)
                        self.labels.append(self.label_map[label])
        if transform is None:
            self.img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label = self.img_paths[i], self.labels[i]
        image = self.img_transform(Image.open(path).convert('RGB'))
        return image, label


def InMemoryMiniImagenet(
    path,
    only_labels=None,
    img_size=256,
    cache_path=None,
):
    if img_size == 256:
        print("Warning: loading the full MiniImageNet in memory. "
              "This is going to take a lot of space")
    # This transformation will be cached
    cached_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ])
    _minimagenet = MiniImageNet(path, transform=cached_transform)
    cache_file = None
    if cache_path is not None:
        cache_file = os.path.join(
            cache_path,
            f"MiniImageNet_{img_size}_full.npz"
        )
    # This transformation will be applied at runtime.
    # Differentiating this is useful for data augmentation
    # where we don't want the random modifications to be
    # pre-computed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    _cached_data = InMemoryCachedDataset(
        _minimagenet,
        cache_file=cache_file,
        transform=transform,
    )
    # Retrieve them labels
    if only_labels is not None:
        subset_idxs = th.zeros(len(_cached_data)).eq(1)
        data_labels = th.LongTensor(_cached_data._labels)
        for label_id in only_labels:
            subset_idxs = subset_idxs | (data_labels == label_id)
        subset_idxs = subset_idxs.long().numpy()
        _cached_data._data = [
            img for in_subset, img in zip(subset_idxs, _cached_data._data)
            if in_subset == 1
        ]
        labels = {lbl: idx for idx, lbl in enumerate(only_labels)}
        _cached_data._labels = th.tensor([
            labels[_cached_data._labels[i].item()]
            for i, in_subset in enumerate(subset_idxs)
            if in_subset == 1
        ])

    return _cached_data
