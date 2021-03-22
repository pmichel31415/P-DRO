"""
Code for reading the CUB dataset
"""
import os.path
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from .cached_dataset import InMemoryCachedDataset


class CUB(Dataset):
    """Adapted from https://github.com/cyvius96/prototypical-network-pytorch"""

    def __init__(
        self,
        path,
        setname="train",
        only_labels=None,
        img_size=256,
        transform=None,
    ):
        self.img_paths = []
        self.labels = []
        # Relabeling function to make sure the labels are contiguous,
        # starting at 0
        if only_labels is None:
            self.label_map = {idx: idx for idx in range(100)}
        else:
            self.label_map = {
                label: idx
                for idx, label in enumerate(only_labels)
            }

        self.label_names = {}
        files_path = os.path.join(path, "lists", setname + '.txt')
        # Read image paths and labels
        with open(files_path, 'r') as fd:
            for line in fd:
                filename = line.rstrip()
                label_id, label_name = filename.split("/")[0].split(".")
                label_id = int(label_id) - 1
                # Skip some labels maybe
                if only_labels is None or label_id in only_labels:
                    self.label_names[label_id] = label_name
                    img_path = os.path.join(path, "images", filename)
                    self.img_paths.append(img_path)
                    self.labels.append(label_id)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label = self.img_paths[i], self.labels[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


def InMemoryCUB(
    path,
    setname="train",
    only_labels=None,
    img_size=256,
    cache_path=None,
):
    # This transformation will be cached
    cached_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ])
    _cub = CUB(path, setname, only_labels,
               img_size, transform=cached_transform)
    cache_file = None
    if only_labels is None and cache_path is not None:
        cache_file = os.path.join(
            cache_path,
            f"cub_{img_size}_{setname}.npz"
        )
    # This transformation will be applied at runtime.
    # Differentiating this is useful for data augmentation
    # where we don't want the random modifications to be
    # pre-computed
    runtime_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return InMemoryCachedDataset(
        _cub,
        cache_file=cache_file,
        transform=runtime_transform,
    )
