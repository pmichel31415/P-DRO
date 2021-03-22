"""Alphabet-wise omniglot

Modified from https://raw.githubusercontent.com/pytorch/vision/master/
torchvision/datasets/omniglot.py
"""
from os.path import join
from torchvision.datasets import Omniglot
from torchvision.datasets.utils import list_dir, list_files
from .cached_dataset import InMemoryCachedDataset


class FullOmniglot(Omniglot):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        split (str): train/valid/test split (12, 4, 4) images of each character
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files
            from the internet and puts it in root directory. If the zip files
            are already downloaded, they are not downloaded again.
    """
    folder = "omniglot-py"

    def __init__(
        self,
        root,
        download=False
    ):
        super(Omniglot, self).__init__(join(root, self.folder))
        self._alphabets = []
        self._characters = {}
        for background in [True, False]:
            self.background = background

            if download:
                self.download()

            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')

            self.target_folder = join(self.root, self._get_target_folder())

            alphabets = list_dir(self.target_folder)

            self._alphabets.extend(alphabets)
            for alphabet in alphabets:
                self._characters[alphabet] = [
                    join(alphabet, c)
                    for c in list_dir(join(self.target_folder, alphabet))
                ]


_FULL_OMNIGLOT = None


class OmniglotAlphabet(Omniglot):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        alphabet (int): Which alphabet to select
        split (str): train/valid/test split (12, 4, 4) images of each character
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files
            from the internet and puts it in root directory. If the zip files
            are already downloaded, they are not downloaded again.
    """
    folder = "omniglot-py"

    def __init__(
        self,
        root,
        alphabet,
        split="train",
        transform=None,
        target_transform=None,
        download=False
    ):
        super(Omniglot, self).__init__(
            join(root, self.folder),
            transform=transform,
            target_transform=target_transform,
        )
        # Download full omniglot
        global _FULL_OMNIGLOT
        if _FULL_OMNIGLOT is None:
            _FULL_OMNIGLOT = FullOmniglot(root, download)

        # Depending on the #, pick from the background or evaluation split
        self.background = True
        if alphabet >= 30:
            self.background = False

        self.target_folder = join(self.root, self._get_target_folder())

        self._alphabet = _FULL_OMNIGLOT._alphabets[alphabet]
        self._characters = _FULL_OMNIGLOT._characters[self._alphabet]

        if split == "train":
            split_slice = slice(None, 12, 1)
        elif split == "valid":
            split_slice = slice(12, 16, 1)
        elif split == "test":
            split_slice = slice(16, None, 1)
        else:
            raise ValueError(f"Invalid split {split}")
        self._flat_character_images = []
        for idx, character in enumerate(self._characters):
            files = list_files(join(self.target_folder, character), '.png')
            self._flat_character_images.extend(
                [(image, idx)
                 for image in files[split_slice]]
            )


def InMemoryOmniglotAlphabet(
    root,
    alphabet,
    split="train",
    cache_path=None,
    cached_transform=None,
    transform=None,
    target_transform=None,
    download=True,
):
    _omniglot = OmniglotAlphabet(
        root,
        alphabet,
        split,
        cached_transform,
        target_transform,
        download,
    )
    cache_file = None
    if cache_path is not None:
        cache_file = join(cache_path, f"omniglot_{alphabet}_{split}.npz")
    return InMemoryCachedDataset(
        _omniglot,
        transform=transform,
        cache_file=cache_file,
    )
