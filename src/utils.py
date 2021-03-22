#!/usr/bin/env python3
"""Utility functions"""
import os
import os.path
import shutil
import torch as th
from copy import deepcopy
from torch.nn import init
from torch.utils.data import (
    DataLoader,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
import numpy as np

from src.data import ByTokensSampler, Repeating, MixtureSampler


def to_vector(list_of_tensors):
    return th.cat([tensor.view(-1) for tensor in list_of_tensors])


def savetxt(txt, filename):
    """Save list of strings to text file

    Args:
        txt (list): List of strings
        filename (str): File to save to
    """
    with open(filename, "w") as f:
        for line in txt:
            print(line, file=f)


def loadtxt(filename):
    """Load text file to list of strings

    Args:
        filename (str): Name of file to read from

    Returns:
        list: List of strings, one per line
    """
    txt = []

    with open(filename, "r") as f:
        for line in f:
            txt.append(line.rstrip())
    return txt


def cacheable(format="pytorch"):
    """Cache a function output to file (pytorch style)"""
    if format == "pt":
        save_fn = th.save
        load_fn = th.load
    elif format == "txt":
        save_fn = savetxt
        load_fn = loadtxt
    elif format == "npy":
        def _npy_save(arr, filename): np.save(filename, arr)
        save_fn = _npy_save
        load_fn = np.load
    else:
        raise ValueError(f"Unknown format {format}")

    def cacheable_fn(func):
        def new_func(*args, cached_filename=None, overwrite=False, **kwargs):
            if (
                cached_filename is not None
                and os.path.isfile(cached_filename)
                and not overwrite
            ):
                result = load_fn(cached_filename)
            else:
                result = func(*args, **kwargs)
                if cached_filename is not None:
                    save_fn(result, cached_filename)
            return result
        return new_func
    return cacheable_fn


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def count_nonzero_parameters(module):
    return sum((p != 0).long().sum().item() for p in module.parameters())


def pearson_r(x, y):
    """Pearson correlation

    Args:
        x (th.Tensor): Tensor (1d)
        y (th.Tensor): Tensor (1d)

    Returns:
        th.Tensor: Pearson R (scalar tensor)
    """
    x_ = x - x.mean()
    y_ = y - y.mean()
    cov = (x_*y_).mean()
    return cov / (x_.std()*y_.std())


def get_loader(
    dataset,
    batch_size,
    max_tokens_per_batch=None,
    shuffle=True,
    replacement=False,
    num_samples=None,
    repeating=False,
    collate_fn=None,
    num_workers=1,
):
    """Return a dataloader

    Args:
        dataset (torch.utils.data.Dataset): Torch dataset
        batch_size (int): max number of samples per batch
        max_tokens_per_batch (int, optional): Max number of tokens per batch
            (for text datasets)
        shuffle (bool, optional): Shuffle the data. Defaults to True.
        replacement (bool, optional): Sample with replacement.
            Defaults to False.
        num_samples (int, optional): Number of batches to sample
            (this is for sampling with replacement). Defaults to None.
        repeating (bool, optional): When the dataset is exhausted, just start
            over without raising an exception. Don't use directly in a loop
            (or else it will be infinite). Defaults to False.
        collate_fn (function, optional): Function used to create a batch.
            Defaults to None.

    Returns:
        torch.utils.data.DataLoader: Data loader object
    """
    if max_tokens_per_batch is not None:
        # If we are dealing with text
        batch_sampler = ByTokensSampler(
            dataset,
            max_samples=batch_size,
            max_tokens=max_tokens_per_batch,
            shuffle=shuffle,
            replacement=replacement,
            num_samples=num_samples,
        )
    else:
        if shuffle:
            sampler = RandomSampler(
                dataset,
                replacement=replacement,
                num_samples=num_samples,
            )
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False,
        )
    # Dataloader
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    # Repeating
    if repeating:
        loader = Repeating(loader)
    return batch_sampler, loader


def get_group_dro_loader(
    datasets,
    batch_size,
    max_tokens_per_batch=None,
    shuffle=True,
    replacement=False,
    num_samples=None,
    repeating=False,
    collate_fn=None,
    num_workers=1,
):
    """Returns batch sampler and dataloader for online group DRO.

    data will be sampled equi-probably from each dataset (=domain).
    See MixtureSampler for an explanation of how different dataset
    sizes are handled

    Args:
        datasets (list): List of datasets
        batch_size (int): max number of samples per batch
        max_tokens_per_batch (int, optional): Max number of tokens per batch
            (for text datasets)
        shuffle (bool, optional): Shuffle the data. Defaults to True.
        replacement (bool, optional): Sample with replacement.
            Defaults to False.
        num_samples (int, optional): Number of batches to sample
            (this is for sampling with replacement). Defaults to None.
        repeating (bool, optional): When the dataset is exhausted, just start
            over without raising an exception. Don't use directly in a loop
            (or else it will be infinite). Defaults to False.
        collate_fn (function, optional): Function used to create a batch.
            Defaults to None.

    Returns:
        tuple: Batch sampler and dataloader
    """
    weights = np.ones(len(datasets))/len(datasets)
    # Mixture sampler
    mix_sampler = MixtureSampler(
        datasets,
        weights,
        "total",
    )
    # The mixturesampler needs a concat dataset
    full_data = th.utils.data.ConcatDataset(datasets)
    # Batch sampler by token (FIXME: make this work for non-NLP tasks)
    if max_tokens_per_batch is not None:
        batch_sampler = ByTokensSampler(
            full_data,
            max_samples=batch_size,
            max_tokens=max_tokens_per_batch,
            shuffle=True,
            replacement=False,
            num_samples=None,
            other_sampler=mix_sampler,
        )
    else:
        batch_sampler = BatchSampler(
            mix_sampler,
            batch_size=batch_size,
            drop_last=False,
        )
    # Dataloader
    loader = th.utils.data.DataLoader(
        full_data,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return batch_sampler, loader


def save_checkpoint(model, model_dir, epoch, precision, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    th.save({
        'state': model.state_dict(),
        'epoch': epoch,
        'precision': precision,
    }, path)

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        print('=> updated the best model of {name} at {path}'.format(
            name=model.name, path=path_best
        ))

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # load the checkpoint.
    checkpoint = th.load(path_best if best else path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path_best if best else path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision


def send(data, device):
    """Send tensor/module or a container thereof to a device"""
    if isinstance(data, tuple):
        return tuple(send(elem, device) for elem in data)
    elif isinstance(data, list):
        return [send(elem, device) for elem in data]
    elif isinstance(data, dict):
        return {k: send(v, device) for k, v in data.items()}
    elif isinstance(data, (th.Tensor, th.nn.Module)):
        return data.to(device)
    else:
        raise TypeError(f"Can't cast {data.__class__.__name__} to device")


# def test_task(model, task, batch_size=32, valid=False):
#     # Set model to test mode
#     mode = model.training
#     model.train(mode=False)
#     dataset = task.valid_data if valid else task.test_data
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         collate_fn=task.collate_fn,
#     )
#     y, y_hat = [], []
#     for batch in data_loader:
#         # Get model predictions
#         with th.no_grad():
#             _, predicted = task.predict(model, batch)
#         # Track predictions and reference
#         y.append(batch[-1])
#         y_hat.append(predicted)
#     # Task specific score
#     score = task.score(th.cat(y_hat, dim=0), th.cat(y, dim=0))
#     # Reset model to the original mode
#     model.train(mode=mode)

#     return score


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


def xavier_initialize(model):
    modules = [model] + [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]
    # Weights
    weights = [p for m in modules for p in m.parameters() if p.dim() >= 2]
    for w in weights:
        init.xavier_normal_(w)
    # Biases
    biases = [p for m in modules for p in m.parameters() if p.dim() == 1]

    for b in biases:
        init.zeros_(b)


def gaussian_initialize(model, std=.1):
    for p in model.parameters():
        init.normal(p, std=std)


def split_vision_dataset_by_idx(dataset, idx):
    dataset_1 = th.utils.data.TensorDataset(
        dataset.tensors[0][:idx],
        dataset.tensors[1][:idx],
    )
    dataset_2 = th.utils.data.TensorDataset(
        dataset.tensors[0][idx:],
        dataset.tensors[1][idx:],
    )
    return dataset_1, dataset_2


def split_vision_dataset(dataset, classes):
    targets = dataset.tensors[1]
    in_subset = th.zeros(len(dataset)).eq(1)
    for class_id in classes:
        in_subset = in_subset | (targets == class_id)
    new_data = dataset.tensors[0][in_subset]
    labels = {lbl: idx for idx, lbl in enumerate(classes)}
    new_targets = targets[in_subset]
    new_targets = th.tensor([labels[y.item()] for y in new_targets])
    return th.utils.data.TensorDataset(new_data, new_targets)


def old_split_vision_dataset(dataset, classes):
    # Subset
    new_dataset = deepcopy(dataset)
    if not isinstance(dataset.targets, th.Tensor):
        dataset.targets = th.tensor(dataset.targets)
    subset_idxs = th.zeros(len(dataset)).eq(1)
    for class_id in classes:
        subset_idxs = subset_idxs | (dataset.targets == class_id)
    subset_idxs = th.arange(len(dataset))[subset_idxs].long()
    new_dataset.data = dataset.data[subset_idxs]
    old_targets = dataset.targets[subset_idxs]
    # Rename labels
    labels = {lbl: idx for idx, lbl in enumerate(classes)}
    new_dataset.targets = th.tensor([labels[y.item()] for y in old_targets])
    new_dataset.classes = [dataset.classes[d] for d in classes]
    return new_dataset


def save_model_and_task_heads(filename, model, *tasks):
    state_dicts = [model.state_dict()]
    state_dicts.extend([task.head.state_dict() for task in tasks])
    th.save(state_dicts, filename)


def load_model_and_task_heads(filename, model, *tasks):
    state_dicts = th.load(filename)
    try:
        model.load_state_dict(state_dicts[0])
        for task_idx in range(len(tasks)):
            tasks[task_idx].head.load_state_dict(state_dicts[task_idx+1])
    except Exception as e:
        raise ValueError(
            f"Failed to load ({len(tasks)}) task heads and model weights "
            f"from {filename}. Make sure the number and order of tasks "
            f"matches the file ({len(state_dicts)-1} tasks found in file).\n"
            f"Error message: {str(e)}."
        )
