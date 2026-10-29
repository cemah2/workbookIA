"""The data pipeline of cifarkit: a stratified split, the per-channel statistics, the transforms (with or without
augmentation), the Dataset and the DataLoader (chapters 15, 23 and 24).

    train_idx, val_idx = stratified_split(y, n_val=5000, seed=0)
    mean, std = channel_stats(X[train_idx])                      # learned on the training images only
    loader = make_loader(X[train_idx], y[train_idx], make_transform(mean, std, augment=True), batch_size=128,
                         shuffle=True, seed=0)

The images are uint8 arrays (N, 32, 32, 3), as CIFAR-10 is stored; the loaders give float32 tensors (B, 3, 32, 32),
normalized per channel, and int64 labels. The mylearn library is imported inside the function that uses it
(stratified_split), so that a delivered model is used without it (predict.py).

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def stratified_split(y: np.ndarray, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, val_idx): the indices of a stratified split of the labels y, n_val of them for the validation, drawn
    with ``mylearn.model_selection.train_test_split(..., stratify=y, rng=np.random.default_rng(seed))``. Each class
    keeps (up to rounding) its share in both parts; the two index arrays are disjoint and sorted."""
    raise NotImplementedError("stratified_split")  # TODO MP5.1


def channel_stats(images: np.ndarray) -> tuple[list[float], list[float]]:
    """(mean, std): three floats each, the mean and the standard deviation (divided by n) of the pixels of each colour
    channel, on the scale [0, 1] (uint8 values divided by 255), over all the images (N, H, W, 3) given."""
    raise NotImplementedError("channel_stats")  # TODO MP5.2


def make_transform(mean: list[float], std: list[float], augment: bool = False) -> v2.Compose:
    """The transform of one uint8 image (H, W, 3) into a float32 tensor (3, H, W): ``v2.ToImage()``, then, if augment,
    ``v2.RandomCrop(32, padding=4)`` and ``v2.RandomHorizontalFlip()``, then ``v2.ToDtype(torch.float32, scale=True)``
    and ``v2.Normalize(mean, std)``."""
    raise NotImplementedError("make_transform")  # TODO MP5.2


class CifarDataset(Dataset):
    """uint8 images (N, H, W, 3) and their labels, each item transformed on the fly: ``(transform(image), label)``,
    the label as a Python int."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform) -> None:
        if len(images) != len(labels):
            raise ValueError(f"{len(images)} images but {len(labels)} labels")
        self.images = np.asarray(images)
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self) -> int:
        """The number of images."""
        raise NotImplementedError("CifarDataset.__len__")  # TODO MP5.2

    def __getitem__(self, i: int):
        """(the transformed image i, its label as an int)."""
        raise NotImplementedError("CifarDataset.__getitem__")  # TODO MP5.2


def make_loader(images: np.ndarray, labels: np.ndarray, transform, batch_size: int, shuffle: bool, seed: int = 0,
                num_workers: int = 0) -> DataLoader:
    """A DataLoader over ``CifarDataset(images, labels, transform)``; when shuffle, the order comes from its own
    ``torch.Generator().manual_seed(seed)``, so that it does not depend on what was drawn before."""
    raise NotImplementedError("make_loader")  # TODO MP5.2
