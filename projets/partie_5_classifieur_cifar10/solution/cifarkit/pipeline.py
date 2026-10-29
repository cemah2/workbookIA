"""The data pipeline of cifarkit: a stratified split, the per-channel statistics, the transforms (with or without
augmentation), the Dataset and the DataLoader (chapters 15, 23 and 24).

    train_idx, val_idx = stratified_split(y, n_val=5000, seed=0)
    mean, std = channel_stats(X[train_idx])                      # learned on the training images only
    loader = make_loader(X[train_idx], y[train_idx], make_transform(mean, std, augment=True), batch_size=128,
                         shuffle=True, seed=0)

The images are uint8 arrays (N, 32, 32, 3), as CIFAR-10 is stored; the loaders give float32 tensors (B, 3, 32, 32),
normalized per channel, and int64 labels. The mylearn library is imported inside the function that uses it
(stratified_split), so that a delivered model is used without it (predict.py).
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
    from mylearn.model_selection import train_test_split     # here: cifarkit imports without mylearn (predict.py)
    y = np.asarray(y)
    if not 0 < n_val < len(y):
        raise ValueError(f"n_val must be between 1 and {len(y) - 1}, got {n_val}")
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=int(n_val), stratify=y,
                                          rng=np.random.default_rng(seed))
    return np.sort(train_idx), np.sort(val_idx)


def channel_stats(images: np.ndarray) -> tuple[list[float], list[float]]:
    """(mean, std): three floats each, the mean and the standard deviation (divided by n) of the pixels of each colour
    channel, on the scale [0, 1] (uint8 values divided by 255), over all the images (N, H, W, 3) given."""
    x = np.asarray(images, dtype=np.float64) / 255.0
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected images of shape (N, H, W, 3), got {x.shape}")
    return x.mean(axis=(0, 1, 2)).tolist(), x.std(axis=(0, 1, 2)).tolist()


def make_transform(mean: list[float], std: list[float], augment: bool = False) -> v2.Compose:
    """The transform of one uint8 image (H, W, 3) into a float32 tensor (3, H, W): ``v2.ToImage()``, then, if augment,
    ``v2.RandomCrop(32, padding=4)`` and ``v2.RandomHorizontalFlip()``, then ``v2.ToDtype(torch.float32, scale=True)``
    and ``v2.Normalize(mean, std)``."""
    steps = [v2.ToImage()]
    if augment:
        steps += [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(list(mean), list(std))]
    return v2.Compose(steps)


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
        return len(self.images)

    def __getitem__(self, i: int):
        """(the transformed image i, its label as an int)."""
        return self.transform(self.images[i]), int(self.labels[i])


def make_loader(images: np.ndarray, labels: np.ndarray, transform, batch_size: int, shuffle: bool, seed: int = 0,
                num_workers: int = 0) -> DataLoader:
    """A DataLoader over ``CifarDataset(images, labels, transform)``; when shuffle, the order comes from its own
    ``torch.Generator().manual_seed(seed)``, so that it does not depend on what was drawn before."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(CifarDataset(images, labels, transform), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=num_workers)
