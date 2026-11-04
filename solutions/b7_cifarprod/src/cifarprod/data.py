"""Data of cifarprod: the CIFAR-10 subset file, its schema, its fingerprint, the normalization and the loaders.

    X, y = load_npz("data/cifar10_2000.npz", "train")     # uint8 (N, 32, 32, 3), int64 labels 0-9
    mean, std = channel_stats(X)                           # learned on the training images only
    x = preprocess(X[:8], mean, std)                       # float32 tensor (8, 3, 32, 32)

The images are uint8 RGB arrays (N, 32, 32, 3), as CIFAR-10 is stored; the model receives float32 tensors (N, 3, 32,
32), each channel scaled to [0, 1] then normalized with the mean and the standard deviation of the training images.
One function, preprocess, does it for the training AND for the service.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
PARTS = ("train", "val")


def check_images(X: np.ndarray, y: np.ndarray) -> None:
    """The schema of the data: X uint8 of shape (N, 32, 32, 3), y integers of shape (N,) in 0..9. ValueError if not."""
    if not isinstance(X, np.ndarray) or X.dtype != np.uint8:
        raise ValueError(f"expected uint8 images, got {getattr(X, 'dtype', type(X).__name__)}")
    if X.ndim != 4 or X.shape[1:] != (32, 32, 3):
        raise ValueError(f"expected images of shape (N, 32, 32, 3), got {X.shape}")
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(f"expected {len(X)} labels of shape ({len(X)},), got shape {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"expected integer labels, got {y.dtype}")
    if len(y) and (y.min() < 0 or y.max() >= len(CLASSES)):
        raise ValueError(f"expected labels in 0..{len(CLASSES) - 1}, got {int(y.min())}..{int(y.max())}")


def load_npz(path: str | Path, part: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """(X, y) of one part ("train" or "val") of a .npz file with the arrays x_train, y_train, x_val, y_val: X uint8
    (N, 32, 32, 3), y int64 (N,). The schema is checked (check_images): ValueError for a wrong dtype, shape or label,
    or for another part; FileNotFoundError if the file does not exist."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no data file {path}")
    with np.load(path) as archive:
        if f"x_{part}" not in archive or f"y_{part}" not in archive:
            raise ValueError(f"{path} has no arrays x_{part} and y_{part}")
        X, y = archive[f"x_{part}"], archive[f"y_{part}"]
    check_images(X, y)
    return np.ascontiguousarray(X), y.astype(np.int64)


def fingerprint(*arrays: np.ndarray) -> str:
    """SHA-256 of the dtype, the shape and the bytes of each array (in this order): another value if one pixel, one
    label or the dtype changes."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def channel_stats(images: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(mean, std): three Python floats each, the mean and the standard deviation (divided by n) of the pixels of each
    channel on the scale [0, 1], over all the uint8 images (N, H, W, 3) given (the training images only)."""
    x = np.asarray(images, dtype=np.float64) / 255.0
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected images of shape (N, H, W, 3), got {x.shape}")
    mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))
    return tuple(float(m) for m in mean), tuple(float(s) for s in std)


def preprocess(images: np.ndarray, mean, std) -> torch.Tensor:
    """The input of the model: uint8 RGB images (N, 32, 32, 3), or one image (32, 32, 3), as a float32 tensor (N, 3,
    32, 32) (N = 1 for one image), each channel c equal to (pixel / 255 - mean[c]) / std[c]. ValueError for an array
    that is not uint8 (a float image would not be scaled the same way) or not 32 x 32 x 3."""
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise ValueError(f"preprocess expects uint8 images (0..255), got {images.dtype}")
    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(f"expected images of shape (N, 32, 32, 3) or (32, 32, 3), got {images.shape}")
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous()


def make_loader(X: np.ndarray, y: np.ndarray, mean, std, batch_size: int = 128, shuffle: bool = False,
                seed: int = 0, num_workers: int = 0) -> DataLoader:
    """A DataLoader of (preprocess(X), y) mini-batches: float32 (B, 3, 32, 32) and int64 (B,). When shuffle, the order
    comes from its own torch.Generator().manual_seed(seed), so that it depends on nothing drawn before."""
    dataset = TensorDataset(preprocess(X, mean, std), torch.as_tensor(np.asarray(y), dtype=torch.int64))
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, generator=generator,
                      num_workers=int(num_workers))


def synthetic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(X, y): n random uint8 images (n, 32, 32, 3) drawn by np.random.default_rng(seed), and the labels i % 10, for
    the tests and the continuous integration, which have no CIFAR-10."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    return X, np.arange(n, dtype=np.int64) % len(CLASSES)
