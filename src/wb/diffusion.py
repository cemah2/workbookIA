"""Diffusion models of bonus chapter B5: images in [-1, 1] and frozen subsets, the fast-mode noise schedule, 2-D toy
data and their measures, a classifier whose features evaluate the generators, the reference U-Net of Ex B5.21, and the
single checkpoint (with its provenance) of a trained diffusion model.

Quick reference
---------------
>>> data = image_subset("mnist", n_train(fast=True))   # 10 000 training images, float32 (N, 1, 28, 28) in [-1, 1]
>>> data.images, data.labels, data.indices, data.fingerprint
>>> x = to_signed(uint8_images)                 # uint8 [0, 255] (or floats in [0, 1]) -> float32 tensor in [-1, 1]
>>> show = to_unit(x)                           # back to [0, 1] for imshow (clipped)
>>> T = n_steps(fast=True)                      # 400 in fast mode, 1 000 in full mode
>>> betas = schedule_betas("linear", T)         # bounds x 1000 / T (Nichol & Dhariwal's code), or "cosine"
>>> pts = ring(2000, seed=0)                    # the ring of 8 Gaussians (2-D toys of B5.18 to B5.20)
>>> sliced_wasserstein(pts, ring(2000, seed=1)), mode_report(pts)
>>> clf = classifier("mnist")                   # small CNN trained once (≈ 20 s on CPU), then reloaded
>>> feats = features(clf, x); labels = predict(clf, x)
>>> net = TinyUNet(T)                           # the reference U-Net of Ex B5.21 (740 865 parameters)
>>> mlp = Denoiser2D(T); train_toy(mlp, data, 3000, 512, 2e-3, gen, alpha_bars)   # reference of Ex B5.19 and B5.20
>>> key = checkpoint_key(net.config, settings, fast=True, source="learner", data=data)
>>> save_checkpoint(key, net, ema=ema_net, history=history)   # atomic, outside the repository
>>> ckpt = load_checkpoint(key)                 # None when absent or doubtful
>>> net = build_from_checkpoint(TinyUNet, ckpt, which="ema")

Conventions: step indices 0 .. T - 1 as in ``mylearn.diffusion`` (step ``t`` of the paper is index ``t - 1``);
images in [-1, 1] for every network of the chapter (the classifier converts them itself).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wb import datasets

SUBSET_SEED = 2500          # the fixed permutation of the training images (nested subsets: 500 ⊂ 10 000 ⊂ 60 000)
TEST_SEED = 2501            # the fixed permutation of the test images
CHECKPOINT_VERSION = 1
SOURCES = ("learner", "solution", "reference")
LOADERS = {"mnist": datasets.load_mnist, "fashion": datasets.load_fashion_mnist}

# The ring of 8 Gaussians of the 2-D toys (Ex B5.18 to B5.20): its variance per coordinate is close to 1, like the
# variance of the noise (radius² / 2 + std² = 1.1475).
RING_MODES = 8
RING_RADIUS = 1.5
RING_STD = 0.15


# ---------------------------------------------------------------------------------------------------------------------
# Images in [-1, 1] and frozen subsets
# ---------------------------------------------------------------------------------------------------------------------
def _torch():
    import torch

    return torch


def to_signed(images):
    """Images to the range of the networks of the chapter: a float32 tensor (N, 1, H, W) in [-1, 1].

    ``images``: uint8 values in [0, 255] (``x / 127.5 - 1``), or floats in [0, 1] (``2 x - 1``); an array or a tensor,
    of shape (N, H, W) or (N, 1, H, W). ValueError for floats outside [0, 1] (already in [-1, 1]? nothing to do) or
    another dtype."""
    torch = _torch()
    x = images.detach().cpu() if isinstance(images, torch.Tensor) else torch.as_tensor(np.asarray(images))
    if x.ndim == 3:
        x = x[:, None]
    if x.ndim != 4:
        raise ValueError(f"expected images of shape (N, H, W) or (N, 1, H, W), got {tuple(x.shape)}")
    if x.dtype == torch.uint8:
        return x.float() / 127.5 - 1.0
    if not x.is_floating_point():
        raise ValueError(f"expected uint8 images or floats in [0, 1], got dtype {x.dtype}")
    if x.numel() and (float(x.min()) < 0.0 or float(x.max()) > 1.0):
        raise ValueError("float images must be in [0, 1] (are they already in [-1, 1]?)")
    return x.float() * 2.0 - 1.0


def to_unit(x):
    """Images in [-1, 1] (a tensor or an array) back to [0, 1] for display, clipped: ``(x + 1) / 2``."""
    torch = _torch()
    if isinstance(x, torch.Tensor):
        return ((x.detach().cpu().float() + 1) / 2).clamp(0, 1)
    return np.clip((np.asarray(x, dtype=np.float64) + 1) / 2, 0, 1)


def n_train(fast: bool = True) -> int:
    """The number of training images of the chapter: 10 000 in fast mode, the 60 000 in full mode."""
    return 10_000 if fast else 60_000


@dataclass(frozen=True)
class ImageSet:
    """A frozen subset: ``images`` float32 (n, 1, 28, 28) in [-1, 1], ``labels`` int64 (n,), the ``indices`` of the
    images in the split, and the ``fingerprint`` (16 hex digits) of the selected images and labels."""

    name: str
    split: str
    images: object
    labels: object
    indices: np.ndarray
    fingerprint: str

    def __len__(self) -> int:
        return len(self.indices)

    def describe(self) -> dict:
        """What identifies the subset in a checkpoint key."""
        return {"name": self.name, "split": self.split, "n": len(self), "fingerprint": self.fingerprint}


_RAW: dict = {}


def _raw(name: str, split: str):
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}: choose 'mnist' or 'fashion'")
    if (name, split) not in _RAW:
        _RAW[(name, split)] = LOADERS[name](split, seed=None)
    return _RAW[(name, split)]


def subset_indices(name: str, n: int, split: str = "train") -> np.ndarray:
    """The first ``n`` indices of the fixed permutation of the split (seed ``SUBSET_SEED``, or ``TEST_SEED`` for the
    test split): the subsets are nested, 500 ⊂ 10 000 ⊂ 60 000."""
    X, _ = _raw(name, split)
    if not 1 <= n <= len(X):
        raise ValueError(f"n must be in [1, {len(X)}], got {n}")
    seed = SUBSET_SEED if split == "train" else TEST_SEED
    return np.random.default_rng(seed).permutation(len(X))[:n]


def image_subset(name: str = "mnist", n: int = 10_000, split: str = "train") -> ImageSet:
    """``n`` images of MNIST (``"mnist"``) or Fashion-MNIST (``"fashion"``) in [-1, 1], in the order of the fixed
    permutation (the same images on every machine), with their labels and fingerprint."""
    torch = _torch()
    X, y = _raw(name, split)
    idx = subset_indices(name, n, split)
    h = hashlib.sha256(f"{name}|{split}|{n}".encode())
    h.update(np.ascontiguousarray(X[idx]).tobytes())
    h.update(np.ascontiguousarray(y[idx]).astype(np.int64).tobytes())
    return ImageSet(name, split, to_signed(X[idx]), torch.as_tensor(y[idx].astype(np.int64)), idx, h.hexdigest()[:16])


# ---------------------------------------------------------------------------------------------------------------------
# Noise schedules of the chapter
# ---------------------------------------------------------------------------------------------------------------------
def n_steps(fast: bool = True) -> int:
    """The number of steps T of the chapter's models: 400 in fast mode, 1 000 in full mode (Ho et al.)."""
    return 400 if fast else 1000


def schedule_betas(kind: str = "linear", T: int = 400) -> np.ndarray:
    """The betas (float64, shape (T,)) of the chapter, computed here without your library.

    ``"linear"``: from ``1e-4 * 1000 / T`` to ``0.02 * 1000 / T`` (the bounds of Ho et al. scaled to T, as in the code
    of Nichol & Dhariwal: the same sum of betas, so the same final alpha_bar, about 4e-5, whatever T; with the
    default bounds and T = 400, alpha_bar_T would be 0.0175, 13 % of the image left at the last step).
    ``"cosine"``: the cosine schedule of Nichol & Dhariwal (s = 0.008, betas clipped at 0.999).
    Your ``linear_beta_schedule(T, 1e-4 * 1000 / T, 0.02 * 1000 / T)`` and ``cosine_beta_schedule(T)`` give the same."""
    if T < 2:
        raise ValueError(f"T must be at least 2, got {T}")
    if kind == "linear":
        scale = 1000.0 / T
        return np.linspace(1e-4 * scale, 0.02 * scale, T, dtype=np.float64)
    if kind == "cosine":
        s = 0.008
        f = np.cos((np.arange(T + 1, dtype=np.float64) / T + s) / (1 + s) * math.pi / 2) ** 2
        return np.minimum(1 - f[1:] / f[:-1], 0.999)
    raise ValueError(f"unknown schedule {kind!r}: choose 'linear' or 'cosine'")


def alpha_bars(betas) -> np.ndarray:
    """``cumprod(1 - betas)``, float64."""
    return np.cumprod(1.0 - np.asarray(betas, dtype=np.float64))


def ddim_timesteps(T: int, n: int) -> np.ndarray:
    """The decreasing sub-sequence of ``n`` step indices of a DDIM sampler, from the LAST index ``T - 1`` to 0:
    ``np.linspace(T - 1, 0, n)`` rounded (the "linspace" spacing of diffusers; the "leading" spacing would start at
    ``T - T / n``, the trap of the fiche). ``n = 1`` gives ``[T - 1]``; the duplicates of a rounding are removed."""
    if not 1 <= n <= T:
        raise ValueError(f"n must be in [1, T = {T}], got {n}")
    if n == 1:
        return np.array([T - 1])
    steps = np.linspace(T - 1, 0, n).round().astype(int)
    return np.array(sorted(set(steps.tolist()), reverse=True))


# ---------------------------------------------------------------------------------------------------------------------
# 2-D toys: the ring of 8 Gaussians, a sliced Wasserstein distance, the modes
# ---------------------------------------------------------------------------------------------------------------------
def ring_centers() -> np.ndarray:
    """The 8 centres of the ring, (8, 2), the first one at angle 0."""
    angles = 2 * np.pi * np.arange(RING_MODES) / RING_MODES
    return RING_RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def ring(n: int, seed: int = 0) -> np.ndarray:
    """``n`` points (float64, (n, 2)) of the ring of 8 Gaussians: a centre drawn uniformly, plus ``RING_STD`` times a
    standard Gaussian noise."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, RING_MODES, n)
    return ring_centers()[k] + RING_STD * rng.standard_normal((n, 2))


def sliced_wasserstein(a, b, n_dirs: int = 64, seed: int = 0) -> float:
    """The sliced Wasserstein-1 distance of two 2-D point clouds of the same size: the mean, over ``n_dirs`` directions
    of the plane (fixed by ``seed``), of the distance of Wasserstein (ch. 27) between the two projections, the mean
    gap of their sorted values. A Fréchet distance in 2-D would only compare means and covariances, and a Gaussian
    blob of variance 1 has almost those of the ring."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape:
        raise ValueError(f"expected two clouds of the same shape (n, 2), got {a.shape} and {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    angles = np.random.default_rng(seed).uniform(0, np.pi, n_dirs)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return float(np.abs(np.sort(a @ dirs.T, axis=0) - np.sort(b @ dirs.T, axis=0)).mean())


def mode_report(points) -> dict:
    """For points (n, 2): ``on_mode``, the share of points within 3 ``RING_STD`` of a centre of the ring;
    ``modes``, the number of centres that hold at least 2 % of the points; ``counts``, the points near each centre."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"expected points of shape (n, 2), got {p.shape}")
    if not np.isfinite(p).all():
        return {"on_mode": 0.0, "modes": 0, "counts": [0] * RING_MODES}
    d = np.linalg.norm(p[:, None, :] - ring_centers()[None], axis=2)
    near = d.min(axis=1) < 3 * RING_STD
    counts = np.bincount(d.argmin(axis=1)[near], minlength=RING_MODES)
    return {"on_mode": float(near.mean()), "modes": int((counts >= 0.02 * len(p)).sum()), "counts": counts.tolist()}


# ---------------------------------------------------------------------------------------------------------------------
# The classifier of the evaluations (features of the Fréchet distance, recognition of noisy digits)
# ---------------------------------------------------------------------------------------------------------------------
CLASSIFIER_SETTINGS = {"n": 10_000, "epochs": 3, "batch_size": 64, "lr": 1e-3, "seed": 2502}


def _classifier_class():
    torch = _torch()
    nn = torch.nn

    class Classifier(nn.Module):
        """The small CNN of ch. 27 (part B): ``features(x)`` is its second-to-last layer (64 values per image),
        ``forward(x)`` the 10 logits; images in [-1, 1]."""

        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU())
            self.head = nn.Linear(64, 10)

        def features(self, x):
            return self.body(x)

        def forward(self, x):
            return self.head(self.body(x))

    return Classifier


def classifier_dir() -> Path:
    path = datasets.cache_dir() / "diffusion"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical(key: dict) -> str:
    return json.dumps(key, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def _atomic_save(payload: dict, path: Path) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read(path: Path, key: dict) -> dict | None:
    torch = _torch()
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("__complete__") is not True or payload.get("key") != _canonical(key):
            return None
    except Exception:  # noqa: BLE001 - truncated, corrupt, another format: as if absent
        return None
    payload["path"] = path
    return payload


_CLASSIFIERS: dict = {}


def classifier(name: str = "mnist", folder: str | Path | None = None, verbose: bool = True):
    """The classifier of the dataset ``name`` (``"mnist"`` or ``"fashion"``), in eval mode: the CNN of ch. 27 trained
    with ``CLASSIFIER_SETTINGS`` (Adam, 3 epochs) on the first 10 000 images of the frozen subset, once (≈ 20 s on two
    CPU cores), then read from its file (a key with the settings and the fingerprint of the images). Kept in memory
    for the session."""
    torch = _torch()
    F = torch.nn.functional
    data = image_subset(name, CLASSIFIER_SETTINGS["n"])
    key = {"version": CHECKPOINT_VERSION, "kind": "classifier", "settings": CLASSIFIER_SETTINGS, "data": data.describe()}
    path = Path(folder) if folder is not None else classifier_dir()
    path = path / f"classifier-{name}-{hashlib.sha256(_canonical(key).encode()).hexdigest()[:16]}.pt"
    if path in _CLASSIFIERS:
        return _CLASSIFIERS[path]
    Classifier = _classifier_class()
    clf = Classifier()
    payload = _read(path, key)
    if payload is None:
        if verbose:
            print(f"ℹ️ entraînement du classifieur de référence de {name} (une seule fois, ≈ 20 s)", flush=True)
        s = CLASSIFIER_SETTINGS
        with torch.random.fork_rng():
            torch.manual_seed(s["seed"])
            clf = Classifier()
            opt = torch.optim.Adam(clf.parameters(), lr=s["lr"])
            gen = torch.Generator().manual_seed(s["seed"])
            x, y = data.images, data.labels
            for _ in range(s["epochs"]):
                perm = torch.randperm(len(x), generator=gen)
                for i in range(0, len(x), s["batch_size"]):
                    idx = perm[i:i + s["batch_size"]]
                    loss = F.cross_entropy(clf(x[idx]), y[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
        _atomic_save({"key": _canonical(key), "state_dict": state, "__complete__": True}, path)
    else:
        clf.load_state_dict(payload["state_dict"])
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    _CLASSIFIERS[path] = clf
    return clf


def _as_images(images):
    torch = _torch()
    x = images.detach().cpu() if isinstance(images, torch.Tensor) else torch.as_tensor(np.asarray(images))
    return x.float().reshape(-1, 1, 28, 28)


def features(clf, images, batch_size: int = 500) -> np.ndarray:
    """The features (float64, (N, 64)) of images in [-1, 1] (a tensor or an array (N, 1, 28, 28) or (N, 784)), in
    eval mode, without a graph, by mini-batches."""
    torch = _torch()
    x = _as_images(images)
    with torch.no_grad():
        out = [clf.features(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
    return torch.cat(out).double().numpy() if out else np.zeros((0, 64))


def predict(clf, images, batch_size: int = 500) -> np.ndarray:
    """The class predicted for each image in [-1, 1] (int64 array)."""
    torch = _torch()
    x = _as_images(images)
    with torch.no_grad():
        out = [clf(x[i:i + batch_size]).argmax(dim=1) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------------
# The reference U-Net of Ex B5.21
# ---------------------------------------------------------------------------------------------------------------------
def sinusoidal_table(n: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """The table (n, dim), float64, of ``positional_encoding`` of B3: column 2i is ``sin(p / base**(2i/dim))``,
    column 2i + 1 its cosine (the reference U-Net computes it here, so as not to depend on your library)."""
    if dim % 2 or n < 1 or dim < 2:
        raise ValueError(f"expected n >= 1 and an even dim >= 2, got n = {n}, dim = {dim}")
    pos = np.arange(n, dtype=np.float64)[:, None]
    angles = pos / base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    table = np.empty((n, dim))
    table[:, 0::2], table[:, 1::2] = np.sin(angles), np.cos(angles)
    return table


def _unet_classes():
    torch = _torch()
    nn = torch.nn
    F = nn.functional

    class ResBlock(nn.Module):
        """The residual block of the fiche (§205.4): group norm, SiLU, 3x3 convolution, plus one bias per channel
        from the step embedding, then group norm, SiLU, 3x3 convolution; a 1x1 convolution on the residual path when
        the number of channels changes."""

        def __init__(self, c_in: int, c_out: int, t_dim: int):
            super().__init__()
            self.norm1 = nn.GroupNorm(8, c_in)
            self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
            self.time = nn.Linear(t_dim, c_out)
            self.norm2 = nn.GroupNorm(8, c_out)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
            self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x, temb):
            h = self.conv1(F.silu(self.norm1(x)))
            h = h + self.time(temb)[:, :, None, None]
            h = self.conv2(F.silu(self.norm2(h)))
            return h + self.skip(x)

    class TinyUNet(nn.Module):
        """The reference U-Net of Ex B5.21 (the one of the fiche, §205.4), with an optional class embedding.

        ``TinyUNet(n_steps, channels=(32, 64, 128), t_dim=128, n_classes=None)``; ``forward(x, t, y=None)``: images
        (N, 1, 28, 28), step indices (N,) int64 in [0, n_steps), class labels (N,) int64 in [0, n_classes] where
        ``n_classes`` is the empty class ∅ (also used when ``y`` is None) -> the predicted noise (N, 1, 28, 28)."""

        def __init__(self, n_steps: int, channels=(32, 64, 128), t_dim: int = 128, n_classes: int | None = None):
            super().__init__()
            c1, c2, c3 = (int(c) for c in channels)
            self.config = {"n_steps": int(n_steps), "channels": [c1, c2, c3], "t_dim": int(t_dim),
                           "n_classes": None if n_classes is None else int(n_classes)}
            self.n_steps, self.n_classes = int(n_steps), self.config["n_classes"]
            self.register_buffer("pe", torch.tensor(sinusoidal_table(n_steps, t_dim), dtype=torch.float32),
                                 persistent=False)
            self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
            if n_classes is not None:
                self.class_emb = nn.Embedding(n_classes + 1, t_dim)
            self.inp = nn.Conv2d(1, c1, 3, padding=1)
            self.d1 = ResBlock(c1, c1, t_dim)
            self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
            self.d2 = ResBlock(c1, c2, t_dim)
            self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
            self.mid = ResBlock(c2, c3, t_dim)
            self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
            self.u2 = ResBlock(2 * c2, c2, t_dim)
            self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
            self.u1 = ResBlock(2 * c1, c1, t_dim)
            self.out = nn.Conv2d(c1, 1, 3, padding=1)

        def embed(self, t, y=None):
            temb = self.t_mlp(self.pe[t])
            if self.n_classes is not None:
                if y is None:
                    y = torch.full_like(t, self.n_classes)
                temb = temb + self.class_emb(y)
            return temb

        def forward(self, x, t, y=None):
            temb = self.embed(t, y)
            h1 = self.d1(self.inp(x), temb)                         # (N, c1, 28, 28), kept for the skip
            h2 = self.d2(self.down1(h1), temb)                      # (N, c2, 14, 14), kept for the skip
            h = self.mid(self.down2(h2), temb)                      # (N, c3, 7, 7)
            h = self.u2(torch.cat([self.up2(h), h2], dim=1), temb)  # (N, 2 c2, 14, 14) -> (N, c2, 14, 14)
            h = self.u1(torch.cat([self.up1(h), h1], dim=1), temb)  # (N, 2 c1, 28, 28) -> (N, c1, 28, 28)
            return self.out(h)

    return ResBlock, TinyUNet


TOY_SETTINGS = {"n_iters": 3000, "batch_size": 512, "lr": 2e-3}
FM_POSITIONS = 1001           # flow matching: s = u / 1000 for the integer positions u = 1 .. 1000 (Ex B5.20)


def _toy_class():
    torch = _torch()
    nn = torch.nn

    class Denoiser2D(nn.Module):
        """The reference MLP of Ex B5.19 and B5.20: the sinusoidal encoding of the integer position ``t`` (a row of
        ``sinusoidal_table(n_positions, t_dim)``, a non-persistent buffer ``pe``) concatenated to the two coordinates,
        then ``net``: three hidden layers of ``hidden`` SiLU units and a dense output of 2 values."""

        def __init__(self, n_positions: int, t_dim: int = 32, hidden: int = 128):
            super().__init__()
            self.register_buffer("pe", torch.tensor(sinusoidal_table(n_positions, t_dim), dtype=torch.float32),
                                 persistent=False)
            self.net = nn.Sequential(nn.Linear(2 + t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

        def forward(self, x, t):
            return self.net(torch.cat([x, self.pe[t]], dim=1))

    return Denoiser2D


def train_toy(model, data, n_iters: int, batch_size: int, lr: float, generator, alpha_bars=None, *,
              flow: bool = False) -> list:
    """The reference training of Ex B5.19 (``flow=False``: predict the noise; ``alpha_bars`` of the schedule) and of
    Ex B5.20 (``flow=True``: predict the velocity ``eps - x0`` at ``s = u / 1000``). AdamW(lr) with its defaults; at
    each iteration, in this order: the indices of the batch ``torch.randint(len(data), (batch_size,))``, the step
    (``torch.randint(0, T, ...)``, or the position ``torch.randint(1, 1001, ...)``), the noise ``torch.randn``, all
    from ``generator``; then the MSE, its gradients and one step. Returns the losses (Python floats)."""
    torch = _torch()
    F = torch.nn.functional
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ab = None if flow else torch.as_tensor(np.asarray(alpha_bars, dtype=np.float64), dtype=torch.float32)
    losses = []
    model.train()
    for _ in range(n_iters):
        x0 = data[torch.randint(len(data), (batch_size,), generator=generator)]
        if flow:
            u = torch.randint(1, FM_POSITIONS, (batch_size,), generator=generator)
            eps = torch.randn(x0.shape, generator=generator)
            s = (u.float() / (FM_POSITIONS - 1))[:, None]
            loss = F.mse_loss(model((1 - s) * x0 + s * eps, u), eps - x0)
        else:
            t = torch.randint(0, len(ab), (batch_size,), generator=generator)
            eps = torch.randn(x0.shape, generator=generator)
            a = ab[t][:, None]
            loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _eps(model, x: np.ndarray, t: int) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        out = model(torch.as_tensor(x, dtype=torch.float32), torch.full((len(x),), int(t), dtype=torch.long))
    return out.double().numpy()


def sample_toy_ddpm(model, n: int, alpha_bars, generator, rng) -> np.ndarray:
    """Ancestral DDPM sampling (posterior variance, no clipping, no noise at index 0) of the reference, in float64,
    from ``torch.randn((n, 2), generator=generator)``; ``z`` drawn by ``rng.standard_normal``."""
    torch = _torch()
    ab = np.asarray(alpha_bars, dtype=np.float64)
    ab_prev = np.concatenate([[1.0], ab[:-1]])
    alphas = ab / ab_prev
    x = torch.randn((n, 2), generator=generator).double().numpy()
    for t in range(len(ab) - 1, -1, -1):
        eps = _eps(model, x, t)
        mean = (x - (1 - alphas[t]) / math.sqrt(1 - ab[t]) * eps) / math.sqrt(alphas[t])
        if t > 0:
            var = (1 - ab_prev[t]) / (1 - ab[t]) * (1 - alphas[t])
            mean = mean + math.sqrt(var) * rng.standard_normal(x.shape)
        x = mean
    return x


def sample_toy_ddim(model, n: int, n_steps: int, alpha_bars, generator) -> np.ndarray:
    """Deterministic DDIM (eta = 0) of the reference on ``ddim_timesteps(T, n_steps)``, then to index -1."""
    torch = _torch()
    ab = np.asarray(alpha_bars, dtype=np.float64)
    steps = ddim_timesteps(len(ab), n_steps)
    x = torch.randn((n, 2), generator=generator).double().numpy()
    for i, t in enumerate(steps):
        a_prev = ab[steps[i + 1]] if i + 1 < len(steps) else 1.0
        eps = _eps(model, x, t)
        x0 = (x - math.sqrt(1 - ab[t]) * eps) / math.sqrt(ab[t])
        x = math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps
    return x


def sample_toy_euler(model, n: int, n_steps: int, generator) -> np.ndarray:
    """Euler integration of the reference flow from s = 1 (noise) to s = 0 in ``n_steps`` equal steps: at step k,
    the position ``round(1000 (1 - k / n_steps))`` and ``x <- x - v / n_steps``."""
    torch = _torch()
    x = torch.randn((n, 2), generator=generator)
    with torch.no_grad():
        for k in range(n_steps):
            u = round((FM_POSITIONS - 1) * (1 - k / n_steps))
            x = x - model(x, torch.full((n,), u, dtype=torch.long)) / n_steps
    return x.double().numpy()


def __getattr__(name: str):
    """``ResBlock``, ``TinyUNet`` and ``Denoiser2D`` are built at first use (importing this module does not import PyTorch)."""
    if name in ("ResBlock", "TinyUNet"):
        classes = dict(zip(("ResBlock", "TinyUNet"), _unet_classes()))
        globals().update(classes)
        return classes[name]
    if name == "Denoiser2D":
        globals()["Denoiser2D"] = _toy_class()
        return globals()["Denoiser2D"]
    raise AttributeError(f"module 'wb.diffusion' has no attribute {name!r}")


def count_params(model) -> int:
    """The number of parameters (buffers excluded) of a module."""
    return sum(p.numel() for p in model.parameters())


UNET_STAGES = ("inp", "d1", "down1", "d2", "down2", "mid", "up2", "u2", "up1", "u1", "out")


def stage_shapes(model, x, t, y=None) -> dict:
    """The output shape of each stage of a U-Net with the module names of Ex B5.21 (``UNET_STAGES``), read by forward
    hooks during one call without a graph, plus the inputs of ``u2`` and ``u1`` (after the concatenation of the skip):
    ``{"inp": (N, 32, 28, 28), ..., "u2.in": (N, 128, 14, 14), ...}``. The hooks are always removed."""
    torch = _torch()
    shapes, handles = {}, []
    try:
        for name in UNET_STAGES:
            module = getattr(model, name, None)
            if module is None:
                continue
            handles.append(module.register_forward_hook(
                lambda mod, inp, out, name=name: shapes.__setitem__(name, tuple(out.shape))))
            if name in ("u2", "u1"):
                handles.append(module.register_forward_pre_hook(
                    lambda mod, inp, name=name: shapes.__setitem__(f"{name}.in", tuple(inp[0].shape))))
        with torch.no_grad():
            model(x, t) if y is None else model(x, t, y)
    finally:
        for handle in handles:
            handle.remove()
    return shapes


# ---------------------------------------------------------------------------------------------------------------------
# The single checkpoint of a trained diffusion model (outside the repository, atomic, with its provenance)
# ---------------------------------------------------------------------------------------------------------------------
def checkpoints_dir() -> Path:
    """Where the trained diffusion models go: ``wb.datasets.cache_dir() / "diffusion"`` (never versioned; on Colab,
    ``/content/wb_cache/diffusion``, erased with the session)."""
    return classifier_dir()


def checkpoint_key(config: dict, settings: dict, *, fast: bool, source: str, data: ImageSet | dict) -> dict:
    """The complete key of a trained model: the configuration of the network, the training settings (seed and
    schedule included), the mode, who trained it (``"learner"``, ``"solution"`` or ``"reference"``) and the
    fingerprint of the training images. Two models whose keys differ never share a file."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    described = data.describe() if isinstance(data, ImageSet) else dict(data)
    return {"version": CHECKPOINT_VERSION, "source": source, "fast": bool(fast), "config": dict(config),
            "settings": dict(settings), "data": described}


def checkpoint_file(key: dict, folder: str | Path | None = None) -> Path:
    """The ``.pt`` file of a key: its source, its mode and 16 hexadecimal digits of the SHA-256 of the key."""
    digest = hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()[:16]
    mode = "fast" if key.get("fast") else "full"
    base = Path(folder) if folder is not None else checkpoints_dir()
    return base / f"ddpm-{key.get('source', 'model')}-{mode}-{digest}.pt"


def save_checkpoint(key: dict, model, ema=None, history: dict | None = None, folder: str | Path | None = None,
                    extra: dict | None = None) -> Path:
    """Save the ``state_dict`` of ``model`` (and of its EMA copy ``ema``, a module or a state dict), the configuration
    of the key, the history of the training and ``extra`` in the file of ``key``: written to a temporary file, then
    renamed (``os.replace``), so that an interrupted save never leaves a half-written checkpoint. Returns the path."""
    torch = _torch()
    path = checkpoint_file(key, folder)

    def state(obj):
        items = obj.state_dict().items() if hasattr(obj, "state_dict") else dict(obj).items()
        return {name: t.detach().cpu().clone() for name, t in items}

    payload = {"key": _canonical(key), "config": dict(key["config"]), "history": history or {},
               "state_dict": state(model), "ema_state_dict": state(ema) if ema is not None else None,
               "extra": dict(extra or {}), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "torch": str(torch.__version__), "__complete__": True}
    _atomic_save(payload, path)
    return path


def load_checkpoint(key: dict, folder: str | Path | None = None) -> dict | None:
    """The checkpoint of ``key`` (``state_dict``, ``ema_state_dict``, ``config``, ``history``, ``extra``, ``saved_at``,
    ``path``), or None when there is none, when the file holds another key, or when it is incomplete or unreadable."""
    return _read(checkpoint_file(key, folder), key)


def delete_checkpoint(key: dict, folder: str | Path | None = None) -> bool:
    """Remove the checkpoint of ``key`` if there is one (a new training removes the old model first, so that a failed
    run never leaves the previous one in place); True when a file was removed."""
    path = checkpoint_file(key, folder)
    if path.exists():
        path.unlink()
        return True
    return False


def build_from_checkpoint(model_class, ckpt: dict, which: str = "ema"):
    """``model_class(**config)`` with the weights of ``ckpt`` loaded strictly, in eval mode: the EMA copy
    (``which="ema"``, the one that generates) or the raw weights (``which="raw"``)."""
    if which not in ("ema", "raw"):
        raise ValueError(f"which must be 'ema' or 'raw', got {which!r}")
    state = ckpt["ema_state_dict"] if which == "ema" else ckpt["state_dict"]
    if state is None:
        raise ValueError("this checkpoint has no EMA copy: use which='raw'")
    model = model_class(**ckpt["config"])
    model.load_state_dict(state)
    return model.eval()


def trim_memory() -> bool:
    """Collect the garbage and give the freed heap back to the system (``wb.llm.trim_memory``)."""
    from wb.llm import trim_memory as _trim

    return _trim()
