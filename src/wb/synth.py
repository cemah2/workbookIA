"""Synthetic data generators for controlled experiments (BIBLE §9).

Every generator takes a ``seed`` and only needs NumPy. Classification
generators return ``(X, y)`` with ``X`` of shape ``(n, n_features)`` (float64)
and ``y`` of shape ``(n,)`` (int64). Signatures are stable: chapters rely on them.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_moons",
    "make_circles",
    "make_blobs",
    "make_spirals",
    "make_xor",
    "logic_gate",
    "make_linear",
    "make_polynomial",
    "noisy_sine",
    "rosenbrock",
    "rosenbrock_grad",
    "coin_flips",
    "gaussian_1d",
    "loss_curves",
    "optimizer_curves",
    "training_curves",
]


def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(seed)


def _shuffle(X, y, rng):
    order = rng.permutation(len(y))
    return X[order], y[order]


def make_moons(n: int = 200, noise: float = 0.1, seed: int | None = 0):
    """Two interleaving half circles (2 classes, 2 features)."""
    rng = _rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = rng.uniform(0, np.pi, n_out)
    t_in = rng.uniform(0, np.pi, n_in)
    outer = np.column_stack([np.cos(t_out), np.sin(t_out)])
    inner = np.column_stack([1 - np.cos(t_in), 0.5 - np.sin(t_in)])
    X = np.vstack([outer, inner]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(n_out, dtype=np.int64), np.ones(n_in, dtype=np.int64)])
    return _shuffle(X, y, rng)


def make_circles(n: int = 200, noise: float = 0.05, factor: float = 0.5, seed: int | None = 0):
    """A small circle inside a big one (2 classes). ``factor`` = inner/outer radius."""
    rng = _rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = rng.uniform(0, 2 * np.pi, n_out)
    t_in = rng.uniform(0, 2 * np.pi, n_in)
    outer = np.column_stack([np.cos(t_out), np.sin(t_out)])
    inner = factor * np.column_stack([np.cos(t_in), np.sin(t_in)])
    X = np.vstack([outer, inner]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(n_out, dtype=np.int64), np.ones(n_in, dtype=np.int64)])
    return _shuffle(X, y, rng)


def make_blobs(
    n: int = 300,
    centers=3,
    std: float = 1.0,
    n_features: int = 2,
    box: tuple[float, float] = (-8.0, 8.0),
    seed: int | None = 0,
):
    """Gaussian clusters. ``centers`` is a number of clusters or an array of centres."""
    rng = _rng(seed)
    if np.isscalar(centers):
        centers = rng.uniform(box[0], box[1], (int(centers), n_features))
    centers = np.asarray(centers, dtype=float)
    k = len(centers)
    counts = np.full(k, n // k)
    counts[: n % k] += 1
    stds = np.broadcast_to(np.asarray(std, dtype=float), (k,))
    X = np.vstack([rng.normal(c, s, (m, centers.shape[1])) for c, s, m in zip(centers, stds, counts)])
    y = np.repeat(np.arange(k, dtype=np.int64), counts)
    return _shuffle(X, y, rng)


def make_spirals(n: int = 300, n_classes: int = 2, noise: float = 0.2, turns: float = 1.5,
                 seed: int | None = 0):
    """Intertwined spirals, one arm per class (hard for linear models)."""
    rng = _rng(seed)
    counts = np.full(n_classes, n // n_classes)
    counts[: n % n_classes] += 1
    parts, labels = [], []
    for k, m in enumerate(counts):
        r = np.linspace(0.05, 1.0, m)
        theta = r * turns * 2 * np.pi + 2 * np.pi * k / n_classes
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        parts.append(pts + rng.normal(0, noise * 0.1, pts.shape))
        labels.append(np.full(m, k, dtype=np.int64))
    return _shuffle(np.vstack(parts), np.concatenate(labels), rng)


def make_xor(n: int = 200, noise: float = 0.15, seed: int | None = 0):
    """Noisy points around the 4 corners of the unit square; label = x1 XOR x2."""
    rng = _rng(seed)
    corners = rng.integers(0, 2, (n, 2))
    X = corners + rng.normal(0, noise, (n, 2))
    y = (corners[:, 0] ^ corners[:, 1]).astype(np.int64)
    return X, y


_GATES = {
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "nand": lambda a, b: 1 - (a & b),
    "nor": lambda a, b: 1 - (a | b),
}


def logic_gate(name: str = "and"):
    """Truth table of a 2-input logic gate: ``X`` (4, 2) and ``y`` (4,)."""
    key = name.lower()
    if key not in _GATES:
        raise ValueError(f"unknown gate {name!r}; choose from {sorted(_GATES)}")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    y = np.array([_GATES[key](a, b) for a, b in X], dtype=np.int64)
    return X.astype(float), y


def make_linear(n: int = 100, w=(2.0,), b: float = 1.0, noise: float = 0.5,
                x_range: tuple[float, float] = (-3.0, 3.0), seed: int | None = 0):
    """Linear regression data ``y = X @ w + b + noise``. Returns ``(X, y)``."""
    rng = _rng(seed)
    w = np.atleast_1d(np.asarray(w, dtype=float))
    X = rng.uniform(x_range[0], x_range[1], (n, len(w)))
    y = X @ w + b + rng.normal(0, noise, n)
    return X, y


def make_polynomial(n: int = 30, coefs=(0.5, -1.0, 0.0, 2.0), noise: float = 0.2,
                    x_range: tuple[float, float] = (-1.0, 1.0), seed: int | None = 0):
    """Noisy samples of a polynomial ``sum(coefs[k] * x**k)``.

    Returns ``(x, y, f)`` where ``f`` is the true (noise-free) function,
    useful to show overfitting (ch. 9).
    """
    rng = _rng(seed)
    coefs = np.asarray(coefs, dtype=float)

    def f(x):
        x = np.asarray(x, dtype=float)
        return sum(c * x**k for k, c in enumerate(coefs))

    x = np.sort(rng.uniform(x_range[0], x_range[1], n))
    y = f(x) + rng.normal(0, noise, n)
    return x, y, f


def noisy_sine(n: int = 500, freq: float = 0.5, noise: float = 0.1, t_max: float = 10.0,
               trend: float = 0.0, seed: int | None = 0):
    """A regularly sampled noisy sine wave ``sin(2π·freq·t) + trend·t + noise``.

    ``freq`` is in cycles per time unit. Returns ``(t, y)``.
    """
    rng = _rng(seed)
    t = np.linspace(0, t_max, n)
    y = np.sin(2 * np.pi * freq * t) + trend * t + rng.normal(0, noise, n)
    return t, y


def rosenbrock(x, y, a: float = 1.0, b: float = 100.0):
    """Rosenbrock function ``(a - x)^2 + b (y - x^2)^2``; minimum 0 at ``(a, a^2)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (a - x) ** 2 + b * (y - x**2) ** 2


def rosenbrock_grad(x, y, a: float = 1.0, b: float = 100.0):
    """Gradient of :func:`rosenbrock`: returns ``(df/dx, df/dy)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dy = 2 * b * (y - x**2)
    return dx, dy


def coin_flips(n: int = 100, p: float = 0.5, seed: int | None = 0):
    """``n`` coin flips with probability ``p`` of heads (1). Returns an int array."""
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    return (_rng(seed).random(n) < p).astype(np.int64)


def gaussian_1d(n: int = 1000, mu: float = 4.0, sigma: float = 1.25, seed: int | None = 0):
    """Samples from a 1D normal distribution (target of the toy GAN, ch. 27)."""
    return _rng(seed).normal(mu, sigma, n)


def loss_curves(seed: int | None = 1822) -> dict:
    """Six synthetic training-loss curves (mean cross-entropy in nats, 60 epochs), labelled "A" to "F".

    Material of a reading exercise (chapter 18): each curve has the shape of a typical training problem. The
    labels are deliberately neutral; the solutions of the exercise say which curve is which.
    """
    rng = _rng(seed)
    t = np.arange(1, 61, dtype=float)
    start = np.log(10.0)
    noise = lambda: 1 + 0.01 * rng.normal(size=60)
    a = np.where(t < 6, start - (start - 1.4) * t / 6, 1.4)
    a = np.where(t > 32, 1.4 - 1.25 * (1 - np.exp(-(t - 32) / 4)), a) * noise()
    e = start + 0.12 * 1.4 ** t * (1 + np.sin(2.4 * t))
    e[t > 11] = np.nan
    return {"A": a,
            "B": (0.08 + (start - 0.08) * np.exp(-t / 6)) * noise(),
            "C": (start + 0.05 * t + 0.0004 * t ** 2) * noise(),
            "D": (start - 0.007 * t) * noise(),
            "E": e,
            "F": np.full(60, start)}


def optimizer_curves(seed: int | None = 1923, n_updates: int = 1200) -> dict:
    """Six synthetic mini-batch loss curves (cross-entropy in nats, one value per update), labelled "A" to "F".

    Material of a reading exercise (chapter 19): each curve has the shape of a typical run of a 10-class
    classifier with a given optimizer or learning-rate schedule (all start near log 10). The labels are
    deliberately neutral; the solutions of the exercise say which curve is which.
    """
    rng = _rng(seed)
    t = np.arange(n_updates, dtype=float)
    start = np.log(10.0)

    def noisy(mean, rel):
        return np.maximum(mean * (1 + rel * rng.normal(size=n_updates)), 1e-3)

    # curve A
    k1, k2 = n_updates // 3, 2 * n_updates // 3
    floor_a = np.where(t < k1, 0.40, np.where(t < k2, 0.16, 0.07))
    mean_a = floor_a + (start - 0.40) * np.exp(-t / 70)
    mean_a = np.where(t >= k1, floor_a + (0.40 - 0.16) * np.exp(-(t - k1) / 8) * (t < k2), mean_a)
    mean_a = np.where(t >= k2, 0.07 + (0.16 - 0.07) * np.exp(-(t - k2) / 8), mean_a)
    rel_a = np.where(t < k1, 0.30, np.where(t < k2, 0.12, 0.05))
    a = np.maximum(mean_a * (1 + rel_a * rng.normal(size=n_updates)), 1e-3)
    # curve B
    b = np.full(n_updates, start) * (1 + 0.0004 * rng.normal(size=n_updates))
    b[:30] = start - 1.1 * np.sin(np.pi * t[:30] / 60) * (1 + 0.05 * rng.normal(size=30))
    b[30:70] = start + np.abs(6 * rng.standard_cauchy(40)).clip(0, 25)
    # curve C
    c = noisy(0.05 + (start - 0.05) * np.exp(-t / 110), 0.12)
    # curve D
    warm = int(0.12 * n_updates)
    lr = np.where(t < warm, t / warm, 0.5 * (1 + np.cos(np.pi * (t - warm) / (n_updates - warm))))
    progress = np.cumsum(lr) / 110
    d = np.maximum((0.03 + (start - 0.03) * np.exp(-progress)) * (1 + 0.25 * lr * rng.normal(size=n_updates)), 1e-3)
    # curve E
    e = (0.15 + (start - 0.15) * np.exp(-t / 120)) * rng.exponential(1.0, size=n_updates)
    # curve F
    f = noisy(0.38 + 0.25 / np.sqrt(1 + t / 60) + (start - 0.63) * np.exp(-t / 25), 0.25 / np.sqrt(1 + t / 100))
    return {"A": a, "B": b, "C": c, "D": d, "E": e, "F": f}


def training_curves(seed: int | None = 2025, n_epochs: int = 30) -> dict:
    """Five synthetic training histories of a Fashion-MNIST classifier, labelled "A" to "E".

    Each value is a dict of three arrays of shape ``(n_epochs,)``, one value per epoch (epoch 1 first):
    ``"train_loss"`` (mean cross-entropy in nats over the epoch, measured during training, so with dropout
    when the model has some), ``"val_loss"`` and ``"val_acc"`` (measured on a validation set after the
    epoch). Material of a reading exercise (chapter 20): each history has the shape of a typical situation.
    The labels are deliberately neutral; the solutions of the exercise say which history is which.
    """
    rng = _rng(seed)
    t = np.arange(1, n_epochs + 1, dtype=float)

    def jitter(curve, rel=0.01):
        return curve * (1 + rel * rng.normal(size=n_epochs))

    # A
    a_train = jitter(0.32 + 0.40 * np.exp(-(t - 1) / 5))
    a_val = (0.31 + 0.36 * np.exp(-(t - 1) / 5)) * (1 + 0.09 * rng.normal(size=n_epochs))
    a_acc = 0.845 + 0.03 * (1 - np.exp(-(t - 1) / 4)) + 0.018 * rng.normal(size=n_epochs)
    # B
    b_train = jitter(0.04 + 0.62 * np.exp(-(t - 1) / 6))
    b_val = jitter(0.30 + 0.35 * np.exp(-(t - 1) / 2.2) + 0.035 * np.maximum(0.0, t - 7), 0.003)
    b_acc = 0.80 + 0.095 * (1 - np.exp(-(t - 1) / 3)) - 0.0009 * np.maximum(0.0, t - 9) ** 1.3 + 0.002 * rng.normal(size=n_epochs)
    # C
    c_train = jitter(0.82 + 0.30 * np.exp(-(t - 1) / 2))
    c_val = jitter(0.83 + 0.29 * np.exp(-(t - 1) / 2))
    c_acc = 0.715 + 0.06 * (1 - np.exp(-(t - 1) / 2)) + 0.003 * rng.normal(size=n_epochs)
    # D
    d_train = jitter(0.33 + 0.40 * np.exp(-(t - 1) / 5))
    d_val = jitter(0.27 + 0.36 * np.exp(-(t - 1) / 4), 0.006)
    d_acc = 0.835 + 0.065 * (1 - np.exp(-(t - 1) / 4)) + 0.002 * rng.normal(size=n_epochs)
    # E
    e_train = jitter(0.02 + 0.62 * np.exp(-(t - 1) / 5))
    e_val = e_train * (0.98 + 0.01 * rng.normal(size=n_epochs))
    e_acc = np.minimum(0.999, 0.80 + 0.195 * (1 - np.exp(-(t - 1) / 5)) + 0.001 * rng.normal(size=n_epochs))
    out = {}
    for key, (tr, va, acc) in {"A": (a_train, a_val, a_acc), "B": (b_train, b_val, b_acc), "C": (c_train, c_val, c_acc),
                               "D": (d_train, d_val, d_acc), "E": (e_train, e_val, e_acc)}.items():
        out[key] = {"train_loss": np.maximum(tr, 1e-3), "val_loss": np.maximum(va, 1e-3),
                    "val_acc": np.clip(acc, 0.0, 1.0)}
    return out
