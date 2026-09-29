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
