"""Your tests of cifarkit (step MP5.8): at least three, each with a clear name and a message whose first line says
what was expected (the notebook only prints that line).

    python -m pytest mon_travail/projets/partie_5_classifieur_cifar10 -q

Run them on small random images: what they check must not depend on CIFAR-10, and they must take a few seconds. The
oracles are NumPy, PyTorch or a property written by hand, never another function of cifarkit. One example is given;
the notebook runs your tests again against five mutants of cifarkit (MP5.8), and good tests catch at least two.
"""

import numpy as np
import pytest

from cifarkit import channel_stats


@pytest.fixture
def images():
    """60 random uint8 images (60, 32, 32, 3) and their labels: 6 per class."""
    rng = np.random.default_rng(5)
    return rng.integers(0, 256, size=(60, 32, 32, 3), dtype=np.uint8), np.repeat(np.arange(10), 6)


def test_channel_stats_match_numpy(images):
    X, _ = images
    mean, std = channel_stats(X)
    x = X.astype(np.float64) / 255
    want_mean, want_std = [x[..., c].mean() for c in range(3)], [x[..., c].std() for c in range(3)]
    assert np.allclose(mean, want_mean) and np.allclose(std, want_std), \
        f"expected per-channel mean {np.round(want_mean, 4)} and std {np.round(want_std, 4)}, got {mean} and {std}"


# TODO MP5.8: your tests (evaluate, fit, fgsm, stratified_split, the models...)
