"""Tests of the data: the preprocessing on a constant image, the schema of the data file, no image shared by the
training and the validation parts, the fingerprint."""

import numpy as np
import pytest
import torch

from cifarprod.data import channel_stats, fingerprint, load_npz, preprocess

MEAN, STD = (0.5, 0.4, 0.3), (0.25, 0.2, 0.1)


@pytest.mark.parametrize("value", [0, 255])
def test_constant_image_preprocessed(value):
    x = preprocess(np.full((32, 32, 3), value, dtype=np.uint8), MEAN, STD)
    want = torch.tensor([(value / 255 - m) / s for m, s in zip(MEAN, STD)]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    assert x.shape == (1, 3, 32, 32), f"expected the shape (1, 3, 32, 32), got {tuple(x.shape)}"
    assert x.dtype == torch.float32, f"expected float32, got {x.dtype}"
    assert torch.allclose(x, want, atol=1e-6), f"expected {want[0, :, 0, 0].tolist()} per channel, got {x[0, :, 0, 0].tolist()}"


def test_channels_stay_in_rgb_order():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 255                                   # a red image
    x = preprocess(image, (0, 0, 0), (1, 1, 1))
    assert x[0, 0].min() == 1 and x[0, 1:].max() == 0, f"expected red in channel 0 only, got the means {x[0].mean(dim=(1, 2)).tolist()}"


@pytest.mark.parametrize("part, n", [("train", 2000), ("val", 500)])
def test_data_schema(data_file, part, n):
    X, y = load_npz(data_file, part)
    assert X.dtype == np.uint8, f"expected uint8 images, got {X.dtype}"
    assert X.shape == (n, 32, 32, 3), f"expected the shape ({n}, 32, 32, 3), got {X.shape}"
    assert y.shape == (n,) and np.issubdtype(y.dtype, np.integer), f"expected {n} integer labels, got {y.shape} {y.dtype}"
    assert set(np.unique(y)) == set(range(10)), f"expected the labels 0 to 9, got {sorted(set(np.unique(y)))}"


def test_no_image_in_both_parts(data_file):
    train = {fingerprint(image) for image in load_npz(data_file, "train")[0]}
    shared = sum(fingerprint(image) in train for image in load_npz(data_file, "val")[0])
    assert shared == 0, f"expected no validation image in the training part, got {shared}"


@pytest.mark.parametrize("bad", ["float", "label", "shape"])
def test_load_npz_refuses_a_bad_schema(tmp_path, bad):
    X, y = np.zeros((4, 32, 32, 3), np.uint8), np.arange(4)
    X = X.astype(np.float32) if bad == "float" else X[:, :28] if bad == "shape" else X
    y = y + 7 if bad == "label" else y
    np.savez(tmp_path / "bad.npz", x_train=X, y_train=y, x_val=X, y_val=y)
    with pytest.raises(ValueError):
        load_npz(tmp_path / "bad.npz", "train")


def test_fingerprint_sees_one_pixel():
    X = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    Y = X.copy()
    Y[1, 5, 5, 2] = 1
    assert fingerprint(X) != fingerprint(Y), "expected another fingerprint when one pixel changes"
    assert fingerprint(X) != fingerprint(X.astype(np.int64)), "expected another fingerprint for another dtype"


def test_channel_stats_on_training_images():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (10, 32, 32, 3), dtype=np.uint8)
    mean, std = channel_stats(X)
    want = (X / 255).mean(axis=(0, 1, 2)), (X / 255).std(axis=(0, 1, 2))
    assert np.allclose(mean, want[0]) and np.allclose(std, want[1]), f"expected {want}, got {(mean, std)}"
