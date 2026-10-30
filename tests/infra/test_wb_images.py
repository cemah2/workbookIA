"""Images and pre-trained networks of chapter 28 (wb.datasets)."""

import hashlib

import numpy as np
import pytest

from wb import datasets


@pytest.mark.parametrize("name", sorted(datasets.CONTENT_IMAGES))
def test_load_sample_image_format(name):
    img = datasets.load_sample_image(name, size=96)
    assert img.dtype == np.float32 and img.ndim == 3 and img.shape[0] == 3
    assert max(img.shape[1:]) == 96, f"expected a longer side of 96, got {img.shape}"
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_load_sample_image_keeps_proportions_and_full_size():
    full = datasets.load_sample_image("china")
    assert full.shape == (3, 427, 640), f"expected (3, 427, 640), got {full.shape}"
    small = datasets.load_sample_image("china", size=128)
    assert small.shape == (3, 85, 128), f"expected (3, 85, 128), got {small.shape}"


def test_unknown_images_are_refused():
    with pytest.raises(ValueError):
        datasets.load_sample_image("picasso")
    with pytest.raises(ValueError):
        datasets.load_style_image("nighthawks")
    with pytest.raises(ValueError):
        datasets.load_sample_image("china", size=2)


@pytest.mark.parametrize("name", sorted(datasets.STYLE_IMAGES))
def test_versioned_paintings_match_their_checksum(name):
    path = datasets.data_dir() / "images" / f"{name}.jpg"
    assert path.exists(), f"expected the versioned copy {path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == datasets.STYLE_IMAGES[name]["sha256"]
    img = datasets.load_style_image(name, size=64, allow_download=False)
    assert img.dtype == np.float32 and img.shape[0] == 3 and max(img.shape[1:]) == 64


def test_style_image_falls_back_to_a_procedural_texture(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "data_dir", lambda: tmp_path)
    monkeypatch.setenv("WB_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.warns(UserWarning, match="SUBSTITUT"):
        img = datasets.load_style_image("starry_night", size=48, allow_download=False)
    assert img.shape == (3, 48, 48) and img.dtype == np.float32
    assert np.array_equal(img, datasets.procedural_style(48, seed=sorted(datasets.STYLE_IMAGES).index("starry_night")))


def test_procedural_style_is_deterministic_and_textured():
    a, b = datasets.procedural_style(64, seed=3), datasets.procedural_style(64, seed=3)
    assert np.array_equal(a, b)
    assert 0.0 <= a.min() and a.max() <= 1.0 and a.std() > 0.05


@pytest.mark.parametrize("name", sorted(datasets.CONTENT_IMAGES) + sorted(datasets.STYLE_IMAGES))
def test_every_image_has_a_credit(name):
    credit = datasets.image_credit(name)
    assert "http" in credit and ("CC BY" in credit or "domaine public" in credit)


def test_paintings_are_in_the_eu_public_domain():
    for name, info in datasets.STYLE_IMAGES.items():
        assert info["artist_died"] + 70 < 2026, f"{name}: {info['artist']} is still protected in the EU"


@pytest.mark.network
def test_load_pretrained_squeezenet_is_frozen():
    torch = pytest.importorskip("torch")
    net, mean, std = datasets.load_pretrained("squeezenet1_1")
    assert not net.training
    assert all(not p.requires_grad for p in net.parameters())
    assert all(not m.inplace for m in net.modules() if isinstance(m, torch.nn.ReLU))
    assert mean.shape == (1, 3, 1, 1) and std.shape == (1, 3, 1, 1)
    out = net(torch.zeros(1, 3, 128, 128))
    assert tuple(out.shape) == (1, 512, 7, 7), f"expected (1, 512, 7, 7), got {tuple(out.shape)}"


def test_load_pretrained_refuses_unknown_networks():
    with pytest.raises(ValueError):
        datasets.load_pretrained("resnet9000")
