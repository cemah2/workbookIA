"""Nested subsets, preprocessing, backbones and the embeddings cache of bonus chapter B1 (wb.datasets, wb.transfer)."""

import numpy as np
import pytest

from wb import datasets, transfer

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Nested, stratified subsets
# ---------------------------------------------------------------------------
def _labels(n_classes=4, per_class=60, seed=0):
    return np.random.default_rng(seed).permutation(np.repeat(np.arange(n_classes), per_class))


def test_stratified_indices_are_stratified_sorted_and_deterministic():
    y = _labels()
    idx = datasets.stratified_indices(y, 7, seed=3)
    assert np.all(np.diff(idx) > 0), "expected sorted indices without repetition"
    assert np.bincount(y[idx]).tolist() == [7, 7, 7, 7], f"expected 7 per class, got {np.bincount(y[idx])}"
    assert np.array_equal(idx, datasets.stratified_indices(y, 7, seed=3)), "expected the same indices for the same seed"
    assert not np.array_equal(idx, datasets.stratified_indices(y, 7, seed=4)), "expected other indices for another seed"


def test_stratified_indices_are_nested():
    y = _labels()
    previous = set()
    for k in (1, 3, 10, 30, 60):
        current = set(datasets.stratified_indices(y, k, seed=11).tolist())
        assert previous <= current, f"expected the {k}-per-class subset to contain the smaller one"
        previous = current


@pytest.mark.parametrize("bad, why", [(0, "zero"), (-2, "negative"), (2.5, "not whole"), (True, "a boolean"),
                                      (61, "more than a class has")], ids=["zero", "negative", "fraction", "bool", "too-many"])
def test_stratified_indices_refuse_bad_sizes(bad, why):
    with pytest.raises(ValueError):
        datasets.stratified_indices(_labels(), bad, seed=0)


def test_subset_parts_train_and_val_never_overlap():
    y = np.repeat(np.arange(3), 600)                       # 600 per class: 100 for train at most, 500 for val
    train = datasets._subset_indices(y, 100, "train", seed=5)
    val = datasets._subset_indices(y, 500, "val", seed=5)
    assert not set(train) & set(val), "expected disjoint train and validation subsets"
    assert set(datasets._subset_indices(y, 20, "val", seed=5)) <= set(val), "expected nested validation subsets"
    with pytest.raises(ValueError):
        datasets._subset_indices(y, 101, "train", seed=5)
    with pytest.raises(ValueError):
        datasets._subset_indices(y, 501, "val", seed=5)
    with pytest.raises(ValueError):
        datasets._subset_indices(y, 10, "dev", seed=5)


@pytest.mark.network
def test_cifar10_subsets_nested_stratified_and_disjoint():
    sizes = datasets.CIFAR10_SUBSET_SIZES
    subsets = [datasets.cifar10_subset_indices(k) for k in sizes]
    for k, small, large in zip(sizes, subsets, subsets[1:]):
        assert set(small) <= set(large), f"expected the {k}-per-class subset inside the next one"
    X, y = datasets.load_cifar10_subset(30)
    assert X.shape == (300, 32, 32, 3) and X.dtype == np.uint8, f"expected uint8 (300, 32, 32, 3), got {X.dtype} {X.shape}"
    assert np.bincount(y).tolist() == [30] * 10, f"expected 30 images per class, got {np.bincount(y)}"
    assert np.array_equal(datasets.cifar10_subset_indices(30), datasets.cifar10_subset_indices(30))
    val = datasets.cifar10_subset_indices(datasets.SUBSET_VAL_POOL, "val")
    assert not set(val) & set(datasets.cifar10_subset_indices(4500)), "expected no validation image in a train subset"
    Xte, yte = datasets.load_cifar10_subset(200, "test")
    assert np.bincount(yte).tolist() == [200] * 10
    assert len(datasets.cifar10_subset_indices(1000, "test")) == 10_000, "expected 1 000 per class = the whole test split"
    Xtr, _ = datasets.load_cifar10_subset(200)
    train_keys = {img.tobytes() for img in Xtr}
    assert not any(img.tobytes() in train_keys for img in Xte), "expected no test image identical to a train image"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_to_input_from_uint8_nhwc_normalises_each_channel():
    images = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
    x = transfer.to_input(images, 16)
    assert tuple(x.shape) == (2, 3, 16, 16) and x.dtype == torch.float32, f"got {x.dtype} {tuple(x.shape)}"
    want = (1 - np.array(datasets.IMAGENET_MEAN)) / np.array(datasets.IMAGENET_STD)
    assert np.allclose(x[0, :, 5, 5].numpy(), want, atol=1e-6), f"expected (1 - mean) / std = {want}"


def test_to_input_repeats_a_gray_channel_after_resizing():
    gray = np.random.default_rng(0).integers(0, 256, (3, 28, 28), dtype=np.uint8)
    x = transfer.to_input(gray, 20, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    assert tuple(x.shape) == (3, 3, 20, 20), f"expected (3, 3, 20, 20), got {tuple(x.shape)}"
    assert torch.equal(x[:, 0], x[:, 1]) and torch.equal(x[:, 1], x[:, 2]), "expected three identical channels"
    rgb = datasets.fashion_mnist_rgb(gray, size=20)
    assert tuple(rgb.shape) == (3, 3, 20, 20)
    back = rgb * torch.tensor(datasets.IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(datasets.IMAGENET_MEAN).view(1, 3, 1, 1)
    assert torch.allclose(back[:, 0], back[:, 2], atol=1e-6), "expected the same gray image in the three channels"


def test_to_input_keeps_the_size_and_accepts_float_tensors():
    x = torch.rand(2, 3, 12, 12)
    out = transfer.to_input(x, None, mean=(0, 0, 0), std=(1, 1, 1))
    assert torch.allclose(out, x), "expected the same tensor with size=None and a neutral normalisation"
    bic = transfer.to_input(x, 30, resample="bicubic", mean=(0, 0, 0), std=(1, 1, 1))
    assert bic.min() >= 0 and bic.max() <= 1, "expected bicubic values clamped to [0, 1]"


@pytest.mark.parametrize("images, why", [
    (np.zeros((2, 8, 8, 4), dtype=np.uint8), "four channels"),
    (np.zeros((8, 8), dtype=np.uint8), "a single image without the batch axis"),
    (np.zeros((2, 2, 8, 8), dtype=np.float32), "two channels"),
], ids=["4-channels", "no-batch-axis", "2-channels"])
def test_to_input_refuses_bad_shapes(images, why):
    with pytest.raises(ValueError):
        transfer.to_input(images, 16)


def test_to_input_refuses_an_unknown_interpolation():
    with pytest.raises(ValueError):
        transfer.to_input(np.zeros((1, 4, 4, 3), dtype=np.uint8), 8, resample="nearest")


# ---------------------------------------------------------------------------
# A fake backbone: embeddings and the cache, offline
# ---------------------------------------------------------------------------
def _fake_backbone():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 5, 3, padding=1), torch.nn.ReLU(),
                              torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()).eval()
    return transfer.Backbone("fake", net, 5, datasets.IMAGENET_MEAN, datasets.IMAGENET_STD, "bilinear", "v0", "test",
                             default_size=16)


def _images(n=7, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, 10, 10, 3), dtype=np.uint8)


def test_backbone_embed_does_not_depend_on_the_batch_size():
    bb = _fake_backbone()
    a, b = bb.embed(_images(), 16, batch_size=2), bb.embed(_images(), 16, batch_size=64)
    assert a.shape == (7, 5) and a.dtype == np.float32, f"got {a.dtype} {a.shape}"
    assert np.allclose(a, b, atol=1e-6), "expected the same embeddings whatever the batch size"
    with torch.inference_mode():
        want = bb.model(bb.preprocess(_images(), 16)).numpy()
    assert np.allclose(a, want, atol=1e-6)
    with pytest.raises(ValueError):
        bb.embed(_images(0), 16)
    with pytest.raises(ValueError):
        bb.embed(_images(), 16, block="layer2")          # the fake backbone has no blocks


def test_cached_arrays_compute_once_then_read(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"a": np.arange(4.0), "b": np.ones((2, 2), dtype=np.float32)}

    key = {"model": "fake", "size": 16, "subset": "s"}
    first = transfer.cached_arrays(key, compute, folder=tmp_path, verbose=False)
    second = transfer.cached_arrays(key, compute, folder=tmp_path, verbose=False)
    assert len(calls) == 1, f"expected one computation, got {len(calls)}"
    assert set(second) == {"a", "b"} and np.array_equal(first["a"], second["a"])
    assert not list(tmp_path.glob("*.tmp")), "expected no temporary file left behind"
    transfer.cached_arrays({**key, "size": 32}, compute, folder=tmp_path, verbose=False)
    assert len(calls) == 2, "expected another key to be computed again"


def test_cached_arrays_recompute_a_file_of_another_key_or_incomplete(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"a": np.arange(3.0)}

    key = {"model": "fake", "subset": "s"}
    path = transfer.cache_file(key, tmp_path)
    np.savez(path, __key__=np.array('{"model":"other"}'), __names__=np.array(["a"]), __complete__=np.array(1),
             a=np.zeros(3))
    out = transfer.cached_arrays(key, compute, folder=tmp_path, verbose=False)
    assert len(calls) == 1 and np.array_equal(out["a"], np.arange(3.0)), "expected a file of another key recomputed"
    np.savez(path, __key__=np.array(transfer._canonical(key)), __names__=np.array(["a"]), __complete__=np.array(0),
             a=np.zeros(3))
    transfer.cached_arrays(key, compute, folder=tmp_path, verbose=False)
    assert len(calls) == 2, "expected an incomplete file recomputed"
    path.write_bytes(path.read_bytes()[:40])                 # a truncated file
    out = transfer.cached_arrays(key, compute, folder=tmp_path, verbose=False)
    assert len(calls) == 3 and np.array_equal(out["a"], np.arange(3.0)), "expected a truncated file recomputed"


def test_cached_arrays_refuse_bad_results(tmp_path):
    with pytest.raises(ValueError):
        transfer.cached_arrays({"k": 1}, lambda: {}, folder=tmp_path, verbose=False)
    with pytest.raises(ValueError):
        transfer.cached_arrays({"k": 2}, lambda: {"__key__": np.zeros(1)}, folder=tmp_path, verbose=False)


def test_cached_embeddings_key_follows_images_size_and_seed(tmp_path):
    bb = _fake_backbone()
    a = transfer.cached_embeddings(bb, _images(), size=16, subset="s", seed=0, folder=tmp_path, verbose=False)
    assert np.allclose(a, bb.embed(_images(), 16), atol=1e-6)
    keys = {transfer.cache_file(transfer.embedding_key(bb, imgs, size=size, block=None, subset="s", seed=seed), tmp_path)
            for imgs, size, seed in [(_images(), 16, 0), (_images(seed=1), 16, 0), (_images(), 24, 0), (_images(), 16, 1)]}
    assert len(keys) == 4, "expected a different file for other images, another size or another seed"
    transfer.cached_embeddings(bb, _images(seed=1), size=16, subset="s", seed=0, folder=tmp_path, verbose=False)
    assert len(list(tmp_path.glob("*.npz"))) == 2
    assert transfer.clear_embeddings_cache(tmp_path) == 2 and not list(tmp_path.glob("*.npz"))


def test_embeddings_dir_lives_in_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("WB_CACHE_DIR", str(tmp_path))
    assert transfer.embeddings_dir() == tmp_path / "embeddings" and (tmp_path / "embeddings").is_dir()


# ---------------------------------------------------------------------------
# Loading and the fallback
# ---------------------------------------------------------------------------
def test_unknown_backbone_is_refused():
    with pytest.raises(ValueError):
        transfer.load_backbone("vgg11")


def test_try_backbone_falls_back_to_none_with_a_pending_line(monkeypatch, capsys):
    def offline(name, spec):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(transfer, "_LOADED", {})
    monkeypatch.setattr(transfer, "_load_hub", offline)
    with pytest.raises(transfer.BackboneUnavailable, match="clip_b32"):
        transfer.load_backbone("clip_b32")
    assert transfer.try_backbone("clip_b32", "B1.24") is None, "expected None when the weights cannot be loaded"
    out = capsys.readouterr().out
    assert out.startswith("⏳ Ex B1.24 :") and "Colab" in out, f"expected a ⏳ line about Colab, got {out!r}"


@pytest.mark.network
def test_torchvision_backbones_are_frozen_with_their_preprocessing():
    for name, dim in (("resnet18", 512), ("mobilenet_v3_small", 576)):
        bb = transfer.load_backbone(name)
        assert bb.dim == dim and not bb.model.training and not any(p.requires_grad for p in bb.model.parameters())
        assert bb.mean == datasets.IMAGENET_MEAN and bb.resample == "bilinear" and len(bb.categories) == 1000
        e = bb.embed(_images(3), 64)
        assert e.shape == (3, dim), f"expected (3, {dim}), got {e.shape}"
    bb = transfer.load_backbone("resnet18")
    assert np.allclose(bb.embed(_images(3), 64, block="layer4"), bb.embed(_images(3), 64), atol=1e-5)
    shapes = {block: bb.embed(_images(2), 64, block=block).shape[1] for block in transfer.RESNET_BLOCKS}
    assert shapes == {"stem": 64, "layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}, shapes
    together = bb.embed_blocks(_images(3), 64, batch_size=2)
    for block in transfer.RESNET_BLOCKS:
        assert np.allclose(together[block], bb.embed(_images(3), 64, block=block), atol=1e-5), block
    two = bb.embed_blocks(_images(3), 64, blocks=("layer2", "stem"))
    assert set(two) == {"layer2", "stem"}, "expected only the blocks asked"
    with pytest.raises(ValueError):
        bb.embed_blocks(_images(3), 64, blocks=("layer5",))


@pytest.mark.network
def test_cached_block_embeddings(tmp_path):
    bb = transfer.load_backbone("resnet18")
    got = transfer.cached_block_embeddings(bb, _images(4), size=48, blocks=("stem", "layer3"), subset="s",
                                           folder=tmp_path, verbose=False)
    again = transfer.cached_block_embeddings(bb, _images(4), size=48, blocks=("stem", "layer3"), subset="s",
                                             folder=tmp_path, verbose=False)
    assert set(got) == {"stem", "layer3"} and got["layer3"].shape == (4, 256)
    assert np.array_equal(got["stem"], again["stem"]) and len(list(tmp_path.glob("*.npz"))) == 1


@pytest.mark.network
def test_hub_backbones_at_their_revisions():
    dino = transfer.load_backbone("dinov2_small")
    assert dino.dim == 384 and dino.revision == transfer.DINOV2_REVISION and dino.resample == "bicubic"
    x = dino.preprocess(_images(2), 56)
    with torch.inference_mode():
        out = dino.full(pixel_values=x)
    assert torch.allclose(out.pooler_output, out.last_hidden_state[:, 0], atol=1e-6), "pooler_output = CLS token"
    assert dino.embed(_images(2), 56).shape == (2, 384)
    clip = transfer.load_backbone("clip_b32")
    assert clip.dim == 512 and clip.mean == transfer.CLIP_MEAN
    assert abs(float(clip.full.logit_scale.detach().exp()) - 100.0) < 1e-3, "expected exp(logit_scale) = 100"
    assert clip.embed(_images(2), 224).shape == (2, 512)
