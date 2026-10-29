"""Tests of cifarkit: the pipeline, the models, the training loop, the saved model and the attack.

    python -m pytest projets/partie_5_classifieur_cifar10/solution -q

They run on small random images (a few seconds on a CPU): what they check does not depend on CIFAR-10. The oracles
are NumPy, PyTorch (torchvision's transforms, F.cross_entropy) and properties written by hand; never another function
of cifarkit.
"""

import csv

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cifarkit import (RUN_COLUMNS, ResidualBlock, build_model, channel_stats, evaluate, fgsm, fit, flat_size,
                      load_model, log_run, make_loader, make_transform, n_parameters, predict_proba, save_model,
                      stratified_split)


@pytest.fixture
def images():
    """60 random uint8 images (60, 32, 32, 3) and their labels: 6 per class."""
    rng = np.random.default_rng(5)
    return rng.integers(0, 256, size=(60, 32, 32, 3), dtype=np.uint8), np.repeat(np.arange(10), 6)


def tiny_loaders(n=96, batch_size=32, seed=0):
    """Loaders of normalized random tensors (no transform), for the training tests."""
    g = torch.Generator().manual_seed(7)
    x, y = torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)
    train = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(x[:40], y[:40]), batch_size=16)
    return train, val


# ---------------------------------------------------------------- the pipeline
def test_stratified_split_keeps_each_class_share():
    y = np.repeat(np.arange(10), 30)                                        # 300 labels, 30 per class
    train_idx, val_idx = stratified_split(y, 50, seed=0)
    assert len(val_idx) == 50 and len(train_idx) == 250, f"expected 250 + 50 indices, got {len(train_idx)} + {len(val_idx)}"
    assert not set(train_idx) & set(val_idx), "expected disjoint train and validation indices"
    counts = np.bincount(y[val_idx], minlength=10)
    assert counts.min() >= 4 and counts.max() <= 6, f"expected about 5 validation images per class, got {counts.tolist()}"


def test_channel_stats_match_numpy(images):
    X, _ = images
    mean, std = channel_stats(X)
    x = X.astype(np.float64) / 255
    want_mean, want_std = [x[..., c].mean() for c in range(3)], [x[..., c].std() for c in range(3)]
    assert np.allclose(mean, want_mean) and np.allclose(std, want_std), \
        f"expected per-channel mean {np.round(want_mean, 4)} and std {np.round(want_std, 4)}, got {mean} and {std}"


def test_transform_without_augmentation_normalizes_per_channel(images):
    X, _ = images
    mean, std = [0.5, 0.4, 0.3], [0.2, 0.25, 0.3]
    out = make_transform(mean, std, augment=False)(X[0])
    want = (torch.from_numpy(X[0].astype(np.float32) / 255).permute(2, 0, 1) - torch.tensor(mean).view(3, 1, 1)) \
        / torch.tensor(std).view(3, 1, 1)
    assert out.shape == (3, 32, 32) and out.dtype == torch.float32, f"expected a float32 (3, 32, 32), got {out.dtype} {tuple(out.shape)}"
    assert torch.allclose(out, want, atol=1e-6), "expected (x / 255 - mean) / std, channel by channel"


def test_augmented_transform_keeps_the_shape_and_varies(images):
    X, _ = images
    transform = make_transform([0.5] * 3, [0.25] * 3, augment=True)
    torch.manual_seed(0)
    outs = [transform(X[0]) for _ in range(8)]
    assert all(o.shape == (3, 32, 32) for o in outs), "expected (3, 32, 32) after RandomCrop(32, padding=4)"
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), "expected random crops and flips: 8 draws all equal"


def test_loader_gives_normalized_batches_and_int_labels(images):
    X, y = images
    loader = make_loader(X, y, make_transform([0.5] * 3, [0.25] * 3), batch_size=16, shuffle=True, seed=3)
    xb, yb = next(iter(loader))
    assert tuple(xb.shape) == (16, 3, 32, 32) and yb.dtype == torch.int64, \
        f"expected (16, 3, 32, 32) images and int64 labels, got {tuple(xb.shape)} and {yb.dtype}"
    again = next(iter(make_loader(X, y, make_transform([0.5] * 3, [0.25] * 3), batch_size=16, shuffle=True, seed=3)))
    assert torch.equal(yb, again[1]), "expected the same order for the same seed (the loader's own generator)"


# ---------------------------------------------------------------- the models
@pytest.mark.parametrize("config, params", [
    ({"arch": "mlp", "hidden": 64}, 3072 * 64 + 64 + 64 * 10 + 10),
    ({"arch": "cnn", "widths": [8, 16]}, (3 * 9 * 8 + 8) + (8 * 9 * 16 + 16) + 16 * 8 * 8 * 10 + 10),
    ({"arch": "vgg", "widths": [8, 16], "downsample": "stride", "batchnorm": False}, None),
    ({"arch": "vgg", "widths": [8, 16], "downsample": "pool", "batchnorm": True, "dropout": 0.3}, None),
    ({"arch": "resnet", "widths": [8, 16], "blocks": 2}, None),
], ids=["mlp", "cnn", "vgg-stride", "vgg-pool", "resnet"])
def test_models_give_ten_logits(config, params):
    model = build_model(config).eval()
    out = model(torch.zeros(2, 3, 32, 32))
    assert tuple(out.shape) == (2, 10), f"{config['arch']}: expected logits (2, 10), got {tuple(out.shape)}"
    if params is not None:
        assert n_parameters(model) == params, f"{config['arch']}: expected {params} parameters, got {n_parameters(model)}"


def test_flat_size_follows_the_pooling():
    assert flat_size(32, [8, 16]) == 16 * 8 * 8, f"expected 16 * 8 * 8 = 1024 values, got {flat_size(32, [8, 16])}"
    assert flat_size(32, [4, 4, 4]) == 4 * 4 * 4, f"expected 4 * 4 * 4 = 64 values, got {flat_size(32, [4, 4, 4])}"


def test_residual_block_adds_its_shortcut():
    torch.manual_seed(0)
    block = ResidualBlock(4, 4).eval()
    x = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        inner = block.bn2(block.conv2(torch.relu(block.bn1(block.conv1(x)))))
        assert torch.allclose(block(x), torch.relu(inner + x), atol=1e-6), "expected relu(F(x) + x) with the identity"
        down = ResidualBlock(4, 8, stride=2).eval()
        assert tuple(down(x).shape) == (2, 8, 4, 4), f"expected (2, 8, 4, 4) with stride 2, got {tuple(down(x).shape)}"


def test_unknown_arch_is_refused():
    with pytest.raises(ValueError):
        build_model({"arch": "transformer"})


# ---------------------------------------------------------------- training
def test_evaluate_is_a_mean_per_example():
    torch.manual_seed(0)
    model = build_model({"arch": "mlp", "hidden": 16})
    x, y = torch.randn(50, 3, 32, 32), torch.randint(0, 10, (50,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)                 # 32 + 18: unequal mini-batches
    loss, acc = evaluate(model, loader)
    with torch.no_grad():
        logits = model.eval()(x)
    want = F.cross_entropy(logits, y).item()
    assert abs(loss - want) < 1e-5, f"expected the mean over the 50 examples, {want:.6f}, got {loss:.6f}"
    assert abs(acc - (logits.argmax(1) == y).sum().item() / 50) < 1e-12, "expected the accuracy over the 50 examples"


def test_fit_keeps_the_best_epoch(tmp_path):
    torch.manual_seed(0)
    model = build_model({"arch": "cnn", "widths": [4, 8]})
    train, val = tiny_loaders()
    history = fit(model, train, val, epochs=4, lr=0.05, schedule="constant", folder=tmp_path)
    best = history["best_epoch"]
    assert len(history["val_acc"]) == 4 and best == int(np.argmax(history["val_acc"])) + 1, \
        f"expected best_epoch = the first epoch of highest validation accuracy, got {best} for {history['val_acc']}"
    _, acc = evaluate(model, val)
    assert acc == history["val_acc"][best - 1], f"expected the final model to have the accuracy of epoch {best}"
    saved = torch.load(tmp_path / "best.pt", weights_only=True)
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items()), "expected best.pt = the final weights"


def test_resume_gives_the_same_weights(tmp_path):
    def run(stop, folder, resume=False, seed_model=True):
        if seed_model:
            torch.manual_seed(0)
        model = build_model({"arch": "vgg", "widths": [4, 8], "dropout": 0.3})
        train, val = tiny_loaders()
        fit(model, train, val, epochs=4, lr=0.05, schedule="onecycle", folder=folder, resume=resume, stop_after=stop)
        return model
    straight = run(None, tmp_path / "a")
    run(2, tmp_path / "b")                                                   # interrupted after 2 epochs out of 4
    torch.manual_seed(123)                                                   # another state: resume must restore it
    resumed = run(None, tmp_path / "b", resume=True, seed_model=False)
    gaps = [(k, (a - b).abs().max().item()) for (k, a), b in
            zip(straight.state_dict().items(), resumed.state_dict().values()) if a.is_floating_point()]
    worst = max(gaps, key=lambda kv: kv[1])
    assert worst[1] == 0.0, f"expected 2 + 2 epochs = 4 epochs in a row bit for bit, largest gap {worst[1]:.2e} on {worst[0]}"


def test_save_and_load_give_the_same_predictions(tmp_path, images):
    X, _ = images
    torch.manual_seed(0)
    config = {"arch": "resnet", "widths": [4, 8]}
    model = build_model(config).eval()
    mean, std = channel_stats(X)
    save_model(tmp_path, model, config, mean, std, {"val_acc": 0.5})
    loaded, info = load_model(tmp_path)
    assert not loaded.training, "expected load_model to return a model in evaluation mode"
    assert info["mean"] == pytest.approx(mean) and info["config"] == config, "expected model.json to keep config, mean, std"
    assert np.allclose(predict_proba(model, X[:5], mean, std), predict_proba(loaded, X[:5], mean, std)), \
        "expected the same probabilities after save_model and load_model"


def test_fgsm_stays_in_the_epsilon_ball_and_raises_the_loss():
    torch.manual_seed(0)
    model = build_model({"arch": "cnn", "widths": [4, 8]}).eval()
    x, y = torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,))
    mean, std = [0.5] * 3, [0.25] * 3
    adv = fgsm(model, x, y, 4 / 255, mean, std)
    assert (adv - x).abs().max() <= 4 / 255 + 1e-6 and adv.min() >= 0 and adv.max() <= 1, \
        "expected |x' - x| <= epsilon and pixels in [0, 1]"
    with torch.no_grad():
        before = F.cross_entropy(model((x - 0.5) / 0.25), y).item()
        after = F.cross_entropy(model((adv - 0.5) / 0.25), y).item()
    assert after > before, f"expected the attack to raise the loss, {before:.4f} -> {after:.4f}"
    assert all(p.grad is None for p in model.parameters()), "expected fgsm to leave the parameters' gradients alone"


def test_log_run_appends_rows(tmp_path):
    path = tmp_path / "runs.csv"
    log_run(path, {"run": "a", "val_acc": 0.5, "config": {"arch": "mlp"}})
    log_run(path, {"run": "b", "val_acc": 0.6})
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["run"] for r in rows] == ["a", "b"] and list(rows[0]) == RUN_COLUMNS, \
        "expected two rows under the header RUN_COLUMNS"
    with pytest.raises(ValueError):
        log_run(path, {"accuracy": 0.7})
