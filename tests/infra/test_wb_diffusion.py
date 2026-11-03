"""Tests of wb.diffusion: images in [-1, 1] and frozen subsets, the noise schedules of the chapter, the 2-D toys and
their measures, the classifier of the evaluations, the reference U-Net of Ex B5.21 and the single checkpoint of a
trained diffusion model (bonus chapter B5)."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wb import diffusion as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _reference(module: str):
    """A module of the reference library, imported by its own path (learner mode has no solutions/ on sys.path)."""
    if str(ROOT / "solutions") not in sys.path:
        sys.path.insert(0, str(ROOT / "solutions"))
    return importlib.import_module(f"mylearn_ref.{module}")


# --------------------------------------------------------------------------------------------------- images and subsets
def test_to_signed_maps_uint8_and_unit_floats_to_minus_one_one():
    x = D.to_signed(np.array([[[0, 255], [127, 128]]], dtype=np.uint8))
    assert x.dtype == torch.float32 and tuple(x.shape) == (1, 1, 2, 2)
    assert x[0, 0, 0, 0] == -1 and x[0, 0, 0, 1] == 1
    assert torch.allclose(D.to_signed(torch.tensor([[[0.0, 0.5, 1.0]]])), torch.tensor([[[[-1.0, 0.0, 1.0]]]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        D.to_signed(np.full((1, 2, 2), -0.5))
    with pytest.raises(ValueError, match="dtype"):
        D.to_signed(np.ones((1, 2, 2), dtype=np.int64))
    assert np.allclose(D.to_unit(np.array([-2.0, -1.0, 0.0, 1.0])), [0, 0, 0.5, 1])
    assert torch.equal(D.to_unit(torch.tensor([-1.0, 3.0])), torch.tensor([0.0, 1.0]))


def test_image_subsets_are_frozen_nested_and_in_range():
    small, big = D.image_subset("mnist", 500), D.image_subset("mnist", 2000)
    assert small.images.shape == (500, 1, 28, 28) and small.images.dtype == torch.float32
    assert float(small.images.min()) == -1.0 and float(small.images.max()) == 1.0
    assert np.array_equal(big.indices[:500], small.indices), "expected nested subsets (a fixed permutation)"
    assert torch.equal(big.images[:500], small.images) and torch.equal(big.labels[:500], small.labels)
    assert D.image_subset("mnist", 500).fingerprint == small.fingerprint and len(small.fingerprint) == 16
    assert small.fingerprint != big.fingerprint
    X, y = D._raw("mnist", "train")
    assert np.array_equal(small.labels.numpy(), y[small.indices])
    assert small.describe() == {"name": "mnist", "split": "train", "n": 500, "fingerprint": small.fingerprint}
    assert D.n_train(True) == 10_000 and D.n_train(False) == 60_000
    with pytest.raises(ValueError, match="unknown dataset"):
        D.image_subset("cifar", 10)
    with pytest.raises(ValueError, match="n must be"):
        D.image_subset("mnist", 0)


def test_test_subset_uses_its_own_permutation():
    test = D.image_subset("mnist", 300, split="test")
    assert test.split == "test" and len(test) == 300 and test.indices.max() < 10_000


# ---------------------------------------------------------------------------------------------------------- schedules
def test_schedules_match_the_reference_library():
    ref = _reference("diffusion")
    assert D.n_steps(True) == 400 and D.n_steps(False) == 1000
    np.testing.assert_allclose(D.schedule_betas("linear", 400), ref.linear_beta_schedule(400, 2.5e-4, 0.05), rtol=1e-12)
    np.testing.assert_allclose(D.schedule_betas("linear", 1000), ref.linear_beta_schedule(1000), rtol=1e-12)
    for T in (5, 400, 1000):
        np.testing.assert_allclose(D.schedule_betas("cosine", T), ref.cosine_beta_schedule(T), rtol=1e-9)
    ab = D.alpha_bars(D.schedule_betas("linear", 400))
    np.testing.assert_allclose(ab, ref.noise_schedule(D.schedule_betas("linear", 400))["alpha_bars"], rtol=1e-12)
    assert 3e-5 < ab[-1] < 5e-5, "expected the scaled bounds to keep alpha_bar_T near 4e-5"
    with pytest.raises(ValueError, match="unknown schedule"):
        D.schedule_betas("quadratic", 10)


def test_ddim_timesteps_start_at_the_last_index():
    assert D.ddim_timesteps(400, 1).tolist() == [399]
    assert D.ddim_timesteps(400, 2).tolist() == [399, 0]
    steps = D.ddim_timesteps(1000, 50)
    assert steps[0] == 999 and steps[-1] == 0 and len(steps) == 50 and np.all(np.diff(steps) < 0)
    assert D.ddim_timesteps(10, 10).tolist() == list(range(9, -1, -1))
    with pytest.raises(ValueError):
        D.ddim_timesteps(10, 11)


# ---------------------------------------------------------------------------------------------------------- 2-D toys
def test_ring_and_its_measures():
    pts = D.ring(4000, seed=0)
    assert pts.shape == (4000, 2) and np.array_equal(pts, D.ring(4000, seed=0))
    assert abs(pts.var(axis=0).mean() - (D.RING_RADIUS ** 2 / 2 + D.RING_STD ** 2)) < 0.05
    report = D.mode_report(pts)
    assert report["modes"] == 8 and report["on_mode"] > 0.98 and sum(report["counts"]) <= 4000
    blob = np.random.default_rng(1).standard_normal((4000, 2))
    assert D.mode_report(blob)["on_mode"] < 0.5
    floor = D.sliced_wasserstein(D.ring(2000, 1), D.ring(2000, 2))
    assert floor < 0.05 < D.sliced_wasserstein(blob[:2000], D.ring(2000, 2))
    assert D.sliced_wasserstein(pts[:10], pts[:10]) == 0.0
    assert D.sliced_wasserstein(np.full((3, 2), np.nan), pts[:3]) == float("inf")
    assert D.mode_report(np.full((3, 2), np.inf))["modes"] == 0
    with pytest.raises(ValueError):
        D.sliced_wasserstein(pts[:10], pts[:11])
    one_mode = D.ring_centers()[[0] * 100] + 0.01
    assert D.mode_report(one_mode)["modes"] == 1


def test_sliced_wasserstein_is_the_mean_of_one_dimensional_distances():
    scipy_stats = pytest.importorskip("scipy.stats")
    a, b = D.ring(300, 3), D.ring(300, 4) * 1.2
    angles = np.random.default_rng(0).uniform(0, np.pi, 8)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    want = np.mean([scipy_stats.wasserstein_distance(a @ d, b @ d) for d in dirs])
    assert abs(D.sliced_wasserstein(a, b, n_dirs=8, seed=0) - want) < 1e-12


# ---------------------------------------------------------------------------------------------------------- the U-Net
def test_sinusoidal_table_is_positional_encoding_of_b3():
    ref = _reference("attention")
    np.testing.assert_allclose(D.sinusoidal_table(400, 128), ref.positional_encoding(400, 128), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        D.sinusoidal_table(4, 3)


def test_reference_unet_is_the_one_of_the_fiche():
    torch.manual_seed(0)
    net = D.TinyUNet(400)
    assert D.count_params(net) == 740_865, "expected the 740 865 parameters of the U-Net of the fiche"
    assert D.count_params(D.ResBlock(64, 64, 128)) == 82_368
    assert "pe" not in net.state_dict(), "the table of the encoding is a non-persistent buffer"
    x, t = torch.randn(3, 1, 28, 28), torch.tensor([0, 17, 399])
    assert net(x, t).shape == (3, 1, 28, 28)
    shapes = D.stage_shapes(net, x, t)
    assert shapes["mid"] == (3, 128, 7, 7) and shapes["u2.in"] == (3, 128, 14, 14) and shapes["u1.in"] == (3, 64, 28, 28)
    assert shapes["out"] == (3, 1, 28, 28) and not net._forward_hooks and not net.u2._forward_pre_hooks
    cond = D.TinyUNet(400, n_classes=10)
    assert D.count_params(cond) == 740_865 + 11 * 128
    cond.load_state_dict({**net.state_dict(), "class_emb.weight": torch.zeros(11, 128)})
    with torch.no_grad():
        assert torch.allclose(cond(x, t), net(x, t), atol=1e-6), "a zero class embedding changes nothing"
        cond.class_emb.weight[10] = 1.0
        assert torch.allclose(cond(x, t), cond(x, t, torch.full((3,), 10)))   # y=None is the empty class
        assert not torch.allclose(cond(x, t, torch.zeros(3, dtype=torch.long)), cond(x, t))
    small = D.TinyUNet(100, channels=(16, 32, 64), t_dim=64)
    assert small.config == {"n_steps": 100, "channels": [16, 32, 64], "t_dim": 64, "n_classes": None}


# ---------------------------------------------------------------------------------------------------------- checkpoint
def test_checkpoint_round_trip_atomic_and_keyed_by_provenance(tmp_path):
    torch.manual_seed(1)
    net = D.TinyUNet(50, channels=(8, 16, 16), t_dim=16)
    ema = D.TinyUNet(50, channels=(8, 16, 16), t_dim=16)
    data = {"name": "mnist", "split": "train", "n": 10, "fingerprint": "0123456789abcdef"}
    settings = {"seed": 3, "schedule": "linear", "epochs": 2}
    key = D.checkpoint_key(net.config, settings, fast=True, source="learner", data=data)
    other = D.checkpoint_key(net.config, settings, fast=True, source="solution", data=data)
    assert D.checkpoint_file(key, tmp_path) != D.checkpoint_file(other, tmp_path)
    assert "learner" in D.checkpoint_file(key, tmp_path).name and "fast" in D.checkpoint_file(key, tmp_path).name
    assert D.load_checkpoint(key, tmp_path) is None
    path = D.save_checkpoint(key, net, ema=ema, history={"loss": [1.0, 0.5]}, folder=tmp_path, extra={"s": 2.0})
    assert path.exists() and not list(tmp_path.glob("*.tmp"))
    ckpt = D.load_checkpoint(key, tmp_path)
    assert ckpt["history"] == {"loss": [1.0, 0.5]} and ckpt["extra"] == {"s": 2.0} and ckpt["path"] == path
    assert D.load_checkpoint(other, tmp_path) is None, "another provenance never reads this file"
    raw, smooth = D.build_from_checkpoint(D.TinyUNet, ckpt, "raw"), D.build_from_checkpoint(D.TinyUNet, ckpt)
    assert not raw.training
    for (name, a), b in zip(raw.state_dict().items(), net.state_dict().values()):
        assert torch.equal(a, b), name
    for a, b in zip(smooth.state_dict().values(), ema.state_dict().values()):
        assert torch.equal(a, b)
    path.write_bytes(path.read_bytes()[:100])                      # a truncated file is as if absent
    assert D.load_checkpoint(key, tmp_path) is None
    assert D.delete_checkpoint(key, tmp_path) and not D.delete_checkpoint(key, tmp_path)
    D.save_checkpoint(key, net, folder=tmp_path)
    with pytest.raises(ValueError, match="no EMA"):
        D.build_from_checkpoint(D.TinyUNet, D.load_checkpoint(key, tmp_path))
    with pytest.raises(ValueError, match="source"):
        D.checkpoint_key(net.config, settings, fast=True, source="someone", data=data)
    image_set = D.image_subset("mnist", 20)
    assert D.checkpoint_key(net.config, settings, fast=False, source="reference", data=image_set)["data"] == \
        image_set.describe()


# ---------------------------------------------------------------------------------------------------------- classifier
def test_classifier_is_trained_once_then_reloaded(tmp_path, monkeypatch):
    monkeypatch.setitem(D.CLASSIFIER_SETTINGS, "n", 2000)
    monkeypatch.setitem(D.CLASSIFIER_SETTINGS, "epochs", 1)
    D._CLASSIFIERS.clear()
    clf = D.classifier("mnist", folder=tmp_path, verbose=False)
    assert not clf.training and not any(p.requires_grad for p in clf.parameters())
    files = list(tmp_path.glob("classifier-mnist-*.pt"))
    assert len(files) == 1
    D._CLASSIFIERS.clear()
    again = D.classifier("mnist", folder=tmp_path, verbose=False)
    assert again is not clf
    for a, b in zip(clf.state_dict().values(), again.state_dict().values()):
        assert torch.equal(a, b)
    test = D.image_subset("mnist", 200, split="test")
    feats = D.features(again, test.images)
    assert feats.shape == (200, 64) and feats.dtype == np.float64
    assert D.features(again, test.images.numpy().reshape(200, 784)).shape == (200, 64)
    accuracy = (D.predict(again, test.images) == test.labels.numpy()).mean()
    assert accuracy > 0.7, f"expected a classifier better than chance after one epoch, got {accuracy:.2f}"
    D._CLASSIFIERS.clear()


# ---------------------------------------------------------------------------------------------------------- 2-D models
def test_toy_reference_training_and_samplers_follow_the_library():
    ref = _reference("diffusion")
    sched = ref.noise_schedule(D.schedule_betas("linear", 40))
    torch.manual_seed(0)
    model = D.Denoiser2D(40)
    assert D.count_params(model) == (2 + 32) * 128 + 128 + 2 * (128 * 128 + 128) + 128 * 2 + 2
    assert "pe" not in model.state_dict()
    data = torch.tensor(D.ring(1000, 0), dtype=torch.float32)
    losses = D.train_toy(model, data, 5, 64, 1e-3, torch.Generator().manual_seed(1), sched["alpha_bars"])
    assert len(losses) == 5 and all(isinstance(v, float) for v in losses)
    torch.manual_seed(0)
    again = D.Denoiser2D(40)
    assert D.train_toy(again, data, 5, 64, 1e-3, torch.Generator().manual_seed(1), sched["alpha_bars"]) == losses
    model.eval()
    got = D.sample_toy_ddpm(model, 7, sched["alpha_bars"], torch.Generator().manual_seed(2), np.random.default_rng(3))
    x = torch.randn((7, 2), generator=torch.Generator().manual_seed(2)).double().numpy()
    rng = np.random.default_rng(3)
    for t in range(39, -1, -1):
        x = ref.ddpm_step(x, t, D._eps(model, x, t), sched, rng=rng)
    np.testing.assert_allclose(got, x, rtol=1e-10, atol=1e-12)
    got = D.sample_toy_ddim(model, 7, 5, sched["alpha_bars"], torch.Generator().manual_seed(4))
    x = torch.randn((7, 2), generator=torch.Generator().manual_seed(4)).double().numpy()
    steps = D.ddim_timesteps(40, 5).tolist() + [-1]
    for t, t_prev in zip(steps[:-1], steps[1:]):
        x = ref.ddim_step(x, t, t_prev, D._eps(model, x, t), sched["alpha_bars"])
    np.testing.assert_allclose(got, x, rtol=1e-10, atol=1e-12)
    flow = D.Denoiser2D(D.FM_POSITIONS)
    assert len(D.train_toy(flow, data, 3, 32, 1e-3, torch.Generator().manual_seed(5), flow=True)) == 3
    flow.eval()
    start = torch.randn((6, 2), generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        want = (start - flow(start, torch.full((6,), 1000))).double().numpy()
    np.testing.assert_allclose(D.sample_toy_euler(flow, 6, 1, torch.Generator().manual_seed(6)), want, rtol=1e-6)


# ------------------------------------------------------------------------------- training and sampling of the U-Net
def _small_unet(n_classes=None, seed=0):
    torch.manual_seed(seed)
    net = D.TinyUNet(12, channels=(8, 8, 16), t_dim=16, n_classes=n_classes)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn(p.shape))
    return net.eval()


def test_ema_update_is_the_formula_in_place():
    ema, model = _small_unet(seed=0), _small_unet(seed=1)
    before = [p.detach().clone() for p in ema.parameters()]
    ids = [id(p) for p in ema.parameters()]
    D.ema_update(ema, model, 0.9)
    assert [id(p) for p in ema.parameters()] == ids
    for new, old, p in zip(ema.parameters(), before, model.parameters()):
        assert torch.allclose(new, 0.9 * old + 0.1 * p)


def test_train_unet_follows_its_recipe_and_stops():
    """Two steps written by hand (same draws, same order) against train_unet; max_steps and the label dropout."""
    F = torch.nn.functional
    images = torch.rand(6, 1, 28, 28) * 2 - 1
    ab = D.alpha_bars(np.linspace(1e-3, 0.2, 12))
    settings = {"epochs": 1, "batch_size": 4, "lr": 1e-3, "ema": 0.5}
    a, ea, b, eb = (_small_unet() for _ in range(4))
    hist = D.train_unet(a, ea, images, settings, torch.Generator().manual_seed(3), ab)
    assert hist["steps"] == 2 and len(hist["loss"]) == 2 and all(isinstance(v, float) for v in hist["loss"])
    gen = torch.Generator().manual_seed(3)
    opt = torch.optim.AdamW(b.parameters(), lr=1e-3)
    abt = torch.tensor(ab, dtype=torch.float32)
    b.train()
    perm = torch.randperm(6, generator=gen)
    for idx in (perm[:4], perm[4:]):
        t = torch.randint(0, 12, (len(idx),), generator=gen)
        eps = torch.randn((len(idx), 1, 28, 28), generator=gen)
        x_t = abt[t].sqrt().view(-1, 1, 1, 1) * images[idx] + (1 - abt[t]).sqrt().view(-1, 1, 1, 1) * eps
        loss = F.mse_loss(b(x_t, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        D.ema_update(eb, b, 0.5)
    for p, q in zip(list(a.parameters()) + list(ea.parameters()), list(b.parameters()) + list(eb.parameters())):
        assert torch.allclose(p, q, atol=1e-6)
    c, ec = _small_unet(), _small_unet()
    assert D.train_unet(c, ec, images, settings | {"epochs": 5}, torch.Generator().manual_seed(3), ab, max_steps=3)["steps"] == 3
    m, em = _small_unet(n_classes=3), _small_unet(n_classes=3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    hist = D.train_unet(m, em, images, settings | {"p_uncond": 0.5}, torch.Generator().manual_seed(3), ab, labels=labels)
    assert hist["steps"] == 2 and not m.training and not em.training


def test_predict_eps_guidance_is_one_batched_call():
    net = _small_unet(n_classes=3)
    x, y = torch.randn(2, 1, 28, 28), torch.tensor([0, 2])
    calls = []
    handle = net.register_forward_pre_hook(lambda mod, args: calls.append(int(args[0].shape[0])))
    try:
        guided = D.predict_eps(net, x, 5, y, w=2.0)
    finally:
        handle.remove()
    assert calls == [4]
    t = torch.full((2,), 5)
    with torch.no_grad():
        want = 3 * net(x, t, y) - 2 * net(x, t, torch.full((2,), 3))
    assert torch.allclose(guided, want, atol=1e-5)
    assert torch.allclose(D.predict_eps(net, x, 5, y, w=0.0), net(x, t, y).detach(), atol=1e-6)


def test_samplers_follow_the_reference_library():
    """sample_ddpm: a loop of ddpm_step (posterior variance, clipped x̂₀, z from the same rng); sample_ddim: x̂₀
    clipped, the noise recomputed, then ddim_step."""
    ref = _reference("diffusion")
    net = _small_unet()
    betas = np.linspace(1e-3, 0.2, 12)
    ab = D.alpha_bars(betas)
    sched = ref.noise_schedule(betas)
    x_T = torch.randn(3, 1, 28, 28, generator=torch.Generator().manual_seed(1))

    def eps(x, t):
        with torch.no_grad():
            return net(torch.as_tensor(x, dtype=torch.float32), torch.full((len(x),), t)).double().numpy()

    x, rng = x_T.double().numpy(), np.random.default_rng(4)
    for t in range(11, -1, -1):
        x = ref.ddpm_step(x, t, eps(x, t), sched, rng=rng, clip_x0=1.0)
    assert np.allclose(D.sample_ddpm(net, x_T, ab, np.random.default_rng(4)).numpy(), x, atol=1e-5)
    x = x_T.double().numpy()
    seq = list(D.ddim_timesteps(12, 4)) + [-1]
    for t, t_prev in zip(seq[:-1], seq[1:]):
        e = eps(x, t)
        x0 = np.clip(ref.predict_x0_from_eps(x, np.full(3, t), e, ab), -1, 1)
        e = (x - np.sqrt(ab[t]) * x0) / np.sqrt(1 - ab[t])
        x = ref.ddim_step(x, t, t_prev, e, ab)
    assert np.allclose(D.sample_ddim(net, x_T, ab, 4).numpy(), x, atol=1e-5)
    unclipped = D.sample_ddim(net, x_T, ab, 4, clip_x0=None)
    assert unclipped.shape == (3, 1, 28, 28)


def test_ideal_denoiser_is_the_posterior_mean_of_a_finite_dataset():
    images = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).reshape(2, 1, 1, 2)
    c1, c2 = np.array([0.6]), np.array([0.8])
    eps_fn = D.ideal_denoiser(images, c1, c2)
    x = torch.tensor([[0.5, 0.1]]).reshape(1, 1, 1, 2)
    d = [((x.reshape(-1) - 0.6 * img.reshape(-1)) ** 2).sum() for img in images]
    w = torch.softmax(-torch.tensor(d) / (2 * 0.64), dim=0)
    x0 = w[0] * images[0].reshape(-1) + w[1] * images[1].reshape(-1)
    assert torch.allclose(eps_fn(x, 0).reshape(-1), (x.reshape(-1) - 0.6 * x0) / 0.8, atol=1e-6)
    far = 100 * images[:1]                       # x_t far on the side of the first image: its noise exactly
    assert torch.allclose(eps_fn(far, 0).reshape(-1), (far.reshape(-1) - 0.6 * images[0].reshape(-1)) / 0.8, atol=1e-4)


def test_copy_ratio_on_points_of_the_plane():
    train = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [10.0, 10.0]]).reshape(4, 1, 1, 2)
    queries = torch.tensor([[0.0, 0.5], [3.0, 3.0], [9.0, 9.0]]).reshape(3, 1, 1, 2)
    assert np.allclose(D.copy_ratio(queries, train, batch_size=2), [0.5 / 3, 1 / 4, np.sqrt(2) / np.sqrt(85)])


def test_cached_module_trains_once(tmp_path):
    calls = []

    def train():
        calls.append(1)
        net = torch.nn.Linear(2, 2)
        with torch.no_grad():
            net.weight.fill_(0.5)
        return net

    key = {"model": "toy", "settings": {"a": 1}}
    first = D.cached_module(key, lambda: torch.nn.Linear(2, 2), train, folder=tmp_path, verbose=False)
    second = D.cached_module(key, lambda: torch.nn.Linear(2, 2), train, folder=tmp_path, verbose=False)
    assert calls == [1] and torch.equal(first.weight, second.weight) and not second.training
    D.cached_module(key | {"settings": {"a": 2}}, lambda: torch.nn.Linear(2, 2), train, folder=tmp_path, verbose=False)
    assert calls == [1, 1]


def test_precomputed_protocol_of_the_guidance():
    table = D.precomputed("guidance_26")
    assert set(table["by_w"]) == {"0", "1", "3", "7"}
    assert all(len(v) == table["n_seeds"] for row in table["by_w"].values() for v in row.values())
    with pytest.raises(KeyError):
        D.precomputed("nothing")


def test_linear_schedule_refuses_a_too_short_chain():
    with pytest.raises(ValueError, match="T > 20"):
        D.schedule_betas("linear", 20)
    assert D.schedule_betas("linear", 21)[-1] < 1


def test_unet_settings_and_reference_model(tmp_path, monkeypatch):
    assert D.unet_settings(True)["T"] == 400 and D.unet_settings(False)["n_train"] == 60_000
    assert D.n_updates(10_000, D.unet_settings(True)) == 4 * 313
    monkeypatch.setitem(D.UNET_SETTINGS, True, D.UNET_SETTINGS[True] | {"epochs": 1, "channels": [8, 8, 16], "t_dim": 16})
    monkeypatch.setattr(D, "n_train", lambda fast=True: 64)
    model, ckpt = D.reference_model(True, folder=tmp_path, verbose=False)
    assert not model.training and ckpt["config"]["channels"] == [8, 8, 16]
    assert len(ckpt["history"]["loss"]) == 2 and ckpt["path"].name.startswith("ddpm-reference-fast-")
    again, _ = D.reference_model(True, folder=tmp_path, verbose=False)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), again.parameters()))
