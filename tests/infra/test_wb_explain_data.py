"""Tests of wb.explain_data: the frozen tabular splits, the cache of the two trained networks (with their provenance),
the learner's ResNet of Ex B1.27 read from its folder, and the reference extractors used by the checks of chapter B6
(hooks and autograd, integrated gradients, Grad-CAM, cascading randomisation, rank correlations)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from wb import explain_data as X  # noqa: E402


# ------------------------------------------------------------------------------------------------------- tabular data
def test_california_split_is_frozen_disjoint_and_float32_exact():
    cal = X.california()
    assert cal.sizes() == (12384, 4128, 4128)
    assert cal.features == ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude",
                            "Longitude"]
    idx = [set(part.index) for part in (cal.X_train, cal.X_val, cal.X_test)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert sum(map(len, idx)) == 20640
    values = cal.X_train.to_numpy()
    assert values.dtype == np.float64 and np.array_equal(values, values.astype(np.float32).astype(np.float64))
    assert X.california() is cal                                     # read once per session
    assert (cal.y_train.index == cal.X_train.index).all()


def test_california_treeshap_matches_the_exact_shapley_values():
    """The reason of the float32 rounding: on these values TreeSHAP (interventional) is exact."""
    shap = pytest.importorskip("shap")
    from sklearn.ensemble import HistGradientBoostingRegressor

    cal = X.california()
    model = HistGradientBoostingRegressor(max_iter=30, random_state=0).fit(cal.X_train, cal.y_train)
    bg = cal.X_train.iloc[:20].to_numpy()
    rows = cal.X_val.iloc[:3].to_numpy()
    tree = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")(rows)
    exact = shap.explainers.Exact(model.predict, shap.maskers.Independent(bg, max_samples=len(bg)))(rows)
    assert np.abs(tree.values - exact.values).max() < 1e-6
    assert np.allclose(tree.values.sum(1) + tree.base_values, model.predict(rows), atol=1e-6)


def test_penguins_split_is_stratified_with_sorted_classes():
    peng = X.penguins()
    assert peng.sizes() == (249, 0, 84) and peng.X_val is None
    assert peng.features == X.PENGUIN_FEATURES and peng.classes == ["Adelie", "Chinstrap", "Gentoo"]
    assert not peng.X_train.isna().any().any()
    share = lambda y: y.value_counts(normalize=True).sort_index().to_numpy()
    assert np.abs(share(peng.y_train) - share(peng.y_test)).max() < 0.02


def test_adult_keeps_the_test_part_of_the_mini_project():
    from sklearn.model_selection import train_test_split

    census = X.adult()
    assert census.sizes() == (29304, 9769, 9769)
    y_all = np.concatenate([census.y_train, census.y_val, census.y_test])
    assert set(np.unique(y_all)) == {0, 1} and census.classes == [0, 1]
    n = 48842
    from wb import datasets

    y = (datasets.load_adult()["class"] == ">50K").astype(int).to_numpy()
    _, test = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=2026)   # income.make_test_indices
    assert np.array_equal(np.sort(test), census.X_test.index.to_numpy())
    assert "sex" in census.features and "class" not in census.features


# --------------------------------------------------------------------------------------------------- tiny test models
def _tiny_cnn(seed=0):
    torch.manual_seed(seed)
    return X._cnn_class()((4, 4, 4)).eval()


class _Linear(nn.Module):
    """logits = W x.flatten(): the gradient of logit k is W[k], exactly."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 3, bias=True)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _images(n=2, seed=0, shape=(3, 32, 32)):
    return torch.rand((n, *shape), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------------------------------------- reference extractors
def test_activations_and_gradients_match_a_manual_backward_and_leave_no_trace():
    model = _tiny_cnn()
    x, target = _images(3), [1, 4, 7]
    layer = model.features[14]
    A, G, logits = X.activations_and_gradients(model, layer, x, target)
    store = {}
    handle = layer.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))
    out = model(x)
    (want,) = torch.autograd.grad(out[torch.arange(3), torch.tensor(target)].sum(), store["A"])
    handle.remove()
    assert A.shape == G.shape == (3, 4, 8, 8) and logits.shape == (3, 10) and A.dtype == np.float64
    assert np.allclose(A, store["A"].detach().numpy(), atol=1e-6) and np.allclose(G, want.numpy(), atol=1e-6)
    assert not layer._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def test_activations_and_gradients_work_on_a_frozen_model_and_broadcast_one_target():
    model = X._freeze(_tiny_cnn())
    A, G, _ = X.activations_and_gradients(model, model.features[3], _images(2), 5)
    A2, G2, _ = X.activations_and_gradients(model, model.features[3], _images(2), [5, 5])
    assert np.array_equal(G, G2) and np.abs(G).sum() > 0
    with pytest.raises(ValueError, match="one target class per image"):
        X.activations_and_gradients(model, model.features[3], _images(2), [1, 2, 3])
    unused = nn.Conv2d(3, 3, 1)
    with pytest.raises(ValueError, match="not used"):
        X.activations_and_gradients(model, unused, _images(2), 0)


def test_leftover_hooks_finds_every_kind_and_clear_hooks_removes_them():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    assert X.leftover_hooks(net) == {}
    calls = []
    net[0].register_forward_hook(lambda m, i, o: calls.append("forward"))
    net[0].register_forward_pre_hook(lambda m, i: calls.append("pre"))
    net[2].register_full_backward_hook(lambda m, gi, go: calls.append("backward"))
    net.register_forward_hook(lambda m, i, o, **kw: None, with_kwargs=True)
    left = X.leftover_hooks(net)
    assert left == {"": ["forward_hooks"], "0": ["forward_hooks", "forward_pre_hooks"], "2": ["backward_hooks"]}
    assert X.clear_hooks(net) == left and X.leftover_hooks(net) == {}
    net(torch.ones(1, 3, requires_grad=True)).sum().backward()      # no hook fires, nothing breaks
    assert calls == []
    net[2].register_full_backward_hook(lambda m, gi, go: calls.append("again"))   # a new hook works after clearing
    net(torch.ones(1, 3, requires_grad=True)).sum().backward()
    assert calls == ["again"]


def test_input_gradients_and_integrated_gradients_on_a_linear_model_are_exact():
    model = X._freeze(_Linear().eval())
    x = _images(2, shape=(3, 2, 2))
    ref = torch.full_like(x, 0.25)
    W = model.fc.weight.detach().double().numpy()
    g = X.input_gradients(model, x, [0, 2])
    assert np.allclose(g.reshape(2, -1), W[[0, 2]], atol=1e-6)
    ig = X.integrated_gradients(model, x, [0, 2], baseline=ref, n_steps=3)
    want = (x - ref).double().numpy().reshape(2, -1) * W[[0, 2]]
    assert np.allclose(ig.reshape(2, -1), want, atol=1e-6)
    with pytest.raises(ValueError, match="baseline"):
        X.integrated_gradients(model, x, 0, baseline=ref[:1])


def test_integrated_gradients_are_complete_on_the_small_cnn():
    model = X._freeze(_tiny_cnn(1))
    x = _images(2, seed=3)
    ig = X.integrated_gradients(model, x, [2, 6], n_steps=200)
    with torch.no_grad():
        gap = (model(x) - model(torch.zeros_like(x)))[torch.arange(2), torch.tensor([2, 6])].double().numpy()
    assert np.allclose(ig.reshape(2, -1).sum(1), gap, rtol=0.02, atol=1e-3)


def test_grad_cam_is_the_normalised_relu_of_the_weighted_sum():
    model = X._freeze(_tiny_cnn(2))
    x = _images(2, seed=4)
    A, G, _ = X.activations_and_gradients(model, model.features[17], x, [0, 3])
    raw = np.einsum("nc,nchw->nhw", G.mean(axis=(2, 3)), A)
    cam = X.grad_cam(model, model.features[17], x, [0, 3])
    want = np.maximum(raw, 0)
    peak = want.max(axis=(1, 2), keepdims=True)
    want = np.where(peak > 0, want / np.where(peak > 0, peak, 1), 0)
    assert cam.shape == (2, 8, 8) and np.allclose(cam, want)
    signed = X.grad_cam(model, model.features[17], x, [0, 3], relu=False)
    assert np.allclose(signed, raw / np.abs(raw).max(axis=(1, 2), keepdims=True))
    big = X.grad_cam(model, model.features[17], x, [0, 3], size=32)
    assert big.shape == (2, 32, 32) and big.min() >= 0 and big.max() <= 1 + 1e-12


def test_grad_cam_of_a_zero_gradient_is_zero_without_nan():
    model = X._freeze(_tiny_cnn(2))
    with torch.no_grad():
        model.fc.weight.zero_()
    cam = X.grad_cam(model, model.features[17], _images(1), 0)
    assert np.array_equal(cam, np.zeros_like(cam))


def test_predict_and_accuracy_read_uint8_images():
    model = X._freeze(_tiny_cnn())
    images = (np.random.default_rng(0).random((5, 32, 32, 3)) * 255).astype(np.uint8)
    logits = X.predict(model, images, batch_size=2)
    with torch.no_grad():
        want = model(torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255).double().numpy()
    assert logits.shape == (5, 10) and np.allclose(logits, want, atol=1e-6)
    assert X.accuracy(model, images, logits.argmax(1)) == 1.0


# -------------------------------------------------------------------------------------------- cascading randomisation
def test_cascade_layers_go_from_the_output_down():
    from torchvision.models import resnet18

    assert X.cascade_layers(resnet18(weights=None)) == ["fc", "layer4.1", "layer4.0", "layer3.1", "layer3.0",
                                                        "layer2.1", "layer2.0", "layer1.1", "layer1.0", "conv1"]
    assert X.cascade_layers(_tiny_cnn()) == ["fc", "features.17", "features.14", "features.10", "features.7",
                                             "features.3", "features.0"]


def test_randomize_layers_changes_only_the_named_parts_and_not_the_global_generator():
    model = X._freeze(_tiny_cnn())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.random.get_rng_state()
    out = X.randomize_layers(model, ["fc", "features.17"], seed=3)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())       # the original is untouched
    changed = {k for k, v in out.state_dict().items() if not torch.equal(v, before[k])}
    assert changed == {"fc.weight", "fc.bias", "features.17.weight", "features.17.bias"}
    again = X.randomize_layers(model, ["fc", "features.17"], seed=3)
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in out.state_dict().items())
    assert not out.training and not any(p.requires_grad for p in out.parameters())
    with_bn = X.randomize_layers(model, ["features.14"], seed=0, include_batchnorm=True)
    assert torch.equal(with_bn.features[15].running_var, torch.ones(4))


def test_spearman_per_image():
    a = np.arange(8.0).reshape(2, 2, 2)
    assert np.allclose(X.spearman_per_image(a, a), [1, 1])
    assert np.allclose(X.spearman_per_image(a, -a), [-1, -1])
    assert np.allclose(X.spearman_per_image(a, -a, absolute=True), [1, 1])
    rho = X.spearman_per_image(a, np.stack([a[0], np.ones((2, 2))]))
    assert rho[0] == 1 and np.isnan(rho[1])
    with pytest.raises(ValueError, match="same shape"):
        X.spearman_per_image(a, a[:1])


# ---------------------------------------------------------------------------------------- cache with its provenance
def _tiny_data(monkeypatch, per_class_seen):
    rng = np.random.default_rng(0)

    def load(per_class, part="train", seed=2011, channels_first=False):
        per_class_seen.append((per_class, part))
        n = 10 * min(per_class, 3)
        return (rng.random((n, 32, 32, 3)) * 255).astype(np.uint8), np.arange(n) % 10

    monkeypatch.setattr(X.datasets, "load_cifar10_subset", load)


def test_cifar_cnn_is_trained_once_then_read_back(tmp_path, monkeypatch, capsys):
    seen = []
    _tiny_data(monkeypatch, seen)
    monkeypatch.setitem(X.CNN_SETTINGS, "fast", {"widths": (4, 4, 4), "per_class": 2, "epochs": 1,
                                                  "batch_size": 8, "max_lr": 1e-3, "seed": 0})
    state = torch.random.get_rng_state()
    model, info = X.cifar_cnn(True, folder=tmp_path)
    assert torch.equal(torch.random.get_rng_state(), state)                  # training inside fork_rng
    assert "entraînement" in capsys.readouterr().out
    assert info["source"] == "reference" and info["path"].parent == tmp_path and 0 <= info["test_accuracy"] <= 1
    assert info["path"].name.startswith("cifar_cnn-reference-")
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    again, info2 = X.cifar_cnn(True, folder=tmp_path)
    assert "relu" in capsys.readouterr().out and len(seen) == 2              # no new training
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in model.state_dict().items())
    assert info2["test_accuracy"] == info["test_accuracy"]
    info["path"].write_bytes(b"truncated")                                    # a broken file: trained again
    X.cifar_cnn(True, folder=tmp_path, verbose=False)
    assert len(seen) == 4
    assert X.cnn_key(True)["source"] == "reference" and X.cnn_key(True) != X.cnn_key(False)
    assert X.cache_file(X.cnn_key(True), tmp_path) != X.cache_file(X.cnn_key(False), tmp_path)


def test_train_resnet_follows_the_recipe_of_b1_17(monkeypatch):
    rng = np.random.default_rng(0)
    images, labels = (rng.random((8, 32, 32, 3)) * 255).astype(np.uint8), np.arange(8) % 10
    settings = {"size": 32, "per_class": 1, "epochs": 1, "batch_size": 4, "head_lr": 1e-3, "gamma": 0.1,
                "weight_decay": 0.01, "seed": 0}
    from torchvision.models import resnet18

    torch.manual_seed(0)
    start = resnet18(weights=None).state_dict()
    monkeypatch.setattr("torchvision.models.resnet18", lambda weights=None: _seeded_resnet(resnet18, start))
    model = X.train_resnet(settings, images, labels, weights=None)
    after = model.state_dict()
    moved = {k.split(".")[0] for k, v in after.items() if k in start and not k.startswith("fc.")
             and not torch.equal(v, start[k])}
    assert moved == {"layer4"}                          # the frozen blocks and every running statistic stay
    assert not any(torch.equal(after[k], start[k]) for k in start if k.startswith("layer4.") and "conv" in k)
    assert after["fc.weight"].shape == (10, 512) and not model.training


def _seeded_resnet(real, state):
    net = real(weights=None)
    net.load_state_dict(state)
    return net


# --------------------------------------------------------------------------------------------- the learner's B1 model
def _publish(folder, size=64):
    from safetensors.torch import save_file
    from torchvision.models import resnet18

    torch.manual_seed(1)
    net = resnet18(weights=None)
    net.fc = nn.Linear(512, 10)
    folder.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(folder / "model.safetensors"))
    config = {"architecture": "resnet18", "num_classes": 10, "image_size": size, "class_names": list("abcdefghij"),
              "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25], "test_accuracy": 0.9}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return net.eval()


def test_the_learner_model_of_b1_27_is_found_read_and_frozen(tmp_path, monkeypatch):
    folder = tmp_path / "b1_transfer" / X.B1_FOLDER_NAME
    net = _publish(folder)
    monkeypatch.setattr("wb.core.repo_root", lambda *a, **k: tmp_path / "repo")
    assert X.find_b1_folder(tmp_path / "b6_explicabilite") == folder.resolve()
    assert X.find_b1_folder(tmp_path / "elsewhere" / "deeper") is None
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
    model, info = X.load_b1_folder(folder)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()} == before   # read, never written
    assert info["source"] == "learner" and info["size"] == 64 and info["classes"] == list("abcdefghij")
    assert info["mean"] == (0.5, 0.5, 0.5) and info["std"] == (0.25, 0.25, 0.25)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    images = (np.random.default_rng(0).random((2, 32, 32, 3)) * 255).astype(np.uint8)
    x = X.model_input(images, info)
    assert tuple(x.shape) == (2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), net(x), atol=1e-5)
    with pytest.raises(RuntimeError, match="does not require grad|doesn't require gradient"):
        model.layer4.register_forward_hook(lambda m, i, o: o.register_hook(lambda g: None))
        model(x)                                       # the trap of the fiche: a frozen model, an input without grad


def test_b1_resnet_falls_back_on_the_reference_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(X, "reference_resnet", lambda fast=True, folder=None, verbose=True: ("ref", {"source": "reference"}))
    (tmp_path / "broken").mkdir()
    model, info = X.b1_resnet(True, folder=tmp_path / "broken")
    assert model == "ref" and "ne se charge pas" in capsys.readouterr().out
    model, info = X.b1_resnet(True, folder=None)
    assert model == "ref" and "pas de modèle publié" in capsys.readouterr().out
    monkeypatch.setattr("wb.core.repo_root", lambda *a, **k: tmp_path)       # never the real mon_travail/
    assert X.find_b1_folder(tmp_path / "nowhere") is None


def test_the_solutions_notebook_never_reads_the_learner_model_of_b1():
    """The solutions notebook of B6 sets B1_FOLDER = None and never looks for the folder of Ex B1.27; the exercise
    notebook looks for it (and only reads it)."""
    from wb.core import repo_root

    folder = Path(repo_root()) / "chapitres" / "b6_explicabilite"
    if not (folder / "05_solutions.ipynb").exists():
        pytest.skip("chapter B6 not built")
    sources = {}
    for name in ("03_notebook", "05_solutions"):
        nb = json.loads((folder / f"{name}.ipynb").read_text(encoding="utf-8"))
        sources[name] = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "B1_FOLDER = None" in sources["05_solutions"] and "find_b1_folder(" not in sources["05_solutions"]
    assert "load_b1_folder(" not in sources["05_solutions"]
    assert "B1_FOLDER = wbx.find_b1_folder()" in sources["03_notebook"]
