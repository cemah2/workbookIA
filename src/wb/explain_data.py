"""Data, models and hooks of bonus chapter B6 (explainability and fairness).

Quick reference
---------------
>>> cal = california()                      # frozen split of California Housing (12 384 / 4 128 / 4 128 districts)
>>> cal.X_train, cal.y_train, cal.X_val, cal.y_val, cal.X_test, cal.y_test, cal.features
>>> peng = penguins()                       # frozen split of Penguins: 4 measures, the species (249 / 84 penguins)
>>> census = adult()                        # Adult: the test part of the MP3 (seed 2026), then train / validation
>>> cnn, info = cifar_cnn(fast=True)        # the small CNN of Ex B6.19: trained once (about 35 s), then read back
>>> net, info = b1_resnet(fast=True)        # ResNet-18 fine-tuned with the recipe of Ex B1.17 (layer4 + head)
>>> net, info = b1_resnet(fast=True, folder=find_b1_folder())   # or the learner's model published in Ex B1.27
>>> x = model_input(images_uint8, info)     # the float tensor the model expects (size, mean and std of `info`)
>>> A, G, logits = activations_and_gradients(net, net.layer4, x, target)   # the reference extractor
>>> g = input_gradients(cnn, x, target)     # d logit[target] / d input, (N, 3, H, W)
>>> ig = integrated_gradients(cnn, x, target, baseline=None, n_steps=50)
>>> cams = grad_cam(net, net.layer4, x, target, size=96)
>>> names = cascade_layers(net)             # the blocks with weights, from the output to the input
>>> copy = randomize_layers(net, names[:3], seed=0)   # a copy whose three top blocks are re-initialised
>>> rho = spearman_per_image(cams_a, cams_b)
>>> leftover_hooks(net), clear_hooks(net)   # the hooks a learner's code left on the modules (checks of B6.20, B6.21)

**Tabular data.** ``california`` rounds the eight features to float32 (then keeps them as float64). The reason:
``HistGradientBoostingRegressor`` puts some of its thresholds exactly on repeated values of the data (latitude and
longitude: midpoints of equal neighbours), and ``shap.TreeExplainer`` reads the data in float32 when it walks the
trees; on the float64 values of scikit-learn's copy, a value equal to a threshold can go to the other side, and
TreeSHAP loses its additivity (gaps up to 0.5 measured with shap 0.54.1). On float32 values, TreeSHAP
(interventional) and the exact Shapley values agree to about 1e-8.

**Models.** The two networks are trained with fixed settings (``CNN_SETTINGS``, ``RESNET_SETTINGS``) the first time,
then read from ``wb.datasets.cache_dir() / "explain"`` (never versioned), under a key that holds their provenance
(``source="reference"``: the workbook trained them, neither the learner nor the solutions notebook), the settings, the
mode, the data and the library versions; the file is written atomically. The learner's own ResNet of chapter B1 is
read from its published folder (``model.safetensors`` and ``config.json`` of Ex B1.27) and never written: when it is
missing or unreadable, :func:`b1_resnet` says so and falls back on the reference model (``source`` in ``info``).
Every model is returned in ``eval()`` mode with its parameters frozen (``requires_grad=False``), like a backbone of
chapter B1: a hook that wants the gradient of a layer needs an input that requires grad.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wb import datasets

CACHE_VERSION = 1
SOURCES = ("reference", "learner")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CNN_SETTINGS = {
    "fast": {"widths": (16, 32, 64), "per_class": 1000, "epochs": 4, "batch_size": 64, "max_lr": 4e-3, "seed": 1906},
    "full": {"widths": (32, 64, 128), "per_class": 2000, "epochs": 8, "batch_size": 64, "max_lr": 4e-3, "seed": 1906},
}
RESNET_SETTINGS = {
    "fast": {"size": 96, "per_class": 100, "epochs": 3, "batch_size": 32, "head_lr": 1e-3, "gamma": 0.1,
             "weight_decay": 0.01, "seed": 2006},
    "full": {"size": 128, "per_class": 300, "epochs": 3, "batch_size": 32, "head_lr": 1e-3, "gamma": 0.1,
             "weight_decay": 0.01, "seed": 2006},
}
EVAL_PER_CLASS = 50                 # the test images (per class) on which a trained model's accuracy is measured once
B1_FOLDER_NAME = "resnet18_cifar10_b1"   # the folder of Ex B1.27 (publish_27)


def _torch():
    import torch

    return torch


# ---------------------------------------------------------------------------------------------------------------------
# Tabular data
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class TabularSplit:
    """A frozen split: DataFrames of features and Series of targets (``X_val``/``y_val`` are None when there is no
    validation part), the names of the features, the name of the target and, for a classification, its classes."""

    name: str
    X_train: object
    y_train: object
    X_val: object
    y_val: object
    X_test: object
    y_test: object
    features: list
    target: str
    classes: list = field(default_factory=list)

    def sizes(self) -> tuple:
        """``(n_train, n_val, n_test)`` (``n_val`` is 0 without a validation part)."""
        return (len(self.X_train), 0 if self.X_val is None else len(self.X_val), len(self.X_test))


_TABULAR: dict = {}


def california(seed: int = 0) -> TabularSplit:
    """California Housing (``wb.datasets.load_california``), features rounded to float32 (module docstring): 20 % of
    the districts for the test (``train_test_split(..., test_size=0.2, random_state=seed)``), then 25 % of the rest
    for the validation (same seed): 12 384, 4 128 and 4 128 districts. Target ``MedHouseVal`` (hundreds of thousands
    of dollars). The same objects are returned again within a session (do not modify them in place)."""
    from sklearn.model_selection import train_test_split

    key = ("california", seed)
    if key not in _TABULAR:
        X, y = datasets.load_california(return_X_y=True)
        X = X.astype("float32").astype("float64")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)
        _TABULAR[key] = TabularSplit("california", X_tr, y_tr, X_va, y_va, X_te, y_te, list(X.columns),
                                     "MedHouseVal")
    return _TABULAR[key]


PENGUIN_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def penguins(seed: int = 0) -> TabularSplit:
    """Palmer Penguins without missing values (333 penguins), the four measures ``PENGUIN_FEATURES`` (float64) and the
    species: 25 % for the test, stratified (``train_test_split(..., test_size=0.25, stratify=species,
    random_state=seed)``): 249 and 84 penguins, no validation part. ``classes``: the species in sorted order
    (Adelie, Chinstrap, Gentoo), the order of ``predict_proba``."""
    from sklearn.model_selection import train_test_split

    key = ("penguins", seed)
    if key not in _TABULAR:
        df = datasets.load_penguins(dropna=True)
        X, y = df[PENGUIN_FEATURES].astype("float64"), df["species"].astype(str)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        _TABULAR[key] = TabularSplit("penguins", X_tr, y_tr, None, None, X_te, y_te, list(PENGUIN_FEATURES),
                                     "species", sorted(y.unique()))
    return _TABULAR[key]


def adult(seed: int = 2026, val_seed: int = 0) -> TabularSplit:
    """Adult (``wb.datasets.load_adult``): the 14 features as loaded (NaN kept) and the target ``y`` = 1 for
    ``">50K"`` (int64). The test part is the one of the mini-project MP3 (20 %, stratified, ``random_state=seed``:
    ``income.make_test_indices``, 9 769 people); the other rows are split again, 25 % for the validation (stratified,
    ``random_state=val_seed``): 29 304, 9 769 and 9 769 people. ``classes``: ``[0, 1]``."""
    from sklearn.model_selection import train_test_split

    key = ("adult", seed, val_seed)
    if key not in _TABULAR:
        df = datasets.load_adult()
        y = (df["class"] == ">50K").astype("int64")
        X = df.drop(columns="class")
        rest, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        rest, test = np.sort(rest), np.sort(test)
        train, val = train_test_split(rest, test_size=0.25, stratify=y.iloc[rest], random_state=val_seed)
        train, val = np.sort(train), np.sort(val)
        _TABULAR[key] = TabularSplit("adult", X.iloc[train], y.iloc[train], X.iloc[val], y.iloc[val], X.iloc[test],
                                     y.iloc[test], list(X.columns), "class", [0, 1])
    return _TABULAR[key]


# ---------------------------------------------------------------------------------------------------------------------
# Cache of the trained networks (atomic, outside the repository, with the provenance in the key)
# ---------------------------------------------------------------------------------------------------------------------
def cache_folder() -> Path:
    """``wb.datasets.cache_dir() / "explain"`` (never versioned; on Colab, erased with the session)."""
    path = datasets.cache_dir() / "explain"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical(key: dict) -> str:
    return json.dumps(key, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def cache_file(key: dict, folder: str | Path | None = None) -> Path:
    """The ``.pt`` file of a key: its kind, its source and 16 hexadecimal digits of the SHA-256 of the whole key."""
    digest = hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()[:16]
    base = Path(folder) if folder is not None else cache_folder()
    return base / f"{key.get('kind', 'model')}-{key.get('source', 'reference')}-{digest}.pt"


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


def _versions() -> dict:
    import torch
    import torchvision

    return {"torch": ".".join(torch.__version__.split("+")[0].split(".")[:2]),
            "torchvision": ".".join(torchvision.__version__.split("+")[0].split(".")[:2])}


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _quiet_build(make):
    """``make()`` inside ``torch.random.fork_rng()``: building a module to load weights into draws an initialisation
    from the global generator, which must not move (the cells after it keep their random draws)."""
    torch = _torch()
    with torch.random.fork_rng():
        return make()


def _mode(fast: bool) -> str:
    return "fast" if fast else "full"


# ---------------------------------------------------------------------------------------------------------------------
# The small CNN of Ex B6.19
# ---------------------------------------------------------------------------------------------------------------------
def _cnn_class():
    torch = _torch()
    nn = torch.nn

    class SmallCNN(nn.Module):
        """Three blocks (two 3 x 3 convolutions with batchnorm and ReLU, then a 2 x 2 max pooling), a global average
        pooling and a linear head of 10 logits. It takes images in [0, 1], (N, 3, 32, 32): the normalisation by the
        statistics of CIFAR-10 is its first operation (so a black image is the zero tensor). ``features[k]`` are its
        layers in order; ``blocks`` the three blocks (``features[0:7]``, ``[7:14]``, ``[14:21]``)."""

        def __init__(self, widths=(16, 32, 64), num_classes: int = 10):
            super().__init__()
            layers, c = [], 3
            for w in widths:
                layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
                           nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2)]
                c = w
            self.features = nn.Sequential(*layers)
            self.fc = nn.Linear(c, num_classes)
            self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

        @property
        def blocks(self):
            return [self.features[i:i + 7] for i in range(0, len(self.features), 7)]

        def forward(self, x):
            h = self.features((x - self.mean) / self.std)
            return self.fc(h.mean(dim=(2, 3)))

    return SmallCNN


def pixels(images):
    """uint8 images (N, 32, 32, 3) -> a float32 tensor (N, 3, 32, 32) in [0, 1] (the input of the small CNN)."""
    torch = _torch()
    if torch.is_tensor(images):
        return images.float()
    arr = np.asarray(images)
    if arr.dtype == np.uint8:
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float() / 255.0
    return torch.as_tensor(arr, dtype=torch.float32)


def cnn_key(fast: bool = True) -> dict:
    """The cache key of the reference CNN of the mode."""
    s = CNN_SETTINGS[_mode(fast)]
    return {"version": CACHE_VERSION, "kind": "cifar_cnn", "source": "reference", "fast": bool(fast),
            "settings": {k: list(v) if isinstance(v, tuple) else v for k, v in s.items()},
            "data": {"dataset": "cifar10", "per_class": s["per_class"], "subset_seed": datasets.CIFAR10_SUBSET_SEED,
                     "eval_per_class": EVAL_PER_CLASS}, "versions": _versions()}


def train_cnn(settings: dict, images, labels, verbose: bool = False):
    """Train a ``SmallCNN(settings["widths"])`` on uint8 images (N, 32, 32, 3): Adam with a one-cycle learning rate
    (peak ``max_lr``), mini-batches of ``batch_size``, random horizontal flips, ``epochs`` passes, seed ``seed``
    (inside ``torch.random.fork_rng``: the global generator does not move). Returns the model in eval() mode."""
    torch = _torch()
    F = torch.nn.functional
    x, y = pixels(images), torch.as_tensor(np.asarray(labels), dtype=torch.long)
    with torch.random.fork_rng():
        torch.manual_seed(settings["seed"])
        model = _cnn_class()(tuple(settings["widths"]))
        opt = torch.optim.Adam(model.parameters(), lr=settings["max_lr"] / 10)
        steps = settings["epochs"] * math.ceil(len(x) / settings["batch_size"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, settings["max_lr"], total_steps=steps)
        gen = torch.Generator().manual_seed(settings["seed"])
        for epoch in range(settings["epochs"]):
            model.train()
            perm = torch.randperm(len(x), generator=gen)
            total = 0.0
            for i in range(0, len(x), settings["batch_size"]):
                idx = perm[i:i + settings["batch_size"]]
                xb = x[idx]
                flip = torch.rand(len(idx), generator=gen) < 0.5
                xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
                loss = F.cross_entropy(model(xb), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            if verbose:
                print(f"   epoch {epoch + 1} : loss {total / len(x):.3f}", flush=True)
    return model.eval()


def accuracy(model, images, labels, info: dict | None = None, batch_size: int = 128) -> float:
    """Accuracy of a model of this module on uint8 images (``info`` of :func:`b1_resnet` for a ResNet, None for the
    small CNN), in eval() mode, without gradient."""
    torch = _torch()
    model.eval()
    correct = 0
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = model_input(images[i:i + batch_size], info)
            correct += int((model(x).argmax(1).numpy() == np.asarray(labels[i:i + batch_size])).sum())
    return correct / len(images)


def cifar_cnn(fast: bool = True, folder: str | Path | None = None, verbose: bool = True):
    """``(model, info)``: the small CNN of Ex B6.19 for the mode (``CNN_SETTINGS``), trained on the nested subset of
    ``per_class`` CIFAR-10 images per class the first time (about 35 s on two CPU cores in fast mode), then read back
    from its file. The model is frozen and in eval() mode; ``info``: ``source`` ("reference"), ``settings``,
    ``test_accuracy`` (on 50 test images per class, measured once), ``seconds`` (of the training), ``path``."""
    SmallCNN = _cnn_class()
    key = cnn_key(fast)
    path = cache_file(key, folder)
    payload = _read(path, key)
    if payload is None:
        s = CNN_SETTINGS[_mode(fast)]
        if verbose:
            print(f"ℹ️ entraînement du CNN de référence ({s['per_class']} images par classe, {s['epochs']} epochs, "
                  "une seule fois)", flush=True)
        X, y = datasets.load_cifar10_subset(s["per_class"])
        start = time.perf_counter()
        model = train_cnn(s, X, y)
        seconds = time.perf_counter() - start
        X_te, y_te = datasets.load_cifar10_subset(EVAL_PER_CLASS, "test")
        extra = {"test_accuracy": accuracy(model, X_te, y_te), "seconds": seconds}
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _atomic_save({"key": _canonical(key), "state_dict": state, "extra": extra, "__complete__": True}, path)
        payload = _read(path, key)
    elif verbose:
        print(f"ℹ️ CNN de référence relu ({path.name})")
    model = _quiet_build(lambda: SmallCNN(tuple(CNN_SETTINGS[_mode(fast)]["widths"])))
    model.load_state_dict(payload["state_dict"])
    info = {"source": "reference", "settings": dict(CNN_SETTINGS[_mode(fast)]), "path": path, "size": 32,
            "classes": list(datasets.CIFAR10_CLASSES), **payload["extra"]}
    return _freeze(model), info


# ---------------------------------------------------------------------------------------------------------------------
# The fine-tuned ResNet-18 of chapter B1 (Ex B6.20)
# ---------------------------------------------------------------------------------------------------------------------
def model_input(images, info: dict | None = None):
    """The input tensor of a model of this module: for the small CNN (``info`` None, or of size 32 without ``mean``),
    :func:`pixels`; for a ResNet, ``wb.transfer.to_input(images, info["size"], info["mean"], info["std"])`` (resized,
    bilinear, normalised by the statistics of ImageNet). A float tensor already prepared is returned as it is."""
    torch = _torch()
    if torch.is_tensor(images) and images.dtype != torch.uint8:
        return images
    if info is None or "mean" not in info:
        return pixels(images)
    from wb.transfer import to_input

    return to_input(np.asarray(images), info["size"], info["mean"], info["std"])


def resnet_key(fast: bool = True) -> dict:
    """The cache key of the reference fine-tuned ResNet-18 of the mode."""
    s = RESNET_SETTINGS[_mode(fast)]
    return {"version": CACHE_VERSION, "kind": "b1_resnet", "source": "reference", "fast": bool(fast),
            "weights": "ResNet18_Weights.IMAGENET1K_V1", "recipe": "Ex B1.17 (layer4 + head)", "settings": dict(s),
            "data": {"dataset": "cifar10", "per_class": s["per_class"], "subset_seed": datasets.CIFAR10_SUBSET_SEED,
                     "eval_per_class": EVAL_PER_CLASS}, "versions": _versions()}


def _resnet_info(source: str, size: int, path, extra: dict) -> dict:
    return {"source": source, "size": int(size), "mean": tuple(datasets.IMAGENET_MEAN),
            "std": tuple(datasets.IMAGENET_STD), "classes": list(datasets.CIFAR10_CLASSES), "path": path, **extra}


def train_resnet(settings: dict, images, labels, weights="DEFAULT", verbose: bool = False):
    """The recipe of Ex B1.17: ResNet-18 (``weights``: "DEFAULT" for IMAGENET1K_V1, None for a random start, as the
    tests do), a new head of 10 outputs, only ``layer4`` and the head trainable, AdamW with ``head_lr`` for the head
    and ``head_lr * gamma`` for layer4 (weight decay on the weights only), batchnorms kept in eval() mode,
    mini-batches of ``batch_size`` images resized to ``size`` pixels, ``epochs`` passes, seed ``seed`` (inside
    ``fork_rng``). Returns the model in eval() mode."""
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional
    from torchvision.models import ResNet18_Weights, resnet18

    from wb.transfer import to_input

    w = ResNet18_Weights.IMAGENET1K_V1 if weights == "DEFAULT" else weights
    with torch.random.fork_rng():
        torch.manual_seed(settings["seed"])
        model = resnet18(weights=w)
        model.fc = nn.Linear(model.fc.in_features, 10)
        for p in model.parameters():
            p.requires_grad_(False)
        groups = []
        for module, lr in ((model.fc, settings["head_lr"]), (model.layer4, settings["head_lr"] * settings["gamma"])):
            decay = [p for n, p in module.named_parameters() if p.ndim > 1]
            no_decay = [p for n, p in module.named_parameters() if p.ndim <= 1]
            for p in decay + no_decay:
                p.requires_grad_(True)
            groups += [{"params": decay, "lr": lr, "weight_decay": settings["weight_decay"]},
                       {"params": no_decay, "lr": lr, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups)
        X, y = np.asarray(images), np.asarray(labels)
        for epoch in range(settings["epochs"]):
            model.train()
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
            order = torch.randperm(len(X), generator=torch.Generator().manual_seed(settings["seed"] + epoch)).numpy()
            total = 0.0
            for i in range(0, len(X), settings["batch_size"]):
                idx = order[i:i + settings["batch_size"]]
                loss = F.cross_entropy(model(to_input(X[idx], settings["size"])), torch.from_numpy(y[idx]).long())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if verbose:
                print(f"   epoch {epoch + 1} : loss {total / len(X):.3f}", flush=True)
    return model.eval()


def _resnet_shell(resnet18):
    """A ResNet-18 without weights and with a head of 10 outputs, to load a state dict into."""
    model = resnet18(weights=None)
    model.fc = _torch().nn.Linear(model.fc.in_features, 10)
    return model


def find_b1_folder(start: str | Path | None = None) -> Path | None:
    """The folder of the learner's model published in Ex B1.27 (``model.safetensors`` and ``config.json``), looked
    for next to the notebooks of ``mon_travail``: ``<start>/../b1_transfer/resnet18_cifar10_b1``, then
    ``<repository>/mon_travail/b1_transfer/resnet18_cifar10_b1`` (``start``: the current folder by default). None
    when there is none (it is only read, never written)."""
    from wb.core import repo_root

    start = Path(start) if start is not None else Path.cwd()
    candidates = [start.parent / "b1_transfer" / B1_FOLDER_NAME, start / B1_FOLDER_NAME]
    try:
        candidates.append(Path(repo_root()) / "mon_travail" / "b1_transfer" / B1_FOLDER_NAME)
    except Exception:  # noqa: BLE001 - no repository: only the folders near `start`
        pass
    for folder in candidates:
        if (folder / "model.safetensors").is_file() and (folder / "config.json").is_file():
            return folder.resolve()
    return None


def load_b1_folder(folder: str | Path):
    """``(model, info)``: the ResNet-18 of a folder published by Ex B1.27, rebuilt from its ``config.json``
    (architecture resnet18, ``num_classes``, ``image_size``, ``mean``, ``std``) and its ``model.safetensors``, frozen
    and in eval() mode; ``info["source"]`` is "learner". Raises ValueError when the folder does not hold such a
    model."""
    from safetensors.torch import load_file
    from torchvision.models import resnet18

    folder = Path(folder)
    config_path, weights_path = folder / "config.json", folder / "model.safetensors"
    if not (config_path.is_file() and weights_path.is_file()):
        raise ValueError(f"{folder} does not hold config.json and model.safetensors (Ex B1.27)")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architecture") != "resnet18" or int(config.get("num_classes", 0)) != 10:
        raise ValueError(f"{config_path}: a resnet18 with 10 outputs is expected, got "
                         f"{config.get('architecture')!r} with {config.get('num_classes')!r} outputs")
    model = _quiet_build(lambda: _resnet_shell(resnet18))
    model.load_state_dict(load_file(str(weights_path)))
    info = _resnet_info("learner", config.get("image_size", 96), folder.resolve(),
                        {"test_accuracy": config.get("test_accuracy"), "settings": {}})
    info["mean"] = tuple(config.get("mean", datasets.IMAGENET_MEAN))
    info["std"] = tuple(config.get("std", datasets.IMAGENET_STD))
    if config.get("class_names"):
        info["classes"] = list(config["class_names"])
    return _freeze(model), info


def reference_resnet(fast: bool = True, folder: str | Path | None = None, verbose: bool = True):
    """``(model, info)``: ResNet-18 fine-tuned with the recipe of Ex B1.17 (:func:`train_resnet`,
    ``RESNET_SETTINGS``: 100 images per class at 96 pixels, 3 epochs, about 35 s in fast mode) the first time, then
    read back from its file (source "reference"). ``info``: ``size``, ``mean``, ``std`` (ImageNet),
    ``test_accuracy`` (50 test images per class), ``seconds``, ``settings``, ``path``."""
    from torchvision.models import resnet18

    key = resnet_key(fast)
    path = cache_file(key, folder)
    payload = _read(path, key)
    s = RESNET_SETTINGS[_mode(fast)]
    if payload is None:
        if verbose:
            print(f"ℹ️ fine-tuning du ResNet-18 de référence (recette de B1.17 : layer4 et tête, {s['per_class']} "
                  f"images par classe à {s['size']} pixels, {s['epochs']} epochs, une seule fois)", flush=True)
        X, y = datasets.load_cifar10_subset(s["per_class"])
        start = time.perf_counter()
        model = train_resnet(s, X, y)
        seconds = time.perf_counter() - start
        X_te, y_te = datasets.load_cifar10_subset(EVAL_PER_CLASS, "test")
        extra = {"test_accuracy": accuracy(model, X_te, y_te, _resnet_info("reference", s["size"], None, {})),
                 "seconds": seconds}
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _atomic_save({"key": _canonical(key), "state_dict": state, "extra": extra, "__complete__": True}, path)
        payload = _read(path, key)
    elif verbose:
        print(f"ℹ️ ResNet-18 de référence relu ({path.name})")
    model = _quiet_build(lambda: _resnet_shell(resnet18))
    model.load_state_dict(payload["state_dict"])
    return _freeze(model), _resnet_info("reference", s["size"], path, {"settings": dict(s), **payload["extra"]})


def b1_resnet(fast: bool = True, folder: str | Path | None = None, verbose: bool = True):
    """``(model, info)`` of Ex B6.20: the learner's ResNet of Ex B1.27 when ``folder`` is given and readable
    (:func:`load_b1_folder`, source "learner"), else the reference model (:func:`reference_resnet`), with a line that
    says which one is used (the explicit fallback)."""
    if folder is not None:
        try:
            model, info = load_b1_folder(folder)
        except Exception as exc:  # noqa: BLE001 - a broken copy of the learner: say it, then fall back
            if verbose:
                print(f"ℹ️ ton modèle du B1 ({folder}) ne se charge pas ({type(exc).__name__} : {str(exc)[:200]}) : "
                      "le ResNet-18 de référence le remplace.")
        else:
            if verbose:
                print(f"ℹ️ ton ResNet du B1 (Ex B1.27), lu dans {info['path']} : {info['size']} pixels.")
            return model, info
    elif verbose:
        print("ℹ️ pas de modèle publié par Ex B1.27 : le ResNet-18 de référence (recette de B1.17) le remplace.")
    return reference_resnet(fast, verbose=verbose)


# ---------------------------------------------------------------------------------------------------------------------
# The reference extractors (hooks and autograd), for the checks of the notebook
# ---------------------------------------------------------------------------------------------------------------------
def _targets(target, n: int):
    torch = _torch()
    t = torch.as_tensor(np.asarray(target)).long().reshape(-1)
    if t.numel() == 1:
        t = t.expand(n)
    if t.numel() != n:
        raise ValueError(f"one target class per image is expected ({n}), got {t.numel()}")
    return t


def activations_and_gradients(model, layer, x, target):
    """``(A, G, logits)``: the output ``A`` of ``layer`` (a module of ``model``) for the input tensor ``x``, the
    gradient ``G`` of the sum over the images of the logit of ``target`` (one class per image, or one for all) with
    respect to ``A``, and the logits, as float64 NumPy arrays ((N, C, H, W) twice, then (N, K)). A forward hook
    keeps ``A`` (it returns nothing, so the output is unchanged) and is removed afterwards; the gradient comes from
    ``torch.autograd.grad``, so no parameter receives a ``.grad``; on a frozen model the input is made to require
    grad. ``model`` must be in eval() mode."""
    torch = _torch()
    x = torch.as_tensor(x, dtype=torch.float32).detach().clone().requires_grad_(True)
    t = _targets(target, len(x))
    store = {}

    def keep(module, inputs, output):
        store["A"] = output

    handle = layer.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            logits = model(x)
            if "A" not in store:
                raise ValueError("the layer was not used by the forward pass of the model")
            score = logits.gather(1, t[:, None]).sum()
            (G,) = torch.autograd.grad(score, store["A"])
    finally:
        handle.remove()
    A = store["A"].detach()
    return A.double().numpy(), G.detach().double().numpy(), logits.detach().double().numpy()


HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_HOOK_COMPANIONS = ("_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks_with_kwargs")


def leftover_hooks(model) -> dict:
    """``{module name: [kinds]}``: the hooks still registered on ``model`` or one of its submodules, by kind
    (``forward_hooks``, ``forward_pre_hooks``, ``backward_hooks``, ``backward_pre_hooks``); ``""`` names ``model``
    itself. Empty when no hook is left. (A hook on a tensor dies with the tensor: it is not looked for.)"""
    out = {}
    for name, module in model.named_modules():
        kinds = [kind.strip("_") for kind in HOOK_DICTS if getattr(module, kind, None)]
        if kinds:
            out[name] = kinds
    return out


def clear_hooks(model) -> dict:
    """Remove every module hook left on ``model`` and its submodules (forward, forward pre, backward, backward pre),
    as ``handle.remove()`` would; returns what was there (:func:`leftover_hooks`, before)."""
    left = leftover_hooks(model)
    for module in model.modules():
        for kind in HOOK_DICTS + _HOOK_COMPANIONS:
            hooks = getattr(module, kind, None)
            if hooks:
                hooks.clear()
        if getattr(module, "_is_full_backward_hook", None) is not None:
            module._is_full_backward_hook = None
    return left


def input_gradients(model, x, target):
    """The gradient of the logit of ``target`` (one class per image, or one for all) with respect to the input
    tensor ``x``, as a float64 NumPy array of the shape of ``x`` (autograd, no parameter ``.grad``)."""
    torch = _torch()
    x = torch.as_tensor(x, dtype=torch.float32).detach().clone().requires_grad_(True)
    t = _targets(target, len(x))
    with torch.enable_grad():
        score = model(x).gather(1, t[:, None]).sum()
        (g,) = torch.autograd.grad(score, x)
    return g.double().numpy()


def integrated_gradients(model, x, target, baseline=None, n_steps: int = 50, batch_size: int = 128):
    """Integrated gradients of the logit of ``target`` for each image of the input tensor ``x`` (N, C, H, W), with the
    trapezoid rule of ``explain.integrated_gradients`` (``n_steps`` intervals, ``n_steps + 1`` points, the ends
    weighted by one half); ``baseline``: a tensor of the shape of ``x`` (zeros if None). float64 NumPy (N, C, H, W)."""
    torch = _torch()
    x = torch.as_tensor(x, dtype=torch.float32)
    ref = torch.zeros_like(x) if baseline is None else torch.as_tensor(baseline, dtype=torch.float32)
    if ref.shape != x.shape:
        raise ValueError(f"baseline must have the shape of x {tuple(x.shape)}, got {tuple(ref.shape)}")
    t = _targets(target, len(x))
    alphas = torch.linspace(0, 1, n_steps + 1)
    weights = torch.full((n_steps + 1,), 1.0 / n_steps, dtype=torch.float64)
    weights[[0, -1]] /= 2
    out = []
    for i in range(len(x)):
        path = ref[i:i + 1] + alphas.view(-1, 1, 1, 1) * (x[i:i + 1] - ref[i:i + 1])
        grads = np.concatenate([input_gradients(model, path[j:j + batch_size], int(t[i]))
                                for j in range(0, len(path), batch_size)])
        avg = np.tensordot(weights.numpy(), grads, axes=1)
        out.append((x[i] - ref[i]).double().numpy() * avg)
    return np.stack(out)


def grad_cam(model, layer, x, target, size: int | None = None, relu: bool = True):
    """Reference Grad-CAM maps of ``layer`` for the input tensor ``x``: channel weights = spatial mean of ``G``, map =
    ReLU of the weighted sum of ``A`` (``relu=False`` keeps the sign), each map divided by its largest absolute value
    (an all-zero map stays zero), then, when ``size`` is given, resized to ``size`` x ``size`` (bilinear,
    ``align_corners=False``, as ``wb.transfer.to_input`` resizes the images). float64 NumPy (N, H, W)."""
    torch = _torch()
    A, G, _ = activations_and_gradients(model, layer, x, target)
    cam = np.einsum("nc,nchw->nhw", G.mean(axis=(2, 3)), A)
    if relu:
        cam = np.maximum(cam, 0.0)
    peak = np.abs(cam).max(axis=(1, 2), keepdims=True)
    cam = np.divide(cam, peak, out=np.zeros_like(cam), where=peak > 0)
    if size is not None:
        cam = torch.nn.functional.interpolate(torch.from_numpy(cam)[:, None], size=(int(size), int(size)),
                                              mode="bilinear", align_corners=False)[:, 0].numpy()
    return cam


def predict(model, images, info: dict | None = None, batch_size: int = 128) -> np.ndarray:
    """float64 logits (N, K) of uint8 images (or a prepared tensor), batch by batch, without gradient."""
    torch = _torch()
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            out.append(model(model_input(images[i:i + batch_size], info)).double().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------------------------------------------------
# Sanity checks (Ex B6.22): the layers from the output down, re-initialised copies, rank correlations
# ---------------------------------------------------------------------------------------------------------------------
def cascade_layers(model) -> list:
    """The names of the parts of ``model`` that hold weights, from the output to the input (the order of a cascading
    randomisation): for a torchvision ResNet, ``["fc", "layer4.1", "layer4.0", ..., "layer1.0", "conv1"]`` (the basic
    blocks; ``conv1`` is the first convolution, its batchnorm ``bn1`` is left out); for the small CNN of this module,
    ``["fc", "features.17", "features.14", ...]`` (each convolution); for another model, every ``Conv2d`` and
    ``Linear`` module, last defined first."""
    torch = _torch()
    nn = torch.nn
    if all(hasattr(model, name) for name in ("conv1", "layer1", "layer4", "fc")):
        blocks = [f"{stage}.{i}" for stage in ("layer1", "layer2", "layer3", "layer4")
                  for i in range(len(getattr(model, stage)))]
        return ["fc"] + blocks[::-1] + ["conv1"]
    return [name for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))][::-1]


def randomize_layers(model, names, seed: int = 0, include_batchnorm: bool = False):
    """A frozen copy of ``model`` (eval() mode) in which every ``Conv2d`` and ``Linear`` inside the parts ``names``
    gets new weights from its own ``reset_parameters()`` (PyTorch's default initialisation), drawn after
    ``torch.manual_seed(seed)`` inside ``torch.random.fork_rng()`` (the global generator does not move); with
    ``include_batchnorm=True``, the batchnorms of these parts are reset too (weight 1, bias 0, running statistics 0
    and 1). ``model`` itself is not changed."""
    torch = _torch()
    nn = torch.nn
    out = copy.deepcopy(model)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for name in names:
            part = out.get_submodule(name)
            for m in part.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)) or (include_batchnorm and
                                                              isinstance(m, nn.modules.batchnorm._BatchNorm)):
                    m.reset_parameters()
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.reset_running_stats()
    return _freeze(out)


def spearman_per_image(maps_a, maps_b, absolute: bool = False) -> np.ndarray:
    """The rank correlation of Spearman (``scipy.stats.spearmanr``) between the two maps of each image, flattened:
    float64 (N,); ``absolute=True`` compares the absolute values (as Adebayo et al. for the gradients). A constant map
    gives nan."""
    from scipy.stats import spearmanr

    a, b = np.asarray(maps_a, dtype=float), np.asarray(maps_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"the two sets of maps must have the same shape, got {a.shape} and {b.shape}")
    if absolute:
        a, b = np.abs(a), np.abs(b)
    out = []
    for u, v in zip(a.reshape(len(a), -1), b.reshape(len(b), -1)):
        if np.ptp(u) == 0 or np.ptp(v) == 0:
            out.append(np.nan)
        else:
            out.append(float(spearmanr(u, v).statistic))
    return np.array(out)
