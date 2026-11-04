"""Code cells of the chapter B7 notebook, part A (the project: B7.9 to B7.15) and the start of part B (🐛 B7.16),
shared by build_chB7.py (through chB7_parts.py) and history_chB7.py (used by Claude).

Each constant is the source of one cell: TOOLS_* and GIVEN_* (both notebooks), TODO_* (exercise notebook), SOLUTION_*
(solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions). The files of the
learner's package are strings defined in the TODO cells (``DATA_PY = r'''...'''``) and written into PROJECT by the
check cells (``wbp.write_files``); the solutions notebook defines the same strings from the reference package
``solutions/b7_cifarprod`` (``reference_file``), so that the reference is written once, in that package.
Placeholders «NAME» are filled by chB7_parts.py.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "solutions" / "b7_cifarprod"


def reference_file(rel: str) -> str:
    """The text of a file of the reference package (``solutions/b7_cifarprod/<rel>``)."""
    text = (REFERENCE / rel).read_text(encoding="utf-8")
    if "'''" in text or text.endswith("\\"):
        raise ValueError(f"{rel}: a file put in a cell as r'''...''' cannot contain ''' nor end with a backslash")
    return text


def file_cell(name: str, text: str, comment: str = "") -> str:
    """``NAME = r'''<text>'''`` (a file of the package defined in a cell), with a comment line before it."""
    if "'''" in text:
        raise ValueError(f"{name}: ''' inside the text of a file")
    return (f"# {comment}\n" if comment else "") + f"{name} = r'''{text}'''"


# =====================================================================================================================
# Part A · the project
# =====================================================================================================================
TOOLS_A = r'''# Tools of part A: the folder of your project, the data of the chapter, the commands; nothing here is your package
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from wb import prod as wbp
from wb.nbtools import StopCheck, checked, fr, refuses, run_pytest_group, same, verdict, written

PROJECT = wbp.project_dir("«KIND»")             # the folder of your project
DATA_FILE = wbp.write_subset(PROJECT / "data")  # the 2 000 + 500 images of the chapter (.npz, 7.7 MB, not for git)
SUBSET = wbp.cifar_subset()                     # the same arrays, for the checks
CLASSES = wbp.CLASSES
GITIGNORE = "data/\nruns/\n__pycache__/\n*.egg-info/\n.pytest_cache/\n"
DEFAULTS = {"seed": 0, "epochs": 1, "lr": 0.05, "batch_size": 128, "n_train": 2000, "arch": "cnn", "widths": [16, 32],
            "data": "data/cifar10_2000.npz", "deterministic": True, "num_workers": 0}   # TrainConfig().to_dict()


def scratch(prefix):
    """A new temporary folder (outside the project): the checks write their test files there."""
    return Path(tempfile.mkdtemp(prefix=f"wb_b7_{prefix}_"))


def run_line(ex_id, result, ok, success, failure):
    """One verdict about a command of your package: what was run, then the end of stderr when it went wrong."""
    shown = result.command().replace(f"python -m {wbp.PACKAGE}", "cifarprod")
    if result.timed_out:
        ok, failure = False, f"délai dépassé ({result.seconds:.0f} s) : {failure}"
    verdict(ex_id, ok, f"{shown} → {success}", f"{shown} → {failure} (code {result.returncode} ; stderr : "
                                                f"{result.tail(2)[:400]})")
    return ok


print(wbp.project_line(PROJECT, "«KIND»"))
print(f"ℹ️ données : data/{DATA_FILE.name} ({len(SUBSET['x_train'])} images d'entraînement, {len(SUBSET['x_val'])} de "
      f"validation, 200 et 50 par classe ; empreinte {wbp.SUBSET_FINGERPRINT[:12]}…)")'''

# --- 📦 B7.9 ---------------------------------------------------------------------------------------------------------
SKEL_INIT = '''"""cifarprod: a CIFAR-10 classifier as an installable package (bonus chapter B7)."""

__version__ = "0.1.0"
'''

SKEL_DATA = '''"""Data of cifarprod: the CIFAR-10 subset file, its schema, its fingerprint, the normalization and the loaders."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_npz(path, part="train"):
    """(X, y) of one part ("train" or "val") of a .npz file with the arrays x_train, y_train, x_val, y_val: X uint8
    (N, 32, 32, 3), y int64 (N,). The schema is checked: ValueError for images that are not uint8 (N, 32, 32, 3), for
    labels that are not N integers in 0..9, or for another part; FileNotFoundError if the file does not exist."""
    raise NotImplementedError("load_npz")


def fingerprint(*arrays):
    """SHA-256 (64 hexadecimal characters) of the dtype, the shape and the bytes of each array, in this order (207.4)."""
    raise NotImplementedError("fingerprint")


def channel_stats(images):
    """(mean, std): three Python floats each, the mean and the standard deviation (divided by n) of the pixels of each
    channel on the scale [0, 1], over all the uint8 images (N, H, W, 3) given."""
    raise NotImplementedError("channel_stats")


def preprocess(images, mean, std):
    """uint8 RGB images (N, 32, 32, 3), or one image (32, 32, 3), as a float32 tensor (N, 3, 32, 32) (N = 1 for one
    image), each channel c equal to (pixel / 255 - mean[c]) / std[c]. ValueError for an array that is not uint8."""
    raise NotImplementedError("preprocess")


def make_loader(X, y, mean, std, batch_size=128, shuffle=False, seed=0, num_workers=0):
    """A DataLoader of (preprocess(X), y) mini-batches, float32 (B, 3, 32, 32) and int64 (B,); when shuffle, the order
    comes from its own torch.Generator().manual_seed(seed)."""
    raise NotImplementedError("make_loader")


def synthetic(n, seed=0):
    """(X, y): n random uint8 images (n, 32, 32, 3) drawn by np.random.default_rng(seed), and the labels i % 10 (for the
    tests and the continuous integration, which have no CIFAR-10)."""
    raise NotImplementedError("synthetic")
'''

SKEL_MODEL = '''"""The models of cifarprod: "cnn" and "vgg", the CNNs of the MP5, rebuilt from a few settings."""

from __future__ import annotations

import torch.nn as nn


def build_model(arch="cnn", widths=(12, 24), n_classes=10):
    """The network `arch` with one stage per width, from float32 (N, 3, 32, 32) to the logits (N, n_classes).
    - "cnn": for each width w, Conv2d(c, w, 3, padding=1), ReLU, MaxPool2d(2); then Flatten and Linear;
    - "vgg": for each width w, twice Conv2d(c, w, 3, padding=1, bias=False), BatchNorm2d(w), ReLU; a MaxPool2d(2)
      after every stage but the last; then AdaptiveAvgPool2d(1), Flatten, Linear(widths[-1], n_classes).
    ValueError for another arch or for empty widths."""
    raise NotImplementedError("build_model")


def n_parameters(model):
    """The number of trainable parameters of a model."""
    raise NotImplementedError("n_parameters")
'''

SKEL_TRAIN = '''"""The training of cifarprod: the seeds, one evaluation, and train(), which returns the model with its normalization."""

from __future__ import annotations

import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import channel_stats, make_loader
from .model import build_model

log = logging.getLogger(__name__)


def seed_everything(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch, switch the deterministic algorithms on or off, and return a generator seeded
    with `seed` for the DataLoader (207.4)."""
    raise NotImplementedError("seed_everything")


def evaluate(model, loader):
    """(mean cross-entropy, accuracy) of the model on all the examples of the loader, in eval() mode and without a
    graph; the mode of the model is restored afterwards."""
    raise NotImplementedError("evaluate")


def train(X, y, X_val, y_val, *, arch="cnn", widths=(16, 32), epochs=1, lr=0.05, batch_size=128, seed=0,
          deterministic=True, num_workers=0):
    """Train build_model(arch, widths) on the uint8 images X (N, 32, 32, 3) and their labels y: seed_everything(seed,
    deterministic) first, the normalization learned on X only, SGD(lr, momentum=0.9), the cross-entropy, the
    mini-batches of make_loader(..., shuffle=True, seed=seed). No print: one INFO line per epoch and one DEBUG line
    per mini-batch through `log`. Returns {"model": the trained model in eval() mode, "mean", "std", "history":
    {"train_loss", "val_loss", "val_acc"} (one float per epoch), "seconds"}."""
    raise NotImplementedError("train")
'''

TODO_9 = "\n\n".join([
    "# Ex B7.9: the files of your package, as strings. The check cell writes them into PROJECT, then imports and runs\n"
    "# them; to change a file, change its string here and run both cells again.\n"
    "PYPROJECT_TOML = ...   # the pyproject.toml of the contract (start from the one of the fiche, 207.1)",
    file_cell("INIT_PY", SKEL_INIT, "src/cifarprod/__init__.py (given)"),
    file_cell("DATA_PY", SKEL_DATA, "src/cifarprod/data.py"),
    file_cell("MODEL_PY", SKEL_MODEL, "src/cifarprod/model.py"),
    file_cell("TRAIN_PY", SKEL_TRAIN, "src/cifarprod/train.py"),
])

SOLUTION_9 = "\n\n".join([
    "# Ex B7.9: the files of the package, as strings (the reference package of the chapter, solutions/b7_cifarprod)",
    file_cell("PYPROJECT_TOML", reference_file("pyproject.toml"), "pyproject.toml"),
    file_cell("INIT_PY", reference_file("src/cifarprod/__init__.py"), "src/cifarprod/__init__.py"),
    file_cell("DATA_PY", reference_file("src/cifarprod/data.py"), "src/cifarprod/data.py"),
    file_cell("MODEL_PY", reference_file("src/cifarprod/model.py"), "src/cifarprod/model.py"),
    file_cell("TRAIN_PY", reference_file("src/cifarprod/train.py"), "src/cifarprod/train.py"),
])

CHECK_9 = r'''FILES_9 = {"pyproject.toml": PYPROJECT_TOML, ".gitignore": GITIGNORE, "src/cifarprod/__init__.py": INIT_PY,
           "src/cifarprod/data.py": DATA_PY, "src/cifarprod/model.py": MODEL_PY, "src/cifarprod/train.py": TRAIN_PY}
ok_9 = {}
with checked("B7.9"):
    wbp.write_files(PROJECT, FILES_9)
    missing_9 = [rel for rel, text in FILES_9.items() if text is ...]
    print("ℹ️ fichiers écrits dans le projet : " + ", ".join(r for r, t in FILES_9.items() if t is not ...)
          + (f" ; pas encore écrits : {', '.join(missing_9)}" if missing_9 else ""))
    pkg_9 = wbp.import_package(PROJECT)
    ok_9["import"] = getattr(pkg_9, "__version__", None) == "0.1.0"
    verdict("B7.9", ok_9["import"], "import cifarprod depuis src/ : version 0.1.0.",
            f"cifarprod.__version__ vaut {getattr(pkg_9, '__version__', None)!r}, attendu '0.1.0'.")

with checked("B7.9"):                                    # data.py
    data_9 = wbp.import_module(PROJECT, "data")
    X_9, y_9 = data_9.load_npz(DATA_FILE, "train")
    Xv_9, yv_9 = data_9.load_npz(DATA_FILE, "val")
    got_9 = [(np.asarray(a).dtype, np.asarray(a).shape) for a in (X_9, y_9, Xv_9, yv_9)]
    ok_9["load"] = (isinstance(X_9, np.ndarray) and X_9.dtype == np.uint8 and np.asarray(y_9).dtype == np.int64
                    and np.array_equal(X_9, SUBSET["x_train"]) and np.array_equal(y_9, SUBSET["y_train"])
                    and np.array_equal(Xv_9, SUBSET["x_val"]) and np.array_equal(yv_9, SUBSET["y_val"]))
    verdict("B7.9", ok_9["load"], "load_npz : les 2 000 images d'entraînement et les 500 de validation, uint8 et int64.",
            f"load_npz ne rend pas les tableaux du fichier en uint8 et int64 : reçu (dtype, forme) {got_9}.")
    bad_9 = scratch("bad")
    zeros_9 = np.zeros((4, 32, 32, 3), np.uint8)
    for name_9, (xb_9, yb_9) in {"des images float32": (zeros_9.astype(np.float32), np.arange(4)),
                                 "des images 32 × 32 × 1": (zeros_9[..., :1], np.arange(4)),
                                 "un label 10": (zeros_9, np.array([0, 1, 2, 10])),
                                 "3 labels pour 4 images": (zeros_9, np.arange(3))}.items():
        np.savez(bad_9 / "bad.npz", x_train=xb_9, y_train=yb_9, x_val=xb_9, y_val=yb_9)
        ok_9[name_9] = refuses("B7.9", f"load_npz d'un fichier avec {name_9}",
                               lambda: data_9.load_npz(bad_9 / "bad.npz", "train"))
    ok_9["part"] = refuses("B7.9", 'load_npz(..., "test")', lambda: data_9.load_npz(DATA_FILE, "test"))
    ok_9["absent"] = refuses("B7.9", "load_npz d'un fichier absent", lambda: data_9.load_npz(bad_9 / "absent.npz"),
                             error=FileNotFoundError)
    fp_9 = data_9.fingerprint(SUBSET["x_train"][:5], SUBSET["y_train"][:5])
    ok_9["fingerprint"] = fp_9 == wbp.array_fingerprint(SUBSET["x_train"][:5], SUBSET["y_train"][:5])
    verdict("B7.9", ok_9["fingerprint"], f"fingerprint de 5 images et de leurs labels : {str(fp_9)[:12]}…, comme "
            "hashlib sur le type, la forme et les octets.", f"fingerprint de 5 images et de leurs labels : "
            f"{str(fp_9)[:16]!r}…, attendu un autre SHA-256 (le type, la forme, puis les octets de chaque tableau).")
    mean_9, std_9 = data_9.channel_stats(SUBSET["x_train"])
    want_9 = (SUBSET["x_train"] / 255.0).mean(axis=(0, 1, 2)), (SUBSET["x_train"] / 255.0).std(axis=(0, 1, 2))
    ok_9["stats"] = (len(mean_9) == len(std_9) == 3 and all(type(v) is float for v in [*mean_9, *std_9])
                     and np.allclose(mean_9, want_9[0], atol=1e-10) and np.allclose(std_9, want_9[1], atol=1e-10))
    verdict("B7.9", ok_9["stats"], f"channel_stats : moyennes {np.round(mean_9, 4).tolist()}, écarts-types "
            f"{np.round(std_9, 4).tolist()} (trois floats Python chacun).",
            f"channel_stats rend {mean_9!r}, {std_9!r} : attendu trois floats Python chacun, moyenne et écart-type "
            "(divisé par n) des pixels de chaque canal dans [0, 1].")
    four_9 = SUBSET["x_val"][:4]
    want_x_9 = ((four_9 / 255.0 - np.array(want_9[0])) / np.array(want_9[1])).transpose(0, 3, 1, 2)
    x_9 = data_9.preprocess(four_9.copy(), tuple(want_9[0]), tuple(want_9[1]))
    ok_9["preprocess"] = (isinstance(x_9, torch.Tensor) and x_9.dtype == torch.float32
                          and same("B7.9", "preprocess de 4 images de validation", x_9, want_x_9, rtol=0, atol=1e-5))
    if not (isinstance(x_9, torch.Tensor) and x_9.dtype == torch.float32):
        verdict("B7.9", False, "", f"preprocess doit rendre un tenseur float32, reçu {type(x_9).__name__} "
                f"{getattr(x_9, 'dtype', '')}.")
    one_9 = data_9.preprocess(four_9[0], tuple(want_9[0]), tuple(want_9[1]))
    ok_9["one"] = tuple(getattr(one_9, "shape", ())) == (1, 3, 32, 32)
    verdict("B7.9", ok_9["one"], "preprocess d'une seule image (32, 32, 3) : (1, 3, 32, 32).",
            f"preprocess d'une seule image (32, 32, 3) : attendu (1, 3, 32, 32), reçu {tuple(getattr(one_9, 'shape', ()))}.")
    ok_9["float"] = refuses("B7.9", "preprocess d'images en float", lambda: data_9.preprocess(four_9 / 255.0, *want_9))

    def order_9(seed, global_seed):
        torch.manual_seed(global_seed)                  # the order must not depend on the global generator
        loader = data_9.make_loader(SUBSET["x_val"][:64], np.arange(64), *want_9, batch_size=16, shuffle=True, seed=seed)
        return np.concatenate([np.asarray(t) for _, t in loader])

    with wbp.torch_state_kept():
        plain_9 = data_9.make_loader(SUBSET["x_val"][:64], SUBSET["y_val"][:64], *want_9, batch_size=16)
        xb_9, tb_9 = next(iter(plain_9))
        a_9, b_9, c_9 = order_9(0, 1), order_9(0, 2), order_9(1, 1)
    ok_9["loader"] = (tuple(xb_9.shape) == (16, 3, 32, 32) and xb_9.dtype == torch.float32 and tb_9.dtype == torch.int64
                      and np.allclose(xb_9.numpy(), data_9.preprocess(SUBSET["x_val"][:16], *want_9).numpy())
                      and np.array_equal(tb_9.numpy(), SUBSET["y_val"][:16]))
    verdict("B7.9", ok_9["loader"], "make_loader sans mélange : les 16 premières images prétraitées et leurs labels, "
            "float32 et int64.", f"make_loader sans mélange : attendu le batch (16, 3, 32, 32) float32 des 16 premières "
            f"images prétraitées et leurs labels int64, reçu {tuple(xb_9.shape)} {xb_9.dtype}, {tb_9.dtype}.")
    ok_9["order"] = sorted(a_9) == list(range(64)) and np.array_equal(a_9, b_9) and not np.array_equal(a_9, c_9)
    verdict("B7.9", ok_9["order"], "make_loader mélangé : le même ordre pour la même graine, quel que soit le "
            "générateur global ; un autre pour une autre graine.",
            "make_loader mélangé : attendu le même ordre des 64 images pour seed=0 après torch.manual_seed(1) ou (2), "
            "et un autre pour seed=1 ; " + ("l'ordre suit le générateur global." if not np.array_equal(a_9, b_9)
                                           else "seed ne change pas l'ordre." if np.array_equal(a_9, c_9)
                                           else "des images manquent ou sont répétées."))
    Xs_9, ys_9 = data_9.synthetic(12, seed=3)
    ok_9["synthetic"] = (np.asarray(Xs_9).dtype == np.uint8 and np.asarray(Xs_9).shape == (12, 32, 32, 3)
                         and np.array_equal(Xs_9, np.random.default_rng(3).integers(0, 256, (12, 32, 32, 3), dtype=np.uint8))
                         and np.array_equal(ys_9, np.arange(12) % 10))
    verdict("B7.9", ok_9["synthetic"], "synthetic(12, seed=3) : 12 images uint8 de default_rng(3) et les labels i % 10.",
            "synthetic(12, seed=3) : attendu np.random.default_rng(3).integers(0, 256, (12, 32, 32, 3), "
            "dtype=np.uint8) et les labels np.arange(12) % 10.")

with checked("B7.9"):                                    # model.py
    model_9 = wbp.import_module(PROJECT, "model")
    for arch_9, widths_9 in (("cnn", (12, 24)), ("cnn", (8, 16, 32)), ("vgg", (16, 32, 64)), ("vgg", (8,))):
        mine_9 = model_9.build_model(arch_9, widths_9)
        want_m_9 = wbp.build_reference(arch_9, widths_9)
        shapes_9 = {k: tuple(v.shape) for k, v in mine_9.state_dict().items()}
        ref_shapes_9 = {k: tuple(v.shape) for k, v in want_m_9.state_dict().items()}
        out_9 = mine_9.eval()(torch.zeros(5, 3, 32, 32))
        ok_9[arch_9 + str(widths_9)] = shapes_9 == ref_shapes_9 and tuple(out_9.shape) == (5, 10)
        verdict("B7.9", ok_9[arch_9 + str(widths_9)],
                f"build_model({arch_9!r}, {widths_9}) : {model_9.n_parameters(mine_9):_} paramètres, les mêmes couches "
                "que l'architecture du contrat, sortie (5, 10).".replace("_", " "),
                f"build_model({arch_9!r}, {widths_9}) : attendu les tenseurs {ref_shapes_9} et une sortie (5, 10), "
                f"reçu {shapes_9} et {tuple(out_9.shape)}.")
    n_9 = model_9.n_parameters(wbp.build_reference("cnn", (12, 24)))
    ok_9["n_parameters"] = n_9 == sum(p.numel() for p in wbp.build_reference("cnn", (12, 24)).parameters())
    verdict("B7.9", ok_9["n_parameters"], "n_parameters compte les paramètres entraînables.",
            f"n_parameters du CNN (12, 24) de la fiche : reçu {n_9}, attendu le nombre de valeurs de ses paramètres.")
    ok_9["default"] = ({k: tuple(v.shape) for k, v in model_9.build_model().state_dict().items()}
                       == {k: tuple(v.shape) for k, v in wbp.build_reference("cnn", (12, 24)).state_dict().items()})
    verdict("B7.9", ok_9["default"], "build_model() : le CNN (12, 24) de la fiche.",
            "build_model() sans argument doit construire le CNN (12, 24) de la fiche (arch=\"cnn\", widths=(12, 24)).")
    ok_9["arch"] = refuses("B7.9", 'build_model("resnet")', lambda: model_9.build_model("resnet"))

TRAINED_9 = None
with checked("B7.9"):                                    # train.py, in this notebook
    train_9 = wbp.import_module(PROJECT, "train")
    small_9 = dict(arch="cnn", widths=(8, 16), epochs=2, lr=0.05, batch_size=64, seed=0)
    printed_9 = io.StringIO()
    with wbp.torch_state_kept(), contextlib.redirect_stdout(printed_9):
        r1_9 = train_9.train(SUBSET["x_train"][:256], SUBSET["y_train"][:256], SUBSET["x_val"][:100],
                             SUBSET["y_val"][:100], **small_9)
        r2_9 = train_9.train(SUBSET["x_train"][:256], SUBSET["y_train"][:256], SUBSET["x_val"][:100],
                             SUBSET["y_val"][:100], **{**small_9, "seed": 1})
        r3_9 = train_9.train(SUBSET["x_train"][:256], SUBSET["y_train"][:256], SUBSET["x_val"][:100],
                             SUBSET["y_val"][:100], **small_9)
    keys_9 = {"model", "mean", "std", "history", "seconds"}
    if not (isinstance(r1_9, dict) and keys_9 <= set(r1_9)):
        verdict("B7.9", False, "", f"train doit rendre un dictionnaire avec les clés {sorted(keys_9)}, reçu "
                f"{type(r1_9).__name__} {sorted(r1_9) if isinstance(r1_9, dict) else ''}.")
        raise StopCheck
    h_9 = r1_9["history"]
    ok_9["history"] = (isinstance(h_9, dict) and all(len(h_9.get(k, [])) == 2 for k in ("train_loss", "val_loss", "val_acc"))
                       and all(type(v) is float for k in ("train_loss", "val_loss", "val_acc") for v in h_9[k])
                       and all(0 <= v <= 1 for v in h_9["val_acc"]))
    verdict("B7.9", ok_9["history"], "train sur 256 images, 2 epochs : un historique de deux floats par courbe.",
            f"train sur 256 images, 2 epochs : attendu history = {{train_loss, val_loss, val_acc}}, deux floats "
            f"chacune (val_acc dans [0, 1]), reçu {h_9!r:.200}.")
    stats_256_9 = (SUBSET["x_train"][:256] / 255.0).mean(axis=(0, 1, 2))
    ok_9["mean"] = np.allclose(r1_9["mean"], stats_256_9, atol=1e-10)
    verdict("B7.9", ok_9["mean"], "la normalisation est apprise sur les images d'entraînement reçues.",
            "result['mean'] doit être channel_stats des images d'entraînement reçues par train (ni la validation, ni "
            "les statistiques d'ImageNet).")
    ok_9["eval"] = not r1_9["model"].training
    verdict("B7.9", ok_9["eval"], "le modèle rendu est en mode eval().", "le modèle rendu est encore en mode "
            "entraînement : relis la docstring de train.")
    ok_9["seed"] = wbp.same_weights(r1_9["model"].state_dict(), r3_9["model"].state_dict()) and \
        not wbp.same_weights(r1_9["model"].state_dict(), r2_9["model"].state_dict())
    verdict("B7.9", ok_9["seed"], "deux appels avec seed=0 donnent les mêmes poids, seed=1 d'autres.",
            "deux appels de train avec seed=0 doivent donner exactement les mêmes poids, et seed=1 d'autres poids "
            "(seed_everything au début de train, le générateur de make_loader).")
    ok_9["print"] = printed_9.getvalue() == ""
    verdict("B7.9", ok_9["print"], "train n'écrit rien avec print.",
            f"train écrit avec print ({printed_9.getvalue().strip()[:80]!r}…) : passe par log.info et log.debug.")
    with wbp.torch_state_kept():
        TRAINED_9 = train_9.train(SUBSET["x_train"], SUBSET["y_train"], SUBSET["x_val"], SUBSET["y_val"])
    acc_9, s_9 = TRAINED_9["history"]["val_acc"][-1], TRAINED_9["seconds"]
    ok_9["smoke"] = s_9 < 60 and acc_9 > 0.15
    verdict("B7.9", ok_9["smoke"], f"entraînement fumée (réglages par défaut, 2 000 images, 1 epoch) : val_acc "
            f"{fr(acc_9, 3)} en {fr(s_9, 1)} s.", f"entraînement fumée (réglages par défaut, 2 000 images, 1 epoch) : "
            f"val_acc {fr(acc_9, 3)} en {fr(s_9, 1)} s ; attendu plus de 0,15 (le hasard : 0,10) en moins de 60 s.")

with checked("B7.9"):                                    # pip install -e, in a new environment
    if PYPROJECT_TOML is ...:
        print("⏳ Ex B7.9 : l'installation attend ton pyproject.toml.")
        raise StopCheck
    install_9 = wbp.install_editable(PROJECT)
    try:
        if install_9.ok is None:
            print("⏳ Ex B7.9 : ni pip ni uv dans cet environnement : l'installation éditable n'est pas vérifiée ici.")
            raise StopCheck
        ok_9["install"] = bool(install_9.ok)
        verdict("B7.9", install_9.ok, f"pip install -e . dans un environnement neuf ({install_9.method}, "
                f"{fr(install_9.seconds, 1)} s).", "pip install -e . échoue : " + install_9.log[-600:])
        if not install_9.ok:
            raise StopCheck
        probe_9 = ("import cifarprod, importlib.metadata as m, json; print(json.dumps([cifarprod.__file__, "
                   "m.version('cifarprod'), [e.value for e in m.entry_points(group='console_scripts') if e.name == "
                   "'cifarprod']]))")
        seen_9 = install_9.run(["python", "-c", probe_9], scratch("elsewhere"))
        where_9, version_9, entry_9 = json.loads(seen_9.stdout) if seen_9.ok else (None, None, None)
        ok_9["editable"] = seen_9.ok and Path(where_9).resolve().is_relative_to(PROJECT / "src")
        verdict("B7.9", ok_9["editable"], "depuis un autre dossier, import cifarprod lit ton dossier src/ (une "
                "installation éditable, pas une copie).", f"depuis un autre dossier, import cifarprod : {where_9 or seen_9.tail(1)}.")
        ok_9["metadata"] = version_9 == "0.1.0" and entry_9 == ["cifarprod.cli:main"]
        verdict("B7.9", ok_9["metadata"], "métadonnées installées : version 0.1.0, commande cifarprod → cifarprod.cli:main.",
                f"métadonnées installées : version {version_9!r} (attendu '0.1.0'), point d'entrée {entry_9!r} "
                "(attendu ['cifarprod.cli:main']).")
    finally:
        install_9.cleanup()
if ok_9 and all(ok_9.values()) and {"smoke", "editable", "metadata", "synthetic", "default"} <= set(ok_9):
    print("✅ Ex B7.9 : ton paquet s'installe, s'importe et s'entraîne : le notebook appelle, le paquet fait.")'''

# --- 📦 B7.10 ---------------------------------------------------------------------------------------------------------
CONFIG_V0 = '''"""The configuration of cifarprod, STARTING VERSION (given in Ex B7.10, replaced in Ex B7.11): the settings and their
defaults, read from a YAML file and the overrides "KEY=VALUE", without any check of the types or of the ranges."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 1
    lr: float = 0.05
    batch_size: int = 128
    n_train: int = 2000
    arch: str = "cnn"
    widths: tuple = (16, 32)
    data: str = "data/cifar10_2000.npz"
    deterministic: bool = True
    num_workers: int = 0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["widths"] = list(self.widths)
        return out


def load_config(path=None, overrides=()):
    raw = dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}) if path is not None else {}
    for item in overrides:
        key, sep, value = str(item).partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        raw[key.strip()] = yaml.safe_load(value)
    unknown = set(raw) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return TrainConfig(**raw)
'''

GIVEN_10 = file_cell("CONFIG_V0", CONFIG_V0, "the starting version of src/cifarprod/config.py, written by the check "
                     "cell only if config.py does not exist yet (Ex B7.11 replaces it)")

SKEL_MAIN = '''"""python -m cifarprod ...: the same command line as the `cifarprod` entry point."""

raise NotImplementedError("__main__.py")
'''

SKEL_CLI = '''"""The command line of cifarprod: cifarprod train [--config FILE] [--set KEY=VALUE] [--out DIR] [-v]; cifarprod --version."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import load_config

log = logging.getLogger(__name__)


def build_parser():
    """The parser of the contract: prog "cifarprod", --version, the sub-command train (required) and its options."""
    raise NotImplementedError("build_parser")


def main(argv=None):
    """Parse argv (None: sys.argv), configure the logging once, run the command and return its exit code: 0 success,
    2 invalid configuration (an ERROR line), 1 data that cannot be read or a training that fails (an ERROR line). On
    success, the only line written on stdout is the JSON summary {"status", "val_acc", "weights"}."""
    raise NotImplementedError("main")


if __name__ == "__main__":
    sys.exit(main())
'''

TODO_10 = "\n\n".join([
    "# Ex B7.10: two more files of your package (written by the check cell); config.py comes from the cell above",
    file_cell("MAIN_PY", SKEL_MAIN, "src/cifarprod/__main__.py"),
    file_cell("CLI_PY", SKEL_CLI, "src/cifarprod/cli.py (Ex B7.14 will ask you to complete this same cell)"),
])

SOLUTION_10 = "\n\n".join([
    "# Ex B7.10: the command line of the reference package. Its run journal (the import of .tracking, --runs, the "
    "record\n# of each run) belongs to Ex B7.14: the solutions notebook writes the final cli.py at once, and tracking.py "
    "with it.",
    file_cell("MAIN_PY", reference_file("src/cifarprod/__main__.py"), "src/cifarprod/__main__.py"),
    file_cell("CLI_PY", reference_file("src/cifarprod/cli.py"), "src/cifarprod/cli.py"),
    file_cell("TRACKING_PY", reference_file("src/cifarprod/tracking.py"), "src/cifarprod/tracking.py (Ex B7.14)"),
])

CHECK_10 = r'''ok_10 = {}
with checked("B7.10"):
    wbp.write_file(PROJECT, "src/cifarprod/config.py", CONFIG_V0, if_absent=True)
    files_10 = {"src/cifarprod/__main__.py": MAIN_PY, "src/cifarprod/cli.py": CLI_PY}
    if "TRACKING_PY" in globals() and TRACKING_PY is not ...:
        files_10["src/cifarprod/tracking.py"] = TRACKING_PY          # Ex B7.14, if it is already done
    wbp.write_files(PROJECT, files_10)
    cli_10 = wbp.import_module(PROJECT, "cli")
    cli_10.build_parser()                                 # ⏳ while it raises NotImplementedError
    out_10 = scratch("out")
    r_10 = wbp.cli(PROJECT, "--version")
    ok_10["version"] = run_line("B7.10", r_10, r_10.ok and "0.1.0" in r_10.stdout, "code 0, « cifarprod 0.1.0 ».",
                                "attendu le code 0 et la version 0.1.0 sur stdout")
    r_10 = wbp.cli(PROJECT, "--help")
    ok_10["help"] = run_line("B7.10", r_10, r_10.ok and "train" in r_10.stdout, "code 0, l'aide cite train.",
                             "attendu le code 0 et une aide qui cite la sous-commande train")
    r_10 = wbp.cli(PROJECT)
    ok_10["no command"] = run_line("B7.10", r_10, r_10.returncode == 2 and "usage" in r_10.stderr.lower(),
                                   "code 2 et l'usage sur stderr (la sous-commande est obligatoire).",
                                   "attendu le code 2 d'argparse : la sous-commande train est obligatoire")
    r_10 = wbp.cli(PROJECT, "train", "--epochs", "3")
    ok_10["unknown option"] = run_line("B7.10", r_10, r_10.returncode == 2, "code 2 (option inconnue).",
                                       "attendu le code 2 d'argparse pour une option inconnue")
    r_10 = wbp.cli(PROJECT, "train", "--set", "lr", "--out", out_10 / "a")
    ok_10["bad set"] = run_line("B7.10", r_10, r_10.returncode == 2 and r_10.stderr_lines("ERROR") and not r_10.stdout.strip(),
                                "code 2, une ligne ERROR, rien sur stdout.",
                                "attendu le code 2 (configuration invalide), une ligne ERROR sur stderr et rien sur stdout")
    r_10 = wbp.cli(PROJECT, "train", "--set", "data=data/absent.npz", "--out", out_10 / "b")
    ok_10["no data"] = run_line("B7.10", r_10, r_10.returncode == 1 and r_10.stderr_lines("ERROR") and not r_10.stdout.strip(),
                                "code 1, une ligne ERROR, rien sur stdout.",
                                "attendu le code 1 (données illisibles), une ligne ERROR sur stderr et rien sur stdout")
    r_10 = wbp.cli(PROJECT, "train", "--set", "n_train=200", "--out", out_10 / "c")
    lines_10 = [line for line in r_10.stdout.splitlines() if line.strip()]
    summary_10 = r_10.last_json()
    ok_json_10 = (len(lines_10) == 1 and isinstance(summary_10, dict) and summary_10.get("status") == "ok"
                  and isinstance(summary_10.get("val_acc"), float) and 0 <= summary_10["val_acc"] <= 1)
    ok_10["train"] = run_line("B7.10", r_10, r_10.ok and ok_json_10,
                              f"code 0, une ligne JSON sur stdout (val_acc {fr(summary_10['val_acc'], 3) if ok_json_10 else '?'}).",
                              "attendu le code 0 et UNE ligne JSON sur stdout, {\"status\": \"ok\", \"val_acc\": <float>, "
                              f"\"weights\": ...}} ; reçu {r_10.stdout.strip()[:200]!r}")
    if ok_10["train"]:
        weights_10 = Path(summary_10.get("weights", ""))
        from safetensors.torch import load_file
        ok_10["weights"] = (weights_10.is_absolute() and weights_10 == (out_10 / "c" / "model.safetensors").resolve()
                            and weights_10.is_file() and set(load_file(str(weights_10)))
                            == set(wbp.build_reference("cnn", (16, 32)).state_dict()))
        verdict("B7.10", ok_10["weights"], "les poids sont dans <out>/model.safetensors (chemin absolu dans le JSON), "
                "ceux du CNN (16, 32).", f"attendu \"weights\" = le chemin absolu de <out>/model.safetensors, un "
                f"fichier safetensors du CNN (16, 32) ; reçu {str(weights_10)!r}.")
        infos_10 = [line for line in r_10.stderr_lines("INFO") if "cifarprod.train" in line]
        ok_10["logging"] = bool(infos_10) and not r_10.stderr_lines("DEBUG")
        verdict("B7.10", ok_10["logging"], f"stderr : {len(r_10.stderr_lines('INFO'))} lignes INFO, dont "
                f"« {infos_10[-1].split(': ', 1)[-1][:60] if infos_10 else ''} » de cifarprod.train ; aucune DEBUG.",
                "attendu sur stderr des lignes INFO (au moins une du logger cifarprod.train, avec le niveau et le nom "
                f"du logger) et aucune ligne DEBUG sans -v ; reçu {r_10.tail(3)[:300]!r}")
        r_10 = wbp.cli(PROJECT, "train", "--set", "n_train=200", "--out", out_10 / "d", "-v")
        ok_10["verbose"] = run_line("B7.10", r_10, r_10.ok and any("cifarprod" in line for line in r_10.stderr_lines("DEBUG")),
                                    "avec -v, les lignes DEBUG des mini-batches.",
                                    "attendu, avec -v, des lignes DEBUG du paquet sur stderr")
    with wbp.isolated_logging():                         # main() configures the logging of THIS process
        code_10 = cli_10.main(["train", "--set", "lr"])
    ok_10["main"] = type(code_10) is int and code_10 == 2
    verdict("B7.10", ok_10["main"], 'main(["train", "--set", "lr"]) rend l\'entier 2 dans le notebook.',
            f'main(["train", "--set", "lr"]) doit rendre l\'entier 2 (sans sys.exit) ; reçu {code_10!r}.')
if ok_10 and all(ok_10.values()):
    print("✅ Ex B7.10 : ta ligne de commande parle avec des codes de sortie et du logging.")'''

# --- 📦 B7.11 ---------------------------------------------------------------------------------------------------------
SKEL_CONFIG = '''"""The configuration of cifarprod: a typed, frozen dataclass, read from a YAML file and overridden on the command line."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 1
    lr: float = 0.05
    batch_size: int = 128
    n_train: int = 2000
    arch: str = "cnn"
    widths: tuple = (16, 32)
    data: str = "data/cifar10_2000.npz"
    deterministic: bool = True
    num_workers: int = 0

    def __post_init__(self):
        """Check every setting (the contract of Ex B7.11): ValueError naming the setting."""
        raise NotImplementedError("TrainConfig.__post_init__")

    def to_dict(self):
        """The settings as a plain dictionary that json.dumps accepts (widths as a list)."""
        raise NotImplementedError("TrainConfig.to_dict")


def load_config(path=None, overrides=()):
    """The TrainConfig of the YAML file `path` (None: the defaults), then of the overrides "KEY=VALUE" (each VALUE read
    by yaml.safe_load). ValueError for a file that is not a mapping, an override without "=", an unknown key or a
    refused setting; FileNotFoundError for a missing file."""
    raise NotImplementedError("load_config")
'''

TODO_11 = "\n\n".join([
    "# Ex B7.11: the typed configuration (it replaces the starting version of B7.10) and two configuration files",
    file_cell("CONFIG_PY", SKEL_CONFIG, "src/cifarprod/config.py"),
    "FAST_YAML = ...    # configs/fast.yaml: the smoke training of the notebook (the contract)\n"
    "SMOKE_YAML = ...   # configs/smoke.yaml: the end-to-end test of the continuous integration (the contract)",
])

SOLUTION_11 = "\n\n".join([
    "# Ex B7.11: the typed configuration of the reference package, and its two configuration files",
    file_cell("CONFIG_PY", reference_file("src/cifarprod/config.py"), "src/cifarprod/config.py"),
    file_cell("FAST_YAML", reference_file("configs/fast.yaml"), "configs/fast.yaml"),
    file_cell("SMOKE_YAML", reference_file("configs/smoke.yaml"), "configs/smoke.yaml"),
])

CHECK_11 = r'''import dataclasses

ok_11 = {}
DEFAULTS_11 = DEFAULTS
with checked("B7.11"):
    wbp.write_files(PROJECT, {"src/cifarprod/config.py": CONFIG_PY, "configs/fast.yaml": FAST_YAML,
                              "configs/smoke.yaml": SMOKE_YAML})
    config_11 = wbp.import_module(PROJECT, "config")
    default_11 = config_11.TrainConfig()
    ok_11["defaults"] = default_11.to_dict() == DEFAULTS_11 and json.loads(json.dumps(default_11.to_dict())) == DEFAULTS_11
    verdict("B7.11", ok_11["defaults"], "TrainConfig() et to_dict() : les valeurs par défaut du contrat, en JSON.",
            f"TrainConfig().to_dict() doit valoir {DEFAULTS_11} ; reçu {default_11.to_dict()!r:.300}.")
    try:
        default_11.lr = 0.1
        ok_11["frozen"] = False
    except dataclasses.FrozenInstanceError:
        ok_11["frozen"] = True
    verdict("B7.11", ok_11["frozen"], "la configuration est figée (frozen=True).",
            "config.lr = 0.1 doit lever FrozenInstanceError : un run ne modifie pas sa configuration en route.")
    cast_11 = config_11.TrainConfig(lr=1 / 2 ** 4, epochs=2, widths=[8, 16])
    ok_11["cast"] = type(config_11.TrainConfig(seed=0, lr=0.5).lr) is float and cast_11.widths == (8, 16) \
        and type(config_11.TrainConfig(**{"lr": 1e-1}).lr) is float
    lr_int_11 = config_11.load_config(None, ["batch_size=64", "lr=0.25"])
    verdict("B7.11", ok_11["cast"] and lr_int_11.batch_size == 64, "un entier accepté pour un float, une liste YAML "
            "devient un tuple.", "attendu lr en float, widths=[8, 16] devenu (8, 16), et --set batch_size=64.")
    for override_11, name_11 in [("epochs=0", "epochs"), ("epochs=1.5", "epochs"), ("lr=0", "lr"), ("lr=1", "lr"),
                                 ("lr=1e-3", "lr"), ("batch_size=0", "batch_size"), ("batch_size=true", "batch_size"),
                                 ("n_train=0", "n_train"), ("seed=-1", "seed"), ("arch=resnet", "arch"),
                                 ("widths=[]", "widths"), ("widths=[8, 0]", "widths"), ("deterministic=yes2", "deterministic"),
                                 ("num_workers=-1", "num_workers"), ("lerning_rate=0.01", "lerning_rate"), ("lr", "lr")]:
        try:
            config_11.load_config(None, [override_11])
            ok_11[override_11] = False
            print(f"❌ Ex B7.11 : --set {override_11} : attendu ValueError, mais rien n'a été levé.")
        except ValueError as err_11:
            ok_11[override_11] = name_11 in str(err_11)
            if not ok_11[override_11]:
                print(f"❌ Ex B7.11 : --set {override_11} : ValueError levée, mais son message ne nomme pas {name_11!r} : "
                      f"« {str(err_11)[:120]} ».")
        except NotImplementedError:
            raise
        except Exception as err_11:  # noqa: BLE001 - the learner's code
            ok_11[override_11] = False
            print(f"❌ Ex B7.11 : --set {override_11} : attendu ValueError, reçu {type(err_11).__name__} : {str(err_11)[:120]}")
    refused_11 = [k for k in ok_11 if "=" in k or k == "lr"]
    if all(ok_11[k] for k in refused_11):
        print(f"✅ Ex B7.11 : {len(refused_11)} réglages invalides refusés, chacun par une ValueError qui le nomme "
              "(types, bornes, clé inconnue, --set sans =).")
    yaml_11 = scratch("yaml")
    (yaml_11 / "a.yaml").write_text("lr: 0.1\nepochs: 3\nwidths: [4, 8]\n", encoding="utf-8")
    both_11 = config_11.load_config(yaml_11 / "a.yaml", ["lr=0.01", "epochs=2"])
    ok_11["precedence"] = (both_11.lr, both_11.epochs, both_11.widths, both_11.seed) == (0.01, 2, (4, 8), 0)
    verdict("B7.11", ok_11["precedence"], "les valeurs par défaut, puis le fichier, puis les --set.",
            f"fichier lr 0.1, epochs 3, widths [4, 8] puis --set lr=0.01 epochs=2 : attendu (0.01, 2, (4, 8), seed 0), "
            f"reçu {(both_11.lr, both_11.epochs, both_11.widths, both_11.seed)}.")
    for text_11, what_11 in [("lr: 1e-3\n", "lr: 1e-3 (une chaîne pour PyYAML)"), ("lerning_rate: 0.01\n", "une faute de frappe"),
                             ("- lr\n- 0.1\n", "une liste au lieu d'un dictionnaire")]:
        (yaml_11 / "bad.yaml").write_text(text_11, encoding="utf-8")
        ok_11[what_11] = refuses("B7.11", f"un fichier avec {what_11}", lambda: config_11.load_config(yaml_11 / "bad.yaml"))
    (yaml_11 / "empty.yaml").write_text("# nothing\n", encoding="utf-8")
    ok_11["empty"] = config_11.load_config(yaml_11 / "empty.yaml").to_dict() == DEFAULTS_11
    verdict("B7.11", ok_11["empty"], "un fichier vide : les valeurs par défaut.", "un fichier vide (un commentaire) doit "
            "donner les valeurs par défaut (yaml.safe_load rend None).")
    ok_11["absent"] = refuses("B7.11", "un fichier absent", lambda: config_11.load_config(yaml_11 / "absent.yaml"),
                              error=FileNotFoundError)
    want_files_11 = {"configs/fast.yaml": DEFAULTS_11,
                     "configs/smoke.yaml": {**DEFAULTS_11, "n_train": 256, "batch_size": 64, "widths": [8, 16]}}
    for rel_11, want_11 in want_files_11.items():
        got_11 = config_11.load_config(PROJECT / rel_11).to_dict()
        ok_11[rel_11] = got_11 == want_11
        verdict("B7.11", ok_11[rel_11], f"{rel_11} : les réglages du contrat.",
                f"{rel_11} : attendu {want_11}, reçu {got_11}.")
    out_11 = scratch("out")
    r_11 = wbp.cli(PROJECT, "train", "--set", "epochs=0", "--out", out_11 / "a")
    ok_11["cli epochs"] = run_line("B7.11", r_11, r_11.returncode == 2 and any("epochs" in l for l in r_11.stderr_lines("ERROR")),
                                   "code 2 et une ligne ERROR qui nomme epochs, avant toute seconde de calcul.",
                                   "attendu le code 2 et une ligne ERROR qui nomme epochs")
    r_11 = wbp.cli(PROJECT, "train", "--config", "configs/smoke.yaml", "--out", out_11 / "b")
    ok_11["cli smoke"] = run_line("B7.11", r_11, r_11.ok, f"code 0 en {fr(r_11.seconds, 1)} s.", "attendu le code 0")
if ok_11 and all(ok_11.values()):
    print("✅ Ex B7.11 : ta configuration est typée, validée au chargement et surchargeable.")'''

# --- 🔮 B7.12 ---------------------------------------------------------------------------------------------------------
TODO_12 = r'''replay0_12 = ...     # a) two runs, seed 0, num_workers=0: exactly the same weights? True or False
replay4_12 = ...     # b) two runs, seed 0, num_workers=4: exactly the same weights? True or False
zero_four_12 = ...   # c) the run with num_workers=0 and the run with num_workers=4: the same weights? True or False
plain_12 = ...       # d) WITHOUT the random flip, num_workers=0 against num_workers=4: the same weights? True or False'''

EXPERIMENT_12 = r'''start_12 = time.perf_counter()
RUNS_12 = {"0, essai 1": wbp.workers_run(0, seed=0), "0, essai 2": wbp.workers_run(0, seed=0),
           "4, essai 1": wbp.workers_run(4, seed=0), "4, essai 2": wbp.workers_run(4, seed=0),
           "0, sans retournement": wbp.workers_run(0, seed=0, augment=False),
           "4, sans retournement": wbp.workers_run(4, seed=0, augment=False)}
PAIRS_12 = {"a": ("0, essai 1", "0, essai 2"), "b": ("4, essai 1", "4, essai 2"), "c": ("0, essai 1", "4, essai 1"),
            "d": ("0, sans retournement", "4, sans retournement")}
print(f"Six entraînements courts en {fr(time.perf_counter() - start_12, 1)} s (512 images, 2 epochs, graine 0) :")
for letter_12, (first_12, second_12) in PAIRS_12.items():
    a_12, b_12 = RUNS_12[first_12], RUNS_12[second_12]
    print(f"  {letter_12}) num_workers {first_12} contre {second_12} : "
          + ("mêmes poids au bit près" if wbp.same_weights(a_12, b_12) else f"poids différents (écart maximal "
             f"{wbp.max_gap(a_12, b_12):.2e})"))
if torch.cuda.is_available():
    gpu_12 = [wbp.workers_run(0, seed=0, device="cuda") for _ in range(2)]
    print(f"  e) sur le GPU ({torch.cuda.get_device_name(0)}) : deux runs "
          + ("identiques" if wbp.same_weights(*gpu_12) else f"différents ({wbp.max_gap(*gpu_12):.2e})")
          + " ; contre le CPU : " + ("identiques" if wbp.same_weights(gpu_12[0], RUNS_12["0, essai 1"])
                                     else f"différents ({wbp.max_gap(gpu_12[0], RUNS_12['0, essai 1']):.2e})"))
else:
    print("  e) pas de GPU ici : ta prédiction e) se compare à 05_solutions.md (ou relance cette cellule sur un GPU "
          "de Colab).")
wb.check("B7.12a", replay0_12)
wb.check("B7.12b", replay4_12)
wb.check("B7.12c", zero_four_12)
wb.check("B7.12d", plain_12)'''

# --- 🔬 B7.13 ---------------------------------------------------------------------------------------------------------
TODO_13 = r'''def train_run_13(seed, deterministic=True):
    """Run YOUR command line once in PROJECT: python -m cifarprod train --config configs/fast.yaml --set seed=<seed>
    --set deterministic=<true|false> --out <a new temporary folder>, through wbp.cli. Returns {"weights": the dict of
    tensors of the model.safetensors it wrote, "val_acc": the val_acc of its JSON line, "seconds": its duration};
    RuntimeError with the end of stderr if the command does not end with the code 0."""
    raise NotImplementedError("train_run_13")


def same_weights_13(a, b):
    """True when the two dicts of tensors have the same names and exactly the same values (torch.equal), else False."""
    raise NotImplementedError("same_weights_13")


def max_gap_13(a, b):
    """The largest absolute difference between the tensors of the same name of two dicts, as a Python float."""
    raise NotImplementedError("max_gap_13")'''

SOLUTION_13 = r'''from safetensors.torch import load_file


def train_run_13(seed, deterministic=True):
    """Run YOUR command line once in PROJECT: python -m cifarprod train --config configs/fast.yaml --set seed=<seed>
    --set deterministic=<true|false> --out <a new temporary folder>, through wbp.cli. Returns {"weights": the dict of
    tensors of the model.safetensors it wrote, "val_acc": the val_acc of its JSON line, "seconds": its duration};
    RuntimeError with the end of stderr if the command does not end with the code 0."""
    out = tempfile.mkdtemp(prefix="wb_b7_13_")
    result = wbp.cli(PROJECT, "train", "--config", "configs/fast.yaml", "--set", f"seed={seed}",
                     "--set", f"deterministic={str(deterministic).lower()}", "--out", out)
    if not result.ok:
        raise RuntimeError(f"cifarprod train exited with {result.returncode}: {result.tail(3)}")
    summary = result.last_json()
    return {"weights": load_file(summary["weights"]), "val_acc": summary["val_acc"], "seconds": result.seconds}


def same_weights_13(a, b):
    """True when the two dicts of tensors have the same names and exactly the same values (torch.equal), else False."""
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def max_gap_13(a, b):
    """The largest absolute difference between the tensors of the same name of two dicts, as a Python float."""
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)'''

CHECK_13 = r'''tools_ok_13 = False
with checked("B7.13"):
    base_13 = {"w": torch.zeros(2, 3), "b": torch.ones(4), "n": torch.tensor(7)}
    twin_13 = {k: v.clone() for k, v in base_13.items()}
    tiny_13 = {**twin_13, "w": torch.tensor([[0.0, 0.0, 1e-6], [0.0, 0.0, 0.0]])}
    cases_13 = [("deux copies identiques", base_13, twin_13, True), ("un poids qui diffère de 1e-6", base_13, tiny_13, False),
                ("un tenseur en moins", base_13, {k: v for k, v in twin_13.items() if k != "b"}, False)]
    tools_ok_13 = True
    for label_13, a_13, b_13, want_13 in cases_13:
        got_13 = same_weights_13(a_13, b_13)
        good_13 = isinstance(got_13, (bool, np.bool_)) and bool(got_13) == want_13
        tools_ok_13 &= good_13
        verdict("B7.13", good_13, f"same_weights_13 sur {label_13} : {want_13}.",
                f"same_weights_13 sur {label_13} : attendu {want_13}, reçu {got_13!r}.")
    gap_13 = max_gap_13(base_13, tiny_13)
    ok_gap_13 = type(gap_13) is float and abs(gap_13 - float(torch.tensor(1e-6))) < 1e-12
    tools_ok_13 &= ok_gap_13
    verdict("B7.13", ok_gap_13, "max_gap_13 : 1e-6 (un float Python).",
            f"max_gap_13 sur un poids qui diffère de 1e-6 : reçu {gap_13!r}, attendu le float 1e-6 (à l'arrondi du float32 près).")
if not written(lambda: train_run_13(0)):
    print("⏳ Ex B7.13 : l'expérience attend ta fonction train_run_13.")
    tools_ok_13 = False'''

EXPERIMENT_13 = r'''RUNS_13 = {}
if not tools_ok_13:
    print("⏳ Ex B7.13 : l'expérience attend same_weights_13, max_gap_13 et train_run_13 justes (cellule précédente).")
else:
    with checked("B7.13"):
        PLAN_13 = {"A": (0, True), "B": (0, True), "C": (0, False), "D": (1, True), "E": (2, True)}
        for key_13, (seed_13, det_13) in PLAN_13.items():
            RUNS_13[key_13] = train_run_13(seed_13, det_13)
            print(f"run {key_13} : graine {seed_13}, algorithmes déterministes {'oui' if det_13 else 'non'} → val_acc "
                  f"{fr(RUNS_13[key_13]['val_acc'], 3)} en {fr(RUNS_13[key_13]['seconds'], 1)} s")
        for first_13, second_13, what_13 in [("A", "B", "même graine, deux fois"), ("A", "C", "déterministe ou non"),
                                             ("A", "D", "graine 0 contre graine 1"), ("D", "E", "graine 1 contre graine 2")]:
            wa_13, wb_13 = RUNS_13[first_13]["weights"], RUNS_13[second_13]["weights"]
            print(f"  {first_13} contre {second_13} ({what_13}) : " + ("poids identiques au bit près"
                  if same_weights_13(wa_13, wb_13) else f"écart maximal {max_gap_13(wa_13, wb_13):.3g}"))
        accs_13 = np.array([RUNS_13[k]["val_acc"] for k in ("A", "D", "E")])
        half_13 = 4.303 * accs_13.std(ddof=1) / np.sqrt(3)                # t(0.975; 2), the table of 207.4
        print(f"accuracy sur 3 graines : {fr(100 * accs_13.mean(), 1)} % ± {fr(100 * half_13, 1)} (intervalle à 95 %, "
              f"t = 4,303 pour n = 3 ; valeurs {', '.join(fr(100 * a, 1) for a in accs_13)})")
        wb.check("B7.13", same_weights_13(RUNS_13["A"]["weights"], RUNS_13["B"]["weights"]))'''

RECORD_13 = r'''wb.record("B7.13", True, mistakes={"tes deux runs à même graine n'ont pas exactement les mêmes poids : qu'est-ce qui, "
                                   "dans ton entraînement, ne dépend pas de la graine de la configuration ?": False})'''

# --- 📦 B7.14 ---------------------------------------------------------------------------------------------------------
SKEL_TRACKING = '''"""The run journal of cifarprod: one JSON line per run, appended to a JSONL file, never rewritten."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def new_run_id():
    """A unique run identifier: the UTC date and time, then six random hexadecimal characters
    ("20261104-161731-e06339")."""
    raise NotImplementedError("new_run_id")


def file_sha256(path):
    """The SHA-256 of the bytes of a file (64 hexadecimal characters)."""
    raise NotImplementedError("file_sha256")


def artifact(path):
    """{"path": str(path), "bytes": its size, "sha256": file_sha256(path)} of a file written by the run."""
    raise NotImplementedError("artifact")


def environment(cwd=None):
    """{"python", "numpy", "torch", "platform", "commit", "dirty"}: the versions (platform.python_version(),
    importlib.metadata.version), platform.platform(), the short git commit and whether `git status --porcelain` is not
    empty (None and None outside a git repository)."""
    raise NotImplementedError("environment")


def log_run(path, record):
    """Append one JSON line (keys sorted) to the journal `path`; its folder is created if needed."""
    raise NotImplementedError("log_run")


def read_runs(path):
    """The runs of the journal as a list of dicts, oldest first ([] if the file does not exist)."""
    raise NotImplementedError("read_runs")
'''

TODO_14 = file_cell("TRACKING_PY", SKEL_TRACKING, "Ex B7.14: src/cifarprod/tracking.py; then complete CLI_PY in the "
                    "cell of Ex B7.10 (the contract) and run that cell again")

SOLUTION_14 = "\n\n".join([
    "# Ex B7.14: tracking.py (the solutions notebook has written it with cli.py in Ex B7.10: the same text again)",
    file_cell("TRACKING_PY", reference_file("src/cifarprod/tracking.py"), "src/cifarprod/tracking.py"),
])

CHECK_14 = r'''import hashlib
import platform
from importlib import metadata

import pandas as pd

ok_14 = {}
with checked("B7.14"):
    wbp.write_files(PROJECT, {"src/cifarprod/tracking.py": TRACKING_PY, "src/cifarprod/cli.py": CLI_PY})
    tracking_14 = wbp.import_module(PROJECT, "tracking")
    ids_14 = [tracking_14.new_run_id() for _ in range(3)]
    today_14 = time.strftime("%Y%m%d", time.gmtime())
    ok_14["id"] = (all(isinstance(i, str) and re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", i) for i in ids_14)
                   and len(set(ids_14)) == 3 and ids_14[0][:8] == today_14)
    verdict("B7.14", ok_14["id"], f"new_run_id : {ids_14[0]}, {ids_14[1]}… (date UTC, heure, six caractères au hasard).",
            f"new_run_id : attendu trois identifiants distincts de la forme {today_14}-HHMMSS-xxxxxx (date UTC), "
            f"reçu {ids_14!r:.120}.")
    blob_14 = scratch("blob") / "weights.bin"
    blob_14.write_bytes(np.random.default_rng(0).bytes(300_000))
    want_art_14 = {"path": str(blob_14), "bytes": 300_000, "sha256": hashlib.sha256(blob_14.read_bytes()).hexdigest()}
    ok_14["artifact"] = tracking_14.file_sha256(blob_14) == want_art_14["sha256"] and tracking_14.artifact(blob_14) == want_art_14
    verdict("B7.14", ok_14["artifact"], "file_sha256 et artifact d'un fichier de 300 000 octets : comme hashlib.",
            f"artifact(fichier de 300 000 octets) : attendu {{'path', 'bytes', 'sha256'}} = {want_art_14['bytes']} octets "
            f"et le SHA-256 de hashlib, reçu {tracking_14.artifact(blob_14)!r:.200}.")
    env_14 = tracking_14.environment()
    ok_14["environment"] = (isinstance(env_14, dict) and {"python", "numpy", "torch", "platform", "commit", "dirty"} <= set(env_14)
                            and env_14["python"] == platform.python_version() and env_14["torch"] == metadata.version("torch"))
    verdict("B7.14", ok_14["environment"], f"environment : Python {env_14.get('python')}, torch {env_14.get('torch')}, "
            f"commit {env_14.get('commit')!r}.", "environment doit rendre {python, numpy, torch, platform, commit, dirty} "
            f"avec platform.python_version() et importlib.metadata.version ; reçu {env_14!r:.200}.")
    journal_14 = scratch("journal") / "sub" / "runs.jsonl"
    tracking_14.log_run(journal_14, {"run_id": "a", "config": {"lr": 0.1}})
    first_14 = journal_14.read_text(encoding="utf-8")
    tracking_14.log_run(journal_14, {"run_id": "b", "metrics": [1, 2]})
    lines_14 = journal_14.read_text(encoding="utf-8").splitlines()
    ok_14["append"] = (len(lines_14) == 2 and journal_14.read_text(encoding="utf-8").startswith(first_14)
                       and [json.loads(l) for l in lines_14] == [{"run_id": "a", "config": {"lr": 0.1}}, {"run_id": "b", "metrics": [1, 2]}]
                       and tracking_14.read_runs(journal_14) == [json.loads(l) for l in lines_14]
                       and tracking_14.read_runs(journal_14.with_name("absent.jsonl")) == [])
    verdict("B7.14", ok_14["append"], "log_run ajoute une ligne JSON par run (dossier créé), read_runs les relit ([] sans fichier).",
            f"log_run puis read_runs : attendu deux lignes JSON, la première inchangée, relues dans l'ordre, et [] pour "
            f"un journal absent ; le fichier contient {lines_14!r:.200}.")

with checked("B7.14"):                                   # the command line logs every run
    out_14 = scratch("runs")
    journal_14 = out_14 / "runs.jsonl"
    r1_14 = wbp.cli(PROJECT, "train", "--set", "n_train=200", "--out", out_14 / "a", "--runs", journal_14)
    ok_14["run 1"] = run_line("B7.14", r1_14, r1_14.ok, "code 0.", "attendu le code 0")
    first_line_14 = journal_14.read_text(encoding="utf-8").splitlines()[0] if journal_14.is_file() else None
    r2_14 = wbp.cli(PROJECT, "train", "--set", "n_train=99999", "--runs", journal_14)
    ok_14["run 2"] = run_line("B7.14", r2_14, r2_14.returncode == 1, "code 1 (n_train plus grand que les données).",
                              "attendu le code 1 : n_train = 99 999 dépasse les 2 000 images")
    r3_14 = wbp.cli(PROJECT, "train", "--set", "n_train=200", "--set", "seed=1", "--runs", journal_14)
    ok_14["run 3"] = run_line("B7.14", r3_14, r3_14.ok, "code 0, sans --out.", "attendu le code 0")
    runs_14 = [json.loads(l) for l in journal_14.read_text(encoding="utf-8").splitlines()] if journal_14.is_file() else []
    ok_14["three"] = [r.get("status") for r in runs_14] == ["ok", "failed", "ok"] and \
        journal_14.read_text(encoding="utf-8").splitlines()[0] == first_line_14
    verdict("B7.14", ok_14["three"], "trois lignes dans le journal, statuts ok, failed, ok ; la première jamais réécrite.",
            f"attendu trois lignes de statuts ['ok', 'failed', 'ok'], la première inchangée ; reçu "
            f"{[r.get('status') for r in runs_14]}.")
    if not ok_14["three"]:
        raise StopCheck
    keys_14 = {"run_id", "started", "status", "seconds", "config", "environment", "data", "metrics", "normalization",
               "artifacts"}
    ok_rec_14 = all(keys_14 <= set(r) for r in (runs_14[0], runs_14[2])) and {"run_id", "status", "seconds", "config",
                                                                               "error"} <= set(runs_14[1])
    verdict("B7.14", ok_rec_14, "chaque ligne a les champs du contrat (un run raté : son erreur).",
            f"champs manquants : run réussi {sorted(keys_14 - set(runs_14[0]))}, run raté "
            f"{sorted({'run_id', 'status', 'seconds', 'config', 'error'} - set(runs_14[1]))}.")
    if not ok_rec_14:
        raise StopCheck
    rec_14 = runs_14[0]
    ok_14["ids"] = len({r["run_id"] for r in runs_14}) == 3 and r1_14.last_json().get("run_id") == rec_14["run_id"] \
        and r3_14.last_json().get("run_id") == runs_14[2]["run_id"]
    verdict("B7.14", ok_14["ids"], "trois identifiants distincts, ceux des lignes JSON de stdout.",
            "attendu trois run_id distincts, et sur stdout le run_id de la ligne du journal.")
    ok_14["config"] = rec_14["config"] == {**DEFAULTS, "n_train": 200} and runs_14[2]["config"] == {**DEFAULTS, "n_train": 200, "seed": 1}
    verdict("B7.14", ok_14["config"], "la configuration complète du run (to_dict), surcharges comprises.",
            f"config du premier run : attendu {{**DEFAULTS, 'n_train': 200}}, reçu {rec_14['config']!r:.200}.")
    want_sha_14 = wbp.array_fingerprint(SUBSET["x_train"][:200], SUBSET["y_train"][:200])
    ok_14["data"] = rec_14["data"].get("sha256") == want_sha_14 and rec_14["data"].get("n_train") == 200
    verdict("B7.14", ok_14["data"], f"l'empreinte des 200 images d'entraînement et de leurs labels : {want_sha_14[:12]}….",
            f"data : attendu n_train 200 et sha256 = fingerprint(X[:200], y[:200]) des images ET des labels utilisés, "
            f"reçu {rec_14['data']!r:.200}.")
    arts_14 = rec_14["artifacts"]
    art_path_14 = (PROJECT / arts_14[0]["path"]).resolve() if arts_14 else None
    ok_14["artifacts"] = (len(arts_14) >= 1 and art_path_14 == (out_14 / "a" / "model.safetensors").resolve()
                          and arts_14[0].get("sha256") == hashlib.sha256(art_path_14.read_bytes()).hexdigest()
                          and arts_14[0].get("bytes") == art_path_14.stat().st_size)
    verdict("B7.14", ok_14["artifacts"], "artifacts : le fichier des poids, sa taille et son SHA-256.",
            f"artifacts : attendu [{{path, bytes, sha256}}] du fichier <out>/model.safetensors, reçu {arts_14!r:.200}.")
    default_out_14 = (PROJECT / runs_14[2]["artifacts"][0]["path"]).resolve() if runs_14[2]["artifacts"] else None
    ok_14["default out"] = default_out_14 == (PROJECT / "runs" / runs_14[2]["run_id"] / "model.safetensors").resolve()
    verdict("B7.14", ok_14["default out"], "sans --out, les poids vont dans runs/<run_id>/.",
            f"sans --out, attendu runs/<run_id>/model.safetensors, reçu {default_out_14}.")
    metrics_14 = rec_14["metrics"]
    ok_14["metrics"] = (len(metrics_14.get("val_acc", [])) == 1 and metrics_14.get("val_acc_final") == r1_14.last_json().get("val_acc")
                        and np.allclose(rec_14["normalization"]["mean"], (SUBSET["x_train"][:200] / 255).mean(axis=(0, 1, 2))))
    verdict("B7.14", ok_14["metrics"], "metrics (une valeur par epoch, val_acc_final) et la normalisation apprise.",
            "metrics : attendu train_loss, val_loss, val_acc (une valeur par epoch) et val_acc_final, la val_acc de "
            "stdout ; normalization : la moyenne et l'écart-type des images d'entraînement.")
    ok_14["failed"] = "n_train" in runs_14[1].get("error", "")
    verdict("B7.14", ok_14["failed"], f"le run raté garde son erreur : « {runs_14[1].get('error', '')[:70]} ».",
            f"le run raté doit garder son erreur (qui nomme n_train) ; reçu {runs_14[1].get('error')!r}.")
    TABLE_14 = pd.json_normalize(runs_14)[["run_id", "status", "config.seed", "config.n_train", "metrics.val_acc_final",
                                           "seconds"]]
    print(TABLE_14.to_string(index=False))
if ok_14 and all(ok_14.values()) and len(ok_14) >= 12:
    print("✅ Ex B7.14 : chaque run laisse sa ligne : ce qui le définit, ce qu'il mesure, ce qu'il produit.")'''

# --- 📦 B7.15 ---------------------------------------------------------------------------------------------------------
GIVEN_TEST_MODEL = '''"""Tests of the model: the shape of the output, and a mini-batch learned by heart (the two tests of the fiche, 207.6)."""

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("batch", [1, 7])
def test_output_shape(model, batch):
    out = model(torch.zeros(batch, 3, 32, 32))
    assert out.shape == (batch, 10), f"expected ({batch}, 10), got {tuple(out.shape)}"


def test_overfits_one_batch(model):
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(16, 3, 32, 32, generator=g), torch.arange(16) % 10
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"expected a loss < 0.05 on 16 memorised images, got {loss.item():.3f}"
'''

TODO_15 = "\n\n".join([
    "# Ex B7.15: the tests of your package (written into PROJECT/tests by the check cell)",
    file_cell("CONFTEST_PY", reference_file("tests/conftest.py"), "tests/conftest.py (given: the fixtures model, "
              "data_file, synthetic_file)"),
    file_cell("TEST_MODEL_PY", GIVEN_TEST_MODEL, "tests/test_model.py (given: the two tests of the fiche; add yours "
              "if you like)"),
    "TEST_DATA_PY = ...   # tests/test_data.py: your tests of the data (the contract)\n"
    "TEST_CLI_PY = ...    # tests/test_cli.py: your tests of the configuration and of the command line, one marked slow",
])

SOLUTION_15 = "\n\n".join([
    "# Ex B7.15: the tests of the reference package",
    file_cell("CONFTEST_PY", reference_file("tests/conftest.py"), "tests/conftest.py"),
    file_cell("TEST_MODEL_PY", reference_file("tests/test_model.py"), "tests/test_model.py"),
    file_cell("TEST_DATA_PY", reference_file("tests/test_data.py"), "tests/test_data.py"),
    file_cell("TEST_CLI_PY", reference_file("tests/test_cli.py"), "tests/test_cli.py"),
])

CHECK_15 = r'''ok_15 = {}
PYTEST_ARGS_15 = ["-c", str(PROJECT / "pyproject.toml"), "--rootdir", str(PROJECT), "--strict-markers"]
ENV_15 = {"PYTHONPATH": str(PROJECT / "src")}
TESTS_15 = {"tests/conftest.py": CONFTEST_PY, "tests/test_model.py": TEST_MODEL_PY, "tests/test_data.py": TEST_DATA_PY,
            "tests/test_cli.py": TEST_CLI_PY}
if TEST_DATA_PY is ... or TEST_CLI_PY is ...:
    print("⏳ Ex B7.15 : écris d'abord TEST_DATA_PY et TEST_CLI_PY.")
else:
    with checked("B7.15"):
        wbp.write_files(PROJECT, TESTS_15)
        own_15 = {}
        for rel_15 in ("tests/test_model.py", "tests/test_data.py", "tests/test_cli.py"):
            print(f"— {rel_15}, sur ton paquet :")
            own_15[rel_15] = run_pytest_group(rel_15, root=PROJECT, env=ENV_15, args=PYTEST_ARGS_15)
        ok_15["own"] = all(r.ok for r in own_15.values())
        verdict("B7.15", ok_15["own"], "tes tests passent sur ton paquet.", "tes tests doivent d'abord passer sur ton "
                "paquet (les lignes ci-dessus) ; si c'est ton paquet qui a tort, corrige-le dans sa cellule.")
        collected_15 = {}
        for marker_15 in ("", "slow"):
            listing_15 = wbp.run([sys.executable, "-m", "pytest", "tests/test_data.py", "tests/test_cli.py", "--collect-only",
                                  "-q", "-p", "no:cacheprovider", *PYTEST_ARGS_15, *(["-m", marker_15] if marker_15 else [])],
                                 PROJECT, src=[PROJECT / "src"], env={"COLUMNS": "1000"})
            collected_15[marker_15] = sum("::" in line for line in listing_15.stdout.splitlines())
        ok_15["count"] = collected_15[""] >= 8 and collected_15["slow"] >= 1
        verdict("B7.15", ok_15["count"], f"{collected_15['']} tests dans test_data.py et test_cli.py (paramètres compris), "
                f"dont {collected_15['slow']} marqué(s) slow.", f"{collected_15['']} tests dans test_data.py et test_cli.py, "
                f"{collected_15['slow']} marqué(s) slow : attendu au moins 8 tests (paramètres compris), dont un marqué slow.")
        if not ok_15["own"]:
            raise StopCheck
        base_15 = scratch("mutants")
        reference_15 = wbp.run_tests(wbp.mutant_package(None, base_15 / "reference"), PROJECT / "tests", data=DATA_FILE)
        ok_15["reference"] = reference_15.ok
        verdict("B7.15", reference_15.ok, f"tes tests passent aussi sur le paquet de référence ({reference_15.summary}).",
                f"tes tests doivent passer sur le paquet de référence, qui suit le contrat ({reference_15.summary}) :")
        for name_15, reason_15 in reference_15.failures[:6]:
            print(f"   ❌ {name_15}\n      {reason_15[:300]}")
        if not reference_15.ok:
            print("   un test qui échoue ici exige plus (ou autre chose) que le contrat de B7.9 à B7.14.")
            raise StopCheck
        for k_15, mutant_15 in enumerate(wbp.MUTANTS, start=1):
            package_15 = wbp.mutant_package(mutant_15.key, base_15 / mutant_15.key)
            run_15 = wbp.run_tests(package_15, PROJECT / "tests", data=DATA_FILE,
                                   marker="slow" if mutant_15.slow else "not slow", maxfail=1)
            if run_15.failed + run_15.errors == 0:
                ok_15[mutant_15.key] = False
                print(f"❌ Ex B7.15 : version fautive {k_15} ({mutant_15.what}) : elle passe tous tes tests"
                      + (" marqués slow" if mutant_15.slow else "") + f" ({run_15.summary}). Quel test aurait dû la voir ?")
                continue
            name_15, reason_15 = run_15.failures[0] if run_15.failures else ("?", "")
            ok_15[mutant_15.key] = wbp.says_expected(reason_15)
            verdict("B7.15", ok_15[mutant_15.key], f"version fautive {k_15} ({mutant_15.what}) : attrapée par {name_15} "
                    f"(« {reason_15[:110]} »).", f"version fautive {k_15} ({mutant_15.what}) : attrapée par {name_15}, "
                    f"mais la première ligne de son message ne dit pas ce qui était attendu : « {reason_15[:160]} ».")
if ok_15 and all(ok_15.values()) and len(ok_15) == 3 + len(wbp.MUTANTS):
    print(f"✅ Ex B7.15 : tes tests passent sur un paquet juste et attrapent les {len(wbp.MUTANTS)} versions fautives.")'''

# =====================================================================================================================
# Part B (start) · 🐛 B7.16
# =====================================================================================================================
GIVEN_16 = r'''# The colleague's model (the VGG 16-32-64 delivered by the MP5 solution, trained in their notebook on uint8 RGB images
# divided by 255, then normalized with the mean and the std of its training images, in float32; output k = CLASSES[k]),
# 200 test images saved as the PNG files a client uploads, what the evaluation of the notebook gives on them, and the
# five diagnoses, in a row
from safetensors.torch import load_file, save_file

MODEL_16, INFO_16 = wbp.reference_model()               # a new copy at each call: the diagnoses never share it
X_16, Y_16 = wbp.test_images(20)                         # 200 test images (20 per class), uint8 RGB (200, 32, 32, 3)
UPLOADS_16 = scratch("uploads")
PATHS_16 = [wbp.write_png(image, UPLOADS_16 / f"upload_{i:03d}.png") for i, image in enumerate(X_16)]
MEAN_16, STD_16 = np.asarray(INFO_16["mean"], np.float32), np.asarray(INFO_16["std"], np.float32)
FRENCH_16 = {"airplane": "avion", "automobile": "automobile", "bird": "oiseau", "cat": "chat", "deer": "cerf",
             "dog": "chien", "frog": "grenouille", "horse": "cheval", "ship": "bateau", "truck": "camion"}


def offline_input_16(images):
    """The input of the model in the colleague's notebook: uint8 RGB (N, 32, 32, 3) -> float32 (N, 3, 32, 32)."""
    x = (np.asarray(images, np.float32) / np.float32(255) - MEAN_16) / STD_16
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


with torch.inference_mode():
    OFFLINE_16 = MODEL_16(offline_input_16(X_16)).argmax(dim=1).numpy()      # the evaluation of the notebook
WANT_16 = [FRENCH_16[CLASSES[k]] for k in OFFLINE_16]                         # what the service should answer
print(f"Dans le notebook du collègue : accuracy {fr((OFFLINE_16 == Y_16).mean(), 3)} sur ces 200 images de test.")


def fresh_service_16(model=None, classes=CLASSES):
    """(model, meta) of the service, from a NEW artifact written by YOUR save_artifact_16 (the colleague's model, or
    `model` whose outputs are the classes `classes`), loaded by YOUR load_service_16."""
    folder = scratch("artifact")
    save_artifact_16(model if model is not None else wbp.reference_model()[0], folder, list(INFO_16["mean"]),
                     list(INFO_16["std"]), list(classes))
    return load_service_16(folder)


def diagnose_16(ex_id="B7.16"):
    """The five diagnoses of the service, in a row: the first one that fails stops the chain (one gap can hide
    another). Returns True when the five pass."""

    def point1():                                         # does the service answer?
        model, meta = fresh_service_16()
        seen = []
        handle = model.register_forward_pre_hook(lambda module, args: seen.append(args[0].dtype))
        try:
            names = predict_16(model, meta, PATHS_16[:8])
        except Exception as error:  # noqa: BLE001 - the learner's code: the symptom
            return False, f"le service plante sur 8 images envoyées : {type(error).__name__} : {str(error)[:160]}", \
                "lis la dernière ligne de l'erreur : quel tenseur ne va pas avec les poids du modèle ?"
        finally:
            handle.remove()
        if not (isinstance(names, list) and len(names) == 8 and all(isinstance(n, str) for n in names)):
            return False, f"predict_16 rend {type(names).__name__} {str(names)[:80]}", "attendu une liste de 8 noms (str)"
        weights = {p.dtype for p in model.parameters()}
        if weights != {torch.float32} or set(seen) != {torch.float32}:
            return False, (f"le modèle du service a des poids {sorted(map(str, weights))} et reçoit {[str(d) for d in seen]}, "
                           "alors que l'artefact et l'entraînement sont en float32"), "où ce type apparaît-il ?"
        return True, "le service répond 8 noms, en float32 comme l'entraînement", ""

    def point2():                                         # the image read from the file
        image = read_upload_16(PATHS_16[0])
        got = np.asarray(image)
        want = X_16[0].astype(np.float32) / 255
        if got.shape != (32, 32, 3):
            return False, f"l'image lue a la forme {got.shape}", "attendu (32, 32, 3), comme à l'entraînement"
        got = got.astype(np.float64)
        if np.allclose(got, want, atol=1e-6):
            return True, "l'image lue est celle de l'entraînement : RGB, des floats dans [0, 1]", ""
        symptoms, scaled = [], got
        if got.max() > 1.5:
            symptoms.append(f"ses valeurs vont de {got.min():.0f} à {got.max():.0f}, alors que l'entraînement voit des "
                            "pixels dans [0, 1]")
            scaled = got / 255
        if not np.allclose(scaled, want, atol=1e-6) and np.allclose(scaled, want[..., ::-1], atol=1e-6):
            symptoms.append("son canal 0 est le canal 2 de l'image envoyée (le bleu à la place du rouge)")
        elif not np.allclose(scaled, want, atol=1e-6):
            symptoms.append(f"elle s'écarte de l'image envoyée de {np.abs(scaled - want).max():.3f}")
        return False, "l'image lue n'est pas celle que voit l'entraînement : " + " ; ".join(symptoms), \
            "compare-la à X_16[0] / 255 : que rend la fonction de lecture ?"

    def point3():                                         # the preprocessing
        model, meta = fresh_service_16()
        images = X_16[:8].astype(np.float32) / 255
        batch = torch.as_tensor(preprocess_16(images.copy(), meta)).double()
        if tuple(batch.shape) != (8, 3, 32, 32):
            return False, f"preprocess_16 de 8 images rend la forme {tuple(batch.shape)}", "attendu (8, 3, 32, 32)"
        alone = torch.as_tensor(preprocess_16(images[:1].copy(), meta)).double()
        want = offline_input_16(X_16[:8]).double()
        if not torch.allclose(alone[0], batch[0], atol=1e-5):
            return False, (f"une image prétraitée seule n'a pas les mêmes valeurs que dans un batch de 8 (écart "
                           f"{float((alone[0] - batch[0]).abs().max()):.3f}) : son prétraitement dépend des autres images "
                           "envoyées"), "d'où viennent la moyenne et l'écart-type qu'utilise le service ?"
        if not torch.allclose(batch, want, atol=1e-4):
            return False, (f"le prétraitement du service s'écarte de celui de l'entraînement de "
                           f"{float((batch - want).abs().max()):.3f}"), "relis comment le notebook préparait les images"
        return True, "le prétraitement est celui de l'entraînement, image par image", ""

    def point4():                                         # the model of the service
        model, meta = fresh_service_16()
        x = offline_input_16(X_16[:8])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            together, alone = model(x), model(x[:1])
        changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
        if changed or not torch.allclose(together[:1], alone, atol=1e-5):
            return False, ("la même image, bien prétraitée, reçoit d'autres scores seule et dans un batch de 8 (écart "
                           f"{float((together[:1] - alone).abs().max()):.3f})"
                           + (", et chaque appel change les statistiques internes du modèle (running_mean)" if changed else "")), \
                "compare le modèle que rend load_service_16 à celui de l'évaluation hors ligne : qu'est-ce qui les distingue ?"
        with torch.inference_mode():
            want = MODEL_16(x)
        if not torch.allclose(together, want, atol=1e-4):
            return False, f"les scores du service s'écartent de ceux du notebook de {float((together - want).abs().max()):.3f}", \
                "les poids chargés sont-ils ceux de l'artefact ?"
        return True, "le modèle du service donne les scores du notebook, seule ou en batch", ""

    def point5():                                         # the names returned: the non-regression test
        model, meta = fresh_service_16()
        names = predict_16(model, meta, PATHS_16)
        agree = sum(a == b for a, b in zip(names, WANT_16))
        if agree != len(WANT_16):
            pairs = {}
            for got, want in zip(names, WANT_16):
                if got != want:
                    pairs[(want, got)] = pairs.get((want, got), 0) + 1
            (want, got), count = max(pairs.items(), key=lambda item: item[1])
            return False, (f"sur 200 images, le service ne rend le nom du notebook que pour {agree} ; par exemple, "
                           f"{count} images que le notebook classe « {want} » reçoivent « {got} »"), \
                "le nom que rend le service correspond-il à la sortie du modèle ?"
        perm = np.random.default_rng(16).permutation(10)
        shuffled = wbp.reference_model()[0]
        with torch.no_grad():
            shuffled[-1].weight.copy_(shuffled[-1].weight[perm].clone())
            shuffled[-1].bias.copy_(shuffled[-1].bias[perm].clone())
        model, meta = fresh_service_16(shuffled, [CLASSES[k] for k in perm])
        agree = sum(a == b for a, b in zip(predict_16(model, meta, PATHS_16), WANT_16))
        if agree != len(WANT_16):
            return False, (f"le même modèle, ses sorties rangées dans un autre ordre (et classes dans cet ordre), ne "
                           f"reçoit le bon nom que pour {agree} images sur 200"), \
                "d'où le service tient-il l'ordre des classes ?"
        return True, ("les 200 noms sont ceux du notebook, même pour un modèle dont les sorties sont rangées dans un "
                      "autre ordre"), ""

    points = [("le service répond", point1), ("l'image lue", point2), ("le prétraitement", point3),
              ("le modèle du service", point4), ("les noms rendus (test de non-régression)", point5)]
    for k, (name, run) in enumerate(points, start=1):
        try:
            with wbp.torch_state_kept():
                ok, measured, why = run()
        except NotImplementedError:
            raise
        except Exception as error:  # noqa: BLE001 - the learner's code: one ❌ line, then the chain stops
            ok, measured, why = False, f"une erreur {type(error).__name__} : {str(error)[:200]}", "relis la fonction en cause"
        if ok:
            print(f"✅ Ex {ex_id} : point {k}, {name} ({measured}).")
        else:
            print(f"❌ Ex {ex_id} : point {k}, {name} : {measured} ; {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex {ex_id} : point {later} : corrige d'abord le point {k}.")
            return False
    print(f"✅ Ex {ex_id} : les cinq diagnostics passent : le service répond comme le notebook.")
    return True'''

TODO_16 = r'''# The colleague's service, to repair IN PLACE (same names, same signatures). It works in their notebook, not in production.
def save_artifact_16(model, folder, mean, std, classes):
    """Training side: the artifact of `model` in `folder`: model.safetensors (the weights) and meta.json (what the
    service needs). mean, std: the normalization learned on the training images; classes: the names of the outputs,
    in their order."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(folder / "model.safetensors"))
    meta = {"arch": "vgg", "widths": [16, 32, 64], "mean": list(mean), "std": list(std)}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_service_16(folder):
    """Service side: (model, meta), the model rebuilt from the artifact, ready to predict."""
    meta = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    model = wbp.build_reference(meta["arch"], meta["widths"])
    model.load_state_dict(load_file(str(Path(folder) / "model.safetensors")))
    return model, meta


def read_upload_16(path):
    """Service side: an uploaded image file, as the array the model was trained on."""
    return wbp.imread_bgr(path)


def preprocess_16(images, meta):
    """Service side: the images (N, 32, 32, 3) read by read_upload_16, as the input tensor (N, 3, 32, 32) of the model."""
    x = np.asarray(images, dtype=np.float64)                      # double precision, to be safe
    mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))
    x = (x - mean) / std
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def predict_16(model, meta, paths):
    """Service side: the French name of the class of each uploaded image (a list of str)."""
    images = np.stack([read_upload_16(p) for p in paths])
    with torch.inference_mode():
        k = model(preprocess_16(images, meta)).argmax(dim=1)
    names = sorted(FRENCH_16.values())
    return [names[i] for i in k.tolist()]'''

# The five repairs, as replacements of the colleague's lines (history_chB7.py replays the 2**5 partly repaired versions);
# in the order of the diagnoses that see them: float64 (point 1), reading (2), normalization (3), eval (4), classes (5).
FIXES_16 = {
    "float64": [('    x = np.asarray(images, dtype=np.float64)                      # double precision, to be safe\n',
                 '    x = np.asarray(images, dtype=np.float32)                      # 5. float32, as in training\n')],
    "reading": [('    return wbp.imread_bgr(path)\n',
                 '    return wbp.imread_bgr(path)[..., ::-1].astype(np.float32) / 255   # 4. RGB, floats in [0, 1]\n')],
    "normalization": [('    mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))\n',
                       '    mean = np.asarray(meta["mean"], dtype=x.dtype)                # 1. the statistics of the\n'
                       '    std = np.asarray(meta["std"], dtype=x.dtype)                  #    training, from the artifact\n')],
    "eval": [('    model.load_state_dict(load_file(str(Path(folder) / "model.safetensors")))\n    return model, meta',
              '    model.load_state_dict(load_file(str(Path(folder) / "model.safetensors")))\n'
              '    model.eval()                                                  # 2. the evaluation mode\n'
              '    return model, meta')],
    "classes": [('    meta = {"arch": "vgg", "widths": [16, 32, 64], "mean": list(mean), "std": list(std)}\n',
                 '    meta = {"arch": "vgg", "widths": [16, 32, 64], "mean": list(mean), "std": list(std),\n'
                 '            "classes": list(classes)}                             # 3. the order of the outputs\n'),
                ('    names = sorted(FRENCH_16.values())\n    return [names[i] for i in k.tolist()]',
                 '    return [FRENCH_16[meta["classes"][i]] for i in k.tolist()]   # 3. the order of the artifact')],
}
FIX_ORDER_16 = ("float64", "reading", "normalization", "eval", "classes")


def repaired_16(fixed=FIX_ORDER_16) -> str:
    """The colleague's code with the repairs ``fixed`` applied (all of them: the solution)."""
    text = TODO_16.replace("# The colleague's service, to repair IN PLACE (same names, same signatures). It works in "
                           "their notebook, not in production.",
                           "# The colleague's service, repaired: five changes, numbered in the comments")
    for name in fixed:
        for old, new in FIXES_16[name]:
            if text.count(old) != 1:
                raise ValueError(f"repair {name}: {old!r} is not found once")
            text = text.replace(old, new)
    return text


SOLUTION_16 = repaired_16()

CHECK_16 = r'''repaired_16_ok = False
with checked("B7.16"):
    repaired_16_ok = diagnose_16()'''

EXPERIMENT_16 = r'''# The service, as it is now (it shows the symptoms before the repair), on 8 uploads and on the 200 images
try:
    with wbp.torch_state_kept():
        model_16, meta_16 = fresh_service_16()
        names_16 = predict_16(model_16, meta_16, PATHS_16)
except Exception as error_16:  # noqa: BLE001 - the learner's code
    names_16 = None
    print(f"⏳ Ex B7.16 : le service lève {type(error_16).__name__} ({str(error_16)[:160]}) : rien à montrer.")
if isinstance(names_16, list) and len(names_16) == len(PATHS_16):
    print("image   vraie classe   notebook   service")
    for i_16 in range(8):
        print(f"{i_16:5d}   {FRENCH_16[CLASSES[Y_16[i_16]]]:12s}   {WANT_16[i_16]:10s} {names_16[i_16]}")
    truth_16 = [FRENCH_16[CLASSES[k]] for k in Y_16]
    print(f"accuracy sur 200 images : notebook {fr((OFFLINE_16 == Y_16).mean(), 3)}, service "
          f"{fr(np.mean([a == b for a, b in zip(names_16, truth_16)]), 3)} ; accord service-notebook "
          f"{sum(a == b for a, b in zip(names_16, WANT_16))} sur 200.")'''
