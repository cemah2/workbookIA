"""The training of cifarkit: one epoch, the measure of a model, a reusable fit() (best checkpoint, early stopping,
learning-rate schedule, exact resumption), the saved model with its preprocessing, the experiment journal, and the
FGSM attack (chapters 21, 23 and 24).

    history = fit(model, train_loader, val_loader, epochs=30, lr=0.1, optimizer="sgd", schedule="onecycle",
                  label_smoothing=0.1, folder=Path("runs/resnet"))
    save_model(Path("."), model, config, mean, std, {"val_acc": max(history["val_acc"])})
    model, info = load_model(Path("."))

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import build_model
from .pipeline import CLASSES

RUN_COLUMNS = ["run", "date", "arch", "config", "n_train", "epochs", "optimizer", "lr", "weight_decay", "schedule",
               "augment", "label_smoothing", "seed", "params", "best_epoch", "val_acc", "val_loss", "seconds", "note"]


def library_versions() -> dict:
    """Provided: the versions of Python, NumPy, PyTorch and torchvision that run the project."""
    import torchvision
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "torchvision": torchvision.__version__}


def quick_fit(model: nn.Module, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 3,
              lr: float = 1e-3, batch_size: int = 128, seed: int = 0) -> dict:
    """Provided: the training of the baselines of MP5.1, before the pipeline and fit() exist. uint8 images (N, 32, 32,
    3) are scaled to [-1, 1] and put in (N, 3, 32, 32); Adam(lr), mini-batches in the order of
    ``torch.Generator().manual_seed(seed)``, no augmentation. Returns {"train_loss", "val_loss", "val_acc", "seconds"},
    one value per epoch (the validation in evaluation mode)."""
    def prepare(images):
        return torch.from_numpy(np.asarray(images, dtype=np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).contiguous()

    x, t = prepare(X), torch.from_numpy(np.asarray(y, dtype=np.int64))
    x_val, t_val = prepare(X_val), torch.from_numpy(np.asarray(y_val, dtype=np.int64))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "seconds": []}
    for _ in range(epochs):
        started, total = time.perf_counter(), 0.0
        model.train()
        for idx in torch.randperm(len(x), generator=generator).split(batch_size):
            loss = F.cross_entropy(model(x[idx]), t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(x_val[i:i + 1000]) for i in range(0, len(x_val), 1000)])
        history["train_loss"].append(total / len(x))
        history["val_loss"].append(F.cross_entropy(logits, t_val).item())
        history["val_acc"].append((logits.argmax(1) == t_val).float().mean().item())
        history["seconds"].append(time.perf_counter() - started)
    return history


def evaluate(model: nn.Module, loader, device: str = "cpu") -> tuple[float, float]:
    """(mean loss per example, accuracy) of the model over the whole loader, in evaluation mode and without a graph
    (``torch.inference_mode``); the cross-entropy is summed over the examples, then divided by their number, so that a
    smaller last mini-batch weighs what it should. The model goes back to its previous mode afterwards."""
    raise NotImplementedError("evaluate")  # TODO MP5.3


def train_one_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer, scheduler=None,
                    label_smoothing: float = 0.0, device: str = "cpu") -> tuple[float, float]:
    """One pass over the loader in training mode: for each mini-batch, the cross-entropy of the logits (with
    ``label_smoothing``), zero_grad, backward, step, then ``scheduler.step()`` if a scheduler is given (it counts in
    steps). Returns (the mean loss per example, the accuracy), measured during the epoch."""
    raise NotImplementedError("train_one_epoch")  # TODO MP5.3


def make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """name "sgd": ``torch.optim.SGD(..., lr, momentum=0.9, nesterov=True, weight_decay)``; name "adamw":
    ``torch.optim.AdamW(..., lr, weight_decay)``; ValueError otherwise. Only the parameters that require a gradient
    are given to the optimizer."""
    raise NotImplementedError("make_optimizer")  # TODO MP5.3


def make_scheduler(optimizer: torch.optim.Optimizer, name: str, lr: float, total_steps: int):
    """The learning-rate schedule, stepped once per mini-batch: "onecycle" (``OneCycleLR(optimizer, max_lr=lr,
    total_steps=total_steps)``), "cosine" (``CosineAnnealingLR(optimizer, T_max=total_steps)``), or None for
    "constant"; ValueError for another name."""
    raise NotImplementedError("make_scheduler")  # TODO MP5.3


def fit(model: nn.Module, train_loader, val_loader, *, epochs: int, lr: float, optimizer: str = "sgd",
        weight_decay: float = 5e-4, schedule: str = "onecycle", label_smoothing: float = 0.0, patience: int | None = None,
        folder: Path | None = None, resume: bool = False, stop_after: int | None = None, device: str = "cpu") -> dict:
    """Train the model for at most ``epochs`` epochs and keep the weights of its best epoch.

    - The optimizer and the schedule come from make_optimizer and make_scheduler (total_steps = epochs times the
      number of mini-batches of train_loader); each epoch is train_one_epoch, then evaluate on val_loader.
    - The best epoch is the one with the highest validation accuracy (a STRICT improvement); its weights are kept
      (a deep copy of the state_dict, and ``folder / "best.pt"`` when folder is given). With ``patience``, training
      stops once that many epochs in a row have not improved.
    - When folder is given, ``folder / "last.pt"`` holds, after each epoch, everything needed to resume exactly: the
      model, the optimizer, the scheduler, the epoch, the history, the best accuracy and weights, the patience counter,
      the global RNG state of PyTorch and the state of ``train_loader.generator``. With ``resume=True``, fit starts from
      it and goes on up to ``epochs``. ``stop_after=k`` stops after epoch k, as an interrupted run would (the
      schedule still spans ``epochs``): fit(epochs=4, stop_after=2), then fit(epochs=4, resume=True), give the same
      weights as 4 epochs in a row.
    - At the end, the model gets the weights of the best epoch.

    Returns the history: lists "train_loss", "train_acc", "val_loss", "val_acc", "lr" (at the end of each epoch),
    "seconds" (one value per epoch done), and "best_epoch" (counted from 1).
    """
    raise NotImplementedError("fit")  # TODO MP5.3


def save_model(folder: Path, model: nn.Module, config: dict, mean: list[float], std: list[float], metrics: dict) -> None:
    """The delivered model: ``folder / "model.pt"`` (the state_dict only) and ``folder / "model.json"`` (the config
    that build_model rebuilds it from, the normalization mean and std, the class names, the metrics, the date and
    library_versions())."""
    raise NotImplementedError("save_model")  # TODO MP5.5


def load_model(folder: Path) -> tuple[nn.Module, dict]:
    """(model, info) from save_model's two files: the model rebuilt by build_model(info["config"]), its weights loaded
    with ``torch.load(..., weights_only=True)``, in evaluation mode; info is the content of model.json."""
    raise NotImplementedError("load_model")  # TODO MP5.5


def predict_proba(model: nn.Module, images: np.ndarray, mean: list[float], std: list[float],
                  batch_size: int = 500) -> np.ndarray:
    """The probabilities (N, 10) of uint8 images (N, 32, 32, 3): scaled to [0, 1], put in (N, 3, 32, 32), normalized
    with mean and std, then the softmax of the logits, in evaluation mode and without a graph."""
    raise NotImplementedError("predict_proba")  # TODO MP5.6


def fgsm(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float, mean: list[float],
         std: list[float]) -> torch.Tensor:
    """The FGSM attack (chapter 21) in pixel space: images are float tensors (N, 3, H, W) in [0, 1]; the loss is the
    cross-entropy of model((x - mean) / std), in evaluation mode; returns ``clamp(x + epsilon * sign(dL/dx), 0, 1)``,
    detached. The gradients of the model's parameters are not touched (``torch.autograd.grad`` on x only)."""
    raise NotImplementedError("fgsm")  # TODO MP5.6


def log_run(path: Path, row: dict) -> None:
    """Append one run to the CSV journal path (its header, RUN_COLUMNS, is written when the file is new). The keys of
    row must be columns of RUN_COLUMNS (a missing one stays empty); ValueError for an unknown key. A dict or list value
    is written as JSON."""
    raise NotImplementedError("log_run")  # TODO MP5.3


def read_runs(path: Path) -> list[dict]:
    """Provided: the rows of the CSV journal, as dictionaries of strings (an empty list if it does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
