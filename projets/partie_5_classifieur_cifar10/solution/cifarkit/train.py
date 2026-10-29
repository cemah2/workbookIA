"""The training of cifarkit: one epoch, the measure of a model, a reusable fit() (best checkpoint, early stopping,
learning-rate schedule, exact resumption), the saved model with its preprocessing, the experiment journal, and the
FGSM attack (chapters 21, 23 and 24).

    history = fit(model, train_loader, val_loader, epochs=30, lr=0.1, optimizer="sgd", schedule="onecycle",
                  label_smoothing=0.1, folder=Path("runs/resnet"))
    save_model(Path("."), model, config, mean, std, {"val_acc": max(history["val_acc"])})
    model, info = load_model(Path("."))
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
    was_training = model.training
    model.eval()
    total, correct, count = 0.0, 0, 0
    try:
        with torch.inference_mode():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                total += F.cross_entropy(logits, y, reduction="sum").item()
                correct += (logits.argmax(1) == y).sum().item()
                count += len(y)
    finally:
        model.train(was_training)
    return total / count, correct / count


def train_one_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer, scheduler=None,
                    label_smoothing: float = 0.0, device: str = "cpu") -> tuple[float, float]:
    """One pass over the loader in training mode: for each mini-batch, the cross-entropy of the logits (with
    ``label_smoothing``), zero_grad, backward, step, then ``scheduler.step()`` if a scheduler is given (it counts in
    steps). Returns (the mean loss per example, the accuracy), measured during the epoch."""
    model.train()
    total, correct, count = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return total / count, correct / count


def make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """name "sgd": ``torch.optim.SGD(..., lr, momentum=0.9, nesterov=True, weight_decay)``; name "adamw":
    ``torch.optim.AdamW(..., lr, weight_decay)``; ValueError otherwise. Only the parameters that require a gradient
    are given to the optimizer."""
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}: expected 'sgd' or 'adamw'")


def make_scheduler(optimizer: torch.optim.Optimizer, name: str, lr: float, total_steps: int):
    """The learning-rate schedule, stepped once per mini-batch: "onecycle" (``OneCycleLR(optimizer, max_lr=lr,
    total_steps=total_steps)``), "cosine" (``CosineAnnealingLR(optimizer, T_max=total_steps)``), or None for
    "constant"; ValueError for another name."""
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if name == "constant":
        return None
    raise ValueError(f"unknown schedule {name!r}: expected 'onecycle', 'cosine' or 'constant'")


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
    model.to(device)
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    sched = make_scheduler(opt, schedule, lr, epochs * len(train_loader))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "seconds": [],
               "best_epoch": None}
    best_acc, best_state, waited, start = -1.0, None, 0, 1
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
    if resume:
        if folder is None or not (folder / "last.pt").exists():
            raise FileNotFoundError("resume=True needs folder / 'last.pt', written by an earlier fit with this folder")
        state = torch.load(folder / "last.pt", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if sched is not None:
            sched.load_state_dict(state["scheduler"])
        history, best_acc, best_state, waited = state["history"], state["best_acc"], state["best_state"], state["waited"]
        torch.set_rng_state(state["torch_rng"])
        if getattr(train_loader, "generator", None) is not None and state["loader_rng"] is not None:
            train_loader.generator.set_state(state["loader_rng"])
        start = state["epoch"] + 1
    for epoch in range(start, epochs + 1):
        if (patience is not None and waited >= patience) or (stop_after is not None and epoch > stop_after):
            break
        started = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, sched, label_smoothing, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        for key, value in (("train_loss", train_loss), ("train_acc", train_acc), ("val_loss", val_loss),
                           ("val_acc", val_acc), ("lr", opt.param_groups[0]["lr"]),
                           ("seconds", time.perf_counter() - started)):
            history[key].append(value)
        if val_acc > best_acc:
            best_acc, waited, history["best_epoch"] = val_acc, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
            if folder is not None:
                torch.save(best_state, folder / "best.pt")
        else:
            waited += 1
        if folder is not None:
            generator = getattr(train_loader, "generator", None)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict() if sched is not None else None, "epoch": epoch,
                        "history": history, "best_acc": best_acc, "best_state": best_state, "waited": waited,
                        "torch_rng": torch.get_rng_state(),
                        "loader_rng": generator.get_state() if generator is not None else None}, folder / "last.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_model(folder: Path, model: nn.Module, config: dict, mean: list[float], std: list[float], metrics: dict) -> None:
    """The delivered model: ``folder / "model.pt"`` (the state_dict only) and ``folder / "model.json"`` (the config
    that build_model rebuilds it from, the normalization mean and std, the class names, the metrics, the date and
    library_versions())."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "model.pt")
    info = {"config": dict(config), "mean": [float(m) for m in mean], "std": [float(s) for s in std],
            "classes": CLASSES, "metrics": metrics, "date": dt.date.today().isoformat(), "versions": library_versions()}
    (folder / "model.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_model(folder: Path) -> tuple[nn.Module, dict]:
    """(model, info) from save_model's two files: the model rebuilt by build_model(info["config"]), its weights loaded
    with ``torch.load(..., weights_only=True)``, in evaluation mode; info is the content of model.json."""
    folder = Path(folder)
    info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    model = build_model(info["config"])
    model.load_state_dict(torch.load(folder / "model.pt", weights_only=True))
    model.eval()
    return model, info


def predict_proba(model: nn.Module, images: np.ndarray, mean: list[float], std: list[float],
                  batch_size: int = 500) -> np.ndarray:
    """The probabilities (N, 10) of uint8 images (N, 32, 32, 3): scaled to [0, 1], put in (N, 3, 32, 32), normalized
    with mean and std, then the softmax of the logits, in evaluation mode and without a graph."""
    model.eval()
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(np.asarray(images[i:i + batch_size], dtype=np.float32) / 255.0).permute(0, 3, 1, 2)
            out.append(torch.softmax(model((x - m) / s), dim=1))
    return torch.cat(out).numpy()


def fgsm(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float, mean: list[float],
         std: list[float]) -> torch.Tensor:
    """The FGSM attack (chapter 21) in pixel space: images are float tensors (N, 3, H, W) in [0, 1]; the loss is the
    cross-entropy of model((x - mean) / std), in evaluation mode; returns ``clamp(x + epsilon * sign(dL/dx), 0, 1)``,
    detached. The gradients of the model's parameters are not touched (``torch.autograd.grad`` on x only)."""
    model.eval()
    m = torch.tensor(mean, dtype=images.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=images.dtype).view(1, 3, 1, 1)
    x = images.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model((x - m) / s), labels)
    (grad,) = torch.autograd.grad(loss, x)
    return (x + epsilon * grad.sign()).clamp(0.0, 1.0).detach()


def log_run(path: Path, row: dict) -> None:
    """Append one run to the CSV journal path (its header, RUN_COLUMNS, is written when the file is new). The keys of
    row must be columns of RUN_COLUMNS (a missing one stays empty); ValueError for an unknown key. A dict or list value
    is written as JSON."""
    unknown = set(row) - set(RUN_COLUMNS)
    if unknown:
        raise ValueError(f"unknown columns {sorted(unknown)}: the journal has {RUN_COLUMNS}")
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})


def read_runs(path: Path) -> list[dict]:
    """Provided: the rows of the CSV journal, as dictionaries of strings (an empty list if it does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
