"""Seeds, cross-validation, the training loop, checkpoints and the run journal (steps PF.5 to PF.7).

    set_seed(config.seed)
    scores = cross_validate(build_classical(config), train_df, config)      # PF.5
    result = fit(net, train_ds, val_ds, config)                              # PF.6
    log_run(config.runs_file, {"run_id": new_run_id(), "config": config.to_dict(), ...})   # PF.7

Every run goes in the journal ``experiments/runs.jsonl``, the failed ones too (B7.14): the summary table of the
report is generated from it by ``scripts/summarize_runs.py``, never written by hand.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import Config


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix the seeds of ``random``, NumPy and PyTorch (CPU and GPU); with ``deterministic``, also ask PyTorch for
    deterministic algorithms (``torch.use_deterministic_algorithms(True, warn_only=True)``, cuDNN benchmark off).
    Same seed, same weights and same batches (B7.12, B7.13)."""
    # TODO PF.6: random.seed, np.random.seed, torch.manual_seed (+ cuda), the deterministic switches
    raise NotImplementedError("TODO PF.6: set_seed")


def cross_validate(estimator, df, config: Config) -> dict:
    """The cross-validation of an UNFITTED scikit-learn estimator (a Pipeline: ``models.build_classical``) on ``df``,
    the training examples only (never the test; validation too if the protocol says so).

    The folds follow the protocol: ``GroupKFold`` with ``config.group_column``, ``TimeSeriesSplit`` with
    ``config.time_column``, otherwise ``StratifiedKFold`` (classification) or ``KFold``, with ``config.seed``
    (8.19, 15.17). Returns ``{"metric": config.metric, "scores": [one per fold], "mean": ..., "std": ...}``; the
    same estimator and the same ``df`` give the same scores.
    """
    # TODO PF.5: sklearn.model_selection.cross_validate with the folds of the protocol and the metric of config.metric
    raise NotImplementedError("TODO PF.5: cross_validate")


def fit(model, train_ds, val_ds, config: Config) -> dict:
    """Train ``model`` on ``train_ds``, choose the best epoch on ``val_ds``, and return
    ``{"model": the model at its best epoch, "history": {"train_loss": [...], "val_loss": [...], "val_metric": [...]},
    "best_epoch": int, "best_score": float, "checkpoint": path of best.pt, "seconds": float}``.

    Contract: AdamW with ``config.lr`` and ``config.weight_decay`` (or the optimizer your model needs, 19.18); a
    learning rate schedule if it helps (19.19, 23.26); ``model.train()`` then ``model.eval()`` and
    ``torch.inference_mode()`` for validation; early stopping after ``config.patience`` epochs without improvement;
    the best checkpoint in ``config.checkpoint_dir`` (``save_checkpoint``); at most ``config.epochs`` epochs; the
    device of ``config.device`` (``"auto"``: cuda, then mps, then cpu). Never reads the test.
    """
    # TODO PF.6: take your fit() of ch. 23 (23.18, 23.25) or of MP5, then make it read config
    raise NotImplementedError("TODO PF.6: fit")


def save_checkpoint(path: str | Path, model, optimizer=None, *, epoch: int, extra: dict | None = None) -> Path:
    """Save everything needed to resume a training exactly: the weights, the optimizer (and the scheduler, the
    random states of torch and NumPy if you resume mid-run), the epoch and ``extra`` (the config, the metrics).
    Write to ``<path>.tmp`` then rename: an interrupted save leaves the previous checkpoint intact (23.23)."""
    # TODO PF.6: torch.save({...}, tmp); tmp.replace(path)
    raise NotImplementedError("TODO PF.6: save_checkpoint")


def load_checkpoint(path: str | Path, model, optimizer=None) -> dict:
    """Load a checkpoint written by ``save_checkpoint`` into ``model`` (and ``optimizer``) and return the rest of its
    content (``epoch``, ``extra``). ``torch.load(..., weights_only=True)`` if your checkpoint allows it."""
    # TODO PF.6: torch.load, model.load_state_dict, optimizer.load_state_dict
    raise NotImplementedError("TODO PF.6: load_checkpoint")


def new_run_id() -> str:
    """A unique, sortable run id: the UTC time and 6 random hex digits, e.g. ``20261105T142233-3fa2c1``. Provided."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def environment() -> dict:
    """What makes a run reproducible besides its config: Python, the platform, the versions of the main libraries
    (those installed), the git commit and whether the working tree had uncommitted changes. Provided."""
    versions = {}
    for name in ("numpy", "pandas", "sklearn", "torch", "torchvision", "transformers", "peft", "gymnasium"):
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name)
            except ImportError:
                continue
        versions[name] = getattr(module, "__version__", "?")

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions,
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def log_run(path: str | Path, record: dict) -> None:
    """Append ``record`` to the journal ``path`` as ONE line of JSON (JSONL), creating the folder if needed.

    A record holds at least ``run_id``, ``started`` (UTC, ISO 8601), ``status`` (``"ok"`` or ``"failed"``),
    ``config`` (``config.to_dict()``), ``environment`` (``environment()``), ``metrics`` (validation only, never the
    test) and ``seconds``; a failed run also has ``error``. Values that JSON refuses (NumPy numbers, paths) are
    converted, a NaN is written as null; one line per run, never rewritten (B7.14).
    """
    # TODO PF.7: json.dumps(record, default=...), mode "a", a final "\n"
    raise NotImplementedError("TODO PF.7: log_run")


def read_runs(path: str | Path) -> list[dict]:
    """Every record of the journal, in the order of the file; ``[]`` if the file does not exist yet. A damaged line
    raises ValueError with its line number (a journal is never silently truncated)."""
    # TODO PF.7: one json.loads per non-empty line
    raise NotImplementedError("TODO PF.7: read_runs")


def _json_default(value):
    """Helper for json.dumps: NumPy scalars and arrays, paths. Provided (use it in log_run if you wish)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def dumps(record: dict) -> str:
    """``json.dumps`` with ``_json_default`` and ``allow_nan=False`` after replacing NaN by None. Provided."""
    def clean(obj):
        if isinstance(obj, float) and obj != obj:
            return None
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    return json.dumps(clean(record), default=_json_default, ensure_ascii=False, allow_nan=False)
