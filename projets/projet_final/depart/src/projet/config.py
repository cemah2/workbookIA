"""The settings of the project: a typed, frozen dataclass, read from a YAML file and overridden on the command line.

    config = load_config("configs/fast.yaml", ["lr=0.001", "model_params={hidden: 64}"])
    config.lr, config.to_dict()

Provided (from the B7 package): every setting is checked when the configuration is built, its type, its range and no
unknown key (a typo fails instead of being ignored). Add the settings of YOUR project as new fields with a default
(TODO PF.10), or under ``model_params`` / ``data_params`` while they still move. A secret (an API key) never goes
here: a configuration is versioned and copied into the run journal.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

TASKS = ("classification", "regression", "forecasting", "generation", "rl")
DEVICES = ("auto", "cpu", "cuda", "mps")


@dataclass(frozen=True)
class Config:
    """All the settings of one run. ``fast`` = the CPU mode of a few minutes (configs/fast.yaml)."""

    name: str = "projet"
    seed: int = 0
    fast: bool = True
    task: str = "classification"
    # data (PF.2 to PF.4)
    data_dir: str = "data/raw"                  # downloads, never in git (.gitignore)
    split_file: str = "data/splits.json"        # the frozen split, in git, its SHA-256 in docs/protocole.md
    target: str = "label"                       # the column of the target in load_examples
    group_column: str = ""                      # "" = no groups (otherwise: a patient, a photo series, a user...)
    time_column: str = ""                       # "" = no time order (otherwise: the test comes after the training)
    test_size: float = 0.2
    val_size: float = 0.1
    max_train_examples: int = 0                 # 0 = all; the fast mode trains on a subsample
    data_params: dict = field(default_factory=dict)
    # evaluation (PF.4, PF.8)
    metric: str = "f1_macro"                    # the main technical metric, written in docs/protocole.md
    n_bootstrap: int = 1000
    # model and training (PF.5 to PF.7)
    model: str = "deep"                         # a name understood by models.build_model
    model_params: dict = field(default_factory=dict)
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 3                           # early stopping, in epochs without improvement on validation
    device: str = "auto"
    num_workers: int = 0
    deterministic: bool = True
    # outputs
    out_dir: str = "artifacts/model"            # the delivered artifact (PF.11); heavy: not in git by default
    checkpoint_dir: str = "checkpoints"         # never in git
    runs_file: str = "experiments/runs.jsonl"   # the run journal (PF.7), in git
    test_log: str = "experiments/test_access.jsonl"   # every opening of the sealed test (PF.8), in git

    def __post_init__(self) -> None:
        for f in fields(self):                          # a dataclass does not check the types: we do
            value = getattr(self, f.name)
            default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
            kind = type(default)
            if kind is float and type(value) is int:
                object.__setattr__(self, f.name, float(value))          # 1 -> 1.0 is fine
            elif kind is dict and value is None:
                object.__setattr__(self, f.name, {})                    # an empty YAML mapping
            elif type(value) is not kind:                               # True is not an int here
                raise ValueError(f"{f.name}: expected {kind.__name__}, got {value!r}")
        if self.task not in TASKS:
            raise ValueError(f"task: expected one of {TASKS}, got {self.task!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device: expected one of {DEVICES}, got {self.device!r}")
        if self.seed < 0 or self.epochs < 1 or self.batch_size < 1 or self.patience < 1 or self.num_workers < 0:
            raise ValueError(f"need seed >= 0, epochs >= 1, batch_size >= 1, patience >= 1 and num_workers >= 0, got "
                             f"{self.seed}, {self.epochs}, {self.batch_size}, {self.patience} and {self.num_workers}")
        if self.max_train_examples < 0 or self.n_bootstrap < 1:
            raise ValueError(f"need max_train_examples >= 0 and n_bootstrap >= 1, got {self.max_train_examples} "
                             f"and {self.n_bootstrap}")
        if not (0 < self.test_size < 1 and 0 <= self.val_size < 1 and self.test_size + self.val_size < 1):
            raise ValueError(f"need 0 < test_size < 1, 0 <= val_size < 1 and test_size + val_size < 1, got "
                             f"{self.test_size} and {self.val_size}")
        if not 0 < self.lr < 1 or self.weight_decay < 0:
            raise ValueError(f"need 0 < lr < 1 and weight_decay >= 0, got {self.lr} and {self.weight_decay}")

    def to_dict(self) -> dict:
        """The settings as a plain dictionary that json.dumps accepts."""
        return asdict(self)


def load_config(path: str | Path | None = None, overrides=()) -> Config:
    """The Config of the YAML file ``path`` (None: the defaults), then of the overrides "KEY=VALUE" (each VALUE read by
    yaml.safe_load: ``epochs=5``, ``model_params={hidden: 64}``).

    ValueError for a file that is not a mapping, an override without "=", an unknown key or a setting refused by
    Config; FileNotFoundError for a missing file.
    """
    raw: dict = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"{path}: expected a mapping of settings, got {type(loaded).__name__}")
        raw.update(loaded or {})
    for item in overrides:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        raw[key.strip()] = yaml.safe_load(value)
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return Config(**raw)


def default_config_path(fast: bool, root: str | Path = ".") -> Path:
    """``configs/fast.yaml`` or ``configs/full.yaml`` under ``root`` (the notebooks choose it from FAST_MODE)."""
    return Path(root) / "configs" / ("fast.yaml" if fast else "full.yaml")
