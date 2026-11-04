"""The configuration of cifarprod: a typed, frozen dataclass, read from a YAML file and overridden on the command line.

    config = load_config("configs/fast.yaml", ["lr=0.01", "widths=[8, 16]"])
    config.lr, config.to_dict()

Every setting is checked when the configuration is built: its type, its range, and no unknown key (a typo fails
instead of being ignored). A secret never goes here: a configuration is versioned and copied into the run journal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

ARCHS = ("cnn", "vgg")


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

    def __post_init__(self) -> None:
        for f in fields(self):                          # a dataclass does not check the types: we do
            value, kind = getattr(self, f.name), type(f.default)
            if kind is float and type(value) is int:
                object.__setattr__(self, f.name, float(value))          # 1 -> 1.0 is fine
            elif kind is tuple and type(value) is list:
                object.__setattr__(self, f.name, tuple(value))          # a YAML list -> a tuple
            elif type(value) is not kind:                               # True is not an int here
                raise ValueError(f"{f.name}: expected {kind.__name__}, got {value!r}")
        if not self.widths or any(type(w) is not int or w < 1 for w in self.widths):
            raise ValueError(f"widths: expected positive whole numbers, got {self.widths!r}")
        if self.arch not in ARCHS:
            raise ValueError(f"arch: expected one of {ARCHS}, got {self.arch!r}")
        if self.seed < 0 or self.epochs < 1 or self.batch_size < 1 or self.n_train < 1 or self.num_workers < 0:
            raise ValueError(f"need seed >= 0, epochs >= 1, batch_size >= 1, n_train >= 1 and num_workers >= 0, got "
                             f"{self.seed}, {self.epochs}, {self.batch_size}, {self.n_train} and {self.num_workers}")
        if not 0 < self.lr < 1:
            raise ValueError(f"lr: expected 0 < lr < 1, got {self.lr}")

    def to_dict(self) -> dict:
        """The settings as a plain dictionary that json.dumps accepts (widths as a list)."""
        out = asdict(self)
        out["widths"] = list(self.widths)
        return out


def load_config(path: str | Path | None = None, overrides=()) -> TrainConfig:
    """The TrainConfig of the YAML file ``path`` (None: the defaults), then of the overrides "KEY=VALUE" (each VALUE
    read by yaml.safe_load). ValueError for a file that is not a mapping, an override without "=", an unknown key or a
    setting refused by TrainConfig; FileNotFoundError for a missing file."""
    raw = {}
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
    unknown = set(raw) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return TrainConfig(**raw)
