"""Download, load, split and seal the dataset (steps PF.2 to PF.4).

The rest of the package only sees the dataset through ``load_examples`` (one row per example) and the frozen split
(``data/splits.json``). Nothing here is specific to a kind of data: an image is a row with the path of its file, a
text a row with its string, a time series a row per time step (or per window), a table a row per individual.

    df = load_examples(config)                    # every example, with an "id" column
    splits = make_splits(df, config)              # {"train": [...], "val": [...], "test": [...]}  (ids)
    digest = save_splits(splits, config.split_file)    # its SHA-256 goes in docs/protocole.md (PF.4)
    train = select(df, load_splits(config.split_file, digest), "train")

The test split is never selected here: ``evaluate.open_sealed_test`` opens it, once, and writes it in the test log.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config

SPLIT_NAMES = ("train", "val", "test")
ID_COLUMN = "id"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """The SHA-256 of a file, as 64 lowercase hexadecimal characters, read in chunks of ``chunk_size`` bytes.

    Take your ``sha256_15`` of Ex 29.15. FileNotFoundError if the file does not exist.
    """
    # TODO PF.2: hashlib.sha256(), update() chunk by chunk, hexdigest()
    raise NotImplementedError("TODO PF.2: sha256_file")


def download(config: Config) -> list[Path]:
    """Download the raw files of the dataset into ``config.data_dir``, and return their paths.

    Contract (Ex 29.15, the data card ``data/card.md``, section « Téléchargement »):
    - a fixed source: a URL with a fixed version, a Hugging Face revision (``revision=<commit sha>``), a Kaggle
      dataset version, an OpenML id with its version;
    - each file checked against its SHA-256, written in the data card and in this module (a constant);
    - idempotent: a file already there with the right SHA-256 is not downloaded again; a wrong one is downloaded
      again into ``<name>.part``, checked, then renamed; a wrong SHA-256 after download raises ValueError and leaves
      nothing half-written;
    - no secret in the code: an API key (Kaggle, Hugging Face) is read from the environment.
    """
    # TODO PF.2: the source, the expected SHA-256 of each file, the download into a .part file, the check, the rename
    raise NotImplementedError("TODO PF.2: download")


def load_examples(config: Config) -> pd.DataFrame:
    """Every example of the dataset, as a DataFrame with one row per example.

    Contract:
    - a column ``"id"``: unique, stable from one run to another (a file name, a row number of the raw file, an id of
      the source), never a position after a shuffle;
    - the column ``config.target`` (absent for an unsupervised or RL project: say so in docs/protocole.md);
    - the column ``config.group_column`` if it is set (examples that must stay together: a patient, a photo series, a
      user, a document), and ``config.time_column`` if it is set;
    - the inputs: feature columns, or a column with the path of a file (image, audio) or the text itself;
    - the raw data, before anything learned (no scaler, no imputation here: that is ``preprocess``, fitted on the
      training split only);
    - the same rows and the same order at every call (sort by ``id``).

    Reads the files of ``config.data_dir`` (``download`` first). Never reads the split.
    """
    # TODO PF.2: read the raw files, build the "id" column, rename the target, sort by id
    raise NotImplementedError("TODO PF.2: load_examples")


def make_splits(df: pd.DataFrame, config: Config) -> dict[str, list]:
    """The split of the examples, as ``{"train": ids, "val": ids, "test": ids}`` (lists of values of ``df["id"]``).

    Contract (written in docs/protocole.md before any experiment, PF.4):
    - the three lists are disjoint and together hold every id of ``df``;
    - their sizes follow ``config.test_size`` and ``config.val_size`` (fractions of ``len(df)``, up to rounding and to
      the size of the groups);
    - the same ``df`` and the same ``config.seed`` give the same split;
    - with ``config.group_column``: no group in two splits (GroupShuffleSplit, StratifiedGroupKFold);
    - with ``config.time_column``: every test example comes after every training and validation example (and
      validation after training); no shuffle;
    - otherwise, for a classification: stratified on ``config.target``.

    Done once (notebook 01), saved by ``save_splits`` and never redone: the experiments read ``data/splits.json``.
    """
    # TODO PF.3: choose the split from what the audit found (duplicates, groups, time); ch. 8 (8.11, 8.19), 29.13
    raise NotImplementedError("TODO PF.3: make_splits")


def save_splits(splits: dict[str, list], path: str | Path) -> str:
    """Write the split as canonical JSON and return the SHA-256 of the file.

    Canonical: keys ``train``, ``val``, ``test`` in that order, each list of ids sorted and written as strings, UTF-8,
    ``indent=1``, a final newline; the same split always gives the same bytes, hence the same SHA-256. ValueError if
    the lists are not disjoint or a key is missing. Refuses to overwrite an existing file with a different content
    (FileExistsError): a frozen split is never redone by accident.
    """
    # TODO PF.4: json.dumps(..., indent=1), write, then sha256_file
    raise NotImplementedError("TODO PF.4: save_splits")


def load_splits(path: str | Path, expected_sha256: str | None = None) -> dict[str, list[str]]:
    """Read the split written by ``save_splits``; with ``expected_sha256``, check the file first (ValueError if it
    differs: the split changed since docs/protocole.md was frozen). The ids come back as strings."""
    # TODO PF.4: check the SHA-256, then json.loads
    raise NotImplementedError("TODO PF.4: load_splits")


def select(df: pd.DataFrame, splits: dict[str, list], name: str, *, _sealed: bool = False) -> pd.DataFrame:
    """The rows of ``df`` whose id is in ``splits[name]``, in the order of ``df``, with a fresh index.

    ``name`` is "train" or "val"; "test" raises PermissionError unless ``_sealed=True``, which only
    ``evaluate.open_sealed_test`` passes. ValueError for another name, or if an id of the split is not in ``df`` (the
    dataset changed). Compare the ids as strings (``df["id"].astype(str)``).
    """
    # TODO PF.4: the PermissionError first, then df[df["id"].astype(str).isin(...)]
    raise NotImplementedError("TODO PF.4: select")


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At most ``n`` rows of ``df`` drawn without replacement with ``seed`` (``n = 0``: all of them), stratified on
    the target for a classification if you can; used by the fast mode (``config.max_train_examples``)."""
    # TODO PF.5: df.sample(n, random_state=seed) or a stratified draw; keep the order of df
    raise NotImplementedError("TODO PF.5: subsample")
