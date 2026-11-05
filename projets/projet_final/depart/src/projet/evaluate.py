"""Metrics, bootstrap intervals, subgroups, errors, and the one opening of the sealed test (steps PF.8 and PF.9).

    test_df = open_sealed_test(config, reason="final evaluation of the model of run 20261201T...")   # once
    metrics = compute_metrics(y_true, y_pred, config, y_score=proba)
    low, high = bootstrap_ci(y_true, y_pred, metric_fn, n_boot=config.n_bootstrap, seed=config.seed)
    diff = paired_bootstrap(y_true, y_pred_model, y_pred_baseline, metric_fn)                         # vs the baseline

Every number of the « Résultats » section comes with its interval and next to the naive and classical baselines.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from . import data
from .config import Config

PROTOCOL = "docs/protocole.md"
PROTOCOL_TAG = "protocole"
SPLIT_DIGEST = re.compile(r"SHA-256 du découpage\s*:\s*`?([0-9a-f]{64})`?")


def compute_metrics(y_true, y_pred, config: Config, *, y_score=None) -> dict:
    """The metrics of the protocol, as ``{name: float}``: the main metric ``config.metric`` first, then the secondary
    ones written in docs/protocole.md.

    Classification: accuracy, ``f1_macro``, the precision and recall of the class that matters, ROC-AUC and PR-AUC if
    ``y_score`` is given (3.16, 3.24 to 3.26), and the calibration if decisions read the probabilities (3.28);
    regression and forecasting: MAE, RMSE, and a relative error that a user understands. Use ``sklearn.metrics``:
    their definitions are the ones a reader will check.
    """
    # TODO PF.8: a dictionary of sklearn.metrics; the key config.metric must be there
    raise NotImplementedError("TODO PF.8: compute_metrics")


def bootstrap_ci(y_true, y_pred, metric_fn: Callable, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                 groups=None) -> tuple[float, float]:
    """The percentile bootstrap interval at level ``1 - alpha`` of ``metric_fn(y_true, y_pred)``: ``n_boot`` resamples
    of the examples with replacement (2.22, 2.24), drawn with ``numpy.random.default_rng(seed)``.

    With ``groups``, resample whole groups (a cluster bootstrap): examples of one patient or one series are not
    independent, and resampling them one by one gives an interval too narrow. Returns ``(low, high)``; same seed,
    same interval.
    """
    # TODO PF.8: rng.integers(0, n, size=n) n_boot times, np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    raise NotImplementedError("TODO PF.8: bootstrap_ci")


def paired_bootstrap(y_true, pred_a, pred_b, metric_fn: Callable, *, n_boot: int = 1000, alpha: float = 0.05,
                     seed: int = 0, groups=None) -> dict:
    """Compare two models on the SAME test examples: ``metric(a) - metric(b)`` on each resample (the same indices for
    both, 8.26), and return ``{"diff": observed difference, "low": ..., "high": ..., "p_a_not_better": share of
    resamples where the difference is <= 0}``. An interval that contains 0 does not show a difference."""
    # TODO PF.8: as bootstrap_ci, with one draw of indices for both predictions
    raise NotImplementedError("TODO PF.8: paired_bootstrap")


def subgroup_table(y_true, y_pred, groups, metric_fn: Callable, *, min_size: int = 30, n_boot: int = 1000,
                   seed: int = 0) -> pd.DataFrame:
    """One row per value of ``groups`` (a subgroup: a class, a source, a period, a sensitive attribute if you have
    one and the right to use it): ``n``, the metric, its bootstrap interval, and ``small = n < min_size`` (a metric on
    15 examples is noise). Sorted by metric, worst first (B6.23 to B6.25)."""
    # TODO PF.9: a loop over the values of groups, bootstrap_ci for each
    raise NotImplementedError("TODO PF.9: subgroup_table")


def worst_errors(df: pd.DataFrame, y_true, y_pred, *, y_score=None, k: int = 20) -> pd.DataFrame:
    """The ``k`` errors to read first: for a classification, the wrong predictions with the highest confidence
    (``y_score`` of the predicted class); for a regression, the largest absolute errors. Returns the rows of ``df``
    with the columns ``y_true``, ``y_pred`` and ``score`` or ``abs_error`` added (23.20, MP5)."""
    # TODO PF.8: a mask of the errors, a sort, head(k)
    raise NotImplementedError("TODO PF.8: worst_errors")


# --------------------------------------------------------------------------------------------- the sealed test
def _git(root: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None


def protocol_status(root: str | Path = ".") -> dict:
    """What git says about the protocol: the commit of the tag ``protocole`` (None if there is no tag), the commit of
    HEAD, whether docs/protocole.md changed since the tag, and whether the protocol or the split file have changes
    not committed yet. Provided."""
    root = Path(root)
    tag = _git(root, "rev-list", "-n", "1", PROTOCOL_TAG)
    head = _git(root, "rev-parse", "HEAD")
    tag_commit = tag.stdout.strip() if tag is not None and tag.returncode == 0 else None
    changed = None
    if tag_commit:
        diff = _git(root, "diff", "--quiet", PROTOCOL_TAG, "HEAD", "--", PROTOCOL)
        changed = None if diff is None else diff.returncode != 0
    dirty = _git(root, "status", "--porcelain", "--", PROTOCOL, "data/splits.json")
    return {"tag_commit": tag_commit,
            "head_commit": head.stdout.strip() if head is not None and head.returncode == 0 else None,
            "protocol_changed_since_tag": changed,
            "uncommitted": None if dirty is None or dirty.returncode != 0 else bool(dirty.stdout.strip())}


def sealed_test_accesses(config: Config, root: str | Path = ".") -> list[dict]:
    """Every opening of the sealed test written in ``config.test_log``, oldest first (``[]`` if none). Provided."""
    path = Path(root) / config.test_log
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def open_sealed_test(config: Config, *, reason: str, root: str | Path = ".", allow_reopen: bool = False):
    """Open the sealed test split, ONCE, and write the opening in ``config.test_log``. Provided.

    1. reads the SHA-256 of the split in docs/protocole.md (the line « SHA-256 du découpage : `<64 hex>` ») and loads
       ``config.split_file`` with it (``data.load_splits``: ValueError if the split changed);
    2. loads the examples and selects the test rows (``data.load_examples``, ``data.select``);
    3. refuses a second opening (RuntimeError) unless ``allow_reopen=True``, which is written in the log: say it, and
       why, in the report (BIBLE: a test seen twice is a validation set);
    4. appends one line to the test log: the UTC time, ``reason``, the number of test examples, the SHA-256 of the
       split, the git commit, the commit of the tag ``protocole`` and whether the protocol changed since the tag.

    Nothing is written if a step fails (a function not written yet, a wrong split): the test stays sealed. Commit the
    test log right after: it is the proof that the test was opened once, after the protocol.
    """
    root = Path(root)
    if not reason or not reason.strip():
        raise ValueError("reason: say why the test is opened (e.g. « final evaluation of run <id> »)")
    protocol = root / PROTOCOL
    found = SPLIT_DIGEST.search(protocol.read_text(encoding="utf-8")) if protocol.exists() else None
    if found is None:
        raise RuntimeError(f"{protocol}: no line « SHA-256 du découpage : `<64 hex>` »; freeze the protocol first (PF.4)")
    digest = found.group(1)
    split_path = Path(config.split_file)
    splits = data.load_splits(split_path if split_path.is_absolute() else root / split_path, digest)
    test = data.select(data.load_examples(config), splits, "test", _sealed=True)
    previous = sealed_test_accesses(config, root)
    if previous and not allow_reopen:
        raise RuntimeError(f"the sealed test was already opened {len(previous)} time(s) (first: "
                           f"{previous[0].get('opened')}); reopening it needs allow_reopen=True and a line in the report")
    status = protocol_status(root)
    record = {"opened": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": reason.strip(),
              "n_examples": int(len(test)), "split_sha256": digest, "git_commit": status["head_commit"],
              "protocol_tag_commit": status["tag_commit"],
              "protocol_changed_since_tag": status["protocol_changed_since_tag"],
              "uncommitted_protocol_or_split": status["uncommitted"], "reopening": bool(previous)}
    log = root / config.test_log
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if status["tag_commit"] is None:
        print("⚠️ no git tag « protocole »: the protocol was not frozen before the experiments (PF.4)")
    elif status["protocol_changed_since_tag"]:
        print("⚠️ docs/protocole.md changed since the tag « protocole »: report every change in the report")
    if status["uncommitted"]:
        print("⚠️ docs/protocole.md or data/splits.json has uncommitted changes: commit them, and say it in the report")
    return test
