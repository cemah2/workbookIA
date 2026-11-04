"""The run journal of cifarprod: one JSON line per run, appended to a JSONL file, never rewritten.

    run_id = new_run_id()                              # "20261104-161731-e06339"
    log_run(Path("runs/runs.jsonl"), {"run_id": run_id, "status": "ok", ...})
    runs = read_runs(Path("runs/runs.jsonl"))          # a list of dictionaries, oldest first

A run records what defines it (configuration, environment, data fingerprint), what it measures (metrics per epoch,
duration) and what it produces (its artifacts, with their path, size and fingerprint).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def new_run_id() -> str:
    """A unique run identifier: the UTC date and time, then six random hexadecimal characters."""
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def file_sha256(path: str | Path) -> str:
    """The SHA-256 of the bytes of a file (64 hexadecimal characters)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path: str | Path) -> dict:
    """{"path", "bytes", "sha256"} of a file written by the run."""
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _git(*args: str, cwd=None) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def environment(cwd=None) -> dict:
    """What runs the code: the versions of Python, NumPy and PyTorch, the platform, the git commit (None outside a
    repository) and whether the working tree has uncommitted changes."""
    versions = {}
    for name in ("numpy", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    status = _git("status", "--porcelain", cwd=cwd)
    return {"python": platform.python_version(), **versions, "platform": platform.platform(),
            "commit": _git("rev-parse", "--short", "HEAD", cwd=cwd), "dirty": bool(status) if status is not None else None}


def log_run(path: str | Path, record: dict) -> None:
    """Append one JSON line (keys sorted) to the journal ``path``; its folder is created if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def read_runs(path: str | Path) -> list[dict]:
    """The runs of the journal, oldest first ([] if the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
