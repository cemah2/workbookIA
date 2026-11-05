#!/usr/bin/env python
"""Download the raw data of the project and check it (PF.2): the data card (data/card.md) says what and why.

    python scripts/download_data.py                     # configs/fast.yaml
    python scripts/download_data.py --config configs/full.yaml

The work is in ``projet.data.download`` (a fixed revision, a SHA-256 per file, a .part file renamed once checked);
this script only calls it, so that a reader finds the download where they expect it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without pip install -e

from projet.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["download", *sys.argv[1:]]))
