#!/usr/bin/env python
"""Predict from the command line with the delivered model (PF.11).

    python scripts/predict.py --model artifacts/model --input examples.jsonl      # one JSON record per line
    echo '{"text": "..."}' | python scripts/predict.py --model artifacts/model --input -

One JSON prediction per line on stdout, in the order of the inputs. The work is in ``projet.predict``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from projet.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["predict", *sys.argv[1:]]))
