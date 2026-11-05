#!/usr/bin/env python
"""Generate the summary table of the experiments from the run journal (PF.7): never write it by hand.

    python scripts/summarize_runs.py                                   # experiments/runs.jsonl -> stdout
    python scripts/summarize_runs.py --out experiments/summary.md      # the table of the report

Contract (TODO PF.7): one row per configuration, that is the same ``command``, the same ``model`` and the same
``config`` except its ``seed`` (the naive and the classical baselines share their config: ``model`` tells them apart);
with the number of seeds, the mean and the standard deviation of the validation metric ``metrics["best_val"]`` (the
same validation split for the baselines and the deep model; the cross-validation of the baselines is in
``metrics["cv"]``), the mean duration (``seconds``); the failed runs counted apart; two runs of the same seed are a
check of reproducibility, not two seeds: keep one; sorted by mean, best first (``get_scorer`` convention: higher is
better); Markdown. A row with fewer than three seeds is marked. The test never appears here: its numbers are
in experiments/test_results.json, written once by notebooks/03_evaluation.ipynb.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from projet.train import read_runs  # noqa: E402


def summarize(runs: list[dict]) -> str:
    """The Markdown table of the contract above, from the records of the journal."""
    # TODO PF.7: group by (command, model, config without its seed: json.dumps(..., sort_keys=True)), mean and std
    raise NotImplementedError("TODO PF.7: summarize")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="experiments/runs.jsonl")
    parser.add_argument("--out", default=None, help="Markdown file to write (default: stdout)")
    args = parser.parse_args(argv)
    try:
        table = summarize(read_runs(args.runs))
    except NotImplementedError as err:
        print(f"⏳ not written yet: {err}", file=sys.stderr)
        return 3
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
