#!/usr/bin/env python
"""Generate the summary table of the experiments from the run journal (PF.7): never write it by hand.

    python scripts/summarize_runs.py                                   # experiments/runs.jsonl -> stdout
    python scripts/summarize_runs.py --out experiments/summary.md      # the table of the report

Contract (TODO PF.7): one row per configuration (the same settings except the seed), with the number of seeds, the
mean and the standard deviation of the validation metric, the mean duration; the failed runs counted apart; sorted by
mean, best first; Markdown. A row with fewer than three seeds is marked. The test never appears here: its numbers are
in experiments/test_results.json, written once by notebooks/03_evaluation.ipynb.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from projet.train import read_runs  # noqa: E402


def summarize(runs: list[dict]) -> str:
    """The Markdown table of the contract above, from the records of the journal."""
    # TODO PF.7: group by the config without its seed (json.dumps(..., sort_keys=True)), then mean and std
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
