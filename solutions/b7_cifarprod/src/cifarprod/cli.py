"""The command line of cifarprod.

    cifarprod train --config configs/fast.yaml --set lr=0.01 -v      # or: python -m cifarprod train ...
    cifarprod --version

Exit codes: 0 success; 2 bad usage (argparse) or invalid configuration; 1 the data cannot be read or the training
failed. Messages go through logging (stderr); the only line written on stdout is the JSON summary of the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import load_config
from .tracking import artifact, environment, log_run, new_run_id

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cifarprod", description="Train a CIFAR-10 classifier.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train a model, save its weights and log the run")
    train.add_argument("--config", default=None, help="YAML file of settings (default: the defaults of TrainConfig)")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a setting")
    train.add_argument("--out", default=None, help="folder of the weights (default: runs/<run id>)")
    train.add_argument("--runs", default="runs/runs.jsonl", help="the run journal (default: %(default)s)")
    train.add_argument("-v", "--verbose", action="store_true", help="also show the DEBUG messages")
    return parser


def run_train(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config, args.set)
    except (OSError, ValueError) as err:
        log.error("invalid configuration: %s", err)
        return 2
    run_id = new_run_id()
    out = Path(args.out) if args.out else Path("runs") / run_id
    record = {"run_id": run_id, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": config.to_dict(), "environment": environment(), "status": "failed"}
    start = time.perf_counter()
    try:
        from safetensors.torch import save_file

        from .data import fingerprint, load_npz
        from .train import train

        try:
            X, y = load_npz(config.data, "train")
            X_val, y_val = load_npz(config.data, "val")
            if config.n_train > len(X):
                raise ValueError(f"n_train = {config.n_train}, but {config.data} has {len(X)} training images")
        except (OSError, ValueError) as err:
            log.error("cannot load the data: %s", err)
            record["error"] = f"{type(err).__name__}: {err}"
            return 1
        X, y = X[:config.n_train], y[:config.n_train]
        log.info("loaded %d training and %d validation images from %s", len(X), len(X_val), config.data)
        record["data"] = {"path": config.data, "n_train": len(X), "sha256": fingerprint(X, y)}
        result = train(X, y, X_val, y_val, arch=config.arch, widths=config.widths, epochs=config.epochs,
                       lr=config.lr, batch_size=config.batch_size, seed=config.seed,
                       deterministic=config.deterministic, num_workers=config.num_workers)
        out.mkdir(parents=True, exist_ok=True)
        weights = out / "model.safetensors"
        save_file(result["model"].state_dict(), str(weights))
        history = result["history"]
        record.update(status="ok", metrics={**history, "val_acc_final": history["val_acc"][-1]},
                      normalization={"mean": list(result["mean"]), "std": list(result["std"])},
                      artifacts=[artifact(weights)])
        log.info("run %s: val_acc %.4f, weights in %s", run_id, history["val_acc"][-1], weights)
        print(json.dumps({"run_id": run_id, "status": "ok", "val_acc": history["val_acc"][-1],
                          "weights": str(weights.resolve())}))
        return 0
    except Exception as err:                      # noqa: BLE001 - logged with its traceback, exit code 1
        log.exception("training failed")
        record["error"] = f"{type(err).__name__}: {err}"
        return 1
    finally:
        record["seconds"] = round(time.perf_counter() - start, 3)
        log_run(args.runs, record)                # every run is logged, the failed ones too


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)        # a bad option: a message and exit code 2
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run_train(args)


if __name__ == "__main__":
    sys.exit(main())
