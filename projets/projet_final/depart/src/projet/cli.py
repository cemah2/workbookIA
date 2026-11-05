"""The command line of the project (provided wiring, to adapt to your project: PF.10).

    projet download --config configs/fast.yaml          # PF.2   (or: python -m projet.cli download ...)
    projet split --config configs/fast.yaml             # PF.3   writes data/splits.json once, prints its SHA-256
    projet baselines --config configs/fast.yaml         # PF.5   cross-validation of the naive and classical models
    projet train --config configs/fast.yaml --set seed=1 --set lr=3e-4      # PF.6, PF.7
    projet train --set "model_params={hidden: 32}"      # a mapping: quotes, and a space after each colon (YAML)
    projet predict --model artifacts/model --input examples.jsonl           # PF.11
    projet latency --model artifacts/model --input examples.jsonl           # PF.11
    projet serve --model artifacts/model --port 8000                        # PF.11

Exit codes: 0 success; 2 bad usage or invalid configuration; 1 the command failed (the run is logged as failed);
3 a function of the package is not written yet (« ⏳ »). Messages go to stderr (logging); stdout only receives the
JSON summary of the command (or the predictions), so that a script can read it. Every ``baselines`` and ``train`` run
is appended to the journal ``config.runs_file``, failed ones included; both write ``metrics["best_val"]``, the main
metric on the same validation split (the baselines refitted on the same training rows as the deep model), which is
what the summary table compares. ``train`` logs its run even before ``predict.save_artifact`` is written (PF.11). The sealed test is not opened here: that is
notebooks/03_evaluation.ipynb (``evaluate.open_sealed_test``), once.
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

log = logging.getLogger(__name__)
NOT_WRITTEN = 3


def build_parser() -> argparse.ArgumentParser:
    """The parser of the command line: one sub-command per step (add yours, PF.10)."""
    parser = argparse.ArgumentParser(prog="projet", description="The final project: data, baselines, model, demo.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(name: str, help_text: str) -> argparse.ArgumentParser:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/fast.yaml", help="YAML settings (default: %(default)s)")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a setting")
        command.add_argument("-v", "--verbose", action="store_true", help="also show the DEBUG messages")
        return command

    with_config("download", "download the raw data and check its SHA-256 (data card)")
    with_config("split", "make the split once and write it (data/splits.json); prints its SHA-256")
    with_config("baselines", "cross-validate the naive and the classical models on the training split")
    train = with_config("train", "train the deep model, keep its best checkpoint, write the artifact")
    train.add_argument("--out", default=None, help="artifact folder (default: config.out_dir)")
    for name, help_text in (("predict", "predict the records of a JSONL file (one JSON object per line)"),
                            ("latency", "measure the latency of the predictor on the records of a JSONL file")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--model", default="artifacts/model", help="artifact folder (default: %(default)s)")
        command.add_argument("--input", required=True, help="JSONL file of raw records, or - for stdin")
        command.add_argument("--batch-size", type=int, default=1)
        command.add_argument("-v", "--verbose", action="store_true")
    serve = sub.add_parser("serve", help="serve the model over HTTP (FastAPI, extra [serve])")
    serve.add_argument("--model", default="artifacts/model")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("-v", "--verbose", action="store_true")
    return parser


# ------------------------------------------------------------------------------------------------ the commands
def _training_frames(config):
    """The training and validation rows of the frozen split (the training ones subsampled in the fast mode). Once
    docs/protocole.md holds the SHA-256 of the split, the split is checked against it (a changed split fails)."""
    from .data import load_examples, load_splits, select, subsample
    from .evaluate import PROTOCOL, SPLIT_DIGEST

    df = load_examples(config)
    found = SPLIT_DIGEST.search(Path(PROTOCOL).read_text(encoding="utf-8")) if Path(PROTOCOL).exists() else None
    splits = load_splits(config.split_file, found.group(1) if found else None)
    train_df = subsample(select(df, splits, "train"), config.max_train_examples, config.seed)
    return train_df, select(df, splits, "val")


def _download(config, args) -> dict:
    from .data import download, sha256_file

    return {"files": {str(path): sha256_file(path) for path in download(config)}}


def _split(config, args) -> dict:
    from .data import load_examples, make_splits, save_splits

    splits = make_splits(load_examples(config), config)
    digest = save_splits(splits, config.split_file)
    log.info("split written to %s: write its SHA-256 in docs/protocole.md, commit, then tag « protocole »",
             config.split_file)
    return {"split_file": config.split_file, "sha256": digest, "sizes": {k: len(v) for k, v in splits.items()}}


def _baseline(kind: str):
    def work(config, args) -> dict:
        from .models import build_classical, build_naive
        from .train import cross_validate, set_seed

        from sklearn.metrics import get_scorer

        set_seed(config.seed, config.deterministic)
        train_df, val_df = _training_frames(config)
        estimator = build_naive(config) if kind == "naive" else build_classical(config)
        cv = cross_validate(estimator, train_df, config)
        # then refit on the same training rows as the deep model and score on the same validation (PF.6 compares them)
        fitted = estimator.fit(train_df, train_df[config.target])
        best_val = float(get_scorer(config.metric)(fitted, val_df, val_df[config.target]))
        return {"cv": cv, "best_val": best_val, "n_train": int(len(train_df))}
    return work


def _train(config, args) -> dict:
    from .models import build_model, n_parameters
    from .predict import save_artifact
    from .preprocess import build_preprocessor, make_dataset
    from .train import fit, set_seed

    set_seed(config.seed, config.deterministic)
    train_df, val_df = _training_frames(config)
    preprocessor = build_preprocessor(config).fit(train_df)
    train_ds = make_dataset(train_df, preprocessor, config, train=True)
    val_ds = make_dataset(val_df, preprocessor, config)
    model = build_model(config)                                  # TODO PF.6: pass n_inputs / n_outputs if needed
    result = fit(model, train_ds, val_ds, config)
    metrics = {"best_epoch": result["best_epoch"], "best_val": result["best_score"], "history": result["history"],
               "n_parameters": n_parameters(result["model"]), "n_train": int(len(train_df))}
    try:
        meta = save_artifact(args.out or config.out_dir, result["model"], preprocessor, config,
                             metrics={"val_" + config.metric: result["best_score"]})
    except NotImplementedError as err:              # PF.11 not written yet: the run still counts (PF.6, PF.7)
        log.warning("⏳ %s: no artifact yet; the best checkpoint is %s", err, result["checkpoint"])
        meta = None
    return {**metrics, "artifact": None if meta is None else str(meta), "checkpoint": str(result["checkpoint"])}


def _logged(command: str, model: str, work, config, args) -> int:
    """Run ``work(config, args)``, print its JSON summary, and append the run to the journal (failed ones too)."""
    from .train import dumps, environment, log_run, new_run_id

    run_id = new_run_id()
    record = {"run_id": run_id, "command": command, "model": model,
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": config.to_dict(), "environment": environment(), "status": "failed"}
    if record["environment"]["git_dirty"]:
        log.warning("uncommitted changes: this run cannot be rebuilt from commit %s; commit before the runs you report",
                    record["environment"]["git_commit"])
    start = time.perf_counter()
    try:
        record["metrics"] = work(config, args)
        record["status"] = "ok"
        print(dumps({"run_id": run_id, "model": model, "status": "ok", **record["metrics"]}))
        return 0
    except NotImplementedError as err:
        log.error("⏳ not written yet: %s", err)
        record = None                                             # not a run: nothing to log
        return NOT_WRITTEN
    except Exception as err:                                      # noqa: BLE001 - logged with its traceback
        log.exception("%s failed", command)
        record["error"] = f"{type(err).__name__}: {err}"
        return 1
    finally:
        if record is not None:
            record["seconds"] = round(time.perf_counter() - start, 3)
            try:
                log_run(config.runs_file, record)
            except NotImplementedError:
                log.warning("⏳ train.log_run is not written yet: this run is not in %s", config.runs_file)


def _read_records(path: str) -> list[dict]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _predict(args) -> int:
    from .predict import load_artifact
    from .train import dumps

    predictor = load_artifact(args.model)
    records = _read_records(args.input)
    for start in range(0, len(records), max(1, args.batch_size)):
        for prediction in predictor.predict(records[start:start + max(1, args.batch_size)]):
            print(dumps(prediction))
    return 0


def _latency(args) -> int:
    from .predict import load_artifact
    from .serve import measure_latency
    from .train import dumps, environment

    import os
    import platform

    import torch

    result = measure_latency(load_artifact(args.model), _read_records(args.input), batch_size=args.batch_size)
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():                       # Linux: platform.processor() is often empty
        names = [l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                 if l.startswith("model name")]
        cpu = names[0] if names else cpu
    print(dumps({**result, "platform": environment()["platform"], "cpu": cpu, "cpu_count": os.cpu_count(),
                 "torch_threads": torch.get_num_threads()}))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command of ``argv`` (default: ``sys.argv[1:]``) and return its exit code (module docstring)."""
    args = build_parser().parse_args(argv)                       # a bad option: a message and exit code 2
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.command in ("predict", "latency"):
            return _predict(args) if args.command == "predict" else _latency(args)
        if args.command == "serve":
            from .serve import main as serve_main

            return serve_main(["--model", args.model, "--host", args.host, "--port", str(args.port)])
        try:
            config = load_config(args.config, args.set)
        except (OSError, ValueError) as err:
            log.error("invalid configuration: %s", err)
            return 2
        if args.command == "download":
            print(json.dumps(_download(config, args)))
            return 0
        if args.command == "split":
            print(json.dumps(_split(config, args)))
            return 0
        if args.command == "baselines":
            codes = [_logged("baselines", kind, _baseline(kind), config, args) for kind in ("naive", "classical")]
            return max(codes)
        return _logged("train", config.model, _train, config, args)
    except NotImplementedError as err:
        log.error("⏳ not written yet: %s", err)
        return NOT_WRITTEN
    except (OSError, ValueError) as err:
        log.error("%s failed: %s", args.command, err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
