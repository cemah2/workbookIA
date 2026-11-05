"""The HTTP API of the model and the measurement of its latency (step PF.11).

    python -m projet.serve --model artifacts/model --port 8000      # or: projet serve --model artifacts/model
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"records": [{...}]}'

FastAPI and uvicorn are optional dependencies (``pip install -e ".[serve]"``), imported inside the functions: the
package and its tests import without them. A Gradio demo is accepted instead of the API (README, PF.11).
"""

from __future__ import annotations

import argparse
import sys

from .predict import Predictor


def create_app(predictor: Predictor):
    """A FastAPI application around ``predictor`` (B7.18):

    - ``GET /health`` -> 200 ``{"status": "ok", "model_version": ...}``;
    - ``POST /predict`` with ``{"records": [ {...}, ... ]}`` -> 200 ``{"predictions": [...], "model_version": ...}``;
      a body that does not follow the schema (pydantic) -> 422, without echoing the input back; at most 64 records per
      request (422 above); an error of the model -> 500 with a short message, the details in the log only.

    The schema of a record is that of ``Predictor.predict``; never log the content of the inputs (B7.7).
    """
    # TODO PF.11: from fastapi import FastAPI; pydantic models of the request and the response; two routes
    raise NotImplementedError("TODO PF.11: create_app")


def measure_latency(predictor: Predictor, records: list[dict], *, n: int = 200, warmup: int = 10,
                    batch_size: int = 1) -> dict:
    """The latency of ``predictor.predict`` on batches of ``batch_size`` records drawn in turn from ``records``:
    ``warmup`` calls not counted, then ``n`` calls timed with ``time.perf_counter``. Returns
    ``{"batch_size": ..., "n": ..., "p50_ms": ..., "p95_ms": ..., "mean_ms": ..., "per_record_ms": ...}``. Report it with
    the machine (CPU, threads) in the README (B7.19, B7.20)."""
    # TODO PF.11: a loop, a list of durations, np.percentile
    raise NotImplementedError("TODO PF.11: measure_latency")


def main(argv: list[str] | None = None) -> int:
    """``python -m projet.serve --model DIR [--host H] [--port P]``: load the artifact and serve it with uvicorn.
    Provided (it calls ``load_artifact`` and ``create_app``)."""
    parser = argparse.ArgumentParser(prog="projet.serve", description="Serve the model over HTTP.")
    parser.add_argument("--model", default="artifacts/model", help="artifact folder (default: %(default)s)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    from .predict import load_artifact

    app = create_app(load_artifact(args.model))
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
