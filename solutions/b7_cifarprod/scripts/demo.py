"""From a fresh clone to a served prediction, in one command: python scripts/demo.py

1. the data: data/cifar10_2000.npz if it is there, else a synthetic file (scripts/make_synthetic.py);
2. a smoke training with configs/smoke.yaml, through the command line (cifarprod train): the artifact of the model
   (model.safetensors, meta.json) in runs/demo/;
3. the API on a free port, in a thread (cifarprod.serve.running, always stopped at the end): GET /health, then
   POST /predict with one image, 50 times, for the latency.
The last line of stdout is a JSON summary: {"status", "label", "probability", "p50_ms", "p95_ms", "artifact",
"seconds"}; the exit code is 0 only if every step worked.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def step(message: str) -> None:
    print(f"[demo] {message}", file=sys.stderr, flush=True)


def main() -> int:
    import requests

    from cifarprod.data import load_npz
    from cifarprod.serve import running

    start = time.perf_counter()
    data = ROOT / "data" / "cifar10_2000.npz"
    if not data.is_file():
        data = ROOT / "data" / "synthetic.npz"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), str(data)], check=True,
                       stdout=subprocess.DEVNULL)
    step(f"data: {data.relative_to(ROOT)}")
    out = ROOT / "runs" / "demo"
    train = subprocess.run([sys.executable, "-m", "cifarprod", "train", "--config", str(ROOT / "configs" / "smoke.yaml"),
                            "--set", f"data={data}", "--out", str(out), "--runs", str(ROOT / "runs" / "runs.jsonl")],
                           cwd=ROOT, capture_output=True, text=True)
    if train.returncode != 0:
        step(f"training failed (exit code {train.returncode}): {train.stderr.strip()[-500:]}")
        return 1
    step(f"trained: {train.stdout.strip()}")
    image = load_npz(data, "val")[0][0]
    with running(out) as url:
        session = requests.Session()
        health = session.get(f"{url}/health", timeout=10)
        step(f"health: {health.status_code} {health.json().get('package_version')}")
        times, answer = [], None
        for _ in range(55):
            t0 = time.perf_counter()
            answer = session.post(f"{url}/predict", json={"pixels": image.tolist()}, timeout=10)
            times.append(time.perf_counter() - t0)
    if health.status_code != 200 or answer.status_code != 200:
        step(f"the API answered {health.status_code} and {answer.status_code}")
        return 1
    p50, p95 = np.percentile(1000 * np.array(times[5:]), [50, 95])
    body = answer.json()
    print(json.dumps({"status": answer.status_code, "label": body["label"], "probability": body["probability"],
                      "p50_ms": round(float(p50), 2), "p95_ms": round(float(p95), 2), "artifact": str(out),
                      "seconds": round(time.perf_counter() - start, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
