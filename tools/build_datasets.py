#!/usr/bin/env python
"""(Re)build the light datasets versioned in data/ from their official sources.

    python tools/build_datasets.py            # all datasets
    python tools/build_datasets.py sunspots   # only one
    python tools/build_datasets.py --list

Documents the provenance of every versioned file (see data/cards/). Needs
internet. The learner never has to run it: the files are already in git.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wb import datasets as D  # noqa: E402

DATA = ROOT / "data"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_penguins() -> list[Path]:
    out = []
    for key in ("penguins", "penguins_raw"):
        dest = DATA / D.FILES[key]
        D.download(D.URLS[key], dest)
        out.append(dest)
    return out


def _build_text(key: str) -> list[Path]:
    dest = DATA / D.FILES[key]
    with tempfile.TemporaryDirectory() as tmp:
        raw = D.download(D.URLS[key], Path(tmp) / "raw.txt")
        text = raw.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8", newline="\n")  # full text, licence kept
    return [dest]


def build_holmes() -> list[Path]:
    return _build_text("holmes")


def build_verne() -> list[Path]:
    return _build_text("verne")


def build_sunspots() -> list[Path]:
    dest = DATA / D.FILES["sunspots"]
    with tempfile.TemporaryDirectory() as tmp:
        raw = D.download(D.URLS["sunspots"], Path(tmp) / "SN_m_tot_V2.0.csv")
        df = D.parse_silso(raw.read_text(encoding="utf-8"))
    df.to_csv(dest, index=False)
    return [dest]


def build_california() -> list[Path]:
    from sklearn.datasets import fetch_california_housing

    dest = DATA / D.FILES["california"]
    with tempfile.TemporaryDirectory() as tmp:
        frame = fetch_california_housing(data_home=tmp, as_frame=True).frame
    frame.to_csv(dest, index=False)  # full float precision: exact round trip
    return [dest]


def build_mnist() -> list[Path]:
    import numpy as np
    import torchvision

    dest = DATA / D.FILES["mnist"]
    with tempfile.TemporaryDirectory() as tmp:
        train = torchvision.datasets.MNIST(tmp, train=True, download=True)
        test = torchvision.datasets.MNIST(tmp, train=False, download=True)
        np.savez_compressed(
            dest,
            x_train=train.data.numpy(), y_train=train.targets.numpy().astype(np.uint8),
            x_test=test.data.numpy(), y_test=test.targets.numpy().astype(np.uint8),
        )
    size = dest.stat().st_size / 1e6
    if size > 20:
        dest.unlink()
        raise RuntimeError(f"mnist.npz fait {size:.1f} Mo (> 20 Mo) : non versionné")
    return [dest]


BUILDERS = {
    "penguins": build_penguins,
    "holmes": build_holmes,
    "verne": build_verne,
    "sunspots": build_sunspots,
    "california": build_california,
    "mnist": build_mnist,
}


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # emoji on Windows consoles and pipes
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"datasets among {', '.join(BUILDERS)}")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(BUILDERS))
        return 0
    status = 0
    for name in args.names or BUILDERS:
        try:
            for path in BUILDERS[name]():
                size = path.stat().st_size / 1e6
                print(f"✅ {name:11s} {path.relative_to(ROOT)}  {size:.2f} Mo  sha256:{_sha(path)}")
        except Exception as exc:
            status = 1
            print(f"❌ {name}: {exc}")
    return status


if __name__ == "__main__":
    sys.exit(main())
