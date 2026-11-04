"""Write a small data file with the schema of the project (random images): python scripts/make_synthetic.py PATH.

The continuous integration has no CIFAR-10 (the data are not in git): its smoke training runs on this file.
"""

import sys
from pathlib import Path

import numpy as np

from cifarprod.data import synthetic


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0] if args else "data/synthetic.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    X, y = synthetic(256, seed=0)
    X_val, y_val = synthetic(64, seed=1)
    np.savez(path, x_train=X, y_train=y, x_val=X_val, y_val=y_val)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
