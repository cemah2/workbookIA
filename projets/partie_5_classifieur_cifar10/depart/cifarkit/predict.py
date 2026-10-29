"""Classify images with the delivered model (model.pt and model.json, written by save_model).

    python -m cifarkit.predict photo.jpg autre.png --model .     # from the project folder

Any image (JPEG, PNG...) is converted to RGB and resized to 32 x 32, like the images of CIFAR-10.

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .train import load_model, predict_proba


def load_image(path: Path) -> np.ndarray:
    """The image at path as a uint8 array (32, 32, 3): converted to RGB, then resized to 32 x 32 with
    ``Image.Resampling.BICUBIC`` (the whole image, without cropping)."""
    raise NotImplementedError("load_image")  # TODO MP5.8


def classify(paths: list[Path], folder: Path, top: int = 3) -> list[tuple[str, list[tuple[str, float]]]]:
    """For each image path, (path as a string, its ``top`` most probable classes as (name, probability) pairs, the most
    probable first), with the model saved in folder (load_model) and its own normalization."""
    raise NotImplementedError("classify")  # TODO MP5.8


def main(argv: list[str] | None = None) -> int:
    """Provided: the command line, ``python -m cifarkit.predict IMAGE [IMAGE ...] --model FOLDER [--top K]``."""
    parser = argparse.ArgumentParser(description="Classe des images avec le modèle CIFAR-10 sauvegardé.")
    parser.add_argument("images", nargs="+", type=Path, help="les images à classer (JPEG, PNG...)")
    parser.add_argument("--model", type=Path, default=Path("."), help="le dossier de model.pt et model.json")
    parser.add_argument("--top", type=int, default=3, help="le nombre de classes affichées par image")
    args = parser.parse_args(argv)
    for path, guesses in classify(args.images, args.model, args.top):
        print(f"{path} : " + " · ".join(f"{name} {p:.1%}" for name, p in guesses))
    return 0


if __name__ == "__main__":
    sys.exit(main())
