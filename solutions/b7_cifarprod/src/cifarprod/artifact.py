"""The artifact of a model: everything needed to predict without the notebook, in one folder.

    save_artifact("artifact", model, arch="cnn", widths=(16, 32), mean=mean, std=std, classes=CLASSES,
                  data={"path": "data/cifar10_2000.npz", "n_train": 2000, "sha256": "..."})
    model, meta = load_artifact("artifact")                 # in eval() mode, the weights checked
    labels, proba = predict_images(model, meta, images)     # uint8 RGB (N, 32, 32, 3) -> names, probabilities

The folder holds model.safetensors (the weights: named tensors, nothing executed when loading) and meta.json (the
architecture, the input format, the normalization learned on the training images, the classes in the order of the
outputs, the data fingerprint, the metrics, the versions and the SHA-256 of the weights file).
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .data import preprocess
from .model import build_model
from .tracking import file_sha256

WEIGHTS, META = "model.safetensors", "meta.json"
INPUT = {"shape": [32, 32, 3], "dtype": "uint8", "channels": "RGB"}


def save_artifact(folder, model, *, arch, widths, mean, std, classes, data, metrics=None) -> Path:
    """Write the artifact of ``model`` into ``folder`` (created): model.safetensors (its state_dict, with the
    metadata {"package_version": __version__}) and meta.json (package_version, created, arch, widths, input, mean,
    std, classes, data, metrics, torch, python, weights_sha256). Returns the folder."""
    import torch
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(folder / WEIGHTS), metadata={"package_version": __version__})
    meta = {"package_version": __version__, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "arch": str(arch), "widths": [int(w) for w in widths], "input": dict(INPUT),
            "mean": [float(m) for m in mean], "std": [float(s) for s in std], "classes": [str(c) for c in classes],
            "data": dict(data), "metrics": dict(metrics or {}), "torch": torch.__version__,
            "python": platform.python_version(), "weights_sha256": file_sha256(folder / WEIGHTS)}
    (folder / META).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return folder


def load_artifact(folder):
    """(model, meta): the model rebuilt from meta.json, its weights loaded from model.safetensors, in eval() mode.
    ValueError if the weights file is not the one meta.json describes (another SHA-256); FileNotFoundError if a file
    is missing."""
    from safetensors.torch import load_file

    folder = Path(folder)
    meta = json.loads((folder / META).read_text(encoding="utf-8"))
    sha = file_sha256(folder / WEIGHTS)
    if sha != meta.get("weights_sha256"):
        raise ValueError(f"the weights of {folder} do not match meta.json: SHA-256 {sha[:12]}…, expected "
                         f"{str(meta.get('weights_sha256'))[:12]}…")
    model = build_model(meta["arch"], meta["widths"], n_classes=len(meta["classes"]))
    model.load_state_dict(load_file(str(folder / WEIGHTS)))
    model.eval()
    return model, meta


def predict_images(model, meta, images):
    """(labels, probabilities) for uint8 RGB images (N, 32, 32, 3), or one image (32, 32, 3): the class names of
    meta["classes"] and a float32 array (N, n_classes) of softmax probabilities, computed without a graph, with the
    preprocessing of the training (data.preprocess and the normalization of the artifact)."""
    import torch

    x = preprocess(np.asarray(images), meta["mean"], meta["std"])
    with torch.inference_mode():
        proba = torch.softmax(model(x), dim=1).numpy()
    return [meta["classes"][k] for k in proba.argmax(axis=1)], proba
