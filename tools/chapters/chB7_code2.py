"""Code cells of the chapter B7 notebook, second half, part B (the service: B7.17 to B7.21), shared by build_chB7.py
(through chB7_parts2.py) and history_chB7.py (used by Claude).

Same conventions as chB7_code.py: GIVEN_* (both notebooks), TODO_* (exercise notebook), SOLUTION_* (solutions
notebook, from the reference package solutions/b7_cifarprod), CHECK_* and EXPERIMENT_* (both), RECORD_* (solutions).
The cells reuse the tools of part A (PROJECT, DATA_FILE, SUBSET, CLASSES, DEFAULTS, scratch, run_line, wbp, checked,
verdict, same, refuses, written, StopCheck, fr). Placeholders «NAME» are filled by chB7_parts2.py.
"""

from __future__ import annotations

from chB7_code import file_cell, reference_file

# =====================================================================================================================
# 📦 B7.17 · the artifact
# =====================================================================================================================
GIVEN_17 = r'''# The model that part B serves: the VGG 16-32-64 delivered by the MP5 solution (versioned with the workbook), with its
# normalization, its classes and the fingerprint of its 45 000 training images; 200 test images it has never seen
import hashlib
import importlib.util
import threading

from safetensors.torch import load_file, save_file

SOURCE_B = "reference"      # or "learner": YOUR model of the MP5, if it is a "vgg" (its arch and widths go in the artifact)
MODEL_B, INFO_B = wbp.base_model(SOURCE_B)
CONFIG_B = {"arch": INFO_B["config"]["arch"], "widths": list(INFO_B["config"]["widths"])}
DATA_B = INFO_B.get("data") or {"path": "CIFAR-10 train (your MP5 split)", "n_train": None, "sha256": None}
X_B, Y_B = wbp.test_images(20)                         # 200 test images (20 per class), uint8 RGB (200, 32, 32, 3)


def model_input(images, mean, std):
    """uint8 RGB images (N, 32, 32, 3) -> the float32 input (N, 3, 32, 32) of the model, as in training."""
    x = (np.asarray(images, np.float32) / np.float32(255) - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def offline_b(model, images, mean, std):
    """What the notebook computes, the reference of every check: (softmax probabilities float64 (N, 10), argmax)."""
    with torch.inference_mode():
        out = model(model_input(images, mean, std))
    return torch.softmax(out, dim=1).double().numpy(), out.argmax(dim=1).numpy()


def permuted_b(seed=17):
    """The reference VGG with its outputs in another order, and the classes in that order (the same predictions)."""
    perm = np.random.default_rng(seed).permutation(10)
    model = wbp.reference_model()[0]
    with torch.no_grad():
        model[-1].weight.copy_(model[-1].weight[perm].clone())
        model[-1].bias.copy_(model[-1].bias[perm].clone())
    return model, [CLASSES[k] for k in perm]


REF_INFO = wbp.reference_model()[1]                     # the reference VGG (B7.19 to B7.21 measure it)
PROBA_B, OFFLINE_B = offline_b(MODEL_B, X_B, INFO_B["mean"], INFO_B["std"])
NAMES_B = [INFO_B["classes"][k] for k in OFFLINE_B]     # what the service must answer for X_B
print(f"ℹ️ le modèle servi : {CONFIG_B}, accuracy {fr((OFFLINE_B == Y_B).mean(), 3)} sur ces 200 images de test ; "
      f"données d'entraînement : {DATA_B['n_train']} images, empreinte {str(DATA_B['sha256'])[:12]}….")'''

SKEL_ARTIFACT = '''"""The artifact of a model: everything needed to predict without the notebook, in one folder (model.safetensors and
meta.json)."""

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


def save_artifact(folder, model, *, arch, widths, mean, std, classes, data, metrics=None):
    """Write the artifact of `model` into `folder` (created) and return the folder (a Path):
    - model.safetensors: the state_dict of the model, with the safetensors metadata {"package_version": __version__};
    - meta.json: {"package_version", "created" (ISO 8601, UTC), "arch", "widths" (a list), "input": {"shape": [32, 32,
      3], "dtype": "uint8", "channels": "RGB"}, "mean", "std" (lists of 3 floats), "classes" (a list, in the order of
      the outputs), "data" (the dict received), "metrics" (the dict received, {} for None), "torch", "python",
      "weights_sha256" (file_sha256 of model.safetensors)}."""
    raise NotImplementedError("save_artifact")


def load_artifact(folder):
    """(model, meta): meta.json read; the SHA-256 of model.safetensors checked against meta["weights_sha256"]
    (ValueError if it differs: another weights file); the model rebuilt by build_model(meta["arch"], meta["widths"],
    len(meta["classes"])), its weights loaded, in eval() mode."""
    raise NotImplementedError("load_artifact")


def predict_images(model, meta, images):
    """(labels, probabilities) of uint8 RGB images (N, 32, 32, 3), or of one image (32, 32, 3): the names
    meta["classes"][k] of the argmax, and a float32 NumPy array (N, n_classes) of softmax probabilities, computed
    without a graph with preprocess(images, meta["mean"], meta["std"]) (the same function as the training)."""
    raise NotImplementedError("predict_images")
'''

TODO_17 = file_cell("ARTIFACT_PY", SKEL_ARTIFACT, "Ex B7.17: src/cifarprod/artifact.py; then make `cifarprod train` "
                    "write the artifact (CLI_PY, in the cell of Ex B7.10)")

SOLUTION_17 = file_cell("ARTIFACT_PY", reference_file("src/cifarprod/artifact.py"),
                        "Ex B7.17: src/cifarprod/artifact.py (cli.py, written in Ex B7.10, already calls save_artifact)")

CHECK_17 = r'''ok_17 = {}
ARTIFACT_B = None                                        # the artifact of the served model, for B7.18 to B7.21
with checked("B7.17"):
    wbp.write_files(PROJECT, {"src/cifarprod/artifact.py": ARTIFACT_PY})
    art_17 = wbp.import_module(PROJECT, "artifact")
    folder_17 = scratch("artifact") / "vgg"
    with wbp.torch_state_kept():
        given_17 = art_17.save_artifact(folder_17, wbp.base_model(SOURCE_B, quiet=True)[0], **CONFIG_B,
                                        mean=INFO_B["mean"], std=INFO_B["std"], classes=INFO_B["classes"],
                                        data=DATA_B, metrics={"val_acc": INFO_B["val_acc"]})
    weights_17, meta_path_17 = folder_17 / "model.safetensors", folder_17 / "meta.json"
    ok_17["files"] = weights_17.is_file() and meta_path_17.is_file() and given_17 is not None \
        and Path(given_17).resolve() == folder_17.resolve()
    verdict("B7.17", ok_17["files"], "save_artifact écrit model.safetensors et meta.json, et rend le dossier.",
            f"save_artifact doit écrire {folder_17.name}/model.safetensors et {folder_17.name}/meta.json, et rendre le "
            f"dossier ; fichiers présents : {sorted(p.name for p in folder_17.glob('*')) if folder_17.exists() else []}, "
            f"rendu {given_17!r}.")
    if not ok_17["files"]:
        raise StopCheck
    from safetensors import safe_open
    try:
        with safe_open(str(weights_17), "pt") as f_17:
            header_17 = f_17.metadata() or {}
        tensors_17 = load_file(str(weights_17))
    except Exception as err_17:  # noqa: BLE001 - not a safetensors file: say it plainly
        ok_17["weights"] = False
        verdict("B7.17", False, "", f"model.safetensors n'est pas un fichier safetensors ({type(err_17).__name__} : "
                f"{str(err_17)[:80]}) : un pickle de torch.save ? Écris-le par safetensors.torch.save_file("
                "model.state_dict(), ...).")
        raise StopCheck
    state_17 = MODEL_B.state_dict()
    ok_17["weights"] = tensors_17.keys() == state_17.keys() and all(torch.equal(tensors_17[k], state_17[k]) for k in state_17)
    verdict("B7.17", ok_17["weights"], f"model.safetensors : les {len(state_17)} tenseurs du state_dict, au bit près.",
            f"model.safetensors doit contenir exactement le state_dict du modèle ({len(state_17)} tenseurs, buffers des "
            f"batchnorms compris) ; reçu {len(tensors_17)} tenseurs.")
    ok_17["header"] = header_17.get("package_version") == "0.1.0"
    verdict("B7.17", ok_17["header"], "l'en-tête safetensors porte package_version 0.1.0.",
            f"métadonnées safetensors : attendu {{'package_version': '0.1.0'}} (des chaînes), reçu {header_17}.")
    meta_17 = json.loads(meta_path_17.read_text(encoding="utf-8"))
    want_17 = {"package_version": "0.1.0", "arch": CONFIG_B["arch"], "widths": CONFIG_B["widths"],
               "input": {"shape": [32, 32, 3], "dtype": "uint8", "channels": "RGB"}, "classes": list(INFO_B["classes"]),
               "data": json.loads(json.dumps(DATA_B)), "metrics": {"val_acc": INFO_B["val_acc"]}, "torch": torch.__version__,
               "weights_sha256": hashlib.sha256(weights_17.read_bytes()).hexdigest()}
    wrong_17 = [k for k, v in want_17.items() if meta_17.get(k) != v]
    for key_17 in ("mean", "std"):
        if not (isinstance(meta_17.get(key_17), list) and len(meta_17[key_17]) == 3
                and np.allclose(meta_17[key_17], INFO_B[key_17], rtol=0, atol=1e-12)):
            wrong_17.append(key_17)
    try:
        from datetime import datetime
        stamp_17 = datetime.fromisoformat(str(meta_17.get("created")))
        if stamp_17.utcoffset() is None or stamp_17.utcoffset().total_seconds() != 0:
            wrong_17.append("created")
    except ValueError:
        wrong_17.append("created")
    if not isinstance(meta_17.get("python"), str):
        wrong_17.append("python")
    ok_17["meta"] = not wrong_17
    verdict("B7.17", ok_17["meta"], "meta.json : version, date UTC, architecture, format d'entrée, normalisation, "
            "classes, données, métriques, versions et SHA-256 des poids.",
            f"meta.json : champs absents ou faux : {wrong_17} (relis la docstring de save_artifact ; widths et classes "
            "en listes, created en ISO 8601 avec le fuseau UTC, weights_sha256 du fichier écrit).")
    ARTIFACT_B = folder_17

with checked("B7.17"):                                    # a NEW process reloads the artifact and predicts
    if ARTIFACT_B is None:
        raise StopCheck
    io_17 = scratch("io")
    np.save(io_17 / "x.npy", X_B)
    probe_17 = ("import json, sys\nimport numpy as np\nfrom cifarprod.artifact import load_artifact, predict_images\n"
                "model, meta = load_artifact(sys.argv[1])\nlabels, proba = predict_images(model, meta, np.load(sys.argv[2]))\n"
                "np.save(sys.argv[3], np.asarray(proba, dtype=np.float64))\n"
                "print(json.dumps({'labels': [str(v) for v in labels], 'training': model.training, "
                "'dtype': str(np.asarray(proba).dtype)}))\n")
    (io_17 / "reload.py").write_text(probe_17, encoding="utf-8")
    r_17 = wbp.run([sys.executable, io_17 / "reload.py", ARTIFACT_B, io_17 / "x.npy", io_17 / "p.npy"], PROJECT,
                   src=[PROJECT / "src"])
    seen_17 = r_17.last_json() if r_17.ok else None
    if not run_line("B7.17", r_17, r_17.ok and isinstance(seen_17, dict), "un processus neuf recharge l'artefact.",
                    "attendu un processus neuf qui recharge l'artefact (load_artifact) et prédit les 200 images"):
        raise StopCheck
    agree_17 = sum(a == b for a, b in zip(seen_17["labels"], NAMES_B))
    proba_17 = np.load(io_17 / "p.npy")
    ok_17["fresh"] = (agree_17 == len(NAMES_B) and not seen_17["training"] and proba_17.shape == PROBA_B.shape
                      and np.allclose(proba_17, PROBA_B, rtol=0, atol=1e-5) and seen_17["dtype"] == "float32")
    verdict("B7.17", ok_17["fresh"], "dans un processus neuf, les 200 images reçoivent les noms et les probabilités du "
            "notebook (à 1e-5 près), le modèle en mode eval().",
            f"dans un processus neuf : {agree_17} noms sur 200 comme le notebook, modèle en mode "
            f"{'entraînement' if seen_17['training'] else 'eval()'}, probabilités {seen_17['dtype']} de forme "
            f"{proba_17.shape}, écart {float(np.abs(proba_17 - PROBA_B).max()) if proba_17.shape == PROBA_B.shape else '?'} "
            "(attendu 200, eval(), float32 (200, 10), au plus 1e-5)"
            + (" : un écart sur les probabilités, predict_images normalise-t-il par meta['mean'] et meta['std'] ?"
               if not seen_17["training"] and proba_17.shape == PROBA_B.shape else "") + ".")
    with wbp.torch_state_kept():
        model_17, meta_l_17 = art_17.load_artifact(ARTIFACT_B)
        alone_17, p_alone_17 = art_17.predict_images(model_17, meta_l_17, X_B[3])
    ok_17["one"] = list(alone_17) == [NAMES_B[3]] and np.allclose(np.asarray(p_alone_17).reshape(-1), PROBA_B[3], atol=1e-5)
    verdict("B7.17", ok_17["one"], "une image seule (32, 32, 3) reçoit le nom et les probabilités qu'elle a dans le batch.",
            f"predict_images d'une image (32, 32, 3) : attendu [{NAMES_B[3]!r}] et ses probabilités du batch, reçu "
            f"{list(alone_17)!r}, écart des probabilités "
            f"{float(np.abs(np.asarray(p_alone_17, dtype=float).reshape(-1)[:10] - PROBA_B[3]).max()) if np.asarray(p_alone_17).size == 10 else '?'}.")
    perm_model_17, perm_classes_17 = permuted_b()
    folder_p_17 = scratch("artifact") / "permuted"
    with wbp.torch_state_kept():
        art_17.save_artifact(folder_p_17, perm_model_17, arch="vgg", widths=[16, 32, 64], mean=REF_INFO["mean"],
                             std=REF_INFO["std"], classes=perm_classes_17, data=DATA_B)
        names_p_17 = art_17.predict_images(*art_17.load_artifact(folder_p_17), X_B)[0]
    ref_names_17 = [CLASSES[k] for k in offline_b(wbp.reference_model()[0], X_B, REF_INFO["mean"], REF_INFO["std"])[1]]
    ok_17["classes"] = list(names_p_17) == ref_names_17
    verdict("B7.17", ok_17["classes"], "un modèle aux sorties rangées dans un autre ordre garde les bons noms : les "
            "classes voyagent avec l'artefact.",
            ("(les prédictions sont déjà fausses ci-dessus) " if not ok_17.get("fresh", True) else "") + f"un modèle dont les sorties sont rangées dans un autre ordre (et classes dans cet ordre) ne reçoit le bon "
            f"nom que pour {sum(a == b for a, b in zip(names_p_17, ref_names_17))} images sur 200 : d'où viennent les "
            "noms de predict_images ?")
    save_file(wbp.reference_model()[0].state_dict(), str(folder_p_17 / "model.safetensors"))   # another weights file
    ok_17["sha"] = refuses("B7.17", "load_artifact d'un dossier dont on a remplacé model.safetensors",
                           lambda: art_17.load_artifact(folder_p_17))

with checked("B7.17"):                                    # the command line writes the artifact of each run
    if ARTIFACT_B is None:
        raise StopCheck
    out_17 = scratch("out") / "run"
    r_17 = wbp.cli(PROJECT, "train", "--set", "n_train=200", "--out", out_17, "--runs", scratch("runs") / "runs.jsonl")
    if not run_line("B7.17", r_17, r_17.ok and (out_17 / "meta.json").is_file(),
                    "cifarprod train écrit l'artefact (model.safetensors et meta.json).",
                    "attendu le code 0 et <out>/meta.json à côté de model.safetensors : dans CLI_PY (cellule de B7.10), "
                    "remplace l'écriture des poids par save_artifact, puis réexécute la cellule de vérification de B7.10"):
        raise StopCheck
    meta_c_17 = json.loads((out_17 / "meta.json").read_text(encoding="utf-8"))
    want_mean_17 = (SUBSET["x_train"][:200] / 255.0).mean(axis=(0, 1, 2))
    want_sha_17 = wbp.array_fingerprint(SUBSET["x_train"][:200], SUBSET["y_train"][:200])
    got_c_17 = {"arch": meta_c_17.get("arch"), "widths": meta_c_17.get("widths"), "classes": meta_c_17.get("classes"),
                "sha256": (meta_c_17.get("data") or {}).get("sha256")}
    ok_17["cli"] = (got_c_17 == {"arch": "cnn", "widths": [16, 32], "classes": list(CLASSES), "sha256": want_sha_17}
                    and np.allclose(meta_c_17.get("mean", [0, 0, 0]), want_mean_17, rtol=0, atol=1e-12))
    verdict("B7.17", ok_17["cli"], f"l'artefact du run : cnn [16, 32], les classes, la normalisation des 200 images et "
            f"l'empreinte de leurs données ({want_sha_17[:12]}…).",
            f"l'artefact de cifarprod train --set n_train=200 : attendu arch cnn, widths [16, 32], classes = CLASSES, "
            f"mean = channel_stats des 200 images et data.sha256 = fingerprint(X[:200], y[:200]) ; reçu "
            f"{ {k: (v[:12] + '…' if isinstance(v, str) and len(v) > 40 else v) for k, v in got_c_17.items()} }.")
    with wbp.torch_state_kept():
        art_17.load_artifact(out_17)
if ok_17 and all(ok_17.values()) and len(ok_17) == 9:
    print("✅ Ex B7.17 : ton artefact se suffit : un autre processus le recharge et prédit comme le notebook.")'''

# =====================================================================================================================
# 📦 B7.18 · the API
# =====================================================================================================================
SKEL_SERVE = '''"""The prediction API of cifarprod: FastAPI (create_app), and the same routes with the standard library (make_server),
served in a thread by running(), or by python -m cifarprod.serve."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from pydantic import BaseModel, ValidationError, field_validator

from . import __version__
from .artifact import load_artifact, predict_images

log = logging.getLogger(__name__)
MAX_BATCH = 64


class Image(BaseModel):
    """{"pixels": 32 x 32 x 3 integers in 0..255 (RGB)}; the validator raises ValueError("expected shape (32, 32, 3),
    got ...") or ValueError("pixel values must be in 0..255") (the model of the fiche, 207.7)."""
    pixels: list[list[list[int]]]


class Batch(BaseModel):
    """{"images": [1 to MAX_BATCH images, each 32 x 32 x 3 integers in 0..255]}; ValueError otherwise."""
    images: list[list[list[list[int]]]]


class Service:
    """The model of an artifact and its answers, shared by the two servers. Attributes: model, meta (load_artifact),
    version (the first 12 characters of meta["weights_sha256"])."""

    def __init__(self, artifact_dir):
        raise NotImplementedError("Service.__init__")

    def health(self):
        """{"status": "ok", "package_version", "model": {"arch", "widths", "classes", "weights_sha256", "data_sha256",
        "created"}, "torch"}."""
        raise NotImplementedError("Service.health")

    def predict(self, images):
        """[{"label", "probability"}, ...] for a list of images (lists of 32 x 32 x 3 integers): predict_images on the
        uint8 array; "probability" is the probability of the label, a float rounded to 4 decimals."""
        raise NotImplementedError("Service.predict")


def create_app(artifact_dir):
    """The FastAPI application of the artifact, its Service in app.state.service: GET /health -> Service.health();
    POST /predict (Image) -> {"label", "probability", "model_version"}; POST /predict_batch (Batch) -> {"predictions":
    [{"label", "probability"}, ...], "model_version"}. A body that does not follow the schema: 422 {"detail": [{"loc",
    "msg", "type"}, ...]} WITHOUT the input that was sent; an exception of the model: 500 {"detail": "internal error"}
    (and log.exception)."""
    raise NotImplementedError("create_app")


class Handler(BaseHTTPRequestHandler):
    """The same routes with http.server: the same validation (Image, Batch), the same answers and status codes (and 404
    for another path); the Service is self.server.service."""

    protocol_version = "HTTP/1.1"          # keep-alive: one connection for many requests (send Content-Length)
    disable_nagle_algorithm = True         # the headers and the body leave together (no 40 ms wait)

    def do_GET(self):
        raise NotImplementedError("Handler.do_GET")

    def do_POST(self):
        raise NotImplementedError("Handler.do_POST")


def make_server(artifact_dir, host="127.0.0.1", port=0):
    """The fallback server, not started: a ThreadingHTTPServer((host, port), Handler), its Service in server.service
    (port 0: a free port, server.server_address[1])."""
    raise NotImplementedError("make_server")


@contextlib.contextmanager
def running(artifact_dir, backend="auto", host="127.0.0.1"):
    """Serve the artifact in a thread on a free port and yield its base URL ("http://127.0.0.1:<port>"); the server is
    ALWAYS stopped and its port freed at the end of the block, even after an error. backend: "fastapi" (uvicorn),
    "stdlib" (make_server) or "auto" (FastAPI if fastapi and uvicorn import)."""
    raise NotImplementedError("running")
    yield


def main(argv=None):
    """python -m cifarprod.serve [--artifact DIR] [--host HOST] [--port PORT] [--backend auto|fastapi|stdlib]: serve the
    artifact (default: the variable MODEL_DIR, else "artifact"; port 8000) until stopped; returns 1, with an ERROR line,
    if the artifact cannot be loaded."""
    raise NotImplementedError("main")


if __name__ == "__main__":
    sys.exit(main())
'''

TODO_18 = file_cell("SERVE_PY", SKEL_SERVE, "Ex B7.18: src/cifarprod/serve.py")

SOLUTION_18 = file_cell("SERVE_PY", reference_file("src/cifarprod/serve.py"), "Ex B7.18: src/cifarprod/serve.py")

CHECK_18 = r'''ok_18 = {}
warnings.filterwarnings("ignore", message=r".*Using `httpx` with `starlette.testclient`.*")   # Starlette 1.10
BAD_18 = {"une image 28 × 28": {"pixels": np.zeros((28, 28, 3), int).tolist()},
          "un pixel à 256": {"pixels": np.full((32, 32, 3), 256).tolist()},
          "un pixel à -1": {"pixels": np.full((32, 32, 3), -1).tolist()},
          "pixels en chaîne": {"pixels": "a cat"},
          "sans le champ pixels": {"image": X_B[0].tolist()}}
BAD_BATCH_18 = {"un batch vide": {"images": []}, "65 images": {"images": np.zeros((65, 32, 32, 3), int).tolist()},
                "une image 28 × 28 dans le batch": {"images": [X_B[0].tolist(), np.zeros((28, 28, 3), int).tolist()]}}


class Failing_18(torch.nn.Module):
    """A model that fails at every call (a bug, an out-of-memory...): the API must answer a clean 500."""

    def forward(self, x):
        raise RuntimeError("the model failed (a test of the check cell)")


@contextlib.contextmanager
def logged_18():
    """The records of the logger cifarprod.serve during the block, kept here instead of being printed."""
    import logging
    logger, records = logging.getLogger("cifarprod.serve"), []
    handler = logging.Handler(level=logging.DEBUG)
    handler.emit = records.append
    keep = logger.propagate, logger.level
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.propagate, logger.level = keep[0], keep[1]


def api_cases_18(get, post, label, records=()):
    """The same cases for both servers: get(path) and post(path, body) return (status, body as a dict or a text);
    records: the log records of the server (logged_18), to show the cause of a 500."""
    oks = {}
    status, body = get("/health")
    body = body if isinstance(body, dict) else {}
    model_info = body.get("model", {}) if isinstance(body.get("model"), dict) else {}
    want_health = {"status": "ok", "package_version": "0.1.0"}
    want_model = {"arch": META_18["arch"], "widths": META_18["widths"], "classes": META_18["classes"],
                  "weights_sha256": META_18["weights_sha256"], "data_sha256": META_18["data"].get("sha256"),
                  "created": META_18.get("created")}
    wrong = [k for k, v in want_health.items() if body.get(k) != v] + \
        [f"model.{k}" for k, v in want_model.items() if model_info.get(k) != v] + \
        ([] if isinstance(body.get("torch"), str) else ["torch"])
    oks["health"] = status == 200 and not wrong
    verdict("B7.18", oks["health"], f"{label} GET /health : 200, la version du paquet, le SHA-256 des poids et des données.",
            f"{label} GET /health : attendu 200 et {{status, package_version, model: {{arch, widths, classes, "
            f"weights_sha256, data_sha256, created}}, torch}} ; reçu {status}, champs absents ou faux : {wrong}.")
    answers = [post("/predict", {"pixels": X_B[i].tolist()}) for i in range(8)]
    labels = [b.get("label") if isinstance(b, dict) else None for _, b in answers]
    probs_ok = all(isinstance(b, dict) and abs(float(b.get("probability", -1)) - PROBA_B[i].max()) <= 6e-5
                   and b.get("model_version") == META_18["weights_sha256"][:12] for i, (_, b) in enumerate(answers))
    oks["predict"] = all(s == 200 for s, _ in answers) and labels == NAMES_B[:8] and probs_ok
    verdict("B7.18", oks["predict"], f"{label} POST /predict : les 8 noms du notebook, leur probabilité (4 décimales) "
            "et la version du modèle (test de non-régression).",
            f"{label} POST /predict sur 8 images : attendu 200, les noms {NAMES_B[:8]}, la probabilité de chacun à 4 "
            f"décimales et model_version = les 12 premiers caractères du SHA-256 des poids ; reçu "
            f"{[s for s, _ in answers]} {labels} {str(answers[0][1])[:120]}.")
    if any(s == 500 for s, _ in answers):
        last = next((r for r in reversed(list(records)) if r.exc_info), None)
        if last is not None:
            print(f"   côté serveur : {last.exc_info[0].__name__} : {str(last.exc_info[1])[:200]}")
    refused = {}
    for what, body_bad in {**BAD_18, **{k: v for k, v in BAD_BATCH_18.items()}}.items():
        path = "/predict_batch" if what in BAD_BATCH_18 else "/predict"
        status, answer = post(path, body_bad)
        text = json.dumps(answer) if isinstance(answer, dict) else str(answer)
        refused[what] = (status, status == 422 and isinstance(answer, dict) and '"input"' not in text
                         and "a cat" not in text)
    oks["422"] = all(good for _, good in refused.values())
    not_422 = [f"{w} ({s})" for w, (s, good) in refused.items() if s != 422]
    echoed = [w for w, (s, good) in refused.items() if s == 422 and not good]
    verdict("B7.18", oks["422"], f"{label} {len(refused)} requêtes mal formées : 422, sans l'entrée envoyée dans la réponse.",
            f"{label} attendu 422 en JSON, sans l'entrée envoyée, pour chaque requête mal formée ; "
            + " ; ".join(part for part in [
                f"pas de 422 : {', '.join(not_422)} (les modèles pydantic doivent refuser ces entrées : forme, bornes, "
                "taille du batch)" if not_422 else "",
                f"422 avec l'entrée renvoyée : {', '.join(echoed)} (retire input de chaque erreur : une donnée "
                "personnelle, peut-être)" if echoed else ""] if part) + ".")
    status, body = post("/predict_batch", {"images": X_B[:32].tolist()})
    batch_labels = [p.get("label") for p in body.get("predictions", [])] if isinstance(body, dict) else []
    oks["batch"] = status == 200 and batch_labels == NAMES_B[:32] and body.get("model_version") == META_18["weights_sha256"][:12]
    verdict("B7.18", oks["batch"], f"{label} POST /predict_batch de 32 images : les 32 noms du notebook.",
            f"{label} POST /predict_batch de 32 images : attendu 200, {{predictions: [{{label, probability}}] × 32, "
            f"model_version}} aux noms du notebook ; reçu {status}, {sum(a == b for a, b in zip(batch_labels, NAMES_B))} "
            "noms justes.")
    return oks


with checked("B7.18"):
    if ARTIFACT_B is None:
        print("⏳ Ex B7.18 : l'API sert l'artefact de B7.17 : fais d'abord passer B7.17.")
        raise StopCheck
    wbp.write_files(PROJECT, {"src/cifarprod/serve.py": SERVE_PY})
    serve_18 = wbp.import_module(PROJECT, "serve")
    META_18 = json.loads((ARTIFACT_B / "meta.json").read_text(encoding="utf-8"))
    ok_18["service"] = serve_18.Service(ARTIFACT_B).version == META_18["weights_sha256"][:12]
    verdict("B7.18", ok_18["service"], "Service charge l'artefact ; sa version : les 12 premiers caractères du SHA-256 "
            "des poids.", "Service(dossier).version doit valoir meta['weights_sha256'][:12].")
    if importlib.util.find_spec("fastapi") is None:
        print("ℹ️ FastAPI n'est pas installé ici : seul le repli http.server est vérifié.")
    else:
        from fastapi.testclient import TestClient
        app_18 = serve_18.create_app(ARTIFACT_B)
        client_18 = TestClient(app_18, raise_server_exceptions=False)

        def as_body_18(response):
            try:
                return response.json()
            except ValueError:
                return response.text

        with logged_18() as kept_records_18:              # the server's log lines are kept, not printed
            ok_18.update({f"fastapi {k}": v for k, v in api_cases_18(
                lambda path: (lambda r: (r.status_code, as_body_18(r)))(client_18.get(path)),
                lambda path, body: (lambda r: (r.status_code, as_body_18(r)))(client_18.post(path, json=body)),
                "FastAPI :", kept_records_18).items()})
        kept_18 = app_18.state.service.model
        app_18.state.service.model = Failing_18()
        try:
            with logged_18() as records_18:
                r_18 = client_18.post("/predict", json={"pixels": X_B[0].tolist()})
        finally:
            app_18.state.service.model = kept_18
        traced_18 = any(r.levelname == "ERROR" and r.exc_info for r in records_18)
        ok_18["fastapi 500"] = r_18.status_code == 500 and as_body_18(r_18) == {"detail": "internal error"} and traced_18
        verdict("B7.18", ok_18["fastapi 500"], "FastAPI : un modèle qui plante donne un 500 propre, {\"detail\": "
                "\"internal error\"}, et une ligne log.exception (ici retenue, pas affichée).",
                f"FastAPI : un modèle qui lève une exception doit donner 500 et le JSON {{\"detail\": \"internal "
                f"error\"}} (sans trace pour le client), et une ligne log.exception côté serveur ; reçu {r_18.status_code} "
                f"{str(as_body_18(r_18))[:120]}, {'une' if traced_18 else 'aucune'} ligne ERROR avec sa trace.")

with checked("B7.18"):                                    # the fallback, started and ALWAYS stopped by this cell
    if "serve_18" not in globals() or ARTIFACT_B is None:
        raise StopCheck
    import requests
    server_18 = serve_18.make_server(ARTIFACT_B, "127.0.0.1", 0)
    thread_18 = threading.Thread(target=server_18.serve_forever, daemon=True)
    thread_18.start()
    try:
        url_18 = f"http://127.0.0.1:{server_18.server_address[1]}"
        session_18 = requests.Session()

        def answer_18(response):
            try:
                return response.status_code, response.json()
            except ValueError:
                return response.status_code, response.text

        with logged_18() as kept_records_18:
            ok_18.update({f"stdlib {k}": v for k, v in api_cases_18(
                lambda path: answer_18(session_18.get(url_18 + path, timeout=10)),
                lambda path, body: answer_18(session_18.post(url_18 + path, json=body, timeout=10)),
                "http.server :", kept_records_18).items()})
        kept_18 = server_18.service.model
        server_18.service.model = Failing_18()
        try:
            with logged_18():
                status_18, body_18 = answer_18(session_18.post(url_18 + "/predict", json={"pixels": X_B[0].tolist()},
                                                               timeout=10))
        finally:
            server_18.service.model = kept_18
        ok_18["stdlib 500"] = status_18 == 500 and body_18 == {"detail": "internal error"}
        verdict("B7.18", ok_18["stdlib 500"], "http.server : un 500 propre aussi.",
                f"http.server : un modèle qui lève une exception doit donner 500 {{\"detail\": \"internal error\"}} ; reçu "
                f"{status_18} {str(body_18)[:120]}.")
        status_18, _ = answer_18(session_18.get(url_18 + "/nowhere", timeout=10))
        ok_18["stdlib 404"] = status_18 == 404
        verdict("B7.18", ok_18["stdlib 404"], "http.server : 404 pour une autre route.",
                f"http.server : GET /nowhere doit rendre 404, reçu {status_18}.")
    finally:
        if "session_18" in globals():
            session_18.close()                               # the keep-alive connection: its handler thread ends with it
        server_18.shutdown()
        server_18.server_close()
        thread_18.join(10)

with checked("B7.18"):                                    # running(): always stopped, even after an error
    if "serve_18" not in globals() or ARTIFACT_B is None:
        raise StopCheck
    import requests
    backends_18 = ["stdlib"] + (["fastapi"] if importlib.util.find_spec("fastapi") and importlib.util.find_spec("uvicorn") else [])
    for backend_18 in backends_18:
        threads_before_18 = threading.active_count()
        port_18, health_18 = None, None
        try:
            with serve_18.running(ARTIFACT_B, backend=backend_18) as url_18:
                port_18 = int(str(url_18).rsplit(":", 1)[-1])
                health_18 = requests.get(f"{url_18}/health", timeout=10).status_code
                raise KeyError("an error inside the block (a test of the check cell)")
        except KeyError:
            pass
        time.sleep(0.2)
        stopped_18 = port_18 is not None and not wbp.port_open(port_18)
        ok_18[f"running {backend_18}"] = health_18 == 200 and stopped_18 and threading.active_count() <= threads_before_18
        verdict("B7.18", ok_18[f"running {backend_18}"], f"running(backend=\"{backend_18}\") : /health répond, puis le "
                "serveur s'arrête et libère son port, même après une erreur dans le bloc.",
                f"running(backend=\"{backend_18}\") : /health {health_18}, port {port_18} "
                f"{'encore ouvert' if port_18 and not stopped_18 else 'fermé'}, threads {threads_before_18} → "
                f"{threading.active_count()} : attendu 200, puis le serveur arrêté (shutdown, server_close, ou "
                "should_exit pour uvicorn) dans un finally.")
        url_a_18 = url_b_18 = None
        try:
            with serve_18.running(ARTIFACT_B, backend=backend_18) as url_a_18, \
                    serve_18.running(ARTIFACT_B, backend=backend_18) as url_b_18:
                pass
            twice_18 = url_a_18 is not None and url_b_18 is not None and url_a_18 != url_b_18
        except Exception as err_18:  # noqa: BLE001 - the learner's code: "Address already in use", a server that did not start
            twice_18, url_b_18 = False, f"une erreur ({type(err_18).__name__} : {str(err_18)[:80]})"
        ok_18[f"free port {backend_18}"] = twice_18
        verdict("B7.18", ok_18[f"free port {backend_18}"], f"running(backend=\"{backend_18}\") deux fois à la fois : deux "
                "ports libres.", f"deux running(backend=\"{backend_18}\") à la fois donnent {url_a_18} et {url_b_18} : un "
                "port en dur ne sert qu'une fois (port 0 : le système choisit un port libre).")
    try:
        with wbp.serve_process(PROJECT / "src", ARTIFACT_B, backend="stdlib", timeout=60) as proc_18:
            code_health_18 = requests.get(f"{proc_18.url}/health", timeout=10).status_code
    except RuntimeError as err_18:
        code_health_18 = f"pas de réponse ({str(err_18)[:200]})"
    ok_18["main"] = code_health_18 == 200
    verdict("B7.18", ok_18["main"], "python -m cifarprod.serve --artifact ... --port ... sert l'artefact dans un autre "
            "processus (arrêté ensuite).", f"python -m cifarprod.serve --artifact ... --port ... --backend stdlib : "
            f"/health {code_health_18} ; attendu 200.")
    r_18 = wbp.run([sys.executable, "-m", "cifarprod.serve", "--artifact", scratch("empty"), "--port",
                    str(wbp.free_port())], PROJECT, src=[PROJECT / "src"], timeout=60)
    ok_18["main error"] = run_line("B7.18", r_18, r_18.returncode == 1 and bool(r_18.stderr_lines("ERROR")),
                                   "un dossier sans artefact : code 1 et une ligne ERROR.",
                                   "attendu, pour un dossier sans artefact, le code 1 et une ligne ERROR")
if ok_18 and all(ok_18.values()) and len(ok_18) >= 9:
    print("✅ Ex B7.18 : ton API répond comme le notebook, refuse proprement ce qui est mal formé, et s'arrête toujours.")'''

# =====================================================================================================================
# 🔮 B7.19 · the latency of the API
# =====================================================================================================================
TODO_19 = r'''http_19 = ...    # a) p95 of POST /predict (one image) ÷ p95 of a direct call of the model on one image: "A", "B" or "C"
batch_19 = ...   # b) p95 of POST /predict_batch (32 images) ÷ p95 of POST /predict (one image): "A", "B" or "C"'''

EXPERIMENT_19 = r'''import requests

start_19 = time.perf_counter()
CHOICES_19 = {"a": {"A": "< 2", "B": "2 à 20", "C": "> 20"}, "b": {"A": "< 3", "B": "3 à 32", "C": "> 32"}}


def category_19(question, ratio):
    bounds = {"a": (2, 20), "b": (3, 32)}[question]
    return "A" if ratio < bounds[0] else "B" if ratio <= bounds[1] else "C"


folder_19 = wbp.reference_artifact(scratch("artifact19") / "vgg")
model_19, info_19 = wbp.reference_model()
x_19 = model_input(X_B, info_19["mean"], info_19["std"])
with wbp.torch_state_kept(), torch.inference_mode():        # one thread, as the server: one image at a time
    torch.set_num_threads(1)
    direct1_19 = wbp.latencies(lambda: model_19(x_19[:1]), 200, 20)
    direct32_19 = wbp.latencies(lambda: model_19(x_19[:32]), 50, 5)
one_body_19, batch_body_19 = {"pixels": X_B[0].tolist()}, {"images": X_B[:32].tolist()}
with wbp.serve_process(wbp.reference_package() / "src", folder_19, backend="auto", env={"OMP_NUM_THREADS": "1"}) as server_19:
    with requests.Session() as session_19:                  # closed at the end: no connection left open
        http1_19 = wbp.latencies(lambda: session_19.post(f"{server_19.url}/predict", json=one_body_19, timeout=30), 200, 20)
        http32_19 = wbp.latencies(lambda: session_19.post(f"{server_19.url}/predict_batch", json=batch_body_19, timeout=30),
                                  40, 3)
rows_19 = {"appel direct, 1 image": (direct1_19, 1), "appel direct, batch de 32": (direct32_19, 32),
           "POST /predict, 1 image": (http1_19, 1), "POST /predict_batch, 32 images": (http32_19, 32)}
print(f"{'':34s} {'p50 (ms)':>9s} {'p95 (ms)':>9s} {'images/s':>9s}")
for name_19, (ms_19, n_19) in rows_19.items():
    p50_19, p95_19 = np.percentile(ms_19, [50, 95])
    print(f"{name_19:34s} {p50_19:9.2f} {p95_19:9.2f} {n_19 * 1000 / p50_19:9.0f}")
ratio_a_19 = np.percentile(http1_19, 95) / np.percentile(direct1_19, 95)
ratio_b_19 = np.percentile(http32_19, 95) / np.percentile(http1_19, 95)
print(f"a) p95 de /predict ÷ p95 de l'appel direct : {fr(ratio_a_19, 1)} (catégorie {category_19('a', ratio_a_19)})")
print(f"b) p95 de /predict_batch (32 images) ÷ p95 de /predict : {fr(ratio_b_19, 1)} (catégorie {category_19('b', ratio_b_19)}) ; "
      f"débit : {fr(32 * np.percentile(http1_19, 50) / np.percentile(http32_19, 50), 1)} fois celui des requêtes d'une image")
print(f"ℹ️ vitesse de cette machine : un appel direct à une image, p50 {fr(np.percentile(direct1_19, 50), 2)} ms (machine de "
      f"génération du chapitre : «DIRECT_P50» ms) ; les catégories sont des rapports, qui dépendent peu de la machine. "
      f"Mesures en {fr(time.perf_counter() - start_19, 0)} s, serveur arrêté.")
if (category_19("a", ratio_a_19), category_19("b", ratio_b_19)) != (str(http_19).strip().upper(), str(batch_19).strip().upper()):
    print("ℹ️ ta prédiction et la mesure de cette machine diffèrent : la vérification ci-dessous juge la catégorie "
          "mesurée sur la machine de génération ; compare aussi aux mesures de plusieurs exécutions (05_solutions.md).")
wb.check("B7.19a", http_19)
wb.check("B7.19b", batch_19)'''

EXPERIMENT_19_OWN = r'''# The same measures on YOUR API (B7.18), if it passed: nothing is checked here
if "ok_18" in globals() and ok_18 and all(ok_18.values()):
    try:
        with serve_18.running(ARTIFACT_B) as url_own_19, requests.Session() as s_own_19:
            own1_19 = wbp.latencies(lambda: s_own_19.post(f"{url_own_19}/predict", json=one_body_19, timeout=30), 100, 10)
            own32_19 = wbp.latencies(lambda: s_own_19.post(f"{url_own_19}/predict_batch", json=batch_body_19, timeout=30), 20, 2)
        print(f"ton API : /predict p50 {fr(np.percentile(own1_19, 50), 2)} ms, p95 {fr(np.percentile(own1_19, 95), 2)} ms ; "
              f"/predict_batch p50 {fr(np.percentile(own32_19, 50), 1)} ms, p95 {fr(np.percentile(own32_19, 95), 1)} ms.")
        slow_19 = np.percentile(own1_19, 50) / np.percentile(http1_19, 50)
        if slow_19 > 3:
            print(f"ℹ️ ton API est {fr(slow_19, 0)} fois plus lente que celle de référence pour la même requête : ≈ 40 ms de "
                  "plus par requête est le signe d'un accusé de réception retardé (algorithme de Nagle) : un socket sans "
                  "TCP_NODELAY (uvicorn avec sockets=[...]), ou des en-têtes et un corps envoyés en plusieurs fois ; "
                  "compare avec running(..., backend=\"stdlib\").")
    except Exception as error_19:  # noqa: BLE001 - the learner's code: say it, nothing else depends on it
        print(f"ℹ️ ton API n'a pas pu être mesurée : {type(error_19).__name__} : {str(error_19)[:160]}")
else:
    print("⏳ Ex B7.19 : la mesure de ton API attend B7.18 (toutes ses vérifications).")'''

# =====================================================================================================================
# 🔬 B7.20 · faster inference on CPU
# =====================================================================================================================
TODO_20 = r'''import copy


def latency_20(fn, x, n=50, warmup=5):
    """(p50, p95), two Python floats in ms, of n calls fn(x) measured with time.perf_counter, after `warmup` calls of
    fn(x) that are not measured."""
    raise NotImplementedError("latency_20")


def bfloat16_20(model):
    """A function x -> logits: a COPY of the model converted to torch.bfloat16, applied to x converted to bfloat16; the
    logits are returned as float32. The model given is not changed."""
    raise NotImplementedError("bfloat16_20")


def quantized_20(model):
    """A COPY of the model whose nn.Linear layers are quantized to int8 by torch.ao.quantization.quantize_dynamic
    (dtype=torch.qint8); the model given is not changed."""
    raise NotImplementedError("quantized_20")


def agreement_20(a, b):
    """(share, gap) for two tensors of logits (N, C): the share of the rows whose argmax agree and the largest absolute
    difference between them, two Python floats."""
    raise NotImplementedError("agreement_20")'''

SOLUTION_20 = r'''import copy


def latency_20(fn, x, n=50, warmup=5):
    """(p50, p95), two Python floats in ms, of n calls fn(x) measured with time.perf_counter, after `warmup` calls of
    fn(x) that are not measured."""
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn(x)
        times.append(1000 * (time.perf_counter() - start))
    p50, p95 = np.percentile(times, [50, 95])
    return float(p50), float(p95)


def bfloat16_20(model):
    """A function x -> logits: a COPY of the model converted to torch.bfloat16, applied to x converted to bfloat16; the
    logits are returned as float32. The model given is not changed."""
    low = copy.deepcopy(model).to(torch.bfloat16)
    return lambda x: low(x.to(torch.bfloat16)).float()


def quantized_20(model):
    """A COPY of the model whose nn.Linear layers are quantized to int8 by torch.ao.quantization.quantize_dynamic
    (dtype=torch.qint8); the model given is not changed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)      # « torch.ao.quantization is deprecated » (2.11)
        return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)


def agreement_20(a, b):
    """(share, gap) for two tensors of logits (N, C): the share of the rows whose argmax agree and the largest absolute
    difference between them, two Python floats."""
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    return float((a.argmax(dim=1) == b.argmax(dim=1)).float().mean()), float((a - b).abs().max())'''

CHECK_20 = r'''tools_ok_20 = False
warnings.filterwarnings("ignore", message=r".*torch\.ao\.quantization is deprecated.*")
with checked("B7.20"):
    calls_20 = []

    def probe_20(x):                                      # slow while warming up, then fast: what is measured?
        calls_20.append(time.perf_counter())
        time.sleep(0.03 if len(calls_20) <= 3 else 0.001)

    got_20 = latency_20(probe_20, None, n=8, warmup=3)
    ok_lat_20 = (isinstance(got_20, tuple) and len(got_20) == 2 and all(type(v) is float for v in got_20)
                 and len(calls_20) == 11 and 0.5 < got_20[0] < 15 and got_20[0] <= got_20[1] < 25)
    verdict("B7.20", ok_lat_20, f"latency_20 : (p50, p95) en ms, les 3 appels à blanc écartés ({fr(got_20[0], 1)} et "
            f"{fr(got_20[1], 1)} ms pour un appel d'environ 1 ms).",
            f"latency_20(fn, x, n=8, warmup=3) sur une fonction qui dort 30 ms pendant les 3 premiers appels puis 1 ms : "
            f"attendu 11 appels et (p50, p95) en ms, deux floats d'environ 1 ms (les 3 appels à blanc, lents, ne sont pas "
            f"mesurés) ; reçu {len(calls_20)} appels et {got_20!r}.")
    model_20 = wbp.reference_model()[0]
    x_20 = model_input(X_B[:16], REF_INFO["mean"], REF_INFO["std"])
    with torch.inference_mode():
        ref_20 = model_20(x_20)
        low_20 = bfloat16_20(model_20)(x_20)
    ok_bf_20 = (isinstance(low_20, torch.Tensor) and low_20.dtype == torch.float32 and tuple(low_20.shape) == (16, 10)
                and next(model_20.parameters()).dtype == torch.float32 and 0 < float((low_20 - ref_20).abs().max()) < 1)
    verdict("B7.20", ok_bf_20, "bfloat16_20 : des logits float32 calculés en bfloat16 (proches, pas égaux), le modèle "
            "d'origine intact.", f"bfloat16_20 : attendu une fonction qui rend des logits float32 (16, 10) calculés en "
            f"bfloat16, sans changer le modèle donné ; reçu {getattr(low_20, 'dtype', type(low_20).__name__)}, modèle "
            f"{next(model_20.parameters()).dtype}.")
    model_20 = wbp.reference_model()[0]                     # a new copy: bfloat16_20 may have changed the first one
    q_20 = quantized_20(model_20)
    kinds_20 = {type(m).__module__ + "." + type(m).__name__ for m in q_20.modules()}
    linear_q_20 = any("quantized" in k and k.endswith("Linear") for k in kinds_20)
    intact_20 = type(model_20[-1]) is torch.nn.Linear and q_20 is not model_20
    convs_20 = any(type(m) is torch.nn.Conv2d for m in q_20.modules())
    ok_q_20 = linear_q_20 and intact_20 and convs_20
    verdict("B7.20", ok_q_20, "quantized_20 : une copie dont la couche dense est en int8 dynamique, les convolutions en "
            "float32, le modèle d'origine intact.", "quantized_20 : attendu une COPIE dont les nn.Linear sont "
            "quantifiées (quantize_dynamic, qint8) et les convolutions inchangées ; "
            + "; ".join(part for part in ["aucune couche dense quantifiée" if not linear_q_20 else "",
                                          "le modèle donné a changé (inplace=True, ou pas de copie ?)" if not intact_20 else "",
                                          "les convolutions ont changé" if not convs_20 else ""] if part) + ".")
    a_20 = torch.tensor([[3.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    b_20 = torch.tensor([[2.5, 0, 0], [0, 2.0, 0], [0, 0, 1.0], [0, 1.5, 0]])
    got_ag_20 = agreement_20(a_20, b_20)
    ok_ag_20 = (isinstance(got_ag_20, tuple) and len(got_ag_20) == 2 and all(type(v) is float for v in got_ag_20)
                and abs(got_ag_20[0] - 0.75) < 1e-9 and abs(got_ag_20[1] - 1.5) < 1e-6)
    verdict("B7.20", ok_ag_20, "agreement_20 : 0,75 et 1,5 sur un petit cas (trois argmax sur quatre en accord).",
            f"agreement_20 sur quatre lignes dont trois argmax en accord et un écart maximal de 1,5 : attendu (0.75, 1.5), "
            f"deux floats ; reçu {got_ag_20!r}.")
    tools_ok_20 = bool(ok_lat_20 and ok_bf_20 and ok_q_20 and ok_ag_20)'''

EXPERIMENT_20 = r'''if not tools_ok_20:
    print("⏳ Ex B7.20 : l'expérience attend tes quatre fonctions justes (cellule précédente).")
else:
    with checked("B7.20"), wbp.torch_state_kept(), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        start_20 = time.perf_counter()
        fp32_20 = wbp.reference_model()[0]
        bf16_20, int8_20 = bfloat16_20(fp32_20), quantized_20(fp32_20)
        X1000_20, Y1000_20 = wbp.test_images(100)
        x_all_20 = model_input(X1000_20, REF_INFO["mean"], REF_INFO["std"])
        x1_20, x32_20 = x_all_20[:1], x_all_20[:32]
        threads_20 = torch.get_num_threads()

        def timed_20(fn, x, mode="inference", threads=None):
            torch.set_num_threads(threads or threads_20)
            try:
                if mode == "inference":
                    with torch.inference_mode():
                        return latency_20(fn, x)
                if mode == "no_grad":
                    with torch.no_grad():
                        return latency_20(fn, x)
                return latency_20(fn, x)                # autograd on: a graph is built at each call
            finally:
                torch.set_num_threads(threads_20)

        grid_20 = [("batch 1, float32, inference_mode", fp32_20, x1_20, "inference", None),
                   ("batch 1, float32, no_grad", fp32_20, x1_20, "no_grad", None),
                   ("batch 1, float32, autograd actif", fp32_20, x1_20, "autograd", None),
                   ("batch 1, float32, 1 thread", fp32_20, x1_20, "inference", 1),
                   ("batch 1, bfloat16", bf16_20, x1_20, "inference", None),
                   ("batch 1, int8 dynamique", int8_20, x1_20, "inference", None),
                   ("batch 32, float32, inference_mode", fp32_20, x32_20, "inference", None),
                   ("batch 32, float32, 1 thread", fp32_20, x32_20, "inference", 1),
                   ("batch 32, bfloat16", bf16_20, x32_20, "inference", None),
                   ("batch 32, int8 dynamique", int8_20, x32_20, "inference", None)]
        print(f"{threads_20} threads par défaut ici. {'':10s} {'p50 (ms)':>9s} {'p95 (ms)':>9s} {'images/s':>9s}")
        for name_20, fn_20, x_20, mode_20, th_20 in grid_20:
            p50_20, p95_20 = timed_20(fn_20, x_20, mode_20, th_20)
            print(f"{name_20:36s} {p50_20:9.2f} {p95_20:9.2f} {len(x_20) * 1000 / p50_20:9.0f}")
        with torch.inference_mode():
            logits_20 = {"float32": fp32_20(x_all_20), "bfloat16": bf16_20(x_all_20), "int8 dynamique": int8_20(x_all_20)}
        print("\nSur 1 000 images de test de CIFAR-10 :")
        for name_20, out_20 in logits_20.items():
            share_20, gap_20 = agreement_20(logits_20["float32"], out_20)
            acc_20 = float((out_20.argmax(dim=1).numpy() == Y1000_20).mean())
            print(f"  {name_20:15s} accuracy {fr(acc_20, 3)} ; accord des argmax avec float32 {fr(100 * share_20, 1)} % ; "
                  f"écart maximal des logits {gap_20:.3g}")
        print(f"(mesures en {fr(time.perf_counter() - start_20, 0)} s ; aucune n'est enregistrée : elles dépendent de la "
              "machine, et d'une exécution à l'autre)")'''

# =====================================================================================================================
# 📦 B7.21 · torch.export
# =====================================================================================================================
SKEL_EXPORT = '''"""Export a model with torch.export, and check that the export computes what the model computes."""

from __future__ import annotations

from pathlib import Path

from .artifact import load_artifact


def export_model(model, path, max_batch=1024):
    """Export `model` in eval() mode (switch it to eval() first) with torch.export.export, the batch size free from 1 to
    `max_batch` (dynamic_shapes on the dimension 0 of the argument "input" of nn.Sequential), from an example batch of
    2 images (2, 3, 32, 32); save the ExportedProgram in `path` (.pt2, torch.export.save) and return the path (a Path)."""
    raise NotImplementedError("export_model")


def export_artifact(folder, max_batch=1024):
    """The model of the artifact `folder` (load_artifact) exported into folder / "model.pt2"; returns that path."""
    raise NotImplementedError("export_artifact")


def max_gap(model, exported, x):
    """The largest absolute difference between the logits of `model` and those of `exported` (a callable) on the batch
    x (float32 (N, 3, 32, 32)), computed without a graph, as a Python float."""
    raise NotImplementedError("max_gap")
'''

TODO_21 = file_cell("EXPORT_PY", SKEL_EXPORT, "Ex B7.21: src/cifarprod/export.py")

SOLUTION_21 = file_cell("EXPORT_PY", reference_file("src/cifarprod/export.py"), "Ex B7.21: src/cifarprod/export.py")

CHECK_21 = r'''ok_21 = {}
GAP_21 = None                                            # the largest gap over the batches 1, 7 and 64
with checked("B7.21"), warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*given buffer is not writable.*")
    if ARTIFACT_B is None:
        print("⏳ Ex B7.21 : l'export part de l'artefact de B7.17 : fais d'abord passer B7.17.")
        raise StopCheck
    wbp.write_files(PROJECT, {"src/cifarprod/export.py": EXPORT_PY})
    export_21 = wbp.import_module(PROJECT, "export")
    with wbp.torch_state_kept():
        path_21 = export_21.export_artifact(ARTIFACT_B)
    ok_21["file"] = path_21 is not None and Path(path_21).resolve() == (ARTIFACT_B / "model.pt2").resolve() \
        and Path(path_21).is_file()
    if not ok_21["file"]:
        verdict("B7.21", False, "", f"export_artifact doit écrire et rendre {ARTIFACT_B.name}/model.pt2 ; rendu {path_21!r}.")
        raise StopCheck
    sizes_21 = (Path(path_21).stat().st_size, (ARTIFACT_B / "model.safetensors").stat().st_size)
    bytes_21 = [f"{n:_}".replace("_", " ") for n in sizes_21]
    verdict("B7.21", True, f"export_artifact écrit model.pt2 dans l'artefact ({bytes_21[0]} octets, contre "
            f"{bytes_21[1]} pour les poids seuls).", "")
    io_21 = scratch("export")
    x_21 = model_input(X_B[:64], INFO_B["mean"], INFO_B["std"]).numpy()
    np.save(io_21 / "x.npy", x_21)
    probe_21 = ("import json, sys, warnings\nsys.modules['cifarprod'] = None   # not importable here, even if installed\n"
                "import numpy as np\nimport torch\nwarnings.simplefilter('ignore')\n"
                "exported = torch.export.load(sys.argv[1]).module()\nx = torch.from_numpy(np.load(sys.argv[2]))\n"
                "with torch.inference_mode():\n    for b in (1, 7, 64):\n        np.save(f'{sys.argv[3]}/out{b}.npy', exported(x[:b]).numpy())\n"
                "refused = {}\nfor name, bad in {'1025': torch.zeros(1025, 3, 32, 32), '28': torch.zeros(2, 3, 28, 28)}.items():\n"
                "    try:\n        exported(bad)\n        refused[name] = None\n    except Exception as err:\n"
                "        refused[name] = str(err).splitlines()[0][:120]\n"
                "try:\n    import cifarprod\n    reachable = True\nexcept ImportError:\n    reachable = False\n"
                "print(json.dumps({'refused': refused, 'cifarprod': reachable}))\n")
    (io_21 / "reload_pt2.py").write_text(probe_21, encoding="utf-8")
    r_21 = wbp.run([sys.executable, io_21 / "reload_pt2.py", path_21, io_21 / "x.npy", io_21], scratch("elsewhere"))
    seen_21 = r_21.last_json() if r_21.ok else None
    if not run_line("B7.21", r_21, r_21.ok and isinstance(seen_21, dict) and not seen_21["cifarprod"],
                    "un processus SANS ton paquet recharge model.pt2 et calcule.",
                    "attendu un processus sans cifarprod qui recharge model.pt2 (torch.export.load) et calcule les "
                    "batches de 1, 7 et 64 images"):
        raise StopCheck
    with torch.inference_mode():                             # the model on the SAME batches as the export
        want_21 = {b: MODEL_B(torch.from_numpy(x_21[:b])).double().numpy() for b in (1, 7, 64)}
    gaps_21 = {b: float(np.abs(np.load(io_21 / f"out{b}.npy").astype(np.float64) - want_21[b]).max()) for b in (1, 7, 64)}
    GAP_21 = max(gaps_21.values())
    ok_21["equal"] = GAP_21 <= 1e-6
    verdict("B7.21", ok_21["equal"], "l'export calcule ce que calcule le modèle, à 1e-6 près, pour 1, 7 et 64 images "
            f"(écarts : {', '.join(f'{v:.2g}' for v in gaps_21.values())}).",
            f"écarts entre le modèle et model.pt2 : {', '.join(f'batch {b} : {v:.3g}' for b, v in gaps_21.items())} ; "
            "attendu au plus 1e-6 : la taille de batch est-elle libre (dynamic_shapes), le modèle en mode eval() ?")
    ok_21["guards"] = all(seen_21["refused"].values())
    verdict("B7.21", ok_21["guards"], f"les gardes refusent un batch de 1 025 images et une image de 28 × 28 "
            f"(« {seen_21['refused'].get('28')} »).",
            f"un modèle exporté doit refuser ce qui sort de ses gardes : batch de 1 025 images "
            f"({'refusé' if seen_21['refused'].get('1025') else 'accepté'}), image de 28 × 28 "
            f"({'refusée' if seen_21['refused'].get('28') else 'acceptée'}) : max_batch=1024 dans Dim.")
    trained_21 = wbp.reference_model()[0].train()           # a model left in training mode by its caller
    with wbp.torch_state_kept():
        path_t_21 = export_21.export_model(trained_21, io_21 / "train.pt2")
        exported_t_21 = torch.export.load(str(path_t_21)).module()
        with torch.inference_mode():
            gap_t_21 = float((wbp.reference_model()[0](torch.from_numpy(x_21)) - exported_t_21(torch.from_numpy(x_21))).abs().max())
    ok_21["eval"] = gap_t_21 <= 1e-6 and not trained_21.training
    verdict("B7.21", ok_21["eval"], "export_model met le modèle en mode eval() avant d'exporter (un modèle reçu en mode "
            "entraînement donne le même export).", f"export_model d'un modèle reçu en mode train() : écart {gap_t_21:.3g} "
            "avec le modèle en mode eval() (attendu au plus 1e-6) : en mode entraînement, la batchnorm de l'export "
            "normalise par les statistiques de chaque batch.")
    shifted_21 = wbp.reference_model()[0]
    with torch.no_grad():
        shifted_21[-1].bias += 0.01
    got_gap_21 = export_21.max_gap(wbp.reference_model()[0], shifted_21, torch.from_numpy(x_21[:8]))
    ok_21["max_gap"] = type(got_gap_21) is float and abs(got_gap_21 - 0.01) < 1e-5
    verdict("B7.21", ok_21["max_gap"], "max_gap : 0,01 pour un modèle dont on a décalé les biais de sortie de 0,01.",
            f"max_gap(modèle, modèle aux biais de sortie décalés de 0.01, x) : attendu le float 0.01, reçu {got_gap_21!r}.")
    if importlib.util.find_spec("onnxscript") and importlib.util.find_spec("onnxruntime"):
        import onnxruntime
        onnx_path_21 = io_21 / "model.onnx"
        torch.onnx.export(wbp.reference_model()[0], (torch.from_numpy(x_21[:2]),), str(onnx_path_21),
                          dynamic_shapes={"input": {0: torch.export.Dim("batch", min=1, max=1024)}})
        session_21 = onnxruntime.InferenceSession(str(onnx_path_21))
        out_onnx_21 = session_21.run(None, {session_21.get_inputs()[0].name: x_21})[0]
        print(f"ℹ️ ONNX (option) : écart maximal avec le modèle {float(np.abs(out_onnx_21 - want_21[64]).max()):.2g} sur 64 images.")
    else:
        print("ℹ️ ONNX (option) : onnxscript et onnxruntime ne sont pas installés ici (extra prod) : l'export ONNX n'est "
              "pas essayé.")
if ok_21 and all(ok_21.values()) and len(ok_21) == 5:
    print("✅ Ex B7.21 : ton modèle exporté se recharge sans ton code et calcule exactement ce que calcule le modèle.")'''

VERIFY_21 = r'''# Is the largest gap at most 1e-6? (nothing to type: the value comes from your code)
if GAP_21 is None:
    print("⏳ Ex B7.21 : l'écart attend ton export (cellule précédente).")
else:
    wb.check("B7.21", bool(GAP_21 <= 1e-6), computed=True)'''

RECORD_21 = r'''wb.record("B7.21", True, mistakes={"ton export s'écarte du modèle de plus de 1e-6 : la taille de batch est-elle libre, "
                                   "et le modèle en mode eval() quand tu l'exportes ?": False})'''
