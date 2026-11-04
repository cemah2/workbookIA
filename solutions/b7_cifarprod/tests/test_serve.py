"""Tests of the artifact and of the API: a reloaded artifact predicts like the model, a weights file swapped is
refused, the routes answer, a bad request gets a 422 without its input, both servers give the same answers."""

import json

import numpy as np
import pytest
import torch

from cifarprod.artifact import load_artifact, predict_images, save_artifact
from cifarprod.data import CLASSES, synthetic
from cifarprod.model import build_model

pytestmark = pytest.mark.filterwarnings("ignore:Using .httpx. with .starlette.testclient.:UserWarning")


@pytest.fixture
def artifact_dir(tmp_path):
    torch.manual_seed(0)
    model = build_model("cnn", (4, 8)).eval()
    return save_artifact(tmp_path / "artifact", model, arch="cnn", widths=(4, 8), mean=(0.5, 0.5, 0.5),
                         std=(0.25, 0.25, 0.25), classes=CLASSES, data={"sha256": "0" * 64}, metrics={"val_acc": 0.1})


@pytest.fixture
def images():
    return synthetic(5, seed=3)[0]


def test_artifact_reloads_the_same_predictions(artifact_dir, images):
    model, meta = load_artifact(artifact_dir)
    assert not model.training, "expected the reloaded model in eval() mode"
    labels, proba = predict_images(model, meta, images)
    assert proba.shape == (5, 10), f"expected probabilities of shape (5, 10), got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5), f"expected rows of sum 1, got {proba.sum(axis=1)}"
    assert labels == [CLASSES[k] for k in proba.argmax(axis=1)], f"expected the names of the argmax, got {labels}"
    one, _ = predict_images(model, meta, images[0])
    assert one == labels[:1], f"expected the same label for one image alone, got {one} and {labels[:1]}"


def test_artifact_meta_is_complete(artifact_dir):
    meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
    want = {"package_version", "arch", "widths", "input", "mean", "std", "classes", "data", "weights_sha256"}
    assert want <= set(meta), f"expected the keys {sorted(want)}, missing {sorted(want - set(meta))}"


def test_swapped_weights_are_refused(artifact_dir):
    from safetensors.torch import save_file
    save_file(build_model("cnn", (4, 8)).state_dict(), str(artifact_dir / "model.safetensors"))
    with pytest.raises(ValueError):
        load_artifact(artifact_dir)


def test_api_routes_and_validation(artifact_dir, images):
    from fastapi.testclient import TestClient

    from cifarprod.serve import create_app

    client = TestClient(create_app(artifact_dir))
    health = client.get("/health")
    assert health.status_code == 200, f"expected 200 on /health, got {health.status_code}"
    assert health.json()["package_version"] == "0.1.0", f"expected the version 0.1.0, got {health.json()}"
    r = client.post("/predict", json={"pixels": images[0].tolist()})
    assert r.status_code == 200 and r.json()["label"] in CLASSES, f"expected 200 and a class, got {r.status_code} {r.text[:200]}"
    bad = np.zeros((28, 28, 3), int).tolist()
    r = client.post("/predict", json={"pixels": bad})
    assert r.status_code == 422, f"expected 422 for a 28 x 28 image, got {r.status_code}"
    assert "input" not in r.text, f"expected a 422 without the input sent, got {r.text[:200]}"
    many = client.post("/predict_batch", json={"images": images.tolist()}).json()["predictions"]
    labels, _ = predict_images(*load_artifact(artifact_dir), images)
    assert [p["label"] for p in many] == labels, f"expected the labels {labels}, got {[p['label'] for p in many]}"


def test_stdlib_server_answers_like_fastapi(artifact_dir, images):
    import requests

    from cifarprod.serve import running

    labels, _ = predict_images(*load_artifact(artifact_dir), images)
    with running(artifact_dir, backend="stdlib") as url:
        got = [requests.post(f"{url}/predict", json={"pixels": image.tolist()}, timeout=10).json()["label"]
               for image in images]
        status = requests.post(f"{url}/predict", json={"pixels": "no"}, timeout=10).status_code
    assert got == labels, f"expected the labels {labels}, got {got}"
    assert status == 422, f"expected 422 for pixels that are not a list, got {status}"
