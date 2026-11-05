"""Tests of the API (PF.11), marked « todo »: a fake predictor, so that they need neither the model nor the data."""

import pytest

from conftest import PACKAGES

pytestmark = pytest.mark.todo
PACKAGE = sorted(PACKAGES)[0]
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
serve = pytest.importorskip(f"{PACKAGE}.serve")


class FakePredictor:
    """Predicts « a » with a score of 1 for every record; enough to test the transport."""

    meta = {"version": "test"}

    def predict(self, records):
        return [{"label": "a", "score": 1.0} for _ in records]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(serve.create_app(FakePredictor()))


def test_health_answers(client):
    response = client.get("/health")
    assert response.status_code == 200, f"expected 200 on /health, got {response.status_code}"


def test_predict_answers_one_prediction_per_record(client):
    # TODO PF.11: records with the fields of YOUR data (the schema of create_app refuses the others)
    response = client.post("/predict", json={"records": [{}, {}]})
    assert response.status_code == 200, f"expected 200, got {response.status_code}: {response.text[:200]}"
    assert len(response.json()["predictions"]) == 2, "expected two predictions for two records"


def test_a_malformed_body_is_refused_with_422(client):
    response = client.post("/predict", json={"records": "not a list"})
    assert response.status_code == 422, f"expected 422 for a body that is not {{'records': [...]}}, got {response.status_code}"
