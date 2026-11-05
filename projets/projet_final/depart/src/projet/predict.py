"""The artifact of the delivered model and its predictor (step PF.11).

An artifact is a folder that is enough to predict, without the notebooks, the training code or the data (B7.17):

    artifact/
      model.safetensors (or model.pt, or model.joblib for a scikit-learn model)
      preprocessor.joblib (or its parameters in meta.json: normalization, vocabulary, classes...)
      meta.json        the config, the classes and their order, the version of the package, the metrics of
                       validation, the SHA-256 of the split and of the weights, the date

    predictor = load_artifact("artifacts/model")
    predictor.predict([{"text": "..."}])   ->  [{"label": "...", "score": 0.93}]
"""

from __future__ import annotations

from pathlib import Path

from .config import Config


def save_artifact(out_dir: str | Path, model, preprocessor, config: Config, *, metrics: dict,
                  classes: list | None = None) -> Path:
    """Write the artifact of ``model`` and its FITTED ``preprocessor`` into ``out_dir`` and return the path of
    ``meta.json``. Weights with ``safetensors`` (or ``torch.save`` of the ``state_dict``), never the pickled module;
    ``meta.json`` holds everything ``load_artifact`` needs to rebuild the model (``config.to_dict()``), the classes in
    the order of the outputs, ``metrics`` (validation), the version of the package and the SHA-256 of each file."""
    # TODO PF.11: B7.17 (artifact.py of cifarprod) is the model
    raise NotImplementedError("TODO PF.11: save_artifact")


class Predictor:
    """A loaded model that predicts from raw inputs, as an API receives them."""

    def __init__(self, model, preprocessor, meta: dict):
        self.model = model
        self.preprocessor = preprocessor
        self.meta = meta

    def predict(self, records: list[dict]) -> list[dict]:
        """One prediction per input record (a dict with the raw fields of an example, as ``load_examples`` gives them,
        without the target), in the same order: ``{"label": ..., "score": ..., "proba": {...}}`` for a classification
        (``score``: the probability of the predicted class, a confidence; ``proba``: the probability of EVERY class,
        keyed by ``str(class)``, what ROC-AUC and PR-AUC need), ``{"prediction": ...}`` for a regression. ValueError with a clear message
        for a record without a required field. The model in ``eval()`` mode, ``torch.inference_mode()``, the same
        preprocessing as in training (B7.16)."""
        # TODO PF.11: preprocessor.transform, model forward, softmax, classes of meta
        raise NotImplementedError("TODO PF.11: Predictor.predict")


def load_artifact(path: str | Path) -> Predictor:
    """The Predictor of the artifact folder ``path`` (written by ``save_artifact``). FileNotFoundError if a file is
    missing, ValueError if a SHA-256 of ``meta.json`` does not match its file."""
    # TODO PF.11: read meta.json, check the SHA-256, rebuild the model (models.build_model), load the weights
    raise NotImplementedError("TODO PF.11: load_artifact")
