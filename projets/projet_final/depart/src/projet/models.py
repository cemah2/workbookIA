"""The three models of the comparison (steps PF.5 and PF.6): naive, classical, deep learning.

The order is the protocol: a deep model is only worth its cost if it beats the classical baseline, which is only
worth something if it beats the naive one, on the same split, with the same metric (docs/protocole.md).

    naive = build_naive(config)            # majority class, mean, persistence...
    classical = build_classical(config)    # a scikit-learn Pipeline: preprocessor + model
    net = build_model(config, n_inputs=..., n_outputs=...)   # a torch.nn.Module
"""

from __future__ import annotations

from .config import Config


def build_naive(config: Config):
    """The naive predictor, an UNFITTED scikit-learn estimator (``fit``, ``predict``).

    Classification: always the majority class of the training split (``DummyClassifier(strategy="most_frequent")``,
    or ``"prior"`` for probabilities); regression: the mean or the median; forecasting: the persistence (the last value
    seen) or the seasonal persistence (22.23, 22.29); RL: the random policy and a hand-written rule (B8.12). It is the
    floor that every other model must beat, reported in every table (PF.8).
    """
    # TODO PF.5: sklearn.dummy, or a tiny estimator class of your own for the persistence
    raise NotImplementedError("TODO PF.5: build_naive")


def build_classical(config: Config):
    """The classical baseline: an UNFITTED scikit-learn ``Pipeline`` (preprocessor of ``preprocess.build_preprocessor``,
    then the model), so that the cross-validation refits the preprocessing in each fold (15.24, 24.17).

    Tabular: a gradient boosting (``HistGradientBoostingClassifier``, XGBoost or LightGBM, 14.26); text: TF-IDF and a
    logistic regression (13.30, B4.28); images: a linear probe on frozen embeddings (B1.13); series: a gradient boosting
    on lags. Its hyperparameters come from ``config.model_params`` (key ``"classical"``) or are left at their defaults.
    """
    # TODO PF.5: sklearn.pipeline.make_pipeline(build_preprocessor(config), <model>)
    raise NotImplementedError("TODO PF.5: build_classical")


def build_model(config: Config, *, n_inputs: int | None = None, n_outputs: int | None = None):
    """The deep learning model, a ``torch.nn.Module`` built from ``config.model`` and ``config.model_params``.

    Adapted to the data and justified in the report (PF.6): a pre-trained CNN or ViT fine-tuned (B1.17, B1.19), an
    encoder or a small LLM with LoRA (B4.25 to B4.28), an LSTM or a small Transformer for a series (22.23, B3.19), an MLP
    with embeddings for a table, a DQN or PPO agent for RL (B8.13, B8.20). Its output is the logits (classification) or
    the prediction (regression); same seed, same weights (``train.set_seed`` before the call).
    """
    # TODO PF.6: a few named variants selected by config.model (for the ablations of PF.7)
    raise NotImplementedError("TODO PF.6: build_model")


def n_parameters(model, trainable_only: bool = False) -> int:
    """The number of parameters of a ``torch.nn.Module`` (only those with ``requires_grad`` if ``trainable_only``).
    Provided."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
