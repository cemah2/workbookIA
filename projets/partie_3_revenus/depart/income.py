"""Income over 50 000 $ from the 1994 census, with scikit-learn pipelines (mini-project of part III): YOUR module.

Everything that learns from the data lives in one scikit-learn ``Pipeline``: the
imputation of the missing values, the standardization, the encoding of the categories
and the model. A cross-validation clones and refits the whole pipeline in each fold, so
the validation rows never leak into the preprocessing. Two encodings:

- ``"onehot"``: numeric columns imputed by their median and standardized; categorical
  columns with a category ``"missing"`` for the missing values, then one-hot encoded
  (the categories seen fewer than ``MIN_FREQUENCY`` times grouped into one column; a
  category unknown at prediction time goes to that column, or to zeros when the feature
  has no rare category). For every model of chapters 13 and 14;
- ``"ordinal"``: categorical columns first, coded 0, 1, 2… (missing and unknown values
  as NaN), numeric columns unchanged: for ``HistGradientBoostingClassifier`` and its
  native categories (``build_pipeline`` tells the model which columns they are).

Two columns of the census are left out: ``fnlwgt``, a sampling weight of the survey (how
many people the row stands for, not a trait of the person), and ``education``, a copy of
``education-num`` in words. The helpers measure the models in cross-validation, choose a
decision threshold, audit the errors by subgroup with the learner's own
``mylearn.metrics``, and save the final pipeline with the versions of the libraries.

    pipeline = build_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_val)[:, 1]

Write every function that raises NotImplementedError (the step of the notebook is given
in the TODO). Keep the signatures and the docstrings: the notebook and the tests rely on
them. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import platform
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from mylearn import metrics

NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
DROPPED = ["fnlwgt", "education"]       # a sampling weight, and a copy of education-num in words
ENCODINGS = ("onehot", "ordinal")
MIN_FREQUENCY = 20                      # rarer categories (in the training rows) share one column
SCORING = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision"]
AGE_BINS = [25, 35, 45, 55, 65]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
REPORT_COLUMNS = ["n", "prevalence", "selection_rate", "accuracy", "tpr", "fpr", "precision", "roc_auc"]


def make_test_indices(y, test_size: float = 0.2, seed: int = 2026) -> np.ndarray:
    """Row positions of the test set, drawn once and for all before any modelling, stratified on y.

    The test part of ``sklearn.model_selection.train_test_split(np.arange(len(y)),
    test_size=test_size, stratify=y, random_state=seed)``, sorted in increasing order: the
    share of positives is the same (to one row) in the test set and in the other rows.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The labels, with at least two classes.
    test_size : float, default=0.2
        Share of the rows kept for the test, strictly between 0 and 1.
    seed : int, default=2026
        Seed of the split: the same seed gives the same test set.

    Returns
    -------
    np.ndarray of int
        The sorted test positions, without repetition, between 0 and ``len(y) - 1``.

    Raises
    ------
    ValueError
        If ``test_size`` is not strictly between 0 and 1, or if y has a single class.
    """
    raise NotImplementedError("make_test_indices")  # TODO MP3.1


def build_preprocessor(encoding: str = "onehot", numeric: Sequence[str] = NUMERIC,
                       categorical: Sequence[str] = CATEGORICAL) -> ColumnTransformer:
    """The preprocessing of the census columns, as an unfitted ``ColumnTransformer``.

    The columns not listed (``DROPPED``) are left out. Output feature names keep the names
    of the columns (``verbose_feature_names_out=False``), and the output is a dense array.
    The parts and the steps have fixed names (the notebook and the tests read them):

    - ``encoding="onehot"``: first the part ``"numeric"``, a ``Pipeline`` of two steps,
      ``"impute"`` (``SimpleImputer(strategy="median")``) then ``"scale"``
      (``StandardScaler()``); then the part ``"categorical"``, a ``Pipeline`` of
      ``"impute"`` (``SimpleImputer(strategy="constant", fill_value="missing")``) then
      ``"encode"`` (``OneHotEncoder(handle_unknown="infrequent_if_exist",
      min_frequency=MIN_FREQUENCY, sparse_output=False)``);
    - ``encoding="ordinal"``: first the part ``"categorical"``
      (``OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
      min_frequency=MIN_FREQUENCY)``: missing values stay NaN), then the part
      ``"numeric"``, the numeric columns unchanged (``"passthrough"``).

    Parameters
    ----------
    encoding : {"onehot", "ordinal"}, default="onehot"
    numeric, categorical : sequences of str
        The columns to keep, by kind (default ``NUMERIC`` and ``CATEGORICAL``).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Not fitted: it learns its medians, means, standard deviations and categories in
        ``fit``, from the rows it receives.

    Raises
    ------
    ValueError
        For an unknown encoding, no column at all, or a column in both lists.
    """
    raise NotImplementedError("build_preprocessor")  # TODO MP3.3


def build_pipeline(model, encoding: str = "onehot", numeric: Sequence[str] = NUMERIC,
                   categorical: Sequence[str] = CATEGORICAL) -> Pipeline:
    """``Pipeline([("prep", build_preprocessor(encoding, numeric, categorical)), ("model", model)])``.

    With ``encoding="ordinal"`` and a ``HistGradientBoostingClassifier``, the model is told
    which columns are categorical: ``model.set_params(categorical_features=[0, 1, …,
    len(categorical) - 1])`` (the categorical columns come first).

    Parameters
    ----------
    model : a scikit-learn classifier (not fitted)
    encoding, numeric, categorical
        As in ``build_preprocessor``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Not fitted, with two steps named ``"prep"`` and ``"model"``.
    """
    raise NotImplementedError("build_pipeline")  # TODO MP3.3


def cross_validate_all(pipelines: Mapping[str, Pipeline], X, y, cv, scoring: Sequence[str] = SCORING) -> dict:
    """Score each pipeline in cross-validation, on the same folds.

    For each pipeline, ``sklearn.model_selection.cross_validate(pipeline, X, y, cv=cv,
    scoring=list(scoring))``: a clone of the pipeline is fitted in each fold, so the
    pipelines received stay unfitted.

    Parameters
    ----------
    pipelines : mapping of str to Pipeline
        The name of each model and its pipeline.
    X : pandas.DataFrame
    y : array-like of shape (n_samples,)
    cv : a cross-validation splitter or a list of (train, validation) index pairs
        The same folds for every pipeline (give a splitter with a fixed random_state).
    scoring : sequence of str, default=SCORING
        Names of scikit-learn scorers.

    Returns
    -------
    dict
        name -> {scorer name: np.ndarray of the validation score of each fold, plus
        ``"fit_time"`` and ``"score_time"``: the time to fit the pipeline and the time to
        score the validation rows (predictions included), fold by fold, in seconds}.
    """
    raise NotImplementedError("cross_validate_all")  # TODO MP3.3


def f1_threshold(y_true, proba) -> tuple[float, float]:
    """The decision threshold that maximizes the F1 score, and that score.

    A person is predicted positive when ``proba >= threshold``. The candidates are the
    distinct values of ``proba``; the F1 score of each comes from
    ``sklearn.metrics.precision_recall_curve`` (F1 = 0 where precision and recall are 0).
    When several thresholds reach the maximum, the smallest of them is returned.

    Parameters
    ----------
    y_true : array-like of 0/1, shape (n_samples,)
        Both classes present.
    proba : array-like of float, shape (n_samples,)
        Estimated probability of the class 1 (out-of-fold predictions, never test ones).

    Returns
    -------
    (float, float)
        The threshold and its F1 score.

    Raises
    ------
    ValueError
        If the lengths differ, if y_true is not made of 0 and 1 (both present), or if
        proba contains NaN.
    """
    raise NotImplementedError("f1_threshold")  # TODO MP3.6


def age_groups(age) -> np.ndarray:
    """The age band of each person: ``AGE_LABELS`` cut at ``AGE_BINS`` ("<25", "25-34", …, "65+").

    An age equal to a bound starts the next band (25 is in "25-34").

    Parameters
    ----------
    age : array-like of numbers, shape (n_samples,)

    Returns
    -------
    np.ndarray of str (dtype object), shape (n_samples,)

    Raises
    ------
    ValueError
        If an age is missing (NaN).
    """
    raise NotImplementedError("age_groups")  # TODO MP3.7


def subgroup_report(y_true, y_pred, proba, groups) -> pd.DataFrame:
    """The errors of a classifier, group by group (a fairness audit).

    For each group (missing group values become ``"missing"``; groups sorted by name),
    with its confusion matrix ``mylearn.metrics.confusion_matrix(y, pred, labels=[0, 1])``:

    - ``n``: number of people; ``prevalence``: share of true positives (y = 1);
    - ``selection_rate``: share predicted positive; ``accuracy``;
    - ``tpr`` = TP / (TP + FN) (the recall), ``fpr`` = FP / (FP + TN),
      ``precision`` = TP / (TP + FP): NaN when the denominator is 0;
    - ``roc_auc``: ``mylearn.metrics.roc_auc(y, proba)`` on the group, NaN when the group
      holds a single class.

    Parameters
    ----------
    y_true, y_pred : array-like of 0/1, shape (n_samples,)
    proba : array-like of float, shape (n_samples,)
        The probabilities that gave y_pred.
    groups : array-like, shape (n_samples,)
        The group of each person (sex, age band…).

    Returns
    -------
    pandas.DataFrame
        One row per group (the index, named ``"group"``), the columns ``REPORT_COLUMNS``.

    Raises
    ------
    ValueError
        If the lengths differ.
    """
    raise NotImplementedError("subgroup_report")  # TODO MP3.7


def library_versions() -> dict:
    """The versions of Python and of the libraries the pipeline depends on.

    Returns
    -------
    dict
        Keys ``"python"``, ``"numpy"``, ``"pandas"``, ``"scikit-learn"``, ``"joblib"``;
        values: the version strings (``platform.python_version()``, ``np.__version__``…).
    """
    raise NotImplementedError("library_versions")  # TODO MP3.8


def save_model(pipeline: Pipeline, path, metadata: Mapping | None = None) -> Path:
    """Save a fitted pipeline with joblib, together with the library versions and metadata.

    The file holds the dict ``{"pipeline": pipeline, "versions": library_versions(),
    "metadata": dict(metadata or {})}``, written by ``joblib.dump``.

    Parameters
    ----------
    pipeline : a fitted scikit-learn Pipeline
    path : str or Path
        The file to write (usually ``model.joblib``).
    metadata : mapping, optional
        For example the threshold, the date and the cross-validation scores.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If the pipeline is not fitted.
    """
    raise NotImplementedError("save_model")  # TODO MP3.8


def load_model(path) -> tuple[Pipeline, dict]:
    """Load a pipeline saved by ``save_model``; warn when the scikit-learn version differs.

    joblib runs code while it loads a file (it is a pickle): load only files you trust.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    (Pipeline, dict)
        The pipeline, and ``{"versions": …, "metadata": …}`` as saved.

    Raises
    ------
    ValueError
        If the file does not hold what ``save_model`` writes.

    Warns
    -----
    UserWarning
        If the installed scikit-learn is not the version that saved the pipeline: the
        pipeline may then fail or predict differently.
    """
    raise NotImplementedError("load_model")  # TODO MP3.8
