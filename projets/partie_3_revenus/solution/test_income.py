"""Tests of income.py (mini-project of part III, reference solution).

    python -m pytest projets/partie_3_revenus/solution -q

scikit-learn, NumPy and pandas are the oracles: they check, the module builds. The data
come from the copy of the repository (``source="csv"``): no network, and 4 000 rows of the
training part only, never the frozen test rows.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import data
import income


@pytest.fixture(scope="module")
def census():
    """4 000 people of the training part (the test split of MP3.1 is left out): the tests stay fast."""
    X, y = data.load_census(source="csv")
    test = income.make_test_indices(y, test_size=0.2, seed=2026)
    rows = np.random.default_rng(7).choice(np.setdiff1d(np.arange(len(y)), test), 4000, replace=False)
    return X.iloc[rows], y[rows]


def assert_close(actual, expected, what, atol=1e-8):
    """Equal arrays, with the expected values on the first line of the message."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, equal_nan=True), (
        f"expected {what} = {np.round(expected, 6).tolist()}, got {np.round(actual, 6).tolist()}")


# --- the frozen test set --------------------------------------------------------------
def test_test_indices_are_sorted_distinct_reproducible_and_stratified():
    _, y = data.load_census(source="csv")
    first = income.make_test_indices(y, test_size=0.2, seed=2026)
    assert len(first) == int(np.ceil(0.2 * len(y))), f"expected ceil(0.2 * {len(y)}) test rows, got {len(first)}"
    assert np.all(np.diff(first) > 0), "expected positions sorted in increasing order, without repetition"
    assert np.array_equal(first, income.make_test_indices(y, 0.2, 2026)), "expected the same rows for the same seed"
    assert not np.array_equal(first, income.make_test_indices(y, 0.2, 2027)), "expected other rows for another seed"
    rest = np.setdiff1d(np.arange(len(y)), first)
    assert abs(y[first].mean() - y[rest].mean()) < 1e-3, (
        f"expected the same share of positives in the test and in the rest (stratified split), got "
        f"{y[first].mean():.4f} and {y[rest].mean():.4f}")


@pytest.mark.parametrize("test_size", [0, 1, 1.5], ids=["zero", "one", "above_one"])
def test_test_indices_refuse_a_test_size_outside_0_1(test_size):
    with pytest.raises(ValueError):
        income.make_test_indices(np.array([0, 1] * 10), test_size=test_size)


# --- the preprocessing: learnt in fit, from the rows received only ---------------------
def test_imputer_and_scaler_learn_from_the_training_rows_only(census):
    X, _ = census
    prep = income.build_preprocessor("onehot").fit(X.iloc[:3000])
    numeric = prep.named_transformers_["numeric"]
    assert_close(numeric.named_steps["impute"].statistics_, X.iloc[:3000][income.NUMERIC].median(),
                 "the medians of the 3 000 training rows")
    assert_close(numeric.named_steps["scale"].mean_, X.iloc[:3000][income.NUMERIC].mean(),
                 "the means of the 3 000 training rows")


def test_a_row_is_transformed_alike_alone_or_with_others(census):
    X, _ = census
    prep = income.build_preprocessor("onehot").fit(X.iloc[:3000])
    together = prep.transform(X.iloc[3000:3010])
    alone = np.vstack([prep.transform(X.iloc[[3000 + i]]) for i in range(10)])
    assert_close(alone, together, "the rows transformed together (transform must reuse what fit learnt)")


def test_onehot_columns_and_dropped_columns(census):
    X, _ = census
    prep = income.build_preprocessor("onehot").fit(X)
    names = list(prep.get_feature_names_out())
    assert names[:len(income.NUMERIC)] == income.NUMERIC, f"expected the numeric columns first, got {names[:5]}"
    assert not any(name.startswith(tuple(income.DROPPED)) for name in names if name != "education-num"), (
        "expected fnlwgt and education to be left out")
    assert "occupation_missing" in names, "expected a column for the missing occupations (category 'missing')"
    out = prep.transform(X)
    assert isinstance(out, np.ndarray) and not np.isnan(out).any(), "expected a dense array without NaN"


def test_unknown_categories_and_missing_values_do_not_break_prediction(census):
    X, y = census
    pipeline = income.build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)
    odd = X.iloc[:3].copy()
    odd["native-country"] = "Atlantis"                                  # a category never seen in fit
    odd.loc[odd.index[0], ["workclass", "occupation"]] = np.nan
    odd.loc[odd.index[1], "age"] = np.nan
    proba = pipeline.predict_proba(odd)[:, 1]
    assert proba.shape == (3,) and np.all((proba >= 0) & (proba <= 1)), f"expected 3 probabilities, got {proba}"


def test_ordinal_encoding_keeps_missing_values_for_the_native_categories(census):
    X, y = census
    pipeline = income.build_pipeline(HistGradientBoostingClassifier(max_iter=20, random_state=0), "ordinal")
    assert pipeline.named_steps["model"].categorical_features == list(range(len(income.CATEGORICAL))), (
        f"expected categorical_features = {list(range(len(income.CATEGORICAL)))} (the categorical columns come first)")
    pipeline.fit(X, y)
    out = pipeline.named_steps["prep"].transform(X)
    missing = X[income.CATEGORICAL].isna().to_numpy()
    assert np.array_equal(np.isnan(out[:, :len(income.CATEGORICAL)]), missing), (
        "expected NaN exactly where a categorical value is missing")
    assert_close(out[:, len(income.CATEGORICAL):], X[income.NUMERIC], "the numeric columns, unchanged")


@pytest.mark.parametrize("kwargs, why", [
    ({"encoding": "binary"}, "an unknown encoding"),
    ({"numeric": [], "categorical": []}, "no column at all"),
    ({"numeric": ["age"], "categorical": ["age"]}, "a column both numeric and categorical"),
], ids=["unknown_encoding", "no_column", "column_in_both_lists"])
def test_build_preprocessor_refuses_invalid_columns(kwargs, why):
    with pytest.raises(ValueError):
        income.build_preprocessor(**kwargs)


# --- cross-validation without leakage, reproducibility ---------------------------------
def test_cross_validation_refits_the_preprocessing_in_each_fold(census):
    X, y = census
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    pipeline = income.build_pipeline(LogisticRegression(max_iter=2000))
    scores = income.cross_validate_all({"logreg": pipeline}, X, y, folds, scoring=["roc_auc"])["logreg"]["roc_auc"]
    expected = []
    for train, val in folds:            # by hand: everything fitted on the training rows of the fold
        fitted = income.build_pipeline(LogisticRegression(max_iter=2000)).fit(X.iloc[train], y[train])
        expected.append(roc_auc_score(y[val], fitted.predict_proba(X.iloc[val])[:, 1]))
    assert_close(scores, expected, "the ROC-AUC of a pipeline refitted on the training rows of each fold")
    with pytest.raises(NotFittedError):
        pipeline.predict(X)                                             # the pipeline received stays unfitted


def test_cross_validate_all_returns_one_score_per_fold_and_metric(census):
    X, y = census
    results = income.cross_validate_all({"dummy": income.build_pipeline(DummyClassifier())}, X, y,
                                        StratifiedKFold(4, shuffle=True, random_state=0))
    assert set(results["dummy"]) == set(income.SCORING) | {"fit_time", "score_time"}, f"got the keys {sorted(results['dummy'])}"
    assert all(len(v) == 4 for v in results["dummy"].values()), "expected one value per fold (4 folds)"
    assert_close(results["dummy"]["roc_auc"], [0.5] * 4, "the ROC-AUC of a constant classifier")


def test_same_seed_same_probabilities(census):
    X, y = census
    make = lambda: income.build_pipeline(HistGradientBoostingClassifier(max_iter=30, random_state=0), "ordinal")
    first = make().fit(X, y).predict_proba(X.iloc[:200])[:, 1]
    second = make().fit(X, y).predict_proba(X.iloc[:200])[:, 1]
    assert_close(second, first, "the probabilities of a second fit with the same seed")


# --- the threshold ---------------------------------------------------------------------
def test_f1_threshold_is_the_best_of_all_thresholds():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.25).astype(int)
    proba = np.clip(0.25 + 0.3 * (y - 0.25) + rng.normal(0, 0.2, 500), 0, 1).round(2)
    threshold, best = income.f1_threshold(y, proba)
    candidates = np.unique(proba)
    scores = np.array([f1_score(y, proba >= t, zero_division=0) for t in candidates])
    assert np.isclose(best, scores.max()), f"expected the best F1 over all thresholds, {scores.max():.6f}, got {best}"
    assert np.isclose(threshold, candidates[np.flatnonzero(np.isclose(scores, scores.max()))[0]]), (
        f"expected the smallest threshold reaching the best F1, got {threshold}")


@pytest.mark.parametrize("y, proba, why", [
    ([0, 1, 1], [0.2, 0.8], "lengths differ"),
    ([1, 1, 1], [0.2, 0.8, 0.5], "a single class"),
    ([0, 1, 2], [0.2, 0.8, 0.5], "labels other than 0 and 1"),
    ([0, 1, 1], [0.2, np.nan, 0.5], "a NaN probability"),
], ids=["lengths", "one_class", "labels_0_1_2", "nan"])
def test_f1_threshold_refuses_invalid_inputs(y, proba, why):
    with pytest.raises(ValueError):
        income.f1_threshold(y, proba)


# --- the audit -------------------------------------------------------------------------
def test_age_groups_on_the_bounds():
    got = income.age_groups([17, 24, 25, 34, 35, 64, 65, 90]).tolist()
    expected = ["<25", "<25", "25-34", "25-34", "35-44", "55-64", "65+", "65+"]
    assert got == expected, f"expected {expected}, got {got}"


def test_subgroup_report_matches_scikit_learn_group_by_group():
    rng = np.random.default_rng(1)
    groups = rng.choice(["Female", "Male", None], 600, p=[0.3, 0.6, 0.1])
    y = (rng.random(600) < 0.3).astype(int)
    proba = np.clip(0.3 + 0.4 * (y - 0.3) + rng.normal(0, 0.25, 600), 0, 1)
    pred = (proba >= 0.4).astype(int)
    report = income.subgroup_report(y, pred, proba, groups)
    assert list(report.index) == ["Female", "Male", "missing"], f"expected the groups sorted, got {list(report.index)}"
    assert list(report.columns) == income.REPORT_COLUMNS, f"expected the columns {income.REPORT_COLUMNS}"
    labels = np.array(["missing" if g is None else g for g in groups])
    for group in report.index:
        mask = labels == group
        (tn, fp), (fn, tp) = confusion_matrix(y[mask], pred[mask], labels=[0, 1])
        expected = [mask.sum(), (tp + fn) / mask.sum(), (tp + fp) / mask.sum(), (tp + tn) / mask.sum(),
                    tp / (tp + fn), fp / (fp + tn), tp / (tp + fp), roc_auc_score(y[mask], proba[mask])]
        assert_close(report.loc[group].to_numpy(dtype=float), expected, f"the row {group!r}")


def test_subgroup_report_gives_nan_for_a_group_with_one_class():
    report = income.subgroup_report([0, 0, 1, 0], [0, 1, 1, 0], [0.1, 0.6, 0.9, 0.2], ["a", "a", "b", "b"])
    assert np.isnan(report.loc["a", "tpr"]) and np.isnan(report.loc["a", "roc_auc"]), (
        "expected NaN for the TPR and the ROC-AUC of a group without positives")
    assert report.loc["a", "fpr"] == 0.5, f"expected the FPR of group a = 0.5, got {report.loc['a', 'fpr']}"


# --- saving the model ------------------------------------------------------------------
def test_saved_model_predicts_the_same_and_keeps_the_versions(census, tmp_path):
    X, y = census
    pipeline = income.build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)
    path = income.save_model(pipeline, tmp_path / "model.joblib", {"threshold": 0.4})
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # same versions: no warning
        loaded, info = income.load_model(path)
    assert_close(loaded.predict_proba(X.iloc[:50]), pipeline.predict_proba(X.iloc[:50]), "the probabilities")
    assert info["metadata"] == {"threshold": 0.4}, f"expected the metadata back, got {info['metadata']}"
    assert info["versions"] == income.library_versions(), f"expected the versions {income.library_versions()}"


def test_save_model_refuses_an_unfitted_pipeline(tmp_path):
    with pytest.raises(NotFittedError):
        income.save_model(income.build_pipeline(DummyClassifier()), tmp_path / "model.joblib")


def test_load_model_warns_when_scikit_learn_differs(census, tmp_path):
    import joblib

    X, y = census
    pipeline = income.build_pipeline(DummyClassifier()).fit(X, y)
    joblib.dump({"pipeline": pipeline, "versions": {**income.library_versions(), "scikit-learn": "0.1"},
                 "metadata": {}}, tmp_path / "old.joblib")
    with pytest.warns(UserWarning, match="0.1"):
        income.load_model(tmp_path / "old.joblib")
    joblib.dump(pipeline, tmp_path / "bare.joblib")
    with pytest.raises(ValueError):
        income.load_model(tmp_path / "bare.joblib")
