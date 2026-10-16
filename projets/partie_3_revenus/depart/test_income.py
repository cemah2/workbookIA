"""Tests of my module income.py (mini-project of part III).

    python -m pytest mon_travail/projets/partie_3_revenus -q

Two tests are given as examples. Write at least four more (step MP3.8 of the notebook
checks that there are six or more), each with a clear name and a message whose first line
says what was expected; at least one on leakage and one on reproducibility. scikit-learn,
NumPy and pandas may serve as oracles: they check, the module builds. Ideas, from the
deliverables of the brief:
- make_test_indices is reproducible, keeps the share of positives (stratified), and
  refuses test_size = 0 or 1;
- the medians of the imputer and the means of the scaler are those of the rows received
  by fit only (no leakage), and a row is transformed alike alone or with others;
- cross_validate_all gives, fold by fold, the score of a pipeline fitted by hand on the
  training rows of the fold, and leaves the pipelines it receives unfitted;
- an unknown category or a missing value at prediction time does not break the pipeline;
- with encoding="ordinal", missing categories stay NaN and the numeric columns are unchanged;
- two fits with the same random_state give the same probabilities (reproducibility);
- f1_threshold returns the best F1 over all the thresholds, and refuses invalid inputs;
- age_groups puts 25 in "25-34" and 65 in "65+";
- subgroup_report gives, group by group, the rates of sklearn.metrics.confusion_matrix,
  and NaN for a group without positives;
- a model saved by save_model and reloaded by load_model predicts the same, and
  load_model warns when the scikit-learn version differs.
"""

import numpy as np
import pytest

import data
import income


@pytest.fixture(scope="module")
def census():
    """4 000 people of the training part (the test split of MP3.1 is left out): the tests stay fast."""
    X, y = data.load_census(source="csv")
    test = income.make_test_indices(y, test_size=0.2, seed=2026)
    rows = np.random.default_rng(7).choice(np.setdiff1d(np.arange(len(y)), test), 4000, replace=False)
    return X.iloc[rows], y[rows]


def test_test_indices_are_sorted_distinct_and_stratified():
    _, y = data.load_census(source="csv")
    test = income.make_test_indices(y, test_size=0.2, seed=2026)
    assert np.all(np.diff(test) > 0), "expected positions sorted in increasing order, without repetition"
    rest = np.setdiff1d(np.arange(len(y)), test)
    assert abs(y[test].mean() - y[rest].mean()) < 1e-3, (
        f"expected the same share of positives in the test and in the rest (stratified split), got "
        f"{y[test].mean():.4f} and {y[rest].mean():.4f}")


def test_imputer_learns_the_medians_of_the_training_rows_only(census):
    X, _ = census
    prep = income.build_preprocessor("onehot").fit(X.iloc[:3000])
    medians = prep.named_transformers_["numeric"].named_steps["impute"].statistics_
    expected = X.iloc[:3000][income.NUMERIC].median().to_numpy()
    assert np.allclose(medians, expected), (
        f"expected the medians of the 3 000 rows received by fit, {expected.tolist()}, got {list(medians)}")


# TODO MP3.8: your tests (at least four more)
