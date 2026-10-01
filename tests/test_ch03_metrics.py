"""Tests of mylearn.metrics (chapter 3): oracle tests and properties.

    pytest tests/test_ch03_metrics.py              # your code (mon_travail/mylearn/metrics.py)
    pytest tests/test_ch03_metrics.py --impl=ref   # the reference

Oracles: scikit-learn 1.6 (sklearn.metrics and sklearn.calibration), NumPy
(np.trapezoid) and direct counts written here with NumPy.
Every test name starts with the name of the function it tests (``-k "test_accuracy_"``).
For precision, recall, fbeta and f1, the tests of the two-class case contain
``binary`` in their name (exercise 3.16) and the tests of the averages over several
classes contain ``multiclass`` (exercise 3.25). The tests of
``precision_recall_curve`` contain ``curve``: exclude them with ``and not curve``.
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest
from sklearn import calibration as sk_calibration
from sklearn import metrics as skm


@pytest.fixture
def mt(mylearn_module):
    return mylearn_module("metrics")


@pytest.fixture(autouse=True)
def _quiet_sklearn():
    """scikit-learn warns about ill-defined scores (zero_division): not the learner's fault."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def random_labels(seed: int, n_classes: int = 2, n: int | None = None, kind: str = "int"):
    """Random true and predicted labels, the predictions right about 60 % of the time."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60)) if n is None else n
    y_true = rng.integers(0, n_classes, size=n)
    noise = rng.integers(0, n_classes, size=n)
    y_pred = np.where(rng.random(n) < 0.6, y_true, noise)
    if kind == "str":
        names = np.array(["cat", "dog", "fish", "bird", "lynx", "mole"][:n_classes])
        return names[y_true], names[y_pred]
    if kind == "pm1" and n_classes == 2:
        return 2 * y_true - 1, 2 * y_pred - 1
    return y_true, y_pred


def random_scores(seed: int, n: int | None = None, ties: bool = False, kind: str = "int"):
    """Random 0/1 truths and scores, the positives scoring higher on average."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 80)) if n is None else n
    y_true = rng.integers(0, 2, size=n)
    y_true[0], y_true[1] = 0, 1  # both classes present
    y_score = rng.normal(loc=y_true * 1.2, scale=1.0)
    if ties:
        y_score = np.round(y_score, 1)
    if kind == "str":
        return np.array(["ham", "spam"])[y_true], y_score
    if kind == "pm1":
        return 2 * y_true - 1, y_score
    return y_true, y_score


def listed(values) -> list:
    """Plain Python values for the messages (NumPy 2 would print np.int64(1))."""
    return np.asarray(values).tolist()


def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for the first line of a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    if arr.dtype.kind in "biuf":
        text = np.array2string(arr, precision=digits, separator=", ", threshold=12, max_line_width=10**6)
    else:
        text = repr(arr.tolist())
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str, details: str):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data and details."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    lines.append(details)
    raise AssertionError("\n".join(lines)) from None


def _shape_head(got, expected):
    """When two arrays have different shapes, say so first: truncated views of them can look identical."""
    want = np.asarray(expected)
    if got.ndim >= 1 and want.ndim >= 1 and got.shape != want.shape:
        return (f"expected {want.size} values (shape {want.shape}), got {got.size} (shape {got.shape}): "
                f"expected {_short(expected)}, got {_short(got)}")
    return None


def _first_difference(got, want, tol) -> str:
    """' (first difference at index i: expected a, got b)' for two arrays of the same shape, '' otherwise."""
    if got.ndim == 0 or got.shape != want.shape:
        return ""
    bad = np.flatnonzero(~np.isclose(got, want, rtol=tol, atol=tol, equal_nan=True))
    if bad.size == 0:
        return ""
    i = int(bad[0])
    index = np.unravel_index(i, got.shape)
    where = index[0] if got.ndim == 1 else tuple(int(k) for k in index)
    return f" (first difference at index {where}: expected {want.ravel()[i]:.6g}, got {got.ravel()[i]:.6g})"


def assert_close(result, expected, tol=1e-10, msg="", data=""):
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected {_short(expected)}, got an object of type {type(result).__name__}", data, "")
    head = _shape_head(got, expected)
    if head:
        _fail(msg, head, data, "")
    try:
        np.testing.assert_allclose(got, np.asarray(expected, dtype=float), rtol=tol, atol=tol)
    except AssertionError as exc:
        want, have = _short(expected), _short(result)
        if want == have:   # they differ beyond the 6th digit (float32, rounding): show every digit
            want, have = _short(expected, 17), _short(result, 17)
        where = _first_difference(got, np.asarray(expected, dtype=float), tol)
        _fail(msg, f"expected {want}, got {have}{where}", data, str(exc))


def assert_equal(result, expected, msg="", data=""):
    head = _shape_head(np.asarray(result), expected)
    if head:
        _fail(msg, head, data, "")
    try:
        np.testing.assert_array_equal(result, expected)
    except AssertionError as exc:
        _fail(msg, f"expected {_short(expected)}, got {_short(result)}", data, str(exc))


def assert_python_float(value, name):
    assert isinstance(value, float), f"{name} must return a Python float, got {type(value).__name__}"


def assert_raises_value_error(function, *args, why="", **kwargs):
    """The call must raise ValueError (or a subclass); `why` names the case in the failure message."""
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    name = getattr(function, "__name__", "the function")
    raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else ""))


# ------------------------------------------------------------------ confusion_matrix (3.15)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("kind", ["int", "str"])
def test_confusion_matrix_matches_sklearn(mt, n_classes, kind):
    for seed in range(15):
        y_true, y_pred = random_labels(seed, n_classes, kind=kind)
        result = mt.confusion_matrix(y_true, y_pred)
        expected = skm.confusion_matrix(y_true, y_pred)
        assert np.shape(result) == expected.shape, "one row and one column per label found in y_true and y_pred"
        assert_equal(result, expected, msg="C[i, j] counts the samples of true label i predicted as label j",
                     data=f"y_true={listed(y_true)}, y_pred={listed(y_pred)}")


def test_confusion_matrix_rows_are_the_truth(mt):
    # one sample of true class 0 predicted as 1: it must be counted in row 0, column 1
    result = mt.confusion_matrix([0, 0, 0, 1], [0, 0, 1, 1])
    assert_equal(result, [[2, 1], [0, 1]], msg="C[i, j] counts the samples of TRUE label i PREDICTED as label j")


def test_confusion_matrix_includes_labels_found_only_in_the_predictions(mt):
    # the label 2 is never true, but it is predicted once: it has its own (empty) row
    y_true, y_pred = [0, 0, 1, 1], [0, 2, 1, 1]
    assert_equal(mt.confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred),
                 msg="the labels are the sorted union of y_true AND y_pred")
    y_true, y_pred = ["b", "b", "c"], ["a", "b", "c"]
    assert_equal(mt.confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred),
                 msg="the labels are the sorted union of y_true AND y_pred")


def test_confusion_matrix_returns_integers_that_sum_to_n(mt):
    y_true, y_pred = random_labels(3, 4, n=50)
    result = np.asarray(mt.confusion_matrix(y_true, y_pred))
    assert np.issubdtype(result.dtype, np.integer), f"the counts must be integers, got dtype {result.dtype}"
    assert result.sum() == 50, f"the counts must sum to the number of samples (50), got {result.sum()}"


def test_confusion_matrix_follows_the_order_of_labels(mt):
    y_true = ["spam", "ham", "spam", "ham", "ham", "eggs"]
    y_pred = ["spam", "spam", "ham", "ham", "eggs", "eggs"]
    for labels in (["spam", "ham", "eggs"], ["eggs", "spam", "ham"], ["ham", "eggs", "spam", "other"]):
        assert_equal(mt.confusion_matrix(y_true, y_pred, labels=labels),
                     skm.confusion_matrix(y_true, y_pred, labels=labels),
                     msg=f"labels={labels}: rows and columns follow this order")


def test_confusion_matrix_accepts_lists_arrays_and_pandas(mt):
    pd = pytest.importorskip("pandas")
    expected = [[1, 1], [0, 2]]
    for y_true, y_pred in [([0, 0, 1, 1], [0, 1, 1, 1]), (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])),
                           (pd.Series([0, 0, 1, 1]), pd.Series([0, 1, 1, 1])),
                           (pd.Series([0, 0, 1, 1], index=[7, 3, 9, 1]), pd.Series([0, 1, 1, 1], index=[7, 3, 9, 1]))]:
        assert_equal(mt.confusion_matrix(y_true, y_pred), expected,
                     msg=f"inputs of type {type(y_true).__name__} (convert them with np.asarray)")


@pytest.mark.parametrize("y_true, y_pred, labels, why", [
    pytest.param([0, 1, 1], [0, 1], None, "different lengths", id="different-lengths"),
    pytest.param([0, 1, 1], [1], None, "different lengths (NumPy would broadcast [1])", id="length-1"),
    pytest.param([], [], None, "empty inputs", id="empty"),
    pytest.param([0, 1, 2], [0, 1, 1], [0, 1], "the label 2 of y_true is missing from labels", id="true-label-missing"),
    pytest.param([0, 1, 1], [0, 1, 2], [0, 1], "the label 2 of y_pred is missing from labels", id="pred-label-missing"),
])
def test_confusion_matrix_rejects_bad_inputs(mt, y_true, y_pred, labels, why):
    assert_raises_value_error(mt.confusion_matrix, y_true, y_pred, labels=labels, why=why)


# ------------------------------------------------------------------ accuracy (3.16)
@pytest.mark.parametrize("n_classes", [2, 4])
def test_accuracy_matches_sklearn(mt, n_classes):
    for seed in range(20):
        y_true, y_pred = random_labels(seed, n_classes, kind="str" if seed % 2 else "int")
        result = mt.accuracy(y_true, y_pred)
        assert_python_float(result, "accuracy")
        assert_close(result, skm.accuracy_score(y_true, y_pred), msg="share of predictions equal to the truth",
                     data=f"y_true={listed(y_true)}, y_pred={listed(y_pred)}")


@pytest.mark.parametrize("y_true, y_pred, why", [
    pytest.param([0, 1, 1], [0, 1], "different lengths", id="different-lengths"),
    pytest.param([0, 1, 1], [1], "different lengths (NumPy would broadcast [1])", id="length-1"),
    pytest.param([], [], "empty inputs", id="empty"),
])
def test_accuracy_rejects_bad_inputs(mt, y_true, y_pred, why):
    assert_raises_value_error(mt.accuracy, y_true, y_pred, why=why)


# ------------------------------------------------------------------ precision, recall, fbeta, f1: binary (3.16)
BINARY_CASES = [
    ("int", {}),                          # labels 0/1, positive class 1 (the default)
    ("int", {"pos_label": 0}),            # the other class as positive
    ("pm1", {"pos_label": -1}),           # labels -1/+1
    ("str", {"pos_label": "dog"}),        # text labels
]


def _binary_data(seed, kind):
    return random_labels(seed, 2, kind=kind)


@pytest.mark.parametrize("kind, kwargs", BINARY_CASES)
def test_precision_binary_matches_sklearn(mt, kind, kwargs):
    for seed in range(20):
        y_true, y_pred = _binary_data(seed, kind)
        result = mt.precision(y_true, y_pred, **kwargs)
        assert_python_float(result, "precision")
        assert_close(result, skm.precision_score(y_true, y_pred, zero_division=0.0, **kwargs),
                     msg=f"precision = TP / (TP + FP) {kwargs}", data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


@pytest.mark.parametrize("kind, kwargs", BINARY_CASES)
def test_recall_binary_matches_sklearn(mt, kind, kwargs):
    for seed in range(20):
        y_true, y_pred = _binary_data(seed, kind)
        result = mt.recall(y_true, y_pred, **kwargs)
        assert_python_float(result, "recall")
        assert_close(result, skm.recall_score(y_true, y_pred, zero_division=0.0, **kwargs),
                     msg=f"recall = TP / (TP + FN) {kwargs}", data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("kind, kwargs", BINARY_CASES)
def test_fbeta_binary_matches_sklearn(mt, kind, kwargs, beta):
    for seed in range(15):
        y_true, y_pred = _binary_data(seed, kind)
        result = mt.fbeta(y_true, y_pred, beta=beta, **kwargs)
        assert_python_float(result, "fbeta")
        assert_close(result, skm.fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0, **kwargs),
                     msg=f"beta={beta} {kwargs}: F = (1 + beta²) TP / ((1 + beta²) TP + beta² FN + FP)",
                     data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


@pytest.mark.parametrize("kind, kwargs", BINARY_CASES)
def test_f1_binary_matches_sklearn(mt, kind, kwargs):
    for seed in range(20):
        y_true, y_pred = _binary_data(seed, kind)
        result = mt.f1(y_true, y_pred, **kwargs)
        assert_python_float(result, "f1")
        assert_close(result, skm.f1_score(y_true, y_pred, zero_division=0.0, **kwargs),
                     msg=f"F1 = 2 TP / (2 TP + FP + FN) {kwargs}", data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


def test_f1_binary_is_the_harmonic_mean_of_precision_and_recall(mt):
    for seed in range(20):
        y_true, y_pred = _binary_data(seed, "int")
        p, r = skm.precision_score(y_true, y_pred), skm.recall_score(y_true, y_pred)
        expected = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        assert_close(mt.f1(y_true, y_pred), expected, msg="F1 is the harmonic mean 2PR / (P + R)",
                     data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


# degenerate cases: a ratio 0/0 takes the value zero_division
ZERO_DIVISION_CASES = [
    pytest.param([1, 1, 0, 0], [0, 0, 0, 0], id="no-positive-prediction"),    # precision 0/0
    pytest.param([0, 0, 0, 0], [0, 1, 0, 0], id="no-actual-positive"),        # recall 0/0
    pytest.param([0, 0, 0, 0], [0, 0, 0, 0], id="nothing-positive"),          # precision, recall and F 0/0
    pytest.param([1, 1, 0, 0], [0, 0, 1, 1], id="tp-0-but-errors"),           # P = R = 0 (defined), F = 0 / 4
]


@pytest.mark.parametrize("zero_division", [0.0, 1.0])
@pytest.mark.parametrize("y_true, y_pred", ZERO_DIVISION_CASES)
def test_precision_binary_zero_division(mt, y_true, y_pred, zero_division):
    assert_close(mt.precision(y_true, y_pred, zero_division=zero_division),
                 skm.precision_score(y_true, y_pred, zero_division=zero_division),
                 msg=f"precision = zero_division ({zero_division}) only when TP + FP = 0",
                 data=f"y_true={y_true} y_pred={y_pred}")


@pytest.mark.parametrize("zero_division", [0.0, 1.0])
@pytest.mark.parametrize("y_true, y_pred", ZERO_DIVISION_CASES)
def test_recall_binary_zero_division(mt, y_true, y_pred, zero_division):
    assert_close(mt.recall(y_true, y_pred, zero_division=zero_division),
                 skm.recall_score(y_true, y_pred, zero_division=zero_division),
                 msg=f"recall = zero_division ({zero_division}) only when TP + FN = 0",
                 data=f"y_true={y_true} y_pred={y_pred}")


@pytest.mark.parametrize("zero_division", [0.0, 1.0])
@pytest.mark.parametrize("y_true, y_pred", ZERO_DIVISION_CASES)
def test_fbeta_binary_zero_division(mt, y_true, y_pred, zero_division):
    for beta in (0.5, 1.0, 2.0):
        assert_close(mt.fbeta(y_true, y_pred, beta=beta, zero_division=zero_division),
                     skm.fbeta_score(y_true, y_pred, beta=beta, zero_division=zero_division),
                     msg=f"beta={beta}: F = zero_division ({zero_division}) only when TP + FN + FP = 0; "
                         "if TP = 0 but there are errors, F = 0",
                     data=f"y_true={y_true} y_pred={y_pred}")


@pytest.mark.parametrize("zero_division", [0.0, 1.0])
@pytest.mark.parametrize("y_true, y_pred", ZERO_DIVISION_CASES)
def test_f1_binary_zero_division(mt, y_true, y_pred, zero_division):
    assert_close(mt.f1(y_true, y_pred, zero_division=zero_division),
                 skm.f1_score(y_true, y_pred, zero_division=zero_division),
                 msg=f"F1 = zero_division ({zero_division}) only when TP + FN + FP = 0; "
                     "if TP = 0 but there are errors, F1 = 0",
                 data=f"y_true={y_true} y_pred={y_pred}")


@pytest.mark.parametrize("name", ["precision", "recall", "fbeta", "f1"])
def test_precision_recall_fbeta_f1_binary_reject_more_than_two_labels(mt, name):
    assert_raises_value_error(getattr(mt, name), [0, 1, 2, 2], [0, 1, 2, 1],
                              why="three labels with average='binary'")
    assert_raises_value_error(getattr(mt, name), [0, 1, 0, 1], [0, 2, 0, 1],
                              why="the labels present are those of y_true AND y_pred: here 0, 1 and 2")


@pytest.mark.parametrize("name", ["precision", "recall", "fbeta", "f1"])
def test_precision_recall_fbeta_f1_binary_reject_an_absent_pos_label(mt, name):
    assert_raises_value_error(getattr(mt, name), [0, 2, 2, 0], [0, 2, 0, 0],
                              why="pos_label=1 (the default) is not one of the two labels 0 and 2")
    assert_raises_value_error(getattr(mt, name), ["a", "b"], ["a", "a"], pos_label="c",
                              why="pos_label='c' is not one of the two labels 'a' and 'b'")


@pytest.mark.parametrize("name", ["precision", "recall", "fbeta", "f1", "accuracy"])
def test_precision_recall_fbeta_f1_accuracy_binary_reject_different_lengths(mt, name):
    assert_raises_value_error(getattr(mt, name), [0, 1, 1], [0, 1], why="different lengths")
    assert_raises_value_error(getattr(mt, name), [0, 1, 1], [1],
                              why="different lengths (NumPy would silently broadcast [1])")


@pytest.mark.parametrize("beta", [0, -1.0])
def test_fbeta_binary_rejects_a_non_positive_beta(mt, beta):
    assert_raises_value_error(mt.fbeta, [0, 1, 1], [0, 1, 0], beta=beta, why=f"beta={beta} is not > 0")


# ------------------------------------------------------------------ averages over the classes (3.25)
AVERAGES = [None, "macro", "micro", "weighted"]


def _multiclass_data(seed):
    n_classes = 2 + seed % 4
    kind = "str" if seed % 3 == 0 else "int"
    return random_labels(seed, n_classes, kind=kind)


def _check_average(result, expected, average, name, data=""):
    if average is None:
        assert isinstance(result, np.ndarray), f"{name}(average=None) must return a NumPy array"
        assert result.shape == np.shape(expected), \
            "one value per label found in y_true OR y_pred, in the sorted order of the labels"
    else:
        assert_python_float(result, f"{name}(average={average!r})")
    assert_close(result, expected, msg=f"{name}(average={average!r})", data=data)


@pytest.mark.parametrize("average", AVERAGES)
def test_precision_multiclass_matches_sklearn(mt, average):
    for seed in range(24):
        y_true, y_pred = _multiclass_data(seed)
        _check_average(mt.precision(y_true, y_pred, average=average),
                       skm.precision_score(y_true, y_pred, average=average, zero_division=0.0), average, "precision",
                       data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


@pytest.mark.parametrize("average", AVERAGES)
def test_recall_multiclass_matches_sklearn(mt, average):
    for seed in range(24):
        y_true, y_pred = _multiclass_data(seed)
        _check_average(mt.recall(y_true, y_pred, average=average),
                       skm.recall_score(y_true, y_pred, average=average, zero_division=0.0), average, "recall",
                       data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


@pytest.mark.parametrize("beta", [0.5, 2.0])
@pytest.mark.parametrize("average", AVERAGES)
def test_fbeta_multiclass_matches_sklearn(mt, average, beta):
    for seed in range(24):
        y_true, y_pred = _multiclass_data(seed)
        _check_average(mt.fbeta(y_true, y_pred, beta=beta, average=average),
                       skm.fbeta_score(y_true, y_pred, beta=beta, average=average, zero_division=0.0),
                       average, f"fbeta(beta={beta})", data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


@pytest.mark.parametrize("average", AVERAGES)
def test_f1_multiclass_matches_sklearn(mt, average):
    for seed in range(24):
        y_true, y_pred = _multiclass_data(seed)
        _check_average(mt.f1(y_true, y_pred, average=average),
                       skm.f1_score(y_true, y_pred, average=average, zero_division=0.0), average, "f1",
                       data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")


@pytest.mark.parametrize("zero_division", [0.0, 1.0])
@pytest.mark.parametrize("average", AVERAGES)
def test_precision_recall_f1_multiclass_zero_division(mt, average, zero_division):
    # class 1 is never predicted (its precision is 0/0) and class 3 never appears in y_true (its recall is 0/0)
    y_true, y_pred = [0, 1, 2, 2, 0, 1], [0, 0, 2, 2, 3, 2]
    for name, oracle in [("precision", skm.precision_score), ("recall", skm.recall_score), ("f1", skm.f1_score)]:
        _check_average(getattr(mt, name)(y_true, y_pred, average=average, zero_division=zero_division),
                       oracle(y_true, y_pred, average=average, zero_division=zero_division), average, name,
                       data=f"zero_division={zero_division}: class 1 is never predicted, class 3 is never true")


def test_precision_recall_f1_multiclass_micro_equals_accuracy(mt):
    for seed in range(10):
        y_true, y_pred = random_labels(seed, 4)
        expected = skm.accuracy_score(y_true, y_pred)
        for name in ("precision", "recall", "f1"):
            assert_close(getattr(mt, name)(y_true, y_pred, average="micro"), expected,
                         msg=f"{name}(average='micro') pools the counts of every class: it equals the accuracy")


@pytest.mark.parametrize("name", ["precision", "recall", "fbeta", "f1"])
@pytest.mark.parametrize("average", ["mean", "samples", "Macro"])
def test_precision_recall_fbeta_f1_multiclass_reject_an_unknown_average(mt, name, average):
    assert_raises_value_error(getattr(mt, name), [0, 1, 2], [0, 2, 1], average=average,
                              why=f"average={average!r} is unknown")
    assert_raises_value_error(getattr(mt, name), [0, 1, 1], [0, 1, 0], average=average,
                              why=f"average={average!r} is unknown, even with two labels")


# ------------------------------------------------------------------ classification_rates (3.19)
RATE_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "npv", "fpr", "fnr",
             "fdr", "false_omission_rate", "prevalence", "f1", "mcc"]


def _rates_oracle(y_true, y_pred, pos_label):
    """Every rate with scikit-learn (and the identities fpr = 1 - specificity, ...)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    t, p = y_true == pos_label, y_pred == pos_label   # booleans: True = positive class
    specificity = skm.recall_score(t, p, pos_label=False)
    npv = skm.precision_score(t, p, pos_label=False)
    precision = skm.precision_score(t, p)
    recall = skm.recall_score(t, p)
    return {
        "accuracy": skm.accuracy_score(t, p),
        "balanced_accuracy": skm.balanced_accuracy_score(t, p),
        "precision": precision, "recall": recall, "specificity": specificity, "npv": npv,
        "fpr": 1 - specificity, "fnr": 1 - recall, "fdr": 1 - precision, "false_omission_rate": 1 - npv,
        "prevalence": float(np.mean(t)), "f1": skm.f1_score(t, p), "mcc": skm.matthews_corrcoef(t, p),
    }


def _nondegenerate(y_true, y_pred, pos_label):
    """Both classes in y_true and in y_pred: every rate is defined."""
    t, p = np.asarray(y_true) == pos_label, np.asarray(y_pred) == pos_label
    return 0 < t.sum() < len(t) and 0 < p.sum() < len(p)


@pytest.mark.parametrize("kind, pos_label", [("int", 1), ("int", 0), ("pm1", -1), ("str", "cat")])
def test_classification_rates_match_sklearn(mt, kind, pos_label):
    checked = 0
    for seed in range(40):
        y_true, y_pred = random_labels(seed, 2, kind=kind)
        if not _nondegenerate(y_true, y_pred, pos_label):
            continue
        result = mt.classification_rates(y_true, y_pred, pos_label=pos_label)
        expected = _rates_oracle(y_true, y_pred, pos_label)
        for key in RATE_KEYS:
            assert key in result, f"key {key!r} missing"
            assert_close(result[key], expected[key], msg=f"{key} (pos_label={pos_label!r})",
                         data=f"y_true={listed(y_true)} y_pred={listed(y_pred)}")
        checked += 1
    assert checked >= 20


def test_classification_rates_keys_come_in_the_documented_order(mt):
    result = mt.classification_rates([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0])
    assert list(result) == RATE_KEYS, f"the 13 keys, in the order of the docstring: {RATE_KEYS}"


def test_classification_rates_values_are_python_floats(mt):
    result = mt.classification_rates([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0])
    for key, value in result.items():
        assert_python_float(value, f"classification_rates(...)[{key!r}]")


@pytest.mark.parametrize("zero_division", [0.0, 1.0])
def test_classification_rates_zero_division(mt, zero_division):
    # The docstring's rule, "zero_division for any 0/0 ratio", is used everywhere, even where scikit-learn
    # has its own convention (balanced_accuracy_score drops a missing class, matthews_corrcoef returns 0).
    # no positive prediction: precision, fdr and mcc are 0/0; all the other rates are defined
    result = mt.classification_rates([1, 1, 0, 0, 0], [0, 0, 0, 0, 0], zero_division=zero_division)
    for key in ("precision", "fdr", "mcc"):
        assert_close(result[key], zero_division, msg=f"{key} is 0/0 here: it takes the value zero_division")
    assert_close(result["recall"], 0.0, msg="recall = TP / (TP + FN) = 0 / 2")
    assert_close(result["specificity"], 1.0, msg="specificity = TN / (TN + FP) = 3 / 3")
    assert_close(result["npv"], 3 / 5, msg="npv = TN / (TN + FN) = 3 / 5")
    assert_close(result["f1"], 0.0, msg="F1 = 2 TP / (2 TP + FP + FN) = 0 / 2: defined")
    assert_close(result["balanced_accuracy"], 0.5, msg="balanced_accuracy = (recall + specificity) / 2 = (0 + 1) / 2")
    # no actual positive: recall, fnr are 0/0, and balanced accuracy uses zero_division for the recall
    result = mt.classification_rates([0, 0, 0, 0], [0, 1, 0, 0], zero_division=zero_division)
    for key in ("recall", "fnr", "mcc"):
        assert_close(result[key], zero_division, msg=f"{key} is 0/0 here: it takes the value zero_division")
    assert_close(result["balanced_accuracy"], (zero_division + 0.75) / 2,
                 msg="balanced_accuracy = (recall + specificity) / 2, with recall = zero_division here "
                     "(scikit-learn's balanced_accuracy_score would drop the missing class instead)")
    assert_close(result["prevalence"], 0.0, msg="prevalence = (TP + FN) / n = 0 / 4")


@pytest.mark.parametrize("y_true, y_pred, pos_label, why", [
    pytest.param([0, 1, 2], [0, 1, 1], 1, "three labels", id="three-labels"),
    pytest.param([0, 1, 0, 1], [0, 2, 0, 1], 1, "three labels in y_true and y_pred together", id="third-label-in-pred"),
    pytest.param([0, 2, 2], [0, 2, 0], 1, "pos_label=1 is not one of the two labels", id="absent-pos-label"),
    pytest.param([0, 1, 1], [0, 1], 1, "different lengths", id="different-lengths"),
    pytest.param([0, 1, 1], [1], 1, "different lengths (NumPy would broadcast [1])", id="length-1"),
    pytest.param([], [], 1, "empty inputs", id="empty"),
])
def test_classification_rates_reject_bad_inputs(mt, y_true, y_pred, pos_label, why):
    assert_raises_value_error(mt.classification_rates, y_true, y_pred, pos_label=pos_label, why=why)


# ------------------------------------------------------------------ roc_curve, auc, roc_auc (3.24)
SCORE_CASES = [("int", {}), ("pm1", {"pos_label": -1}), ("str", {"pos_label": "spam"}), ("int", {"pos_label": 0})]


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("kind, kwargs", SCORE_CASES)
def test_roc_curve_matches_sklearn(mt, kind, kwargs, ties):
    for seed in range(15):
        y_true, y_score = random_scores(seed, ties=ties, kind=kind)
        fpr, tpr, thresholds = mt.roc_curve(y_true, y_score, **kwargs)
        fpr_sk, tpr_sk, thr_sk = skm.roc_curve(y_true, y_score, drop_intermediate=False, **kwargs)
        data = f"{kwargs} y_true={listed(y_true)} y_score={listed(y_score)}"
        assert np.shape(thresholds) == thr_sk.shape, "one point per distinct score, plus the starting point at +inf"
        assert_close(thresholds[1:], thr_sk[1:], msg="thresholds: the distinct scores, decreasing", data=data)
        assert np.isposinf(thresholds[0]), "thresholds[0] must be np.inf"
        assert_close(fpr, fpr_sk, msg="false positive rates FP / (FP + TN), predicting positive when score >= t",
                     data=data)
        assert_close(tpr, tpr_sk, msg="true positive rates TP / (TP + FN), predicting positive when score >= t",
                     data=data)


def test_roc_curve_goes_from_0_0_to_1_1(mt):
    fpr, tpr, _ = mt.roc_curve([0, 0, 1, 1, 0, 1], [0.2, 0.6, 0.7, 0.3, 0.1, 0.9])
    assert (fpr[0], tpr[0]) == (0.0, 0.0), f"the curve starts at (0, 0), got ({fpr[0]}, {tpr[0]})"
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0), f"the curve ends at (1, 1), got ({fpr[-1]}, {tpr[-1]})"
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0), "fpr and tpr never decrease"


def test_roc_curve_accepts_pandas_series_with_any_index(mt):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 40)
    y[:2] = [0, 1]
    s = rng.normal(y * 1.0, 1.0)                             # no ties: this test is about the index only
    index = rng.permutation(40)                              # as after train_test_split or DataFrame.sample
    fpr, tpr, thresholds = mt.roc_curve(pd.Series(y, index=index), pd.Series(s, index=index))
    fpr_sk, tpr_sk, thresholds_sk = skm.roc_curve(y, s, drop_intermediate=False)
    assert_close(tpr, tpr_sk, msg="Series with a shuffled index: convert the inputs with np.asarray first")
    assert_close(fpr, fpr_sk, msg="Series with a shuffled index: convert the inputs with np.asarray first")
    assert_close(np.asarray(thresholds)[1:], thresholds_sk[1:],
                 msg="Series with a shuffled index: convert BOTH inputs with np.asarray first (thresholds)")


@pytest.mark.parametrize("y_true, y_score, kwargs, why", [
    pytest.param([1, 1, 1], [0.2, 0.5, 0.9], {}, "a single class", id="single-class"),
    pytest.param([0, 1, 2, 1], [0.1, 0.5, 0.9, 0.3], {}, "three classes", id="three-classes"),
    pytest.param([0, 1, 1], [0.2, 0.5], {}, "different lengths", id="different-lengths"),
    pytest.param([0, 1, 1], [0.5], {}, "different lengths (NumPy would broadcast [0.5])", id="length-1"),
    pytest.param([0, 1, 1], [0.2, np.nan, 0.9], {}, "a NaN score", id="nan-score"),
    pytest.param([0, 2, 2], [0.2, 0.5, 0.9], {}, "pos_label=1 is not one of the classes", id="absent-pos-label"),
    pytest.param(["a", "b", "b"], [0.2, 0.5, 0.9], {"pos_label": "c"}, "pos_label='c' is not one of the classes",
                 id="absent-text-pos-label"),
])
def test_roc_curve_rejects_bad_inputs(mt, y_true, y_score, kwargs, why):
    assert_raises_value_error(mt.roc_curve, y_true, y_score, why=why, **kwargs)


def _monotonic_points(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 30))
    x = np.sort(np.round(rng.random(n), 1))       # repeated x values happen, as in a ROC curve
    y = rng.random(n)
    return x, y


def test_auc_matches_sklearn(mt):
    for seed in range(30):
        x, y = _monotonic_points(seed)
        result = mt.auc(x, y)
        assert_python_float(result, "auc")
        assert_close(result, skm.auc(x, y), msg="increasing x: sum of the trapezoids", data=f"x={listed(x)}")
        assert_close(mt.auc(x[::-1], y[::-1]), skm.auc(x[::-1], y[::-1]),
                     msg="decreasing x: the area does not depend on the direction", data=f"x={listed(x[::-1])}")


def test_auc_matches_numpy_trapezoid(mt):
    x = np.array([0.0, 0.1, 0.1, 0.4, 1.0])
    y = np.array([0.0, 0.5, 0.6, 0.9, 1.0])
    assert_close(mt.auc(x, y), np.trapezoid(y, x), msg="trapezoids (x_{i+1} - x_i) (y_i + y_{i+1}) / 2")


@pytest.mark.parametrize("x, y, why", [
    pytest.param([0.5], [1.0], "a single point", id="single-point"),
    pytest.param([0, 1, 2], [0, 1], "different lengths", id="different-lengths"),
    pytest.param([0, 1, 0.5], [0, 1, 1], "x neither non-decreasing nor non-increasing", id="x-not-monotonic"),
])
def test_auc_rejects_bad_inputs(mt, x, y, why):
    assert_raises_value_error(mt.auc, x, y, why=why)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("kind, kwargs", SCORE_CASES)
def test_roc_auc_matches_sklearn(mt, kind, kwargs, ties):
    for seed in range(15):
        y_true, y_score = random_scores(seed, ties=ties, kind=kind)
        pos_label = kwargs.get("pos_label", 1)
        result = mt.roc_auc(y_true, y_score, **kwargs)
        assert_python_float(result, "roc_auc")
        assert_close(result, skm.roc_auc_score(np.asarray(y_true) == pos_label, y_score),
                     msg=f"area under the ROC curve {kwargs}", data=f"y_true={listed(y_true)} y_score={listed(y_score)}")


def test_roc_auc_is_the_probability_that_a_positive_outscores_a_negative(mt):
    for seed in range(20):
        y_true, y_score = random_scores(seed, n=15, ties=True)
        pos, neg = y_score[y_true == 1], y_score[y_true == 0]
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        assert_close(mt.roc_auc(y_true, y_score), wins / (len(pos) * len(neg)),
                     msg="count the (positive, negative) pairs; a tie counts 1/2",
                     data=f"y_true={listed(y_true)} y_score={listed(y_score)}")


def test_roc_auc_rejects_a_single_class(mt):
    assert_raises_value_error(mt.roc_auc, [0, 0, 0], [0.1, 0.5, 0.9], why="a single class")


def test_roc_auc_rejects_three_classes_or_an_absent_pos_label(mt):
    assert_raises_value_error(mt.roc_auc, [0, 1, 2, 1], [0.1, 0.5, 0.9, 0.3], why="three classes")
    assert_raises_value_error(mt.roc_auc, [0, 2, 2], [0.2, 0.5, 0.9], why="pos_label=1 is not one of the classes")


# ------------------------------------------------------------------ precision_recall_curve, average_precision (3.26)
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("kind, kwargs", SCORE_CASES)
def test_precision_recall_curve_matches_sklearn(mt, kind, kwargs, ties):
    for seed in range(15):
        y_true, y_score = random_scores(seed, ties=ties, kind=kind)
        precision, recall, thresholds = mt.precision_recall_curve(y_true, y_score, **kwargs)
        p_sk, r_sk, t_sk = skm.precision_recall_curve(y_true, y_score, drop_intermediate=False, **kwargs)
        data = f"{kwargs} y_true={listed(y_true)} y_score={listed(y_score)}"
        assert np.shape(thresholds) == t_sk.shape, "one threshold per distinct score"
        assert_close(thresholds, t_sk, msg="thresholds: the distinct scores, increasing", data=data)
        assert np.shape(precision) == p_sk.shape, "precision and recall have one more value: the closing point"
        assert_close(precision, p_sk, msg="precision at each threshold (positive when score >= t), then 1",
                     data=data)
        assert_close(recall, r_sk, msg="recall at each threshold (positive when score >= t), then 0", data=data)


def test_precision_recall_curve_ends_with_precision_1_and_recall_0(mt):
    precision, recall, _ = mt.precision_recall_curve([0, 1, 1, 0, 1], [0.1, 0.8, 0.4, 0.5, 0.9])
    assert (precision[-1], recall[-1]) == (1.0, 0.0), \
        f"the closing point is (precision 1, recall 0), got ({precision[-1]}, {recall[-1]})"
    assert np.all(np.diff(recall) <= 0), "recall never increases along increasing thresholds"


def test_precision_recall_curve_accepts_pandas_series_with_any_index(mt):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 40)
    y[:2] = [0, 1]
    s = rng.normal(y * 1.0, 1.0)                             # no ties: this test is about the index only
    index = rng.permutation(40)
    p, r, thresholds = mt.precision_recall_curve(pd.Series(y, index=index), pd.Series(s, index=index))
    p_sk, r_sk, thresholds_sk = skm.precision_recall_curve(y, s, drop_intermediate=False)
    assert_close(p, p_sk, msg="Series with a shuffled index: convert the inputs with np.asarray first")
    assert_close(r, r_sk, msg="Series with a shuffled index: convert the inputs with np.asarray first")
    assert_close(thresholds, thresholds_sk,
                 msg="Series with a shuffled index: convert BOTH inputs with np.asarray first (thresholds)")


@pytest.mark.parametrize("y_true, y_score, why", [
    pytest.param([1, 1], [0.2, 0.5], "a single class", id="single-class"),
    pytest.param([0, 1, 2, 1], [0.1, 0.5, 0.9, 0.3], "three classes", id="three-classes"),
    pytest.param([0, 2, 2], [0.2, 0.5, 0.9], "pos_label=1 is not one of the classes", id="absent-pos-label"),
    pytest.param([0, 1, 1], [0.2, 0.5], "different lengths", id="different-lengths"),
    pytest.param([0, 1, 1], [0.5], "different lengths (NumPy would broadcast [0.5])", id="length-1"),
])
def test_precision_recall_curve_rejects_bad_inputs(mt, y_true, y_score, why):
    assert_raises_value_error(mt.precision_recall_curve, y_true, y_score, why=why)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("kind, kwargs", SCORE_CASES)
def test_average_precision_matches_sklearn(mt, kind, kwargs, ties):
    for seed in range(15):
        y_true, y_score = random_scores(seed, ties=ties, kind=kind)
        result = mt.average_precision(y_true, y_score, **kwargs)
        assert_python_float(result, "average_precision")
        assert_close(result, skm.average_precision_score(y_true, y_score, **kwargs),
                     msg=f"AP = sum of (R_n - R_(n-1)) P_n over the thresholds, from the highest {kwargs}",
                     data=f"y_true={listed(y_true)} y_score={listed(y_score)}")


def test_average_precision_uses_steps_not_trapezoids(mt):
    y_true, y_score = [1, 0, 1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    precision, recall, _ = skm.precision_recall_curve(y_true, y_score)
    trapezoids = skm.auc(recall, precision)
    result = mt.average_precision(y_true, y_score)
    assert not np.isclose(result, trapezoids), "AP is a sum of steps (R_n - R_(n-1)) P_n, not the trapezoidal area"
    assert_close(result, skm.average_precision_score(y_true, y_score), msg="AP as a sum of steps")


def test_average_precision_rejects_a_single_class(mt):
    assert_raises_value_error(mt.average_precision, [1, 1, 1], [0.1, 0.5, 0.9], why="a single class")


def test_average_precision_rejects_three_classes_or_an_absent_pos_label(mt):
    assert_raises_value_error(mt.average_precision, [0, 1, 2, 1], [0.1, 0.5, 0.9, 0.3], why="three classes")
    assert_raises_value_error(mt.average_precision, [0, 2, 2], [0.2, 0.5, 0.9],
                              why="pos_label=1 is not one of the classes")


# ------------------------------------------------------------------ calibration_curve, brier_score (3.28)
def _probabilities(seed, n_bins):
    """Outcomes and probabilities, some of them exactly on the bin edges.

    Only the edges that equal k / n_bins exactly are used: np.linspace(0, 1, 11)[3] is
    0.30000000000000004, and a value there would test floating-point noise, not the edge rule.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    p = rng.random(n)
    edges = np.linspace(0, 1, n_bins + 1)                    # the documented edges
    exact = edges[edges == np.arange(n_bins + 1) / n_bins]
    on_edge = rng.random(n) < 0.3
    p[on_edge] = rng.choice(exact, size=int(on_edge.sum()))
    y = (rng.random(n) < p).astype(int)
    return y, p


@pytest.mark.parametrize("n_bins", [1, 2, 5, 10, 13])
def test_calibration_curve_matches_sklearn(mt, n_bins):
    for seed in range(20):
        y, p = _probabilities(seed, n_bins)
        prob_true, prob_pred = mt.calibration_curve(y, p, n_bins=n_bins)
        pt_sk, pp_sk = sk_calibration.calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
        data = f"y_true={listed(y)} y_prob={listed(p)}"
        assert np.shape(prob_true) == pt_sk.shape, "one value per NON-EMPTY bin"
        assert_close(prob_true, pt_sk, msg=f"share of positives per bin (n_bins={n_bins}; a value on an inner "
                                           "edge goes to the lower bin)", data=data)
        assert_close(prob_pred, pp_sk, msg=f"mean predicted probability per bin (n_bins={n_bins})", data=data)


def test_calibration_curve_puts_an_inner_edge_in_the_lower_bin(mt):
    # 0.4 is exactly np.linspace(0, 1, 11)[4] (whereas np.linspace(0, 1, 11)[3] is 0.30000000000000004)
    prob_true, prob_pred = mt.calibration_curve([1, 0, 0], [0.4, 0.41, 0.49], n_bins=10)
    assert np.shape(prob_true) == (2,), "the value 0.4, on an inner edge, goes to the bin (0.3, 0.4], not (0.4, 0.5]"
    assert_close(prob_true, [1.0, 0.0], msg="bins (0.3, 0.4] then (0.4, 0.5]")


@pytest.mark.parametrize("y_true, y_prob, n_bins, why", [
    pytest.param([0, 2, 1], [0.1, 0.5, 0.9], 10, "an outcome that is neither 0 nor 1", id="outcome-2"),
    pytest.param([0, 1, 1], [0.1, 1.5, 0.9], 10, "a probability above 1", id="probability-above-1"),
    pytest.param([0, 1, 1], [-0.1, 0.5, 0.9], 10, "a probability below 0", id="probability-below-0"),
    pytest.param([0, 1, 1], [0.1, 0.5], 10, "different lengths", id="different-lengths"),
    pytest.param([0, 1, 1], [0.5], 10, "different lengths (NumPy would broadcast [0.5])", id="length-1"),
    pytest.param([0, 1, 1], [0.1, 0.5, 0.9], 0, "n_bins < 1", id="no-bin"),
])
def test_calibration_curve_rejects_bad_inputs(mt, y_true, y_prob, n_bins, why):
    assert_raises_value_error(mt.calibration_curve, y_true, y_prob, n_bins=n_bins, why=why)


def test_brier_score_matches_sklearn(mt):
    for seed in range(20):
        y, p = _probabilities(seed, 10)
        result = mt.brier_score(y, p)
        assert_python_float(result, "brier_score")
        assert_close(result, skm.brier_score_loss(y, p), msg="mean of (y_prob - y_true)²",
                     data=f"y_true={listed(y)} y_prob={listed(p)}")


def test_brier_score_of_a_constant_one_half_is_one_quarter(mt):
    assert_close(mt.brier_score([0, 1, 1, 0, 1], [0.5] * 5), 0.25, msg="always 0.5: every squared gap is 0.25")


@pytest.mark.parametrize("y_true, y_prob, why", [
    pytest.param([0, 2, 1], [0.1, 0.5, 0.9], "an outcome that is neither 0 nor 1", id="outcome-2"),
    pytest.param([0, 1, 1], [0.1, 1.5, 0.9], "a probability above 1", id="probability-above-1"),
    pytest.param([0, 1, 1], [-0.1, 0.5, 0.9], "a probability below 0", id="probability-below-0"),
    pytest.param([0, 1, 1], [0.1, 0.5], "different lengths", id="different-lengths"),
    pytest.param([0, 1, 1], [0.5], "different lengths (NumPy would broadcast [0.5])", id="length-1"),
])
def test_brier_score_rejects_bad_inputs(mt, y_true, y_prob, why):
    assert_raises_value_error(mt.brier_score, y_true, y_prob, why=why)
