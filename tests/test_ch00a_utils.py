"""Tests of mylearn.utils (chapter 0A): oracle tests and properties.

    pytest tests/test_ch00a_utils.py              # your code (mon_travail/mylearn/utils.py)
    pytest tests/test_ch00a_utils.py --impl=ref   # the reference

Oracles: collections.Counter and pandas.value_counts (count_values), np.argmax
(argmax), np.eye, scikit-learn OneHotEncoder and torch one_hot (one_hot),
torch.utils.data.BatchSampler (iterate_minibatches).
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def utils(mylearn_module):
    return mylearn_module("utils")


def assert_raises_value_error(function, *args, why="", **kwargs):
    """The call must raise ValueError (or a subclass); `why` names the case in the failure message."""
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    name = getattr(function, "__name__", "the function")
    raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else ""))


def type_names(values):
    return sorted({type(v).__name__ for v in values})


# ------------------------------------------------------------------ count_values
def test_count_values_matches_counter_pairs_and_order(utils):
    rng = np.random.default_rng(0)
    for _ in range(20):
        values = rng.choice(list("abcdef"), size=int(rng.integers(1, 40))).tolist()
        result = utils.count_values(list(values))
        expected = dict(Counter(values))
        assert result == expected, f"expected the counts {expected}, got {result}"
        assert list(result) == list(expected), "keys must keep the order of first appearance"


def test_count_values_normalize_matches_pandas(utils):
    values = ["Adelie", "Gentoo", "Adelie", "Chinstrap", "Adelie", "Gentoo"]
    result = utils.count_values(values, normalize=True)
    expected = pd.Series(values).value_counts(normalize=True).to_dict()
    assert set(result) == set(expected)
    for key in expected:
        assert result[key] == pytest.approx(expected[key])
    assert sum(result.values()) == pytest.approx(1.0)


def test_count_values_accepts_strings_tuples_arrays_series(utils):
    assert utils.count_values("abca") == {"a": 2, "b": 1, "c": 1}
    assert utils.count_values((3, 1, 3)) == {3: 2, 1: 1}
    assert utils.count_values(np.array([2, 2, 5])) == {2: 2, 5: 1}
    assert utils.count_values(pd.Series(["x", "y", "x"])) == {"x": 2, "y": 1}


def test_count_values_counts_are_ints_and_sum_to_length(utils):
    values = [1, 1, 2, 3, 3, 3]
    result = utils.count_values(list(values))
    assert all(isinstance(c, int) for c in result.values()), (
        f"the counts must be Python ints (int(...)), got {type_names(result.values())}")
    assert sum(result.values()) == len(values), f"the counts must sum to {len(values)}, got {sum(result.values())}"


@pytest.mark.parametrize("values, why", [
    pytest.param([], "an empty list", id="empty-list"),
    pytest.param("", "an empty string", id="empty-string"),
    pytest.param(np.array([]), "an empty array", id="empty-array"),
])
def test_count_values_empty_raises(utils, values, why):
    assert_raises_value_error(utils.count_values, values, why=why)


@pytest.mark.parametrize("values, why", [
    pytest.param([1.0, float("nan")], "a NaN in a list", id="nan-in-list"),
    pytest.param(np.array([2.0, np.nan, 2.0]), "a NaN in an array", id="nan-in-array"),
])
def test_count_values_nan_raises(utils, values, why):
    assert_raises_value_error(utils.count_values, values, why=why)


# ------------------------------------------------------------------------ argmax
def test_argmax_1d_matches_numpy_with_ties(utils):
    rng = np.random.default_rng(1)
    for _ in range(50):
        values = rng.integers(0, 4, size=int(rng.integers(1, 12)))
        result = utils.argmax(values.copy())
        assert result == np.argmax(values), (
            f"expected the index {np.argmax(values)} (the FIRST maximum), got {result}, for {values.tolist()}")
        assert isinstance(result, int), f"argmax must return a Python int (int(...)), got {type(result).__name__}"


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_argmax_2d_matches_numpy(utils, axis):
    rng = np.random.default_rng(2)
    for _ in range(30):
        values = rng.integers(0, 3, size=(int(rng.integers(1, 6)), int(rng.integers(1, 6))))
        result = utils.argmax(values.copy(), axis=axis)
        expected = np.argmax(values, axis=axis)
        if axis is None:
            assert result == expected, f"expected the flat index {expected}, got {result}, for {values.tolist()}"
            assert isinstance(result, int), f"argmax must return a Python int (int(...)), got {type(result).__name__}"
        else:
            assert isinstance(result, np.ndarray) and result.shape == expected.shape, (
                f"with axis={axis}, expected an array of shape {expected.shape}, got {type(result).__name__} "
                f"of shape {np.shape(result)}")
            np.testing.assert_array_equal(result, expected, err_msg=f"axis={axis}, values {values.tolist()}")


def test_argmax_accepts_lists_and_floats(utils):
    assert utils.argmax([3, 7, 7, 1]) == 1
    assert utils.argmax([0.5, -2.0, 0.75]) == 2
    np.testing.assert_array_equal(utils.argmax([[1, 9, 9], [8, 2, 8]], axis=0), [1, 0, 0])


@pytest.mark.parametrize("axis", [0, -1])
def test_argmax_1d_axis_0_and_minus_1(utils, axis):
    assert utils.argmax([4, 9, 2], axis=axis) == 1


@pytest.mark.parametrize("values, axis, why", [
    pytest.param([], None, "an empty input", id="empty"),
    pytest.param([[1, 2]], 2, "axis=2 (only None, 0, 1 and -1 exist)", id="axis-2"),
    pytest.param([1, 2], 1, "axis=1 for a 1-D input", id="axis-1-on-1d"),
    pytest.param([1.0, float("nan")], None, "a NaN", id="nan"),
    pytest.param(np.zeros((2, 2, 2)), None, "a 3-D input (only 1-D and 2-D)", id="3d"),
])
def test_argmax_invalid_inputs_raise(utils, values, axis, why):
    assert_raises_value_error(utils.argmax, values, axis=axis, why=why)


# ----------------------------------------------------------------------- one_hot
def test_one_hot_matches_numpy_eye(utils):
    rng = np.random.default_rng(3)
    for _ in range(20):
        k = int(rng.integers(1, 6))
        y = rng.integers(0, k, size=int(rng.integers(1, 15)))
        result = utils.one_hot(y, n_classes=k)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.eye(k)[y])


def test_one_hot_matches_sklearn(utils):
    from sklearn.preprocessing import OneHotEncoder

    y = np.array([2, 0, 1, 2, 2])
    encoder = OneHotEncoder(categories=[list(range(4))], sparse_output=False)
    expected = encoder.fit_transform(y.reshape(-1, 1))
    np.testing.assert_array_equal(utils.one_hot(y, n_classes=4), expected)


def test_one_hot_matches_torch(utils):
    torch = pytest.importorskip("torch")
    y = [1, 0, 3, 3]
    expected = torch.nn.functional.one_hot(torch.as_tensor(y), 4).numpy()
    np.testing.assert_array_equal(utils.one_hot(y, n_classes=4, dtype=np.int64), expected)


def test_one_hot_default_n_classes_dtype_and_float_labels(utils):
    result = utils.one_hot([0, 2, 1])
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result.sum(axis=1), [1, 1, 1])
    assert utils.one_hot([1, 0], n_classes=3, dtype=int).dtype == np.dtype(int)
    np.testing.assert_array_equal(utils.one_hot([2.0, 0.0]), [[0, 0, 1], [1, 0, 0]])


def test_one_hot_empty_with_n_classes(utils):
    assert utils.one_hot([], n_classes=3).shape == (0, 3)


@pytest.mark.parametrize("y, n_classes, why", [
    pytest.param([[0, 1]], None, "a 2-D y", id="2d"),
    pytest.param([], None, "an empty y without n_classes", id="empty"),
    pytest.param([0, 1.5], None, "a non-integer label (1.5)", id="non-integer"),
    pytest.param([0, -1], None, "a negative label", id="negative"),
    pytest.param([0, 3], 3, "a label >= n_classes (3 with n_classes=3)", id="too-large"),
])
def test_one_hot_invalid_labels_raise(utils, y, n_classes, why):
    assert_raises_value_error(utils.one_hot, y, n_classes=n_classes, why=why)


# ----------------------------------------------------------- iterate_minibatches
@pytest.mark.parametrize("n, b, drop_last", [(10, 3, False), (10, 3, True), (12, 4, False),
                                             (12, 4, True), (5, 10, False), (1, 1, False)])
def test_iterate_minibatches_matches_torch_batch_sampler(utils, n, b, drop_last):
    torch_data = pytest.importorskip("torch.utils.data")
    order = np.random.default_rng(7).permutation(n)
    expected = list(torch_data.BatchSampler(order.tolist(), b, drop_last))
    result = utils.iterate_minibatches(n, b, rng=np.random.default_rng(7), drop_last=drop_last)
    got = [np.asarray(batch).tolist() for batch in result]
    want = [list(map(int, batch)) for batch in expected]
    assert got == want, f"expected the batches {want} (permutation of default_rng(7), cut in order), got {got}"


def test_iterate_minibatches_without_shuffle(utils):
    result = [np.asarray(batch).tolist() for batch in utils.iterate_minibatches(5, 2, shuffle=False)]
    assert result == [[0, 1], [2, 3], [4]], f"expected [[0, 1], [2, 3], [4]] without shuffling, got {result}"
    result = [np.asarray(batch).tolist() for batch in utils.iterate_minibatches(5, 2, shuffle=False, drop_last=True)]
    assert result == [[0, 1], [2, 3]], f"expected [[0, 1], [2, 3]] with drop_last=True, got {result}"


def test_iterate_minibatches_partition_and_sizes(utils):
    batches = utils.iterate_minibatches(344, 32, rng=np.random.default_rng(0))
    assert len(batches) == 11
    assert [len(b) for b in batches] == [32] * 10 + [24]
    assert sorted(np.concatenate(batches).tolist()) == list(range(344))
    assert all(isinstance(b, np.ndarray) and b.dtype == np.int64 and b.ndim == 1 for b in batches), (
        "iterate_minibatches must return a list of 1-D NumPy arrays of int64 indices, got "
        + ", ".join(sorted({f"{type(b).__name__} {getattr(b, 'dtype', '')}" for b in batches})))
    assert len(utils.iterate_minibatches(344, 32, rng=np.random.default_rng(0), drop_last=True)) == 10


def test_iterate_minibatches_reproducible_and_new_order_each_call(utils):
    first = utils.iterate_minibatches(20, 5, rng=np.random.default_rng(3))
    again = utils.iterate_minibatches(20, 5, rng=np.random.default_rng(3))
    assert all((a == b).all() for a, b in zip(first, again))
    rng = np.random.default_rng(3)
    epoch1 = np.concatenate(utils.iterate_minibatches(20, 5, rng=rng))
    epoch2 = np.concatenate(utils.iterate_minibatches(20, 5, rng=rng))
    assert not (epoch1 == epoch2).all(), "a generator must give a new order at every call (epoch)"


@pytest.mark.parametrize("n, b, why", [
    pytest.param(0, 4, "n_samples = 0", id="no-samples"),
    pytest.param(10, 0, "batch_size = 0", id="batch-size-0"),
    pytest.param(-3, 2, "a negative n_samples", id="negative-n"),
])
def test_iterate_minibatches_invalid_arguments_raise(utils, n, b, why):
    assert_raises_value_error(utils.iterate_minibatches, n, b, why=why)
