"""Tests of mylearn.stats (chapter 2): oracle tests and properties.

    pytest tests/test_ch02_stats.py              # your code (mon_travail/mylearn/stats.py)
    pytest tests/test_ch02_stats.py --impl=ref   # the reference

Oracles: NumPy (mean, median, var, std, percentile, histogram, cov, corrcoef, unique),
SciPy (stats.zscore, stats.pearsonr, stats.chisquare, stats.bootstrap), pandas
(Series.mode, DataFrame.corr) and statistics.multimode. The random functions follow
the algorithm written in their docstring: with the same seed, they must give exactly
the same draws as that algorithm (written again here with NumPy).
Every test name starts with the name of the function it tests (``-k "test_mean_"``).
"""

import statistics

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats


@pytest.fixture
def st(mylearn_module):
    return mylearn_module("stats")


def random_data(seed: int, shape=None):
    """Random numbers of a random shape (1-D to 3-D), sometimes integers."""
    rng = np.random.default_rng(seed)
    if shape is None:
        ndim = int(rng.integers(1, 4))
        shape = tuple(int(s) for s in rng.integers(1, 7, size=ndim))
    if seed % 3 == 0:
        return rng.integers(-20, 21, size=shape).astype(float)
    return rng.normal(rng.uniform(-50, 50), rng.uniform(0.5, 10), size=shape)


def assert_close(result, expected, rtol=1e-10, atol=1e-10, msg=""):
    np.testing.assert_allclose(np.asarray(result, dtype=float), np.asarray(expected, dtype=float),
                               rtol=rtol, atol=atol, err_msg=msg)


def assert_python_float(value, name):
    assert isinstance(value, float), f"{name} must return a float here (convert with float(...)), got {type(value).__name__}"


def assert_unchanged(array, before, call, hint):
    """The function must not change the caller's array (first line: what was expected)."""
    assert np.array_equal(array, before), (
        f"{call} changed the caller's array: expected it unchanged ({before.tolist()}), got {array.tolist()}; "
        f"{hint} (np.asarray(x, dtype=float) is x itself when x is already a float array, so -= and /= change it)")


def assert_raises_value_error(function, *args, why="", **kwargs):
    """The call must raise ValueError (or a subclass); `why` names the case in the failure message."""
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    name = getattr(function, "__name__", "the function")
    raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else ""))


def assert_shape(result, expected, call):
    assert np.shape(result) == expected, f"{call} must have shape {expected}, got {np.shape(result)}"


# ------------------------------------------------------------------ mean, median, mode
def test_mean_matches_numpy(st):
    for seed in range(40):
        x = random_data(seed)
        result = st.mean(x.copy())
        assert_python_float(result, "mean(x)")
        assert_close(result, np.mean(x), msg="mean(x) with axis=None: the mean of all the values")
        for axis in range(-x.ndim, x.ndim):
            assert_close(st.mean(x.copy(), axis=axis), np.mean(x, axis=axis), msg=f"mean(x, axis={axis})")


def test_mean_returns_an_array_without_the_reduced_axis(st):
    x = random_data(1, shape=(4, 3))
    assert_shape(st.mean(x, axis=0), (3,), "mean of a (4, 3) array with axis=0")
    assert_shape(st.mean(x, axis=1), (4,), "mean of a (4, 3) array with axis=1")


def test_mean_accepts_lists_and_pandas(st):
    assert_close(st.mean([1, 2, 3, 4]), 2.5)
    assert_close(st.mean(pd.Series([2.0, 4.0, 9.0])), 5.0)
    assert_close(st.mean([[1, 2], [3, 4]], axis=0), [2.0, 3.0])


@pytest.mark.parametrize("bad, why", [
    pytest.param([], "an empty list", id="empty"),
    pytest.param([1.0, np.nan, 3.0], "a NaN (remove the missing values first)", id="nan"),
    pytest.param(np.empty((0, 3)), "an array with no row", id="no-row"),
])
def test_mean_rejects_empty_or_nan(st, bad, why):
    assert_raises_value_error(st.mean, bad, why=why)


def test_median_matches_numpy(st):
    for seed in range(40):
        x = random_data(seed)
        result = st.median(x.copy())
        assert_python_float(result, "median(x)")
        assert_close(result, np.median(x), msg="median(x) with axis=None: the median of all the values")
        for axis in range(-x.ndim, x.ndim):
            assert_close(st.median(x.copy(), axis=axis), np.median(x, axis=axis), msg=f"median(x, axis={axis})")


def test_median_of_odd_and_even_counts(st):
    assert_close(st.median([7, 1, 3]), 3.0, msg="median([7, 1, 3]): the middle value once sorted")
    assert_close(st.median([7, 1, 3, 5]), 4.0, msg="median([7, 1, 3, 5]): the mean of the two middle values once sorted")


def test_median_does_not_modify_its_input(st):
    x = np.array([5.0, 1.0, 4.0, 2.0])
    before = x.copy()
    st.median(x)
    assert_unchanged(x, before, "median(x)", "sort a copy (np.sort(x)), not x itself (x.sort())")


@pytest.mark.parametrize("bad, why", [
    pytest.param([], "an empty list", id="empty"),
    pytest.param([2.0, np.nan], "a NaN", id="nan"),
])
def test_median_rejects_empty_or_nan(st, bad, why):
    assert_raises_value_error(st.median, bad, why=why)


@pytest.mark.parametrize("values", [
    [1, 2, 2, 3], [4, 1, 4, 1, 7], [3, 3, 3], [5, 1, 2, 9], [2.5, 2.5, -1.0, -1.0, 0.0],
    ["Adelie", "Gentoo", "Adelie", "Chinstrap"], ["b", "a", "b", "a"],
])
def test_mode_matches_statistics_and_pandas(st, values):
    result = st.mode(values)
    assert isinstance(result, np.ndarray), f"mode must return a NumPy array, got {type(result).__name__}"
    assert result.ndim == 1, f"mode must return a 1-D array (all the most frequent values), got {result.ndim} dimensions"
    expected = sorted(statistics.multimode(values))
    assert result.tolist() == expected, (
        f"mode({values}) must be {expected} (every most frequent value, sorted), got {result.tolist()}")
    assert result.tolist() == pd.Series(values).mode().tolist()


def test_mode_on_random_integers(st):
    for seed in range(20):
        values = np.random.default_rng(seed).integers(0, 6, size=15).tolist()
        expected = sorted(statistics.multimode(values))
        assert st.mode(values).tolist() == expected, (
            f"mode({values}) must be {expected} (every most frequent value, sorted), got {st.mode(values).tolist()}")


@pytest.mark.parametrize("bad, why", [
    pytest.param([], "an empty list", id="empty"),
    pytest.param([[1, 2], [2, 2]], "a 2-D list (mode works on one variable)", id="2d"),
])
def test_mode_rejects_empty_or_2d(st, bad, why):
    assert_raises_value_error(st.mode, bad, why=why)


# ------------------------------------------------------ variance, std, percentile, zscore
@pytest.mark.parametrize("ddof", [0, 1])
def test_variance_matches_numpy(st, ddof):
    for seed in range(40):
        x = random_data(seed, shape=(int(np.random.default_rng(seed).integers(2, 9)), 3))
        result = st.variance(x.copy(), ddof=ddof)
        assert_python_float(result, "variance(x)")
        assert_close(result, np.var(x, ddof=ddof), msg=f"variance(x, ddof={ddof}) with axis=None")
        for axis in (0, 1, -1):
            assert_close(st.variance(x.copy(), ddof=ddof, axis=axis), np.var(x, ddof=ddof, axis=axis),
                         msg=f"variance(x, ddof={ddof}, axis={axis})")


def test_variance_does_not_modify_its_input(st):
    x = np.array([2.0, 4.0, 4.0, 5.0])
    before = x.copy()
    st.variance(x, ddof=1)
    assert_unchanged(x, before, "variance(x, ddof=1)", "compute x - mean in a NEW array")


def test_variance_is_accurate_for_large_values(st):
    x = 1e6 + np.random.default_rng(5).normal(0, 1, size=200)    # large values, small spread
    assert_close(st.variance(x), np.var(x), rtol=1e-6, msg=(
        "compute the deviations from the mean first: mean(x**2) - mean(x)**2 loses precision "
        "when the values are large compared with their spread"))


def test_variance_default_divides_by_n(st):
    assert_close(st.variance([1, 2, 3, 4]), 1.25, msg="variance([1, 2, 3, 4]): ddof=0 by default, divide by n")
    assert_close(st.variance([1, 2, 3, 4], ddof=1), 5 / 3, msg="variance([1, 2, 3, 4], ddof=1): divide by n - 1")


@pytest.mark.parametrize("bad, ddof, why", [
    pytest.param([], 0, "an empty list", id="empty"),
    pytest.param([3.0], 1, "n - ddof = 0 (one value with ddof=1)", id="n-minus-ddof-0"),
    pytest.param([1.0, np.nan], 0, "a NaN", id="nan"),
])
def test_variance_rejects_bad_inputs(st, bad, ddof, why):
    assert_raises_value_error(st.variance, bad, ddof=ddof, why=why)


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_matches_numpy(st, ddof):
    for seed in range(40):
        x = random_data(seed, shape=(5, 4))
        result = st.std(x.copy(), ddof=ddof)
        assert_python_float(result, "std(x)")
        assert_close(result, np.std(x, ddof=ddof), msg=f"std(x, ddof={ddof}) with axis=None")
        for axis in (0, 1):
            assert_close(st.std(x.copy(), ddof=ddof, axis=axis), np.std(x, ddof=ddof, axis=axis),
                         msg=f"std(x, ddof={ddof}, axis={axis})")


def test_std_rejects_a_single_value_with_ddof_1(st):
    assert_raises_value_error(st.std, [7.0], ddof=1, why="n - ddof = 0 (one value with ddof=1)")


def test_percentile_matches_numpy(st):
    for seed in range(40):
        x = random_data(seed)
        q = float(np.random.default_rng(seed + 100).uniform(0, 100))
        result = st.percentile(x.copy(), q)
        assert_python_float(result, "percentile(x, q) with a scalar q")
        assert_close(result, np.percentile(x, q), msg=f"percentile(x, {q:.3f}) with axis=None")
        qs = [0, 10, 25, 50, 75, 90, 100]
        assert_close(st.percentile(x.copy(), qs), np.percentile(x, qs), msg=f"percentile(x, {qs})")
        for axis in range(x.ndim):
            assert_close(st.percentile(x.copy(), q, axis=axis), np.percentile(x, q, axis=axis),
                         msg=f"percentile(x, {q:.3f}, axis={axis})")
            assert_close(st.percentile(x.copy(), qs, axis=axis), np.percentile(x, qs, axis=axis),
                         msg=f"percentile(x, {qs}, axis={axis}): one row per q, then the axes that remain")


def test_percentile_extremes_are_min_and_max(st):
    x = [4.0, -2.0, 9.5, 3.0]
    assert_close(st.percentile(x, 0), -2.0, msg="percentile(x, 0) is the minimum")
    assert_close(st.percentile(x, 100), 9.5, msg="percentile(x, 100) is the maximum")
    assert_close(st.percentile(x, [25, 75]), np.percentile(x, [25, 75]), msg="percentile(x, [25, 75])")


def test_percentile_does_not_modify_its_input(st):
    x = np.array([3.0, 1.0, 2.0])
    before = x.copy()
    st.percentile(x, 50)
    assert_unchanged(x, before, "percentile(x, 50)", "sort a copy (np.sort(x)), not x itself (x.sort())")


@pytest.mark.parametrize("q, why", [
    pytest.param(-1, "q = -1 is below 0", id="q-negative"),
    pytest.param(100.5, "q = 100.5 is above 100", id="q-above-100"),
    pytest.param([10, 101], "q = 101, in a list, is above 100", id="q-list-above-100"),
])
def test_percentile_rejects_q_outside_0_100(st, q, why):
    assert_raises_value_error(st.percentile, [1.0, 2.0, 3.0], q, why=why)


@pytest.mark.parametrize("ddof", [0, 1])
def test_zscore_matches_scipy(st, ddof):
    for seed in range(30):
        x = np.random.default_rng(seed).normal(5, 3, size=(6, 3))   # no constant row or column
        assert_close(st.zscore(x.copy(), ddof=ddof), scipy_stats.zscore(x, axis=None, ddof=ddof))
        assert_close(st.zscore(x.copy(), ddof=ddof, axis=0), scipy_stats.zscore(x, axis=0, ddof=ddof))
        assert_close(st.zscore(x.copy(), ddof=ddof, axis=1), scipy_stats.zscore(x, axis=1, ddof=ddof),
                     msg="with axis=1, the mean and the standard deviation are computed row by row "
                         "(put the reduced axis back, e.g. with np.expand_dims, before subtracting)")


def test_zscore_does_not_modify_its_input(st):
    x = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 3.0]])
    before = x.copy()
    for axis in (None, 0, 1):
        st.zscore(x, axis=axis)
        assert_unchanged(x, before, f"zscore(x, axis={axis})", "compute (x - mean) / std in a NEW array")


def test_zscore_has_mean_0_and_std_1(st):
    x = random_data(7, shape=(50, 4))
    z = np.asarray(st.zscore(x.copy(), axis=0))
    assert_shape(z, x.shape, "zscore(x, axis=0)")
    assert_close(z.mean(axis=0), np.zeros(4), atol=1e-12, msg="after zscore(x, axis=0), every column has mean 0")
    assert_close(z.std(axis=0), np.ones(4), msg="after zscore(x, axis=0), every column has standard deviation 1")


@pytest.mark.parametrize("bad, axis, why", [
    pytest.param([5.0, 5.0, 5.0], None, "constant data: the standard deviation is 0", id="constant"),
    pytest.param([[1.0, 2.0], [3.0, 2.0]], 0, "the second column is constant (axis=0)", id="constant-column"),
    pytest.param([], None, "an empty list", id="empty"),
])
def test_zscore_rejects_constant_or_empty_data(st, bad, axis, why):
    assert_raises_value_error(st.zscore, bad, axis=axis, why=why)


# -------------------------------------------------------------------------- histogram
@pytest.mark.parametrize("density", [False, True])
def test_histogram_matches_numpy(st, density):
    # Continuous random data: no value sits exactly on an inner edge, where two correct
    # methods (comparisons with the edges, or the floor of a relative position) may
    # disagree by a rounding error. Values on edges are tested with exact numbers below.
    for seed in range(40):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 3, size=int(rng.integers(5, 200)))
        bins = int(rng.integers(1, 15))
        for bin_range in (None, (-4.0, 4.0), (float(x.min()), float(x.max()) + 1)):
            counts, edges = st.histogram(x.copy(), bins=bins, bin_range=bin_range, density=density)
            expected_counts, expected_edges = np.histogram(x, bins=bins, range=bin_range, density=density)
            call = f"histogram(x, bins={bins}, bin_range={bin_range}, density={density})"
            assert_shape(counts, (bins,), f"the counts of {call}")
            assert_shape(edges, (bins + 1,), f"the edges of {call}")
            assert_close(edges, expected_edges, msg=f"the edges of {call}")
            assert_close(counts, expected_counts, msg=f"the counts of {call}")


def test_histogram_counts_are_integers_and_last_bin_keeps_its_right_edge(st):
    counts, edges = st.histogram([0, 1, 2, 3, 4], bins=4)
    assert np.issubdtype(np.asarray(counts).dtype, np.integer), (
        f"counts must be integers when density=False, got dtype {np.asarray(counts).dtype}")
    assert np.asarray(counts).tolist() == [1, 1, 1, 2], (
        f"histogram([0, 1, 2, 3, 4], bins=4) must count [1, 1, 1, 2] (the last bin keeps its right edge, 4), "
        f"got {np.asarray(counts).tolist()}")
    assert_close(edges, [0, 1, 2, 3, 4], msg="the edges of histogram([0, 1, 2, 3, 4], bins=4)")


@pytest.mark.parametrize("x, bins, bin_range", [
    (list(range(11)), 5, (0, 10)),          # edges 0, 2, 4, 6, 8, 10: every even value is on an edge
    ([0.5, 1.0, 1.5, 2.0, 2.0], 4, (0, 2)),  # edges 0, 0.5, 1, 1.5, 2
    ([-4, -2, 0, 2, 4], 2, None),           # edges -4, 0, 4
])
def test_histogram_puts_values_on_an_edge_in_the_right_bin(st, x, bins, bin_range):
    counts, edges = st.histogram(x, bins=bins, bin_range=bin_range)
    expected_counts, expected_edges = np.histogram(x, bins=bins, range=bin_range)
    assert np.asarray(counts).tolist() == expected_counts.tolist(), (
        f"expected the counts {expected_counts.tolist()}, got {np.asarray(counts).tolist()}: a value equal to an "
        f"edge goes to the bin on its right (except the last edge)")
    assert_close(edges, expected_edges)


def test_histogram_ignores_values_outside_the_range(st):
    counts, _ = st.histogram([-5, 0.5, 1.5, 2.5, 99], bins=3, bin_range=(0, 3))
    assert np.asarray(counts).tolist() == [1, 1, 1], (
        f"expected the counts [1, 1, 1] (-5 and 99 are outside bin_range=(0, 3)), got {np.asarray(counts).tolist()}")


def test_histogram_density_has_area_1(st):
    x = np.random.default_rng(3).exponential(2.0, size=300)
    density, edges = st.histogram(x, bins=12, density=True)
    assert_close(np.sum(np.asarray(density) * np.diff(edges)), 1.0,
                 msg="with density=True, the area of the histogram (sum of height * width) is 1")


@pytest.mark.parametrize("kwargs, why", [
    pytest.param({"bins": 0}, "bins = 0", id="bins-0"),
    pytest.param({"bin_range": (3.0, 3.0)}, "an empty range, low = high = 3", id="empty-range"),
    pytest.param({"bin_range": (4.0, 1.0)}, "a reversed range, low = 4 > high = 1", id="reversed-range"),
])
def test_histogram_rejects_bad_bins_or_range(st, kwargs, why):
    assert_raises_value_error(st.histogram, [1.0, 2.0, 3.0], why=why, **kwargs)


@pytest.mark.parametrize("bad, why", [
    pytest.param([], "an empty list", id="empty"),
    pytest.param([2.0, 2.0, 2.0], "constant data and no bin_range: low = high", id="constant"),
    pytest.param([1.0, np.nan], "a NaN", id="nan"),
])
def test_histogram_rejects_empty_constant_or_nan_data_without_range(st, bad, why):
    assert_raises_value_error(st.histogram, bad, why=why)


# --------------------------------------------------------------- covariance, correlation
@pytest.mark.parametrize("ddof", [0, 1])
def test_covariance_matches_numpy(st, ddof):
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 40))
        x = rng.normal(0, 5, size=n)
        y = 0.7 * x + rng.normal(0, 2, size=n) if seed % 2 else rng.normal(10, 1, size=n)
        result = st.covariance(x.copy(), y.copy(), ddof=ddof)
        assert_python_float(result, "covariance(x, y)")
        assert_close(result, np.cov(x, y, ddof=ddof)[0, 1], msg=f"covariance(x, y, ddof={ddof})")
        assert_close(st.covariance(x.copy(), x.copy(), ddof=ddof), np.var(x, ddof=ddof),
                     msg=f"covariance(x, x, ddof={ddof}) is the variance of x")


def test_covariance_does_not_modify_its_inputs(st):
    x, y = np.array([1.0, 2.0, 4.0, 7.0]), np.array([3.0, 1.0, 2.0, 6.0])
    x_before, y_before = x.copy(), y.copy()
    st.covariance(x, y, ddof=1)
    assert_unchanged(x, x_before, "covariance(x, y)", "compute x - mean(x) in a NEW array")
    assert_unchanged(y, y_before, "covariance(x, y)", "compute y - mean(y) in a NEW array")


def test_covariance_default_is_ddof_0(st):
    assert_close(st.covariance([1, 2, 3, 4], [2, 4, 6, 8]), 2.5,
                 msg="covariance([1, 2, 3, 4], [2, 4, 6, 8]): ddof=0 by default, divide by n")


@pytest.mark.parametrize("x, y, ddof, why", [
    pytest.param([1.0, 2.0], [1.0, 2.0, 3.0], 0, "lengths 2 and 3 differ", id="different-lengths"),
    pytest.param([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], 0, "2-D inputs (x and y must be 1-D)", id="2d"),
    pytest.param([1.0], [2.0], 1, "n - ddof = 0 (one pair with ddof=1)", id="n-minus-ddof-0"),
    pytest.param([1.0, np.nan, 3.0], [1.0, 2.0, 3.0], 0, "a NaN in x", id="nan"),
])
def test_covariance_rejects_bad_inputs(st, x, y, ddof, why):
    assert_raises_value_error(st.covariance, x, y, ddof=ddof, why=why)


def test_correlation_matches_numpy_and_scipy(st):
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 60))
        x = rng.normal(0, 5, size=n)
        y = rng.uniform(-2, 2) * x + rng.normal(0, 3, size=n)
        result = st.correlation(x.copy(), y.copy())
        assert_python_float(result, "correlation(x, y)")
        assert -1.0 - 1e-12 <= result <= 1.0 + 1e-12, f"a correlation lies between -1 and 1, got {result}"
        assert_close(result, np.corrcoef(x, y)[0, 1], msg="correlation(x, y): covariance / (std(x) * std(y)), same ddof")
        assert_close(result, scipy_stats.pearsonr(x, y).statistic)


def test_correlation_of_exact_lines_is_plus_or_minus_one(st):
    x = np.arange(10.0)
    assert_close(st.correlation(x, 3 * x - 7), 1.0, msg="points on a rising line: correlation 1")
    assert_close(st.correlation(x, -0.5 * x + 2), -1.0, msg="points on a falling line: correlation -1")


def test_correlation_does_not_depend_on_units(st):
    rng = np.random.default_rng(11)
    x, y = rng.normal(0, 1, 30), rng.normal(0, 1, 30)
    x_changed, y_changed = 1000 * x + 3, 0.01 * y - 40      # other units, other origins
    assert_close(st.correlation(x_changed, y_changed), np.corrcoef(x, y)[0, 1], rtol=1e-7, atol=1e-7,
                 msg="the correlation does not change when x and y change units (a * x + b with a > 0)")


@pytest.mark.parametrize("x, y, why", [
    pytest.param([1.0, 2.0, 3.0], [4.0, 4.0, 4.0], "y is constant: its standard deviation is 0", id="y-constant"),
    pytest.param([5.0, 5.0], [1.0, 2.0], "x is constant: its standard deviation is 0", id="x-constant"),
    pytest.param([1.0], [2.0], "a single pair (n < 2)", id="one-pair"),
    pytest.param([1.0, 2.0], [1.0, 2.0, 3.0], "lengths 2 and 3 differ", id="different-lengths"),
    pytest.param([1.0, 2.0, np.nan], [1.0, 2.0, 3.0], "a NaN in x", id="nan"),
])
def test_correlation_rejects_constant_short_or_bad_inputs(st, x, y, why):
    assert_raises_value_error(st.correlation, x, y, why=why)


# --------------------------------------------------------------------------- matrices
@pytest.mark.parametrize("ddof", [0, 1])
def test_covariance_matrix_matches_numpy(st, ddof):
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n, p = int(rng.integers(3, 30)), int(rng.integers(1, 6))
        X = rng.normal(0, 3, size=(n, p)) @ rng.normal(0, 1, size=(p, p))
        C = np.asarray(st.covariance_matrix(X.copy(), ddof=ddof))
        assert_shape(C, (p, p), f"covariance_matrix of a ({n}, {p}) array (one row and one column per feature)")
        assert_close(C, np.atleast_2d(np.cov(X, rowvar=False, ddof=ddof)), msg=f"covariance_matrix(X, ddof={ddof})")
        assert_close(C, C.T, atol=1e-12, msg="a covariance matrix is symmetric")
        assert_close(np.diag(C), np.var(X, axis=0, ddof=ddof), msg="the diagonal holds the variances of the columns")
        assert_close(st.covariance_matrix(X.copy()), np.atleast_2d(np.cov(X, rowvar=False, ddof=0)),
                     msg="by default, covariance_matrix divides by n (ddof=0), unlike np.cov")


def test_covariance_matrix_does_not_modify_its_input(st):
    X = np.array([[1.0, 2.0], [2.0, 5.0], [4.0, 3.0]])
    before = X.copy()
    st.covariance_matrix(X, ddof=1)
    assert_unchanged(X, before, "covariance_matrix(X)", "center the columns in a NEW array")


def test_covariance_matrix_of_the_docstring(st):
    assert_close(st.covariance_matrix([[1, 2], [2, 4], [3, 6], [4, 8]]), [[1.25, 2.5], [2.5, 5.0]],
                 msg="the example of the docstring")


@pytest.mark.parametrize("X, ddof, why", [
    pytest.param([1.0, 2.0, 3.0], 0, "a 1-D list (X must be 2-D: one row per sample)", id="1d"),
    pytest.param([[1.0, 2.0]], 1, "n_samples - ddof = 0 (one row with ddof=1)", id="n-minus-ddof-0"),
    pytest.param([[1.0, np.nan], [2.0, 3.0]], 0, "a NaN", id="nan"),
])
def test_covariance_matrix_rejects_bad_inputs(st, X, ddof, why):
    assert_raises_value_error(st.covariance_matrix, X, ddof=ddof, why=why)


def test_correlation_matrix_matches_numpy_and_pandas(st):
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n, p = int(rng.integers(3, 40)), int(rng.integers(1, 6))
        X = rng.normal(0, 3, size=(n, p)) @ rng.normal(0, 1, size=(p, p)) + rng.normal(0, 50, size=p)
        R = np.asarray(st.correlation_matrix(X.copy()))
        assert_shape(R, (p, p), f"correlation_matrix of a ({n}, {p}) array (one row and one column per feature)")
        assert_close(R, np.atleast_2d(np.corrcoef(X, rowvar=False)), msg="correlation_matrix(X)")
        assert_close(R, pd.DataFrame(X).corr().to_numpy())
        assert_close(np.diag(R), np.ones(p), msg="the diagonal of a correlation matrix holds 1")
        assert np.all(np.abs(R) <= 1.0 + 1e-12), f"a correlation lies between -1 and 1, got {np.abs(R).max()}"


@pytest.mark.parametrize("X, why", [
    pytest.param([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], "the first column is constant", id="constant-column"),
    pytest.param([[1.0, 2.0]], "a single row", id="one-row"),
    pytest.param([1.0, 2.0, 3.0], "a 1-D list (X must be 2-D)", id="1d"),
])
def test_correlation_matrix_rejects_bad_inputs(st, X, why):
    assert_raises_value_error(st.correlation_matrix, X, why=why)


# ---------------------------------------------------------------------------- sampling
def documented_sample(population, size, replace, seed):
    rng = np.random.default_rng(seed)
    population = np.asarray(population)
    n = population.shape[0]
    if replace:
        return population[rng.integers(0, n, size)]
    return population[rng.permutation(n)[:size]]


@pytest.mark.parametrize("replace", [True, False])
def test_sample_follows_the_documented_algorithm(st, replace):
    for seed in range(20):
        rng = np.random.default_rng(seed + 500)
        n = int(rng.integers(1, 30))
        size = int(rng.integers(0, n + 1)) if not replace else int(rng.integers(0, 3 * n))
        for population in (np.arange(n) * 10, rng.normal(size=(n, 3)), [f"item{i}" for i in range(n)]):
            result = st.sample(population, size, replace=replace, rng=np.random.default_rng(seed))
            expected = documented_sample(population, size, replace, seed)
            assert_shape(result, np.shape(expected), f"sample(population, {size}, replace={replace})")
            assert np.array_equal(np.asarray(result), expected), (
                "with the same seed, your draws must be exactly those of the algorithm of the docstring")


def test_sample_without_replacement_never_repeats(st):
    for seed in range(20):
        drawn = np.asarray(st.sample(np.arange(50), 50, replace=False, rng=np.random.default_rng(seed)))
        assert sorted(drawn.tolist()) == list(range(50)), (
            "sample(np.arange(50), 50, replace=False) must return each of the 50 values once, in some order")


def test_sample_with_replacement_keeps_about_63_percent_distinct(st):
    shares = [len(np.unique(st.sample(np.arange(1000), 1000, rng=np.random.default_rng(seed)))) / 1000
              for seed in range(10)]
    expected = 1 - (1 - 1 / 1000) ** 1000
    assert abs(np.mean(shares) - expected) < 0.01, (
        f"expected about {expected:.3f} of distinct values with replacement (n = size = 1000), got {np.mean(shares):.3f}")


def test_sample_works_without_a_generator(st):
    assert_shape(st.sample([1, 2, 3], 5), (5,), "sample([1, 2, 3], 5) without rng (use np.random.default_rng())")


@pytest.mark.parametrize("population, size, replace, why", [
    pytest.param([1, 2, 3], -1, True, "size = -1 is negative", id="negative-size"),
    pytest.param([], 2, True, "an empty population", id="empty-population"),
    pytest.param([1, 2, 3], 4, False, "4 draws without replacement from 3 elements", id="too-many-without-replacement"),
])
def test_sample_rejects_bad_arguments(st, population, size, replace, why):
    assert_raises_value_error(st.sample, population, size, replace=replace, rng=np.random.default_rng(0), why=why)


def documented_categorical(p, size, seed):
    rng = np.random.default_rng(seed)
    u = rng.random(size)
    k = np.minimum(np.searchsorted(np.cumsum(p), u, side="right"), len(p) - 1)
    return int(k) if size is None else k


def test_sample_categorical_follows_the_documented_algorithm(st):
    for seed in range(30):
        rng = np.random.default_rng(seed + 900)
        p = rng.dirichlet(np.ones(int(rng.integers(1, 8))))
        for size in (None, 0, 1, 25):
            result = st.sample_categorical(p, size=size, rng=np.random.default_rng(seed))
            expected = documented_categorical(p, size, seed)
            if size is None:
                assert isinstance(result, int), f"a single draw must be a Python int, got {type(result).__name__}"
                assert result == expected, (
                    f"expected the draw {expected} (the algorithm of the docstring, same seed), got {result}")
            else:
                assert_shape(result, (size,), f"sample_categorical(p, size={size})")
                assert np.array_equal(np.asarray(result), expected), (
                    "with the same seed, your draws must be exactly those of the algorithm of the docstring")


def test_sample_categorical_frequencies_pass_a_chi_square_test(st):
    p = np.array([0.34, 0.26, 0.2, 0.12, 0.08])
    draws = np.asarray(st.sample_categorical(p, size=100_000, rng=np.random.default_rng(2)))
    observed = np.bincount(draws, minlength=len(p))
    assert scipy_stats.chisquare(observed, 100_000 * p).pvalue > 0.001, (
        f"expected frequencies close to p = {p.tolist()}, got {(observed / 100_000).round(3).tolist()}")


def test_sample_categorical_never_draws_an_impossible_category(st):
    draws = np.asarray(st.sample_categorical([0.5, 0.0, 0.5, 0.0], size=2000, rng=np.random.default_rng(4)))
    assert set(draws.tolist()) <= {0, 2}, (
        f"expected only the categories 0 and 2 (p = [0.5, 0, 0.5, 0]), got {sorted(set(draws.tolist()))}")


@pytest.mark.parametrize("p, why", [
    pytest.param([[0.5, 0.5]], "a 2-D p", id="2d"),
    pytest.param([0.5, -0.1, 0.6], "a negative probability", id="negative"),
    pytest.param([0.2, 0.2], "probabilities that sum to 0.4", id="sum-not-1"),
    pytest.param([], "an empty p (sums to 0)", id="empty"),
])
def test_sample_categorical_rejects_bad_probabilities(st, p, why):
    assert_raises_value_error(st.sample_categorical, p, size=3, rng=np.random.default_rng(0), why=why)


# --------------------------------------------------------------------------- bootstrap
def documented_bootstrap(x, statistic, n_boot, sample_size, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = x.shape[0]
    size = n if sample_size is None else sample_size
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, n, size=size)
        values.append(statistic(x[index]))
    return np.array(values, dtype=float)


def pair_correlation(rows):
    return np.corrcoef(rows[:, 0], rows[:, 1])[0, 1]


@pytest.mark.parametrize("statistic, sample_size", [(np.mean, None), (np.median, 7), (np.std, None)])
def test_bootstrap_distribution_follows_the_documented_algorithm(st, statistic, sample_size):
    for seed in range(10):
        x = np.random.default_rng(seed + 50).normal(20, 4, size=15)
        result = st.bootstrap_distribution(x.copy(), statistic, n_boot=40, sample_size=sample_size,
                                           rng=np.random.default_rng(seed))
        assert_shape(result, (40,), "bootstrap_distribution(x, ..., n_boot=40)")
        assert_close(result, documented_bootstrap(x, statistic, 40, sample_size, seed), msg=(
            "with the same seed, the resamples must be those of the algorithm of the docstring "
            "(rng.integers(0, n, size=...) once per resample)"))


def test_bootstrap_distribution_keeps_the_rows_paired(st):
    rng = np.random.default_rng(8)
    x = rng.normal(size=30)
    rows = np.column_stack([x, 2 * x + rng.normal(0, 0.5, size=30)])
    result = st.bootstrap_distribution(rows.copy(), pair_correlation, n_boot=25, rng=np.random.default_rng(1))
    assert_close(result, documented_bootstrap(rows, pair_correlation, 25, None, 1),
                 msg="a 2-D x is resampled by rows: draw row indices, keep each row whole")


def test_bootstrap_distribution_spread_matches_scipy(st):
    x = np.random.default_rng(12).exponential(3.0, size=80)
    values = st.bootstrap_distribution(x, np.mean, n_boot=4000, rng=np.random.default_rng(3))
    reference = scipy_stats.bootstrap((x,), np.mean, n_resamples=4000, method="percentile",
                                      rng=np.random.default_rng(3))
    ratio = np.std(values, ddof=1) / reference.standard_error
    assert abs(ratio - 1) < 0.1, f"expected the spread of SciPy's bootstrap (ratio 1 +/- 0.1), got a ratio of {ratio:.2f}"


@pytest.mark.parametrize("kwargs, why", [
    pytest.param({"n_boot": 0}, "n_boot = 0", id="n-boot-0"),
    pytest.param({"sample_size": 0}, "sample_size = 0", id="sample-size-0"),
])
def test_bootstrap_distribution_rejects_bad_arguments(st, kwargs, why):
    assert_raises_value_error(st.bootstrap_distribution, [1.0, 2.0, 3.0], np.mean,
                              rng=np.random.default_rng(0), why=why, **kwargs)


def test_bootstrap_distribution_rejects_empty_data(st):
    assert_raises_value_error(st.bootstrap_distribution, [], np.mean, rng=np.random.default_rng(0), why="an empty x")


@pytest.mark.parametrize("confidence", [0.8, 0.95])
def test_bootstrap_ci_follows_the_documented_algorithm(st, confidence):
    for seed in range(10):
        x = np.random.default_rng(seed + 70).normal(50, 10, size=25)
        low, high = st.bootstrap_ci(x.copy(), np.mean, confidence=confidence, n_boot=300, rng=np.random.default_rng(seed))
        assert_python_float(low, "bootstrap_ci(...)[0]")
        assert_python_float(high, "bootstrap_ci(...)[1]")
        values = documented_bootstrap(x, np.mean, 300, None, seed)
        expected = np.percentile(values, [50 * (1 - confidence), 50 * (1 + confidence)])
        assert_close([low, high], expected, msg=(
            "low and high are the percentiles 50 * (1 - confidence) and 50 * (1 + confidence) of the bootstrap "
            "distribution, with linear interpolation (reuse your percentile), not sorted values cut by hand"))
        assert low <= high, f"expected low <= high, got ({low}, {high})"


def test_bootstrap_ci_is_close_to_scipy(st):
    x = np.random.default_rng(21).normal(100, 15, size=60)
    low, high = st.bootstrap_ci(x, np.mean, confidence=0.9, n_boot=5000, rng=np.random.default_rng(5))
    reference = scipy_stats.bootstrap((x,), np.mean, confidence_level=0.9, n_resamples=5000,
                                      method="percentile", rng=np.random.default_rng(6)).confidence_interval
    width = reference.high - reference.low
    assert abs(low - reference.low) < 0.1 * width and abs(high - reference.high) < 0.1 * width, (
        f"expected an interval close to SciPy's ({reference.low:.2f}, {reference.high:.2f}), "
        f"got ({low:.2f}, {high:.2f})")


def test_bootstrap_ci_smaller_resamples_give_a_wider_interval(st):
    x = np.random.default_rng(30).normal(0, 1, size=200)
    low_n, high_n = st.bootstrap_ci(x, np.mean, n_boot=2000, rng=np.random.default_rng(1))
    low_20, high_20 = st.bootstrap_ci(x, np.mean, n_boot=2000, sample_size=20, rng=np.random.default_rng(1))
    assert (high_20 - low_20) > 2 * (high_n - low_n), (
        f"expected resamples of 20 values to give an interval more than twice as wide as resamples of 200, "
        f"got widths {high_20 - low_20:.3f} and {high_n - low_n:.3f} (is sample_size used?)")


@pytest.mark.parametrize("confidence", [0, 1, 1.5, -0.2])
def test_bootstrap_ci_rejects_a_bad_confidence(st, confidence):
    assert_raises_value_error(st.bootstrap_ci, [1.0, 2.0, 3.0], np.mean, confidence=confidence,
                              rng=np.random.default_rng(0), why=f"confidence = {confidence} is not in (0, 1)")
