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
    assert isinstance(value, float), f"{name} must return a Python float here, got {type(value).__name__}"


def assert_raises_value_error(function, *args, **kwargs):
    with pytest.raises(ValueError):
        function(*args, **kwargs)


# ------------------------------------------------------------------ mean, median, mode
def test_mean_matches_numpy(st):
    for seed in range(40):
        x = random_data(seed)
        result = st.mean(x)
        assert_python_float(result, "mean(x)")
        assert_close(result, np.mean(x))
        for axis in range(-x.ndim, x.ndim):
            assert_close(st.mean(x, axis=axis), np.mean(x, axis=axis))


def test_mean_returns_an_array_without_the_reduced_axis(st):
    x = random_data(1, shape=(4, 3))
    assert np.shape(st.mean(x, axis=0)) == (3,)
    assert np.shape(st.mean(x, axis=1)) == (4,)


def test_mean_accepts_lists_and_pandas(st):
    assert_close(st.mean([1, 2, 3, 4]), 2.5)
    assert_close(st.mean(pd.Series([2.0, 4.0, 9.0])), 5.0)
    assert_close(st.mean([[1, 2], [3, 4]], axis=0), [2.0, 3.0])


@pytest.mark.parametrize("bad", [[], [1.0, np.nan, 3.0], np.empty((0, 3))])
def test_mean_rejects_empty_or_nan(st, bad):
    assert_raises_value_error(st.mean, bad)


def test_median_matches_numpy(st):
    for seed in range(40):
        x = random_data(seed)
        result = st.median(x)
        assert_python_float(result, "median(x)")
        assert_close(result, np.median(x))
        for axis in range(-x.ndim, x.ndim):
            assert_close(st.median(x, axis=axis), np.median(x, axis=axis))


def test_median_of_odd_and_even_counts(st):
    assert_close(st.median([7, 1, 3]), 3.0)
    assert_close(st.median([7, 1, 3, 5]), 4.0)


def test_median_does_not_modify_its_input(st):
    x = np.array([5.0, 1.0, 4.0, 2.0])
    st.median(x)
    assert x.tolist() == [5.0, 1.0, 4.0, 2.0], "median must not sort its input in place"


@pytest.mark.parametrize("bad", [[], [2.0, np.nan]])
def test_median_rejects_empty_or_nan(st, bad):
    assert_raises_value_error(st.median, bad)


@pytest.mark.parametrize("values", [
    [1, 2, 2, 3], [4, 1, 4, 1, 7], [3, 3, 3], [5, 1, 2, 9], [2.5, 2.5, -1.0, -1.0, 0.0],
    ["Adelie", "Gentoo", "Adelie", "Chinstrap"], ["b", "a", "b", "a"],
])
def test_mode_matches_statistics_and_pandas(st, values):
    result = st.mode(values)
    assert isinstance(result, np.ndarray), "mode must return a NumPy array"
    assert result.ndim == 1
    expected = sorted(statistics.multimode(values))
    assert result.tolist() == expected
    assert result.tolist() == pd.Series(values).mode().tolist()


def test_mode_on_random_integers(st):
    for seed in range(20):
        values = np.random.default_rng(seed).integers(0, 6, size=15).tolist()
        assert st.mode(values).tolist() == sorted(statistics.multimode(values))


@pytest.mark.parametrize("bad", [[], [[1, 2], [2, 2]]])
def test_mode_rejects_empty_or_2d(st, bad):
    assert_raises_value_error(st.mode, bad)


# ------------------------------------------------------ variance, std, percentile, zscore
@pytest.mark.parametrize("ddof", [0, 1])
def test_variance_matches_numpy(st, ddof):
    for seed in range(40):
        x = random_data(seed, shape=(int(np.random.default_rng(seed).integers(2, 9)), 3))
        result = st.variance(x, ddof=ddof)
        assert_python_float(result, "variance(x)")
        assert_close(result, np.var(x, ddof=ddof))
        for axis in (0, 1, -1):
            assert_close(st.variance(x, ddof=ddof, axis=axis), np.var(x, ddof=ddof, axis=axis))


def test_variance_is_accurate_for_large_values(st):
    x = 1e6 + np.random.default_rng(5).normal(0, 1, size=200)    # large values, small spread
    assert_close(st.variance(x), np.var(x), rtol=1e-6, msg=(
        "compute the deviations from the mean first: mean(x**2) - mean(x)**2 loses precision "
        "when the values are large compared with their spread"))


def test_variance_default_divides_by_n(st):
    assert_close(st.variance([1, 2, 3, 4]), 1.25)
    assert_close(st.variance([1, 2, 3, 4], ddof=1), 5 / 3)


@pytest.mark.parametrize("bad, ddof", [([], 0), ([3.0], 1), ([1.0, np.nan], 0)])
def test_variance_rejects_bad_inputs(st, bad, ddof):
    assert_raises_value_error(st.variance, bad, ddof=ddof)


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_matches_numpy(st, ddof):
    for seed in range(40):
        x = random_data(seed, shape=(5, 4))
        result = st.std(x, ddof=ddof)
        assert_python_float(result, "std(x)")
        assert_close(result, np.std(x, ddof=ddof))
        for axis in (0, 1):
            assert_close(st.std(x, ddof=ddof, axis=axis), np.std(x, ddof=ddof, axis=axis))


def test_std_rejects_a_single_value_with_ddof_1(st):
    assert_raises_value_error(st.std, [7.0], ddof=1)


def test_percentile_matches_numpy(st):
    for seed in range(40):
        x = random_data(seed)
        q = float(np.random.default_rng(seed + 100).uniform(0, 100))
        result = st.percentile(x, q)
        assert_python_float(result, "percentile(x, q) with a scalar q")
        assert_close(result, np.percentile(x, q))
        qs = [0, 10, 25, 50, 75, 90, 100]
        assert_close(st.percentile(x, qs), np.percentile(x, qs))
        for axis in range(x.ndim):
            assert_close(st.percentile(x, q, axis=axis), np.percentile(x, q, axis=axis))
            assert_close(st.percentile(x, qs, axis=axis), np.percentile(x, qs, axis=axis))


def test_percentile_extremes_are_min_and_max(st):
    x = [4.0, -2.0, 9.5, 3.0]
    assert_close(st.percentile(x, 0), -2.0)
    assert_close(st.percentile(x, 100), 9.5)
    assert_close(st.percentile(x, [25, 75]), np.percentile(x, [25, 75]))


def test_percentile_does_not_modify_its_input(st):
    x = np.array([3.0, 1.0, 2.0])
    st.percentile(x, 50)
    assert x.tolist() == [3.0, 1.0, 2.0], "percentile must not sort its input in place"


@pytest.mark.parametrize("q", [-1, 100.5, [10, 101]])
def test_percentile_rejects_q_outside_0_100(st, q):
    assert_raises_value_error(st.percentile, [1.0, 2.0, 3.0], q)


@pytest.mark.parametrize("ddof", [0, 1])
def test_zscore_matches_scipy(st, ddof):
    for seed in range(30):
        x = np.random.default_rng(seed).normal(5, 3, size=(6, 3))   # no constant row or column
        assert_close(st.zscore(x, ddof=ddof), scipy_stats.zscore(x, axis=None, ddof=ddof))
        assert_close(st.zscore(x, ddof=ddof, axis=0), scipy_stats.zscore(x, axis=0, ddof=ddof))
        assert_close(st.zscore(x, ddof=ddof, axis=1), scipy_stats.zscore(x, axis=1, ddof=ddof),
                     msg="with axis=1, the mean and the standard deviation are computed row by row "
                         "(put the reduced axis back, e.g. with np.expand_dims, before subtracting)")


def test_zscore_has_mean_0_and_std_1(st):
    x = random_data(7, shape=(50, 4))
    z = np.asarray(st.zscore(x, axis=0))
    assert z.shape == x.shape
    assert_close(z.mean(axis=0), np.zeros(4), atol=1e-12)
    assert_close(z.std(axis=0), np.ones(4))


@pytest.mark.parametrize("bad, axis", [([5.0, 5.0, 5.0], None), ([[1.0, 2.0], [3.0, 2.0]], 0), ([], None)])
def test_zscore_rejects_constant_or_empty_data(st, bad, axis):
    assert_raises_value_error(st.zscore, bad, axis=axis)


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
            counts, edges = st.histogram(x, bins=bins, bin_range=bin_range, density=density)
            expected_counts, expected_edges = np.histogram(x, bins=bins, range=bin_range, density=density)
            assert np.shape(counts) == (bins,) and np.shape(edges) == (bins + 1,)
            assert_close(edges, expected_edges)
            assert_close(counts, expected_counts)


def test_histogram_counts_are_integers_and_last_bin_keeps_its_right_edge(st):
    counts, edges = st.histogram([0, 1, 2, 3, 4], bins=4)
    assert np.issubdtype(np.asarray(counts).dtype, np.integer), "counts must be integers when density=False"
    assert np.asarray(counts).tolist() == [1, 1, 1, 2]
    assert_close(edges, [0, 1, 2, 3, 4])


@pytest.mark.parametrize("x, bins, bin_range", [
    (list(range(11)), 5, (0, 10)),          # edges 0, 2, 4, 6, 8, 10: every even value is on an edge
    ([0.5, 1.0, 1.5, 2.0, 2.0], 4, (0, 2)),  # edges 0, 0.5, 1, 1.5, 2
    ([-4, -2, 0, 2, 4], 2, None),           # edges -4, 0, 4
])
def test_histogram_puts_values_on_an_edge_in_the_right_bin(st, x, bins, bin_range):
    counts, edges = st.histogram(x, bins=bins, bin_range=bin_range)
    expected_counts, expected_edges = np.histogram(x, bins=bins, range=bin_range)
    assert np.asarray(counts).tolist() == expected_counts.tolist(), (
        "a value equal to an edge goes to the bin on its right (except the last edge)")
    assert_close(edges, expected_edges)


def test_histogram_ignores_values_outside_the_range(st):
    counts, _ = st.histogram([-5, 0.5, 1.5, 2.5, 99], bins=3, bin_range=(0, 3))
    assert np.asarray(counts).tolist() == [1, 1, 1]


def test_histogram_density_has_area_1(st):
    x = np.random.default_rng(3).exponential(2.0, size=300)
    density, edges = st.histogram(x, bins=12, density=True)
    assert_close(np.sum(np.asarray(density) * np.diff(edges)), 1.0)


@pytest.mark.parametrize("kwargs", [{"bins": 0}, {"bin_range": (3.0, 3.0)}, {"bin_range": (4.0, 1.0)}])
def test_histogram_rejects_bad_bins_or_range(st, kwargs):
    assert_raises_value_error(st.histogram, [1.0, 2.0, 3.0], **kwargs)


@pytest.mark.parametrize("bad", [[], [2.0, 2.0, 2.0], [1.0, np.nan]])
def test_histogram_rejects_empty_constant_or_nan_data_without_range(st, bad):
    assert_raises_value_error(st.histogram, bad)


# --------------------------------------------------------------- covariance, correlation
@pytest.mark.parametrize("ddof", [0, 1])
def test_covariance_matches_numpy(st, ddof):
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 40))
        x = rng.normal(0, 5, size=n)
        y = 0.7 * x + rng.normal(0, 2, size=n) if seed % 2 else rng.normal(10, 1, size=n)
        result = st.covariance(x, y, ddof=ddof)
        assert_python_float(result, "covariance(x, y)")
        assert_close(result, np.cov(x, y, ddof=ddof)[0, 1])
        assert_close(st.covariance(x, x, ddof=ddof), np.var(x, ddof=ddof))


def test_covariance_default_is_ddof_0(st):
    assert_close(st.covariance([1, 2, 3, 4], [2, 4, 6, 8]), 2.5)


@pytest.mark.parametrize("x, y, ddof", [
    ([1.0, 2.0], [1.0, 2.0, 3.0], 0),
    ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], 0),
    ([1.0], [2.0], 1),
    ([1.0, np.nan, 3.0], [1.0, 2.0, 3.0], 0),
])
def test_covariance_rejects_bad_inputs(st, x, y, ddof):
    assert_raises_value_error(st.covariance, x, y, ddof=ddof)


def test_correlation_matches_numpy_and_scipy(st):
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 60))
        x = rng.normal(0, 5, size=n)
        y = rng.uniform(-2, 2) * x + rng.normal(0, 3, size=n)
        result = st.correlation(x, y)
        assert_python_float(result, "correlation(x, y)")
        assert -1.0 - 1e-12 <= result <= 1.0 + 1e-12, "a correlation lies between -1 and 1"
        assert_close(result, np.corrcoef(x, y)[0, 1])
        assert_close(result, scipy_stats.pearsonr(x, y).statistic)


def test_correlation_of_exact_lines_is_plus_or_minus_one(st):
    x = np.arange(10.0)
    assert_close(st.correlation(x, 3 * x - 7), 1.0)
    assert_close(st.correlation(x, -0.5 * x + 2), -1.0)


def test_correlation_does_not_depend_on_units(st):
    rng = np.random.default_rng(11)
    x, y = rng.normal(0, 1, 30), rng.normal(0, 1, 30)
    x_changed, y_changed = 1000 * x + 3, 0.01 * y - 40      # other units, other origins
    assert_close(st.correlation(x_changed, y_changed), np.corrcoef(x, y)[0, 1], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("x, y", [
    ([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]),
    ([5.0, 5.0], [1.0, 2.0]),
    ([1.0], [2.0]),
    ([1.0, 2.0], [1.0, 2.0, 3.0]),
    ([1.0, 2.0, np.nan], [1.0, 2.0, 3.0]),
])
def test_correlation_rejects_constant_short_or_bad_inputs(st, x, y):
    assert_raises_value_error(st.correlation, x, y)


# --------------------------------------------------------------------------- matrices
@pytest.mark.parametrize("ddof", [0, 1])
def test_covariance_matrix_matches_numpy(st, ddof):
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n, p = int(rng.integers(3, 30)), int(rng.integers(1, 6))
        X = rng.normal(0, 3, size=(n, p)) @ rng.normal(0, 1, size=(p, p))
        C = np.asarray(st.covariance_matrix(X, ddof=ddof))
        assert C.shape == (p, p)
        assert_close(C, np.atleast_2d(np.cov(X, rowvar=False, ddof=ddof)))
        assert_close(C, C.T, atol=1e-12)
        assert_close(np.diag(C), np.var(X, axis=0, ddof=ddof))
        assert_close(st.covariance_matrix(X), np.atleast_2d(np.cov(X, rowvar=False, ddof=0)),
                     msg="by default, covariance_matrix divides by n (ddof=0), unlike np.cov")


def test_covariance_matrix_of_the_docstring(st):
    assert_close(st.covariance_matrix([[1, 2], [2, 4], [3, 6], [4, 8]]), [[1.25, 2.5], [2.5, 5.0]])


@pytest.mark.parametrize("X, ddof", [([1.0, 2.0, 3.0], 0), ([[1.0, 2.0]], 1), ([[1.0, np.nan], [2.0, 3.0]], 0)])
def test_covariance_matrix_rejects_bad_inputs(st, X, ddof):
    assert_raises_value_error(st.covariance_matrix, X, ddof=ddof)


def test_correlation_matrix_matches_numpy_and_pandas(st):
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n, p = int(rng.integers(3, 40)), int(rng.integers(1, 6))
        X = rng.normal(0, 3, size=(n, p)) @ rng.normal(0, 1, size=(p, p)) + rng.normal(0, 50, size=p)
        R = np.asarray(st.correlation_matrix(X))
        assert R.shape == (p, p)
        assert_close(R, np.atleast_2d(np.corrcoef(X, rowvar=False)))
        assert_close(R, pd.DataFrame(X).corr().to_numpy())
        assert_close(np.diag(R), np.ones(p))
        assert np.all(np.abs(R) <= 1.0 + 1e-12), "a correlation lies between -1 and 1"


@pytest.mark.parametrize("X", [
    [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],   # a constant column
    [[1.0, 2.0]],                           # a single row
    [1.0, 2.0, 3.0],                        # not 2-D
])
def test_correlation_matrix_rejects_bad_inputs(st, X):
    assert_raises_value_error(st.correlation_matrix, X)


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
            assert np.shape(result) == np.shape(expected), f"shape {np.shape(result)} instead of {np.shape(expected)}"
            assert np.array_equal(np.asarray(result), expected), (
                "with the same seed, your draws must be exactly those of the algorithm of the docstring")


def test_sample_without_replacement_never_repeats(st):
    for seed in range(20):
        drawn = np.asarray(st.sample(np.arange(50), 50, replace=False, rng=np.random.default_rng(seed)))
        assert sorted(drawn.tolist()) == list(range(50))


def test_sample_with_replacement_keeps_about_63_percent_distinct(st):
    shares = [len(np.unique(st.sample(np.arange(1000), 1000, rng=np.random.default_rng(seed)))) / 1000
              for seed in range(10)]
    assert abs(np.mean(shares) - (1 - (1 - 1 / 1000) ** 1000)) < 0.01


def test_sample_works_without_a_generator(st):
    assert np.shape(st.sample([1, 2, 3], 5)) == (5,)


@pytest.mark.parametrize("population, size, replace", [
    ([1, 2, 3], -1, True), ([], 2, True), ([1, 2, 3], 4, False),
])
def test_sample_rejects_bad_arguments(st, population, size, replace):
    assert_raises_value_error(st.sample, population, size, replace=replace, rng=np.random.default_rng(0))


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
                assert result == expected
            else:
                assert np.shape(result) == (size,)
                assert np.array_equal(np.asarray(result), expected), (
                    "with the same seed, your draws must be exactly those of the algorithm of the docstring")


def test_sample_categorical_frequencies_pass_a_chi_square_test(st):
    p = np.array([0.34, 0.26, 0.2, 0.12, 0.08])
    draws = np.asarray(st.sample_categorical(p, size=100_000, rng=np.random.default_rng(2)))
    observed = np.bincount(draws, minlength=len(p))
    assert scipy_stats.chisquare(observed, 100_000 * p).pvalue > 0.001


def test_sample_categorical_never_draws_an_impossible_category(st):
    draws = np.asarray(st.sample_categorical([0.5, 0.0, 0.5, 0.0], size=2000, rng=np.random.default_rng(4)))
    assert set(draws.tolist()) <= {0, 2}


@pytest.mark.parametrize("p", [[[0.5, 0.5]], [0.5, -0.1, 0.6], [0.2, 0.2], []])
def test_sample_categorical_rejects_bad_probabilities(st, p):
    assert_raises_value_error(st.sample_categorical, p, size=3, rng=np.random.default_rng(0))


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
        result = st.bootstrap_distribution(x, statistic, n_boot=40, sample_size=sample_size,
                                           rng=np.random.default_rng(seed))
        assert np.shape(result) == (40,)
        assert_close(result, documented_bootstrap(x, statistic, 40, sample_size, seed))


def test_bootstrap_distribution_keeps_the_rows_paired(st):
    rng = np.random.default_rng(8)
    x = rng.normal(size=30)
    rows = np.column_stack([x, 2 * x + rng.normal(0, 0.5, size=30)])
    result = st.bootstrap_distribution(rows, pair_correlation, n_boot=25, rng=np.random.default_rng(1))
    assert_close(result, documented_bootstrap(rows, pair_correlation, 25, None, 1))


def test_bootstrap_distribution_spread_matches_scipy(st):
    x = np.random.default_rng(12).exponential(3.0, size=80)
    values = st.bootstrap_distribution(x, np.mean, n_boot=4000, rng=np.random.default_rng(3))
    reference = scipy_stats.bootstrap((x,), np.mean, n_resamples=4000, method="percentile",
                                      rng=np.random.default_rng(3))
    assert abs(np.std(values, ddof=1) / reference.standard_error - 1) < 0.1


@pytest.mark.parametrize("kwargs", [{"n_boot": 0}, {"sample_size": 0}])
def test_bootstrap_distribution_rejects_bad_arguments(st, kwargs):
    assert_raises_value_error(st.bootstrap_distribution, [1.0, 2.0, 3.0], np.mean,
                              rng=np.random.default_rng(0), **kwargs)


def test_bootstrap_distribution_rejects_empty_data(st):
    assert_raises_value_error(st.bootstrap_distribution, [], np.mean, rng=np.random.default_rng(0))


@pytest.mark.parametrize("confidence", [0.8, 0.95])
def test_bootstrap_ci_follows_the_documented_algorithm(st, confidence):
    for seed in range(10):
        x = np.random.default_rng(seed + 70).normal(50, 10, size=25)
        low, high = st.bootstrap_ci(x, np.mean, confidence=confidence, n_boot=300, rng=np.random.default_rng(seed))
        assert_python_float(low, "bootstrap_ci(...)[0]")
        assert_python_float(high, "bootstrap_ci(...)[1]")
        values = documented_bootstrap(x, np.mean, 300, None, seed)
        expected = np.percentile(values, [50 * (1 - confidence), 50 * (1 + confidence)])
        assert_close([low, high], expected, msg=(
            "low and high are the percentiles 50 * (1 - confidence) and 50 * (1 + confidence) of the bootstrap "
            "distribution, with linear interpolation (reuse your percentile), not sorted values cut by hand"))
        assert low <= high


def test_bootstrap_ci_is_close_to_scipy(st):
    x = np.random.default_rng(21).normal(100, 15, size=60)
    low, high = st.bootstrap_ci(x, np.mean, confidence=0.9, n_boot=5000, rng=np.random.default_rng(5))
    reference = scipy_stats.bootstrap((x,), np.mean, confidence_level=0.9, n_resamples=5000,
                                      method="percentile", rng=np.random.default_rng(6)).confidence_interval
    width = reference.high - reference.low
    assert abs(low - reference.low) < 0.1 * width and abs(high - reference.high) < 0.1 * width


def test_bootstrap_ci_smaller_resamples_give_a_wider_interval(st):
    x = np.random.default_rng(30).normal(0, 1, size=200)
    low_n, high_n = st.bootstrap_ci(x, np.mean, n_boot=2000, rng=np.random.default_rng(1))
    low_20, high_20 = st.bootstrap_ci(x, np.mean, n_boot=2000, sample_size=20, rng=np.random.default_rng(1))
    assert (high_20 - low_20) > 2 * (high_n - low_n)


@pytest.mark.parametrize("confidence", [0, 1, 1.5, -0.2])
def test_bootstrap_ci_rejects_a_bad_confidence(st, confidence):
    assert_raises_value_error(st.bootstrap_ci, [1.0, 2.0, 3.0], np.mean, confidence=confidence,
                              rng=np.random.default_rng(0))
