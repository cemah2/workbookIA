"""Tests of mylearn.bayes (chapter 4): oracle tests and properties.

    pytest tests/test_ch04_bayes.py              # your code (mon_travail/mylearn/bayes.py)
    pytest tests/test_ch04_bayes.py --impl=ref   # the reference

Oracles: exact fractions (``fractions.Fraction``), NumPy (``np.dot``), SciPy
(``scipy.special.logsumexp``, ``scipy.special.xlogy``, ``scipy.stats.beta``) and
sequential updates written here with NumPy. Every test name starts with the name of the
function it tests (``-k "test_update_discrete_"``); the tests of one exercise never
call the functions of another one.
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

from fractions import Fraction

import numpy as np
import pytest
from scipy import special, stats


@pytest.fixture
def by(mylearn_module):
    return mylearn_module("bayes")


# ------------------------------------------------------------------ helpers
def listed(values) -> list:
    """Plain Python values for the messages (NumPy 2 would print np.float64(0.5))."""
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
    if details:
        lines.append(details)
    raise AssertionError("\n".join(lines)) from None


def _shape_head(got, expected):
    """When two arrays have different shapes, say so first: truncated views of them can look identical."""
    want = np.asarray(expected)
    if got.ndim != want.ndim or got.shape != want.shape:
        return (f"expected shape {want.shape}, got shape {got.shape}: "
                f"expected {_short(expected)}, got {_short(got)}")
    return None


def _first_difference(got, want, rtol, atol) -> str:
    """' (first difference at index i: expected a, got b)' for two arrays of the same shape, '' otherwise."""
    if got.ndim == 0 or got.shape != want.shape:
        return ""
    bad = np.flatnonzero(~np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=False))
    if bad.size == 0:
        return ""
    i = int(bad[0])
    index = np.unravel_index(i, got.shape)
    where = index[0] if got.ndim == 1 else tuple(int(k) for k in index)
    return f" (first difference at index {where}: expected {want.ravel()[i]:.6g}, got {got.ravel()[i]:.6g})"


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg="", data=""):
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected {_short(expected)}, got an object of type {type(result).__name__}", data, "")
    head = _shape_head(got, expected)
    if head:
        _fail(msg, head, data, "")
    want = np.asarray(expected, dtype=float)
    if np.isnan(got).any():
        _fail(msg, f"expected {_short(expected)}, got NaN values: {_short(result)}", data, "")
    try:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    except AssertionError as exc:
        have, need = _short(result), _short(expected)
        if have == need:   # they differ beyond the 6th digit: show every digit
            have, need = _short(result, 17), _short(expected, 17)
        _fail(msg, f"expected {need}, got {have}{_first_difference(got, want, rtol, atol)}", data, str(exc))


def assert_python_float(value, name):
    assert isinstance(value, float), f"{name} must return a Python float, got {type(value).__name__}"


def assert_distribution(value, name, n):
    """A 1-D float array of n probabilities summing to 1, without NaN."""
    assert isinstance(value, np.ndarray), f"{name} must return a NumPy array, got {type(value).__name__}"
    assert value.shape == (n,), f"{name} must return an array of shape ({n},), got shape {value.shape}"
    assert not np.isnan(value).any(), f"{name} returned NaN values: {_short(value)}"
    assert (value >= 0).all(), f"{name} returned negative probabilities: {_short(value)}"
    assert abs(value.sum() - 1) < 1e-9, f"the probabilities must sum to 1, got a sum of {value.sum()!r}"


def assert_raises_value_error(function, *args, why="", **kwargs):
    """The call must raise ValueError (or a subclass); `why` names the case in the failure message."""
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    except NotImplementedError:
        raise                  # shown as "⏳ pas encore implémenté" by conftest.py
    except Exception as exc:  # noqa: BLE001 - say which error was raised instead
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else "")
                             + f" (it raised {type(exc).__name__}: {exc})") from None
    name = getattr(function, "__name__", "the function")
    raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else ""))


def random_prior(rng, n):
    """A random distribution of n probabilities (Dirichlet), sometimes with a zero."""
    prior = rng.dirichlet(np.ones(n))
    if n > 2 and rng.random() < 0.3:
        prior[rng.integers(n)] = 0.0
        prior /= prior.sum()
    return prior


def sequential_oracle(prior, table, observations):
    """Bayes' rule applied observation by observation, written here with NumPy (oracle)."""
    posterior = np.asarray(prior, dtype=float)
    rows = [posterior]
    for o in observations:
        joint = posterior * np.asarray(table, dtype=float)[:, o]
        posterior = joint / joint.sum()
        rows.append(posterior)
    return np.array(rows)


def batch_log_oracle(prior, table, observations):
    """log prior + sum of the log-likelihoods, normalized with scipy.special.logsumexp (oracle)."""
    table = np.asarray(table, dtype=float)
    with np.errstate(divide="ignore"):
        log_post = np.log(np.asarray(prior, dtype=float)) + np.log(table[:, observations]).sum(axis=1)
    return np.exp(log_post - special.logsumexp(log_post))


def coin_oracle(flips, grid, prior=None):
    """prior x theta^h x (1 - theta)^t in log space (xlogy handles 0 log 0 = 0), normalized (oracle)."""
    flips, grid = np.asarray(flips), np.asarray(grid, dtype=float)
    h = int(np.sum(flips == 1))
    t = len(flips) - h
    prior = np.full(len(grid), 1 / len(grid)) if prior is None else np.asarray(prior, dtype=float)
    with np.errstate(divide="ignore"):
        log_post = np.log(prior) + special.xlogy(h, grid) + special.xlogy(t, 1 - grid)
    return np.exp(log_post - special.logsumexp(log_post))


# ------------------------------------------------------------------ evidence (4.14)
def test_evidence_matches_numpy_dot(by):
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(1, 12))
        prior, likelihood = random_prior(rng, n), rng.random(n)
        result = by.evidence(prior, likelihood)
        assert_python_float(result, "evidence")
        assert_close(result, np.dot(prior, likelihood), msg="the evidence is the sum of likelihood x prior",
                     data=f"prior={listed(prior)}, likelihood={listed(likelihood)}")


@pytest.mark.parametrize("prior, likelihood, exact", [
    pytest.param(["1/2", "1/2"], ["1/2", "2/3"], "7/12", id="fair-or-rigged-heads"),
    pytest.param(["1/2", "1/2"], ["1/2", "1/3"], "5/12", id="fair-or-rigged-tails"),
    pytest.param(["1/5"] * 5, ["0", "1/4", "1/2", "3/4", "1"], "1/2", id="five-biases"),
    pytest.param(["101/1000", "899/1000"], ["1/101", "869/899"], "870/1000", id="probe-nothing-detected"),
    pytest.param(["1"], ["3/10"], "3/10", id="a-single-hypothesis"),
])
def test_evidence_exact_values(by, prior, likelihood, exact):
    prior_f = [float(Fraction(p)) for p in prior]
    likelihood_f = [float(Fraction(p)) for p in likelihood]
    assert_close(by.evidence(prior_f, likelihood_f), float(Fraction(exact)),
                 msg="the evidence is the sum of likelihood x prior over all hypotheses",
                 data=f"prior={prior}, likelihood={likelihood}")


def test_evidence_likelihoods_need_not_sum_to_one(by):
    # P(heads | H_i) for three coins: 0.9 + 0.8 + 0.6 > 1, and that is fine
    assert_close(by.evidence([0.2, 0.3, 0.5], [0.9, 0.8, 0.6]), 0.2 * 0.9 + 0.3 * 0.8 + 0.5 * 0.6,
                 msg="likelihoods are P(obs | H_i), one per hypothesis: they need not sum to 1 (do not reject them)")


def test_evidence_tolerates_rounding_in_the_prior(by):
    prior = np.array([0.6, 0.3, 0.1])       # prior.sum() is 0.9999999999999999 in floating point
    assert prior.sum() != 1.0               # (the premise of this test)
    assert_close(by.evidence(prior, [0.5] * 3), 0.5,
                 msg="a prior that sums to 1 up to rounding (np.array([0.6, 0.3, 0.1]).sum() is "
                     "0.9999999999999999) is a valid distribution: compare the sum with a tolerance of 1e-8")


def test_evidence_accepts_lists_tuples_and_arrays(by):
    for prior, likelihood in [([0.25, 0.75], [0.4, 0.8]), ((0.25, 0.75), (0.4, 0.8)),
                              (np.array([0.25, 0.75]), np.array([0.4, 0.8]))]:
        assert_close(by.evidence(prior, likelihood), 0.7,
                     msg=f"inputs of type {type(prior).__name__} (convert them with np.asarray)")


@pytest.mark.parametrize("prior, likelihood, why", [
    pytest.param([0.5, 0.5], [0.5, 0.5, 0.5], "prior and likelihood of different lengths", id="different-lengths"),
    pytest.param([0.5, 0.5], [0.7], "different lengths (NumPy would broadcast [0.7])", id="length-1"),
    pytest.param([0.5, 0.6], [0.5, 0.5], "the prior sums to 1.1, it is not a distribution", id="prior-sum"),
    pytest.param([0.3, 0.3], [0.5, 0.5], "the prior sums to 0.6, it is not a distribution", id="prior-sum-low"),
    pytest.param([1.5, -0.5], [0.5, 0.5], "a negative prior probability (even if the sum is 1)", id="negative-prior"),
    pytest.param([0.5, 0.5], [0.5, 1.2], "a likelihood of 1.2 is not a probability", id="likelihood-above-1"),
    pytest.param([0.5, 0.5], [-0.1, 0.5], "a negative likelihood", id="negative-likelihood"),
    pytest.param([0.5, 0.5], [np.nan, 0.5], "a NaN likelihood", id="nan-likelihood"),
    pytest.param([np.nan, 1.0], [0.5, 0.5], "a NaN prior", id="nan-prior"),
    pytest.param([], [], "empty inputs: no hypothesis at all", id="empty"),
])
def test_evidence_rejects_bad_inputs(by, prior, likelihood, why):
    assert_raises_value_error(by.evidence, prior, likelihood, why=why)


# ------------------------------------------------------------------ bayes_posterior (4.14)
@pytest.mark.parametrize("likelihood, expected", [
    pytest.param(["1/2", "2/3"], ["3/7", "4/7"], id="heads"),
    pytest.param(["1/2", "1/3"], ["3/5", "2/5"], id="tails"),
])
def test_bayes_posterior_book_values(by, likelihood, expected):
    # the fair coin and the coin of bias 2/3 of the book, equally likely (§4.4)
    result = by.bayes_posterior([0.5, 0.5], [float(Fraction(p)) for p in likelihood])
    assert_close(result, [float(Fraction(p)) for p in expected],
                 msg="P(H_i | obs) = P(obs | H_i) P(H_i) / evidence", data=f"prior=[0.5, 0.5], likelihood={likelihood}")


def test_bayes_posterior_exact_values_with_fractions(by):
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(2, 7))
        prior = [Fraction(int(k), 1) for k in rng.integers(1, 10, size=n)]
        prior = [p / sum(prior) for p in prior]
        likelihood = [Fraction(int(k), 12) for k in rng.integers(0, 13, size=n)]
        if sum(p * q for p, q in zip(prior, likelihood)) == 0:
            continue
        total = sum(p * q for p, q in zip(prior, likelihood))
        expected = [float(p * q / total) for p, q in zip(prior, likelihood)]
        result = by.bayes_posterior([float(p) for p in prior], [float(q) for q in likelihood])
        assert_close(result, expected, msg="P(H_i | obs) = P(obs | H_i) P(H_i) / evidence",
                     data=f"prior={[str(p) for p in prior]}, likelihood={[str(q) for q in likelihood]}")


def test_bayes_posterior_returns_a_distribution(by):
    rng = np.random.default_rng(2)
    for _ in range(20):
        n = int(rng.integers(1, 15))
        prior, likelihood = random_prior(rng, n), rng.random(n)
        result = by.bayes_posterior(prior, likelihood)
        assert_distribution(result, "bayes_posterior", n)


def test_bayes_posterior_zero_prior_stays_zero(by):
    result = by.bayes_posterior([0.0, 0.4, 0.6], [0.9, 0.5, 0.1])
    assert np.asarray(result)[0] == 0, (f"a hypothesis with prior 0 must keep posterior exactly 0, "
                                        f"got {_short(result)}")


def test_bayes_posterior_uniform_likelihood_leaves_the_prior_unchanged(by):
    prior = [0.1, 0.2, 0.3, 0.4]
    for value in (0.3, 0.8, 1.0):
        assert_close(by.bayes_posterior(prior, [value] * 4), prior,
                     msg=f"an observation that is equally likely under every hypothesis ({value}) teaches nothing")


def test_bayes_posterior_does_not_modify_its_inputs(by):
    prior, likelihood = np.array([0.25, 0.25, 0.5]), np.array([0.2, 0.6, 0.4])
    by.bayes_posterior(prior, likelihood)
    assert listed(prior) == [0.25, 0.25, 0.5], f"prior was modified in place: {listed(prior)} (work on a copy)"
    assert listed(likelihood) == [0.2, 0.6, 0.4], f"likelihood was modified in place: {listed(likelihood)}"


@pytest.mark.parametrize("prior, likelihood, why", [
    pytest.param([1.0, 0.0], [0.0, 1.0], "the evidence is 0: the observation is impossible under every hypothesis",
                 id="evidence-0"),
    pytest.param([0.5, 0.5], [0.0, 0.0], "the evidence is 0: every likelihood is 0", id="all-likelihoods-0"),
    pytest.param([0.5, 0.5], [0.5, 0.5, 0.5], "prior and likelihood of different lengths", id="different-lengths"),
    pytest.param([0.5, 0.5], [0.7], "different lengths (NumPy would broadcast [0.7])", id="length-1"),
    pytest.param([0.5, 0.6], [0.5, 0.5], "the prior sums to 1.1, it is not a distribution", id="prior-sum"),
    pytest.param([1.5, -0.5], [0.5, 0.5], "a negative prior probability (even if the sum is 1)", id="negative-prior"),
    pytest.param([0.5, 0.5], [0.5, 1.2], "a likelihood of 1.2 is not a probability", id="likelihood-above-1"),
])
def test_bayes_posterior_rejects_bad_inputs(by, prior, likelihood, why):
    assert_raises_value_error(by.bayes_posterior, prior, likelihood, why=why)


# ------------------------------------------------------------------ update_discrete (4.16)
def test_update_discrete_docstring_example(by):
    likelihoods = [[0.5, 0.5], [1 / 3, 2 / 3]]   # columns: tails, heads
    expected = sequential_oracle([0.5, 0.5], likelihoods, [1, 1, 0])
    assert_close(by.update_discrete([0.5, 0.5], likelihoods, [1, 1, 0]), expected[-1],
                 msg="final posterior after heads, heads, tails")
    assert_close(by.update_discrete([0.5, 0.5], likelihoods, [1, 1, 0], return_history=True), expected,
                 msg="with return_history=True: row 0 is the prior, row k the posterior after k observations")


@pytest.mark.parametrize("n_outcomes", [2, 3, 6])
def test_update_discrete_matches_the_batch_formula_in_log_space(by, n_outcomes):
    rng = np.random.default_rng(10 + n_outcomes)
    for _ in range(15):
        n_hyp = int(rng.integers(1, 9))
        prior = rng.dirichlet(np.ones(n_hyp))
        table = rng.uniform(0.05, 1.0, size=(n_hyp, n_outcomes))
        table /= table.sum(axis=1, keepdims=True)   # each row: a distribution over the outcomes
        observations = rng.integers(0, n_outcomes, size=int(rng.integers(0, 40)))
        result = by.update_discrete(prior, table, observations)
        assert_close(result, batch_log_oracle(prior, table, observations),
                     msg="P(H_i | o_1..o_T) is proportional to P(H_i) x the product of the P(o_t | H_i)",
                     data=f"prior={listed(prior)}, observations={listed(observations)}")


def test_update_discrete_history_has_the_prior_then_one_row_per_observation(by):
    rng = np.random.default_rng(20)
    for _ in range(10):
        n_hyp, n_out = int(rng.integers(2, 6)), int(rng.integers(2, 4))
        prior = rng.dirichlet(np.ones(n_hyp))
        table = rng.uniform(0.05, 1.0, size=(n_hyp, n_out))
        table /= table.sum(axis=1, keepdims=True)
        observations = rng.integers(0, n_out, size=int(rng.integers(1, 12)))
        result = np.asarray(by.update_discrete(prior, table, observations, return_history=True))
        expected = sequential_oracle(prior, table, observations)
        assert_close(result, expected, msg="row 0 = the prior; row k = the posterior after the first k observations",
                     data=f"prior={listed(prior)}, observations={listed(observations)}")


def test_update_discrete_returns_the_final_posterior_by_default(by):
    result = by.update_discrete([0.5, 0.5], [[0.5, 0.5], [0.2, 0.8]], [1, 0, 1, 1])
    assert np.shape(result) == (2,), (f"without return_history, return only the final posterior (shape (2,)), "
                                      f"got shape {np.shape(result)}")
    assert_distribution(np.asarray(result, dtype=float), "update_discrete", 2)


def test_update_discrete_order_of_the_observations_does_not_matter(by):
    rng = np.random.default_rng(30)
    prior = rng.dirichlet(np.ones(5))
    table = rng.uniform(0.1, 1.0, size=(5, 3))
    table /= table.sum(axis=1, keepdims=True)
    observations = rng.integers(0, 3, size=60)
    first = by.update_discrete(prior, table, observations)
    for _ in range(3):
        shuffled = rng.permutation(observations)
        assert_close(by.update_discrete(prior, table, shuffled), first, rtol=1e-8,
                     msg="the same observations in another order must give the same final posterior",
                     data=f"observations={listed(observations)}")


def test_update_discrete_does_not_underflow_after_thousands_of_observations(by):
    # 5 000 flips of a coin of bias 0.6, hypotheses 0.5 and 0.6: the raw products would underflow to 0
    flips = np.random.default_rng(40).random(5000) < 0.6
    observations = flips.astype(int)
    table = [[0.5, 0.5], [0.4, 0.6]]           # columns: tails, heads
    result = by.update_discrete([0.5, 0.5], table, observations)
    assert not np.isnan(np.asarray(result, dtype=float)).any(), (
        "got NaN after 5 000 observations: normalize the posterior at EVERY step (it becomes the next prior), "
        "do not multiply all the likelihoods first")
    assert_close(result, batch_log_oracle([0.5, 0.5], table, observations), rtol=1e-6, atol=1e-12,
                 msg="after 5 000 observations, normalize at every step")


def test_update_discrete_with_no_observation_returns_the_prior(by):
    prior = [0.2, 0.3, 0.5]
    table = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    assert_close(by.update_discrete(prior, table, []), prior, msg="no observation: the posterior is the prior")
    history = np.asarray(by.update_discrete(prior, table, [], return_history=True))
    assert history.shape == (1, 3), (f"no observation: the history is the prior alone, shape (1, 3), "
                                     f"got shape {history.shape}")


def test_update_discrete_a_zero_prior_stays_zero(by):
    result = by.update_discrete([0.0, 0.5, 0.5], [[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]], [1, 1, 0, 1])
    assert np.asarray(result)[0] == 0, f"a hypothesis with prior 0 must stay at 0, got {_short(result)}"


def test_update_discrete_accepts_lists_tuples_and_arrays(by):
    table = [[0.5, 0.5], [1 / 3, 2 / 3]]
    expected = sequential_oracle([0.5, 0.5], table, [1, 1, 0])[-1]
    for observations in ([1, 1, 0], (1, 1, 0), np.array([1, 1, 0]), np.array([1, 1, 0], dtype=np.int32)):
        assert_close(by.update_discrete([0.5, 0.5], np.array(table), observations), expected,
                     msg=f"observations of type {type(observations).__name__} "
                         f"(dtype {np.asarray(observations).dtype}): convert them with np.asarray")


def test_update_discrete_does_not_modify_its_inputs(by):
    prior = np.array([0.5, 0.25, 0.25])
    table = np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
    by.update_discrete(prior, table, [1, 0, 1])
    assert listed(prior) == [0.5, 0.25, 0.25], f"prior was modified in place: {listed(prior)} (work on a copy)"
    assert listed(table) == [[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]], f"likelihoods was modified in place: {listed(table)}"


@pytest.mark.parametrize("prior, table, observations, why", [
    pytest.param([0.5, 0.5], [[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]], [1], "3 rows of likelihoods for 2 hypotheses",
                 id="rows-mismatch"),
    pytest.param([0.5, 0.5], [0.5, 0.8], [1], "likelihoods must be a 2-D table (n_hypotheses, n_outcomes)",
                 id="likelihoods-1d"),
    pytest.param([0.5, 0.5], [[0.5, 0.5], [0.2, 0.8]], [0, 2], "the outcome 2 does not exist (2 outcomes: 0 and 1)",
                 id="index-too-large"),
    pytest.param([0.5, 0.5], [[0.5, 0.5], [0.2, 0.8]], [1, -1],
                 "a negative outcome index (NumPy would silently read the last column)", id="negative-index"),
    pytest.param([0.5, 0.5], [[0.5, 0.5], [0.2, 0.8]], [1, 1.5], "1.5 is not an outcome index", id="non-integer"),
    pytest.param([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [0, 1],
                 "the 2nd observation is impossible under every hypothesis still possible (evidence 0)",
                 id="evidence-0"),
    pytest.param([0.5, 0.6], [[0.5, 0.5], [0.2, 0.8]], [1], "the prior sums to 1.1", id="prior-sum"),
    pytest.param([0.5, 0.5], [[0.5, 0.5], [0.2, 1.8]], [1], "a likelihood of 1.8 is not a probability",
                 id="likelihood-above-1"),
    pytest.param([0.5, 0.5], [[0.5, 0.5], [0.2, 0.8]], [[1, 0]], "observations must be a 1-D sequence",
                 id="observations-2d"),
])
def test_update_discrete_rejects_bad_inputs(by, prior, table, observations, why):
    assert_raises_value_error(by.update_discrete, prior, table, observations, why=why)


# ------------------------------------------------------------------ coin_bias_posterior (4.24)
def test_coin_bias_posterior_docstring_example(by):
    assert_close(by.coin_bias_posterior([1, 1, 0], np.linspace(0, 1, 5)), [0, 0.15, 0.4, 0.45, 0],
                 msg="2 heads and 1 tail on the grid 0, 0.25, 0.5, 0.75, 1 (uniform prior)")


@pytest.mark.parametrize("n_flips", [0, 1, 10, 200, 3000])
def test_coin_bias_posterior_matches_scipy_beta(by, n_flips):
    rng = np.random.default_rng(n_flips)
    grid = np.linspace(0.001, 0.999, 999)
    for _ in range(5):
        flips = (rng.random(n_flips) < rng.uniform(0.05, 0.95)).astype(int)
        heads = int(flips.sum())
        pdf = stats.beta(heads + 1, n_flips - heads + 1).pdf(grid)
        result = by.coin_bias_posterior(flips, grid)
        assert_distribution(np.asarray(result, dtype=float), "coin_bias_posterior", len(grid))
        assert_close(result, pdf / pdf.sum(), rtol=1e-7, atol=1e-12,
                     msg=f"uniform prior, {heads} heads and {n_flips - heads} tails: the posterior is the Beta("
                         f"{heads + 1}, {n_flips - heads + 1}) density, normalized on the grid")


def test_coin_bias_posterior_with_a_prior_matches_sequential_bayes(by):
    rng = np.random.default_rng(50)
    for _ in range(10):
        grid = np.sort(rng.random(int(rng.integers(2, 12))))
        prior = rng.dirichlet(np.ones(len(grid)))
        flips = (rng.random(int(rng.integers(0, 25))) < 0.4).astype(int)
        table = np.column_stack([1 - grid, grid])           # columns: tails, heads
        expected = sequential_oracle(prior, table, flips)[-1]
        assert_close(by.coin_bias_posterior(flips, grid, prior=prior), expected,
                     msg="posterior proportional to prior x theta^heads x (1 - theta)^tails",
                     data=f"grid={listed(grid)}, prior={listed(prior)}, flips={listed(flips)}")


@pytest.mark.parametrize("flips, expected, why", [
    pytest.param([1, 1, 1], ["0", "1/9", "8/9"],
                 "3 heads and no tail: theta = 0 is ruled out, and theta = 1 keeps (1 - theta)^0 = 1 "
                 "(0 x log 0 must not give NaN)", id="only-heads"),
    pytest.param([0, 0, 0], ["8/9", "1/9", "0"],
                 "3 tails and no head: theta^0 = 1 even for theta = 0 (0 x log 0 must not give NaN)", id="only-tails"),
    pytest.param([1, 0], ["0", "1", "0"], "one head and one tail rule out both ends", id="both-ends-out"),
    pytest.param([], ["1/3", "1/3", "1/3"], "no flip: the posterior is the prior", id="no-flip"),
])
def test_coin_bias_posterior_at_the_ends_of_the_grid(by, flips, expected, why):
    result = by.coin_bias_posterior(flips, [0.0, 0.5, 1.0])
    assert_close(result, [float(Fraction(p)) for p in expected], msg=why, data="grid=[0.0, 0.5, 1.0], uniform prior")


def test_coin_bias_posterior_does_not_underflow_after_thousands_of_flips(by):
    rng = np.random.default_rng(60)
    flips = (rng.random(20_000) < 0.3).astype(int)
    grid = np.linspace(0, 1, 501)
    result = by.coin_bias_posterior(flips, grid)
    assert not np.isnan(np.asarray(result, dtype=float)).any(), (
        "got NaN after 20 000 flips: theta^h underflows to 0 (and 0 / 0 = NaN); add LOG-probabilities, "
        "subtract their maximum, then take exp")
    assert_close(result, coin_oracle(flips, grid), rtol=1e-6, atol=1e-12,
                 msg="after 20 000 flips: compute in log space, subtract the maximum before exp, then normalize")


def test_coin_bias_posterior_a_zero_prior_stays_zero(by):
    grid = [0.2, 0.4, 0.6, 0.8]
    result = by.coin_bias_posterior([1, 1, 0, 1], grid, prior=[0.5, 0.0, 0.25, 0.25])
    assert np.asarray(result)[1] == 0, f"a hypothesis with prior 0 must keep posterior 0, got {_short(result)}"
    assert_close(result, coin_oracle([1, 1, 0, 1], grid, [0.5, 0.0, 0.25, 0.25]),
                 msg="posterior proportional to prior x theta^heads x (1 - theta)^tails")


def test_coin_bias_posterior_depends_only_on_the_counts(by):
    grid = np.linspace(0, 1, 11)
    a = by.coin_bias_posterior([1, 1, 0, 0, 0, 1, 0], grid)
    b = by.coin_bias_posterior([0, 0, 0, 0, 1, 1, 1], grid)
    assert_close(b, a, msg="the same numbers of heads and tails in another order give the same posterior")


def test_coin_bias_posterior_accepts_lists_arrays_and_booleans(by):
    grid = [0.25, 0.5, 0.75]
    expected = coin_oracle([1, 0, 1], grid)
    for flips in ([1, 0, 1], (1, 0, 1), np.array([1, 0, 1]), np.array([True, False, True])):
        assert_close(by.coin_bias_posterior(flips, grid), expected,
                     msg=f"flips of type {type(flips).__name__} (dtype {np.asarray(flips).dtype}): "
                         f"convert them with np.asarray")


@pytest.mark.parametrize("flips, grid, prior, why", [
    pytest.param([1, 2, 0], [0.5, 0.6], None, "a flip of 2 (only 0 and 1 are allowed)", id="flip-2"),
    pytest.param([1, 0.5], [0.5, 0.6], None, "a flip of 0.5", id="flip-half"),
    pytest.param([1, 0], [0.5, 1.2], None, "a grid value of 1.2 is not a bias", id="grid-above-1"),
    pytest.param([1, 0], [-0.1, 0.5], None, "a negative grid value", id="grid-negative"),
    pytest.param([1, 0], [np.nan, 0.5], None, "a NaN grid value", id="grid-nan"),
    pytest.param([1, 0], [0.4, 0.6], [0.2, 0.3, 0.5], "a prior with 3 values for a grid of 2", id="prior-shape"),
    pytest.param([1, 0], [0.4, 0.6], [0.5, 0.6], "the prior sums to 1.1", id="prior-sum"),
    pytest.param([1, 0], [0.4, 0.6], [1.5, -0.5], "a negative prior probability", id="prior-negative"),
    pytest.param([1, 0], [0.0, 1.0], None, "one head and one tail rule out both hypotheses 0 and 1",
                 id="every-hypothesis-ruled-out"),
    pytest.param([1, 0], [0.0, 0.5, 1.0], [0.5, 0.0, 0.5],
                 "the only possible hypothesis (0.5) has prior 0", id="only-possible-has-prior-0"),
])
def test_coin_bias_posterior_rejects_bad_inputs(by, flips, grid, prior, why):
    assert_raises_value_error(by.coin_bias_posterior, flips, grid, prior=prior, why=why)


# ------------------------------------------------------------------ credible_interval (4.25)
def test_credible_interval_docstring_examples(by):
    grid, posterior = np.linspace(0, 1, 5), [0.0, 0.15, 0.4, 0.45, 0.0]
    result = by.credible_interval(grid, posterior)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"credible_interval must return a tuple (low, high), got {type(result).__name__}")
    assert_close(result, (0.25, 0.75), msg="mass 0.95: levels 0.025 and 0.975 of the cumulative probability")
    assert_close(by.credible_interval(grid, posterior, mass=0.5), (0.5, 0.75),
                 msg="mass 0.5: levels 0.25 and 0.75 of the cumulative probability")


def test_credible_interval_returns_python_floats(by):
    low, high = by.credible_interval(np.array([1, 2, 3, 4]), [0.25, 0.25, 0.25, 0.25], mass=0.5)
    assert_python_float(low, "credible_interval (low)")
    assert_python_float(high, "credible_interval (high)")


@pytest.mark.parametrize("mass, expected", [
    pytest.param(0.5, (1.0, 3.0), id="levels-exactly-reached"),    # levels 0.25 and 0.75 = cumulative sums
    pytest.param(0.25, (2.0, 3.0), id="levels-in-between"),        # levels 0.375 and 0.625
])
def test_credible_interval_reached_means_greater_or_equal(by, mass, expected):
    # cumulative probabilities 0.25, 0.5, 0.75, 1 (exact in binary): a level is "reached" when cdf >= level
    result = by.credible_interval([1.0, 2.0, 3.0, 4.0], [0.25, 0.25, 0.25, 0.25], mass=mass)
    assert_close(result, expected, msg=f"mass {mass}: the first grid value whose cumulative probability is >= "
                                       f"each level (reaching a level exactly counts)",
                 data="grid=[1, 2, 3, 4], posterior=[0.25, 0.25, 0.25, 0.25]")


@pytest.mark.parametrize("mass", [0.5, 0.8, 0.95, 0.99])
def test_credible_interval_matches_scipy_beta_within_one_grid_step(by, mass):
    rng = np.random.default_rng(int(mass * 100))
    grid = np.linspace(0, 1, 2001)
    step = grid[1] - grid[0]
    for _ in range(8):
        heads, tails = int(rng.integers(0, 60)), int(rng.integers(0, 60))
        pdf = stats.beta(heads + 1, tails + 1).pdf(grid)
        low, high = by.credible_interval(grid, pdf / pdf.sum(), mass=mass)
        want = stats.beta(heads + 1, tails + 1).ppf([(1 - mass) / 2, (1 + mass) / 2])
        assert abs(low - want[0]) <= 1.01 * step and abs(high - want[1]) <= 1.01 * step, (
            f"expected about ({want[0]:.4f}, {want[1]:.4f}) (scipy.stats.beta, within one grid step), "
            f"got ({low:.4f}, {high:.4f}) for {heads} heads and {tails} tails, mass {mass}")


def test_credible_interval_contains_at_least_the_mass(by):
    rng = np.random.default_rng(70)
    for _ in range(30):
        n = int(rng.integers(2, 40))
        grid = np.cumsum(rng.uniform(0.1, 1.0, size=n))          # strictly increasing, uneven steps
        posterior = rng.dirichlet(np.ones(n) * 0.5)
        mass = float(rng.uniform(0.05, 0.98))
        low, high = by.credible_interval(grid, posterior, mass=mass)
        assert low <= high, f"low ({low}) must not exceed high ({high})"
        inside = posterior[(grid >= low) & (grid <= high)].sum()
        assert inside >= mass - 1e-12, (
            f"the interval must hold at least the mass {mass:.3f} of probability, it holds {inside:.3f} "
            f"(grid={listed(grid)}, posterior={listed(posterior)})")


def test_credible_interval_widens_when_the_mass_grows(by):
    grid = np.linspace(0, 1, 101)
    pdf = stats.beta(8, 5).pdf(grid)
    posterior = pdf / pdf.sum()
    previous = None
    for mass in (0.2, 0.5, 0.8, 0.95, 0.99):
        low, high = by.credible_interval(grid, posterior, mass=mass)
        if previous:
            assert low <= previous[0] and high >= previous[1], (
                f"mass {mass}: the interval ({low}, {high}) must contain the one of a smaller mass {previous}")
        previous = (low, high)


def test_credible_interval_of_a_point_mass(by):
    assert_close(by.credible_interval([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 1.0, 0.0]), (0.3, 0.3),
                 msg="all the probability on 0.3: the interval is (0.3, 0.3)")


@pytest.mark.parametrize("grid, posterior, mass, why", [
    pytest.param([0.1, 0.2, 0.2, 0.4], [0.25] * 4, 0.9, "the grid repeats 0.2 (not strictly increasing)",
                 id="repeated-value"),
    pytest.param([0.4, 0.3, 0.2, 0.1], [0.25] * 4, 0.9, "a decreasing grid", id="decreasing"),
    pytest.param([0.1, 0.2, 0.3], [0.25] * 4, 0.9, "grid and posterior of different lengths", id="different-lengths"),
    pytest.param([0.1, 0.2, 0.3, 0.4], [0.3] * 4, 0.9, "the posterior sums to 1.2", id="posterior-sum"),
    pytest.param([0.1, 0.2, 0.3, 0.4], [0.5, 0.75, -0.25, 0.0], 0.9, "a negative posterior probability",
                 id="posterior-negative"),
    pytest.param([0.1, 0.2, 0.3, 0.4], [0.25] * 4, 0.0, "mass 0 (it must be strictly between 0 and 1)",
                 id="mass-0"),
    pytest.param([0.1, 0.2, 0.3, 0.4], [0.25] * 4, 1.0, "mass 1 (it must be strictly between 0 and 1)",
                 id="mass-1"),
    pytest.param([0.1, 0.2, 0.3, 0.4], [0.25] * 4, 95, "mass 95: a percentage instead of a probability",
                 id="mass-percent"),
    pytest.param([0.1, 0.2, 0.3, 0.4], [0.25] * 4, -0.5, "a negative mass", id="mass-negative"),
])
def test_credible_interval_rejects_bad_inputs(by, grid, posterior, mass, why):
    assert_raises_value_error(by.credible_interval, grid, posterior, mass=mass, why=why)
