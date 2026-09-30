"""Tests of mylearn.linalg_basics (chapter 0B): oracle tests and properties.

    pytest tests/test_ch00b_linalg_basics.py              # your code (mon_travail/mylearn/linalg_basics.py)
    pytest tests/test_ch00b_linalg_basics.py --impl=ref   # the reference

Oracles: NumPy (np.add, np.subtract, np.multiply, np.dot, np.linalg.norm,
np.transpose, np.eye, @), math.dist, scikit-learn cosine_similarity and SciPy
distance.cosine. Every function must return NEW lists of Python floats and leave
its inputs unchanged.
"""

import copy
import math

import numpy as np
import pytest
from scipy.spatial import distance as scipy_distance
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


@pytest.fixture
def lb(mylearn_module):
    return mylearn_module("linalg_basics")


def random_vectors(seed: int, count: int = 2, integers: bool = False):
    """A few random vectors of the same random length (1 to 8)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 9))
    if integers:
        return [rng.integers(-9, 10, size=n).tolist() for _ in range(count)]
    return [rng.normal(0, 3, size=n).tolist() for _ in range(count)]


def random_matrix(rng, n_rows: int, n_cols: int, integers: bool = False):
    if integers:
        return rng.integers(-5, 6, size=(n_rows, n_cols)).tolist()
    return rng.normal(0, 2, size=(n_rows, n_cols)).tolist()


def assert_float_list(result, expected, rel=1e-12, abs_=1e-12):
    assert isinstance(result, list), f"expected a list, got {type(result).__name__}"
    assert all(isinstance(x, float) for x in result), "the entries must be Python floats"
    np.testing.assert_allclose(result, expected, rtol=rel, atol=abs_)


def assert_float_matrix(result, expected, rel=1e-12, abs_=1e-12):
    assert isinstance(result, list) and all(isinstance(row, list) for row in result), (
        "expected a list of lists (a list of rows)")
    assert all(isinstance(x, float) for row in result for x in row), "the entries must be Python floats"
    assert np.shape(result) == np.shape(expected), f"shape {np.shape(result)} instead of {np.shape(expected)}"
    np.testing.assert_allclose(result, expected, rtol=rel, atol=abs_)


# ------------------------------------------------------------ element-wise operations
@pytest.mark.parametrize("name, oracle", [
    ("vector_add", np.add), ("vector_subtract", np.subtract), ("hadamard", np.multiply),
])
def test_elementwise_operations_match_numpy(lb, name, oracle):
    function = getattr(lb, name)
    for seed in range(30):
        u, v = random_vectors(seed, integers=seed % 2 == 0)
        u_before, v_before = copy.deepcopy(u), copy.deepcopy(v)
        assert_float_list(function(u, v), oracle(u, v))
        assert (u, v) == (u_before, v_before), f"{name} must not modify its inputs"


@pytest.mark.parametrize("name", ["vector_add", "vector_subtract", "hadamard", "dot"])
def test_different_lengths_raise(lb, name):
    with pytest.raises(ValueError):
        getattr(lb, name)([1.0, 2.0], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("name", ["vector_add", "vector_subtract", "hadamard"])
def test_elementwise_operations_return_a_new_list(lb, name):
    u, v = [1.0, 2.0], [0.0, 0.0]
    result = getattr(lb, name)(u, v)
    assert result is not u and result is not v


def test_elementwise_operations_accept_tuples_and_arrays(lb):
    assert lb.vector_add((1, 2), np.array([3.0, 0.5])) == [4.0, 2.5]
    assert lb.vector_subtract(np.array([1, 2]), (3.0, 0.5)) == [-2.0, 1.5]
    assert lb.hadamard((1, 2, 3), (4, 5, 6)) == [4.0, 10.0, 18.0]


def test_scalar_multiply_matches_numpy(lb):
    for seed in range(20):
        (v,) = random_vectors(seed, count=1)
        c = float(np.random.default_rng(seed + 100).normal())
        v_before = list(v)
        assert_float_list(lb.scalar_multiply(c, v), c * np.asarray(v))
        assert v == v_before
    assert lb.scalar_multiply(2, [1, -2]) == [2.0, -4.0]
    assert lb.scalar_multiply(3.0, []) == []


def test_scalar_multiply_gives_floats_for_integer_inputs(lb):
    assert_float_list(lb.scalar_multiply(2, [1, -2]), [2, -4])


def test_hadamard_sums_to_dot_product(lb):
    for seed in range(10):
        u, v = random_vectors(seed)
        assert sum(lb.hadamard(u, v)) == pytest.approx(float(np.dot(u, v)), abs=1e-9)


# ------------------------------------------------------------ dot, norm, distance, cosine
def test_dot_matches_numpy(lb):
    for seed in range(30):
        u, v = random_vectors(seed, integers=seed % 3 == 0)
        u_before = list(u)
        result = lb.dot(u, v)
        assert isinstance(result, float), "dot must return a Python float"
        assert result == pytest.approx(float(np.dot(u, v)), rel=1e-12, abs=1e-12)
        assert u == u_before
    assert lb.dot([1, 2, 3], [4, 5, 6]) == 32.0


def test_dot_of_empty_vectors_is_zero(lb):
    assert lb.dot([], []) == 0.0


@pytest.mark.parametrize("p", [1, 2, 3, math.inf])
def test_norm_matches_numpy(lb, p):
    for seed in range(20):
        (v,) = random_vectors(seed, count=1, integers=seed % 2 == 0)
        result = lb.norm(v, p=p)
        assert isinstance(result, float)
        assert result == pytest.approx(float(np.linalg.norm(v, ord=p)), rel=1e-12)


def test_norm_default_is_euclidean(lb):
    assert lb.norm([3.0, 4.0]) == 5.0
    assert lb.norm([3, -4], p=1) == 7.0
    assert lb.norm([3, -4], p=math.inf) == 4.0


def test_norm_of_empty_vector_is_zero(lb):
    for p in (1, 2, 3, math.inf):
        assert lb.norm([], p=p) == 0.0


@pytest.mark.parametrize("p", [0, 0.5, -1])
def test_norm_rejects_p_below_one(lb, p):
    with pytest.raises(ValueError):
        lb.norm([1.0, 2.0], p=p)


def test_distance_matches_math_dist_and_numpy(lb):
    for seed in range(30):
        u, v = random_vectors(seed)
        result = lb.distance(u, v)
        assert isinstance(result, float)
        assert result == pytest.approx(math.dist(u, v), rel=1e-12, abs=1e-12)
        assert result == pytest.approx(float(np.linalg.norm(np.subtract(u, v))), rel=1e-12, abs=1e-12)
    assert lb.distance([1, 1], [4, 5]) == 5.0


def test_distance_different_lengths_raise(lb):
    with pytest.raises(ValueError):
        lb.distance([1.0], [1.0, 2.0])


def test_cosine_similarity_matches_sklearn_and_scipy(lb):
    for seed in range(30):
        u, v = random_vectors(seed)
        result = lb.cosine_similarity(u, v)
        assert isinstance(result, float)
        assert result == pytest.approx(float(sk_cosine([u], [v])[0, 0]), abs=1e-12)
        assert result == pytest.approx(1 - scipy_distance.cosine(u, v), abs=1e-12)


def test_cosine_similarity_examples_and_range(lb):
    assert lb.cosine_similarity([3.0, 4.0], [4.0, 3.0]) == pytest.approx(0.96)
    assert lb.cosine_similarity([1.0, 0.0], [0.0, 2.0]) == pytest.approx(0.0)
    assert lb.cosine_similarity([3.0, 4.0], [-6.0, -8.0]) == pytest.approx(-1.0)
    for seed in range(20):
        (v,) = random_vectors(seed, count=1)
        same = lb.cosine_similarity(v, [2.5 * x for x in v])
        assert -1.0 <= same <= 1.0, "rounding errors must not push the value outside [-1, 1]"
        assert same == pytest.approx(1.0)


def test_cosine_similarity_ignores_positive_scaling(lb):
    u, v = random_vectors(7)
    assert lb.cosine_similarity([10 * x for x in u], v) == pytest.approx(lb.cosine_similarity(u, v))


def test_cosine_similarity_zero_vector_raises(lb):
    with pytest.raises(ValueError):
        lb.cosine_similarity([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        lb.cosine_similarity([1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        lb.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# ------------------------------------------------------------ matrices
def test_shape_matches_numpy(lb):
    rng = np.random.default_rng(0)
    for _ in range(20):
        m, n = (int(k) for k in rng.integers(1, 7, size=2))
        A = random_matrix(rng, m, n)
        result = lb.shape(A)
        assert isinstance(result, tuple)
        assert result == np.asarray(A).shape
    assert lb.shape([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == (2, 3)
    assert lb.shape([[7]]) == (1, 1)


@pytest.mark.parametrize("bad", [[], [[]], [[1.0, 2.0], [3.0]], [[1.0], [2.0, 3.0]]])
def test_shape_rejects_empty_or_ragged_matrices(lb, bad):
    with pytest.raises(ValueError):
        lb.shape(bad)


def test_transpose_matches_numpy(lb):
    rng = np.random.default_rng(1)
    for _ in range(20):
        m, n = (int(k) for k in rng.integers(1, 7, size=2))
        A = random_matrix(rng, m, n, integers=m % 2 == 0)
        A_before = copy.deepcopy(A)
        result = lb.transpose(A)
        assert_float_matrix(result, np.transpose(A))
        assert A == A_before, "transpose must not modify its input"
        assert lb.transpose(result) == np.asarray(A, dtype=float).tolist()


def test_transpose_examples_and_errors(lb):
    assert lb.transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert lb.transpose([[1, 2, 3]]) == [[1.0], [2.0], [3.0]]
    with pytest.raises(ValueError):
        lb.transpose([[1.0, 2.0], [3.0]])


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_identity_matches_numpy(lb, n):
    result = lb.identity(n)
    assert_float_matrix(result, np.eye(n))
    assert result == np.eye(n).tolist()


def test_identity_rows_are_independent(lb):
    result = lb.identity(3)
    result[0][1] = 5.0
    assert result[1][1] == 1.0 and result[2][1] == 0.0, "each row must be a new list (no [row] * n)"


@pytest.mark.parametrize("n", [0, -2])
def test_identity_rejects_sizes_below_one(lb, n):
    with pytest.raises(ValueError):
        lb.identity(n)


def test_matvec_matches_numpy(lb):
    rng = np.random.default_rng(2)
    for trial in range(25):
        m, n = (int(k) for k in rng.integers(1, 7, size=2))
        A = random_matrix(rng, m, n, integers=trial % 2 == 0)
        v = rng.normal(size=n).tolist()
        A_before, v_before = copy.deepcopy(A), list(v)
        assert_float_list(lb.matvec(A, v), np.asarray(A) @ np.asarray(v), rel=1e-10, abs_=1e-10)
        assert (A, v) == (A_before, v_before), "matvec must not modify its inputs"


def test_matvec_with_the_identity_returns_the_vector(lb):
    v = [1.5, -2.0, 3.25]
    assert lb.matvec(np.eye(3).tolist(), v) == v
    assert lb.matvec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0]) == [3.0, 7.0]


def test_matvec_gives_floats_for_integer_inputs(lb):
    assert_float_list(lb.matvec([[1, 2], [3, 4]], [1, 1]), [3, 7])


def test_matvec_shape_mismatch_message(lb):
    with pytest.raises(ValueError, match=r"\(2, 3\).*\(2,\)"):
        lb.matvec([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        lb.matvec([[1.0, 2.0], [3.0]], [1.0, 2.0])


def test_matmul_matches_numpy(lb):
    rng = np.random.default_rng(3)
    for trial in range(30):
        m, n, p = (int(k) for k in rng.integers(1, 6, size=3))
        integers = trial % 2 == 0
        A, B = random_matrix(rng, m, n, integers), random_matrix(rng, n, p, integers)
        A_before, B_before = copy.deepcopy(A), copy.deepcopy(B)
        assert_float_matrix(lb.matmul(A, B), np.matmul(A, B), rel=1e-10, abs_=1e-10)
        assert (A, B) == (A_before, B_before), "matmul must not modify its inputs"


def test_matmul_examples(lb):
    assert lb.matmul([[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [1.0, 0.0]]) == [[2.0, 1.0], [4.0, 3.0]]
    assert lb.matmul([[1.0, 2.0, 3.0]], [[1.0], [0.0], [2.0]]) == [[7.0]]
    assert lb.matmul([[2, 0], [1, 3]], [[1, -1], [2, 4]]) == [[2.0, -2.0], [7.0, 11.0]]


def test_matmul_properties_against_numpy(lb):
    rng = np.random.default_rng(4)
    A, B = random_matrix(rng, 3, 4), random_matrix(rng, 4, 2)
    assert_float_matrix(lb.matmul(A, np.eye(4).tolist()), A)
    # (AB)^T = B^T A^T, computed with the learner's matmul on NumPy transposes
    expected = (np.asarray(A) @ np.asarray(B)).T
    assert_float_matrix(lb.matmul(np.transpose(B).tolist(), np.transpose(A).tolist()), expected,
                        rel=1e-10, abs_=1e-10)


def test_matmul_shape_mismatch_message(lb):
    with pytest.raises(ValueError, match=r"\(2, 3\).*\(2, 2\)"):
        lb.matmul([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        lb.matmul([[1.0, 2.0], [3.0]], [[1.0], [2.0]])
    with pytest.raises(ValueError):
        lb.matmul([[1.0, 2.0]], [])
