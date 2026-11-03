"""Explainability and fairness — mylearn, chapter B6 (Explainability, fairness and ethics).

Model explanation and fairness indicators, in NumPy: exact (interventional) Shapley values
by enumerating the coalitions, integrated gradients from a gradient function, the Grad-CAM
combination of activations and gradients already extracted (the hooks stay on the PyTorch
side), and per-group rates of a binary classifier with two fairness gaps.

Reference implementation.
"""

from __future__ import annotations

import math
import numbers
import operator
from typing import Callable, Hashable

import numpy as np
from numpy.typing import ArrayLike

# Your own function from chapter 3.
from .metrics import confusion_matrix

_IG_METHODS = ("left", "right", "midpoint", "trapezoid")
_ROWS_PER_CALL = 1 << 16          # rows given to f in one call (memory bound of shapley_values)


def _integer(value, name: str) -> int:
    """An integer argument (Python or NumPy integer; bool refused)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got a bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None


def _float_array(values, name: str) -> np.ndarray:
    """A new float64 array (a copy: the caller's data is never modified)."""
    try:
        return np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric array") from None


def _binary(values, name: str) -> np.ndarray:
    """A 1-D array of 0/1 values (ints, floats or bools), as int64; NaN and other values are refused."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain only 0 and 1, got values of dtype {arr.dtype}")
    ok = (arr == 0) | (arr == 1)          # NaN is neither 0 nor 1
    if not ok.all():
        bad = arr[~ok][0]
        raise ValueError(f"{name} must contain only 0 and 1, got {bad!r}")
    return arr.astype(np.int64)


def _groups(groups, n: int, names: str) -> tuple[list, np.ndarray]:
    """The sorted groups (as np.unique, Python scalars) and the group index of each example."""
    g = np.asarray(groups)
    if g.ndim != 1:
        raise ValueError(f"groups must be 1-D, got shape {g.shape}")
    if len(g) != n:
        raise ValueError(f"{names} and groups must have the same length, got {n} and {len(g)}")
    uniques, inverse = np.unique(g, return_inverse=True)
    keys = [u.item() if isinstance(u, np.generic) else u for u in uniques]
    return keys, inverse.reshape(-1)


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def shapley_values(
    f: Callable[[np.ndarray], np.ndarray],
    x: ArrayLike,
    background: ArrayLike,
    max_features: int = 12,
) -> tuple[np.ndarray, float]:
    """Exact interventional Shapley values of the model output ``f`` at ``x``.

    A coalition ``S`` of features is worth ``v(S)`` = the mean of ``f`` over the background
    rows in which the features of ``S`` are replaced by those of ``x``. Then
    ``phi_i = sum over S not containing i of |S|! (M - |S| - 1)! / M! (v(S + {i}) - v(S))``,
    enumerating the ``2**M`` coalitions.

    Parameters
    ----------
    f : callable
        Vectorised model output, ``(n, M) -> (n,)``, e.g.
        ``lambda X: model.predict_proba(X)[:, 1]``. It may be called several times.
    x : array-like of shape (M,)
        Instance to explain.
    background : array-like of shape (B, M)
        Reference rows (e.g. a sample of the training set).
    max_features : int, default=12
        Guard against a combinatorial explosion: ``2**M * B`` predictions are needed.

    Returns
    -------
    phi : np.ndarray of shape (M,)
        Shapley value of each feature.
    base_value : float
        ``v(empty set)``, the mean of ``f`` over the background.
        Efficiency: ``phi.sum() + base_value == f(x)`` (up to rounding).

    Raises
    ------
    ValueError
        If ``M > max_features``, ``x`` is not 1-D, or ``background`` is not 2-D with
        ``M`` columns.

    Notes
    -----
    Tested against ``shap.explainers.Exact(f, shap.maskers.Independent(background,
    max_samples=len(background)))``, which gives the same values, and with the properties
    of efficiency, symmetry, null player, and ``phi_i = w_i (x_i - mean_i)`` for a linear
    model.

    Examples
    --------
    >>> w = np.array([2.0, -1.0, 0.5])
    >>> background = np.array([[0., 0., 0.], [2., 2., 2.]])  # feature means: 1, 1, 1
    >>> phi, base = shapley_values(lambda X: X @ w, [3., 1., 1.], background)
    >>> phi, base
    (array([4., 0., 0.]), 1.5)
    >>> phi, base = shapley_values(lambda X: X[:, 0] * X[:, 1], [1., 1.], [[0., 0.]])
    >>> phi  # the interaction is shared equally
    array([0.5, 0.5])
    """
    x = _float_array(x, "x")
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D (one instance), got shape {x.shape}")
    M = x.shape[0]
    bg = _float_array(background, "background")
    if bg.ndim != 2 or bg.shape[1] != M:
        raise ValueError(f"background must be 2-D with {M} columns (the features of x), got shape {bg.shape}")
    if bg.shape[0] == 0:
        raise ValueError("background must have at least one row")
    max_features = _integer(max_features, "max_features")
    if M > max_features:
        raise ValueError(f"{M} features > max_features = {max_features}: 2**{M} coalitions x {bg.shape[0]} rows "
                         "would be needed; raise max_features or use an approximation (KernelSHAP, TreeSHAP)")
    if np.isnan(x).any() or np.isnan(bg).any():
        raise ValueError("x and background contain NaN: remove the missing values first")
    B = bg.shape[0]
    n_coalitions = 1 << M
    codes = np.arange(n_coalitions)
    masks = ((codes[:, None] >> np.arange(M)) & 1).astype(bool)      # bit i of the code = feature i in S
    v = np.empty(n_coalitions)
    step = max(1, _ROWS_PER_CALL // B)
    for start in range(0, n_coalitions, step):
        block = masks[start:start + step]
        rows = np.where(block[:, None, :], x, bg[None, :, :]).reshape(-1, M)
        out = np.asarray(f(rows), dtype=np.float64)
        if out.shape != (rows.shape[0],):
            raise ValueError(f"f must return one value per row, shape ({rows.shape[0]},), got shape {out.shape}")
        v[start:start + len(block)] = out.reshape(len(block), B).mean(axis=1)
    sizes = masks.sum(axis=1)
    weight = np.array([math.factorial(k) * math.factorial(M - k - 1) / math.factorial(M) for k in range(M)])
    phi = np.empty(M)
    for i in range(M):
        without = codes[(codes >> i) & 1 == 0]
        phi[i] = np.sum(weight[sizes[without]] * (v[without | (1 << i)] - v[without]))
    return phi, float(v[0])


def integrated_gradients(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x: ArrayLike,
    baseline: ArrayLike | None = None,
    n_steps: int = 50,
    method: str = "trapezoid",
) -> np.ndarray:
    """Integrated gradients ``(x - x') * integral from 0 to 1 of grad f(x' + a (x - x')) da``.

    The integral over ``a`` is approximated with ``n_steps`` intervals of width
    ``1 / n_steps``: 'left' uses the points ``k / n_steps`` for ``k = 0 .. n_steps - 1``,
    'right' ``k = 1 .. n_steps``, 'midpoint' ``(k + 0.5) / n_steps`` for
    ``k = 0 .. n_steps - 1`` (weights ``1 / n_steps``), and 'trapezoid' the
    ``n_steps + 1`` points ``k / n_steps`` (weights ``1 / n_steps``, halved at both ends).

    Parameters
    ----------
    grad_fn : callable
        Batched gradient of the model output, ``(n, *x.shape) -> (n, *x.shape)``. It may
        be called once on all the points or on several smaller batches.
    x : array-like
        Input to explain.
    baseline : array-like with the shape of x, or None, default=None
        Reference input ``x'``; zeros if None.
    n_steps : int, default=50
        Number of intervals.
    method : {'left', 'right', 'midpoint', 'trapezoid'}, default='trapezoid'
        Integration rule.

    Returns
    -------
    np.ndarray
        Attributions, with the shape of ``x``. Completeness: their sum is close to
        ``f(x) - f(x')``, and closer as ``n_steps`` grows.

    Raises
    ------
    ValueError
        If ``n_steps < 1``, ``method`` is unknown, or ``baseline`` does not have the shape
        of ``x``.

    Notes
    -----
    Tested against closed forms for linear and quadratic models, and with the
    completeness property. ``captum.attr.IntegratedGradients`` (if installed) uses the
    same rules, but its ``n_steps`` counts points: 'trapezoid' here equals captum's
    'riemann_trapezoid' with ``n_steps + 1``.

    Examples
    --------
    For ``f(x) = x0**2 + 3 x1`` (gradient ``(2 x0, 3)``), with ``x = (1, 1)`` and a zero
    baseline, the exact attributions are ``(1, 3)``:

    >>> grad_fn = lambda X: np.stack([2 * X[:, 0], np.full(len(X), 3.0)], axis=1)
    >>> integrated_gradients(grad_fn, [1.0, 1.0], n_steps=4, method="left")
    array([0.75, 3.  ])
    >>> integrated_gradients(grad_fn, [1.0, 1.0], n_steps=4, method="trapezoid")
    array([1., 3.])
    """
    x = _float_array(x, "x")
    if baseline is None:
        base = np.zeros_like(x)
    else:
        base = _float_array(baseline, "baseline")
        if base.shape != x.shape:
            raise ValueError(f"baseline must have the shape of x {x.shape}, got shape {base.shape}")
    n_steps = _integer(n_steps, "n_steps")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if method not in _IG_METHODS:
        raise ValueError(f"method must be one of {_IG_METHODS}, got {method!r}")
    k = np.arange(n_steps + 1, dtype=np.float64)
    if method == "left":
        alphas, weights = k[:-1] / n_steps, np.full(n_steps, 1.0 / n_steps)
    elif method == "right":
        alphas, weights = k[1:] / n_steps, np.full(n_steps, 1.0 / n_steps)
    elif method == "midpoint":
        alphas, weights = (k[:-1] + 0.5) / n_steps, np.full(n_steps, 1.0 / n_steps)
    else:
        alphas, weights = k / n_steps, np.full(n_steps + 1, 1.0 / n_steps)
        weights[0] /= 2
        weights[-1] /= 2
    diff = x - base
    points = base + alphas.reshape((-1,) + (1,) * x.ndim) * diff
    grads = np.asarray(grad_fn(points), dtype=np.float64)
    if grads.shape != points.shape:
        raise ValueError(f"grad_fn must return the shape of its input {points.shape}, got shape {grads.shape}")
    return diff * np.tensordot(weights, grads, axes=1)


def grad_cam(
    activations: ArrayLike, gradients: ArrayLike, relu: bool = True, normalize: bool = True
) -> np.ndarray:
    """Grad-CAM map from the activations of a layer and the gradients of a class score.

    Channel weights ``w_c`` = spatial mean of the gradients of channel ``c``; map =
    ``ReLU(sum over c of w_c A_c)``. With ``normalize=True``, each map is divided by its
    largest absolute value, so it lies in [0, 1] (in [-1, 1] if ``relu=False``); a map
    that is all zero stays zero.

    Parameters
    ----------
    activations : array-like of shape (N, C, H, W)
        Feature maps ``A`` of the chosen layer (usually the last convolutional one).
    gradients : array-like of shape (N, C, H, W)
        Gradients of the target class score with respect to these feature maps.
    relu : bool, default=True
        Keep only the positive evidence.
    normalize : bool, default=True
        Rescale each map as described above.

    Returns
    -------
    np.ndarray of shape (N, H, W)
        One map per image, at the resolution of the layer (the caller upsamples it to
        the image size).

    Raises
    ------
    ValueError
        If the two arrays are not 4-D with the same shape.

    Notes
    -----
    Tested against an explicit ``torch.einsum`` computation, and with properties: with
    ``normalize=True`` the map does not change when the gradients are multiplied by a
    positive number; the map is zero when the gradients are zero.

    Examples
    --------
    >>> A = np.array([[[[1., 2.], [0., 1.]], [[0., 1.], [3., 0.]]]])  # N=1, C=2, H=W=2
    >>> G = np.array([[[[1., 1.], [1., 1.]], [[-1., -1.], [-1., -1.]]]])
    >>> grad_cam(A, G, relu=False, normalize=False)  # channel 0 minus channel 1
    array([[[ 1.,  1.],
            [-3.,  1.]]])
    >>> grad_cam(A, G)
    array([[[1., 1.],
            [0., 1.]]])
    """
    A = _float_array(activations, "activations")
    G = _float_array(gradients, "gradients")
    if A.ndim != 4 or G.shape != A.shape:
        raise ValueError("activations and gradients must be 4-D (N, C, H, W) with the same shape, "
                         f"got shapes {A.shape} and {G.shape}")
    if A.shape[2] == 0 or A.shape[3] == 0:
        raise ValueError(f"the maps must have at least one pixel, got H, W = {A.shape[2:]}")
    weights = G.mean(axis=(2, 3))                       # (N, C): spatial mean of the gradients
    cam = np.einsum("nc,nchw->nhw", weights, A)
    if relu:
        cam = np.maximum(cam, 0.0)
    if normalize and cam.shape[0] > 0:
        peak = np.abs(cam).max(axis=(1, 2), keepdims=True)
        cam = cam / np.where(peak > 0, peak, 1.0)       # an all-zero map stays zero (NaN propagates)
    return cam


def group_rates(
    y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike
) -> dict[Hashable, dict[str, float]]:
    """Per-group rates of a binary classifier.

    For each group ``g``: ``'n'`` (number of examples), ``'base_rate'`` P(y = 1),
    ``'selection_rate'`` P(y_hat = 1), ``'tpr'`` TP / (TP + FN), ``'fpr'`` FP / (FP + TN),
    ``'precision'`` TP / (TP + FP) and ``'accuracy'`` (TP + TN) / n, all computed on the
    examples of ``g`` only. A rate whose denominator is 0 is ``nan``.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True labels in {0, 1}.
    y_pred : array-like of shape (n,)
        Predictions in {0, 1}.
    groups : array-like of shape (n,)
        Group of each example (e.g. the sex or an age bracket).

    Returns
    -------
    dict
        ``{group: {'n', 'base_rate', 'selection_rate', 'tpr', 'fpr', 'precision',
        'accuracy'}}``, with the groups in sorted order (as ``np.unique``). ``'n'`` is an
        int, the rates are floats.

    Raises
    ------
    ValueError
        If the lengths differ or a label or prediction is not 0 or 1.

    Notes
    -----
    Tested against ``sklearn.metrics.confusion_matrix`` computed on each subgroup, and
    ``fairlearn.metrics.MetricFrame`` if fairlearn is installed. Reuses your
    ``confusion_matrix`` (chapter 3).

    Examples
    --------
    >>> y_true = [1, 0, 1, 0, 1, 1, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
    >>> rates = group_rates(y_true, y_pred, groups)
    >>> sorted(rates["a"])
    ['accuracy', 'base_rate', 'fpr', 'n', 'precision', 'selection_rate', 'tpr']
    >>> rates["a"]["n"], rates["a"]["tpr"], rates["a"]["fpr"]
    (4, 0.5, 0.5)
    >>> rates["b"]["n"], rates["b"]["tpr"], rates["b"]["fpr"]
    (4, 1.0, 0.0)
    """
    yt = _binary(y_true, "y_true")
    yp = _binary(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred must have the same length, got {len(yt)} and {len(yp)}")
    keys, inverse = _groups(groups, len(yt), "y_true, y_pred")
    rates: dict[Hashable, dict[str, float]] = {}
    for j, key in enumerate(keys):
        mask = inverse == j
        (tn, fp), (fn, tp) = np.asarray(confusion_matrix(yt[mask], yp[mask], labels=[0, 1])).tolist()
        n = tn + fp + fn + tp
        rates[key] = {
            "n": int(n),
            "base_rate": _ratio(tp + fn, n),
            "selection_rate": _ratio(tp + fp, n),
            "tpr": _ratio(tp, tp + fn),
            "fpr": _ratio(fp, fp + tn),
            "precision": _ratio(tp, tp + fp),
            "accuracy": _ratio(tp + tn, n),
        }
    return rates


def demographic_parity_difference(y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Demographic parity gap: largest minus smallest selection rate P(y_hat = 1 | group).

    Parameters
    ----------
    y_pred : array-like of shape (n,)
        Predictions in {0, 1}.
    groups : array-like of shape (n,)
        Group of each example.

    Returns
    -------
    float
        Gap in [0, 1]; 0 means the same selection rate in every group.

    Raises
    ------
    ValueError
        If the lengths differ, a prediction is not 0 or 1, or there are fewer than two
        groups.

    Notes
    -----
    Tested against hand-computed values, and
    ``fairlearn.metrics.demographic_parity_difference`` if fairlearn is installed.

    Examples
    --------
    >>> demographic_parity_difference([1, 0, 0, 1, 1, 1, 0, 0], ["a"] * 4 + ["b"] * 4)
    0.0
    >>> demographic_parity_difference([1, 1, 1, 0, 1, 0, 0, 0], ["a"] * 4 + ["b"] * 4)
    0.5
    """
    yp = _binary(y_pred, "y_pred")
    keys, inverse = _groups(groups, len(yp), "y_pred")
    if len(keys) < 2:
        raise ValueError(f"at least two groups are needed, got {len(keys)}")
    selection = [yp[inverse == j].mean() for j in range(len(keys))]
    return float(max(selection) - min(selection))


def equalized_odds_difference(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Equalized odds gap: the larger of the TPR gap and the FPR gap between groups.

    ``max(max_g TPR_g - min_g TPR_g, max_g FPR_g - min_g FPR_g)``.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True labels in {0, 1}.
    y_pred : array-like of shape (n,)
        Predictions in {0, 1}.
    groups : array-like of shape (n,)
        Group of each example.

    Returns
    -------
    float
        Gap in [0, 1]; 0 means the same TPR and the same FPR in every group.

    Raises
    ------
    ValueError
        If the lengths differ, a label or prediction is not 0 or 1, there are fewer than
        two groups, or a group has no positive or no negative example (its TPR or FPR is
        undefined).

    Notes
    -----
    Tested against hand-computed values, and
    ``fairlearn.metrics.equalized_odds_difference`` (``agg='worst_case'``) if fairlearn is
    installed.

    Examples
    --------
    Same selection rate in both groups (demographic parity holds), but not the same
    errors:

    >>> y_true = [1, 0, 1, 0, 1, 1, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> equalized_odds_difference(y_true, y_pred, ["a"] * 4 + ["b"] * 4)
    0.5
    """
    rates = group_rates(y_true, y_pred, groups)
    if len(rates) < 2:
        raise ValueError(f"at least two groups are needed, got {len(rates)}")
    for key, r in rates.items():
        if math.isnan(r["tpr"]) or math.isnan(r["fpr"]):
            missing = "positive" if math.isnan(r["tpr"]) else "negative"
            raise ValueError(f"group {key!r} has no {missing} example: its "
                             f"{'TPR' if missing == 'positive' else 'FPR'} is undefined")
    tpr = [r["tpr"] for r in rates.values()]
    fpr = [r["fpr"] for r in rates.values()]
    return float(max(max(tpr) - min(tpr), max(fpr) - min(fpr)))
