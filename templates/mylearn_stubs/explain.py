"""Explainability and fairness — mylearn, chapter B6 (Explainability, fairness and ethics).

Model explanation and fairness indicators, in NumPy: exact (interventional) Shapley values
by enumerating the coalitions, integrated gradients from a gradient function, the Grad-CAM
combination of activations and gradients already extracted (the hooks stay on the PyTorch
side), and per-group rates of a binary classifier with two fairness gaps.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: shap, scikit-learn and PyTorch).
"""

from __future__ import annotations

from typing import Callable, Hashable

import numpy as np
from numpy.typing import ArrayLike

# Your own function from chapter 3.
from .metrics import confusion_matrix


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
    # TODO: compute v(S) once for every coalition (a boolean mask of length M), then
    # combine them with the Shapley weights.
    raise NotImplementedError("shapley_values() is not implemented yet")


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
    # TODO: build the interpolation points and their weights, one call to grad_fn,
    # weighted sum, multiply by (x - baseline).
    raise NotImplementedError("integrated_gradients() is not implemented yet")


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
    # TODO: weights by averaging, weighted sum over the channels, then ReLU and
    # normalisation.
    raise NotImplementedError("grad_cam() is not implemented yet")


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
    # TODO: for each group, a 2 x 2 confusion matrix, then the ratios (nan when the
    # denominator is 0).
    raise NotImplementedError("group_rates() is not implemented yet")


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
    # TODO: one selection rate per group, then max - min.
    raise NotImplementedError("demographic_parity_difference() is not implemented yet")


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
    # TODO: reuse group_rates, check that every TPR and FPR is defined, then the gaps.
    raise NotImplementedError("equalized_odds_difference() is not implemented yet")
