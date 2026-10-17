"""Losses and backpropagation — mylearn, chapter 18 (Backpropagation).

Loss functions that return both their value and their gradient, the backward
pass of a dense layer and of a whole MLP, and a finite-difference gradient
checker. Everything is explicit: the backward pass reads the cache
``(a_prev, z)`` returned by ``mylearn.nn.layers.mlp_forward`` (chapter 16) and
the ``<name>_derivative`` functions of ``mylearn.nn.activations`` (chapter 17).

All losses use the natural logarithm and are averaged over the batch, like
PyTorch's ``reduction="mean"``.

Reference implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .activations import log_softmax, sigmoid  # chapter 17, to reuse below


def _as_float(values: ArrayLike, name: str) -> np.ndarray:
    """A new float64 array (never a view of the caller's data)."""
    return np.array(values, dtype=np.float64, copy=True)


def _check_same_shape(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a} has shape {a.shape} but {name_b} has shape {b.shape}: "
                         "they must be equal (no broadcasting)")


def mse_loss(y_pred: ArrayLike, y_true: ArrayLike) -> tuple[float, np.ndarray]:
    """Compute the mean squared error and its gradient with respect to y_pred.

    L = mean((y_pred - y_true)**2), the mean being taken over ALL elements, so
    dL/dy_pred = 2 * (y_pred - y_true) / y_pred.size.

    Parameters
    ----------
    y_pred : array-like of any shape
        Predictions.
    y_true : array-like of the same shape
        Targets.

    Returns
    -------
    loss : float
        The mean squared error.
    grad : ndarray of the shape of y_pred
        dL/dy_pred, as float64.

    Raises
    ------
    ValueError
        If the two shapes differ (no broadcasting).

    Notes
    -----
    Tested against ``torch.nn.functional.mse_loss`` and its autograd gradient
    (float64).

    Examples
    --------
    >>> mse_loss([1.0, 2.0], [0.0, 0.0])
    (2.5, array([1., 2.]))
    """
    pred = _as_float(y_pred, "y_pred")
    true = _as_float(y_true, "y_true")
    _check_same_shape(pred, true, "y_pred", "y_true")
    if pred.size == 0:
        raise ValueError("y_pred is empty")
    diff = pred - true
    return float(np.mean(diff ** 2)), 2.0 * diff / diff.size


def binary_cross_entropy_with_logits(
    logits: ArrayLike, y_true: ArrayLike
) -> tuple[float, np.ndarray]:
    """Compute the mean binary cross-entropy from logits, and its gradient.

    With p = sigmoid(z): L = -mean(y log p + (1 - y) log(1 - p)). Working on the
    logits z (never on rounded probabilities) keeps the loss finite, even for
    |z| = 1000. The gradient is dL/dz = (sigmoid(z) - y) / n, where n is the
    number of elements.

    Parameters
    ----------
    logits : array-like of shape (n,) or (n, 1)
        Raw scores z (before the sigmoid).
    y_true : array-like of the same shape
        Targets in [0, 1] (usually 0 or 1).

    Returns
    -------
    loss : float
        The mean binary cross-entropy, in nats.
    grad : ndarray of the shape of logits
        dL/dlogits, as float64.

    Raises
    ------
    ValueError
        If the shapes differ or a target is outside [0, 1].

    Notes
    -----
    Tested against ``torch.nn.functional.binary_cross_entropy_with_logits`` and
    its autograd gradient; finite for logits of +/-1000.

    Examples
    --------
    >>> binary_cross_entropy_with_logits([0.0, 0.0], [1.0, 0.0])
    (0.6931471805599453, array([-0.25,  0.25]))
    >>> binary_cross_entropy_with_logits([1000.0, -1000.0], [0.0, 0.0])[0]
    500.0
    """
    z = _as_float(logits, "logits")
    y = _as_float(y_true, "y_true")
    _check_same_shape(z, y, "logits", "y_true")
    if z.size == 0:
        raise ValueError("logits is empty")
    if not np.all((y >= 0.0) & (y <= 1.0)):
        raise ValueError("every target must lie in [0, 1]")
    # log(1 + e^z) - y z, written so that exp only sees non-positive numbers
    losses = np.maximum(z, 0.0) - y * z + np.log1p(np.exp(-np.abs(z)))
    grad = (np.asarray(sigmoid(z), dtype=np.float64) - y) / z.size
    return float(np.mean(losses)), grad


def softmax_cross_entropy(logits: ArrayLike, y_true: ArrayLike) -> tuple[float, np.ndarray]:
    """Compute the mean cross-entropy of softmax(logits), and its gradient.

    L = -mean over samples of sum_k Y[i, k] * log softmax(z_i)_k, where Y holds
    one-hot labels or target probabilities. The gradient has a remarkably
    simple form: dL/dz = (softmax(z) - Y) / n.

    Parameters
    ----------
    logits : array-like of shape (n, K)
        Raw scores (before the softmax), one row per sample.
    y_true : array-like of shape (n,) or (n, K)
        Either integer class labels in ``[0, K)``, or target probabilities
        (each row sums to 1), as ``torch.nn.functional.cross_entropy`` accepts
        both.

    Returns
    -------
    loss : float
        The mean cross-entropy, in nats.
    grad : ndarray of shape (n, K)
        dL/dlogits, as float64.

    Raises
    ------
    ValueError
        If a label is outside ``[0, K)`` or the shapes are inconsistent.

    Notes
    -----
    Tested against ``torch.nn.functional.cross_entropy`` and its autograd
    gradient, with labels and with probabilities; finite for logits of 1e4.

    Examples
    --------
    >>> softmax_cross_entropy([[0.0, 0.0]], [0])
    (0.6931471805599453, array([[-0.5,  0.5]]))
    >>> softmax_cross_entropy([[0.0, 0.0]], [[0.25, 0.75]])
    (0.6931471805599453, array([[ 0.25, -0.25]]))
    """
    z = _as_float(logits, "logits")
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError(f"logits must have shape (n, K) with n, K >= 1, got {z.shape}")
    n, k = z.shape
    target = np.asarray(y_true)
    if target.ndim == 1:
        if target.shape[0] != n:
            raise ValueError(f"{target.shape[0]} labels for {n} rows of logits")
        if target.dtype.kind not in "iu":
            as_float = np.asarray(target, dtype=np.float64)
            if not np.all(np.isfinite(as_float)) or not np.all(as_float == np.round(as_float)):
                raise ValueError("labels of shape (n,) must be integers in [0, K)")
            target = as_float.astype(np.int64)
        if np.any(target < 0) or np.any(target >= k):
            raise ValueError(f"every label must lie in [0, {k}), got values from {target.min()} to {target.max()}")
        Y = np.zeros((n, k))
        Y[np.arange(n), target] = 1.0
    elif target.ndim == 2:
        if target.shape != (n, k):
            raise ValueError(f"target probabilities have shape {target.shape}, logits have shape {(n, k)}")
        Y = np.array(target, dtype=np.float64, copy=True)
    else:
        raise ValueError(f"y_true must have shape (n,) or (n, K), got {target.shape}")
    log_p = np.asarray(log_softmax(z, axis=1), dtype=np.float64)
    loss = -float(np.sum(Y * log_p)) / n
    return loss, (np.exp(log_p) - Y) / n


def dense_backward(
    dout: np.ndarray, x: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the gradients of a dense layer z = x @ W + b.

    Given the upstream gradient dL/dz, return the gradients with respect to
    the input, the weights and the bias. The bias itself is not needed.

    Parameters
    ----------
    dout : ndarray of shape (n, n_out)
        Upstream gradient dL/dz.
    x : ndarray of shape (n, n_in)
        Input of the layer in the forward pass.
    W : ndarray of shape (n_in, n_out)
        Weights of the layer.

    Returns
    -------
    dx : ndarray of shape (n, n_in)
        dL/dx.
    dW : ndarray of shape (n_in, n_out)
        dL/dW (summed over the samples).
    db : ndarray of shape (n_out,)
        dL/db (summed over the samples).

    Raises
    ------
    ValueError
        If the shapes do not match.

    Notes
    -----
    Tested against ``torch.autograd`` through
    ``torch.nn.functional.linear(x, W.T, b)`` (float64).

    Examples
    --------
    >>> dx, dW, db = dense_backward(np.array([[1.0, -1.0]]), np.array([[1.0, 2.0]]),
    ...                             np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> dx
    array([[-1., -1.]])
    >>> dW
    array([[ 1., -1.],
           [ 2., -2.]])
    >>> db
    array([ 1., -1.])
    """
    g = _as_float(dout, "dout")
    a = _as_float(x, "x")
    w = _as_float(W, "W")
    if g.ndim != 2 or a.ndim != 2 or w.ndim != 2:
        raise ValueError(f"dout, x and W must be 2-D, got shapes {g.shape}, {a.shape} and {w.shape}")
    if a.shape[0] != g.shape[0]:
        raise ValueError(f"x has {a.shape[0]} rows but dout has {g.shape[0]}: one row per sample in both")
    if w.shape != (a.shape[1], g.shape[1]):
        raise ValueError(f"W has shape {w.shape}, expected (n_in, n_out) = {(a.shape[1], g.shape[1])}")
    return g @ w.T, a.T @ g, g.sum(axis=0)


def mlp_backward(
    dout: np.ndarray,
    cache: Sequence[tuple[np.ndarray, np.ndarray]],
    params: Sequence[tuple[np.ndarray, np.ndarray]],
    activation_derivative: Callable[[np.ndarray], np.ndarray] | None = None,
    output_activation_derivative: Callable[[np.ndarray], np.ndarray] | None = None,
    return_input_grad: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]] | tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Backpropagate through the whole MLP of ``mylearn.nn.layers.mlp_forward``.

    Walk the layers from the last to the first: turn the gradient with respect
    to the output of a layer into the gradient with respect to its
    pre-activation z (multiply by the derivative evaluated at z), then apply
    ``dense_backward``.

    Parameters
    ----------
    dout : ndarray of shape (n, n_L)
        dL/d(network output), for example the gradient returned by a loss
        computed on the logits.
    cache : sequence of (a_prev, z) tuples
        The cache returned by ``mlp_forward(..., return_cache=True)``.
    params : sequence of (W, b) tuples
        The parameters used in that forward pass.
    activation_derivative : callable or None, default=None
        Derivative of the hidden activation, evaluated at z (a
        ``<name>_derivative`` function of chapter 17). None means identity.
    output_activation_derivative : callable or None, default=None
        Elementwise derivative of the output activation, evaluated at the last
        z. None means a linear output: the usual case, since the loss takes
        the logits.
    return_input_grad : bool, default=False
        Also return dL/dx (used for adversarial examples in chapter 21).

    Returns
    -------
    grads : list of (dW, db) tuples
        Same order and shapes as ``params``.
    dx : ndarray of shape (n, n_0)
        dL/dx, only if ``return_input_grad`` is True.

    Raises
    ------
    ValueError
        If ``len(cache) != len(params)`` or the shapes are inconsistent.

    Notes
    -----
    Tested against ``torch.autograd`` on the equivalent ``torch.nn.Sequential``
    with the same weights (float64), for identity, tanh, ReLU and sigmoid
    hidden activations.

    Examples
    --------
    The network of the ``mlp_forward`` example (2 inputs, 2 ReLU neurons,
    1 output), with its cache written out:

    >>> x = np.array([[1.0, 1.0]])
    >>> params = [(np.array([[1.0, -1.0], [0.5, -2.0]]), np.array([0.0, 1.0])),
    ...           (np.array([[1.0], [1.0]]), np.array([-1.0]))]
    >>> cache = [(x, np.array([[1.5, -2.0]])),
    ...          (np.array([[1.5, 0.0]]), np.array([[0.5]]))]
    >>> relu_derivative = lambda z: (z > 0).astype(float)
    >>> grads, dx = mlp_backward(np.array([[1.0]]), cache, params,
    ...                          activation_derivative=relu_derivative,
    ...                          return_input_grad=True)
    >>> grads[0][0]
    array([[1., 0.],
           [1., 0.]])
    >>> grads[1][0]
    array([[1.5],
           [0. ]])
    >>> dx
    array([[1. , 0.5]])
    """
    if len(cache) != len(params):
        raise ValueError(f"len(cache) = {len(cache)} but len(params) = {len(params)}: one entry per layer in both")
    if len(params) == 0:
        raise ValueError("params is empty: the network needs at least one layer")
    grad = _as_float(dout, "dout")
    last = len(params) - 1
    grads: list = [None] * len(params)
    for index in range(last, -1, -1):
        a_prev, z = cache[index]
        W, b = params[index]
        z = np.asarray(z, dtype=np.float64)
        if grad.shape != z.shape:
            raise ValueError(f"layer {index}: the incoming gradient has shape {grad.shape}, "
                             f"its pre-activation z has shape {z.shape}")
        derivative = output_activation_derivative if index == last else activation_derivative
        delta = grad if derivative is None else grad * np.asarray(derivative(z), dtype=np.float64)
        dx, dW, db = dense_backward(delta, np.asarray(a_prev, dtype=np.float64), np.asarray(W, dtype=np.float64))
        if np.size(b) != db.size:
            raise ValueError(f"layer {index}: b has {np.size(b)} entries, the layer has {db.size} neurons")
        grads[index] = (dW, db.reshape(np.shape(b)))
        grad = dx
    return (grads, grad) if return_input_grad else grads


def gradient_check(
    f: Callable[[], float],
    param: np.ndarray,
    analytic_grad: ArrayLike,
    eps: float = 1e-6,
    n_checks: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Compare an analytic gradient with centered finite differences.

    For each checked entry i, the numerical derivative is
    (f() with param[i] + eps  -  f() with param[i] - eps) / (2 * eps), and the
    relative error is |a - n| / max(|a| + |n|, 1e-12), where a is the analytic
    value.

    Parameters
    ----------
    f : callable with no argument, returning float
        Recomputes the scalar loss, reading the current values of ``param``.
    param : ndarray of floats
        Parameter array. It is modified IN PLACE (+eps, then -eps) and then
        restored exactly: after the call, it is identical bit for bit.
    analytic_grad : array-like of the shape of param
        Gradient to check.
    eps : float, default=1e-6
        Finite-difference step.
    n_checks : int or None, default=None
        Number of randomly chosen entries to check. None checks every entry.
    rng : numpy.random.Generator or None, default=None
        Generator used to choose the entries. None means
        ``np.random.default_rng()``.

    Returns
    -------
    float
        Maximum relative error over the checked entries: below 1e-7 the
        gradient is correct; above 1e-3 there is almost certainly a bug.

    Raises
    ------
    ValueError
        If the shapes of ``param`` and ``analytic_grad`` differ, or ``param`` is
        not an array of floats.

    Notes
    -----
    Tested on properties, in the spirit of ``torch.autograd.gradcheck``: exact
    gradients of quadratic and dense-layer losses give less than 1e-7, a sign
    error gives about 1, and ``param`` is restored bit for bit.

    Examples
    --------
    >>> w = np.array([1.0, -2.0, 3.0])
    >>> loss = lambda: float(np.sum(w ** 2))
    >>> gradient_check(loss, w, 2 * w) < 1e-7
    True
    >>> gradient_check(loss, w, -2 * w) > 0.9    # wrong sign
    True
    >>> w
    array([ 1., -2.,  3.])
    """
    if not isinstance(param, np.ndarray) or param.dtype.kind != "f":
        raise ValueError("param must be a NumPy array of floats (it is modified in place)")
    analytic = np.asarray(analytic_grad, dtype=np.float64)
    if analytic.shape != param.shape:
        raise ValueError(f"analytic_grad has shape {analytic.shape} but param has shape {param.shape}")
    if n_checks is None:
        indices = range(param.size)
    else:
        if int(n_checks) < 1:
            raise ValueError(f"n_checks must be >= 1 or None, got {n_checks}")
        generator = np.random.default_rng(rng)
        indices = generator.choice(param.size, size=min(int(n_checks), param.size), replace=False)
    flat_analytic = analytic.ravel()
    worst = 0.0
    for i in indices:
        i = int(i)
        old = param.flat[i]
        try:
            param.flat[i] = old + eps
            plus = float(f())
            param.flat[i] = old - eps
            minus = float(f())
        finally:
            param.flat[i] = old
        numeric = (plus - minus) / (2.0 * eps)
        a = float(flat_analytic[i])
        error = abs(a - numeric) / max(abs(a) + abs(numeric), 1e-12)
        worst = max(worst, error)
    return float(worst)
