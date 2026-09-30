"""Losses and backpropagation — mylearn, chapter 18 (Backpropagation).

Loss functions that return both their value and their gradient, the backward
pass of a dense layer and of a whole MLP, and a finite-difference gradient
checker. Everything is explicit: the backward pass reads the cache
``(a_prev, z)`` returned by ``mylearn.nn.layers.mlp_forward`` (chapter 16) and
the ``<name>_derivative`` functions of ``mylearn.nn.activations`` (chapter 17).

All losses use the natural logarithm and are averaged over the batch, like
PyTorch's ``reduction="mean"``.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .activations import log_softmax, sigmoid  # chapter 17, to reuse below


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
    raise NotImplementedError("mse_loss() is not implemented yet")


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
    # TODO: write the loss as a function of z that never overflows; reuse sigmoid
    #   (from .activations) for the gradient.
    raise NotImplementedError("binary_cross_entropy_with_logits() is not implemented yet")


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
    # TODO: turn integer labels into one-hot rows; log_softmax (from .activations)
    #   keeps the loss finite.
    raise NotImplementedError("softmax_cross_entropy() is not implemented yet")


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
    raise NotImplementedError("dense_backward() is not implemented yet")


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
    # TODO: walk the layers from the last to the first; the input of layer l is
    #   cache[l][0], its pre-activation cache[l][1].
    raise NotImplementedError("mlp_backward() is not implemented yet")


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
    # TODO: for each checked index: save the value, evaluate f at +eps and -eps,
    #   then put the saved value back (never old + eps - eps).
    raise NotImplementedError("gradient_check() is not implemented yet")
