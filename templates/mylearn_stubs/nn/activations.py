"""Activation functions — mylearn, chapter 17 (Activation Functions).

Elementwise activation functions as pure NumPy functions, each with its
derivative ``<name>_derivative(z)``, plus a numerically stable softmax and
log-softmax, the Jacobian of the softmax, and the registry ``get_activation``
used by the training loops of chapters 18 to 20.

Convention: every derivative takes the PRE-activation ``z`` (the input of the
activation), never its output ``a``; this is what
``mylearn.nn.backward.mlp_backward`` expects. At a kink (for example ReLU at
0), the value chosen is the one PyTorch uses.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


def identity(z: ArrayLike) -> np.ndarray:
    """Identity activation (no non-linearity): return z as float64.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.

    Returns
    -------
    ndarray of the same shape as z
        A new float64 array equal to ``z`` (modifying it never changes ``z``).

    Notes
    -----
    Tested against ``np.asarray(z, dtype=float)``.

    Examples
    --------
    >>> identity([-2, 0, 2])
    array([-2.,  0.,  2.])
    """
    raise NotImplementedError("identity() is not implemented yet")


def identity_derivative(z: ArrayLike) -> np.ndarray:
    """Derivative of ``identity`` with respect to z: always 1.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).

    Returns
    -------
    ndarray of the same shape as z
        Ones, as float64.

    Notes
    -----
    Tested against ``np.ones_like(z, dtype=float)``.

    Examples
    --------
    >>> identity_derivative([-2.0, 0.0, 2.0])
    array([1., 1., 1.])
    """
    raise NotImplementedError("identity_derivative() is not implemented yet")


def step(z: ArrayLike, threshold: float = 0.0, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Step function: ``high`` where z >= threshold, ``low`` elsewhere.

    With the defaults it is the unit (Heaviside) step of the perceptron;
    ``low=-1`` gives the sign function.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.
    threshold : float, default=0.0
        Position of the jump.
    low : float, default=0.0
        Value left of the threshold (z < threshold).
    high : float, default=1.0
        Value at and right of the threshold (z >= threshold).

    Returns
    -------
    ndarray of the same shape as z
        Values ``low`` or ``high``, as float64.

    Notes
    -----
    Tested against ``np.heaviside(z - threshold, 1.0)`` rescaled to
    ``[low, high]``.

    Examples
    --------
    >>> step([-2.0, 0.0, 2.0])
    array([0., 1., 1.])
    >>> step([-2.0, 0.0, 2.0], low=-1.0)
    array([-1.,  1.,  1.])
    """
    raise NotImplementedError("step() is not implemented yet")


def step_derivative(
    z: ArrayLike, threshold: float = 0.0, low: float = 0.0, high: float = 1.0
) -> np.ndarray:
    """Derivative of ``step`` with respect to z: zero everywhere.

    The function is flat on both sides of the jump and the jump itself has no
    derivative (0 is returned there too). A zero gradient carries no
    information: this is why a network of perceptrons cannot be trained by
    gradient descent.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).
    threshold : float, default=0.0
        Position of the jump.
    low : float, default=0.0
        Value left of the threshold.
    high : float, default=1.0
        Value at and right of the threshold.

    Returns
    -------
    ndarray of the same shape as z
        Zeros, as float64.

    Notes
    -----
    Tested against ``np.zeros_like(z, dtype=float)`` (``torch.heaviside`` has
    no backward); central finite differences are 0 away from the threshold.

    Examples
    --------
    >>> step_derivative([-2.0, 0.0, 2.0])
    array([0., 0., 0.])
    """
    raise NotImplementedError("step_derivative() is not implemented yet")


def relu(z: ArrayLike) -> np.ndarray:
    """Rectified linear unit: max(0, z).

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.relu``.

    Examples
    --------
    >>> relu([-2.0, 0.0, 2.0])
    array([0., 0., 2.])
    """
    raise NotImplementedError("relu() is not implemented yet")


def relu_derivative(z: ArrayLike) -> np.ndarray:
    """Derivative of ``relu`` with respect to z: 1 for z > 0, else 0.

    Convention at z = 0: the derivative is 0, as in PyTorch.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.autograd.grad`` of ``torch.relu(z).sum()``, and
    against central finite differences away from 0.

    Examples
    --------
    >>> relu_derivative([-2.0, 0.0, 2.0])
    array([0., 0., 1.])
    """
    raise NotImplementedError("relu_derivative() is not implemented yet")


def leaky_relu(z: ArrayLike, negative_slope: float = 0.01) -> np.ndarray:
    """Leaky ReLU: z for z > 0, ``negative_slope * z`` otherwise.

    The small slope keeps a non-zero gradient for negative inputs. The book's
    example uses a slope of 0.1; a slope learned during training gives PReLU.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.
    negative_slope : float, default=0.01
        Slope for z < 0 (PyTorch default). A slope of 1 would make the
        function linear, so it is never used in practice.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.nn.functional.leaky_relu``.

    Examples
    --------
    >>> leaky_relu([-2.0, 0.0, 2.0], negative_slope=0.1)
    array([-0.2,  0. ,  2. ])
    """
    raise NotImplementedError("leaky_relu() is not implemented yet")


def leaky_relu_derivative(z: ArrayLike, negative_slope: float = 0.01) -> np.ndarray:
    """Derivative of ``leaky_relu`` with respect to z: 1 for z > 0, else negative_slope.

    Convention at z = 0: the derivative is ``negative_slope``, as in PyTorch.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).
    negative_slope : float, default=0.01
        Slope for z < 0.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.autograd.grad`` of
    ``torch.nn.functional.leaky_relu(z).sum()``, and against central finite
    differences away from 0.

    Examples
    --------
    >>> leaky_relu_derivative([-2.0, 0.0, 2.0], negative_slope=0.1)
    array([0.1, 0.1, 1. ])
    """
    raise NotImplementedError("leaky_relu_derivative() is not implemented yet")


def sigmoid(z: ArrayLike) -> np.ndarray:
    """Logistic sigmoid 1 / (1 + exp(-z)), without overflow for large |z|.

    A direct ``1 / (1 + np.exp(-z))`` overflows (with a RuntimeWarning) for
    very negative z: your version must not.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.

    Returns
    -------
    ndarray of the same shape as z
        Values in [0, 1] (in (0, 1) up to rounding), as float64.
        ``sigmoid(-1000)`` is 0.0 and ``sigmoid(1000)`` is 1.0, without warnings.

    Notes
    -----
    Tested against ``torch.sigmoid`` and ``scipy.special.expit``.

    Examples
    --------
    >>> sigmoid([-2.0, 0.0, 2.0])
    array([0.11920292, 0.5       , 0.88079708])
    >>> sigmoid([-1000.0, 0.0, 1000.0])
    array([0. , 0.5, 1. ])
    """
    raise NotImplementedError("sigmoid() is not implemented yet")


def sigmoid_derivative(z: ArrayLike) -> np.ndarray:
    """Derivative of ``sigmoid`` with respect to z: sigmoid(z) * (1 - sigmoid(z)).

    Its maximum is 0.25, at z = 0.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.autograd.grad`` of ``torch.sigmoid(z).sum()``, and
    against central finite differences.

    Examples
    --------
    >>> sigmoid_derivative([-2.0, 0.0, 2.0])
    array([0.10499359, 0.25      , 0.10499359])
    """
    raise NotImplementedError("sigmoid_derivative() is not implemented yet")


def tanh(z: ArrayLike) -> np.ndarray:
    """Hyperbolic tangent, with values in (-1, 1).

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.tanh`` and ``np.tanh``.

    Examples
    --------
    >>> tanh([-2.0, 0.0, 2.0])
    array([-0.96402758,  0.        ,  0.96402758])
    """
    raise NotImplementedError("tanh() is not implemented yet")


def tanh_derivative(z: ArrayLike) -> np.ndarray:
    """Derivative of ``tanh`` with respect to z: 1 - tanh(z)**2.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.autograd.grad`` of ``torch.tanh(z).sum()``, and
    against central finite differences.

    Examples
    --------
    >>> tanh_derivative([-2.0, 0.0, 2.0])
    array([0.07065082, 1.        , 0.07065082])
    """
    raise NotImplementedError("tanh_derivative() is not implemented yet")


def softplus(z: ArrayLike, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """Softplus log(1 + exp(beta * z)) / beta, a smooth version of ReLU.

    Where ``beta * z > threshold`` the function returns z itself (PyTorch
    convention): the difference is negligible and ``exp`` cannot overflow.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.
    beta : float, default=1.0
        Sharpness, > 0 (a large beta gets close to ReLU).
    threshold : float, default=20.0
        Above this value of ``beta * z`` the function is linear.

    Returns
    -------
    ndarray of the same shape as z
        Non-negative values (> 0 up to underflow), as float64.

    Notes
    -----
    Tested against ``torch.nn.functional.softplus(z, beta, threshold)``.

    Examples
    --------
    >>> softplus([-2.0, 0.0, 2.0])
    array([0.12692801, 0.69314718, 2.12692801])
    >>> softplus([1000.0])
    array([1000.])
    """
    raise NotImplementedError("softplus() is not implemented yet")


def softplus_derivative(z: ArrayLike, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """Derivative of ``softplus`` with respect to z: sigmoid(beta * z).

    It is exactly 1 in the linear regime (``beta * z > threshold``).

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).
    beta : float, default=1.0
        Sharpness, > 0.
    threshold : float, default=20.0
        Above this value of ``beta * z`` the function is linear.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.autograd.grad`` of
    ``torch.nn.functional.softplus(z, beta, threshold).sum()``, and against
    central finite differences.

    Examples
    --------
    >>> softplus_derivative([-2.0, 0.0, 2.0])
    array([0.11920292, 0.5       , 0.88079708])
    """
    raise NotImplementedError("softplus_derivative() is not implemented yet")


def elu(z: ArrayLike, alpha: float = 1.0) -> np.ndarray:
    """Exponential linear unit: z for z > 0, alpha * (exp(z) - 1) otherwise.

    A smoothed ReLU whose outputs can be negative (down to -alpha), which
    keeps the mean activation closer to 0.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.
    alpha : float, default=1.0
        The function tends to -alpha when z tends to minus infinity.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.nn.functional.elu``.

    Examples
    --------
    >>> elu([-2.0, 0.0, 2.0])
    array([-0.86466472,  0.        ,  2.        ])
    """
    raise NotImplementedError("elu() is not implemented yet")


def elu_derivative(z: ArrayLike, alpha: float = 1.0) -> np.ndarray:
    """Derivative of ``elu`` with respect to z: 1 for z > 0, alpha * exp(z) otherwise.

    Convention at z = 0: the derivative is ``alpha``, as in PyTorch.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).
    alpha : float, default=1.0
        The function tends to -alpha when z tends to minus infinity.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.autograd.grad`` of
    ``torch.nn.functional.elu(z, alpha).sum()``, and against central finite
    differences away from 0.

    Examples
    --------
    >>> elu_derivative([-2.0, 0.0, 2.0])
    array([0.13533528, 1.        , 1.        ])
    """
    raise NotImplementedError("elu_derivative() is not implemented yet")


def silu(z: ArrayLike) -> np.ndarray:
    """SiLU, also called swish (with beta = 1): z * sigmoid(z).

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.nn.functional.silu``.

    Examples
    --------
    >>> silu([-2.0, 0.0, 2.0])
    array([-0.23840584,  0.        ,  1.76159416])
    """
    raise NotImplementedError("silu() is not implemented yet")


def silu_derivative(z: ArrayLike) -> np.ndarray:
    """Derivative of ``silu`` with respect to z: s * (1 + z * (1 - s)), s = sigmoid(z).

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Notes
    -----
    Tested against ``torch.autograd.grad`` of
    ``torch.nn.functional.silu(z).sum()``, and against central finite
    differences.

    Examples
    --------
    >>> silu_derivative([-2.0, 0.0, 2.0])
    array([-0.09078425,  0.5       ,  1.09078425])
    """
    raise NotImplementedError("silu_derivative() is not implemented yet")


def gelu(z: ArrayLike, approximate: str = "none") -> np.ndarray:
    """Gaussian error linear unit z * Phi(z), exact or with the tanh approximation.

    ``Phi`` is the cumulative distribution function of N(0, 1):
    Phi(z) = 0.5 * (1 + erf(z / sqrt(2))). The tanh approximation is
    0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z**3))).

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values.
    approximate : {"none", "tanh"}, default="none"
        ``"none"`` for the exact formula, ``"tanh"`` for the approximation
        (same names as PyTorch).

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Raises
    ------
    ValueError
        If ``approximate`` is neither ``"none"`` nor ``"tanh"``.

    Notes
    -----
    NumPy has no ``erf``: vectorise ``math.erf`` from the standard library.
    Tested against ``torch.nn.functional.gelu(z, approximate=...)`` in float64.

    Examples
    --------
    >>> gelu([-2.0, 0.0, 2.0])
    array([-0.04550026,  0.        ,  1.95449974])
    >>> gelu([-2.0, 0.0, 2.0], approximate="tanh")
    array([-0.04540231,  0.        ,  1.95459769])
    """
    raise NotImplementedError("gelu() is not implemented yet")


def gelu_derivative(z: ArrayLike, approximate: str = "none") -> np.ndarray:
    """Derivative of ``gelu`` with respect to z.

    Exact form: Phi(z) + z * phi(z), where phi is the density of N(0, 1);
    with ``approximate="tanh"``, the derivative of the tanh approximation.

    Parameters
    ----------
    z : array-like of any shape
        Pre-activation values (not the output of the activation).
    approximate : {"none", "tanh"}, default="none"
        Which version of GELU to differentiate.

    Returns
    -------
    ndarray of the same shape as z
        As float64.

    Raises
    ------
    ValueError
        If ``approximate`` is neither ``"none"`` nor ``"tanh"``.

    Notes
    -----
    Tested against ``torch.autograd.grad`` of
    ``torch.nn.functional.gelu(z, approximate=...).sum()``, and against
    central finite differences.

    Examples
    --------
    >>> gelu_derivative([-2.0, 0.0, 2.0])
    array([-0.0852318,  0.5      ,  1.0852318])
    >>> gelu_derivative([-2.0, 0.0, 2.0], approximate="tanh")
    array([-0.08609926,  0.5       ,  1.08609926])
    """
    raise NotImplementedError("gelu_derivative() is not implemented yet")


def softmax(z: ArrayLike, axis: int = -1) -> np.ndarray:
    """Softmax along ``axis``, computed in a numerically stable way.

    softmax(z)_i = exp(z_i) / sum_j exp(z_j). Subtracting the maximum along
    ``axis`` first changes nothing mathematically and prevents overflow.

    Parameters
    ----------
    z : array-like
        Scores (logits), for example of shape (n, K) with the K classes along
        the last axis. At least 1-D.
    axis : int, default=-1
        Axis holding the classes.

    Returns
    -------
    ndarray of the same shape as z
        Non-negative probabilities summing to 1 along ``axis``. Adding a
        constant to z does not change them; no nan for logits of order 1e3.

    Raises
    ------
    ValueError
        If ``z`` has 0 dimensions (a single number).

    Notes
    -----
    Tested against ``torch.softmax(z, dim=axis)`` and
    ``scipy.special.softmax(z, axis=axis)``.

    Examples
    --------
    >>> softmax([1.0, 2.0, 3.0])
    array([0.09003057, 0.24472847, 0.66524096])
    >>> softmax([1001.0, 1002.0, 1003.0])
    array([0.09003057, 0.24472847, 0.66524096])
    """
    raise NotImplementedError("softmax() is not implemented yet")


def log_softmax(z: ArrayLike, axis: int = -1) -> np.ndarray:
    """Logarithm of the softmax, computed with the log-sum-exp trick.

    log_softmax(z) = z - logsumexp(z), with
    logsumexp(z) = m + log(sum_j exp(z_j - m)) and m = max_j z_j.
    Never computes ``log(0)``, even when a probability underflows to 0.

    Parameters
    ----------
    z : array-like
        Scores (logits), at least 1-D.
    axis : int, default=-1
        Axis holding the classes.

    Returns
    -------
    ndarray of the same shape as z
        Log-probabilities (<= 0), finite for logits up to 1e4.

    Raises
    ------
    ValueError
        If ``z`` has 0 dimensions (a single number).

    Notes
    -----
    Tested against ``torch.log_softmax(z, dim=axis)`` and
    ``scipy.special.log_softmax(z, axis=axis)``.

    Examples
    --------
    >>> log_softmax([1.0, 2.0, 3.0])
    array([-2.40760596, -1.40760596, -0.40760596])
    >>> log_softmax([0.0, 10000.0])
    array([-10000.,      0.])
    """
    raise NotImplementedError("log_softmax() is not implemented yet")


def softmax_derivative(z: ArrayLike) -> np.ndarray:
    """Jacobian of the softmax along the last axis: J[..., i, j] = p_i (delta_ij - p_j).

    ``p = softmax(z)`` and ``delta_ij`` is 1 if i == j, else 0. Unlike the
    elementwise activations, each output depends on every input, hence a
    K x K matrix per sample.

    Parameters
    ----------
    z : array-like of shape (K,) or (n, K)
        Logits of one sample or of n samples.

    Returns
    -------
    ndarray of shape (K, K) or (n, K, K)
        Jacobian of each sample: entry ``[i, j]`` is d p_i / d z_j. The
        matrices are symmetric and each of their rows sums to 0.

    Raises
    ------
    ValueError
        If ``z`` is not 1-D or 2-D.

    Notes
    -----
    Tested against
    ``torch.autograd.functional.jacobian(lambda t: torch.softmax(t, -1), z)``.

    Examples
    --------
    >>> softmax_derivative([0.0, 0.0])
    array([[ 0.25, -0.25],
           [-0.25,  0.25]])
    >>> softmax_derivative(np.zeros((4, 3))).shape
    (4, 3, 3)
    """
    raise NotImplementedError("softmax_derivative() is not implemented yet")


def get_activation(
    name: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return the pair (function, derivative) of an elementwise activation.

    The pair is made of this module's own functions, used with their default
    parameters: ``get_activation("relu")`` is ``(relu, relu_derivative)``.

    Parameters
    ----------
    name : str
        One of ``"identity"``, ``"step"``, ``"relu"``, ``"leaky_relu"``,
        ``"elu"``, ``"softplus"``, ``"sigmoid"``, ``"tanh"``, ``"silu"``
        (alias ``"swish"``) or ``"gelu"``. Case-insensitive.

    Returns
    -------
    f : callable
        The activation, usable as ``mlp_forward(..., activation=f)`` (chapter 16).
    f_derivative : callable
        Its derivative, usable as
        ``mlp_backward(..., activation_derivative=f_derivative)`` (chapter 18).

    Raises
    ------
    ValueError
        For an unknown name; the message lists the valid names.

    Notes
    -----
    Tested by identity with the module functions (``f is relu``) and on an
    unknown name.

    Examples
    --------
    >>> f, df = get_activation("ReLU")
    >>> f(np.array([-1.0, 2.0]))
    array([0., 2.])
    >>> df(np.array([-1.0, 2.0]))
    array([0., 1.])
    """
    raise NotImplementedError("get_activation() is not implemented yet")
