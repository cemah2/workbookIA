"""Numerical calculus and gradient descent — mylearn, chapter 5 (Curves and surfaces).

Derivatives and gradients estimated with finite differences, gradient descent (and
ascent), the local extrema of a sampled curve, and the nature of a critical point
(minimum, maximum, saddle or flat). Reused for gradient checking (chapter 18) and
for the loss landscapes of the optimizers (chapter 19).

Reference implementation: read it only after trying (``mon_travail/mylearn/calculus.py``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


_METHODS = ("central", "forward", "backward")


def _check_step(h: float) -> None:
    """A finite-difference step must be > 0 (``not h > 0`` also rejects NaN)."""
    if not h > 0:
        raise ValueError(f"h must be > 0, got {h!r}")


def _as_point(x: float | ArrayLike) -> tuple[float | np.ndarray, bool]:
    """(x as a float, True) for a scalar; (x as a float array, False) otherwise."""
    if np.ndim(x) == 0:
        return float(x), True
    return np.asarray(x, dtype=float), False


def numerical_derivative(
    f: Callable[[float], float],
    x: float | ArrayLike,
    h: float = 1e-5,
    method: str = "central",
) -> float | np.ndarray:
    """Estimate the derivative (slope) of f at x with a difference quotient.

    - ``'central'``: ``(f(x + h) - f(x - h)) / (2 h)`` (the most accurate);
    - ``'forward'``: ``(f(x + h) - f(x)) / h``;
    - ``'backward'``: ``(f(x) - f(x - h)) / h``.

    Parameters
    ----------
    f : callable
        Function of one real variable. If ``x`` is an array, ``f`` must accept
        NumPy arrays and work element by element (like ``np.sin``).
    x : float or array-like of any shape
        Point, or array of points, where the slope is estimated.
    h : float, default=1e-5
        Step, ``> 0``.
    method : {'central', 'forward', 'backward'}, default='central'
        Difference quotient to use.

    Returns
    -------
    float or np.ndarray
        A Python float for a scalar ``x``, otherwise a float array with the shape
        of ``x``.

    Raises
    ------
    ValueError
        If ``h <= 0`` or ``method`` is unknown.

    Notes
    -----
    Tested against analytic derivatives (``cos`` for ``sin``, ``2x`` for ``x²``,
    ``exp`` for ``exp``) and against ``torch.autograd.grad`` on the same function.

    Examples
    --------
    >>> round(numerical_derivative(lambda x: x ** 2, 3.0), 6)
    6.0
    >>> round(numerical_derivative(lambda x: x ** 2, 3.0, h=0.1), 6)
    6.0
    >>> round(numerical_derivative(lambda x: x ** 2, 3.0, h=0.1, method="forward"), 6)
    6.1
    >>> np.round(numerical_derivative(np.sin, np.array([0.0, 1.0, 2.0])), 6)
    array([ 1.      ,  0.540302, -0.416147])
    """
    _check_step(h)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    point, scalar = _as_point(x)
    if method == "central":
        slope = (f(point + h) - f(point - h)) / (2 * h)
    elif method == "forward":
        slope = (f(point + h) - f(point)) / h
    else:
        slope = (f(point) - f(point - h)) / h
    return float(slope) if scalar else np.asarray(slope, dtype=float)


def second_derivative(
    f: Callable[[float], float], x: float | ArrayLike, h: float = 1e-4
) -> float | np.ndarray:
    """Estimate the second derivative (curvature) of f at x.

    ``(f(x + h) - 2 f(x) + f(x - h)) / h²``: positive at the bottom of a valley,
    negative on a hilltop.

    Parameters
    ----------
    f : callable
        Function of one real variable (element-wise on arrays if ``x`` is an array).
    x : float or array-like of any shape
        Point, or array of points.
    h : float, default=1e-4
        Step, ``> 0`` (larger than for the first derivative: dividing by ``h²``
        amplifies rounding errors).

    Returns
    -------
    float or np.ndarray
        A Python float for a scalar ``x``, otherwise a float array with the shape
        of ``x``.

    Raises
    ------
    ValueError
        If ``h <= 0``.

    Notes
    -----
    Tested against analytic second derivatives (``6x`` for ``x³``, ``-sin`` for
    ``sin``).

    Examples
    --------
    >>> round(second_derivative(lambda x: x ** 3, 2.0), 3)
    12.0
    """
    _check_step(h)
    point, scalar = _as_point(x)
    curvature = (f(point + h) - 2 * f(point) + f(point - h)) / h ** 2
    return float(curvature) if scalar else np.asarray(curvature, dtype=float)


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: ArrayLike, h: float = 1e-5
) -> np.ndarray:
    """Estimate the gradient of a scalar function: one central difference per coordinate.

    Entry ``i`` of the gradient is ``(f(x + h e_i) - f(x - h e_i)) / (2 h)``, where
    ``e_i`` is the array with a 1 at position ``i`` and 0 elsewhere. ``x`` can have
    any shape (a weight matrix, for instance). The caller's ``x`` is never
    modified: the function works on a float copy.

    Parameters
    ----------
    f : callable
        Maps an array shaped like ``x`` to a float.
    x : array-like of any shape
        The point.
    h : float, default=1e-5
        Step, ``> 0``.

    Returns
    -------
    np.ndarray
        Float array with the shape of ``x``: the partial derivatives.

    Raises
    ------
    ValueError
        If ``h <= 0``.

    Notes
    -----
    Tested against ``wb.synth.rosenbrock_grad`` and ``torch.autograd.grad`` on the
    same function, plus the property that the caller's ``x`` is unchanged.

    Examples
    --------
    >>> f = lambda v: v[0] ** 2 + 3 * v[1]
    >>> np.round(numerical_gradient(f, [1.0, 2.0]), 6)
    array([2., 3.])
    """
    _check_step(h)
    point = np.array(x, dtype=float)          # a copy: the caller's x is never modified
    grad = np.zeros_like(point)
    flat_point = point.reshape(-1)            # views of the copy, one entry per coordinate
    flat_grad = grad.reshape(-1)
    for i in range(flat_point.size):
        saved = flat_point[i]
        flat_point[i] = saved + h
        f_plus = f(point)
        flat_point[i] = saved - h
        f_minus = f(point)
        flat_point[i] = saved                 # put the coordinate back exactly
        flat_grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


def gradient_descent(
    grad: Callable[[np.ndarray], np.ndarray],
    x0: ArrayLike,
    lr: float = 0.01,
    n_steps: int = 100,
    tol: float | None = None,
    maximize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize (or maximize) a function by gradient descent (or ascent).

    Starting from ``x0``, repeat ``x <- x - lr * grad(x)`` (``x <- x + lr * grad(x)``
    if ``maximize``), at most ``n_steps`` times. Before each step, the gradient is
    evaluated at the current point; if ``tol`` is given and the Euclidean norm of
    this gradient (over all its entries) is ``< tol``, the loop stops without
    moving.

    Parameters
    ----------
    grad : callable
        Returns the gradient at a point, as an array shaped like ``x0`` (e.g.
        ``lambda x: numerical_gradient(f, x)``).
    x0 : array-like of any shape
        Starting point (converted to a float copy; the caller's array is unchanged).
    lr : float, default=0.01
        Learning rate, ``> 0``.
    n_steps : int, default=100
        Maximum number of steps, ``>= 0``.
    tol : float or None, default=None
        Stop early when the norm of the gradient is ``< tol``; never stop early if
        None.
    maximize : bool, default=False
        If True, climb (gradient ascent) instead of descending.

    Returns
    -------
    x_final : np.ndarray
        Float array with the shape of ``x0``: the last point reached.
    path : np.ndarray of shape (n_done + 1, *x0.shape)
        Every point visited, ``path[0] == x0`` and ``path[-1] == x_final``;
        ``n_done`` is the number of steps actually taken.

    Raises
    ------
    ValueError
        If ``lr <= 0``, ``n_steps < 0`` or ``tol < 0``.

    Notes
    -----
    Tested against ``torch.optim.SGD(params, lr=lr, maximize=maximize)`` fed with the
    same gradients: the trajectories must be identical.

    Examples
    --------
    >>> x_final, path = gradient_descent(lambda x: 2 * x, [1.0], lr=0.25, n_steps=3)
    >>> x_final
    array([0.125])
    >>> path
    array([[1.   ],
           [0.5  ],
           [0.25 ],
           [0.125]])
    >>> _, path = gradient_descent(lambda x: 2 * x, [1.0], lr=0.25, n_steps=100, tol=0.3)
    >>> path.shape
    (4, 1)
    >>> grad_hill = lambda x: -2 * (x - 2)  # gradient of -(x - 2)**2, top at x = 2
    >>> gradient_descent(grad_hill, [0.0], lr=0.25, n_steps=3, maximize=True)[0]
    array([1.75])
    """
    if not lr > 0:
        raise ValueError(f"lr must be > 0, got {lr!r}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps!r}")
    if tol is not None and tol < 0:
        raise ValueError(f"tol must be >= 0 or None, got {tol!r}")
    x = np.array(x0, dtype=float)             # a float copy of the starting point
    path = [x.copy()]
    sign = 1.0 if maximize else -1.0          # ascent climbs along the gradient, descent against it
    for _ in range(int(n_steps)):
        g = np.asarray(grad(x), dtype=float)
        if tol is not None and np.linalg.norm(g) < tol:
            break                             # (almost) flat: stop without moving
        x = x + sign * lr * g
        path.append(x.copy())
    return x, np.array(path)


def find_local_extrema(y: ArrayLike, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Find the strict local minima and maxima of a sampled curve.

    Sample ``i`` is a local minimum (maximum) if it is strictly smaller (larger)
    than every neighbour within ``order`` samples on each side. Neighbours beyond
    the ends of the array are ignored, but the first and last samples are never
    extrema. Equal neighbours (a plateau) prevent an extremum.

    Parameters
    ----------
    y : array-like of shape (n,)
        Samples of the curve (e.g. a smoothed time series).
    order : int, default=1
        How many neighbours to compare on each side, ``>= 1``.

    Returns
    -------
    argmin_indices : np.ndarray of shape (n_minima,)
        Sorted int indices of the local minima.
    argmax_indices : np.ndarray of shape (n_maxima,)
        Sorted int indices of the local maxima.

    Raises
    ------
    ValueError
        If ``y`` is not 1-D or ``order < 1``.

    Notes
    -----
    Tested against ``scipy.signal.argrelextrema(y, np.less, order=order)`` and
    ``scipy.signal.argrelextrema(y, np.greater, order=order)``.

    Examples
    --------
    >>> y = [0, 2, 1, 3, 1, 0, 1]
    >>> find_local_extrema(y)
    (array([2, 5]), array([1, 3]))
    >>> find_local_extrema(y, order=2)
    (array([5]), array([3]))
    """
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError(f"y must be a 1-D array, got shape {values.shape}")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f"order must be an integer >= 1, got {order!r}")
    n = values.size
    minima, maxima = [], []
    for i in range(1, n - 1):                 # the first and last samples are never extrema
        neighbours = np.concatenate([values[max(0, i - order):i], values[i + 1:i + order + 1]])
        if (values[i] < neighbours).all():
            minima.append(i)
        elif (values[i] > neighbours).all():
            maxima.append(i)
    return np.array(minima, dtype=int), np.array(maxima, dtype=int)


def classify_critical_point(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    h: float = 1e-3,
    tol: float = 1e-6,
    grad_tol: float = 1e-4,
) -> str:
    """Classify a critical point of f: minimum, maximum, saddle or flat.

    At a point where the gradient vanishes, the function looks at its curvature in
    several directions ``d``: the coordinate axes ``e_i`` and the diagonals
    ``e_i + e_j`` and ``e_i - e_j`` (``i < j``), without computing a Hessian. In each
    direction, the second difference is
    ``D_d = (f(x + h d) - 2 f(x) + f(x - h d)) / h²``; a value with ``|D_d| <= tol``
    counts as zero. The diagonals are needed to detect saddles such as ``x·y``,
    which is flat along both axes.

    Parameters
    ----------
    f : callable
        Scalar function of a 1-D array.
    x : array-like of shape (n,)
        The critical point.
    h : float, default=1e-3
        Step of the second differences.
    tol : float, default=1e-6
        Second differences with absolute value ``<= tol`` count as zero.
    grad_tol : float, default=1e-4
        Largest norm of ``numerical_gradient(f, x)`` (default step) for ``x`` to
        count as a critical point.

    Returns
    -------
    str
        ``'minimum'`` if every ``D_d > tol``, ``'maximum'`` if every ``D_d < -tol``,
        ``'saddle'`` if some ``D_d > tol`` and some ``D_d < -tol``, otherwise
        ``'flat'`` (inconclusive).

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, or if the norm of the numerical gradient at ``x``
        exceeds ``grad_tol`` (``x`` is not a critical point).

    Notes
    -----
    Tested on known cases: ``x² + y²`` (minimum), ``-x² - y²`` (maximum), ``x² - y²``
    and ``x·y`` (saddle), a constant (flat), and against the signs of
    ``np.linalg.eigvalsh`` of the analytic Hessian of quadratic forms.

    Examples
    --------
    >>> classify_critical_point(lambda v: v[0] ** 2 + v[1] ** 2, [0.0, 0.0])
    'minimum'
    >>> classify_critical_point(lambda v: v[0] * v[1], [0.0, 0.0])
    'saddle'
    >>> classify_critical_point(lambda v: 5.0, [1.0, 2.0])
    'flat'
    """
    point = np.array(x, dtype=float)
    if point.ndim != 1:
        raise ValueError(f"x must be a 1-D array, got shape {point.shape}")
    gradient_norm = float(np.linalg.norm(numerical_gradient(f, point)))
    if gradient_norm > grad_tol:
        raise ValueError(f"x is not a critical point: the norm of the gradient is {gradient_norm:.3g} "
                         f"(> grad_tol = {grad_tol:g})")
    n = point.size
    eye = np.eye(n)
    directions = [eye[i] for i in range(n)]   # the axes, then the diagonals e_i + e_j and e_i - e_j
    for i in range(n):
        for j in range(i + 1, n):
            directions += [eye[i] + eye[j], eye[i] - eye[j]]
    f_x = f(point)
    curvatures = np.array([(f(point + h * d) - 2 * f_x + f(point - h * d)) / h ** 2 for d in directions])
    up, down = curvatures > tol, curvatures < -tol
    if up.any() and down.any():
        return "saddle"
    if up.all():
        return "minimum"
    if down.all():
        return "maximum"
    return "flat"
