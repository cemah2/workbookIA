"""Regularization and normalization — mylearn, chapter 20 (Deep Learning).

Dropout, batch normalization and layer normalization as pure NumPy functions,
with explicit forward and backward passes (like ``mylearn.nn.backward``) and
the conventions of PyTorch (inverted dropout, running statistics of batch
norm). They are the "from scratch" reference before ``torch.nn.Dropout``,
``torch.nn.BatchNorm1d/2d`` and ``torch.nn.LayerNorm``. The L2 penalty and
``EarlyStopping`` work both with the NumPy training loop and with a PyTorch
loop (``EarlyStopping`` is used again in chapters 23 and 24).

Reference implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def _as_float(values: ArrayLike) -> np.ndarray:
    """A new float64 array (never a view of the caller's data)."""
    return np.array(values, dtype=np.float64, copy=True)


def _feature_axes(ndim: int) -> tuple[int, ...]:
    """The axes reduced by batch normalization: every axis except axis 1."""
    if ndim == 2:
        return (0,)
    if ndim == 4:
        return (0, 2, 3)
    raise ValueError(f"batch normalization expects a 2-D (N, D) or 4-D (N, C, H, W) input, got {ndim}-D")


def dropout_forward(
    x: ArrayLike,
    p: float = 0.5,
    training: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Inverted dropout: zero each value with probability p, scale the others by 1 / (1 - p).

    The scaling keeps the expected value of each activation unchanged, so
    nothing has to be rescaled at evaluation time. A new random mask is drawn
    at EVERY call (every mini-batch), not once per epoch.

    Parameters
    ----------
    x : array-like of any shape
        Activations.
    p : float, default=0.5
        Drop probability, in [0, 1].
    training : bool, default=True
        False means evaluation mode: identity, no mask.
    rng : numpy.random.Generator or None, default=None
        Generator used to draw the mask. None means ``np.random.default_rng()``.

    Returns
    -------
    out : ndarray of the shape of x
        The activations after dropout, as float64.
    mask : ndarray of the shape of x, or None
        Values 0 or 1 / (1 - p), such that ``out = x * mask``. None in
        evaluation mode or when p = 0 (``out`` is then a copy of x). When
        p = 1, ``out`` and ``mask`` are all zeros.

    Raises
    ------
    ValueError
        If ``p`` is not in [0, 1].

    Notes
    -----
    Tested on the properties of ``torch.nn.functional.dropout``: same scaling,
    fraction of zeros close to p (within 1 % on 1e6 values), mean preserved,
    identity in evaluation mode, zeros when p = 1.

    Examples
    --------
    >>> out, mask = dropout_forward(np.ones((2, 4)), p=0.5, rng=np.random.default_rng(0))
    >>> set(np.unique(out).tolist()) <= {0.0, 2.0}
    True
    >>> bool(np.array_equal(out, np.ones((2, 4)) * mask))
    True
    >>> dropout_forward(np.ones(3), training=False)
    (array([1., 1., 1.]), None)
    """
    if not 0.0 <= p <= 1.0:          # also refuses nan
        raise ValueError(f"p must be in [0, 1], got {p}")
    x = _as_float(x)
    if not training or p == 0.0:
        return x, None
    if p == 1.0:
        return np.zeros_like(x), np.zeros_like(x)
    rng = np.random.default_rng(rng)
    mask = (rng.random(x.shape) >= p) / (1.0 - p)
    return x * mask, mask


def dropout_backward(dout: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Gradient of dropout: the upstream gradient times the same mask.

    Parameters
    ----------
    dout : ndarray
        Upstream gradient dL/dout, with the shape of x.
    mask : ndarray or None
        The mask returned by ``dropout_forward`` for this forward pass. None
        means identity (evaluation mode or p = 0).

    Returns
    -------
    ndarray of the shape of dout
        dL/dx.

    Raises
    ------
    ValueError
        If ``mask`` is not None and its shape differs from ``dout.shape``.

    Notes
    -----
    Tested against ``torch.autograd`` through ``x * mask`` with the same mask.

    Examples
    --------
    >>> dropout_backward(np.array([1.0, 1.0, 1.0]), np.array([0.0, 2.0, 2.0]))
    array([0., 2., 2.])
    """
    dout = _as_float(dout)
    if mask is None:
        return dout
    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != dout.shape:
        raise ValueError(f"mask has shape {mask.shape} but dout has shape {dout.shape}")
    return dout * mask


def batchnorm_forward(
    x: ArrayLike,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> tuple[np.ndarray, dict]:
    """Batch normalization per feature (N, D) or per channel (N, C, H, W), as PyTorch.

    The statistics are computed over every axis except axis 1 (the features or
    channels): out = gamma * (x - mean) / sqrt(var + eps) + beta.

    - Training mode: ``mean`` and ``var`` are the batch mean and BIASED batch
      variance; the running statistics are updated IN PLACE with
      ``running = (1 - momentum) * running + momentum * batch_stat``, where the
      variance used for ``running_var`` is the UNBIASED batch variance.
    - Evaluation mode: ``running_mean`` and ``running_var`` are used and left
      unchanged.

    Parameters
    ----------
    x : array-like of shape (N, D) or (N, C, H, W)
        Input batch.
    gamma : ndarray of shape (D,) or (C,)
        Learned scale.
    beta : ndarray of shape (D,) or (C,)
        Learned shift.
    running_mean : ndarray of shape (D,) or (C,)
        Running mean, updated in place in training mode.
    running_var : ndarray of shape (D,) or (C,)
        Running variance, updated in place in training mode.
    training : bool, default=True
        Training mode (batch statistics) or evaluation mode (running ones).
    momentum : float, default=0.1
        Weight of the new batch statistic in the running update (PyTorch
        default; beware, Keras uses the opposite convention).
    eps : float, default=1e-5
        Added to the variance inside the square root.

    Returns
    -------
    out : ndarray of the shape of x
        Normalized, scaled and shifted batch.
    cache : dict
        Everything ``batchnorm_backward`` needs, for example the normalized
        input x_hat, 1 / sqrt(var + eps), gamma, the reduced axes and the mode.
        Its content is up to you: only your ``batchnorm_backward`` reads it.

    Raises
    ------
    ValueError
        If ``x`` is not 2-D or 4-D, if a parameter does not have the shape
        (x.shape[1],), or if there are fewer than 2 values per feature in
        training mode (as PyTorch).

    Notes
    -----
    Tested against ``torch.nn.BatchNorm1d`` and ``torch.nn.BatchNorm2d`` in
    float64: same outputs, same running statistics after 3 training steps,
    same output in evaluation mode.

    Examples
    --------
    >>> x = np.array([[1.0, 2.0], [3.0, 6.0]])
    >>> running_mean, running_var = np.zeros(2), np.ones(2)
    >>> out, cache = batchnorm_forward(x, np.ones(2), np.zeros(2), running_mean, running_var)
    >>> out
    array([[-0.999995  , -0.99999875],
           [ 0.999995  ,  0.99999875]])
    >>> running_mean, running_var
    (array([0.2, 0.4]), array([1.1, 1.7]))
    """
    x = _as_float(x)
    axes = _feature_axes(x.ndim)
    n_features = x.shape[1]
    for name, value in (("gamma", gamma), ("beta", beta), ("running_mean", running_mean),
                        ("running_var", running_var)):
        if np.shape(value) != (n_features,):
            raise ValueError(f"{name} must have shape ({n_features},), got {np.shape(value)}")
    shape = [1] * x.ndim
    shape[1] = n_features                                     # (1, D) or (1, C, 1, 1)
    count = x.size // n_features                              # values per feature
    if training:
        if count < 2:
            raise ValueError(f"expected more than 1 value per feature in training mode, got {count}")
        mean = x.mean(axis=axes)
        var = x.var(axis=axes)                                # biased: used to normalize
        running_mean *= 1.0 - momentum
        running_mean += momentum * mean
        running_var *= 1.0 - momentum
        running_var += momentum * var * count / (count - 1)   # unbiased: used for the running estimate
    else:
        mean = np.asarray(running_mean, dtype=np.float64)
        var = np.asarray(running_var, dtype=np.float64)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean.reshape(shape)) * inv_std.reshape(shape)
    gamma = np.asarray(gamma, dtype=np.float64)
    out = gamma.reshape(shape) * x_hat + np.asarray(beta, dtype=np.float64).reshape(shape)
    cache = {"x_hat": x_hat, "inv_std": inv_std, "gamma": gamma.copy(), "axes": axes, "shape": shape,
             "count": count, "training": training}
    return out, cache


def batchnorm_backward(dout: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the gradients of ``batchnorm_forward``.

    In training mode the gradient flows through the batch mean and variance
    too (each output depends on the whole batch); in evaluation mode the
    normalization is a fixed affine map.

    Parameters
    ----------
    dout : ndarray of the shape of x
        Upstream gradient dL/dout.
    cache : dict
        The cache returned by ``batchnorm_forward``.

    Returns
    -------
    dx : ndarray of the shape of x
        dL/dx. In training mode it sums to 0 over the normalized axes.
    dgamma : ndarray of shape (D,) or (C,)
        dL/dgamma.
    dbeta : ndarray of shape (D,) or (C,)
        dL/dbeta.

    Notes
    -----
    Tested against ``torch.autograd`` through ``torch.nn.BatchNorm1d`` and
    ``torch.nn.BatchNorm2d`` (float64, atol 1e-9), in both modes.

    Examples
    --------
    >>> x = np.array([[1.0, 2.0], [3.0, 6.0]])
    >>> out, cache = batchnorm_forward(x, np.ones(2), np.zeros(2), np.zeros(2), np.ones(2))
    >>> dx, dgamma, dbeta = batchnorm_backward(np.eye(2), cache)
    >>> dgamma, dbeta
    (array([-0.999995  ,  0.99999875]), array([1., 1.]))
    >>> bool(np.allclose(dx.sum(axis=0), 0.0))
    True
    """
    dout = _as_float(dout)
    x_hat, axes, shape = cache["x_hat"], cache["axes"], cache["shape"]
    if dout.shape != x_hat.shape:
        raise ValueError(f"dout has shape {dout.shape} but the forward input had shape {x_hat.shape}")
    dbeta = dout.sum(axis=axes)
    dgamma = (dout * x_hat).sum(axis=axes)
    scale = (cache["gamma"] * cache["inv_std"]).reshape(shape)
    if not cache["training"]:
        return dout * scale, dgamma, dbeta
    m = cache["count"]
    dx = scale / m * (m * dout - dbeta.reshape(shape) - x_hat * dgamma.reshape(shape))
    return dx, dgamma, dbeta


def layer_norm(
    x: ArrayLike,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Layer normalization over the last axis of each sample.

    out = gamma * (x - mean) / sqrt(var + eps) + beta, with the mean and the
    BIASED variance computed over the last axis only. No running statistics:
    the same computation in training and in evaluation.

    Parameters
    ----------
    x : array-like of shape (..., D)
        Input; each vector along the last axis is normalized on its own.
    gamma : ndarray of shape (D,) or None, default=None
        Scale. None means ones.
    beta : ndarray of shape (D,) or None, default=None
        Shift. None means zeros.
    eps : float, default=1e-5
        Added to the variance inside the square root.

    Returns
    -------
    ndarray of the shape of x
        The normalized array, as float64.

    Raises
    ------
    ValueError
        If ``gamma`` or ``beta`` is given with a shape other than (D,).

    Notes
    -----
    Tested against ``torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps)``
    in float64.

    Examples
    --------
    >>> layer_norm([[1.0, 2.0, 3.0]])
    array([[-1.22473569,  0.        ,  1.22473569]])
    >>> layer_norm([[1.0, 2.0, 3.0]], gamma=np.full(3, 2.0), beta=np.ones(3))
    array([[-1.44947137,  1.        ,  3.44947137]])
    """
    x = _as_float(x)
    d = x.shape[-1]
    for name, value in (("gamma", gamma), ("beta", beta)):
        if value is not None and np.shape(value) != (d,):
            raise ValueError(f"{name} must have shape ({d},), got {np.shape(value)}")
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * np.asarray(gamma, dtype=np.float64)
    if beta is not None:
        out = out + np.asarray(beta, dtype=np.float64)
    return out


def l2_penalty(weights: Sequence[np.ndarray], lam: float) -> tuple[float, list[np.ndarray]]:
    """L2 penalty (lam / 2) * (sum of the squared weights), and its gradient lam * W.

    With the factor 1/2, adding these gradients to the data gradient is
    exactly the ``weight_decay=lam`` of ``torch.optim.SGD`` (and of
    ``mylearn.optim.SGD``). Beware: ``Ridge`` (chapter 9, like scikit-learn)
    penalizes alpha * ||w||**2, without the 1/2.

    Parameters
    ----------
    weights : sequence of ndarray
        Arrays to penalize (usually the W matrices only, not the biases).
    lam : float
        Regularization strength, >= 0.

    Returns
    -------
    penalty : float
        (lam / 2) * sum over all arrays of the sum of their squared entries.
    grads : list of ndarray
        ``grads[i] = lam * weights[i]``, same shapes.

    Raises
    ------
    ValueError
        If ``lam < 0``.

    Notes
    -----
    Tested against ``torch.autograd`` of ``0.5 * lam * sum(w**2)``, and by
    equivalence with ``torch.optim.SGD(weight_decay=lam)`` and
    ``mylearn.optim.SGD(weight_decay=lam)``.

    Examples
    --------
    >>> l2_penalty([np.array([1.0, 2.0]), np.array([[3.0]])], lam=0.5)
    (3.5, [array([0.5, 1. ]), array([[1.5]])])
    """
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    arrays = [np.asarray(w, dtype=np.float64) for w in weights]
    penalty = 0.5 * lam * sum(float(np.sum(w * w)) for w in arrays)
    return float(penalty), [lam * w for w in arrays]


class EarlyStopping:
    """Stop training when a validation metric has not improved for ``patience`` checks.

    Call ``step(value)`` once per check (usually once per epoch) with the
    monitored metric. A check is an improvement when
    ``value < best_ - min_delta`` (mode ``"min"``, for a loss) or
    ``value > best_ + min_delta`` (mode ``"max"``, for an accuracy); the first
    check is always an improvement.

    - Improvement: ``best_``, ``best_epoch_`` are updated, ``counter_`` goes
      back to 0, ``improved_`` is True, and deep copies of ``params`` (if
      given and ``restore_best`` is True) are kept in ``best_params_``.
    - No improvement: ``counter_`` increases by 1 and ``improved_`` is False.
      When ``counter_`` reaches ``patience``, ``step`` returns True: stop now.

    ``__init__`` is already written: it checks the hyperparameters and sets
    the attributes below to their initial values.

    Parameters
    ----------
    patience : int, default=5
        Number of consecutive checks without improvement after which training
        stops, >= 1. With ``patience=10``, a best value at check 13 stops the
        training at check 23.
    min_delta : float, default=0.0
        Minimal change counted as an improvement, >= 0.
    mode : {"min", "max"}, default="min"
        Whether the metric should decrease (loss) or increase (accuracy).
    restore_best : bool, default=True
        Keep deep copies of the parameters of the best check (when ``step``
        receives them), so that ``restore`` can put them back.

    Attributes
    ----------
    best_ : float
        Best value seen so far (inf in mode "min", -inf in mode "max" before
        the first check).
    best_epoch_ : int
        0-based index of the best check (-1 before the first check).
    counter_ : int
        Number of consecutive checks without improvement.
    improved_ : bool
        Whether the last check was an improvement. With PyTorch, save
        ``model.state_dict()`` when it is True.
    best_params_ : list of ndarray or None
        Copies of the parameters at the best check, or None.
    stopped_ : bool
        True once ``step`` has returned True.
    n_checks_ : int
        Number of calls to ``step`` so far.

    Raises
    ------
    ValueError
        If ``patience < 1``, ``min_delta < 0`` or ``mode`` is unknown.

    Notes
    -----
    Tested on scripted metric sequences (modes "min" and "max", min_delta,
    patience, restore). Same semantics as ``keras.callbacks.EarlyStopping``
    (patience, min_delta, restore_best_weights).

    Examples
    --------
    >>> stopper = EarlyStopping(patience=2)
    >>> [stopper.step(v) for v in [1.0, 0.8, 0.9, 0.85]]
    [False, False, False, True]
    >>> stopper.best_, stopper.best_epoch_, stopper.counter_
    (0.8, 1, 2)
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best_ = math.inf if mode == "min" else -math.inf
        self.best_epoch_ = -1
        self.counter_ = 0
        self.improved_ = False
        self.best_params_: list[np.ndarray] | None = None
        self.stopped_ = False
        self.n_checks_ = 0

    def step(self, value: float, params: Sequence[np.ndarray] | None = None) -> bool:
        """Record one check of the monitored metric and tell whether to stop.

        Parameters
        ----------
        value : float
            Value of the monitored metric at this check (for example the
            validation loss of the epoch). A nan is never an improvement.
        params : sequence of ndarray or None, default=None
            Current parameters. When this check is an improvement and
            ``restore_best`` is True, deep copies are stored in
            ``best_params_``.

        Returns
        -------
        bool
            True when ``counter_`` has reached ``patience``: stop training now.

        Examples
        --------
        >>> stopper = EarlyStopping(patience=1, min_delta=0.1)
        >>> stopper.step(1.0), stopper.step(0.95)    # 0.95 is not below 1.0 - 0.1
        (False, True)
        """
        value = float(value)
        if self.mode == "min":
            better = value < self.best_ - self.min_delta
        else:
            better = value > self.best_ + self.min_delta
        if self.n_checks_ == 0 and not math.isnan(value):
            better = True                                    # the first check is always an improvement
        if better:
            self.best_ = value
            self.best_epoch_ = self.n_checks_
            self.counter_ = 0
            self.improved_ = True
            if params is not None and self.restore_best:
                self.best_params_ = [np.array(p, dtype=np.float64, copy=True) for p in params]
        else:
            self.counter_ += 1
            self.improved_ = False
        self.n_checks_ += 1
        if self.counter_ >= self.patience:
            self.stopped_ = True
        return bool(self.counter_ >= self.patience)

    def restore(self, params: Sequence[np.ndarray]) -> None:
        """Copy the best parameters back into ``params``, in place.

        Parameters
        ----------
        params : sequence of ndarray
            The parameter arrays to overwrite (same order and shapes as the
            ones given to ``step``).

        Raises
        ------
        ValueError
            If no parameters were saved (``restore_best`` is False, or
            ``step`` never received ``params``), or if the shapes differ.

        Examples
        --------
        >>> w = np.array([1.0])
        >>> stopper = EarlyStopping(patience=1)
        >>> stopper.step(0.5, [w])
        False
        >>> w[0] = 7.0                 # training goes on and gets worse
        >>> stopper.step(0.6, [w])
        True
        >>> stopper.restore([w])
        >>> w
        array([1.])
        """
        if self.best_params_ is None:
            raise ValueError("no parameters were saved: pass them to step() (with restore_best=True)")
        if len(params) != len(self.best_params_):
            raise ValueError(f"got {len(params)} arrays, {len(self.best_params_)} were saved")
        for p, saved in zip(params, self.best_params_):
            if np.shape(p) != saved.shape:
                raise ValueError(f"an array has shape {np.shape(p)}, the saved one has shape {saved.shape}")
        for p, saved in zip(params, self.best_params_):
            p[...] = saved
