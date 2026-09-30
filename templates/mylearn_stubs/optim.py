"""Optimizers and learning-rate schedules — mylearn, chapter 19 (Optimizers).

Gradient-descent optimizers in NumPy that follow the EXACT conventions of
``torch.optim`` (same formulas, same defaults as torch 2.11, eps inside or
outside the square root), so that the tests can compare them step by step.
Usage: ``opt = Adam(params)``, then ``opt.step(grads)`` after each backward
pass. The parameters are updated IN PLACE; the ``(W, b)`` list of
``mylearn.nn.layers.init_mlp`` is flattened into ``[W1, b1, W2, b2, ...]``.
Learning-rate schedules and gradient clipping are pure functions.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _check_non_negative(**values: float) -> None:
    """Raise ValueError if one of the named hyperparameters is negative."""
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")


def _check_fraction(name: str, value: float, *, allow_one: bool = False) -> None:
    """Raise ValueError if value is outside [0, 1) (or [0, 1] when allow_one is True)."""
    ok = 0.0 <= value <= 1.0 if allow_one else 0.0 <= value < 1.0
    if not ok:
        interval = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"{name} must be in {interval}, got {value}")


class Optimizer:
    """Base class of the optimizers: parameters, learning rate and step counter.

    ``__init__`` is already written (it checks its arguments and creates the
    attributes below). Each subclass implements ``step``.

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE by ``step`` (typically every W and b of
        ``mylearn.nn.layers.init_mlp``, flattened into ``[W1, b1, W2, b2, ...]``).
        The list and the arrays are kept by reference, never copied.
    lr : float
        Learning rate, >= 0.

    Attributes
    ----------
    params : list of ndarray
        The parameters (the very list that was passed).
    lr : float
        Learning rate. Public: a schedule may change it between two steps,
        for example ``opt.lr = cosine_lr(step, total_steps, base_lr)``.
    t : int
        Number of steps already applied (0 before the first step).
    state : list of dict
        One dict of buffers per parameter (momentum, running averages...),
        empty until the first step, when the buffers are created (lazily, as
        in ``torch.optim``). The buffer names are up to you; the torch names
        are suggested (``"momentum_buffer"``, ``"exp_avg"``, ``"exp_avg_sq"``...).

    Raises
    ------
    ValueError
        If ``params`` is empty, contains something other than a float NumPy
        array, or if ``lr < 0``.

    Notes
    -----
    Tested through the subclasses.
    """

    def __init__(self, params: list[np.ndarray], lr: float) -> None:
        if len(params) == 0:
            raise ValueError("params is empty: pass the list of arrays to optimize")
        for p in params:
            if not isinstance(p, np.ndarray) or not np.issubdtype(p.dtype, np.floating):
                raise ValueError("every parameter must be a NumPy array of floats")
        _check_non_negative(lr=lr)
        self.params = params
        self.lr = lr
        self.t = 0
        self.state: list[dict[str, np.ndarray]] = [{} for _ in params]

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one update to every parameter, in place.

        Provided: abstract method, nothing to write here. It describes the
        contract of the ``step`` method that each subclass below implements.

        Parameters
        ----------
        grads : sequence of ndarray
            One gradient per parameter, in the same order and with the same
            shapes. They are not modified (weight decay, when used, is added
            to a copy).

        Returns
        -------
        None
            The arrays of ``self.params`` are modified in place and ``self.t``
            is incremented by 1.

        Raises
        ------
        ValueError
            If ``len(grads) != len(self.params)`` or a shape differs.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement step()")


class SGD(Optimizer):
    """Plain (mini-batch) gradient descent: p -= lr * (g + weight_decay * p).

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE, kept by reference (see ``Optimizer``).
    lr : float, default=1e-3
        Learning rate (torch 2.11 default).
    weight_decay : float, default=0.0
        L2 coefficient: ``weight_decay * p`` is added to the gradient.

    Attributes
    ----------
    params, lr, t, state
        See ``Optimizer``.
    weight_decay : float
        As given.

    Raises
    ------
    ValueError
        As ``Optimizer``, or if ``weight_decay < 0``.

    Notes
    -----
    Equivalent to ``torch.optim.SGD(lr=lr, momentum=0, weight_decay=weight_decay)``.
    Tested against it on a copy of the parameters: identical parameters
    (float64, rtol 1e-10) after 20 steps of random gradients, and after 100
    steps on the Rosenbrock function.

    Examples
    --------
    >>> w = np.array([1.0, -2.0])
    >>> opt = SGD([w], lr=0.1)
    >>> opt.step([np.array([0.5, -1.0])])
    >>> w
    array([ 0.95, -1.9 ])
    >>> opt.t
    1
    """

    def __init__(
        self, params: list[np.ndarray], lr: float = 1e-3, weight_decay: float = 0.0
    ) -> None:
        super().__init__(params, lr)
        _check_non_negative(weight_decay=weight_decay)
        self.weight_decay = weight_decay

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one SGD update in place (see ``Optimizer.step``).

        With g = grad + weight_decay * p: p -= lr * g.
        """
        # TODO: check grads, increment self.t, then update each p IN PLACE (p -= ...),
        #   never p = p - ... (that would create a new array).
        raise NotImplementedError("step() is not implemented yet")


class Momentum(Optimizer):
    """Heavy-ball momentum, PyTorch convention.

    For each parameter, with g = grad + weight_decay * p:
    b = g at the first step, then b = momentum * b + g; and p -= lr * b.
    (The book folds the learning rate into the velocity; both versions agree
    as long as lr stays constant.)

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE, kept by reference (see ``Optimizer``).
    lr : float, default=1e-3
        Learning rate.
    momentum : float, default=0.9
        Coefficient mu (the book's gamma), in [0, 1).
    weight_decay : float, default=0.0
        L2 coefficient added to the gradient.

    Attributes
    ----------
    params, lr, t, state
        See ``Optimizer``; ``state[i]`` holds the velocity buffer b.
    momentum, weight_decay : float
        As given.

    Raises
    ------
    ValueError
        As ``Optimizer``, if ``momentum`` is outside [0, 1) or if
        ``weight_decay < 0``.

    Notes
    -----
    Equivalent to ``torch.optim.SGD(lr, momentum, dampening=0, nesterov=False,
    weight_decay)``. Tested against it: identical parameters (float64, rtol
    1e-10) after 20 steps of random gradients, and after 100 steps on the
    Rosenbrock function.

    Examples
    --------
    >>> w = np.array([1.0])
    >>> opt = Momentum([w], lr=0.1, momentum=0.9)
    >>> opt.step([np.array([1.0])])     # b = 1.0
    >>> opt.step([np.array([1.0])])     # b = 0.9 * 1.0 + 1.0 = 1.9
    >>> w
    array([0.71])
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        _check_fraction("momentum", momentum)
        _check_non_negative(weight_decay=weight_decay)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one momentum update in place (see ``Optimizer.step``)."""
        # TODO: check grads, increment self.t; create the buffer at the first step, then
        #   update it and p IN PLACE.
        raise NotImplementedError("step() is not implemented yet")


class Nesterov(Optimizer):
    """Nesterov momentum, in PyTorch's reformulation.

    The gradient is taken at the current point (not at the look-ahead point
    of the book). With g = grad + weight_decay * p: b = g at the first step,
    then b = momentum * b + g; and p -= lr * (g + momentum * b).

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE, kept by reference (see ``Optimizer``).
    lr : float, default=1e-3
        Learning rate.
    momentum : float, default=0.9
        Coefficient mu, in [0, 1).
    weight_decay : float, default=0.0
        L2 coefficient added to the gradient.

    Attributes
    ----------
    params, lr, t, state
        See ``Optimizer``; ``state[i]`` holds the velocity buffer b.
    momentum, weight_decay : float
        As given.

    Raises
    ------
    ValueError
        As ``Optimizer``, if ``momentum`` is outside [0, 1) or if
        ``weight_decay < 0``.

    Notes
    -----
    Equivalent to ``torch.optim.SGD(lr, momentum, dampening=0, nesterov=True,
    weight_decay)``. Tested against it: identical parameters (float64, rtol
    1e-10) after 20 steps of random gradients, and after 100 steps on the
    Rosenbrock function.

    Examples
    --------
    >>> w = np.array([1.0])
    >>> opt = Nesterov([w], lr=0.1, momentum=0.9)
    >>> opt.step([np.array([1.0])])     # b = 1.0, update 1.0 + 0.9 * 1.0
    >>> opt.step([np.array([1.0])])     # b = 1.9, update 1.0 + 0.9 * 1.9
    >>> w
    array([0.539])
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        _check_fraction("momentum", momentum)
        _check_non_negative(weight_decay=weight_decay)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one Nesterov update in place (see ``Optimizer.step``)."""
        # TODO: same structure as Momentum.step, with the Nesterov update.
        raise NotImplementedError("step() is not implemented yet")


class Adagrad(Optimizer):
    """Adagrad: s += g**2; p -= lr * g / (sqrt(s) + eps).

    Each parameter entry gets its own step size, which shrinks as its squared
    gradients accumulate. With weight decay, g = grad + weight_decay * p.

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE, kept by reference (see ``Optimizer``).
    lr : float, default=1e-2
        Learning rate.
    eps : float, default=1e-10
        Added AFTER the square root (torch convention).
    initial_accumulator_value : float, default=0.0
        Initial value of every entry of s.
    weight_decay : float, default=0.0
        L2 coefficient added to the gradient.

    Attributes
    ----------
    params, lr, t, state
        See ``Optimizer``; ``state[i]`` holds the accumulator s.
    eps, initial_accumulator_value, weight_decay : float
        As given.

    Raises
    ------
    ValueError
        As ``Optimizer``, or if ``eps``, ``initial_accumulator_value`` or
        ``weight_decay`` is negative.

    Notes
    -----
    Equivalent to ``torch.optim.Adagrad(lr, lr_decay=0, weight_decay,
    initial_accumulator_value, eps)``. Tested against it: identical parameters
    (float64, rtol 1e-10) after 20 steps of random gradients, and after 100
    steps on the Rosenbrock function.

    Examples
    --------
    >>> w = np.array([1.0])
    >>> opt = Adagrad([w], lr=0.1)
    >>> opt.step([np.array([2.0])])     # s = 4: step of 0.1 * 2 / 2
    >>> w
    array([0.9])
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-2,
        eps: float = 1e-10,
        initial_accumulator_value: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        _check_non_negative(
            eps=eps,
            initial_accumulator_value=initial_accumulator_value,
            weight_decay=weight_decay,
        )
        self.eps = eps
        self.initial_accumulator_value = initial_accumulator_value
        self.weight_decay = weight_decay

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one Adagrad update in place (see ``Optimizer.step``)."""
        # TODO: check grads, increment self.t; the accumulator starts at
        #   initial_accumulator_value; update p IN PLACE.
        raise NotImplementedError("step() is not implemented yet")


class RMSprop(Optimizer):
    """RMSprop: v = alpha * v + (1 - alpha) * g**2; p -= lr * g / (sqrt(v) + eps).

    Like Adagrad, but with an exponential moving average of the squared
    gradients, so the step size does not shrink forever. v starts at 0. With
    weight decay, g = grad + weight_decay * p.

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE, kept by reference (see ``Optimizer``).
    lr : float, default=1e-2
        Learning rate (torch default).
    alpha : float, default=0.99
        Smoothing constant, in [0, 1] (the book's gamma, which suggests 0.9;
        torch default 0.99).
    eps : float, default=1e-8
        Added AFTER the square root.
    weight_decay : float, default=0.0
        L2 coefficient added to the gradient.

    Attributes
    ----------
    params, lr, t, state
        See ``Optimizer``; ``state[i]`` holds the running average v.
    alpha, eps, weight_decay : float
        As given.

    Raises
    ------
    ValueError
        As ``Optimizer``, if ``alpha`` is outside [0, 1], or if ``eps`` or
        ``weight_decay`` is negative.

    Notes
    -----
    Equivalent to ``torch.optim.RMSprop(lr, alpha, eps, weight_decay,
    momentum=0, centered=False)``. Tested against it: identical parameters
    (float64, rtol 1e-10) after 20 steps of random gradients, and after 100
    steps on the Rosenbrock function.

    Examples
    --------
    >>> w = np.array([1.0])
    >>> opt = RMSprop([w], lr=0.01)
    >>> opt.step([np.array([1.0])])     # v = 0.01: step of about 0.01 / 0.1
    >>> w
    array([0.90000001])
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        _check_fraction("alpha", alpha, allow_one=True)
        _check_non_negative(eps=eps, weight_decay=weight_decay)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one RMSprop update in place (see ``Optimizer.step``)."""
        # TODO: check grads, increment self.t; v starts at 0; update p IN PLACE.
        raise NotImplementedError("step() is not implemented yet")


class Adadelta(Optimizer):
    """Adadelta: a step size in the units of the parameters, no learning rate needed.

    With g = grad + weight_decay * p, and v = u = 0 at the start:
    v = rho * v + (1 - rho) * g**2;
    d = sqrt(u + eps) / sqrt(v + eps) * g;
    u = rho * u + (1 - rho) * d**2;
    p -= lr * d.

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE, kept by reference (see ``Optimizer``).
    lr : float, default=1.0
        Multiplier of the update (torch default 1.0; the original algorithm
        has none).
    rho : float, default=0.9
        Decay of both running averages, in [0, 1].
    eps : float, default=1e-6
        Added INSIDE both square roots (unlike the other optimizers).
    weight_decay : float, default=0.0
        L2 coefficient added to the gradient.

    Attributes
    ----------
    params, lr, t, state
        See ``Optimizer``; ``state[i]`` holds the running averages v and u.
    rho, eps, weight_decay : float
        As given.

    Raises
    ------
    ValueError
        As ``Optimizer``, if ``rho`` is outside [0, 1], or if ``eps`` or
        ``weight_decay`` is negative.

    Notes
    -----
    Equivalent to ``torch.optim.Adadelta(lr, rho, eps, weight_decay)``. Tested
    against it: identical parameters (float64, rtol 1e-10) after 20 steps of
    random gradients, and after 100 steps on the Rosenbrock function.

    Examples
    --------
    >>> w = np.array([1.0])
    >>> opt = Adadelta([w])
    >>> opt.step([np.array([1.0])])
    >>> w
    array([0.99683774])
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        _check_fraction("rho", rho, allow_one=True)
        _check_non_negative(eps=eps, weight_decay=weight_decay)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one Adadelta update in place (see ``Optimizer.step``)."""
        # TODO: check grads, increment self.t; v and u start at 0; update p IN PLACE.
        raise NotImplementedError("step() is not implemented yet")


class Adam(Optimizer):
    """Adam, with bias correction; weight decay is a coupled L2 term.

    With g = grad + weight_decay * p (coupled: the decay goes through the
    moments), m = v = 0 at the start and t the step number (1 at the first
    step): m = beta1 * m + (1 - beta1) * g; v = beta2 * v + (1 - beta2) * g**2;
    m_hat = m / (1 - beta1**t); v_hat = v / (1 - beta2**t);
    p -= lr * m_hat / (sqrt(v_hat) + eps).

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE, kept by reference (see ``Optimizer``).
    lr : float, default=1e-3
        Learning rate.
    betas : tuple of 2 floats, default=(0.9, 0.999)
        (beta1, beta2): decays of the first and second moments, each in [0, 1).
    eps : float, default=1e-8
        Added AFTER sqrt(v_hat).
    weight_decay : float, default=0.0
        L2 coefficient ADDED TO THE GRADIENT (coupled). See ``AdamW`` for the
        decoupled version.

    Attributes
    ----------
    params, lr, t, state
        See ``Optimizer``; ``state[i]`` holds the moments m and v.
    betas : tuple of 2 floats
        As given.
    eps, weight_decay : float
        As given.

    Raises
    ------
    ValueError
        As ``Optimizer``, if a beta is outside [0, 1), or if ``eps`` or
        ``weight_decay`` is negative.

    Notes
    -----
    Equivalent to ``torch.optim.Adam(lr, betas, eps, weight_decay,
    amsgrad=False)``. Tested against it: identical parameters (float64, rtol
    1e-10) after 20 steps of random gradients, and after 100 steps on the
    Rosenbrock function.

    Examples
    --------
    The first step of Adam moves each entry by about lr, whatever the size of
    its gradient:

    >>> w = np.array([1.0])
    >>> Adam([w], lr=0.1).step([np.array([0.5])])
    >>> w
    array([0.9])
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if len(betas) != 2:
            raise ValueError(f"betas must be a pair (beta1, beta2), got {betas!r}")
        _check_fraction("beta1", betas[0])
        _check_fraction("beta2", betas[1])
        _check_non_negative(eps=eps, weight_decay=weight_decay)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one Adam update in place (see ``Optimizer.step``)."""
        # TODO: check grads, increment self.t (the bias correction uses the new t);
        #   m and v start at 0; update p IN PLACE.
        raise NotImplementedError("step() is not implemented yet")


class AdamW(Optimizer):
    """Adam with decoupled weight decay.

    First p *= 1 - lr * weight_decay, then the Adam step computed on the RAW
    gradient (the decay does not go through the moments).

    Parameters
    ----------
    params : list of ndarray
        Float arrays updated IN PLACE, kept by reference (see ``Optimizer``).
    lr : float, default=1e-3
        Learning rate.
    betas : tuple of 2 floats, default=(0.9, 0.999)
        (beta1, beta2), each in [0, 1).
    eps : float, default=1e-8
        Added AFTER sqrt(v_hat).
    weight_decay : float, default=1e-2
        Decoupled decay (torch default 1e-2).

    Attributes
    ----------
    params, lr, t, state
        See ``Optimizer``; ``state[i]`` holds the moments m and v.
    betas : tuple of 2 floats
        As given.
    eps, weight_decay : float
        As given.

    Raises
    ------
    ValueError
        As ``Optimizer``, if a beta is outside [0, 1), or if ``eps`` or
        ``weight_decay`` is negative.

    Notes
    -----
    Equivalent to ``torch.optim.AdamW(lr, betas, eps, weight_decay,
    amsgrad=False)``. Tested against it: identical parameters (float64, rtol
    1e-10) after 20 steps of random gradients, and after 100 steps on the
    Rosenbrock function.

    Examples
    --------
    With a zero gradient, only the decay acts (Adam with the same
    ``weight_decay`` would move w by about lr = 0.1):

    >>> w = np.array([1.0])
    >>> AdamW([w], lr=0.1, weight_decay=0.01).step([np.array([0.0])])
    >>> w
    array([0.999])
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr)
        if len(betas) != 2:
            raise ValueError(f"betas must be a pair (beta1, beta2), got {betas!r}")
        _check_fraction("beta1", betas[0])
        _check_fraction("beta2", betas[1])
        _check_non_negative(eps=eps, weight_decay=weight_decay)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one AdamW update in place (see ``Optimizer.step``)."""
        # TODO: decay p IN PLACE first, then the Adam update on the raw gradient.
        raise NotImplementedError("step() is not implemented yet")


def exponential_decay_lr(step: int, base_lr: float, gamma: float) -> float:
    """Exponential decay of the learning rate: base_lr * gamma**step.

    Parameters
    ----------
    step : int
        Number of decays already applied (counted in updates or in epochs), >= 0.
    base_lr : float
        Initial learning rate, >= 0.
    gamma : float
        Decay factor, in (0, 1].

    Returns
    -------
    float
        The learning rate for this step.

    Raises
    ------
    ValueError
        If ``step < 0``, ``base_lr < 0`` or ``gamma`` is not in (0, 1].

    Notes
    -----
    Tested against ``torch.optim.lr_scheduler.ExponentialLR``: same sequence
    over 50 steps.

    Examples
    --------
    >>> [exponential_decay_lr(s, 0.1, 0.5) for s in range(4)]
    [0.1, 0.05, 0.025, 0.0125]
    """
    raise NotImplementedError("exponential_decay_lr() is not implemented yet")


def step_decay_lr(step: int, base_lr: float, step_size: int, gamma: float = 0.1) -> float:
    """Interval (step) decay: base_lr * gamma**(step // step_size).

    The learning rate is multiplied by ``gamma`` every ``step_size`` steps.

    Parameters
    ----------
    step : int
        Current epoch (or update), >= 0.
    base_lr : float
        Initial learning rate, >= 0.
    step_size : int
        Number of steps between two decays, >= 1.
    gamma : float, default=0.1
        Decay factor, in (0, 1].

    Returns
    -------
    float
        The learning rate for this step.

    Raises
    ------
    ValueError
        If ``step < 0``, ``base_lr < 0``, ``step_size < 1`` or ``gamma`` is not
        in (0, 1].

    Notes
    -----
    Tested against ``torch.optim.lr_scheduler.StepLR``: same sequence over 50
    steps.

    Examples
    --------
    >>> step_decay_lr(9, 0.1, step_size=10, gamma=0.5)
    0.1
    >>> step_decay_lr(25, 0.1, step_size=10, gamma=0.5)
    0.025
    """
    raise NotImplementedError("step_decay_lr() is not implemented yet")


def bold_driver_lr(
    lr: float,
    loss: float,
    prev_loss: float,
    increase: float = 1.05,
    decrease: float = 0.5,
    tolerance: float = 0.0,
) -> float:
    """Bold-driver rule (book section 19.3.3): adapt the learning rate to the epoch loss.

    Grow the learning rate while the loss decreases; cut it when the loss
    rises by more than the tolerance:

    - ``loss < prev_loss``: return ``lr * increase``;
    - ``loss > prev_loss * (1 + tolerance)``: return ``lr * decrease``;
    - otherwise (inside the tolerance band): return ``lr`` unchanged.

    Parameters
    ----------
    lr : float
        Current learning rate.
    loss : float
        Loss of the epoch that just finished.
    prev_loss : float
        Loss of the previous epoch.
    increase : float, default=1.05
        Growth factor, > 1.
    decrease : float, default=0.5
        Cut factor, in (0, 1).
    tolerance : float, default=0.0
        Relative increase of the loss tolerated without cutting, >= 0.

    Returns
    -------
    float
        The new learning rate.

    Raises
    ------
    ValueError
        If ``increase <= 1``, ``decrease`` is not in (0, 1) or ``tolerance < 0``.

    Notes
    -----
    No library equivalent: tested on scripted loss sequences.

    Examples
    --------
    >>> bold_driver_lr(1.0, loss=0.8, prev_loss=1.0)
    1.05
    >>> bold_driver_lr(1.0, loss=1.2, prev_loss=1.0)
    0.5
    >>> bold_driver_lr(1.0, loss=1.05, prev_loss=1.0, tolerance=0.1)
    1.0
    """
    raise NotImplementedError("bold_driver_lr() is not implemented yet")


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup from 0 to base_lr, then cosine decay from base_lr to min_lr.

    - ``step < warmup_steps``: ``base_lr * step / warmup_steps``;
    - ``warmup_steps <= step < total_steps``: with
      ``progress = (step - warmup_steps) / (total_steps - warmup_steps)``,
      ``min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))``;
    - ``step >= total_steps``: ``min_lr`` (no restart).

    Parameters
    ----------
    step : int
        Current update index, >= 0.
    total_steps : int
        Total number of updates, > warmup_steps.
    base_lr : float
        Peak learning rate, reached at the end of the warmup.
    warmup_steps : int, default=0
        Length of the linear warmup, >= 0 (0: no warmup).
    min_lr : float, default=0.0
        Final learning rate, <= base_lr.

    Returns
    -------
    float
        The learning rate for this step.

    Raises
    ------
    ValueError
        If ``total_steps <= warmup_steps``, ``step < 0``, ``warmup_steps < 0``
        or ``min_lr > base_lr``.

    Notes
    -----
    Tested against ``transformers.get_cosine_schedule_with_warmup`` (with
    ``min_lr=0``) and ``torch.optim.lr_scheduler.CosineAnnealingLR`` (with
    ``warmup_steps=0``, ``T_max=total_steps`` and ``eta_min=min_lr``): same
    values.

    Examples
    --------
    >>> [cosine_lr(s, 100, 0.1, warmup_steps=10) for s in (0, 5, 10, 55, 100)]
    [0.0, 0.05, 0.1, 0.05, 0.0]
    """
    # TODO: check the arguments, then handle the three cases in order.
    raise NotImplementedError("cosine_lr() is not implemented yet")


def clip_grad_norm(grads: Sequence[np.ndarray], max_norm: float, eps: float = 1e-6) -> float:
    """Rescale all the gradients IN PLACE so that their global L2 norm is at most max_norm.

    The global norm is the L2 norm of all the gradients concatenated into one
    vector. When it exceeds ``max_norm``, every gradient is multiplied by
    ``max_norm / (norm + eps)`` (the same factor for all), as PyTorch does.

    Parameters
    ----------
    grads : sequence of ndarray
        Gradients, modified in place.
    max_norm : float
        Maximum global norm, > 0.
    eps : float, default=1e-6
        Added to the norm in the scaling factor.

    Returns
    -------
    float
        The global norm BEFORE clipping. The gradients are left unchanged
        when it is below ``max_norm``.

    Raises
    ------
    ValueError
        If ``max_norm <= 0``.

    Notes
    -----
    Tested against ``torch.nn.utils.clip_grad_norm_``: same returned norm and
    same clipped gradients.

    Examples
    --------
    >>> g = [np.array([3.0, 4.0])]
    >>> clip_grad_norm(g, max_norm=1.0)
    5.0
    >>> g[0]
    array([0.59999988, 0.79999984])
    """
    # TODO: one global norm over all the arrays; rescale in place only when it
    #   exceeds max_norm.
    raise NotImplementedError("clip_grad_norm() is not implemented yet")
