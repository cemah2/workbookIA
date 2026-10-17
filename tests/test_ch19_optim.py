"""Tests of mylearn.optim (chapter 19): oracle tests and properties.

    pytest tests/test_ch19_optim.py              # your code (mon_travail/mylearn/optim.py)
    pytest tests/test_ch19_optim.py --impl=ref   # the reference

Oracles: PyTorch (``torch.optim.SGD``, ``Adagrad``, ``RMSprop``, ``Adadelta``, ``Adam``, ``AdamW`` run on
copies of the same parameters with the same gradients; ``torch.optim.lr_scheduler.ExponentialLR``,
``StepLR`` and ``CosineAnnealingLR``; ``torch.nn.utils.clip_grad_norm_``; autograd on the Rosenbrock
function), Hugging Face ``transformers.get_cosine_schedule_with_warmup`` when it is installed, and NumPy:
the pseudo-code of the torch.optim documentation, transcribed in this file. With PyTorch, the transcribed
pseudo-code is checked against torch at every test (a disagreement is a bug of this file, not yours);
without PyTorch, it is the oracle, so that no test is skipped.

Every test name starts with the name of what it tests, so that each exercise runs its own group:
``-k "test_exponential_decay_lr_ or test_step_decay_lr_ or test_bold_driver_lr_"`` (19.15),
``-k "test_sgd_ or test_momentum_ or test_nesterov_"`` (19.16),
``-k "test_adagrad_ or test_rmsprop_ or test_adadelta_"`` (19.17), ``-k "test_adam_ or test_adamw_"``
(19.18), ``-k "test_cosine_lr_ or test_clip_grad_norm_"`` (19.19).

Contract precisions written in the statements of 19.15 to 19.19 and checked here: ``step`` modifies the
very arrays of ``params`` (never a new array: ``p -= ...``), never modifies the gradients it receives,
raises ``ValueError`` for a wrong number of gradients or a wrong shape before changing anything
(no parameter updated, ``t`` unchanged), increments ``t`` by one, and reads
``self.lr`` at every step (a schedule may change it between two steps); two optimizers never share their
buffers; the schedules and ``clip_grad_norm`` return Python floats; ``clip_grad_norm`` leaves the
gradients untouched when the global norm does not exceed ``max_norm`` (the tests avoid the band of width
``eps`` just below it, where PyTorch already rescales by a factor of 1 - 1e-6).
The first line of every failure message says what was expected (it is the line that ``pytest -rf``
shows).
"""

import math

import numpy as np
import pytest

try:
    import torch
except ImportError:  # the PyTorch oracles run on Colab; the NumPy pseudo-code takes over here
    torch = None

try:
    from transformers import get_cosine_schedule_with_warmup
except Exception:  # noqa: BLE001 - transformers is optional (Colab has it)
    get_cosine_schedule_with_warmup = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")


@pytest.fixture
def optim(mylearn_module):
    return mylearn_module("optim")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3,
                           max_line_width=10**6)
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


def assert_close(result, expected, rtol=1e-10, atol=1e-13, msg=""):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected a number or an array, got an object of type {type(result).__name__}")
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}")
    ok = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=False)
    if not ok.all():
        bad = np.flatnonzero(~ok)
        i = int(bad[0])
        index = np.unravel_index(i, want.shape) if want.ndim > 1 else (i,)
        where = f" at index {tuple(int(k) for k in index)}" if want.ndim else ""
        head = (f"expected {want.ravel()[i] + 0.0:.12g}{where}, got {got.ravel()[i] + 0.0:.12g} "
                f"({bad.size} wrong value(s) out of {want.size})")
        data = f"expected {_short(want, 10)}\n     got {_short(got, 10)}" if want.size <= 12 else ""
        _fail(msg, head, data)


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
    try:
        function(*args, **kwargs)
    except error:
        return
    except NotImplementedError:
        raise                  # shown as "⏳ pas encore implémenté" by conftest.py
    except Exception as exc:  # noqa: BLE001 - say which error was raised instead
        raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else "")
                             + f" (it raised {type(exc).__name__}: {exc})") from None
    raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else ""))


def call(function, *args, expected="a result", **kwargs):
    """Call the learner's function; an unexpected exception becomes a failure that says what was expected."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def assert_python_float(value, what):
    if value is None:
        _fail("", f"expected a float from {what}, got None (did you forget the return?)")
    if not isinstance(value, float):
        _fail("", f"expected a Python float from {what}, got {type(value).__name__} (convert with float(...))")


# ------------------------------------------------------------------ the optimizers and their oracles
# name -> (class name in mylearn.optim, torch factory, kwargs of the tested settings)
SETTINGS = {
    "SGD": [{}, {"lr": 0.05}, {"lr": 0.02, "weight_decay": 0.1}],
    "Momentum": [{}, {"lr": 0.05, "momentum": 0.8}, {"lr": 0.02, "momentum": 0.95, "weight_decay": 0.1}],
    "Nesterov": [{}, {"lr": 0.05, "momentum": 0.8}, {"lr": 0.02, "momentum": 0.95, "weight_decay": 0.1}],
    "Adagrad": [{}, {"lr": 0.05}, {"lr": 0.05, "eps": 1e-6, "initial_accumulator_value": 0.3,
                                   "weight_decay": 0.1}],
    "RMSprop": [{}, {"lr": 0.01, "alpha": 0.9}, {"lr": 0.005, "alpha": 0.99, "eps": 1e-6, "weight_decay": 0.1}],
    "Adadelta": [{}, {"lr": 0.5, "rho": 0.8}, {"lr": 2.0, "rho": 0.95, "eps": 1e-4, "weight_decay": 0.1}],
    "Adam": [{}, {"lr": 0.01, "betas": (0.8, 0.95)}, {"lr": 0.02, "eps": 1e-6, "weight_decay": 0.1}],
    "AdamW": [{}, {"lr": 0.01, "betas": (0.8, 0.95), "weight_decay": 0.1}, {"lr": 0.02, "eps": 1e-6,
                                                                          "weight_decay": 0.0}],
}
NAMES = list(SETTINGS)
DEFAULTS = {  # torch 2.11 defaults, as in the docstrings
    "SGD": {"lr": 1e-3, "weight_decay": 0.0},
    "Momentum": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0},
    "Nesterov": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0},
    "Adagrad": {"lr": 1e-2, "eps": 1e-10, "initial_accumulator_value": 0.0, "weight_decay": 0.0},
    "RMSprop": {"lr": 1e-2, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0.0},
    "Adadelta": {"lr": 1.0, "rho": 0.9, "eps": 1e-6, "weight_decay": 0.0},
    "Adam": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
    "AdamW": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-2},
}
TORCH_DOC = {  # where to read the pseudo-code, for the messages
    "SGD": "torch.optim.SGD (momentum=0)", "Momentum": "torch.optim.SGD (momentum, dampening=0)",
    "Nesterov": "torch.optim.SGD (nesterov=True)", "Adagrad": "torch.optim.Adagrad",
    "RMSprop": "torch.optim.RMSprop", "Adadelta": "torch.optim.Adadelta", "Adam": "torch.optim.Adam",
    "AdamW": "torch.optim.AdamW",
}


def full_settings(name, kwargs):
    out = dict(DEFAULTS[name])
    out.update(kwargs)
    return out


def setting_id(kwargs):
    return "defaults" if not kwargs else ",".join(f"{k}={v}" for k, v in kwargs.items())


def torch_optimizer(name, tensors, kw):
    """The torch.optim optimizer that the docstring of `name` names."""
    kw = full_settings(name, kw)
    if name == "SGD":
        return torch.optim.SGD(tensors, lr=kw["lr"], momentum=0.0, weight_decay=kw["weight_decay"])
    if name in ("Momentum", "Nesterov"):
        return torch.optim.SGD(tensors, lr=kw["lr"], momentum=kw["momentum"], dampening=0.0,
                               nesterov=name == "Nesterov", weight_decay=kw["weight_decay"])
    if name == "Adagrad":
        return torch.optim.Adagrad(tensors, lr=kw["lr"], lr_decay=0.0, weight_decay=kw["weight_decay"],
                                   initial_accumulator_value=kw["initial_accumulator_value"], eps=kw["eps"])
    if name == "RMSprop":
        return torch.optim.RMSprop(tensors, lr=kw["lr"], alpha=kw["alpha"], eps=kw["eps"],
                                   weight_decay=kw["weight_decay"], momentum=0.0, centered=False)
    if name == "Adadelta":
        return torch.optim.Adadelta(tensors, lr=kw["lr"], rho=kw["rho"], eps=kw["eps"],
                                    weight_decay=kw["weight_decay"])
    cls = torch.optim.Adam if name == "Adam" else torch.optim.AdamW
    return cls(tensors, lr=kw["lr"], betas=kw["betas"], eps=kw["eps"], weight_decay=kw["weight_decay"],
               amsgrad=False)


class PseudoCode:
    """The pseudo-code of the torch.optim documentation, transcribed in NumPy (one object per run)."""

    def __init__(self, name, params, kw):
        self.name, self.kw = name, full_settings(name, kw)
        self.params = [np.array(p, dtype=float, copy=True) for p in params]
        self.state = [{} for _ in params]
        self.t = 0

    def step(self, grads, lr=None):
        kw, name = self.kw, self.name
        lr = kw["lr"] if lr is None else lr
        self.t += 1
        t = self.t
        for i, (p, g0) in enumerate(zip(self.params, grads)):
            st = self.state[i]
            g = np.array(g0, dtype=float)
            if name == "AdamW":
                p = p * (1 - lr * kw["weight_decay"])
            elif kw["weight_decay"]:
                g = g + kw["weight_decay"] * p
            if name == "SGD":
                p = p - lr * g
            elif name in ("Momentum", "Nesterov"):
                st["b"] = g.copy() if t == 1 else kw["momentum"] * st["b"] + g
                p = p - lr * (g + kw["momentum"] * st["b"] if name == "Nesterov" else st["b"])
            elif name == "Adagrad":
                st["s"] = st.get("s", np.full(p.shape, kw["initial_accumulator_value"])) + g * g
                p = p - lr * g / (np.sqrt(st["s"]) + kw["eps"])
            elif name == "RMSprop":
                st["v"] = kw["alpha"] * st.get("v", np.zeros(p.shape)) + (1 - kw["alpha"]) * g * g
                p = p - lr * g / (np.sqrt(st["v"]) + kw["eps"])
            elif name == "Adadelta":
                v = kw["rho"] * st.get("v", np.zeros(p.shape)) + (1 - kw["rho"]) * g * g
                u = st.get("u", np.zeros(p.shape))
                d = np.sqrt(u + kw["eps"]) / np.sqrt(v + kw["eps"]) * g
                st["v"], st["u"] = v, kw["rho"] * u + (1 - kw["rho"]) * d * d
                p = p - lr * d
            else:  # Adam, AdamW
                b1, b2 = kw["betas"]
                st["m"] = b1 * st.get("m", np.zeros(p.shape)) + (1 - b1) * g
                st["v"] = b2 * st.get("v", np.zeros(p.shape)) + (1 - b2) * g * g
                m_hat, v_hat = st["m"] / (1 - b1**t), st["v"] / (1 - b2**t)
                p = p - lr * m_hat / (np.sqrt(v_hat) + kw["eps"])
            self.params[i] = p


def oracle_trajectory(name, kw, params0, grads_seq, lrs=None):
    """The parameters after each step, from torch (checked against the pseudo-code) or from the pseudo-code."""
    pseudo = PseudoCode(name, params0, kw)
    traj_pseudo = []
    for k, grads in enumerate(grads_seq):
        pseudo.step(grads, None if lrs is None else lrs[k])
        traj_pseudo.append([p.copy() for p in pseudo.params])
    if torch is None:
        return traj_pseudo
    tensors = [torch.tensor(np.array(p, dtype=float), dtype=torch.float64, requires_grad=True) for p in params0]
    opt = torch_optimizer(name, tensors, kw)
    traj = []
    for k, grads in enumerate(grads_seq):
        if lrs is not None:
            for group in opt.param_groups:
                group["lr"] = lrs[k]
        for tensor, g in zip(tensors, grads):
            tensor.grad = torch.tensor(np.array(g, dtype=float), dtype=torch.float64)
        opt.step()
        traj.append([tensor.detach().numpy().copy() for tensor in tensors])
    for k, (a, b) in enumerate(zip(traj, traj_pseudo)):  # the oracle of this file must agree with torch
        for i, (x, y) in enumerate(zip(a, b)):
            assert np.allclose(x, y, rtol=1e-11, atol=1e-14), (
                f"bug in tests/test_ch19_optim.py (not in your code): the transcribed pseudo-code of {name} "
                f"disagrees with torch at step {k + 1}, parameter {i}")
    return traj


def make_optimizer(optim, name, params, kw):
    cls = getattr(optim, name)
    return call(cls, params, **kw, expected=f"a {name} optimizer")


def run_learner(optim, name, kw, params0, grads_seq, lrs=None):
    """Run the learner's optimizer on copies; return the parameters after each step and the objects."""
    params = [np.array(p, dtype=float, copy=True) for p in params0]
    opt = make_optimizer(optim, name, params, kw)
    traj = []
    for k, grads in enumerate(grads_seq):
        if lrs is not None:
            opt.lr = lrs[k]
        call(opt.step, [np.array(g, dtype=float, copy=True) for g in grads], expected="step() to run")
        traj.append([p.copy() for p in params])
    return traj, params, opt


def compare_trajectories(name, kw, got, want, what):
    for k, (a, b) in enumerate(zip(got, want)):
        for i, (x, y) in enumerate(zip(a, b)):
            shape = f", shape {np.shape(y)}" + (" (a 0-d array: a NumPy scalar is not updated in place)"
                                                if np.ndim(y) == 0 else "")
            assert_close(x, y, msg=f"{name}({setting_id(kw)}) {what}: parameter {i}{shape} after step {k + 1} "
                                   f"(compare with the pseudo-code of {TORCH_DOC[name]})")


SHAPES = [(3, 2), (4,), ()]


def random_problem(seed, n_steps=20):
    rng = np.random.default_rng(seed)
    params0 = [rng.normal(size=s) for s in SHAPES]
    grads_seq = [[rng.normal(scale=2.0, size=s) for s in SHAPES] for _ in range(n_steps)]
    return params0, grads_seq


def rosenbrock_grad(x, y):
    return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])


ROSENBROCK = {  # settings that stay finite for 100 steps from (-1.2, 1)
    "SGD": {"lr": 1e-3}, "Momentum": {"lr": 5e-4, "momentum": 0.9}, "Nesterov": {"lr": 5e-4, "momentum": 0.9},
    "Adagrad": {"lr": 0.1}, "RMSprop": {"lr": 0.01}, "Adadelta": {"lr": 1.0}, "Adam": {"lr": 0.05},
    "AdamW": {"lr": 0.05, "weight_decay": 0.01},
}


def torch_rosenbrock(name, kw, start, n_steps):
    """Oracle: torch.optim with autograd on f(x, y) = (1 - x)^2 + 100 (y - x^2)^2 (or the pseudo-code)."""
    if torch is None:
        pseudo = PseudoCode(name, [np.array(start, dtype=float)], kw)
        out = []
        for _ in range(n_steps):
            pseudo.step([rosenbrock_grad(*pseudo.params[0])])
            out.append(pseudo.params[0].copy())
        return out
    w = torch.tensor(np.array(start, dtype=float), dtype=torch.float64, requires_grad=True)
    opt = torch_optimizer(name, [w], kw)
    out = []
    for _ in range(n_steps):
        opt.zero_grad()
        f = (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2
        f.backward()
        opt.step()
        out.append(w.detach().numpy().copy())
    return out


def _optimizer_tests():
    """The family tests, generated for each optimizer so that their names start with its name."""
    def matches_torch_on_random_gradients(optim, name, kw):
        params0, grads_seq = random_problem(1900 + NAMES.index(name))
        want = oracle_trajectory(name, kw, params0, grads_seq)
        got, _, _ = run_learner(optim, name, kw, params0, grads_seq)
        compare_trajectories(name, kw, got, want, "on 20 steps of random gradients")

    def matches_torch_on_rosenbrock(optim, name):
        kw = ROSENBROCK[name]
        want = torch_rosenbrock(name, kw, (-1.2, 1.0), 100)
        w = np.array([-1.2, 1.0])
        opt = make_optimizer(optim, name, [w], kw)
        for k in range(100):
            call(opt.step, [rosenbrock_grad(*w)], expected="step() to run")
            assert_close(w, want[k], rtol=1e-9, atol=1e-12,
                         msg=f"{name}({setting_id(kw)}) on the Rosenbrock function, (x, y) after step {k + 1}")

    def follows_lr_changes(optim, name):
        params0, grads_seq = random_problem(1950 + NAMES.index(name), n_steps=8)
        lrs = [0.05, 0.02, 0.1, 0.0, 0.03, 0.08, 0.01, 0.05]
        kw = {"weight_decay": 0.1} if name != "SGD" else {}
        want = oracle_trajectory(name, kw, params0, grads_seq, lrs=lrs)
        got, _, _ = run_learner(optim, name, kw, params0, grads_seq, lrs=lrs)
        compare_trajectories(name, kw, got, want,
                             "when opt.lr changes between steps (read self.lr at every step)")

    def updates_the_arrays_in_place(optim, name):
        params = [np.ones((2, 2)), np.zeros(3)]
        ids = [id(p) for p in params]
        before = [p.copy() for p in params]
        opt = make_optimizer(optim, name, params, {"lr": 0.1})
        call(opt.step, [np.full((2, 2), 0.5), np.full(3, -0.5)], expected="step() to run")
        if opt.params is not params:
            _fail("", f"expected opt.params to be the very list that was passed, got another object "
                      "(keep the list by reference)")
        if [id(p) for p in params] != ids:
            _fail("", "expected the same array objects in the list after step(), got new arrays "
                      "(update with p -= ..., not params[i] = p - ...)")
        if all(np.array_equal(p, b) for p, b in zip(params, before)):
            _fail("", f"expected {name}.step to change the arrays of params in place, got unchanged arrays "
                      "(p = p - ... creates a new array: write p -= ...)")

    def does_not_modify_the_gradients(optim, name):
        params = [np.array([1.0, -2.0]), np.array([[0.5]])]
        grads = [np.array([0.3, -0.7]), np.array([[2.0]])]
        saved = [g.copy() for g in grads]
        opt = make_optimizer(optim, name, params, {"lr": 0.1, "weight_decay": 0.5})
        for _ in range(3):
            call(opt.step, grads, expected="step() to run")
        for i, (g, s) in enumerate(zip(grads, saved)):
            if not np.array_equal(g, s):
                _fail("", f"expected the gradient {i} unchanged after step(), got {_short(g)} instead of "
                          f"{_short(s)} (add the weight decay to a copy: g = g + wd * p, not g += wd * p)")

    def increments_t(optim, name):
        params = [np.array([1.0])]
        opt = make_optimizer(optim, name, params, {"lr": 0.1})
        if opt.t != 0:
            _fail("", f"expected t = 0 before the first step, got {opt.t!r}")
        for k in range(1, 4):
            call(opt.step, [np.array([0.5])], expected="step() to run")
            if opt.t != k:
                _fail("", f"expected t = {k} after {k} step(s), got {opt.t!r} (increment self.t once per step)")

    def refuses_wrong_number_of_gradients(optim, name, n_grads):
        params = [np.ones(2), np.ones(3)]
        opt = make_optimizer(optim, name, params, {"lr": 0.1})
        grads = [np.ones(2), np.ones(3), np.ones(1)][:n_grads]
        assert_raises(ValueError, opt.step, grads, why=f"{n_grads} gradient(s) for 2 parameters")

    def refuses_a_wrong_shape(optim, name, shape):
        params = [np.ones(3), np.ones((2, 3))]
        opt = make_optimizer(optim, name, params, {"lr": 0.1})
        assert_raises(ValueError, opt.step, [np.ones(3), np.ones(shape)],
                      why=f"a gradient of shape {shape} for a parameter of shape (2, 3) (no broadcasting)")
        if not np.array_equal(params[0], np.ones(3)):
            _fail("", "expected step() to check ALL the gradients before updating anything, got parameter 0 "
                      "already modified when gradient 1 has the wrong shape")
        if opt.t != 0:
            _fail("", f"expected t = 0 after a refused step, got {opt.t!r} (check the gradients first)")

    def two_instances_do_not_share_buffers(optim, name):
        params0, grads_seq = random_problem(1970 + NAMES.index(name), n_steps=5)
        kw = {"lr": 0.05}
        want = oracle_trajectory(name, kw, params0, grads_seq)
        a = [np.array(p, copy=True) for p in params0]
        b = [np.array(p, copy=True) for p in params0]
        opt_a = make_optimizer(optim, name, a, kw)
        opt_b = make_optimizer(optim, name, b, kw)
        for grads in grads_seq:            # interleaved: each optimizer must keep its own buffers
            call(opt_a.step, [g.copy() for g in grads], expected="step() to run")
            call(opt_b.step, [g.copy() for g in grads], expected="step() to run")
        for i, (x, y) in enumerate(zip(a, want[-1])):
            assert_close(x, y, msg=f"{name}: two optimizers stepped in turn, parameter {i} of the first "
                                   "(keep the buffers in self.state, never in a class attribute)")
        for i, (x, y) in enumerate(zip(b, want[-1])):
            assert_close(x, y, msg=f"{name}: two optimizers stepped in turn, parameter {i} of the second")

    return {
        "matches_torch_on_random_gradients": matches_torch_on_random_gradients,
        "matches_torch_on_rosenbrock": matches_torch_on_rosenbrock,
        "follows_lr_changes_between_steps": follows_lr_changes,
        "updates_the_arrays_in_place": updates_the_arrays_in_place,
        "does_not_modify_the_gradients": does_not_modify_the_gradients,
        "increments_t": increments_t,
        "refuses_wrong_number_of_gradients": refuses_wrong_number_of_gradients,
        "refuses_a_wrong_shape": refuses_a_wrong_shape,
        "two_instances_do_not_share_buffers": two_instances_do_not_share_buffers,
    }


_FAMILY = _optimizer_tests()


def _register(name):
    """Create test_<name>_... functions in this module for one optimizer."""
    prefix = "test_" + name.lower() + "_"
    g = globals()

    @pytest.mark.parametrize("kw", SETTINGS[name], ids=[setting_id(k) for k in SETTINGS[name]])
    def t1(optim, kw):
        _FAMILY["matches_torch_on_random_gradients"](optim, name, kw)

    def t2(optim):
        _FAMILY["matches_torch_on_rosenbrock"](optim, name)

    def t3(optim):
        _FAMILY["follows_lr_changes_between_steps"](optim, name)

    def t4(optim):
        _FAMILY["updates_the_arrays_in_place"](optim, name)

    def t5(optim):
        _FAMILY["does_not_modify_the_gradients"](optim, name)

    def t6(optim):
        _FAMILY["increments_t"](optim, name)

    @pytest.mark.parametrize("n_grads", [1, 3], ids=["too-few", "too-many"])
    def t7(optim, n_grads):
        _FAMILY["refuses_wrong_number_of_gradients"](optim, name, n_grads)

    @pytest.mark.parametrize("shape", [(3, 2), (2,), (1, 3)], ids=["transposed", "1d", "broadcastable"])
    def t8(optim, shape):
        _FAMILY["refuses_a_wrong_shape"](optim, name, shape)

    def t9(optim):
        _FAMILY["two_instances_do_not_share_buffers"](optim, name)

    for fn, suffix in zip((t1, t2, t3, t4, t5, t6, t7, t8, t9), _FAMILY):
        fn.__name__ = prefix + suffix
        g[fn.__name__] = fn


for _name in NAMES:
    _register(_name)


# ------------------------------------------------------------------ specific optimizer tests
def _docstring(optim, name, kw, start, grads, expected):
    w = np.array(start, dtype=float)
    opt = make_optimizer(optim, name, [w], kw)
    for g in grads:
        call(opt.step, [np.array(g, dtype=float)], expected="step() to run")
    assert_close(w, expected, rtol=1e-9, msg=f"docstring example of {name}")


def test_sgd_docstring_example(optim):
    _docstring(optim, "SGD", {"lr": 0.1}, [1.0, -2.0], [[0.5, -1.0]], [0.95, -1.9])


def test_momentum_docstring_example(optim):
    _docstring(optim, "Momentum", {"lr": 0.1, "momentum": 0.9}, [1.0], [[1.0], [1.0]], [0.71])


def test_momentum_first_buffer_is_the_gradient(optim):
    # PyTorch (dampening=0): b = g at the first step, NOT (1 - momentum) * g, NOT momentum * 0 + lr * g
    w = np.array([0.0])
    opt = make_optimizer(optim, "Momentum", [w], {"lr": 0.5, "momentum": 0.9})
    call(opt.step, [np.array([2.0])], expected="step() to run")
    assert_close(w, [-1.0], msg="Momentum(lr=0.5, momentum=0.9), first step with g = 2: p -= lr * b with b = g")


def test_nesterov_docstring_example(optim):
    _docstring(optim, "Nesterov", {"lr": 0.1, "momentum": 0.9}, [1.0], [[1.0], [1.0]], [0.539])


def test_nesterov_differs_from_momentum_by_the_look_ahead_term(optim):
    w = np.array([0.0])
    opt = make_optimizer(optim, "Nesterov", [w], {"lr": 1.0, "momentum": 0.5})
    call(opt.step, [np.array([1.0])], expected="step() to run")
    assert_close(w, [-1.5], msg="Nesterov(lr=1, momentum=0.5), first step with g = 1: p -= lr * (g + mu * b), "
                                "b = g")


def test_adagrad_docstring_example(optim):
    _docstring(optim, "Adagrad", {"lr": 0.1}, [1.0], [[2.0]], [0.9])


def test_adagrad_starts_from_initial_accumulator_value(optim):
    w = np.array([1.0])
    opt = make_optimizer(optim, "Adagrad", [w], {"lr": 0.1, "initial_accumulator_value": 5.0, "eps": 0.0})
    call(opt.step, [np.array([2.0])], expected="step() to run")
    assert_close(w, [1.0 - 0.1 * 2.0 / 3.0], msg="Adagrad(lr=0.1, initial_accumulator_value=5, eps=0), first "
                                                 "step with g = 2: s = 5 + 4")


def test_adagrad_steps_shrink_with_a_constant_gradient(optim):
    w = np.array([0.0])
    opt = make_optimizer(optim, "Adagrad", [w], {"lr": 1.0, "eps": 0.0})
    positions = []
    for _ in range(4):
        call(opt.step, [np.array([1.0])], expected="step() to run")
        positions.append(float(w[0]))
    expected = -np.cumsum(1 / np.sqrt(np.arange(1, 5)))
    assert_close(positions, expected, msg="Adagrad(lr=1, eps=0) with g = 1 four times: steps 1/sqrt(k)")


def test_rmsprop_docstring_example(optim):
    _docstring(optim, "RMSprop", {"lr": 0.01}, [1.0], [[1.0]], [1.0 - 0.01 / (0.1 + 1e-8)])


def test_rmsprop_eps_is_outside_the_square_root(optim):
    w = np.array([0.0])
    opt = make_optimizer(optim, "RMSprop", [w], {"lr": 1.0, "alpha": 0.0, "eps": 1.0})
    call(opt.step, [np.array([3.0])], expected="step() to run")
    assert_close(w, [-0.75], msg="RMSprop(lr=1, alpha=0, eps=1), g = 3: g / (sqrt(v) + eps) = 3 / (3 + 1)")


def test_adadelta_docstring_example(optim):
    _docstring(optim, "Adadelta", {}, [1.0], [[1.0]], [1.0 - math.sqrt(1e-6) / math.sqrt(0.1 + 1e-6)])


def test_adadelta_eps_is_inside_both_square_roots(optim):
    w = np.array([0.0])
    opt = make_optimizer(optim, "Adadelta", [w], {"lr": 1.0, "rho": 0.0, "eps": 1.0})
    call(opt.step, [np.array([3.0])], expected="step() to run")
    assert_close(w, [-3.0 * math.sqrt(1.0) / math.sqrt(9.0 + 1.0)],
                 msg="Adadelta(lr=1, rho=0, eps=1), g = 3, u = 0: d = sqrt(u + eps) / sqrt(v + eps) * g")


def test_adam_docstring_example(optim):
    _docstring(optim, "Adam", {"lr": 0.1}, [1.0], [[0.5]], [1.0 - 0.1 * 0.5 / (0.5 + 1e-8)])


@pytest.mark.parametrize("g", [1e-4, 3.0, -50.0], ids=["tiny", "medium", "large-negative"])
def test_adam_first_step_moves_each_entry_by_about_lr(optim, g):
    w = np.array([0.0])
    opt = make_optimizer(optim, "Adam", [w], {"lr": 0.01, "eps": 0.0})
    call(opt.step, [np.array([g])], expected="step() to run")
    assert_close(w, [-0.01 * np.sign(g)],
                 msg=f"Adam(lr=0.01, eps=0), first step with g = {g}: the bias correction makes m_hat = g and "
                     "v_hat = g**2 (use the NEW t, starting at 1)")


def test_adam_weight_decay_goes_through_the_moments(optim):
    w = np.array([2.0])
    opt = make_optimizer(optim, "Adam", [w], {"lr": 0.1, "weight_decay": 0.5, "eps": 0.0})
    call(opt.step, [np.array([0.0])], expected="step() to run")
    assert_close(w, [1.9], msg="Adam(lr=0.1, weight_decay=0.5, eps=0), zero gradient: g = 0 + 0.5 * p goes "
                               "through m and v, so the first step is about lr (coupled L2)")


def test_adamw_docstring_example(optim):
    _docstring(optim, "AdamW", {"lr": 0.1, "weight_decay": 0.01}, [1.0], [[0.0]], [0.999])


def test_adamw_decays_before_the_adam_step(optim):
    w = np.array([2.0])
    opt = make_optimizer(optim, "AdamW", [w], {"lr": 0.1, "weight_decay": 0.5, "eps": 0.0})
    call(opt.step, [np.array([1.0])], expected="step() to run")
    assert_close(w, [2.0 * (1 - 0.05) - 0.1], msg="AdamW(lr=0.1, weight_decay=0.5, eps=0), g = 1: "
                                                  "p *= 1 - lr * wd, then the Adam step on the RAW g")


def test_adamw_default_weight_decay_is_torch_default(optim):
    w = np.array([1.0])
    opt = make_optimizer(optim, "AdamW", [w], {"lr": 0.1})
    call(opt.step, [np.array([0.0])], expected="step() to run")
    assert_close(w, [1.0 - 0.1 * 1e-2], msg="AdamW(lr=0.1) with a zero gradient: only the decay acts "
                                            "(default weight_decay = 1e-2, as torch)")


# ================================================================== schedules (19.15)
def torch_schedule(make, base_lr, n):
    """The learning rates of a torch scheduler before each of n epochs."""
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([t], lr=base_lr)
    sched = make(opt)
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


@pytest.mark.parametrize("base_lr, gamma", [(0.1, 0.93), (1.0, 0.5), (0.03, 1.0)],
                         ids=["0.1x0.93", "1x0.5", "gamma-1"])
def test_exponential_decay_lr_matches_torch_exponentiallr(optim, base_lr, gamma):
    if torch is not None:
        want = torch_schedule(lambda o: torch.optim.lr_scheduler.ExponentialLR(o, gamma), base_lr, 50)
    else:
        want = [base_lr * gamma**k for k in range(50)]
    got = [call(optim.exponential_decay_lr, k, base_lr, gamma, expected="a learning rate") for k in range(50)]
    assert_close(got, want, rtol=1e-12, msg=f"exponential_decay_lr(step, {base_lr}, {gamma}) for steps 0 to 49 "
                                            "(against ExponentialLR)")


def test_exponential_decay_lr_docstring_example(optim):
    got = [call(optim.exponential_decay_lr, s, 0.1, 0.5) for s in range(4)]
    assert_close(got, [0.1, 0.05, 0.025, 0.0125], msg="exponential_decay_lr(s, 0.1, 0.5) for s = 0..3")


def test_exponential_decay_lr_returns_a_python_float(optim):
    assert_python_float(call(optim.exponential_decay_lr, 3, 0.1, 0.5), "exponential_decay_lr")


@pytest.mark.parametrize("args, why", [
    pytest.param((-1, 0.1, 0.5), "step = -1", id="negative-step"),
    pytest.param((0, -0.1, 0.5), "base_lr = -0.1", id="negative-base-lr"),
    pytest.param((0, 0.1, 0.0), "gamma = 0 (outside (0, 1])", id="gamma-0"),
    pytest.param((0, 0.1, 1.5), "gamma = 1.5 (outside (0, 1])", id="gamma-above-1"),
])
def test_exponential_decay_lr_refuses_invalid_arguments(optim, args, why):
    assert_raises(ValueError, optim.exponential_decay_lr, *args, why=why)


@pytest.mark.parametrize("base_lr, step_size, gamma", [(0.1, 7, 0.5), (0.5, 1, 0.9), (0.2, 10, 0.1)],
                         ids=["size7", "size1", "size10"])
def test_step_decay_lr_matches_torch_steplr(optim, base_lr, step_size, gamma):
    if torch is not None:
        want = torch_schedule(lambda o: torch.optim.lr_scheduler.StepLR(o, step_size, gamma), base_lr, 50)
    else:
        want = [base_lr * gamma ** (k // step_size) for k in range(50)]
    got = [call(optim.step_decay_lr, k, base_lr, step_size, gamma, expected="a learning rate") for k in range(50)]
    assert_close(got, want, rtol=1e-12, msg=f"step_decay_lr(step, {base_lr}, {step_size}, {gamma}) for steps "
                                            "0 to 49 (against StepLR)")


def test_step_decay_lr_docstring_example(optim):
    assert_close(call(optim.step_decay_lr, 9, 0.1, step_size=10, gamma=0.5), 0.1,
                 msg="step_decay_lr(9, 0.1, step_size=10, gamma=0.5)")
    assert_close(call(optim.step_decay_lr, 25, 0.1, step_size=10, gamma=0.5), 0.025,
                 msg="step_decay_lr(25, 0.1, step_size=10, gamma=0.5)")


def test_step_decay_lr_decays_exactly_at_a_multiple_of_step_size(optim):
    assert_close(call(optim.step_decay_lr, 10, 0.1, step_size=10, gamma=0.5), 0.05,
                 msg="step_decay_lr(10, 0.1, step_size=10, gamma=0.5): the first decay happens AT step 10")


def test_step_decay_lr_default_gamma_is_0_1(optim):
    assert_close(call(optim.step_decay_lr, 5, 1.0, 5), 0.1, msg="step_decay_lr(5, 1.0, 5) with the default gamma")


def test_step_decay_lr_returns_a_python_float(optim):
    assert_python_float(call(optim.step_decay_lr, 3, 0.1, 2), "step_decay_lr")


@pytest.mark.parametrize("args, why", [
    pytest.param((-1, 0.1, 5), "step = -1", id="negative-step"),
    pytest.param((0, -0.1, 5), "base_lr = -0.1", id="negative-base-lr"),
    pytest.param((0, 0.1, 0), "step_size = 0", id="step-size-0"),
    pytest.param((0, 0.1, 5, 0.0), "gamma = 0", id="gamma-0"),
    pytest.param((0, 0.1, 5, 2.0), "gamma = 2", id="gamma-above-1"),
])
def test_step_decay_lr_refuses_invalid_arguments(optim, args, why):
    assert_raises(ValueError, optim.step_decay_lr, *args, why=why)


def test_bold_driver_lr_docstring_example(optim):
    assert_close(call(optim.bold_driver_lr, 1.0, loss=0.8, prev_loss=1.0), 1.05, msg="loss went down")
    assert_close(call(optim.bold_driver_lr, 1.0, loss=1.2, prev_loss=1.0), 0.5, msg="loss went up")
    assert_close(call(optim.bold_driver_lr, 1.0, loss=1.05, prev_loss=1.0, tolerance=0.1), 1.0,
                 msg="loss went up by 5 %, inside a tolerance of 10 %")


def test_bold_driver_lr_keeps_lr_when_the_loss_is_equal(optim):
    assert_close(call(optim.bold_driver_lr, 0.2, loss=0.5, prev_loss=0.5), 0.2,
                 msg="bold_driver_lr(0.2, loss=0.5, prev_loss=0.5): neither a decrease nor an increase, lr unchanged")


def test_bold_driver_lr_tolerance_boundary_is_inside(optim):
    # 1.25 = 1.0 * (1 + 0.25) exactly: "rises by MORE than the tolerance" is false
    assert_close(call(optim.bold_driver_lr, 0.4, loss=1.25, prev_loss=1.0, tolerance=0.25), 0.4,
                 msg="bold_driver_lr(0.4, loss=1.25, prev_loss=1.0, tolerance=0.25): exactly on the band, unchanged")
    assert_close(call(optim.bold_driver_lr, 0.4, loss=1.375, prev_loss=1.0, tolerance=0.25), 0.2,
                 msg="bold_driver_lr(0.4, loss=1.375, prev_loss=1.0, tolerance=0.25): above the band, cut")


def test_bold_driver_lr_follows_a_scripted_sequence(optim):
    losses = [1.0, 0.8, 0.7, 0.75, 0.6, 0.6, 0.9, 0.5]
    increase, decrease, tol = 1.1, 0.25, 0.05
    lr, want = 0.08, []
    expected_lr = lr
    for prev, loss in zip(losses, losses[1:]):
        if loss < prev:
            expected_lr *= increase
        elif loss > prev * (1 + tol):
            expected_lr *= decrease
        want.append(expected_lr)
    got, lr = [], 0.08
    for prev, loss in zip(losses, losses[1:]):
        lr = call(optim.bold_driver_lr, lr, loss, prev, increase=increase, decrease=decrease, tolerance=tol,
                  expected="a learning rate")
        got.append(lr)
    assert_close(got, want, rtol=1e-12, msg="bold_driver_lr along the losses [1.0, 0.8, 0.7, 0.75, 0.6, 0.6, 0.9, "
                                           "0.5] (increase 1.1, decrease 0.25, tolerance 0.05)")


def test_bold_driver_lr_returns_a_python_float(optim):
    assert_python_float(call(optim.bold_driver_lr, 1.0, 0.5, 1.0), "bold_driver_lr")


@pytest.mark.parametrize("kwargs, why", [
    pytest.param({"increase": 1.0}, "increase = 1 (must be > 1)", id="increase-1"),
    pytest.param({"decrease": 1.0}, "decrease = 1 (must be in (0, 1))", id="decrease-1"),
    pytest.param({"decrease": 0.0}, "decrease = 0", id="decrease-0"),
    pytest.param({"tolerance": -0.1}, "tolerance = -0.1", id="negative-tolerance"),
])
def test_bold_driver_lr_refuses_invalid_arguments(optim, kwargs, why):
    assert_raises(ValueError, optim.bold_driver_lr, 1.0, 0.5, 1.0, why=why, **kwargs)


# ================================================================== cosine_lr and clip_grad_norm (19.19)
def cosine_formula(step, total, base, warmup=0, min_lr=0.0):
    """The closed form of the docstring, in NumPy (used when torch and transformers are absent)."""
    if step < warmup:
        return base * step / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + (base - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


@pytest.mark.parametrize("warmup, total", [(10, 60), (3, 7), (0, 25)], ids=["10of60", "3of7", "no-warmup"])
def test_cosine_lr_matches_hugging_face_with_warmup(optim, warmup, total):
    if get_cosine_schedule_with_warmup is not None and torch is not None:
        want = torch_schedule(lambda o: get_cosine_schedule_with_warmup(o, warmup, total), 0.1, total + 1)
    else:
        want = [cosine_formula(k, total, 0.1, warmup) for k in range(total + 1)]
    got = [call(optim.cosine_lr, k, total, 0.1, warmup_steps=warmup, expected="a learning rate")
           for k in range(total + 1)]
    assert_close(got, want, rtol=1e-12, atol=1e-15,
                 msg=f"cosine_lr(step, {total}, 0.1, warmup_steps={warmup}) for steps 0 to {total} (against "
                     "transformers.get_cosine_schedule_with_warmup)")


@pytest.mark.parametrize("total, min_lr", [(40, 0.01), (13, 0.0)], ids=["min0.01", "min0"])
def test_cosine_lr_matches_torch_cosineannealinglr(optim, total, min_lr):
    if torch is not None:
        want = torch_schedule(lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=total, eta_min=min_lr),
                              0.1, total + 1)
    else:
        want = [cosine_formula(k, total, 0.1, 0, min_lr) for k in range(total + 1)]
    got = [call(optim.cosine_lr, k, total, 0.1, min_lr=min_lr, expected="a learning rate") for k in range(total + 1)]
    assert_close(got, want, rtol=1e-12, atol=1e-15,
                 msg=f"cosine_lr(step, {total}, 0.1, min_lr={min_lr}) for steps 0 to {total} (against "
                     "CosineAnnealingLR)")


def test_cosine_lr_docstring_example(optim):
    got = [call(optim.cosine_lr, s, 100, 0.1, warmup_steps=10) for s in (0, 5, 10, 55, 100)]
    assert_close(got, [0.0, 0.05, 0.1, 0.05, 0.0], atol=1e-15,
                 msg="cosine_lr(s, 100, 0.1, warmup_steps=10) for s in (0, 5, 10, 55, 100)")


def test_cosine_lr_stays_at_min_lr_after_total_steps(optim):
    got = [call(optim.cosine_lr, s, 50, 0.1, warmup_steps=5, min_lr=0.002) for s in (50, 51, 75, 100, 1000)]
    assert_close(got, [0.002] * 5, msg="cosine_lr(s, 50, 0.1, warmup_steps=5, min_lr=0.002) for s >= 50: no "
                                       "restart (torch and Hugging Face would climb again)")


def test_cosine_lr_warmup_end_and_middle_values(optim):
    assert_close(call(optim.cosine_lr, 4, 20, 0.8, warmup_steps=4, min_lr=0.2), 0.8,
                 msg="cosine_lr(4, 20, 0.8, warmup_steps=4, min_lr=0.2): end of the warmup, the peak")
    assert_close(call(optim.cosine_lr, 12, 20, 0.8, warmup_steps=4, min_lr=0.2), 0.5,
                 msg="cosine_lr(12, 20, 0.8, warmup_steps=4, min_lr=0.2): half-way through the cosine, "
                     "(base + min) / 2")
    assert_close(call(optim.cosine_lr, 1, 20, 0.8, warmup_steps=4, min_lr=0.2), 0.2,
                 msg="cosine_lr(1, 20, 0.8, warmup_steps=4, min_lr=0.2): the warmup goes from 0 to base_lr "
                     "(it ignores min_lr)")


def test_cosine_lr_returns_a_python_float(optim):
    assert_python_float(call(optim.cosine_lr, 3, 10, 0.1), "cosine_lr")


@pytest.mark.parametrize("args, kwargs, why", [
    pytest.param((5, 10, 0.1), {"warmup_steps": 10}, "total_steps = warmup_steps = 10", id="total-equals-warmup"),
    pytest.param((5, 10, 0.1), {"warmup_steps": 12}, "warmup_steps = 12 > total_steps = 10", id="warmup-too-long"),
    pytest.param((-1, 10, 0.1), {}, "step = -1", id="negative-step"),
    pytest.param((0, 10, 0.1), {"warmup_steps": -2}, "warmup_steps = -2", id="negative-warmup"),
    pytest.param((0, 10, 0.1), {"min_lr": 0.2}, "min_lr = 0.2 > base_lr = 0.1", id="min-above-base"),
])
def test_cosine_lr_refuses_invalid_arguments(optim, args, kwargs, why):
    assert_raises(ValueError, optim.cosine_lr, *args, why=why, **kwargs)


def torch_clip(grads, max_norm):
    tensors = []
    for g in grads:
        t = torch.zeros(np.shape(g), dtype=torch.float64, requires_grad=True)
        t.grad = torch.tensor(np.array(g, dtype=float), dtype=torch.float64)
        tensors.append(t)
    norm = torch.nn.utils.clip_grad_norm_(tensors, max_norm)
    return float(norm), [t.grad.numpy().copy() for t in tensors]


def numpy_clip(grads, max_norm, eps=1e-6):
    norm = float(np.sqrt(sum(np.sum(np.square(g)) for g in grads)))
    factor = max_norm / (norm + eps) if norm > max_norm else 1.0
    return norm, [np.array(g, dtype=float) * factor for g in grads]


def clip_learner(optim, grads, max_norm, **kwargs):
    work = [np.array(g, dtype=float, copy=True) for g in grads]
    norm = call(optim.clip_grad_norm, work, max_norm, expected="the norm before clipping", **kwargs)
    assert_python_float(norm, "clip_grad_norm")
    return norm, work


@pytest.mark.parametrize("max_norm", [1.0, 0.25, 5.0], ids=["1", "0.25", "5"])
def test_clip_grad_norm_matches_torch(optim, max_norm):
    rng = np.random.default_rng(1919)
    grads = [rng.normal(size=s) for s in [(3, 2), (4,), ()]]  # global norm ~ 2.8
    want_norm, want = torch_clip(grads, max_norm) if torch is not None else numpy_clip(grads, max_norm)
    norm, got = clip_learner(optim, grads, max_norm)
    assert_close(norm, want_norm, rtol=1e-12, msg=f"clip_grad_norm(..., {max_norm}): the returned GLOBAL norm "
                                                  "(before clipping, over all the arrays)")
    for i, (x, y) in enumerate(zip(got, want)):
        assert_close(x, y, rtol=1e-12, msg=f"clip_grad_norm(..., {max_norm}): gradient {i} after clipping "
                                           "(against torch.nn.utils.clip_grad_norm_)")


def test_clip_grad_norm_docstring_example(optim):
    norm, got = clip_learner(optim, [np.array([3.0, 4.0])], 1.0)
    assert_close(norm, 5.0, msg="clip_grad_norm([[3, 4]], max_norm=1.0): returned norm")
    assert_close(got[0], [0.59999988, 0.79999984], rtol=1e-7, msg="clip_grad_norm([[3, 4]], max_norm=1.0): "
                                                                  "the gradient, rescaled in place")


def test_clip_grad_norm_uses_one_global_norm(optim):
    grads = [np.array([0.6, 0.0]), np.array([0.0, 0.8])]  # each norm < 1, global norm exactly 1
    norm, got = clip_learner(optim, grads, 0.5)
    assert_close(norm, 1.0, msg="clip_grad_norm of [0.6, 0] and [0, 0.8]: the norm of all the entries together, "
                                "not one norm per array")
    factor = 0.5 / (1.0 + 1e-6)
    assert_close(got[0], [0.6 * factor, 0.0], msg="clip_grad_norm(..., 0.5): array 0, the same factor for all")
    assert_close(got[1], [0.0, 0.8 * factor], msg="clip_grad_norm(..., 0.5): array 1, the same factor for all")


def test_clip_grad_norm_leaves_small_gradients_unchanged(optim):
    grads = [np.array([0.3, -0.4]), np.array([[0.1]])]
    norm, got = clip_learner(optim, grads, 1.0)
    assert_close(norm, math.sqrt(0.26), msg="clip_grad_norm with a global norm of about 0.51 < 1: returned norm")
    for i, (x, y) in enumerate(zip(got, grads)):
        if not np.array_equal(x, y):
            _fail("", f"expected gradient {i} unchanged (norm 0.51 <= max_norm = 1), got {_short(x)} instead of "
                      f"{_short(y)}")


def test_clip_grad_norm_works_in_place(optim):
    grads = [np.array([30.0, 40.0]), np.array([0.0])]
    ids = [id(g) for g in grads]
    call(optim.clip_grad_norm, grads, 1.0, expected="the norm before clipping")
    if [id(g) for g in grads] != ids:
        _fail("", "expected the same array objects after clip_grad_norm, got new arrays (multiply in place: g *= ...)")
    assert_close(grads[0], [30.0 / (50 + 1e-6), 40.0 / (50 + 1e-6)],
                 msg="clip_grad_norm([[30, 40], [0]], 1.0): the caller's array must be rescaled in place")


def test_clip_grad_norm_uses_its_eps(optim):
    norm, got = clip_learner(optim, [np.array([6.0, 8.0])], 2.0, eps=0.5)
    assert_close(norm, 10.0, msg="clip_grad_norm([[6, 8]], 2.0, eps=0.5): returned norm")
    assert_close(got[0], [6.0 * 2.0 / 10.5, 8.0 * 2.0 / 10.5],
                 msg="clip_grad_norm([[6, 8]], 2.0, eps=0.5): factor max_norm / (norm + eps)")


def test_clip_grad_norm_of_zero_gradients(optim):
    norm, got = clip_learner(optim, [np.zeros(3), np.zeros((2, 2))], 1.0)
    assert_close(norm, 0.0, msg="clip_grad_norm of zero gradients: returned norm")
    if any(np.any(g != 0) or np.any(np.isnan(g)) for g in got):
        _fail("", "expected zero gradients to stay zero (no division by a zero norm)")


@pytest.mark.parametrize("max_norm", [0.0, -1.0], ids=["zero", "negative"])
def test_clip_grad_norm_refuses_a_non_positive_max_norm(optim, max_norm):
    assert_raises(ValueError, optim.clip_grad_norm, [np.ones(2)], max_norm, why=f"max_norm = {max_norm}")
