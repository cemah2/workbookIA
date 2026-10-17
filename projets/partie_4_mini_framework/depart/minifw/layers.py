"""The layers of minifw: Dense, the activations, Dropout and BatchNorm1d, each with forward and backward.

Every layer follows the same small contract, the one the model (``model.py``) and the
optimizers of mylearn rely on:

- ``forward(x)`` computes the output of a mini-batch ``x`` of shape ``(n, n_in)`` and keeps
  in the layer what ``backward`` will need (the cache);
- ``backward(dout)`` receives dL/d(output), stores the gradients of the parameters in the
  layer and returns dL/d(input);
- ``parameters()`` and ``gradients()`` return the lists of the parameter arrays and of their
  gradients, in the same order. ``parameters()`` returns the arrays themselves (an optimizer
  updates them in place), never copies;
- ``training`` is True in training mode and False in evaluation mode (``Sequential.train()``
  and ``Sequential.eval()`` set it); only Dropout and BatchNorm1d behave differently.

The computations come from your mylearn library (chapters 16 to 20); this module only
wraps them in objects. All the arrays are float64.

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import numpy as np

from mylearn.nn.activations import get_activation
from mylearn.nn.backward import dense_backward
from mylearn.nn.layers import dense_forward, init_weights
from mylearn.nn.regularization import batchnorm_backward, batchnorm_forward, dropout_backward, dropout_forward


class Layer:
    """Base class: a layer without parameters, in training mode."""

    def __init__(self) -> None:
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Provided (abstract): the output of the mini-batch x, and the cache that backward needs."""
        raise NotImplementedError(f"{type(self).__name__}.forward")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Provided (abstract): dL/d(input) from dL/d(output), after the forward pass of the same mini-batch."""
        raise NotImplementedError(f"{type(self).__name__}.backward")

    def parameters(self) -> list[np.ndarray]:
        """Provided: the parameter arrays of the layer (the arrays themselves, not copies); [] without parameters."""
        return []

    def gradients(self) -> list[np.ndarray]:
        """Provided: the gradients of the last backward pass, in the order of parameters(); [] without parameters."""
        return []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Provided: layer(x) is layer.forward(x)."""
        return self.forward(x)


class Dense(Layer):
    """A fully connected layer z = x @ W + b, with W of shape (n_in, n_out) (the convention of chapter 16).

    W is drawn by ``mylearn.nn.layers.init_weights(n_in, n_out, init, rng)`` and b starts at zero.
    After ``backward``, ``dW`` and ``db`` hold the gradients of the mean loss.
    """

    def __init__(self, n_in: int, n_out: int, init: str = "he_normal", rng: np.random.Generator | None = None) -> None:
        super().__init__()
        self.W = np.asarray(init_weights(n_in, n_out, init, rng=rng), dtype=np.float64)
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """z = x @ W + b, for x of shape (n, n_in); keeps x for backward."""
        raise NotImplementedError("Dense.forward")  # TODO MP4.2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Stores dW = x.T @ dout and db = dout.sum(axis=0) (in dW and db, in place), and returns dout @ W.T."""
        raise NotImplementedError("Dense.backward")  # TODO MP4.2

    def parameters(self) -> list[np.ndarray]:
        """[W, b], the arrays themselves."""
        raise NotImplementedError("Dense.parameters")  # TODO MP4.2

    def gradients(self) -> list[np.ndarray]:
        """[dW, db]."""
        raise NotImplementedError("Dense.gradients")  # TODO MP4.2


class Activation(Layer):
    """An elementwise activation of mylearn, by name ("relu", "tanh", "gelu", ...): ``get_activation(name)``.

    backward multiplies dout by the derivative at the PRE-activation z kept by forward
    (the convention of chapters 17 and 18).
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.f, self.f_derivative = get_activation(name)
        self._z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x), elementwise; keeps x (the pre-activation) for backward."""
        raise NotImplementedError("Activation.forward")  # TODO MP4.2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout * f'(z)."""
        raise NotImplementedError("Activation.backward")  # TODO MP4.2


class ReLU(Activation):
    """Activation("relu")."""

    def __init__(self) -> None:
        super().__init__("relu")


class Tanh(Activation):
    """Activation("tanh")."""

    def __init__(self) -> None:
        super().__init__("tanh")


class Dropout(Layer):
    """Inverted dropout (``mylearn.nn.regularization.dropout_forward``): in training mode, each value is set to zero
    with probability p and the others are divided by 1 - p, with a new mask at every forward pass, drawn from rng;
    in evaluation mode, the identity."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p
        self.rng = np.random.default_rng() if rng is None else rng
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x * mask in training mode (a new mask), x itself (a copy) in evaluation mode."""
        raise NotImplementedError("Dropout.forward")  # TODO MP4.2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout * mask: the mask of the last forward pass (dout itself, copied, in evaluation mode)."""
        raise NotImplementedError("Dropout.backward")  # TODO MP4.2


class BatchNorm1d(Layer):
    """Batch normalization of each feature, like ``torch.nn.BatchNorm1d(n_features)``
    (``mylearn.nn.regularization.batchnorm_forward``).

    In training mode, the statistics of the mini-batch, and the running statistics updated in
    place (momentum 0.1, unbiased variance); in evaluation mode, the running statistics.
    gamma starts at 1, beta at 0, running_mean at 0 and running_var at 1. ``dgamma`` and
    ``dbeta`` hold the gradients after backward. The running statistics are buffers, not
    parameters: the optimizer never sees them, but ``Sequential.state_dict`` saves them.
    """

    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.dgamma = np.zeros(n_features)
        self.dbeta = np.zeros(n_features)
        self.momentum = momentum
        self.eps = eps
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """The normalized mini-batch, scaled by gamma and shifted by beta (see the class docstring)."""
        raise NotImplementedError("BatchNorm1d.forward")  # TODO MP4.2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Stores dgamma and dbeta (in place) and returns dL/dx, through the batch statistics in training mode."""
        raise NotImplementedError("BatchNorm1d.backward")  # TODO MP4.2

    def parameters(self) -> list[np.ndarray]:
        """[gamma, beta], the arrays themselves."""
        raise NotImplementedError("BatchNorm1d.parameters")  # TODO MP4.2

    def gradients(self) -> list[np.ndarray]:
        """[dgamma, dbeta]."""
        raise NotImplementedError("BatchNorm1d.gradients")  # TODO MP4.2
