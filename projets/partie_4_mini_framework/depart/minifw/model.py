"""The model of minifw: a Sequential stack of layers, with chained forward and backward passes.

    model = Sequential(Dense(784, 256, rng=rng), BatchNorm1d(256), ReLU(), Dropout(0.2, rng=rng),
                       Dense(256, 10, rng=rng))
    logits = model(X)                       # forward, layer after layer
    loss, dlogits = SoftmaxCrossEntropy()(logits, y)
    model.backward(dlogits)                 # backward, from the last layer to the first
    optimizer = mylearn.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step(model.gradients())       # the arrays of the layers are updated in place

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import copy

import numpy as np

from .layers import Layer


class Sequential:
    """A stack of layers applied in order. Its parameters are those of its layers, in order."""

    def __init__(self, *layers: Layer) -> None:
        if not layers:
            raise ValueError("a Sequential needs at least one layer")
        self.layers = list(layers)
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """The output of the last layer: each layer receives the output of the previous one."""
        raise NotImplementedError("Sequential.forward")  # TODO MP4.3

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Provided: model(x) is model.forward(x)."""
        return self.forward(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagation: dout goes through the layers in REVERSE order; returns dL/d(input of the model)."""
        raise NotImplementedError("Sequential.backward")  # TODO MP4.3

    def parameters(self) -> list[np.ndarray]:
        """The parameter arrays of all the layers, in order (the arrays themselves: an optimizer updates them)."""
        raise NotImplementedError("Sequential.parameters")  # TODO MP4.3

    def gradients(self) -> list[np.ndarray]:
        """The gradients of the last backward pass, in the order of parameters()."""
        raise NotImplementedError("Sequential.gradients")  # TODO MP4.3

    def train(self) -> Sequential:
        """Training mode for the model and every layer (Dropout draws masks, BatchNorm1d uses the batch); returns self."""
        raise NotImplementedError("Sequential.train")  # TODO MP4.3

    def eval(self) -> Sequential:
        """Evaluation mode for the model and every layer (Dropout is the identity, BatchNorm1d uses its running
        statistics); returns self."""
        raise NotImplementedError("Sequential.eval")  # TODO MP4.3

    def n_parameters(self) -> int:
        """The number of learned values (the sizes of all the parameter arrays), as an int."""
        raise NotImplementedError("Sequential.n_parameters")  # TODO MP4.3

    def state_dict(self) -> dict[str, np.ndarray]:
        """Deep copies of the parameters AND of the buffers (running_mean, running_var), by name:
        "0.W", "0.b", "1.gamma", "1.running_mean"... (the index of the layer, a dot, the attribute)."""
        raise NotImplementedError("Sequential.state_dict")  # TODO MP4.3

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Copy the arrays of state into the layers, IN PLACE (the optimizer keeps its references); the names and
        the shapes must match those of state_dict()."""
        raise NotImplementedError("Sequential.load_state_dict")  # TODO MP4.3
