"""The model of minifw: a Sequential stack of layers, with chained forward and backward passes.

    model = Sequential(Dense(784, 256, rng=rng), BatchNorm1d(256), ReLU(), Dropout(0.2, rng=rng),
                       Dense(256, 10, rng=rng))
    logits = model(X)                       # forward, layer after layer
    loss, dlogits = SoftmaxCrossEntropy()(logits, y)
    model.backward(dlogits)                 # backward, from the last layer to the first
    optimizer = mylearn.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step(model.gradients())       # the arrays of the layers are updated in place
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
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Provided: model(x) is model.forward(x)."""
        return self.forward(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagation: dout goes through the layers in REVERSE order; returns dL/d(input of the model)."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self) -> list[np.ndarray]:
        """The parameter arrays of all the layers, in order (the arrays themselves: an optimizer updates them)."""
        return [p for layer in self.layers for p in layer.parameters()]

    def gradients(self) -> list[np.ndarray]:
        """The gradients of the last backward pass, in the order of parameters()."""
        return [g for layer in self.layers for g in layer.gradients()]

    def train(self) -> Sequential:
        """Training mode for the model and every layer (Dropout draws masks, BatchNorm1d uses the batch); returns self."""
        self.training = True
        for layer in self.layers:
            layer.training = True
        return self

    def eval(self) -> Sequential:
        """Evaluation mode for the model and every layer (Dropout is the identity, BatchNorm1d uses its running
        statistics); returns self."""
        self.training = False
        for layer in self.layers:
            layer.training = False
        return self

    def n_parameters(self) -> int:
        """The number of learned values (the sizes of all the parameter arrays), as an int."""
        return int(sum(p.size for p in self.parameters()))

    def state_dict(self) -> dict[str, np.ndarray]:
        """Deep copies of the parameters AND of the buffers (running_mean, running_var), by name:
        "0.W", "0.b", "1.gamma", "1.running_mean"... (the index of the layer, a dot, the attribute)."""
        state = {}
        for i, layer in enumerate(self.layers):
            for name in ("W", "b", "gamma", "beta", "running_mean", "running_var"):
                if hasattr(layer, name):
                    state[f"{i}.{name}"] = copy.deepcopy(getattr(layer, name))
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Copy the arrays of state into the layers, IN PLACE (the optimizer keeps its references); the names and
        the shapes must match those of state_dict()."""
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError(f"the names differ: expected {sorted(expected)}, got {sorted(state)}")
        for key, value in state.items():
            i, name = key.split(".")
            target = getattr(self.layers[int(i)], name)
            if np.shape(value) != target.shape:
                raise ValueError(f"{key}: shape {np.shape(value)} instead of {target.shape}")
            target[...] = value
