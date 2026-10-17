"""Micro-autograd — mylearn, chapter 18 (Backpropagation).

A scalar automatic-differentiation engine in the spirit of micrograd
(A. Karpathy). Each ``Value`` remembers the Values it was computed from and
the operation that produced it; ``backward()`` walks this graph in reverse
topological order (see ``topological_order`` in chapter 16) and applies the
chain rule. It is the only place in mylearn where the gradient graph is built
automatically, as PyTorch does. ``MiniMLP`` assembles Values into neurons and
layers to train a small network.

Reference implementation.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Sequence

import numpy as np


def _as_value(other) -> Value:
    """Wrap a plain number in a Value; refuse anything else with a TypeError."""
    if isinstance(other, Value):
        return other
    if isinstance(other, numbers.Real):
        return Value(float(other))
    raise TypeError(f"unsupported operand for Value: {type(other).__name__} "
                    "(use a Value, an int or a float)")


class Value:
    """A scalar that records the operations applied to it, for reverse-mode autodiff.

    Build expressions with ``+``, ``-``, ``*``, ``/``, unary ``-`` and ``**``
    (a Value or an int/float on either side, except for ``**`` whose exponent
    must be an int or a float) and with the methods ``exp``, ``log``, ``tanh``,
    ``relu`` and ``sigmoid``. Each operation returns a new Value whose parents
    are its operands. Then ``out.backward()`` fills the ``grad`` attribute of
    every Value the result depends on.

    Parameters
    ----------
    data : float
        The scalar value (stored as a Python float).
    _children : tuple of Value, default=()
        Internal: the Values this one was computed from (its parents in the
        graph). Leave empty for a leaf (an input or a parameter).
    _op : str, default=""
        Internal: name of the operation that produced this Value, for example
        ``"+"``, ``"*"`` or ``"tanh"`` (for display and debugging only).
    label : str, default=""
        Optional name for display, for example ``"w1"``.

    Attributes
    ----------
    data : float
        The value.
    grad : float
        d(output)/d(this Value) after ``output.backward()``; 0.0 before.
    label : str
        Display name.
    _prev : tuple of Value
        The parents (``_children`` as given; a Value used twice, as in
        ``a * a``, appears twice).
    _op : str
        Name of the producing operation.
    _backward : callable with no argument
        Function that propagates ``self.grad`` to the parents' ``grad``. It
        does nothing for a leaf; each operation sets it on the Value it
        creates (a closure over the operands).

    Notes
    -----
    Values are hashed and compared by identity (no ``__eq__``), so they can be
    stored in a ``set``. Tested against PyTorch scalar tensors (float64,
    ``requires_grad=True``): same ``data`` and same ``.grad`` on random
    expression graphs, including a node reused several times (its gradients
    must ACCUMULATE).

    Examples
    --------
    >>> a = Value(2.0)
    >>> b = Value(-3.0)
    >>> c = a * b + a        # a is used twice: its gradients add up
    >>> c.backward()
    >>> c.data, a.grad, b.grad
    (-4.0, -2.0, 2.0)
    >>> (1 - a / 4).data     # plain numbers work on either side
    0.5
    """

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        """Return ``"Value(data=..., grad=...)"``, for example ``Value(data=2.0, grad=0.0)``.

        Examples
        --------
        >>> Value(2.0)
        Value(data=2.0, grad=0.0)
        """
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Value | float) -> Value:
        """Return ``self + other``.

        Raises
        ------
        TypeError
            If ``other`` is neither a Value nor an int or a float.
        """
        other = _as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Value:
        """Return ``other + self`` when the left operand is a plain number."""
        return self + other

    def __neg__(self) -> Value:
        """Return ``-self``."""
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        """Return ``self - other``.

        Raises
        ------
        TypeError
            If ``other`` is neither a Value nor an int or a float.
        """
        return self + (-_as_value(other))

    def __rsub__(self, other: float) -> Value:
        """Return ``other - self`` when the left operand is a plain number."""
        return _as_value(other) + (-self)

    def __mul__(self, other: Value | float) -> Value:
        """Return ``self * other``.

        Raises
        ------
        TypeError
            If ``other`` is neither a Value nor an int or a float.
        """
        other = _as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> Value:
        """Return ``other * self`` when the left operand is a plain number."""
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        """Return ``self / other``.

        Raises
        ------
        TypeError
            If ``other`` is neither a Value nor an int or a float.
        """
        return self * _as_value(other) ** -1

    def __rtruediv__(self, other: float) -> Value:
        """Return ``other / self`` when the left operand is a plain number."""
        return _as_value(other) * self ** -1

    def __pow__(self, exponent: float) -> Value:
        """Return ``self ** exponent`` for a constant exponent.

        Raises
        ------
        TypeError
            If ``exponent`` is not an int or a float (``Value ** Value`` is not
            supported).
        """
        if isinstance(exponent, bool) or not isinstance(exponent, numbers.Real):
            raise TypeError(f"the exponent must be an int or a float, got {type(exponent).__name__} "
                            "(Value ** Value is not supported)")
        exponent = float(exponent)
        out = Value(self.data ** exponent, (self,), f"**{exponent:g}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1.0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Return exp(self)."""
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Return the natural logarithm of self.

        Raises
        ------
        ValueError
            If ``self.data <= 0``.
        """
        if self.data <= 0.0:
            raise ValueError(f"log of a non-positive value ({self.data})")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Return tanh(self)."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """Return max(0, self); its derivative at 0 is 0, as in PyTorch."""
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        """Return 1 / (1 + exp(-self)), without overflow for large |self.data|."""
        if self.data >= 0.0:
            s = 1.0 / (1.0 + math.exp(-self.data))
        else:
            e = math.exp(self.data)
            s = e / (1.0 + e)
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Compute d(self)/d(node) for every node of the graph that leads to self.

        Set ``self.grad = 1.0``, order the graph topologically (every node after
        the Values it was computed from), then call each node's ``_backward``
        in reverse order. Gradients ACCUMULATE (``+=``): a node used several
        times receives the sum of its contributions. ``backward`` does not
        reset the other gradients: set them to 0 before each new forward pass
        (as ``optimizer.zero_grad()`` in PyTorch).

        Returns
        -------
        None
            The gradients are stored in the ``grad`` attribute of each node.

        Notes
        -----
        Tested against ``.backward()`` of PyTorch on the same expression.

        Examples
        --------
        >>> x = Value(3.0)
        >>> y = x * x + 2 * x     # dy/dx = 2x + 2
        >>> y.backward()
        >>> x.grad
        8.0
        """
        order: list[Value] = []
        visited: set[int] = set()
        # iterative depth-first search: a node goes into `order` after all its parents
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


class MiniMLP:
    """A tiny multilayer perceptron made of Value neurons.

    Hidden layers apply ``activation``; the last layer is linear (it returns
    logits or regression outputs). Weights follow the mylearn convention: in
    layer ``l``, ``weights_[l][i][j]`` connects input ``i`` to neuron ``j``, so
    each layer computes ``z_j = sum_i x_i * W[i][j] + b_j``.

    ``__init__`` is already written: it checks the hyperparameters and
    creates the parameters as Values (weights drawn from U(-1/sqrt(fan_in),
    1/sqrt(fan_in)), the scale of ``torch.nn.Linear``; biases at 0).

    Parameters
    ----------
    n_in : int
        Number of inputs, >= 1.
    layer_sizes : sequence of int
        Width of each layer; the last one is the number of outputs.
        At least one layer, all widths >= 1.
    activation : {"relu", "tanh", "sigmoid", "linear"}, default="relu"
        Activation of the hidden layers (``"linear"``: none).
    random_state : int or None, default=None
        Seed of the weight initialisation.

    Attributes
    ----------
    weights_ : list of list of list of Value
        ``weights_[l]`` is a (fan_in x fan_out) nested list of Values.
    biases_ : list of list of Value
        ``biases_[l]`` holds the fan_out biases of layer ``l``.

    Raises
    ------
    ValueError
        For an unknown activation or an invalid size (in ``__init__``), or an
        input of the wrong length (in ``__call__``).

    Notes
    -----
    Tested against a ``torch.nn.Sequential`` loaded with ``to_params()``: same
    outputs and same gradients; ``mylearn.nn.layers.mlp_forward`` on
    ``to_params()`` gives the same outputs.

    Examples
    --------
    >>> model = MiniMLP(2, [3, 1], activation="tanh", random_state=0)
    >>> [W.shape for W, b in model.to_params()]
    [(2, 3), (3, 1)]
    >>> len(model.parameters())
    13
    >>> out = model([1.0, -1.0])
    >>> isinstance(out, Value)
    True
    >>> out.backward()
    >>> model.zero_grad()
    >>> all(p.grad == 0.0 for p in model.parameters())
    True
    """

    def __init__(
        self,
        n_in: int,
        layer_sizes: Sequence[int],
        activation: str = "relu",
        random_state: int | None = None,
    ) -> None:
        if activation not in ("relu", "tanh", "sigmoid", "linear"):
            raise ValueError(
                f"unknown activation {activation!r}: use 'relu', 'tanh', 'sigmoid' or 'linear'"
            )
        sizes = [int(n_in), *(int(s) for s in layer_sizes)]
        if len(sizes) < 2 or min(sizes) < 1:
            raise ValueError("n_in and every layer size must be >= 1, with at least one layer")
        self.n_in = n_in
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        self.weights_: list[list[list[Value]]] = []
        self.biases_: list[list[Value]] = []
        for layer, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            bound = 1.0 / np.sqrt(fan_in)
            W = rng.uniform(-bound, bound, size=(fan_in, fan_out))
            self.weights_.append(
                [[Value(W[i, j], label=f"W{layer}[{i},{j}]") for j in range(fan_out)]
                 for i in range(fan_in)]
            )
            self.biases_.append([Value(0.0, label=f"b{layer}[{j}]") for j in range(fan_out)])

    def __call__(self, x: Sequence[float | Value]) -> Value | list[Value]:
        """Run the forward pass on ONE sample.

        Parameters
        ----------
        x : sequence of float or Value, of length n_in
            The input sample. Values are allowed (to get gradients with
            respect to the input).

        Returns
        -------
        Value or list of Value
            A single Value when the last layer has one neuron, else a list of
            its outputs.

        Raises
        ------
        ValueError
            If ``len(x) != n_in``.
        """
        if len(x) != self.n_in:
            raise ValueError(f"the input has {len(x)} values, the model expects n_in = {self.n_in}")
        values = [v if isinstance(v, Value) else Value(float(v)) for v in x]
        last = len(self.weights_) - 1
        for layer, (W, b) in enumerate(zip(self.weights_, self.biases_, strict=True)):
            outputs = []
            for j in range(len(b)):
                z = b[j]
                for i, v in enumerate(values):
                    z = z + v * W[i][j]
                if layer < last and self.activation != "linear":
                    z = getattr(z, self.activation)()
                outputs.append(z)
            values = outputs
        return values[0] if len(values) == 1 else values

    def parameters(self) -> list[Value]:
        """Return every parameter as a flat list of Values.

        Order: layer by layer; within a layer, the weights row by row
        (``weights_[l][0][0], weights_[l][0][1], ...``), then the biases.

        Returns
        -------
        list of Value
            ``mylearn.nn.layers.count_parameters([n_in, *layer_sizes])`` Values,
            the objects themselves (not copies), so that updating their
            ``data`` trains the model.
        """
        params: list[Value] = []
        for W, b in zip(self.weights_, self.biases_, strict=True):
            for row in W:
                params.extend(row)
            params.extend(b)
        return params

    def zero_grad(self) -> None:
        """Set the ``grad`` of every parameter to 0.0 (call it before each backward)."""
        for p in self.parameters():
            p.grad = 0.0

    def to_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Export the current parameter values as NumPy arrays.

        Returns
        -------
        list of (W, b) tuples, one per layer
            ``W`` of shape (fan_in, fan_out) and ``b`` of shape (fan_out,),
            float64 copies of the ``data`` of the Values, ready for
            ``mylearn.nn.layers.mlp_forward``.
        """
        return [(np.array([[w.data for w in row] for row in W], dtype=np.float64),
                 np.array([v.data for v in b], dtype=np.float64))
                for W, b in zip(self.weights_, self.biases_, strict=True)]
