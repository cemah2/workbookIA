"""Dense layers — mylearn, chapter 16 (Feed-Forward Networks).

Fully connected layers in pure NumPy: parameter counting, weight
initialisation (LeCun, Glorot, He...), the forward pass of one layer and of a
whole MLP (with the cache that backpropagation will need in chapter 18), and
the evaluation order of a network graph.

Convention used in the whole of Part IV: weights ``W`` of shape
``(n_in, n_out)``, one sample per row, ``z = x @ W + b`` (like ``coefs_`` in
scikit-learn; ``torch.nn.Linear`` stores the transpose ``W.T``).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def count_parameters(layer_sizes: Sequence[int], bias: bool = True) -> int:
    """Count the trainable parameters of a fully connected network.

    Layer ``l`` connects ``n_(l-1)`` inputs to ``n_l`` neurons: it has
    ``n_(l-1) * n_l`` weights, plus ``n_l`` biases when ``bias`` is True.
    The total is P = sum over l of (n_(l-1) * n_l + n_l).

    Parameters
    ----------
    layer_sizes : sequence of int
        ``[n_0, n_1, ..., n_L]``: ``n_0`` is the number of input features, then
        the width of each layer (the last one is the output layer). At least 2
        entries, all >= 1.
    bias : bool, default=True
        Whether each neuron has a bias.

    Returns
    -------
    int
        Total number of weights (and biases).

    Raises
    ------
    ValueError
        If ``layer_sizes`` has fewer than 2 entries or a size is < 1.

    Notes
    -----
    Tested against ``sum(p.numel() for p in model.parameters())`` for the
    equivalent ``torch.nn.Sequential`` of ``torch.nn.Linear`` layers.

    Examples
    --------
    >>> count_parameters([2, 4, 4, 1])
    37
    >>> count_parameters([2, 4, 4, 1], bias=False)
    28
    >>> count_parameters([784, 256, 128, 10])
    235146
    """
    raise NotImplementedError("count_parameters() is not implemented yet")


def dense_forward(x: ArrayLike, W: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Compute the affine map of a fully connected layer, z = x @ W + b.

    Each row of ``x`` is one sample; the bias vector is broadcast over the rows.

    Parameters
    ----------
    x : array-like of shape (n, n_in) or (n_in,)
        Inputs: one sample per row, or a single sample as a 1-D array.
    W : ndarray of shape (n_in, n_out)
        Weights: column ``j`` holds the weights of neuron ``j``.
    b : ndarray of shape (n_out,) or None, default=None
        Biases, one per neuron. None means no bias.

    Returns
    -------
    ndarray of shape (n, n_out), or (n_out,) for a 1-D ``x``
        Pre-activations ``z``, as float64.

    Raises
    ------
    ValueError
        If ``x.shape[-1] != W.shape[0]`` or ``b.shape != (W.shape[1],)``.

    Notes
    -----
    Tested against ``torch.nn.functional.linear(x, W.T, b)`` in float64.

    Examples
    --------
    >>> W = np.array([[1.0, 0.0, -1.0],
    ...               [0.5, 1.0, 2.0]])
    >>> dense_forward([[1.0, 2.0]], W, np.array([0.0, 0.0, 1.0]))
    array([[2., 2., 4.]])
    >>> dense_forward([1.0, 2.0], W)
    array([2., 2., 3.])
    """
    # TODO: convert x to float64 and check the shapes before computing z (b is optional).
    raise NotImplementedError("dense_forward() is not implemented yet")


def init_weights(
    n_in: int,
    n_out: int,
    method: str = "he_normal",
    rng: np.random.Generator | None = None,
    *,
    scale: float = 0.05,
) -> np.ndarray:
    """Create the initial weight matrix of a dense layer.

    The variance-preserving schemes set the spread of the weights from the
    fan-in ``n_in`` (and the fan-out ``n_out`` for Glorot), so that the
    activations keep the same scale from layer to layer. A uniform law
    U(-a, a) has standard deviation a / sqrt(3).

    Parameters
    ----------
    n_in : int
        Fan-in: number of inputs of the layer, >= 1.
    n_out : int
        Fan-out: number of neurons of the layer, >= 1.
    method : str, default="he_normal"
        One of the following (case-sensitive) names:

        - ``"zeros"``: every weight is 0;
        - ``"constant"``: every weight equals ``scale``;
        - ``"uniform"``: U(-scale, scale);
        - ``"normal"``: N(0, scale**2);
        - ``"lecun_uniform"``: U(-a, a) with a = sqrt(3 / n_in);
        - ``"lecun_normal"``: N(0, s**2) with s = sqrt(1 / n_in);
        - ``"glorot_uniform"`` (alias ``"xavier_uniform"``):
          U(-a, a) with a = sqrt(6 / (n_in + n_out));
        - ``"glorot_normal"`` (alias ``"xavier_normal"``):
          N(0, s**2) with s = sqrt(2 / (n_in + n_out));
        - ``"he_uniform"``: U(-a, a) with a = sqrt(6 / n_in);
        - ``"he_normal"``: N(0, s**2) with s = sqrt(2 / n_in).

        Normal draws are NOT truncated (PyTorch convention; Keras truncates).
    rng : numpy.random.Generator or None, default=None
        Source of randomness. None means ``np.random.default_rng()``
        (a different draw at every call, not reproducible).
    scale : float, default=0.05
        Bound (``"uniform"``), standard deviation (``"normal"``) or value
        (``"constant"``); ignored by the other methods.

    Returns
    -------
    ndarray of shape (n_in, n_out)
        The weights, as float64.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or ``n_in < 1`` or ``n_out < 1``.

    Notes
    -----
    Tested against the bounds and standard deviations of
    ``torch.nn.init.xavier_uniform_``, ``xavier_normal_``, ``kaiming_uniform_``
    and ``kaiming_normal_`` (fan_in mode; ``nonlinearity="linear"`` for LeCun,
    ``"relu"`` for He): the empirical standard deviation of a 1000 x 1000 draw
    is within 2 % of the target, uniform draws stay inside their bound, and
    the same seed gives identical arrays.

    Examples
    --------
    >>> init_weights(2, 3, "constant", scale=0.5)
    array([[0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5]])
    >>> W = init_weights(100, 50, "glorot_uniform", rng=np.random.default_rng(0))
    >>> W.shape
    (100, 50)
    >>> bool(np.abs(W).max() <= np.sqrt(6 / 150))
    True
    >>> W1 = init_weights(3, 2, rng=np.random.default_rng(42))
    >>> W2 = init_weights(3, 2, rng=np.random.default_rng(42))
    >>> bool(np.array_equal(W1, W2))
    True
    """
    # TODO: validate the arguments, compute the bound or the standard deviation of the
    #   method, then draw ONE array of shape (n_in, n_out) with rng.
    raise NotImplementedError("init_weights() is not implemented yet")


def init_mlp(
    layer_sizes: Sequence[int],
    method: str = "he_normal",
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Initialise every layer of an MLP: weights with ``init_weights``, biases at 0.

    Parameters
    ----------
    layer_sizes : sequence of int
        ``[n_0, n_1, ..., n_L]`` as in ``count_parameters``.
    method : str, default="he_normal"
        Any ``init_weights`` method, used for every layer.
    rng : numpy.random.Generator or None, default=None
        Generator shared by all layers, drawn in layer order (first layer
        first). None means ``np.random.default_rng()``.

    Returns
    -------
    list of (W, b) tuples, one per layer
        ``W`` of shape ``(n_(l-1), n_l)`` and ``b = np.zeros(n_l)``.

    Raises
    ------
    ValueError
        As ``count_parameters`` and ``init_weights``.

    Notes
    -----
    Tested on properties: shapes, total size equal to
    ``count_parameters(layer_sizes)``, all biases zero, same seed gives the
    same parameters.

    Examples
    --------
    >>> params = init_mlp([2, 4, 4, 1], rng=np.random.default_rng(0))
    >>> [W.shape for W, b in params]
    [(2, 4), (4, 4), (4, 1)]
    >>> sum(W.size + b.size for W, b in params)
    37
    >>> params[0][1]
    array([0., 0., 0., 0.])
    """
    # TODO: validate layer_sizes, then call init_weights once per layer, in order.
    raise NotImplementedError("init_mlp() is not implemented yet")


def mlp_forward(
    x: ArrayLike,
    params: Sequence[tuple[np.ndarray, np.ndarray]],
    activation: Callable[[np.ndarray], np.ndarray] | None = None,
    output_activation: Callable[[np.ndarray], np.ndarray] | None = None,
    return_cache: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run the forward pass of a stack of dense layers.

    For each layer ``l``: ``z = a_prev @ W + b``, then ``a = activation(z)`` for
    the hidden layers and ``a = output_activation(z)`` for the last layer.

    Parameters
    ----------
    x : array-like of shape (n, n_0)
        Inputs, one sample per row.
    params : sequence of (W, b) tuples
        Parameters of each layer, as returned by ``init_mlp``.
    activation : callable or None, default=None
        Elementwise function applied to the hidden layers: ``np.tanh``,
        ``lambda z: np.maximum(0, z)``, or ``relu`` from
        ``mylearn.nn.activations`` (chapter 17). None means identity.
    output_activation : callable or None, default=None
        Function applied to the pre-activation of the last layer (e.g. a
        softmax). None means identity: the network returns logits.
    return_cache : bool, default=False
        Also return the cache needed by ``mylearn.nn.backward.mlp_backward``
        (chapter 18).

    Returns
    -------
    out : ndarray of shape (n, n_L)
        Output of the network.
    cache : list of (a_prev, z) tuples, one per layer
        Only if ``return_cache`` is True. ``a_prev`` is the input of layer
        ``l`` (``x`` for the first layer) and ``z = a_prev @ W + b`` its
        pre-activation, before any activation function.

    Raises
    ------
    ValueError
        If ``params`` is empty, or on inconsistent shapes (as ``dense_forward``).

    Notes
    -----
    Tested against ``torch.nn.Sequential(Linear, act, ..., Linear)`` with the
    same weights (``W.T``) in float64, and against ``predict`` of a fitted
    scikit-learn ``MLPClassifier`` (argmax of the logits computed from its
    ``coefs_`` and ``intercepts_``).

    Examples
    --------
    >>> params = [(np.array([[1.0, -1.0], [0.5, -2.0]]), np.array([0.0, 1.0])),
    ...           (np.array([[1.0], [1.0]]), np.array([-1.0]))]
    >>> x = np.array([[1.0, 1.0]])
    >>> relu = lambda z: np.maximum(0.0, z)
    >>> mlp_forward(x, params, activation=relu)
    array([[0.5]])
    >>> out, cache = mlp_forward(x, params, activation=relu, return_cache=True)
    >>> len(cache)
    2
    >>> cache[0][1]   # pre-activation of the hidden layer
    array([[ 1.5, -2. ]])
    >>> cache[1][0]   # input of the output layer: relu of the line above
    array([[1.5, 0. ]])
    """
    # TODO: loop over the layers with dense_forward; store (a_prev, z) BEFORE applying
    #   the activation (the last layer uses output_activation).
    raise NotImplementedError("mlp_forward() is not implemented yet")


def topological_order(graph: Mapping[str, Sequence[str]]) -> list[str]:
    """Return an evaluation order of a directed acyclic graph (Kahn's algorithm).

    An edge ``parent -> child`` means "the output of parent feeds child": in
    the returned order every parent comes before all its children. At each
    step, the next node is the alphabetically smallest of the nodes whose
    parents have all been placed already, so the order is unique.

    Parameters
    ----------
    graph : mapping of str to sequence of str
        Adjacency lists ``{parent: [children, ...]}``. Nodes that only appear
        as children are included in the result.

    Returns
    -------
    list of str
        All the nodes of the graph, each exactly once.

    Raises
    ------
    ValueError
        If the graph contains a cycle (a feedback loop).

    Notes
    -----
    Tested against ``graphlib.TopologicalSorter`` (standard library): same set
    of nodes, every edge respected, and ``graphlib.CycleError`` exactly when
    this function raises ``ValueError``.

    Examples
    --------
    >>> topological_order({"x": ["h1", "h2"], "h1": ["y"], "h2": ["y"]})
    ['x', 'h1', 'h2', 'y']
    >>> topological_order({"b": ["c"], "a": ["c"]})
    ['a', 'b', 'c']
    """
    # TODO: count the parents of each node, then repeatedly place the smallest ready
    #   node (heapq keeps them sorted); nodes never placed reveal a cycle.
    raise NotImplementedError("topological_order() is not implemented yet")
