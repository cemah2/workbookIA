"""Recurrent cells — mylearn, chapter 22 (Recurrent Neural Networks).

Forward passes of the Elman RNN, LSTM and GRU cells written from scratch in
NumPy, one time step at a time, for batch-first sequences of shape (N, T, D).
The weights follow the mylearn convention of dense layers, ``z = x @ W + b``
with ``W`` of shape (n_in, n_out), so ``W_ih = weight_ih.T`` of PyTorch; the
gate order (i, f, g, o for the LSTM; r, z, n for the GRU) and the two bias
vectors ``b_ih`` and ``b_hh`` are those of PyTorch, so its weights can be
copied directly. No backward pass here: backpropagation through time is done
on paper and measured with autograd.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rnn_cell_forward(
    x_t: ArrayLike,
    h_prev: ArrayLike,
    W_ih: ArrayLike,
    W_hh: ArrayLike,
    b_ih: ArrayLike | None = None,
    b_hh: ArrayLike | None = None,
    nonlinearity: str = "tanh",
) -> np.ndarray:
    """One step of an Elman RNN: h_t = f(x_t @ W_ih + b_ih + h_prev @ W_hh + b_hh).

    Parameters
    ----------
    x_t : array-like of shape (N, D)
        Inputs at time t.
    h_prev : array-like of shape (N, H)
        Previous hidden state.
    W_ih : array-like of shape (D, H)
        Input-to-hidden weights (``weight_ih.T`` of PyTorch).
    W_hh : array-like of shape (H, H)
        Hidden-to-hidden weights (``weight_hh.T`` of PyTorch).
    b_ih : array-like of shape (H,) or None, default=None
        Input bias. None means no bias.
    b_hh : array-like of shape (H,) or None, default=None
        Hidden bias (PyTorch keeps two bias vectors). None means no bias.
    nonlinearity : {"tanh", "relu"}, default="tanh"
        The function f.

    Returns
    -------
    ndarray of shape (N, H)
        The new state h_t, which is also the output of the cell.

    Raises
    ------
    ValueError
        On a shape mismatch or an unknown ``nonlinearity``.

    Notes
    -----
    Tested against ``torch.nn.RNNCell`` (weights copied with ``.T``).

    Examples
    --------
    >>> rnn_cell_forward([[1.0, 0.0]], [[0.0, 0.5]], np.eye(2), np.eye(2))
    array([[0.76159416, 0.46211716]])
    """
    raise NotImplementedError("rnn_cell_forward() is not implemented yet")


def rnn_forward(
    x: ArrayLike,
    W_ih: ArrayLike,
    W_hh: ArrayLike,
    b_ih: ArrayLike | None = None,
    b_hh: ArrayLike | None = None,
    h0: ArrayLike | None = None,
    nonlinearity: str = "tanh",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the RNN cell over a batch of sequences, time step after time step.

    The same weights are used at every time step.

    Parameters
    ----------
    x : array-like of shape (N, T, D)
        Batch-first sequences.
    W_ih : array-like of shape (D, H)
        Input-to-hidden weights.
    W_hh : array-like of shape (H, H)
        Hidden-to-hidden weights.
    b_ih : array-like of shape (H,) or None, default=None
        Input bias.
    b_hh : array-like of shape (H,) or None, default=None
        Hidden bias.
    h0 : array-like of shape (N, H) or None, default=None
        Initial state; None means zeros.
    nonlinearity : {"tanh", "relu"}, default="tanh"
        As in ``rnn_cell_forward``.

    Returns
    -------
    outputs : ndarray of shape (N, T, H)
        Every state h_1, ..., h_T, stacked along axis 1.
    h_T : ndarray of shape (N, H)
        The last state (equal to ``outputs[:, -1]``).

    Raises
    ------
    ValueError
        On a shape mismatch or an unknown ``nonlinearity``.

    Notes
    -----
    Tested against ``torch.nn.RNN(batch_first=True)`` (whose final state has
    an extra leading layer axis: ``h_n[0]``).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(2, 5, 3))
    >>> outputs, h_T = rnn_forward(x, rng.normal(size=(3, 4)), rng.normal(size=(4, 4)))
    >>> outputs.shape, h_T.shape
    ((2, 5, 4), (2, 4))
    >>> bool(np.array_equal(outputs[:, -1], h_T))
    True
    """
    # TODO: loop over the time axis with rnn_cell_forward and collect the states.
    raise NotImplementedError("rnn_forward() is not implemented yet")


def lstm_cell_forward(
    x_t: ArrayLike,
    h_prev: ArrayLike,
    c_prev: ArrayLike,
    W_ih: ArrayLike,
    W_hh: ArrayLike,
    b_ih: ArrayLike | None = None,
    b_hh: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One LSTM step, with the gates stacked in PyTorch order: i, f, g, o.

    With a = x_t @ W_ih + b_ih + h_prev @ W_hh + b_hh, split into four blocks
    of H columns [a_i | a_f | a_g | a_o]:
    i = sigmoid(a_i) (input gate), f = sigmoid(a_f) (forget gate),
    g = tanh(a_g) (candidate), o = sigmoid(a_o) (output gate);
    c_t = f * c_prev + i * g and h_t = o * tanh(c_t).

    Parameters
    ----------
    x_t : array-like of shape (N, D)
        Inputs at time t.
    h_prev : array-like of shape (N, H)
        Previous hidden state.
    c_prev : array-like of shape (N, H)
        Previous cell memory.
    W_ih : array-like of shape (D, 4H)
        Input weights, column blocks [i | f | g | o] (``weight_ih.T``).
    W_hh : array-like of shape (H, 4H)
        Hidden weights, same blocks (``weight_hh.T``).
    b_ih : array-like of shape (4H,) or None, default=None
        Input bias.
    b_hh : array-like of shape (4H,) or None, default=None
        Hidden bias.

    Returns
    -------
    h_t : ndarray of shape (N, H)
        New hidden state (the output of the cell).
    c_t : ndarray of shape (N, H)
        New cell memory.

    Raises
    ------
    ValueError
        On a shape mismatch (for example ``W_ih`` without 4H columns).

    Notes
    -----
    Tested against ``torch.nn.LSTMCell`` (weights copied with ``.T``).

    Examples
    --------
    With all weights at 0, every gate is sigmoid(0) = 0.5 and g = 0:

    >>> h_t, c_t = lstm_cell_forward(np.zeros((1, 3)), np.zeros((1, 2)), np.ones((1, 2)),
    ...                              np.zeros((3, 8)), np.zeros((2, 8)))
    >>> c_t
    array([[0.5, 0.5]])
    >>> h_t
    array([[0.23105858, 0.23105858]])
    """
    # TODO: compute the pre-activations of the four gates at once, then split the
    #   4H columns into blocks of H.
    raise NotImplementedError("lstm_cell_forward() is not implemented yet")


def lstm_forward(
    x: ArrayLike,
    W_ih: ArrayLike,
    W_hh: ArrayLike,
    b_ih: ArrayLike | None = None,
    b_hh: ArrayLike | None = None,
    h0: ArrayLike | None = None,
    c0: ArrayLike | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Run the LSTM cell over a batch of sequences; same return structure as ``nn.LSTM``.

    Parameters
    ----------
    x : array-like of shape (N, T, D)
        Batch-first sequences.
    W_ih : array-like of shape (D, 4H)
        Input weights, column blocks [i | f | g | o].
    W_hh : array-like of shape (H, 4H)
        Hidden weights, same blocks.
    b_ih : array-like of shape (4H,) or None, default=None
        Input bias.
    b_hh : array-like of shape (4H,) or None, default=None
        Hidden bias.
    h0 : array-like of shape (N, H) or None, default=None
        Initial hidden state; None means zeros.
    c0 : array-like of shape (N, H) or None, default=None
        Initial cell memory; None means zeros.

    Returns
    -------
    outputs : ndarray of shape (N, T, H)
        Every hidden state h_1, ..., h_T.
    (h_T, c_T) : tuple of 2 ndarrays of shape (N, H)
        Last hidden state and last cell memory.

    Raises
    ------
    ValueError
        On a shape mismatch.

    Notes
    -----
    Tested against a single-layer ``torch.nn.LSTM(batch_first=True)`` (whose
    final states have an extra leading layer axis).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(2, 5, 3))
    >>> outputs, (h_T, c_T) = lstm_forward(x, rng.normal(size=(3, 16)),
    ...                                    rng.normal(size=(4, 16)))
    >>> outputs.shape, h_T.shape, c_T.shape
    ((2, 5, 4), (2, 4), (2, 4))
    """
    # TODO: loop over the time axis with lstm_cell_forward and collect h_t.
    raise NotImplementedError("lstm_forward() is not implemented yet")


def gru_cell_forward(
    x_t: ArrayLike,
    h_prev: ArrayLike,
    W_ih: ArrayLike,
    W_hh: ArrayLike,
    b_ih: ArrayLike | None = None,
    b_hh: ArrayLike | None = None,
) -> np.ndarray:
    """One GRU step, with the gates in PyTorch order: reset r, update z, new n.

    With the column blocks [r | z | n] of the weights and biases:
    r = sigmoid(x_t @ W_ir + b_ir + h_prev @ W_hr + b_hr),
    z = sigmoid(x_t @ W_iz + b_iz + h_prev @ W_hz + b_hz),
    n = tanh(x_t @ W_in + b_in + r * (h_prev @ W_hn + b_hn)),
    h_t = (1 - z) * n + z * h_prev.
    The reset gate multiplies ``h_prev @ W_hn + b_hn``: this is why the two
    bias vectors must stay separate.

    Parameters
    ----------
    x_t : array-like of shape (N, D)
        Inputs at time t.
    h_prev : array-like of shape (N, H)
        Previous hidden state.
    W_ih : array-like of shape (D, 3H)
        Input weights, column blocks [r | z | n] (``weight_ih.T``).
    W_hh : array-like of shape (H, 3H)
        Hidden weights, same blocks (``weight_hh.T``).
    b_ih : array-like of shape (3H,) or None, default=None
        Input bias.
    b_hh : array-like of shape (3H,) or None, default=None
        Hidden bias.

    Returns
    -------
    ndarray of shape (N, H)
        The new state h_t.

    Raises
    ------
    ValueError
        On a shape mismatch.

    Notes
    -----
    Tested against ``torch.nn.GRUCell`` (weights copied with ``.T``).

    Examples
    --------
    With all weights at 0: r = z = 0.5 and n = 0, so h_t = 0.5 * h_prev.

    >>> gru_cell_forward(np.zeros((1, 3)), [[1.0, -1.0]], np.zeros((3, 6)), np.zeros((2, 6)))
    array([[ 0.5, -0.5]])
    """
    # TODO: keep x_t @ W_ih + b_ih and h_prev @ W_hh + b_hh separate before
    #   splitting them into the r, z, n blocks.
    raise NotImplementedError("gru_cell_forward() is not implemented yet")
