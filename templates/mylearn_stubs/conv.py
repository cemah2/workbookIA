"""Convolution and pooling — mylearn, chapter 21 (Convolutional Neural Networks).

2-D convolution, its backward pass, transposed convolution and pooling,
written from scratch in NumPy with the PyTorch layout: batches of images of
shape (N, C, H, W) and convolution weights of shape (C_out, C_in, k_h, k_w),
so the weights of a PyTorch layer can be copied as they are. ``im2col``
materialises the "fly's-eye view" (every sliding patch as a column) and turns
the convolution into a single matrix product.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def conv_output_size(
    size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    """Output length along one spatial axis of a convolution or pooling window.

    out = floor((size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1.
    A dilated kernel of length k spans dilation * (k - 1) + 1 input cells.

    Parameters
    ----------
    size : int
        Input length (H or W), >= 1.
    kernel_size : int
        Window length k, >= 1.
    stride : int, default=1
        Step s between two windows, >= 1.
    padding : int, default=0
        Zeros added on EACH side, p >= 0.
    dilation : int, default=1
        Spacing d between two taps of the kernel, >= 1 (1: contiguous kernel).

    Returns
    -------
    int
        Number of window positions along this axis, >= 1.

    Raises
    ------
    ValueError
        If an argument is out of range, or if the window does not fit in the
        padded input (the result would be < 1).

    Notes
    -----
    Tested against the output shapes of ``torch.nn.functional.conv2d`` and
    ``torch.nn.functional.max_pool2d``.

    Examples
    --------
    >>> conv_output_size(28, 3)
    26
    >>> conv_output_size(28, 3, padding=1)
    28
    >>> conv_output_size(28, 2, stride=2)
    14
    >>> conv_output_size(300, 3, stride=3, padding=2)
    101
    """
    raise NotImplementedError("conv_output_size() is not implemented yet")


def pad2d(x: ArrayLike, padding: int | tuple[int, int], value: float = 0.0) -> np.ndarray:
    """Pad the two spatial axes of a (N, C, H, W) batch with a constant.

    Parameters
    ----------
    x : array-like of shape (N, C, H, W)
        Batch of images.
    padding : int or tuple of 2 ints
        ``p`` (the same on all four sides) or ``(p_h, p_w)``: ``p_h`` rows at
        the top and at the bottom, ``p_w`` columns on the left and on the
        right. Each >= 0.
    value : float, default=0.0
        Fill value (0 gives zero-padding).

    Returns
    -------
    ndarray of shape (N, C, H + 2 * p_h, W + 2 * p_w)
        A new float array; ``x`` is not modified.

    Raises
    ------
    ValueError
        If ``x`` is not 4-D or a padding is negative.

    Notes
    -----
    Tested against ``numpy.pad`` and ``torch.nn.functional.pad``.

    Examples
    --------
    >>> pad2d(np.ones((1, 1, 2, 2)), 1).shape
    (1, 1, 4, 4)
    >>> pad2d(np.ones((1, 1, 1, 1)), (0, 1))
    array([[[[0., 1., 0.]]]])
    """
    raise NotImplementedError("pad2d() is not implemented yet")


def im2col(
    x: ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> np.ndarray:
    """Extract every sliding patch of a batch (the "fly's-eye view") as a column.

    Column ``l`` holds the patch at output position ``l = row * W_out + col``,
    flattened in the order (channel, kernel row, kernel column). A convolution
    then becomes one matrix product: ``weight.reshape(C_out, -1) @ columns``.

    Parameters
    ----------
    x : array-like of shape (N, C, H, W)
        Batch of images.
    kernel_size : int or tuple of 2 ints
        ``k`` or ``(k_h, k_w)``.
    stride : int or tuple of 2 ints, default=1
        ``s`` or ``(s_h, s_w)``.
    padding : int or tuple of 2 ints, default=0
        ``p`` or ``(p_h, p_w)``, zeros added on each side.
    dilation : int or tuple of 2 ints, default=1
        ``d`` or ``(d_h, d_w)``.

    Returns
    -------
    ndarray of shape (N, C * k_h * k_w, H_out * W_out)
        The patches as columns, with the same layout as
        ``torch.nn.functional.unfold``.

    Raises
    ------
    ValueError
        If ``x`` is not 4-D or the window does not fit.

    Notes
    -----
    Tested against ``torch.nn.functional.unfold``.

    Examples
    --------
    >>> x = np.arange(9.0).reshape(1, 1, 3, 3)
    >>> im2col(x, 2)
    array([[[0., 1., 3., 4.],
            [1., 2., 4., 5.],
            [3., 4., 6., 7.],
            [4., 5., 7., 8.]]])
    """
    # TODO: pad once, then for each kernel offset (u, v) copy a strided slice of the
    #   padded input: two loops over the kernel, none over the output positions.
    raise NotImplementedError("im2col() is not implemented yet")


def conv2d(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | str = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> np.ndarray:
    """2-D convolution as computed by deep-learning layers (a cross-correlation).

    out[n, o, i, j] = bias[o] + sum over c, u, v of
    weight[o, c, u, v] * x_padded[n, c, i * s_h + u * d_h, j * s_w + v * d_w]
    (within the group of channel o). The kernel is NOT flipped, unlike the
    convolution of mathematics.

    Parameters
    ----------
    x : array-like of shape (N, C_in, H, W)
        Batch of images.
    weight : array-like of shape (C_out, C_in // groups, k_h, k_w)
        Filters, in the PyTorch layout.
    bias : array-like of shape (C_out,) or None, default=None
        One bias per output channel.
    stride : int or tuple of 2 ints, default=1
        ``s`` or ``(s_h, s_w)``.
    padding : int, tuple of 2 ints or str, default=0
        ``p``, ``(p_h, p_w)``, ``"valid"`` (no padding) or ``"same"`` (stride 1
        only: the output keeps H and W; for an even kernel span the extra row
        and column of zeros go at the bottom and on the right, as in PyTorch).
    dilation : int or tuple of 2 ints, default=1
        ``d`` or ``(d_h, d_w)``: spacing between kernel taps.
    groups : int, default=1
        Number of channel groups: C_in and C_out must both be divisible by it;
        input group g only feeds output group g. ``groups = C_in`` gives one
        filter per channel (depthwise convolution).

    Returns
    -------
    ndarray of shape (N, C_out, H_out, W_out)
        Output feature maps, as float64.

    Raises
    ------
    ValueError
        On incompatible shapes or channel counts, an unknown padding string,
        or ``padding="same"`` with a stride > 1.

    Notes
    -----
    Tested against ``torch.nn.functional.conv2d``.

    Examples
    --------
    >>> x = np.arange(9.0).reshape(1, 1, 3, 3)
    >>> conv2d(x, np.ones((1, 1, 2, 2)))
    array([[[[ 8., 12.],
             [20., 24.]]]])
    >>> conv2d(np.zeros((2, 3, 28, 28)), np.zeros((8, 3, 3, 3)), padding="same").shape
    (2, 8, 28, 28)
    """
    # TODO: normalise the int / tuple / str arguments first, then im2col and one
    #   matrix product per group.
    raise NotImplementedError("conv2d() is not implemented yet")


def conv2d_backward(
    dout: ArrayLike,
    x: ArrayLike,
    weight: ArrayLike,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradients of ``conv2d`` (groups=1, dilation=1) with respect to input, weights and bias.

    Weight sharing seen from backpropagation: each weight is used at every
    position, so its gradient sums the contributions of all positions (and of
    all the images of the batch).

    Parameters
    ----------
    dout : array-like of shape (N, C_out, H_out, W_out)
        Upstream gradient dL/dout.
    x : array-like of shape (N, C_in, H, W)
        Input of the forward pass.
    weight : array-like of shape (C_out, C_in, k_h, k_w)
        Weights of the forward pass.
    stride : int or tuple of 2 ints, default=1
        As in the forward pass.
    padding : int or tuple of 2 ints, default=0
        As in the forward pass (integers only).

    Returns
    -------
    dx : ndarray of shape (N, C_in, H, W)
        dL/dx.
    dW : ndarray of shape (C_out, C_in, k_h, k_w)
        dL/dweight.
    db : ndarray of shape (C_out,)
        dL/dbias.

    Raises
    ------
    ValueError
        If ``dout`` does not have the shape of the forward output.

    Notes
    -----
    Tested against ``torch.autograd.grad`` through
    ``torch.nn.functional.conv2d``.

    Examples
    --------
    >>> x = np.arange(9.0).reshape(1, 1, 3, 3)
    >>> dx, dW, db = conv2d_backward(np.ones((1, 1, 2, 2)), x, np.ones((1, 1, 2, 2)))
    >>> dx    # how many windows see each pixel
    array([[[[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]]])
    >>> dW
    array([[[[ 8., 12.],
             [20., 24.]]]])
    >>> db
    array([4.])
    """
    # TODO: db and dW sum over the images and positions; dx sends each output
    #   gradient back to the input cells of its window (then remove the padding).
    raise NotImplementedError("conv2d_backward() is not implemented yet")


def max_pool2d(
    x: ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    """Maximum over each window, channel by channel.

    Parameters
    ----------
    x : array-like of shape (N, C, H, W)
        Batch of feature maps.
    kernel_size : int or tuple of 2 ints
        ``k`` or ``(k_h, k_w)``.
    stride : int, tuple of 2 ints or None, default=None
        None means ``kernel_size`` (non-overlapping tiles).
    padding : int or tuple of 2 ints, default=0
        Added on each side; padded cells act as -inf (they are never the
        maximum). At most half the kernel size (PyTorch rule).

    Returns
    -------
    ndarray of shape (N, C, H_out, W_out)
        Channels are pooled independently.

    Raises
    ------
    ValueError
        If ``x`` is not 4-D, the padding exceeds ``kernel_size // 2``, or the
        window does not fit.

    Notes
    -----
    Tested against ``torch.nn.functional.max_pool2d``.

    Examples
    --------
    >>> x = np.arange(16.0).reshape(1, 1, 4, 4)
    >>> max_pool2d(x, 2)
    array([[[[ 5.,  7.],
             [13., 15.]]]])
    """
    # TODO: pad with -inf, then take the maximum over each window.
    raise NotImplementedError("max_pool2d() is not implemented yet")


def avg_pool2d(
    x: ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    """Mean over each window, channel by channel.

    Zero-padded cells are counted in the mean (``count_include_pad=True``, the
    PyTorch default): every window is divided by k_h * k_w.

    Parameters
    ----------
    x : array-like of shape (N, C, H, W)
        Batch of feature maps.
    kernel_size : int or tuple of 2 ints
        ``k`` or ``(k_h, k_w)``.
    stride : int, tuple of 2 ints or None, default=None
        None means ``kernel_size`` (non-overlapping tiles).
    padding : int or tuple of 2 ints, default=0
        Zeros added on each side, at most half the kernel size.

    Returns
    -------
    ndarray of shape (N, C, H_out, W_out)
        Channels are pooled independently.

    Raises
    ------
    ValueError
        If ``x`` is not 4-D, the padding exceeds ``kernel_size // 2``, or the
        window does not fit.

    Notes
    -----
    Tested against ``torch.nn.functional.avg_pool2d``.

    Examples
    --------
    >>> x = np.arange(16.0).reshape(1, 1, 4, 4)
    >>> avg_pool2d(x, 2)
    array([[[[ 2.5,  4.5],
             [10.5, 12.5]]]])
    """
    # TODO: pad with zeros, then take the mean over each window.
    raise NotImplementedError("avg_pool2d() is not implemented yet")


def conv_transpose2d(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    """Transposed ("fractionally strided") convolution, the upsampling layer.

    Equivalent recipe: insert ``stride - 1`` zeros between the input cells,
    pad with ``k - 1 - padding`` zeros on each side, then cross-correlate
    with the kernel flipped in both spatial directions (and its two channel
    axes swapped).

    Parameters
    ----------
    x : array-like of shape (N, C_in, H, W)
        Batch of feature maps.
    weight : array-like of shape (C_in, C_out, k_h, k_w)
        Filters, in the PyTorch layout for TRANSPOSED convolutions (input
        channels first).
    bias : array-like of shape (C_out,) or None, default=None
        One bias per output channel.
    stride : int or tuple of 2 ints, default=1
        Upsampling factor s >= 1.
    padding : int or tuple of 2 ints, default=0
        ``p`` removed from each side of the output, with 0 <= p <= k - 1.

    Returns
    -------
    ndarray of shape (N, C_out, (H - 1) * s_h - 2 * p_h + k_h, (W - 1) * s_w - 2 * p_w + k_w)
        Upsampled feature maps, as float64.

    Raises
    ------
    ValueError
        On a shape mismatch or an out-of-range padding.

    Notes
    -----
    Tested against ``torch.nn.functional.conv_transpose2d``.

    Examples
    --------
    >>> x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    >>> conv_transpose2d(x, np.ones((1, 1, 2, 2)), stride=2)
    array([[[[1., 1., 2., 2.],
             [1., 1., 2., 2.],
             [3., 3., 4., 4.],
             [3., 3., 4., 4.]]]])
    >>> conv_transpose2d(np.ones((1, 1, 2, 2)), np.ones((1, 1, 2, 2)))
    array([[[[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]]])
    """
    # TODO: follow the recipe above and reuse conv2d.
    raise NotImplementedError("conv_transpose2d() is not implemented yet")
