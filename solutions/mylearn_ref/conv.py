"""Convolution and pooling — mylearn, chapter 21 (Convolutional Neural Networks).

2-D convolution, its backward pass, transposed convolution and pooling,
written from scratch in NumPy with the PyTorch layout: batches of images of
shape (N, C, H, W) and convolution weights of shape (C_out, C_in, k_h, k_w),
so the weights of a PyTorch layer can be copied as they are. ``im2col``
materialises the "fly's-eye view" (every sliding patch as a column) and turns
the convolution into a single matrix product.

Reference implementation.
"""

from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike


def _is_int(value) -> bool:
    """A true integer (Python or NumPy), not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _pair(value, name: str, minimum: int) -> tuple[int, int]:
    """``v`` or ``(v_h, v_w)`` as a tuple of two Python ints, each >= minimum."""
    if _is_int(value):
        pair = (int(value), int(value))
    elif isinstance(value, (tuple, list)) and len(value) == 2 and all(_is_int(v) for v in value):
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{name} must be an int or a tuple of 2 ints, got {value!r}")
    if min(pair) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return pair


def _batch(x: ArrayLike, name: str = "x", layout: str = "(N, C, H, W)") -> np.ndarray:
    """A new float64 copy of a 4-D array: a batch (N, C, H, W), or a weight with its own ``layout``."""
    arr = np.array(x, dtype=np.float64, copy=True)
    if arr.ndim != 4:
        raise ValueError(f"{name} must be 4-D {layout}, got {arr.ndim}-D of shape {arr.shape}")
    return arr


def _out_sizes(shape, kernel, stride, padding, dilation) -> tuple[int, int]:
    """(H_out, W_out) for an input of spatial shape ``shape`` (raises if the window does not fit)."""
    return tuple(conv_output_size(shape[a], kernel[a], stride[a], padding[a], dilation[a]) for a in (0, 1))


def _windows(xp: np.ndarray, kernel, stride, dilation, out) -> np.ndarray:
    """All windows of an already padded batch, shape (N, C, k_h, k_w, H_out, W_out).

    Two loops over the kernel offsets (u, v); each copies one strided slice of ``xp``.
    """
    (kh, kw), (sh, sw), (dh, dw), (ho, wo) = kernel, stride, dilation, out
    n, c = xp.shape[:2]
    patches = np.empty((n, c, kh, kw, ho, wo), dtype=np.float64)
    for u in range(kh):
        top = u * dh
        for v in range(kw):
            left = v * dw
            patches[:, :, u, v] = xp[:, :, top:top + sh * (ho - 1) + 1:sh, left:left + sw * (wo - 1) + 1:sw]
    return patches


def _pool_args(x, kernel_size, stride, padding):
    """The input (a float64 copy) and the normalised arguments of the pooling functions."""
    arr = _batch(x)
    kernel = _pair(kernel_size, "kernel_size", 1)
    stride = kernel if stride is None else _pair(stride, "stride", 1)
    padding = _pair(padding, "padding", 0)
    if padding[0] > kernel[0] // 2 or padding[1] > kernel[1] // 2:
        raise ValueError(f"padding {padding} must be at most half the kernel size {kernel} (kernel_size // 2)")
    out = _out_sizes(arr.shape[2:], kernel, stride, padding, (1, 1))
    return arr, kernel, stride, padding, out


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
    values = {"size": size, "kernel_size": kernel_size, "stride": stride, "padding": padding,
              "dilation": dilation}
    for name, value in values.items():
        if not _is_int(value):
            raise ValueError(f"{name} must be an int, got {value!r}")
    for name, minimum in (("size", 1), ("kernel_size", 1), ("stride", 1), ("padding", 0), ("dilation", 1)):
        if values[name] < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {values[name]}")
    span = dilation * (kernel_size - 1) + 1
    if span > size + 2 * padding:
        raise ValueError(f"the window (span {span}) does not fit in the padded input "
                         f"(size {size} + 2 * {padding})")
    return int((size + 2 * padding - span) // stride + 1)


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
    arr = _batch(x)
    ph, pw = _pair(padding, "padding", 0)
    n, c, h, w = arr.shape
    out = np.full((n, c, h + 2 * ph, w + 2 * pw), value, dtype=np.float64)
    out[:, :, ph:ph + h, pw:pw + w] = arr
    return out


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
    arr = _batch(x)
    kernel = _pair(kernel_size, "kernel_size", 1)
    stride = _pair(stride, "stride", 1)
    padding = _pair(padding, "padding", 0)
    dilation = _pair(dilation, "dilation", 1)
    out = _out_sizes(arr.shape[2:], kernel, stride, padding, dilation)
    patches = _windows(pad2d(arr, padding), kernel, stride, dilation, out)
    n, c = arr.shape[:2]
    return patches.reshape(n, c * kernel[0] * kernel[1], out[0] * out[1])


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
    arr = _batch(x)
    w = _batch(weight, "weight", "(C_out, C_in // groups, k_h, k_w)")
    n, c_in, h, width = arr.shape
    c_out, c_group, kh, kw = w.shape
    if not _is_int(groups) or groups < 1:
        raise ValueError(f"groups must be an int >= 1, got {groups!r}")
    if c_in % groups or c_out % groups:
        raise ValueError(f"groups={groups} must divide C_in={c_in} and C_out={c_out}")
    if c_group != c_in // groups:
        raise ValueError(f"weight of shape {w.shape} expects {c_group * groups} input channels "
                         f"(groups={groups}), got x with C_in={c_in}")
    b = None
    if bias is not None:
        b = np.array(bias, dtype=np.float64)
        if b.shape != (c_out,):
            raise ValueError(f"bias must have shape ({c_out},), got {b.shape}")
    stride = _pair(stride, "stride", 1)
    dilation = _pair(dilation, "dilation", 1)
    if isinstance(padding, str):
        if padding == "valid":
            padding = (0, 0)
        elif padding == "same":
            if stride != (1, 1):
                raise ValueError(f'padding="same" needs stride 1, got stride {stride}')
            total_h, total_w = dilation[0] * (kh - 1), dilation[1] * (kw - 1)
            top, left = total_h // 2, total_w // 2        # the extra row / column at the bottom / right
            padded = np.zeros((n, c_in, h + total_h, width + total_w))
            padded[:, :, top:top + h, left:left + width] = arr
            arr, padding = padded, (0, 0)
        else:
            raise ValueError(f'padding must be an int, a tuple of 2 ints, "valid" or "same", got {padding!r}')
    padding = _pair(padding, "padding", 0)
    out = _out_sizes(arr.shape[2:], (kh, kw), stride, padding, dilation)
    cols = im2col(arr, (kh, kw), stride, padding, dilation)           # (N, C_in * kh * kw, L)
    rows, per_out = c_group * kh * kw, c_out // groups
    parts = [w[g * per_out:(g + 1) * per_out].reshape(per_out, rows) @ cols[:, g * rows:(g + 1) * rows]
             for g in range(groups)]                                   # each (N, per_out, L)
    result = np.concatenate(parts, axis=1).reshape(n, c_out, out[0], out[1])
    if b is not None:
        result += b.reshape(1, c_out, 1, 1)
    return result


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
    arr = _batch(x)
    w = _batch(weight, "weight", "(C_out, C_in, k_h, k_w)")
    n, c_in, h, width = arr.shape
    c_out, c_w, kh, kw = w.shape
    if c_w != c_in:
        raise ValueError(f"weight of shape {w.shape} expects C_in={c_w}, got x with C_in={c_in}")
    stride = _pair(stride, "stride", 1)
    padding = _pair(padding, "padding", 0)
    ho, wo = _out_sizes((h, width), (kh, kw), stride, padding, (1, 1))
    g = np.array(dout, dtype=np.float64)
    if g.shape != (n, c_out, ho, wo):
        raise ValueError(f"dout must have the shape of the forward output {(n, c_out, ho, wo)}, got {g.shape}")
    db = g.sum(axis=(0, 2, 3))
    g = g.reshape(n, c_out, ho * wo)
    cols = im2col(arr, (kh, kw), stride, padding)                       # (N, C_in * kh * kw, L)
    dw = np.einsum("nol,nkl->ok", g, cols).reshape(w.shape)             # sum over images and positions
    dcols = (w.reshape(c_out, -1).T @ g).reshape(n, c_in, kh, kw, ho, wo)
    (ph, pw), (sh, sw) = padding, stride
    dxp = np.zeros((n, c_in, h + 2 * ph, width + 2 * pw))
    for u in range(kh):                                                 # col2im: accumulate the windows
        for v in range(kw):
            dxp[:, :, u:u + sh * (ho - 1) + 1:sh, v:v + sw * (wo - 1) + 1:sw] += dcols[:, :, u, v]
    return dxp[:, :, ph:ph + h, pw:pw + width].copy(), dw, db


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
    arr, kernel, stride, padding, out = _pool_args(x, kernel_size, stride, padding)
    patches = _windows(pad2d(arr, padding, value=-np.inf), kernel, stride, (1, 1), out)
    return patches.max(axis=(2, 3))


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
    arr, kernel, stride, padding, out = _pool_args(x, kernel_size, stride, padding)
    patches = _windows(pad2d(arr, padding), kernel, stride, (1, 1), out)
    return patches.sum(axis=(2, 3)) / (kernel[0] * kernel[1])        # count_include_pad=True


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
    arr = _batch(x)
    w = _batch(weight, "weight", "(C_in, C_out, k_h, k_w)")
    n, c_in, h, width = arr.shape
    if w.shape[0] != c_in:
        raise ValueError(f"weight of shape {w.shape} expects C_in={w.shape[0]} (input channels first), "
                         f"got x with C_in={c_in}")
    kh, kw = w.shape[2:]
    sh, sw = _pair(stride, "stride", 1)
    ph, pw = _pair(padding, "padding", 0)
    if ph > kh - 1 or pw > kw - 1:
        raise ValueError(f"padding {(ph, pw)} must be between 0 and k - 1 = {(kh - 1, kw - 1)}")
    spread = np.zeros((n, c_in, (h - 1) * sh + 1, (width - 1) * sw + 1))
    spread[:, :, ::sh, ::sw] = arr                                      # stride - 1 zeros between the cells
    kernel = w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)                  # flipped, (C_out, C_in, kh, kw)
    return conv2d(spread, kernel, bias, padding=(kh - 1 - ph, kw - 1 - pw))
