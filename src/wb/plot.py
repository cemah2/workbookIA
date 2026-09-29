"""Plot helpers shared by all chapters: style, decision boundaries,
training curves, image grids, contour plots and confusion matrices.

All functions return the matplotlib ``Axes`` (or ``Figure``) and never call
``plt.show()``: a notebook displays the figure at the end of the cell; in a
script, call ``plt.show()`` yourself.
"""

from __future__ import annotations

import numpy as np

# Colour-blind-aware categorical palette (fixed order, never cycled beyond 8).
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#a32a29"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
GOOD = "#008300"
BAD = "#c62828"


def _plt():
    import matplotlib.pyplot as plt

    return plt


def class_colors(n: int) -> list[str]:
    """Colours for ``n`` classes: the workbook palette up to 8, ``tab10``/``tab20`` beyond.

    Beyond 8 classes, colour alone cannot separate the classes: add direct
    labels (e.g. the digit at each cluster centre).
    """
    if n <= len(CATEGORICAL):
        return CATEGORICAL[:n]
    import matplotlib

    cmap = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return [matplotlib.colors.to_hex(cmap(i % cmap.N)) for i in range(n)]


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("wb_blues", SEQUENTIAL)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("wb_diverging", DIVERGING)


def set_style() -> None:
    """Apply the workbook style (readable sizes, discreet grid, fixed palette)."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.figsize": (7.0, 4.2),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": TEXT_MUTED,
            "axes.labelcolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "image.cmap": "viridis",
            "font.size": 10.5,
        }
    )


# ---------------------------------------------------------------------------
# Decision boundaries
# ---------------------------------------------------------------------------
def _predict_grid(model, grid: np.ndarray, proba: bool) -> np.ndarray:
    """Evaluate a model on grid points; supports sklearn, torch and callables."""
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(model, torch.nn.Module):
        params = list(model.parameters())
        device = params[0].device if params else "cpu"
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(torch.as_tensor(grid, dtype=torch.float32, device=device)).cpu().numpy()
        model.train(was_training)
        if out.ndim == 2 and out.shape[1] > 1:
            return out.argmax(axis=1) if not proba else np.exp(out) / np.exp(out).sum(1, keepdims=True)
        out = out.ravel()
        if proba:
            return 1 / (1 + np.exp(-out)) if (out.min() < 0 or out.max() > 1) else out
        return (out > (0.0 if (out.min() < 0 or out.max() > 1) else 0.5)).astype(int)
    if proba and hasattr(model, "predict_proba"):
        return model.predict_proba(grid)
    if hasattr(model, "predict"):
        return model.predict(grid)
    return np.asarray(model(grid))


def plot_decision_boundary(
    model,
    X,
    y=None,
    ax=None,
    *,
    resolution: int = 200,
    padding: float = 0.5,
    proba: bool = False,
    title: str | None = None,
    class_names=None,
    alpha: float = 0.25,
    show_points: bool = True,
):
    """Colour the plane by the class a 2D model predicts, and overlay the data.

    ``model`` can be a scikit-learn estimator, a PyTorch module (logits or
    probabilities) or any function ``f(points) -> labels``. With ``proba=True``
    and two classes, the probability of class 1 is shown as a gradient.
    """
    plt = _plt()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("plot_decision_boundary needs X with exactly 2 columns")
    ax = ax or plt.subplots()[1]
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    pred = np.asarray(_predict_grid(model, grid, proba))

    labels_known = np.unique(y) if y is not None else None
    if proba and pred.ndim == 2 and pred.shape[1] == 2:
        pred = pred[:, 1]
    is_proba = proba and pred.ndim == 1 and np.issubdtype(pred.dtype, np.floating)
    if not is_proba and pred.ndim == 2:
        pred = pred.argmax(axis=1)  # class scores -> class index
        if labels_known is not None and pred.max() < len(labels_known):
            pred = labels_known[pred]
    if labels_known is not None:
        classes = labels_known
    else:
        classes = np.unique(pred) if not is_proba else np.array([0, 1])
    n_classes = max(len(classes), 2)
    colors = class_colors(n_classes)
    from matplotlib.colors import ListedColormap

    if is_proba:
        cs = ax.contourf(xx, yy, pred.reshape(xx.shape), levels=np.linspace(0, 1, 11),
                         cmap=diverging_cmap(), alpha=0.6)
        ax.contour(xx, yy, pred.reshape(xx.shape), levels=[0.5], colors=[TEXT], linewidths=1.5)
        plt.colorbar(cs, ax=ax, label="P(classe 1)")
    else:
        # map any labels (-1/+1, 1/2, strings...) to indices 0..K-1 for colouring
        lookup = {label: index for index, label in enumerate(classes.tolist())}
        Z = np.array([lookup.get(v, np.nan) for v in pred.tolist()], dtype=float).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=np.arange(n_classes + 1) - 0.5,
                    cmap=ListedColormap(colors), alpha=alpha)
        ax.contour(xx, yy, Z, levels=np.arange(n_classes) + 0.5, colors=[TEXT_MUTED], linewidths=0.8)

    if show_points and y is not None:
        y = np.asarray(y)
        for k, cls in enumerate(classes):
            mask = y == cls
            name = class_names[k] if class_names is not None else f"classe {cls}"
            ax.scatter(X[mask, 0], X[mask, 1], s=28, color=colors[k % len(colors)],
                       edgecolor="white", linewidth=1.0, label=name, zorder=3)
        ax.legend(loc="best")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def _history_to_dict(history) -> dict[str, list[float]]:
    if hasattr(history, "to_dict") and hasattr(history, "columns"):  # pandas DataFrame
        return {c: list(history[c]) for c in history.columns}
    if isinstance(history, dict):
        return {k: list(np.asarray(v, dtype=float)) for k, v in history.items()}
    if isinstance(history, (list, tuple)) and history and isinstance(history[0], dict):
        keys = history[0].keys()
        return {k: [row.get(k, np.nan) for row in history] for k in keys}
    raise TypeError("history must be a dict of lists, a list of dicts or a DataFrame")


def plot_training_curves(history, metrics=None, *, axes=None, title: str | None = None,
                         xlabel: str = "epoch", log_loss: bool = False):
    """Plot training vs validation curves, one panel per metric.

    ``history`` is e.g. ``{"loss": [...], "val_loss": [...], "accuracy": [...],
    "val_accuracy": [...]}`` (or a list of per-epoch dicts, or a DataFrame).
    Keys ``x`` and ``val_x`` are paired automatically. The best validation
    point is marked.
    """
    plt = _plt()
    hist = _history_to_dict(history)
    hist.pop("epoch", None)
    if metrics is None:
        metrics = [k for k in hist if not k.startswith("val_")]
        metrics += [k[4:] for k in hist if k.startswith("val_") and k[4:] not in hist]
    if axes is None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.8), squeeze=False)
        axes = axes[0]
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure
    for ax, metric in zip(axes, metrics):
        train, val = hist.get(metric), hist.get("val_" + metric)
        if train is not None:
            ax.plot(np.arange(1, len(train) + 1), train, color=CATEGORICAL[0], label="entraînement")
        if val is not None:
            epochs = np.arange(1, len(val) + 1)
            ax.plot(epochs, val, color=CATEGORICAL[1], label="validation")
            values = np.asarray(val, dtype=float)
            if np.isfinite(values).any():
                lower_is_better = "loss" in metric or "error" in metric
                best = int(np.nanargmin(values) if lower_is_better else np.nanargmax(values))
                ax.scatter([epochs[best]], [values[best]], s=64, color=CATEGORICAL[1],
                           edgecolor="white", linewidth=2, zorder=3)
                ax.annotate(f"meilleur : {values[best]:.3g}", (epochs[best], values[best]),
                            textcoords="offset points", xytext=(6, 8), fontsize=9, color=TEXT_MUTED)
        ax.set_title(metric)
        ax.set_xlabel(xlabel)
        if log_loss and "loss" in metric:
            ax.set_yscale("log")
        if train is not None and val is not None:
            ax.legend()
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Image grids
# ---------------------------------------------------------------------------
def _to_numpy_images(images) -> np.ndarray:
    if type(images).__module__.startswith("torch"):
        images = images.detach().cpu().numpy()
    images = np.asarray(images)
    if images.ndim == 2:  # (N, H*W) flattened square images
        side = int(round(np.sqrt(images.shape[1])))
        if side * side != images.shape[1]:
            raise ValueError("cannot reshape flattened images into squares")
        images = images.reshape(-1, side, side)
    if images.ndim == 4 and images.shape[1] in (1, 3) and images.shape[-1] not in (1, 3):
        images = images.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    return images


def _display_range(img: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    lo, hi = img.min(), img.max()
    if lo >= 0 and hi <= 1:
        return img
    if lo >= 0 and hi <= 255:
        return img / 255.0
    return (img - lo) / (hi - lo + 1e-12)  # normalised data: rescale for display


def show_images(images, labels=None, preds=None, *, class_names=None, ncols: int = 8,
                n: int | None = None, cmap: str = "gray_r", title: str | None = None,
                size: float = 1.3):
    """Display a grid of images (MNIST, CIFAR...) with optional labels and predictions.

    Accepts arrays of shape (N, H, W), (N, H, W, C), (N, C, H, W) or (N, H*W),
    NumPy or PyTorch. Wrong predictions are shown in red with a ✗.
    """
    plt = _plt()
    images = _to_numpy_images(images)
    total = len(images) if n is None else min(n, len(images))
    ncols = max(1, min(ncols, total))
    nrows = int(np.ceil(total / ncols))
    extra = 0.35 if (labels is not None or preds is not None) else 0.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, (size + extra) * nrows), squeeze=False)

    def name(k):
        k = int(k)
        return class_names[k] if class_names is not None else str(k)

    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= total:
            continue
        img = _display_range(images[i])
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None, vmin=0, vmax=1)
        if preds is not None and labels is not None:
            ok = int(preds[i]) == int(labels[i])
            text = name(preds[i]) if ok else f"✗ {name(preds[i])} ({name(labels[i])})"
            ax.set_title(text, fontsize=8.5, color=GOOD if ok else BAD)
        elif labels is not None:
            ax.set_title(name(labels[i]), fontsize=8.5, color=TEXT)
        elif preds is not None:
            ax.set_title(name(preds[i]), fontsize=8.5, color=TEXT_MUTED)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Functions of two variables (ch. 5 and 19)
# ---------------------------------------------------------------------------
def plot_contour(f, xlim=(-2.0, 2.0), ylim=(-1.0, 3.0), *, path=None, ax=None,
                 levels: int = 30, log: bool = True, resolution: int = 300,
                 title: str | None = None, minimum=None):
    """Contour plot of ``f(x, y)`` (vectorised), optionally with an optimisation path.

    ``path`` is an array of shape (steps, 2), e.g. the successive points of a
    gradient descent. ``log=True`` spaces the levels logarithmically (useful
    for Rosenbrock).
    """
    plt = _plt()
    ax = ax or plt.subplots(figsize=(6.2, 5.0))[1]
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.asarray(f(XX, YY), dtype=float)
    if log:
        Zp = Z - Z.min() + 1e-3
        lv = np.logspace(np.log10(Zp.min()), np.log10(Zp.max()), levels)
        cs = ax.contourf(XX, YY, Zp, levels=lv, cmap=sequential_cmap().reversed(),
                         norm=__import__("matplotlib").colors.LogNorm())
    else:
        cs = ax.contourf(XX, YY, Z, levels=levels, cmap=sequential_cmap().reversed())
    ax.contour(XX, YY, Z if not log else Zp, levels=cs.levels, colors="white", linewidths=0.4, alpha=0.6)
    if path is not None:
        path = np.asarray(path, dtype=float)
        ax.plot(path[:, 0], path[:, 1], "-", color=CATEGORICAL[1], lw=2, label="trajectoire")
        ax.scatter(path[0, 0], path[0, 1], s=64, color=CATEGORICAL[1], edgecolor="white",
                   linewidth=2, zorder=3, label="départ")
        ax.scatter(path[-1, 0], path[-1, 1], s=64, marker="s", color=CATEGORICAL[1],
                   edgecolor="white", linewidth=2, zorder=3, label="arrivée")
    if minimum is not None:
        ax.scatter([minimum[0]], [minimum[1]], s=140, marker="*", color=TEXT,
                   edgecolor="white", linewidth=1.5, zorder=4, label="minimum")
    if path is not None or minimum is not None:
        ax.legend(loc="upper left")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(False)
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Confusion matrix (ch. 3, 13)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names=None, *, normalize: bool = False, ax=None,
                          title: str | None = None):
    """Heat map of a confusion matrix (rows = true class, columns = predicted)."""
    plt = _plt()
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-12) if normalize else cm
    ax = ax or plt.subplots(figsize=(1.0 + 0.6 * len(cm), 0.8 + 0.6 * len(cm)))[1]
    im = ax.imshow(shown, cmap=sequential_cmap())
    names = class_names if class_names is not None else [str(i) for i in range(len(cm))]
    ax.set_xticks(range(len(cm)), names, rotation=45 if len(cm) > 4 else 0, ha="right" if len(cm) > 4 else "center")
    ax.set_yticks(range(len(cm)), names)
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("classe réelle")
    ax.grid(False)
    threshold = shown.max() / 2 if shown.size else 0
    for i in range(len(cm)):
        for j in range(len(cm)):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > threshold else TEXT)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    return ax
