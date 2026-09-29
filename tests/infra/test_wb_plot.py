"""Smoke tests of wb.plot (rendering with the Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wb import plot, synth  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    plot.set_style()
    yield
    plt.close("all")


def test_decision_boundary_sklearn_and_callable():
    from sklearn.linear_model import LogisticRegression

    X, y = synth.make_moons(120, seed=0)
    model = LogisticRegression().fit(X, y)
    ax = plot.plot_decision_boundary(model, X, y, resolution=60, title="logreg")
    assert ax.get_title() == "logreg"
    plot.plot_decision_boundary(model, X, y, resolution=60, proba=True)
    plot.plot_decision_boundary(lambda P: (P[:, 0] > 0.5).astype(int), X, y, resolution=40)
    X3, y3 = synth.make_blobs(90, centers=3, seed=0)
    plot.plot_decision_boundary(LogisticRegression().fit(X3, y3), X3, y3, resolution=40)
    with pytest.raises(ValueError):
        plot.plot_decision_boundary(model, np.zeros((5, 3)), None)


def test_decision_boundary_torch():
    torch = pytest.importorskip("torch")
    X, y = synth.make_xor(80, seed=0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    plot.plot_decision_boundary(net, X, y, resolution=40)
    net1 = torch.nn.Sequential(torch.nn.Linear(2, 1))
    plot.plot_decision_boundary(net1, X, y, resolution=40, proba=True)


def test_training_curves_formats():
    hist = {"loss": [1.0, 0.6, 0.4], "val_loss": [1.1, 0.7, 0.8],
            "accuracy": [0.5, 0.7, 0.8], "val_accuracy": [0.5, 0.65, 0.6]}
    fig = plot.plot_training_curves(hist, title="courbes")
    assert len(fig.axes) == 2
    rows = [dict(zip(hist, vals)) for vals in zip(*hist.values())]
    assert len(plot.plot_training_curves(rows).axes) == 2
    import pandas as pd

    assert len(plot.plot_training_curves(pd.DataFrame(hist), metrics=["loss"]).axes) == 1


@pytest.mark.parametrize("shape", [(6, 28, 28), (6, 784), (6, 32, 32, 3), (6, 3, 32, 32), (6, 1, 28, 28)])
def test_show_images_layouts(shape):
    images = np.random.default_rng(0).integers(0, 255, shape).astype(np.uint8)
    labels = np.array([0, 1, 2, 3, 4, 5])
    preds = np.array([0, 1, 9, 3, 4, 0])
    fig = plot.show_images(images, labels, preds, ncols=3)
    assert len(fig.axes) == 6


def test_show_images_normalised_and_class_names():
    images = np.random.default_rng(0).normal(size=(4, 28, 28))
    fig = plot.show_images(images, labels=[0, 1, 2, 3], class_names=list("abcd"), ncols=4)
    assert fig.axes[0].get_title() == "a"


def test_contour_and_confusion_matrix():
    path = np.array([[-1.5, 2.0], [-1.0, 1.0], [0.5, 0.3], [1.0, 1.0]])
    plot.plot_contour(synth.rosenbrock, path=path, minimum=(1, 1), resolution=80)
    plot.plot_contour(synth.rosenbrock, log=False, resolution=60)
    cm = np.array([[5, 1], [2, 7]])
    plot.plot_confusion_matrix(cm, ["chat", "chien"], normalize=True)
    plot.plot_confusion_matrix(cm)


def test_class_colors_beyond_palette():
    assert len(plot.class_colors(3)) == 3
    assert len(set(plot.class_colors(10))) == 10


def test_decision_boundary_with_arbitrary_labels():
    from sklearn.linear_model import LogisticRegression, Perceptron

    X, y = synth.make_moons(80, seed=0)
    signed = np.where(y == 1, 1, -1)
    ax = plot.plot_decision_boundary(Perceptron().fit(X, signed), X, signed, resolution=30)
    assert len(ax.collections) > 0
    names = np.where(y == 1, "Gentoo", "Adelie")
    plot.plot_decision_boundary(LogisticRegression().fit(X, names), X, names, resolution=30)
