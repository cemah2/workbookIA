"""minifw: a mini deep learning framework in NumPy, built on the mylearn library (chapters 16 to 20).

    from minifw import Dense, ReLU, Dropout, BatchNorm1d, Sequential, SoftmaxCrossEntropy, fit, evaluate

The layers (layers.py), the losses (losses.py), the model (model.py) and the training loop
(train.py) are checked against PyTorch by test_minifw.py.
"""

from .layers import Activation, BatchNorm1d, Dense, Dropout, Layer, ReLU, Tanh
from .losses import MSELoss, SoftmaxCrossEntropy
from .model import Sequential
from .train import evaluate, fit, iterate_minibatches, library_versions

__all__ = ["Activation", "BatchNorm1d", "Dense", "Dropout", "Layer", "MSELoss", "ReLU", "Sequential",
           "SoftmaxCrossEntropy", "Tanh", "evaluate", "fit", "iterate_minibatches", "library_versions"]
