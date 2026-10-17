"""Tests of my package minifw (mini-project of part IV): equivalence with PyTorch.

    python -m pytest mon_travail/projets/partie_4_mini_framework -q

Two tests are given as examples. Write at least eight more (step MP4.4 of the notebook checks
that there are ten or more, and runs them against six mutants of minifw), each with a clear name
and a message whose first line says what was expected. PyTorch is the oracle: it checks, minifw
computes. Work in float64 (``.double()``, ``torch.float64``) and give the PyTorch modules the
weights of minifw (``nn.Linear`` stores W transposed). Ideas, from the deliverables of the brief:
- Dense: the gradients dW and db after backward, against autograd; parameters() returns the
  arrays themselves; two layers built with the same seed start alike;
- an activation (ReLU, Tanh): forward and backward against torch.relu, torch.tanh (ReLU'(0) = 0);
- Dropout: about p zeros and 1 / (1 - p) elsewhere in training mode, the mask of the last
  forward in backward, the identity in evaluation mode;
- BatchNorm1d: the output and the three gradients in training mode, the running statistics
  after a few mini-batches, the output in evaluation mode, against nn.BatchNorm1d;
- SoftmaxCrossEntropy: the mean loss (a float) and its gradient, against F.cross_entropy;
- Sequential: a whole network (Dense, BatchNorm1d, ReLU, Tanh) against its nn.Sequential twin,
  the loss and EVERY gradient; train() and eval(); state_dict() copies, load_state_dict()
  restores in place;
- one or two steps of mylearn.optim.AdamW on model.parameters() against torch.optim.AdamW;
- fit: the learning rate of each step follows cosine_lr; the best epoch is restored.
"""

import numpy as np
import pytest

from minifw import Dense, ReLU

torch = pytest.importorskip("torch")
nn = torch.nn


def assert_close(got, want, what):
    """The first line of the failure says what was expected (pytest -rf only prints that line)."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{what}: expected shape {want.shape}, got {got.shape}"
    gap = np.abs(got - want).max() if got.size else 0.0
    assert np.allclose(got, want, rtol=1e-6, atol=1e-8), f"{what}: largest gap to PyTorch {gap:.2e} (expected < 1e-6)"


def test_dense_forward_matches_nn_linear():
    rng = np.random.default_rng(0)
    layer = Dense(5, 3, rng=rng)
    x = rng.normal(size=(4, 5))
    lin = nn.Linear(5, 3).double()
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(layer.W.T.copy()))      # nn.Linear stores W as (n_out, n_in)
        lin.bias.copy_(torch.from_numpy(layer.b.copy()))
    assert_close(layer.forward(x), lin(torch.from_numpy(x)).detach(), "Dense.forward")


def test_relu_backward_matches_autograd():
    rng = np.random.default_rng(1)
    x, dout = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    x[0, 0] = 0.0                                                  # ReLU'(0) = 0, as in PyTorch
    xt = torch.tensor(x, requires_grad=True)
    torch.relu(xt).backward(torch.from_numpy(dout))
    layer = ReLU()
    layer.forward(x)
    assert_close(layer.backward(dout), xt.grad, "ReLU.backward")
