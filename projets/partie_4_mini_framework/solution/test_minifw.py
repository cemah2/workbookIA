"""Equivalence tests of minifw against PyTorch: same weights, same inputs => same outputs and same gradients.

    python -m pytest projets/partie_4_mini_framework/solution -q

Every comparison is made in float64, at 1e-6 relative and 1e-8 absolute (in practice, the gaps are
around 1e-15). The PyTorch modules receive the weights of minifw (nn.Linear stores W transposed).
"""

import math

import numpy as np
import pytest

import data
from minifw import BatchNorm1d, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy, Tanh, evaluate, fit
from minifw.train import iterate_minibatches
from mylearn.optim import AdamW, cosine_lr

torch = pytest.importorskip("torch")
nn = torch.nn
TOL = {"rtol": 1e-6, "atol": 1e-8}


def assert_close(got, want, what):
    """The first line of the failure says what was expected (pytest -rf only prints that line)."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{what}: expected shape {want.shape}, got {got.shape}"
    gap = np.abs(got - want).max() if got.size else 0.0
    assert np.allclose(got, want, **TOL), f"{what}: largest gap to PyTorch {gap:.2e} (expected < 1e-6 relative)"


def tensor(a, grad=False):
    return torch.tensor(np.asarray(a, dtype=np.float64), requires_grad=grad)


def torch_linear(layer):
    """An nn.Linear in float64 with the weights of a minifw Dense (W transposed)."""
    lin = nn.Linear(*layer.W.shape).double()
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(layer.W.T.copy()))
        lin.bias.copy_(torch.from_numpy(layer.b.copy()))
    return lin


@pytest.fixture
def rng():
    return np.random.default_rng(4)


# ---------------------------------------------------------------- Dense
def test_dense_forward_matches_nn_linear(rng):
    layer = Dense(5, 3, rng=rng)
    x = rng.normal(size=(4, 5))
    assert_close(layer.forward(x), torch_linear(layer)(tensor(x)).detach(), "Dense.forward")


def test_dense_backward_matches_autograd(rng):
    layer = Dense(5, 3, rng=rng)
    x, dout = rng.normal(size=(4, 5)), rng.normal(size=(4, 3))
    xt, lin = tensor(x, grad=True), torch_linear(layer)
    lin(xt).backward(tensor(dout))
    layer.forward(x)
    dx = layer.backward(dout)
    assert_close(dx, xt.grad, "Dense.backward: dL/dx")
    assert_close(layer.dW, lin.weight.grad.T, "Dense.backward: dW (shape (n_in, n_out))")
    assert_close(layer.db, lin.bias.grad, "Dense.backward: db")


def test_dense_init_is_he_normal_from_the_rng():
    layer = Dense(400, 300, rng=np.random.default_rng(0))
    assert layer.W.shape == (400, 300) and layer.b.shape == (300,), "W must have the shape (n_in, n_out), b (n_out,)"
    assert np.all(layer.b == 0), "b must start at zero"
    assert abs(layer.W.std() - math.sqrt(2 / 400)) < 0.002, f"He normal: std sqrt(2/400) = 0.0707, got {layer.W.std():.4f}"
    same = Dense(400, 300, rng=np.random.default_rng(0))
    assert np.array_equal(layer.W, same.W), "two Dense with the same rng seed must start with the same W"


def test_dense_parameters_are_the_arrays_themselves(rng):
    layer = Dense(3, 2, rng=rng)
    params = layer.parameters()
    assert params[0] is layer.W and params[1] is layer.b, "parameters() must return [W, b] themselves, not copies"
    layer.forward(rng.normal(size=(2, 3)))
    layer.backward(np.ones((2, 2)))
    assert layer.gradients()[0] is layer.dW, "gradients() must return [dW, db], the arrays updated by backward"


# ---------------------------------------------------------------- activations
@pytest.mark.parametrize("cls, fn", [(ReLU, torch.relu), (Tanh, torch.tanh)], ids=["relu", "tanh"])
def test_activation_forward_and_backward_match_autograd(cls, fn, rng):
    x, dout = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    x[0, 0] = 0.0                                                    # ReLU'(0) = 0, as in PyTorch
    xt = tensor(x, grad=True)
    fn(xt).backward(tensor(dout))
    layer = cls()
    assert_close(layer.forward(x), fn(tensor(x)), f"{cls.__name__}.forward")
    assert_close(layer.backward(dout), xt.grad, f"{cls.__name__}.backward")


def test_activation_has_no_parameters():
    assert ReLU().parameters() == [] and ReLU().gradients() == [], "an activation has no parameter"


# ---------------------------------------------------------------- Dropout
def test_dropout_in_training_keeps_or_scales_each_value(rng):
    layer = Dropout(0.25, rng=rng)
    out = layer.forward(np.ones((200, 50)))
    values = set(np.round(np.unique(out), 12))
    assert values <= {0.0, round(1 / 0.75, 12)}, f"inverted dropout: every value is 0 or 1/(1-p) = 1.3333, got {values}"
    assert abs((out == 0).mean() - 0.25) < 0.02, f"about 25 % of zeros expected, got {(out == 0).mean():.3f}"


def test_dropout_backward_uses_the_mask_of_the_forward(rng):
    layer = Dropout(0.5, rng=rng)
    x = rng.normal(size=(8, 6))
    out = layer.forward(x)
    dx = layer.backward(np.ones_like(x))
    assert np.array_equal(dx == 0, out == 0) and np.allclose(dx[dx != 0], 2.0), \
        "backward must multiply by the mask of the LAST forward: zeros at the same places, 1/(1-p) elsewhere"


def test_dropout_in_evaluation_is_the_identity(rng):
    layer = Dropout(0.5, rng=rng)
    layer.training = False
    x = rng.normal(size=(4, 3))
    assert_close(layer.forward(x), x, "Dropout in evaluation mode")
    assert_close(layer.backward(np.ones_like(x)), np.ones_like(x), "Dropout.backward in evaluation mode")


# ---------------------------------------------------------------- BatchNorm1d
def torch_batchnorm(layer):
    bn = nn.BatchNorm1d(len(layer.gamma), momentum=layer.momentum, eps=layer.eps).double()
    with torch.no_grad():
        bn.weight.copy_(torch.from_numpy(layer.gamma.copy()))
        bn.bias.copy_(torch.from_numpy(layer.beta.copy()))
    return bn


def test_batchnorm_training_matches_nn_batchnorm1d(rng):
    layer = BatchNorm1d(4)
    layer.gamma[...] = rng.normal(size=4)
    layer.beta[...] = rng.normal(size=4)
    bn = torch_batchnorm(layer)
    x, dout = rng.normal(2.0, 3.0, size=(16, 4)), rng.normal(size=(16, 4))
    xt = tensor(x, grad=True)
    out_t = bn(xt)
    out_t.backward(tensor(dout))
    out = layer.forward(x)
    dx = layer.backward(dout)
    assert_close(out, out_t.detach(), "BatchNorm1d.forward in training mode")
    assert_close(dx, xt.grad, "BatchNorm1d.backward: dL/dx")
    assert_close(layer.dgamma, bn.weight.grad, "BatchNorm1d.backward: dgamma")
    assert_close(layer.dbeta, bn.bias.grad, "BatchNorm1d.backward: dbeta")


def test_batchnorm_running_statistics_and_evaluation_match(rng):
    layer, bn = BatchNorm1d(3), nn.BatchNorm1d(3).double()
    for _ in range(3):
        x = rng.normal(1.0, 2.0, size=(10, 3))
        layer.forward(x)
        bn(tensor(x))
    assert_close(layer.running_mean, bn.running_mean, "BatchNorm1d.running_mean after 3 mini-batches")
    assert_close(layer.running_var, bn.running_var, "BatchNorm1d.running_var after 3 mini-batches (unbiased)")
    layer.training = False
    bn.eval()
    x = rng.normal(size=(5, 3))
    assert_close(layer.forward(x), bn(tensor(x)).detach(), "BatchNorm1d.forward in evaluation mode")


# ---------------------------------------------------------------- loss
def test_softmax_cross_entropy_matches_torch(rng):
    logits, y = rng.normal(size=(7, 5)), rng.integers(0, 5, size=7)
    lt = tensor(logits, grad=True)
    want = nn.functional.cross_entropy(lt, torch.from_numpy(y))
    want.backward()
    loss, grad = SoftmaxCrossEntropy()(logits, y)
    assert isinstance(loss, float), f"the loss must be a Python float, got {type(loss).__name__}"
    assert_close(loss, want.item(), "SoftmaxCrossEntropy: the mean loss")
    assert_close(grad, lt.grad, "SoftmaxCrossEntropy: dL/dlogits")


# ---------------------------------------------------------------- Sequential
def build_pair(rng, dropout=0.0):
    """A minifw MLP 6-8-8-3 (BatchNorm1d, ReLU, Tanh) and the same nn.Sequential in float64."""
    layers = [Dense(6, 8, rng=rng), BatchNorm1d(8), ReLU(), Dense(8, 8, rng=rng), Tanh(), Dense(8, 3, rng=rng)]
    model = Sequential(*layers)
    twin = nn.Sequential(torch_linear(layers[0]), torch_batchnorm(layers[1]), nn.ReLU(), torch_linear(layers[3]),
                         nn.Tanh(), torch_linear(layers[5]))
    return model, twin


def test_sequential_forward_and_all_gradients_match_pytorch(rng):
    model, twin = build_pair(rng)
    x, y = rng.normal(size=(12, 6)), rng.integers(0, 3, size=12)
    loss_t = nn.functional.cross_entropy(twin(tensor(x)), torch.from_numpy(y))
    loss_t.backward()
    loss, dout = SoftmaxCrossEntropy()(model(x), y)
    model.backward(dout)
    assert_close(loss, loss_t.item(), "Sequential: the loss")
    want = []
    for module in twin:
        if isinstance(module, nn.Linear):
            want += [module.weight.grad.T, module.bias.grad]
        elif isinstance(module, nn.BatchNorm1d):
            want += [module.weight.grad, module.bias.grad]
    got = model.gradients()
    assert len(got) == len(want), f"gradients(): expected {len(want)} arrays (W, b, gamma, beta in order), got {len(got)}"
    for k, (g, w) in enumerate(zip(got, want)):
        assert_close(g, w, f"Sequential.gradients()[{k}]")


def test_sequential_parameters_order_and_count(rng):
    model, twin = build_pair(rng)
    assert model.n_parameters() == sum(p.numel() for p in twin.parameters()), "n_parameters() must count every value"
    assert isinstance(model.n_parameters(), int), "n_parameters() must return an int"
    shapes = [tuple(p.shape) for p in model.parameters()]
    assert shapes == [(6, 8), (8,), (8,), (8,), (8, 8), (8,), (8, 3), (3,)], \
        f"parameters(): W, b of each Dense and gamma, beta of the batchnorm, in the order of the layers; got {shapes}"


def test_eval_switches_dropout_and_batchnorm(rng):
    model = Sequential(Dense(4, 6, rng=rng), BatchNorm1d(6), ReLU(), Dropout(0.5, rng=rng), Dense(6, 2, rng=rng))
    x = rng.normal(size=(5, 4))
    model.train()
    model(rng.normal(size=(32, 4)))
    model.eval()
    assert all(not layer.training for layer in model.layers), "eval() must set training = False in every layer"
    assert np.array_equal(model(x), model(x)), "in evaluation mode, two forward passes must give the same output"
    model.train()
    assert all(layer.training for layer in model.layers), "train() must set training = True in every layer"


def test_state_dict_copies_and_load_restores_in_place(rng):
    model = Sequential(Dense(3, 4, rng=rng), BatchNorm1d(4), ReLU(), Dense(4, 2, rng=rng))
    model(rng.normal(size=(8, 3)))
    state = model.state_dict()
    W = model.layers[0].W
    assert state["0.W"] is not W and np.array_equal(state["0.W"], W), "state_dict() must hold COPIES of the arrays"
    assert "1.running_mean" in state, "state_dict() must also save the buffers (running_mean, running_var)"
    W += 1.0
    model.load_state_dict(state)
    assert model.layers[0].W is W and np.array_equal(W, state["0.W"]), \
        "load_state_dict() must copy IN PLACE (the optimizer keeps references to these arrays)"


# ---------------------------------------------------------------- optimizer and training loop
def test_one_adamw_step_on_the_model_matches_torch_optim(rng):
    model, twin = build_pair(rng)
    x, y = rng.normal(size=(12, 6)), rng.integers(0, 3, size=12)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    torch_opt = torch.optim.AdamW(twin.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(2):
        loss, dout = SoftmaxCrossEntropy()(model(x), y)
        model.backward(dout)
        optimizer.step(model.gradients())
        torch_opt.zero_grad()
        nn.functional.cross_entropy(twin(tensor(x)), torch.from_numpy(y)).backward()
        torch_opt.step()
    assert_close(model.layers[0].W, twin[0].weight.detach().T, "after two AdamW steps: W of the first layer")
    assert_close(model.layers[1].gamma, twin[1].weight.detach(), "after two AdamW steps: gamma of the batchnorm")


def test_fit_follows_warmup_cosine_and_clips(rng):
    X, y = data.load_moons(200, seed=1)
    model = Sequential(Dense(2, 16, rng=rng), ReLU(), Dense(16, 2, rng=rng))
    history = fit(model, SoftmaxCrossEntropy(), AdamW(model.parameters(), lr=1e-2), X, y, X, y, epochs=3,
                  batch_size=50, base_lr=1e-2, warmup_steps=4, max_norm=0.5, rng=np.random.default_rng(0))
    want = [cosine_lr(t, 12, 1e-2, 4) for t in range(12)]
    assert_close(history["lr"], want, "the learning rate of each step (warmup + cosine)")
    assert all(n is not None and n > 0 for n in history["grad_norm"]), "grad_norm: the norm before clipping, each step"
    assert len(history["train_loss"]) == 3 and len(history["val_acc"]) == 3, "one value per epoch"


def test_fit_with_early_stopping_restores_the_best_epoch(rng):
    X, y = data.load_moons(300, noise=0.35, seed=2)
    X_val, y_val = data.load_moons(100, noise=0.35, seed=3)
    model = Sequential(Dense(2, 64, rng=rng), ReLU(), Dense(64, 64, rng=rng), ReLU(), Dense(64, 2, rng=rng))
    history = fit(model, SoftmaxCrossEntropy(), AdamW(model.parameters(), lr=3e-2, weight_decay=0.0), X, y,
                  X_val, y_val, epochs=60, batch_size=32, base_lr=3e-2, patience=3, rng=np.random.default_rng(0))
    best = history["best_epoch"]
    assert len(history["val_loss"]) - best == 3, "with a patience of 3, the training stops 3 epochs after the best one"
    loss, _ = evaluate(model, SoftmaxCrossEntropy(), X_val, y_val)
    assert_close(loss, history["val_loss"][best - 1], "the model must end with the weights of the best epoch")


def test_iterate_minibatches_covers_every_index_once():
    batches = list(iterate_minibatches(10, 4, np.random.default_rng(0)))
    assert [len(b) for b in batches] == [4, 4, 2], "consecutive slices of batch_size, the last one smaller"
    assert np.array_equal(np.concatenate(batches), np.random.default_rng(0).permutation(10)), \
        "the order must be rng.permutation(n)"
