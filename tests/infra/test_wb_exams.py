"""Tests of wb.exams: the oracles of the code questions of the mock exams (checkpoints IV to VI), and the guard that
keeps their solutions out of the exam notebooks (audit P5 R3).

Each oracle is compared with PyTorch, NumPy or a computation written another way; the last tests read the exam
notebooks the learner has open during the exam: no oracle, no formula to remember and no letter of CP6.9 a) with its
fix may be readable there.
"""

import ast
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wb import exams

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CP4.9
# ---------------------------------------------------------------------------
def softmax_loss_and_grads(params, X, y):
    """(mean cross-entropy, [dW, db]) of the softmax regression z = X @ W + b (the given tool of CP4.9)."""
    W, b = params
    z = X @ W + b
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y]).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    d /= n
    return float(loss), [X.T @ d, d.sum(axis=0)]


def test_importing_exams_does_not_import_torch():
    code = "import sys; import wb.exams; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""}, check=True).stdout.strip()
    assert out == "False", f"expected wb.exams to import without PyTorch (CP4 runs without it), got torch imported: {out}"


def test_cp4_adamw_step_is_torch_adamw():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    start = [rng.normal(size=(3, 2)), rng.normal(size=2)]
    grads = [[rng.normal(size=(3, 2)), rng.normal(size=2)] for _ in range(4)]
    mine, state = [p.copy() for p in start], {}
    tensors = [torch.tensor(p, requires_grad=True) for p in start]
    opt = torch.optim.AdamW(tensors, lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.1)
    for g in grads:
        objects = [id(p) for p in mine]
        assert exams.cp4_adamw_step(mine, [x.copy() for x in g], state, 0.01, betas=(0.8, 0.95), eps=1e-6,
                                    weight_decay=0.1) is None, "expected an in-place step that returns None"
        assert [id(p) for p in mine] == objects, "expected the arrays of params to be updated in place"
        for t, x in zip(tensors, g):
            t.grad = torch.from_numpy(x.copy())
        opt.step()
    gap = max(np.abs(p - t.detach().numpy()).max() for p, t in zip(mine, tensors))
    assert gap < 1e-12, f"expected the values of torch.optim.AdamW after 4 steps, largest gap {gap:.2e}"
    assert state["t"] == 4, f"expected the step counter t = 4 in state, got {state.get('t')!r}"


def test_cp4_adamw_step_with_zero_gradient_only_decays():
    w = [np.array([2.0, -4.0])]
    exams.cp4_adamw_step(w, [np.zeros(2)], {}, 0.1, weight_decay=0.5)
    assert np.allclose(w[0], [1.9, -3.8], rtol=1e-12, atol=0), f"expected w * (1 - lr * wd) = [1.9, -3.8], got {w[0]}"


def test_cp4_train_epoch_is_torch_adamw_with_clipping():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(70, 6)), rng.integers(0, 4, size=70)
    start = [rng.normal(0, 0.5, size=(6, 4)), np.zeros(4)]
    order = np.random.default_rng(2).permutation(70)
    mine, state = [p.copy() for p in start], {}
    mean = exams.cp4_train_epoch(mine, X, y, state, 0.05, 16, 0.3, order, softmax_loss_and_grads)

    W, b = (torch.tensor(p, requires_grad=True) for p in start)
    opt = torch.optim.AdamW([W, b], lr=0.05, weight_decay=0.01)
    losses = []
    for start_i in range(0, 70, 16):
        idx = torch.from_numpy(order[start_i:start_i + 16])
        loss = torch.nn.functional.cross_entropy(torch.from_numpy(X)[idx] @ W + b, torch.from_numpy(y)[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([W, b], 0.3)
        opt.step()
        losses.append((loss.item(), len(idx)))
    gap = max(np.abs(mine[0] - W.detach().numpy()).max(), np.abs(mine[1] - b.detach().numpy()).max())
    assert gap < 1e-9, f"expected the parameters of torch AdamW + clip_grad_norm_ after one epoch, largest gap {gap:.2e}"
    want = sum(loss * n for loss, n in losses) / 70                  # the last mini-batch has 6 examples
    assert abs(mean - want) < 1e-12, f"expected the mean loss per example {want:.12f}, got {mean:.12f}"
    assert state["t"] == 5, f"expected 5 steps (70 examples by 16), got t = {state.get('t')!r}"


def test_cp4_torch_epoch_matches_the_numpy_epoch():
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(50, 5)), rng.integers(0, 3, size=50)
    start = [rng.normal(0, 0.5, size=(5, 3)), np.zeros(3)]
    order = np.random.default_rng(4).permutation(50)
    numpy_params = [p.copy() for p in start]
    exams.cp4_train_epoch(numpy_params, X, y, {}, 0.02, 12, 0.4, order, softmax_loss_and_grads)
    torch_params = exams.cp4_torch_epoch([p.copy() for p in start], X, y, order, 0.02, 12, 0.4)
    gap = max(np.abs(a - b).max() for a, b in zip(numpy_params, torch_params))
    assert gap < 1e-9, f"expected the same parameters after the PyTorch epoch and the NumPy one, largest gap {gap:.2e}"
    assert all(isinstance(p, np.ndarray) for p in torch_params), "expected NumPy arrays [W, b]"


# ---------------------------------------------------------------------------
# CP5.5 and CP5.6
# ---------------------------------------------------------------------------
def test_cp5_rnn_is_the_simple_rnn_recurrence():
    pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    X, h0 = rng.normal(size=(2, 4, 3)), rng.normal(size=(2, 5))
    Wx, Wh, b = rng.normal(0, 0.5, (3, 5)), rng.normal(0, 0.5, (5, 5)), rng.normal(0, 0.5, 5)
    X_in = X.copy()
    out, h_n = exams.cp5_rnn(X, h0, Wx, Wh, b)
    h, states = h0, []
    for t in range(4):
        h = np.tanh(X[:, t] @ Wx + h @ Wh + b)
        states.append(h)
    want = np.stack(states, axis=1)
    assert out.shape == (2, 4, 5) and h_n.shape == (2, 5), f"expected (2, 4, 5) and (2, 5), got {out.shape}, {h_n.shape}"
    assert np.abs(out - want).max() < 1e-12, f"expected tanh(x_t Wx + h Wh + b) at each step, gap {np.abs(out - want).max():.2e}"
    assert np.array_equal(h_n, out[:, -1]), "expected h_n to be the state after the last step"
    assert np.array_equal(X, X_in), "expected cp5_rnn to leave X unchanged"


def _tiny_problem(torch, seed=0):
    """(make_model, make_loaders): a small float64 classifier with dropout, and fixed mini-batches."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(seed)
    X = torch.randn(96, 6, generator=g, dtype=torch.float64)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).long() + (X[:, 2] > 1).long()

    def make_model():
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 3)).double()

    def make_loaders():
        train = DataLoader(TensorDataset(X[:64], y[:64]), batch_size=20, shuffle=True,
                           generator=torch.Generator().manual_seed(seed))
        return train, DataLoader(TensorDataset(X[64:], y[64:]), batch_size=10)     # the last batch has 2

    return make_model, make_loaders


def test_cp5_fit_early_stopping_follows_the_contract(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    make_model, make_loaders = _tiny_problem(torch)
    model = make_model()
    history = exams.cp5_fit_early_stopping(model, *make_loaders(), 12, 2, tmp_path / "best.pt", lr=0.05)

    ref = make_model()                                                # the same training, written another way
    optimizer = torch.optim.Adam(ref.parameters(), lr=0.05)
    train_loader, val_loader = make_loaders()
    train_loss, val_loss, states = [], [], []
    for _ in range(len(history["val_loss"])):
        ref.train()
        pairs = []
        for X, y in train_loader:
            loss = -F.log_softmax(ref(X), dim=1).gather(1, y[:, None]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pairs.append((loss.item(), len(y)))
        train_loss.append(sum(v * n for v, n in pairs) / sum(n for _, n in pairs))
        ref.eval()
        with torch.no_grad():
            per_example = torch.cat([F.cross_entropy(ref(X), y, reduction="none") for X, y in val_loader])
        val_loss.append(per_example.mean().item())
        states.append(copy.deepcopy(ref.state_dict()))
    assert np.allclose(history["train_loss"], train_loss, rtol=1e-10, atol=0), \
        f"expected the training losses per example {train_loss}, got {history['train_loss']}"
    assert np.allclose(history["val_loss"], val_loss, rtol=1e-10, atol=0), \
        f"expected the validation losses per example {val_loss}, got {history['val_loss']}"
    best = int(np.argmin(val_loss)) + 1
    assert history["best_epoch"] == best, f"expected best_epoch = {best} (from 1), got {history['best_epoch']!r}"
    done = len(val_loss)
    assert done == 12 or done - best == 2, f"expected a stop 2 epochs after the best one ({best}), got {done} epochs"
    saved = torch.load(tmp_path / "best.pt", weights_only=True)
    for key, value in states[best - 1].items():
        assert torch.equal(saved[key], value), f"expected the saved state_dict of epoch {best} ({key})"
        assert torch.equal(model.state_dict()[key], value), f"expected the model reloaded with epoch {best} ({key})"


def test_cp5_fit_early_stopping_with_a_flat_loss_stops_after_patience(tmp_path):
    torch = pytest.importorskip("torch")
    make_model, make_loaders = _tiny_problem(torch, seed=1)
    history = exams.cp5_fit_early_stopping(make_model(), *make_loaders(), 6, 2, tmp_path / "best.pt", lr=0.0)
    assert history["best_epoch"] == 1 and len(history["val_loss"]) == 3, \
        f"expected best_epoch = 1 and 3 epochs with lr = 0 and patience = 2, got {history['best_epoch']!r}, " \
        f"{len(history['val_loss'])} epochs"
    assert all(isinstance(v, float) for v in history["train_loss"] + history["val_loss"]), "expected Python floats"


# ---------------------------------------------------------------------------
# CP6.8 and CP6.9
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape, d, beta", [((3, 784), 5, 0.5), ((4, 1, 28, 28), 4, 1.0), ((2, 1, 3, 3), 2, 3.0)],
                         ids=["flat-images", "mnist-shape", "beta-3"])
def test_cp6_vae_loss_is_the_per_image_bce_and_kl(shape, d, beta):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(8)
    logits = 2.0 * torch.randn(*shape, generator=g, dtype=torch.float64)
    x = torch.rand(*shape, generator=g, dtype=torch.float64)
    mu = torch.randn(shape[0], d, generator=g, dtype=torch.float64)
    log_var = 0.5 * torch.randn(shape[0], d, generator=g, dtype=torch.float64)
    loss, rec, kl = exams.cp6_vae_loss(logits, x, mu, log_var, beta=beta)

    lo, xo, m, lv = (t.numpy() for t in (logits, x, mu, log_var))
    p = 1 / (1 + np.exp(-lo))
    want_rec = -(xo * np.log(p) + (1 - xo) * np.log(1 - p)).reshape(shape[0], -1).sum(axis=1).mean()
    want_kl = (0.5 * (m ** 2 + np.exp(lv) - 1 - lv)).sum(axis=1).mean()
    assert all(isinstance(v, float) for v in (loss, rec, kl)), "expected three floats (loss, rec, kl)"
    assert abs(rec - want_rec) < 1e-9 * want_rec, f"expected rec = {want_rec:.10f} (BCE summed per image), got {rec:.10f}"
    assert abs(kl - want_kl) < 1e-9 * want_kl, f"expected kl = {want_kl:.10f} (KL summed per image), got {kl:.10f}"
    assert abs(loss - (want_rec + beta * want_kl)) < 1e-9 * loss, f"expected loss = rec + {beta} * kl, got {loss}"


def _small_gan(torch, seed):
    from torch import nn

    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
    D = nn.Sequential(nn.Linear(1, 8), nn.LeakyReLU(0.2), nn.Linear(8, 1))
    return (G, D, torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.5, 0.999)),
            torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.5, 0.999)))


def test_cp6_gan_step_is_the_non_saturating_step_with_detached_fakes():
    torch = pytest.importorskip("torch")
    G, D, opt_d, opt_g = _small_gan(torch, 3)
    G2, D2, opt_d2, opt_g2 = _small_gan(torch, 3)
    torch.manual_seed(30)
    reals = [4.0 + 1.25 * torch.randn(32, 1) for _ in range(3)]
    for step, real in enumerate(reals):
        torch.manual_seed(100 + step)
        out = exams.cp6_gan_step(G, D, real, opt_d, opt_g)
        assert isinstance(out, tuple) and all(isinstance(v, float) for v in out), f"expected (d_loss, g_loss), got {out!r}"
        torch.manual_seed(100 + step)                                 # the same draw of z, written another way
        fake = G2(torch.randn(len(real), 1))
        d_real, d_fake = torch.sigmoid(D2(real)), torch.sigmoid(D2(fake.detach()))
        d_loss = -(torch.log(d_real).mean() + torch.log(1 - d_fake).mean())
        opt_d2.zero_grad()
        d_loss.backward()
        opt_d2.step()
        g_loss = -torch.log(torch.sigmoid(D2(fake))).mean()          # non-saturating: -log D(G(z))
        opt_g2.zero_grad()
        g_loss.backward()
        opt_g2.step()
        assert np.allclose(out, (d_loss.item(), g_loss.item()), rtol=1e-5), \
            f"expected the losses {(d_loss.item(), g_loss.item())} at step {step + 1}, got {out}"
    for name, a, b in (("D", D, D2), ("G", G, G2)):
        for p, q in zip(a.parameters(), b.parameters()):
            assert torch.allclose(p, q, rtol=1e-5, atol=1e-7), f"expected the weights of {name} after 3 steps"


# ---------------------------------------------------------------------------
# The exam notebooks hold no solution in clear
# ---------------------------------------------------------------------------
def _exam_sources(part: int) -> str:
    nb = json.loads((ROOT / f"checkpoints/partie_{part}/02_examen_notebook.ipynb").read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in nb["cells"])


# Snippets of the solutions, of the hints moved to wb.exams and the letters of the 🐛 questions (CP4.7 a, CP5.4 a,
# CP6.9 a) that must not be readable in the exam notebook.
IN_CLEAR = {
    4: ["def oracle_step", "p *= 1 - lr * weight_decay", "np.sqrt(v / (1 - betas[1]", "clip_grad_norm_([W_t",
        "norm_i = np.sqrt", "DIKM"],
    5: ["def oracle_5", "def reference_fit", "optimizer.step()", 'reduction="sum"', "weight_ih_l0", "np.stack(states", "BDFI"],
    6: ["def oracle_8", "def reference_step", "fake.detach()", "opt_g.step()", "log_var.exp() - 1", "CGJ", "ligne J",
        "½(μ²", "exp(log_var) − 1", "label 1 aux faux", "BCE(D(faux), 1)", "détaché les faux", "opt_d.step() de trop"],
}


@pytest.mark.parametrize("part", [4, 5, 6], ids=["CP4", "CP5", "CP6"])
def test_exam_notebook_holds_no_solution_in_clear(part):
    sources = _exam_sources(part)
    found = [snippet for snippet in IN_CLEAR[part] if snippet in sources]
    assert not found, (f"expected no solution in clear in checkpoints/partie_{part}/02_examen_notebook.ipynb, found "
                       f"{found}; why: the learner has it open during the exam, the oracles live in wb.exams")
    assert "from wb import exams" in sources, f"expected part B of CP{part} to import its oracles from wb.exams"


def test_the_hints_used_by_the_exam_scripts_exist():
    source = (ROOT / "tools/chapters/build_cp6.py").read_text(encoding="utf-8")
    used = {(f"CP6.{n}", key) for n, expr in re.findall(r"hints_([89])\[([^\]]+)\]", source)
            for key in re.findall(r'"(\w+)"', expr)}                 # hints_9["a" if x else "b"] gives a and b
    missing = sorted(f"{q} {key}" for q, key in used if key not in exams.HINTS[q])
    assert used and not missing, f"expected every hint used by build_cp6.py in wb.exams.HINTS, missing {missing}"
    assert exams.HINTS["CP6.8"]["kl_wrong"].format(got="1,0", want="2,0").startswith("kl vaut 1,0, attendu 2,0")


def test_every_oracle_is_used_by_an_exam_script():
    scripts = "".join((ROOT / f"tools/chapters/build_cp{n}.py").read_text(encoding="utf-8") for n in (4, 5, 6))
    tree = ast.parse((ROOT / "src/wb/exams.py").read_text(encoding="utf-8"))
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")]
    unused = [name for name in functions if f"exams.{name}" not in scripts]
    assert functions and not unused, f"expected every oracle of wb.exams to be used by build_cp4/5/6.py, unused {unused}"
