"""Oracles of the code questions of the mock exams (part B of the exam notebooks, checkpoints IV to VI).

The learner keeps the exam notebook open DURING the exam: part A, the code questions, is done in it. Part B,
the checks run after the exam, sits just below, so it cannot hold in clear the code it compares against, nor
the hints that name a faulty line and its fix (BIBLE §22, audit P5 R3). Part B imports them from here::

    from wb import exams

    want = exams.cp6_vae_loss(LOGITS_V, X_V, MU_V, LOGVAR_V)          # (loss, rec, kl) as floats
    _, ref, ref_g, ref_d = run_steps(exams.cp6_gan_step)            # the GAN step of chapter 27
    print(exams.HINTS["CP6.9"]["d_other"])

Each oracle is the computation that part B of its exam used to define in its own cell, unchanged: the verdicts
of the checks do not depend on where it lives. Nothing here imports PyTorch at import time: the PyTorch oracles
import it when called (CP4 runs without it).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HINTS",
    "cp4_adamw_step",
    "cp4_torch_epoch",
    "cp4_train_epoch",
    "cp5_fit_early_stopping",
    "cp5_rnn",
    "cp6_gan_step",
    "cp6_vae_loss",
]


# ---------------------------------------------------------------------------
# CP4.9: AdamW and gradient clipping in NumPy
# ---------------------------------------------------------------------------
def cp4_adamw_step(params, grads, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """torch.optim.AdamW transcribed in NumPy (its pseudo-code, PyTorch 2.11), in place on the arrays of params.

    ``state`` is a dict, empty before the first step; it keeps ``t``, ``m`` and ``v``.
    """
    if not state:
        state.update(t=0, m=[np.zeros_like(p) for p in params], v=[np.zeros_like(p) for p in params])
    state["t"] += 1
    for p, g, m, v in zip(params, grads, state["m"], state["v"]):
        p *= 1 - lr * weight_decay
        m[...] = betas[0] * m + (1 - betas[0]) * g
        v[...] = betas[1] * v + (1 - betas[1]) * g * g
        p -= lr * (m / (1 - betas[0] ** state["t"])) / (np.sqrt(v / (1 - betas[1] ** state["t"])) + eps)


def cp4_train_epoch(params, X, y, state, lr, batch_size, max_norm, order, loss_and_grads):
    """One epoch in the given order: for each mini-batch of ``batch_size`` examples (the last one may be smaller),
    ``loss_and_grads``, the clipping by the global norm, then :func:`cp4_adamw_step` (its other settings by
    default). In place on params; returns the mean loss per example of the epoch."""
    total = 0.0
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        loss_i, grads_i = loss_and_grads(params, X[idx], y[idx])
        norm_i = np.sqrt(sum(np.sum(g ** 2) for g in grads_i))
        cp4_adamw_step(params, [g * min(1.0, max_norm / (norm_i + 1e-6)) for g in grads_i], state, lr)
        total += loss_i * len(idx)
    return total / len(X)


def cp4_torch_epoch(start_params, X, y, order, lr, batch_size, max_norm):
    """[W, b] (NumPy) after the same epoch of the softmax regression z = X @ W + b with torch.optim.AdamW
    (weight_decay 0.01) and torch.nn.utils.clip_grad_norm_. Raises ImportError when PyTorch is not installed."""
    import torch

    W_t, b_t = (torch.tensor(a, requires_grad=True) for a in start_params)
    opt_t = torch.optim.AdamW([W_t, b_t], lr=lr, weight_decay=0.01)
    X_t, Y_t = torch.from_numpy(X), torch.from_numpy(y)
    for start in range(0, len(X), batch_size):
        idx = torch.from_numpy(order[start:start + batch_size])
        loss_t = torch.nn.functional.cross_entropy(X_t[idx] @ W_t + b_t, Y_t[idx])
        opt_t.zero_grad()
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_([W_t, b_t], max_norm)
        opt_t.step()
    return [W_t.detach().numpy(), b_t.detach().numpy()]


# ---------------------------------------------------------------------------
# CP5.5 and CP5.6
# ---------------------------------------------------------------------------
def cp5_rnn(X, h0, Wx, Wh, b):
    """(output, h_n) of nn.RNN(batch_first=True) in float64 with W_ih = Wx.T, W_hh = Wh.T, b_ih = b, b_hh = 0:
    output (N, T, H), the states after each step, and h_n (N, H), the last one (NumPy arrays)."""
    import torch
    from torch import nn

    layer = nn.RNN(Wx.shape[0], Wx.shape[1], batch_first=True).double()
    with torch.no_grad():
        layer.weight_ih_l0.copy_(torch.from_numpy(Wx.T.copy()))
        layer.weight_hh_l0.copy_(torch.from_numpy(Wh.T.copy()))
        layer.bias_ih_l0.copy_(torch.from_numpy(b.copy()))
        layer.bias_hh_l0.zero_()
        out, h_n = layer(torch.from_numpy(X.copy()), torch.from_numpy(h0.copy()).unsqueeze(0))
    return out.numpy(), h_n[0].numpy()


def cp5_fit_early_stopping(model, train_loader, val_loader, max_epochs, patience, path, lr=1e-3):
    """The contract of CP5.6 a), the oracle of its checks: Adam(lr), per epoch the mean training loss per example
    (model.train()) then the validation cross-entropy per example (model.eval(), no gradient); the state_dict saved
    in ``path`` at each strict progress, a stop after ``patience`` epochs in a row without one, the saved weights
    reloaded at the end. Returns {"train_loss": [...], "val_loss": [...], "best_epoch": k} (epochs from 1)."""
    import torch
    from torch import nn

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best, waited = {"train_loss": [], "val_loss": [], "best_epoch": None}, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for X, y in train_loader:
            loss = nn.functional.cross_entropy(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, count = total + loss.item() * len(y), count + len(y)
        history["train_loss"].append(total / count)
        model.eval()
        with torch.no_grad():
            total = sum(nn.functional.cross_entropy(model(X), y, reduction="sum").item() for X, y in val_loader)
        history["val_loss"].append(total / len(val_loader.dataset))
        if history["val_loss"][-1] < best:
            best, waited, history["best_epoch"] = history["val_loss"][-1], 0, epoch
            torch.save(model.state_dict(), path)
        else:
            waited += 1
            if waited == patience:
                break
    model.load_state_dict(torch.load(path, weights_only=True))
    return history


# ---------------------------------------------------------------------------
# CP6.8 and CP6.9
# ---------------------------------------------------------------------------
def cp6_vae_loss(logits, x, mu, log_var, beta=1.0):
    """(loss, rec, kl) as floats: the binary cross-entropy on the logits summed over the pixels and the Gaussian KL
    summed over the latents, both per example (divided by the number of images); loss = rec + beta * kl."""
    import torch.nn.functional as F

    n = x.shape[0]
    rec = float(F.binary_cross_entropy_with_logits(logits, x, reduction="sum")) / n
    kl = float(0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum()) / n
    return rec + beta * kl, rec, kl


def cp6_gan_step(G, D, real, opt_d, opt_g):
    """The GAN step of chapter 27: D on real and detached fakes, then G with the label 1 (non-saturating loss).
    One call to torch.randn, before the rest; the losses are the means over the mini-batch (BCEWithLogitsLoss).
    Returns (d_loss, g_loss) as floats."""
    import torch
    import torch.nn.functional as F

    def bce(logits, target):
        return F.binary_cross_entropy_with_logits(logits, target)        # what nn.BCEWithLogitsLoss() computes

    ones, zeros = torch.ones(len(real), 1), torch.zeros(len(real), 1)
    fake = G(torch.randn(len(real), 1))
    d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()
    g_loss = bce(D(fake), ones)
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()
    return d_loss.item(), g_loss.item()


# ---------------------------------------------------------------------------
# The hints of part B that give a formula to remember or name a faulty line with its fix
# ---------------------------------------------------------------------------
HINTS = {
    "CP6.8": {
        "kl_ok": "kl : ½(μ² + σ² − 1 − ln σ²) sommée sur les latents, moyennée sur les images.",
        "kl_wrong": "kl vaut {got}, attendu {want} : ½(μ² + exp(log_var) − 1 − log_var), sommée sur les latents, "
                    "divisée par le nombre d'images.",
    },
    "CP6.9": {
        "freed_graph": " (un backward à travers un graphe déjà libéré : le pas de D a-t-il détaché les faux ?)",
        "d_after_first": "le premier pas de D est juste ; l'écart vient ensuite du pas du générateur "
                         "(un opt_d.step() de trop, ou un générateur qui a bougé autrement).",
        "d_still_g": "l'écart vient du pas du générateur : le générateur n'a pas bougé, et D a fait un second pas avec "
                     "les gradients de g_loss (ligne J : c'est opt_g qui fait le pas du générateur).",
        "d_other": "vrais exemples avec le label 1, faux DÉTACHÉS avec le label 0, et un seul tirage torch.randn (ligne B).",
        "g_still": "il n'a pas bougé, c'est opt_g qui fait son pas (ligne J).",
        "g_other": "sa loss donne le label 1 aux faux (non saturante), et c'est opt_g qui fait le pas.",
        "losses_wrong": "les losses rendues diffèrent de celles du pas de référence : d_loss = BCE(D(vrais), 1) + "
                        "BCE(D(faux), 0), g_loss = BCE(D(faux), 1).",
    },
}
