#!/usr/bin/env python
"""Draw the figure of the mock exam of part VI (question CP6.7), used by Claude.

    python tools/chapters/figures_cp6.py

Writes checkpoints/partie_6/figures/cp6_7_courbes.png and tools/chapters/history_cp6.json: three real trainings,
made here (about 30 s on CPU), nothing from the book:
- panel 1: a dense VAE 784-256-(8)-256-784 on 6 000 MNIST images, Adam(1e-3), mini-batches of 128, 15 epochs, with
  beta = 60 (the reconstruction summed over the pixels and the KL, both per example, as in chapter 25);
- panel 2: a 1D GAN (target N(4, 1.25^2), G 1-16-1 tanh, D 1-32-32-1 LeakyReLU), Adam(beta1 = 0.5), the
  discriminator at lr 1e-2 and the generator at lr 1e-5, 3 000 steps of 64, losses averaged per 100 steps;
- panel 3: tabular Q-learning (mylearn_ref.rl.train_tabular) on the slippery FrozenLake 8x8, alpha = 0.1,
  gamma = 0.99, epsilon = 0.1, episodes cut at 60 steps, 3 000 episodes, return averaged per 100 episodes.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import wb  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "checkpoints" / "partie_6" / "figures"
HISTORY_FILE = ROOT / "tools" / "chapters" / "history_cp6.json"
C = wbplot.CATEGORICAL
DPI = 110


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma and a true minus sign."""
    return f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def run_vae(beta: float = 60.0, epochs: int = 15) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(2)
    X, _ = wb.datasets.load_mnist("train", 6000, seed=61)
    X = torch.tensor(X / 255.0, dtype=torch.float32).flatten(1)

    class VAE(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
            self.mu, self.lv = nn.Linear(256, d), nn.Linear(256, d)
            self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 784))

        def forward(self, x):
            h = self.enc(x)
            mu, lv = self.mu(h), self.lv(h)
            return self.dec(mu + torch.exp(0.5 * lv) * torch.randn_like(mu)), mu, lv

    torch.manual_seed(61)
    model, gen = VAE(), torch.Generator().manual_seed(61)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    rec_h, kl_h = [], []
    for _ in range(epochs):
        tr = tk = 0.0
        for idx in torch.randperm(len(X), generator=gen).split(128):
            x = X[idx]
            logits, mu, lv = model(x)
            rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
            kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum() / len(x)
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr, tk = tr + rec.item() * len(x), tk + kl.item() * len(x)
        rec_h.append(tr / len(X))
        kl_h.append(tk / len(X))
    return {"rec": rec_h, "kl": kl_h, "beta": beta}


def run_gan(lr_d: float = 1e-2, lr_g: float = 1e-5, steps: int = 3000) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(61)
    G = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
    D = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
    od = torch.optim.Adam(D.parameters(), lr_d, betas=(0.5, 0.999))
    og = torch.optim.Adam(G.parameters(), lr_g, betas=(0.5, 0.999))
    ones, zeros = torch.ones(64, 1), torch.zeros(64, 1)
    d_h, g_h, d_run, g_run = [], [], [], []
    for s in range(steps):
        real, fake = 4 + 1.25 * torch.randn(64, 1), G(torch.randn(64, 1))
        ld = F.binary_cross_entropy_with_logits(D(real), ones) + F.binary_cross_entropy_with_logits(D(fake.detach()), zeros)
        od.zero_grad()
        ld.backward()
        od.step()
        lg = F.binary_cross_entropy_with_logits(D(fake), ones)
        og.zero_grad()
        lg.backward()
        og.step()
        d_run.append(ld.item())
        g_run.append(lg.item())
        if (s + 1) % 100 == 0:
            d_h.append(float(np.mean(d_run)))
            g_h.append(float(np.mean(g_run)))
            d_run, g_run = [], []
    with torch.no_grad():
        sample = G(torch.randn(2000, 1))
    return {"d_loss": d_h, "g_loss": g_h, "sample_mean": float(sample.mean()), "sample_std": float(sample.std())}


def run_rl(episodes: int = 3000) -> dict:
    import mylearn_ref.rl as rl
    env = wb.datasets.make_env("FrozenLake-v1", seed=61, map_name="8x8")
    _, hist = rl.train_tabular(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1, max_steps=60,
                               rng=np.random.default_rng(61))
    returns = np.asarray(hist["returns"], dtype=float)
    return {"returns_per_100": [float(returns[i:i + 100].mean()) for i in range(0, episodes, 100)],
            "wins": int(returns.sum())}


def run_flippers(seeds=(0, 1, 2)) -> dict:
    """CP6.6: the share of the 506 non-winning boards that the greedy policy wins in fewer than 100 moves, after
    3 000 games of Q-learning (alpha = 0.5, Q = 0 at the start), for three settings of (gamma, epsilon)."""
    import mylearn_ref.rl as rl
    from wb import envs
    out = {}
    for gamma, epsilon in [(0.9, 0.1), (0.0, 0.1), (0.9, 0.0)]:
        shares, lengths = [], []
        for seed in seeds:
            Q, _ = rl.train_tabular(envs.Flippers(seed=seed), 3000, alpha=0.5, gamma=gamma, epsilon=epsilon,
                                    max_steps=100, rng=np.random.default_rng(seed))
            policy, env, won, moves = rl.greedy_policy(Q), envs.Flippers(), 0, []
            for s in range(512):
                if s in env.winning:
                    continue
                state, _ = env.reset(options={"state": s})
                for k in range(100):
                    state, _, terminated, _, _ = env.step(int(policy[state]))
                    if terminated:
                        won += 1
                        moves.append(k + 1)
                        break
            shares.append(won / (512 - len(env.winning)))
            lengths.append(float(np.mean(moves)))
        out[f"gamma={gamma},epsilon={epsilon}"] = {"shares": shares, "mean_moves": lengths}
    return out


def fig_curves(h: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    epochs = np.arange(1, len(h["vae"]["rec"]) + 1)
    ax.plot(epochs, h["vae"]["rec"], color=C[0], lw=2, marker="o", ms=3, label="reconstruction (par image)")
    ax.set(title="Courbes ① : un VAE sur MNIST", xlabel="epoch", ylabel="reconstruction (nats)")
    ax.set_ylim(0, 340)
    twin = ax.twinx()
    twin.plot(epochs, h["vae"]["kl"], color=C[1], lw=2, ls="--", marker="s", ms=3, label="KL (par image)")
    twin.set_ylabel("KL (nats)")
    twin.set_ylim(0, 15)
    twin.grid(False)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax.grid(True, alpha=0.45)
    ax = axes[1]
    steps = np.arange(1, len(h["gan"]["d_loss"]) + 1) * 100
    ax.plot(steps, h["gan"]["d_loss"], color=C[0], lw=2, label="loss de D")
    ax.plot(steps, h["gan"]["g_loss"], color=C[1], lw=2, ls="--", label="loss de G")
    ax.set(title="Courbes ② : un GAN sur une loi 1D", xlabel="pas d'entraînement", ylabel="loss (BCE, moyenne sur 100 pas)")
    ax.set_ylim(0, 10.5)
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.45)
    ax = axes[2]
    episodes = np.arange(1, len(h["rl"]["returns_per_100"]) + 1) * 100
    ax.plot(episodes, h["rl"]["returns_per_100"], color=C[2], lw=2, marker="o", ms=3, label="retour moyen (100 épisodes)")
    ax.set(title="Courbes ③ : Q-learning sur FrozenLake 8 × 8", xlabel="épisode", ylabel="retour moyen")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.45)
    fig.suptitle("Question CP6.7 : trois entraînements réels", y=1.02)
    fig.tight_layout()
    save(fig, "cp6_7_courbes.png")


def main() -> int:
    warnings.filterwarnings("ignore")
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    history = {"vae": run_vae(), "gan": run_gan(), "rl": run_rl(), "flippers": run_flippers()}
    HISTORY_FILE.write_text(json.dumps(history, indent=1), encoding="utf-8")
    fig_curves(history)
    v, g, r = history["vae"], history["gan"], history["rl"]
    print(f"  VAE: rec {v['rec'][0]:.1f} -> {v['rec'][-1]:.1f}, KL {v['kl'][0]:.2f} -> {v['kl'][-1]:.3f}")
    print(f"  GAN: D {g['d_loss'][0]:.3f} -> {g['d_loss'][-1]:.3f}, G {g['g_loss'][0]:.2f} -> {g['g_loss'][-1]:.2f}, "
          f"samples {g['sample_mean']:.2f} ± {g['sample_std']:.2f}")
    print(f"  RL: return per 100 episodes {r['returns_per_100'][0]:.2f} -> {r['returns_per_100'][-1]:.2f}, "
          f"{r['wins']} wins in 3000 episodes")
    for key, value in history["flippers"].items():
        print(f"  Flippers {key}: shares {np.round(value['shares'], 3).tolist()}, moves {np.round(value['mean_moves'], 2).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
