#!/usr/bin/env python
"""Draw the figures of chapter 26 (used by Claude).

    python tools/chapters/figures_ch26.py

Writes small PNG files into chapitres/ch26_rl/figures/. Every figure is computed or drawn here (never copied from
the book): the agent-environment loop with the names of the Gymnasium API, the weights gamma**k of four discount
factors, a Flippers board, a move and the six winning boards, one Q-learning update on a small table, and the
action probabilities of epsilon-greedy and softmax for one row of a table. None of them uses the numbers of an
exercise: the gammas, the table values, epsilon and the temperatures differ from those of 02_exercices.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch26_rl" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma, a thin space for thousands and a true minus sign."""
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


def arrow(ax, start, end, color=INK, lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def box(ax, xy, w, h, text, color, text_color="white", fontsize=11):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=color, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def figure_loop() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    box(ax, (0.4, 1.4), 2.8, 1.3, "agent\n(politique, table Q)", C[0])
    box(ax, (6.8, 1.4), 2.8, 1.3, "environnement", C[1])
    arrow(ax, (3.3, 2.5), (6.7, 2.5))
    ax.text(5.0, 2.75, "action $a_t$", ha="center", fontsize=10, color=INK)
    ax.text(5.0, 3.35, "env.step(action)", ha="center", fontsize=9, color=MUTED, family="monospace")
    arrow(ax, (6.7, 1.65), (3.3, 1.65))
    ax.text(5.0, 1.05, "état $s_{t+1}$, récompense $r_t$,\nterminated, truncated, info", ha="center", fontsize=9.5,
            color=INK)
    ax.text(5.0, 0.25, "state, reward, terminated, truncated, info = env.step(action)", ha="center", fontsize=8.5,
            color=MUTED, family="monospace")
    ax.text(1.8, 3.6, "state, info = env.reset(seed=…)", ha="center", fontsize=8.5, color=MUTED, family="monospace")
    arrow(ax, (1.8, 3.4), (1.8, 2.75), color=MUTED, lw=1)
    save(fig, "boucle.png")


def figure_gamma_weights() -> None:
    k = np.arange(0, 25)
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for gamma, color in zip((0.4, 0.75, 0.95, 1.0), C):
        ax.plot(k, gamma ** k, "o-", ms=3.5, lw=1.3, color=color, label=f"γ = {french_number(gamma)}")
    ax.set_xlabel("nombre de pas $k$ avant la récompense")
    ax.set_ylabel("poids $\\gamma^k$")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("Une récompense reçue dans $k$ pas compte pour $\\gamma^k$ fois sa valeur", fontsize=10)
    save(fig, "poids_gamma.png")


def _draw_board(ax, bits, x0, y0, size=1.0, highlight=None, title=None):
    """A 3 x 3 Flippers board: bit i (row-major) = 1 means a dot."""
    cell = size / 3
    for i in range(9):
        r, c = divmod(i, 3)
        x, y = x0 + c * cell, y0 + (2 - r) * cell
        edge = C[3] if highlight == i else INK
        ax.add_patch(Rectangle((x, y), cell, cell, fc="#f4f4f4", ec=edge, lw=2.4 if highlight == i else 0.8))
        if bits[i]:
            ax.add_patch(plt.Circle((x + cell / 2, y + cell / 2), cell * 0.22, color=C[0]))
    if title:
        ax.text(x0 + size / 2, y0 - 0.18, title, ha="center", va="top", fontsize=8.5, color=INK)


def figure_flippers() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    ax.set_xlim(-0.1, 10.2)
    ax.set_ylim(-0.6, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    start = [1, 0, 0, 0, 1, 0, 0, 1, 1]                 # an example board (not one of the exercises)
    _draw_board(ax, start, 0.0, 0.1, highlight=0, title="départ : on retourne\nla tuile encadrée")
    after = start.copy()
    after[0] = 0
    arrow(ax, (1.15, 0.6), (1.6, 0.6))
    _draw_board(ax, after, 1.75, 0.1, title="après le coup")
    ax.text(3.15, 0.6, "gagnants :", ha="left", va="center", fontsize=9.5, color=INK)
    winners = []
    for r in range(3):
        b = [0] * 9
        b[3 * r:3 * r + 3] = [1, 1, 1]
        winners.append(b)
    for c in range(3):
        b = [0] * 9
        b[c], b[c + 3], b[c + 6] = 1, 1, 1
        winners.append(b)
    for j, b in enumerate(winners):
        _draw_board(ax, b, 4.3 + j * 0.98, 0.25, size=0.8)
    save(fig, "flippers.png")


def figure_q_update() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    q_s = [0.30, 0.55, 0.20, 0.10]
    q_next = [0.25, 0.70, 0.40, 0.05]
    a = 1
    alpha, gamma, r = 0.25, 0.9, 0.2
    target = r + gamma * max(q_next)
    new = q_s[a] + alpha * (target - q_s[a])

    def row(y, values, label, mark=None, color=C[0]):
        ax.text(0.3, y + 0.3, label, ha="left", va="center", fontsize=10, color=INK)
        for j, v in enumerate(values):
            fc = color if j == mark else "#f4f4f4"
            ax.add_patch(Rectangle((1.6 + j * 1.0, y), 1.0, 0.6, fc=fc, ec=INK, lw=0.8))
            ax.text(2.1 + j * 1.0, y + 0.3, french_number(v), ha="center", va="center", fontsize=10,
                    color="white" if j == mark else INK)
        for j in range(len(values)):
            ax.text(2.1 + j * 1.0, 4.15, f"$a_{j}$", ha="center", fontsize=9, color=MUTED)

    row(3.2, q_s, "ligne de $s$", mark=a, color=C[3])
    row(1.6, q_next, "ligne de $s'$", mark=int(np.argmax(q_next)), color=C[0])
    text = (f"cible $y = r + \\gamma \\max_{{a'}} Q(s', a')$ = {french_number(r)} + {french_number(gamma)} × "
            f"{french_number(max(q_next))} = {french_number(round(target, 3))}\n"
            f"$Q(s, a_1) \\leftarrow$ {french_number(q_s[a])} + {french_number(alpha)} × "
            f"({french_number(round(target, 3))} − {french_number(q_s[a])}) = {french_number(round(new, 4))}")
    ax.text(6.1, 2.55, text, ha="left", va="center", fontsize=9.5, color=INK)
    ax.text(6.1, 1.3, f"récompense $r$ = {french_number(r)}, $\\alpha$ = {french_number(alpha)}, "
            f"$\\gamma$ = {french_number(gamma)}", ha="left", va="center", fontsize=9, color=MUTED)
    arrow(ax, (2.6, 2.25), (2.6, 3.15), color=C[0], rad=0.0)
    save(fig, "mise_a_jour_q.png")


def figure_policies() -> None:
    q = np.array([0.6, 1.0, 0.2, 0.9, 0.4])
    n = len(q)
    eps = 0.2
    greedy = int(np.argmax(q))
    p_eps = np.full(n, eps / n)
    p_eps[greedy] += 1 - eps
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    panels = [(p_eps, f"ε-greedy, ε = {french_number(eps)}")]
    for tau in (0.5, 0.1):
        z = q / tau
        p = np.exp(z - z.max())
        panels.append((p / p.sum(), f"softmax, τ = {french_number(tau)}"))
    for ax, (p, title) in zip(axes, panels):
        ax.bar(range(n), p, color=[C[3] if j == greedy else C[0] for j in range(n)])
        ax.set_xticks(range(n), [f"$a_{j}$\n{french_number(v)}" for j, v in enumerate(q)], fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("probabilité de jouer l'action")
    fig.suptitle("Une même ligne de la table (valeurs Q sous chaque action), trois politiques", fontsize=10)
    save(fig, "politiques.png")


def main() -> int:
    figure_loop()
    figure_gamma_weights()
    figure_flippers()
    figure_q_update()
    figure_policies()
    print(f"Figures -> {OUT.relative_to(ROOT)}: {sorted(p.name for p in OUT.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
