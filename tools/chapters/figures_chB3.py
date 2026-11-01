#!/usr/bin/env python
"""Draw the figures of chapter B3 (used by Claude).

    python tools/chapters/figures_chB3.py        # draw the seven figures (offline, a few seconds)

Writes small PNG files into chapitres/b3_transformers/figures/:

- ``goulot.png``: the recurrent encoder-decoder and its single context vector, next to Bahdanau's attention (the
  decoder weights every encoder state at each step; the weights drawn are illustrative, not learned);
- ``qkv.png``: the worked example of section 203.2 (one query, four keys and values): scores, softmax weights, and
  the output as the weighted average of the value vectors, all computed here;
- ``masques.png``: boolean masks (True = may attend): causal, padding of a batch, both combined, and the causal
  mask of 2 queries against 6 keys aligned bottom-right (KV cache) or top-left (``is_causal=True``);
- ``tetes.png``: split_heads as a diagram, (N, T, d) -> (N, T, H, d_h) -> (N, H, T, d_h);
- ``positions.png``: the sinusoidal encoding of 64 positions in dimension 64, and one RoPE pair rotated at
  positions 0 to 5;
- ``bloc.png``: a post-LN block (2017) next to a pre-LN block (GPT-2);
- ``cout.png``: theoretical costs, no timing: the memory of the attention weights of one layer against the length,
  and the share of attention in the FLOPs of a causal decoder layer, T / (T + 12 d).

Nothing here is trained or measured: the notebook measures (🔬 B3.27) and draws the learned attention maps
(📈 B3.24). None of the figures uses the numbers of an exercise of 02_exercices.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "b3_transformers" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED
ALLOWED, MASKED = "#3987e5", "#f0efec"

# --- the worked example of section 203.2 (also in 01_fiche.md) ------------------------------------------------
Q_EX = np.array([1.0, 0.0, 1.0, 0.0])
K_EX = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [2.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
V_EX = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, 1.0]])


def french(x: float, digits: int = 2) -> str:
    """A number with a French decimal comma and a true minus sign."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fontsize=9.5, color=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)


def arrow(ax, start, end, color=INK, lw=1.0, rad=0.0, style="-|>", alpha=1.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9, color=color, lw=lw, alpha=alpha,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


# --- 1. the bottleneck and the attention ------------------------------------------------------------------------
def figure_bottleneck() -> None:
    src, tgt = ["the", "black", "cat", "sleeps"], ["le", "chat", "noir", "dort"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for k, ax in enumerate(axes):
        xs = np.arange(4) * 1.25
        for i, (x, w) in enumerate(zip(xs, src)):
            box(ax, x, 0.6, 0.9, 0.5, f"$\\mathbf{{h}}_{i + 1}$", fc="#cde2fb")
            ax.text(x, 0.05, w, ha="center", va="center", fontsize=9.5, color=MUTED)
            arrow(ax, (x, 0.2), (x, 0.33), color=MUTED, lw=0.8)
            if i:
                arrow(ax, (xs[i - 1] + 0.45, 0.6), (x - 0.45, 0.6), lw=0.9)
        dx = np.arange(4) * 1.25 + 0.6
        for i, (x, w) in enumerate(zip(dx, tgt)):
            box(ax, x, 2.6, 0.9, 0.5, f"$\\mathbf{{s}}_{i + 1}$", fc="#fde0cf")
            ax.text(x, 3.15, w, ha="center", va="center", fontsize=9.5, color=INK)
            arrow(ax, (x, 2.85), (x, 3.0), color=MUTED, lw=0.8)
            if i:
                arrow(ax, (dx[i - 1] + 0.45, 2.6), (x - 0.45, 2.6), lw=0.9)
        if k == 0:
            box(ax, 4.65, 1.6, 0.8, 0.55, "$\\mathbf{c}$", fc="#fbe8b5", lw=1.6)
            arrow(ax, (xs[3] + 0.45, 0.7), (4.3, 1.45), lw=1.2)
            arrow(ax, (4.3, 1.8), (dx[0] + 0.1, 2.33), lw=1.2, rad=0.25)
            ax.text(5.15, 1.6, "goulot : un seul\nvecteur pour\ntoute la phrase", ha="left", va="center",
                    fontsize=8.8, color=MUTED)
            ax.set_title("Encodeur-décodeur récurrent", fontsize=10.5)
        else:
            target, weights = 2, [0.05, 0.78, 0.12, 0.05]          # illustrative weights for « noir »
            for i, (x, wgt) in enumerate(zip(xs, weights)):
                ax.add_patch(FancyArrowPatch((x, 0.88), (dx[target] - 0.15 + 0.1 * i, 2.33), arrowstyle="-|>",
                                             mutation_scale=9, color=C[1], lw=0.6 + 5.0 * wgt, alpha=0.35 + 0.65 * wgt,
                                             shrinkA=2, shrinkB=2))
                ax.text(x + 0.12, 1.25 + 0.06 * i, "$\\alpha$ = " + french(wgt), fontsize=8.3, color=C[1],
                        ha="left")
            ax.text(5.75, 1.75, "contexte propre au pas :\n$\\mathbf{c}_3 = \\sum_j \\alpha_{3j}\\,\\mathbf{h}_j$",
                    ha="center", va="center", fontsize=9, color=INK)
            ax.set_title("Avec attention (Bahdanau et coll.) : le pas « noir »", fontsize=10.5)
        ax.text(-0.75, 0.6, "encodeur", rotation=90, ha="center", va="center", fontsize=9, color=MUTED)
        ax.text(-0.75, 2.6, "décodeur", rotation=90, ha="center", va="center", fontsize=9, color=MUTED)
        ax.set_xlim(-1.0, 6.9)
        ax.set_ylim(-0.25, 3.45)
        ax.axis("off")
    fig.text(0.75, 0.0, "poids illustratifs, non appris", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "goulot.png")


# --- 2. one query against four keys ------------------------------------------------------------------------------
def figure_qkv() -> None:
    scores = K_EX @ Q_EX / np.sqrt(Q_EX.size)
    w = np.exp(scores - scores.max())
    w /= w.sum()
    out = w @ V_EX
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    labels = [f"token {j}" for j in range(4)]
    colors = [C[0], C[2], C[1], C[4]]
    ax = axes[0]
    ax.bar(labels, scores, color=colors)
    for j, s in enumerate(scores):
        ax.text(j, s + 0.05, french(s, 1), ha="center", va="bottom", fontsize=9)
    ax.set_title("1. scores $\\mathbf{q}\\cdot\\mathbf{k}_j / \\sqrt{d_k}$", fontsize=10.5)
    ax.set_ylim(0, 2.4)
    ax = axes[1]
    ax.bar(labels, w, color=colors)
    for j, a in enumerate(w):
        ax.text(j, a + 0.015, french(a, 3), ha="center", va="bottom", fontsize=9)
    ax.set_title("2. softmax sur les clés : poids de somme 1", fontsize=10.5)
    ax.set_ylim(0, 0.75)
    ax = axes[2]
    spots = [(1.08, -0.02, "left"), (0.0, 1.55, "center"), (2.06, 2.08, "left"), (-1.2, 1.2, "center")]
    for j, (v, col) in enumerate(zip(V_EX, colors)):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
        x, y, ha = spots[j]
        ax.text(x, y, f"$\\mathbf{{v}}_{j}$ (poids {french(w[j], 2)})", fontsize=8.8, color=col, ha=ha, va="center")
    ax.annotate("", xy=out, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.4))
    ax.text(1.45, 0.62, f"sortie $\\sum_j \\alpha_j \\mathbf{{v}}_j$\n= ({french(out[0])} ; "
            f"{french(out[1])})", fontsize=9, color=INK)
    ax.set_xlim(-2.4, 3.2)
    ax.set_ylim(-0.4, 2.5)
    ax.set_aspect("equal")
    ax.axhline(0, color=wbplot.GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=wbplot.GRID, lw=0.8, zorder=0)
    ax.set_title("3. moyenne des valeurs pondérée", fontsize=10.5)
    fig.suptitle("La requête du token 2 face aux quatre clés ($d_k = 4$, $d_v = 2$)", fontsize=11, y=1.02)
    save(fig, "qkv.png")


# --- 3. masks ------------------------------------------------------------------------------------------------
def draw_mask(ax, mask, title, xlabel="clé $j$", ylabel="requête $i$"):
    cmap = ListedColormap([MASKED, ALLOWED])
    ax.imshow(mask.astype(int), cmap=cmap, vmin=0, vmax=1)
    rows, cols = mask.shape
    for i in range(rows):
        for j in range(cols):
            if not mask[i, j]:
                ax.text(j, i, "×", ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks(np.arange(-0.5, cols), minor=True)
    ax.set_yticks(np.arange(-0.5, rows), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.8)


def figure_masks() -> None:
    T = 6
    causal = np.tril(np.ones((T, T), dtype=bool))
    lengths = np.array([4, 6])
    pad = np.arange(T)[None, :] < lengths[:, None]
    L, S = 2, 6
    bottom_right = np.tril(np.ones((L, S), dtype=bool), k=S - L)
    top_left = np.tril(np.ones((L, S), dtype=bool), k=0)
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.5], hspace=0.55, wspace=0.9)
    draw_mask(fig.add_subplot(gs[0, 0:2]), causal, "causal : $j \\leq i$")
    draw_mask(fig.add_subplot(gs[0, 2:4]), pad, "padding : longueurs 4 et 6", ylabel="séquence $n$")
    draw_mask(fig.add_subplot(gs[0, 4:6]), causal & pad[0][None, :], "causal ET padding (séquence 0)")
    draw_mask(fig.add_subplot(gs[1, 0:3]), bottom_right,
              "2 requêtes, 6 clés, aligné en bas à droite :\n$j \\leq i + S - L$ (KV cache, causal_mask(2, 6))")
    draw_mask(fig.add_subplot(gs[1, 3:6]), top_left,
              "2 requêtes, 6 clés, aligné en haut à gauche :\n$j \\leq i$ (is_causal=True quand $L \\neq S$)")
    fig.suptitle("Masques booléens : bleu = la requête peut regarder la clé (True), × = masqué (False)",
                 fontsize=11, y=0.99)
    save(fig, "masques.png")


# --- 4. split into heads -------------------------------------------------------------------------------------
def figure_heads() -> None:
    T, H, dh = 3, 4, 2
    d = H * dh
    head_colors = [C[0], C[1], C[2], C[3]]
    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    cw, chh = 0.42, 0.42

    def cell(x, y, h, tok, feat):
        ax.add_patch(Rectangle((x, y), cw, chh, fc=head_colors[h], ec="white", lw=1.2, alpha=0.35 + 0.18 * tok))
        ax.text(x + cw / 2, y + chh / 2, f"{tok},{feat}", ha="center", va="center", fontsize=6.8, color=INK)

    # (T, d)
    x0, y0 = 0.0, 1.6
    for t in range(T):
        for f in range(d):
            cell(x0 + f * cw, y0 + (T - 1 - t) * chh, f // dh, t, f)
        ax.text(x0 - 0.12, y0 + (T - 1 - t) * chh + chh / 2, f"token {t}", ha="right", va="center", fontsize=8.5)
    ax.text(x0 + d * cw / 2, y0 + T * chh + 0.25, "$(T, d)$ = (3, 8)", ha="center", fontsize=10)
    ax.text(x0 + d * cw / 2, y0 - 0.35, "cases « token, feature » ; couleur = tête", ha="center", fontsize=8.3,
            color=MUTED)
    arrow(ax, (x0 + d * cw + 0.15, y0 + 0.6), (x0 + d * cw + 1.05, y0 + 0.6), lw=1.2)
    ax.text(x0 + d * cw + 0.6, y0 + 0.85, "reshape", ha="center", fontsize=8.8)
    # (T, H, d_h): each token row cut into H blocks
    x1 = x0 + d * cw + 1.2
    for t in range(T):
        for h in range(H):
            for k in range(dh):
                cell(x1 + h * (dh * cw + 0.15) + k * cw, y0 + (T - 1 - t) * chh, h, t, h * dh + k)
    ax.text(x1 + H * (dh * cw + 0.15) / 2, y0 + T * chh + 0.25, "$(T, H, d_h)$ = (3, 4, 2)", ha="center",
            fontsize=10)
    x_end = x1 + H * (dh * cw + 0.15)
    arrow(ax, (x_end + 0.1, y0 + 0.6), (x_end + 1.0, y0 + 0.6), lw=1.2)
    ax.text(x_end + 0.55, y0 + 0.85, "échanger les\naxes 1 et 2", ha="center", fontsize=8.8)
    # (H, T, d_h): one full sequence per head
    x2 = x_end + 1.2
    for h in range(H):
        bx = x2 + h * (dh * cw + 0.35)
        for t in range(T):
            for k in range(dh):
                cell(bx + k * cw, y0 + (T - 1 - t) * chh, h, t, h * dh + k)
        ax.text(bx + dh * cw / 2, y0 - 0.2, f"tête {h}", ha="center", va="top", fontsize=8.5, color=head_colors[h])
    ax.text(x2 + H * (dh * cw + 0.35) / 2, y0 + T * chh + 0.25, "$(H, T, d_h)$ = (4, 3, 2)", ha="center",
            fontsize=10)
    ax.text(x2 + H * (dh * cw + 0.35) / 2, y0 - 0.65, "chaque tête : une séquence complète\nde $T$ vecteurs de "
            "taille $d_h$", ha="center", va="top", fontsize=8.3, color=MUTED)
    arrow(ax, (x2 + H * (dh * cw + 0.35) / 2, 0.45), (x0 + d * cw / 2, 0.45), lw=1.0, color=MUTED, rad=0.0)
    ax.text((x2 + x0 + d * cw) / 2 + 1.0, 0.18, "merge_heads : échanger à nouveau, puis reshape en $(T, d)$ "
            "(la concaténation)", ha="center", fontsize=8.8, color=MUTED)
    ax.set_xlim(-1.2, x2 + H * (dh * cw + 0.35) + 0.2)
    ax.set_ylim(-0.1, y0 + T * chh + 0.6)
    ax.axis("off")
    ax.set_title("split_heads pour une séquence ($N$ = 1) : $d$ = 8 features, $H$ = 4 têtes de $d_h$ = 2",
                 fontsize=10.5)
    save(fig, "tetes.png")


# --- 5. positions -------------------------------------------------------------------------------------------
def sinusoidal(T: int, d: int, base: float = 10000.0) -> np.ndarray:
    p = np.arange(T)[:, None]
    omega = base ** (-np.arange(0, d, 2) / d)
    pe = np.zeros((T, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(p * omega), np.cos(p * omega)
    return pe


def figure_positions() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.45})
    pe = sinusoidal(64, 64)
    ax = axes[0]
    im = ax.imshow(pe, cmap=wbplot.diverging_cmap(), vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("colonne (paire $i$ = colonne // 2)")
    ax.set_ylabel("position $p$")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Encodage sinusoïdal, 64 positions, $d$ = 64", fontsize=10.5)
    ax = axes[1]
    theta = 0.5
    v = np.array([1.0, 0.35])
    t = np.linspace(0, 2 * np.pi, 200)
    r = np.linalg.norm(v)
    ax.plot(r * np.cos(t), r * np.sin(t), color=wbplot.GRID, lw=1.0)
    for m in range(6):
        a = m * theta
        rv = np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[1] * np.cos(a) + v[0] * np.sin(a)])
        col = wbplot.sequential_cmap()(0.35 + 0.12 * m)
        ax.annotate("", xy=rv, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.text(rv[0] * 1.13, rv[1] * 1.13, f"$m = {m}$", ha="center", va="center", fontsize=8.8, color=INK)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.3, 1.45)
    ax.set_aspect("equal")
    ax.axhline(0, color=wbplot.GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=wbplot.GRID, lw=0.8, zorder=0)
    ax.set_xlabel("coordonnée $i$")
    ax.set_ylabel("coordonnée $i + d_h/2$")
    ax.set_title("RoPE : une paire tournée de $m\\,\\theta_i$ ($\\theta_i$ = 0,5 rad)", fontsize=10.5)
    save(fig, "positions.png")


# --- 6. post-LN and pre-LN blocks ------------------------------------------------------------------------------
def figure_blocks() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2))
    attn_fc, mlp_fc, ln_fc = "#cde2fb", "#fde0cf", "#d3f0e4"

    def plus(ax, x, y):
        ax.add_patch(plt.Circle((x, y), 0.17, fc="white", ec=INK, lw=1.0))
        ax.text(x, y, "+", ha="center", va="center", fontsize=12)

    # post-LN
    ax = axes[0]
    ys = {"x": 0.2, "mha": 1.3, "p1": 2.2, "ln1": 2.9, "mlp": 4.0, "p2": 4.9, "ln2": 5.6, "y": 6.5}
    ax.text(0, ys["x"], "$\\mathbf{x}$", ha="center", va="center", fontsize=12)
    box(ax, 0, ys["mha"], 1.9, 0.55, "attention multi-têtes", fc=attn_fc)
    plus(ax, 0, ys["p1"])
    box(ax, 0, ys["ln1"], 1.9, 0.5, "layer norm", fc=ln_fc)
    box(ax, 0, ys["mlp"], 1.9, 0.55, "MLP (GELU, $4d$)", fc=mlp_fc)
    plus(ax, 0, ys["p2"])
    box(ax, 0, ys["ln2"], 1.9, 0.5, "layer norm", fc=ln_fc)
    ax.text(0, ys["y"], "$\\mathbf{y}$", ha="center", va="center", fontsize=12)
    for a, b in [("x", "mha"), ("mha", "p1"), ("p1", "ln1"), ("ln1", "mlp"), ("mlp", "p2"), ("p2", "ln2"),
                 ("ln2", "y")]:
        arrow(ax, (0, ys[a] + 0.2), (0, ys[b] - 0.25))
    arrow(ax, (0, 0.55), (-0.17, ys["p1"] - 0.05), color=C[1], lw=1.4, rad=-0.6)
    arrow(ax, (0, ys["ln1"] + 0.3), (-0.17, ys["p2"] - 0.05), color=C[1], lw=1.4, rad=-0.6)
    ax.text(-1.35, 1.25, "résiduel", color=C[1], fontsize=9, rotation=90, va="center")
    ax.text(-1.35, 3.95, "résiduel", color=C[1], fontsize=9, rotation=90, va="center")
    ax.set_title("Post-LN (Vaswani et coll., 2017)\nla layer norm est sur le chemin principal", fontsize=10)
    # pre-LN
    ax = axes[1]
    ys = {"x": 0.2, "ln1": 1.0, "mha": 1.8, "p1": 2.7, "ln2": 3.5, "mlp": 4.3, "p2": 5.2, "lnf": 5.85, "y": 6.5}
    ax.text(0, ys["x"], "$\\mathbf{x}$", ha="center", va="center", fontsize=12)
    box(ax, 0.95, ys["ln1"], 1.7, 0.45, "layer norm", fc=ln_fc)
    box(ax, 0.95, ys["mha"], 1.7, 0.55, "attention multi-têtes", fc=attn_fc)
    plus(ax, 0, ys["p1"])
    box(ax, 0.95, ys["ln2"], 1.7, 0.45, "layer norm", fc=ln_fc)
    box(ax, 0.95, ys["mlp"], 1.7, 0.55, "MLP (GELU, $4d$)", fc=mlp_fc)
    plus(ax, 0, ys["p2"])
    ax.text(0, ys["y"], "$\\mathbf{y}$", ha="center", va="center", fontsize=12)
    ax.plot([0, 0], [ys["x"] + 0.2, ys["p1"] - 0.17], color=C[1], lw=2.2)
    ax.plot([0, 0], [ys["p1"] + 0.17, ys["p2"] - 0.17], color=C[1], lw=2.2)
    arrow(ax, (0, ys["p2"] + 0.17), (0, ys["y"] - 0.2), color=C[1], lw=2.2)
    arrow(ax, (0, 0.55), (0.95, ys["ln1"] - 0.22), rad=-0.3)
    arrow(ax, (0.95, ys["ln1"] + 0.22), (0.95, ys["mha"] - 0.27))
    arrow(ax, (0.95, ys["mha"] + 0.27), (0.17, ys["p1"]), rad=0.3)
    arrow(ax, (0, ys["p1"] + 0.3), (0.95, ys["ln2"] - 0.22), rad=-0.3)
    arrow(ax, (0.95, ys["ln2"] + 0.22), (0.95, ys["mlp"] - 0.27))
    arrow(ax, (0.95, ys["mlp"] + 0.27), (0.17, ys["p2"]), rad=0.3)
    ax.text(-0.3, 3.95, "chemin résiduel direct", color=C[1], fontsize=9, rotation=90, va="center", ha="right")
    ax.text(1.4, 6.35, "(une layer norm finale\naprès le dernier bloc)", fontsize=8.3, color=MUTED, ha="center")
    ax.set_title("Pré-LN (GPT-2 et suivants, ta transformer_block)\nla layer norm ouvre chaque branche",
                 fontsize=10)
    for ax in axes:
        ax.set_xlim(-1.6, 2.4)
        ax.set_ylim(-0.2, 6.9)
        ax.axis("off")
    save(fig, "bloc.png")


# --- 7. theoretical costs ------------------------------------------------------------------------------------
def figure_costs() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    T = 2.0 ** np.arange(7, 18)                       # 128 to 131 072 tokens
    mem = 32 * T**2 * 4                               # one sequence, 32 heads, float32 weights of one layer
    ax = axes[0]
    ax.loglog(T, mem / 2**30, marker="o", color=C[0])
    ticks = [2.0**-10, 2.0**-5, 1.0, 2.0**5, 2.0**10]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["1 Mio", "32 Mio", "1 Gio", "32 Gio", "1 Tio"])
    ax.minorticks_off()
    for gib, label in [(16, "16 Gio"), (80, "80 Gio (un grand GPU)")]:
        ax.axhline(gib, color=MUTED, lw=0.9, ls="--")
        ax.text(T[0], gib * 1.25, label, fontsize=8.5, color=MUTED)
    ax.set_xlabel("longueur $T$ (tokens)")
    ax.set_ylabel("mémoire")
    ax.set_xticks(T[::2])
    ax.set_xticklabels([f"{int(t):,}".replace(",", " ") for t in T[::2]], fontsize=8.5)
    ax.set_title("Mémoire des poids d'attention d'une couche\n(1 séquence, 32 têtes, float32) : pente 2",
                 fontsize=10.5)
    ax = axes[1]
    Ts = np.logspace(2, 6, 300)
    for d, col in zip((512, 2048, 8192), (C[2], C[0], C[6])):
        share = Ts / (Ts + 12 * d)
        ax.semilogx(Ts, 100 * share, color=col, label=f"$d$ = {d:,}".replace(",", " "))
        ax.plot([12 * d], [50], "o", color=col)
    ax.axhline(50, color=MUTED, lw=0.8, ls="--")
    ax.text(110, 53, "la moitié du calcul, à $T = 12\\,d$", fontsize=8.5, color=MUTED)
    ax.set_xlabel("longueur $T$ (tokens)")
    ax.set_ylabel("part de l'attention (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.set_title("Part de l'attention dans les FLOPs d'une couche causale\n$2Td \\,/\\, (24d^2 + 2Td)$",
                 fontsize=10.5)
    save(fig, "cout.png")


def main() -> int:
    wbplot.set_style()
    figure_bottleneck()
    figure_qkv()
    figure_masks()
    figure_heads()
    figure_positions()
    figure_blocks()
    figure_costs()
    print(f"✅ figures written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
