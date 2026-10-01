#!/usr/bin/env python
"""Draw the figures of the course sheet of chapter 6 (used by Claude).

    python tools/chapters/figures_ch06.py

Writes small PNG files into chapitres/ch06_information/figures/. Every figure is computed
(never copied from the book): the surprise -log2(p) and the entropy of a coin, the word
WATSON in Morse and in a fixed-length code, a Huffman tree for a toy alphabet, the terms
of a cross-entropy and of the two KL divergences for two small distributions, and the
bits per letter that context saves on the French text of Verne. None of them uses the
numbers of an exercise (the exercises use Holmes, or Verne coded with the letters of
Holmes).
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import datasets  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch06_information" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
LETTERS = "abcdefghijklmnopqrstuvwxyz"
MORSE = dict(a=".-", b="-...", c="-.-.", d="-..", e=".", f="..-.", g="--.", h="....", i="..", j=".---", k="-.-",
             l=".-..", m="--", n="-.", o="---", p=".--.", q="--.-", r=".-.", s="...", t="-", u="..-", v="...-",
             w=".--", x="-..-", y="-.--", z="--..")


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma, a thin space for thousands and a true minus sign."""
    return f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")


def fr(value: float, decimals: int) -> str:
    """A number written the French way, for titles and legends."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:  # decimal commas on every numeric linear axis (not on named ticks)
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    path = OUT / name
    print(f"  {name} ({path.stat().st_size // 1000} ko)")


def entropy_bits(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
def fig_surprise() -> None:
    """Left: the surprise -log2(p). Right: the entropy of a coin as a function of p."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))
    p = np.linspace(0.005, 1, 400)
    ax1.plot(p, -np.log2(p), color=C[0], lw=2)
    for prob, label in [(1, "1"), (0.5, "1/2"), (0.25, "1/4"), (1 / 16, "1/16")]:
        bits = abs(float(np.log2(prob)))                # abs: p = 1 gives 0, not -0
        ax1.plot(prob, bits, "o", color=C[1])
        ax1.annotate(f"p = {label} : {fr(bits, 0)} bit{'s' if bits > 1 else ''}", (prob, bits),
                     xytext=(8, 6), textcoords="offset points", fontsize=9)
    ax1.set(xlabel="probabilité p de l'événement", ylabel="surprise −log₂ p (bits)", ylim=(-0.3, 7.9),
            title="Plus c'est rare, plus ça informe")
    q = np.linspace(0.0005, 0.9995, 400)
    ax2.plot(q, -(q * np.log2(q) + (1 - q) * np.log2(1 - q)), color=C[0], lw=2)
    ax2.plot(0.5, 1, "o", color=C[1])
    ax2.annotate("maximum : 1 bit (pièce équilibrée)", (0.5, 1), xytext=(14, 4), textcoords="offset points",
                 fontsize=9)
    h25 = entropy_bits([0.25, 0.75])
    ax2.plot(0.25, h25, "o", color=C[2])
    ax2.annotate(f"« rhubarb » 0,25, « sassafras » 0,75 : {fr(h25, 2)} bit", (0.25, h25),
                 xytext=(-10, -34), textcoords="offset points", fontsize=9)
    ax2.set(xlabel="probabilité p de pile", ylabel="entropie de la pièce (bits)", ylim=(0, 1.12),
            title="L'incertitude moyenne d'une pièce")
    fig.tight_layout()
    save(fig, "surprise.png")


def _cell(ax, x, y, text, face, edge="0.35", color="black", size=12, hatch=None) -> None:
    """One square cell of a tape, with a symbol written in it."""
    ax.add_patch(Rectangle((x, y - 0.4), 0.8, 0.8, facecolor=face, edgecolor=edge, lw=1, hatch=hatch))
    if text:
        ax.text(x + 0.4, y, text, ha="center", va="center", fontsize=size, color=color, weight="bold")


def fig_morse() -> None:
    """The word WATSON on two tapes: a 5-digit fixed-length code, and Morse with the silences it needs."""
    word = "watson"
    fixed = {c: format(i, "05b") for i, c in enumerate(LETTERS)}       # a = 00000, b = 00001, ...
    fig, ax = plt.subplots(figsize=(12, 4.4))
    y_fixed, y_morse = 2.6, 0.6
    x = 0.0
    for c in word:                                                    # fixed code: 5 binary digits per letter
        start = x
        for digit in fixed[c]:
            _cell(ax, x, y_fixed, digit, face="#e8eef7")
            x += 0.85
        ax.text((start + x - 0.05) / 2, y_fixed + 0.65, c.upper(), ha="center", fontsize=11)
        x += 0.45                                                     # a gap for the eye only
    x = 0.0
    for k, c in enumerate(word):                                      # Morse: dots, dashes, then a silence
        start = x
        for symbol in MORSE[c]:
            _cell(ax, x, y_morse, "·" if symbol == "." else "−", face="#fde9df", size=16)
            x += 0.85
        ax.text((start + x - 0.05) / 2, y_morse - 0.75, c.upper(), ha="center", fontsize=11)
        if k < len(word) - 1:
            _cell(ax, x, y_morse, "", face="white", edge="0.55", hatch="///")
            x += 0.85
    n_fixed, n_morse = 5 * len(word), sum(len(MORSE[c]) for c in word)
    ax.text(-0.5, y_fixed, f"code fixe\n{n_fixed} symboles", ha="right", va="center", fontsize=10)
    ax.text(-0.5, y_morse, f"Morse\n{n_morse} points et traits\n+ {len(word) - 1} silences (hachurés)", ha="right",
            va="center", fontsize=10)
    ax.text(0, y_fixed - 0.75, "les écarts entre les groupes du code fixe ne servent qu'à la lecture : on lit les "
            "symboles 5 par 5", fontsize=8.5, color="0.35")
    ax.set_xlim(-0.3, 6 * 5 * 0.85 + 6 * 0.45)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")
    ax.set_title("WATSON : un code fixe de 5 symboles par lettre, ou le Morse et ses silences")
    save(fig, "morse_fixe.png")


def _huffman_tree(symbols, probs):
    """Huffman merges, kept as a tree: leaves are symbols, nodes are (left, right, weight)."""
    heap = [(p, i, s) for i, (s, p) in enumerate(zip(symbols, probs))]
    heapq.heapify(heap)
    order = len(heap)
    while len(heap) > 1:
        w0, _, t0 = heapq.heappop(heap)
        w1, _, t1 = heapq.heappop(heap)
        heapq.heappush(heap, (w0 + w1, order, (t0, t1, w0 + w1)))
        order += 1
    return heap[0][2]


def fig_huffman() -> None:
    """A Huffman tree for a toy alphabet of six symbols: weights, bits on the edges, codewords at the leaves."""
    symbols = ["A", "B", "C", "D", "E", "F"]
    probs = [0.32, 0.26, 0.17, 0.12, 0.08, 0.05]
    weight = dict(zip(symbols, probs))
    tree = _huffman_tree(symbols, probs)
    positions, codes = {}, {}
    leaves = []

    spacing = 1.7                                       # room for two lines of text under each leaf

    def walk(node, depth, prefix):
        if isinstance(node, str):
            leaves.append(node)
            positions[id(node), node] = (spacing * (len(leaves) - 1), -depth)
            codes[node] = prefix
            return spacing * (len(leaves) - 1)
        left = walk(node[0], depth + 1, prefix + "0")
        right = walk(node[1], depth + 1, prefix + "1")
        positions[id(node), None] = ((left + right) / 2, -depth)
        node_x = (left + right) / 2
        return node_x

    def x_of(node):
        if isinstance(node, str):
            return positions[id(node), node][0]
        return positions[id(node), None][0]

    walk(tree, 0, "")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    def draw(node, depth):
        x, y = x_of(node), -depth
        if isinstance(node, str):
            ax.add_patch(Circle((x, y), 0.3, color=C[0]))
            ax.text(x, y, node, ha="center", va="center", color="white", fontsize=11, weight="bold")
            ax.text(x, y - 0.55, f"p = {fr(weight[node], 2)}\n{codes[node]}", ha="center", va="top", fontsize=9)
            return
        ax.add_patch(Circle((x, y), 0.3, color="0.85"))
        ax.text(x, y, fr(node[2], 2), ha="center", va="center", fontsize=8)
        for child, bit in ((node[0], "0"), (node[1], "1")):
            cx, cy = x_of(child), -(depth + 1)
            ax.plot([x, cx], [y - 0.3, cy + 0.3], color="0.4", lw=1)
            ax.text((x + cx) / 2 + (-0.12 if bit == "0" else 0.12), (y + cy) / 2, bit, color=C[1],
                    ha="center", va="center", fontsize=10, weight="bold")
            draw(child, depth + 1)

    draw(tree, 0)
    mean = sum(weight[s] * len(codes[s]) for s in symbols)
    ax.set_xlim(-0.9, spacing * (len(symbols) - 1) + 0.9)
    ax.set_ylim(-max(len(c) for c in codes.values()) - 1.4, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Code de Huffman de six symboles\nlongueur moyenne {fr(mean, 2)} bits, "
                 f"entropie {fr(entropy_bits(probs), 2)} bits, code fixe 3 bits", fontsize=11)
    save(fig, "huffman_arbre.png")


def fig_cross_entropy() -> None:
    """Two distributions on four outcomes: the terms of KL(p || q) and of KL(q || p)."""
    outcomes = ["sec", "bruine", "averse", "orage"]
    p = np.array([0.8, 0.15, 0.04, 0.01])       # the weather of a dry town (the data)
    q = np.array([0.3, 0.3, 0.25, 0.15])        # the weather of a rainy town (the code)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.7))
    x = np.arange(len(outcomes))
    ax1.bar(x - 0.2, p, width=0.4, color=C[0], label="p : la ville sèche (les données)")
    ax1.bar(x + 0.2, q, width=0.4, color=C[1], label="q : la ville pluvieuse (le code)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(outcomes)
    ax1.set(ylabel="probabilité", title="Deux distributions du temps qu'il fait")
    ax1.legend(fontsize=8)
    forward = p * np.log2(p / q)
    backward = q * np.log2(q / p)
    ax2.bar(x - 0.2, forward, width=0.4, color=C[0],
            label=f"KL(p ‖ q) = {fr(forward.sum(), 2)} bit : p envoyé avec le code de q")
    ax2.bar(x + 0.2, backward, width=0.4, color=C[1],
            label=f"KL(q ‖ p) = {fr(backward.sum(), 2)} bit : q envoyé avec le code de p")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcomes)
    ax2.set(ylabel="contribution de chaque issue (bits)", title="Chaque issue pèse selon la PREMIÈRE distribution")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "cross_entropie.png")


def fig_context() -> None:
    """Bits per letter that 1, 2 or 3 letters of context save on the second half of Verne (models learned on the first)."""
    text = "".join(ch for ch in datasets.load_verne().lower() if "a" <= ch <= "z")
    index = np.frombuffer(text.encode("ascii"), dtype=np.uint8) - ord("a")
    half = len(index) // 2
    train, test = index[:half], index[half:]

    def mean_surprise(n: int) -> float:
        """Laplace-smoothed n-gram model (n = 1: no context), judged on the test half."""
        counts = np.ones((26,) * n)
        np.add.at(counts, tuple(train[k:len(train) - n + 1 + k] for k in range(n)), 1)
        probs = counts / counts.sum(axis=-1, keepdims=True)
        return float(np.mean(-np.log2(probs[tuple(test[k:len(test) - n + 1 + k] for k in range(n))])))

    base = mean_surprise(1)
    gains = {f"{n - 1} lettre{'s' if n > 2 else ''} de contexte": base - mean_surprise(n) for n in (2, 3, 4)}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bars = ax.bar(range(len(gains)), list(gains.values()), color=C[0], width=0.55)
    for bar, value in zip(bars, gains.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, fr(value, 2), ha="center", fontsize=9)
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(list(gains), fontsize=9)
    ax.set(ylabel="bits gagnés par lettre", ylim=(0, 1.35),
           title="Verne : ce que le contexte fait gagner, par rapport à un modèle qui l'ignore")
    save(fig, "contexte.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    fig_surprise()
    fig_morse()
    fig_huffman()
    fig_cross_entropy()
    fig_context()
    return 0


if __name__ == "__main__":
    sys.exit(main())
