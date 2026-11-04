#!/usr/bin/env python
"""Draw the figures of chapter B7 (used by Claude).

    python tools/chapters/figures_chB7.py        # draw the five figures (offline, a few seconds)

Writes small PNG files into chapitres/b7_production/figures/:

- ``cycle_de_vie.png``: the life of a model in production: data, training (configuration, seed, run), artifact
  (weights, preprocessing, classes, versions, fingerprint), serving (API, batches, latency), monitoring (drift,
  quality, logs); two return arrows (retrain, roll back) and the tests and CI band under the whole loop;
- ``integration_continue.png``: the steps of a continuous integration job on every push or pull request, from the
  checkout to the smoke training, and what a red step means;
- ``latences.png``: a histogram of 2 000 SYNTHETIC latencies (a log-normal body around 8 ms plus 3 % of slow
  requests, three times longer; seed 7), with the mean, p50, p95 and p99 marked: the mean and the median differ,
  and the slowest requests decide p95 and p99. These are not measurements of any model of the workbook;
- ``psi.png``: the PSI example of section 207.10 (five bins, reference shares 0.2 each, production shares 0.10,
  0.15, 0.20, 0.25, 0.30): the two histograms side by side, then the contribution of each bin to the PSI;
- ``ks.png``: the Kolmogorov-Smirnov example of section 207.10 (two samples of five brightness values): the two
  empirical cumulative distribution functions and the largest vertical gap D = 0.4.

Nothing here is downloaded or trained. None of these figures uses CIFAR-10, the MP5 model or the numbers of the
paper exercises (B7.1: sizes; B7.2: five seeds; B7.3: twenty latencies; B7.4: a PSI on four bins).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "b7_production" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED

# --- data shared with 01_fiche.md --------------------------------------------------------------------------------
PSI_REF = np.array([0.20, 0.20, 0.20, 0.20, 0.20])          # reference shares (quintiles of the reference)
PSI_PROD = np.array([0.10, 0.15, 0.20, 0.25, 0.30])         # production shares
KS_REF = np.array([0.31, 0.42, 0.47, 0.52, 0.58])           # mean brightness of five reference images
KS_PROD = np.array([0.22, 0.29, 0.35, 0.44, 0.61])          # and of five production images


def french(x: float, digits: int = 2) -> str:
    """A number with a French decimal comma and a true minus sign."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=INK, lw=1.0))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.52, "\n".join(lines), ha="center", va="top", fontsize=8.5, color=INK,
            linespacing=1.35)


def arrow(ax, p, q, text="", rad=0.0, color=INK, dy=0.12, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=8.5,
                color=color, style="italic", bbox=dict(fc="white", ec="none", pad=1.5))


# --- 207.1 and 207.10 --------------------------------------------------------------------------------------------
def figure_lifecycle() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-0.6, 4.9)
    ax.axis("off")
    w, h, y = 2.15, 2.2, 1.35
    xs = [0.15, 2.65, 5.15, 7.65, 10.15]
    specs = [("Données", ["source, licence", "révision, empreinte", "découpage figé"], C[5]),
             ("Entraînement", ["configuration YAML", "graine, versions", "un run = une ligne", "du journal"], C[0]),
             ("Artefact", ["poids (safetensors)", "prétraitement", "classes dans l'ordre", "versions, empreinte"],
              C[1]),
             ("Service", ["API /health, /predict", "validation des entrées", "batches ou en ligne",
                          "latence p50, p95"], C[2]),
             ("Surveillance", ["dérive des entrées", "qualité quand les", "labels arrivent",
                               "journaux, alertes"], C[3])]
    light = [matplotlib.colors.to_rgba(c, 0.25) for _, _, c in specs]
    for x, (title, lines, _), col in zip(xs, specs, light):
        box(ax, x, y, w, h, title, lines, col)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w + 0.03, y + h / 2), (b - 0.03, y + h / 2))
    arrow(ax, (xs[4] + w / 2, y + h + 0.02), (xs[1] + w / 2, y + h + 0.02), "dérive confirmée : réentraîner",
          rad=0.22, color=C[3], dy=0.95)
    arrow(ax, (xs[4] + 0.35, y - 0.03), (xs[2] + w - 0.35, y - 0.03), "incident : retour arrière à l'artefact précédent",
          rad=-0.35, color=wbplot.BAD, dy=-0.95)
    ax.add_patch(FancyBboxPatch((0.15, -0.55), 12.15, 0.42, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=matplotlib.colors.to_rgba(MUTED, 0.12), ec=MUTED, lw=0.8))
    ax.text(6.22, -0.34, "tests et intégration continue à chaque commit  ·  tout est versionné : code, "
            "configuration, verrou des dépendances, artefacts", ha="center", va="center", fontsize=8.5, color=INK)
    save(fig, "cycle_de_vie.png")


# --- 207.6 -------------------------------------------------------------------------------------------------------
def figure_ci() -> None:
    steps = [("push ou\npull request", "déclencheur"), ("checkout", "le dépôt\nau commit testé"),
             ("setup-python", "Python 3.13,\ncache pip"), ("installer", "torch CPU, verrou,\npip install -e ."),
             ("lint\n(option)", "ruff : style,\nimports"), ("pytest", "formes, données,\noverfitting"),
             ("entraînement\nfumée", "1 epoch,\npeu d'images")]
    fig, ax = plt.subplots(figsize=(12.5, 2.9))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-0.9, 2.0)
    ax.axis("off")
    w, gap = 1.55, 0.25
    for i, (name, note) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        col = matplotlib.colors.to_rgba(C[0] if i else MUTED, 0.22)
        ax.add_patch(FancyBboxPatch((x, 0.6), w, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec=INK,
                                    lw=1.0))
        ax.text(x + w / 2, 1.1, name, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
        ax.text(x + w / 2, 0.35, note, ha="center", va="top", fontsize=8, color=MUTED)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.02, 1.1), (x + w + gap - 0.02, 1.1))
    ax.text(6.25, -0.72, "une étape échoue (code de sortie ≠ 0) : le job s'arrête et le commit est marqué ✗ ; "
            "tout passe : ✓, et la branche peut être fusionnée", ha="center", fontsize=8.5, color=INK)
    save(fig, "integration_continue.png")


# --- 207.7 -------------------------------------------------------------------------------------------------------
def figure_latency() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    t = np.exp(rng.normal(np.log(8.0), 0.18, n))
    slow = rng.random(n) < 0.03
    t[slow] *= 3.0
    mean = t.mean()
    p50, p95, p99 = np.percentile(t, [50, 95, 99])
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.hist(t, bins=np.arange(4, 36, 0.5), color=C[0], alpha=0.75)
    ax.set_xlim(3, 36)
    top = 1.3 * ax.get_ylim()[1]             # headroom: the labels sit above the tallest bars
    ax.set_ylim(0, top)
    for v, lab, col, ls, (dx, yf, ha) in [(p50, "p50", INK, "-", (-0.3, 0.93, "right")),
                                          (mean, "moyenne", MUTED, ":", (0.3, 0.82, "left")),
                                          (p95, "p95", C[1], "--", (0.3, 0.68, "left")),
                                          (p99, "p99", wbplot.BAD, "--", (0.3, 0.93, "left"))]:
        ax.axvline(v, color=col, ls=ls, lw=1.6)
        ax.text(v + dx, top * yf, f"{lab} = {french(v, 1)} ms", color=col, fontsize=9, ha=ha,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_xlabel("latence d'une requête (ms) — données synthétiques")
    ax.set_ylabel("nombre de requêtes")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("2 000 requêtes : 97 % autour de 8 ms, 3 % trois fois plus lentes", fontsize=10)
    save(fig, "latences.png")
    return mean, p50, p95, p99


# --- 207.10 ------------------------------------------------------------------------------------------------------
def figure_psi() -> None:
    contrib = (PSI_PROD - PSI_REF) * np.log(PSI_PROD / PSI_REF)
    k = np.arange(1, 6)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.bar(k - 0.19, PSI_REF, 0.38, color=MUTED, alpha=0.6, label="référence $q_k$")
    a1.bar(k + 0.19, PSI_PROD, 0.38, color=C[0], label="production $p_k$")
    a1.set_xticks(k, [f"intervalle {i}" for i in k], fontsize=8.5)
    a1.set_ylabel("part des exemples")
    a1.set_ylim(0, 0.36)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    a1.set_title("les intervalles sont les quintiles de la référence", fontsize=10)
    a2.bar(k, contrib, 0.6, color=C[1])
    for i, c in zip(k, contrib):
        a2.text(i, c + 0.0015, french(c, 4) if c > 0 else "0", ha="center", fontsize=8.5, color=INK)
    a2.set_xticks(k, [str(i) for i in k])
    a2.set_xlabel("intervalle")
    a2.set_ylabel("$(p_k - q_k)\\ln(p_k / q_k)$")
    a2.set_ylim(0, 0.08)
    a2.set_title(f"PSI = somme = {french(contrib.sum(), 3)}", fontsize=10)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "psi.png")


def figure_ks() -> None:
    xs = np.linspace(0.15, 0.68, 600)

    def ecdf(sample, x):
        return np.searchsorted(np.sort(sample), x, side="right") / len(sample)

    fr, fp = ecdf(KS_REF, xs), ecdf(KS_PROD, xs)
    i = np.argmax(np.abs(fr - fp))
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.step(xs, fr, where="post", color=MUTED, lw=2, label="référence $F_{\\mathrm{réf}}$")
    ax.step(xs, fp, where="post", color=C[0], lw=2, label="production $F_{\\mathrm{prod}}$")
    x0 = 0.30
    ax.annotate("", xy=(x0, ecdf(KS_PROD, x0)), xytext=(x0, ecdf(KS_REF, x0)),
                arrowprops=dict(arrowstyle="<->", color=wbplot.BAD, lw=1.6))
    ax.text(x0 + 0.012, 0.1, "D = 0,4", color=wbplot.BAD, fontsize=10)
    ax.set_xlabel("luminosité moyenne d'une image")
    ax.set_ylabel("part des images ≤ x")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("deux échantillons de cinq images : le plus grand écart vertical", fontsize=10)
    save(fig, "ks.png")
    return float(np.abs(fr - fp)[i])


def main() -> None:
    figure_lifecycle()
    figure_ci()
    stats = figure_latency()
    figure_psi()
    d = figure_ks()
    print("latency mean, p50, p95, p99:", [round(float(s), 2) for s in stats], "| KS D:", d)
    print("figures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
