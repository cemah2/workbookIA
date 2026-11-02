#!/usr/bin/env python
"""Draw the figures of chapter B5 (used by Claude).

    python tools/chapters/figures_chB5.py        # draw the five figures (offline, a few seconds)

Writes small PNG files into chapitres/b5_diffusion/figures/:

- ``chaine.png``: the forward process on a 2-D spiral cloud, x_t = sqrt(ab) x0 + sqrt(1 - ab) eps, at six levels of
  alpha_bar (not at step numbers), with the direction of the learned reverse process;
- ``plannings.png``: alpha_bar_t and ln SNR(t) against t / T for the linear schedule of Ho et al. (T = 1000, and
  T = 200 with the same betas, to show that it depends on T) and the cosine schedule of Nichol & Dhariwal;
- ``unet.png``: the small U-Net of section 205.4 (28 -> 14 -> 7 -> 14 -> 28, skip connections by concatenation,
  step embedding added to every residual block); its parameter count was measured with PyTorch 2.11;
- ``ddpm_ddim.png``: reverse trajectories in 2-D with the *exact* noise predictor of a Gaussian mixture (no
  network): DDPM with 1000 stochastic steps, twice from the same starting points, against DDIM with 50
  deterministic steps;
- ``guidance.png``: the same mixture with class labels; unconditional samples, samples of one class, and samples of
  that class with classifier-free guidance w = 0.5 (exact conditional and unconditional predictors, DDIM 50 steps).

Nothing here is trained, downloaded or timed. The 2-D data are neither the ring of Gaussians of the notebook
(B5.18 to B5.20) nor MNIST; the guidance scale (w = 0.5) is none of those of B5.1 and B5.26; no figure shows a
noisy digit or marks a step number (B5.3, B5.13, B5.15); plannings.png has no grid and no tick at ln SNR = 0
or alpha_bar = 0.5, so that the crossings asked in B5.3 cannot be read off it to two decimals.
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

OUT = ROOT / "chapitres" / "b5_diffusion" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED
BLUE, ORANGE, GREEN, GREY = "#cde2fb", "#fde0cf", "#d5efd2", "#efeeea"

# --- data shared with 01_fiche.md --------------------------------------------------------------------------------
LEVELS = [1.0, 0.9, 0.5, 0.15, 0.02, 0.0]                  # alpha_bar of the panels of chaine.png
MIX_MEANS = np.array([[-1.6, -0.6], [1.4, -1.0], [0.2, 1.5]])   # 3 classes of the 2-D mixture
MIX_STD = 0.35
MIX_WEIGHTS = np.array([0.5, 0.3, 0.2])
GUIDANCE_W = 0.5


def french(x: float, digits: int = 2) -> str:
    """A number with a French decimal comma and a true minus sign."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fontsize=9.0, color=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)


def arrow(ax, start, end, color=INK, lw=1.0, rad=0.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


# --- schedules (the formulas of the docstrings of diffusion.py, indices 0 .. T - 1) --------------------------------
def linear_betas(T: int, start: float = 1e-4, end: float = 0.02) -> np.ndarray:
    return np.linspace(start, end, T)


def cosine_betas(T: int, s: float = 0.008, max_beta: float = 0.999) -> np.ndarray:
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return np.minimum(1 - ab[1:] / ab[:-1], max_beta)


def alpha_bars(betas: np.ndarray) -> np.ndarray:
    return np.cumprod(1 - betas)


# --- exact noise predictor of a Gaussian mixture -----------------------------------------------------------------
def mixture_eps(x: np.ndarray, ab: float, means=MIX_MEANS, weights=MIX_WEIGHTS, std=MIX_STD) -> np.ndarray:
    """E[eps | x_t = x] for x0 drawn from the mixture sum_k w_k N(mu_k, std^2 I), x_t = a x0 + b eps.

    Given the component k, x_t ~ N(a mu_k, (a^2 std^2 + b^2) I) and E[x0 | x_t, k] = mu_k + a std^2 / (a^2 std^2 +
    b^2) (x_t - a mu_k); the components are weighted by their posterior probability. Then eps = (x_t - a x0) / b.
    """
    a, b = np.sqrt(ab), np.sqrt(1 - ab)
    var = a**2 * std**2 + b**2
    d2 = ((x[:, None, :] - a * means[None, :, :]) ** 2).sum(-1)              # (N, K)
    logw = np.log(weights)[None, :] - d2 / (2 * var)
    post = np.exp(logw - logw.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    x0_k = means[None, :, :] + (a * std**2 / var) * (x[:, None, :] - a * means[None, :, :])
    x0 = (post[:, :, None] * x0_k).sum(1)
    return (x - a * x0) / b


def ddpm_path(x_T, abs_, betas, rng, eps_fn):
    """Ancestral sampling (posterior variance), keeping every state."""
    ab_prev = np.r_[1.0, abs_[:-1]]
    x, path = x_T.copy(), [x_T.copy()]
    for t in range(len(betas) - 1, -1, -1):
        eps = eps_fn(x, abs_[t])
        mean = (x - betas[t] / np.sqrt(1 - abs_[t]) * eps) / np.sqrt(1 - betas[t])
        if t > 0:
            var = betas[t] * (1 - ab_prev[t]) / (1 - abs_[t])
            x = mean + np.sqrt(var) * rng.standard_normal(x.shape)
        else:
            x = mean
        path.append(x.copy())
    return np.array(path)


def ddim_path(x_T, abs_, steps, eps_fn):
    """Deterministic DDIM (eta = 0) on the decreasing indices ``steps``, then down to the clean image (index -1)."""
    x, path = x_T.copy(), [x_T.copy()]
    for t, t_prev in zip(steps, list(steps[1:]) + [-1]):
        eps = eps_fn(x, abs_[t])
        x0 = (x - np.sqrt(1 - abs_[t]) * eps) / np.sqrt(abs_[t])
        ab_prev = 1.0 if t_prev < 0 else abs_[t_prev]
        x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * eps
        path.append(x.copy())
    return np.array(path)


def spiral(n: int, rng) -> np.ndarray:
    u = np.sqrt(rng.uniform(0.04, 1, n)) * 3.2 * np.pi
    pts = np.c_[u * np.cos(u), u * np.sin(u)] / 6.0
    pts += 0.06 * rng.standard_normal(pts.shape)
    return (pts - pts.mean(0)) / pts.std()


# --- 1. the chain -------------------------------------------------------------------------------------------------
def figure_chain() -> None:
    rng = np.random.default_rng(3)
    x0 = spiral(1500, rng)
    eps = rng.standard_normal(x0.shape)
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(13.0, 3.3))
    for ax, ab in zip(axes, LEVELS):
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        ax.scatter(xt[:, 0], xt[:, 1], s=2.5, color=C[0], alpha=0.55, lw=0)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        label = "données $\\mathbf{x}_0$" if ab == 1 else ("bruit pur" if ab == 0 else "")
        ax.set_title(f"$\\bar\\alpha$ = {french(ab, 2)}\n{label}", fontsize=10)
    fig.subplots_adjust(wspace=0.08, top=0.78, bottom=0.2)
    left, right = axes[0].get_position(), axes[-1].get_position()
    y_top, y_bot = 0.95, 0.07
    fig.add_artist(FancyArrowPatch((left.x0, y_top), (right.x1, y_top), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=12, color=C[1], lw=1.4))
    fig.text(0.5, y_top + 0.02, "processus direct $q$ (fixé, sans paramètres) : on ajoute un peu de bruit à chaque pas",
             ha="center", va="bottom", fontsize=10.2, color=C[1])
    fig.add_artist(FancyArrowPatch((right.x1, y_bot), (left.x0, y_bot), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=12, color=C[2], lw=1.4))
    fig.text(0.5, y_bot - 0.02, "processus inverse $p_\\theta$ (appris) : un réseau retire un peu de bruit à chaque pas",
             ha="center", va="top", fontsize=10.2, color=C[2])
    save(fig, "chaine.png")


# --- 2. the schedules ---------------------------------------------------------------------------------------------
def figure_schedules() -> None:
    curves = [("linéaire, $T$ = 1000", linear_betas(1000), C[0], "-"),
              ("cosinus, $T$ = 1000", cosine_betas(1000), C[1], "-"),
              ("linéaire, mêmes $\\beta$, $T$ = 200", linear_betas(200), C[0], "--")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    for name, betas, color, ls in curves:
        ab = alpha_bars(betas)
        T = len(betas)
        x = np.arange(1, T + 1) / T                          # paper step t = index + 1
        axes[0].plot(x, ab, color=color, ls=ls, lw=1.6, label=name)
        snr = ab / (1 - ab)
        keep = snr > 0
        axes[1].plot(x[keep][:-1], np.log(snr[keep][:-1]), color=color, ls=ls, lw=1.6, label=name)
    axes[0].set_ylabel("$\\bar\\alpha_t$ : part de variance qui vient de l'image")
    axes[0].set_title("$\\bar\\alpha_t$ des deux plannings", fontsize=10.5)
    axes[1].set_ylabel("$\\ln \\mathrm{SNR}(t) = \\ln \\frac{\\bar\\alpha_t}{1 - \\bar\\alpha_t}$")
    axes[1].set_title("Rapport signal sur bruit, en échelle logarithmique", fontsize=10.5)
    axes[1].set_ylim(-8, 10)
    # No grid, and no tick at ln SNR = 0 or alpha_bar = 0.5: the crossings of B5.3 must be computed, not read.
    axes[0].set_yticks([0.0, 0.3, 0.6, 0.9])
    axes[1].set_yticks([-7, -4, -1, 2, 5, 8])
    for ax in axes:
        ax.set_xlabel("$t / T$ (avancement de la chaîne)")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.legend(fontsize=8.5, frameon=False)
        ax.set_xlim(0, 1)
    fig.tight_layout()
    save(fig, "plannings.png")


# --- 3. the U-Net -------------------------------------------------------------------------------------------------
def figure_unet() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    # (x, y, text, colour) of the main path; y goes down with the resolution
    enc = [(0.4, 3.0, "$\\mathbf{x}_t$\n1 × 28 × 28", "#ffffff"),
           (2.2, 3.0, "conv 3×3\n+ bloc résiduel\n32 × 28 × 28", BLUE),
           (4.0, 1.9, "conv stride 2\n+ bloc résiduel\n64 × 14 × 14", BLUE),
           (6.25, 0.8, "conv stride 2\n+ bloc du milieu\n128 × 7 × 7", GREY)]
    dec = [(8.5, 1.9, "conv. transposée,\n‖, bloc résiduel\n64 × 14 × 14", ORANGE),
           (10.3, 3.0, "conv. transposée,\n‖, bloc résiduel\n32 × 28 × 28", ORANGE),
           (12.1, 3.0, "conv 3×3\n$\\boldsymbol{\\varepsilon}_\\theta$ : 1 × 28 × 28", "#ffffff")]
    nodes = enc + dec
    for x, y, text, fc in nodes:
        box(ax, x, y, 1.6, 0.86, text, fc=fc, fontsize=8.4)
    for (x1, y1, *_), (x2, y2, *_) in zip(nodes[:-1], nodes[1:]):
        arrow(ax, (x1 + 0.81, y1), (x2 - 0.81, y2), lw=1.1)
    # skip connections (concatenation), drawn above
    for (xa, ya), (xb, yb) in [((2.2, 3.0), (10.3, 3.0)), ((4.0, 1.9), (8.5, 1.9))]:
        arrow(ax, (xa, ya + 0.42), (xb, yb + 0.42), color=C[0], lw=1.4, rad=-0.18)
    ax.text(6.25, 4.55, "connexions de saut : la carte d'activation de l'encodeur est recopiée et concaténée "
            "(‖) à celle du décodeur, à la même résolution", ha="center", fontsize=9.0, color=C[0])
    # step embedding
    box(ax, 6.25, -0.75, 3.6, 0.7, "pas $t$ → encodage sinusoïdal (128) → MLP → $\\mathbf{e}_t$\n"
        "(ajouté à chaque bloc résiduel : un biais par canal)", fc=GREEN, fontsize=8.6)
    for x, y in [(2.2, 3.0), (4.0, 1.9), (6.25, 0.8), (8.5, 1.9), (10.3, 3.0)]:
        arrow(ax, (6.25, -0.38), (x, y - 0.42), color=C[5], lw=0.9, ls=(0, (3, 2)), rad=0.0)
    ax.text(0.0, -0.75, "≈ 0,74 M paramètres\n≈ 0,2 GFLOP par image", fontsize=8.6, color=MUTED, va="center")
    ax.set_xlim(-0.55, 13.0)
    ax.set_ylim(-1.3, 4.9)
    ax.axis("off")
    save(fig, "unet.png")


# --- 4. DDPM against DDIM -----------------------------------------------------------------------------------------
def _mixture_background(ax) -> None:
    th = np.linspace(0, 2 * np.pi, 100)
    for k, m in enumerate(MIX_MEANS):
        for r in (1, 2):
            ax.plot(m[0] + r * MIX_STD * np.cos(th), m[1] + r * MIX_STD * np.sin(th), color=MUTED, lw=0.6,
                    alpha=0.7)


def figure_ddpm_ddim() -> None:
    betas = cosine_betas(1000)
    abs_ = alpha_bars(betas)
    starts = np.array([[-0.3, 0.2], [0.9, -0.4], [-0.9, -1.4]])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    ax = axes[0]
    _mixture_background(ax)
    for run, (rng_seed, ls) in enumerate([(1, "-"), (2, "--")]):
        path = ddpm_path(starts, abs_, betas, np.random.default_rng(rng_seed), mixture_eps)
        for j in range(len(starts)):
            ax.plot(path[::5, j, 0], path[::5, j, 1], color=C[j % 8], lw=0.7, ls=ls, alpha=0.8)
            ax.plot(*path[-1, j], marker="o", ms=8, color=C[j % 8], mec="black", mew=0.9, zorder=5)
    ax.set_title("DDPM : 1 000 pas, un bruit $\\mathbf{z}$ neuf à chaque pas\n(deux tirages depuis les mêmes "
                 "départs : trait plein, tirets)", fontsize=10)
    ax = axes[1]
    _mixture_background(ax)
    steps = np.linspace(999, 0, 50).round().astype(int)
    path = ddim_path(starts, abs_, steps, mixture_eps)
    for j in range(len(starts)):
        ax.plot(path[:, j, 0], path[:, j, 1], color=C[j % 8], lw=1.2, marker=".", ms=2.5)
        ax.plot(*path[-1, j], marker="o", ms=8, color=C[j % 8], mec="black", mew=0.9, zorder=5)
    ax.set_title("DDIM ($\\eta$ = 0) : 50 pas, aucun tirage\n(même départ → même point d'arrivée)", fontsize=10)
    for ax in axes:
        ax.plot(starts[:, 0], starts[:, 1], "kx", ms=6, mew=1.2)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal")
        ax.grid(color=wbplot.GRID, lw=0.5)
    fig.text(0.5, 0.01, "croix : départs $\\mathbf{x}_T$ ; ronds cerclés de noir : arrivées ; cercles gris : les trois "
             "gaussiennes des données (1 et 2 écarts-types) ; prédicteur de bruit exact, sans réseau",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "ddpm_ddim.png")


# --- 5. classifier-free guidance ------------------------------------------------------------------------------------
def figure_guidance() -> None:
    abs_ = alpha_bars(cosine_betas(1000))
    steps = np.linspace(999, 0, 50).round().astype(int)
    k = 2                                                         # the class we ask for
    cond = lambda x, ab: mixture_eps(x, ab, MIX_MEANS[k:k + 1], np.array([1.0]))       # noqa: E731
    uncond = lambda x, ab: mixture_eps(x, ab)                                             # noqa: E731
    guided = lambda x, ab: (1 + GUIDANCE_W) * cond(x, ab) - GUIDANCE_W * uncond(x, ab)  # noqa: E731
    x_T = np.random.default_rng(5).standard_normal((600, 2))
    panels = [("sans condition : $\\boldsymbol{\\varepsilon}_\\theta(\\mathbf{x}_t, \\varnothing)$", uncond, MUTED),
              (f"classe {k}, $w$ = 0", cond, C[0]),
              (f"classe {k}, guidance $w$ = {french(GUIDANCE_W, 1)}", guided, C[1])]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
    for ax, (title, fn, color) in zip(axes, panels):
        _mixture_background(ax)
        x0 = ddim_path(x_T, abs_, steps, fn)[-1]
        ax.scatter(x0[:, 0], x0[:, 1], s=4, color=color, alpha=0.6, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-2.6, 3.0)
        ax.set_aspect("equal")
        ax.grid(color=wbplot.GRID, lw=0.5)
        if fn is not uncond:
            sd = x0.std(0).mean()
            ax.text(0.03, 0.03, f"écart-type moyen : {french(sd, 2)}", transform=ax.transAxes, fontsize=8.6,
                    color=MUTED)
    fig.text(0.5, 0.0, "600 départs identiques dans les trois panneaux, DDIM 50 pas ; prédicteurs de bruit exacts "
             "du mélange (sans condition) et de la classe (avec condition)", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "guidance.png")


def main() -> None:
    figure_chain()
    figure_schedules()
    figure_unet()
    figure_ddpm_ddim()
    figure_guidance()
    print("figures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
