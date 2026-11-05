#!/usr/bin/env python
"""Draw the figures of chapter B8 (used by Claude).

    python tools/chapters/figures_chB8.py        # draw the seven figures (offline, a few seconds)

Writes small PNG files into chapitres/b8_rl_moderne/figures/:

- ``boucle_dqn.png``: the DQN loop: the environment, the epsilon-greedy action of the online network, the replay
  buffer, a random mini-batch, the target computed by the frozen target network (no gradient), the Huber loss and
  Adam, and the periodic copy of the online weights into the target network;
- ``surestimation.png``: the maximum of m noisy estimates of m actions that are all worth 0 (noise N(0, 1),
  200 000 draws per m, seed 0): the max of one set of estimates is biased upwards (the exact expectation of the max
  of m standard normals is drawn too), the double estimator (argmax on one set, value read on an independent set)
  is not. Nothing is trained;
- ``variance_baseline.png``: single-sample policy-gradient estimates for a two-armed bandit with a uniform softmax
  policy (logits 0, 0) and Gaussian rewards N(10, 1) and N(11, 1): the gradient with respect to the logit of
  arm 1, without baseline and with the baseline b = 10.5 (100 000 draws, seed 1). Same mean (0.25), very different
  spreads. This toy is not CartPole and says nothing about which REINFORCE variant learns faster there;
- ``gae_lambda.png``: the weight (gamma * lambda)**l of the TD error delta_{t+l} in the GAE advantage, for
  gamma = 0.99 and five values of lambda, over 60 steps;
- ``clip_ppo.png``: one term of L^CLIP as a function of the ratio rho, for A = +1 and A = -1 (eps = 0.2), with the
  zones where the clipped term is the active minimum (zero gradient) shaded; the start of each update is rho = 1;
- ``rlhf_dpo.png``: three pipelines side by side as rows: RLHF (preferences -> Bradley-Terry reward model -> PPO
  with a KL penalty towards the reference), DPO (preferences + frozen reference -> one supervised loss) and GRPO
  (a group of answers -> verifiable rewards -> normalised within the group -> clipped update, no critic);
- ``autoreset.png``: a timeline of two sub-environments of ``gym.make_vec`` in the default next-step autoreset mode:
  after an episode ends, the next ``step`` only resets (reward 0, action ignored), and that transition must not be
  learned from.

Nothing here is downloaded or trained. None of these figures uses the numbers of the paper exercises (B8.1 to B8.6:
targets, Huber, softmax gradient, GAE on four steps, four clip cases, a preference pair) nor the results of the
notebook experiments (B8.12 to B8.28).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy import integrate, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "b8_rl_moderne" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED

# --- data shared with 01_fiche.md --------------------------------------------------------------------------------
BANDIT_MEANS = (10.0, 11.0)        # two arms, reward N(mean, 1)
BANDIT_BASELINE = 10.5             # the mean reward under the uniform policy
GAE_GAMMA = 0.99
GAE_LAMBDAS = (0.0, 0.5, 0.9, 0.95, 1.0)
CLIP_EPS = 0.2


def french(x: float, digits: int = 2) -> str:
    """A number with a French decimal comma and a true minus sign."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, title, lines, color, title_size=10.0, line_size=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=INK, lw=1.0))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK)
    if lines:
        ax.text(x + w / 2, y + h - 0.45, "\n".join(lines), ha="center", va="top", fontsize=line_size, color=INK,
                linespacing=1.3)


def arrow(ax, p, q, text="", rad=0.0, color=INK, dx=0.0, dy=0.1, style="-|>", ls="-", size=8.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=1.3, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=size,
                color=color, style="italic", bbox=dict(fc="white", ec="none", pad=1.2))


def light(c, a=0.22):
    return matplotlib.colors.to_rgba(c, a)


# --- 208.2 -------------------------------------------------------------------------------------------------------
def figure_dqn_loop() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    box(ax, 0.2, 3.2, 2.3, 1.6, "Environnement", ["CartPole-v1", "état s (4 nombres)", "récompense r"], light(C[5]))
    box(ax, 4.0, 3.2, 2.9, 1.6, "Réseau en ligne $Q_\\theta$", ["s → une valeur par action", "action : ε-greedy",
                                                                "(ε décroît)"], light(C[0]))
    box(ax, 0.2, 0.3, 2.9, 1.9, "Mémoire de rejeu", ["les N dernières transitions", "(s, a, r, s', terminé)",
                                                     "ReplayBuffer de rl.py"], light(C[3]))
    box(ax, 4.0, 0.3, 2.9, 1.9, "Mini-batch tiré au hasard", ["B transitions", "décorrélées",
                                                              "chacune resservira"], light(C[3], 0.12))
    box(ax, 8.4, 3.2, 3.4, 1.6, "Réseau cible $Q_{\\bar\\theta}$", ["copie figée de $Q_\\theta$", "calcule la cible y",
                                                                    "sans gradient (no_grad)"], light(C[6]))
    box(ax, 8.4, 0.3, 3.4, 1.9, "Loss et mise à jour", ["$y = r + \\gamma\\,(1-d)\\,\\max_{a'} Q_{\\bar\\theta}(s', a')$",
                                                        "Huber($Q_\\theta(s, a) - y$)", "Adam sur θ seulement"],
        light(C[1]))
    arrow(ax, (2.55, 4.2), (3.95, 4.2), "s", dy=0.05)
    arrow(ax, (3.95, 3.55), (2.55, 3.55), "action a", dy=-0.38)
    arrow(ax, (1.3, 3.15), (1.3, 2.25), "transition", dx=0.55, dy=-0.15)
    arrow(ax, (3.15, 1.25), (3.95, 1.25))
    arrow(ax, (6.95, 1.0), (8.35, 1.0), "s, a, r, s', d", dy=0.05)
    arrow(ax, (10.1, 3.15), (10.1, 2.25), "y (calculée sur s')", dx=0.75, dy=-0.15)
    arrow(ax, (8.35, 1.9), (6.95, 3.3), "gradient", rad=-0.25, color=C[1], dx=0.05, dy=-0.05)
    arrow(ax, (6.95, 4.55), (8.35, 4.55), "copie $\\theta \\to \\bar\\theta$ toutes les C mises à jour", color=C[6],
          dy=0.08, ls="--")
    save(fig, "boucle_dqn.png")


def figure_overestimation() -> dict[int, float]:
    rng = np.random.default_rng(0)
    ms = np.arange(1, 21)
    single, double, exact = [], [], []
    for m in ms:
        a = rng.normal(size=(200_000, m))
        b = rng.normal(size=(200_000, m))
        single.append(a.max(axis=1).mean())
        double.append(b[np.arange(len(b)), a.argmax(axis=1)].mean())
        f = lambda x, m=m: x * m * stats.norm.pdf(x) * stats.norm.cdf(x) ** (m - 1)  # noqa: E731
        exact.append(integrate.quad(f, -12, 12)[0])
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(ms, single, "o", color=C[1], ms=5, label="max d'un seul jeu d'estimations (Q-learning, DQN)")
    ax.plot(ms, exact, "-", color=C[1], lw=1.2, alpha=0.7, label="espérance exacte du max de m gaussiennes")
    ax.plot(ms, double, "s", color=C[0], ms=5, label="choisir sur un jeu, lire la valeur sur l'autre (Double)")
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.text(20.3, 0.02, "vraie valeur : 0", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    ax.set_xlabel("nombre d'actions m (toutes valent 0 ; chaque estimation a un bruit N(0, 1))")
    ax.set_ylabel("valeur moyenne retenue")
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Le max d'estimations bruitées surestime ; le double estimateur, non", fontsize=10.5)
    save(fig, "surestimation.png")
    return {int(m): float(e) for m, e in zip(ms, exact)}


# --- 208.3 -------------------------------------------------------------------------------------------------------
def figure_variance_baseline() -> dict[str, float]:
    rng = np.random.default_rng(1)
    n = 100_000
    p1 = 0.5                                        # uniform softmax policy: logits (0, 0)
    a = (rng.random(n) < p1).astype(int)
    r = rng.normal(np.take(BANDIT_MEANS, a), 1.0)
    score = a - p1                                  # d log pi(a) / d z_1 = 1[a = 1] - pi(1)
    g0, gb = r * score, (r - BANDIT_BASELINE) * score
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.4), sharey=True)
    bins = np.linspace(-7.5, 7.5, 121)
    for ax, g, title, col in [(a1, g0, "sans baseline (b = 0)", C[1]),
                              (a2, gb, f"avec baseline b = {french(BANDIT_BASELINE, 1)}", C[0])]:
        ax.hist(g, bins=bins, color=col, alpha=0.8)
        ax.axvline(g.mean(), color=INK, lw=1.4, ls="--")
        ax.set_title(f"{title} : moyenne {french(g.mean(), 2)}, écart-type {french(g.std(), 2)}", fontsize=9.5)
        ax.set_xlabel("estimation à un tirage de $\\partial J / \\partial z_1$")
    a1.set_ylabel("nombre de tirages")
    fig.suptitle("Bandit à deux bras, récompenses N(10, 1) et N(11, 1), politique uniforme : même gradient moyen",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "variance_baseline.png")
    return {"mean0": float(g0.mean()), "std0": float(g0.std()), "meanb": float(gb.mean()), "stdb": float(gb.std())}


# --- 208.4 -------------------------------------------------------------------------------------------------------
def figure_gae_lambda() -> None:
    ls = np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    for lam, col in zip(GAE_LAMBDAS, [C[7], C[1], C[3], C[0], C[6]]):
        w = (GAE_GAMMA * lam) ** ls
        style = dict(marker="o", ms=4) if lam == 0 else {}
        ax.plot(ls, w, color=col, lw=1.8, label=f"λ = {french(lam, 2).rstrip('0').rstrip(',')}", **style)
    ax.set_xlabel("l : l'erreur TD $\\delta_{t+l}$, l pas plus loin")
    ax.set_ylabel("poids $(\\gamma\\lambda)^l$ dans $\\hat A_t$")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(title=f"γ = {french(GAE_GAMMA, 2)}", fontsize=8.5, title_fontsize=8.5, loc="upper right")
    ax.set_title("GAE : λ règle jusqu'où l'avantage regarde (petit λ : biais du critique ; grand λ : variance)",
                 fontsize=10)
    save(fig, "gae_lambda.png")


# --- 208.5 -------------------------------------------------------------------------------------------------------
def figure_clip() -> None:
    rho = np.linspace(0.0, 2.0, 801)
    eps = CLIP_EPS
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    for ax, A, title in [(axes[0], 1.0, "avantage positif (A = +1)"), (axes[1], -1.0, "avantage négatif (A = −1)")]:
        term = np.minimum(rho * A, np.clip(rho, 1 - eps, 1 + eps) * A)
        ax.plot(rho, rho * A, color=MUTED, lw=1.1, ls=":", label="ρA (sans clip)")
        ax.plot(rho, term, color=C[0] if A > 0 else C[1], lw=2.4, label="min(ρA, clip(ρ, 1−ε, 1+ε)A)")
        flat = (rho > 1 + eps) if A > 0 else (rho < 1 - eps)
        lo, hi = (1 + eps, 2.0) if A > 0 else (0.0, 1 - eps)
        ax.axvspan(lo, hi, color=light(MUTED, 0.13))
        ax.text((lo + hi) / 2, -1.4 if A > 0 else 1.4, "terme clippé actif :\ngradient nul", ha="center",
                va="center", fontsize=8.5, color=MUTED)
        ax.plot([1.0], [A], "o", ms=8, mfc="none", mec=wbplot.BAD, mew=2)
        ax.annotate("départ (ρ = 1)", (1.0, A), (1.05, A * 0.35), fontsize=8.5, color=wbplot.BAD,
                    arrowprops=dict(arrowstyle="-", color=wbplot.BAD, lw=0.8))
        for v in (1 - eps, 1 + eps):
            ax.axvline(v, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks([0, 1 - eps, 1, 1 + eps, 2], ["0", "1−ε", "1", "1+ε", "2"])
        ax.set_xlabel("ratio ρ = π_θ(a|s) / π_old(a|s)")
        ax.set_ylabel("un terme de $L^{CLIP}$")
        ax.set_ylim(-2.1, 2.1)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left" if A > 0 else "upper right")
        del flat
    fig.suptitle(f"L'objectif clippé de PPO (ε = {french(eps, 1)}) : on maximise ; plus rien ne pousse au-delà du clip",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "clip_ppo.png")


# --- 208.7 -------------------------------------------------------------------------------------------------------
def figure_rlhf_dpo() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    w, h = 2.25, 1.35
    rows = [
        (4.9, "RLHF\n(InstructGPT, 2022)", C[0], [
            ("Préférences", ["prompt x, réponses", "$y_w$ préférée à $y_l$", "(annotateurs)"]),
            ("Modèle de récompense", ["Bradley-Terry :", "$-\\log\\sigma(r_\\phi(y_w) - r_\\phi(y_l))$"]),
            ("PPO", ["récompense $r_\\phi$", "− β · KL(π ‖ π_réf)", "par token ; un critique"]),
            ("Modèle aligné", ["π, proche de π_réf", "(le modèle SFT)"])]),
        (2.75, "DPO\n(2023)", C[1], [
            ("Préférences", ["les mêmes paires", "$(x, y_w, y_l)$"]),
            ("Référence figée", ["π_réf (SFT) :", "log-probabilités", "des deux réponses"]),
            ("Une loss supervisée", ["$-\\log\\sigma(\\beta[\\Delta_w - \\Delta_l])$", "$\\Delta = \\log\\pi - \\log\\pi_{réf}$",
                                     "ni récompense ni rollout"]),
            ("Modèle aligné", ["π"])]),
        (0.6, "GRPO\n(2024)", C[2], [
            ("Une question", ["G réponses tirées", "de la politique"]),
            ("Récompenses", ["vérifiables (réponse", "juste, tests passés)", "ou modèle appris"]),
            ("Avantage de groupe", ["$(r_i - \\mathrm{moy}) / \\mathrm{écart\\text{-}type}$", "pas de critique"]),
            ("Mise à jour clippée", ["comme PPO,", "+ β · KL dans la loss"])]),
    ]
    for y, label, col, boxes in rows:
        ax.text(0.55, y + h / 2, label, ha="center", va="center", fontsize=10, weight="bold", color=col)
        xs = [1.35 + i * (w + 0.62) for i in range(len(boxes))]
        for x, (title, lines) in zip(xs, boxes):
            box(ax, x, y, w, h, title, lines, light(col), title_size=9.3, line_size=8.0)
        for a, b in zip(xs[:-1], xs[1:]):
            arrow(ax, (a + w + 0.03, y + h / 2), (b - 0.03, y + h / 2))
    save(fig, "rlhf_dpo.png")


def figure_autoreset() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    T = 9
    ax.set_xlim(-1.6, T + 0.2)
    ax.set_ylim(-0.9, 2.1)
    ax.axis("off")
    ends = {0: [3], 1: [6]}                     # step at which each sub-environment ends its episode
    for env, y in [(0, 1.2), (1, 0.2)]:
        ax.text(-1.5, y + 0.3, f"sous-env. {env}", fontsize=9.5, va="center", color=INK)
        reset_steps = [e + 1 for e in ends[env]]
        for t in range(T):
            if t in ends[env]:
                col, txt = light(wbplot.BAD, 0.35), "fin\n(terminé ou\ntronqué)"
            elif t in reset_steps:
                col, txt = light(MUTED, 0.25), "reset :\nr = 0, action\nignorée"
            else:
                col, txt = light(C[0], 0.25), "pas\nnormal"
            ax.add_patch(FancyBboxPatch((t + 0.05, y), 0.9, 0.62, boxstyle="round,pad=0.01,rounding_size=0.05",
                                        fc=col, ec=INK, lw=0.7))
            ax.text(t + 0.5, y + 0.31, txt, ha="center", va="center", fontsize=6.6, color=INK)
    for t in range(T):
        ax.text(t + 0.5, 1.95, f"step {t}", ha="center", fontsize=8, color=MUTED)
    ax.text(T / 2 - 0.4, -0.6, "mode « next-step » (défaut de Gymnasium 1.x) : l'observation rendue à la fin est la "
            "vraie dernière observation ; l'appel suivant ne fait que le reset.\nCette transition grise (de la "
            "dernière observation vers la première du nouvel épisode) ne doit entrer ni dans la mémoire ni dans un "
            "rollout.", ha="center", va="center", fontsize=8.3, color=INK)
    save(fig, "autoreset.png")


def main() -> None:
    figure_dqn_loop()
    exact = figure_overestimation()
    stats_b = figure_variance_baseline()
    figure_gae_lambda()
    figure_clip()
    figure_rlhf_dpo()
    figure_autoreset()
    print("E[max of m N(0,1)]:", {m: round(exact[m], 4) for m in (2, 4, 10, 20)})
    print("bandit gradient:", {k: round(v, 3) for k, v in stats_b.items()})
    print("figures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
