#!/usr/bin/env python
"""Build the two notebooks of bonus chapter B6 from a single source (used by Claude).

    python tools/chapters/build_chB6.py
    python tools/run_all_notebooks.py chapitres/b6_explicabilite/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/b6_explicabilite/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3 and of the ✏️ paper exercises
B6.1, B6.3, B6.4 and B6.5 (02_exercices.md). The derivations ∂ B6.2 and B6.6, the estimate 🧮 B6.7, the reflection
(🗣️ B6.8, ⚖️ B6.9 and B6.10, 📄 B6.11) and the interview are corrected with 05_solutions.md. The answers are computed
by hand formulas in the context cell of the solutions notebook, then checked against the reference library
(mylearn.explain, mylearn.metrics), shap (explainers.Exact with an Independent masker that keeps every background
row), scikit-learn (confusion_matrix) and torchvision (the layer sizes of an untrained ResNet-18): part 0 downloads
nothing. captum is not installed: its trapezoid (✏️ B6.3 k) is computed from its rule (N points, weights 1/N halved at
both ends), the rule the tests of explain.py checked against captum 0.9.0.
The notebook itself (exercises B6.12 to B6.29) comes at the second generation session (NEXT_SESSION), in three phases:
tabular models, networks, fairness.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (Paper, badge, check_unique_keys, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "B6"
FOLDER = "chapitres/b6_explicabilite"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = r'''# The short answers only use the numbers of their statements (02_exercices.md). Features are numbered 1..M in
# the formulas, columns 0..M-1 in the code (feature 1 is column 0).
import itertools
import math

import numpy as np


def coalition_values(f, x, background):
    """v(S) for every mask S (a tuple of 0/1): the mean of f over the background rows where S takes x's values."""
    x, background = np.asarray(x, float), np.asarray(background, float)
    values = {}
    for mask in itertools.product((0, 1), repeat=len(x)):
        rows = background.copy()
        keep = np.array(mask, bool)
        rows[:, keep] = x[keep]
        values[mask] = float(np.mean(f(rows)))
    return values


def shapley_from_values(values, M, weight=None):
    """phi_i = sum over S without i of weight(|S|) (v(S + i) - v(S)); weight defaults to |S|! (M-|S|-1)! / M!."""
    weight = weight or (lambda k: math.factorial(k) * math.factorial(M - k - 1) / math.factorial(M))
    phi = []
    for i in range(M):
        total = 0.0
        for mask, value in values.items():
            if not mask[i]:
                with_i = tuple(1 if j == i else mask[j] for j in range(M))
                total += weight(sum(mask)) * (values[with_i] - value)
        phi.append(total)
    return phi


# ✏️ B6.1: f(x) = 2 x1 + x1 x3 + x2, x = (2, 1, 3), background (0, 2, 0) and (2, 2, 2)
F_B61 = lambda X: 2 * X[:, 0] + X[:, 0] * X[:, 2] + X[:, 1]
X_B61, BG_B61 = np.array([2.0, 1.0, 3.0]), np.array([[0.0, 2.0, 0.0], [2.0, 2.0, 2.0]])
V_B61 = coalition_values(F_B61, X_B61, BG_B61)
V0_B61, V1_B61, V13_B61 = V_B61[(0, 0, 0)], V_B61[(1, 0, 0)], V_B61[(1, 0, 1)]
W3_B61 = math.factorial(1) * math.factorial(1) / math.factorial(3)        # |S| = 1, M = 3
PHI_B61 = shapley_from_values(V_B61, 3)
FIRST_B61 = [V_B61[tuple(int(j == i) for j in range(3))] - V0_B61 for i in range(3)]   # contribution to the empty set
LAST_B61 = [V_B61[(1, 1, 1)] - V_B61[tuple(int(j != i) for j in range(3))] for i in range(3)]
EVALS_B61 = 2 ** 3 * len(BG_B61)
MEAN_ROW_B61 = BG_B61.mean(axis=0, keepdims=True)                          # (1, 2, 1)
PHI_MEAN_B61 = shapley_from_values(coalition_values(F_B61, X_B61, MEAN_ROW_B61), 3)
INTER_SHARE_B61 = [PHI_B61[0] - 2 * (X_B61[0] - BG_B61[:, 0].mean()), PHI_B61[2]]   # the share of x1 x3 in phi_1, phi_3

# ✏️ B6.3: f(x) = x1**2 + 3 x2 at x = (2, -1), zero baseline unless stated otherwise
X_B63 = np.array([2.0, -1.0])
F_B63 = lambda P: P[..., 0] ** 2 + 3 * P[..., 1]
GRAD_B63 = lambda P: np.stack([2 * P[..., 0], np.full(np.shape(P)[:-1], 3.0)], axis=-1)


def ig_rule_b63(n, method, x=X_B63, ref=(0.0, 0.0)):
    """Integrated gradients with n intervals (the conventions of the docstring of integrated_gradients)."""
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    k = np.arange(n + 1) / n
    alphas = {"left": k[:-1], "right": k[1:], "midpoint": (np.arange(n) + 0.5) / n, "trapezoid": k}[method]
    weights = np.full(len(alphas), 1 / n)
    if method == "trapezoid":
        weights[[0, -1]] /= 2
    grads = GRAD_B63(ref + alphas[:, None] * (x - ref))
    return (x - ref) * (weights[:, None] * grads).sum(axis=0)


GRAD_AT_X_B63 = [float(g) for g in GRAD_B63(X_B63)]
GXI_B63 = [float(g) for g in GRAD_B63(X_B63) * X_B63]
GAP_B63 = float(F_B63(X_B63) - F_B63(np.zeros(2)))
IG_EXACT_B63 = [float(X_B63[0] ** 2), float(3 * X_B63[1])]                 # x1 * integral of 2 a x1 = x1**2 ; 3 x2
IG1_LEFT_B63, IG1_RIGHT_B63, IG1_TRAP_B63 = (float(ig_rule_b63(5, m)[0]) for m in ("left", "right", "trapezoid"))
N_LEFT_B63 = math.ceil(round(X_B63[0] ** 2 / 0.03, 9))                     # the left error of IG_1 is x1**2 / n
REF2_B63 = (1.0, 0.0)
IG1_REF2_B63 = float(X_B63[0] ** 2 - REF2_B63[0] ** 2)                     # (x1 - x1') (x1 + x1')
SUM_REF2_B63 = float(F_B63(X_B63) - F_B63(np.array(REF2_B63)))
CAPTUM_B63 = IG1_TRAP_B63 * 5 / 6                                          # 6 points, weights 1/6 halved at both ends

# ✏️ B6.4: two 3 x 3 channels (index 0 = channel 1)
A_B64 = np.array([[[1, 0, 0], [2, 1, 0], [3, 2, 1]], [[0, 1, 2], [0, 0, 1], [0, 0, 0]]], dtype=float)
G_B64 = np.array([[[0, 1, 2], [1, 1, 1], [2, 1, 0]], [[-1, -1, -1], [0, 0, 0], [-1, -0.5, 0]]], dtype=float)
ALPHA_B64 = G_B64.mean(axis=(1, 2))
RAW_B64 = np.tensordot(ALPHA_B64, A_B64, axes=1)                           # before the ReLU
CAM_B64 = np.maximum(RAW_B64, 0)
NORM_B64 = CAM_B64 / np.abs(CAM_B64).max()
TOPRIGHT_B64 = float(RAW_B64[0, 2])
ZEROS_B64 = int((CAM_B64 == 0).sum())
CENTER_B64 = float(NORM_B64[1, 1])
NORELU_TOPRIGHT_B64 = float(RAW_B64[0, 2] / np.abs(RAW_B64).max())
OTHER_B64 = np.tensordot(np.array([-0.5, 1.0]), A_B64, axes=1)             # class k': alpha' = (-0.5, 1)
CORNERS_B64 = {(0, 0): "A", (2, 0): "B", (0, 2): "C"}
MAX_B64 = CORNERS_B64[tuple(int(i) for i in np.unravel_index(np.argmax(CAM_B64), CAM_B64.shape))]
MAX_OTHER_B64 = CORNERS_B64[tuple(int(i) for i in np.unravel_index(np.argmax(OTHER_B64), OTHER_B64.shape))]
SHIFT_B64 = 72 / 3 / 2                                                     # half a layer pixel, in image pixels

# ✏️ B6.5: two groups (TP, FN, FP, TN)
COUNTS_B65 = {"F": (72, 48, 19, 361), "H": (256, 64, 40, 280)}


def rates_b65(tp, fn, fp, tn):
    n = tp + fn + fp + tn
    return {"base": (tp + fn) / n, "sel": (tp + fp) / n, "tpr": tp / (tp + fn), "fpr": fp / (fp + tn),
            "ppv": tp / (tp + fp), "acc": (tp + tn) / n}


R_B65 = {g: rates_b65(*c) for g, c in COUNTS_B65.items()}
DP_B65 = abs(R_B65["H"]["sel"] - R_B65["F"]["sel"])
TPR_GAP_B65 = abs(R_B65["H"]["tpr"] - R_B65["F"]["tpr"])
FPR_GAP_B65 = abs(R_B65["H"]["fpr"] - R_B65["F"]["fpr"])
EO_B65 = max(TPR_GAP_B65, FPR_GAP_B65)
RATIO_B65 = R_B65["F"]["sel"] / R_B65["H"]["sel"]
ACC_GAP_B65 = abs(R_B65["F"]["acc"] - R_B65["H"]["acc"])
POS_F_B65 = COUNTS_B65["F"][0] + COUNTS_B65["F"][1]
EXTRA_B65 = round(R_B65["H"]["tpr"] * POS_F_B65 - COUNTS_B65["F"][0])

# quizzes and recalls with a computation
Q1E = 2 ** 4
Q2E = 0.83 - float(np.mean([0.66, 0.62, 0.65, 0.63, 0.64]))
Q3C = 3 * 2 + float(np.mean(np.array([0, 1, 2]) ** 2))
Q4A = 0.72 - 0.40 - 0.15 - (-0.05)
Q4C = 0.3 + (-0.1)
Q4E = 2 ** (5 - 1)
Q5C, Q5D = 2 * (4 - 3), 2 * (4 - 5)
Q6A, Q6B = 2 * (1.5 - 0.5), 0 * (1.5 - 0.5)
Q7A = 0                                                     # min(x, 1) is flat beyond 1
Q7B = 3 * (1 / 3)                                           # (x - 0) * integral of f'(3 a): f' = 1 for a < 1/3
Q7C = 0                                                     # the path from 2 to 3 stays where f is flat
Q8A, Q8C = 160 // 32, 224 // 4
Q9E = 1 / 10
Q11A = 0.6 - 0.25                                           # a perfect classifier selects exactly the positives
Q11D = 0.5 / (1 - 0.5) * (1 - 0.8) / 0.8 * (1 - 0.2)        # Chouldechova
EPS_R1, X_R1, G_R1 = 0.03, np.array([0.40, 0.98, 0.10]), np.array([-0.7, 0.2, 0.0])
R1A = [float(v) for v in np.clip(X_R1 + EPS_R1 * np.sign(G_R1), 0, 1)]
R1D = 0.01 * 3072 * 0.01
R2E = round((0.88 - 0.86) * 400)
CM_R3 = np.array([[88, 12], [10, 40]])                      # rows: truth 0, 1 ; columns: prediction 0, 1
(TN_R3, FP_R3), (FN_R3, TP_R3) = CM_R3
R3A, R3B = TP_R3 / (TP_R3 + FN_R3), FP_R3 / (FP_R3 + TN_R3)
R3C, R3D = TP_R3 / (TP_R3 + FP_R3), (TP_R3 + FP_R3) / CM_R3.sum()
'''

LIBRARY_CHECKS = r'''
# The same answers from the libraries (no download): mylearn (reference), shap, scikit-learn and torchvision
E = mylearn.explain

# ✏️ B6.1: shapley_values, then shap's exact explainer with every background row
phi, base = E.shapley_values(F_B61, X_B61, BG_B61)
assert np.allclose(phi, PHI_B61) and math.isclose(base, V0_B61)
assert math.isclose(phi.sum() + base, float(F_B61(X_B61[None])[0]))       # efficiency
assert np.allclose(E.shapley_values(F_B61, X_B61, MEAN_ROW_B61)[0], PHI_MEAN_B61)
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")   # tqdm, imported by shap
try:
    import shap
except ImportError:
    print("ℹ️ shap is not installed: the comparison with shap.explainers.Exact is skipped")
else:
    exact = shap.explainers.Exact(F_B61, shap.maskers.Independent(BG_B61, max_samples=len(BG_B61)))(X_B61[None])
    assert np.allclose(exact.values[0], PHI_B61) and math.isclose(float(exact.base_values[0]), V0_B61)

# quiz Q4 to Q6: the linear models of the statements (phi_i = w_i (x_i - mean_i))
lin = lambda w: (lambda X: X @ np.asarray(w, float))
assert np.allclose(E.shapley_values(lin([2, -3]), [4.0, 1.0], [[3.0, 1.0]])[0], [Q5C, -3 * 0])
assert math.isclose(E.shapley_values(lin([2, -3]), [4.0, 1.0], [[4.0, 1.0], [6.0, 3.0]])[0][0], Q5D)
phi_a = E.shapley_values(lin([2, 2]), [1.5, 1.5], [[0.0, 0.0], [1.0, 1.0]])[0]
phi_b = E.shapley_values(lin([4, 0]), [1.5, 1.5], [[0.0, 0.0], [1.0, 1.0]])[0]
assert math.isclose(phi_a[0], Q6A) and math.isclose(phi_b[1], Q6B) and math.isclose(phi_a.sum(), phi_b.sum())
# ∂ B6.2, question 5 (an application, not checked by wb.check)
phi_d2, base_d2 = E.shapley_values(lambda X: 1 + X @ np.array([0.5, -2.0, 0.0]), [4.0, 1.0, 7.0],
                                   [[2.0, 0.0, 1.0], [4.0, 1.0, 5.0], [0.0, 2.0, 3.0]])
assert np.allclose(phi_d2, [1.0, 0.0, 0.0]) and math.isclose(base_d2, 0.0, abs_tol=1e-12)

# ✏️ B6.3: integrated_gradients with the rules of its docstring
ig = lambda **kw: E.integrated_gradients(GRAD_B63, X_B63, **kw)
for method, answer in (("left", IG1_LEFT_B63), ("right", IG1_RIGHT_B63), ("trapezoid", IG1_TRAP_B63)):
    assert math.isclose(ig(n_steps=5, method=method)[0], answer)
assert np.allclose(ig(n_steps=5, method="trapezoid"), IG_EXACT_B63)       # exact: the gradient is linear on the path
assert abs(GAP_B63 - ig(n_steps=N_LEFT_B63, method="left").sum()) <= 0.03 - 1e-4      # far from the edge (4/134)
assert abs(GAP_B63 - ig(n_steps=N_LEFT_B63 - 1, method="left").sum()) > 0.03 + 1e-5
assert np.allclose(ig(baseline=list(REF2_B63), n_steps=3), [IG1_REF2_B63, IG_EXACT_B63[1]])
assert math.isclose(ig(baseline=list(REF2_B63), n_steps=3).sum(), SUM_REF2_B63, abs_tol=1e-12)
alphas = np.linspace(0, 1, 6)                                             # captum's riemann_trapezoid, 6 points
weights = np.full(6, 1 / 6)
weights[[0, -1]] /= 2
assert math.isclose(X_B63[0] * float(weights @ GRAD_B63(alphas[:, None] * X_B63)[:, 0]), CAPTUM_B63)

# quiz Q7: min(x, 1) at x = 3 (its gradient: 1 below 1, 0 beyond)
grad_min = lambda P: (P < 1).astype(float)
assert math.isclose(float(E.integrated_gradients(grad_min, [3.0], n_steps=3000, method="midpoint")[0]), Q7B,
                    rel_tol=1e-3)
assert float(E.integrated_gradients(grad_min, [3.0], baseline=[2.0], n_steps=10)[0]) == Q7C

# ✏️ B6.4: grad_cam, with and without the ReLU
assert np.allclose(E.grad_cam(A_B64[None], G_B64[None])[0], NORM_B64)
assert math.isclose(float(E.grad_cam(A_B64[None], G_B64[None], relu=False)[0, 0, 2]), NORELU_TOPRIGHT_B64)
assert np.allclose(E.grad_cam(A_B64[None], 10 * G_B64[None])[0], NORM_B64)
assert np.allclose(E.grad_cam(A_B64[None], G_B64[None], relu=False, normalize=False)[0], RAW_B64)

# ✏️ B6.5: group_rates and the two gaps, from the labels rebuilt out of the counts
def labels_from_counts(counts):
    y_true, y_pred, groups = [], [], []
    for group, (tp, fn, fp, tn) in counts.items():
        for t, p, k in ((1, 1, tp), (1, 0, fn), (0, 1, fp), (0, 0, tn)):
            y_true += [t] * k
            y_pred += [p] * k
            groups += [group] * k
    return np.array(y_true), np.array(y_pred), np.array(groups)


y_true, y_pred, groups = labels_from_counts(COUNTS_B65)
rates = E.group_rates(y_true, y_pred, groups)
for g in ("F", "H"):
    for ours, theirs in (("base", "base_rate"), ("sel", "selection_rate"), ("tpr", "tpr"), ("fpr", "fpr"),
                         ("ppv", "precision"), ("acc", "accuracy")):
        assert math.isclose(rates[g][theirs], R_B65[g][ours])
assert math.isclose(E.demographic_parity_difference(y_pred, groups), DP_B65)
assert math.isclose(E.equalized_odds_difference(y_true, y_pred, groups), EO_B65)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

for g, (tp, fn, fp, tn) in COUNTS_B65.items():
    assert (sk_confusion_matrix(y_true[groups == g], y_pred[groups == g]) == [[tn, fp], [fn, tp]]).all()
for p_, ppv, fnr, fpr in ((R_B65[g]["base"], R_B65[g]["ppv"], 1 - R_B65[g]["tpr"], R_B65[g]["fpr"]) for g in "FH"):
    assert math.isclose(p_ / (1 - p_) * (1 - ppv) / ppv * (1 - fnr), fpr)    # ∂ B6.6, question 5

# recall R3: the confusion matrix of ch. 3 (the reference metrics.py) and scikit-learn
y3 = np.repeat([0, 0, 1, 1], [TN_R3, FP_R3, FN_R3, TP_R3])
p3 = np.repeat([0, 1, 0, 1], [TN_R3, FP_R3, FN_R3, TP_R3])
assert (np.asarray(mylearn.metrics.confusion_matrix(y3, p3)) == CM_R3).all()
assert (sk_confusion_matrix(y3, p3) == CM_R3).all()
r3 = E.group_rates(y3, p3, np.zeros(len(y3), int))[0]
assert math.isclose(r3["tpr"], R3A) and math.isclose(r3["fpr"], R3B) and math.isclose(r3["precision"], R3C)
assert math.isclose(r3["selection_rate"], R3D)

# quiz Q8: the sizes of the maps of an untrained ResNet-18 (no download)
try:
    import torch
    from torchvision.models import resnet18
except ImportError:
    print("ℹ️ torchvision is not installed: the sizes of quiz Q8 are not checked")
else:
    net, sizes = resnet18(weights=None).eval(), {}
    hooks = [getattr(net, name).register_forward_hook(lambda m, i, o, name=name: sizes.__setitem__(name, o.shape[-1]))
             for name in ("layer1", "layer4")]
    with torch.no_grad():
        net(torch.zeros(1, 3, 160, 160))
        assert sizes["layer4"] == Q8A
        net(torch.zeros(1, 3, 224, 224))
        assert sizes["layer1"] == Q8C
    for hook in hooks:
        hook.remove()
print("✅ the paper answers agree with mylearn (reference: explain, metrics), shap, scikit-learn and torchvision")
'''

mylearn = importlib.import_module("mylearn_ref")   # the builder computes the mistakes with the same helpers
for _module in ("explain", "metrics"):
    importlib.import_module(f"mylearn_ref.{_module}")
exec(PAPER_CONTEXT)

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
WHOLE = "a whole number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
VEC2 = "a list of two numbers"
VEC3 = "a list of three numbers, 2 decimals"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    Paper("B6.Q1", "Interprétable par construction ou expliqué après coup ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis « Interprétable par construction, ou expliqué après coup » (§206.1) : la forêt se lit-elle elle-même, ou faut-il TreeSHAP ?",
                                           "C": "une carte de saillance est-elle le modèle, ou une explication ajoutée ? Relis 206.1"})),
        ("b", TF, "False", r'''mistakes={"relis la définition de l'explication post hoc (§206.1) : est-elle la règle du modèle ?": True}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "Grad-CAM se contente-t-il des prédictions ? Relis la phrase sur les méthodes spécifiques (§206.1)",
                                           "C": "Grad-CAM change-t-il le modèle ? Relis « Interprétable par construction, ou expliqué après coup »"})),
        ("d", TF, "True", r'''mistakes={"relis 206.1 : de quoi l'importance par permutation a-t-elle besoin ?": False}'''),
        ("e", WHOLE, "Q1E", r'''mistakes={"c'est la profondeur : on demande le nombre de feuilles": 4,
          "c'est le nombre de nœuds internes : on demande les feuilles": 15,
          "c'est le nombre de feuilles d'un arbre de profondeur 3": 8,
          "c'est le nombre total de nœuds : on demande les feuilles": 31}'''),
    ]),
    Paper("B6.Q2", "Explication globale ou locale : quel outil répond à quelle question ?", [
        ("a", LETTER, '"C"', letters("C", {"A": "la question porte-t-elle sur tout le modèle ou sur un dossier ? Relis « Global ou local » (§206.1)",
                                           "B": "la question compare-t-elle des groupes ? Relis le tableau de « Global ou local »"})),
        ("b", LETTER, '"A"', letters("A", {"B": "un waterfall explique combien de prédictions ? Relis le tableau de « Global ou local »",
                                           "C": "une courbe ICE d'un seul client dit-elle ce que le modèle utilise en tout ? Relis 206.2"})),
        ("c", TF, "False", r'''mistakes={"relis la fin de « Global ou local » (§206.1) : une importance moyenne dit-elle tout d'un cas particulier ?": True}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "une moyenne sur 1 000 clients explique-t-elle encore un client ? Relis la puce sur le beeswarm (§206.4)",
                                           "C": "la moyenne des |φ_i| mélange-t-elle une colonne ? Relis la puce sur le beeswarm (§206.4)"})),
        ("e", DEC2, "Q2E", r'''decimals=2, mistakes={"c'est le score après mélange : l'importance est une baisse": 0.64,
          "fais la moyenne des cinq mélanges avant de soustraire": 0.83 - 0.66,
          "vérifie le sens de la soustraction": -Q2E}'''),
    ]),
    Paper("B6.Q3", "Dépendance partielle et courbes ICE : ce que montre la moyenne, ce qu'elle cache", [
        ("a", LETTER, '"C"', letters("C", {"A": "fais la moyenne de v · x₂ sur tous les exemples : que vaut la moyenne de x₂ ?",
                                           "B": "relis la formule de PD_j(v) : on moyenne v · x₂ sur tous les exemples ; que vaut la moyenne de x₂ ?"})),
        ("b", WHOLE, "2", r'''mistakes={"chaque exemple a sa courbe ICE : quelles pentes prennent-elles selon x₂ ?": 1,
          "une courbe ICE de ce modèle est une droite : combien de valeurs de x₂ donnent-elles des pentes différentes ?": 4}'''),
        ("c", DEC2, "Q3C", r'''decimals=2, mistakes={"tu as mis la moyenne de x₂ au carré : la dépendance partielle fait la moyenne des prédictions": 6 + 1.0,
          "relis la question : on impose v = 2 à x₁ ; n'oublie pas le terme x₂²": 6.0,
          "fais la moyenne de x₂² sur les trois valeurs, pas leur somme": 6 + 5.0}'''),
        ("d", TF, "True", r'''mistakes={"relis le paragraphe qui suit le code de 206.2 : que fait centered=True à chaque courbe ?": False}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "relis la formule de PD_j(v) : sur quels exemples fait-on la moyenne ?",
                                           "C": "relis le piège classique « une dépendance partielle hors des données » (§206.2)"})),
    ]),
    Paper("B6.Q4", "Les quatre propriétés de Shapley : efficacité, symétrie, joueur nul, additivité", [
        ("a", DEC2, "Q4A", r'''decimals=2, mistakes={"l'efficacité répartit f(x) moins la valeur de base, pas f(x)": 0.72 - 0.15 + 0.05,
          "vérifie le signe de φ₂ : il est négatif": 0.72 - 0.40 - 0.15 - 0.05,
          "c'est l'écart à répartir entre les trois features : retire les deux valeurs connues": 0.72 - 0.40}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "l'efficacité porte sur la somme des parts : relis les quatre propriétés (§206.3)",
                                           "C": "la symétrie compare deux joueurs : relis les quatre propriétés (§206.3)"})),
        ("c", DEC1, "Q4C", r'''decimals=1, mistakes={"relis l'additivité : les parts s'ajoutent, elles ne se moyennent pas": 0.1,
          "vérifie le signe de la part de h": 0.4}'''),
        ("d", TF, "False", r'''mistakes={"la symétrie exige que les deux joueurs apportent la même chose à toute coalition : avec x₁ = 1 et x₂ = 5, est-ce le cas ? Essaie f = x₁ + x₂": True}'''),
        ("e", WHOLE, "Q4E", r'''mistakes={"c'est le nombre de toutes les coalitions : retire celles qui contiennent i": 32,
          "c'est le nombre d'ordres d'arrivée des cinq features": 120,
          "relis l'encadré 🧮 de 206.3 : un ensemble de m éléments a 2^m sous-ensembles": 5}'''),
    ]),
    Paper("B6.Q5", "Changer le fond (background) change les valeurs SHAP : vrai ou faux", [
        ("a", TF, "False", r'''mistakes={"relis l'efficacité avec un fond (§206.3) : de quoi dépend la somme des φ_i ?": True}'''),
        ("b", LETTER, '"C"', letters("C", {"A": "relis la définition de la valeur de base (§206.3) : sur quelles lignes la moyenne est-elle prise ?",
                                           "B": "relis la définition de la valeur de base (§206.3) : v(∅) est-il nul ?"})),
        ("c", WHOLE, "Q5C", r'''mistakes={"vérifie le sens (x₁ moins la moyenne du fond) et le fond utilisé : le fond A": -Q5C,
          "n'oublie pas le coefficient w₁ = 2": 1}'''),
        ("d", WHOLE, "Q5D", r'''mistakes={"vérifie le sens (x₁ moins la moyenne du fond) et le fond utilisé : le fond B": -Q5D,
          "n'oublie pas le coefficient w₁ = 2": -1}'''),
        ("e", TF, "False", r'''mistakes={"relis les deux modes de TreeExplainer (§206.4) : répondent-ils à la même question ?": True}'''),
        ("f", WHOLE, "100", r'''mistakes={"relis le paragraphe qui suit le code de shap (§206.3) : que fait max_samples par défaut ?": 300}'''),
    ]),
    Paper("B6.Q6", "Features corrélées : ce que SHAP répartit et ce qu'il ne dit pas", [
        ("a", WHOLE, "round(Q6A)", r'''mistakes={"relis le cas linéaire : φ₁ = w₁ (x₁ − moyenne), avec w₁ = 2": 4,
          "n'oublie pas de retirer la moyenne du fond": 3}'''),
        ("b", WHOLE, "round(Q6B)", r'''mistakes={"le modèle B lit-il x₂ ? Relis le cas linéaire": 2,
          "le modèle B lit-il x₂ ? Relis « Features corrélées : ce que SHAP répartit »": 4}'''),
        ("c", TF, "True", r'''mistakes={"calcule φ₁ + φ₂ pour les deux modèles, avec la formule du cas linéaire": False}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "relis « Features corrélées : ce que SHAP répartit » : le Shapley observationnel est-il fidèle au modèle ou aux données ?",
                                           "B": "relis « Features corrélées : ce que SHAP répartit » : pour qui le Shapley observationnel a-t-il été conçu ?"})),
        ("e", LETTER, '"A"', letters("A", {"B": "la plus grande des deux dépend du modèle choisi : relis la fin de « Features corrélées »",
                                           "C": "la répartition entre deux copies est-elle stable ? Relis la fin de « Features corrélées »"})),
    ]),
    Paper("B6.Q7", "Gradient, gradient × entrée, gradients intégrés : pourquoi une ligne de référence ?", [
        ("a", WHOLE, "Q7A", r'''mistakes={"au-delà de 1, f vaut toujours 1 : quelle est sa pente ?": 1,
          "relis la question : on demande la dérivée, pas la valeur de f": 3}'''),
        ("b", WHOLE, "round(Q7B)", r'''mistakes={"le gradient est nul en x, mais pas tout le long du chemin depuis 0 : relis la formule des gradients intégrés": 0,
          "c'est x − x′ : multiplie par la moyenne de la pente le long du chemin": 3}'''),
        ("c", WHOLE, "Q7C", r'''mistakes={"le chemin va de 2 à 3 : quelle est la pente de f sur ce segment ?": 1,
          "relis la question : la référence a changé": 3}'''),
        ("d", TF, "True", r'''mistakes={"pour f = wᵀx, le gradient est le même en tout point : calcule les deux attributions": False}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "un pixel égal à sa référence a-t-il un x_i − x′_i non nul ? Relis la formule des gradients intégrés",
                                           "C": "relis la formule des gradients intégrés : que vaut le facteur x_i − x′_i pour un pixel du chat ?"})),
    ]),
    Paper("B6.Q8", "Grad-CAM : pourquoi la dernière couche convolutive ?", [
        ("a", WHOLE, "Q8A", r'''mistakes={"c'est la taille de layer3, qui divise par 16 : relis les quatre détails de « Avec PyTorch : deux hooks »": 160 // 16,
          "c'est une division par 8 : de combien layer4 divise-t-il la taille de l'image ?": 160 // 8,
          "c'est la taille de layer1 : relis la question": 160 // 4}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "relis « Pourquoi la dernière couche convolutive ? » (§206.6) : que vient-il après elle ?",
                                           "C": "une softmax rend un vecteur de probabilités, sans position : relis « Pourquoi la dernière couche convolutive ? »"})),
        ("c", WHOLE, "Q8C", r'''mistakes={"relis l'énoncé : layer1 divise par 4": 224 // 8,
          "c'est la taille de layer4 à 224 pixels : relis la question": 7,
          "relis l'énoncé : layer1 divise par 4, pas par 2": 112}'''),
        ("d", TF, "True", r'''mistakes={"un poids α_c peut-il être négatif ? Relis l'exemple à la main de 206.6": False}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "relis la phrase qui suit la formule de Grad-CAM (§206.6) : de quel score prend-on le gradient ?",
                                           "B": "relis la phrase qui suit la formule de Grad-CAM (§206.6) : avant ou après la softmax ?"})),
    ]),
    Paper("B6.Q9", "Une carte de saillance qui ne dépend pas du modèle : que prouve le test de randomisation ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis « Les tests de santé » (§206.7) : qu'exige-t-on d'une explication quand le modèle change ?",
                                           "C": "le test de randomisation mesure-t-il la robustesse aux attaques ? Relis 206.7"})),
        ("b", TF, "False", r'''mistakes={"relis le paragraphe « Ce que prouve un test de santé réussi » (§206.7)": True}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "c'est la randomisation indépendante : relis les deux variantes (§206.7)",
                                           "B": "c'est la randomisation des données : relis les deux tests (§206.7)"})),
        ("d", TF, "False", r'''mistakes={"une corrélation de rangs compare l'ordre des pixels : deux cartes peuvent-elles avoir le même ordre et des valeurs différentes ?": True}'''),
        ("e", DEC1, "Q9E", r'''decimals=1, mistakes={"relis l'énoncé : CIFAR-10 a dix classes, et les labels du test ne sont pas mélangés": 0.5,
          "un réseau qui répond au hasard a-t-il une accuracy nulle ?": 0.0,
          "c'est l'accuracy d'entraînement d'un réseau qui apprend par cœur : relis la question, on demande le test": 1.0}'''),
    ]),
    Paper("B6.Q10", "Sources de biais : données, labels, variables de substitution, boucles de rétroaction", [
        ("a", LETTER, '"A"', letters("A", {"B": "les décisions du modèle fabriquent-elles ses données ? Relis les puces de « D'où vient un biais » (§206.8)",
                                           "C": "les transcriptions sont-elles fausses ? Relis les puces de « D'où vient un biais » (§206.8)"})),
        ("b", LETTER, '"B"', letters("B", {"A": "la note des managers est-elle une feature ou la cible ? Relis 206.8",
                                           "C": "le modèle sert-il une autre population que celle de son évaluation ? Relis 206.8"})),
        ("c", LETTER, '"C"', letters("C", {"A": "qui choisit les dossiers contrôlés ? Relis les puces de « D'où vient un biais »",
                                           "B": "une colonne porte-t-elle un attribut sensible à sa place ? Relis la question : qui choisit les contrôles ?"})),
        ("d", TF, "True", r'''mistakes={"une colonne qui n'est pas le sexe peut-elle en porter une part de l'information ? Relis la puce sur les variables de substitution (§206.8)": False}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "le modèle sert-il une autre population ? Relis les puces de 206.8 : qu'est-ce qui est mal mesuré ?",
                                           "C": "les décisions du modèle changent-elles le capteur ? Relis les puces de 206.8"})),
    ]),
    Paper("B6.Q11", "Parité démographique, égalité des chances, calibration : peut-on tout avoir ?", [
        ("a", DEC2, "Q11A", r'''decimals=2, mistakes={"un classifieur parfait sélectionne-t-il tout le monde ? Son taux de sélection vaut le taux de base": 0.0,
          "relis la définition de Δ_DP : la différence des taux de sélection, pas leur rapport": 0.6 / 0.25}'''),
        ("b", TF, "True", r'''mistakes={"calcule le TPR et le FPR d'un classifieur parfait dans chaque groupe": False}'''),
        ("c", TF, "True", r'''mistakes={"calcule la precision d'un classifieur parfait dans chaque groupe": False}'''),
        ("d", DEC2, "Q11D", r'''decimals=2, mistakes={"relis la relation de Chouldechova : le dernier facteur est 1 − FNR": 0.5 / 0.5 * 0.25 * 0.2,
          "relis la relation : (1 − PPV) / PPV, pas PPV / (1 − PPV)": 0.5 / 0.5 * 4 * 0.8,
          "relis la relation : le premier facteur est p / (1 − p)": 0.5 * 0.25 * 0.8}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "la parité démographique tient-elle compte de qui rembourserait ? Relis « Les critères » (§206.9)",
                                           "C": "un faux positif est ici un prêt accordé à qui ne rembourserait pas : est-ce l'erreur la plus coûteuse pour les demandeurs ?"})),
    ]),
    Paper("B6.Q12", "AI Act : dans quelle catégorie de risque tombe un scoring de crédit ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis la puce « haut risque » de « L'AI Act » (§206.11) : que vise le point 5 b) ?",
                                           "C": "le scoring de crédit figure-t-il parmi les pratiques interdites ? Relis « L'AI Act » (§206.11)"})),
        ("b", LETTER, '"A"', letters("A", {"B": "relis le point 5 b) cité en 206.11 : quelle exception contient-il ?",
                                           "C": "la détection de fraude figure-t-elle à l'article 5 ? Relis « L'AI Act »"})),
        ("c", LETTER, '"C"', letters("C", {"A": "relis la puce « risque inacceptable » de « L'AI Act » (§206.11)",
                                           "B": "l'article 50 parle de transparence envers l'utilisateur : relis la puce « risque inacceptable »"})),
        ("d", TF, "False", r'''mistakes={"relis l'encadré 🕰️ sur la régulation (§206.11) : quelles dates le règlement 2026/1744 a-t-il fixées ?": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "relis le dernier paragraphe de « L'AI Act » : qui, du fournisseur ou du déployeur, réalise l'analyse d'impact ?",
                                           "C": "relis le dernier paragraphe de « L'AI Act » : l'article 27 s'adresse à qui ?"})),
        ("f", TF, "False", r'''mistakes={"relis le texte du point 5 b) cité en 206.11 : de quelles personnes évalue-t-il la solvabilité ?": True}'''),
    ]),
    Paper("B6.R1", "Visualiser ce qu'a appris un CNN, et l'attaque FGSM", [
        ("a", VEC3, "R1A", r'''decimals=2, mistakes={"ramène chaque pixel dans [0, 1] après le pas": [0.37, 1.01, 0.1],
          "vérifie le signe : FGSM monte la loss, x + ε sign(∇L)": [0.43, 0.95, 0.1],
          "FGSM ajoute ε fois le signe du gradient, pas ε fois le gradient": [0.38, 0.99, 0.1],
          "le signe de 0 vaut 0 : le troisième pixel ne bouge pas": [0.37, 1.0, 0.13]}'''),
        ("b", TF, "True", r'''mistakes={"relis le début de 206.5 et la section FGSM du ch. 21 : les deux calculs dérivent-ils par rapport à la même chose, avec la même passe arrière ?": False}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "relis le rappel maths « le gradient par rapport à l'entrée » du ch. 21 : les poids bougent-ils ?",
                                           "C": "la maximisation d'activation a-t-elle besoin de labels ? Relis le ch. 21"})),
        ("d", DEC2, "R1D", r'''decimals=2, mistakes={"relis la formule ε Σ|w_j| du ch. 21 : il manque un facteur": 0.01 * 0.01,
          "c'est ε fois le nombre de pixels : n'oublie pas |w_j|": 0.01 * 3072,
          "relis l'énoncé : ε = 0,01 et |w_j| = 0,01": 3.072}'''),
        ("e", TF, "False", r'''mistakes={"relis le rappel maths du ch. 21 : la montée de gradient cherche-t-elle un minimum ?": True}'''),
    ]),
    Paper("B6.R2", "feature_importances_ contre importance par permutation", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis l'importance des features au ch. 14 : qu'arrive-t-il aux features à forte cardinalité ?",
                                           "C": "une baisse d'impureté peut-elle être négative ? Relis le ch. 14"})),
        ("b", TF, "True", r'''mistakes={"relis le ch. 14 : sur quelles données l'importance MDI est-elle calculée ?": False}'''),
        ("c", WHOLE, "1", r'''mistakes={"relis le ch. 14 : les importances MDI sont normalisées": 100}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "mélanger un identifiant fait-il perdre quelque chose sur des exemples jamais vus ? Relis le ch. 14",
                                           "C": "une importance par permutation fortement négative voudrait dire que le mélange aide beaucoup : est-ce plausible ?"})),
        ("e", WHOLE, "R2E", r'''mistakes={"c'est la baisse d'accuracy : multiplie-la par le nombre d'exemples": 0,
          "vérifie l'ordre de grandeur : 2 points de 400 exemples": 80}'''),
    ]),
    Paper("B6.R3", "TPR, FPR et precision à partir d'une matrice de confusion", [
        ("a", DEC2, "R3A", r'''decimals=2, mistakes={"le TPR divise TP par les positifs réels (la ligne 1), pas par les sélectionnés": R3C,
          "le TPR divise par les positifs réels, pas par tous les exemples": 40 / 150}'''),
        ("b", DEC2, "R3B", r'''decimals=2, mistakes={"c'est la part de faux positifs parmi les sélectionnés : le FPR divise par les négatifs réels": 12 / 52,
          "relis la disposition : la case [0, 1] contient quoi ?": 10 / 50,
          "le FPR divise par les négatifs réels, pas par tous les exemples": 12 / 150}'''),
        ("c", DEC3, "R3C", r'''decimals=3, mistakes={"la precision divise TP par les sélectionnés (la colonne 1), pas par les positifs réels": R3A,
          "c'est la part de faux positifs parmi les sélectionnés : on demande la part de vrais positifs": 12 / 52}'''),
        ("d", DEC3, "R3D", r'''decimals=3, mistakes={"c'est la part de positifs réels (la ligne 1) : on demande la part de prédictions positives": 50 / 150,
          "le taux de sélection divise par tous les exemples, pas par les négatifs réels": 52 / 100}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "relis la disposition : la ligne 0 est la vérité 0, la colonne 1 la prédiction 1",
                                           "C": "relis la disposition : la ligne 0 est la vérité 0"})),
    ]),
    Paper("B6.1", "Valeurs de Shapley exactes à la main : trois features, huit coalitions", [
        ("a", WHOLE, "round(V0_B61)", r'''mistakes={"tu as calculé f sur la ligne moyenne du fond : v(∅) est la moyenne de f sur les deux lignes": 5,
          "relis la question : v(∅) n'a aucune feature du client": 11,
          "tu as additionné les deux lignes : v(S) est une moyenne": 12}'''),
        ("b", WHOLE, "round(V1_B61)", r'''mistakes={"dans v({1}), seule la feature 1 vient du client : as-tu inversé le masque ?": round(V_B61[(0, 1, 1)]),
          "tu as additionné les deux lignes : v(S) est une moyenne": 16,
          "relis la question : v({1}) garde x₂ et x₃ du fond": 11}'''),
        ("c", WHOLE, "round(V13_B61)", r'''mistakes={"dans v({1, 3}), les features 1 et 3 viennent du client : as-tu inversé le masque ?": round(V_B61[(0, 1, 0)]),
          "tu as additionné les deux lignes : v(S) est une moyenne": 24}'''),
        ("d", DEC2, "W3_B61", r'''decimals=2, mistakes={"c'est le poids des coalitions de taille 0 ou 2 : {3} a un membre": 1 / 3,
          "les poids ne sont pas uniformes sur les quatre coalitions sans la feature 1 : relis la formule": 1 / 4,
          "les poids ne sont pas uniformes sur les huit coalitions : relis la formule": 1 / 8}'''),
        ("e", DEC1, "PHI_B61[0]", r'''decimals=1, mistakes={"tu n'as gardé que l'apport de la feature 1 à la coalition vide (ou le seul terme 2x₁) : il y a quatre coalitions sans elle": FIRST_B61[0],
          "tu n'as gardé que son apport quand elle arrive la dernière : il y a quatre coalitions sans elle": LAST_B61[0]}'''),
        ("f", WHOLE, "round(PHI_B61[1])", r'''mistakes={"vérifie le signe : x₂ est plus petit que ses valeurs du fond": 1,
          "x₂ entre dans f : un joueur nul n'apporte jamais rien, est-ce le cas ?": 0}'''),
        ("g", DEC1, "PHI_B61[2]", r'''decimals=1, mistakes={"tu n'as gardé que l'apport de la feature 3 à la coalition vide": FIRST_B61[2],
          "tu n'as gardé que son apport quand elle arrive la dernière": LAST_B61[2],
          "as-tu pris pour valeur de base f sur la ligne moyenne du fond ? Relis a)": PHI_B61[2] + 1}'''),
        ("h", TF, "False", r'''mistakes={"calcule ce que l'interaction apporte à φ₁ (retire le terme 2(x₁ − moyenne)) et compare à φ₃": True}'''),
        ("i", WHOLE, "EVALS_B61", r'''mistakes={"chaque v(S) demande une évaluation par ligne du fond : il y en a deux": 8,
          "chaque v(S) se calcule une fois et sert aux trois features": 3 * EVALS_B61,
          "compte toutes les coalitions, la vide et la pleine comprises": 6 * 2,
          "la coalition pleine se calcule aussi sur les deux lignes du fond, comme les autres": EVALS_B61 - 1}'''),
        ("j", DEC1, "PHI_MEAN_B61[0]", r'''decimals=1, mistakes={"relis la question : le fond n'a plus qu'une ligne, recalcule les v(S)": 3.5,
          "tu n'as gardé que le terme 2x₁ : la feature 1 entre aussi dans l'interaction": 2.0}'''),
        ("k", LETTER, '"B"', letters("B", {"A": "relis l'encadré 📐 de la fiche : la feature 1 est la colonne 0",
                                           "C": "relis la signature de shapley_values : que rend-elle à part phi ?"})),
    ]),
    Paper("B6.3", "Gradients intégrés à la main sur f(x) = x₁² + 3x₂ et vérification de la complétude", [
        ("a", VEC2, "GRAD_AT_X_B63", r'''decimals=1, mistakes={"la dérivée de x₁² est 2x₁": [2.0, 3.0],
          "la dérivée de 3x₂ par rapport à x₂ ne dépend pas de x₂": [4.0, -3.0]}'''),
        ("b", VEC2, "GXI_B63", r'''decimals=1, mistakes={"multiplie chaque dérivée par la valeur de sa feature": GRAD_AT_X_B63,
          "vérifie le signe : x₂ = −1": [8.0, 3.0]}'''),
        ("c", WHOLE, "round(GAP_B63)", r'''mistakes={"c'est la somme de gradient × entrée : on demande l'écart de prédiction f(x) − f(x′)": 5,
          "vérifie le signe de 3x₂ : x₂ = −1": 7,
          "vérifie le sens de la soustraction : f(x) − f(x′)": -1}'''),
        ("d", VEC2, "IG_EXACT_B63", r'''decimals=1, mistakes={"c'est gradient × entrée : intègre la dérivée le long du chemin": GXI_B63,
          "multiplie l'intégrale par x_i − x′_i": [2.0, 3.0],
          "vérifie le signe de la seconde attribution : x₂ = −1": [4.0, 3.0]}'''),
        ("e", DEC1, "IG1_LEFT_B63", r'''decimals=1, mistakes={"c'est la règle 'right' : 'left' prend les points k/n pour k = 0 à n − 1": IG1_RIGHT_B63,
          "multiplie la moyenne des gradients par x₁ − x′₁": IG1_LEFT_B63 / 2,
          "n = 5 intervalles : les points 0 ; 0,2 ; 0,4 ; 0,6 ; 0,8": float(ig_rule_b63(4, "left")[0]),
          "relis la question : la règle de gauche, avec n = 5": 4.0}'''),
        ("f", DEC1, "IG1_RIGHT_B63", r'''decimals=1, mistakes={"c'est la règle 'left' : 'right' prend les points k/n pour k = 1 à n": IG1_LEFT_B63,
          "multiplie la moyenne des gradients par x₁ − x′₁": IG1_RIGHT_B63 / 2,
          "relis la question : la règle de droite, avec n = 5": 4.0}'''),
        ("g", DEC1, "IG1_TRAP_B63", r'''decimals=1, mistakes={"relis la question : la règle des trapèzes, dont les deux extrémités comptent pour moitié": 4.8,
          "multiplie la somme pondérée par x₁ − x′₁": 2.0,
          "les poids valent 1/n, divisés par deux aux extrémités : relis l'énoncé": 2 * (0 + 0.8 + 1.6 + 2.4 + 3.2 + 2) / 6}'''),
        ("h", WHOLE, "N_LEFT_B63", r'''mistakes={"avec ce n, l'erreur dépasse encore un peu 0,03 : prends le plus petit entier qui convient": N_LEFT_B63 - 1,
          "l'erreur de la règle de gauche sur IG₁ dépend de x₁² : recalcule-la": 34,
          "recalcule l'erreur de la règle de gauche en fonction de n : où intervient x₁ ?": 67},
          fractional="le nombre d'intervalles est entier : prends le plus petit n qui convient"'''),
        ("i", WHOLE, "round(IG1_REF2_B63)", r'''mistakes={"relis la question : la référence a changé ; un gradient pris en x seul ne suffit pas non plus": 4,
          "le chemin part de x′ : son point est x′ + α(x − x′)": 2}'''),
        ("j", WHOLE, "round(SUM_REF2_B63)", r'''mistakes={"relis la question : avec cette référence, la somme vaut f(x) − f(x′)": 1,
          "relis la question : on demande la somme des deux attributions": 3,
          "c'est la seconde attribution seule : ajoute IG₁": -3}'''),
        ("k", DEC2, "CAPTUM_B63", r'''decimals=2, mistakes={"c'est le trapèze de ta docstring : relis l'encadré 🕰️ de 206.6 sur captum": 4.0,
          "le facteur est n/(n + 1), pas (n + 1)/n : relis l'encadré 🕰️ de 206.6": 4.8,
          "6 points font 5 intervalles, pas 6 : relis l'énoncé": 4 * 6 / 7}'''),
        ("l", LETTER, '"C"', letters("C", {"A": "n_steps compte les intervalles dans ta librairie, pas les points",
                                           "B": "relis la question : e) utilise la règle 'left'"})),
    ]),
    Paper("B6.4", "Grad-CAM à la main : deux cartes 3×3 et leurs gradients", [
        ("a", DEC1, "float(ALPHA_B64[0])", r'''decimals=1, mistakes={"α_c est la moyenne des gradients, pas leur somme": float(G_B64[0].sum()),
          "α_c est la moyenne des gradients seuls, sans les multiplier par les activations": float((G_B64[0] * A_B64[0]).mean())}'''),
        ("b", DEC1, "float(ALPHA_B64[1])", r'''decimals=1, mistakes={"α_c est la moyenne des gradients, pas leur somme": float(G_B64[1].sum()),
          "vérifie le signe : les gradients du canal 2 sont négatifs ou nuls": -float(ALPHA_B64[1]),
          "α_c est la moyenne des gradients seuls, sans les multiplier par les activations": float((G_B64[1] * A_B64[1]).mean())}'''),
        ("c", DEC1, "TOPRIGHT_B64", r'''decimals=1, mistakes={"relis la question : avant le ReLU": 0.0,
          "vérifie le signe de α₂": -TOPRIGHT_B64,
          "relis la case : en haut à droite, A¹ vaut 0 et A² vaut 2": -0.5}'''),
        ("d", WHOLE, "ZEROS_B64", r'''mistakes={"compte aussi les cases dont la somme pondérée est négative mais proche de 0": 2,
          "une case positive n'est pas mise à 0 : compte les cases négatives de la somme pondérée": 4}'''),
        ("e", DEC2, "CENTER_B64", r'''decimals=2, mistakes={"n'oublie pas de diviser par la plus grande valeur de la carte": 1.0,
          "on divise par la plus grande valeur absolue, pas par la somme de la carte": float(CAM_B64[1, 1] / CAM_B64.sum())}'''),
        ("f", LETTER, f'"{MAX_B64}"', letters(MAX_B64, {"A": "cherche la plus grande valeur de α₁A¹ + α₂A² après le ReLU",
                                                         "B": "compare les valeurs de α₁A¹ + α₂A² dans les trois coins",
                                                         "C": "en haut à droite, quel est le signe de la somme pondérée ?"})),
        ("g", DEC2, "NORELU_TOPRIGHT_B64", r'''decimals=2, mistakes={"sans le ReLU, la case garde son signe": -NORELU_TOPRIGHT_B64,
          "n'oublie pas de diviser par la plus grande valeur absolue": TOPRIGHT_B64,
          "relis la question : sans le ReLU, la case n'est pas mise à 0": 0.0}'''),
        ("h", LETTER, f'"{MAX_OTHER_B64}"', letters(MAX_OTHER_B64, {"A": "calcule −0,5 A¹ + A² case par case : que vaut-elle en haut à gauche ?",
                                                                     "B": "calcule −0,5 A¹ + A² case par case : que vaut-elle en bas à gauche ?",
                                                                     "C": "calcule −0,5 A¹ + A² case par case : que vaut-elle en haut à droite ?"})),
        ("i", TF, "False", r'''mistakes={"multiplier les gradients par 10 multiplie les α, donc toute la carte : que fait la division par le maximum ?": True}'''),
        ("j", WHOLE, "round(SHIFT_B64)", r'''mistakes={"c'est la taille d'un pixel de la couche : on demande un demi-pixel": 24,
          "relis l'énoncé : la carte fait 3 × 3 et l'image 72 × 72": 36}'''),
    ]),
    Paper("B6.5", "Audit à la main : deux groupes, deux matrices de confusion, cinq indicateurs", [
        ("a", DEC2, 'R_B65["F"]["base"]', r'''decimals=2, mistakes={"relis la question : on demande la part de positifs réels (TP + FN)": R_B65["F"]["sel"],
          "le taux de base divise par tout le groupe, pas par les négatifs": 120 / 380}'''),
        ("b", DEC3, 'R_B65["F"]["sel"]', r'''decimals=3, mistakes={"le taux de sélection compte tous les sélectionnés, TP et FP": 72 / 500,
          "relis la question : on demande la part sélectionnée (TP + FP)": R_B65["F"]["base"]}'''),
        ("c", DEC2, 'R_B65["F"]["tpr"]', r'''decimals=2, mistakes={"c'est le FNR : le TPR divise TP par les positifs réels": 0.4,
          "c'est la precision : le TPR divise par les positifs réels, pas par les sélectionnés": R_B65["F"]["ppv"],
          "le TPR divise par les positifs réels, pas par tout le groupe": 72 / 500}'''),
        ("d", DEC3, 'R_B65["H"]["fpr"]', r'''decimals=3, mistakes={"le FPR divise par les négatifs réels, pas par tout le groupe": 40 / 640,
          "c'est la part de vrais négatifs : le FPR compte les faux positifs": 280 / 320,
          "le FPR divise par les négatifs réels (FP + TN), pas par les sélectionnés": 40 / 296}'''),
        ("e", DEC3, 'R_B65["F"]["ppv"]', r'''decimals=3, mistakes={"la precision divise TP par les sélectionnés (TP + FP)": R_B65["F"]["tpr"],
          "c'est la part de faux positifs parmi les sélectionnés : on demande la precision": 19 / 91}'''),
        ("f", DEC2, "DP_B65", r'''decimals=2, mistakes={"c'est l'écart des taux de base : la parité démographique compare les taux de sélection": 0.5 - 0.24,
          "relis la question : la parité démographique compare les taux de sélection": TPR_GAP_B65,
          "les taux de sélection divisent TP + FP par l'effectif du groupe": 296 / 640 - 72 / 500}'''),
        ("g", DEC2, "EO_B65", r'''decimals=2, mistakes={"l'equalized odds prend le plus grand des deux écarts, TPR et FPR": FPR_GAP_B65,
          "l'equalized odds prend le plus grand des deux écarts, pas leur somme": TPR_GAP_B65 + FPR_GAP_B65,
          "l'equalized odds prend le plus grand des deux écarts, pas leur moyenne": (TPR_GAP_B65 + FPR_GAP_B65) / 2}'''),
        ("h", DEC2, "RATIO_B65", r'''decimals=2, mistakes={"relis la question : le taux des femmes divisé par celui des hommes": 1 / RATIO_B65,
          "relis la question : on demande le rapport, pas son complément": 1 - RATIO_B65}'''),
        ("i", TF, "False", r'''mistakes={"compare ton rapport h) à 4/5 = 0,8": True}'''),
        ("j", TF, "True", r'''mistakes={"calcule l'accuracy (TP + TN) / n de chaque groupe": False}'''),
        ("k", WHOLE, "EXTRA_B65", r'''mistakes={"c'est le nombre total de qualifiées sélectionnées : on demande combien de plus": 96,
          "le TPR des hommes s'applique aux 120 candidates qualifiées, pas à toutes": round(0.8 * 500) - 72,
          "relis l'énoncé : le TPR à atteindre est celui des hommes": 120 - 72}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ B6.1, B6.3, B6.4 et B6.5)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; un "
               "vecteur : une liste de nombres (`[0.5, -1.25]`). Un taux s'écrit comme une proportion (`0.25`). "
               "Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore "
               "remplies affichent ⏳. Les questions « dans ta copie », les dérivations ∂ B6.2 et B6.6, "
               "l'estimation 🧮 B6.7, la réflexion (🗣️ B6.8, ⚖️ B6.9 et B6.10, 📄 B6.11) et l'entretien se "
               "corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to C (exercises B6.12 to B6.29): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "B6.12 à B6.17", "les modèles tabulaires : explications globales sur California, Shapley exact avec ta "
                           "librairie, shap sur un gradient boosting, le rôle du fond, Penguins"),
    ("B", "B6.18 à B6.22", "les réseaux : gradients intégrés et Grad-CAM dans ta librairie, attributions sur un CNN "
                           "CIFAR-10, Grad-CAM par hooks sur le ResNet du B1, un Grad-CAM à réparer, le test de santé"),
    ("C", "B6.23 à B6.29", "l'équité : tes indicateurs par groupe, l'audit d'Adult avec intervalles, l'effet du "
                           "retrait du sexe, l'atténuation et son coût, la fiche modèle et le défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | B6.Q1 à B6.Q12, B6.R1 à B6.R3, B6.1, B6.3 à B6.5 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(ex.table_row(part.key if i == 0 else ""))
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# B6 · Explicabilité, équité et éthique — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/b6_explicabilite/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py B6`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# B6 · Explicabilité, équité et éthique — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Calculer à la main des valeurs de Shapley exactes, des gradients intégrés et une carte Grad-CAM, "
               "et les relier à ta librairie (`explain.py`) et à shap.\n"
               "- Expliquer un modèle tabulaire globalement (permutation, dépendance partielle, ICE) et localement "
               "(Shapley, waterfall), en disant quel fond a servi.\n"
               "- Produire des cartes d'attribution sur un CNN et les soumettre à un test de santé.\n"
               "- Auditer un classifieur groupe par groupe, avec des intervalles, essayer des corrections, chiffrer "
               "leur coût et documenter le tout dans une fiche modèle.\n\n"
               "**Rappel express.** Shapley : $v(S) = \\frac{1}{B}\\sum_k f(\\mathbf{x}_S, \\mathbf{b}^{(k)}_{\\bar S})$, "
               "$\\phi_i = \\sum_{S \\not\\ni i} \\frac{|S|!\\,(M - |S| - 1)!}{M!}\\,[v(S \\cup \\{i\\}) - v(S)]$, "
               "$\\sum_i \\phi_i = f(\\mathbf{x}) - v(\\varnothing)$ ; modèle linéaire : $\\phi_i = w_i(x_i - \\bar b_i)$. "
               "Gradients intégrés : $\\operatorname{IG}_i = (x_i - x'_i)\\int_0^1 \\partial_i f(\\mathbf{x}' + "
               "\\alpha(\\mathbf{x} - \\mathbf{x}'))\\,d\\alpha$, de somme $f(\\mathbf{x}) - f(\\mathbf{x}')$. Grad-CAM : "
               "$\\alpha_c$ = moyenne spatiale de $\\partial y^k / \\partial A^c$, carte $\\operatorname{ReLU}(\\sum_c "
               "\\alpha_c A^c)$. Équité : $\\Delta_{\\text{DP}}$ = écart des taux de sélection, $\\Delta_{\\text{EO}}$ = "
               "le plus grand des écarts de TPR et de FPR ; $\\operatorname{FPR} = \\frac{p}{1-p}\\cdot\\frac{1 - "
               "\\operatorname{PPV}}{\\operatorname{PPV}}\\cdot(1 - \\operatorname{FNR})$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code B6.12 à B6.29 (modèles tabulaires, réseaux, équité) seront ajoutés à la "
             "prochaine session de génération ; relance alors `python tools/start_chapter.py B6` pour obtenir le "
             "notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu calculer à la main les valeurs de Shapley d'un petit jeu, avec un fond, et dire ce que "
               "le fond change ?\n"
               "2. Sais-tu calculer des gradients intégrés et une carte Grad-CAM à la main, et vérifier la "
               "complétude ?\n"
               "3. Sais-tu auditer un classifieur par groupe (taux de sélection, TPR, FPR, precision, écarts) et "
               "expliquer pourquoi tous les critères ne peuvent pas tenir ensemble ?\n"
               "4. Sais-tu dire ce qu'un scoring de crédit implique au regard du RGPD et de l'AI Act ?\n\n"
               "**Pour aller plus loin** : l'article de Lundberg et Lee et le livre en ligne de C. Molnar, cités dans "
               "la fiche. La suite : le bonus B7, le MLOps."
               + later)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT + LIBRARY_CHECKS, PAPER_INTRO)
    for part in PARTS:
        cells += part_cells(part, kind)
    cells += footer_cells(kind)
    return cells


def main() -> int:
    write_notebook(f"{FOLDER}/03_notebook.ipynb", build("exercise"))
    write_notebook(f"{FOLDER}/05_solutions.ipynb", build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER)
    n_ex = sum(len(part.exercises) for part in PARTS)
    print(f"✅ {FOLDER}/03_notebook.ipynb and 05_solutions.ipynb written ({len(PAPER)} paper exercises, "
          f"{n_checks} paper checks, {n_ex} notebook exercises)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
