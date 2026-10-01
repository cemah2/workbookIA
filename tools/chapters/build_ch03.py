#!/usr/bin/env python
"""Build the two notebooks of chapter 3 from a single source (used by Claude).

    python tools/chapters/build_ch03.py
    python tools/run_all_notebooks.py chapitres/ch03_probabilites/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch03_probabilites/03_notebook.ipynb

Part 0 checks the ✏️ paper exercises (3.1, 3.2, 3.3, 3.5, 3.6, 3.7). Parts A to D are the
notebook exercises 3.12 to 3.29: probabilities with areas and tables (A), the confusion
matrix and the binary measures of mylearn.metrics (B), threshold, prevalence and the
scikit-learn tools (C), ROC and precision-recall curves, averages, calibration and the
challenge (D).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, guarded, md, paper_cells,  # noqa: E402
                         part_cells, setup_cell, write_notebook)

CHAPTER = "3"
FOLDER = "chapitres/ch03_probabilites"

# ---------------------------------------------------------------------------
# Part 0: checks of the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The paper exercises only use the numbers of their statements (02_exercices.md)
import numpy as np

WALL = 4 * 2.5                                     # 3.1: area of the wall (m²)
AREA_A, AREA_B, AREA_AB = 2 * 1.5, 2 * 1, 1 * 1    # 3.1: rectangles A, B and their overlap
TRUTH_32 = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0])   # 3.2
PRED_32 = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
TP_32 = int(np.sum((TRUTH_32 == 1) & (PRED_32 == 1)))
FP_32 = int(np.sum((TRUTH_32 == 0) & (PRED_32 == 1)))
FN_32 = int(np.sum((TRUTH_32 == 1) & (PRED_32 == 0)))
TN_32 = int(np.sum((TRUTH_32 == 0) & (PRED_32 == 0)))
ICE = np.array([[42, 18], [38, 52]])              # 3.3: rows vanilla, chocolate; columns cone, cup
TP_35, FN_35, FP_35, TN_35 = 40, 10, 60, 890      # 3.5: fraud detector, 1 000 transactions
SPECIES_36 = np.array([[41, 3, 1], [7, 13, 0], [0, 2, 33]])   # 3.6: rows truth, columns prediction


def rates(tp, fn, fp, tn):
    """Every measure of a binary confusion matrix (3.5)."""
    n = tp + fn + fp + tn
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    specificity, npv = tn / (tn + fp), tn / (tn + fn)
    return {"prevalence": (tp + fn) / n, "accuracy": (tp + tn) / n, "precision": precision, "recall": recall,
            "specificity": specificity, "npv": npv, "fpr": fp / (fp + tn), "fnr": fn / (fn + tp),
            "fdr": fp / (fp + tp), "for": fn / (fn + tn), "f1": 2 * tp / (2 * tp + fp + fn),
            "balanced": (recall + specificity) / 2,
            "mcc": (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))}


R35 = rates(TP_35, FN_35, FP_35, TN_35)
PRECISION_36 = np.diag(SPECIES_36) / SPECIES_36.sum(axis=0)
RECALL_36 = np.diag(SPECIES_36) / SPECIES_36.sum(axis=1)
F1_36 = 2 * PRECISION_36 * RECALL_36 / (PRECISION_36 + RECALL_36)
SUPPORT_36 = SPECIES_36.sum(axis=1)


def screening(n, prevalence, sensitivity, specificity):
    """Expected counts of a screening test (natural frequencies): TP, FN, FP, TN (3.7)."""
    sick = n * prevalence
    healthy = n - sick
    return sensitivity * sick, (1 - sensitivity) * sick, (1 - specificity) * healthy, specificity * healthy


TP_37, FN_37, FP_37, TN_37 = screening(50_000, 0.01, 0.99, 0.99)
BOOK = screening(10_000, 0.01, 0.99, 0.98)        # 3.7 part 2: the book's test (§3.8)'''

PAPER = [
    Paper("3.1", "Fléchettes et aires : probabilités simples et conditionnelles", [
        ("a", "P(A)", "AREA_A / WALL",
         'decimals=2, mistakes={"c\'est l\'aire de A : divise-la par l\'aire du mur": AREA_A}'),
        ("b", "P(B)", "AREA_B / WALL",
         'decimals=2, mistakes={"c\'est l\'aire de B : divise-la par l\'aire du mur": AREA_B}'),
        ("c", "P(A, B)", "AREA_AB / WALL",
         'decimals=2, mistakes={"c\'est P(A) × P(B) : ce produit ne vaut P(A, B) que si A et B sont indépendants ; '
         'mesure plutôt l\'aire de la partie commune": (AREA_A / WALL) * (AREA_B / WALL), '
         '"c\'est l\'aire de la partie commune : divise-la par l\'aire du mur": AREA_AB, '
         '"vérifie les bornes de la partie commune : pour chaque coordonnée, garde le plus grand des deux minimums '
         'et le plus petit des deux maximums": 1 * 1.5 / WALL}'),
        ("d", "P(A | B)", "AREA_AB / AREA_B",
         'decimals=2, mistakes={"c\'est P(B | A) : ici, on sait que la fléchette est dans B, divise par l\'aire de B": '
         'AREA_AB / AREA_A, "c\'est P(A, B) : sachant B, on divise par l\'aire de B, pas par celle du mur": '
         'AREA_AB / WALL}'),
        ("e", "P(B | A), 3 decimals", "AREA_AB / AREA_A",
         'decimals=3, mistakes={"c\'est P(A | B) : ici, on sait que la fléchette est dans A, divise par l\'aire de A": '
         'AREA_AB / AREA_B, "c\'est P(A, B) : sachant A, on divise par l\'aire de A, pas par celle du mur": '
         'AREA_AB / WALL}'),
        ("f", "P(neither A nor B)", "1 - (AREA_A + AREA_B - AREA_AB) / WALL",
         'decimals=2, mistakes={"la partie commune est comptée deux fois dans P(A) + P(B) : '
         'P(A ou B) = P(A) + P(B) − P(A, B)": 1 - (AREA_A + AREA_B) / WALL, '
         '"c\'est P(A ou B) : on demande l\'événement contraire, ni A ni B": (AREA_A + AREA_B - AREA_AB) / WALL, '
         '"(1 − P(A)) × (1 − P(B)) n\'est juste que si A et B sont indépendants, ce que rien ne garantit ici '
         '(tu le vérifieras en j) : passe par P(A ou B)": (1 - AREA_A / WALL) * (1 - AREA_B / WALL)}'),
        ("g", "expected darts in B out of 400", "round(400 * AREA_B / WALL)",
         'mistakes={"c\'est le nombre attendu dans A et B à la fois": round(400 * AREA_AB / WALL), '
         '"c\'est le nombre attendu dans A": round(400 * AREA_A / WALL)}'),
        ("h", "expected darts in A and B out of 400", "round(400 * AREA_AB / WALL)",
         'mistakes={"c\'est le nombre attendu dans B : on demande A et B à la fois": round(400 * AREA_B / WALL)}'),
        ("i", "the student's estimate of P(A | B), 3 decimals", "41 / 76",
         'decimals=3, mistakes={"sachant B : divise par le nombre de fléchettes dans B, pas par toutes": 41 / 400, '
         '"divise les fléchettes dans A et B par celles dans B, pas l\'inverse": 76 / 41}'),
        ("j", "are A and B independent? True or False", "bool(np.isclose(AREA_AB / WALL, (AREA_A / WALL) * (AREA_B / WALL)))",
         'mistakes={"compare P(A, B) au produit P(A) × P(B)": True}'),
    ]),
    Paper("3.2", "Les 20 points : matrice de confusion et quatre mesures", [
        ("a", "TP", "TP_32",
         'mistakes={"ça, c\'est le nombre de faux positifs : un vrai positif est positif ET prédit positif": FP_32, '
         '"c\'est le nombre total de positifs : ne compte que ceux qui sont aussi prédits positifs": 10, '
         '"ça, c\'est le nombre de vrais négatifs : un vrai positif est positif ET prédit positif": TN_32}'),
        ("b", "FP", "FP_32",
         'mistakes={"ça, c\'est le nombre de faux négatifs : un faux positif est un NÉGATIF prédit positif": FN_32}'),
        ("c", "FN", "FN_32",
         'mistakes={"ça, c\'est le nombre de faux positifs : un faux négatif est un POSITIF prédit négatif": FP_32}'),
        ("d", "TN", "TN_32",
         'mistakes={"c\'est le nombre total de négatifs : ne compte que ceux qui sont aussi prédits négatifs": 10, '
         '"ça, c\'est le nombre de vrais positifs : un vrai négatif est négatif ET prédit négatif": TP_32}'),
        ("e", "the confusion matrix as scikit-learn returns it, a list of 2 rows",
         "[[TN_32, FP_32], [FN_32, TP_32]]",
         'mistakes={"c\'est la disposition du livre (TP en haut à gauche) ; scikit-learn trie les étiquettes, '
         '0 puis 1, donc la ligne de la vérité 0 vient d\'abord": [[TP_32, FN_32], [FP_32, TN_32]], '
         '"vérité en LIGNES, prédiction en COLONNES : ta matrice est transposée": [[TN_32, FN_32], [FP_32, TP_32]]}'),
        ("f", "the accuracy, 2 decimals", "(TP_32 + TN_32) / 20",
         'decimals=2, mistakes={"l\'accuracy compte les DEUX sortes de bonnes réponses, TP et TN": TP_32 / 20, '
         '"c\'est le recall : l\'accuracy compte les bonnes réponses, TP + TN, parmi les 20 points": '
         'TP_32 / (TP_32 + FN_32), "c\'est la precision : l\'accuracy compte les bonnes réponses, TP + TN, parmi '
         'les 20 points": TP_32 / (TP_32 + FP_32)}'),
        ("g", "the precision, 3 decimals", "TP_32 / (TP_32 + FP_32)",
         'decimals=3, mistakes={"c\'est le recall : la precision divise par le nombre de prédictions positives, '
         'TP + FP": TP_32 / (TP_32 + FN_32)}'),
        ("h", "the recall, 2 decimals", "TP_32 / (TP_32 + FN_32)",
         'decimals=2, mistakes={"c\'est la precision : le recall divise par le nombre de positifs réels, TP + FN": '
         'TP_32 / (TP_32 + FP_32), "c\'est l\'accuracy : le recall ne regarde que les positifs réels, TP + FN": '
         '(TP_32 + TN_32) / 20}'),
        ("i", "the F1 score, 3 decimals", "2 * TP_32 / (2 * TP_32 + FP_32 + FN_32)",
         'decimals=3, mistakes={"c\'est la moyenne ordinaire de la precision et du recall : '
         'le F1 est leur moyenne HARMONIQUE": (TP_32 / (TP_32 + FP_32) + TP_32 / (TP_32 + FN_32)) / 2, '
         '"deux causes possibles : tu as calculé avec la precision arrondie à 3 décimales, ou tu as tronqué au lieu '
         'd\'arrondir au plus proche ; garde la fraction exacte jusqu\'au bout (ou la forme 2 TP / (2 TP + FP + FN)), '
         'puis arrondis au plus proche": 0.666}'),
    ]),
    Paper("3.3", "Le glacier : jointes, marginales et conditionnelles", [
        ("a", "P(V)", "ICE[0].sum() / ICE.sum()",
         'decimals=2, mistakes={"c\'est un effectif : divise par le nombre de clients": int(ICE[0].sum())}'),
        ("b", "P(C), 3 decimals", "ICE[:, 0].sum() / ICE.sum()",
         'decimals=3, mistakes={"c\'est la probabilité d\'un pot : C est l\'événement « cornet »": '
         'ICE[:, 1].sum() / ICE.sum()}'),
        ("c", "P(V, C)", "ICE[0, 0] / ICE.sum()",
         'decimals=2, mistakes={"c\'est P(V | C) : la probabilité jointe divise par le nombre total de clients": '
         'ICE[0, 0] / ICE[:, 0].sum(), "c\'est P(C | V) : la probabilité jointe divise par le nombre total de clients": '
         'ICE[0, 0] / ICE[0].sum(), "c\'est P(V) × P(C) : ce produit ne vaut P(V, C) que si V et C sont '
         'indépendants ; lis plutôt la case de la table": (ICE[0].sum() / ICE.sum()) * (ICE[:, 0].sum() / ICE.sum())}'),
        ("d", "P(V | C), 3 decimals", "ICE[0, 0] / ICE[:, 0].sum()",
         'decimals=3, mistakes={"c\'est P(C | V) : sachant C (un cornet), divise par le nombre de cornets": '
         'ICE[0, 0] / ICE[0].sum(), "c\'est P(V, C) : sachant C, divise par le nombre de cornets, pas de clients": '
         'ICE[0, 0] / ICE.sum()}'),
        ("e", "P(C | V)", "ICE[0, 0] / ICE[0].sum()",
         'decimals=2, mistakes={"c\'est P(V | C) : sachant V (vanille), divise par le nombre de clients vanille": '
         'ICE[0, 0] / ICE[:, 0].sum(), "c\'est P(V, C) : sachant V, divise par le nombre de clients vanille": '
         'ICE[0, 0] / ICE.sum()}'),
        ("f", "P(chocolate | cup), 3 decimals", "ICE[1, 1] / ICE[:, 1].sum()",
         'decimals=3, mistakes={"c\'est P(pot | chocolat) : sachant « pot », divise par le nombre de pots": '
         'ICE[1, 1] / ICE[1].sum(), "c\'est la probabilité jointe : sachant « pot », divise par le nombre de pots": '
         'ICE[1, 1] / ICE.sum()}'),
        ("g", "are the flavour and the container independent? True or False",
         "bool(np.isclose(ICE[0, 0] / ICE.sum(), (ICE[0].sum() / ICE.sum()) * (ICE[:, 0].sum() / ICE.sum())))",
         'mistakes={"compare P(V, C) au produit P(V) × P(C)": True}'),
        ("h", "vanilla cones to plan for 300 customers", "round(300 * ICE[0, 0] / ICE.sum())",
         'mistakes={"c\'est le nombre d\'hier, pour 150 clients : demain, ils seront 300": int(ICE[0, 0])}'),
    ]),
    Paper("3.5", "Toutes les mesures du tableau récapitulatif", [
        ("a", "the prevalence, 3 decimals", 'R35["prevalence"]',
         'decimals=3, mistakes={"c\'est la part de prédictions positives : la prévalence compte les positifs réels '
         '(la vérité), TP + FN": (TP_35 + FP_35) / 1000}'),
        ("b", "the accuracy, 3 decimals", 'R35["accuracy"]',
         'decimals=3, mistakes={"l\'accuracy compte les DEUX sortes de bonnes réponses, TP et TN": TP_35 / 1000}'),
        ("c", "the precision, 3 decimals", 'R35["precision"]',
         'decimals=3, mistakes={"c\'est le recall : la precision divise par les prédictions positives, TP + FP": '
         'R35["recall"]}'),
        ("d", "the recall, 3 decimals", 'R35["recall"]',
         'decimals=3, mistakes={"c\'est la precision : le recall divise par les positifs réels, TP + FN": '
         'R35["precision"]}'),
        ("e", "the specificity, 3 decimals", 'R35["specificity"]',
         'decimals=3, mistakes={"c\'est la NPV : la spécificité divise par le nombre de négatifs RÉELS, TN + FP": '
         'R35["npv"], "c\'est le FPR, son complément à 1": R35["fpr"]}'),
        ("f", "the NPV, 3 decimals", 'R35["npv"]',
         'decimals=3, mistakes={"c\'est la spécificité : la NPV divise par le nombre de prédictions négatives, TN + FN": '
         'R35["specificity"], "c\'est le FOR, son complément à 1": R35["for"]}'),
        ("g", "the FPR, 3 decimals", 'R35["fpr"]',
         'decimals=3, mistakes={"c\'est le FDR : le FPR divise par le nombre de négatifs réels, FP + TN": R35["fdr"], '
         '"c\'est la spécificité, son complément à 1": R35["specificity"], '
         '"tu as divisé par les 1 000 transactions : le FPR divise par les négatifs réels, FP + TN": FP_35 / 1000}'),
        ("h", "the FNR, 3 decimals", 'R35["fnr"]',
         'decimals=3, mistakes={"c\'est le FOR : le FNR divise par le nombre de positifs réels, FN + TP": R35["for"], '
         '"c\'est le recall, son complément à 1": R35["recall"]}'),
        ("i", "the FDR, 3 decimals", 'R35["fdr"]',
         'decimals=3, mistakes={"c\'est le FPR : le FDR divise par le nombre de prédictions positives, FP + TP": '
         'R35["fpr"], "c\'est la precision, son complément à 1": R35["precision"]}'),
        ("j", "the FOR (false omission rate), 3 decimals", 'R35["for"]',
         'decimals=3, mistakes={"c\'est le FNR : le FOR divise par le nombre de prédictions négatives, FN + TN": '
         'R35["fnr"], "c\'est la NPV, son complément à 1": R35["npv"], '
         '"tu as divisé par les 1 000 transactions : le FOR divise par les prédictions négatives, FN + TN": '
         'FN_35 / 1000}'),
        ("k", "the F1 score, 3 decimals", 'R35["f1"]',
         'decimals=3, mistakes={"c\'est la moyenne ordinaire de la precision et du recall : le F1 est leur moyenne '
         'HARMONIQUE": (R35["precision"] + R35["recall"]) / 2}'),
        ("l", "the balanced accuracy, 3 decimals", 'R35["balanced"]',
         'decimals=3, mistakes={"c\'est l\'accuracy : la balanced accuracy est la moyenne du recall et de la '
         'spécificité": R35["accuracy"], "tu as fait la moyenne avec une spécificité déjà arrondie : garde la '
         'fraction exacte jusqu\'au bout, puis arrondis": 0.869}'),
        ("m", "the MCC, 3 decimals", 'R35["mcc"]',
         'decimals=3, mistakes={"le dénominateur est la RACINE CARRÉE du produit des quatre sommes": '
         '(TP_35 * TN_35 - FP_35 * FN_35) / ((TP_35 + FP_35) * (TP_35 + FN_35) * (TN_35 + FP_35) * (TN_35 + FN_35))}'),
    ]),
    Paper("3.6", "Trois espèces : moyennes macro, micro et pondérée", [
        ("a", "the support of each species, a list", "SUPPORT_36.tolist()",
         'mistakes={"ce sont les nombres de PRÉDICTIONS de chaque espèce (totaux des colonnes) : le support compte '
         'les vrais manchots de chaque espèce (totaux des lignes)": SPECIES_36.sum(axis=0).tolist()}'),
        ("b", "the precision of each species, a list (3 decimals)", "PRECISION_36.tolist()",
         'decimals=3, mistakes={"ce sont les recalls : la precision d\'une espèce divise par le total de sa COLONNE '
         '(les prédictions)": RECALL_36.tolist()}'),
        ("c", "the recall of each species, a list (3 decimals)", "RECALL_36.tolist()",
         'decimals=3, mistakes={"ce sont les precisions : le recall d\'une espèce divise par le total de sa LIGNE '
         '(les vrais manchots)": PRECISION_36.tolist()}'),
        ("d", "the F1 score of each species, a list (3 decimals)", "F1_36.tolist()",
         'decimals=3, mistakes={"ce sont des moyennes ordinaires : le F1 est la moyenne HARMONIQUE de la precision '
         'et du recall": ((PRECISION_36 + RECALL_36) / 2).tolist()}'),
        ("e", "the macro precision, 3 decimals", "PRECISION_36.mean()",
         'decimals=3, mistakes={"c\'est la precision pondérée : la moyenne macro est la moyenne SIMPLE des trois '
         'espèces": np.average(PRECISION_36, weights=SUPPORT_36), "c\'est la precision micro, qui vaut l\'accuracy": '
         'np.trace(SPECIES_36) / SPECIES_36.sum()}'),
        ("f", "the macro F1 score, 3 decimals", "F1_36.mean()",
         'decimals=3, mistakes={"c\'est le F1 pondéré : la moyenne macro est la moyenne SIMPLE des trois espèces": '
         'np.average(F1_36, weights=SUPPORT_36), "c\'est le F1 micro, qui vaut l\'accuracy": '
         'np.trace(SPECIES_36) / SPECIES_36.sum(), "c\'est le F1 de la precision macro et du recall macro : '
         'fais plutôt la moyenne des trois F1": 2 * PRECISION_36.mean() * RECALL_36.mean() / '
         '(PRECISION_36.mean() + RECALL_36.mean())}'),
        ("g", "the weighted F1 score, 3 decimals", "np.average(F1_36, weights=SUPPORT_36)",
         'decimals=3, mistakes={"c\'est le F1 macro : la moyenne pondérée multiplie le F1 de chaque espèce par son '
         'support": F1_36.mean(), "c\'est une moyenne pondérée par les PRÉDICTIONS : le poids d\'une espèce est son '
         'support (vrais manchots)": np.average(F1_36, weights=SPECIES_36.sum(axis=0)), '
         '"tu as fait la moyenne des F1 arrondis de d : repars des fractions exactes des trois F1 et n\'arrondis '
         'qu\'à la fin": 0.869, "c\'est le F1 micro, qui vaut l\'accuracy : le F1 pondéré fait la moyenne des F1 des '
         'espèces, pondérée par leur support": np.trace(SPECIES_36) / SPECIES_36.sum()}'),
        ("h", "the micro F1 score, 3 decimals", "np.trace(SPECIES_36) / SPECIES_36.sum()",
         'decimals=3, mistakes={"c\'est le F1 macro : le F1 micro additionne d\'abord les TP, FP et FN des trois '
         'espèces": F1_36.mean(), "c\'est le F1 pondéré : le F1 micro additionne d\'abord les TP, FP et FN des trois '
         'espèces": np.average(F1_36, weights=SUPPORT_36)}'),
    ]),
    Paper("3.7", "Le test « fiable à 99 % » dans une ville à 1 % de malades", [
        ("a", "the number of sick people", "round(50_000 * 0.01)", ""),
        ("b", "TP", "round(TP_37)",
         'mistakes={"c\'est le nombre de malades : le test n\'en détecte que 99 %": round(50_000 * 0.01), '
         '"c\'est 99 % de toute la ville (et aussi le nombre de personnes saines) : les vrais positifs sont les '
         'MALADES que le test détecte, 99 % des seuls malades": round(0.99 * 50_000)}'),
        ("c", "FN", "round(FN_37)",
         'mistakes={"c\'est le nombre de faux positifs : un faux négatif est un MALADE déclaré négatif": round(FP_37), '
         '"c\'est 1 % de toute la ville (ou le nombre de malades) : les faux négatifs sont 1 % des seuls MALADES": '
         'round(0.01 * 50_000)}'),
        ("d", "FP", "round(FP_37)",
         'mistakes={"le taux de 1 % de faux positifs s\'applique aux 49 500 personnes SAINES, pas à toute la ville": '
         'round(0.01 * 50_000), "c\'est le nombre de faux négatifs : un faux positif est une personne SAINE '
         'déclarée positive": round(FN_37)}'),
        ("e", "TN", "round(TN_37)",
         'mistakes={"les vrais négatifs sont 99 % des personnes SAINES (49 500), pas de toute la ville": '
         'round(0.99 * 50_000)}'),
        ("f", "the precision P(sick | positive), 2 decimals", "TP_37 / (TP_37 + FP_37)",
         'decimals=2, mistakes={"c\'est la sensibilité P(positif | malade) : la precision est P(malade | positif), '
         'divise TP par le nombre de positifs": 0.99}'),
        ("g", "the NPV P(healthy | negative), 4 decimals", "TN_37 / (TN_37 + FN_37)",
         'decimals=4, mistakes={"c\'est la spécificité P(négatif | sain) : la NPV divise TN par le nombre de '
         'résultats négatifs": 0.99}'),
        ("h", "the accuracy, 2 decimals", "(TP_37 + TN_37) / 50_000",
         'decimals=2, mistakes={"c\'est la precision : l\'accuracy compte les bonnes réponses parmi TOUTES les '
         'personnes": TP_37 / (TP_37 + FP_37), "c\'est la NPV : l\'accuracy compte les bonnes réponses, TP + TN, '
         'parmi les 50 000 habitants": TN_37 / (TN_37 + FN_37)}'),
        ("i", "the precision with a prevalence of 0.2 %, 3 decimals",
         "screening(50_000, 0.002, 0.99, 0.99)[0] / (screening(50_000, 0.002, 0.99, 0.99)[0] "
         "+ screening(50_000, 0.002, 0.99, 0.99)[2])",
         'decimals=3, mistakes={"c\'est la sensibilité : la precision dépend aussi de la prévalence ; refais l\'arbre '
         'avec 100 malades": 0.99, "deux causes possibles : les faux positifs sont 1 % des personnes SAINES de cette ville, '
         'pas de tous ses habitants ; ou bien tu as tronqué au lieu d\'arrondir au plus proche": 99 / (99 + 500)}'),
        ("j", "the precision with a prevalence of 10 %, 3 decimals",
         "screening(50_000, 0.10, 0.99, 0.99)[0] / (screening(50_000, 0.10, 0.99, 0.99)[0] "
         "+ screening(50_000, 0.10, 0.99, 0.99)[2])",
         'decimals=3, mistakes={"c\'est la sensibilité : la precision dépend aussi de la prévalence ; refais l\'arbre '
         'avec 5 000 malades": 0.99, "les faux positifs sont 1 % des personnes SAINES de cette ville, pas de tous '
         'ses habitants": 4950 / (4950 + 500)}'),
        ("k", "the book's specificity TN / (TN + FP), 2 decimals", "BOOK[3] / (BOOK[3] + BOOK[2])",
         'decimals=2, mistakes={"c\'est la NPV : la spécificité divise TN par le nombre de personnes SAINES, '
         'TN + FP": BOOK[3] / (BOOK[3] + BOOK[1]), "ce test n\'est pas celui de la partie 1 : calcule TN / (TN + FP) '
         'avec les comptages du livre": 0.99, "tu as divisé TN par toute la ville : la spécificité divise par les '
         'personnes SAINES, TN + FP": BOOK[3] / 10_000}'),
        ("l", "the book's NPV TN / (TN + FN), 4 decimals", "BOOK[3] / (BOOK[3] + BOOK[1])",
         'decimals=4, mistakes={"c\'est la spécificité : la NPV divise TN par le nombre de résultats NÉGATIFS, '
         'TN + FN": BOOK[3] / (BOOK[3] + BOOK[2])}'),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes exercices papier (✏️ 3.1, 3.2, 3.3, 3.5, 3.6 et 3.7)\n\n"
               "Fais d'abord les exercices ✏️ de `02_exercices.md` **sur papier**, dans ta copie de "
               "`06_mes_reponses.md`. Reporte ensuite chaque réponse ici : **la valeur** que tu as trouvée "
               "(par exemple `42`, `0.125`, `[1, 2, 3]` ou `True`), pas l'expression Python, sinon tu ne vérifies "
               "rien. Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore "
               "remplies affichent ⏳. "
               "Les exercices ∂ 3.4 et 3.8 se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Part A: probabilities with areas and tables (3.12 to 3.14)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools and data for the notebook exercises (parts A to D)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as skm

penguins = wb.datasets.load_penguins()                    # the 344 penguins
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
measured = penguins.dropna(subset=MEASURES).reset_index(drop=True)   # the 342 penguins with their 4 measures
species = measured["species"].to_numpy()                  # their true species
flipper = measured["flipper_length_mm"].to_numpy()        # in mm (whole numbers)
is_gentoo = (species == "Gentoo").astype(int)             # 1 = Gentoo, 0 = another species (3.20, 3.24)


def verdict(ex_id, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message."""
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")


def fr(value, decimals=2):
    """A number written the French way, for the messages: fr(0.25) -> '0,25'."""
    return f"{value:.{decimals}f}".replace(".", ",")


def run_metrics_tests(keyword, impl="learner"):
    """Run the tests of mylearn.metrics selected by `keyword` (on YOUR code by default)."""
    command = [sys.executable, "-m", "pytest", "tests/test_ch03_metrics.py", "-k", keyword, "-q",
               "-p", "no:cacheprovider", "--color=no", "-rf", "--tb=no"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "1000"})   # long lines: the reason of each failure
    lines = result.stdout.strip().splitlines()
    failed = [line for line in lines if line.startswith("FAILED ")]
    for line in failed[:8]:                                  # the test, then the reason of its failure
        name, _, reason = line.removeprefix("FAILED tests/test_ch03_metrics.py::").partition(" - ")
        print(f"❌ {name}\n   {reason[:400]}")
    if len(failed) > 8:
        print(f"   ... and {len(failed) - 8} other failed test(s)")
    print("pytest:", lines[-1] if lines else result.stderr.strip()[-300:])


def error_name(func, *args, **kwargs):
    """Name of the exception raised by func(*args, **kwargs), or "no error"."""
    try:
        func(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as error:  # noqa: BLE001 - we want the name of whatever is raised
        return type(error).__name__
    return "no error"


print(f"{len(penguins)} penguins, {len(measured)} with their 4 measures")'''

MYLEARN_HOWTO = ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre `mon_travail/mylearn/metrics.py` "
                 "(créé par `python tools/start_chapter.py 3`), lis la docstring de chaque fonction, remplace les "
                 "`raise NotImplementedError(...)` par ton code et **enregistre**. NumPy est permis (`np.asarray`, "
                 "`np.unique`, `np.concatenate`, `np.sum`, `np.argsort`, `np.cumsum`…), scikit-learn non : "
                 "`sklearn.metrics` est l'**oracle** des tests. Écris une petite fonction d'aide, par exemple "
                 "`_check_pair(y_true, y_pred)`, qui convertit les deux entrées (`np.asarray`), lève une `ValueError` "
                 "si leurs longueurs diffèrent ou si elles sont vides, et les renvoie : toutes les fonctions de mesure "
                 "l'appelleront. Une mesure se renvoie en `float` Python (`float(...)` ; les tests acceptent aussi un "
                 "`np.float64`). La cellule de vérification recharge ta librairie, vérifie quelques valeurs, puis lance "
                 "les tests de tes fonctions ; `python -m pytest tests/test_ch03_metrics.py -q` les lance tous dans un "
                 "terminal.")

MYLEARN_SHORT = ("> **mylearn** : dans `mon_travail/mylearn/metrics.py`, mêmes règles qu'en 3.15 (NumPy permis, "
                 "scikit-learn non). Enregistre, puis relance la cellule de vérification.")

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="3")   # reload your saved file\n'

DISCS_13 = r'''rng_13 = np.random.default_rng(13)
darts_13 = 4 * rng_13.random((100_000, 2))                          # the 4 m x 4 m wall
in_a = ((darts_13 - [1.5, 2.0]) ** 2).sum(axis=1) <= 0.5 ** 2      # disc A: centre (1.5, 2), radius 0.5
in_b = ((darts_13 - [2.2, 2.0]) ** 2).sum(axis=1) <= 1.0 ** 2      # disc B: centre (2.2, 2), radius 1
p_a_given_b = np.sum(in_a & in_b) / np.sum(in_b)
p_b_given_a = np.sum(in_a & in_b) / np.sum(in_a)
print(f"darts in A: {in_a.sum()} · in B: {in_b.sum()} · in A and B: {np.sum(in_a & in_b)}")
print(f"P(A | B) ≈ {p_a_given_b:.3f}   P(B | A) ≈ {p_b_given_a:.3f}")

shown = slice(0, 4000)                                              # 4 000 darts are enough for the picture
colour = np.where(in_a & in_b, "C2", np.where(in_a, "C0", np.where(in_b, "C1", "0.8")))[shown]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(*darts_13[shown].T, s=3, c=colour)
ax.add_patch(plt.Circle((1.5, 2.0), 0.5, fill=False, color="C0", lw=2))
ax.add_patch(plt.Circle((2.2, 2.0), 1.0, fill=False, color="C1", lw=2))
ax.text(1.15, 2.6, "A", color="C0", fontsize=14)
ax.text(2.9, 2.9, "B", color="C1", fontsize=14)
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_aspect("equal")
ax.set_title("4 000 of the 100 000 darts")
plt.show()'''

PART_A = Part("A", "Probabilités par les aires et par les tables",
              "Fiche §3.2 à §3.6. Tu estimes une aire en lançant des fléchettes simulées, tu vérifies quand "
              "$P(A \\mid B)$ et $P(B \\mid A)$ diffèrent, puis tu lis des probabilités jointes, marginales et "
              "conditionnelles dans une table de contingence des manchots. La cellule ci-dessous charge les données "
              "et les outils de tout le notebook : exécute-la d'abord.",
              given=PART_A_GIVEN, exercises=[
    Ex("3.12", "🔬", 1, 15, "Dix mille fléchettes : estimer des aires (et π)",
       "estimer une aire en comptant des fléchettes simulées, et mesurer comment l'erreur diminue avec le nombre "
       "de lancers.",
       "ch. 2 (loi uniforme, graine, écart-type, vitesse en $1/\\sqrt{n}$) · fiche §3.2, §3.3 · livre §3.2, §3.3",
       thread="synthétique", tracks="M, C",
       body=r"""Le mur est un carré de 2 m de côté, avec un disque de rayon 1 m peint en son centre. Une fléchette tombe en un point $(x, y)$ tiré uniformément sur le mur (fiche §3.2) ; elle touche le disque si $(x - 1)^2 + (y - 1)^2 \le 1$. D'après la fiche, la probabilité de toucher le disque est le rapport de son aire, $\pi$, à celle du mur, 4 : compter les fléchettes dans le disque **estime** donc $\pi$.

Pour que tout le monde obtienne les mêmes nombres, tire les points **exactement** ainsi : `points = 2 * rng.random((n, 2))` (une ligne par fléchette, colonnes $x$ et $y$), avec le générateur `rng` indiqué.

a) `pi_hat_12` : avec `rng = np.random.default_rng(3)` et $n = 10\,000$ fléchettes, l'estimation $\hat{\pi} = 4 \times$ (part des fléchettes tombées dans le disque) (3 décimales).

Écris ensuite `estimate_pi(n, rng)`, qui fait la même chose pour $n$ fléchettes et renvoie un `float`. La vérification contrôle qu'elle redonne la bonne réponse de a) avec la graine 3, et qu'avec un million de fléchettes elle tombe à moins de 0,01 de $\pi$.

b) `spread_12` : crée **un seul** générateur, `rng = np.random.default_rng(12)`, puis calcule 200 estimations de 10 000 fléchettes chacune, l'une après l'autre avec ce même générateur (`[estimate_pi(10_000, rng) for _ in range(200)]`). Donne leur écart-type (`np.std`, 3 décimales) : c'est l'erreur typique d'une estimation.
c) `factor_12` : par combien faut-il multiplier le nombre de fléchettes pour diviser cette erreur typique par 10 ? (un entier ; souviens-toi de la vitesse en $1/\sqrt{n}$ du ch. 2). Une fois c) rempli, la vérification le mesure sur d'autres nombres de fléchettes.""",
       todo=r'''pi_hat_12 = ...   # a)


def estimate_pi(n, rng):
    """Estimate pi with n darts thrown uniformly on the 2 m x 2 m wall: 4 x (share of darts in the disc)."""
    raise NotImplementedError("estimate_pi() is not written yet")


spread_12 = ...   # b)
factor_12 = ...   # c) an integer''',
       check=r'''wb.check("3.12a", pi_hat_12)
wb.check("3.12b", spread_12)
wb.check("3.12c", factor_12)
with wb.attempt("3.12"):
    first_12 = estimate_pi(10_000, np.random.default_rng(3))
    verdict("3.12", isinstance(first_12, float) and bool(wb.check("3.12a", first_12, quiet=True)),
            "estimate_pi redonne la bonne réponse de a) avec la graine 3, sous forme de float.",
            "estimate_pi(10_000, np.random.default_rng(3)) doit redonner la bonne réponse de a), en float : tire les "
            "points exactement comme l'énoncé, et renvoie 4 fois la part des fléchettes dans le disque.")
    big_12 = estimate_pi(1_000_000, np.random.default_rng(0))
    verdict("3.12", abs(big_12 - np.pi) < 0.01, f"avec un million de fléchettes : {fr(big_12, 4)}, à moins de 0,01 de π.",
            f"avec un million de fléchettes, ton estimation vaut {fr(big_12, 4)} : trop loin de π.")
    if factor_12 is not ...:
        rng_demo = np.random.default_rng(0)
        for n_demo in [25, 250, 2_500]:
            spread_demo = np.std([estimate_pi(n_demo, rng_demo) for _ in range(500)])
            print(f"   {n_demo:>6} darts: typical error of an estimate ≈ {spread_demo:.4f}")''',
       solution=r'''rng = np.random.default_rng(3)
points = 2 * rng.random((10_000, 2))
in_disc = ((points - 1) ** 2).sum(axis=1) <= 1
pi_hat_12 = 4 * in_disc.mean()


def estimate_pi(n, rng):
    """Estimate pi with n darts thrown uniformly on the 2 m x 2 m wall: 4 x (share of darts in the disc)."""
    points = 2 * rng.random((n, 2))
    return float(4 * np.mean(((points - 1) ** 2).sum(axis=1) <= 1))


rng = np.random.default_rng(12)
estimates_12 = [estimate_pi(10_000, rng) for _ in range(200)]
spread_12 = float(np.std(estimates_12))
factor_12 = 100
print(pi_hat_12, spread_12, "mean of the 200 estimates:", np.mean(estimates_12))
rng_demo = np.random.default_rng(0)
for n_demo in [25, 250, 2_500]:
    spread_demo = np.std([estimate_pi(n_demo, rng_demo) for _ in range(500)])
    print(f"   {n_demo:>6} darts: typical error of an estimate ≈ {spread_demo:.4f}")''',
       record=r'''wb.record("3.12a", pi_hat_12, decimals=3, mistakes={"c'est la part des fléchettes dans le disque : multiplie-la par 4, le rapport de l'aire du mur (4) à celle d'un disque de rayon 1 (π)": in_disc.mean(),
                                                    "tire les points exactement comme l'énoncé, 2 * rng.random((10_000, 2)), une ligne par fléchette : un autre découpage des nombres (deux appels, ou la forme (2, n)) donne d'autres fléchettes": 4 * np.mean((2 * np.random.default_rng(3).random(10_000) - 1) ** 2 + (2 * np.random.default_rng(3).random(20_000)[10_000:] - 1) ** 2 <= 1)})
wb.record("3.12b", spread_12, decimals=3, mistakes={"ta réponse s'arrondit à 0 : soit tes 200 estimations sont identiques (crée le générateur UNE fois, avant la boucle, et passe le même à chaque appel, ch. 2), soit tu as donné leur variance (on demande l'écart-type, sa racine)": 0.0})
wb.record("3.12c", factor_12, mistakes={"10 fois plus de fléchettes ne divisent pas l'erreur typique par 10 : relis à quelle vitesse elle diminue quand n augmente (ch. 2)": 10})''',
       note="Une estimation par comptage est une variable aléatoire : elle fluctue autour de la vraie valeur, avec une "
            "erreur typique proportionnelle à $1/\\sqrt{n}$. Diviser l'erreur par 10 coûte 100 fois plus de lancers : "
            "c'est la même loi que l'erreur typique d'une moyenne au ch. 2, et celle d'une accuracy mesurée sur un jeu de "
            "test (un jeu de test de 100 exemples donne une accuracy à ± 0,05 près environ). Estimer une aire par des "
            "tirages au hasard s'appelle une méthode de **Monte-Carlo** : elle marche pour des formes dont l'aire n'a "
            "pas de formule simple."),

    Ex("3.13", "🔮", 1, 10, "Deux disques : P(A|B) = P(B|A) ?",
       "prévoir quand $P(A \\mid B)$ et $P(B \\mid A)$ sont égales, puis le vérifier avec des fléchettes.",
       "Ex 3.12, Ex 3.1 · fiche §3.4 · livre §3.4", thread="synthétique", tracks="M, C", hypothesis=True,
       body=r"""Un mur carré de 4 m de côté porte deux disques qui se chevauchent en partie : $A$, de rayon 0,5 m, et $B$, de rayon 1 m. **Sans rien exécuter**, prévois :
a) `prediction_3_13a` : laquelle est la plus grande, $P(A \mid B)$ ou $P(B \mid A)$ ? Réponds `1` si c'est $P(A \mid B)$, `-1` si c'est $P(B \mid A)$, `0` si elles sont égales ;
b) `prediction_3_13b` : même question si l'on agrandit $A$ jusqu'au rayon de $B$, les deux disques se chevauchant toujours en partie ;
c) `prediction_3_13c` : dans la situation de départ, le rapport $\frac{P(A \mid B)}{P(B \mid A)}$ (2 décimales). Écris les deux probabilités comme des rapports d'aires (fiche §3.4) : l'aire de la partie commune, difficile à calculer, n'est pas nécessaire.

Écris ton hypothèse (cellule 📝), puis tes trois prédictions ; la vérification ne regarde que tes prédictions. Ensuite seulement, exécute l'**Expérience** : 100 000 fléchettes.""",
       todo=r'''prediction_3_13a = ...   # 1, -1 or 0
prediction_3_13b = ...   # 1, -1 or 0
prediction_3_13c = ...   # a number''',
       check=r'''wb.check("3.13a", prediction_3_13a)
wb.check("3.13b", prediction_3_13b)
wb.check("3.13c", prediction_3_13c)''',
       solution=r'''prediction_3_13a, prediction_3_13b, prediction_3_13c = -1, 0, 0.25''',
       record=r'''wb.record("3.13a", prediction_3_13a, mistakes={"les deux probabilités ont le même numérateur, l'aire de la partie commune : compare leurs dénominateurs": 1,
                                                "elles ne sont égales que si leurs dénominateurs le sont : A et B ont-ils la même aire ?": 0})
wb.record("3.13b", prediction_3_13b, mistakes={"deux disques de même rayon ont la même aire : les deux probabilités ont alors le même numérateur ET le même dénominateur": 1,
                                                "même numérateur (la partie commune) et, maintenant, même dénominateur : l'aire de A est devenue celle de B": -1})
wb.record("3.13c", prediction_3_13c, decimals=2, mistakes={"c'est le rapport des rayons : une aire varie comme le CARRÉ du rayon": 0.5,
                                                            "c'est l'inverse : P(B | A) / P(A | B)": 4})''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", guarded(DISCS_13, ["prediction_3_13a", "prediction_3_13b", "prediction_3_13c"],
                               "⏳ Ex 3.13 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "d) `ratio_13` : le rapport mesuré $\\frac{P(A \\mid B)}{P(B \\mid A)}$, calculé avec les "
                     "variables de l'expérience (2 décimales). Compare-le à ta prédiction c. Dans tes notes : ce "
                     "rapport d'estimations est exactement un rapport de deux comptages de fléchettes ; lesquels ?"),
              ("todo", "ratio_13 = ...   # d)"),
              ("check", 'wb.check("3.13d", ratio_13)'),
              ("solution", "ratio_13 = p_a_given_b / p_b_given_a\nprint(ratio_13, in_a.sum() / in_b.sum())"),
              ("record", 'wb.record("3.13d", ratio_13, decimals=2, mistakes={"c\'est P(B | A) / P(A | B) : inverse le rapport": 1 / ratio_13})')],
       note="$P(A \\mid B) = \\frac{\\text{aire}(A \\cap B)}{\\text{aire}(B)}$ et $P(B \\mid A) = "
            "\\frac{\\text{aire}(A \\cap B)}{\\text{aire}(A)}$ : même numérateur, donc le rapport vaut "
            "$\\frac{\\text{aire}(A)}{\\text{aire}(B)} = \\frac{P(A)}{P(B)} = \\left(\\frac{0{,}5}{1}\\right)^2$, quelle "
            "que soit la partie commune. Avec des fléchettes, le rapport des deux estimations vaut exactement "
            "(fléchettes dans A) / (fléchettes dans B) : le nombre de fléchettes dans la partie commune se simplifie. "
            "Les deux conditionnelles sont égales seulement si $P(A) = P(B)$ (ou si les disques ne se touchent pas : "
            "0 = 0). C'est la règle du produit, $P(A \\mid B)\\,P(B) = P(B \\mid A)\\,P(A)$, que le ch. 4 transforme "
            "en règle de Bayes."),

    Ex("3.14", "📦", 2, 20, "Penguins : espèce × île avec pd.crosstab",
       "lire des probabilités jointes, marginales et conditionnelles dans une table de contingence, et tester "
       "l'indépendance de deux variables.",
       "Ex 3.3, rappel 3.R3 · fiche §3.4 à §3.6 (🧮 Rappel outil : `pd.crosstab`)", thread="Penguins", tracks="C",
       body=r"""La fiche a croisé le sexe et l'espèce des manchots (§3.6). Fais de même avec l'**espèce** et l'**île**, pour les 344 manchots de `penguins` (aucune de ces deux colonnes ne manque).

a) `table_14` : la table de contingence (`pd.crosstab`), espèces en lignes, îles en colonnes, **sans** les marges ;
b) `p_gentoo_given_biscoe` : $P(\text{Gentoo} \mid \text{Biscoe})$ (3 décimales) ;
c) `p_dream_given_chinstrap` : $P(\text{Dream} \mid \text{Chinstrap})$ (2 décimales) ;
d) `p_adelie_and_dream` : la probabilité jointe $P(\text{Adelie}, \text{Dream})$ (3 décimales) ;
e) `p_adelie_given_dream` : $P(\text{Adelie} \mid \text{Dream})$ (3 décimales) ;
f) `independent_14` : l'espèce et l'île sont-elles indépendantes ? (`True` ou `False` ; justifie-le dans tes notes par un calcul, fiche §3.6) ;
g) `normalize_14` : la valeur de l'argument `normalize` de `pd.crosstab` qui donne, dans chaque **ligne** de la table, $P(\text{île} \mid \text{espèce})$ (une chaîne).

Calcule b à e à partir de la table (`.loc`, sommes de lignes ou de colonnes), pas en recopiant des nombres à la main : c'est ainsi qu'on évite les fautes de frappe.""",
       todo=r'''table_14 = ...                  # a) species in rows, islands in columns
p_gentoo_given_biscoe = ...     # b)
p_dream_given_chinstrap = ...   # c)
p_adelie_and_dream = ...        # d)
p_adelie_given_dream = ...      # e)
independent_14 = ...            # f) True or False
normalize_14 = ...              # g) a string''',
       check=r'''for letter, answer in zip("abcdefg", [table_14, p_gentoo_given_biscoe, p_dream_given_chinstrap, p_adelie_and_dream,
                                     p_adelie_given_dream, independent_14, normalize_14]):
    wb.check(f"3.14{letter}", answer)''',
       solution=r'''table_14 = pd.crosstab(penguins["species"], penguins["island"])
print(pd.crosstab(penguins["species"], penguins["island"], margins=True))
n_14 = table_14.to_numpy().sum()
p_gentoo_given_biscoe = table_14.loc["Gentoo", "Biscoe"] / table_14["Biscoe"].sum()
p_dream_given_chinstrap = table_14.loc["Chinstrap", "Dream"] / table_14.loc["Chinstrap"].sum()
p_adelie_and_dream = table_14.loc["Adelie", "Dream"] / n_14
p_adelie_given_dream = table_14.loc["Adelie", "Dream"] / table_14["Dream"].sum()
joint_14 = table_14.to_numpy() / n_14                                               # P(species, island)
product_14 = np.outer(table_14.sum(axis=1), table_14.sum(axis=0)) / n_14 ** 2      # P(species) x P(island)
print((joint_14 - product_14).round(3))
independent_14 = bool(np.allclose(joint_14, product_14, atol=0.01))
normalize_14 = "index"
print(pd.crosstab(penguins["species"], penguins["island"], normalize=normalize_14).round(3))''',
       record=r'''wb.record("3.14a", table_14)
table_measured_14 = pd.crosstab(measured["species"], measured["island"])   # the classic slip: 342 penguins
MEASURED_14 = "tu as utilisé les 342 manchots de `measured` : la table porte sur les 344 manchots de `penguins`"
wb.record("3.14b", p_gentoo_given_biscoe, decimals=3, mistakes={MEASURED_14: table_measured_14.loc["Gentoo", "Biscoe"] / table_measured_14["Biscoe"].sum(),
                                                                 "c'est P(Biscoe | Gentoo) : sachant Biscoe, divise par le nombre de manchots de Biscoe (total de la colonne)": table_14.loc["Gentoo", "Biscoe"] / table_14.loc["Gentoo"].sum(),
                                                                 "c'est la probabilité jointe : sachant Biscoe, divise par le total de la colonne Biscoe, pas par tous les manchots": table_14.loc["Gentoo", "Biscoe"] / n_14})
wb.record("3.14c", p_dream_given_chinstrap, decimals=2, mistakes={"c'est P(Chinstrap | Dream) : sachant Chinstrap, divise par le nombre de Chinstrap (total de la ligne)": table_14.loc["Chinstrap", "Dream"] / table_14["Dream"].sum()})
wb.record("3.14d", p_adelie_and_dream, decimals=3, mistakes={MEASURED_14: table_measured_14.loc["Adelie", "Dream"] / table_measured_14.to_numpy().sum(),
                                                              "c'est P(Adelie | Dream), une probabilité conditionnelle : la jointe divise la case par le nombre TOTAL de manchots": p_adelie_given_dream,
                                                              "c'est P(Dream | Adelie), une probabilité conditionnelle : la jointe divise la case par le nombre TOTAL de manchots": table_14.loc["Adelie", "Dream"] / table_14.loc["Adelie"].sum(),
                                                              "c'est P(Adelie) × P(Dream) : ce produit ne vaut la jointe que si les deux variables sont indépendantes ; lis plutôt la case": table_14.loc["Adelie"].sum() * table_14["Dream"].sum() / n_14 ** 2})
wb.record("3.14e", p_adelie_given_dream, decimals=3, mistakes={"c'est P(Dream | Adelie) : sachant Dream, divise par le nombre de manchots de Dream (total de la colonne)": table_14.loc["Adelie", "Dream"] / table_14.loc["Adelie"].sum(),
                                                                "c'est la probabilité jointe : sachant Dream, divise par le total de la colonne Dream": p_adelie_and_dream})
wb.record("3.14f", independent_14, mistakes={"compare, case par case, la probabilité jointe au produit des deux marginales : une seule case très différente suffit (regarde les zéros de la table)": True})
wb.record("3.14g", normalize_14, mistakes={"avec \"columns\", chaque COLONNE somme à 1 : on obtient P(espèce | île)": "columns",
                                            "avec \"all\", toute la table somme à 1 : on obtient les probabilités jointes": "all"})''',
       note="Les deux variables sont très loin de l'indépendance : les Chinstrap ne vivent (dans ces données) que sur "
            "Dream, les Gentoo que sur Biscoe, et seuls les Adélie sont sur les trois îles. Une case nulle suffit "
            "à le voir : si les variables étaient indépendantes, $P(\\text{Gentoo}, \\text{Dream})$ vaudrait "
            "$P(\\text{Gentoo})\\,P(\\text{Dream}) > 0$. Conséquence pour le ML : l'île est une feature très "
            "informative… mais dangereuse, car elle décrit l'échantillonnage (où l'équipe a compté) plutôt que les "
            "manchots eux-mêmes. Un modèle qui s'en servirait se tromperait sur des Gentoo observés ailleurs."),
])

# ---------------------------------------------------------------------------
# Part B: the confusion matrix and the binary measures in mylearn.metrics (3.15 to 3.19)
# ---------------------------------------------------------------------------
PART_B_GIVEN = r'''def expert_rule(bill_length, bill_depth, flipper_length, body_mass):
    """The biologist's three rules of exercise 1.15: returns "Gentoo", "Chinstrap" or "Adelie"."""
    if body_mass > 4700:
        return "Gentoo"
    if bill_length > 45:
        return "Chinstrap"
    return "Adelie"


def predict_with(rule, X):
    """Apply rule(bill_length, bill_depth, flipper_length, body_mass) to every row of the DataFrame X."""
    return np.array([rule(*row) for row in X[MEASURES].to_numpy()])


expert_pred = predict_with(expert_rule, measured)       # the species given by the rules, for the 342 penguins

# 3.16 and 3.19: a spam filter tested on 300 e-mails (labels "spam" and "ham")
rng_spam = np.random.default_rng(16)
is_spam = rng_spam.random(300) < 0.25
u_spam = rng_spam.random(300)
spam_true = np.where(is_spam, "spam", "ham")                                          # the truth
spam_pred = np.where(np.where(is_spam, u_spam < 0.70, u_spam < 0.06), "spam", "ham")  # the filter's answers
print(f"{len(expert_pred)} predictions of the expert rules · {len(spam_true)} e-mails for the spam filter")'''

SCREENING_17 = r'''rng_17 = np.random.default_rng(17)
sick_17 = rng_17.random(2000) < 0.10
u_17 = rng_17.random(2000)
test_truth = np.where(sick_17, "malade", "sain")                                          # the truth
test_result = np.where(np.where(sick_17, u_17 < 0.85, u_17 < 0.05), "malade", "sain")   # the rapid test


def screening_report(y_true, y_pred):
    """Sensitivity and precision of a screening test (the colleague's version)."""
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
    return {"sensitivity": tp / (tp + fn), "precision": tp / (tp + fp)}


print("the colleague's report:", {key: round(float(value), 3) for key, value in screening_report(test_truth, test_result).items()})'''

EXTREMES_18 = r'''y_18 = np.array([1] * 8 + [0] * 32)          # 8 positives, then 32 negatives
all_positive = np.ones(40, dtype=int)          # classifier 1
one_positive = np.zeros(40, dtype=int)         # classifier 2: only the most obvious case, a real positive
one_positive[0] = 1
for name, pred in [("1: all positive", all_positive), ("2: one positive", one_positive)]:
    print(f"{name:<16} accuracy {skm.accuracy_score(y_18, pred):.3f} · F1 {skm.f1_score(y_18, pred):.3f} · "
          f"balanced accuracy {skm.balanced_accuracy_score(y_18, pred):.4f} · "
          f"precision {skm.precision_score(y_18, pred):.3f} · recall {skm.recall_score(y_18, pred):.3f}")'''

KEYS_19 = ('["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "npv", "fpr", "fnr", "fdr", '
           '"false_omission_rate", "prevalence", "f1", "mcc"]')

PART_B = Part("B", "La matrice de confusion et les mesures binaires",
              "Fiche §3.7.1 à §3.7.11. Tu écris le cœur de `mylearn.metrics` : la matrice de confusion (3.15), puis "
              "accuracy, precision, recall, F-beta et F1 dans le cas binaire (3.16), et enfin le tableau complet des "
              "mesures (3.19). Entre les deux : une matrice lue à l'envers (3.17) et deux classifieurs extrêmes "
              "(3.18). La cellule ci-dessous prépare les données de la partie : les prédictions des règles de la "
              "biologiste (1.15) pour les 342 manchots, et les réponses d'un filtre anti-spam testé sur 300 e-mails.",
              given=PART_B_GIVEN, exercises=[
    Ex("3.15", "🔨", 2, 20, "confusion_matrix à la manière de scikit-learn",
       "écrire la matrice de confusion, vérité en lignes et prédictions en colonnes, comme scikit-learn.",
       "Ex 3.2 · Ex 1.15 (les règles de la biologiste) · fiche §3.7.1, §3.7.2 (🕰️ conventions)",
       thread="Penguins", tracks="R, M, C", mylearn="metrics.py",
       body=MYLEARN_HOWTO + r"""

Écris `confusion_matrix(y_true, y_pred, labels=None)` (lis sa docstring) :
- sans `labels`, les étiquettes sont celles qu'on trouve dans `y_true` **et** dans `y_pred`, triées : `np.unique` sur les deux réunies (`np.concatenate`) ;
- la case `C[i, j]` compte les échantillons de vérité `labels[i]` prédits `labels[j]` : un dictionnaire `{étiquette: numéro}` donne la ligne et la colonne de chaque échantillon, puis une boucle (ou `np.add.at`) compte ;
- lève une `ValueError` si les longueurs diffèrent, si les entrées sont vides, ou si une étiquette des données manque dans `labels` (scikit-learn, lui, écarterait ces échantillons sans rien dire).

Vérifications, sur les 342 manchots (la cellule de vérification appelle ta fonction : il n'y a rien à recopier) :
a) la matrice de confusion des règles de la biologiste (`species` contre `expert_pred`) ;
b) la même, avec les lignes et les colonnes dans l'ordre `["Gentoo", "Chinstrap", "Adelie"]` ;
c) `gentoo_as_chinstrap` : combien de Gentoo les règles prennent-elles pour des Chinstrap ? Lis-le dans la matrice que la vérification affiche (un entier).

Puis les tests de `confusion_matrix`.""",
       todo=r'''gentoo_as_chinstrap = ...   # c) read it in the matrix displayed by the check (an integer)''',
       check=RELOAD + r'''with wb.attempt("3.15"):
    C_15 = mylearn.metrics.confusion_matrix(species, expert_pred)
    wb.check("3.15a", C_15, computed=True)
    wb.check("3.15b", mylearn.metrics.confusion_matrix(species, expert_pred, labels=["Gentoo", "Chinstrap", "Adelie"]), computed=True)
    verdict("3.15", error_name(mylearn.metrics.confusion_matrix, species, expert_pred, labels=["Adelie", "Gentoo"]) == "ValueError",
            "une étiquette absente de labels lève bien une ValueError.",
            "confusion_matrix(species, expert_pred, labels=[\"Adelie\", \"Gentoo\"]) doit lever une ValueError : les "
            "Chinstrap n'ont ni ligne ni colonne.")
    wb.plot.plot_confusion_matrix(C_15, class_names=["Adelie", "Chinstrap", "Gentoo"], title="Expert rules (1.15), 342 penguins")
    plt.show()
    run_metrics_tests("test_confusion_matrix_")
wb.check("3.15c", gentoo_as_chinstrap)''',
       solution=r'''C_15 = mylearn.metrics.confusion_matrix(species, expert_pred)
C_15_ordered = mylearn.metrics.confusion_matrix(species, expert_pred, labels=["Gentoo", "Chinstrap", "Adelie"])
print(C_15, C_15_ordered, sep="\n\n")
gentoo_as_chinstrap = int(C_15[2, 1])   # row = truth (Gentoo, 3rd label), column = prediction (Chinstrap, 2nd label)
wb.plot.plot_confusion_matrix(C_15, class_names=["Adelie", "Chinstrap", "Gentoo"], title="Expert rules (1.15), 342 penguins")
plt.show()
run_metrics_tests("test_confusion_matrix_", impl="ref")''',
       record=r'''wb.record("3.15a", C_15)
wb.record("3.15b", C_15_ordered, mistakes={"ta fonction ignore labels : les lignes et les colonnes doivent suivre l'ordre de labels": C_15})
wb.record("3.15c", gentoo_as_chinstrap, mistakes={"c'est l'inverse, les Chinstrap pris pour des Gentoo : ligne = vérité (Gentoo), colonne = prédiction (Chinstrap)": int(C_15[1, 2])})''',
       note="La diagonale compte les bonnes réponses : 297 sur 342, soit une accuracy de 0,868. Mais la matrice dit "
            "**où** sont les erreurs : 33 des 45 erreurs viennent de Gentoo trop légers (4 700 g ou moins), que la "
            "règle 2 envoie chez les Chinstrap (bec de plus de 45 mm) ou que la règle 3 laisse chez les Adélie. C'est ce que le ch. 1 avait "
            "entrevu en 1.15 ; la matrice le chiffre. La référence est dans `solutions/mylearn_ref/metrics.py` : "
            "lis-la **après** avoir réussi les tests."),

    Ex("3.16", "🔨", 3, 40, "accuracy, precision, recall, F-beta et F1 (cas binaire)",
       "écrire accuracy, precision, recall, F-beta et F1 pour deux classes, avec une classe positive au choix.",
       "Ex 3.15 · fiche §3.7.5 à §3.7.11", thread="synthétique", tracks="R, M, C", mylearn="metrics.py",
       body=MYLEARN_SHORT + r"""

Écris `accuracy`, `precision`, `recall`, `fbeta` et `f1` pour le cas **binaire** (`average="binary"`, la valeur par défaut) ; les autres valeurs d'`average` viendront en 3.25 : d'ici là, fais-leur lever une `NotImplementedError`. Relis les docstrings :
- la classe positive est `pos_label`. En binaire, il y a au plus deux étiquettes, cherchées dans `y_true` **et** dans `y_pred` ; s'il y en a deux, `pos_label` doit en faire partie (sinon `ValueError`) ;
- `zero_division` est la valeur renvoyée quand un dénominateur est nul : aucune prédiction positive (precision), aucun positif réel (recall), $TP + FN + FP = 0$ (F-beta) ;
- $F_\beta = \frac{(1 + \beta^2)\,TP}{(1 + \beta^2)\,TP + \beta^2\,FN + FP}$ (fiche §3.7.11) ; `ValueError` si $\beta \le 0$, et `f1` appelle `fbeta` avec $\beta = 1$.

Compte TP, FP et FN dans **une** fonction d'aide que les quatre mesures partagent : moins de code, moins de bugs.

Vérifications, sur le filtre anti-spam (`spam_true`, `spam_pred` ; étiquettes `"spam"` et `"ham"`, classe positive `"spam"`) ; la cellule de vérification appelle tes fonctions :
a) l'accuracy ; b) la precision ; c) le recall ; d) le F1 ; e) le $F_2$ (`fbeta` avec `beta=2`) ;
puis un contrôle : sans `pos_label`, `precision(spam_true, spam_pred)` doit lever une `ValueError`, car l'étiquette positive par défaut, 1, n'existe pas ici ; enfin, les tests des cinq fonctions (cas binaire).

Dans tes notes : le $F_2$ est-il plus proche de la precision ou du recall ? Pourquoi ?""",
       check=RELOAD + r'''with wb.attempt("3.16"):
    mm = mylearn.metrics
    wb.check("3.16a", mm.accuracy(spam_true, spam_pred), computed=True)
    wb.check("3.16b", mm.precision(spam_true, spam_pred, pos_label="spam"), computed=True)
    wb.check("3.16c", mm.recall(spam_true, spam_pred, pos_label="spam"), computed=True)
    wb.check("3.16d", mm.f1(spam_true, spam_pred, pos_label="spam"), computed=True)
    wb.check("3.16e", mm.fbeta(spam_true, spam_pred, beta=2, pos_label="spam"), computed=True)
    verdict("3.16", error_name(mm.precision, spam_true, spam_pred) == "ValueError",
            "sans pos_label, la classe positive 1 n'existe pas : ValueError, comme scikit-learn.",
            "precision(spam_true, spam_pred) doit lever une ValueError : pos_label=1 n'est pas une des étiquettes.")
    run_metrics_tests("(test_accuracy_ or test_precision_ or test_recall_ or test_fbeta_ or test_f1_) and not multiclass and not curve")''',
       solution=r'''mm = mylearn.metrics
values_16 = {"accuracy": mm.accuracy(spam_true, spam_pred),
             "precision": mm.precision(spam_true, spam_pred, pos_label="spam"),
             "recall": mm.recall(spam_true, spam_pred, pos_label="spam"),
             "f1": mm.f1(spam_true, spam_pred, pos_label="spam"),
             "f2": mm.fbeta(spam_true, spam_pred, beta=2, pos_label="spam"),
             "f0.5": mm.fbeta(spam_true, spam_pred, beta=0.5, pos_label="spam")}
print({key: round(value, 3) for key, value in values_16.items()})
print(error_name(mm.precision, spam_true, spam_pred))
run_metrics_tests("(test_accuracy_ or test_precision_ or test_recall_ or test_fbeta_ or test_f1_) and not multiclass and not curve", impl="ref")''',
       record=r'''n_spam_tp = int(np.sum((spam_true == "spam") & (spam_pred == "spam")))
wb.record("3.16a", values_16["accuracy"], decimals=4, mistakes={"l'accuracy compte les DEUX sortes de bonnes réponses, TP et TN, parmi tous les e-mails": n_spam_tp / len(spam_true)})
wb.record("3.16b", values_16["precision"], decimals=4, mistakes={"c'est le recall : la precision divise TP par le nombre de prédictions positives, TP + FP": values_16["recall"],
                                                                  "c'est la precision de la classe \"ham\" : la classe positive est pos_label": mm.precision(spam_true, spam_pred, pos_label="ham")})
wb.record("3.16c", values_16["recall"], decimals=4, mistakes={"c'est la precision : le recall divise TP par le nombre de positifs réels, TP + FN": values_16["precision"],
                                                               "c'est le recall de la classe \"ham\" : la classe positive est pos_label": mm.recall(spam_true, spam_pred, pos_label="ham")})
wb.record("3.16d", values_16["f1"], decimals=4, mistakes={"c'est la moyenne ordinaire de la precision et du recall : le F1 est leur moyenne HARMONIQUE": (values_16["precision"] + values_16["recall"]) / 2})
wb.record("3.16e", values_16["f2"], decimals=4, mistakes={"c'est le F0,5 : dans la formule, β² multiplie FN (β = 2 favorise le recall)": values_16["f0.5"],
                                                           "c'est le F1 : ta fonction utilise-t-elle bien beta ?": values_16["f1"]})''',
       note="Ce filtre a une bonne accuracy (0,870), une precision correcte (0,761) et un recall plus faible (0,689) : "
            "il laisse passer près d'un spam sur trois. Le $F_2$ (0,702) est plus proche du recall, le plus faible des "
            "deux, parce que $\\beta = 2$ donne 4 fois plus de poids aux faux négatifs ; le $F_{0,5}$ (0,746) est plus "
            "proche de la precision. Toutes ces mesures sont des fractions de la même matrice : avec une seule "
            "fonction d'aide qui compte TP, FP et FN, les cinq fonctions tiennent en quelques lignes."),

    Ex("3.17", "🐛", 2, 15, "La matrice à l'envers",
       "diagnostiquer une matrice de confusion lue dans le mauvais sens, puis écrire une version juste quelles que "
       "soient les étiquettes.",
       "Ex 3.16 · fiche §3.7.2 (🕰️ conventions de scikit-learn), §3.7.3", thread="synthétique", tracks="C",
       body=r"""Un laboratoire évalue un test de dépistage rapide sur 2 000 personnes : `test_truth` contient la vérité (`"malade"` ou `"sain"`) et `test_result` le résultat du test. Un collègue a écrit `screening_report`, qui renvoie la sensibilité (le recall des malades) et la precision (la part de malades parmi les tests positifs). Son code tourne, ses chiffres semblent plausibles… et ils sont faux.

a) `true_sensitivity_17` : la vraie sensibilité du test, calculée avec **tes** fonctions de `mylearn.metrics` (3 décimales) ;
b) `true_precision_17` : sa vraie precision, avec tes fonctions aussi (3 décimales) ;
c) `top_left_17` : dans `skm.confusion_matrix(test_truth, test_result)`, quelle case se trouve en haut à gauche : `"TP"`, `"FN"`, `"FP"` ou `"TN"` (la classe positive étant `"malade"`) ? C'est la clé du bug.

Puis corrige : écris `screening_report_fixed(y_true, y_pred, positive)`, qui renvoie `{"sensitivity": ..., "precision": ...}`, juste quelles que soient les étiquettes, la classe positive étant `positive`. La vérification l'essaie avec quatre jeux d'étiquettes. Dans tes notes : quelles mesures le collègue a-t-il calculées en réalité ? Pourquoi son code marche-t-il avec des étiquettes 0 et 1, et pas ici ? Comment l'aurais-tu repéré ?""",
       given=SCREENING_17,
       todo=r'''true_sensitivity_17 = ...   # a) with your mylearn.metrics functions
true_precision_17 = ...     # b)
top_left_17 = ...           # c) "TP", "FN", "FP" or "TN"


def screening_report_fixed(y_true, y_pred, positive):
    """{"sensitivity": ..., "precision": ...} for any labels; `positive` is the positive class."""
    raise NotImplementedError("screening_report_fixed() is not written yet")''',
       check=r'''wb.check("3.17a", true_sensitivity_17)
wb.check("3.17b", true_precision_17)
wb.check("3.17c", top_left_17)
with wb.attempt("3.17"):
    sick_truth, sick_result = test_truth == "malade", test_result == "malade"
    cases_17 = {"malade / sain": (test_truth, test_result, "malade"),
                "1 / 0": (sick_truth.astype(int), sick_result.astype(int), 1),
                "yes / no": (np.where(sick_truth, "yes", "no"), np.where(sick_result, "yes", "no"), "yes"),
                "fraude / normal": (np.where(sick_truth, "fraude", "normal"), np.where(sick_result, "fraude", "normal"), "fraude")}
    for name, (truth, result, positive) in cases_17.items():
        report_17 = screening_report_fixed(truth, result, positive)
        expected_17 = (skm.recall_score(truth, result, pos_label=positive), skm.precision_score(truth, result, pos_label=positive))
        verdict("3.17", np.allclose([report_17["sensitivity"], report_17["precision"]], expected_17),
                f"étiquettes {name} : sensibilité et precision justes.",
                f"étiquettes {name} : ta sensibilité ou ta precision est fausse.")''',
       solution=r'''true_sensitivity_17 = mylearn.metrics.recall(test_truth, test_result, pos_label="malade")
true_precision_17 = mylearn.metrics.precision(test_truth, test_result, pos_label="malade")
print(skm.confusion_matrix(test_truth, test_result))    # labels sorted: "malade" < "sain", so row 0 = the sick
top_left_17 = "TP"


def screening_report_fixed(y_true, y_pred, positive):
    """{"sensitivity": ..., "precision": ...} for any labels; `positive` is the positive class."""
    truth = np.asarray(y_true) == positive
    alarm = np.asarray(y_pred) == positive
    tp, fn, fp = np.sum(truth & alarm), np.sum(truth & ~alarm), np.sum(~truth & alarm)
    return {"sensitivity": float(tp / (tp + fn)), "precision": float(tp / (tp + fp))}


print(true_sensitivity_17, true_precision_17, screening_report_fixed(test_truth, test_result, "malade"))''',
       record=r'''wrong_17 = screening_report(test_truth, test_result)
wb.record("3.17a", true_sensitivity_17, decimals=3, mistakes={"c'est la sensibilité annoncée par le collègue : recalcule-la avec tes fonctions (le recall, classe positive \"malade\")": float(wrong_17["sensitivity"]),
                                                              "c'est la precision : la sensibilité est le RECALL des malades": true_precision_17})
wb.record("3.17b", true_precision_17, decimals=3, mistakes={"c'est la precision annoncée par le collègue : recalcule-la avec tes fonctions (classe positive \"malade\")": float(wrong_17["precision"]),
                                                            "c'est la sensibilité (le recall) : la precision divise par les tests POSITIFS": true_sensitivity_17})
wb.record("3.17c", top_left_17, mistakes={"c'est vrai pour des étiquettes 0 et 1, rangées 0 puis 1 ; ici, scikit-learn trie des mots : lequel vient en premier, \"malade\" ou \"sain\" ?": "TN",
                                           "la case en haut à gauche est sur la diagonale : une bonne réponse, pas un faux négatif": "FN",
                                           "la case en haut à gauche est sur la diagonale : une bonne réponse, pas un faux positif": "FP"})''',
       note="`confusion_matrix` range les étiquettes dans l'ordre **trié** : `\"malade\" < \"sain\"`, donc la ligne "
            "0 est celle des malades et la matrice vaut `[[TP, FN], [FP, TN]]`. Le `.ravel()` du collègue range "
            "TP dans sa variable `tn`, FN dans `fp`, FP dans `fn` et TN dans `tp` : sa « sensibilité » est en réalité "
            "la spécificité, $\\frac{TN}{TN + FP}$, et sa « precision » la NPV, $\\frac{TN}{TN + FN}$. Avec des "
            "étiquettes 0/1 (ou `\"no\"`/`\"yes\"`), le positif est trié en second et le code tombe juste… par "
            "chance. Corrections sûres : `labels=[negative, positive]` dans `confusion_matrix`, ou des masques "
            "booléens comme ici. Pour le repérer : tester la fonction sur un petit exemple calculé à la main, avec les "
            "étiquettes réelles du projet."),

    Ex("3.18", "🔮", 2, 15, "Tout positif, un seul positif : prédire les scores",
       "prévoir l'accuracy, le F1 et la balanced accuracy de deux classifieurs extrêmes, et voir que ces mesures "
       "ne les classent pas dans le même ordre.",
       "Ex 3.16 · fiche §3.7.9 (balanced accuracy), §3.7.10", thread="synthétique", tracks="C", hypothesis=True,
       body=r"""Un jeu de test de 40 échantillons contient 8 positifs. Deux classifieurs paresseux :
- le classifieur 1 répond « positif » pour **tous** les échantillons ;
- le classifieur 2 ne répond « positif » que pour **un seul** échantillon, le plus évident, qui est bien un positif ; « négatif » pour tous les autres.

La fiche (§3.7.10) donne déjà leur precision et leur recall. **Sans rien exécuter**, prévois :
a) `prediction_3_18a` : l'accuracy du classifieur 1 (2 décimales) ;
b) `prediction_3_18b` : son F1 (3 décimales) ;
c) `prediction_3_18c` : l'accuracy du classifieur 2 (3 décimales) ;
d) `prediction_3_18d` : son F1 (3 décimales) ;
e) `prediction_3_18e` : lequel a la meilleure balanced accuracy (fiche §3.7.9) : `1` ou `2` ?

Écris ton hypothèse (cellule 📝), puis tes cinq prédictions ; la vérification ne regarde que tes prédictions. Ensuite seulement, exécute l'**Expérience**.""",
       todo=r'''prediction_3_18a = ...   # the accuracy of classifier 1
prediction_3_18b = ...   # its F1
prediction_3_18c = ...   # the accuracy of classifier 2
prediction_3_18d = ...   # its F1
prediction_3_18e = ...   # 1 or 2''',
       check=r'''for letter, prediction in zip("abcde", [prediction_3_18a, prediction_3_18b, prediction_3_18c,
                                        prediction_3_18d, prediction_3_18e]):
    wb.check(f"3.18{letter}", prediction)''',
       solution=r'''prediction_3_18a, prediction_3_18b = 8 / 40, 2 * 8 / (2 * 8 + 32 + 0)        # TP = 8, FP = 32, FN = 0
prediction_3_18c, prediction_3_18d = 33 / 40, 2 * 1 / (2 * 1 + 0 + 7)       # TP = 1, FP = 0, FN = 7, TN = 32
prediction_3_18e = 2                                                        # (1/8 + 1) / 2 against (1 + 0) / 2''',
       record=r'''wb.record("3.18a", prediction_3_18a, decimals=2, mistakes={"c'est son recall : l'accuracy compte les bonnes réponses parmi les 40 échantillons": 1.0,
                                                              "c'est la part des négatifs : le classifieur 1 se trompe justement sur eux": 0.8})
wb.record("3.18b", prediction_3_18b, decimals=3, mistakes={"c'est la moyenne ordinaire de la precision et du recall : le F1 est leur moyenne HARMONIQUE": 0.6})
wb.record("3.18c", prediction_3_18c, decimals=3, mistakes={"c'est sa precision : l'accuracy compte TOUTES les bonnes réponses, négatifs compris": 1.0,
                                                            "l'accuracy compte TOUTES les bonnes réponses, négatifs compris": 1 / 40})
wb.record("3.18d", prediction_3_18d, decimals=3, mistakes={"c'est la moyenne ordinaire de la precision et du recall : le F1 est leur moyenne HARMONIQUE": 0.5625,
                                                            "c'est la moyenne ordinaire (arrondie) de la precision et du recall : le F1 est leur moyenne HARMONIQUE": 0.563,
                                                            "c'est son recall : le F1 combine precision ET recall": 0.125})
wb.record("3.18e", prediction_3_18e, mistakes={"la balanced accuracy fait la moyenne du recall ET de la spécificité : que vaut la spécificité du classifieur 1 ?": 1})''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", guarded(EXTREMES_18, [f"prediction_3_18{letter}" for letter in "abcde"],
                               "⏳ Ex 3.18 : écris d'abord tes cinq prédictions, puis relance cette cellule."))],
       note="L'accuracy préfère largement le classifieur 2 (0,825 contre 0,2), le F1 préfère le 1 (0,333 contre "
            "0,222), et la balanced accuracy les trouve presque aussi mauvais l'un que l'autre (0,5 et 0,5625, près "
            "de 0,5, le score du hasard). Aucun n'est utile : le 1 ne distingue rien, le 2 rate 7 positifs sur 8. "
            "Leçon : une mesure seule se laisse berner par un classifieur dégénéré, et deux mesures peuvent classer "
            "deux modèles en sens inverse. Compare toujours à une référence triviale (la classe majoritaire, tout "
            "positif) et regarde la matrice."),

    Ex("3.19", "🔨", 2, 20, "Le tableau de bord complet : classification_rates",
       "calculer d'un coup les 13 mesures d'une matrice binaire, avec une seule règle pour les divisions par zéro.",
       "Ex 3.16, Ex 3.5 · fiche §3.7.9 (tableau et 🕰️)", thread="synthétique", tracks="M, C", mylearn="metrics.py",
       body=MYLEARN_SHORT + r"""

Écris `classification_rates(y_true, y_pred, pos_label=1, zero_division=0.0)`, qui renvoie un dictionnaire des 13 mesures du tableau de la fiche (§3.7.9), **dans l'ordre de la docstring** (un dictionnaire Python garde l'ordre d'insertion). Compte TP, FN, FP et TN une seule fois, puis écris une petite fonction d'aide `ratio(num, den)` qui renvoie `zero_division` quand `den` vaut 0 : **tous** les rapports passent par elle, ceux du MCC et de la balanced accuracy compris (scikit-learn a pour ces deux-là ses propres conventions ; les tests suivent la docstring). Mêmes contrôles des étiquettes qu'en 3.16 ; chaque valeur est un `float`.

Vérifications, sur le filtre anti-spam de 3.16 (classe positive `"spam"`) :
a) `[specificity, npv, balanced_accuracy, mcc]`, calculés par ta fonction (la cellule de vérification l'appelle) ;
b) `swap_19` : si l'on prend `"ham"` comme classe positive, la precision devient l'une des autres mesures du tableau calculé avec `"spam"` : laquelle ? (sa clé, une chaîne) ;
puis l'ordre des clés, et les tests de `classification_rates`.""",
       todo=r'''swap_19 = ...   # b) a key of the dictionary''',
       check=RELOAD + r'''with wb.attempt("3.19"):
    rates_19 = mylearn.metrics.classification_rates(spam_true, spam_pred, pos_label="spam")
    if {"specificity", "npv", "balanced_accuracy", "mcc"} <= set(rates_19):
        wb.check("3.19a", [rates_19["specificity"], rates_19["npv"], rates_19["balanced_accuracy"], rates_19["mcc"]],
                 computed=True)
    verdict("3.19", list(rates_19) == ''' + KEYS_19 + r''',
            "les 13 clés sont là, dans l'ordre de la docstring.",
            "les clés doivent être exactement celles de la docstring, dans le même ordre.")
    run_metrics_tests("test_classification_rates_")
wb.check("3.19b", swap_19)''',
       solution=r'''rates_19 = mylearn.metrics.classification_rates(spam_true, spam_pred, pos_label="spam")
print({key: round(value, 3) for key, value in rates_19.items()})
rates_ham = mylearn.metrics.classification_rates(spam_true, spam_pred, pos_label="ham")
print("precision with 'ham' as the positive class:", round(rates_ham["precision"], 3))
swap_19 = "npv"
run_metrics_tests("test_classification_rates_", impl="ref")''',
       record=r'''r19 = rates_19
wb.record("3.19a", [r19["specificity"], r19["npv"], r19["balanced_accuracy"], r19["mcc"]], decimals=4,
          mistakes={"la spécificité et la NPV sont échangées : la spécificité divise TN par les négatifs RÉELS (TN + FP), la NPV par les prédictions négatives (TN + FN)": [r19["npv"], r19["specificity"], r19["balanced_accuracy"], r19["mcc"]],
                    "la balanced accuracy est la moyenne du recall et de la SPÉCIFICITÉ (pas de la precision)": [r19["specificity"], r19["npv"], (r19["precision"] + r19["recall"]) / 2, r19["mcc"]]})
wb.record("3.19b", swap_19, mistakes={"c'est ce que devient le RECALL de \"ham\", pas sa precision : une precision divise par des PRÉDICTIONS, pas par des cas réels": "specificity",
                                      "une precision divise par des PRÉDICTIONS : écris la precision de \"ham\" avec les quatre cases de la matrice \"spam\"": "recall"})''',
       note="Changer de classe positive échange les rôles : la precision de « ham » est la NPV de « spam », son "
            "recall est la spécificité de « spam ». Le tableau entier se déduit des quatre cases ; `classification_rates` "
            "le calcule une fois pour toutes. La règle unique `zero_division` rend la fonction prévisible dans les cas "
            "dégénérés (une classe jamais prédite, un jeu sans positif), là où scikit-learn a des conventions qui "
            "varient d'une fonction à l'autre."),
])

# ---------------------------------------------------------------------------
# Part C: threshold, prevalence, scikit-learn tools and documentation (3.20 to 3.23)
# ---------------------------------------------------------------------------
PLOT_20 = r'''    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for column in ["precision", "recall", "f1"]:
        ax.plot(table_20["threshold"], table_20[column], marker=".", label=column)
    ax.set_xlabel("threshold t (mm): Gentoo when flipper >= t")
    ax.set_ylim(0, 1.02)
    ax.legend()
    plt.show()'''

PLOT_21 = r'''    prevalences_21 = np.logspace(-4, np.log10(0.5), 200)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(prevalences_21, [theoretical_precision(p, 0.99, 0.98) for p in prevalences_21], label="theory")
    ax.scatter([0.01], [np.sum(sick_21 & positive_21) / np.sum(positive_21)], color="C1", zorder=3,
               label="simulation, 100 000 people")
    ax.set_xscale("log")
    ax.set_xlabel("prevalence (share of sick people), log scale")
    ax.set_ylabel("precision P(sick | positive)")
    ax.set_title("The book's test: sensitivity 0.99, specificity 0.98")
    ax.legend()
    plt.show()'''

DOC_23 = r'''def numbers_23(value):
    """The numbers of an answer written as a learner would: 0.75, "0,75", "3/4", or lists of them."""
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if "/" in text:
            top, bottom = text.split("/")
            return float(top) / float(bottom)
        return float(text)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [numbers_23(item) for item in value]
    return float(value)


WORDS_23 = {"column": "colonnes", "columns": "colonnes", "colonne": "colonnes", "colonnes": "colonnes",
            "row": "lignes", "rows": "lignes", "ligne": "lignes", "lignes": "lignes",
            "all": "tout", "tout": "tout", "toute": "tout", "total": "tout"}


def compare_23(letter, answer, actual, explanation):
    """Compare your prediction with what scikit-learn really does."""
    if answer is ... or answer is None:
        print(f"⏳ Ex 3.23{letter} : pas encore fait.")
        return
    try:
        mine, real = np.asarray(numbers_23(answer), dtype=float), np.asarray(numbers_23(actual), dtype=float)
        ok = mine.shape == real.shape and bool(np.allclose(mine, real))
    except (TypeError, ValueError, ZeroDivisionError):
        def word(text):
            text = str(text).strip().strip("'\"").lower()
            return WORDS_23.get(text, text)
        ok = word(answer) == word(actual)
    verdict(f"3.23{letter}", ok, f"juste ({explanation}).",
            f"scikit-learn donne {actual!r} ({explanation}) : relis la documentation.")


def raised_23(func, *args, **kwargs):
    """Name of the error raised by a scikit-learn call, or "no error"."""
    try:
        func(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - we want the name of whatever is raised
        return type(error).__name__
    return "no error"


cm_23 = skm.confusion_matrix(["a", "b", "b", "a", "b"], ["a", "b", "a", "a", "a"], normalize="pred")
sums_23 = "colonnes" if np.allclose(cm_23.sum(axis=0), 1) else "lignes" if np.allclose(cm_23.sum(axis=1), 1) else "tout"
compare_23("a", answer_23a, skm.precision_score([1, 1, 0], [0, 0, 0], zero_division=1), "zero_division")
compare_23("b", answer_23b, skm.confusion_matrix(["a", "b", "c", "a"], ["a", "c", "b", "a"], labels=["a", "b"]).tolist(), "labels")
compare_23("c", answer_23c, raised_23(skm.recall_score, ["spam", "ham", "spam"], ["spam", "spam", "ham"]), "pos_label")
compare_23("d", answer_23d, sums_23, "normalize")
compare_23("e", answer_23e, skm.accuracy_score([1, 0, 1], [1, 1, 1], sample_weight=[1, 1, 2]), "sample_weight")
compare_23("f", answer_23f, skm.precision_score(["b", "a", "c", "a"], ["b", "a", "a", "c"], average=None).tolist(), "average=None")'''

PART_C = Part("C", "Seuil, prévalence et outils de scikit-learn",
              "Fiche §3.7.4, §3.7.10 et §3.8. Un classifieur donne souvent un **score** ; le seuil qui en fait une "
              "décision se choisit selon le coût des erreurs (3.20). La precision dépend de la prévalence : tu le "
              "vérifies par la simulation (3.21). Enfin, tu contrôles tes résultats avec les outils de scikit-learn "
              "et tu apprends à lire leur documentation (3.22, 3.23).",
              exercises=[
    Ex("3.20", "🔬", 2, 25, "Un seuil sur la nageoire : precision et recall en balance",
       "balayer les seuils d'un score, puis choisir un seuil selon le F1 ou selon le coût des erreurs.",
       "Ex 3.16 · fiche §3.7.4, §3.7.10 (🧮 score et seuil), §3.7.11", thread="Penguins", tracks="R, M, C",
       body=r"""Pas encore de modèle entraîné (ce sera au ch. 7) : le **score** est ici une simple mesure, la longueur de la nageoire (`flipper`, en mm, des nombres entiers). La classe positive est « Gentoo » (`is_gentoo` : 1 pour un Gentoo, 0 sinon), contre les deux autres espèces. Au seuil $t$, on prédit « Gentoo » quand `flipper >= t` (supérieur **ou égal**).

Écris d'abord `sweep_20(thresholds)`, qui renvoie un DataFrame avec une ligne par seuil et les colonnes `threshold`, `precision`, `recall`, `f1`, `fn` et `fp`, calculées avec tes fonctions de `mylearn.metrics` (les prédictions en 0/1 : `(flipper >= t).astype(int)`). La vérification trace precision, recall et F1 en fonction du seuil, avec ton tableau `sweep_20(np.arange(190, 231))`. Puis :

a) `pr_205` : `[precision, recall]` au seuil $t = 205$ mm (3 décimales) ;
b) `best_f1_t` : parmi les seuils entiers de 190 à 230 mm, celui qui maximise le F1 ;
c) `best_cost_t` : un Gentoo manqué (FN) coûte 10, un faux Gentoo (FP) coûte 1 ; parmi les mêmes seuils, celui qui minimise le coût total $10 \times FN + 1 \times FP$.

Lis b et c dans ton tableau (`idxmax`, `idxmin`, 0A). Dans tes notes : pourquoi les seuils de b et c diffèrent-ils ? Lequel choisirais-tu pour un recensement où rater un Gentoo coûte cher ?""",
       todo=r'''def sweep_20(thresholds):
    """One row per threshold t (Gentoo when flipper >= t): threshold, precision, recall, f1, fn, fp."""
    raise NotImplementedError("sweep_20() is not written yet")


pr_205 = ...        # a) [precision, recall] at t = 205
best_f1_t = ...     # b)
best_cost_t = ...   # c)''',
       check=r'''wb.check("3.20a", pr_205)
wb.check("3.20b", best_f1_t)
wb.check("3.20c", best_cost_t)
with wb.attempt("3.20"):
    table_20 = sweep_20(np.arange(190, 231))
''' + PLOT_20,
       solution=r'''def sweep_20(thresholds):
    """One row per threshold t (Gentoo when flipper >= t): threshold, precision, recall, f1, fn, fp."""
    rows = []
    for t in thresholds:
        pred = (flipper >= t).astype(int)
        rows.append({"threshold": int(t),
                     "precision": mylearn.metrics.precision(is_gentoo, pred),
                     "recall": mylearn.metrics.recall(is_gentoo, pred),
                     "f1": mylearn.metrics.f1(is_gentoo, pred),
                     "fn": int(np.sum((is_gentoo == 1) & (pred == 0))),
                     "fp": int(np.sum((is_gentoo == 0) & (pred == 1)))})
    return pd.DataFrame(rows)


table_20 = sweep_20(np.arange(190, 231))
row_205 = table_20.set_index("threshold").loc[205]
pr_205 = [row_205["precision"], row_205["recall"]]
best_f1_t = int(table_20.loc[table_20["f1"].idxmax(), "threshold"])
table_20["cost"] = 10 * table_20["fn"] + table_20["fp"]
best_cost_t = int(table_20.loc[table_20["cost"].idxmin(), "threshold"])
print(table_20[(table_20["threshold"] >= 200) & (table_20["threshold"] <= 213)].round(3).to_string(index=False))
print(pr_205, best_f1_t, best_cost_t)
''' + PLOT_20.replace("\n    ", "\n").lstrip(),
       record=r'''row_206 = table_20.set_index("threshold").loc[206]  # "flipper > 205" is "flipper >= 206" for whole numbers
wb.record("3.20a", pr_205, decimals=3, mistakes={"c'est [recall, precision] : l'ordre demandé est [precision, recall]": pr_205[::-1],
                                                  "au seuil t, on prédit Gentoo quand flipper >= t (supérieur OU ÉGAL), pas >": [row_206["precision"], row_206["recall"]]})
IDX_20 = "c'est le numéro de la ligne, pas le seuil : idxmax et idxmin renvoient l'étiquette de la ligne ; lis la colonne threshold de cette ligne"
wb.record("3.20b", best_f1_t, mistakes={"on prédit Gentoo quand flipper >= t (supérieur OU ÉGAL), pas > : ton seuil est décalé d'un millimètre": best_f1_t - 1,
                                         "le F1 de ce seuil est presque le meilleur, mais pas le meilleur : compare les F1 sans les arrondir": best_f1_t + 1,
                                         IDX_20: best_f1_t - 190})
wb.record("3.20c", best_cost_t, mistakes={"on prédit Gentoo quand flipper >= t (supérieur OU ÉGAL), pas > : ton seuil est décalé d'un millimètre": best_cost_t - 1,
                                           "c'est le seuil du meilleur F1 : ici, on minimise le coût 10 × FN + FP": best_f1_t,
                                           IDX_20: best_cost_t - 190,
                                           "un FN coûte 10 et un FP coûte 1, pas l'inverse": int(table_20.loc[(table_20["fn"] + 10 * table_20["fp"]).idxmin(), "threshold"])})''',
       note="Au seuil 205 mm, un seul Gentoo est manqué, mais 11 manchots d'autres espèces sont pris pour des "
            "Gentoo. En montant le seuil, la precision monte et le recall baisse ; le F1 est maximal à 207 mm, où il "
            "reste un FN et 7 FP. Si rater un Gentoo coûte 10 fois plus cher qu'une fausse alerte, le meilleur seuil "
            "descend à 203 mm : aucun Gentoo manqué, 15 fausses alertes. Le F1 suppose implicitement qu'un FN et un "
            "FP ont à peu près le même poids ; dès que les coûts sont connus, on minimise le coût. Attention : "
            "choisir le seuil sur les données où l'on mesure le résultat est optimiste. Il faudrait le choisir sur "
            "un jeu de validation (ch. 8) : c'est l'objet du défi 3.29."),

    Ex("3.21", "🔬", 2, 30, "Simuler le dépistage : la prévalence fait la precision",
       "vérifier par la simulation que la precision d'un test dépend de la prévalence, et trouver la prévalence à "
       "partir de laquelle un positif est plus souvent malade que sain.",
       "Ex 3.7, Ex 3.19 · ch. 2 (tirages de Bernoulli, graine) · fiche §3.8", thread="synthétique", tracks="M, C",
       body=r"""Le test du livre (§3.8) a une sensibilité de 0,99 et une spécificité de 0,98 ; 1 % de la population est malade. Tu as calculé ses mesures sur papier (3.7, 2ᵉ partie) ; ici, tu mesures sa precision sur une population simulée.

Écris `simulate_screening(n, prevalence, sensitivity, specificity, rng)`, qui renvoie deux tableaux de booléens `(sick, positive)` de longueur $n$, tirés **exactement** ainsi (pour que tout le monde obtienne les mêmes nombres) :
1. `sick = rng.random(n) < prevalence` ;
2. puis `u = rng.random(n)` : une personne est positive si `u < sensitivity` quand elle est malade, si `u < 1 - specificity` quand elle est saine (`np.where`).

a) `precision_21` : la precision mesurée sur $n = 100\,000$ personnes, avec `rng = np.random.default_rng(21)` (4 décimales) ; par exemple avec ta fonction `classification_rates` de 3.19 (convertis d'abord les booléens en 0/1 avec `.astype(int)`), ou directement en comptant ;
b) écris `theoretical_precision(prevalence, sensitivity, specificity)`, la precision attendue (fréquences naturelles, fiche §3.8), puis `theory_21` : ses valeurs pour une prévalence de 0,1 % et de 10 % (liste, 3 décimales) ;
c) `half_21` : la prévalence pour laquelle la precision vaut exactement 0,5 (4 décimales) : un positif a alors autant de chances d'être malade que sain. Résous l'équation à la main, ou cherche-la sur une grille de prévalences (un pas d'au plus 0,000 01, et la prévalence dont la precision est la plus proche de 0,5) ;
d) `double_21` : juste après le premier test (avec le même générateur, sans le recréer), fais passer un **second** test indépendant, de mêmes caractéristiques, à toute la population : `u2 = rng.random(n)`, puis la même règle qu'au point 2. Quelle est la precision parmi les personnes positives **aux deux** tests (3 décimales) ?

La vérification trace la precision théorique en fonction de la prévalence, avec ton point simulé. Dans tes notes : pourquoi un second test change-t-il autant la precision ? (Le ch. 4, règle de Bayes, y reviendra.)""",
       todo=r'''def simulate_screening(n, prevalence, sensitivity, specificity, rng):
    """(sick, positive): two boolean arrays of length n, drawn exactly as in the statement."""
    raise NotImplementedError("simulate_screening() is not written yet")


def theoretical_precision(prevalence, sensitivity, specificity):
    """Expected precision P(sick | positive) of the test (natural frequencies)."""
    raise NotImplementedError("theoretical_precision() is not written yet")


precision_21 = ...   # a)
theory_21 = ...      # b) [prevalence 0.1 %, prevalence 10 %]
half_21 = ...        # c)
double_21 = ...      # d)''',
       check=r'''wb.check("3.21a", precision_21)
wb.check("3.21b", theory_21)
wb.check("3.21c", half_21)
wb.check("3.21d", double_21)
with wb.attempt("3.21"):
    sick_21, positive_21 = simulate_screening(100_000, 0.01, 0.99, 0.98, np.random.default_rng(21))
    verdict("3.21", bool(wb.check("3.21a", np.sum(sick_21 & positive_21) / np.sum(positive_21), quiet=True)),
            "simulate_screening tire comme l'énoncé : sa precision est la bonne réponse de a).",
            "avec la graine 21, simulate_screening ne redonne pas la bonne precision : tire exactement comme l'énoncé "
            "(les malades d'abord, puis un seul tableau u pour le test).")
    verdict("3.21", bool(wb.check("3.21b", [theoretical_precision(0.001, 0.99, 0.98), theoretical_precision(0.1, 0.99, 0.98)], quiet=True)),
            "theoretical_precision donne les bonnes valeurs de b).",
            "theoretical_precision ne donne pas les bonnes valeurs de b) : suis l'arbre des fréquences naturelles.")
''' + PLOT_21,
       solution=r'''def simulate_screening(n, prevalence, sensitivity, specificity, rng):
    """(sick, positive): two boolean arrays of length n, drawn exactly as in the statement."""
    sick = rng.random(n) < prevalence
    u = rng.random(n)
    positive = np.where(sick, u < sensitivity, u < 1 - specificity)
    return sick, positive


def theoretical_precision(prevalence, sensitivity, specificity):
    """Expected precision P(sick | positive) of the test (natural frequencies)."""
    true_positives = sensitivity * prevalence                 # share of the population: sick and positive
    false_positives = (1 - specificity) * (1 - prevalence)    # healthy and positive
    return true_positives / (true_positives + false_positives)


rng = np.random.default_rng(21)
sick_21, positive_21 = simulate_screening(100_000, 0.01, 0.99, 0.98, rng)
rates_21 = mylearn.metrics.classification_rates(sick_21.astype(int), positive_21.astype(int))
precision_21 = rates_21["precision"]
theory_21 = [theoretical_precision(0.001, 0.99, 0.98), theoretical_precision(0.1, 0.99, 0.98)]
half_21 = 0.02 / (0.99 + 0.02)            # 0.99 p = 0.02 (1 - p): as many true as false positives
u2 = rng.random(100_000)                  # the second test, with the same generator
positive_2 = np.where(sick_21, u2 < 0.99, u2 < 0.02)
both_21 = positive_21 & positive_2
double_21 = np.sum(sick_21 & both_21) / np.sum(both_21)
print(f"{sick_21.sum()} sick, {positive_21.sum()} positive · precision {precision_21:.4f} (theory {theoretical_precision(0.01, 0.99, 0.98):.4f})")
print(theory_21, half_21, theoretical_precision(half_21, 0.99, 0.98), f"· two tests: {np.sum(both_21)} people, precision {double_21:.4f}")
''' + PLOT_21.replace("\n    ", "\n").lstrip(),
       record=r'''sick_x = np.random.default_rng(21).random(100_000) < 0.01           # classic slip: the generator re-created for u
u_x = np.random.default_rng(21).random(100_000)
positive_x = np.where(sick_x, u_x < 0.99, u_x < 0.02)
u2_x = np.random.default_rng(21).random(100_000)                     # classic slip: re-created for the second test
both_x = positive_21 & np.where(sick_21, u2_x < 0.99, u2_x < 0.02)
wb.record("3.21a", precision_21, decimals=4, mistakes={"tire u avec le MÊME générateur, juste après les malades, sans le recréer : recréé, il redonne les mêmes nombres, et le test « sait » qui est malade": np.sum(sick_x & positive_x) / np.sum(positive_x),
                                                          "c'est la precision THÉORIQUE : on demande celle que tu mesures sur ta population simulée": theoretical_precision(0.01, 0.99, 0.98),
                                                          "c'est la part des positifs dans toute la population : la precision ne regarde que les positifs, et compte ceux qui sont malades": positive_21.mean()})
wb.record("3.21b", theory_21, decimals=3, mistakes={"les faux positifs viennent des personnes SAINES : multiplie 1 − spécificité par 1 − prévalence": [0.99 * p / (0.99 * p + 0.02) for p in (0.001, 0.1)]})
wb.record("3.21c", half_21, decimals=4, mistakes={"c'est 1 − spécificité : la precision vaut 0,5 quand les vrais positifs sont aussi nombreux que les faux ; écris les deux en fonction de la prévalence p": 0.02,
                                                  "les faux positifs viennent des personnes SAINES, une part 1 − p de la population, pas de toute la population": 0.02 / 0.99})
wb.record("3.21d", double_21, decimals=3, mistakes={"c'est la valeur THÉORIQUE pour deux tests indépendants : on demande la precision mesurée sur ta simulation": theoretical_precision(theoretical_precision(0.01, 0.99, 0.98), 0.99, 0.98),
                                                     "c'est la precision d'UN seul test : garde les personnes positives aux deux tests, et tire le second test à la suite du premier, avec le même générateur (recréé, il rejouerait le premier test)": precision_21,
                                                     "tu as recréé le générateur pour le second test : il rejoue des nombres déjà utilisés, et les deux tests ne sont plus indépendants ; tire u2 à la suite, avec le même générateur": np.sum(sick_21 & both_x) / np.sum(both_x)})''',
       note="Sur 100 000 personnes, environ 1 000 malades donnent 990 vrais positifs, mais les 99 000 personnes saines "
            "en donnent près de 2 000 faux : la precision mesurée (0,3255) est proche de la valeur théorique, 1/3. "
            "Elle vaut moins de 5 % pour une maladie dix fois plus rare, et 85 % pour une maladie dix fois plus "
            "fréquente : c'est le même test, la sensibilité et la spécificité n'ont pas bougé. Le second test "
            "s'applique, en pratique, à une population où un tiers des gens sont malades : la precision passe au-dessus "
            "de 0,95. C'est la logique du test de confirmation (fiche §3.8) : chaque test positif « augmente la "
            "prévalence » du groupe suivant, à condition que les deux tests se trompent indépendamment."),

    Ex("3.22", "📦", 2, 20, "Vérifier avec scikit-learn : classification_report et affichages",
       "lire un rapport de classification et une matrice de confusion normalisée de scikit-learn.",
       "Ex 3.16, Ex 3.15 · fiche « Au-delà du livre : plusieurs classes », §3.7.9 (🕰️)", thread="Penguins", tracks="R, C",
       body=r"""scikit-learn résume tout en un tableau : `classification_report`. Applique-le aux règles de la biologiste (`species` contre `expert_pred`), avec `digits=3`, et affiche aussi leur matrice de confusion avec `skm.ConfusionMatrixDisplay.from_predictions(...)`, suivi de `plt.show()`.

a) `recall_gentoo_22` : le recall des Gentoo, lu dans le rapport (3 décimales) ;
b) `macro_f1_22` : le F1 macro (3 décimales) ;
c) `weighted_f1_22` : le F1 pondéré (3 décimales) ;
d) `normalize_22` : la valeur de l'argument `normalize` de `ConfusionMatrixDisplay.from_predictions` qui fait apparaître sur la diagonale le recall de chaque espèce (une chaîne ; lis sa documentation : `help(skm.ConfusionMatrixDisplay.from_predictions)`).

Récupère enfin le rapport sous forme de dictionnaire, `output_dict=True`, dans `report_22` : la vérification y lit le recall des Gentoo. Dans tes notes : pourquoi le F1 macro est-il plus bas que le F1 pondéré ? Quelle espèce le tire vers le bas ?""",
       todo=r'''# Display the report and the confusion matrix here, then fill in:
recall_gentoo_22 = ...   # a)
macro_f1_22 = ...        # b)
weighted_f1_22 = ...     # c)
normalize_22 = ...       # d) a string
report_22 = ...          # the report as a dictionary (output_dict=True)''',
       check=r'''wb.check("3.22a", recall_gentoo_22)
wb.check("3.22b", macro_f1_22)
wb.check("3.22c", weighted_f1_22)
wb.check("3.22d", normalize_22)
if report_22 is ...:
    print("⏳ Ex 3.22 : report_22 pas encore fait.")
else:
    gentoo_22 = report_22.get("Gentoo", {}) if isinstance(report_22, dict) else {}
    verdict("3.22", bool(wb.check("3.22a", gentoo_22.get("recall"), quiet=True)),
            "report_22[\"Gentoo\"][\"recall\"] est bien le recall des Gentoo.",
            "report_22 doit être le dictionnaire de classification_report(..., output_dict=True), avec une entrée par espèce.")''',
       solution=r'''print(skm.classification_report(species, expert_pred, digits=3))
normalize_22 = "true"
display_22 = skm.ConfusionMatrixDisplay.from_predictions(species, expert_pred, normalize=normalize_22, values_format=".2f")
display_22.ax_.grid(False)
plt.title("Expert rules: each row divided by its total")
plt.show()
report_22 = skm.classification_report(species, expert_pred, output_dict=True)
recall_gentoo_22 = report_22["Gentoo"]["recall"]
macro_f1_22 = report_22["macro avg"]["f1-score"]
weighted_f1_22 = report_22["weighted avg"]["f1-score"]
print(recall_gentoo_22, macro_f1_22, weighted_f1_22)''',
       record=r'''r22 = report_22
wb.record("3.22a", recall_gentoo_22, decimals=3, mistakes={"c'est la precision des Gentoo : le recall est dans la 2ᵉ colonne du rapport": r22["Gentoo"]["precision"],
                                                            "c'est le F1 des Gentoo : le recall est dans la 2ᵉ colonne du rapport": r22["Gentoo"]["f1-score"]})
wb.record("3.22b", macro_f1_22, decimals=3, mistakes={"c'est le F1 pondéré (ligne weighted avg) : on demande la ligne macro avg": r22["weighted avg"]["f1-score"],
                                                       "c'est la precision macro : le F1 est dans la 3ᵉ colonne": r22["macro avg"]["precision"],
                                                       "c'est l'accuracy (qui vaut aussi le F1 micro) : on demande la ligne macro avg": r22["accuracy"]})
wb.record("3.22c", weighted_f1_22, decimals=3, mistakes={"c'est le F1 macro : on demande la ligne weighted avg": r22["macro avg"]["f1-score"],
                                                          "c'est l'accuracy (qui vaut aussi le recall pondéré) : on demande le F1 de la ligne weighted avg": r22["accuracy"]})
wb.record("3.22d", normalize_22, mistakes={"avec \"pred\", chaque COLONNE somme à 1 : la diagonale donne la precision de chaque espèce": "pred",
                                           "avec \"all\", toute la matrice somme à 1 : les cases sont des probabilités jointes": "all"})''',
       note="`normalize=\"true\"` divise chaque ligne (une vraie espèce) par son total : la diagonale donne le recall "
            "de chaque espèce, et l'on voit d'un coup d'œil que 27 % des Gentoo sont manqués. Le F1 macro (0,854) "
            "est sous le F1 pondéré (0,867) parce que le Chinstrap, l'espèce au plus petit F1, pèse autant que les "
            "autres dans la moyenne macro, mais peu dans la pondérée (68 manchots sur 342). `output_dict=True` sert "
            "à récupérer ces chiffres dans du code, pour un tableau de résultats ou un suivi d'expériences."),

    Ex("3.23", "🛠️", 2, 25, "Lire la documentation de sklearn.metrics",
       "trouver dans la documentation officielle le comportement exact d'une fonction avant de s'y fier.",
       "Ex 3.22 · fiche (🕰️, « Pour aller plus loin »)", thread="—", tracks="C",
       body=r"""Réflexe professionnel : avant de faire confiance à un chiffre, vérifie **ce que calcule exactement** la fonction. La documentation de chaque fonction de `sklearn.metrics` décrit ses paramètres ; lis-la pour la version du workbook (1.6) : sur le site de scikit-learn, choisis la version en haut de la page, ou, dans le notebook, `help(skm.precision_score)`.

Pour chaque question, **prévois** la réponse en lisant la documentation, sans exécuter l'appel ; écris-la dans la variable, puis exécute la vérification, qui compare ta prévision au comportement réel de scikit-learn.
a) `zero_division` : que renvoie `skm.precision_score([1, 1, 0], [0, 0, 0], zero_division=1)` ? (un nombre)
b) `labels` : que renvoie `skm.confusion_matrix(["a", "b", "c", "a"], ["a", "c", "b", "a"], labels=["a", "b"])` ? (une liste de listes) Que deviennent les échantillons dont une étiquette n'est pas dans `labels` ? Ta `confusion_matrix` fait autrement : comment, et pourquoi ?
c) `pos_label` : quelle erreur lève `skm.recall_score(["spam", "ham", "spam"], ["spam", "spam", "ham"])` ? (son nom, une chaîne comme `"KeyError"`)
d) `normalize` : avec `skm.confusion_matrix(y_true, y_pred, normalize="pred")`, qu'est-ce qui somme à 1 : les `"lignes"`, les `"colonnes"` ou `"tout"` ?
e) `sample_weight` : que renvoie `skm.accuracy_score([1, 0, 1], [1, 1, 1], sample_weight=[1, 1, 2])` ? (un nombre)
f) `average=None` : que renvoie `skm.precision_score(["b", "a", "c", "a"], ["b", "a", "a", "c"], average=None)` ? (une liste de nombres) Dans quel ordre sont rangées les classes ?

Dans ta cellule 📝, recopie pour chaque question la phrase de la documentation qui y répond.""",
       todo=r'''answer_23a = ...   # a) a number
answer_23b = ...   # b) a list of lists
answer_23c = ...   # c) the name of the error, a string
answer_23d = ...   # d) "lignes", "colonnes" or "tout"
answer_23e = ...   # e) a number
answer_23f = ...   # f) a list of numbers''',
       check=DOC_23,
       solution=r'''answer_23a = 1.0                  # no positive prediction: precision = 0/0, replaced by zero_division
answer_23b = [[2, 0], [0, 0]]     # the samples with a label outside `labels` are left out, silently
answer_23c = "ValueError"         # pos_label=1 (the default) is not one of the labels "spam" and "ham"
answer_23d = "colonnes"           # "pred": each column (a predicted class) is divided by its total
answer_23e = 0.75                 # (1 + 0 + 2) / (1 + 1 + 2): each sample counts with its weight
answer_23f = [0.5, 1.0, 0.0]      # one value per class, in the sorted order of the labels: a, b, c
''' + DOC_23,
       after=[("todo_md", "📝 **Les phrases de la documentation** (une par question) : …")],
       note="Trois comportements à retenir. `labels` peut **écarter des échantillons** sans prévenir : une matrice "
            "dont la somme n'est pas le nombre d'échantillons doit alerter (`mylearn` préfère lever une erreur). "
            "`zero_division` fixe la valeur d'une mesure indéfinie (0/0) ; sans lui, scikit-learn renvoie 0 avec un "
            "avertissement. `average=None` range les classes dans l'ordre **trié** des étiquettes (ou dans l'ordre "
            "de `labels`) : ne suppose jamais l'ordre d'apparition. Pour lire la documentation de la bonne version : "
            "le sélecteur de version du site, ou `help()` dans le notebook, qui montre toujours la version installée."),
])

# ---------------------------------------------------------------------------
# Part D: beyond the book: curves, averages, calibration, challenge (3.24 to 3.29)
# ---------------------------------------------------------------------------
ROC_PLOT_24 = r'''    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for measure in MEASURES:
        fpr, tpr, _ = mylearn.metrics.roc_curve(is_gentoo, measured[measure])
        ax.plot(fpr, tpr, label=measure)
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", label="random")
    ax.set(xlabel="false positive rate (FPR)", ylabel="recall (TPR)", title="Gentoo against the rest: ROC curves")
    ax.legend(loc="lower right")
    plt.show()'''

MODELS_26 = r'''rng_26 = np.random.default_rng(26)
y_26 = (rng_26.random(5000) < 0.04).astype(int)           # 1 = fraud, 0 = normal transaction
easy_26 = rng_26.random(5000) < 0.4                       # the frauds that model A recognises at once
score_a = np.where((y_26 == 1) & easy_26, rng_26.normal(4, 0.5, 5000), rng_26.normal(0, 1, 5000) + 0.8 * y_26)
score_b = rng_26.normal(0, 1, 5000) + 2.0 * y_26          # model B: every fraud a little higher
print(f"{y_26.sum()} frauds among {len(y_26)} transactions")'''

PR_PLOT_26 = r'''    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, score in [("A", score_a), ("B", score_b)]:
        precision, recall, _ = mylearn.metrics.precision_recall_curve(y_26, score)
        ax.plot(recall, precision, label=f"model {name}")
    ax.set(xlabel="recall", ylabel="precision", title="Two fraud models: precision-recall curves", ylim=(0, 1.02))
    ax.legend()
    plt.show()'''

CURVES_27 = r'''rng_27 = np.random.default_rng(27)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for share, colour in [(0.02, "C0"), (0.005, "C1")]:
    y = rng_27.random(200_000) < share                     # the positives of this population
    score = rng_27.normal(0, 1, 200_000) + 2.0 * y         # the same laws of scores in both populations
    fpr, tpr, _ = skm.roc_curve(y, score)
    precision, recall, _ = skm.precision_recall_curve(y, score)
    label = f"{share:.1%} of positives"
    axes[0].plot(fpr, tpr, color=colour, label=label)
    axes[1].plot(fpr, tpr, color=colour, label=label)
    axes[2].plot(recall, precision, color=colour, label=label)
axes[0].plot([0, 1], [0, 1], ls="--", color="0.5", label="random")
axes[0].set(title="ROC", xlabel="false positive rate (FPR)", ylabel="recall (TPR)")
axes[1].set(title="ROC, zoom on the small FPR", xlabel="false positive rate (FPR)", ylabel="recall (TPR)",
            xlim=(0, 0.05), xticks=np.arange(0, 0.051, 0.01), ylim=(0, 1.02))
axes[1].axhline(0.5, color="0.5", ls=":")
axes[2].set(title="precision-recall", xlabel="recall", ylabel="precision", ylim=(0, 1.02), yticks=np.arange(0, 1.01, 0.1))
axes[2].axvline(0.5, color="0.5", ls=":")
for ax in axes:
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower right" if ax is not axes[2] else "upper right")
plt.show()'''

WEATHER_28 = r'''rng_28 = np.random.default_rng(28)
chance_28 = rng_28.beta(1.2, 2.0, 3650)                  # the true chance of rain of each day (known to nobody)
rain = (rng_28.random(3650) < chance_28).astype(int)     # 1 = it rained that day
forecast_a = chance_28                                   # A announces the true chance: honest
forecast_b = 1 / (1 + np.exp(-2.5 * np.log(chance_28 / (1 - chance_28))))   # B pushes it towards 0 or 1
forecast_c = np.full(3650, 0.375)                        # C always announces the average chance of rain
print(f"{rain.sum()} rainy days out of {len(rain)}")'''

CALIBRATION_PLOT_28 = r'''    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", label="perfectly calibrated")
    for name, forecast in [("A", forecast_a), ("B", forecast_b), ("C", forecast_c)]:
        prob_true, prob_pred = mylearn.metrics.calibration_curve(rain, forecast, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=f"forecaster {name}")
    ax.set(xlabel="announced probability (mean of each bin)", ylabel="observed frequency of rain",
           title="Reliability diagram (10 bins)", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    plt.show()'''

TRANSACTIONS_29 = r'''def make_transactions(n, seed):
    """n transactions, about 1 % of frauds, with the scores of two models A and B."""
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < 0.01
    kind_2 = rng.random(n) < 0.06
    score_a = rng.normal(0, 1, n) + np.where(fraud & ~kind_2, 5.0, 0.0)
    score_b = rng.normal(0, 1, n) + 4.0 * fraud
    return pd.DataFrame({"score_a": score_a, "score_b": score_b, "fraud": fraud.astype(int)})


def evaluate_29(column, threshold, data):
    """(recall, precision, number of alerts) when we raise an alert for data[column] >= threshold."""
    alert = data[column].to_numpy() >= threshold
    fraud = data["fraud"].to_numpy() == 1
    caught = int(np.sum(alert & fraud))
    return caught / int(np.sum(fraud)), caught / max(int(np.sum(alert)), 1), int(np.sum(alert))


val_29 = make_transactions(50_000, seed=291)     # to choose the model and the threshold
test_29 = make_transactions(50_000, seed=391)    # only for the final grade: do not look at it to choose
print(f"validation: {val_29['fraud'].sum()} frauds · test: {test_29['fraud'].sum()} frauds (50 000 transactions each)")'''

GRADE_29 = r'''    column_29, threshold_29 = choose_29(val_29)
    recall_29, precision_29, alerts_29 = evaluate_29(column_29, threshold_29, test_29)
    print(f"your choice: {column_29}, threshold {threshold_29:.4f} · test set: {alerts_29} alerts, "
          f"recall {recall_29:.4f}, precision {precision_29:.4f}")
    verdict("3.29", recall_29 >= 0.99, "recall ≥ 0,99 sur le test : au moins 99 % des fraudes sont bloquées.",
            "recall < 0,99 sur le test : trop de fraudes passent ; prends plus de marge (ou vérifie le modèle).")
    verdict("3.29", precision_29 >= 0.10, "precision ≥ 0,10 sur le test : au plus 9 fausses alertes par fraude bloquée.",
            "precision < 0,10 sur le test : trop d'alertes ; ta marge est trop grande, ou le modèle n'est pas le bon.")
    held_29 = [evaluate_29(*choose_29(make_transactions(50_000, 1000 + 2 * k)), make_transactions(50_000, 1001 + 2 * k))
               for k in range(20)]
    print(f"for information, your method on 20 other (validation, test) pairs: recall ≥ 0.99 in "
          f"{sum(r >= 0.99 for r, _, _ in held_29)}, precision ≥ 0.10 in {sum(p >= 0.10 for _, p, _ in held_29)}, "
          f"both in {sum(r >= 0.99 and p >= 0.10 for r, p, _ in held_29)}")'''

PART_D = Part("D", "Au-delà du livre : courbes, moyennes, calibration, défi",
              "Sections « au-delà du livre » de la fiche. Tu programmes la courbe ROC et son aire (3.24), les moyennes "
              "sur plusieurs classes (3.25), la courbe precision-recall et l'average precision (3.26), puis la "
              "calibration (3.28) ; tu apprends à lire ces courbes quand la classe positive est rare (3.27), et tu "
              "finis par un défi : bloquer 99 % des fraudes avec le moins de fausses alertes possible (3.29).",
              exercises=[
    Ex("3.24", "🔨", 3, 40, "Courbe ROC et AUC",
       "programmer la courbe ROC et l'aire sous la courbe, et s'en servir pour comparer des scores.",
       "Ex 3.20 · fiche « Au-delà du livre (1) » (🧮 méthode des trapèzes)", thread="Penguins", tracks="R, M, C", mylearn="metrics.py",
       body=MYLEARN_SHORT + r"""

Écris trois fonctions (lis leurs docstrings) :
- `roc_curve(y_true, y_score, pos_label=1)` → `(fpr, tpr, thresholds)`. Convertis les entrées avec `np.asarray` (une Series pandas peut avoir un index quelconque). Trie les échantillons par score décroissant (`np.argsort(-scores, kind="mergesort")`), cumule les positifs et les négatifs rencontrés (`np.cumsum`), puis ne garde qu'un point par score distinct : des scores égaux forment un seul seuil (fiche, mini-exemple). Ajoute enfin le point de départ $(0, 0)$, de seuil `np.inf` ;
- `auc(x, y)` : l'aire par la méthode des trapèzes (fiche, 🧮), que les `x` soient croissants ou décroissants ;
- `roc_auc(y_true, y_score, pos_label=1)`, qui réutilise les deux autres.

Vérifications, pour « Gentoo contre le reste » (`is_gentoo`, comme en 3.20) :
a) l'AUC de chacune des quatre mesures de `MEASURES` prise comme score, calculée par ta fonction `roc_auc` (la cellule de vérification l'appelle) ;
b) `best_measure_24` : la mesure qui sépare le mieux les Gentoo des autres, si l'on a le droit de **retourner** un score (prendre son opposé) : son nom, une chaîne ;
c) `auc_minus_depth_24` : l'AUC de l'opposé de l'épaisseur du bec, `-measured["bill_depth_mm"]` (4 décimales), calculée avec ta fonction. Quel lien avec l'AUC de l'épaisseur elle-même ?

La vérification trace les courbes ROC des quatre mesures avec ta fonction, et contrôle sur la nageoire que l'AUC est la probabilité qu'un Gentoo tiré au hasard ait une nageoire plus longue qu'un autre manchot tiré au hasard, un ex-æquo comptant pour moitié (fiche). Puis les tests de ces trois fonctions.""",
       todo=r'''best_measure_24 = ...      # b) a column name
auc_minus_depth_24 = ...   # c)''',
       check=RELOAD + r'''with wb.attempt("3.24"):
    aucs_24 = [mylearn.metrics.roc_auc(is_gentoo, measured[measure]) for measure in MEASURES]
    wb.check("3.24a", aucs_24, computed=True)
    gentoo_fl, other_fl = flipper[is_gentoo == 1], flipper[is_gentoo == 0]
    pairs_24 = np.mean(gentoo_fl[:, None] > other_fl[None, :]) + 0.5 * np.mean(gentoo_fl[:, None] == other_fl[None, :])
    verdict("3.24", abs(aucs_24[2] - pairs_24) < 1e-9,
            "l'AUC de la nageoire est bien la part des paires (Gentoo, autre) bien rangées, ex-æquo comptés pour moitié.",
            "l'AUC de la nageoire devrait valoir la part des paires (Gentoo, autre) bien rangées : vérifie roc_curve "
            "(un point par score distinct) et auc.")
''' + ROC_PLOT_24 + r'''
    run_metrics_tests("test_roc_curve_ or test_auc_ or test_roc_auc_")
wb.check("3.24b", best_measure_24)
wb.check("3.24c", auc_minus_depth_24)''',
       solution=r'''aucs_24 = [mylearn.metrics.roc_auc(is_gentoo, measured[measure]) for measure in MEASURES]
print(dict(zip(MEASURES, np.round(aucs_24, 4))))
separation_24 = {measure: max(a, 1 - a) for measure, a in zip(MEASURES, aucs_24)}   # a score may be turned round
best_measure_24 = max(separation_24, key=separation_24.get)
auc_minus_depth_24 = mylearn.metrics.roc_auc(is_gentoo, -measured["bill_depth_mm"])
print(best_measure_24, auc_minus_depth_24, 1 - aucs_24[1])
with wb.attempt("3.24"):
''' + ROC_PLOT_24 + r'''
run_metrics_tests("test_roc_curve_ or test_auc_ or test_roc_auc_", impl="ref")''',
       record=r'''def untied_auc_24(score):
    """The classic slip: one ROC point per penguin instead of one per distinct score."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    positive = (is_gentoo == 1)[order]
    tpr = np.r_[0, np.cumsum(positive)] / positive.sum()
    fpr = np.r_[0, np.cumsum(~positive)] / (~positive).sum()
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


UNTIED_24 = "ta courbe a un point par manchot au lieu d'un point par score distinct : des scores égaux forment un seul seuil (fiche, mini-exemple)"
wb.record("3.24a", aucs_24, decimals=4, mistakes={UNTIED_24: [untied_auc_24(measured[measure]) for measure in MEASURES],
                                                  "tes AUC valent 1 − les bonnes : la courbe part de (0, 0) au seuil +inf, puis les seuils DÉCROISSENT (score >= seuil)": [1 - a for a in aucs_24]})
wb.record("3.24b", best_measure_24, mistakes={"retournée, l'épaisseur du bec sépare bien les Gentoo, mais une autre mesure fait mieux : compare les AUC (ou 1 − AUC)": "bill_depth_mm",
                                              "la masse sépare bien les Gentoo, mais une autre mesure fait mieux : compare les AUC": "body_mass_g"})
wb.record("3.24c", auc_minus_depth_24, decimals=4, mistakes={UNTIED_24: untied_auc_24(-measured["bill_depth_mm"]),
                                                             "c'est l'AUC de l'épaisseur elle-même : on demande celle de son OPPOSÉ (-measured[\"bill_depth_mm\"])": aucs_24[1]})''',
       note="La nageoire range presque parfaitement les Gentoo (AUC 0,996) : un Gentoo tiré au hasard a une nageoire "
            "plus longue qu'un autre manchot dans 99,6 % des paires, un ex-æquo comptant pour moitié. L'épaisseur du "
            "bec a une AUC de 0,012, bien pire "
            "que le hasard… parce que les Gentoo ont le bec le **moins** épais : retournée, elle donne "
            "$1 - 0{,}012 = 0{,}988$. Une AUC sous 0,5 signale un score rangé à l'envers, pas un score inutile. La "
            "masse (0,980) est bonne aussi ; la longueur du bec (0,794) sépare mal les Gentoo des Chinstrap, qui ont "
            "aussi un long bec. L'AUC juge le **classement** sur tous les seuils à la fois ; elle ne dit pas quel "
            "seuil choisir (3.20)."),

    Ex("3.25", "🔨", 3, 35, "Moyennes macro, micro et pondérée",
       "étendre precision, recall, F-beta et F1 à plusieurs classes, et voir quelle moyenne remarque une classe "
       "oubliée.",
       "Ex 3.16, Ex 3.6, Ex 3.15 · Ex 1.15 · fiche « Au-delà du livre : plusieurs classes »", thread="Penguins",
       tracks="M, C", mylearn="metrics.py",
       body=MYLEARN_SHORT + r"""

Complète `precision`, `recall`, `fbeta` et `f1` pour les autres valeurs d'`average` (docstrings) :
- `None` : une valeur par classe, chaque classe devenant tour à tour la classe positive (*one-vs-rest*), dans l'ordre trié des étiquettes présentes dans `y_true` **et** `y_pred`. Ta `confusion_matrix` (3.15) donne tout d'un coup : les TP sur sa diagonale, les FP dans le reste de chaque colonne, les FN dans le reste de chaque ligne ;
- `"macro"` : la moyenne simple des valeurs par classe ; `"weighted"` : leur moyenne pondérée par le support, le nombre de vrais échantillons de chaque classe (`np.average(..., weights=...)`) ;
- `"micro"` : additionne d'abord les TP, les FP et les FN de toutes les classes, puis calcule **une seule** mesure ;
- toute autre valeur d'`average` lève une `ValueError`.

Une classe jamais prédite a une precision 0/0 : c'est `zero_division` qui décide.

Deux classifieurs à trois classes sur les 342 manchots : les règles de la biologiste (`expert_pred`) et une règle à **deux** espèces, `two_species_pred` : « Gentoo » si la nageoire mesure au moins 210 mm, « Adelie » sinon ; elle n'annonce donc jamais « Chinstrap ». Vérifications (la cellule de vérification appelle tes fonctions) :
a) le F1 de chaque espèce pour la règle à deux espèces (`average=None`) ;
b) `[macro, micro, weighted]` du F1 des règles de la biologiste ;
c) la même liste pour la règle à deux espèces ;
d) `most_sensitive_25` : la moyenne qui baisse le plus d'un classifieur à l'autre : `"macro"`, `"micro"` ou `"weighted"`.

La vérification contrôle aussi que le F1 micro vaut l'accuracy (fiche), puis lance les tests des moyennes. Dans tes notes : quand préférer la moyenne macro ? (fiche, E4)""",
       given=r'''two_species_pred = np.where(flipper >= 210, "Gentoo", "Adelie")   # never answers "Chinstrap"''',
       todo=r'''most_sensitive_25 = ...   # d) "macro", "micro" or "weighted"''',
       check=RELOAD + r'''with wb.attempt("3.25"):
    mm = mylearn.metrics
    wb.check("3.25a", mm.f1(species, two_species_pred, average=None), computed=True)
    wb.check("3.25b", [mm.f1(species, expert_pred, average=average) for average in ("macro", "micro", "weighted")], computed=True)
    wb.check("3.25c", [mm.f1(species, two_species_pred, average=average) for average in ("macro", "micro", "weighted")], computed=True)
    verdict("3.25", abs(mm.f1(species, expert_pred, average="micro") - np.mean(species == expert_pred)) < 1e-12,
            "le F1 micro vaut l'accuracy : chaque erreur est à la fois un FP (pour la classe prédite) et un FN (pour la vraie).",
            "le F1 micro devrait valoir l'accuracy : additionne les TP, FP et FN de toutes les classes AVANT de calculer.")
    run_metrics_tests("(test_precision_ or test_recall_ or test_fbeta_ or test_f1_) and multiclass and not curve")
wb.check("3.25d", most_sensitive_25)''',
       solution=r'''mm = mylearn.metrics
f1_two_25 = mm.f1(species, two_species_pred, average=None)
averages_expert_25 = [mm.f1(species, expert_pred, average=average) for average in ("macro", "micro", "weighted")]
averages_two_25 = [mm.f1(species, two_species_pred, average=average) for average in ("macro", "micro", "weighted")]
drops_25 = dict(zip(("macro", "micro", "weighted"), np.subtract(averages_expert_25, averages_two_25)))
most_sensitive_25 = max(drops_25, key=drops_25.get)
print("F1 per species (Adelie, Chinstrap, Gentoo):", f1_two_25.round(3))
print("expert:", np.round(averages_expert_25, 3), "· two species:", np.round(averages_two_25, 3), "· drops:",
      {key: round(value, 3) for key, value in drops_25.items()})
run_metrics_tests("(test_precision_ or test_recall_ or test_fbeta_ or test_f1_) and multiclass and not curve", impl="ref")''',
       record=r'''support_25 = [int(np.sum(species == label)) for label in ("Adelie", "Chinstrap", "Gentoo")]
predicted_25 = [int(np.sum(expert_pred == label)) for label in ("Adelie", "Chinstrap", "Gentoo")]
f1_expert_25 = mm.f1(species, expert_pred, average=None)
wb.record("3.25a", f1_two_25, decimals=4, mistakes={"il manque une classe : les étiquettes sont celles de y_true ET de y_pred, même une classe jamais prédite": [f1_two_25[0], f1_two_25[2]]})
wb.record("3.25b", averages_expert_25, decimals=4, mistakes={"la moyenne pondérée utilise le SUPPORT de chaque classe (ses vrais échantillons, y_true), pas le nombre de ses prédictions": [averages_expert_25[0], averages_expert_25[1], float(np.average(f1_expert_25, weights=predicted_25))]})
wb.record("3.25c", averages_two_25, decimals=4, mistakes={"la moyenne macro compte TOUTES les classes, même celle qui n'est jamais prédite (son F1 vaut 0)": [(f1_two_25[0] + f1_two_25[2]) / 2, averages_two_25[1], averages_two_25[2]]})
wb.record("3.25d", most_sensitive_25, mistakes={"compare les baisses des trois moyennes : dans la pondérée, les Chinstrap ne pèsent que 68 manchots sur 342": "weighted",
                                                "compare les baisses des trois moyennes : la micro, qui vaut l'accuracy, ne regarde pas les classes une par une": "micro"})''',
       note="La règle à deux espèces est meilleure pour les Gentoo, mais elle ignore une espèce : son F1 Chinstrap "
            "vaut 0. La moyenne macro, où chaque espèce compte pour un tiers, s'effondre (de 0,854 à 0,570) ; la "
            "pondérée baisse moins (les Chinstrap ne font que 20 % des manchots) ; la micro, qui vaut l'accuracy, "
            "baisse le moins. Si chaque classe compte autant (une espèce rare à protéger, une maladie rare), c'est la "
            "macro qu'il faut suivre, et toujours avec les scores par classe."),

    Ex("3.26", "🔨", 3, 40, "Courbe precision-recall et average precision",
       "programmer la courbe precision-recall et l'average precision, puis comparer deux modèles selon le nombre "
       "d'alertes qu'on peut traiter.",
       "Ex 3.24 · fiche « Au-delà du livre (2) »", thread="synthétique", tracks="M, C", mylearn="metrics.py",
       body=MYLEARN_SHORT + r"""

Écris `precision_recall_curve(y_true, y_score, pos_label=1)` et `average_precision(y_true, y_score, pos_label=1)` (docstrings). Tu peux reprendre le cœur de ta `roc_curve` (tri par score décroissant, cumuls, un point par score distinct) : une fonction d'aide commune évite de l'écrire deux fois. Attention aux conventions de scikit-learn : les seuils sont rangés par ordre **croissant**, et un dernier point (precision 1, recall 0), sans seuil, ferme la courbe. L'AP est une somme **en escalier** (fiche), pas des trapèzes.

Une équipe antifraude compare deux modèles, A et B, sur 5 000 transactions dont environ 4 % de fraudes (`y_26`, `score_a`, `score_b`).
a) `[AP de A, AP de B]`, calculées par ta fonction `average_precision` (la cellule de vérification l'appelle) ;
b) `top50_26` : la precision de chaque modèle parmi ses 50 transactions les plus suspectes, c'est-à-dire ses 50 plus hauts scores (liste `[A, B]`, 2 décimales) ;
c) `top300_26` : la même chose parmi ses 300 plus hauts scores (3 décimales) ;
d) `choice_26` : l'équipe peut vérifier 300 alertes par jour : quel modèle choisir, `"A"` ou `"B"` ?
e) `auc_26` : `[AUC de A, AUC de B]` (2 décimales), avec ta fonction `roc_auc` de 3.24.

La vérification trace les deux courbes precision-recall avec ta fonction. Dans tes notes : l'AP, l'AUC et le choix de d désignent-ils le même modèle ? Que « voit » chacune de ces mesures ?""",
       given=MODELS_26,
       todo=r'''top50_26 = ...    # b) [A, B]
top300_26 = ...   # c) [A, B]
choice_26 = ...   # d) "A" or "B"
auc_26 = ...      # e) [A, B]''',
       check=RELOAD + r'''with wb.attempt("3.26"):
    wb.check("3.26a", [mylearn.metrics.average_precision(y_26, score) for score in (score_a, score_b)], computed=True)
''' + PR_PLOT_26 + r'''
    run_metrics_tests("test_precision_recall_curve_ or test_average_precision_")
wb.check("3.26b", top50_26)
wb.check("3.26c", top300_26)
wb.check("3.26d", choice_26)
wb.check("3.26e", auc_26)''',
       solution=r'''def top_precision(y, score, k):
    """Share of frauds among the k highest scores."""
    return float(np.mean(y[np.argsort(-score)[:k]]))


def top_recall(y, score, k):
    """Share of all the frauds found among the k highest scores."""
    return float(np.sum(y[np.argsort(-score)[:k]]) / np.sum(y))


ap_26 = [mylearn.metrics.average_precision(y_26, score) for score in (score_a, score_b)]
top50_26 = [top_precision(y_26, score, 50) for score in (score_a, score_b)]
top300_26 = [top_precision(y_26, score, 300) for score in (score_a, score_b)]
choice_26 = "AB"[int(np.argmax(top300_26))]
auc_26 = [mylearn.metrics.roc_auc(y_26, score) for score in (score_a, score_b)]
print("AP:", np.round(ap_26, 3), "· top 50:", top50_26, "· top 300:", top300_26, "· AUC:", np.round(auc_26, 3))
with wb.attempt("3.26"):
''' + PR_PLOT_26 + r'''
run_metrics_tests("test_precision_recall_curve_ or test_average_precision_", impl="ref")''',
       record=r'''trapezoid_26 = []
for score in (score_a, score_b):
    precision, recall, _ = mylearn.metrics.precision_recall_curve(y_26, score)
    trapezoid_26.append(mylearn.metrics.auc(recall, precision))
wb.record("3.26a", ap_26, decimals=4, mistakes={"tu as relié les points par des trapèzes : l'AP est une somme EN ESCALIER, chaque gain de recall multiplié par la precision atteinte": trapezoid_26,
                                                 "ce sont les AUC des courbes ROC : on demande les AP": auc_26})
wb.record("3.26b", top50_26, decimals=2, mistakes={"l'ordre demandé est [A, B]": top50_26[::-1],
                                                   "ce sont des recalls (part de TOUTES les fraudes trouvées) : on demande la part de fraudes parmi les 50 alertes": [top_recall(y_26, score, 50) for score in (score_a, score_b)]})
wb.record("3.26c", top300_26, decimals=3, mistakes={"l'ordre demandé est [A, B]": top300_26[::-1],
                                                     "ce sont des recalls (part de TOUTES les fraudes trouvées) : on demande la part de fraudes parmi les 300 alertes": [top_recall(y_26, score, 300) for score in (score_a, score_b)]})
wb.record("3.26d", choice_26, mistakes={"compare les precisions parmi 300 alertes (c), pas parmi 50 ni l'AP": "A"})
wb.record("3.26e", auc_26, decimals=2, mistakes={"ce sont les AP : on demande les AUC (roc_auc)": ap_26,
                                                 "l'ordre demandé est [A, B]": auc_26[::-1]})''',
       note="Le modèle A repère d'emblée 40 % des fraudes, avec des scores très hauts : ses 50 premières alertes "
            "sont toutes des fraudes, et son AP (0,502) dépasse un peu celle de B (0,481), car l'AP pèse surtout le "
            "haut de la liste. Mais les autres fraudes, A les voit mal : à 300 alertes, B en trouve davantage "
            "(precision 0,377 contre 0,320), et B a une bien meilleure AUC (0,914 contre 0,827), car l'AUC juge le "
            "classement dans toute la liste. Aucune mesure résumée ne choisit à ta place : le bon modèle dépend du "
            "**point de fonctionnement**, ici le nombre d'alertes que l'équipe peut traiter."),

    Ex("3.27", "📈", 2, 25, "ROC ou PR ? Lire les courbes d'un problème déséquilibré",
       "lire des courbes ROC et precision-recall quand la classe positive est rare, et traduire un taux de faux "
       "positifs en nombre de fausses alertes.",
       "Ex 3.24 · fiche « Au-delà du livre (1) et (2) » (🕰️ PR et AP)", thread="synthétique", tracks="R, M",
       body=r"""Un même modèle (les mêmes lois de scores pour les positifs et pour les négatifs) est évalué sur deux populations de 200 000 cas : l'une compte 2 % de positifs, l'autre 0,5 %. La cellule ci-dessous trace, avec scikit-learn, les courbes ROC (à gauche, puis un zoom sur les petits taux de faux positifs) et precision-recall (à droite). Lis les graphiques, sans recalculer :

a) `same_roc_27` : les deux courbes ROC sont-elles presque confondues ? (`True` ou `False`)
b) `fpr_at_half_27` : sur le zoom, le taux de faux positifs quand le recall (TPR) vaut 0,5 (2 décimales) ;
c) `precision_at_half_27` : sur les courbes precision-recall, la precision quand le recall vaut 0,5, pour 2 % puis pour 0,5 % de positifs (liste, 1 décimale) ;
d) `baseline_27` : la precision d'un classifieur qui répond au hasard, avec 0,5 % de positifs (3 décimales) : à quelle hauteur serait sa « courbe » PR ?
e) `curve_27` : laquelle des deux courbes, `"ROC"` ou `"PR"`, montre que la plupart des alertes seraient fausses avec 0,5 % de positifs ?

Dans tes notes, relie b et c : avec 0,5 % de positifs, combien y a-t-il de négatifs parmi les 200 000 cas, donc combien de fausses alertes au taux lu en b, pour combien de vrais positifs trouvés au recall 0,5 ? Retrouves-tu la precision lue en c ?""",
       given=CURVES_27,
       todo=r'''same_roc_27 = ...            # a) True or False
fpr_at_half_27 = ...         # b)
precision_at_half_27 = ...   # c) [2 %, 0.5 %]
baseline_27 = ...            # d)
curve_27 = ...               # e) "ROC" or "PR"''',
       check=r'''for letter, answer in zip("abcde", [same_roc_27, fpr_at_half_27, precision_at_half_27, baseline_27, curve_27]):
    wb.check(f"3.27{letter}", answer)''',
       solution=r'''same_roc_27 = True                   # the ROC only uses rates: it ignores the share of positives
fpr_at_half_27 = 0.02                 # read on the zoom, for both populations
precision_at_half_27 = [0.3, 0.1]     # read on the precision-recall curves
baseline_27 = 0.005                   # a random classifier: precision = share of positives
curve_27 = "PR"
negatives, positives = 0.995 * 200_000, 0.005 * 200_000
false_alerts, caught = fpr_at_half_27 * negatives, 0.5 * positives
print(f"{false_alerts:.0f} false alerts for {caught:.0f} positives found: precision ≈ {caught / (caught + false_alerts):.2f}")''',
       record=r'''wb.record("3.27a", same_roc_27, mistakes={"compare les deux courbes du graphique de gauche : l'écart entre elles se voit-il ?": False})
wb.record("3.27b", fpr_at_half_27, decimals=2, mistakes={"c'est le recall : on demande le taux de faux positifs (l'abscisse) au point de recall 0,5": 0.5,
                                                          "c'est la spécificité (1 − FPR) : on demande le FPR lui-même": 0.98})
wb.record("3.27c", precision_at_half_27, decimals=1, mistakes={"l'ordre demandé est [2 %, 0,5 %]": precision_at_half_27[::-1]})
wb.record("3.27d", baseline_27, decimals=3, mistakes={"0,5 est l'AUC de la diagonale de la ROC (le hasard), pas la hauteur de la ligne de base d'une courbe precision-recall (ou bien 0,5 % recopié sans le convertir en proportion) : relis ce que vaut la precision d'alertes tirées au hasard": 0.5,
                                                       "c'est l'autre population : on demande celle à 0,5 % de positifs": 0.02})
wb.record("3.27e", curve_27, mistakes={"sur la ROC, les deux populations donnent la même courbe : elle ne voit pas la part des positifs, donc pas les fausses alertes": "ROC"})''',
       note="La ROC n'utilise que des taux (le FPR divise par les négatifs, le recall par les positifs) : elle ne "
            "dépend pas de la prévalence, et les deux courbes se confondent. Mais un FPR de 2 % appliqué à 199 000 "
            "négatifs fait environ 4 000 fausses alertes, contre 500 positifs trouvés au recall 0,5 : une precision "
            "d'environ 0,1, que la courbe PR montre directement. Avec 2 % de positifs, la même lecture donne environ "
            "0,3. D'où le conseil de la fiche : quand la classe positive est rare et que la qualité des alertes "
            "compte, montrer la courbe PR (et sa ligne de base, la prévalence) à côté de la ROC."),

    Ex("3.28", "🔨", 3, 35, "Calibration : quand la météo annonce 70 %",
       "programmer le diagramme de fiabilité et le score de Brier, et distinguer calibration et qualité du "
       "classement.",
       "Ex 3.16, Ex 2.21 · fiche §3.1 et « Au-delà du livre (3) »", thread="synthétique", tracks="M, C", mylearn="metrics.py",
       body=MYLEARN_SHORT + r"""

Écris `calibration_curve(y_true, y_prob, n_bins=10)` et `brier_score(y_true, y_prob)` (docstrings). Pour trouver l'intervalle de chaque probabilité, **compare-la aux bords** `edges = np.linspace(0, 1, n_bins + 1)` : `np.searchsorted(edges[1:-1], p)` (ou `np.digitize(p, edges[1:-1], right=True)`) donne le numéro de l'intervalle, une valeur posée sur un bord intérieur allant dans l'intervalle du bas. Ne calcule pas ce numéro par `int(p * n_bins)` : une probabilité posée sur un bord intérieur (0,4 avec 10 intervalles) irait dans l'intervalle du haut au lieu de celui du bas, et $p = 1$ tomberait hors des intervalles ; un test le vérifie. `np.bincount(..., weights=..., minlength=n_bins)` fait ensuite les sommes par intervalle ; ne garde que les intervalles non vides.

Dix ans de prévisions de pluie (3 650 jours) : `rain` vaut 1 les jours de pluie, 0 sinon. Trois prévisionnistes annoncent chaque matin une probabilité de pluie, fabriquée par la cellule ci-dessous : A annonce la vraie probabilité du jour, B la pousse vers 0 ou vers 1, C annonce tous les jours la même valeur, la probabilité moyenne de pluie.
a) `[Brier de A, Brier de B, Brier de C]`, calculés par ta fonction `brier_score` (la cellule de vérification l'appelle) ;
b) `shows_70_28` : une application météo affiche les probabilités arrondies à 10 % : elle affiche « 70 % » quand la prévision est entre 0,65 (inclus) et 0,75 (exclu). Ces jours-là, quelle est la fréquence de la pluie, pour A puis pour B (liste, 3 décimales) ?
c) `auc_28` : `[AUC de A, AUC de B, AUC de C]` (3 décimales), avec ta fonction `roc_auc` de 3.24 ou `skm.roc_auc_score`.

La vérification trace le diagramme de fiabilité des trois prévisionnistes avec ta fonction. Dans tes notes : lequel est calibré ? Lequel est trop sûr de lui, et où le voit-on sur le diagramme ? Lequel est calibré mais inutile, et pourquoi ?""",
       given=WEATHER_28,
       todo=r'''shows_70_28 = ...   # b) [A, B]
auc_28 = ...        # c) [A, B, C]''',
       check=RELOAD + r'''with wb.attempt("3.28"):
    wb.check("3.28a", [mylearn.metrics.brier_score(rain, forecast) for forecast in (forecast_a, forecast_b, forecast_c)], computed=True)
''' + CALIBRATION_PLOT_28 + r'''
    run_metrics_tests("test_calibration_curve_ or test_brier_score_")
wb.check("3.28b", shows_70_28)
wb.check("3.28c", auc_28)''',
       solution=r'''brier_28 = [mylearn.metrics.brier_score(rain, forecast) for forecast in (forecast_a, forecast_b, forecast_c)]
shows_70_28 = [float(rain[(forecast >= 0.65) & (forecast < 0.75)].mean()) for forecast in (forecast_a, forecast_b)]
auc_28 = [mylearn.metrics.roc_auc(rain, forecast) for forecast in (forecast_a, forecast_b, forecast_c)]
print("Brier:", np.round(brier_28, 3), "· rain when the app shows 70 %:", np.round(shows_70_28, 3), "· AUC:", np.round(auc_28, 3))
for name, forecast in [("A", forecast_a), ("B", forecast_b)]:
    prob_true, prob_pred = mylearn.metrics.calibration_curve(rain, forecast, n_bins=10)
    print(name, "announced:", prob_pred.round(2), "\n  observed:", prob_true.round(2))
with wb.attempt("3.28"):
''' + CALIBRATION_PLOT_28 + r'''
run_metrics_tests("test_calibration_curve_ or test_brier_score_", impl="ref")''',
       record=r'''wb.record("3.28a", brier_28, decimals=4, mistakes={"ce sont les racines des scores de Brier : le score de Brier est la moyenne des CARRÉS des écarts, sans racine": list(np.sqrt(brier_28)),
                                                     "tu as fait la moyenne des écarts sans les élever au carré": [float(np.mean(np.abs(forecast - rain))) for forecast in (forecast_a, forecast_b, forecast_c)]})
announced_28 = [float(forecast[(forecast >= 0.65) & (forecast < 0.75)].mean()) for forecast in (forecast_a, forecast_b)]
wb.record("3.28b", shows_70_28, decimals=3, mistakes={"l'ordre demandé est [A, B]": shows_70_28[::-1],
                                                      "ce sont les probabilités annoncées ces jours-là (en moyenne) : on demande la fréquence OBSERVÉE de la pluie": announced_28})
wb.record("3.28c", auc_28, decimals=3, mistakes={"ce sont les scores de Brier : on demande les AUC": brier_28})''',
       note="A suit la diagonale : quand l'application affiche « 70 % », il pleut environ 71 % de ces jours-là. B "
            "range les jours exactement dans le même ordre que A (même AUC, 0,782), mais il exagère : les jours où "
            "il affiche « 70 % », il ne pleut que 58 % du temps, et sa courbe est plus plate que la diagonale (trop "
            "haute à gauche, trop basse à droite). C est parfaitement calibré (un seul point, sur la diagonale) mais "
            "inutile : AUC 0,5, il ne distingue aucun jour d'un autre. Le score de Brier les classe A (0,180), B "
            "(0,200), C (0,234) : il mélange calibration et classement. B se corrigerait par une recalibration "
            "(fiche, 🕰️), sans rien changer à son classement."),

    Ex("3.29", "🏆", 3, 45, "Recall ≥ 0,99 au meilleur prix",
       "choisir, sur un jeu de validation seulement, un modèle et un seuil qui bloquent 99 % des fraudes du jeu de "
       "test avec le moins de fausses alertes possible.",
       "Ex 3.26, Ex 3.20 · ch. 2 (percentiles, bootstrap) · fiche §3.7.4, « Au-delà du livre (2) », pièges",
       thread="synthétique", tracks="C",
       body=r"""**Défi.** Une banque exige que son système bloque **au moins 99 %** des fraudes (recall ≥ 0,99). Chaque alerte coûte un appel au client : elle veut aussi **au moins une vraie fraude pour 10 alertes** (precision ≥ 0,10).

Deux modèles, A et B, donnent un score à chaque transaction (1 % de fraudes). Tu disposes d'un jeu de **validation** (`val_29` : 50 000 transactions, colonnes `score_a`, `score_b` et `fraud`) pour faire tes choix, et d'un jeu de **test** (`test_29` : 50 000 autres transactions) qui ne sert qu'à la note finale. Ne le regarde pas pour choisir : régler un seuil sur le test, c'est tricher (fiche, pièges).

Écris `choose_29(val)`, qui reçoit un DataFrame de validation et renvoie `(column, threshold)` : la colonne de score retenue (`"score_a"` ou `"score_b"`) et le seuil (une alerte quand score ≥ seuil). La vérification applique ton choix au jeu de test : il faut un recall ≥ 0,99 **et** une precision ≥ 0,10.

Deux pièges t'attendent :
1. le meilleur modèle selon l'AP n'est pas forcément le meilleur quand il faut attraper presque **toutes** les fraudes : regarde les scores des fraudes les plus basses de chaque modèle ;
2. un seuil qui donne exactement 99 % de recall sur la validation en donne souvent moins sur le test : les fraudes du test ne sont pas celles de la validation, et le 1 % le plus bas d'environ 500 fraudes varie beaucoup d'un tirage à l'autre. Prends une **marge**, mais pas trop grande, sinon la precision s'effondre. Le bootstrap du ch. 2 peut t'aider à mesurer cette variabilité.

Pour aller plus loin, la vérification rejoue ta méthode sur 20 autres couples (validation, test) : combien de fois tient-elle ses deux promesses ?""",
       given=TRANSACTIONS_29,
       todo=r'''def choose_29(val):
    """(score column, threshold) chosen with the validation DataFrame only."""
    raise NotImplementedError("choose_29() is not written yet")''',
       check=r'''with wb.attempt("3.29"):
''' + GRADE_29,
       solution=r'''frauds_val = val_29.loc[val_29["fraud"] == 1]
for column in ["score_a", "score_b"]:
    ap = mylearn.metrics.average_precision(val_29["fraud"], val_29[column])
    lowest = np.sort(frauds_val[column].to_numpy())[:6].round(2)
    print(f"{column}: AP {ap:.3f} · the 6 lowest fraud scores: {lowest}")

exactly_99 = np.sort(frauds_val["score_b"].to_numpy())[int(0.01 * len(frauds_val))]   # at most 1 % of the frauds below
for name, (column, threshold) in {"exactly 99 %, no margin": ("score_b", exactly_99),
                                  "0.5 % quantile": ("score_b", np.quantile(frauds_val["score_b"], 0.005)),
                                  "model A, 0.5 % quantile": ("score_a", np.quantile(frauds_val["score_a"], 0.005))}.items():
    r, p, n = evaluate_29(column, threshold, val_29)
    print(f"{name:<24} validation: recall {r:.4f}, precision {p:.4f}, {n} alerts")


def choose_29(val):
    """(score column, threshold) chosen with the validation DataFrame only."""
    scores = val.loc[val["fraud"] == 1, "score_b"].to_numpy()    # model B: every fraud gets a higher score
    return "score_b", float(np.quantile(scores, 0.005))           # a margin: the 0.5 % quantile, not the 1 % one


with wb.attempt("3.29"):
''' + GRADE_29,
       note="Le modèle A a la meilleure AP, mais 6 % des fraudes (un nouveau type) ont chez lui des scores de "
            "transactions normales : pour en bloquer 99 %, il faudrait alerter sur presque tout (precision ≈ 0,01). "
            "B relève le score de **toutes** les fraudes. Avec B, un seuil qui garde exactement 99 % des fraudes de "
            "validation n'en garde que 98,8 % sur le test : il faut une marge. Le quantile à 0,5 % "
            "tient ici (recall 0,996, precision 0,115), mais seulement dans 13 des 20 autres couples. Plus "
            "stable : estimer le bas de la distribution des scores de fraudes à partir de **toutes** les fraudes "
            "(moyenne − 2,58 écarts-types, si cette distribution est à peu près normale : à vérifier sur un "
            "histogramme), qui tient 17 fois sur 20. Garantir 99 % à coup sûr est impossible avec 500 fraudes : on "
            "annonce une garantie avec sa marge d'erreur."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 3.1, 3.2, 3.3, 3.5, 3.6, 3.7 | vérifier tes exercices papier ✏️ | ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")          # "P(A|B)" must not split the table row
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if kind == "exercise":
        title = "# 3 · Probabilités et mesure de la qualité — notebook d'exercices"
        how = ("La partie 0 vérifie tes exercices papier. Chaque exercice de code : un énoncé, une cellule à "
               "compléter (les `...` et les `raise NotImplementedError`), puis une cellule de vérification "
               "(`wb.check`, ou les tests de ta librairie `mylearn`). « Exécuter tout » va jusqu'au bout même si rien "
               "n'est rempli : ce qui n'est pas fait affiche ⏳. Les questions « pourquoi ? » qui n'ont pas de "
               "cellule 📝 se notent dans la section « Notes sur le notebook » de ta copie de `06_mes_reponses.md`. "
               "Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch03_probabilites/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 3`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 3 · Probabilités et mesure de la qualité — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Calculer des probabilités simples, conditionnelles, jointes et marginales, et ne pas confondre "
               "$P(A \\mid B)$ avec $P(B \\mid A)$.\n"
               "- Construire une matrice de confusion et en tirer accuracy, precision, recall, F1 et les autres "
               "mesures, avec les conventions de scikit-learn.\n"
               "- Choisir une mesure et un seuil selon le coût des erreurs et la prévalence.\n"
               "- Évaluer des scores sur tous les seuils (courbes ROC et precision-recall) et vérifier des "
               "probabilités (calibration).\n\n"
               "**Rappel express.** $P(A \\mid B) = \\frac{P(A, B)}{P(B)}$ ; $P(A, B) = P(A \\mid B)\\,P(B)$ ; "
               "matrice de confusion de scikit-learn pour des étiquettes 0/1 : `[[TN, FP], [FN, TP]]` ; "
               "accuracy $= \\frac{TP + TN}{n}$ ; precision $= \\frac{TP}{TP + FP}$ ; recall $= \\frac{TP}{TP + FN}$ ; "
               "spécificité $= \\frac{TN}{TN + FP}$ ; $F_1 = \\frac{2\\,TP}{2\\,TP + FP + FN}$. En Python : "
               "`pd.crosstab`, `sklearn.metrics.confusion_matrix`, `classification_report`, `roc_curve`, "
               "`precision_recall_curve`.")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu calculer une probabilité conditionnelle dans une table, et dire laquelle de "
               "$P(A \\mid B)$ ou $P(B \\mid A)$ répond à une question ?\n"
               "2. Sais-tu construire une matrice de confusion, en tirer les mesures, et choisir une mesure et un seuil "
               "selon le coût des erreurs ?\n"
               "3. Sais-tu expliquer pourquoi, quand la maladie est rare, la plupart des résultats positifs d'un très bon "
               "test peuvent être faux, et quand montrer une courbe precision-recall plutôt qu'une ROC ?\n\n"
               "**Pour aller plus loin** : les pages du *Machine Learning Crash Course* de Google et le guide "
               "« Metrics and scoring » de scikit-learn, cités dans la fiche ; l'article de Fawcett (3.11). La suite : "
               "le ch. 4 (règle de Bayes), qui passe de $P(\\text{positif} \\mid \\text{malade})$ à "
               "$P(\\text{malade} \\mid \\text{positif})$ ; `mylearn.metrics` servira dans tous les chapitres suivants, "
               "dès le ch. 7 (classification) et le ch. 8 (choisir un seuil sur un jeu de validation).")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT, PAPER_INTRO)
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
