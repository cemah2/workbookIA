#!/usr/bin/env python
"""Build the two notebooks of chapter 3 from a single source (used by Claude).

    python tools/chapters/build_ch03.py
    python tools/run_all_notebooks.py chapitres/ch03_probabilites/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch03_probabilites/03_notebook.ipynb

Part 0 checks the ✏️ paper exercises (3.1, 3.2, 3.3, 3.5, 3.6, 3.7). Parts A to D
(exercises 3.12 to 3.29, mylearn.metrics) come with the second generation session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, badge, md, paper_cells, part_cells, setup_cell,  # noqa: E402
                         write_notebook, Paper)

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
# Parts A to D (exercises 3.12 to 3.29): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "3.12 à 3.14", "probabilités par les aires et les tables : fléchettes simulées, deux disques, pd.crosstab"),
    ("B", "3.15 à 3.19", "la matrice de confusion et les mesures binaires dans mylearn.metrics"),
    ("C", "3.20 à 3.23", "seuil de décision, prévalence, outils et documentation de scikit-learn"),
    ("D", "3.24 à 3.29", "au-delà du livre : courbes ROC et precision-recall, moyennes sur plusieurs classes, "
                         "calibration, défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 3.1, 3.2, 3.3, 3.5, 3.6, 3.7 | vérifier tes exercices papier ✏️ | ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {ex.title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |" for key, ids, title in NEXT_SESSION]
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
    later = ("" if PARTS else
             "\n\n*Les exercices de code 3.12 à 3.29 (parties A à D, ta librairie `mylearn.metrics`) seront ajoutés à "
             "la prochaine session de génération ; relance alors `python tools/start_chapter.py 3` pour obtenir le "
             "notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu calculer une probabilité conditionnelle dans une table, et dire laquelle de "
               "$P(A \\mid B)$ ou $P(B \\mid A)$ répond à une question ?\n"
               "2. Sais-tu construire une matrice de confusion et choisir entre precision et recall selon le coût "
               "des erreurs ?\n"
               "3. Sais-tu expliquer pourquoi un bon test donne surtout des faux positifs quand la maladie est rare ?\n\n"
               "**Pour aller plus loin** : les pages du *Machine Learning Crash Course* de Google et le guide "
               "« Metrics and scoring » de scikit-learn, cités dans la fiche. La suite : le ch. 4 (règle de Bayes), "
               "qui passe de $P(\\text{positif} \\mid \\text{malade})$ à $P(\\text{malade} \\mid \\text{positif})$ ; "
               "`mylearn.metrics` servira dans tous les chapitres suivants." + later)]


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
