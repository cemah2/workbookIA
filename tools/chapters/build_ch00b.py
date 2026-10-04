#!/usr/bin/env python
"""Build the two notebooks of chapter 0B from a single source (used by Claude).

    python tools/chapters/build_ch00b.py
    python tools/run_all_notebooks.py chapitres/ch00b_maths/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch00b_maths/03_notebook.ipynb

Part 0 checks the ✏️ paper exercises (0B.1 to 0B.27); parts A to D are the notebook
exercises 0B.33 to 0B.54 (mylearn.linalg_basics in part B).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "0B"
FOLDER = "chapitres/ch00b_maths"

# ---------------------------------------------------------------------------
# Part 0: checks of the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER = [
    Paper("0B.1", "Puissances, racines et notation scientifique sans calculatrice", [
        ("a", "2^5 x 2^3", "2 ** 5 * 2 ** 3", 'mistakes={"on ADDITIONNE les exposants : 2^(5+3)": 2 ** 15}'),
        ("b", "(10^2)^3 / 10^4", "(10 ** 2) ** 3 / 10 ** 4",
         'decimals=0, mistakes={"(10^2)^3 = 10^6 : on MULTIPLIE les exposants": 10}'),
        ("c", "5^(-2), as a decimal number", "5 ** -2",
         'decimals=2, mistakes={"un exposant négatif donne un inverse, pas un nombre négatif": -25}'),
        ("d", "sqrt(49 x 16)", "math.sqrt(49 * 16)", "decimals=0"),
        ("e", "8^(2/3)", "round(8 ** (2 / 3))", 'mistakes={"8^(1/3) = 2 (racine cubique), puis 2^2": 2}'),
        ("f", "(3.2 x 10^5) x (2 x 10^-3), as a decimal number", "3.2e5 * 2e-3", "decimals=0"),
        ("g", "the exponent k such that 0.00056 = 5.6 x 10^k", "math.floor(math.log10(0.00056))",
         'mistakes={"pour un nombre plus petit que 1, l\'exposant est négatif": 4}'),
        ("h", "the exponent k such that 2^30 is about 10^k", "round(30 * math.log10(2))",
         'mistakes={"on cherche l\'exposant de 10, pas celui de 2 : 2^30 = (2^10)^3 et 2^10 ≈ 10^3": 30}'),
    ]),
    Paper("0B.2", "Valeur absolue, partie entière et signe : tableau de valeurs", [
        ("a", "|-7| + |3 - 5|", "abs(-7) + abs(3 - 5)", 'mistakes={"une valeur absolue est toujours positive : |3 − 5| = 2": 5}'),
        ("b", "floor(3.7)", "math.floor(3.7)", ""),
        ("c", "floor(-3.7)", "math.floor(-3.7)", 'mistakes={"le plancher va vers le BAS (vers −∞) : prends l\'entier juste en dessous de −3,7": -3}'),
        ("d", "ceil(-3.7)", "math.ceil(-3.7)", 'mistakes={"le plafond va vers le HAUT : prends l\'entier juste au-dessus de −3,7": -4}'),
        ("e", "ceil(2.01)", "math.ceil(2.01)", ""),
        ("f", "sign(-0.5) + sign(0) + sign(12)", "int(np.sign(-0.5) + np.sign(0) + np.sign(12))",
         'mistakes={"sign(0) vaut 0": 1}'),
        ("g", "lower bound of the x such that |x - 4| <= 1.5", "4 - 1.5", "decimals=1"),
        ("h", "upper bound", "4 + 1.5", "decimals=1"),
        ("i", "number of integers x such that |x| < 3", "sum(1 for x in range(-10, 11) if abs(x) < 3)",
         'mistakes={"inégalité stricte : 3 et −3 sont exclus": 7, "les entiers négatifs comptent aussi": 3}'),
    ]),
    Paper("0B.3", "Lire et calculer des Σ et des Π", [
        ("a", "sum of i for i = 1..5", "sum(range(1, 6))", ""),
        ("b", "sum of (2i - 1) for i = 1..4", "sum(2 * i - 1 for i in range(1, 5))", ""),
        ("c", "sum of 3^k for k = 0..3", "sum(3 ** k for k in range(4))",
         'mistakes={"k commence à 0 : le terme 3^0 = 1 compte aussi": 39}'),
        ("d", "product of (i + 1) for i = 1..4", "math.prod(i + 1 for i in range(1, 5))", ""),
        ("e", "sum of x_i^2", "sum(v ** 2 for v in x3)", 'mistakes={"le carré de −1 vaut +1": 28}'),
        ("f", "sum of (3 x_i + 1)", "sum(3 * v + 1 for v in x3)",
         'mistakes={"le +1 est ajouté à chacun des 4 termes, pas une seule fois": 25}'),
        ("g", "sum of 5 for i = 1..100", "5 * 100", ""),
        ("h", "sum of i x_i for i = 2..4", "sum(i * x3[i - 1] for i in range(2, 5))",
         'mistakes={"la somme commence à i = 2 : le terme 1 × x_1 n\'y est pas": 24}'),
    ]),
    Paper("0B.4", "Moyenne pondérée, somme pondérée et moyenne mobile à la main", [
        ("a", "the mean of 4, 8, 6, 10, 2", "np.mean([4, 8, 6, 10, 2])", "decimals=0"),
        ("b", "the weighted mean", "np.average([14, 8, 11], weights=[3, 1, 2])",
         'decimals=0, mistakes={"divise par la somme des coefficients (6), pas par le nombre de notes": 24}'),
        ("c", "the output z of the neuron", "0.2 * 10 - 0.5 * 4 + 1 * 3 - 1",
         'decimals=1, mistakes={"n\'oublie pas le biais b = −1": 3}'),
        ("d", "the list [m3, m4, m5, m6], 2 decimals", "[round(np.mean(s4[t - 3:t]), 2) for t in range(3, 7)]", "decimals=2"),
        ("e", "number of values of the moving average", "50 - 7 + 1",
         'mistakes={"la première moyenne mobile existe à t = k : il y a n − k + 1 valeurs": 43}'),
    ]),
    Paper("0B.5", "Ensembles : union, intersection, complémentaire et cardinal", [
        ("a", "|A ∩ B|", "len(A5 & B5)", ""),
        ("b", "|A ∪ B|", "len(A5 | B5)", 'mistakes={"retire l\'intersection, comptée deux fois": 10}'),
        ("c", "|complement of A|", "len(omega5 - A5)", ""),
        ("d", "|complement of A ∩ complement of B|", "len((omega5 - A5) & (omega5 - B5))",
         'mistakes={"ni multiple de 2 NI multiple de 3 : c\'est le complémentaire de A ∪ B": 8, "« ni… ni… » exclut aussi les nombres qui ne sont que dans A, ou que dans B": 10}'),
        ("e", "the set A ∩ C, as a Python set {...}", "A5 & C5", "ordered=False"),
        ("f", "|A ∪ B ∪ C|", "len(A5 | B5 | C5)", 'mistakes={"2 et 3 sont déjà dans A ∪ B : seul 1 est nouveau": 11}'),
    ]),
    Paper("0B.6", "Droites et paraboles : pente, ordonnée à l'origine, racines, sommet", [
        ("a", "the slope of d", "(-3 - 5) / (3 - (-1))",
         'decimals=0, mistakes={"pente = Δy / Δx, avec les points pris dans le même ordre en haut et en bas": 2}'),
        ("b", "its intercept", "5 - (-2) * (-1)", ""),
        ("c", "its root", "-3 / -2", "decimals=1"),
        ("d", "the discriminant of f", "(-8) ** 2 - 4 * 2 * 6", 'mistakes={"Δ = b² − 4ac avec b = −8 : (−8)² = 64": -112}'),
        ("e", "the smaller root", "(8 - math.sqrt(16)) / (2 * 2)", "decimals=0"),
        ("f", "the larger root", "(8 + math.sqrt(16)) / (2 * 2)", "decimals=0"),
        ("g", "the x of the vertex", "8 / (2 * 2)", 'decimals=0, mistakes={"le sommet est en −b/(2a), avec b = −8": -2}'),
        ("h", "the minimum value of f", "2 * 2 ** 2 - 8 * 2 + 6", ""),
    ]),
    Paper("0B.7", "Cosinus : cercle, période et planning en cosinus", [
        ("a", "60 degrees in radians, 2 decimals", "math.radians(60)", 'decimals=2, mistakes={"π rad = 180°, donc 60° = π/3": 60 / 180}'),
        ("b", "cos(pi)", "math.cos(math.pi)", "decimals=1"),
        ("c", "cos(2 pi / 3)", "math.cos(2 * math.pi / 3)", "decimals=1"),
        ("d", "cos(-pi / 3)", "math.cos(-math.pi / 3)", 'decimals=1, mistakes={"le cosinus est pair : cos(−x) = cos(x)": -0.5}'),
        ("e", "cos(4 pi)", "math.cos(4 * math.pi)", 'decimals=1, mistakes={"cos a une période de 2π : 4π, c\'est deux tours complets": -1}'),
        ("f", "the cosine factor at t = 50", "0.5 * (1 + math.cos(math.pi * 50 / 100))", "decimals=2"),
        ("g", "the factor at t = 25, 2 decimals", "0.5 * (1 + math.cos(math.pi * 25 / 100))", "decimals=2"),
        ("h", "the learning rate at t = 75, 4 decimals", "0.01 * 0.5 * (1 + math.cos(math.pi * 75 / 100))",
         'decimals=4, mistakes={"multiplie le facteur par le learning rate maximal 0,01": 0.5 * (1 + math.cos(math.pi * 75 / 100)), "multiplie le facteur (≈ 0,146) par le learning rate maximal 0,01": 0.146, "multiplie le facteur (≈ 0,1465) par le learning rate maximal 0,01": 0.1465, "cos(3π/4) est négatif : cos(π − x) = −cos x": 0.0085}'),
    ]),
    Paper("0B.8", "Vecteurs : somme, multiple, norme et distance entre deux manchots", [
        ("a", "p1 + p2, a list", "p1 + p2", ""),
        ("b", "(p1 + p2) / 2, a list", "(p1 + p2) / 2", "decimals=1"),
        ("c", "the L2 norm of u", "np.linalg.norm(u8)", 'decimals=1, mistakes={"7 est la norme L1 (somme des valeurs absolues) ; L2 : racine de la somme des carrés": 7}'),
        ("d", "the L1 norm of u", "int(np.abs(u8).sum())", 'mistakes={"L1 : somme des VALEURS ABSOLUES": -1}'),
        ("e", "the L-infinity norm of u", "int(np.abs(u8).max())", 'mistakes={"L∞ : la plus grande valeur ABSOLUE": 3}'),
        ("f", "the distance between p1 and p2", "np.linalg.norm(p1 - p2)", "decimals=1"),
        ("g", "the same distance with the flipper in cm, 2 decimals",
         "np.linalg.norm(np.array([40, 19]) - np.array([48, 19.6]))", "decimals=2"),
        ("h", "the unit vector of u, a list", "u8 / np.linalg.norm(u8)", "decimals=1"),
    ]),
    Paper("0B.9", "Transposée et produit matrice-vecteur : deux lectures", [
        ("a", "the shape of A^T, a tuple", "A9.T.shape", 'mistakes={"la transposée échange lignes et colonnes": (2, 3)}'),
        ("b", "(A^T)_31", "A9.T[2, 0]", ""),
        ("c", "A v, a list", "A9 @ v9", ""),
        ("d", "A^T t with t = (1, -1), a list", "A9.T @ np.array([1, -1])", ""),
        ("e", "is A s defined for s = (1, 2)? True or False", "A9.shape[1] == 2", ""),
        ("f", "the predictions X w + b, a list", "X9 @ w9 + 1", 'decimals=1, mistakes={"n\'oublie pas le biais b = 1": X9 @ w9}'),
    ]),
    Paper("0B.10", "Taux d'accroissement : de la sécante à la tangente", [
        ("a", "the rate for h = 1", "(f10(3) - f10(2)) / 1", "decimals=0"),
        ("b", "the rate for h = 0.1", "(f10(2.1) - f10(2)) / 0.1", "decimals=1"),
        ("c", "the rate for h = 0.01", "(f10(2.01) - f10(2)) / 0.01", "decimals=2"),
        ("d", "f'(2)", "2 * 2 + 1", 'mistakes={"développe f(2 + h) − f(2), divise par h, puis fais tendre h vers 0": 4}'),
        ("e", "the intercept of the tangent at 2", "f10(2) - 5 * 2", ""),
        ("f", "the linear approximation of f(2.05) given by the tangent (exact value)", "f10(2) + 5 * 0.05",
         'decimals=4, mistakes={"c\'est la vraie valeur f(2,05) : on demande l\'approximation donnée par la tangente": f10(2.05)}'),
    ]),
    Paper("0B.11", "Probabilités : issues, complémentaire, union", [
        ("a", "P(sum = 8), 3 decimals", "prob(lambda d: sum(d) == 8)", "decimals=3"),
        ("b", "P(double)", "prob(lambda d: d[0] == d[1])", "decimals=3"),
        ("c", "P(at least one 1)", "prob(lambda d: 1 in d)",
         'decimals=3, mistakes={"au moins un 1 = 1 − P(aucun 1) = 1 − (5/6)²": 2 / 6}'),
        ("d", "P(sum = 8 or double)", "prob(lambda d: sum(d) == 8 or d[0] == d[1])",
         'decimals=3, mistakes={"(4, 4) est à la fois un double et une somme 8 : ne le compte qu\'une fois": 11 / 36}'),
        ("e", "P(sum >= 10)", "prob(lambda d: sum(d) >= 10)", "decimals=3"),
        ("f", "P(heart or face card)", "(13 + 12 - 3) / 52",
         'decimals=3, mistakes={"les 3 figures de cœur sont comptées deux fois : retire-les une fois": 25 / 52}'),
    ]),
    Paper("0B.12", "Dénombrer : choix successifs, factorielle et C(n, k)", [
        ("a", "number of 4-digit codes", "10 ** 4", ""),
        ("b", "number of 4-digit codes with all digits different", "10 * 9 * 8 * 7",
         'mistakes={"tous différents : 10 choix, puis 9, puis 8, puis 7": 10 * 9 * 8}'),
        ("c", "6!", "math.factorial(6)", ""),
        ("d", "C(6, 2)", "math.comb(6, 2)", 'mistakes={"C(n, k) ne tient pas compte de l\'ordre : divise par 2!": 30}'),
        ("e", "C(8, 3)", "math.comb(8, 3)", ""),
        ("f", "number of one-vs-one classifiers for 5 classes", "math.comb(5, 2)", ""),
        ("g", "number of subsets of 5 features", "2 ** 5", 'mistakes={"n\'oublie ni l\'ensemble vide ni l\'ensemble complet": 30}'),
        ("h", "number of committees of 3 with a president", "10 * math.comb(9, 2)",
         'mistakes={"le président est un rôle à part : choisis-le, puis les deux autres membres": math.comb(10, 3)}'),
    ]),
    Paper("0B.13", "Suites géométriques : ce qui fond, ce qui explose", [
        ("a", "u_5", "1000 * 0.5 ** 5", 'decimals=2, mistakes={"u_5 = u_0 × q^5 (5 multiplications depuis u_0)": 1000 * 0.5 ** 4}'),
        ("b", "0.9^20, 3 decimals", "0.9 ** 20", "decimals=3"),
        ("c", "1.1^10, 2 decimals", "1.1 ** 10", 'decimals=2, mistakes={"1,1^10 n\'est pas 1 + 10 × 0,1 : les intérêts se cumulent": 2.0}'),
        ("d", "sum of 0.5^k for k = 0..9, 3 decimals", "(1 - 0.5 ** 10) / (1 - 0.5)",
         'decimals=3, mistakes={"de k = 0 à 9, il y a 10 termes : q^n avec n = 10": (1 - 0.5 ** 9) / (1 - 0.5)}'),
        ("e", "the infinite sum of 0.95^k", "1 / (1 - 0.95)", "decimals=0"),
        ("f", "the smallest k such that 0.5^k < 0.001", "next(k for k in range(100) if 0.5 ** k < 0.001)", ""),
        ("g", "(-0.8)^7, 3 decimals", "(-0.8) ** 7",
         'decimals=3, mistakes={"une raison négative à une puissance IMPAIRE donne un nombre négatif": 0.8 ** 7}'),
    ]),
    Paper("0B.15", "Exponentielles et logarithmes : règles de calcul en bases 2, e et 10", [
        ("a", "ln(e^3)", "math.log(math.e ** 3)", "decimals=0"),
        ("b", "e^(ln 5)", "math.exp(math.log(5))", "decimals=0"),
        ("c", "ln 1 + log2 32", "math.log(1) + math.log2(32)", 'decimals=0, mistakes={"ln 1 = 0 (et non 1)": 6}'),
        ("d", "log10 0.001", "math.log10(0.001)", "decimals=0"),
        ("e", "ln(e^2 x e^5)", "math.log(math.e ** 2 * math.e ** 5)",
         'decimals=0, mistakes={"e^2 × e^5 = e^(2+5) : on additionne les exposants": 10}'),
        ("f", "the solution of e^x = 20, 2 decimals", "math.log(20)", "decimals=2"),
        ("g", "the solution of 2^x = 1000, 2 decimals", "math.log2(1000)",
         'decimals=2, mistakes={"x = log₂ 1000 = ln 1000 / ln 2, pas log₁₀ 1000": 3.0, "10 est le nombre de bits (arrondi à l\'entier supérieur) ; ici on demande la solution x, à 2 décimales": 10}'),
        ("h", "ln(0.01^100), 1 decimal", "100 * math.log(0.01)", "decimals=1"),
        ("i", "1 nat in bits, 3 decimals", "1 / math.log(2)",
         'decimals=3, mistakes={"1 nat vaut 1 / ln 2 bit, et non ln 2": math.log(2)}'),
    ]),
    Paper("0B.17", "Sigmoïde et tanh : valeurs, limites, symétries", [
        ("a", "sigma(0)", "sigmoid(0)", "decimals=3"),
        ("b", "sigma(3), 3 decimals", "sigmoid(3)", "decimals=3"),
        ("c", "sigma(-3), 3 decimals", "sigmoid(-3)", 'decimals=3, mistakes={"σ(−x) = 1 − σ(x), et non −σ(x)": -sigmoid(3)}'),
        ("d", "tanh(0.5), 3 decimals", "math.tanh(0.5)", "decimals=3"),
        ("e", "2 sigma(1) - 1, 3 decimals", "2 * sigmoid(1) - 1", "decimals=3"),
        ("f", "the limit of sigma(x) when x -> -infinity", "0", 'mistakes={"quand x → −∞, e^(−x) explose et σ(x) → 0": 1}'),
        ("g", "the x such that sigma(x) = 0.9, 2 decimals", "math.log(9)",
         'decimals=2, mistakes={"σ(x) = 0,9 ⟺ e^(−x) = 1/9 ⟺ x = ln 9": -math.log(9)}'),
        ("h", "sigma(x) + sigma(-x)", "round(sigmoid(1.7) + sigmoid(-1.7))", ""),
    ]),
    Paper("0B.18", "Produit scalaire, similarité cosinus et produit de Hadamard", [
        ("a", "a . b", "int(va @ vb)", ""),
        ("b", "are a and b orthogonal? True or False", "bool(va @ vb == 0)", ""),
        ("c", "||a||", "np.linalg.norm(va)", "decimals=1"),
        ("d", "a . c", "int(va @ vc)", ""),
        ("e", "cos(a, c)", "cosine(va, vc)", "decimals=1"),
        ("f", "cos(a, d), 3 decimals", "cosine(va, vd)", 'decimals=3, mistakes={"divise par les DEUX normes : ‖a‖ = 3 et ‖d‖ = 1": 1.0}'),
        ("g", "a ⊙ b, a list", "va * vb", ""),
        ("h", "the sum of the components of a ⊙ c", "int((va * vc).sum())", ""),
        ("i", "cos(a, -a)", "cosine(va, -va)", "decimals=1"),
    ]),
    Paper("0B.20", "Produit matriciel : calculer et vérifier les formes", [
        ("a", "the shape of AB, a tuple", "(A20 @ B20).shape", ""),
        ("b", "the shape of BA, a tuple", "(B20 @ A20).shape",
         'mistakes={"(m, n) × (n, p) → (m, p) : on garde les dimensions extérieures, ici celles de B puis de A": (3, 3)}'),
        ("c", "BA, a list of rows", "B20 @ A20", ""),
        ("d", "(AB)_11", "int((A20 @ B20)[0, 0])", ""),
        ("e", "(AB)_32", "int((A20 @ B20)[2, 1])", 'mistakes={"(AB)₃₂ : ligne 3 de A, colonne 2 de B": int((A20 @ B20)[1, 2])}'),
        ("f", "is CA defined? True or False", "C20.shape[1] == A20.shape[0]", ""),
        ("g", "AC, a list of rows", "A20 @ C20", ""),
        ("h", "the number of multiplications for AB", "3 * 2 * 3", 'mistakes={"9 est le nombre d\'éléments de AB ; chacun coûte n multiplications (m × n × p au total)": 9}'),
    ]),
    Paper("0B.21", "Identité, inverse 2 × 2 et système de deux équations", [
        ("a", "det M", "np.linalg.det(M21)", "decimals=0"),
        ("b", "M^-1, a list of rows", "np.linalg.inv(M21)",
         'decimals=1, mistakes={"n\'oublie pas de diviser par le déterminant (2)": [[2, -1], [-4, 3]]}'),
        ("c", "the solution [x, y]", "np.linalg.solve(M21, [5, 6])", "decimals=0"),
        ("d", "is [[2, 4], [1, 2]] invertible? True or False", "bool(abs(np.linalg.det(np.array([[2, 4], [1, 2]]))) > 1e-12)", ""),
        ("e", "the value of k", "6 / 2", 'decimals=0, mistakes={"det = 1 × 6 − k × 2 = 0": -3}'),
        ("f", "the inverse of [[2, 0], [0, 5]], a list of rows", "np.linalg.inv(np.array([[2.0, 0], [0, 5]]))", "decimals=1"),
    ]),
    Paper("0B.22", "Dériver avec les règles : somme, produit, quotient, exp, ln", [
        ("a", "f'(1)", "12 * 1 ** 2 - 2", ""),
        ("b", "g'(4), 4 decimals", "1 / (2 * math.sqrt(4)) - 1 / 4 ** 2",
         'decimals=4, mistakes={"(1/x)\' = −1/x² : le signe est négatif": 1 / (2 * math.sqrt(4)) + 1 / 4 ** 2}'),
        ("c", "h'(1), 3 decimals", "(2 * 1 + 1 ** 2) * math.e",
         'decimals=3, mistakes={"(uv)\' = u\'v + uv\' : il manque un terme": 2 * math.e}'),
        ("d", "k'(e)", "math.log(math.e) + 1", "decimals=0"),
        ("e", "q'(3)", "-2 / (3 - 1) ** 2", 'decimals=1, mistakes={"quotient : (u\'v − uv\') / v², attention au signe": 0.5}'),
        ("f", "r'(1)", "0", ""),
        ("g", "s'(2)", "20 * 2 ** 3", 'mistakes={"(x⁴)\' = 4x³, puis × 5": 80}'),
    ]),
    Paper("0B.23", "Règle de la chaîne : décomposer, puis dériver", [
        ("a", "value at x = 1", "12 * (3 * 1 - 1) ** 3", 'mistakes={"n\'oublie pas la dérivée intérieure (× 3)": 32}'),
        ("b", "value at x = 0, 3 decimals", "2 * math.exp(1)",
         'decimals=3, mistakes={"(e^u)\' = e^u × u\', avec u\' = 2": math.exp(1)}'),
        ("c", "value at x = 2", "2 * 2 / (2 ** 2 + 1)", "decimals=1"),
        ("d", "value at x = 2, 3 decimals", "2 / math.sqrt(1 + 4 * 2)",
         'decimals=3, mistakes={"(√u)\' = u\' / (2√u), avec u\' = 4": 1 / (2 * math.sqrt(9))}'),
        ("e", "value at x = 1, 3 decimals", "-1 * math.exp(-0.5)", "decimals=3"),
        ("f", "value at x = 1", "-2 * 1 / (1 ** 2 + 1) ** 2", "decimals=1"),
        ("g", "dL/dw at w = 2", "2 * (3 * 2 + 1 - 4) * 3", 'mistakes={"dérivée intérieure : d(3w + 1 − 4)/dw = 3": 6}'),
        ("h", "value at x = e, 3 decimals", "2 * math.log(math.e) / math.e", "decimals=3"),
    ]),
    Paper("0B.25", "Lignes de niveau, dérivées partielles, gradient et un pas de descente", [
        ("a", "f(3, 1)", "f25(3, 1)", ""),
        ("b", "df/dx at (3, 1)", "2 * (3 - 1)", ""),
        ("c", "df/dy at (3, 1)", "4 * 1", 'mistakes={"∂(2y²)/∂y = 4y": 2}'),
        ("d", "the gradient, a list", "[2 * (3 - 1), 4 * 1]", ""),
        ("e", "its norm, 2 decimals", "math.hypot(4, 4)", "decimals=2"),
        ("f", "the point after one step, a list", "[3 - 0.1 * 4, 1 - 0.1 * 4]",
         'decimals=1, mistakes={"on descend : on SOUSTRAIT η∇f": [3.4, 1.4]}'),
        ("g", "f at the new point, 2 decimals", "f25(3 - 0.1 * 4, 1 - 0.1 * 4)", "decimals=2"),
        ("h", "the minimum point, a list", "[1, 0]", ""),
        ("i", "dg/dx at (1, 2)", "2 * 1 * 2", ""),
        ("j", "dg/dy at (1, 2)", "1 ** 2 + 3 * 2 ** 2", 'mistakes={"∂(x²y)/∂y = x² et ∂(y³)/∂y = 3y²": 12}'),
    ]),
    Paper("0B.26", "Indépendance : tester P(A ∩ B) = P(A) P(B) avec deux dés", [
        ("a", "P(A), 3 decimals", "prob(first_even)", "decimals=3"),
        ("b", "P(B)", "prob(sum_is(7))", "decimals=3"),
        ("c", "P(A ∩ B)", "prob(lambda d: first_even(d) and sum_is(7)(d))", "decimals=3"),
        ("d", "are A and B independent? True or False",
         "math.isclose(prob(lambda d: first_even(d) and sum_is(7)(d)), prob(first_even) * prob(sum_is(7)))", ""),
        ("e", "P(C)", "prob(sum_is(8))", "decimals=3"),
        ("f", "P(A ∩ C)", "prob(lambda d: first_even(d) and sum_is(8)(d))", "decimals=3"),
        ("g", "are A and C independent? True or False",
         "math.isclose(prob(lambda d: first_even(d) and sum_is(8)(d)), prob(first_even) * prob(sum_is(8)))", ""),
        ("h", "P(A ∩ C) - P(A) P(C), 3 decimals",
         "prob(lambda d: first_even(d) and sum_is(8)(d)) - prob(first_even) * prob(sum_is(8))", "decimals=3"),
    ]),
    Paper("0B.27", "Espérance et variance d'une variable discrète", [
        ("a", "E[X]", "values27 @ probs27", "decimals=1"),
        ("b", "E[X^2]", "values27 ** 2 @ probs27", 'decimals=1, mistakes={"E[X²] = Σ x² p (et non E[X]²)": 1.44}'),
        ("c", "Var(X)", "values27 ** 2 @ probs27 - (values27 @ probs27) ** 2",
         'decimals=2, mistakes={"Var = E[X²] − E[X]² : n\'oublie pas d\'élever E[X] au carré": 3.6 - 1.2}'),
        ("d", "the standard deviation, 2 decimals", "math.sqrt(values27 ** 2 @ probs27 - (values27 @ probs27) ** 2)", "decimals=2"),
        ("e", "E[3X + 2]", "3 * (values27 @ probs27) + 2", "decimals=1"),
        ("f", "Var(3X + 2)", "9 * (values27 ** 2 @ probs27 - (values27 @ probs27) ** 2)",
         'decimals=2, mistakes={"Var(aX + b) = a² Var(X) : le b disparaît, le a est au carré": 3 * 2.16 + 2}'),
        ("g", "E of this die, 3 decimals", "np.mean([1, 1, 2, 3, 3, 3])", "decimals=3"),
        ("h", "Var of this die, 3 decimals", "np.var([1, 1, 2, 3, 3, 3])",
         'decimals=3, mistakes={"garde E[X] = 13/6 exact : l\'arrondir à 2,167 avant de l\'élever au carré fausse le résultat": 5.5 - 2.167 ** 2}'),
    ]),
]

PAPER_CONTEXT = '''# Data of the paper exercises (the same as in 02_exercices.md)
import math
from itertools import product

import numpy as np

x3 = [2, -1, 4, 3]                                                   # 0B.3
s4 = [2, 4, 9, 1, 5, 6]                                              # 0B.4
omega5 = set(range(1, 13))                                           # 0B.5
A5 = {n for n in omega5 if n % 2 == 0}
B5 = {n for n in omega5 if n % 3 == 0}
C5 = {1, 2, 3}
p1, p2, u8 = np.array([40, 190]), np.array([48, 196]), np.array([3, -4])   # 0B.8
A9 = np.array([[2, 0, 1], [-1, 3, 2]])                               # 0B.9
v9 = np.array([1, 2, -1])
X9 = np.array([[1, 2], [3, 0], [0, -1]])
w9 = np.array([0.5, 2])


def f10(x):                                                          # 0B.10
    return x ** 2 + x


DICE = list(product(range(1, 7), repeat=2))                         # 0B.11, 0B.26


def prob(event):
    """Probability of an event (a function of the pair of dice) with two fair dice."""
    return sum(1 for d in DICE if event(d)) / len(DICE)


def first_even(d):
    return d[0] % 2 == 0


def sum_is(total):
    return lambda d: d[0] + d[1] == total


def sigmoid(x):                                                      # 0B.17
    return 1 / (1 + math.exp(-x))


va, vb, vc, vd = (np.array(v) for v in ([1, 2, 2], [2, 0, -1], [2, 4, 4], [1, 0, 0]))   # 0B.18


def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


A20 = np.array([[1, 2], [0, -1], [3, 1]])                            # 0B.20
B20 = np.array([[2, 1, 0], [1, -1, 4]])
C20 = np.array([[1, 1], [2, 0]])
M21 = np.array([[3.0, 1.0], [4.0, 2.0]])                             # 0B.21


def f25(x, y):                                                       # 0B.25
    return (x - 1) ** 2 + 2 * y ** 2


values27, probs27 = np.array([0, 1, 2, 5]), np.array([0.4, 0.3, 0.2, 0.1])   # 0B.27'''

PAPER_INTRO = ("## Partie 0 · Vérifier tes exercices papier (✏️ de 0B.1 à 0B.27)\n\n"
               "Fais d'abord les exercices ✏️ de `02_exercices.md` **sur papier**, dans ta copie de "
               "`06_mes_reponses.md`. Reporte ensuite chaque réponse ici : **la valeur** que tu as trouvée "
               "(par exemple `42`, `0.125`, `[7, -2]`, `[[1, 0], [0, 1]]`, `(4, 5)`, `True`), pas l'expression Python, "
               "sinon tu ne vérifies rien. Arrondis comme l'énoncé le demande. Les réponses pas encore "
               "remplies affichent ⏳.")

# ---------------------------------------------------------------------------
# Part A: numbers and functions in code (0B.33 to 0B.37)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools for the notebook exercises (parts A to D)
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def verdict(ex_id, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message."""
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")


def error_name(func, *args):
    """Name of the exception raised by func(*args), or "no error"."""
    try:
        func(*args)
    except NotImplementedError:
        raise
    except Exception as error:  # noqa: BLE001 - we want the name of whatever is raised
        return type(error).__name__
    return "no error"


def measure(func, *args, repeat=3):
    """Best time (in seconds) of `repeat` runs of func(*args)."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        best = min(best, time.perf_counter() - start)
    return best'''

PART_A = Part("A", "Nombres et fonctions en code",
              "Fiche §101.1 et §101.2. Tu retrouves en Python les calculs faits à la main : puissances, arrondis, "
              "Σ et Π, moyennes, fonctions usuelles, et les pièges du calcul en virgule flottante.",
              given=PART_A_GIVEN, exercises=[
    Ex("0B.33", "📦", 1, 10, "Calculer avec Python : puissances, arrondis, `abs`, signe et `C(n, k)`",
       "traduire en Python les calculs de 101.1 et repérer les pièges de priorité et d'arrondi.",
       "0A · Ex 0B.2, Ex 0B.12 · fiche §101.1.1, §101.1.2, §101.1.7", tracks="M, C",
       body=r"""Range dans chaque variable le résultat **calculé par Python** : écris l'expression, pas la valeur que tu as en tête.

a) `powers` : la liste `[(-3) ** 2, -3 ** 2]`. Pourquoi les deux nombres diffèrent-ils ? (réponds dans ta copie)
b) `roundings` : la liste `[math.floor(-3.7), math.ceil(-3.7), int(-3.7), round(-3.7)]`. Lesquelles de ces fonctions vont vers le bas, vers le haut, vers zéro, vers l'entier le plus proche ?
c) `n_small` : le nombre de valeurs de `values_33` telles que $|x| \leq 1$ (un masque `np.abs(values_33) <= 1`, puis `.sum()`).
d) `signs` : le signe de chaque valeur de `values_33` (`np.sign`).
e) `poker_hands` : le nombre de mains de 5 cartes qu'on peut tirer d'un jeu de 52 cartes (`math.comb`).
f) `digits` : le nombre de chiffres de $2^{100}$ (`len(str(...))`). Compare avec $100 \log_{10} 2 \approx 30{,}1$ (101.1.1) : comment passe-t-on de l'un à l'autre ?""",
       given=r'''values_33 = np.array([-2.5, 0.0, 3.1, -0.2, 7.0, 1.0])''',
       todo=r'''powers = ...        # a)
roundings = ...     # b)
n_small = ...       # c)
signs = ...         # d)
poker_hands = ...   # e)
digits = ...        # f)''',
       check=r'''wb.check("0B.33a", powers)
wb.check("0B.33b", roundings)
wb.check("0B.33c", n_small)
wb.check("0B.33d", signs)
wb.check("0B.33e", poker_hands)
wb.check("0B.33f", digits)''',
       solution=r'''powers = [(-3) ** 2, -3 ** 2]            # -3 ** 2 is -(3 ** 2): the power comes first
roundings = [math.floor(-3.7), math.ceil(-3.7), int(-3.7), round(-3.7)]
n_small = int((np.abs(values_33) <= 1).sum())
signs = np.sign(values_33)
poker_hands = math.comb(52, 5)
digits = len(str(2 ** 100))
print(powers, roundings, n_small, signs, poker_hands, digits, 100 * math.log10(2))''',
       record=r'''wb.record("0B.33a", powers, mistakes={"-3 ** 2 se lit -(3 ** 2) : la puissance passe avant le signe moins": [9, 9]})
wb.record("0B.33b", roundings, mistakes={"math.floor va vers le BAS (vers −∞), pas vers zéro": [-3, -3, -3, -4]})
wb.record("0B.33c", n_small, mistakes={"≤ 1 : la valeur 1.0 compte aussi": 2})
wb.record("0B.33d", signs)
wb.record("0B.33e", poker_hands, mistakes={"une main ne dépend pas de l'ordre des cartes : math.comb, pas math.perm": math.perm(52, 5)})
wb.record("0B.33f", digits, mistakes={"combien de chiffres a un nombre compris entre 10^30 et 10^31 ? (10^1 = 10 en a déjà 2)": 30})''',
       note="`round` arrondit à l'entier le plus proche (et, à égalité, au pair : `round(2.5)` vaut 2, 0A.1) ; "
            "`int` tronque vers zéro ; `floor` et `ceil` vont vers le bas et vers le haut. Un nombre $x \\geq 1$ "
            "s'écrit avec $\\lfloor \\log_{10} x \\rfloor + 1$ chiffres : $\\lfloor 30{,}1 \\rfloor + 1 = 31$."),

    Ex("0B.34", "🔮", 1, 10, "0,99 puissance 1000 : presque 1 ou presque 0 ?",
       "prévoir l'ordre de grandeur de puissances d'un nombre proche de 1, puis le vérifier.",
       "Ex 0B.13 · fiche §101.1.5", tracks="M, C", hypothesis=True,
       body=r"""**Sans rien calculer**, classe chacun de ces nombres dans une catégorie : `"zero"` (moins de 0,01), `"petit"` (de 0,01 à 0,5), `"proche de 1"` (de 0,5 à 2) ou `"grand"` (plus de 2).

a) $0{,}99^{10}$ · b) $0{,}99^{100}$ · c) $0{,}99^{1000}$ · d) $1{,}01^{100}$ · e) $1{,}01^{1000}$ · f) $0{,}99^{1000} \times 1{,}01^{1000}$

Écris ton hypothèse (cellule 📝), puis tes six prédictions ; la vérification ne regarde que tes prédictions. Ensuite seulement, exécute l'**Expérience**, qui calcule les vrais nombres.

g) Après l'expérience : `k_small`, le plus petit entier $k$ tel que $0{,}99^k < 0{,}01$, trouvé avec une boucle `while` ; puis retrouve-le avec un logarithme (101.2.4).""",
       todo=r'''prediction_0B_34a = ...   # a) 0.99 ** 10: "zero", "petit", "proche de 1" or "grand"
prediction_0B_34b = ...   # b) 0.99 ** 100
prediction_0B_34c = ...   # c) 0.99 ** 1000
prediction_0B_34d = ...   # d) 1.01 ** 100
prediction_0B_34e = ...   # e) 1.01 ** 1000
prediction_0B_34f = ...   # f) 0.99 ** 1000 * 1.01 ** 1000''',
       check=r'''predictions_0B_34 = [prediction_0B_34a, prediction_0B_34b, prediction_0B_34c,
                     prediction_0B_34d, prediction_0B_34e, prediction_0B_34f]
for letter, prediction in zip("abcdef", predictions_0B_34):
    wb.check(f"0B.34{letter}", prediction)''',
       solution=r'''prediction_0B_34a = "proche de 1"
prediction_0B_34b = "petit"
prediction_0B_34c = "zero"
prediction_0B_34d = "grand"
prediction_0B_34e = "grand"
prediction_0B_34f = "proche de 1"''',
       record=r'''wb.record("0B.34a", prediction_0B_34a, mistakes={"10 fois −1 % : on ne perd qu'environ 10 %": "petit"})
wb.record("0B.34b", prediction_0B_34b, mistakes={"100 fois −1 % : on perd bien plus que 1 %, mais on ne tombe pas à 0": "zero"})
wb.record("0B.34c", prediction_0B_34c, mistakes={"0,99 est proche de 1, mais 1000 multiplications font fondre le résultat (101.1.5)": "proche de 1"})
wb.record("0B.34d", prediction_0B_34d, mistakes={"+1 % cent fois de suite : les hausses se cumulent (comme des intérêts composés)": "proche de 1"})
wb.record("0B.34e", prediction_0B_34e, mistakes={"+1 % mille fois de suite : les hausses se cumulent et explosent": "proche de 1"})
wb.record("0B.34f", prediction_0B_34f, mistakes={"regroupe les facteurs deux par deux, puis regarde le produit de ces deux nombres": "zero"})''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", r'''def category(x):
    """The category of a positive number, as in the statement."""
    if x < 0.01:
        return "zero"
    if x <= 0.5:
        return "petit"
    if x <= 2:
        return "proche de 1"
    return "grand"


numbers_0B_34 = {"a": 0.99 ** 10, "b": 0.99 ** 100, "c": 0.99 ** 1000,
                 "d": 1.01 ** 100, "e": 1.01 ** 1000, "f": 0.99 ** 1000 * 1.01 ** 1000}
for letter, x in numbers_0B_34.items():
    print(f"{letter}) {x:.4g} -> {category(x)}")'''),
              ("md", "g) Le plus petit $k$ tel que $0{,}99^k < 0{,}01$ : une boucle `while`, puis le même nombre avec un logarithme."),
              ("todo", r'''k_small = ...   # g) the smallest k such that 0.99 ** k < 0.01 (a while loop)'''),
              ("check", r'''wb.check("0B.34g", k_small)'''),
              ("solution", r'''k_small = 0
while 0.99 ** k_small >= 0.01:
    k_small += 1
print(k_small, math.log(0.01) / math.log(0.99))   # 0.99 ** k < 0.01  <=>  k > ln 0.01 / ln 0.99'''),
              ("record", r'''wb.record("0B.34g", k_small, mistakes={"vérifie la condition de ta boucle : on cherche le premier k où 0,99^k passe SOUS 0,01": 458})''')],
       note="$0{,}99^{1000} \\approx 4 \\times 10^{-5}$ alors que $1{,}01^{1000} \\approx 21\\,000$ : une petite "
            "différence de raison, répétée mille fois, devient énorme. C'est le cœur des gradients qui "
            "s'évanouissent ou explosent (ch. 22). Pour f, $0{,}99 \\times 1{,}01 = 0{,}9999$ : les deux effets se "
            "compensent presque. En g, $k > \\frac{\\ln 0{,}01}{\\ln 0{,}99} \\approx 458{,}2$, donc $k = 459$ : "
            "attention au sens de l'inégalité, car $\\ln 0{,}99 < 0$."),

    Ex("0B.35", "📦", 2, 15, "Σ, Π et moyennes en code : `sum`, `math.prod`, `np.average`, moyenne mobile",
       "programmer sommes, produits, moyennes pondérées et moyennes mobiles, avec et sans NumPy.",
       "Ex 0B.3, Ex 0B.4 · fiche §101.1.3, §101.1.4", thread="synthétique", tracks="M, C",
       body=r"""a) `sum_squares` : $\sum_{i=1}^{50} i^2$ avec `sum` et une expression génératrice ; compare avec la formule $\frac{n(n+1)(2n+1)}{6}$.
b) `product_terms` : $\prod_{k=1}^{10} \left(1 + \frac{1}{k}\right)$ avec `math.prod` (6 décimales). Écris les premiers facteurs sous forme de fractions ($\frac{2}{1}$, $\frac{3}{2}$…) : pourquoi le résultat est-il si simple ?
c) `weighted` : la moyenne pondérée des notes 12, 15, 9 et 17 avec les coefficients 2, 1, 3 et 4 (`np.average`).
d) `z` : la sortie d'un neurone, $\mathbf{w} \cdot \mathbf{x} + b$, avec `w = [0.5, -1, 2, 0.1]`, `x = [4, 1, 0.5, 10]` et `b = 0.5` (`np.dot(w, x)`, ou `@` entre deux arrays NumPy).
e) et f) `moving_average(x, k)` : les moyennes mobiles d'ordre `k` d'une série, un array de `len(x) - k + 1` valeurs. Écris-la **avec une boucle** sur les fenêtres `x[t - k + 1 : t + 1]`, puis compare-la à `np.convolve(x, np.ones(k) / k, mode="valid")` (fiche §101.1.4). La vérification regarde, pour la série bruitée `series` (200 valeurs) et $k = 9$ : e) le nombre de moyennes mobiles, f) les trois premières (3 décimales). Elle trace ensuite `series` avec ses moyennes mobiles d'ordre 3 et 21 : laquelle suit le mieux la sinusoïde cachée ? laquelle réagit en retard ?""",
       given=r'''t_35, series = wb.synth.noisy_sine(n=200, freq=0.02, noise=0.4, t_max=100, seed=0)   # a noisy sine wave
print(series.shape, series[:4].round(3))''',
       todo=r'''sum_squares = ...     # a)
product_terms = ...   # b)
weighted = ...        # c)
z = ...               # d)


def moving_average(x, k):
    """Moving averages of order k of the series x (with a loop): len(x) - k + 1 values."""
    raise NotImplementedError("moving_average() is not written yet")''',
       check=r'''wb.check("0B.35a", sum_squares)
wb.check("0B.35b", product_terms)
wb.check("0B.35c", weighted)
wb.check("0B.35d", z)
with wb.attempt("0B.35e"):
    wb.check("0B.35e", len(moving_average(series, 9)))
    wb.check("0B.35f", np.asarray(moving_average(series, 9))[:3])
    same = np.allclose(moving_average(series, 5), np.convolve(series, np.ones(5) / 5, mode="valid"))
    verdict("0B.35", same, "ta boucle donne les mêmes valeurs que np.convolve.",
            "ta boucle ne donne pas les mêmes valeurs que np.convolve (fenêtres, bornes ?).")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t_35, series, color="0.6", lw=1, label="series")
    for k in (3, 21):
        ax.plot(t_35[k - 1:], moving_average(series, k), lw=2, label=f"moving average, k = {k}")
    ax.legend()
    plt.show()''',
       solution=r'''sum_squares = sum(i ** 2 for i in range(1, 51))
product_terms = math.prod(1 + 1 / k for k in range(1, 11))   # 2/1 × 3/2 × ... × 11/10 = 11
weighted = np.average([12, 15, 9, 17], weights=[2, 1, 3, 4])
z = np.array([0.5, -1, 2, 0.1]) @ np.array([4, 1, 0.5, 10]) + 0.5


def moving_average(x, k):
    """Moving averages of order k of the series x (with a loop): len(x) - k + 1 values."""
    return np.array([np.mean(x[t - k + 1:t + 1]) for t in range(k - 1, len(x))])


print(sum_squares, 50 * 51 * 101 // 6, round(product_terms, 6), weighted, z)
print(len(moving_average(series, 9)), moving_average(series, 9)[:3].round(3),
      np.allclose(moving_average(series, 5), np.convolve(series, np.ones(5) / 5, mode="valid")))
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(t_35, series, color="0.6", lw=1, label="series")
for k in (3, 21):
    ax.plot(t_35[k - 1:], moving_average(series, k), lw=2, label=f"moving average, k = {k}")
ax.legend()
plt.show()''',
       record=r'''wb.record("0B.35a", sum_squares, mistakes={"range(1, 50) s'arrête à 49 : la borne haute d'un Σ est incluse": sum(i ** 2 for i in range(1, 50))})
wb.record("0B.35b", product_terms, decimals=6)
wb.record("0B.35c", weighted, decimals=1, mistakes={"divise par la somme des coefficients (10), pas par le nombre de notes": np.mean([12, 15, 9, 17])})
wb.record("0B.35d", z, decimals=1, mistakes={"n'oublie pas le biais b = 0,5": z - 0.5})
wb.record("0B.35e", len(moving_average(series, 9)), mistakes={"la première moyenne mobile existe à t = k : il y en a n − k + 1": 191})
wb.record("0B.35f", moving_average(series, 9)[:3], decimals=3)''',
       note="Le produit « télescopique » $\\frac{2}{1} \\times \\frac{3}{2} \\times \\dots \\times \\frac{11}{10}$ se "
            "simplifie en $\\frac{11}{1}$. Sur la figure, l'ordre 3 garde beaucoup de bruit ; l'ordre 21 suit la "
            "sinusoïde, mais avec un retard d'environ 10 pas, car chaque moyenne ne regarde que le passé. "
            "`np.convolve` fait le même calcul en code compilé ; `np.cumsum` permet aussi un calcul en temps "
            "linéaire (0A)."),

    Ex("0B.36", "📦", 2, 20, "Galerie des fonctions usuelles : de l'affine au cosinus",
       "tracer et reconnaître les fonctions usuelles du ML avec NumPy et matplotlib.",
       "Ex 0B.6, Ex 0B.17, Ex 0B.7 · fiche §101.2.1 à §101.2.6, §101.1.2", thread="synthétique", tracks="M, C",
       body=r"""1. Écris `sigmoid(x)`, qui calcule $\sigma(x) = \frac{1}{1 + e^{-x}}$ pour un **array** NumPy d'un coup (avec `np.exp`, sans boucle).
2. Écris `gallery()`, qui trace une grille 3 × 3 (`fig, axes = plt.subplots(3, 3, figsize=(11, 8))`, puis `axes.ravel()`) avec, dans l'ordre : $2x - 1$, $x^2 - 2x - 3$, $e^x$, $\ln x$, $\sigma(x)$, $\tanh(x)$, $\cos x$, $|x|$ et $\lfloor x \rfloor$. Pour chaque panneau : un titre (le nom de la fonction) et les axes passant par l'origine (`ax.axhline(0)` et `ax.axvline(0)`). Prends `x = np.linspace(-4, 4, 401)`, sauf pour $\ln$, définie seulement pour $x > 0$ : `np.linspace(0.05, 4, 400)`. `gallery()` renvoie la figure.

Vérifications : a) `sigmoid(np.array([-2.0, 0.0, 2.0]))` (3 décimales) · b) $\tanh(1)$ et $2\sigma(2) - 1$ sont-ils égaux (`np.isclose`, 0B.17) ? · c) la structure de la figure : 9 panneaux, chacun avec un titre et une courbe, et aucune valeur `nan` ou infinie tracée (sinon, un $\ln$ de nombre négatif s'est glissé quelque part).

Regarde ensuite la figure (ta copie) : quelles fonctions sont bornées ? croissantes ? paires ($f(-x) = f(x)$) ? impaires ?""",
       todo=r'''def sigmoid(x):
    """1 / (1 + exp(-x)), computed element-wise on a NumPy array."""
    raise NotImplementedError("sigmoid() is not written yet")


def gallery():
    """A 3 x 3 figure of the usual functions; returns the figure."""
    raise NotImplementedError("gallery() is not written yet")''',
       check=r'''def gallery_report(fig):
    """Problems found in the gallery figure (an empty list if it is fine)."""
    problems = []
    if len(fig.axes) != 9:
        problems.append(f"{len(fig.axes)} panneaux au lieu de 9")
    for ax in fig.axes:
        title = " ".join(ax.get_title(loc) for loc in ("left", "center", "right")).strip()
        name = title or "un panneau"
        if not title:
            problems.append("un panneau n'a pas de titre")
        curves = [np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()
                  if len(line.get_xdata()) > 2]                      # not the axhline/axvline
        curves += [np.asarray(item.get_offsets(), dtype=float)[:, 1] for item in ax.collections
                   if len(item.get_offsets()) > 2]                   # points drawn with scatter
        if not curves:
            problems.append(f"{name} : aucune courbe")
        elif any(np.isinf(y).any() or np.isnan(y).mean() > 0.05 for y in curves):
            problems.append(f"{name} : des valeurs infinies ou beaucoup de nan (ln d'un nombre négatif ou nul ?)")
    return problems


with wb.attempt("0B.36a"):
    wb.check("0B.36a", sigmoid(np.array([-2.0, 0.0, 2.0])))
    wb.check("0B.36b", bool(np.isclose(np.tanh(1), 2 * sigmoid(np.array(2.0)) - 1)))
with wb.attempt("0B.36c"):
    fig = gallery()
    plt.show()
    problems = gallery_report(fig)
    verdict("0B.36c", not problems, "la galerie a la bonne structure.", " ; ".join(problems))''',
       solution=r'''def sigmoid(x):
    """1 / (1 + exp(-x)), computed element-wise on a NumPy array."""
    return 1 / (1 + np.exp(-x))


def gallery():
    """A 3 x 3 figure of the usual functions; returns the figure."""
    x = np.linspace(-4, 4, 401)
    x_pos = np.linspace(0.05, 4, 400)
    panels = [("2x - 1", x, 2 * x - 1), ("x² - 2x - 3", x, x ** 2 - 2 * x - 3), ("exp(x)", x, np.exp(x)),
              ("ln(x)", x_pos, np.log(x_pos)), ("sigmoid(x)", x, sigmoid(x)), ("tanh(x)", x, np.tanh(x)),
              ("cos(x)", x, np.cos(x)), ("|x|", x, np.abs(x)), ("floor(x)", x, np.floor(x))]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, (title, xs, ys) in zip(axes.ravel(), panels):
        ax.plot(xs, ys, lw=2)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_title(title)
    fig.tight_layout()
    return fig


print(sigmoid(np.array([-2.0, 0.0, 2.0])).round(3), np.tanh(1), 2 * sigmoid(np.array(2.0)) - 1)
fig = gallery()
plt.show()''',
       record=r'''wb.record("0B.36a", sigmoid(np.array([-2.0, 0.0, 2.0])), decimals=3)
wb.record("0B.36b", bool(np.isclose(np.tanh(1), 2 * sigmoid(np.array(2.0)) - 1)))''',
       note="Bornées : σ (entre 0 et 1), tanh (entre −1 et 1), cos (entre −1 et 1). Croissantes : l'affine, "
            "exp, ln, σ, tanh et la partie entière (en escalier). Paires : $x^2$ n'y est pas (le terme $-2x$ la "
            "décentre), mais $|x|$ et $\\cos x$ le sont ; impaire : $\\tanh$. `np.floor` trace un escalier : "
            "matplotlib relie les marches par des segments presque verticaux."),

    Ex("0B.37", "🐛", 2, 15, "`exp` et `log` en NumPy : `-inf`, `nan` et dépassements",
       "diagnostiquer les `-inf`, `inf` et `nan` du calcul flottant et les éviter avec les règles des logarithmes.",
       "Ex 0B.36, Ex 0B.15 · fiche §101.2.3, §101.2.4", tracks="M, C",
       body=r"""Les trois fonctions ci-dessous renvoient `-inf`, `inf` ou `nan` avec un simple avertissement (`RuntimeWarning`) : NumPy ne lève **pas** d'erreur, et le calcul continue avec une valeur absurde. Pour chacune, explique la cause dans ta copie (quel nombre sort des limites d'un `float64`, ou du domaine de $\ln$ ?), puis écris la version corrigée, sans changer son nom :

1. `log_prob_all` : utilise $\ln(p_1 p_2 \cdots p_n) = \sum_i \ln p_i$ ;
2. `geometric_mean` : passe par les logarithmes, $\left(\prod_i x_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_i \ln x_i\right)$ ;
3. `log_shift` doit renvoyer $\ln(1 + x - \min x)$ : 0 pour la plus petite valeur, jamais `nan` ni `-inf` (la fonction `np.log1p(u)` calcule $\ln(1 + u)$ ; cette transformation est courante pour des données positives très étalées).

Vérifications : a) `log_prob_all(probs_37)` (1 décimale) · b) `geometric_mean(values_37)` (1 décimale) · c) `log_shift(durations_37)` (3 décimales) · d) `max_exponent` : le plus grand entier $n$ tel que `np.exp(n)` est encore fini (`np.isfinite`), trouvé avec une boucle. Le dernier essai affiche un `RuntimeWarning: overflow` : c'est normal (tu peux le masquer en plaçant la boucle dans un bloc `with np.errstate(over="ignore"):`).""",
       given=r'''probs_37 = np.full(1000, 0.3)                 # 1000 independent events, each of probability 0.3
values_37 = np.linspace(100, 10_000, 500)      # 500 positive measurements
durations_37 = np.array([12.0, 15.5, 12.0, 30.0, 18.25])


def log_prob_all(probs):
    """ln of the probability that ALL the independent events happen."""
    return np.log(np.prod(probs))


def geometric_mean(values):
    """(x_1 × x_2 × ... × x_n) ** (1 / n)."""
    return np.prod(values) ** (1 / len(values))


def log_shift(values):
    """ln(1 + x - min x): 0 for the smallest value (buggy version)."""
    return np.log(values - values.mean())


print("log_prob_all:", log_prob_all(probs_37))
print("geometric_mean:", geometric_mean(values_37))
print("log_shift:", log_shift(durations_37))''',
       todo=r'''def log_prob_all(probs):
    raise NotImplementedError("log_prob_all() is not fixed yet")


def geometric_mean(values):
    raise NotImplementedError("geometric_mean() is not fixed yet")


def log_shift(values):
    raise NotImplementedError("log_shift() is not fixed yet")


max_exponent = ...   # d) the largest integer n such that np.exp(n) is finite''',
       check=r'''with wb.attempt("0B.37a"):
    wb.check("0B.37a", log_prob_all(probs_37))
with wb.attempt("0B.37b"):
    wb.check("0B.37b", geometric_mean(values_37))
with wb.attempt("0B.37c"):
    wb.check("0B.37c", log_shift(durations_37))
wb.check("0B.37d", max_exponent)''',
       solution=r'''def log_prob_all(probs):
    return np.sum(np.log(probs))            # the product underflowed to 0, and ln 0 = -inf


def geometric_mean(values):
    return np.exp(np.mean(np.log(values)))  # the product overflowed to inf


def log_shift(values):
    return np.log1p(values - values.min())  # values - mean < 0 below the mean, and the log of a negative number is nan


max_exponent = 0
with np.errstate(over="ignore"):            # hide the overflow warning of the last try
    while np.isfinite(np.exp(max_exponent + 1)):
        max_exponent += 1
print(log_prob_all(probs_37), geometric_mean(values_37), log_shift(durations_37).round(3), max_exponent)''',
       record=r'''wb.record("0B.37a", log_prob_all(probs_37), decimals=1)
wb.record("0B.37b", geometric_mean(values_37), decimals=1, mistakes={"c'est la moyenne ordinaire : la moyenne géométrique passe par les logarithmes": values_37.mean()})
wb.record("0B.37c", log_shift(durations_37), decimals=3)
wb.record("0B.37d", max_exponent, mistakes={"vérifie ta condition d'arrêt : on veut le DERNIER n pour lequel np.exp(n) est encore fini": 710})''',
       note="1. $0{,}3^{1000} \\approx 10^{-523}$ est trop petit pour un `float64` : `np.prod` renvoie 0 "
            "(underflow), et $\\ln 0 = -\\infty$. 2. Le produit de 500 nombres entre 100 et 10 000 dépasse "
            "$10^{308}$ : `inf`. 3. Soustraire la moyenne rend négatives les valeurs sous la moyenne : `nan`. "
            "`np.exp` déborde au-delà de $\\ln(1{,}8 \\times 10^{308}) \\approx 709{,}8$. Règle : on garde les "
            "grands produits sous forme de sommes de logarithmes (0B.E3)."),
])


# ---------------------------------------------------------------------------
# Part B: mylearn.linalg_basics, then NumPy (0B.38 to 0B.46)
# ---------------------------------------------------------------------------
PART_B_GIVEN = r'''# Tools and data for part B
p1, p2 = [40, 190], [48, 196]      # two penguins: (bill length, flipper length) in mm (0B.8)


def run_linalg_tests(keyword, impl="learner"):
    """Run the tests of mylearn.linalg_basics selected by `keyword` (on YOUR code by default)."""
    command = [sys.executable, "-m", "pytest", "tests/test_ch00b_linalg_basics.py", "-k", keyword, "-q",
               "-p", "no:cacheprovider", "--color=no", "-rf", "--tb=line"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "200"})   # long lines: the reason of each failure
    lines = result.stdout.strip().splitlines()
    for line in [line for line in lines if line.startswith("FAILED")][:8]:
        print(line[:200])
    print("pytest:", lines[-1] if lines else result.stderr.strip()[-300:])


def error_message(func, *args):
    """Message of the exception raised by func(*args), or "no error"."""
    try:
        func(*args)
    except NotImplementedError:
        raise
    except Exception as error:  # noqa: BLE001 - we want the message of whatever is raised
        return str(error)
    return "no error"'''

MYLEARN_HOWTO = ("> **Mode d'emploi mylearn** (comme en 0A.26 et 0A.63) : ouvre `mon_travail/mylearn/linalg_basics.py` "
                 "(créé par `python tools/start_chapter.py 0B`), lis la docstring de chaque fonction, remplace les "
                 "`raise NotImplementedError(...)` par ton code et **enregistre**. Tout en **Python pur** : des listes, "
                 "des boucles ou des compréhensions, `math` si besoin, mais **pas de NumPy** dans ce fichier (NumPy est "
                 "l'oracle des tests). La cellule de vérification recharge ta librairie, vérifie quelques valeurs, puis "
                 "lance les tests de ces fonctions.")

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="0B")   # reload your saved file\n'

PART_B = Part("B", "mylearn.linalg_basics : vecteurs et matrices en Python pur, puis NumPy",
              "Fiche §101.3 et §101.4. Tu écris toi-même, avec des listes et des boucles, ce que cache le `@` de "
              "NumPy : sommes de vecteurs, produit scalaire, normes, cosinus, transposée, produit matriciel. Puis tu "
              "compares avec NumPy, et tu chasses les bugs de formes.", given=PART_B_GIVEN, exercises=[
    Ex("0B.38", "🔨", 2, 15, "`linalg_basics` (1) : additionner, soustraire, multiplier des vecteurs",
       "écrire les opérations composante par composante sur des vecteurs stockés en listes.",
       "Ex 0B.8 · 0A (listes, `zip`, compréhensions) · fiche §101.3.1, §101.3.4", tracks="M, C",
       body=MYLEARN_HOWTO + r"""

Écris `vector_add`, `vector_subtract`, `scalar_multiply` et `hadamard`. Tout se fait **composante par composante** : une compréhension sur `zip(u, v)` suffit. Vérifie d'abord que les deux vecteurs ont la même longueur (sinon, `ValueError`), et renvoie une **nouvelle** liste de `float` (`float(a + b)`), sans modifier les entrées.

Vérifications, avec les deux manchots de 0B.8, `p1` et `p2` : a) `vector_add(p1, p2)` · b) `vector_subtract(p1, p2)` · c) le « manchot moyen » `scalar_multiply(0.5, vector_add(p1, p2))` · d) `hadamard([1, 2, 2], [2, 0, -1])` · e) le nom de l'exception levée par `vector_add([1, 2], [1, 2, 3])` · puis les tests.""",
       check=RELOAD + r'''with wb.attempt("0B.38"):
    lb = mylearn.linalg_basics
    wb.check("0B.38a", lb.vector_add(p1, p2))
    wb.check("0B.38b", lb.vector_subtract(p1, p2))
    wb.check("0B.38c", lb.scalar_multiply(0.5, lb.vector_add(p1, p2)))
    wb.check("0B.38d", lb.hadamard([1, 2, 2], [2, 0, -1]))
    wb.check("0B.38e", error_name(lb.vector_add, [1, 2], [1, 2, 3]))
    run_linalg_tests("vector_add or vector_subtract or scalar_multiply or hadamard or elementwise")''',
       solution=r'''lb = mylearn.linalg_basics
print(lb.vector_add(p1, p2), lb.vector_subtract(p1, p2), lb.scalar_multiply(0.5, lb.vector_add(p1, p2)),
      lb.hadamard([1, 2, 2], [2, 0, -1]), error_name(lb.vector_add, [1, 2], [1, 2, 3]))
print("with lists, + concatenates:", p1 + p2)
run_linalg_tests("vector_add or vector_subtract or scalar_multiply or hadamard or elementwise", impl="ref")''',
       record=r'''wb.record("0B.38a", lb.vector_add(p1, p2), mistakes={"sur des listes, + les COLLE : additionne composante par composante": p1 + p2})
wb.record("0B.38b", lb.vector_subtract(p1, p2))
wb.record("0B.38c", lb.scalar_multiply(0.5, lb.vector_add(p1, p2)), decimals=1)
wb.record("0B.38d", lb.hadamard([1, 2, 2], [2, 0, -1]))
wb.record("0B.38e", error_name(lb.vector_add, [1, 2], [1, 2, 3]), mistakes={"zip s'arrête silencieusement au plus court : vérifie les longueurs avant": "no error"})''',
       note="La référence est dans `solutions/mylearn_ref/linalg_basics.py` : lis-la **après** avoir réussi les "
            "tests. Le piège de e : `zip` ne signale pas des longueurs différentes, il s'arrête au plus court "
            "(depuis Python 3.10, `zip(u, v, strict=True)` lève une `ValueError`)."),

    Ex("0B.39", "🔨", 2, 20, "`linalg_basics` (2) : produit scalaire, norme, distance, cosinus",
       "écrire le produit scalaire, les normes L1, L2 et L∞, la distance et la similarité cosinus.",
       "Ex 0B.18 · 0A · fiche §101.3.2, §101.3.3", tracks="R, M, C",
       body=MYLEARN_HOWTO + r"""

Écris `dot`, `norm`, `distance` et `cosine_similarity`.
- `norm(v, p=2)` couvre plusieurs normes : `p = 2` (euclidienne), `p = 1` (somme des valeurs absolues), `p = math.inf` (plus grande valeur absolue), et plus généralement $\left(\sum_i |v_i|^p\right)^{1/p}$ pour tout $p \geq 1$.
- `distance` : réutilise `vector_subtract` et `norm` si tu as fait 0B.38 ; sinon, calcule-la directement.
- `cosine_similarity` refuse un vecteur nul (`ValueError`) : l'angle n'est pas défini. Les erreurs d'arrondi peuvent donner `1.0000000000000002` : ramène le résultat dans $[-1, 1]$.

Vérifications : a) `dot([1, 2, 2], [2, 0, -1])` · b) la liste des normes L2, L1 et L∞ de `[3, -4]` · c) `distance(p1, p2)` · d) `cosine_similarity([1, 2, 2], [2, 4, 4])` · e) `cosine_similarity([3, 1], [1, 2])` (3 décimales) · puis les tests.""",
       check=RELOAD + r'''with wb.attempt("0B.39"):
    lb = mylearn.linalg_basics
    wb.check("0B.39a", lb.dot([1, 2, 2], [2, 0, -1]))
    wb.check("0B.39b", [lb.norm([3, -4]), lb.norm([3, -4], p=1), lb.norm([3, -4], p=math.inf)])
    wb.check("0B.39c", lb.distance(p1, p2))
    wb.check("0B.39d", lb.cosine_similarity([1, 2, 2], [2, 4, 4]))
    wb.check("0B.39e", lb.cosine_similarity([3, 1], [1, 2]))
    run_linalg_tests("(dot or norm or distance or cosine) and not hadamard")''',
       solution=r'''lb = mylearn.linalg_basics
print(lb.dot([1, 2, 2], [2, 0, -1]), [lb.norm([3, -4]), lb.norm([3, -4], p=1), lb.norm([3, -4], p=math.inf)],
      lb.distance(p1, p2), lb.cosine_similarity([1, 2, 2], [2, 4, 4]), lb.cosine_similarity([3, 1], [1, 2]))
run_linalg_tests("(dot or norm or distance or cosine) and not hadamard", impl="ref")''',
       record=r'''wb.record("0B.39a", int(lb.dot([1, 2, 2], [2, 0, -1])))
wb.record("0B.39b", [lb.norm([3, -4]), lb.norm([3, -4], p=1), lb.norm([3, -4], p=math.inf)],
          mistakes={"L1 et L∞ utilisent des VALEURS ABSOLUES": [5, -1, 3]})
wb.record("0B.39c", lb.distance(p1, p2), decimals=1)
wb.record("0B.39d", lb.cosine_similarity([1, 2, 2], [2, 4, 4]), decimals=3, mistakes={"le produit scalaire seul ne suffit pas : divise-le par le produit des deux normes": 18.0, "divise par le produit des DEUX normes, pas par une seule": 6.0, "divise par le produit des deux normes, pas par une seule": 3.0})
wb.record("0B.39e", lb.cosine_similarity([3, 1], [1, 2]), decimals=3)''',
       note="d vaut 1 : les deux vecteurs ont la même direction (le second est le double du premier). e retrouve "
            "l'exemple de la fiche, un angle de 45°. `norm` avec `p = math.inf` se traite à part : "
            "$|v_i|^{\\infty}$ n'a pas de sens en calcul flottant."),

    Ex("0B.40", "📦", 2, 15, "Tes fonctions contre NumPy : mêmes résultats, autre vitesse",
       "traduire ses fonctions en appels NumPy, vérifier qu'ils concordent et mesurer l'écart de vitesse.",
       "Ex 0B.39 · fiche §101.3", tracks="R, M, C",
       body=r"""NumPy fait tout cela en une instruction. Pour les arrays `a` et `b` fournis :

a) `np_dot` : leur produit scalaire (`a @ b`) · b) `np_norms` : la liste des normes L2, L1 et L∞ de `a` (`np.linalg.norm(a, ord=...)`, avec `ord=np.inf` pour L∞ ; 2 décimales) · c) `np_cos` : leur similarité cosinus (3 décimales).

Puis compare avec ta librairie (la cellule de vérification s'en charge) : d) sur deux vecteurs aléatoires de dimension 1000 (`u_1000`, `v_1000`), l'écart maximal entre ton `dot` et `np.dot`, ton `norm` et `np.linalg.norm`, ta `cosine_similarity` et la formule NumPy est-il inférieur à $10^{-9}$ ?

e) Écris `speedup_dot()` : avec `measure` (fournie en partie A), le temps de **ton** `dot` sur `u_big` et `v_big` (100 000 composantes, convertis en listes avec `.tolist()` pour ta fonction), divisé par le temps de `np.dot` sur les arrays. La vérification regarde si NumPy est au moins 10 fois plus rapide. D'où vient l'écart (0A.55) ?""",
       given=r'''a = np.array([2.0, -1.0, 4.0, 0.5])
b = np.array([1.0, 3.0, -2.0, 4.0])
rng_40 = np.random.default_rng(0)
u_1000, v_1000 = rng_40.normal(size=1000), rng_40.normal(size=1000)
u_big, v_big = rng_40.normal(size=100_000), rng_40.normal(size=100_000)''',
       todo=r'''np_dot = ...     # a)
np_norms = ...   # b) [L2, L1, L∞] of a
np_cos = ...     # c)


def speedup_dot():
    """(time of YOUR dot on lists) / (time of np.dot on arrays), for u_big and v_big."""
    raise NotImplementedError("speedup_dot() is not written yet")''',
       check=RELOAD + r'''wb.check("0B.40a", np_dot)
wb.check("0B.40b", np_norms)
wb.check("0B.40c", np_cos)
with wb.attempt("0B.40d"):
    lb = mylearn.linalg_basics
    u_list, v_list = u_1000.tolist(), v_1000.tolist()
    gaps = [abs(lb.dot(u_list, v_list) - np.dot(u_1000, v_1000)),
            abs(lb.norm(u_list) - np.linalg.norm(u_1000)),
            abs(lb.cosine_similarity(u_list, v_list)
                - u_1000 @ v_1000 / (np.linalg.norm(u_1000) * np.linalg.norm(v_1000)))]
    print("largest gap:", max(gaps))
    wb.check("0B.40d", bool(max(gaps) < 1e-9))
with wb.attempt("0B.40e"):
    speedup = speedup_dot()
    print(f"NumPy is {speedup:.0f} times faster")
    verdict("0B.40e", speedup >= 10, f"NumPy est {speedup:.0f} fois plus rapide que ta boucle : objectif atteint.",
            f"NumPy n'est que {speedup:.0f} fois plus rapide : relance la cellule (une mesure de temps fluctue quand "
            "la machine est occupée) ; si l'écart reste faible, vérifie que ton dot reçoit des listes et np.dot des "
            "arrays.")''',
       solution=r'''np_dot = a @ b
np_norms = [np.linalg.norm(a), np.linalg.norm(a, ord=1), np.linalg.norm(a, ord=np.inf)]
np_cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def speedup_dot():
    """(time of YOUR dot on lists) / (time of np.dot on arrays), for u_big and v_big."""
    u_list, v_list = u_big.tolist(), v_big.tolist()
    return measure(mylearn.linalg_basics.dot, u_list, v_list) / measure(np.dot, u_big, v_big)


lb = mylearn.linalg_basics
u_list, v_list = u_1000.tolist(), v_1000.tolist()
gaps = [abs(lb.dot(u_list, v_list) - np.dot(u_1000, v_1000)),
        abs(lb.norm(u_list) - np.linalg.norm(u_1000)),
        abs(lb.cosine_similarity(u_list, v_list) - u_1000 @ v_1000 / (np.linalg.norm(u_1000) * np.linalg.norm(v_1000)))]
speedup = speedup_dot()
if speedup < 10:      # a timing measured on a busy machine: never publish it, run the notebook again
    raise RuntimeError(f"NumPy only {speedup:.0f} times faster: the machine is busy, run this notebook again later")
print(np_dot, np.round(np_norms, 2), round(np_cos, 3), max(gaps), f"NumPy is {speedup:.0f} times faster")''',
       record=r'''wb.record("0B.40a", np_dot, decimals=1)
wb.record("0B.40b", np_norms, decimals=2)
wb.record("0B.40c", np_cos, decimals=3)
wb.record("0B.40d", bool(max(gaps) < 1e-9))''',
       note="Les écarts sont nuls ou minuscules (au plus $10^{-14}$ environ) : quand les additions ne se font pas dans "
            "le même ordre, l'arrondi flottant peut différer au dernier chiffre. Le rapport de vitesse (souvent 50 à 500) vient, comme en 0A.55, "
            "de l'interprétation de chaque instruction Python et des objets `float` créés un par un, que NumPy "
            "évite en travaillant sur un bloc de mémoire en code compilé."),

    Ex("0B.41", "🔬", 2, 20, "Distance ou similarité cosinus : l'effet de la longueur",
       "comparer distance euclidienne et similarité cosinus sur des sacs de mots, et voir l'effet de la normalisation.",
       "Ex 0B.40 · fiche §101.3.2, §101.3.3", thread="synthétique", tracks="M, C",
       body=r"""Un texte peut être représenté par un **sac de mots** (*bag of words*) : le nombre de fois où chaque mot d'un vocabulaire y apparaît (tu en construiras pour classer des phrases au ch. 13). La requête `query` parle de manchots ; `docs` contient trois documents : un long texte sur les manchots, un court sur les matrices, et un texte mélangé. Tu peux utiliser NumPy ou ta librairie.

a) `nearest_distance` : le **nom** du document le plus proche de `query` au sens de la distance euclidienne (par exemple `min(docs, key=lambda name: ...)`) · b) `nearest_cosine` : le nom de celui qui a la plus grande similarité cosinus avec `query` · c) `nearest_normalized` : le nom du plus proche en distance après avoir **normalisé** tous les vecteurs (chacun divisé par sa norme).

d) Expérience : écris `length_experiment()`, qui multiplie `query` par $k = 1, 2, \dots, 10$ (le même texte, $k$ fois plus long) et renvoie deux listes, `distances` et `cosines` : la distance et la similarité cosinus entre `k * query` et `docs["penguins_long"]`. La vérification regarde si les cosinus sont tous égaux, puis e) la distance pour $k = 10$ (2 décimales), et trace les deux courbes.

Dans ta copie : laquelle des deux mesures dépend de la longueur du texte ? Laquelle choisir pour comparer des textes de longueurs différentes, et que change la normalisation ?""",
       given=r'''vocabulary = ["penguin", "ice", "fish", "matrix", "vector", "product"]
query = np.array([2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
docs = {
    "penguins_long": np.array([6.0, 4.0, 3.0, 0.0, 1.0, 0.0]),
    "matrices_short": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    "mixed": np.array([1.0, 0.0, 1.0, 2.0, 1.0, 0.0]),
}''',
       todo=r'''nearest_distance = ...     # a) a name (a key of docs), as a string
nearest_cosine = ...       # b)
nearest_normalized = ...   # c)


def length_experiment():
    """Distances and cosine similarities between k * query and docs["penguins_long"], for k = 1..10."""
    raise NotImplementedError("length_experiment() is not written yet")''',
       check=r'''wb.check("0B.41a", nearest_distance)
wb.check("0B.41b", nearest_cosine)
wb.check("0B.41c", nearest_normalized)
with wb.attempt("0B.41d"):
    distances, cosines = length_experiment()
    wb.check("0B.41d", bool(np.ptp(cosines) < 1e-9))
    wb.check("0B.41e", distances[-1])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(range(1, 11), distances, marker="o")
    axes[0].set_title("distance to penguins_long")
    axes[1].plot(range(1, 11), cosines, marker="o")
    axes[1].set_title("cosine similarity")
    for ax in axes:
        ax.set_xlabel("k (the query repeated k times)")
    fig.tight_layout()
    plt.show()''',
       solution=r'''def cosine(u, v):
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))


nearest_distance = min(docs, key=lambda name: np.linalg.norm(query - docs[name]))
nearest_cosine = max(docs, key=lambda name: cosine(query, docs[name]))
unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
nearest_normalized = min(docs, key=lambda name: np.linalg.norm(unit(query) - unit(docs[name])))


def length_experiment():
    """Distances and cosine similarities between k * query and docs["penguins_long"], for k = 1..10."""
    target = docs["penguins_long"]
    distances = [np.linalg.norm(k * query - target) for k in range(1, 11)]
    cosines = [cosine(k * query, target) for k in range(1, 11)]
    return distances, cosines


for name, doc in docs.items():
    print(f"{name:15} distance {np.linalg.norm(query - doc):.2f}  cosine {cosine(query, doc):.3f}  "
          f"normalized distance {np.linalg.norm(unit(query) - unit(doc)):.3f}")
distances, cosines = length_experiment()
print(nearest_distance, nearest_cosine, nearest_normalized, np.round(distances, 2), np.round(cosines, 3))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(range(1, 11), distances, marker="o")
axes[0].set_title("distance to penguins_long")
axes[1].plot(range(1, 11), cosines, marker="o")
axes[1].set_title("cosine similarity")
for ax in axes:
    ax.set_xlabel("k (the query repeated k times)")
fig.tight_layout()
plt.show()''',
       record=r'''wb.record("0B.41a", nearest_distance, mistakes={"le plus PROCHE en distance, c'est la plus PETITE distance (min)": "penguins_long"})
wb.record("0B.41b", nearest_cosine, mistakes={"la plus grande similarité cosinus (max), pas la plus petite": "matrices_short"})
wb.record("0B.41c", nearest_normalized, mistakes={"après normalisation, seule la direction des vecteurs compte, plus leur longueur": "mixed"})
wb.record("0B.41d", bool(np.ptp(cosines) < 1e-9))
wb.record("0B.41e", distances[-1], decimals=2)''',
       note="La distance classe en premier le texte « mixed », simplement parce qu'il est court comme la "
            "requête ; le texte long sur les manchots est loin, alors qu'il parle du même sujet. La similarité "
            "cosinus ne regarde que la direction (les proportions des mots) : répéter un texte ne la change pas. "
            "Après normalisation, les deux critères donnent le même classement ($\\|\\mathbf{a} - \\mathbf{b}\\|^2 "
            "= 2 - 2\\cos$, 0B.19) : c'est pourquoi on normalise les embeddings (0B.E2, B2 et B4)."),

    Ex("0B.42", "🔨", 2, 20, "`linalg_basics` (3) : forme, transposée, identité, matrice × vecteur",
       "écrire la forme, la transposée, l'identité et le produit matrice-vecteur d'une matrice stockée en liste de lignes.",
       "Ex 0B.39, Ex 0B.9 · fiche §101.4.1, §101.4.2, §101.4.4", tracks="R, M, C",
       body=MYLEARN_HOWTO + r"""

Écris `shape`, `transpose`, `identity` et `matvec`. Une matrice est une **liste de lignes**.
- `shape` vérifie que la matrice est bien rectangulaire (au moins une ligne, des lignes non vides et toutes de même longueur ; sinon `ValueError`) ; `transpose` et `matvec` s'en servent pour valider leurs entrées.
- `transpose` : la ligne $j$ du résultat est la colonne $j$ de la matrice.
- `identity(n)` : attention, `[[0.0] * n] * n` crée $n$ fois **la même** liste (fiche 0A, §100.3.1, « mutabilité et alias ») : modifier une ligne les modifierait toutes.
- `matvec(A, v)` : un produit scalaire par ligne (réutilise `dot`) ; si les formes ne vont pas, une `ValueError` dont le message montre les deux formes, par exemple `cannot multiply (2, 3) by (2,)`.

Vérifications, avec les matrices de 0B.9 : a) `shape(A9)` · b) `transpose(A9)` · c) `matvec(A9, [1, 2, -1])` · d) les prédictions $\mathbf{X}\mathbf{w} + b$ de 0B.9 f, avec `matvec`, puis `b` ajouté à chaque composante · e) `matvec(identity(3), [1.5, -2.0, 3.25])` · f) le message d'erreur de `matvec(A9, [1, 2])` contient-il les deux formes `(2, 3)` et `(2,)` ? · puis les tests.""",
       given=r'''A9 = [[2, 0, 1], [-1, 3, 2]]          # the matrices of 0B.9, as lists of rows
X9 = [[1, 2], [3, 0], [0, -1]]
w9, b9 = [0.5, 2], 1''',
       check=RELOAD + r'''with wb.attempt("0B.42"):
    lb = mylearn.linalg_basics
    wb.check("0B.42a", lb.shape(A9))
    wb.check("0B.42b", lb.transpose(A9))
    wb.check("0B.42c", lb.matvec(A9, [1, 2, -1]))
    wb.check("0B.42d", [y + b9 for y in lb.matvec(X9, w9)])
    wb.check("0B.42e", lb.matvec(lb.identity(3), [1.5, -2.0, 3.25]))
    message = error_message(lb.matvec, A9, [1, 2])
    print("message:", message)
    wb.check("0B.42f", "(2, 3)" in message and "(2,)" in message)
    run_linalg_tests("shape_matches or shape_rejects or transpose or identity or matvec")''',
       solution=r'''lb = mylearn.linalg_basics
predictions_9 = [y + b9 for y in lb.matvec(X9, w9)]
message = error_message(lb.matvec, A9, [1, 2])
print(lb.shape(A9), lb.transpose(A9), lb.matvec(A9, [1, 2, -1]), predictions_9,
      lb.matvec(lb.identity(3), [1.5, -2.0, 3.25]), message)
run_linalg_tests("shape_matches or shape_rejects or transpose or identity or matvec", impl="ref")''',
       record=r'''wb.record("0B.42a", lb.shape(A9), mistakes={"(nombre de lignes, nombre de colonnes)": (3, 2)})
wb.record("0B.42b", lb.transpose(A9))
wb.record("0B.42c", lb.matvec(A9, [1, 2, -1]))
wb.record("0B.42d", predictions_9, decimals=1, mistakes={"n'oublie pas le biais b = 1": lb.matvec(X9, w9)})
wb.record("0B.42e", lb.matvec(lb.identity(3), [1.5, -2.0, 3.25]), decimals=2)
wb.record("0B.42f", "(2, 3)" in message and "(2,)" in message)''',
       note="Un message d'erreur qui montre les formes fait gagner un temps précieux : c'est ce que font NumPy et "
            "PyTorch (`mat1 and mat2 shapes cannot be multiplied (64x784 and 128x10)`). `matvec(identity(n), v)` "
            "redonne `v` : l'identité joue le rôle du nombre 1."),

    Ex("0B.43", "🔨", 2, 25, "`linalg_basics` (4) : `matmul` et vérification des formes",
       "écrire le produit matriciel en Python pur, avec la vérification des formes.",
       "Ex 0B.42, Ex 0B.20 · fiche §101.4.3", tracks="R, M, C",
       body=MYLEARN_HOWTO + r"""

Écris `matmul(A, B)`. Méthode conseillée :
1. les formes, avec `shape` : si le nombre de colonnes de `A` diffère du nombre de lignes de `B`, une `ValueError` dont le message montre les deux formes (`cannot multiply (2, 3) by (2, 2)`) ;
2. `transpose(B)`, pour accéder facilement aux **colonnes** de `B` ;
3. l'élément $(i, j)$ du résultat est `dot(A[i], colonne_j)` : une compréhension imbriquée suffit.

Vérifications, avec les matrices de 0B.20 : a) `matmul(B20, A20)` · b) `shape(matmul(A20, B20))` · c) `matmul(A20, C20)` · d) le nom de l'exception levée par `matmul(C20, A20)` · puis les tests, qui comparent aussi `matmul` à `np.matmul` sur des matrices aléatoires.""",
       given=r'''A20 = [[1, 2], [0, -1], [3, 1]]      # the matrices of 0B.20
B20 = [[2, 1, 0], [1, -1, 4]]
C20 = [[1, 1], [2, 0]]''',
       check=RELOAD + r'''with wb.attempt("0B.43"):
    lb = mylearn.linalg_basics
    wb.check("0B.43a", lb.matmul(B20, A20))
    wb.check("0B.43b", lb.shape(lb.matmul(A20, B20)))
    wb.check("0B.43c", lb.matmul(A20, C20))
    wb.check("0B.43d", error_name(lb.matmul, C20, A20))
    run_linalg_tests("matmul")''',
       solution=r'''lb = mylearn.linalg_basics
print(lb.matmul(B20, A20), lb.shape(lb.matmul(A20, B20)), lb.matmul(A20, C20), error_message(lb.matmul, C20, A20))
print(np.array_equal(lb.matmul(A20, B20), np.array(A20) @ np.array(B20)))
run_linalg_tests("matmul", impl="ref")''',
       record=r'''wb.record("0B.43a", lb.matmul(B20, A20))
wb.record("0B.43b", lb.shape(lb.matmul(A20, B20)), mistakes={"(m, n) × (n, p) → (m, p) : on garde les dimensions extérieures": (2, 2)})
wb.record("0B.43c", lb.matmul(A20, C20))
wb.record("0B.43d", error_name(lb.matmul, C20, A20), mistakes={"C est (2, 2) et A (3, 2) : 2 ≠ 3, le produit n'existe pas ; vérifie les formes AVANT de calculer": "no error"})''',
       note="Ta fonction fait exactement $m \\times n \\times p$ multiplications (0B.20 h), comme la formule "
            "$(\\mathbf{A}\\mathbf{B})_{ij} = \\sum_k A_{ik} B_{kj}$. NumPy fait le même calcul, mais découpé en "
            "blocs qui tiennent dans la mémoire cache du processeur, et en parallèle : le résultat est identique, "
            "la vitesse sans commune mesure (0B.54)."),

    Ex("0B.44", "🔮", 2, 15, "AB = BA ? (AB)ᵀ = BᵀAᵀ ? Prédire, puis tester",
       "prévoir quelles règles de calcul valent pour le produit matriciel, puis les tester sur des exemples.",
       "Ex 0B.43 · fiche §101.4.3", tracks="R, M, C", hypothesis=True,
       body=r"""**Sans calculer**, prédis `True` (toujours vrai) ou `False` (faux en général) pour des matrices carrées $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$ quelconques de même taille :

a) $\mathbf{A}\mathbf{B} = \mathbf{B}\mathbf{A}$ · b) $(\mathbf{A}\mathbf{B})^\top = \mathbf{A}^\top\mathbf{B}^\top$ · c) $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$ · d) $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$ · e) $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{A}\mathbf{B} + \mathbf{A}\mathbf{C}$ · f) pour deux matrices **diagonales** $\mathbf{D}_1$ et $\mathbf{D}_2$ (des zéros hors de la diagonale) : $\mathbf{D}_1\mathbf{D}_2 = \mathbf{D}_2\mathbf{D}_1$.

Puis l'**Expérience** teste chaque égalité sur des matrices aléatoires à coefficients entiers (des entiers, pour que les égalités soient exactes). Un seul contre-exemple suffit à réfuter une règle ; des milliers d'exemples qui la vérifient ne la démontrent pas, mais donnent confiance.""",
       todo=r'''prediction_0B_44a = ...   # a) AB = BA: True or False
prediction_0B_44b = ...   # b) (AB)^T = A^T B^T
prediction_0B_44c = ...   # c) (AB)^T = B^T A^T
prediction_0B_44d = ...   # d) (AB)C = A(BC)
prediction_0B_44e = ...   # e) A(B + C) = AB + AC
prediction_0B_44f = ...   # f) D1 D2 = D2 D1 for diagonal matrices''',
       check=r'''predictions_0B_44 = [prediction_0B_44a, prediction_0B_44b, prediction_0B_44c,
                     prediction_0B_44d, prediction_0B_44e, prediction_0B_44f]
for letter, prediction in zip("abcdef", predictions_0B_44):
    wb.check(f"0B.44{letter}", prediction)''',
       solution=r'''prediction_0B_44a = False   # not commutative
prediction_0B_44b = False   # the order must be reversed...
prediction_0B_44c = True    # ...like this
prediction_0B_44d = True    # associative
prediction_0B_44e = True    # distributive
prediction_0B_44f = True    # diagonal matrices commute (a special case)''',
       record=r'''wb.record("0B.44a", prediction_0B_44a, mistakes={"le produit matriciel n'est pas commutatif (fiche §101.4.3) : l'ordre compte": True})
wb.record("0B.44b", prediction_0B_44b, mistakes={"la transposée d'un produit INVERSE l'ordre des facteurs": True})
wb.record("0B.44c", prediction_0B_44c)
wb.record("0B.44d", prediction_0B_44d, mistakes={"le produit est associatif : seul le coût du calcul change (0B.30)": False})
wb.record("0B.44e", prediction_0B_44e)
wb.record("0B.44f", prediction_0B_44f, mistakes={"multiplie deux matrices diagonales 2 × 2 à la main : que devient chaque élément de la diagonale ?": False})''',
       after=[("md", "**Expérience** : 1 000 essais par règle, sur des matrices 3 × 3 aléatoires. Une règle est « toujours vraie » ici si aucun essai ne la contredit."),
              ("code", r'''rng_44 = np.random.default_rng(0)
counterexamples = {"a) AB = BA": 0, "b) (AB)T = AT BT": 0, "c) (AB)T = BT AT": 0,
                   "d) (AB)C = A(BC)": 0, "e) A(B + C) = AB + AC": 0, "f) D1 D2 = D2 D1": 0}
for trial in range(1000):
    A, B, C = (rng_44.integers(-5, 6, size=(3, 3)) for _ in range(3))
    D1, D2 = np.diag(rng_44.integers(-5, 6, size=3)), np.diag(rng_44.integers(-5, 6, size=3))
    equalities = [np.array_equal(A @ B, B @ A), np.array_equal((A @ B).T, A.T @ B.T),
                  np.array_equal((A @ B).T, B.T @ A.T), np.array_equal((A @ B) @ C, A @ (B @ C)),
                  np.array_equal(A @ (B + C), A @ B + A @ C), np.array_equal(D1 @ D2, D2 @ D1)]
    for rule, equal in zip(counterexamples, equalities):
        counterexamples[rule] += not equal
for rule, count in counterexamples.items():
    print(f"{rule:24} {'always true here' if count == 0 else f'false in {count} trials out of 1000'}")''')],
       note="La non-commutativité est la règle, la commutativité l'exception (matrices diagonales, identité, une "
            "matrice et elle-même…). c) se démontre à partir de la formule $(\\mathbf{A}\\mathbf{B})_{ij} = \\sum_k "
            "A_{ik} B_{kj}$ ; b) n'a même pas de sens en général pour des matrices rectangulaires, car les formes ne "
            "s'emboîtent pas. Retiens : pour transposer un produit, on transpose chaque facteur **et** on inverse "
            "l'ordre : on met les chaussettes puis les chaussures, on enlève les chaussures puis les chaussettes."),

    Ex("0B.45", "🐛", 2, 20, "Le produit qui n'en est pas un : `*`, `@`, `(n,)` et `(n, 1)`",
       "diagnostiquer les confusions entre produit élément par élément, produit matriciel et formes de vecteurs.",
       "Ex 0B.44 · fiche §101.4.1, §101.4.3, §101.3.4", tracks="R, C",
       body=r"""Trois fonctions d'un modèle linéaire, trois bugs. La cellule fournie les essaie et montre seulement les symptômes : une forme inattendue, une erreur, pas de matrice.

1. `predict(X, w, b)` doit renvoyer **un nombre par exemple**, $\mathbf{X}\mathbf{w} + b$.
2. `layer(X, W, b)` doit renvoyer $\mathbf{X}\mathbf{W} + \mathbf{b}$ : une ligne par exemple, une colonne par neurone.
3. `outer(u, v)` doit renvoyer la matrice de **tous** les produits $u_i v_j$, de forme `(len(u), len(v))` : le produit extérieur (*outer product*) $\mathbf{u}\mathbf{v}^\top$ d'une colonne par une ligne.

Diagnostique d'abord **en raisonnant sur les formes** (écris-les), puis vérifie dans une cellule à toi, avant de corriger : a) la forme renvoyée par la version buggée de `predict`, c'est-à-dire celle de `X45 * w45 + 1.0` (un tuple) · b) le nom de l'exception levée par `W45 @ X45` · c) la valeur de `u45 @ v45.T` (que fait `.T` sur un vecteur de forme `(3,)` ?).

Corrige ensuite les trois fonctions (sans `np.outer`) ; vérifications : d) `predict(X45, w45, 1.0)` · e) `layer(X45, W45, bias45)` · f) `outer(u45, v45)`.""",
       given=r'''X45 = np.array([[1.0, 2.0, 0.0],
                [0.0, 1.0, 1.0],
                [2.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [3.0, 0.0, 2.0]])              # 5 examples, 3 features
w45 = np.array([0.5, -1.0, 2.0])
W45 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])   # a layer of 2 neurons: (n_in, n_out) = (3, 2)
bias45 = np.array([0.5, -0.5])
u45, v45 = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])


def predict(X, w, b):
    """Predictions X w + b of a linear model: one number per example."""
    return X * w + b


def layer(X, W, b):
    """Outputs X W + b of a dense layer: one row per example, one column per neuron."""
    return W @ X + b


def outer(u, v):
    """The outer product u v^T: the matrix of all the products u_i v_j."""
    return u @ v.T


# Symptoms (without the details: finding them is your job)
print("predict: shape (5,) expected ->", predict(X45, w45, 1.0).shape == (5,))
try:
    layer(X45, W45, bias45)
    print("layer: no error")
except Exception:  # noqa: BLE001
    print("layer: an exception is raised")
print("outer: a 3 x 3 matrix expected ->", np.shape(outer(u45, v45)) == (3, 3))''',
       todo=r'''buggy_predict_shape = ...   # a) a tuple
buggy_layer_error = ...     # b) the name of the exception
buggy_outer_value = ...     # c)


def predict(X, w, b):
    raise NotImplementedError("predict() is not fixed yet")


def layer(X, W, b):
    raise NotImplementedError("layer() is not fixed yet")


def outer(u, v):
    raise NotImplementedError("outer() is not fixed yet")''',
       check=r'''wb.check("0B.45a", buggy_predict_shape)
wb.check("0B.45b", buggy_layer_error)
wb.check("0B.45c", buggy_outer_value)
with wb.attempt("0B.45d"):
    wb.check("0B.45d", predict(X45, w45, 1.0))
with wb.attempt("0B.45e"):
    wb.check("0B.45e", layer(X45, W45, bias45))
with wb.attempt("0B.45f"):
    wb.check("0B.45f", outer(u45, v45))''',
       solution=r'''buggy_predict_shape = (5, 3)      # X * w broadcasts w along each row: element-wise, not a product
buggy_layer_error = "ValueError"  # (3, 2) @ (5, 3): 2 != 5
buggy_outer_value = 32.0          # .T does nothing on a 1-D array: u @ v is the dot product


def predict(X, w, b):
    return X @ w + b                         # (5, 3) @ (3,) -> (5,)


def layer(X, W, b):
    return X @ W + b                         # (5, 3) @ (3, 2) -> (5, 2), then b (2,) broadcast on each row


def outer(u, v):
    return u[:, None] @ v[None, :]           # (3, 1) @ (1, 3) -> (3, 3)


print(predict(X45, w45, 1.0), layer(X45, W45, bias45), outer(u45, v45), sep="\n")''',
       record=r'''wb.record("0B.45a", buggy_predict_shape)
wb.record("0B.45b", buggy_layer_error)
wb.record("0B.45c", buggy_outer_value, decimals=1)
wb.record("0B.45d", predict(X45, w45, 1.0), decimals=1, mistakes={"le biais b est ajouté à chaque prédiction": predict(X45, w45, 0.0)})
wb.record("0B.45e", layer(X45, W45, bias45), decimals=1)
wb.record("0B.45f", outer(u45, v45), decimals=1)''',
       note="Trois réflexes : `*` est élément par élément, `@` est le produit matriciel ; écris les formes avant "
            "d'écrire le calcul (`(5, 3) @ (3, 2) → (5, 2)`) ; un vecteur `(n,)` n'est ni une ligne ni une "
            "colonne, et `.T` ne le change pas : pour une colonne, `u[:, None]` ou `u.reshape(-1, 1)`. Le bug 1 est "
            "le plus dangereux : aucune erreur, juste une forme fausse qui se propagera plus loin."),

    Ex("0B.46", "📦", 2, 15, "Inverse et systèmes : `np.linalg.inv` et `np.linalg.solve`",
       "calculer déterminant, inverse et solution d'un système avec NumPy, et savoir pourquoi on préfère `solve`.",
       "Ex 0B.21, Ex 0B.43 · fiche §101.4.4", tracks="M, C",
       body=r"""a) `det_M` : le déterminant de $\mathbf{M}$ (0B.21) avec `np.linalg.det` (6 décimales) ; le calcul est fait en virgule flottante, le résultat peut s'écarter de la valeur exacte au 16ᵉ chiffre.
b) `M_inv` : $\mathbf{M}^{-1}$ avec `np.linalg.inv` (2 décimales) ; vérifie dans ta copie que `M46 @ M_inv` est proche de l'identité (`np.allclose(..., np.eye(2))`).
c) `x_2` : la solution de $3x + y = 5$, $4x + 2y = 6$ avec `np.linalg.solve`.
d) `singular_error` : le nom (court) de l'exception levée par `np.linalg.inv` sur la matrice `singular46` de 0B.21 d, par exemple avec `error_name` (partie A).
e) `x_3` : la solution du système $2x + y - z = 8$, $-3x - y + 2z = -11$, $-2x + y + 2z = -3$ (écris la matrice des coefficients et le second membre).
f) Pour le système aléatoire `A_big` ($500 \times 500$) et `b_big` : `solve_is_accurate`, l'écart $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|$ de `np.linalg.solve` est-il inférieur à $10^{-8}$ ? Dans ta copie, compare aussi avec `np.linalg.inv(A_big) @ b_big` : écarts et temps (`measure`). Lequel choisir ?""",
       given=r'''M46 = np.array([[3.0, 1.0], [4.0, 2.0]])
singular46 = np.array([[2.0, 4.0], [1.0, 2.0]])
rng_46 = np.random.default_rng(0)
A_big, b_big = rng_46.normal(size=(500, 500)), rng_46.normal(size=500)''',
       todo=r'''det_M = ...               # a)
M_inv = ...               # b)
x_2 = ...                 # c)
singular_error = ...      # d)
x_3 = ...                 # e)
solve_is_accurate = ...   # f)''',
       check=r'''wb.check("0B.46a", det_M)
wb.check("0B.46b", M_inv)
wb.check("0B.46c", x_2)
wb.check("0B.46d", singular_error)
wb.check("0B.46e", x_3)
wb.check("0B.46f", solve_is_accurate)''',
       solution=r'''det_M = np.linalg.det(M46)
M_inv = np.linalg.inv(M46)
x_2 = np.linalg.solve(M46, [5.0, 6.0])
try:
    np.linalg.inv(singular46)
    singular_error = "no error"
except np.linalg.LinAlgError as error:
    singular_error = type(error).__name__
    print("LinAlgError:", error)
A3 = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
x_3 = np.linalg.solve(A3, [8.0, -11.0, -3.0])
x_solve = np.linalg.solve(A_big, b_big)
x_inv = np.linalg.inv(A_big) @ b_big
solve_is_accurate = bool(np.linalg.norm(A_big @ x_solve - b_big) < 1e-8)
print(det_M, M_inv, np.allclose(M46 @ M_inv, np.eye(2)), x_2, x_3, sep="\n")
print(f"solve: gap {np.linalg.norm(A_big @ x_solve - b_big):.1e}, {measure(np.linalg.solve, A_big, b_big) * 1000:.1f} ms")
print(f"inv:   gap {np.linalg.norm(A_big @ x_inv - b_big):.1e}, "
      f"{measure(lambda: np.linalg.inv(A_big) @ b_big) * 1000:.1f} ms")''',
       record=r'''wb.record("0B.46a", det_M, decimals=6)
wb.record("0B.46b", M_inv, decimals=2, mistakes={"n'oublie pas de diviser par le déterminant": [[2, -1], [-4, 3]]})
wb.record("0B.46c", x_2, decimals=2)
wb.record("0B.46d", singular_error)
wb.record("0B.46e", x_3, decimals=2)
wb.record("0B.46f", solve_is_accurate)''',
       note="`solve` résout le système sans former l'inverse (par élimination, comme au lycée, mais organisée) : "
            "environ trois fois moins de calculs, et un résultat au moins aussi précis. Règle pratique : si tu "
            "écris `inv(A) @ b`, remplace-le par `solve(A, b)`. L'inverse sert surtout en théorie : par exemple "
            "dans la formule des moindres carrés de la régression linéaire (ch. 9)."),
])


# ---------------------------------------------------------------------------
# Part C: derivatives and gradients in code (0B.47 to 0B.51)
# ---------------------------------------------------------------------------
PART_C = Part("C", "Dérivées et gradients en code",
              "Fiche §101.5 et §101.6. Une pente centrée de trois lignes suffit pour **vérifier** toutes tes "
              "dérivées calculées à la main, lire les variations d'une fonction, dessiner un gradient et tester la "
              "somme sur les chemins. (Le choix du pas $h$ et la descente de gradient complète viendront au ch. 5.)",
              exercises=[
    Ex("0B.47", "🔨", 2, 20, "Pentes numériques : vérifier tes dérivées à la main",
       "estimer une dérivée par une pente centrée et s'en servir pour vérifier des dérivées calculées à la main.",
       "Ex 0B.22, Ex 0B.23 · fiche §101.5.1 à §101.5.3, §101.2.7", tracks="M, C",
       body=r"""a) Écris `centered_slope(f, a, h=1e-5)`, la pente centrée (*central difference*) $\frac{f(a + h) - f(a - h)}{2h}$ (fiche §101.5.1) : une estimation de $f'(a)$, en trois lignes. Vérification a) : `centered_slope(lambda x: x ** 3, 2)` (4 décimales) ; quelle valeur exacte attendais-tu ?

b) Le dictionnaire `functions` (fourni) contient sept fonctions de 0B.22 et 0B.23, chacune avec un intervalle. Remplis le dictionnaire `derivatives` avec **les dérivées que tu as trouvées à la main**, sous forme de `lambda` (par exemple `lambda x: 3 * x ** 2 - 2` pour la fonction $x^3 - 2x$). La vérification compare ta formule à la pente centrée en neuf points de l'intervalle, et signale un point où elle se trompe. C'est le principe du *gradient checking* (ch. 18) : on vérifie une dérivée calculée à la main par une dérivée numérique.

c) La pente centrée de $f(x) = x^2 e^x$ en $x = 1$ (3 décimales) : la même que ta réponse papier à 0B.22 c ?""",
       given=r'''functions = {   # name: (function, interval where your derivative is compared with the slope)
    "0B.22c": (lambda x: x ** 2 * np.exp(x), (-2, 2)),
    "0B.22d": (lambda x: x * np.log(x), (0.5, 3)),
    "0B.22e": (lambda x: (x + 1) / (x - 1), (1.5, 4)),
    "0B.23b": (lambda x: np.exp(2 * x + 1), (-1, 1)),
    "0B.23c": (lambda x: np.log(x ** 2 + 1), (-2, 2)),
    "0B.23e": (lambda x: np.exp(-x ** 2 / 2), (-2, 2)),
    "0B.23h": (lambda x: np.log(x) ** 2, (0.5, 3)),
}''',
       todo=r'''def centered_slope(f, a, h=1e-5):
    """(f(a + h) - f(a - h)) / (2h): a numerical estimate of f'(a)."""
    raise NotImplementedError("centered_slope() is not written yet")


derivatives = {   # the derivatives you found BY HAND (0B.22, 0B.23), as lambdas
    "0B.22c": ...,
    "0B.22d": ...,
    "0B.22e": ...,
    "0B.23b": ...,
    "0B.23c": ...,
    "0B.23e": ...,
    "0B.23h": ...,
}''',
       check=r'''with wb.attempt("0B.47a"):
    wb.check("0B.47a", centered_slope(lambda x: x ** 3, 2))
    wb.check("0B.47c", centered_slope(functions["0B.22c"][0], 1))
with wb.attempt("0B.47b"):
    for name, (f, (low, high)) in functions.items():
        derivative = derivatives[name]
        if derivative is ...:
            print(f"⏳ Ex 0B.47b : {name} pas encore fait.")
            continue
        points = np.linspace(low, high, 9)
        slopes = np.array([centered_slope(f, x) for x in points])
        mine = np.array([derivative(x) for x in points], dtype=float)
        wrong = np.flatnonzero(~np.isclose(mine, slopes, rtol=1e-5, atol=1e-6))
        verdict("0B.47b", wrong.size == 0, f"{name} : ta dérivée coïncide avec la pente.",
                f"{name} : en x = {points[wrong[0]]:.3f} ta dérivée vaut {mine[wrong[0]]:.5f}, "
                f"la pente {slopes[wrong[0]]:.5f}." if wrong.size else "")''',
       solution=r'''def centered_slope(f, a, h=1e-5):
    """(f(a + h) - f(a - h)) / (2h): a numerical estimate of f'(a)."""
    return (f(a + h) - f(a - h)) / (2 * h)


derivatives = {
    "0B.22c": lambda x: (x ** 2 + 2 * x) * np.exp(x),          # product rule
    "0B.22d": lambda x: np.log(x) + 1,                          # product rule
    "0B.22e": lambda x: -2 / (x - 1) ** 2,                      # quotient rule
    "0B.23b": lambda x: 2 * np.exp(2 * x + 1),                  # chain rule, u = 2x + 1
    "0B.23c": lambda x: 2 * x / (x ** 2 + 1),                   # chain rule, u = x² + 1
    "0B.23e": lambda x: -x * np.exp(-x ** 2 / 2),               # chain rule, u = -x²/2
    "0B.23h": lambda x: 2 * np.log(x) / x,                      # chain rule, u = ln x
}
for name, (f, (low, high)) in functions.items():
    points = np.linspace(low, high, 9)
    gap = max(abs(derivatives[name](x) - centered_slope(f, x)) for x in points)
    print(f"{name}: largest gap {gap:.1e}")
print(centered_slope(lambda x: x ** 3, 2), centered_slope(functions["0B.22c"][0], 1), 3 * np.e)''',
       record=r'''wb.record("0B.47a", centered_slope(lambda x: x ** 3, 2), decimals=4)
wb.record("0B.47c", centered_slope(functions["0B.22c"][0], 1), decimals=3)''',
       note="Les écarts sont de l'ordre de $10^{-9}$ : la pente centrée est une très bonne approximation (le ch. 5 "
            "explique pourquoi, et comment choisir $h$). Une dérivée fausse, elle, se trompe dès le premier chiffre. "
            "C'est ainsi qu'on teste une rétropropagation écrite à la main (ch. 18)."),

    Ex("0B.48", "📈", 2, 15, "Lire les variations : f, f′ et les points où f′ s'annule",
       "relier le signe de la dérivée aux variations d'une fonction, sur un graphique puis par le calcul.",
       "Ex 0B.47 · fiche §101.5.4", thread="synthétique", tracks="M, C",
       body=r"""Pour $f(x) = x^3 - 3x^2 - 9x + 2$ sur $[-2 ; 4{,}5]$ (fournie : `f48`, et la grille `x48`), écris `plot_f_and_slope()`, qui trace deux panneaux l'un sous l'autre (`fig, axes = plt.subplots(2, 1, sharex=True)`) : $f$ en haut, sa pente centrée (0B.47) en bas, avec la droite $y = 0$ (`axhline(0)`) sur les deux ; la fonction renvoie la figure.

Puis **lis les graphiques** :
a) `zeros_of_slope` : les deux abscisses où $f'$ s'annule (des entiers, dans l'ordre croissant) ;
b) `nature_left` : en la plus petite, $f$ a-t-elle un `"maximum"` ou un `"minimum"` local ?
c) `f_min` : la plus petite valeur de $f$ sur l'intervalle, calculée sur la grille (`f48(x48).min()`, 1 décimale) ;
d) `f_local_max` : la valeur de $f$ en son maximum local.

Enfin, dans ta copie : dérive $f$ à la main, factorise $f'(x)$ et retrouve a).""",
       given=r'''def f48(x):
    return x ** 3 - 3 * x ** 2 - 9 * x + 2


x48 = np.linspace(-2, 4.5, 651)   # a grid with a step of 0.01''',
       todo=r'''def plot_f_and_slope():
    """Two panels: f48 on top, its centered slope below (with y = 0 on both); returns the figure."""
    raise NotImplementedError("plot_f_and_slope() is not written yet")


zeros_of_slope = ...   # a) [x1, x2]
nature_left = ...      # b) "maximum" or "minimum"
f_min = ...            # c)
f_local_max = ...      # d)''',
       check=r'''with wb.attempt("0B.48"):
    fig = plot_f_and_slope()
    plt.show()
    verdict("0B.48", len(fig.axes) == 2 and all(ax.get_lines() for ax in fig.axes),
            "deux panneaux, chacun avec ses courbes.", "il faut deux panneaux (f, puis sa pente), avec des courbes.")
wb.check("0B.48a", zeros_of_slope)
wb.check("0B.48b", nature_left)
wb.check("0B.48c", f_min)
wb.check("0B.48d", f_local_max)''',
       solution=r'''def plot_f_and_slope():
    """Two panels: f48 on top, its centered slope below (with y = 0 on both); returns the figure."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    axes[0].plot(x48, f48(x48), lw=2)
    axes[0].set_title("f(x) = x³ - 3x² - 9x + 2")
    axes[1].plot(x48, centered_slope(f48, x48), lw=2, color="C1")   # centered_slope works on arrays too
    axes[1].set_title("its centered slope ≈ f'(x)")
    for ax in axes:
        ax.axhline(0, color="0.5", lw=0.8)
        ax.grid(True)
    axes[1].set_xlabel("x")
    fig.tight_layout()
    return fig


fig = plot_f_and_slope()
plt.show()
zeros_of_slope = [-1, 3]       # read on the lower panel; f'(x) = 3x² - 6x - 9 = 3(x + 1)(x - 3)
nature_left = "maximum"        # f' goes from + to -
f_min = f48(x48).min()
f_local_max = f48(-1)
print(zeros_of_slope, nature_left, f_min, f_local_max, x48[np.argmin(f48(x48))])''',
       record=r'''wb.record("0B.48a", zeros_of_slope)
wb.record("0B.48b", nature_left, mistakes={"f′ passe de positive à négative : f monte, puis descend": "minimum"})
wb.record("0B.48c", f_min, decimals=1)
wb.record("0B.48d", f_local_max, decimals=1)''',
       note="$f'(x) = 3x^2 - 6x - 9 = 3(x + 1)(x - 3)$ : positive, puis négative entre −1 et 3, puis positive. "
            "Le minimum sur l'intervalle est atteint en $x = 3$ ($f(3) = -25$), qui est aussi le minimum local ; "
            "sur un intervalle plus grand, il faudrait aussi regarder les bornes. Pour une loss, on cherche de même "
            "les points où la dérivée s'annule, mais on ne sait pas les calculer : on les approche pas à pas (ch. 5)."),

    Ex("0B.49", "📈", 2, 20, "Carte de lignes de niveau et flèches du gradient",
       "tracer les lignes de niveau et le champ de gradient d'une fonction de deux variables, et les lire.",
       "Ex 0B.25, Ex 0B.48 · fiche §101.6.1, §101.6.3", thread="synthétique", tracks="M, C",
       body=r"""Pour $f(x, y) = (x - 1)^2 + 2y^2$ (0B.25, fournie : `f49`) :

1. Écris `grad_f49(x, y)`, qui renvoie le gradient **calculé à la main**, sous forme de tuple `(∂f/∂x, ∂f/∂y)` ; elle doit marcher avec des arrays NumPy (pas de `if`).
2. Écris `level_map()`, qui trace sur une même figure : les lignes de niveau 1, 2, 3, 4, 5 et 6 (`ax.contour(X, Y, Z, levels=[...])`, avec `X, Y = np.meshgrid(...)` sur $[-2 ; 4] \times [-2 ; 2]$) ; les flèches du gradient sur une grille plus grossière (`ax.quiver(Xc, Yc, Gx, Gy)`) ; le point $(3, 1)$ ; et `ax.set_aspect("equal")`, sans quoi les angles sont déformés. La fonction renvoie la figure.

Lis ensuite la carte :
a) le gradient en $(3, 1)$ : la vérification appelle ta fonction, `grad_f49(3, 1)` ;
b) `arrows_vanish_at` : le point où les flèches disparaissent (le gradient est nul) ;
c) `perpendicular` : les flèches coupent-elles les lignes de niveau à angle droit ? (`True`/`False`)
d) `steeper_axis` : en s'éloignant du minimum d'une même distance, $f$ monte-t-elle plus vite selon l'axe `"x"` ou selon l'axe `"y"` ? (regarde dans quelle direction les lignes de niveau sont les plus serrées)""",
       given=r'''def f49(x, y):
    return (x - 1) ** 2 + 2 * y ** 2''',
       todo=r'''def grad_f49(x, y):
    """The gradient of f49 at (x, y), computed by hand: (df/dx, df/dy)."""
    raise NotImplementedError("grad_f49() is not written yet")


def level_map():
    """Level lines 1 to 6 of f49, gradient arrows and the point (3, 1); returns the figure."""
    raise NotImplementedError("level_map() is not written yet")


arrows_vanish_at = ...   # b) [x, y]
perpendicular = ...      # c) True or False
steeper_axis = ...       # d) "x" or "y"''',
       check=r'''import matplotlib


with wb.attempt("0B.49a"):
    wb.check("0B.49a", list(grad_f49(3.0, 1.0)))
    Gx, Gy = grad_f49(np.array([0.0, 1.0]), np.array([1.0, 0.0]))    # also on arrays
    verdict("0B.49", np.allclose(Gx, [-2, 0]) and np.allclose(Gy, [4, 0]), "grad_f49 marche aussi sur des arrays.",
            "grad_f49 doit accepter des arrays (pas de if, des opérations NumPy).")
with wb.attempt("0B.49"):
    fig = level_map()
    plt.show()
    ax = fig.axes[0]
    kinds = {type(item).__name__ for item in ax.collections}
    has_quiver = any(isinstance(item, matplotlib.quiver.Quiver) for item in ax.collections)
    has_contour = any(isinstance(item, matplotlib.contour.ContourSet) for item in ax.collections)
    verdict("0B.49", has_quiver and has_contour, "la carte contient des lignes de niveau et des flèches.",
            f"il manque des lignes de niveau (contour) ou des flèches (quiver) : {sorted(kinds)}")
wb.check("0B.49b", arrows_vanish_at)
wb.check("0B.49c", perpendicular)
wb.check("0B.49d", steeper_axis)''',
       solution=r'''import matplotlib


def grad_f49(x, y):
    """The gradient of f49 at (x, y), computed by hand: (df/dx, df/dy)."""
    return 2 * (x - 1), 4 * y


def level_map():
    """Level lines 1 to 6 of f49, gradient arrows and the point (3, 1); returns the figure."""
    X, Y = np.meshgrid(np.linspace(-2, 4, 200), np.linspace(-2, 2, 200))
    Xc, Yc = np.meshgrid(np.linspace(-1.5, 3.5, 11), np.linspace(-1.5, 1.5, 7))
    Gx, Gy = grad_f49(Xc, Yc)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    contours = ax.contour(X, Y, f49(X, Y), levels=[1, 2, 3, 4, 5, 6])
    ax.clabel(contours, fontsize=8)
    ax.quiver(Xc, Yc, Gx, Gy, color="0.4", angles="xy")
    ax.scatter([3], [1], color="C1", zorder=3, label="(3, 1)")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("level lines of f and its gradient")
    return fig


fig = level_map()
plt.show()
grad_at_3_1 = list(grad_f49(3.0, 1.0))
arrows_vanish_at = [1, 0]
perpendicular = True
steeper_axis = "y"
print(grad_at_3_1, arrows_vanish_at, perpendicular, steeper_axis)''',
       record=r'''wb.record("0B.49a", grad_at_3_1, mistakes={"∂(2y²)/∂y = 4y": [4, 2]})
wb.record("0B.49b", arrows_vanish_at)
wb.record("0B.49c", perpendicular)
wb.record("0B.49d", steeper_axis, mistakes={"les ellipses sont plus étroites dans quelle direction ? Regarde le coefficient devant y²": "x"})''',
       note="Le gradient est perpendiculaire aux lignes de niveau et pointe vers les niveaux croissants ; il "
            "s'annule au minimum $(1, 0)$. Les ellipses sont $\\sqrt{2}$ fois plus « serrées » selon $y$ (coefficient 2 "
            "devant $y^2$) : $f$ monte plus vite dans cette direction. Une loss en forme d'ellipses très allongées "
            "rend la descente de gradient lente (elle zigzague) : c'est une raison de standardiser les features "
            "(ch. 12) et d'utiliser les optimiseurs du ch. 19."),

    Ex("0B.50", "🔮", 2, 15, "Contre le gradient, avec lui ou le long d'une ligne de niveau : où va f ?",
       "prévoir l'effet d'un petit pas selon la direction choisie par rapport au gradient, puis le mesurer.",
       "Ex 0B.49 · fiche §101.6.3", thread="synthétique", tracks="M", hypothesis=True,
       body=r"""Toujours $f(x, y) = (x - 1)^2 + 2y^2$. Depuis $\mathbf{p} = (3, 1)$, où $\nabla f = (4, 4)$, on fait un petit pas de longueur 0,01 dans quatre directions : a) **contre** le gradient ; b) **dans le sens** du gradient ; c) **le long de la ligne de niveau** qui passe par $\mathbf{p}$ (perpendiculairement au gradient) ; d) selon l'axe des $x$, vers la droite.

Pour chacune, prédis si $f$ va `"baisser"`, `"monter"` ou rester `"stable"` (variation négligeable devant celle des autres directions). Écris ton hypothèse, tes prédictions, puis lance l'**Expérience**, qui calcule les variations $\Delta f$.

e) et f) Après l'expérience : parmi les 360 directions $(\cos\theta, \sin\theta)$, pour $\theta = 0°, 1°, \dots, 359°$, cherche celle où $f$ baisse le plus après un pas de longueur donnée. `best_angle_small` : l'angle (en degrés, un entier) pour un pas de 0,01 ; `best_angle_big` : le même pour un grand pas, de 0,5. Dans ta copie : compare ces deux angles avec celui de la direction $-\nabla f$ et avec celui de la direction qui va de $\mathbf{p}$ au minimum $(1, 0)$ (l'angle d'un vecteur $(d_x, d_y)$ : `np.degrees(np.arctan2(dy, dx)) % 360`). Que garantit le gradient, et pour quels pas ?""",
       given=r'''def f50(x, y):
    return (x - 1) ** 2 + 2 * y ** 2


p50 = np.array([3.0, 1.0])
unit_grad = np.array([4.0, 4.0]) / np.linalg.norm([4.0, 4.0])    # the gradient at p50, scaled to length 1''',
       todo=r'''prediction_0B_50a = ...   # a) against the gradient: "baisser", "monter" or "stable"
prediction_0B_50b = ...   # b) along the gradient
prediction_0B_50c = ...   # c) along the level line
prediction_0B_50d = ...   # d) along the x axis, to the right''',
       check=r'''predictions_0B_50 = [prediction_0B_50a, prediction_0B_50b, prediction_0B_50c, prediction_0B_50d]
for letter, prediction in zip("abcd", predictions_0B_50):
    wb.check(f"0B.50{letter}", prediction)''',
       solution=r'''prediction_0B_50a = "baisser"
prediction_0B_50b = "monter"
prediction_0B_50c = "stable"
prediction_0B_50d = "monter"   # df/dx = 4 > 0 at p50''',
       record=r'''wb.record("0B.50a", prediction_0B_50a, mistakes={"contre le gradient, on va dans la direction de plus forte DESCENTE": "monter"})
wb.record("0B.50b", prediction_0B_50b, mistakes={"le gradient indique la direction où f MONTE le plus vite": "baisser"})
wb.record("0B.50c", prediction_0B_50c, mistakes={"le long d'une ligne de niveau, f garde (presque) la même valeur": "baisser"})
wb.record("0B.50d", prediction_0B_50d, mistakes={"regarde le signe de ∂f/∂x en (3, 1)": "baisser", "∂f/∂x n'est pas nul en (3, 1) : regarde son signe": "stable"})''',
       after=[("md", "**Expérience** : la variation de $f$ après un pas de 0,01 dans chaque direction."),
              ("code", r'''directions_50 = {
    "a) against the gradient": -unit_grad,
    "b) along the gradient": unit_grad,
    "c) along the level line": np.array([-unit_grad[1], unit_grad[0]]),
    "d) along the x axis": np.array([1.0, 0.0]),
}
for name, direction in directions_50.items():
    new_point = p50 + 0.01 * direction
    print(f"{name:26} delta f = {f50(*new_point) - f50(*p50):+.6f}")'''),
              ("md", "e) et f) Les 360 directions, un degré à la fois."),
              ("todo", r'''best_angle_small = ...   # e) in degrees, an integer between 0 and 359 (a step of 0.01)
best_angle_big = ...     # f) the same for a step of 0.5'''),
              ("check", r'''wb.check("0B.50e", best_angle_small)
wb.check("0B.50f", best_angle_big)'''),
              ("solution", r'''def best_angle(step):
    """The direction (in degrees) where f50 decreases the most after a step of this length from p50."""
    angles = np.arange(360)
    radians = np.radians(angles)
    changes = f50(p50[0] + step * np.cos(radians), p50[1] + step * np.sin(radians)) - f50(*p50)
    return int(angles[np.argmin(changes)])


best_angle_small = best_angle(0.01)
best_angle_big = best_angle(0.5)
print(best_angle_small, best_angle_big, np.degrees(np.arctan2(0 - 1, 1 - 3)) % 360)'''),
              ("record", r'''wb.record("0B.50e", best_angle_small, mistakes={"cherche la plus forte BAISSE (argmin), pas la plus forte hausse": 45})
wb.record("0B.50f", best_angle_big, mistakes={"refais la recherche avec un pas de 0,5 : la meilleure direction n'est plus exactement celle du gradient": 225})''')],
       note="Le pas le long de la ligne de niveau change $f$ d'environ $10^{-4}$, près de 400 fois moins que les "
            "autres : à cette échelle, $f$ ne varie qu'au second ordre. Pour un petit pas, la meilleure direction est "
            "225°, exactement $-\\nabla f$, et la baisse vaut environ $0{,}01 \\times \\|\\nabla f\\| \\approx 0{,}057$ : "
            "la norme du gradient mesure la pente la plus forte (fiche §101.6.3). Pour un pas de 0,5, la meilleure "
            "direction tourne vers le minimum (220°) : le gradient ne donne que la meilleure direction **locale**. "
            "D'où les petits pas (le learning rate) et les pas répétés de la descente de gradient (ch. 5)."),

    Ex("0B.51", "🔨", 2, 25, "Dérivées partielles numériques et somme sur les chemins",
       "estimer des dérivées partielles par des pentes centrées et vérifier des dérivées par la somme sur les chemins.",
       "Ex 0B.29, Ex 0B.47 · fiche §101.6.2, §101.6.4", tracks="M, C",
       body=r"""a) Écris `partial_x(f, x, y, h=1e-5)` et `partial_y(f, x, y, h=1e-5)` : des pentes centrées selon une seule variable, l'autre restant fixe. Puis `numerical_gradient(f, x, y)`, qui renvoie `[∂f/∂x, ∂f/∂y]`. Vérification a) : le gradient numérique de $g(x, y) = x^2 y + y^3$ en $(1, 2)$ (fournie : `g51` ; 2 décimales ; compare avec 0B.25 i et j).

b) Partie A de 0B.29 : $z = u^2 + uv$ avec $u = 2x$ et $v = x^2$ (fournie : `z_of_x`). Écris `dz_dx_paths(x)`, la dérivée par la **somme sur les chemins**, avec les dérivées locales de ton graphe de 0B.29 (calcule d'abord $u$ et $v$, puis ajoute les deux chemins). La vérification la compare à la pente centrée de `z_of_x` en plusieurs points, puis regarde b) `dz_dx_paths(1)` (2 décimales).

c) à e) Partie B de 0B.29 : la loss `loss51(w, b)` $= (w \cdot 1 + b - 3)^2 + (w \cdot 2 + b - 4)^2$ est fournie. Écris `grad_paths(w, b)`, qui renvoie `[∂L/∂w, ∂L/∂b]` sous forme de sommes sur les chemins (à travers $\hat{y}_1$ et $\hat{y}_2$), puis `one_step(w, b, eta)`, qui renvoie le point `[w, b]` après un pas de descente de gradient. Vérifications : c) `numerical_gradient(loss51, 1, 0)` (2 décimales) · d) `grad_paths` coïncide-t-elle avec le gradient numérique, en 20 points au hasard ? · e) la loss après un pas depuis $(1, 0)$ avec $\eta = 0{,}1$ (2 décimales).""",
       given=r'''def g51(x, y):
    return x ** 2 * y + y ** 3


def z_of_x(x):
    u, v = 2 * x, x ** 2
    return u ** 2 + u * v


def loss51(w, b):
    """Squared errors of y_hat = w x + b on the two examples (1, 3) and (2, 4)."""
    return (w * 1 + b - 3) ** 2 + (w * 2 + b - 4) ** 2''',
       todo=r'''def partial_x(f, x, y, h=1e-5):
    raise NotImplementedError("partial_x() is not written yet")


def partial_y(f, x, y, h=1e-5):
    raise NotImplementedError("partial_y() is not written yet")


def numerical_gradient(f, x, y):
    """[df/dx, df/dy] estimated with centered slopes."""
    raise NotImplementedError("numerical_gradient() is not written yet")


def dz_dx_paths(x):
    """dz/dx for z = u² + u v, u = 2x, v = x², as a sum over the two paths."""
    raise NotImplementedError("dz_dx_paths() is not written yet")


def grad_paths(w, b):
    """[dL/dw, dL/db] for loss51, as sums over the paths through y_hat_1 and y_hat_2."""
    raise NotImplementedError("grad_paths() is not written yet")


def one_step(w, b, eta):
    """The point [w, b] after one gradient descent step of learning rate eta."""
    raise NotImplementedError("one_step() is not written yet")''',
       check=r'''rng_51 = np.random.default_rng(51)
with wb.attempt("0B.51a"):
    wb.check("0B.51a", numerical_gradient(g51, 1, 2))
with wb.attempt("0B.51b"):
    points = np.linspace(-2, 2, 9)
    ok = all(np.isclose(dz_dx_paths(x), (z_of_x(x + 1e-5) - z_of_x(x - 1e-5)) / 2e-5, rtol=1e-6, atol=1e-6)
             for x in points)
    verdict("0B.51b", ok, "la somme sur les chemins coïncide avec la pente de z.",
            "dz_dx_paths ne coïncide pas avec la pente de z_of_x : revois les dérivées locales des arêtes.")
    wb.check("0B.51b", dz_dx_paths(1))
with wb.attempt("0B.51c"):
    wb.check("0B.51c", numerical_gradient(loss51, 1, 0))
with wb.attempt("0B.51d"):
    trials = rng_51.uniform(-3, 3, size=(20, 2))
    wb.check("0B.51d", all(np.allclose(grad_paths(w, b), numerical_gradient(loss51, w, b), rtol=1e-5, atol=1e-5)
                           for w, b in trials))
with wb.attempt("0B.51e"):
    wb.check("0B.51e", loss51(*one_step(1, 0, 0.1)))''',
       solution=r'''def partial_x(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


def partial_y(f, x, y, h=1e-5):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


def numerical_gradient(f, x, y):
    """[df/dx, df/dy] estimated with centered slopes."""
    return [partial_x(f, x, y), partial_y(f, x, y)]


def dz_dx_paths(x):
    """dz/dx for z = u² + u v, u = 2x, v = x², as a sum over the two paths."""
    u, v = 2 * x, x ** 2
    return (2 * u + v) * 2 + u * (2 * x)     # path through u + path through v


def grad_paths(w, b):
    """[dL/dw, dL/db] for loss51, as sums over the paths through y_hat_1 and y_hat_2."""
    (x1, y1), (x2, y2) = (1, 3), (2, 4)
    e1, e2 = w * x1 + b - y1, w * x2 + b - y2          # dL/dy_hat_i = 2 e_i
    return [2 * e1 * x1 + 2 * e2 * x2, 2 * e1 * 1 + 2 * e2 * 1]


def one_step(w, b, eta):
    """The point [w, b] after one gradient descent step of learning rate eta."""
    dw, db = grad_paths(w, b)
    return [w - eta * dw, b - eta * db]


rng_51 = np.random.default_rng(51)
trials = rng_51.uniform(-3, 3, size=(20, 2))
paths_ok = all(np.allclose(grad_paths(w, b), numerical_gradient(loss51, w, b), rtol=1e-5, atol=1e-5) for w, b in trials)
print(numerical_gradient(g51, 1, 2), dz_dx_paths(1), numerical_gradient(loss51, 1, 0), paths_ok,
      one_step(1, 0, 0.1), loss51(1, 0), loss51(*one_step(1, 0, 0.1)))''',
       record=r'''wb.record("0B.51a", numerical_gradient(g51, 1, 2), decimals=2)
wb.record("0B.51b", dz_dx_paths(1), decimals=2, mistakes={"deux chemins mènent de x à z : additionne-les": (2 * 2 + 1) * 2})
wb.record("0B.51c", numerical_gradient(loss51, 1, 0), decimals=2)
wb.record("0B.51d", paths_ok)
wb.record("0B.51e", loss51(*one_step(1, 0, 0.1)), decimals=2, mistakes={"on descend : on SOUSTRAIT eta fois le gradient": loss51(1 + 0.1 * -12, 0 + 0.1 * -8)})''',
       note="La somme sur les chemins donne exactement la dérivée ; la pente centrée ne fait que l'estimer, mais "
            "suffit à détecter une erreur. Un réseau de neurones a des millions de chemins : la rétropropagation "
            "(ch. 18) organise ce calcul en partant de la loss, et la pente numérique sert à la vérifier sur de "
            "petits exemples."),
])


# ---------------------------------------------------------------------------
# Part D: simulated probabilities and final challenge (0B.52 to 0B.54)
# ---------------------------------------------------------------------------
PART_D = Part("D", "Probabilités simulées et défi final",
              "Fiche §101.7 et §101.4.3. Simuler, c'est vérifier un calcul de probabilité, d'espérance ou de "
              "variance par l'expérience. Quand un exercice tire des nombres au hasard, l'énoncé impose **exactement** "
              "les appels au générateur, pour que tes nombres soient ceux du corrigé.", exercises=[
    Ex("0B.52", "🔬", 2, 25, "Simuler des dés : fréquences, indépendance, loi des grands nombres",
       "estimer des probabilités par simulation, tester une indépendance et voir la loi des grands nombres à l'œuvre.",
       "Ex 0B.26 · 0A (`default_rng`, masques) · fiche §101.7.1, §101.7.2, §101.7.5", thread="synthétique", tracks="R, M, C",
       body=r"""On simule $n = 100\,000$ lancers de deux dés, **exactement** ainsi : `rng = np.random.default_rng(0)`, puis `rolls = rng.integers(1, 7, size=(n, 2))`, `d1 = rolls[:, 0]` et `d2 = rolls[:, 1]`. La fréquence d'un événement est la moyenne d'un masque booléen (par exemple `(d1 == 1).mean()`).

a) `freq_sum_7` : la fréquence de « la somme vaut 7 » (3 décimales) ; compare avec $\frac{1}{6}$.
b) `gaps` : avec $A$ = « le premier dé est pair », $B$ = « la somme vaut 7 » et $C$ = « la somme vaut 8 », la liste des deux écarts $f(A \cap B) - f(A)\,f(B)$ et $f(A \cap C) - f(A)\,f(C)$ ($f$ désigne la fréquence ; 3 décimales). Compare avec 0B.26 : lequel est presque nul ?
c) `running_6` : la fréquence des 6 sur le premier dé après $k$ lancers, pour $k = 1, 2, \dots, n$ (un array de $n$ valeurs : `np.cumsum` et `np.arange`). La vérification regarde ses valeurs après 100, 1 000 et 100 000 lancers (3 décimales) et trace la courbe.
d) On peut aussi mesurer **à quelle vitesse** la fréquence se rapproche de $\frac{1}{6}$. Avec un second générateur, `rng2 = np.random.default_rng(1)`, et pour chaque nombre de lancers `n_rolls` de la liste `[100, 400, 1600, 6400]`, **dans cet ordre** : `freqs = (rng2.integers(1, 7, size=(200, n_rolls)) == 6).mean(axis=1)` (200 expériences de $n$ lancers), puis l'écart-type de ces 200 fréquences (`freqs.std()`). `ratios` : la liste des 3 rapports entre deux écarts-types successifs, arrondis à l'unité. Que se passe-t-il quand on multiplie le nombre de lancers par 4 ?""",
       todo=r'''freq_sum_7 = ...   # a)
gaps = ...         # b) [f(A∩B) - f(A) f(B), f(A∩C) - f(A) f(C)]
running_6 = ...    # c) an array of n running frequencies
ratios = ...       # d) [std(100) / std(400), std(400) / std(1600), std(1600) / std(6400)], rounded''',
       check=r'''wb.check("0B.52a", freq_sum_7)
wb.check("0B.52b", gaps)
if running_6 is ...:
    print("⏳ Ex 0B.52c : pas encore fait.")
else:
    wb.check("0B.52c", [running_6[99], running_6[999], running_6[-1]])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(np.arange(1, len(running_6) + 1), running_6)
    ax.axhline(1 / 6, color="0.3", ls="--", label="1/6")
    ax.set_xscale("log")
    ax.set_xlabel("number of rolls (log scale)")
    ax.set_ylabel("frequency of 6")
    ax.legend()
    plt.show()
wb.check("0B.52d", ratios)''',
       solution=r'''n = 100_000
rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=(n, 2))
d1, d2 = rolls[:, 0], rolls[:, 1]
A, B, C = d1 % 2 == 0, d1 + d2 == 7, d1 + d2 == 8
freq_sum_7 = B.mean()
gaps = [(A & B).mean() - A.mean() * B.mean(), (A & C).mean() - A.mean() * C.mean()]
running_6 = np.cumsum(d1 == 6) / np.arange(1, n + 1)

rng2 = np.random.default_rng(1)
stds = []
for n_rolls in [100, 400, 1600, 6400]:
    freqs = (rng2.integers(1, 7, size=(200, n_rolls)) == 6).mean(axis=1)
    stds.append(freqs.std())
ratios = [round(stds[i] / stds[i + 1]) for i in range(3)]
print(freq_sum_7, np.round(gaps, 4), running_6[[99, 999, -1]], np.round(stds, 4), ratios)

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(np.arange(1, n + 1), running_6)
ax.axhline(1 / 6, color="0.3", ls="--", label="1/6")
ax.set_xscale("log")
ax.set_xlabel("number of rolls (log scale)")
ax.set_ylabel("frequency of 6")
ax.legend()
plt.show()''',
       record=r'''wb.record("0B.52a", freq_sum_7, decimals=3, mistakes={"c'est la probabilité théorique : on demande la fréquence observée dans ta simulation": 1 / 6})
wb.record("0B.52b", gaps, decimals=3)
wb.record("0B.52c", [running_6[99], running_6[999], running_6[-1]], decimals=3)
wb.record("0B.52d", ratios, mistakes={"un rapport entre deux écarts-types successifs, pas entre deux variances": [round(stds[i] ** 2 / stds[i + 1] ** 2) for i in range(3)]})''',
       note="L'écart $f(A \\cap B) - f(A)f(B)$ vaut environ 0,001 : du bruit de simulation, $A$ et $B$ sont "
            "indépendants. L'autre vaut environ 0,013, proche de $\\frac{1}{72} \\approx 0{,}014$ (0B.26) : "
            "une vraie dépendance, petite mais visible avec 100 000 lancers. Multiplier le nombre de lancers par 4 "
            "divise l'écart-type des fréquences par 2 : l'erreur décroît comme $\\frac{1}{\\sqrt{n}}$ (fiche "
            "§101.7.5). Pour gagner une décimale de précision, il faut 100 fois plus de lancers."),

    Ex("0B.53", "🔨", 2, 20, "Espérance et variance : le calcul exact contre la simulation",
       "calculer espérance et variance d'une loi discrète, puis les retrouver par simulation.",
       "Ex 0B.27, Ex 0B.52 · fiche §101.7.3, §101.7.4, §101.7.5", thread="synthétique", tracks="R, M, C",
       body=r"""a) et b) Écris `expectation(values, probs)` et `variance(values, probs)` (avec la formule $\mathbb{E}[X^2] - \mathbb{E}[X]^2$) pour une variable discrète donnée par deux arrays : ses valeurs et leurs probabilités. Vérifications, sur la loi de 0B.27 (`values_53`, `probs_53`) : a) $\mathbb{E}[X]$ · b) $\mathrm{Var}(X)$ · c) la variance de $3X + 2$, avec ta fonction `variance` appliquée aux valeurs transformées `3 * values_53 + 2` (2 décimales) : compare avec 0B.27 f.

Simule ensuite $n = 100\,000$ tirages, exactement ainsi : `n = 100_000`, `rng = np.random.default_rng(0)`, `x = rng.choice(values_53, size=n, p=probs_53)`, puis `y = rng.choice(values_53, size=n, p=probs_53)` (une deuxième variable de même loi, **indépendante** de la première).

d) `sample_stats` : la moyenne et la variance (`.var()`) des tirages `x` (2 décimales) ; proches de a) et b) ?
e) `var_sum_indep` : la variance de `x + y` (2 décimales) ; compare avec $\mathrm{Var}(X) + \mathrm{Var}(Y)$.
f) `var_sum_same` : la variance de `x + x` (2 décimales). Pourquoi n'est-ce plus la somme des deux variances ?""",
       given=r'''values_53 = np.array([0, 1, 2, 5])          # the law of 0B.27
probs_53 = np.array([0.4, 0.3, 0.2, 0.1])''',
       todo=r'''def expectation(values, probs):
    raise NotImplementedError("expectation() is not written yet")


def variance(values, probs):
    raise NotImplementedError("variance() is not written yet")


sample_stats = ...    # d) [mean of x, variance of x]
var_sum_indep = ...   # e)
var_sum_same = ...    # f)''',
       check=r'''with wb.attempt("0B.53a"):
    wb.check("0B.53a", expectation(values_53, probs_53))
with wb.attempt("0B.53b"):
    wb.check("0B.53b", variance(values_53, probs_53))
    wb.check("0B.53c", variance(3 * values_53 + 2, probs_53))
wb.check("0B.53d", sample_stats)
wb.check("0B.53e", var_sum_indep)
wb.check("0B.53f", var_sum_same)''',
       solution=r'''def expectation(values, probs):
    return np.sum(values * probs)


def variance(values, probs):
    return expectation(values ** 2, probs) - expectation(values, probs) ** 2


n = 100_000
rng = np.random.default_rng(0)
x = rng.choice(values_53, size=n, p=probs_53)
y = rng.choice(values_53, size=n, p=probs_53)
sample_stats = [x.mean(), x.var()]
var_sum_indep = (x + y).var()
var_sum_same = (x + x).var()
print(expectation(values_53, probs_53), variance(values_53, probs_53), variance(3 * values_53 + 2, probs_53))
print(np.round(sample_stats, 3), round(var_sum_indep, 3), round(var_sum_same, 3))''',
       record=r'''wb.record("0B.53a", expectation(values_53, probs_53), decimals=2)
wb.record("0B.53b", variance(values_53, probs_53), decimals=2, mistakes={"Var = E[X²] − E[X]² : n'oublie pas d'élever E[X] au carré": 3.6 - 1.2})
wb.record("0B.53c", variance(3 * values_53 + 2, probs_53), decimals=2)
wb.record("0B.53d", sample_stats, decimals=2, mistakes={"ce sont les valeurs théoriques : on demande celles de la simulation": [1.2, 2.16]})
wb.record("0B.53e", var_sum_indep, decimals=2, mistakes={"c'est la valeur théorique : on demande la variance des tirages simulés": 4.32})
wb.record("0B.53f", var_sum_same, decimals=2, mistakes={"c'est la valeur théorique : on demande la variance des tirages simulés": 8.64, "x + x = 2x : ses deux termes ne sont pas indépendants, les variances ne s'additionnent plus": 4.32, "x + x = 2x : ses deux termes varient ensemble, les variances ne s'additionnent plus": 2 * x.var()})''',
       note="La simulation donne 1,20 et 2,14 au lieu de 1,2 et 2,16 : l'écart diminue comme "
            "$\\frac{1}{\\sqrt{n}}$ quand le nombre de tirages grandit. Pour des variables **indépendantes**, les variances s'additionnent "
            "($\\approx 4{,}32$ ; démontré au ch. 16). Pour $X + X = 2X$, la règle $\\mathrm{Var}(aX) = "
            "a^2\\,\\mathrm{Var}(X)$ donne $4 \\times 2{,}16 = 8{,}64$ : deux fois plus que la somme des variances, "
            "car les deux termes varient ensemble. L'espérance, elle, s'additionne toujours."),

    Ex("0B.54", "🏆", 3, 40, "L'ordre des produits : calculer A·B·C·v des dizaines de fois plus vite",
       "choisir l'ordre des produits matriciels qui minimise le nombre de multiplications, et mesurer le gain.",
       "Ex 0B.30, Ex 0B.46 · fiche §101.4.3, §101.1.1", thread="synthétique", tracks="M, C",
       body=r"""🏆 **Objectif** : calculer $\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{v}$ ($\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$ de taille $1000 \times 1000$, $\mathbf{v}$ de dimension 1000) **au moins 20 fois plus vite** que de gauche à droite, avec le même résultat. (En `FAST_MODE = False`, les matrices font $2000 \times 2000$.)

a) Écris `left_to_right(A, B, C, v)`, qui calcule $((\mathbf{A}\mathbf{B})\mathbf{C})\mathbf{v}$, et `right_to_left(A, B, C, v)`, qui calcule $\mathbf{A}(\mathbf{B}(\mathbf{C}\mathbf{v}))$, avec `@`. Vérification a) : les deux résultats sont-ils égaux (`np.allclose`) ?

b) Écris `matmul_cost(shape_a, shape_b)` : le nombre de multiplications d'un produit $(m, n) \times (n, p)$, soit $m\,n\,p$ ; un vecteur de forme `(n,)` compte comme une matrice $(n, 1)$. Puis `chain_costs(n)`, qui renvoie `[coût de gauche à droite, coût de droite à gauche]` pour des matrices $n \times n$ et un vecteur de dimension $n$ : additionne les coûts des trois produits de chaque ordre, en suivant les formes des résultats intermédiaires. Vérification b) : `chain_costs(1000)` (compare avec 0B.30).

c) La cellule de vérification mesure les deux temps (le meilleur de 3 essais) et calcule `speedup` = temps de gauche à droite / temps de droite à gauche. 🏆 Objectif atteint si `speedup >= 20`.

d) Dans ta copie : le gain en temps est-il aussi grand que le gain en nombre de multiplications ? Pourquoi ? (Indice : un produit matrice-vecteur lit toute la matrice en mémoire pour très peu de calcul, alors qu'un produit matriciel réutilise chaque nombre lu des centaines de fois.)

e) Bonus : un réseau de trois couches **linéaires** (sans fonction d'activation) calcule $\mathbf{X}\mathbf{W}_1\mathbf{W}_2\mathbf{W}_3$, avec $\mathbf{X}$ de forme $(64, 784)$ (un mini-lot d'images MNIST), $\mathbf{W}_1$ $(784, 512)$, $\mathbf{W}_2$ $(512, 512)$ et $\mathbf{W}_3$ $(512, 10)$. `network_costs` : la liste `[coût de gauche à droite, coût de X(W1(W2 W3))]`, avec `matmul_cost`. Dans ta copie : pourquoi ce regroupement n'est-il plus possible dès qu'une fonction d'activation s'intercale entre les couches (ch. 16 et 17) ?""",
       given=r'''rng_54 = np.random.default_rng(0)
N54 = 1000 if FAST_MODE else 2000
A54, B54, C54 = (rng_54.normal(size=(N54, N54)) for _ in range(3))
v54 = rng_54.normal(size=N54)''',
       todo=r'''def left_to_right(A, B, C, v):
    raise NotImplementedError("left_to_right() is not written yet")


def right_to_left(A, B, C, v):
    raise NotImplementedError("right_to_left() is not written yet")


def matmul_cost(shape_a, shape_b):
    """Number of multiplications of a (m, n) x (n, p) product; a vector (n,) counts as (n, 1)."""
    raise NotImplementedError("matmul_cost() is not written yet")


def chain_costs(n):
    """[cost of ((AB)C)v, cost of A(B(Cv))] for n x n matrices and a vector of dimension n."""
    raise NotImplementedError("chain_costs() is not written yet")


network_costs = ...   # e) [cost from left to right, cost of X(W1(W2 W3))]''',
       check=r'''with wb.attempt("0B.54a"):
    wb.check("0B.54a", bool(np.allclose(left_to_right(A54, B54, C54, v54), right_to_left(A54, B54, C54, v54))))
with wb.attempt("0B.54b"):
    wb.check("0B.54b", chain_costs(1000))
with wb.attempt("0B.54c"):
    t_left = measure(left_to_right, A54, B54, C54, v54)
    t_right = measure(right_to_left, A54, B54, C54, v54)
    speedup = t_left / t_right
    print(f"left to right: {t_left * 1000:.1f} ms, right to left: {t_right * 1000:.2f} ms, speedup: {speedup:.0f}")
    verdict("0B.54c", speedup >= 20, f"🏆 objectif atteint : {speedup:.0f} fois plus rapide.",
            f"{speedup:.0f} fois plus rapide seulement : vérifie que right_to_left ne multiplie jamais deux matrices "
            "entre elles, puis relance (les mesures fluctuent).")
wb.check("0B.54e", network_costs)''',
       solution=r'''def left_to_right(A, B, C, v):
    return ((A @ B) @ C) @ v


def right_to_left(A, B, C, v):
    return A @ (B @ (C @ v))


def matmul_cost(shape_a, shape_b):
    """Number of multiplications of a (m, n) x (n, p) product; a vector (n,) counts as (n, 1)."""
    m, n = shape_a if len(shape_a) == 2 else (shape_a[0], 1)
    n_b, p = shape_b if len(shape_b) == 2 else (shape_b[0], 1)
    if n != n_b:
        raise ValueError(f"cannot multiply {shape_a} by {shape_b}")
    return m * n * p


def chain_costs(n):
    """[cost of ((AB)C)v, cost of A(B(Cv))] for n x n matrices and a vector of dimension n."""
    square, vector = (n, n), (n,)
    left = matmul_cost(square, square) + matmul_cost(square, square) + matmul_cost(square, vector)
    right = 3 * matmul_cost(square, vector)
    return [left, right]


X_shape, W1_shape, W2_shape, W3_shape = (64, 784), (784, 512), (512, 512), (512, 10)
network_costs = [matmul_cost(X_shape, W1_shape) + matmul_cost((64, 512), W2_shape) + matmul_cost((64, 512), W3_shape),
                 matmul_cost(W2_shape, W3_shape) + matmul_cost(W1_shape, (512, 10)) + matmul_cost(X_shape, (784, 10))]

same = np.allclose(left_to_right(A54, B54, C54, v54), right_to_left(A54, B54, C54, v54))
t_left = measure(left_to_right, A54, B54, C54, v54)
t_right = measure(right_to_left, A54, B54, C54, v54)
speedup = t_left / t_right
if speedup < 20:      # a timing measured on a busy machine: never publish it, run the notebook again
    raise RuntimeError(f"speedup {speedup:.0f} only: the machine is busy, run this notebook again later")
costs = chain_costs(1000)
print(same, costs, costs[0] / costs[1], network_costs, network_costs[0] / network_costs[1])
print(f"left to right: {t_left * 1000:.1f} ms, right to left: {t_right * 1000:.2f} ms, speedup: {speedup:.0f}")''',
       record=r'''wb.record("0B.54a", bool(same))
wb.record("0B.54b", costs, mistakes={"de droite à gauche, chaque résultat intermédiaire est un vecteur : quel est le coût d'un produit matrice-vecteur ?": [2_001_000_000, 3_000_000_000]})
wb.record("0B.54e", network_costs)''',
       note="Le calcul de droite à gauche fait environ 670 fois moins de multiplications, mais n'est « que » 30 à "
            "100 fois plus rapide selon la machine : un produit matrice-vecteur est limité par la vitesse de la "
            "mémoire (chaque nombre lu ne sert qu'une fois), alors qu'un produit matriciel réutilise chaque nombre "
            "et exploite à fond le processeur. En e, regrouper les poids divise le coût par 6 : un réseau linéaire "
            "équivaut à **une seule** matrice $\\mathbf{W}_1\\mathbf{W}_2\\mathbf{W}_3$, ce qui montre qu'empiler "
            "des couches linéaires n'apporte rien. Dès qu'une activation non linéaire s'intercale, "
            "$f(\\mathbf{X}\\mathbf{W}_1)\\mathbf{W}_2 \\neq \\mathbf{X}(\\mathbf{W}_1\\mathbf{W}_2)$ : on ne peut plus "
            "regrouper, et c'est justement ce qui donne sa puissance au réseau (ch. 16 et 17). La rétropropagation "
            "applique l'idée de a à c : partir d'un nombre (la loss) et n'enchaîner que des produits "
            "vecteur-matrice (ch. 18)."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 0B.1 à 0B.27 | vérifier tes exercices papier ✏️ | ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {ex.title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if kind == "exercise":
        title = "# 0B · Maths du lycée au ML — notebook d'exercices"
        how = ("La partie 0 vérifie tes exercices papier. Chaque exercice de code : un énoncé, une cellule à "
               "compléter (les `...` et les `raise NotImplementedError`), puis une cellule de vérification "
               "(`wb.check`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas "
               "fait affiche ⏳. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch00b_maths/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 0B`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 0B · Maths du lycée au ML — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Vérifier tes calculs à la main (partie 0), puis les retrouver en code.\n"
               "- Écrire l'algèbre linéaire en Python pur (`mylearn.linalg_basics`) et la comparer à NumPy.\n"
               "- Estimer des dérivées et des gradients par des pentes numériques, et lire des lignes de niveau.\n"
               "- Confirmer des probabilités, des espérances et des variances par simulation.\n\n"
               "**Rappel express.** $\\sum$ est une boucle ; $(m, n) \\times (n, p) \\to (m, p)$ ; "
               "$\\frac{dz}{dx} = \\frac{dz}{dy}\\frac{dy}{dx}$ ; on descend en soustrayant le gradient ; "
               "en code, `np.log` est $\\ln$ et `A @ B` est le produit matriciel (`A * B` est élément par élément).")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu vérifier qu'un produit matriciel existe et donner sa forme, sans calculer, puis l'écrire "
               "en Python pur et avec `@` ?\n"
               "2. Sais-tu vérifier une dérivée calculée à la main avec une pente centrée, et dériver une composition "
               "avec la règle de la chaîne ?\n"
               "3. Sais-tu calculer une espérance et une variance à partir d'un tableau de probabilités, et les "
               "retrouver par simulation ?\n\n"
               "**Pour aller plus loin** : la série vidéo *Essence of linear algebra* de 3Blue1Brown (les matrices "
               "vues comme des transformations du plan) et le livre libre *Mathematics for Machine Learning* "
               "(Deisenroth, Faisal et Ong), cités dans la fiche. La suite : le ch. 2 (statistiques) réutilise la "
               "moyenne, la variance et la simulation ; le ch. 5, les dérivées et le gradient ; `linalg_basics` "
               "t'aidera à lire ce que font les couches des ch. 16 à 18.")]


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
