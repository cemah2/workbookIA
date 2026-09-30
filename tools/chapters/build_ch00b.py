#!/usr/bin/env python
"""Build the two notebooks of chapter 0B from a single source (used by Claude).

    python tools/chapters/build_ch00b.py
    python tools/run_all_notebooks.py chapitres/ch00b_maths/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch00b_maths/03_notebook.ipynb

Session 1 of chapter 0B: Part 0 (checks of the ✏️ paper exercises 0B.1 to 0B.27).
The notebook exercises 0B.33 to 0B.54 are added in session 2 (PARTS below).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Paper, badge, md, paper_cells, part_cells,  # noqa: E402
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
# Notebook parts (0B.33 to 0B.54): session 2
# ---------------------------------------------------------------------------
PARTS: list = []

NEXT_SESSION = [
    ("A · Nombres et fonctions en code", "0B.33 à 0B.37", "📦 🔮 🐛"),
    ("B · mylearn.linalg_basics : vecteurs et matrices en Python pur", "0B.38 à 0B.46", "🔨 📦 🔬 🔮 🐛"),
    ("C · Dérivées et gradients en code", "0B.47 à 0B.51", "🔨 📈 🔮"),
    ("D · Probabilités simulées et défi final", "0B.52 à 0B.54", "🔬 🔨 🏆"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 0B.1 à 0B.27 | vérifier tes exercices papier ✏️ | ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {ex.title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if not PARTS:
        rows += [f"| {title} | {ids} | *prochaine session de génération* | {types} | | |" for title, ids, types in NEXT_SESSION]
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
    return [md("## ✅ Bilan (partie 0)\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu vérifier qu'un produit matriciel existe et donner sa forme, sans calculer ?\n"
               "2. Sais-tu dériver une composition avec la règle de la chaîne, en nommant les étapes ?\n"
               "3. Sais-tu calculer une espérance et une variance à partir d'un tableau de probabilités ?\n\n"
               "*Les exercices de code 0B.33 à 0B.54 (parties A à D) seront ajoutés à la prochaine session "
               "de génération.*")]


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
