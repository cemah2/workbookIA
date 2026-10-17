#!/usr/bin/env python
"""Build the two notebooks of chapter 17 from a single source (used by Claude).

    python tools/chapters/build_ch17.py
    python tools/run_all_notebooks.py chapitres/ch17_activations/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch17_activations/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper
exercises 17.1, 17.2, 17.5 and 17.6, of the numerical checks of the proof ∂ 17.4 and of the 🧮 estimate 17.8
(02_exercices.md). The proofs ∂ 17.3 and 17.7, the derivations of ∂ 17.4, the reflection (🗣️ 17.9, 📄 17.10)
and the interview are corrected with 05_solutions.md.
Parts A to D: exercises 17.11 to 17.24 (the learner's mylearn.nn.activations, scikit-learn's MLPClassifier,
saturation and gradient experiments on synthetic data).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "17"
FOLDER = "chapitres/ch17_activations"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np
from scipy import special


def sigma(z):
    return special.expit(z)


def dsigma(z):
    return special.expit(z) * special.expit(-z)


def Phi(z):
    return 0.5 * (1.0 + special.erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def softmax(z):
    return special.softmax(np.asarray(z, dtype=float))


# 17.1: the 3-2-2-1 network without activation
W1_171 = np.array([[1.0, -2.0], [0.0, 1.0], [3.0, 1.0]])
B1_171 = np.array([1.0, -1.0])
W2_171 = np.array([[2.0, 1.0], [-1.0, 3.0]])
B2_171 = np.array([0.0, 2.0])
W3_171 = np.array([[1.0], [-2.0]])
B3_171 = 0.5
X_171 = np.array([1.0, 2.0, -1.0])
W_EQ_171 = (W1_171 @ W2_171 @ W3_171).ravel()
B_EQ_171 = float((B1_171 @ W2_171 + B2_171) @ W3_171 + B3_171)


def out_171(x, f=lambda z: z):
    h = f(x @ W1_171 + B1_171)
    g = f(h @ W2_171 + B2_171)
    return float((g @ W3_171).item() + B3_171)


# 17.2: nine activations at two points (Phi rounded as in the statement)
PHI_172 = {-1.0: 0.1587, 1.5: 0.9332}


def nine_172(z):
    return [max(0.0, z), z if z > 0 else 0.1 * z, max(z, -0.5), math.log1p(math.exp(z)),
            z if z > 0 else math.expm1(z), float(sigma(z)), math.tanh(z), z * float(sigma(z)), z * PHI_172[z]]


# 17.5: the maxout of three lines
LINES_175 = [(-2.0, -1.0), (0.5, -0.5), (3.0, -4.0)]


def maxout_175(z):
    return max(a * z + b for a, b in LINES_175)


# 17.6: the six scores of figure 17.24
Z_176 = np.array([0.3, 0.1, 0.8, 0.4, 0.2, 0.6])          # A to F
P_176 = softmax(Z_176)

# 17.8: a 1024 -> 1024 layer on a batch of 256
MULTS_178 = 256 * 1024 * 1024
VALUES_178 = 256 * 1024
BYTES_178 = VALUES_178 * 4 * 2'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
LETTERS = 'letters glued in alphabetical order, e.g. "BD"'
SHAPE = 'a shape between quotes, e.g. "(5, 7)"'

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("17.Q1", "À quoi sert la fonction d'activation ?", [
        ("a", "a whole number", "4 * 2 + 2",
         r'''mistakes={"c'est le compte du MLP 4-8-8-2 de départ ; la couche unique relie directement 4 entrées à 2 sorties": 4 * 8 + 8 + 8 * 8 + 8 + 8 * 2 + 2,
          "une couche dense a aussi un biais par neurone de sortie": 8,
          "tu as compté la première couche du réseau de départ ; la couche unique relie les entrées aux sorties": 40,
          "la couche unique relie les 4 entrées aux 2 sorties : combien de poids, combien de biais ?": 6}'''),
        ("b", TF, "False",
         r'''mistakes={"une activation affine se fond-elle dans la couche suivante ? Compose deux applications affines": True}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"les poids multiplient les entrées brutes : l'activation vient-elle avant ?": "A",
          "le neurone additionne d'abord tous ses produits, et son biais": "B",
          "chaque couche cachée a son activation, pas seulement la sortie": "D"}'''),
        ("d", TF, "False",
         r'''mistakes={"sans activation entre elles, les trois couches cachées se composent-elles en une seule application affine ?": True}'''),
    ]),
    Paper("17.Q2", "Un nombre en entrée, un nombre en sortie : la forme d'une activation", [
        ("a", SHAPE, '"(32, 10)"',
         r'''mistakes={"une activation élément par élément agit sur chaque case : la forme change-t-elle ?": "(32,)",
          "chaque case reste à sa place : une ligne par exemple, une colonne par neurone": "(10, 32)",
          "aucune réduction : une valeur de sortie par valeur d'entrée": "(10,)"}'''),
        ("b", TF, "True",
         r'''mistakes={"relis le §17.2.1 : que fait-on en pratique, neurone par neurone ou couche par couche ?": False}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"un noyau est une notion du ch. 13 (SVM)": "B", "un logit est un score de sortie, avant la softmax": "C",
          "une fonction de répartition est une notion de probabilités (GELU en utilise une)": "D"}'''),
        ("d", TF, "False",
         r'''mistakes={"élément par élément : de quoi dépend chaque case du résultat ?": True}'''),
    ]),
    Paper("17.Q3", "Linéaire ou non linéaire ? Classer six courbes", [
        ("a", 'six letters L or N, glued, e.g. "NNLLNL"', '"LLNNNL"',
         r'''mistakes={"relis l'élément 2 : une droite qui ne passe pas par l'origine est-elle « linéaire » au sens du livre ?": "LNNNNL",
          "relis l'élément 6 : deux morceaux qui suivent la même formule forment-ils une seule droite ?": "LLNNNN",
          "relis l'élément 3 : un escalier est-il une seule droite ?": "LLLNNL",
          "relis l'élément 4 : la valeur absolue a-t-elle un coude ?": "LLNLNL",
          "relis l'élément 5 : ReLU a-t-elle un coude ?": "LLNNLL"}'''),
    ]),
    Paper("17.Q4", "Pourquoi une activation doit-elle donner une seule valeur par entrée ?", [
        ("a", TF, "False",
         r'''mistakes={"combien de points du cercle se trouvent au-dessus de z = 0 ?": True}'''),
        ("b", TF, "True",
         r'''mistakes={"relis le §17.3.1 : une fonction doit-elle être continue, ou avoir une seule valeur par entrée ?": False}'''),
        ("c", LETTERS, '"AD"',
         r'''mistakes={"une verticale coupe-t-elle la parabole couchée en un seul point ?": "ABD",
          "il en manque une : refais le test de la verticale sur chaque courbe": "A",
          "une verticale coupe-t-elle un cercle en un seul point ?": "ACD",
          "une droite verticale donne une infinité de valeurs pour z = 1": "ADE",
          "il en manque une : refais le test de la verticale pour chaque courbe": "D"}'''),
    ]),
    Paper("17.Q5", "Marche unité, Heaviside, signe : les distinguer", [
        ("a", "a number", "0.0",
         r'''decimals=1, mistakes={"z = −0,5 est à gauche du seuil 0 : quelle valeur prend la marche de ce côté ?": 1,
          "la marche de Heaviside vaut 0 ou 1, jamais −1": -1, "la marche de Heaviside ne prend jamais la valeur ½ hors du seuil": 0.5}'''),
        ("b", "a number", "1.0",
         r'''decimals=1, mistakes={"relis la convention du livre au seuil (le point plein) : quel côté l'emporte ?": 0,
          "la marche unité vaut 0 ou 1": 1.5}'''),
        ("c", "a number", "0",
         r'''mistakes={"essaie np.sign(0.0) : NumPy suit l'une des deux variantes de la fonction signe": 1,
          "la fonction signe ne rend −1 que pour z < 0": -1}'''),
        ("d", "a number", "1",
         r'''mistakes={"relis la convention de ta marche unité au seuil : quel côté l'emporte ?": -1,
          "c'est la convention de np.sign ; step suit celle de la marche unité": 0}'''),
        ("e", TF, "True",
         r'''mistakes={"une marche est plate de chaque côté du seuil : quelle est la pente d'un palier ?": False}'''),
    ]),
    Paper("17.Q6", "ReLU, leaky, paramétrique, décalée : qui fait quoi ?", [
        ("a", "a number", "0.1 * -3",
         r'''decimals=2, mistakes={"la leaky ReLU garde une petite fraction de z négatif, elle ne rend pas 0": 0,
          "pour z < 0, la leaky ReLU ne rend pas z lui-même": -3, "le signe de z est conservé": 0.3,
          "c'est la pente 0,01 de PyTorch ; l'énoncé dit 0,1": -0.03}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"le coude d'une PReLU reste en 0": "B", "la pente des z positifs reste 1": "C",
          "relis l'encadré « Le livre, corrigé » du §17.5 : tirée au hasard, ou apprise ?": "D"}'''),
        ("c", "a number", "max(-3, -2)",
         r'''mistakes={"relis la définition de la ReLU décalée : jusqu'où descend-elle ?": -3, "la ReLU décalée ne remonte pas à 0": 0}'''),
        ("d", TF, "False",
         r'''mistakes={"avec une pente 1 des deux côtés, reste-t-il un coude ?": True}'''),
        ("e", "a number", "0.0",
         r'''decimals=1, mistakes={"relis l'encadré sur la dérivée en un point anguleux : PyTorch prend la pente de quel côté ?": 1,
          "PyTorch choisit l'une des deux pentes, 0 ou 1": 0.5}'''),
    ]),
    Paper("17.Q7", "Maxout et ReLU bruitée", [
        ("a", "a number", "float(max(0.5 * -3, 2 * -3 - 1, 3 - 2))",
         r'''decimals=1, mistakes={"le maxout garde la PLUS HAUTE des trois droites en z = −3": -7,
          "calcule les trois valeurs en z = −3, puis prends la plus grande": -1.5,
          "−z − 2 en z = −3 vaut 3 − 2": 5}'''),
        ("b", "a number", "float(max(0.5 * 2, 2 * 2 - 1, -2 - 2))",
         r'''decimals=1, mistakes={"le maxout garde la plus haute des trois droites en z = 2": 1, "calcule chaque droite en z = 2": -4}'''),
        ("c", "a whole number", "2",
         r'''mistakes={"une seule droite reste une droite : il faut un coude": 1, "ReLU a deux morceaux de droite": 3}'''),
        ("d", TF, "False",
         r'''mistakes={"la ReLU bruitée tire un nouveau bruit à chaque passe : la sortie peut-elle changer ?": True}'''),
    ]),
    Paper("17.Q8", "Saturation : quelles fonctions, et de quel côté ?", [
        ("a", 'six characters among G, D, 2, 0, glued, e.g. "D20G2D"', '"22G0G0"',
         r'''mistakes={"relis l'élément 3 : ReLU a-t-elle une pente nulle d'un côté ?": "2200G0",
          "relis l'élément 4 : la pente de la leaky ReLU s'annule-t-elle ?": "22GGG0",
          "relis l'élément 5 : la pente de l'ELU tend-elle vers 0 pour z très négatif ?": "22G000",
          "relis l'élément 3 : de quel côté ReLU est-elle plate ?": "22D0G0",
          "relis la consigne : quelle lettre désigne le côté des z négatifs ?": "22D0D0",
          "relis l'élément 5 : de quel côté l'ELU devient-elle plate ?": "22G0D0",
          "relis l'élément 1 : la sigmoïde devient-elle plate d'un seul côté ?": "D2G0G0",
          "relis les éléments 1 et 2 : la sigmoïde et tanh deviennent-elles plates d'un seul côté ?": "GGG0G0"}'''),
        ("b", "a number with 4 decimals", "float(dsigma(6.0))",
         r'''decimals=4, mistakes={"c'est σ(6) ; on demande sa dérivée σ(6)(1 − σ(6))": 0.9975,
          "σ′ = σ(1 − σ), pas σ(1 + σ)": 1.9926, "as-tu calculé la bonne fonction au bon point ?": 0.0099}'''),
        ("c", "a number with 4 decimals", "1 - math.tanh(3) ** 2",
         r'''decimals=4, mistakes={"tanh′ = 1 − tanh², pas 1 − tanh": 0.0049, "c'est tanh(3) ; on demande sa dérivée": 0.9951,
          "as-tu calculé la bonne fonction au bon point ?": 0.0025}'''),
        ("d", "a number with 2 decimals", "0.25",
         r'''decimals=2, mistakes={"relis la question : on demande la dérivée de la sigmoïde": 1,
          "c'est σ(0) ; on demande σ′(0)": 0.5}'''),
    ]),
    Paper("17.Q9", "Swish, SiLU, GELU : des ReLU lissées", [
        ("a", "a number with 4 decimals", "-0.5 * float(sigma(-0.5))",
         r'''decimals=4, mistakes={"c'est σ(−0,5) ; SiLU multiplie encore par z": 0.3775,
          "SiLU(z) = z σ(z), avec σ(−0,5), pas σ(0,5)": -0.3112, "le signe : z = −0,5 est négatif": 0.1888}'''),
        ("b", TF, "False",
         r'''mistakes={"regarde SiLU pour z négatif : descend-elle sous 0 avant de remonter ?": True}'''),
        ("c", "a number with 4 decimals", "1 * 0.8413",
         r'''decimals=4, mistakes={"GELU(z) = z Φ(z), pas z (1 − Φ(z))": 0.1587,
          "GELU(z) = z Φ(z), pas (1 + Φ(z))/2": 0.9207, "c'est la densité φ(1) ; GELU utilise la fonction de répartition Φ": 0.2420,
          "c'est SiLU(1) : GELU utilise Φ, pas σ": 0.7311}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"l'identité garderait z pour z < 0 : que vaut σ(βz) pour z < 0 et β grand ?": "B",
          "z σ(βz) garde le facteur z : ce n'est pas une marche": "C", "z/2 : que vaut σ(βz) quand β devient très grand ?": "D"}'''),
        ("e", LETTER, '"D"',
         r'''mistakes={"σ(0) = ½ : que vaut z σ(0) ?": "A", "σ(0) = ½, pas 1": "B", "z σ(0) garde le facteur z": "C"}'''),
    ]),
    Paper("17.Q10", "Reconnaître les courbes de la figure 17.22", [
        ("a", 'six letters glued, e.g. "FEDCBA"', '"BEFDAC"',
         r'''mistakes={"sigmoïde et tanh : regarde la valeur la plus basse de chaque courbe en S (0 ou −1)": "AEFDBC",
          "softplus et ELU : laquelle descend sous 0 pour z négatif ?": "BEFCAD",
          "SiLU et ELU : laquelle remonte vers 0 pour z très négatif, et laquelle tend vers −1 ?": "BDFEAC",
          "la leaky ReLU a un coude net en 0 ; les autres sont lisses": "BFEDAC",
          "SiLU et softplus : laquelle descend sous 0 ?": "BCFDAE"}'''),
    ]),
    Paper("17.Q11", "Softmax : somme à 1, ordre conservé, mais pas une activation", [
        ("a", TF, "True",
         r'''mistakes={"l'exponentielle est croissante : si z_i > z_j, que dire de e^(z_i) et e^(z_j) ?": False}'''),
        ("b", TF, "True",
         r'''mistakes={"ajouter 10 à chaque score multiplie numérateur et dénominateur par le même facteur": False}'''),
        ("c", TF, "False",
         r'''mistakes={"doubler les scores double leurs écarts : les rapports e^(z_i − z_j) changent-ils ?": True}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"la softmax est dérivable partout (∂ 17.7)": "B", "ses sorties sont des probabilités, entre 0 et 1": "C",
          "ReLU n'a pas de paramètre non plus, et c'est une activation": "D"}'''),
    ]),
    Paper("17.Q12", "Vrai ou faux : des scores 0,6 / 0,3 / 0,1 se lisent comme des probabilités", [
        ("a", TF, "False",
         r'''mistakes={"relis le §17.8 : des scores bruts se comparent-ils comme des probabilités ?": True}'''),
        ("b", "a number with 2 decimals", "math.exp(0.6 - 0.3)",
         r'''decimals=2, mistakes={"c'est le rapport des scores bruts ; après softmax, le rapport vaut e^(z_A − z_B)": 2,
          "c'est e^(z_B − z_A), le rapport inverse": 0.74, "c'est e^(z_A) ; le rapport vaut e^(z_A − z_B)": 1.82}'''),
        ("c", "a number with 3 decimals", "float(softmax([0.6, 0.3, 0.1])[0])",
         r'''decimals=3, mistakes={"c'est le score brut ; la softmax divise e^(0,6) par la somme des trois exponentielles": 0.6,
          "c'est e^(0,6) : divise par la somme des trois exponentielles": 1.822}'''),
        ("d", TF, "False",
         r'''mistakes={"multiplier les scores par 10 multiplie leurs écarts : relis la propriété des rapports": True}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("17.R1", "Quelle initialisation va avec ReLU, et pourquoi ?", [
        ("a", "a number", "2 / 50",
         r'''decimals=3, mistakes={"relis la question : on demande une variance, pas un écart-type": 0.2, "c'est la règle de LeCun (1/n_in) ; relis celle de He": 0.02,
          "relis la règle de He : le fan-in multiplie-t-il ou divise-t-il ?": 100}'''),
        ("b", "a number", "math.sqrt(2 / 50)",
         r'''decimals=3, mistakes={"relis la question : la racine de quoi ?": 0.04, "c'est l'écart-type de LeCun, √(1/50)": 0.141}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"la règle de He ne regarde que la passe avant et le fan-in": "B", "le fan-out n'apparaît pas dans la règle de He": "C",
          "les biais sont nuls à l'initialisation": "D"}'''),
        ("d", TF, "True",
         r'''mistakes={"relis le ch. 16 : He compense le facteur ½ de ReLU ; tanh met-elle la moitié des valeurs à 0 ?": False}'''),
    ]),
    Paper("17.R2", "D'où viennent les probabilités d'une forêt aléatoire ?", [
        ("a", LETTER, '"A"',
         r'''mistakes={"relis le ch. 14 : scikit-learn fait-il un vote dur ou un vote souple (une moyenne de probabilités) ?": "B",
          "aucune softmax dans une forêt : chaque arbre donne déjà des proportions": "C", "aucune sigmoïde dans une forêt": "D"}'''),
        ("b", "a number with 2 decimals", "(0.9 + 0.6 + 0.0) / 3",
         r'''decimals=2, mistakes={"c'est la proportion d'arbres qui prédisent la classe 1 (vote dur) ; scikit-learn moyenne les probabilités": 2 / 3,
          "la moyenne des trois probabilités, sans en oublier": 0.75}'''),
        ("c", TF, "False",
         r'''mistakes={"chaque arbre donne une distribution qui somme à 1 : que dire de leur moyenne ?": True}'''),
    ]),
    Paper("17.R3", "Pourquoi un perceptron à seuil ne s'entraîne pas par gradient ?", [
        ("a", "a number", "0",
         r'''mistakes={"la marche est plate de chaque côté du seuil : z = 0,3 est sur un palier": 1}'''),
        ("b", TF, "False",
         r'''mistakes={"relis le ch. 10 : la règle du perceptron corrige les poids d'après quoi ?": True}'''),
        ("c", LETTERS, '"ACE"',
         r'''mistakes={"ReLU a une pente nulle sur toute une demi-droite : est-ce « presque partout » non nul ?": "ACEF",
          "la fonction signe est plate de chaque côté de 0": "ABCE", "la marche de Heaviside est plate de chaque côté de 0": "ACDE",
          "il en manque une : regarde la pente de chaque fonction pour z < 0": "AC", "il en manque une : relis les courbes en S": "AE"}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("17.1", "Effondrer un réseau linéaire en un seul neurone", [
        ("a", "a list of three numbers", "W_EQ_171.tolist()",
         r'''decimals=1, mistakes={"as-tu oublié la matrice de la couche du milieu, W₂ ?": [5.0, -2.0, 1.0],
          "W₂ s'utilise telle quelle, pas transposée : W₁ W₂ W₃": [14.0, -5.0, 7.0]}'''),
        ("b", "a number", "B_EQ_171",
         r'''decimals=2, mistakes={"le biais b₁ traverse aussi les couches suivantes : multiplie-le par W₂ puis W₃": -3.5,
          "le biais b₂ traverse aussi la dernière couche : multiplie-le par W₃": 7.5,
          "as-tu ajouté le biais de sortie b₃ = 0,5 ?": 3.0, "les biais ne s'additionnent pas tels quels : chacun est multiplié par les matrices qui le suivent": 2.5}'''),
        ("c", "a number", "out_171(X_171)",
         r'''decimals=2, mistakes={"as-tu ajouté le biais équivalent ?": 7.0}'''),
        ("d", "a number", "out_171(X_171, lambda z: np.maximum(z, 0.0))",
         r'''decimals=2, mistakes={"ReLU change-t-elle quelque chose ici ? Calcule les pré-activations de la première couche cachée": out_171(X_171),
          "ReLU ne s'applique pas à la sortie : elle peut être négative": 0,
          "as-tu appliqué ReLU après la première couche cachée aussi ?": 0.5}'''),
        ("e", "a whole number", "3 * 2 + 2 + 2 * 2 + 2 + 2 * 1 + 1",
         r'''mistakes={"ajoute les biais : un par neurone de chaque couche": 12, "l'entrée n'a pas de biais": 20}'''),
        ("f", "a whole number", "3 + 1",
         r'''mistakes={"le neurone équivalent a aussi un biais": 3}'''),
    ]),
    Paper("17.2", "Tableau de valeurs de neuf activations", [
        ("a", "a list of nine numbers, 4 decimals", "[round(v, 4) for v in nine_172(-1.0)]",
         r'''decimals=4, mistakes={"softplus(−1) = ln(1 + e^(−1)), pas ln(1 + e)": [0.0, -0.1, -0.5, 1.3133, -0.6321, 0.2689, -0.7616, -0.2689, -0.1587],
          "ELU(−1) = e^(−1) − 1, négatif": [0.0, -0.1, -0.5, 0.3133, 0.3679, 0.2689, -0.7616, -0.2689, -0.1587],
          "la leaky ReLU de pente 0,1 garde 0,1 × z pour z < 0": [0.0, 0.0, -0.5, 0.3133, -0.6321, 0.2689, -0.7616, -0.2689, -0.1587]}'''),
        ("b", "a list of nine numbers, 4 decimals", "[round(v, 4) for v in nine_172(1.5)]",
         r'''decimals=4, mistakes={"SiLU(1,5) = 1,5 × σ(1,5), pas σ(1,5)": [1.5, 1.5, 1.5, 1.7014, 1.5, 0.8176, 0.9051, 0.8176, 1.3998],
          "GELU(1,5) = 1,5 × Φ(1,5), pas Φ(1,5)": [1.5, 1.5, 1.5, 1.7014, 1.5, 0.8176, 0.9051, 1.2264, 0.9332]}'''),
        ("c", "a whole number", "sum(v < 0 for v in nine_172(-1.0))",
         r'''mistakes={"compte aussi les fonctions qui descendent un peu sous 0 (SiLU, GELU)": 4,
          "softplus et la sigmoïde restent positives ; ReLU vaut 0, qui n'est pas négatif": 7, "relis ta liste a) : compte les valeurs strictement négatives": 5}'''),
    ]),
    Paper("17.4", "Dériver la sigmoïde, tanh et softplus", [
        ("a", "a number", "0.75 * 0.25",
         r'''decimals=4, mistakes={"c'est σ(ln 3) ; on demande σ′(ln 3) = σ(1 − σ)": 0.75, "relis la question : on demande σ′ au point ln 3": 0.25,
          "0,5625 = σ² : relis le second facteur de σ(1 − σ)": 0.5625}'''),
        ("b", "a number", "1 - 0.6 ** 2",
         r'''decimals=4, mistakes={"c'est tanh(ln 2) ; on demande sa dérivée": 0.6, "tanh′ = 1 − tanh², pas 1 − tanh": 0.4, "c'est tanh² ; tanh′ = 1 − tanh²": 0.36}'''),
        ("c", "a number", "0.8",
         r'''decimals=4, mistakes={"softplus′ = σ, pas σ′": 0.16, "c'est softplus(ln 4) = ln 5 ; on demande sa dérivée": 1.6094, "σ(ln 4) = 4/5, pas 1/5": 0.2}'''),
        ("d", "a number with 5 decimals", "0.75 * 0.25 * (1 - 2 * 0.75)",
         r'''decimals=5, mistakes={"le signe : 1 − 2σ est négatif quand σ = 0,75": 0.09375, "σ″ = σ′(1 − 2σ), pas σ′(1 − σ)": 0.046875}'''),
        ("e", "a number", "1 / 0.25",
         r'''decimals=2, mistakes={"le rapport de tanh′ à σ′, pas l'inverse": 0.25, "tanh′(z) = 4σ′(2z) : où chaque dérivée atteint-elle son maximum ?": 2}'''),
        ("f", "a number with 4 decimals", "0.75 * (1 + math.log(3) * 0.25)",
         r'''decimals=4, mistakes={"SiLU′ = σ(1 + z(1 − σ)) : as-tu oublié le facteur z ?": 0.9375, "SiLU′ = σ(1 + z(1 − σ)), pas σ(1 + z)": 1.5740,
          "c'est σ′(ln 3) ; SiLU′ a aussi un terme σ": 0.1875, "c'est σ(ln 3) seul : il manque le facteur 1 + z(1 − σ)": 0.75,
          "as-tu multiplié par σ(ln 3) ? SiLU′ = σ(1 + z(1 − σ))": 1.2747}'''),
    ]),
    Paper("17.5", "Maxout et ReLU : construire |x|, une cuvette et une bosse", [
        ("a", "a list of two numbers, in increasing order", "[-0.2, 1.4]",
         r'''decimals=2, mistakes={"−2z − 1 et 3z − 4 se croisent en 0,6, mais sous la droite 0,5z − 0,5 : est-ce un coude de la cuvette ?": [-0.2, 0.6, 1.4],
          "les coudes sont là où la droite la plus haute change : compare les trois droites deux à deux": [0.6],
          "dans l'ordre croissant": [1.4, -0.2]}'''),
        ("b", "a number", "maxout_175(-0.2)",
         r'''decimals=2, mistakes={"en z = 0,6, la droite la plus haute n'est ni −2z − 1 ni 3z − 4": -2.2,
          "c'est la valeur au coude de droite ; la cuvette est plus basse ailleurs": 0.2, "le minimum d'un maximum de droites n'est pas 0 ici": 0,
          "−0,2 n'est pas la valeur la plus basse : calcule la cuvette en chacun de ses coudes": -0.2}'''),
        ("c", "a list [a, b, c]", "[1, -2, 1]",
         r'''decimals=1, mistakes={"après le sommet, la pente doit passer de +1 à −1 : de combien la ReLU(z) doit-elle changer la pente ?": [1, -1, 1],
          "le chapeau est positif : quel signe pour le premier coefficient ?": [-1, 2, -1]}'''),
        ("d", TF, "False",
         r'''mistakes={"un maximum de droites peut-il monter puis redescendre ? Relis le §17.5 sur la convexité": True}'''),
    ]),
    Paper("17.6", "Softmax à la main : translation, température, rapports", [
        ("a", "a number with 2 decimals", "float(P_176[2] / P_176[1])",
         r'''decimals=2, mistakes={"c'est le rapport des scores bruts 0,8 / 0,1 ; après softmax, le rapport vaut e^(z_C − z_B)": 8,
          "c'est l'écart des scores ; le rapport des probabilités vaut e^(z_C − z_B)": 0.7, "c'est e^(z_B − z_C), le rapport inverse": 0.50,
          "c'est la conclusion du livre ; refais le calcul": 1.99}'''),
        ("b", "a number with 3 decimals", "float(P_176[2])",
         r'''decimals=3, mistakes={"c'est 0,8 divisé par la somme des scores ; la softmax prend les exponentielles": 0.333,
          "c'est e^(0,8) : divise par la somme des six exponentielles": 2.226}'''),
        ("c", "a number with 3 decimals", "float(softmax(Z_176 + 5)[2])",
         r'''decimals=3, mistakes={"ajoute 5 à CHACUN des six scores, pas seulement à celui de C": 0.979}'''),
        ("d", "a number with 3 decimals", "float(softmax(Z_176 / 0.1)[2])",
         r'''decimals=3, mistakes={"la température DIVISE les scores : z / 0,1, pas z × 0,1": 0.173,
          "as-tu divisé les scores par T avant la softmax ?": 0.241}'''),
        ("e", "a number with 3 decimals", "1 / 6",
         r'''decimals=3, mistakes={"quand T grandit, les scores divisés par T tendent tous vers 0 : vers quelle distribution ?": 1,
          "p_C ne tend pas vers 0 : que deviennent les écarts de scores divisés par T ?": 0}'''),
        ("f", "a number with 2 decimals", "math.exp(2 * (0.8 - 0.1))",
         r'''decimals=2, mistakes={"doubler les scores double l'écart dans l'exposant, pas le rapport lui-même": 4.03,
          "c'est le rapport d'avant : doubler les scores change-t-il l'écart z_C − z_B ?": 2.01}'''),
    ]),
    Paper("17.8", "Fermi : une activation coûte-t-elle cher face au produit matriciel ?", [
        ("a", "a whole number of millions", "round(MULTS_178 / 1e6)",
         r'''mistakes={"en millions : 256 × 1 024 × 1 024 divisé par 10⁶": MULTS_178,
          "environ 1 million, c'est le nombre de poids : combien de fois chacun sert-il dans un batch ?": 1}'''),
        ("b", "a whole number of thousands", "round(VALUES_178 / 1e3)",
         r'''mistakes={"c'est le nombre de poids, en milliers ; l'activation agit sur les valeurs de Z": 1049,
          "en milliers : divise par 10³": VALUES_178, "une valeur par neurone ET par exemple du batch": 1,
          "relis la question : une valeur par neurone et par exemple, en milliers": 0}'''),
        ("c", "a whole number", "MULTS_178 // VALUES_178",
         r'''mistakes={"le rapport de a) à b), pas l'inverse": 0, "c'est la taille du batch : combien de multiplications donnent une seule valeur de Z ?": 256,
          "c'est le nombre de poids ; on demande des multiplications par valeur d'activation": 1048576}'''),
        ("d", "a percentage with 1 decimal", "100 * 30 * VALUES_178 / (2 * MULTS_178 + 30 * VALUES_178)",
         r'''decimals=1, mistakes={"une multiplication-addition compte pour 2 opérations : 2 × a)": 2.8,
          "on demande une part en pourcentage : multiplie par 100": 0.0144, "la part dans le TOTAL des opérations de la couche (produit et activation)": 1.5}'''),
        ("e", "a number of MB, 1 decimal", "BYTES_178 / 1e6",
         r'''decimals=1, mistakes={"l'activation lit Z ET écrit A : as-tu compté les deux ?": 1.0,
          "4 octets par nombre en float32, pas 8": 4.2, "un float32 occupe 4 octets (32 bits) : as-tu compté des bits ?": 8.4}'''),
        ("f", "a number of microseconds, 1 decimal", "BYTES_178 / 2e12 * 1e6",
         r'''decimals=1, mistakes={"le GPU déplace 2 × 10¹² octets par seconde : divise, puis convertis en microsecondes": 2.1,
          "1 seconde vaut 10⁶ microsecondes": 0.000001}'''),
        ("g", "a number of microseconds, 1 decimal", "2 * MULTS_178 / 5e13 * 1e6",
         r'''decimals=1, mistakes={"une multiplication-addition compte pour 2 opérations": 5.4, "1 seconde vaut 10⁶ microsecondes": 0.0000107}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 17.1, 17.2, 17.5 et 17.6, ∂ 17.4 et 🧮 17.8)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "plusieurs choix ou une suite de codes : les caractères collés, dans l'ordre demandé (`\"XYZ\"`) ; "
               "une forme NumPy : entre guillemets (`\"(5, 7)\"`) ; plusieurs nombres : une liste (`[4, 7]`). "
               "Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore "
               "remplies affichent ⏳. Les questions « dans ta copie », les preuves ∂ 17.3 et 17.7, les "
               "démonstrations de ∂ 17.4, la réflexion (🗣️ 17.9, 📄 17.10) et l'entretien se corrigent avec "
               "`05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 17.11 to 17.24)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools for the notebook exercises (parts A to D)
import contextlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from wb.errors import MylearnMissingError

try:
    import torch
except ImportError:                       # the comparisons with PyTorch are then left to the tests
    torch = None


def verdict(ex_id, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message."""
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")


def filled(*values):
    """True when none of the values is still `...` (or None): the answer has been written."""
    return all(value is not ... and value is not None for value in values)


def fr(x, digits=3):
    """A number written the French way, with a decimal comma (for the messages in French)."""
    return f"{x:.{digits}f}".replace(".", ",")


def print_answer(ex_id, value, **_):
    """Solutions notebook: show the value that the exercise notebook checks with wb.check."""
    try:
        shown = np.round(np.asarray(value, dtype=float), 4).tolist() if isinstance(value, (np.ndarray, list, tuple)) else value
    except (TypeError, ValueError):
        shown = value
    print(f"{ex_id}:", shown)


@contextlib.contextmanager
def checked(ex_id):
    """Like wb.attempt (⏳ while the code is not written yet), but any other error raised by YOUR code becomes one ❌
    line instead of stopping the cell, so that the tests run after it can say more."""
    with wb.attempt(ex_id):
        try:
            yield
        except (NotImplementedError, ImportError, MylearnMissingError):
            raise
        except Exception as exc:                              # a bug in the learner's code: report it, go on
            if isinstance(exc, TypeError) and "NoneType" in str(exc):
                raise                                         # wb.attempt explains the forgotten return
            print(f"❌ Ex {ex_id} : ton code lève {type(exc).__name__} : {str(exc)[:300]}")


def written(probe):
    """False while probe() stops on code not written yet (NotImplementedError, or no mylearn at all); True otherwise,
    even when it raises another error (the tests then say which)."""
    try:
        probe()
    except (NotImplementedError, ImportError, MylearnMissingError):
        return False
    except Exception:
        return True
    return True


def quiet(f, *args, **kwargs):
    """Call f and return (result, the RuntimeWarnings it emitted: overflow, invalid value, division by zero)."""
    with warnings.catch_warnings(record=True) as caught, np.errstate(over="warn", invalid="warn", divide="warn",
                                                                     under="ignore"):
        warnings.simplefilter("always")
        result = f(*args, **kwargs)
    return result, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


def same(ex_id, label, got, want, rtol=1e-9, atol=1e-12):
    """✅ if `got` is an array equal to `want` up to rounding, ❌ with both values otherwise; returns the verdict."""
    if got is None:
        print(f"❌ Ex {ex_id} : {label} renvoie None : as-tu oublié le return ?")
        return False
    try:
        arr = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        print(f"❌ Ex {ex_id} : {label} : attendu un tableau de nombres, reçu un objet {type(got).__name__}.")
        return False
    want = np.asarray(want, dtype=float)
    if arr.shape != want.shape:
        print(f"❌ Ex {ex_id} : {label} : attendu la forme {want.shape}, reçu la forme {arr.shape}.")
        return False
    ok = bool(np.allclose(arr, want, rtol=rtol, atol=atol, equal_nan=True))
    shown = np.round(arr, 8).tolist() if arr.size <= 6 else f"un tableau {arr.shape}"
    verdict(ex_id, ok, f"{label} : {shown}, comme attendu.",
            f"{label} : attendu {np.round(want, 8).tolist() if want.size <= 6 else 'autre chose'}, reçu {shown}"
            + ("" if want.size <= 6 else f" (plus grand écart {np.nanmax(np.abs(arr - want)):.3g})") + ".")
    return ok


def fit_quietly(model, X, y):
    """Fit a scikit-learn model without its ConvergenceWarning (the number of iterations is part of the protocol)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)


GRID = np.arange(-600, 601) / 100.0                      # z from -6 to 6, with an exact 0.0
EXTREME = np.array([-1000.0, -700.0, -40.0, 0.0, 40.0, 700.0, 1000.0])
TEST_FILES = {"activations": "tests/test_ch17_activations.py"}


def run_mylearn_tests(module, keyword, impl="learner"):
    """Run the tests of mylearn.nn.<module> selected by `keyword` (on YOUR code by default)."""
    test_file = TEST_FILES[module]
    command = [sys.executable, "-m", "pytest", test_file, "-k", keyword, "-q", "-p", "no:cacheprovider",
               "--color=no", "-rf", "--tb=no"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "1000"})   # long lines: the reason of each failure
    lines = result.stdout.strip().splitlines()
    failed = [line for line in lines if line.startswith("FAILED ")]
    for line in failed[:8]:                                  # the test, then the reason of its failure
        name, _, reason = line.removeprefix(f"FAILED {test_file}::").partition(" - ")
        print(f"❌ {name}\n   {reason[:800]}")
    if len(failed) > 8:
        print(f"   ... and {len(failed) - 8} other failed test(s)")
    print("pytest:", lines[-1] if lines else result.stderr.strip()[-300:])


print("tools ready · PyTorch", "available" if torch is not None else "absent (the tests will compare with SciPy)")'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="17")   # reload your saved file\n'


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre `mon_travail/mylearn/nn/activations.py` "
            "(créé par `python tools/start_chapter.py 17`), lis la docstring de chaque fonction, remplace les "
            "`raise NotImplementedError(...)` par ton code et **enregistre**. NumPy et la bibliothèque standard "
            "seulement (`math.erf` pour GELU) ; SciPy et PyTorch sont les **oracles** des tests. La cellule de "
            "vérification recharge ta librairie, essaie les exemples de la docstring, compare quelques résultats à "
            "SciPy, puis lance les tests de l'exercice ; `python -m pytest tests/test_ch17_activations.py -q` les lance "
            "tous dans un terminal. Les tests sont écrits comme te le fera faire 🛠️ 17.22 : un test paramétré par "
            "propriété, exécuté sur toute la famille (`[relu]`, `[softplus-beta2]`, `[gelu-approx_derivative]`…).")


COMMON_CONTRACT = ("**Contrat commun** (les docstrings, plus ces précisions que vérifient les tests) : chaque fonction "
                   "rend un **nouveau** tableau `float64` de la forme de `z` (un tableau de dimension 0, ou un "
                   "`np.float64`, pour un nombre seul), jamais une vue de `z`, qu'elle laisse intact ; une liste "
                   "d'entiers est acceptée ; aucune fonction n'émet de `RuntimeWarning` (débordement, valeur "
                   "invalide, division par zéro) ni ne rend de valeur infinie pour des entrées entre $-1\\,000$ et "
                   "$1\\,000$ ; chaque dérivée prend la **pré-activation** `z` et vaut, aux points anguleux, la "
                   "valeur de PyTorch.")


def ref_check(check: str, ex_id: str) -> str:
    """The solutions version of a 🔨 check cell: no wb.attempt, and the tests run on the reference."""
    return (check.replace(f'with checked("{ex_id}"):\n', "if True:\n")
            .replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", module="'))


def as_printed(check: str, ex_id: str) -> str:
    """The solutions version of a check cell that reads the learner's values: print them instead of checking."""
    return (check.replace(f'with wb.attempt("{ex_id}"):\n', "if True:\n").replace(f'with checked("{ex_id}"):\n', "if True:\n")
            .replace("wb.check(", "print_answer("))


# ---- 17.11 -----------------------------------------------------------------
GIVEN_11 = r'''X_11, y_11 = wb.synth.make_moons(n=500, noise=0.2, seed=1711)
Xtr_11, Xte_11, ytr_11, yte_11 = train_test_split(X_11, y_11, test_size=0.4, random_state=1711, stratify=y_11)
print("train", Xtr_11.shape, "test", Xte_11.shape)'''

EXPERIMENT_11 = r'''logreg_11 = fit_quietly(LogisticRegression(), Xtr_11, ytr_11)
flat_11 = fit_quietly(MLPClassifier((100, 100), activation="identity", solver="lbfgs", max_iter=3000, random_state=0),
                      Xtr_11, ytr_11)
bent_11 = fit_quietly(MLPClassifier((100, 100), activation="relu", solver="lbfgs", max_iter=3000, random_state=0),
                      Xtr_11, ytr_11)
n_params_11 = sum(W.size for W in flat_11.coefs_) + sum(b.size for b in flat_11.intercepts_)
acc_11 = {name: model.score(Xte_11, yte_11) for name, model in
          [("logistic regression", logreg_11), ("MLP without activation", flat_11), ("MLP with ReLU", bent_11)]}
for name, acc in acc_11.items():
    print(f"{name:24s} test accuracy {acc:.3f}")
agreement_11 = float(np.mean(flat_11.predict(Xte_11) == logreg_11.predict(Xte_11)))
print(f"MLP without activation: {n_params_11} parameters · same class as the logistic regression on {agreement_11:.1%} of the test points")
W_eq_11 = flat_11.coefs_[0] @ flat_11.coefs_[1] @ flat_11.coefs_[2]       # the collapsed weights (2, 1)
print("direction of the collapsed weights:", np.round(W_eq_11.ravel() / np.linalg.norm(W_eq_11), 3).tolist(),
      "· of the logistic regression:", np.round(logreg_11.coef_.ravel() / np.linalg.norm(logreg_11.coef_), 3).tolist())
fig_11, axes_11 = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (name, model) in zip(axes_11, [("régression logistique", logreg_11), ("MLP sans activation", flat_11),
                                       ("MLP avec ReLU", bent_11)]):
    wb.plot.plot_decision_boundary(model, Xte_11, yte_11, ax=ax, title=name)
plt.tight_layout()
plt.show()
wb.check("17.11a", beats_lr_11)
wb.check("17.11b", agree_11)
wb.check("17.11c", relu_11)'''

# ---- 17.12 -----------------------------------------------------------------
CHECK_12 = r'''with checked("17.12"):
    A = mylearn.nn.activations
    same("17.12", "identity([-2, 0, 2])", A.identity([-2, 0, 2]), [-2.0, 0.0, 2.0])
    same("17.12", "step([-2, 0, 2])", A.step([-2.0, 0.0, 2.0]), [0.0, 1.0, 1.0])
    same("17.12", "step([-2, 0, 2], low=-1)", A.step([-2.0, 0.0, 2.0], low=-1.0), [-1.0, 1.0, 1.0])
    same("17.12", "relu_derivative([-2, 0, 2])", A.relu_derivative([-2.0, 0.0, 2.0]), [0.0, 0.0, 1.0])
    same("17.12", "leaky_relu([-2, 0, 2], negative_slope=0.1)", A.leaky_relu([-2.0, 0.0, 2.0], negative_slope=0.1), [-0.2, 0.0, 2.0])
    same("17.12", "leaky_relu_derivative([-2, 0, 2], negative_slope=0.1)",
         A.leaky_relu_derivative([-2.0, 0.0, 2.0], negative_slope=0.1), [0.1, 0.1, 1.0])
with checked("17.12"):
    A = mylearn.nn.activations
    same("17.12", "relu sur z de -6 à 6, comme np.maximum", A.relu(GRID), np.maximum(GRID, 0.0))
    same("17.12", "step(z, threshold=0.5, low=-1, high=2) sur la grille", A.step(GRID, threshold=0.5, low=-1.0, high=2.0),
         -1.0 + 3.0 * np.heaviside(GRID - 0.5, 1.0))
    z_12 = np.array([1.0, 2.0, 3.0])
    out_12 = A.identity(z_12)
    verdict("17.12", isinstance(out_12, np.ndarray) and not np.shares_memory(out_12, z_12),
            "identity rend un nouveau tableau (le modifier ne touche pas z).",
            "identity(z) partage la mémoire de z : modifier le résultat modifierait z (np.asarray ne copie pas un tableau float64).")
if written(lambda: mylearn.nn.activations.relu([1.0])):
    run_mylearn_tests("activations", keyword="identity or step or relu")
else:
    print("⏳ Ex 17.12 : les tests attendent tes fonctions.")'''

# ---- 17.13 -----------------------------------------------------------------
CHECK_13 = r'''with checked("17.13"):
    A = mylearn.nn.activations
    same("17.13", "sigmoid([-2, 0, 2])", A.sigmoid([-2.0, 0.0, 2.0]), special.expit([-2.0, 0.0, 2.0]))
    same("17.13", "sigmoid_derivative([-2, 0, 2])", A.sigmoid_derivative([-2.0, 0.0, 2.0]), [0.10499358540350652, 0.25, 0.10499358540350652])
    same("17.13", "tanh_derivative([-2, 0, 2])", A.tanh_derivative([-2.0, 0.0, 2.0]), 1 - np.tanh([-2.0, 0.0, 2.0]) ** 2)
    same("17.13", "softplus([-2, 0, 2])", A.softplus([-2.0, 0.0, 2.0]), np.logaddexp(0.0, [-2.0, 0.0, 2.0]))
    same("17.13", "softplus_derivative([-2, 0, 2])", A.softplus_derivative([-2.0, 0.0, 2.0]), special.expit([-2.0, 0.0, 2.0]))
with checked("17.13"):
    A = mylearn.nn.activations
    for name_13 in ["sigmoid", "sigmoid_derivative", "tanh", "tanh_derivative", "softplus", "softplus_derivative"]:
        values_13, alarms_13 = quiet(getattr(A, name_13), EXTREME.copy())
        finite_13 = bool(np.all(np.isfinite(np.asarray(values_13, dtype=float))))
        verdict("17.13", finite_13 and not alarms_13, f"{name_13} sur z = ±1000, ±700, ±40, 0 : fini et sans avertissement.",
                f"{name_13} sur z = ±1000, ±700, ±40, 0 : " + (f"avertissement « {alarms_13[0]} »" if alarms_13 else "une valeur infinie ou nan")
                + " (relis le premier principe de la section sur le calcul sans débordement).")
    same("17.13", "sigmoid(-700) (tolérance relative)", A.sigmoid([-700.0]), special.expit([-700.0]), rtol=1e-6, atol=0.0)
    same("17.13", "softplus(z, beta=2, threshold=10) sur la grille (seuil compris, en z = 5)", A.softplus(GRID, beta=2.0, threshold=10.0),
         np.where(2 * GRID > 10, GRID, np.logaddexp(0.0, 2 * GRID) / 2), rtol=1e-9, atol=0.0)
if written(lambda: mylearn.nn.activations.sigmoid([1.0])):
    run_mylearn_tests("activations", keyword="sigmoid or tanh or softplus")
else:
    print("⏳ Ex 17.13 : les tests attendent tes fonctions.")'''

# ---- 17.14 -----------------------------------------------------------------
CHECK_14 = r'''with checked("17.14"):
    A = mylearn.nn.activations
    same("17.14", "elu([-2, 0, 2])", A.elu([-2.0, 0.0, 2.0]), [-0.8646647167633873, 0.0, 2.0])
    same("17.14", "elu_derivative([0], alpha=1.5)", A.elu_derivative([0.0], alpha=1.5), [1.5])
    same("17.14", "silu_derivative([-2, 0, 2])", A.silu_derivative([-2.0, 0.0, 2.0]), [-0.09078424878489121, 0.5, 1.0907842487848914])
    exact_14 = GRID * 0.5 * (1 + special.erf(GRID / np.sqrt(2)))
    same("17.14", "gelu sur la grille, comme z Φ(z) avec scipy.special.erf", A.gelu(GRID), exact_14)
    same("17.14", 'gelu([-2, 0, 2], approximate="tanh")', A.gelu([-2.0, 0.0, 2.0], approximate="tanh"),
         [-0.04540230591222698, 0.0, 1.954597694087773])
    same("17.14", 'gelu_derivative([-2, 0, 2], approximate="tanh")', A.gelu_derivative([-2.0, 0.0, 2.0], approximate="tanh"),
         [-0.08609925660415786, 0.5, 1.0860992566041579])
with checked("17.14"):
    A = mylearn.nn.activations
    try:
        A.gelu([1.0], approximate="exact")
        print('❌ Ex 17.14 : gelu(z, approximate="exact") devrait lever ValueError (les deux noms valides sont "none" et "tanh").')
    except ValueError as exc_14:
        print(f'✅ Ex 17.14 : gelu(z, approximate="exact") lève ValueError : {exc_14}')
    pair_14 = A.get_activation("ReLU")
    verdict("17.14", tuple(pair_14) == (A.relu, A.relu_derivative), 'get_activation("ReLU") rend (relu, relu_derivative), les fonctions du module.',
            f'get_activation("ReLU") rend {pair_14!r} : attendu (relu, relu_derivative), les fonctions du module elles-mêmes.')
    verdict("17.14", A.get_activation("swish")[0] is A.silu, '"swish" est un alias de silu.', 'get_activation("swish") doit rendre silu.')
    try:
        A.get_activation("sigmoïde")
        print('❌ Ex 17.14 : get_activation("sigmoïde") devrait lever ValueError.')
    except ValueError as exc_14:
        print(f"✅ Ex 17.14 : un nom inconnu lève ValueError : {exc_14}")
if written(lambda: mylearn.nn.activations.gelu([1.0])):
    run_mylearn_tests("activations", keyword="(elu or silu or gelu or get_activation) and not relu")
else:
    print("⏳ Ex 17.14 : les tests attendent tes fonctions.")'''

PART_A = Part("A", "Les activations élément par élément",
              "Prédire ce que devient un MLP sans activation (17.11), puis écrire, avec leurs dérivées, les dix "
              "activations de ta librairie : les fonctions linéaires par morceaux (17.12), les fonctions lisses de "
              "base, sans débordement (17.13), puis ELU, SiLU, GELU et le registre `get_activation` (17.14). La "
              "cellule suivante définit les outils des parties A à D. Ne recopie jamais un nombre à la main : "
              "chaque réponse est calculée par ton code, et pas arrondie.",
              given=PART_A_GIVEN, exercises=[
    Ex("17.11", "🔮", 1, 15, "Un MLP sans activation sur les lunes : quelle frontière ?",
       "prédire ce que vaut un réseau profond sans activation, face à la régression logistique et au même réseau avec ReLU.",
       "✏️ 17.1 · ch. 15 (`MLPClassifier`, les deux lunes) · fiche §17.2", thread="synth", tracks="R, C",
       hypothesis=True,
       body=r"""Sur les deux lunes du ch. 15 (500 points, bruit 0,2 ; 40 % pour le test), on compare trois modèles : une régression logistique (3 paramètres), un `MLPClassifier` à deux couches cachées de 100 neurones **sans activation** (`activation="identity"`, plus de 10 000 paramètres) et le même réseau avec ReLU. Les deux MLP sont entraînés par `solver="lbfgs"` (une méthode de second ordre, bien adaptée aux petits datasets), avec `random_state=0`.

Prédis, **avant** d'exécuter quoi que ce soit :
a) `beats_lr_11` : le MLP sans activation a-t-il une accuracy de test **supérieure de plus de 2 points** (0,02) à celle de la régression logistique ? (`True` ou `False`) ;
b) `agree_11` : sur quelle part des points de test le MLP sans activation et la régression logistique prédisent-ils la même classe ? (A) moins de 80 % ; (B) de 80 % à 95 % ; (C) plus de 95 % ;
c) `relu_11` : l'accuracy de test du MLP avec ReLU : (A) moins de 0,85 ; (B) de 0,85 à 0,90 ; (C) plus de 0,90.

Puis exécute l'expérience : elle entraîne les trois modèles, affiche leurs accuracies, leur accord, la direction des poids « effondrés » du MLP sans activation et les trois frontières de décision, et vérifie tes prédictions.""",
       given=GIVEN_11,
       todo=r'''beats_lr_11 = ...   # a) True or False
agree_11 = ...      # b) "A", "B" or "C"
relu_11 = ...       # c) "A", "B" or "C"''',
       solution=r'''beats_lr_11, agree_11, relu_11 = False, "C", "C"   # the answers, for the record''',
       after=[("code", guarded(EXPERIMENT_11, ["beats_lr_11", "agree_11", "relu_11"],
                               "⏳ Ex 17.11 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : combien de nombres suffisent à décrire la frontière du MLP sans activation, "
                     "malgré ses milliers de paramètres ? Compare les deux directions de poids affichées : d'où "
                     "peuvent venir les écarts entre ce MLP et la régression logistique ? Qu'est-ce que la ReLU a "
                     "changé à la frontière ?")],
       record=r'''wb.record("17.11a", False, mistakes={"sans activation, le MLP calcule une application affine de ses deux entrées : sa frontière peut-elle être autre chose qu'une droite ?": True})
wb.record("17.11b", "C", mistakes={"deux modèles linéaires entraînés sur les mêmes points : leurs droites sont-elles très différentes ?": "A",
                                   "regarde la direction des poids effondrés et celle de la régression logistique": "B"})
wb.record("17.11c", "C", mistakes={"une frontière en ligne brisée peut suivre les lunes : regarde la troisième figure": "A",
                                   "la ReLU permet une frontière courbe (en ligne brisée) : bien mieux qu'une droite": "B"})''',
       note="Le MLP sans activation calcule une application affine de ses deux entrées, suivie de la sigmoïde de sortie : "
            "c'est une régression logistique, et sa frontière est une droite, décrite par trois nombres (deux poids et "
            "un biais), quels que soient ses 10 000 paramètres. Ici, son accuracy de test (0,860) est même un peu "
            "**inférieure** à celle de la régression logistique (0,875), et les deux modèles donnent la même classe "
            "sur 97,5 % des points de test. Les petits écarts viennent de l'entraînement, pas du modèle : la pénalité "
            "L2 de `MLPClassifier` (`alpha`) porte sur les 10 000 poids et non sur les deux poids effondrés, et "
            "`lbfgs` s'arrête quand la loss ne baisse plus. Avec ReLU, la frontière devient une ligne brisée qui suit "
            "les lunes : 0,955 d'accuracy. Sur quatre autres tirages des lunes, le MLP sans activation reste à moins "
            "de 1,5 point de la régression logistique, et le MLP ReLU au-dessus de 0,91."),

    Ex("17.12", "🔨", 1, 15, "identity, step, relu, leaky_relu et leurs dérivées",
       "écrire les activations linéaires par morceaux et leurs dérivées, avec la convention de PyTorch aux coudes.",
       "✏️ 17.2 · fiche §17.3.1, §17.4, §17.5 (encadré sur la dérivée en un point anguleux)", thread="synth", tracks="R, M, C",
       mylearn="nn/activations.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `identity`, `step`, `relu` et `leaky_relu`, et leurs quatre dérivées `..._derivative`. **Contrat** propre à ces fonctions : `step(z, threshold, low, high)` rend `high` dès que $z \ge$ `threshold` et `low` strictement en dessous ; sa dérivée est nulle partout ; $\mathrm{ReLU}'(0) = 0$ et $\mathrm{leaky}'(0) =$ `negative_slope`, comme PyTorch ; la pente par défaut de `leaky_relu` est 0,01.

La vérification essaie les exemples de la docstring, compare `relu` et `step` à NumPy sur une grille de $-6$ à $6$, vérifie que `identity` rend une copie, puis lance les tests de l'exercice (`-k "identity or step or relu"`).

Dans tes notes : pourquoi `identity` ne peut-elle pas se contenter de `return np.asarray(z, dtype=float)` ? Quel test de la batterie le montre ?""",
       check=RELOAD + CHECK_12, solution=ref_check(CHECK_12, "17.12"),
       note="`np.asarray(z, dtype=float)` ne copie pas un tableau qui est déjà en `float64` : la fonction rendrait **le** "
            "tableau de l'appelant, et un `out[...] = ...` ultérieur modifierait ses données. Le test "
            "`test_elementwise_does_not_modify_or_share_its_input[identity]` le vérifie avec `np.shares_memory`. "
            "`np.array(z, dtype=float)` copie toujours ; pour les autres fonctions, le calcul crée déjà un nouveau "
            "tableau, mais pas pour l'identité."),

    Ex("17.13", "🔨", 2, 25, "Sigmoïde, tanh et softplus sans débordement",
       "écrire les fonctions lisses de base et leurs dérivées sans aucun débordement, même pour des entrées de ±1 000.",
       "Ex 17.12 · fiche §17.6, section sur le calcul sans débordement · ch. 13 (sigmoïde stable)", thread="synth",
       tracks="R, M, C", mylearn="nn/activations.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `sigmoid`, `tanh` et `softplus`, et leurs trois dérivées. **Contrat** propre à ces fonctions : `sigmoid(-1000)` vaut 0 et `sigmoid(1000)` vaut 1, sans avertissement, et les petites probabilités sont gardées (`sigmoid(-700)` est comparé à SciPy avec une tolérance **relative**) ; `softplus(z, beta, threshold)` rend `z` lui-même seulement quand `beta * z` est **strictement** supérieur à `threshold` (convention de PyTorch), et $\log(1 + e^{\beta z})/\beta$ sinon, seuil compris ; sa dérivée vaut exactement 1 au-dessus du seuil.

La vérification essaie les exemples de la docstring, appelle chaque fonction sur des valeurs extrêmes en guettant les avertissements, compare `sigmoid(-700)` et `softplus` à SciPy et NumPy, puis lance les tests (`-k "sigmoid or tanh or softplus"`).

Dans tes notes : quelle est la plus petite valeur strictement positive que rend ta sigmoïde, et pour quel $z$ environ ? Pourquoi la forme $1 - 1/(1 + e^{z})$, mathématiquement égale à $\sigma(z)$, ne la garderait-elle pas ?""",
       check=RELOAD + CHECK_13, solution=ref_check(CHECK_13, "17.13"),
       note="Avec la forme $e^{z}/(1 + e^{z})$ pour $z < 0$, la sigmoïde suit $e^{z}$ jusqu'au plus petit nombre "
            "représentable : environ $5 \\times 10^{-324}$ (un nombre « sous-normal ») en $z = -745$, puis 0 dès "
            "$z = -746$. La forme $1 - 1/(1 + e^{z})$ soustrait deux nombres presque égaux à 1 : dès que $e^{z}$ "
            "passe sous $10^{-16}$ environ (vers $z = -37$), $1/(1 + e^{z})$ s'arrondit à 1 et la différence vaut "
            "exactement 0. On "
            "perd alors toutes les petites probabilités, et un `log` de la sortie vaut `-inf`."),

    Ex("17.14", "🔨", 2, 25, "ELU, SiLU, GELU et le registre get_activation",
       "écrire les ReLU lissées, GELU exacte et approchée, et le registre qui associe un nom à un couple (f, f′).",
       "Ex 17.13 · fiche §17.6 (encadré sur Φ et erf), §17.7", thread="synth", tracks="R, M, C",
       mylearn="nn/activations.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `elu`, `silu` et `gelu`, leurs dérivées, et `get_activation`. **Contrat** propre à ces fonctions : $\mathrm{ELU}'(0) = \alpha$ (PyTorch) ; `gelu` et `gelu_derivative` acceptent `approximate="none"` (la formule exacte, avec `math.erf` vectorisé) ou `approximate="tanh"` (l'approximation de la fiche), et lèvent `ValueError` pour toute autre valeur, `None` compris ; `get_activation(name)` ignore la casse, connaît l'alias `"swish"` de `"silu"`, rend un couple fait des fonctions **du module elles-mêmes** (`get_activation("relu")` est `(relu, relu_derivative)`), et lève `ValueError` pour un nom inconnu, avec un message qui donne les dix noms valides.

La vérification essaie les exemples de la docstring, compare `gelu` à $z\,\Phi(z)$ calculé avec `scipy.special.erf`, essaie des noms et une approximation inconnus, puis lance les tests (`-k "(elu or silu or gelu or get_activation) and not relu"`).

Dans tes notes : pourquoi `get_activation("softmax")` doit-il lever une erreur, alors que ta librairie a une fonction `softmax` ?""",
       check=RELOAD + CHECK_14, solution=ref_check(CHECK_14, "17.14"),
       note="Le registre rend des couples `(f, f′)` de fonctions **élément par élément**, que `mlp_forward` applique à "
            "chaque case et que `mlp_backward` multipliera case par case (ch. 18). La softmax n'en est pas une : "
            "chaque sortie dépend de toute la ligne, et sa « dérivée » est une Jacobienne $(K, K)$ par exemple, pas "
            "un tableau de la forme de `z`. La mettre dans le registre ferait croire qu'on peut l'utiliser comme une "
            "activation cachée ; en sortie, elle se traite à part, avec la loss (ch. 18)."),
])

# ---- Part B ----------------------------------------------------------------
CHECK_15 = r'''with checked("17.15"):
    A = mylearn.nn.activations
    same("17.15", "softmax([1, 2, 3])", A.softmax([1.0, 2.0, 3.0]), special.softmax([1.0, 2.0, 3.0]))
    big_15, alarms_15 = quiet(A.softmax, [1001.0, 1002.0, 1003.0])
    same("17.15", "softmax([1001, 1002, 1003])", big_15, special.softmax([1.0, 2.0, 3.0]))
    verdict("17.15", not alarms_15, "softmax de logits de l'ordre de 1 000 : aucun avertissement.",
            f"softmax([1001, 1002, 1003]) émet « {alarms_15[0] if alarms_15 else ''} » : soustrais d'abord le maximum.")
    logp_15, alarms_15 = quiet(A.log_softmax, [0.0, 10000.0])
    same("17.15", "log_softmax([0, 10000])", logp_15, [-10000.0, 0.0])
    verdict("17.15", not alarms_15, "log_softmax([0, 10000]) : aucun avertissement.",
            f"log_softmax([0, 10000]) émet « {alarms_15[0] if alarms_15 else ''} » : z − logsumexp(z), jamais log(softmax(z)).")
with checked("17.15"):
    A = mylearn.nn.activations
    batch_15 = np.random.default_rng(1715).normal(scale=3.0, size=(4, 6))
    same("17.15", "softmax d'un batch (4, 6), axe des classes -1", A.softmax(batch_15), special.softmax(batch_15, axis=-1))
    same("17.15", "softmax du même batch avec axis=0", A.softmax(batch_15, axis=0), special.softmax(batch_15, axis=0))
    same("17.15", "log_softmax du batch", A.log_softmax(batch_15), special.log_softmax(batch_15, axis=-1))
    try:
        A.softmax(3.0)
        print("❌ Ex 17.15 : softmax(3.0) devrait lever ValueError (un nombre seul n'a pas d'axe de classes).")
    except ValueError as exc_15:
        print(f"✅ Ex 17.15 : softmax d'un nombre seul lève ValueError : {exc_15}")
if written(lambda: mylearn.nn.activations.softmax([1.0, 2.0])):
    run_mylearn_tests("activations", keyword="softmax and not derivative")
else:
    print("⏳ Ex 17.15 : les tests attendent tes fonctions.")'''

CHECK_16 = r'''def numeric_jacobian_16(z, h=1e-6):
    """Central differences of scipy.special.softmax along the last axis: entry [..., i, j] = dp_i / dz_j."""
    K = z.shape[-1]
    J = np.empty(z.shape[:-1] + (K, K))
    for j in range(K):
        step = np.zeros(K)
        step[j] = h
        J[..., :, j] = (special.softmax(z + step, axis=-1) - special.softmax(z - step, axis=-1)) / (2 * h)
    return J


with checked("17.16"):
    A = mylearn.nn.activations
    same("17.16", "softmax_derivative([0, 0])", A.softmax_derivative([0.0, 0.0]), [[0.25, -0.25], [-0.25, 0.25]])
    z_16 = np.random.default_rng(1716).normal(scale=2.0, size=(5, 4))
    J_16 = np.asarray(A.softmax_derivative(z_16), dtype=float)
    verdict("17.16", J_16.shape == (5, 4, 4), "5 exemples de 4 logits : un tableau (5, 4, 4), une Jacobienne par exemple.",
            f"attendu la forme (5, 4, 4), reçu {J_16.shape}.")
    if J_16.shape == (5, 4, 4):
        same("17.16", "Jacobiennes du batch, comme les différences centrées de scipy.special.softmax", J_16,
             numeric_jacobian_16(z_16), rtol=1e-6, atol=1e-8)
        verdict("17.16", np.allclose(J_16, np.swapaxes(J_16, 1, 2)) and np.allclose(J_16.sum(axis=2), 0.0, atol=1e-12),
                "chaque Jacobienne est symétrique, et ses lignes somment à 0.",
                "une Jacobienne de la softmax est symétrique, et ses lignes somment à 0 : vérifie les axes de ton produit.")
if written(lambda: mylearn.nn.activations.softmax_derivative([0.0, 1.0])):
    run_mylearn_tests("activations", keyword="softmax_derivative")
else:
    print("⏳ Ex 17.16 : les tests attendent ta fonction.")'''

GIVEN_17 = r'''rng_17 = np.random.default_rng(1717)
X_17 = rng_17.normal(size=(8, 3))
PARAMS_17 = [(rng_17.normal(size=(3, 4)), rng_17.normal(size=4)), (rng_17.normal(size=(4, 3)), rng_17.normal(size=3))]
BIG_17 = [PARAMS_17[0], (400.0 * PARAMS_17[1][0], 400.0 * PARAMS_17[1][1])]      # logits of order 1000
Y_17 = np.array([0, 1, 2, 0, 1, 2, 0, 1])


def reference_17(X, y, params):
    """What forward_17 must return, computed with SciPy (the oracle of the diagnostics)."""
    (W1, b1), (W2, b2) = params
    z1 = X @ W1 + b1
    logits = np.tanh(z1) @ W2 + b2
    return special.softmax(logits, axis=1), float(-np.mean(special.log_softmax(logits, axis=1)[np.arange(len(y)), y])), 1 - np.tanh(z1) ** 2'''

TODO_17 = r'''def forward_17(X, y, params):
    """Probabilities, mean cross-entropy and local slopes of a 2-layer tanh network (one row per sample).

    params = [(W1, b1), (W2, b2)], W of shape (n_in, n_out); y holds the class of each row (0 to K - 1).
    Returns (proba of shape (n, K), loss as a float, slope = tanh'(z1) of shape (n, n_hidden)).
    """
    (W1, b1), (W2, b2) = params
    z1 = X @ W1 + b1
    h = np.tanh(z1)
    slope = 1.0 - np.tanh(h) ** 2                     # tanh'(z1), for the backward pass of chapter 18
    logits = h @ W2 + b2
    e = np.exp(logits)
    proba = e / e.sum(axis=0, keepdims=True)
    loss = -np.mean(np.log(proba[np.arange(len(y)), y]))
    return proba, loss, slope'''

SOLUTION_17 = r'''def forward_17(X, y, params):
    """Probabilities, mean cross-entropy and local slopes of a 2-layer tanh network (one row per sample)."""
    (W1, b1), (W2, b2) = params
    z1 = X @ W1 + b1
    h = np.tanh(z1)
    slope = 1.0 - np.tanh(z1) ** 2                    # fix 4: the derivative is taken at the pre-activation z1
    logits = h @ W2 + b2
    shifted = logits - logits.max(axis=1, keepdims=True)   # fix 1: no overflow
    log_proba = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))   # fix 2 (axis=1) and fix 3 (log-sum-exp)
    proba = np.exp(log_proba)
    loss = -np.mean(log_proba[np.arange(len(y)), y])
    return proba, float(loss), slope

'''

CHECK_17 = r'''def diagnose_17(forward):
    """The four points of 17.17, in the order in which the bugs hide each other; returns the first failing point (or 0)."""
    points = [
        (1, "aucun débordement (« overflow ») de l'exponentielle pour des logits de l'ordre de 1 000"),
        (2, "chaque LIGNE de probabilités somme à 1 et vaut la softmax de SciPy"),
        (3, "une loss finie, égale à la cross-entropy de SciPy, quand la bonne classe a une probabilité qui s'arrondit à 0"),
        (4, "slope vaut tanh'(z1), la dérivée prise en la pré-activation"),
    ]
    for k, what in points:
        try:
            if k == 1:
                (proba, _, _), alarms = quiet(forward, X_17, Y_17, BIG_17)
                ok = not [a for a in alarms if "overflow" in a]      # nan from the axis is point 2, inf from the log point 3
            elif k == 2:
                (proba, _, _), _ = quiet(forward, X_17, Y_17, BIG_17)      # its warnings belong to the other points
                want, _, _ = reference_17(X_17, Y_17, BIG_17)
                ok = np.shape(proba) == want.shape and bool(np.allclose(proba, want, atol=1e-12))
            elif k == 3:
                logits = np.tanh(X_17 @ BIG_17[0][0] + BIG_17[0][1]) @ BIG_17[1][0] + BIG_17[1][1]
                y_worst = np.argmin(logits, axis=1)               # the least likely class of each row
                (_, loss, _), alarms = quiet(forward, X_17, y_worst, BIG_17)
                _, want, _ = reference_17(X_17, y_worst, BIG_17)
                ok = not alarms and np.isfinite(float(loss)) and bool(np.isclose(float(loss), want, rtol=1e-9))
            else:
                (_, _, slope), _ = quiet(forward, X_17, Y_17, PARAMS_17)
                _, _, want = reference_17(X_17, Y_17, PARAMS_17)
                ok = np.shape(slope) == want.shape and bool(np.allclose(slope, want, atol=1e-12))
        except (NotImplementedError, MylearnMissingError):
            raise
        except Exception as exc:                                # a crash counts as a failed point
            ok, what = False, f"{what} (ton code lève {type(exc).__name__} : {str(exc)[:120]})"
        verdict("17.17", ok, f"point {k} : {what}.", f"point {k} : attendu : {what}.")
        if not ok:
            return k
    return 0


with wb.attempt("17.17"):
    first_17 = diagnose_17(forward_17)
    if first_17:
        print(f"⏳ Ex 17.17 : corrige d'abord le point {first_17} ; les points suivants seront vérifiés ensuite.")
    else:
        print("✅ Ex 17.17 : les quatre points sont corrigés.")'''

PART_B = Part("B", "La softmax",
              "La softmax stable et son logarithme (17.15), sa Jacobienne pour un batch (17.16), puis une chasse aux "
              "`nan` dans la sortie d'un petit réseau (17.17).",
              exercises=[
    Ex("17.15", "🔨", 2, 25, "Softmax et log_softmax stables (astuce log-sum-exp)",
       "écrire une softmax et une log-softmax qui ne débordent jamais, le long de n'importe quel axe.",
       "Ex 17.13 · fiche §17.8, section sur le calcul sans débordement · ch. 13 (`logsumexp`)", thread="synth",
       tracks="R, M, C", mylearn="nn/activations.py",
       body=mylearn_howto() + r"""

Écris `softmax` et `log_softmax`. **Contrat** (les docstrings, plus ces précisions que vérifient les tests) : les deux fonctions rendent un nouveau tableau `float64` de la forme de `z`, laissent `z` intact, et lèvent `ValueError` pour un nombre seul (0 dimension) ; `axis` désigne l'axe des classes (par défaut le dernier), et n'importe quel axe d'un tableau 1-D, 2-D ou 3-D doit marcher ; les probabilités sont positives et somment à 1 le long de `axis` ; aucun avertissement ni `nan` pour des logits de l'ordre de $\pm 1\,000$ (`softmax`) ou de $10^4$ (`log_softmax`, qui ne calcule jamais `log(0)`).

La vérification essaie les exemples de la docstring et des logits énormes, compare tes fonctions à `scipy.special.softmax` et `log_softmax` sur un batch le long des deux axes, essaie un nombre seul, puis lance les tests (`-k "softmax and not derivative"`).

Dans tes notes : la softmax de `[-1000, -1001, -1003]` sans soustraction du maximum, que donne-t-elle, et pourquoi ? Le soustraire change-t-il le résultat mathématique ?""",
       check=RELOAD + CHECK_15, solution=ref_check(CHECK_15, "17.15"),
       note="Sans soustraction, les trois exponentielles s'arrondissent à 0 (elles sont sous $e^{-745}$), et le quotient "
            "$0/0$ vaut `nan`, avec un avertissement « invalid value ». Soustraire le maximum ne change rien "
            "mathématiquement ($\\mathrm{softmax}(\\mathbf{z} - m) = \\mathrm{softmax}(\\mathbf{z})$) : le plus grand terme "
            "devient $e^0 = 1$, la somme vaut au moins 1, et le résultat est $(0{,}7054 ;\\ 0{,}2595 ;\\ 0{,}0351)$, comme "
            "pour les logits $0, -1, -3$."),

    Ex("17.16", "🔨", 3, 30, "La Jacobienne de la softmax en NumPy",
       "écrire la Jacobienne de la softmax pour un exemple et pour un batch, sans boucle sur les exemples.",
       "Ex 17.15 · ∂ 17.7 · fiche §17.8 (encadré sur la Jacobienne)", thread="synth", tracks="M, C",
       mylearn="nn/activations.py",
       body=mylearn_howto() + r"""

Écris `softmax_derivative`. **Contrat** : pour des logits de forme `(K,)`, la fonction rend la matrice `(K, K)` de case `[i, j]` $= \partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$ ; pour un batch `(n, K)`, un tableau `(n, K, K)`, une matrice par exemple ; `ValueError` pour toute autre dimension ; aucun avertissement pour des logits de l'ordre de $\pm 1\,000$ ; `z` reste intact.

La vérification essaie les exemples de la docstring, compare ta Jacobienne à des différences centrées de `scipy.special.softmax` sur un batch de 5 exemples, vérifie la symétrie et les lignes de somme nulle, puis lance les tests (`-k softmax_derivative`).

Dans tes notes : combien de nombres contient le résultat pour un batch de 64 exemples et un vocabulaire de 50 000 tokens ? Pourquoi aucune bibliothèque ne forme ce tableau pendant l'entraînement ?""",
       check=RELOAD + CHECK_16, solution=ref_check(CHECK_16, "17.16"),
       note="$64 \\times 50\\,000^2 = 1{,}6 \\times 10^{11}$ nombres, soit 640 Go en `float32` : impossible. La "
            "rétropropagation n'a besoin que du produit d'un vecteur par la Jacobienne, $\\mathbf{g}\\mathbf{J} = "
            "\\mathbf{p} \\odot (\\mathbf{g} - (\\mathbf{g}\\cdot\\mathbf{p}))$, en $O(K)$ opérations par exemple (∂ 17.7) ; "
            "avec la cross-entropy, il se simplifie même en $\\mathbf{p} - \\mathbf{y}$."),

    Ex("17.17", "🐛", 2, 25, "Chasse au bug : nan, inf et mauvais axe",
       "trouver quatre erreurs dans la sortie d'un petit réseau, dont trois qui ne se voient que sur des valeurs extrêmes.",
       "Ex 17.13 · Ex 17.15 · fiche §17.6 (encadré sur les dérivées), section sur le calcul sans débordement",
       tracks="C",
       body=r"""Un collègue a écrit `forward_17`, la passe avant d'un petit réseau tanh à une couche cachée : elle rend les probabilités des classes, la cross-entropy moyenne du batch, et la pente $\tanh'(z_1)$ de chaque neurone caché, que la rétropropagation du ch. 18 utilisera. Sur ses données de test, tout semble marcher. Il y a pourtant **quatre** erreurs : trouve-les et corrige-les dans la cellule ci-dessous, sans changer la signature ni ce que rend la fonction. Tu peux utiliser les fonctions de ta librairie (`mylearn.nn.activations`).

La vérification contrôle quatre propriétés, dans l'ordre où les erreurs se masquent (un point ne se teste que si les précédents passent) : 1) aucun débordement de l'exponentielle pour des logits de l'ordre de 1 000 ; 2) chaque ligne de probabilités somme à 1 et vaut la softmax de SciPy ; 3) une loss finie et juste quand la bonne classe d'un exemple a une probabilité qui s'arrondit à 0 ; 4) la pente vaut $\tanh'(z_1)$. Cet exercice se corrige avec `05_solutions.md`.

Dans tes notes : pour chaque erreur, quel symptôme aurait-elle eu pendant un vrai entraînement ? Lesquelles un test sur des données « normales » ne voit-il pas ?""",
       given=GIVEN_17, todo=TODO_17, check=CHECK_17, solution=SOLUTION_17 + CHECK_17.replace('with wb.attempt("17.17"):\n', "if True:\n"),
       note="Les quatre erreurs : `np.exp(logits)` déborde pour des logits de l'ordre de 1 000 (`inf / inf = nan`) : "
            "soustraire le maximum de chaque ligne ; `sum(axis=0)` normalise les **colonnes** (les classes sur tout le "
            "batch) au lieu des lignes : `axis=1` ; `np.log(proba)` vaut `-inf` quand la probabilité de la bonne "
            "classe s'arrondit à 0, et la loss devient infinie : calculer `log_softmax` par log-sum-exp ; "
            "`1 - np.tanh(h) ** 2` prend la dérivée en la **sortie** $h = \\tanh(z_1)$ au lieu de la pré-activation : "
            "`1 - np.tanh(z1) ** 2` (ou `1 - h ** 2`). Sur des données ordinaires, seule l'erreur d'axe change les "
            "probabilités ; les erreurs 1 et 3 n'apparaissent qu'avec des logits extrêmes, et la 4 donne des pentes "
            "plausibles (toutes entre 0 et 1), donc des gradients faux sans aucun message."),
])

# ---- Part C ----------------------------------------------------------------
TODO_18 = r'''# 1) The gallery: a 2 x 5 grid, one panel per activation of get_activation, f in a solid line and f' dashed,
#    on z from -4 to 4, with the same axis limits as the figure of the sheet.
NAMES_18 = ["identity", "step", "relu", "leaky_relu", "elu", "softplus", "sigmoid", "tanh", "silu", "gelu"]
# your plotting code here, then plt.show()

# 2) Four values read on fine grids with your functions (do not round them)
zmin_silu_18 = ...      # a) the z where SiLU is smallest
min_gelu_18 = ...       # b) the smallest value of GELU
max_dgelu_18 = ...      # c) the largest value of the derivative of GELU
zmax_dsilu_18 = ...     # d) the z > 0 where the derivative of SiLU is largest'''

SOLUTION_18 = r'''A = mylearn.nn.activations
NAMES_18 = ["identity", "step", "relu", "leaky_relu", "elu", "softplus", "sigmoid", "tanh", "silu", "gelu"]
z_18 = np.linspace(-4, 4, 801)
fig_18, axes_18 = plt.subplots(2, 5, figsize=(14, 5.4), sharex=True)
for ax, name in zip(axes_18.ravel(), NAMES_18):
    f, df = A.get_activation(name)
    ax.plot(z_18, f(z_18), lw=2, label="f(z)")
    ax.plot(z_18, df(z_18), "--", lw=1.3, label="f′(z)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_title(name)
    ax.set_ylim(-1.5, 2.5)
axes_18[0, 0].legend()
plt.tight_layout()
plt.show()

fine_18 = np.linspace(-4, 4, 800_001)                 # a step of 1e-5
zmin_silu_18 = float(fine_18[np.argmin(A.silu(fine_18))])
min_gelu_18 = float(np.min(A.gelu(fine_18)))
max_dgelu_18 = float(np.max(A.gelu_derivative(fine_18)))
positive_18 = fine_18[fine_18 > 0]
zmax_dsilu_18 = float(positive_18[np.argmax(A.silu_derivative(positive_18))])
print(zmin_silu_18, min_gelu_18, max_dgelu_18, zmax_dsilu_18)'''

CHECK_18 = r'''wb.check("17.18a", zmin_silu_18)
wb.check("17.18b", min_gelu_18)
wb.check("17.18c", max_dgelu_18)
wb.check("17.18d", zmax_dsilu_18)'''

GIVEN_19 = r'''SCORES_19 = np.array([3.0, 1.5, 1.0, 0.0, -1.0])     # five classes, numbered 0 to 4
TEMPERATURES_19 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]'''

EXPERIMENT_19 = r'''with wb.attempt("17.19"):
    A = mylearn.nn.activations
    table_19 = {T: np.asarray(A.softmax(SCORES_19 / T), dtype=float) for T in TEMPERATURES_19 + [5.0, -1.0]}
    entropy_19 = {T: float(-(p * np.log2(p)).sum()) for T, p in table_19.items()}
    for T in TEMPERATURES_19 + [5.0, -1.0]:
        print(f"T = {T:5.2f} : p = {np.round(table_19[T], 3).tolist()} · entropy {entropy_19[T]:.3f} bits "
              f"(maximum log2(5) = {np.log2(5):.3f})")
    fig_19, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 3.6))
    for T in [0.5, 1.0, 5.0]:
        ax_a.plot(range(5), table_19[T], "o-", label=f"T = {T}")
    ax_a.set_xticks(range(5))
    ax_a.set_xlabel("classe")
    ax_a.set_ylabel("probabilité")
    ax_a.legend()
    ax_b.plot(TEMPERATURES_19, [entropy_19[T] for T in TEMPERATURES_19], "o-")
    ax_b.axhline(np.log2(5) / 2, color="grey", ls="--", label="la moitié du maximum")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("température T (échelle log)")
    ax_b.set_ylabel("entropie (bits)")
    ax_b.legend()
    plt.tight_layout()
    plt.show()
wb.check("17.19a", sharp_19)
wb.check("17.19b", flat_19)
wb.check("17.19c", half_19)
wb.check("17.19d", negative_19)'''

GIVEN_20 = r'''def split_20(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=1720, stratify=y)


DATA_20 = {"lunes": split_20(*wb.synth.make_moons(n=600, noise=0.25, seed=1720)),
           "spirales": split_20(*wb.synth.make_spirals(n=600, noise=0.2, turns=1.5, seed=1720))}
ACTIVATIONS_20 = ["identity", "logistic", "tanh", "relu"]
for name, (Xtr, Xte, ytr, yte) in DATA_20.items():
    print(name, "train", Xtr.shape, "test", Xte.shape)'''

TODO_20 = r'''def mean_accuracy_20(dataset, activation):
    """Mean test accuracy over random_state = 0, 1, 2, 3, 4 of the protocol of the statement."""
    raise NotImplementedError("mean_accuracy_20")


best_moons_20 = ...      # b) the best activation on the moons, e.g. "identity"
best_spirals_20 = ...    # c) the best activation on the spirals'''

SOLUTION_20 = r'''def mean_accuracy_20(dataset, activation):
    """Mean test accuracy over random_state = 0, 1, 2, 3, 4 of the protocol of the statement."""
    Xtr, Xte, ytr, yte = DATA_20[dataset]
    scores = []
    for seed in range(5):
        model = MLPClassifier((16,), activation=activation, solver="lbfgs", max_iter=3000, random_state=seed)
        scores.append(fit_quietly(model, Xtr, ytr).score(Xte, yte))
    return float(np.mean(scores))


best_moons_20, best_spirals_20 = "relu", "tanh"
'''

CHECK_20 = r'''results_20 = {}
with checked("17.20"):
    for dataset_20 in DATA_20:
        results_20[dataset_20] = {a: mean_accuracy_20(dataset_20, a) for a in ACTIVATIONS_20}
        print(f"{dataset_20:9s}", " · ".join(f"{a} {v:.3f}" for a, v in results_20[dataset_20].items()))
if len(results_20) == 2:
    wb.check("17.20a", results_20["spirales"]["identity"], computed=True)
    wb.check("17.20d", results_20["spirales"]["relu"] - results_20["spirales"]["tanh"], computed=True)
with wb.attempt("17.20"):
    wb.check("17.20b", best_moons_20)
    wb.check("17.20c", best_spirals_20)'''

GIVEN_21 = r'''rng_21 = np.random.default_rng(1721)
SPECS_21 = {"P": (0.0, 1.0), "Q": (4.0, 1.0), "R": (0.0, 4.0), "S": (-1.0, 0.3)}       # mean, standard deviation
Z_21 = {name: rng_21.normal(m, s, size=(1000, 5)) for name, (m, s) in SPECS_21.items()}  # 1000 samples, 5 neurons
fig_21, axes_21 = plt.subplots(1, 4, figsize=(14, 3.0), sharex=True)
for ax, (name, z) in zip(axes_21, Z_21.items()):
    ax.hist(z.ravel(), bins=60, range=(-14, 14), color="#2a78d6")
    ax.axvspan(-14, -4.6, color="#f4a3a2", alpha=0.3)
    ax.axvspan(4.6, 14, color="#f4a3a2", alpha=0.3)
    ax.set_title(f"couche {name}")
    ax.set_xlabel("pré-activation z")
fig_21.suptitle("Pré-activations de quatre couches sigmoïdes (en rose : |z| > 4,6, où σ′(z) < 0,01)", fontsize=10)
plt.tight_layout()
plt.show()'''

TODO_21 = r'''most_21 = ...       # a) the letter of the most saturated layer, read on the histograms
linear_21 = ...     # b) the letter of the layer that works in the almost linear part of the sigmoid


def saturation_21(z):
    """(mean of sigmoid'(z), fraction of the values where sigmoid'(z) < 0.01), with YOUR sigmoid_derivative."""
    raise NotImplementedError("saturation_21")'''

SOLUTION_21 = r'''most_21, linear_21 = "Q", "S"


def saturation_21(z):
    """(mean of sigmoid'(z), fraction of the values where sigmoid'(z) < 0.01), with YOUR sigmoid_derivative."""
    slope = np.asarray(mylearn.nn.activations.sigmoid_derivative(z), dtype=float)
    return float(slope.mean()), float(np.mean(slope < 0.01))
'''

CHECK_21 = r'''with wb.attempt("17.21"):
    wb.check("17.21a", most_21)
    wb.check("17.21b", linear_21)
with checked("17.21"):
    stats_21 = {name: saturation_21(z) for name, z in Z_21.items()}
    for name, (mean_slope, frac) in stats_21.items():
        print(f"couche {name} : pente moyenne {mean_slope:.4f} · part des pentes < 0,01 : {frac:.3f}")
    wb.check("17.21c", [stats_21[name][0] for name in "PQRS"], computed=True)
    wb.check("17.21d", stats_21["R"][1], computed=True)'''

PART_C = Part("C", "Comparer les activations",
              "Refaire la galerie avec ton code et y mesurer quatre valeurs (17.18), prédire l'effet d'une température "
              "(17.19), comparer les quatre activations de scikit-learn sur deux problèmes (17.20), puis lire des "
              "histogrammes de pré-activations (17.21).",
              exercises=[
    Ex("17.18", "🎨", 2, 25, "Reproduire la galerie de la figure 17.22, dérivées comprises",
       "tracer les dix activations de ta librairie avec leurs dérivées, et lire sur des grilles fines quatre points remarquables.",
       "Ex 17.14 · fiche §17.7", tracks="C",
       body=r"""Refais la galerie de la fiche (§17.7) avec **ta** librairie : une grille de 2 × 5 graphiques, un par nom de `NAMES_18`, avec `f` en trait plein et `f′` en tirets (obtenus par `get_activation`), pour $z$ de $-4$ à $4$ et une ordonnée de $-1{,}5$ à $2{,}5$. Puis, avec tes fonctions, calcule sur des grilles **fines** (un pas de $10^{-4}$ ou moins) :
a) `zmin_silu_18` : la valeur de $z$ où SiLU est la plus petite ;
b) `min_gelu_18` : la plus petite valeur de GELU ;
c) `max_dgelu_18` : la plus grande valeur de la dérivée de GELU ;
d) `zmax_dsilu_18` : la valeur de $z > 0$ où la dérivée de SiLU est la plus grande.

Garde les valeurs telles quelles : la vérification arrondit elle-même (2 décimales pour a) et d), 3 pour b) et c)).

Dans tes notes : quelles fonctions de la galerie ont une dérivée qui dépasse 1 ? Qu'est-ce que cela change pour le produit des dérivées de la rétropropagation, par rapport à la sigmoïde ?""",
       todo=TODO_18, check=CHECK_18, solution=SOLUTION_18,
       record=r'''wb.record("17.18a", zmin_silu_18, decimals=2, mistakes={"c'est la valeur minimale de SiLU ; on demande OÙ elle est atteinte": -0.28,
                                                            "c'est l'abscisse du minimum de GELU": -0.75})
wb.record("17.18b", min_gelu_18, decimals=3, mistakes={"c'est le minimum de SiLU ; on demande celui de GELU": -0.278,
                                                       "c'est l'abscisse du minimum ; on demande la valeur": -0.752})
wb.record("17.18c", max_dgelu_18, decimals=3, mistakes={"c'est le maximum de la dérivée de SiLU ; on demande celui de GELU": 1.100,
                                                        "c'est l'abscisse du maximum ; on demande la valeur": 1.414})
wb.record("17.18d", zmax_dsilu_18, decimals=2, mistakes={"c'est la valeur maximale de la dérivée ; on demande OÙ elle est atteinte": 1.10,
                                                         "c'est l'abscisse du maximum de la dérivée de GELU": 1.41})''',
       note="SiLU atteint son minimum $-0{,}278$ en $z \\approx -1{,}278$ ; GELU, moins creusée, descend à $-0{,}170$ en "
            "$z \\approx -0{,}752$. Leurs dérivées dépassent 1 : celle de GELU culmine à $1{,}129$ en $z = \\sqrt{2}$, "
            "celle de SiLU à $1{,}100$ en $z \\approx 2{,}40$. Avec une dérivée qui atteint 1 et la dépasse un peu, un "
            "gradient qui traverse des neurones « actifs » n'est pas réduit par leurs pentes, alors que celle d'une "
            "sigmoïde apporte un facteur d'au plus 1/4 à chaque couche."),

    Ex("17.19", "🔮", 2, 15, "Softmax et température : prédire la distribution",
       "prédire l'effet d'une température sur une distribution de probabilités, chiffres à l'appui.",
       "Ex 17.15 · ch. 6 (entropie) · fiche §17.8 (la température)", tracks="M, C", hypothesis=True,
       body=r"""Un classifieur à cinq classes, numérotées de 0 à 4, sort les scores `SCORES_19` $= (3 ;\ 1{,}5 ;\ 1 ;\ 0 ;\ -1)$. On calcule $\mathrm{softmax}(\mathbf{z}/T)$ avec ta `softmax`, pour plusieurs températures $T$, et l'entropie de chaque distribution, en bits (ch. 6 ; son maximum, pour cinq classes, vaut $\log_2 5 \approx 2{,}32$).

Prédis, **avant** d'exécuter quoi que ce soit :
a) `sharp_19` : à $T = 0{,}5$, la classe en tête a-t-elle une probabilité supérieure à 0,9 ? (`True` ou `False`) ;
b) `flat_19` : à $T = 5$, la classe la moins probable a-t-elle une probabilité supérieure à 0,1 ? (`True` ou `False`) ;
c) `half_19` : parmi les températures (A) 0,25 ; (B) 0,5 ; (C) 1 ; (D) 2 ; (E) 4 ; (F) 8, la première dont l'entropie dépasse la moitié du maximum (la lettre) ;
d) `negative_19` : avec une température **négative**, $T = -1$, le numéro de la classe en tête (un entier de 0 à 4).

Puis exécute l'expérience : elle affiche les distributions et leurs entropies, trace l'entropie en fonction de $T$, et vérifie tes prédictions.""",
       given=GIVEN_19,
       todo=r'''sharp_19 = ...      # a) True or False
flat_19 = ...       # b) True or False
half_19 = ...       # c) "A" to "F"
negative_19 = ...   # d) an integer from 0 to 4''',
       solution=r'''sharp_19, flat_19, half_19, negative_19 = True, True, "C", 4   # the answers, for the record''',
       after=[("code", guarded(EXPERIMENT_19, ["sharp_19", "flat_19", "half_19", "negative_19"],
                               "⏳ Ex 17.19 : écris d'abord tes quatre prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : comment l'entropie évolue-t-elle quand $T$ grandit, et vers quelle valeur "
                     "tend-elle ? Que fait une température négative à l'ordre des classes ? Un modèle de langage "
                     "« créatif » utilise-t-il une petite ou une grande température ?")],
       record=r'''wb.record("17.19a", True, mistakes={"à T = 0,5, les écarts de scores doublent : l'écart de 1,5 entre les deux premières classes devient 3 ; que vaut e³ ?": False})
wb.record("17.19b", True, mistakes={"à T = 5, les écarts sont divisés par 5 : les cinq probabilités se rapprochent de 0,2": False})
wb.record("17.19c", "C", mistakes={"à T = 0,25, une classe prend presque toute la probabilité : l'entropie est proche de 0": "A",
                                   "à T = 0,5, la classe en tête garde plus de 90 % : l'entropie reste faible": "B",
                                   "calcule aussi l'entropie à T = 1 : dépasse-t-elle déjà la moitié du maximum ?": "D",
                                   "l'entropie dépasse la moitié du maximum bien avant T = 4": "E", "regarde les températures les plus basses d'abord": "F"})
wb.record("17.19d", 4, mistakes={"diviser par un nombre négatif inverse l'ordre des scores : quelle classe a le PLUS PETIT score ?": 0,
                                 "le plus petit score, −1, est celui de la classe 4": 3, "la classe 1 a le deuxième plus grand score": 1,
                                 "la classe 2 est au milieu": 2}, fractional="un numéro de classe est un entier de 0 à 4")''',
       note="À $T = 0{,}5$, la classe 0 prend 0,934 de la probabilité ; à $T = 5$, la moins probable garde 0,132 "
            "(l'uniforme donnerait 0,2). L'entropie croît avec $T$ : 0,03 bit à $T = 0{,}25$, 0,42 à $T = 0{,}5$, 1,35 à "
            "$T = 1$ (au-dessus de la moitié du maximum, 1,16), 2,30 à $T = 8$, et tend vers $\\log_2 5$. Une "
            "température négative inverse les écarts, donc l'ordre : à $T = -1$, la classe 4, au plus petit score, "
            "prend 0,62. Un modèle de langage « créatif » utilise une température plus grande (proche de 1 ou "
            "au-dessus), un modèle « sûr » une petite température."),

    Ex("17.20", "📦", 2, 25, "Quatre activations de scikit-learn sur les lunes et les spirales",
       "comparer, avec un protocole reproductible, les quatre activations de `MLPClassifier` sur deux problèmes.",
       "Ex 17.11 · ch. 15 (`MLPClassifier`) · fiche §17.6, section sur le choix d'une activation", thread="synth",
       tracks="R, C",
       body=r"""`MLPClassifier` propose quatre activations cachées : `"identity"`, `"logistic"` (la sigmoïde), `"tanh"` et `"relu"`. Compare-les sur deux problèmes : les deux lunes (600 points, bruit 0,25) et deux spirales (600 points, bruit 0,2, 1,5 tour), avec 30 % pour le test (`DATA_20`).

Écris `mean_accuracy_20(dataset, activation)` : pour `random_state` valant 0, 1, 2, 3 et 4, entraîne `MLPClassifier((16,), activation=activation, solver="lbfgs", max_iter=3000, random_state=random_state)` sur la partie d'entraînement de `DATA_20[dataset]` (avec `fit_quietly`), mesure son accuracy de test, et rends la **moyenne** des cinq accuracies (un `float`). La vérification calcule les huit moyennes avec ta fonction, affiche le tableau, et vérifie :
a) la moyenne de l'identité sur les spirales ;
b) `best_moons_20` : l'activation de meilleure moyenne sur les lunes (son nom, entre guillemets) ;
c) `best_spirals_20` : l'activation de meilleure moyenne sur les spirales ;
d) l'écart entre la moyenne de ReLU et celle de tanh sur les spirales (ReLU moins tanh).

Remplis b) et c) **après** avoir lu le tableau.

Dans tes notes : le classement est-il le même sur les deux problèmes ? Explique l'écart entre les activations sur les spirales (pense à la forme de la frontière que 16 neurones de chaque sorte peuvent tracer). Que dit l'écart entre les cinq graines (relance une ligne avec `np.std`) de la fiabilité d'une comparaison sur une seule graine ?""",
       given=GIVEN_20, todo=TODO_20, check=CHECK_20, solution=SOLUTION_20 + as_printed(CHECK_20, "17.20"),
       record=r'''wb.record("17.20a", results_20["spirales"]["identity"], decimals=4,
          mistakes={"c'est la moyenne de l'identité sur les lunes ; relis la question : on demande les spirales": results_20["lunes"]["identity"]})
wb.record("17.20b", "relu", mistakes={"relis le tableau : sur les lunes, laquelle a la plus grande moyenne ?": "tanh",
                                      "la sigmoïde (logistic) n'est pas en tête sur les lunes": "logistic",
                                      "sans activation, le MLP trace une droite : il ne peut pas suivre les lunes": "identity"})
wb.record("17.20c", "tanh", mistakes={"relis le tableau : sur les spirales, laquelle a la plus grande moyenne ?": "relu",
                                      "la sigmoïde (logistic) n'est pas en tête sur les spirales": "logistic",
                                      "sans activation, le MLP trace une droite": "identity"})
wb.record("17.20d", results_20["spirales"]["relu"] - results_20["spirales"]["tanh"], decimals=4,
          mistakes={"ReLU moins tanh, pas l'inverse": results_20["spirales"]["tanh"] - results_20["spirales"]["relu"]})''',
       note="Sur les lunes, ReLU est en tête (0,929), devant tanh (0,907) et la sigmoïde (0,898) ; l'identité, une "
            "droite, plafonne à 0,817. Sur les spirales, tanh domine nettement (0,992), la sigmoïde est très instable "
            "d'une graine à l'autre (0,853 en moyenne, avec un écart-type de 0,17), et ReLU ne fait que 0,837 : avec 16 "
            "neurones, une ReLU ne dispose que de 16 coudes pour approcher une spirale, alors que 16 courbes tanh "
            "s'additionnent en une frontière lisse. Le classement dépend donc du problème et de la taille du "
            "réseau : « ReLU par défaut » est un point de départ, pas une loi. L'écart-type entre graines (0,05 pour "
            "ReLU sur les spirales, 0,17 pour la sigmoïde) montre qu'une comparaison sur une seule graine peut s'inverser."),

    Ex("17.21", "📈", 2, 20, "Diagnostiquer la saturation sur des histogrammes de pré-activations",
       "repérer sur des histogrammes les couches sigmoïdes saturées, puis le chiffrer avec ta dérivée.",
       "Ex 17.13 · ch. 16 (histogrammes d'activations) · fiche §17.6 (la saturation)", thread="synth", tracks="M",
       body=r"""La cellule ci-dessous trace les pré-activations $z$ de quatre couches sigmoïdes P, Q, R et S (1 000 exemples, 5 neurones chacune), comme le ferait un diagnostic avant l'entraînement (ch. 16). En rose : la zone $|z| > 4{,}6$, où $\sigma'(z) < 0{,}01$.

En lisant les histogrammes :
a) `most_21` : la lettre de la couche la plus saturée (celle dont les neurones ont, en moyenne, la plus petite pente) ;
b) `linear_21` : la lettre de la couche qui travaille presque dans la partie linéaire de la sigmoïde (une pente presque constante).

Puis écris `saturation_21(z)`, qui rend, avec **ta** `sigmoid_derivative`, la pente moyenne $\overline{\sigma'(z)}$ et la part des valeurs où $\sigma'(z) < 0{,}01$. La vérification calcule :
c) les pentes moyennes des quatre couches, dans l'ordre P, Q, R, S (une liste) ;
d) la part des pentes inférieures à 0,01 dans la couche R.

Dans tes notes : la couche R a-t-elle plus de valeurs dans la zone rose que Q ? Pourquoi sa pente moyenne est-elle pourtant plus grande ? Qu'est-ce qui, dans une initialisation ou dans les données, peut produire chacun des quatre histogrammes ?""",
       given=GIVEN_21, todo=TODO_21, check=CHECK_21, solution=SOLUTION_21 + as_printed(CHECK_21, "17.21"),
       record=r'''wb.record("17.21a", "Q", mistakes={"une distribution large a des valeurs loin de 0, mais plus de la moitié restent près de 0 : compare aussi les centres": "R",
                                    "cette couche est centrée en 0, avec un écart-type de 1 : sa pente reste forte": "P",
                                    "cette couche est étroite : ses pentes sont-elles petites ?": "S"})
wb.record("17.21b", "S", mistakes={"un écart-type de 1 couvre une bonne partie de la courbe en S, courbure comprise": "P",
                                   "cette couche est-elle dans la partie centrale de la courbe en S ?": "Q",
                                   "cette couche couvre toute la courbe, plateaux compris": "R"})
wb.record("17.21c", [stats_21[name][0] for name in "PQRS"], decimals=4)
wb.record("17.21d", stats_21["R"][1], decimals=4,
          mistakes={"c'est la part de la couche Q ; on demande la couche R": stats_21["Q"][1]})''',
       note="Q est centrée en 4 : presque tous ses $z$ sont sur le plateau du haut, et sa pente moyenne (0,027) est "
            "huit fois plus faible que celle de P (0,207). R, très large (écart-type 4), a presque autant de valeurs "
            "dans la zone rose (25 % contre 28 %), mais ses autres valeurs tombent souvent près de 0, où la pente est "
            "forte : sa pente moyenne (0,091) est plus de trois fois celle de Q. S, étroite autour de $-1$, est dans "
            "la partie presque linéaire (pente 0,195, presque constante). Q ressemble à une couche dont les biais "
            "sont trop grands (ou dont les entrées ne sont pas centrées) ; R à des poids trop grands ; S à des "
            "poids très petits et un biais négatif ; P à une bonne initialisation sur des entrées standardisées."),
])

# ---- Part D ----------------------------------------------------------------
ACT_OK_22 = r'''def act_ok(name):
    """(f, f') for "relu", "sigmoid", "tanh" or "softplus": four activations and their derivatives (correct)."""
    def relu(z):
        z = np.array(z, dtype=float)
        return np.where(z > 0, z, 0.0)

    def relu_derivative(z):
        z = np.array(z, dtype=float)
        return (z > 0).astype(float)

    def sigmoid(z):
        z = np.array(z, dtype=float)
        e = np.exp(-np.abs(z))
        return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

    def sigmoid_derivative(z):
        s = sigmoid(z)
        return s * (1 - s)

    def tanh(z):
        return np.tanh(np.array(z, dtype=float))

    def tanh_derivative(z):
        return 1 - np.tanh(np.array(z, dtype=float)) ** 2

    def softplus(z):
        z = np.array(z, dtype=float)
        return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))

    def softplus_derivative(z):
        return sigmoid(z)

    return {"relu": (relu, relu_derivative), "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative), "softplus": (softplus, softplus_derivative)}[name]'''

BUGS_22 = [  # (name, the lines of act_ok it replaces, the buggy lines)
    ("act_bug_1", "        return (z > 0).astype(float)", "        return (z >= 0).astype(float)"),
    ("act_bug_2", "        return np.where(z >= 0, 1 / (1 + e), e / (1 + e))", "        return np.where(z >= 0, e / (1 + e), 1 / (1 + e))"),
    ("act_bug_3", "        return 1 - np.tanh(np.array(z, dtype=float)) ** 2",
     "        return 1 - np.tanh(np.tanh(np.array(z, dtype=float))) ** 2"),
    ("act_bug_4", "        z = np.array(z, dtype=float)\n        return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))",
     "        return np.log1p(np.exp(np.array(z, dtype=float)))"),
    ("act_bug_5", "    def sigmoid_derivative(z):\n        s = sigmoid(z)",
     "    def sigmoid_derivative(z):\n        z = np.asarray(z, dtype=float)\n        z *= -1.0                       # sigma'(-z) = sigma'(z): \"saves\" a copy\n        s = sigmoid(z)"),
]


def _bug_22(name: str, old: str, new: str) -> str:
    assert ACT_OK_22.count(old) == 1, name
    return ACT_OK_22.replace(old, new).replace("def act_ok(name):", f"def {name}(name):").replace(
        "four activations and their derivatives (correct).", "the same, with one bug.")


GIVEN_22 = ("import math\n\nimport pytest                                     # for the @pytest.mark.parametrize decorators of your tests\n\n\n"
            + ACT_OK_22 + "\n\n\n" + "\n\n\n".join(_bug_22(*bug) for bug in BUGS_22))

TODO_22 = r'''# Write your test functions here (their names start with test_). They test a function act(name) whose contract
# is the one of act_ok (read it): act(name) returns the pair (f, f'). At least ONE of your tests must be
# parametrized with @pytest.mark.parametrize, for example over the four names:
#
# @pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "softplus"])
# def test_something(name):
#     f, df = act(name)
#     assert ...

my_tests_22 = ...   # the list of your test functions'''

SOLUTION_22 = r'''@pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "softplus"])
def test_derivative_matches_central_differences(name):
    f, df = act(name)
    z = np.linspace(-6, 6, 241)                                  # -6 to 6, step 0.05
    z = z[np.abs(z) > 1e-3]                                      # away from the kink of ReLU
    h = 1e-5
    assert np.allclose(df(z), (f(z + h) - f(z - h)) / (2 * h), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "softplus"])
def test_function_is_non_decreasing(name):
    f, _ = act(name)
    assert np.all(np.diff(f(np.linspace(-6, 6, 241))) >= -1e-15)


@pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "softplus"])
@pytest.mark.parametrize("which", [0, 1], ids=["f", "df"])
def test_large_inputs_stay_finite_without_warnings(name, which):
    g = act(name)[which]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = g(np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0]))
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "softplus"])
@pytest.mark.parametrize("which", [0, 1], ids=["f", "df"])
def test_input_is_left_unchanged(name, which):
    z = np.array([-2.0, -0.5, 0.0, 1.5])
    before = z.copy()
    act(name)[which](z)
    assert np.array_equal(z, before)


def test_relu_derivative_is_zero_at_zero_as_in_pytorch():
    _, df = act("relu")
    assert df(np.array([0.0]))[0] == 0.0


my_tests_22 = [test_derivative_matches_central_differences, test_function_is_non_decreasing,
               test_large_inputs_stay_finite_without_warnings, test_input_is_left_unchanged,
               test_relu_derivative_is_zero_at_zero_as_in_pytorch]
'''

CHECK_22 = r'''def parametrized_22(tests):
    """True when at least one test function carries a pytest.mark.parametrize mark."""
    return any(mark.name == "parametrize" for t in tests for mark in getattr(t, "pytestmark", []))


if my_tests_22 is ...:
    print("⏳ Ex 17.22 : tests pas encore écrits.")
else:
    good_22 = wb.run_pytest(my_tests_22, subject=act_ok, name="act")
    verdict("17.22", parametrized_22(my_tests_22), "au moins un de tes tests est paramétré.",
            "aucun de tes tests n'est paramétré : utilise @pytest.mark.parametrize pour en exécuter un sur toute la famille.")
    verdict("17.22", good_22.ok, f"tes tests passent sur la version juste ({good_22.passed} cas).",
            "tes tests doivent tous passer sur act_ok, la version juste (voir le détail au-dessus).")
    if good_22.ok:
        for bug_22 in [act_bug_1, act_bug_2, act_bug_3, act_bug_4, act_bug_5]:
            result_22 = wb.run_pytest(my_tests_22, subject=bug_22, name="act", quiet=True)
            verdict("17.22", result_22.failed + result_22.errors > 0, f"{bug_22.__name__} est attrapé ({result_22.failed} cas en échec).",
                    f"tes tests passent sur {bug_22.__name__}, qui est faux : lis-le, puis ajoute un test qui le fait échouer.")'''

GIVEN_23 = r'''X_23 = np.random.default_rng(1723).normal(size=(500, 64))      # 500 standardized samples, 64 features
G_23 = np.random.default_rng(17231).normal(size=(500, 64))      # a gradient arriving on the last layer


def weights_23(name):
    """Ten square 64 x 64 layers, without bias: Glorot for sigmoid and tanh, He for relu and gelu (chapter 16)."""
    gain = 2.0 if name in ("relu", "gelu") else 1.0
    rng = np.random.default_rng(17230)
    return [rng.normal(0.0, np.sqrt(gain / 64), size=(64, 64)) for _ in range(10)]


def backward_ratio_23(Ws, slopes):
    """||gradient on the input|| / ||gradient on the last layer||, by the chain rule through the ten layers (ch. 18)."""
    g = G_23
    for W, s in zip(reversed(Ws), reversed(slopes)):
        g = (g * s) @ W.T
    return float(np.linalg.norm(g) / np.linalg.norm(G_23))'''

TODO_23 = r'''def slopes_23(name):
    """The ten arrays f'(z) (each of shape (500, 64)) of a forward pass of X_23 through weights_23(name),
    a = f(z) with z = a @ W at each layer, f and f' from YOUR get_activation(name)."""
    raise NotImplementedError("slopes_23")'''

SOLUTION_23 = r'''def slopes_23(name):
    """The ten arrays f'(z) (each of shape (500, 64)) of a forward pass of X_23 through weights_23(name),
    a = f(z) with z = a @ W at each layer, f and f' from YOUR get_activation(name)."""
    f, df = mylearn.nn.activations.get_activation(name)
    a, slopes = X_23, []
    for W in weights_23(name):
        z = a @ W
        slopes.append(np.asarray(df(z), dtype=float))
        a = np.asarray(f(z), dtype=float)
    return slopes
'''

CHECK_23 = r'''summary_23 = {}
with checked("17.23"):
    for name_23 in ["sigmoid", "tanh", "relu", "gelu"]:
        s_23 = slopes_23(name_23)
        if len(s_23) != 10 or any(np.shape(s) != (500, 64) for s in s_23):
            print(f"❌ Ex 17.23 : slopes_23({name_23!r}) doit rendre 10 tableaux de forme (500, 64).")
            continue
        means_23 = [float(np.mean(s)) for s in s_23]
        summary_23[name_23] = (means_23, float(np.prod(means_23)), backward_ratio_23(weights_23(name_23), s_23))
        print(f"{name_23:8s} mean f' per layer {np.round(means_23, 3).tolist()}")
        print(f"{'':8s} product of the means {summary_23[name_23][1]:.2e} · gradient ratio input/output {summary_23[name_23][2]:.2e}")
if len(summary_23) == 4:
    fig_23, ax_23 = plt.subplots(figsize=(7, 3.6))
    for name_23, (means_23, _, _) in summary_23.items():
        ax_23.plot(range(1, 11), np.cumprod(means_23), "o-", label=name_23)
    ax_23.set_yscale("log")
    ax_23.set_xlabel("couche")
    ax_23.set_ylabel("produit des pentes moyennes")
    ax_23.legend()
    plt.tight_layout()
    plt.show()
    verdict("17.23", summary_23["sigmoid"][2] < 1e-4, f"sigmoïde : le gradient est divisé par plus de 10⁴ ({summary_23['sigmoid'][2]:.1e}).",
            f"sigmoïde : un rapport de {summary_23['sigmoid'][2]:.1e} est inattendu ; vérifie tes dérivées (σ′ ≤ 1/4).")
    verdict("17.23", 0.1 < summary_23["relu"][2] < 10, f"ReLU avec He : le gradient garde son ordre de grandeur ({summary_23['relu'][2]:.2f}).",
            f"ReLU avec He : un rapport de {summary_23['relu'][2]:.2e} est inattendu ; vérifie relu_derivative.")
    verdict("17.23", summary_23["relu"][1] < 1e-2 < summary_23["relu"][2],
            "ReLU : le produit des pentes moyennes s'effondre, mais pas le gradient : les poids de He compensent.",
            "le produit des pentes de ReLU devrait être petit (≈ 0,5 par couche) et le rapport de gradients proche de 1.")'''

GIVEN_24 = r'''def spirals_24(seed):
    """Two spirals of 1600 points (noise 0.3, 2.5 turns), half for training and half for testing."""
    X, y = wb.synth.make_spirals(n=1600, n_classes=2, noise=0.3, turns=2.5, seed=seed)
    return train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)


X_24, X_HIDDEN_24, y_24, Y_HIDDEN_24 = spirals_24(1724)          # the test half is the hidden test: do not look at it
OTHER_SEEDS_24 = [2401, 2402, 2403, 2404]
print("training points:", X_24.shape, "· classes:", np.bincount(y_24).tolist())


def n_params_24(model):
    """The number of trained parameters of the MLPClassifier of a fitted model (or of the last step of a Pipeline)."""
    mlp = model[-1] if hasattr(model, "steps") else model
    return sum(W.size for W in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)'''

TODO_24 = r'''def build_24():
    """A NEW, unfitted scikit-learn model: an MLPClassifier, or a Pipeline whose last step is an MLPClassifier."""
    raise NotImplementedError("build_24")'''

SOLUTION_24 = r'''from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_24():
    """A NEW, unfitted scikit-learn model: an MLPClassifier, or a Pipeline whose last step is an MLPClassifier."""
    return make_pipeline(StandardScaler(),
                         MLPClassifier((24, 24), activation="tanh", solver="lbfgs", alpha=0.1, max_iter=3000, random_state=0))
'''

CHECK_24 = r'''from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

with checked("17.24"):
    model_24 = build_24()
    last_24 = model_24[-1] if hasattr(model_24, "steps") else model_24
    rules_24 = isinstance(last_24, MLPClassifier)
    verdict("17.24", rules_24, "le modèle se termine par un MLPClassifier.",
            f"le dernier élément du modèle est un {type(last_24).__name__} : le défi demande un MLPClassifier.")
    try:
        check_is_fitted(model_24)
        rules_24 = False
        print("❌ Ex 17.24 : build_24 rend un modèle déjà entraîné : rends un modèle neuf.")
    except NotFittedError:
        pass
    if build_24() is model_24:
        rules_24 = False
        print("❌ Ex 17.24 : build_24 rend deux fois le même objet : crée un nouveau modèle à chaque appel.")
    if rules_24:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cv_24 = cross_val_score(clone(model_24), X_24, y_24, cv=5)
        fitted_24 = fit_quietly(build_24(), X_24, y_24)
        params_24 = n_params_24(fitted_24)
        print(f"validation croisée sur l'entraînement : {cv_24.mean():.3f} ± {cv_24.std():.3f} · {params_24} paramètres")
        verdict("17.24", params_24 <= 2000, f"{params_24} paramètres, sous la limite de 2 000.",
                f"{params_24} paramètres : la limite est de 2 000 (poids et biais du MLPClassifier).")
        if params_24 <= 2000:
            if globals().get("REVEALED_24") is None:              # revealed once per session (an honour rule)
                others_24 = []
                for seed_24 in OTHER_SEEDS_24:
                    Xa, Xb, ya, yb = spirals_24(seed_24)
                    others_24.append(fit_quietly(build_24(), Xa, ya).score(Xb, yb))
                REVEALED_24 = (fitted_24.score(X_HIDDEN_24, Y_HIDDEN_24), float(np.mean(others_24)), params_24)
            test_24, mean_24, frozen_params_24 = REVEALED_24
            print(f"test caché : {test_24:.3f} · moyenne sur quatre autres tirages : {mean_24:.3f} "
                  f"(révélés une seule fois par session, pour le modèle à {frozen_params_24} paramètres)")
            verdict("17.24", test_24 >= 0.97 and mean_24 >= 0.97,
                    "défi réussi : au moins 0,97 sur le test caché et en moyenne sur quatre autres tirages."
                    + (" 🌟 Palier atteint : au moins 0,99 sur le test caché." if test_24 >= 0.99 else ""),
                    "pas encore : il faut au moins 0,97 sur le test caché ET en moyenne sur les quatre autres tirages.")'''

PART_D = Part("D", "Tester et défier",
              "Écrire une batterie de tests paramétrés pour toute une famille de fonctions (17.22), mesurer ce que "
              "chaque activation fait à un gradient qui traverse dix couches (17.23), puis relever le défi des "
              "spirales sous contrainte de paramètres (17.24).",
              exercises=[
    Ex("17.22", "🛠️", 2, 25, "Une batterie de tests paramétrés pour toutes les activations",
       "écrire des tests pytest paramétrés et des tests de propriétés qui attrapent cinq bugs différents dans une famille de fonctions.",
       "Ex 17.14 · 0A (`pytest.mark.parametrize`, `wb.run_pytest`) · fiche, section sur le choix et les tests", tracks="C",
       body=r"""La fonction `act_ok(name)` (fournie) rend le couple `(f, f′)` de quatre activations, `"relu"`, `"sigmoid"`, `"tanh"` et `"softplus"`, avec le contrat de ta librairie : chaque fonction rend un nouveau tableau `float64`, laisse son entrée intacte, reste finie sans avertissement pour $|z| \le 1\,000$, chaque dérivée vaut la pente de sa fonction, et $\mathrm{ReLU}'(0) = 0$ comme dans PyTorch. Écris une **batterie de tests** pytest pour une fonction `act(name)` qui a ce contrat : au moins un test doit être **paramétré** (`@pytest.mark.parametrize`, par exemple sur les quatre noms), et tu peux tester des **propriétés** (une fonction croissante, une dérivée égale à la pente numérique) plutôt que des valeurs.

Range tes tests dans la liste `my_tests_22`. Comme en 16.21, la vérification lance vraiment pytest : tes tests doivent **passer** sur `act_ok` et **échouer** sur chacune des cinq versions buggées fournies, `act_bug_1` à `act_bug_5` (lis-les : que fait chacune de travers ?). Tes tests ne peuvent utiliser que `act`, `np`, `math` et `pytest`. Cet exercice se corrige avec `05_solutions.md`.

Dans tes notes : pour chaque version buggée, le test qui l'attrape. Lesquelles passeraient un test qui compare les valeurs en trois points « ordinaires » comme $-1$, $0$ et $2$ ?""",
       given=GIVEN_22, todo=TODO_22, check=CHECK_22, solution=SOLUTION_22 + CHECK_22,
       note="Une batterie de cinq tests suffit, dont quatre paramétrés : la pente numérique (attrape `act_bug_3`, la "
            "dérivée de tanh prise en sa sortie) ; une fonction croissante (attrape `act_bug_2`, qui rend $\\sigma(-z)$ : "
            "ses valeurs restent dans $]0, 1[$ et vérifient même $\\sigma(-z) = 1 - \\sigma(z)$, seule la monotonie ou une "
            "comparaison à un oracle la trahit) ; des valeurs finies pour $\\pm 1\\,000$ (attrape `act_bug_4`, la softplus "
            "naïve, qui vaut `inf` en 1 000) ; une entrée intacte (attrape `act_bug_5`, qui retourne $z$ en place pour "
            "« économiser » une copie) ; et la convention en 0 (attrape `act_bug_1`, $\\mathrm{ReLU}'(0) = 1$, que la "
            "pente numérique ne peut pas voir). En trois points ordinaires, `act_bug_1`, `act_bug_4` et `act_bug_5` passent "
            "(`act_bug_5` rend même les bonnes valeurs), et `act_bug_3` aussi si l'on ne teste que les fonctions."),

    Ex("17.23", "🔬", 3, 45, "Produit des dérivées sur 10 couches : sigmoïde, tanh, ReLU, GELU",
       "mesurer, couche par couche, les pentes des activations d'un réseau profond et ce qu'il reste d'un gradient après dix couches.",
       "Ex 17.14 · ch. 16 (initialisations de Glorot et He) · fiche §17.6 (la saturation)", thread="synth",
       tracks="R, M, C",
       body=r"""Un réseau de dix couches carrées de 64 neurones, sans biais, reçoit 500 exemples standardisés (`X_23`). Ses poids viennent de `weights_23(name)` : Glorot pour la sigmoïde et tanh, He pour ReLU et GELU, comme le recommande le ch. 16. Écris `slopes_23(name)` : la passe avant de `X_23` à travers ces dix couches ($\mathbf{z} = \mathbf{a}\mathbf{W}$, puis $\mathbf{a} = f(\mathbf{z})$), qui rend la liste des dix tableaux $f'(\mathbf{z})$, avec $f$ et $f'$ tirés de **ton** `get_activation`.

La vérification calcule, pour chaque activation, la pente moyenne de chaque couche, le **produit** de ces dix pentes moyennes, et le rapport entre la norme du gradient qui arrive sur l'entrée et celle du gradient parti de la dernière couche, calculé par `backward_ratio_23` avec la règle de la chaîne (le ch. 18 la détaillera : à chaque couche, le gradient est multiplié case par case par $f'(\mathbf{z})$, puis par $\mathbf{W}^\top$). Elle trace le produit cumulé des pentes moyennes, et contrôle trois ordres de grandeur.

Dans tes notes : pour quelle activation le gradient s'évanouit-il ? Pourquoi le produit des pentes moyennes de ReLU est-il si petit, alors que son gradient arrive presque intact ? Que se passerait-il avec ReLU et l'initialisation de Glorot (copie `weights_23` avec un paramètre `gain`, et refais l'expérience) ?""",
       given=GIVEN_23, todo=TODO_23, check=CHECK_23, solution=SOLUTION_23 + CHECK_23.replace('with checked("17.23"):\n', "if True:\n"),
       note="La sigmoïde a des pentes moyennes de 0,21 à 0,24 à chaque couche (jamais plus de 1/4) : leur produit vaut "
            "$4 \\times 10^{-7}$, et le gradient arrive sur l'entrée divisé par deux millions environ ($5 \\times 10^{-7}$) : "
            "les premières couches n'apprendraient rien. tanh garde des pentes moyennes de 0,6 à 0,95 (ses "
            "activations restent près de 0, où sa pente vaut presque 1) : produit 0,21, gradient 0,27. ReLU a une "
            "pente moyenne d'environ 0,5 (la moitié des neurones sont actifs, de pente 1), soit un produit de "
            "$9 \\times 10^{-4}$ ; mais l'initialisation de He double la variance des poids, et chaque $\\mathbf{W}^\\top$ "
            "compense ce facteur $1/2$ : le gradient arrive avec un rapport de 0,96. GELU, avec He, donne 0,7. Avec "
            "Glorot et ReLU, chaque couche diviserait la variance du gradient par 2 : un rapport de l'ordre de "
            "$2^{-5} \\approx 0{,}03$ après dix couches. Le produit des pentes seul ne dit donc pas tout : c'est la "
            "combinaison de l'activation et de l'initialisation qui compte."),

    Ex("17.24", "🏆", 3, 60, "Défi spirales : au moins 97 % avec au plus 2 000 paramètres",
       "choisir activation, architecture, solveur et régularisation d'un MLP sous une contrainte de paramètres, et le vérifier sur des données jamais vues.",
       "Ex 17.20 · ch. 16 (compter les paramètres) · ch. 8 (validation croisée) · ch. 15 (`Pipeline`)", thread="synth",
       tracks="C",
       body=r"""Deux spirales entrelacées de 1 600 points (bruit 0,3, 2,5 tours) : la moitié pour l'entraînement (`X_24`, `y_24`), l'autre moitié forme un **test caché**. Écris `build_24()`, qui rend un modèle scikit-learn **neuf** (pas encore entraîné) : un `MLPClassifier`, éventuellement précédé d'un prétraitement dans un `Pipeline` (ch. 15). Tu choisis l'activation (parmi les quatre de scikit-learn), l'architecture, le solveur, la régularisation (`alpha`) et le nombre d'itérations.

Règles : au plus **2 000 paramètres** entraînés (poids et biais du `MLPClassifier`, comptés par `n_params_24`) ; le défi est réussi avec une accuracy d'au moins **0,97** sur le test caché **et** en moyenne sur quatre autres tirages des spirales (ta méthode est réentraînée sur chaque tirage) ; palier 🌟 : au moins 0,99 sur le test caché. La vérification entraîne ton modèle une dizaine de fois (validation croisée, test caché, quatre tirages) : garde un entraînement sous 5 secondes environ (`max_iter` de 1 000 à 3 000). Règle d'honneur : choisis tes réglages par validation croisée sur `X_24` seulement (la vérification l'affiche), et ne regarde le test caché qu'une fois : la vérification ne le révèle qu'une fois par session, et fige le résultat.

Dans tes notes : ta démarche (ce que tu as essayé, dans quel ordre, et les scores de validation croisée), puis ce qui a le plus compté : l'activation, la taille, `alpha` ou le solveur ? Le modèle par défaut, `MLPClassifier()`, qu'obtient-il en validation croisée ?""",
       given=GIVEN_24, todo=TODO_24, check=CHECK_24, solution=SOLUTION_24 + CHECK_24.replace('with checked("17.24"):\n', "if True:\n"),
       note="Le modèle de référence, une standardisation suivie d'un MLP tanh 24-24 entraîné par `lbfgs` avec "
            "`alpha = 0.1` (697 paramètres), obtient 0,985 en validation croisée, 0,994 sur le test caché et 0,996 en "
            "moyenne sur les quatre autres tirages. Deux réglages comptent plus que l'activation : le **solveur** et la "
            "**régularisation**. Sans régularisation (`alpha` par défaut, $10^{-4}$), le même réseau colle aux 800 points "
            "d'entraînement (accuracy de 1) et tombe vers 0,95 en validation croisée ; avec `alpha` entre 0,01 et 0,1, il "
            "atteint 0,985. Et sans un solveur qui converge, rien ne marche : avec `adam` et 200 "
            "itérations, le `MLPClassifier()` par défaut (ReLU, 100 neurones) reste vers 0,60 en validation croisée, faute d'avoir "
            "convergé sur des entrées aussi petites (des rayons inférieurs à 1). Une ReLU 24-24 avec standardisation "
            "et `alpha = 0.1` fait presque aussi bien que tanh (0,991 en moyenne) ; sans standardisation (même `alpha`), elle est "
            "beaucoup moins stable (de 0,90 à 0,99 selon le tirage)."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 17.Q1 à 17.Q12, 17.R1 à 17.R3, 17.1, 17.2, 17.4, 17.5, 17.6, 17.8 | vérifier tes réponses courtes | "
            "🧠 🔁 ✏️ ∂ 🧮 | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if kind == "exercise":
        title = "# 17 · Fonctions d'activation — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch17_activations/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 17`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 17 · Fonctions d'activation — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Voir qu'un réseau sans activation ne vaut qu'une couche, et ce qu'une ReLU y change.\n"
               "- Écrire et vérifier contre SciPy et PyTorch les fonctions de `mylearn.nn.activations` : dix activations "
               "et leurs dérivées, `get_activation`, `softmax`, `log_softmax` et `softmax_derivative`, sans aucun "
               "débordement.\n"
               "- Comparer les activations : saturation, température, choix pour un problème donné, gradients à "
               "travers dix couches.\n"
               "- Tester toute une famille de fonctions avec des tests paramétrés et des tests de propriétés.\n\n"
               "**Rappel express.** $\\mathrm{ReLU}(z) = \\max(0, z)$, $\\mathrm{ReLU}'(0) = 0$ (PyTorch) ; "
               "$\\sigma(z) = 1/(1 + e^{-z})$, $\\sigma' = \\sigma(1 - \\sigma)$ ; $\\tanh' = 1 - \\tanh^2$ ; "
               "$\\mathrm{softplus}(z) = \\log(1 + e^z)$, $\\mathrm{softplus}' = \\sigma$ ; $\\mathrm{ELU}(z) = \\alpha(e^z - 1)$ "
               "pour $z \\le 0$ ; $\\mathrm{SiLU}(z) = z\\,\\sigma(z)$ ; $\\mathrm{GELU}(z) = z\\,\\Phi(z)$, "
               "$\\Phi(z) = \\frac{1}{2}(1 + \\mathrm{erf}(z/\\sqrt{2}))$ ; $p_i = e^{z_i}/\\sum_j e^{z_j}$, "
               "$\\partial p_i/\\partial z_j = p_i(\\delta_{ij} - p_j)$ ; $\\log\\sum_j e^{z_j} = m + \\log\\sum_j e^{z_j - m}$. "
               "En NumPy : `np.exp(-np.abs(z))`, `np.log1p`, `np.expm1`, `np.where` (qui calcule les deux branches), "
               "`z.max(axis=-1, keepdims=True)`, `np.vectorize(math.erf)` ; en SciPy : `special.expit`, "
               "`special.softmax`, `special.log_softmax`, `special.erf`.")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu montrer qu'un réseau sans activation s'effondre, et calculer la couche qui le remplace ?\n"
               "2. Sais-tu écrire chaque activation et sa dérivée sans débordement, et la softmax, la log-softmax et sa "
               "Jacobienne ?\n"
               "3. Sais-tu choisir l'activation des couches cachées et celle de la sortie, et justifier ce choix par la "
               "saturation, les neurones morts et les gradients ?\n\n"
               "**Pour aller plus loin** : les articles de Hendrycks et Gimpel (2016), de Ramachandran, Zoph et Le "
               "(2017) et de Shazeer (2020), cités dans la fiche. La suite : le ch. 18, où les couples `(f, f′)` de ton "
               "`get_activation` et le cache de ton `mlp_forward` serviront à la rétropropagation.")]


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
