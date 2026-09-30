#!/usr/bin/env python
"""Build the two notebooks of chapter 0A from a single source (used by Claude).

    python tools/chapters/build_ch00a.py
    python tools/run_all_notebooks.py chapitres/ch00a_python/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch00a_python/03_notebook.ipynb

Each exercise is described once (statement, shared data cell, exercise code,
check code, solution code, recorded answers), so the exercise notebook and the
solutions notebook can never disagree on an ID, a variable name or a format.
The exercise notebook only DEFINES things in its TODO cells (``...`` or
``raise NotImplementedError``); every call to the learner's code happens in a
check cell, inside ``wb.attempt`` when it can raise.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from nbbuild import badge, code, md, setup_cell, write_notebook  # noqa: E402

CHAPTER = "0A"
FOLDER = "chapitres/ch00a_python"
STARS = {1: "★", 2: "★★", 3: "★★★", 4: "★★★★"}


@dataclass
class Ex:
    id: str
    type: str
    stars: int
    minutes: int
    title: str
    goal: str
    prereq: str
    body: str                      # the statement (Markdown, French)
    todo: str = ""                 # exercise notebook: code to complete (definitions only)
    check: str = ""                # exercise notebook: calls and wb.check
    solution: str = ""             # solutions notebook: the worked solution
    record: str = ""               # solutions notebook: wb.record (cell tagged "answer")
    given: str = ""                # code shared by both notebooks (data preparation)
    thread: str = "—"
    tracks: str = ""
    hypothesis: bool = False       # 🔮: a "my hypothesis" cell before running
    after: list = field(default_factory=list)   # extra (markdown, code) cells, both notebooks
    note: str = ""                 # solutions notebook: short remark after the answer

    def header(self) -> str:
        fil = f" · **Fil rouge :** {self.thread}" if self.thread != "—" else ""
        return (f"### Ex {self.id} — {self.title} {self.type} {STARS[self.stars]} ⏱️ {self.minutes} min\n"
                f"**Objectif :** {self.goal}  \n**Prérequis :** {self.prereq}{fil}"
                + (f" · **Parcours :** {self.tracks}" if self.tracks else ""))


@dataclass
class Part:
    key: str
    title: str
    intro: str
    given: str = ""
    exercises: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Part 0: checks of the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER = [
    ("0A.1", "abcdefg", "Évaluer des expressions à la main : //, %, **, conversions", [
        ("a", "17 // 5", "17 // 5", ""),
        ("b", "17 % 5", "17 % 5", ""),
        ("c", "-17 // 5", "-17 // 5", 'mistakes={"// arrondit vers le bas (vers moins l\'infini), pas vers zéro": -3}'),
        ("d", "(2 + 3) ** 2 - 10 // 3", "(2 + 3) ** 2 - 10 // 3", 'mistakes={"les parenthèses passent en premier : (2 + 3) ** 2 vaut 25": 8}'),
        ("e", "int(7.9) + round(7.5)", "int(7.9) + round(7.5)", 'mistakes={"int(7.9) tronque : il vaut 7, pas 8": 16, '
         '"round(7.5) vaut 8 : au milieu, Python arrondit au nombre pair le plus proche": 14}'),
        ("f", "10 / 4 * 2", "10 / 4 * 2", 'decimals=1, mistakes={"/ et * ont la même priorité : on calcule de gauche à droite, (10 / 4) * 2": 1.25}'),
        ("g", "the type of 7 / 7, as a string such as \"int\"", 'type(7 / 7).__name__',
         'mistakes={"/ donne toujours un float, même quand la division tombe juste": "int"}'),
    ]),
    ("0A.2", "abcdefgh", "Indices et tranches à la main", [
        ("a", "masses[2]", "masses[2]", ""),
        ("b", "masses[-2]", "masses[-2]", 'mistakes={"-1 est le dernier élément, -2 l\'avant-dernier": masses[-3]}'),
        ("c", "masses[1:4] (a list)", "masses[1:4]", 'mistakes={"la fin d\'une tranche est exclue": masses[1:5]}'),
        ("d", "masses[::2] (a list)", "masses[::2]", ""),
        ("e", "name[:5]", "name[:5]", ""),
        ("f", "name[-4:]", "name[-4:]", ""),
        ("g", "len(masses[2:])", "len(masses[2:])", ""),
        ("h", "point[2] - point[0], rounded to 1 decimal", "point[2] - point[0]", "decimals=1"),
    ]),
    ("0A.3", "abcde", "Dérouler une boucle et une compréhension", [
        ("a", "total", "total", ""),
        ("b", "count", "count", ""),
        ("c", "[v * 2 for v in values if v > 4] (a list)", "[v * 2 for v in values if v > 4]", ""),
        ("d", "sum(v for v in values if v % 3 == 0)", "sum(v for v in values if v % 3 == 0)", ""),
        ("e", "steps", "steps", ""),
    ]),
    ("0A.4", "abcdef", "Un groupby à la main sur huit manchots", [
        ("a", "mean mass of Gentoo, 1 decimal", 'df.groupby("species")["body_mass_g"].mean()["Gentoo"]', "decimals=1"),
        ("b", "mean mass of Adelie, 1 decimal", 'df.groupby("species")["body_mass_g"].mean()["Adelie"]', "decimals=1"),
        ("c", "number of penguins on Dream", '(df["island"] == "Dream").sum()', ""),
        ("d", "maximum mass on Dream", 'df.groupby("island")["body_mass_g"].max()["Dream"]', ""),
        ("e", "number of male Gentoo", 'df[df["sex"] == "male"]["species"].value_counts()["Gentoo"]', ""),
        ("f", "lightest species on average (a string)", 'df.groupby("species")["body_mass_g"].mean().idxmin()', ""),
    ]),
    ("0A.5", "abcdef", "Mini-lots : combien de lots, de quelle taille, combien de mises à jour ?", [
        ("a", "batches per epoch (last one kept)", "math.ceil(333 / 64)",
         'mistakes={"le dernier lot, incomplet, compte aussi : arrondis vers le haut": 5}'),
        ("b", "size of the last batch", "333 - 64 * (333 // 64)", 'mistakes={"c\'est le reste de la division de 333 par 64": 5}'),
        ("c", "batches per epoch with drop_last=True", "333 // 64", ""),
        ("d", "updates in 20 epochs (last batch kept)", "20 * math.ceil(333 / 64)", 'mistakes={"avec drop_last=True, ce serait 100 : ici le dernier lot est gardé": 100}'),
        ("e", "updates per epoch with batches of 1", "333", ""),
        ("f", "updates in 20 epochs with one full batch", "20", ""),
    ]),
    ("0A.6", "abcdef", "Portée, valeurs par défaut et arguments nommés", [
        ("a", "a", "a", 'mistakes={"dans price, rate est une variable locale qui vaut 5": 60}'),
        ("b", "b", "b", 'mistakes={"dans price, rate est une variable locale qui vaut 5": 120}'),
        ("c", "c", "c", ""),
        ("d", "d", "d", ""),
        ("e", "e", "e", 'mistakes={"rate = 5 est locale à la fonction : la variable globale rate n\'a pas changé": 5}'),
        ("f", 'the name of the error raised by price(1, 2, 3), as a string such as "ValueError"', "error_name", ""),
    ]),
    ("0A.7", "abcdefghi", "Formes NumPy à la main", [
        ("a", "shape of A[1:3], a tuple such as (4, 6)", "A[1:3].shape", ""),
        ("b", "shape of A[:, 2]", "A[:, 2].shape", 'mistakes={"un seul indice (2) retire l\'axe ; une tranche (2:3) le garderait": (4, 1)}'),
        ("c", "shape of A[:, 2:3]", "A[:, 2:3].shape", 'mistakes={"une tranche garde l\'axe, même si elle ne contient qu\'une colonne": (4,)}'),
        ("d", "shape of A.sum(axis=0)", "A.sum(axis=0).shape", 'mistakes={"axis=0 fait disparaître l\'axe des lignes : il reste une valeur par colonne": (4,)}'),
        ("e", "shape of A.mean(axis=1, keepdims=True)", "A.mean(axis=1, keepdims=True).shape", ""),
        ("f", "shape of A.reshape(2, -1, 3)", "A.reshape(2, -1, 3).shape", ""),
        ("g", "shape of B.reshape(len(B), -1)", "B.reshape(len(B), -1).shape", ""),
        ("h", "shape of A[A > 20]", "A[A > 20].shape", 'mistakes={"un masque booléen renvoie un tableau à UNE dimension des éléments retenus": (4, 6)}'),
        ("i", "the value of A[2, 3]", "A[2, 3]", ""),
    ]),
    ("0A.8", "abcdefg", "Broadcasting : compatibles ou non, et quelle forme ?", [
        ("a", '(5, 3) + (3,), a string such as "(5, 3)" or "erreur"', "broadcast((5, 3), (3,))", ""),
        ("b", "(5, 3) + (5,)", "broadcast((5, 3), (5,))",
         'mistakes={"compare les formes de droite à gauche : 3 et 5 sont différents et aucun ne vaut 1": "(5, 3)"}'),
        ("c", "(5, 1) * (1, 4)", "broadcast((5, 1), (1, 4))", ""),
        ("d", "(2, 1, 3) - (4, 1)", "broadcast((2, 1, 3), (4, 1))", ""),
        ("e", "(64, 28, 28) - (28, 28)", "broadcast((64, 28, 28), (28, 28))", ""),
        ("f", "(10,) - (10, 1)", "broadcast((10,), (10, 1))",
         'mistakes={"les axes de taille 1 sont étirés : (10,) se lit (1, 10), et le résultat est un tableau de toutes les différences": "erreur"}'),
        ("g", "(3, 4) * (4, 3)", "broadcast((3, 4), (4, 3))", ""),
    ]),
]

PAPER_CONTEXT = '''# Data of the paper exercises (the same as in 02_exercices.md)
import math

import numpy as np
import pandas as pd

masses = [3750, 3800, 3250, 3450, 3650, 3625]          # 0A.2
name = "Chinstrap"
point = (39.1, 18.7, 181)

values = [4, 7, 1, 8, 3, 6]                              # 0A.3
total = 0
count = 0
for i, v in enumerate(values):
    if v % 2 == 0:
        total += v * i
    elif v > 5:
        count += 1
n = 100
steps = 0
while n > 1:
    n = n // 3
    steps += 1

df = pd.DataFrame({                                      # 0A.4
    "species": ["Adelie", "Adelie", "Gentoo", "Chinstrap", "Gentoo", "Adelie", "Chinstrap", "Gentoo"],
    "island": ["Torgersen", "Dream", "Biscoe", "Dream", "Biscoe", "Biscoe", "Dream", "Biscoe"],
    "body_mass_g": [3750, 3400, 5200, 3700, 4650, 3900, 3950, 5550],
    "sex": ["male", "female", "male", "female", "female", "male", "male", "male"],
})

rate = 10                                                # 0A.6


def price(quantity, unit=2, *, discount=0):
    rate = 5
    return quantity * unit * rate - discount


a = price(3)
b = price(3, 4)
c = price(2, discount=7)
d = price(unit=1, quantity=4)
e = rate
try:
    price(1, 2, 3)
    error_name = "no error"
except Exception as error:  # noqa: BLE001 - we want the name of whatever is raised
    error_name = type(error).__name__

A = np.arange(24).reshape(4, 6)                          # 0A.7
B = np.zeros((3, 28, 28))


def broadcast(shape_a, shape_b):                         # 0A.8
    """Shape of the result as a string, or "erreur"."""
    try:
        return str(np.broadcast_shapes(shape_a, shape_b))
    except ValueError:
        return "erreur"'''


def paper_cells(kind: str) -> list:
    cells = [md("## Partie 0 · Vérifier tes exercices papier (0A.1 à 0A.8)\n\n"
                "Fais d'abord les exercices ✏️ de `02_exercices.md` **sur papier**, dans ta copie de "
                "`06_mes_reponses.md`. Reporte ensuite chaque réponse ici : **la valeur** que tu as trouvée "
                "(`3`, `\"float\"`, `[1, 2]`, `(4, 6)`…), pas l'expression Python, sinon tu ne vérifies rien. "
                "Les réponses pas encore remplies affichent ⏳.")]
    if kind == "solution":
        cells.append(code(PAPER_CONTEXT))
    for ex_id, letters, title, subs in PAPER:
        cells.append(md(f"**Ex {ex_id} — {title}**"))
        if kind == "exercise":
            lines = [f"# Ex {ex_id}: the values you found on paper"]
            for letter, hint, _, _ in subs:
                var = f"answer_{ex_id.replace('.', '_')}{letter}"
                lines.append(f"{var} = ...  # {letter}) {hint}")
            names = ", ".join(f"answer_{ex_id.replace('.', '_')}{letter}" for letter, *_ in subs)
            lines += ["", f"for letter, answer in zip(\"{letters}\", [{names}]):",
                      f"    wb.check(f\"{ex_id}{{letter}}\", answer)"]
            cells.append(code("\n".join(lines)))
        else:
            lines = []
            for letter, _, expr, options in subs:
                args = f", {options}" if options else ""
                lines.append(f'wb.record("{ex_id}{letter}", {expr}{args})')
            cells.append(code("\n".join(lines), tags=["answer"]))
    return cells


# ---------------------------------------------------------------------------
# Parts A to E (0A.13 to 0A.36)
# ---------------------------------------------------------------------------
PART_A = Part("A", "Prise en main du notebook", "Fiche §100.1. Un notebook garde ses variables en mémoire : "
              "c'est l'ordre d'exécution qui compte.", exercises=[
    Ex("0A.13", "🔮", 1, 10, "Ordre d'exécution des cellules : que vaut `x` ?",
       "comprendre que l'état du noyau dépend de l'ordre dans lequel on exécute les cellules.",
       "fiche §100.1.1", tracks="R, C", hypothesis=True,
       body="""Voici trois cellules, dans cet ordre sur la page :

| Cellule | Code |
|---|---|
| ① | `x = 2` |
| ② | `x = x * 3` |
| ③ | `x = x + 1` |

a) Tu exécutes ①, puis ②, puis ③, puis **encore** ②, puis **encore** ③. Que vaut `x` ?
b) Tu fais ensuite *Redémarrer et tout exécuter* (*Restart and run all*). Que vaut `x` à la fin ?

Écris ton hypothèse **avant** d'exécuter quoi que ce soit, puis réponds dans la cellule suivante.""",
       todo="""prediction_0A_13a = ...  # a) your prediction (a number)
prediction_0A_13b = ...  # b) your prediction after "Restart and run all\"""",
       check="""wb.check("0A.13a", prediction_0A_13a)
wb.check("0A.13b", prediction_0A_13b)""",
       solution="""# a) 2 -> 2 * 3 = 6 -> 6 + 1 = 7 -> 7 * 3 = 21 -> 21 + 1 = 22
# b) restart: memory is empty; "Run all" runs (1), (2), (3) once each -> 7
prediction_0A_13a = 22
prediction_0A_13b = 7""",
       record="""wb.record("0A.13a", prediction_0A_13a, mistakes={"tu as suivi l'ordre de la page, pas l'ordre d'exécution": 7})
wb.record("0A.13b", prediction_0A_13b, mistakes={"après un redémarrage, la mémoire repart de zéro : les exécutions précédentes sont oubliées": 22})""",
       after=[("md", "**Expérience.** Les trois cellules ci-dessous sont ①, ② et ③. Exécute-les **à la main** "
               "dans l'ordre de la question a (① ② ③ ② ③), puis lance la dernière cellule, qui affiche `x`. "
               "Enfin, fais *Redémarrer et tout exécuter* et regarde la valeur affichée."),
              ("code", "x = 2  # cell (1)"), ("code", "x = x * 3  # cell (2)"), ("code", "x = x + 1  # cell (3)"),
              ("code", "print(\"x =\", x)")],
       note="Un notebook qui ne donne le bon résultat que si l'on exécute les cellules « dans le bon désordre » "
            "n'est pas reproductible : avant de le partager, *Restart and run all*."),
])

PART_B_GIVEN = '''# Data for part B, as plain Python lists (pandas comes in part E)
import pandas as pd

penguins = wb.datasets.load_penguins()          # 344 penguins
penguins_raw = wb.datasets.load_penguins_raw()  # the raw version, 17 columns

species_list = penguins["species"].tolist()
island_list = penguins["island"].tolist()
mass_list = [None if pd.isna(m) else int(m) for m in penguins["body_mass_g"]]         # None = missing
flipper_list = [None if pd.isna(f) else int(f) for f in penguins["flipper_length_mm"]]
print(len(species_list), "penguins; first masses:", mass_list[:5])'''

PART_B = Part("B", "Python de base : types, structures de données, contrôle du flux",
              "Fiche §100.2 à §100.4. Tout se fait avec du Python « pur » (listes, dictionnaires…) : "
              "pandas et NumPy arrivent dans les parties D et E. Les données des manchots sont préparées "
              "ci-dessous sous forme de listes ; `None` marque une mesure manquante.",
              given=PART_B_GIVEN, exercises=[
    Ex("0A.14", "🔨", 1, 13, "Nombres et f-strings : la fiche d'un manchot",
       "calculer avec des nombres et mettre en forme un résultat avec une f-string.",
       "Ex 0A.1 · fiche §100.2.1-2", thread="Penguins", tracks="R, M, C",
       body="""Un manchot est décrit par quatre variables (déjà écrites dans la cellule).

a) Sa masse en **kilogrammes**, arrondie à 2 décimales (`round`).
b) Sa **fiche**, construite avec une f-string, exactement au format suivant (exemple pour un autre manchot, un Adelie de Dream de 3400 g à la nageoire de 190 mm) : `Adelie (Dream): 3.4 kg, flipper 19.0 cm`. La masse est en kg avec 1 décimale, la nageoire en **cm** avec 1 décimale.
c) Combien de manchots de cette masse peut-on charger, **en entier**, dans un traîneau limité à 50 kg ?
d) Combien de grammes de charge reste-t-il alors ?""",
       todo="""species, island, flipper_mm, mass_g = "Gentoo", "Biscoe", 217, 5076

mass_kg = ...        # a)
card = ...           # b) an f-string
n_penguins = ...     # c) use // (50 kg = 50_000 g)
remaining_g = ...    # d) use %""",
       check="""wb.check("0A.14a", mass_kg)
wb.check("0A.14b", card)
wb.check("0A.14c", n_penguins)
wb.check("0A.14d", remaining_g)""",
       solution="""species, island, flipper_mm, mass_g = "Gentoo", "Biscoe", 217, 5076

mass_kg = round(mass_g / 1000, 2)
card = f"{species} ({island}): {mass_g / 1000:.1f} kg, flipper {flipper_mm / 10:.1f} cm"
n_penguins = 50_000 // mass_g
remaining_g = 50_000 % mass_g
print(mass_kg, "|", card, "|", n_penguins, "|", remaining_g)""",
       record="""wb.record("0A.14a", mass_kg, decimals=2, mistakes={"arrondis à 2 décimales, pas 1": 5.1})
wb.record("0A.14b", card)
wb.record("0A.14c", n_penguins)
wb.record("0A.14d", remaining_g, mistakes={"c'est le reste qui est demandé, pas la masse des manchots chargés": 45684})"""),

    Ex("0A.15", "🔨", 1, 13, "Chaînes : nettoyer les noms d'espèces de Penguins brut",
       "transformer du texte avec les méthodes des chaînes (`split`, `find`, `replace`, indices).",
       "Ex 0A.14 · fiche §100.2.2", thread="Penguins", tracks="C",
       body="""Dans la version brute du dataset, l'espèce est écrite en toutes lettres : `'Adelie Penguin (Pygoscelis adeliae)'`. La liste `raw_names` contient les trois noms bruts.

Écris deux fonctions :
- `short_name(raw)` renvoie le premier mot : `'Adelie'` ;
- `latin_name(raw)` renvoie le texte **entre les parenthèses** : `'Pygoscelis adeliae'` (indice : `raw.find("(")` et `raw.find(")")`, puis une tranche).

a) `", ".join(short_name(r) for r in raw_names)` (les trois noms courts séparés par une virgule).
b) `latin_name(raw_names[2])`.
c) Le nom latin de la **deuxième** espèce, avec `Pygoscelis` remplacé par `P.` (par exemple `'P. adeliae'` pour la première).""",
       given='''raw_names = penguins_raw["Species"].unique().tolist()
raw_names''',
       todo='''def short_name(raw):
    """Return the first word of a raw species name."""
    raise NotImplementedError("short_name() is not written yet")


def latin_name(raw):
    """Return the text between the parentheses of a raw species name."""
    raise NotImplementedError("latin_name() is not written yet")''',
       check='''with wb.attempt("0A.15"):
    wb.check("0A.15a", ", ".join(short_name(r) for r in raw_names))
    wb.check("0A.15b", latin_name(raw_names[2]))
    wb.check("0A.15c", latin_name(raw_names[1]).replace("Pygoscelis", "P."))''',
       solution='''def short_name(raw):
    """Return the first word of a raw species name."""
    return raw.split()[0]


def latin_name(raw):
    """Return the text between the parentheses of a raw species name."""
    start = raw.find("(")
    end = raw.find(")")
    return raw[start + 1:end]


print([short_name(r) for r in raw_names])
print([latin_name(r) for r in raw_names])''',
       record='''wb.record("0A.15a", ", ".join(short_name(r) for r in raw_names))
wb.record("0A.15b", latin_name(raw_names[2]), mistakes={"garde seulement le texte ENTRE les parenthèses": "(Pygoscelis antarctica)"})
wb.record("0A.15c", latin_name(raw_names[1]).replace("Pygoscelis", "P."))'''),

    Ex("0A.16", "🔨", 1, 13, "Listes : les nageoires de dix manchots",
       "calculer sur une liste avec `min`, `max`, `sum`, `len`, `sorted` et des indices.",
       "Ex 0A.2 · fiche §100.3.1", thread="Penguins", tracks="R, M, C",
       body="""La liste `flippers` contient les longueurs de nageoire (mm) des dix premiers manchots mesurés.

a) L'**étendue** : la plus grande valeur moins la plus petite.
b) La **moyenne**, avec `sum` et `len` (wb.check arrondit à 1 décimale).
c) La liste des **trois plus grandes** valeurs, de la plus grande à la plus petite (`sorted` avec `reverse=True`, puis une tranche).
d) La **médiane** : avec 10 valeurs, c'est la moyenne des 5ᵉ et 6ᵉ valeurs de la liste **triée** (attention : les indices commencent à 0).""",
       given='''flippers = [f for f in flipper_list if f is not None][:10]
flippers''',
       todo='''flipper_range = ...    # a)
flipper_mean = ...     # b)
top3 = ...             # c) a list of 3 values
flipper_median = ...   # d) sorted(flippers)[...] ...''',
       check='''wb.check("0A.16a", flipper_range)
wb.check("0A.16b", flipper_mean)
wb.check("0A.16c", top3)
wb.check("0A.16d", flipper_median)''',
       solution='''flipper_range = max(flippers) - min(flippers)
flipper_mean = sum(flippers) / len(flippers)
top3 = sorted(flippers, reverse=True)[:3]
ordered = sorted(flippers)
flipper_median = (ordered[4] + ordered[5]) / 2   # the 5th and 6th values have indices 4 and 5
print(flipper_range, flipper_mean, top3, flipper_median)''',
       record='''wb.record("0A.16a", flipper_range)
wb.record("0A.16b", flipper_mean, decimals=1)
wb.record("0A.16c", top3, mistakes={"de la plus grande à la plus petite : sorted(..., reverse=True)": sorted(flippers)[:3]})
wb.record("0A.16d", flipper_median, decimals=1)'''),

    Ex("0A.17", "🔨", 1, 13, "Tuples et déballage : renvoyer et échanger plusieurs valeurs",
       "renvoyer plusieurs valeurs sous forme de tuple et les récupérer par déballage.",
       "Ex 0A.16 · fiche §100.3.2", tracks="C",
       body="""a) Écris `min_max(values)`, qui renvoie le **tuple** `(minimum, maximum)` d'une liste. Vérification : `min_max(flippers)`.
b) Avec `mass_known` (les masses connues), récupère le minimum et le maximum par déballage (`low, high = ...`) : que vaut `high - low` ?
c) Échange les valeurs de `first_species` et `second_species` **en une seule ligne**, sans variable intermédiaire. Vérification : `first_species + "/" + second_species`.
d) Avec `head, *middle, tail = flippers`, combien d'éléments contient `middle` ?""",
       given='''mass_known = [m for m in mass_list if m is not None]''',
       todo='''def min_max(values):
    """Return the tuple (smallest value, largest value) of a list."""
    raise NotImplementedError("min_max() is not written yet")


mass_span = ...      # b) high - low, after low, high = min_max(mass_known)

first_species, second_species = "Adelie", "Gentoo"
# c) swap first_species and second_species in ONE line (no temporary variable), then:
swapped = ...        # c) first_species + "/" + second_species, after the swap
n_middle = ...       # d)''',
       check='''with wb.attempt("0A.17"):
    wb.check("0A.17a", min_max(flippers))
wb.check("0A.17b", mass_span)
wb.check("0A.17c", swapped)
wb.check("0A.17d", n_middle)''',
       solution='''def min_max(values):
    """Return the tuple (smallest value, largest value) of a list."""
    return min(values), max(values)


low, high = min_max(mass_known)
mass_span = high - low

first_species, second_species = "Adelie", "Gentoo"
first_species, second_species = second_species, first_species
swapped = first_species + "/" + second_species
head, *middle, tail = flippers
n_middle = len(middle)
print(min_max(flippers), mass_span, first_species, second_species, n_middle)''',
       record='''wb.record("0A.17a", min_max(flippers))
wb.record("0A.17b", mass_span)
wb.record("0A.17c", swapped, mistakes={"les deux variables n'ont pas été échangées": "Adelie/Gentoo"})
wb.record("0A.17d", n_middle)'''),

    Ex("0A.18", "🔨", 1, 15, "Dictionnaires : une fiche par espèce",
       "regrouper des valeurs par clé dans un dictionnaire, puis en tirer une statistique.",
       "Ex 0A.16 · fiche §100.3.3", thread="Penguins", tracks="R, M, C",
       body="""On veut, pour chaque espèce, la liste de ses masses connues, puis la masse moyenne de chaque espèce.

Complète le squelette : pour chaque manchot, ajoute sa masse à la liste de son espèce (crée la liste la première fois qu'une espèce apparaît : `masses_by_species[species] = []`, ou utilise `setdefault`), puis remplis `mean_by_species`.

a) Le nombre de clés de `masses_by_species`.
b) Le nombre de masses connues pour les Adelie.
c) La masse moyenne des Chinstrap (wb.check arrondit à 0 décimale).
d) L'espèce dont la masse moyenne est la plus élevée (une chaîne).""",
       todo='''masses_by_species = {}
for species, mass in zip(species_list, mass_list):
    if mass is None:
        continue
    ...  # TODO: add `mass` to the list of `species` (create the list the first time)

mean_by_species = {}
for species, masses in masses_by_species.items():
    ...  # TODO: store the mean of `masses` in mean_by_species[species]

n_keys = ...          # a)
n_adelie = ...        # b)
chinstrap_mean = ...  # c)
heaviest = ...        # d) a string''',
       check='''wb.check("0A.18a", n_keys)
wb.check("0A.18b", n_adelie)
wb.check("0A.18c", chinstrap_mean)
wb.check("0A.18d", heaviest)''',
       solution='''masses_by_species = {}
for species, mass in zip(species_list, mass_list):
    if mass is None:
        continue
    masses_by_species.setdefault(species, []).append(mass)

mean_by_species = {}
for species, masses in masses_by_species.items():
    mean_by_species[species] = sum(masses) / len(masses)

n_keys = len(masses_by_species)
n_adelie = len(masses_by_species["Adelie"])
chinstrap_mean = mean_by_species["Chinstrap"]
heaviest = None
for species, mean in mean_by_species.items():
    if heaviest is None or mean > mean_by_species[heaviest]:
        heaviest = species
print(n_keys, n_adelie, round(chinstrap_mean), heaviest)''',
       record='''wb.record("0A.18a", n_keys)
wb.record("0A.18b", n_adelie, mistakes={"compte seulement les masses connues (saute les None)": 152})
wb.record("0A.18c", chinstrap_mean, decimals=0)
wb.record("0A.18d", heaviest)'''),

    Ex("0A.19", "🔨", 1, 13, "Ensembles : quelles espèces sur quelles îles ?",
       "dédoublonner et croiser des informations avec des ensembles.",
       "Ex 0A.18 · fiche §100.3.4", thread="Penguins", tracks="C",
       body="""`pairs` est la liste des couples `(espèce, île)` des 344 manchots. La première réponse est donnée en exemple.

a) L'ensemble des îles où vivent des Adelie (déjà écrit).
b) L'ensemble des îles où vivent des Gentoo.
c) L'ensemble des îles où vivent **à la fois** des Adelie et des Chinstrap (opérateur `&`).
d) Le nombre de couples `(espèce, île)` **différents**.""",
       given='''pairs = list(zip(species_list, island_list))
pairs[:3]''',
       todo='''adelie_islands = {island for species, island in pairs if species == "Adelie"}   # a) example
gentoo_islands = ...        # b)
chinstrap_islands = ...     # (needed for c)
both_islands = ...          # c)
n_distinct_pairs = ...      # d)''',
       check='''wb.check("0A.19a", adelie_islands)
wb.check("0A.19b", gentoo_islands)
wb.check("0A.19c", both_islands)
wb.check("0A.19d", n_distinct_pairs)''',
       solution='''adelie_islands = {island for species, island in pairs if species == "Adelie"}
gentoo_islands = {island for species, island in pairs if species == "Gentoo"}
chinstrap_islands = {island for species, island in pairs if species == "Chinstrap"}
both_islands = adelie_islands & chinstrap_islands
n_distinct_pairs = len(set(pairs))
print(sorted(adelie_islands), sorted(gentoo_islands), both_islands, n_distinct_pairs)''',
       record='''wb.record("0A.19a", adelie_islands, ordered=False)
wb.record("0A.19b", gentoo_islands, ordered=False)
wb.record("0A.19c", both_islands, ordered=False, mistakes={"& garde ce qui est commun aux deux ensembles ; | les réunit": adelie_islands | chinstrap_islands})
wb.record("0A.19d", n_distinct_pairs, mistakes={"un ensemble ne garde qu'un exemplaire de chaque couple : utilise set(pairs)": 344})'''),

    Ex("0A.20", "🔨", 1, 13, "Conditions : classer un manchot selon sa masse",
       "écrire une suite de tests `if` / `elif` / `else` dans le bon ordre, en traitant le cas « valeur manquante ».",
       "Ex 0A.14 · fiche §100.4.1", thread="Penguins", tracks="R, M, C",
       body="""Écris `mass_category(mass_g)`, qui renvoie :
- `"unknown"` si `mass_g` vaut `None` (masse manquante) ;
- `"light"` en dessous de 3500 g ;
- `"medium"` de 3500 g (inclus) à 4500 g (exclu) ;
- `"heavy"` à partir de 4500 g.

a) `mass_category(3500)` · b) `mass_category(None)` · c) le nombre de manchots `"heavy"` parmi les 344 · d) le nombre de `"light"`.""",
       todo='''def mass_category(mass_g):
    """Return "unknown", "light", "medium" or "heavy" for a mass in grams (or None)."""
    raise NotImplementedError("mass_category() is not written yet")''',
       check='''with wb.attempt("0A.20"):
    wb.check("0A.20a", mass_category(3500))
    wb.check("0A.20b", mass_category(None))
    categories = [mass_category(m) for m in mass_list]
    wb.check("0A.20c", categories.count("heavy"))
    wb.check("0A.20d", categories.count("light"))''',
       solution='''def mass_category(mass_g):
    """Return "unknown", "light", "medium" or "heavy" for a mass in grams (or None)."""
    if mass_g is None:          # test the missing value FIRST: None < 3500 would raise a TypeError
        return "unknown"
    if mass_g < 3500:
        return "light"
    elif mass_g < 4500:
        return "medium"
    else:
        return "heavy"


categories = [mass_category(m) for m in mass_list]
print(mass_category(3500), mass_category(None), categories.count("heavy"), categories.count("light"))''',
       record='''wb.record("0A.20a", mass_category(3500), mistakes={"3500 g est inclus dans « medium » : le test est mass_g < 3500 pour « light »": "light"})
wb.record("0A.20b", mass_category(None))
wb.record("0A.20c", categories.count("heavy"))
wb.record("0A.20d", categories.count("light"))'''),

    Ex("0A.21", "🔨", 1, 15, "Boucles : `for`, `range`, `enumerate`, `zip` et `while`",
       "choisir la bonne boucle pour parcourir des données ou répéter un calcul.",
       "Ex 0A.3, Ex 0A.20 · fiche §100.4.2", thread="Penguins", tracks="R, M, C",
       body="""a) Avec `zip(species_list, mass_list)` : la masse totale des Chinstrap, en **kg** (saute les masses manquantes ; wb.check arrondit à 2 décimales).
b) Avec `enumerate(mass_list)` : l'**indice** du premier manchot de plus de 6000 g (utilise `break`).
c) Avec `while` : une colonie de 344 manchots grandit de 8 % par an (`n = n * 1.08`). Au bout de combien d'**années entières** dépasse-t-elle 1000 manchots ?
d) Avec `range` : la somme des nombres **pairs** de 0 à 100 inclus.
e) L'indice du premier manchot dont la nageoire est manquante (`None`) dans `flipper_list`.""",
       todo='''chinstrap_total_kg = ...   # a)
first_heavy_index = ...    # b)
years = ...                # c)
even_sum = ...             # d)
first_missing = ...        # e)''',
       check='''wb.check("0A.21a", chinstrap_total_kg)
wb.check("0A.21b", first_heavy_index)
wb.check("0A.21c", years)
wb.check("0A.21d", even_sum)
wb.check("0A.21e", first_missing)''',
       solution='''chinstrap_total_kg = 0
for species, mass in zip(species_list, mass_list):
    if species == "Chinstrap" and mass is not None:
        chinstrap_total_kg += mass / 1000

first_heavy_index = None
for i, mass in enumerate(mass_list):
    if mass is not None and mass > 6000:
        first_heavy_index = i
        break

colony, years = 344, 0
while colony <= 1000:
    colony = colony * 1.08
    years += 1

even_sum = 0
for k in range(0, 101, 2):     # 0, 2, ..., 100 (the stop value 101 is excluded)
    even_sum += k

first_missing = flipper_list.index(None)
print(round(chinstrap_total_kg, 2), first_heavy_index, years, even_sum, first_missing)''',
       record='''wb.record("0A.21a", chinstrap_total_kg, decimals=2)
wb.record("0A.21b", first_heavy_index)
wb.record("0A.21c", years, mistakes={"compte les années jusqu'à DÉPASSER 1000 : la boucle tourne tant que n <= 1000": 13})
wb.record("0A.21d", even_sum, mistakes={"range(0, 100, 2) s'arrête à 98 : la fin est exclue": 2450})
wb.record("0A.21e", first_missing)'''),

    Ex("0A.22", "🔨", 1, 15, "Compréhensions : filtrer et transformer en une ligne",
       "remplacer une boucle d'accumulation par une compréhension de liste, de dictionnaire ou d'ensemble.",
       "Ex 0A.21 · fiche §100.4.3", thread="Penguins", tracks="R, M, C",
       body="""Une ligne par réponse, avec une compréhension.

a) La liste des masses **en kg** des manchots d'au moins 6000 g (`heavy_kg`), puis sa somme (wb.check arrondit à 2 décimales).
b) Le dictionnaire `{espèce: nombre de manchots}` (`counts` : parcours `sorted(set(species_list))` et utilise `species_list.count`), puis sa valeur pour `"Chinstrap"`.
c) La liste des longueurs de nageoire des Gentoo en **cm** (`gentoo_cm`, via `zip(species_list, flipper_list)`, en sautant les `None`), puis sa plus grande valeur.
d) L'ensemble des noms d'îles en **majuscules**.""",
       todo='''heavy_kg = ...            # a) a list comprehension
heavy_kg_total = ...      # a) its sum
counts = ...              # b) a dict comprehension
chinstrap_count = ...     # b) counts["Chinstrap"]
gentoo_cm = ...           # c) a list comprehension
gentoo_cm_max = ...       # c) its largest value
islands_upper = ...       # d) a set comprehension''',
       check='''wb.check("0A.22a", heavy_kg_total)
wb.check("0A.22b", chinstrap_count)
wb.check("0A.22c", gentoo_cm_max)
wb.check("0A.22d", islands_upper)''',
       solution='''heavy_kg = [m / 1000 for m in mass_list if m is not None and m >= 6000]
counts = {sp: species_list.count(sp) for sp in sorted(set(species_list))}
gentoo_cm = [f / 10 for sp, f in zip(species_list, flipper_list) if sp == "Gentoo" and f is not None]
islands_upper = {island.upper() for island in island_list}
heavy_kg_total = sum(heavy_kg)
chinstrap_count = counts["Chinstrap"]
gentoo_cm_max = max(gentoo_cm)
print(heavy_kg, counts, gentoo_cm_max, islands_upper)''',
       record='''wb.record("0A.22a", heavy_kg_total, decimals=2)
wb.record("0A.22b", chinstrap_count)
wb.record("0A.22c", gentoo_cm_max, decimals=1, mistakes={"la réponse est en cm : divise les mm par 10": gentoo_cm_max * 10})
wb.record("0A.22d", islands_upper, ordered=False)'''),
])

PART_C = Part("C", "Fonctions, modules, erreurs et premier module mylearn",
              "Fiche §100.5.1, §100.6.1-2 et §100.11.5.", exercises=[
    Ex("0A.23", "🔨", 1, 15, "Tes premières fonctions : paramètres, valeurs par défaut, `return`",
       "écrire des fonctions avec des paramètres obligatoires et par défaut, qui **renvoient** un résultat.",
       "Ex 0A.22 · fiche §100.5.1", thread="Penguins", tracks="R, M, C",
       body="""Écris deux fonctions :
- `flipper_per_kg(flipper_mm, mass_g)` : la longueur de nageoire (mm) par kilogramme de masse ;
- `describe(species, mass_g, unit="kg", decimals=1)` : une chaîne `"<espèce>: <masse> <unité>"`. Avec `unit="kg"`, la masse est en kg arrondie à `decimals` décimales ; avec `unit="g"`, c'est la masse en grammes, **entière**. Toute autre unité lève une `ValueError`.

a) `flipper_per_kg(181, 3750)` (wb.check arrondit à 2 décimales) · b) `describe("Gentoo", 5076)` · c) `describe("Gentoo", 5076, unit="g")` · d) `describe("Adelie", 3750, decimals=2)`.""",
       todo='''def flipper_per_kg(flipper_mm, mass_g):
    """Return the flipper length (mm) per kilogram of body mass."""
    raise NotImplementedError("flipper_per_kg() is not written yet")


def describe(species, mass_g, unit="kg", decimals=1):
    """Return "<species>: <mass> <unit>" with the mass in kg (rounded) or in g (integer)."""
    raise NotImplementedError("describe() is not written yet")''',
       check='''with wb.attempt("0A.23"):
    wb.check("0A.23a", flipper_per_kg(181, 3750))
    wb.check("0A.23b", describe("Gentoo", 5076))
    wb.check("0A.23c", describe("Gentoo", 5076, unit="g"))
    wb.check("0A.23d", describe("Adelie", 3750, decimals=2))''',
       solution='''def flipper_per_kg(flipper_mm, mass_g):
    """Return the flipper length (mm) per kilogram of body mass."""
    return flipper_mm / (mass_g / 1000)


def describe(species, mass_g, unit="kg", decimals=1):
    """Return "<species>: <mass> <unit>" with the mass in kg (rounded) or in g (integer)."""
    if unit == "kg":
        value = round(mass_g / 1000, decimals)
    elif unit == "g":
        value = int(mass_g)
    else:
        raise ValueError(f"unit must be 'kg' or 'g', not {unit!r}")
    return f"{species}: {value} {unit}"


print(flipper_per_kg(181, 3750), describe("Gentoo", 5076), describe("Gentoo", 5076, unit="g"),
      describe("Adelie", 3750, decimals=2), sep=" | ")''',
       record='''wb.record("0A.23a", flipper_per_kg(181, 3750), decimals=2, mistakes={"convertis d'abord la masse en kg": 181 / 3750})
wb.record("0A.23b", describe("Gentoo", 5076))
wb.record("0A.23c", describe("Gentoo", 5076, unit="g"))
wb.record("0A.23d", describe("Adelie", 3750, decimals=2))'''),

    Ex("0A.24", "🔨", 1, 13, "Importer des modules : `math`, `random`, `statistics` et `Counter`",
       "trouver et utiliser la bonne fonction de la bibliothèque standard plutôt que la réécrire.",
       "Ex 0A.23 · fiche §100.6.1 et §100.3.5", thread="Penguins", tracks="R, M, C",
       body="""Importe ce qu'il faut (`import math`, `import statistics`, `from collections import Counter`…) dans la cellule, puis :

a) Le nombre de mini-lots de 50 manchots nécessaires pour les 344 (`math.ceil`).
b) La médiane des dix nageoires `flippers` (`statistics.median`).
c) L'île la plus fréquente et d) son nombre de manchots (`Counter(island_list).most_common`).
e) L'écart-type de `mass_known` (`statistics.stdev` ; wb.check arrondit à 1 décimale).

La dernière cellule simule dix lancers de dé avec `random` : exécute-la deux fois, puis retire la ligne `random.seed(0)` et recommence. Qu'observes-tu ?""",
       todo='''# TODO: imports

n_batches = ...        # a)
median_flipper = ...   # b)
top_island = ...       # c) a string
top_count = ...        # d)
mass_stdev = ...       # e)''',
       check='''wb.check("0A.24a", n_batches)
wb.check("0A.24b", median_flipper)
wb.check("0A.24c", top_island)
wb.check("0A.24d", top_count)
wb.check("0A.24e", mass_stdev)''',
       solution='''import math
import statistics
from collections import Counter

n_batches = math.ceil(344 / 50)
median_flipper = statistics.median(flippers)
top_island, top_count = Counter(island_list).most_common(1)[0]
mass_stdev = statistics.stdev(mass_known)
print(n_batches, median_flipper, top_island, top_count, round(mass_stdev, 1))''',
       record='''wb.record("0A.24a", n_batches, mistakes={"le dernier lot incomplet compte aussi : math.ceil, pas //": 6})
wb.record("0A.24b", median_flipper, decimals=1)
wb.record("0A.24c", top_island)
wb.record("0A.24d", top_count)
wb.record("0A.24e", mass_stdev, decimals=1)''',
       after=[("code", '''import random

random.seed(0)
print([random.randint(1, 6) for _ in range(10)])''')],
       note="Avec la même graine, `random` redonne exactement la même suite : c'est ce qui rend une expérience reproductible (fiche §100.8.7)."),

    Ex("0A.25", "🔨", 1, 15, "Exceptions : lever une `ValueError` et la rattraper",
       "refuser une entrée invalide avec un message clair, et traiter l'erreur là où on sait quoi faire.",
       "Ex 0A.23 · fiche §100.6.2", tracks="R, M, C",
       body="""Des masses ont été saisies à la main, sous forme de texte (`texts`). Certaines sont invalides.

- `parse_mass(text)` : retire les espaces autour, lève `ValueError("empty mass")` si le texte est vide, convertit en `float` (ce qui lève déjà une `ValueError` pour `"NA"` ou `"3,450"`), et lève `ValueError("negative mass")` si le résultat est négatif. Elle renvoie la masse.
- `parse_all(texts)` : renvoie la liste des masses, avec `None` à la place de chaque texte invalide (`try` / `except ValueError`).

a) Le nombre de masses valides · b) leur somme · c) vérifié automatiquement : `parse_mass("")` doit lever une `ValueError` (la cellule de vérification rattrape l'exception et contrôle son nom).""",
       given='''texts = ["3750", " 3800 ", "", "3,450", "-12", "4.2e3", "NA", "5076"]''',
       todo='''def parse_mass(text):
    """Convert a text to a mass in grams; raise ValueError if it is empty, not a number or negative."""
    raise NotImplementedError("parse_mass() is not written yet")


def parse_all(texts):
    """Parse every text; invalid ones become None."""
    raise NotImplementedError("parse_all() is not written yet")''',
       check='''with wb.attempt("0A.25"):
    parsed = parse_all(texts)
    valid = [m for m in parsed if m is not None]
    wb.check("0A.25a", len(valid))
    wb.check("0A.25b", sum(valid))
    try:
        parse_mass("")
        error_name = "no error"
    except Exception as error:  # noqa: BLE001 - here we want to see whatever is raised
        error_name = type(error).__name__
    wb.check("0A.25c", error_name)''',
       solution='''def parse_mass(text):
    """Convert a text to a mass in grams; raise ValueError if it is empty, not a number or negative."""
    text = text.strip()
    if text == "":
        raise ValueError("empty mass")
    mass = float(text)           # raises ValueError for "NA" or "3,450"
    if mass < 0:
        raise ValueError("negative mass")
    return mass


def parse_all(texts):
    """Parse every text; invalid ones become None."""
    masses = []
    for text in texts:
        try:
            masses.append(parse_mass(text))
        except ValueError:
            masses.append(None)
    return masses


parsed = parse_all(texts)
valid = [m for m in parsed if m is not None]
print(parsed)''',
       record='''wb.record("0A.25a", len(valid), mistakes={"une masse négative doit être refusée": 5})
wb.record("0A.25b", sum(valid), decimals=1, mistakes={"4.2e3 est un nombre valide (4200.0)": sum(valid) - 4200})
wb.record("0A.25c", "ValueError")'''),

    Ex("0A.26", "🔨", 1, 15, "Ton premier module mylearn : `mean` et son test",
       "compléter une fonction de mylearn à partir de son squelette et la valider avec pytest.",
       "Ex 0A.25 · fiche §100.11.5 et §100.11.3", tracks="R, M, C",
       body="""C'est ton premier contact avec **mylearn**, ta librairie.

1. Si ce n'est pas fait : `python tools/start_chapter.py 0A` (ou `--init`) crée `mon_travail/mylearn/`.
2. Ouvre `mon_travail/mylearn/_example.py`. Lis la docstring de `mean` : elle dit ce que la fonction doit faire, y compris refuser une liste vide.
3. Remplace la ligne `raise NotImplementedError(...)` par ton code, **enregistre** le fichier.
4. Redémarre le noyau (le module est chargé au setup), relance la cellule de setup, puis les deux cellules ci-dessous : la première vérifie ta fonction sur les dix nageoires (wb.check arrondit à 2 décimales), la seconde lance les **tests** (`tests/test_example_mylearn.py`) : ils doivent tous passer (`5 passed`).""",
       check='''with wb.attempt("0A.26"):
    wb.check("0A.26", mylearn._example.mean(flippers))

# the tests of your mylearn._example (--impl=learner: YOUR code)
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_example_mylearn.py", "-q", "-p", "no:cacheprovider", "--color=no"],
    cwd=ROOT, capture_output=True, text=True,
)
print(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr)''',
       solution='''print(mylearn._example.mean(flippers))

# the same tests, run on the reference implementation (--impl=ref)
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_example_mylearn.py", "--impl=ref", "-q", "-p", "no:cacheprovider",
     "--color=no"],
    cwd=ROOT, capture_output=True, text=True,
)
print(result.stdout.strip().splitlines()[-1])''',
       record='''wb.record("0A.26", mylearn._example.mean(flippers), decimals=2)''',
       note="Dans le notebook de solutions, `mylearn` est la **référence** : les tests y passent donc. Dans ton notebook, c'est **ton** code qui est testé."),
])

PART_D_GIVEN = '''import numpy as np

numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
table = penguins.dropna()[numeric_cols].values.tolist()   # 333 rows of 4 numbers (plain Python lists)
table[:2]'''

PART_D = Part("D", "Premiers pas NumPy", "Fiche §100.8.1 à §100.8.3 et §100.8.7. On part de `table`, une "
              "liste de 333 listes de 4 mesures (bec, profondeur du bec, nageoire, masse) : les manchots sans "
              "valeur manquante.", given=PART_D_GIVEN, exercises=[
    Ex("0A.27", "📦", 1, 13, "Premiers arrays NumPy : `dtype`, `shape`, `ndim`",
       "créer un array et lire ses attributs de base.",
       "Ex 0A.16 · fiche §100.8.1", thread="Penguins", tracks="R, M, C",
       body="""Crée `X = np.array(table)`, puis donne :

a) sa forme · b) le **nom** de son type d'éléments (`X.dtype.name`, une chaîne) · c) son nombre de dimensions · d) son nombre total d'éléments · e) la 4ᵉ valeur (indice 3) de `np.linspace(0, 1, 11)`.""",
       todo='''X = ...            # the array built from `table`

x_shape = ...      # a)
x_dtype = ...      # b) a string
x_ndim = ...       # c)
x_size = ...       # d)
fourth = ...       # e)''',
       check='''wb.check("0A.27a", x_shape)
wb.check("0A.27b", x_dtype)
wb.check("0A.27c", x_ndim)
wb.check("0A.27d", x_size)
wb.check("0A.27e", fourth)''',
       solution='''X = np.array(table)
x_shape, x_dtype, x_ndim, x_size = X.shape, X.dtype.name, X.ndim, X.size
fourth = np.linspace(0, 1, 11)[3]
print(x_shape, x_dtype, x_ndim, x_size, fourth)''',
       record='''wb.record("0A.27a", x_shape, mistakes={"(lignes, colonnes) : 333 manchots, 4 mesures": (4, 333)})
wb.record("0A.27b", x_dtype)
wb.record("0A.27c", x_ndim)
wb.record("0A.27d", x_size)
wb.record("0A.27e", fourth, decimals=1)'''),

    Ex("0A.28", "📦", 1, 15, "Indexation, tranches et masques booléens",
       "extraire des lignes, des colonnes et des sous-ensembles d'un array avec des indices et des masques.",
       "Ex 0A.27 · fiche §100.8.2", thread="Penguins", tracks="R, M, C",
       body="""Les colonnes de `X` sont, dans l'ordre : bec (mm), profondeur du bec (mm), nageoire (mm), masse (g).

a) La longueur de nageoire du 11ᵉ manchot (indice 10).
b) La masse moyenne des manchots d'indices 100 à 199 inclus (wb.check arrondit à 2 décimales).
c) Le nombre de manchots dont le bec dépasse 45 mm **et** la nageoire mesure moins de 200 mm.
d) La proportion de manchots d'**au moins** 4500 g (4500 inclus, comme « heavy » en 0A.20 ; moyenne d'un masque ; wb.check arrondit à 3 décimales).""",
       todo='''flipper_10 = ...        # a)
mean_mass_100s = ...    # b)
n_long_bill = ...       # c) combine two masks with & (parentheses!)
share_heavy = ...       # d)''',
       check='''wb.check("0A.28a", flipper_10)
wb.check("0A.28b", mean_mass_100s)
wb.check("0A.28c", n_long_bill)
wb.check("0A.28d", share_heavy)''',
       solution='''flipper_10 = X[10, 2]
mean_mass_100s = X[100:200, 3].mean()          # 100 to 199 inclusive: the stop 200 is excluded
n_long_bill = ((X[:, 0] > 45) & (X[:, 2] < 200)).sum()
share_heavy = (X[:, 3] >= 4500).mean()
print(flipper_10, mean_mass_100s, n_long_bill, share_heavy)''',
       record='''wb.record("0A.28a", flipper_10, decimals=1)
wb.record("0A.28b", mean_mass_100s, decimals=2, mistakes={"199 est inclus : la tranche s'arrête à 200 (exclu)": X[100:199, 3].mean()})
wb.record("0A.28c", n_long_bill)
wb.record("0A.28d", share_heavy, decimals=3, mistakes={
    "c'est une proportion (entre 0 et 1), pas un nombre de manchots": float((X[:, 3] >= 4500).sum()),
    "« au moins 4500 g » inclut 4500 : >= et non >": (X[:, 3] > 4500).mean()})'''),

    Ex("0A.29", "🔮", 1, 10, "Vue ou copie : qui est modifié ?",
       "prévoir quand modifier un morceau d'array modifie l'array d'origine.",
       "Ex 0A.28 · fiche §100.8.2", tracks="C", hypothesis=True,
       body="""```python
a = np.arange(6)
b = a[2:5]
b[0] = 100
c = a[[0, 1]]
c[0] = -1
```

Sans exécuter : a) que vaut `a[2]` à la fin ? b) que vaut `a[0]` ? c) que vaut `a.sum()` ?

Écris ton hypothèse, réponds, puis exécute la cellule « Expérience » pour vérifier.""",
       todo='''prediction_0A_29a = ...   # a[2]
prediction_0A_29b = ...   # a[0]
prediction_0A_29c = ...   # a.sum()''',
       check='''wb.check("0A.29a", prediction_0A_29a)
wb.check("0A.29b", prediction_0A_29b)
wb.check("0A.29c", prediction_0A_29c)''',
       solution='''a = np.arange(6)
b = a[2:5]       # a slice: a VIEW on a
b[0] = 100       # changes a[2]
c = a[[0, 1]]    # a list of indices: a COPY
c[0] = -1        # does not change a
prediction_0A_29a, prediction_0A_29b, prediction_0A_29c = a[2], a[0], a.sum()
print(a, a.sum())''',
       record='''wb.record("0A.29a", prediction_0A_29a, mistakes={"une tranche est une VUE : la modifier modifie a": 2})
wb.record("0A.29b", prediction_0A_29b, mistakes={"une liste d'indices crée une COPIE : a n'est pas modifié": -1})
wb.record("0A.29c", prediction_0A_29c)''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", '''a = np.arange(6)
b = a[2:5]
b[0] = 100
c = a[[0, 1]]
c[0] = -1
print("a =", a, "| a.sum() =", a.sum(), "| shares memory with b:", np.shares_memory(a, b))''')]),

    Ex("0A.30", "📦", 1, 15, "Calcul vectorisé : unités, normalisation, fonctions universelles",
       "transformer des colonnes entières d'un coup, sans boucle.",
       "Ex 0A.28 · fiche §100.8.3", thread="Penguins", tracks="R, M, C",
       body="""Sans aucune boucle :

a) La masse moyenne des manchots de `X`, en **kg** (wb.check arrondit à 3 décimales).
b) La **normalisation min-max** des nageoires, $(f - f_{min}) / (f_{max} - f_{min})$, qui ramène les valeurs entre 0 et 1 : la valeur du premier manchot (3 décimales).
c) Le rapport longueur / profondeur du bec de chaque manchot : sa valeur maximale (2 décimales).
d) Avec `np.where`, l'étiquette `"heavy"` (au moins 4500 g, comme en 0A.20) ou `"not heavy"` de chaque manchot : combien de `"heavy"` ?""",
       todo='''mean_kg = ...          # a)
flipper_scaled = ...   # b) an array of 333 values between 0 and 1
first_scaled = ...     # b) its first value
bill_ratio = ...       # c) an array of 333 ratios
max_ratio = ...        # c) its largest value
labels = ...           # d) an array of 333 strings
n_heavy = ...          # d) how many "heavy"''',
       check='''wb.check("0A.30a", mean_kg)
wb.check("0A.30b", first_scaled)
wb.check("0A.30c", max_ratio)
wb.check("0A.30d", n_heavy)''',
       solution='''mean_kg = (X[:, 3] / 1000).mean()
f = X[:, 2]
flipper_scaled = (f - f.min()) / (f.max() - f.min())
bill_ratio = X[:, 0] / X[:, 1]
labels = np.where(X[:, 3] >= 4500, "heavy", "not heavy")
first_scaled, max_ratio, n_heavy = flipper_scaled[0], bill_ratio.max(), (labels == "heavy").sum()
print(round(mean_kg, 3), flipper_scaled[:3].round(3), max_ratio.round(2), n_heavy)''',
       record='''wb.record("0A.30a", mean_kg, decimals=3, mistakes={"utilise la colonne des masses de X (333 manchots complets), pas mass_known": sum(mass_known) / len(mass_known) / 1000})
wb.record("0A.30b", first_scaled, decimals=3)
wb.record("0A.30c", max_ratio, decimals=2)
wb.record("0A.30d", n_heavy, mistakes={"« heavy » commence à 4500 g inclus : >= et non >": int((X[:, 3] > 4500).sum())})'''),

    Ex("0A.31", "📦", 1, 15, "Aléatoire reproductible : `default_rng`, graine, `permutation`, `choice`",
       "tirer au hasard de façon reproductible avec un générateur NumPy.",
       "Ex 0A.27 · fiche §100.8.7", thread="synth", tracks="R, C",
       body="""a) Avec `rng = np.random.default_rng(2026)`, simule 10 000 lancers de dé (`rng.integers(1, 7, size=10_000)`) : la **proportion** de 6 (3 décimales). Est-elle proche de ce que tu attendais ?
b) `np.random.default_rng(0).permutation(5)`.
c) Trois indices de manchots tirés **sans remise** parmi 333 : `np.random.default_rng(1).choice(333, size=3, replace=False)`.
d) Deux générateurs créés avec la graine 7 donnent-ils le même premier nombre (`.random()`) ? (`True` ou `False`)
e) Tire 50 manchots sans remise avec `np.random.default_rng(42).choice(len(X), size=50, replace=False)` (des indices de lignes de `X`), puis calcule la moyenne de leurs nageoires (1 décimale).""",
       todo='''share_six = ...        # a)
perm = ...             # b)
picked = ...           # c)
same_first = ...       # d) True or False, computed
sample_mean = ...      # e)''',
       check='''wb.check("0A.31a", share_six)
wb.check("0A.31b", perm)
wb.check("0A.31c", picked)
wb.check("0A.31d", same_first)
wb.check("0A.31e", sample_mean)''',
       solution='''rng = np.random.default_rng(2026)
rolls = rng.integers(1, 7, size=10_000)       # 7 is excluded: faces 1 to 6
share_six = (rolls == 6).mean()               # close to 1/6 = 0.167
perm = np.random.default_rng(0).permutation(5)
picked = np.random.default_rng(1).choice(333, size=3, replace=False)
same_first = np.random.default_rng(7).random() == np.random.default_rng(7).random()
idx = np.random.default_rng(42).choice(len(X), size=50, replace=False)
sample_mean = X[idx, 2].mean()
print(share_six, perm, picked, same_first, round(sample_mean, 1))''',
       record='''wb.record("0A.31a", share_six, decimals=3)
wb.record("0A.31b", perm)
wb.record("0A.31c", picked)
wb.record("0A.31d", same_first)
wb.record("0A.31e", sample_mean, decimals=1, mistakes={
    "permutation(...)[:50] est aussi un tirage sans remise, mais il n'utilise pas le hasard comme choice : utilise choice, comme demandé":
    X[np.random.default_rng(42).permutation(len(X))[:50], 2].mean()})''',
       note="La proportion de 6 est proche de 1/6 ≈ 0,167 sans être exactement égale : c'est la fluctuation d'échantillonnage (ch. 2)."),
])

PART_E_GIVEN = '''import matplotlib.pyplot as plt

numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
penguins = wb.datasets.load_penguins()   # the same table as data/penguins.csv, used from 0A.33 on'''

PART_E = Part("E", "Premiers pas pandas et matplotlib", "Fiche §100.9 et §100.10.1. On lit maintenant le "
              "fichier avec pandas, comme on le fera dans tout le workbook.", given=PART_E_GIVEN, exercises=[
    Ex("0A.32", "📦", 1, 13, "Premier contact avec Penguins : `read_csv`, `head`, `info`, `describe`",
       "lire un fichier CSV et en faire le premier état des lieux.",
       "Ex 0A.24 · fiche §100.9.1", thread="Penguins", tracks="R, M, C",
       body="""Lis `data/penguins.csv` avec `pd.read_csv` (le chemin `wb.datasets.data_dir() / "penguins.csv"` marche partout), regarde `df.head()`, `df.info()` et `df.describe()`, puis donne :

a) la forme du DataFrame · b) le nombre de valeurs **non manquantes** de la colonne `sex` (lis-le dans `info`) · c) la masse moyenne (wb.check arrondit à l'unité) · d) la plus grande longueur de nageoire · e) le nombre de colonnes numériques résumées par `describe`.""",
       todo='''df = ...              # pd.read_csv(...)

df_shape = ...        # a)
sex_non_null = ...    # b)
mean_mass = ...       # c)
max_flipper = ...     # d)
n_numeric = ...       # e)''',
       check='''wb.check("0A.32a", df_shape)
wb.check("0A.32b", sex_non_null)
wb.check("0A.32c", mean_mass)
wb.check("0A.32d", max_flipper)
wb.check("0A.32e", n_numeric)''',
       solution='''df = pd.read_csv(wb.datasets.data_dir() / "penguins.csv")
df.info()
summary = df.describe()
df_shape = df.shape
sex_non_null = df["sex"].count()        # count() ignores missing values
mean_mass = summary.loc["mean", "body_mass_g"]
max_flipper = summary.loc["max", "flipper_length_mm"]
n_numeric = summary.shape[1]
summary.round(1)''',
       record='''wb.record("0A.32a", df_shape)
wb.record("0A.32b", sex_non_null, mistakes={"compte les valeurs NON manquantes, pas toutes les lignes": 344})
wb.record("0A.32c", mean_mass, decimals=0)
wb.record("0A.32d", max_flipper, decimals=0)
wb.record("0A.32e", n_numeric, mistakes={"year est aussi une colonne numérique": 4})'''),

    Ex("0A.33", "📦", 1, 15, "Sélectionner : colonnes, `loc`, `iloc` et filtres",
       "sélectionner des lignes et des colonnes par étiquette, par position et par condition.",
       "Ex 0A.32, Ex 0A.4 · fiche §100.9.2", thread="Penguins", tracks="R, M, C",
       body="""Travaille sur `penguins` (le même tableau que ton `df`).

a) L'île du manchot d'index 100 (`loc`).
b) L'espèce du **dernier** manchot (`iloc`).
c) Le nombre de Gentoo de plus de 5500 g.
d) La longueur moyenne du bec des manchots de l'île Dream (2 décimales).
e) La masse moyenne des Adelie femelles (1 décimale).""",
       todo='''island_100 = ...          # a)
last_species = ...        # b)
n_big_gentoo = ...        # c) conditions combined with & and parentheses
dream_bill = ...          # d)
adelie_female_mass = ...  # e)''',
       check='''wb.check("0A.33a", island_100)
wb.check("0A.33b", last_species)
wb.check("0A.33c", n_big_gentoo)
wb.check("0A.33d", dream_bill)
wb.check("0A.33e", adelie_female_mass)''',
       solution='''island_100 = penguins.loc[100, "island"]
last_species = penguins.iloc[-1]["species"]
n_big_gentoo = len(penguins[(penguins["species"] == "Gentoo") & (penguins["body_mass_g"] > 5500)])
dream_bill = penguins.loc[penguins["island"] == "Dream", "bill_length_mm"].mean()
adelie_female_mass = penguins.loc[(penguins["species"] == "Adelie") & (penguins["sex"] == "female"), "body_mass_g"].mean()
print(island_100, last_species, n_big_gentoo, round(dream_bill, 2), round(adelie_female_mass, 1))''',
       record='''wb.record("0A.33a", island_100)
wb.record("0A.33b", last_species)
wb.record("0A.33c", n_big_gentoo)
wb.record("0A.33d", dream_bill, decimals=2)
wb.record("0A.33e", adelie_female_mass, decimals=1)'''),

    Ex("0A.34", "📦", 1, 15, "Valeurs manquantes et doublons : `isna`, `dropna`, `fillna`, `duplicated`",
       "repérer, compter et traiter les valeurs manquantes et les lignes en double.",
       "Ex 0A.33 · fiche §100.9.3", thread="Penguins", tracks="R, M, C",
       body="""a) Le nombre **total** de valeurs manquantes dans `penguins`.
b) Le nombre de lignes après `dropna()`.
c) Le nombre de lignes après `dropna(subset=["body_mass_g"])`.
d) Après avoir remplacé les sexes manquants par `"unknown"` (`fillna`), combien de `"unknown"` ?
e) `doubled` contient `penguins` plus 5 lignes recopiées : combien de lignes signale `duplicated()` ?
f) La moyenne des nageoires après avoir remplacé leurs valeurs manquantes par la **médiane** (2 décimales).""",
       given='''doubled = pd.concat([penguins, penguins.sample(5, random_state=0)], ignore_index=True)''',
       todo='''n_missing = ...          # a)
rows_dropna = ...        # b)
rows_mass_known = ...    # c)
n_unknown = ...          # d)
n_duplicates = ...       # e)
filled_mean = ...        # f)''',
       check='''wb.check("0A.34a", n_missing)
wb.check("0A.34b", rows_dropna)
wb.check("0A.34c", rows_mass_known)
wb.check("0A.34d", n_unknown)
wb.check("0A.34e", n_duplicates)
wb.check("0A.34f", filled_mean)''',
       solution='''n_missing = penguins.isna().sum().sum()           # per column, then in total
rows_dropna = len(penguins.dropna())
rows_mass_known = len(penguins.dropna(subset=["body_mass_g"]))
n_unknown = (penguins["sex"].fillna("unknown") == "unknown").sum()
n_duplicates = doubled.duplicated().sum()
flipper = penguins["flipper_length_mm"]
filled_mean = flipper.fillna(flipper.median()).mean()
print(n_missing, rows_dropna, rows_mass_known, n_unknown, n_duplicates, round(filled_mean, 2))''',
       record='''wb.record("0A.34a", n_missing, mistakes={"additionne sur TOUTES les colonnes : penguins.isna().sum().sum()": 11})
wb.record("0A.34b", rows_dropna)
wb.record("0A.34c", rows_mass_known)
wb.record("0A.34d", n_unknown)
wb.record("0A.34e", n_duplicates)
wb.record("0A.34f", filled_mean, decimals=2)'''),

    Ex("0A.35", "📦", 1, 13, "De pandas à NumPy : construire `X` et `y`",
       "préparer les features `X` et les labels `y` au format attendu par les modèles.",
       "Ex 0A.34, Ex 0A.27 · fiche §100.9.5", thread="Penguins", tracks="R, M, C",
       body="""À partir des manchots **sans valeur manquante** :
- `X` : les 4 colonnes de `numeric_cols`, en array NumPy ;
- `y` : l'espèce, en array NumPy.

a) La forme de `X` · b) la forme de `y` · c) le nombre de valeurs distinctes de `y` (`np.unique`) · d) la moyenne de la colonne des nageoires de `X` (1 décimale).""",
       todo='''clean = ...     # the rows without missing values
X = ...         # (n_samples, n_features)
y = ...         # (n_samples,)''',
       check='''if X is ... or y is ...:   # not done yet
    print("⏳ Ex 0A.35 : pas encore fait.")
else:
    wb.check("0A.35a", X.shape)
    wb.check("0A.35b", y.shape)
    wb.check("0A.35c", len(np.unique(y)))
    wb.check("0A.35d", X[:, 2].mean())''',
       solution='''clean = penguins.dropna()
X = clean[numeric_cols].to_numpy()
y = clean["species"].to_numpy()
print(X.shape, y.shape, np.unique(y), X[:, 2].mean().round(1))''',
       record='''only_numeric = penguins.dropna(subset=numeric_cols)   # a frequent mistake: keeps the rows where only sex is missing
wb.record("0A.35a", X.shape, mistakes={"enlève TOUTES les lignes incomplètes, même celles où seul sex manque : dropna() sans subset": only_numeric[numeric_cols].shape})
wb.record("0A.35b", y.shape, mistakes={"y est un vecteur : une seule dimension": (333, 1),
                                       "enlève TOUTES les lignes incomplètes, même celles où seul sex manque : dropna() sans subset": (len(only_numeric),)})
wb.record("0A.35c", len(np.unique(y)))
wb.record("0A.35d", X[:, 2].mean(), decimals=1, mistakes={"enlève TOUTES les lignes incomplètes, même celles où seul sex manque : dropna() sans subset": only_numeric["flipper_length_mm"].mean()})'''),

    Ex("0A.36", "📦", 1, 15, "Premiers graphiques : `plot`, `scatter`, `hist`",
       "tracer les trois graphiques de base, avec des axes nommés et une légende.",
       "Ex 0A.35 · fiche §100.10.1", thread="Penguins", tracks="R, C",
       body="""Trace, avec `fig, ax = plt.subplots()` à chaque fois :

1. l'**histogramme** des masses (25 classes : `bins=25`), axes nommés avec leurs unités ;
2. le **nuage de points** longueur du bec (x) contre profondeur du bec (y), **une couleur par espèce**, avec une légende ;
3. la **courbe** de la masse moyenne par année, une courbe par espèce (le tableau `yearly` est calculé pour toi).

Vérifie ensuite toi-même (réponses dans les solutions) : quelle espèce a le bec le **moins** profond ? La masse moyenne a-t-elle beaucoup changé entre 2007 et 2009 ?""",
       given='''yearly = penguins.groupby(["year", "species"])["body_mass_g"].mean().unstack()
yearly.round(0)''',
       todo='''# 1. histogram of body_mass_g
# 2. scatter bill_length_mm vs bill_depth_mm, one color per species (loop over penguins.groupby("species"))
# 3. one line per species: yearly[species] against yearly.index
''',
       solution='''fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(penguins["body_mass_g"].dropna(), bins=25)
ax.set_xlabel("body mass (g)")
ax.set_ylabel("number of penguins")
ax.set_title("Distribution of body mass")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
for species, group in penguins.groupby("species"):
    ax.scatter(group["bill_length_mm"], group["bill_depth_mm"], s=14, label=species)
ax.set_xlabel("bill length (mm)")
ax.set_ylabel("bill depth (mm)")
ax.set_title("Bill shape by species")
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(6, 3.5))
for species in yearly.columns:
    ax.plot(yearly.index, yearly[species], marker="o", label=species)
ax.set_xticks(yearly.index)
ax.set_xlabel("year")
ax.set_ylabel("mean body mass (g)")
ax.set_title("Mean body mass per year")
ax.legend()
plt.show()''',
       note="Les Gentoo ont le bec le moins profond (≈ 15 mm, contre ≈ 18,4 mm pour les Adelie et les Chinstrap) ; "
            "d'une année à l'autre, la masse moyenne de chaque espèce varie de 120 g au plus, moins de 3 %."),
])

# ---------------------------------------------------------------------------
# Part F (0A.37 to 0A.50): intermediate and advanced Python
# ---------------------------------------------------------------------------
PART_F_GIVEN = '''# Data and modules for part F
import json
import math
import pickle
import re
import tempfile
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
import heapq

import numpy as np
import pandas as pd

numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
penguins = wb.datasets.load_penguins()
penguins_raw = wb.datasets.load_penguins_raw()
clean = penguins.dropna()
work_dir = Path(tempfile.mkdtemp(prefix="wb_0A_"))   # a scratch folder for the files you write here
print("scratch folder:", work_dir)'''

PART_F = Part("F", "Python intermédiaire et avancé",
              "Fiche §100.5.2 à §100.7.4. Les exercices montent d'un cran (★★) : moins de squelette, plus de "
              "fonctions à écrire entièrement. Les fichiers que tu écris ici vont dans un dossier temporaire du système "
              "(`work_dir`), à l'écart de tes fichiers et du dépôt.", given=PART_F_GIVEN, exercises=[
    Ex("0A.37", "🐛", 2, 15, "Lire un traceback : cinq bugs de débutant",
       "lire un traceback en partant de la dernière ligne, nommer l'erreur et corriger le code.",
       "Ex 0A.25 · fiche §100.6.2, §100.2.3, §100.5.1", tracks="C",
       body="""Chacune des cinq fonctions ci-dessous contient **un** bug. La cellule fournie les appelle une par une et affiche le **traceback** de chaque erreur, sans arrêter le notebook.

Ce qu'elles devraient faire :
- `total_mass_kg(texts)` : la masse totale, en kg, de masses lues comme **texte** (`["3750", " 3800", "3250"]`) ;
- `last_flipper(flippers)` : la dernière valeur d'une liste ;
- `island_count(counts, island)` : l'effectif d'une île, quelle que soit la façon d'écrire son nom (`"torgersen"`, `"DREAM"`…) ;
- `shout(species)` : le nom de l'espèce en majuscules ;
- `flipper_cm(flipper_mm)` : la longueur en cm, avec la constante `MM_PER_CM`.

a) Lis chaque traceback **en commençant par la dernière ligne**, et range les noms des cinq exceptions, dans l'ordre, dans la liste `error_names` (par exemple `["ValueError", ...]`).
b) à f) Réécris les cinq fonctions **corrigées** dans la cellule suivante (même nom) : la cellule de vérification teste chacune.""",
       given='''MM_PER_CM = 10
island_counts = {"Biscoe": 168, "Dream": 124, "Torgersen": 52}


def total_mass_kg(texts):
    return sum(texts) / 1000


def last_flipper(flippers):
    return flippers[len(flippers)]


def island_count(counts, island):
    return counts[island]


def shout(species):
    return species.uppper()


def flipper_cm(flipper_mm):
    return flipper_mm / MM_PER_CMS


def diagnose(func, *args):
    """Call func(*args); print the traceback of the error, if any, without stopping the notebook."""
    print(f"----- {func.__name__}{args}")
    try:
        result = func(*args)
    except Exception:  # noqa: BLE001 - here we want to see every error
        traceback.print_exc(file=sys.stdout)
    else:
        print("no error:", repr(result))


diagnose(total_mass_kg, ["3750", " 3800", "3250"])
diagnose(last_flipper, [181, 186, 195])
diagnose(island_count, island_counts, "torgersen")
diagnose(shout, "gentoo")
diagnose(flipper_cm, 217)''',
       todo='''error_names = ...   # a) the five exception names, in order: ["...", "...", "...", "...", "..."]


def total_mass_kg(texts):
    raise NotImplementedError("total_mass_kg() is not fixed yet")


def last_flipper(flippers):
    raise NotImplementedError("last_flipper() is not fixed yet")


def island_count(counts, island):
    raise NotImplementedError("island_count() is not fixed yet")


def shout(species):
    raise NotImplementedError("shout() is not fixed yet")


def flipper_cm(flipper_mm):
    raise NotImplementedError("flipper_cm() is not fixed yet")''',
       check='''wb.check("0A.37a", error_names if error_names is ... or isinstance(error_names, str) else ", ".join(error_names))
with wb.attempt("0A.37b"):
    wb.check("0A.37b", total_mass_kg(["3750", " 3800", "3250"]))
with wb.attempt("0A.37c"):
    wb.check("0A.37c", last_flipper([181, 186, 195]))
with wb.attempt("0A.37d"):
    wb.check("0A.37d", island_count(island_counts, "torgersen") + island_count(island_counts, "DREAM"))
with wb.attempt("0A.37e"):
    wb.check("0A.37e", shout("gentoo") == "GENTOO")
with wb.attempt("0A.37f"):
    wb.check("0A.37f", flipper_cm(217))''',
       solution='''error_names = ["TypeError", "IndexError", "KeyError", "AttributeError", "NameError"]


def total_mass_kg(texts):
    return sum(float(text) for text in texts) / 1000     # texts must become numbers first


def last_flipper(flippers):
    return flippers[-1]                                  # the last index is len - 1


def island_count(counts, island):
    return counts[island.strip().capitalize()]           # "torgersen" -> "Torgersen"


def shout(species):
    return species.upper()                               # the method is upper, not uppper


def flipper_cm(flipper_mm):
    return flipper_mm / MM_PER_CM                        # the constant is MM_PER_CM


print(total_mass_kg(["3750", " 3800", "3250"]), last_flipper([181, 186, 195]),
      island_count(island_counts, "torgersen"), shout("gentoo"), flipper_cm(217))''',
       record='''wb.record("0A.37a", ", ".join(error_names), mistakes={
    "l'ordre compte : dans l'ordre des fonctions, de total_mass_kg à flipper_cm": "IndexError, TypeError, KeyError, AttributeError, NameError"})
wb.record("0A.37b", total_mass_kg(["3750", " 3800", "3250"]), decimals=2,
          mistakes={"le résultat est en kg : divise par 1000": 10800.0})
wb.record("0A.37c", last_flipper([181, 186, 195]))
wb.record("0A.37d", island_count(island_counts, "torgersen") + island_count(island_counts, "DREAM"))
wb.record("0A.37e", shout("gentoo") == "GENTOO")
wb.record("0A.37f", flipper_cm(217), decimals=1)''',
       note="Les cinq erreurs à reconnaître d'un coup d'œil : `TypeError` (mauvais type, ici du texte additionné à un "
            "nombre), `IndexError` (indice trop grand), `KeyError` (clé absente), `AttributeError` (méthode mal "
            "écrite) et `NameError` (nom inconnu)."),

    Ex("0A.38", "🔨", 2, 20, "Lire penguins.csv comme un simple fichier texte (`pathlib`, `with`)",
       "lire et écrire un fichier texte ligne par ligne, sans pandas.",
       "Ex 0A.21, Ex 0A.24 · fiche §100.6.3", thread="Penguins", tracks="C",
       body="""Avant pandas, un CSV n'est qu'un **fichier texte** : une ligne d'en-tête, puis une ligne par manchot, les valeurs séparées par des virgules, et `NA` pour une valeur manquante. C'est ce que `pd.read_csv` lit pour toi ; savoir le faire à la main sert le jour où un fichier est mal formé.

Écris trois fonctions, **sans pandas** :
- `read_rows(path)` : ouvre le fichier avec `with open(path, encoding="utf-8") as f:`, lit la première ligne (l'en-tête), puis toutes les suivantes ; renvoie le couple `(header, rows)` : `header` est la liste des noms de colonnes, `rows` la liste des lignes de données, chacune découpée en liste de chaînes (`line.strip().split(",")`) ;
- `column_values(header, rows, name)` : la liste des valeurs **connues** de la colonne `name`, converties en `float` (on saute les `"NA"`) ; `header.index(name)` donne la position de la colonne ;
- `write_rows(path, header, rows)` : écrit l'en-tête puis les lignes dans un fichier CSV (les valeurs recollées par `",".join(...)`, une ligne par manchot).

Vérifications : a) le nombre de lignes de données · b) le nombre de masses manquantes · c) la moyenne des longueurs de nageoire connues (2 décimales) · d) le nombre de lignes du fichier `gentoo.csv` écrit avec les seules lignes des Gentoo, en-tête compris (relu avec `Path.read_text().splitlines()`) · e) relire `gentoo.csv` avec `read_rows` redonne-t-il exactement les mêmes lignes ?""",
       given='''csv_path = wb.datasets.data_dir() / "penguins.csv"
print(csv_path.name, csv_path.exists())''',
       todo='''def read_rows(path):
    """Return (header, rows): the column names and the data lines split on commas."""
    raise NotImplementedError("read_rows() is not written yet")


def column_values(header, rows, name):
    """Return the known values of column `name` as floats ("NA" is skipped)."""
    raise NotImplementedError("column_values() is not written yet")


def write_rows(path, header, rows):
    """Write the header, then one line per row, to a CSV file."""
    raise NotImplementedError("write_rows() is not written yet")''',
       check='''with wb.attempt("0A.38"):
    header, rows = read_rows(csv_path)
    wb.check("0A.38a", len(rows))
    wb.check("0A.38b", len(rows) - len(column_values(header, rows, "body_mass_g")))
    flippers_known = column_values(header, rows, "flipper_length_mm")
    wb.check("0A.38c", sum(flippers_known) / len(flippers_known))
    gentoo_rows = [row for row in rows if row[0] == "Gentoo"]
    write_rows(work_dir / "gentoo.csv", header, gentoo_rows)
    wb.check("0A.38d", len((work_dir / "gentoo.csv").read_text(encoding="utf-8").splitlines()))
    wb.check("0A.38e", read_rows(work_dir / "gentoo.csv") == (header, gentoo_rows))''',
       solution='''def read_rows(path):
    """Return (header, rows): the column names and the data lines split on commas."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        rows = [line.strip().split(",") for line in f if line.strip()]
    return header, rows


def column_values(header, rows, name):
    """Return the known values of column `name` as floats ("NA" is skipped)."""
    j = header.index(name)
    return [float(row[j]) for row in rows if row[j] != "NA"]


def write_rows(path, header, rows):
    """Write the header, then one line per row, to a CSV file."""
    lines = [",".join(header)] + [",".join(row) for row in rows]
    Path(path).write_text("\\n".join(lines) + "\\n", encoding="utf-8")


header, rows = read_rows(csv_path)
flippers_known = column_values(header, rows, "flipper_length_mm")
gentoo_rows = [row for row in rows if row[0] == "Gentoo"]
write_rows(work_dir / "gentoo.csv", header, gentoo_rows)
print(header)
print(len(rows), "rows;", rows[0])
print((work_dir / "gentoo.csv").read_text(encoding="utf-8").splitlines()[:2])''',
       record='''wb.record("0A.38a", len(rows), mistakes={"l'en-tête n'est pas une ligne de données": 345})
wb.record("0A.38b", len(rows) - len(column_values(header, rows, "body_mass_g")))
wb.record("0A.38c", sum(flippers_known) / len(flippers_known), decimals=2,
          mistakes={"divise par le nombre de valeurs CONNUES (les NA sont sautés)": sum(flippers_known) / len(rows)})
wb.record("0A.38d", len((work_dir / "gentoo.csv").read_text(encoding="utf-8").splitlines()),
          mistakes={"compte aussi la ligne d'en-tête": 124})
wb.record("0A.38e", read_rows(work_dir / "gentoo.csv") == (header, gentoo_rows))''',
       note="Le bloc `with` referme le fichier même en cas d'erreur. Le « `\\n` » final est une convention : "
            "la dernière ligne d'un fichier texte se termine elle aussi par un retour à la ligne."),

    Ex("0A.39", "🔨", 2, 20, "Sauvegarder et recharger des résultats : `json` et `pickle`",
       "enregistrer des résultats dans un fichier lisible (JSON) ou binaire (pickle), et les relire.",
       "Ex 0A.38 · fiche §100.6.4, §100.5.6", thread="Penguins", tracks="C",
       body="""La cellule fournie résume Penguins dans un dictionnaire `results`, calculé avec pandas.

a) Essaie `json.dumps(results)` dans un `try` / `except TypeError as error` : range le **nom** de l'exception dans `json_error` (`type(error).__name__`). Regarde le message : quel type de valeur JSON refuse-t-il ? (regarde `type(results["n_missing"]["sex"])`)
b) Écris `to_jsonable(results)`, qui renvoie une **copie** de `results` où chaque nombre NumPy devient un nombre Python (`int(...)` ou `float(...)`, pense à `isinstance(v, np.integer)` et `isinstance(v, np.floating)`) et chaque tuple une liste ; les dictionnaires imbriqués sont convertis de la même façon. Écris ensuite `save_json(obj, path)` (`json.dump(obj, f, indent=2)`, fichier ouvert en `"w"`) et `load_json(path)`. Vérification : la masse moyenne des Gentoo relue dans le fichier (1 décimale).
c) Dans l'objet relu, quel est le **nom** du type de `loaded["shape"]` ?
d) Écris `save_pickle(obj, path)` et `load_pickle(path)` (fichiers ouverts en `"wb"` et `"rb"`) et enregistre `results` **tel quel** : l'objet relu est-il égal à `results` ? (`True` ou `False`)

⚠️ Et si le `.pkl` venait d'Internet ? Relis l'encadré sécurité de la fiche avant de répondre dans ta tête.""",
       given='''results = {
    "dataset": "penguins",
    "n_rows": penguins.shape[0],
    "shape": penguins.shape,
    "species_counts": penguins["species"].value_counts().to_dict(),
    "mean_mass_by_species": penguins.groupby("species")["body_mass_g"].mean().round(1).to_dict(),
    "n_missing": {"sex": penguins["sex"].isna().sum(), "body_mass_g": penguins["body_mass_g"].isna().sum()},
}
results''',
       todo='''json_error = ...   # a) the NAME of the exception raised by json.dumps(results)


def to_jsonable(results):
    """Return a copy of results with NumPy numbers turned into Python numbers and tuples into lists."""
    raise NotImplementedError("to_jsonable() is not written yet")


def save_json(obj, path):
    raise NotImplementedError("save_json() is not written yet")


def load_json(path):
    raise NotImplementedError("load_json() is not written yet")


def save_pickle(obj, path):
    raise NotImplementedError("save_pickle() is not written yet")


def load_pickle(path):
    raise NotImplementedError("load_pickle() is not written yet")''',
       check='''wb.check("0A.39a", json_error)
with wb.attempt("0A.39b"):
    save_json(to_jsonable(results), work_dir / "results.json")
    loaded = load_json(work_dir / "results.json")
    wb.check("0A.39b", loaded["mean_mass_by_species"]["Gentoo"])
    wb.check("0A.39c", type(loaded["shape"]).__name__)
with wb.attempt("0A.39d"):
    save_pickle(results, work_dir / "results.pkl")
    wb.check("0A.39d", load_pickle(work_dir / "results.pkl") == results)''',
       solution='''try:
    json.dumps(results)
    json_error = "no error"
except TypeError as error:
    json_error = type(error).__name__
    print("json.dumps:", error)


def convert(value):
    """A NumPy number becomes a Python number, a tuple a list; anything else is kept."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return [convert(v) for v in value]
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    return value


def to_jsonable(results):
    """Return a copy of results with NumPy numbers turned into Python numbers and tuples into lists."""
    return {key: convert(value) for key, value in results.items()}


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


save_json(to_jsonable(results), work_dir / "results.json")
loaded = load_json(work_dir / "results.json")
save_pickle(results, work_dir / "results.pkl")
print((work_dir / "results.json").read_text(encoding="utf-8")[:230], "...")
print(loaded == results, load_pickle(work_dir / "results.pkl") == results)''',
       record='''wb.record("0A.39a", json_error)
wb.record("0A.39b", loaded["mean_mass_by_species"]["Gentoo"], decimals=1)
wb.record("0A.39c", type(loaded["shape"]).__name__, mistakes={"JSON ne connaît pas les tuples : relu, c'est une liste": "tuple"})
wb.record("0A.39d", load_pickle(work_dir / "results.pkl") == results)''',
       note="`convert` s'appelle elle-même pour les tuples et les dictionnaires imbriqués : c'est une fonction "
            "**récursive** (0A.44). Autre solution courante : `json.dumps(results, default=lambda v: v.item())`. "
            "Le JSON relu n'est pas égal à `results` (`(344, 8)` est devenu `[344, 8]`), le pickle si : il garde "
            "les types Python, mais ne se relit qu'en Python et jamais depuis une source inconnue."),

    Ex("0A.40", "🔨", 2, 26, "Arguments variables : `*args`, `**kwargs` et keyword-only",
       "écrire des fonctions qui acceptent un nombre variable d'arguments, et des arguments uniquement nommés.",
       "Ex 0A.6, Ex 0A.23 · fiche §100.5.2", tracks="C",
       body="""a) Écris `mean_of(*values)`, qui accepte **n'importe quel nombre** d'arguments et renvoie leur moyenne ; sans aucun argument, elle lève une `ValueError`.
b) Écris `make_config(*, lr=0.01, epochs=10, **extra)` : ses paramètres ne se passent que **par leur nom** (le `*` seul), et elle renvoie un dictionnaire qui contient `lr`, `epochs` et tous les réglages supplémentaires reçus dans `extra`.

Vérifications : a) `mean_of(181, 186, 195)` (1 décimale) · b) `mean_of(*masses)`, où l'étoile **déballe** la liste · c) le nombre de clés de `make_config(lr=0.1, batch_size=32)` · d) le nom de l'erreur levée par `make_config(0.1)` · e) avec `settings = {"lr": 0.5, "epochs": 3}`, le produit `lr × epochs` du dictionnaire `make_config(**settings, dropout=0.2)` · f) le nom de l'erreur levée par `mean_of()`.""",
       given='''masses = [3750, 3800, 3250, 3450]
settings = {"lr": 0.5, "epochs": 3}


def error_name(func, *args, **kwargs):
    """The name of the exception raised by func(*args, **kwargs), or "no error"."""
    try:
        func(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as error:  # noqa: BLE001 - we want the name of whatever is raised
        return type(error).__name__
    return "no error"''',
       todo='''def mean_of(*values):
    """Return the mean of any number of values; raise ValueError if there is none."""
    raise NotImplementedError("mean_of() is not written yet")


def make_config(*, lr=0.01, epochs=10, **extra):
    """Return a dict with lr, epochs and every extra setting."""
    raise NotImplementedError("make_config() is not written yet")''',
       check='''with wb.attempt("0A.40"):
    wb.check("0A.40a", mean_of(181, 186, 195))
    wb.check("0A.40b", mean_of(*masses))
    wb.check("0A.40c", len(make_config(lr=0.1, batch_size=32)))
    wb.check("0A.40d", error_name(make_config, 0.1))
    config = make_config(**settings, dropout=0.2)
    wb.check("0A.40e", config["lr"] * config["epochs"])
    wb.check("0A.40f", error_name(mean_of))''',
       solution='''def mean_of(*values):
    """Return the mean of any number of values; raise ValueError if there is none."""
    if len(values) == 0:                 # values is a tuple
        raise ValueError("mean_of() needs at least one value")
    return sum(values) / len(values)


def make_config(*, lr=0.01, epochs=10, **extra):
    """Return a dict with lr, epochs and every extra setting."""
    return {"lr": lr, "epochs": epochs, **extra}


config = make_config(**settings, dropout=0.2)
print(mean_of(181, 186, 195), mean_of(*masses), make_config(lr=0.1, batch_size=32), config)
print(error_name(make_config, 0.1), error_name(mean_of))''',
       record='''wb.record("0A.40a", mean_of(181, 186, 195), decimals=1)
wb.record("0A.40b", mean_of(*masses), decimals=1)
wb.record("0A.40c", len(make_config(lr=0.1, batch_size=32)), mistakes={"lr et epochs sont toujours dans le dictionnaire, en plus des réglages supplémentaires": 1})
wb.record("0A.40d", error_name(make_config, 0.1))
wb.record("0A.40e", config["lr"] * config["epochs"], decimals=1)
wb.record("0A.40f", error_name(mean_of), mistakes={
    "sans argument, la fonction doit refuser l'entrée avec raise ValueError(...)": "ZeroDivisionError",
    "statistics.mean lève sa propre erreur : teste toi-même l'absence d'argument et lève une ValueError": "StatisticsError",
    "sans argument, mean_of doit lever une ValueError, pas renvoyer une valeur": "no error"})''',
       note="`{\"lr\": lr, \"epochs\": epochs, **extra}` déballe un dictionnaire **dans** un autre. Les estimateurs "
            "de scikit-learn rendent leurs réglages *keyword-only* pour la même raison : `LogisticRegression(0.1)` "
            "ne dirait pas de quel réglage il s'agit."),

    Ex("0A.41", "🔨", 2, 20, "Fonctions en argument : `lambda`, `key=` et `Callable`",
       "passer une fonction en argument (`key=`, `lambda`) et écrire une fonction qui en reçoit une.",
       "Ex 0A.40 · fiche §100.5.3", thread="Penguins", tracks="C",
       body="""`penguin_records` est la liste des manchots complets sous forme de tuples `(espèce, île, nageoire, masse)`.

a) Le manchot le plus lourd : `heaviest = max(penguin_records, key=...)`. La vérification affiche son espèce et sa masse.
b) Les trois manchots aux nageoires les plus **courtes** (`sorted(penguin_records, key=...)`, puis une tranche) : `shortest3`.
c) Les noms d'îles triés du plus peuplé au moins peuplé : `sorted(island_sizes, key=island_sizes.get, reverse=True)` (une **méthode** passée comme clé, sans parenthèses).
d) Écris `apply(func, values)`, annotée `func: Callable[[float], float]`, qui renvoie la liste des `func(v)`. Vérification : la somme des masses converties en kg avec `lambda m: m / 1000` (2 décimales).
e) Écris `count_if(predicate, values)`, qui compte les valeurs pour lesquelles `predicate(v)` est vrai. Vérification : le nombre de nageoires d'au moins 200 mm.""",
       given='''penguin_records = list(zip(clean["species"], clean["island"], clean["flipper_length_mm"], clean["body_mass_g"]))
island_sizes = dict(Counter(clean["island"]))
flipper_values = [r[2] for r in penguin_records]
mass_values = [r[3] for r in penguin_records]
print(len(penguin_records), penguin_records[0], island_sizes)''',
       todo='''heaviest = ...        # a) max(penguin_records, key=lambda r: ...)
shortest3 = ...       # b) the 3 records with the shortest flippers
islands_sorted = ...  # c) a list of island names


def apply(func: Callable[[float], float], values: list[float]) -> list[float]:
    """Return [func(v) for each v in values]."""
    raise NotImplementedError("apply() is not written yet")


def count_if(predicate, values):
    """Return how many values v satisfy predicate(v)."""
    raise NotImplementedError("count_if() is not written yet")''',
       check='''wb.check("0A.41a", heaviest if heaviest is ... else f"{heaviest[0]} {heaviest[3]:.0f}")
wb.check("0A.41b", shortest3 if shortest3 is ... else [r[2] for r in shortest3])
wb.check("0A.41c", islands_sorted if islands_sorted is ... or isinstance(islands_sorted, str) else ", ".join(islands_sorted))
with wb.attempt("0A.41d"):
    wb.check("0A.41d", sum(apply(lambda m: m / 1000, mass_values)))
with wb.attempt("0A.41e"):
    wb.check("0A.41e", count_if(lambda f: f >= 200, flipper_values))''',
       solution='''heaviest = max(penguin_records, key=lambda r: r[3])
shortest3 = sorted(penguin_records, key=lambda r: r[2])[:3]
islands_sorted = sorted(island_sizes, key=island_sizes.get, reverse=True)


def apply(func: Callable[[float], float], values: list[float]) -> list[float]:
    """Return [func(v) for each v in values]."""
    return [func(v) for v in values]


def count_if(predicate, values):
    """Return how many values v satisfy predicate(v)."""
    return sum(1 for v in values if predicate(v))


print(heaviest, shortest3, islands_sorted)
print(round(sum(apply(lambda m: m / 1000, mass_values)), 2), count_if(lambda f: f >= 200, flipper_values))''',
       record='''wb.record("0A.41a", f"{heaviest[0]} {heaviest[3]:.0f}")
wb.record("0A.41b", [r[2] for r in shortest3], mistakes={"les plus COURTES : tri croissant, puis les 3 premiers": [231.0, 230.0, 230.0]})
wb.record("0A.41c", ", ".join(islands_sorted), mistakes={"du plus peuplé au moins peuplé : reverse=True": "Torgersen, Dream, Biscoe"})
wb.record("0A.41d", sum(apply(lambda m: m / 1000, mass_values)), decimals=2)
wb.record("0A.41e", count_if(lambda f: f >= 200, flipper_values), mistakes={"« au moins 200 » inclut 200 : >=": sum(1 for f in flipper_values if f > 200)})'''),

    Ex("0A.42", "🔨", 2, 26, "Fermetures : une fabrique de fonctions",
       "écrire des fonctions qui fabriquent et renvoient d'autres fonctions, avec une mémoire (`nonlocal`).",
       "Ex 0A.41 · fiche §100.5.4", tracks="C",
       body="""a) Écris `make_scaler(low, high)`, qui renvoie une **fonction** `scale(x)` calculant $(x - \\text{low}) / (\\text{high} - \\text{low})$ : une normalisation min-max « préréglée » (0A.30), qui se souvient de ses bornes.
b) Écris `make_running_mean()`, qui renvoie une fonction `add(x)` : chaque appel ajoute `x` et renvoie la moyenne de **toutes** les valeurs reçues jusque-là. La fonction intérieure garde un total et un compteur dans les variables de la fonction englobante (`nonlocal`).

Vérifications, avec `scale = make_scaler(172, 231)` (les nageoires extrêmes) : a) `scale(200)` (3 décimales) · b) `[scale(172), scale(231)]` · c) après `running(3750)`, `running(3800)`, la valeur de `running(3250)` · d) une deuxième fabrique indépendante : `other = make_running_mean()`, `other(5000)`, puis `running(4000)` (la première continue sa propre moyenne).""",
       todo='''def make_scaler(low, high):
    """Return a function x -> (x - low) / (high - low)."""
    raise NotImplementedError("make_scaler() is not written yet")


def make_running_mean():
    """Return a function add(x) that returns the mean of every value received so far."""
    raise NotImplementedError("make_running_mean() is not written yet")''',
       check='''with wb.attempt("0A.42"):
    scale = make_scaler(172, 231)
    wb.check("0A.42a", scale(200))
    wb.check("0A.42b", [scale(172), scale(231)])
    running = make_running_mean()
    running(3750)
    running(3800)
    wb.check("0A.42c", running(3250))
    other = make_running_mean()
    other(5000)
    wb.check("0A.42d", running(4000))''',
       solution='''def make_scaler(low, high):
    """Return a function x -> (x - low) / (high - low)."""
    def scale(x):
        return (x - low) / (high - low)
    return scale


def make_running_mean():
    """Return a function add(x) that returns the mean of every value received so far."""
    total = 0.0
    count = 0

    def add(x):
        nonlocal total, count          # without nonlocal: UnboundLocalError
        total += x
        count += 1
        return total / count
    return add


scale = make_scaler(172, 231)
running = make_running_mean()
running(3750)
running(3800)
third = running(3250)
other = make_running_mean()
other(5000)
fourth = running(4000)
print(scale(200), [scale(172), scale(231)], third, fourth)''',
       record='''wb.record("0A.42a", scale(200), decimals=3)
wb.record("0A.42b", [scale(172), scale(231)])
wb.record("0A.42c", third, decimals=1, mistakes={"la moyenne de TOUTES les valeurs reçues, pas seulement de la dernière": 3250.0})
wb.record("0A.42d", fourth, decimals=1,
          mistakes={"chaque appel de make_running_mean crée un total et un compteur NEUFS : other ne touche pas à running": (3750 + 3800 + 3250 + 5000 + 4000) / 5})''',
       note="Sans `nonlocal`, `total += x` crée une variable **locale** à `add` et Python lève "
            "`UnboundLocalError`. Chaque appel de la fabrique crée un nouvel environnement : `running` et `other` ne "
            "partagent rien. Au ch. 18, chaque opération de l'autodifférentiation fabrique ainsi sa propre fonction "
            "de dérivation."),

    Ex("0A.43", "🔮", 2, 15, "Le piège des lambdas créées dans une boucle",
       "prévoir le comportement d'une fermeture créée dans une boucle, et le corriger.",
       "Ex 0A.42 · fiche §100.5.4", tracks="C", hypothesis=True,
       body="""```python
multipliers = [lambda x: x * k for k in range(1, 4)]
results = [f(10) for f in multipliers]
```

a) **Sans exécuter** : que vaut `results` ? Écris ton hypothèse, puis ta prédiction (une liste).
**Expérience** : exécute la cellule qui suit l'exercice. La fonction se souvient-elle de la **valeur** de `k` au moment de sa création, ou de la **variable** `k` ?
b) Construis `fixed`, une liste de trois fonctions qui donnent vraiment `[10, 20, 30]`. Deux corrections classiques : une valeur par défaut, `lambda x, k=k: x * k` (elle est calculée au moment où la fonction est créée), ou une fabrique de fonctions comme en 0A.42.""",
       todo='''prediction_0A_43 = ...   # a) your prediction for results (a list)
fixed = ...              # b) a list of 3 functions giving [10, 20, 30]''',
       check='''wb.check("0A.43a", prediction_0A_43)
wb.check("0A.43b", fixed if fixed is ... else [f(10) for f in fixed])''',
       solution='''multipliers = [lambda x: x * k for k in range(1, 4)]
prediction_0A_43 = [f(10) for f in multipliers]      # every lambda reads k when it is CALLED: k = 3
fixed = [lambda x, k=k: x * k for k in range(1, 4)]  # the default value is computed at creation
print(prediction_0A_43, [f(10) for f in fixed])''',
       record='''wb.record("0A.43a", prediction_0A_43, mistakes={"la lambda lit k au moment où on l'APPELLE, quand la boucle est finie": [10, 20, 30]})
wb.record("0A.43b", [f(10) for f in fixed])''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec ta prédiction."),
              ("code", '''multipliers = [lambda x: x * k for k in range(1, 4)]
print([f(10) for f in multipliers])
print("k is read when the function is called; the loop is over, k =", 3)''')],
       note="Une fermeture garde un accès à la **variable**, pas une photo de sa valeur (fiche §100.5.4). Le même piège "
            "guette les boucles qui créent des callbacks ou des couches en PyTorch."),

    Ex("0A.44", "🔨", 2, 25, "Fonctions récursives : parcourir un arbre de dictionnaires (profondeur, nombre de feuilles)",
       "écrire une fonction récursive avec un cas de base et un appel sur une partie plus petite.",
       "Ex 0A.18, Ex 0A.23 · fiche §100.5.6", thread="Penguins", tracks="R, M, C",
       body="""`colonies` range les effectifs de Penguins dans un **arbre** : continent → région → île → espèce → nombre de manchots. Les nombres sont les **feuilles** de l'arbre ; chaque dictionnaire est un **nœud**.

Écris quatre fonctions **récursives** (cas de base : `tree` n'est pas un dictionnaire, c'est une feuille) :
- `total(tree)` : la somme de toutes les feuilles ;
- `count_leaves(tree)` : le nombre de feuilles ;
- `depth(tree)` : une feuille a la profondeur 0 ; un dictionnaire a la profondeur 1 + la plus grande profondeur de ses enfants ;
- `largest_leaf(tree)` : la plus grande feuille.

Teste d'abord chaque fonction sur un petit arbre, comme `{"a": 1, "b": {"c": 2}}` (total 3, 2 feuilles, profondeur 2), avant le grand.

Vérifications sur `colonies` : a) `total` · b) `count_leaves` · c) `depth` · d) `largest_leaf` · e) `depth(7)` (une feuille seule).""",
       given='''colonies = {
    "Antarctica": {
        "Anvers": {
            "Biscoe": {"Adelie": 44, "Gentoo": 124},
            "Dream": {"Adelie": 56, "Chinstrap": 68},
            "Torgersen": {"Adelie": 52},
        },
    },
}''',
       todo='''def total(tree):
    """Sum of all the leaves (numbers) of a tree of nested dictionaries."""
    raise NotImplementedError("total() is not written yet")


def count_leaves(tree):
    """Number of leaves of the tree."""
    raise NotImplementedError("count_leaves() is not written yet")


def depth(tree):
    """0 for a leaf; 1 + the largest depth of the children for a dictionary."""
    raise NotImplementedError("depth() is not written yet")


def largest_leaf(tree):
    """The largest leaf of the tree."""
    raise NotImplementedError("largest_leaf() is not written yet")''',
       check='''with wb.attempt("0A.44"):
    wb.check("0A.44a", total(colonies))
    wb.check("0A.44b", count_leaves(colonies))
    wb.check("0A.44c", depth(colonies))
    wb.check("0A.44d", largest_leaf(colonies))
    wb.check("0A.44e", depth(7))''',
       solution='''def total(tree):
    """Sum of all the leaves (numbers) of a tree of nested dictionaries."""
    if not isinstance(tree, dict):          # base case: a leaf
        return tree
    return sum(total(child) for child in tree.values())


def count_leaves(tree):
    """Number of leaves of the tree."""
    if not isinstance(tree, dict):
        return 1
    return sum(count_leaves(child) for child in tree.values())


def depth(tree):
    """0 for a leaf; 1 + the largest depth of the children for a dictionary."""
    if not isinstance(tree, dict):
        return 0
    return 1 + max(depth(child) for child in tree.values())


def largest_leaf(tree):
    """The largest leaf of the tree."""
    if not isinstance(tree, dict):
        return tree
    return max(largest_leaf(child) for child in tree.values())


small = {"a": 1, "b": {"c": 2}}
print(total(small), count_leaves(small), depth(small))
print(total(colonies), count_leaves(colonies), depth(colonies), largest_leaf(colonies), depth(7))''',
       record='''wb.record("0A.44a", total(colonies))
wb.record("0A.44b", count_leaves(colonies), mistakes={"compte les FEUILLES (les nombres), pas les îles": 3})
wb.record("0A.44c", depth(colonies), mistakes={"compte chaque niveau de dictionnaire : continent, région, île, espèce": 3})
wb.record("0A.44d", largest_leaf(colonies))
wb.record("0A.44e", depth(7), mistakes={"une feuille seule n'a aucun niveau de dictionnaire : profondeur 0": 1})''',
       note="Un dictionnaire vide ferait planter `max` : dans un vrai code, on traiterait ce cas à part. Tu "
            "retrouveras exactement ce schéma (cas de base, appel sur chaque enfant) pour les arbres de décision du "
            "ch. 13 et l'arbre de calcul de la rétropropagation au ch. 18."),

    Ex("0A.45", "🔨", 2, 26, "Expressions régulières : identifiants et dates de Penguins brut",
       "vérifier un format et extraire des morceaux de texte avec le module `re`.",
       "Ex 0A.15, Ex 0A.38 · fiche §100.6.5", thread="Penguins", tracks="C",
       body="""Dans `penguins_raw`, `Individual ID` identifie un adulte : `N1A1` se lit « nid n° 1, adulte n° 1 ». `Date Egg` est la date de ponte (`2007-11-11`) et `Comments` des remarques libres des chercheurs.

a) Écris `is_valid_id(text)` : `True` si **tout** le texte suit le motif « `N`, un ou plusieurs chiffres, `A`, puis `1` ou `2` » (`re.fullmatch`), `False` sinon. Combien d'identifiants sont valides ?
b) Tes motifs doivent refuser les pièges : `"N12A3"`, `"n1a1"`, `"N1A1 "` (espace final) et `"xN1A1"`. La vérification calcule `any(...)` sur ces pièges : il doit valoir `False`.
c) Écris `nest_number(text)`, qui renvoie le numéro de nid (un `int`) grâce à un **groupe** `( )`. Quel est le plus grand numéro de nid ?
d) Écris `egg_month(date_text)`, qui renvoie le mois (un `int`) d'une date `AAAA-MM-JJ` (trois groupes). Combien de pontes en décembre ?
e) Combien de commentaires parlent du **sexage** (le mot « sex », dans `Sexing` ou `sexing`), en majuscules ou en minuscules ? (`re.search(motif, texte, flags=re.IGNORECASE)` ; la cellule fournit `comments`, les commentaires non vides.)""",
       given='''ids = penguins_raw["Individual ID"].tolist()
dates = penguins_raw["Date Egg"].tolist()
comments = penguins_raw["Comments"].dropna().tolist()
traps = ["N12A3", "n1a1", "N1A1 ", "xN1A1"]
print(ids[:4], dates[:2], len(comments), comments[:2])''',
       todo='''def is_valid_id(text):
    """True if text is exactly N<digits>A1 or N<digits>A2."""
    raise NotImplementedError("is_valid_id() is not written yet")


def nest_number(text):
    """The nest number of an ID such as N12A1 (an int)."""
    raise NotImplementedError("nest_number() is not written yet")


def egg_month(date_text):
    """The month (an int) of a date written YYYY-MM-DD."""
    raise NotImplementedError("egg_month() is not written yet")


n_sexing = ...   # e) how many comments mention "sex" (any case)''',
       check='''with wb.attempt("0A.45a"):
    wb.check("0A.45a", sum(is_valid_id(t) for t in ids))
    wb.check("0A.45b", any(is_valid_id(t) for t in traps))
with wb.attempt("0A.45c"):
    wb.check("0A.45c", max(nest_number(t) for t in ids))
with wb.attempt("0A.45d"):
    wb.check("0A.45d", sum(1 for d in dates if egg_month(d) == 12))
wb.check("0A.45e", n_sexing)''',
       solution='''def is_valid_id(text):
    """True if text is exactly N<digits>A1 or N<digits>A2."""
    return re.fullmatch(r"N\\d+A[12]", text) is not None


def nest_number(text):
    """The nest number of an ID such as N12A1 (an int)."""
    return int(re.fullmatch(r"N(\\d+)A[12]", text).group(1))


def egg_month(date_text):
    """The month (an int) of a date written YYYY-MM-DD."""
    year, month, day = re.fullmatch(r"(\\d{4})-(\\d{2})-(\\d{2})", date_text).groups()
    return int(month)


n_sexing = sum(1 for c in comments if re.search(r"sex", c, flags=re.IGNORECASE))
print([is_valid_id(t) for t in traps], nest_number("N12A1"), egg_month("2007-11-11"), n_sexing)
print([c for c in comments if re.search(r"sex", c, flags=re.IGNORECASE)][:3])''',
       record='''wb.record("0A.45a", sum(is_valid_id(t) for t in ids))
wb.record("0A.45b", any(is_valid_id(t) for t in traps))
wb.record("0A.45c", max(nest_number(t) for t in ids), mistakes={"le groupe est du TEXTE : convertis-le avec int() avant de comparer": 99})
wb.record("0A.45d", sum(1 for d in dates if egg_month(d) == 12))
wb.record("0A.45e", n_sexing, mistakes={"« Sexing » et « sexing » comptent tous les deux : flags=re.IGNORECASE": sum(1 for c in comments if "sex" in c)})''',
       note="`re.search` accepte un piège comme `\"xN1A1\"` (il trouve le motif **dans** le texte) ; `re.fullmatch` "
            "exige que **tout** le texte suive le motif : c'est ce qu'il faut pour valider un format. Sans `int()`, "
            "`max` compare des chaînes : `\"99\" > \"100\"` !"),

    Ex("0A.46", "🔨", 2, 26, "`itertools` et `heapq` : paires de features, grille, top-k",
       "générer des combinaisons et des grilles sans boucles imbriquées, et trouver les k plus grands éléments.",
       "Ex 0A.41 · fiche §100.6.6", thread="Penguins", tracks="C",
       body="""a) Le nombre de **paires** de features parmi les quatre de `numeric_cols` (`combinations`). 🧮 C'est aussi $\\binom{4}{2} = \\frac{4 \\times 3}{2}$.
b) Parmi les paires d'**espèces**, celle dont les masses moyennes sont les plus éloignées : `max(combinations(species_names, 2), key=...)`, avec `mean_mass` fourni (un tuple de deux noms, que la vérification affiche sous la forme `"Espèce1, Espèce2"`).
c) Le nombre de réglages d'une **grille** d'hyperparamètres : learning rate dans `[0.1, 0.01, 0.001]`, taille de lot dans `[16, 32, 64]`, epochs dans `[5, 10]` (`product`).
d) Les trois plus grandes masses (`heapq.nlargest`), dans l'ordre décroissant.
e) Les deux manchots les plus **légers** : `heapq.nsmallest(2, mass_records, key=...)` sur les tuples `(espèce, masse)` ; la vérification affiche leurs espèces sous la forme `"Espèce1, Espèce2"`.""",
       given='''species_names = sorted(clean["species"].unique())
mean_mass = clean.groupby("species")["body_mass_g"].mean().to_dict()
mass_records = list(zip(clean["species"], clean["body_mass_g"]))
print(species_names, {k: round(v) for k, v in mean_mass.items()})''',
       todo='''n_pairs = ...          # a)
farthest = ...         # b) a tuple of two species
n_settings = ...       # c)
top3_masses = ...      # d) a list of 3 masses
lightest2 = ...        # e) the 2 lightest (species, mass) records''',
       check='''wb.check("0A.46a", n_pairs)
wb.check("0A.46b", farthest if farthest is ... or isinstance(farthest, str) else ", ".join(farthest))
wb.check("0A.46c", n_settings)
wb.check("0A.46d", top3_masses)
wb.check("0A.46e", lightest2 if lightest2 is ... or isinstance(lightest2, str) else ", ".join(r[0] for r in lightest2))''',
       solution='''n_pairs = len(list(combinations(numeric_cols, 2)))
farthest = max(combinations(species_names, 2), key=lambda pair: abs(mean_mass[pair[0]] - mean_mass[pair[1]]))
grid = list(product([0.1, 0.01, 0.001], [16, 32, 64], [5, 10]))
n_settings = len(grid)
top3_masses = heapq.nlargest(3, clean["body_mass_g"])
lightest2 = heapq.nsmallest(2, mass_records, key=lambda r: r[1])
print(n_pairs, farthest, n_settings, grid[:2], top3_masses, lightest2)''',
       record='''wb.record("0A.46a", n_pairs, mistakes={"combinations ne compte chaque paire qu'une fois, sans ordre": 12})
wb.record("0A.46b", ", ".join(farthest))
wb.record("0A.46c", n_settings, mistakes={"product combine les TROIS listes : 3 × 3 × 2": 9})
wb.record("0A.46d", top3_masses)
wb.record("0A.46e", ", ".join(r[0] for r in lightest2))''',
       note="Une grille de 18 réglages, c'est 18 entraînements : au ch. 15, `GridSearchCV` fait exactement ce "
            "`product` (fois le nombre de plis de la validation croisée)."),

    Ex("0A.47", "🔨", 2, 26, "Une classe `RunningStats` : `__init__`, attributs, méthodes, puis la même en `@dataclass`",
       "écrire une classe qui garde un état (attributs) et le met à jour avec des méthodes ; utiliser `@dataclass`.",
       "Ex 0A.23 · fiche §100.7.1", thread="Penguins", tracks="R, M, C",
       body="""a) Écris la classe `RunningStats`, qui résume une série de nombres **au fil de l'eau**, sans les garder en mémoire (utile quand les données arrivent une par une, ou ne tiennent pas en mémoire) :
- `__init__(self)` met `n`, `total` et `total_sq` (la somme des carrés) à 0, et `minimum` et `maximum` à `None` ;
- `add(self, x)` met à jour les cinq attributs ;
- `mean(self)` renvoie la moyenne ; `std(self)` renvoie l'écart-type.

> 🧮 **Rappel maths** — L'écart-type $\\sigma$ mesure la dispersion autour de la moyenne $\\bar{x}$ ; sa formule « au fil de l'eau » n'a besoin que de $n$, de $\\sum x$ et de $\\sum x^2$ : $\\sigma = \\sqrt{\\frac{1}{n}\\sum x_i^2 - \\bar{x}^2}$ (moyenne des carrés moins carré de la moyenne ; détails au ch. 2).

b) Écris la **dataclass** `PenguinRecord` : `species: str`, `island: str`, `mass_g: float | None = None`, avec une méthode `mass_kg(self)` qui renvoie la masse en kg, ou `None` si elle manque.

Vérifications, après avoir ajouté les 342 masses connues une par une : a) `stats.n` · b) `stats.mean()` (2 décimales) · c) `stats.std()` (2 décimales, à comparer à `np.std(known_masses)`) · d) `(stats.minimum, stats.maximum)` · e) `PenguinRecord("Gentoo", "Biscoe", 5076).mass_kg()` (3 décimales) · f) `PenguinRecord("Adelie", "Dream") == PenguinRecord("Adelie", "Dream", None)` · g) `repr(PenguinRecord("Adelie", "Dream"))`.""",
       given='''known_masses = penguins["body_mass_g"].dropna().tolist()
print(len(known_masses), known_masses[:3])''',
       todo='''class RunningStats:
    """Count, mean, standard deviation, min and max of a stream of numbers."""

    def __init__(self):
        raise NotImplementedError("RunningStats is not written yet")

    def add(self, x):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def std(self):
        raise NotImplementedError


# b) the dataclass PenguinRecord (remove the line below and write it)
PenguinRecord = None''',
       check='''with wb.attempt("0A.47"):
    stats = RunningStats()
    for m in known_masses:
        stats.add(m)
    wb.check("0A.47a", stats.n)
    wb.check("0A.47b", stats.mean())
    wb.check("0A.47c", stats.std())
    wb.check("0A.47d", (stats.minimum, stats.maximum))
if PenguinRecord is None:
    print("⏳ Ex 0A.47e : pas encore fait (la dataclass PenguinRecord).")
else:
    wb.check("0A.47e", PenguinRecord("Gentoo", "Biscoe", 5076).mass_kg())
    wb.check("0A.47f", PenguinRecord("Adelie", "Dream") == PenguinRecord("Adelie", "Dream", None))
    wb.check("0A.47g", repr(PenguinRecord("Adelie", "Dream")))''',
       solution='''class RunningStats:
    """Count, mean, standard deviation, min and max of a stream of numbers."""

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, x):
        self.n += 1
        self.total += x
        self.total_sq += x * x
        self.minimum = x if self.minimum is None else min(self.minimum, x)
        self.maximum = x if self.maximum is None else max(self.maximum, x)

    def mean(self):
        return self.total / self.n

    def std(self):
        return math.sqrt(self.total_sq / self.n - self.mean() ** 2)


@dataclass
class PenguinRecord:
    species: str
    island: str
    mass_g: float | None = None

    def mass_kg(self):
        return None if self.mass_g is None else self.mass_g / 1000


stats = RunningStats()
for m in known_masses:
    stats.add(m)
print(stats.n, stats.mean(), stats.std(), np.std(known_masses), stats.minimum, stats.maximum)
print(PenguinRecord("Adelie", "Dream"), PenguinRecord("Gentoo", "Biscoe", 5076).mass_kg())''',
       record='''wb.record("0A.47a", stats.n)
wb.record("0A.47b", stats.mean(), decimals=2)
wb.record("0A.47c", stats.std(), decimals=2, mistakes={"c'est l'écart-type (la racine carrée de la variance)": stats.std() ** 2})
wb.record("0A.47d", (stats.minimum, stats.maximum))
wb.record("0A.47e", PenguinRecord("Gentoo", "Biscoe", 5076).mass_kg(), decimals=3)
wb.record("0A.47f", PenguinRecord("Adelie", "Dream") == PenguinRecord("Adelie", "Dream", None))
wb.record("0A.47g", repr(PenguinRecord("Adelie", "Dream")))''',
       note="`@dataclass` a écrit `__init__`, `__repr__` et `__eq__` à ta place. Pour de très grandes séries, la "
            "formule « moyenne des carrés moins carré de la moyenne » perd en précision ; l'algorithme de Welford, "
            "qui met à jour la moyenne pas à pas, est la version robuste."),

    Ex("0A.48", "🔨", 2, 30, "Méthodes spéciales : une classe `Vector2D` qui s'additionne",
       "donner à une classe le comportement d'un type natif (affichage, égalité, opérateurs, `len`, `[]`, `sum`).",
       "Ex 0A.47 · fiche §100.7.2", tracks="C",
       body="""Écris la classe `Vector2D(x, y)`, un vecteur du plan, avec les méthodes spéciales :
- `__repr__` : `Vector2D(x=1, y=2)` ;
- `__eq__` : deux vecteurs sont égaux si leurs `x` et leurs `y` le sont ;
- `__add__` et `__sub__` : somme et différence de deux vecteurs, coordonnée par coordonnée ;
- `__mul__` (vecteur × nombre) et `__rmul__` (nombre × vecteur) ;
- `__abs__` : la longueur $\\sqrt{x^2 + y^2}$ (`math.hypot(x, y)`), appelée par `abs(v)` ;
- `__getitem__` : `v[0]` renvoie `x`, `v[1]` renvoie `y` ;
- `__radd__` : `sum(vecteurs)` commence par calculer `0 + v1`, ce qui appelle `v1.__radd__(0)` ; renvoie le vecteur lui-même quand `other == 0`, sinon `self + other`.

Vérifications : a) `repr(Vector2D(1, 2) + Vector2D(3, 4))` · b) `abs(Vector2D(3, 4))` · c) `2 * Vector2D(1, -1) == Vector2D(2, -2)` · d) `Vector2D(7, 9)[1]` · e) le **centre** des becs des Gentoo : la vérification construit un `Vector2D(longueur, profondeur)` par couple de `gentoo_bills` et calcule `center_gentoo = sum(vecteurs) * (1 / len(vecteurs))` ; elle affiche sa longueur de bec, `center_gentoo[0]` (2 décimales) · f) la distance entre les centres des Gentoo et des Adelie, `abs(center_gentoo - center_adelie)` (2 décimales).""",
       given='''def bills_of(species):
    """The (bill length, bill depth) pairs of one species, as a list of tuples."""
    group = clean[clean["species"] == species]
    return list(zip(group["bill_length_mm"], group["bill_depth_mm"]))


gentoo_bills, adelie_bills = bills_of("Gentoo"), bills_of("Adelie")
print(len(gentoo_bills), gentoo_bills[:2])''',
       todo='''class Vector2D:
    """A vector of the plane with the usual operations."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        raise NotImplementedError("Vector2D.__repr__ is not written yet")

    def __eq__(self, other):
        raise NotImplementedError("Vector2D.__eq__ is not written yet")

    def __add__(self, other):
        raise NotImplementedError("Vector2D.__add__ is not written yet")

    def __radd__(self, other):
        raise NotImplementedError("Vector2D.__radd__ is not written yet")

    def __sub__(self, other):
        raise NotImplementedError("Vector2D.__sub__ is not written yet")

    def __mul__(self, scalar):
        raise NotImplementedError("Vector2D.__mul__ is not written yet")

    def __rmul__(self, scalar):
        raise NotImplementedError("Vector2D.__rmul__ is not written yet")

    def __abs__(self):
        raise NotImplementedError("Vector2D.__abs__ is not written yet")

    def __getitem__(self, i):
        raise NotImplementedError("Vector2D.__getitem__ is not written yet")''',
       check='''with wb.attempt("0A.48"):
    wb.check("0A.48a", repr(Vector2D(1, 2) + Vector2D(3, 4)))
    wb.check("0A.48b", abs(Vector2D(3, 4)))
    wb.check("0A.48c", 2 * Vector2D(1, -1) == Vector2D(2, -2))
    wb.check("0A.48d", Vector2D(7, 9)[1])
    center_gentoo = sum(Vector2D(l, d) for l, d in gentoo_bills) * (1 / len(gentoo_bills))
    center_adelie = sum(Vector2D(l, d) for l, d in adelie_bills) * (1 / len(adelie_bills))
    wb.check("0A.48e", center_gentoo[0])
    wb.check("0A.48f", abs(center_gentoo - center_adelie))''',
       solution='''class Vector2D:
    """A vector of the plane with the usual operations."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector2D(x={self.x}, y={self.y})"

    def __eq__(self, other):
        return isinstance(other, Vector2D) and self.x == other.x and self.y == other.y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        if other == 0:              # sum() starts with 0 + first vector
            return self
        return self + other

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __getitem__(self, i):
        return (self.x, self.y)[i]


center_gentoo = sum(Vector2D(l, d) for l, d in gentoo_bills) * (1 / len(gentoo_bills))
center_adelie = sum(Vector2D(l, d) for l, d in adelie_bills) * (1 / len(adelie_bills))
print(Vector2D(1, 2) + Vector2D(3, 4), abs(Vector2D(3, 4)), 2 * Vector2D(1, -1))
print(center_gentoo, center_adelie, abs(center_gentoo - center_adelie))''',
       record='''wb.record("0A.48a", repr(Vector2D(1, 2) + Vector2D(3, 4)))
wb.record("0A.48b", abs(Vector2D(3, 4)), decimals=1)
wb.record("0A.48c", 2 * Vector2D(1, -1) == Vector2D(2, -2))
wb.record("0A.48d", Vector2D(7, 9)[1])
wb.record("0A.48e", center_gentoo[0], decimals=2)
wb.record("0A.48f", abs(center_gentoo - center_adelie), decimals=2)''',
       note="`2 * v` : `int` ne sait pas multiplier un `Vector2D`, Python essaie alors `v.__rmul__(2)`. Même "
            "mécanisme pour `sum` et `__radd__`. La classe `Value` du ch. 18 repose entièrement sur ces méthodes."),

    Ex("0A.49", "🔨", 2, 30, "Héritage et `super()` : un mini-estimateur `fit`/`predict` appelable",
       "écrire des classes filles qui héritent d'une interface commune `fit`/`predict`, comme scikit-learn.",
       "Ex 0A.47 · fiche §100.7.3, §100.7.2 ; §100.8.4-5 (réductions par axe et broadcasting : un aperçu suffit)", thread="Penguins", tracks="R, M, C",
       body="""`BaseClassifier` (fournie) décrit ce qu'ont en commun tous les classifieurs : `fit(X, y)` apprend, `predict(X)` prédit, `score(X, y)` calcule la proportion de bonnes réponses, et l'objet est **appelable** (`model(X)` appelle `predict`). Écris deux classes **filles** :

- `MajorityClassifier` : `fit` range dans `self.majority_` l'étiquette la plus fréquente de `y` (`Counter(y).most_common(1)`) et **renvoie `self`** ; `predict` renvoie un array rempli de cette étiquette, une par ligne de `X` (`np.full(len(X), self.majority_)`). C'est la référence la plus simple : un modèle qui fait moins bien ne sert à rien.
- `NearestCentroidClassifier` : `__init__(self, verbose=False)` appelle `super().__init__(verbose)` ; `fit` calcule `self.classes_` (les étiquettes distinctes, triées) et `self.centroids_` (un array `(n_classes, n_features)` : la moyenne des lignes de `X` de chaque classe), affiche un message si `self.verbose`, et renvoie `self` ; `predict` renvoie, pour chaque ligne, la classe dont le centre est le plus **proche** (distance euclidienne ; une boucle sur les classes suffit).

Données : `X2`, la longueur du bec et de la nageoire des 333 manchots complets, et `y`, leur espèce.

Vérifications : a) `MajorityClassifier().fit(X2, y).majority_` · b) son `score(X2, y)` (3 décimales) · c) le `score` du `NearestCentroidClassifier` (3 décimales) · d) la prédiction obtenue en **appelant** le modèle sur le premier manchot, `model(X2[:1])[0]` · e) `isinstance(model, BaseClassifier)`.""",
       given='''class BaseClassifier:
    """Common interface: fit, predict, score, and calling the model like a function."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y):
        raise NotImplementedError(f"{type(self).__name__}.fit is not written yet")

    def predict(self, X):
        raise NotImplementedError(f"{type(self).__name__}.predict is not written yet")

    def score(self, X, y):
        """Accuracy: the proportion of correct predictions."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __call__(self, X):
        return self.predict(X)


X2 = clean[["bill_length_mm", "flipper_length_mm"]].to_numpy()
y = clean["species"].to_numpy()''',
       todo='''class MajorityClassifier(BaseClassifier):
    """Always predicts the most frequent label seen in fit."""


class NearestCentroidClassifier(BaseClassifier):
    """Predicts the class whose mean point (centroid) is the closest."""''',
       check='''with wb.attempt("0A.49a"):
    baseline = MajorityClassifier().fit(X2, y)
    wb.check("0A.49a", baseline.majority_)
    wb.check("0A.49b", baseline.score(X2, y))
with wb.attempt("0A.49c"):
    model = NearestCentroidClassifier(verbose=True).fit(X2, y)
    wb.check("0A.49c", model.score(X2, y))
    wb.check("0A.49d", model(X2[:1])[0])
    wb.check("0A.49e", isinstance(model, BaseClassifier))''',
       solution='''class MajorityClassifier(BaseClassifier):
    """Always predicts the most frequent label seen in fit."""

    def fit(self, X, y):
        self.majority_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)


class NearestCentroidClassifier(BaseClassifier):
    """Predicts the class whose mean point (centroid) is the closest."""

    def __init__(self, verbose=False):
        super().__init__(verbose)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == label].mean(axis=0) for label in self.classes_])
        if self.verbose:
            print(f"fit: {len(self.classes_)} centroids, one per class")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.array([np.sqrt(((X - c) ** 2).sum(axis=1)) for c in self.centroids_])  # (n_classes, n)
        return self.classes_[distances.argmin(axis=0)]


baseline = MajorityClassifier().fit(X2, y)
model = NearestCentroidClassifier(verbose=True).fit(X2, y)
print(baseline.majority_, baseline.score(X2, y), model.score(X2, y), model(X2[:3]))
print(model.centroids_.round(1))''',
       record='''wb.record("0A.49a", baseline.majority_)
wb.record("0A.49b", baseline.score(X2, y), decimals=3)
wb.record("0A.49c", model.score(X2, y), decimals=3)
wb.record("0A.49d", model(X2[:1])[0])
wb.record("0A.49e", isinstance(model, BaseClassifier))''',
       note="Ce « plus proche centre » atteint 89 % de bonnes réponses contre 44 % pour la classe majoritaire, "
            "**sur les données d'entraînement** : on apprendra au ch. 8 pourquoi il faut évaluer sur des données "
            "jamais vues. scikit-learn propose le même modèle (`sklearn.neighbors.NearestCentroid`)."),

    Ex("0A.50", "🔨", 2, 30, "Générateurs et itérables : un mini-Dataset de manchots",
       "rendre une classe itérable et indexable, et produire des lots à la demande avec `yield`.",
       "Ex 0A.49 · fiche §100.7.4, §100.7.2", thread="Penguins", tracks="C",
       body="""Les `Dataset` et `DataLoader` de PyTorch (ch. 20) reposent sur deux idées : un objet qui a une **longueur** et qu'on **indexe**, et des lots produits **à la demande**. Écris-en une version miniature :

- `PenguinDataset(frame, features, target)` : `__init__` range `self.X` (l'array des colonnes `features`) et `self.y` (l'array de la colonne `target`) ; `__len__` renvoie le nombre d'exemples ; `__getitem__(i)` renvoie le couple `(self.X[i], self.y[i])` ; `__iter__` est une **fonction génératrice** qui fournit les couples un par un avec `yield`.
- `batches(dataset, batch_size)` : une **fonction génératrice** qui fournit, dans l'ordre, les couples `(X_lot, y_lot)` de `batch_size` exemples consécutifs (le dernier lot peut être plus petit ; les tranches de `dataset.X` et `dataset.y` suffisent).

Vérifications, avec `ds = PenguinDataset(clean, numeric_cols, "species")` : a) `len(ds)` · b) l'espèce du premier exemple, `ds[0][1]` · c) la moyenne des nageoires calculée en **parcourant** `ds` avec `for` (2 décimales) · d) le nombre de lots de 64 · e) la taille du dernier lot · f) un générateur ne se parcourt qu'une fois : après `list(gen)`, que vaut `len(list(gen))` ? · g) `__iter__` est-elle bien une fonction génératrice ? (la vérification regarde si `iter(ds)` est un générateur)""",
       todo='''class PenguinDataset:
    """A tiny dataset: len(ds), ds[i] -> (features, label), and iteration with for."""

    def __init__(self, frame, features, target):
        raise NotImplementedError("PenguinDataset is not written yet")

    def __len__(self):
        raise NotImplementedError("PenguinDataset.__len__ is not written yet")

    def __getitem__(self, i):
        raise NotImplementedError("PenguinDataset.__getitem__ is not written yet")

    def __iter__(self):
        raise NotImplementedError("PenguinDataset.__iter__ is not written yet")
        yield  # this line makes __iter__ a generator function


def batches(dataset, batch_size):
    """Yield (X_batch, y_batch) pairs of batch_size consecutive examples."""
    raise NotImplementedError("batches() is not written yet")
    yield  # this line makes batches a generator function''',
       check='''with wb.attempt("0A.50"):
    ds = PenguinDataset(clean, numeric_cols, "species")
    wb.check("0A.50a", len(ds))
    wb.check("0A.50b", ds[0][1])
    wb.check("0A.50c", sum(x[2] for x, label in ds) / len(ds))
    all_batches = list(batches(ds, 64))
    wb.check("0A.50d", len(all_batches))
    wb.check("0A.50e", len(all_batches[-1][0]))
    gen = batches(ds, 64)
    list(gen)
    wb.check("0A.50f", len(list(gen)))
    wb.check("0A.50g", type(iter(ds)).__name__ == "generator")''',
       solution='''class PenguinDataset:
    """A tiny dataset: len(ds), ds[i] -> (features, label), and iteration with for."""

    def __init__(self, frame, features, target):
        self.X = frame[features].to_numpy()
        self.y = frame[target].to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def batches(dataset, batch_size):
    """Yield (X_batch, y_batch) pairs of batch_size consecutive examples."""
    for start in range(0, len(dataset), batch_size):
        yield dataset.X[start:start + batch_size], dataset.y[start:start + batch_size]


ds = PenguinDataset(clean, numeric_cols, "species")
all_batches = list(batches(ds, 64))
gen = batches(ds, 64)
first = next(gen)          # a generator gives its values one at a time
list(gen)                  # ... until it is exhausted
exhausted = len(list(gen))
print(len(ds), ds[0], [len(b[0]) for b in all_batches], first[1][:3], exhausted)''',
       record='''wb.record("0A.50a", len(ds))
wb.record("0A.50b", ds[0][1])
wb.record("0A.50c", sum(x[2] for x, label in ds) / len(ds), decimals=2)
wb.record("0A.50d", len(all_batches), mistakes={"le dernier lot, incomplet, compte aussi (0A.5)": 5})
wb.record("0A.50e", len(all_batches[-1][0]))
wb.record("0A.50g", type(iter(ds)).__name__ == "generator")
wb.record("0A.50f", exhausted, mistakes={"un générateur épuisé ne recommence pas : le deuxième list(gen) est vide": 6})''',
       note="Avec `__len__` et `__getitem__`, `PenguinDataset` a déjà l'interface d'un `torch.utils.data.Dataset` ; "
            "le `DataLoader` y ajoute le mélange et le découpage en lots, comme `iterate_minibatches` (0A.66)."),
])

# ---------------------------------------------------------------------------
# Part G (0A.51 to 0A.60): NumPy, pandas and matplotlib, further
# ---------------------------------------------------------------------------
PART_G_GIVEN = '''# Data for part G
import time

import matplotlib.pyplot as plt

X = clean[numeric_cols].to_numpy()             # (333, 4): bill length, bill depth, flipper, mass
y = clean["species"].to_numpy()                # (333,)
species_names = np.unique(y)                   # ['Adelie' 'Chinstrap' 'Gentoo']
images, labels = wb.datasets.load_mnist("train", n=1000)   # 1000 handwritten digits
print(X.shape, y.shape, species_names, images.shape, images.dtype, labels.shape)'''

PART_G = Part("G", "NumPy, pandas et matplotlib : aller plus loin",
              "Fiche §100.8.3 à §100.10.2. On reprend les 333 manchots complets (`X`, `y`) et 1000 chiffres "
              "manuscrits de MNIST (`images`, `labels`).", given=PART_G_GIVEN, exercises=[
    Ex("0A.51", "📦", 2, 26, "Réductions par axe, tri, `argmax` et `unique`",
       "résumer un tableau par axe, trier, et retrouver la position d'un maximum.",
       "Ex 0A.7, Ex 0A.35 · fiche §100.8.5", thread="Penguins", tracks="R, M, C",
       body="""a) La moyenne de chaque colonne de `X` (un array de 4 valeurs ; 2 décimales).
b) `class_means`, un array `(3, 4)` : la ligne `k` contient la moyenne des colonnes pour l'espèce `species_names[k]` (une boucle ou une compréhension sur les espèces, avec un masque `y == nom`). Arrondi à 2 décimales.
c) Pour chaque **mesure**, l'indice de l'espèce qui a la plus grande moyenne : `class_means.argmax(axis=...)` (réfléchis : quel axe doit disparaître ?).
d) L'espèce du manchot le plus lourd (`argmax` sur la colonne des masses, puis `y[...]`).
e) Le nombre de manchots de chaque espèce, avec `np.unique(..., return_counts=True)`.
f) Les trois plus petites masses (`np.sort`).""",
       todo='''col_means = ...          # a) shape (4,)
class_means = ...        # b) shape (3, 4)
best_species_idx = ...   # c) shape (4,)
heaviest_species = ...   # d) a string
species_counts = ...     # e) shape (3,)
lightest3 = ...          # f) shape (3,)''',
       check='''wb.check("0A.51a", col_means)
wb.check("0A.51b", class_means)
wb.check("0A.51c", best_species_idx)
wb.check("0A.51d", heaviest_species)
wb.check("0A.51e", species_counts)
wb.check("0A.51f", lightest3)''',
       solution='''col_means = X.mean(axis=0)
class_means = np.array([X[y == name].mean(axis=0) for name in species_names])
best_species_idx = class_means.argmax(axis=0)          # the species axis (0) disappears: one index per column
heaviest_species = y[X[:, 3].argmax()]
species, species_counts = np.unique(y, return_counts=True)
lightest3 = np.sort(X[:, 3])[:3]
print(col_means.round(2))
print(class_means.round(2))
print(best_species_idx, species_names[best_species_idx], heaviest_species, dict(zip(species, species_counts)), lightest3)''',
       record='''wb.record("0A.51a", col_means, decimals=2)
wb.record("0A.51b", class_means, decimals=2)
wb.record("0A.51c", best_species_idx, mistakes={"axis=1 donne la meilleure MESURE de chaque espèce ; on veut la meilleure espèce pour chaque mesure : axis=0": class_means.argmax(axis=1)})
wb.record("0A.51d", heaviest_species)
wb.record("0A.51e", species_counts)
wb.record("0A.51f", lightest3)''',
       note="`best_species_idx` vaut `[1, 1, 2, 2]` : les Chinstrap ont le bec le plus long **et** (de peu) le plus "
            "profond, les Gentoo la nageoire la plus longue et la plus grande masse. `species_names[best_species_idx]` "
            "traduit les indices en noms : c'est ainsi qu'on passe d'un `argmax` à une prédiction de classe."),

    Ex("0A.52", "📦", 2, 26, "Broadcasting : standardiser toutes les colonnes d'un coup",
       "utiliser le broadcasting pour standardiser un tableau et calculer toutes les distances entre des points.",
       "Ex 0A.8, Ex 0A.51 · fiche §100.8.4", thread="Penguins", tracks="R, M, C",
       body="""Les mesures n'ont pas la même échelle (des mm, des g) : on les **standardise** en retirant à chaque colonne sa moyenne et en divisant par son écart-type, $z = (x - \\bar{x}) / \\sigma$ (ch. 12). Avec le broadcasting, c'est une ligne, sans boucle.

a) `Z` : `X` standardisé colonne par colonne (`X.mean(axis=0)` et `X.std(axis=0)` ont la forme `(4,)`). Sa forme ?
b) Les moyennes des colonnes de `Z` sont-elles toutes nulles, à `1e-10` près ? (`True` ou `False`, avec `np.abs(...).max() < 1e-10`)
c) Les écarts-types des colonnes de `Z` (3 décimales).
d) La première ligne de `Z` (2 décimales) : ce premier manchot a-t-il une nageoire plus courte ou plus longue que la moyenne ?
e) **Distances** entre les 5 premiers manchots de `Z` : `P = Z[:5]` a la forme `(5, 4)`. `P[:, np.newaxis, :]` a la forme `(5, 1, 4)` et `P[np.newaxis, :, :]` la forme `(1, 5, 4)` : leur différence a la forme `(5, 5, 4)` (toutes les paires !). Mets au carré, somme sur le dernier axe, prends la racine : `D`, de forme `(5, 5)`, où `D[i, j]` est la distance euclidienne entre les manchots `i` et `j`. Vérification : `D[0, 4]` (2 décimales).""",
       todo='''Z = ...            # a) standardized X, shape (333, 4)
means_zero = ...   # b) True or False
z_stds = ...       # c) shape (4,)
D = ...            # e) distances between the first 5 penguins, shape (5, 5)''',
       check='''if Z is ...:
    print("⏳ Ex 0A.52 : pas encore fait.")
else:
    wb.check("0A.52a", Z.shape)
    wb.check("0A.52b", means_zero)
    wb.check("0A.52c", z_stds)
    wb.check("0A.52d", Z[0])
    wb.check("0A.52e", D if D is ... else D[0, 4])''',
       solution='''Z = (X - X.mean(axis=0)) / X.std(axis=0)          # (333, 4) with (4,): each row gets the same correction
means_zero = bool(np.abs(Z.mean(axis=0)).max() < 1e-10)
z_stds = Z.std(axis=0)
P = Z[:5]
D = np.sqrt(((P[:, np.newaxis, :] - P[np.newaxis, :, :]) ** 2).sum(axis=2))   # (5, 1, 4) - (1, 5, 4) -> (5, 5, 4)
print(Z.shape, means_zero, z_stds.round(3), Z[0].round(2))
print(D.round(2))''',
       record='''wb.record("0A.52a", Z.shape)
wb.record("0A.52b", means_zero)
wb.record("0A.52c", z_stds, decimals=3)
wb.record("0A.52d", Z[0], decimals=2, mistakes={"divise aussi par l'écart-type de chaque colonne": X[0] - X.mean(axis=0)})
wb.record("0A.52e", D[0, 4], decimals=2)''',
       note="La matrice `D` est symétrique, avec des zéros sur la diagonale. Le calcul de toutes les distances par "
            "broadcasting est le cœur des k plus proches voisins (ch. 13) ; il crée un tableau `(n, n, p)` : "
            "attention à la mémoire quand `n` est grand."),

    Ex("0A.53", "📦", 2, 20, "`reshape`, transposée et empilement",
       "changer la forme d'un tableau et assembler des tableaux sans se tromper d'axe.",
       "Ex 0A.52 · fiche §100.8.6", tracks="R, M, C",
       body="""Avec `v = np.arange(12)` :
a) `v.reshape(3, 4)[2, 1]` · b) la forme de `v.reshape(2, -1)` · c) la ligne 1 de la **transposée** de `v.reshape(3, 4)`.

Avec `X` :
d) Reconstruis un tableau `(333, 2)` à partir des deux colonnes `X[:, 0]` et `X[:, 2]` avec `np.stack(..., axis=1)` : sa forme ?
e) et f) Ajoute à `X` une première colonne de 1 (le « biais » des modèles linéaires, ch. 13 et 16) avec `np.hstack` et `np.ones((len(X), 1))` : e) la forme de `X_bias`, f) la somme de sa première colonne.
g) Colle les lignes `X[:100]` et `X[200:]` avec `np.concatenate` : la forme du résultat.

Range dans chaque variable le **tableau** lui-même : la vérification lit sa forme.""",
       given='''v = np.arange(12)''',
       todo='''value_21 = ...       # a)
shape_2 = ...        # b)
row_1_of_T = ...     # c)
two_cols = ...       # d) the array, shape (333, 2)
X_bias = ...         # e) and f): the array, shape (333, 5)
glued = ...          # g) the array''',
       check='''wb.check("0A.53a", value_21)
wb.check("0A.53b", shape_2)
wb.check("0A.53c", row_1_of_T)
wb.check("0A.53d", two_cols if two_cols is ... else two_cols.shape)
wb.check("0A.53e", X_bias if X_bias is ... else X_bias.shape)
wb.check("0A.53f", X_bias if X_bias is ... else X_bias[:, 0].sum())
wb.check("0A.53g", glued if glued is ... else glued.shape)''',
       solution='''value_21 = v.reshape(3, 4)[2, 1]
shape_2 = v.reshape(2, -1).shape
row_1_of_T = v.reshape(3, 4).T[1]
two_cols = np.stack([X[:, 0], X[:, 2]], axis=1)
X_bias = np.hstack([np.ones((len(X), 1)), X])
glued = np.concatenate([X[:100], X[200:]])
print(v.reshape(3, 4), value_21, shape_2, row_1_of_T, two_cols.shape, X_bias.shape, X_bias[:2], glued.shape, sep="\\n")''',
       record='''wb.record("0A.53a", value_21)
wb.record("0A.53b", shape_2)
wb.record("0A.53c", row_1_of_T, mistakes={"la ligne 1 de la TRANSPOSÉE est la colonne 1 de v.reshape(3, 4)": v.reshape(3, 4)[1]})
wb.record("0A.53d", two_cols.shape, mistakes={"np.stack(..., axis=0) empile en lignes ; axis=1 range chaque colonne côte à côte": (2, 333)})
wb.record("0A.53e", X_bias.shape)
wb.record("0A.53f", X_bias[:, 0].sum(), decimals=1)
wb.record("0A.53g", glued.shape, mistakes={"X[200:] contient les lignes 200 à 332, soit 133 lignes": (300, 4)})'''),

    Ex("0A.54", "📦", 2, 26, "Images MNIST : un tableau `(N, 28, 28)`",
       "manipuler des images comme des arrays : forme, type, intensités, moyennes par classe, aplatissement.",
       "Ex 0A.53 · fiche §100.8.8", thread="MNIST", tracks="R, C",
       body="""`images` contient 1000 chiffres manuscrits de MNIST (`uint8`, de 0 à 255) et `labels` le chiffre dessiné.

a) La forme de `images` · b) le nom de son `dtype` · c) l'intensité moyenne de la première image (2 décimales) · d) la proportion de pixels **nuls** (le fond noir) sur toutes les images (3 décimales) · e) la forme de `flat`, les images aplaties en vecteurs de 784 valeurs · f) `ink`, un array de 10 valeurs : l'intensité moyenne des images de chaque chiffre (`images[labels == d].mean()`), puis le chiffre qui utilise le **plus** d'encre et celui qui en utilise le **moins** (`argmax`, `argmin`) · g) combien de fois plus de mémoire prend `images / 255` que `images` (`.nbytes`) ?

La cellule suivante affiche l'image moyenne de chaque chiffre : regarde-la, elle explique f).""",
       todo='''images_shape = ...     # a)
images_dtype = ...     # b) a string
mean_first = ...       # c)
share_zero = ...       # d)
flat = ...             # e)
ink = ...              # f) shape (10,)
memory_ratio = ...     # g)''',
       check='''wb.check("0A.54a", images_shape)
wb.check("0A.54b", images_dtype)
wb.check("0A.54c", mean_first)
wb.check("0A.54d", share_zero)
wb.check("0A.54e", flat if flat is ... else flat.shape)
wb.check("0A.54f", ink if ink is ... else [int(ink.argmax()), int(ink.argmin())])
wb.check("0A.54g", memory_ratio)''',
       solution='''images_shape = images.shape
images_dtype = images.dtype.name
mean_first = images[0].mean()
share_zero = (images == 0).mean()
flat = images.reshape(len(images), -1)
ink = np.array([images[labels == d].mean() for d in range(10)])
memory_ratio = (images / 255).nbytes / images.nbytes       # float64 (8 bytes) against uint8 (1 byte)
print(images_shape, images_dtype, round(mean_first, 2), round(share_zero, 3), flat.shape)
print(ink.round(1), ink.argmax(), ink.argmin(), memory_ratio)''',
       record='''wb.record("0A.54a", images_shape)
wb.record("0A.54b", images_dtype)
wb.record("0A.54c", mean_first, decimals=2)
wb.record("0A.54d", share_zero, decimals=3)
wb.record("0A.54e", flat.shape)
wb.record("0A.54f", [int(ink.argmax()), int(ink.argmin())])
wb.record("0A.54g", memory_ratio, decimals=1)''',
       after=[("code", '''mean_digits = np.array([images[labels == d].mean(axis=0) for d in range(10)])   # (10, 28, 28)
wb.plot.show_images(mean_digits, labels=np.arange(10), ncols=10)
plt.show()''')],
       note="Le 0 est un grand anneau (beaucoup d'encre), le 1 un simple trait. 81 % des pixels sont nuls : "
            "l'information tient dans une petite partie de l'image, ce qu'exploiteront les réseaux convolutifs (ch. 21)."),

    Ex("0A.55", "🔬", 2, 20, "Boucle Python contre NumPy : mesurer le gain",
       "mesurer l'écart de vitesse entre une boucle Python et le calcul vectorisé, et en tirer une règle.",
       "Ex 0A.30, Ex 0A.54 · fiche §100.8.3", thread="MNIST", tracks="C",
       body="""On veut l'intensité moyenne de **chaque** image.

1. Écris `mean_per_image_loop(images)` avec des boucles `for` : pour chaque image, additionne ses 784 pixels un par un, puis divise par leur nombre ; renvoie la liste des moyennes. ⚠️ Additionne des `int(pixel)` : un `uint8` ne dépasse pas 255 et « déborde » (`np.uint8(200) + np.uint8(100)` vaut 44, avec au mieux un avertissement).
2. Écris `mean_per_image_numpy(images)` en **une** ligne, sans boucle (`mean` avec `axis=(1, 2)`).
3. La cellule de vérification compare les deux résultats, puis mesure les deux durées pour 100, 300 et 1000 images (`time.perf_counter`) et trace le temps en fonction du nombre d'images.

**Mon analyse** (dans une cellule Markdown) : combien de fois plus rapide est NumPy sur ta machine ? Le rapport change-t-il avec le nombre d'images ? D'où vient l'écart (fiche §100.8.3) ?""",
       todo='''def mean_per_image_loop(images):
    """Mean intensity of each image, computed pixel by pixel with Python loops."""
    raise NotImplementedError("mean_per_image_loop() is not written yet")


def mean_per_image_numpy(images):
    """Mean intensity of each image, without any Python loop."""
    raise NotImplementedError("mean_per_image_numpy() is not written yet")''',
       check='''def measure(func, data, repeat=3):
    """Best time (s) of `repeat` runs of func(data)."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func(data)
        best = min(best, time.perf_counter() - start)
    return best


with wb.attempt("0A.55"):
    same = np.allclose(mean_per_image_loop(images[:50]), mean_per_image_numpy(images[:50]))
    print("✅ Ex 0A.55 : les deux versions donnent les mêmes moyennes." if same else
          "❌ Ex 0A.55 : les deux versions ne donnent pas les mêmes moyennes (débordement des uint8 ?).")
    sizes, t_loop, t_numpy = [100, 300, 1000], [], []
    for n in sizes:
        t_loop.append(measure(mean_per_image_loop, images[:n], repeat=1))
        t_numpy.append(measure(mean_per_image_numpy, images[:n]))
        print(f"{n:>5} images: loop {t_loop[-1]:.3f} s, NumPy {t_numpy[-1] * 1000:.2f} ms, "
              f"NumPy is {t_loop[-1] / t_numpy[-1]:.0f} times faster")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(sizes, t_loop, marker="o", label="Python loops")
    ax.plot(sizes, t_numpy, marker="o", label="NumPy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of images")
    ax.set_ylabel("time (s, log scale)")
    ax.legend()
    plt.show()''',
       solution='''def mean_per_image_loop(images):
    """Mean intensity of each image, computed pixel by pixel with Python loops."""
    means = []
    for image in images:
        total = 0
        count = 0
        for row in image:
            for pixel in row:
                total += int(pixel)     # int(): a uint8 would overflow above 255
                count += 1
        means.append(total / count)
    return means


def mean_per_image_numpy(images):
    """Mean intensity of each image, without any Python loop."""
    return images.mean(axis=(1, 2))


def measure(func, data, repeat=3):
    """Best time (s) of `repeat` runs of func(data)."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func(data)
        best = min(best, time.perf_counter() - start)
    return best


print("same results:", np.allclose(mean_per_image_loop(images[:50]), mean_per_image_numpy(images[:50])))
sizes, t_loop, t_numpy = [100, 300, 1000], [], []
for n in sizes:
    t_loop.append(measure(mean_per_image_loop, images[:n], repeat=1))
    t_numpy.append(measure(mean_per_image_numpy, images[:n]))
    print(f"{n:>5} images: loop {t_loop[-1]:.3f} s, NumPy {t_numpy[-1] * 1000:.2f} ms, "
          f"NumPy is {t_loop[-1] / t_numpy[-1]:.0f} times faster")
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(sizes, t_loop, marker="o", label="Python loops")
ax.plot(sizes, t_numpy, marker="o", label="NumPy")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of images")
ax.set_ylabel("time (s, log scale)")
ax.legend()
plt.show()''',
       note="NumPy est ici de l'ordre de 100 à 300 fois plus rapide ; les mesures fluctuent d'une exécution à "
            "l'autre (autres programmes, mémoire cache), d'où le meilleur de plusieurs essais. Les deux temps croissent "
            "à peu près proportionnellement au nombre d'images (des droites presque parallèles en échelle log), mais "
            "pour 100 images le temps de NumPy est surtout un coût fixe d'appel. L'écart vient de l'interprétation de "
            "chaque instruction Python et de la conversion de chaque pixel en objet Python, que NumPy évite. Règle : "
            "pas de boucle Python sur les éléments d'un array."),

    Ex("0A.56", "🐛", 2, 20, "Bugs NumPy : `axis` oublié, formes `(n,)` et `(n, 1)`, vue modifiée",
       "diagnostiquer trois bugs NumPy silencieux (aucun message d'erreur) et les corriger.",
       "Ex 0A.52, Ex 0A.29 · fiche §100.8.2, §100.8.4, §100.8.5", thread="Penguins", tracks="C",
       body="""Les trois fonctions ci-dessous **ne lèvent aucune erreur**… et renvoient pourtant un résultat faux : ce sont les bugs les plus dangereux. La cellule fournie les essaie sur une **copie** de `X` et montre le symptôme.

- `column_means(X)` devrait renvoyer la moyenne de chaque colonne ;
- `mse(y_true, y_pred)` devrait renvoyer l'erreur quadratique moyenne $\\frac{1}{n}\\sum (y_i - \\hat{y}_i)^2$ entre deux séries de même longueur, même si l'une est rangée en colonne `(n, 1)` ;
- `centered_flippers(X, k)` devrait renvoyer les nageoires des `k` premiers manchots moins leur moyenne, **sans modifier `X`**.

Pour chaque fonction : trouve la cause (quelle forme, quelle vue ?), puis réécris-la corrigée. Vérifications : a) `column_means(X)` (2 décimales) · b) `mse(y_true_col, y_pred)`, où les prédictions sont toutes trop fortes de 100 g · c) `X` est-il intact après `centered_flippers` ? (`True`/`False`) · d) la première valeur renvoyée : la nageoire du premier manchot moins la moyenne des dix premières (1 décimale).""",
       given='''def column_means(X):
    """Mean of each column of X."""
    return X.mean()


def mse(y_true, y_pred):
    """Mean squared error between two series of the same length."""
    return ((y_true - y_pred) ** 2).mean()


def centered_flippers(X, k=10):
    """The flipper lengths of the first k penguins minus their mean (X must not change)."""
    part = X[:k, 2]
    part -= part.mean()
    return part


# Symptoms, on a COPY of X (so that the bugs cannot damage X)
X_demo = X.copy()
y_true_col = X_demo[:, 3:4]          # the true masses, as a COLUMN: shape (333, 1)
y_pred = X_demo[:, 3] + 100          # predictions 100 g too high: shape (333,)
print("column_means:", column_means(X_demo), "<- one number, not four")
print("mse:", mse(y_true_col, y_pred), "<- should be 100 ** 2 = 10000")
before = X_demo[0, 2]
centered_flippers(X_demo)
print("X_demo[0, 2] before:", before, "after:", X_demo[0, 2], "<- the data have changed!")''',
       todo='''def column_means(X):
    raise NotImplementedError("column_means() is not fixed yet")


def mse(y_true, y_pred):
    raise NotImplementedError("mse() is not fixed yet")


def centered_flippers(X, k=10):
    raise NotImplementedError("centered_flippers() is not fixed yet")''',
       check='''with wb.attempt("0A.56a"):
    wb.check("0A.56a", column_means(X))
with wb.attempt("0A.56b"):
    wb.check("0A.56b", mse(y_true_col, y_pred))
with wb.attempt("0A.56c"):
    X_test = X.copy()
    centered = centered_flippers(X_test)
    wb.check("0A.56c", np.array_equal(X_test, X))
    wb.check("0A.56d", centered[0])''',
       solution='''def column_means(X):
    return X.mean(axis=0)                            # one mean per column


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()  # (n, 1) -> (n,): no silent (n, n) broadcasting
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shapes differ: {y_true.shape} and {y_pred.shape}")
    return ((y_true - y_pred) ** 2).mean()


def centered_flippers(X, k=10):
    part = X[:k, 2].copy()                           # a slice is a VIEW: copy before modifying
    part -= part.mean()
    return part


X_test = X.copy()
centered = centered_flippers(X_test)
print(column_means(X).round(2), mse(y_true_col, y_pred), np.array_equal(X_test, X), centered.round(1))
print("the buggy mse compared", (y_true_col - y_pred).shape, "pairs instead of", y_pred.shape)''',
       record='''wb.record("0A.56a", column_means(X), decimals=2)
wb.record("0A.56b", mse(y_true_col, y_pred), decimals=1)
wb.record("0A.56c", np.array_equal(X_test, X))
wb.record("0A.56d", centered[0], decimals=1)''',
       note="Trois réflexes : préciser `axis` ; vérifier les formes (`.shape`) avant toute opération entre deux "
            "tableaux, et lever une erreur si elles diffèrent ; copier (`.copy()`) une tranche avant de la modifier. "
            "La fonction `mse` corrigée refuse désormais deux séries de longueurs différentes au lieu de renvoyer un "
            "nombre faux."),

    Ex("0A.57", "📦", 2, 26, "Compter et regrouper : `value_counts`, `groupby`, `agg`, `sort_values`",
       "répondre à des questions sur un dataset en comptant, en regroupant et en triant avec pandas.",
       "Ex 0A.34 · fiche §100.9.4", thread="Penguins", tracks="R, C",
       body="""Sur `penguins` (les 344 manchots, valeurs manquantes comprises) :

a) Le nombre de manchots sur l'île Biscoe (`value_counts`).
b) La proportion d'Adelie (3 décimales).
c) La masse moyenne des Gentoo **mâles** : `groupby(["species", "sex"])`, puis `.loc[("Gentoo", "male")]` (1 décimale).
d) L'écart-type de la longueur de nageoire des Chinstrap, lu dans le tableau `agg(["count", "mean", "std"])` de la nageoire par espèce (2 décimales).
e) L'espèce du manchot au bec le plus **long** (`sort_values(..., ascending=False)`, puis la première ligne).
f) Le nombre d'espèces **différentes** sur l'île Torgersen (`groupby("island")["species"].nunique()`).""",
       todo='''n_biscoe = ...              # a)
share_adelie = ...          # b)
gentoo_male_mass = ...      # c)
flipper_stats = ...         # d) the agg table (3 rows, 3 columns)
longest_bill_species = ...  # e)
n_species_torgersen = ...   # f)''',
       check='''wb.check("0A.57a", n_biscoe)
wb.check("0A.57b", share_adelie)
wb.check("0A.57c", gentoo_male_mass)
wb.check("0A.57d", flipper_stats if flipper_stats is ... else flipper_stats.loc["Chinstrap", "std"])
wb.check("0A.57e", longest_bill_species)
wb.check("0A.57f", n_species_torgersen)''',
       solution='''n_biscoe = penguins["island"].value_counts()["Biscoe"]
share_adelie = penguins["species"].value_counts(normalize=True)["Adelie"]
gentoo_male_mass = penguins.groupby(["species", "sex"])["body_mass_g"].mean().loc[("Gentoo", "male")]
flipper_stats = penguins.groupby("species")["flipper_length_mm"].agg(["count", "mean", "std"])
longest_bill_species = penguins.sort_values("bill_length_mm", ascending=False).iloc[0]["species"]
n_species_torgersen = penguins.groupby("island")["species"].nunique()["Torgersen"]
print(n_biscoe, round(share_adelie, 3), round(gentoo_male_mass, 1), longest_bill_species, n_species_torgersen)
flipper_stats.round(2)''',
       record='''wb.record("0A.57a", n_biscoe)
wb.record("0A.57b", share_adelie, decimals=3, mistakes={"une proportion est entre 0 et 1 : normalize=True": 152})
wb.record("0A.57c", gentoo_male_mass, decimals=1)
wb.record("0A.57d", flipper_stats.loc["Chinstrap", "std"], decimals=2)
wb.record("0A.57e", longest_bill_species, mistakes={"le bec le plus LONG : ascending=False, puis la première ligne": "Adelie"})
wb.record("0A.57f", n_species_torgersen)''',
       note="`count` vaut 151 pour les Adelie (et non 152) : une nageoire manque. `groupby` à deux clés donne un index "
            "à deux niveaux, qu'on lit avec un tuple ; `.unstack()` le transforme en tableau croisé (0A.36)."),

    Ex("0A.58", "🐛", 2, 15, "Le filtre qui ne filtre pas : `and`, `&`, parenthèses et copies",
       "corriger les trois erreurs classiques des filtres pandas : `and`, priorité de `&`, affectation en chaîne.",
       "Ex 0A.57 · fiche §100.9.2, §100.9.3", thread="Penguins", tracks="C",
       body="""La cellule fournie tente trois opérations et affiche ce qui se passe :
1. `penguins[penguins["species"] == "Gentoo" and penguins["body_mass_g"] > 5000]` ;
2. la même avec `&` mais **sans parenthèses** ;
3. remplacer les sexes manquants par `"unknown"` avec une affectation « en chaîne » : `demo[demo["sex"].isna()]["sex"] = "unknown"` (sur une copie, `demo`).

a) et b) Les **noms** des exceptions levées par 1 et 2 (lis les messages : pourquoi ?).
c) Écris le bon filtre : combien de Gentoo pèsent **plus** de 5000 g ?
d) Sur `fixed_df = penguins.copy()`, remplace les sexes manquants par `"unknown"` avec **une seule** instruction `.loc[masque, "sex"] = ...` : combien de `"unknown"` ensuite ?
e) Avec `isin`, combien de manchots vivent sur Dream **ou** Torgersen ?""",
       given='''def try_it(code):
    """Run a line of code; return the name of the exception it raises, or "no error"."""
    try:
        exec(code, globals())
    except Exception as error:  # noqa: BLE001 - we want to see what goes wrong
        print(f"{code}\\n   -> {type(error).__name__}: {str(error)[:90]}")
        return type(error).__name__
    print(f"{code}\\n   -> no error")
    return "no error"


try_it('penguins[penguins["species"] == "Gentoo" and penguins["body_mass_g"] > 5000]')
try_it('penguins[penguins["species"] == "Gentoo" & penguins["body_mass_g"] > 5000]')
demo = penguins.copy()
try_it('demo[demo["sex"].isna()]["sex"] = "unknown"')
print('"unknown" in demo after the chained assignment:', (demo["sex"] == "unknown").sum())''',
       todo='''error_and = ...          # a) the name of the exception of attempt 1
error_no_parens = ...    # b) the name of the exception of attempt 2
n_heavy_gentoo = ...     # c)
fixed_df = penguins.copy()
# d) one line: fixed_df.loc[..., "sex"] = "unknown"
n_unknown = ...          # d) how many "unknown" in fixed_df
n_dream_torgersen = ...  # e)''',
       check='''wb.check("0A.58a", error_and)
wb.check("0A.58b", error_no_parens)
wb.check("0A.58c", n_heavy_gentoo)
wb.check("0A.58d", n_unknown)
wb.check("0A.58e", n_dream_torgersen)''',
       solution='''error_and = try_it('penguins[penguins["species"] == "Gentoo" and penguins["body_mass_g"] > 5000]')
error_no_parens = try_it('penguins[penguins["species"] == "Gentoo" & penguins["body_mass_g"] > 5000]')
n_heavy_gentoo = len(penguins[(penguins["species"] == "Gentoo") & (penguins["body_mass_g"] > 5000)])
fixed_df = penguins.copy()
fixed_df.loc[fixed_df["sex"].isna(), "sex"] = "unknown"
n_unknown = (fixed_df["sex"] == "unknown").sum()
n_dream_torgersen = penguins["island"].isin(["Dream", "Torgersen"]).sum()
print(error_and, error_no_parens, n_heavy_gentoo, n_unknown, n_dream_torgersen)''',
       record='''wb.record("0A.58a", error_and)
wb.record("0A.58b", error_no_parens)
wb.record("0A.58c", n_heavy_gentoo)
wb.record("0A.58d", n_unknown, mistakes={"l'affectation en chaîne modifie une copie temporaire : utilise fixed_df.loc[masque, \\"sex\\"]": 0})
wb.record("0A.58e", n_dream_torgersen)''',
       note="1. `and` demande **un** booléen à chaque Series : pandas refuse de trancher (`ValueError`). 2. `&` passe "
            "avant `==` et `>` : Python calcule d'abord `\"Gentoo\" & penguins[\"body_mass_g\"]`, qui n'a pas de sens "
            "(`TypeError`). 3. `demo[masque]` crée une copie temporaire : l'affectation la modifie, elle, et `demo` ne "
            "change pas (avec un `SettingWithCopyWarning`)."),

    Ex("0A.59", "📦", 2, 26, "Figures à plusieurs panneaux : `subplots`, `imshow`, `show_images`",
       "construire une figure à plusieurs panneaux, afficher des images et comparer des distributions.",
       "Ex 0A.36, Ex 0A.54 · fiche §100.10.2", thread="MNIST", tracks="C",
       body="""Trace trois figures :

1. `fig_digits` : une grille 2 × 5 (`plt.subplots(2, 5, figsize=(9, 4))`) avec, dans chaque case, **l'image moyenne** d'un chiffre (`images[labels == d].mean(axis=0)`), affichée avec `imshow(..., cmap="gray_r")`, le chiffre en titre, et sans axes (`ax.axis("off")`). Parcours les cases avec `axes.ravel()`, qui aplatit la grille `(2, 5)` en 10 cases.
2. les 16 premières images avec leur étiquette, en une seule instruction : `wb.plot.show_images(images[:16], labels=labels[:16])`.
3. `fig_hist` : une ligne de trois panneaux (`plt.subplots(1, 3, figsize=(12, 3.5))`), un par mesure (bec, nageoire, masse), avec dans chacun l'histogramme **de chaque espèce** (`alpha=0.5`, `label=espèce`), un titre et une légende.

La cellule de vérification contrôle la structure de `fig_digits` et `fig_hist` (nombre de panneaux, images, titres).""",
       todo='''fig_digits = ...   # 1. the 2 x 5 figure of the mean digits (then plt.show())
# 2. wb.plot.show_images(...)
fig_hist = ...     # 3. the 1 x 3 figure of histograms per species''',
       check='''def figure_report(fig, n_panels, need_images=False):
    """Check the structure of a matplotlib figure; return a list of problems."""
    problems = []
    axes = fig.axes
    if len(axes) != n_panels:
        problems.append(f"{len(axes)} panneaux au lieu de {n_panels}")
    if need_images and not all(len(ax.images) == 1 for ax in axes):
        problems.append("chaque panneau doit contenir une image (imshow)")
    if not all(ax.get_title() for ax in axes):
        problems.append("un panneau n'a pas de titre")
    return problems


for name, fig, n, need in [("fig_digits", fig_digits, 10, True), ("fig_hist", fig_hist, 3, False)]:
    if fig is ...:
        print(f"⏳ Ex 0A.59 : {name} pas encore fait.")
        continue
    problems = figure_report(fig, n, need)
    print(f"✅ Ex 0A.59 : {name} a la bonne structure." if not problems else
          f"❌ Ex 0A.59 : {name} : " + " ; ".join(problems))''',
       solution='''fig_digits, axes = plt.subplots(2, 5, figsize=(9, 4))
for d, ax in enumerate(axes.ravel()):
    ax.imshow(images[labels == d].mean(axis=0), cmap="gray_r")
    ax.set_title(str(d))
    ax.axis("off")
fig_digits.suptitle("Mean image of each digit (1000 MNIST images)")
fig_digits.tight_layout()
plt.show()

wb.plot.show_images(images[:16], labels=labels[:16])
plt.show()

fig_hist, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col in zip(axes, ["bill_length_mm", "flipper_length_mm", "body_mass_g"]):
    for species, group in clean.groupby("species"):
        ax.hist(group[col], bins=20, alpha=0.5, label=species)
    ax.set_title(col)
    ax.set_xlabel(col.split("_")[-1])
    ax.legend()
axes[0].set_ylabel("number of penguins")
fig_hist.tight_layout()
plt.show()'''),

    Ex("0A.60", "📈", 2, 15, "Quelle mesure sépare le mieux les espèces ?",
       "lire des histogrammes et un nuage de points pour choisir les mesures qui distinguent les classes.",
       "Ex 0A.57, Ex 0A.59 · fiche §100.10.1, §100.9.4", thread="Penguins", tracks="C",
       body="""La cellule fournie trace, pour chaque mesure, les histogrammes des trois espèces, puis le nuage longueur × profondeur du bec. **Lis les graphiques** (sans calculer) et réponds par une chaîne :

a) Sur quelle mesure les Adelie et les Chinstrap se distinguent-ils **nettement** ? (le nom de la colonne)
b) Un manchot a une nageoire de 225 mm : de quelle espèce est-il ?
c) Un autre a un bec de 50 mm de long et de 19 mm de profondeur : de quelle espèce ?
d) Vérifie a) par le calcul : pour chaque mesure, l'écart entre les moyennes des Adelie et des Chinstrap divisé par leur écart-type moyen, $|\\bar{x}_A - \\bar{x}_C| / \\sqrt{(s_A^2 + s_C^2)/2}$ ; la mesure où ce rapport est le plus grand (fonction `separation`, puis `max(numeric_cols, key=...)`).""",
       given='''fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, col in zip(axes, numeric_cols):
    for species, group in clean.groupby("species"):
        ax.hist(group[col], bins=20, alpha=0.5, label=species)
    ax.set_title(col)
axes[0].legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(5.5, 4))
for species, group in clean.groupby("species"):
    ax.scatter(group["bill_length_mm"], group["bill_depth_mm"], s=12, label=species)
ax.set_xlabel("bill length (mm)")
ax.set_ylabel("bill depth (mm)")
ax.legend()
plt.show()''',
       todo='''best_feature = ...        # a) a column name
species_225 = ...         # b)
species_50_19 = ...       # c)


def separation(col):
    """|mean Adelie - mean Chinstrap| / sqrt((var Adelie + var Chinstrap) / 2) for one column."""
    raise NotImplementedError("separation() is not written yet")''',
       check='''wb.check("0A.60a", best_feature)
wb.check("0A.60b", species_225)
wb.check("0A.60c", species_50_19)
with wb.attempt("0A.60d"):
    wb.check("0A.60d", max(numeric_cols, key=separation))''',
       solution='''best_feature = "bill_length_mm"
species_225 = "Gentoo"
species_50_19 = "Chinstrap"


def separation(col):
    """|mean Adelie - mean Chinstrap| / sqrt((var Adelie + var Chinstrap) / 2) for one column."""
    a = clean.loc[clean["species"] == "Adelie", col]
    c = clean.loc[clean["species"] == "Chinstrap", col]
    return abs(a.mean() - c.mean()) / np.sqrt((a.var() + c.var()) / 2)


print({col: round(separation(col), 2) for col in numeric_cols}, max(numeric_cols, key=separation))''',
       record='''wb.record("0A.60a", best_feature)
wb.record("0A.60b", species_225)
wb.record("0A.60c", species_50_19, mistakes={"un bec de 50 mm est trop long pour un Adelie, et 19 mm trop profond pour un Gentoo": "Gentoo"})
wb.record("0A.60d", max(numeric_cols, key=separation))''',
       note="Le rapport vaut 3,3 pour le bec contre moins de 0,9 pour les autres mesures : les deux histogrammes du bec "
            "ne se chevauchent presque pas. C'est l'idée du critère de Fisher, et c'est pourquoi un modèle a besoin "
            "de **plusieurs** mesures : aucune ne sépare seule les trois espèces (ch. 7)."),
])

# ---------------------------------------------------------------------------
# Part H (0A.61 to 0A.67): tools of the trade, mylearn and the final challenge
# ---------------------------------------------------------------------------
PART_H_GIVEN = '''# Tools for part H
import doctest
import inspect

import pytest

species_list = penguins["species"].tolist()
island_list = penguins["island"].tolist()


def verdict(ex_id, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message."""
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")


def run_utils_tests(keyword, impl="learner"):
    """Run the tests of mylearn.utils whose name contains `keyword` (on YOUR code by default)."""
    command = [sys.executable, "-m", "pytest", "tests/test_ch00a_utils.py", "-k", keyword, "-q",
               "-p", "no:cacheprovider", "--color=no", "-rf", "--tb=line"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    for line in [line for line in lines if line.startswith("FAILED")][:8]:
        print(line[:200])
    print("pytest:", lines[-1] if lines else result.stderr.strip()[-300:])'''

MYLEARN_HOWTO = ("> **Mode d'emploi mylearn** (comme en 0A.26) : ouvre `mon_travail/mylearn/utils.py` (créé par "
                 "`python tools/start_chapter.py 0A`), lis la docstring de la fonction, remplace le "
                 "`raise NotImplementedError(...)` par ton code et **enregistre**. La cellule de vérification "
                 "**recharge** ta librairie (inutile de redémarrer le noyau), vérifie quelques valeurs, puis lance "
                 "les tests de la fonction (`pytest -k <nom>`), qui comparent ton code à un oracle.")

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="0A")   # reload your saved file\n'

PART_H = Part("H", "Outils du pro, mylearn et défi final",
              "Fiche §100.5.5 et §100.11.3 à §100.11.5. Tu documentes et tu testes comme un développeur, puis tu "
              "écris les quatre fonctions de `mylearn.utils` qui serviront jusqu'au ch. 20, et tu finis par une "
              "enquête chiffrée sur les manchots.", given=PART_H_GIVEN, exercises=[
    Ex("0A.61", "🛠️", 2, 20, "Une docstring au format NumPy, vérifiée par doctest",
       "documenter une fonction au format NumPy, avec des exemples exécutables vérifiés par doctest.",
       "Ex 0A.26 · fiche §100.5.5, §100.11.4", tracks="R, C",
       body="""Écris `min_max_scale(values)` : elle renvoie un **array** NumPy des valeurs ramenées entre 0 et 1, $(x - \\min) / (\\max - \\min)$, et lève une `ValueError` si la séquence est vide ou si toutes les valeurs sont égales (division par zéro).

Donne-lui une docstring **complète** au format NumPy (modèle : la fonction `mean` de la fiche §100.5.5) : une phrase de résumé, puis les sections `Parameters`, `Returns`, `Raises` et `Examples`, avec au moins **deux** exemples `>>>` dont un qui montre la `ValueError` :

```text
>>> min_max_scale([5, 5])
Traceback (most recent call last):
    ...
ValueError: all values are equal
```

Pour la sortie attendue d'un exemple, exécute-le d'abord et **recopie exactement** ce que Python affiche (par exemple `array([0. , 0.5, 1. ])`). La cellule de vérification contrôle le calcul, les quatre sections, puis exécute tes exemples avec doctest.""",
       todo='''def min_max_scale(values):
    """TODO: write the docstring here."""
    raise NotImplementedError("min_max_scale() is not written yet")''',
       check='''with wb.attempt("0A.61"):
    result = min_max_scale([2, 4, 6])
    verdict("0A.61", np.allclose(result, [0, 0.5, 1]) and isinstance(result, np.ndarray),
            "le calcul est juste et renvoie un array.", "min_max_scale([2, 4, 6]) doit renvoyer l'array [0, 0.5, 1].")
    raised = []
    for bad in ([], [3, 3]):
        try:
            min_max_scale(bad)
        except ValueError:
            raised.append(bad)
    verdict("0A.61", len(raised) == 2, "les entrées invalides lèvent une ValueError.",
            "une liste vide et des valeurs toutes égales doivent lever une ValueError.")
    doc = inspect.getdoc(min_max_scale) or ""
    missing = [s for s in ("Parameters", "Returns", "Raises", "Examples")
               if not re.search(rf"^{s}\\n-{{3,}}$", doc, flags=re.MULTILINE)]
    verdict("0A.61", not missing, "les quatre sections de la docstring sont là.", f"sections manquantes : {missing}")
    runner = doctest.DocTestRunner(verbose=False)
    for test in doctest.DocTestFinder().find(min_max_scale, "min_max_scale", globs={"np": np, "min_max_scale": min_max_scale}):
        runner.run(test)
    failed, attempted = runner.summarize(verbose=False)
    verdict("0A.61", attempted >= 2 and failed == 0, f"doctest : {attempted} exemples, tous justes.",
            f"doctest : {failed} exemple(s) faux sur {attempted} (détail au-dessus), ou moins de 2 exemples.")''',
       solution='''def min_max_scale(values):
    """Rescale values to the interval [0, 1].

    Parameters
    ----------
    values : sequence of float
        The numbers to rescale (list, tuple or 1-D array), not all equal.

    Returns
    -------
    np.ndarray
        ``(values - min) / (max - min)``: the smallest value becomes 0, the largest 1.

    Raises
    ------
    ValueError
        If ``values`` is empty or if all the values are equal.

    Examples
    --------
    >>> min_max_scale([2, 4, 6])
    array([0. , 0.5, 1. ])
    >>> min_max_scale([5, 5])
    Traceback (most recent call last):
        ...
    ValueError: all values are equal
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("min_max_scale() needs at least one value")
    span = values.max() - values.min()
    if span == 0:
        raise ValueError("all values are equal")
    return (values - values.min()) / span


runner = doctest.DocTestRunner(verbose=False)
for test in doctest.DocTestFinder().find(min_max_scale, "min_max_scale", globs={"np": np, "min_max_scale": min_max_scale}):
    runner.run(test)
print(runner.summarize(verbose=False), min_max_scale([181, 190, 231]))
help(min_max_scale)'''),

    Ex("0A.62", "🛠️", 2, 25, "Écrire tes propres tests : `assert`, `approx`, `raises`, `parametrize`",
       "écrire des tests pytest qui passent sur un code juste et attrapent des codes faux.",
       "Ex 0A.61 · fiche §100.11.3", tracks="R, C",
       body="""Un bon test **échoue quand le code est faux**. Tu disposes d'une version juste de `min_max_scale` (`reference_scale`) et de trois versions **buggées** (lis-les : où est l'erreur ?).

Écris au moins **quatre** tests pytest pour une fonction nommée `min_max_scale` :
- un test simple avec `assert` (trois valeurs entières) ;
- un test sur des flottants avec `pytest.approx` ;
- un test d'erreur avec `with pytest.raises(ValueError):` ;
- un test paramétré avec `@pytest.mark.parametrize` qui vérifie, sur plusieurs listes, que le minimum du résultat vaut 0 et le maximum 1.

Range-les dans la liste `my_tests`. La cellule de vérification lance **vraiment** pytest (`wb.run_pytest`, qui copie tes tests dans un fichier temporaire) : tes tests doivent **passer** sur `reference_scale` et **échouer** sur chacune des trois versions buggées. Tes tests ne peuvent utiliser que `min_max_scale`, `np`, `math` et `pytest`.""",
       given='''def reference_scale(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.max() == values.min():
        raise ValueError("all values are equal")
    return (values - values.min()) / (values.max() - values.min())


def scale_bug_1(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.max() == values.min():
        raise ValueError("all values are equal")
    return (values - values.min()) / values.max()


def scale_bug_2(values):
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    return (values - values.min()) / span if span else np.zeros(len(values))


def scale_bug_3(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.max() == values.min():
        raise ValueError("all values are equal")
    return (values.max() - values) / (values.max() - values.min())''',
       todo='''# Write your test functions here; they call min_max_scale. For example:
# def test_three_integers():
#     assert list(min_max_scale([2, 4, 6])) == [0.0, 0.5, 1.0]

my_tests = ...   # the list of your test functions, e.g. [test_three_integers, ...]''',
       check='''if my_tests is ...:
    print("⏳ Ex 0A.62 : pas encore fait.")
else:
    good = wb.run_pytest(my_tests, subject=reference_scale, name="min_max_scale")
    verdict("0A.62", good.ok and good.passed >= 4, f"tes tests passent sur la version juste ({good.passed} cas).",
            "tes tests doivent tous passer sur la version juste, et être au moins 4 (voir le détail au-dessus).")
    if good.ok:
        for bug in [scale_bug_1, scale_bug_2, scale_bug_3]:
            result = wb.run_pytest(my_tests, subject=bug, name="min_max_scale", quiet=True)
            caught = result.failed + result.errors > 0
            verdict("0A.62", caught, f"{bug.__name__} est attrapé ({result.failed} test(s) en échec).",
                    f"tes tests passent sur {bug.__name__}, qui est faux : ajoute un test qui le fait échouer.")''',
       solution='''def test_three_integers():
    assert list(min_max_scale([2, 4, 6])) == [0.0, 0.5, 1.0]


def test_floats():
    result = min_max_scale([0.1, 0.2, 0.4])
    assert result[1] == pytest.approx(1 / 3)


def test_equal_values_raise():
    with pytest.raises(ValueError):
        min_max_scale([5, 5, 5])


@pytest.mark.parametrize("values", [[1, 2], [3, -1, 7], [181, 231, 190, 172]])
def test_bounds(values):
    result = min_max_scale(values)
    assert result.min() == 0 and result.max() == 1


my_tests = [test_three_integers, test_floats, test_equal_values_raise, test_bounds]

good = wb.run_pytest(my_tests, subject=reference_scale, name="min_max_scale")
for bug in [scale_bug_1, scale_bug_2, scale_bug_3]:
    result = wb.run_pytest(my_tests, subject=bug, name="min_max_scale", quiet=True)
    print(f"{bug.__name__}: {result.summary}")''',
       note="Chaque bug est attrapé par un test différent : `scale_bug_1` (dénominateur faux) par les valeurs "
            "attendues, `scale_bug_2` (pas d'erreur) par `pytest.raises`, `scale_bug_3` (échelle inversée) par le "
            "test simple. Écrire un test qui échoue sur un code faux connu, c'est le principe du *mutation testing*."),

    Ex("0A.63", "🔨", 2, 26, "`utils.count_values` : compter sans pandas",
       "implémenter une fonction de mylearn à partir de sa docstring et la valider contre un oracle.",
       "Ex 0A.26, Ex 0A.24 · fiche §100.11.5, §100.3.3, §100.3.5", thread="Penguins", tracks="M, C",
       body=MYLEARN_HOWTO + """

Écris `count_values(values, normalize=False)` **sans** `Counter` ni pandas : un dictionnaire et une boucle (fiche §100.3.3). Relis la docstring : l'ordre des clés, `normalize`, et les deux cas d'erreur (vide, et `NaN`, qui n'est égal à rien, pas même à lui-même : `x != x` est vrai seulement pour `NaN`).

Vérifications : a) `count_values(species_list)["Adelie"]` · b) `count_values(island_list, normalize=True)["Torgersen"]` (3 décimales) · c) le nombre de clés de `count_values("Gentoo")` · d) `count_values(penguins["sex"])` doit lever une `ValueError`, car 11 sexes manquent : la cellule affiche le nom de l'exception levée · puis les tests `-k count_values`.""",
       check=RELOAD + '''with wb.attempt("0A.63"):
    wb.check("0A.63a", mylearn.utils.count_values(species_list)["Adelie"])
    wb.check("0A.63b", mylearn.utils.count_values(island_list, normalize=True)["Torgersen"])
    wb.check("0A.63c", len(mylearn.utils.count_values("Gentoo")))
    try:
        mylearn.utils.count_values(penguins["sex"])
        raised = "no error"
    except NotImplementedError:
        raise
    except Exception as error:  # noqa: BLE001 - we want the name of whatever is raised
        raised = type(error).__name__
    wb.check("0A.63d", raised)
    run_utils_tests("count_values")''',
       solution='''print(mylearn.utils.count_values(species_list))
print(mylearn.utils.count_values(island_list, normalize=True))
print(mylearn.utils.count_values("Gentoo"))
try:
    mylearn.utils.count_values(penguins["sex"])
    raised = "no error"
except ValueError as error:
    raised = type(error).__name__
    print("ValueError:", error)
run_utils_tests("count_values", impl="ref")''',
       record='''wb.record("0A.63a", mylearn.utils.count_values(species_list)["Adelie"])
wb.record("0A.63b", mylearn.utils.count_values(island_list, normalize=True)["Torgersen"], decimals=3)
wb.record("0A.63c", len(mylearn.utils.count_values("Gentoo")), mistakes={"une clé par lettre DIFFÉRENTE : le o apparaît deux fois": 6})
wb.record("0A.63d", raised, mistakes={"NaN n'est égal à rien, pas même à lui-même : repère-le avec v != v et lève une ValueError": "no error"})''',
       note="La référence est dans `solutions/mylearn_ref/utils.py` : lis-la **après** avoir réussi les tests. Le "
            "test de l'ordre des clés compare ton dictionnaire à `dict(Counter(values))` : un dictionnaire Python "
            "garde l'ordre d'insertion."),

    Ex("0A.64", "🔨", 3, 35, "`utils.argmax` : le premier maximum, avec des boucles",
       "écrire `argmax` avec des boucles, pour 1 ou 2 dimensions et selon un axe, en respectant la règle des égalités.",
       "Ex 0A.63, Ex 0A.51 · fiche §100.8.5, §100.11.5", tracks="M, C",
       body=MYLEARN_HOWTO + """

Écris `argmax(values, axis=None)` avec des boucles `for` : `np.argmax` et `np.max` sont interdits dans le corps de la fonction (ce sont les oracles des tests). Avance par étapes, en relançant la vérification à chaque fois :
1. le cas 1-D : garde l'indice du meilleur élément vu jusqu'ici ; un élément **strictement** plus grand le remplace (ainsi, en cas d'égalité, le premier gagne) ;
2. `axis=None` en 2-D : la position dans le tableau aplati (`np.asarray(values).ravel()`) ;
3. `axis=1` (et `-1`) : un indice par ligne ; `axis=0` : un indice par colonne (pense à la transposée `.T`) ;
4. les erreurs (`ValueError`) : tableau vide, `NaN`, plus de deux dimensions, axe invalide.

La cellule compare ton `argmax` à `np.argmax` sur les scores `scores` (cinq manchots, trois classes), puis lance les tests `-k argmax`.""",
       given='''scores = np.array([[0.2, 0.5, 0.3],
                   [0.7, 0.1, 0.2],
                   [0.1, 0.1, 0.8],
                   [0.4, 0.4, 0.2],
                   [0.3, 0.3, 0.4]])   # one row per penguin, one column per class (with ties)''',
       check=RELOAD + '''with wb.attempt("0A.64"):
    for axis in [None, 0, 1]:
        mine = mylearn.utils.argmax(scores, axis=axis)
        verdict("0A.64", np.array_equal(mine, np.argmax(scores, axis=axis)),
                f"axis={axis} : {mine} comme np.argmax.", f"axis={axis} : {mine} au lieu de {np.argmax(scores, axis=axis)}.")
    run_utils_tests("argmax")''',
       solution='''for axis in [None, 0, 1]:
    print(axis, mylearn.utils.argmax(scores, axis=axis), np.argmax(scores, axis=axis))
run_utils_tests("argmax", impl="ref")''',
       note="Ligne 4 de `scores` : `0.4` apparaît deux fois, et c'est l'indice 0 qui gagne. Cette règle (« le "
            "premier ») est celle de NumPy et de PyTorch : avec `>=` au lieu de `>`, tu obtiendrais le dernier."),

    Ex("0A.65", "🔨", 2, 26, "`utils.one_hot` : des étiquettes aux vecteurs",
       "encoder des étiquettes entières en vecteurs one-hot, avec validation des entrées.",
       "Ex 0A.64 · fiche §100.8.2, §100.11.5", thread="Penguins", tracks="C",
       body=MYLEARN_HOWTO + """

Écris `one_hot(y, n_classes=None, dtype=np.float64)` : la ligne `i` du résultat contient un 1 dans la colonne `y[i]` et des 0 ailleurs. Méthode conseillée : une matrice de zéros `(n, n_classes)`, puis un seul 1 par ligne grâce à l'indexation `M[np.arange(n), y] = 1`. Valide d'abord les étiquettes (docstring).

On code les espèces des 333 manchots complets par des entiers (`y_codes` : 0 = Adelie, 1 = Chinstrap, 2 = Gentoo). Vérifications : a) la forme de `one_hot(y_codes)` · b) la somme de chaque colonne · c) `np.argmax(one_hot(y_codes), axis=1)` redonne-t-il `y_codes` ? · d) le type des éléments de `one_hot([1, 0], n_classes=3, dtype=int)` (son nom, par exemple `"int64"`) · puis les tests `-k one_hot`.""",
       given='''codes = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
y_codes = np.array([codes[s] for s in clean["species"]])
print(y_codes[:5], y_codes.shape)''',
       check=RELOAD + '''with wb.attempt("0A.65"):
    Y = mylearn.utils.one_hot(y_codes)
    wb.check("0A.65a", Y.shape)
    wb.check("0A.65b", Y.sum(axis=0))
    wb.check("0A.65c", np.array_equal(np.argmax(Y, axis=1), y_codes))
    wb.check("0A.65d", mylearn.utils.one_hot([1, 0], n_classes=3, dtype=int).dtype.name)
    run_utils_tests("one_hot")''',
       solution='''Y = mylearn.utils.one_hot(y_codes)
print(Y[:4], Y.shape, Y.sum(axis=0), mylearn.utils.one_hot([1, 0], n_classes=3, dtype=int))
run_utils_tests("one_hot", impl="ref")''',
       record='''wb.record("0A.65a", Y.shape, mistakes={"une ligne par exemple, une colonne par classe": (3, 333)})
wb.record("0A.65b", Y.sum(axis=0))
wb.record("0A.65c", np.array_equal(np.argmax(Y, axis=1), y_codes))
wb.record("0A.65d", mylearn.utils.one_hot([1, 0], n_classes=3, dtype=int).dtype.name)''',
       note="`argmax` est l'inverse de `one_hot` : c'est ainsi qu'on passe des probabilités d'un réseau (une ligne "
            "par exemple) à une classe prédite (ch. 18). Le one-hot des entrées catégorielles (l'île, par exemple) "
            "revient au ch. 12."),

    Ex("0A.66", "🔨", 3, 39, "`utils.iterate_minibatches` : découper un dataset en mini-lots",
       "découper les indices d'un dataset en mini-lots mélangés, reproductibles, avec ou sans le dernier lot incomplet.",
       "Ex 0A.5, Ex 0A.31, Ex 0A.26 · fiche §100.8.7, §100.11.5", thread="Penguins", tracks="R, C",
       body=MYLEARN_HOWTO + """

Écris `iterate_minibatches(n_samples, batch_size, shuffle=True, rng=None, drop_last=False)`. La docstring impose l'algorithme exact (les tests le comparent au `BatchSampler` de PyTorch) : l'ordre des indices est `rng.permutation(n_samples)` (ou `0, 1, …` sans mélange), puis on le coupe en tranches consécutives de `batch_size`. Relis 0A.5 (combien de lots ?) et 0A.31 (le générateur).

Vérifications, avec `rng = np.random.default_rng(0)` recréé à chaque question : a) le nombre de lots pour 333 manchots et des lots de 64 · b) la liste des tailles des lots · c) les 5 premiers indices du premier lot · d) le nombre de lots avec `drop_last=True` · e) une boucle de 3 epochs avec **un seul** générateur `np.random.default_rng(1)` : le nombre total de mises à jour (un lot = une mise à jour) · f) les deux premières epochs ont-elles un ordre différent ? · puis les tests `-k iterate_minibatches`.""",
       check=RELOAD + '''with wb.attempt("0A.66"):
    iterate = mylearn.utils.iterate_minibatches
    batches_0 = iterate(333, 64, rng=np.random.default_rng(0))
    wb.check("0A.66a", len(batches_0))
    wb.check("0A.66b", [len(b) for b in batches_0])
    wb.check("0A.66c", batches_0[0][:5])
    wb.check("0A.66d", len(iterate(333, 64, rng=np.random.default_rng(0), drop_last=True)))
    rng = np.random.default_rng(1)
    epochs = [iterate(333, 64, rng=rng) for epoch in range(3)]
    wb.check("0A.66e", sum(len(batches) for batches in epochs))
    wb.check("0A.66f", not np.array_equal(np.concatenate(epochs[0]), np.concatenate(epochs[1])))
    run_utils_tests("iterate_minibatches")''',
       solution='''iterate = mylearn.utils.iterate_minibatches
batches_0 = iterate(333, 64, rng=np.random.default_rng(0))
rng = np.random.default_rng(1)
epochs = [iterate(333, 64, rng=rng) for epoch in range(3)]
print([len(b) for b in batches_0], batches_0[0][:5], np.random.default_rng(0).permutation(333)[:5])
print([len(batches) for batches in epochs], [e[0][:3] for e in epochs])

# how it is used in a training loop (chapters 18 to 20):
X_train, y_train = X, y_codes
for idx in iterate(len(X_train), 64, rng=np.random.default_rng(0))[:2]:
    X_batch, y_batch = X_train[idx], y_train[idx]
    print(X_batch.shape, y_batch[:5])
run_utils_tests("iterate_minibatches", impl="ref")''',
       record='''wb.record("0A.66a", len(batches_0), mistakes={"le dernier lot, incomplet, est gardé : 6 lots": 5})
wb.record("0A.66b", [len(b) for b in batches_0])
wb.record("0A.66c", batches_0[0][:5], mistakes={"l'ordre doit être rng.permutation(n_samples), puis des tranches consécutives": np.arange(5)})
wb.record("0A.66d", len(iterate(333, 64, rng=np.random.default_rng(0), drop_last=True)))
wb.record("0A.66e", sum(len(batches) for batches in epochs), mistakes={"3 epochs × 6 lots : une mise à jour par lot": 3})
wb.record("0A.66f", not np.array_equal(np.concatenate(epochs[0]), np.concatenate(epochs[1])))''',
       note="Le **même** générateur donne un nouvel ordre à chaque epoch, et toute la suite reste reproductible : "
            "relancer le notebook redonne exactement les mêmes lots. Recréer `default_rng(1)` à chaque epoch "
            "donnerait trois fois le même ordre, une erreur classique."),

    Ex("0A.67", "🏆", 3, 45, "Enquête : dix questions sur les manchots, dix réponses vérifiées",
       "répondre seul à dix questions sur un dataset en combinant filtres, groupby, tri et réductions.",
       "Ex 0A.51, Ex 0A.57, Ex 0A.60 · fiche §100.9.4, §100.8.5, §100.10.1", thread="Penguins", tracks="C",
       body="""🏆 **Défi** : dix questions sur les 344 manchots de `penguins`, **dix ✅** en moins de 45 minutes, sans regarder les solutions. Chaque réponse tient en une à trois lignes de pandas ; relis la question deux fois (quelle population ? quel arrondi ?).

a) L'île qui accueille le plus de manchots **femelles**.
b) L'espèce dont le bec est, en moyenne, le plus **long**.
c) Le manchot le plus **léger** : son espèce et son île, au format `"Espèce, Île"`.
d) L'écart entre la masse moyenne des Gentoo mâles et celle des Gentoo femelles, en grammes (0 décimale).
e) L'année où la masse moyenne des Adelie a été la plus élevée.
f) Parmi les 342 nageoires mesurées, la proportion d'au moins 200 mm (3 décimales).
g) La médiane de la masse des Chinstrap.
h) Le nombre de manchots de sexe inconnu sur l'île Dream.
i) Parmi les manchots d'au moins 4500 g, la proportion de Gentoo (3 décimales).
j) L'espèce dont la longueur de nageoire est la plus **dispersée** (le plus grand écart-type).

**Bonus portfolio** : une figure qui illustre ta réponse la plus intéressante, et trois phrases de conclusion dans une cellule Markdown, comme pour un rapport.""",
       todo='''answer_a = ...
answer_b = ...
answer_c = ...   # "Species, Island"
answer_d = ...
answer_e = ...
answer_f = ...
answer_g = ...
answer_h = ...
answer_i = ...
answer_j = ...''',
       check='''for letter, answer in zip("abcdefghij", [answer_a, answer_b, answer_c, answer_d, answer_e,
                                          answer_f, answer_g, answer_h, answer_i, answer_j]):
    wb.check(f"0A.67{letter}", answer)''',
       solution='''answer_a = penguins.loc[penguins["sex"] == "female", "island"].value_counts().idxmax()
answer_b = penguins.groupby("species")["bill_length_mm"].mean().idxmax()
lightest = penguins.loc[penguins["body_mass_g"].idxmin()]
answer_c = f"{lightest['species']}, {lightest['island']}"
gentoo_by_sex = penguins[penguins["species"] == "Gentoo"].groupby("sex")["body_mass_g"].mean()
answer_d = gentoo_by_sex["male"] - gentoo_by_sex["female"]
answer_e = penguins[penguins["species"] == "Adelie"].groupby("year")["body_mass_g"].mean().idxmax()
flippers_known = penguins["flipper_length_mm"].dropna()
answer_f = (flippers_known >= 200).mean()
answer_g = penguins.loc[penguins["species"] == "Chinstrap", "body_mass_g"].median()
answer_h = (penguins["sex"].isna() & (penguins["island"] == "Dream")).sum()
heavy = penguins[penguins["body_mass_g"] >= 4500]
answer_i = (heavy["species"] == "Gentoo").mean()
answer_j = penguins.groupby("species")["flipper_length_mm"].std().idxmax()
print(answer_a, answer_b, answer_c, round(answer_d), answer_e, round(answer_f, 3), answer_g, answer_h,
      round(answer_i, 3), answer_j, sep=" | ")''',
       record='''wb.record("0A.67a", answer_a)
wb.record("0A.67b", answer_b, mistakes={"Gentoo a le plus grand bec en masse, mais le plus LONG en moyenne est celui d'une autre espèce": "Gentoo"})
wb.record("0A.67c", answer_c)
wb.record("0A.67d", answer_d, decimals=0)
wb.record("0A.67e", answer_e)
wb.record("0A.67f", answer_f, decimals=3, mistakes={"divise par les 342 nageoires mesurées, pas par 344": (flippers_known >= 200).sum() / 344,
                                                    "« au moins 200 mm » inclut 200 : >=": (flippers_known > 200).mean()})
wb.record("0A.67g", answer_g, decimals=0)
wb.record("0A.67h", answer_h)
wb.record("0A.67i", answer_i, decimals=3, mistakes={"la proportion de Gentoo PARMI les manchots lourds, pas l'inverse": (heavy["species"] == "Gentoo").sum() / (penguins["species"] == "Gentoo").sum()})
wb.record("0A.67j", answer_j)''',
       after=[("md", "**Mon rapport d'enquête** (bonus) : ma figure, puis trois phrases de conclusion.")],
       note="Plusieurs chemins mènent aux mêmes réponses (`value_counts`, `groupby`, masques). Ce qui compte en "
            "entreprise : la population exacte de chaque question (les manquants !) et un résultat relu avant d'être "
            "annoncé."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D, PART_E, PART_F, PART_G, PART_H]


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------
def all_exercises() -> list[Ex]:
    return [ex for part in PARTS for ex in part.exercises]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 0A.1 à 0A.8 | vérifier tes exercices papier | ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {ex.title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if kind == "exercise":
        title = "# 0A · Python, notebooks et outils — notebook d'exercices"
        how = ("Chaque exercice : un énoncé, une cellule à compléter (les `...` et les `raise NotImplementedError`), "
               "puis une cellule de vérification (`wb.check`). « Exécuter tout » va jusqu'au bout même si rien "
               "n'est rempli : les exercices pas faits affichent ⏳. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch00a_python/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 0A`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 0A · Python, notebooks et outils — solutions (notebook exécuté)"
        how = ("Les solutions des exercices du notebook, exécutées. Les explications détaillées (le *pourquoi*, "
               "les erreurs fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées "
               "`answer` enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Utiliser un notebook sans se perdre : ordre d'exécution, « Run all », `wb.check`.\n"
               "- Écrire du Python lisible : types, listes, dictionnaires, ensembles, boucles, compréhensions, fonctions, exceptions.\n"
               "- Faire ses premiers pas avec NumPy (arrays, masques, vectorisation, aléatoire) et pandas (lire, sélectionner, valeurs manquantes, `X` et `y`).\n"
               "- Lire un traceback, travailler avec des fichiers (texte, JSON), écrire des fonctions et des classes plus riches (fermetures, récursivité, méthodes spéciales, héritage, générateurs).\n"
               "- Maîtriser les axes, le broadcasting et les formes de NumPy ; regrouper avec pandas ; composer des figures à plusieurs panneaux.\n"
               "- Documenter et tester comme un développeur (docstring NumPy, doctest, pytest), et écrire les premières fonctions de ta librairie mylearn.\n\n"
               "**Rappel express.** L'état du notebook dépend de l'ordre d'exécution ; les indices commencent à 0 et la fin "
               "d'une tranche est exclue ; une fonction **renvoie** avec `return` ; un array NumPy n'a qu'un type et se "
               "calcule sans boucle ; `axis=0` donne une valeur par colonne.")]


def exercise_cells(ex: Ex, kind: str) -> list:
    cells = [md(ex.header() + "\n\n" + ex.body)]
    if ex.given:
        cells.append(code(ex.given))
    if kind == "exercise":
        if ex.hypothesis:
            cells.append(md("📝 **Mon hypothèse** (à écrire **avant** d'exécuter quoi que ce soit) : …"))
        if ex.todo:
            cells.append(code(ex.todo))
        if ex.check:
            cells.append(code(ex.check))
    else:
        if ex.solution:
            cells.append(code(ex.solution))
        if ex.record:
            cells.append(code(ex.record, tags=["answer"]))
    for cell_kind, text in ex.after:
        cells.append(md(text) if cell_kind == "md" else code(text))
    if kind == "solution" and ex.note:
        cells.append(md(f"💡 {ex.note}"))
    return cells


def footer_cells(kind: str) -> list:
    text = ("## ✅ Bilan du chapitre 0A\n\n"
            "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
            "1. Sais-tu dire, sans essayer, ce que valent `l[2:5]`, `l[-1]` et `len(l[2:])`, et la forme de `X.mean(axis=0)` ?\n"
            "2. Sais-tu lire un traceback, écrire une fonction qui refuse une entrée invalide, et un test pytest qui le vérifie ?\n"
            "3. Sais-tu passer d'un DataFrame pandas à `X` et `y`, standardiser `X` par broadcasting et le découper en mini-lots ?\n\n"
            "**Pour aller plus loin** : refais 0A.67 avec un autre dataset (`wb.datasets.load_california()`) ; lis le code de "
            "la référence `solutions/mylearn_ref/utils.py` et compare-le au tien ; commite ta librairie (`git add mon_travail/mylearn`).\n\n"
            "**Et maintenant ?** Le chapitre 0B (maths du lycée au machine learning) utilise tout ce que tu viens d'apprendre : "
            "vecteurs et matrices en NumPy, fonctions, graphiques.")
    return [md(text)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind)
    for part in PARTS:
        cells.append(md(f"## Partie {part.key} · {part.title}\n\n{part.intro}"))
        if part.given:
            cells.append(code(part.given))
        for ex in part.exercises:
            cells += exercise_cells(ex, kind)
    cells += footer_cells(kind)
    return cells


def main() -> int:
    write_notebook(f"{FOLDER}/03_notebook.ipynb", build("exercise"))
    write_notebook(f"{FOLDER}/05_solutions.ipynb", build("solution"))
    print(f"✅ {FOLDER}/03_notebook.ipynb and 05_solutions.ipynb written "
          f"({len(all_exercises())} notebook exercises + 8 paper checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
