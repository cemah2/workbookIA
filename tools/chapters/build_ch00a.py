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

PARTS = [PART_A, PART_B, PART_C, PART_D, PART_E]


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
    rows += ["| F, G, H | 0A.37 à 0A.67 | Python avancé, NumPy/pandas avancés, outils et mylearn | | | *prochaine session* |"]
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
               "- Écrire et tester ta première fonction de mylearn.\n\n"
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
    text = ("## ✅ Bilan des parties A à E\n\n"
            "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
            "1. Sais-tu dire, sans essayer, ce que valent `l[2:5]`, `l[-1]` et `len(l[2:])` ?\n"
            "2. Sais-tu écrire une fonction qui lève une `ValueError` sur une entrée invalide, et l'appeler dans un `try` ?\n"
            "3. Sais-tu passer d'un DataFrame pandas à `X` et `y` en NumPy, et donner leurs formes ?\n\n"
            "**Pour aller plus loin** : refais 0A.21 et 0A.22 en inversant les rôles (la boucle en compréhension, "
            "et inversement) ; puis explore `penguins_raw` (dates, commentaires) : c'est le terrain de jeu des parties F et G.\n\n"
            "*Les parties F (Python intermédiaire et avancé), G (NumPy, pandas et matplotlib avancés) et H (outils du pro, "
            "mylearn et défi final) seront ajoutées à la prochaine session de génération.*")
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
