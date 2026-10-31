#!/usr/bin/env python
"""Build the two notebooks of chapter 29 from a single source (used by Claude).

    python tools/chapters/build_ch29.py
    python tools/run_all_notebooks.py chapitres/ch29_datasets/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch29_datasets/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q8, of the recalls R1 to R3 and of the ✏️ paper exercises 29.1
and 29.2 (02_exercices.md). The estimate 🧮 29.3, the reflection (🗣️ 29.4, ⚖️ 29.5, 📄 29.6) and the interview are
corrected with 05_solutions.md. Parts A and B (exercises 29.7 to 29.16, tools/chapters/ch29_parts.py and
ch29_code.py): loading and searching datasets (scikit-learn, Hugging Face, OpenML), then auditing and documenting
them (profile, near-duplicates, distribution shift, data card and verified download, the trap dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Paper, badge, check_unique_keys, code, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "29"
FOLDER = "chapitres/ch29_datasets"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
MB, GB = 1e6, 1e9                       # the conventions of the chapter: 1 MB = 10**6 bytes
MNIST_VALUES = 70_000 * 28 * 28         # uint8 values of MNIST (one channel)
CIFAR_VALUES = 60_000 * 32 * 32 * 3     # uint8 values of CIFAR-10 (three channels)
COUNTS_29_2 = {"A": 1200, "B": 600, "C": 200}
N_29_2 = sum(COUNTS_29_2.values())
'''

exec(PAPER_CONTEXT)  # the builder computes a few mistakes with the same constants

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "B"'
LETTERS = 'the letters, e.g. "ABD"'
WHOLE = "a whole number"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    Paper("29.Q1", "Partir de l'existant : modèle pré-entraîné, dataset public ou collecte ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "la collecte est la voie la plus chère : le livre la garde pour la fin",
                                           "C": "entraîner de zéro coûte le plus de calcul et de données : c'est le dernier recours"})),
        ("b", TF, "True", r'''mistakes={"réutiliser un réseau déjà entraîné en l'ajustant à sa tâche, c'est la définition du transfer learning": False}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "toujours, non : un modèle pré-entraîné sur des données très différentes peut faire moins bien",
                                           "B": "un modèle pré-entraîné hérite des biais de ses données d'entraînement"})),
        ("d", TF, "True", r'''mistakes={"public ne veut pas dire libre : c'est la licence qui dit si l'usage commercial est permis": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "entraîner avant de regarder, c'est découvrir les défauts trop tard",
                                           "C": "l'augmentation multiplie les défauts au lieu de les révéler"})),
    ]),
    Paper("29.Q2", "Licences et restrictions d'usage : vrai ou faux", [
        ("a", TF, "True", r'''mistakes={"CC BY n'interdit rien : elle demande seulement de citer la source": False}'''),
        ("b", TF, "False", r'''mistakes={"NC, c'est « non commercial » : pas de produit vendu": True}'''),
        ("c", TF, "False", r'''mistakes={"un prototype destiné à être vendu n'est pas de la recherche : la restriction s'applique": True}'''),
        ("d", TF, "False", r'''mistakes={"la licence porte sur les droits de celui qui publie, pas sur le consentement des personnes photographiées": True}'''),
        ("e", TF, "True", r'''mistakes={"publier des données personnelles ne les fait pas sortir du RGPD": False}'''),
    ]),
    Paper("29.Q3", "Les défauts qu'une inspection manuelle révèle", [
        ("a", LETTERS, '"ABC"', r'''mistakes={"les poids d'un réseau ne sont pas un défaut des données": "ABCD",
          "il manque les fautes d'orthographe": "AB", "il manque les champs vides": "BC",
          "il manque les labels qui n'apparaissent qu'une fois": "AC"}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "un poids négatif est impossible : un nombre rond et absurde code souvent une absence"})),
        ("c", TF, "False", r'''mistakes={"isna() ne voit que NaN et None : −999 est un nombre comme un autre pour pandas": True}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "un diamètre en millimètres n'est pas manquant : isna() ne le voit pas",
                                           "C": "un diamètre en millimètres ne crée pas de ligne en double"})),
        ("e", TF, "True", r'''mistakes={"relis l'encadré sur les audits : même les jeux de test des benchmarks ont des labels faux": False}'''),
    ]),
    Paper("29.Q4", "MNIST et Fashion-MNIST : même format, difficulté différente", [
        ("a", WHOLE, "70000", r'''mistakes={"c'est le seul jeu d'entraînement : ajoute les 10 000 images de test": 60000}'''),
        ("b", WHOLE, "28", r'''mistakes={"784 est le nombre de pixels d'une image, pas son côté": 784}'''),
        ("c", WHOLE, "10", r'''mistakes={"Fashion-MNIST garde les dix classes de MNIST, en vêtements": 9}'''),
        ("d", TF, "True", r'''mistakes={"même format, mêmes formes de tableaux : seule la source change": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "Fashion-MNIST est en niveaux de gris, comme MNIST",
                                           "C": "il a exactement autant d'exemples que MNIST"})),
    ]),
    Paper("29.Q5", "Les datasets de scikit-learn : lequel pour quelle tâche, et pourquoi Boston a disparu", [
        ("a", WHOLE, "1797", r'''mistakes={"c'est le chiffre du livre : load_digits n'en livre qu'une partie": 5620}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "en 0.24, load_boston était encore là, sans même d'avertissement",
                                           "C": "scikit-learn n'a pas de version 2.0 : le retrait date de la 1.2"})),
        ("c", LETTER, '"A"', letters("A", {"B": "des données anciennes ne justifient pas un retrait : relis la description de la variable B",
                                           "C": "le problème n'est pas la justesse des prix, mais une variable construite sur un critère racial"})),
        ("d", LETTER, '"C"', letters("C", {"A": "Iris est une classification de fleurs",
                                           "B": "Breast Cancer est une classification de tumeurs"})),
        ("e", TF, "False", r'''mistakes={"Diabetes est un tableau de dix mesures numériques et une cible : pas un texte": True}'''),
    ]),
    Paper("29.Q6", "Des datasets de Keras à torchvision et Hugging Face", [
        ("a", LETTER, '"C"', letters("C", {"A": "keras.datasets rend des tableaux NumPy, pas des DataFrames",
                                           "B": "le DataLoader est un objet de PyTorch, pas de Keras"})),
        ("b", WHOLE, "60000", r'''mistakes={"50 000, c'est l'entraînement seul : ajoute les 10 000 images de test": 50000}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "Keras livre IMDB déjà encodé : un entier par mot"})),
        ("d", LETTER, '"B"', letters("B", {"A": "torch.load relit un objet sauvegardé, il ne télécharge pas de dataset"})),
        ("e", TF, "True", r'''mistakes={"revision= désigne un commit précis du dépôt du dataset : la version est fixée": False}'''),
    ]),
    Paper("29.Q7", "Collections de datasets : UCI, Kaggle, OpenML, Hugging Face Hub et liens morts", [
        ("a", LETTERS, '"ABC"', r'''mistakes={"deeplearning.net est un lien mort (encadré 🕰️ du §29.4)": "ABCD",
          "il manque OpenML": "AB", "il manque Kaggle": "AC", "il manque le Hugging Face Hub": "BC"}'''),
        ("b", TF, "False", r'''mistakes={"l'outil kaggle lit une clé d'API personnelle, liée à un compte (encadré 🕰️ du §29.4)": True}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "torchvision fournit des datasets d'images, pas l'accès à OpenML",
                                           "B": "datasets charge le Hugging Face Hub, pas OpenML"})),
        ("d", TF, "False", r'''mistakes={"relis l'encadré 🕰️ du §29.4 : plusieurs liens du livre sont morts ou redirigés": True}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "une fiche modèle décrit un modèle, pas un dataset",
                                           "C": "le README d'un dépôt de code ne dit pas la licence ni les limites des données"})),
    ]),
    Paper("29.Q8", "Les « nouveaux » datasets de 2018 : tâches, tailles et retraits", [
        ("a", WHOLE, "9", r'''mistakes={"Open Images en compte environ neuf millions, pas un": 1}'''),
        ("b", LETTERS, '"AC"', r'''mistakes={"COCO est toujours disponible": "ABC", "Fashion-MNIST est toujours disponible": "ACD",
          "il manque 80 Million Tiny Images": "A", "il manque MS-Celeb-1M": "C"}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "LAION-6B n'existe pas : relis l'encadré 🕰️ du §29.5",
                                           "C": "il a été republié, nettoyé, en 2024"})),
        ("d", LETTER, '"B"', letters("B", {"A": "la licence de CelebA exclut l'usage commercial"})),
        ("e", TF, "True", r'''mistakes={"une copie ne lève ni le retrait ni ses raisons : l'utiliser reste un problème": False}'''),
    ]),
    Paper("29.R1", "Ch. 28 : qui possède les droits d'une image de style ?", [
        ("a", WHOLE, "2044", r'''mistakes={"70 ans après 1973, puis le 1ᵉʳ janvier suivant : 2044, pas 2043": 2043}'''),
        ("b", TF, "False", r'''mistakes={"la directive de 2019 (art. 14) : une reproduction fidèle d'une œuvre du domaine public n'est pas protégée": True}'''),
        ("c", TF, "False", r'''mistakes={"le droit d'auteur protège des œuvres, pas un style": True}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "les photos de scikit-learn sont sous CC BY 2.0 : il faut créditer leurs auteurs"})),
        ("e", WHOLE, "1961", r'''mistakes={"1890 + 70 = 1960, puis le 1ᵉʳ janvier suivant": 1960}'''),
    ]),
    Paper("29.R2", "Ch. 26 : en RL, d'où viennent les données ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "en RL, aucun dataset annoté : l'agent produit ses données en agissant"})),
        ("b", TF, "True", r'''mistakes={"une autre politique visite d'autres états : les données changent avec elle": False}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "le rejeu stocke des transitions vécues, pas des images annotées",
                                           "C": "les poids sont dans le réseau, pas dans la mémoire de rejeu"})),
        ("d", WHOLE, "16", r'''mistakes={"4 × 4 cases, c'est 16 états ; 4, c'est le nombre d'actions": 4}'''),
        ("e", TF, "False", r'''mistakes={"les données d'un agent changent avec sa politique : un découpage figé n'a pas de sens": True}'''),
    ]),
    Paper("29.R3", "Ch. 2 : biais d'échantillonnage : un dataset représente-t-il la population visée ?", [
        ("a", WHOLE, "344", r'''mistakes={"333, c'est après avoir retiré les lignes incomplètes : compte-les toutes": 333}'''),
        ("b", WHOLE, "3", r'''mistakes={"trois espèces (Adélie, Chinstrap, Gentoo) ; trois îles aussi": 2}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "une erreur de mesure fausse chaque valeur, pas le choix des exemples",
                                           "C": "un modèle trop simple, c'est de l'underfitting"})),
        ("d", TF, "False", r'''mistakes={"trois espèces, trois îles, trois saisons : ce n'est pas tous les manchots du monde": True}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "l'overfitting concerne l'écart entre entraînement et test d'une même distribution",
                                           "C": "une fuite fait passer de l'information du test à l'entraînement"})),
    ]),
    Paper("29.1", "Poids en mémoire de MNIST, Fashion-MNIST et CIFAR-10", [
        ("a", DEC2, "MNIST_VALUES / MB", r'''decimals=2, mistakes={"compte les 70 000 images, pas seulement les 60 000 d'entraînement": 60_000 * 784 / MB,
          "1 Mo = 10**6 octets ici, pas 2**20": MNIST_VALUES / 2 ** 20}'''),
        ("b", DEC2, "4 * MNIST_VALUES / MB", r'''decimals=2, mistakes={"un float32 occupe 4 octets, pas 8": 8 * MNIST_VALUES / MB}'''),
        ("c", DEC2, "CIFAR_VALUES / MB", r'''decimals=2, mistakes={"CIFAR-10 a trois canaux de couleur": CIFAR_VALUES / 3 / MB}'''),
        ("d", DEC3, "4 * CIFAR_VALUES / GB", r'''decimals=3, mistakes={"en float32, multiplie par 4 octets": CIFAR_VALUES / GB}'''),
        ("e", WHOLE, "int(GB // (784 * 4))", r'''mistakes={"en float32, chaque valeur occupe 4 octets": int(GB // 784),
          "arrondis vers le bas : une image incomplète ne tient pas": int(GB // (784 * 4)) + 1}'''),
        ("f", TF, "True", r'''mistakes={"même nombre d'images, même taille, même type : même mémoire": False}'''),
    ]),
    Paper("29.2", "Découpage stratifié et référence « classe majoritaire »", [
        ("a", WHOLE, "round(0.70 * COUNTS_29_2['C'])", r'''mistakes={"70 % des 200 exemples de C, pas 70 % du total": 1400}'''),
        ("b", WHOLE, "round(0.15 * COUNTS_29_2['C'])", r'''mistakes={"15 % des 200 exemples de C, pas 15 % du total": 300}'''),
        ("c", DEC2, "COUNTS_29_2['A'] / N_29_2", r'''decimals=2, mistakes={"la classe majoritaire est A : 1 200 sur 2 000": 0.30}'''),
        ("d", DEC2, "(COUNTS_29_2['A'] + COUNTS_29_2['B']) / N_29_2", r'''decimals=2, mistakes={"c'est l'accuracy du modèle de c) : un modèle peut aussi avoir juste sur tous les B": 0.60,
          "il rate au moins tous les C, soit 10 % du total": 1.0}'''),
        ("e", DEC2, "0.0", r'''decimals=2, mistakes={"le modèle ne prédit jamais C : il n'en trouve aucun": 1.0}'''),
        ("f", TF, "True", r'''mistakes={"stratifier, c'est garder les proportions des classes dans chaque partie": False}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 29.1 et 29.2)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"B\"`) ; "
               "plusieurs lettres : entre guillemets (`\"ABD\"`). Arrondis comme l'énoncé le demande, et seulement "
               "à la fin du calcul. Les réponses pas encore remplies affichent ⏳. Les questions « dans ta copie », "
               "l'estimation 🧮 29.3, la réflexion (🗣️ 29.4, ⚖️ 29.5, 📄 29.6) et l'entretien se corrigent avec "
               "`05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A and B (exercises 29.7 to 29.16)
# ---------------------------------------------------------------------------
from ch29_parts import PARTS, ROCKETS  # noqa: E402
import ch29_code  # noqa: E402


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 29.Q1 à 29.Q8, 29.R1 à 29.R3, 29.1, 29.2 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rocket = " 🚀" if ex.id in ROCKETS else ""
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title}{rocket} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if kind == "exercise":
        title = "# 29 · Datasets et préparation du projet final — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » se notent dans la section « Notes sur le notebook » de ta copie "
               "de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`. Tout tourne sur CPU en quelques "
               "minutes. 29.10 et 29.11 interrogent le Hugging Face Hub et OpenML : sans réseau, leurs vérifications "
               "passent quand même (copie locale, services simulés) et les parties en direct sont sautées.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch29_datasets/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 29`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 29 · Datasets et préparation du projet final — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Charger des datasets avec scikit-learn, torchvision et Hugging Face `datasets` (à une révision "
               "fixée), et chercher sur le Hub et OpenML en lisant la licence.\n"
               "- Profiler un tableau inconnu, détecter les doublons qui fuient entre entraînement et test, et "
               "découper par groupes.\n"
               "- Mesurer un décalage de distribution et ce que récupèrent une meilleure préparation et quelques "
               "exemples étiquetés.\n"
               "- Écrire un téléchargement vérifié par SHA-256 et la data card du dataset de ton projet final.\n\n"
               "**Rappel express.** Mesurer sur les données chargées, pas croire la description. Un dataset public "
               "n'est pas forcément libre : lire sa licence. Découper **avant** toute transformation ajustée sur les "
               "données, et **par groupes** quand des exemples sont liés. Mémoire d'un dataset d'images : "
               "$N \\times H \\times W \\times C \\times$ (octets par valeur).")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu charger un même dataset par deux API et vérifier qu'elles rendent les mêmes données ?\n"
               "2. Sais-tu dire, en lisant sa page, si un dataset peut servir à ton projet (licence, provenance, "
               "retraits) ?\n"
               "3. Sais-tu repérer une fuite entre entraînement et test, et découper par groupes ?\n\n"
               "**Avant le checkpoint** : choisis le dataset de ton projet final, avec sa data card et un "
               "téléchargement vérifié (🛠️ 29.15).\n\n"
               "**Pour aller plus loin** : Gebru et coll., « Datasheets for Datasets », et l'article de Northcutt et "
               "coll. sur les erreurs de labels des benchmarks, cités dans la fiche. La suite : le checkpoint de la "
               "partie VI, puis le projet final.")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT, PAPER_INTRO)
    cells += [md("## Outils du notebook\n\nLa cellule suivante importe pandas et PyTorch et définit les petits outils "
                 "des deux parties : `train_mlp` et `evaluate` (le petit réseau de 29.8, 29.9 et 29.14), `fine_tune`, "
                 "`digits_naive`, `show_digits`, `online`, un Hub et un OpenML simulés pour 29.11 (`FakeHubApi`, "
                 "`fake_fetch_openml`) et `check_card` pour la data card de 29.15. Lis leurs docstrings au besoin."),
              code(ch29_code.TOOLS)]
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
