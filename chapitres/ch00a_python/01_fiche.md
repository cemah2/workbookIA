# 0A · Python, notebooks et outils — fiche de cours

> Ce chapitre n'a pas d'équivalent dans le livre de Glassner : **cette fiche est le cours**. Elle t'apprend tout le Python dont le workbook a besoin, des variables jusqu'à git et pytest.

| | |
|---|---|
| **Sections** | 100.1 à 100.11 (numérotation interne du workbook : 100 = chapitre 0A) |
| **Temps total estimé** | ≈ 28 h : lecture et essais de la fiche ≈ 5 h, exercices ≈ 22 h, flashcards ≈ 1 h |
| **Prérequis** | aucun, à part le calcul du collège (rappels 🧮 dans la fiche) |
| **Fichiers du chapitre** | `02_exercices.md` (quiz, papier, entretien) · `03_notebook.ipynb` · `04_indices.md` · `05_solutions.md` et `05_solutions.ipynb` · `06_mes_reponses.md` · `flashcards.csv` |
| **mylearn** | `_example.py` (`mean`, 0A.26) et `utils.py` (`count_values`, `argmax`, `one_hot`, `iterate_minibatches`, 0A.63 à 0A.66) |

## Comment utiliser ce chapitre

Tu débutes en Python : c'est normal que ce chapitre soit long. **Tu n'as pas à tout retenir.** Tu dois savoir que chaque outil existe, et où le retrouver : cette fiche, les cheatsheets de `annexes/cheatsheets/` et la documentation officielle.

**Ordre conseillé.**
1. Lance `python tools/start_chapter.py 0A` (sur Colab : `00_setup/COLAB.md` §2) et ouvre ta copie du notebook, `mon_travail/ch00a_python/03_notebook.ipynb` ; puis lis la section 100.1 et fais l'exercice 0A.13 du notebook et les quiz 0A.Q1 et 0A.Q2.
2. Ensuite, **section par section** : lis la section, **tape toi-même** les exemples dans une cellule de notebook, puis fais les exercices de la section (tableau ci-dessous). Commence par les quiz 🧠, puis le papier ✏️, puis le notebook.
3. Garde les questions d'entretien 💼 et les exercices git 🛠️ pour la fin de chaque partie.

**Lire les exemples.** Dans cette fiche, une ligne qui commence par `>>>` est ce que tu tapes ; les lignes juste en dessous, sans `>>>`, sont ce que Python affiche. Une ligne `...` continue l'instruction précédente (un bloc indenté). Dans un notebook, tape le code **sans** `>>>` ni `...`. Tous les exemples ont été exécutés avec les versions du workbook (Python 3.13, NumPy 2.1, pandas 2.2).

| Section | Quiz 🧠 | Papier ✏️ et réflexion | Notebook | Entretien 💼 |
|---|---|---|---|---|
| 100.1 Prise en main : notebooks, terminal et workbook | Q1, Q2 | 0A.10 | 0A.13, 0A.26 | E1, E4 |
| 100.2 Valeurs, variables et types | Q3 | 0A.1, 0A.2 | 0A.14, 0A.15, 0A.37 | |
| 100.3 Structures de données | Q4 | 0A.2, 0A.9 | 0A.16 à 0A.19, 0A.24, 0A.63 | E2 |
| 100.4 Contrôle du flux | Q5 | 0A.3 | 0A.20 à 0A.22 | |
| 100.5 Fonctions | Q6, Q7 | 0A.6 | 0A.23, 0A.37, 0A.40 à 0A.44, 0A.61 | |
| 100.6 Modules, erreurs et fichiers | Q8 | | 0A.24, 0A.25, 0A.37 à 0A.39, 0A.45, 0A.46 | |
| 100.7 Classes et objets | Q9 | | 0A.47 à 0A.50 | |
| 100.8 NumPy | Q10, Q11 | 0A.5, 0A.7, 0A.8, 0A.9 | 0A.27 à 0A.31, 0A.51 à 0A.56, 0A.64 à 0A.67 | E3, E4 |
| 100.9 pandas | Q12 | 0A.4 | 0A.32 à 0A.35, 0A.57, 0A.58, 0A.60, 0A.67 | |
| 100.10 matplotlib | Q12 | | 0A.36, 0A.59, 0A.60, 0A.67 | |
| 100.11 Outils du développeur | Q2, Q12 | 0A.10 à 0A.12 | 0A.26, 0A.61 à 0A.66 | E5 |

## Objectifs d'apprentissage

À la fin du chapitre, tu sais :
- **utiliser** un notebook (Colab ou Jupyter) et le terminal sans te perdre : ordre d'exécution, « Tout exécuter », `wb.check`, pytest, git ;
- **écrire** des programmes Python lisibles avec les types de base, les structures de données, les boucles, les compréhensions et les fonctions (lambda, `*args`/`**kwargs` et fermetures compris) ;
- **lire et écrire** des classes (attributs, méthodes, méthodes spéciales, héritage, générateurs) comme celles de scikit-learn et PyTorch ;
- **manipuler** des tableaux NumPy : formes, indexation, masques, broadcasting, réductions par axe, `reshape`, aléatoire reproductible ;
- **explorer** un dataset (*jeu de données*) avec pandas (lecture, sélection, valeurs manquantes, `groupby`) et le **visualiser** avec matplotlib ;
- **lire, écrire et sérialiser** des fichiers (`pathlib`, `json`, `pickle`) et **extraire** de l'information avec des expressions régulières ;
- **documenter, tester et versionner** ton code : docstrings au format NumPy, doctest, pytest, commits git propres ;
- **implémenter** les premières fonctions de mylearn (`count_values`, `argmax`, `one_hot`, `iterate_minibatches`) et les valider par des tests à oracle.

## L'essentiel en 10 lignes

1. Un notebook garde ses variables en mémoire dans un **noyau** : c'est l'**ordre d'exécution** qui compte, pas l'ordre des cellules. « Redémarrer et tout exécuter » est le test de vérité.
2. Python manipule des **valeurs typées** (`int`, `float`, `str`, `bool`, `None`) ; une erreur de type se lit dans la dernière ligne du message.
3. Quatre structures couvrent presque tout : **liste** (ordonnée, modifiable), **tuple** (figé), **dictionnaire** (clé → valeur), **ensemble** (sans doublon).
4. Une **fonction** a des paramètres, renvoie une valeur avec `return` et se documente par une docstring ; on la teste dès qu'on l'a écrite.
5. Les **classes** regroupent des données (attributs) et des actions (méthodes) : `fit`/`predict` de scikit-learn et `forward` de PyTorch sont des méthodes.
6. **NumPy** calcule sur des tableaux entiers d'un coup (*vectorisation*) : plus court, plus lisible et de quelques dizaines à quelques centaines de fois plus rapide qu'une boucle, selon l'opération.
7. La **forme** (`shape`) d'un tableau est la première chose à vérifier ; `axis=0` travaille « le long des lignes » (une valeur par colonne).
8. **pandas** manipule des tableaux de données nommées (DataFrame) : lire, filtrer, regrouper, puis passer à NumPy avec `X` de forme `(n_samples, n_features)`.
9. Une graine (`np.random.default_rng(42)`) rend les expériences **reproductibles**.
10. Un pro **versionne** (git), **documente** (docstrings) et **teste** (pytest) : ce sont aussi tes premiers pas dans mylearn.

## 100.1 · Prise en main : notebooks, terminal et workbook

### 100.1.1 · Cellules, noyau, ordre d'exécution et « Tout exécuter »

Un **notebook** (*carnet*), Jupyter en local ou Colab dans le navigateur, est un document fait de **cellules** :
- une **cellule de code** contient du Python ; on l'exécute avec **Maj + Entrée** (exécuter et passer à la suivante) ou **Ctrl + Entrée** (exécuter et rester) ;
- une **cellule Markdown** contient du texte mis en forme (titres `#`, listes `-`, **gras**, formules `$x^2$`) ; l'exécuter l'affiche joliment.

Le code n'est pas exécuté par la page elle-même, mais par un programme Python qui tourne en arrière-plan : le **noyau** (*kernel*). Le noyau garde en mémoire toutes les variables créées depuis son démarrage. Le numéro affiché à gauche d'une cellule, `[5]`, indique qu'elle a été la 5ᵉ exécutée.

Conséquence essentielle : **l'état du noyau dépend de l'ordre dans lequel tu as exécuté les cellules**, pas de leur ordre sur la page. Si tu exécutes deux fois une cellule `total = total + 10`, `total` augmente deux fois. Si tu supprimes une cellule, la variable qu'elle avait créée existe encore jusqu'au prochain redémarrage.

La dernière expression d'une cellule est affichée automatiquement ; pour afficher autre chose, utilise `print(...)`.

| Commande (Colab / Jupyter) | Effet |
|---|---|
| Exécution → Tout exécuter (*Run all*) | exécute toutes les cellules de haut en bas, **en gardant** l'état actuel du noyau |
| Exécution → Redémarrer la session et tout exécuter (*Restart and run all*) | vide la mémoire, puis exécute tout de haut en bas : **le test de reproductibilité** |
| Exécution → Interrompre | arrête une cellule trop longue (boucle infinie) |

> ⚠️ **Piège classique — un notebook qui ne marche que chez toi** — Un notebook qui « marche » après une heure de travail peut planter chez quelqu'un d'autre, parce qu'une variable venait d'une cellule effacée ou exécutée dans le désordre. Avant de partager ou de rendre un notebook : *Restart and run all*.

> 💼 **En entreprise** — « Ton notebook est-il reproductible ? » est une question d'entretien classique (0A.E4). La réponse tient en trois points : il tourne de haut en bas après redémarrage, les graines aléatoires sont fixées, les versions des bibliothèques sont notées.

### 100.1.2 · Le workbook : setup, `wb.check`, `wb.attempt`, `start_chapter.py`, `mon_travail`

**Où tu écris.** Le dépôt contient deux mondes :
- `chapitres/`, `solutions/`, `src/`… sont mis à jour par Claude : tu les **lis**, tu ne les modifies pas (sinon `git pull` se plaindra) ;
- `mon_travail/` est **ton espace** : copies de travail des notebooks, ta librairie `mylearn`, tes fichiers de suivi. Claude n'y écrit jamais.

`python tools/start_chapter.py 0A` copie dans `mon_travail/` le notebook du chapitre, le fichier `06_mes_reponses.md` et les squelettes `mylearn` du chapitre, **sans jamais écraser** un fichier existant.

**La cellule de setup.** Chaque notebook commence par la même cellule. Elle retrouve le dépôt (et, sur Colab, monte Google Drive et récupère la dernière version), puis appelle `wb.setup()`, qui fixe les graines aléatoires, choisit le processeur (`cpu` ou `cuda`) et affiche un rapport. `FAST_MODE = True` garde des calculs courts.

**Vérifier une réponse sans voir la solution.** `wb.check("id", valeur)` compare une empreinte (*hash*) de ta valeur à celle de la solution :

```python
import wb

wb.check("demo.1", 344)
wb.check("demo.1", 345)
wb.check("demo.1", ...)
```

```text
✅ Ex demo.1 : Exact !
❌ Ex demo.1 : Tu es à 1 près : une petite erreur de calcul, ou, si tu comptes des éléments, une erreur de bornes (off-by-one : bornes incluses ou exclues ?).
⏳ Ex demo.1 : pas encore fait (remplace `...` par ta réponse).
```

- ✅ juste · ❌ faux, avec un indice qui ne donne pas la réponse · ⏳ pas encore fait (la valeur est encore `...`).
- Les arrondis, les majuscules et les accents sont gérés : l'énoncé dit toujours combien de décimales garder.

**Écrire du code qui n'existe pas encore.** Dans les notebooks, les fonctions à écrire contiennent `raise NotImplementedError`. Le bloc `with wb.attempt("id"):` transforme cette erreur en « ⏳ pas encore fait » : c'est ce qui permet de faire « Tout exécuter » même quand rien n'est rempli.

```python
def count_islands(df):
    """Return the number of distinct islands in the dataset."""
    raise NotImplementedError("count_islands() is not written yet")


with wb.attempt("demo.4"):
    wb.check("demo.4", count_islands(None))
```

```text
⏳ Ex demo.4 : pas encore fait (count_islands() is not written yet).
```

**mylearn.** Au fil des chapitres, tu écris ta propre librairie de machine learning dans `mon_travail/mylearn/`. Chaque fonction arrive sous forme de **squelette** (*stub*) : la signature et la documentation sont écrites, le corps est à compléter. Les tests (`python -m pytest tests/`) comparent ton code à une bibliothèque de confiance (§100.11.5).

**Suivi.** `mon_travail/suivi/tableau_de_bord.md` (une case par exercice), `journal.md` (ce que tu as fait et compris à chaque séance) et `auto_evaluation.md` (tes compétences de 0 à 3).

**Les métiers visés.** Le workbook prépare à quatre métiers : *data analyst* (répondre à des questions métier avec des données), *data scientist* (construire et évaluer des modèles), *ML engineer* (mettre les modèles en production) et *AI engineer* (construire des applications autour de modèles pré-entraînés). Le détail, avec les compétences de chaque chapitre, est dans `annexes/metiers.md` ; la question 💼 0A.E1 t'entraîne à en parler.

### 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest et git

Le **terminal** (Terminal sur Mac, PowerShell sur Windows, un terminal sous Linux) est une fenêtre où tu tapes des commandes. Il travaille toujours **dans un dossier**, le *dossier courant*, affiché avant l'invite.

| Action | Mac / Linux | Windows (PowerShell) |
|---|---|---|
| Où suis-je ? | `pwd` | `pwd` |
| Lister le dossier | `ls` | `ls` (ou `dir`) |
| Entrer dans un dossier | `cd workbookIA` | `cd workbookIA` |
| Remonter d'un niveau | `cd ..` | `cd ..` |
| Aller dans ton dossier personnel | `cd ~` | `cd ~` |
| Lancer un script | `python3 tools/start_chapter.py 0A` | `python tools/start_chapter.py 0A` |
| Lancer les tests d'un chapitre | `python3 -m pytest tests/test_ch00a_utils.py -q` | `python -m pytest tests/test_ch00a_utils.py -q` |
| État du dépôt git | `git status` | `git status` |

Un **chemin** désigne un fichier :
- **relatif** : à partir du dossier courant, par exemple `data/penguins.csv` (ou `data\penguins.csv` sous Windows) ;
- **absolu** : à partir de la racine du disque, par exemple `/home/lea/workbookIA/data/penguins.csv` ou `C:\Users\Lea\workbookIA\data\penguins.csv`.

`..` désigne le dossier parent et `.` le dossier courant. La plupart des commandes du workbook se lancent **depuis la racine du dépôt** (le dossier qui contient `README.md`).

> ⚠️ **Piège classique — « No such file or directory »** — `No such file or directory` veut presque toujours dire « tu n'es pas dans le bon dossier » : fais `pwd`, puis `cd` vers la racine du dépôt.

Colab propose aussi un terminal, gratuit pour tous depuis 2025, mais le workbook lance ses commandes depuis une cellule, avec le module `subprocess`, pour qu'elles restent dans le notebook (les notebooks du workbook n'utilisent pas les raccourcis `!` et `%`, qui ne fonctionnent pas partout ; sur Colab, une cellule qui commence par `!` lance une commande du terminal, comme dans `00_setup/COLAB.md`).

```python
import subprocess
import sys

result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
print(result.stdout)
```

## 100.2 · Valeurs, variables et types

### 100.2.1 · Nombres, opérateurs arithmétiques, booléens et `None`

Une **variable** est un nom qui désigne une valeur. On la crée avec `=` (l'affectation) : à droite la valeur, à gauche le nom.

```python
>>> flipper = 181
>>> mass = 3750.0
>>> flipper
181
>>> mass / 1000
3.75
```

Python connaît deux sortes de nombres : les entiers (`int`, sans limite de taille) et les nombres à virgule (`float`, avec un **point** décimal). `1e-3` s'écrit aussi `0.001` ; `1_000_000` est un entier lisible.

| Opérateur | Sens | Exemple | Résultat |
|---|---|---|---|
| `+ - *` | addition, soustraction, multiplication | `7 * 2` | `14` |
| `/` | division (donne toujours un `float`) | `7 / 2` | `3.5` |
| `//` | quotient de la division euclidienne | `7 // 2` | `3` |
| `%` | reste (modulo) | `7 % 2` | `1` |
| `**` | puissance | `2 ** 10` | `1024` |

Les priorités sont celles des maths : `**` d'abord, puis `* / // %`, puis `+ -`. Dans le doute, mets des parenthèses. Attention au signe moins : `-2 ** 2` vaut `-4`, car la puissance passe avant le signe ; écris `(-2) ** 2` pour obtenir 4.

```python
>>> 2 + 3 * 4
14
>>> (2 + 3) * 4
20
>>> -2 ** 2
-4
>>> 344 // 32, 344 % 32
(10, 24)
```

> 🧮 **Rappel maths — division euclidienne** — Diviser $a$ par $b$ donne un quotient $q$ et un reste $r$ tels que $a = b \times q + r$, avec $0 \le r < b$. En Python : $a = b \times (a \,//\, b) + (a \,\%\, b)$. Pour 344 manchots rangés par batches de 32 : $344 = 32 \times 10 + 24$, soit 10 batches pleins et un dernier batch de 24. L'**arrondi supérieur** $\lceil 344 / 32 \rceil = 11$ donne le nombre de batches en comptant le batch incomplet. Avec un nombre négatif, `//` arrondit **vers le bas** (vers $-\infty$) : `-7 // 2` vaut `-4`, et le reste reste positif (`-7 % 2` vaut `1`).

```python
>>> -7 // 2, -7 % 2
(-4, 1)
>>> import math
>>> math.ceil(344 / 32)
11
```

Les **booléens** `True` et `False` viennent des comparaisons (`==` égal, `!=` différent, `<`, `<=`, `>`, `>=`) et se combinent avec `and`, `or`, `not`. `None` veut dire « pas de valeur » : c'est ce que renvoie une fonction sans `return`.

```python
>>> mass > 4000
False
>>> 3000 <= mass < 4000
True
>>> not (mass > 4000) and flipper > 180
True
>>> x = None
>>> x is None
True
```

> 🧮 **Rappel maths — les nombres à virgule sont approchés** — L'ordinateur stocke les `float` en base 2, où 0,1 n'a pas d'écriture finie (comme 1/3 en base 10). D'où des écarts minuscules : ne compare jamais deux `float` avec `==`, utilise `math.isclose` (ou `np.isclose`). La précision des calculs est détaillée au ch. 5.

```python
>>> 0.1 + 0.2
0.30000000000000004
>>> 0.1 + 0.2 == 0.3
False
>>> math.isclose(0.1 + 0.2, 0.3)
True
```

### 100.2.2 · Chaînes de caractères et f-strings

Une **chaîne** (`str`) est un texte entre guillemets simples ou doubles. `len` donne sa longueur ; les **méthodes** (des fonctions attachées à la valeur, appelées avec un point) la transforment sans la modifier : une chaîne est **immuable**.

```python
>>> species = "  Adelie Penguin (Pygoscelis adeliae) "
>>> species.strip()
'Adelie Penguin (Pygoscelis adeliae)'
>>> species.strip().lower()
'adelie penguin (pygoscelis adeliae)'
>>> species.split("(")
['  Adelie Penguin ', 'Pygoscelis adeliae) ']
>>> species.replace("Penguin", "").split()
['Adelie', '(Pygoscelis', 'adeliae)']
>>> "Adelie" in species
True
>>> len("Gentoo")
6
>>> "ab" + "cd", "-" * 10
('abcd', '----------')
```

| Méthode | Effet |
|---|---|
| `s.strip()` | retire les espaces (et retours à la ligne) au début et à la fin |
| `s.lower()`, `s.upper()`, `s.title()` | minuscules, majuscules, majuscule à chaque mot |
| `s.split(sep)` | découpe en liste de morceaux (sur les espaces si `sep` est absent) |
| `sep.join(liste)` | recolle une liste de chaînes avec `sep` entre elles |
| `s.replace(a, b)` | remplace toutes les occurrences de `a` par `b` |
| `s.startswith(a)`, `s.endswith(a)` | teste le début ou la fin |
| `s.find(a)` | position de `a` (`-1` si absent) |

Une **f-string** (chaîne précédée de `f`) insère des valeurs entre accolades. Après `:`, un **format** règle l'affichage des nombres.

```python
>>> name, mass, flipper = "Adelie", 3750.0, 181
>>> f"{name}: {mass} g, flipper {flipper} mm"
'Adelie: 3750.0 g, flipper 181 mm'
>>> f"{name} weighs {mass / 1000:.1f} kg"
'Adelie weighs 3.8 kg'
>>> f"{mass:,.0f} g | {0.4567:.1%} | {flipper:>6} | {flipper:06d}"
'3,750 g | 45.7% |    181 | 000181'
>>> f"{mass=}"
'mass=3750.0'
```

| Format | Effet | Exemple |
|---|---|---|
| `:.2f` | 2 décimales | `f"{3.14159:.2f}"` → `3.14` |
| `:.1%` | pourcentage à 1 décimale | `f"{0.4567:.1%}"` → `45.7%` |
| `:,` | séparateur de milliers | `f"{1234567:,}"` → `1,234,567` |
| `:>6` / `:<6` | aligné à droite / à gauche sur 6 caractères | |
| `:.2e` | notation scientifique | `f"{0.000123:.2e}"` → `1.23e-04` |

`"\n"` est un retour à la ligne et `"\t"` une tabulation ; une chaîne entre triples guillemets `"""…"""` peut s'étendre sur plusieurs lignes (c'est ce qu'on utilise pour les docstrings).

### 100.2.3 · Types, conversions et erreurs de type

`type(x)` donne le type d'une valeur ; `isinstance(x, int)` teste un type. On **convertit** avec le nom du type :

```python
>>> type(181), type(181.0), type("181"), type(True), type(None)
(<class 'int'>, <class 'float'>, <class 'str'>, <class 'bool'>, <class 'NoneType'>)
>>> int("181") + 1
182
>>> float("3.5"), str(3.5), int(3.9), round(3.9)
(3.5, '3.5', 3, 4)
>>> round(2.5), round(3.5), round(2.675, 2)
(2, 4, 2.67)
>>> bool(0), bool(""), bool("0"), bool([])
(False, False, True, False)
```

Trois remarques :
- `int(3.9)` **tronque** (vers zéro), il n'arrondit pas ;
- `round` arrondit « au pair le plus proche » quand on tombe pile au milieu (`round(2.5)` vaut 2) ; et `round(2.675, 2)` donne 2.67 parce que 2.675 est stocké comme 2.67499999… ;
- les valeurs « vides » (`0`, `""`, `[]`, `None`) valent `False` dans un test ; une chaîne non vide, même `"0"`, vaut `True`.

Additionner un texte et un nombre n'a pas de sens : Python lève une **erreur de type** (`TypeError`). Convertir un texte qui ne ressemble pas à un nombre lève une `ValueError`.

```python
>>> "Flipper: " + 181
Traceback (most recent call last):
  ...
TypeError: can only concatenate str (not "int") to str
>>> "Flipper: " + str(181)
'Flipper: 181'
>>> int("3.5")
Traceback (most recent call last):
  ...
ValueError: invalid literal for int() with base 10: '3.5'
```

> ⚠️ **Piège classique — un nombre lu est une chaîne** — Lire un nombre dans un fichier ou dans `input()` donne une **chaîne** : `"181" + "1"` vaut `"1811"`. Convertis avec `int(...)` ou `float(...)` avant de calculer.

## 100.3 · Structures de données

Quatre structures (plus `Counter`) couvrent presque tous les besoins. Le tableau résume ; les sous-sections détaillent.

| Structure | Écriture | Ordonnée | Modifiable (*mutable*) | Doublons | Usage type |
|---|---|---|---|---|---|
| liste `list` | `[181, 186, 195]` | oui | oui | oui | une suite de valeurs qu'on complète |
| tuple `tuple` | `(181, 3750.0)` | oui | **non** | oui | un enregistrement figé, plusieurs valeurs renvoyées |
| dictionnaire `dict` | `{"Adelie": 152}` | ordre d'insertion | oui | clés uniques | associer une valeur à une clé |
| ensemble `set` | `{"Biscoe", "Dream"}` | non | oui | **non** | éliminer les doublons, tester l'appartenance |
| `Counter` | `Counter(liste)` | ordre de 1ʳᵉ apparition | oui | clés uniques | compter des occurrences |

### 100.3.1 · Listes : indices, tranches, méthodes, mutabilité

Une **liste** range des valeurs dans l'ordre. Les **indices** commencent à **0** ; un indice négatif compte depuis la fin (`-1` est le dernier élément).

```python
>>> flippers = [181, 186, 195, 193, 190]
>>> flippers[0], flippers[-1], len(flippers)
(181, 190, 5)
>>> flippers[1:3]
[186, 195]
>>> flippers[:2], flippers[2:], flippers[::2], flippers[::-1]
([181, 186], [195, 193, 190], [181, 195, 190], [190, 193, 195, 186, 181])
```

Une **tranche** (*slice*) `liste[début:fin:pas]` prend les éléments de `début` **inclus** à `fin` **exclue**. `début` absent = depuis le début ; `fin` absente = jusqu'au bout ; `pas` = 2 prend un élément sur deux, `-1` parcourt à l'envers. Le nombre d'éléments de `liste[a:b]` est `b - a` : c'est la raison du « fin exclue ». Les mêmes règles valent pour les chaînes et les tuples.

```python
>>> "Pygoscelis"[0:4], "Pygoscelis"[-3:]
('Pygo', 'lis')
>>> flippers.append(181)
>>> flippers
[181, 186, 195, 193, 190, 181]
>>> flippers.sort()
>>> flippers
[181, 181, 186, 190, 193, 195]
>>> sorted(flippers, reverse=True)
[195, 193, 190, 186, 181, 181]
>>> 195 in flippers, flippers.index(195), flippers.count(181)
(True, 5, 2)
>>> flippers.pop(), flippers
(195, [181, 181, 186, 190, 193])
```

| Méthode ou fonction | Effet |
|---|---|
| `l.append(x)` | ajoute `x` à la fin |
| `l.extend(autre)` | ajoute tous les éléments d'`autre` |
| `l.insert(i, x)` | insère `x` à la position `i` |
| `l.pop()`, `l.pop(i)` | retire et renvoie le dernier élément (ou celui d'indice `i`) |
| `l.remove(x)` | retire la première occurrence de `x` |
| `l.sort()` | trie **la liste elle-même** et renvoie `None` |
| `sorted(l)` | renvoie une **nouvelle** liste triée, `l` ne change pas |
| `sum(l)`, `min(l)`, `max(l)`, `len(l)` | somme, minimum, maximum, longueur |

**Mutabilité et alias.** Une liste est *mutable* : on peut la modifier sur place. Attention : `b = a` ne copie pas la liste, il donne un **deuxième nom** au même objet. Pour une vraie copie : `a.copy()` ou `list(a)`.

```python
>>> a = [1, 2, 3]
>>> b = a
>>> b.append(4)
>>> a
[1, 2, 3, 4]
>>> c = a.copy()
>>> c.append(5)
>>> a, c
([1, 2, 3, 4], [1, 2, 3, 4, 5])
```

> ⚠️ **Piège classique — `l = l.sort()`** — `l = l.sort()` met `None` dans `l` : `sort` modifie la liste et ne renvoie rien. Écris `l.sort()` tout court, ou `l = sorted(l)`.

### 100.3.2 · Tuples et déballage

Un **tuple** ressemble à une liste mais ne peut plus changer une fois créé (*immuable*). On l'utilise pour un enregistrement figé (un point `(x, y)`, une forme de tableau `(344, 8)`) ou pour renvoyer plusieurs valeurs. Un tuple d'un seul élément s'écrit avec une virgule : `(181,)`.

Le **déballage** (*unpacking*) affecte chaque élément à une variable :

```python
>>> penguin = ("Adelie", "Torgersen", 181)
>>> species, island, flipper = penguin
>>> island
'Torgersen'
>>> first, *rest = [181, 186, 195]
>>> first, rest
(181, [186, 195])
>>> a, b = 1, 2
>>> a, b = b, a
>>> a, b
(2, 1)
>>> penguin[0] = "Gentoo"
Traceback (most recent call last):
  ...
TypeError: 'tuple' object does not support item assignment
```

La ligne `a, b = b, a` échange deux variables sans variable temporaire : Python construit d'abord le tuple de droite, puis le déballe.

### 100.3.3 · Dictionnaires

Un **dictionnaire** associe des **clés** (souvent des chaînes) à des **valeurs**. On y accède par la clé, pas par une position. Les clés sont uniques et doivent être immuables (chaîne, nombre, tuple).

```python
>>> penguin = {"species": "Adelie", "island": "Torgersen", "flipper_length_mm": 181}
>>> penguin["island"]
'Torgersen'
>>> penguin["body_mass_g"] = 3750
>>> penguin
{'species': 'Adelie', 'island': 'Torgersen', 'flipper_length_mm': 181, 'body_mass_g': 3750}
>>> penguin["sex"]
Traceback (most recent call last):
  ...
KeyError: 'sex'
>>> penguin.get("sex"), penguin.get("sex", "unknown")
(None, 'unknown')
>>> "species" in penguin, len(penguin)
(True, 4)
>>> list(penguin.keys())
['species', 'island', 'flipper_length_mm', 'body_mass_g']
>>> for key, value in penguin.items():
...     print(f"{key:>18}: {value}")
           species: Adelie
            island: Torgersen
 flipper_length_mm: 181
       body_mass_g: 3750
```

| Méthode | Effet |
|---|---|
| `d[k]` | valeur de la clé `k` (`KeyError` si elle n'existe pas) |
| `d.get(k, défaut)` | valeur de `k`, ou `défaut` (par défaut `None`) si elle n'existe pas |
| `d[k] = v` | crée ou remplace la clé `k` |
| `k in d` | la clé `k` existe-t-elle ? (recherche instantanée, même dans un très grand dictionnaire) |
| `d.keys()`, `d.values()`, `d.items()` | les clés, les valeurs, les paires `(clé, valeur)` |
| `d.pop(k)` | retire la clé `k` et renvoie sa valeur |
| `d.setdefault(k, [])` | renvoie la valeur de `k` ; si la clé n'existe pas, la crée d'abord avec la valeur `[]` (pratique pour regrouper : `d.setdefault(espèce, []).append(masse)`) |

Un dictionnaire peut contenir des listes et d'autres dictionnaires : c'est la forme des fichiers JSON (§100.6.4) et des configurations. On lit alors les niveaux l'un après l'autre :

```python
>>> colony = {"Biscoe": {"Adelie": 44, "Gentoo": 124}, "Dream": {"Adelie": 56, "Chinstrap": 68}}
>>> colony["Dream"]["Chinstrap"]
68
>>> sum(colony["Biscoe"].values())
168
```

### 100.3.4 · Ensembles

Un **ensemble** (`set`) contient des éléments **sans doublon et sans ordre**. Il sert à dédoublonner et à tester très vite l'appartenance. `set()` crée un ensemble vide (`{}` crée un dictionnaire vide !).

```python
>>> islands = ["Torgersen", "Biscoe", "Dream", "Biscoe", "Torgersen"]
>>> set(islands) == {"Biscoe", "Dream", "Torgersen"}
True
>>> len(set(islands))
3
>>> adelie = {"Torgersen", "Biscoe", "Dream"}
>>> gentoo = {"Biscoe"}
>>> sorted(adelie & gentoo), sorted(adelie | gentoo), sorted(adelie - gentoo)
(['Biscoe'], ['Biscoe', 'Dream', 'Torgersen'], ['Dream', 'Torgersen'])
```

`&` est l'intersection (dans les deux), `|` la réunion (dans l'un ou l'autre), `-` la différence (dans le premier, pas dans le second). On trie avec `sorted` pour afficher dans un ordre stable, car un ensemble n'a pas d'ordre garanti.

### 100.3.5 · `collections.Counter`

`Counter` compte les occurrences de chaque valeur. C'est un dictionnaire dont les valeurs sont des comptes ; ses clés gardent l'ordre de première apparition.

```python
>>> from collections import Counter
>>> species = ["Adelie", "Gentoo", "Adelie", "Chinstrap", "Adelie", "Gentoo"]
>>> counts = Counter(species)
>>> counts
Counter({'Adelie': 3, 'Gentoo': 2, 'Chinstrap': 1})
>>> counts["Adelie"], counts["Emperor"]
(3, 0)
>>> counts.most_common(1)
[('Adelie', 3)]
```

Un `Counter` renvoie 0 pour une clé absente, au lieu de lever une `KeyError`. Tu recoderas cette fonction à la main dans mylearn (`utils.count_values`, 0A.63).

## 100.4 · Contrôle du flux

### 100.4.1 · Conditions : `if`, `elif`, `else`, opérateurs logiques

Un bloc `if` n'est exécuté que si sa condition est vraie. Le contenu du bloc est **indenté** de 4 espaces : en Python, c'est l'indentation qui délimite les blocs, pas des accolades. `elif` (« sinon si ») et `else` (« sinon ») sont facultatifs ; seul le **premier** bloc dont la condition est vraie s'exécute.

```python
>>> mass = 4800
>>> if mass < 3500:
...     category = "light"
... elif mass < 4500:
...     category = "medium"
... else:
...     category = "heavy"
>>> category
'heavy'
```

L'ordre des tests compte : si l'on testait `mass < 4500` avant `mass < 3500`, un manchot de 3000 g serait classé « medium ». Pour une condition courte, l'expression conditionnelle tient sur une ligne :

```python
>>> "heavy" if mass >= 4500 else "not heavy"
'heavy'
```

`and` et `or` s'arrêtent dès que le résultat est connu : dans `x is not None and x > 0`, le test `x > 0` n'est jamais fait si `x` vaut `None`.

### 100.4.2 · Boucles : `for`, `range`, `enumerate`, `zip`, `while`, `break`, `continue`

Une boucle **`for`** répète un bloc pour chaque élément d'une séquence (liste, chaîne, `range`, dictionnaire…).

```python
>>> total = 0
>>> for flipper in [181, 186, 195]:
...     total = total + flipper
>>> total
562
>>> list(range(5)), list(range(2, 10, 3))
([0, 1, 2, 3, 4], [2, 5, 8])
```

`range(début, fin, pas)` produit les entiers de `début` inclus à `fin` exclue (comme une tranche). `range(5)` va de 0 à 4.

| Outil | Ce qu'il donne à chaque tour | Exemple |
|---|---|---|
| `for x in liste` | l'élément | `for f in flippers:` |
| `for i in range(n)` | un indice 0, 1, …, n−1 | `for i in range(len(flippers)):` |
| `for i, x in enumerate(liste)` | l'indice **et** l'élément | le plus lisible quand on a besoin des deux |
| `for a, b in zip(l1, l2)` | un élément de chaque liste, en parallèle | s'arrête à la plus courte |
| `for k, v in d.items()` | clé et valeur d'un dictionnaire | |

```python
>>> names = ["Adelie", "Gentoo", "Chinstrap"]
>>> masses = [3700, 5076, 3733]
>>> for i, name in enumerate(names):
...     print(i, name)
0 Adelie
1 Gentoo
2 Chinstrap
>>> for name, mass in zip(names, masses):
...     print(f"{name:<10}{mass:>6} g")
Adelie      3700 g
Gentoo      5076 g
Chinstrap   3733 g
```

Une boucle **`while`** répète tant qu'une condition est vraie : on l'utilise quand on ne connaît pas le nombre de tours à l'avance. Le premier exemple suit la suite de Syracuse (si `n` est pair on le divise par 2, sinon on le remplace par `3n + 1`) et compte les étapes jusqu'à 1 : personne ne sait prédire ce nombre sans faire le calcul. `break` sort de la boucle immédiatement ; `continue` passe directement au tour suivant.

```python
>>> n, steps = 27, 0
>>> while n != 1:
...     n = n // 2 if n % 2 == 0 else 3 * n + 1
...     steps += 1
>>> steps
111
>>> for flipper in [181, -1, 195, 0, 230]:
...     if flipper <= 0:
...         continue
...     if flipper > 220:
...         break
...     print(flipper)
181
195
```

`steps += 1` est un raccourci pour `steps = steps + 1` (de même `-=`, `*=`…).

> ⚠️ **Piège classique — la boucle sans fin** — Une boucle `while` dont la condition ne devient jamais fausse tourne indéfiniment : interromps la cellule (■ ou *Interrompre*) et vérifie que la variable testée change bien à chaque tour. Autre piège : ne retire pas d'éléments d'une liste pendant que tu la parcours avec `for` ; construis plutôt une nouvelle liste.

### 100.4.3 · Compréhensions de listes, de dictionnaires et d'ensembles

Une **compréhension** construit une liste en une ligne à partir d'une autre séquence, avec un filtre facultatif. `[expression for x in séquence if condition]` se lit « la liste des *expression* pour chaque *x* de la séquence qui vérifie *condition* ».

```python
>>> flippers = [181, 186, 195, 193, 190, 217]
>>> [f / 10 for f in flippers]
[18.1, 18.6, 19.5, 19.3, 19.0, 21.7]
>>> [f for f in flippers if f >= 190]
[195, 193, 190, 217]
>>> {name: len(name) for name in ["Adelie", "Gentoo", "Chinstrap"]}
{'Adelie': 6, 'Gentoo': 6, 'Chinstrap': 9}
>>> {f // 10 * 10 for f in flippers} == {180, 190, 210}
True
```

Elle équivaut à une boucle avec `append`, en plus court :

```python
long_flippers = []
for f in flippers:
    if f >= 190:
        long_flippers.append(f)
```

Avec des accolades et `clé: valeur`, on obtient un dictionnaire ; avec des accolades seules, un ensemble. Si la compréhension ne tient plus sur une ligne lisible (deux boucles, plusieurs conditions), reviens à une boucle `for` classique : la lisibilité passe avant la concision.

## 100.5 · Fonctions

### 100.5.1 · Définir une fonction : paramètres, valeurs par défaut, `return`, portée

Une **fonction** donne un nom à un calcul qu'on veut réutiliser. On la définit avec `def`, on lui donne des **paramètres** entre parenthèses, et elle renvoie un résultat avec `return`. Le corps est indenté.

```python
>>> def kg(mass_g, decimals=1):
...     """Convert a mass from grams to kilograms, rounded."""
...     return round(mass_g / 1000, decimals)
>>> kg(3750)
3.8
>>> kg(3750, 2), kg(3750, decimals=3)
(3.75, 3.75)
```

- `mass_g` est un paramètre **obligatoire**, `decimals=1` a une **valeur par défaut** : on peut l'omettre à l'appel.
- À l'appel, on passe les **arguments** par position (`kg(3750, 2)`) ou par nom (`kg(3750, decimals=3)`) ; le nom rend l'appel plus lisible.
- `return` arrête la fonction et renvoie la valeur. Une fonction **sans** `return` renvoie `None` : `print` affiche, mais ne renvoie rien.

```python
>>> def show(mass_g):
...     print(f"{mass_g} g")
>>> result = show(3750)
3750 g
>>> print(result)
None
```

**Portée** (*scope*). Une variable créée dans une fonction est **locale** : elle n'existe que pendant l'appel et ne touche pas une variable du même nom à l'extérieur. La fonction peut lire les variables extérieures, mais c'est une mauvaise habitude : passe-lui plutôt ce dont elle a besoin en paramètre.

```python
>>> total = 100
>>> def add_ten(x):
...     total = x + 10
...     return total
>>> add_ten(5), total
(15, 100)
```

> ⚠️ **Piège classique — une liste comme valeur par défaut** — N'utilise jamais une liste ou un dictionnaire comme valeur par défaut (`def add(x, seen=[])`) : la valeur par défaut est créée **une seule fois**, à la définition, et partagée entre tous les appels. Écris `def add(x, seen=None):` puis `if seen is None: seen = []` dans le corps.

### 100.5.2 · Arguments positionnels et nommés, `*args`, `**kwargs`, arguments *keyword-only*

Une fonction peut accepter un nombre variable d'arguments :
- `*args` rassemble les arguments positionnels en trop dans un **tuple** ;
- `**kwargs` rassemble les arguments nommés en trop dans un **dictionnaire** ;
- un paramètre placé **après** `*` (ou après `*args`) ne peut être passé que **par son nom** (*keyword-only*) : c'est une protection contre les erreurs d'ordre, très utilisée par scikit-learn.

```python
>>> def describe(*values, unit="mm", **options):
...     label = options.get("label", "values")
...     return f"{label}: {len(values)} values, max {max(values)} {unit}"
>>> describe(181, 186, 195)
'values: 3 values, max 195 mm'
>>> describe(3750, 5076, unit="g", label="masses")
'masses: 2 values, max 5076 g'
>>> def train(X, y, *, learning_rate=0.1, epochs=10):
...     return learning_rate * epochs
>>> train([1], [0], learning_rate=0.01)
0.1
>>> train([1], [0], 0.01)
Traceback (most recent call last):
  ...
TypeError: train() takes 2 positional arguments but 3 were given
```

À l'inverse, `*` et `**` **déballent** une séquence ou un dictionnaire au moment de l'appel : `f(*liste)` passe chaque élément comme argument positionnel, `f(**dico)` chaque paire comme argument nommé.

```python
>>> settings = {"learning_rate": 0.5, "epochs": 4}
>>> train([1], [0], **settings)
2.0
>>> max(*[3, 9, 4])
9
```

### 100.5.3 · Fonctions comme valeurs : `lambda`, `key=`, `Callable`

En Python, une fonction est une valeur comme une autre : on peut la ranger dans une variable, la mettre dans une liste, la **passer en argument** à une autre fonction. `lambda` crée une petite fonction anonyme d'une seule expression : `lambda x: x * 2` équivaut à une fonction `def` qui renvoie `x * 2`.

Usage le plus fréquent : l'argument `key=` de `sorted`, `min` et `max`, qui dit **selon quoi** comparer.

```python
>>> penguins = [("Adelie", 3700), ("Gentoo", 5076), ("Chinstrap", 3733)]
>>> sorted(penguins, key=lambda p: p[1])
[('Adelie', 3700), ('Chinstrap', 3733), ('Gentoo', 5076)]
>>> max(penguins, key=lambda p: p[1])[0]
'Gentoo'
>>> sorted(["gentoo", "Adelie", "chinstrap"], key=str.lower)
['Adelie', 'chinstrap', 'gentoo']
>>> def apply_twice(f, x):
...     return f(f(x))
>>> apply_twice(lambda x: x + 3, 10)
16
```

Pour annoter un paramètre qui attend une fonction, on écrit `Callable` (du module `collections.abc`) : `Callable[[float], float]` se lit « une fonction qui prend un `float` et renvoie un `float` ». Tu le verras dans les signatures de mylearn (par exemple la fonction à dériver au ch. 5).

### 100.5.4 · Fermetures (*closures*)

Une fonction définie **à l'intérieur** d'une autre peut utiliser les variables de la fonction englobante, même après la fin de celle-ci : on appelle cela une **fermeture** (*closure*). C'est une façon de fabriquer des fonctions « préréglées ».

```python
>>> def make_scaler(factor):
...     def scale(x):
...         return x * factor
...     return scale
>>> to_kg = make_scaler(0.001)
>>> to_cm = make_scaler(0.1)
>>> to_kg(3750), to_cm(181)
(3.75, 18.1)
```

`make_scaler` renvoie une **fonction** ; `to_kg` se souvient que `factor` valait 0,001. Pour qu'une fonction intérieure **modifie** une variable de la fonction englobante (un compteur, par exemple), il faut la déclarer `nonlocal` :

```python
>>> def make_counter():
...     count = 0
...     def increment():
...         nonlocal count
...         count += 1
...         return count
...     return increment
>>> tick = make_counter()
>>> tick(), tick(), tick()
(1, 2, 3)
```

Retiens qu'une fermeture garde un **accès à la variable**, pas une photo de sa valeur au moment où la fonction intérieure est créée : l'exercice 🔮 0A.43 te montre la conséquence. Les fermetures reviennent au ch. 18, où chaque opération de la mini-bibliothèque d'autodifférentiation fabrique sa propre fonction de dérivation.

### 100.5.5 · Docstrings au format NumPy et annotations de type

Une **docstring** est la chaîne entre triples guillemets placée juste sous `def`. Elle explique ce que fait la fonction ; `help(fonction)` l'affiche. Le workbook (comme NumPy, SciPy, scikit-learn et mylearn) suit le **format NumPy** : une phrase de résumé, puis des sections soulignées de tirets.

Les **annotations de type** (`values: list[float]`, `-> float`) indiquent les types attendus. Python ne les vérifie pas à l'exécution : elles servent à la lecture, à l'éditeur et aux outils de vérification.

```python
def mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of numbers.

    Parameters
    ----------
    values : list of float
        The numbers to average. Must not be empty.

    Returns
    -------
    float
        The sum of the values divided by their count.

    Raises
    ------
    ValueError
        If ``values`` is empty.

    Examples
    --------
    >>> mean([1, 2, 3, 4])
    2.5
    """
    if len(values) == 0:
        raise ValueError("mean() of an empty list")
    return sum(values) / len(values)
```

| Annotation | Se lit |
|---|---|
| `x: int`, `x: float`, `x: str`, `x: bool` | un entier, un nombre à virgule, une chaîne, un booléen |
| `x: list[float]`, `x: dict[str, int]`, `x: tuple[int, int]` | une liste de `float`, un dictionnaire chaîne → entier, un couple d'entiers |
| `x: int \| None = None` | un entier **ou** `None` (valeur par défaut `None`) |
| `-> float`, `-> None` | ce que renvoie la fonction |
| `ArrayLike`, `np.ndarray` | « tout ce qui ressemble à un tableau » (liste, tuple, array) en entrée ; un array NumPy en sortie |

> 🕰️ **Mise à jour (2026) — les annotations de type** — **Le livre et les tutoriels anciens :** `typing.Optional[int]`, `typing.List[float]`, `typing.Callable`. · **Aujourd'hui :** `int | None` (Python ≥ 3.10, PEP 604), `list[float]` (Python ≥ 3.9, PEP 585) et `collections.abc.Callable`. · **Faut-il quand même l'apprendre ?** Savoir **lire** l'ancienne écriture, oui (beaucoup de code existant) ; écrire la nouvelle. · *Sources :* [PEP 604](https://peps.python.org/pep-0604/), [PEP 585](https://peps.python.org/pep-0585/), [doc `typing`](https://docs.python.org/3/library/typing.html).

### 100.5.6 · Fonctions récursives

Une fonction **récursive** s'appelle elle-même sur un problème plus petit. Elle a toujours deux parties :
- un **cas de base**, qui se résout sans appel récursif (sinon, la fonction s'appellerait sans fin) ;
- un **appel récursif**, sur une donnée plus petite, qui se rapproche du cas de base.

```python
>>> def factorial(n):
...     if n == 0:
...         return 1
...     return n * factorial(n - 1)
>>> factorial(5)
120
```

`factorial(5)` appelle `factorial(4)`, qui appelle `factorial(3)`… jusqu'à `factorial(0)`, qui renvoie 1 ; les résultats remontent ensuite : $1 \times 1 \times 2 \times 3 \times 4 \times 5 = 120$.

La récursivité est naturelle pour les **structures imbriquées** : un dictionnaire qui contient des dictionnaires, un arbre. Pour additionner tous les nombres d'une structure imbriquée, on traite le cas de base (un nombre) et on délègue le reste :

```python
>>> def total(tree):
...     if isinstance(tree, dict):
...         return sum(total(child) for child in tree.values())
...     return tree
>>> colony = {"Biscoe": {"Adelie": 44, "Gentoo": 124}, "Dream": {"Adelie": 56, "Chinstrap": 68}}
>>> total(colony)
292
```

Chaque appel en cours occupe une place dans la **pile d'appels** ; Python limite sa profondeur (environ 1 000 appels) et lève une `RecursionError` au-delà, souvent parce que le cas de base est oublié. Tu retrouveras la récursivité pour le codage de Huffman (ch. 6), les arbres de décision (ch. 13), l'ordre de calcul d'un réseau (ch. 16) et la rétropropagation automatique (ch. 18).

## 100.6 · Modules, erreurs et fichiers

### 100.6.1 · Modules, imports et bibliothèque standard

Un **module** est un fichier de code réutilisable ; un **paquet** (*package*) est un dossier de modules. On les charge avec `import`. Python est livré avec une grande **bibliothèque standard** ; les bibliothèques de data science (NumPy, pandas, scikit-learn, PyTorch) s'installent en plus (`pip install`, déjà fait pour toi sur Colab et par `requirements.txt` en local).

```python
>>> import math
>>> math.sqrt(16), math.pi, math.log(math.e)
(4.0, 3.141592653589793, 1.0)
>>> from statistics import mean, median
>>> mean([181, 186, 195]), median([181, 186, 195, 200])
(187.33333333333334, 190.5)
>>> import random
>>> random.seed(0)
>>> random.choice(["Adelie", "Gentoo", "Chinstrap"])
'Gentoo'
```

| Écriture | Effet |
|---|---|
| `import math` | on écrit ensuite `math.sqrt(2)` : on voit d'où vient la fonction |
| `from math import sqrt` | on écrit `sqrt(2)` directement |
| `import numpy as np` | alias : c'est la convention universelle (`np`, `pd` pour pandas, `plt` pour matplotlib) |

| Module standard | Pour quoi faire | Voir |
|---|---|---|
| `math`, `statistics`, `random` | fonctions mathématiques, statistiques simples, hasard | §100.2, §100.6 |
| `collections` | `Counter`, `defaultdict` | §100.3.5 |
| `pathlib`, `json`, `pickle` | chemins et fichiers, sérialisation | §100.6.3-4 |
| `re` | expressions régulières | §100.6.5 |
| `itertools`, `heapq` | combinaisons, produits, top-k | §100.6.6 |
| `time`, `datetime` | mesurer une durée, manipuler des dates | |
| `subprocess`, `sys`, `os` | lancer une commande, informations sur Python et le système | §100.1.3 |

> ⚠️ **Piège classique — un fichier qui masque un module** — Ne nomme jamais un de tes fichiers comme un module existant (`random.py`, `math.py`, `numpy.py`) : Python importerait **ton** fichier à la place du vrai module.

### 100.6.2 · Exceptions : lire un traceback, `raise`, `try`/`except`

Quand une erreur survient, Python lève une **exception** et affiche un **traceback** : la liste des appels en cours, du plus ancien au plus récent. **Lis d'abord la dernière ligne** (le type d'erreur et sa description), puis remonte jusqu'à la première ligne qui parle de **ton** code.

```python
>>> def mean(values):
...     return sum(values) / len(values)
>>> mean([])
Traceback (most recent call last):
  ...
ZeroDivisionError: division by zero
```

Ici : `ZeroDivisionError`, parce que `len([])` vaut 0. Les erreurs les plus courantes :

| Exception | Cause typique |
|---|---|
| `NameError` | variable ou fonction pas définie (faute de frappe, cellule pas exécutée) |
| `TypeError` | opération sur un mauvais type (`"a" + 1`), mauvais nombre d'arguments |
| `ValueError` | bon type, mauvaise valeur (`int("abc")`, liste vide refusée) |
| `IndexError` | indice hors de la liste (`l[10]` pour une liste de 5 éléments) |
| `KeyError` | clé absente d'un dictionnaire |
| `AttributeError` | méthode ou attribut inexistant (`[1, 2].push(3)`) |
| `ZeroDivisionError` | division par zéro |
| `FileNotFoundError` | fichier introuvable (mauvais dossier courant, faute dans le chemin) |
| `ModuleNotFoundError` | module pas installé, ou cellule de setup pas exécutée |
| `IndentationError`, `SyntaxError` | code mal écrit : parenthèse ou `:` oubliés, indentation incohérente |

**Lever** une exception, c'est refuser une entrée invalide avec un message clair, plutôt que renvoyer un résultat faux : `raise ValueError("message")`. **Rattraper** une exception se fait avec `try`/`except` : le bloc `try` est tenté, et si l'exception indiquée survient, le bloc `except` s'exécute à la place.

```python
>>> def safe_mean(values):
...     if len(values) == 0:
...         raise ValueError("safe_mean() needs at least one value")
...     return sum(values) / len(values)
>>> safe_mean([])
Traceback (most recent call last):
  ...
ValueError: safe_mean() needs at least one value
>>> try:
...     safe_mean([])
... except ValueError as error:
...     print("Problem:", error)
Problem: safe_mean() needs at least one value
```

> ⚠️ **Piège classique — `except:` tout seul** — N'écris pas `except:` tout seul (ni `except Exception:` pour tout) : tu cacherais aussi les vraies erreurs, y compris les fautes de frappe. Rattrape **l'exception précise** que tu sais traiter.

### 100.6.3 · Fichiers texte et chemins avec `pathlib`

Le module `pathlib` représente un chemin par un objet `Path`. L'opérateur `/` assemble les morceaux, avec le bon séparateur sur chaque système (sous Windows, Python affiche `WindowsPath('data\\penguins.csv')` au lieu de `PosixPath('data/penguins.csv')`). Les chemins relatifs partent du dossier courant : ici, la racine du dépôt.

```python
>>> from pathlib import Path
>>> path = Path("data") / "penguins.csv"
>>> path, path.name, path.suffix, path.exists()
(PosixPath('data/penguins.csv'), 'penguins.csv', '.csv', True)
>>> text = path.read_text(encoding="utf-8")
>>> text.splitlines()[:3]
['species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year', 'Adelie,Torgersen,39.1,18.7,181,3750,male,2007', 'Adelie,Torgersen,39.5,17.4,186,3800,female,2007']
```

Pour les gros fichiers, on lit **ligne par ligne** avec `open` dans un bloc `with`, qui referme le fichier à la sortie du bloc, même en cas d'erreur :

```python
>>> with open(path, encoding="utf-8") as f:
...     header = f.readline().strip()
...     n_lines = 1 + sum(1 for line in f)
>>> header.split(","), n_lines
(['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year'], 345)
```

| Opération | Code |
|---|---|
| lire tout le texte | `Path(p).read_text(encoding="utf-8")` |
| écrire du texte (remplace le contenu) | `Path(p).write_text(texte, encoding="utf-8")` |
| lire ligne par ligne | `with open(p, encoding="utf-8") as f: for line in f: ...` |
| écrire ligne par ligne | `with open(p, "w", encoding="utf-8") as f: f.write(ligne + "\n")` |
| créer un dossier | `Path(d).mkdir(parents=True, exist_ok=True)` |
| lister des fichiers | `sorted(Path("data").glob("*.csv"))` |

Précise **toujours** `encoding="utf-8"` : sans lui, Windows lit parfois les accents de travers. Le mode `"w"` écrase le fichier existant ; `"a"` ajoute à la fin.

> 🕰️ **Mise à jour (2026) — les chemins de fichiers** — **Les tutoriels anciens :** `os.path.join("data", "penguins.csv")` et des chemins en chaînes de caractères. · **Aujourd'hui :** `pathlib.Path("data") / "penguins.csv"`, avec `.read_text()`, `.exists()`, `.glob()`. · **Faut-il quand même l'apprendre ?** Savoir **lire** `os.path` (encore partout), écrire `pathlib`. · *Source :* [doc `pathlib`](https://docs.python.org/3/library/pathlib.html).

### 100.6.4 · Sérialiser : `json` et `pickle`

**Sérialiser**, c'est transformer des données en texte ou en octets pour les enregistrer dans un fichier, puis les relire plus tard.

- **JSON** est un format **texte**, lisible par un humain et par tous les langages. Il accepte les dictionnaires, listes, chaînes, nombres, booléens et `None` (qui devient `null`). C'est le bon choix pour des résultats d'expérience et des configurations.
- **pickle** est un format **binaire** propre à Python, qui enregistre presque n'importe quel objet (un modèle scikit-learn, par exemple).

```python
>>> import json
>>> results = {"model": "baseline", "accuracy": 0.73, "features": ["bill_length_mm", "flipper_length_mm"], "notes": None}
>>> text = json.dumps(results, indent=2)
>>> print(text)
{
  "model": "baseline",
  "accuracy": 0.73,
  "features": [
    "bill_length_mm",
    "flipper_length_mm"
  ],
  "notes": null
}
>>> json.loads(text) == results
True
>>> json.loads(json.dumps({"shape": (344, 8)}))
{'shape': [344, 8]}
```

Un tuple devient une liste en JSON (le format ne connaît pas les tuples). Pour un fichier : `json.dump(objet, f)` et `json.load(f)` avec un fichier ouvert en texte ; pour pickle : `pickle.dump(objet, f)` et `pickle.load(f)` avec un fichier ouvert en **binaire** (`"wb"`, `"rb"`).

> ⚠️ **Piège classique — charger un pickle inconnu** — Charger un pickle **exécute du code** contenu dans le fichier. Ne charge jamais un pickle dont tu ne connais pas l'origine (téléchargé, reçu par mail).

> 🕰️ **Mise à jour (2026) — sauvegarder des résultats** — **Les tutoriels anciens :** `pickle` pour tout sauvegarder. · **Aujourd'hui :** JSON pour les résultats et les configurations ; pickle ou `joblib` seulement pour tes propres objets ; pour les poids d'un réseau, des formats sûrs : `safetensors`, et `torch.load(..., weights_only=True)`, le comportement par défaut depuis PyTorch 2.6. · **Faut-il quand même l'apprendre ?** Oui : pickle reste très utilisé, mais avec l'avertissement de sécurité en tête. · *Sources :* [doc `pickle` (encadré d'avertissement)](https://docs.python.org/3/library/pickle.html), [PyTorch 2.6 release blog](https://pytorch.org/blog/pytorch2-6/).

### 100.6.5 · Expressions régulières (`re`) : les bases

Une **expression régulière** (*regex*) décrit un **motif** de texte : « des chiffres, puis un tiret, puis deux lettres »… Elle sert à vérifier un format, à extraire des morceaux, à remplacer. On l'écrit dans une chaîne brute `r"..."`, pour que les `\` soient pris tels quels.

| Motif | Correspond à |
|---|---|
| `\d` | un chiffre (`0` à `9` pour du texte courant ; exactement `[0-9]` avec `flags=re.ASCII`) |
| `\w` | un caractère de mot (lettre, chiffre ou `_`) |
| `\s` | un espace, une tabulation, un retour à la ligne |
| `[A-Z]`, `[aeiou]` | un caractère parmi ceux listés (ou dans l'intervalle) |
| `.` | n'importe quel caractère (sauf retour à la ligne) |
| `+`, `*`, `?` | le motif précédent 1 fois ou plus, 0 fois ou plus, 0 ou 1 fois |
| `{3}`, `{2,4}` | exactement 3 fois, de 2 à 4 fois |
| `^`, `$` | début, fin du texte |
| `( )` | un **groupe** : une partie à extraire |

```python
>>> import re
>>> log = "run 7 finished on 2026-09-14 with loss 0.312, run 8 on 2026-09-15 with loss 0.297"
>>> re.findall(r"\d{4}-\d{2}-\d{2}", log)
['2026-09-14', '2026-09-15']
>>> re.findall(r"loss (\d+\.\d+)", log)
['0.312', '0.297']
>>> match = re.search(r"run (\d+) finished on (\d{4})-(\d{2})", log)
>>> match.group(1), match.groups()
('7', ('7', '2026', '09'))
>>> bool(re.fullmatch(r"[A-Z]{3}-\d{2}", "ABC-12")), bool(re.fullmatch(r"[A-Z]{3}-\d{2}", "AB-123"))
(True, False)
>>> re.sub(r"\s+", " ", "too    many   spaces")
'too many spaces'
```

| Fonction | Renvoie |
|---|---|
| `re.search(motif, texte)` | la **première** correspondance (un objet `Match`), ou `None` |
| `re.fullmatch(motif, texte)` | une correspondance si **tout** le texte suit le motif, sinon `None` |
| `re.findall(motif, texte)` | la liste de **toutes** les correspondances (ou des groupes, s'il y en a) |
| `re.sub(motif, remplacement, texte)` | le texte avec les correspondances remplacées |

Les regex deviennent vite illisibles : commence simple, teste sur quelques exemples, et préfère `str.split` ou `str.startswith` quand ils suffisent.

### 100.6.6 · `itertools` et `heapq` (aperçu)

`itertools` génère des combinaisons sans boucles imbriquées ; `heapq` gère une **file de priorité** (on retire toujours le plus petit élément) et trouve les *k* plus grands ou plus petits éléments.

```python
>>> from itertools import combinations, product
>>> list(combinations(["a", "b", "c"], 2))
[('a', 'b'), ('a', 'c'), ('b', 'c')]
>>> list(product([0.1, 0.01], [16, 32]))
[(0.1, 16), (0.1, 32), (0.01, 16), (0.01, 32)]
>>> import heapq
>>> scores = [0.71, 0.93, 0.64, 0.88, 0.97, 0.52]
>>> heapq.nlargest(3, scores), heapq.nsmallest(2, scores)
([0.97, 0.93, 0.88], [0.52, 0.64])
>>> queue = []
>>> for item in [(3, "c"), (1, "a"), (2, "b")]:
...     heapq.heappush(queue, item)
>>> heapq.heappop(queue), heapq.heappop(queue)
((1, 'a'), (2, 'b'))
```

- `combinations(l, k)` : toutes les façons de choisir `k` éléments sans ordre (utile pour les paires de features) ;
- `product(a, b)` : toutes les combinaisons d'un élément de `a` et d'un élément de `b` (une **grille** de réglages, comme en recherche d'hyperparamètres au ch. 15) ;
- `heapq.nlargest(k, l)` : les `k` plus grands (le *top-k*) ; `heappush`/`heappop` servent pour le codage de Huffman au ch. 6.

## 100.7 · Classes et objets

Toutes les bibliothèques du workbook sont faites de **classes** : un `DataFrame` pandas, un modèle scikit-learn, une couche PyTorch. Savoir lire (et écrire) une classe, c'est comprendre `model.fit(X, y)`.

### 100.7.1 · `class`, `__init__`, attributs et méthodes ; `@dataclass`

Une **classe** est un plan de fabrication ; un **objet** (ou *instance*) est un exemplaire fabriqué selon ce plan. Un objet regroupe des **attributs** (ses données) et des **méthodes** (des fonctions qui agissent sur lui).

```python
>>> class Penguin:
...     """A penguin with a species and a body mass."""
...
...     def __init__(self, species, mass_g):
...         self.species = species
...         self.mass_g = mass_g
...
...     def mass_kg(self):
...         return self.mass_g / 1000
...
...     def eat(self, food_g):
...         self.mass_g += food_g
>>> pingu = Penguin("Adelie", 3750)
>>> pingu.species, pingu.mass_kg()
('Adelie', 3.75)
>>> pingu.eat(250)
>>> pingu.mass_g
4000
```

- `__init__` est appelée automatiquement à la création (`Penguin("Adelie", 3750)`) : elle **initialise** les attributs.
- `self` désigne l'objet lui-même. Il est le premier paramètre de chaque méthode, mais on ne le passe pas à l'appel : `pingu.mass_kg()` devient `Penguin.mass_kg(pingu)`.
- Une méthode peut modifier l'objet (`eat`) ou calculer à partir de lui (`mass_kg`).

Quand une classe sert surtout à **ranger des données**, le décorateur `@dataclass` écrit `__init__` (et un affichage lisible) à ta place à partir des annotations :

```python
>>> from dataclasses import dataclass
>>> @dataclass
... class Measure:
...     species: str
...     flipper_mm: float
...     island: str = "Biscoe"
>>> m = Measure("Gentoo", 217.0)
>>> m
Measure(species='Gentoo', flipper_mm=217.0, island='Biscoe')
>>> m.flipper_mm, m == Measure("Gentoo", 217.0)
(217.0, True)
```

Tu retrouveras `@dataclass` pour les nœuds d'un arbre de décision (ch. 13) et les configurations d'un projet (B7).

### 100.7.2 · Méthodes spéciales : `__repr__`, `__len__`, `__getitem__`, `__call__`, opérateurs

Les méthodes dont le nom est entouré de doubles tirets bas (*dunder methods*, « double underscore ») sont appelées par Python lui-même : `len(obj)` appelle `obj.__len__()`, `obj[i]` appelle `obj.__getitem__(i)`, `a + b` appelle `a.__add__(b)`, `obj(x)` appelle `obj.__call__(x)`. En les définissant, tes objets se comportent comme des objets natifs.

```python
>>> class Polynomial:
...     """A polynomial c0 + c1 x + c2 x**2 + ..., stored by its coefficients."""
...
...     def __init__(self, coefs):
...         self.coefs = list(coefs)
...
...     def __repr__(self):
...         return f"Polynomial({self.coefs})"
...
...     def __len__(self):
...         return len(self.coefs)
...
...     def __getitem__(self, i):
...         return self.coefs[i]
...
...     def __call__(self, x):
...         return sum(c * x ** k for k, c in enumerate(self.coefs))
...
...     def __eq__(self, other):
...         return isinstance(other, Polynomial) and self.coefs == other.coefs
...
...     def __mul__(self, scalar):
...         return Polynomial([c * scalar for c in self.coefs])
...
...     def __rmul__(self, scalar):
...         return self * scalar
>>> p = Polynomial([1, 0, 2])
>>> p
Polynomial([1, 0, 2])
>>> len(p), p[2], p(3)
(3, 2, 19)
>>> 3 * p, p * 3 == 3 * p
(Polynomial([3, 0, 6]), True)
```

| Méthode | Appelée par | Rôle |
|---|---|---|
| `__init__(self, ...)` | `Classe(...)` | initialiser |
| `__repr__(self)` | l'affichage dans le notebook, `repr(obj)` | une représentation lisible |
| `__len__(self)` | `len(obj)` | une longueur |
| `__getitem__(self, i)` | `obj[i]` | accès par indice ou par clé |
| `__call__(self, ...)` | `obj(...)` | rendre l'objet appelable comme une fonction |
| `__eq__(self, other)` | `obj == other` | égalité |
| `__add__`, `__mul__` | `obj + other`, `obj * other` | opérateurs |
| `__radd__`, `__rmul__` | `other + obj`, `other * obj` | opérateurs quand l'objet est **à droite** |

`3 * p` appelle d'abord `int.__mul__(3, p)`, qui ne sait pas multiplier un entier par un `Polynomial` ; Python essaie alors `p.__rmul__(3)`. C'est exactement ce que fera la classe `Value` du ch. 18 pour que `2 * x + 1` fonctionne avec des objets qui calculent leurs dérivées.

### 100.7.3 · Héritage et `super()`

Une classe peut **hériter** d'une autre : elle récupère ses attributs et méthodes, en ajoute et en redéfinit. La classe mère décrit ce qui est commun ; `super()` appelle la version de la classe mère.

```python
>>> class Transformer:
...     """Base class: learn something with fit, apply it with transform."""
...
...     def fit(self, values):
...         raise NotImplementedError
...
...     def transform(self, values):
...         raise NotImplementedError
...
...     def fit_transform(self, values):
...         return self.fit(values).transform(values)
>>> class Centerer(Transformer):
...     """Subtract the mean learned by fit."""
...
...     def fit(self, values):
...         self.mean_ = sum(values) / len(values)
...         return self
...
...     def transform(self, values):
...         return [v - self.mean_ for v in values]
>>> Centerer().fit_transform([180, 186, 195])
[-7.0, -1.0, 8.0]
>>> class Shifter(Centerer):
...     """Center, then add a constant."""
...
...     def __init__(self, shift):
...         super().__init__()
...         self.shift = shift
...
...     def transform(self, values):
...         return [v + self.shift for v in super().transform(values)]
>>> Shifter(100).fit([2, 4, 6]).transform([2, 4, 6])
[98.0, 100.0, 102.0]
>>> isinstance(Shifter(1), Transformer)
True
```

`Centerer` n'a pas écrit `fit_transform` : il l'**hérite** de `Transformer`. C'est l'organisation de scikit-learn : tous les estimateurs partagent `fit`, `predict` ou `transform`, et `fit` renvoie `self` pour pouvoir enchaîner (`model.fit(X, y).predict(X)`). Par convention, les attributs **appris** pendant `fit` finissent par un tiret bas (`mean_`). Les couches PyTorch héritent de même de `torch.nn.Module` et appellent `super().__init__()`.

### 100.7.4 · Itérables et générateurs (`yield`)

Un objet est **itérable** quand on peut le parcourir avec `for` : listes, chaînes, dictionnaires, fichiers, `range`… La boucle demande un *itérateur* avec `iter(obj)`, puis appelle `next(...)` jusqu'à l'exception `StopIteration`.

```python
>>> it = iter(["Adelie", "Gentoo"])
>>> next(it), next(it)
('Adelie', 'Gentoo')
>>> next(it)
Traceback (most recent call last):
  ...
StopIteration
```

Une **fonction génératrice** contient `yield` au lieu de `return` : chaque `yield` fournit une valeur **à la demande** et met la fonction en pause jusqu'à la suivante. Rien n'est calculé à l'avance : on peut produire des millions de valeurs, ou lire un énorme fichier, sans tout garder en mémoire.

```python
>>> def read_numbers(lines):
...     """Yield the numbers of text lines, skipping blank lines and comments."""
...     for line in lines:
...         line = line.strip()
...         if line and not line.startswith("#"):
...             yield float(line)
>>> gen = read_numbers(["181", "", "# comment", "186.5"])
>>> type(gen).__name__
'generator'
>>> list(gen)
[181.0, 186.5]
>>> list(gen)
[]
>>> sum(x * x for x in range(4))
14
```

Un générateur ne se parcourt **qu'une fois** : le deuxième `list(gen)` est vide. Une expression entre parenthèses comme `(x * x for x in range(4))` est un générateur en une ligne. Les `DataLoader` de PyTorch (ch. 20) livrent les mini-batches de cette façon, à la demande.

## 100.8 · NumPy

NumPy fournit l'**array** (`np.ndarray`) : un tableau de nombres **tous du même type**, rangés côte à côte en mémoire. Les calculs se font sur le tableau entier, dans du code compilé : c'est la base de pandas, scikit-learn et PyTorch.

> 🧮 **Rappel maths — vecteurs et matrices** — Un tableau à une dimension est un **vecteur** (une liste de nombres, par exemple les mesures d'un manchot) ; à deux dimensions, c'est une **matrice** : des lignes et des colonnes (un manchot par ligne, une mesure par colonne). On note sa forme « lignes × colonnes ». Le chapitre 0B détaille les opérations sur les vecteurs et les matrices ; ici, retiens l'image d'un tableau de nombres.

### 100.8.1 · Créer des arrays : `dtype`, `shape`, `ndim`

```python
>>> import numpy as np
>>> a = np.array([181, 186, 195])
>>> a, a.dtype, a.shape, a.ndim
(array([181, 186, 195]), dtype('int64'), (3,), 1)
>>> X = np.array([[39.1, 181.0], [39.5, 186.0], [40.3, 195.0], [36.7, 193.0]])
>>> X.shape, X.ndim, X.size, X.dtype
((4, 2), 2, 8, dtype('float64'))
>>> np.zeros(3), np.ones((2, 3)).shape, np.full(2, 7)
(array([0., 0., 0.]), (2, 3), array([7, 7]))
>>> np.arange(0, 10, 3), np.linspace(0, 1, 5)
(array([0, 3, 6, 9]), array([0.  , 0.25, 0.5 , 0.75, 1.  ]))
>>> np.array([1, 2.5]).dtype, np.array([True, False]).dtype, a.astype(float)
(dtype('float64'), dtype('bool'), array([181., 186., 195.]))
```

| Attribut ou fonction | Donne |
|---|---|
| `a.shape` | la forme : un tuple, `(4, 2)` pour 4 lignes et 2 colonnes |
| `a.ndim` | le nombre de dimensions (*axes*) |
| `a.size` | le nombre total d'éléments |
| `a.dtype` | le type des éléments : `int64`, `float64`, `bool`, `uint8` (entiers de 0 à 255, pour les pixels) |
| `np.zeros(forme)`, `np.ones(forme)`, `np.full(forme, v)` | un tableau rempli de 0, de 1, de `v` |
| `np.arange(début, fin, pas)` | comme `range`, mais en array (fin exclue) |
| `np.linspace(a, b, n)` | `n` valeurs régulièrement espacées de `a` à `b` **inclus** |
| `a.astype(float)` | une copie convertie dans un autre type |

Si tu mélanges entiers et nombres à virgule, tout devient `float64` : un array n'a qu'un seul type.

### 100.8.2 · Indexation, tranches, masques booléens ; vues et copies

On indexe un tableau 2D avec **deux indices séparés par une virgule** : `X[ligne, colonne]`. Les tranches marchent sur chaque axe ; `:` seul veut dire « tout ». NumPy 2 affiche un nombre extrait d'un array sous la forme `np.float64(181.0)` : c'est un nombre NumPy, qui s'utilise exactement comme un `float`.

```python
>>> X[0, 1], X[-1, 0]
(np.float64(181.0), np.float64(36.7))
>>> X[1], X[:, 1], X[1:3, :]
(array([ 39.5, 186. ]), array([181., 186., 195., 193.]), array([[ 39.5, 186. ],
       [ 40.3, 195. ]]))
>>> X[[0, 3]]
array([[ 39.1, 181. ],
       [ 36.7, 193. ]])
```

Un **masque booléen** est un tableau de `True`/`False` de la même forme ; il sélectionne les éléments où il vaut `True`. On combine les conditions avec `&` (et), `|` (ou), `~` (non), **entre parenthèses**.

```python
>>> flippers = X[:, 1]
>>> flippers > 185
array([False,  True,  True,  True])
>>> flippers[flippers > 185]
array([186., 195., 193.])
>>> X[(X[:, 0] > 38) & (X[:, 1] < 190)]
array([[ 39.1, 181. ],
       [ 39.5, 186. ]])
>>> (flippers > 185).sum(), (flippers > 185).mean()
(np.int64(3), np.float64(0.75))
```

`True` compte pour 1 et `False` pour 0 : `.sum()` d'un masque **compte** les éléments qui vérifient la condition, `.mean()` donne leur **proportion**.

**Vues et copies.** Une tranche (`X[1:3]`, `X[:, 1]`) est une **vue** : elle partage les données du tableau d'origine, donc la modifier modifie l'original. Un masque ou une liste d'indices (`X[[0, 3]]`) crée une **copie**. En cas de doute, `.copy()`.

```python
>>> b = np.arange(5)
>>> view = b[1:3]
>>> view[0] = 99
>>> b
array([ 0, 99,  2,  3,  4])
>>> c = b[b > 2]
>>> c[0] = -1
>>> b
array([ 0, 99,  2,  3,  4])
```

> ⚠️ **Piège classique — modifier une colonne sans le vouloir** — `X[:, 1] = 0` met à zéro une colonne **de `X` lui-même**. Si tu veux travailler sur une colonne sans toucher aux données, écris `col = X[:, 1].copy()`.

### 100.8.3 · Calcul vectorisé et fonctions universelles

Les opérations arithmétiques s'appliquent **élément par élément**, sans boucle. Les **fonctions universelles** (*ufuncs*) de NumPy (`np.sqrt`, `np.exp`, `np.log`, `np.abs`, `np.round`…) aussi.

```python
>>> masses = np.array([3750.0, 3800.0, 3250.0])
>>> masses / 1000
array([3.75, 3.8 , 3.25])
>>> masses - masses.min()
array([500., 550.,   0.])
>>> np.sqrt(np.array([4.0, 9.0])), np.log(np.array([1.0, np.e]))
(array([2., 3.]), array([0., 1.]))
>>> np.array([1, 2, 3]) * np.array([10, 20, 30])
array([10, 40, 90])
>>> np.where(masses > 3500, "heavy", "light")
array(['heavy', 'heavy', 'light'], dtype='<U5')
```

`np.where(condition, a, b)` choisit `a` là où la condition est vraie et `b` ailleurs. Ce style, qu'on appelle **vectorisation**, est plus court à écrire et beaucoup plus rapide qu'une boucle Python (l'exercice 🔬 0A.55 te fait mesurer l'écart) : la boucle est faite en langage compilé, sur des nombres rangés côte à côte.

### 100.8.4 · Broadcasting

Que se passe-t-il quand on combine deux tableaux de **formes différentes** ? NumPy « étire » virtuellement le plus petit, sans copier de données : c'est le **broadcasting**. La règle :

> Compare les formes **de droite à gauche**. Deux tailles sont compatibles si elles sont **égales** ou si l'une d'elles vaut **1** (une dimension absente compte comme 1). La forme du résultat prend, sur chaque axe, la plus grande des deux tailles.

| Forme de `A` | Forme de `B` | Résultat |
|---|---|---|
| `(4, 2)` | `()` (un nombre) | `(4, 2)` |
| `(4, 2)` | `(2,)` | `(4, 2)` : `B` est ajouté à chaque ligne |
| `(4, 2)` | `(4, 1)` | `(4, 2)` : `B` est ajouté à chaque colonne |
| `(3, 1)` | `(1, 4)` | `(3, 4)` : tableau de toutes les combinaisons |
| `(4, 2)` | `(4,)` | **erreur** : 2 et 4 incompatibles |

```python
>>> X - X.mean(axis=0)
array([[ 0.2 , -7.75],
       [ 0.6 , -2.75],
       [ 1.4 ,  6.25],
       [-2.2 ,  4.25]])
>>> X * np.array([1.0, 0.1])
array([[39.1, 18.1],
       [39.5, 18.6],
       [40.3, 19.5],
       [36.7, 19.3]])
>>> np.arange(3)[:, np.newaxis] + np.arange(4)
array([[0, 1, 2, 3],
       [1, 2, 3, 4],
       [2, 3, 4, 5]])
>>> X + np.ones(4)
Traceback (most recent call last):
  ...
ValueError: operands could not be broadcast together with shapes (4,2) (4,)
```

`X.mean(axis=0)` a la forme `(2,)` (une moyenne par colonne) : la soustraction **centre** chaque colonne d'un coup. `np.newaxis` (ou `None`) ajoute un axe de taille 1 : `np.arange(3)[:, np.newaxis]` a la forme `(3, 1)`.

> ⚠️ **Piège classique — les formes `(n,)` et `(n, 1)`** — Les formes `(n,)` et `(n, 1)` ne sont pas la même chose : `(n,) - (n, 1)` donne un tableau `(n, n)`, sans erreur ! Vérifie toujours la forme d'un résultat avec `.shape`.

### 100.8.5 · Réductions par axe, tri, `argmax`, `unique`

Une **réduction** résume un tableau par un nombre : `sum`, `mean`, `std`, `min`, `max`. Le paramètre `axis` dit **quel axe disparaît** :
- `axis=0` réduit **le long des lignes** : on obtient **une valeur par colonne** (la moyenne de chaque feature) ;
- `axis=1` réduit **le long des colonnes** : on obtient **une valeur par ligne** (par exemple par manchot) ;
- sans `axis`, on réduit tout le tableau.

```python
>>> M = np.array([[1, 5, 2], [4, 0, 6]])
>>> M.sum(), M.sum(axis=0), M.sum(axis=1)
(np.int64(18), array([5, 5, 8]), array([ 8, 10]))
>>> M.mean(axis=0, keepdims=True).shape
(1, 3)
>>> M.max(axis=1), M.argmax(axis=1), M.argmax()
(array([5, 6]), array([1, 2]), np.int64(5))
>>> np.sort(np.array([3, 1, 2])), np.argsort(np.array([3, 1, 2]))
(array([1, 2, 3]), array([1, 2, 0]))
>>> np.unique(np.array(["b", "a", "b", "c", "b"]), return_counts=True)
(array(['a', 'b', 'c'], dtype='<U1'), array([1, 3, 1]))
```

- `argmax` donne la **position** du maximum (le premier, en cas d'égalité) ; sans `axis`, c'est une position dans le tableau aplati. Il sert à transformer des scores en prédiction de classe (ch. 13 à 18).
- `argsort` donne les positions qui trieraient le tableau ; `np.unique` donne les valeurs distinctes triées, et leurs comptes avec `return_counts=True`.
- `keepdims=True` garde l'axe réduit avec une taille 1 : `(1, 3)` au lieu de `(3,)`, pratique pour le broadcasting.

### 100.8.6 · `reshape`, transposition, empilement

```python
>>> v = np.arange(6)
>>> v.reshape(2, 3)
array([[0, 1, 2],
       [3, 4, 5]])
>>> v.reshape(3, -1).shape, v.reshape(-1, 1).shape
((3, 2), (6, 1))
>>> v.reshape(2, 3).T
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> v.reshape(2, 3).ravel()
array([0, 1, 2, 3, 4, 5])
>>> a, b = np.array([1, 2]), np.array([3, 4])
>>> np.stack([a, b]), np.stack([a, b], axis=1)
(array([[1, 2],
       [3, 4]]), array([[1, 3],
       [2, 4]]))
>>> np.concatenate([a, b]), np.vstack([a, b]).shape, np.hstack([a, b]).shape
(array([1, 2, 3, 4]), (2, 2), (4,))
```

| Opération | Effet |
|---|---|
| `a.reshape(forme)` | même données, autre forme (le nombre d'éléments doit rester le même) ; `-1` = « calcule cette taille » |
| `a.T` | transposée : les lignes deviennent des colonnes |
| `a.ravel()`, `a.flatten()` | aplatit en une dimension (`ravel` renvoie une vue si possible, `flatten` une copie) |
| `np.stack(liste, axis)` | empile des tableaux de même forme **sur un nouvel axe** |
| `np.concatenate(liste, axis)` | colle des tableaux **le long d'un axe existant** |
| `np.vstack`, `np.hstack` | raccourcis : empiler verticalement (lignes), horizontalement (colonnes) |
| `a[:, np.newaxis]`, `a.squeeze()` | ajouter un axe de taille 1, retirer les axes de taille 1 |

### 100.8.7 · Aléatoire reproductible et mini-batches

Pour tirer des nombres au hasard, on crée un **générateur** avec une **graine** (*seed*) : la même graine donne toujours la même suite de nombres. C'est ce qui rend une expérience **reproductible**.

```python
>>> rng = np.random.default_rng(42)
>>> rng.integers(0, 10, size=5)
array([0, 7, 6, 4, 4])
>>> rng.random(3).round(3)
array([0.697, 0.094, 0.976])
>>> rng.normal(loc=0.0, scale=1.0, size=2).round(3)
array([ 0.128, -0.316])
>>> rng.permutation(5), rng.choice(["a", "b", "c"], size=2, replace=False)
(array([1, 0, 4, 3, 2]), array(['b', 'a'], dtype='<U1'))
>>> np.random.default_rng(42).integers(0, 10, size=5)
array([0, 7, 6, 4, 4])
```

| Méthode du générateur | Tire |
|---|---|
| `rng.random(n)` | `n` réels uniformes dans [0, 1) |
| `rng.integers(a, b, size)` | des entiers de `a` inclus à `b` exclu |
| `rng.normal(loc, scale, size)` | des valeurs selon une loi normale (moyenne `loc`, écart-type `scale` ; ch. 2) |
| `rng.choice(liste, size, replace)` | des éléments tirés, avec ou sans remise |
| `rng.permutation(n)` | les entiers de 0 à n−1 dans un ordre aléatoire |
| `rng.shuffle(a)` | mélange `a` **sur place** |

Le générateur avance à chaque tirage : deux appels successifs donnent des nombres différents, mais toute la séquence est la même d'une exécution à l'autre. Recréer le générateur avec la même graine redonne la même séquence (dernière ligne).

> 🕰️ **Mise à jour (2026) — le générateur aléatoire de NumPy** — **Les tutoriels anciens :** `np.random.seed(0)` puis `np.random.rand(...)`, un état aléatoire **global** partagé par tout le programme. · **Aujourd'hui :** `rng = np.random.default_rng(0)`, un générateur explicite que l'on passe aux fonctions (paramètre `rng` dans mylearn). · **Faut-il quand même l'apprendre ?** Savoir lire l'ancien style, écrire le nouveau : NumPy le recommande pour tout code neuf. · *Sources :* [doc NumPy « Random sampling »](https://numpy.org/doc/stable/reference/random/index.html), [NEP 19](https://numpy.org/neps/nep-0019-rng-policy.html).

**Mini-batches** (*mini-batches*). Pour entraîner un modèle, on ne présente pas toutes les données d'un coup : on les **mélange**, puis on les découpe en **mini-batches** de taille `b` (32, 64…). Un passage complet sur toutes les données s'appelle une **epoch**. Mélanger une permutation des **indices** (plutôt que les données) permet de découper `X` et `y` de la même façon.

> 🧮 **Rappel maths — compter les batches** — Avec $n$ exemples et des batches de taille $b$ : il y a $\lceil n / b \rceil$ batches par epoch, dont un dernier batch incomplet de $n - b \times \lfloor n / b \rfloor$ exemples s'il en reste. Avec `drop_last=True` (on jette le batch incomplet), il y a $\lfloor n / b \rfloor$ batches. Le nombre de **mises à jour** des poids est (nombre d'epochs) × (batches par epoch).

```python
>>> rng = np.random.default_rng(0)
>>> order = rng.permutation(10)
>>> [order[i:i + 4] for i in range(0, 10, 4)]
[array([4, 6, 2, 7]), array([3, 5, 9, 0]), array([8, 1])]
```

Tu coderas la version complète, `utils.iterate_minibatches`, dans mylearn (0A.66) : elle servira à toutes les boucles d'entraînement en NumPy des ch. 18 à 20.

### 100.8.8 · Les images sont des arrays : MNIST

Une image en niveaux de gris est un tableau 2D : une valeur par pixel, de 0 (noir) à 255 (blanc) en `uint8`. Un batch de $N$ images forme un tableau `(N, H, W)` (nombre, hauteur, largeur). Les images en couleur ont en plus des **canaux** (rouge, vert, bleu) : matplotlib affiche une image `(H, W, C)` (canaux en dernier), PyTorch range un batch en `(N, C, H, W)` (canaux en premier, même pour une image en niveaux de gris : `C = 1`).

```python
>>> import wb
>>> images, labels = wb.datasets.load_mnist("train", n=100)
>>> images.shape, images.dtype, labels.shape, labels[:10]
((100, 28, 28), dtype('uint8'), (100,), array([4, 0, 5, 4, 5, 1, 1, 2, 5, 8]))
>>> images.min(), images.max()
(np.uint8(0), np.uint8(255))
>>> images[0].shape, images[0, 14, 10:18]
((28, 28), array([254, 254, 254, 254, 254, 254, 254, 254], dtype=uint8))
>>> images.reshape(len(images), -1).shape, (images / 255.0).max()
((100, 784), np.float64(1.0))
```

- `images[0]` est la première image, `(28, 28)` ; `images[0, 14]` sa 15ᵉ ligne de pixels.
- `images.reshape(len(images), -1)` aplatit chaque image en un vecteur de $28 \times 28 = 784$ valeurs : c'est la forme `(n_samples, n_features)` qu'attendent les modèles classiques.
- Diviser par 255 ramène les pixels dans [0, 1] : c'est une **normalisation** (ch. 12). Attention, cela crée un tableau `float64`, huit fois plus lourd qu'en `uint8`.

## 100.9 · pandas

pandas manipule des **tableaux de données** dont les colonnes ont un nom et peuvent avoir des types différents (texte, nombres, dates) : c'est l'outil de tous les jours pour lire, nettoyer et explorer un dataset, avant de passer à NumPy pour les calculs des modèles.

### 100.9.1 · `DataFrame` et `Series` : `read_csv`, `head`, `info`, `describe`

Un **DataFrame** est un tableau : des lignes (une par exemple, repérées par un **index**) et des colonnes nommées. Une colonne seule est une **Series**. On lit un fichier CSV avec `pd.read_csv` ; dans le workbook, `wb.datasets.load_penguins()` fait la même chose pour Palmer Penguins.

```python
>>> import numpy as np
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "species": ["Adelie", "Gentoo", "Adelie", "Chinstrap", "Gentoo", "Adelie"],
...     "island": ["Dream", "Biscoe", "Torgersen", "Dream", "Biscoe", "Dream"],
...     "flipper_length_mm": [190, 215, np.nan, 196, 222, 186],
...     "body_mass_g": [3600, 5200, 3450, 3700, 5550, np.nan],
... })
>>> df
     species     island  flipper_length_mm  body_mass_g
0     Adelie      Dream              190.0       3600.0
1     Gentoo     Biscoe              215.0       5200.0
2     Adelie  Torgersen                NaN       3450.0
3  Chinstrap      Dream              196.0       3700.0
4     Gentoo     Biscoe              222.0       5550.0
5     Adelie      Dream              186.0          NaN
>>> df.shape, list(df.columns)
((6, 4), ['species', 'island', 'flipper_length_mm', 'body_mass_g'])
>>> df.dtypes
species               object
island                object
flipper_length_mm    float64
body_mass_g          float64
dtype: object
>>> df.describe().round(1)
       flipper_length_mm  body_mass_g
count                5.0          5.0
mean               201.8       4300.0
std                 15.8        993.1
min                186.0       3450.0
25%                190.0       3600.0
50%                196.0       3700.0
75%                215.0       5200.0
max                222.0       5550.0
>>> df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 4 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   species            6 non-null      object 
 1   island             6 non-null      object 
 2   flipper_length_mm  5 non-null      float64
 3   body_mass_g        5 non-null      float64
dtypes: float64(2), object(2)
memory usage: 324.0+ bytes
```

| Commande | Donne |
|---|---|
| `pd.read_csv("data/penguins.csv")` | le DataFrame lu depuis un fichier CSV |
| `df.head(n)`, `df.tail(n)` | les `n` premières ou dernières lignes (5 par défaut) |
| `df.shape` | `(nombre de lignes, nombre de colonnes)` |
| `df.columns`, `df.dtypes` | les noms des colonnes, leurs types (`object` = texte) |
| `df.info()` | types, nombre de valeurs **non manquantes** par colonne, mémoire |
| `df.describe()` | nombre, moyenne, écart-type, minimum, quartiles, maximum des colonnes numériques |

`NaN` (*Not a Number*) marque une **valeur manquante** ; une colonne d'entiers qui contient un `NaN` devient une colonne de `float64`.

```python
>>> penguins = pd.read_csv("data/penguins.csv")
>>> penguins.head(3)
  species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex  year
0  Adelie  Torgersen            39.1           18.7              181.0       3750.0    male  2007
1  Adelie  Torgersen            39.5           17.4              186.0       3800.0  female  2007
2  Adelie  Torgersen            40.3           18.0              195.0       3250.0  female  2007
```

### 100.9.2 · Sélectionner : colonnes, `loc`, `iloc`, filtres

```python
>>> df["species"]
0       Adelie
1       Gentoo
2       Adelie
3    Chinstrap
4       Gentoo
5       Adelie
Name: species, dtype: object
>>> df[["species", "body_mass_g"]].head(2)
  species  body_mass_g
0  Adelie       3600.0
1  Gentoo       5200.0
>>> df.loc[1, "island"], df.iloc[1, 1]
('Biscoe', 'Biscoe')
>>> df.loc[df["species"] == "Gentoo", "body_mass_g"]
1    5200.0
4    5550.0
Name: body_mass_g, dtype: float64
>>> df.iloc[:2, 2:]
   flipper_length_mm  body_mass_g
0              190.0       3600.0
1              215.0       5200.0
>>> df[(df["island"] == "Dream") & (df["body_mass_g"] > 3650)]
     species island  flipper_length_mm  body_mass_g
3  Chinstrap  Dream              196.0       3700.0
>>> df[df["species"].isin(["Gentoo", "Chinstrap"])].shape
(3, 4)
```

| Écriture | Sélectionne |
|---|---|
| `df["col"]` | une colonne (une Series) |
| `df[["a", "b"]]` | plusieurs colonnes (un DataFrame) : note les **doubles** crochets |
| `df.loc[lignes, colonnes]` | par **labels** : valeurs de l'index et noms de colonnes ; une tranche `loc` **inclut** sa fin |
| `df.iloc[lignes, colonnes]` | par **positions** entières, comme NumPy (fin exclue) |
| `df[masque]` | les lignes où le masque booléen vaut `True` |
| `df.loc[masque, "col"] = valeur` | modifie la colonne sur les lignes choisies (la bonne façon d'écrire) |

Comme en NumPy, on combine les conditions avec `&`, `|` et `~`, **chaque condition entre parenthèses** : `and` et `or` ne marchent pas sur des colonnes entières (l'exercice 🐛 0A.58 te montre pourquoi).

### 100.9.3 · Valeurs manquantes et doublons

```python
>>> df.isna().sum()
species              0
island               0
flipper_length_mm    1
body_mass_g          1
dtype: int64
>>> df.dropna().shape, df.dropna(subset=["body_mass_g"]).shape
((4, 4), (5, 4))
>>> df["flipper_length_mm"].fillna(df["flipper_length_mm"].median()).tolist()
[190.0, 215.0, 196.0, 196.0, 222.0, 186.0]
>>> df["body_mass_g"].mean(), df["body_mass_g"].count()
(np.float64(4300.0), np.int64(5))
>>> dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
>>> dup.duplicated().sum(), dup.drop_duplicates().shape
(np.int64(1), (6, 4))
```

| Commande | Effet |
|---|---|
| `df.isna()` | tableau de booléens : `True` là où la valeur manque ; `.sum()` compte par colonne |
| `df.dropna()` | retire les lignes qui ont **au moins une** valeur manquante ; `subset=[...]` ne regarde que certaines colonnes |
| `s.fillna(v)` | remplace les manquants d'une colonne par `v` (une constante, la moyenne, la médiane…) |
| `df.duplicated()` | `True` pour chaque ligne qui répète une ligne précédente |
| `df.drop_duplicates()` | retire les lignes en double |

Les calculs de pandas (`mean`, `sum`, `count`…) **ignorent** les valeurs manquantes : la moyenne ci-dessus porte sur 5 masses, pas 6. Supprimer ou remplacer des manquants n'est jamais neutre (qui disparaît du dataset ?) : on y revient en détail au ch. 12.

> ⚠️ **Piège classique — `NaN` n'est égal à rien** — `NaN` n'est égal à rien, pas même à lui-même : `np.nan == np.nan` vaut `False`. Pour tester une valeur manquante, utilise `pd.isna(x)` ou `np.isnan(x)`, jamais `x == np.nan`.

### 100.9.4 · Compter et regrouper : `value_counts`, `groupby`, tri

```python
>>> df["species"].value_counts()
species
Adelie       3
Gentoo       2
Chinstrap    1
Name: count, dtype: int64
>>> df["island"].value_counts(normalize=True).round(2)
island
Dream        0.50
Biscoe       0.33
Torgersen    0.17
Name: proportion, dtype: float64
>>> df.groupby("species")["body_mass_g"].mean()
species
Adelie       3525.0
Chinstrap    3700.0
Gentoo       5375.0
Name: body_mass_g, dtype: float64
>>> df.groupby("island")["flipper_length_mm"].agg(["count", "mean", "max"])
           count        mean    max
island                             
Biscoe         2  218.500000  222.0
Dream          3  190.666667  196.0
Torgersen      0         NaN    NaN
>>> df.sort_values("body_mass_g", ascending=False).head(2)
  species  island  flipper_length_mm  body_mass_g
4  Gentoo  Biscoe              222.0       5550.0
1  Gentoo  Biscoe              215.0       5200.0
```

Sur l'île de Torgersen, la seule longueur de nageoire manque : `count` vaut 0 et la moyenne `NaN`. `groupby` suit le schéma **séparer, appliquer, combiner** : pandas sépare les lignes par valeur de la clé (`species`), applique un calcul à chaque groupe (`mean`), puis combine les résultats dans une Series indexée par la clé. `agg` applique plusieurs calculs à la fois. `value_counts` trie par effectif décroissant ; `normalize=True` donne des proportions. Enfin, `s.idxmax()` et `s.idxmin()` renvoient le **label** (l'index) du maximum et du minimum d'une Series, pas leur valeur : `df.groupby("species")["body_mass_g"].mean().idxmax()` vaut `'Gentoo'`.

### 100.9.5 · De pandas à NumPy : `X` et `y`

Les modèles (scikit-learn, mylearn, PyTorch) attendent des tableaux NumPy :
- `X`, les **features**, de forme `(n_samples, n_features)` : une ligne par exemple, une colonne par mesure ;
- `y`, les **labels** (la réponse à prédire), de forme `(n_samples,)`.

```python
>>> clean = df.dropna()
>>> X = clean[["flipper_length_mm", "body_mass_g"]].to_numpy()
>>> y = clean["species"].to_numpy()
>>> X.shape, y.shape, X.dtype
((4, 2), (4,), dtype('float64'))
>>> X[:2], y[:2]
(array([[ 190., 3600.],
       [ 215., 5200.]]), array(['Adelie', 'Gentoo'], dtype=object))
```

On retire d'abord les lignes incomplètes (ou on les complète), et l'on sélectionne les mêmes lignes pour `X` et `y` : ici, `clean` sert aux deux.

> 🕰️ **Mise à jour (2026) — les affectations en chaîne de pandas** — **Les tutoriels anciens :** des affectations « en chaîne » comme `df[df["x"] > 0]["y"] = 0`, qui modifient une copie temporaire et déclenchent un `SettingWithCopyWarning` (et parfois ne modifient rien). · **Aujourd'hui :** le mode *Copy-on-Write* est une option en pandas 2.2 (la version du workbook) et le comportement par défaut depuis pandas 3.0 ; l'affectation en chaîne ne fonctionne plus du tout. Écris toujours `df.loc[df["x"] > 0, "y"] = 0`, ou fais une copie explicite avec `.copy()`. · **Faut-il quand même l'apprendre ?** Oui : cette bonne habitude marche avec les deux versions. · *Source :* [doc pandas « Copy-on-Write (CoW) »](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).

## 100.10 · matplotlib

### 100.10.1 · `plot`, `scatter`, `hist` ; titres, axes et légendes

matplotlib dessine des graphiques. Le workbook utilise toujours le même schéma : on crée une **figure** (`fig`, la page) contenant un ou plusieurs **axes** (`ax`, les zones de tracé), on dessine sur `ax`, on nomme les axes, puis `plt.show()` affiche.

```python
import matplotlib.pyplot as plt
import numpy as np
import wb

penguins = wb.datasets.load_penguins(dropna=True)

fig, ax = plt.subplots(figsize=(6, 4))
for species, group in penguins.groupby("species"):
    ax.scatter(group["flipper_length_mm"], group["body_mass_g"], s=12, label=species)
ax.set_xlabel("flipper length (mm)")
ax.set_ylabel("body mass (g)")
ax.set_title("Palmer penguins")
ax.legend()
plt.show()
```

| Méthode de `ax` | Dessine |
|---|---|
| `ax.plot(x, y, label=...)` | une courbe (points reliés) : une évolution, une fonction |
| `ax.scatter(x, y, s=..., c=...)` | un nuage de points : la relation entre deux variables |
| `ax.hist(valeurs, bins=20)` | un histogramme : la répartition d'une variable |
| `ax.bar(noms, hauteurs)` | un diagramme en barres : comparer des catégories |
| `ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title` | les légendes des axes et le titre |
| `ax.legend()` | la légende des séries (celles qui ont un `label`) |

```python
x = np.linspace(-3, 3, 100)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x, x ** 2, label="x²")
ax.plot(x, np.exp(x) / 10, label="exp(x) / 10", linestyle="--")
ax.hist(np.random.default_rng(0).normal(size=500), bins=30, density=True, alpha=0.4, label="normal sample")
ax.legend()
plt.show()
```

Un graphique utile a toujours : des axes nommés **avec leurs unités**, un titre qui dit ce qu'on regarde, une légende si plusieurs séries. `fig.savefig("figure.png", dpi=150)` l'enregistre dans un fichier (avant `plt.show()`).

### 100.10.2 · Figures à plusieurs panneaux et images : `subplots`, `imshow`, `show_images`

`plt.subplots(lignes, colonnes)` crée une grille d'axes ; `axes` est alors un tableau NumPy d'axes (`axes[0]`, ou `axes[i, j]` pour une grille 2D). `ax.imshow(image, cmap="gray_r")` affiche un tableau 2D comme une image.

```python
images, labels = wb.datasets.load_mnist("train", n=100)

fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
for ax, image, label in zip(axes, images, labels):
    ax.imshow(image, cmap="gray_r")
    ax.set_title(f"label {label}")
    ax.axis("off")
fig.tight_layout()
plt.show()

wb.plot.show_images(images[:16], labels=labels[:16], ncols=8)
plt.show()
```

`fig.tight_layout()` évite que les titres se chevauchent. `wb.plot.show_images` fait la même grille en une ligne, avec les labels (et, plus tard, les prédictions du modèle).

## 100.11 · Outils du développeur

### 100.11.1 · git : `status`, `add`, `commit`, `push`, `pull` ; messages de commit ; `.gitignore`

**git** garde l'historique de ton travail sous forme de **commits** : des photos du projet, chacune avec un message qui dit ce qui a changé et pourquoi. Tu peux revenir en arrière, comparer deux versions, travailler à plusieurs. **GitHub** héberge une copie du dépôt en ligne (le *remote*) : c'est ta sauvegarde et ton **portfolio**.

Le cycle de base, dans un terminal, à la racine du dépôt :

```bash
git status                         # what changed? (red: not staged, green: staged)
git add mon_travail/ch00a_python/  # stage the files for the next commit
git commit -m "0A: finish exercises 0A.14 to 0A.22"
git push                           # send the commits to GitHub
git pull --rebase --autostash      # get the new chapters (the setup cell does it on Colab)
```

| Commande | Effet |
|---|---|
| `git clone URL` | copie un dépôt GitHub sur ton ordinateur (une seule fois) |
| `git status` | l'état : fichiers modifiés, fichiers prêts à être commités |
| `git add chemin` | ajoute un fichier (ou un dossier) au prochain commit (*staging*) |
| `git commit -m "message"` | enregistre une photo des fichiers ajoutés, avec un message |
| `git log --oneline` | l'historique, un commit par ligne |
| `git diff` | les lignes modifiées **pas encore ajoutées** avec `git add` (`git diff --staged` : celles qui sont ajoutées ; `git diff HEAD` : tout depuis le dernier commit) |
| `git restore fichier` | ⚠️ efface **définitivement** les modifications pas encore ajoutées d'un fichier (après un `git add` : `git restore --staged --worktree fichier`) |
| `git push` / `git pull` | envoie tes commits sur GitHub / récupère ceux de GitHub |

Avant ton premier commit, présente-toi une fois pour toutes : `git config --global user.name "Ton Nom"` et `git config --global user.email "ton.email@exemple.fr"`.

**Un bon message de commit** tient en une ligne courte qui dit **ce que fait** le commit, à l'impératif ou au présent : « 0A: implement mean and its test », pas « modifs » ni « update ». Un commit = une étape cohérente.

**Dans le workbook**, ne commite que `mon_travail/` : les autres dossiers sont mis à jour par Claude, et `git pull --rebase --autostash` rapatrie ses nouveautés sans toucher à ton travail (règle d'or n° 4 du README).

**`.gitignore`** liste les fichiers que git doit **ignorer** : fichiers temporaires, caches, données lourdes, secrets. Une ligne par motif :

```text
__pycache__/
.ipynb_checkpoints/
*.pyc
data/downloads/
*.pdf
.env
```

> ⚠️ **Piège classique — un secret commité** — Un mot de passe, une clé d'API ou un gros fichier commité reste **dans l'historique**, même si tu le supprimes ensuite. Mets-le dans `.gitignore` **avant** le premier commit.

### 100.11.2 · git : branches (aperçu)

Une **branche** est une ligne de travail parallèle : tu essaies une idée sans toucher à la branche principale `main`, puis tu fusionnes si c'est concluant, ou tu jettes la branche sinon.

```bash
git switch -c essai-normalisation   # create the branch and move onto it
# ... edit, then git add / git commit as usual ...
git switch main                      # back to the main branch
git merge essai-normalisation        # bring the branch's commits into main
git branch -d essai-normalisation    # delete the merged branch
```

En entreprise, on ne fusionne pas soi-même dans `main` : on pousse sa branche sur GitHub et on ouvre une **pull request**, que des collègues relisent avant la fusion (0A.E5).

> 🕰️ **Mise à jour (2026) — la branche par défaut et `git switch`** — **Les tutoriels anciens :** branche par défaut `master`, et `git checkout` pour tout (changer de branche, annuler une modification). · **Aujourd'hui :** la branche par défaut des nouveaux dépôts GitHub s'appelle `main` depuis le 1ᵉʳ octobre 2020 ; `git switch` change de branche et `git restore` annule une modification (Git 2.23, 2019). · **Faut-il quand même l'apprendre ?** Savoir lire `checkout`, écrire `switch` et `restore`. · *Sources :* [GitHub Changelog (2020-10-01)](https://github.blog/changelog/2020-10-01-the-default-branch-for-newly-created-repositories-is-now-main/), [doc `git switch`](https://git-scm.com/docs/git-switch).

### 100.11.3 · pytest : lancer, lire et écrire des tests

Un **test** est une petite fonction qui vérifie qu'une autre fonction fait ce qu'elle doit. pytest trouve et exécute automatiquement toutes les fonctions `test_...` des fichiers `test_*.py`.

```python
import pytest


def mean(values):
    if len(values) == 0:
        raise ValueError("mean() of an empty list")
    return sum(values) / len(values)


def test_mean_small_list():
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_floats():
    assert mean([0.1, 0.2]) == pytest.approx(0.15)


def test_mean_empty_raises():
    with pytest.raises(ValueError):
        mean([])


@pytest.mark.parametrize("values, expected", [([5], 5), ([1, 3], 2), ([-1, 1], 0)])
def test_mean_several_cases(values, expected):
    assert mean(values) == expected
```

| Outil | Sert à |
|---|---|
| `assert condition` | échouer si la condition est fausse |
| `pytest.approx(x)` | comparer des `float` avec une tolérance (jamais `==` sur des `float`) |
| `with pytest.raises(ValueError):` | vérifier qu'une erreur est bien levée |
| `@pytest.mark.parametrize(...)` | lancer le même test sur plusieurs cas |

On lance `python -m pytest tests/test_ch00a_utils.py -q` depuis la racine pour les tests d'un chapitre (`python -m pytest tests/ -q` les lance tous, y compris ceux de l'outillage du workbook, dans `tests/infra/`). Chaque point `.` est un test réussi, `F` un échec (*failed*, avec le détail en dessous), `s` un test ignoré (*skipped*, par exemple parce que ton module mylearn n'existe pas encore), `E` une erreur pendant la préparation du test. La dernière ligne résume : `5 passed, 1 failed in 0.12s` ; juste avant, une ligne ℹ️ liste les modules mylearn pas encore créés, dont les tests sont ignorés.

### 100.11.4 · doctest : des exemples exécutables dans les docstrings

Les exemples `>>>` de la section *Examples* d'une docstring (§100.5.5) peuvent être **vérifiés automatiquement** : le module `doctest` exécute chaque ligne `>>>` et compare ce qui s'affiche avec la ligne attendue. La documentation reste ainsi juste.

```python
>>> import doctest
>>> def half(x):
...     """Return half of x.
...
...     >>> half(10)
...     5.0
...     """
...     return x / 2
>>> doctest.run_docstring_examples(half, {"half": half}, name="half")
>>> def double(x):
...     """Return twice x.
...
...     >>> double(10)
...     20
...     """
...     return x + x + 1
>>> doctest.run_docstring_examples(double, {"double": double}, name="double")
**********************************************************************
File "__main__", line 4, in double
Failed example:
    double(10)
Expected:
    20
Got:
    21
```

Rien ne s'affiche quand tout est juste ; en cas d'écart, doctest montre l'exemple, la valeur attendue et la valeur obtenue (ici, `double` contient volontairement une erreur : `x + x + 1`). Sur un fichier entier : `python -m doctest -v mon_module.py` (ou `python -m pytest --doctest-modules mon_module.py`).

### 100.11.5 · mylearn : stubs, référence et tests fondés sur un oracle

Ta librairie `mylearn` se construit chapitre par chapitre avec trois ingrédients :

| Élément | Où | Rôle |
|---|---|---|
| le **stub** (squelette) | `templates/mylearn_stubs/`, copié dans `mon_travail/mylearn/` | signature, docstring complète, et `raise NotImplementedError` à remplacer par ton code |
| la **référence** | `solutions/mylearn_ref/` | une solution complète, à lire **après** avoir essayé |
| les **tests** | `tests/test_*.py` | comparent ta fonction à un **oracle** : une bibliothèque de confiance (NumPy, scikit-learn, PyTorch) qui calcule la même chose |

Un test « à oracle » tire des entrées au hasard et vérifie que ta fonction donne le même résultat que l'oracle (par exemple `mylearn._example.mean` contre `np.mean`), puis ajoute des tests de **propriétés** et de **cas limites** (liste d'un seul élément, liste vide qui doit lever `ValueError`). Il ne révèle pas la solution : il dit seulement si ton code se comporte comme attendu.

Le cycle, que tu fais pour la première fois en 0A.26 :
1. `python tools/start_chapter.py 0A` copie les stubs du chapitre dans `mon_travail/mylearn/` ;
2. ouvre le stub, lis la docstring, remplace `raise NotImplementedError(...)` par ton code ;
3. `python -m pytest tests/test_example_mylearn.py -q` : rouge (`F`) tant que ce n'est pas juste, vert (`.`) ensuite ;
4. `python -m pytest --impl=ref` fait tourner les mêmes tests sur la référence (utile pour vérifier qu'un test est bien possible à réussir).

Si tu sautes un chapitre, ses modules manquants sont pris dans la référence quand un chapitre suivant en a besoin, avec un message ; les modules du chapitre en cours ne sont jamais remplacés : une coche ✅ vient toujours de **ton** code.

## Les pièges classiques ⚠️ (récapitulatif)

| Piège | Exemple | Ce qu'il faut faire |
|---|---|---|
| cellules exécutées dans le désordre | le notebook marche chez toi, pas après redémarrage | *Restart and run all* avant de partager |
| `l = l.sort()` | `l` vaut `None` | `l.sort()` seul, ou `l = sorted(l)` |
| `b = a` sur une liste ou un array | modifier `b` modifie `a` | `a.copy()` |
| liste comme valeur par défaut | la « valeur par défaut » grossit d'un appel à l'autre | `def f(x, acc=None)` |
| `"3" + 1`, nombre lu dans un fichier | `TypeError`, ou `"31"` au lieu de 4 | convertir avec `int` / `float` |
| `0.1 + 0.2 == 0.3` | `False` | `math.isclose`, `np.isclose`, `pytest.approx` |
| `x == np.nan` | toujours `False` | `pd.isna(x)`, `np.isnan(x)` |
| `and` / `or` entre colonnes ou tableaux | `ValueError: The truth value ... is ambiguous` | `&` / `|` et des parenthèses autour de chaque condition |
| `axis` oublié ou inversé | un seul nombre au lieu d'un par colonne, ou l'inverse | `axis=0` : une valeur par colonne ; vérifier `.shape` |
| `(n,)` combiné avec `(n, 1)` | un tableau `(n, n)` inattendu, sans erreur | vérifier les formes, `reshape` explicite |
| tranche NumPy modifiée | les données d'origine changent | `.copy()` |
| affectation en chaîne pandas | un avertissement (`SettingWithCopyWarning` ou `FutureWarning`), et selon les cas rien ne change | `df.loc[masque, "col"] = v` |
| fichier introuvable | `FileNotFoundError` | `pwd` ; lancer depuis la racine du dépôt |
| `except:` qui attrape tout | une vraie erreur passe inaperçue | rattraper l'exception précise |
| secret ou gros fichier commité | reste dans l'historique git | `.gitignore` avant le premier commit |
| charger un pickle inconnu | exécution de code arbitraire | JSON, ou des fichiers de confiance seulement |

## Liens avec les autres chapitres 🔗

- **0B (maths)** : les arrays deviennent des vecteurs et des matrices ; `@` fait le produit matriciel ; `axis` et `reshape` servent à chaque formule.
- **Ch. 2 (statistiques)** : moyenne, variance, bootstrap avec `rng` ; `groupby` pour comparer les espèces ; histogrammes.
- **Ch. 3 et 6** : `count_values` et `argmax` de mylearn pour les matrices de confusion et les distributions de caractères ; `Counter` sur les textes de Holmes et Verne.
- **Ch. 8 et 15** : l'API `fit` / `predict` (héritage, `fit` qui renvoie `self`, attributs appris en `_`) ; `**kwargs` des estimateurs.
- **Ch. 12** : valeurs manquantes, encodage `one_hot`, standardisation par broadcasting ; fuites de données.
- **Ch. 13** : récursivité et `@dataclass` pour l'arbre de décision.
- **Ch. 18** : méthodes spéciales (`__add__`, `__radd__`, `__mul__`) et fermetures pour l'autodifférentiation ; `iterate_minibatches` pour l'entraînement.
- **Ch. 20 et suivants** : générateurs et `__len__` / `__getitem__` pour `Dataset` et `DataLoader` ; images `(N, C, H, W)`.
- **B7** : git, tests, configuration (`@dataclass`, JSON), reproductibilité.

## Guide de lecture et de travail

Ce chapitre n'a pas de lecture dans le livre : la fiche est le cours. Pour chaque section, dans l'ordre :

1. **Lis** la section, et tape les exemples dans une cellule (sans les `>>>`) : modifie-les, casse-les, regarde les messages d'erreur.
2. Fais le **quiz** 🧠 correspondant (dans `02_exercices.md`), sans regarder la fiche.
3. Fais les exercices **papier** ✏️ dans `mon_travail/ch00a_python/06_mes_reponses.md`, puis vérifie-les dans la partie 0 du notebook (`wb.check`).
4. Fais les exercices du **notebook** de la section.
5. En fin de journée : 10 minutes de **flashcards**, et une ligne dans ton journal.

**Parcours rapide.** Il lit toute la fiche et fait l'essentiel des exercices : ce chapitre est le socle de tous les autres. Il laisse de côté les quiz Q7 à Q9, les exercices papier de « traçage » (0A.1 à 0A.4, 0A.6), l'explication 🗣️ 0A.9, les exercices git 0A.11 et 0A.12, quelques variantes (0A.15, 0A.17, 0A.19, 0A.29), une partie du Python intermédiaire et avancé (0A.37 à 0A.43, 0A.45, 0A.46, 0A.48, 0A.50), une partie de NumPy, pandas et matplotlib avancés (0A.55, 0A.56, 0A.58 à 0A.60), trois fonctions de mylearn (0A.63 à 0A.65) et le défi 0A.67 ; la liste exacte est dans `docs/PARCOURS.md`.

**Si tu bloques** : règle des 15 minutes (indice 1, puis indice 2…, dans `04_indices.md`). La documentation officielle est ton alliée : prends l'habitude d'y chercher le nom d'une fonction.

## Pour aller plus loin

- [Le tutoriel Python officiel, en français](https://docs.python.org/fr/3.13/tutorial/index.html) : la référence pour tout ce qui est « Python pur » (sections 100.2 à 100.7).
- [NumPy : the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) : le guide officiel pour débuter avec NumPy (en anglais).
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) : le tour rapide officiel de pandas (en anglais ; la documentation en ligne décrit pandas 3 : le texte y est de type `str` et non `object`, et Copy-on-Write y est actif (100.9) ; la documentation de la 2.2 du workbook : https://pandas.pydata.org/pandas-docs/version/2.2/).
- [Pro Git, en français](https://git-scm.com/book/fr/v2), chapitres 1 à 3 : les bases de git et des branches.
- Pour comprendre la place de NumPy : C. R. Harris et coll., « [Array programming with NumPy](https://www.nature.com/articles/s41586-020-2649-2) », *Nature*, 2020 (en accès libre) ; et, pour l'histoire de pandas, W. McKinney, « Data Structures for Statistical Computing in Python », *Proceedings of the 9th Python in Science Conference*, 2010.
