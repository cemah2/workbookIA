# 0A · Python, notebooks et outils — exercices papier, quiz et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch00a_python/06_mes_reponses.md` (créée par `python tools/start_chapter.py 0A`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses des exercices ✏️ se vérifient dans la **partie 0** du notebook (`wb.check`) ; les quiz, les exercices 🛠️ et 🗣️ et les questions d'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.

**Sommaire** : [🧠 Quiz](#quiz) · [✏️ Papier-crayon](#papier) · [🗣️🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, puis vérifie dans `05_solutions.md`.

### 0A.Q1 — Cellules, noyau, « Run all » et terminal : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §100.1 · parcours R*

Vrai ou faux ?
1. Dans un notebook, c'est l'ordre des cellules sur la page qui détermine la valeur des variables.
2. Supprimer une cellule supprime aussi les variables qu'elle avait créées.
3. « Redémarrer et tout exécuter » (*Restart and run all*) vérifie qu'un notebook tourne de haut en bas à partir d'une mémoire vide.
4. Dans un terminal, `cd ..` remonte au dossier parent.
5. `data/penguins.csv` est un chemin absolu.

### 0A.Q2 — Le workbook : où j'écris, comment je vérifie 🧠 ⏱️ 3 min
*Fiche §100.1.2, §100.11.5 · parcours R*

1. Dans quel dossier écris-tu ton code et tes réponses : `chapitres/`, `mon_travail/` ou `solutions/` ?
2. `wb.check` affiche ⏳ : qu'est-ce que cela veut dire ?
3. À quoi sert le bloc `with wb.attempt("0A.20"):` ?
4. Un test de mylearn compare ta fonction à quoi ?
5. `python tools/start_chapter.py 0A` peut-il écraser un fichier que tu as déjà modifié ?

### 0A.Q3 — Que vaut cette expression ? Types et opérateurs 🧠 ⏱️ 3 min
*Fiche §100.2 · parcours R, C*

Donne la valeur **et** le type de chaque expression.
1. `9 / 3`
2. `9 // 4 * 4 + 9 % 4`
3. `"ab" * 3`
4. `int("12") + float("0.5")`
5. `3 == 3.0`

### 0A.Q4 — Liste, tuple, dict, set ou Counter ? 🧠 ⏱️ 3 min
*Fiche §100.3 · parcours R, C*

Quelle structure choisis-tu pour ranger…
1. les 344 masses des manchots, dans l'ordre du fichier, en ajoutant les nouvelles mesures au fur et à mesure ?
2. la forme d'un tableau, `344` lignes et `8` colonnes ?
3. le nom latin de chaque espèce, à retrouver à partir de son nom court ?
4. les noms des îles visitées, chacun une seule fois ?
5. le nombre de manchots observés sur chaque île ?

### 0A.Q5 — Conditions, boucles, compréhensions : qu'affiche ce code ? 🧠 ⏱️ 3 min
*Fiche §100.4 · parcours R, C*

```python
# 1
for i in range(1, 8, 3):
    print(i, end=" ")

# 2
x = 5
if x > 3:
    print("a")
elif x > 1:
    print("b")
else:
    print("c")

# 3
print([c for c in "manchot" if c in "aeiou"])

# 4
for i, v in enumerate("ab"):
    print(i, v)

# 5
print(len([n for n in range(10) if n % 3 == 0]))
```

### 0A.Q6 — Paramètres, valeurs par défaut, `*args` et `**kwargs` 🧠 ⏱️ 3 min
*Fiche §100.5.1-2 · parcours R, C*

```python
def f(a, b=2, *args, c=3, **kwargs):
    return a, b, args, c, kwargs
```
Que renvoient ces appels ?
1. `f(1)`
2. `f(1, 5, 6, 7)`
3. `f(1, c=9, d=4)`
4. `f(b=1)`
5. Et que renvoie une fonction qui n'a pas de `return` ?

### 0A.Q7 — Lire une signature : annotations, `Callable`, `lambda`, fermeture 🧠 ⏱️ 3 min
*Fiche §100.5.3-5 · parcours C*

```python
def fit(X: np.ndarray, y: np.ndarray | None = None, *,
        callback: Callable[[int, float], None] | None = None) -> "Model":
    ...
```
1. L'argument `y` est-il obligatoire ?
2. Peut-on passer `callback` en troisième argument positionnel, `fit(X, y, my_function)` ?
3. Quelle sorte de fonction faut-il donner à `callback` ?
4. Selon quoi `sorted(words, key=len)` trie-t-il la liste ?
5. Avec `def make_adder(n): return lambda x: x + n`, que vaut `make_adder(3)(4)` ?

### 0A.Q8 — Quel outil de la bibliothèque standard pour quelle tâche ? 🧠 ⏱️ 3 min
*Fiche §100.6 · parcours C*

Quel module (ou quelle fonction) utilises-tu pour…
1. assembler le chemin `data/penguins.csv` de façon portable (Windows et Mac) ?
2. enregistrer un dictionnaire de résultats dans un fichier lisible par un humain ?
3. extraire toutes les dates `2007-11-11` d'un texte ?
4. obtenir toutes les paires possibles parmi 4 features ?
5. garder les 5 meilleurs scores d'une longue liste ?
6. compter combien de fois apparaît chaque espèce ?

### 0A.Q9 — Classes : `self`, méthodes spéciales, héritage, `yield` 🧠 ⏱️ 3 min
*Fiche §100.7 · parcours C*

1. Que désigne `self` dans une méthode ?
2. Quelle méthode Python appelle-t-il pour `len(obj)` ? pour `obj(x)` ? pour `3 * obj` quand `int` ne sait pas faire ?
3. Dans `class B(A):`, que fait `super().__init__()` écrit dans `B.__init__` ?
4. Peut-on parcourir deux fois le même générateur ?
5. Pourquoi la méthode `fit` des estimateurs scikit-learn renvoie-t-elle `self` ?

### 0A.Q10 — NumPy : `dtype`, `shape`, vue ou copie, images 🧠 ⏱️ 3 min
*Fiche §100.8.1-3, §100.8.8 · parcours R, C*

1. Quel est le `dtype` de `np.array([1, 2, 3.0])` ?
2. Quelle est la forme d'un lot de 64 images MNIST en niveaux de gris, chargé en NumPy comme dans la fiche (`wb.datasets.load_mnist`) ?
3. Pour un array `a`, `a[2:4]` est-il une vue ou une copie ? Et `a[a > 0]` ?
4. Quel `dtype` ont les pixels MNIST bruts, et entre quelles valeurs ?
5. `X.shape` vaut `(333, 4)` : combien d'exemples, combien de features ?

### 0A.Q11 — `axis`, broadcasting, `reshape` et graine 🧠 ⏱️ 3 min
*Fiche §100.8.4-7 · parcours R, M, C*

`X` a la forme `(333, 4)`.
1. Quelle est la forme de `X.mean(axis=0)` ?
2. Quelle est la forme de `X.max(axis=1)` ?
3. `X - X.mean(axis=0)` fonctionne-t-il ? Quelle est la forme du résultat ?
4. Pour `v` de forme `(12,)`, quelle est la forme de `v.reshape(-1, 1)` ?
5. Deux générateurs créés avec `np.random.default_rng(7)` donnent-ils les mêmes nombres ?

### 0A.Q12 — pandas, matplotlib, git, pytest : quelle commande pour quoi ? 🧠 ⏱️ 4 min
*Fiche §100.9-11 · parcours R, C*

Associe chaque besoin à une commande de la liste : `plt.subplots(2, 3)`, `pytest.raises(ValueError)`, `df.isna().sum()`, `git switch -c essai`, `df.groupby("species")["body_mass_g"].mean()`, `git diff`.
1. Compter les valeurs manquantes de chaque colonne.
2. Calculer la masse moyenne de chaque espèce.
3. Préparer une grille de 2 × 3 graphiques.
4. Voir les lignes modifiées depuis le dernier commit.
5. Créer une branche et s'y placer.
6. Vérifier dans un test qu'une fonction refuse une entrée invalide.

<a id="papier"></a>

## ✏️ Papier-crayon

Fais-les **sans ordinateur** : c'est le moyen le plus sûr de vérifier que tu comprends ce que fait Python. Écris chaque réponse dans ta copie de `06_mes_reponses.md`, puis reporte-la dans la **partie 0 du notebook**, qui la vérifie avec `wb.check` (par exemple `answer_0A_1a` pour la question a de 0A.1).

### Ex 0A.1 — Évaluer des expressions à la main : `//`, `%`, `**`, conversions ✏️ ★ ⏱️ 10 min
**Objectif :** prévoir le résultat des opérateurs arithmétiques et des conversions de type.
**Prérequis :** fiche §100.2.1 et §100.2.3 · **Parcours :** M, C

Donne la valeur de chaque expression, telle que Python l'afficherait.

| | Expression |
|---|---|
| a | `17 // 5` |
| b | `17 % 5` |
| c | `-17 // 5` |
| d | `(2 + 3) ** 2 - 10 // 3` |
| e | `int(7.9) + round(7.5)` |
| f | `10 / 4 * 2` |
| g | le **type** de `7 / 7` (réponds par son nom, par exemple `"int"`) |

### Ex 0A.2 — Indices et tranches à la main : listes, tuples, chaînes ✏️ ★ ⏱️ 10 min
**Objectif :** lire sans hésiter un indice négatif et une tranche `[début:fin:pas]`.
**Prérequis :** fiche §100.3.1-2 et §100.2.2 · **Parcours :** M, C

```python
masses = [3750, 3800, 3250, 3450, 3650, 3625]
name = "Chinstrap"
point = (39.1, 18.7, 181)
```

| | Expression |
|---|---|
| a | `masses[2]` |
| b | `masses[-2]` |
| c | `masses[1:4]` |
| d | `masses[::2]` |
| e | `name[:5]` |
| f | `name[-4:]` |
| g | `len(masses[2:])` |
| h | `point[2] - point[0]` (arrondi à 1 décimale) |

### Ex 0A.3 — Dérouler une boucle et une compréhension pas à pas ✏️ ★ ⏱️ 15 min
**Objectif :** exécuter du code « à la main », tour de boucle par tour de boucle, en notant les variables.
**Prérequis :** Ex 0A.1 · fiche §100.4 · **Parcours :** M, C

```python
values = [4, 7, 1, 8, 3, 6]
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
```

Conseil : fais un tableau avec une colonne par variable (`i`, `v`, `total`, `count`) et une ligne par tour.

| | Question |
|---|---|
| a | Que vaut `total` à la fin de la boucle `for` ? |
| b | Que vaut `count` ? |
| c | Que vaut `[v * 2 for v in values if v > 4]` ? |
| d | Que vaut `sum(v for v in values if v % 3 == 0)` ? |
| e | Que vaut `steps` à la fin de la boucle `while` ? |

### Ex 0A.4 — Un `groupby` à la main sur huit manchots ✏️ ★ ⏱️ 10 min
**Objectif :** comprendre ce que calculent un filtre et un `groupby` en les faisant à la main.
**Prérequis :** fiche §100.9.2 et §100.9.4 · **Fil rouge :** Penguins · **Parcours :** M, C

Le DataFrame `df` contient huit manchots :

| index | species | island | body_mass_g | sex |
|---|---|---|---|---|
| 0 | Adelie | Torgersen | 3750 | male |
| 1 | Adelie | Dream | 3400 | female |
| 2 | Gentoo | Biscoe | 5200 | male |
| 3 | Chinstrap | Dream | 3700 | female |
| 4 | Gentoo | Biscoe | 4650 | female |
| 5 | Adelie | Biscoe | 3900 | male |
| 6 | Chinstrap | Dream | 3950 | male |
| 7 | Gentoo | Biscoe | 5550 | male |

| | Expression | Format de la réponse |
|---|---|---|
| a | `df.groupby("species")["body_mass_g"].mean()["Gentoo"]` | arrondi à 1 décimale |
| b | `df.groupby("species")["body_mass_g"].mean()["Adelie"]` | arrondi à 1 décimale |
| c | `(df["island"] == "Dream").sum()` | entier |
| d | `df.groupby("island")["body_mass_g"].max()["Dream"]` | entier |
| e | `df[df["sex"] == "male"]["species"].value_counts()["Gentoo"]` | entier |
| f | `df.groupby("species")["body_mass_g"].mean().idxmin()` (l'espèce la plus légère en moyenne) | nom de l'espèce |

### Ex 0A.5 — Mini-lots : combien de lots, de quelle taille, combien de mises à jour ? ✏️ ★ ⏱️ 10 min
**Objectif :** relier la taille d'un dataset, la taille des mini-lots et le nombre de mises à jour.
**Prérequis :** fiche §100.8.7 (🧮 division euclidienne, §100.2.1) · **Fil rouge :** Penguins · **Parcours :** R, M

Après suppression des lignes incomplètes, il reste **333** manchots. On les découpe en mini-lots de **64**.

| | Question |
|---|---|
| a | Combien de mini-lots par epoch (le dernier lot, incomplet, est gardé) ? |
| b | Combien d'exemples contient le dernier lot ? |
| c | Combien de mini-lots par epoch avec `drop_last=True` ? |
| d | Combien de mises à jour des poids en 20 epochs, sans `drop_last` ? |
| e | Avec des lots de taille 1, combien de mises à jour par epoch ? |
| f | Avec un seul lot contenant tout le dataset, combien de mises à jour en 20 epochs ? |

### Ex 0A.6 — Portée, valeurs par défaut et arguments nommés : qui vaut quoi ? ✏️ ★★ ⏱️ 15 min
**Objectif :** prévoir les valeurs des paramètres à chaque appel et distinguer variable locale et variable globale.
**Prérequis :** Ex 0A.3 · fiche §100.5.1-2 · **Parcours :** M, C

```python
rate = 10


def price(quantity, unit=2, *, discount=0):
    rate = 5
    return quantity * unit * rate - discount


a = price(3)
b = price(3, 4)
c = price(2, discount=7)
d = price(unit=1, quantity=4)
e = rate
```

| | Question |
|---|---|
| a à e | Que valent `a`, `b`, `c`, `d` et `e` ? |
| f | Que se passe-t-il pour `price(1, 2, 3)` ? Réponds par le **nom** de l'erreur levée (par exemple `"ValueError"`), ou par la valeur si l'appel réussit. |

### Ex 0A.7 — Formes NumPy à la main : indexation, réductions, `reshape` ✏️ ★★ ⏱️ 15 min
**Objectif :** prévoir la forme de n'importe quel résultat avant de l'exécuter : le réflexe qui évite la moitié des bugs.
**Prérequis :** Ex 0A.2 · fiche §100.8.2, §100.8.5, §100.8.6 · **Parcours :** R, M, C

```python
A = np.arange(24).reshape(4, 6)
B = np.zeros((3, 28, 28))
```

Donne la **forme** (un tuple, par exemple `(4, 6)`) de chaque expression, sauf pour la question i.

| | Expression |
|---|---|
| a | `A[1:3]` |
| b | `A[:, 2]` |
| c | `A[:, 2:3]` |
| d | `A.sum(axis=0)` |
| e | `A.mean(axis=1, keepdims=True)` |
| f | `A.reshape(2, -1, 3)` |
| g | `B.reshape(len(B), -1)` |
| h | `A[A > 20]` |
| i | la **valeur** de `A[2, 3]` |

### Ex 0A.8 — Broadcasting : compatibles ou non, et quelle forme ? ✏️ ★★ ⏱️ 15 min
**Objectif :** appliquer la règle du broadcasting (comparer les formes de droite à gauche).
**Prérequis :** Ex 0A.7 · fiche §100.8.4 · **Parcours :** R, M, C

Pour chaque opération entre un tableau de forme `A` et un tableau de forme `B`, donne la forme du résultat, ou `erreur` si les formes sont incompatibles. Dans le notebook, réponds par une **chaîne** : `"(5, 3)"` ou `"erreur"`.

| | Forme de `A` | Forme de `B` | Opération |
|---|---|---|---|
| a | `(5, 3)` | `(3,)` | `A + B` |
| b | `(5, 3)` | `(5,)` | `A + B` |
| c | `(5, 1)` | `(1, 4)` | `A * B` |
| d | `(2, 1, 3)` | `(4, 1)` | `A - B` |
| e | `(64, 28, 28)` | `(28, 28)` | `A - B` |
| f | `(10,)` | `(10, 1)` | `A - B` |
| g | `(3, 4)` | `(4, 3)` | `A * B` |

<a id="reflexion"></a>

## 🗣️ 🛠️ Réflexion et outils

### Ex 0A.9 — Liste Python ou array NumPy : l'expliquer en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer simplement la différence entre une liste et un array, et quand utiliser chacun.
**Prérequis :** fiche §100.3.1 et §100.8.3 · **Parcours :** complet

Une amie qui débute te demande : « Pourquoi tout le monde utilise NumPy, alors que Python a déjà des listes ? » Réponds-lui **en cinq lignes au plus**, sans jargon inutile, avec un exemple concret sur les masses des 344 manchots. Ta réponse doit contenir au moins trois des idées suivantes : un seul type par array ; calcul vectorisé (sans boucle) ; vitesse ; mémoire ; les listes restent utiles pour des éléments de types différents ou une taille qui change souvent.

### Ex 0A.10 — Premier commit propre depuis le terminal 🛠️ ★★ ⏱️ 20 min
**Objectif :** enregistrer ton travail dans git avec un message clair, et le pousser sur GitHub.
**Prérequis :** fiche §100.1.3 et §100.11.1 · **Parcours :** R, C

À faire **dans un terminal sur ton ordinateur**, à la racine du dépôt (sur Colab, suis la section 5 « Sauvegarder ton travail sur GitHub » de `00_setup/COLAB.md`, qui passe par Python et un jeton secret).

1. Présente-toi à git une fois pour toutes (nom et e-mail, voir la fiche §100.11.1).
2. Écris ta réponse à 0A.9 dans `mon_travail/ch00a_python/06_mes_reponses.md` et enregistre le fichier.
3. Tape `git status` : quel fichier apparaît, et dans quelle couleur ?
4. Ajoute **seulement** ce fichier, puis refais `git status` : qu'est-ce qui a changé ?
5. Commite avec un message qui dit ce que fait le commit (pas « update »).
6. Tape `git log --oneline -3`, puis `git push`.

**À rendre** (dans `06_mes_reponses.md`) : la sortie de `git log --oneline -3` et, en une phrase, la différence entre `git add` et `git commit`.

### Ex 0A.11 — `.gitignore` : ce qui ne doit jamais entrer dans le dépôt 🛠️ ★★ ⏱️ 15 min
**Objectif :** décider ce qui mérite d'être versionné, et le faire respecter avec un `.gitignore`.
**Prérequis :** Ex 0A.10 · fiche §100.11.1 · **Parcours :** C

Voici des fichiers apparus dans `mon_travail/` au fil de tes séances :

```text
mon_travail/ch00a_python/03_notebook.ipynb
mon_travail/ch00a_python/.ipynb_checkpoints/03_notebook-checkpoint.ipynb
mon_travail/mylearn/utils.py
mon_travail/mylearn/__pycache__/utils.cpython-313.pyc
mon_travail/notes/idees.md
mon_travail/.env                       (contains OPENAI_API_KEY=sk-...)
mon_travail/models/cnn_cifar.pt        (180 MB)
mon_travail/data_perso/releve_bancaire.csv
```

1. Pour chaque fichier : versionner ou ignorer ? Justifie en quelques mots.
2. Écris le contenu d'un fichier `mon_travail/.gitignore` (4 à 6 lignes de motifs) qui ignore ce qui doit l'être.
3. Sur ton ordinateur : crée ce fichier, puis vérifie avec `git check-ignore -v mon_travail/.env` qu'il est bien pris en compte. Que se serait-il passé si `.env` avait déjà été commité avant ?

### Ex 0A.12 — Une branche pour essayer sans risque (aperçu) 🛠️ ★★ ⏱️ 15 min
**Objectif :** créer une branche, y committer, revenir sur `main`, puis fusionner ou abandonner l'essai.
**Prérequis :** Ex 0A.10 · fiche §100.11.2 · **Parcours :** C

Dans un terminal, à la racine du dépôt :
1. Crée une branche `essai-notes` et place-toi dessus.
2. Ajoute une ligne à `mon_travail/notes/idees.md` (crée le fichier s'il n'existe pas), puis commite.
3. Reviens sur `main` : la ligne est-elle encore dans le fichier ? Pourquoi ?
4. Fusionne `essai-notes` dans `main`, vérifie avec `git log --oneline -3`, puis supprime la branche.
5. Question : quelle est la différence entre `git branch -d` et `git branch -D` ? Dans quel cas l'une refuse-t-elle de supprimer la branche ?

⚠️ Termine toujours sur `main` : c'est sur `main` que `git pull --rebase --autostash` récupère les nouveaux chapitres.

**À rendre** : la sortie de `git log --oneline -3` après la fusion, et ta réponse à la question 5.

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 0A.E1 — Data analyst, data scientist, ML engineer : qui fait quoi, avec quels outils ? 💼 ★★ ⏱️ 10 min
*Fiche §100.1.2, `annexes/metiers.md` · parcours R*

« Quelle différence faites-vous entre un data analyst, un data scientist et un ML engineer ? Lequel de ces postes visez-vous, et pourquoi ? »

### 0A.E2 — Liste, tuple, dictionnaire ou ensemble : comment choisir ? 💼 ★★ ⏱️ 10 min
*Fiche §100.3 · prérequis 0A.Q4 · parcours R*

« En Python, comment choisissez-vous entre une liste, un tuple, un dictionnaire et un ensemble ? Donnez un exemple de situation pour chacun. »

### 0A.E3 — Pourquoi vectoriser avec NumPy plutôt qu'écrire une boucle ? 💼 ★★ ⏱️ 10 min
*Fiche §100.8.3 · prérequis 0A.55 (notebook, prochaine session) · parcours R*

« Pourquoi dit-on qu'il faut vectoriser son code NumPy ? D'où vient le gain de vitesse, et y a-t-il des cas où une boucle reste acceptable ? »

### 0A.E4 — Ton notebook est-il reproductible ? Run all, graine, versions 💼 ★★ ⏱️ 10 min
*Fiche §100.1.1, §100.8.7 · prérequis 0A.13, 0A.31 · parcours R*

« Un collègue relance votre notebook et n'obtient pas les mêmes résultats que vous. Quelles peuvent être les causes, et comment rendre un notebook reproductible ? »

### 0A.E5 — Ton workflow git : commit, branche, pull request 💼 ★★ ⏱️ 10 min
*Fiche §100.11.1-2 · prérequis 0A.12 · parcours R*

« Décrivez votre façon de travailler avec git sur un projet en équipe, depuis la création d'une branche jusqu'à l'intégration de votre code. »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch00a_python/03_notebook.ipynb`). La partie 0 du notebook vérifie tes réponses aux exercices ✏️ ci-dessus.

| Partie | ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|---|
| A · Prise en main | 0A.13 | Ordre d'exécution des cellules : que vaut `x` ? | 🔮 | ★ | 10 |
| B · Python de base | 0A.14 | Nombres et f-strings : la fiche d'un manchot | 🔨 | ★ | 13 |
| | 0A.15 | Chaînes : nettoyer les noms d'espèces de Penguins brut | 🔨 | ★ | 13 |
| | 0A.16 | Listes : les nageoires de dix manchots | 🔨 | ★ | 13 |
| | 0A.17 | Tuples et déballage : renvoyer et échanger plusieurs valeurs | 🔨 | ★ | 13 |
| | 0A.18 | Dictionnaires : une fiche par espèce | 🔨 | ★ | 15 |
| | 0A.19 | Ensembles : quelles espèces sur quelles îles ? | 🔨 | ★ | 13 |
| | 0A.20 | Conditions : classer un manchot selon sa masse | 🔨 | ★ | 13 |
| | 0A.21 | Boucles : `for`, `range`, `enumerate`, `zip` et `while` | 🔨 | ★ | 15 |
| | 0A.22 | Compréhensions : filtrer et transformer en une ligne | 🔨 | ★ | 15 |
| C · Fonctions, modules, erreurs | 0A.23 | Tes premières fonctions : paramètres, valeurs par défaut, `return` | 🔨 | ★ | 15 |
| | 0A.24 | Importer des modules : `math`, `random`, `statistics` et `Counter` | 🔨 | ★ | 13 |
| | 0A.25 | Exceptions : lever une `ValueError` et la rattraper | 🔨 | ★ | 15 |
| | 0A.26 | Ton premier module mylearn : `mean` et son test | 🔨 | ★ | 15 |
| D · Premiers pas NumPy | 0A.27 | Premiers arrays NumPy : `dtype`, `shape`, `ndim` | 📦 | ★ | 13 |
| | 0A.28 | Indexation, tranches et masques booléens | 📦 | ★ | 15 |
| | 0A.29 | Vue ou copie : qui est modifié ? | 🔮 | ★ | 10 |
| | 0A.30 | Calcul vectorisé : unités, normalisation, fonctions universelles | 📦 | ★ | 15 |
| | 0A.31 | Aléatoire reproductible : `default_rng`, graine, `permutation`, `choice` | 📦 | ★ | 15 |
| E · Premiers pas pandas et matplotlib | 0A.32 | Premier contact avec Penguins : `read_csv`, `head`, `info`, `describe` | 📦 | ★ | 13 |
| | 0A.33 | Sélectionner : colonnes, `loc`, `iloc` et filtres | 📦 | ★ | 15 |
| | 0A.34 | Valeurs manquantes et doublons : `isna`, `dropna`, `fillna`, `duplicated` | 📦 | ★ | 15 |
| | 0A.35 | De pandas à NumPy : construire `X` et `y` | 📦 | ★ | 13 |
| | 0A.36 | Premiers graphiques : `plot`, `scatter`, `hist` | 📦 | ★ | 15 |
| F · Python intermédiaire et avancé | 0A.37 à 0A.50 | traceback, fichiers, `json` et `pickle`, `*args`/`**kwargs`, `lambda`, fermetures, récursivité, regex, `itertools`, classes, méthodes spéciales, héritage, générateurs | 🐛 🔨 🔮 | ★★ | |
| G · NumPy, pandas et matplotlib avancés | 0A.51 à 0A.60 | réductions par axe, broadcasting, `reshape`, images MNIST, vectorisation mesurée, bugs NumPy, `groupby`, filtres, figures à panneaux, lecture de graphique | 📦 🔬 🐛 📈 | ★★ | |
| H · Outils du pro et mylearn | 0A.61 à 0A.67 | doctest, pytest, `utils.count_values`, `argmax`, `one_hot`, `iterate_minibatches`, défi final | 🛠️ 🔨 🏆 | ★★ à ★★★ | |

Les parties F à H sont ajoutées à la prochaine session de génération : le détail de leurs exercices est dans `docs/SYLLABUS.md` (chapitre 0A).
