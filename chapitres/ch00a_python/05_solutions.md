# 0A · Python, notebooks et outils — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Le code complet des exercices du notebook est dans `05_solutions.ipynb`, exécuté.

**Sommaire** : [🧠 Quiz](#quiz) · [✏️ Papier-crayon](#papier) · [🗣️🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à H](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 0A.Q1 — Cellules, noyau, « Run all » et terminal
1. **Faux** : c'est l'ordre d'**exécution** qui compte ; le noyau garde l'état laissé par les cellules exécutées, dans l'ordre où tu les as lancées.
2. **Faux** : la variable reste dans la mémoire du noyau jusqu'au prochain redémarrage.
3. **Vrai** : c'est le test de reproductibilité.
4. **Vrai.**
5. **Faux** : c'est un chemin **relatif** (il part du dossier courant). Un chemin absolu commence à la racine du disque (`/` ou `C:\`).

### 0A.Q2 — Le workbook : où j'écris, comment je vérifie
1. `mon_travail/`. Les autres dossiers sont mis à jour par Claude.
2. Pas encore fait : la réponse vaut encore `...` (ou `None`), ou la fonction lève encore `NotImplementedError`.
3. Il transforme « pas encore écrit » (`NotImplementedError`, module mylearn absent) en ⏳, pour que « Run all » continue ; les vraies erreurs s'affichent normalement.
4. À un **oracle** : une bibliothèque de confiance (NumPy, scikit-learn, PyTorch) qui calcule la même chose.
5. Non, jamais : un fichier déjà présent est conservé (🔒).

### 0A.Q3 — Que vaut cette expression ?
1. `3.0`, un `float` : `/` donne toujours un `float`.
2. `9`, un `int` : `9 // 4 * 4` vaut 8 et `9 % 4` vaut 1 (c'est la division euclidienne : $9 = 4 \times 2 + 1$).
3. `'ababab'`, un `str` : `*` répète une chaîne.
4. `12.5`, un `float` : un `int` plus un `float` donne un `float`.
5. `True`, un `bool` : `3` et `3.0` sont égaux en valeur.

### 0A.Q4 — Liste, tuple, dict, set ou Counter ?
1. Une **liste** (ordonnée, on y ajoute avec `append`).
2. Un **tuple** `(344, 8)` : un enregistrement figé (c'est ce que renvoie `.shape`).
3. Un **dictionnaire** `{"Adelie": "Pygoscelis adeliae", ...}`.
4. Un **ensemble** : chaque île une seule fois, test d'appartenance instantané.
5. Un **`Counter`** (ou un dictionnaire île → nombre).

### 0A.Q5 — Qu'affiche ce code ?
1. `1 4 7` (de 1 à 8 exclu, de 3 en 3).
2. `a` : seul le premier bloc dont la condition est vraie s'exécute.
3. `['a', 'o']`.
4. `0 a` puis `1 b`.
5. `4` (0, 3, 6 et 9).

### 0A.Q6 — Paramètres, valeurs par défaut, `*args` et `**kwargs`
1. `(1, 2, (), 3, {})`.
2. `(1, 5, (6, 7), 3, {})` : 6 et 7 sont en trop, ils vont dans `args`.
3. `(1, 2, (), 9, {'d': 4})` : `c` est nommé, `d` n'est pas un paramètre, il va dans `kwargs`.
4. `TypeError` : `a` est obligatoire et n'a pas été donné.
5. `None`.

### 0A.Q7 — Lire une signature
1. Non : sa valeur par défaut est `None`.
2. Non : `callback` est après `*`, il est *keyword-only* ; il faut écrire `fit(X, y, callback=my_function)`.
3. Une fonction qui reçoit un `int` et un `float` (par exemple un numéro d'epoch et une loss) et ne renvoie rien.
4. Selon la **longueur** des mots : `key` reçoit chaque élément et renvoie la valeur à comparer.
5. `7` : `make_adder(3)` renvoie une fonction qui ajoute 3 (une fermeture qui se souvient de `n`).

### 0A.Q8 — Quel outil pour quelle tâche ?
1. `pathlib` (`Path("data") / "penguins.csv"`). 2. `json`. 3. `re` (`re.findall(r"\d{4}-\d{2}-\d{2}", texte)`). 4. `itertools.combinations(features, 2)`. 5. `heapq.nlargest(5, scores)` (ou `sorted(scores, reverse=True)[:5]`). 6. `collections.Counter`.

### 0A.Q9 — Classes
1. L'objet sur lequel la méthode est appelée : dans `pingu.mass_kg()`, `self` est `pingu`.
2. `__len__` ; `__call__` ; `__rmul__` (la version « objet à droite » de `__mul__`).
3. Il appelle le `__init__` de la classe mère `A`, pour que la partie « A » de l'objet soit correctement initialisée.
4. Non : une fois parcouru, un générateur est épuisé.
5. Pour pouvoir enchaîner les appels : `model.fit(X, y).predict(X_new)`.

### 0A.Q10 — NumPy
1. `float64` : un seul type par array ; l'entier est converti.
2. `(64, 28, 28)` : nombre d'images, hauteur, largeur (pas d'axe de canal en NumPy pour des niveaux de gris). PyTorch ajoutera un axe de canal : `(64, 1, 28, 28)`.
3. `a[2:4]` est une **vue** (elle partage les données) ; `a[a > 0]` est une **copie**.
4. `uint8`, entiers de 0 à 255.
5. 333 exemples (lignes), 4 features (colonnes).

### 0A.Q11 — `axis`, broadcasting, `reshape`, graine
1. `(4,)` : une moyenne par colonne. 2. `(333,)` : un maximum par ligne. 3. Oui : `(333, 4)` et `(4,)` sont compatibles (le vecteur des moyennes est soustrait à chaque ligne) ; résultat `(333, 4)`. 4. `(12, 1)`. 5. Oui, exactement les mêmes.

### 0A.Q12 — Quelle commande pour quoi ?
1. `df.isna().sum()` · 2. `df.groupby("species")["body_mass_g"].mean()` · 3. `plt.subplots(2, 3)` · 4. `git diff` · 5. `git switch -c essai` · 6. `pytest.raises(ValueError)`.

<a id="papier"></a>

## ✏️ Papier-crayon

### Ex 0A.1 — Évaluer des expressions à la main

| | Expression | Réponse | Pourquoi |
|---|---|---|---|
| a | `17 // 5` | `3` | $17 = 5 \times 3 + 2$ |
| b | `17 % 5` | `2` | le reste de la même division |
| c | `-17 // 5` | `-4` | `//` arrondit **vers le bas** : $-17 / 5 = -3{,}4$, arrondi vers le bas : $-4$ (et le reste vaut $-17 - 5 \times (-4) = 3$) |
| d | `(2 + 3) ** 2 - 10 // 3` | `22` | $25 - 3$ : parenthèses, puis puissance, puis `//`, puis soustraction |
| e | `int(7.9) + round(7.5)` | `15` | `int` tronque (7) ; `round(7.5)` arrondit au pair le plus proche (8) |
| f | `10 / 4 * 2` | `5.0` | même priorité, de gauche à droite : $(10 / 4) \times 2 = 2{,}5 \times 2$ ; `/` donne un `float` |
| g | type de `7 / 7` | `"float"` | `/` donne toujours un `float` (`1.0`) |

**Erreurs fréquentes** : `-3` en c (penser que `//` tronque vers zéro, comme `int`) ; `16` en e (croire que `int` arrondit) ; `14` en e (croire que `round(7.5)` donne 7) ; `1.25` en f (calculer $10 / (4 \times 2)$).
**Variante** : que valent `-17 % 5` et `divmod(17, 5)` ? (Réponses : `3` et `(3, 2)`.)

### Ex 0A.2 — Indices et tranches à la main

| | Expression | Réponse |
|---|---|---|
| a | `masses[2]` | `3250` (le **3ᵉ** élément : les indices commencent à 0) |
| b | `masses[-2]` | `3650` (l'avant-dernier) |
| c | `masses[1:4]` | `[3800, 3250, 3450]` (indices 1, 2, 3 : la fin 4 est exclue) |
| d | `masses[::2]` | `[3750, 3250, 3650]` (un sur deux, à partir du premier) |
| e | `name[:5]` | `"Chins"` |
| f | `name[-4:]` | `"trap"` |
| g | `len(masses[2:])` | `4` (6 éléments moins les 2 premiers) |
| h | `point[2] - point[0]` | `141.9` ($181 - 39{,}1$) |

**Erreurs fréquentes** : compter à partir de 1 (a) ; inclure la fin de la tranche (c : 4 éléments au lieu de 3).
**Variante** : `masses[::-1][:2]` donne les deux derniers éléments dans l'ordre inverse : `[3625, 3650]`.

### Ex 0A.3 — Dérouler une boucle et une compréhension

Le tableau de suivi de la boucle `for` :

| `i` | `v` | pair ? | `v > 5` ? | `total` | `count` |
|---|---|---|---|---|---|
| 0 | 4 | oui | | 0 + 4 × 0 = 0 | 0 |
| 1 | 7 | non | oui | 0 | 1 |
| 2 | 1 | non | non | 0 | 1 |
| 3 | 8 | oui | | 0 + 8 × 3 = 24 | 1 |
| 4 | 3 | non | non | 24 | 1 |
| 5 | 6 | oui | (non testé : `elif`) | 24 + 6 × 5 = 54 | 1 |

a) `54` · b) `1` (6 est pair : il passe par le `if`, jamais par le `elif`) · c) `[14, 16, 12]` (7, 8 et 6 dépassent 4) · d) `9` (3 et 6 sont divisibles par 3) · e) `4` : $100 \to 33 \to 11 \to 3 \to 1$.

**Erreurs fréquentes** : compter 6 dans `count` (oublier que `elif` n'est testé que si le `if` a échoué) ; multiplier par la position « humaine » (1, 2, 3…) au lieu de l'indice.
**Variante** : que vaut `count` si l'on remplace `elif` par un second `if` ? (Réponse : `3` : 7, 8 et 6 dépassent 5, et les pairs sont maintenant testés aussi.)

### Ex 0A.4 — Un `groupby` à la main

a) Gentoo : $(5200 + 4650 + 5550) / 3 = 15\,400 / 3 \approx$ **5133,3** · b) Adelie : $(3750 + 3400 + 3900) / 3 = 11\,050 / 3 \approx$ **3683,3** · c) **3** manchots sur Dream (lignes 1, 3, 6) · d) **3950** (le plus lourd de Dream, ligne 6) · e) **2** Gentoo mâles (lignes 2 et 7) · f) **Adelie** : les moyennes sont Adelie 3683,3, Chinstrap $(3700 + 3950) / 2 = 3825$, Gentoo 5133,3.

**Démarche** : `groupby("species")` sépare les lignes en trois groupes, `["body_mass_g"]` garde la colonne, `.mean()` calcule une moyenne par groupe, `["Gentoo"]` lit la valeur de l'index `Gentoo`. Un filtre (`df[df["sex"] == "male"]`) garde les lignes où la condition est vraie, **avant** le comptage.
**Variante** : que donne `df.groupby("island")["species"].nunique()` ? (Biscoe 2, Dream 2, Torgersen 1.)

### Ex 0A.5 — Mini-lots

a) $\lceil 333 / 64 \rceil =$ **6** lots · b) $333 = 64 \times 5 + 13$ : le dernier lot contient **13** exemples · c) avec `drop_last=True`, **5** lots (les 13 exemples restants sont ignorés à cette epoch ; ils changent à chaque epoch si les données sont mélangées) · d) $20 \times 6 =$ **120** mises à jour · e) **333** (une mise à jour par exemple : c'est la descente de gradient « stochastique ») · f) **20** (une mise à jour par epoch : c'est le *full batch*).

**Erreurs fréquentes** : 5 en a (oublier le lot incomplet) ; 100 en d (compter avec `drop_last`).
**À retenir** : plus les lots sont petits, plus il y a de mises à jour par epoch, chacune plus « bruitée » (ch. 19).

### Ex 0A.6 — Portée, valeurs par défaut, arguments nommés

Dans `price`, `rate = 5` crée une variable **locale** : c'est elle qui sert au calcul, et la variable globale `rate` (10) n'est jamais modifiée.

a) `price(3)` : $3 \times 2 \times 5 - 0 =$ **30** · b) `price(3, 4)` : $3 \times 4 \times 5 =$ **60** · c) `price(2, discount=7)` : $2 \times 2 \times 5 - 7 =$ **13** · d) `price(unit=1, quantity=4)` : **20** (les arguments nommés peuvent être dans n'importe quel ordre) · e) **10** · f) **`TypeError`** : `discount` est *keyword-only* (il est après `*`), donc `price` n'accepte que 2 arguments positionnels (`price() takes from 1 to 2 positional arguments but 3 were given`).

**Erreurs fréquentes** : utiliser `rate = 10` dans les calculs (a = 60) ; croire que `e` vaut 5.
**Variante** : écris `price(1, 2, discount=3)` : **7**.

### Ex 0A.7 — Formes NumPy à la main

`A` est `[[0..5], [6..11], [12..17], [18..23]]`, de forme `(4, 6)`.

| | Expression | Forme | Pourquoi |
|---|---|---|---|
| a | `A[1:3]` | `(2, 6)` | 2 lignes, toutes les colonnes |
| b | `A[:, 2]` | `(4,)` | un **indice** entier retire l'axe des colonnes |
| c | `A[:, 2:3]` | `(4, 1)` | une **tranche** garde l'axe, même d'une seule colonne |
| d | `A.sum(axis=0)` | `(6,)` | l'axe 0 (les lignes) disparaît : une somme par colonne |
| e | `A.mean(axis=1, keepdims=True)` | `(4, 1)` | une moyenne par ligne, l'axe gardé avec une taille 1 |
| f | `A.reshape(2, -1, 3)` | `(2, 4, 3)` | $24 / (2 \times 3) = 4$ |
| g | `B.reshape(len(B), -1)` | `(3, 784)` | chaque image 28 × 28 aplatie |
| h | `A[A > 20]` | `(3,)` | un masque renvoie un tableau 1D des éléments retenus : 21, 22, 23 |
| i | `A[2, 3]` | valeur `15` | ligne 2, colonne 3 : $2 \times 6 + 3$ |

**Erreur fréquente** : confondre b et c ; c'est l'origine de beaucoup de bugs de formes `(n,)` contre `(n, 1)` (fiche §100.8.4).

### Ex 0A.8 — Broadcasting

On aligne les formes **à droite** et on compare axe par axe (égales, ou l'une vaut 1 ; un axe absent compte comme 1).

| | `A` | `B` | Résultat | Détail |
|---|---|---|---|---|
| a | `(5, 3)` | `(3,)` | `(5, 3)` | 3 = 3 ; l'axe absent de `B` compte comme 1 |
| b | `(5, 3)` | `(5,)` | **erreur** | 3 ≠ 5 et aucun ne vaut 1 |
| c | `(5, 1)` | `(1, 4)` | `(5, 4)` | chaque 1 est étiré : toutes les combinaisons |
| d | `(2, 1, 3)` | `(4, 1)` | `(2, 4, 3)` | droite : 3 et 1 → 3 ; milieu : 1 et 4 → 4 ; gauche : 2 et (absent) → 2 |
| e | `(64, 28, 28)` | `(28, 28)` | `(64, 28, 28)` | la même image `B` est soustraite des 64 images (par exemple l'image moyenne) |
| f | `(10,)` | `(10, 1)` | `(10, 10)` | `(10,)` se lit `(1, 10)` : un tableau de toutes les différences, sans erreur |
| g | `(3, 4)` | `(4, 3)` | **erreur** | à droite : 4 ≠ 3 (ce n'est pas un produit matriciel, qui s'écrit `@`) |

**Piège à retenir** : f ne lève **pas** d'erreur, ce qui en fait un bug silencieux classique.

<a id="reflexion"></a>

## 🗣️ 🛠️ Réflexion et outils

### Ex 0A.9 — Liste Python ou array NumPy (réponse modèle)

> Une liste Python peut contenir n'importe quoi (des nombres, du texte, d'autres listes) : chaque élément est un objet séparé en mémoire. Un array NumPy ne contient que des nombres d'**un seul type**, rangés côte à côte : il prend beaucoup moins de place. Surtout, on calcule sur **tout le tableau d'un coup** : `masses / 1000` convertit les 344 masses en kg sans écrire de boucle, et c'est 50 à 100 fois plus rapide, car la boucle est faite en langage compilé. On garde les listes pour des éléments de types différents ou une collection qui grandit au fur et à mesure (`append`) ; dès qu'on calcule, on passe à NumPy.

**Critères** : un seul type ; calcul vectorisé ; vitesse ; mémoire ; quand garder une liste. Trois idées suffisent, sans jargon.

### Ex 0A.10 — Premier commit propre

```bash
git config --global user.name "Ton Nom"
git config --global user.email "ton.email@exemple.fr"
git status                                                  # in red, under "Untracked files": mon_travail/ch00a_python/
git add mon_travail/ch00a_python/06_mes_reponses.md
git status                                                  # in green, "new file": staged for the next commit
git commit -m "0A: answer exercise 0A.9 (list vs NumPy array)"
git log --oneline -3
git push
```

**Couleurs** : au premier `git status`, le dossier `mon_travail/ch00a_python/` apparaît **en rouge** sous *Untracked files* : git ne suit pas encore ces fichiers (`start_chapter.py` vient de les créer). Après `git add`, le fichier passe **en vert** sous *Changes to be committed* (`new file`). Pour un fichier déjà commité puis modifié, `git status` le montre en rouge sous *Changes not staged for commit* (`modified`).
`git log --oneline -3` affiche trois lignes du type `a1b2c3d 0A: answer exercise 0A.9 (list vs NumPy array)` : l'identifiant court du commit et son message, le plus récent en haut.
**`git add` contre `git commit`** : `add` choisit **ce qui entrera** dans le prochain commit (la zone de préparation, *staging area*) ; `commit` **enregistre** la photo de ces fichiers dans l'historique, avec un message.
**Erreurs fréquentes** : `git add .` à la racine, qui embarque tout ce qui traîne ; un message vide de sens (« update ») ; oublier `git push` (le commit n'existe alors que sur ton ordinateur) ; `Author identity unknown` : l'étape 1 a été oubliée.

### Ex 0A.11 — `.gitignore`

| Fichier | Décision | Pourquoi |
|---|---|---|
| `03_notebook.ipynb` | versionner | c'est ton travail |
| `.ipynb_checkpoints/…` | ignorer | sauvegardes automatiques de Jupyter |
| `mylearn/utils.py` | versionner | ta librairie : ton portfolio |
| `mylearn/__pycache__/…` | ignorer | cache généré automatiquement par Python |
| `notes/idees.md` | versionner | tes notes (sauf si elles sont privées) |
| `.env` | **ignorer absolument** | contient une clé d'API secrète |
| `models/cnn_cifar.pt` | ignorer | 180 Mo : trop lourd pour git (GitHub refuse les fichiers de plus de 100 Mo), et régénérable |
| `data_perso/releve_bancaire.csv` | **ignorer absolument** | données personnelles |

```text
.ipynb_checkpoints/
__pycache__/
.env
*.pt
data_perso/
```

`git check-ignore -v mon_travail/.env` affiche la ligne du `.gitignore` qui ignore le fichier (rien s'il n'est pas ignoré).
**Si `.env` avait déjà été commité** : le `.gitignore` n'y changerait rien, car git continue de suivre un fichier déjà versionné. Il faudrait `git rm --cached mon_travail/.env`, puis commiter, et surtout **révoquer la clé** : elle reste lisible dans l'historique (et sur GitHub si elle a été poussée). Toujours considérer comme compromis un secret qui a été poussé.

### Ex 0A.12 — Une branche pour essayer

```bash
git switch -c essai-notes
echo "Idée : comparer les trois espèces sur la masse" >> mon_travail/notes/idees.md
git add mon_travail/notes/idees.md
git commit -m "notes: add an idea to compare species"
git switch main                  # the line is gone: it only exists on the branch
git merge essai-notes            # fast-forward: main now has the commit
git log --oneline -3
git branch -d essai-notes
```

3. Sur `main`, la ligne a **disparu** du fichier (et même le fichier entier, si tu l'as créé sur la branche) : `git switch` remet les fichiers dans l'état de la branche choisie, et le commit n'existe que sur `essai-notes`. Tout revient après la fusion.
5. `git branch -d` supprime une branche **déjà fusionnée** et refuse sinon (pour ne pas perdre de commits) ; `git branch -D` force la suppression, même si ses commits ne sont fusionnés nulle part.

<a id="entretien"></a>

## 💼 Entretien

### 0A.E1 — Data analyst, data scientist, ML engineer

**Réponse modèle en 60 secondes** : « Les trois métiers travaillent avec les données, mais pas au même endroit de la chaîne. Le data analyst répond à des questions métier : il explore, résume et visualise, avec SQL, pandas et des tableaux de bord ; son livrable est une analyse ou une recommandation. Le data scientist construit et évalue des modèles prédictifs : préparation des données, choix du modèle, évaluation honnête, avec scikit-learn, du gradient boosting et parfois du deep learning. Le ML engineer met ces modèles en production et les maintient : code testé, pipelines d'entraînement, API, suivi des performances. Je vise [le poste], parce que [ce qui m'attire : un problème, une compétence déjà acquise dans mon métier précédent] ; je m'y prépare avec [un projet concret de ton portfolio]. »
**Relances possibles** : « Et l'AI engineer ? » (construire des applications autour de modèles pré-entraînés : LLM, RAG, évaluation) · « Quelle compétence vous manque encore, et comment la travaillez-vous ? » · « Qu'apporte votre ancien métier ? »

### 0A.E2 — Liste, tuple, dictionnaire ou ensemble

**Réponse modèle en 60 secondes** : « Je choisis selon ce que je dois faire des données. Une liste pour une suite ordonnée qui évolue, comme des mesures que j'ajoute au fil de l'eau. Un tuple pour un enregistrement figé ou plusieurs valeurs renvoyées par une fonction, comme la forme d'un tableau : il est immuable, donc il peut servir de clé de dictionnaire. Un dictionnaire pour retrouver une valeur à partir d'une clé, comme une configuration ou le nombre d'individus par espèce : l'accès est en temps constant. Un ensemble pour dédoublonner ou tester l'appartenance rapidement, et pour les intersections. Et dès qu'il s'agit de calculer sur beaucoup de nombres, je passe à NumPy ou pandas. »
**Relances possibles** : « Pourquoi `x in liste` est-il lent sur une grande liste et `x in ensemble` rapide ? » (parcours de toute la liste contre accès par hachage) · « Pourquoi une liste ne peut-elle pas être une clé de dictionnaire ? » (elle est modifiable, donc non hachable).

### 0A.E3 — Pourquoi vectoriser avec NumPy ?

**Réponse modèle en 60 secondes** : « Vectoriser, c'est écrire une opération sur un tableau entier plutôt qu'une boucle sur ses éléments. Le gain vient de deux choses : la boucle est exécutée dans du code compilé, sans interpréter une instruction Python à chaque élément, et les données sont rangées côte à côte en mémoire, dans un seul type, ce que le processeur traite très efficacement. En pratique, on gagne souvent un facteur 50 à 100, et le code est plus court et plus proche de la formule mathématique. Une boucle reste acceptable quand les éléments dépendent les uns des autres de façon séquentielle, pour de petits volumes, ou pour une première version qu'on vectorise ensuite en vérifiant que les résultats sont identiques. »
**Relances possibles** : « Qu'est-ce que le broadcasting ? » · « Comment mesurez-vous le gain ? » (`time.perf_counter` ou `%timeit`, sur des données de taille réaliste ; voir 0A.55) · « Et sur GPU ? » (même principe, à plus grande échelle, avec PyTorch).

### 0A.E4 — Ton notebook est-il reproductible ?

**Réponse modèle en 60 secondes** : « Je vérifie trois choses. D'abord l'état : un notebook peut marcher parce que ses cellules ont été exécutées dans le désordre, ou grâce à une variable venue d'une cellule supprimée ; je fais toujours *Restart and run all* avant de le partager. Ensuite le hasard : je fixe les graines (NumPy `default_rng(42)`, `random`, PyTorch) et je passe le générateur aux fonctions. Enfin l'environnement : je note les versions des bibliothèques dans un `requirements.txt` et les données dans leur version exacte, car un changement de version peut modifier les résultats. En plus, j'évite les chemins absolus propres à ma machine et je versionne le tout avec git. Sur GPU, certaines opérations restent légèrement non déterministes : on l'accepte ou on active le mode déterministe, plus lent. »
**Relances possibles** : « Deux exécutions donnent 0,912 et 0,907 : est-ce grave ? » (variabilité liée à la graine : on répète l'expérience avec plusieurs graines et on donne un écart-type ; ch. 8) · « Que mettez-vous dans le README pour qu'un collègue relance votre travail ? »

### 0A.E5 — Ton workflow git en équipe

**Réponse modèle en 60 secondes** : « Je pars toujours d'une branche `main` à jour (`git pull`). Pour chaque tâche, je crée une branche au nom explicite (`git switch -c fix-missing-values`). J'y fais des commits petits et cohérents, avec des messages qui disent ce que fait le commit ; les données, les secrets et les fichiers générés restent hors du dépôt grâce au `.gitignore`. Je pousse ma branche et j'ouvre une pull request : les tests automatiques tournent, un ou deux collègues relisent, je corrige selon leurs remarques. Une fois la pull request acceptée, elle est fusionnée dans `main` et je supprime la branche. En cas de conflit, je rapatrie `main` dans ma branche (merge ou rebase) et je le résous en local. »
**Relances possibles** : « Merge ou rebase ? » · « Comment annuler un commit déjà poussé ? » (`git revert`, qui crée un commit inverse, plutôt que réécrire l'historique partagé) · « Qu'est-ce qu'une bonne revue de code ? »

<a id="notebook"></a>

## Notebook, parties A à H

Le code complet et exécuté est dans `05_solutions.ipynb`. Ici : les réponses, le *pourquoi*, les erreurs fréquentes et une variante.

### Ex 0A.13 — Ordre d'exécution des cellules 🔮
a) **22** : $x = 2 \to 2 \times 3 = 6 \to 6 + 1 = 7 \to 7 \times 3 = 21 \to 21 + 1 = 22$. b) **7** : après un redémarrage, la mémoire est vide et « Run all » exécute ①, ② et ③ une seule fois.
**Erreur fréquente** : répondre 7 en a, en suivant l'ordre de la page. **À retenir** : c'est la raison d'être de *Restart and run all*.

### Ex 0A.14 — Nombres et f-strings
a) **5.08** (`round(5076 / 1000, 2)`) · b) **`Gentoo (Biscoe): 5.1 kg, flipper 21.7 cm`**, avec `f"{species} ({island}): {mass_g / 1000:.1f} kg, flipper {flipper_mm / 10:.1f} cm"` · c) **9** (`50_000 // 5076`) · d) **4316** g (`50_000 % 5076`).
**Pourquoi** : `:.1f` arrondit à l'affichage, sans modifier la variable ; `//` et `%` sont le quotient et le reste de la division euclidienne ($50\,000 = 5076 \times 9 + 4316$).
**Erreurs fréquentes** : `round(mass_g / 1000, 1)` en a ; oublier de convertir la nageoire en cm ; `50 // 5076` (des kg divisés par des g : 0).
**Variante** : affiche la masse avec un séparateur de milliers : `f"{mass_g:,} g"` donne `5,076 g`.

### Ex 0A.15 — Chaînes
a) **`Adelie, Gentoo, Chinstrap`** · b) **`Pygoscelis antarctica`** · c) **`P. papua`**.
**Démarche** : `raw.split()[0]` découpe sur les espaces et garde le premier mot ; pour le nom latin, `start = raw.find("(")` et `end = raw.find(")")`, puis `raw[start + 1:end]` (le `+ 1` saute la parenthèse ouvrante ; la fin exclue écarte la fermante).
**Erreurs fréquentes** : garder les parenthèses (`raw[start:end + 1]`) ; utiliser `split("(")[1]`, qui garde la parenthèse fermante.
**Variante** : `short_name(r).lower()` pour obtenir des étiquettes en minuscules, comme dans `penguins.csv`… qui les écrit pourtant avec une majuscule : vérifie toujours le format réel des données.

### Ex 0A.16 — Listes
a) **14** ($195 - 181$) · b) **189.0** ($1890 / 10$) · c) **[195, 195, 193]** · d) **190.0**.
**Médiane** : triée, la liste vaut `[181, 181, 186, 186, 190, 190, 193, 193, 195, 195]` ; les 5ᵉ et 6ᵉ valeurs ont les indices 4 et 5 : $(190 + 190) / 2 = 190$.
**Erreurs fréquentes** : prendre les indices 5 et 6 (compter à partir de 1) ; `sorted(flippers)[:3]` (les trois plus petites) ; `flippers.sort()` puis utiliser la valeur renvoyée (`None`).
**Variante** : compare avec `statistics.median(flippers)` (0A.24).

### Ex 0A.17 — Tuples et déballage
a) **(181, 195)** avec `return min(values), max(values)` (la virgule crée le tuple) · b) **3600** ($6300 - 2700$) · c) **`Gentoo/Adelie`** avec `first_species, second_species = second_species, first_species` · d) **8** (10 éléments moins le premier et le dernier).
**Erreur fréquente** : échanger en deux lignes (`a = b` puis `b = a`), ce qui donne deux fois la même valeur.

### Ex 0A.18 — Dictionnaires
a) **3** · b) **151** (152 Adelie, dont une sans masse) · c) **3733** (moyenne des Chinstrap : 3733,1 g) · d) **Gentoo**.
**Démarche** : `masses_by_species.setdefault(species, []).append(mass)` crée la liste vide la première fois, puis ajoute ; équivalent avec un test : `if species not in masses_by_species: masses_by_species[species] = []`.
**Erreurs fréquentes** : `masses_by_species[species].append(mass)` sans créer la liste (`KeyError`) ; compter les `None`.
**Variante** : `collections.defaultdict(list)` fait la même chose sans test ; `max(mean_by_species, key=mean_by_species.get)` trouve l'espèce la plus lourde en une ligne (0A.41).

### Ex 0A.19 — Ensembles
a) **{Biscoe, Dream, Torgersen}** · b) **{Biscoe}** · c) **{Dream}** · d) **5** couples différents.
**Pourquoi** : les Gentoo ne vivent que sur Biscoe et les Chinstrap que sur Dream ; les Adelie sont sur les trois îles : $3 + 1 + 1 = 5$ couples. C'est un **biais** du dataset (l'île « trahit » l'espèce), qui reviendra au ch. 7 (fiche du dataset : `data/cards/penguins.md`).
**Erreur fréquente** : `|` (la réunion) au lieu de `&` (l'intersection) en c.

### Ex 0A.20 — Conditions
a) **medium** (3500 n'est pas `< 3500`) · b) **unknown** · c) **118** « heavy » · d) **71** « light ».
**Pourquoi tester `None` en premier** : `None < 3500` lève une `TypeError` ; l'ordre des tests est la logique du programme.
**Variante** : ajoute une catégorie `"very heavy"` au-delà de 5500 g : où faut-il placer le test ?

### Ex 0A.21 — Boucles
a) **253.85** kg (68 Chinstrap) · b) indice **169** (le premier manchot de plus de 6000 g, un Gentoo de 6300 g) · c) **14** ans ($344 \times 1{,}08^{13} \approx 935$, $344 \times 1{,}08^{14} \approx 1010$) · d) **2550** · e) indice **3** (`flipper_list.index(None)`).
**Erreurs fréquentes** : `range(0, 100, 2)`, qui s'arrête à 98 (2450) ; `while colony < 1000` donne aussi 14 ici, mais la condition exacte de l'énoncé est « dépasser » ; oublier `break` en b (on obtient alors le **dernier** manchot lourd).
**Variante** : avec `math.log`, calcule directement le nombre d'années : $n = \lceil \ln(1000/344) / \ln(1{,}08) \rceil$.

### Ex 0A.22 — Compréhensions
a) **24.35** kg (quatre manchots : 6,3 ; 6,05 ; 6,0 ; 6,0) · b) **68** · c) **23.1** cm · d) **{TORGERSEN, BISCOE, DREAM}**.
**Exemples** : `[m / 1000 for m in mass_list if m is not None and m >= 6000]` (l'ordre des tests compte : `None >= 6000` lèverait une erreur) ; `{sp: species_list.count(sp) for sp in sorted(set(species_list))}`.
**Variante** : `counts` en une ligne avec `Counter(species_list)` (0A.24).

### Ex 0A.23 — Premières fonctions
a) **48.27** mm/kg ($181 / 3{,}75$) · b) **`Gentoo: 5.1 kg`** · c) **`Gentoo: 5076 g`** · d) **`Adelie: 3.75 kg`**.
**Démarche** : un `if`/`elif`/`else` sur `unit`, et un `raise ValueError(...)` dans le `else` : mieux vaut refuser une unité inconnue que renvoyer un résultat faux.
**Erreur fréquente** : `print` au lieu de `return` : la fonction renvoie alors `None`.

### Ex 0A.24 — Modules de la bibliothèque standard
a) **7** (`math.ceil(344 / 50)` : 6 lots pleins et un de 44) · b) **190.0** · c) **Biscoe** et d) **168** · e) **802.0** g (`statistics.stdev`, écart-type d'échantillon : on y revient au ch. 2).
**Observation sur `random`** : avec `random.seed(0)`, les dix lancers sont identiques d'une exécution à l'autre ; sans graine, ils changent à chaque fois.

### Ex 0A.25 — Exceptions
a) **4** masses valides : 3750, 3800, 4200 (`"4.2e3"` est une écriture scientifique valide) et 5076 · b) **16826.0** · c) la cellule de vérification affiche **`ValueError`**, levée par ton `raise ValueError("empty mass")`.
Les textes refusés : `""` (vide), `"3,450"` (virgule décimale : `float` ne l'accepte pas), `"-12"` (négatif), `"NA"` (pas un nombre).
**Erreurs fréquentes** : `except:` tout seul, qui cacherait aussi une faute de frappe dans le code ; oublier le `strip()` (`" 3800 "` passe quand même, car `float` tolère les espaces, mais un texte fait seulement d'espaces ne serait pas refusé proprement).
**Variante** : accepte aussi la virgule décimale française en remplaçant `","` par `"."` avant la conversion : combien de masses valides alors ? (5)

### Ex 0A.26 — Premier module mylearn
Une solution possible dans `_example.py` :
```python
values = list(values)
if not values:
    raise ValueError("mean() of an empty sequence")
return float(sum(values)) / len(values)
```
`mean(flippers)` vaut **189.0** ; les tests affichent `5 passed`. La conversion en liste évite le piège de `if not values` sur un array NumPy (« truth value … is ambiguous ») ; `if len(values) == 0` marche aussi.
**Erreurs fréquentes** : oublier d'enregistrer le fichier ; oublier de redémarrer le noyau (le module chargé en mémoire est l'ancien) ; lever autre chose qu'une `ValueError` pour la liste vide (un test le vérifie).

### Ex 0A.27 — Premiers arrays
a) **(333, 4)** · b) **float64** · c) **2** · d) **1332** ($333 \times 4$) · e) **0.3** (NumPy affiche `0.30000000000000004` : les flottants sont approchés, fiche §100.2.1).

### Ex 0A.28 — Indexation et masques
a) **185.0** · b) **4406.75** (`X[100:200, 3].mean()` : la fin 200 est exclue, donc les indices 100 à 199) · c) **46** avec `((X[:, 0] > 45) & (X[:, 2] < 200)).sum()` · d) **0.345** avec `(X[:, 3] >= 4500).mean()` (115 manchots sur 333).
**Erreurs fréquentes** : `>` au lieu de `>=` en d (0.336 : « au moins » inclut 4500) ; `and` au lieu de `&` (`ValueError: The truth value of an array ... is ambiguous`) ; oublier les parenthèses autour de chaque condition (`&` est prioritaire sur `>`).

### Ex 0A.29 — Vue ou copie 🔮
a) **100** : `b = a[2:5]` est une vue, `b[0]` est `a[2]` · b) **0** : `c = a[[0, 1]]` est une copie · c) **113** ($0 + 1 + 100 + 3 + 4 + 5$).
**À retenir** : tranche = vue ; masque ou liste d'indices = copie. `np.shares_memory(a, b)` le vérifie.

### Ex 0A.30 — Calcul vectorisé
a) **4.207** kg · b) **0.153** · c) **3.61** · d) **115** (`np.where(X[:, 3] >= 4500, "heavy", "not heavy")`, la même règle qu'en 0A.20).
**Pourquoi** : chaque opération (`/`, `-`, `np.where`) s'applique aux 333 valeurs d'un coup. La normalisation min-max (b) ramène les valeurs dans [0, 1] : le plus petit manchot vaut 0, le plus grand 1 (ch. 12).
**Erreurs fréquentes** : `(f - f.min()) / f.max() - f.min()` : sans parenthèses, la division passe avant la soustraction ; en a, partir de `mass_known` (344 − 2 manchots) au lieu de `X` (333) donne 4.202.

### Ex 0A.31 — Aléatoire reproductible
a) **0.170** (1696 six sur 10 000 : proche de $1/6 \approx 0{,}167$, sans être égal : c'est la fluctuation d'échantillonnage) · b) **[2, 4, 3, 0, 1]** · c) **[169, 156, 251]** · d) **True** · e) **202.8** mm.
**Erreurs fréquentes** : en e, `rng.permutation(333)[:50]` est aussi un tirage sans remise valable, mais il n'utilise pas le hasard de la même façon que `choice` (202.2 mm) : deux méthodes correctes, deux échantillons différents ; `rng.integers(1, 6)` (la borne haute est exclue : jamais de 6) ; appeler `np.random.default_rng(2026)` à chaque tirage (on retombe toujours sur les mêmes premiers nombres).

### Ex 0A.32 — Premier contact avec Penguins
a) **(344, 8)** · b) **333** (11 sexes manquants) · c) **4202** g (4201,75) · d) **231** mm · e) **5** colonnes numériques (les 4 mesures et `year`).

### Ex 0A.33 — Sélectionner
a) **Biscoe** · b) **Chinstrap** · c) **28** · d) **44.17** mm · e) **3368.8** g.
**Démarche** : `penguins.loc[100, "island"]` (étiquette), `penguins.iloc[-1]["species"]` (position), `penguins.loc[(penguins["species"] == "Adelie") & (penguins["sex"] == "female"), "body_mass_g"].mean()`.

### Ex 0A.34 — Valeurs manquantes et doublons
a) **19** ($4 \times 2$ mesures manquantes et 11 sexes) · b) **333** · c) **342** · d) **11** · e) **5** · f) **200.89** mm.
**Pourquoi c ≠ b** : `dropna()` retire toute ligne qui a **au moins une** valeur manquante (y compris le sexe) ; `subset` ne regarde que la masse.

### Ex 0A.35 — `X` et `y`
a) **(333, 4)** · b) **(333,)** · c) **3** · d) **201.0** mm.
**Erreurs fréquentes** : `penguins[["species"]]` (doubles crochets) donne un DataFrame, donc un `y` de forme `(333, 1)` : les modèles attendent un vecteur `(333,)` ; `dropna(subset=numeric_cols)` garde les 9 manchots dont seul le sexe manque (342 lignes) : l'énoncé demande des manchots **sans aucune** valeur manquante.

### Ex 0A.36 — Premiers graphiques
Les trois figures attendues sont dans `05_solutions.ipynb`. Réponses aux questions : les **Gentoo** ont le bec le moins profond (≈ 15,0 mm en moyenne, contre ≈ 18,4 mm pour les Adelie et les Chinstrap), alors qu'ils sont bien plus lourds ; d'une année à l'autre, la masse moyenne de chaque espèce varie de 120 g au plus (Gentoo : 5071 g en 2007, 5020 g en 2008, 5141 g en 2009), soit moins de 3 % : pas de tendance nette sur trois ans.
**Check-list** : axes nommés avec leurs unités ; un titre ; une légende quand il y a plusieurs séries ; `plt.show()` à la fin de chaque cellule graphique.

### Ex 0A.37 — Lire un traceback 🐛
a) **`TypeError, IndexError, KeyError, AttributeError, NameError`** · b) **10.8** kg · c) **195** · d) **176** (52 + 124) · e) **True** · f) **21.7** cm.
**Les cinq causes** : `sum` part de `0` et ne sait pas ajouter `0 + "3750"` (convertir avec `float`, qui tolère l'espace de `" 3800"`) ; le dernier indice est `len - 1` (ou `-1`) ; la clé exacte est `"Torgersen"` (normaliser avec `.strip().capitalize()`) ; `uppper` n'existe pas (`AttributeError: 'str' object has no attribute 'uppper'. Did you mean: 'upper'?`) ; `MM_PER_CMS` n'est défini nulle part (Python 3.13 suggère `MM_PER_CM`).
**Méthode** : la dernière ligne donne le type et le message ; la ligne juste au-dessus, marquée `^^^`, montre l'expression fautive dans **ta** fonction ; les lignes plus hautes (ici `diagnose`) disent seulement par où l'on est passé.
**Variante** : provoque toi-même une `ZeroDivisionError` et une `ValueError` (`int("3,5")`) et lis leurs tracebacks.

### Ex 0A.38 — Lire penguins.csv comme un simple fichier texte
a) **344** lignes de données · b) **2** masses manquantes · c) **200.92** mm · d) **125** lignes (l'en-tête et 124 Gentoo) · e) **True**.
**Pourquoi** : `f.readline()` consomme la première ligne, et la boucle `for line in f` reprend **après** elle. Les valeurs lues sont des chaînes : `"181"`, pas `181` ; il faut `float` pour calculer, et sauter `"NA"` avant de convertir.
**Erreurs fréquentes** : compter l'en-tête (345) ; diviser par 344 au lieu du nombre de valeurs connues ; oublier le `"\n"` entre les lignes écrites (tout sur une ligne) ; ouvrir sans `encoding="utf-8"` (accents mal lus sous Windows).
**Limites** : ce découpage naïf casse dès qu'une valeur contient une virgule entre guillemets (`"Nest never observed, full clutch"`) : c'est tout l'intérêt du module `csv` et de `pd.read_csv`.

### Ex 0A.39 — `json` et `pickle`
a) **`TypeError`** (`Object of type int64 is not JSON serializable` : les valeurs de `results["n_missing"]` sont des `np.int64`) · b) **5076.0** · c) **`list`** · d) **True**.
**Pourquoi** : JSON ne connaît que les types simples de Python ; un nombre NumPy doit être converti (`int(v)`, `float(v)` ou `v.item()`). Un tuple est écrit comme une liste, et relu comme une liste : l'objet relu n'est donc pas égal à l'original. pickle, lui, enregistre les objets Python tels quels.
**Erreurs fréquentes** : ouvrir le fichier pickle en mode texte (`"w"` au lieu de `"wb"`) ; oublier de convertir les valeurs du dictionnaire **imbriqué** `n_missing` (c'est là que sont les `np.int64` : `value_counts().to_dict()` renvoie déjà des nombres Python) ; croire qu'un `np.float64` pose problème (c'est une sous-classe de `float`, JSON l'accepte, contrairement à `np.int64`).
**Sécurité** : non, on ne relit jamais un `.pkl` d'origine inconnue : le charger peut exécuter du code (fiche §100.6.4).
**Variante** : `json.dumps(results, default=lambda v: v.item())` confie la conversion à JSON pour les objets qu'il ne connaît pas.

### Ex 0A.40 — `*args`, `**kwargs` et keyword-only
a) **187.3** · b) **3562.5** · c) **3** clés (`lr`, `epochs`, `batch_size`) · d) **`TypeError`** (`make_config() takes 0 positional arguments but 1 was given`) · e) **1.5** (0,5 × 3) · f) **`ValueError`**.
**Pourquoi** : `*values` range les arguments positionnels dans un tuple (vide si l'on n'en passe aucun : il faut le tester avant de diviser, sinon `ZeroDivisionError`). Le `*` seul de `make_config` interdit tout argument positionnel. À l'appel, `*masses` et `**settings` déballent une liste et un dictionnaire.
**Erreur fréquente** : `def mean_of(values)` avec une liste en paramètre : on devrait alors appeler `mean_of([181, 186, 195])`, et `mean_of(181, 186, 195)` lève une `TypeError`.

### Ex 0A.41 — `lambda`, `key=` et `Callable`
a) **Gentoo 6300** · b) **[172, 174, 176]** · c) **Biscoe, Dream, Torgersen** · d) **1400.95** kg · e) **148** nageoires d'au moins 200 mm.
**Pourquoi** : `key=` reçoit une fonction appliquée à chaque élément ; `max`, `min` et `sorted` comparent les résultats, mais renvoient les **éléments** eux-mêmes (le tuple entier). `island_sizes.get` est une méthode passée sans parenthèses : on donne la fonction, pas son résultat.
**Erreurs fréquentes** : `key=island_sizes.get()` (on appelle la fonction au lieu de la donner : `TypeError`) ; oublier `reverse=True` en c ; `>` au lieu de `>=` en e (144).
**Variante** : `sorted(penguin_records, key=lambda r: (r[0], -r[3]))` trie par espèce, puis du plus lourd au plus léger : une clé peut être un tuple.

### Ex 0A.42 — Fermetures
a) **0.475** · b) **[0.0, 1.0]** · c) **3600.0** · d) **3700.0** (la moyenne de 3750, 3800, 3250 et 4000 : `other` a son propre total).
**Pourquoi** : `make_scaler` renvoie `scale` **sans l'appeler** ; `scale` se souvient de `low` et `high`. Dans `add`, `nonlocal total, count` permet de **modifier** les variables de la fabrique ; chaque appel à `make_running_mean()` crée un nouvel environnement.
**Erreurs fréquentes** : `return scale()` (on renvoie un résultat, pas une fonction) ; oublier `nonlocal` (`UnboundLocalError: cannot access local variable 'total'`) ; utiliser des variables **globales** : tous les compteurs partageraient alors le même total.
**Variante** : la même chose avec une classe (`RunningStats`, 0A.47) : une fermeture est un petit objet à une seule méthode.

### Ex 0A.43 — Le piège des lambdas créées dans une boucle 🔮
a) **[30, 30, 30]** · b) **[10, 20, 30]** avec `fixed = [lambda x, k=k: x * k for k in range(1, 4)]`.
**Pourquoi** : chaque `lambda` garde un accès à la **variable** `k`, pas à sa valeur ; elle la lit quand on l'appelle, une fois la boucle finie, et `k` vaut alors 3. Une valeur par défaut, elle, est calculée **au moment de la création** de la fonction : `k=k` fige la valeur courante.
**Autre correction** : une fabrique, `def make_multiplier(k): return lambda x: x * k`, puis `[make_multiplier(k) for k in range(1, 4)]` : chaque appel crée son propre `k`.

### Ex 0A.44 — Fonctions récursives
a) **344** · b) **5** feuilles · c) **4** niveaux · d) **124** · e) **0**.
**Démarche** : chaque fonction a un cas de base (une feuille, `not isinstance(tree, dict)`) et un appel récursif sur chaque enfant (`tree.values()`) ; seule la façon de **combiner** change : `sum`, `sum` de 1, `1 + max`, `max`.
**Erreurs fréquentes** : oublier le cas de base (`AttributeError: 'int' object has no attribute 'values'` en arrivant sur une feuille ; dans d'autres problèmes, une récursion sans fin lève `RecursionError`) ; compter les nœuds au lieu des feuilles ; `depth` qui renvoie 1 pour une feuille (on obtient 5 au lieu de 4).
**Variante** : écris `paths(tree)`, qui renvoie la liste des chemins `"Antarctica/Anvers/Biscoe/Adelie"` de chaque feuille (le chemin courant se passe en paramètre).

### Ex 0A.45 — Expressions régulières
a) **344** identifiants valides · b) **False** (aucun piège accepté) · c) **100** · d) **14** pontes en décembre · e) **7** commentaires sur le sexage.
**Démarche** : `re.fullmatch(r"N\d+A[12]", text)` valide tout le texte ; un groupe `(\d+)` isole le numéro, que `int()` convertit ; `(\d{4})-(\d{2})-(\d{2})` découpe une date en trois groupes.
**Erreurs fréquentes** : `re.search` ou `re.match` au lieu de `fullmatch` (`"xN1A1"` ou `"N1A1 "` passeraient) ; `[1-2]` est juste, mais `[1,2]` accepte aussi la virgule ; comparer des chaînes (`max` de `"99"` et `"100"` donne `"99"`) ; sans `re.IGNORECASE`, seulement 2 commentaires (`"sexing"` en minuscules), car 5 commencent par `"Sexing"`.
**Variante** : pour des dates, `pd.to_datetime(penguins_raw["Date Egg"]).dt.month` fait la même chose que `egg_month` ; les regex restent l'outil des formats maison (identifiants, journaux d'expérience).

### Ex 0A.46 — `itertools` et `heapq`
a) **6** paires ($\binom{4}{2} = 6$) · b) **Adelie, Gentoo** (≈ 1386 g d'écart) · c) **18** réglages (3 × 3 × 2) · d) **[6300, 6050, 6000]** · e) **Chinstrap, Adelie** (2700 g puis 2850 g).
**Pourquoi** : `combinations` et `product` évitent des boucles imbriquées et produisent leurs éléments à la demande (ce sont des itérateurs : `len(list(...))` pour compter). `heapq.nlargest(k, ...)` trouve les *k* plus grands sans tout trier.
**Erreurs fréquentes** : `permutations` au lieu de `combinations` (12 paires ordonnées) ; `len(product(...))` (un itérateur n'a pas de longueur).

### Ex 0A.47 — `RunningStats` et `@dataclass`
a) **342** · b) **4201.75** · c) **800.78** (identique à `np.std(known_masses)`, l'écart-type « population », `ddof=0`) · d) **(2700, 6300)** · e) **5.076** · f) **True** · g) **`PenguinRecord(species='Adelie', island='Dream', mass_g=None)`**.
**Pourquoi** : les attributs (`self.n`, `self.total`…) gardent l'état entre deux appels de `add` ; la formule $\sigma^2 = \overline{x^2} - \bar{x}^2$ n'a besoin que de trois sommes. `@dataclass` écrit `__init__`, `__repr__` et `__eq__` à partir des annotations.
**Erreurs fréquentes** : oublier `self.` (la variable reste locale à la méthode) ; initialiser `minimum` à 0 (le minimum serait faux pour des masses positives) ; `std` sans racine carrée (la variance, 641 250) ; garder la ligne `PenguinRecord = None` après la classe.
**Variante** : l'algorithme de Welford met à jour la moyenne et la somme des carrés des écarts pas à pas, sans perte de précision pour les très grandes séries.

### Ex 0A.48 — Méthodes spéciales : `Vector2D`
a) **`Vector2D(x=4, y=6)`** · b) **5.0** · c) **True** · d) **9** · e) **47.57** mm · f) **9.36** mm.
**Pourquoi** : `sum(vectors)` calcule `0 + v1` : `int.__add__` renvoie `NotImplemented`, et Python appelle `v1.__radd__(0)`. De même, `2 * v` appelle `v.__rmul__(2)`. Sans `__eq__`, `==` compare les identités (deux objets différents ne sont jamais égaux).
**Erreurs fréquentes** : modifier `self` dans `__add__` au lieu de renvoyer un nouveau vecteur ; oublier `__radd__` (`TypeError: unsupported operand type(s) for +: 'int' and 'Vector2D'`) ; `__repr__` qui renvoie autre chose qu'une chaîne.
**Variante** : `sum(vectors, Vector2D(0, 0))` marche sans `__radd__`, en donnant la valeur de départ.

### Ex 0A.49 — Héritage : un mini-estimateur
a) **Adelie** · b) **0.438** · c) **0.892** · d) **Adelie** · e) **True**.
**Pourquoi** : les classes filles n'écrivent que `fit` et `predict` ; `score` et `__call__` viennent de `BaseClassifier`, et `super().__init__(verbose)` laisse la classe mère initialiser ce qui la concerne. `fit` renvoie `self`, ce qui permet `NearestCentroidClassifier().fit(X2, y).score(X2, y)`.
**Erreurs fréquentes** : oublier `return self` (`'NoneType' object has no attribute 'score'`) ; recopier `score` dans chaque classe ; oublier de convertir `y` en array (`X[y == label]` avec une liste échoue).
**Remarque** : comparer les distances au carré donne le même classement ; la racine est inutile pour trouver le centre le plus proche.
**À retenir** : le classifieur majoritaire est la **référence minimale** (*baseline*) : un modèle qui ne la bat pas n'a rien appris. Ces scores sont mesurés sur les données d'entraînement ; on apprendra au ch. 8 à évaluer honnêtement.

### Ex 0A.50 — Générateurs et itérables : un mini-Dataset
a) **333** · b) **Adelie** · c) **200.97** mm · d) **6** lots · e) **13** exemples · f) **0** · g) **True**.
**À noter** : sans `__iter__`, `for x in ds` marcherait quand même (Python se rabat sur `__getitem__` avec 0, 1, 2… jusqu'à une `IndexError`) ; `__iter__` avec `yield` rend l'intention explicite, et c'est ce que vérifie g).
**Pourquoi** : `len`, `[]` et `for` appellent `__len__`, `__getitem__` et `__iter__`. Une fonction avec `yield` renvoie un générateur : les lots sont produits un par un, quand la boucle les demande, puis le générateur est épuisé.
**Erreurs fréquentes** : `return` au lieu de `yield` (un seul lot) ; `range(0, len(dataset) - batch_size, batch_size)` (on perd le dernier lot) ; parcourir deux fois le même générateur.
**Variante** : ajoute un paramètre `shuffle` à `batches`, avec les indices de `iterate_minibatches` (0A.66) : c'est exactement un `DataLoader`.

### Ex 0A.51 — Réductions par axe, tri, `argmax` et `unique`
a) **[43.99, 17.16, 200.97, 4207.06]** · b) les moyennes par espèce, par exemple **[38.82, 18.35, 190.10, 3706.16]** pour les Adelie · c) **[1, 1, 2, 2]** · d) **Gentoo** · e) **[146, 68, 119]** · f) **[2700, 2850, 2850]**.
**Pourquoi** : sur `class_means` `(3, 4)`, `argmax(axis=0)` fait disparaître l'axe des espèces : il reste une réponse par mesure. `argmax(axis=1)` donnerait la « meilleure mesure » de chaque espèce, ce qui n'a pas de sens (les unités diffèrent).
**Erreurs fréquentes** : `X.mean()` sans `axis` (un seul nombre) ; `X[:, 3].max()` au lieu de `argmax` en d (on obtient la masse, pas la position) ; confondre `np.sort` (les valeurs) et `np.argsort` (les positions).

### Ex 0A.52 — Broadcasting : standardiser
a) **(333, 4)** · b) **True** · c) **[1.0, 1.0, 1.0, 1.0]** · d) **[-0.90, 0.78, -1.43, -0.57]** : bec plus court, bec plus profond, nageoire plus courte et masse plus faible que la moyenne · e) **1.17**.
**Pourquoi** : `(333, 4)` et `(4,)` sont compatibles ; chaque ligne reçoit la même correction. En e), `(5, 1, 4) - (1, 5, 4)` s'étire en `(5, 5, 4)` : toutes les différences entre paires, sans boucle.
**Erreurs fréquentes** : `X.mean()` sans axe (on retire la moyenne de **tout** le tableau) ; `axis=1` (une moyenne par manchot, forme `(333,)`, incompatible avec `(333, 4)`) ; sommer sur le mauvais axe en e (`axis=0` ou `axis=1` au lieu de `axis=2`).
**Variante** : `sklearn.preprocessing.StandardScaler` fait la même chose (ch. 12), en retenant les moyennes du **jeu d'entraînement** pour les appliquer ensuite au jeu de test.

### Ex 0A.53 — `reshape`, transposée et empilement
a) **9** · b) **(2, 6)** · c) **[1, 5, 9]** · d) **(333, 2)** · e) **(333, 5)** · f) **333.0** · g) **(233, 4)**.
**Pourquoi** : `reshape` relit les données ligne par ligne ; la ligne 1 de la transposée est la colonne 1 de `v.reshape(3, 4)`. `np.stack(..., axis=1)` crée un nouvel axe en dernière position (les colonnes) ; `np.hstack` colle deux tableaux `(333, 1)` et `(333, 4)` côte à côte.
**Erreurs fréquentes** : `np.stack` sans `axis=1` (forme `(2, 333)`) ; `np.ones(len(X))` de forme `(333,)` dans `hstack` (`ValueError` : les tableaux n'ont pas le même nombre de dimensions) ; oublier que `X[:100]` s'arrête à la ligne 99 et que `X[200:]` va jusqu'à la dernière (100 + 133 = 233 lignes).

### Ex 0A.54 — Images MNIST
a) **(1000, 28, 28)** · b) **uint8** · c) **34.57** · d) **0.809** · e) **(1000, 784)** · f) **[0, 1]** : le 0 utilise le plus d'encre, le 1 le moins · g) **8.0**.
**Pourquoi** : 81 % des pixels sont noirs ; une image moyenne par chiffre (la cellule qui suit l'exercice) montre un anneau épais pour le 0 et un trait fin pour le 1. Diviser par 255 crée des `float64` : 8 octets par pixel au lieu de 1.
**Erreurs fréquentes** : `images.mean(axis=0)` en c (l'image moyenne, pas l'intensité de la première image) ; `images.reshape(-1, 784)` est juste aussi, mais `images.reshape(784, -1)` mélange les pixels de plusieurs images.
**Variante** : `images.astype(np.float32) / 255` divise la mémoire par deux par rapport à `float64` : c'est le type par défaut de PyTorch.

### Ex 0A.55 — Boucle Python contre NumPy 🔬
Il n'y a pas de valeur unique : les deux versions doivent donner les **mêmes** moyennes, et NumPy est ici de l'ordre de 100 à 300 fois plus rapide (le notebook de solutions affiche les mesures d'une machine de test). Les deux temps croissent à peu près comme le nombre d'images ; pour 100 images, le temps de NumPy est surtout un coût fixe d'appel.
**Mon analyse (modèle)** : « NumPy est environ 200 fois plus rapide ; le rapport varie selon les mesures, mais reste de cet ordre. La boucle Python interprète une instruction et crée un objet Python pour chacun des 784 000 pixels ; NumPy parcourt un bloc de mémoire contigu dans du code compilé. »
**Erreurs fréquentes** : additionner des `uint8` sans `int()` (les sommes « débordent » et les moyennes sont fausses) ; mesurer une seule fois une durée de quelques millisecondes (trop bruité : on garde le meilleur de plusieurs essais).

### Ex 0A.56 — Bugs NumPy 🐛
a) **[43.99, 17.16, 200.97, 4207.06]** · b) **10000.0** · c) **True** · d) **-8.2**.
**Les trois causes** : `X.mean()` réduit tout le tableau (il faut `axis=0`) ; `(333, 1) - (333,)` donne un tableau `(333, 333)` de toutes les différences croisées, dont la moyenne des carrés n'a aucun sens (1,3 million au lieu de 10 000) ; `X[:k, 2]` est une **vue** : `part -= ...` modifie `X` lui-même.
**Correction robuste de `mse`** : aplatir avec `ravel()` **et** lever une `ValueError` si les formes diffèrent encore : mieux vaut une erreur qu'un nombre faux.
**Remarque** : `part = part - part.mean()` corrige aussi `centered_flippers` (la soustraction crée un nouveau tableau, alors que `-=` modifie la vue sur place), mais `.copy()` rend l'intention explicite.

### Ex 0A.57 — Compter et regrouper
a) **168** · b) **0.442** · c) **5484.8** g · d) **7.13** mm · e) **Gentoo** (59,6 mm) · f) **1**.
**Pourquoi** : `groupby(["species", "sex"])` crée un index à deux niveaux, lu avec un tuple ; `agg` calcule plusieurs statistiques d'un coup, et `count` montre au passage les valeurs manquantes (151 nageoires d'Adelie sur 152).
**Erreurs fréquentes** : `value_counts()["Adelie"]` sans `normalize=True` en b (152) ; `.loc["Gentoo", "male"]` fonctionne aussi, mais `["Gentoo"]["male"]` enchaîne deux sélections ; `size()` compte les lignes, `count()` les valeurs non manquantes.

### Ex 0A.58 — Le filtre qui ne filtre pas 🐛
a) **`ValueError`** · b) **`TypeError`** · c) **61** Gentoo de plus de 5000 g · d) **11** (avec `fixed_df.loc[fixed_df["sex"].isna(), "sex"] = "unknown"`) · e) **176**.
**Les trois causes** : `and` demande à chaque Series d'être `True` ou `False` en entier, ce que pandas refuse (« truth value of a Series is ambiguous ») ; `&` passe **avant** `==` et `>` : Python calcule d'abord `"Gentoo" & penguins["body_mass_g"]` ; `demo[masque]` renvoie une copie temporaire, que l'affectation `["sex"] = ...` modifie, puis qui disparaît (d'où le `SettingWithCopyWarning` et 0 `"unknown"`).
**À retenir** : filtres avec `&`, `|`, `~` et des parenthèses ; modifications avec `.loc[masque, colonne] = valeur`, en une seule instruction.

### Ex 0A.59 — Figures à plusieurs panneaux
Les trois figures de référence sont dans `05_solutions.ipynb`. **Check-list** : `fig_digits` a 10 panneaux, chacun avec une image et un titre (le chiffre), sans axes ; `fig_hist` a 3 panneaux avec un titre, une légende et des histogrammes semi-transparents (`alpha`) pour voir les chevauchements.
**Erreurs fréquentes** : `axes[d]` sur une grille `(2, 5)` (il faut `axes.ravel()[d]` ou `axes[d // 5, d % 5]`) ; oublier `plt.show()` ; des histogrammes opaques qui se cachent les uns les autres.

### Ex 0A.60 — Quelle mesure sépare le mieux les espèces ? 📈
a) **bill_length_mm** · b) **Gentoo** · c) **Chinstrap** · d) **bill_length_mm** : le rapport vaut 3,31 pour le bec, 0,84 pour la nageoire, 0,06 pour la profondeur du bec et la masse.
**Lecture** : sur la longueur du bec, les histogrammes des Adelie (≈ 39 mm) et des Chinstrap (≈ 49 mm) se chevauchent à peine ; sur la masse et la profondeur du bec, ils se superposent presque entièrement. Une nageoire de 225 mm n'existe que chez les Gentoo ; un bec long (50 mm) **et** profond (19 mm) n'appartient qu'aux Chinstrap. Attention aux points proches d'une frontière : un bec de 46 × 19 mm serait ambigu (des Adelie et des Chinstrap l'entourent).
**À retenir** : une mesure seule sépare deux espèces au mieux ; deux mesures bien choisies (le nuage du bec) séparent les trois. C'est l'intuition de la classification (ch. 7) et du critère de Fisher.

### Ex 0A.61 — Une docstring au format NumPy 🛠️
La docstring complète est dans `05_solutions.ipynb` ; les quatre vérifications doivent afficher ✅. Les points clés : chaque titre de section est souligné par autant de tirets que de lettres ; les sorties des exemples sont recopiées **exactement** (`array([0. , 0.5, 1. ])`, avec ses espaces) ; l'exemple d'erreur commence par `Traceback (most recent call last):`, puis `...`, puis la dernière ligne réelle.
**Erreurs fréquentes** : une sortie tapée « à peu près » (`array([0, 0.5, 1])`) ; un message d'erreur différent de celui que lève le code ; renvoyer une liste au lieu d'un array.
**Variante** : `python -m pytest --doctest-modules mon_module.py` lance les doctests d'un fichier entier, comme on le fera pour mylearn.

### Ex 0A.62 — Écrire tes propres tests 🛠️
Quatre tests suffisent (6 cas avec la paramétrisation) : ils passent sur `reference_scale`, et chaque version buggée en fait échouer au moins un. `scale_bug_1` (dénominateur `max` au lieu de `max - min`) échoue sur les valeurs attendues ; `scale_bug_2` (pas d'erreur sur des valeurs égales) sur `pytest.raises` ; `scale_bug_3` (échelle inversée) sur le test simple.
**Erreurs fréquentes** : comparer des flottants avec `==` (utiliser `pytest.approx`) ; un test sans `assert` (il passe toujours) ; utiliser dans un test une variable du notebook (`X`, `penguins`), inconnue du fichier de test.
**À retenir** : un test utile est un test qui **peut** échouer. Vérifier que ses tests attrapent des bugs volontaires s'appelle le *mutation testing*.

### Ex 0A.63 — `utils.count_values`
a) **152** · b) **0.151** · c) **5** (`G`, `e`, `n`, `t`, `o` : le `o` compte deux fois) · d) **`ValueError`**, puis `9 passed` pour les tests `-k count_values`.
**Une solution** :
```python
counts = {}
for v in values:
    if isinstance(v, (float, np.floating)) and v != v:   # NaN is the only value different from itself
        raise ValueError("count_values() cannot count NaN: drop the missing values first")
    counts[v] = counts.get(v, 0) + 1
if not counts:
    raise ValueError("count_values() needs at least one value")
if normalize:
    total = sum(counts.values())
    return {k: c / total for k, c in counts.items()}
return counts
```
**Erreurs fréquentes** : `if not values` au début (plante sur un array ou une Series de plusieurs éléments) ; tester `v == np.nan` (toujours faux) ; renvoyer les proportions en arrondissant.

### Ex 0A.64 — `utils.argmax`
Les trois comparaisons avec `np.argmax` affichent ✅ (`8`, `[1, 0, 2]`, `[1, 0, 2, 0, 2]`), puis `13 passed` pour `-k argmax`. La référence (`solutions/mylearn_ref/utils.py`) écrit une fonction interne `first_max(items)` pour le cas 1-D, puis l'applique au tableau aplati (`axis=None`), à chaque ligne (`axis=1`) ou à chaque colonne, reconstruite en liste (`axis=0` ; `values.T` marche aussi).
**Erreurs fréquentes** : `>=` au lieu de `>` (le **dernier** maximum gagne) ; partir de `best_value = 0` (faux si toutes les valeurs sont négatives : partir du premier élément) ; renvoyer un `np.int64` au lieu d'un `int` quand le résultat est un seul indice (un test le vérifie).

### Ex 0A.65 — `utils.one_hot`
a) **(333, 3)** · b) **[146, 68, 119]** · c) **True** · d) **int64** (sous Windows avec NumPy 2, aussi `int64`) · puis `10 passed` pour `-k one_hot`.
**Démarche** : valider (une dimension, des entiers, positifs, `< n_classes`), puis `M = np.zeros((n, n_classes), dtype=dtype)` et `M[np.arange(n), y] = 1` : l'indexation par deux tableaux met un 1 à chaque couple (ligne `i`, colonne `y[i]`).
**Erreurs fréquentes** : `M[:, y] = 1` (met des 1 dans des colonnes entières) ; accepter `1.5` ou `-1` en silence ; `n_classes = len(np.unique(y))` (faux si une classe est absente de l'échantillon : `[0, 2]` doit donner 3 colonnes).

### Ex 0A.66 — `utils.iterate_minibatches`
a) **6** · b) **[64, 64, 64, 64, 64, 13]** · c) **[182, 254, 212, 323, 292]** (les 5 premiers de `np.random.default_rng(0).permutation(333)`) · d) **5** · e) **18** mises à jour · f) **True** · puis `12 passed` pour `-k iterate_minibatches`.
**Démarche** : `order = rng.permutation(n_samples)` (ou `np.arange`), puis `[order[s:s + batch_size] for s in range(0, n_samples, batch_size)]`, et retirer le dernier lot s'il est incomplet avec `drop_last`.
**Erreurs fréquentes** : mélanger avec `rng.choice(n, n, replace=False)` (valable, mais pas le même ordre que l'oracle : la docstring impose `permutation` ; `rng.shuffle(np.arange(n))` donne, lui, le même ordre) ; recréer le générateur à chaque epoch (même ordre à chaque fois) ; `range(0, n_samples - batch_size, batch_size)` (perd des lots).
**Pour aller plus loin** : c'est le cœur d'un `DataLoader` PyTorch (`shuffle=True`, `drop_last`) ; tu t'en serviras dans toutes les boucles d'entraînement NumPy des ch. 18 à 20.

### Ex 0A.67 — Enquête 🏆
a) **Biscoe** (80 femelles) · b) **Chinstrap** (48,8 mm, contre 47,5 pour les Gentoo) · c) **Chinstrap, Dream** (2700 g) · d) **805** g · e) **2008** · f) **0.444** · g) **3700** g · h) **1** · i) **0.907** (107 Gentoo sur 118 manchots d'au moins 4500 g) · j) **Chinstrap** (7,1 mm).
**Pièges** : b) les Gentoo sont les plus grands (nageoire, masse), mais leur bec n'est pas le plus long en moyenne ; f) la population est celle des nageoires **mesurées** (342) ; i) la proportion de Gentoo **parmi** les lourds (0,907), pas la proportion de lourds parmi les Gentoo (0,863) ; h) deux conditions combinées avec `&`.
**Bonus portfolio** : par exemple, un diagramme en barres de la masse moyenne par espèce et par sexe (`penguins.groupby(["species", "sex"])["body_mass_g"].mean().unstack().plot.bar()`), et trois phrases : les Gentoo sont les plus lourds (≈ 5 kg), les mâles pèsent 10 à 20 % de plus que les femelles dans chaque espèce, et l'île trahit en partie l'espèce (les Gentoo ne vivent que sur Biscoe).
