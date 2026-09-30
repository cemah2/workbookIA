# 0A · Python, notebooks et outils — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Le code complet des exercices du notebook est dans `05_solutions.ipynb`, exécuté.

**Sommaire** : [🧠 Quiz](#quiz) · [✏️ Papier-crayon](#papier) · [🗣️🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à E](#notebook)

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

## Notebook, parties A à E

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
