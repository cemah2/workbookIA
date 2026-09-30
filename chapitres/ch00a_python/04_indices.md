# 0A · Python, notebooks et outils — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [✏️ Papier-crayon](#papier) · [🗣️🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à E](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 0A.Q1 — Cellules, noyau, « Run all » et terminal

<details><summary>Indice 1</summary>

Le noyau est une mémoire qui vit à côté de la page : il ne « voit » pas la page, il reçoit seulement les cellules que tu exécutes.

</details>
<details><summary>Indice 2</summary>

Pour 1 à 3 : demande-toi ce qui se passe dans la mémoire du noyau quand on exécute, supprime ou redémarre. Pour 4 et 5 : `..` désigne le dossier parent ; un chemin absolu part de la racine du disque.

</details>
<details><summary>Indice 3</summary>

Deux réponses sont « vrai » : celle sur *Restart and run all* et celle sur `cd ..`. Pour les trois autres, trouve pourquoi elles sont fausses.

</details>

### 0A.Q2 — Le workbook : où j'écris, comment je vérifie

<details><summary>Indice 1</summary>

Relis le schéma des dossiers (fiche §100.1.2) et la section sur mylearn (§100.11.5).

</details>
<details><summary>Indice 2</summary>

Un seul dossier t'appartient ; les autres sont réécrits par les mises à jour. ⏳ n'est pas une erreur : c'est un état. Un test compare ton résultat à une bibliothèque « de confiance ».

</details>
<details><summary>Indice 3</summary>

1 : le dossier dont le nom dit « mon ». 3 : sans `wb.attempt`, une fonction pas encore écrite lève `NotImplementedError` : que deviendrait « Run all » ? 5 : l'outil affiche 🔒 devant certains fichiers.

</details>

### 0A.Q3 — Que vaut cette expression ? Types et opérateurs

<details><summary>Indice 1</summary>

`/` et `//` ne font pas la même division ; un calcul qui mélange `int` et `float` donne un `float`.

</details>
<details><summary>Indice 2</summary>

Pour 2 : `*` et `//` ont la même priorité et se calculent de gauche à droite, `%` aussi ; `+` vient après. Pour 3 : sur une chaîne, `*` répète.

</details>
<details><summary>Indice 3</summary>

2 : `9 // 4` vaut 2 et `9 % 4` vaut 1 : c'est la division euclidienne $9 = 4 \times 2 + 1$. 5 : `==` compare des valeurs, pas des types.

</details>

### 0A.Q4 — Liste, tuple, dict, set ou Counter ?

<details><summary>Indice 1</summary>

Pose-toi trois questions : l'ordre compte-t-il ? La taille change-t-elle ? Cherche-t-on une valeur à partir d'une clé ?

</details>
<details><summary>Indice 2</summary>

Liste : ordonnée, modifiable. Tuple : ordonné, figé. Dictionnaire : clé → valeur. Ensemble : sans doublon, sans ordre. `Counter` : un dictionnaire spécialisé dans les comptages.

</details>
<details><summary>Indice 3</summary>

La forme d'un array est justement renvoyée par NumPy sous forme de tuple ; « chacun une seule fois » est la définition d'un ensemble.

</details>

### 0A.Q5 — Conditions, boucles, compréhensions : qu'affiche ce code ?

<details><summary>Indice 1</summary>

`range(début, fin, pas)` s'arrête **avant** `fin` ; un `if`/`elif`/`else` n'exécute qu'une seule branche.

</details>
<details><summary>Indice 2</summary>

1 : écris 1, puis ajoute 3 tant que tu restes sous 8. 3 : parcours `"manchot"` lettre par lettre et garde les voyelles. 5 : quels nombres de 0 à 9 sont divisibles par 3 ?

</details>
<details><summary>Indice 3</summary>

1 : `1 4 7`. 2 : la première condition vraie gagne. 4 : `enumerate` donne des couples (indice, élément) en commençant à 0. 5 : pense à 0.

</details>

### 0A.Q6 — Paramètres, valeurs par défaut, `*args` et `**kwargs`

<details><summary>Indice 1</summary>

Les arguments positionnels remplissent les paramètres dans l'ordre ; le surplus part dans `args` (un tuple) ; les arguments nommés inconnus partent dans `kwargs` (un dictionnaire).

</details>
<details><summary>Indice 2</summary>

`c` est après `*args` : on ne peut le donner que **par son nom**. `a` n'a pas de valeur par défaut : il est obligatoire.

</details>
<details><summary>Indice 3</summary>

2 : `a=1`, `b=5`, et il reste `6, 7`. 4 : que se passe-t-il quand un paramètre obligatoire manque ? (le nom de l'erreur commence par `Type`).

</details>

### 0A.Q7 — Lire une signature : annotations, `Callable`, `lambda`, fermeture

<details><summary>Indice 1</summary>

Dans une signature, `= None` donne une valeur par défaut, et `*` seul marque le début des paramètres « uniquement nommés ».

</details>
<details><summary>Indice 2</summary>

`Callable[[int, float], None]` se lit : « une fonction qui prend un `int` et un `float` et ne renvoie rien ». `key=len` applique `len` à chaque élément avant de comparer.

</details>
<details><summary>Indice 3</summary>

5 : `make_adder(3)` renvoie la fonction `lambda x: x + 3`, qui se souvient de `n = 3` ; on l'appelle ensuite avec 4.

</details>

### 0A.Q8 — Quel outil de la bibliothèque standard pour quelle tâche ?

<details><summary>Indice 1</summary>

Relis la liste des modules de la fiche §100.6 : chemins, fichiers, texte, combinaisons, comptages.

</details>
<details><summary>Indice 2</summary>

Chemins : un module moderne remplace `os.path`. Fichier lisible par un humain : pas `pickle`. Motif dans un texte : expressions régulières. Paires : un module d'itérateurs. « Les 5 meilleurs » : une fonction du module `heapq`… ou `sorted`.

</details>
<details><summary>Indice 3</summary>

Les six réponses, dans le désordre : `collections.Counter`, `re.findall`, `pathlib.Path`, `heapq.nlargest` (ou `sorted(...)[:5]`), `json.dump`, `itertools.combinations`. À toi de les associer.

</details>

### 0A.Q9 — Classes : `self`, méthodes spéciales, héritage, `yield`

<details><summary>Indice 1</summary>

Les méthodes spéciales ont des noms entourés de deux tirets bas (`__len__`…) : Python les appelle pour toi quand tu utilises un opérateur ou une fonction intégrée.

</details>
<details><summary>Indice 2</summary>

`3 * obj` : Python essaie d'abord la multiplication de `int`, puis la multiplication « à droite » de l'objet (le `r` de *right*). Un générateur produit ses valeurs une seule fois.

</details>
<details><summary>Indice 3</summary>

2 : `__len__`, `__call__`, `__rmul__`. 5 : pense à l'écriture `model.fit(X, y).predict(X_new)`.

</details>

### 0A.Q10 — NumPy : `dtype`, `shape`, vue ou copie, images

<details><summary>Indice 1</summary>

Un array n'a qu'un seul type d'éléments : NumPy choisit le type « le plus large » qui convient à tous.

</details>
<details><summary>Indice 2</summary>

Une tranche `a[2:4]` regarde la même mémoire ; un masque booléen fabrique un nouvel array. Un lot d'images en NumPy : `(nombre d'images, hauteur, largeur)`.

</details>
<details><summary>Indice 3</summary>

1 : un seul flottant suffit pour que tout devienne flottant. 2 : 64 images de 28 × 28 pixels, sans axe de canal en NumPy. 4 : les pixels bruts tiennent sur un octet non signé.

</details>

### 0A.Q11 — `axis`, broadcasting, `reshape` et graine

<details><summary>Indice 1</summary>

`axis=0` fait disparaître l'axe 0 (les lignes) : il reste une valeur par colonne.

</details>
<details><summary>Indice 2</summary>

Broadcasting : compare les formes en partant de la droite ; deux dimensions sont compatibles si elles sont égales ou si l'une vaut 1 (ou manque). `-1` dans `reshape` : « calcule cette dimension pour moi ».

</details>
<details><summary>Indice 3</summary>

1 : `(4,)`. 3 : `(333, 4)` et `(4,)` : la dimension manquante à gauche est ajoutée. 5 : c'est tout l'intérêt d'une graine.

</details>

### 0A.Q12 — pandas, matplotlib, git, pytest : quelle commande pour quoi ?

<details><summary>Indice 1</summary>

Chaque commande de la liste sert exactement une fois.

</details>
<details><summary>Indice 2</summary>

Repère les mots-clés : `isna` (manquant), `groupby` (par groupe), `subplots` (grille), `diff` (différences), `switch -c` (créer et changer), `raises` (erreur attendue).

</details>
<details><summary>Indice 3</summary>

1 ↔ `df.isna().sum()` ; 3 ↔ `plt.subplots(2, 3)` ; associe les quatre autres de la même façon.

</details>

<a id="papier"></a>

## ✏️ Papier-crayon

### Ex 0A.1 — Évaluer des expressions à la main

<details><summary>Indice 1</summary>

`//` est le quotient de la division euclidienne, `%` le reste, `**` la puissance ; `int` **tronque** vers zéro, `round` arrondit (au pair le plus proche en cas d'égalité : `round(2.5)` vaut 2).

</details>
<details><summary>Indice 2</summary>

c : `//` arrondit vers **moins l'infini**, pas vers zéro. d : parenthèses, puis puissance, puis `//`, puis la soustraction. g : `/` renvoie toujours le même type.

</details>
<details><summary>Indice 3</summary>

a et b : $17 = 5 \times 3 + 2$. c : $-17 / 5 = -3{,}4$ ; prends l'entier juste **en dessous** (attention au sens, on est dans les négatifs). e : `int(7.9)` vaut 7 ; `round(7.5)` : 7 ou 8, lequel est pair ? f : de gauche à droite, `10 / 4` vaut 2.5.

</details>

### Ex 0A.2 — Indices et tranches à la main

<details><summary>Indice 1</summary>

Les indices commencent à 0 ; `-1` est le dernier élément ; une tranche `[début:fin]` s'arrête **avant** `fin`.

</details>
<details><summary>Indice 2</summary>

Écris les indices sous chaque élément : `3750` (0), `3800` (1), `3250` (2)… puis les indices négatifs, en partant de la fin : le dernier élément a l'indice −1. Même chose pour les lettres de `"Chinstrap"`.

</details>
<details><summary>Indice 3</summary>

c : indices 1, 2 et 3. d : un élément sur deux à partir de l'indice 0. f : les quatre dernières lettres. g : de l'indice 2 à la fin, il reste `6 - 2` éléments. h : `181 - 39.1`.

</details>

### Ex 0A.3 — Dérouler une boucle et une compréhension pas à pas

<details><summary>Indice 1</summary>

Fais un tableau : une ligne par tour, une colonne pour `i`, `v`, « pair ? », « > 5 ? », `total`, `count`.

</details>
<details><summary>Indice 2</summary>

Le `elif` n'est testé **que si** `v` est impair. Pour la boucle `while`, note `n` et `steps` après chaque tour : `100 → 33 → …`.

</details>
<details><summary>Indice 3</summary>

a : les valeurs paires sont 4 (i=0), 8 (i=3) et 6 (i=5) : `total = 4×0 + 8×3 + 6×5`. b : parmi les impairs 7, 1, 3, lesquels dépassent 5 ? e : `100 → 33 → 11 → 3 → 1`, compte les flèches.

</details>

### Ex 0A.4 — Un `groupby` à la main sur huit manchots

<details><summary>Indice 1</summary>

Un `groupby("species")` range les lignes en paquets, un par espèce, puis calcule la statistique dans chaque paquet.

</details>
<details><summary>Indice 2</summary>

Recopie les masses par espèce : Adelie (lignes 0, 1, 5), Gentoo (2, 4, 7), Chinstrap (3, 6). Pour e : garde d'abord les mâles, **puis** compte les Gentoo.

</details>
<details><summary>Indice 3</summary>

a : $(5200 + 4650 + 5550) / 3$. b : $(3750 + 3400 + 3900) / 3$. d : îles Dream = lignes 1, 3, 6. f : `idxmin` renvoie le **nom** de l'espèce dont la moyenne est la plus petite, pas la valeur.

</details>

### Ex 0A.5 — Mini-lots

<details><summary>Indice 1</summary>

Division euclidienne : $333 = 64 \times q + r$. Le quotient $q$ compte les lots **complets**, le reste $r$ la taille du lot incomplet.

</details>
<details><summary>Indice 2</summary>

Nombre de lots en gardant le dernier : $\lceil 333 / 64 \rceil$ (on arrondit vers le haut). Une mise à jour des poids a lieu après **chaque** mini-lot.

</details>
<details><summary>Indice 3</summary>

$64 \times 5 = 320$, reste $13$. a : 5 lots complets + 1 incomplet. d : (lots par epoch) × 20. e : un exemple par lot. f : un seul lot par epoch.

</details>

### Ex 0A.6 — Portée, valeurs par défaut et arguments nommés

<details><summary>Indice 1</summary>

Une variable affectée **dans** une fonction est locale à cette fonction. Demande-toi si le `rate = 5` de `price` peut modifier le `rate` défini au début du programme.

</details>
<details><summary>Indice 2</summary>

Pour chaque appel, écris les valeurs de `quantity`, `unit` et `discount`, puis calcule `quantity * unit * 5 - discount`. `discount` est après `*` : on ne peut le donner que par son nom.

</details>
<details><summary>Indice 3</summary>

a : `3 * 2 * 5 - 0`. d : les arguments nommés peuvent venir dans n'importe quel ordre. f : `price(1, 2, 3)` donne trois arguments positionnels, mais la fonction n'en accepte que deux : quelle erreur signale un mauvais appel ?

</details>

### Ex 0A.7 — Formes NumPy à la main

<details><summary>Indice 1</summary>

`A` a la forme `(4, 6)` et contient 0, 1, …, 23 ligne par ligne. Un indice entier **supprime** un axe, une tranche le **garde**.

</details>
<details><summary>Indice 2</summary>

d et e : `axis=0` supprime l'axe des lignes, `axis=1` celui des colonnes ; `keepdims=True` le garde avec la taille 1. f et g : `reshape` ne change pas le nombre total d'éléments, et `-1` le laisse calculer une dimension. h : un masque de même forme que `A` renvoie un array à une dimension (les valeurs gardées, à la suite).

</details>
<details><summary>Indice 3</summary>

b : `(4,)` mais c : `(4, 1)`. f : $24 / (2 \times 3) = 4$. h : combien de nombres de 0 à 23 sont plus grands que 20 ? i : ligne 2, colonne 3, c'est $2 \times 6 + 3$.

</details>

### Ex 0A.8 — Broadcasting

<details><summary>Indice 1</summary>

Aligne les formes **à droite**, puis compare dimension par dimension : égales → OK ; l'une vaut 1 → elle est étirée ; l'une manque → on ajoute un 1 à gauche ; sinon → erreur.

</details>
<details><summary>Indice 2</summary>

b : `(5, 3)` contre `(5,)` : on compare 3 et 5 en premier. d : `(2, 1, 3)` contre `(1, 4, 1)` après ajout d'un 1 à gauche. f : `(10,)` devient `(1, 10)`.

</details>
<details><summary>Indice 3</summary>

a : `3 = 3` → OK, résultat `(5, 3)`. c : `(5, 1)` et `(1, 4)` s'étirent l'un l'autre. g : 4 contre 3 à droite… Réponds par une chaîne : `"(5, 3)"` ou `"erreur"`.

</details>

<a id="reflexion"></a>

## 🗣️ 🛠️ Réflexion et outils

### Ex 0A.9 — Liste Python ou array NumPy : l'expliquer en cinq lignes

<details><summary>Indice 1</summary>

Pars d'un exemple concret : convertir les 344 masses de grammes en kilogrammes.

</details>
<details><summary>Indice 2</summary>

Compare les deux écritures (une boucle ou une compréhension d'un côté, `masses / 1000` de l'autre), puis dis **pourquoi** la seconde va plus vite (un seul type, des nombres rangés côte à côte, une boucle écrite en C). Termine par ce qui reste le point fort des listes.

</details>
<details><summary>Indice 3</summary>

Plan en cinq lignes : 1) une liste peut tout contenir, un array ne contient que des nombres du même type ; 2) exemple `masses / 1000` sans boucle ; 3) pourquoi c'est plus rapide ; 4) moins de mémoire ; 5) quand garder une liste.

</details>

### Ex 0A.10 — Premier commit propre depuis le terminal

<details><summary>Indice 1</summary>

Trois zones : ton dossier de travail, la zone de préparation (*staging*, remplie par `git add`) et l'historique (rempli par `git commit`).

</details>
<details><summary>Indice 2</summary>

`git config --global user.name "…"` et `user.email "…"` une seule fois ; puis `git status`, `git add <le fichier>`, `git status`, `git commit -m "…"`, `git log --oneline -3`, `git push`.

</details>
<details><summary>Indice 3</summary>

Un bon message dit ce que fait le commit : `git commit -m "0A: answer exercise 0A.9"`. En rouge : ce que git ne suit pas encore (*Untracked files*) ou ce qui est modifié mais pas préparé ; en vert : ce qui est préparé pour le prochain commit.

</details>

### Ex 0A.11 — `.gitignore` : ce qui ne doit jamais entrer dans le dépôt

<details><summary>Indice 1</summary>

On versionne ce qu'on a **écrit** et qu'on veut retrouver ; on ignore ce qui est **généré automatiquement**, **trop lourd** ou **secret**.

</details>
<details><summary>Indice 2</summary>

Classe chaque fichier : code ou notes (garder), fichier de sauvegarde automatique de Jupyter, fichier compilé par Python, secret, modèle entraîné de 180 Mo, donnée personnelle. Un motif `.gitignore` peut viser un dossier (`__pycache__/`) ou une extension (`*.pt`).

</details>
<details><summary>Indice 3</summary>

Six lignes possibles : `.ipynb_checkpoints/`, `__pycache__/`, `.env`, `*.pt`, `models/`, `data_perso/`. Pour la dernière question : `.gitignore` n'agit que sur les fichiers **pas encore suivis** ; un secret déjà commité reste dans l'historique.

</details>

### Ex 0A.12 — Une branche pour essayer sans risque

<details><summary>Indice 1</summary>

Une branche est une ligne d'historique parallèle : les commits faits sur `essai-notes` n'existent pas sur `main` tant qu'on n'a pas fusionné.

</details>
<details><summary>Indice 2</summary>

`git switch -c essai-notes`, modifie, `git add` + `git commit`, `git switch main` (regarde le fichier), `git merge essai-notes`, `git branch -d essai-notes`.

</details>
<details><summary>Indice 3</summary>

Question 5 : `-d` est prudent, `-D` force. Essaie mentalement `git branch -d` sur une branche dont les commits n'ont **pas** été fusionnés : git risquerait-il de perdre du travail ?

</details>

<a id="entretien"></a>

## 💼 Entretien

### 0A.E1 — Data analyst, data scientist, ML engineer

<details><summary>Indice 1</summary>

Distingue les postes par leur **livrable** : un tableau de bord et une analyse ; un modèle et une étude ; un modèle en production.

</details>
<details><summary>Indice 2</summary>

Pour chacun : une phrase sur la mission, deux outils typiques (SQL, BI, pandas, scikit-learn, PyTorch, Docker, cloud…). Puis ton choix, relié à ton parcours (voir `annexes/metiers.md`).

</details>
<details><summary>Indice 3</summary>

Structure : « Le data analyst répond à des questions métier avec les données existantes… le data scientist construit et évalue des modèles… le ML engineer les met en production et les maintient… Je vise … parce que … ».

</details>

### 0A.E2 — Liste, tuple, dictionnaire ou ensemble : comment choisir ?

<details><summary>Indice 1</summary>

Reprends les trois questions du quiz Q4 : ordre ? modification ? accès par clé ?

</details>
<details><summary>Indice 2</summary>

Donne un exemple « data » pour chaque structure, et ajoute un argument de performance : chercher dans un ensemble ou un dictionnaire est quasi instantané, chercher dans une liste oblige à la parcourir.

</details>
<details><summary>Indice 3</summary>

Liste : une série de mesures. Tuple : une forme `(344, 8)` ou un couple renvoyé par une fonction (et une clé de dictionnaire possible). Dictionnaire : des hyperparamètres `{"lr": 0.01}`. Ensemble : les valeurs distinctes d'une colonne.

</details>

### 0A.E3 — Pourquoi vectoriser avec NumPy plutôt qu'écrire une boucle ?

<details><summary>Indice 1</summary>

Une boucle Python traite un objet Python à la fois ; NumPy délègue la boucle à du code compilé qui travaille sur un bloc de mémoire contigu.

</details>
<details><summary>Indice 2</summary>

Trois raisons : pas de vérification de type à chaque élément, mémoire contiguë (le processeur lit vite), instructions vectorielles (SIMD). Puis les cas où la boucle reste raisonnable.

</details>
<details><summary>Indice 3</summary>

Donne un ordre de grandeur (souvent 50 à 100 fois plus rapide) et cite deux exceptions : une boucle sur quelques éléments, ou un calcul où chaque étape dépend de la précédente (une simulation pas à pas, une boucle d'entraînement sur les epochs).

</details>

### 0A.E4 — Ton notebook est-il reproductible ?

<details><summary>Indice 1</summary>

Liste tout ce qui peut différer entre deux exécutions : l'ordre des cellules, le hasard, les données, les versions des bibliothèques, la machine.

</details>
<details><summary>Indice 2</summary>

Pour chaque cause, un remède : *Restart and run all*, une graine (`np.random.default_rng(42)`, `torch.manual_seed`), des versions figées (`requirements.txt`), des chemins relatifs, les données versionnées ou téléchargées par le code.

</details>
<details><summary>Indice 3</summary>

Conclus par la limite : sur GPU, certains calculs restent légèrement non déterministes ; on peut les rendre déterministes au prix de la vitesse.

</details>

### 0A.E5 — Ton workflow git en équipe

<details><summary>Indice 1</summary>

Décris le chemin d'une modification, de `main` à `main` : branche → commits → push → *pull request* → relecture → fusion.

</details>
<details><summary>Indice 2</summary>

Cite les bonnes pratiques : une branche par fonctionnalité, de petits commits avec des messages clairs, les tests qui passent avant la *pull request*, la relecture par un collègue, la mise à jour depuis `main` avant de fusionner.

</details>
<details><summary>Indice 3</summary>

« Je pars d'un `main` à jour, je crée une branche `feature/...`, je fais de petits commits… j'ouvre une *pull request*, les tests tournent automatiquement, un collègue relit… ». Prépare une relance : « Et en cas de conflit ? ».

</details>

<a id="notebook"></a>

## Notebook, parties A à E

### Ex 0A.13 — Ordre d'exécution des cellules : que vaut `x` ? 🔮

<details><summary>Indice 1</summary>

Le noyau garde `x` en mémoire entre les cellules : chaque exécution repart de la **dernière** valeur de `x`, quel que soit l'endroit de la cellule sur la page.

</details>
<details><summary>Indice 2</summary>

a : suis la séquence ① ② ③ ② ③ en notant `x` après chaque étape. b : après un redémarrage, la mémoire est vide et « Run all » exécute chaque cellule une fois, de haut en bas.

</details>
<details><summary>Indice 3</summary>

a : `2 → 6 → 7 → 21 → …`. b : la séquence est ① ② ③, une seule fois.

</details>

### Ex 0A.14 — Nombres et f-strings : la fiche d'un manchot

<details><summary>Indice 1</summary>

Conversion d'unités : grammes → kilogrammes, divise par 1000 ; millimètres → centimètres, divise par 10. Dans une f-string, `{valeur:.1f}` affiche 1 décimale.

</details>
<details><summary>Indice 2</summary>

a : `round(..., 2)`. b : recopie le modèle `Adelie (Dream): 3.4 kg, flipper 19.0 cm` en remplaçant chaque morceau variable par `{...}`. c et d : division euclidienne de 50 000 g par la masse.

</details>
<details><summary>Indice 3</summary>

`card = f"{species} ({island}): {mass_g / 1000:.1f} kg, flipper ..."` ; `n_penguins = 50_000 // mass_g` ; `remaining_g = 50_000 % mass_g`.

</details>

### Ex 0A.15 — Chaînes : nettoyer les noms d'espèces

<details><summary>Indice 1</summary>

`split()` découpe une chaîne en mots ; `find` renvoie la position d'un caractère ; une tranche `s[i:j]` extrait un morceau.

</details>
<details><summary>Indice 2</summary>

`short_name` : le premier élément de `raw.split()`. `latin_name` : la tranche commence **juste après** `(` et s'arrête **juste avant** `)`.

</details>
<details><summary>Indice 3</summary>

```python
start = raw.find("(")
end = raw.find(")")
return raw[start + 1:end]
```
Pour c : `latin_name(raw_names[1]).replace("Pygoscelis", "P.")`.

</details>

### Ex 0A.16 — Listes : les nageoires de dix manchots

<details><summary>Indice 1</summary>

`min`, `max`, `sum`, `len` et `sorted` font tout le travail ; aucune boucle n'est nécessaire.

</details>
<details><summary>Indice 2</summary>

c : `sorted(..., reverse=True)` trie du plus grand au plus petit, puis une tranche garde les trois premiers. d : trie la liste, puis prends les éléments en 5ᵉ et 6ᵉ position, c'est-à-dire aux indices 4 et 5.

</details>
<details><summary>Indice 3</summary>

`top3 = sorted(flippers, reverse=True)[:3]` ; `ordered = sorted(flippers)` puis `flipper_median = (ordered[4] + ordered[5]) / 2`.

</details>

### Ex 0A.17 — Tuples et déballage

<details><summary>Indice 1</summary>

`return a, b` renvoie un tuple ; `x, y = un_tuple` le déballe en deux variables.

</details>
<details><summary>Indice 2</summary>

c : l'échange en une ligne s'écrit `a, b = b, a` (Python construit d'abord le tuple de droite). d : dans `head, *middle, tail = ...`, l'étoile ramasse tout ce qui n'est ni le premier ni le dernier.

</details>
<details><summary>Indice 3</summary>

`return min(values), max(values)` ; `low, high = min_max(mass_known)` ; `first_species, second_species = second_species, first_species` ; `n_middle = len(middle)` (sachant que `flippers` a 10 éléments, tu peux prévoir la réponse).

</details>

### Ex 0A.18 — Dictionnaires : une fiche par espèce

<details><summary>Indice 1</summary>

Un dictionnaire d'accumulation : la clé est l'espèce, la valeur est la liste de ses masses, qu'on allonge avec `append`.

</details>
<details><summary>Indice 2</summary>

Dans la boucle : si l'espèce n'est pas encore une clé, crée une liste vide ; puis ajoute la masse. `d.setdefault(clé, [])` fait les deux en un appel. d : parcours `mean_by_species.items()` en retenant l'espèce qui a la plus grande moyenne.

</details>
<details><summary>Indice 3</summary>

```python
masses_by_species.setdefault(species, []).append(mass)
...
mean_by_species[species] = sum(masses) / len(masses)
```
d : `max(mean_by_species, key=mean_by_species.get)` marche aussi.

</details>

### Ex 0A.19 — Ensembles : quelles espèces sur quelles îles ?

<details><summary>Indice 1</summary>

Un ensemble ne garde qu'un exemplaire de chaque valeur ; `&` garde ce qui est commun à deux ensembles.

</details>
<details><summary>Indice 2</summary>

b : recopie la compréhension de l'exemple (a) en changeant l'espèce. c : construis les îles des Chinstrap de la même façon, puis combine avec `&`. d : transforme la liste des couples en ensemble et mesure sa taille.

</details>
<details><summary>Indice 3</summary>

`both_islands = adelie_islands & chinstrap_islands` ; `n_distinct_pairs = len(set(pairs))` (les tuples peuvent entrer dans un ensemble).

</details>

### Ex 0A.20 — Conditions : classer un manchot selon sa masse

<details><summary>Indice 1</summary>

L'ordre des tests compte : la première condition vraie l'emporte. Et `None < 3500` provoque une erreur.

</details>
<details><summary>Indice 2</summary>

Teste d'abord `mass_g is None`, puis les seuils du plus petit au plus grand avec `<`. Pour 3500 g, relis bien : « de 3500 g (inclus) ».

</details>
<details><summary>Indice 3</summary>

```python
if mass_g is None:
    return "unknown"
if mass_g < 3500:
    return "light"
elif mass_g < 4500:
    ...
```

</details>

### Ex 0A.21 — Boucles : `for`, `range`, `enumerate`, `zip` et `while`

<details><summary>Indice 1</summary>

`zip` parcourt deux listes en parallèle, `enumerate` donne l'indice avec l'élément, `while` répète tant qu'une condition est vraie, `break` sort de la boucle.

</details>
<details><summary>Indice 2</summary>

a : accumule `mass / 1000` quand l'espèce est Chinstrap **et** la masse n'est pas `None`. c : compte les tours pendant que la colonie est `<= 1000`. d : `range(0, 101, 2)` (la fin 101 est exclue, 100 est donc inclus). e : `list.index` renvoie la position de la première occurrence.

</details>
<details><summary>Indice 3</summary>

```python
colony, years = 344, 0
while colony <= 1000:
    colony = colony * 1.08
    years += 1
```
b :
```python
for i, mass in enumerate(mass_list):
    if mass is not None and mass > 6000:
        first_heavy_index = i
        ...  # stop the loop here
```

</details>

### Ex 0A.22 — Compréhensions : filtrer et transformer en une ligne

<details><summary>Indice 1</summary>

Forme générale : `[expression for élément in itérable if condition]` ; avec des accolades `{clé: valeur for ...}` pour un dictionnaire, `{expression for ...}` pour un ensemble.

</details>
<details><summary>Indice 2</summary>

a : la condition doit d'abord écarter `None` (`m is not None and m >= 6000`). c : parcours `zip(species_list, flipper_list)` avec deux variables. d : `.upper()` sur chaque île.

</details>
<details><summary>Indice 3</summary>

`heavy_kg = [m / 1000 for m in mass_list if m is not None and m >= 6000]` ; `counts = {sp: species_list.count(sp) for sp in sorted(set(species_list))}` ; `gentoo_cm = [f / 10 for sp, f in zip(...) if ...]`.

</details>

### Ex 0A.23 — Tes premières fonctions

<details><summary>Indice 1</summary>

Une fonction **renvoie** son résultat avec `return` ; `print` ne fait qu'afficher. Les paramètres avec `=` ont une valeur par défaut utilisée quand on ne les donne pas.

</details>
<details><summary>Indice 2</summary>

`flipper_per_kg` : convertis la masse en kg avant de diviser. `describe` : un `if unit == "kg"` / `elif unit == "g"` / `else: raise ValueError(...)`, puis une f-string.

</details>
<details><summary>Indice 3</summary>

```python
if unit == "kg":
    value = round(mass_g / 1000, decimals)
elif unit == "g":
    value = int(mass_g)
else:
    raise ValueError(f"unit must be 'kg' or 'g', not {unit!r}")
return f"{species}: {value} {unit}"
```

</details>

### Ex 0A.24 — Importer des modules

<details><summary>Indice 1</summary>

Chaque question a sa fonction toute faite, citée dans l'énoncé : l'exercice consiste à l'importer et à lire ce qu'elle renvoie.

</details>
<details><summary>Indice 2</summary>

`math.ceil` arrondit vers le haut. `Counter(...).most_common(1)` renvoie une **liste** contenant un seul couple `(valeur, nombre)`.

</details>
<details><summary>Indice 3</summary>

`n_batches = math.ceil(344 / 50)` ; `top_island, top_count = Counter(island_list).most_common(1)[0]` ; `mass_stdev = statistics.stdev(mass_known)`.

</details>

### Ex 0A.25 — Exceptions : lever une `ValueError` et la rattraper

<details><summary>Indice 1</summary>

On **lève** une erreur là où on détecte le problème (`raise ValueError("...")`), on la **rattrape** là où on sait quoi faire (`try` / `except ValueError`).

</details>
<details><summary>Indice 2</summary>

`parse_mass` : `text.strip()`, test du texte vide, `float(text)` (qui lève seul une `ValueError` si le texte n'est pas un nombre), test du signe. `parse_all` : une boucle avec un `try` autour de l'appel, et `None` dans le `except`.

</details>
<details><summary>Indice 3</summary>

```python
for text in texts:
    try:
        masses.append(parse_mass(text))
    except ValueError:
        masses.append(None)
```
Rappel : `"4.2e3"` est une écriture scientifique valide (4200).

</details>

### Ex 0A.26 — Ton premier module mylearn : `mean` et son test

<details><summary>Indice 1</summary>

La docstring de `mean` est ton cahier des charges : moyenne = somme / nombre d'éléments, et une `ValueError` si la séquence est vide.

</details>
<details><summary>Indice 2</summary>

Teste le cas vide avec `len(values) == 0` (pas avec `if not values`, qui plante sur un array NumPy), puis renvoie `sum(values) / len(values)`. N'oublie pas d'**enregistrer** le fichier et de **redémarrer le noyau**.

</details>
<details><summary>Indice 3</summary>

```python
if len(values) == 0:
    raise ValueError("mean() of an empty sequence")
return ...
```
Si les tests échouent encore, lis la **dernière** ligne de l'erreur pytest : elle dit quel test et pourquoi.

</details>

### Ex 0A.27 — Premiers arrays NumPy

<details><summary>Indice 1</summary>

`np.array(liste_de_listes)` crée un tableau à deux dimensions ; ses attributs `shape`, `dtype`, `ndim` et `size` décrivent sa forme et son contenu.

</details>
<details><summary>Indice 2</summary>

`shape` est un tuple `(lignes, colonnes)` ; `dtype.name` donne le nom du type sous forme de chaîne ; `size` = produit des dimensions. e : `np.linspace(0, 1, 11)` découpe [0, 1] en 10 intervalles égaux.

</details>
<details><summary>Indice 3</summary>

`X = np.array(table)` ; `x_shape, x_dtype, x_ndim, x_size = X.shape, X.dtype.name, X.ndim, X.size` ; e : les valeurs sont 0, 0.1, 0.2, …

</details>

### Ex 0A.28 — Indexation, tranches et masques booléens

<details><summary>Indice 1</summary>

`X[ligne, colonne]` ; `X[:, j]` est une colonne entière ; une comparaison sur une colonne donne un **masque** de `True`/`False`.

</details>
<details><summary>Indice 2</summary>

b : les indices 100 à 199 inclus s'écrivent `100:200`. c : combine deux masques avec `&`, **chaque** condition entre parenthèses, puis `.sum()` compte les `True`. d : « au moins 4500 g » s'écrit `>= 4500` ; `.mean()` d'un masque donne la proportion de `True`.

</details>
<details><summary>Indice 3</summary>

`X[10, 2]` ; `X[100:200, 3].mean()` ; `((X[:, 0] > 45) & (X[:, 2] < 200)).sum()` ; `(X[:, 3] >= 4500).mean()`.

</details>

### Ex 0A.29 — Vue ou copie : qui est modifié ? 🔮

<details><summary>Indice 1</summary>

Une **tranche** (`a[2:5]`) est une vue : elle partage la mémoire de `a`. Une **liste d'indices** (`a[[0, 1]]`) fabrique une copie.

</details>
<details><summary>Indice 2</summary>

Écris `a` au départ : `[0 1 2 3 4 5]`. `b[0]` est la même case que `a[2]`. `c` est un nouvel array : le modifier ne change pas `a`.

</details>
<details><summary>Indice 3</summary>

À la fin, `a` vaut `[0 1 100 3 4 5]` ; il reste à lire `a[2]`, `a[0]` et à faire la somme.

</details>

### Ex 0A.30 — Calcul vectorisé

<details><summary>Indice 1</summary>

Une opération entre une colonne et un nombre (ou entre deux colonnes) s'applique à tous les éléments d'un coup.

</details>
<details><summary>Indice 2</summary>

b : `f = X[:, 2]`, puis la formule de l'énoncé avec `f.min()` et `f.max()`. c : divise la colonne 0 par la colonne 1. d : `np.where(condition, valeur_si_vrai, valeur_si_faux)`, puis compte avec `(labels == "heavy").sum()`.

</details>
<details><summary>Indice 3</summary>

`mean_kg = (X[:, 3] / 1000).mean()` ; `flipper_scaled = (f - f.min()) / (f.max() - f.min())` ; `labels = np.where(X[:, 3] >= 4500, "heavy", "not heavy")` (même règle qu'en 0A.20).

</details>

### Ex 0A.31 — Aléatoire reproductible

<details><summary>Indice 1</summary>

Un générateur `rng = np.random.default_rng(graine)` donne toujours la même suite de nombres pour la même graine ; `integers(1, 7)` exclut 7.

</details>
<details><summary>Indice 2</summary>

a : la proportion de 6 est la moyenne du masque `rolls == 6`. d : crée **deux** générateurs avec la même graine et compare leur premier `.random()`. e : `choice(len(X), size=50, replace=False)` renvoie 50 indices distincts ; utilise-les pour indexer les lignes de `X`.

</details>
<details><summary>Indice 3</summary>

`rolls = rng.integers(1, 7, size=10_000)` ; `share_six = (rolls == 6).mean()` ; `idx = np.random.default_rng(42).choice(len(X), size=50, replace=False)` puis `X[idx, 2].mean()`. Attention : chaque appel à `rng` fait avancer le générateur, respecte l'ordre de l'énoncé.

</details>

### Ex 0A.32 — Premier contact avec Penguins

<details><summary>Indice 1</summary>

`pd.read_csv(chemin)` lit le fichier ; `info()` affiche le nombre de valeurs non manquantes par colonne ; `describe()` renvoie un tableau de statistiques des colonnes numériques.

</details>
<details><summary>Indice 2</summary>

b : `df["sex"].count()` compte les valeurs non manquantes. c et d : lis `describe()` avec `.loc["mean", "body_mass_g"]` et `.loc["max", ...]`. e : le nombre de **colonnes** du tableau renvoyé par `describe()`.

</details>
<details><summary>Indice 3</summary>

`df = pd.read_csv(wb.datasets.data_dir() / "penguins.csv")` ; `summary = df.describe()` ; `n_numeric = summary.shape[1]`. N'oublie pas `year`, qui est aussi numérique.

</details>

### Ex 0A.33 — Sélectionner : colonnes, `loc`, `iloc` et filtres

<details><summary>Indice 1</summary>

`loc` sélectionne par **étiquette** (le nom de l'index et des colonnes), `iloc` par **position** ; un filtre est un masque booléen placé entre crochets.

</details>
<details><summary>Indice 2</summary>

a : `penguins.loc[100, "island"]`. b : la position `-1` avec `iloc`. c : deux conditions avec `&` et des parenthèses, puis `len` ou `.sum()` du masque. d et e : `penguins.loc[masque, "colonne"].mean()`.

</details>
<details><summary>Indice 3</summary>

`((penguins["species"] == "Gentoo") & (penguins["body_mass_g"] > 5500)).sum()` ; `penguins.loc[penguins["island"] == "Dream", "bill_length_mm"].mean()`.

</details>

### Ex 0A.34 — Valeurs manquantes et doublons

<details><summary>Indice 1</summary>

`isna()` renvoie un tableau de `True`/`False` ; `sum()` compte les `True` **par colonne**.

</details>
<details><summary>Indice 2</summary>

a : une première somme par colonne, une seconde pour le total. b et c : `len(...)` après `dropna()`, avec ou sans `subset`. d : `fillna("unknown")` sur la colonne `sex`, puis compte. e : `duplicated()` marque chaque ligne déjà vue plus haut. f : remplace par `flipper.median()` avant de faire la moyenne.

</details>
<details><summary>Indice 3</summary>

`penguins.isna().sum().sum()` ; `len(penguins.dropna(subset=["body_mass_g"]))` ; `doubled.duplicated().sum()` ; `flipper.fillna(flipper.median()).mean()`.

</details>

### Ex 0A.35 — De pandas à NumPy : construire `X` et `y`

<details><summary>Indice 1</summary>

Convention : `X` est un tableau à deux dimensions (un exemple par ligne), `y` un vecteur à une dimension (un label par exemple).

</details>
<details><summary>Indice 2</summary>

Enlève les lignes incomplètes avec `dropna()`, sélectionne les colonnes avec une **liste** de noms (`clean[numeric_cols]`) et l'espèce avec un seul nom (`clean["species"]`), puis convertis avec `.to_numpy()`.

</details>
<details><summary>Indice 3</summary>

```python
clean = penguins.dropna()
X = clean[numeric_cols].to_numpy()
y = clean["species"].to_numpy()
```

</details>

### Ex 0A.36 — Premiers graphiques : `plot`, `scatter`, `hist`

<details><summary>Indice 1</summary>

Chaque graphique suit le même moule : `fig, ax = plt.subplots()`, un appel de tracé sur `ax`, les noms des axes, `plt.show()`.

</details>
<details><summary>Indice 2</summary>

1 : `ax.hist(..., bins=25)` sur les masses **sans** les valeurs manquantes. 2 : une boucle `for species, group in penguins.groupby("species"):` avec un `ax.scatter(..., label=species)` par espèce, puis `ax.legend()`. 3 : une boucle sur `yearly.columns` avec `ax.plot(yearly.index, yearly[species], label=species)`.

</details>
<details><summary>Indice 3</summary>

```python
fig, ax = plt.subplots()
for species, group in penguins.groupby("species"):
    ax.scatter(group["bill_length_mm"], group["bill_depth_mm"], label=species)
ax.set_xlabel("bill length (mm)")
ax.set_ylabel("bill depth (mm)")
ax.legend()
plt.show()
```

</details>
