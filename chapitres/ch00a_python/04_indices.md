# 0A · Python, notebooks et outils — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [✏️ Papier-crayon](#papier) · [🗣️🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à H](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 0A.Q1 — Cellules, noyau, « Tout exécuter » et terminal : vrai ou faux

<details><summary>Indice 1</summary>

Le noyau est une mémoire qui vit à côté de la page : il ne « voit » pas la page, il reçoit seulement les cellules que tu exécutes.

</details>
<details><summary>Indice 2</summary>

Pour 1 à 3 : demande-toi ce qui se passe dans la mémoire du noyau quand on exécute, supprime ou redémarre. Pour 4 et 5 : relis la fiche §100.1.3 : que désigne `..` ? D'où part un chemin absolu ?

</details>
<details><summary>Indice 3</summary>

1 est **faux** : le noyau ne lit pas la page, il garde ce que les cellules ont laissé en mémoire, dans l'ordre où tu les as **exécutées** (relance une cellule du haut après une cellule du bas : elle repart des valeurs laissées par celle du bas). Juge les quatre autres de la même façon. 2 : supprimer une cellule touche-t-il la mémoire du noyau, ou seulement la page ? 3 : avec quelle mémoire *Restart and run all* commence-t-il, et dans quel ordre exécute-t-il les cellules ? 4 et 5 : réponds aux deux questions de l'indice 2, puis compare avec `cd ..` et avec le début de `data/penguins.csv`.

</details>

### 0A.Q2 — Le workbook : où j'écris, comment je vérifie

<details><summary>Indice 1</summary>

Relis le schéma des dossiers (fiche §100.1.2) et la section sur mylearn (§100.11.5).

</details>
<details><summary>Indice 2</summary>

Un seul dossier t'appartient ; les autres sont réécrits par les mises à jour. ⏳ n'est pas une erreur : c'est un état. Pour 4 : un test ne contient pas de réponses écrites à la main ; demande-toi qui calcule la valeur attendue.

</details>
<details><summary>Indice 3</summary>

1 : `mon_travail/`, le seul dossier que les mises à jour ne touchent jamais (le dossier dont le nom dit « mon »). 2 : tant que tu n'as rien écrit, que vaut encore ta réponse, ou que lève encore ta fonction ? 3 : sans `wb.attempt`, une fonction pas encore écrite lève `NotImplementedError` : que deviendrait « Tout exécuter » ? 4 : le paragraphe « mylearn » de la fiche §100.1.2 et le tableau du §100.11.5 disent à quoi les tests comparent ta fonction. 5 : relis, au §100.1.2, ce que `start_chapter.py` fait d'un fichier **déjà présent** dans `mon_travail/`.

</details>

### 0A.Q3 — Que vaut cette expression ? Types et opérateurs

<details><summary>Indice 1</summary>

`/` et `//` ne font pas la même division ; un calcul qui mélange `int` et `float` donne un `float`.

</details>
<details><summary>Indice 2</summary>

Pour 2 : `*` et `//` ont la même priorité et se calculent de gauche à droite, `%` aussi ; `+` vient après. Pour 3 : sur une chaîne, `*` répète.

</details>
<details><summary>Indice 3</summary>

1 : `/` donne toujours un `float`, même quand la division tombe juste : `9 / 3` vaut `3.0`, de type `float`. 2 : calcule à part `9 // 4` et `9 % 4` (le quotient et le reste de la division euclidienne de 9 par 4), puis fais `*` avant `+`. 3 : combien de fois `"ab"` est-il répété, et quel est le type d'un texte ? 4 : convertis chaque morceau, puis applique la règle de l'indice 1 sur `int` et `float`. 5 : `==` compare des valeurs, pas des types ; et quel type a le résultat d'une comparaison ?

</details>

### 0A.Q4 — Liste, tuple, dict, set ou Counter ?

<details><summary>Indice 1</summary>

Pose-toi trois questions : l'ordre compte-t-il ? La taille change-t-elle ? Cherche-t-on une valeur à partir d'une clé ?

</details>
<details><summary>Indice 2</summary>

Liste : ordonnée, modifiable. Tuple : ordonné, figé. Dictionnaire : clé → valeur. Ensemble : sans doublon, sans ordre. `Counter` : un dictionnaire spécialisé dans les comptages.

</details>
<details><summary>Indice 3</summary>

1 : l'ordre compte et la taille change (on ajoute les mesures au fur et à mesure) : une **liste**. Pour les quatre autres, pose les trois questions de l'indice 1, puis compare avec les définitions de l'indice 2. 2 : deux nombres figés, qui ne changeront plus. 3 : on cherche le nom latin **à partir** du nom court. 4 : l'ordre ne compte pas, et chaque nom ne doit apparaître qu'une fois. 5 : un nombre à associer à chaque île, obtenu en comptant.

</details>

### 0A.Q5 — Conditions, boucles, compréhensions : qu'affiche ce code ?

<details><summary>Indice 1</summary>

`range(début, fin, pas)` s'arrête **avant** `fin` ; un `if`/`elif`/`else` n'exécute qu'une seule branche.

</details>
<details><summary>Indice 2</summary>

1 : écris 1, puis ajoute 3 tant que tu restes sous 8. 3 : parcours `"manchot"` lettre par lettre et garde les voyelles. 5 : quels nombres de 0 à 9 sont divisibles par 3 ?

</details>
<details><summary>Indice 3</summary>

1 : `range(1, 8, 3)` part de 1, avance de 3 en 3 et s'arrête **avant** 8 : `1 4 7`. 2 : la première condition vraie gagne, les suivantes ne sont même pas testées. 3 : garde, dans l'ordre, chaque lettre de `"manchot"` qui est dans `"aeiou"` ; `print` affiche une liste. 4 : `enumerate` donne des couples (indice, élément) en commençant à 0. 5 : écris les nombres de 0 à 9 divisibles par 3, sans oublier 0, puis compte-les.

</details>

### 0A.Q6 — Paramètres, valeurs par défaut, `*args` et `**kwargs`

<details><summary>Indice 1</summary>

Les arguments positionnels remplissent les paramètres dans l'ordre ; le surplus part dans `args` (un tuple) ; les arguments nommés inconnus partent dans `kwargs` (un dictionnaire).

</details>
<details><summary>Indice 2</summary>

`c` est après `*args` : on ne peut le donner que **par son nom**. `a` n'a pas de valeur par défaut : il est obligatoire.

</details>
<details><summary>Indice 3</summary>

1 : `f(1)` remplit `a` ; `b` garde sa valeur par défaut, `args` reste vide, `c` garde 3 et `kwargs` reste vide : `(1, 2, (), 3, {})`. 2 : remplis `a` puis `b` dans l'ordre ; le surplus part dans `args`. 3 : `c=9` vise un paramètre qui existe, `d=4` non : où va-t-il ? 4 : `a`, obligatoire, n'est pas donné : l'appel peut-il aboutir ? Sinon, l'exemple `train([1], [0], 0.01)` de la fiche §100.5.2 montre quelle erreur Python lève pour un appel qui ne respecte pas la signature. 5 : relis la fiche §100.5.1 : que renvoie une fonction qui arrive à sa fin sans `return` ?

</details>

### 0A.Q7 — Lire une signature : annotations, `Callable`, `lambda`, fermeture

<details><summary>Indice 1</summary>

Dans une signature, `= None` donne une valeur par défaut, et `*` seul marque le début des paramètres « uniquement nommés ».

</details>
<details><summary>Indice 2</summary>

`Callable[[A, B], R]` se lit : « une fonction qui prend un `A` et un `B` et renvoie un `R` » ; si `R` vaut `None`, elle ne renvoie rien. `key=` reçoit une fonction, appliquée à chaque élément avant la comparaison : c'est **son résultat** qu'on compare.

</details>
<details><summary>Indice 3</summary>

1 : `y` a une valeur par défaut (`= None`) : on peut l'omettre, il n'est donc **pas** obligatoire. 2 : où est `callback` par rapport au `*` seul (indice 1) ? 3 : remplace `A`, `B` et `R` de l'indice 2 par ce qui est écrit dans la signature. 4 : que renvoie `len` pour un mot ? 5 : `make_adder(3)` renvoie la fonction `lambda x: x + 3`, qui se souvient de `n = 3` ; on l'appelle ensuite avec 4.

</details>

### 0A.Q8 — Quel outil de la bibliothèque standard pour quelle tâche ?

<details><summary>Indice 1</summary>

Relis la liste des modules de la fiche §100.6 : chemins, fichiers, texte, combinaisons, comptages.

</details>
<details><summary>Indice 2</summary>

Chemins : un module moderne remplace `os.path`. Fichier lisible par un humain : pas `pickle`. Motif dans un texte : expressions régulières. Paires : un module d'itérateurs. « Les 5 meilleurs » : un module trouve les *k* plus grands sans tout trier.

</details>
<details><summary>Indice 3</summary>

1 : `pathlib` : `Path("data") / "penguins.csv"` assemble le chemin avec le bon séparateur sur chaque système. Pour les cinq autres, trouve la ligne du tableau des modules de la fiche §100.6.1 qui correspond au besoin, puis la fonction dans la section indiquée. 2 : sérialiser en **texte** (§100.6.4). 3 : extraire un motif (§100.6.5). 4 : des combinaisons sans ordre (§100.6.6). 5 : un *top-k* (§100.6.6). 6 : un comptage (§100.3.5).

</details>

### 0A.Q9 — Classes : `self`, méthodes spéciales, héritage, `yield`

<details><summary>Indice 1</summary>

Les méthodes spéciales ont des noms entourés de deux tirets bas (`__len__`…) : Python les appelle pour toi quand tu utilises un opérateur ou une fonction intégrée.

</details>
<details><summary>Indice 2</summary>

`3 * obj` : Python essaie d'abord la multiplication de `int`, puis la multiplication « à droite » de l'objet (le `r` de *right*). Pour un générateur, relis la fin de la fiche §100.7.4 : que donne un deuxième `list(gen)` ?

</details>
<details><summary>Indice 3</summary>

1 : `self` est l'objet sur lequel la méthode est appelée : dans `pingu.mass_kg()`, `self` est `pingu`. 2 : chaque méthode spéciale porte le nom de l'opération, entouré de `__` : cherche celui de `len`, celui de l'appel (*call*) et celui de la multiplication « à droite » (indice 2). 3 : la classe mère `A` a son propre `__init__` : qui l'exécute quand `B` définit le sien ? 4 : que reste-t-il à produire dans un générateur après un premier parcours complet ? 5 : si `fit` renvoie l'objet lui-même, que peut-on écrire juste après `model.fit(X, y)`, sur la même ligne ?

</details>

### 0A.Q10 — NumPy : `dtype`, `shape`, vue ou copie, images

<details><summary>Indice 1</summary>

Un array n'a qu'un seul type d'éléments : NumPy choisit le type « le plus large » qui convient à tous.

</details>
<details><summary>Indice 2</summary>

Pour 3, relis « Vues et copies » (fiche §100.8.2) : quelles façons d'indexer partagent la mémoire de l'array d'origine, et lesquelles en fabriquent un nouveau ? Un batch d'images en NumPy : `(nombre d'images, hauteur, largeur)`.

</details>
<details><summary>Indice 3</summary>

1 : un seul flottant suffit pour que tout l'array devienne flottant : `float64`. 2 : remplis le format de l'indice 2 avec les nombres de l'énoncé, sans axe de canal. 3 : `a[2:4]` est une tranche, `a[a > 0]` un masque : classe chacun avec la règle de la fiche. 4 : le tableau des attributs de la fiche §100.8.1 donne le type des pixels et ses valeurs possibles. 5 : `shape` se lit `(lignes, colonnes)` : que représente une ligne de `X`, et une colonne ?

</details>

### 0A.Q11 — `axis`, broadcasting, `reshape` et graine

<details><summary>Indice 1</summary>

`axis=0` fait disparaître l'axe 0 (les lignes) : il reste une valeur par colonne.

</details>
<details><summary>Indice 2</summary>

Broadcasting : compare les formes en partant de la droite ; deux dimensions sont compatibles si elles sont égales ou si l'une vaut 1 (ou manque). `-1` dans `reshape` : « calcule cette dimension pour moi ».

</details>
<details><summary>Indice 3</summary>

1 : `axis=0` fait disparaître l'axe des 333 lignes : il reste une moyenne par colonne, `(4,)`. 2 : quel axe disparaît avec `axis=1`, et combien de valeurs restent ? 3 : aligne à droite `(333, 4)` et la forme trouvée en 1, puis compare-les axe par axe avec la règle de l'indice 2 (une dimension manquante à gauche compte comme 1). 4 : `-1` laisse NumPy calculer la dimension : 12 éléments, rangés sur 1 colonne. 5 : un générateur pseudo-aléatoire est un calcul : mêmes entrées, même suite de nombres. Quelle est son entrée ?

</details>

### 0A.Q12 — pandas, matplotlib, git, pytest : quelle commande pour quoi ?

<details><summary>Indice 1</summary>

Chaque commande de la liste sert exactement une fois.

</details>
<details><summary>Indice 2</summary>

Chaque commande contient un mot-clé anglais qui dit ce qu'elle fait ; pour chacune, cherche le besoin qui correspond à sa traduction. `isna` : *is NA*, « est manquant » ; `groupby` : « regrouper selon » ; `subplots` : « sous-graphiques » ; `diff` : « différence » ; `switch` : « basculer » ; `raises` : « lève » (une exception).

</details>
<details><summary>Indice 3</summary>

1 ↔ `df.isna().sum()` : `isna` marque chaque valeur manquante, `sum` les compte colonne par colonne. Associe les cinq autres de la même façon, grâce aux mots-clés de l'indice 2 ; chaque commande ne sert qu'une fois.

</details>

<a id="papier"></a>

## ✏️ Papier-crayon

### Ex 0A.1 — Évaluer des expressions à la main : `//`, `%`, `**`, conversions ✏️

<details><summary>Indice 1</summary>

`//` est le quotient de la division euclidienne, `%` le reste, `**` la puissance ; `int` **tronque** vers zéro, `round` arrondit (au pair le plus proche en cas d'égalité : `round(2.5)` vaut 2).

</details>
<details><summary>Indice 2</summary>

c : `//` arrondit vers **moins l'infini**, pas vers zéro. d : parenthèses, puis puissance, puis `//`, puis la soustraction. g : `/` renvoie toujours le même type.

</details>
<details><summary>Indice 3</summary>

a : 5 tient 3 fois entièrement dans 17 ($5 \times 3 = 15$, et $5 \times 4 = 20$ dépasse) : `17 // 5` vaut 3. b : ce qui reste de 17 une fois retirés ces $5 \times 3$. c : $-17 / 5 = -3{,}4$ ; prends l'entier juste **en dessous** (attention au sens, on est dans les négatifs). d : $(2 + 3)^2$ d'abord, puis `10 // 3`, puis la soustraction. e : `int(7.9)` vaut 7 ; `round(7.5)` : 7 ou 8, lequel est pair ? f : de gauche à droite, $(10 / 4) \times 2$ ; et quel type donne `/` ? g : même question que la fin de f.

</details>

### Ex 0A.2 — Indices et tranches à la main : listes, tuples, chaînes ✏️

<details><summary>Indice 1</summary>

Les indices commencent à 0 ; `-1` est le dernier élément ; une tranche `[début:fin]` s'arrête **avant** `fin`.

</details>
<details><summary>Indice 2</summary>

Écris les indices sous chaque élément : `3750` (0), `3800` (1), `3250` (2)… puis les indices négatifs, en partant de la fin : le dernier élément a l'indice −1. Même chose pour les lettres de `"Chinstrap"`.

</details>
<details><summary>Indice 3</summary>

a : les indices partent de 0, donc `masses[2]` est le **3ᵉ** élément : `3250`. b : compte depuis la fin : −1, puis −2. c : indices 1, 2 et 3. d : un élément sur deux à partir de l'indice 0. e : les lettres d'indices 0 à 4. f : les quatre dernières lettres. g : de l'indice 2 à la fin, il reste `6 - 2` éléments. h : `181 - 39.1`.

</details>

### Ex 0A.3 — Dérouler une boucle et une compréhension pas à pas ✏️

<details><summary>Indice 1</summary>

Fais un tableau : une ligne par tour, une colonne pour `i`, `v`, « pair ? », « > 5 ? », `total`, `count`.

</details>
<details><summary>Indice 2</summary>

Le `elif` n'est testé **que si** `v` est impair. Pour la boucle `while`, note `n` et `steps` après chaque tour : `100 → 33 → …`.

</details>
<details><summary>Indice 3</summary>

a : les valeurs paires sont 4 (i=0), 8 (i=3) et 6 (i=5) : `total = 4×0 + 8×3 + 6×5`. b : parmi les impairs 7, 1, 3, lesquels dépassent 5 ? c et d : filtre d'abord (la condition après `if`), puis double ou additionne ce qui reste. e : prolonge la suite de l'indice 2, `100 → 33 → …`, en divisant par 3 (division entière) tant que `n > 1`, et compte une flèche par tour.

</details>

### Ex 0A.4 — Un `groupby` à la main sur huit manchots ✏️

<details><summary>Indice 1</summary>

Un `groupby("species")` range les lignes en paquets, un par espèce, puis calcule la statistique dans chaque paquet.

</details>
<details><summary>Indice 2</summary>

Recopie les masses par espèce : Adelie (lignes 0, 1, 5), Gentoo (2, 4, 7), Chinstrap (3, 6). Pour e : garde d'abord les mâles, **puis** compte les Gentoo.

</details>
<details><summary>Indice 3</summary>

a : $(5200 + 4650 + 5550) / 3 = 15\,400 / 3 \approx 5133{,}3$. b : $(3750 + 3400 + 3900) / 3$. c : compte les lignes dont l'île est Dream. d : parmi ces mêmes lignes, la plus grande masse. f : calcule aussi la moyenne des Chinstrap, puis compare les trois moyennes ; `idxmin` renvoie le **nom** de l'espèce dont la moyenne est la plus petite, pas la valeur.

</details>

### Ex 0A.5 — Mini-batches : combien de batches, de quelle taille, combien de mises à jour ? ✏️

<details><summary>Indice 1</summary>

Division euclidienne : $333 = 64 \times q + r$. Le quotient $q$ compte les batches **complets**, le reste $r$ la taille du batch incomplet.

</details>
<details><summary>Indice 2</summary>

Nombre de batches en gardant le dernier : $\lceil 333 / 64 \rceil$ (on arrondit vers le haut). Une mise à jour des poids a lieu après **chaque** mini-batch.

</details>
<details><summary>Indice 3</summary>

a : $333 / 64 \approx 5{,}2$ n'est pas un entier : les batches pleins ne suffisent pas, il en faut un de plus pour les exemples qui restent. On arrondit donc vers le haut : $\lceil 5{,}2 \rceil = 6$ batches. b : le reste de la division euclidienne, $333 - 64 \times q$, avec $q$ le nombre de batches pleins. c : `drop_last=True` jette le batch incomplet de a. d : (batches par epoch, trouvés en a) × 20. e : un exemple par batch : combien de batches par epoch ? f : un seul batch par epoch, pendant 20 epochs.

</details>

### Ex 0A.6 — Portée, valeurs par défaut et arguments nommés : qui vaut quoi ? ✏️

<details><summary>Indice 1</summary>

Une variable affectée **dans** une fonction est locale à cette fonction. Demande-toi si le `rate = 5` de `price` peut modifier le `rate` défini au début du programme.

</details>
<details><summary>Indice 2</summary>

Pour chaque appel, écris les valeurs de `quantity`, `unit` et `discount`, puis calcule `quantity * unit * 5 - discount`. `discount` est après `*` : on ne peut le donner que par son nom.

</details>
<details><summary>Indice 3</summary>

a : `quantity = 3`, `unit = 2` (valeur par défaut), `discount = 0`, et le `rate` **local** vaut 5 : `3 * 2 * 5 - 0`, soit 30. b et c : le même calcul avec les valeurs de chaque appel. d : les arguments nommés peuvent venir dans n'importe quel ordre. f : `price(1, 2, 3)` donne trois arguments positionnels, mais la fonction n'en accepte que deux : quelle erreur signale un mauvais appel ?

</details>

### Ex 0A.7 — Formes NumPy à la main : indexation, réductions, `reshape` ✏️

<details><summary>Indice 1</summary>

`A` a la forme `(4, 6)` et contient 0, 1, …, 23 ligne par ligne. Un indice entier **supprime** un axe, une tranche le **garde**.

</details>
<details><summary>Indice 2</summary>

d et e : `axis=0` supprime l'axe des lignes, `axis=1` celui des colonnes ; `keepdims=True` le garde avec la taille 1. f et g : `reshape` ne change pas le nombre total d'éléments, et `-1` le laisse calculer une dimension. h : un masque de même forme que `A` renvoie un array à une dimension (les valeurs gardées, à la suite).

</details>
<details><summary>Indice 3</summary>

a : `A[1:3]` garde les lignes 1 et 2 (la fin 3 est exclue) et toutes les colonnes : `(2, 6)`. b : un indice entier (`2`) retire l'axe des colonnes ; c : une tranche (`2:3`) le garde, même pour une seule colonne. d et e : quel axe disparaît, et `keepdims=True` le garde-t-il ? f : le `-1` vaut $24 / (2 \times 3)$. g : `len(B)` vaut 3, et `-1` regroupe tout le reste d'une image, $28 \times 28$ valeurs. h : combien de nombres de 0 à 23 sont plus grands que 20 ? i : ligne 2, colonne 3, c'est $2 \times 6 + 3$.

</details>

### Ex 0A.8 — Broadcasting : compatibles ou non, et quelle forme ? ✏️

<details><summary>Indice 1</summary>

Aligne les formes **à droite**, puis compare dimension par dimension : égales → OK ; l'une vaut 1 → elle est étirée ; l'une manque → on ajoute un 1 à gauche ; sinon → erreur.

</details>
<details><summary>Indice 2</summary>

b : `(5, 3)` contre `(5,)` : on compare 3 et 5 en premier. d : `(2, 1, 3)` contre `(1, 4, 1)` après ajout d'un 1 à gauche. f : `(10,)` devient `(1, 10)`.

</details>
<details><summary>Indice 3</summary>

a : à droite, 3 et 3 sont égaux ; à gauche, l'axe absent de `B` compte comme 1 et s'étire à 5 : résultat `"(5, 3)"`. Fais de même pour les six autres, en partant de la droite. b : compare d'abord 3 et 5. c : compare 1 et 4 à droite, puis 5 et 1 à gauche. d : ajoute un 1 à gauche de `(4, 1)`, puis compare axe par axe. e : l'axe absent de `B` compte comme 1. f : `(10,)` se lit `(1, 10)`. g : 4 contre 3 à droite… Réponds par une chaîne, comme `"(5, 3)"`, ou par `"erreur"`.

</details>

<a id="reflexion"></a>

## 🗣️ 🛠️ Réflexion et outils

### Ex 0A.9 — Liste Python ou array NumPy : l'expliquer en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Pars d'un exemple concret : convertir les 344 masses de grammes en kilogrammes.

</details>
<details><summary>Indice 2</summary>

Compare les deux écritures (une boucle ou une compréhension d'un côté, `masses / 1000` de l'autre), puis dis **pourquoi** la seconde va plus vite (un seul type, des nombres rangés côte à côte, une boucle écrite en C). Termine par ce qui reste le point fort des listes.

</details>
<details><summary>Indice 3</summary>

Plan en cinq lignes : 1) une liste peut tout contenir, un array ne contient que des nombres du même type ; 2) exemple `masses / 1000` sans boucle ; 3) pourquoi c'est plus rapide ; 4) moins de mémoire ; 5) quand garder une liste.

</details>

### Ex 0A.10 — Premier commit propre depuis le terminal 🛠️

<details><summary>Indice 1</summary>

Git range ton travail en trois zones : ton dossier de travail, la zone de préparation (*staging*) et l'historique. Chaque commande fait passer des fichiers d'une zone à la suivante : de quelle zone à quelle zone pour `git add` ? pour `git commit` ?

</details>
<details><summary>Indice 2</summary>

`git config --global user.name "…"` et `user.email "…"` une seule fois ; puis `git status`, `git add <le fichier>`, `git status`, `git commit -m "…"`, `git log --oneline -3`, `git push`.

</details>
<details><summary>Indice 3</summary>

Un bon message dit ce que fait le commit : `git commit -m "0A: answer exercise 0A.9"`. En rouge : ce que git ne suit pas encore (*Untracked files*) ou ce qui est modifié mais pas préparé ; en vert : ce qui est préparé pour le prochain commit.

</details>

### Ex 0A.11 — `.gitignore` : ce qui ne doit jamais entrer dans le dépôt 🛠️

<details><summary>Indice 1</summary>

On versionne ce qu'on a **écrit** et qu'on veut retrouver ; on ignore ce qui est **généré automatiquement**, **trop lourd** ou **secret**.

</details>
<details><summary>Indice 2</summary>

Classe chaque fichier : code ou notes écrits par toi, fichier de sauvegarde automatique de Jupyter, fichier compilé par Python, secret, modèle entraîné de 180 Mo, donnée personnelle. Un motif `.gitignore` peut viser un nom de fichier (`notes.tmp`), un dossier, avec un `/` final (`build/`), ou une extension, avec une étoile (`*.log`).

</details>
<details><summary>Indice 3</summary>

1 : `03_notebook.ipynb`, c'est ton travail, écrit par toi : **versionner**. Pour les sept autres, demande-toi : écrit par toi, ou fabriqué tout seul ? lourd ? secret ou personnel ? 2 : une ligne par fichier ou dossier à ignorer ; la première peut être `.ipynb_checkpoints/` (un dossier, d'où le `/` final) ; continue avec la syntaxe de l'indice 2. 3 : `.gitignore` ne concerne que les fichiers que git **ne suit pas encore** : que devient un fichier déjà commité, et que reste-t-il de son contenu dans l'historique ?

</details>

### Ex 0A.12 — Une branche pour essayer sans risque (aperçu) 🛠️

<details><summary>Indice 1</summary>

Une branche est une ligne d'historique parallèle : les commits faits sur `essai-notes` n'existent pas sur `main` tant qu'on n'a pas fusionné.

</details>
<details><summary>Indice 2</summary>

`git switch -c essai-notes`, modifie, `git add` + `git commit`, `git switch main` (regarde le fichier), `git merge essai-notes`, `git branch -d essai-notes`.

</details>
<details><summary>Indice 3</summary>

Question 3 : la ligne a disparu, car `git switch main` remet les fichiers dans l'état de `main`, où ton commit n'existe pas encore. Question 5 : l'une des deux options vérifie quelque chose avant de supprimer, l'autre non (`git branch -h` les décrit en une ligne chacune). Essaie mentalement `git branch -d` sur une branche dont les commits n'ont **pas** été fusionnés : git risquerait-il de perdre du travail ?

</details>

<a id="entretien"></a>

## 💼 Entretien

### 0A.E1 — Data analyst, data scientist, ML engineer : qui fait quoi, avec quels outils ?

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

Donne un ordre de grandeur (de quelques dizaines à quelques centaines de fois plus rapide selon l'opération) et cite deux exceptions : une boucle sur quelques éléments, ou un calcul où chaque étape dépend de la précédente (une simulation pas à pas, une boucle d'entraînement sur les epochs).

</details>

### 0A.E4 — Ton notebook est-il reproductible ? « Tout exécuter », graine, versions

<details><summary>Indice 1</summary>

Liste tout ce qui peut différer entre deux exécutions : l'ordre des cellules, le hasard, les données, les versions des bibliothèques, la machine.

</details>
<details><summary>Indice 2</summary>

Pour chaque cause, un remède : *Restart and run all*, une graine (`np.random.default_rng(42)`, `torch.manual_seed`), des versions figées (`requirements.txt`), des chemins relatifs, les données versionnées ou téléchargées par le code.

</details>
<details><summary>Indice 3</summary>

Conclus par la limite : sur GPU, certains calculs restent légèrement non déterministes ; on peut les rendre déterministes au prix de la vitesse.

</details>

### 0A.E5 — Ton workflow git : commit, branche, pull request

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

## Notebook, parties A à H

### Ex 0A.13 — Ordre d'exécution des cellules : que vaut `x` ? 🔮

<details><summary>Indice 1</summary>

Le noyau garde `x` en mémoire entre les cellules : chaque exécution repart de la **dernière** valeur de `x`, quel que soit l'endroit de la cellule sur la page.

</details>
<details><summary>Indice 2</summary>

a : suis la séquence ① ② ③ ② ③ en notant `x` après chaque étape. b : après un redémarrage, la mémoire est vide et « Tout exécuter » exécute chaque cellule une fois, de haut en bas.

</details>
<details><summary>Indice 3</summary>

a : écris la valeur de `x` sous chacune des cinq exécutions. La 4ᵉ (② de nouveau) multiplie par 3 la valeur laissée par la 3ᵉ, pas le 2 du départ ; la 5ᵉ ajoute 1 à ce résultat. b : refais le calcul de a en t'arrêtant après la 3ᵉ exécution : c'est exactement ce que fait *Restart and run all* (mémoire vide, puis ①, ② et ③ une seule fois).

</details>

### Ex 0A.14 — Nombres et f-strings : la fiche d'un manchot 🔨

<details><summary>Indice 1</summary>

Conversion d'unités : grammes → kilogrammes, divise par 1000 ; millimètres → centimètres, divise par 10. Dans une f-string, `{valeur:.1f}` affiche 1 décimale.

</details>
<details><summary>Indice 2</summary>

a : `round(..., 2)`. b : recopie le modèle `Adelie (Dream): 3.4 kg, flipper 19.0 cm` en remplaçant chaque morceau variable par `{...}`. c et d : division euclidienne de 50 000 g par la masse.

</details>
<details><summary>Indice 3</summary>

`card = f"{species} ({island}): {mass_g / 1000:.1f} kg, flipper ..."` ; `n_penguins = 50_000 // mass_g` ; `remaining_g = 50_000 % mass_g`.

</details>

### Ex 0A.15 — Chaînes : nettoyer les noms d'espèces de Penguins brut 🔨

<details><summary>Indice 1</summary>

`split()` découpe une chaîne en mots ; `find` renvoie la position d'un caractère ; une tranche `s[i:j]` extrait un morceau.

</details>
<details><summary>Indice 2</summary>

`short_name` : le premier élément de `raw.split()`. `latin_name` : la tranche commence **juste après** `(` et s'arrête **juste avant** `)`.

</details>
<details><summary>Indice 3</summary>

```python
start = raw.find("(")   # position of the opening parenthesis
end = raw.find(")")     # position of the closing one
```
Il reste à renvoyer la tranche de `raw` qui commence **une case après** `start` et s'arrête à `end` (la fin d'une tranche est déjà exclue). Pour c, la cellule de vérification applique `.replace("Pygoscelis", "P.")` à ton `latin_name(raw_names[1])`.

</details>

### Ex 0A.16 — Listes : les nageoires de dix manchots 🔨

<details><summary>Indice 1</summary>

`min`, `max`, `sum`, `len` et `sorted` font tout le travail ; aucune boucle n'est nécessaire.

</details>
<details><summary>Indice 2</summary>

c : `sorted(..., reverse=True)` trie du plus grand au plus petit, puis une tranche garde les trois premiers. d : trie la liste, puis prends les éléments en 5ᵉ et 6ᵉ position, c'est-à-dire aux indices 4 et 5.

</details>
<details><summary>Indice 3</summary>

`top3 = sorted(flippers, reverse=True)[:3]` ; `ordered = sorted(flippers)` puis `flipper_median = (ordered[4] + ordered[5]) / 2`.

</details>

### Ex 0A.17 — Tuples et déballage : renvoyer et échanger plusieurs valeurs 🔨

<details><summary>Indice 1</summary>

`return a, b` renvoie un tuple ; `x, y = un_tuple` le déballe en deux variables.

</details>
<details><summary>Indice 2</summary>

c : l'échange en une ligne s'écrit `a, b = b, a` (Python construit d'abord le tuple de droite). d : dans `head, *middle, tail = ...`, l'étoile ramasse tout ce qui n'est ni le premier ni le dernier.

</details>
<details><summary>Indice 3</summary>

a : une virgule suffit à fabriquer un tuple : `return ..., ...`, avec le minimum puis le maximum. b : `low, high = min_max(mass_known)`, puis l'écart entre les deux. c : la forme `a, b = b, a` de l'indice 2, avec tes deux noms de variables, puis `swapped` comme l'indique la cellule. d : écris d'abord `head, *middle, tail = flippers`, puis compte les éléments de `middle` (sachant que `flippers` a 10 éléments, tu peux prévoir la réponse).

</details>

### Ex 0A.18 — Dictionnaires : une fiche par espèce 🔨

<details><summary>Indice 1</summary>

Un dictionnaire d'accumulation : la clé est l'espèce, la valeur est la liste de ses masses, qu'on allonge avec `append`.

</details>
<details><summary>Indice 2</summary>

Dans la boucle : si l'espèce n'est pas encore une clé, crée une liste vide ; puis ajoute la masse. `d.setdefault(clé, [])` fait les deux en un appel. d : parcours `mean_by_species.items()` en retenant l'espèce qui a la plus grande moyenne.

</details>
<details><summary>Indice 3</summary>

Les deux lignes à écrire dans les boucles :
```python
masses_by_species.setdefault(species, []).append(mass)
...
mean_by_species[species] = sum(masses) / len(masses)
```
d :
```python
heaviest = None
for species, mean in mean_by_species.items():
    ...   # replace heaviest when it is still None, or when mean > mean_by_species[heaviest]
```

</details>

### Ex 0A.19 — Ensembles : quelles espèces sur quelles îles ? 🔨

<details><summary>Indice 1</summary>

Un ensemble ne garde qu'un exemplaire de chaque valeur ; `&` garde ce qui est commun à deux ensembles.

</details>
<details><summary>Indice 2</summary>

b : recopie la compréhension de l'exemple (a) en changeant l'espèce. c : construis les îles des Chinstrap de la même façon, puis combine avec `&`. d : transforme la liste des couples en ensemble et mesure sa taille.

</details>
<details><summary>Indice 3</summary>

`both_islands = adelie_islands & chinstrap_islands` ; `n_distinct_pairs = len(set(pairs))` (les tuples peuvent entrer dans un ensemble).

</details>

### Ex 0A.20 — Conditions : classer un manchot selon sa masse 🔨

<details><summary>Indice 1</summary>

L'ordre des tests compte : la première condition vraie l'emporte. Et `None < 3500` provoque une erreur.

</details>
<details><summary>Indice 2</summary>

Teste d'abord `mass_g is None`, puis les seuils du plus petit au plus grand avec `<`. Pour 3500 g, relis bien : « de 3500 g (inclus) ».

</details>
<details><summary>Indice 3</summary>

```python
if mass_g is None:      # test the missing value FIRST
    return "unknown"
if mass_g < 3500:
    return "light"
# then "medium" below 4500 (elif), and "heavy" for every other mass (else)
```

</details>

### Ex 0A.21 — Boucles : `for`, `range`, `enumerate`, `zip` et `while` 🔨

<details><summary>Indice 1</summary>

`zip` parcourt deux listes en parallèle, `enumerate` donne l'indice avec l'élément, `while` répète tant qu'une condition est vraie, `break` sort de la boucle.

</details>
<details><summary>Indice 2</summary>

a : accumule `mass / 1000` quand l'espèce est Chinstrap **et** la masse n'est pas `None`. c : compte les tours pendant que la colonie est `<= 1000`. d : `range(0, 101, 2)` (la fin 101 est exclue, 100 est donc inclus). e : `list.index` renvoie la position de la première occurrence.

</details>
<details><summary>Indice 3</summary>

b :
```python
for i, mass in enumerate(mass_list):
    if mass is not None and mass > 6000:
        ...   # remember i, then leave the loop with break
```
c :
```python
colony, years = 344, 0
while colony <= 1000:      # "exceed 1000": go on while it is still <= 1000
    ...                    # grow by 8 %, then count one more year
```

</details>

### Ex 0A.22 — Compréhensions : filtrer et transformer en une ligne 🔨

<details><summary>Indice 1</summary>

Forme générale : `[expression for élément in itérable if condition]` ; avec des accolades `{clé: valeur for ...}` pour un dictionnaire, `{expression for ...}` pour un ensemble.

</details>
<details><summary>Indice 2</summary>

a : la condition doit d'abord écarter `None` (`m is not None and m >= 6000`). c : parcours `zip(species_list, flipper_list)` avec deux variables. d : `.upper()` sur chaque île.

</details>
<details><summary>Indice 3</summary>

`heavy_kg = [m / 1000 for m in mass_list if m is not None and m >= 6000]` ; `counts = {sp: species_list.count(sp) for sp in sorted(set(species_list))}` ; `gentoo_cm = [f / 10 for sp, f in zip(...) if ...]`.

</details>

### Ex 0A.23 — Tes premières fonctions : paramètres, valeurs par défaut, `return` 🔨

<details><summary>Indice 1</summary>

Une fonction **renvoie** son résultat avec `return` ; `print` ne fait qu'afficher. Les paramètres avec `=` ont une valeur par défaut utilisée quand on ne les donne pas.

</details>
<details><summary>Indice 2</summary>

`flipper_per_kg` : convertis la masse en kg avant de diviser. `describe` : un `if unit == "kg"` / `elif unit == "g"` / `else: raise ValueError(...)`, puis une f-string.

</details>
<details><summary>Indice 3</summary>

Le squelette de `describe` :
```python
if unit == "kg":
    value = round(mass_g / 1000, decimals)
elif unit == "g":
    ...   # the mass in grams, as an integer
else:
    ...   # any other unit: raise ValueError(...)
return ...   # an f-string "<species>: <value> <unit>"
```

</details>

### Ex 0A.24 — Importer des modules : `math`, `random`, `statistics` et `Counter` 🔨

<details><summary>Indice 1</summary>

Chaque question a sa fonction toute faite, citée dans l'énoncé : l'exercice consiste à l'importer et à lire ce qu'elle renvoie.

</details>
<details><summary>Indice 2</summary>

`math.ceil` arrondit vers le haut. `Counter(...).most_common(1)` renvoie une **liste** contenant un seul couple `(valeur, nombre)`.

</details>
<details><summary>Indice 3</summary>

Commence par les trois `import` de l'énoncé. La seule ligne délicate est celle de c et d : `top_island, top_count = Counter(island_list).most_common(1)[0]` (`most_common(1)` renvoie une **liste** d'un seul couple : `[0]` sort le couple de la liste, puis le déballage le coupe en deux). Les autres réponses sont un appel direct de la fonction citée dans l'énoncé : `math.ceil` sur le nombre de batches **non arrondi**, `statistics.median` sur `flippers`, `statistics.stdev` sur `mass_known`.

</details>

### Ex 0A.25 — Exceptions : lever une `ValueError` et la rattraper 🔨

<details><summary>Indice 1</summary>

On **lève** une erreur là où on détecte le problème (`raise ValueError("...")`), on la **rattrape** là où on sait quoi faire (`try` / `except ValueError`).

</details>
<details><summary>Indice 2</summary>

`parse_mass` : `text.strip()`, test du texte vide, `float(text)` (qui lève seul une `ValueError` si le texte n'est pas un nombre), test du signe. `parse_all` : une boucle avec un `try` autour de l'appel, et `None` dans le `except`.

</details>
<details><summary>Indice 3</summary>

Le squelette de `parse_all` :
```python
masses = []
for text in texts:
    try:
        ...   # parse_mass(text), appended to masses
    except ValueError:
        ...   # an invalid text gives None
return masses
```
Rappel : `"4.2e3"` est une écriture scientifique valide (4200).

</details>

### Ex 0A.26 — Ton premier module mylearn : `mean` et son test 🔨

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

### Ex 0A.27 — Premiers arrays NumPy : `dtype`, `shape`, `ndim` 📦

<details><summary>Indice 1</summary>

`np.array(liste_de_listes)` crée un tableau à deux dimensions ; ses attributs `shape`, `dtype`, `ndim` et `size` décrivent sa forme et son contenu.

</details>
<details><summary>Indice 2</summary>

`shape` est un tuple `(lignes, colonnes)` ; `dtype.name` donne le nom du type sous forme de chaîne ; `size` = produit des dimensions. e : `np.linspace(0, 1, 11)` découpe [0, 1] en 10 intervalles égaux.

</details>
<details><summary>Indice 3</summary>

`X = np.array(table)`, puis `x_shape = X.shape` ; `x_ndim` et `x_size` se lisent de la même façon, sans parenthèses. b : l'attribut `dtype` a lui-même un attribut `name`, qui est une chaîne. e : `np.linspace(0, 1, 11)` fait 10 pas égaux de 0 à 1 ; la 4ᵉ valeur (indice 3) est à trois pas de 0.

</details>

### Ex 0A.28 — Indexation, tranches et masques booléens 📦

<details><summary>Indice 1</summary>

`X[ligne, colonne]` ; `X[:, j]` est une colonne entière ; une comparaison sur une colonne donne un **masque** de `True`/`False`.

</details>
<details><summary>Indice 2</summary>

b : les indices 100 à 199 inclus s'écrivent `100:200`. c : combine deux masques avec `&`, **chaque** condition entre parenthèses, puis `.sum()` compte les `True`. d : « au moins 4500 g » s'écrit `>= 4500` ; `.mean()` d'un masque donne la proportion de `True`.

</details>
<details><summary>Indice 3</summary>

a : `X[10, 2]` (ligne 10, colonne 2 : la nageoire). b : `X[100:200, 3]` (la fin 200 est exclue, donc 199 est inclus), puis sa moyenne. c : `((X[:, 0] > 45) & (...)).sum()`, en complétant la seconde condition, sur la colonne des nageoires. d : la moyenne d'un masque `>=` sur la colonne des masses.

</details>

### Ex 0A.29 — Vue ou copie : qui est modifié ? 🔮

<details><summary>Indice 1</summary>

Une **tranche** (`a[2:5]`) est une vue : elle partage la mémoire de `a`. Une **liste d'indices** (`a[[0, 1]]`) fabrique une copie.

</details>
<details><summary>Indice 2</summary>

Écris `a` au départ : `[0 1 2 3 4 5]`. Pour chaque ligne, demande-toi si elle crée une vue ou une copie (indice 1), et quelle case elle modifie. `b` commence à l'indice 2 de `a` : à quelle case de `a` correspond `b[0]` ? `c` est-il une vue de `a` ou un nouvel array ?

</details>
<details><summary>Indice 3</summary>

a : si `b` est une **vue**, écrire dans `b[0]` écrit aussi dans la case de `a` qui lui correspond. b : si `c` est une **copie**, `c[0] = -1` ne modifie que `c`. c : pars de $0 + 1 + 2 + 3 + 4 + 5$ et corrige chaque case de `a` qui a changé.

</details>

### Ex 0A.30 — Calcul vectorisé : unités, normalisation, fonctions universelles 📦

<details><summary>Indice 1</summary>

Une opération entre une colonne et un nombre (ou entre deux colonnes) s'applique à tous les éléments d'un coup.

</details>
<details><summary>Indice 2</summary>

b : `f = X[:, 2]`, puis la formule de l'énoncé avec `f.min()` et `f.max()`. c : divise la colonne 0 par la colonne 1. d : `np.where(condition, valeur_si_vrai, valeur_si_faux)`, puis compte avec `(labels == "heavy").sum()`.

</details>
<details><summary>Indice 3</summary>

a : la colonne des masses divisée par 1000, puis `.mean()`. b : `flipper_scaled = (f - f.min()) / (f.max() - f.min())`, avec `f = X[:, 2]` : les parenthèses comptent. d : `labels = np.where(X[:, 3] >= 4500, "heavy", "not heavy")` (même règle qu'en 0A.20), puis compte les `"heavy"`.

</details>

### Ex 0A.31 — Aléatoire reproductible : `default_rng`, graine, `permutation`, `choice` 📦

<details><summary>Indice 1</summary>

Un générateur `rng = np.random.default_rng(graine)` donne toujours la même suite de nombres pour la même graine ; `integers(1, 7)` exclut 7.

</details>
<details><summary>Indice 2</summary>

a : la proportion de 6 est la moyenne du masque `rolls == 6`. d : crée **deux** générateurs avec la même graine et compare leur premier `.random()`. e : `choice(len(X), size=50, replace=False)` renvoie 50 indices distincts ; utilise-les pour indexer les lignes de `X`.

</details>
<details><summary>Indice 3</summary>

a : `rolls = rng.integers(1, 7, size=10_000)` (7 est exclu : les faces 1 à 6), puis la proportion de 6, c'est-à-dire la moyenne du masque `rolls == 6`. e : `idx`, le résultat de l'appel de l'énoncé, contient 50 numéros de lignes ; `X[idx, 2]` garde la nageoire de ces 50 manchots, il reste la moyenne. Attention : chaque appel à `rng` fait avancer le générateur, respecte l'ordre de l'énoncé.

</details>

### Ex 0A.32 — Premier contact avec Penguins : `read_csv`, `head`, `info`, `describe` 📦

<details><summary>Indice 1</summary>

`pd.read_csv(chemin)` lit le fichier ; `info()` affiche le nombre de valeurs non manquantes par colonne ; `describe()` renvoie un tableau de statistiques des colonnes numériques.

</details>
<details><summary>Indice 2</summary>

b : `df["sex"].count()` compte les valeurs non manquantes. c et d : lis `describe()` avec `.loc["mean", "body_mass_g"]` et `.loc["max", ...]`. e : le nombre de **colonnes** du tableau renvoyé par `describe()`.

</details>
<details><summary>Indice 3</summary>

`df = pd.read_csv(wb.datasets.data_dir() / "penguins.csv")` ; `summary = df.describe()` ; `n_numeric = summary.shape[1]`. N'oublie pas `year`, qui est aussi numérique.

</details>

### Ex 0A.33 — Sélectionner : colonnes, `loc`, `iloc` et filtres 📦

<details><summary>Indice 1</summary>

`loc` sélectionne par **label** (le nom de l'index et des colonnes), `iloc` par **position** ; un filtre est un masque booléen placé entre crochets.

</details>
<details><summary>Indice 2</summary>

a : `penguins.loc[100, "island"]`. b : la position `-1` avec `iloc`. c : deux conditions avec `&` et des parenthèses, puis `len` ou `.sum()` du masque. d et e : `penguins.loc[masque, "colonne"].mean()`.

</details>
<details><summary>Indice 3</summary>

`((penguins["species"] == "Gentoo") & (penguins["body_mass_g"] > 5500)).sum()` ; `penguins.loc[penguins["island"] == "Dream", "bill_length_mm"].mean()`.

</details>

### Ex 0A.34 — Valeurs manquantes et doublons : `isna`, `dropna`, `fillna`, `duplicated` 📦

<details><summary>Indice 1</summary>

`isna()` renvoie un tableau de `True`/`False` ; `sum()` compte les `True` **par colonne**.

</details>
<details><summary>Indice 2</summary>

a : une première somme par colonne, une seconde pour le total. b et c : `len(...)` après `dropna()`, avec ou sans `subset`. d : `fillna("unknown")` sur la colonne `sex`, puis compte. e : `duplicated()` marque chaque ligne déjà vue plus haut. f : remplace par `flipper.median()` avant de faire la moyenne.

</details>
<details><summary>Indice 3</summary>

a : `penguins.isna().sum()` donne un total **par colonne** ; un second `.sum()` additionne ces totaux. c : `len(penguins.dropna(subset=["body_mass_g"]))` (b : la même chose, sans `subset`). e : `duplicated()` renvoie un masque : compte ses `True`. f : `flipper = penguins["flipper_length_mm"]`, puis `flipper.fillna(...)` avec sa médiane, avant la moyenne.

</details>

### Ex 0A.35 — De pandas à NumPy : construire `X` et `y` 📦

<details><summary>Indice 1</summary>

Convention : `X` est un tableau à deux dimensions (un exemple par ligne), `y` un vecteur à une dimension (un label par exemple).

</details>
<details><summary>Indice 2</summary>

Enlève les lignes incomplètes avec `dropna()`, sélectionne les colonnes avec une **liste** de noms (`clean[numeric_cols]`) et l'espèce avec un seul nom (`clean["species"]`), puis convertis avec `.to_numpy()`.

</details>
<details><summary>Indice 3</summary>

```python
clean = penguins.dropna()            # no subset: rows with NO missing value at all
X = clean[numeric_cols].to_numpy()   # a LIST of names gives a 2-D table
```
`y` : la même chose avec **un seul** nom de colonne, sans doubles crochets (`clean[["species"]]` donnerait la forme `(333, 1)`).

</details>

### Ex 0A.36 — Premiers graphiques : `plot`, `scatter`, `hist` 📦

<details><summary>Indice 1</summary>

Chaque graphique suit le même moule : `fig, ax = plt.subplots()`, un appel de tracé sur `ax`, les noms des axes, `plt.show()`.

</details>
<details><summary>Indice 2</summary>

1 : `ax.hist(..., bins=25)` sur les masses **sans** les valeurs manquantes. 2 : une boucle `for species, group in penguins.groupby("species"):` avec un `ax.scatter(..., label=species)` par espèce, puis `ax.legend()`. 3 : une boucle sur `yearly.columns` avec `ax.plot(yearly.index, yearly[species], label=species)`.

</details>
<details><summary>Indice 3</summary>

Le squelette de la figure 2 :
```python
fig, ax = plt.subplots()
for species, group in penguins.groupby("species"):
    ax.scatter(...)   # x: the bill length of the group, y: its bill depth, label=species
# then the axis labels with their units, ax.legend(), plt.show()
```
Même moule pour 1 (`ax.hist` sur les masses sans valeurs manquantes) et pour 3 (une boucle sur `yearly.columns`, avec `ax.plot`).

</details>

### Ex 0A.37 — Lire un traceback : cinq bugs de débutant 🐛

<details><summary>Indice 1</summary>

La dernière ligne d'un traceback donne le **type** d'erreur et sa description ; la ligne juste au-dessus montre l'instruction fautive dans ta fonction.

</details>
<details><summary>Indice 2</summary>

Pour chaque erreur, pose-toi une question : quel type la fonction reçoit-elle vraiment (texte ou nombre) ? quel est le dernier indice valide d'une liste ? la clé existe-t-elle avec cette orthographe ? la méthode est-elle bien écrite ? le nom est-il défini ?

</details>
<details><summary>Indice 3</summary>

Le premier bug, en modèle : `sum` part de 0 et ne sait pas ajouter un texte à un nombre ; il faut convertir chaque texte d'abord, `sum(float(t) for t in texts) / 1000` (`float` tolère l'espace de `" 3800"`). Pour les quatre autres, lis la dernière ligne de chaque traceback. Quel est le dernier indice valide d'une liste de 3 éléments ? Sous quelle forme exacte la clé est-elle écrite dans `island_counts` (majuscule, espaces) ? La méthode existe-t-elle avec cette orthographe ? Quel nom de constante est défini au-dessus ? a) : recopie les cinq noms d'exceptions dans l'ordre des tracebacks ; le premier est `TypeError`.

</details>

### Ex 0A.38 — Lire penguins.csv comme un simple fichier texte (`pathlib`, `with`) 🔨

<details><summary>Indice 1</summary>

Un fichier ouvert avec `open` se parcourt ligne par ligne ; `f.readline()` lit une seule ligne (l'en-tête), puis `for line in f` continue **après** elle.

</details>
<details><summary>Indice 2</summary>

`read_rows` : dans un bloc `with`, découpe l'en-tête avec `.strip().split(",")`, puis chaque ligne de la même façon. `column_values` : `j = header.index(name)`, puis garde `float(row[j])` quand `row[j] != "NA"`. `write_rows` : recolle chaque ligne avec `",".join(row)`.

</details>
<details><summary>Indice 3</summary>

Le squelette de `read_rows` :
```python
with open(path, encoding="utf-8") as f:
    header = f.readline().strip().split(",")
    rows = ...   # every remaining line of f, split the same way (skip the empty ones)
return header, rows
```
Pour écrire : construis la liste `lines` (l'en-tête, puis chaque ligne recollée avec `",".join(...)`), puis écris-la d'un coup avec `Path(path).write_text(..., encoding="utf-8")`, les lignes séparées par `"\n"` et suivies d'un dernier `"\n"`.

</details>

### Ex 0A.39 — Sauvegarder et recharger des résultats : `json` et `pickle` 🔨

<details><summary>Indice 1</summary>

JSON ne connaît que des types simples : dictionnaire, liste, chaîne, nombre Python, booléen, `None`. Un nombre NumPy (`np.int64`) n'en fait pas partie.

</details>
<details><summary>Indice 2</summary>

Écris d'abord `convert(value)` pour **une** valeur : `int(value)` si `isinstance(value, np.integer)`, `float(value)` si `np.floating`, une liste si c'est un tuple, et la valeur telle quelle sinon. `to_jsonable` applique `convert` à chaque valeur, et aux valeurs des dictionnaires imbriqués.

</details>
<details><summary>Indice 3</summary>

b) Le cœur de `to_jsonable`, c'est `convert`, qui traite **une** valeur :
```python
def convert(value):
    if isinstance(value, np.integer):
        return int(value)
    ...   # np.floating -> float(value); tuple -> list of converted items; dict -> convert each value
    return value
```
`to_jsonable` applique ensuite `convert` à chaque valeur de `results`. Fichiers : chaque fonction commence par `with open(path, "w", encoding="utf-8") as f:` (en lecture : sans `"w"`), puis appelle `json.dump` ou `json.load` ; pour pickle, même schéma en `"wb"` / `"rb"` (sans `encoding`), avec `pickle.dump` / `pickle.load`. c) : que devient un tuple en JSON ?

</details>

### Ex 0A.40 — Arguments variables : `*args`, `**kwargs` et keyword-only 🔨

<details><summary>Indice 1</summary>

Dans `def f(*values)`, `values` est un **tuple** ; dans `def f(**extra)`, `extra` est un **dictionnaire**. Au moment de l'appel, `*liste` et `**dico` font l'inverse : ils déballent.

</details>
<details><summary>Indice 2</summary>

`mean_of` : teste `len(values) == 0` avant de diviser. `make_config` : construis un dictionnaire avec `lr` et `epochs`, puis ajoute le contenu de `extra` (`d.update(extra)`, ou `{..., **extra}`).

</details>
<details><summary>Indice 3</summary>

`make_config` : `return {"lr": lr, "epochs": epochs, ...}`, où les `...` déballent le dictionnaire `extra` (avec deux étoiles, comme à l'appel). `mean_of` : teste `len(values) == 0` et lève la `ValueError` **avant** de diviser. d) : un argument positionnel alors que la fonction n'en accepte aucun, quel type d'erreur ? e) : `make_config(**settings, dropout=0.2)` reçoit `lr=0.5`, `epochs=3` et `dropout=0.2`.

</details>

### Ex 0A.41 — Fonctions en argument : `lambda`, `key=` et `Callable` 🔨

<details><summary>Indice 1</summary>

`key=` reçoit une **fonction** appliquée à chaque élément avant la comparaison ; `lambda r: r[3]` renvoie la masse d'un tuple `(espèce, île, nageoire, masse)`.

</details>
<details><summary>Indice 2</summary>

a) `max(..., key=lambda r: r[3])` ; b) `sorted(..., key=lambda r: r[2])` puis `[:3]` ; c) `island_sizes.get` est déjà une fonction (sans parenthèses !) ; d) et e) : une compréhension qui appelle `func(v)` ou `predicate(v)`.

</details>
<details><summary>Indice 3</summary>

d) `apply` renvoie une liste : une compréhension dont l'expression est l'**appel** `func(v)`, pour chaque `v` de `values`. e) `count_if` : `sum(1 for v in values if ...)`, où la condition est l'appel de `predicate` sur `v` (une boucle avec un compteur marche aussi).

</details>

### Ex 0A.42 — Fermetures : une fabrique de fonctions 🔨

<details><summary>Indice 1</summary>

Une fonction peut définir une fonction **à l'intérieur** d'elle-même et la **renvoyer** (sans parenthèses) : la fonction intérieure se souvient des variables de la fonction englobante.

</details>
<details><summary>Indice 2</summary>

`make_running_mean` : crée `total = 0.0` et `count = 0` dans la fabrique, puis une fonction `add(x)` qui les met à jour et renvoie `total / count`. Pour **modifier** ces variables depuis `add`, déclare-les `nonlocal`.

</details>
<details><summary>Indice 3</summary>

```python
def make_scaler(low, high):
    def scale(x):
        ...          # the formula of the statement, with low and high remembered from make_scaler
    return scale     # the function itself, without parentheses
```
Dans `add`, la première ligne est `nonlocal total, count` ; ensuite, mets à jour le total et le compteur, puis renvoie la moyenne. Le `make_counter` de la fiche §100.5.4 a exactement cette forme.

</details>

### Ex 0A.43 — Le piège des lambdas créées dans une boucle 🔮

<details><summary>Indice 1</summary>

Relis le dernier paragraphe de la fiche §100.5.4 : une fermeture garde-t-elle une photo de la valeur, ou un accès à la variable ?

</details>
<details><summary>Indice 2</summary>

Distingue deux moments : celui où chaque `lambda` est **créée** (pendant la boucle) et celui où elle est **appelée** (dans `results`). Que vaut `k` au second moment ?

</details>
<details><summary>Indice 3</summary>

a : chaque `lambda` contient le **nom** `k`, pas un nombre. Calcule `results` dans les deux hypothèses : si chaque fonction lisait `k` au moment de sa **création**, puis si elle le lisait au moment de son **appel**, une fois la boucle finie (que vaut alors `k` ?). Le dernier paragraphe de la fiche §100.5.4 dit laquelle est la bonne. b : garde la compréhension de `multipliers` et remplace seulement la `lambda` par la version à valeur par défaut de l'énoncé.

</details>

### Ex 0A.44 — Fonctions récursives : parcourir un arbre de dictionnaires (profondeur, nombre de feuilles) 🔨

<details><summary>Indice 1</summary>

Chaque fonction a deux cas : si `tree` n'est pas un dictionnaire (`not isinstance(tree, dict)`), c'est une feuille et la réponse est immédiate ; sinon, on combine les réponses des enfants (`tree.values()`).

</details>
<details><summary>Indice 2</summary>

Cas de base : `total` renvoie la feuille, `count_leaves` renvoie 1, `depth` renvoie 0, `largest_leaf` renvoie la feuille. Cas récursif : `sum(...)`, `sum(...)`, `1 + max(...)`, `max(...)` sur les appels récursifs de chaque enfant.

</details>
<details><summary>Indice 3</summary>

Les quatre fonctions ont la même forme ; seules changent la réponse pour une feuille et la façon de combiner les réponses des enfants :
```python
def f(tree):
    if not isinstance(tree, dict):   # base case: a leaf
        return ...                   # the answer for ONE leaf
    return COMBINE(f(child) for child in tree.values())   # one recursive call per child
```
`COMBINE` vaut `sum` ou `max`, et pour `depth` « 1 + `max` » ; la réponse pour une feuille est dans l'indice 2. Teste chaque fonction sur le petit arbre de l'énoncé avant `colonies`.

</details>

### Ex 0A.45 — Expressions régulières : identifiants et dates de Penguins brut 🔨

<details><summary>Indice 1</summary>

`\d+` : un ou plusieurs chiffres ; `[12]` : le caractère 1 ou 2 ; `re.fullmatch` exige que **tout** le texte suive le motif ; des parenthèses créent un groupe que `.group(1)` renvoie.

</details>
<details><summary>Indice 2</summary>

a) traduis le motif morceau par morceau : une lettre fixe, des chiffres répétés, une lettre fixe, un caractère parmi deux ; `re.fullmatch` renvoie `None` quand le texte ne suit pas le motif. c) entoure la partie à extraire de parenthèses, puis convertis le groupe. d) trois groupes, un par morceau de la date. e) compte les commentaires pour lesquels `re.search` trouve quelque chose, en ignorant la casse.

</details>
<details><summary>Indice 3</summary>

c) le motif, avec un groupe autour du numéro : `r"N(\d+)A[12]"`. `re.fullmatch(motif, text)` renvoie un objet « match » (ou `None`) ; sa méthode `.group(1)` donne le texte capturé par les parenthèses, à convertir avec `int()`. a) le même motif sans les parenthèses : `is_valid_id` renvoie `True` quand `fullmatch` ne renvoie pas `None`. e) compte les commentaires `c` pour lesquels `re.search(r"sex", c, flags=re.IGNORECASE)` trouve quelque chose.

</details>

### Ex 0A.46 — `itertools` et `heapq` : paires de features, grille, top-k 🔨

<details><summary>Indice 1</summary>

`combinations(liste, 2)` donne toutes les paires sans ordre ; `product(a, b, c)` toutes les combinaisons d'un élément de chaque liste ; `heapq.nlargest(k, …)` et `heapq.nsmallest(k, …, key=…)` les extrêmes.

</details>
<details><summary>Indice 2</summary>

Ce sont des itérateurs : pour compter, `len(list(...))`. b) `key=lambda pair: abs(mean_mass[pair[0]] - mean_mass[pair[1]])`. e) la clé est la masse, `r[1]`.

</details>
<details><summary>Indice 3</summary>

`n_settings = len(list(product([0.1, 0.01, 0.001], [16, 32, 64], [5, 10])))` ; `lightest2 = heapq.nsmallest(2, mass_records, key=lambda r: r[1])`.

</details>

### Ex 0A.47 — Une classe `RunningStats` : `__init__`, attributs, méthodes, puis la même en `@dataclass` 🔨

<details><summary>Indice 1</summary>

Les attributs (`self.n`, `self.total`…) gardent l'état de l'objet entre deux appels de `add` ; `__init__` les crée, `add` les met à jour, `mean` et `std` les lisent.

</details>
<details><summary>Indice 2</summary>

Dans `add` : `n` augmente de 1, `total` de `x`, `total_sq` de `x * x` ; `minimum` devient `x` s'il vaut `None` ou si `x` est plus petit. `std` : `math.sqrt(self.total_sq / self.n - self.mean() ** 2)`. Pour la dataclass : `@dataclass` au-dessus de `class PenguinRecord:`, puis les trois champs annotés, et **supprime** la ligne `PenguinRecord = None`.

</details>
<details><summary>Indice 3</summary>

a) Dans `add`, la ligne délicate est celle du minimum, qui vaut `None` au départ : `self.minimum = x if self.minimum is None else min(self.minimum, x)` (même chose pour le maximum). `std` applique la formule de l'encadré avec `math.sqrt` et `self.mean()`.

b) Le squelette de la dataclass (supprime d'abord la ligne `PenguinRecord = None`) :
```python
@dataclass
class PenguinRecord:
    species: str
    island: str
    mass_g: float | None = None

    def mass_kg(self):
        ...   # None if the mass is missing, else the mass in kg
```

</details>

### Ex 0A.48 — Méthodes spéciales : une classe `Vector2D` qui s'additionne 🔨

<details><summary>Indice 1</summary>

Chaque opérateur appelle une méthode : `a + b` → `a.__add__(b)`, `2 * v` → `v.__rmul__(2)`, `abs(v)` → `v.__abs__()`, `v[1]` → `v.__getitem__(1)`, `sum(vs)` → `0 + v1`, donc `v1.__radd__(0)`.

</details>
<details><summary>Indice 2</summary>

Les opérations renvoient un **nouveau** `Vector2D` (sans modifier `self`). `__rmul__` peut simplement renvoyer `self * scalar` ; `__radd__` renvoie `self` si `other == 0`, sinon `self + other`.

</details>
<details><summary>Indice 3</summary>

`__add__` renvoie `Vector2D(..., ...)`, un **nouveau** vecteur, avec la somme des `x` puis celle des `y` ; `__sub__` et `__mul__` suivent le même modèle. La méthode piège est `__radd__` :
```python
def __radd__(self, other):
    if other == 0:     # sum() starts with 0 + first vector
        return self
    ...                # otherwise, an ordinary addition
```
`__repr__` renvoie une f-string au format exact de l'énoncé, `Vector2D(x=1, y=2)`.

</details>

### Ex 0A.49 — Héritage et `super()` : un mini-estimateur `fit`/`predict` appelable 🔨

<details><summary>Indice 1</summary>

Une classe fille n'écrit que ce qui change : `fit` et `predict`. `score` et `__call__` sont **hérités** de `BaseClassifier`.

</details>
<details><summary>Indice 2</summary>

`MajorityClassifier.fit` : `self.majority_ = Counter(y).most_common(1)[0][0]`, puis `return self`. `NearestCentroidClassifier.fit` : `self.classes_ = np.unique(y)`, et une ligne de moyennes par classe `X[y == label].mean(axis=0)`. `predict` : pour chaque centre `c`, les distances `np.sqrt(((X - c) ** 2).sum(axis=1))`, puis la classe du centre le plus proche.

</details>
<details><summary>Indice 3</summary>

La ligne clé de `NearestCentroidClassifier.predict` :
```python
def predict(self, X):
    distances = np.array([np.sqrt(((X - c) ** 2).sum(axis=1)) for c in self.centroids_])   # (n_classes, n)
    ...   # for each penguin (each column), the class of the closest centroid: argmin, then self.classes_[...]
```
Et `__init__(self, verbose=False)` : `super().__init__(verbose)`.

</details>

### Ex 0A.50 — Générateurs et itérables : un mini-Dataset de manchots 🔨

<details><summary>Indice 1</summary>

`len(ds)` appelle `ds.__len__()`, `ds[i]` appelle `ds.__getitem__(i)`, `for x in ds` appelle `ds.__iter__()` ; une fonction qui contient `yield` renvoie un générateur.

</details>
<details><summary>Indice 2</summary>

`__iter__` : une boucle sur `range(len(self))` qui fait `yield self[i]`. `batches` : une boucle sur `range(0, len(dataset), batch_size)` qui fait `yield` des tranches `dataset.X[start:start + batch_size]` et `dataset.y[...]`.

</details>
<details><summary>Indice 3</summary>

```python
def batches(dataset, batch_size):
    for start in range(0, len(dataset), batch_size):
        ...   # yield the pair (slice start:start + batch_size of dataset.X, same slice of dataset.y)
```
`__iter__` a la même forme : une boucle sur `range(len(self))`, et un `yield` par exemple.

</details>

### Ex 0A.51 — Réductions par axe, tri, `argmax` et `unique` 📦

<details><summary>Indice 1</summary>

`axis=k` fait **disparaître** l'axe `k` : sur un tableau `(3, 4)` (espèces × mesures), une valeur par mesure s'obtient en faisant disparaître l'axe des espèces.

</details>
<details><summary>Indice 2</summary>

b) pour chaque nom d'espèce, sélectionne ses lignes avec un masque sur `y`, puis fais la moyenne **par colonne** ; empile les trois résultats dans un array. c) sur `class_means`, quel axe faut-il faire disparaître pour garder une valeur par mesure ? d) la position du maximum de la colonne des masses, puis l'espèce à cette position. e) `np.unique` renvoie deux tableaux avec `return_counts=True`.

</details>
<details><summary>Indice 3</summary>

`best_species_idx = class_means.argmax(axis=0)` donne 4 indices, un par colonne ; `species_names[best_species_idx]` les traduit en noms. f) `np.sort(X[:, 3])[:3]`.

</details>

### Ex 0A.52 — Broadcasting : standardiser toutes les colonnes d'un coup 📦

<details><summary>Indice 1</summary>

`(333, 4)` et `(4,)` sont compatibles : la même moyenne (et le même écart-type) de chaque colonne s'applique à toutes les lignes.

</details>
<details><summary>Indice 2</summary>

a) `(X - X.mean(axis=0)) / X.std(axis=0)`. e) suis les formes pas à pas : `(5, 1, 4) - (1, 5, 4)` → `(5, 5, 4)` ; `** 2` ; `.sum(axis=2)` → `(5, 5)` ; `np.sqrt`.

</details>
<details><summary>Indice 3</summary>

```python
P = Z[:5]
diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]   # (5, 1, 4) - (1, 5, 4) -> (5, 5, 4): every pair
D = ...   # square diff, sum over the LAST axis (one value per pair), then the square root -> (5, 5)
```

</details>

### Ex 0A.53 — `reshape`, transposée et empilement 📦

<details><summary>Indice 1</summary>

`reshape` garde les données dans le même ordre (ligne par ligne) ; `.T` échange lignes et colonnes ; `np.stack` crée un **nouvel** axe, `np.concatenate` et `np.hstack` collent le long d'un axe **existant**.

</details>
<details><summary>Indice 2</summary>

Écris `v.reshape(3, 4)` à la main : trois lignes de quatre nombres, remplies dans l'ordre, ligne après ligne. La ligne 1 de la transposée est la colonne 1 de ce tableau. e) les deux tableaux collés côte à côte doivent avoir le même nombre de lignes : `np.ones((len(X), 1))`.

</details>
<details><summary>Indice 3</summary>

Les trois fonctions d'assemblage reçoivent **une liste** de tableaux, entre crochets, comme en d : `two_cols = np.stack([X[:, 0], X[:, 2]], axis=1)`. Même forme d'appel pour e (`np.hstack`, la colonne de 1 d'abord, puis `X`) et pour g (`np.concatenate`, avec les deux morceaux de lignes). Les pièges : la colonne de 1 doit avoir deux dimensions, `(len(X), 1)`, comme `X` (avec `np.ones(len(X))`, `np.hstack` lève une `ValueError`) ; pour prévoir g, `X[200:]` va de la ligne 200 à la dernière, la 332.

</details>

### Ex 0A.54 — Images MNIST : un tableau `(N, 28, 28)` 📦

<details><summary>Indice 1</summary>

Une réduction sans `axis` porte sur **tout** le tableau ; un masque `images == 0` a la même forme que `images`, et sa moyenne est une proportion.

</details>
<details><summary>Indice 2</summary>

c) la première image est `images[0]` ; d) compare `images` à 0, puis fais la moyenne du masque ; e) `reshape` avec `-1` ; f) pour chaque chiffre `d` de 0 à 9, sélectionne ses images avec un masque sur `labels`, puis fais leur moyenne ; g) `.nbytes` donne la taille en octets : combien d'octets pour un `float64`, pour un `uint8` ?

</details>
<details><summary>Indice 3</summary>

`ink = np.array([images[labels == d].mean() for d in range(10)])`, puis `ink.argmax()` et `ink.argmin()` ; `memory_ratio = (images / 255).nbytes / images.nbytes`.

</details>

### Ex 0A.55 — Boucle Python contre NumPy : mesurer le gain 🔬

<details><summary>Indice 1</summary>

Trois boucles imbriquées : les images, les lignes d'une image, les pixels d'une ligne. Et une seule ligne NumPy : une moyenne sur les deux axes des pixels.

</details>
<details><summary>Indice 2</summary>

Dans la boucle, additionne `int(pixel)` (un `uint8` déborde au-delà de 255) et compte les pixels ; ajoute `total / count` à la liste des moyennes. Version NumPy : `images.mean(axis=(1, 2))`.

</details>
<details><summary>Indice 3</summary>

Le squelette de la version en boucles :
```python
means = []
for image in images:
    total, count = 0, 0
    ...   # two nested loops: each row of the image, then each pixel of the row;
          # add int(pixel) to total and 1 to count
    means.append(total / count)
return means
```

</details>

### Ex 0A.56 — Bugs NumPy : `axis` oublié, formes `(n,)` et `(n, 1)`, vue modifiée 🐛

<details><summary>Indice 1</summary>

Regarde les **formes** : combien de nombres renvoie `X.mean()` ? Quelle forme a `(333, 1) - (333,)` ? Une tranche `X[:10, 2]` est-elle une vue ou une copie ?

</details>
<details><summary>Indice 2</summary>

Mesure chaque symptôme avant de corriger : combien de valeurs renvoie `X_demo.mean()` ? Quelle forme a `y_true_col - y_pred` (relis le cas f de 0A.8) ? Après `part = X_demo[:10, 2]`, que répond `np.shares_memory(part, X_demo)` (0A.29) ? Chaque réponse désigne la cause d'un des trois bugs.

</details>
<details><summary>Indice 3</summary>

`column_means`, en modèle : sans `axis`, `X.mean()` réduit **tout** le tableau en un seul nombre ; avec `X.mean(axis=0)`, l'axe des lignes disparaît et il reste une moyenne par colonne, de forme `(4,)`. `mse` : `(333, 1) - (333,)` s'étire en `(333, 333)` ; ramène d'abord chaque entrée à **une** dimension (une méthode du tableau de la fiche §100.8.6 aplatit un array), puis lève une `ValueError` si les deux formes diffèrent encore. `centered_flippers` : `X[:k, 2]` est une vue, donc `part -= ...` écrit dans `X` ; comment obtenir un tableau indépendant **avant** de le modifier (fiche §100.8.2) ?

</details>

### Ex 0A.57 — Compter et regrouper : `value_counts`, `groupby`, `agg`, `sort_values` 📦

<details><summary>Indice 1</summary>

`value_counts` compte les valeurs d'une colonne ; `groupby(clé)[colonne]` puis une réduction (`mean`, `agg`, `nunique`) calcule une statistique par groupe.

</details>
<details><summary>Indice 2</summary>

b) une proportion : pense à l'option de `value_counts`. c) avec deux clés, l'index a deux niveaux : on lit une case avec un **tuple**. d) le tableau `agg` a une ligne par espèce et une colonne par statistique : `.loc[ligne, colonne]`. e) trie par longueur de bec, du plus grand au plus petit, puis lis la première ligne.

</details>
<details><summary>Indice 3</summary>

`penguins.groupby(["species", "sex"])["body_mass_g"].mean().loc[("Gentoo", "male")]` ; `penguins.groupby("island")["species"].nunique()["Torgersen"]`.

</details>

### Ex 0A.58 — Le filtre qui ne filtre pas : `and`, `&`, parenthèses et copies 🐛

<details><summary>Indice 1</summary>

Lis les deux messages d'erreur affichés : leur nom est la réponse de a) et b). Le troisième essai ne lève aucune erreur… mais combien de `"unknown"` y a-t-il ensuite ?

</details>
<details><summary>Indice 2</summary>

Le bon filtre : `(condition1) & (condition2)`, chaque condition entre parenthèses. Pour modifier des lignes filtrées : `fixed_df.loc[masque, "sex"] = "unknown"`, en **une** instruction.

</details>
<details><summary>Indice 3</summary>

c : `penguins[(penguins["species"] == "Gentoo") & (...)]`, la seconde condition entre parenthèses elle aussi, puis la longueur du résultat. d : `fixed_df.loc[fixed_df["sex"].isna(), "sex"] = "unknown"`, puis compte les `"unknown"`. e : `isin` reçoit la **liste** des deux îles et renvoie un masque : compte ses `True`.

</details>

### Ex 0A.59 — Figures à plusieurs panneaux : `subplots`, `imshow`, `show_images` 📦

<details><summary>Indice 1</summary>

`fig, axes = plt.subplots(2, 5)` : `axes` est un tableau `(2, 5)` d'axes ; `axes.ravel()` le parcourt case par case, dans le même ordre que les chiffres 0 à 9.

</details>
<details><summary>Indice 2</summary>

Figure 1 : `for d, ax in enumerate(axes.ravel()):` puis `ax.imshow(...)`, `ax.set_title(str(d))`, `ax.axis("off")`. Figure 3 : `for ax, col in zip(axes, [...]):`, et dans chaque panneau une boucle sur `clean.groupby("species")`.

</details>
<details><summary>Indice 3</summary>

Le squelette de la figure 1 :
```python
fig_digits, axes = plt.subplots(2, 5, figsize=(9, 4))
for d, ax in enumerate(axes.ravel()):
    ...   # imshow of the mean image of digit d (cmap="gray_r"), the digit as title, no axes
plt.show()
```
Figure 3 : même schéma, avec `zip(axes, [...])` sur les trois colonnes et, dans chaque panneau, un histogramme par espèce (`alpha=0.5`, `label=species`), un titre et une légende.

</details>

### Ex 0A.60 — Quelle mesure sépare le mieux les espèces ? 📈

<details><summary>Indice 1</summary>

Deux espèces sont bien séparées sur une mesure quand leurs histogrammes **ne se chevauchent presque pas**.

</details>
<details><summary>Indice 2</summary>

Compare les histogrammes bleu (Adelie) et orange (Chinstrap) sur chaque panneau. Pour b) et c), place le point sur l'axe (ou dans le nuage) et regarde quelle couleur l'entoure.

</details>
<details><summary>Indice 3</summary>

a à c se lisent sur les graphiques. a : sur quel panneau les histogrammes bleu et orange se chevauchent-ils le **moins** ? b : sur le panneau des nageoires, quelle couleur trouve-t-on encore vers 225 mm ? c : dans le nuage, de quelle couleur sont les points autour de (50, 19) ? d :
```python
def separation(col):
    a = clean.loc[clean["species"] == "Adelie", col]
    c = ...      # the same for Chinstrap
    return ...   # the ratio of the statement, with .mean(), .var() and np.sqrt
```

</details>

### Ex 0A.61 — Une docstring au format NumPy, vérifiée par doctest 🛠️

<details><summary>Indice 1</summary>

Pars du modèle de la fiche §100.5.5 : une phrase de résumé, puis chaque section soulignée par des tirets de la même longueur que son titre.

</details>
<details><summary>Indice 2</summary>

Code : `values = np.asarray(values, dtype=float)`, deux `raise ValueError(...)` (vide, puis `max == min`), puis la formule. Exemples : exécute `min_max_scale([2, 4, 6])` dans une cellule et recopie **exactement** la sortie sous la ligne `>>>`.

</details>
<details><summary>Indice 3</summary>

```text
Examples
--------
>>> min_max_scale([2, 4, 6])
array([0. , 0.5, 1. ])
```
Le message de la `ValueError` de ton exemple doit être celui que lève vraiment ton code.

</details>

### Ex 0A.62 — Écrire tes propres tests : `assert`, `approx`, `raises`, `parametrize` 🛠️

<details><summary>Indice 1</summary>

Un test est une fonction `test_...` qui appelle `min_max_scale` et fait un `assert` ; pytest le compte comme réussi s'il ne lève rien. Lis les trois versions buggées : chacune a un défaut différent.

</details>
<details><summary>Indice 2</summary>

`scale_bug_1` donne de mauvaises valeurs, `scale_bug_2` ne lève pas d'erreur, `scale_bug_3` inverse l'échelle : il faut un test de valeurs exactes, un test `pytest.raises` et un test qui vérifie l'ordre (le plus petit élément devient 0).

</details>
<details><summary>Indice 3</summary>

```python
@pytest.mark.parametrize("values", [[1, 2], [3, -1, 7]])
def test_bounds(values):
    ...   # result = min_max_scale(values), then assert that its min is 0 and its max is 1

my_tests = [test_three_integers, ..., test_bounds]   # your test functions, without parentheses
```
Chaque version buggée doit faire échouer au moins un de tes tests : l'indice 2 dit quel test vise quel bug.

</details>

### Ex 0A.63 — `utils.count_values` : compter sans pandas 🔨

<details><summary>Indice 1</summary>

Un dictionnaire d'accumulation, comme en 0A.18 : une clé par valeur rencontrée, qu'on augmente de 1 à chaque passage.

</details>
<details><summary>Indice 2</summary>

Une seule boucle : pour chaque valeur, refuse `NaN` (`isinstance(v, float)` ou `np.floating`, et `v != v`), puis `counts[v] = counts.get(v, 0) + 1`. Après la boucle : `ValueError` si `counts` est vide ; avec `normalize`, divise chaque compte par le total.

</details>
<details><summary>Indice 3</summary>

```python
counts = {}
for v in values:
    if isinstance(v, (float, np.floating)) and v != v:   # NaN is the only value different from itself
        ...   # refuse it: raise ValueError(...)
    counts[v] = counts.get(v, 0) + 1
# after the loop: ValueError if counts is empty; with normalize, divide each count by the total
```

</details>

### Ex 0A.64 — `utils.argmax` : le premier maximum, avec des boucles 🔨

<details><summary>Indice 1</summary>

Écris d'abord une petite fonction interne pour le cas 1-D : parcourir une liste en gardant l'indice du meilleur élément. Les autres cas s'en servent.

</details>
<details><summary>Indice 2</summary>

`axis=None` : applique-la à `values.ravel()`. `axis=1` : applique-la à chaque ligne. `axis=0` : à chaque ligne de `values.T` (les colonnes). Commence par valider (`values.size == 0`, `np.isnan(values).any()`, `values.ndim`, `axis`).

</details>
<details><summary>Indice 3</summary>

```python
def first_max(row):
    best = 0
    for i in range(1, len(row)):
        ...   # a STRICTLY larger value replaces best: the first maximum wins ties
    return best
```
Puis `np.array([first_max(row) for row in values])` pour `axis=1` ; pour `axis=0`, la même chose sur les lignes de `values.T` (les colonnes) ; pour `axis=None`, `first_max` sur le tableau aplati.

</details>

### Ex 0A.65 — `utils.one_hot` : des labels aux vecteurs 🔨

<details><summary>Indice 1</summary>

Une matrice de zéros d'une ligne par exemple et d'une colonne par classe, puis un seul 1 par ligne.

</details>
<details><summary>Indice 2</summary>

Valide : `y` à une dimension, entiers (`np.all(y == np.round(y))`), positifs, `< n_classes`. `n_classes` par défaut : `int(y.max()) + 1`. Puis `M = np.zeros((len(y), n_classes), dtype=dtype)`.

</details>
<details><summary>Indice 3</summary>

`M[np.arange(len(y)), y.astype(int)] = 1` : la ligne `i` reçoit un 1 dans la colonne `y[i]`.

</details>

### Ex 0A.66 — `utils.iterate_minibatches` : découper un dataset en mini-batches 🔨

<details><summary>Indice 1</summary>

Deux étapes : fabriquer l'**ordre** des indices (mélangé ou non), puis le **couper** en tranches consécutives de `batch_size`.

</details>
<details><summary>Indice 2</summary>

`order = rng.permutation(n_samples)` si `shuffle` (avec `rng = np.random.default_rng()` si `rng` vaut `None`), sinon `np.arange(n_samples)`. Tranches : `order[start:start + batch_size]` pour `start` dans `range(0, n_samples, batch_size)`. Avec `drop_last`, retire le dernier batch s'il est plus court.

</details>
<details><summary>Indice 3</summary>

```python
batches = [order[start:start + batch_size] for start in range(0, n_samples, batch_size)]
# drop_last: remove the last batch if it is shorter than batch_size
return batches
```
N'oublie pas les `ValueError` pour `n_samples < 1` ou `batch_size < 1`.

</details>

### Ex 0A.67 — Enquête : dix questions sur les manchots, dix réponses vérifiées 🏆

<details><summary>Indice 1</summary>

Pour chaque question, identifie la **population** (tous les manchots ? les femelles ? les Gentoo ? les masses connues ?), puis le calcul (compter, moyenne, maximum…) et le format (nom, nombre, proportion).

</details>
<details><summary>Indice 2</summary>

Outils : un filtre, puis `value_counts().idxmax()` ou `groupby(...).mean().idxmax()` ; `idxmin()` donne le **label** de la ligne du minimum, qu'on lit avec `.loc` ; une proportion est la moyenne d'un masque.

</details>
<details><summary>Indice 3</summary>

a) `penguins.loc[penguins["sex"] == "female", "island"].value_counts().idxmax()` ; c) `lightest = penguins.loc[penguins["body_mass_g"].idxmin()]` ; h) `(penguins["sex"].isna() & (penguins["island"] == "Dream")).sum()`.

</details>
