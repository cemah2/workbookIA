# 0B · Maths du lycée au ML — quiz, rappels, exercices papier et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch00b_maths/06_mes_reponses.md` (créée par `python tools/start_chapter.py 0B`), jamais dans ce fichier : il est mis à jour par Claude. Commence par **0B.32** (LaTeX), pour écrire tes formules proprement.
> Les réponses des exercices ✏️ se vérifient dans la **partie 0** du notebook (`wb.check`) ; les quiz, les rappels, les démonstrations ∂, les exercices 🧮 🗣️ 🛠️ et les questions d'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée, sauf mention « sans calculatrice ».

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🧮 🗣️ 🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, puis vérifie dans `05_solutions.md`.

### 0B.Q1 — Puissances, racines et notation scientifique : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §101.1.1 · parcours R, M*

1. $2^3 \times 2^4 = 2^{12}$.
2. $10^{-2} = 0{,}01$.
3. $\sqrt{a + b} = \sqrt{a} + \sqrt{b}$ pour tous nombres positifs $a$ et $b$.
4. $2^{20}$ vaut environ un million.
5. $4{,}7 \times 10^{7}$ s'écrit `4.7e7` en Python.

### 0B.Q2 — |x|, ⌊x⌋, ⌈x⌉ et sign(x) : pièges de signe 🧠 ⏱️ 3 min
*Fiche §101.1.2 · parcours R, M*

Vrai ou faux ?
1. $|-5| = -5$.
2. $\lfloor -1{,}5 \rfloor = -1$.
3. $\lceil 4 \rceil = 4$.
4. Si $-3x > 12$, alors $x < -4$.
5. $|x - 2| < 1$ veut dire que $x$ est dans l'intervalle $]1, 3[$.

### 0B.Q3 — Lire une formule avec Σ, Π et une moyenne pondérée 🧠 ⏱️ 3 min
*Fiche §101.1.3, §101.1.4 · parcours R, M*

1. Combien de termes compte $\sum_{i=3}^{7} x_i$ ?
2. Que vaut $\prod_{k=1}^{3} k$ ?
3. $\sum_{i=1}^{n} 2 x_i = 2 \sum_{i=1}^{n} x_i$ : vrai ou faux ?
4. Dans $\bar{x}_w = \frac{\sum_i w_i x_i}{\sum_i w_i}$, que devient la formule si tous les poids $w_i$ valent 1 ?
5. Dans $z = \sum_i w_i x_i + b$, que représentent les $w_i$ et $b$ pour un neurone ?

### 0B.Q4 — Suite géométrique : elle fond ou elle explose ? 🧠 ⏱️ 3 min
*Fiche §101.1.5 · parcours R, M*

Pour chaque raison $q$, dis si, quand $k$ grandit, $q^k$ **fond** vers 0, **explose** (sa valeur absolue devient aussi grande qu'on veut), reste **constant**, ou **oscille** sans fondre ni exploser.
1. $q = 0{,}99$
2. $q = 1{,}01$
3. $q = -0{,}5$
4. $q = -1$
5. $q = 1$

### 0B.Q5 — Ensembles et dénombrement : le bon réflexe 🧠 ⏱️ 3 min
*Fiche §101.1.6, §101.1.7 · parcours R, M*

1. Quelle opération utilises-tu pour « les éléments communs à $A$ et $B$ » ?
2. $|A \cup B| = |A| + |B|$ : dans quel cas est-ce vrai ?
3. Combien de façons de ranger 4 livres différents sur une étagère ?
4. Combien de paires différentes peut-on former avec 5 features ?
5. $\binom{n}{k}$ tient-il compte de l'ordre des objets choisis ?

### 0B.Q6 — Reconnaître l'allure d'une courbe 🧠 ⏱️ 3 min
*Fiche §101.2 · parcours R, M*

Associe chaque description à une fonction parmi : $2x - 1$, $x^2 - 4$, $e^x$, $\ln x$, $\sigma(x)$, $\tanh(x)$, $\cos x$.
1. Une courbe en « S » qui va de 0 à 1 et passe par $\frac{1}{2}$ en 0.
2. Une vague qui se répète tous les $2\pi$.
3. Une courbe définie seulement pour $x > 0$, qui passe par $(1, 0)$.
4. Une parabole tournée vers le haut, qui coupe l'axe en $-2$ et $2$.
5. Une courbe en « S » qui va de $-1$ à $1$ et passe par l'origine.

### 0B.Q7 — Règles des logarithmes et des exponentielles : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §101.2.3, §101.2.4, §101.2.7 · parcours R, M*

1. $e^{a+b} = e^a + e^b$.
2. $\ln(ab) = \ln a + \ln b$ pour $a, b > 0$.
3. $\ln(a + b) = \ln a + \ln b$ pour $a, b > 0$.
4. `np.log(100)` vaut 2.
5. Si $f(x) = x + 1$ et $g(u) = u^2$, alors $(g \circ f)(2) = 9$.

### 0B.Q8 — Vecteurs : norme, produit scalaire, cosinus, Hadamard 🧠 ⏱️ 3 min
*Fiche §101.3 · parcours R, M*

1. Le produit scalaire de deux vecteurs est-il un nombre ou un vecteur ? Et leur produit de Hadamard ?
2. Que vaut $\|(6, 8)\|$ ?
3. Deux vecteurs de produit scalaire nul sont… ?
4. Si l'on multiplie $\mathbf{a}$ par 10, que devient $\cos(\mathbf{a}, \mathbf{b})$ ?
5. Quelle est la similarité cosinus d'un vecteur non nul avec lui-même ?

### 0B.Q9 — Formes compatibles : ce produit existe-t-il ? 🧠 ⏱️ 3 min
*Fiche §101.4 · parcours R, M*

$\mathbf{A}$ a la forme $(3, 4)$, $\mathbf{B}$ la forme $(4, 2)$, $\mathbf{v}$ est un vecteur de dimension 4. Pour chaque produit, donne la forme du résultat ou dis qu'il n'existe pas.
1. $\mathbf{A}\mathbf{B}$
2. $\mathbf{B}\mathbf{A}$
3. $\mathbf{A}\mathbf{v}$
4. $\mathbf{A}^\top\mathbf{A}$
5. $\mathbf{B}^\top\mathbf{A}^\top$

### 0B.Q10 — Dérivée : pente, signe, extremum 🧠 ⏱️ 3 min
*Fiche §101.5 · parcours R, M*

1. Que représente géométriquement $f'(a)$ ?
2. Si $f'(x) < 0$ sur un intervalle, que fait $f$ ?
3. $(x^3)'$ = ?
4. Si $f'(a) = 0$, $f$ a-t-elle forcément un minimum ou un maximum en $a$ ?
5. $(e^{2x})'$ = ?

### 0B.Q11 — Gradient et lignes de niveau : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §101.6 · parcours R, M*

1. Le gradient d'une fonction de deux variables est un nombre.
2. Le gradient pointe dans la direction où la fonction augmente le plus vite.
3. Pour faire baisser une loss, on se déplace dans le sens du gradient.
4. Le long d'une ligne de niveau, la fonction garde la même valeur.
5. Pour calculer $\frac{\partial f}{\partial x}$, on traite $y$ comme une constante.

### 0B.Q12 — Probabilités : indépendance, espérance, variance 🧠 ⏱️ 3 min
*Fiche §101.7 · parcours R, M*

1. Pour deux événements indépendants, $P(A \cap B)$ = ?
2. Deux événements incompatibles, chacun de probabilité non nulle, peuvent-ils être indépendants ?
3. Quelle est l'espérance d'un dé équilibré à 6 faces ?
4. Si $\mathrm{Var}(X) = 4$, que vaut $\mathrm{Var}(3X + 1)$ ?
5. Qu'affirme la loi des grands nombres, en une phrase ?

<a id="rappels"></a>

## 🔁 Rappels

Des outils Python du chapitre 0A, relus avec les yeux de 0B. Réponds sur papier, puis vérifie dans une cellule de notebook.

### 0B.R1 — 0A : une somme en Python, trois façons (boucle, sum, compréhension) 🔁 ★ ⏱️ 5 min
*Chapitre 0A (0A.21, 0A.22) · fiche §101.1.3 · parcours R, C*

La formule $S = \sum_{i=1}^{5} i^2$ se programme de trois façons.
1. Écris-la avec une boucle `for` et une variable `total`.
2. Écris-la en une ligne avec `sum` et une expression génératrice.
3. Écris $\prod_{i=1}^{5} i$ avec `math.prod`.
4. Pourquoi `range(1, 6)` et pas `range(1, 5)` ?

### 0B.R2 — 0A : shape, ndim et axis d'un array 🔁 ★ ⏱️ 5 min
*Chapitre 0A (0A.27, 0A.51) · fiche §101.4.1 · parcours R, C*

`X` est le tableau des 333 manchots complets, avec 4 mesures par manchot, dans l'ordre : longueur du bec, épaisseur du bec, longueur de la nageoire, masse (0A.35).
1. Que valent `X.shape` et `X.ndim` ? Combien de lignes et de colonnes a la matrice $\mathbf{X}$ ?
2. Que représente `X[:, 2]`, et quelle est sa forme ?
3. Quelle est la forme de `X.mean(axis=0)` ? Que représente-t-il ?
4. Quelle est la forme de `X.T` ?

### 0B.R3 — 0A : une fonction qui prend une fonction (lambda, Callable) 🔁 ★ ⏱️ 5 min
*Chapitre 0A (0A.41, 0A.42) · fiche §101.2.7 · parcours R, C*

1. Écris `compose(g, f)`, qui renvoie la fonction $x \mapsto g(f(x))$.
2. Avec `f = lambda x: 2 * x` et `g = lambda u: u + 3`, que valent `compose(g, f)(5)` et `compose(f, g)(5)` ?
3. Quelle annotation de type écrirais-tu pour le paramètre `f`, qui prend un nombre et renvoie un nombre ?

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Fais-les **à la main**, en écrivant chaque étape dans ta copie de `06_mes_reponses.md` (en LaTeX, 0B.32). Pour les ✏️, reporte ensuite **la valeur** trouvée dans la **partie 0 du notebook**, qui la vérifie avec `wb.check` (par exemple `answer_0B_1a = 42` si tu as trouvé 42). Les démonstrations ∂ se corrigent avec `05_solutions.md`.

Dans le notebook, écris un vecteur comme une liste (`[7, -2]`), une matrice comme une liste de lignes (`[[1, 0], [0, 1]]`), une forme comme un tuple (`(4, 5)`), et « vrai » ou « faux » comme `True` ou `False`. Arrondis comme indiqué ; sinon, donne la valeur exacte.

### Premier passage (★)

### Ex 0B.1 — Puissances, racines et notation scientifique sans calculatrice ✏️ ★ ⏱️ 10 min
**Objectif :** calculer avec les règles des puissances et la notation scientifique, sans calculatrice.
**Prérequis :** fiche §101.1.1 · **Parcours :** M

| | Calcul |
|---|---|
| a | $2^5 \times 2^3$ |
| b | $\frac{(10^2)^3}{10^4}$ |
| c | $5^{-2}$ (sous forme décimale) |
| d | $\sqrt{49 \times 16}$ |
| e | $8^{2/3}$ (indice : $8 = 2^3$) |
| f | $(3{,}2 \times 10^5) \times (2 \times 10^{-3})$ (sous forme décimale) |
| g | l'exposant $k$ tel que $0{,}00056 = 5{,}6 \times 10^{k}$ |
| h | l'exposant de l'ordre de grandeur de $2^{30}$ (la puissance $k$ telle que $2^{30} \approx 10^k$) |

### Ex 0B.2 — Valeur absolue, partie entière et signe : tableau de valeurs ✏️ ★ ⏱️ 10 min
**Objectif :** évaluer sans erreur de signe valeurs absolues, parties entières et fonction signe.
**Prérequis :** fiche §101.1.2 · **Parcours :** M

| | Calcul |
|---|---|
| a | $\lvert -7 \rvert + \lvert 3 - 5 \rvert$ |
| b | $\lfloor 3{,}7 \rfloor$ |
| c | $\lfloor -3{,}7 \rfloor$ |
| d | $\lceil -3{,}7 \rceil$ |
| e | $\lceil 2{,}01 \rceil$ |
| f | $\mathrm{sign}(-0{,}5) + \mathrm{sign}(0) + \mathrm{sign}(12)$ |
| g et h | les bornes inférieure (g) et supérieure (h) de l'intervalle des $x$ tels que $\lvert x - 4 \rvert \leq 1{,}5$ |
| i | le nombre d'entiers relatifs $x$ (positifs, négatifs ou nuls) tels que $\lvert x \rvert < 3$ |

### Ex 0B.3 — Lire et calculer des Σ et des Π ✏️ ★ ⏱️ 15 min
**Objectif :** traduire $\Sigma$ et $\Pi$ en additions et multiplications, et utiliser leurs règles de calcul.
**Prérequis :** 0B.R1 · fiche §101.1.3 · **Parcours :** R, M

Avec $\mathbf{x} = (x_1, x_2, x_3, x_4) = (2, -1, 4, 3)$ :

| | Calcul |
|---|---|
| a | $\sum_{i=1}^{5} i$ |
| b | $\sum_{i=1}^{4} (2i - 1)$ |
| c | $\sum_{k=0}^{3} 3^k$ |
| d | $\prod_{i=1}^{4} (i + 1)$ |
| e | $\sum_{i=1}^{4} x_i^2$ |
| f | $\sum_{i=1}^{4} (3x_i + 1)$ (sers-toi de $\sum_i x_i$) |
| g | $\sum_{i=1}^{100} 5$ |
| h | $\sum_{i=2}^{4} i\,x_i$ (attention aux bornes) |

### Ex 0B.4 — Moyenne pondérée, somme pondérée et moyenne mobile à la main ✏️ ★ ⏱️ 15 min
**Objectif :** calculer moyenne, moyenne pondérée, somme pondérée d'un neurone et moyenne mobile.
**Prérequis :** Ex 0B.3 · fiche §101.1.4 · **Parcours :** R, M

| | Question |
|---|---|
| a | La moyenne de 4, 8, 6, 10 et 2. |
| b | La moyenne pondérée des notes 14 (coefficient 3), 8 (coefficient 1) et 11 (coefficient 2). |
| c | La sortie $z = \sum_i w_i x_i + b$ d'un neurone, avec $\mathbf{w} = (0{,}2 ; -0{,}5 ; 1)$, $\mathbf{x} = (10 ; 4 ; 3)$ et $b = -1$. |
| d | Les moyennes mobiles d'ordre 3 de la série $2, 4, 9, 1, 5, 6$ : la liste $(m_3, m_4, m_5, m_6)$, à 2 décimales. |
| e | Combien de valeurs donne la moyenne mobile d'ordre $k = 7$ d'une série de $n = 50$ valeurs ? |

### Ex 0B.5 — Ensembles : union, intersection, complémentaire et cardinal ✏️ ★ ⏱️ 10 min
**Objectif :** lister et compter les éléments d'unions, d'intersections et de complémentaires.
**Prérequis :** fiche §101.1.6 · **Parcours :** M

Dans $\Omega = \{1, 2, \dots, 12\}$, soient $A$ les multiples de 2, $B$ les multiples de 3 et $C = \{1, 2, 3\}$.

| | Question |
|---|---|
| a | $\lvert A \cap B \rvert$ |
| b | $\lvert A \cup B \rvert$ (avec la formule, puis vérifie en listant) |
| c | $\lvert \bar{A} \rvert$ |
| d | $\lvert \bar{A} \cap \bar{B} \rvert$ (ni multiple de 2, ni multiple de 3) |
| e | l'ensemble $A \cap C$ (dans le notebook, un ensemble Python : `{…}`) |
| f | $\lvert A \cup B \cup C \rvert$ |

### Ex 0B.6 — Droites et paraboles : pente, ordonnée à l'origine, racines, sommet ✏️ ★ ⏱️ 15 min
**Objectif :** trouver l'équation d'une droite et les éléments remarquables d'une parabole.
**Prérequis :** fiche §101.2.1, §101.2.2 · **Parcours :** R, M

La droite $d$ passe par $(-1, 5)$ et $(3, -3)$. La parabole est $f(x) = 2x^2 - 8x + 6$.

| | Question |
|---|---|
| a | La pente de $d$. |
| b | Son ordonnée à l'origine. |
| c | Sa racine (où elle coupe l'axe des abscisses). |
| d | Le discriminant de $f$. |
| e et f | La plus petite (e) et la plus grande (f) racine de $f$. |
| g | L'abscisse du sommet de $f$. |
| h | La valeur minimale de $f$. |

### Ex 0B.7 — Cosinus : cercle, période et planning en cosinus ✏️ ★ ⏱️ 15 min
**Objectif :** lire des cosinus sur le cercle trigonométrique et évaluer un planning en cosinus.
**Prérequis :** fiche §101.2.6 · **Parcours :** M

On donne $\cos\frac{\pi}{4} = \frac{\sqrt{2}}{2} \approx 0{,}707$, et on rappelle que $\cos(\pi - x) = -\cos x$ (deux points du cercle symétriques par rapport à l'axe vertical).

| | Question |
|---|---|
| a | 60° en radians (2 décimales). |
| b | $\cos \pi$ |
| c | $\cos\frac{2\pi}{3}$ |
| d | $\cos\left(-\frac{\pi}{3}\right)$ |
| e | $\cos(4\pi)$ |
| f | Le facteur du planning en cosinus $\frac{1}{2}\left(1 + \cos\frac{\pi t}{T}\right)$ pour $T = 100$ et $t = 50$. |
| g | Le même facteur pour $t = 25$ (2 décimales). |
| h | Le learning rate à $t = 75$, si le learning rate maximal (le facteur 1) vaut 0,01 (4 décimales). |

### Ex 0B.8 — Vecteurs : somme, multiple, norme et distance entre deux manchots ✏️ ★ ⏱️ 10 min
**Objectif :** calculer sommes, normes et distances de vecteurs, et voir l'effet des unités.
**Prérequis :** fiche §101.3.1, §101.3.2 · **Fil rouge :** Penguins · **Parcours :** R, M

Deux manchots, décrits par (longueur du bec, longueur de la nageoire) en mm : $\mathbf{p}_1 = (40 ; 190)$ et $\mathbf{p}_2 = (48 ; 196)$. Et $\mathbf{u} = (3, -4)$.

| | Question |
|---|---|
| a | $\mathbf{p}_1 + \mathbf{p}_2$ |
| b | Le « manchot moyen » $\frac{1}{2}(\mathbf{p}_1 + \mathbf{p}_2)$. |
| c, d, e | $\lVert \mathbf{u} \rVert_2$, $\lVert \mathbf{u} \rVert_1$ et $\lVert \mathbf{u} \rVert_\infty$. |
| f | La distance entre $\mathbf{p}_1$ et $\mathbf{p}_2$. |
| g | La même distance si la nageoire est mesurée en **cm** : $\mathbf{p}_1 = (40 ; 19)$, $\mathbf{p}_2 = (48 ; 19{,}6)$ (2 décimales). Qu'est-ce qui a changé ? |
| h | Le vecteur unitaire de même direction que $\mathbf{u}$. |

### Ex 0B.9 — Transposée et produit matrice-vecteur : deux lectures ✏️ ★ ⏱️ 15 min
**Objectif :** transposer une matrice et calculer un produit matrice-vecteur, par lignes et par colonnes.
**Prérequis :** 0B.R2, Ex 0B.8 · fiche §101.4.1, §101.4.2 · **Parcours :** R, M

$$\mathbf{A} = \begin{pmatrix} 2 & 0 & 1 \\ -1 & 3 & 2 \end{pmatrix} \qquad \mathbf{v} = (1, 2, -1) \qquad \mathbf{X} = \begin{pmatrix} 1 & 2 \\ 3 & 0 \\ 0 & -1 \end{pmatrix} \qquad \mathbf{w} = (0{,}5 ; 2)$$

| | Question |
|---|---|
| a | La forme de $\mathbf{A}^\top$. |
| b | L'élément $(\mathbf{A}^\top)_{31}$ (ligne 3, colonne 1, indices mathématiques). |
| c | $\mathbf{A}\mathbf{v}$, calculé ligne par ligne ; puis refais-le comme combinaison des colonnes de $\mathbf{A}$. |
| d | $\mathbf{A}^\top\mathbf{t}$ avec $\mathbf{t} = (1, -1)$. |
| e | Le produit $\mathbf{A}\mathbf{s}$ avec $\mathbf{s} = (1, 2)$ est-il défini ? (`True` ou `False`) |
| f | Les prédictions $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$ d'un modèle linéaire pour les 3 exemples de $\mathbf{X}$, avec $b = 1$. |

### Ex 0B.10 — Taux d'accroissement : de la sécante à la tangente ✏️ ★ ⏱️ 15 min
**Objectif :** calculer des taux d'accroissement de plus en plus proches et en déduire une dérivée.
**Prérequis :** Ex 0B.6 · fiche §101.5.1 · **Parcours :** R, M

Soit $f(x) = x^2 + x$ et $a = 2$ (on a $f(2) = 6$).

| | Question |
|---|---|
| a | Le taux d'accroissement entre $2$ et $2 + h$ pour $h = 1$. |
| b | Pour $h = 0{,}1$. |
| c | Pour $h = 0{,}01$. |
| d | Simplifie $\frac{f(2 + h) - f(2)}{h}$ en fonction de $h$ (dans ta copie), et déduis $f'(2)$ (la valeur à reporter dans le notebook). |
| e | L'ordonnée à l'origine de la tangente en $a = 2$ (écris son équation $y = f(2) + f'(2)(x - 2)$). |
| f | L'approximation linéaire de $f(2{,}05)$ par la tangente (valeur exacte). Compare-la avec la vraie valeur $f(2{,}05)$ (4 décimales). |

### Ex 0B.11 — Probabilités : issues, complémentaire, union ✏️ ★ ⏱️ 10 min
**Objectif :** calculer des probabilités en comptant des issues équiprobables, avec le complémentaire et l'union.
**Prérequis :** Ex 0B.5 · fiche §101.7.1 · **Parcours :** R, M

On lance deux dés équilibrés (36 issues équiprobables). Réponses à 3 décimales.

| | Question |
|---|---|
| a | $P(\text{somme} = 8)$ |
| b | $P(\text{double})$ (les deux dés donnent le même nombre) |
| c | $P(\text{au moins un } 1)$ (passe par le complémentaire) |
| d | $P(\text{somme} = 8 \text{ ou double})$ |
| e | $P(\text{somme} \geq 10)$ |
| f | On tire une carte d'un jeu de 52 cartes : $P(\text{cœur ou figure})$ (les figures sont les valets, dames et rois). |

### Deuxième passage (★★)

### Ex 0B.12 — Dénombrer : choix successifs, factorielle et C(n, k) ✏️ ★★ ⏱️ 20 min
**Objectif :** choisir entre principe multiplicatif, factorielle et coefficient binomial pour compter.
**Prérequis :** Ex 0B.1 · fiche §101.1.7 · **Parcours :** M

| | Question |
|---|---|
| a | Combien de codes à 4 chiffres (de 0000 à 9999) ? |
| b | Combien de codes à 4 chiffres **tous différents** ? |
| c | $6!$ |
| d | $\binom{6}{2}$ |
| e | $\binom{8}{3}$ |
| f | Combien de classifieurs « un contre un » faut-il pour 5 classes (un par paire de classes) ? |
| g | Combien de sous-ensembles de features peut-on former avec 5 features (l'ensemble vide et l'ensemble complet compris) ? |
| h | De combien de façons peut-on choisir, parmi 10 personnes, un comité de 3 dont un président ? |

### Ex 0B.13 — Suites géométriques : ce qui fond, ce qui explose ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer des termes et des sommes de suites géométriques et estimer leur vitesse.
**Prérequis :** Ex 0B.1 · fiche §101.1.5 · **Parcours :** M

| | Question |
|---|---|
| a | $u_0 = 1000$ et $q = 0{,}5$ : $u_5$. |
| b | $0{,}9^{20}$ (3 décimales) : que reste-t-il d'un signal multiplié par 0,9 à chacune des 20 couches d'un réseau ? |
| c | $1{,}1^{10}$ (2 décimales). |
| d | $\sum_{k=0}^{9} 0{,}5^k$ avec la formule (3 décimales). |
| e | La somme infinie $\sum_{k=0}^{\infty} 0{,}95^k$ (la valeur totale d'une récompense 1 à chaque étape, actualisée par $\gamma = 0{,}95$). |
| f | Le plus petit entier $k$ tel que $0{,}5^k < 0{,}001$. |
| g | $(-0{,}8)^7$ (3 décimales) : quel est son signe, et pourquoi ? |

### Ex 0B.14 — La somme géométrique démontrée pas à pas ∂ ★★ ⏱️ 15 min
**Objectif :** démontrer la formule de la somme géométrique et en déduire la somme infinie.
**Prérequis :** Ex 0B.3, Ex 0B.13 · fiche §101.1.5 · **Parcours :** M

Soit $q \neq 1$ et $S_n = \sum_{k=0}^{n-1} q^k = 1 + q + q^2 + \dots + q^{n-1}$.
1. Écris $q\,S_n$ en développant, terme par terme.
2. Calcule $S_n - q\,S_n$ : quels termes se simplifient ? Déduis-en $S_n = \frac{1 - q^n}{1 - q}$.
3. Vérifie la formule pour $q = 2$ et $n = 4$ en additionnant directement.
4. On suppose $|q| < 1$ : que devient $q^n$ quand $n$ grandit ? Déduis-en $\sum_{k=0}^{\infty} q^k = \frac{1}{1 - q}$.
5. Application : combien vaut $0{,}999\ldots = 0{,}9 + 0{,}09 + 0{,}009 + \dots$ ? (écris-la comme $0{,}9 \sum_{k=0}^{\infty} 0{,}1^k$)

### Ex 0B.15 — Exponentielles et logarithmes : règles de calcul en bases 2, e et 10 ✏️ ★★ ⏱️ 20 min
**Objectif :** appliquer les règles des exponentielles et des logarithmes, et résoudre $b^x = c$.
**Prérequis :** Ex 0B.1 · fiche §101.2.3, §101.2.4 · **Parcours :** R, M

Sans calculatrice pour a à e.

| | Question |
|---|---|
| a | $\ln(e^3)$ |
| b | $e^{\ln 5}$ |
| c | $\ln 1 + \log_2 32$ |
| d | $\log_{10} 0{,}001$ |
| e | $\ln(e^2 \times e^5)$ |
| f | La solution de $e^x = 20$ (2 décimales). |
| g | La solution de $2^x = 1000$ (2 décimales). Puis, dans ta copie seulement : combien de bits faut-il pour numéroter 1000 objets ? |
| h | $\ln(0{,}01^{100})$ (1 décimale), calculé **sans** calculer $0{,}01^{100}$. |
| i | 1 nat exprimé en bits (3 décimales). |

### Ex 0B.16 — Changer de base : log₂ x = ln x / ln 2, bits et nats ∂ ★★ ⏱️ 15 min
**Objectif :** démontrer la formule de changement de base et convertir des bits en nats.
**Prérequis :** Ex 0B.15 · fiche §101.2.4 · **Parcours :** M

1. Soit $y = \log_2 x$, c'est-à-dire $2^y = x$. Prends le logarithme népérien des deux membres et déduis-en $\log_2 x = \frac{\ln x}{\ln 2}$.
2. Généralise : montre que $\log_b x = \frac{\ln x}{\ln b}$ pour toute base $b > 0$, $b \neq 1$.
3. Montre que $\log_2 x = \log_2 10 \times \log_{10} x$.
4. Une quantité d'information vaut $I$ nats, c'est-à-dire $I = \ln\frac{1}{p}$ pour un événement de probabilité $p$ ; la même quantité en bits vaut $\log_2\frac{1}{p}$. Montre que la valeur en bits est égale à $\frac{I}{\ln 2} \approx 1{,}443\,I$.
5. Application : un événement de probabilité $\frac{1}{8}$ apporte combien de bits ? Combien de nats (2 décimales) ?

### Ex 0B.17 — Sigmoïde et tanh : valeurs, limites, symétries ✏️ ★★ ⏱️ 15 min
**Objectif :** évaluer la sigmoïde et tanh, utiliser leurs symétries et inverser la sigmoïde.
**Prérequis :** Ex 0B.15 · fiche §101.2.5 · **Parcours :** M

Réponses à 3 décimales, sauf mention.

| | Question |
|---|---|
| a | $\sigma(0)$ |
| b | $\sigma(3)$ |
| c | $\sigma(-3)$, en utilisant la symétrie $\sigma(-x) = 1 - \sigma(x)$. |
| d | $\tanh(0{,}5)$ |
| e | $2\sigma(1) - 1$ : compare avec d. |
| f | La limite de $\sigma(x)$ quand $x \to -\infty$. |
| g | Le nombre $x$ tel que $\sigma(x) = 0{,}9$ (2 décimales ; résous $\frac{1}{1 + e^{-x}} = 0{,}9$ pas à pas). |
| h | $\sigma(x) + \sigma(-x)$, pour tout $x$. |

### Ex 0B.18 — Produit scalaire, similarité cosinus et produit de Hadamard ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer produits scalaires, normes, similarités cosinus et produits de Hadamard, et les interpréter.
**Prérequis :** Ex 0B.8, Ex 0B.7 · fiche §101.3.3, §101.3.4 · **Parcours :** R, M

Avec $\mathbf{a} = (1, 2, 2)$, $\mathbf{b} = (2, 0, -1)$, $\mathbf{c} = (2, 4, 4)$ et $\mathbf{d} = (1, 0, 0)$ :

| | Question |
|---|---|
| a | $\mathbf{a} \cdot \mathbf{b}$ |
| b | $\mathbf{a}$ et $\mathbf{b}$ sont-ils orthogonaux ? (`True` ou `False`) |
| c | $\lVert \mathbf{a} \rVert$ |
| d | $\mathbf{a} \cdot \mathbf{c}$ |
| e | $\cos(\mathbf{a}, \mathbf{c})$ : pourquoi ce résultat, sans calcul ? |
| f | $\cos(\mathbf{a}, \mathbf{d})$ (3 décimales) |
| g | $\mathbf{a} \odot \mathbf{b}$ |
| h | La somme des composantes de $\mathbf{a} \odot \mathbf{c}$ : compare avec d. |
| i | $\cos(\mathbf{a}, -\mathbf{a})$ |

### Ex 0B.19 — Développer ‖a − b‖² avec le produit scalaire ∂ ★★ ⏱️ 15 min
**Objectif :** relier distance et produit scalaire par un calcul algébrique.
**Prérequis :** Ex 0B.18 · fiche §101.3.2, §101.3.3 · **Parcours :** M

1. En partant de $\lVert \mathbf{x} \rVert^2 = \mathbf{x} \cdot \mathbf{x}$ et des propriétés du produit scalaire, montre que $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = \lVert \mathbf{a} \rVert^2 + \lVert \mathbf{b} \rVert^2 - 2\,\mathbf{a} \cdot \mathbf{b}$.
2. Vérifie la formule sur $\mathbf{a} = (1, 2, 2)$ et $\mathbf{b} = (2, 0, -1)$ (0B.18).
3. Déduis-en que si $\mathbf{a}$ et $\mathbf{b}$ sont **unitaires**, $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = 2 - 2\cos(\mathbf{a}, \mathbf{b})$ : pour des vecteurs normalisés, chercher le plus proche voisin revient à chercher la plus grande similarité cosinus.
4. Quel théorème du collège retrouves-tu quand $\mathbf{a} \cdot \mathbf{b} = 0$ ?

### Ex 0B.20 — Produit matriciel : calculer et vérifier les formes ✏️ ★★ ⏱️ 20 min
**Objectif :** vérifier la compatibilité des formes et calculer des produits matriciels à la main.
**Prérequis :** Ex 0B.9 · fiche §101.4.3 · **Parcours :** R, M

$$\mathbf{A} = \begin{pmatrix} 1 & 2 \\ 0 & -1 \\ 3 & 1 \end{pmatrix} \qquad \mathbf{B} = \begin{pmatrix} 2 & 1 & 0 \\ 1 & -1 & 4 \end{pmatrix} \qquad \mathbf{C} = \begin{pmatrix} 1 & 1 \\ 2 & 0 \end{pmatrix}$$

| | Question |
|---|---|
| a | La forme de $\mathbf{A}\mathbf{B}$. |
| b | La forme de $\mathbf{B}\mathbf{A}$. |
| c | $\mathbf{B}\mathbf{A}$ |
| d | $(\mathbf{A}\mathbf{B})_{11}$ (indices mathématiques), sans calculer tout $\mathbf{A}\mathbf{B}$. |
| e | $(\mathbf{A}\mathbf{B})_{32}$ |
| f | $\mathbf{C}\mathbf{A}$ est-il défini ? (`True` ou `False`) |
| g | $\mathbf{A}\mathbf{C}$ |
| h | Le nombre de multiplications nécessaires pour calculer $\mathbf{A}\mathbf{B}$ entièrement. |

### Ex 0B.21 — Identité, inverse 2 × 2 et système de deux équations ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer un déterminant et un inverse $2 \times 2$, et résoudre un système linéaire.
**Prérequis :** Ex 0B.20 · fiche §101.4.4 · **Parcours :** M

Soit $\mathbf{M} = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}$.

| | Question |
|---|---|
| a | Le déterminant de $\mathbf{M}$. |
| b | $\mathbf{M}^{-1}$ ; vérifie que $\mathbf{M}\mathbf{M}^{-1} = \mathbf{I}$. |
| c | La solution $(x, y)$ du système $3x + y = 5$, $4x + 2y = 6$. |
| d | $\begin{pmatrix} 2 & 4 \\ 1 & 2 \end{pmatrix}$ est-elle inversible ? (`True` ou `False`) |
| e | La valeur de $k$ pour laquelle $\begin{pmatrix} 1 & k \\ 2 & 6 \end{pmatrix}$ n'est pas inversible. |
| f | L'inverse de $\begin{pmatrix} 2 & 0 \\ 0 & 5 \end{pmatrix}$. |

### Ex 0B.22 — Dériver avec les règles : somme, produit, quotient, exp, ln ✏️ ★★ ⏱️ 25 min
**Objectif :** dériver avec le tableau des dérivées usuelles et les règles du produit et du quotient.
**Prérequis :** Ex 0B.10, Ex 0B.15 · fiche §101.5.2 · **Parcours :** R, M

Pour chaque fonction, écris la dérivée dans ta copie, puis donne sa **valeur** au point indiqué (3 décimales si elle n'est pas exacte ; prends $e$ avec la touche de ta calculatrice, pas 2,718).

| | Fonction | Point |
|---|---|---|
| a | $f(x) = 4x^3 - 2x + 7$ | $x = 1$ |
| b | $g(x) = \sqrt{x} + \frac{1}{x}$ | $x = 4$ (4 décimales) |
| c | $h(x) = x^2\,e^x$ | $x = 1$ |
| d | $k(x) = x \ln x$ | $x = e$ |
| e | $q(x) = \frac{x + 1}{x - 1}$ | $x = 3$ |
| f | $r(x) = \frac{e^x}{x}$ | $x = 1$ |
| g | $s(x) = 5x^4 - 3$ | $x = 2$ |

### Ex 0B.23 — Règle de la chaîne : décomposer, puis dériver ✏️ ★★ ⏱️ 25 min
**Objectif :** dériver une composition en la décomposant en étapes, jusqu'à la dérivée d'une loss.
**Prérequis :** Ex 0B.22, 0B.R3 · fiche §101.5.3, §101.2.7 · **Parcours :** R, M

Pour chaque fonction : 1. nomme l'étape intérieure ($u = \dots$) ; 2. dérive chaque étape ; 3. multiplie ; 4. donne la valeur au point indiqué (3 décimales si elle n'est pas exacte ; $e$ et $e^{-0{,}5}$ à la calculatrice).

| | Fonction | Point |
|---|---|---|
| a | $(3x - 1)^4$ | $x = 1$ |
| b | $e^{2x + 1}$ | $x = 0$ |
| c | $\ln(x^2 + 1)$ | $x = 2$ |
| d | $\sqrt{1 + 4x}$ | $x = 2$ |
| e | $e^{-x^2/2}$ | $x = 1$ |
| f | $\frac{1}{x^2 + 1}$ | $x = 1$ |
| g | la loss $L(w) = (3w + 1 - 4)^2$, dérivée par rapport à $w$ | $w = 2$ |
| h | $(\ln x)^2$ | $x = e$ |

### Ex 0B.24 — La moyenne minimise la somme des carrés des écarts ∂ ★★ ⏱️ 20 min
**Objectif :** montrer, en annulant une dérivée, que la moyenne est la meilleure prédiction constante au sens des moindres carrés.
**Prérequis :** Ex 0B.23, Ex 0B.4 · fiche §101.5.4, §101.1.3 · **Parcours :** M

On a $n$ mesures $y_1, \dots, y_n$ et on cherche le nombre $a$ qui les résume le mieux, au sens où il minimise $S(a) = \sum_{i=1}^{n} (y_i - a)^2$.
1. Calcule $S'(a)$ (dérive chaque terme avec la règle de la chaîne).
2. Résous $S'(a) = 0$ : montre que la solution est $a = \bar{y} = \frac{1}{n}\sum_i y_i$.
3. Justifie que c'est un **minimum** : étudie le signe de $S'(a)$ de part et d'autre de $\bar{y}$ (ou remarque que $S$ est une parabole ; dans quel sens est-elle tournée ?).
4. Vérifie sur les mesures $2, 3, 7$ : calcule $S(4)$, $S(3)$ et $S(5)$.
5. En ML, prédire toujours $\bar{y}$ est le modèle le plus simple pour une régression (la *baseline*) : quelle erreur quadratique moyenne obtient-il, en fonction de la variance des $y_i$ (celle de 101.7.4, qui divise par $n$) ?

### Ex 0B.25 — Lignes de niveau, dérivées partielles, gradient et un pas de descente ✏️ ★★ ⏱️ 25 min
**Objectif :** calculer dérivées partielles et gradient, et effectuer un pas de descente de gradient.
**Prérequis :** Ex 0B.22 · fiche §101.6.1 à §101.6.3 · **Parcours :** R, M

Soit $f(x, y) = (x - 1)^2 + 2y^2$.

| | Question |
|---|---|
| a | $f(3, 1)$ |
| b | $\frac{\partial f}{\partial x}(3, 1)$ |
| c | $\frac{\partial f}{\partial y}(3, 1)$ |
| d | Le gradient $\nabla f(3, 1)$. |
| e | Sa norme (2 décimales). |
| f | Le point obtenu après un pas de descente de gradient depuis $(3, 1)$, avec $\eta = 0{,}1$. |
| g | La valeur de $f$ en ce nouveau point : a-t-elle baissé ? |
| h | Le point où $f$ est minimale (dans ta copie, dis aussi ce que vaut le gradient en ce point). |
| i et j | Pour $g(x, y) = x^2 y + y^3$ : $\frac{\partial g}{\partial x}(1, 2)$ (i) et $\frac{\partial g}{\partial y}(1, 2)$ (j). |

Dessine enfin, à main levée, les lignes de niveau 2 et 8 de $f$ (des ellipses centrées en $(1, 0)$) et la flèche du gradient en $(3, 1)$.

### Ex 0B.26 — Indépendance : tester P(A ∩ B) = P(A) P(B) avec deux dés ✏️ ★★ ⏱️ 15 min
**Objectif :** tester l'indépendance de deux événements par le calcul.
**Prérequis :** Ex 0B.11 · fiche §101.7.2 · **Parcours :** M

On lance deux dés équilibrés. $A$ = « le premier dé est pair », $B$ = « la somme vaut 7 », $C$ = « la somme vaut 8 ». Réponses à 3 décimales ; pour décider de l'indépendance, compare les **fractions exactes**, pas les arrondis.

| | Question |
|---|---|
| a | $P(A)$ |
| b | $P(B)$ |
| c | $P(A \cap B)$ (liste les issues) |
| d | $A$ et $B$ sont-ils indépendants ? (`True` ou `False`) |
| e | $P(C)$ |
| f | $P(A \cap C)$ |
| g | $A$ et $C$ sont-ils indépendants ? (`True` ou `False`) |
| h | L'écart $P(A \cap C) - P(A)\,P(C)$ (3 décimales). |

### Ex 0B.27 — Espérance et variance d'une variable discrète ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer espérance, variance et écart-type d'une variable discrète, et utiliser leurs propriétés.
**Prérequis :** Ex 0B.11, Ex 0B.3 · fiche §101.7.3, §101.7.4 · **Parcours :** R, M

Le nombre $X$ de messages reçus en une minute suit la loi :

| $x_k$ | 0 | 1 | 2 | 5 |
|---|---|---|---|---|
| $p_k$ | 0,4 | 0,3 | 0,2 | 0,1 |

| | Question |
|---|---|
| a | $\mathbb{E}[X]$ |
| b | $\mathbb{E}[X^2]$ |
| c | $\mathrm{Var}(X)$ (avec la formule $\mathbb{E}[X^2] - \mathbb{E}[X]^2$) |
| d | L'écart-type (2 décimales). |
| e | $\mathbb{E}[3X + 2]$ |
| f | $\mathrm{Var}(3X + 2)$ |
| g et h | Un dé équilibré dont les six faces portent 1, 1, 2, 3, 3, 3 : l'espérance (g) et la variance (h) du nombre obtenu, à 3 décimales (garde les fractions jusqu'au bout). |

### Ex 0B.28 — Variance : deux formules, et l'espérance est linéaire ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer la linéarité de l'espérance et la formule $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$.
**Prérequis :** Ex 0B.27 · fiche §101.7.3, §101.7.4 · **Parcours :** M

$X$ prend les valeurs $x_1, \dots, x_K$ avec les probabilités $p_1, \dots, p_K$ (et $\sum_k p_k = 1$) ; $a$ et $b$ sont deux nombres ; on note $\mu = \mathbb{E}[X]$.
1. À partir de $\mathbb{E}[aX + b] = \sum_k (a x_k + b)\,p_k$, montre que $\mathbb{E}[aX + b] = a\,\mu + b$.
2. Développe $(x_k - \mu)^2$ et montre que $\mathrm{Var}(X) = \sum_k (x_k - \mu)^2\,p_k = \mathbb{E}[X^2] - \mu^2$.
3. Montre que $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$ (commence par calculer $(aX + b) - \mathbb{E}[aX + b]$).
4. Vérifie les trois résultats sur la loi de 0B.27.
5. Explique pourquoi une variance est toujours positive ou nulle (regarde sa définition), puis déduis-en, avec la question 2, que $\mathbb{E}[X^2] \geq \mathbb{E}[X]^2$.

### Pour finir (★★★)

### Ex 0B.29 — Règle de la chaîne à deux variables : la somme sur les chemins ∂ ★★★ ⏱️ 30 min
**Objectif :** calculer une dérivée par la somme sur les chemins d'un graphe de calcul, comme le fait la rétropropagation.
**Prérequis :** Ex 0B.23, Ex 0B.25 · fiche §101.6.4, §101.6.2 · **Parcours :** M

**Partie A.** Soit $z = u^2 + u\,v$, avec $u = 2x$ et $v = x^2$.
1. Dessine le graphe de calcul ($x \to u$, $x \to v$, $u \to z$, $v \to z$) et écris la dérivée locale sur chaque arête.
2. Calcule $\frac{dz}{dx}$ par la somme sur les chemins, puis sa valeur en $x = 1$.
3. Vérifie en exprimant d'abord $z$ en fonction de $x$ seul, puis en dérivant.

**Partie B.** Un modèle $\hat{y} = w\,x + b$ est évalué sur **deux** exemples, $(x_1, y_1) = (1, 3)$ et $(x_2, y_2) = (2, 4)$, avec la loss $L = (\hat{y}_1 - y_1)^2 + (\hat{y}_2 - y_2)^2$.
1. Dessine le graphe : $w$ et $b$ influencent $L$ par **deux** chemins chacun, à travers $\hat{y}_1$ et $\hat{y}_2$.
2. Écris $\frac{\partial L}{\partial w}$ et $\frac{\partial L}{\partial b}$ comme des sommes sur les chemins.
3. Calcule-les en $w = 1$, $b = 0$.
4. Fais un pas de descente de gradient avec $\eta = 0{,}1$ : quels sont les nouveaux $w$ et $b$, et la loss a-t-elle baissé ?
5. Pourquoi cette méthode, appliquée à un réseau de millions de poids, s'appelle-t-elle la **rétro**propagation ? (une phrase, en pensant au sens dans lequel on parcourt le graphe)

<a id="reflexion"></a>

## 🧮 🗣️ 🛠️ Réflexion et outils

### Ex 0B.30 — Fermi : combien de multiplications dans un produit matriciel ? 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer le coût de produits matriciels et l'effet de l'ordre des calculs.
**Prérequis :** Ex 0B.20 · fiche §101.4.3, §101.1.1 · **Fil rouge :** MNIST · **Parcours :** M

Ordres de grandeur seulement (une puissance de 10 suffit). Rappel : un produit $(m, n) \times (n, p)$ coûte $m\,n\,p$ multiplications.
1. Une couche dense reçoit tout MNIST d'un coup : $\mathbf{X}$ de forme $(60\,000, 784)$ et $\mathbf{W}$ de forme $(784, 128)$. Combien de multiplications pour $\mathbf{X}\mathbf{W}$ ?
2. Une boucle Python fait environ $10^7$ multiplications par seconde ; une bibliothèque optimisée sur un processeur récent, environ $10^{10}$ à $10^{11}$ ; un GPU, plus de $10^{13}$. Combien de temps prend le calcul de la question 1 dans chaque cas ?
3. $\mathbf{A}$, $\mathbf{B}$ et $\mathbf{C}$ sont des matrices $1000 \times 1000$ et $\mathbf{v}$ un vecteur de dimension 1000. Combien de multiplications pour calculer $((\mathbf{A}\mathbf{B})\mathbf{C})\mathbf{v}$ (de gauche à droite) ? Et $\mathbf{A}(\mathbf{B}(\mathbf{C}\mathbf{v}))$ (de droite à gauche) ?
4. Le résultat est-il le même ? Lequel choisir, et combien de fois plus rapide est-il, environ ?
5. Le mode « inverse » de la rétropropagation (ch. 18) profite exactement de cette idée : laquelle ?

### Ex 0B.31 — Le gradient expliqué à un randonneur dans le brouillard 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer simplement le gradient et la descente de gradient, sans formule.
**Prérequis :** Ex 0B.25 · fiche §101.6.3 · **Parcours :** R

Un ami randonneur est perdu dans le brouillard, en montagne ; il veut rejoindre la vallée et ne voit qu'à un mètre autour de lui. Explique-lui **en cinq lignes au plus**, sans formule, ce qu'est le gradient et comment « descendre le gradient » l'aide. Ta réponse doit parler de : la pente sous ses pieds, la direction de plus forte montée, la taille de ses pas (le learning rate), et un risque (par exemple une cuvette qui n'est pas la vallée, ou des pas trop grands).

### Ex 0B.32 — Écrire des maths en LaTeX dans Markdown 🛠️ ★ ⏱️ 10 min
**Objectif :** écrire des formules lisibles en LaTeX dans un fichier Markdown ou une cellule de notebook.
**Prérequis :** fiche §101.1.3 · **Parcours :** R, M, C

**Fais-le en premier** : tu t'en serviras pour toutes tes réponses papier.

Entre deux `$`, le texte est une formule en ligne ; entre deux `$$`, une formule centrée sur sa ligne. Quelques commandes :

| Tu écris | Tu obtiens |
|---|---|
| `x^2`, `x_i`, `x_{i+1}`, `e^{-x}` | $x^2$, $x_i$, $x_{i+1}$, $e^{-x}$ |
| `\frac{a}{b}`, `\sqrt{x}` | $\frac{a}{b}$, $\sqrt{x}$ |
| `\sum_{i=1}^{n} x_i`, `\prod_{k=1}^{n} k` | $\sum_{i=1}^{n} x_i$, $\prod_{k=1}^{n} k$ |
| `\mathbf{x}`, `\mathbf{W}`, `\hat{y}`, `\bar{x}` | $\mathbf{x}$, $\mathbf{W}$, $\hat{y}$, $\bar{x}$ |
| `\sigma`, `\eta`, `\theta`, `\nabla f`, `\partial` | $\sigma$, $\eta$, $\theta$, $\nabla f$, $\partial$ |
| `\times`, `\cdot`, `\leq`, `\neq`, `\approx`, `\to` | $\times$, $\cdot$, $\leq$, $\neq$, $\approx$, $\to$ |
| `\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}` | une matrice $2 \times 2$ |

Dans ta copie de `06_mes_reponses.md` (section 0B.32), écris en LaTeX :
1. la moyenne $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ ;
2. la somme pondérée d'un neurone, avec des vecteurs en gras : $z = \mathbf{w} \cdot \mathbf{x} + b$ ;
3. la sigmoïde ;
4. un pas de descente de gradient $\mathbf{x} \leftarrow \mathbf{x} - \eta\,\nabla f(\mathbf{x})$ ;
5. la matrice $\mathbf{M}$ de 0B.21.

Vérifie le rendu : aperçu Markdown de VS Code, cellule Markdown de Jupyter ou de Colab, ou page GitHub de ton dépôt une fois le fichier poussé. ⚠️ Dans un tableau Markdown, remplace `|x|` par `\lvert x \rvert` : la barre verticale y sépare les colonnes.

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 0B.E1 — Qu'est-ce qu'un gradient, et à quoi sert-il pour entraîner un modèle ? 💼 ★★ ⏱️ 10 min
*Fiche §101.6.3 · prérequis 0B.25 · parcours R*

« Pouvez-vous m'expliquer ce qu'est un gradient et comment il sert à entraîner un réseau de neurones ? »

### 0B.E2 — Produit scalaire et similarité cosinus : à quoi servent-ils en ML ? 💼 ★★ ⏱️ 10 min
*Fiche §101.3.3 · prérequis 0B.41 (notebook) · parcours R*

« Dans un système de recherche par embeddings, pourquoi utilise-t-on souvent la similarité cosinus plutôt que la distance euclidienne ? »

### 0B.E3 — Pourquoi manipuler des log-probabilités plutôt que des probabilités ? 💼 ★★ ⏱️ 10 min
*Fiche §101.2.4, §101.7.2 · prérequis 0B.15, 0B.26 · parcours R*

« Pourquoi les modèles de langage et les classifieurs travaillent-ils avec des log-probabilités ? »

### 0B.E4 — La règle de la chaîne, et pourquoi la rétropropagation en dépend 💼 ★★ ⏱️ 10 min
*Fiche §101.5.3, §101.6.4 · prérequis 0B.29 · parcours R*

« Quel est le lien entre la règle de la chaîne et la rétropropagation ? »

### 0B.E5 — Espérance, moyenne d'un échantillon, variance : quelles différences ? 💼 ★★ ⏱️ 10 min
*Fiche §101.7.3 à §101.7.5 · prérequis 0B.53 (notebook) · parcours R*

« Quelle différence faites-vous entre l'espérance d'une variable aléatoire et la moyenne d'un échantillon ? Et que mesure la variance ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch00b_maths/03_notebook.ipynb`). La partie 0 du notebook vérifie tes réponses aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 0B.33 | Calculer avec Python : puissances, arrondis, `abs`, signe et `C(n, k)` | 📦 | ★ | 10 |
| 0B.34 | 0,99 puissance 1000 : presque 1 ou presque 0 ? | 🔮 | ★ | 10 |
| 0B.35 | Σ, Π et moyennes en code : `sum`, `math.prod`, `np.average`, moyenne mobile | 📦 | ★★ | 15 |
| 0B.36 | Galerie des fonctions usuelles : de l'affine au cosinus | 📦 | ★★ | 20 |
| 0B.37 | `exp` et `log` en NumPy : `-inf`, `nan` et dépassements | 🐛 | ★★ | 15 |
| 0B.38 | `linalg_basics` (1) : additionner, soustraire, multiplier des vecteurs | 🔨 | ★★ | 15 |
| 0B.39 | `linalg_basics` (2) : produit scalaire, norme, distance, cosinus | 🔨 | ★★ | 20 |
| 0B.40 | Tes fonctions contre NumPy : mêmes résultats, autre vitesse | 📦 | ★★ | 15 |
| 0B.41 | Distance ou similarité cosinus : l'effet de la longueur | 🔬 | ★★ | 20 |
| 0B.42 | `linalg_basics` (3) : forme, transposée, identité, matrice × vecteur | 🔨 | ★★ | 20 |
| 0B.43 | `linalg_basics` (4) : `matmul` et vérification des formes | 🔨 | ★★ | 25 |
| 0B.44 | AB = BA ? (AB)ᵀ = BᵀAᵀ ? Prédire, puis tester | 🔮 | ★★ | 15 |
| 0B.45 | Le produit qui n'en est pas un : `*`, `@`, `(n,)` et `(n, 1)` | 🐛 | ★★ | 20 |
| 0B.46 | Inverse et systèmes : `np.linalg.inv` et `np.linalg.solve` | 📦 | ★★ | 15 |
| 0B.47 | Pentes numériques : vérifier tes dérivées à la main | 🔨 | ★★ | 20 |
| 0B.48 | Lire les variations : f, f′ et les points où f′ s'annule | 📈 | ★★ | 15 |
| 0B.49 | Carte de lignes de niveau et flèches du gradient | 📈 | ★★ | 20 |
| 0B.50 | Contre le gradient, avec lui ou le long d'une ligne de niveau : où va f ? | 🔮 | ★★ | 15 |
| 0B.51 | Dérivées partielles numériques et somme sur les chemins | 🔨 | ★★ | 25 |
| 0B.52 | Simuler des dés : fréquences, indépendance, loi des grands nombres | 🔬 | ★★ | 25 |
| 0B.53 | Espérance et variance : le calcul exact contre la simulation | 🔨 | ★★ | 20 |
| 0B.54 | L'ordre des produits : calculer A·B·C·v des dizaines de fois plus vite | 🏆 | ★★★ | 40 |

Indices : `04_indices.md` (section « Notebook ») ; solutions commentées : `05_solutions.md` et `05_solutions.ipynb`.
