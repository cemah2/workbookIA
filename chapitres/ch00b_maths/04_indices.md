# 0B · Maths du lycée au ML — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🧮 🗣️ 🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à D](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 0B.Q1 — Puissances, racines et notation scientifique : vrai ou faux

<details><summary>Indice 1</summary>

Relis les règles de 101.1.1 : ce qui se multiplie, ce qui s'additionne. Pour chaque affirmation qui te semble fausse, cherche un contre-exemple avec de petits nombres.

</details>
<details><summary>Indice 2</summary>

1 : quand on multiplie deux puissances de même base, que fait-on des exposants ? 3 : essaie $a = 9$ et $b = 16$. 4 : $2^{10} = 1024 \approx 10^3$.

</details>
<details><summary>Indice 3</summary>

Trois affirmations sont vraies : la 2, la 4 et la 5. Trouve pourquoi la 1 et la 3 sont fausses.

</details>

### 0B.Q2 — |x|, ⌊x⌋, ⌈x⌉ et sign(x) : pièges de signe

<details><summary>Indice 1</summary>

Place chaque nombre sur une droite graduée : ⌊x⌋ est l'entier juste **à gauche** (ou x lui-même s'il est entier), ⌈x⌉ l'entier juste **à droite**.

</details>
<details><summary>Indice 2</summary>

1 : une valeur absolue est une distance. 4 : que se passe-t-il quand on divise les deux membres d'une inégalité par un nombre **négatif** ? 5 : $|x - 2|$ est la distance entre $x$ et 2.

</details>
<details><summary>Indice 3</summary>

Trois affirmations sont vraies : la 3, la 4 et la 5. Pour la 2 : −1,5 est entre −2 et −1 ; lequel est à gauche ?

</details>

### 0B.Q3 — Lire une formule avec Σ, Π et une moyenne pondérée

<details><summary>Indice 1</summary>

Une somme $\sum_{i=m}^{n}$ va de $m$ à $n$ **inclus**. Un $\Pi$ se lit comme un $\Sigma$, avec des multiplications.

</details>
<details><summary>Indice 2</summary>

1 : écris les indices 3, 4, 5, … et compte. 3 : un facteur commun à tous les termes peut sortir de la somme. 4 : remplace chaque $w_i$ par 1 ; que vaut alors $\sum_i w_i$ ?

</details>
<details><summary>Indice 3</summary>

1 : $n - m + 1$ termes. 2 : $1 \times 2 \times 3$. 4 : la somme des poids devient $n$, le nombre de valeurs. 5 : fiche §101.1.4, le paragraphe sur la somme pondérée.

</details>

### 0B.Q4 — Suite géométrique : elle fond ou elle explose ?

<details><summary>Indice 1</summary>

Ce qui compte, c'est la **valeur absolue** de $q$ (plus petite ou plus grande que 1), puis son **signe** (qui fait alterner les signes des termes).

</details>
<details><summary>Indice 2</summary>

Calcule mentalement les premiers termes : $(-0{,}5)^k$ donne $1, -0{,}5, 0{,}25, -0{,}125, \dots$ ; $(-1)^k$ donne $1, -1, 1, -1, \dots$

</details>
<details><summary>Indice 3</summary>

$|q| < 1$ : ça fond (en alternant les signes si $q < 0$). $|q| > 1$ : ça explose (en alternant si $q < 0$). $|q| = 1$ : la valeur absolue reste égale à 1 ; reste à voir si le signe change.

</details>

### 0B.Q5 — Ensembles et dénombrement : le bon réflexe

<details><summary>Indice 1</summary>

« Et » (communs aux deux) se traduit par une intersection, « ou » par une union. Pour compter, demande-toi si l'**ordre** compte.

</details>
<details><summary>Indice 2</summary>

2 : dans la formule $|A \cup B| = |A| + |B| - |A \cap B|$, quand le dernier terme est-il nul ? 3 : 4 choix pour la première place, puis… 4 : une paire ne dépend pas de l'ordre.

</details>
<details><summary>Indice 3</summary>

3 : $4!$. 4 : $\binom{5}{2} = \frac{5 \times 4}{2}$. 5 : relis la définition de $\binom{n}{k}$ en 101.1.7 (des **sous-ensembles**).

</details>

### 0B.Q6 — Reconnaître l'allure d'une courbe

<details><summary>Indice 1</summary>

Regarde la figure des fonctions usuelles de la fiche (101.2) : note pour chaque courbe son ensemble de valeurs et un point remarquable.

</details>
<details><summary>Indice 2</summary>

Deux courbes en « S » : l'une va de 0 à 1, l'autre de −1 à 1. Une seule courbe n'existe que pour $x > 0$. Une parabole qui coupe l'axe en $\pm 2$ s'annule en $x = \pm 2$.

</details>
<details><summary>Indice 3</summary>

1 : la sigmoïde ($\sigma(0) = \frac{1}{2}$). 3 : $\ln 1 = 0$. 4 : $x^2 - 4 = (x - 2)(x + 2)$. Deux fonctions de la liste ne servent pas.

</details>

### 0B.Q7 — Règles des logarithmes et des exponentielles : vrai ou faux

<details><summary>Indice 1</summary>

L'exponentielle transforme une somme (dans l'exposant) en produit ; le logarithme transforme un produit en somme. Pour chaque affirmation qui te semble fausse, cherche un contre-exemple avec des nombres simples.

</details>
<details><summary>Indice 2</summary>

1 : teste avec $a = b = 0$ : $e^0 = 1$ alors que $e^0 + e^0 = 2$. 3 : teste avec $a = b = 1$. 4 : dans NumPy, `np.log` est quel logarithme ? 5 : calcule d'abord $f(2)$, puis applique $g$.

</details>
<details><summary>Indice 3</summary>

Deux affirmations sont vraies : la 2 et la 5. Pour la 4 : `np.log` est $\ln$ ; $\log_{10} 100 = 2$ s'obtient avec `np.log10`.

</details>

### 0B.Q8 — Vecteurs : norme, produit scalaire, cosinus, Hadamard

<details><summary>Indice 1</summary>

Le produit scalaire additionne les produits des composantes ; le produit de Hadamard les garde **séparés**.

</details>
<details><summary>Indice 2</summary>

2 : $\sqrt{6^2 + 8^2}$. 4 : $\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}$ : que deviennent le numérateur et le dénominateur si $\mathbf{a}$ est multiplié par 10 ?

</details>
<details><summary>Indice 3</summary>

2 : $\sqrt{100}$. 3 : le cosinus de leur angle est nul : l'angle est droit. 5 : l'angle entre un vecteur et lui-même est nul.

</details>

### 0B.Q9 — Formes compatibles : ce produit existe-t-il ?

<details><summary>Indice 1</summary>

$(m, n) \times (n, p) \to (m, p)$ : les deux dimensions **intérieures** doivent être égales, et le résultat garde les deux dimensions **extérieures**.

</details>
<details><summary>Indice 2</summary>

Écris les formes côte à côte : $(3, 4)(4, 2)$, $(4, 2)(3, 4)$… La transposée échange les deux nombres de la forme : $\mathbf{A}^\top$ a la forme $(4, 3)$. Un vecteur de dimension 4 se comporte comme une forme $(4, 1)$.

</details>
<details><summary>Indice 3</summary>

Un seul des cinq produits n'existe pas. Le 5 donne la même forme que $(\mathbf{A}\mathbf{B})^\top$ : ce n'est pas un hasard (0B.44).

</details>

### 0B.Q10 — Dérivée : pente, signe, extremum

<details><summary>Indice 1</summary>

$f'(a)$ est une **pente** ; son signe dit si la courbe monte ou descend.

</details>
<details><summary>Indice 2</summary>

3 : $(x^n)' = n\,x^{n-1}$. 4 : cherche une fonction dont la dérivée s'annule en 0 mais qui monte avant **et** après 0. 5 : règle de la chaîne, avec $u = 2x$.

</details>
<details><summary>Indice 3</summary>

4 : regarde $f(x) = x^3$ en 0. 5 : $(e^u)' = u'\,e^u$ et $u' = 2$.

</details>

### 0B.Q11 — Gradient et lignes de niveau : vrai ou faux

<details><summary>Indice 1</summary>

Relis les trois propriétés du gradient en 101.6.3.

</details>
<details><summary>Indice 2</summary>

1 : combien de dérivées partielles a une fonction de deux variables ? 3 : le gradient indique la **montée** la plus rapide ; pour faire baisser la loss, où faut-il aller ?

</details>
<details><summary>Indice 3</summary>

Trois affirmations sont vraies : la 2, la 4 et la 5.

</details>

### 0B.Q12 — Probabilités : indépendance, espérance, variance

<details><summary>Indice 1</summary>

Indépendance : $P(A \cap B) = P(A)\,P(B)$. Espérance : moyenne pondérée par les probabilités. Variance : $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$.

</details>
<details><summary>Indice 2</summary>

2 : si $A$ et $B$ sont incompatibles, que vaut $P(A \cap B)$ ? Et $P(A)\,P(B)$ ? 3 : $\frac{1 + 2 + \dots + 6}{6}$. 4 : le $+1$ ne change pas la dispersion.

</details>
<details><summary>Indice 3</summary>

2 : $0 \neq P(A)\,P(B)$ dès que les deux probabilités sont non nulles. 4 : $3^2 \times 4$. 5 : fiche §101.7.5, première phrase.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 0B.R1 — 0A : une somme en Python, trois façons (boucle, sum, compréhension)

<details><summary>Indice 1</summary>

Une somme $\Sigma$ se traduit par une boucle qui accumule dans une variable initialisée à 0 ; un produit $\Pi$, dans une variable initialisée à 1.

</details>
<details><summary>Indice 2</summary>

`total = 0`, puis `for i in range(...)`, puis `total += ...`. En une ligne : `sum(expression for i in range(...))`. `math.prod` prend un itérable, comme `sum`.

</details>
<details><summary>Indice 3</summary>

`range(a, b)` s'arrête **avant** `b` : pour aller de 1 à 5 inclus, il faut `range(1, 6)`. Tu dois trouver 55, puis 120.

</details>

### 0B.R2 — 0A : shape, ndim et axis d'un array

<details><summary>Indice 1</summary>

Une ligne par manchot, une colonne par mesure : la forme est `(nombre de lignes, nombre de colonnes)`.

</details>
<details><summary>Indice 2</summary>

`X[:, 2]` : toutes les lignes, la colonne d'indice 2 (un indice entier retire un axe). `axis=0` fait **disparaître** l'axe 0, celui des lignes.

</details>
<details><summary>Indice 3</summary>

`X.shape == (333, 4)`. Une moyenne par colonne donne 4 nombres. `.T` échange les deux nombres de la forme.

</details>

### 0B.R3 — 0A : une fonction qui prend une fonction (lambda, Callable)

<details><summary>Indice 1</summary>

`compose` doit **renvoyer une fonction**, pas un nombre : une `lambda` ou une fonction définie à l'intérieur (une fermeture, 0A.42).

</details>
<details><summary>Indice 2</summary>

`def compose(g, f): return lambda x: ...`. Pour `compose(g, f)(5)`, on applique d'abord `f`, puis `g`.

</details>
<details><summary>Indice 3</summary>

$g(f(5)) = g(10)$ et $f(g(5)) = f(8)$. L'annotation se construit avec `Callable[[type des arguments], type du résultat]`, importé de `collections.abc`.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 0B.1 — Puissances, racines et notation scientifique sans calculatrice ✏️

<details><summary>Indice 1</summary>

Trois règles suffisent : $a^m \times a^n = a^{m+n}$, $(a^m)^n = a^{mn}$, $a^{-n} = \frac{1}{a^n}$. Et $a^{1/n}$ est la racine $n$-ième de $a$.

</details>
<details><summary>Indice 2</summary>

b : simplifie d'abord le numérateur en une seule puissance de 10. d : $\sqrt{ab} = \sqrt{a}\,\sqrt{b}$. e : $8^{2/3} = (2^3)^{2/3}$. f : multiplie les nombres entre eux, puis les puissances de 10 entre elles.

</details>
<details><summary>Indice 3</summary>

c : $\frac{1}{25}$. f : $6{,}4 \times 10^{2}$. g : pour passer de 5,6 à 0,000 56, la virgule recule de combien de rangs ? h : $2^{30} = (2^{10})^3$ et $2^{10} \approx 10^3$.

</details>

### Ex 0B.2 — Valeur absolue, partie entière et signe : tableau de valeurs ✏️

<details><summary>Indice 1</summary>

$|x|$ est la distance de $x$ à 0. $\lfloor x \rfloor$ est l'entier juste en dessous (vers la gauche de la droite graduée), $\lceil x \rceil$ l'entier juste au-dessus. $\mathrm{sign}(x)$ vaut −1, 0 ou 1.

</details>
<details><summary>Indice 2</summary>

c et d : dessine −4, −3,7 et −3 sur une droite graduée. g et h : $|x - 4| \leq 1{,}5$ veut dire que $x$ est à une distance au plus 1,5 de 4. i : l'inégalité est **stricte**.

</details>
<details><summary>Indice 3</summary>

a : $7 + 2$. c : −4 (en dessous de −3,7) ; d : −3. f : $-1 + 0 + 1$. g et h : $4 - 1{,}5$ et $4 + 1{,}5$. i : liste les entiers de −2 à 2.

</details>

### Ex 0B.3 — Lire et calculer des Σ et des Π ✏️

<details><summary>Indice 1</summary>

Écris chaque somme **en entier**, terme par terme, avant de calculer : par exemple $\sum_{i=1}^{3} (i + 2) = 3 + 4 + 5$.

</details>
<details><summary>Indice 2</summary>

c : le premier terme est $3^0$. f : $\sum_i (3x_i + 1) = 3\sum_i x_i + \sum_i 1$, et la seconde somme compte 4 termes. g : on ajoute 100 fois le même nombre. h : trois termes seulement, pour $i = 2, 3, 4$.

</details>
<details><summary>Indice 3</summary>

c : $1 + 3 + 9 + 27$. d : $2 \times 3 \times 4 \times 5$. e : $4 + 1 + 16 + 9$. f : $\sum_i x_i = 8$. h : $2 \times (-1) + 3 \times 4 + 4 \times 3$.

</details>

### Ex 0B.4 — Moyenne pondérée, somme pondérée et moyenne mobile à la main ✏️

<details><summary>Indice 1</summary>

Moyenne pondérée : $\frac{\sum_i w_i x_i}{\sum_i w_i}$. Somme pondérée d'un neurone : $\sum_i w_i x_i + b$, **sans** division. Moyenne mobile d'ordre 3 : la moyenne des 3 dernières valeurs.

</details>
<details><summary>Indice 2</summary>

b : la somme des coefficients vaut 6. d : $m_3$ est la moyenne de $x_1, x_2, x_3$ ; $m_4$ celle de $x_2, x_3, x_4$, etc. e : la première moyenne mobile existe à $t = k$, la dernière à $t = n$.

</details>
<details><summary>Indice 3</summary>

b : $\frac{14 \times 3 + 8 \times 1 + 11 \times 2}{6}$. c : $2 - 2 + 3 - 1$. d : $\frac{2 + 4 + 9}{3}$, $\frac{4 + 9 + 1}{3}$, $\frac{9 + 1 + 5}{3}$, $\frac{1 + 5 + 6}{3}$. e : $n - k + 1$.

</details>

### Ex 0B.5 — Ensembles : union, intersection, complémentaire et cardinal ✏️

<details><summary>Indice 1</summary>

Commence par **lister** $A$, $B$ et $C$ en entier : avec 12 éléments, c'est rapide et ça évite toutes les erreurs.

</details>
<details><summary>Indice 2</summary>

$A = \{2, 4, 6, 8, 10, 12\}$, $B = \{3, 6, 9, 12\}$. b : $|A \cup B| = |A| + |B| - |A \cap B|$. d : « ni l'un ni l'autre » est le complémentaire de « l'un ou l'autre ».

</details>
<details><summary>Indice 3</summary>

a : $A \cap B = \{6, 12\}$. d : $12 - |A \cup B|$. e : parmi 1, 2 et 3, lesquels sont pairs ? f : $A \cup B$, plus les éléments de $C$ qui n'y sont pas encore.

</details>

### Ex 0B.6 — Droites et paraboles : pente, ordonnée à l'origine, racines, sommet ✏️

<details><summary>Indice 1</summary>

Pente d'une droite : $\frac{\Delta y}{\Delta x}$, les deux points pris dans le même ordre en haut et en bas. Parabole $ax^2 + bx + c$ : $\Delta = b^2 - 4ac$, racines $\frac{-b \pm \sqrt{\Delta}}{2a}$, sommet en $-\frac{b}{2a}$.

</details>
<details><summary>Indice 2</summary>

a : $\frac{-3 - 5}{3 - (-1)}$. b : écris $y = a\,x + b$ (101.2.1) avec la pente trouvée en a, puis remplace par le point $(-1, 5)$. c : résous $a\,x + b = 0$. d : pour la parabole, les lettres changent de rôle : ici $a = 2$, $b = -8$, $c = 6$.

</details>
<details><summary>Indice 3</summary>

a : $\frac{-8}{4}$. b : $5 = -2 \times (-1) + b$. d : $64 - 48$. e et f : $\frac{8 \pm 4}{4}$. g : $\frac{8}{4}$. h : calcule $f$ au sommet.

</details>

### Ex 0B.7 — Cosinus : cercle, période et planning en cosinus ✏️

<details><summary>Indice 1</summary>

$\pi$ rad = 180°. Sur le cercle trigonométrique, le cosinus est l'**abscisse** du point. Le cosinus est pair ($\cos(-x) = \cos x$) et $2\pi$-périodique.

</details>
<details><summary>Indice 2</summary>

a : $60° = \frac{\pi}{3}$. c : $\frac{2\pi}{3}$ est le symétrique de $\frac{\pi}{3}$ par rapport à l'axe vertical. e : $4\pi$, c'est deux tours complets. f à h : remplace $t$ et $T$, puis simplifie $\frac{\pi t}{T}$.

</details>
<details><summary>Indice 3</summary>

a : $\frac{3{,}1416}{3}$. c : $-\cos\frac{\pi}{3}$ et $\cos\frac{\pi}{3} = \frac{1}{2}$. f : $\cos\frac{\pi}{2} = 0$. g : $\frac{1}{2}(1 + 0{,}707)$. h : $\cos\frac{3\pi}{4} = -0{,}707$, puis multiplie par 0,01.

</details>

### Ex 0B.8 — Vecteurs : somme, multiple, norme et distance entre deux manchots ✏️

<details><summary>Indice 1</summary>

On additionne des vecteurs **composante par composante**. Norme L2 : $\sqrt{\sum_i u_i^2}$ ; L1 : $\sum_i |u_i|$ ; L∞ : $\max_i |u_i|$. Distance : la norme de la différence.

</details>
<details><summary>Indice 2</summary>

f : $\mathbf{p}_1 - \mathbf{p}_2 = (-8, -6)$. g : même calcul avec $(-8, -0{,}6)$. h : divise chaque composante de $\mathbf{u}$ par sa norme.

</details>
<details><summary>Indice 3</summary>

c : $\sqrt{9 + 16}$. d : $3 + 4$. f : $\sqrt{64 + 36}$. g : $\sqrt{64 + 0{,}36}$ : la nageoire ne pèse presque plus rien dans la distance. h : $\left(\frac{3}{5}, \frac{-4}{5}\right)$.

</details>

### Ex 0B.9 — Transposée et produit matrice-vecteur : deux lectures ✏️

<details><summary>Indice 1</summary>

La transposée échange lignes et colonnes : $(\mathbf{A}^\top)_{ij} = \mathbf{A}_{ji}$. Le produit $\mathbf{A}\mathbf{v}$ exige que le nombre de **colonnes** de $\mathbf{A}$ soit égal à la dimension de $\mathbf{v}$.

</details>
<details><summary>Indice 2</summary>

c, lecture par lignes : chaque composante du résultat est le produit scalaire d'une ligne de $\mathbf{A}$ avec $\mathbf{v}$. Lecture par colonnes : $1 \times (\text{colonne } 1) + 2 \times (\text{colonne } 2) + (-1) \times (\text{colonne } 3)$. f : chaque ligne de $\mathbf{X}$ est un exemple.

</details>
<details><summary>Indice 3</summary>

b : $(\mathbf{A}^\top)_{31} = \mathbf{A}_{13}$. c : $(2 + 0 - 1,\; -1 + 6 - 2)$. d : $\mathbf{A}^\top = \begin{pmatrix} 2 & -1 \\ 0 & 3 \\ 1 & 2 \end{pmatrix}$. e : $\mathbf{A}$ a 3 colonnes. f : $\mathbf{X}\mathbf{w} = (4{,}5 ; 1{,}5 ; -2)$, puis ajoute $b$.

</details>

### Ex 0B.10 — Taux d'accroissement : de la sécante à la tangente ✏️

<details><summary>Indice 1</summary>

Le taux d'accroissement entre $a$ et $a + h$ est $\frac{f(a + h) - f(a)}{h}$ : la pente de la corde (la sécante) entre les deux points.

</details>
<details><summary>Indice 2</summary>

a : $f(3) = 12$. b : $f(2{,}1) = 4{,}41 + 2{,}1$. d : développe $(2 + h)^2 + (2 + h)$, retire 6, puis divise par $h$. e : développe $y = 6 + 5(x - 2)$.

</details>
<details><summary>Indice 3</summary>

d : le numérateur vaut $5h + h^2$, donc le taux vaut $5 + h$. e : $y = 5x - 4$. f : $6 + 5 \times 0{,}05$ ; la vraie valeur est $f(2{,}05) = 4{,}2025 + 2{,}05$.

</details>

### Ex 0B.11 — Probabilités : issues, complémentaire, union ✏️

<details><summary>Indice 1</summary>

Dessine le tableau $6 \times 6$ des 36 issues (dé 1 en lignes, dé 2 en colonnes) et entoure les issues favorables. $P = \frac{\text{favorables}}{36}$.

</details>
<details><summary>Indice 2</summary>

a : les couples de somme 8 sont sur une diagonale du tableau. c : « au moins un 1 » est le complémentaire de « aucun 1 », qui compte $5 \times 5$ issues. d : $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. f : 13 cœurs, 12 figures, dont 3 figures de cœur.

</details>
<details><summary>Indice 3</summary>

a : $\frac{5}{36}$. c : $1 - \frac{25}{36}$. d : $(4, 4)$ est à la fois un double et une somme 8 : $\frac{5 + 6 - 1}{36}$. e : sommes 10, 11 et 12 : $3 + 2 + 1$ issues. f : $\frac{13 + 12 - 3}{52}$.

</details>

### Ex 0B.12 — Dénombrer : choix successifs, factorielle et C(n, k) ✏️

<details><summary>Indice 1</summary>

Trois outils : choix successifs (on multiplie les nombres de choix), $n!$ pour ranger $n$ objets, $\binom{n}{k}$ pour choisir $k$ objets **sans ordre**. Demande-toi à chaque fois : l'ordre compte-t-il ? peut-on répéter ?

</details>
<details><summary>Indice 2</summary>

a : 10 choix pour chaque chiffre. b : 10 choix, puis un de moins à chaque fois. f : une paire de classes. g : chaque feature est dedans ou pas : 2 choix par feature. h : choisis d'abord le président, puis les deux autres membres, sans ordre.

</details>
<details><summary>Indice 3</summary>

b : $10 \times 9 \times 8 \times 7$. d : $\frac{6 \times 5}{2}$. e : $\frac{8 \times 7 \times 6}{3 \times 2 \times 1}$. g : $2^5$. h : $10 \times \binom{9}{2}$.

</details>

### Ex 0B.13 — Suites géométriques : ce qui fond, ce qui explose ✏️

<details><summary>Indice 1</summary>

$u_n = u_0\,q^n$ et $\sum_{k=0}^{n-1} q^k = \frac{1 - q^n}{1 - q}$ ; si $|q| < 1$, la somme infinie vaut $\frac{1}{1 - q}$.

</details>
<details><summary>Indice 2</summary>

a : 5 divisions par 2 à partir de 1000. d : de $k = 0$ à 9, il y a **10** termes ($n = 10$). f : $0{,}5^{10} = \frac{1}{1024}$. g : une puissance impaire d'un nombre négatif est négative.

</details>
<details><summary>Indice 3</summary>

a : $\frac{1000}{32}$. d : $\frac{1 - 0{,}5^{10}}{0{,}5} = 2\left(1 - \frac{1}{1024}\right)$. e : $\frac{1}{0{,}05}$. f : compare $0{,}5^9 = \frac{1}{512}$ et $0{,}5^{10} = \frac{1}{1024}$ avec $0{,}001$. g : $-(0{,}8^7)$.

</details>

### Ex 0B.14 — La somme géométrique démontrée pas à pas ∂

<details><summary>Indice 1</summary>

Écris $S_n$ et $q\,S_n$ l'une sous l'autre, en décalant la seconde d'un cran vers la droite : presque tous les termes se retrouvent en face d'un terme identique.

</details>
<details><summary>Indice 2</summary>

$q\,S_n = q + q^2 + \dots + q^{n}$. Dans $S_n - q\,S_n$, il ne reste que le premier terme de $S_n$ et le dernier de $q\,S_n$. Puis factorise $S_n$ à gauche.

</details>
<details><summary>Indice 3</summary>

$S_n(1 - q) = 1 - q^n$, et on peut diviser par $1 - q$ puisque $q \neq 1$. 3 : $1 + 2 + 4 + 8$. 4 : si $|q| < 1$, $q^n \to 0$ (101.1.5). 5 : $0{,}9 \times \frac{1}{1 - 0{,}1}$.

</details>

### Ex 0B.15 — Exponentielles et logarithmes : règles de calcul en bases 2, e et 10 ✏️

<details><summary>Indice 1</summary>

$\ln$ et $\exp$ se défont l'une l'autre : $\ln(e^a) = a$ et $e^{\ln b} = b$. $\log_b x$ est l'exposant qu'il faut donner à $b$ pour obtenir $x$.

</details>
<details><summary>Indice 2</summary>

c : $\ln 1 = 0$ et $32 = 2^5$. d : $0{,}001 = 10^{-3}$. e : $e^2 \times e^5 = e^{7}$. f et g : prends le logarithme des deux membres. h : $\ln(a^n) = n \ln a$.

</details>
<details><summary>Indice 3</summary>

f : $x = \ln 20$. g : $x = \log_2 1000 = \frac{\ln 1000}{\ln 2}$ ; comme $2^{10} = 1024$, il faut 10 bits. h : $100 \times \ln 0{,}01 = 100 \times (-4{,}605)$. i : $\frac{1}{\ln 2}$.

</details>

### Ex 0B.16 — Changer de base : log₂ x = ln x / ln 2, bits et nats ∂

<details><summary>Indice 1</summary>

L'idée clé : $\ln(a^y) = y \ln a$. Elle fait « descendre » l'exposant.

</details>
<details><summary>Indice 2</summary>

1 : $\ln(2^y) = \ln x$ donne $y \ln 2 = \ln x$. 2 : remplace 2 par $b$ ; pourquoi faut-il $b \neq 1$ ? 3 : écris les deux membres avec des $\ln$ grâce à la question 2. 4 : $\log_2 \frac{1}{p} = \frac{\ln(1/p)}{\ln 2}$.

</details>
<details><summary>Indice 3</summary>

2 : on divise par $\ln b$, qui est nul si $b = 1$. 3 : $\frac{\ln 10}{\ln 2} \times \frac{\ln x}{\ln 10}$. 5 : $\frac{1}{p} = 8 = 2^3$ ; en nats, $\ln 8 = 3 \ln 2$.

</details>

### Ex 0B.17 — Sigmoïde et tanh : valeurs, limites, symétries ✏️

<details><summary>Indice 1</summary>

$\sigma(x) = \frac{1}{1 + e^{-x}}$ et $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. Calcule d'abord la ou les exponentielles, puis la fraction.

</details>
<details><summary>Indice 2</summary>

b : $e^{-3} \approx 0{,}0498$. d : $e^{0{,}5} \approx 1{,}6487$ et $e^{-0{,}5} \approx 0{,}6065$. e : $\sigma(1) \approx 0{,}7311$. f : quand $x \to -\infty$, que devient $e^{-x}$ ? g : isole $e^{-x}$ dans $\frac{1}{1 + e^{-x}} = 0{,}9$.

</details>
<details><summary>Indice 3</summary>

c : $1 - \sigma(3)$. e : même valeur qu'en d, car $\tanh(x) = 2\sigma(2x) - 1$. g : $1 + e^{-x} = \frac{10}{9}$, donc $e^{-x} = \frac{1}{9}$ et $x = \ln 9$. h : utilise la symétrie de c.

</details>

### Ex 0B.18 — Produit scalaire, similarité cosinus et produit de Hadamard ✏️

<details><summary>Indice 1</summary>

$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ (un nombre) ; $\mathbf{a} \odot \mathbf{b} = (a_1 b_1, a_2 b_2, \dots)$ (un vecteur) ; $\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}$.

</details>
<details><summary>Indice 2</summary>

a : $2 + 0 - 2$. c : $\sqrt{1 + 4 + 4}$. e : compare $\mathbf{c}$ et $\mathbf{a}$ composante par composante. f : $\|\mathbf{d}\| = 1$. h : additionne les composantes de $\mathbf{a} \odot \mathbf{c}$.

</details>
<details><summary>Indice 3</summary>

d : $2 + 8 + 8$. e : $\mathbf{c} = 2\mathbf{a}$, même direction : $\frac{18}{3 \times 6}$. f : $\frac{1}{3 \times 1}$. h : la somme des composantes du produit de Hadamard, c'est le produit scalaire. i : $-\mathbf{a}$ pointe dans la direction opposée.

</details>

### Ex 0B.19 — Développer ‖a − b‖² avec le produit scalaire ∂

<details><summary>Indice 1</summary>

Le produit scalaire se développe comme un produit de nombres : $(\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b})$ se calcule comme $(a - b)(a - b)$, et $\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}$.

</details>
<details><summary>Indice 2</summary>

1 : $(\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b}) = \mathbf{a} \cdot \mathbf{a} - \mathbf{a} \cdot \mathbf{b} - \mathbf{b} \cdot \mathbf{a} + \mathbf{b} \cdot \mathbf{b}$. 2 : $\mathbf{a} - \mathbf{b} = (-1, 2, 3)$. 3 : pour des vecteurs unitaires, $\|\mathbf{a}\| = \|\mathbf{b}\| = 1$ et $\mathbf{a} \cdot \mathbf{b} = \cos(\mathbf{a}, \mathbf{b})$.

</details>
<details><summary>Indice 3</summary>

2 : $\|\mathbf{a} - \mathbf{b}\|^2 = 1 + 4 + 9$ ; à droite, $9 + 5 - 2 \times 0$. 3 : la distance diminue quand le cosinus augmente. 4 : un triangle rectangle, et ses trois côtés.

</details>

### Ex 0B.20 — Produit matriciel : calculer et vérifier les formes ✏️

<details><summary>Indice 1</summary>

$(m, n) \times (n, p) \to (m, p)$. L'élément $(i, j)$ du produit est le produit scalaire de la **ligne** $i$ de la matrice de gauche et de la **colonne** $j$ de la matrice de droite.

</details>
<details><summary>Indice 2</summary>

Formes : $\mathbf{A}$ est $(3, 2)$, $\mathbf{B}$ est $(2, 3)$, $\mathbf{C}$ est $(2, 2)$. c : $\mathbf{B}\mathbf{A}$ a 4 éléments ; le premier est (ligne 1 de $\mathbf{B}$) · (colonne 1 de $\mathbf{A}$) $= (2, 1, 0) \cdot (1, 0, 3)$. h : chaque élément de $\mathbf{A}\mathbf{B}$ coûte $n$ multiplications.

</details>
<details><summary>Indice 3</summary>

c : les autres éléments sont $(2, 1, 0) \cdot (2, -1, 1)$, $(1, -1, 4) \cdot (1, 0, 3)$ et $(1, -1, 4) \cdot (2, -1, 1)$. d : $(1, 2) \cdot (2, 1)$. e : $(3, 1) \cdot (1, -1)$. f : $(2, 2) \times (3, 2)$. g : 3 lignes, 2 colonnes. h : $m \times n \times p$.

</details>

### Ex 0B.21 — Identité, inverse 2 × 2 et système de deux équations ✏️

<details><summary>Indice 1</summary>

Pour $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ : $\det = ad - bc$, et, si $\det \neq 0$, l'inverse est $\frac{1}{\det}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$ (on échange $a$ et $d$, on change le signe de $b$ et $c$).

</details>
<details><summary>Indice 2</summary>

a : $3 \times 2 - 1 \times 4$. c : le système s'écrit $\mathbf{M}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 5 \\ 6 \end{pmatrix}$, donc $\begin{pmatrix} x \\ y \end{pmatrix} = \mathbf{M}^{-1}\begin{pmatrix} 5 \\ 6 \end{pmatrix}$. d et e : une matrice est inversible si et seulement si son déterminant est non nul.

</details>
<details><summary>Indice 3</summary>

b : $\frac{1}{2}\begin{pmatrix} 2 & -1 \\ -4 & 3 \end{pmatrix}$. c : $(1 \times 5 - 0{,}5 \times 6,\; -2 \times 5 + 1{,}5 \times 6)$, puis vérifie dans les deux équations. e : $6 - 2k = 0$. f : pour une matrice diagonale, on inverse chaque élément de la diagonale.

</details>

### Ex 0B.22 — Dériver avec les règles : somme, produit, quotient, exp, ln ✏️

<details><summary>Indice 1</summary>

Le tableau de 101.5.2 : $(x^n)' = n x^{n-1}$, $(\sqrt{x})' = \frac{1}{2\sqrt{x}}$, $\left(\frac{1}{x}\right)' = -\frac{1}{x^2}$, $(e^x)' = e^x$, $(\ln x)' = \frac{1}{x}$ ; $(uv)' = u'v + uv'$ ; $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$.

</details>
<details><summary>Indice 2</summary>

c, d : règle du produit, avec $u = x^2$ et $v = e^x$ (puis $u = x$ et $v = \ln x$). e, f : règle du quotient. Dérive d'abord, **puis** remplace $x$ par sa valeur.

</details>
<details><summary>Indice 3</summary>

a : $12x^2 - 2$. b : $\frac{1}{4} - \frac{1}{16}$. c : $(2x + x^2)e^x$, soit $3e$ en 1. d : $\ln x + 1$. e : $\frac{(x - 1) - (x + 1)}{(x - 1)^2} = \frac{-2}{(x - 1)^2}$. f : $\frac{e^x(x - 1)}{x^2}$. g : $20x^3$.

</details>

### Ex 0B.23 — Règle de la chaîne : décomposer, puis dériver ✏️

<details><summary>Indice 1</summary>

$(g \circ u)' = g'(u) \times u'$ : dérivée de l'étape **extérieure** (évaluée en $u$), multipliée par la dérivée de l'étape **intérieure**. Nomme toujours $u$ en premier.

</details>
<details><summary>Indice 2</summary>

a : $u = 3x - 1$ et $g(u) = u^4$. b : $u = 2x + 1$. c : $u = x^2 + 1$ et $g = \ln$. d : $u = 1 + 4x$ et $g = \sqrt{\ }$. e : $u = -\frac{x^2}{2}$. f : $u = x^2 + 1$ et $g(u) = \frac{1}{u}$. g : $u = 3w - 3$. h : $u = \ln x$ et $g(u) = u^2$.

</details>
<details><summary>Indice 3</summary>

a : $4u^3 \times 3$. b : $e^u \times 2$. c : $\frac{1}{u} \times 2x$. d : $\frac{1}{2\sqrt{u}} \times 4$. e : $e^u \times (-x)$. f : $-\frac{1}{u^2} \times 2x$. g : $2u \times 3$. h : $2u \times \frac{1}{x}$. Remplace ensuite par la valeur du point.

</details>

### Ex 0B.24 — La moyenne minimise la somme des carrés des écarts ∂

<details><summary>Indice 1</summary>

La variable est $a$ ; les $y_i$ sont des **constantes**. Chaque terme $(y_i - a)^2$ se dérive avec la règle de la chaîne, avec $u = y_i - a$.

</details>
<details><summary>Indice 2</summary>

$\frac{d}{da}(y_i - a)^2 = 2(y_i - a) \times (-1)$. Puis $S'(a) = 0 \iff \sum_i (y_i - a) = 0 \iff \sum_i y_i - n\,a = 0$ (la somme de $n$ fois $a$). 3 : écris $S'(a) = 2n(a - \bar{y})$.

</details>
<details><summary>Indice 3</summary>

3 : $S'(a) < 0$ pour $a < \bar{y}$ et $S'(a) > 0$ pour $a > \bar{y}$ ; ou : en développant, $S(a) = n\,a^2 - 2a\sum_i y_i + \sum_i y_i^2$, une parabole avec $n > 0$ devant $a^2$. 4 : $\bar{y} = 4$ ; $S(4) = 4 + 1 + 9$. 5 : $\frac{S(\bar{y})}{n}$ est exactement la formule de la variance (101.7.4).

</details>

### Ex 0B.25 — Lignes de niveau, dérivées partielles, gradient et un pas de descente ✏️

<details><summary>Indice 1</summary>

$\frac{\partial f}{\partial x}$ : dérive par rapport à $x$ en traitant $y$ comme une constante (et inversement). Le gradient est le vecteur des deux dérivées partielles ; un pas de descente : $\mathbf{p} \leftarrow \mathbf{p} - \eta\,\nabla f(\mathbf{p})$.

</details>
<details><summary>Indice 2</summary>

b : $\frac{\partial}{\partial x}(x - 1)^2 = 2(x - 1)$, et $2y^2$ est une constante. c : $\frac{\partial}{\partial y} 2y^2 = 4y$. f : $(3, 1) - 0{,}1 \times (4, 4)$. h : une somme de deux carrés est minimale quand les deux carrés sont nuls. i, j : pour $\frac{\partial g}{\partial y}$, $x^2$ est un coefficient constant.

</details>
<details><summary>Indice 3</summary>

a : $4 + 2$. e : $\sqrt{16 + 16} = 4\sqrt{2}$. f : $(2{,}6 ; 0{,}6)$. g : $1{,}6^2 + 2 \times 0{,}6^2$. h : $x = 1$, $y = 0$. i : $2xy$. j : $x^2 + 3y^2$. Dessin : $(x - 1)^2 + 2y^2 = 2$ est une ellipse de demi-axes $\sqrt{2}$ (horizontal) et 1 (vertical) ; pour 8, $2\sqrt{2}$ et 2.

</details>

### Ex 0B.26 — Indépendance : tester P(A ∩ B) = P(A) P(B) avec deux dés ✏️

<details><summary>Indice 1</summary>

Calcule les deux membres de $P(A \cap B) = P(A)\,P(B)$ séparément, **en fractions**, puis compare-les exactement.

</details>
<details><summary>Indice 2</summary>

a : 18 issues sur 36. c : liste les couples de somme 7 dont le premier dé est pair. f : même chose pour la somme 8. Garde les fractions : $\frac{3}{36} = \frac{1}{12}$ ; $\frac{1}{2} \times \frac{5}{36} = \frac{5}{72}$.

</details>
<details><summary>Indice 3</summary>

c : $(2, 5), (4, 3), (6, 1)$, et $\frac{1}{2} \times \frac{1}{6} = \frac{1}{12}$. f : $(2, 6), (4, 4), (6, 2)$. h : $\frac{6}{72} - \frac{5}{72}$.

</details>

### Ex 0B.27 — Espérance et variance d'une variable discrète ✏️

<details><summary>Indice 1</summary>

$\mathbb{E}[X] = \sum_k x_k p_k$ ; $\mathbb{E}[X^2] = \sum_k x_k^2 p_k$ ; $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ ; $\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b$ ; $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$.

</details>
<details><summary>Indice 2</summary>

a : $0 \times 0{,}4 + 1 \times 0{,}3 + 2 \times 0{,}2 + 5 \times 0{,}1$. b : même chose avec $0, 1, 4, 25$. g et h : chacune des 6 faces a la probabilité $\frac{1}{6}$ ; le nombre 3 est inscrit sur trois faces, donc $P(X = 3) = \frac{3}{6}$.

</details>
<details><summary>Indice 3</summary>

c : $3{,}6 - 1{,}2^2$. e : $3 \times 1{,}2 + 2$. f : $9 \times 2{,}16$. g : $\frac{1 + 1 + 2 + 3 + 3 + 3}{6} = \frac{13}{6}$. h : $\mathbb{E}[X^2] = \frac{33}{6}$, puis retire $\left(\frac{13}{6}\right)^2$.

</details>

### Ex 0B.28 — Variance : deux formules, et l'espérance est linéaire ∂

<details><summary>Indice 1</summary>

Tout repose sur deux règles des sommes : on peut couper une somme en deux, et sortir un facteur constant. Et n'oublie pas $\sum_k p_k = 1$.

</details>
<details><summary>Indice 2</summary>

1 : $\sum_k (a x_k + b) p_k = a \sum_k x_k p_k + b \sum_k p_k$. 2 : $(x_k - \mu)^2 = x_k^2 - 2\mu x_k + \mu^2$ ; coupe la somme en trois. 3 : $(aX + b) - (a\mu + b) = a(X - \mu)$.

</details>
<details><summary>Indice 3</summary>

2 : les trois morceaux valent $\mathbb{E}[X^2]$, $-2\mu \times \mu$ et $\mu^2 \times 1$. 3 : $\sum_k a^2 (x_k - \mu)^2 p_k$. 4 : la variable $3X + 2$ prend les valeurs $2, 5, 8, 17$ ; calcule son espérance et sa variance directement. 5 : une somme de termes $\geq 0$.

</details>

### Ex 0B.29 — Règle de la chaîne à deux variables : la somme sur les chemins ∂

<details><summary>Indice 1</summary>

Un terme par chemin ; le long d'un chemin, on **multiplie** les dérivées locales ; puis on **additionne** les chemins (101.6.4).

</details>
<details><summary>Indice 2</summary>

A : les arêtes portent $\frac{du}{dx} = 2$, $\frac{dv}{dx} = 2x$, $\frac{\partial z}{\partial u} = 2u + v$ et $\frac{\partial z}{\partial v} = u$. B : $\frac{\partial L}{\partial \hat{y}_1} = 2(\hat{y}_1 - y_1)$, $\frac{\partial \hat{y}_1}{\partial w} = x_1$ et $\frac{\partial \hat{y}_1}{\partial b} = 1$ ; de même pour l'exemple 2.

</details>
<details><summary>Indice 3</summary>

A : $\frac{dz}{dx} = (2u + v) \times 2 + u \times 2x$, avec $u = 2$ et $v = 1$ en $x = 1$ ; vérification : $z = 4x^2 + 2x^3$. B : en $w = 1$, $b = 0$ : $\hat{y}_1 = 1$ et $\hat{y}_2 = 2$, donc les écarts valent $-2$ et $-2$ ; $\frac{\partial L}{\partial w} = 2(-2)(1) + 2(-2)(2)$ et $\frac{\partial L}{\partial b} = 2(-2) + 2(-2)$. Le pas : $w - 0{,}1 \times \frac{\partial L}{\partial w}$.

</details>

<a id="reflexion"></a>

## 🧮 🗣️ 🛠️ Réflexion et outils

### Ex 0B.30 — Fermi : combien de multiplications dans un produit matriciel ? 🧮

<details><summary>Indice 1</summary>

Un produit $(m, n) \times (n, p)$ coûte $m\,n\,p$ multiplications. Arrondis chaque facteur à une puissance de 10 ou à un petit multiple (60 000 ≈ $6 \times 10^4$, 784 ≈ $8 \times 10^2$, 128 ≈ $10^2$).

</details>
<details><summary>Indice 2</summary>

2 : temps = nombre de multiplications / vitesse. 3 : $\mathbf{A}\mathbf{B}$ est un produit $(1000, 1000) \times (1000, 1000)$ ; $\mathbf{C}\mathbf{v}$ est un produit $(1000, 1000) \times (1000, 1)$. Qu'est-ce qui reste une matrice, qu'est-ce qui devient un vecteur ?

</details>
<details><summary>Indice 3</summary>

1 : environ $6 \times 10^9$. 3 : de gauche à droite, deux produits matrice-matrice ($10^9$ chacun) et un matrice-vecteur ($10^6$) ; de droite à gauche, trois produits matrice-vecteur. 4 : le produit matriciel est associatif. 5 : la rétropropagation part de la loss, un seul nombre ; ce qu'elle fait passer d'une couche à l'autre joue le rôle du vecteur $\mathbf{v}$ tout à droite.

</details>

### Ex 0B.31 — Le gradient expliqué à un randonneur dans le brouillard 🗣️

<details><summary>Indice 1</summary>

Pars de ce qu'il **peut** faire : sentir la pente sous ses pieds, dans toutes les directions autour de lui.

</details>
<details><summary>Indice 2</summary>

Le gradient, c'est « la direction où ça monte le plus, et à quel point ça monte ». Descendre le gradient, c'est faire un pas dans la direction **opposée**, puis recommencer. La taille du pas est un choix.

</details>
<details><summary>Indice 3</summary>

Cinq lignes : 1) il tâte le sol autour de lui ; 2) il repère la direction de plus forte montée ; 3) il fait un pas dans l'autre sens ; 4) des pas trop grands le font passer par-dessus la vallée, des pas trop petits le font avancer très lentement ; 5) il peut s'arrêter dans une cuvette qui n'est pas la vallée.

</details>

### Ex 0B.32 — Écrire des maths en LaTeX dans Markdown 🛠️

<details><summary>Indice 1</summary>

Tout ce dont tu as besoin est dans le tableau de l'énoncé. Commence par une formule simple, vérifie son rendu, puis complique.

</details>
<details><summary>Indice 2</summary>

Les accolades regroupent : `x_{i+1}` et non `x_i+1`, `e^{-x}` et non `e^-x`. Une fraction : `\frac{numérateur}{dénominateur}`. Dans une matrice, `&` sépare les colonnes et `\\` les lignes.

</details>
<details><summary>Indice 3</summary>

La sigmoïde : `$\sigma(x) = \frac{1}{1 + e^{-x}}$`. La flèche vers la gauche : `\leftarrow`. Une petite espace avant $\nabla$ : `\,`.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 0B.E1 — Qu'est-ce qu'un gradient, et à quoi sert-il pour entraîner un modèle ?

<details><summary>Indice 1</summary>

Trois temps : la définition (un vecteur de dérivées partielles), la propriété (la direction de plus forte montée), l'usage (la descente de gradient sur la loss).

</details>
<details><summary>Indice 2</summary>

Parle de la loss comme d'une fonction des **poids** ; du learning rate ; du fait que le gradient est calculé par la rétropropagation, en pratique sur des mini-batches.

</details>
<details><summary>Indice 3</summary>

Structure : « Pour une fonction de plusieurs variables, le gradient est… Il pointe vers… Pour entraîner un modèle, on… On répète… Le calcul efficace du gradient, c'est… »

</details>

### 0B.E2 — Produit scalaire et similarité cosinus : à quoi servent-ils en ML ?

<details><summary>Indice 1</summary>

La similarité cosinus ne regarde que l'**angle** entre deux vecteurs, pas leur longueur.

</details>
<details><summary>Indice 2</summary>

Dans un embedding, la direction porte le sens ; la longueur dépend souvent d'autre chose (longueur du texte, fréquence du mot). Et pour des vecteurs normalisés, les deux classements coïncident (0B.19).

</details>
<details><summary>Indice 3</summary>

Structure : définition du produit scalaire et du cosinus ; pourquoi l'angle compte plus que la longueur ; le lien $\|\mathbf{a} - \mathbf{b}\|^2 = 2 - 2\cos$ pour des vecteurs unitaires ; le côté pratique (un produit scalaire est très rapide à calculer, index de recherche vectorielle).

</details>

### 0B.E3 — Pourquoi manipuler des log-probabilités plutôt que des probabilités ?

<details><summary>Indice 1</summary>

Deux raisons : une **numérique** (que devient le produit de milliers de probabilités ?) et une **pratique** (que devient ce produit une fois passé au logarithme ?).

</details>
<details><summary>Indice 2</summary>

$0{,}01^{100} = 10^{-200}$, et un `float64` ne descend pas beaucoup plus bas que $10^{-308}$ (0B.15 h). $\ln$ transforme un produit en somme, et il est croissant : maximiser $\ln L$ revient à maximiser $L$.

</details>
<details><summary>Indice 3</summary>

Structure : les exemples sont supposés indépendants, donc la vraisemblance est un produit → underflow (*sous-dépassement*) ; le log le transforme en somme, stable et facile à dériver ; même maximum ; c'est l'origine de la cross-entropy (ch. 6), et PyTorch fournit `log_softmax` pour la même raison.

</details>

### 0B.E4 — La règle de la chaîne, et pourquoi la rétropropagation en dépend

<details><summary>Indice 1</summary>

Un réseau de neurones est une **composition** de fonctions simples : couche après couche.

</details>
<details><summary>Indice 2</summary>

Règle de la chaîne : la dérivée d'une composition est le produit des dérivées des étapes. Avec plusieurs chemins, on additionne les chemins (0B.29). La rétropropagation organise ce calcul en partant de la loss.

</details>
<details><summary>Indice 3</summary>

Structure : un réseau = une composition → la dérivée de la loss par rapport à un poids = un produit de dérivées locales le long des chemins, additionné sur les chemins → la rétropropagation parcourt le graphe de la sortie vers l'entrée et réutilise les résultats intermédiaires → coût du même ordre qu'une passe avant.

</details>

### 0B.E5 — Espérance, moyenne d'un échantillon, variance : quelles différences ?

<details><summary>Indice 1</summary>

L'une est une propriété de la **loi** (un nombre fixe, souvent inconnu), l'autre se calcule sur des **données** (et change d'un échantillon à l'autre).

</details>
<details><summary>Indice 2</summary>

Relie-les par la loi des grands nombres. La variance mesure la dispersion autour de l'espérance ; l'écart-type est dans la même unité que les données.

</details>
<details><summary>Indice 3</summary>

Structure : définition de l'espérance (moyenne pondérée par les probabilités) ; la moyenne d'échantillon l'estime et s'en rapproche quand $n$ grandit, avec une erreur en $\frac{1}{\sqrt{n}}$ ; la variance ; un exemple ML (un score de test est une moyenne d'échantillon ; on répète avec plusieurs graines et on donne un écart-type).

</details>

<a id="notebook"></a>

## Notebook, parties A à D

La partie 0 du notebook (vérification des exercices ✏️) n'a pas d'indices propres : ce sont ceux des exercices papier ci-dessus.

### Ex 0B.33 — Calculer avec Python : puissances, arrondis, `abs`, signe et `C(n, k)` 📦

<details><summary>Indice 1</summary>

Écris les expressions telles quelles dans les variables : le but est de voir ce que **Python** répond, surtout quand ça te surprend.

</details>
<details><summary>Indice 2</summary>

a : en Python, `**` passe avant le signe moins placé devant. b : `floor` et `ceil` sont dans `math`. c : `np.abs(values_33) <= 1` donne un tableau de `True`/`False`, et `True` compte pour 1 dans une somme. f : `str(2 ** 100)` est la chaîne des chiffres.

</details>
<details><summary>Indice 3</summary>

`powers = [(-3) ** 2, -3 ** 2]`, `n_small = (np.abs(values_33) <= 1).sum()`, `signs = np.sign(values_33)`, `poker_hands = math.comb(52, 5)`, `digits = len(str(2 ** 100))`.

</details>

### Ex 0B.34 — 0,99 puissance 1000 : presque 1 ou presque 0 ? 🔮

<details><summary>Indice 1</summary>

Repense à 0B.13 : une raison un peu plus petite que 1 fait fondre la suite, une raison un peu plus grande la fait exploser ; mais il faut beaucoup d'étapes.

</details>
<details><summary>Indice 2</summary>

Un repère de la fiche (101.2.4) : $0{,}99^k$ est divisé par 2 environ tous les 69 pas. Combien de fois divise-t-on par 2 en 100 pas ? en 1000 ? Pour f, regroupe : $0{,}99^{1000} \times 1{,}01^{1000} = (0{,}99 \times 1{,}01)^{1000}$.

</details>
<details><summary>Indice 3</summary>

100 pas : $100 / 69 \approx 1{,}45$ division par 2, soit une division par 2,7 environ ; 1000 pas : environ 14,5 divisions par 2 ($2^{14{,}5} \approx 23\,000$). $0{,}99 \times 1{,}01 = 0{,}9999$. g : `k = 0`, puis `while 0.99 ** k >= 0.01: k += 1`.

</details>

### Ex 0B.35 — Σ, Π et moyennes en code : `sum`, `math.prod`, `np.average`, moyenne mobile 📦

<details><summary>Indice 1</summary>

Chaque $\Sigma$ ou $\Pi$ se traduit par `sum(...)` ou `math.prod(...)` sur une expression génératrice ; attention à la borne haute de `range`.

</details>
<details><summary>Indice 2</summary>

b : `math.prod(1 + 1 / k for k in range(1, 11))`. c : `np.average(notes, weights=coefficients)`. e : pour chaque `t` de `k - 1` à `len(x) - 1`, la fenêtre est `x[t - k + 1 : t + 1]` ; ranger les moyennes dans une liste, puis `np.array(...)`.

</details>
<details><summary>Indice 3</summary>

`return np.array([np.mean(x[t - k + 1:t + 1]) for t in range(k - 1, len(x))])`. Il y a bien `len(x) - k + 1` valeurs de `t`.

</details>

### Ex 0B.36 — Galerie des fonctions usuelles : de l'affine au cosinus 📦

<details><summary>Indice 1</summary>

Les opérations NumPy (`np.exp`, `np.log`, `np.tanh`, `np.cos`, `np.abs`, `np.floor`) s'appliquent à tout un array : une ligne par fonction suffit.

</details>
<details><summary>Indice 2</summary>

`sigmoid` : `1 / (1 + np.exp(-x))`. Pour la galerie, prépare une liste de triplets `(titre, xs, ys)`, puis parcours-la en même temps que les cases : `for ax, (title, xs, ys) in zip(axes.ravel(), panels):`.

</details>
<details><summary>Indice 3</summary>

Dans la boucle : `ax.plot(xs, ys)`, `ax.axhline(0)`, `ax.axvline(0)`, `ax.set_title(title)`. Pour $\ln$, `xs = np.linspace(0.05, 4, 400)`. N'oublie pas `return fig`.

</details>

### Ex 0B.37 — `exp` et `log` en NumPy : `-inf`, `nan` et dépassements 🐛

<details><summary>Indice 1</summary>

Un `float64` ne représente que des nombres entre environ $10^{-308}$ et $10^{308}$ en valeur absolue (et 0) : au-delà, on obtient `inf` ; en dessous, 0. Et $\ln$ n'est défini que pour des nombres strictement positifs.

</details>
<details><summary>Indice 2</summary>

1 : $0{,}3^{1000}$ est-il représentable ? 2 : que vaut le produit de 500 nombres plus grands que 100 ? 3 : que devient $x - \bar{x}$ pour les valeurs plus petites que la moyenne ? Les corrections sont écrites dans l'énoncé.

</details>
<details><summary>Indice 3</summary>

1 : `np.sum(np.log(probs))` ; 2 : `np.exp(np.mean(np.log(values)))` ; 3 : `np.log1p(values - values.min())`. d : `n = 0`, puis `while np.isfinite(np.exp(n + 1)): n += 1`.

</details>

### Ex 0B.38 — `linalg_basics` (1) : additionner, soustraire, multiplier des vecteurs 🔨

<details><summary>Indice 1</summary>

Composante par composante : on parcourt les deux vecteurs en même temps, avec `zip(u, v)`.

</details>
<details><summary>Indice 2</summary>

Commence par `if len(u) != len(v): raise ValueError(...)`, puis une compréhension. Pour `scalar_multiply`, un seul vecteur : `for x in v`.

</details>
<details><summary>Indice 3</summary>

`return [float(a + b) for a, b in zip(u, v)]` ; même modèle pour `-` et `*` ; `return [float(c * x) for x in v]`.

</details>

### Ex 0B.39 — `linalg_basics` (2) : produit scalaire, norme, distance, cosinus 🔨

<details><summary>Indice 1</summary>

Tout part du produit scalaire : une somme de produits. La norme est une racine de somme de carrés (pour $p = 2$), la distance une norme de différence, le cosinus un produit scalaire divisé par deux normes.

</details>
<details><summary>Indice 2</summary>

`norm` : vérifie `p >= 1` (sinon `ValueError`) ; renvoie `0.0` pour un vecteur vide ; traite `p == math.inf` à part (`float(max(abs(x) for x in v))`) ; sinon `float(sum(abs(x) ** p for x in v) ** (1 / p))`. `cosine_similarity` : calcule les deux normes, refuse une norme nulle, puis divise.

</details>
<details><summary>Indice 3</summary>

`dot` : vérifie les longueurs (sinon `ValueError`), puis `total = 0.0` et `for a, b in zip(u, v): total += a * b`. `distance` : `return norm(vector_subtract(u, v))` si tu as fait 0B.38, sinon `math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))` après le même contrôle des longueurs. Cosinus : `value = dot(u, v) / (norm(u) * norm(v))`, puis `max(-1.0, min(1.0, value))`.

</details>

### Ex 0B.40 — Tes fonctions contre NumPy : mêmes résultats, autre vitesse 📦

<details><summary>Indice 1</summary>

Chaque fonction de ta librairie a un équivalent NumPy d'une ligne ; la documentation de `np.linalg.norm` liste les valeurs possibles de `ord`.

</details>
<details><summary>Indice 2</summary>

a : `a @ b`. b : `np.linalg.norm(a)`, `np.linalg.norm(a, ord=1)`, `np.linalg.norm(a, ord=np.inf)`. c : `a @ b / (np.linalg.norm(a) * np.linalg.norm(b))`. e : `measure(fonction, argument1, argument2)` renvoie le meilleur temps.

</details>
<details><summary>Indice 3</summary>

```python
def speedup_dot():
    u_list, v_list = u_big.tolist(), v_big.tolist()
    return measure(mylearn.linalg_basics.dot, u_list, v_list) / measure(np.dot, u_big, v_big)
```

</details>

### Ex 0B.41 — Distance ou similarité cosinus : l'effet de la longueur 🔬

<details><summary>Indice 1</summary>

`min(docs, key=...)` parcourt les **noms** du dictionnaire et renvoie celui dont la clé calculée est la plus petite ; `max` pour la plus grande.

</details>
<details><summary>Indice 2</summary>

a : `key=lambda name: np.linalg.norm(query - docs[name])`. b : `max` avec la similarité cosinus. c : une fonction `unit(v) = v / np.linalg.norm(v)`, puis la distance entre `unit(query)` et `unit(docs[name])`.

</details>
<details><summary>Indice 3</summary>

d : `distances = [np.linalg.norm(k * query - target) for k in range(1, 11)]`, idem pour les cosinus, avec `target = docs["penguins_long"]` ; `return distances, cosines`.

</details>

### Ex 0B.42 — `linalg_basics` (3) : forme, transposée, identité, matrice × vecteur 🔨

<details><summary>Indice 1</summary>

Une matrice est une liste de lignes : `len(A)` est le nombre de lignes, `len(A[0])` le nombre de colonnes, `A[i][j]` l'élément ligne `i`, colonne `j`.

</details>
<details><summary>Indice 2</summary>

`shape` : vérifie `len(A) > 0`, `len(A[0]) > 0`, puis que chaque ligne a la même longueur. `transpose` : la nouvelle ligne `j` est `[float(A[i][j]) for i in range(n_rows)]`. `identity` : `ValueError` si `n < 1`, puis une compréhension imbriquée (une nouvelle liste par ligne). `matvec` : `shape(A)`, comparaison avec `len(v)`, puis un `dot` par ligne.

</details>
<details><summary>Indice 3</summary>

`return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]` ; `raise ValueError(f"cannot multiply {(n_rows, n_cols)} by ({len(v)},)")` ; `return [dot(row, v) for row in A]`.

</details>

### Ex 0B.43 — `linalg_basics` (4) : `matmul` et vérification des formes 🔨

<details><summary>Indice 1</summary>

L'élément $(i, j)$ de $\mathbf{A}\mathbf{B}$ est le produit scalaire de la ligne $i$ de $\mathbf{A}$ et de la colonne $j$ de $\mathbf{B}$ : tu as déjà `dot` et `transpose`.

</details>
<details><summary>Indice 2</summary>

`shape_a, shape_b = shape(A), shape(B)` ; si `shape_a[1] != shape_b[0]`, `ValueError` avec les deux formes. Les colonnes de `B` sont les lignes de `transpose(B)`.

</details>
<details><summary>Indice 3</summary>

```python
columns = transpose(B)
return [[dot(row, column) for column in columns] for row in A]
```

</details>

### Ex 0B.44 — AB = BA ? (AB)ᵀ = BᵀAᵀ ? Prédire, puis tester 🔮

<details><summary>Indice 1</summary>

Relis la fiche §101.4.3 : les propriétés du produit matriciel et la transposée d'un produit. Refais son exemple à la main dans les deux ordres.

</details>
<details><summary>Indice 2</summary>

b et c : pense aux formes avec des matrices rectangulaires, $(m, n)$ et $(n, p)$ : laquelle des deux écritures a encore un sens ? f : multiplie à la main deux matrices diagonales $2 \times 2$.

</details>
<details><summary>Indice 3</summary>

$\begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix}\begin{pmatrix} c & 0 \\ 0 & d \end{pmatrix} = \begin{pmatrix} ac & 0 \\ 0 & bd \end{pmatrix}$ : et dans l'autre ordre ? Quatre des six affirmations sont vraies.

</details>

### Ex 0B.45 — Le produit qui n'en est pas un : `*`, `@`, `(n,)` et `(n, 1)` 🐛

<details><summary>Indice 1</summary>

Écris les formes de chaque opérande : `X` est `(5, 3)`, `w` est `(3,)`, `W` est `(3, 2)`, `u` et `v` sont `(3,)`. Puis demande-toi ce que fait `*` (élément par élément, avec broadcasting) et ce que fait `@`.

</details>
<details><summary>Indice 2</summary>

1 : `X * w` multiplie chaque ligne de `X` par `w`, élément par élément. 2 : `W @ X` demande que `W` ait autant de colonnes que `X` a de lignes. 3 : `.T` ne change rien à un tableau à une seule dimension.

</details>
<details><summary>Indice 3</summary>

Corrections : `X @ w + b`, `X @ W + b`, et pour `outer` une colonne fois une ligne : `u[:, None] @ v[None, :]` (ou `u.reshape(-1, 1) @ v.reshape(1, -1)`).

</details>

### Ex 0B.46 — Inverse et systèmes : `np.linalg.inv` et `np.linalg.solve` 📦

<details><summary>Indice 1</summary>

Un système linéaire s'écrit $\mathbf{M}\mathbf{x} = \mathbf{y}$ : la matrice des coefficients (une ligne par équation) et le second membre.

</details>
<details><summary>Indice 2</summary>

c : `np.linalg.solve(M46, [5, 6])`. d : `error_name(np.linalg.inv, singular46)` (partie A). e : la matrice est `[[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]` et le second membre `[8, -11, -3]`.

</details>
<details><summary>Indice 3</summary>

f : `x = np.linalg.solve(A_big, b_big)`, puis `np.linalg.norm(A_big @ x - b_big) < 1e-8`.

</details>

### Ex 0B.47 — Pentes numériques : vérifier tes dérivées à la main 🔨

<details><summary>Indice 1</summary>

La pente centrée compare $f$ juste avant et juste après $a$ : `f(a + h)` et `f(a - h)`.

</details>
<details><summary>Indice 2</summary>

`return (f(a + h) - f(a - h)) / (2 * h)`. Pour `derivatives`, reprends tes réponses de 0B.22 et 0B.23 (la **formule** de la dérivée, pas sa valeur en un point) et écris-les avec `np.exp` et `np.log`.

</details>
<details><summary>Indice 3</summary>

Par exemple `"0B.22d": lambda x: np.log(x) + 1` et `"0B.23c": lambda x: 2 * x / (x ** 2 + 1)`. Si une dérivée est signalée fausse, compare au point indiqué ta formule avec la pente : un facteur oublié (dérivée intérieure) ou un signe ?

</details>

### Ex 0B.48 — Lire les variations : f, f′ et les points où f′ s'annule 📈

<details><summary>Indice 1</summary>

Sur le panneau du bas, cherche où la courbe de la pente coupe la droite $y = 0$ ; sur celui du haut, ces abscisses correspondent aux sommets et aux creux.

</details>
<details><summary>Indice 2</summary>

`fig, axes = plt.subplots(2, 1, sharex=True)`, puis `axes[0].plot(x48, f48(x48))` et `axes[1].plot(x48, [centered_slope(f48, x) for x in x48])`. b : la pente passe-t-elle de positive à négative, ou l'inverse ?

</details>
<details><summary>Indice 3</summary>

a : deux entiers lus sur le graphique. c : `f48(x48).min()`. d : calcule $f$ en la plus petite des deux abscisses.

</details>

### Ex 0B.49 — Carte de lignes de niveau et flèches du gradient 📈

<details><summary>Indice 1</summary>

`np.meshgrid(xs, ys)` fabrique deux tableaux `X` et `Y` qui couvrent tous les points de la grille ; `f49(X, Y)` calcule alors $f$ partout d'un coup.

</details>
<details><summary>Indice 2</summary>

`grad_f49` : `return 2 * (x - 1), 4 * y`. Pour la carte : `fig, ax = plt.subplots()`, `ax.contour(X, Y, f49(X, Y), levels=[1, 2, 3, 4, 5, 6])`, puis une grille plus grossière `Xc, Yc` et `ax.quiver(Xc, Yc, *grad_f49(Xc, Yc))`.

</details>
<details><summary>Indice 3</summary>

b : là où les deux composantes du gradient sont nulles. d : sur la carte, depuis $(1, 0)$, compare la distance à parcourir jusqu'à la ligne de niveau 2 vers la droite et vers le haut.

</details>

### Ex 0B.50 — Contre le gradient, avec lui ou le long d'une ligne de niveau : où va f ? 🔮

<details><summary>Indice 1</summary>

Relis la fiche §101.6.3, puis regarde la carte de 0B.49 : comment les flèches sont-elles placées par rapport aux lignes de niveau et aux niveaux croissants ?

</details>
<details><summary>Indice 2</summary>

d : le signe de $\frac{\partial f}{\partial x}$ en $(3, 1)$ dit si $f$ monte ou descend quand on avance selon $x$. e et f : les directions s'écrivent `(np.cos(t), np.sin(t))` avec `t = np.radians(angle)`.

</details>
<details><summary>Indice 3</summary>

Une fonction `best_angle(step)` : pour `angles = np.arange(360)`, calcule `f50(3 + step * cos, 1 + step * sin) - f50(3, 1)` et renvoie l'angle de la plus petite valeur (`np.argmin`) ; appelle-la avec 0.01, puis 0.5.

</details>

### Ex 0B.51 — Dérivées partielles numériques et somme sur les chemins 🔨

<details><summary>Indice 1</summary>

Une dérivée partielle, c'est une pente centrée où l'on ne fait bouger qu'une seule variable.

</details>
<details><summary>Indice 2</summary>

`partial_x` : `(f(x + h, y) - f(x - h, y)) / (2 * h)`. `dz_dx_paths` : calcule `u` et `v`, puis `∂z/∂u · du/dx + ∂z/∂v · dv/dx` avec tes dérivées locales de 0B.29. `grad_paths` : les écarts $e_i = w x_i + b - y_i$, puis $\frac{\partial L}{\partial w} = \sum_i 2 e_i x_i$ et $\frac{\partial L}{\partial b} = \sum_i 2 e_i$.

</details>
<details><summary>Indice 3</summary>

`dz_dx_paths` : `return (2 * u + v) * 2 + u * (2 * x)`. `one_step` : `dw, db = grad_paths(w, b)`, puis `return [w - eta * dw, b - eta * db]`.

</details>

### Ex 0B.52 — Simuler des dés : fréquences, indépendance, loi des grands nombres 🔬

<details><summary>Indice 1</summary>

Une fréquence est la moyenne d'un masque booléen : `(condition).mean()`. Pour une intersection, combine deux masques avec `&`.

</details>
<details><summary>Indice 2</summary>

`A = d1 % 2 == 0`, `B = d1 + d2 == 7`, `C = d1 + d2 == 8`, puis `(A & B).mean() - A.mean() * B.mean()`. c : `np.cumsum(d1 == 6)` compte les 6 au fil des lancers ; divise par le nombre de lancers, `np.arange(1, n + 1)`.

</details>
<details><summary>Indice 3</summary>

d : une boucle `for n_rolls in [100, 400, 1600, 6400]:` (pas `n`, qui sert au 0B.53) qui ajoute `freqs.std()` à une liste `stds`, puis `ratios = [round(stds[i] / stds[i + 1]) for i in range(3)]`.

</details>

### Ex 0B.53 — Espérance et variance : le calcul exact contre la simulation 🔨

<details><summary>Indice 1</summary>

L'espérance est une somme pondérée : les valeurs multipliées par leurs probabilités, puis additionnées (101.7.3).

</details>
<details><summary>Indice 2</summary>

`expectation` : `np.sum(values * probs)`. `variance` : réutilise `expectation` deux fois, avec `values ** 2` puis `values`. Les tirages : suis exactement l'énoncé (le même générateur, `x` d'abord, `y` ensuite).

</details>
<details><summary>Indice 3</summary>

`return expectation(values ** 2, probs) - expectation(values, probs) ** 2` ; `sample_stats = [x.mean(), x.var()]`, `var_sum_indep = (x + y).var()`, `var_sum_same = (x + x).var()`.

</details>

### Ex 0B.54 — L'ordre des produits : calculer A·B·C·v des dizaines de fois plus vite 🏆

<details><summary>Indice 1</summary>

Les parenthèses décident de l'ordre des calculs : de droite à gauche, chaque produit fait intervenir un **vecteur**, jamais deux matrices $n \times n$.

</details>
<details><summary>Indice 2</summary>

`matmul_cost` : si une forme n'a qu'une dimension, remplace `(n,)` par `(n, 1)` ; puis `m * n * p`. `chain_costs` : de gauche à droite, $(n, n) \times (n, n)$ deux fois, puis $(n, n) \times (n,)$ ; de droite à gauche, trois fois $(n, n) \times (n,)$.

</details>
<details><summary>Indice 3</summary>

e : de gauche à droite, `matmul_cost((64, 784), (784, 512)) + matmul_cost((64, 512), (512, 512)) + matmul_cost((64, 512), (512, 10))` ; dans l'autre ordre, `W2 W3` est `(512, 10)`, puis `W1 (W2 W3)` est `(784, 10)`, puis `X (…)` est `(64, 10)`.

</details>
