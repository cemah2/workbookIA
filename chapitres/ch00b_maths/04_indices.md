# 0B · Maths du lycée au ML — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🧮 🗣️ 🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 0B.Q1 — Puissances, racines et notation scientifique

<details><summary>Indice 1</summary>

Relis les règles de 101.1.1 : ce qui se multiplie, ce qui s'additionne. Pour chaque affirmation qui te semble fausse, cherche un contre-exemple avec de petits nombres.

</details>
<details><summary>Indice 2</summary>

1 : quand on multiplie deux puissances de même base, que fait-on des exposants ? 3 : essaie $a = 9$ et $b = 16$. 4 : $2^{10} = 1024 \approx 10^3$.

</details>
<details><summary>Indice 3</summary>

Trois affirmations sont vraies : la 2, la 4 et la 5. Trouve pourquoi la 1 et la 3 sont fausses.

</details>

### 0B.Q2 — |x|, ⌊x⌋, ⌈x⌉ et sign(x)

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

### 0B.Q7 — Règles des logarithmes et des exponentielles

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

### 0B.Q11 — Gradient et lignes de niveau

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

### 0B.R1 — Une somme en Python, trois façons

<details><summary>Indice 1</summary>

Une somme $\Sigma$ se traduit par une boucle qui accumule dans une variable initialisée à 0 ; un produit $\Pi$, dans une variable initialisée à 1.

</details>
<details><summary>Indice 2</summary>

`total = 0`, puis `for i in range(...)`, puis `total += ...`. En une ligne : `sum(expression for i in range(...))`. `math.prod` prend un itérable, comme `sum`.

</details>
<details><summary>Indice 3</summary>

`range(a, b)` s'arrête **avant** `b` : pour aller de 1 à 5 inclus, il faut `range(1, 6)`. Tu dois trouver 55, puis 120.

</details>

### 0B.R2 — shape, ndim et axis d'un array

<details><summary>Indice 1</summary>

Une ligne par manchot, une colonne par mesure : la forme est `(nombre de lignes, nombre de colonnes)`.

</details>
<details><summary>Indice 2</summary>

`X[:, 2]` : toutes les lignes, la colonne d'indice 2 (un indice entier retire un axe). `axis=0` fait **disparaître** l'axe 0, celui des lignes.

</details>
<details><summary>Indice 3</summary>

`X.shape == (333, 4)`. Une moyenne par colonne donne 4 nombres. `.T` échange les deux nombres de la forme.

</details>

### 0B.R3 — Une fonction qui prend une fonction

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

### Ex 0B.1 — Puissances, racines et notation scientifique sans calculatrice

<details><summary>Indice 1</summary>

Trois règles suffisent : $a^m \times a^n = a^{m+n}$, $(a^m)^n = a^{mn}$, $a^{-n} = \frac{1}{a^n}$. Et $a^{1/n}$ est la racine $n$-ième de $a$.

</details>
<details><summary>Indice 2</summary>

b : simplifie d'abord le numérateur en une seule puissance de 10. d : $\sqrt{ab} = \sqrt{a}\,\sqrt{b}$. e : $8^{2/3} = (2^3)^{2/3}$. f : multiplie les nombres entre eux, puis les puissances de 10 entre elles.

</details>
<details><summary>Indice 3</summary>

c : $\frac{1}{25}$. f : $6{,}4 \times 10^{2}$. g : pour passer de 5,6 à 0,000 56, la virgule recule de combien de rangs ? h : $2^{30} = (2^{10})^3$ et $2^{10} \approx 10^3$.

</details>

### Ex 0B.2 — Valeur absolue, partie entière et signe

<details><summary>Indice 1</summary>

$|x|$ est la distance de $x$ à 0. $\lfloor x \rfloor$ est l'entier juste en dessous (vers la gauche de la droite graduée), $\lceil x \rceil$ l'entier juste au-dessus. $\mathrm{sign}(x)$ vaut −1, 0 ou 1.

</details>
<details><summary>Indice 2</summary>

c et d : dessine −4, −3,7 et −3 sur une droite graduée. g et h : $|x - 4| \leq 1{,}5$ veut dire que $x$ est à une distance au plus 1,5 de 4. i : l'inégalité est **stricte**.

</details>
<details><summary>Indice 3</summary>

a : $7 + 2$. c : −4 (en dessous de −3,7) ; d : −3. f : $-1 + 0 + 1$. g et h : $4 - 1{,}5$ et $4 + 1{,}5$. i : liste les entiers de −2 à 2.

</details>

### Ex 0B.3 — Lire et calculer des Σ et des Π

<details><summary>Indice 1</summary>

Écris chaque somme **en entier**, terme par terme, avant de calculer : par exemple $\sum_{i=1}^{3} (i + 2) = 3 + 4 + 5$.

</details>
<details><summary>Indice 2</summary>

c : le premier terme est $3^0$. f : $\sum_i (3x_i + 1) = 3\sum_i x_i + \sum_i 1$, et la seconde somme compte 4 termes. g : on ajoute 100 fois le même nombre. h : trois termes seulement, pour $i = 2, 3, 4$.

</details>
<details><summary>Indice 3</summary>

c : $1 + 3 + 9 + 27$. d : $2 \times 3 \times 4 \times 5$. e : $4 + 1 + 16 + 9$. f : $\sum_i x_i = 8$. h : $2 \times (-1) + 3 \times 4 + 4 \times 3$.

</details>

### Ex 0B.4 — Moyenne pondérée, somme pondérée et moyenne mobile

<details><summary>Indice 1</summary>

Moyenne pondérée : $\frac{\sum_i w_i x_i}{\sum_i w_i}$. Somme pondérée d'un neurone : $\sum_i w_i x_i + b$, **sans** division. Moyenne mobile d'ordre 3 : la moyenne des 3 dernières valeurs.

</details>
<details><summary>Indice 2</summary>

b : la somme des coefficients vaut 6. d : $m_3$ est la moyenne de $x_1, x_2, x_3$ ; $m_4$ celle de $x_2, x_3, x_4$, etc. e : la première moyenne mobile existe à $t = k$, la dernière à $t = n$.

</details>
<details><summary>Indice 3</summary>

b : $\frac{14 \times 3 + 8 \times 1 + 11 \times 2}{6}$. c : $2 - 2 + 3 - 1$. d : $\frac{2 + 4 + 9}{3}$, $\frac{4 + 9 + 1}{3}$, $\frac{9 + 1 + 5}{3}$, $\frac{1 + 5 + 6}{3}$. e : $n - k + 1$.

</details>

### Ex 0B.5 — Ensembles : union, intersection, complémentaire et cardinal

<details><summary>Indice 1</summary>

Commence par **lister** $A$, $B$ et $C$ en entier : avec 12 éléments, c'est rapide et ça évite toutes les erreurs.

</details>
<details><summary>Indice 2</summary>

$A = \{2, 4, 6, 8, 10, 12\}$, $B = \{3, 6, 9, 12\}$. b : $|A \cup B| = |A| + |B| - |A \cap B|$. d : « ni l'un ni l'autre » est le complémentaire de « l'un ou l'autre ».

</details>
<details><summary>Indice 3</summary>

a : $A \cap B = \{6, 12\}$. d : $12 - |A \cup B|$. e : parmi 1, 2 et 3, lesquels sont pairs ? f : $A \cup B$, plus les éléments de $C$ qui n'y sont pas encore.

</details>

### Ex 0B.6 — Droites et paraboles : pente, ordonnée à l'origine, racines, sommet

<details><summary>Indice 1</summary>

Pente d'une droite : $\frac{\Delta y}{\Delta x}$, les deux points pris dans le même ordre en haut et en bas. Parabole $ax^2 + bx + c$ : $\Delta = b^2 - 4ac$, racines $\frac{-b \pm \sqrt{\Delta}}{2a}$, sommet en $-\frac{b}{2a}$.

</details>
<details><summary>Indice 2</summary>

a : $\frac{-3 - 5}{3 - (-1)}$. b : écris $y = mx + p$ et remplace par le point $(-1, 5)$. c : résous $mx + p = 0$. d : ici $a = 2$, $b = -8$, $c = 6$.

</details>
<details><summary>Indice 3</summary>

a : $\frac{-8}{4}$. b : $5 = -2 \times (-1) + p$. d : $64 - 48$. e et f : $\frac{8 \pm 4}{4}$. g : $\frac{8}{4}$. h : calcule $f$ au sommet.

</details>

### Ex 0B.7 — Cosinus : cercle, période et planning en cosinus

<details><summary>Indice 1</summary>

$\pi$ rad = 180°. Sur le cercle trigonométrique, le cosinus est l'**abscisse** du point. Le cosinus est pair ($\cos(-x) = \cos x$) et $2\pi$-périodique.

</details>
<details><summary>Indice 2</summary>

a : $60° = \frac{\pi}{3}$. c : $\frac{2\pi}{3}$ est le symétrique de $\frac{\pi}{3}$ par rapport à l'axe vertical. e : $4\pi$, c'est deux tours complets. f à h : remplace $t$ et $T$, puis simplifie $\frac{\pi t}{T}$.

</details>
<details><summary>Indice 3</summary>

a : $\frac{3{,}1416}{3}$. c : $-\cos\frac{\pi}{3}$ et $\cos\frac{\pi}{3} = \frac{1}{2}$. f : $\cos\frac{\pi}{2} = 0$. g : $\frac{1}{2}(1 + 0{,}707)$. h : $\cos\frac{3\pi}{4} = -0{,}707$, puis multiplie par 0,01.

</details>

### Ex 0B.8 — Vecteurs : somme, multiple, norme et distance

<details><summary>Indice 1</summary>

On additionne des vecteurs **composante par composante**. Norme L2 : $\sqrt{\sum_i u_i^2}$ ; L1 : $\sum_i |u_i|$ ; L∞ : $\max_i |u_i|$. Distance : la norme de la différence.

</details>
<details><summary>Indice 2</summary>

f : $\mathbf{p}_1 - \mathbf{p}_2 = (-8, -6)$. g : même calcul avec $(-8, -0{,}6)$. h : divise chaque composante de $\mathbf{u}$ par sa norme.

</details>
<details><summary>Indice 3</summary>

c : $\sqrt{9 + 16}$. d : $3 + 4$. f : $\sqrt{64 + 36}$. g : $\sqrt{64 + 0{,}36}$ : la nageoire ne pèse presque plus rien dans la distance. h : $\left(\frac{3}{5}, \frac{-4}{5}\right)$.

</details>

### Ex 0B.9 — Transposée et produit matrice-vecteur

<details><summary>Indice 1</summary>

La transposée échange lignes et colonnes : $(\mathbf{A}^\top)_{ij} = \mathbf{A}_{ji}$. Le produit $\mathbf{A}\mathbf{v}$ exige que le nombre de **colonnes** de $\mathbf{A}$ soit égal à la dimension de $\mathbf{v}$.

</details>
<details><summary>Indice 2</summary>

c, lecture par lignes : chaque composante du résultat est le produit scalaire d'une ligne de $\mathbf{A}$ avec $\mathbf{v}$. Lecture par colonnes : $1 \times (\text{colonne } 1) + 2 \times (\text{colonne } 2) + (-1) \times (\text{colonne } 3)$. f : chaque ligne de $\mathbf{X}$ est un exemple.

</details>
<details><summary>Indice 3</summary>

b : $(\mathbf{A}^\top)_{31} = \mathbf{A}_{13}$. c : $(2 + 0 - 1,\; -1 + 6 - 2)$. d : $\mathbf{A}^\top = \begin{pmatrix} 2 & -1 \\ 0 & 3 \\ 1 & 2 \end{pmatrix}$. e : $\mathbf{A}$ a 3 colonnes. f : $\mathbf{X}\mathbf{w} = (4{,}5 ; 1{,}5 ; -2)$, puis ajoute $b$.

</details>

### Ex 0B.10 — Taux d'accroissement : de la sécante à la tangente

<details><summary>Indice 1</summary>

Le taux d'accroissement entre $a$ et $a + h$ est $\frac{f(a + h) - f(a)}{h}$ : la pente de la corde (la sécante) entre les deux points.

</details>
<details><summary>Indice 2</summary>

a : $f(3) = 12$. b : $f(2{,}1) = 4{,}41 + 2{,}1$. d : développe $(2 + h)^2 + (2 + h)$, retire 6, puis divise par $h$. e : développe $y = 6 + 5(x - 2)$.

</details>
<details><summary>Indice 3</summary>

d : le numérateur vaut $5h + h^2$, donc le taux vaut $5 + h$. e : $y = 5x - 4$. f : $6 + 5 \times 0{,}05$ ; la vraie valeur est $f(2{,}05) = 4{,}2025 + 2{,}05$.

</details>

### Ex 0B.11 — Probabilités : issues, complémentaire, union

<details><summary>Indice 1</summary>

Dessine le tableau $6 \times 6$ des 36 issues (dé 1 en lignes, dé 2 en colonnes) et entoure les issues favorables. $P = \frac{\text{favorables}}{36}$.

</details>
<details><summary>Indice 2</summary>

a : les couples de somme 8 sont sur une diagonale du tableau. c : « au moins un 1 » est le complémentaire de « aucun 1 », qui compte $5 \times 5$ issues. d : $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. f : 13 cœurs, 12 figures, dont 3 figures de cœur.

</details>
<details><summary>Indice 3</summary>

a : $\frac{5}{36}$. c : $1 - \frac{25}{36}$. d : $(4, 4)$ est à la fois un double et une somme 8 : $\frac{5 + 6 - 1}{36}$. e : sommes 10, 11 et 12 : $3 + 2 + 1$ issues. f : $\frac{13 + 12 - 3}{52}$.

</details>

### Ex 0B.12 — Dénombrer : choix successifs, factorielle et C(n, k)

<details><summary>Indice 1</summary>

Trois outils : choix successifs (on multiplie les nombres de choix), $n!$ pour ranger $n$ objets, $\binom{n}{k}$ pour choisir $k$ objets **sans ordre**. Demande-toi à chaque fois : l'ordre compte-t-il ? peut-on répéter ?

</details>
<details><summary>Indice 2</summary>

a : 10 choix pour chaque chiffre. b : 10 choix, puis un de moins à chaque fois. f : une paire de classes. g : chaque feature est dedans ou pas : 2 choix par feature. h : choisis d'abord le président, puis les deux autres membres, sans ordre.

</details>
<details><summary>Indice 3</summary>

b : $10 \times 9 \times 8 \times 7$. d : $\frac{6 \times 5}{2}$. e : $\frac{8 \times 7 \times 6}{3 \times 2 \times 1}$. g : $2^5$. h : $10 \times \binom{9}{2}$.

</details>

### Ex 0B.13 — Suites géométriques : ce qui fond, ce qui explose

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

### Ex 0B.15 — Exponentielles et logarithmes : règles de calcul

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

### Ex 0B.17 — Sigmoïde et tanh : valeurs, limites, symétries

<details><summary>Indice 1</summary>

$\sigma(x) = \frac{1}{1 + e^{-x}}$ et $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. Calcule d'abord la ou les exponentielles, puis la fraction.

</details>
<details><summary>Indice 2</summary>

b : $e^{-3} \approx 0{,}0498$. d : $e^{0{,}5} \approx 1{,}6487$ et $e^{-0{,}5} \approx 0{,}6065$. e : $\sigma(1) \approx 0{,}7311$. f : quand $x \to -\infty$, que devient $e^{-x}$ ? g : isole $e^{-x}$ dans $\frac{1}{1 + e^{-x}} = 0{,}9$.

</details>
<details><summary>Indice 3</summary>

c : $1 - \sigma(3)$. e : même valeur qu'en d, car $\tanh(x) = 2\sigma(2x) - 1$. g : $1 + e^{-x} = \frac{10}{9}$, donc $e^{-x} = \frac{1}{9}$ et $x = \ln 9$. h : utilise la symétrie de c.

</details>

### Ex 0B.18 — Produit scalaire, similarité cosinus et produit de Hadamard

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

### Ex 0B.20 — Produit matriciel : calculer et vérifier les formes

<details><summary>Indice 1</summary>

$(m, n) \times (n, p) \to (m, p)$. L'élément $(i, j)$ du produit est le produit scalaire de la **ligne** $i$ de la matrice de gauche et de la **colonne** $j$ de la matrice de droite.

</details>
<details><summary>Indice 2</summary>

Formes : $\mathbf{A}$ est $(3, 2)$, $\mathbf{B}$ est $(2, 3)$, $\mathbf{C}$ est $(2, 2)$. c : $\mathbf{B}\mathbf{A}$ a 4 éléments ; le premier est (ligne 1 de $\mathbf{B}$) · (colonne 1 de $\mathbf{A}$) $= (2, 1, 0) \cdot (1, 0, 3)$. h : chaque élément de $\mathbf{A}\mathbf{B}$ coûte $n$ multiplications.

</details>
<details><summary>Indice 3</summary>

c : les autres éléments sont $(2, 1, 0) \cdot (2, -1, 1)$, $(1, -1, 4) \cdot (1, 0, 3)$ et $(1, -1, 4) \cdot (2, -1, 1)$. d : $(1, 2) \cdot (2, 1)$. e : $(3, 1) \cdot (1, -1)$. f : $(2, 2) \times (3, 2)$. g : 3 lignes, 2 colonnes. h : $m \times n \times p$.

</details>

### Ex 0B.21 — Identité, inverse 2 × 2 et système de deux équations

<details><summary>Indice 1</summary>

Pour $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ : $\det = ad - bc$, et, si $\det \neq 0$, l'inverse est $\frac{1}{\det}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$ (on échange $a$ et $d$, on change le signe de $b$ et $c$).

</details>
<details><summary>Indice 2</summary>

a : $3 \times 2 - 1 \times 4$. c : le système s'écrit $\mathbf{M}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 5 \\ 6 \end{pmatrix}$, donc $\begin{pmatrix} x \\ y \end{pmatrix} = \mathbf{M}^{-1}\begin{pmatrix} 5 \\ 6 \end{pmatrix}$. d et e : une matrice est inversible si et seulement si son déterminant est non nul.

</details>
<details><summary>Indice 3</summary>

b : $\frac{1}{2}\begin{pmatrix} 2 & -1 \\ -4 & 3 \end{pmatrix}$. c : $(1 \times 5 - 0{,}5 \times 6,\; -2 \times 5 + 1{,}5 \times 6)$, puis vérifie dans les deux équations. e : $6 - 2k = 0$. f : pour une matrice diagonale, on inverse chaque élément de la diagonale.

</details>

### Ex 0B.22 — Dériver avec les règles

<details><summary>Indice 1</summary>

Le tableau de 101.5.2 : $(x^n)' = n x^{n-1}$, $(\sqrt{x})' = \frac{1}{2\sqrt{x}}$, $\left(\frac{1}{x}\right)' = -\frac{1}{x^2}$, $(e^x)' = e^x$, $(\ln x)' = \frac{1}{x}$ ; $(uv)' = u'v + uv'$ ; $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$.

</details>
<details><summary>Indice 2</summary>

c, d : règle du produit, avec $u = x^2$ et $v = e^x$ (puis $u = x$ et $v = \ln x$). e, f : règle du quotient. Dérive d'abord, **puis** remplace $x$ par sa valeur.

</details>
<details><summary>Indice 3</summary>

a : $12x^2 - 2$. b : $\frac{1}{4} - \frac{1}{16}$. c : $(2x + x^2)e^x$, soit $3e$ en 1. d : $\ln x + 1$. e : $\frac{(x - 1) - (x + 1)}{(x - 1)^2} = \frac{-2}{(x - 1)^2}$. f : $\frac{e^x(x - 1)}{x^2}$. g : $20x^3$.

</details>

### Ex 0B.23 — Règle de la chaîne : décomposer, puis dériver

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

### Ex 0B.25 — Lignes de niveau, dérivées partielles, gradient et un pas de descente

<details><summary>Indice 1</summary>

$\frac{\partial f}{\partial x}$ : dérive par rapport à $x$ en traitant $y$ comme une constante (et inversement). Le gradient est le vecteur des deux dérivées partielles ; un pas de descente : $\mathbf{p} \leftarrow \mathbf{p} - \eta\,\nabla f(\mathbf{p})$.

</details>
<details><summary>Indice 2</summary>

b : $\frac{\partial}{\partial x}(x - 1)^2 = 2(x - 1)$, et $2y^2$ est une constante. c : $\frac{\partial}{\partial y} 2y^2 = 4y$. f : $(3, 1) - 0{,}1 \times (4, 4)$. h : une somme de deux carrés est minimale quand les deux carrés sont nuls. i, j : pour $\frac{\partial g}{\partial y}$, $x^2$ est un coefficient constant.

</details>
<details><summary>Indice 3</summary>

a : $4 + 2$. e : $\sqrt{16 + 16} = 4\sqrt{2}$. f : $(2{,}6 ; 0{,}6)$. g : $1{,}6^2 + 2 \times 0{,}6^2$. h : $x = 1$, $y = 0$. i : $2xy$. j : $x^2 + 3y^2$. Dessin : $(x - 1)^2 + 2y^2 = 2$ est une ellipse de demi-axes $\sqrt{2}$ (horizontal) et 1 (vertical) ; pour 8, $2\sqrt{2}$ et 2.

</details>

### Ex 0B.26 — Indépendance : tester P(A ∩ B) = P(A) P(B)

<details><summary>Indice 1</summary>

Calcule les deux membres de $P(A \cap B) = P(A)\,P(B)$ séparément, **en fractions**, puis compare-les exactement.

</details>
<details><summary>Indice 2</summary>

a : 18 issues sur 36. c : liste les couples de somme 7 dont le premier dé est pair. f : même chose pour la somme 8. Garde les fractions : $\frac{3}{36} = \frac{1}{12}$ ; $\frac{1}{2} \times \frac{5}{36} = \frac{5}{72}$.

</details>
<details><summary>Indice 3</summary>

c : $(2, 5), (4, 3), (6, 1)$, et $\frac{1}{2} \times \frac{1}{6} = \frac{1}{12}$. f : $(2, 6), (4, 4), (6, 2)$. h : $\frac{6}{72} - \frac{5}{72}$.

</details>

### Ex 0B.27 — Espérance et variance d'une variable discrète

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

### Ex 0B.30 — Fermi : combien de multiplications dans un produit matriciel ?

<details><summary>Indice 1</summary>

Un produit $(m, n) \times (n, p)$ coûte $m\,n\,p$ multiplications. Arrondis chaque facteur à une puissance de 10 ou à un petit multiple (60 000 ≈ $6 \times 10^4$, 784 ≈ $8 \times 10^2$, 128 ≈ $10^2$).

</details>
<details><summary>Indice 2</summary>

2 : temps = nombre de multiplications / vitesse. 3 : $\mathbf{A}\mathbf{B}$ est un produit $(1000, 1000) \times (1000, 1000)$ ; $\mathbf{C}\mathbf{v}$ est un produit $(1000, 1000) \times (1000, 1)$. Qu'est-ce qui reste une matrice, qu'est-ce qui devient un vecteur ?

</details>
<details><summary>Indice 3</summary>

1 : environ $6 \times 10^9$. 3 : de gauche à droite, deux produits matrice-matrice ($10^9$ chacun) et un matrice-vecteur ($10^6$) ; de droite à gauche, trois produits matrice-vecteur. 4 : le produit matriciel est associatif. 5 : la rétropropagation part de la loss, un seul nombre ; ce qu'elle fait passer d'une couche à l'autre joue le rôle du vecteur $\mathbf{v}$ tout à droite.

</details>

### Ex 0B.31 — Le gradient expliqué à un randonneur

<details><summary>Indice 1</summary>

Pars de ce qu'il **peut** faire : sentir la pente sous ses pieds, dans toutes les directions autour de lui.

</details>
<details><summary>Indice 2</summary>

Le gradient, c'est « la direction où ça monte le plus, et à quel point ça monte ». Descendre le gradient, c'est faire un pas dans la direction **opposée**, puis recommencer. La taille du pas est un choix.

</details>
<details><summary>Indice 3</summary>

Cinq lignes : 1) il tâte le sol autour de lui ; 2) il repère la direction de plus forte montée ; 3) il fait un pas dans l'autre sens ; 4) des pas trop grands le font passer par-dessus la vallée, des pas trop petits le font avancer très lentement ; 5) il peut s'arrêter dans une cuvette qui n'est pas la vallée.

</details>

### Ex 0B.32 — Écrire des maths en LaTeX dans Markdown

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

Parle de la loss comme d'une fonction des **poids** ; du learning rate ; du fait que le gradient est calculé par la rétropropagation, en pratique sur des mini-lots.

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

### 0B.E3 — Pourquoi manipuler des log-probabilités ?

<details><summary>Indice 1</summary>

Deux raisons : une **numérique** (que devient le produit de milliers de probabilités ?) et une **pratique** (que devient ce produit une fois passé au logarithme ?).

</details>
<details><summary>Indice 2</summary>

$0{,}01^{100} = 10^{-200}$, et un `float64` ne descend pas beaucoup plus bas que $10^{-308}$ (0B.15 h). $\ln$ transforme un produit en somme, et il est croissant : maximiser $\ln L$ revient à maximiser $L$.

</details>
<details><summary>Indice 3</summary>

Structure : les exemples sont supposés indépendants, donc la vraisemblance est un produit → sous-dépassement (*underflow*) ; le log le transforme en somme, stable et facile à dériver ; même maximum ; c'est l'origine de la cross-entropy (ch. 6), et PyTorch fournit `log_softmax` pour la même raison.

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

## Notebook

Les indices des exercices du notebook (0B.33 à 0B.54) arrivent avec le notebook, à la prochaine session de génération. La partie 0 du notebook (vérification des exercices ✏️) n'a pas besoin d'indices : ce sont ceux des exercices papier ci-dessus.
