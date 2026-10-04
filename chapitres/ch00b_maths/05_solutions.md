# 0B · Maths du lycée au ML — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses des exercices ✏️ se vérifient aussi dans la partie 0 du notebook ; les calculs de cette page y sont refaits en Python dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🧮 🗣️ 🛠️ Réflexion et outils](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à D](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 0B.Q1 — Puissances, racines et notation scientifique
1. **Faux** : on **additionne** les exposants, $2^3 \times 2^4 = 2^{7} = 128$ (on les multiplie seulement pour une puissance de puissance, $(2^3)^4 = 2^{12}$).
2. **Vrai** : $10^{-2} = \frac{1}{100}$.
3. **Faux** : $\sqrt{9 + 16} = \sqrt{25} = 5$, alors que $\sqrt{9} + \sqrt{16} = 7$. La racine d'un **produit** se sépare, pas celle d'une somme (c'est vrai seulement si $a$ ou $b$ vaut 0).
4. **Vrai** : $2^{20} = (2^{10})^2 \approx (10^3)^2 = 10^6$ ; exactement 1 048 576.
5. **Vrai** : `e7` se lit « fois $10^7$ ».

### 0B.Q2 — |x|, ⌊x⌋, ⌈x⌉ et sign(x)
1. **Faux** : $|-5| = 5$ ; une valeur absolue n'est jamais négative.
2. **Faux** : $\lfloor -1{,}5 \rfloor = -2$ (l'entier juste **en dessous** de −1,5 ; −1 est au-dessus).
3. **Vrai** : un entier est sa propre partie entière, par défaut comme par excès.
4. **Vrai** : on divise par −3, un nombre **négatif**, donc le sens de l'inégalité change : $x < \frac{12}{-3} = -4$.
5. **Vrai** : la distance de $x$ à 2 est plus petite que 1, donc $1 < x < 3$.

### 0B.Q3 — Lire une formule avec Σ, Π et une moyenne pondérée
1. **5 termes** : $x_3, x_4, x_5, x_6, x_7$ ($7 - 3 + 1$).
2. $1 \times 2 \times 3 = $ **6**.
3. **Vrai** : un facteur commun sort de la somme.
4. La **moyenne ordinaire** : $\frac{\sum_i x_i}{n}$, puisque la somme des poids vaut alors $n$.
5. Les $w_i$ sont les **poids** (l'importance de chaque entrée, appris pendant l'entraînement) ; $b$ est le **biais** (un décalage qui déplace le seuil d'activation, appris lui aussi).

### 0B.Q4 — Suite géométrique : elle fond ou elle explose ?
1. $q = 0{,}99$ : **fond** vers 0, lentement ($0{,}99^{100} \approx 0{,}37$ ; 0B.34).
2. $q = 1{,}01$ : **explose**, lentement ($1{,}01^{1000} \approx 21\,000$).
3. $q = -0{,}5$ : **fond** vers 0 en alternant les signes ($1, -0{,}5, 0{,}25, -0{,}125, \dots$).
4. $q = -1$ : **oscille** entre 1 et −1 ($1, -1, 1, -1, \dots$), sans fondre ni exploser.
5. $q = 1$ : **constant**.

**À retenir** : c'est $|q|$ qui décide (fond si $|q| < 1$, explose si $|q| > 1$, garde la même valeur absolue si $|q| = 1$) ; le signe de $q$ fait seulement alterner : $q = -2$ donne $1, -2, 4, -8, \dots$, qui explose en alternant. C'est le mécanisme des gradients qui s'évanouissent ou explosent dans un réseau profond (ch. 22).

### 0B.Q5 — Ensembles et dénombrement : le bon réflexe
1. L'**intersection** $A \cap B$.
2. Quand $A$ et $B$ sont **disjoints** ($A \cap B = \emptyset$) ; sinon on compte deux fois les éléments communs.
3. $4! = $ **24** (4 choix pour la première place, 3 pour la deuxième…).
4. $\binom{5}{2} = $ **10**.
5. **Non** : $\binom{n}{k}$ compte des sous-ensembles ; $\{a, b\}$ et $\{b, a\}$ sont le même choix.

### 0B.Q6 — Reconnaître l'allure d'une courbe
1. $\sigma(x)$ · 2. $\cos x$ · 3. $\ln x$ · 4. $x^2 - 4$ · 5. $\tanh(x)$. ($2x - 1$ et $e^x$ ne servent pas.)

### 0B.Q7 — Règles des logarithmes et des exponentielles
1. **Faux** : $e^{a+b} = e^a \times e^b$ (une somme dans l'exposant devient un produit).
2. **Vrai** : c'est la propriété qui fait tout l'intérêt du logarithme.
3. **Faux** : il n'y a aucune règle pour $\ln(a + b)$ ; par exemple $\ln(1 + 1) = \ln 2 \approx 0{,}69$, alors que $\ln 1 + \ln 1 = 0$.
4. **Faux** : `np.log` est le logarithme **népérien** : `np.log(100)` vaut environ 4,605. Le logarithme décimal est `np.log10` (et `np.log2` en base 2).
5. **Vrai** : $f(2) = 3$, puis $g(3) = 9$.

### 0B.Q8 — Vecteurs : norme, produit scalaire, cosinus, Hadamard
1. Le produit scalaire est un **nombre** ; le produit de Hadamard est un **vecteur** de même dimension.
2. $\sqrt{36 + 64} = $ **10**.
3. **Orthogonaux** (perpendiculaires).
4. **Il ne change pas** : le produit scalaire et la norme de $\mathbf{a}$ sont tous les deux multipliés par 10, qui se simplifie (pour un facteur négatif, le cosinus changerait de signe).
5. **1** (l'angle est nul).

### 0B.Q9 — Formes compatibles : ce produit existe-t-il ?
1. $\mathbf{A}\mathbf{B}$ : $(3, 4) \times (4, 2) \to$ **(3, 2)**.
2. $\mathbf{B}\mathbf{A}$ : $(4, 2) \times (3, 4)$, $2 \neq 3$ : **n'existe pas**.
3. $\mathbf{A}\mathbf{v}$ : un vecteur de dimension **3**.
4. $\mathbf{A}^\top\mathbf{A}$ : $(4, 3) \times (3, 4) \to$ **(4, 4)** (toujours défini, et toujours carré).
5. $\mathbf{B}^\top\mathbf{A}^\top$ : $(2, 4) \times (4, 3) \to$ **(2, 3)**, la forme de $(\mathbf{A}\mathbf{B})^\top$ : en fait $\mathbf{B}^\top\mathbf{A}^\top = (\mathbf{A}\mathbf{B})^\top$ (0B.44).

### 0B.Q10 — Dérivée : pente, signe, extremum
1. La **pente de la tangente** à la courbe au point d'abscisse $a$ (le taux de variation instantané).
2. $f$ est **décroissante** sur cet intervalle.
3. $3x^2$.
4. **Non** : $f(x) = x^3$ a une dérivée nulle en 0, mais elle monte avant et après : pas d'extremum (un « plat » en passant). Pour conclure à un extremum, on vérifie que $f'$ **change de signe** en $a$ (101.5.4).
5. $2e^{2x}$ (règle de la chaîne).

### 0B.Q11 — Gradient et lignes de niveau
1. **Faux** : c'est un **vecteur**, celui des dérivées partielles.
2. **Vrai.**
3. **Faux** : dans le sens **opposé** au gradient (descente de gradient : $\mathbf{x} - \eta\,\nabla f$).
4. **Vrai** : c'est la définition d'une ligne de niveau.
5. **Vrai.**

### 0B.Q12 — Probabilités : indépendance, espérance, variance
1. $P(A)\,P(B)$.
2. **Non** : $P(A \cap B) = 0$ alors que $P(A)\,P(B) > 0$. Ils sont même très dépendants : si l'un arrive, l'autre ne peut plus arriver.
3. **3,5**.
4. $3^2 \times 4 = $ **36** (le $+1$ ne change pas la dispersion).
5. Quand on répète une expérience un grand nombre de fois, la fréquence d'un événement se rapproche de sa probabilité (et la moyenne des résultats se rapproche de l'espérance).

<a id="rappels"></a>

## 🔁 Rappels

### 0B.R1 — Une somme en Python, trois façons

```python
import math

total = 0                                  # 1. boucle
for i in range(1, 6):
    total += i ** 2
print(total)                               # 55

print(sum(i ** 2 for i in range(1, 6)))    # 2. une ligne : 55
print(math.prod(range(1, 6)))              # 3. 1 × 2 × 3 × 4 × 5 = 120
```

4. `range(a, b)` s'arrête **avant** `b` : `range(1, 5)` donnerait 1, 2, 3, 4 et oublierait le terme $i = 5$ (on trouverait 30). La borne haute d'un $\Sigma$ est incluse, celle d'un `range` est exclue : c'est la source n° 1 des erreurs « à un près ».

### 0B.R2 — shape, ndim et axis d'un array
1. `X.shape == (333, 4)` et `X.ndim == 2` : la matrice $\mathbf{X}$ a **333 lignes** (une par manchot) et **4 colonnes** (une par mesure).
2. `X[:, 2]` est la **troisième colonne** (indice 2), c'est-à-dire la troisième mesure de tous les manchots (la longueur de la nageoire, dans l'ordre habituel des colonnes de 0A.35) ; forme `(333,)`.
3. `X.mean(axis=0)` a la forme `(4,)` : la moyenne de chaque mesure sur tous les manchots, c'est-à-dire le « manchot moyen ».
4. `X.T.shape == (4, 333)`.

### 0B.R3 — Une fonction qui prend une fonction

```python
from collections.abc import Callable

def compose(g: Callable[[float], float], f: Callable[[float], float]) -> Callable[[float], float]:
    """Return the function x -> g(f(x))."""
    return lambda x: g(f(x))

f = lambda x: 2 * x
g = lambda u: u + 3
print(compose(g, f)(5), compose(f, g)(5))   # 13 16
```

2. `compose(g, f)(5)` $= g(f(5)) = g(10) = $ **13** ; `compose(f, g)(5)` $= f(g(5)) = f(8) = $ **16** : l'ordre compte, $g \circ f \neq f \circ g$.
3. `Callable[[float], float]` : une fonction qui prend un `float` et renvoie un `float`.

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 0B.1 — Puissances, racines et notation scientifique sans calculatrice

| | Calcul | Réponse | Pourquoi |
|---|---|---|---|
| a | $2^5 \times 2^3$ | **256** | $2^{5+3} = 2^8$ |
| b | $\frac{(10^2)^3}{10^4}$ | **100** | $\frac{10^6}{10^4} = 10^{6-4} = 10^2$ |
| c | $5^{-2}$ | **0,04** | $\frac{1}{5^2} = \frac{1}{25} = \frac{4}{100}$ |
| d | $\sqrt{49 \times 16}$ | **28** | $\sqrt{49} \times \sqrt{16} = 7 \times 4$ |
| e | $8^{2/3}$ | **4** | $(2^3)^{2/3} = 2^{3 \times 2/3} = 2^2$ |
| f | $(3{,}2 \times 10^5) \times (2 \times 10^{-3})$ | **640** | $6{,}4 \times 10^{5 - 3} = 6{,}4 \times 10^2$ |
| g | $0{,}00056 = 5{,}6 \times 10^k$ | $k = $ **−4** | la virgule recule de 4 rangs |
| h | $2^{30} \approx 10^k$ | $k = $ **9** | $2^{30} = (2^{10})^3 \approx (10^3)^3$ ; exactement 1 073 741 824 |

**Erreurs fréquentes** : $2^{15}$ en a (multiplier les exposants) ; −25 en c (un exposant négatif donne un **inverse**, pas un nombre négatif) ; 4 en g (oublier que l'exposant est négatif pour un nombre plus petit que 1).
**Variante** : combien d'octets dans un gigaoctet « informatique » de $2^{30}$ octets, en ordre de grandeur ? ($10^9$ : c'est pour cela qu'on confond Go et Gio.)

### Ex 0B.2 — Valeur absolue, partie entière et signe

| | Calcul | Réponse | Pourquoi |
|---|---|---|---|
| a | $\lvert -7 \rvert + \lvert 3 - 5 \rvert$ | **9** | $7 + \lvert -2 \rvert = 7 + 2$ |
| b | $\lfloor 3{,}7 \rfloor$ | **3** | l'entier juste en dessous |
| c | $\lfloor -3{,}7 \rfloor$ | **−4** | vers le **bas** : −4 est en dessous de −3,7 |
| d | $\lceil -3{,}7 \rceil$ | **−3** | vers le **haut** |
| e | $\lceil 2{,}01 \rceil$ | **3** | le plus petit entier $\geq 2{,}01$ |
| f | $\mathrm{sign}(-0{,}5) + \mathrm{sign}(0) + \mathrm{sign}(12)$ | **0** | $-1 + 0 + 1$ |
| g, h | $\lvert x - 4 \rvert \leq 1{,}5$ | **2,5** et **5,5** | $-1{,}5 \leq x - 4 \leq 1{,}5$, soit $x \in [2{,}5 ; 5{,}5]$ |
| i | entiers relatifs tels que $\lvert x \rvert < 3$ | **5** | −2, −1, 0, 1, 2 (3 et −3 exclus : inégalité stricte) |

**Erreurs fréquentes** : 3 en i (oublier les entiers négatifs) ; −3 en c (confondre partie entière et troncature : `int(-3.7)` vaut −3, mais `math.floor(-3.7)` vaut −4) ; 7 en i (inclure ±3).
**Variante** : `math.floor(-3.7)`, `int(-3.7)` et `round(-3.7)` ? (−4, −3 et −4.)

### Ex 0B.3 — Lire et calculer des Σ et des Π

| | Calcul | Réponse | Pourquoi |
|---|---|---|---|
| a | $\sum_{i=1}^{5} i$ | **15** | $1 + 2 + 3 + 4 + 5$ |
| b | $\sum_{i=1}^{4} (2i - 1)$ | **16** | $1 + 3 + 5 + 7$ (la somme des $n$ premiers impairs vaut $n^2$) |
| c | $\sum_{k=0}^{3} 3^k$ | **40** | $1 + 3 + 9 + 27$ |
| d | $\prod_{i=1}^{4} (i + 1)$ | **120** | $2 \times 3 \times 4 \times 5$ |
| e | $\sum x_i^2$ | **30** | $4 + 1 + 16 + 9$ |
| f | $\sum (3x_i + 1)$ | **28** | $3 \sum_i x_i + 4 = 3 \times 8 + 4$ |
| g | $\sum_{i=1}^{100} 5$ | **500** | 100 termes égaux à 5 |
| h | $\sum_{i=2}^{4} i\,x_i$ | **22** | $2 \times (-1) + 3 \times 4 + 4 \times 3$ |

**Erreurs fréquentes** : 39 en c (oublier le terme $k = 0$) ; 25 en f (ajouter le 1 une seule fois au lieu de 4) ; 28 en e (écrire $(-1)^2 = -1$) ; 24 en h (commencer à $i = 1$).
**Variante** : vérifie b, e et h en Python avec `sum(...)` et une expression génératrice (0B.R1).

### Ex 0B.4 — Moyenne pondérée, somme pondérée et moyenne mobile

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | moyenne | **6** | $\frac{30}{5}$ |
| b | moyenne pondérée | **12** | $\frac{42 + 8 + 22}{3 + 1 + 2} = \frac{72}{6}$ |
| c | sortie du neurone | **2** | $2 - 2 + 3 + (-1)$ |
| d | moyennes mobiles | **[5 ; 4,67 ; 5 ; 4]** | $\frac{15}{3}$, $\frac{14}{3}$, $\frac{15}{3}$, $\frac{12}{3}$ |
| e | nombre de valeurs | **44** | de $t = 7$ à $t = 50$ : $50 - 7 + 1$ |

**Erreurs fréquentes** : 24 en b (diviser par le nombre de notes, 3, au lieu de la somme des coefficients, 6) ; 3 en c (oublier le biais) ; 43 en e (oublier le « + 1 »).
**Remarque** : la moyenne mobile lisse les variations rapides (la série oscille entre 1 et 9, les moyennes mobiles entre 4 et 5) ; c'est l'idée de la moyenne exponentielle utilisée par les optimiseurs Momentum et Adam (ch. 19).

### Ex 0B.5 — Ensembles

$A = \{2, 4, 6, 8, 10, 12\}$, $B = \{3, 6, 9, 12\}$, $C = \{1, 2, 3\}$.

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $\lvert A \cap B \rvert$ | **2** | $\{6, 12\}$ (les multiples de 6) |
| b | $\lvert A \cup B \rvert$ | **8** | $6 + 4 - 2$ ; en listant : $\{2, 3, 4, 6, 8, 9, 10, 12\}$ |
| c | $\lvert \bar{A} \rvert$ | **6** | les impairs |
| d | $\lvert \bar{A} \cap \bar{B} \rvert$ | **4** | $\{1, 5, 7, 11\}$ : c'est $\overline{A \cup B}$, de cardinal $12 - 8$ |
| e | $A \cap C$ | **{2}** | le seul élément pair de $C$ |
| f | $\lvert A \cup B \cup C \rvert$ | **9** | $A \cup B$ plus l'élément 1 (2 et 3 y sont déjà) |

**Erreurs fréquentes** : 10 en b (compter 6 et 12 deux fois) ; 8 en d (donner $\lvert A \cup B \rvert$ au lieu de son complémentaire) ; 10 en d (confondre « ni… ni… » avec « pas les deux à la fois », qui est le complémentaire de $A \cap B$) ; 11 en f (ajouter les 3 éléments de $C$, alors que 2 et 3 sont déjà dans $A \cup B$).
**Variante** : $\overline{A \cap B} = \bar{A} \cup \bar{B}$ (loi de De Morgan) ; vérifie-la en listant.

### Ex 0B.6 — Droites et paraboles

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | pente | **−2** | $\frac{-3 - 5}{3 - (-1)} = \frac{-8}{4}$ |
| b | ordonnée à l'origine | **3** | $5 = -2 \times (-1) + p$, donc $p = 3$ : $y = -2x + 3$ |
| c | racine | **1,5** | $-2x + 3 = 0$ |
| d | discriminant | **16** | $(-8)^2 - 4 \times 2 \times 6 = 64 - 48$ |
| e, f | racines | **1** et **3** | $\frac{8 \pm 4}{4}$ ; en effet $f(x) = 2(x - 1)(x - 3)$ |
| g | abscisse du sommet | **2** | $-\frac{b}{2a} = \frac{8}{4}$, exactement au milieu des racines |
| h | minimum | **−2** | $f(2) = 8 - 16 + 6$ |

**Erreurs fréquentes** : 2 en a (mélanger l'ordre des points en haut et en bas) ; −112 en d (écrire $-8^2 = -64$ au lieu de $(-8)^2 = 64$) ; −2 en g (oublier le signe de $b$).
**Vérification** : $(3, -3)$ est bien sur la droite : $-2 \times 3 + 3 = -3$.

### Ex 0B.7 — Cosinus : cercle, période et planning en cosinus

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | 60° en radians | **1,05** | $\frac{\pi}{3} \approx \frac{3{,}1416}{3}$ |
| b | $\cos \pi$ | **−1** | le point à l'opposé de $(1, 0)$ sur le cercle |
| c | $\cos\frac{2\pi}{3}$ | **−0,5** | symétrique de $\frac{\pi}{3}$ par rapport à l'axe vertical : $-\cos\frac{\pi}{3}$ |
| d | $\cos(-\frac{\pi}{3})$ | **0,5** | le cosinus est pair |
| e | $\cos(4\pi)$ | **1** | deux tours complets : $\cos 0$ |
| f | facteur en $t = 50$ | **0,5** | $\frac{1}{2}(1 + \cos\frac{\pi}{2}) = \frac{1}{2}(1 + 0)$ |
| g | facteur en $t = 25$ | **0,85** | $\frac{1}{2}(1 + \cos\frac{\pi}{4}) = \frac{1}{2}(1 + 0{,}707) \approx 0{,}854$ |
| h | learning rate en $t = 75$ | **0,0015** | $0{,}01 \times \frac{1}{2}(1 - 0{,}707) \approx 0{,}01 \times 0{,}146$ |

**Erreurs fréquentes** : 0,33 en a (calculer $\frac{60}{180}$ sans le $\pi$) ; −0,5 en d (croire que le cosinus est impair) ; 0,146 en h (oublier de multiplier par le learning rate maximal) ; 0,0085 en h (prendre $\cos\frac{3\pi}{4} = +0{,}707$ : il est négatif, car $\cos(\pi - x) = -\cos x$).
**Remarque** : le planning en cosinus part du learning rate maximal ($t = 0$, facteur 1), passe à la moitié à mi-parcours et termine en douceur vers 0 ($t = T$). On le retrouvera au ch. 19 (`CosineAnnealingLR` dans PyTorch).

### Ex 0B.8 — Vecteurs : somme, multiple, norme et distance

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $\mathbf{p}_1 + \mathbf{p}_2$ | **(88 ; 386)** | composante par composante |
| b | manchot moyen | **(44 ; 193)** | la moitié de a |
| c | $\lVert \mathbf{u} \rVert_2$ | **5** | $\sqrt{9 + 16}$ |
| d | $\lVert \mathbf{u} \rVert_1$ | **7** | $3 + 4$ |
| e | $\lVert \mathbf{u} \rVert_\infty$ | **4** | $\max(3, 4)$ |
| f | distance | **10** | $\mathbf{p}_1 - \mathbf{p}_2 = (-8 ; -6)$ ; $\sqrt{64 + 36}$ |
| g | distance, nageoire en cm | **8,02** | $\sqrt{64 + 0{,}36}$ |
| h | vecteur unitaire | **(0,6 ; −0,8)** | $\frac{\mathbf{u}}{5}$ |

**Qu'est-ce qui a changé en g ?** Le même écart de nageoire (6 mm = 0,6 cm) pèse 100 fois moins dans la somme des carrés : la distance ne « voit » presque plus que le bec. Changer d'unité change donc les plus proches voisins d'un manchot : c'est pourquoi on **standardise** les features (0A.52, ch. 12) avant un k plus proches voisins (ch. 13) ou une descente de gradient.
**Erreurs fréquentes** : 7 en c (calculer la norme L1, la somme des valeurs absolues, au lieu de la norme L2) ; −1 en d (oublier les valeurs absolues) ; 3 en e (prendre le maximum sans valeur absolue).

### Ex 0B.9 — Transposée et produit matrice-vecteur

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | forme de $\mathbf{A}^\top$ | **(3, 2)** | $\mathbf{A}$ est $(2, 3)$ |
| b | $(\mathbf{A}^\top)_{31}$ | **1** | $= \mathbf{A}_{13}$ (ligne 1, colonne 3 de $\mathbf{A}$) |
| c | $\mathbf{A}\mathbf{v}$ | **(1 ; 3)** | lignes : $(2 + 0 - 1,\; -1 + 6 - 2)$ |
| d | $\mathbf{A}^\top\mathbf{t}$ | **(3 ; −3 ; −1)** | $(2 + 1,\; 0 - 3,\; 1 - 2)$ |
| e | $\mathbf{A}\mathbf{s}$ défini ? | **False** | $\mathbf{A}$ a 3 colonnes, $\mathbf{s}$ a 2 composantes |
| f | $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$ | **(5,5 ; 2,5 ; −1)** | $\mathbf{X}\mathbf{w} = (0{,}5 + 4 ;\; 1{,}5 + 0 ;\; 0 - 2)$, plus 1 |

**Lecture par colonnes de c** : $1 \times \begin{pmatrix} 2 \\ -1 \end{pmatrix} + 2 \times \begin{pmatrix} 0 \\ 3 \end{pmatrix} + (-1) \times \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 2 + 0 - 1 \\ -1 + 6 - 2 \end{pmatrix} = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$ : même résultat. La lecture par lignes dit « chaque sortie est un produit scalaire » ; la lecture par colonnes dit « le résultat est un mélange des colonnes, dosé par $\mathbf{v}$ ».
**En ML** : f est exactement la prédiction d'une régression linéaire sur un lot de 3 exemples, en **une** opération (ch. 9).
**Erreur fréquente** : (2, 3) en a ; en b, chercher $\mathbf{A}_{31}$, qui n'existe pas ($\mathbf{A}$ n'a que 2 lignes) : $(\mathbf{A}^\top)_{31} = \mathbf{A}_{13}$.

### Ex 0B.10 — Taux d'accroissement : de la sécante à la tangente

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $h = 1$ | **6** | $\frac{f(3) - f(2)}{1} = 12 - 6$ |
| b | $h = 0{,}1$ | **5,1** | $\frac{6{,}51 - 6}{0{,}1}$ |
| c | $h = 0{,}01$ | **5,01** | $\frac{6{,}0501 - 6}{0{,}01}$ |
| d | $f'(2)$ | **5** | $\frac{(2 + h)^2 + (2 + h) - 6}{h} = \frac{5h + h^2}{h} = 5 + h \to 5$ |
| e | ordonnée à l'origine de la tangente | **−4** | $y = 6 + 5(x - 2) = 5x - 4$ |
| f | approximation de $f(2{,}05)$ | **6,25** (exactement) | $6 + 5 \times 0{,}05$ |

**Comparaison (f)** : la vraie valeur est $f(2{,}05) = 4{,}2025 + 2{,}05 = 6{,}2525$ (c'est l'erreur fréquente : donner la vraie valeur au lieu de l'approximation) ; l'erreur vaut $0{,}0025 = 0{,}05^2$, exactement le terme $h^2$ oublié. Plus $h$ est petit, plus la tangente colle à la courbe : c'est ce qui rend la descente de gradient fiable avec de **petits** pas.
**Variante** : avec la formule $(x^2 + x)' = 2x + 1$, retrouve $f'(2) = 5$.

### Ex 0B.11 — Probabilités : issues, complémentaire, union

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | somme = 8 | **0,139** | $(2, 6), (3, 5), (4, 4), (5, 3), (6, 2)$ : $\frac{5}{36}$ |
| b | double | **0,167** | $\frac{6}{36}$ |
| c | au moins un 1 | **0,306** | $1 - \frac{5 \times 5}{36} = \frac{11}{36}$ |
| d | somme 8 ou double | **0,278** | $\frac{5 + 6 - 1}{36} = \frac{10}{36}$ ($(4, 4)$ est dans les deux) |
| e | somme $\geq 10$ | **0,167** | sommes 10, 11, 12 : $3 + 2 + 1 = 6$ issues |
| f | cœur ou figure | **0,423** | $\frac{13 + 12 - 3}{52} = \frac{22}{52}$ |

**Erreurs fréquentes** : 0,333 en c ($\frac{1}{6} + \frac{1}{6}$ compte $(1, 1)$ deux fois) ; 0,306 en d (compter $(4, 4)$ deux fois) ; 0,481 en f (compter deux fois les 3 figures de cœur).
**Variante** : $P(\text{somme} = 7)$ ? ($\frac{6}{36}$ : c'est la somme la plus probable.)

### Deuxième passage

### Ex 0B.12 — Dénombrer

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | codes à 4 chiffres | **10 000** | $10^4$ (répétitions permises, ordre important) |
| b | tous différents | **5 040** | $10 \times 9 \times 8 \times 7$ |
| c | $6!$ | **720** | $6 \times 5 \times 4 \times 3 \times 2 \times 1$ |
| d | $\binom{6}{2}$ | **15** | $\frac{6 \times 5}{2}$ |
| e | $\binom{8}{3}$ | **56** | $\frac{8 \times 7 \times 6}{6}$ |
| f | un-contre-un, 5 classes | **10** | une paire de classes = $\binom{5}{2}$ |
| g | sous-ensembles de 5 features | **32** | chaque feature est dedans ou non : $2^5$ |
| h | comité de 3 avec un président | **360** | 10 choix de président, puis $\binom{9}{2} = 36$ pour les deux autres |

**Erreurs fréquentes** : 720 en b (s'arrêter à trois facteurs) ; 30 en d (compter les paires ordonnées) ; 30 en g (oublier l'ensemble vide et l'ensemble complet) ; 120 en h (oublier le rôle de président).
**Autre méthode pour h** : $\binom{10}{3} = 120$ comités, puis 3 choix de président dans chacun : $120 \times 3 = 360$. Deux raisonnements, un seul résultat : c'est la meilleure vérification.
**En ML** : g explique pourquoi on ne teste pas tous les sous-ensembles de features : avec 30 features, $2^{30} \approx 10^9$ modèles.

### Ex 0B.13 — Suites géométriques

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $u_5$ | **31,25** | $1000 \times 0{,}5^5 = \frac{1000}{32}$ |
| b | $0{,}9^{20}$ | **0,122** | il reste 12 % du signal après 20 couches |
| c | $1{,}1^{10}$ | **2,59** | +10 % par étape, 10 fois : le double, et plus |
| d | $\sum_{k=0}^{9} 0{,}5^k$ | **1,998** | $\frac{1 - 0{,}5^{10}}{1 - 0{,}5} = 2 \times \frac{1023}{1024}$ |
| e | $\sum_{k=0}^{\infty} 0{,}95^k$ | **20** | $\frac{1}{1 - 0{,}95} = \frac{1}{0{,}05}$ |
| f | plus petit $k$ avec $0{,}5^k < 0{,}001$ | **10** | $0{,}5^9 = \frac{1}{512} \approx 0{,}002$ ; $0{,}5^{10} = \frac{1}{1024} \approx 0{,}00098$ |
| g | $(-0{,}8)^7$ | **−0,210** | négatif : un nombre négatif à une puissance **impaire** |

**Erreurs fréquentes** : 62,5 en a (s'arrêter à $q^4$) ; 2,00 en c (croire que 10 hausses de 10 % font +100 %) ; 1,996 en d (prendre $n = 9$ au lieu de 10 termes).
**En ML** : b et c sont les gradients qui **s'évanouissent** et qui **explosent** quand on les multiplie couche après couche (ch. 22) ; e est la « valeur » d'une suite de récompenses en apprentissage par renforcement, avec le facteur d'actualisation $\gamma$ (ch. 26).

### Ex 0B.14 — La somme géométrique démontrée pas à pas ∂

1. $q\,S_n = q + q^2 + q^3 + \dots + q^{n}$ (chaque terme de $S_n$ multiplié par $q$ : les exposants avancent d'un cran).
2. On soustrait, en alignant les termes identiques :

$$S_n - q\,S_n = (1 + q + \dots + q^{n-1}) - (q + q^2 + \dots + q^{n}) = 1 - q^{n}$$

Tous les termes de $q$ à $q^{n-1}$ apparaissent une fois avec $+$ et une fois avec $-$ : il ne reste que le premier terme de $S_n$ et le dernier de $q\,S_n$. En factorisant, $S_n(1 - q) = 1 - q^n$, et comme $q \neq 1$ on peut diviser par $1 - q$ :

$$S_n = \frac{1 - q^n}{1 - q}$$

3. $q = 2$, $n = 4$ : directement, $1 + 2 + 4 + 8 = 15$ ; avec la formule, $\frac{1 - 16}{1 - 2} = \frac{-15}{-1} = 15$. ✔️
4. Si $|q| < 1$, $q^n \to 0$ quand $n \to \infty$ (101.1.5), donc $S_n \to \frac{1 - 0}{1 - q} = \frac{1}{1 - q}$.
5. $0{,}999\ldots = 0{,}9 \sum_{k=0}^{\infty} 0{,}1^k = 0{,}9 \times \frac{1}{1 - 0{,}1} = \frac{0{,}9}{0{,}9} = $ **1**. Ce n'est pas « presque 1 » : c'est exactement 1, écrit autrement.

**Critères** : le décalage des termes est explicite (1) ; la simplification est justifiée (2) ; la division par $1 - q$ est justifiée par $q \neq 1$ ; le passage à la limite utilise $|q| < 1$.
**Variante** : que vaut $S_n$ quand $q = 1$ ? ($n$ : tous les termes valent 1, et la formule, qui divise par 0, ne s'applique pas.)

### Ex 0B.15 — Exponentielles et logarithmes

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $\ln(e^3)$ | **3** | $\ln$ défait $\exp$ |
| b | $e^{\ln 5}$ | **5** | $\exp$ défait $\ln$ |
| c | $\ln 1 + \log_2 32$ | **5** | $0 + 5$, car $2^5 = 32$ |
| d | $\log_{10} 0{,}001$ | **−3** | $0{,}001 = 10^{-3}$ |
| e | $\ln(e^2 \times e^5)$ | **7** | $\ln(e^7)$ ; ou $\ln e^2 + \ln e^5 = 2 + 5$ |
| f | $e^x = 20$ | **3,00** | $x = \ln 20 \approx 2{,}996$ |
| g | $2^x = 1000$ | **9,97** | $x = \log_2 1000 = \frac{\ln 1000}{\ln 2} = \frac{6{,}908}{0{,}693}$ |
| h | $\ln(0{,}01^{100})$ | **−460,5** | $100 \times \ln 0{,}01 = 100 \times (-4{,}605)$ |
| i | 1 nat en bits | **1,443** | $\frac{1}{\ln 2}$ (0B.16) |

**Bits (g)** : il faut **10 bits** pour numéroter 1000 objets, car $2^9 = 512 < 1000 \leq 2^{10} = 1024$ : on arrondit $\log_2 1000$ à l'entier **supérieur**.
**Pourquoi h compte** : $0{,}01^{100} = 10^{-200}$ est encore représentable en `float64` (qui descend jusqu'à environ $10^{-308}$, et même $5 \times 10^{-324}$ en perdant des chiffres), mais $0{,}01^{200}$ ne l'est plus : Python l'arrondit à 0. Son logarithme, −921, ne pose, lui, aucun problème. C'est pour cela qu'on additionne des log-probabilités au lieu de multiplier des probabilités (0B.E3, ch. 6).
**Erreurs fréquentes** : 6 en c ($\ln 1 = 0$, pas 1) ; 10 en e (multiplier les exposants) ; 3 en g (confondre $\log_2$ et $\log_{10}$) ; 0,693 en i (inverser la conversion).

### Ex 0B.16 — Changer de base ∂

1. $y = \log_2 x$ veut dire $2^y = x$. On prend le logarithme népérien des deux membres : $\ln(2^y) = \ln x$, soit $y \ln 2 = \ln x$ (l'exposant « descend »). Comme $\ln 2 \neq 0$ :

$$\log_2 x = y = \frac{\ln x}{\ln 2}$$

2. Même calcul avec $b$ : $b^y = x \Rightarrow y \ln b = \ln x \Rightarrow \log_b x = \frac{\ln x}{\ln b}$. On divise par $\ln b$, non nul **parce que** $b \neq 1$ (et $b > 0$ pour que $b^y$ ait un sens pour tout $y$).
3. $\log_2 10 \times \log_{10} x = \frac{\ln 10}{\ln 2} \times \frac{\ln x}{\ln 10} = \frac{\ln x}{\ln 2} = \log_2 x$. ✔️ Changer de base, c'est multiplier par une constante.
4. En bits : $\log_2\frac{1}{p} = \frac{\ln(1/p)}{\ln 2} = \frac{I}{\ln 2}$, et $\frac{1}{\ln 2} \approx \frac{1}{0{,}6931} \approx 1{,}443$.
5. $p = \frac{1}{8}$ : $\log_2 8 = $ **3 bits** (trois questions oui/non suffisent pour trouver une issue parmi 8) ; en nats, $\ln 8 = 3 \ln 2 \approx $ **2,08 nats** (et $2{,}08 \times 1{,}443 \approx 3$ ✔️).

**Critères** : l'étape « on prend le $\ln$ des deux membres » est écrite ; la condition $b \neq 1$ est justifiée.
**À retenir** : bits et nats, c'est la même information dans deux unités, comme des centimètres et des pouces. PyTorch calcule la cross-entropy en nats (avec $\ln$) ; la théorie de l'information parle souvent en bits (ch. 6).

### Ex 0B.17 — Sigmoïde et tanh

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $\sigma(0)$ | **0,5** | $\frac{1}{1 + 1}$ |
| b | $\sigma(3)$ | **0,953** | $\frac{1}{1 + 0{,}0498}$ |
| c | $\sigma(-3)$ | **0,047** | $1 - 0{,}953$ |
| d | $\tanh(0{,}5)$ | **0,462** | $\frac{1{,}6487 - 0{,}6065}{1{,}6487 + 0{,}6065} = \frac{1{,}0422}{2{,}2552}$ |
| e | $2\sigma(1) - 1$ | **0,462** | $2 \times 0{,}7311 - 1$ : la même valeur que d |
| f | limite en $-\infty$ | **0** | $e^{-x} \to +\infty$, donc $\frac{1}{1 + e^{-x}} \to 0$ |
| g | $\sigma(x) = 0{,}9$ | **2,20** | $1 + e^{-x} = \frac{1}{0{,}9} = \frac{10}{9}$, $e^{-x} = \frac{1}{9}$, $x = \ln 9 \approx 2{,}197$ |
| h | $\sigma(x) + \sigma(-x)$ | **1** | la symétrie de c |

**Pourquoi d = e** : $\tanh(x) = 2\sigma(2x) - 1$, ici avec $x = 0{,}5$. La tangente hyperbolique est une sigmoïde étirée de $[0, 1]$ vers $[-1, 1]$.
**Preuve de h** (pour les curieux) : $\sigma(-x) = \frac{1}{1 + e^{x}} = \frac{e^{-x}}{e^{-x} + 1}$ (on multiplie en haut et en bas par $e^{-x}$), donc $\sigma(x) + \sigma(-x) = \frac{1 + e^{-x}}{1 + e^{-x}} = 1$.
**En ML** : g est la **fonction logit**, l'inverse de la sigmoïde : un classifieur qui sort une probabilité 0,9 a calculé en interne un score $\ln 9 \approx 2{,}2$ (ch. 13 et 17).
**Erreur fréquente** : −0,953 en c (croire que $\sigma$ est impaire, comme tanh) ; −2,20 en g (erreur de signe en isolant $x$).

### Ex 0B.18 — Produit scalaire, similarité cosinus et produit de Hadamard

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $\mathbf{a} \cdot \mathbf{b}$ | **0** | $2 + 0 - 2$ |
| b | orthogonaux ? | **True** | produit scalaire nul |
| c | $\lVert \mathbf{a} \rVert$ | **3** | $\sqrt{1 + 4 + 4}$ |
| d | $\mathbf{a} \cdot \mathbf{c}$ | **18** | $2 + 8 + 8$ |
| e | $\cos(\mathbf{a}, \mathbf{c})$ | **1** | $\mathbf{c} = 2\mathbf{a}$ : même direction ; $\frac{18}{3 \times 6}$ |
| f | $\cos(\mathbf{a}, \mathbf{d})$ | **0,333** | $\frac{1}{3 \times 1}$ |
| g | $\mathbf{a} \odot \mathbf{b}$ | **(2 ; 0 ; −2)** | produits composante par composante |
| h | somme des composantes de $\mathbf{a} \odot \mathbf{c}$ | **18** | $2 + 8 + 8 = \mathbf{a} \cdot \mathbf{c}$ |
| i | $\cos(\mathbf{a}, -\mathbf{a})$ | **−1** | directions opposées |

**À retenir** : le produit scalaire, c'est « Hadamard, puis somme » (h) ; en NumPy, `(a * c).sum()` et `a @ c` donnent le même nombre.
**Erreur fréquente** : 1 en f (oublier de diviser par $\lVert \mathbf{a} \rVert = 3$).
**En ML** : e montre que la similarité cosinus ignore la longueur : un document et le même document copié deux fois ont une similarité de 1 (0B.41, 0B.E2).

### Ex 0B.19 — Développer ‖a − b‖² avec le produit scalaire ∂

1. Le produit scalaire se développe comme un produit ordinaire et il est symétrique ($\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}$) :

$$\lVert \mathbf{a} - \mathbf{b} \rVert^2 = (\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b}) = \mathbf{a} \cdot \mathbf{a} - \mathbf{a} \cdot \mathbf{b} - \mathbf{b} \cdot \mathbf{a} + \mathbf{b} \cdot \mathbf{b} = \lVert \mathbf{a} \rVert^2 + \lVert \mathbf{b} \rVert^2 - 2\,\mathbf{a} \cdot \mathbf{b}$$

C'est l'identité remarquable $(a - b)^2 = a^2 + b^2 - 2ab$, version vecteurs.
2. $\mathbf{a} - \mathbf{b} = (-1, 2, 3)$, donc $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = 1 + 4 + 9 = 14$. À droite : $\lVert \mathbf{a} \rVert^2 = 9$, $\lVert \mathbf{b} \rVert^2 = 4 + 0 + 1 = 5$ et $\mathbf{a} \cdot \mathbf{b} = 0$ : $9 + 5 - 0 = 14$. ✔️
3. Si $\lVert \mathbf{a} \rVert = \lVert \mathbf{b} \rVert = 1$, alors $\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{1 \times 1} = \mathbf{a} \cdot \mathbf{b}$ et $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = 1 + 1 - 2\cos(\mathbf{a}, \mathbf{b})$. La distance est une fonction **décroissante** du cosinus : le vecteur le plus proche est celui qui a la plus grande similarité cosinus. C'est pourquoi les bases de données vectorielles normalisent souvent les embeddings.
4. **Pythagore** : si $\mathbf{a} \cdot \mathbf{b} = 0$, $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = \lVert \mathbf{a} \rVert^2 + \lVert \mathbf{b} \rVert^2$ (le triangle formé par $\mathbf{a}$, $\mathbf{b}$ et $\mathbf{a} - \mathbf{b}$ est rectangle). Ici $14 = 9 + 5$.

**Critères** : les quatre termes du développement sont écrits ; la symétrie est invoquée pour regrouper $-\mathbf{a} \cdot \mathbf{b} - \mathbf{b} \cdot \mathbf{a}$.

### Ex 0B.20 — Produit matriciel

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | forme de $\mathbf{A}\mathbf{B}$ | **(3, 3)** | $(3, 2) \times (2, 3)$ |
| b | forme de $\mathbf{B}\mathbf{A}$ | **(2, 2)** | $(2, 3) \times (3, 2)$ |
| c | $\mathbf{B}\mathbf{A}$ | $\begin{pmatrix} 2 & 3 \\ 13 & 7 \end{pmatrix}$ | voir le détail ci-dessous |
| d | $(\mathbf{A}\mathbf{B})_{11}$ | **4** | $(1, 2) \cdot (2, 1)$ |
| e | $(\mathbf{A}\mathbf{B})_{32}$ | **2** | ligne 3 de $\mathbf{A}$, colonne 2 de $\mathbf{B}$ : $(3, 1) \cdot (1, -1)$ |
| f | $\mathbf{C}\mathbf{A}$ défini ? | **False** | $(2, 2) \times (3, 2)$ : $2 \neq 3$ |
| g | $\mathbf{A}\mathbf{C}$ | $\begin{pmatrix} 5 & 1 \\ -2 & 0 \\ 5 & 3 \end{pmatrix}$ | $(1, 2) \cdot (1, 2) = 5$, $(1, 2) \cdot (1, 0) = 1$, etc. |
| h | multiplications pour $\mathbf{A}\mathbf{B}$ | **18** | 9 éléments × 2 multiplications ($m\,n\,p = 3 \times 2 \times 3$) |

**Détail de c** : $(2, 1, 0) \cdot (1, 0, 3) = 2$ ; $(2, 1, 0) \cdot (2, -1, 1) = 3$ ; $(1, -1, 4) \cdot (1, 0, 3) = 13$ ; $(1, -1, 4) \cdot (2, -1, 1) = 7$.
**À retenir** : $\mathbf{A}\mathbf{B}$ est $3 \times 3$ et $\mathbf{B}\mathbf{A}$ est $2 \times 2$ : même quand les deux produits existent, ils n'ont pas la même forme. Le produit matriciel n'est **pas commutatif** (0B.44).
**Erreurs fréquentes** : (3, 3) en b (garder les dimensions intérieures) ; −4 en e (lire la ligne 2 et la colonne 3 au lieu de la ligne 3 et la colonne 2) ; 9 en h (compter les éléments au lieu des multiplications).

### Ex 0B.21 — Identité, inverse 2 × 2 et système

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $\det \mathbf{M}$ | **2** | $3 \times 2 - 1 \times 4$ |
| b | $\mathbf{M}^{-1}$ | $\begin{pmatrix} 1 & -0{,}5 \\ -2 & 1{,}5 \end{pmatrix}$ | $\frac{1}{2}\begin{pmatrix} 2 & -1 \\ -4 & 3 \end{pmatrix}$ |
| c | solution | **(2 ; −1)** | $\mathbf{M}^{-1}\begin{pmatrix} 5 \\ 6 \end{pmatrix} = (5 - 3 ;\; -10 + 9)$ |
| d | inversible ? | **False** | $\det = 4 - 4 = 0$ (la ligne 1 est le double de la ligne 2) |
| e | $k$ | **3** | $\det = 6 - 2k = 0$ |
| f | inverse de la diagonale | $\begin{pmatrix} 0{,}5 & 0 \\ 0 & 0{,}2 \end{pmatrix}$ | on inverse chaque élément diagonal |

**Vérification de b** : $\mathbf{M}\mathbf{M}^{-1} = \begin{pmatrix} 3 - 2 & -1{,}5 + 1{,}5 \\ 4 - 4 & -2 + 3 \end{pmatrix} = \mathbf{I}$. **Vérification de c** : $3 \times 2 - 1 = 5$ et $4 \times 2 - 2 = 6$. ✔️
**Erreurs fréquentes** : $\begin{pmatrix} 2 & -1 \\ -4 & 3 \end{pmatrix}$ en b (oublier de diviser par le déterminant) ; échanger les signes au lieu des éléments diagonaux.
**En pratique** : en code, on ne calcule presque jamais l'inverse ; pour résoudre $\mathbf{M}\mathbf{x} = \mathbf{y}$, on appelle `np.linalg.solve(M, y)`, plus rapide et plus précis (0B.46).

### Ex 0B.22 — Dériver avec les règles

| | Fonction | Dérivée | Valeur |
|---|---|---|---|
| a | $4x^3 - 2x + 7$ | $12x^2 - 2$ | **10** |
| b | $\sqrt{x} + \frac{1}{x}$ | $\frac{1}{2\sqrt{x}} - \frac{1}{x^2}$ | $\frac{1}{4} - \frac{1}{16} = $ **0,1875** |
| c | $x^2 e^x$ | $2x\,e^x + x^2 e^x = (x^2 + 2x)\,e^x$ | $3e \approx $ **8,155** |
| d | $x \ln x$ | $1 \times \ln x + x \times \frac{1}{x} = \ln x + 1$ | **2** |
| e | $\frac{x + 1}{x - 1}$ | $\frac{1 \times (x - 1) - (x + 1) \times 1}{(x - 1)^2} = \frac{-2}{(x - 1)^2}$ | **−0,5** |
| f | $\frac{e^x}{x}$ | $\frac{e^x \times x - e^x \times 1}{x^2} = \frac{e^x (x - 1)}{x^2}$ | **0** |
| g | $5x^4 - 3$ | $20x^3$ | **160** |

**Erreurs fréquentes** : $+\frac{1}{x^2}$ en b ; $2x\,e^x$ seul en c (la dérivée d'un produit n'est **pas** le produit des dérivées) ; $+0{,}5$ en e (inverser l'ordre des termes du numérateur) ; oublier que la constante disparaît (a, g).
**Remarque (f)** : $r'(1) = 0$ et $r'$ change de signe en 1 (négative avant, positive après) : $\frac{e^x}{x}$ a un minimum en $x = 1$ sur $]0, +\infty[$ (101.5.4).

### Ex 0B.23 — Règle de la chaîne

| | Fonction | Décomposition | Dérivée | Valeur |
|---|---|---|---|---|
| a | $(3x - 1)^4$ | $u = 3x - 1$, $u^4$ | $4u^3 \times 3 = 12(3x - 1)^3$ | **96** |
| b | $e^{2x + 1}$ | $u = 2x + 1$, $e^u$ | $2e^{2x + 1}$ | $2e \approx $ **5,437** |
| c | $\ln(x^2 + 1)$ | $u = x^2 + 1$, $\ln u$ | $\frac{2x}{x^2 + 1}$ | $\frac{4}{5} = $ **0,8** |
| d | $\sqrt{1 + 4x}$ | $u = 1 + 4x$, $\sqrt{u}$ | $\frac{4}{2\sqrt{1 + 4x}} = \frac{2}{\sqrt{1 + 4x}}$ | $\frac{2}{3} \approx $ **0,667** |
| e | $e^{-x^2/2}$ | $u = -\frac{x^2}{2}$, $e^u$ | $-x\,e^{-x^2/2}$ | $-e^{-0{,}5} \approx $ **−0,607** |
| f | $\frac{1}{x^2 + 1}$ | $u = x^2 + 1$, $\frac{1}{u}$ | $-\frac{2x}{(x^2 + 1)^2}$ | **−0,5** |
| g | $(3w + 1 - 4)^2$ | $u = 3w - 3$, $u^2$ | $2(3w - 3) \times 3$ | **18** |
| h | $(\ln x)^2$ | $u = \ln x$, $u^2$ | $\frac{2 \ln x}{x}$ | $\frac{2}{e} \approx $ **0,736** |

**Erreur fréquente** : oublier la dérivée **intérieure** (le « × 3 » de a, le « × 2 » de b, le « × 4 » de d) : 32 en a, 2,718 en b, 0,167 en d, 6 en g.
**En ML** : g est la dérivée d'une erreur quadratique pour un seul exemple ($x = 3$, $y = 4$, $\hat{y} = 3w + 1$) : $\frac{dL}{dw} = 2(\hat{y} - y) \times x$. En $w = 2$, la prédiction (7) est trop grande, la dérivée est positive, et la descente de gradient fera baisser $w$. C'est la formule de 0B.29 B pour un seul exemple.

### Ex 0B.24 — La moyenne minimise la somme des carrés des écarts ∂

1. Pour chaque terme, règle de la chaîne avec $u = y_i - a$ (et $\frac{du}{da} = -1$) : $\frac{d}{da}(y_i - a)^2 = 2(y_i - a) \times (-1)$. Donc

$$S'(a) = -2\sum_{i=1}^{n} (y_i - a)$$

2. $S'(a) = 0 \iff \sum_i (y_i - a) = 0 \iff \sum_i y_i - n\,a = 0 \iff a = \frac{1}{n}\sum_i y_i = \bar{y}$ (la somme de $n$ termes égaux à $a$ vaut $n\,a$).
3. $S'(a) = -2\left(\sum_i y_i - n\,a\right) = 2n(a - \bar{y})$ : négative pour $a < \bar{y}$ ($S$ décroît), positive pour $a > \bar{y}$ ($S$ croît). $S$ atteint donc son **minimum** en $\bar{y}$. Autre argument : en développant, $S(a) = n\,a^2 - 2a\sum_i y_i + \sum_i y_i^2$ est une parabole en $a$, tournée vers le haut car $n > 0$ ; son sommet est en $\frac{2\sum_i y_i}{2n} = \bar{y}$.
4. $\bar{y} = \frac{2 + 3 + 7}{3} = 4$. $S(4) = 4 + 1 + 9 = $ **14** ; $S(3) = 1 + 0 + 16 = $ **17** ; $S(5) = 9 + 4 + 4 = $ **17**. Le minimum est bien en 4, et $S$ est symétrique autour de 4 (une parabole).
5. L'erreur quadratique moyenne de la baseline vaut $\frac{S(\bar{y})}{n} = \frac{1}{n}\sum_i (y_i - \bar{y})^2$ : c'est exactement la **variance** des $y_i$ (ici $\frac{14}{3} \approx 4{,}67$). Un modèle de régression n'est utile que s'il fait mieux ; le score $R^2 = 1 - \frac{\text{MSE du modèle}}{\text{variance}}$ mesure justement de combien (ch. 9).

**Critères** : dérivée intérieure −1 présente ; la résolution utilise $\sum_i a = n\,a$ ; le minimum est justifié (signe ou parabole), pas seulement la dérivée nulle.
**Variante** : quel nombre minimise $\sum_i |y_i - a|$ ? (la **médiane** : ici 3 ; c'est la baseline de l'erreur absolue moyenne, MAE.)

### Ex 0B.25 — Lignes de niveau, dérivées partielles, gradient et un pas de descente

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $f(3, 1)$ | **6** | $4 + 2$ |
| b | $\frac{\partial f}{\partial x}(3, 1)$ | **4** | $2(x - 1)$ |
| c | $\frac{\partial f}{\partial y}(3, 1)$ | **4** | $4y$ |
| d | $\nabla f(3, 1)$ | **(4 ; 4)** | |
| e | sa norme | **5,66** | $\sqrt{32} = 4\sqrt{2}$ |
| f | nouveau point | **(2,6 ; 0,6)** | $(3, 1) - 0{,}1 \times (4, 4)$ |
| g | $f$ au nouveau point | **3,28** | $1{,}6^2 + 2 \times 0{,}36 = 2{,}56 + 0{,}72$ : elle a baissé (de 6 à 3,28) |
| h | minimum | **(1 ; 0)** | $f \geq 0$ et $f(1, 0) = 0$ ; le gradient y vaut $(0, 0)$ |
| i | $\frac{\partial g}{\partial x}(1, 2)$ | **4** | $2xy$ |
| j | $\frac{\partial g}{\partial y}(1, 2)$ | **13** | $x^2 + 3y^2 = 1 + 12$ |

**Le dessin** : la ligne de niveau 2 est l'ellipse $(x - 1)^2 + 2y^2 = 2$, centrée en $(1, 0)$, de demi-axes $\sqrt{2} \approx 1{,}41$ horizontalement et 1 verticalement ; la ligne de niveau 8 a les demi-axes $2\sqrt{2} \approx 2{,}83$ et 2. Le point $(3, 1)$ ($f = 6$) est entre les deux ; la flèche du gradient $(4, 4)$ part de ce point vers l'extérieur, perpendiculaire à la ligne de niveau 6 qui y passe. Le pas de descente va dans l'autre sens, vers le centre.
**Erreurs fréquentes** : 2 en c (dériver $2y^2$ en $2y$) ; (3,4 ; 1,4) en f (ajouter le gradient au lieu de le soustraire) ; 12 en j (oublier le $x^2$).
**Remarque** : le gradient ne pointe pas **vers** le minimum $(1, 0)$ (il faudrait la direction $(-2, -1)$), mais dans la direction de plus forte pente : sur des ellipses allongées, la descente de gradient zigzague. Les optimiseurs du ch. 19 (Momentum, Adam) corrigent en partie ce défaut.

### Ex 0B.26 — Indépendance

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $P(A)$ | **0,5** | 3 faces paires sur 6 (18 issues sur 36) |
| b | $P(B)$ | **0,167** | $\frac{6}{36}$ |
| c | $P(A \cap B)$ | **0,083** | $(2, 5), (4, 3), (6, 1)$ : $\frac{3}{36} = \frac{1}{12}$ |
| d | indépendants ? | **True** | $\frac{1}{2} \times \frac{1}{6} = \frac{1}{12}$ |
| e | $P(C)$ | **0,139** | $\frac{5}{36}$ |
| f | $P(A \cap C)$ | **0,083** | $(2, 6), (4, 4), (6, 2)$ : $\frac{3}{36}$ |
| g | indépendants ? | **False** | $\frac{1}{2} \times \frac{5}{36} = \frac{5}{72} \neq \frac{6}{72}$ |
| h | écart | **0,014** | $\frac{6}{72} - \frac{5}{72} = \frac{1}{72}$ |

**Interprétation** : quel que soit le premier dé, exactement une valeur du second donne une somme 7 : connaître la parité du premier dé ne change rien. Pour la somme 8, un premier dé égal à 1 la rend impossible (et 1 est impair) : savoir que le premier dé est pair rend la somme 8 **un peu** plus probable. L'écart est petit ($\frac{1}{72}$), mais la dépendance est réelle : une simulation devra être grande pour la voir (0B.52).
**Erreur fréquente** : arrondir trop tôt et comparer 0,083 à 0,069 « à vue » ; garde les fractions.

### Ex 0B.27 — Espérance et variance

| | Question | Réponse | Pourquoi |
|---|---|---|---|
| a | $\mathbb{E}[X]$ | **1,2** | $0 + 0{,}3 + 0{,}4 + 0{,}5$ |
| b | $\mathbb{E}[X^2]$ | **3,6** | $0 + 0{,}3 + 0{,}8 + 2{,}5$ |
| c | $\mathrm{Var}(X)$ | **2,16** | $3{,}6 - 1{,}44$ |
| d | écart-type | **1,47** | $\sqrt{2{,}16}$ |
| e | $\mathbb{E}[3X + 2]$ | **5,6** | $3 \times 1{,}2 + 2$ |
| f | $\mathrm{Var}(3X + 2)$ | **19,44** | $9 \times 2{,}16$ |
| g | espérance du dé aux faces 1, 1, 2, 3, 3, 3 | **2,167** | $\frac{1 + 1 + 2 + 3 + 3 + 3}{6} = \frac{13}{6}$ |
| h | sa variance | **0,806** | $\frac{33}{6} - \left(\frac{13}{6}\right)^2 = \frac{198 - 169}{36} = \frac{29}{36}$ |

**Erreurs fréquentes** : 1,44 en b (calculer $\mathbb{E}[X]^2$ au lieu de $\mathbb{E}[X^2]$) ; 2,4 en c (oublier le carré de $\mathbb{E}[X]$) ; 8,48 en f ($3 \times 2{,}16 + 2$ : le $b$ ne compte pas et le $a$ est au carré) ; 0,804 en h (arrondir $\mathbb{E}[X]$ à 2,167 avant de l'élever au carré : garde $\frac{13}{6}$).
**Variante** : quelle est la variance du nombre total de messages en deux minutes, si les deux minutes sont indépendantes ? ($2 \times 2{,}16 = 4{,}32$ : pour des variables **indépendantes**, les variances s'additionnent. Tu l'observeras par simulation en 0B.53 ; la preuve viendra au ch. 16.)

### Ex 0B.28 — Variance : deux formules, et l'espérance est linéaire ∂

1. On coupe la somme en deux et on sort les constantes :

$$\mathbb{E}[aX + b] = \sum_k (a x_k + b)\,p_k = a \sum_k x_k p_k + b \sum_k p_k = a\,\mu + b \times 1$$

2. On développe $(x_k - \mu)^2 = x_k^2 - 2\mu\,x_k + \mu^2$, puis on coupe en trois sommes :

$$\mathrm{Var}(X) = \sum_k x_k^2 p_k - 2\mu \sum_k x_k p_k + \mu^2 \sum_k p_k = \mathbb{E}[X^2] - 2\mu \times \mu + \mu^2 \times 1 = \mathbb{E}[X^2] - \mu^2$$

3. Par la question 1, $(aX + b) - \mathbb{E}[aX + b] = aX + b - a\mu - b = a(X - \mu)$. Donc

$$\mathrm{Var}(aX + b) = \sum_k a^2 (x_k - \mu)^2\,p_k = a^2 \sum_k (x_k - \mu)^2\,p_k = a^2\,\mathrm{Var}(X)$$

4. Sur la loi de 0B.27 ($\mu = 1{,}2$) :
   - $3X + 2$ prend les valeurs $2, 5, 8, 17$, d'où $\mathbb{E}[3X + 2] = 0{,}8 + 1{,}5 + 1{,}6 + 1{,}7 = 5{,}6 = 3 \times 1{,}2 + 2$ ✔️ ;
   - $\sum_k (x_k - 1{,}2)^2 p_k = 1{,}44 \times 0{,}4 + 0{,}04 \times 0{,}3 + 0{,}64 \times 0{,}2 + 14{,}44 \times 0{,}1 = 0{,}576 + 0{,}012 + 0{,}128 + 1{,}444 = 2{,}16 = 3{,}6 - 1{,}44$ ✔️ ;
   - $\mathrm{Var}(3X + 2) = 3{,}6^2 \times 0{,}4 + 0{,}6^2 \times 0{,}3 + 2{,}4^2 \times 0{,}2 + 11{,}4^2 \times 0{,}1 = 5{,}184 + 0{,}108 + 1{,}152 + 12{,}996 = 19{,}44 = 9 \times 2{,}16$ ✔️.
5. $\mathrm{Var}(X) = \sum_k (x_k - \mu)^2\,p_k$ est une somme de termes positifs ou nuls (un carré fois une probabilité), donc $\mathrm{Var}(X) \geq 0$, et par la question 2, $\mathbb{E}[X^2] - \mathbb{E}[X]^2 \geq 0$. Elle est nulle seulement si $X$ est constante.

**Critères** : $\sum_k p_k = 1$ est utilisé explicitement (1 et 2) ; le développement de 2 a ses trois termes ; en 3, le $b$ disparaît **avant** de mettre au carré.

### Pour finir

### Ex 0B.29 — Règle de la chaîne à deux variables : la somme sur les chemins ∂

**Partie A.**
1. Le graphe a quatre arêtes : $x \xrightarrow{\;2\;} u$, $x \xrightarrow{\;2x\;} v$, $u \xrightarrow{\;2u + v\;} z$, $v \xrightarrow{\;u\;} z$ (sur chaque arête, la dérivée locale : $\frac{du}{dx} = 2$, $\frac{dv}{dx} = 2x$, $\frac{\partial z}{\partial u} = 2u + v$, $\frac{\partial z}{\partial v} = u$).
2. Deux chemins de $x$ à $z$ :

$$\frac{dz}{dx} = \underbrace{(2u + v) \times 2}_{\text{par } u} + \underbrace{u \times 2x}_{\text{par } v}$$

En $x = 1$ : $u = 2$ et $v = 1$, donc $\frac{dz}{dx} = 5 \times 2 + 2 \times 2 = $ **14**.
3. Directement : $z = (2x)^2 + 2x \times x^2 = 4x^2 + 2x^3$, donc $\frac{dz}{dx} = 8x + 6x^2$, qui vaut $8 + 6 = 14$ en $x = 1$. ✔️

**Partie B.**
1. $w$ et $b$ sont reliés à $\hat{y}_1$ et à $\hat{y}_2$ ; $\hat{y}_1$ et $\hat{y}_2$ sont reliés à $L$. Dérivées locales : $\frac{\partial \hat{y}_i}{\partial w} = x_i$, $\frac{\partial \hat{y}_i}{\partial b} = 1$, $\frac{\partial L}{\partial \hat{y}_i} = 2(\hat{y}_i - y_i)$.
2. Un terme par chemin :

$$\frac{\partial L}{\partial w} = 2(\hat{y}_1 - y_1)\,x_1 + 2(\hat{y}_2 - y_2)\,x_2 \qquad \frac{\partial L}{\partial b} = 2(\hat{y}_1 - y_1) + 2(\hat{y}_2 - y_2)$$

3. En $w = 1$, $b = 0$ : $\hat{y}_1 = 1$, $\hat{y}_2 = 2$, les écarts valent $1 - 3 = -2$ et $2 - 4 = -2$, et $L = 4 + 4 = 8$.
$\frac{\partial L}{\partial w} = 2(-2)(1) + 2(-2)(2) = $ **−12** et $\frac{\partial L}{\partial b} = 2(-2) + 2(-2) = $ **−8**.
4. $w \leftarrow 1 - 0{,}1 \times (-12) = $ **2,2** et $b \leftarrow 0 - 0{,}1 \times (-8) = $ **0,8**. Nouvelles prédictions : $\hat{y}_1 = 3{,}0$ et $\hat{y}_2 = 5{,}2$ ; $L = 0^2 + 1{,}2^2 = $ **1,44** : la loss a bien baissé (de 8 à 1,44). Les deux dérivées étaient négatives (le modèle prédisait trop bas), donc le pas a **augmenté** $w$ et $b$.
5. Parce qu'on parcourt le graphe **à l'envers**, de la loss vers les poids : on calcule d'abord $\frac{\partial L}{\partial \hat{y}_i}$ à la sortie, puis on remonte arête par arête en multipliant par les dérivées locales ; chaque résultat intermédiaire sert à tous les chemins qui passent par ce nœud, au lieu d'être recalculé pour chaque poids (ch. 18).

**Critères** : un terme par chemin, un produit le long de chaque chemin ; les valeurs numériques vérifiées par un calcul direct (A3) ou par la baisse de la loss (B4).
**Variante** : refais B en moyennant la loss sur les deux exemples ($L/2$, la MSE) : les dérivées sont divisées par 2 (−6 et −4) et, avec le même $\eta$, le pas est deux fois plus petit.

<a id="reflexion"></a>

## 🧮 🗣️ 🛠️ Réflexion et outils

### Ex 0B.30 — Fermi : combien de multiplications dans un produit matriciel ? 🧮

1. $60\,000 \times 784 \times 128 \approx 6 \times 10^4 \times 8 \times 10^2 \times 1{,}3 \times 10^2 \approx 6 \times 10^9$ multiplications (exactement 6 021 120 000) : de l'ordre de $10^{10}$.
2. Temps = multiplications / vitesse :
   - boucle Python ($10^7$ par seconde) : $6 \times 10^2$ s, soit **une dizaine de minutes** ;
   - bibliothèque optimisée ($10^{10}$ à $10^{11}$ par seconde) : **0,06 à 0,6 s** ;
   - GPU ($10^{13}$ par seconde et plus) : **moins d'une milliseconde**.

   Même calcul, un rapport de un à un million entre le premier et le dernier : c'est pour cela qu'on ne code jamais un produit matriciel avec des boucles Python (0B.40), et qu'on entraîne les gros réseaux sur GPU.
3. De gauche à droite : $\mathbf{A}\mathbf{B}$ coûte $10^3 \times 10^3 \times 10^3 = 10^9$, puis $(\mathbf{A}\mathbf{B})\mathbf{C}$ encore $10^9$, puis le produit par $\mathbf{v}$, $10^6$ : **environ $2 \times 10^9$**. De droite à gauche : $\mathbf{C}\mathbf{v}$ coûte $10^6$ et donne un **vecteur** ; $\mathbf{B}(\mathbf{C}\mathbf{v})$ encore $10^6$ ; $\mathbf{A}(\dots)$ encore $10^6$ : **$3 \times 10^6$**.
4. Le résultat est **le même** (le produit matriciel est associatif : $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$), mais le calcul de droite à gauche est environ $\frac{2 \times 10^9}{3 \times 10^6} \approx 700$ fois plus rapide. On ne multiplie jamais deux matrices quand on peut multiplier une matrice par un vecteur. (Les arrondis des `float` peuvent différer au dernier chiffre, mais c'est le même nombre en théorie.)
5. La loss est un **nombre** ; la rétropropagation part d'elle et multiplie, étape par étape, un **vecteur** de dérivées par les dérivées locales de chaque couche (des matrices) : comme $\mathbf{A}(\mathbf{B}(\mathbf{C}\mathbf{v}))$, on enchaîne des produits matrice-vecteur, sans jamais former le produit des matrices entre elles. C'est ce qui rend le calcul du gradient du même ordre de coût qu'une passe avant, environ deux à trois fois plus, quel que soit le nombre de poids (ch. 18). Tu le mesureras en 0B.54.

**Critères** : les ordres de grandeur sont justes à un facteur 10 près ; l'associativité est citée en 4 ; en 5, l'idée « partir d'un scalaire, faire des produits matrice-vecteur » est présente.

### Ex 0B.31 — Le gradient expliqué à un randonneur (réponse modèle) 🗣️

> Tu ne vois rien, mais tu sens la pente sous tes pieds : tâte le sol autour de toi pour trouver la direction où ça **monte le plus fort** ; c'est ça, le gradient (une direction, et une raideur). Fais un pas dans la direction **opposée** : c'est la descente la plus rapide depuis là où tu es. Recommence à chaque pas : la pente change, donc la direction aussi. Choisis bien la **longueur de tes pas** : trop longs, tu passes par-dessus le fond de la vallée et tu remontes de l'autre côté ; trop courts, tu y seras demain. Et méfie-toi : tu peux t'arrêter au fond d'une **cuvette** où tout remonte autour de toi, sans être dans la vallée.

**Critères** : la pente locale (information seulement autour de soi) ; la direction de plus forte montée ; aller dans le sens opposé ; répéter ; la taille des pas (le learning rate) et ses deux excès ; un risque (minimum local, cuvette). Pas de formule, pas de jargon non expliqué.

### Ex 0B.32 — Écrire des maths en LaTeX dans Markdown 🛠️

Une copie possible de la section 0B.32 de `06_mes_reponses.md` :

```markdown
1. La moyenne : $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
2. Un neurone : $z = \mathbf{w} \cdot \mathbf{x} + b$
3. La sigmoïde : $\sigma(x) = \frac{1}{1 + e^{-x}}$
4. Un pas de descente de gradient : $\mathbf{x} \leftarrow \mathbf{x} - \eta\,\nabla f(\mathbf{x})$
5. La matrice de 0B.21 :

$$\mathbf{M} = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}$$
```

Ce qui s'affiche :
1. La moyenne : $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
2. Un neurone : $z = \mathbf{w} \cdot \mathbf{x} + b$
3. La sigmoïde : $\sigma(x) = \frac{1}{1 + e^{-x}}$
4. Un pas de descente de gradient : $\mathbf{x} \leftarrow \mathbf{x} - \eta\,\nabla f(\mathbf{x})$
5. La matrice de 0B.21 : $\mathbf{M} = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}$

**Erreurs fréquentes** : `e^-x` (seul le `-` passe en exposant : écris `e^{-x}`) ; `\frac 1 1+e^{-x}` (sans accolades, seul le premier caractère est pris) ; une matrice sans `\\` entre les lignes ; un `$$` non refermé, qui transforme tout le reste du fichier en formule.
**Astuce** : sur GitHub, une formule entre `$` qui commence ou finit par une espace (`$ x $`) ne s'affiche pas toujours ; colle les `$` à la formule.

<a id="entretien"></a>

## 💼 Entretien

### 0B.E1 — Qu'est-ce qu'un gradient, et à quoi sert-il pour entraîner un modèle ?

**Réponse modèle en 60 secondes** : « Pour une fonction de plusieurs variables, le gradient est le vecteur de ses dérivées partielles : une par variable. Il indique la direction dans laquelle la fonction augmente le plus vite, et sa norme dit à quel point elle augmente. Pour entraîner un modèle, la fonction qui nous intéresse est la loss, vue comme une fonction de tous les poids du modèle. On calcule son gradient par rapport aux poids, puis on fait un petit pas dans la direction opposée : chaque poids est corrigé de moins le learning rate fois sa dérivée partielle. On répète ce pas des milliers de fois, en général sur des mini-lots de données, ce qui donne la descente de gradient stochastique. Pour un réseau de neurones, le gradient est calculé efficacement par la rétropropagation, qui applique la règle de la chaîne de la sortie vers l'entrée. »
**Relances possibles** : « Que se passe-t-il si le learning rate est trop grand ? » (la loss oscille ou diverge ; trop petit : convergence très lente) · « Le gradient nul garantit-il un minimum ? » (non : maximum, point-selle ou plateau ; en grande dimension, les points-selles sont fréquents) · « Pourquoi des mini-lots plutôt que tout le dataset ? » (un gradient approché bien moins cher, et un bruit qui aide parfois à sortir des mauvais minima ; ch. 19).

### 0B.E2 — Produit scalaire et similarité cosinus : à quoi servent-ils en ML ?

**Réponse modèle en 60 secondes** : « Le produit scalaire de deux vecteurs est la somme des produits de leurs composantes ; divisé par le produit des normes, il donne le cosinus de leur angle, entre −1 et 1 : c'est la similarité cosinus. Dans un espace d'embeddings, la **direction** d'un vecteur porte le sens, alors que sa **longueur** dépend souvent d'autre chose, comme la longueur du document ou la fréquence du mot. La similarité cosinus ignore cette longueur : deux textes sur le même sujet, l'un long et l'autre court, restent proches. La distance euclidienne, elle, les séparerait. En pratique, on normalise souvent les embeddings : pour des vecteurs de norme 1, la distance au carré vaut deux moins deux fois le cosinus, donc les deux critères donnent exactement le même classement, et la similarité se réduit à un produit scalaire, très rapide à calculer, y compris par les index de recherche vectorielle. »
**Relances possibles** : « Que vaut la similarité de deux vecteurs orthogonaux ? » (0 : aucun lien) · « Quand préférer la distance euclidienne ? » (quand la norme a un sens, par exemple des mesures physiques standardisées pour un k plus proches voisins) · « Où voit-on un produit scalaire dans un transformer ? » (l'attention compare requêtes et clés par des produits scalaires ; chapitre bonus B3).

### 0B.E3 — Pourquoi manipuler des log-probabilités plutôt que des probabilités ?

**Réponse modèle en 60 secondes** : « Pour deux raisons, une numérique et une pratique. Quand on suppose les exemples indépendants, la probabilité d'un dataset entier, la vraisemblance, est le produit des probabilités de chaque exemple. Avec des milliers d'exemples, ce produit devient si petit qu'il est arrondi à zéro en virgule flottante : un `float64` ne descend pas beaucoup sous $10^{-308}$. Le logarithme transforme ce produit en une somme de nombres raisonnables, sans underflow. Ensuite, le logarithme est croissant : maximiser la log-vraisemblance revient à maximiser la vraisemblance, avec le même optimum, et une somme se dérive terme à terme, ce qui est bien plus simple pour le gradient. Minimiser l'opposé de la log-vraisemblance, c'est exactement minimiser la cross-entropy. C'est aussi pour cela que les bibliothèques fournissent des fonctions comme `log_softmax`, qui calculent directement le logarithme de façon stable. »
**Relances possibles** : « Pourquoi ne pas calculer `softmax` puis `log` ? » (une probabilité arrondie à 0 donne $\log 0 = -\infty$ ; `log_softmax` évite ce passage) · « Que mesure la perplexité d'un modèle de langage ? » (l'exponentielle de la log-vraisemblance moyenne négative par token ; ch. 6 et 22).

### 0B.E4 — La règle de la chaîne, et pourquoi la rétropropagation en dépend

**Réponse modèle en 60 secondes** : « Un réseau de neurones est une composition de fonctions simples : chaque couche transforme la sortie de la précédente, puis la loss compare la sortie finale à la cible. La règle de la chaîne dit que la dérivée d'une composition est le produit des dérivées de chaque étape. Quand une variable influence la loss par plusieurs chemins, on additionne les contributions de tous les chemins. La dérivée de la loss par rapport à un poids est donc une somme, sur les chemins, de produits de dérivées locales. La rétropropagation organise ce calcul : elle part de la loss et remonte le graphe vers l'entrée, en calculant à chaque nœud la dérivée de la loss par rapport à ce nœud à partir de celle du nœud suivant. Chaque résultat intermédiaire est calculé une seule fois et réutilisé par tous les chemins qui passent par là, si bien que le gradient de millions de poids ne coûte que deux à trois passes avant environ. »
**Relances possibles** : « Pourquoi en arrière et pas en avant ? » (une seule sortie scalaire, beaucoup d'entrées : le mode inverse enchaîne des produits vecteur-matrice ; 0B.30) · « Que faut-il garder en mémoire pendant la passe avant ? » (les valeurs intermédiaires, dont les dérivées locales ont besoin : c'est ce qui coûte de la mémoire à l'entraînement) · « Qu'est-ce qu'un gradient qui s'évanouit ? » (un long produit de dérivées plus petites que 1, qui fond comme une suite géométrique ; 0B.13).

### 0B.E5 — Espérance, moyenne d'un échantillon, variance : quelles différences ?

**Réponse modèle en 60 secondes** : « L'espérance est une propriété de la loi d'une variable aléatoire : la moyenne de ses valeurs possibles, pondérée par leurs probabilités. C'est un nombre fixe, en général inconnu. La moyenne d'un échantillon se calcule sur des données observées ; c'est une estimation de l'espérance, qui change d'un échantillon à l'autre. La loi des grands nombres dit qu'elle se rapproche de l'espérance quand la taille de l'échantillon grandit, avec une erreur typique qui décroît comme un sur racine de $n$ : pour être dix fois plus précis, il faut cent fois plus de données. La variance mesure la dispersion autour de l'espérance : c'est l'espérance du carré de l'écart ; l'écart-type, sa racine, est dans la même unité que les données. En ML, un score de test est une moyenne d'échantillon : il faut donner son incertitude, par exemple en répétant l'expérience avec plusieurs graines et en rapportant l'écart-type. »
**Relances possibles** : « Pourquoi divise-t-on parfois par $n - 1$ ? » (pour estimer sans biais la variance de la population à partir d'un échantillon ; `np.var(x, ddof=1)` ; ch. 2) · « Votre modèle gagne 0,5 point d'accuracy : est-ce significatif ? » (ça dépend de la taille du jeu de test et de la variabilité entre graines ; ch. 8).

<a id="notebook"></a>

## Notebook, parties A à D

Le code complet et exécuté est dans `05_solutions.ipynb` ; voici les réponses, le *pourquoi* et les pièges. Les réponses de la partie 0 sont celles des exercices papier ci-dessus.

### Ex 0B.33 — Calculer avec Python : puissances, arrondis, `abs`, signe et `C(n, k)`
a) **[9, −9]** · b) **[−4, −3, −3, −4]** · c) **3** · d) **[−1, 0, 1, −1, 1, 1]** · e) **2 598 960** · f) **31**.
**Pourquoi** : a) en Python comme en maths, la puissance passe avant le signe moins : `-3 ** 2` se lit $-(3^2)$ ; écris `(-3) ** 2` pour élever −3 au carré. b) `floor` va vers le bas, `ceil` vers le haut, `int` tronque vers zéro, `round` va à l'entier le plus proche. c) −0,2, 0 et 1 (l'inégalité est large). e) une main ne dépend pas de l'ordre : $\binom{52}{5}$. f) $2^{100} \approx 1{,}27 \times 10^{30}$ : un nombre $x \geq 1$ a $\lfloor \log_{10} x \rfloor + 1$ chiffres.
**Erreurs fréquentes** : [9, 9] en a ; 2 en c (oublier que $|1| \leq 1$) ; `math.perm(52, 5)`, 120 fois trop grand, en e ; 30 en f.

### Ex 0B.34 — 0,99 puissance 1000 : presque 1 ou presque 0 ? 🔮
a) **"proche de 1"** (0,904) · b) **"petit"** (0,366) · c) **"zero"** ($4{,}3 \times 10^{-5}$) · d) **"grand"** (2,70) · e) **"grand"** (≈ 21 000) · f) **"proche de 1"** (0,905) · g) **459**.
**Pourquoi** : une petite différence de raison, répétée des centaines de fois, devient énorme : on perd environ 1 % à chaque pas, mais sur 1000 pas cela fait un facteur 23 000. Pour f, $0{,}99 \times 1{,}01 = 0{,}9999$ : la baisse et la hausse se compensent presque. g : $0{,}99^k < 0{,}01 \iff k > \frac{\ln 0{,}01}{\ln 0{,}99} \approx 458{,}2$ (l'inégalité change de sens, car on divise par $\ln 0{,}99 < 0$), donc $k = 459$.
**Erreur fréquente** : 458 en g (boucle qui s'arrête un tour trop tôt, ou arrondi de 458,2 vers le bas).
**En ML** : c'est le mécanisme des gradients qui s'évanouissent ou explosent au fil des couches ou des pas de temps (ch. 22).

### Ex 0B.35 — Σ, Π et moyennes en code
a) **42 925** (la formule donne $\frac{50 \times 51 \times 101}{6}$, le même nombre) · b) **11** · c) **13,4** · d) **3,5** · e) **192** · f) **[0,340 ; 0,338 ; 0,375]**.
**Pourquoi b vaut 11** : $\frac{2}{1} \times \frac{3}{2} \times \frac{4}{3} \times \dots \times \frac{11}{10}$ : chaque numérateur se simplifie avec le dénominateur suivant (produit « télescopique »), il reste $\frac{11}{1}$.
**Une moyenne mobile** : `np.array([np.mean(x[t - k + 1:t + 1]) for t in range(k - 1, len(x))])`. Sur la figure, l'ordre 3 laisse passer le bruit ; l'ordre 21 suit la sinusoïde, avec un retard d'environ 10 pas.
**Erreurs fréquentes** : `range(1, 50)` en a (42 925 − 2 500 = 40 425) ; `np.mean` au lieu de `np.average` en c (13,25) ; oublier le biais en d (3,0) ; une fenêtre `x[t - k : t]` décalée d'un pas en e et f.
**Variante** : avec `c = np.cumsum(np.insert(x, 0, 0))`, les moyennes mobiles sont `(c[k:] - c[:-k]) / k`, un calcul en temps linéaire.

### Ex 0B.36 — Galerie des fonctions usuelles
a) **[0,119 ; 0,5 ; 0,881]** · b) **True** · c) une figure de 9 panneaux, chacun titré, sans `nan` ni valeur infinie.
**Lecture de la galerie** : bornées : $\sigma$ (entre 0 et 1), $\tanh$ et $\cos$ (entre −1 et 1) ; croissantes : l'affine, $e^x$, $\ln$, $\sigma$, $\tanh$, la partie entière (en escalier) ; paires : $|x|$ et $\cos$ ; impaire : $\tanh$ (et $\sigma - \frac{1}{2}$).
**Erreurs fréquentes** : une `sigmoid` écrite avec `math.exp`, qui refuse les arrays (`TypeError: only length-1 arrays…`) ; $\ln$ tracé sur $[-4 ; 4]$, qui produit des `nan` et un `RuntimeWarning`.

### Ex 0B.37 — `exp` et `log` en NumPy 🐛
a) **−1204,0** · b) **3843,5** · c) **[0 ; 1,504 ; 0 ; 2,944 ; 1,981]** · d) **709**.
**Les trois causes** : 1. $0{,}3^{1000} \approx 10^{-523}$ est trop petit pour un `float64` : `np.prod` renvoie 0 (underflow) et $\ln 0 = -\infty$ ; la somme des logarithmes, $1000 \ln 0{,}3$, ne pose aucun problème. 2. Le produit de 500 nombres plus grands que 100 dépasse $10^{308}$ : `inf`. 3. $x - \bar{x}$ est négatif pour les valeurs sous la moyenne, et $\ln$ d'un nombre négatif vaut `nan`. d) `np.exp(709)` vaut environ $8 \times 10^{307}$, `np.exp(710)` dépasse le plus grand `float64` ($\approx 1{,}8 \times 10^{308}$).
**À retenir** : NumPy ne s'arrête pas sur ces erreurs, il continue avec `inf` ou `nan`, qui contaminent tous les calculs suivants. Un `RuntimeWarning` est un signal d'alarme à ne jamais ignorer. En ML, on garde les produits de probabilités sous forme de sommes de logarithmes (0B.E3).

### Ex 0B.38 — `linalg_basics` (1)
a) **[88 ; 386]** · b) **[−8 ; −6]** · c) **[44 ; 193]** · d) **[2 ; 0 ; −2]** · e) **`ValueError`**, puis les 13 tests passent.
**Une solution** :
```python
def vector_add(u, v):
    if len(u) != len(v):
        raise ValueError(f"vectors of different lengths: {len(u)} and {len(v)}")
    return [float(a + b) for a, b in zip(u, v)]
```
(même modèle pour `vector_subtract` et `hadamard` ; `scalar_multiply` : `[float(c * x) for x in v]`).
**Erreurs fréquentes** : `u + v` sur des listes, qui les **colle** ; oublier le test des longueurs (`zip` s'arrête sans rien dire au plus court : « no error » en e) ; modifier `u` sur place au lieu de créer une nouvelle liste ; renvoyer des entiers.

### Ex 0B.39 — `linalg_basics` (2)
a) **0** · b) **[5 ; 7 ; 4]** · c) **10** · d) **1** · e) **0,707**, puis les 18 tests passent.
**Points délicats** : `norm` traite `p = math.inf` à part et refuse `p < 1` ; `cosine_similarity` refuse un vecteur nul **avant** de diviser, et ramène le résultat dans $[-1, 1]$ (avec `max(-1.0, min(1.0, value))`), car l'arrondi peut donner `1.0000000000000002`.
**Erreurs fréquentes** : L1 et L∞ sans valeurs absolues ([5, −1, 3]) ; oublier de diviser en d (18), ou diviser par une seule norme (6 ou 3) ; `distance` écrite comme `norm(u) - norm(v)` (la différence des longueurs, pas la distance entre les points).

### Ex 0B.40 — Tes fonctions contre NumPy
a) **−7** · b) **[4,61 ; 7,5 ; 4]** · c) **−0,277** · d) **True** (écarts nuls ou minuscules, au plus $10^{-14}$ environ) · e) une mesure, pas une réponse à saisir : la cellule vérifie que NumPy est au moins 10 fois plus rapide (ici, environ 150 à 300 fois selon la machine et sa charge).
**Pourquoi des écarts parfois non nuls** : quand les additions ne se font pas dans le même ordre, l'arrondi flottant peut différer au dernier chiffre ; c'est pourquoi on compare des flottants avec une tolérance.
**Pourquoi NumPy va plus vite** : comme en 0A.55, la boucle Python interprète une instruction et manipule un objet `float` à chaque composante ; NumPy parcourt un bloc de mémoire contigu en code compilé, avec des instructions qui traitent plusieurs nombres à la fois.

### Ex 0B.41 — Distance ou similarité cosinus 🔬
a) **"mixed"** · b) **"penguins_long"** · c) **"penguins_long"** · d) **True** · e) **16,79**.
**Analyse (modèle)** : « La distance dépend de la longueur du texte : répéter la requête $k$ fois l'éloigne ou la rapproche du texte long, alors que le sujet ne change pas. La similarité cosinus ne regarde que les proportions des mots (la direction du vecteur) : elle reste 0,985 pour tout $k$. Pour comparer des textes de longueurs différentes, on prend donc la similarité cosinus, ou la distance entre vecteurs normalisés, qui donne le même classement ($\|\mathbf{a} - \mathbf{b}\|^2 = 2 - 2\cos$, 0B.19). »
**Remarque** : la distance à `penguins_long` diminue d'abord ($k = 1$ à 3), puis augmente : `3 * query` $= (6, 3, 3, 0, 0, 0)$ est le plus proche de $(6, 4, 3, 0, 1, 0)$.

### Ex 0B.42 — `linalg_basics` (3)
a) **(2, 3)** · b) **[[2, −1], [0, 3], [1, 2]]** · c) **[1 ; 3]** · d) **[5,5 ; 2,5 ; −1]** · e) **[1,5 ; −2 ; 3,25]** · f) **True**, puis les 18 tests passent.
**Une solution pour `identity`** : après `if n < 1: raise ValueError(...)`, `[[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]` crée une **nouvelle** liste par ligne.
**Erreurs fréquentes** : `[[0.0] * n] * n` (les $n$ lignes sont la même liste : modifier `I[0][1]` modifie toutes les lignes, un test le vérifie) ; `shape` qui ne regarde que la première ligne (une matrice « irrégulière » passerait) ; un message d'erreur sans les formes.

### Ex 0B.43 — `linalg_basics` (4) : `matmul`
a) **[[2, 3], [13, 7]]** (0B.20 c) · b) **(3, 3)** · c) **[[5, 1], [−2, 0], [5, 3]]** · d) **`ValueError`**, puis les 4 tests passent.
**Une solution** :
```python
def matmul(A, B):
    shape_a, shape_b = shape(A), shape(B)
    if shape_a[1] != shape_b[0]:
        raise ValueError(f"cannot multiply {shape_a} by {shape_b}")
    columns = transpose(B)
    return [[dot(row, column) for column in columns] for row in A]
```
**Erreurs fréquentes** : vérifier les formes **après** le calcul (un `IndexError` ou un résultat faux avant l'erreur attendue) ; échanger les rôles des lignes et des colonnes (on obtient $\mathbf{B}^\top\mathbf{A}^\top$ ou une erreur) ; une triple boucle correcte mais qui oublie de convertir en `float`.

### Ex 0B.44 — AB = BA ? (AB)ᵀ = BᵀAᵀ ? 🔮
a) **False** · b) **False** · c) **True** · d) **True** · e) **True** · f) **True**.
**Pourquoi** : le produit matriciel n'est pas commutatif (a), mais il est associatif (d) et distributif (e). La transposée d'un produit inverse l'ordre (c) : avec des formes $(m, n)$ et $(n, p)$, $\mathbf{A}^\top\mathbf{B}^\top$ n'existe même pas en général (b). Deux matrices diagonales commutent (f) : le produit multiplie simplement les éléments diagonaux deux à deux.
**À retenir** : un seul contre-exemple réfute une règle ; des milliers d'exemples ne la démontrent pas, mais la démonstration de c se fait en une ligne avec la formule $(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$.

### Ex 0B.45 — Le produit qui n'en est pas un 🐛
a) **(5, 3)** · b) **`ValueError`** · c) **32** · d) **[−0,5 ; 2 ; 4 ; 2,5 ; 6,5]** · e) **[[1,5 ; 1,5], [1,5 ; −0,5], [3,5 ; −1,5], [2,5 ; −0,5], [5,5 ; −2,5]]** · f) **[[4, 5, 6], [8, 10, 12], [12, 15, 18]]**.
**Les trois causes** : 1. `X * w` multiplie chaque ligne de `X` par `w` élément par élément (broadcasting) : aucune erreur, mais une forme `(5, 3)` au lieu de `(5,)`. 2. `W @ X` est un produit $(3, 2) \times (5, 3)$ : $2 \neq 5$. 3. `v.T` ne change rien à un tableau `(3,)` : `u @ v` est le produit scalaire, $4 + 10 + 18 = 32$.
**Corrections** : `X @ w + b`, `X @ W + b` (le biais `(2,)` est ajouté à chaque ligne par broadcasting), `u[:, None] @ v[None, :]`.
**À retenir** : le bug 1 est le plus dangereux, parce qu'il est silencieux ; vérifie `.shape` après chaque étape d'un calcul matriciel.

### Ex 0B.46 — Inverse et systèmes
a) **2,0** · b) **[[1, −0,5], [−2, 1,5]]** · c) **[2 ; −1]** · d) **`LinAlgError`** · e) **[2 ; 3 ; −1]** · f) **True**.
**Pourquoi `solve`** : il résout le système par élimination, sans former l'inverse : environ trois fois moins de calculs, et un résultat au moins aussi précis (sur le système $500 \times 500$, un écart de l'ordre de $10^{-11}$ dans les deux cas ici, et `solve` environ 3 fois plus rapide). Règle pratique : `inv(A) @ b` s'écrit `solve(A, b)`.
**Erreur fréquente** : oublier qu'une matrice dont une ligne est proportionnelle à une autre n'est pas inversible (`LinAlgError: Singular matrix`) ; en ML, cela arrive avec deux features identiques (ch. 9).

### Ex 0B.47 — Pentes numériques : vérifier tes dérivées à la main
a) **12,0000** · b) les sept dérivées coïncident avec la pente (écarts inférieurs à $10^{-8}$) · c) **8,155**.
**Les dérivées attendues** : $(x^2 + 2x)e^x$ ; $\ln x + 1$ ; $\frac{-2}{(x - 1)^2}$ ; $2e^{2x + 1}$ ; $\frac{2x}{x^2 + 1}$ ; $-x\,e^{-x^2/2}$ ; $\frac{2 \ln x}{x}$. Toute forme équivalente passe (par exemple $2x e^x + x^2 e^x$).
**Pourquoi la pente centrée** : elle est plus précise que le taux d'accroissement simple (fiche §101.5.1) ; le ch. 5 explique pourquoi, et comment choisir $h$.
**Erreur fréquente** : donner la **valeur** trouvée en 0B.22 au lieu de la **formule** (`lambda x: 8.155`) : la vérification la compare en neuf points et la refuse.

### Ex 0B.48 — Lire les variations 📈
a) **[−1, 3]** · b) **"maximum"** · c) **−25,0** · d) **7**.
**Démarche** : sur le panneau du bas, la pente s'annule en −1 et en 3 ; elle est positive avant −1, négative entre −1 et 3, positive après : $f$ monte, descend, puis remonte. Par le calcul : $f'(x) = 3x^2 - 6x - 9 = 3(x^2 - 2x - 3) = 3(x + 1)(x - 3)$.
**Remarque** : le minimum de $f$ **sur l'intervalle** (−25, en $x = 3$) est aussi son minimum local ; sur $[-5 ; 4{,}5]$, ce serait $f(-5) = -153$, atteint au bord. Toujours regarder aussi les bornes.

### Ex 0B.49 — Carte de lignes de niveau et flèches du gradient 📈
a) **[4, 4]** · b) **[1, 0]** · c) **True** · d) **"y"**.
**Lecture** : les flèches sont perpendiculaires aux ellipses et pointent vers les niveaux croissants ; elles disparaissent au minimum $(1, 0)$. Les ellipses sont plus serrées selon $y$ : pour atteindre le niveau 2 depuis $(1, 0)$, il suffit de monter de 1 selon $y$, alors qu'il faut avancer de $\sqrt{2} \approx 1{,}41$ selon $x$ (à cause du coefficient 2 devant $y^2$).
**Erreur fréquente** : oublier `ax.set_aspect("equal")` : les unités des deux axes n'ont plus la même longueur et les flèches ne paraissent plus perpendiculaires aux lignes de niveau.

### Ex 0B.50 — Contre le gradient, avec lui ou le long d'une ligne de niveau 🔮
a) **"baisser"** · b) **"monter"** · c) **"stable"** · d) **"monter"** · e) **225** · f) **220**.
**L'expérience** : $\Delta f \approx -0{,}0564$ contre le gradient, $+0{,}0567$ avec lui, $+0{,}00015$ le long de la ligne de niveau (400 fois moins : un effet du second ordre), $+0{,}0401$ selon $x$ ($\frac{\partial f}{\partial x} = 4$, et $4 \times 0{,}01 = 0{,}04$).
**e et f** : pour un petit pas, la meilleure direction est exactement $-\nabla f$ (225°). Pour un pas de 0,5, elle tourne vers le minimum (220°, alors que le minimum est vu sous 206,6°). Le gradient ne garantit que la meilleure direction **locale** : c'est pourquoi la descente de gradient fait de petits pas, et les répète (ch. 5 et 19).

### Ex 0B.51 — Dérivées partielles numériques et somme sur les chemins
a) **[4 ; 13]** · b) **14** · c) **[−12 ; −8]** · d) **True** · e) **1,44**.
**Démarche** : `partial_x` fait bouger $x$ seul : `(f(x + h, y) - f(x - h, y)) / (2 * h)`. La somme sur les chemins de 0B.29 s'écrit `(2 * u + v) * 2 + u * (2 * x)` pour A, et, pour B, avec les écarts $e_i = w x_i + b - y_i$ : $\frac{\partial L}{\partial w} = 2e_1 x_1 + 2e_2 x_2$ et $\frac{\partial L}{\partial b} = 2e_1 + 2e_2$. Un pas : `[w - eta * dw, b - eta * db]`, qui fait passer la loss de 8 à 1,44.
**Erreurs fréquentes** : un seul chemin en b (10 au lieu de 14) ; ajouter le gradient au lieu de le soustraire en e (la loss monte à 43,04).
**En ML** : une rétropropagation écrite à la main se vérifie exactement ainsi, en comparant sur de petits exemples ses dérivées aux pentes numériques (*gradient checking*, ch. 18).

### Ex 0B.52 — Simuler des dés 🔬
a) **0,168** · b) **[0,001 ; 0,013]** · c) **[0,190 ; 0,179 ; 0,168]** · d) **[2, 2, 2]**.
**Analyse (modèle)** : « La fréquence de la somme 7 est proche de $\frac{1}{6} \approx 0{,}167$. L'écart $f(A \cap B) - f(A)f(B)$ vaut 0,001 : du bruit de simulation, $A$ et $B$ sont indépendants. L'autre vaut 0,013, proche de la valeur théorique $\frac{1}{72} \approx 0{,}014$ : une vraie dépendance, petite mais visible avec 100 000 lancers. La fréquence des 6 oscille beaucoup au début (0,19 après 100 lancers), puis se stabilise près de $\frac{1}{6}$. Multiplier le nombre de lancers par 4 divise l'écart-type par 2 : l'erreur décroît comme $\frac{1}{\sqrt{n}}$. »
**Erreurs fréquentes** : ne pas suivre exactement les appels au générateur (d'autres nombres, tout aussi justes, mais différents du corrigé) ; comparer des variances au lieu d'écarts-types en d (les rapports sont alors les carrés : [5, 3, 4] ici).

### Ex 0B.53 — Espérance et variance : exact contre simulation
a) **1,2** · b) **2,16** · c) **19,44** · d) **[1,20 ; 2,14]** · e) **4,27** · f) **8,58**.
**Pourquoi** : la simulation retrouve l'espérance et la variance à peu près, et l'écart diminue comme $\frac{1}{\sqrt{n}}$ quand le nombre de tirages grandit. Pour deux variables **indépendantes**, les variances s'additionnent : $2{,}16 + 2{,}16 = 4{,}32$, et la simulation donne 4,27 (démonstration au ch. 16). Pour $X + X = 2X$, $\mathrm{Var}(2X) = 4\,\mathrm{Var}(X) = 8{,}64$ : les deux termes varient ensemble, les variances ne s'additionnent plus. L'espérance, elle, s'additionne toujours.
**Erreurs fréquentes** : oublier le carré de $\mathbb{E}[X]$ (2,4 en b) ; donner les valeurs théoriques à la place des valeurs simulées en d à f ; tirer `x` et `y` dans un autre ordre que celui de l'énoncé.

### Ex 0B.54 — L'ordre des produits 🏆
a) **True** · b) **[2 001 000 000 ; 3 000 000]** · c) un gain de l'ordre de 50 sur la machine de test (30 à 100 selon les machines) : objectif atteint · e) **[42 795 008 ; 7 137 280]**.
**Démarche** : de droite à gauche, $\mathbf{C}\mathbf{v}$ est un vecteur, donc chacun des trois produits est un produit matrice-vecteur ($n^2$ multiplications) ; de gauche à droite, on forme deux produits de matrices ($n^3$ chacun).
**d) Pourquoi le gain en temps (≈ 50) est plus petit que le gain en multiplications (≈ 670)** : un produit matrice-vecteur lit toute la matrice (8 Mo pour $1000 \times 1000$) pour ne faire qu'une multiplication par nombre lu : il est limité par la vitesse de la mémoire. Un produit matriciel réutilise chaque nombre lu des centaines de fois et fait tourner le processeur à plein régime.
**e)** Regrouper les poids divise le coût par 6 : un réseau **linéaire** de trois couches équivaut à une seule matrice $\mathbf{W}_1\mathbf{W}_2\mathbf{W}_3$ de forme $(784, 10)$ ; empiler des couches linéaires n'apporte donc rien. Avec une activation $f$ entre les couches, $f(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2 \neq \mathbf{X}(\mathbf{W}_1\mathbf{W}_2)$ : on ne peut plus regrouper, et c'est la non-linéarité qui donne sa puissance au réseau (ch. 16 et 17).
**En ML** : la rétropropagation part d'un nombre, la loss, et n'enchaîne que des produits vecteur-matrice, comme $\mathbf{A}(\mathbf{B}(\mathbf{C}\mathbf{v}))$ : c'est ce qui rend le calcul du gradient du même ordre de coût qu'une passe avant (ch. 18, et 0B.30).
