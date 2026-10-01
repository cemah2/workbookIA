# 5 · Courbes et surfaces — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 🧮 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 5.Q1 — Pourquoi dérivée et gradient sont au cœur de l'apprentissage

<details><summary>Indice 1</summary>

Relis le §5.1 de la fiche (et du livre) : il dit en quelques phrases ce que la dérivée et le gradient nous apprennent, et quel algorithme du ch. 18 s'en sert.

</details>
<details><summary>Indice 2</summary>

Pour la question 2, pense à ce qu'on règle pendant l'entraînement (les entrées de la fonction) et à ce qu'on mesure (sa sortie). Pour la question 5, combien de poids a un réseau, et combien de valeurs faudrait-il essayer pour chacun ?

</details>
<details><summary>Indice 3</summary>

Les deux outils disent dans quel sens monter ou descendre, et à quel point ça monte. La surface est la loss vue comme une fonction des poids. Le livre annonce lui-même qu'il évite les équations. Au hasard, dans un espace à des millions de dimensions, on ne tombe presque jamais sur de bons poids ; la pente, elle, indique à chaque pas une direction qui fait baisser la loss.

</details>

### 5.Q2 — Une fonction, une table d'entrées et de sorties

<details><summary>Indice 1</summary>

Relis le §5.2 de la fiche : la fonction vue comme une table, et ce qui change entre une courbe et une surface.

</details>
<details><summary>Indice 2</summary>

Compte les nombres qu'on donne et ceux qu'on reçoit. Pour la question 4, cherche dans le livre la condition « tant que… » ; puis pense aux étapes d'un entraînement qui tirent quelque chose au hasard (0A).

</details>
<details><summary>Indice 3</summary>

Courbe : 1 et 1 ; surface : 2 et 1 ; le modèle du prix : 8 entrées et 1 sortie ; la loss : autant d'entrées que de poids. Le hasard volontaire : le tirage des mini-batches, le *dropout*, l'augmentation de données. Une abscisse peut prendre une infinité de valeurs réelles.

</details>

### 5.Q3 — Continue, lisse, univoque : reconnaître les courbes

<details><summary>Indice 1</summary>

Relis la liste des quatre règles du §5.2 de la fiche, et regarde la figure `courbes_interdites.png`.

</details>
<details><summary>Indice 2</summary>

Dessine chaque courbe à main levée. Peux-tu la tracer sans lever le crayon ? A-t-elle un coin ? Une droite verticale la coupe-t-elle deux fois ? Sa tangente devient-elle verticale quelque part ?

</details>
<details><summary>Indice 3</summary>

La partie entière fait des sauts ; $|x - 1|$ a un coin en 1 ; le cercle a deux valeurs pour chaque $x$ de $]-2 ; 2[$ ; la racine cubique a une tangente verticale en 0 ; $x^2 + \sin x$ respecte tout. Pour la question 6 : que doit trouver l'algorithme en chaque point ?

</details>

### 5.Q4 — Minimum local ou minimum global ?

<details><summary>Indice 1</summary>

Relis « Extrema globaux et locaux » dans la fiche (§5.3), et l'encadré ⚠️ sur la phrase du livre.

</details>
<details><summary>Indice 2</summary>

Distingue la **valeur** d'un extremum et les **points** où il est atteint. Pour $(x^2 - 4)^2$, cherche où elle s'annule et ce qu'elle vaut en 0. Sur un intervalle fermé, n'oublie pas les bornes.

</details>
<details><summary>Indice 3</summary>

$\cos x$ vaut 1 en tous les multiples de $2\pi$. $x^3$ descend sans fin vers $-\infty$. $(x^2 - 4)^2 \ge 0$ et s'annule en $\pm 2$ ; que vaut-elle en 0, et que fait-elle autour ? Sur $[0 ; 3]$, $x^2 - 2x = (x - 1)^2 - 1$ : compare la valeur en 1 et les valeurs aux bornes 0 et 3.

</details>

### 5.Q5 — Le signe de la dérivée indique le chemin

<details><summary>Indice 1</summary>

Relis « Le signe de la dérivée » et l'encadré ⚠️ « Pas fixe ou pas proportionnel » dans la fiche (§5.3).

</details>
<details><summary>Indice 2</summary>

Une dérivée négative veut dire que la courbe descend vers la droite : de quel côté monte-t-elle ? Pour la question 4, imagine qu'on est à une demi-longueur de pas du sommet : où arrive-t-on au pas suivant ?

</details>
<details><summary>Indice 3</summary>

On monte dans le sens du signe de la dérivée : ici, vers la gauche. Avec un pas fixe, on enjambe le sommet, puis on revient, indéfiniment. Avec un pas $\eta\,|f'(x)|$, le pas rétrécit à mesure que la pente s'aplatit.

</details>

### 5.Q6 — Dérivée nulle : sommet, creux ou plateau ?

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la dérivée seconde et l'encadré ⚠️ « Plateau » de la fiche (§5.3).

</details>
<details><summary>Indice 2</summary>

Pour chaque fonction, regarde son signe juste à gauche et juste à droite de 0, comparé à $f(0)$. Calcule ensuite $f''(0)$ : quand vaut-elle 0 ?

</details>
<details><summary>Indice 3</summary>

$-x^2$ : maximum ; $x^4$ : minimum ; $x^3$ : ni l'un ni l'autre ; une constante : tous ses points sont à la fois des maxima et des minima, au sens large. $f''(0)$ vaut $-2$, 0 et 0 : seule la première conclut. Le nom du point de la question 3 est dans l'encadré ⚠️ de la fiche.

</details>

### 5.Q7 — Le gradient : une direction et une longueur

<details><summary>Indice 1</summary>

Relis le §5.4 de la fiche et l'encadré 🧮 « La pente dans une direction ».

</details>
<details><summary>Indice 2</summary>

$f$ est une fonction affine : ses dérivées partielles sont des constantes. La pente dans la direction $\mathbf{u}$ est le produit scalaire $\nabla f \cdot \mathbf{u}$ ; elle est nulle quand $\mathbf{u}$ est perpendiculaire au gradient.

</details>
<details><summary>Indice 3</summary>

$\nabla f = (3, -4)$, de norme 5 ; la plus grande montée suit $\frac{(3, -4)}{5}$ ; dans la direction $(0, 1)$, la pente vaut la seconde composante du gradient ; un vecteur perpendiculaire à $(3, -4)$ : $(4, 3)$, à diviser par sa norme. Pour descendre, on va contre le gradient.

</details>

### 5.Q8 — Point selle : un gradient nul sans extremum

<details><summary>Indice 1</summary>

Relis « Les points où le gradient s'annule » dans la fiche (§5.4), avec son mini-exemple sur $xy$.

</details>
<details><summary>Indice 2</summary>

Pour $y^2 - x^2$, regarde $f(x, 0)$ puis $f(0, y)$. Pour $xy$, essaie les diagonales $y = x$ et $y = -x$. Pour la question 4, que vaut le gradient en $(0, 0)$ : peut-il désigner une direction ? Pour la question 5, relis l'encadré 🕰️ du §5.4 de la fiche.

</details>
<details><summary>Indice 3</summary>

Le long de l'axe des $x$, $y^2 - x^2 = -x^2$ (maximum) ; le long de l'axe des $y$, $y^2$ (minimum). Le long de $y = x$, $xy = x^2$ ; le long de $y = -x$, $xy = -x^2$. Pour descendre depuis $(0, 0)$, il faut partir à moins de 45° de l'axe des $x$ (le long de cet axe, $f = -x^2$), mais en $(0, 0)$ le gradient est nul. En grande dimension, un minimum doit monter dans **toutes** les directions à la fois. En une dimension, pense à $x^3$ en 0.

</details>

### 5.Q9 — La fonction max(0, x) a un coin en 0 : est-ce grave ?

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §5.2 de la fiche, sur ReLU.

</details>
<details><summary>Indice 2</summary>

À gauche de 0, $\max(0, x) = 0$ ; à droite, $\max(0, x) = x$. Pour la question 3, calcule $\mathrm{relu}(h)$ et $\mathrm{relu}(-h)$ pour $h > 0$. Pour la question 5, à quelle fréquence une entrée réelle tombe-t-elle **exactement** sur 0 ?

</details>
<details><summary>Indice 3</summary>

Les pentes valent 0 et 1 ; la pente centrée vaut $\frac{h - 0}{2h}$. PyTorch prend, pour une fonction convexe, le sous-gradient de plus petite norme. Pour la question 5, la théorie dit qu'une entrée réelle ne tombe presque jamais pile sur 0 ; l'encadré 🕰️ dit ce qui se passe en `float32`. Autres fonctions anguleuses : la valeur absolue (l'erreur absolue moyenne, la pénalité L1), la *hinge loss* des SVM, le max d'un *max pooling*.

</details>

### 5.Q10 — Ce que fait un réseau quand il « descend le gradient »

<details><summary>Indice 1</summary>

Relis le §5.1 et la fin du §5.4 de la fiche (« Descendre une surface pas à pas »).

</details>
<details><summary>Indice 2</summary>

Dans la mise à jour, quel signe faut-il pour **descendre** ? Pour la question 3, combien d'évaluations de la loss coûte une différence centrée par poids (🧮 5.9) ? Pour la question 4, souviens-toi des points où le gradient s'annule.

</details>
<details><summary>Indice 3</summary>

Les coordonnées sont les poids, l'altitude est la loss ; $w \leftarrow w - \eta\,\frac{\partial L}{\partial w}$. Deux évaluations par poids, des millions de poids : beaucoup trop cher. Un gradient presque nul avec une loss élevée fait penser à un plateau ou à un point selle. Les chapitres : celui de la rétropropagation, et celui des optimiseurs.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 5.R1 — Ch. 4 : mettre à jour un prior après une observation

<details><summary>Indice 1</summary>

Les hypothèses sont les deux machines, l'observation « défectueuse ». Écris le prior et la vraisemblance de chaque machine.

</details>
<details><summary>Indice 2</summary>

L'évidence additionne les deux façons d'obtenir une pièce défectueuse : $P(D) = P(D \mid A)\,P(A) + P(D \mid B)\,P(B)$. Le posterior de B est $\frac{P(D \mid B)\,P(B)}{P(D)}$.

</details>
<details><summary>Indice 3</summary>

$P(D) = 0{,}6 \times 0{,}02 + 0{,}4 \times 0{,}05$. Pour la question 3, refais le calcul avec 0,5 et 0,5. Pour la question 4, relis ∂ 4.7 : que devient un prior nul ?

</details>

### 5.R2 — Ch. 2 : où la courbe en cloche atteint-elle son maximum ?

<details><summary>Indice 1</summary>

La règle de la chaîne : la dérivée de $e^{u(x)}$ est $u'(x)\,e^{u(x)}$.

</details>
<details><summary>Indice 2</summary>

Ici $u(x) = -\frac{(x - \mu)^2}{2\sigma^2}$, donc $u'(x) = -\frac{x - \mu}{\sigma^2}$. L'exponentielle est toujours positive : le signe de $f'$ est celui de $u'$.

</details>
<details><summary>Indice 3</summary>

$f'(x) = -\frac{x - \mu}{\sigma^2}\,f(x)$ s'annule en $x = \mu$, positive avant, négative après : un maximum, qui vaut $\frac{1}{\sigma\sqrt{2\pi}}$. Les points d'inflexion sont à une distance $\sigma$ de la moyenne.

</details>

### 5.R3 — 0B : règle de la chaîne, dériver (3x + 1)²

<details><summary>Indice 1</summary>

La dérivée de $u(x)^2$ est $2\,u(x)\,u'(x)$.

</details>
<details><summary>Indice 2</summary>

Avec $u(x) = 3x + 1$, $u'(x) = 3$. Pour vérifier, développe $(3x + 1)^2 = 9x^2 + 6x + 1$ et dérive terme à terme.

</details>
<details><summary>Indice 3</summary>

$g'(x) = 6(3x + 1)$, nulle en $x = -\frac{1}{3}$, où $g$ vaut 0, sa plus petite valeur possible (un carré). Le pas de descente : $x_1 = 0 - 0{,}05 \times g'(0)$ ; compare ensuite $g(x_1)$ et $g(0)$.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 5.1 — La sécante qui se resserre sur la tangente

<details><summary>Indice 1</summary>

La pente d'une droite qui passe par deux points de la courbe : la différence des hauteurs divisée par la différence des abscisses. Pour la sécante symétrique, les abscisses sont $2 - h$ et $2 + h$ : elles sont à $2h$ l'une de l'autre.

</details>
<details><summary>Indice 2</summary>

Avec $f(x) = \frac{1}{x}$ : $\frac{f(2 + h) - f(2 - h)}{2h}$. Mets les deux fractions au même dénominateur, $(2 + h)(2 - h) = 4 - h^2$, pour obtenir une expression simple en $h$ (question d). Pour f), écris aussi la pente avant comme une fraction simple de $h$, puis calcule les deux écarts avec les fractions exactes.

</details>
<details><summary>Indice 3</summary>

La pente symétrique vaut $-\frac{1}{4 - h^2}$ ; sa limite quand $h$ tend vers 0 est $f'(2)$. La pente avant, entre 2 et $2 + h$, vaut $-\frac{1}{2(2 + h)}$. Pour f), les écarts à $-\frac{1}{4}$ valent $\frac{h}{4(2 + h)}$ (avant) et $\frac{h^2}{4(4 - h^2)}$ (symétrique) : leur rapport se simplifie.

</details>

### Ex 5.2 — Gradient à la main et direction de plus grande pente

<details><summary>Indice 1</summary>

Le gradient rassemble les dérivées partielles : dérive par rapport à $x$ en traitant $y$ comme une constante, puis l'inverse. Le terme $xy$ compte dans les deux.

</details>
<details><summary>Indice 2</summary>

Pour $\frac{\partial f}{\partial x}$, $y$ est une constante : $xy$ se dérive en $y$, et $2y^2$ en 0. La plus grande descente va contre le gradient ; pour la rendre unitaire, divise par la norme. La pente dans la direction $\mathbf{u}$ est $\nabla f \cdot \mathbf{u}$ (fiche, 🧮).

</details>
<details><summary>Indice 3</summary>

$\frac{\partial f}{\partial x} = 2x + y$ et $\frac{\partial f}{\partial y} = x + 4y$ : en $P$, le gradient vaut $(1, -3)$, de norme $\sqrt{10}$. Pour f), un vecteur perpendiculaire à $(a, b)$ est $(b, -a)$ ou $(-b, a)$ : choisis celui de première composante positive, puis divise par sa norme.

</details>

### Ex 5.3 — Trois pas de descente de gradient à la main

<details><summary>Indice 1</summary>

Applique la formule pas à pas : $x_1 = x_0 - \eta\,f'(x_0)$, puis recommence depuis $x_1$. Pour la montée, le signe change.

</details>
<details><summary>Indice 2</summary>

Dérive $2(x - 1)^2$ avec la règle de la chaîne, sans oublier le facteur 2 qui est devant. Pour f), remplace $f'(x_t)$ par son expression dans la formule et retranche 1 des deux côtés. La suite $x_t - 1$ est alors multipliée par le même nombre à chaque pas : quand se rapproche-t-elle de 0 ?

</details>
<details><summary>Indice 3</summary>

$f'(x) = 4(x - 1)$, d'où $x_{t+1} - 1 = (1 - 4\eta)(x_t - 1)$ : la distance au minimum diminue à chaque pas si et seulement si $|1 - 4\eta| < 1$. Avec $\eta = 0{,}5$, le facteur vaut $-1$ : la suite saute d'un côté à l'autre sans se rapprocher. Le minimum est atteint en un pas quand le facteur est nul.

</details>

### Ex 5.4 — Tableau de variations : extrema locaux et globaux de x³ − 3x

<details><summary>Indice 1</summary>

Dérive deux fois : le signe de $f'$ donne le sens de variation, celui de $f''$ la courbure (fiche, 🧮 dérivée seconde).

</details>
<details><summary>Indice 2</summary>

Factorise $f'(x)$ pour trouver ses zéros. Sur un intervalle fermé, le maximum global est soit un maximum local, soit une valeur aux bornes : calcule $f$ aux points critiques et aux deux bornes, puis compare. Même chose sur $[-2 ; 2]$.

</details>
<details><summary>Indice 3</summary>

$f'(x) = 3x^2 - 3 = 3(x - 1)(x + 1)$ et $f''(x) = 6x$ ; $f(-1) = 2$, $f(1) = -2$, $f(2{,}5) = 8{,}125$, $f(2) = 2$ et $f(-2) = -2$. Sur $[-2 ; 2]$, la valeur maximale 2 est atteinte deux fois. Pour h), $f'(0{,}5) < 0$ : vers la gauche, $f$ monte jusqu'à $x = -1$ ; vers la droite, elle descend jusqu'à $x = 1$.

</details>

### Ex 5.5 — Point selle : x² − y² vu dans deux directions

<details><summary>Indice 1</summary>

Calcule les deux dérivées partielles, puis évalue-les en $(0, 0)$. Pour b), restreins $f$ à chaque axe : tu obtiens une fonction d'une seule variable $t$.

</details>
<details><summary>Indice 2</summary>

Pour c), développe $f(t\cos\theta, t\sin\theta)$ : c'est $t^2$ fois un nombre qui ne dépend que de $\theta$, qu'une formule de trigonométrie de 0B simplifie. Pour e), écris un pas pour chaque coordonnée : chacune est multipliée par un facteur fixe.

</details>
<details><summary>Indice 3</summary>

$f(t\cos\theta, t\sin\theta) = t^2(\cos^2\theta - \sin^2\theta) = t^2\cos 2\theta$, et la dérivée seconde de $a\,t^2$ vaut $2a$. Le gradient est $(2x, -2y)$ : les facteurs valent $1 - 2\eta = 0{,}5$ pour $x$ et $1 + 2\eta = 1{,}5$ pour $y$ ; $x$ se rapproche de 0, $y$ s'en éloigne. En g), $y$ vaut 0 et le reste : la descente ne voit jamais la direction qui descend.

</details>

### Ex 5.6 — Pourquoi la différence centrée est plus précise (calcul exact sur x³)

<details><summary>Indice 1</summary>

$(a + h)^3 = a^3 + 3a^2h + 3ah^2 + h^3$ (le binôme, 0B) ; pour $(a - h)^3$, remplace $h$ par $-h$.

</details>
<details><summary>Indice 2</summary>

Dans $(a + h)^3 - (a - h)^3$, les termes de degré pair en $h$ s'annulent ; dans $(a + h)^3 - a^3$, non. Pour la question 7, l'erreur sur le numérateur peut atteindre $2\delta$, et le numérateur est ensuite divisé par $2h$.

</details>
<details><summary>Indice 3</summary>

$D_+(h) = 3a^2 + 3ah + h^2$ et $D_0(h) = 3a^2 + h^2$ : l'erreur avant contient un terme en $h$, l'erreur centrée commence en $h^2$. Pour $x^2$, le numérateur centré vaut exactement $4ah$. La fonction $h^2 + \frac{\delta}{h}$ décroît puis croît : elle a un minimum.

</details>

### Ex 5.7 — Rosenbrock : gradient et minimum à la main

<details><summary>Indice 1</summary>

Pour dériver $b\,(y - x^2)^2$ par rapport à $x$, la règle de la chaîne : $2b\,(y - x^2) \times (-2x)$.

</details>
<details><summary>Indice 2</summary>

$\frac{\partial f}{\partial y} = 2b\,(y - x^2)$ s'annule seulement si $y = x^2$ ; reporte dans $\frac{\partial f}{\partial x}$. Pour la question 5, développe chaque fonction de $u$ (ou de $t$) jusqu'au terme en $u^2$ : sa dérivée seconde en 0 vaut deux fois le coefficient de $u^2$.

</details>
<details><summary>Indice 3</summary>

$\frac{\partial f}{\partial x} = -2(a - x) - 4bx\,(y - x^2)$. En $(1 + u, 1)$, $y - x^2 = -2u - u^2$ ; en $(1 + t, 1 + 2t)$, $y - x^2 = -t^2$ : le second terme ne compte plus qu'à l'ordre 4. Pour la question 6, sur une parabole de dérivée seconde $c$, l'écart au minimum est multiplié à chaque pas par $1 - \eta c$ (✏️ 5.3).

</details>

<a id="reflexion"></a>

## 🗣️ 🧮 📄 Réflexion

### Ex 5.8 — Le gradient expliqué avec de l'eau sur un drap

<details><summary>Indice 1</summary>

Reprends l'image du livre (§5.4) : un drap figé, de l'eau versée dessus. Que fait l'eau ? Qu'est-ce que le gradient, dans cette image ?

</details>
<details><summary>Indice 2</summary>

Une phrase par contrainte : la pente et sa direction, le pas répété, l'endroit où l'eau s'arrête, et une différence entre l'eau et l'algorithme (relis l'encadré ⚠️ « Les limites de l'image de l'eau » de la fiche).

</details>
<details><summary>Indice 3</summary>

L'eau s'arrête au fond du premier creux, qui n'est pas forcément le plus bas du drap. Différences possibles : l'eau coule en continu, l'algorithme fait des pas qui peuvent enjamber un creux ; l'eau a de l'élan, pas l'algorithme.

</details>

### Ex 5.9 — Fermi : le prix d'un gradient numérique pour un million de paramètres

<details><summary>Indice 1</summary>

Avance pas à pas : le coût d'une évaluation, puis le nombre d'évaluations, puis le temps.

</details>
<details><summary>Indice 2</summary>

Une évaluation sur le mini-batch : $64 \times 2P$ opérations. Une différence centrée par paramètre : 2 évaluations par paramètre. Le temps : le nombre d'opérations divisé par $10^{11}$.

</details>
<details><summary>Indice 3</summary>

Environ $10^8$ opérations par évaluation, $2 \times 10^6$ évaluations par gradient : de l'ordre de $2{,}6 \times 10^{14}$ opérations, soit environ 40 minutes par pas. La rétropropagation : trois évaluations, quelques millisecondes. Pour la question 6, le coût des différences finies est proportionnel à $P^2$, celui de la rétropropagation à $P$.

</details>

### Ex 5.10 — Dauphin et al. (2014) : les points selles en grande dimension

<details><summary>Indice 1</summary>

Le résumé répond déjà aux questions 1 et 4 ; la section 2 définit l'indice, la section 3 présente la figure 1.

</details>
<details><summary>Indice 2</summary>

L'indice compte les directions dans lesquelles la surface descend, en proportion. Pour la méthode de Newton, regarde ce que devient un pas qui divise le gradient par une courbure **négative**.

</details>
<details><summary>Indice 3</summary>

L'indice est la proportion de valeurs propres négatives de la matrice des dérivées secondes : 0 pour un minimum, 1 pour un maximum. La figure 1 place les points critiques dans le plan (erreur, indice) : plus l'erreur est grande, plus l'indice l'est. *Saddle-free Newton* divise par la **valeur absolue** des courbures.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 5.E1 — Descente de gradient : que se passe-t-il avec un learning rate trop grand, puis trop petit ?

<details><summary>Indice 1</summary>

Un plan en trois temps : l'algorithme en une phrase et sa formule, les deux cas extrêmes, puis comment on choisit en pratique.

</details>
<details><summary>Indice 2</summary>

Trop grand : que fait un pas qui dépasse le minimum (✏️ 5.3 e, 🔬 5.19) ? Trop petit : combien de pas faut-il ? Quelle direction fixe la limite sur un bol allongé ?

</details>
<details><summary>Indice 3</summary>

Trop grand : oscillation, puis divergence (la loss explose, parfois jusqu'à `nan`) ; trop petit : convergence très lente, arrêt sur un plateau. En pratique : essayer plusieurs valeurs sur une échelle logarithmique en traçant la loss, garder la plus grande qui reste stable, puis la faire décroître ; Adam rend le réglage moins délicat.

</details>

### 5.E2 — Minimum local : un vrai problème en deep learning ?

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ sur les points selles (fiche §5.4) et ta lecture de l'article 5.10.

</details>
<details><summary>Indice 2</summary>

En grande dimension, que faut-il pour qu'un point critique soit un minimum ? Quels obstacles ralentissent vraiment la descente ? Qu'est-ce qui aide à s'en sortir ?

</details>
<details><summary>Indice 3</summary>

Un minimum exige que la surface monte dans **toutes** les directions : à loss élevée, la plupart des points critiques sont des points selles. Les minima trouvés en pratique ont des loss voisines. Les vrais obstacles : plateaux, points selles, vallées étroites ; ce qui aide : le bruit des mini-batches, le momentum, l'initialisation. Termine par la généralisation.

</details>

### 5.E3 — Comment vérifier un gradient calculé ?

<details><summary>Indice 1</summary>

Le nom de la technique : le *gradient check*. Avec quoi compare-t-on le gradient calculé ?

</details>
<details><summary>Indice 2</summary>

Une différence centrée par coordonnée : quel pas, quelle précision de calcul, quelle mesure de l'écart ? Relis « Au-delà du livre (2) » et l'encadré 🕰️ de « Au-delà du livre (3) ».

</details>
<details><summary>Indice 3</summary>

`float64`, $h$ autour de $10^{-5}$, une erreur **relative** $\frac{|a - b|}{\max(|a|, |b|)}$ ; quelques coordonnées tirées au hasard plutôt que toutes ; attention aux points anguleux et au *dropout*. Cite `torch.autograd.gradcheck`.

</details>

### 5.E4 — Point selle : définition et effet sur l'optimisation

<details><summary>Indice 1</summary>

Une définition, un exemple, puis pourquoi on en parle pour les réseaux.

</details>
<details><summary>Indice 2</summary>

Définition par les directions (monte dans l'une, descend dans l'autre), puis par les dérivées secondes. Effet : que vaut le gradient autour du point selle, et que voit-on sur la courbe de loss ?

</details>
<details><summary>Indice 3</summary>

$x^2 - y^2$ en $(0, 0)$ ; la matrice des dérivées secondes a des valeurs propres des deux signes. Les points selles sont nombreux en grande dimension (Dauphin et coll., 2014) ; autour d'eux, un plateau ralentit la descente. Partie d'un point aléatoire, la descente ne s'y arrête presque jamais (Lee et coll., 2016), mais elle peut y perdre beaucoup de temps (Du et coll., 2017) ; le bruit aide.

</details>

<a id="notebook"></a>

## Notebook, parties A à D

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/calculus.py`, la docstring de chaque fonction décrit déjà ce qu'elle doit faire, cas d'erreur compris : relis-la avant d'ouvrir un indice.

### Ex 5.11 — Dérivées numériques : première et seconde 🔨

<details><summary>Indice 1</summary>

Trois étapes dans chaque fonction : contrôler les arguments, préparer `x` (un nombre ou un tableau), appliquer la formule. Écris d'abord `_check_step(h)` : les trois fonctions qui prennent un pas l'appelleront.

</details>
<details><summary>Indice 2</summary>

`np.ndim(x) == 0` vaut `True` pour `2`, `2.0` ou `np.float64(2.0)`, et `False` pour une liste ou un tableau. Pour un scalaire, travaille avec `float(x)` et renvoie `float(...)` ; sinon, `np.asarray(x, dtype=float)` (qui ne modifie rien : `x + h` crée un nouveau tableau). Les trois formules sont dans la docstring ; un `if`/`elif` choisit la bonne.

</details>
<details><summary>Indice 3</summary>

```python
def _check_step(h):
    if not h > 0:
        raise ValueError(f"h must be > 0, got {h!r}")


def numerical_derivative(f, x, h=1e-5, method="central"):
    _check_step(h)
    if method not in ("central", "forward", "backward"):
        raise ValueError(f"unknown method {method!r}")
    scalar = np.ndim(x) == 0
    x = float(x) if scalar else np.asarray(x, dtype=float)
    if method == "central":
        slope = (f(x + h) - f(x - h)) / (2 * h)
    elif method == "forward":
        slope = (f(x + h) - f(x)) / h
    else:
        slope = (f(x) - f(x - h)) / h
    return float(slope) if scalar else np.asarray(slope, dtype=float)


def second_derivative(f, x, h=1e-4):
    _check_step(h)
    scalar = np.ndim(x) == 0
    x = float(x) if scalar else np.asarray(x, dtype=float)
    curvature = (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2
    return float(curvature) if scalar else np.asarray(curvature, dtype=float)
```

</details>

### Ex 5.12 — Quel pas h choisir ? Prédire la courbe d'erreur 🔮

<details><summary>Indice 1</summary>

Deux erreurs se disputent : celle de la formule (la troncature, ∂ 5.6), qui diminue quand $h$ diminue, et celle des arrondis : un `float` ne garde qu'environ 16 chiffres, et le numérateur soustrait deux nombres presque égaux. Comment chacune varie-t-elle avec $h$ ?

</details>
<details><summary>Indice 2</summary>

Un `float` garde environ 16 chiffres. L'erreur d'arrondi d'une différence centrée est de l'ordre de $\frac{10^{-16}}{h}$ ; sa troncature, de l'ordre de $h^2$. Pour la différence avant, la troncature est de l'ordre de $h$. Pour quel $h$ chaque paire d'erreurs est-elle du même ordre ?

</details>
<details><summary>Indice 3</summary>

Centrée : $h^2 \approx \frac{10^{-16}}{h}$ donne $h^3 \approx 10^{-16}$, soit $h$ vers $10^{-5}$. Avant : $h \approx \frac{10^{-16}}{h}$ donne $h$ vers $10^{-8}$. Avec $h = 10^{-15}$, l'erreur d'arrondi est de l'ordre de $\frac{10^{-16}}{10^{-15}} \times |f(x)|$, à comparer à la troncature avec $h = 0{,}1$.

</details>

### Ex 5.13 — Erreur de troncature contre erreur d'arrondi 🔬

<details><summary>Indice 1</summary>

`errors_13` est une boucle (ou une compréhension de liste) sur `ks` : un appel à ta fonction `numerical_derivative` par pas, et la valeur absolue de l'écart à `df(x)`.

</details>
<details><summary>Indice 2</summary>

Le pas : `10.0 ** -k` (avec `10 ** -k`, NumPy refuse un exposant entier négatif). Pour e) et f), dérive $g(h) = h^2 + \frac{\varepsilon}{h}$ par rapport à $h$ et annule la dérivée ; même chose pour $h + \frac{\varepsilon}{h}$. Puis prends le logarithme décimal du résultat avec `np.log10`.

</details>
<details><summary>Indice 3</summary>

```python
def errors_13(f, df, x, ks, method):
    return np.array([abs(mylearn.calculus.numerical_derivative(f, x, h=10.0 ** -k, method=method) - df(x))
                     for k in ks])


eps = np.finfo(float).eps
log_h_central_13 = np.log10((eps / 2) ** (1 / 3))   # 2h - eps / h**2 = 0
log_h_forward_13 = np.log10(np.sqrt(eps))           # 1 - eps / h**2 = 0
```

</details>

### Ex 5.14 — Les maxima des cycles solaires 🔨

<details><summary>Indice 1</summary>

Pour chaque échantillon intérieur $i$ (de 1 à $n - 2$), rassemble ses voisins : jusqu'à `order` échantillons à gauche et à droite, en t'arrêtant aux bords du tableau. Puis compare.

</details>
<details><summary>Indice 2</summary>

Les tranches de NumPy s'arrêtent d'elles-mêmes au bout du tableau : `y[i + 1:i + order + 1]` ; à gauche, protège le début avec `max(0, i - order)`. `np.all(y[i] < voisins)` teste la condition **stricte** : sur un plateau, un voisin égal la fait échouer. Contrôle `order` (un entier au moins égal à 1 ; `True` est un `bool`, refuse-le) et la dimension de `y`, puis renvoie des tableaux d'entiers (`np.array(liste, dtype=int)`, même vides). Pour e), compte les valeurs de `smooth_14` égales à 0.

</details>
<details><summary>Indice 3</summary>

```python
def find_local_extrema(y, order=1):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f"order must be an integer >= 1, got {order!r}")
    minima, maxima = [], []
    for i in range(1, len(y) - 1):
        neighbours = np.concatenate([y[max(0, i - order):i], y[i + 1:i + order + 1]])
        if np.all(y[i] < neighbours):
            minima.append(i)
        if np.all(y[i] > neighbours):
            maxima.append(i)
    return np.array(minima, dtype=int), np.array(maxima, dtype=int)
```

Et pour e) : `zero_months_14 = int(np.sum(smooth_14 == 0))`.

</details>

### Ex 5.15 — numerical_gradient sur la vallée de Rosenbrock 🔨

<details><summary>Indice 1</summary>

Une copie en flottants, une boucle sur toutes les positions, deux évaluations de `f` par position. Le gradient a la forme de `x`.

</details>
<details><summary>Indice 2</summary>

`point = np.array(x, dtype=float)` copie toujours ; `flat = point.reshape(-1)` est une vue à plat : modifier `flat[i]` modifie `point`, quelle que soit sa forme. Même chose pour le gradient : `grad = np.zeros_like(point)` et `grad.reshape(-1)`. Pour chaque `i` : garde `old = flat[i]`, pose `old + h`, évalue, pose `old - h`, évalue, remets `old`.

</details>
<details><summary>Indice 3</summary>

```python
def numerical_gradient(f, x, h=1e-5):
    _check_step(h)
    point = np.array(x, dtype=float)       # a float COPY: the caller's x never changes
    flat = point.reshape(-1)               # a flat view: changing flat changes point
    grad = np.zeros_like(point)
    flat_grad = grad.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + h
        f_plus = f(point)
        flat[i] = old - h
        f_minus = f(point)
        flat[i] = old                      # put the old value back, exactly
        flat_grad[i] = (f_plus - f_minus) / (2 * h)
    return grad
```

</details>

### Ex 5.16 — Le gradient qui abîme son entrée 🐛

<details><summary>Indice 1</summary>

Deux questions à se poser sur la fonction du collègue : travaille-t-elle sur le tableau de l'appelant, ou sur une copie ? Et quel est le type des nombres de ce tableau ?

</details>
<details><summary>Indice 2</summary>

`np.asarray` renvoie le tableau lui-même quand c'est déjà un tableau NumPy, sans copie. Dans un tableau d'entiers, `x[i] += 1e-5` stocke un entier : la valeur est tronquée (arrondie vers 0). Pour c), après l'appel, regarde `point_16 == np.array([0.1, 0.2])`, ou `repr(point_16[0])`.

</details>
<details><summary>Indice 3</summary>

En b), $-1 + 10^{-5}$ est tronqué en 0, puis toutes les modifications suivantes restent à 0. En c), $0{,}1 + h - 2h + h$ ne redonne pas exactement 0,1 en flottants. La correction : remplacer `x = np.asarray(x)` par `x = np.array(x, dtype=float)` ; `np.zeros_like(x)` est alors un tableau de flottants, lui aussi.

</details>

### Ex 5.17 — Lire des lignes de niveau : où pointe le gradient ? 📈

<details><summary>Indice 1</summary>

Repère d'abord, grâce à la barre de couleurs, les zones hautes (claires) et basses (sombres). Un maximum est entouré de lignes fermées dans une zone claire ; un point selle est un croisement de lignes en X.

</details>
<details><summary>Indice 2</summary>

La pente est forte là où les lignes de niveau sont serrées. Le gradient est perpendiculaire aux lignes de niveau et pointe vers les valeurs plus hautes. Pour f), dérive $g$ par rapport à $x$, puis par rapport à $y$, en gardant les signes.

</details>
<details><summary>Indice 3</summary>

Le maximum et le minimum sont les deux centres de lignes fermées, l'un clair, l'autre sombre ; les deux autres points critiques sont des croisements. En D, les couleurs claires sont vers le haut et vers la gauche. Pour f), $\frac{\partial g}{\partial x} = 3x^2 - 3$ et $\frac{\partial g}{\partial y} = -3y^2 + 3$ : remplace $x$ par 1,8 et $y$ par $-0{,}2$.

</details>

### Ex 5.18 — gradient_descent, et sa version qui monte 🔨

<details><summary>Indice 1</summary>

Contrôle les arguments, fais une copie en flottants de `x0`, puis une boucle d'au plus `n_steps` tours ; garde chaque point visité dans une liste.

</details>
<details><summary>Indice 2</summary>

Dans la boucle : `g = np.asarray(grad(x), dtype=float)` ; si `tol` n'est pas `None` et `np.linalg.norm(g) < tol`, `break` **avant** de bouger ; sinon `x = x - lr * g` (ou `+` si `maximize`), et `path.append(x.copy())`. Renvoie `x` et `np.array(path)`.

</details>
<details><summary>Indice 3</summary>

```python
def gradient_descent(grad, x0, lr=0.01, n_steps=100, tol=None, maximize=False):
    if not lr > 0:
        raise ValueError(f"lr must be > 0, got {lr!r}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps!r}")
    if tol is not None and tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol!r}")
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    sign = 1.0 if maximize else -1.0
    for _ in range(int(n_steps)):
        g = np.asarray(grad(x), dtype=float)
        if tol is not None and np.linalg.norm(g) < tol:
            break
        x = x + sign * lr * g
        path.append(x.copy())
    return x, np.array(path)
```

</details>

### Ex 5.19 — Learning rate sur un bol : trop petit, juste, trop grand 🔬

<details><summary>Indice 1</summary>

`distances_19` appelle ta fonction `gradient_descent`, garde le chemin, et mesure la distance de chaque point à $(0, 0)$.

</details>
<details><summary>Indice 2</summary>

Le chemin a la forme `(51, 2)` : `np.linalg.norm(path, axis=1)` donne les 51 distances. Pour les questions, calcule les deux facteurs $1 - 2\eta$ et $1 - 20\eta$ pour chaque learning rate : la coordonnée converge si son facteur est strictement entre $-1$ et 1.

</details>
<details><summary>Indice 3</summary>

```python
def distances_19(lr):
    _, path = mylearn.calculus.gradient_descent(grad_bowl_19, START_19, lr=lr, n_steps=50)
    return np.linalg.norm(path, axis=1)
```

La vitesse est fixée par le plus grand des deux facteurs en valeur absolue. Avec 0,01, $v_0$ est multiplié par 0,98 à chaque pas : cherche $t$ tel que $2 \times 0{,}98^t < 0{,}01$.

</details>

### Ex 5.20 — Démarrer pile sur un point selle 🔮

<details><summary>Indice 1</summary>

Relis la règle de mise à jour : de quoi dépend la taille d'un pas ? Que se passe-t-il là où cette quantité est nulle, ou minuscule ?

</details>
<details><summary>Indice 2</summary>

Pour un $y$ minuscule, compare $y^3$ et $y$ : un pas fait-il grandir ou diminuer $|y|$, et de combien, en proportion ? Le signe de $y$ peut-il changer ?

</details>
<details><summary>Indice 3</summary>

Près de l'axe, un pas fait $y \leftarrow y - \eta\,(y^3 - y) \approx (1 + \eta)\,y = 1{,}1\,y$ : le signe de $y$ ne change pas, et $|y|$ grandit de 10 % par pas. $1{,}1^n = 5 \times 10^5$ donne $n = \frac{\ln(5 \times 10^5)}{\ln 1{,}1}$, de l'ordre de la centaine (la croissance ralentit un peu quand $y$ approche de 1). Pour d), il faut gagner un facteur $10^6$ de plus : $n = \frac{\ln 10^6}{\ln 1{,}1}$.

</details>

### Ex 5.21 — Le même gradient avec torch.autograd 📦

<details><summary>Indice 1</summary>

`rosenbrock_torch` ressemble à la formule de Rosenbrock écrite en Python, avec `p[0]` et `p[1]` à la place de `x` et `y`. N'appelle aucune fonction NumPy (ni `wb.synth.rosenbrock`, qui en utilise).

</details>
<details><summary>Indice 2</summary>

Les opérateurs `-`, `*` et `**` marchent sur les tenseurs. Pour b), `torch_gradient(lambda p: torch.relu(p[0]), [0.0])` donne la pente de ReLU en 0 ; fais de même avec `torch.clamp(p[0], min=0)` et `torch.maximum(p[0], torch.zeros_like(p[0]))`, puis appelle ta `numerical_derivative` sur `lambda x: max(0.0, x)`.

</details>
<details><summary>Indice 3</summary>

```python
def rosenbrock_torch(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


corner_21 = [float(torch_gradient(lambda p: torch.relu(p[0]), [0.0])[0]),
             float(torch_gradient(lambda p: torch.clamp(p[0], min=0), [0.0])[0]),
             float(torch_gradient(lambda p: torch.maximum(p[0], torch.zeros_like(p[0])), [0.0])[0]),
             mylearn.calculus.numerical_derivative(lambda x: max(0.0, x), 0.0)]
```

</details>

### Ex 5.22 — L'eau qui descend la surface (figure 5.18) 🎨

<details><summary>Indice 1</summary>

Deux fonctions de dessin indépendantes, qui reçoivent l'axe et le chemin. Pour la surface, une grille de points (`np.meshgrid`), puis les hauteurs ; pour la carte, la fonction `wb.plot.plot_contour` fait presque tout.

</details>
<details><summary>Indice 2</summary>

Hauteurs : `np.log10(1 + wb.synth.rosenbrock(X, Y))` pour la surface, et la même formule pour les points du chemin (`path[:, 0]`, `path[:, 1]`). Une grille de 60 × 60 points suffit et dessine vite. Flèches : `path[::300]` donne un point sur 300 ; calcule $-\nabla f$ en chacun (`rosenbrock_gradient`), divise chaque flèche par sa norme (`np.linalg.norm(..., axis=1, keepdims=True)`), puis `ax.quiver(xs, ys, us, vs)`.

</details>
<details><summary>Indice 3</summary>

```python
def draw_surface_22(ax, path):
    X, Y = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-1, 3, 60))
    ax.plot_surface(X, Y, np.log10(1 + wb.synth.rosenbrock(X, Y)), cmap="viridis", alpha=0.6, linewidth=0)
    path = np.asarray(path)
    heights = np.log10(1 + wb.synth.rosenbrock(path[:, 0], path[:, 1]))
    ax.plot(path[:, 0], path[:, 1], heights, color="red", lw=2)
    ax.scatter(path[0, 0], path[0, 1], heights[0], color="red", s=40)
    ax.scatter(path[-1, 0], path[-1, 1], heights[-1], color="black", marker="s", s=40)


def draw_map_22(ax, path):
    path = np.asarray(path)
    wb.plot.plot_contour(wb.synth.rosenbrock, xlim=(-2, 2), ylim=(-1, 3), path=path, ax=ax, minimum=(1, 1))
    marks = path[::300]
    arrows = -np.array([rosenbrock_gradient(point) for point in marks])
    arrows = arrows / np.linalg.norm(arrows, axis=1, keepdims=True)
    ax.quiver(marks[:, 0], marks[:, 1], arrows[:, 0], arrows[:, 1], color="white")
```

</details>

### Ex 5.23 — Tests de propriétés paramétrés avec pytest 🛠️

<details><summary>Indice 1</summary>

Lis les trois versions buggées : chacune échoue sur une propriété différente. Écris un test par propriété de l'énoncé.

</details>
<details><summary>Indice 2</summary>

Bug 1 : la différence avant, fausse d'environ $10^{-5}$ sur une forme quadratique, alors que la différence centrée y est exacte aux arrondis près : une tolérance de $10^{-6}$ l'attrape. Bug 2 : un point **entier** dont les coordonnées ne sont pas nulles. Bug 3 : un point **matrice**, et la forme du résultat. Le décorateur : `@pytest.mark.parametrize("n, seed", [(1, 0), (2, 1), (5, 2)])` au-dessus d'une fonction `test_...(n, seed)`.

</details>
<details><summary>Indice 3</summary>

```python
@pytest.mark.parametrize("n, seed", [(1, 0), (2, 1), (5, 2)])
def test_matches_the_gradient_of_a_quadratic_form(n, seed):
    rng = np.random.default_rng(seed)
    A, b, x = rng.normal(size=(n, n)), rng.normal(size=n), rng.normal(size=n)
    result = numerical_gradient(lambda v: v @ A @ v + b @ v, x)
    assert np.allclose(result, (A + A.T) @ x + b, atol=1e-6)


def test_an_integer_point_gives_the_same_gradient():
    def f(v):
        return v[0] ** 2 * v[1] + 3 * v[1]
    assert np.allclose(numerical_gradient(f, np.array([-1, 2])), numerical_gradient(f, np.array([-1.0, 2.0])),
                       atol=1e-6)
```

Il te reste le troisième test (la matrice), puis `my_tests_23 = [...]`.

</details>

### Ex 5.24 — Minimum, maximum, selle ou plat : classify_critical_point 🔨

<details><summary>Indice 1</summary>

Trois étapes : contrôler `x` (une dimension, un point critique), construire la liste des directions, calculer une différence seconde par direction et décider d'après leurs signes.

</details>
<details><summary>Indice 2</summary>

`np.eye(n)[i]` est $\mathbf{e}_i$ ; deux boucles `for i in range(n)` et `for j in range(i + 1, n)` ajoutent $\mathbf{e}_i + \mathbf{e}_j$ et $\mathbf{e}_i - \mathbf{e}_j$. Calcule $f(\mathbf{x})$ une seule fois. Puis `up = second > tol` et `down = second < -tol` (des tableaux de booléens) : selle si `up.any() and down.any()`, minimum si `up.all()`, maximum si `down.all()`, sinon plat. L'ordre des tests compte-t-il ?

</details>
<details><summary>Indice 3</summary>

```python
def classify_critical_point(f, x, h=1e-3, tol=1e-6, grad_tol=1e-4):
    point = np.array(x, dtype=float)
    if point.ndim != 1:
        raise ValueError("x must be a 1-D array")
    norm = np.linalg.norm(numerical_gradient(f, point))
    if norm > grad_tol:
        raise ValueError(f"x is not a critical point: the norm of the gradient is {norm:.3g}")
    n = point.size
    eye = np.eye(n)
    directions = [eye[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            directions += [eye[i] + eye[j], eye[i] - eye[j]]
    f0 = f(point)
    second = np.array([(f(point + h * d) - 2 * f0 + f(point - h * d)) / h ** 2 for d in directions])
    up, down = second > tol, second < -tol
    if up.any() and down.any():
        return "saddle"
    if up.all():
        return "minimum"
    if down.all():
        return "maximum"
    return "flat"
```

</details>

### Ex 5.25 — Atteindre le fond de la vallée de Rosenbrock 🏆

<details><summary>Indice 1</summary>

Commence par mesurer : pour quelques learning rates, combien de pas faut-il pour arriver à moins de $10^{-3}$ de $(1, 1)$ ? Écris une petite fonction qui le calcule à partir du chemin de `gradient_descent`.

</details>
<details><summary>Indice 2</summary>

`np.linalg.norm(path - np.array([1.0, 1.0]), axis=1) < 1e-3`, puis le premier indice `True` (`np.flatnonzero`). Essaie 0,001, 0,0015, 0,0018, 0,0019, 0,002. Pour chaque candidat qui arrive à temps, vérifie aussi qu'il **reste** : `grade_25([(lr, n)])` te renvoie le nombre de pas, la distance finale, puis la pire distance pendant les 10 000 pas suivants et la distance à leur fin.

</details>
<details><summary>Indice 3</summary>

Au fond de la vallée, la courbure la plus forte vaut environ 1 000 : au-delà de $\frac{2}{1\,000}$, la descente oscille en travers de la vallée sans s'amortir. Un learning rate un peu en dessous, vers 0,0019, arrive à temps et reste. Un programme en deux phases, rapide puis prudent, est aussi possible.

</details>
