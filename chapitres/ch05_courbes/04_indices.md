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

1. La dérivée (une variable) et le gradient (plusieurs) mesurent la **pente locale** : de combien la sortie change quand on bouge un peu l'entrée, donc dans quel sens aller pour monter ou pour descendre. Pour les autres : 2. ce que l'entraînement fait varier, ce sont les entrées de la surface ; ce qu'il cherche à rendre petit, c'est sa sortie ; 3. le nom de l'algorithme est dans la fiche §5.1, avec son nom anglais ; 4. relis le début du livre (§5.1) : qu'annonce l'auteur sur les équations ? 5. compte les combinaisons à essayer avec des millions de poids, puis demande-toi ce que la pente t'apprend à chaque pas, et que le hasard ne t'apprend pas.

</details>

### 5.Q2 — Une fonction, une table d'entrées et de sorties

<details><summary>Indice 1</summary>

Relis le §5.2 de la fiche : la fonction vue comme une table, et ce qui change entre une courbe et une surface.

</details>
<details><summary>Indice 2</summary>

Compte les nombres qu'on donne et ceux qu'on reçoit. Pour la question 4, cherche dans le livre la condition « tant que… » ; puis pense aux étapes d'un entraînement qui tirent quelque chose au hasard (0A).

</details>
<details><summary>Indice 3</summary>

1. Une courbe $y = f(x)$ reçoit un nombre et en rend un : 1 entrée, 1 sortie ; une surface $z = f(x, y)$ en reçoit deux et en rend un : 2 entrées, 1 sortie. Compte de la même façon pour les autres : 2. combien de nombres le modèle reçoit-il, et combien en rend-il ? Compare avec les deux cas de 1 ; 3. qu'est-ce qu'on change quand on entraîne (ce sont les entrées), et combien de nombres la loss renvoie-t-elle ? 4. relis la phrase du livre (§5.2) qui pose la condition « tant que… » ; puis cherche, dans une boucle d'entraînement (0A, ch. 1), une étape où le programme tire quelque chose au sort exprès ; 5. combien de valeurs différentes une entrée réelle peut-elle prendre ?

</details>

### 5.Q3 — Continue, lisse, univoque : reconnaître les courbes

<details><summary>Indice 1</summary>

Relis la liste des quatre règles du §5.2 de la fiche, et regarde la figure `courbes_interdites.png`.

</details>
<details><summary>Indice 2</summary>

Dessine chaque courbe à main levée. Peux-tu la tracer sans lever le crayon ? A-t-elle un coin ? Une droite verticale la coupe-t-elle deux fois ? Sa tangente devient-elle verticale quelque part ?

</details>
<details><summary>Indice 3</summary>

1. La partie entière vaut 0 sur $[0 ; 1[$, puis 1 dès $x = 1$ : il faut lever le crayon, elle n'est **pas continue**. Applique les mêmes tests aux autres, un par un : 2. compare la pente juste à gauche et juste à droite de $x = 1$ ; 3. pour un $x$ de $]-2 ; 2[$, résous $x^2 + y^2 = 4$ en $y$ : combien de solutions ? 4. calcule la pente de la sécante symétrique en 0, $\frac{\sqrt[3]{h} - \sqrt[3]{-h}}{2h}$, pour $h = 0{,}001$ puis $h = 0{,}000\,001$ : que devient-elle quand $h$ diminue ? 5. $x^2$ et $\sin x$ ont-elles un saut, un coin, deux valeurs ou une pente infinie quelque part ? Et leur somme ? 6. Que doit connaître l'algorithme, en chaque point, pour faire son pas, et que lui arrive-t-il à un saut, à un coin ou devant une pente infinie ?

</details>

### 5.Q4 — Minimum local ou minimum global ?

<details><summary>Indice 1</summary>

Relis « Extrema globaux et locaux » dans la fiche (§5.3), et l'encadré ⚠️ sur la phrase du livre.

</details>
<details><summary>Indice 2</summary>

Distingue la **valeur** d'un extremum et les **points** où il est atteint. Pour $(x^2 - 4)^2$, cherche où elle s'annule et ce qu'elle vaut en 0. Sur un intervalle fermé, n'oublie pas les bornes.

</details>
<details><summary>Indice 3</summary>

1. $\cos x \le 1$ partout, et $\cos x = 1$ en $x = 0$, $\pm 2\pi$, $\pm 4\pi$… : le maximum global vaut 1, atteint en une infinité de points. Pour les autres : 2. sépare, dans ta phrase, la **valeur** du maximum et les **points** où il est atteint (ta réponse 1 sert de contre-exemple) ; 3. que devient $x^3$ quand $x$ tend vers $-\infty$ ? Puis compare $x^3$, juste à gauche et juste à droite de 0, à sa valeur en 0 ; 4. $(x^2 - 4)^2$ est un carré : quelle est sa plus petite valeur possible, et pour quels $x$ ? Calcule ensuite $f(0)$, $f(-0{,}5)$ et $f(0{,}5)$, et regarde ce que fait $f$ quand $|x|$ grandit ; 5. $x^2 - 2x = (x - 1)^2 - 1$ : compare la valeur en 1 et les valeurs aux bornes, en 0 et en 3 ; 6. relis « zones d'influence » (fiche §5.3) : où s'arrête une descente ?

</details>

### 5.Q5 — Le signe de la dérivée indique le chemin

<details><summary>Indice 1</summary>

Relis « Le signe de la dérivée » et l'encadré ⚠️ « Pas fixe ou pas proportionnel » dans la fiche (§5.3).

</details>
<details><summary>Indice 2</summary>

Une dérivée négative veut dire que la courbe descend vers la droite : de quel côté monte-t-elle ? Pour la question 4, imagine qu'on est à une demi-longueur de pas du sommet : où arrive-t-on au pas suivant ?

</details>
<details><summary>Indice 3</summary>

1. On monte dans le sens du **signe** de la dérivée : $\mathrm{sign}(-3) = -1$, il faut aller vers la **gauche**. Pour les autres : 2. même raisonnement, dans le sens opposé au signe ; 3. applique la définition de la fonction signe (fiche §5.3) à chacun des trois nombres, sans oublier le cas de 0 ; 4. place-toi à une demi-longueur de pas du sommet, et fais deux pas de la règle : où arrives-tu à chaque fois ? 5. écris la longueur d'un pas, $\eta\,|f'(x)|$ : que devient $f'(x)$ à l'approche du minimum ? 6. compare une droite qui monte de 3 par unité et une autre qui monte de 0,1 : laquelle est la plus raide, et laquelle a la plus grande dérivée ?

</details>

### 5.Q6 — Dérivée nulle : sommet, creux ou plateau ?

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la dérivée seconde et l'encadré ⚠️ « Plateau » de la fiche (§5.3).

</details>
<details><summary>Indice 2</summary>

Pour chaque fonction, regarde son signe juste à gauche et juste à droite de 0, comparé à $f(0)$. Calcule ensuite $f''(0)$ : quand vaut-elle 0 ?

</details>
<details><summary>Indice 3</summary>

1. $-x^2 \le 0 = f(0)$ pour tout $x$ : aucune valeur ne dépasse $f(0)$, c'est un **maximum**. Pour 2 à 4, compare de même $f(x)$ et $f(0)$, juste à gauche et juste à droite de 0 : toujours au-dessus, toujours au-dessous, ou au-dessous d'un côté et au-dessus de l'autre ? Pour la constante, pense aux inégalités larges ($\ge$ et $\le$). 5. Dérive deux fois chacune des trois premières, remplace $x$ par 0, puis relis l'encadré 🧮 : quel signe de $f''(0)$ permet de conclure, et lequel ne dit rien ? 6. Le nom est dans l'encadré ⚠️ « plateau » de la fiche (§5.3).

</details>

### 5.Q7 — Le gradient : une direction et une longueur

<details><summary>Indice 1</summary>

Relis le §5.4 de la fiche et l'encadré 🧮 « La pente dans une direction ».

</details>
<details><summary>Indice 2</summary>

$f$ est une fonction affine : ses dérivées partielles sont des constantes. La pente dans la direction $\mathbf{u}$ est le produit scalaire $\nabla f \cdot \mathbf{u}$ ; elle est nulle quand $\mathbf{u}$ est perpendiculaire au gradient.

</details>
<details><summary>Indice 3</summary>

1. $\frac{\partial f}{\partial x} = 3$ et $\frac{\partial f}{\partial y} = -4$, quels que soient $x$ et $y$ : $\nabla f = (3, -4)$ en tout point. Pour les autres : 2. la norme de $(a, b)$ vaut $\sqrt{a^2 + b^2}$ ; 3. divise le gradient par sa norme ; 4. calcule le produit scalaire $\nabla f \cdot (0, 1)$ ; 5. un vecteur perpendiculaire à $(a, b)$ est $(b, -a)$ ou $(-b, a)$ : prends-en un, divise-le par sa norme, puis relis l'encadré 🧮 de la fiche (pente nulle et lignes de niveau) ; 6. relis le même encadré : dans quelle direction la pente est-elle la plus négative ?

</details>

### 5.Q8 — Point selle : un gradient nul sans extremum

<details><summary>Indice 1</summary>

Relis « Les points où le gradient s'annule » dans la fiche (§5.4), avec son mini-exemple sur $xy$.

</details>
<details><summary>Indice 2</summary>

Pour $y^2 - x^2$, regarde $f(x, 0)$ puis $f(0, y)$. Pour $xy$, essaie les diagonales $y = x$ et $y = -x$. Pour la question 4, que vaut le gradient en $(0, 0)$ : peut-il désigner une direction ? Pour la question 5, relis l'encadré 🕰️ du §5.4 de la fiche.

</details>
<details><summary>Indice 3</summary>

1. Un point selle est un point où le gradient est **nul**, mais qui n'est **ni un maximum ni un minimum** : la surface y monte dans certaines directions et descend dans d'autres, comme une selle de cheval. Pour les autres : 2. calcule les deux dérivées partielles en $(0, 0)$, puis écris $f(x, 0)$ et $f(0, y)$ : une bosse ou un creux ? 3. ne t'arrête pas aux axes : écris $f(x, x)$ et $f(x, -x)$, et compare-les à $f(0, 0)$ ; 4. pour quels points proches de $(0, 0)$ a-t-on $y^2 - x^2 < 0$ ? Compare $|y|$ et $|x|$, puis traduis-le en angle avec l'axe des $x$ ; et le gradient, nul en $(0, 0)$, peut-il désigner une direction ? 5. relis l'encadré 🕰️ du §5.4 : en grande dimension, que faut-il pour qu'un point critique soit un minimum, et que trouve-t-on surtout là où la loss est élevée ? 6. sur une courbe, combien de sens de déplacement y a-t-il ? Cherche, parmi les fonctions de Q6, celle dont la pente s'annule en un point d'où elle monte d'un côté et descend de l'autre.

</details>

### 5.Q9 — La fonction max(0, x) a un coin en 0 : est-ce grave ?

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §5.2 de la fiche, sur ReLU.

</details>
<details><summary>Indice 2</summary>

À gauche de 0, $\max(0, x) = 0$ ; à droite, $\max(0, x) = x$. Pour la question 3, calcule $\mathrm{relu}(h)$ et $\mathrm{relu}(-h)$ pour $h > 0$. Pour la question 5, à quelle fréquence une entrée réelle tombe-t-elle **exactement** sur 0 ?

</details>
<details><summary>Indice 3</summary>

1. Juste à gauche de 0, $\mathrm{relu}(x) = 0$, une constante : pente **0** ; juste à droite, $\mathrm{relu}(x) = x$ : pente **1**. Pour les autres : 2. parmi les quatre règles de la fiche (§5.2), laquelle parle d'un changement brusque de direction ? 3. pour $h > 0$, $\mathrm{relu}(h) = h$ et $\mathrm{relu}(-h) = 0$ : remplace dans la fraction et simplifie ; 4. l'encadré 🕰️ du §5.2 cite la documentation de PyTorch : quelle règle choisit une pente parmi celles qui sont comprises entre la pente de gauche et celle de droite, et laquelle désigne-t-elle ici ? 5. pour la théorie, à quelle fréquence une entrée réelle tombe-t-elle **exactement** sur 0 ? Pour l'expérience, relis ce que l'encadré dit du `float32`, de la descente de gradient simple, de la batchnorm et d'Adam ; 6. cherche, parmi les losses, les pénalités ou les couches que tu connais, une formule qui contient une valeur absolue ou un maximum.

</details>

### 5.Q10 — Ce que fait un réseau quand il « descend le gradient »

<details><summary>Indice 1</summary>

Relis le §5.1 et la fin du §5.4 de la fiche (« Descendre une surface pas à pas »).

</details>
<details><summary>Indice 2</summary>

Dans la mise à jour, quel signe faut-il pour **descendre** ? Pour la question 3, combien d'évaluations de la loss coûte une différence centrée par poids (🧮 5.9) ? Pour la question 4, souviens-toi des points où le gradient s'annule.

</details>
<details><summary>Indice 3</summary>

1. Le point qui se déplace, c'est le réseau lui-même : ses coordonnées sont ses **poids** (et ses biais), son altitude est la **loss**. Pour les autres : 2. pars de la formule de la fiche, $\mathbf{x} \leftarrow \mathbf{x} - \eta\,\nabla f(\mathbf{x})$, et écris-la pour un seul poids $w$ ; 3. compte : combien d'évaluations de la loss par poids pour une différence centrée, combien de poids, donc combien d'évaluations par pas (🧮 5.9) ? 4. que devient un pas, $\eta\,\left|\frac{\partial L}{\partial w}\right|$, quand le gradient est presque nul ? Quels points de la fiche §5.4 ont un gradient nul sans être un minimum ? 5. relis « zones d'influence » (§5.3) : une descente sait-elle où est le point le plus bas ? 6. relis les liens 🔗 de la fiche : quel chapitre calcule le gradient, lequel compare les façons de descendre ?

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

1. $P(D) = 0{,}6 \times 0{,}02 + 0{,}4 \times 0{,}05 = 0{,}012 + 0{,}020 = 0{,}032$ : c'est l'**évidence**, le dénominateur de la règle de Bayes. Pour les autres : 2. divise le terme de B, $P(D \mid B)\,P(B)$, par cette évidence ; 3. refais les deux calculs avec 0,5 et 0,5, puis compare ; 4. relis ∂ 4.7 : que devient un prior nul ?

</details>

### 5.R2 — Ch. 2 : où la courbe en cloche atteint-elle son maximum ?

<details><summary>Indice 1</summary>

La règle de la chaîne : la dérivée de $e^{u(x)}$ est $u'(x)\,e^{u(x)}$.

</details>
<details><summary>Indice 2</summary>

Ici $u(x) = -\frac{(x - \mu)^2}{2\sigma^2}$, donc $u'(x) = -\frac{x - \mu}{\sigma^2}$. L'exponentielle est toujours positive : le signe de $f'$ est celui de $u'$.

</details>
<details><summary>Indice 3</summary>

1. Avec $u(x) = -\frac{(x - \mu)^2}{2\sigma^2}$, la règle de la chaîne donne $f'(x) = \frac{1}{\sigma\sqrt{2\pi}}\,u'(x)\,e^{u(x)} = -\frac{x - \mu}{\sigma^2}\,f(x)$. Pour les autres : 2. $f(x) > 0$ partout, donc $f'(x)$ a le signe de $-(x - \mu)$ : étudie ce signe avant et après le point où il s'annule ; 3. remplace $x$ par $\mu$ dans $f$ (l'exponentielle vaut alors $e^0$), puis $\sigma$ par 2 ; 4. regarde où $\sigma$ apparaît dans la formule de 3 ; 5. dérive $f'(x) = -\frac{x - \mu}{\sigma^2}\,f(x)$ comme un produit, mets $f(x)$ en facteur, puis cherche où le facteur restant s'annule.

</details>

### 5.R3 — 0B : règle de la chaîne, dériver (3x + 1)²

<details><summary>Indice 1</summary>

La dérivée de $u(x)^2$ est $2\,u(x)\,u'(x)$.

</details>
<details><summary>Indice 2</summary>

Avec $u(x) = 3x + 1$, $u'(x) = 3$. Pour vérifier, développe $(3x + 1)^2 = 9x^2 + 6x + 1$ et dérive terme à terme.

</details>
<details><summary>Indice 3</summary>

1. $g'(x) = 2\,(3x + 1) \times 3 = 6\,(3x + 1)$. Pour les autres : 2. dérive terme à terme le développement de l'indice 2, puis compare ; 3. remplace $x$ par 1 dans $g'$ ; 4. résous $6\,(3x + 1) = 0$, calcule $g$ en ce point, et souviens-toi qu'un carré n'est jamais négatif ; 5. $x_1 = 0 - 0{,}05 \times g'(0)$, puis compare $g(x_1)$ et $g(0)$.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 5.1 — La sécante qui se resserre sur la tangente ✏️

<details><summary>Indice 1</summary>

La pente d'une droite qui passe par deux points de la courbe : la différence des hauteurs divisée par la différence des abscisses. Pour la sécante symétrique, les abscisses sont $2 - h$ et $2 + h$ : elles sont à $2h$ l'une de l'autre.

</details>
<details><summary>Indice 2</summary>

Avec $f(x) = \frac{1}{x}$ : $\frac{f(2 + h) - f(2 - h)}{2h}$. Mets les deux fractions au même dénominateur, $(2 + h)(2 - h) = 4 - h^2$, pour obtenir une expression simple en $h$ (question d). Pour f), écris aussi la pente avant comme une fraction simple de $h$, puis calcule les deux écarts avec les fractions exactes.

</details>
<details><summary>Indice 3</summary>

a) Les points d'abscisses 1 et 3 ont pour hauteurs $f(1) = 1$ et $f(3) = \frac{1}{3}$ : la pente vaut $\frac{\frac{1}{3} - 1}{3 - 1} = -\frac{1}{3} \approx -0{,}333$. Pour les autres : b) et c) même calcul, avec les abscisses $2 - h$ et $2 + h$, et une division par $2h$ ; d) au même dénominateur, $\frac{1}{2 + h} - \frac{1}{2 - h} = \frac{-2h}{4 - h^2}$ : divise par $2h$, puis fais tendre $h$ vers 0 ; e) entre 2 et $2 + h$, la pente avant se simplifie de même en une fraction de $h$ : remplace ensuite $h$ par 0,1 ; f) écris les deux écarts à $f'(2)$ comme des fractions de $h$, divise l'un par l'autre et simplifie (avec $4 - h^2 = (2 - h)(2 + h)$), et seulement alors remplace $h$ par 0,1.

</details>

### Ex 5.2 — Gradient à la main et direction de plus grande pente ✏️

<details><summary>Indice 1</summary>

Le gradient rassemble les dérivées partielles : dérive par rapport à $x$ en traitant $y$ comme une constante, puis l'inverse. Le terme $xy$ compte dans les deux.

</details>
<details><summary>Indice 2</summary>

Pour $\frac{\partial f}{\partial x}$, $y$ est une constante : $xy$ se dérive en $y$, et $2y^2$ en 0. La plus grande descente va contre le gradient ; pour la rendre unitaire, divise par la norme. La pente dans la direction $\mathbf{u}$ est $\nabla f \cdot \mathbf{u}$ (fiche, 🧮).

</details>
<details><summary>Indice 3</summary>

a) $f(1, -1) = 1^2 + 1 \times (-1) + 2 \times (-1)^2 = 1 - 1 + 2 = 2$. Pour les autres : b) $\frac{\partial f}{\partial x} = 2x + y$ et $\frac{\partial f}{\partial y} = x + 4y$, à évaluer en $P$ ; c) la norme de $(a, b)$ vaut $\sqrt{a^2 + b^2}$ ; d) prends l'opposé du gradient, puis divise-le par sa norme ; e) $\nabla f(P) \cdot \mathbf{u} = \frac{\partial f}{\partial x}(P) \times 0{,}6 + \frac{\partial f}{\partial y}(P) \times 0{,}8$ ; f) un vecteur perpendiculaire à $(a, b)$ est $(b, -a)$ ou $(-b, a)$ : choisis celui de première composante positive, puis divise-le par sa norme.

</details>

### Ex 5.3 — Trois pas de descente de gradient à la main ✏️

<details><summary>Indice 1</summary>

Applique la formule pas à pas : $x_1 = x_0 - \eta\,f'(x_0)$, puis recommence depuis $x_1$. Pour la montée, le signe change.

</details>
<details><summary>Indice 2</summary>

Dérive $2(x - 1)^2$ avec la règle de la chaîne, sans oublier le facteur 2 qui est devant. Pour f), remplace $f'(x_t)$ par son expression dans la formule et retranche 1 des deux côtés. La suite $x_t - 1$ est alors multipliée par le même nombre à chaque pas : quand se rapproche-t-elle de 0 ?

</details>
<details><summary>Indice 3</summary>

a) $f'(x) = 2 \times 2(x - 1) = 4(x - 1)$, donc $f'(5) = 4 \times 4 = 16$. Pour les autres : b) $x_1 = 5 - 0{,}125 \times f'(5)$, puis recommence depuis $x_1$, puis depuis $x_2$ ; c) calcule $f$ en $x_0$ et en chacun de ces points ; d) même méthode, avec $g'(x) = 4 - 2x$ et le signe $+$ ; e) et f) remplace $f'(x_t)$ par $4(x_t - 1)$ et retranche 1 des deux côtés : $x_{t+1} - 1 = (1 - 4\eta)(x_t - 1)$. La distance au minimum diminue à chaque pas si et seulement si $|1 - 4\eta| < 1$ : résous cette double inégalité. Pour la fin de e), calcule ce facteur avec $\eta = 0{,}5$ : que fait une multiplication répétée par ce nombre ? g) Le minimum est atteint en un seul pas quand ce facteur est nul.

</details>

### Ex 5.4 — Tableau de variations : extrema locaux et globaux de x³ − 3x ✏️

<details><summary>Indice 1</summary>

Dérive deux fois : le signe de $f'$ donne le sens de variation, celui de $f''$ la courbure (fiche, 🧮 dérivée seconde).

</details>
<details><summary>Indice 2</summary>

Factorise $f'(x)$ pour trouver ses zéros. Sur un intervalle fermé, le maximum global est soit un maximum local, soit une valeur aux bornes : calcule $f$ aux points critiques et aux deux bornes, puis compare. Même chose sur $[-2 ; 2]$.

</details>
<details><summary>Indice 3</summary>

a) $f'(x) = 3x^2 - 3 = 3(x - 1)(x + 1)$ s'annule en $x = -1$ et en $x = 1$ : la liste est $[-1, 1]$. Pour les autres : b) remplace $x$ par $-1$, puis par 1, dans $x^3 - 3x$ ; c) $f''(x) = 6x$, au plus petit des deux points, puis relis l'encadré 🧮 (dérivée seconde) pour interpréter son signe ; d) et e) compare la valeur au maximum local et les valeurs aux deux bornes, $f(-2{,}5)$ et $f(2{,}5)$ : garde la plus grande, et son abscisse ; f) même comparaison sur $[-2 ; 2]$, puis compte les points qui atteignent la valeur maximale ; g) résous $f''(x) = 0$ ; h) le signe de $f'$ sur chaque intervalle donne le tableau ; depuis $x = 0{,}5$, calcule le signe de $f'(0{,}5)$, puis suis la marche du livre de chaque côté, jusqu'au changement de sens.

</details>

### Ex 5.5 — Point selle : x² − y² vu dans deux directions ✏️

<details><summary>Indice 1</summary>

Calcule les deux dérivées partielles, puis évalue-les en $(0, 0)$. Pour b), restreins $f$ à chaque axe : tu obtiens une fonction d'une seule variable $t$.

</details>
<details><summary>Indice 2</summary>

Pour c), développe $f(t\cos\theta, t\sin\theta)$ : c'est $t^2$ fois un nombre qui ne dépend que de $\theta$, qu'une formule de trigonométrie de 0B simplifie. Pour e), écris un pas pour chaque coordonnée : chacune est multipliée par un facteur fixe.

</details>
<details><summary>Indice 3</summary>

a) $\nabla f = (2x, -2y)$, qui vaut $(0, 0)$ en $(0, 0)$ : la liste est $[0, 0]$. Pour les autres : b) écris $f(t, 0)$ et $f(0, t)$ : ce sont des fonctions $a\,t^2$, dont la dérivée seconde vaut $2a$ ; c) et d) $f(t\cos\theta, t\sin\theta) = t^2(\cos^2\theta - \sin^2\theta)$ : une formule de trigonométrie de 0B réduit la parenthèse à un seul cosinus ; prends le double de ce coefficient en $\theta = 30°$, puis cherche l'angle qui l'annule ; e) un pas multiplie $x$ par $1 - 2\eta$ et $y$ par $1 + 2\eta$ (le signe de $\frac{\partial f}{\partial y} = -2y$ compte) : applique ces facteurs deux fois ; f) remplace dans $x^2 - y^2$, et compare à $f(0{,}5 ;\ 0{,}1)$ ; g) même méthode, trois pas depuis $(0{,}5 ;\ 0)$ : que devient un $y$ nul quand on le multiplie par un facteur ?

</details>

### Ex 5.6 — Pourquoi la différence centrée est plus précise (calcul exact sur x³) ∂

<details><summary>Indice 1</summary>

$(a + h)^3 = a^3 + 3a^2h + 3ah^2 + h^3$ (le binôme, 0B) ; pour $(a - h)^3$, remplace $h$ par $-h$.

</details>
<details><summary>Indice 2</summary>

Dans $(a + h)^3 - (a - h)^3$, les termes de degré pair en $h$ s'annulent ; dans $(a + h)^3 - a^3$, non. Pour la question 7, l'erreur sur le numérateur peut atteindre $2\delta$, et le numérateur est ensuite divisé par $2h$.

</details>
<details><summary>Indice 3</summary>

1. $(a + h)^3 = a^3 + 3a^2h + 3ah^2 + h^3$ et, en remplaçant $h$ par $-h$, $(a - h)^3 = a^3 - 3a^2h + 3ah^2 - h^3$. Pour la suite : 2. retranche $a^3$, puis divise chaque terme par $h$ ; pour $D_-$, fais de même avec $a^3 - (a - h)^3$ ; 3. dans $(a + h)^3 - (a - h)^3$, repère les termes qui s'annulent et ceux qui doublent ; pour la moyenne, additionne tes deux résultats de 2 et divise par 2 ; 4. retranche $3a^2$ à chaque formule, puis remplace $h$ par $\frac{h}{10}$ : quel terme domine quand $h$ est petit ? 5. calcule $(a + h)^2 - (a - h)^2$ et divise par $2h$ ; puis relis la question 3 : quel terme du développement de $x^3$ produisait l'erreur, et existe-t-il pour $x^2$ ? 6. additionne les deux développements de 1, retranche $2a^3$ et divise par $h^2$ ; 7. écris les valeurs calculées $f(a + h) + e_1$ et $f(a - h) + e_2$, avec $|e_1| \le \delta$ et $|e_2| \le \delta$ : quelle est la plus grande valeur possible de $|e_1 - e_2|$, et que devient-elle divisée par $2h$ ? Pour le meilleur $h$, dérive $h^2 + \frac{\delta}{h}$ et étudie le signe de cette dérivée.

</details>

### Ex 5.7 — Rosenbrock : gradient et minimum à la main ∂

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

### Ex 5.8 — Le gradient expliqué avec de l'eau sur un drap 🗣️

<details><summary>Indice 1</summary>

Reprends l'image du livre (§5.4) : un drap figé, de l'eau versée dessus. Que fait l'eau ? Qu'est-ce que le gradient, dans cette image ?

</details>
<details><summary>Indice 2</summary>

Une phrase par contrainte : la pente et sa direction, le pas répété, l'endroit où l'eau s'arrête, et une différence entre l'eau et l'algorithme (relis l'encadré ⚠️ « Les limites de l'image de l'eau » de la fiche).

</details>
<details><summary>Indice 3</summary>

La première ligne, comme modèle : « Imagine un grand drap figé, plein de bosses et de creux, et une goutte d'eau posée dessus. » Écris les quatre autres toi-même, une par contrainte : en chaque point, vers où pointe la flèche du gradient, vers la montée ou vers la descente, et de quel côté part l'eau ? Comment un réseau imite-t-il l'eau, pas après pas ? Où l'eau s'arrête-t-elle, et qu'est-ce qui lui interdit de savoir si c'est le point le plus bas ? Enfin, une différence entre l'eau et l'algorithme : relis l'encadré ⚠️ « Les limites de l'image de l'eau » (en continu ou par pas ? avec ou sans élan ?).

</details>

### Ex 5.9 — Fermi : le prix d'un gradient numérique pour un million de paramètres 🧮

<details><summary>Indice 1</summary>

Avance pas à pas : le coût d'une évaluation, puis le nombre d'évaluations, puis le temps.

</details>
<details><summary>Indice 2</summary>

Une évaluation sur le mini-batch : $64 \times 2P$ opérations. Une différence centrée par paramètre : 2 évaluations par paramètre. Le temps : le nombre d'opérations divisé par $10^{11}$.

</details>
<details><summary>Indice 3</summary>

1. $64 \times 2P = 64 \times 2 \times 10^6 = 1{,}28 \times 10^8$ opérations, environ $10^8$. Pour les autres, enchaîne les multiplications et les divisions : 2. une différence centrée coûte deux évaluations par paramètre : multiplie le nombre d'évaluations par le coût d'une évaluation (question 1) ; 3. divise ce nombre d'opérations par $10^{11}$, puis convertis les secondes en minutes ; 4. trois évaluations au prix de la question 1, divisées par $10^{11}$ ; le rapport des deux durées donne le gain ; 5. multiplie chaque durée par $10^4$, puis convertis en jours ou en mois ; 6. multiplie $P$ par 1 000 : par combien sont multipliés le coût d'une évaluation, puis le nombre d'évaluations, dans chacune des deux méthodes ?

</details>

### Ex 5.10 — Dauphin et coll. (2014) : les points selles en grande dimension 📄

<details><summary>Indice 1</summary>

Le résumé répond déjà aux questions 1 et 4 ; la section 2 définit l'indice, la section 3 présente la figure 1.

</details>
<details><summary>Indice 2</summary>

L'indice se lit sur les valeurs propres de la matrice des dérivées secondes (la hessienne), c'est-à-dire sur les courbures du point critique dans ses directions principales : regarde lesquelles il compte, et comment. Pour la méthode de Newton, regarde ce que devient un pas qui divise le gradient par une courbure **négative**.

</details>
<details><summary>Indice 3</summary>

1. Le résumé le dit : la principale difficulté ne vient pas des minima locaux, mais de la **prolifération des points selles**, surtout en grande dimension, entourés de plateaux de loss élevée. Pour les autres, appuie chaque réponse sur une phrase ou une figure précise : 2. cherche dans la section 2 la phrase qui définit l'indice α, puis demande-toi combien de directions descendent autour d'un minimum, et autour d'un maximum ; 3. lis les deux axes de la figure 1 et sa légende (quels réseaux, quelles données, quelle taille d'images), puis décris la forme du nuage de points ; 4. que vaut le gradient près d'un point selle, donc la taille des pas, et à quoi ressemble alors la courbe de loss ? 5. le pas de Newton divise le gradient par la courbure : dans une direction de courbure négative, va-t-il vers le point critique ou s'en éloigne-t-il ? Cherche ensuite, dans la section 6, ce que *saddle-free Newton* change à cette division ; 6. compare la taille des réseaux et des images de la figure 1 à celle d'un réseau moderne, puis relis l'encadré 🕰️ des points selles (fiche §5.4) ; 7. relis ta réponse à ✏️ 5.5 g) et le résultat de l'expérience 🔮 5.20 : où chaque descente s'arrête-t-elle, et combien de temps perd-elle ?

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

Le squelette de `numerical_derivative` ; `second_derivative` suit le même plan, sans le choix de la méthode :

```python
def numerical_derivative(f, x, h=1e-5, method="central"):
    _check_step(h)
    # 1. refuse a method other than "central", "forward" or "backward" (ValueError)
    # 2. a scalar x: work on float(x); otherwise on np.asarray(x, dtype=float)
    # 3. the formula of the chosen method (if / elif / else)
    # 4. return float(...) for a scalar x, a float array otherwise
```

Les trois lignes clés : `if not h > 0:` dans `_check_step` (cette écriture refuse aussi `nan`), `scalar = np.ndim(x) == 0` au début, et la formule centrée, `(f(x + h) - f(x - h)) / (2 * h)` : le dénominateur est la distance entre les deux points, $2h$.

</details>

### Ex 5.12 — Quel pas h choisir ? Prédire la courbe d'erreur 🔮

<details><summary>Indice 1</summary>

Deux erreurs se disputent : celle de la formule (la troncature, ∂ 5.6), qui diminue quand $h$ diminue, et celle des arrondis : un `float` ne garde qu'environ 16 chiffres, et le numérateur soustrait deux nombres presque égaux. Comment chacune varie-t-elle avec $h$ ?

</details>
<details><summary>Indice 2</summary>

Un `float` garde environ 16 chiffres. L'erreur d'arrondi d'une différence centrée est de l'ordre de $\frac{10^{-16}}{h}$ ; sa troncature, de l'ordre de $h^2$. Pour la différence avant, la troncature est de l'ordre de $h$. Pour quel $h$ chaque paire d'erreurs est-elle du même ordre ?

</details>
<details><summary>Indice 3</summary>

a) Cherche le $h$ où les deux erreurs de la différence centrée sont du même ordre : $h^2 \approx \frac{10^{-16}}{h}$, soit $h^3 \approx 10^{-16}$ ; résous, puis garde l'exposant entier le plus proche. b) Même équation avec la troncature de la différence avant : $h \approx \frac{10^{-16}}{h}$. c) Compare deux ordres de grandeur : l'arrondi avec $h = 10^{-15}$, $\frac{10^{-16}}{10^{-15}} \times |f(2{,}2)|$ (calcule $f(2{,}2)$), et la troncature avec $h = 0{,}1$, de l'ordre de $h^2$. d) Estime l'erreur totale de chaque méthode **à son meilleur pas** (troncature plus arrondi), puis compare.

</details>

### Ex 5.13 — Erreur de troncature contre erreur d'arrondi 🔬

<details><summary>Indice 1</summary>

`errors_13` est une boucle (ou une compréhension de liste) sur `ks` : un appel à ta fonction `numerical_derivative` par pas, et la valeur absolue de l'écart à `df(x)`.

</details>
<details><summary>Indice 2</summary>

Le pas : `10.0 ** -k` (avec `10 ** -k`, NumPy refuse un exposant entier négatif). Pour e) et f), dérive $g(h) = h^2 + \frac{\varepsilon}{h}$ par rapport à $h$ et annule la dérivée ; même chose pour $h + \frac{\varepsilon}{h}$. Puis prends le logarithme décimal du résultat avec `np.log10`.

</details>
<details><summary>Indice 3</summary>

`errors_13` tient en une compréhension de liste sur `ks`, dans un `np.array([...])`. Pour chaque `k`, la pente estimée est

```python
mylearn.calculus.numerical_derivative(f, x, h=10.0 ** -k, method=method)
```

dont tu retranches la pente exacte, `df(x)`, avant de prendre la valeur absolue. Pour e), la dérivée de $h^2 + \frac{\varepsilon}{h}$ est $2h - \frac{\varepsilon}{h^2}$ : elle s'annule pour $h^3 = \frac{\varepsilon}{2}$ (n'oublie pas le facteur 2) ; avec `eps = np.finfo(float).eps`, élève `eps / 2` à la puissance `1 / 3`, puis applique `np.log10`. Pour f), annule de même la dérivée de $h + \frac{\varepsilon}{h}$ : cette fois, pas de facteur 2.

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
    # 1. checks (ValueError): y is 1-D; order is an integer >= 1, and not a bool
    # 2. for every inner sample i, from 1 to len(y) - 2: its neighbours, at most `order` on each side
    # 3. strictly smaller than all of them: a minimum; strictly larger than all of them: a maximum
    # 4. return np.array(minima, dtype=int), np.array(maxima, dtype=int), even when they are empty
```

Les lignes clés : la condition qui refuse `order`, `isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1`, les voisins `np.concatenate([y[max(0, i - order):i], y[i + 1:i + order + 1]])`, et le test strict `np.all(y[i] < neighbours)`. Pour e), `smooth_14 == 0` est un tableau de booléens : `np.sum` compte ses `True`.

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
    # grad: zeros with the shape of point, and a flat view of it too
    # for each position i: keep old = flat[i]; set old + h, evaluate f(point);
    #     set old - h, evaluate f(point); put old back
    # the central difference goes into the flat view of grad; return grad
```

La ligne clé de la boucle : `flat[i] = old`, après les deux évaluations. Elle remet la valeur **exacte** ; refaire `+ h` après `- 2 * h` ne la redonne pas toujours au bit près.

</details>

### Ex 5.16 — Le gradient qui abîme son entrée 🐛

<details><summary>Indice 1</summary>

Deux questions à se poser sur la fonction du collègue : travaille-t-elle sur le tableau de l'appelant, ou sur une copie ? Et quel est le type des nombres de ce tableau ?

</details>
<details><summary>Indice 2</summary>

`np.asarray` renvoie le tableau lui-même quand c'est déjà un tableau NumPy, sans copie. Dans un tableau d'entiers, `x[i] += 1e-5` stocke un entier : la valeur est tronquée (arrondie vers 0). Pour c), après l'appel, regarde `point_16 == np.array([0.1, 0.2])`, ou `repr(point_16[0])`.

</details>
<details><summary>Indice 3</summary>

a) `rosenbrock_gradient([-1.0, 1.0])` donne le vrai gradient (ou ∂ 5.7, à la main). b) Suis la première coordonnée : $-1 + 10^{-5} = -0{,}99999$, rangé dans un tableau d'**entiers**, est tronqué vers 0 ; continue avec $-2h$, puis $+h$, en tronquant à chaque fois, puis fais de même pour la seconde coordonnée (ou affiche simplement `start_16`). c) Après l'appel, `point_16 == np.array([0.1, 0.2])` compare chaque coordonnée à sa valeur d'origine : compte les `False`. d) La correction tient en une ligne, la première de la fonction : il faut une conversion qui **copie toujours** et qui **convertit en flottants** (l'énoncé de 5.15 la donne) ; `np.zeros_like(x)` suit alors le nouveau type.

</details>

### Ex 5.17 — Lire des lignes de niveau : où pointe le gradient ? 📈

<details><summary>Indice 1</summary>

Repère d'abord, grâce à la barre de couleurs, les zones hautes (claires) et basses (sombres). Un maximum est entouré de lignes fermées dans une zone claire ; un point selle est un croisement de lignes en X.

</details>
<details><summary>Indice 2</summary>

La pente est forte là où les lignes de niveau sont serrées. Le gradient est perpendiculaire aux lignes de niveau et pointe vers les valeurs plus hautes. Pour f), dérive $g$ par rapport à $x$, puis par rapport à $y$, en gardant les signes.

</details>
<details><summary>Indice 3</summary>

a) Le maximum est au centre de lignes fermées, dans la zone la plus claire : c'est C. Pour les autres : b) même critère dans la zone la plus sombre ; c) cherche **tous** les croisements de lignes en X ; d) compare l'écartement des lignes de niveau autour de D et autour de F ; e) le gradient est perpendiculaire aux lignes de niveau et pointe vers les couleurs plus claires : regarde de quel côté de D elles se trouvent, puis traduis-le en une des huit directions ; f) $\frac{\partial g}{\partial x} = 3x^2 - 3$ et $\frac{\partial g}{\partial y} = -3y^2 + 3$ : remplace $x$ par 1,8 et $y$ par $-0{,}2$.

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
    # 1. checks (ValueError): lr <= 0, n_steps < 0, tol given and < 0
    x = np.array(x0, dtype=float)          # a float copy of the starting point
    path = [x.copy()]
    sign = 1.0 if maximize else -1.0       # ascent along the gradient, descent against it
    # 2. at most n_steps times: g = the gradient at the current point, as a float array;
    #    if tol is given and the norm of g is < tol: break, BEFORE moving;
    #    otherwise x = x + sign * lr * g, then append a copy of x to path
    # 3. return x and np.array(path)
```

Le piège : `x -= lr * g` modifie `x` sur place ; si tu ranges `x` dans le chemin sans `.copy()`, tous les points du chemin deviennent le dernier.

</details>

### Ex 5.19 — Learning rate sur un bol : trop petit, juste, trop grand 🔬

<details><summary>Indice 1</summary>

`distances_19` appelle ta fonction `gradient_descent`, garde le chemin, et mesure la distance de chaque point à $(0, 0)$.

</details>
<details><summary>Indice 2</summary>

Le chemin a la forme `(51, 2)` : `np.linalg.norm(path, axis=1)` donne les 51 distances. Pour les questions, calcule les deux facteurs $1 - 2\eta$ et $1 - 20\eta$ pour chaque learning rate : la coordonnée converge si son facteur est strictement entre $-1$ et 1.

</details>
<details><summary>Indice 3</summary>

`distances_19` : un appel `mylearn.calculus.gradient_descent(grad_bowl_19, START_19, lr=lr, n_steps=50)`, dont tu gardes le second élément renvoyé (le chemin), puis une ligne avec `np.linalg.norm(..., axis=1)`. Pour les questions : à la longue, la distance est multipliée à chaque pas par le plus grand des deux facteurs **en valeur absolue** ; calcule-le pour chaque learning rate qui converge, puis compare. Avec 0,01, $v_0$ est multiplié par 0,98 à chaque pas : cherche $t$ tel que $2 \times 0{,}98^t < 0{,}01$.

</details>

### Ex 5.20 — Démarrer pile sur un point selle 🔮

<details><summary>Indice 1</summary>

Relis la règle de mise à jour : de quoi dépend la taille d'un pas ? Que se passe-t-il là où cette quantité est nulle, ou minuscule ?

</details>
<details><summary>Indice 2</summary>

Pour un $y$ minuscule, compare $y^3$ et $y$ : un pas fait-il grandir ou diminuer $|y|$, et de combien, en proportion ? Le signe de $y$ peut-il changer ?

</details>
<details><summary>Indice 3</summary>

Près de l'axe, $|y|$ est petit et $y^3$ est négligeable devant $y$ : un pas fait $y \leftarrow y - \eta\,(y^3 - y) \approx (1 + \eta)\,y$. a) Que vaut le gradient en $(0, 0)$, donc le pas ? b) Une multiplication par un facteur positif peut-elle changer le signe de $y$ ? Vers lequel des deux minima mène alors le côté où l'on part ? c) Résous $(1 + \eta)^n = \frac{0{,}5}{10^{-6}}$, soit $n = \frac{\ln(5 \times 10^5)}{\ln 1{,}1}$, puis choisis l'ordre de grandeur le plus proche. d) Même calcul pour le facteur à gagner en plus : $n = \frac{\ln 10^6}{\ln 1{,}1}$, arrondi au plus proche.

</details>

### Ex 5.21 — Le même gradient avec torch.autograd 📦

<details><summary>Indice 1</summary>

`rosenbrock_torch` ressemble à la formule de Rosenbrock écrite en Python, avec `p[0]` et `p[1]` à la place de `x` et `y`. N'appelle aucune fonction NumPy (ni `wb.synth.rosenbrock`, qui en utilise).

</details>
<details><summary>Indice 2</summary>

Les opérateurs `-`, `*` et `**` marchent sur les tenseurs. Pour b), `torch_gradient(lambda p: torch.relu(p[0]), [0.0])` donne la pente de ReLU en 0 ; fais de même avec `torch.clamp(p[0], min=0)` et `torch.maximum(p[0], torch.zeros_like(p[0]))`, puis appelle ta `numerical_derivative` sur `lambda x: max(0.0, x)`.

</details>
<details><summary>Indice 3</summary>

`rosenbrock_torch` tient en une ligne, `return (1 - p[0]) ** 2 + ...`, où tu écris le terme de la vallée, multiplié par $b = 100$, avec `p[0]` et `p[1]` et sans NumPy. Pour `corner_21`, la première pente s'écrit

```python
float(torch_gradient(lambda p: torch.relu(p[0]), [0.0])[0])
```

Écris les deux suivantes sur ce modèle, en changeant seulement la fonction, puis la dernière avec ta `mylearn.calculus.numerical_derivative`, appliquée à `lambda x: max(0.0, x)` en 0. Ne suppose rien : calcule les quatre.

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
    # 1. the heights Z = log10(1 + f) on the grid, then ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)
    # 2. the path at the height of the surface (same formula on path[:, 0], path[:, 1]): ax.plot(xs, ys, zs)
    # 3. the start and the end: two calls of ax.scatter(x, y, z)


def draw_map_22(ax, path):
    wb.plot.plot_contour(wb.synth.rosenbrock, xlim=(-2, 2), ylim=(-1, 3), path=path, ax=ax, minimum=(1, 1))
    # 1. one point every 300 steps: path[::300]
    # 2. minus the exact gradient (rosenbrock_gradient) at each of these points, as an array of shape (k, 2)
    # 3. divide each arrow by its norm, then ax.quiver(xs, ys, us, vs)
```

La ligne clé de l'étape 3 : `arrows / np.linalg.norm(arrows, axis=1, keepdims=True)`. Sans `keepdims=True`, les formes `(k, 2)` et `(k,)` ne se combinent pas.

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
    # draw A (shape (n, n)), b and x (shape (n,)) with rng.normal
    # result = numerical_gradient of the function v -> v @ A @ v + b @ v, at x
    assert np.allclose(result, (A + A.T) @ x + b, atol=1e-6)
```

Le deuxième test compare `numerical_gradient(f, np.array([-1, 2]))` au gradient du même point écrit en flottants, pour une fonction `f` définie **dans** le test et dont les dérivées partielles ne sont pas nulles en ce point. Le troisième vérifie la **forme** et la valeur du gradient d'une fonction d'une matrice. Puis `my_tests_23 = [...]`, la liste des trois fonctions de test.

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
    # 1. ValueError if point is not 1-D, or if the norm of numerical_gradient(f, point) is > grad_tol
    # 2. directions: the rows of np.eye(n), then eye[i] + eye[j] and eye[i] - eye[j] for every i < j
    # 3. f0 = f(point), computed once; then one second difference per direction:
    second = np.array([(f(point + h * d) - 2 * f0 + f(point - h * d)) / h ** 2 for d in directions])
    up, down = second > tol, second < -tol
    # 4. decide: "saddle", "minimum", "maximum" or "flat"
```

Le piège de l'étape 4 : « minimum » et « maximum » demandent `all`. Avec `any`, une selle passerait pour un minimum si ce test venait avant celui de la selle (`up.any() and down.any()`) ; avec `all`, les trois cas s'excluent, et « plat » est simplement le cas restant.

</details>

### Ex 5.25 — Atteindre le fond de la vallée de Rosenbrock 🏆

<details><summary>Indice 1</summary>

Commence par mesurer : pour quelques learning rates, combien de pas faut-il pour arriver à moins de $10^{-3}$ de $(1, 1)$ ? Écris une petite fonction qui le calcule à partir du chemin de `gradient_descent`.

</details>
<details><summary>Indice 2</summary>

`np.linalg.norm(path - np.array([1.0, 1.0]), axis=1) < 1e-3`, puis le premier indice `True` (`np.flatnonzero`). Essaie 0,001, 0,0015, 0,0018, 0,0019, 0,002. Pour chaque candidat qui arrive à temps, vérifie aussi qu'il **reste** : `grade_25([(lr, n)])` te renvoie le nombre de pas, la distance finale, puis la pire distance pendant les 10 000 pas suivants et la distance à leur fin.

</details>
<details><summary>Indice 3</summary>

Mesure d'abord, avec une petite fonction :

```python
def steps_to_reach(lr, max_steps=20_000):
    _, path = mylearn.calculus.gradient_descent(rosenbrock_gradient, START_25, lr=lr, n_steps=max_steps)
    # the first index where the distance to TARGET_25 is < 1e-3 (np.flatnonzero), or None if there is none
```

Au fond de la vallée, la courbure la plus forte vaut environ 1 000 : comme sur le bol de 5.19, au-delà de $\frac{2}{1\,000}$, la descente oscille en travers de la vallée sans s'amortir. Cherche donc, juste **en dessous** de cette limite, un learning rate qui arrive en moins de 10 000 pas, puis vérifie avec `grade_25` qu'il **reste** au fond. Un programme en deux phases, rapide puis prudent, marche aussi : c'est la dernière phase qui doit rester sous la limite.

</details>
