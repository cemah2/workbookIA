# 10 · Neurones — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 🧮 ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 10.Q1 — Neurones artificiels : où sont-ils indispensables, où s'en passe-t-on ?

<details><summary>Indice 1</summary>

Relis la section 10.1 de la fiche : quelles méthodes calculent leur résultat par une formule ou par un algorithme dédié, sans neurone ?

</details>
<details><summary>Indice 2</summary>

Pour chaque méthode, demande-toi ce qu'elle calcule : des moyennes et des distances (ch. 7), une solution de moindres carrés (ch. 9), une suite de questions sur les variables (ch. 13), ou des sommes pondérées enchaînées sur des couches ? Pour c), que veut dire « profond » ?

</details>
<details><summary>Indice 3</summary>

a) Le k-means, les moindres carrés et l'arbre de décision n'ont aucun neurone ; les réseaux convolutifs et les grands modèles de langage sont faits de neurones. b) Une forme fermée n'a besoin d'aucun neurone. c) Un perceptron n'a qu'une couche de poids : « profond » suppose plusieurs couches.

</details>

### 10.Q2 — Le neurone biologique en quatre étapes

<details><summary>Indice 1</summary>

Relis la liste numérotée de la section 10.2 de la fiche, et suis le trajet d'un message : il arrive, il est traité, il repart.

</details>
<details><summary>Indice 2</summary>

Qu'est-ce qui déclenche les signaux électriques ? Que faut-il faire de plusieurs signaux avant de les comparer à un seuil ? Que se passe-t-il quand le seuil est dépassé ? Pour b), les signaux sont-ils tous de même signe ? Pour c), qu'est-ce que la fente synaptique ?

</details>
<details><summary>Indice 3</summary>

a) Fixation des neurotransmetteurs, addition des signaux, comparaison au seuil, libération de nouveaux neurotransmetteurs. b) Certains signaux sont inhibiteurs. c) Dans une synapse chimique, la plus courante, une fente de quelques dizaines de nanomètres sépare les deux neurones.

</details>

### 10.Q3 — Connectome et émulation du cerveau : vrai ou faux justifié

<details><summary>Indice 1</summary>

Relis la fin de la section 10.2 de la fiche et son encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

b) Le livre compare les connectomes à une empreinte digitale. c) Qu'ont produit les puces neuromorphiques et les simulations géantes ? d) Cherche « cognition incarnée ». e) Cherche « FlyWire ».

</details>
<details><summary>Indice 3</summary>

Vrai, faux, faux, vrai, vrai. Pour chaque justification, une phrase suffit : la définition (a), l'unicité (b), le constat du livre et les systèmes neuromorphiques récents (c), l'idée de la cognition incarnée (d), le connectome de la mouche publié en 2024 (e).

</details>

### 10.Q4 — Neurone, unité, « cerveau électronique » : bien nommer les choses

<details><summary>Indice 1</summary>

Relis la section 10.3 de la fiche, en particulier la comparaison avec un plan de métro.

</details>
<details><summary>Indice 2</summary>

Que garde un neurone artificiel d'un vrai neurone, et que laisse-t-il de côté ? Un mot plus neutre que « neurone » éviterait quelle confusion ?

</details>
<details><summary>Indice 3</summary>

a) Le mot « unité » rappelle que le neurone artificiel est une abstraction très simplifiée. b) Un réseau de neurones n'est pas un cerveau. c) Il garde l'addition de signaux suivie d'une décision.

</details>

### 10.Q5 — McCulloch et Pitts (1943) : ce qu'ils ont démontré

<details><summary>Indice 1</summary>

Relis le paragraphe « 1943 : le neurone formel » de la section 10.3.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

a) Leur résultat porte sur la logique, pas sur l'apprentissage. b) Qui a introduit des poids qui s'ajustent à partir d'exemples ? c) Écris la table : pour (0, 0), (0, 1), (1, 0) et (1, 1), la somme atteint-elle 2 ?

</details>
<details><summary>Indice 3</summary>

a) Toute expression logique (sous certaines conditions) peut être réalisée par un réseau de neurones formels. b) Leurs poids et seuils étaient fixés à la main. c) La somme n'atteint 2 que pour (1, 1) : c'est AND.

</details>

### 10.Q6 — Anatomie d'un perceptron

<details><summary>Indice 1</summary>

Relis la formule du perceptron de la section 10.3.1 de la fiche, et la phrase qui la suit.

</details>
<details><summary>Indice 2</summary>

a) Combien de poids un perceptron a-t-il : un par entrée, ou un seul ? b) La condition pour sortir $+1$ est-elle $z > 0$ ou $z \ge 0$ ? c) Le livre mentionne une autre paire de sorties.

</details>
<details><summary>Indice 3</summary>

a) Multiplier chaque entrée par son poids, additionner, comparer à 0. b) Une somme nulle n'est pas $> 0$ : la sortie est $-1$. c) Vrai : certaines versions sortent 1 et 0.

</details>

### 10.Q7 — Mark I, Minsky-Papert, renaissance : la chronologie

<details><summary>Indice 1</summary>

Regarde la chronologie de la section 10.3.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Les dates : 1943, 1957, 1960, 1969, 1986. b) Relis la description de la machine : comment ses cellules étaient-elles disposées ? d) Sur quel type de perceptron portaient les démonstrations de Minsky et Papert ?

</details>
<details><summary>Indice 3</summary>

a) McCulloch et Pitts, le rapport de Rosenblatt, le Mark I, *Perceptrons*, la rétropropagation. b) $20 \times 20$. c) Deux classes qu'un hyperplan sépare. d) Faux : leurs résultats portent sur une seule couche de poids appris, et un réseau à deux couches calcule XOR.

</details>

### 10.Q8 — Du perceptron au neurone moderne : les deux changements

<details><summary>Indice 1</summary>

Relis le début de la section 10.3.3 de la fiche : un changement à l'entrée, un à la sortie.

</details>
<details><summary>Indice 2</summary>

b) Compte un poids par entrée, et n'oublie pas ce qu'on ajoute à la somme. d) Que vaut $\mathbf{w}\cdot\mathbf{0}$ ?

</details>
<details><summary>Indice 3</summary>

a) Le biais et la fonction d'activation. b) $4 + 1$. c) Vrai. d) Vrai : sans biais, l'origine donne toujours $z = 0$, elle est sur la frontière.

</details>

### 10.Q9 — Lire un schéma de réseau : poids implicites et convention AD/DA

<details><summary>Indice 1</summary>

Relis les paragraphes « Dessiner un neurone » et « Nommer les poids » de la section 10.3.3 de la fiche, et regarde la figure de la couche.

</details>
<details><summary>Indice 2</summary>

b) Chaque neurone de la première couche est relié à chacun de la seconde. c) La question ne porte que sur D et E : compte leurs poids d'entrée et leurs biais. d) Dans le livre, le nom d'un poids commence par la source. e) Relis l'encadré 🕰️ sur les conventions.

</details>
<details><summary>Indice 3</summary>

a) Faux : les poids sont implicites, mais toujours là. b) $3 \times 2$. c) Ajoute les biais de D et de E. d) Le poids qui multiplie la sortie de B avant son entrée dans E. e) Vrai : `weight` a la forme `(out_features, in_features)`.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 10.R1 — Ch. 9 : régularisation L2, que deviennent les poids ?

<details><summary>Indice 1</summary>

La formule est donnée dans l'énoncé : il suffit de remplacer.

</details>
<details><summary>Indice 2</summary>

a) Remplace $\lambda$ par 0 dans la formule. b) Le dénominateur grandit avec $\lambda$. d) Repense à l'ordonnée à l'origine de Ridge au ch. 9 : est-elle pénalisée ?

</details>
<details><summary>Indice 3</summary>

a) 2. b) $20 / 20 = 1$. c) Vrai. d) Les poids, pas le biais. e) Le biais ne fait que déplacer la frontière (ou la droite) ; le pénaliser tirerait les prédictions vers 0 sans rien gagner en simplicité, et le résultat changerait si l'on décale toutes les cibles.

</details>

### 10.R2 — Ch. 7 : la frontière du centroïde le plus proche, w·x + b = 0

<details><summary>Indice 1</summary>

Développe $\lVert \mathbf{x} - \boldsymbol{\mu} \rVert^2 = \lVert \mathbf{x} \rVert^2 - 2\,\boldsymbol{\mu}\cdot\mathbf{x} + \lVert \boldsymbol{\mu} \rVert^2$ pour les deux centroïdes.

</details>
<details><summary>Indice 2</summary>

Le terme $\lVert \mathbf{x} \rVert^2$ disparaît des deux côtés. Il reste une inégalité entre $2\,(\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-)\cdot\mathbf{x}$ et $\lVert \boldsymbol{\mu}_+ \rVert^2 - \lVert \boldsymbol{\mu}_- \rVert^2$ ; divise par 2.

</details>
<details><summary>Indice 3</summary>

$\mathbf{w} = \boldsymbol{\mu}_+ - \boldsymbol{\mu}_- = (2, 2)$ et $b = (\lVert \boldsymbol{\mu}_- \rVert^2 - \lVert \boldsymbol{\mu}_+ \rVert^2)/2 = (1 - 13)/2$. Pour $\mathbf{x} = (3, 2)$ : $6 + 4 + b$.

</details>

### 10.R3 — Ch. 3 : matrice de confusion et accuracy d'un classifieur binaire

<details><summary>Indice 1</summary>

Range les dix exemples dans les quatre cases : vrai label $\pm 1$, prédiction $\pm 1$.

</details>
<details><summary>Indice 2</summary>

Les quatre premiers exemples sont les vrais $+1$ ; les six autres, les vrais $-1$. Le recall divise par les vrais positifs plus les faux négatifs ; la precision, par les vrais positifs plus les faux positifs.

</details>
<details><summary>Indice 3</summary>

Parmi les quatre vrais $+1$, trois sont prédits $+1$ (TP = 3, FN = 1). Parmi les six vrais $-1$, deux sont prédits $+1$ (FP = 2, TN = 4). Accuracy $7/10$, recall $3/4$, precision $3/5$.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 10.1 — Sortie d'un perceptron à quatre entrées, avec et sans biais ✏️

<details><summary>Indice 1</summary>

Multiplie terme à terme, additionne, puis applique la règle « $+1$ si $z > 0$ ». Attention aux signes et au cas $z = 0$.

</details>
<details><summary>Indice 2</summary>

Écris chaque somme terme à terme, quatre produits, puis additionne-les, sans sauter le dernier terme. Avec le biais, ajoute $b$ à chaque somme avant de prendre le seuil. i) Calcule les trois sommes sans biais : pour chacune, quelle condition sur $b$ la rend strictement positive ?

</details>
<details><summary>Indice 3</summary>

$z^{(1)} = 0{,}5 \times 2 + (-1) \times 1 + 2 \times 0{,}5 + 0{,}25 \times (-4) = 0$, donc $-1$ ; $z^{(2)} = 0{,}5$, donc $+1$ ; avec le biais : $-0{,}75$, $-0{,}25$ et $0{,}25$, une seule sortie $+1$. h) $z = 0$ n'est pas $> 0$. i) Les sommes sans biais valent 0 ; 0,5 ; 1 : il faut $b > 0$, et le plus petit entier est 1.

</details>

### Ex 10.2 — L'astuce du biais : même neurone, une entrée de plus ✏️

<details><summary>Indice 1</summary>

Relis l'astuce du biais dans la section 10.3.3 de la fiche : le biais devient le **premier** poids, celui d'une entrée constante 1.

</details>
<details><summary>Indice 2</summary>

Le vecteur augmenté commence par le biais, puis les poids ; $\tilde{\mathbf{x}}$ commence par 1, puis les entrées. Pour le batch, chaque ligne devient $(1, x_1, x_2)$. f) La matrice augmentée a une ligne par entrée, plus une ligne de biais, et une colonne par neurone. g) Écris les trois conditions données par $(1, 0)$, $(0, 1)$ et $(1, 1)$ quand $b = 0$.

</details>
<details><summary>Indice 3</summary>

a) $\tilde{\mathbf{w}} = (0{,}5;\ 2;\ -1)$ ; b) $\tilde{\mathbf{x}} = (1, 1, 3)$ et $z = 0{,}5 + 2 - 3$. c) $[-0{,}5;\ 0{,}5;\ 3{,}5]$, donc d) $[-1, 1, 1]$. e) Faux : c'est le même calcul. f) $(3 + 1) \times 5$. g) Il faudrait $w_1 \le 0$, $w_2 \le 0$ et $w_1 + w_2 > 0$ : impossible.

</details>

### Ex 10.3 — Portes logiques à la main : AND, OR, NOT, NAND ✏️

<details><summary>Indice 1</summary>

Pars de l'exemple AND de la fiche ($\mathbf{w} = (1, 1)$, $b = -1{,}5$) : la droite $x_1 + x_2 = 1{,}5$ laisse un seul coin du côté positif. Pour chaque porte, cherche où placer la droite, puis vérifie les entrées une à une.

</details>
<details><summary>Indice 2</summary>

OR : la même direction $\mathbf{w} = (1, 1)$, mais une droite qui laisse trois coins du côté positif. NOT : un seul poids, négatif. NAND : que devient la sortie si l'on change le signe de $z$ ? Majorité : trois poids égaux, et un seuil entre 1 et 2. f) Que devient le signe de $c\,z$ si $c > 0$ ? si $c < 0$ ?

</details>
<details><summary>Indice 3</summary>

OR : $\mathbf{w} = (1, 1)$, $b = -0{,}5$. NOT : $w = -1$, $b = 0{,}5$. NAND : $\mathbf{w} = (-1, -1)$, $b = 1{,}5$ (aucune des quatre sommes d'AND ne vaut 0, donc changer le signe échange exactement les sorties ; une somme nulle donnerait 0 dans les deux cas). NOR : $\mathbf{w} = (-1, -1)$, $b = 0{,}5$. Majorité : $\mathbf{w} = (1, 1, 1)$, $b = -1{,}5$.

</details>

### Ex 10.4 — Pourquoi un seul perceptron ne peut pas calculer XOR ∂

<details><summary>Indice 1</summary>

Écris une inégalité par entrée : sortie 1 si $s > 0$, sortie 0 si $s \le 0$.

</details>
<details><summary>Indice 2</summary>

a) Pour chaque entrée, remplace $x_1$ et $x_2$ par leurs valeurs dans $s$, et écris $s > 0$ ou $s \le 0$ selon la sortie voulue. b) Additionne les deux inégalités strictes, puis compare avec celle de $(1, 1)$ en te servant de celle de $(0, 0)$. c) $s$ est affine : la valeur au milieu d'un segment est la moyenne des valeurs aux deux bouts. e) Avec $x_3 = x_1 x_2$, le point $(1, 1)$ reçoit une coordonnée de plus, qui peut le pousser du côté négatif.

</details>
<details><summary>Indice 3</summary>

a) $b \le 0$ ; $w_2 + b > 0$ ; $w_1 + b > 0$ ; $w_1 + w_2 + b \le 0$. b) La somme des deux du milieu donne $w_1 + w_2 + 2b > 0$, donc $w_1 + w_2 + b > -b \ge 0$, ce qui contredit $w_1 + w_2 + b \le 0$. c) $s(0{,}5;0{,}5) = \frac{1}{2}(s(0,1) + s(1,0)) > 0$ et $= \frac{1}{2}(s(0,0) + s(1,1)) \le 0$. d) Non : XNOR a les mêmes points de chaque côté, labels échangés. e) Par exemple $\mathbf{w} = (1, 1, -2)$, $b = -0{,}5$. f) $2^4 = 16$ fonctions ; seules XOR et XNOR échappent au perceptron.

</details>

### Ex 10.5 — XOR en deux couches : câbler et nommer les poids ✏️

<details><summary>Indice 1</summary>

Calcule le réseau couche par couche : d'abord C et D pour les quatre entrées, puis E à partir des sorties de C et de D (pas des entrées A et B).

</details>
<details><summary>Indice 2</summary>

C : $A + B - 0{,}5 > 0$ ? D : $-A - B + 1{,}5 > 0$ ? E : $C + D - 1{,}5 > 0$ ? e) La convention de mylearn met une ligne par **source** (A, B) et une colonne par neurone (C, D) : $W_{AC}$ en ligne A, colonne C. h) Remplace chaque seuil par l'identité et développe la sortie de E.

</details>
<details><summary>Indice 3</summary>

C : $[0, 1, 1, 1]$ (OR) ; D : $[1, 1, 1, 0]$ (NAND) ; E : $[0, 1, 1, 0]$ (AND des deux, donc XOR). e) $[[1, -1], [1, -1]]$ ; f) sa transposée. g) 6 poids et 3 biais. h) Vrai : $E = (A + B - 0{,}5) + (-A - B + 1{,}5) - 1{,}5 = -0{,}5$, une constante ici, affine en général.

</details>

### Ex 10.6 — Une epoch de la règle du perceptron à la main ✏️

<details><summary>Indice 1</summary>

Tiens le tableau proposé, une ligne par exemple. Pour chaque exemple : calcule $z = \mathbf{w}\cdot\mathbf{x} + b$ avec les poids **courants**, puis $y z$ ; si $y z \le 0$, corrige, sinon passe.

</details>
<details><summary>Indice 2</summary>

Pour chaque exemple : $z$ avec les poids courants, puis $y z$ ; une correction ajoute $\eta\,y\,\mathbf{x}$ aux poids et $\eta\,y$ au biais. Attention au cas $y z = 0$, dès le premier exemple. g) et h) La règle de **prédiction** et la règle d'**apprentissage** ne traitent pas $z = 0$ de la même façon : compare-les sur $(0, 0)$.

</details>
<details><summary>Indice 3</summary>

Après l'exemple 1 : $\mathbf{w} = (0, 0)$, $b = -1$. Après l'exemple 2 : $(0, 1)$, $b = 0$. Après l'exemple 3 : $(1, 1)$, $b = 1$. L'exemple 4 est bien classé. Epoch 2 : seul $(0, 0)$ est corrigé, $b = 0$. Epoch 3 : $(0, 0)$ donne $z = 0$, bien prédit mais $y z = 0$ : correction.

</details>

### Ex 10.7 — Le théorème de convergence du perceptron, guidé pas à pas ∂

<details><summary>Indice 1</summary>

Deux quantités évoluent à chaque correction : le produit scalaire $\mathbf{u}\cdot\mathbf{w}$ (il mesure l'alignement avec le bon séparateur) et la norme $\lVert \mathbf{w} \rVert$. Montre que la première croît vite et que la seconde croît lentement.

</details>
<details><summary>Indice 2</summary>

b) $\mathbf{u}\cdot\mathbf{w}_k = \mathbf{u}\cdot\mathbf{w}_{k-1} + y\,\mathbf{u}\cdot\mathbf{x}$, et $y\,\mathbf{u}\cdot\mathbf{x} \ge \gamma$. c) $\lVert \mathbf{w}_{k-1} + y\mathbf{x} \rVert^2 = \lVert \mathbf{w}_{k-1} \rVert^2 + 2y\,\mathbf{w}_{k-1}\cdot\mathbf{x} + \lVert \mathbf{x} \rVert^2$ ($y^2 = 1$), et le terme du milieu est $\le 0$ à une erreur. d) $\mathbf{u}\cdot\mathbf{w}_k \le \lVert \mathbf{u} \rVert\,\lVert \mathbf{w}_k \rVert$. e) Refais b) et c) en gardant $\eta$ dans chaque correction : que deviennent les deux inégalités ?

</details>
<details><summary>Indice 3</summary>

$k\gamma \le \mathbf{u}\cdot\mathbf{w}_k \le \lVert \mathbf{w}_k \rVert \le \sqrt{k}\,R$, d'où $\sqrt{k} \le R/\gamma$. Avec $\eta$, les deux inégalités sont multipliées par $\eta$ et la borne ne change pas. f) $\lVert \tilde{\mathbf{x}}_i \rVert^2 = 1 + \lVert \mathbf{x}_i \rVert^2 \le 1 + R^2$, et la marge se calcule avec $(b, \mathbf{w})$ normalisé. h) $R \le \sqrt{400} = 20$, donc au plus $(20/0{,}5)^2 = 1\,600$ corrections.

</details>

<a id="reflexion"></a>

## 🗣️ 🧮 ⚖️ 📄 Réflexion

### Ex 10.8 — Pourquoi un neurone artificiel n'est pas un neurone 🗣️

<details><summary>Indice 1</summary>

Pars de la comparaison avec un plan de métro (fiche §10.3) : ce qui reste, ce qui disparaît.

</details>
<details><summary>Indice 2</summary>

La ressemblance : additionner des signaux pondérés, puis décider. Les différences possibles : la chimie, le temps et les impulsions, la forme des cellules, la façon d'apprendre, l'échelle, l'énergie.

</details>
<details><summary>Indice 3</summary>

Un plan en trois temps : « c'est une métaphore » ; « ce qui est vrai : une somme et un seuil » ; « ce qui est faux : trois différences » ; et une conclusion sur le mot « unité ».

</details>

### Ex 10.9 — Fermi : cerveau humain contre grands modèles 🧮

<details><summary>Indice 1</summary>

Travaille en puissances de 10, et garde des fourchettes : un ordre de grandeur suffit.

</details>
<details><summary>Indice 2</summary>

a) $10^{14} / (8{,}6 \times 10^{10})$ et $5 \times 10^{14} / (8{,}6 \times 10^{10})$. b) Divise les synapses par $10^{12}$. c) $10^{12}$ paramètres de 2 octets, à répartir en blocs de 80 Go. d) Nombre de GPU × 700 W, puis compare à 20 W.

</details>
<details><summary>Indice 3</summary>

a) Entre 1 000 et 6 000 synapses par neurone environ. b) Entre 100 et 500 synapses par paramètre. c) 2 To, donc au moins 25 GPU. d) Environ 17,5 kW, 875 fois, près de 900 fois un cerveau, sans compter les processeurs, la mémoire, le refroidissement… e) $20 \times 24 = 480$ Wh. g) Pense à ce qu'est une synapse (dynamique, chimique) et à ce qu'est un paramètre (un nombre), à l'architecture, aux données et à l'apprentissage.

</details>

### Ex 10.10 — « Cerveaux électroniques » : hype, hivers de l'IA et responsabilité ⚖️

<details><summary>Indice 1</summary>

Sépare deux colonnes : ce que la machine a fait devant les journalistes, et ce qu'on a annoncé qu'elle ferait un jour.

</details>
<details><summary>Indice 2</summary>

Pour 3., pense à la chaîne promesse, crédits, déception, retrait des crédits. Pour 4., des critères : une démonstration vérifiable, une mesure sur un test indépendant, des limites annoncées, des chiffres plutôt que des adjectifs.

</details>
<details><summary>Indice 3</summary>

La démonstration : distinguer deux types de cartes après une cinquantaine d'essais. L'annonce : marcher, parler, voir, écrire, se reproduire, être conscient. La responsabilité est partagée entre celui qui finance et communique, le chercheur qui laisse dire, et le journaliste qui amplifie. Tes trois règles peuvent porter sur la tâche exacte, la mesure (sur quelles données) et les limites connues.

</details>

### Ex 10.11 — Rosenblatt (1958) : le perceptron dans le texte 📄

<details><summary>Indice 1</summary>

Les trois questions sont au tout début de l'introduction ; les deux positions sur la mémoire suivent immédiatement.

</details>
<details><summary>Indice 2</summary>

Repère les mots *S-points* (ou *sensory units*), *A-units*, *R-units*, et la phrase qui dit quelles connexions sont aléatoires. Pour les systèmes α, β et γ, regarde ce qui gagne de la « valeur » et à quel moment.

</details>
<details><summary>Indice 3</summary>

Rosenblatt adopte la position « connexionniste » : l'information est stockée dans les connexions. Les connexions de l'aire de projection à l'aire d'association sont aléatoires ; les réponses s'excluent par des connexions inhibitrices en retour. Les systèmes de renforcement augmentent la valeur des cellules **actives**, pas seulement après une erreur. Il reconnaît une limite : la reconnaissance de relations dans l'espace et le temps. Les expériences ont été simulées sur un IBM 704.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 10.E1 — Qu'est-ce qu'un perceptron, et quelle est sa limite fondamentale ?

<details><summary>Indice 1</summary>

Trois temps : ce qu'il calcule, ce qu'il sait séparer, comment on dépasse la limite.

</details>
<details><summary>Indice 2</summary>

Mots-clés : somme pondérée, seuil, hyperplan, séparabilité linéaire, XOR, couches cachées, activation non linéaire, rétropropagation.

</details>
<details><summary>Indice 3</summary>

« Une somme pondérée suivie d'un seuil : une frontière linéaire. Il apprend par corrections et converge si les données sont séparables ; XOR ne l'est pas. Deux couches avec une activation non linéaire, entraînées par rétropropagation, le calculent. »

</details>

### 10.E2 — À quoi sert le biais d'un neurone ?

<details><summary>Indice 1</summary>

Pense à la géométrie de la frontière $\mathbf{w}\cdot\mathbf{x} + b = 0$.

</details>
<details><summary>Indice 2</summary>

Sans biais, que vaut $z$ en $\mathbf{x} = \mathbf{0}$ ? Le biais est un seuil réglable, l'équivalent de l'ordonnée à l'origine d'une régression.

</details>
<details><summary>Indice 3</summary>

« Il déplace la frontière : sans lui, elle passe par l'origine (AND devient impossible avec des entrées 0/1). C'est un seuil appris. On peut le traiter comme le poids d'une entrée constante 1 (l'astuce du biais), et on ne le pénalise pas en général. »

</details>

### 10.E3 — Pourquoi remplacer le seuil par une activation dérivable ?

<details><summary>Indice 1</summary>

Comment entraîne-t-on un réseau à plusieurs couches ? De quoi cette méthode a-t-elle besoin ?

</details>
<details><summary>Indice 2</summary>

La descente de gradient a besoin de la dérivée de la sortie par rapport à chaque poids. Que vaut la dérivée du seuil ? Et sans aucune non-linéarité, que calcule une pile de couches ?

</details>
<details><summary>Indice 3</summary>

« La rétropropagation calcule des gradients : la dérivée du seuil est nulle partout, aucun signal ne passe. Une bonne activation est non linéaire (sinon les couches s'effondrent en une seule fonction affine), dérivable presque partout avec une pente utile, et peu coûteuse : ReLU, GELU, SiLU. »

</details>

### 10.E4 — Un réseau de neurones ressemble-t-il au cerveau ?

<details><summary>Indice 1</summary>

Reprends ton oral 🗣️ 10.8, et ajoute des ordres de grandeur (🧮 10.9).

</details>
<details><summary>Indice 2</summary>

L'inspiration historique, puis les différences (unité, apprentissage, énergie, échelle), puis ce qui fait vraiment marcher les réseaux actuels.

</details>
<details><summary>Indice 3</summary>

« L'inspiration est réelle en 1943 et 1957, la ressemblance est très lointaine aujourd'hui : un neurone artificiel est une somme et une activation ; l'apprentissage par rétropropagation n'a pas d'équivalent établi dans le cerveau ; le cerveau consomme 20 W. Ce qui fait marcher les réseaux, ce sont les données, le calcul et les architectures. »

</details>

<a id="notebook"></a>

## Notebook, parties A à C

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/perceptron.py`, la docstring de chaque fonction décrit déjà ce qu'elle doit faire, cas d'erreur compris : relis-la avant d'ouvrir un indice.

### Ex 10.12 — sign_step et add_bias_column 🔨

<details><summary>Indice 1</summary>

Deux fonctions d'une ou deux lignes chacune, sans boucle : une comparaison vectorisée, puis une concaténation de colonnes.

</details>
<details><summary>Indice 2</summary>

`np.where(condition, valeur_si_vrai, valeur_si_faux)` travaille sur tout un tableau ; convertis d'abord `z` avec `np.asarray(z, dtype=float)`. Pour la colonne : `np.ones((n, 1))` et `np.hstack`, après avoir vérifié `X.ndim`.

</details>
<details><summary>Indice 3</summary>

```python
def sign_step(z):
    return np.where(np.asarray(z, dtype=float) > 0, 1.0, -1.0)


def add_bias_column(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.ndim} dimension(s)")
    return np.hstack([np.ones((X.shape[0], 1)), X])
```

</details>

### Ex 10.13 — AND, OR, XOR : le perceptron va-t-il converger ? 🔮

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la séparabilité et la fin de l'encadré sur la convergence : que se passe-t-il sur des données séparables, et sur des données qui ne le sont pas ?

</details>
<details><summary>Indice 2</summary>

Pour c), fais une epoch de XOR à la main, dans l'ordre (0, 0), (0, 1), (1, 0), (1, 1), à partir de poids nuls : où en sont les poids à la fin ? Pour b), déduis-en ce que prédit le perceptron pour chacune des quatre entrées à la fin d'une epoch.

</details>
<details><summary>Indice 3</summary>

Les quatre corrections de l'epoch s'annulent exactement. Avec des poids nuls, toutes les sommes valent 0, et scikit-learn répond la première classe pour une somme nulle.

</details>

### Ex 10.14 — neuron_forward : un neurone appliqué à tout un batch 🔨

<details><summary>Indice 1</summary>

Trois étapes : convertir et vérifier les formes, calculer toutes les sommes pondérées d'un coup, appliquer l'activation au tableau entier.

</details>
<details><summary>Indice 2</summary>

`X @ w` marche pour `X` de forme `(n, d)` comme `(d,)`. La dernière dimension de `X` est `X.shape[-1]`. Pour un exemple seul, `X @ w + b` est un scalaire NumPy : `np.asarray(...)` en fait un tableau 0-d.

</details>
<details><summary>Indice 3</summary>

```python
def neuron_forward(X, w, b=0.0, activation=sign_step):
    X, w = np.asarray(X, dtype=float), np.asarray(w, dtype=float)
    if X.shape[-1] != len(w):
        raise ValueError(f"X has {X.shape[-1]} feature(s) but w has {len(w)} weight(s)")
    return np.asarray(activation(X @ w + b), dtype=float)
```

</details>

### Ex 10.15 — Des noms de poids (AD, BE…) à la matrice W 🔨

<details><summary>Indice 1</summary>

Une ligne de $\mathbf{W}$ par source, une colonne par neurone : le nom du poids en ligne `j`, colonne `k`, est la lettre de la source `j` suivie de celle du neurone `k`.

</details>
<details><summary>Indice 2</summary>

Une liste de listes en compréhension, `[[weights[s + t] for t in targets] for s in sources]`, puis `np.array`. Pour la couche : `X @ W + b`, où `b` s'ajoute à chaque ligne par *broadcasting*. c) Quelle est la forme de `W`, et celle qu'attend `nn.Linear(3, 3)` ?

</details>
<details><summary>Indice 3</summary>

```python
def names_to_matrix_15(weights, sources, targets):
    return np.array([[weights[s + t] for t in targets] for s in sources], dtype=float)


def layer_forward_15(X, W, b, activation=lambda z: z):
    return activation(np.asarray(X, dtype=float) @ np.asarray(W, dtype=float) + np.asarray(b, dtype=float))
```
c) Les deux formes sont $(3, 3)$ : PyTorch ne voit rien.

</details>

### Ex 10.16 — XOR avec trois neurones câblés à la main 🔨

<details><summary>Indice 1</summary>

Reprends l'idée de ✏️ 10.5 : un neurone OR, un neurone NAND, puis un AND de leurs deux sorties. Calcule chaque neurone avec `neuron_forward`.

</details>
<details><summary>Indice 2</summary>

Les deux neurones cachés reçoivent `X`. Le neurone de sortie reçoit `np.column_stack([h1, h2])`, dont les valeurs sont $-1$ ou $+1$ : sa somme $h_1 + h_2 + b$ vaut $-2 + b$, $b$ ou $2 + b$ ; choisis $b$ pour que seul le dernier cas soit positif.

</details>
<details><summary>Indice 3</summary>

OR : $\mathbf{w} = (1, 1)$, $b = -0{,}5$ ; NAND : $\mathbf{w} = (-1, -1)$, $b = 1{,}5$ ; sortie : $\mathbf{w} = (1, 1)$, $b = -1$ (tout biais strictement entre $-2$ et 0 convient). Trois neurones de deux poids et un biais : 9 nombres.

</details>

### Ex 10.17 — Le learning rate change-t-il un perceptron qui part de zéro ? 🔮

<details><summary>Indice 1</summary>

Pars de la règle d'apprentissage de la fiche (encadré 🧮) : dans une correction, qu'est-ce qui dépend de $\eta$, et qu'est-ce qui décide s'il y a correction ?

</details>
<details><summary>Indice 2</summary>

Partis de zéro, suppose que deux perceptrons, l'un avec $\eta$, l'autre avec $100\,\eta$, ont fait les mêmes corrections jusqu'ici : quelle relation lie leurs poids ? Et leurs sommes $z$ pour l'exemple suivant ? Avec un départ non nul, cette relation tient-elle encore ?

</details>
<details><summary>Indice 3</summary>

Par récurrence, les poids du second sont 100 fois ceux du premier, donc les signes de $z$ sont les mêmes, donc les corrections aussi. Avec un départ $\mathbf{w}_0 \ne \mathbf{0}$, les poids valent $\mathbf{w}_0 + \eta \times (\ldots)$ : le départ, lui, n'est pas multiplié par 100, et le rapport entre le départ et les corrections dépend de $\eta$.

</details>

### Ex 10.18 — Le perceptron de scikit-learn sur portes logiques et manchots 📦

<details><summary>Indice 1</summary>

Une petite fonction qui crée `SklearnPerceptron(max_iter=100, shuffle=False, tol=None)` et l'entraîne t'évitera de répéter les paramètres cinq fois.

</details>
<details><summary>Indice 2</summary>

`GATES["or"]` est un couple `(X, y)` : `model.fit(*GATES["or"])`. `coef_` a la forme `(1, 2)` : prends `coef_[0]`, et `intercept_[0]`. Pour d) et e), entraîne sur les manchots d'entraînement et note avec `score` sur ceux de test.

</details>
<details><summary>Indice 3</summary>

```python
or_model_18 = SklearnPerceptron(max_iter=100, shuffle=False, tol=None).fit(*GATES["or"])
or_params_18 = [*or_model_18.coef_[0].tolist(), float(or_model_18.intercept_[0])]
or_n_iter_18 = or_model_18.n_iter_
raw_accuracy_18 = (SklearnPerceptron(max_iter=100, shuffle=False, tol=None)
                   .fit(X_peng[train_17], y_train_17).score(X_peng[test_17], y_test_17))
```

</details>

### Ex 10.19 — Erreurs par epoch : séparable ou pas ? 📈

<details><summary>Indice 1</summary>

Lis chaque courbe jusqu'au bout : est-elle à 0 à la fin, et y reste-t-elle ?

</details>
<details><summary>Indice 2</summary>

Le graphique du milieu a une échelle logarithmique en abscisse : les epochs 3, 30 et 300 sont espacées régulièrement. Pour d), regarde la courbe A de près, epoch par epoch. Pour e), regarde le nuage de points de E, à droite. Pour f), relis la fin de l'encadré 🧮 sur la convergence : de quoi dépend le nombre de corrections ?

</details>
<details><summary>Indice 3</summary>

Trois courbes finissent à 0. XOR reste sur une valeur constante, celle d'un perceptron aux poids nuls. La courbe A touche 0 puis rebondit. E ne peut pas atteindre 0 (une droite ne sépare pas ces deux espèces avec les deux mesures du bec). Une marge minuscule peut demander un nombre de corrections énorme : une courbe qui n'atteint pas 0 ne prouve rien.

</details>

### Ex 10.20 — Docstring NumPy et doctest pour neuron_forward 🛠️

<details><summary>Indice 1</summary>

Exécute d'abord tes deux exemples dans une cellule du notebook : ce qu'affiche Python est exactement ce que tu dois recopier sous chaque ligne `>>>`.

</details>
<details><summary>Indice 2</summary>

Dans la docstring, les exemples existants définissent déjà `X` et `w` : tu peux les réutiliser. Écris `>>> neuron_forward(np.array([1.0, 2.0]), w, b=-1.0, activation=lambda z: z)` puis la sortie sur la ligne suivante, avec la même indentation ; de même pour `add_bias_column(X)`. Une ligne vide sépare un exemple d'une phrase d'explication.

</details>
<details><summary>Indice 3</summary>

```text
    A single sample gives a 0-d array:

    >>> neuron_forward(np.array([1.0, 2.0]), w, b=-1.0, activation=lambda z: z)
    array(2.)

    The bias trick: the bias becomes the weight of a constant input 1.

    >>> neuron_forward(add_bias_column(X), np.array([-1.0, 1.0, 1.0]))
    array([ 1., -1.])
```
Si doctest affiche `np.float64(2.0)` pour le premier, c'est ta fonction qu'il faut corriger (10.14), pas l'exemple.

</details>

### Ex 10.21 — La classe Perceptron et sa règle d'apprentissage 🔨

<details><summary>Indice 1</summary>

Organise `fit` en quatre temps : les contrôles ; le codage des labels en $\pm 1$ ; la double boucle (les epochs, puis les exemples d'une epoch) ; l'enregistrement des attributs. Refais d'abord ✏️ 10.6 si ce n'est pas fait.

</details>
<details><summary>Indice 2</summary>

`classes = np.unique(y)`, puis `target = np.where(y == classes[1], 1.0, -1.0)`. L'ordre d'une epoch : `rng.permutation(n)` si `shuffle`, sinon `range(n)`, avec `rng` créé **dans** `fit`. Compte les corrections dans un entier Python, ajoute-le à `errors_`, et sors de la boucle des epochs s'il vaut 0. Termine par `self.intercept_ = float(b)` et `return self`.

</details>
<details><summary>Indice 3</summary>

```python
target = np.where(y == classes[1], 1.0, -1.0)
w, b = np.zeros(X.shape[1]), 0.0
rng = np.random.default_rng(self.random_state)
errors = []
for _ in range(self.max_iter):
    order = rng.permutation(len(X)) if self.shuffle else range(len(X))
    mistakes = 0
    for i in order:
        if target[i] * (X[i] @ w + b) <= 0:
            w = w + self.eta0 * target[i] * X[i]
            if self.fit_intercept:
                b += self.eta0 * target[i]
            mistakes += 1
    errors.append(mistakes)
    if mistakes == 0:
        break
```

</details>

### Ex 10.22 — Perceptron piégé : quatre bugs classiques 🐛

<details><summary>Indice 1</summary>

Le collègue part de poids nuls : que vaut $y z$ pour le premier exemple ? Puis relis sa boucle ligne par ligne, avec le tableau des pièges ⚠️ de la fiche sous les yeux.

</details>
<details><summary>Indice 2</summary>

Dans l'ordre des diagnostics, cherche : le comparateur du test d'erreur ; la valeur de `yi` quand les labels sont 0 et 1 ; la ligne qui manque pour le biais ; ce que fait `rng.shuffle(X)` à `X`, et à `y`. Un `np.asarray` sur un tableau déjà en flottants renvoie-t-il une copie ?

</details>
<details><summary>Indice 3</summary>

Les quatre corrections : tester `<= 0` ; coder les labels en $\pm 1$ (`target = np.where(y == self.classes_[1], 1.0, -1.0)`) ; ajouter `self.intercept_ += self.eta0 * target[i]` ; parcourir `rng.permutation(len(X))` au lieu de mélanger `X`.

</details>

### Ex 10.23 — Marge et vitesse de convergence : la borne (R/γ)² à l'épreuve 🔬

<details><summary>Indice 1</summary>

Deux petites fonctions : la marge est un minimum de produits scalaires, le nombre de corrections est la somme de `errors_` de ton `Perceptron`.

</details>
<details><summary>Indice 2</summary>

`margin_23` : normalise `u`, code les labels en $\pm 1$, puis `np.min(signs * (X @ u))`. `updates_23` : `Perceptron(fit_intercept=False, max_iter=100_000, shuffle=True, random_state=seed)`, entraîné, puis `sum(model.errors_)`.

</details>
<details><summary>Indice 3</summary>

```python
def margin_23(X, y, u):
    u = np.asarray(u, dtype=float) / np.linalg.norm(u)
    signs = np.where(np.asarray(y) == 1, 1.0, -1.0)
    return float(np.min(signs * (np.asarray(X, dtype=float) @ u)))


def updates_23(X, y, seed):
    model = mylearn.perceptron.Perceptron(fit_intercept=False, max_iter=100_000, shuffle=True, random_state=seed)
    return int(sum(model.fit(X, y).errors_))
```

</details>

### Ex 10.24 — Trois espèces de manchots avec des perceptrons en un-contre-tous 🔬

<details><summary>Indice 1</summary>

Tes méta-classifieurs du ch. 7 reçoivent un classifieur binaire **non entraîné**, et le copient pour chaque sous-problème.

</details>
<details><summary>Indice 2</summary>

`mylearn.multiclass.OneVsRestClassifier(mylearn.perceptron.Perceptron(max_iter=100)).fit(X_train, y_train)`, et de même pour `OneVsOneClassifier`. b) Un perceptron a convergé s'il finit par une epoch sans correction : regarde ce qu'affiche la vérification.

</details>
<details><summary>Indice 3</summary>

Les deux fonctions tiennent en une ligne chacune. b) Deux des trois perceptrons un-contre-tous corrigent encore à la 100e epoch : leurs initiales, dans l'ordre alphabétique.

</details>

### Ex 10.25 — Défi « Mark I » : un perceptron sur des chiffres de 20 × 20 pixels 🏆

<details><summary>Indice 1</summary>

Commence par `crop_25` (une découpe de tableau et une division) et la part de l'encre (deux sommes). Puis regarde pourquoi le point de départ échoue : après 100 epochs, son perceptron corrige-t-il encore des images ? Que valent alors les poids de la **dernière** epoch ? Relis la fin de l'encadré 🧮 sur la convergence.

</details>
<details><summary>Indice 2</summary>

`images[:, 4:24, 4:24].reshape(len(images), 400) / 255`. Pour la méthode, deux idées classiques : faire la **moyenne** des poids au fil de l'entraînement (le perceptron moyenné), ou garder les poids de la meilleure epoch (l'algorithme « pocket »). Mélanger les exemples à chaque epoch aide aussi. Compare tes variantes avec la validation croisée qu'affiche la vérification, sans révéler le test.

</details>
<details><summary>Indice 3</summary>

Le perceptron moyenné : la règle classique, exemples mélangés, une dizaine d'epochs ; après **chaque** exemple (corrigé ou non), ajoute les poids et le biais courants à deux sommes, et renvoie leur moyenne à la fin. Il dépasse 0,95 au test.

</details>
