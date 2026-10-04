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

a) Le k-means (A) calcule des moyennes et des distances aux centroïdes, sans aucun neurone : A fait partie de ta réponse. Applique le même test à B, C, D et E : la méthode obtient-elle son résultat par une formule ou un algorithme dédié, ou par des couches de sommes pondérées ? Pour B, ne te laisse pas piéger par la **forme** $\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$ : regarde comment la méthode **calcule** ses poids. b) Une forme fermée calcule sa solution directement : a-t-elle besoin d'unités qui ajustent leurs poids ? c) Compte les couches de poids d'un perceptron, puis relis ce que « profond » exige.

</details>

### 10.Q2 — Le neurone biologique en quatre étapes

<details><summary>Indice 1</summary>

Relis la liste numérotée de la section 10.2 de la fiche, et suis le trajet d'un message : il arrive, il est traité, il repart.

</details>
<details><summary>Indice 2</summary>

Qu'est-ce qui déclenche les signaux électriques ? Que faut-il faire de plusieurs signaux avant de les comparer à un seuil ? Que se passe-t-il quand le seuil est dépassé ? Pour b), les signaux sont-ils tous de même signe ? Pour c), qu'est-ce que la fente synaptique ?

</details>
<details><summary>Indice 3</summary>

a) Tout commence quand des neurotransmetteurs se fixent sur les récepteurs du neurone : B est la première étape. Pour placer les trois autres, suis le trajet du message : que faut-il avoir calculé avant de pouvoir comparer un total à un seuil ? Et que déclenche une décharge ? b) Relis l'étape 2 de la liste de la fiche §10.2 : les signaux qui arrivent ont-ils tous le même effet sur la décharge ? c) Relis la définition de deux neurones « connectés » dans la fiche, puis la description d'une synapse chimique : les deux cellules y sont-elles en contact ? Le « en général » de l'énoncé vise le cas le plus courant.

</details>

### 10.Q3 — Connectome et émulation du cerveau : vrai ou faux justifié

<details><summary>Indice 1</summary>

Relis la fin de la section 10.2 de la fiche et son encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

b) Le livre compare les connectomes à une empreinte digitale. c) Qu'ont produit les puces neuromorphiques et les simulations géantes ? d) Cherche « cognition incarnée ». e) Cherche « FlyWire ».

</details>
<details><summary>Indice 3</summary>

a) Vrai : c'est la définition même du connectome, la carte des connexions d'un individu (fiche §10.2). Pour les quatre autres, appuie chaque verdict sur un fait précis, en une phrase : b) deux empreintes digitales sont-elles identiques, et un câblage qui change avec l'expérience peut-il l'être d'une personne à l'autre ? c) compare l'affirmation au constat du livre, puis aux machines de 2024 et 2025 de l'encadré 🕰️ : ont-elles produit une intelligence générale ? d) compare avec la phrase de la fiche sur la cognition incarnée ; e) compare mot à mot avec la phrase de l'encadré 🕰️ sur FlyWire : même animal, même stade (adulte), même année ?

</details>

### 10.Q4 — Neurone, unité, « cerveau électronique » : bien nommer les choses

<details><summary>Indice 1</summary>

Relis la section 10.3 de la fiche, en particulier la comparaison avec un plan de métro.

</details>
<details><summary>Indice 2</summary>

Que garde un neurone artificiel d'un vrai neurone, et que laisse-t-il de côté ? Un mot plus neutre que « neurone » éviterait quelle confusion ?

</details>
<details><summary>Indice 3</summary>

a) « Unité » est un mot neutre : il rappelle que le neurone artificiel n'est qu'une abstraction très simplifiée d'un vrai neurone. C'est A ; les trois autres propositions parlent de vitesse, de poids ou de réseaux impulsionnels, sans rapport avec le choix d'un mot. b) Reprends la comparaison de la fiche §10.3 : un plan de métro est-il la ville ? Transpose au réseau de neurones et au cerveau. c) Garde la proposition qui figure dans « l'idée centrale » de la fiche §10.3, et vérifie que les trois autres sont dans la liste de ce que le neurone artificiel laisse de côté.

</details>

### 10.Q5 — McCulloch et Pitts (1943) : ce qu'ils ont démontré

<details><summary>Indice 1</summary>

Relis le paragraphe « 1943 : le neurone formel » de la section 10.3.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

a) Leur résultat porte sur la logique, pas sur l'apprentissage. b) Qui a introduit des poids qui s'ajustent à partir d'exemples ? c) Écris la table : pour (0, 0), (0, 1), (1, 0) et (1, 1), la somme atteint-elle 2 ?

</details>
<details><summary>Indice 3</summary>

a) Leur article relie neurones et logique : pour toute expression logique (sous certaines conditions), on peut construire un réseau de neurones formels qui se comporte comme elle. C'est B : il ne parle pas d'apprentissage (A), ne dit pas que le cerveau est un ordinateur (C), et il lui faut un réseau, pas un neurone seul (D). b) Dans le modèle de 1943, d'où viennent les poids et les seuils : d'un algorithme qui les ajuste sur des exemples, ou du choix des auteurs ? La fin du paragraphe « 1943 : le neurone formel » de la fiche le dit. c) Écris la table : pour chacune des entrées (0, 0), (0, 1), (1, 0) et (1, 1), la somme $x_1 + x_2$ atteint-elle 2 ? Compare la colonne des sorties aux tables de OR, AND, XOR et NAND.

</details>

### 10.Q6 — Anatomie d'un perceptron

<details><summary>Indice 1</summary>

Relis la formule du perceptron de la section 10.3.1 de la fiche, et la phrase qui la suit.

</details>
<details><summary>Indice 2</summary>

a) Combien de poids un perceptron a-t-il : un par entrée, ou un seul ? b) La condition pour sortir $+1$ est-elle $z > 0$ ou $z \ge 0$ ? c) Le perceptron a-t-il toujours la même paire de sorties, d'un auteur à l'autre ?

</details>
<details><summary>Indice 3</summary>

a) Le perceptron multiplie chaque entrée par **son** poids, additionne les produits, puis compare la somme à 0 : c'est A (B n'a qu'un poids, C fait voter les entrées, D les multiplie entre elles). b) Applique la formule de la fiche §10.3.1 à $z = 0$ : la condition $z > 0$ est-elle remplie ? Lis alors la ligne « sinon » de la formule. c) Relis les deux phrases qui suivent la formule du perceptron dans la fiche, et compare-les à l'affirmation.

</details>

### 10.Q7 — Mark I, Minsky-Papert, renaissance : la chronologie

<details><summary>Indice 1</summary>

Regarde la chronologie de la section 10.3.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Les dates : 1943, 1957, 1960, 1969, 1986. b) Relis la description de la machine : comment ses cellules étaient-elles disposées ? d) Sur quel type de perceptron portaient les démonstrations de Minsky et Papert ?

</details>
<details><summary>Indice 3</summary>

a) L'événement le plus ancien est l'article de McCulloch et Pitts (B, 1943) : c'est la première lettre. Associe chacun des quatre autres à l'une des dates de l'indice 2, puis range-les. b) Les cellules forment un carré de 20 de côté : combien en tout ? (20 n'est que le côté, et 512 le nombre d'unités d'association, pas de cellules.) c) Relis ce que garantit le théorème de convergence, et ce qui arrive sur XOR, qui n'a pourtant aucun bruit : une seule proposition est compatible avec les deux. d) Sur combien de couches de poids appris portaient les démonstrations de *Perceptrons* ? Et un réseau à deux couches peut-il calculer XOR (✏️ 10.5) ?

</details>

### 10.Q8 — Du perceptron au neurone moderne : les deux changements

<details><summary>Indice 1</summary>

Relis le début de la section 10.3.3 de la fiche : un changement à l'entrée, un à la sortie.

</details>
<details><summary>Indice 2</summary>

b) Compte un poids par entrée, et n'oublie pas ce qu'on ajoute à la somme. d) Que vaut $\mathbf{w}\cdot\mathbf{0}$ ?

</details>
<details><summary>Indice 3</summary>

a) Le changement à l'entrée est le **biais**, ajouté à la somme pondérée : A est l'une des deux lettres. Cherche l'autre à la sortie : par quoi remplace-t-on le test « $z > 0$ ? » ? b) Un poids par entrée, plus le nombre propre au neurone que tu as trouvé en a) : fais l'addition. c) Relis l'astuce du biais dans la fiche §10.3.3 : quelle entrée ajoute-t-on, et quel est son poids ? d) Avec $b = 0$, calcule $z$ en $\mathbf{x} = \mathbf{0}$ : où l'origine se trouve-t-elle par rapport à la frontière $z = 0$ ?

</details>

### 10.Q9 — Lire un schéma de réseau : poids implicites et convention AD/DA

<details><summary>Indice 1</summary>

Relis les paragraphes « Dessiner un neurone » et « Nommer les poids » de la section 10.3.3 de la fiche, et regarde la figure de la couche.

</details>
<details><summary>Indice 2</summary>

b) Chaque neurone de la première couche est relié à chacun de la seconde. c) La question ne porte que sur D et E : compte leurs poids d'entrée et leurs biais. d) Dans le livre, le nom d'un poids commence par la source. e) Relis l'encadré 🕰️ sur les conventions.

</details>
<details><summary>Indice 3</summary>

a) Faux : sur un schéma, les poids sont implicites ; chaque flèche entre deux neurones en porte un, même quand rien ne l'indique. b) Chaque neurone de la première couche envoie une flèche à chaque neurone de la seconde : compte les flèches (nombre de sources × nombre de destinations). c) Ajoute à b) les biais des seuls neurones D et E. d) Dans la convention du livre, la première lettre est la source et la seconde la destination : lis BE dans cet ordre, puis cherche la proposition qui lui correspond. e) Relis l'encadré 🕰️ sur les conventions des bibliothèques : quelle forme donne-t-il au `weight` de `nn.Linear` ?

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

a) Avec $\lambda = 0$ : $w = 20 / (10 + 0) = 2$. b) Même formule, $w = 20 / (10 + \lambda)$, avec $\lambda = 10$. c) Le numérateur reste 20 : que devient la fraction quand son dénominateur grandit sans limite ? d) Relis le paragraphe « À l'entrée : le biais » de la fiche §10.3.3 : que dit-il du biais et de la régularisation L2 ? e) Demande-toi ce que le biais change à la frontière (sa position, ou sa souplesse ?), puis ce que ferait sa pénalité si l'on ajoutait une même constante à toutes les cibles.

</details>

### 10.R2 — Ch. 7 : la frontière du centroïde le plus proche, w·x + b = 0

<details><summary>Indice 1</summary>

Développe $\lVert \mathbf{x} - \boldsymbol{\mu} \rVert^2 = \lVert \mathbf{x} \rVert^2 - 2\,\boldsymbol{\mu}\cdot\mathbf{x} + \lVert \boldsymbol{\mu} \rVert^2$ pour les deux centroïdes.

</details>
<details><summary>Indice 2</summary>

Le terme $\lVert \mathbf{x} \rVert^2$ disparaît des deux côtés. Il reste une inégalité entre $2\,(\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-)\cdot\mathbf{x}$ et $\lVert \boldsymbol{\mu}_+ \rVert^2 - \lVert \boldsymbol{\mu}_- \rVert^2$ ; divise par 2.

</details>
<details><summary>Indice 3</summary>

a) $w_1 = \mu_{+,1} - \mu_{-,1} = 2 - 0 = 2$. b) Même calcul sur la seconde coordonnée. c) $b = (\lVert \boldsymbol{\mu}_- \rVert^2 - \lVert \boldsymbol{\mu}_+ \rVert^2)/2$ : calcule les deux normes au carré, $0^2 + 1^2$ et $2^2 + 3^2$, puis leur demi-différence, **dans cet ordre**. d) Calcule $\mathbf{w}\cdot\mathbf{x} + b$ pour $\mathbf{x} = (3, 2)$ avec tes valeurs : son signe donne la classe (contrôle avec les deux distances au carré). e) Compare la règle obtenue, « $+1$ si $\mathbf{w}\cdot\mathbf{x} + b > 0$ », à celle d'un perceptron : est-ce la forme de la décision qui diffère, ou la façon d'obtenir $\mathbf{w}$ et $b$ ?

</details>

### 10.R3 — Ch. 3 : matrice de confusion et accuracy d'un classifieur binaire

<details><summary>Indice 1</summary>

Range les dix exemples dans les quatre cases : vrai label $\pm 1$, prédiction $\pm 1$.

</details>
<details><summary>Indice 2</summary>

Les quatre premiers exemples sont les vrais $+1$ ; les six autres, les vrais $-1$. Le recall divise par les vrais positifs plus les faux négatifs ; la precision, par les vrais positifs plus les faux positifs.

</details>
<details><summary>Indice 3</summary>

a) Les quatre premiers exemples sont les vrais $+1$, prédits $+1, +1, -1, +1$ : TP = 3. b) à d) Même méthode : un FP est un vrai $-1$ prédit $+1$ (cherche-les parmi les six derniers), un FN un vrai $+1$ prédit $-1$, un TN un vrai $-1$ prédit $-1$ ; les quatre cases doivent totaliser 10. e) $(\text{TP} + \text{TN}) / 10$. f) $\text{TP} / (\text{TP} + \text{FN})$. g) $\text{TP} / (\text{TP} + \text{FP})$.

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

a) $z^{(1)} = 0{,}5 \times 2 + (-1) \times 1 + 2 \times 0{,}5 + 0{,}25 \times (-4) = 1 - 1 + 1 - 1 = 0$. b) Applique la règle « $+1$ si $z > 0$, $-1$ sinon » à cette somme. c) Même calcul avec $\mathbf{x}^{(2)}$ : $0{,}5 \times 1 + (-1) \times 3 + 2 \times 1 + 0{,}25 \times 4$. d) Le seuil appliqué à c). e) et f) Ajoute $b = -0{,}75$ à la somme sans biais de chaque entrée avant de prendre le seuil (pour $\mathbf{x}^{(3)}$, calcule d'abord ses quatre produits). g) Compte les $+1$ parmi les trois sorties avec biais. h) La somme de a), avec la règle « 1 si $z > 0$, 0 sinon ». i) Pour chaque entrée, écris la condition « somme sans biais $+\,b > 0$ », garde la plus exigeante des trois, puis cherche le plus petit entier qui la vérifie.

</details>

### Ex 10.2 — L'astuce du biais : même neurone, une entrée de plus ✏️

<details><summary>Indice 1</summary>

Relis l'astuce du biais dans la section 10.3.3 de la fiche : le biais devient le **premier** poids, celui d'une entrée constante 1.

</details>
<details><summary>Indice 2</summary>

Le vecteur augmenté commence par le biais, puis les poids ; $\tilde{\mathbf{x}}$ commence par 1, puis les entrées. Pour le batch, chaque ligne devient $(1, x_1, x_2)$. f) La matrice augmentée a une ligne par entrée, plus une ligne de biais, et une colonne par neurone. g) Écris les trois conditions données par $(1, 0)$, $(0, 1)$ et $(1, 1)$ quand $b = 0$.

</details>
<details><summary>Indice 3</summary>

a) Le biais vient en tête : $\tilde{\mathbf{w}} = (b, w_1, w_2) = (0{,}5;\ 2;\ -1)$. b) $\tilde{\mathbf{x}} = (1, 1, 3)$, puis $z = 0{,}5 \times 1 + 2 \times 1 + (-1) \times 3$. c) Les lignes de $\tilde{\mathbf{X}}$ sont $(1, 1, 3)$, $(1, 0, 0)$ et $(1, 2, 1)$ : fais le produit scalaire de chacune avec $\tilde{\mathbf{w}}$. d) Applique le seuil à chacune des trois sommes de c). e) Développe $\tilde{\mathbf{X}}\tilde{\mathbf{w}}$ ligne par ligne : est-ce autre chose que $\mathbf{X}\mathbf{w} + b$ ? f) $\tilde{\mathbf{W}}$ a la forme $(n_{\text{in}} + 1, n_{\text{out}})$ : remplace par les nombres de l'énoncé et multiplie. g) et h) Sans biais, $(1, 0)$ doit donner 0, donc $w_1 \le 0$ ; écris de même les conditions imposées par $(0, 1)$ et par $(1, 1)$, puis regarde si les trois peuvent être vraies ensemble.

</details>

### Ex 10.3 — Portes logiques à la main : AND, OR, NOT, NAND ✏️

<details><summary>Indice 1</summary>

Pars de l'exemple AND de la fiche ($\mathbf{w} = (1, 1)$, $b = -1{,}5$) : la droite $x_1 + x_2 = 1{,}5$ laisse un seul coin du côté positif. Pour chaque porte, cherche où placer la droite, puis vérifie les entrées une à une.

</details>
<details><summary>Indice 2</summary>

OR : la même direction $\mathbf{w} = (1, 1)$, mais une droite qui laisse trois coins du côté positif. NOT : un seul poids, négatif. NAND : que devient la sortie si l'on change le signe de $z$ ? Majorité : trois poids égaux, et un seuil entre 1 et 2. f) Que devient le signe de $c\,z$ si $c > 0$ ? si $c < 0$ ?

</details>
<details><summary>Indice 3</summary>

a) OR : garde la direction $\mathbf{w} = (1, 1)$ d'AND et déplace la droite pour ne laisser que $(0, 0)$ du côté négatif ; $b = -0{,}5$ convient, avec les sommes $-0{,}5$ ; $0{,}5$ ; $0{,}5$ ; $1{,}5$. b) NOT : un seul poids, négatif ; choisis $b$ pour que la somme soit positive en $x = 0$ et ne le soit plus en $x = 1$. c) NAND : change le signe de $\mathbf{w}$ et de $b$ d'AND, et recalcule les quatre sommes. Pour le « pourquoi », regarde ce que devient une sortie quand sa somme change de signe, puis quand sa somme vaut 0 (son opposé vaut encore 0). d) NOR : la même astuce, appliquée à ton OR. e) Majorité : avec $\mathbf{w} = (1, 1, 1)$, la somme sans biais vaut 0, 1, 2 ou 3 selon le nombre d'entrées à 1 ; choisis $b$ pour que $1 + b \le 0$ et $2 + b > 0$. f) Compare le signe de $c\,z$ à celui de $z$, pour $c > 0$ puis pour $c < 0$, sans oublier le cas $z = 0$. g) Trace la droite $z = 0$ de ton OR et vérifie que seul $(0, 0)$ reste du côté négatif.

</details>

### Ex 10.4 — Pourquoi un seul perceptron ne peut pas calculer XOR ∂

<details><summary>Indice 1</summary>

Écris une inégalité par entrée : sortie 1 si $s > 0$, sortie 0 si $s \le 0$.

</details>
<details><summary>Indice 2</summary>

a) Pour chaque entrée, remplace $x_1$ et $x_2$ par leurs valeurs dans $s$, et écris $s > 0$ ou $s \le 0$ selon la sortie voulue. b) Additionne les deux inégalités strictes, puis compare avec celle de $(1, 1)$ en te servant de celle de $(0, 0)$. c) $s$ est affine : la valeur au milieu d'un segment est la moyenne des valeurs aux deux bouts. e) Avec $x_3 = x_1 x_2$, le point $(1, 1)$ reçoit une coordonnée de plus, qui peut le pousser du côté négatif.

</details>
<details><summary>Indice 3</summary>

a) $(0, 0) \mapsto 0$ donne $s(0, 0) = b \le 0$ ; écris de même les trois autres, en remplaçant $x_1$ et $x_2$ dans $s$ (inégalité stricte pour une sortie 1, large pour une sortie 0). b) Additionne les inégalités de $(0, 1)$ et de $(1, 0)$, fais apparaître $w_1 + w_2 + b$, et sers-toi de $b \le 0$ pour en tirer le signe de cette quantité ; confronte-le à l'inégalité de $(1, 1)$. c) $(0{,}5 ; 0{,}5)$ est le milieu des **deux** diagonales du carré. Comme $s$ est affine, sa valeur au milieu d'un segment est la moyenne de ses valeurs aux deux bouts : écris-le pour chaque diagonale, et compare les signes obtenus. d) XNOR a les sorties de XOR, échangées : refais b) en échangeant le rôle des deux paires de points. e) Le coin $(1, 1)$ reçoit $x_3 = 1$, les trois autres $x_3 = 0$ : garde les poids d'OR pour $x_1$, $x_2$ et $b$, et choisis $w_3$ assez négatif pour que la somme de $(1, 1)$ passe à 0 ou en dessous. Pour les chapitres, pense aux features qu'on ajoute aux données au ch. 9, et à une méthode du ch. 13 qui fait le même travail sans les calculer. f) Une table de vérité à deux entrées a 4 lignes, chacune de sortie 0 ou 1 : combien de tables ? L'argument de c) condamne toute table qui donne une même sortie aux deux bouts d'une diagonale, et l'autre sortie aux deux bouts de l'autre ; pour chacune des autres tables, cherche une droite (les constantes, $x_1$, $x_2$, les portes de 10.3…).

</details>

### Ex 10.5 — XOR en deux couches : câbler et nommer les poids ✏️

<details><summary>Indice 1</summary>

Calcule le réseau couche par couche : d'abord C et D pour les quatre entrées, puis E à partir des sorties de C et de D (pas des entrées A et B).

</details>
<details><summary>Indice 2</summary>

C : $A + B - 0{,}5 > 0$ ? D : $-A - B + 1{,}5 > 0$ ? E : $C + D - 1{,}5 > 0$ ? e) La convention de mylearn met une ligne par **source** (A, B) et une colonne par neurone (C, D) : $W_{AC}$ en ligne A, colonne C. h) Remplace chaque seuil par l'identité et développe la sortie de E.

</details>
<details><summary>Indice 3</summary>

a) Pour C, $A + B - 0{,}5$ vaut $-0{,}5$ ; $0{,}5$ ; $0{,}5$ ; $1{,}5$ sur les quatre entrées, d'où les sorties $[0, 1, 1, 1]$. b) Même méthode pour D, avec $-A - B + 1{,}5$. c) Calcule $C + D - 1{,}5$ entrée par entrée, avec les sorties de a) et de b), pas avec A et B. d) Compare chaque liste aux tables de 10.3 (et à celle de XOR). e) Ligne A : $(AC, AD)$ ; ligne B : $(BC, BD)$ ; remplace chaque nom par sa valeur. f) Une ligne par neurone : ligne C, $(AC, BC)$ ; ligne D, $(AD, BD)$. g) Compte les flèches (un poids chacune), puis un biais par neurone qui calcule (les entrées A et B n'en ont pas). h) Avec l'identité partout, remplace C et D par leurs expressions dans $E = C + D - 1{,}5$ et développe : le résultat a-t-il la forme $v_1 x_1 + v_2 x_2 + c$ ?

</details>

### Ex 10.6 — Une epoch de la règle du perceptron à la main ✏️

<details><summary>Indice 1</summary>

Tiens le tableau proposé, une ligne par exemple. Pour chaque exemple : calcule $z = \mathbf{w}\cdot\mathbf{x} + b$ avec les poids **courants**, puis $y z$ ; si $y z \le 0$, corrige, sinon passe.

</details>
<details><summary>Indice 2</summary>

Pour chaque exemple : $z$ avec les poids courants, puis $y z$ ; une correction ajoute $\eta\,y\,\mathbf{x}$ aux poids et $\eta\,y$ au biais. Attention au cas $y z = 0$, dès le premier exemple. g) et h) La règle de **prédiction** et la règle d'**apprentissage** ne traitent pas $z = 0$ de la même façon : compare-les sur $(0, 0)$.

</details>
<details><summary>Indice 3</summary>

a) Exemple 1, $(0, 0)$ avec $y = -1$ : $z = 0$, donc $y z = 0 \le 0$, c'est une erreur. $\mathbf{w}$ reste $(0, 0)$ (on lui ajoute $-1 \times (0, 0)$) et $b$ devient $0 + 1 \times (-1) = -1$. b) à e) Même méthode, exemple par exemple, avec les poids **courants** : $z = \mathbf{w}\cdot\mathbf{x} + b$, puis $y z$ ; si $y z \le 0$, $\mathbf{w} \leftarrow \mathbf{w} + y\,\mathbf{x}$, $b \leftarrow b + y$, et tu comptes une correction. Surveille l'exemple 3 : sa somme peut valoir exactement 0. f) Refais un parcours complet, en partant des poids de la fin de l'epoch 1. g) Avec les poids de la fin de l'epoch 2, calcule les quatre $z$ et applique la règle de **prédiction** (une somme nulle donne $-1$, qui code la sortie 0 de OR). h) Au début de l'epoch 3, que vaut $y z$ pour $(0, 0)$ ? Compare avec la condition d'erreur de la règle d'**apprentissage**.

</details>

### Ex 10.7 — Le théorème de convergence du perceptron, guidé pas à pas ∂

<details><summary>Indice 1</summary>

Deux quantités évoluent à chaque correction : le produit scalaire $\mathbf{u}\cdot\mathbf{w}$ (il mesure l'alignement avec le bon séparateur) et la norme $\lVert \mathbf{w} \rVert$. Montre que la première croît vite et que la seconde croît lentement.

</details>
<details><summary>Indice 2</summary>

b) $\mathbf{u}\cdot\mathbf{w}_k = \mathbf{u}\cdot\mathbf{w}_{k-1} + y\,\mathbf{u}\cdot\mathbf{x}$, et $y\,\mathbf{u}\cdot\mathbf{x} \ge \gamma$. c) $\lVert \mathbf{w}_{k-1} + y\mathbf{x} \rVert^2 = \lVert \mathbf{w}_{k-1} \rVert^2 + 2y\,\mathbf{w}_{k-1}\cdot\mathbf{x} + \lVert \mathbf{x} \rVert^2$ ($y^2 = 1$), et le terme du milieu est $\le 0$ à une erreur. d) $\mathbf{u}\cdot\mathbf{w}_k \le \lVert \mathbf{u} \rVert\,\lVert \mathbf{w}_k \rVert$. e) Refais b) et c) en gardant $\eta$ dans chaque correction : que deviennent les deux inégalités ?

</details>
<details><summary>Indice 3</summary>

a) Comme $\lVert \mathbf{u} \rVert = 1$, $y_i\,\mathbf{u}\cdot\mathbf{x}_i$ est la distance de $\mathbf{x}_i$ à l'hyperplan $\mathbf{u}\cdot\mathbf{x} = 0$, comptée positivement du bon côté. L'hypothèse dit donc que tous les exemples sont du bon côté, à une distance au moins $\gamma$ de l'hyperplan : une bande vide de demi-largeur $\gamma$ l'entoure, et tous les points tiennent dans la boule de rayon $R$. b) Pars de l'égalité de l'indice 2, minore son dernier terme grâce à l'hypothèse sur $\mathbf{u}$, puis fais une récurrence à partir de $\mathbf{w}_0 = \mathbf{0}$. c) Même schéma pour la norme : borne le terme du milieu grâce à l'erreur, le dernier grâce à $R$, puis récurrence. d) Encadre $\mathbf{u}\cdot\mathbf{w}_k$ : par b) en dessous, par Cauchy-Schwarz puis c) au-dessus. Il reste une inégalité entre $k\gamma$ et $\sqrt{k}\,R$, à résoudre en $k$. e) Refais b) et c) avec $\eta$ : par quelle puissance de $\eta$ chaque borne est-elle multipliée ? Reporte dans d). f) Écris $\lVert \tilde{\mathbf{x}}_i \rVert^2$ en fonction de $\lVert \mathbf{x}_i \rVert^2$. Pour la marge, ramène le séparateur augmenté $(b, \mathbf{u})$ à la norme 1 : sa norme est-elle plus grande ou plus petite que 1 quand $b \ne 0$, et qu'est-ce que cela fait à la marge ? g) Relis la borne : quelles quantités y figurent, et lesquelles n'y figurent pas ? h) Chaque pixel vaut au plus 1 : majore $\lVert \mathbf{x} \rVert^2$ par une somme de 400 termes, prends la racine, puis applique la borne avec $\gamma = 0{,}5$. i) Quelle hypothèse du théorème tombe, et quelle étape de la preuve ne tient plus ?

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

a) Borne basse : $10^{14} / (8{,}6 \times 10^{10}) \approx 1\,200$ synapses par neurone ; calcule la borne haute de la même façon, avec $5 \times 10^{14}$. b) Même méthode : divise chacune des deux bornes du nombre de synapses par $10^{12}$. c) Calcule $10^{12} \times 2$ octets, convertis en Go (1 Go $= 10^9$ octets), divise par 80 Go, et arrondis au nombre entier de GPU supérieur. d) Multiplie ce nombre de GPU par 700 W, puis divise par 20 W ; ensuite, fais la liste de ce qui consomme dans un centre de données en plus des GPU. e) $20\ \text{W} \times 24\ \text{h}$ donne des wattheures : convertis en kWh. f) Calcule le rapport $3{,}2 \times 10^{10} / (1{,}6 \times 10^{10})$, puis demande-toi si un paramètre est l'équivalent d'un neurone ou plutôt d'une synapse, et quelle part de ses neurones le cerveau active à la fois. g) Pense à ce qu'est une synapse (dynamique, chimique) et à ce qu'est un paramètre (un nombre figé), au rôle de l'architecture, des données et de l'apprentissage, et à ce qu'on sait mesurer de « l'intelligence ».

</details>

### Ex 10.10 — « Cerveaux électroniques » : hype, hivers de l'IA et responsabilité ⚖️

<details><summary>Indice 1</summary>

Sépare deux colonnes : ce que la machine a fait devant les journalistes, et ce qu'on a annoncé qu'elle ferait un jour.

</details>
<details><summary>Indice 2</summary>

Pour 3., pense à la chaîne promesse, crédits, déception, retrait des crédits. Pour 4., des critères : une démonstration vérifiable, une mesure sur un test indépendant, des limites annoncées, des chiffres plutôt que des adjectifs.

</details>
<details><summary>Indice 3</summary>

1. Le démontré : une seule décision binaire (des cartes marquées à gauche ou à droite), apprise en une cinquantaine d'essais. L'annoncé : marcher, parler, voir, écrire, se reproduire, être conscient. Pour mesurer l'écart, compare cette décision binaire à chacune des capacités annoncées. 2. Pour chaque acteur (la Marine, Rosenblatt, le journal), demande-toi ce qu'il savait, ce qu'il avait intérêt à dire et ce qu'il pouvait corriger ; puis décide si la faute revient à un seul. 3. Mets dans l'ordre la chaîne de l'indice 2, et demande-toi qui paie la déception : seulement ceux qui avaient trop promis ? 4. Applique les critères de l'indice 2 à tes deux annonces, un par un. 5. Tes trois règles peuvent porter sur la tâche exacte, la mesure (sur quelles données) et les limites connues ; une bonne règle se vérifie, « être honnête » ne se vérifie pas.

</details>

### Ex 10.11 — Rosenblatt (1958) : le perceptron dans le texte 📄

<details><summary>Indice 1</summary>

Les trois questions sont au tout début de l'introduction ; les deux positions sur la mémoire suivent immédiatement.

</details>
<details><summary>Indice 2</summary>

Repère les mots *S-points* (ou *sensory units*), *A-units*, *R-units*, et la phrase qui dit quelles connexions sont aléatoires. Pour les systèmes α, β et γ, regarde ce qui gagne de la « valeur » et à quel moment.

</details>
<details><summary>Indice 3</summary>

1. Les trois questions portent sur la détection de l'information par le système biologique, sur la forme sous laquelle elle est stockée, et sur la façon dont elle influence la reconnaissance et le comportement ; l'article laisse la première à la physiologie des sens et s'attaque aux deux autres. Pour les questions suivantes, appuie chaque réponse sur une phrase précise du texte. 2. Le paragraphe qui suit les trois questions oppose deux conceptions de la trace d'un souvenir : pour chacune, demande-toi si l'on pourrait y retrouver une copie du stimulus, comme dans la mémoire d'un ordinateur, puis cherche la phrase où l'auteur dit laquelle il suit. 3. Dessine les étages, des points sensoriels aux réponses, et note sur chaque flèche ce que l'article en dit (fixée, tirée au hasard, renforcée) ; pour l'exclusion des réponses, regarde les connexions qui repartent des unités de réponse. 4. Pour chaque système, note quelles cellules gagnent de la valeur et à quel moment, puis compare avec la règle de la fiche, qui n'agit qu'après une erreur. 5. Cherche dans les conclusions la capacité que l'auteur juge hors de portée de son modèle, puis relis ce que Minsky et Papert reprochent aux unités qui ne voient qu'une partie de l'image (fiche §10.3.2). 6. Cherche le nom de l'ordinateur et du laboratoire là où l'auteur présente ses simulations. 7. Compare avec un article de machine learning récent : y trouves-tu un dataset de test, des chiffres de performance, une comparaison avec d'autres méthodes ?

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

`sign_step` : un seul `np.where(condition, 1.0, -1.0)` sur `z` converti en flottants, sans boucle. La condition est une inégalité **stricte** : $z = 0$ doit tomber du côté de $-1$ (c'est tout le piège de `np.sign`, et de `>=`).

```python
def add_bias_column(X):
    X = np.asarray(X, dtype=float)
    # 1. if X is not 2-D: raise a ValueError that says how many dimensions X has
    # 2. glue a column of ones in front of X, and return the result
```
La ligne clé de l'étape 2 : `np.hstack([np.ones((X.shape[0], 1)), X])` ; la colonne de 1 a la forme `(n, 1)`, pas `(n,)`.

</details>

### Ex 10.13 — AND, OR, XOR : le perceptron va-t-il converger ? 🔮

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la séparabilité et la fin de l'encadré sur la convergence : que se passe-t-il sur des données séparables, et sur des données qui ne le sont pas ?

</details>
<details><summary>Indice 2</summary>

Pour c), fais une epoch de XOR à la main, dans l'ordre (0, 0), (0, 1), (1, 0), (1, 1), à partir de poids nuls : où en sont les poids à la fin ? Pour b), déduis-en ce que prédit le perceptron pour chacune des quatre entrées à la fin d'une epoch.

</details>
<details><summary>Indice 3</summary>

a) Range chaque porte parmi les données séparables ou non (✏️ 10.3, ∂ 10.4), puis relis ce que le théorème de convergence garantit dans un cas, et ce que la fiche annonce dans l'autre. c) Premier exemple de XOR, $(0, 0)$ avec $y = -1$ : $z = 0$, c'est une erreur ; $\mathbf{w}$ ne bouge pas et $b$ passe à $-1$. Fais de même pour $(0, 1)$, $(1, 0)$ et $(1, 1)$, en vérifiant à chaque fois s'il y a erreur (une somme nulle compte), puis compare les poids de la fin de l'epoch à ceux du début : que fera l'epoch suivante, qui part de là et voit les mêmes exemples dans le même ordre ? b) Avec les poids de la fin d'une epoch, calcule les quatre sommes et applique la règle de prédiction de scikit-learn (la classe 1 si la somme est $> 0$, la classe 0 sinon), puis compte les bonnes réponses.

</details>

### Ex 10.14 — neuron_forward : un neurone appliqué à tout un batch 🔨

<details><summary>Indice 1</summary>

Trois étapes : convertir et vérifier les formes, calculer toutes les sommes pondérées d'un coup, appliquer l'activation au tableau entier.

</details>
<details><summary>Indice 2</summary>

`X @ w` marche pour `X` de forme `(n, p)` comme `(p,)`. La dernière dimension de `X` est `X.shape[-1]`. Pour un exemple seul, `X @ w + b` est un scalaire NumPy : `np.asarray(...)` en fait un tableau 0-d.

</details>
<details><summary>Indice 3</summary>

```python
def neuron_forward(X, w, b=0.0, activation=sign_step):
    X, w = np.asarray(X, dtype=float), np.asarray(w, dtype=float)
    # 1. if X.shape[-1] != len(w): raise a ValueError that gives both sizes
    # 2. all the weighted sums at once, then ONE call of the activation on the whole array
    # 3. wrap the result in np.asarray(..., dtype=float): a single sample then gives a 0-d array
```
La ligne clé des étapes 2 et 3 : `activation(X @ w + b)`, enveloppée dans `np.asarray(..., dtype=float)`. `X.shape[-1]` vaut le nombre de features aussi bien pour un batch de forme `(n, p)` que pour un exemple seul de forme `(p,)`.

</details>

### Ex 10.15 — Des noms de poids (AD, BE…) à la matrice W 🔨

<details><summary>Indice 1</summary>

Une ligne de $\mathbf{W}$ par source, une colonne par neurone : le nom du poids en ligne `j`, colonne `k`, est la lettre de la source `j` suivie de celle du neurone `k`.

</details>
<details><summary>Indice 2</summary>

Une liste de listes en compréhension : une ligne par source, et dans chaque ligne un poids par cible ; le nom d'un poids est la chaîne « source + cible ». Pour la couche : le batch multiplié par $\mathbf{W}$, plus les biais, que NumPy ajoute à chaque ligne par *broadcasting*. c) Quelle est la forme de `W`, et celle qu'attend `nn.Linear(3, 3)` ?

</details>
<details><summary>Indice 3</summary>

`names_to_matrix_15` : l'élément de la ligne `j`, colonne `k`, est `weights[sources[j] + targets[k]]`. En compréhension, `for s in sources` va **à l'extérieur**, pour que chaque source donne une ligne, et `for t in targets` à l'intérieur ; passe le tout à `np.array(..., dtype=float)`. `layer_forward_15` : convertis `X`, `W` et `b` en tableaux de flottants ; `X @ W` a la forme `(4, 3)`, `b` (forme `(3,)`) s'ajoute à chacune de ses lignes, et l'activation s'applique une seule fois, au tableau entier. c) Quand on copie un tableau dans le `weight` d'une couche, PyTorch ne contrôle que sa **forme** : compare celle de `W` à celle qu'attend `nn.Linear(3, 3)`, `(out_features, in_features)`.

</details>

### Ex 10.16 — XOR avec trois neurones câblés à la main 🔨

<details><summary>Indice 1</summary>

Reprends l'idée de ✏️ 10.5 : un neurone OR, un neurone NAND, puis un AND de leurs deux sorties. Calcule chaque neurone avec `neuron_forward`.

</details>
<details><summary>Indice 2</summary>

Les deux neurones cachés reçoivent `X`. Le neurone de sortie reçoit `np.column_stack([h1, h2])`, dont les valeurs sont $-1$ ou $+1$ : sa somme $h_1 + h_2 + b$ vaut $-2 + b$, $b$ ou $2 + b$ ; choisis $b$ pour que seul le dernier cas soit positif.

</details>
<details><summary>Indice 3</summary>

```python
def xor_network_16(X):
    h1 = mylearn.perceptron.neuron_forward(X, np.array([1.0, 1.0]), b=-0.5)   # OR: -1 only for (0, 0)
    # h2: a NAND neuron (the weights and the bias of AND, signs reversed: ✏️ 10.3 c)
    # output neuron: its inputs are np.column_stack([h1, h2]), its weights (1, 1),
    #                and a bias strictly between -2 and 0; return its output
```
b) Pour chacun des trois neurones, compte ses poids et son biais, puis additionne.

</details>

### Ex 10.17 — Le learning rate change-t-il un perceptron qui part de zéro ? 🔮

<details><summary>Indice 1</summary>

Pars de la règle d'apprentissage de la fiche (encadré 🧮) : dans une correction, qu'est-ce qui dépend de $\eta$, et qu'est-ce qui décide s'il y a correction ?

</details>
<details><summary>Indice 2</summary>

Partis de zéro, suppose que deux perceptrons, l'un avec $\eta$, l'autre avec $100\,\eta$, ont fait les mêmes corrections jusqu'ici : quelle relation lie leurs poids ? Et leurs sommes $z$ pour l'exemple suivant ? Avec un départ non nul, cette relation tient-elle encore ?

</details>
<details><summary>Indice 3</summary>

Partis de zéro, les poids après $k$ corrections valent $\mathbf{w}_k = \eta \sum_{j=1}^{k} y_j\,\mathbf{x}_j$, et le biais $b_k = \eta \sum_{j=1}^{k} y_j$, où $j$ parcourt les exemples corrigés. Suppose que deux perceptrons, de learning rates $\eta$ et $100\,\eta$, ont fait les mêmes corrections jusqu'ici : écris leurs poids avec cette formule, compare leurs sommes $z$ sur l'exemple suivant, puis le signe de $y z$. Décident-ils la même chose ? Conclus par récurrence pour a) et b). Pour c), refais le calcul avec un départ commun : $\mathbf{w}_k = \mathbf{w}_0 + \eta \sum_{j=1}^{k} y_j\,\mathbf{x}_j$. Le rapport des poids des deux perceptrons peut-il encore être le même pour toutes les composantes ?

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
or_params_18 = [*or_model_18.coef_[0].tolist(), float(or_model_18.intercept_[0])]   # [w1, w2, b]
```
Les autres valeurs suivent le même schéma : b) un modèle entraîné sur `GATES["xor"]`, noté avec `score` sur les mêmes données ; c) un attribut du modèle de a) ; d) et e) `fit` sur les manchots d'entraînement (`X_peng[train_17]`, puis `Z_train_17`), puis `score` sur ceux de test, avec les mesures de la même sorte.

</details>

### Ex 10.19 — Erreurs par epoch : séparable ou pas ? 📈

<details><summary>Indice 1</summary>

Lis chaque courbe jusqu'au bout : est-elle à 0 à la fin, et y reste-t-elle ?

</details>
<details><summary>Indice 2</summary>

Le graphique du milieu a une échelle logarithmique en abscisse : les epochs 3, 30 et 300 sont espacées régulièrement. Pour d), regarde la courbe A de près, epoch par epoch. Pour e), regarde le nuage de points de E, à droite. Pour f), relis la fin de l'encadré 🧮 sur la convergence : de quoi dépend le nombre de corrections ?

</details>
<details><summary>Indice 3</summary>

a) Lis chaque courbe jusqu'à sa dernière epoch : C est à 0 dès la première epoch et n'en bouge plus, donc C fait partie de la réponse. Fais de même pour A, B, D et E, en regardant la **fin** de chaque courbe, pas seulement son minimum. b) Repère la première epoch où D vaut 0. Sur l'axe logarithmique, l'epoch 5 est aux sept dixièmes de l'intervalle entre 1 et 10, et l'epoch 50 aux sept dixièmes de l'intervalle entre 10 et 100. c) Lis le palier de la courbe B, ou retrouve-le avec 🔮 10.13 : combien d'entrées le perceptron classe-t-il mal à la fin d'une epoch ? d) Suis la courbe A epoch par epoch, de 1 à 8 : une fois à 0, y reste-t-elle ? e) Sur le nuage de points de E, peux-tu tracer une droite qui laisse toutes les Adélie d'un côté et toutes les Chinstrap de l'autre ? Si non, que peut faire la courbe, même avec dix fois plus d'epochs ? f) Que devient la borne $(R/\gamma)^2$ quand la marge $\gamma$ est minuscule ? Une courbe encore au-dessus de 0 permet-elle alors de distinguer « pas séparable » de « séparable, mais lent » ?

</details>

### Ex 10.20 — Docstring NumPy et doctest pour neuron_forward 🛠️

<details><summary>Indice 1</summary>

Exécute d'abord tes deux exemples dans une cellule du notebook : ce qu'affiche Python est exactement ce que tu dois recopier sous chaque ligne `>>>`.

</details>
<details><summary>Indice 2</summary>

Dans la docstring, les exemples existants définissent déjà `X` et `w` : tu peux les réutiliser. Écris `>>> neuron_forward(np.array([1.0, 2.0]), w, b=-1.0, activation=lambda z: z)` puis la sortie sur la ligne suivante, avec la même indentation ; de même pour `add_bias_column(X)`. Une ligne vide sépare un exemple d'une phrase d'explication.

</details>
<details><summary>Indice 3</summary>

Le premier exemple, en modèle, avec l'indentation des exemples existants :

```text
    A single sample gives a 0-d array:

    >>> neuron_forward(np.array([1.0, 2.0]), w, b=-1.0, activation=lambda z: z)
    array(2.)
```
Pour le second, fais de même : une phrase d'explication, une ligne vide, puis la ligne `>>> neuron_forward(add_bias_column(X), np.array([-1.0, 1.0, 1.0]))`. Lance-la dans une cellule, et recopie **exactement** ce que Python affiche sur la ligne qui la suit, espaces compris. Si doctest affiche `np.float64(2.0)` pour le premier exemple, c'est ta fonction qu'il faut corriger (10.14), pas l'exemple.

</details>

### Ex 10.21 — La classe Perceptron et sa règle d'apprentissage 🔨

<details><summary>Indice 1</summary>

Organise `fit` en quatre temps : les contrôles ; le codage des labels en $\pm 1$ ; la double boucle (les epochs, puis les exemples d'une epoch) ; l'enregistrement des attributs. Refais d'abord ✏️ 10.6 si ce n'est pas fait.

</details>
<details><summary>Indice 2</summary>

`classes = np.unique(y)`, puis `target = np.where(y == classes[1], 1.0, -1.0)`. L'ordre d'une epoch : `rng.permutation(n)` si `shuffle`, sinon `range(n)`, avec `rng` créé **dans** `fit`. Compte les corrections dans un entier Python, ajoute-le à `errors_`, et sors de la boucle des epochs s'il vaut 0. Termine par `self.intercept_ = float(b)` et `return self`.

</details>
<details><summary>Indice 3</summary>

Le squelette de la double boucle, une fois `X`, `target` (les labels en $\pm 1$), `w` et `b` (nuls) prêts :

```python
rng = np.random.default_rng(self.random_state)       # created in fit: same seed, same model
for _ in range(self.max_iter):
    # the order of this epoch: rng.permutation(len(X)) if self.shuffle, else range(len(X)); mistakes = 0
    for i in order:
        if target[i] * (X[i] @ w + b) <= 0:           # <= : a sum of 0 is a mistake too
            w = w + self.eta0 * target[i] * X[i]
            # the bias the same way, only if self.fit_intercept; count the mistake
    # append mistakes to the list of errors; break after an epoch without mistakes
```
Range ensuite `classes_`, `coef_`, `intercept_` (un `float` Python), `errors_` et `n_iter_`, et termine par `return self`.

</details>

### Ex 10.22 — Perceptron piégé : quatre bugs classiques 🐛

<details><summary>Indice 1</summary>

Le collègue part de poids nuls : que vaut $y z$ pour le premier exemple ? Puis relis sa boucle ligne par ligne, avec le tableau des pièges ⚠️ de la fiche sous les yeux.

</details>
<details><summary>Indice 2</summary>

Dans l'ordre des diagnostics, cherche : le comparateur du test d'erreur ; la valeur de `yi` quand les labels sont 0 et 1 ; la ligne qui manque pour le biais ; ce que fait `rng.shuffle(X)` à `X`, et à `y`. Un `np.asarray` sur un tableau déjà en flottants renvoie-t-il une copie ?

</details>
<details><summary>Indice 3</summary>

Deux lignes clés, à placer dans la méthode du collègue :

```python
target = np.where(y == self.classes_[1], 1.0, -1.0)                 # bug 2: labels -1 and +1, from classes_
order = rng.permutation(len(X)) if self.shuffle else range(len(X))  # bug 4: shuffle indices, not X
```
La boucle des exemples devient `for i in order:`, avec `X[i]` et `target[i]`, ce qui garde exemples et labels alignés et ne touche plus au tableau de l'appelant. Pour les bugs 1 et 3, relis le test d'erreur et le bloc de correction : un exemple dont la somme vaut exactement 0 doit-il compter comme une erreur ? Quel nombre ajustable le bloc de correction oublie-t-il ?

</details>

### Ex 10.23 — Marge et vitesse de convergence : la borne (R/γ)² à l'épreuve 🔬

<details><summary>Indice 1</summary>

Deux petites fonctions : la marge est un minimum de produits scalaires, le nombre de corrections est la somme de `errors_` de ton `Perceptron`.

</details>
<details><summary>Indice 2</summary>

`margin_23` : trois étapes, ramener `u` à la norme 1 (`np.linalg.norm`), coder les labels 0 et 1 en $-1$ et $+1$, puis prendre le minimum des produits $y_i\,\mathbf{u}\cdot\mathbf{x}_i$, tous calculés d'un coup par `X @ u`. `updates_23` : crée ton `Perceptron` avec les quatre réglages de l'énoncé, entraîne-le, puis additionne sa liste `errors_`.

</details>
<details><summary>Indice 3</summary>

```python
def margin_23(X, y, u):
    u = np.asarray(u, dtype=float) / np.linalg.norm(u)      # u of norm 1 (the check passes a u of norm 2)
    signs = np.where(np.asarray(y) == 1, 1.0, -1.0)         # labels 0/1 -> -1/+1
    # return the smallest signs[i] * (X[i] @ u), as a Python float


def updates_23(X, y, seed):
    model = mylearn.perceptron.Perceptron(fit_intercept=False, max_iter=100_000, shuffle=True, random_state=seed)
    # fit it on (X, y), then return the total number of corrections: the sum of errors_, as an int
```
Le piège : compter les corrections (la somme de `errors_`), pas les epochs (`n_iter_`).

</details>

### Ex 10.24 — Trois espèces de manchots avec des perceptrons en un-contre-tous 🔬

<details><summary>Indice 1</summary>

Tes méta-classifieurs du ch. 7 reçoivent un classifieur binaire **non entraîné**, et le copient pour chaque sous-problème.

</details>
<details><summary>Indice 2</summary>

Crée le méta-classifieur autour d'un `Perceptron(max_iter=100)` neuf, puis entraîne-le avec `fit` sur les données reçues ; `fit` renvoie `self` : tu peux renvoyer directement son résultat. b) Un perceptron a convergé s'il finit par une epoch sans correction : regarde ce qu'affiche la vérification.

</details>
<details><summary>Indice 3</summary>

Le méta-classifieur de `ovr_24` s'écrit `mylearn.multiclass.OneVsRestClassifier(mylearn.perceptron.Perceptron(max_iter=100))` : il reçoit un perceptron **neuf** et le copie pour chaque espèce ; il reste à l'entraîner sur `(X_train, y_train)` et à le renvoyer. `ovo_24` suit le même modèle, avec `OneVsOneClassifier`. b) Dans l'affichage du un-contre-tous, lis pour chaque espèce les corrections de la **dernière** epoch (`errors_[-1]`, pas `errors_[0]`) : 0 veut dire que son perceptron a convergé. Garde les initiales des autres, par ordre alphabétique.

</details>

### Ex 10.25 — Défi « Mark I » : un perceptron sur des chiffres de 20 × 20 pixels 🏆

<details><summary>Indice 1</summary>

Commence par `crop_25` (une découpe de tableau et une division) et la part de l'encre (deux sommes). Puis regarde pourquoi le point de départ échoue : après 100 epochs, son perceptron corrige-t-il encore des images ? Que valent alors les poids de la **dernière** epoch ? Relis la fin de l'encadré 🧮 sur la convergence.

</details>
<details><summary>Indice 2</summary>

Pour `crop_25` : une tranche des lignes et des colonnes 4 à 23 (en Python, la borne de fin d'une tranche est exclue), un `reshape` en 400 colonnes, puis la division par 255. Pour la méthode, deux idées classiques : faire la **moyenne** des poids au fil de l'entraînement (le perceptron moyenné), ou garder les poids de la meilleure epoch (l'algorithme « pocket »). Mélanger les exemples à chaque epoch aide aussi. Compare tes variantes avec la validation croisée qu'affiche la vérification, sans révéler le test.

</details>
<details><summary>Indice 3</summary>

`crop_25` : la tranche `images[:, 4:24, 4:24]` donne le carré central de chaque image ; aplatis-la en une ligne de 400 valeurs par image (`reshape`, avec `len(images)` lignes), puis divise par 255. Pour la méthode, le perceptron moyenné :

```python
def train_mark1_25(X, y):
    # target: the labels coded -1/+1; w, b at zero; w_sum, b_sum, count at zero; rng with a fixed seed, e.g. 25
    for _ in range(10):                               # about ten epochs, each in a new order
        for i in rng.permutation(len(X)):
            # the classic rule: correct w and b when target[i] * (X[i] @ w + b) <= 0
            w_sum += w                                # after EVERY sample, corrected or not
            b_sum += b
            count += 1
    # return the averages w_sum / count and b_sum / count
```
Les poids qui ont duré longtemps sans erreur pèsent le plus dans la moyenne. Construit ainsi, le corrigé atteint les deux objectifs.

</details>
