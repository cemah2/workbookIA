# 16 · Réseaux feed-forward — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch16_feedforward/06_mes_reponses.md` (créée par `python tools/start_chapter.py 16`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 16.1 à 16.5, des vérifications numériques de ∂ 16.7 et de l'estimation 🧮 16.8 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la preuve ∂ 16.6, les démonstrations de ∂ 16.7, la réflexion (🗣️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), des lettres collées pour plusieurs choix ou pour une liste de nœuds (`"AE"`), une suite de codes, une lettre par élément et dans l'ordre (`"XYZ"`), une forme NumPy entre guillemets (`"(4, 7)"`, `"(7,)"`) ou `"erreur"` si l'opération lève une erreur, une liste pour plusieurs nombres (`[2, 5]`).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 16.Q1 — Qu'est-ce qui rend un réseau « feed-forward » ? 🧠 ⏱️ 3 min
*Fiche §16.1, §16.2 · livre §16.1, §16.2 · parcours R*

a) Vrai ou faux : dans un réseau feed-forward, une donnée qui a quitté un nœud n'y revient jamais, quel que soit le chemin suivi.
b) Un MLP « 784-128-10 » a combien de couches de neurones qui calculent (avec des poids et des biais) ? (A) 3 ; (B) 2 ; (C) 1 ; (D) 128.
c) Vrai ou faux : dans un MLP, les neurones d'une même couche s'échangent leurs sorties avant de les envoyer à la couche suivante.
d) Vrai ou faux : on dit que le réseau traite les données de façon hiérarchique, parce que chaque couche construit sur ce qu'a calculé la précédente.

### 16.Q2 — Vocabulaire des graphes : nœuds, arêtes, parents, descendants 🧠 ⏱️ 3 min
*Fiche §16.2 · livre §16.2 · parcours R*

Un graphe orienté a cinq nœuds A, B, C, D, E et les arêtes A → C, B → C, C → D, C → E et B → E.

a) Les parents de E, en lettres collées dans l'ordre alphabétique (`"XY"`).
b) Les descendants de A (les nœuds où mène un chemin parti de A), dans l'ordre alphabétique.
c) Le nombre de nœuds sans parent (les entrées du graphe).
d) Vrai ou faux : B est un ancêtre de D.
e) Vrai ou faux : sur un dessin de réseau sans pointes de flèches, la convention est que les données vont de gauche à droite, ou de bas en haut.

### 16.Q3 — Vrai ou faux : la passe arrière crée une boucle de rétroaction 🧠 ⏱️ 3 min
*Fiche §16.2 · livre §16.2 · parcours R*

a) Vrai ou faux : la passe arrière de l'entraînement (ch. 18), qui fait circuler une information des sorties vers les entrées, crée une boucle de rétroaction dans le graphe du réseau.
b) Vrai ou faux : le livre réserve le mot *feedback* (rétroaction) au cas où une boucle du graphe permet à un nœud de recevoir sa propre sortie.
c) Le sifflement qui monte quand un micro s'approche du haut-parleur qui le diffuse illustre : (A) une passe arrière ; (B) une rétroaction qui s'emballe ; (C) un flux synchrone ; (D) un horodatage.

### 16.Q4 — Pourquoi des graphes sans boucle, et où est l'exception des RNN ? 🧠 ⏱️ 3 min
*Fiche §16.2 · livre §16.2 · parcours R*

a) Vrai ou faux : dans un graphe orienté acyclique, deux chemins différents peuvent mener d'un même nœud à un autre.
b) Un réseau récurrent (RNN) reçoit à chaque pas de temps sa propre sortie du pas précédent. Pour le ramener à un DAG, on : (A) supprime la boucle ; (B) le déroule dans le temps, une copie du réseau par pas de temps ; (C) inverse le sens de toutes les arêtes ; (D) on ne peut pas.
c) Vrai ou faux : tout graphe orienté acyclique fini et non vide a au moins un nœud sans parent.
d) Le plus grand nombre d'arêtes d'un graphe orienté acyclique à 4 nœuds, sans arête en double ni arête d'un nœud vers lui-même.

### 16.Q5 — Réseau synchrone ou asynchrone : reconnaître les deux 🧠 ⏱️ 3 min
*Fiche §16.3 · livre §16.3*

a) Pour chacun des quatre systèmes, S s'il est synchrone, A s'il est asynchrone, dans l'ordre (`"SSSS"`) : 1) un orchestre où tous les musiciens jouent en suivant la battue du chef ; 2) des collègues qui font avancer un projet par un fil d'e-mails ; 3) un tableur qui recalcule une cellule dès qu'une des cellules dont elle dépend change ; 4) un jeu où tous les joueurs révèlent leur coup en même temps, au coup de gong.
b) Vrai ou faux : dans un système synchrone, on règle la période de l'horloge sur le nœud le plus rapide.
c) Vrai ou faux : un système synchrone est un cas particulier de système asynchrone.

### 16.Q6 — Horodatage : quand un nœud doit-il recalculer sa sortie ? 🧠 ⏱️ 3 min
*Fiche §16.3 · livre §16.3*

Un nœud à trois entrées ne calcule que lorsque ses trois entrées sont nouvelles. Il a calculé sa dernière sortie à $t = 10$.

a) Vrai ou faux : ses entrées portent les horodatages 12, 15 et 11 ; il doit recalculer.
b) Vrai ou faux : ses entrées portent les horodatages 12, 9 et 14 ; il doit recalculer.
c) Vrai ou faux : l'horodatage d'une valeur note le moment où le nœud qui la reçoit l'utilise.
d) Vrai ou faux : ses trois entrées portent des horodatages postérieurs à 10, mais leurs valeurs sont exactement celles du dernier calcul ; il doit quand même recalculer.

### 16.Q7 — Dans un programme, qui garantit l'ordre d'évaluation ? 🧠 ⏱️ 3 min
*Fiche §16.3.1 · livre §16.3.1*

a) Dans un programme NumPy qui calcule un MLP couche par couche, qu'est-ce qui garantit qu'une couche ne lit pas ses entrées avant qu'elles soient prêtes ? (A) un horodatage attaché à chaque valeur ; (B) l'ordre des instructions du programme ; (C) une horloge maître du processeur, réglée sur la couche la plus lente ; (D) rien : l'ordre est aléatoire.
b) Vrai ou faux : sur GPU, PyTorch lance les calculs de façon asynchrone, et pour chronométrer correctement une opération, il faut appeler `torch.cuda.synchronize()` avant de lire l'heure.
c) Vrai ou faux : dans `h = np.tanh(x @ W1)` suivi de `y = h @ W2`, Python calcule `h` avant `y`.

### 16.Q8 — Un neurone, trois enfants : combien de poids ? 🧠 ⏱️ 3 min
*Fiche §16.4 · livre §16.4 · parcours R*

a) Un neurone R envoie sa sortie à trois neurones de la couche suivante. Le nombre de poids portés par ces trois arêtes.
b) Vrai ou faux : ces trois poids sont forcément égaux, puisque les trois arêtes transportent la même valeur.
c) Une couche de 4 neurones est entièrement connectée à une couche de 3 neurones. Le nombre de poids entre les deux (sans les biais).
d) Vrai ou faux : sur un schéma de réseau où aucun poids n'est écrit, chaque arête porte quand même son propre poids.

### 16.Q9 — Pourquoi pas le même poids partout au départ ? 🧠 ⏱️ 3 min
*Fiche §16.4.1 · livre §16.4.1 · parcours R*

a) Vrai ou faux : si tous les neurones d'une couche cachée ont les mêmes poids entrants et le même biais, ils calculent la même sortie pour chaque exemple.
b) Pourquoi des poids tous égaux posent-ils problème ? (A) les calculs sont plus lents ; (B) les neurones reçoivent les mêmes corrections, restent identiques pendant tout l'entraînement, et la couche se comporte comme un seul neurone ; (C) les poids deviennent négatifs ; (D) la loss ne peut plus être calculée.
c) Vrai ou faux : des poids tirés au hasard, avec des biais tous nuls, suffisent à briser la symétrie entre les neurones.
d) L'écart-type de la loi uniforme sur $[-0{,}05 ;\ 0{,}05]$ (3 décimales).

### 16.Q10 — LeCun, Glorot, He : quel fan-in, pour quelle activation ? 🧠 ⏱️ 3 min
*Fiche §16.4.1 · livre §16.4.1 · parcours R, M*

Une couche dense a 400 entrées et 100 neurones.

a) L'écart-type de ses poids avec l'initialisation de He normale (3 décimales).
b) La borne $a$ de l'initialisation de Glorot uniforme, $\mathcal{U}(-a, a)$ (3 décimales).
c) La règle conçue pour ReLU : (A) LeCun ; (B) Glorot ; (C) He ; (D) une constante 0,05.
d) La règle qui tient compte à la fois du fan-in et du fan-out, recommandée pour tanh et la sigmoïde : (A) LeCun ; (B) Glorot ; (C) He ; (D) une constante 0,05.
e) Vrai ou faux : la variance de l'initialisation de He ne dépend que du nombre d'entrées de la couche.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 16.R1 — Ch. 15 : que contient un estimateur après fit() ? 🔁 ★ ⏱️ 5 min
*Ch. 15 (fiche §15.4, les estimateurs) et 📦 1.23 (`coefs_`) · parcours R*

a) Vrai ou faux : un `MLPClassifier` qu'on vient de créer, sans appeler `fit`, a déjà un attribut `coefs_`.
b) Dans scikit-learn, un attribut dont le nom finit par `_` est : (A) un attribut appris pendant `fit` ; (B) un hyperparamètre passé au constructeur ; (C) un attribut privé, à ne pas lire ; (D) une méthode.
c) `MLPClassifier(hidden_layer_sizes=(50,))` est entraîné sur 20 features et 3 classes. La longueur de la liste `coefs_`.
d) La forme de `coefs_[0]`, entre guillemets.
e) La forme de `coefs_[1]`, entre guillemets.

### 16.R2 — Ch. 13 : combien de régions pour un arbre de profondeur 2 ? 🔁 ★ ⏱️ 5 min
*Ch. 13 (fiche, les arbres de décision) · parcours R*

a) Le nombre de feuilles d'un arbre de décision binaire complet de profondeur 2.
b) Le plus grand nombre de feuilles (de régions) d'un arbre binaire de profondeur 5.
c) Vrai ou faux : quand chaque question compare une seule feature à un seuil, chaque feuille correspond à une région en forme de boîte, aux côtés parallèles aux axes.
d) Le nombre de nœuds internes (de questions) d'un arbre binaire complet de profondeur 2.

Dans ta copie : un arbre de décision est-il un graphe orienté acyclique ? Combien de parents a chacun de ses nœuds, et qu'est-ce qui le distingue du graphe d'un MLP ?

### 16.R3 — Ch. 0B : formes compatibles dans un produit matriciel 🔁 ★ ⏱️ 5 min
*Ch. 0B (le produit matriciel) · parcours R, M*

On a trois arrays NumPy : `A` de forme (3, 4), `B` de forme (4, 2) et `v` de forme (4,).

a) La forme de `A @ B`, entre guillemets (ou `"erreur"`).
b) La forme de `B @ A`.
c) La forme de `A @ v`.
d) Le nombre de multiplications de nombres pour calculer le produit d'une matrice (3, 4) par une matrice (4, 10).
e) Vrai ou faux : $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 16.1 — Compter les poids et les biais d'un MLP ✏️ ★ ⏱️ 10 min
**Objectif :** compter les paramètres d'un réseau dense, couche par couche, et voir lesquelles pèsent le plus.
**Prérequis :** ch. 10 (neurone, biais) · 0B (matrices) · fiche, section sur la passe avant · **Parcours :** R, M

Un MLP 10-8-6-3 reçoit 10 features, a deux couches cachées de 8 et 6 neurones, et 3 neurones de sortie ; toutes ses couches sont denses, avec un biais par neurone.

a) Le nombre de poids de la première couche (entre l'entrée et la première couche cachée).
b) Le nombre total de poids du réseau.
c) Le nombre total de biais.
d) Le nombre total de paramètres.
e) Le nombre de paramètres d'un MLP 784-512-512-10.
f) Le nombre de paramètres d'un MLP 10-16-12-3.
g) Vrai ou faux : dans un MLP 100-200-200-10, doubler la largeur des deux couches cachées (100-400-400-10) multiplie par 4 le nombre de poids entre les deux couches cachées.

Dans ta copie : dans le réseau de e), quelle couche porte le plus de paramètres, et quelle part du total ? Que dirais-tu à quelqu'un qui veut ajouter une troisième couche cachée de 512 neurones ?

### Ex 16.2 — Passe avant à la main dans un réseau 2-2-1 ✏️ ★ ⏱️ 15 min
**Objectif :** calculer à la main la sortie d'un petit réseau, lire un poids dans une matrice et voir ce que fait une activation.
**Prérequis :** 0B (produit matriciel) · ch. 10 (neurone, biais) · fiche §16.4 et section sur la passe avant · **Parcours :** R, M

D'abord un seul neurone, comme à la figure 16.3 du livre : trois valeurs $(R, G, B) = (0{,}2 ;\ 0{,}6 ;\ 0{,}4)$ arrivent par des arêtes de poids $(1 ;\ -0{,}5 ;\ 2)$, et le neurone a un biais de $0{,}1$.

a) La somme pondérée de ce neurone.

Ensuite un réseau 2-2-1 : deux entrées $(x_1, x_2)$, deux neurones cachés $h_1$ et $h_2$, un neurone de sortie, avec
$$\mathbf{W}^{(1)} = \begin{pmatrix} 0{,}5 & -1 \\ 1 & 0{,}5 \end{pmatrix}, \quad \mathbf{b}^{(1)} = (0 ;\ 1), \quad \mathbf{W}^{(2)} = \begin{pmatrix} 1{,}5 \\ -2 \end{pmatrix}, \quad b^{(2)} = -0{,}5,$$
dans la convention de la fiche : $W^{(1)}_{jk}$ est le poids de l'entrée $x_j$ vers le neurone $h_k$, et $\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}$. L'exemple est $\mathbf{x} = (2 ;\ 1)$, la sortie n'a pas d'activation.

b) Les deux sommes pondérées $(z_1, z_2)$ de la couche cachée (une liste).
c) Avec ReLU dans la couche cachée, la sortie du réseau.
d) Avec tanh dans la couche cachée, la sortie du réseau (3 décimales).
e) Le poids de l'arête qui va de l'entrée $x_2$ au neurone $h_1$.
f) Vrai ou faux : avec ReLU, modifier le poids de l'arête $h_2 \to$ sortie (le $-2$) change la sortie du réseau pour cet exemple.
g) Avec ReLU, les sorties du réseau pour le mini-batch des deux exemples $(2 ;\ 1)$ et $(0 ;\ 1)$, dans l'ordre (une liste).

Dans ta copie : pourquoi la réponse à f) dépend-elle de l'exemple ? Écris le calcul de g) sous forme matricielle, en donnant la forme de chaque tableau.

### Ex 16.3 — Formes et broadcasting d'un mini-batch à travers les couches ✏️ ★★ ⏱️ 15 min
**Objectif :** suivre la forme de chaque tableau d'une passe avant matricielle, et repérer les opérations qui lèvent une erreur.
**Prérequis :** 0A (broadcasting) · 0B (formes compatibles) · fiche, section sur la passe avant · **Fil rouge :** MNIST · **Parcours :** R, M

Un MLP 784-256-64-10 classe des images MNIST aplaties (784 pixels). On lui donne un mini-batch `X` de 32 images. Ses paramètres suivent la convention de la fiche : `W1` relie l'entrée à la première couche cachée, `b1` est le biais de cette couche, et ainsi de suite.

Réponds par une forme entre guillemets, ou `"erreur"` si l'opération lève une erreur.

a) La forme de `X`.
b) La forme de `W1`.
c) La forme de `X @ W1 + b1`.
d) La forme de `b2`.
e) La forme de la sortie du réseau (les logits).
f) Le nombre total de valeurs du cache de la passe avant : pour chaque couche, l'entrée `a_prev` et la somme pondérée `z` (un nombre).
g) Par erreur, `b1` a la forme du batch, `(32,)`. La forme de `X @ W1 + b1`.
h) Un collègue range `W1` comme PyTorch, avec la forme `(256, 784)`. La forme de `X @ W1`.
i) Avec ce `W1` de forme `(256, 784)`, la forme de `X @ W1.T`.

Dans ta copie : pour la question g), que se passerait-il avec un batch de 256 images, et un `b1` toujours de la forme du batch, `(256,)` ? Pourquoi est-ce plus grave qu'une erreur ?

### Ex 16.4 — Ordre d'évaluation d'un graphe et détection de boucle ✏️ ★★ ⏱️ 15 min
**Objectif :** dérouler l'algorithme de Kahn à la main, compter les ordres topologiques possibles et reconnaître une arête qui crée un cycle.
**Prérequis :** fiche §16.2 (encadré sur l'ordre topologique) · **Parcours :** M

Un petit réseau a deux entrées U et V, trois neurones cachés A, B, C et une sortie Y, reliés par les arêtes U → A, V → A, V → B, A → C, B → C, A → Y et C → Y (A envoie sa sortie à la fois à C et à Y).

a) L'ordre topologique que donne l'algorithme de Kahn quand il range toujours, parmi les nœuds prêts, le plus petit dans l'ordre alphabétique (lettres collées, `"XYZ…"`).
b) Le nombre d'arêtes du graphe.
c) Le nombre d'ordres topologiques différents de ce graphe.
d) Vrai ou faux : ajouter l'arête Y → B crée un cycle.
e) Vrai ou faux : ajouter l'arête U → Y crée un cycle.
f) La longueur (en arêtes) du plus long chemin d'une entrée à la sortie.

Dans ta copie : écris le déroulé de Kahn pour a), étape par étape (les nœuds prêts, le nœud rangé, les comptes de parents qui changent). Range maintenant les nœuds « couche par couche » : d'abord toutes les entrées, puis tous les nœuds dont **tous** les parents sont déjà rangés, et ainsi de suite. Quel ordre obtiens-tu, et est-ce celui de a) ?

### Ex 16.5 — Simuler un réseau asynchrone avec horodatage ✏️ ★★ ⏱️ 20 min
**Objectif :** appliquer la règle des horodatages du livre à un nœud qui additionne trois entrées, et mesurer ce que coûterait une règle naïve.
**Prérequis :** fiche §16.3 · **Parcours :** M

Un nœud S additionne trois entrées P, Q et R. Il a fait son dernier calcul à $t = 1$, avec $P = 3$, $Q = 5$ et $R = 2$ (sortie 10). Il ne recalcule que lorsque ses trois entrées portent un horodatage postérieur à son dernier calcul ; il calcule alors aussitôt, au moment où arrive la dernière, et note ce moment. Les messages suivants arrivent, chacun horodaté par son expéditeur :

| $t$ | 2 | 3 | 5 | 6 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| message | $P = 1$ | $Q = 4$ | $R = 2$ | $P = 7$ | $Q = 0$ | $R = 3$ | $P = 2$ |

a) La sortie que calcule S à $t = 5$.
b) La sortie que calcule S à $t = 9$.
c) Le nombre de calculs que fait S entre $t = 2$ et $t = 10$ inclus.
d) Une règle naïve fait recalculer S à chaque fois qu'une de ses entrées **change de valeur**. Le nombre de calculs qu'elle ferait entre $t = 2$ et $t = 10$.
e) Avec la règle naïve, la dernière sortie calculée.

Dans ta copie : la sortie de la règle naïve à $t = 9$ est-elle l'une des sorties de la règle du livre ? Et celle de e) ? Pourquoi les deux règles ne donnent-elles pas les mêmes valeurs intermédiaires ? Quel message de la table montre que comparer les valeurs ne suffit pas ?

### Ex 16.6 — Symétrie : pourquoi une initialisation constante ne se brise jamais ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer que deux neurones qui partent identiques reçoivent les mêmes corrections, et en déduire ce que vaut une couche initialisée par une constante.
**Prérequis :** 0B (dérivée d'une composée) · ch. 5 (dérivées partielles, descente de gradient) · fiche §16.4.1 · **Parcours :** M

Un réseau a $p$ entrées, une couche cachée de $m$ neurones d'activation $f$ (dérivable) et un neurone de sortie sans activation. Le neurone caché $k$ a un vecteur de poids entrants $\mathbf{w}_k$ (la colonne $k$ de $\mathbf{W}^{(1)}$), un biais $b_k$ et un poids sortant $v_k$ :
$$h_k = f(\mathbf{x}\cdot\mathbf{w}_k + b_k), \qquad \hat y = \sum_{k=1}^{m} v_k h_k + c.$$
La loss $\mathcal{L}$ de l'entraînement est une fonction dérivable des prédictions $\hat y$ des exemples (la MSE, par exemple). On suppose qu'au départ, les neurones 1 et 2 sont **jumeaux** : $\mathbf{w}_1 = \mathbf{w}_2$, $b_1 = b_2$ et $v_1 = v_2$. Réponds dans ta copie.

1. Montre que $h_1 = h_2$ pour **tout** exemple $\mathbf{x}$.
2. On échange les paramètres des neurones 1 et 2, $(\mathbf{w}_1, b_1, v_1) \leftrightarrow (\mathbf{w}_2, b_2, v_2)$, sans toucher aux autres. Montre que les prédictions, donc la loss, ne changent pas, quels que soient les paramètres.
3. Déduis-en, à partir de la définition d'une dérivée partielle comme limite d'un taux d'accroissement (ch. 5), qu'en un point où les neurones 1 et 2 sont jumeaux, $\partial \mathcal{L} / \partial v_1 = \partial \mathcal{L} / \partial v_2$, $\partial \mathcal{L} / \partial b_1 = \partial \mathcal{L} / \partial b_2$, et $\partial \mathcal{L} / \partial w_{j1} = \partial \mathcal{L} / \partial w_{j2}$ pour chaque entrée $j$.
4. Montre qu'après un pas de descente de gradient de learning rate $\eta$, les neurones 1 et 2 sont encore jumeaux, puis, par récurrence, après n'importe quel nombre de pas.
5. Les poids entrants, les biais et les poids sortants $v_k$ de tous les neurones cachés valent la même constante $\gamma \ne 0$. Montre que le réseau calcule, pendant tout l'entraînement, la même fonction qu'un réseau à **un seul** neurone caché, et donne le poids sortant de ce neurone en fonction de $m$ et de $v_1$.
6. Tous les paramètres valent 0 et $f = \tanh$. En utilisant la dérivée d'une composée, calcule $\partial \hat y / \partial v_k$ et $\partial \hat y / \partial w_{jk}$ au départ. Quels paramètres change le premier pas de descente de gradient ? Et les suivants ?
7. Cite deux façons de briser la symétrie.

### Ex 16.7 — Conserver la variance : retrouver les règles de LeCun, Glorot et He ∂ ★★★ ⏱️ 40 min
**Objectif :** démontrer la formule de la variance d'une somme pondérée, en déduire les trois règles d'initialisation, et chiffrer l'effet de la profondeur.
**Prérequis :** ch. 2 (variance, espérance, indépendance) · 0B (puissances, racines) · fiche §16.4.1 (encadré sur la variance d'une somme de produits) · **Parcours :** M

Un neurone reçoit $n$ entrées $x_1, \dots, x_n$ et calcule $z = \sum_{j=1}^{n} w_j x_j$ (biais nul). Les poids sont indépendants entre eux et du vecteur des entrées, de moyenne 0 et de même variance $\mathrm{Var}(w)$ ; les entrées ont toutes le même moment d'ordre 2, $\mathbb{E}[x^2]$ (elles peuvent être corrélées entre elles). Réponds dans ta copie.

1. Montre que $\mathbb{E}[z] = 0$ et que $\mathrm{Var}(z) = n\,\mathrm{Var}(w)\,\mathbb{E}[x^2]$ (montre d'abord que deux produits $w_j x_j$ et $w_k x_k$ différents sont décorrélés). Quelles hypothèses utilises-tu, et où ?
2. Les entrées sont centrées et de variance 1. Quelle variance des poids garde $\mathrm{Var}(z) = 1$ ? C'est la règle de LeCun.
3. Si $z$ suit une loi symétrique autour de 0 (par exemple normale), montre que $\mathbb{E}[\mathrm{ReLU}(z)^2] = \frac12 \mathrm{Var}(z)$. Déduis-en la règle de He pour une couche dont les entrées sont les sorties ReLU de la couche précédente.
4. Pendant la passe arrière (ch. 18), un gradient traverse la couche dans l'autre sens en additionnant $n_{out}$ produits ; admets que sa variance est alors multipliée par $n_{out}\,\mathrm{Var}(w)$. Pourquoi ne peut-on pas conserver à la fois la variance des activations et celle des gradients quand $n_{in} \ne n_{out}$ ? Quel compromis donne la règle de Glorot ?
5. Montre que $\mathcal{U}(-a, a)$ a la variance $a^2/3$, et déduis-en la borne de la version uniforme de chaque règle.

Vérifications numériques :

a) 256 entrées centrées de variance 1, des poids de variance $1/256$ : la variance de $z$.
b) La variance des poids de LeCun pour une couche de 16 entrées (4 décimales).
c) $z$ suit la loi normale $\mathcal{N}(0, 4)$ (variance 4) : la valeur de $\mathbb{E}[\mathrm{ReLU}(z)^2]$.
d) La variance des poids de He pour une couche de 32 entrées (4 décimales).
e) La borne $a$ de Glorot uniforme pour une couche de 200 entrées et 200 neurones (3 décimales).
f) Un réseau **linéaire** de 10 couches dont chaque couche multiplie la variance par $n_{in}\,\mathrm{Var}(w) = 1{,}5$. Des entrées centrées de variance 1 : la variance des sorties de la dernière couche (1 décimale).
g) Un réseau de 10 couches **ReLU**, initialisé en LeCun ($\mathrm{Var}(w) = 1/n_{in}$), dont la première somme pondérée a une variance 1. La variance des sommes pondérées de la dixième couche, avec le résultat de 3 (6 décimales).

### Ex 16.8 — Fermi : taille, mémoire et coût d'un MLP pour MNIST 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer de tête la mémoire et le coût de calcul d'un réseau dense, avant d'écrire une ligne de code.
**Prérequis :** Ex 16.1 · fiche, section sur la passe avant (mémoire et calcul) · **Fil rouge :** MNIST · **Parcours :** R, M

Le MLP 784-512-512-10 de 16.1 e) classe les 60 000 images d'entraînement de MNIST. On compte les nombres en `float32` (4 octets), un mégaoctet (Mo) vaut $10^6$ octets, et une passe avant coûte une multiplication par poids et par exemple (on néglige les biais et les activations).

a) La mémoire de ses paramètres, en Mo (1 décimale).
b) Le nombre de multiplications d'une passe avant pour **une** image, en millions (2 décimales).
c) Le nombre de multiplications d'une passe avant sur les 60 000 images, en milliards ($10^9$, arrondi à l'entier).
d) Un processeur fait $10^{10}$ multiplications par seconde. La durée de cette passe avant, en secondes (arrondie à l'entier).
e) Une epoch d'entraînement coûte environ trois passes avant. Sa durée, en secondes (arrondie à l'entier).
f) La mémoire du cache de la passe avant (pour chaque couche, l'entrée `a_prev` et la somme pondérée `z`) pour un batch de 128 images, en Mo (1 décimale).

Dans ta copie : ces estimations sont-elles optimistes ou pessimistes par rapport à un vrai entraînement en NumPy, et pourquoi ? Que deviendrait c) pour un réseau convolutif qui réutilise chaque poids sur les 784 positions de l'image (ch. 21) ?

<a id="reflexion"></a>

## 🗣️ 📄 Réflexion

### Ex 16.9 — Expliquer un réseau feed-forward avec une chaîne d'usine 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce qu'est un réseau feed-forward, avec une image juste et ses limites.
**Prérequis :** fiche §16.1 à §16.4 · **Parcours :** R

Un ami qui ne connaît pas le machine learning te demande ce qu'est « un réseau de neurones à propagation avant ». Explique-le-lui en **cinq phrases** au plus, avec l'image de la chaîne d'usine du livre :
- ce qui entre et ce qui sort, et ce que sont les postes de travail ;
- pourquoi les pièces ne reviennent jamais en arrière ;
- ce que sont les poids, et ce que veut dire « apprendre » ;
- une limite de l'image (ce qu'une usine fait, et qu'un réseau ne fait pas, ou l'inverse).

Enregistre-toi ou écris ta réponse, puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 16.10 — Glorot & Bengio (2010) : pourquoi les réseaux profonds apprenaient mal 📄 ★★★ ⏱️ 45 min
**Objectif :** lire un article fondateur et en tirer ce qu'il montre, comment, et ce qui a changé depuis.
**Prérequis :** Ex 16.7 · fiche §16.4.1 et section sur le diagnostic · **Parcours :** M

Lis l'article de X. Glorot et Y. Bengio, [« Understanding the difficulty of training deep feedforward neural networks »](https://proceedings.mlr.press/v9/glorot10a.html) (AISTATS 2010) : le résumé, l'introduction, la description des expériences, les sections sur la saturation des activations et sur l'étude des gradients, et la conclusion. Réponds dans ta copie.

1. Quelle question pose l'article ? Sur quels datasets et avec quels réseaux (profondeur, largeur, activations) les auteurs travaillent-ils ?
2. Avec la sigmoïde et l'initialisation classique de l'époque, que deviennent les activations de la **dernière** couche cachée au début de l'entraînement ? Quelle explication les auteurs proposent-ils ?
3. Quelle initialisation proposent-ils (la « normalisée ») ? Refais le raisonnement qui y mène avec tes résultats de ∂ 16.7, et compare-la à l'initialisation qu'ils critiquent.
4. Quelles mesures (histogrammes, écarts-types, valeurs singulières des jacobiennes) leur servent de diagnostic, couche par couche ? Lesquelles as-tu refaites dans le notebook (📈 16.22, 🔬 16.23) ?
5. Quelles limites voit-on aujourd'hui à cette étude (activations, datasets, techniques apparues depuis) ? L'article de He et coll. (2015) et l'encadré 🕰️ de la fiche sur les initialisations par défaut t'aident à répondre.

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 16.E1 — Pourquoi ne pas initialiser tous les poids à zéro ? 💼 ★★ ⏱️ 10 min
*Fiche §16.4.1 · parcours R*

« Pourquoi n'initialise-t-on pas tous les poids d'un réseau à zéro ? Et les biais ? »

### 16.E2 — He ou Glorot : quelle initialisation pour quelle activation ? 💼 ★★ ⏱️ 10 min
*Fiche §16.4.1, section sur le diagnostic · parcours R*

« Quelle initialisation choisissez-vous pour un réseau ReLU, et pour un réseau tanh ? D'où viennent ces règles, et que fait PyTorch par défaut ? »

### 16.E3 — Calculer de tête le nombre de paramètres d'un MLP 💼 ★★ ⏱️ 10 min
*Fiche, section sur la passe avant · parcours R*

« Combien de paramètres a un MLP 1000-500-100-10 ? Combien de mémoire en float32 ? Lequel de ces nombres change si l'on double la taille du batch ? »

### 16.E4 — Feed-forward, récurrent, graphe acyclique : expliquer la différence 💼 ★★ ⏱️ 10 min
*Fiche §16.2 · parcours R*

« Qu'est-ce qu'un réseau feed-forward, par opposition à un réseau récurrent ? Est-ce que la rétropropagation en fait un réseau avec des boucles ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch16_feedforward/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie (`mylearn/nn/layers.py`). La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels et aux exercices ✏️, ∂ et 🧮 ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 16.11 | Prédire les formes des sorties avant d'exécuter | 🔮 | ★ | 10 |
| 16.12 | Compter les paramètres par programme : count_parameters | 🔨 | ★ | 10 |
| 16.13 | La couche dense : dense_forward | 🔨 | ★ | 15 |
| 16.14 | Dix façons d'initialiser : init_weights | 🔨 | ★★ | 25 |
| 16.15 | Reproduire les figures 16.5 et 16.6, puis comparer Glorot et He | 🎨 | ★★ | 20 |
| 16.16 | Ordre topologique d'un graphe : topological_order | 🔨 | ★★ | 25 |
| 16.17 | Empiler les couches : init_mlp et mlp_forward | 🔨 | ★★ | 30 |
| 16.18 | Relire un MLPClassifier de scikit-learn avec ton mlp_forward | 📦 | ★★ | 25 |
| 16.19 | Initialisation constante : les neurones restent-ils jumeaux ? | 🔮 | ★★ | 20 |
| 16.20 | Chasse au bug : transposée, biais mal diffusé, variance prise pour un écart-type | 🐛 | ★★ | 25 |
| 16.21 | Rendre l'aléatoire reproductible et le prouver avec pytest | 🛠️ | ★★ | 20 |
| 16.22 | Lire des histogrammes d'activations couche par couche | 📈 | ★★ | 20 |
| 16.23 | Activations à travers 20 couches selon l'initialisation | 🔬 | ★★★ | 45 |
| 16.24 | Défi : garder vivant un réseau ReLU de 50 couches | 🏆 | ★★★ | 60 |
