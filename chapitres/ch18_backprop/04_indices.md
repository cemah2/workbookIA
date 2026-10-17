# 18 · Rétropropagation — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 18.Q1 — Pourquoi apprendre la rétropropagation si PyTorch la fait pour toi ?

<details><summary>Indice 1</summary>

Relis le §18.1 de la fiche : ce que font les bibliothèques, et les raisons du livre.

</details>
<details><summary>Indice 2</summary>

Le livre donne quatre raisons ; pour b), cherche celle qui ne s'y trouve pas. Pour c), compare un réseau aux classifieurs du ch. 13 : qui apporte la méthode d'apprentissage ?

</details>
<details><summary>Indice 3</summary>

a) Faux : dans un projet, ce sont PyTorch ou JAX qui calculent les gradients. b) Élimine les options que le §18.1 de la fiche cite parmi les raisons du livre : il en reste une. c) Un réseau est-il autre chose qu'un assemblage de petits calculs ?

</details>

### 18.Q2 — Perturber un poids au hasard : pourquoi est-ce si lent ?

<details><summary>Indice 1</summary>

Relis le §18.2.1 de la fiche et son pseudo-code.

</details>
<details><summary>Indice 2</summary>

Compte ce que coûte un essai (une passe avant) et combien d'essais il faut pour toucher chaque poids une fois. Pour b), il y a deux sens possibles. Pour d), pense à l'astuce du ch. 10.

</details>
<details><summary>Indice 3</summary>

a) Un essai par poids, une passe avant par essai : $10^6$ poids donnent $10^6$ passes. b) Combien de sens possibles, et combien font baisser l'erreur ? c) Un poids sert-il à un seul exemple ? d) Avec l'astuce du biais, à quoi ressemble le biais vu par l'algorithme ?

</details>

### 18.Q3 — Ce que le gradient dit à chaque poids, et le cas du biais

<details><summary>Indice 1</summary>

Relis le §18.2.2 de la fiche : le signe de la dérivée dit dans quel sens l'erreur bouge.

</details>
<details><summary>Indice 2</summary>

Une dérivée négative : l'erreur baisse quand le poids augmente ; une dérivée positive : l'inverse. Pour d), relis le §18.3.

</details>
<details><summary>Indice 3</summary>

a) La dérivée vaut $-2$ : augmenter le poids fait baisser l'erreur, donc `+1`. b) Même raisonnement avec une dérivée positive. c) Que dit le livre du pari de déplacer tous les poids à la fois, à petits pas ? d) Le livre remet-il les activations plus loin ?

</details>

### 18.Q4 — Le delta d'un neurone : signe et effet sur l'erreur

<details><summary>Indice 1</summary>

Relis le §18.4.1 de la fiche : $\Delta L \approx \delta\,\Delta a$.

</details>
<details><summary>Indice 2</summary>

Le signe d'un produit dit si l'erreur monte ou baisse. Pour d), calcule d'abord le changement de la sortie. Pour c), relis l'exemple des sorties 3 et 3 000 (p. 731).

</details>
<details><summary>Indice 3</summary>

a) Deux nombres de même signe ont un produit positif : l'erreur **monte**, réponse A. b) Quel signe doit avoir $\Delta a$ pour que $\delta\,\Delta a$ soit négatif ? c) De 7 à 7,5 ou de 700 à 700,5 : l'effet change-t-il ? d) $\Delta a = 1\,000{,}4 - 1\,000$, puis $\delta \times \Delta a$. e) Relis le début du §18.4 : combien d'étapes ?

</details>

### 18.Q5 — Le réseau minuscule : couches, poids et leurs noms

<details><summary>Indice 1</summary>

Relis le §18.5 de la fiche : le réseau $X, Y \to A, B \to C, D$ et le nom de ses poids.

</details>
<details><summary>Indice 2</summary>

Compte les fils entre les couches (chaque neurone d'une couche est relié à chaque neurone de la suivante). Le nom d'un poids : départ puis arrivée. Une couche se compte si elle calcule. Pour e), les indices commencent à 0.

</details>
<details><summary>Indice 3</summary>

a) Deux entrées vers deux neurones cachés : $2 \times 2 = 4$ fils ; deux cachés vers deux sorties : $2 \times 2 = 4$ ; sans biais, cela fait 8. b) Dans « $AC$ », quel neurone vient d'abord ? c) La couche d'entrée calcule-t-elle quelque chose ? d) Le réseau prédit-il la plus grande ou la plus petite sortie ? e) Un one-hot a autant de cases que de classes, toutes à 0 sauf une : laquelle, si l'on numérote les cases comme les classes ?

</details>

### 18.Q6 — Delta de sortie, delta caché, mise à jour d'un poids

<details><summary>Indice 1</summary>

Relis les §18.7 et §18.8 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un changement de poids $\Delta w$ change ce qui arrive au neurone de $a\,\Delta w$. Mise à jour : $w \leftarrow w - \eta\,a\,\delta$. Delta caché : la somme des deltas suivants, chacun multiplié par le poids de son fil.

</details>
<details><summary>Indice 3</summary>

a) $\Delta L \approx \delta \times (a\,\Delta w)$, donc la dérivée par rapport à $w$ est $a\,\delta$ : réponse A. b) Calcule $a\,\delta$, puis $1 - 0{,}1 \times a\,\delta$, en gardant le signe. c) $0{,}5 \times 2 + (-1) \times 3$ : fais le calcul. d) Les couches plus lointaines sont-elles déjà « contenues » dans les deltas de la couche suivante ?

</details>

### 18.Q7 — Rétropropager couche par couche, puis avec une activation

<details><summary>Indice 1</summary>

Relis les §18.9 et §18.10 de la fiche.

</details>
<details><summary>Indice 2</summary>

Les deltas partent de la couche dont l'erreur est connue directement, puis reculent d'une couche à la fois. Avec une activation, on multiplie par la pente de l'activation, prise au $z$ de la passe avant.

</details>
<details><summary>Indice 3</summary>

a) Seule la couche de sortie a ses deltas directement à partir de la loss : réponse D. b) Quelle couche vient juste avant la sortie dans 2 → 2 → 4 → 3 → 2 ? c) Une pente forte amplifie un changement : par quoi multiplie-t-on ? d) Calcule $\sigma(0)\,(1 - \sigma(0))$. e) Que vaut la dérivée de l'identité ?

</details>

### 18.Q8 — Learning rate trop grand ou trop petit : reconnaître les symptômes

<details><summary>Indice 1</summary>

Relis les §18.11 et §18.11.1 de la fiche, et la figure de la descente sur une parabole.

</details>
<details><summary>Indice 2</summary>

De trop grands pas sautent par-dessus le minimum ; de trop petits avancent peu. Pour c), pense à un réseau qui répond toujours la même chose. Pour d), relis le paragraphe sur les plateaux.

</details>
<details><summary>Indice 3</summary>

a) Des poids qui sautent d'un bord à l'autre et une loss qui ne baisse jamais : des pas trop **grands**, réponse A. b) Une baisse régulière mais lente : des pas trop grands ou trop petits ? c) Deux classes équilibrées, une seule réponse : quelle accuracy ? d) D'après Dauphin et coll., minimum local ou point selle ? e) Que vaut $\eta\,g$ pour $\eta = 0$ ?

</details>

### 18.Q9 — Ce que la rétropropagation propage vraiment

<details><summary>Indice 1</summary>

Relis les §18.12.1 à §18.12.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

L'erreur ne sert qu'une fois ; ce qui remonte, ce sont des taux de variation. Un pas d'apprentissage : aller, deltas de sortie, deltas cachés, mise à jour. Pour d), qui fixe la taille du pas ?

</details>
<details><summary>Indice 3</summary>

a) Faux : seuls les deltas remontent, l'erreur ne sert qu'au départ. b) Un delta est-il une erreur, une dérivée, un poids ou une sortie ? c) Que faut-il avoir calculé avant les deltas de sortie, et avant la mise à jour ? d) Le learning rate fait-il partie de la rétropropagation ?

</details>

### 18.Q10 — Neurones saturés et ReLU mortes

<details><summary>Indice 1</summary>

Relis le §18.12.4 de la fiche, et la figure des neurones ReLU.

</details>
<details><summary>Indice 2</summary>

$\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$, et $1 - \sigma(z) \approx e^{-z}$ pour un grand $z$. ReLU a une pente de 0 à gauche et de 1 à droite. Pour c), demande-toi si chaque geste rapproche les pré-activations de 0 ou supprime un plateau.

</details>
<details><summary>Indice 3</summary>

a) $1 - \sigma(15) \approx e^{-15} \approx 3 \times 10^{-7}$, et $\sigma(15) \approx 1$ : la dérivée est plus petite que $10^{-6}$, donc Vrai. b) Quelle est la pente de ReLU pour $z < 0$ ? c) Pour chaque option : rapproche-t-elle $z$ de 0, ou retire-t-elle un plateau ? d) ReLU a-t-elle un plateau pour $z > 0$ ?

</details>

### 18.Q11 — Mini-batches, mises à jour parallèles et coût du backward

<details><summary>Indice 1</summary>

Relis les §18.12.5 à §18.12.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une mise à jour par mini-batch. La passe arrière calcule deux choses par couche : les deltas qui reculent et les gradients des poids.

</details>
<details><summary>Indice 3</summary>

a) $6\,400 / 32 = 200$ mini-batches, donc 200 mises à jour. b) Quand se fait la mise à jour avec des mini-batches ? c) Combien de produits matriciels de la taille de la passe avant la passe arrière fait-elle par couche ? d) De quoi dépend la mise à jour d'un poids ?

</details>

### 18.Q12 — Aucune garantie, une longue histoire, des équations à apprivoiser

<details><summary>Indice 1</summary>

Relis les §18.12.8 à §18.12.10 de la fiche.

</details>
<details><summary>Indice 2</summary>

Comparer avec le perceptron du ch. 10, qui a une preuve. L'histoire : Linnainmaa, Werbos, Rumelhart et coll. Pour e), vérifie les formes : $\boldsymbol{\Delta}$ $(n, n_{out})$, et le résultat doit être $(n, n_{in})$.

</details>
<details><summary>Indice 3</summary>

a) Faux : la rétropropagation n'a pas la garantie du perceptron. b) Le livre conseille-t-il de calmer les choses ou de les accélérer ? c) Le mémoire de master de Linnainmaa : relis le §18.12.9. d) L'article dans *Nature* : même section. e) Quel produit de $\boldsymbol{\Delta}$ $(n, n_{out})$ par une matrice de poids donne $(n, n_{in})$ ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 18.R1 — Ch. 17 : La dérivée de la sigmoïde à partir de sa sortie

<details><summary>Indice 1</summary>

Revois la fiche du ch. 17 : les dérivées de la sigmoïde et de tanh s'écrivent avec leur propre valeur.

</details>
<details><summary>Indice 2</summary>

$\sigma' = \sigma(1 - \sigma)$ et $\tanh' = 1 - \tanh^2$. ReLU : pente 0 à gauche, 1 à droite.

</details>
<details><summary>Indice 3</summary>

a) $0{,}8 \times (1 - 0{,}8) = 0{,}16$. b) Même formule avec 0,5. c) $1 - 0{,}6^2$. d) Les formules de a) à c) ont-elles besoin de $z$ ? e) $z = -0{,}3$ est-il à gauche ou à droite de 0 ?

</details>

### 18.R2 — Ch. 15 : Chercher le meilleur learning rate par recherche en grille

<details><summary>Indice 1</summary>

Revois la fiche du ch. 15 sur `GridSearchCV` et `refit`.

</details>
<details><summary>Indice 2</summary>

Chaque valeur de la grille est entraînée sur chacun des folds ; `refit=True` réentraîne le meilleur réglage une fois de plus. Pour d), regarde quels solveurs ont un pas de gradient.

</details>
<details><summary>Indice 3</summary>

a) 4 valeurs × 5 folds = 20 entraînements, plus 1 réentraînement final : 21. b) Un learning rate se juge-t-il par pas constants ou par facteurs ? c) Parmi les attributs de `GridSearchCV` qui finissent par `_`, lequel est un dictionnaire de réglages ? d) `lbfgs` a-t-il un learning rate ?

</details>

### 18.R3 — Ch. 11 : Représentation, évaluation, optimisation dans un MLP

<details><summary>Indice 1</summary>

Revois la fiche du ch. 11 : les trois composantes de Domingos.

</details>
<details><summary>Indice 2</summary>

Représentation : ce que le modèle peut exprimer. Évaluation : la mesure qui juge une solution. Optimisation : la méthode qui cherche une bonne solution.

</details>
<details><summary>Indice 3</summary>

a) La loss juge une solution : c'est l'évaluation, B. b) La descente de gradient cherche-t-elle, juge-t-elle, ou fixe-t-elle ce qui est possible ? c) Même question pour l'architecture. d) Une meilleure recherche peut-elle trouver ce que le modèle ne peut pas exprimer ?

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 18.1 — Le delta, amplificateur de changement ✏️

<details><summary>Indice 1</summary>

$\Delta E \approx \delta\,\Delta o$ : le delta multiplie le **changement** de la sortie.

</details>
<details><summary>Indice 2</summary>

Calcule d'abord $\Delta o$ (nouvelle valeur moins ancienne), puis multiplie par $\delta$. Pour c), inverse la relation. Pour d), le delta est la dérivée $E'(o)$. Pour f), calcule $E$ aux deux points.

</details>
<details><summary>Indice 3</summary>

a) $\Delta o = 1{,}5 - 1{,}2 = 0{,}3$, et $\Delta E \approx -3 \times 0{,}3 = -0{,}9$. b) Même calcul avec $\Delta o = 3{,}6 - 4$. c) $\Delta o = \Delta E / \delta$, avec $\Delta E = -0{,}06$. d) Dérive $(o - 2)^2$, puis prends $o = 3$. e) Le delta de d) fois $0{,}1$. f) $E(3{,}1) - E(3)$. g) Que dit le signe d'une pente : la valeur de l'erreur, ou son sens de variation ?

</details>

### Ex 18.2 — Deltas de sortie avec l'erreur quadratique : attention au signe ✏️

<details><summary>Indice 1</summary>

Relis le §18.6 de la fiche et l'encadré sur le signe du delta de sortie.

</details>
<details><summary>Indice 2</summary>

Avec $L = \frac12\sum (\hat{y}_k - y_k)^2$, $\delta_k = \hat{y}_k - y_k$. Pour e) et f), calcule les nouvelles sorties, puis la nouvelle loss avec le $\frac12$.

</details>
<details><summary>Indice 3</summary>

a) $\frac12\big((0{,}6 - 0)^2 + (-0{,}2 - 1)^2\big) = \frac12(0{,}36 + 1{,}44) = 0{,}9$. b) et c) $\hat{y}_k - y_k$ pour chaque sortie. d) $\delta_2 \times 0{,}01$. e) $\hat{\mathbf{y}} - 0{,}5\,\boldsymbol{\delta}$, puis la loss. f) Même chose avec $\boldsymbol{\delta}' = \mathbf{y} - \hat{\mathbf{y}}$. g) Dérive $(\hat{y}_1 - y_1)^2$ sans le $\frac12$.

</details>

### Ex 18.3 — Descente sur une parabole : quel learning rate ? ✏️

<details><summary>Indice 1</summary>

Relis l'encadré « la descente sur une parabole » du §18.11 de la fiche.

</details>
<details><summary>Indice 2</summary>

$w_{t+1} - w^\star = (1 - \eta a)(w_t - w^\star)$, donc $w_t = w^\star + (1 - \eta a)^t\,(w_0 - w^\star)$. Convergence si $|1 - \eta a| < 1$.

</details>
<details><summary>Indice 3</summary>

a) $1 - 0{,}1 \times 4 = 0{,}6$. b) $w_1 = 1 + 0{,}6 \times (3 - 1)$. c) Même formule avec $0{,}6^3$. d) Pour quel $\eta$ le facteur vaut-il 0 ? e) Résous $|1 - 4\eta| < 1$ ; quelle borne correspond à « diverge » ? f) Calcule le facteur pour $\eta = 0{,}4$, puis élève-le au carré. g) La même condition avec $a = 0{,}1$. h) Cherche le plus petit entier $t$ tel que $2 \times 0{,}6^t < 0{,}01$ (prends le logarithme).

</details>

### Ex 18.4 — Rétropropagation complète du réseau minuscule, sans activation ✏️

<details><summary>Indice 1</summary>

Relis les §18.6 à §18.9 de la fiche : passe avant, deltas de sortie, deltas cachés, gradients, mise à jour.

</details>
<details><summary>Indice 2</summary>

Une colonne de $\mathbf{W}_1$ par neurone caché : $A_o = X \cdot XA + Y \cdot YA$. Deltas de sortie : $\hat{y} - y$. Delta caché : $\delta_A = AC\,\delta_C + AD\,\delta_D$ (les fils qui **partent** de $A$). Gradient d'un poids : sortie de départ × delta d'arrivée. Pour i), refais toute la passe avant avec les nouveaux poids.

</details>
<details><summary>Indice 3</summary>

a) $A_o = 1 \times 0{,}5 + 2 \times (-0{,}5) = -0{,}5$ et $B_o = 1 \times 1 + 2 \times 0{,}25 = 1{,}5$. b) $C_o = A_o\,AC + B_o\,BC$, $D_o = A_o\,AD + B_o\,BD$. c) $\frac12\big((C_o - 1)^2 + (D_o - 0)^2\big)$. d) $C_o - 1$ et $D_o - 0$. e) $\delta_A = AC\,\delta_C + AD\,\delta_D$, $\delta_B = BC\,\delta_C + BD\,\delta_D$. f) $A_o\,\delta_C$, $A_o\,\delta_D$, $B_o\,\delta_C$, $B_o\,\delta_D$. g) $X\,\delta_A$, $X\,\delta_B$, $Y\,\delta_A$, $Y\,\delta_B$. h) Additionne les carrés des huit gradients, multiplie par $-0{,}1$. i) Nouveaux poids $w - 0{,}1\,g$, puis a), b), c) à nouveau.

</details>

### Ex 18.5 — Le même réseau avec une sigmoïde cachée : le facteur σ'(z) ✏️

<details><summary>Indice 1</summary>

Relis le §18.10 de la fiche : on multiplie par $f'(z)$, évaluée au $z$ de la passe avant.

</details>
<details><summary>Indice 2</summary>

Les pré-activations cachées $z_A$ et $z_B$ sont celles de 18.4 a) ; les sorties cachées deviennent $\sigma(z_A)$ et $\sigma(z_B)$ ; les sorties $C$, $D$ restent linéaires. $\delta_A = \sigma'(z_A)\,(AC\,\delta_C + AD\,\delta_D)$.

</details>
<details><summary>Indice 3</summary>

a) $z_A = -0{,}5$, donc $A_o = \sigma(-0{,}5) = 1/(1 + e^{0{,}5}) \approx 0{,}3775$. b) $\sigma(z_B)$. c) $A_o\,AC + B_o\,BC$ avec les nouvelles sorties cachées. d) $C_o - 1$. e) $\sigma(z_A)\,(1 - \sigma(z_A))$. f) $\sigma'(z_A) \times (AC\,\delta_C + AD\,\delta_D)$ ; il te faut aussi $\delta_D = D_o$. g) Même chose pour $B$. h) $Y \times \delta_A$. Garde 5 ou 6 décimales jusqu'au bout.

</details>

### Ex 18.6 — La règle de la chaîne derrière les deltas ∂

<details><summary>Indice 1</summary>

Relis l'encadré « la règle de la chaîne à plusieurs chemins » du §18.8 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour 1., $L$ ne dépend de $w_{ij}$ que par $z_j$. Pour 2., $z_j$ agit sur $L$ par $a_j$, puis par chaque $z_k$ de la couche suivante. Pour 5., compte les produits d'une somme pondérée. Pour 6., écris l'approximation au premier ordre.

</details>
<details><summary>Indice 3</summary>

1. $\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial z_j}\cdot\frac{\partial z_j}{\partial w_{ij}}$, et $\frac{\partial z_j}{\partial w_{ij}} = a_i$, d'où $a_i\,\delta_j$ ; fais de même pour $b_j$. 2. Écris $\frac{\partial L}{\partial z_j} = \sum_k \frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial a_j}\frac{\partial a_j}{\partial z_j}$ et calcule chaque facteur. 3. Dérive $\frac12(a_k - y_k)^2$ par rapport à $a_k$, puis $a_k$ par rapport à $z_k$. 4. Compare $\partial L / \partial a_j$ et $\partial L / \partial z_j$ : quel facteur les sépare ? 5. Combien de multiplications pour une somme de $n$ produits, et combien de sommes ? 6. $\Delta L \approx \frac{\partial L}{\partial w}\,\Delta w$ : remplace $\Delta w$ par $-\eta\,a_i\delta_j$, puis cherche le $\Delta w$ qui donne $\Delta L = -1$.

</details>

### Ex 18.7 — Forme matricielle : XᵀΔ, ΔWᵀ et somme des lignes ∂

<details><summary>Indice 1</summary>

Relis l'encadré « gradients matriciels » du §18.12.10 de la fiche, et vérifie toujours les formes.

</details>
<details><summary>Indice 2</summary>

Écris $Z_{ik} = \sum_j X_{ij} W_{jk} + b_k$ et repère, pour chaque paramètre, **toutes** les cases de $\mathbf{Z}$ qui en dépendent : on somme sur elles. Pour 5., la moyenne d'un produit n'est pas le produit des moyennes. Pour 6., un produit $(p, q) \times (q, r)$ coûte $pqr$ multiplications.

</details>
<details><summary>Indice 3</summary>

1. $W_{jk}$ intervient dans $Z_{ik}$ pour toutes les lignes $i$, avec la dérivée $X_{ij}$ : $\frac{\partial L}{\partial W_{jk}} = \sum_i \Delta_{ik}X_{ij}$ ; reconnais un produit matriciel avec une transposée. 2. $X_{ij}$ intervient dans $Z_{ik}$ pour tous les $k$ ; $b_k$ dans toute la colonne $k$. 3. $A_{ik}$ ne dépend que de $Z_{ik}$. 4. Où apparaît le $\frac{1}{n}$ dans $\partial L / \partial \mathbf{Z}$ ? 5. Essaie deux exemples dont l'entrée et le delta changent de signe ensemble. 6. Compte $\mathbf{X}\mathbf{W}$, $\boldsymbol{\Delta}\mathbf{W}^\top$ et $\mathbf{X}^\top\boldsymbol{\Delta}$.

</details>

### Ex 18.8 — Softmax et cross-entropy : le gradient vaut p − y ∂

<details><summary>Indice 1</summary>

Relis la section sur les losses calculées sur les logits, et ∂ 17.7 (question 4).

</details>
<details><summary>Indice 2</summary>

Écris $\log p_k = z_k - \log S$ et dérive $\log S$ ($\partial \log S / \partial z_j = p_j$). Pour 3., dérive par rapport à $a$, puis multiplie par $\sigma'$. Pour 5., traite séparément $z \ge 0$ et $z < 0$.

</details>
<details><summary>Indice 3</summary>

1. $\ell = -\sum_k q_k z_k + \big(\sum_k q_k\big)\log S$ ; dérive par rapport à $z_j$, et sers-toi de $\sum_k q_k = 1$. 2. Chaque exemple ne dépend que de sa ligne ; n'oublie pas le $\frac{1}{n}$. 3. $\partial \ell / \partial a = -\frac{y}{a} + \frac{1 - y}{1 - a}$, puis multiplie par $a(1 - a)$. 4. Règle de la chaîne, puis remplace $\sigma(5) \approx 0{,}9933$ dans les deux formules. 5. Écris $\log\sigma(z)$ et $\log(1 - \sigma(z))$ avec $\log(1 + e^{-z})$, puis réécris $\log(1 + e^{-z})$ pour $z < 0$. 6. $\log_2 x = \log x / \log 2$.

</details>

### Ex 18.9 — Fermi : rétropropagation contre force brute 🧮

<details><summary>Indice 1</summary>

Relis le §18.12.7 de la fiche, et le décompte des paramètres du ch. 16.

</details>
<details><summary>Indice 2</summary>

Paramètres : $\sum (n_{\ell-1}\,n_\ell + n_\ell)$. Multiplications d'une passe avant : $\sum n_{\ell-1}\,n_\ell$ (sans les biais). Différences centrées : deux passes avant par paramètre. Rétropropagation : une passe avant plus une passe arrière qui en vaut deux.

</details>
<details><summary>Indice 3</summary>

a) $784 \times 256 + 256 + 256 \times 128 + 128 + 128 \times 10 + 10 = 235\,146$. b) Les mêmes produits sans les biais, en milliers. c) $2 \times P$. d) $2P$ passes contre 3, en milliers. e) $3 \times$ (multiplications d'une passe) $\times 60\,000 / 10^{12}$ secondes, en millisecondes. f) $2P \times$ (multiplications d'une passe) $\times 60\,000 / 10^{12}$ secondes, en heures.

</details>

<a id="reflexion"></a>

## 🗣️ 📄 Réflexion

### Ex 18.10 — Expliquer la rétropropagation sans équation 🗣️

<details><summary>Indice 1</summary>

Relis les §18.12.1 à §18.12.3 de la fiche ; cherche une image du quotidien.

</details>
<details><summary>Indice 2</summary>

Trois idées à faire passer : le réseau veut savoir comment régler chaque bouton ; on part de la fin, où l'on connaît l'erreur, et chaque étape renseigne celle d'avant ; un seul aller-retour suffit.

</details>
<details><summary>Indice 3</summary>

Une image possible : une chaîne de montage où un défaut constaté en bout de chaîne fait remonter, poste par poste, la part de chaque étape. Vérifie que ton texte ne dit pas « l'erreur remonte », mais « ce qui remonte, c'est de combien chaque étape fait varier l'erreur ».

</details>

### Ex 18.11 — Rumelhart, Hinton & Williams (1986) : l'article qui a relancé les réseaux 📄

<details><summary>Indice 1</summary>

Lis l'article en entier (quatre pages), avec ses figures ; garde la fiche (§18.6, §18.12.5, §18.12.9) à côté.

</details>
<details><summary>Indice 2</summary>

Les équations numérotées donnent 1. et 3. ; la phrase qui suit l'équation 8 donne 2. ; les légendes des figures 1 à 4 donnent les nombres de 4. et 5. ; l'avant-dernière page parle des minima locaux et des réseaux itératifs ; la dernière phrase répond à 9.

</details>
<details><summary>Indice 3</summary>

1. Cherche l'équation 3 : une double somme, sur les cas et sur les sorties. 2. Que fait-on de $\partial E / \partial w$ avant de changer les poids ? 3. Le terme $\alpha\,\Delta w(t - 1)$ garde une partie du pas précédent : quel optimiseur du ch. 19 fait cela ? 4. Pour le rapport $1 : 2 : 4$, écris la somme pondérée d'une moitié du vecteur et vérifie que les 8 configurations donnent 8 sommes différentes. 5. Les légendes des figures 2 à 4. 6. Que se passerait-il avec des poids tous égaux (ch. 16) ? 7. Que disent les auteurs des réseaux qui ont « tout juste assez » de connexions ? 8. Un pas de temps devient une couche : que faut-il garder, et quels poids doivent rester égaux ? 9. Relis la dernière phrase.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 18.E1 — Explique la rétropropagation en une minute

<details><summary>Indice 1</summary>

Le but, les deux passes, la règle de la chaîne, ce qu'elle n'est pas.

</details>
<details><summary>Indice 2</summary>

Dis ce qu'elle calcule (le gradient de la loss par rapport à tous les paramètres), comment (aller, puis retour couche par couche), son coût, et qu'un optimiseur se sert ensuite de ce gradient.

</details>
<details><summary>Indice 3</summary>

Plan en cinq phrases : 1) le gradient de la loss par rapport à tous les poids ; 2) l'aller, qui garde les valeurs intermédiaires ; 3) le retour, couche par couche, avec la règle de la chaîne ; 4) le coût, comparable à une passe avant ; 5) ce n'est pas un optimiseur, c'est la différentiation automatique en mode inverse.

</details>

### 18.E2 — Pourquoi la rétropropagation bat-elle les différences finies ?

<details><summary>Indice 1</summary>

Compare les coûts : par paramètre, ou pour tous les paramètres d'un coup.

</details>
<details><summary>Indice 2</summary>

Différences centrées : deux passes avant par paramètre. Rétropropagation : environ trois passes avant pour tous. Pense aussi à la précision, et à l'usage qui reste aux différences finies.

</details>
<details><summary>Indice 3</summary>

Plan : 1) le coût des différences finies, proportionnel au nombre de paramètres ; 2) celui de la rétropropagation, une constante ; 3) un ordre de grandeur sur un réseau d'un million de poids ; 4) la précision ; 5) leur bon usage : vérifier un gradient.

</details>

### 18.E3 — Gradient évanescent : causes, symptômes, remèdes

<details><summary>Indice 1</summary>

Relis le §18.12.4 de la fiche, et l'entretien 17.E1 du ch. 17.

</details>
<details><summary>Indice 2</summary>

Le gradient d'une couche est un produit de facteurs pris dans les couches suivantes. Que se passe-t-il si ces facteurs sont presque toujours sous 1 ?

</details>
<details><summary>Indice 3</summary>

Plan : 1) le mécanisme (un produit de dérivées et de poids) ; 2) la cause classique (des sigmoïdes saturées, une mauvaise initialisation) ; 3) le diagnostic (la norme des gradients par couche, la part de neurones saturés ou morts) ; 4) trois ou quatre remèdes (ReLU, He ou Glorot, normalisation, connexions résiduelles) ; 5) le cas inverse, l'explosion, et le gradient clipping.

</details>

### 18.E4 — Comment vérifies-tu une implémentation de gradient ?

<details><summary>Indice 1</summary>

Relis la section sur la vérification d'un gradient de la fiche.

</details>
<details><summary>Indice 2</summary>

La différence centrée, l'erreur relative et ses seuils, puis les pièges (précision, points anguleux, hasard, remise en place du paramètre).

</details>
<details><summary>Indice 3</summary>

Plan : 1) la différence centrée en float64, avec $\varepsilon$ de l'ordre de $10^{-6}$ ; 2) l'erreur relative et ses seuils ($10^{-7}$, $10^{-3}$) ; 3) trois pièges ; 4) l'automatisation dans un test et `torch.autograd.gradcheck`.

</details>

### 18.E5 — Pourquoi calculer la loss sur les logits plutôt qu'après la softmax ?

<details><summary>Indice 1</summary>

Relis la section sur les losses calculées sur les logits, et ∂ 18.8.

</details>
<details><summary>Indice 2</summary>

Deux raisons : la stabilité numérique (un logarithme de probabilité arrondie à 0) et la forme du gradient. Ajoute le piège classique de PyTorch.

</details>
<details><summary>Indice 3</summary>

Plan : 1) ce qui casse si l'on prend le log d'une softmax ou d'une sigmoïde arrondie ; 2) log-softmax par log-sum-exp ; 3) le gradient $\mathbf{p} - \mathbf{y}$, qui ne s'éteint pas ; 4) `CrossEntropyLoss` et `BCEWithLogitsLoss` ; 5) la double softmax.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices 18.12 à 18.29 (parties A à D du notebook) seront ajoutés à la prochaine session de génération, avec le notebook complet.
