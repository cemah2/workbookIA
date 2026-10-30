# 28 · Applications créatives — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch28_creatif/06_mes_reponses.md` (créée par `python tools/start_chapter.py 28`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ 28.1, 28.2 et 28.4 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les dérivations ∂ 28.3 et 28.5, les estimations 🧮 28.6 et 28.7, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"B"`), plusieurs lettres entre guillemets pour un choix multiple (`"ABD"`).
> Conventions du chapitre (fiche) : l'image générée $\mathbf{x}$, l'image de contenu (l'« image de base » du livre) $\mathbf{c}$, la référence de style $\mathbf{s}$ ; à la couche $\ell$, l'activation $A^\ell$ a $C_\ell$ canaux de $H_\ell \times W_\ell$, et $F^\ell \in \mathbb{R}^{C_\ell \times H_\ell W_\ell}$ est la même activation, chaque carte mise à plat en une ligne ; la matrice de Gram est **normalisée** : $G^\ell = \frac{1}{C_\ell H_\ell W_\ell} F^\ell (F^\ell)^\top$. Les couches de VGG16 se nomment comme dans le livre : « bloc 3, conv 1 » est la première convolution du troisième bloc (`block3_conv1` dans Keras).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 28.Q1 — Pourquoi partir d'un réseau pré-entraîné comme VGG16 🧠 ⏱️ 3 min
*Fiche §28.1, §28.2.1 · livre §28.1, §28.2, §28.2.1*

a) Vrai ou faux : dans les expériences de ce chapitre, les poids de VGG16 changent à chaque pas d'optimisation.
b) Combien VGG16 a-t-il de couches à poids (convolutions et couches denses) ?
c) Pourquoi partir d'un réseau entraîné sur ImageNet ? (A) c'est le réseau le plus rapide qui existe ; (B) ses filtres détectent déjà une grande variété de motifs, des traits aux parties d'objets ; (C) il a été entraîné sur des tableaux de maîtres.
d) Le jeu d'entraînement d'ImageNet-1k (ILSVRC) compte environ : (A) 500 000 images ; (B) 1,28 million d'images ; (C) 14 millions d'images.
e) Combien de classes ce jeu distingue-t-il ?

### 28.Q2 — Montée de gradient sur l'entrée, poids gelés 🧠 ⏱️ 3 min
*Fiche §28.2.2 · livre §28.2.2*

a) À chaque pas de la visualisation d'un filtre, on modifie : (A) les poids du réseau ; (B) les pixels de l'image d'entrée ; (C) le learning rate.
b) Vrai ou faux : faire une montée de gradient sur l'activation $a(\mathbf{x})$ revient à faire une descente de gradient sur $-a(\mathbf{x})$.
c) Le livre part : (A) d'un bruit uniforme ; (B) d'une photo ; (C) d'une image toute noire.
d) En PyTorch, l'optimiseur reçoit : (A) `model.parameters()` ; (B) `[x]`, l'image, créée avec `requires_grad=True`.
e) Vrai ou faux : si tous les poids du réseau sont gelés (`requires_grad_(False)`), `loss.backward()` calcule quand même le gradient de la loss par rapport à l'image.

### 28.Q3 — Visualiser une couche entière, pondérer ses filtres 🧠 ⏱️ 3 min
*Fiche §28.2.3 · livre §28.2.3*

a) Pour visualiser une couche entière, on maximise : (A) l'activation d'un seul de ses filtres ; (B) la somme, éventuellement pondérée, des activations de tous ses filtres ; (C) la sortie de la dernière couche dense.
b) Les images obtenues pour les couches du premier bloc montrent surtout : (A) des rayures, des couleurs et des textures fines ; (B) des objets reconnaissables, comme des yeux ou des têtes.
c) Vrai ou faux : donner un poids négatif à un filtre, c'est chercher une image qui l'éteint.
d) Combien de filtres compte la couche bloc 4, conv 3 de VGG16 ?
e) Pour une entrée de 224 × 224, la carte d'activation d'un filtre du bloc 4 mesure combien de pixels de côté ?

### 28.Q4 — Deep dream : ce qui change par rapport à la visualisation d'une couche 🧠 ⏱️ 3 min
*Fiche §28.3 · livre §28.3*

a) Parmi ces différences avec la visualisation d'une couche, lesquelles sont vraies ? (A) on part d'une photo et non d'un bruit ; (B) on combine plusieurs couches, chacune avec son poids ; (C) on met à jour les poids du réseau ; (D) dans la version de Google, on répète l'opération à plusieurs échelles (des octaves). Écris leurs lettres.
b) Vrai ou faux : le deep dream amplifie ce que le réseau croit déjà voir dans l'image.
c) Le nom donné par les chercheurs de Google à leur billet de 2015 : (A) *inceptionism* ; (B) *deep style* ; (C) *neural painting*.
d) Pour ne rêver que dans une partie de l'image, on : (A) coupe une partie du réseau ; (B) baisse le learning rate ; (C) multiplie le gradient par un masque avant de mettre l'image à jour.
e) Vrai ou faux : le deep dream et la visualisation d'un filtre maximisent tous deux une activation, alors que le transfert de style minimise un écart.

### 28.Q5 — Des yeux partout : l'écho des données d'entraînement 🧠 ⏱️ 3 min
*Fiche §28.3 · livre §28.3*

a) Pourquoi le deep dream fait-il apparaître tant d'yeux et de têtes de chiens ? (A) VGG16 a une couche spéciale pour les yeux ; (B) c'est un effet du bruit de départ ; (C) ImageNet contient de très nombreuses photos d'animaux, et le réseau a appris des détecteurs pour leurs parties.
b) Vrai ou faux : plus d'un dixième des 1 000 classes d'ImageNet-1k sont des races de chiens.
c) Un réseau entraîné sur des images satellites ferait plutôt apparaître : (A) des yeux et des museaux ; (B) des routes, des champs et des toits.
d) Rêver avec une couche **profonde** fait apparaître surtout : (A) des traits et des textures ; (B) des parties d'objets (yeux, museaux, pattes).
e) Vrai ou faux : ces hallucinations révèlent un biais des données d'entraînement, dont il faut tenir compte quand on interprète ce qu'un réseau « voit ».

### 28.Q6 — Image de base, référence de style, image générée : que minimise-t-on ? 🧠 ⏱️ 3 min
*Fiche §28.4, §28.4.2 · livre §28.4, §28.4.2*

a) Pendant le transfert de style, on optimise : (A) l'image de contenu ; (B) la référence de style ; (C) l'image générée ; (D) les poids du réseau.
b) La loss de contenu compare les activations de l'image générée à celles de : (A) l'image de contenu ; (B) la référence de style.
c) La loss de style compare les matrices de Gram de l'image générée à celles de : (A) l'image de contenu ; (B) la référence de style.
d) Vrai ou faux : le réseau (VGG16) s'entraîne pendant le transfert de style, comme le discriminateur d'un GAN.
e) Les cibles (activations du contenu, matrices de Gram du style) sont calculées une fois pour toutes avant la boucle. Combien d'images passent alors dans le réseau à chaque pas d'optimisation ?

### 28.Q7 — Ce que la matrice de Gram retient et ce qu'elle oublie 🧠 ⏱️ 3 min
*Fiche §28.4.1, §28.4.6 · livre §28.4.1, §28.4.6*

a) La matrice de Gram d'une couche de $C$ filtres, dont les cartes mesurent $H \times W$, est de taille : (A) $C \times C$ ; (B) $HW \times HW$ ; (C) $C \times HW$.
b) L'entrée $G_{ij}$ mesure : (A) la distance entre les pixels $i$ et $j$ ; (B) à quel point les filtres $i$ et $j$ s'activent ensemble, aux mêmes endroits ; (C) la position où le filtre $i$ s'active le plus.
c) Vrai ou faux : si l'on mélange les positions d'une carte d'activation (la même permutation pour tous les canaux), la matrice de Gram ne change pas.
d) Le coefficient diagonal $G_{ii}$ est proportionnel à : (A) le nombre de positions où le filtre $i$ est actif ; (B) la moyenne des carrés des activations du filtre $i$ ; (C) la plus grande activation du filtre $i$.
e) Vrai ou faux : Li et coll. (2017) montrent qu'égaler les matrices de Gram revient à rapprocher les **distributions** des features des deux images, mesurées par une MMD avec un noyau polynomial du second degré.

### 28.Q8 — Loss de contenu : couches précoces ou profondes ? 🧠 ⏱️ 3 min
*Fiche §28.4.3 · livre §28.4.3*

a) Partir d'un bruit et égaler les activations d'une couche **précoce** de la photo donne : (A) une image presque identique à la photo ; (B) seulement la disposition générale des objets.
b) Avec une couche **profonde**, l'image reconstruite : (A) est identique à la photo, au pixel près ; (B) garde la structure et les objets, mais perd les couleurs et les détails fins.
c) Le livre calcule la loss de contenu sur : (A) la couche bloc 5, conv 2 ; (B) toutes les couches ; (C) la couche bloc 1, conv 2.
d) Vrai ou faux : Gatys et coll. (2015) prennent eux aussi une couche du premier bloc pour le contenu.
e) Pourquoi une couche de contenu profonde laisse-t-elle plus de place au style ? (A) elle a moins de paramètres ; (B) elle ne fixe que la structure, et laisse libres les couleurs et les textures ; (C) elle est plus rapide à calculer.

### 28.Q9 — Loss de style, poids relatifs et variation totale 🧠 ⏱️ 3 min
*Fiche §28.4.4, §28.4.5 · livre §28.4.4, §28.4.5*

a) Le livre calcule la loss de style sur : (A) une seule couche ; (B) toutes les couches de convolution.
b) Avec les poids du livre, $\alpha = 0{,}025$ pour le contenu et $\beta = 1$ pour le style, combien de fois le poids du style est-il plus grand que celui du contenu ?
c) La variation totale pénalise : (A) les couleurs trop saturées ; (B) l'écart à l'image de contenu ; (C) les écarts entre pixels voisins, donc le bruit à haute fréquence.
d) Vrai ou faux : un bon réglage de $\alpha$ et $\beta$ dépend de la façon dont on normalise les matrices de Gram et les losses ; il ne se recopie pas d'un code à l'autre.
e) Vrai ou faux : partir de l'image de contenu plutôt que d'un bruit donne en général, en moins d'itérations, une image plus proche de la photo.

### 28.Q10 — Générer du texte mot à mot : ce qu'implique « le mot le plus probable » 🧠 ⏱️ 3 min
*Fiche §28.5 · livre §28.5*

a) Le livre choisit, à chaque pas : (A) le mot le plus probable ; (B) un mot tiré au hasard selon les probabilités du modèle.
b) Vrai ou faux : le décodage glouton donne toujours la même suite pour un même début de texte.
c) Le défaut typique d'un décodage glouton : (A) des mots qui n'existent pas ; (B) des répétitions et des boucles.
d) Vrai ou faux : un modèle mot à mot peut écrire un mot absent de son vocabulaire.
e) Diviser les logits par une température $T < 1$ avant le softmax rend la distribution : (A) plus piquée, plus proche du glouton ; (B) plus plate, plus proche du hasard uniforme.

<a id="rappels"></a>

## 🔁 Rappels

### 28.R1 — Un réseau qui sert de loss (le discriminateur) 🔁 ⏱️ 5 min
*ch. 27 · fiche §28.4.2*

a) Dans un GAN, le générateur reçoit son signal d'apprentissage : (A) du gradient d'une loss calculée par le discriminateur ; (B) d'une loss pixel à pixel avec une vraie image.
b) Dans le transfert de style, le réseau qui sert à calculer la loss (VGG16) est : (A) entraîné en même temps, comme le discriminateur ; (B) gelé.
c) Vrai ou faux : dans les deux cas, le gradient traverse un réseau pour atteindre ce que l'on optimise.
d) Dans le transfert de style de Gatys, ce que l'on optimise : (A) les poids d'un générateur ; (B) les pixels d'une image.
e) Une loss calculée sur les features d'un réseau pré-entraîné, plutôt que sur les pixels, s'appelle : (A) une loss adversariale ; (B) une loss perceptuelle.

### 28.R2 — Mélange de contenu contre mélange de représentations 🔁 ⏱️ 5 min
*ch. 25 · fiche §28.4*

a) La moyenne, pixel à pixel, d'une image MNIST de 1 et d'une image de 7 donne : (A) un chiffre plausible, entre les deux ; (B) les deux chiffres superposés, en transparence.
b) Le point milieu entre les codes latents d'un 1 et d'un 7, passé dans le décodeur d'un VAE, donne plutôt : (A) une forme intermédiaire plausible ; (B) les deux chiffres superposés.
c) Vrai ou faux : le transfert de style mélange lui aussi des représentations plutôt que des pixels : la structure d'une image (ses activations) et les statistiques des features d'une autre (ses matrices de Gram).
d) L'interpolation $\mathbf{z}(t) = (1 - t)\,\mathbf{z}_1 + t\,\mathbf{z}_2$ avec $\mathbf{z}_1 = (0, 2)$, $\mathbf{z}_2 = (4, -2)$ et $t = 0{,}25$ : sa première coordonnée.
e) Vrai ou faux : la matrice de Gram de la moyenne de deux activations, $\frac{1}{2}(F_1 + F_2)$, est la moyenne de leurs deux matrices de Gram.

### 28.R3 — Champ récepteur et hiérarchie des filtres de VGG16 🔁 ⏱️ 5 min
*ch. 21 · fiche §28.2.1*

a) Deux convolutions 3 × 3 empilées, de pas 1 : le côté du champ récepteur d'un neurone de la seconde, en pixels de l'entrée.
b) Trois convolutions 3 × 3 empilées, de pas 1 : le côté du champ récepteur.
c) Vrai ou faux : plus on va profond dans un CNN, plus le champ récepteur grandit, et plus les motifs détectés sont grands et complexes.
d) Une image CIFAR-10 de 32 × 32 traverse les cinq blocs de VGG16, chacun terminé par un max-pooling 2 × 2 de pas 2 (les convolutions gardent la taille). Le côté de la sortie.
e) Le nombre de paramètres, biais compris, d'une convolution 3 × 3 de 64 vers 64 canaux.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 28.1 — Tailles des activations et des matrices de Gram le long de VGG16 ✏️ ★ ⏱️ 10 min
**Objectif :** prévoir la forme des activations d'un réseau pré-entraîné, et voir combien la matrice de Gram les résume.
**Prérequis :** fiche §28.2.1, §28.4.1 · ch. 21 (convolution, pooling) · **Parcours :** R, M

VGG16 (fiche §28.2.1) : cinq blocs de convolutions 3 × 3 de pas 1 et de padding 1 (2, 2, 3, 3 et 3 convolutions, de 64, 128, 256, 512 et 512 filtres), chaque bloc terminé par un max-pooling 2 × 2 de pas 2. Entrée : une image couleur de 224 × 224.

a) Le nombre de valeurs dans l'activation de la couche bloc 1, conv 2 (tous canaux compris).
b) Le nombre de valeurs dans l'activation de la couche bloc 4, conv 1.
c) Le nombre de coefficients de la matrice de Gram de la couche bloc 4, conv 1.
d) Pour la couche bloc 1, conv 1 : le nombre de valeurs de l'activation divisé par le nombre de coefficients de sa matrice de Gram.
e) La matrice de Gram de la couche bloc 4, conv 1 est symétrique : combien de coefficients **distincts** au plus faut-il stocker (diagonale comprise) ?
f) En mode rapide, le notebook travaille sur des images de 128 × 128. Le côté des cartes d'activation de la couche bloc 5, conv 1.
g) Vrai ou faux : la taille de la matrice de Gram d'une couche dépend de la taille de l'image d'entrée.

Dans ta copie : pourquoi la photo de contenu et la référence de style n'ont-elles pas besoin d'avoir la même taille pour la loss de style ? En ont-elles besoin pour la loss de contenu ?

### Ex 28.2 — Une matrice de Gram à la main ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer une matrice de Gram, la normaliser, et lire ce que ses coefficients disent des filtres.
**Prérequis :** fiche §28.4.1 (encadré 🧮) · 0B (produit matriciel, transposée) · **Parcours :** R, M

Une couche de trois filtres produit des cartes de 2 × 2. Mises à plat ligne par ligne (en haut à gauche, en haut à droite, en bas à gauche, en bas à droite), elles donnent les trois lignes de $F$ :

$$F = \begin{pmatrix} 1 & 0 & 2 & 1 \\ 0 & 3 & 1 & 1 \\ 2 & 0 & 4 & 2 \end{pmatrix}$$

a) Le coefficient $(1, 2)$ du produit **non normalisé** $F F^\top$.
b) Le coefficient $(2, 2)$ de $F F^\top$.
c) Le coefficient $G_{13}$ de la matrice de Gram **normalisée** $G = \frac{1}{CHW} F F^\top$.
d) $G_{22}$, à 3 décimales.
e) La somme des neuf coefficients de $G$, à 3 décimales.
f) La similarité cosinus des filtres 1 et 2, $\frac{G_{12}}{\sqrt{G_{11}\,G_{22}}}$, à 3 décimales.
g) Vrai ou faux : si l'on retourne les trois cartes de gauche à droite (la même opération pour chacune), $G$ ne change pas.

Dans ta copie : que vaut la similarité cosinus des filtres 1 et 3, et pourquoi pouvais-tu la prévoir sans calcul ? Le livre montre une matrice de Gram de six filtres dont certains coefficients dépassent 10 : comment est-ce possible pour des filtres dont les activations restent petites ?

### Ex 28.3 — Mélanger les positions ne change pas la matrice de Gram ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer ce que la matrice de Gram oublie (la disposition), et ce qu'elle garde (des statistiques de second ordre).
**Prérequis :** Ex 28.2 · fiche §28.4.1, §28.4.6 · 0B (matrice de permutation, transposée d'un produit) · **Parcours :** M

Réponds dans ta copie. On note $N = HW$ le nombre de positions et $G = \frac{1}{CN} F F^\top$.

1. Une **matrice de permutation** $P$ ($N \times N$) a un seul 1 par ligne et par colonne. Que fait le produit $F P$ aux colonnes de $F$ ? Montre que $P P^\top = I$.
2. Déduis-en que $G(FP) = G(F)$ : mélanger les positions, de la même façon pour tous les canaux, ne change pas la matrice de Gram.
3. Le résultat reste-t-il vrai pour une matrice $Q$ quelconque telle que $Q Q^\top = I$ (une matrice orthogonale) ? Une telle transformation garde-t-elle encore les valeurs des activations ?
4. Et si l'on permute les positions d'un **seul** canal ? Donne un exemple avec deux canaux et deux positions où $G$ change.
5. On note $\mu_i$ et $\sigma_i$ la moyenne et l'écart-type (sur les $N$ positions) des activations du filtre $i$. Montre que $C\,G_{ii} = \mu_i^2 + \sigma_i^2$. Peut-on retrouver $\mu_i$ à partir de la seule matrice $G$ ?
6. Qu'en conclus-tu sur ce que la loss de style impose à l'image générée, et sur ce qu'elle laisse libre ? Relie ta réponse à l'idée du livre de tables à trois ou quatre filtres (§28.4.6).

### Ex 28.4 — Losses de contenu, de style et de variation totale sur un mini-exemple ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer à la main les trois termes de la loss du transfert de style, et voir l'effet de la normalisation.
**Prérequis :** Ex 28.2 · fiche §28.4.3 à §28.4.5 (encadrés 🧮) · **Parcours :** M

Conventions de la fiche : $L_c = \lVert F(\mathbf{x}) - F(\mathbf{c}) \rVert^2$ (une somme de carrés), $L_s = \lVert G(\mathbf{x}) - G(\mathbf{s}) \rVert_F^2$ (une seule couche, Gram normalisée), $L_{TV}$ la somme des carrés des écarts entre pixels voisins, horizontalement et verticalement.

- **Contenu** : une couche d'un seul canal de 2 × 2 ; mises à plat, $F(\mathbf{x}) = (1, 2, 0, 1)$ et $F(\mathbf{c}) = (1, 0, 1, 1)$.
- **Style** : une couche de deux canaux à deux positions ; $F(\mathbf{x}) = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}$, $F(\mathbf{s}) = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$.
- **Variation totale** : une image d'un canal de 3 × 3, $\mathbf{x} = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 2 \end{pmatrix}$.

a) $L_c$.
b) La valeur que rendrait `F.mse_loss(Fx, Fc)` de PyTorch (une **moyenne** des carrés).
c) Le coefficient $G_{12}(\mathbf{x})$.
d) $L_s$.
e) $L_{TV}$.
f) La loss totale $L = \alpha L_c + \beta L_s + \lambda L_{TV}$ avec les poids du livre, $\alpha = 0{,}025$, $\beta = 1$, $\lambda = 10^{-4}$, à 4 décimales.
g) La loss de style si l'on oublie la normalisation, c'est-à-dire avec $F F^\top$ au lieu de $G$ pour les deux images.

Dans ta copie : dans f), quel terme domine ? Par quel facteur g) multiplie-t-il d) pour une couche de 512 canaux de 28 × 28, et que devient alors le réglage $\alpha = 0{,}025$, $\beta = 1$ ?

### Ex 28.5 — Le gradient de la loss de style ∂ ★★★ ⏱️ 35 min
**Objectif :** dériver le gradient de la loss de style par rapport aux activations, avec la règle de la chaîne sur des sommes indexées.
**Prérequis :** Ex 28.2, Ex 28.4 · fiche §28.4.4 · ch. 18 (règle de la chaîne, symbole de Kronecker) · **Parcours :** M

On note $E(F) = \lVert F F^\top - S \rVert_F^2 = \sum_{i,j} \bigl(\sum_k F_{ik} F_{jk} - S_{ij}\bigr)^2$, où $S$ est une matrice symétrique fixée (la cible). Réponds dans ta copie.

1. Calcule $\frac{\partial}{\partial F_{ab}} \sum_k F_{ik} F_{jk}$ à l'aide du symbole de Kronecker $\delta$.
2. On pose $D = F F^\top - S$. Montre que $\frac{\partial E}{\partial F_{ab}} = 2 (D F)_{ab} + 2 (D^\top F)_{ab}$, puis que $\frac{\partial E}{\partial F} = 4\,(F F^\top - S)\,F$. Où sers-tu de la symétrie de $S$ ?
3. Avec la normalisation de la fiche, $L_s = \bigl\lVert \frac{1}{N'} F F^\top - G(\mathbf{s}) \bigr\rVert_F^2$, où $N' = CHW$. Écris $\frac{\partial L_s}{\partial F}$.
4. Vérifie ta formule sur l'Ex 28.4 : calcule la matrice $\frac{\partial L_s}{\partial F(\mathbf{x})}$ ($2 \times 2$).
5. Que vaut ce gradient si toutes les activations de la couche sont nulles ($F = 0$) ? Pourquoi est-ce un argument contre une image de départ uniforme et sombre, surtout derrière une ReLU ?
6. Ce gradient porte sur les activations $F$. Comment arrive-t-on au gradient par rapport aux pixels de $\mathbf{x}$, et qui le calcule dans le notebook ?

### Ex 28.6 — Combien coûte un transfert de style sur CPU ? 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer en ordres de grandeur le temps et la mémoire d'un transfert de style, et justifier le choix d'un petit réseau sur CPU.
**Prérequis :** Ex 28.1 · fiche §28.4.5 · **Parcours :** R, M

Données (documentation de torchvision) : une passe avant de VGG16 sur une image de 224 × 224 coûte environ 15,5 milliards de multiplications-additions (MAC) ; celle de SqueezeNet 1.1, 0,35 milliard. Les poids de VGG16 pèsent 528 Mo, dont 14,7 millions de paramètres dans les convolutions et 123,6 millions dans les trois couches denses. On suppose qu'un pas d'optimisation (passe avant et rétropropagation) coûte trois passes avant, et que ton CPU tient 50 milliards de MAC par seconde. Réponds dans ta copie, avec des ordres de grandeur.

1. Le temps de 300 pas de transfert de style avec VGG16 à 224 × 224.
2. Le même calcul à 512 × 512. Pourquoi le coût croît-il comme le carré du côté ?
3. Le temps de 150 pas avec SqueezeNet 1.1 à 128 × 128. Pourquoi le temps mesuré dans le notebook sera-t-il plus long que ce calcul ?
4. Le transfert de style n'utilise que les convolutions. Quelle part des 528 Mo de VGG16 sert vraiment ? Combien de Mo faut-il pour les seuls poids des convolutions, en `float32` ?
5. Les activations des treize convolutions de VGG16 à 224 × 224 totalisent environ 13,5 millions de valeurs. Leur mémoire en `float32` ? Pourquoi la rétropropagation demande-t-elle de les garder ?
6. Un réseau de transfert rapide (Johnson et coll., 2016) applique un style en **une** passe avant d'un réseau de coût comparable à celui de VGG16. Combien de fois plus vite que la méthode de Gatys à 300 pas ? Qu'a-t-il fallu payer une fois pour toutes ?

### Ex 28.7 — Vocabulaire de mots et taille de la couche de sortie d'un LSTM 🧮 ★ ⏱️ 10 min
**Objectif :** compter les paramètres d'un générateur de texte mot à mot, et voir où ils se concentrent.
**Prérequis :** fiche §28.5 · ch. 22 (LSTM, nombre de paramètres) · ch. 24 (embeddings) · **Parcours :** M

Le livre entraîne deux couches de LSTM de 128 cellules sur environ 427 000 mots, avec un vocabulaire d'environ 10 300 mots. On suppose une couche d'embedding de dimension 100 à l'entrée et une couche dense de 128 vers le vocabulaire à la sortie. Un LSTM de PyTorch de $h$ cellules sur des entrées de dimension $d$ a $4h(d + h) + 8h$ paramètres. Réponds dans ta copie.

1. Le nombre de paramètres de la couche de sortie, biais compris.
2. Ceux de l'embedding, puis des deux couches de LSTM.
3. Le total, et la part de la couche de sortie. Que t'apprend cette répartition ?
4. Combien de mots d'entraînement par paramètre ? Qu'en déduis-tu sur le risque de sur-apprentissage ?
5. On garde les 5 000 mots les plus fréquents et un token `<unk>` pour les autres (📦 28.27). La taille de la couche de sortie ? Avec la loi de Zipf, la part des mots du texte couverts par ces 5 000 mots est d'environ $\frac{\ln 5\,000 + 0{,}577}{\ln 10\,300 + 0{,}577}$ : calcule-la.
6. Un modèle qui répond au hasard, uniformément sur 10 300 mots : sa cross-entropy (en nats) et sa perplexité. Un modèle de perplexité 150 : combien de mots « hésite-t-il » en moyenne ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 28.8 — Pourquoi la matrice de Gram capture le style, en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce que la matrice de Gram retient d'une image, avec une image à toi.
**Prérequis :** fiche §28.4.1, §28.4.6 · **Parcours :** R

Explique en cinq lignes au plus, à quelqu'un qui ne connaît pas le deep learning, pourquoi compter « quels détecteurs s'allument ensemble » suffit à reconnaître un style de peinture. Ton explication doit faire apparaître (1) ce que mesurent les détecteurs, (2) ce que l'on compte, (3) ce que l'on oublie volontairement, et pourquoi c'est une bonne chose pour un style.

### Ex 28.9 — Imiter le style d'un artiste : droit d'auteur, domaine public, consentement ⚖️ ★★ ⏱️ 25 min
**Objectif :** savoir quelles œuvres on peut utiliser comme référence de style, et ce que l'on doit à leurs auteurs.
**Prérequis :** fiche §28.4.5 (encadré ⚠️), « Au-delà du livre » (encadrés 🕰️) · **Parcours :** R

Les neuf références de style du livre (fig. 28.26) : *La Nuit étoilée* (Van Gogh, mort en 1890), *Le Naufrage d'un navire de transport* (Turner, mort en 1851 ; le livre l'appelle *The Shipwreck of the Minotaur*), *Le Cri* (Munch, mort en 1944), deux œuvres de Picasso (mort en 1973), *Nighthawks* (Hopper, mort en 1967), *Les Nymphéas* (Monet, mort en 1926), *Composition VII* (Kandinsky, mort en 1944), et un tableau de l'auteur du livre. Réponds dans ta copie.

1. Dans l'Union européenne, une œuvre entre dans le domaine public le 1er janvier qui suit les 70 ans de la mort de son auteur. Lesquelles de ces œuvres y sont en 2026 ? En quelle année les autres y entreront-elles ?
2. Une photo fidèle d'un tableau du domaine public (par exemple sur Wikimedia Commons) crée-t-elle de nouveaux droits, dans l'Union européenne ? Que dit l'article 14 de la directive (UE) 2019/790 ?
3. Un **style** est-il protégé par le droit d'auteur ? Qu'est-ce qui l'est ? Pourquoi imiter le style d'un artiste vivant pose-t-il quand même des questions (éthiques, commerciales) ?
4. Pour entraîner un générateur, on copie des millions d'images. Que permet l'exception de fouille de textes et de données (article 4 de la directive 2019/790), et quelle limite lui donne le « opt-out » ? Que demande le règlement sur l'IA (article 53) aux fournisseurs de modèles à usage général ?
5. Tu publies une galerie de transferts de style. Quelles informations dois-tu donner pour chaque image (crédits, licences, paramètres) ? (Le 🛠️ 28.26 du notebook te le fera faire.)

### Ex 28.10 — Lire Gatys, Ecker et Bethge (2015), « A Neural Algorithm of Artistic Style » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article fondateur du transfert de style et comparer ses choix à ceux du livre.
**Prérequis :** Ex 28.2, Ex 28.5 · fiche §28.4 · **Parcours :** M

Lis L. A. Gatys, A. S. Ecker et M. Bethge, « A Neural Algorithm of Artistic Style » ([arXiv:1508.06576](https://arxiv.org/abs/1508.06576)) : le texte principal, la figure 1 et la section *Methods*. Réponds dans ta copie.

1. Quel réseau les auteurs utilisent-ils ? Quelles couches gardent-ils, et quel changement font-ils sur le pooling ? Pourquoi ?
2. Quelle couche sert au contenu, lesquelles au style ? Comment sont pondérées les couches de style ?
3. Écris leur loss de contenu et leur loss de style (avec le facteur $\frac{1}{4 N_\ell^2 M_\ell^2}$) avec les notations de la fiche. Que valent $N_\ell$ et $M_\ell$ ? Compare leur normalisation à celle de la fiche.
4. Quels rapports $\alpha / \beta$ essaient-ils, et que montre la figure 3 de l'article quand ce rapport change ?
5. De quelle image partent-ils, et par quelle méthode d'optimisation ? Compare avec les choix du livre (§28.4.5).
6. Que montre la figure 1 sur les reconstructions de contenu et de style couche par couche ? Relie-la aux fig. 28.19, 28.23 et 28.24 du livre.
7. Quelle affirmation de l'article sur la séparation du contenu et du style te semble la plus discutable ? Pourquoi ?

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### 28.E1 — Comment visualiser ce qu'a appris un CNN ? 💼 ★★ ⏱️ 10 min
« Comment savoir ce qu'un réseau convolutif a appris ? Quelles techniques connaissez-vous, et quelles sont leurs limites ? »

### 28.E2 — Expliquer le transfert de style neuronal et la matrice de Gram 💼 ★★ ⏱️ 10 min
« Expliquez le transfert de style neuronal. Pourquoi la matrice de Gram ? »

### 28.E3 — Quelles stratégies de décodage pour générer du texte ? 💼 ★★ ⏱️ 10 min
« Un modèle de langue donne une distribution sur le mot suivant. Comment choisissez-vous le mot à écrire ? Quels compromis ? »

### 28.E4 — Pourquoi la diffusion a-t-elle remplacé deep dream et le transfert de style de Gatys ? 💼 ★★ ⏱️ 10 min
« Aujourd'hui, comment feriez-vous un outil qui applique un style artistique à une photo ? Pourquoi pas la méthode de Gatys ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 28.11 | Un extracteur multi-couches par hooks sur un réseau pré-entraîné gelé | 📦 | ★★ | 25 |
| 28.12 | Prédire l'image qui excite un filtre précoce, puis un filtre profond | 🔮 | ★★ | 15 |
| 28.13 | Visualiser un filtre d'un réseau ImageNet : la montée de gradient du ch. 21 passe à l'échelle | 📦 | ★★ | 30 |
| 28.14 | Une galerie couche par couche (fig. 28.7) | 🎨 | ★★ | 25 |
| 28.15 | Régulariser la visualisation : décalages, flou, octaves | 🔬 | ★★★ | 40 |
| 28.16 | Deep dream sur une photo libre | 📦 | ★★★ | 40 |
| 28.17 | Choisir couches et poids, ne rêver que dans une zone | 🔬 | ★★★ | 30 |
| 28.18 | La matrice de Gram en PyTorch, vérifiée contre NumPy | 📦 | ★ | 15 |
| 28.19 | Prédire une reconstruction depuis le bruit : contenu profond seul, style seul | 🔮 | ★ | 10 |
| 28.20 | Lire les courbes de losses d'un transfert de style | 📈 | ★★ | 15 |
| 28.21 | Le transfert de style qui ne produisait que du gris | 🐛 | ★★ | 25 |
| 28.22 | Reconstructions de contenu et de style couche par couche (fig. 28.19, 28.23, 28.24) | 🎨 | ★★★ | 45 |
| 28.23 | Transfert de style complet sous budget CPU | 📦 | ★★★ | 45 |
| 28.24 | Poids contenu/style, variation totale et image de départ | 🔬 | ★★★ | 35 |
| 28.25 | Au-delà de Gram : moyennes et écarts-types des canaux | 🔬 | ★★★ | 40 |
| 28.26 | Crédits, licences et paramètres : documenter une galerie générée | 🛠️ | ★★★ | 30 |
| 28.27 | Plus de Verne : réutiliser le code du LSTM mot à mot de 24.30 sur Verne et l'entraîner plus longtemps, du charabia aux phrases | 📦 | ★★★ | 45 |
| 28.28 | Glouton, température, top-k ou top-p : répétitions et diversité | 🔬 | ★★★ | 30 |
| 28.29 | Défi : un transfert de style en moins de 2 minutes de CPU, loss de style divisée par 10 et contenu reconnaissable | 🏆 | ★★★★ | 90 |

Aucun exercice ne demande de GPU : en mode rapide, le notebook utilise SqueezeNet 1.1 (5 Mo de poids) sur des images de 128 à 192 pixels de côté, et quelques centaines de pas d'Adam au plus. Les versions complètes, avec VGG16 et des images plus grandes, sont signalées 🚀 dans le notebook.
