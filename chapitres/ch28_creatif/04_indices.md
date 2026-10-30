# 28 · Applications créatives — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 28.Q1 — Pourquoi partir d'un réseau pré-entraîné comme VGG16

<details><summary>Indice 1</summary>

Relis le §28.2.1 de la fiche et son encadré ⚠️ sur la taille d'ImageNet.

</details>
<details><summary>Indice 2</summary>

Dans tout le chapitre, le réseau sert d'instrument de mesure : demande-toi ce qui change à chaque pas, et ce qui ne change jamais. Pour b), le « 16 » de VGG16 compte des couches qui ont des poids.

</details>
<details><summary>Indice 3</summary>

a) **Faux** : seuls les pixels de l'image changent, les poids sont gelés. Pour b), additionne les convolutions des cinq blocs (2, 2, 3, 3, 3) et les couches denses. Pour c), qu'est-ce qu'un réseau entraîné sur des photos a appris ? Pour d) et e), l'encadré ⚠️ du §28.2.1 donne les chiffres.

</details>

### 28.Q2 — Montée de gradient sur l'entrée, poids gelés

<details><summary>Indice 1</summary>

Relis le §28.2.2 de la fiche et l'encadré 🧮 « dériver par rapport à une image en PyTorch ».

</details>
<details><summary>Indice 2</summary>

La formule $\mathbf{x} \leftarrow \mathbf{x} + \eta\,\nabla_{\mathbf{x}}\, a(\mathbf{x})$ dit ce qui bouge. Un optimiseur de PyTorch ne fait que des descentes : comment s'en servir pour monter ? Pour e), demande-toi ce que `requires_grad_(False)` coupe exactement, et ce qu'il laisse.

</details>
<details><summary>Indice 3</summary>

a) **B** : la formule met à jour $\mathbf{x}$, l'image. Pour b), compare le gradient de $a$ et celui de $-a$. Pour c), relis le début du §28.2.2. Pour d), qu'est-ce que l'optimiseur doit modifier ? Pour e), geler les poids coupe-t-il le chemin du gradient jusqu'à l'image ?

</details>

### 28.Q3 — Visualiser une couche entière, pondérer ses filtres

<details><summary>Indice 1</summary>

Relis le §28.2.3 de la fiche, et le §28.2.1 pour l'architecture de VGG16.

</details>
<details><summary>Indice 2</summary>

Une couche, c'est tous ses filtres : on additionne leurs activations. Pour d) et e), compte les filtres de chaque bloc et le nombre de poolings que l'image a traversés avant le bloc 4.

</details>
<details><summary>Indice 3</summary>

a) **B** : on additionne les activations de tous les filtres de la couche, éventuellement pondérées. Pour b), que détectent les premières couches d'un CNN (ch. 21) ? Pour c), que devient une somme quand un des termes est multiplié par un nombre négatif ? Pour d), compte les filtres bloc par bloc (§28.2.1). Pour e), divise 224 par 2 autant de fois qu'il y a de poolings **avant** le bloc 4.

</details>

### 28.Q4 — Deep dream : ce qui change par rapport à la visualisation d'une couche

<details><summary>Indice 1</summary>

Relis le §28.3 de la fiche, en entier.

</details>
<details><summary>Indice 2</summary>

Le §28.3 donne les deux changements du livre, puis les ingrédients de la version de Google. Pour e), demande-toi, pour chaque technique, si l'on cherche à faire monter quelque chose ou à se rapprocher d'une cible.

</details>
<details><summary>Indice 3</summary>

a) Le deep dream part d'une photo (A) et combine plusieurs couches pondérées (B) ; reste à décider pour (C) (que dit tout le chapitre des poids ?) et pour (D) (le rêve cherche-t-il à renforcer ou à effacer ce que voit le réseau ?). Pour b), que fait la montée de gradient aux motifs que le réseau détecte déjà ? Pour c), d'où vient le nom ? Pour d), quel outil limite une mise à jour à une zone de l'image ? Pour e), compare les formules du §28.2.2 et du §28.4.2.

</details>

### 28.Q5 — Des yeux partout : l'écho des données d'entraînement

<details><summary>Indice 1</summary>

Relis le dernier paragraphe du §28.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Demande-toi d'où viennent les détecteurs d'un réseau entraîné, et ce qu'ils peuvent faire apparaître. Pour d), rappelle-toi la hiérarchie des couches (§28.2.3).

</details>
<details><summary>Indice 3</summary>

a) **C** : les détecteurs d'yeux et de museaux viennent des nombreuses photos d'animaux d'ImageNet. Pour b), la fiche dit combien de classes sont des chiens. Pour c), qu'aurait appris un réseau qui n'a vu que le ciel vu d'en haut ? Pour d), à quelle profondeur apparaissent les parties d'objets ? Pour e), qu'est-ce que ces images disent du réseau ?

</details>

### 28.Q6 — Image de base, référence de style, image générée : que minimise-t-on ?

<details><summary>Indice 1</summary>

Relis les §28.4 et §28.4.2 de la fiche, et la figure des trois images.

</details>
<details><summary>Indice 2</summary>

Pour chacune des trois images, demande-toi si elle change d'un pas à l'autre ou si elle ne sert qu'à fixer une cible. La loss de contenu et la loss de style comparent chacune l'image générée à une **autre** image.

</details>
<details><summary>Indice 3</summary>

a) **C** : on optimise l'image générée ; la photo et le tableau restent fixes. Pour b) et c), quelle image fournit le contenu, laquelle le style ? Pour d), compare le réseau du transfert de style et le discriminateur d'un GAN (§28.4.2). Pour e), si les cibles sont calculées avant la boucle, quelle image reste-t-il à passer dans le réseau à chaque pas ?

</details>

### 28.Q7 — Ce que la matrice de Gram retient et ce qu'elle oublie

<details><summary>Indice 1</summary>

Relis le §28.4.1 de la fiche, son encadré 🧮 et la figure des cartes mélangées ; pour e), l'encadré ⚠️ du §28.4.6.

</details>
<details><summary>Indice 2</summary>

$G_{ij} = \frac{1}{CHW}\sum_k F_{ik} F_{jk}$ : regarde ce que désignent les indices $i$, $j$ et $k$, et sur quoi porte la somme.

</details>
<details><summary>Indice 3</summary>

a) **A** : $i$ et $j$ parcourent les filtres, d'où une matrice $C \times C$. Pour b), quand le produit $F_{ik} F_{jk}$ est-il grand ? Pour c), une somme dépend-elle de l'ordre de ses termes ? Pour d), écris $G_{ii}$ avec $j = i$. Pour e), relis ce que montre l'article de Li et coll.

</details>

### 28.Q8 — Loss de contenu : couches précoces ou profondes ?

<details><summary>Indice 1</summary>

Relis le §28.4.3 de la fiche et ses deux encadrés ⚠️.

</details>
<details><summary>Indice 2</summary>

Relis ce que la fiche dit des reconstructions depuis le bruit, couche par couche (fig. 28.19 du livre). Le second encadré ⚠️ compare trois choix de couche de contenu.

</details>
<details><summary>Indice 3</summary>

a) **A** : égaler les activations d'une couche précoce, c'est presque égaler les pixels. Pour b), que perd-on en allant plus profond ? Pour c) et d), relis le second encadré ⚠️ : la couche du livre, celle de Gatys, celle de Chollet. Pour e), si la loss de contenu ne fixe que la structure, que reste-t-il de libre pour le style ?

</details>

### 28.Q9 — Loss de style, poids relatifs et variation totale

<details><summary>Indice 1</summary>

Relis les §28.4.4 et §28.4.5 de la fiche, avec l'encadré 🧮 sur la variation totale.

</details>
<details><summary>Indice 2</summary>

Le §28.4.5 donne les couches et les poids du livre. Pour d), demande-toi ce que devient la loss de style si l'on oublie de diviser la matrice de Gram par $CHW$. Pour e), relis la remarque sur le point de départ.

</details>
<details><summary>Indice 3</summary>

a) **B** : toutes les convolutions de VGG16. Pour b), divise le poids du style par celui du contenu. Pour c), relis la formule de $L_{TV}$ : que compare-t-elle ? Pour d), un même $\beta$ donne-t-il le même dosage quand la loss de style change d'échelle ? Pour e), d'où part-on, et qu'y a-t-il encore à faire ?

</details>

### 28.Q10 — Générer du texte mot à mot : ce qu'implique « le mot le plus probable »

<details><summary>Indice 1</summary>

Relis le §28.5 de la fiche, ses encadrés ⚠️ et l'encadré 🧮 sur la température.

</details>
<details><summary>Indice 2</summary>

Pour chaque item, une question : y a-t-il du hasard dans le glouton ? Combien de sorties a la couche finale ? Que deviennent les écarts entre logits quand on les divise par $T$ ?

</details>
<details><summary>Indice 3</summary>

a) **A** : le livre prend « le mot suivant le plus probable ». Pour b), que donne deux fois le même calcul sans hasard ? Pour c), qu'est-ce qu'un modèle mot à mot peut écrire, et que se passe-t-il quand une suite devient plus probable à chaque répétition ? Pour d), compte les sorties possibles. Pour e), que deviennent les écarts entre logits quand on les divise par 0,5 ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 28.R1 — Un réseau qui sert de loss (le discriminateur)

<details><summary>Indice 1</summary>

Relis le §27.4 de la fiche du ch. 27 et le §28.4.2 de cette fiche.

</details>
<details><summary>Indice 2</summary>

Dans les deux cas, un réseau dérivable dit « à quel point c'est bon ». Ce qui diffère : ce réseau apprend-il, et que modifie-t-on ?

</details>
<details><summary>Indice 3</summary>

a) **A** : le générateur ne voit jamais de vraie image ; il apprend par le gradient qui traverse le discriminateur. Pour b), relis le §28.4.2 : le réseau est-il entraîné ? Pour c), suis le gradient jusqu'à ce qu'on optimise. Pour d), que modifie l'optimiseur chez Gatys ? Pour e), le §28.4.2 donne le nom d'une loss calculée sur des features.

</details>

### 28.R2 — Mélange de contenu contre mélange de représentations

<details><summary>Indice 1</summary>

Relis, au ch. 25, le mélange de deux images et l'interpolation dans l'espace latent ; et le début du §28.4 de cette fiche.

</details>
<details><summary>Indice 2</summary>

Une moyenne de pixels agit case par case ; un décodeur, lui, a été entraîné à produire des images. Pour e), une matrice de Gram est un produit $F F^\top$ : que donne ce produit pour une somme $F_1 + F_2$ ?

</details>
<details><summary>Indice 3</summary>

a) **B** : la moyenne de deux images superpose leurs traits, chacun à moitié d'intensité. Pour b), que fait un décodeur d'un code intermédiaire ? Pour c), relis le premier paragraphe du §28.4. Pour d), applique la formule à la première coordonnée seulement. Pour e), développe $\frac{1}{4}(F_1 + F_2)(F_1 + F_2)^\top$ et compare-le à $\frac{1}{2}(F_1F_1^\top + F_2F_2^\top)$.

</details>

### 28.R3 — Champ récepteur et hiérarchie des filtres de VGG16

<details><summary>Indice 1</summary>

Relis, au ch. 21, le champ récepteur et l'architecture de VGG16 (§21.9.1).

</details>
<details><summary>Indice 2</summary>

Une convolution 3 × 3 de pas 1 voit 3 sorties de la couche précédente ; chacune de ces sorties voyait déjà une région. Une convolution de $C_{\text{in}}$ vers $C_{\text{out}}$ canaux a $C_{\text{out}}$ filtres de taille $C_{\text{in}} \times 3 \times 3$, et un biais par filtre.

</details>
<details><summary>Indice 3</summary>

a) **5** : la seconde convolution voit 3 sorties de la première, qui voient chacune 3 pixels décalés de 1 ; ensemble, $3 + 2$ pixels. Pour b), chaque convolution de plus ajoute la même quantité. Pour c), relie la taille du champ récepteur à celle des motifs. Pour d), divise 32 par 2 autant de fois qu'il y a de poolings. Pour e), $64 \times 64 \times 3 \times 3$ poids, plus les biais.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 28.1 — Tailles des activations et des matrices de Gram le long de VGG16

<details><summary>Indice 1</summary>

Relis le §28.2.1 (l'architecture) et l'encadré 🧮 du §28.4.1 (la taille de la matrice de Gram).

</details>
<details><summary>Indice 2</summary>

Les convolutions gardent la taille ; seuls les poolings la divisent par 2. Le bloc $b$ travaille donc sur des cartes de $224 / 2^{b-1}$ pixels de côté. Une activation compte $C \times H \times W$ valeurs ; une matrice de Gram, $C \times C$ coefficients.

</details>
<details><summary>Indice 3</summary>

a) $64 \times 224 \times 224 = 3\,211\,264$. Pour b), le bloc 4 travaille sur des cartes de $224 / 8$ pixels, avec 512 filtres. Pour c), une matrice $C \times C$. Pour d), divise le nombre de valeurs de l'activation du bloc 1 par $64^2$. Pour e), une matrice symétrique $n \times n$ a $\frac{n(n+1)}{2}$ coefficients distincts. Pour f), le bloc 5 vient après quatre poolings. Pour g), de quoi dépend $C$ ?

</details>

### Ex 28.2 — Une matrice de Gram à la main

<details><summary>Indice 1</summary>

Relis le §28.4.1 de la fiche, son exemple à deux filtres et son encadré 🧮.

</details>
<details><summary>Indice 2</summary>

Le coefficient $(i, j)$ de $F F^\top$ est le produit scalaire des lignes $i$ et $j$ de $F$. Ici $C = 3$ filtres et $HW = 4$ positions : on divise par $CHW = 12$. Remarque que la ligne 3 est un multiple de la ligne 1.

</details>
<details><summary>Indice 3</summary>

a) $(1, 0, 2, 1) \cdot (0, 3, 1, 1) = 0 + 0 + 2 + 1 = 3$. Pour b), le produit scalaire de la ligne 2 avec elle-même. Pour c) et d), calcule le produit scalaire puis divise par 12. Pour e), calcule les six coefficients distincts, compte deux fois ceux hors de la diagonale, et divise par 12. Pour f), la normalisation se simplifie : $\frac{(FF^\top)_{12}}{\sqrt{(FF^\top)_{11}(FF^\top)_{22}}}$. Pour g), un retournement réordonne-t-il les positions de la même façon pour les trois cartes ?

</details>

### Ex 28.3 — Mélanger les positions ne change pas la matrice de Gram

<details><summary>Indice 1</summary>

Relis le §28.4.1 de la fiche (ce que la matrice de Gram oublie) et la figure des cartes mélangées ; au ch. 2, la formule de la variance.

</details>
<details><summary>Indice 2</summary>

1. Écris $(FP)_{ik} = \sum_m F_{im} P_{mk}$ et utilise le fait que la colonne $k$ de $P$ n'a qu'un seul 1. 2. $(AB)^\top = B^\top A^\top$. 4. Cherche deux filtres qui s'allument à la même position, puis déplace l'activation d'un seul. 5. La moyenne des carrés, la variance et la moyenne sont liées par une identité.

</details>
<details><summary>Indice 3</summary>

1. La colonne $k$ de $FP$ est une colonne de $F$ ; pour $PP^\top$, calcule $\sum_k P_{ik} P_{jk}$ en distinguant $i = j$ et $i \ne j$. 2. $G(FP) = \frac{1}{CN} F (PP^\top) F^\top$ : conclus avec 1. 3. Quelle propriété de $P$ le calcul utilise-t-il, et une matrice orthogonale l'a-t-elle ? Que fait une rotation aux valeurs ? 4. Prends deux filtres allumés à la même position, puis déplace l'activation d'un seul. 5. $C\,G_{ii} = \frac{1}{N}\sum_k F_{ik}^2$, et $\sigma^2 = \overline{F^2} - \mu^2$ ; pour la seconde question, cherche deux canaux de même moyenne des carrés et de moyennes différentes. 6. Qu'est-ce que la loss de style impose, et qu'est-ce qui décide de la disposition ?

</details>

### Ex 28.4 — Losses de contenu, de style et de variation totale sur un mini-exemple

<details><summary>Indice 1</summary>

Relis les formules des §28.4.3 à §28.4.5 de la fiche et les encadrés 🧮 sur la matrice de Gram, la norme de Frobenius et la variation totale.

</details>
<details><summary>Indice 2</summary>

Chaque loss est une somme de **carrés** d'écarts. Pour le style, calcule d'abord les deux matrices de Gram, normalisées par $CHW = 4$, puis l'écart coefficient par coefficient. Pour la variation totale, compte les écarts verticaux (entre une ligne et la suivante) et les écarts horizontaux (entre une colonne et la suivante).

</details>
<details><summary>Indice 3</summary>

a) Écarts $(0, 2, -1, 0)$, carrés $0 + 4 + 1 + 0 = 5$. Pour b), `F.mse_loss` divise par le nombre de valeurs. Pour c), $F(\mathbf{x}) F(\mathbf{x})^\top$ puis la division par 4. Pour d), calcule aussi $G(\mathbf{s})$, puis les quatre écarts au carré. Pour e), 6 écarts verticaux et 6 horizontaux. Pour f), applique les poids. Pour g), si l'on ne divise pas par 4, par combien chaque écart est-il multiplié, et son carré ?

</details>

### Ex 28.5 — Le gradient de la loss de style

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §28.4.4 et, au ch. 18, la règle de la chaîne sur des sommes indexées et le symbole de Kronecker.

</details>
<details><summary>Indice 2</summary>

$\frac{\partial F_{ik}}{\partial F_{ab}} = \delta_{ia}\delta_{kb}$. Dérive le produit $F_{ik} F_{jk}$ comme un produit ordinaire (deux termes), puis la somme des carrés $\sum_{i,j} D_{ij}^2$ par la règle de la chaîne. Pour 3, fais sortir $\frac{1}{N'}$ de la norme avant de dériver.

</details>
<details><summary>Indice 3</summary>

1. $\delta_{ia} F_{jb} + F_{ib}\,\delta_{ja}$. 2. $\frac{\partial E}{\partial F_{ab}} = \sum_{i,j} 2 D_{ij} \frac{\partial D_{ij}}{\partial F_{ab}}$ ; chaque $\delta$ fait disparaître une somme ; reconnais deux produits matriciels, puis utilise $D^\top = D$. 3. $\bigl\lVert \frac{1}{N'} F F^\top - G(\mathbf{s}) \bigr\rVert_F^2 = \frac{1}{N'^2}\bigl\lVert F F^\top - N' G(\mathbf{s}) \bigr\rVert_F^2$ : applique 2 avec $S = N' G(\mathbf{s})$. 4. Reprends la matrice d'écart de l'Ex 28.4 d). 5. Que vaut un produit par $F = 0$ ? Que vaut la dérivée d'une ReLU là où son entrée est négative ? Quels filtres une image uniforme laisse-t-elle éteints ? 6. Comment le gradient passe-t-il des activations aux pixels ?

</details>

### Ex 28.6 — Combien coûte un transfert de style sur CPU ?

<details><summary>Indice 1</summary>

Relis la fin du §28.4.5 de la fiche, l'encadré 🕰️ du §28.2.1 (les poids de VGG16) et celui du §28.4.6.

</details>
<details><summary>Indice 2</summary>

Temps = nombre de pas × coût d'un pas / vitesse du CPU, avec un pas = 3 passes avant. Le coût d'une passe est proportionnel au nombre de pixels, donc au carré du côté.

</details>
<details><summary>Indice 3</summary>

1. $300 \times 3 \times 15{,}5 \cdot 10^9$ MAC, divisé par $5 \cdot 10^{10}$ MAC/s, soit environ 280 s. Pour 2, multiplie par $(512 / 224)^2$. Pour 3, réduis d'abord le coût de SqueezeNet par $(128 / 224)^2$, puis pense à tout ce qui n'est pas du calcul. Pour 4, la part des paramètres des convolutions, et 4 octets par paramètre. Pour 5, 4 octets par valeur ; que faut-il savoir pour dériver une ReLU ou un max-pooling ? Pour 6, compare le nombre de passes avant.

</details>

### Ex 28.7 — Vocabulaire de mots et taille de la couche de sortie d'un LSTM

<details><summary>Indice 1</summary>

Relis le §28.5 de la fiche et, au ch. 22, le compte des paramètres d'un LSTM.

</details>
<details><summary>Indice 2</summary>

Une couche dense de $n$ entrées et $m$ sorties a $(n + 1)\,m$ paramètres ; un embedding de $V$ mots en dimension $d$, $V d$. Pour le LSTM, applique la formule de l'énoncé avec la bonne dimension d'entrée pour chaque couche.

</details>
<details><summary>Indice 3</summary>

1. $(128 + 1) \times 10\,300 = 1\,328\,700$. Pour 2, la première couche reçoit les vecteurs de l'embedding ($d = 100$), la seconde les sorties de la première ($d = 128$). Pour 3, additionne et calcule la part de la sortie. Pour 4, divise le nombre de mots par le nombre de paramètres. Pour 5, les 5 000 mots, plus le token `<unk>`, puis le calcul de la formule. Pour 6, la cross-entropy d'une distribution uniforme sur $V$ mots est $\ln V$ ; que vaut alors $e^{L_{CE}}$ ?

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 28.8 — Pourquoi la matrice de Gram capture le style, en cinq lignes

<details><summary>Indice 1</summary>

Relis le §28.4.1 et le §28.4.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

Trouve une image pour les filtres (des détecteurs de quoi ?), une pour le compte (qu'est-ce qu'on compte, exactement ?), et dis ce qu'on perd en comptant sur toute l'image.

</details>
<details><summary>Indice 3</summary>

Commence par « Imagine des détecteurs qui parcourent le tableau, chacun cherchant un petit motif… » ; puis le compte des paires qui s'allument au même endroit ; puis l'oubli de la position, et pourquoi un style est justement ce qui se retrouve partout sur la toile.

</details>

### Ex 28.9 — Imiter le style d'un artiste : droit d'auteur, domaine public, consentement

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ du §28.4.5 et l'encadré 🕰️ sur le droit d'auteur, à la fin de la fiche.

</details>
<details><summary>Indice 2</summary>

1. Ajoute 70 à l'année de la mort, puis passe au 1ᵉʳ janvier suivant. 3. Distingue une œuvre, une reproduction d'une œuvre, et une manière de peindre. 4. Une exception, sa condition, et ce que le règlement demande en plus aux fournisseurs.

</details>
<details><summary>Indice 3</summary>

1. Van Gogh (1890) : $1890 + 70 = 1960$, domaine public depuis le 1ᵉʳ janvier 1961 ; fais de même pour les autres, et n'oublie pas le tableau d'un auteur vivant. 2. Lis l'article 14 : une reproduction fidèle est-elle originale ? 3. Le droit d'auteur protège-t-il des idées ? Pense à la concurrence, au nom de l'artiste, à son accord. 4. Que veut dire « s'opposer de manière appropriée » pour un contenu en ligne ? 5. Liste ce dont quelqu'un aurait besoin pour **refaire** l'image et pour **savoir** s'il peut la réutiliser.

</details>

### Ex 28.10 — Lire Gatys, Ecker et Bethge (2015), « A Neural Algorithm of Artistic Style »

<details><summary>Indice 1</summary>

La section *Methods* (à la fin de l'article) contient presque toutes les réponses ; la figure 1 et la figure 3 le reste.

</details>
<details><summary>Indice 2</summary>

Cherche dans *Methods* : le nom du réseau, ce qu'ils font du pooling, les couches citées, la formule de $E_\ell$ et le sens de $N_\ell$ et $M_\ell$, le rapport $\alpha / \beta$, l'image de départ.

</details>
<details><summary>Indice 3</summary>

1. VGG-19, dont ils gardent les convolutions et les poolings, sans les couches denses ; cherche ce qu'ils changent au pooling. 2. Combien de couches pour le contenu, lesquelles pour le style ? 3. Compare $\frac{1}{4 N_\ell^2 M_\ell^2}$ au carré de la normalisation de la fiche, $\frac{1}{CHW}$. 4. La figure 3 croise deux réglages. 5. Comparé au livre : réseau, couches, taille des images. 6. Relis les fig. 28.19 et 28.24 du livre. 7. Cherche le mot « séparables » dans l'article, et ce qu'en disent les auteurs plus loin.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 28.E1 — Comment visualiser ce qu'a appris un CNN ?

<details><summary>Indice 1</summary>

Fiche §28.2.2 (et son encadré 🕰️) ; ch. 21 pour les cartes d'activation et Grad-CAM.

</details>
<details><summary>Indice 2</summary>

Trois familles : regarder les poids et les activations ; chercher l'image qui excite un filtre ; expliquer une décision par une carte d'attribution. Puis les limites.

</details>
<details><summary>Indice 3</summary>

Commence par « Je distingue comprendre un filtre et expliquer une prédiction » ; décris la maximisation d'activation avec sa régularisation, puis Grad-CAM, puis une limite (une belle image n'est pas une preuve).

</details>

### 28.E2 — Expliquer le transfert de style neuronal et la matrice de Gram

<details><summary>Indice 1</summary>

Fiche §28.4 à §28.4.5, et la figure des trois images.

</details>
<details><summary>Indice 2</summary>

Trois images, un réseau gelé, deux losses (et la variation totale), ce que l'on optimise ; puis la définition de la matrice de Gram et ce qu'elle oublie.

</details>
<details><summary>Indice 3</summary>

Commence par « On optimise les pixels d'une image pour qu'elle ait les activations profondes d'une photo et les matrices de Gram d'un tableau » ; définis $G = F F^\top$ ; termine par la lenteur et les méthodes rapides.

</details>

### 28.E3 — Quelles stratégies de décodage pour générer du texte ?

<details><summary>Indice 1</summary>

Fiche §28.5, l'encadré 🧮 sur le top-k et le top-p et l'encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

Deux familles : choisir (glouton, faisceau) ou tirer au sort (température, top-k, top-p). Puis : laquelle pour quelle tâche ?

</details>
<details><summary>Indice 3</summary>

Commence par « Le modèle donne une distribution, le décodage choisit » ; décris le glouton et son défaut, puis l'échantillonnage et ses réglages, en expliquant la différence entre top-k et top-p ; termine par le choix selon la tâche.

</details>

### 28.E4 — Pourquoi la diffusion a-t-elle remplacé deep dream et le transfert de style de Gatys ?

<details><summary>Indice 1</summary>

Fiche « Au-delà du livre : la génération d'images aujourd'hui », et l'encadré 🕰️ du §28.4.6.

</details>
<details><summary>Indice 2</summary>

Deux parties : les limites de Gatys (coût, sémantique), puis l'outil que tu construirais aujourd'hui, brique par brique.

</details>
<details><summary>Indice 3</summary>

Commence par « Gatys optimise chaque image et ne connaît du style que des statistiques de texture » ; puis la diffusion latente, ControlNet pour la structure, IP-Adapter ou LoRA pour le style ; termine par une option légère et la question des droits.

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex 28.11 — Un extracteur multi-couches par hooks sur un réseau pré-entraîné gelé

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « les *hooks* de PyTorch » du §28.2.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un dictionnaire vide, un hook par couche qui y range la sortie du module, une seule passe `net(x)`, puis on retire les hooks. Attention à la fermeture (*closure*) d'une `lambda` dans une boucle : passe le nom en argument par défaut, sinon toutes les lambdas rangent sous le dernier nom.

</details>
<details><summary>Indice 3</summary>

```python
acts, handles = {}, []
for name, index in layers.items():
    handles.append(net[index].register_forward_hook(lambda module, inputs, output, name=name: acts.__setitem__(name, output)))
try:
    ...                      # la passe
finally:
    ...                      # retirer chaque hook
```
Pour a) et b), lis `.numel()` et `.shape[-1]` sur les tenseurs rendus.

</details>

### Ex 28.12 — Prédire l'image qui excite un filtre précoce, puis un filtre profond

<details><summary>Indice 1</summary>

Relis les §28.2.2 et §28.2.3 de la fiche, et ton 🔁 R3 sur le champ récepteur.

</details>
<details><summary>Indice 2</summary>

Un filtre ne peut chercher un motif plus grand que ce qu'il voit de l'image. Calcule l'ordre de grandeur du champ récepteur de `conv1` (un noyau 3 × 3, un pas de 2) et de `fire8` (après trois poolings de pas 2).

</details>
<details><summary>Indice 3</summary>

Pour chaque couche, estime deux nombres en pixels de l'image : la région que voit un neurone, et l'écart entre deux neurones voisins (le produit des pas traversés). Place-les ensuite dans les catégories de l'énoncé.

</details>

### Ex 28.13 — Visualiser un filtre d'un réseau ImageNet : la montée de gradient du ch. 21 passe à l'échelle

<details><summary>Indice 1</summary>

Relis le §28.2.2 de la fiche, son encadré 🧮 sur la dérivation par rapport à une image, et ton 21.28.

</details>
<details><summary>Indice 2</summary>

`activation_13` : normalise, passe dans `net[:layer_index + 1]`, prends le canal, la moyenne des carrés. `visualize_filter_13` : l'image est le seul paramètre ; on minimise l'opposé de l'activation ; `clamp_` sous `torch.no_grad()` après chaque pas.

</details>
<details><summary>Indice 3</summary>

```python
x = (torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed)) * 0.2 + 0.4).requires_grad_(True)
optimizer = torch.optim.Adam([x], lr=lr)
for _ in range(steps):
    ...                      # zero_grad, loss = -activation_13(...), backward, step, puis clamp_
return to_image(x)
```

</details>

### Ex 28.14 — Une galerie couche par couche (fig. 28.7)

<details><summary>Indice 1</summary>

Relis le §28.2.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Reprends `visualize_filter_13` ; seule change la quantité qui monte : toute la sortie de `net[:layer_index + 1]`, et non un seul canal.

</details>
<details><summary>Indice 3</summary>

La loss devient `-(net[:layer_index + 1](normalize(x)) ** 2).mean()` ; le reste de la boucle est identique.

</details>

### Ex 28.15 — Régulariser la visualisation : décalages, flou, octaves

<details><summary>Indice 1</summary>

Relis la fin du §28.2.2 de la fiche et son encadré 🕰️ ; l'article « Feature Visualization » de Distill montre chaque régularisation.

</details>
<details><summary>Indice 2</summary>

Construis d'abord la liste des tailles des octaves (de la plus petite à `size`) ; pour chaque taille, agrandis l'image précédente puis fais `steps` pas. Dans un pas : tire le décalage, roule l'image, calcule la loss sur l'image roulée, recule. Le flou se fait sous `torch.no_grad()`, avec `x.copy_(...)` pour garder le même tenseur que l'optimiseur.

</details>
<details><summary>Indice 3</summary>

```python
sizes = [max(16, round(size / octave_scale ** k)) for k in range(octaves - 1, -1, -1)]
dy, dx = torch.randint(-jitter, jitter + 1, (2,), generator=shifts).tolist() if jitter else (0, 0)
rolled = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
```
Sans régularisation, ta fonction doit faire exactement les mêmes opérations que `visualize_filter_13` : même départ, même optimiseur.

</details>

### Ex 28.16 — Deep dream sur une photo libre

<details><summary>Indice 1</summary>

Relis le §28.3 de la fiche (les octaves et le masque).

</details>
<details><summary>Indice 2</summary>

`dream_loss_16` : une passe par `extract_11` pour toutes les couches du dictionnaire, puis la somme pondérée des moyennes des carrés. `deep_dream_16` : une liste de tailles (de la plus petite à celle de la photo) ; à chaque échelle, le point de départ est le rêve précédent agrandi, plus les détails perdus ; on calcule le gradient avec `torch.autograd.grad`, on le normalise, on le masque, on **ajoute**.

</details>
<details><summary>Indice 3</summary>

```python
resize = lambda t, s: F.interpolate(t, size=s, mode="bilinear", align_corners=False)
detail = resize(base, shape) - resize(resize(base, previous_shape), shape)   # ce que la réduction a perdu
(g,) = torch.autograd.grad(dream_loss_16(net, x, layer_weights), x)
```
Le masque se redimensionne avec `mode="nearest"`, et multiplie `g` avant la mise à jour.

</details>

### Ex 28.17 — Choisir couches et poids, ne rêver que dans une zone

<details><summary>Indice 1</summary>

Relis le §28.3 de la fiche et ta `deep_dream_16`.

</details>
<details><summary>Indice 2</summary>

`PHOTO_B.shape` donne `(3, H, W)` : ton masque est un tenseur `(H, W)`. Pour une bande, `torch.zeros(H, W)` puis une tranche mise à 1.

</details>
<details><summary>Indice 3</summary>

```python
mask_17 = torch.zeros(PHOTO_B.shape[1:])
mask_17[: PHOTO_B.shape[1] // 3] = 1        # par exemple : le tiers du haut
```
Pour les couches, garde au moins une couche intermédiaire (`fire5` à `fire7`) et compare deux jeux de poids.

</details>

### Ex 28.18 — La matrice de Gram en PyTorch, vérifiée contre NumPy

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §28.4.1 (la matrice de Gram) et celui du §28.4.5 (la variation totale) ; ✏️ 28.2 et 28.4.

</details>
<details><summary>Indice 2</summary>

Lis `C, H, W` sur les trois derniers axes, aplatis les deux derniers, multiplie par la transposée des **deux derniers** axes (ce qui marche avec ou sans axe de batch). Pour la variation totale, des tranches décalées d'un pixel.

</details>
<details><summary>Indice 3</summary>

```python
def gram_matrix_18(A):
    C, H, W = A.shape[-3:]
    flat = A.reshape(*A.shape[:-2], H * W)       # les cartes en lignes, axe de batch gardé s'il existe
    ...                                          # produit par la transposée des DEUX derniers axes, puis la normalisation
```
Pour `tv_loss_18`, les écarts verticaux sont `x[..., 1:, :] - x[..., :-1, :]`.

</details>

### Ex 28.19 — Prédire une reconstruction depuis le bruit : contenu profond seul, style seul

<details><summary>Indice 1</summary>

Relis les §28.4.3 et §28.4.4 de la fiche, et ∂ 28.3.

</details>
<details><summary>Indice 2</summary>

Pour a), demande-toi ce que l'image doit garder pour que les activations de `fire7` coïncident : des pixels exacts, ou seulement de quoi produire les mêmes détections ? Pour b), qu'est-ce que la loss de style impose à la **disposition** de l'image ?

</details>
<details><summary>Indice 3</summary>

Une erreur pixel moyenne d'environ 0,005 correspond à une image presque identique ; une erreur de l'ordre de 0,1 à 0,2, à une image dont les couleurs et les détails diffèrent beaucoup (un bruit uniforme a une erreur de l'ordre de 0,2 avec une photo). Pour b), la corrélation mesure si l'image reconstruite est claire là où la photo est claire.

</details>

### Ex 28.20 — Lire les courbes de losses d'un transfert de style

<details><summary>Indice 1</summary>

Relis le §28.4.5 de la fiche (le point de départ, les poids) et ✏️ 28.4.

</details>
<details><summary>Indice 2</summary>

Au pas 0, un départ par la photo donne un terme de contenu nul : sur une échelle logarithmique, la courbe part d'en bas (ou de hors du graphique). Une graduation de l'axe vertical, c'est un facteur 10.

</details>
<details><summary>Indice 3</summary>

a) **A** : c'est le seul essai dont la courbe de contenu existe dès le pas 0 (les deux autres partent de la photo, contenu nul). b) Pour chaque essai, compare les courbes de style et de contenu au dernier pas : l'écart dépasse-t-il une graduation ? c) Repère la puissance de dix juste en dessous du point de départ et du point d'arrivée de la courbe de style de l'essai A, et compte les graduations entre les deux.

</details>

### Ex 28.21 — Le transfert de style qui ne produisait que du gris

<details><summary>Indice 1</summary>

Relis les pièges ⚠️ de la fiche : ils listent les erreurs classiques d'une optimisation d'image.

</details>
<details><summary>Indice 2</summary>

Lis les diagnostics dans l'ordre : chacun pointe une famille de lignes. Une ligne fautive n'est pas toujours une ligne qui plante : c'est une ligne dont **ce qu'elle calcule** est faux (l'entrée du réseau, l'échelle de la matrice de Gram, les paramètres de l'optimiseur, la plage des pixels).

</details>
<details><summary>Indice 3</summary>

Le premier diagnostic désigne la ligne K : l'optimiseur reçoit `net.parameters()`, gelés, et l'image ne bouge jamais ; il faut `Adam([x], lr=lr)`. Corrige-la, relance `diagnose_21` sur ta version, et suis le diagnostic suivant : pour chacun, cherche la ligne qui calcule la quantité qu'il mesure.

</details>

### Ex 28.22 — Reconstructions de contenu et de style couche par couche (fig. 28.19, 28.23, 28.24)

<details><summary>Indice 1</summary>

Relis les §28.4.3 et §28.4.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Les cibles se calculent une fois, sans graphe : les activations de `target` (contenu) ou leurs matrices de Gram (style). Puis une boucle d'Adam sur l'image, comme en 28.13, avec la loss qui correspond à `kind`.

</details>
<details><summary>Indice 3</summary>

```python
with torch.no_grad():
    acts_t = extract_11(net, normalize(to_batch(target)), layers)
    goals = {n: (acts_t[n] if kind == "content" else gram_matrix_18(acts_t[n])) for n in layer_names}
```
Dans la boucle : `F.mse_loss(acts[n], goals[n])` pour le contenu, `((gram_matrix_18(acts[n]) - goals[n]) ** 2).sum()` pour le style.

</details>

### Ex 28.23 — Transfert de style complet sous budget CPU

<details><summary>Indice 1</summary>

Relis les §28.4.2 à §28.4.5 de la fiche et la figure des trois losses ; ✏️ 28.4.

</details>
<details><summary>Indice 2</summary>

Trois étapes : les cibles (une fois, sans graphe), le départ (copie du contenu ou bruit de graine `seed`), la boucle. Écris une petite fonction interne qui rend les trois termes **pondérés** pour une image : tu t'en sers pour l'historique et pour la loss. La boucle compte `steps + 1` évaluations (la dernière après le dernier pas, sans pas).

</details>
<details><summary>Indice 3</summary>

```python
layers = {name: LAYERS[name] for name in set(content_layers) | set(style_layers)}
for step in range(steps + 1):
    t_c, t_s, t_tv = terms(x)              # alpha * L_c, beta * L_s, tv * L_tv
    ...                                    # enregistrer si step % every == 0 ou step == steps ; s'arrêter si step == steps
```

</details>

### Ex 28.24 — Poids contenu/style, variation totale et image de départ

<details><summary>Indice 1</summary>

Relis le §28.4.5 de la fiche (les quatre remarques pratiques).

</details>
<details><summary>Indice 2</summary>

Ne change qu'une chose à la fois par rapport aux réglages de la partie C : trois valeurs de `beta` sur une échelle logarithmique, puis `tv` à 0, puis le départ.

</details>
<details><summary>Indice 3</summary>

```python
settings_24 = [{"beta": 1.0}, {"beta": 10.0}, {"beta": 100.0}, {"tv": 0.0}, ...]
```
Complète avec un départ par le bruit, et lis le tableau colonne par colonne.

</details>

### Ex 28.25 — Au-delà de Gram : moyennes et écarts-types des canaux

<details><summary>Indice 1</summary>

Relis ∂ 28.3 (points 5 et 6) et l'encadré 🕰️ du §28.4.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

Aplatis les positions `(N, C, H * W)`, puis moyenne et variance sur le dernier axe ; attention à la variance de **population**.

</details>
<details><summary>Indice 3</summary>

```python
flat = A.reshape(A.shape[0], A.shape[1], -1)     # (N, C, H * W)
```
Puis `flat.mean(dim=2)` ; pour l'écart-type, `flat.var(dim=2, unbiased=False)`, sans oublier le $10^{-5}$ sous la racine. `stats_loss_25` additionne les carrés des écarts des deux statistiques.

</details>

### Ex 28.26 — Crédits, licences et paramètres : documenter une galerie générée

<details><summary>Indice 1</summary>

Relis ⚖️ 28.9 (point 5) et la data card `data/cards/images.md` (`wb.datasets.dataset_card("images_ch28")`).

</details>
<details><summary>Indice 2</summary>

Construis le README comme une liste de lignes, puis `"\n".join(lines)`. Pour chaque entrée : `plt.imsave` du PNG (l'image en `(H, W, 3)`), le lien Markdown, les deux crédits, les réglages. À la fin, une section « Reproduire ».

</details>
<details><summary>Indice 3</summary>

```python
plt.imsave(folder / entry["file"], np.clip(np.asarray(entry["image"]).transpose(1, 2, 0), 0, 1))
lines += [f"![{entry['file']}]({entry['file']})", f"- Contenu : {wb.datasets.image_credit(entry['content'])}"]
lines.append("- Réglages : " + ", ".join(f"{k}: {v}" for k, v in entry["params"].items()))
```

</details>

### Ex 28.27 — Plus de Verne : réutiliser le code du LSTM mot à mot de 24.30 sur Verne et l'entraîner plus longtemps, du charabia aux phrases

<details><summary>Indice 1</summary>

Relis ton 24.30 et le §28.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une seule fonction change vraiment : le tokenizer. Dans une expression régulière, `[^\W\d_]` veut dire « un caractère de mot qui n'est ni un chiffre ni un tiret bas » : une lettre, accentuée ou non. Ajoute une apostrophe facultative après les lettres, et une alternative pour tout autre caractère qui n'est pas une espace.

</details>
<details><summary>Indice 3</summary>

```python
WORD_RE_27 = re.compile(r"[^\W\d_]+['’]?|\S")
```
`train_lm_27` est `train_lm_30` avec `on_epoch(epoch, model)` appelé à la fin de chaque epoch (epoch comptée à partir de 1).

</details>

### Ex 28.28 — Glouton, température, top-k ou top-p : répétitions et diversité

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « température, top-k et top-p » du §28.5 de la fiche, et `sample_from_logits` (ch. 22).

</details>
<details><summary>Indice 2</summary>

Calcule une fois les probabilités (softmax de `logits / temperature`, en soustrayant le max), puis l'ordre décroissant ; chaque stratégie ne fait que choisir quels indices garder. Pour le top-p, la somme cumulée des probabilités triées dit combien en garder.

</details>
<details><summary>Indice 3</summary>

```python
order = np.argsort(-probs, kind="stable")
keep = order[: int(np.searchsorted(np.cumsum(probs[order]), p)) + 1]      # top-p
return int(rng.choice(keep, p=probs[keep] / probs[keep].sum()))
```
Pour `repeated_share_28`, un ensemble des n-grammes déjà vus, parcourus dans l'ordre.

</details>

### Ex 28.29 — Défi : un transfert de style en moins de 2 minutes de CPU, loss de style divisée par 10 et contenu reconnaissable

<details><summary>Indice 1</summary>

Relis le tableau de ton 🔬 28.24 : quelle direction réduit la distance de contenu ?

</details>
<details><summary>Indice 2</summary>

Le critère de style (÷ 10) est facile ; c'est le contenu qui limite. Cherche le plus petit `beta` qui divise encore le style par 10, avec un départ par la photo ; un learning rate plus petit et plus de pas évitent d'abîmer la photo.

</details>
<details><summary>Indice 3</summary>

Règle sur `"china"` et `"starry_night"` : fais varier `beta` sur une échelle logarithmique (1, 0,3, 0,1, 0,03…) et mesure les deux distances avec `measure_24(img, content, style)` ; garde une marge sur le rapport de style, car il change d'un tableau à l'autre. Puis un seul appel à `evaluate_29`.

</details>
