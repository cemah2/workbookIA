# 21 · Réseaux convolutifs (CNN) — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch21_cnn/06_mes_reponses.md` (créée par `python tools/start_chapter.py 21`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 21.1 à 21.6 et de l'estimation 🧮 21.8 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la preuve ∂ 21.7, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), une forme ou une position entre guillemets (`"(3, 5)"`).
> Conventions du chapitre : celles de PyTorch (fiche). La « convolution » d'une couche est une **corrélation croisée** (le noyau n'est pas retourné) ; la sortie en position $(i, j)$ correspond à la fenêtre dont le coin en haut à gauche est en $(i, j)$ dans l'entrée complétée ; lignes et colonnes se numérotent **à partir de 0**, la ligne d'abord ; une image a la forme `(C, H, W)`, un mini-batch `(N, C, H, W)`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 21.Q1 — Les deux sens de « profondeur » 🧠 ⏱️ 3 min
*Fiche §21.1, §21.2, §21.2.1 · livre §21.1 à §21.2.1 · parcours R*

a) La profondeur (au sens des canaux) d'une photo en couleur de 224 × 224 pixels.
b) La profondeur du tenseur de forme `(64, 112, 112)` que produit une couche de 64 filtres.
c) Quand on dit que VGG16 est un réseau de profondeur 16, de quel sens parle-t-on ? (A) le nombre de canaux de sa dernière couche ; (B) son nombre de couches de calcul ; (C) le nombre de bits par pixel ; (D) la taille de ses filtres.
d) Une image en couleur de 32 × 32 pixels, aplatie en un vecteur : son nombre de features.
e) Vrai ou faux : un réseau ne s'appelle CNN que si toutes ses couches sont des convolutions.

### 21.Q2 — Un seul neurone promené sur toute l'image 🧠 ⏱️ 3 min
*Fiche §21.2.2 à §21.2.5 · livre §21.2.2 à §21.2.5 · parcours R*

a) Vrai ou faux : avec le partage de poids, toutes les positions de l'image sont évaluées avec les mêmes poids.
b) Le nombre de poids (sans biais) d'un filtre 5 × 5 sur une image en niveaux de gris.
c) Vrai ou faux : le nombre de poids d'un filtre dépend de la taille de l'image à laquelle on l'applique.
d) Le pixel de l'image qui se trouve sous l'ancre du filtre s'appelle : (A) le noyau ; (B) le pixel focal ; (C) le champ récepteur ; (D) la carte de caractéristiques.
e) Pourquoi préfère-t-on des filtres de côté impair (3, 5, 7) ? (A) ils ont moins de poids ; (B) ils ont un pixel central, où placer l'ancre ; (C) ils s'entraînent plus vite ; (D) les GPU ne savent pas calculer les autres.

### 21.Q3 — Convolution ou corrélation croisée ? 🧠 ★★ ⏱️ 3 min
*Fiche §21.3, §21.3.2 · livre §21.3, §21.3.2 · parcours R, M*

a) Vrai ou faux : `nn.Conv2d` retourne le noyau avant de le faire glisser.
b) Une entrée $x = (1, 2, 3, 4)$ et un noyau $w = (1, 0, -1)$ : la première valeur de leur **corrélation croisée** sans padding (ce que calcule `nn.Conv1d`).
c) La première valeur de leur **convolution** au sens des mathématiciens (noyau retourné), sans padding.
d) Vrai ou faux : pour le noyau symétrique $(1, 2, 1)$, convolution et corrélation croisée donnent le même résultat.
e) La vue « œil de mouche » d'une image 6 × 6 pour un filtre 3 × 3 (sans padding, stride 1) : son nombre de morceaux.
f) Le nombre de valeurs de chaque morceau (une colonne d'`im2col`) pour la même image, si elle a 3 canaux.

### 21.Q4 — Détecteurs de motifs et hiérarchie de filtres 🧠 ⏱️ 3 min
*Fiche §21.3.1, §21.3.3 · livre §21.3.1, §21.3.3 · parcours R*

Un filtre 3 × 3 vaut $+1$ dans sa colonne de **gauche** et $-1$ partout ailleurs. On le pose sur des morceaux 3 × 3 d'une image binaire (0 noir, 1 blanc).

a) Le plus grand score qu'il puisse obtenir.
b) Son score sur un morceau dont la colonne de gauche est blanche, ainsi que le pixel du centre (tout le reste est noir).
c) Vrai ou faux : dans un CNN, on dessine les filtres à la main avant l'entraînement.
d) Pourquoi un filtre 3 × 3 d'une couche haute « voit-il » une région plus grande de l'image qu'un filtre 3 × 3 de la première couche ? (A) il a plus de poids ; (B) l'image a été réduite entre les deux (pooling ou stride) et chacune de ses entrées résume déjà une région ; (C) son learning rate est plus grand ; (D) il reçoit l'image en couleur.
e) Dans ce chapitre, une « feature » détectée par un filtre est : (A) une colonne d'un tableau de données ; (B) un motif de l'image que le filtre cherche ; (C) un hyperparamètre de la couche ; (D) un pixel de l'image d'entrée.

### 21.Q5 — Padding et stride : que devient la taille ? 🧠 ⏱️ 3 min
*Fiche §21.3.4, §21.3.5 · livre §21.3.4, §21.3.5 · parcours R*

a) Une image de 12 × 12, un filtre 5 × 5, sans padding, stride 1 : la largeur de sortie.
b) Le padding (de chaque côté) qui garde la taille d'une image avec un filtre 7 × 7 et un stride de 1.
c) Une image de 7 lignes et 12 colonnes, un filtre 3 × 3 sans padding, un stride de 2 verticalement et de 3 horizontalement : la hauteur de sortie.
d) Sa largeur de sortie.
e) Vrai ou faux : avec un filtre 3 × 3 et un stride de 3, sans padding, chaque pixel d'une image 9 × 9 est utilisé exactement une fois.

### 21.Q6 — Canaux d'entrée, nombre de filtres, canaux de sortie 🧠 ★★ ⏱️ 3 min
*Fiche §21.4, §21.4.1, §21.4.2 · livre §21.4 à §21.4.2 · parcours R*

Une couche reçoit un tenseur à 6 canaux et lui applique 4 filtres 3 × 3 (convolution standard).

a) La profondeur (nombre de canaux) de chaque filtre.
b) Le nombre de canaux de sortie.
c) Le nombre de poids de la couche, sans les biais.
d) Son nombre de paramètres, biais compris.
e) Une convolution **par canal** (`groups=6`, un filtre 3 × 3 par canal, 6 canaux de sortie) sur la même entrée : son nombre de poids, sans les biais.

### 21.Q7 — Convolution 1D ou convolution 1×1 ? 🧠 ⏱️ 3 min
*Fiche §21.5, §21.6 · livre §21.5, §21.6*

Une phrase de 20 mots ; chaque mot est un vecteur de 50 nombres (une matrice de 20 lignes et 50 colonnes).

a) Un filtre de convolution 1D couvre 3 mots consécutifs, sur toute la largeur : son nombre de poids (sans biais).
b) Son nombre de positions le long de la phrase (sans padding, stride 1).
c) Une convolution 1 × 1 de 300 canaux vers 175 canaux : son nombre de poids, sans les biais.
d) Vrai ou faux : une convolution 1 × 1 mélange les canaux d'un même pixel sans regarder les pixels voisins.
e) Cette convolution 1 × 1 reçoit un tenseur de forme `(8, 300, 28, 28)` : la hauteur de sa sortie.

### 21.Q8 — Couche de convolution et initialisation des filtres 🧠 ⏱️ 3 min
*Fiche §21.7, §21.7.1 · livre §21.7, §21.7.1*

a) Vrai ou faux : si **tous** les poids et les biais d'une couche de convolution et des couches suivantes partent d'une même valeur (une pour les poids, une pour les biais), les filtres de cette couche restent identiques pendant tout l'entraînement.
b) Le fan-in d'un neurone d'une convolution 3 × 3 sur une entrée à 64 canaux.
c) L'écart-type de l'initialisation de He pour ce fan-in, $\sqrt{2/\text{fan-in}}$ (3 décimales).
d) La sortie d'une couche de convolution, une tranche par filtre, s'appelle : (A) un noyau ; (B) une carte de caractéristiques (*feature map*) ; (C) un champ récepteur ; (D) un padding.
e) Vrai ou faux : `nn.Conv2d(3, 16, 3)`, sans autre argument, garde la hauteur et la largeur de l'image.

### 21.Q9 — Convolution transposée : les mots piégés 🧠 ★★ ⏱️ 3 min
*Fiche §21.8 · livre §21.8 · parcours M*

a) Vrai ou faux : le paramètre `dilation=2` de `nn.Conv2d` insère des zéros entre les pixels de l'**entrée**, comme la « convolution dilatée » du livre.
b) Une entrée de 4 × 4 et `nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2)` : la largeur de sortie.
c) La même couche avec `padding=1` : la largeur de sortie.
d) Combien de zéros insère-t-on entre deux pixels voisins de l'entrée pour une convolution transposée de stride 3 ?
e) Vrai ou faux : une convolution transposée peut produire des artefacts en damier.
f) Un noyau 3 × 3 de dilatation 3 couvre une région de combien de pixels de côté ?

### 21.Q10 — VGG16 et ce que voient ses filtres 🧠 ⏱️ 3 min
*Fiche §21.9 à §21.9.3 · livre §21.9 à §21.9.3*

a) La forme du tenseur à la sortie du troisième bloc de VGG16 (après son pooling) pour une image de 224 × 224, au format `(C, H, W)` (entre guillemets).
b) Le nombre de max poolings de VGG16.
c) Le côté des cartes d'activation de la première convolution du cinquième bloc (`block5_conv1`), pour une image de 224 × 224.
d) Pour trouver l'image qui excite le plus un filtre, on fait : (A) une descente de gradient sur les poids ; (B) une montée de gradient sur les pixels, les poids restant fixés ; (C) un dropout sur l'image ; (D) une recherche parmi les images d'entraînement seulement.
e) Vrai ou faux : les filtres des premières couches de VGG16 répondent surtout à des bords et des rayures, ceux des couches profondes à des textures et des motifs plus complexes.

### 21.Q11 — Exemples adversariaux : vrai ou faux ? 🧠 ⏱️ 3 min
*Fiche §21.10, section sur l'attaque FGSM · livre §21.10 · parcours R*

a) Vrai ou faux : une perturbation adversariale est en général visible à l'œil nu.
b) Vrai ou faux : une perturbation universelle trompe un même classifieur sur un grand nombre d'images différentes.
c) Vrai ou faux : seuls les réseaux mal entraînés sont sensibles aux exemples adversariaux.
d) FGSM avec $\varepsilon = 0{,}03$, sur une image 28 × 28 en niveaux de gris dont aucun pixel n'est ramené par le `clamp` : la somme des variations absolues des pixels, $\sum_j \lvert x_{adv,j} - x_j \rvert$ (2 décimales). On suppose qu'aucune composante du gradient n'est exactement nulle.
e) Une attaque qui cherche à faire prédire une classe **choisie** (par exemple « grille-pain ») est une attaque : (A) non ciblée ; (B) ciblée ; (C) universelle ; (D) physique.

### 21.Q12 — 🕰️ Après VGG16 : ResNet, ViT et modèles pré-entraînés 🧠 ⏱️ 3 min
*Fiche §21.9.1 (encadré 🕰️) · livre §21.9.1 · parcours R*

a) L'idée clé de ResNet : (A) des filtres de 11 × 11 ; (B) les connexions résiduelles ; (C) l'abandon de la rétropropagation ; (D) des couches denses de 4 096 neurones.
b) Un Vision Transformer découpe une image de 224 × 224 en patchs de 16 × 16 pixels : leur nombre.
c) Un pooling moyen global appliqué à un tenseur `(512, 7, 7)` : la longueur du vecteur obtenu.
d) Vrai ou faux : `torchvision.models.resnet18(weights="DEFAULT")` donne un réseau aux poids pré-entraînés sur ImageNet.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 21.R1 — Ch. 20 : ce que changent model.train() et model.eval() pour le dropout 🔁 ★ ⏱️ 5 min
*Ch. 20 (fiche §20.4.3, section PyTorch en pratique) · parcours R*

a) Vrai ou faux : en mode évaluation, `nn.Dropout(0.5)` rend son entrée inchangée.
b) En mode entraînement, `nn.Dropout(0.25)` : le facteur par lequel il multiplie les valeurs qu'il garde (4 décimales).
c) Le convnet MNIST du livre a un `Dropout(0.25)` après son pooling. En mode entraînement, quelle part des valeurs du tenseur `(64, 12, 12)` met-il à zéro, en moyenne, en % ?
d) On oublie `model.eval()` avant de mesurer l'accuracy de validation : (A) elle est mesurée avec le dropout actif, donc bruitée et en général un peu plus basse ; (B) elle est plus haute, car le réseau est en entraînement ; (C) rien ne change ; (D) PyTorch lève une erreur.

### 21.R2 — Ch. 18 : le gradient d'un poids utilisé à plusieurs endroits 🔁 ★ ⏱️ 5 min
*Ch. 18 (fiche, accumulation des gradients) · parcours R, M*

Un même poids $w$ sert à trois sorties : $y_i = w\,x_i$ pour $\mathbf{x} = (1, 2, 3)$.

a) Si la loss est $L = y_1 + y_2 + y_3$ : la valeur de $\partial L / \partial w$.
b) Si l'on connaît $\partial L / \partial y_i = (1, -1, 2)$ : la valeur de $\partial L / \partial w$.
c) Vrai ou faux : dans le micro-autograd `Value` du ch. 18, les gradients d'un nœud utilisé plusieurs fois s'additionnent.

### 21.R3 — 0B : le produit scalaire, ce que calcule un filtre en chaque position 🔁 ★ ⏱️ 5 min
*0B (produit scalaire) · parcours R*

a) Le produit scalaire de $(1, 2, -1)$ et $(3, 0, 2)$.
b) La somme des produits terme à terme des grilles $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ et $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.
c) Vrai ou faux : parmi des vecteurs $\mathbf{x}$ de même norme, le produit scalaire $\mathbf{w} \cdot \mathbf{x}$ est le plus grand quand $\mathbf{x}$ pointe dans la même direction que $\mathbf{w}$.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 21.1 — Le détecteur de jaune à la main ✏️ ★ ⏱️ 10 min
**Objectif :** calculer à la main le score d'un neurone détecteur de couleur, et le relier à une convolution 1 × 1.
**Prérequis :** 0B (somme pondérée, produit scalaire) · fiche §21.2.2 · **Parcours :** R, M

Le neurone du livre a les poids $(+1, +1, -1)$ pour $(R, G, B)$, pas de biais et pas d'activation : son score vaut $R + G - B$. Les composantes vont de 0 à 1. Donne les scores avec 2 décimales.

a) Le score du pixel $(0{,}9 ;\ 0{,}85 ;\ 0{,}1)$.
b) Le score d'un pixel orange $(1 ;\ 0{,}5 ;\ 0)$.
c) Le score d'un pixel blanc $(1 ;\ 1 ;\ 1)$.
d) Le score d'un pixel noir.
e) Lequel de ces quatre pixels est le plus « jaune » pour ce neurone ? (A) celui de a) ; (B) l'orange ; (C) le blanc ; (D) le noir.
f) Le plus grand score possible.
g) On construit de la même façon un détecteur de **magenta** (rouge et bleu sans vert) : le poids du vert.
h) On applique le détecteur de jaune à chaque pixel d'une image en couleur de 4 × 6 pixels : le nombre de scores obtenus.
i) Vrai ou faux : cette opération est une convolution 1 × 1, à 3 canaux d'entrée et 1 canal de sortie.

Dans ta copie : pourquoi le pixel blanc obtient-il moins que le jaune pur, alors qu'il contient autant de rouge et de vert ? Propose des poids pour un détecteur de « gris clair » (trois composantes égales et grandes) ; un seul neurone linéaire, même avec un biais, peut-il donner à un gris clair un score plus grand qu'au blanc **et** qu'au gris moyen ?

### Ex 21.2 — Une corrélation 5×5 par 3×3 à la main ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer à la main la sortie d'une couche de convolution, avec et sans padding, et voir ce que change le retournement du noyau.
**Prérequis :** 0B (produit scalaire) · fiche §21.3, §21.3.1, §21.3.4 · **Parcours :** R, M

$$\mathbf{X} = \begin{pmatrix} 3 & 1 & 0 & 0 & 1 \\ 2 & 0 & 1 & 2 & 0 \\ 4 & 1 & 0 & 1 & 3 \\ 1 & 2 & 2 & 0 & 1 \\ 0 & 3 & 1 & 2 & 2 \end{pmatrix}, \qquad \mathbf{K} = \begin{pmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{pmatrix}.$$

On calcule $\mathbf{Y}$, la corrélation croisée de l'image $\mathbf{X}$ par le noyau $\mathbf{K}$ (ce que calcule `nn.Conv2d`, sans biais), sans padding et avec un stride de 1. $Y_{i,j}$ est la somme des produits de $\mathbf{K}$ et du morceau 3 × 3 de $\mathbf{X}$ dont le coin en haut à gauche est en $(i, j)$.

a) La largeur de $\mathbf{Y}$.
b) $Y_{0,0}$.
c) $Y_{1,1}$.
d) $Y_{2,1}$.
e) La plus petite valeur de $\mathbf{Y}$.
f) Sa position $(i, j)$ (entre guillemets, ligne puis colonne).
g) Avec un zero-padding de 1 (la sortie fait 5 × 5) : la valeur en position $(0, 0)$.
h) Avec ce padding, la valeur en position $(4, 0)$.
i) La **convolution** (au sens des mathématiciens, noyau retourné dans les deux directions), sans padding, en position $(0, 0)$.
j) Vrai ou faux : pour ce noyau, la convolution vaut l'opposé de la corrélation croisée en toute position.

Dans ta copie : ce noyau est un filtre de Sobel. Quel motif de l'image fait les grandes valeurs positives, et lequel les grandes valeurs négatives ? Où se trouvent-ils dans $\mathbf{X}$ ?

### Ex 21.3 — Tailles de sortie : padding, stride et la hiérarchie 600 → 300 → 100 ✏️ ★★ ⏱️ 20 min
**Objectif :** appliquer la formule des tailles de sortie à des convolutions et à des convolutions transposées, et vérifier un exemple du livre.
**Prérequis :** Ex 21.2 · 0B (partie entière) · fiche §21.3.4, §21.3.5, §21.4.2, §21.8 · **Parcours :** R, M

Rappel : $H_{out} = \left\lfloor \frac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1$ pour une convolution, $H_{out} = (H - 1)\,s - 2p + k$ pour une convolution transposée.

Le livre construit une hiérarchie sur une image de 600 × 600 pixels : 8 filtres 5 × 5, padding 2, stride 2 ; puis 4 filtres 3 × 3, padding 2, stride 3.

a) La largeur de sortie du premier étage.
b) La largeur de sortie du second étage, d'après la formule.
c) Le plus petit padding $p \ge 0$ qui donne exactement 100 au second étage (noyau 3, stride 3).
d) Le plus grand padding $p$ qui donne encore 100.
e) Le nombre de canaux que doivent avoir les filtres du second étage.
f) Une image de 28 × 28, un noyau 3 × 3 de dilatation 2, sans padding, stride 1 : la largeur de sortie.
g) Le padding de chaque côté qui garde la taille de 28 avec ce noyau dilaté (stride 1).
h) Une convolution transposée de noyau 3, stride 3, sans padding, sur une entrée de 100 : la largeur de sortie.
i) Une convolution transposée de noyau 4, stride 2, padding 1, sur une entrée de 7 : la largeur de sortie.
j) Une convolution ordinaire de noyau 4, stride 2, padding 1, sur une entrée de 14 : la largeur de sortie.

Dans ta copie : que remarques-tu en comparant i) et j) ? Une convolution transposée est-elle l'inverse d'une convolution ?

### Ex 21.4 — Compter les paramètres : convolution contre couche dense ✏️ ★★ ⏱️ 15 min
**Objectif :** compter les paramètres d'un CNN couche par couche, et comparer une convolution à la couche dense qui ferait le même travail.
**Prérequis :** ch. 16 (paramètres d'une couche dense) · Ex 21.3 · fiche §21.4.1, §21.9 · **Fil rouge :** MNIST · **Parcours :** R, M

Le convnet MNIST du livre (fiche, §21.9) : entrée 1 × 28 × 28 ; convolution de 32 filtres 3 × 3 ; convolution de 64 filtres 3 × 3 ; max pooling 2 × 2 ; dropout ; aplatissement ; dense de 128 neurones ; dropout ; dense de 10 neurones. Chaque convolution et chaque couche dense a un biais par filtre ou par neurone.

a) Le nombre de paramètres de la première convolution.
b) Celui de la deuxième convolution.
c) Celui de la première couche dense.
d) Celui de la dernière couche dense.
e) Le nombre total de paramètres du réseau.
f) La part de la première couche dense dans ce total, en % (1 décimale).
g) Une couche **dense** qui recevrait les 784 pixels et produirait directement les 32 × 26 × 26 valeurs de la première convolution : son nombre de paramètres (biais compris).
h) Le nombre de multiplications que fait la première convolution pour **une** image (une par poids et par position, sans compter les additions).

Dans ta copie : pourquoi la convolution a-t-elle tant de paramètres en moins que la couche dense de g), pour la même forme de sortie ? Qu'est-ce que la couche dense pourrait faire et que la convolution ne peut pas ?

### Ex 21.5 — Champ récepteur effectif d'une pile convolution + pooling ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer le champ récepteur d'une sortie à travers une pile de convolutions et de poolings, et en tirer l'argument des filtres 3 × 3.
**Prérequis :** Ex 21.3 · fiche §21.3.3 (formule du champ récepteur), §21.4.2, §21.9.1 · **Parcours :** M

Rappel : $r_\ell = r_{\ell-1} + (k_\ell - 1)\prod_{i<\ell} s_i$, avec $r_0 = 1$ ; un pooling compte comme une couche. On donne le côté du champ récepteur, en pixels de l'entrée.

a) Deux convolutions 3 × 3 de stride 1 : le côté du champ récepteur d'une sortie.
b) Trois convolutions 3 × 3 de stride 1.
c) Le nombre de poids (sans biais) de trois convolutions 3 × 3 de 64 canaux vers 64 canaux.
d) Le nombre de poids d'une seule convolution de 64 vers 64 canaux qui aurait le champ récepteur de b).
e) Le détecteur de visage du livre : convolution 3 × 3 (stride 1), max pooling 3 × 3 (stride 3), convolution 3 × 3 (stride 1). Le champ récepteur d'une sortie de la deuxième convolution (le padding ne le change pas).
f) Il continue : max pooling 3 × 3 (stride 3), puis convolution 3 × 3. Le champ récepteur d'une sortie de cette troisième convolution.
g) La hiérarchie 600 → 300 → 100 : convolution 5 × 5 de stride 2, puis convolution 3 × 3 de stride 3. Le champ récepteur d'une sortie du second étage.
h) Le convnet MNIST : convolution 3 × 3, convolution 3 × 3, max pooling 2 × 2. Le champ récepteur d'une valeur à la sortie du pooling.

Dans ta copie : le livre dit qu'un filtre du niveau 2 de son détecteur « répond à une région de 9 × 9 » ; d'où vient l'écart avec e) ? Que vaut f) comparé à la taille de l'image (27 × 27), et est-ce gênant ?

### Ex 21.6 — Convolution transposée à la main : insérer des zéros puis corréler ✏️ ★★★ ⏱️ 30 min
**Objectif :** calculer à la main une convolution transposée de stride 2, par la vue « tampon » et par l'insertion de zéros.
**Prérequis :** Ex 21.2 · Ex 21.3 · fiche §21.8 · **Parcours :** M

$$\mathbf{x} = \begin{pmatrix} 2 & -1 \\ 1 & 3 \end{pmatrix}, \qquad \mathbf{w} = \begin{pmatrix} 1 & 0 & 2 \\ -1 & 3 & 0 \\ 2 & 1 & -1 \end{pmatrix}.$$

On calcule la convolution transposée de $\mathbf{x}$ par $\mathbf{w}$ avec un stride de 2, sans padding (`F.conv_transpose2d` de PyTorch, un canal) : chaque pixel $x_{i,j}$ dépose dans la sortie $\mathbf{y}$ le tampon $x_{i,j}\,\mathbf{w}$, avec son coin en haut à gauche en $(2i, 2j)$, et les tampons qui se chevauchent s'additionnent.

a) La largeur de $\mathbf{y}$.
b) Le nombre de zéros à insérer entre deux pixels voisins de $\mathbf{x}$ pour calculer la même chose par une convolution ordinaire.
c) Le nombre de zéros à ajouter de chaque côté de cette entrée « étirée », avant de la corréler avec le noyau retourné (formule $k - 1 - p$).
d) $y_{0,0}$.
e) $y_{1,1}$.
f) $y_{0,2}$.
g) $y_{2,2}$.
h) La somme de toutes les valeurs de $\mathbf{y}$.
i) Avec `padding=1` (on retire une rangée et une colonne de chaque bord) : la largeur de la sortie.
j) Avec `padding=1` : la valeur au centre de la sortie.

Dans ta copie : vérifie $y_{0,2}$ par la seconde méthode (insérer les zéros, compléter, retourner le noyau, corréler). Combien de tampons recouvrent la case $(2, 2)$, et combien la case $(1, 1)$ ? Qu'est-ce que cela annonce pour une image générée avec ce noyau et ce stride ?

### Ex 21.7 — Rétropropager dans une convolution 1D : la somme sur les positions ∂ ★★★ ⏱️ 35 min
**Objectif :** dériver les gradients d'une convolution par rapport à ses poids, à son biais et à son entrée, et reconnaître une convolution transposée.
**Prérequis :** ch. 18 (rétropropagation, gradient d'un nœud utilisé plusieurs fois) · Ex 21.2 · fiche, section « Au-delà du livre : la rétropropagation d'une convolution » · **Parcours :** M

Une entrée $x_0, \dots, x_{n-1}$, un noyau $w_0, \dots, w_{k-1}$, un biais $b$, sans padding et avec un stride de 1 : $y_i = b + \sum_{u=0}^{k-1} w_u\,x_{i+u}$ pour $i = 0, \dots, m - 1$. On connaît $g_i = \partial L / \partial y_i$. Réponds dans ta copie.

1. Que vaut $m$ en fonction de $n$ et $k$ ? Combien de sorties dépendent du poids $w_u$ ? Montre que $\frac{\partial L}{\partial w_u} = \sum_{i=0}^{m-1} g_i\,x_{i+u}$ et que $\frac{\partial L}{\partial b} = \sum_i g_i$. Quelle opération reconnais-tu dans la première formule ?
2. Pour un pixel $x_j$, quelles sorties $y_i$ en dépendent, et avec quel coefficient ? Montre que $\frac{\partial L}{\partial x_j} = \sum_i g_i\,w_{j-i}$, la somme portant sur les $i$ tels que $0 \le i \le m - 1$ et $0 \le j - i \le k - 1$. Combien de termes a la somme pour $x_0$ ? Pour un pixel du milieu ?
3. Montre que, si l'on complète $g$ de $k - 1$ zéros de chaque côté, $\partial L / \partial \mathbf{x}$ est la corrélation croisée de ce $g$ complété par le noyau **retourné** $(w_{k-1}, \dots, w_0)$, c'est-à-dire une convolution « pleine » de $g$ par $w$. Vérifie que la taille obtenue est bien $n$.
4. Application : $\mathbf{x} = (1, 2, -1, 3)$, $\mathbf{w} = (2, -1)$, $b = 0$ et $\mathbf{g} = (1, 0, 2)$. Calcule $\mathbf{y}$, puis $\partial L / \partial \mathbf{w}$, $\partial L / \partial b$ et $\partial L / \partial \mathbf{x}$ par les formules. Vérifie un des gradients de poids par une différence finie centrée à la main, pour la loss linéaire $L = \sum_i g_i\,y_i$.
5. Montre que la formule de la question 2 est exactement celle d'une convolution **transposée** de $g$ par $w$ (de stride 1), au sens de la vue « tampon » de la fiche. Avec un stride $s$ dans la passe avant, que faut-il faire de $g$ avant la convolution pleine ?
6. Combien de multiplications coûte chacune des trois formules (poids, biais, entrée), comparé à la passe avant ? Retrouve l'ordre de grandeur du ch. 18 (une passe arrière coûte environ deux passes avant).
7. En deux dimensions avec des canaux, $y_{o,i,j} = b_o + \sum_{c,u,v} W_{o,c,u,v}\,x_{c,i+u,j+v}$. Écris $\partial L / \partial W_{o,c,u,v}$ et $\partial L / \partial x_{c,p,q}$ sans démonstration. Sur quoi somme-t-on en plus pour un mini-batch de $N$ images ?

### Ex 21.8 — Fermi : paramètres, mémoire et calcul de VGG16 🧮 ★★ ⏱️ 25 min
**Objectif :** estimer le calcul et la mémoire d'une passe avant d'un grand CNN, et voir qu'ils ne se trouvent pas là où sont les paramètres.
**Prérequis :** Ex 21.4 · ch. 20 (✏️ 20.6, les paramètres de VGG16) · fiche §21.9.1 · **Parcours :** R, M

VGG16 (fiche, §21.9.1) : entrée 3 × 224 × 224 ; cinq blocs de convolutions 3 × 3 de padding 1 (2, 2, 3, 3 et 3 convolutions de 64, 128, 256, 512 et 512 filtres), chacun suivi d'un max pooling 2 × 2 ; puis trois couches denses (25 088 → 4 096 → 4 096 → 1 000). On compte une **multiplication-addition** par poids et par position de sortie, sans les biais.

a) Les multiplications-additions de la première convolution (3 → 64 canaux, sortie 224 × 224), en millions (1 décimale).
b) Celles de la deuxième convolution (64 → 64 canaux, sortie 224 × 224), en milliards (2 décimales).
c) Celles des 13 convolutions ensemble, en milliards (1 décimale).
d) La part des trois couches denses dans le calcul total (convolutions et couches denses), en % (1 décimale).
e) La mémoire des sorties des 13 convolutions pour **une** image, en `float32` (4 octets), en Mo de $10^6$ octets (1 décimale).
f) La même mémoire pour un mini-batch de 64 images, en Go de $10^9$ octets (2 décimales).
g) Vrai ou faux : les couches denses ont l'essentiel des paramètres de VGG16 (✏️ 20.6), mais une petite part de son calcul.

Dans ta copie : qu'oublient ces estimations, côté mémoire (pendant l'entraînement) et côté temps ? D'après b), pourquoi les architectures modernes réduisent-elles l'image dès la première couche (ResNet commence par une convolution 7 × 7 de stride 2, suivie d'un max pooling) ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 21.9 — Expliquer la convolution en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer ce que fait une couche de convolution à quelqu'un qui n'en a jamais vu.
**Prérequis :** fiche §21.3, §21.3.1 · **Parcours :** R

Dans ta copie, en 5 lignes au plus et sans formule, explique à un ami ce que fait une couche de convolution sur une photo, et pourquoi c'est une meilleure idée qu'une couche dense pour des images. Utilise une image du quotidien (un pochoir, une loupe, un tampon…). Puis relis-toi : as-tu dit que les mêmes poids servent partout ? Que les filtres sont appris ? Où ton image s'arrête-t-elle ?

### Ex 21.10 — Deviner l'âge et le genre sur une photo : jusqu'où aller ? ⚖️ ★★ ⏱️ 20 min
**Objectif :** analyser les risques d'un système de vision qui infère des caractéristiques personnelles à partir d'un visage.
**Prérequis :** ch. 3 (matrice de confusion, taux d'erreur par groupe) · fiche §21.1, §21.10 · **Parcours :** R

Le livre cite, parmi les succès des CNN, un réseau qui estime l'âge et le genre d'une personne à partir de sa photo (p. 930). Une chaîne de magasins veut installer des caméras qui estiment l'âge et le genre de chaque client, pour adapter les publicités affichées et compter sa clientèle. Réponds dans ta copie.

1. Qu'est-ce que le système apprend vraiment : le genre d'une personne, ou l'apparence que les annotateurs du dataset ont associée à un genre ? Qui ce système risque-t-il de mal classer, et avec quelles conséquences pour ces personnes ?
2. Comment mesurerais-tu, avec une matrice de confusion par groupe (ch. 3), si le système se trompe plus souvent pour certaines personnes (selon l'âge, la couleur de peau, le port de lunettes ou d'un voile) ? Quelles données te faudrait-il, et lesquelles serait-il délicat de collecter ?
3. Les clients n'ont rien demandé. Quelles questions de consentement et d'information se posent ? Un client peut-il savoir qu'il est analysé, et refuser ?
4. Le système est-il « attaquable » ? Explique, à partir du §21.10, comment une personne pourrait volontairement tromper la caméra, et pourquoi cela pose aussi un problème de fiabilité pour l'entreprise.
5. Le droit et les pratiques : cherche (ou rappelle-toi) au moins un fournisseur de service de vision qui a retiré l'inférence du genre ou des émotions de son offre, et ce que le règlement européen sur l'IA dit de la catégorisation biométrique et de la reconnaissance des émotions. Recommanderais-tu ce projet ? Sous quelle forme, s'il en reste une acceptable (par exemple un simple comptage anonyme, sans inférence sur les personnes) ?

### Ex 21.11 — Lire LeCun et coll. (1998) : LeNet-5 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a fixé l'architecture d'un CNN, et retrouver dans LeNet-5 les idées du chapitre.
**Prérequis :** Ex 21.4 · fiche §21.3 à §21.9 · **Fil rouge :** MNIST · **Parcours :** M

Lis Y. LeCun, L. Bottou, Y. Bengio et P. Haffner, « Gradient-Based Learning Applied to Document Recognition », *Proceedings of the IEEE*, 86(11), 1998 (disponible sur la page de L. Bottou : [leon.bottou.org/publications/pdf/ieee-1998.pdf](https://leon.bottou.org/publications/pdf/ieee-1998.pdf)) : l'introduction (§I, jusqu'à I.B), la section II (son introduction, puis II.A et II.B : les réseaux convolutifs, LeNet-5) et la section III (les résultats sur MNIST). Réponds dans ta copie.

1. Quelles trois « idées architecturales » les auteurs donnent-ils aux réseaux convolutifs (§II.A) ? Relie chacune à une section de la fiche.
2. Pourquoi, selon eux, une couche dense sur les pixels bruts est-elle un mauvais choix pour des images (deux raisons) ?
3. Décris l'architecture de LeNet-5 (figure 2) couche par couche : C1, S2, C3, S4, C5, F6 et la sortie, avec les tailles de leurs cartes. Pourquoi l'entrée fait-elle 32 × 32 alors que les chiffres de MNIST font 28 × 28 ?
4. La couche C1 : vérifie avec la formule de la fiche le nombre de ses paramètres (6 filtres 5 × 5) et la taille de ses cartes.
5. Les couches de sous-échantillonnage S2 et S4 ne sont pas un max pooling : que calculent-elles, et combien de paramètres ont-elles ?
6. C3 ne relie pas chaque carte à toutes les cartes de S2 (tableau 1). Pourquoi les auteurs font-ils ce choix ? Quel paramètre de `nn.Conv2d` fait aujourd'hui quelque chose de comparable, en plus régulier ?
7. Quel taux d'erreur LeNet-5 obtient-il sur le test de MNIST, sans et avec des images déformées artificiellement (§III) ? Compare avec le convnet du livre (figure 21.49).
8. Bilan : qu'est-ce qui, dans LeNet-5, a disparu des CNN actuels (sorties, activations, sous-échantillonnage), et qu'est-ce qui est resté ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 21.E1 — Pourquoi un CNN plutôt qu'un MLP pour des images ? 💼 ★★ ⏱️ 10 min
*Fiche §21.2.3, §21.3.3 · parcours R*

« Pourquoi utilise-t-on des réseaux convolutifs plutôt que des réseaux denses pour des images ? »

### 21.E2 — Taille de sortie et paramètres d'une Conv2d(3, 64, 3, stride=2, padding=1) sur une image 224×224 💼 ★★ ⏱️ 10 min
*Fiche §21.3.4, §21.3.5, §21.7 · parcours R*

« J'ai une couche `nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)` et des images de 224 × 224. Quelle est la forme de la sortie pour un mini-batch de 32 images, et combien de paramètres a la couche ? »

### 21.E3 — À quoi sert une convolution 1×1 ? 💼 ★★ ⏱️ 10 min
*Fiche §21.6 · parcours R*

« À quoi sert une convolution 1 × 1 ? Elle ne regarde qu'un pixel : qu'est-ce qu'elle peut bien apprendre ? »

### 21.E4 — Exemples adversariaux : quels risques, quelles défenses ? 💼 ★★ ⏱️ 10 min
*Fiche §21.10, section sur l'attaque FGSM · parcours R*

« Qu'est-ce qu'un exemple adversarial ? Si vous déployez un modèle de vision, comment évaluez-vous et réduisez-vous ce risque ? »

### 21.E5 — CNN ou Vision Transformer pour un nouveau projet de vision ? 💼 ★★ ⏱️ 10 min
*Fiche §21.9.1 (encadré 🕰️) · parcours R*

« On démarre un projet de classification d'images industrielles, avec 5 000 images étiquetées. Vous partez sur un CNN ou sur un Vision Transformer ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 21.12 | conv_output_size et pad2d | 🔨 | ★ | 15 |
| 21.13 | Prédire où une image CIFAR-10 « jaunit » | 🔮 | ★ | 10 |
| 21.14 | im2col : la vue « œil de mouche » | 🔨 | ★★ | 30 |
| 21.15 | conv2d : stride, padding, dilatation et groupes | 🔨 | ★★★ | 90 |
| 21.16 | Tests pytest paramétrés pour conv2d | 🛠️ | ★★ | 20 |
| 21.17 | Reproduire la chasse aux rayures verticales (Fig. 21.14) | 🎨 | ★★ | 20 |
| 21.18 | max_pool2d et avg_pool2d | 🔨 | ★★ | 25 |
| 21.19 | Flou, Sobel, netteté : des filtres faits main avec F.conv2d | 📦 | ★★ | 20 |
| 21.20 | Conv1d, noyau pleine largeur et 1×1 : prédire formes et paramètres | 🔮 | ★★ | 15 |
| 21.21 | Le détecteur de visage hiérarchique (Fig. 21.18 à 21.23) | 🎨 | ★★★ | 45 |
| 21.22 | Le convnet MNIST du livre en PyTorch, sur Fashion-MNIST | 📦 | ★★★ | 45 |
| 21.23 | Diagnostiquer les courbes et la matrice de confusion du CNN | 📈 | ★★ | 15 |
| 21.24 | Le CNN dont les formes ne passent pas | 🐛 | ★★ | 20 |
| 21.25 | Briser la symétrie : filtres identiques, aléatoires ou He | 🔬 | ★★ | 25 |
| 21.26 | Stride ou pooling : accuracy et temps sur CIFAR-10 | 🔬 | ★★★ | 40 |
| 21.27 | Filtres et cartes d'activation d'un ResNet-18 pré-entraîné | 📦 | ★★ | 30 |
| 21.28 | Ce qu'un filtre cherche : montée de gradient sur l'entrée | 📦 | ★★★ | 45 |
| 21.29 | Attaque FGSM contre ton CNN Fashion-MNIST | 📦 | ★★★ | 40 |
| 21.30 | conv_transpose2d par insertion de zéros | 🔨 | ★★★ | 50 |
| 21.31 | conv2d_backward : rétropropager à travers une convolution | 🔨 | ★★★★ | 100 |
| 21.32 | Défi Fashion-MNIST : au moins 92 % avec moins de 100 000 paramètres | 🏆 | ★★★★ | 120 |

🚀 *GPU Colab conseillé* pour le mode complet (`FAST_MODE = False`) de 21.26 ; le mode rapide tourne sur CPU.
