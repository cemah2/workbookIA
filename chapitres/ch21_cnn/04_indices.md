# 21 · Réseaux convolutifs (CNN) — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 21.Q1 — Les deux sens de « profondeur »

<details><summary>Indice 1</summary>

Relis le §21.2.1 de la fiche et l'encadré « Le livre, corrigé » du §21.2.

</details>
<details><summary>Indice 2</summary>

Pour un tenseur, la profondeur est son nombre de canaux ; pour un réseau, son nombre de couches. Pour d), compte les valeurs d'une image : combien par pixel, combien de pixels ?

</details>
<details><summary>Indice 3</summary>

a) 3 : un canal par couleur primaire. Pour b), lis le premier nombre de la forme `(C, H, W)` ; pour c), demande-toi ce que compte le « 16 » ; pour d), multiplie canaux, hauteur et largeur ; pour e), rappelle-toi de quoi est fait le convnet MNIST du livre.

</details>

### 21.Q2 — Un seul neurone promené sur toute l'image

<details><summary>Indice 1</summary>

Relis les §21.2.2 à §21.2.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un filtre a un poids par case de son noyau et par canal d'entrée ; l'image ne change que le nombre de positions où on le pose. Pour d) et e), relis la définition de l'ancre.

</details>
<details><summary>Indice 3</summary>

a) Vrai : le partage de poids, c'est le même neurone partout. Pour b), compte les cases d'un noyau 5 × 5 sur un seul canal ; pour c), demande-toi ce qui change entre une petite et une grande image ; pour d) et e), cherche le mot qui désigne le pixel sous l'ancre, et ce qu'un côté impair a de plus qu'un côté pair.

</details>

### 21.Q3 — Convolution ou corrélation croisée ?

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 et l'encadré 🕰️ du §21.3, puis le §21.3.2.

</details>
<details><summary>Indice 2</summary>

La corrélation croisée multiplie la première fenêtre $(x_0, x_1, x_2)$ par $(w_0, w_1, w_2)$ ; la convolution la multiplie par le noyau retourné $(w_2, w_1, w_0)$. Pour e) et f), une image $H \times H$ a $(H - k + 1)^2$ morceaux $k \times k$ entiers.

</details>
<details><summary>Indice 3</summary>

a) Faux : la documentation de PyTorch parle de corrélation croisée. Pour b), calcule $1 \times 1 + 2 \times 0 + 3 \times (-1)$ ; pour c), refais-le avec $(-1, 0, 1)$ ; pour d), retourne $(1, 2, 1)$ ; pour e), applique $(H - k + 1)^2$ ; pour f), multiplie les $k^2$ valeurs par le nombre de canaux.

</details>

### 21.Q4 — Détecteurs de motifs et hiérarchie de filtres

<details><summary>Indice 1</summary>

Relis les §21.3.1 et §21.3.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un pixel noir (0) ne compte pas ; un pixel blanc (1) ajoute son poids, $+1$ ou $-1$. Pour d), demande-toi ce qu'il s'est passé entre la première couche et la couche haute.

</details>
<details><summary>Indice 3</summary>

a) 3 : les trois blancs de la colonne de gauche, et rien de blanc sous les $-1$. Pour b), ajoute la contribution du pixel blanc du centre ; pour c), demande-toi qui règle les poids ; pour d), pense au pooling ; pour e), relis le sens que le chapitre donne à « feature ».

</details>

### 21.Q5 — Padding et stride : que devient la taille ?

<details><summary>Indice 1</summary>

Relis les §21.3.4 et §21.3.5 de la fiche et leur figure.

</details>
<details><summary>Indice 2</summary>

$H_{out} = \lfloor (H + 2p - k)/s \rfloor + 1$, direction par direction : le stride vertical pour la hauteur, l'horizontal pour la largeur. Le padding qui garde la taille (stride 1) vaut $(k - 1)/2$.

</details>
<details><summary>Indice 3</summary>

a) $12 - 5 + 1 = 8$. Pour b), applique $(k - 1)/2$ ; pour c), calcule $\lfloor (7 - 3)/2 \rfloor + 1$ ; pour d), la même formule avec la largeur et le stride horizontal ; pour e), demande-toi si des fenêtres de 3 placées tous les 3 pixels se recouvrent ou laissent des trous.

</details>

### 21.Q6 — Canaux d'entrée, nombre de filtres, canaux de sortie

<details><summary>Indice 1</summary>

Relis les §21.4, §21.4.1 et §21.4.2 de la fiche (le paragraphe sur la convolution par canal est au §21.4).

</details>
<details><summary>Indice 2</summary>

Un filtre a la profondeur de son entrée ; une couche a un canal de sortie par filtre ; poids $= C_{out} \times C_{in} \times k^2$, plus un biais par filtre. Une convolution par canal a des filtres d'un seul canal.

</details>
<details><summary>Indice 3</summary>

a) 6 : la profondeur de l'entrée. Pour b), compte les filtres ; pour c), $4 \times 6 \times 9$ ; pour d), ajoute un biais par filtre ; pour e), compte 6 filtres de $1 \times 3 \times 3$ poids.

</details>

### 21.Q7 — Convolution 1D ou convolution 1×1 ?

<details><summary>Indice 1</summary>

Relis les §21.5 et §21.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le filtre 1D couvre 3 lignes entières de la matrice (3 mots de 50 nombres) et glisse le long des mots. Une convolution 1 × 1 a un poids par couple (canal d'entrée, canal de sortie) et garde la taille de l'image.

</details>
<details><summary>Indice 3</summary>

a) $3 \times 50 = 150$. Pour b), applique $n - k + 1$ aux mots ; pour c), multiplie les deux nombres de canaux ; pour d), demande-toi si un noyau 1 × 1 lit un voisin ; pour e), rappelle-toi ce qu'une convolution 1 × 1 fait de la hauteur.

</details>

### 21.Q8 — Couche de convolution et initialisation des filtres

<details><summary>Indice 1</summary>

Relis les §21.7 et §21.7.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Des neurones qui partent des mêmes poids reçoivent les mêmes gradients. Le fan-in compte toutes les valeurs que lit un neurone : $C_{in} \times k_h \times k_w$.

</details>
<details><summary>Indice 3</summary>

a) Vrai : rien ne distingue des filtres égaux, ils restent égaux. Pour b), calcule $64 \times 9$ ; pour c), prends la racine de $2/576$ ; pour d), cherche le nom de la sortie d'un filtre ; pour e), regarde la valeur par défaut de `padding`.

</details>

### 21.Q9 — Convolution transposée : les mots piégés

<details><summary>Indice 1</summary>

Relis le §21.8 de la fiche, son encadré « Le livre, corrigé » et sa figure.

</details>
<details><summary>Indice 2</summary>

Convolution transposée : $H_{out} = (H - 1)\,s - 2p + k$, et $s - 1$ zéros entre deux pixels. Dilatation : le noyau couvre $d(k - 1) + 1$ pixels.

</details>
<details><summary>Indice 3</summary>

a) Faux : la dilatation espace les poids du noyau, pas les pixels de l'entrée. Pour b), calcule $3 \times 2 + 3$ ; pour c), retire $2p$ ; pour d), applique $s - 1$ ; pour e), regarde la figure des chevauchements ; pour f), applique $d(k - 1) + 1$.

</details>

### 21.Q10 — VGG16 et ce que voient ses filtres

<details><summary>Indice 1</summary>

Relis les §21.9.1 à §21.9.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Chaque bloc de VGG16 finit par un pooling qui divise le côté par 2 ; ses blocs ont 64, 128, 256, 512 et 512 filtres. La visualisation des filtres modifie l'image, pas le réseau.

</details>
<details><summary>Indice 3</summary>

a) Après trois blocs, $224/2^3 = 28$ de côté et les 256 canaux du bloc 3. Pour b), compte les blocs ; pour c), compte les poolings **avant** le bloc 5 ; pour d), demande-toi ce qui reste fixé ; pour e), relis ce que montrent les filtres de `block1` et de `block4`.

</details>

### 21.Q11 — Exemples adversariaux : vrai ou faux ?

<details><summary>Indice 1</summary>

Relis le §21.10 de la fiche et la section sur l'attaque FGSM.

</details>
<details><summary>Indice 2</summary>

Avec FGSM, chaque pixel bouge de $+\varepsilon$, $-\varepsilon$ ou 0, selon le signe du gradient. Pour e), distingue « n'importe quelle erreur » et « une classe choisie ».

</details>
<details><summary>Indice 3</summary>

a) Faux : l'image attaquée paraît identique. Pour b), relis la définition d'une perturbation universelle ; pour c), relis quels réseaux le livre attaque au §21.10 ; pour d), chacun des 784 pixels bouge d'exactement $\varepsilon$ : additionne ; pour e), cherche le mot qui désigne une classe choisie.

</details>

### 21.Q12 — 🕰️ Après VGG16 : ResNet, ViT et modèles pré-entraînés

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ « après VGG16 » du §21.9.1.

</details>
<details><summary>Indice 2</summary>

Un ViT découpe l'image en une grille de patchs ; un pooling moyen global fait la moyenne de chaque canal sur toutes ses positions.

</details>
<details><summary>Indice 3</summary>

a) B : les connexions résiduelles (ch. 20). Pour b), compte les patchs par côté, $224/16$, puis mets au carré ; pour c), demande-toi combien de valeurs il reste par canal ; pour d), relis ce que télécharge `weights="DEFAULT"`.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 21.R1 — Ch. 20 : ce que changent model.train() et model.eval() pour le dropout

<details><summary>Indice 1</summary>

Relis le dropout inversé du ch. 20 (fiche §20.4.3) et la section PyTorch en pratique.

</details>
<details><summary>Indice 2</summary>

En entraînement, une valeur est coupée avec la probabilité $p$, et les autres sont divisées par $1 - p$ ; en évaluation, rien n'est coupé.

</details>
<details><summary>Indice 3</summary>

a) Vrai : en évaluation, la couche est l'identité. Pour b), calcule $1/(1 - 0{,}25)$ ; pour c), relis ce que représente $p$ ; pour d), demande-toi dans quel mode le réseau mesure alors son accuracy.

</details>

### 21.R2 — Ch. 18 : le gradient d'un poids utilisé à plusieurs endroits

<details><summary>Indice 1</summary>

Relis, dans la fiche du ch. 18, l'accumulation des gradients d'un nœud utilisé plusieurs fois.

</details>
<details><summary>Indice 2</summary>

$\partial y_i/\partial w = x_i$, et $\partial L/\partial w = \sum_i (\partial L/\partial y_i)\,x_i$.

</details>
<details><summary>Indice 3</summary>

a) Ici $\partial L/\partial y_i = 1$ : $\partial L/\partial w = 1 + 2 + 3$. Pour b), pondère chaque $x_i$ par le $\partial L/\partial y_i$ donné ; pour c), relis comment `backward` met à jour `grad`.

</details>

### 21.R3 — 0B : le produit scalaire, ce que calcule un filtre en chaque position

<details><summary>Indice 1</summary>

Relis le produit scalaire en 0B.

</details>
<details><summary>Indice 2</summary>

Multiplie terme à terme, puis additionne ; pour deux grilles, case par case. Pour c), écris $\mathbf{w} \cdot \mathbf{x}$ avec un cosinus.

</details>
<details><summary>Indice 3</summary>

a) $3 + 0 - 2 = 1$. Pour b), seules les cases où la seconde grille vaut 1 comptent ; pour c), demande-toi pour quel angle $\cos\theta$ est le plus grand.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 21.1 — Le détecteur de jaune à la main

<details><summary>Indice 1</summary>

Relis le §21.2.2 de la fiche (le pixel cyan) et le §21.6.

</details>
<details><summary>Indice 2</summary>

Le score vaut $R + G - B$ : calcule-le pour chaque pixel. Pour f), cherche les valeurs de $R$, $G$, $B$ qui le rendent le plus grand ; pour g), demande-toi quelle couleur le magenta ne contient pas.

</details>
<details><summary>Indice 3</summary>

a) $0{,}9 + 0{,}85 - 0{,}1 = 1{,}65$. Pour b) à d), même calcul ; pour e), compare les quatre scores ; pour f), mets à 1 ce qui a un poids $+1$ et à 0 ce qui a un poids $-1$ ; pour g), demande-toi quel signe donner au poids de la couleur que le magenta ne contient pas ; pour h), un score par pixel ; pour i), relis le §21.6.

</details>

### Ex 21.2 — Une corrélation 5×5 par 3×3 à la main

<details><summary>Indice 1</summary>

Relis les §21.3, §21.3.1 et §21.3.4 de la fiche, et l'encadré 🧮 sur corrélation et convolution.

</details>
<details><summary>Indice 2</summary>

Le noyau ignore sa colonne du milieu : $Y_{i,j}$ = (colonne de gauche du morceau, pondérée par $1, 2, 1$) − (colonne de droite, pondérée de même). Avec le padding, la position $(i, j)$ est centrée sur $X_{i,j}$. Pour i), retourne $\mathbf{K}$ dans les deux directions et compare-le à $\mathbf{K}$.

</details>
<details><summary>Indice 3</summary>

a) $5 - 3 + 1 = 3$. Pour b), le morceau des lignes 0 à 2 et colonnes 0 à 2 a pour colonne de gauche $(3, 2, 4)$ et pour colonne de droite $(0, 1, 0)$ : applique la règle ; pour c) et d), même règle sur les morceaux qui commencent en $(1, 1)$ et $(2, 1)$ ; pour e) et f), calcule les 9 valeurs ; pour g) et h), une colonne de la fenêtre est faite de zéros ; pour i) et j), écris le noyau retourné.

</details>

### Ex 21.3 — Tailles de sortie : padding, stride et la hiérarchie 600 → 300 → 100

<details><summary>Indice 1</summary>

Relis les §21.3.4, §21.3.5, §21.4.2 et §21.8 de la fiche.

</details>
<details><summary>Indice 2</summary>

Applique les deux formules du rappel, en prenant la partie entière **avant** le $+1$. Pour c) et d), essaie $p = 0, 1, 2$. Pour f) et g), remplace $k$ par l'étendue du noyau dilaté, $d(k - 1) + 1$.

</details>
<details><summary>Indice 3</summary>

a) $\lfloor (600 + 4 - 5)/2 \rfloor + 1 = 299 + 1 = 300$. Pour b), refais le calcul avec $300$, $k = 3$, $s = 3$, $p = 2$ ; pour c) et d), teste chaque padding ; pour e), demande-toi d'où vient l'entrée du second étage ; pour f) et g), utilise une étendue de 5 ; pour h) et i), applique $(H - 1)\,s - 2p + k$ ; pour j), la formule de la convolution.

</details>

### Ex 21.4 — Compter les paramètres : convolution contre couche dense

<details><summary>Indice 1</summary>

Relis les §21.4.1 et §21.9 de la fiche (le tableau du convnet MNIST).

</details>
<details><summary>Indice 2</summary>

Convolution : $C_{out} \times (C_{in}\,k^2 + 1)$. Couche dense : entrées × sorties + sorties. Suis les formes : $28 \to 26 \to 24 \to 12$, et 64 canaux après la deuxième convolution.

</details>
<details><summary>Indice 3</summary>

a) $32 \times (9 + 1) = 320$. Pour b), chaque filtre lit 32 canaux ; pour c), l'entrée est l'aplatissement de $64 \times 12 \times 12$ ; pour d), $128 \times 10 + 10$ ; pour e), additionne ; pour f), divise c) par e) ; pour g), relie les 784 pixels aux $32 \times 26 \times 26$ sorties ; pour h), 9 multiplications par position et par filtre.

</details>

### Ex 21.5 — Champ récepteur effectif d'une pile convolution + pooling

<details><summary>Indice 1</summary>

Relis la formule du champ récepteur au §21.3.3 de la fiche, son exemple et sa figure.

</details>
<details><summary>Indice 2</summary>

Tiens deux nombres à jour : $r$ (le champ) et $J$ (le produit des strides déjà franchis). Chaque couche fait $r \leftarrow r + (k - 1)\,J$, **puis** $J \leftarrow J\,s$. Un pooling est une couche comme les autres.

</details>
<details><summary>Indice 3</summary>

a) $1 + 2 + 2 = 5$. Pour b), une couche de plus ; pour c) et d), $C_{out} \times C_{in} \times k^2$, avec pour d) un noyau dont le côté est le champ récepteur trouvé en b) ; pour e) à g), suis $r$ et $J$ couche par couche ; pour h), le pooling 2 × 2 ajoute $(2 - 1)\,J$.

</details>

### Ex 21.6 — Convolution transposée à la main : insérer des zéros puis corréler

<details><summary>Indice 1</summary>

Relis le §21.8 de la fiche : la recette et la vue « tampon ».

</details>
<details><summary>Indice 2</summary>

Les quatre tampons $x_{i,j}\,\mathbf{w}$ ont leurs coins en $(0, 0)$, $(0, 2)$, $(2, 0)$ et $(2, 2)$. Pour chaque case demandée, cherche quels tampons la recouvrent, et quelle case de $\mathbf{w}$ chacun y dépose : la case $(a, b)$ reçoit $x_{i,j}\,w_{a - 2i,\,b - 2j}$.

</details>
<details><summary>Indice 3</summary>

a) $(2 - 1) \times 2 + 3 = 5$. Pour b) et c), applique $s - 1$ et $k - 1 - p$ ; pour d), seul $x_{0,0}$ couvre $(0, 0)$ ; pour e), cherche les tampons qui couvrent $(1, 1)$ ; pour f) et g), cherche de même tous les tampons qui couvrent la case ; pour h), chaque tampon a pour somme $x_{i,j} \times \sum \mathbf{w}$ ; pour i) et j), retire une rangée et une colonne de chaque bord, puis repère le centre.

</details>

### Ex 21.7 — Rétropropager dans une convolution 1D : la somme sur les positions

<details><summary>Indice 1</summary>

Relis la section « Au-delà du livre : la rétropropagation d'une convolution » de la fiche, puis la règle de la chaîne du ch. 18.

</details>
<details><summary>Indice 2</summary>

Écris $\partial L/\partial w_u = \sum_i g_i\,\partial y_i/\partial w_u$ et $\partial L/\partial x_j = \sum_i g_i\,\partial y_i/\partial x_j$, puis cherche quels $\partial y_i/\partial x_j$ sont non nuls. Pour la question 3, pose $i = j + u - (k - 1)$. Pour la question 5, écris ce que dépose chaque $g_i$ dans la vue « tampon ».

</details>
<details><summary>Indice 3</summary>

Question 1 : $y_i$ contient le terme $w_u\,x_{i+u}$ pour **chaque** $i$, donc $\partial y_i/\partial w_u = x_{i+u}$ et $\partial L/\partial w_u = \sum_i g_i\,x_{i+u}$, une somme de $m = n - k + 1$ termes. Pour la question 2, $x_j$ apparaît dans $y_i$ quand $j - i$ est un indice du noyau ; pour la question 3, vérifie que les zéros du padding éliminent les $i$ hors de $0..m-1$ ; pour la question 4, calcule d'abord $y_0 = w_0 x_0 + w_1 x_1$ et ses deux voisins, puis applique les formules ; pour la question 6, compte les produits de chaque formule ; pour la question 7, ajoute les sommes sur les canaux et les deux directions.

</details>

### Ex 21.8 — Fermi : paramètres, mémoire et calcul de VGG16

<details><summary>Indice 1</summary>

Relis le §21.9.1 de la fiche et l'✏️ 20.6 (les paramètres de VGG16).

</details>
<details><summary>Indice 2</summary>

Une convolution coûte $H_{out} \times W_{out} \times C_{out} \times C_{in} \times 9$ multiplications-additions ; une couche dense, une par poids. Les côtés sont 224, 112, 56, 28 et 14 dans les cinq blocs ; la mémoire des sorties d'une convolution vaut $C_{out} \times H \times W$ valeurs.

</details>
<details><summary>Indice 3</summary>

a) $224^2 \times 64 \times 27 = 86\,704\,128$, soit 86,7 millions. Pour b), la même chose avec 64 canaux d'entrée ; pour c), additionne les 13 convolutions, bloc par bloc ; pour d), compte les poids des trois couches denses et compare ; pour e) et f), additionne les $C_{out}\,H\,W$ des 13 convolutions, fois 4 octets, puis fois 64 ; pour g), compare d) à la part des paramètres.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 21.9 — Expliquer la convolution en cinq lignes

<details><summary>Indice 1</summary>

Relis les §21.3 et §21.3.1 de la fiche, puis l'✏️ 21.4 (convolution contre couche dense).

</details>
<details><summary>Indice 2</summary>

Choisis une image où un même petit objet est déplacé partout sur la photo (un tampon, un pochoir) ; dis ce qu'on obtient à la fin (une carte), puis pourquoi c'est économique.

</details>
<details><summary>Indice 3</summary>

Une première phrase possible : « Imagine un petit tampon transparent avec un motif dessus, que tu poses successivement sur chaque coin de la photo. » Il te reste à dire ce qu'on note à chaque position, ce qu'on obtient, qui choisit les motifs, et pourquoi le même tampon partout vaut mieux qu'un réglage par position.

</details>

### Ex 21.10 — Deviner l'âge et le genre sur une photo : jusqu'où aller ?

<details><summary>Indice 1</summary>

Relis l'éthique du ch. 3 (matrice de confusion par groupe) et le §21.10 de la fiche ; pour la question 5, cherche « Azure Face » et le règlement (UE) 2024/1689.

</details>
<details><summary>Indice 2</summary>

Question 1 : d'où viennent les labels du dataset ? Question 2 : il faut un jeu de test où le genre et l'âge sont **déclarés**, et des attributs pour former des groupes. Question 3 : le RGPD s'applique à une image de visage. Question 5 : dans le règlement, lis l'article 5 (pratiques interdites), l'article 50 (transparence) et l'annexe III, point 1 (biométrie).

</details>
<details><summary>Indice 3</summary>

Question 1 : le système reproduit les annotations, l'apparence que des annotateurs ont associée à un genre et à un âge, pas l'identité ni l'âge réel. Pour les questions suivantes, pense aux personnes dont l'apparence s'écarte de la norme du dataset, aux données sensibles qu'il faudrait pour un audit, à la possibilité réelle de refuser, aux perturbations du §21.10, et vérifie dans le texte du règlement si le sexe et l'âge sont dans la liste des catégorisations interdites.

</details>

### Ex 21.11 — Lire LeCun et coll. (1998) : LeNet-5

<details><summary>Indice 1</summary>

Lis d'abord le début du §II.A, la figure 2 et le tableau 1 ; garde la fiche (§21.3 à §21.9) ouverte à côté.

</details>
<details><summary>Indice 2</summary>

Les trois idées sont annoncées en une phrase au §II.A ; les raisons contre une couche dense sont dans l'introduction de la section II, juste avant le §II.A (taille, invariance, topologie). Pour la question 5, cherche ce que fait une unité de S2 avec ses quatre entrées.

</details>
<details><summary>Indice 3</summary>

Question 1 : champs récepteurs locaux, poids partagés, sous-échantillonnage. Pour la question 3, suis la figure 2 couche par couche et note la taille de chaque carte ; pour la question 4, applique $C_{out}\,(k^2 + 1)$ et $32 - 5 + 1$ ; pour la question 6, cherche les deux raisons que les auteurs donnent juste avant le tableau 1 ; pour la question 7, cherche les deux taux d'erreur de LeNet-5 au §III.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 21.E1 — Pourquoi un CNN plutôt qu'un MLP pour des images ?

<details><summary>Indice 1</summary>

Relis les §21.2.3 et §21.3.3 de la fiche, et l'✏️ 21.4.

</details>
<details><summary>Indice 2</summary>

Deux propriétés des images (locales, et un motif peut être n'importe où), deux propriétés des convolutions qui y répondent, puis la conséquence chiffrée.

</details>
<details><summary>Indice 3</summary>

Commence par : « Parce que l'information d'une image est locale, et qu'un même motif peut apparaître n'importe où. » Puis associe à chaque propriété le mécanisme de la convolution, donne un ordre de grandeur des paramètres, et finis sur la hiérarchie de features.

</details>

### 21.E2 — Taille de sortie et paramètres d'une Conv2d(3, 64, 3, stride=2, padding=1) sur une image 224×224

<details><summary>Indice 1</summary>

Relis les §21.3.4, §21.3.5 et §21.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

$H_{out} = \lfloor (H + 2p - k)/s \rfloor + 1$ ; paramètres $= C_{out}\,(C_{in}\,k^2 + 1)$ ; forme NCHW.

</details>
<details><summary>Indice 3</summary>

La hauteur : $\lfloor (224 + 2 - 3)/2 \rfloor + 1 = 112$. Il te reste à écrire la forme complète à quatre nombres, à compter les poids d'un filtre (3 canaux de 3 × 3) et le biais, et à dire ce qui dépend, ou non, de la taille de l'image.

</details>

### 21.E3 — À quoi sert une convolution 1×1 ?

<details><summary>Indice 1</summary>

Relis le §21.6 de la fiche et le paragraphe sur la convolution par canal (§21.4).

</details>
<details><summary>Indice 2</summary>

Elle ne regarde pas les voisins, mais elle lit **tous les canaux** du pixel : que peut-elle en faire ? Cite au moins un réseau qui s'en sert.

</details>
<details><summary>Indice 3</summary>

Commence par : « C'est une couche dense appliquée à chaque pixel, sur ses canaux, avec les mêmes poids partout. » Puis donne son usage principal (changer le nombre de canaux à faible coût) avec un exemple (le bloc bottleneck de ResNet, ou MobileNet), et son coût en poids.

</details>

### 21.E4 — Exemples adversariaux : quels risques, quelles défenses ?

<details><summary>Indice 1</summary>

Relis le §21.10 de la fiche, la section sur FGSM et son encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

Trois temps : la définition (et FGSM en une phrase), l'évaluation (quelle attaque, quel budget ε, quel adversaire), les défenses et leur coût.

</details>
<details><summary>Indice 3</summary>

Commence par : « Une entrée modifiée de façon presque invisible, mais calculée pour que le modèle se trompe. » Puis dis comment FGSM la calcule, comment tu mesurerais l'accuracy sous attaque, et ce que coûte l'entraînement adversarial ; finis par ce qu'on fait quand une erreur provoquée serait grave.

</details>

### 21.E5 — CNN ou Vision Transformer pour un nouveau projet de vision ?

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ « après VGG16 » du §21.9.1.

</details>
<details><summary>Indice 2</summary>

5 000 images, c'est peu : la première question n'est pas l'architecture mais le pré-entraînement. Ensuite, compare les a priori des deux familles, et dis comment tu trancherais.

</details>
<details><summary>Indice 3</summary>

Commence par : « Avec 5 000 images, je pars d'un modèle pré-entraîné, quel qu'il soit. » Puis propose une référence rapide, un second candidat, le critère de décision (validation, contraintes de déploiement), et ce qui compte souvent plus que l'architecture.

</details>

<a id="notebook"></a>

## Notebook

Les indices des exercices 21.12 à 21.32 seront écrits à la session suivante, avec les parties A à D du notebook.
