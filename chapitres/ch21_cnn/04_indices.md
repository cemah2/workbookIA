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

## Exercices du notebook

### Ex 21.12 — conv_output_size et pad2d 🔨

<details><summary>Indice 1</summary>

Relis la formule des tailles (fiche §21.3.4 et §21.3.5) et ta solution de ✏️ 21.3. Pour `pad2d`, il suffit d'un tableau de la nouvelle taille, rempli de `value`, au milieu duquel on recopie `x`.

</details>
<details><summary>Indice 2</summary>

`conv_output_size` : vérifie d'abord chaque argument (un entier, pas un booléen, au moins son minimum), calcule l'étendue du noyau dilaté $d(k - 1) + 1$, refuse une fenêtre plus grande que l'entrée complétée, puis fais la division entière (`//`) et convertis en `int`. `pad2d` : convertis `x` en un nouveau tableau `float64` à quatre axes, ramène `padding` à un couple $(p_h, p_w)$ (un petit outil qui accepte un entier ou un couple te resservira dans tout le module), crée `np.full(...)` de la forme complétée, puis recopie `x` dans la tranche du milieu.

</details>
<details><summary>Indice 3</summary>

```python
def conv_output_size(size, kernel_size, stride=1, padding=0, dilation=1):
    # 1. each argument: an int (not a bool) >= its minimum (size, kernel_size, stride, dilation >= 1; padding >= 0)
    span = dilation * (kernel_size - 1) + 1          # the cells covered by the dilated kernel
    # 2. the window must fit: span <= size + 2 * padding, else ValueError
    return int((size + 2 * padding - span) // stride + 1)


def pad2d(x, padding, value=0.0):
    # x -> a new float64 array of 4 axes; padding -> (ph, pw), each >= 0
    out = np.full((n, c, h + 2 * ph, w + 2 * pw), value, dtype=np.float64)
    out[:, :, ph:ph + h, pw:pw + w] = arr
```

</details>

### Ex 21.13 — Prédire où une image CIFAR-10 « jaunit » 🔮

<details><summary>Indice 1</summary>

Relis ✏️ 21.1 : le score d'un pixel dépend de ses trois composantes, pas de la couleur qu'on lui donne en la regardant. Pour `weight_13`, relis l'ordre des axes d'un poids de convolution (fiche §21.4.1).

</details>
<details><summary>Indice 2</summary>

Pour chaque photo, estime les composantes $R$, $G$ et $B$ de sa couleur dominante (le fond occupe le plus de pixels), puis le score $R + G - B$ ; pour c), cherche quelle couleur donne le score maximal (✏️ 21.1 f) et quelle photo en a des pixels proches. `weight_13` : un poids `(C_out, C_in, k_h, k_w)` avec un canal de sortie, trois d'entrée et un noyau 1 × 1.

</details>
<details><summary>Indice 3</summary>

`weight_13 = np.array([1.0, 1.0, -1.0]).reshape(1, 3, 1, 1)`. Pour les prédictions, écris pour chaque photo une estimation de $(R, G, B)$ de son fond et de son objet, calcule les deux scores, et pondère par la place qu'ils occupent ; calcule le score de chaque couleur dominante sans juger la couleur à l'œil.

</details>

### Ex 21.14 — im2col : la vue « œil de mouche » 🔨

<details><summary>Indice 1</summary>

Relis la vue « œil de mouche » (fiche §21.3.2) et ta solution de ✏️ 21.2 : chaque position de sortie lit un morceau de l'image complétée. `im2col` range ces morceaux côte à côte.

</details>
<details><summary>Indice 2</summary>

Inutile de boucler sur les positions de sortie : boucle sur les $k_h \times k_w$ **décalages** $(i, j)$ du noyau. Pour un décalage, les valeurs lues par toutes les fenêtres forment une tranche à pas fixe de l'image complétée : lignes $i\,d_h, i\,d_h + s_h, \dots$ ($H_{out}$ valeurs), colonnes $j\,d_w, \dots$ ($W_{out}$ valeurs), de forme `(N, C, H_out, W_out)`. Empile ces tranches sur deux nouveaux axes dans l'ordre `(N, C, k_h, k_w, H_out, W_out)`, puis `reshape` vers `(N, C·k_h·k_w, H_out·W_out)` : l'ordre des axes donne exactement l'indice `c·k_h·k_w + i·k_w + j`. Les tailles viennent de `conv_output_size`, le padding de `pad2d`.

</details>
<details><summary>Indice 3</summary>

```python
def im2col(x, kernel_size, stride=1, padding=0, dilation=1):
    # arguments -> pairs (kh, kw), (sh, sw), (ph, pw), (dh, dw); (ho, wo) with conv_output_size
    xp = pad2d(arr, (ph, pw))
    # the view of offset (i, j): one strided slice per offset
    view = xp[:, :, i * dh:i * dh + sh * (ho - 1) + 1:sh, j * dw:j * dw + sw * (wo - 1) + 1:sw]
    # stack the views to (N, C, kh, kw, ho, wo), then
    return patches.reshape(n, c * kh * kw, ho * wo)
```

</details>

### Ex 21.15 — conv2d : stride, padding, dilatation et groupes 🔨

<details><summary>Indice 1</summary>

Une convolution, c'est `im2col` suivi d'un produit matriciel (21.14) ; les groupes, c'est la même chose répétée sur chaque tranche de canaux ; `"same"`, c'est un padding calculé, parfois inégal entre les deux côtés.

</details>
<details><summary>Indice 2</summary>

Dans l'ordre : vérifie les formes (`weight` à quatre axes, `groups` qui divise `C_in` et `C_out`, `weight.shape[1] == C_in // groups`, le biais) ; traite `padding` : `"valid"` vaut 0 ; pour `"same"` (stride 1 seulement), la bordure totale vaut $d(k - 1)$ sur chaque axe, dont `total // 2` en haut (ou à gauche) et le reste en bas (ou à droite) : complète l'image toi-même avec `np.pad`, puis continue avec un padding de 0. Pour chaque groupe $g$ : `im2col` des canaux d'entrée du groupe, puis le produit des poids du groupe aplatis `(C_out/g, C_in/g·k_h·k_w)` par ces colonnes (le produit `@` diffuse sur l'axe des images). Enfin, concatène les groupes sur l'axe des canaux, remets en `(N, C_out, H_out, W_out)` et ajoute le biais.

</details>
<details><summary>Indice 3</summary>

```python
def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # checks of the shapes, of groups and of the bias (ValueError)
    # padding: "valid" -> 0; "same" -> np.pad by (total // 2, total - total // 2) on each axis, then 0
    for g in range(groups):
        cols = im2col(arr[:, g * g_in:(g + 1) * g_in], (kh, kw), stride, padding, dilation)   # (N, g_in*kh*kw, L)
        parts.append(w[g * g_out:(g + 1) * g_out].reshape(g_out, -1) @ cols)                 # (N, g_out, L)
    # concatenate the groups on the channel axis, reshape to (N, C_out, H_out, W_out), add the bias
```

</details>

### Ex 21.16 — Tests pytest paramétrés pour conv2d 🛠️

<details><summary>Indice 1</summary>

Lis la docstring de chacune des cinq versions fausses : chacune ne se trompe que dans certains cas. Un bon test est celui qui place la fonction dans ces cas-là.

</details>
<details><summary>Indice 2</summary>

Un seul test bien paramétré suffit presque : compare `conv(x, w, b, stride, padding)` à `F.conv2d` sur plusieurs strides (1, 2, 3), plusieurs paddings (0, 1, 2) et des tailles d'image qui ne tombent pas juste avec le stride ; prends un noyau **asymétrique** et rectangulaire (pour le retournement et les axes), et un biais non nul. Compare la forme avant les valeurs (`got.shape == want.shape`, puis `np.allclose`).

</details>
<details><summary>Indice 3</summary>

```python
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_conv_matches_torch(stride, padding):
    rng = np.random.default_rng(stride * 10 + padding)
    x, w, b = rng.normal(size=(2, 3, 8, 7)), rng.normal(size=(4, 3, 3, 2)), rng.normal(size=4)
    want = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=stride, padding=padding).numpy()
    # compare the shape, then the values, of conv(x, w, b, stride=stride, padding=padding)
```

</details>

### Ex 21.17 — Reproduire la chasse aux rayures verticales (Fig. 21.14) 🎨

<details><summary>Indice 1</summary>

Ta `conv2d` attend des tableaux `(N, C, H, W)` et des poids `(C_out, C_in, k_h, k_w)` : ajoute deux axes à l'image et au filtre. Relis le §21.3.1 pour le score parfait (et le pire).

</details>
<details><summary>Indice 2</summary>

Le score parfait vaut la somme des $+1$ du filtre (tout $+1$ sur du blanc, tout $-1$ sur du noir) ; le plus bas, l'opposé du nombre de $-1$ (tout $-1$ sur du blanc, tout $+1$ sur du noir). Les comptes se font avec `np.sum(scores_17 == valeur)`. Pour la figure : `plt.subplots(1, 5)`, `imshow` pour les quatre premiers panneaux (une barre de couleurs avec `fig.colorbar`), et pour le dernier un `plt.Rectangle` par correspondance, dont le coin est le coin en haut à gauche de la fenêtre.

</details>
<details><summary>Indice 3</summary>

`scores_17 = mylearn.conv.conv2d(IMG_17[None, None], K_17[None, None])[0, 0]`. Pour b) et c), compte de la même façon les positions qui valent le score le plus bas, puis 2. Pour les cadres : la sortie $(i, j)$ lit les pixels $i$ à $i + 2$ et $j$ à $j + 2$, d'où `plt.Rectangle((j - 0.5, i - 0.5), 3, 3, fill=False)` ajouté par `ax.add_patch(...)`.

</details>

### Ex 21.18 — max_pool2d et avg_pool2d 🔨

<details><summary>Indice 1</summary>

Un pooling lit les mêmes fenêtres qu'une convolution sans poids : reprends les tranches à pas fixe de 21.14, et remplace le produit matriciel par un maximum ou une moyenne.

</details>
<details><summary>Indice 2</summary>

Ramène `kernel_size`, `stride` (`None` → `kernel_size`) et `padding` à des couples, vérifie que chaque padding vaut au plus `k // 2`, calcule les tailles de sortie avec `conv_output_size`, complète l'image avec `pad2d` (avec $-\infty$ pour le max, des zéros pour la moyenne), puis, pour chacun des $k_h \times k_w$ décalages, prends la tranche à pas fixe ; le max est le maximum de ces tranches, la moyenne leur somme divisée par $k_h k_w$ (le padding compte).

</details>
<details><summary>Indice 3</summary>

```python
def max_pool2d(x, kernel_size, stride=None, padding=0):
    # pairs; stride = kernel if None; padding <= kernel // 2 on each axis, else ValueError
    xp = pad2d(arr, padding, value=-np.inf)            # a padded cell is never the maximum
    views = [xp[:, :, i:i + sh * (ho - 1) + 1:sh, j:j + sw * (wo - 1) + 1:sw] for i in range(kh) for j in range(kw)]
    # the maximum of the views, element by element
# avg_pool2d: the same views of a zero-padded image, np.sum(...) / (kh * kw)
```

</details>

### Ex 21.19 — Flou, Sobel, netteté : des filtres faits main avec F.conv2d 📦

<details><summary>Indice 1</summary>

Relis la forme d'un poids de convolution avec groupes (fiche §21.4.1) : `(C_out, C_in / groups, k_h, k_w)`. Un filtre appliqué à chaque canal séparément, c'est `groups=3`.

</details>
<details><summary>Indice 2</summary>

Les appels : `F.pad(x, (1, 1, 1, 1), mode="replicate")` (gauche, droite, haut, bas) ; `F.conv2d(xp, weight, groups=3)` avec un poids `(3, 1, 3, 3)` pour le flou (des $1/9$) et la netteté ; pour Sobel, un poids `(1, 3, 3, 3)` dont le canal $c$ vaut le coefficient de luminance de $c$ multiplié par le noyau de Sobel (`torch.stack`), sans groupes ; `.clamp(0.0, 1.0)` pour la netteté. Crée les poids dans le `dtype` de `x`.

</details>
<details><summary>Indice 3</summary>

```python
def blur_19(x):
    weight = torch.full((3, 1, 3, 3), 1.0 / 9, dtype=x.dtype)       # (C_out, C_in / groups, k, k)
    # F.pad with mode="replicate", then ONE F.conv2d with groups=3
# sobel_19: weight = torch.stack([c * sobel for c in LUMA_19])[None], shape (1, 3, 3, 3), no groups
# sharpen_19: the kernel expanded to (3, 1, 3, 3), groups=3, then .clamp(0.0, 1.0)
```

</details>

### Ex 21.20 — Conv1d, noyau pleine largeur et 1×1 : prédire formes et paramètres 🔮

<details><summary>Indice 1</summary>

Relis les §21.5 et §21.6 de la fiche, et la formule des tailles : elle s'applique à chaque axe qui glisse.

</details>
<details><summary>Indice 2</summary>

Une `Conv1d` lit des tenseurs `(N, C, L)` et ne glisse que le long de `L` ; son poids a la forme `(C_out, C_in, k)`. Une `Conv2d` glisse sur les deux derniers axes : avec un noyau aussi large que l'entrée, combien de positions reste-t-il en largeur ? Les paramètres : `C_out × (C_in × taille du noyau + 1)`.

</details>
<details><summary>Indice 3</summary>

Pour chaque couche, écris la forme d'entrée, retire les axes qui ne glissent pas (exemples, canaux), applique $L - k + 1$ à chacun de ceux qui glissent, et remplace l'axe des canaux par le nombre de filtres. Pour les paramètres, compte d'abord les poids d'**un** filtre (tout ce qu'il lit en une position), ajoute son biais, puis multiplie par le nombre de filtres.

</details>

### Ex 21.21 — Le détecteur de visage hiérarchique (Fig. 21.18 à 21.23) 🎨

<details><summary>Indice 1</summary>

Les niveaux 1 et 2 font exactement les mêmes trois étapes : une convolution avec `padding=1`, un seuil « score égal au nombre de $+1$ », un max pooling 3 × 3. Écris-les une fois, dans une boucle sur les deux bancs de filtres.

</details>
<details><summary>Indice 2</summary>

Le nombre de $+1$ de chaque filtre d'un banc : `(bank > 0).sum(axis=(1, 2, 3))`, un tableau d'une valeur par filtre ; la carte des correspondances : `(scores == wanted[None, :, None, None]).astype(float)` ; puis `max_pool2d(match, 3)`. Au niveau 3, une convolution sans padding sur des cartes 3 × 3 rend un seul score par filtre : `[0, :, 0, 0]`. Pour c) et d), compare chaque score à son nombre de $+1$. La figure : `plt.subplots(1, 10)` et une barre par filtre final (`ax.bar`).

</details>
<details><summary>Indice 3</summary>

```python
def detect_21(image):
    maps = np.asarray(image, dtype=float)[None, None]        # (1, 1, 27, 27)
    for bank in (L1_21, L2_21):
        scores = mylearn.conv.conv2d(maps, bank, padding=1)
        # wanted, the match map, then maps = max_pool2d(match, 3); keep the level-1 result
    # level 3: conv2d(maps, L3_21) without padding, [0, :, 0, 0]
```

</details>

### Ex 21.22 — Le convnet MNIST du livre en PyTorch, sur Fashion-MNIST 📦

<details><summary>Indice 1</summary>

Le tableau du convnet MNIST de la fiche (§21.9) donne chaque couche et sa forme de sortie ; la boucle d'entraînement est celle de 20.23, avec Adam et un ordre aléatoire tiré par un générateur.

</details>
<details><summary>Indice 2</summary>

`nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), ...)` dans l'ordre du contrat, la première couche dense ayant $64 \times 12 \times 12$ entrées. Dans `train_22` : l'optimiseur et la loss, puis **un** générateur `torch.Generator().manual_seed(seed)` avant les epochs ; à chaque epoch, `model.train()`, `torch.randperm(len(X), generator=generator)`, des tranches de `batch_size` indices, `.to(DEVICE)`, puis `zero_grad`, la loss, `backward`, `step` ; accumule `loss.item() * len(batch)` et divise par `len(X)`. Pour a) et b), refais les tailles : $32 \to 30 \to 28$, puis le pooling ; seule la première couche dense change.

</details>
<details><summary>Indice 3</summary>

```python
def train_22(model, X, y, epochs=2, batch_size=128, lr=1e-3, seed=0):
    optimizer, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)          # created once: a new order at each epoch
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        # for each slice of batch_size indices: to(DEVICE), zero_grad, loss, backward, step; total += loss.item() * len(batch)
        # history.append(total / len(X))
```
a) : $64 \times 14 \times 14$ ; pour b), remplace seulement les paramètres de la première couche dense dans ton total de ✏️ 21.4.

</details>

### Ex 21.23 — Diagnostiquer les courbes et la matrice de confusion du CNN 📈

<details><summary>Indice 1</summary>

Relis le recall et la precision (ch. 3) : le recall d'une classe se lit sur sa **ligne** (parmi ses images, combien sont reconnues), la precision sur sa **colonne** (parmi les images prédites dans cette classe, combien en sont vraiment). Pour a) et b), relis 20.25 : l'écart entre les deux courbes, et le minimum de la loss de validation.

</details>
<details><summary>Indice 2</summary>

`per_class_23` : `np.diag(cm)`, les sommes `cm.sum(axis=1)` et `cm.sum(axis=0)`, et `np.divide(..., out=np.zeros_like(...), where=sommes > 0)`, qui ne divise que là où le dénominateur est non nul. Pour les réponses : `np.argmin` rend un indice qui commence à 0 (une epoch commence à 1) ; les erreurs sont la matrice sans sa diagonale, `cm - np.diag(np.diag(cm))` ; e) se lit sur la ligne de `worst_23` de cette matrice des erreurs ; f) additionne la ligne et la colonne de `worst_23` (elles n'ont aucune case commune hors de la diagonale), puis divise par le total des erreurs.

</details>
<details><summary>Indice 3</summary>

```python
def per_class_23(cm):
    cm = np.asarray(cm, dtype=float)
    diag, rows = np.diag(cm), cm.sum(axis=1)
    recall = np.divide(diag, rows, out=np.zeros_like(diag), where=rows > 0)
    # precision: the same with the sums of the columns
    return recall, precision
```
a) : A finit avec une loss d'entraînement presque nulle et une loss de validation qui remonte (de l'overfitting), B avec deux courbes proches : les dropouts sont dans B. Pour b) à f), une ligne de calcul chacune : `np.argmin` sur la loss de validation de A (plus 1, car les epochs commencent à 1), une différence d'accuracies, puis le recall ou la matrice des erreurs.

</details>

### Ex 21.24 — Le CNN dont les formes ne passent pas 🐛

<details><summary>Indice 1</summary>

Lis le premier diagnostic en échec : il dit ce que reçoit une couche et ce qu'elle attend. Les quatre bugs viennent du passage de Keras à PyTorch et des tailles calculées de tête (fiche, encadré 🕰️ « le format des images », §21.4.1).

</details>
<details><summary>Indice 2</summary>

Point 1 : Keras range les images en `(N, H, W, C)`, PyTorch en `(N, C, H, W)`. Point 2 : une convolution lit les canaux que produit la précédente. Point 3 : une couche dense lit une ligne par image. Point 4 : refais les tailles de 28 × 28 jusqu'au pooling pour trouver le nombre de valeurs par image.

</details>
<details><summary>Indice 3</summary>

Point 1 : la première ligne de `forward` devient `x = x.permute(0, 3, 1, 2)`. Pour les points suivants, compare chaque fois le nombre que la couche attend (son `in_channels`, son `in_features`) à la forme que le diagnostic dit qu'elle reçoit, et demande-toi quel appel garde l'axe des exemples quand on aplatit.

</details>

### Ex 21.25 — Briser la symétrie : filtres identiques, aléatoires ou He 🔬

<details><summary>Indice 1</summary>

Relis le §21.7.1 et l'initialisation de He (ch. 16) ; `model.modules()` parcourt toutes les couches d'un modèle.

</details>
<details><summary>Indice 2</summary>

`init_25` : vérifie `kind`, puis pour chaque couche `isinstance(layer, (nn.Conv2d, nn.Linear))` : `layer.reset_parameters()` pour `"default"` ; sinon, sous `torch.no_grad()`, `layer.weight.fill_(0.05)` ou `nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")`, et `layer.bias.zero_()`. `distinct_filters_25` : aplatis chaque filtre (`reshape(len(weight), -1)`), et garde une liste de représentants : un filtre est nouveau s'il diffère de tous ceux déjà gardés de plus de `tol` (`np.max(np.abs(f - g))`).

</details>
<details><summary>Indice 3</summary>

```python
def distinct_filters_25(weight, tol=1e-6):
    filters = as_numpy(weight).reshape(len(weight), -1)
    kept = []
    for f in filters:
        # f is new when it differs from every kept filter by more than tol somewhere
    return len(kept)
```

</details>

### Ex 21.26 — Stride ou pooling : accuracy et temps sur CIFAR-10 🔬

<details><summary>Indice 1</summary>

Relis le §21.4.2 de la fiche : un stride de 2 et un pooling 2 × 2 divisent tous deux la taille par 2 ; la question est **où** se fait le calcul.

</details>
<details><summary>Indice 2</summary>

Suis les tailles : avec le pooling, $32 \to 32 \to 16 \to 16 \to 8$ ; avec le stride, $32 \to 16 \to 8$. Les deux finissent sur 64 cartes de 8 × 8, donc la même couche dense `nn.Linear(64 * 8 * 8, 10)`.

</details>
<details><summary>Indice 3</summary>

```python
def make_pool_26():
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         # the second convolution, ReLU and pooling, then nn.Flatten() and the dense layer
                         )
# make_stride_26: the same two convolutions with stride=2 and padding=1, no pooling
```

</details>

### Ex 21.27 — Filtres et cartes d'activation d'un ResNet-18 pré-entraîné 📦

<details><summary>Indice 1</summary>

Les filtres sont dans `model.conv1.weight` ; les hooks du ch. 20 permettent de lire la sortie d'une couche pendant la passe avant.

</details>
<details><summary>Indice 2</summary>

Les appels : `model.conv1.weight.detach()`, `.permute(0, 2, 3, 1)` (canaux à la fin), `min` et `max` sur les axes 1 à 3 avec `keepdims=True` ; `handle = layer.register_forward_hook(fonction)`, où `fonction(module, inputs, output)` garde `output.detach()` dans un dictionnaire ; puis `model.eval()`, `with torch.no_grad(): model(x)`, et `handle.remove()` dans un `finally`.

</details>
<details><summary>Indice 3</summary>

```python
def maps_27(model, x, layer):
    captured = {}
    hook = layer.register_forward_hook(lambda module, inputs, output: captured.setdefault("out", output.detach()))
    try:
        # eval mode, no autograd, one forward pass
    finally:
        hook.remove()
    return captured["out"]
```

</details>

### Ex 21.28 — Ce qu'un filtre cherche : montée de gradient sur l'entrée 📦

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §21.9.2 : on dérive par rapport à l'**image**, l'optimiseur ne contient qu'elle, et les poids sont gelés.

</details>
<details><summary>Indice 2</summary>

Les appels : `part = model[:upto + 1]` (un `nn.Sequential` se découpe comme une liste) ; `model.eval()` ; `p.requires_grad_(False)` pour chaque poids (garde leurs valeurs d'avant pour les remettre à la fin, dans un `finally`) ; l'image de départ, sur `DEVICE`, avec `requires_grad_(True)` ; `torch.optim.Adam([x], lr=lr)` ; à chaque pas `zero_grad`, la loss `-part(x)[0, channel].mean()`, `backward`, `step`, puis `x.clamp_(0.0, 1.0)` sous `torch.no_grad()` ; enfin `x.detach()`.

</details>
<details><summary>Indice 3</summary>

```python
def ascent_28(model, upto, channel, steps=60, lr=0.1, seed=0):
    part = model[:upto + 1]
    # eval mode; freeze the weights (and restore their requires_grad at the end, in a finally)
    x = 0.4 + 0.2 * torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(seed))
    x = x.to(DEVICE).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    # steps times: zero_grad, loss = -part(x)[0, channel].mean(), backward, step, clamp in no_grad
```

</details>

### Ex 21.29 — Attaque FGSM contre ton CNN Fashion-MNIST 📦

<details><summary>Indice 1</summary>

Relis la section « Au-delà du livre : l'attaque FGSM » de la fiche : un seul gradient, celui de la loss par rapport aux pixels, dont on ne garde que le signe.

</details>
<details><summary>Indice 2</summary>

Les appels : une copie détachée de `x` avec `requires_grad_(True)` ; `nn.functional.cross_entropy(model(x), y)` en mode évaluation ; `torch.autograd.grad(loss, x)`, qui rend le gradient par rapport à `x` **sans** toucher au `.grad` des poids (contrairement à `loss.backward()`) ; `grad.sign()`, puis `.clamp(0.0, 1.0)` et `.detach()`. Pour la courbe : une boucle sur `eps_list`, et pour chaque valeur une boucle sur les mini-batches, qui compte les bonnes prédictions sur les images attaquées.

</details>
<details><summary>Indice 3</summary>

```python
def fgsm_29(model, x, y, eps):
    model.eval()
    x = x.clone().detach().to(DEVICE).requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x), y.to(DEVICE))
    # the gradient of loss with respect to x ONLY (torch.autograd.grad), then x + eps * sign(grad), clamped, detached
```

</details>

### Ex 21.30 — conv_transpose2d par insertion de zéros 🔨

<details><summary>Indice 1</summary>

Relis la recette du §21.8 de la fiche et ta solution de ✏️ 21.6 : insérer des zéros, compléter, retourner le noyau, puis une convolution ordinaire. Ta `conv2d` fait déjà la dernière étape.

</details>
<details><summary>Indice 2</summary>

Dans l'ordre : vérifie les formes et le padding ($0 \le p \le k - 1$ sur chaque axe) ; crée un tableau de zéros de taille $(H - 1)s + 1$ sur chaque axe et pose `x` une case sur $s$ (`spread[:, :, ::sh, ::sw] = x`) ; retourne le noyau sur ses deux axes spatiaux (`[:, :, ::-1, ::-1]`) et échange ses deux axes de canaux (`(C_in, C_out, …)` → `(C_out, C_in, …)`) ; puis `conv2d` avec un padding de $k - 1 - p$ sur chaque axe et le biais.

</details>
<details><summary>Indice 3</summary>

```python
def conv_transpose2d(x, weight, bias=None, stride=1, padding=0):
    # checks: 4-D weight with weight.shape[0] == C_in, bias (C_out,), stride >= 1, 0 <= padding <= k - 1, output not empty
    spread = np.zeros((n, c_in, (h - 1) * sh + 1, (w - 1) * sw + 1))
    spread[:, :, ::sh, ::sw] = arr
    w = np.array(weight, dtype=np.float64)
    flipped = w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
    # then your conv2d, with a padding of k - 1 - p on each axis, and the bias
```

</details>

### Ex 21.31 — conv2d_backward : rétropropager à travers une convolution 🔨

<details><summary>Indice 1</summary>

Relis ∂ 21.7 (questions 1, 2 et 7) et la section « Au-delà du livre : la rétropropagation d'une convolution » : `db` et `dW` somment sur les images et les positions, `dx` renvoie le gradient de chaque fenêtre vers les cases qu'elle a lues.

</details>
<details><summary>Indice 2</summary>

Reprends `im2col` : la passe avant vaut `W.reshape(C_out, -1) @ cols` avec `cols = im2col(x, k, stride, padding)` de forme `(N, C_in·k_h·k_w, L)` et `dout` remis en `(N, C_out, L)`. Alors `db = dout.sum(axis=(0, 2, 3))`, `dW` est la somme sur les images de `dout_n @ cols_n.T` (un `np.einsum`), et le gradient des colonnes vaut `W.reshape(C_out, -1).T @ dout`. Pour `dx` (« col2im ») : remets ce gradient en `(N, C_in, k_h, k_w, H_out, W_out)`, crée `dx` à zéro de la taille **complétée** $(H + 2p_h, W + 2p_w)$, ajoute (`+=`) pour chaque décalage $(i, j)$ sa tranche à pas fixe, comme dans `im2col`, puis retire la bordure. Les cases qu'aucune fenêtre ne lit restent à zéro. Attention : `conv_transpose2d(dout, W, stride, padding)` ne redonne pas toujours `dx` : quand le stride laisse des lignes non lues en bas, sa sortie est trop courte.

</details>
<details><summary>Indice 3</summary>

```python
def conv2d_backward(dout, x, weight, stride=1, padding=0):
    # checks: dout must have the shape of the forward output
    cols = im2col(arr, (kh, kw), stride, (ph, pw))                    # (N, C_in*kh*kw, L)
    g2 = g.reshape(n, c_out, ho * wo)
    # dw: the sum over the images of g2[n] @ cols[n].T (one np.einsum), reshaped like w
    dcols = (w.reshape(c_out, -1).T @ g2).reshape(n, c_in, kh, kw, ho, wo)
    # dxp = zeros of the padded shape; for each (i, j): dxp[:, :, i:i + sh*(ho-1)+1:sh, j:j + sw*(wo-1)+1:sw] += dcols[:, :, i, j]
    # return dxp without its border, dw, db
```

</details>

### Ex 21.32 — Défi Fashion-MNIST : au moins 92 % avec moins de 100 000 paramètres 🏆

<details><summary>Indice 1</summary>

Où sont les paramètres du convnet du livre (21.22) ? Presque tous dans la première couche dense. Réduis l'image davantage avant elle, et garde les convolutions, qui coûtent peu de paramètres.

</details>
<details><summary>Indice 2</summary>

Une piste : deux blocs « convolution 3 × 3 avec padding, batchnorm, ReLU, max pooling 2 × 2 » (28 → 14 → 7), puis l'aplatissement, un dropout et une petite couche dense ; compte les paramètres avec `n_params` avant d'entraîner. Pour l'entraînement : AdamW, un planning qui monte puis descend (`OneCycleLR`, ch. 19), des mini-batches de 128, et un nombre d'epochs choisi d'après le temps d'une epoch sur ta machine. Pendant tes essais, garde 10 000 images d'entraînement comme validation.

</details>
<details><summary>Indice 3</summary>

```python
def train_32(X, y):
    torch.manual_seed(32)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
        # a second block of 32 filters (14 -> 7), nn.Flatten(), nn.Dropout(...), a small dense layer, 10 logits
    ).to(DEVICE)
    # AdamW + OneCycleLR(max_lr=3e-3, total_steps=epochs * number of mini-batches), scheduler.step() after each step
    return model.eval()
```

</details>
