# 25 · Autoencodeurs et VAE — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch25_autoencodeurs/06_mes_reponses.md` (créée par `python tools/start_chapter.py 25`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ 25.1 à 25.4, 25.6 et ∂ 25.7 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les dérivations ∂ 25.5 et 25.8, les estimations 🧮 25.9 et 25.10, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"F"`), plusieurs lettres entre guillemets pour un choix multiple (`"AF"`), une forme ou un vecteur entre guillemets (`"(3, 5, 1)"`), un mot entre guillemets (`"ABCD"`).
> Conventions du chapitre (fiche) : une entrée a $p$ valeurs, un code $d$ ; images en $(N, C, H, W)$ ; $\log$ népérien, KL en nats sauf mention contraire ; $\mathcal{N}(\mu, \sigma^2)$ a pour **variance** $\sigma^2$.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 25.Q1 — Avec ou sans perte : le test de l'inversibilité 🧠 ⏱️ 3 min
*Fiche §25.1 à §25.2.1 · livre §25.1 à §25.2.1*

a) Parmi ces transformations, lesquelles sont **sans perte** ? (A) écrire un entier en binaire au lieu du décimal ; (B) arrondir des températures au degré près ; (C) compresser un fichier texte en ZIP ; (D) garder une lettre sur deux d'un message ; (E) enregistrer une photo en JPEG de qualité 90 ; (F) passer une photo couleur en niveaux de gris. Écris leurs lettres.
b) Vrai ou faux : une transformation est sans perte si, et seulement si, on peut retrouver exactement l'entrée à partir de la sortie.
c) Un autoencodeur MNIST 784 → 20 → 784 reconstruit chaque image avec une petite erreur. Sa compression est : (A) sans perte ; (B) avec perte.
d) Un fichier de 2 400 000 octets est compressé en un fichier de 300 000 octets. Le facteur de compression (taille d'origine divisée par taille compressée).
e) Vrai ou faux : un compresseur sans perte bien conçu peut rendre plus court **n'importe quel** fichier.

### 25.Q2 — Pourquoi JPEG et MP3 battent un autoencodeur généraliste 🧠 ⏱️ 3 min
*Fiche §25.2.2 · livre §25.2.2*

a) Ce qui rend JPEG si efficace sur les photos : (A) il a été entraîné sur des millions de photos ; (B) il est sans perte ; (C) il a été conçu à partir de la structure des images et de la vision humaine, pour jeter d'abord ce que l'œil remarque le moins ; (D) il range chaque image dans un dictionnaire.
b) Un autoencodeur entraîné sur des radios thoraciques compresse une photo de paysage : (A) aussi bien qu'une radio ; (B) mieux, car un paysage est plus simple ; (C) mal, car son code est adapté aux radios.
c) Vrai ou faux : on utilise surtout les autoencodeurs pour remplacer JPEG et MP3.
d) Les deux usages pratiques que le livre donne aux autoencodeurs classiques : (A) débruiter des données ; (B) chiffrer des fichiers ; (C) réduire la dimension d'un dataset ; (D) trier des images par date. Écris leurs lettres.
e) Un morceau de 3 minutes encodé à 128 kbit/s (128 000 bits par seconde) : sa taille en millions d'octets (8 bits par octet), à 2 décimales.

### 25.Q3 — Mélange de contenu ou mélange de paramètres ? 🧠 ⏱️ 3 min
*Fiche §25.2.3 · livre §25.2.3*

a) Superposer deux photos en prenant 50 % de chaque pixel de l'une et 50 % de l'autre est un mélange : (A) de contenu ; (B) paramétrique.
b) On mélange deux mots lettre à lettre avec la règle du livre : pour chaque position, la lettre du milieu de l'alphabet entre les deux lettres (A = 1, …, Z = 26), en comptant à rebours si la seconde vient avant la première, et la plus proche du début de l'alphabet quand il y a deux lettres au milieu. Le mélange de « LUNE » et « SOIR », en majuscules.
c) Vrai ou faux : mélanger ainsi des codes compressés (comme des mots écrits) donne en général un objet intermédiaire sensé.
d) Ce qu'il faut à un code pour que le milieu de deux codes se décode en un objet intermédiaire sensé : (A) un code court ; (B) un code binaire ; (C) un code où des objets proches ont des codes proches, sans trous entre eux ; (D) un code sans perte.

### 25.Q4 — Goulot, code, variables latentes : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §25.3 · livre §25.3*

a) Vrai ou faux : pour entraîner un autoencodeur, il faut des labels.
b) Vrai ou faux : la cible d'un autoencodeur est son entrée elle-même.
c) Vrai ou faux : le code, ou les variables latentes, est la sortie de la couche la plus étroite.
d) Le terme employé aujourd'hui pour l'apprentissage d'un autoencodeur : (A) supervisé ; (B) semi-supervisé ; (C) auto-supervisé ; (D) par renforcement.
e) Vrai ou faux : un autoencodeur dont le code a autant de valeurs que l'entrée peut reconstruire parfaitement sans rien comprimer.

### 25.Q5 — Le réseau qui ne savait redessiner qu'une image 🧠 ⏱️ 3 min
*Fiche §25.3 · livre §25.3*

Un autoencodeur à deux couches denses prend des images en niveaux de gris de $64 \times 64$ pixels et a un goulot de 10 neurones.

a) Le nombre de **poids** de ses deux couches (sans les biais).
b) Le nombre de **biais** de sa couche de sortie.
c) Entraîné longtemps sur une seule image, on lui donne une image entièrement noire (tous les pixels à 0). Il rend : (A) une image noire ; (B) un bruit aléatoire ; (C) une version de l'image apprise, produite surtout par les biais ; (D) une erreur.
d) Vrai ou faux : une erreur de reconstruction très faible sur les images d'entraînement prouve qu'un autoencodeur comprime bien.
e) Ce qui révèle le mieux une mémorisation : (A) l'erreur sur les images d'entraînement ; (B) le nombre d'epochs ; (C) la sortie pour des images jamais vues, très différentes ; (D) la taille du goulot.

### 25.Q6 — Profondeur ou nombre de latents : qui améliore quoi ? 🧠 ⏱️ 3 min
*Fiche §25.4 · livre §25.4*

a) Dans l'autoencodeur MNIST à deux couches, passer de 20 à 3 latents donne : (A) des reconstructions plus nettes ; (B) aucune différence ; (C) des chiffres flous et mélangés.
b) À 20 latents, passer de cet autoencodeur à deux couches à un autoencodeur profond (plusieurs couches de chaque côté) donne : (A) des reconstructions plus floues ; (B) des reconstructions plus nettes ; (C) les mêmes reconstructions.
c) Ce dont un autoencodeur a besoin, selon le livre : (A) assez de poids et de couches pour calculer le code ; (B) des labels ; (C) assez de latents pour porter l'information ; (D) un learning rate nul. Écris leurs lettres.
d) Vrai ou faux : diviser la largeur par deux d'une couche à la suivante est une règle obligatoire des autoencodeurs.
e) L'intérêt d'un goulot de 2 latents seulement : (A) de meilleures reconstructions ; (B) pouvoir dessiner l'espace latent sur une page ; (C) un entraînement sans loss ; (D) supprimer le décodeur.

### 25.Q7 — Du bruit dans l'espace latent d'un autoencodeur classique 🧠 ⏱️ 3 min
*Fiche §25.5, §25.5.1 · livre §25.5, §25.5.1*

Un autoencodeur dense a 20 latents positifs qui montent jusqu'à quelques centaines.

a) On ajoute à chaque latent un bruit uniforme entre −1 et 1, puis on décode. (A) les chiffres sont ruinés ; (B) on ne voit presque aucune différence.
b) Même chose avec un bruit entre −100 et 100 : (A) les chiffres sont ruinés ; (B) on ne voit presque aucune différence.
c) On ajoute un bruit entre −100 et 100 à **un seul** latent : (A) seuls les chiffres qui dépendent beaucoup de ce latent changent nettement ; (B) tous les chiffres sont abîmés de la même façon.
d) Vrai ou faux : dans un autoencodeur classique, toute région de l'espace latent se décode en une image plausible.
e) On décode 20 latents tirés au hasard entre 0 et 25 : (A) des taches qui ne sont pas des chiffres ; (B) des chiffres nets.

### 25.Q8 — Lire une carte latente à deux dimensions 🧠 ⏱️ 3 min
*Fiche §25.5.2, §25.5.3 · livre §25.5.2, §25.5.3*

Un autoencodeur à 2 latents place les codes de 10 000 chiffres sur une carte : les 1 ont leur région, les 4, 7 et 9 se chevauchent, et de grandes zones de la carte sont vides.

a) Le chevauchement des 4, 7 et 9 signifie : (A) le modèle les distingue mal et leur donne des codes voisins ; (B) ces chiffres sont rares ; (C) la carte est mal dessinée ; (D) le modèle les reconstruit mieux que les autres.
b) Décoder un point d'une zone vide, loin de tout exemple, donne : (A) un chiffre net ; (B) une forme imprévisible, car le décodeur n'a jamais reçu de code de cette région ; (C) une image noire.
c) Faire la moyenne des codes de deux images puis la décoder est un mélange : (A) de contenu ; (B) paramétrique.
d) Vrai ou faux : la moyenne des codes d'un 2 et d'un 4 se décode forcément en un chiffre.
e) Le milieu des codes $\mathbf{z}_0 = (2, 6)$ et $\mathbf{z}_1 = (8, 0)$, comme vecteur.

### 25.Q9 — Autoencodeur linéaire et ACP 🧠 ⏱️ 3 min
*Fiche §25.6 · livre §25.6*

a) Un autoencodeur **linéaire** (sans activation) à $d$ latents, entraîné à l'erreur quadratique sur des données centrées, apprend à l'optimum : (A) exactement les $d$ premiers axes principaux, dans l'ordre ; (B) le sous-espace des $d$ premières composantes principales, sans forcément ses axes ; (C) un sous-espace quelconque ; (D) les $d$ dernières composantes.
b) Vrai ou faux : un autoencodeur profond à ReLU donne toujours la même chose que l'ACP.
c) Vrai ou faux : sur des données non centrées (et qu'on ne centre pas avant), l'autoencodeur linéaire a besoin de biais pour retrouver l'ACP.
d) Ce qu'un autoencodeur non linéaire peut faire de plus que l'ACP : (A) suivre une surface courbe (une variété) avec peu de latents ; (B) garantir une erreur nulle ; (C) se passer de données ; (D) fournir des axes orthogonaux classés par variance.
e) Dans le plan, la première composante principale de données centrées est $\mathbf{u} = (0{,}6\,;\ 0{,}8)$. Le code (un seul latent) que donne l'autoencodeur linéaire optimal d'encodeur $\mathbf{u}$ au point $\mathbf{x} = (5, 0)$.

### 25.Q10 — Autoencodeurs convolutifs et débruiteurs 🧠 ⏱️ 3 min
*Fiche §25.7, §25.7.1, §25.8 · livre §25.7, §25.7.1, §25.8*

a) Dans un autoencodeur débruiteur, l'entrée et la cible sont : (A) l'image propre et l'image bruitée ; (B) l'image bruitée et l'image propre ; (C) l'image bruitée deux fois ; (D) l'image propre deux fois.
b) Vrai ou faux : dans un débruiteur, le code doit avoir moins de valeurs que l'image.
c) On ajoute un bruit gaussien à une image, puis on écrête dans $[0, 1]$. Pourquoi ? (A) les pixels d'une image restent dans $[0, 1]$, comme les sorties sigmoïdes du réseau ; (B) pour accélérer l'entraînement ; (C) pour retirer le bruit ; (D) pour que la loss soit nulle.
d) Ce qui remplace aujourd'hui souvent un pooling suivi, au décodage, d'un sur-échantillonnage : (A) des couches denses ; (B) une convolution de stride 2 à l'encodage, une convolution transposée de stride 2 au décodage ; (C) un dropout ; (D) une batchnorm.
e) Vrai ou faux : un débruiteur entraîné sur des chiffres MNIST débruite aussi bien des photos.

### 25.Q11 — Une entrée jamais vue : un tigre chez les chiffres 🧠 ⏱️ 3 min
*Fiche §25.5.4, §25.7.2 · livre §25.5.4, §25.7.2*

a) Un autoencodeur entraîné sur MNIST reçoit une photo réduite en $28 \times 28$. Il rend : (A) la photo fidèlement ; (B) une image noire ; (C) une forme faite de morceaux de chiffres ; (D) un chiffre net, toujours le même.
b) Vrai ou faux : une erreur de reconstruction élevée peut signaler une entrée qui ne ressemble pas aux données d'entraînement.
c) Vrai ou faux : une erreur de reconstruction faible garantit que l'entrée ressemble aux données d'entraînement.
d) Pourquoi la reconstruction de la photo a-t-elle un cadre noir ? (A) les pixels du bord sont presque toujours noirs dans MNIST, et le décodeur a appris à les mettre noirs ; (B) la photo avait un cadre ; (C) la convolution ajoute un padding noir ; (D) par hasard.
e) Pour un autoencodeur entraîné sur MNIST, lesquelles de ces entrées sont hors de sa distribution ? (A) un chiffre du jeu de test MNIST ; (B) une chaussure de Fashion-MNIST ; (C) une photo de chat réduite en $28 \times 28$. Écris leurs lettres.

### 25.Q12 — Les trois exigences d'un VAE et l'astuce de reparamétrisation 🧠 ⏱️ 4 min
*Fiche §25.9 à §25.10 · livre §25.9 à §25.10*

a) Les trois exigences du livre pour un espace latent dont on peut tirer des codes : (A) les codes rassemblés dans une région connue ; (B) des entrées semblables ont des codes proches ; (C) le moins de trous possible ; (D) des latents entiers ; (E) un seul code par classe. Écris leurs lettres.
b) Pour chaque latent, l'encodeur d'un VAE produit : (A) une seule valeur ; (B) une moyenne et une variance (en pratique, sa log-variance) ; (C) un label ; (D) une probabilité entre 0 et 1.
c) Vrai ou faux : on peut rétropropager le gradient à travers un tirage $z \sim \mathcal{N}(\mu, \sigma^2)$ écrit tel quel, sans reformulation.
d) Avec $\mu = 1{,}5$, $\sigma = 0{,}5$ et un bruit tiré $\varepsilon = -2$, la valeur du code $z$ par l'astuce de reparamétrisation.
e) Vrai ou faux : un VAE entraîné rend exactement la même sortie à chaque passage de la même entrée.

<a id="rappels"></a>

## 🔁 Rappels

### 25.R1 — Quelles augmentations garder pour des chiffres manuscrits ? 🔁 ⏱️ 5 min
*Ch. 24 (augmentation de données) · parcours R*

a) Lesquelles de ces transformations sont plausibles pour augmenter des chiffres MNIST ? (A) un décalage de 2 pixels ; (B) une rotation de ±10° ; (C) une symétrie gauche-droite ; (D) une rotation de 180° ; (E) un léger zoom. Écris leurs lettres.
b) Vrai ou faux : on applique aussi l'augmentation aux images de test.
c) Vrai ou faux : tirer un bruit neuf à chaque mini-batch pour un débruiteur est une forme d'augmentation de données.

### 25.R2 — Résumer une séquence entière dans l'état final d'un LSTM 🔁 ⏱️ 5 min
*Ch. 22 (LSTM, formes de `nn.LSTM`) · parcours R*

`lstm = nn.LSTM(input_size=8, hidden_size=32, batch_first=True)` reçoit `x` de forme `(16, 50, 8)` : 16 séquences de 50 pas, 8 features par pas.

a) La forme de `h_n`.
b) Si l'on prend l'état final de chaque séquence comme code, le facteur de compression d'une séquence (nombre de valeurs de la séquence divisé par celui du code), à 1 décimale.
c) Vrai ou faux : un autoencodeur de séquences peut utiliser un LSTM comme encodeur et un autre LSTM comme décodeur.

### 25.R3 — La divergence KL, ce que mesure le terme de régularisation du VAE 🔁 ⏱️ 5 min
*Ch. 6 (divergence KL) · parcours R, M*

a) $\mathrm{KL}(P \,\|\, Q)$ en **bits** pour $P = (0{,}5\ ;\ 0{,}5)$ et $Q = (0{,}25\ ;\ 0{,}75)$, à 2 décimales.
b) Vrai ou faux : $\mathrm{KL}(P \,\|\, Q) = \mathrm{KL}(Q \,\|\, P)$ pour toutes les lois $P$ et $Q$.
c) Vrai ou faux : $\mathrm{KL}(P \,\|\, P) = 0$ pour toute loi $P$.
d) Vrai ou faux : une divergence KL n'est jamais négative.
e) La divergence de a) en **nats**, à 2 décimales.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 25.1 — Compter les paramètres d'un autoencodeur à goulot ✏️ ★ ⏱️ 10 min
**Objectif :** compter les paramètres d'un autoencodeur dense, simple puis profond, et comparer la taille du code à celle de l'image.
**Prérequis :** ch. 16 (paramètres d'une couche dense) · fiche §25.3, §25.4 · **Fil rouge :** MNIST · **Parcours :** R, M

Toutes les couches sont denses, avec un biais par neurone.

a) Le nombre de paramètres de l'encodeur d'un autoencodeur MNIST 784 → 20 → 784 (la couche 784 → 20).
b) Celui de son décodeur (la couche 20 → 784).
c) Le nombre total de paramètres de cet autoencodeur.
d) Un autoencodeur profond a l'encodeur 784 → 512 → 256 → 128 → 20 et le décodeur symétrique 20 → 128 → 256 → 512 → 784. Le nombre de paramètres de son encodeur.
e) Le nombre total de paramètres de cet autoencodeur profond.
f) Le facteur de réduction entre une image MNIST et son code de 20 latents (nombre de valeurs de l'image divisé par celui du code), à 1 décimale.
g) L'autoencodeur du tigre du livre (10 000 → 20 → 10 000) : son nombre total de paramètres, biais compris.

Dans ta copie : le livre dit avoir rangé 10 000 nombres dans 400 000. D'où vient l'écart avec ta réponse g) ? Pourquoi un tel réseau peut-il reproduire une seule image sans rien comprimer ?

### Ex 25.2 — Mélanger deux cercles : pixels contre paramètres ✏️ ★ ⏱️ 10 min
**Objectif :** comparer, sur des nombres, le mélange de contenu et le mélange paramétrique de deux formes.
**Prérequis :** 0B (coordonnées, aire d'un disque) · fiche §25.2.3 · **Parcours :** M

Une image noire contient un disque blanc A de centre $(2, 2)$, de rayon 1 et d'intensité 1,0. Une autre image noire contient un disque gris B de centre $(10, 6)$, de rayon 4 et d'intensité 0,4. Un pixel vaut l'intensité du disque s'il est dans le disque (distance au centre au plus égale au rayon), 0 sinon.

Le mélange **paramétrique** à $t$ mélange chaque paramètre : $(1 - t)\cdot\text{A} + t\cdot\text{B}$ pour le centre, le rayon et l'intensité.

a) Le centre du disque obtenu par mélange paramétrique à $t = 0{,}25$, comme vecteur.
b) Son rayon, à 2 décimales.
c) Son intensité, à 2 décimales.

Le mélange de **contenu** à $t = 0{,}25$ est l'image $0{,}75 \cdot \text{image A} + 0{,}25 \cdot \text{image B}$, pixel par pixel.

d) La valeur du pixel de coordonnées $(2, 2)$ dans le mélange de contenu.
e) Celle du pixel $(10, 6)$.
f) Le nombre de disques visibles (de valeur non nulle) dans le mélange de contenu.
g) L'aire du disque obtenu par mélange paramétrique à $t = 0{,}5$, à 1 décimale.
h) La moyenne des aires des disques A et B, à 1 décimale.

Dans ta copie : que vaut, dans le mélange de contenu, le pixel $(4, 3)$, le centre du disque de a) ? Pourquoi g) et h) diffèrent-ils ? Quel paramètre faudrait-il mélanger linéairement pour que l'aire du disque intermédiaire soit la moyenne des aires ? Qu'en conclus-tu sur le choix des paramètres d'un mélange ?

### Ex 25.3 — Suivre les formes dans l'autoencodeur convolutif ✏️ ★★ ⏱️ 15 min
**Objectif :** prévoir la forme d'un tenseur à chaque étage d'un autoencodeur convolutif, et compter ses paramètres.
**Prérequis :** ch. 21 (tailles de sortie, paramètres d'une convolution) · fiche §25.7, §25.8 · **Fil rouge :** MNIST · **Parcours :** R, M

L'autoencodeur convolutif du livre, en PyTorch, pour un mini-batch de 64 images MNIST `(64, 1, 28, 28)` :
- encodeur : `Conv2d(1, 16, 3, padding=1)`, `ReLU`, `MaxPool2d(2)`, `Conv2d(16, 8, 3, padding=1)`, `ReLU`, `MaxPool2d(2)`, `Conv2d(8, 3, 3, padding=1)`, `ReLU` ;
- décodeur : `Upsample(scale_factor=2)`, `Conv2d(3, 16, 3, padding=1)`, `ReLU`, `Upsample(scale_factor=2)`, `Conv2d(16, 1, 3, padding=1)`.

a) La forme du tenseur après le premier pooling.
b) Sa forme après le second pooling.
c) La forme du code (la sortie de l'encodeur).
d) Le nombre de paramètres de l'encodeur.
e) Le nombre de paramètres du décodeur.
f) Le nombre total de paramètres.
g) Le facteur de réduction entre une image et son code (valeurs de l'image divisées par valeurs du code), à 2 décimales.

Le débruiteur de la fig. 25.47 du livre commence par `Conv2d(1, 32, 3, padding=1)`, `ReLU`, `MaxPool2d(2)`, `Conv2d(32, 32, 3, padding=1)`, `ReLU`, `MaxPool2d(2)`.

h) Le nombre de valeurs de son code pour **une** image (la sortie du second pooling).

Dans ta copie : compare h) au nombre de pixels d'une image. Pourquoi n'est-ce pas un problème pour un débruiteur ? Combien de paramètres compte, à part, l'autoencodeur à 20 latents de l'Ex 25.1 ? Pourquoi le convolutif reconstruit-il mieux avec beaucoup moins de paramètres ?

### Ex 25.4 — Retrouver 28×28 : padding et output_padding d'une convolution transposée ✏️ ★★ ⏱️ 15 min
**Objectif :** choisir les réglages d'une convolution transposée pour retrouver la taille d'une image.
**Prérequis :** ch. 21 (convolution transposée) · fiche §25.7 (rappel maths), §25.8 · **Fil rouge :** MNIST · **Parcours :** M

$H_{out} = (H_{in} - 1)\,s - 2p + k + p_{out}$ pour `ConvTranspose2d(kernel_size=k, stride=s, padding=p, output_padding=` $p_{out}$ `)`, et $H_{out} = \lfloor (H_{in} + 2p - k)/s \rfloor + 1$ pour `Conv2d`. Les images sont carrées : on ne suit que la hauteur.

a) La hauteur de sortie de `ConvTranspose2d(k=3, s=2, p=1)` sur une entrée de hauteur 7.
b) La même avec `output_padding=1`.
c) La hauteur de sortie de `ConvTranspose2d(k=3, s=2, p=1, output_padding=1)` sur une entrée de hauteur 14.
d) La hauteur de sortie de `ConvTranspose2d(k=2, s=2, p=0)` sur une entrée de hauteur 7.
e) La hauteur de sortie de `ConvTranspose2d(k=4, s=2, p=1)` sur une entrée de hauteur 7.
f) À l'encodage, la hauteur de sortie de `Conv2d(k=3, s=2, p=1)` sur une entrée de hauteur 7.
g) La hauteur de sortie de `ConvTranspose2d(k=3, s=2, p=1)` sur une entrée de la hauteur f), sans `output_padding`.
h) Le nombre de paramètres de `ConvTranspose2d(32, 16, 3)` (avec biais).

Dans ta copie : quelle valeur d'`output_padding` donnerait 8 au lieu de g) ? Pourquoi la convolution de f) ne peut-elle pas savoir, à elle seule, si son entrée avait une hauteur 7 ou 8 ? Écris une ligne de PyTorch qui vérifie ta réponse c) sans rien entraîner.

### Ex 25.5 — Dériver à travers un tirage : l'astuce de reparamétrisation ∂ ★★ ⏱️ 20 min
**Objectif :** comprendre, par le calcul, pourquoi la reparamétrisation laisse passer le gradient.
**Prérequis :** ch. 18 (règle de la chaîne) · ch. 2 (loi normale) · fiche §25.9.2 · **Parcours :** R, M

On écrit le code d'un latent $z = \mu + \sigma\,\varepsilon$ avec $\sigma = e^{s/2}$, où $s = \log \sigma^2$ est la sortie de l'encodeur et $\varepsilon \sim \mathcal{N}(0, 1)$ un nombre tiré à part. Réponds dans ta copie, avec la démarche.

1. Calcule $\partial z/\partial \mu$ et $\partial z/\partial \sigma$, puis $\partial \sigma/\partial s$, et déduis-en $\partial z/\partial s$ en fonction de $\sigma$ et $\varepsilon$.
2. Le décodeur et la loss donnent $\partial L/\partial z$, notée $g$. Écris $\partial L/\partial \mu$ et $\partial L/\partial s$ par la règle de la chaîne. Que vaut $\partial L/\partial s$ quand $\varepsilon = 0$ ? Pourquoi est-ce logique ?
3. Application : $\mu = 0{,}2$, $\sigma^2 = 4$, $\varepsilon = 0{,}5$, et $L = \tfrac{1}{2}(z - y)^2$ avec $y = 1$. Calcule $z$, $L$, $\partial L/\partial \mu$ et $\partial L/\partial s$.
4. Pourquoi ne peut-on pas dériver « $z$ tiré dans $\mathcal{N}(\mu, \sigma^2)$ » par rapport à $\mu$, quand le tirage est fait par une fonction aléatoire qui reçoit $\mu$ et $\sigma$ ? Que change le fait d'écrire $z$ comme une fonction de $\mu$, $\sigma$ et d'un $\varepsilon$ qui ne dépend d'aucun poids ?
5. Montre, avec $\mathbb{E}[\varepsilon] = 0$ et $\mathbb{E}[\varepsilon^2] = 1$, que $\mathbb{E}[z^2] = \mu^2 + \sigma^2$. Dérive cette expression par rapport à $\mu$. Puis calcule $\mathbb{E}[\partial (z^2)/\partial \mu]$ en dérivant $z^2 = (\mu + \sigma\varepsilon)^2$ **avant** de prendre l'espérance. Que montre l'égalité des deux résultats sur le gradient estimé avec un seul tirage de $\varepsilon$ ?

### Ex 25.6 — Où tombent les tirages d'une gaussienne en grande dimension ✏️ ★★ ⏱️ 25 min
**Objectif :** calculer où se trouvent les codes typiques d'un prior gaussien, et ce que devient le milieu de deux codes.
**Prérequis :** ch. 2 (loi normale, variance) · 0B (racine carrée, exponentielle) · fiche §25.5.3 (rappels maths), §25.9.1 · **Parcours :** M

On tire $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ : $d$ coordonnées indépendantes de moyenne 0 et de variance 1.

a) $\mathbb{E}\lVert\mathbf{z}\rVert^2$ pour $d = 20$.
b) La norme typique d'un tirage pour $d = 20$, prise égale à $\sqrt{\mathbb{E}\lVert\mathbf{z}\rVert^2}$, à 2 décimales.

En dimension 2, $P(\lVert\mathbf{z}\rVert \le r) = 1 - e^{-r^2/2}$.

c) La probabilité qu'un tirage en dimension 2 tombe à une distance au plus 1 du centre, à 3 décimales.
d) La probabilité qu'il tombe à une distance au plus 3, à 3 décimales.
e) Le rayon $r$ du disque qui contient la moitié des tirages en dimension 2, à 2 décimales.

On prend maintenant deux tirages indépendants $\mathbf{z}_0$ et $\mathbf{z}_1$ en dimension $d = 20$, et leur milieu $\mathbf{m} = (\mathbf{z}_0 + \mathbf{z}_1)/2$.

f) $\mathbb{E}\lVert\mathbf{m}\rVert^2$.
g) La norme typique du milieu, $\sqrt{\mathbb{E}\lVert\mathbf{m}\rVert^2}$, à 2 décimales.
h) La norme typique d'un tirage pour $d = 512$, à 2 décimales.

Dans ta copie : comment se compare le milieu g) aux tirages b) ? Pourquoi un décodeur entraîné sur des codes de norme proche de b) risque-t-il de mal décoder le milieu ? Que fait l'interpolation sphérique pour éviter cela ?

### Ex 25.7 — La divergence KL entre N(μ, σ²) et N(0, 1) ∂ ★★★ ⏱️ 35 min
**Objectif :** dériver la forme fermée de la divergence KL du VAE, puis l'appliquer.
**Prérequis :** ch. 6 (divergence KL) · Ex 25.5 · fiche §25.9.1 (rappel maths sur la gaussienne), §25.9.2 · **Parcours :** M

On note $q = \mathcal{N}(\mu, \sigma^2)$ et $p = \mathcal{N}(0, 1)$, et $\mathrm{KL}(q \,\|\, p) = \mathbb{E}_{z \sim q}\left[\ln q(z) - \ln p(z)\right]$, la version continue de la divergence du ch. 6. Dans ta copie, avec la démarche :

1. Écris $\ln q(z)$ et $\ln p(z)$ à partir de la densité de la gaussienne (fiche).
2. Calcule $\mathbb{E}_q[\ln q(z)]$, puis $\mathbb{E}_q[\ln p(z)]$, à l'aide de $\mathbb{E}[(z - \mu)^2] = \sigma^2$ et $\mathbb{E}[z^2] = \mu^2 + \sigma^2$.
3. Déduis-en $\mathrm{KL}(q \,\|\, p) = \tfrac{1}{2}\left(\mu^2 + \sigma^2 - 1 - \ln \sigma^2\right)$.
4. Écris-la en fonction de $s = \ln\sigma^2$, et calcule ses dérivées partielles par rapport à $\mu$ et à $s$.

Puis les valeurs (en nats sauf mention contraire) :

a) La KL pour $\mu = 2$ et $\sigma = 1$.
b) La KL pour $\mu = 0$ et $\sigma^2 = e$ (soit $s = 1$), à 3 décimales.
c) La KL pour $\mu = 0$ et $\sigma = 0{,}1$, à 3 décimales.
d) La KL pour $\mu = 1$ et $\sigma^2 = 4$, à 3 décimales.
e) La KL d'un code à 3 latents de moyennes $(1, 0, -1)$ et de log-variances toutes nulles (les latents sont indépendants).
f) La KL pour $\mu = 0{,}5$ et $\sigma^2 = 2{,}5$, à 3 décimales.
g) La dérivée de la KL par rapport à $s = \ln\sigma^2$ en $s = \ln 5$.
h) La KL de d) en **bits**, à 3 décimales.

Dans ta copie : pour $\mu$ fixé, quelle valeur de $\sigma^2$ rend la KL minimale ? Pourquoi la KL pénalise-t-elle à la fois un $\sigma$ très petit (c) et une moyenne loin de 0 (a) ? Que se passerait-il dans un VAE sans ce terme ?

### Ex 25.8 — Un autoencodeur linéaire retrouve l'ACP (cas 2D guidé) ∂ ★★★ ⏱️ 40 min
**Objectif :** montrer, dans le plan, que le meilleur autoencodeur linéaire à un latent projette sur la première composante principale, et qu'il n'en retrouve que la droite.
**Prérequis :** ch. 12 (ACP, covariance, vecteurs propres) · fiche §25.6 · **Parcours :** M

Des données $\mathbf{x}_i$ du plan sont centrées, de matrice de covariance $\mathbf{C} = \frac{1}{n}\sum_i \mathbf{x}_i\mathbf{x}_i^\top$. Un autoencodeur linéaire à un latent a un encodeur $\mathbf{a}$ et un décodeur $\mathbf{b}$ (deux vecteurs du plan) : $z = \mathbf{a}^\top\mathbf{x}$ et $\hat{\mathbf{x}} = z\,\mathbf{b}$. Sa loss est $L(\mathbf{a}, \mathbf{b}) = \frac{1}{n}\sum_i \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert^2$. Réponds dans ta copie.

1. Cas **lié** : $\mathbf{a} = \mathbf{b} = \mathbf{w}$ avec $\lVert\mathbf{w}\rVert = 1$. Montre que $\hat{\mathbf{x}}$ est la projection orthogonale de $\mathbf{x}$ sur la droite de $\mathbf{w}$, puis que $L = \operatorname{tr}(\mathbf{C}) - \mathbf{w}^\top\mathbf{C}\,\mathbf{w}$ (Pythagore).
2. Déduis-en que minimiser $L$ revient à maximiser $\mathbf{w}^\top\mathbf{C}\,\mathbf{w}$ sur le cercle unité, et que le maximum est atteint pour un vecteur propre de la plus grande valeur propre de $\mathbf{C}$ (écris $\mathbf{w}$ dans la base des vecteurs propres).
3. Application : $\mathbf{C} = \begin{pmatrix} 5 & 2 \\ 2 & 2 \end{pmatrix}$. Calcule ses valeurs propres et un vecteur propre unitaire de la plus grande. Calcule $L$ pour ce $\mathbf{w}$, puis pour $\mathbf{w} = (1, 0)$.
4. Cas **non lié** : montre que, pour tout nombre $c \ne 0$, les paires $(\mathbf{a}, \mathbf{b})$ et $(c\,\mathbf{a}, \mathbf{b}/c)$ donnent les mêmes reconstructions. Qu'est-ce que l'autoencodeur ne peut donc pas déterminer ? Avec $d$ latents, quelle transformation du code joue le même rôle, et pourquoi dit-on qu'il retrouve le **sous-espace** de l'ACP mais pas ses axes ?
5. Si les données ne sont pas centrées, que faut-il ajouter à l'autoencodeur pour retrouver le même résultat ? Puis : des points répartis sur un cercle de rayon 1 autour de l'origine. Que fait la meilleure ACP à une composante ? Pourquoi un autoencodeur non linéaire à un latent pourrait-il faire bien mieux (pense à l'angle) ?

### Ex 25.9 — Compression réelle : le décodeur n'est pas gratuit 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer ce que coûte vraiment le stockage par un autoencodeur, décodeur compris, et le comparer aux images brutes.
**Prérequis :** Ex 25.1 · fiche §25.1, §25.3 · **Fil rouge :** MNIST · **Parcours :** R, M

On veut stocker 10 000 images MNIST. Une image brute occupe 784 octets (un octet par pixel). Un autoencodeur 784 → 20 → 784 (Ex 25.1) stocke le code de chaque image en 20 nombres `float32` (4 octets chacun), et il faut garder son **décodeur** pour reconstruire. Réponds dans ta copie, avec des ordres de grandeur.

1. La taille des images brutes, en kilo-octets (1 ko = 1 000 octets).
2. La taille des codes, puis celle du décodeur en `float32`, puis leur somme.
3. Le facteur de compression réel (taille brute divisée par codes plus décodeur). Pourquoi est-il bien plus petit que le facteur de l'Ex 25.1 f) ?
4. À partir de combien d'images le stockage par l'autoencodeur devient-il plus petit que les images brutes ?
5. Un format sans perte (PNG, ZIP) comprime déjà fortement les images MNIST, qui sont surtout noires. Que faudrait-il comparer, au-delà de la taille, pour choisir entre l'autoencodeur et un tel format ? Pourquoi, au fond, n'utilise-t-on presque jamais un autoencodeur pour archiver des fichiers ?

### Ex 25.10 — Pourquoi diffuser dans un espace latent : facteur de compression et coût 🧮 ★★ ⏱️ 15 min
**Objectif :** chiffrer ce que gagne un générateur d'images à travailler dans l'espace latent d'un autoencodeur.
**Prérequis :** Ex 25.3 · fiche §25.10 (mise à jour 2026) · **Parcours :** R, M

L'autoencodeur d'un modèle de diffusion latente transforme une image RGB de $512 \times 512$ pixels en un latent de $64 \times 64$ avec 4 canaux. Le réseau de diffusion fait ensuite, disons, 50 passes sur ce latent pour générer une image. Réponds dans ta copie, avec des ordres de grandeur.

1. Le nombre de valeurs d'une image, celui d'un latent, et le facteur de compression.
2. Si le coût d'une passe d'un réseau convolutif est à peu près proportionnel au nombre de positions de l'image (hauteur fois largeur), combien de fois moins cher est une passe sur le latent que sur l'image ?
3. Une couche d'**attention** compare chaque position à toutes les autres : son coût croît comme le carré du nombre de positions. Combien de fois moins chère est une telle couche sur le latent ?
4. Avec 50 passes de diffusion et un seul passage du décodeur à la fin, pourquoi le coût du décodeur compte-t-il peu dans le total ?
5. Les modèles plus récents prennent des latents de 16 canaux au lieu de 4, pour la même résolution. Que gagne-t-on, et que perd-on ? Pourquoi l'autoencodeur doit-il reconstruire presque sans perte visible, alors qu'un VAE du chapitre peut se contenter de chiffres flous ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 25.11 — Le VAE expliqué à un ami en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce qu'est un VAE et pourquoi il sait générer.
**Prérequis :** fiche §25.9 · **Parcours :** R

Dans ta copie, en 5 lignes au plus et sans formule, explique à un ami qui ne fait pas de deep learning ce qu'est un VAE et pourquoi, contrairement à un autoencodeur ordinaire, il peut inventer de nouvelles images. Prends une image du quotidien (autre que le peintre de cerisiers du livre). Puis relis-toi : as-tu dit ce que fait l'encodeur, ce que fait le décodeur, pourquoi on ajoute du hasard, et ce qui oblige les codes à rester dans une même région ?

### Ex 25.12 — Deepfakes par autoencodeurs : un encodeur, deux décodeurs ⚖️ ★★ ⏱️ 20 min
**Objectif :** comprendre le mécanisme des premiers deepfakes de visages, et juger leurs usages et leur encadrement.
**Prérequis :** fiche §25.7, §25.9, « au-delà du livre » · **Parcours :** R

Les premiers logiciels d'échange de visages entraînent **un seul encodeur** sur des milliers d'images de deux personnes A et B, et **deux décodeurs** : l'un reconstruit A à partir du code, l'autre B. Réponds dans ta copie.

1. Pourquoi l'encodeur doit-il être commun aux deux visages ? Qu'apprend son code (pense à ce qui est commun à A et B : position de la tête, expression, éclairage) ?
2. On encode une vidéo de A, image par image, et on décode avec le décodeur de B. Que voit-on, et pourquoi ?
3. Cite deux usages légitimes de cette technique, et deux usages nuisibles. Qui est lésé dans chaque usage nuisible ?
4. Depuis le 2 août 2026, le règlement européen sur l'IA (AI Act, article 50) impose à ceux qui diffusent un *deepfake* d'indiquer que le contenu a été généré ou manipulé artificiellement, avec des obligations allégées pour une œuvre manifestement artistique ou satirique. En France, la loi du 21 mai 2024 (dite SREN) a étendu au contenu généré par algorithme le délit de montage de l'article 226-8 du Code pénal, et créé un délit spécifique pour les montages à caractère sexuel. Ces règles suffisent-elles ? Que ne règlent-elles pas (pense aux auteurs anonymes, aux plateformes, aux victimes) ?
5. Tu travailles pour une entreprise qui veut un outil de doublage : la bouche de l'acteur suit le texte traduit. Quelles conditions poserais-tu (consentement, traçabilité, marquage des vidéos, accès au modèle) ?

### Ex 25.13 — Lire Kingma et Welling (2013), « Auto-Encoding Variational Bayes » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a introduit le VAE, et relier ses notations à la fiche.
**Prérequis :** Ex 25.5 · Ex 25.7 · fiche §25.9.2 · **Fil rouge :** MNIST · **Parcours :** M

Lis D. P. Kingma et M. Welling, « Auto-Encoding Variational Bayes », ICLR 2014 ([arXiv:1312.6114](https://arxiv.org/abs/1312.6114)) : le résumé, la section 2 (2.1 à 2.4), la section 3 (l'exemple du VAE), la section 5 et sa figure 2, et les annexes A et B. Réponds dans ta copie.

1. Résumé : quel problème les auteurs résolvent-ils, et quelles sont leurs deux contributions ?
2. Section 2.1 : qui sont $p_\theta(\mathbf{z})$, $p_\theta(\mathbf{x} \mid \mathbf{z})$ et $q_\phi(\mathbf{z} \mid \mathbf{x})$ ? Lequel est l'encodeur, lequel le décodeur ? Pourquoi le vrai posterior $p_\theta(\mathbf{z} \mid \mathbf{x})$ n'est-il pas calculable ?
3. Section 2.2 et équation (3) : écris la borne inférieure $\mathcal{L}$ avec ses deux termes. Relie-la à la loss de la fiche (le signe, le rôle de chaque terme).
4. Section 2.4 : comment les auteurs écrivent-ils $\mathbf{z}$ pour pouvoir dériver ? Relie leur notation à $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}$, et à l'Ex 25.5.
5. Annexe B : retrouve la forme fermée de la KL, et compare-la à celle de l'Ex 25.7 (attention au signe : les auteurs écrivent $-\mathrm{KL}$).
6. Sections 2.3 et 5 : combien de tirages de $\boldsymbol{\varepsilon}$ par exemple suffisent, selon les auteurs, quand le mini-batch est assez grand ? Sur quels datasets comparent-ils leur méthode, et que montre la figure 2 ? Annexe A : que montrent les figures des variétés à deux latents et des tirages au hasard ? Compare-les aux figures 25.68 et 25.70 du livre.

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 25.E1 — Autoencodeur ou ACP : quand préférer l'un ou l'autre ? 💼 ★★ ⏱️ 10 min
*Fiche §25.6 · parcours R*

« Pour réduire la dimension de vos données, quand utiliseriez-vous une ACP, et quand un autoencodeur ? »

### 25.E2 — Pourquoi un VAE sait-il générer et pas un autoencodeur classique ? 💼 ★★ ⏱️ 10 min
*Fiche §25.5.1, §25.9, §25.9.1 · parcours R*

« Pourquoi ne peut-on pas générer de nouvelles images en donnant des nombres au hasard au décodeur d'un autoencodeur classique ? Qu'est-ce qui change avec un VAE ? »

### 25.E3 — Expliquer l'astuce de reparamétrisation 💼 ★★ ⏱️ 10 min
*Fiche §25.9.2 · parcours R*

« Qu'est-ce que l'astuce de reparamétrisation, et pourquoi un VAE en a-t-il besoin ? »

### 25.E4 — Détecter des anomalies avec un autoencodeur : principe et pièges 💼 ★★ ⏱️ 10 min
*Fiche §25.5.4, « au-delà du livre » · parcours R*

« On vous demande de détecter des pièces défectueuses sur des photos d'une chaîne de production, avec très peu d'exemples de défauts. Comment utiliseriez-vous un autoencodeur, et quels pièges surveilleriez-vous ? »

### 25.E5 — Pourquoi Stable Diffusion travaille-t-il dans un espace latent ? 💼 ★★ ⏱️ 10 min
*Fiche §25.10 (mise à jour 2026) · parcours R*

« Les modèles de diffusion latente, comme Stable Diffusion, ne génèrent pas directement des pixels. Pourquoi, et quel rôle y joue un autoencodeur ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 25.14 | Prédire : un autoencodeur entraîné sur une seule image | 🔮 | ★ | 15 |
| 25.15 | Premier autoencodeur dense 784-20-784 en PyTorch | 📦 | ★★ | 25 |
| 25.16 | Autoencodeur profond 512-256-128-20 | 📦 | ★★ | 25 |
| 25.17 | Prédire : ce qu'un autoencodeur de chiffres fait d'un vêtement | 🔮 | ★★ | 15 |
| 25.18 | Reproduire la carte latente 2D et sa grille décodée (fig. 25.29-25.30) | 🎨 | ★★ | 30 |
| 25.19 | Secouer les variables latentes : bruit global, une seule composante, latents aléatoires | 🔬 | ★★ | 25 |
| 25.20 | Interpoler entre deux codes : linéaire et sphérique | 🔨 | ★★ | 20 |
| 25.21 | Mélanger dans l'espace des pixels ou dans l'espace latent | 🔬 | ★★ | 25 |
| 25.22 | Taille du goulot et profondeur : la courbe d'erreur de reconstruction | 🔬 | ★★★ | 40 |
| 25.23 | L'autoencodeur linéaire face à l'ACP de scikit-learn | 🔬 | ★★★ | 40 |
| 25.24 | Autoencodeur convolutif à 147 latents | 📦 | ★★ | 30 |
| 25.25 | Le débruiteur qui apprenait le bruit | 🐛 | ★★ | 25 |
| 25.26 | Autoencodeur débruiteur (bruit gaussien, écrêtage) | 📦 | ★★ | 30 |
| 25.27 | Sur-échantillonnage explicite ou convolutions à pas : temps et qualité | 🔬 | ★★★ | 35 |
| 25.28 | La reparamétrisation en NumPy : passe avant et passe arrière | 🔨 | ★★ | 25 |
| 25.29 | Divergence KL gaussienne et son gradient | 🔨 | ★★ | 20 |
| 25.30 | Un VAE en PyTorch : reconstruction + KL | 📦 | ★★ | 30 |
| 25.31 | Prédire : décoder des tirages N(0, I) avec un AE puis avec un VAE | 🔮 | ★★ | 15 |
| 25.32 | Un VAE est stochastique : sorties répétées, bruit relatif, interpolations | 🔬 | ★★ | 30 |
| 25.33 | Reproduire le nuage latent d'un VAE 2D et la grille −3…3 (fig. 25.67-25.68) | 🎨 | ★★ | 30 |
| 25.34 | Diagnostiquer des courbes reconstruction/KL (effondrement du posterior) | 📈 | ★★ | 15 |
| 25.35 | Exporter le décodeur comme générateur autonome et le tester | 🛠️ | ★★ | 20 |
| 25.36 | β-VAE : échanger la netteté contre la régularité | 🔬 | ★★★ | 40 |
| 25.37 | Détecter des anomalies par l'erreur de reconstruction | 🔬 | ★★★ | 45 |
| 25.38 | Un VAE entièrement en NumPy avec mylearn | 🔨 | ★★★★ | 150 |
| 25.39 | Défi débruitage Fashion-MNIST : le meilleur PSNR sous contrainte | 🏆 | ★★★★ | 120 |

🚀 *GPU Colab conseillé* pour le mode complet (`FAST_MODE = False`) de 25.27 et 25.39 ; le mode rapide tourne sur CPU.
