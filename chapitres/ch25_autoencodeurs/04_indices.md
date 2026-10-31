# 25 · Autoencodeurs et VAE — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 25.Q1 — Avec ou sans perte : le test de l'inversibilité

<details><summary>Indice 1</summary>

Relis le §25.2.1 de la fiche et son encadré 🔗 sur le ch. 6.

</details>
<details><summary>Indice 2</summary>

Pour chaque transformation, demande-toi si l'on peut retrouver **exactement** l'entrée à partir de la sortie ; la taille de la sortie ne compte pas. Pour d), divise la taille d'origine par la taille compressée. Pour e), compare le nombre de fichiers de $n$ bits au nombre de fichiers plus courts.

</details>
<details><summary>Indice 3</summary>

a) Écrire un entier en binaire se défait sans erreur (A est sans perte) ; arrondir au degré efface les décimales (B est avec perte). Applique le même test à C, D, E et F. Pour b), relis la définition ; pour c), demande-toi si l'erreur est nulle ; pour d), calcule $2\,400\,000 / 300\,000$ ; pour e), il y a $2^n$ fichiers de $n$ bits et $2^n - 1$ fichiers de moins de $n$ bits (vide compris).

</details>

### 25.Q2 — Pourquoi JPEG et MP3 battent un autoencodeur généraliste

<details><summary>Indice 1</summary>

Relis les §25.1 et §25.2.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Demande-toi d'où vient chaque algorithme : de règles écrites à partir de la perception humaine, ou d'un apprentissage sur un dataset ? Un code appris est adapté à son dataset. Pour e), un débit en bits par seconde, fois une durée en secondes, donne des bits ; il y a 8 bits par octet.

</details>
<details><summary>Indice 3</summary>

a) JPEG n'est entraîné sur rien : (A) est faux. Cherche, parmi (B), (C) et (D), ce qui décrit des règles conçues d'après la vision humaine. Pour b), demande-toi sur quoi le code a été appris ; pour c), relis ce que le livre dit des « JPG et MP3 » dès la première page ; pour d), relis les deux usages du §25.1 ; pour e), calcule $128\,000 \times 180 / 8$, puis exprime-le en millions.

</details>

### 25.Q3 — Mélange de contenu ou mélange de paramètres ?

<details><summary>Indice 1</summary>

Relis le §25.2.3 de la fiche et sa figure.

</details>
<details><summary>Indice 2</summary>

Le mélange de contenu combine les données elles-mêmes, le mélange paramétrique les nombres qui les décrivent. Pour b), numérote les lettres (A = 1, …, Z = 26) ; pour chaque paire, cherche le milieu entre les deux numéros ; s'il tombe entre deux lettres, prends la plus petite.

</details>
<details><summary>Indice 3</summary>

a) On combine les pixels eux-mêmes : demande-toi lequel des deux types de mélange cela décrit. b) Première lettre : L = 12 et S = 19, le milieu vaut 15,5, entre O (15) et P (16) ; la règle donne O. Fais de même pour U et O, N et I, E et R. Pour c), relis ce que donne le mélange des noms de fruits du livre ; pour d), demande-toi ce qui fait qu'un milieu de deux codes « ressemble » aux deux objets.

</details>

### 25.Q4 — Goulot, code, variables latentes : vrai ou faux

<details><summary>Indice 1</summary>

Relis le §25.3 de la fiche et son encadré « Le livre, corrigé ».

</details>
<details><summary>Indice 2</summary>

Un autoencodeur se définit par sa cible et par sa contrainte. Pour e), demande-toi ce que le réseau le plus simple ferait si rien ne le forçait à résumer.

</details>
<details><summary>Indice 3</summary>

a) La cible d'un autoencodeur est son entrée : il n'a besoin d'aucun label, donc a) est faux. Pour b) et c), relis la définition du goulot et du code ; pour d), relis le terme employé aujourd'hui ; pour e), pense à la fonction identité.

</details>

### 25.Q5 — Le réseau qui ne savait redessiner qu'une image

<details><summary>Indice 1</summary>

Relis l'histoire du tigre, au §25.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une image de $64 \times 64$ a $64 \times 64$ pixels ; une couche dense de $n_{in}$ vers $n_{out}$ a $n_{in} \times n_{out}$ poids et $n_{out}$ biais. Pour c), demande-toi ce qu'il reste dans le calcul quand l'entrée vaut 0. Pour d) et e), demande-toi ce qu'un réseau qui mémorise fait de ses images d'entraînement, et d'images différentes.

</details>
<details><summary>Indice 3</summary>

a) L'encodeur a $4\,096 \times 10$ poids ; ajoute ceux du décodeur. Pour b), compte un biais par neurone de sortie ; pour c), avec une entrée nulle, seules les sommes de biais restent : relis ce qu'elles contiennent ; pour d), relis ce que la reconstruction parfaite du tigre cachait ; pour e), cherche l'expérience qui distingue un réseau qui mémorise d'un réseau qui comprime.

</details>

### 25.Q6 — Profondeur ou nombre de latents : qui améliore quoi ?

<details><summary>Indice 1</summary>

Relis le §25.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le livre fait varier séparément le nombre de latents (20, 10, 3) et la profondeur (deux couches ou plusieurs) ; il en tire deux conditions.

</details>
<details><summary>Indice 3</summary>

a) Avec 3 latents, le code ne peut porter presque rien : les reconstructions se dégradent, ce n'est donc pas (A) ni (B). Pour b), relis ce que la profondeur change à 20 latents ; pour c), relis les deux conditions de la fin du §25.4 ; pour d), relis comment le livre présente la division par deux ; pour e), demande-toi ce qu'on peut faire de deux nombres qu'on ne peut pas faire de vingt.

</details>

### 25.Q7 — Du bruit dans l'espace latent d'un autoencodeur classique

<details><summary>Indice 1</summary>

Relis le §25.5.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Compare chaque bruit à l'échelle des latents (quelques centaines). Pour c), un chiffre dépend plus ou moins de chaque latent ; pour d) et e), relis ce que donnent des latents tirés au hasard.

</details>
<details><summary>Indice 3</summary>

a) Un bruit de ±1 sur des latents de plusieurs centaines est minuscule : on ne voit presque rien, (B). Pour b), compare ±100 à l'échelle des latents ; pour c), demande-toi si tous les chiffres utilisent le premier latent de la même façon ; pour d) et e), relis la figure 25.28.

</details>

### 25.Q8 — Lire une carte latente à deux dimensions

<details><summary>Indice 1</summary>

Relis les §25.5.2 et §25.5.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Sur la carte, des codes voisins sont des images que l'encodeur juge semblables. Demande-toi quels codes le décodeur a reçus pendant l'entraînement. Le milieu de deux points est la moyenne de leurs coordonnées.

</details>
<details><summary>Indice 3</summary>

a) Des chiffres différents avec des codes voisins : l'encodeur les distingue mal, (A). Pour b), demande-toi si le décodeur a appris quelque chose sur cette zone ; pour c), demande-toi ce qu'on mélange ; pour d), relis ce que donnent certains mélanges au §25.5.3 ; pour e), calcule $((2 + 8)/2,\ (6 + 0)/2)$.

</details>

### 25.Q9 — Autoencodeur linéaire et ACP

<details><summary>Indice 1</summary>

Relis le §25.6 de la fiche et ses deux encadrés.

</details>
<details><summary>Indice 2</summary>

Distingue le sous-espace (la droite, le plan) et les axes (une base particulière de ce sous-espace). Pour e), le code d'un autoencodeur linéaire d'encodeur unitaire $\mathbf{u}$ est le produit scalaire $\mathbf{u}^\top\mathbf{x}$.

</details>
<details><summary>Indice 3</summary>

a) Une rotation du code, compensée par le décodeur, donne la même reconstruction : l'autoencodeur ne peut pas retrouver des axes précis, ce n'est donc pas (A). Pour b), relis l'encadré « Le livre, corrigé » ; pour c), demande-toi qui centre les données ; pour d), pense aux surfaces courbes ; pour e), calcule $0{,}6 \times 5 + 0{,}8 \times 0$.

</details>

### 25.Q10 — Autoencodeurs convolutifs et débruiteurs

<details><summary>Indice 1</summary>

Relis les §25.7 et §25.8 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un débruiteur doit apprendre à retirer le bruit : demande-toi ce qu'il reçoit et ce qu'il doit rendre. Relis ce que le livre dit de la taille de son code, et ce que la fiche dit de l'écrêtage et des convolutions de stride 2.

</details>
<details><summary>Indice 3</summary>

a) Pour apprendre à retirer le bruit, le réseau doit recevoir l'image bruitée et viser l'image propre : (B). Pour b), relis la taille du code du débruiteur ; pour c), pense à l'intervalle des pixels et des sorties sigmoïdes ; pour d), relis la seconde version du débruiteur du livre ; pour e), demande-toi ce que le réseau a vu pendant l'entraînement.

</details>

### 25.Q11 — Une entrée jamais vue : un tigre chez les chiffres

<details><summary>Indice 1</summary>

Relis les §25.5.4 et §25.7.2 de la fiche, puis le paragraphe sur les anomalies de la section « au-delà du livre ».

</details>
<details><summary>Indice 2</summary>

Un autoencodeur ne sait représenter que ce qui ressemble à ses données d'entraînement. Demande-toi ce que mesure exactement une erreur de reconstruction faible.

</details>
<details><summary>Indice 3</summary>

a) Le réseau cherche un chiffre dans la photo : il rend une forme faite de morceaux de chiffres, (C). Pour b) et c), relis les trois pièges de la détection d'anomalies ; pour d), demande-toi à quoi ressemblent les bords des images MNIST ; pour e), demande-toi lesquelles viennent du même genre de données que l'entraînement.

</details>

### 25.Q12 — Les trois exigences d'un VAE et l'astuce de reparamétrisation

<details><summary>Indice 1</summary>

Relis les §25.9.1 et §25.9.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le livre liste trois propriétés pour pouvoir tirer des codes au hasard. Relis ce que l'encodeur d'un VAE produit pour chaque latent. Le code s'écrit $z = \mu + \sigma\varepsilon$.

</details>
<details><summary>Indice 3</summary>

a) La première exigence est que tous les codes soient rassemblés dans une région connue (A en fait partie) ; retrouve les deux autres dans la liste du §25.9.1. Pour b), relis ce que produit l'encodeur ; pour c), demande-toi si un nombre tiré au hasard est une fonction de $\mu$ ; pour d), calcule $1{,}5 + 0{,}5 \times (-2)$ ; pour e), demande-toi si $\varepsilon$ change d'un passage à l'autre.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 25.R1 — Ch. 24 : quelles augmentations garder pour des chiffres manuscrits ?

<details><summary>Indice 1</summary>

Relis le §24.4.6 de la fiche du ch. 24 (augmentation de données).

</details>
<details><summary>Indice 2</summary>

Une augmentation garde le label et reste plausible ; demande-toi aussi à quelles données on l'applique. Pour chaque transformation, demande-toi si un humain lirait encore le même chiffre.

</details>
<details><summary>Indice 3</summary>

a) Un décalage de 2 pixels laisse un chiffre lisible et inchangé (A est plausible). Fais le même test pour une petite rotation, une symétrie gauche-droite (pense à un 7), une rotation de 180° (pense à un 6) et un léger zoom. Pour b), demande-toi ce que le test doit représenter ; pour c), demande-toi si chaque epoch voit les mêmes images bruitées.

</details>

### 25.R2 — Ch. 22 : résumer une séquence entière dans l'état final d'un LSTM

<details><summary>Indice 1</summary>

Relis les formes de `nn.LSTM` (fiche du ch. 22, et le §24.5.8 du ch. 24).

</details>
<details><summary>Indice 2</summary>

`h_n` a la forme $(L, N, H)$, quel que soit `batch_first`. Une séquence a $T \times D$ valeurs.

</details>
<details><summary>Indice 3</summary>

a) Une couche, 16 séquences, 32 valeurs d'état : écris la forme dans l'ordre $(L, N, H)$. Pour b), divise $50 \times 8$ par 32 ; pour c), pense à un encodeur et un décodeur récurrents qui se passent un état.

</details>

### 25.R3 — Ch. 6 : la divergence KL, ce que mesure le terme de régularisation du VAE

<details><summary>Indice 1</summary>

Relis la fiche du ch. 6 (divergence KL).

</details>
<details><summary>Indice 2</summary>

$\mathrm{KL}(P \,\|\, Q) = \sum_i p_i \log_2(p_i/q_i)$ en bits ; en nats, on prend le logarithme népérien, ce qui revient à multiplier la valeur en bits par $\ln 2$.

</details>
<details><summary>Indice 3</summary>

a) Le premier terme vaut $0{,}5 \log_2(0{,}5/0{,}25) = 0{,}5$ ; calcule le second, $0{,}5\log_2(0{,}5/0{,}75)$, et additionne. Pour b), calcule $\mathrm{KL}(Q \,\|\, P)$ et compare ; pour c), que vaut $\log 1$ ? ; pour d), relis ce que le ch. 6 dit du signe de la KL ; pour e), multiplie a) par $\ln 2$.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 25.1 — Compter les paramètres d'un autoencodeur à goulot ✏️

<details><summary>Indice 1</summary>

Relis le comptage des paramètres d'une couche dense (ch. 16).

</details>
<details><summary>Indice 2</summary>

Une couche dense de $n_{in}$ vers $n_{out}$ a $(n_{in} + 1)\,n_{out}$ paramètres. L'encodeur et le décodeur ont les mêmes poids en nombre, mais pas les mêmes biais : un biais par neurone de **sortie** de chaque couche.

</details>
<details><summary>Indice 3</summary>

a) $(784 + 1) \times 20$. Pour b), $(20 + 1) \times 784$ ; pour c), additionne ; pour d), additionne les quatre couches 784 → 512 → 256 → 128 → 20 ; pour e), fais de même pour 20 → 128 → 256 → 512 → 784 et ajoute d) ; pour f), divise 784 par 20 ; pour g), $(10\,000 + 1) \times 20 + (20 + 1) \times 10\,000$.

</details>

### Ex 25.2 — Mélanger deux cercles : pixels contre paramètres ✏️

<details><summary>Indice 1</summary>

Relis le §25.2.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le mélange paramétrique applique $(1 - t)\cdot\text{A} + t\cdot\text{B}$ à chaque paramètre. Le mélange de contenu applique la même formule à chaque pixel : un pixel vaut l'intensité d'un disque s'il est dedans, 0 sinon. Pour savoir si un pixel est dans un disque, compare sa distance au centre au rayon.

</details>
<details><summary>Indice 3</summary>

a) $0{,}75 \times (2, 2) + 0{,}25 \times (10, 6) = (1{,}5 + 2{,}5\,;\ 1{,}5 + 1{,}5) = $ `"(4, 3)"`. Pour b) et c), applique la même formule au rayon et à l'intensité ; pour d) et e), détermine d'abord dans quel(s) disque(s) se trouve chaque pixel ; pour f), demande-toi si les deux disques se recouvrent ; pour g), calcule le rayon à $t = 0{,}5$, puis $\pi r^2$ ; pour h), la moyenne de $\pi \times 1^2$ et $\pi \times 4^2$.

</details>

### Ex 25.3 — Suivre les formes dans l'autoencodeur convolutif ✏️

<details><summary>Indice 1</summary>

Relis les formules de taille et de paramètres d'une convolution (ch. 21), et le §25.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une convolution $3 \times 3$ avec `padding=1` garde la hauteur et la largeur et donne autant de canaux qu'elle a de filtres ; un `MaxPool2d(2)` divise la hauteur et la largeur par 2 et garde les canaux ; un `Upsample(scale_factor=2)` les multiplie par 2. Une convolution a $C_{out}(C_{in} k^2 + 1)$ paramètres ; pooling et sur-échantillonnage n'en ont aucun.

</details>
<details><summary>Indice 3</summary>

a) Après la première convolution : 16 canaux de $28 \times 28$ ; le pooling donne `"(64, 16, 14, 14)"`. Pour b) et c), continue étage par étage ; pour d), additionne $16(1 \cdot 9 + 1)$, $8(16 \cdot 9 + 1)$ et $3(8 \cdot 9 + 1)$ ; pour e), les deux convolutions du décodeur ; pour f), additionne ; pour g), divise 784 par le nombre de valeurs du code ; pour h), suis les deux poolings du débruiteur et multiplie les trois dimensions.

</details>

### Ex 25.4 — Retrouver 28×28 : padding et output_padding d'une convolution transposée ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §25.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Applique $H_{out} = (H_{in} - 1)\,s - 2p + k + p_{out}$ à chaque convolution transposée, et $\lfloor (H_{in} + 2p - k)/s \rfloor + 1$ à la convolution de f). Pour h), les poids de `ConvTranspose2d(C_in, C_out, k)` ont la forme `(C_in, C_out, k, k)`, et il y a un biais par canal de sortie.

</details>
<details><summary>Indice 3</summary>

a) $(7 - 1) \times 2 - 2 \times 1 + 3 + 0 = 13$. Pour b) à e), change les réglages dans la même formule ; pour f), calcule $\lfloor (7 + 2 - 3)/2 \rfloor + 1$ ; pour g), applique la formule de la convolution transposée à la hauteur f) ; pour h), $32 \times 16 \times 3 \times 3$ plus les biais.

</details>

### Ex 25.5 — Dériver à travers un tirage : l'astuce de reparamétrisation ∂

<details><summary>Indice 1</summary>

Relis le §25.9.2 de la fiche et la règle de la chaîne (ch. 18).

</details>
<details><summary>Indice 2</summary>

Dérive $z = \mu + \sigma\varepsilon$ comme une fonction ordinaire, avec $\varepsilon$ constant. Pour $\sigma = e^{s/2}$, la dérivée de $e^{s/2}$ par rapport à $s$ est $\tfrac{1}{2}e^{s/2}$. Pour 5, développe $(\mu + \sigma\varepsilon)^2$ et utilise la linéarité de l'espérance.

</details>
<details><summary>Indice 3</summary>

1. $\partial z/\partial \mu = 1$ ; fais de même pour $\sigma$, puis multiplie par $\partial\sigma/\partial s$. 2. Multiplie par $g$. 3. Calcule $\sigma = \sqrt{4}$, puis $z$, puis $L$, puis $g = z - y$, puis applique 2. 4. Demande-toi ce qu'est une dérivée quand deux appels avec le même $\mu$ donnent deux nombres différents. 5. $\mathbb{E}[z^2] = \mu^2 + 2\mu\sigma\,\mathbb{E}[\varepsilon] + \sigma^2\mathbb{E}[\varepsilon^2]$ ; puis dérive $(\mu + \sigma\varepsilon)^2$ par rapport à $\mu$ et prends l'espérance.

</details>

### Ex 25.6 — Où tombent les tirages d'une gaussienne en grande dimension ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la norme d'un tirage gaussien, au §25.5.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

$\mathbb{E}\lVert\mathbf{z}\rVert^2$ est la somme des $\mathbb{E}[z_j^2]$. En dimension 2, utilise $P(\lVert\mathbf{z}\rVert \le r) = 1 - e^{-r^2/2}$. Pour le milieu, la variance de $(a + b)/2$ pour $a$ et $b$ indépendants est $(\text{Var}\,a + \text{Var}\,b)/4$.

</details>
<details><summary>Indice 3</summary>

a) Chacune des 20 coordonnées a $\mathbb{E}[z_j^2] = 1$, donc $\mathbb{E}\lVert\mathbf{z}\rVert^2 = 20$. Pour b), prends la racine ; pour c) et d), remplace $r$ par 1 puis 3 dans la formule ; pour e), résous $1 - e^{-r^2/2} = 0{,}5$ ; pour f), calcule la variance d'une coordonnée du milieu, puis multiplie par 20 ; pour g) et h), prends les racines.

</details>

### Ex 25.7 — La divergence KL entre N(μ, σ²) et N(0, 1) ∂

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la densité et l'espérance d'une gaussienne (§25.9.1 de la fiche).

</details>
<details><summary>Indice 2</summary>

Écris $\ln q(z) - \ln p(z)$, puis prends l'espérance terme à terme sous $q$ : les constantes restent, $(z - \mu)^2$ devient $\sigma^2$, et $z^2$ devient $\mu^2 + \sigma^2$. Pour les valeurs, applique la formule obtenue en faisant attention à ce qui est donné : $\sigma$ ou $\sigma^2$.

</details>
<details><summary>Indice 3</summary>

1–3. $\mathbb{E}_q[\ln q(z)] = -\tfrac{1}{2}\ln(2\pi\sigma^2) - \tfrac{1}{2}$ ; calcule de même $\mathbb{E}_q[\ln p(z)]$ et soustrais. 4. Remplace $\sigma^2$ par $e^s$ et dérive. a) $\tfrac{1}{2}(2^2 + 1 - 1 - \ln 1)$. Pour b) à d) et f), remplace $\mu$ et $\sigma^2$ dans la formule (en c), $\sigma^2 = 0{,}01$) ; pour e), additionne les trois latents ; pour g), évalue la dérivée de 4 en $e^s = 5$ ; pour h), divise d) par $\ln 2$.

</details>

### Ex 25.8 — Un autoencodeur linéaire retrouve l'ACP (cas 2D guidé) ∂

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « projection orthogonale et ACP » du §25.6 de la fiche, et les vecteurs propres (ch. 12).

</details>
<details><summary>Indice 2</summary>

Pour un $\mathbf{w}$ unitaire, $\mathbf{x} - (\mathbf{w}^\top\mathbf{x})\mathbf{w}$ est orthogonal à $\mathbf{w}$ : applique Pythagore. Une moyenne de $(\mathbf{w}^\top\mathbf{x}_i)^2$ s'écrit $\mathbf{w}^\top\mathbf{C}\,\mathbf{w}$. Pour 2, écris $\mathbf{w}$ dans la base orthonormée des vecteurs propres. Pour 3, les valeurs propres d'une matrice $2 \times 2$ vérifient $\lambda^2 - \operatorname{tr}\,\lambda + \det = 0$.

</details>
<details><summary>Indice 3</summary>

1. $\lVert\mathbf{x} - \hat{\mathbf{x}}\rVert^2 = \lVert\mathbf{x}\rVert^2 - (\mathbf{w}^\top\mathbf{x})^2$ ; prends la moyenne. 2. $\mathbf{w}^\top\mathbf{C}\,\mathbf{w} = \lambda_1 c_1^2 + \lambda_2 c_2^2$ avec $c_1^2 + c_2^2 = 1$ ; majore. 3. $\operatorname{tr} = 7$, $\det = 6$ ; résous, puis cherche un vecteur propre de la plus grande. 4. Calcule $((c\mathbf{a})^\top\mathbf{x})(\mathbf{b}/c)$ ; avec $d$ latents, remplace $c$ par une matrice inversible. 5. Pense à la moyenne pour les biais ; pour le cercle, calcule $\mathbf{C}$, puis pense à un code qui serait l'angle du point.

</details>

### Ex 25.9 — Compression réelle : le décodeur n'est pas gratuit 🧮

<details><summary>Indice 1</summary>

Relis l'Ex 25.1 et le §25.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Compte en octets : un pixel MNIST en occupe un, un `float32` en occupe quatre. Le décodeur est la couche 20 → 784 de l'Ex 25.1. Pour 4, écris l'inégalité « images brutes > codes + décodeur » en fonction du nombre d'images $N$.

</details>
<details><summary>Indice 3</summary>

1. $10\,000 \times 784$ octets. 2. $10\,000 \times 20 \times 4$ octets pour les codes, et le nombre de paramètres du décodeur fois 4. 3. Divise 1 par la somme de 2. 4. Résous $784\,N > 80\,N + 4 \times 16\,464$. 5. Pense à la qualité, au temps de calcul, à la dépendance au modèle et à la durée de vie d'une archive.

</details>

### Ex 25.10 — Pourquoi diffuser dans un espace latent : facteur de compression et coût 🧮

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §25.10 de la fiche.

</details>
<details><summary>Indice 2</summary>

Compte les valeurs (hauteur × largeur × canaux), puis les positions (hauteur × largeur). Un coût proportionnel aux positions se divise par leur rapport ; un coût quadratique, par le carré de ce rapport.

</details>
<details><summary>Indice 3</summary>

1. $512 \times 512 \times 3$ et $64 \times 64 \times 4$, puis leur rapport. 2. $(512 \times 512)/(64 \times 64)$. 3. Élève le rapport de 2 au carré. 4. Compare le nombre de passes du réseau de diffusion et du décodeur. 5. Pense à ce qu'on garde de plus avec 16 canaux, et à ce qui limite la qualité de l'image finale.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 25.11 — Le VAE expliqué à un ami en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis « L'essentiel en 10 lignes » (points 2, 9 et 10) et le §25.9 de la fiche.

</details>
<details><summary>Indice 2</summary>

Trouve une image où quelqu'un résume une chose en quelques réglages et quelqu'un d'autre la refabrique à partir de ces réglages (un portrait-robot, une recette, un réglage d'égaliseur). Puis ajoute les deux règles du VAE : un peu de hasard autour des réglages, et tous les réglages rassemblés au même endroit.

</details>
<details><summary>Indice 3</summary>

Plan en cinq lignes : (1) l'image du quotidien et ce qu'elle résume ; (2) qui résume (l'encodeur) et qui refabrique (le décodeur) ; (3) le hasard autour du résumé, et pourquoi il oblige des résumés voisins à donner des résultats voisins ; (4) la règle qui rassemble les résumés ; (5) comment on invente : tirer un résumé au hasard dans cette zone et le refabriquer.

</details>

### Ex 25.12 — Deepfakes par autoencodeurs : un encodeur, deux décodeurs ⚖️

<details><summary>Indice 1</summary>

Relis l'encadré ⚖️ de la section « au-delà du livre » de la fiche, et le rôle du code dans un autoencodeur (§25.3).

</details>
<details><summary>Indice 2</summary>

Un code partagé par deux décodeurs ne peut garder que ce qui est commun aux deux personnes ; l'identité est dans chaque décodeur. Pour les usages, demande-toi chaque fois qui a consenti et qui est lésé. Pour les lois, distingue ce qu'elles imposent (transparence, sanctions) de ce qu'elles ne peuvent pas garantir (identifier, détecter, retirer vite).

</details>
<details><summary>Indice 3</summary>

1. Liste ce qui est commun à A et B dans une image (pose, expression, lumière). 2. Décoder le code de A avec le décodeur de B : décris le résultat image par image. 3. Deux usages où les personnes filmées consentent, deux où elles ne consentent pas. 4. Ce que disent l'article 50 de l'AI Act et l'article 226-8 du Code pénal, puis ce qui leur échappe (auteurs anonymes, vitesse de diffusion, plateformes, détection). 5. Consentement, contrat, marquage, accès, données.

</details>

### Ex 25.13 — Lire Kingma et Welling (2013), « Auto-Encoding Variational Bayes » 📄

<details><summary>Indice 1</summary>

Lis d'abord le résumé et la section 2.1 en cherchant les trois lois $p_\theta(\mathbf{z})$, $p_\theta(\mathbf{x} \mid \mathbf{z})$ et $q_\phi(\mathbf{z} \mid \mathbf{x})$ ; relis le §25.9.2 de la fiche à côté.

</details>
<details><summary>Indice 2</summary>

L'équation (3) est la borne inférieure, à maximiser : compare-la à la loss de la fiche, à minimiser. La section 2.4 et l'annexe B contiennent les deux formules que tu as dérivées en ∂ 25.5 et 25.7. Le nombre de tirages $L$ est discuté à la fin de la section 2.3.

</details>
<details><summary>Indice 3</summary>

1. Cherche dans le résumé « two contributions ». 2. L'encodeur est le modèle de **reconnaissance** ; le posterior demande une intégrale sur $\mathbf{z}$. 3. $\mathcal{L} = -\mathrm{KL} + \mathbb{E}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$ : change le signe. 4. Cherche $g_\phi(\boldsymbol{\epsilon}, \mathbf{x})$ et l'exemple gaussien. 5. Compare le signe et les termes avec ta formule. 6. Cherche la phrase sur $L$ et $M$ ; les datasets sont nommés au début de la section 5 ; l'annexe A a deux figures.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 25.E1 — Autoencodeur ou ACP : quand préférer l'un ou l'autre ?

<details><summary>Indice 1</summary>

Relis le §25.6 de la fiche et son encadré 💼.

</details>
<details><summary>Indice 2</summary>

Structure : ce qu'ils ont en commun (le cas linéaire), les atouts de l'ACP, les cas où l'autoencodeur gagne, ce qu'il coûte.

</details>
<details><summary>Indice 3</summary>

Commence par « un autoencodeur linéaire retrouve le sous-espace de l'ACP », puis : ACP d'abord (exacte, rapide, interprétable, la variance expliquée), autoencodeur pour des données non linéaires ou pour débruiter, détecter, générer, avec son prix (entraînement, graine, interprétation).

</details>

### 25.E2 — Pourquoi un VAE sait-il générer et pas un autoencodeur classique ?

<details><summary>Indice 1</summary>

Relis les §25.5.1, §25.9 et §25.9.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Structure : le problème de l'espace latent classique (trous, mélanges), puis les deux changements du VAE (une loi par exemple, le terme KL), puis la génération, puis la limite.

</details>
<details><summary>Indice 3</summary>

Commence par « un code au hasard tombe presque toujours là où le décodeur n'a rien appris », puis : la gaussienne par exemple remplit les trous, la KL rassemble tout près du prior, on tire du prior et on décode, au prix d'images floues.

</details>

### 25.E3 — Expliquer l'astuce de reparamétrisation

<details><summary>Indice 1</summary>

Relis le §25.9.2 de la fiche et l'Ex 25.5.

</details>
<details><summary>Indice 2</summary>

Structure : pourquoi un tirage bloque le gradient, la formule, ce qu'elle change, les détails pratiques (log σ², un seul tirage).

</details>
<details><summary>Indice 3</summary>

Commence par « un nombre tiré au hasard n'est pas une fonction de la moyenne », puis : tirer ε à part, $z = \mu + \sigma\varepsilon$, les dérivées 1 et ε, log σ², un tirage par exemple, gradient sans biais.

</details>

### 25.E4 — Détecter des anomalies avec un autoencodeur : principe et pièges

<details><summary>Indice 1</summary>

Relis le paragraphe sur les anomalies de la section « au-delà du livre » de la fiche, et la ROC-AUC (ch. 3).

</details>
<details><summary>Indice 2</summary>

Structure : sur quoi entraîner, quel score, comment fixer le seuil et évaluer, quels pièges, à quoi comparer.

</details>
<details><summary>Indice 3</summary>

Commence par « j'entraîne sur des pièces normales seulement », puis : l'erreur de reconstruction comme score, un seuil sur une validation normale, la ROC-AUC sur les défauts connus, et trois pièges (un réseau trop capable, un petit défaut noyé dans la moyenne, les variations normales absentes de l'entraînement).

</details>

### 25.E5 — Pourquoi Stable Diffusion travaille-t-il dans un espace latent ?

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §25.10 de la fiche et l'Ex 25.10.

</details>
<details><summary>Indice 2</summary>

Structure : ce que coûte une diffusion sur des pixels, ce que fait l'autoencodeur, ce qu'on gagne (passes, attention), ce que le décodeur impose.

</details>
<details><summary>Indice 3</summary>

Commence par « la diffusion fait des dizaines de passes d'un gros réseau », puis : un autoencodeur régularisé par KL comprime fortement (calcule le facteur), chaque passe coûte beaucoup moins (l'attention encore plus), le décodeur ne sert qu'une fois mais plafonne la qualité.

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex 25.14 — Prédire : un autoencodeur entraîné sur une seule image 🔮

<details><summary>Indice 1</summary>

Relis le §25.3 de la fiche, puis suis le chemin d'une entrée dans le réseau : $\mathbf{h} = \text{ReLU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$, puis $\hat{\mathbf{x}} = \sigma(\mathbf{W}_2\mathbf{h} + \mathbf{b}_2)$.

</details>
<details><summary>Indice 2</summary>

Pour a), écris $\mathbf{h}$ pour $\mathbf{x} = \mathbf{0}$ : de quels paramètres dépend-il ? Pour b), compare les signes des pixels du T-shirt et de la botte, et ce qu'ils deviennent à travers $\mathbf{W}_1$ et la ReLU. Pour c), pour un $\mathbf{h}$ fixé, quelles valeurs $\sigma(\mathbf{W}_2\mathbf{h})$ peut-il prendre ?

</details>
<details><summary>Indice 3</summary>

L'entraînement a pu ranger l'image à deux endroits : dans les biais de sortie $\mathbf{b}_2$ (elle sortirait alors quelle que soit l'entrée, même noire), ou dans le produit $\mathbf{W}_2\mathbf{h}_{\text{botte}}$ (elle ne sortirait que pour des codes proches de celui de la botte). Écris pour chacune des trois questions ce que donnerait chaque hypothèse, choisis, puis lis la colonne « code nul → botte » : elle dit ce que portent les biais seuls.

</details>

### Ex 25.15 — Premier autoencodeur dense 784-20-784 en PyTorch 📦

<details><summary>Indice 1</summary>

C'est la boucle de ton `fit_18` du ch. 23, avec une seule différence : la cible est l'entrée. Relis le §25.3 de la fiche pour les couches.

</details>
<details><summary>Indice 2</summary>

`make_dense_ae_15` : deux `nn.Sequential` passés à `AutoEncoder`. `fit_ae_15` : convertis `x` en tenseur (`to_tensor`), crée l'optimiseur et **un** générateur avant la boucle des epochs ; dans chaque epoch, tire l'ordre avec `torch.randperm(len(x), generator=g)`, parcours les mini-batches, calcule `F.binary_cross_entropy(model(xb), xb)`, puis les trois lignes habituelles (`zero_grad`, `backward`, `step`). Accumule `loss.item() * len(xb)` et divise par $N$ en fin d'epoch.

</details>
<details><summary>Indice 3</summary>

```python
def fit_ae_15(model, x, epochs, lr=1e-3, batch_size=128, seed=0):
    device = next(model.parameters()).device
    x = to_tensor(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)          # once, before the epochs
    # for each epoch: order = torch.randperm(len(x), generator=g)
    #     for each mini-batch xb = x[order[i:i + batch_size]].to(device):
    #         loss = F.binary_cross_entropy(model(xb), xb)    # the target is the input
    #         zero_grad, backward, step; total += loss.item() * len(xb)
    #     losses.append(total / len(x))
```

</details>

### Ex 25.16 — Autoencodeur profond 512-256-128-20 📦

<details><summary>Indice 1</summary>

Écris la liste des tailles $[784, 512, 256, 128, 20]$ : l'encodeur relie chaque taille à la suivante, le décodeur parcourt la même liste à l'envers.

</details>
<details><summary>Indice 2</summary>

Construis une liste de couches : pour chaque paire `(n_in, n_out)` de tailles consécutives (`zip(sizes[:-1], sizes[1:])`), ajoute `nn.Linear(n_in, n_out)` puis `nn.ReLU()`. À la fin de l'encodeur, retire la dernière ReLU (le code n'a pas d'activation) ; à la fin du décodeur, remplace-la par une sigmoïde.

</details>
<details><summary>Indice 3</summary>

```python
def make_deep_ae_16(n_latent=20, hidden=(512, 256, 128)):
    sizes = [784, *hidden, n_latent]
    encoder = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        encoder += [nn.Linear(n_in, n_out), nn.ReLU()]
    encoder.pop()                                  # no activation on the code
    # the decoder: the same loop on sizes[::-1], then the last ReLU becomes nn.Sigmoid()
```

</details>

### Ex 25.17 — Prédire : ce qu'un autoencodeur de chiffres fait d'un vêtement 🔮

<details><summary>Indice 1</summary>

Relis le §25.5.4 de la fiche et le quiz 25.Q11 : l'autoencodeur rend toujours quelque chose qui ressemble à ce qu'il a appris.

</details>
<details><summary>Indice 2</summary>

Pour a), un chiffre a peu de pixels allumés, un vêtement beaucoup plus : à quoi ressemblera la reconstruction d'un pull, et combien de pixels seront faux ? Pour b), cherche les vêtements dont la forme est la plus proche d'un trait fin ou d'un petit motif, comme un chiffre. Pour c), une image noire se trompe sur **tous** les pixels allumés du vêtement : l'autoencodeur fait-il au moins mieux que cela ?

</details>
<details><summary>Indice 3</summary>

Pour a), compare l'ordre de grandeur des erreurs : un chiffre bien reconstruit se trompe de quelques centièmes en moyenne ; demande-toi si un vêtement mal reconstruit garde au moins sa silhouette. Pour b), classe mentalement les dix catégories par « surface allumée » et par ressemblance avec des traits. Pour c), compare « prédire 0 partout » à « prédire une forme claire au bon endroit ».

</details>

### Ex 25.18 — Reproduire la carte latente 2D et sa grille décodée (fig. 25.29-25.30) 🎨

<details><summary>Indice 1</summary>

Deux étapes : une grille de codes (`np.linspace` et `np.meshgrid`), puis une mosaïque d'images (un `reshape` et un `transpose`). Relis le §25.5.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour `grid_codes_18`, les $y$ vont de `ymax` à `ymin` (le haut d'une image est la première ligne) : `np.linspace(ymax, ymin, n)`. `np.meshgrid(xs, ys)` rend deux tableaux $(n, n)$ ; aplatis-les ligne par ligne (`ravel`) et colle-les en colonnes. Pour la mosaïque, les $n^2$ images forment un tableau $(n, n, 28, 28)$ : (ligne d'images, colonne d'images, ligne de pixels, colonne de pixels). Pour que chaque ligne de pixels de la mosaïque traverse toutes les images d'une ligne, il faut l'ordre (ligne d'images, ligne de pixels, colonne d'images, colonne de pixels).

</details>
<details><summary>Indice 3</summary>

```python
def grid_codes_18(lims, n):
    xmin, xmax, ymin, ymax = lims
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymax, ymin, n))
    # ...one column per coordinate, rows in the order of ravel()

# in plot_latent_map_18:
images = decode(model, grid_codes_18(lims, n).astype(np.float32)).reshape(n, n, 28, 28)
mosaic = images.transpose(0, 2, 1, 3).reshape(28 * n, 28 * n)
# left panel: one ax.scatter per digit; right panel: ax.imshow(mosaic, cmap="gray", extent=lims)
```

</details>

### Ex 25.19 — Secouer les variables latentes : bruit global, une seule composante, latents aléatoires 🔬

<details><summary>Indice 1</summary>

Un bruit uniforme se tire avec `rng.uniform(low, high, size=...)`. Travaille sur une **copie** des codes. Relis le §25.5.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

`add_noise_19` : `np.array(codes, dtype=np.float32)` fait la copie ; si `column` vaut `None`, ajoute un bruit de forme `codes.shape`, sinon un bruit de $N$ valeurs à la seule colonne `noisy[:, column]`. `noise_mse_19` : encode une fois, puis, pour chaque échelle, crée un **nouveau** générateur `np.random.default_rng(seed)`, bruite, décode, et moyenne `mse_per_image(..., x)`.

</details>
<details><summary>Indice 3</summary>

```python
def noise_mse_19(model, x, scales, seed=0, column=None):
    codes = encode(model, x)
    # for each scale: a NEW np.random.default_rng(seed), add_noise_19, decode, then the mean of mse_per_image(..., x)

# add_noise_19: noisy = np.array(codes, dtype=np.float32), then noisy += ... or noisy[:, column] += ...
```

</details>

### Ex 25.20 — Interpoler entre deux codes : linéaire et sphérique 🔨

<details><summary>Indice 1</summary>

Relis l'encadré slerp du §25.5.3 de la fiche. L'angle entre deux vecteurs vient du produit scalaire : $\cos\Omega = \frac{\mathbf{z}_0 \cdot \mathbf{z}_1}{\lVert\mathbf{z}_0\rVert\,\lVert\mathbf{z}_1\rVert}$.

</details>
<details><summary>Indice 2</summary>

Valide d'abord : deux vecteurs 1-D de même forme, `n_steps` entier (pas un `bool`) ≥ 2, une méthode connue. Calcule `t = np.linspace(0, 1, n_steps)[:, None]`. Pour la ligne droite, une diffusion suffit. Pour le slerp, calcule les deux normes ; si l'une est nulle, reviens à la ligne droite **avant** de diviser. Sinon, $\cos\Omega$ (borné dans $[-1, 1]$ par `np.clip`), $\Omega$ = `np.arccos`, et si $|\sin\Omega|$ est minuscule (de l'ordre de $10^{-6}$), ligne droite aussi. Termine en forçant la première ligne à `z0` et la dernière à `z1`.

</details>
<details><summary>Indice 3</summary>

```python
t = np.linspace(0, 1, n_steps)[:, None]
# linear: (1 - t) * z0 + t * z1
# slerp, after the checks on the norms and on sin(omega):
path = (np.sin((1 - t) * omega) * z0 + np.sin(t * omega) * z1) / np.sin(omega)
path[0], path[-1] = z0, z1          # exactly the ends
```

</details>

### Ex 25.21 — Mélanger dans l'espace des pixels ou dans l'espace latent 🔬

<details><summary>Indice 1</summary>

Trois petites fonctions : un fondu (comme la ligne droite de 25.20, mais sur les pixels), une interpolation de codes encadrée par `encode` et `decode`, et une erreur de reconstruction.

</details>
<details><summary>Indice 2</summary>

`blend_pixels_21` : `t = np.linspace(0, 1, n_steps)[:, None]`, puis $(1 - t)\,\mathbf{x}_0 + t\,\mathbf{x}_1$. `blend_codes_21` : encode les deux images d'un coup (`np.stack([x0, x1])`), passe les deux codes à `mylearn.vae.interpolate_latents`, convertis le chemin en `float32` et décode. `off_manifold_21` : `reconstruct` puis `mse_per_image`.

</details>
<details><summary>Indice 3</summary>

```python
def blend_codes_21(model, x0, x1, n_steps, method="linear"):
    z0, z1 = encode(model, np.stack([x0, x1]))
    path = mylearn.vae.interpolate_latents(z0, z1, n_steps, method=method)
    # ...decode the path (float32)
```

</details>

### Ex 25.22 — Taille du goulot et profondeur : la courbe d'erreur de reconstruction 🔬

<details><summary>Indice 1</summary>

Deux boucles imbriquées (les modèles, puis les tailles), et un dictionnaire dont les clés sont des couples `(nom, taille)`.

</details>
<details><summary>Indice 2</summary>

L'ordre compte pour retrouver le résultat de la vérification : `torch.manual_seed(seed)` **juste avant** `maker(size)` (les poids sont tirés à la création), puis `train_ae(model, x, epochs, seed=seed)`, puis `mse_per_image(reconstruct(model, XM_TEST), XM_TEST).mean()`.

</details>
<details><summary>Indice 3</summary>

```python
x = X_MNIST if x is None else x
results = {}
for name, maker in makers.items():
    for size in sizes:
        torch.manual_seed(seed)
        model = maker(size)
        # ...train, then results[(name, size)] = float(test MSE)
```

</details>

### Ex 25.23 — L'autoencodeur linéaire face à l'ACP de scikit-learn 🔬

<details><summary>Indice 1</summary>

Relis le ∂ 25.8 et le §25.6 de la fiche. Un autoencodeur linéaire est un `AutoEncoder` dont l'encodeur et le décodeur sont deux `nn.Linear` sans biais ; `train_ae` sait l'entraîner avec `loss="mse"`.

</details>
<details><summary>Indice 2</summary>

`fit_linear_ae_23` : `torch.manual_seed(seed)`, puis l'encodeur **avant** le décodeur (l'ordre des tirages), `train_ae(..., lr=lr, batch_size=batch_size, seed=seed, loss="mse")`, et le poids du décodeur : `model.decoder.weight`, de forme `(784, k)` (un `nn.Linear(k, 784)` range ses poids en `(sortie, entrée)`). `subspace_angles_deg_23` : `scipy.linalg.subspace_angles` rend des radians, déjà du plus grand au plus petit.

</details>
<details><summary>Indice 3</summary>

```python
def fit_linear_ae_23(xc, k, epochs=40, lr=3e-3, batch_size=256, seed=2523):
    torch.manual_seed(seed)
    model = AutoEncoder(nn.Linear(784, k, bias=False), nn.Linear(k, 784, bias=False))
    train_ae(model, xc, epochs, lr=lr, batch_size=batch_size, seed=seed, loss="mse")
    # ...return the decoder's weight as a NumPy array (784, k)
```

</details>

### Ex 25.24 — Autoencodeur convolutif à 147 latents 📦

<details><summary>Indice 1</summary>

Relis le ✏️ 25.3 : il a suivi toutes les formes de ce réseau. Il ne reste qu'à écrire les couches dans l'ordre du contrat.

</details>
<details><summary>Indice 2</summary>

Deux `nn.Sequential` : l'encodeur commence par `nn.Unflatten(1, (1, 28, 28))` (une ligne de 784 pixels devient une image à un canal), le décodeur finit par `nn.Flatten()`. Une convolution $3 \times 3$ de `padding=1` garde la taille, un `nn.MaxPool2d(2)` la divise par deux, un `nn.Upsample(scale_factor=2)` la double.

</details>
<details><summary>Indice 3</summary>

```python
encoder = nn.Sequential(nn.Unflatten(1, (1, 28, 28)),
                        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 16 x 14 x 14
                        # ...the 8 filters, the pooling, then the 3 filters of the code
                        )
decoder = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                        # ...the second upsampling, the last convolution, the sigmoid, nn.Flatten()
                        )
```

</details>

### Ex 25.25 — Le débruiteur qui apprenait le bruit 🐛

<details><summary>Indice 1</summary>

Relis le §25.8 de la fiche et le tableau des pièges : un débruiteur reçoit une image bruitée et doit rendre l'image **propre**.

</details>
<details><summary>Indice 2</summary>

Lis les diagnostics dans l'ordre : chacun pointe une ligne. La cible de la loss est-elle la bonne image ? Où le bruit est-il tiré : une fois pour toutes, ou à chaque mini-batch ? Que deviennent les pixels d'une image à laquelle on ajoute un bruit gaussien : restent-ils dans $[0, 1]$ ?

</details>
<details><summary>Indice 3</summary>

Pour le premier point : la ligne (G) passe `inputs` comme cible, alors qu'un débruiteur doit comparer sa sortie à l'image propre. Pour les deux autres, demande-toi quelle ligne fixe le bruit de chaque image pour toutes les epochs, et quelle ligne laisse passer des valeurs hors de $[0, 1]$. La version corrigée tire le bruit dans la boucle :

```python
clean = x[order[i:i + batch_size]]
inputs = torch.clamp(clean + sigma * torch.randn(clean.shape, generator=g), 0, 1)
# loss_fn(model(inputs on the device), ...)
```

</details>

### Ex 25.26 — Autoencodeur débruiteur (bruit gaussien, écrêtage) 📦

<details><summary>Indice 1</summary>

L'architecture est celle de 25.24 avec plus de filtres et sans goulot étroit. Le PSNR se calcule image par image : relis l'encadré PSNR du §25.8.

</details>
<details><summary>Indice 2</summary>

`make_denoiser_26` : comme 25.24, avec $c$ filtres partout, deux `MaxPool2d(2)` dans l'encodeur, et une convolution à $c$ filtres **avant** chaque `Upsample` dans le décodeur. `psnr_26` : la MSE de chaque image (moyenne sur l'axe 1), puis $10\log_{10}(1/\text{MSE})$, puis la moyenne des PSNR.

</details>
<details><summary>Indice 3</summary>

```python
def psnr_26(clean, images):
    mse = ((np.asarray(images, dtype=np.float64) - np.asarray(clean, dtype=np.float64)) ** 2).mean(axis=1)
    # ...10 log10(1 / mse) for each image, then the mean, as a float
```

</details>

### Ex 25.27 — Sur-échantillonnage explicite ou convolutions à pas : temps et qualité 🔬

<details><summary>Indice 1</summary>

Relis le ✏️ 25.4 : une convolution transposée de noyau 3, stride 2, padding 1 et `output_padding=1` double exactement la taille.

</details>
<details><summary>Indice 2</summary>

`make_strided_27` : deux `Conv2d(..., 3, stride=2, padding=1)` (28 → 14 → 7) à l'encodage, deux `ConvTranspose2d(..., 3, stride=2, padding=1, output_padding=1)` (7 → 14 → 28) au décodage, puis une convolution à un filtre. `compare_27` : pour chaque modèle, la graine, la création, l'entraînement, puis un dictionnaire de trois nombres.

</details>
<details><summary>Indice 3</summary>

```python
for name, maker in makers.items():
    torch.manual_seed(seed)
    model = maker()
    history = train_denoiser(model, X_FASHION[:N_DENOISE], epochs, seed=seed)
    # results[name] = {"params": ..., "sec_per_epoch": np.median(history["seconds"]),
    #                  "psnr": psnr(VAL_CLEAN_B, reconstruct(model, VAL_NOISY_B))}
```

</details>

### Ex 25.28 — La reparamétrisation en NumPy : passe avant et passe arrière 🔨

<details><summary>Indice 1</summary>

Relis le ∂ 25.5 : il donne $\partial z / \partial \mu$ et $\partial z / \partial \log\sigma^2$. Un générateur NumPy se fabrique à partir d'une graine avec `np.random.default_rng`.

</details>
<details><summary>Indice 2</summary>

Une petite fonction commune valide deux tableaux (même forme, 2-D, sans NaN) et en fait des copies `float64`. `reparameterize` : si `rng` n'est pas déjà un `np.random.Generator`, crée-le avec `np.random.default_rng(rng)` ; tire `eps = rng.standard_normal(mu.shape)` ; $\mathbf{z} = \boldsymbol{\mu} + e^{\log\boldsymbol{\sigma}^2/2} \odot \boldsymbol{\varepsilon}$. `reparameterize_backward` : la règle de la chaîne terme à terme, avec une **copie** de `dz` pour `dmu`.

</details>
<details><summary>Indice 3</summary>

```python
generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
eps = generator.standard_normal(mu.shape)
z = mu + np.exp(0.5 * log_var) * eps
# backward: dmu = dz.copy(); dlog_var = dz * (the derivative of z with respect to log_var, ∂ 25.5)
```

</details>

### Ex 25.29 — Divergence KL gaussienne et son gradient 🔨

<details><summary>Indice 1</summary>

La formule est celle du ∂ 25.7, avec $\sigma^2 = e^{s}$ et $s = \log\sigma^2$ ; on somme sur les latents (l'axe 1).

</details>
<details><summary>Indice 2</summary>

$\mathrm{KL} = \frac{1}{2}\sum_j (\mu_j^2 + e^{s_j} - 1 - s_j)$ : avec `np.expm1(s)` pour $e^{s} - 1$, le calcul reste précis quand $s$ est proche de 0. Les dérivées : par rapport à $\mu_j$, puis par rapport à $s_j$ (le ∂ 25.7 g l'a calculée en un point).

</details>
<details><summary>Indice 3</summary>

```python
kl = 0.5 * np.sum(mu ** 2 + np.expm1(log_var) - log_var, axis=1)     # shape (n,)
# grad: dmu = a copy of mu; dlog_var = 0.5 * (the derivative of e^s - 1 - s)
```

</details>

### Ex 25.30 — Un VAE en PyTorch : reconstruction + KL 📦

<details><summary>Indice 1</summary>

Le VAE est l'autoencodeur profond de 25.16 avec deux têtes au bout de l'encodeur (`to_mu`, `to_log_var`) et la reparamétrisation de 25.28 entre l'encodeur et le décodeur. La loss est celle du §25.9.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Dans `__init__`, appelle d'abord `super().__init__()`, garde `n_latent` et `tuple(hidden)`, puis construis l'encodeur (des `Linear` + ReLU sur `[784, *hidden]`), les deux têtes, et le décodeur (sur les tailles à l'envers, sans activation après la dernière couche). `forward` : `encode`, puis `torch.randn_like(mu)`, puis le décodeur. La loss : `F.binary_cross_entropy_with_logits(..., reduction="none")`, sommée sur l'axe 1 puis moyennée ; la KL de même. Ne convertis rien en nombre Python : la loss doit rester dérivable.

</details>
<details><summary>Indice 3</summary>

```python
def forward(self, x):
    mu, log_var = self.encode(x)
    z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
    return self.decoder(z), mu, log_var

# vae_loss_30:
rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()
# kl: the same pattern with 0.5 * (mu ** 2 + exp(log_var) - 1 - log_var)
```

</details>

### Ex 25.31 — Prédire : décoder des tirages N(0, I) avec un AE puis avec un VAE 🔮

<details><summary>Indice 1</summary>

Relis les §25.9 et §25.10 de la fiche, et ce que 25.18 et 25.19 t'ont montré de l'espace latent d'un autoencodeur classique.

</details>
<details><summary>Indice 2</summary>

Rien n'a demandé aux codes de `AE_DEEP` de ressembler à $\mathcal{N}(\mathbf{0}, \mathbf{I})$ : où tombent des tirages de cette loi par rapport à ses codes (25.19 a affiché leur étendue) ? Si tous les tirages tombent dans une même région de l'espace latent, que rendra le décodeur ? Pour `VAE_C`, la KL a rapproché les codes de cette loi : les tirages se répartissent-ils entre les chiffres ? Pour c), pense à la netteté des images d'un VAE entraîné 10 epochs sur 10 000 images.

</details>
<details><summary>Indice 3</summary>

Écris, pour chaque modèle, deux nombres que tu attends : la part du chiffre le plus prédit (20 % est deux fois la part d'un chiffre sur dix) et la part d'images « nettes » pour le classifieur. Puis compare tes deux colonnes : laquelle doit être la plus concentrée sur un chiffre ?

</details>

### Ex 25.32 — Un VAE est stochastique : sorties répétées, bruit relatif, interpolations 🔬

<details><summary>Indice 1</summary>

`repeat_32` refait le `forward` du VAE, pour une seule image et $k$ tirages ; `relative_noise_32` est un bruit multiplicatif.

</details>
<details><summary>Indice 2</summary>

`repeat_32` : mode évaluation, `torch.inference_mode()`, `encode` sur l'image en une ligne `(1, 784)`, puis `torch.manual_seed(seed)` **juste avant** `torch.randn(k, n_latent)` ; `mu` `(1, d)` se diffuse sur les $k$ lignes. `relative_noise_32` : $u$ par `np.random.default_rng(seed).uniform(-1, 1, size=mu.shape)`, puis $\mu\,(1 + \text{frac}\cdot u)$ en `float32`.

</details>
<details><summary>Indice 3</summary>

```python
with torch.inference_mode():
    mu, log_var = vae.encode(to_tensor(x).reshape(1, 784).to(DEVICE))
    torch.manual_seed(seed)
    z = mu + torch.exp(0.5 * log_var) * torch.randn(k, mu.shape[1]).to(DEVICE)
    # ...the sigmoid of vae.decoder(z), as a NumPy array
```

</details>

### Ex 25.33 — Reproduire le nuage latent d'un VAE 2D et la grille −3…3 (fig. 25.67-25.68) 🎨

<details><summary>Indice 1</summary>

C'est la figure de 25.18, avec les moyennes $\boldsymbol{\mu}$ au lieu des codes, des limites fixes et trois cercles.

</details>
<details><summary>Indice 2</summary>

`encode_mu(vae, x)` donne les moyennes. Les cercles : `ax.add_patch(plt.Circle((0, 0), r, fill=False))` pour $r$ = 1, 2, 3 ; `ax.set_aspect("equal")` pour qu'ils restent ronds. La grille : `grid_codes((-lim, lim, -lim, lim), n)`, puis `vae_images`, puis la même mosaïque qu'en 25.18.

</details>
<details><summary>Indice 3</summary>

```python
mu = encode_mu(vae, x)
images = vae_images(vae, grid_codes((-lim, lim, -lim, lim), n)).reshape(n, n, 28, 28)
mosaic = images.transpose(0, 2, 1, 3).reshape(28 * n, 28 * n)
# left: one scatter per digit, three circles, limits -lim..lim; right: imshow(mosaic, extent=(-lim, lim, -lim, lim))
```

</details>

### Ex 25.34 — Diagnostiquer des courbes reconstruction/KL (effondrement du posterior) 📈

<details><summary>Indice 1</summary>

Relis la section « Au-delà du livre » de la fiche : le $\beta$-VAE et l'effondrement du posterior, et ses signes.

</details>
<details><summary>Indice 2</summary>

Un $\beta$ plus grand pousse la KL vers le bas et la reconstruction vers le haut ; un $\beta$ qui part de 0 laisse d'abord la KL libre : rien ne la retient pendant les premières epochs. Pour a), cherche la KL qui finit presque à 0 avec zéro latent actif. Pour b), lis la dernière valeur de `"active"`. Pour c), la première et la dernière valeur de `"kl"` du run qui part le plus haut. Pour d), deux dernières valeurs de `"rec"`. Pour e), range les runs par KL finale.

</details>
<details><summary>Indice 3</summary>

a) La KL de chaque run à la dernière epoch : `{k: run["kl"][-1] for k, run in HISTORY_34.items()}` ; le plus petit, s'il est presque nul, désigne l'effondrement. Pour les autres questions, le même genre de lecture : `run["active"][-1]`, `run["kl"][0] - run["kl"][-1]`, une différence de `run["rec"][-1]`, et un classement : plus $\beta$ est grand, plus la KL finale est petite.

</details>

### Ex 25.35 — Exporter le décodeur comme générateur autonome et le tester 🛠️

<details><summary>Indice 1</summary>

Relis l'Ex 23.22 : on sauvegarde le `state_dict`, et on reconstruit l'architecture avant de le recharger. Ici, le fichier doit aussi dire **quelle** architecture reconstruire.

</details>
<details><summary>Indice 2</summary>

`export_decoder_35` : un `torch.save` d'un dictionnaire de trois entrées. `load_generator_35` : `torch.load(path, weights_only=True)`, puis les couches de 25.30 (sur `reversed(hidden)`), `load_state_dict`, `eval()`, et une fonction **imbriquée** `generate(n, seed=0)` qui garde le décodeur. Un test est une fonction sans argument qui appelle le code et vérifie par `assert` (forme, `dtype`, égalité avec `np.array_equal`).

</details>
<details><summary>Indice 3</summary>

```python
def generate(n, seed=0):
    z = torch.randn(n, saved["n_latent"], generator=torch.Generator().manual_seed(seed))
    with torch.inference_mode():
        images = torch.round(torch.sigmoid(decoder(z)) * 255)
    # ...as uint8, shape (n, 28, 28)

def test_generator_seed_35():
    generate = load_generator_35(GEN_PATH)
    assert np.array_equal(generate(3, seed=7), generate(3, seed=7))
    # ...and two different seeds must give different images
```

</details>

### Ex 25.36 — β-VAE : échanger la netteté contre la régularité 🔬

<details><summary>Indice 1</summary>

Une boucle sur les $\beta$, avec la même graine pour chacun, et les outils de référence de 25.30 (`VAE`, `train_vae`, `vae_eval`).

</details>
<details><summary>Indice 2</summary>

Pour chaque $\beta$ : `torch.manual_seed(seed)` puis `VAE(n_latent)`, `train_vae(..., beta=beta, seed=seed)`, `vae_eval(model, XF_TEST)` qui rend `(rec, kl, per_latent)` ; le nombre de latents actifs est `np.sum(per_latent > 0.05)`.

</details>
<details><summary>Indice 3</summary>

```python
for beta in betas:
    torch.manual_seed(seed)
    model = VAE(n_latent)
    # ...train with this beta, evaluate on XF_TEST with vae_eval, count the active latents, store the four entries
```

</details>

### Ex 25.37 — Détecter des anomalies par l'erreur de reconstruction 🔬

<details><summary>Indice 1</summary>

Relis la section « anomalies » de la fiche et ta `roc_auc` du ch. 3 : elle prend des labels vrais (0 ou 1) et des scores.

</details>
<details><summary>Indice 2</summary>

`fit_without_37` : un masque `Y_MNIST != digit` sur `X_MNIST`, la graine avant `make_deep_ae(20)`, puis `train_ae`. `scores_37` : `reconstruct` puis `mse_per_image`. `auc_37` : des labels 0 pour les normaux, 1 pour les anomalies (`np.zeros`, `np.ones`), les scores concaténés dans le même ordre.

</details>
<details><summary>Indice 3</summary>

```python
# fit_without_37: torch.manual_seed(seed), make_deep_ae(20), then train_ae on X_MNIST[Y_MNIST != digit]
# auc_37: y_true = 0 for each normal score, 1 for each anomaly, in the order of np.concatenate([normal, anomalies])
```

</details>

### Ex 25.38 — Un VAE entièrement en NumPy avec mylearn 🔨

<details><summary>Indice 1</summary>

Dessine le graphe de calcul : $\mathbf{x} \to$ encodeur $\to [\boldsymbol{\mu}, \log\boldsymbol{\sigma}^2] \to \mathbf{z} \to$ décodeur $\to$ logits $\to$ reconstruction, et une seconde branche $[\boldsymbol{\mu}, \log\boldsymbol{\sigma}^2] \to$ KL. La passe arrière remonte ce graphe ; $\boldsymbol{\mu}$ et $\log\boldsymbol{\sigma}^2$ reçoivent deux gradients qui s'additionnent.

</details>
<details><summary>Indice 2</summary>

Passe avant : `mlp_forward(..., activation=_relu_38, return_cache=True)` pour l'encodeur, les $d$ premières colonnes de sa sortie sont $\boldsymbol{\mu}$ ; puis $\mathbf{z}$, puis le décodeur. La loss de reconstruction sommée sur les pixels vaut 784 fois la moyenne que rend `binary_cross_entropy_with_logits`, et son gradient aussi. Passe arrière : `mlp_backward` du décodeur **avec** `return_input_grad=True` donne $\partial L / \partial \mathbf{z}$ ; `reparameterize_backward` le ramène sur $\boldsymbol{\mu}$ et $\log\boldsymbol{\sigma}^2$ ; ajoute `gaussian_kl_grad` divisé par $n$ (la KL est moyennée sur le batch) ; concatène, et `mlp_backward` de l'encodeur. Pour l'entraînement, Adam reçoit la liste à plat des paramètres, et `step` la liste à plat des gradients, dans le même ordre.

</details>
<details><summary>Indice 3</summary>

```python
mean_bce, dlogits = backward.binary_cross_entropy_with_logits(logits, x)
rec, dlogits = 784 * mean_bce, 784 * dlogits
grads_dec, dz = backward.mlp_backward(dlogits, cache_dec, params["dec"],
                                      activation_derivative=_relu_derivative_38, return_input_grad=True)
dmu, dlog_var = vae.reparameterize_backward(dz, eps, log_var)
# ...add the gradient of the mean KL, concatenate [dmu, dlog_var] on axis 1, then the encoder's mlp_backward
```

</details>

### Ex 25.39 — Défi débruitage Fashion-MNIST : le meilleur PSNR sous contrainte 🏆

<details><summary>Indice 1</summary>

Pars du débruiteur de 25.26 et note son PSNR de validation : c'est ta baseline. Change une chose à la fois, et garde un œil sur le chronomètre (25.27 a mesuré le coût de chaque choix).

</details>
<details><summary>Indice 2</summary>

Trois leviers, dans l'ordre où ils coûtent le moins : plus d'images d'entraînement (`X_FASHION` en a 10 000 en mode rapide), un learning rate un peu plus grand avec quelques epochs de plus, puis l'architecture. Un débruiteur n'a pas besoin d'un goulot étroit : réduire la résolution perd des détails qu'il faut ensuite réinventer. Pense à un réseau qui garde toute ou une partie de la résolution, et à la largeur qu'autorise la limite de 100 000 paramètres.

</details>
<details><summary>Indice 3</summary>

Un squelette qui garde la résolution (tu choisis $c$ et le nombre de couches, sous 100 000 paramètres) :

```python
layers = [nn.Unflatten(1, (1, 28, 28)), nn.Conv2d(1, c, 3, padding=1), nn.ReLU()]
# ...a few nn.Conv2d(c, c, 3, padding=1) + nn.ReLU()
model = AutoEncoder(nn.Sequential(*layers), nn.Sequential(nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid(), nn.Flatten()))
```

Puis `train_denoiser(model, X_FASHION, epochs, lr=...)` en comptant le temps d'une epoch avant d'en choisir le nombre.

</details>
