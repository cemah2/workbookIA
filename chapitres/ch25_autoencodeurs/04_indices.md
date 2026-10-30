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

Sur la carte, des codes voisins sont des images que l'encodeur juge semblables. Une zone vide est une région que le décodeur n'a jamais reçue. Le milieu de deux points est la moyenne de leurs coordonnées.

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

Un autoencodeur ne sait représenter que ce qui ressemble à ses données d'entraînement. Une erreur de reconstruction faible veut dire « facile à reconstruire », pas forcément « semblable aux données ».

</details>
<details><summary>Indice 3</summary>

a) Le réseau cherche un chiffre dans la photo : il rend une forme faite de morceaux de chiffres, (C). Pour b) et c), relis les trois pièges de la détection d'anomalies ; pour d), demande-toi à quoi ressemblent les bords des images MNIST ; pour e), demande-toi lesquelles viennent du même genre de données que l'entraînement.

</details>

### 25.Q12 — Les trois exigences d'un VAE et l'astuce de reparamétrisation

<details><summary>Indice 1</summary>

Relis les §25.9.1 et §25.9.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le livre liste trois propriétés pour pouvoir tirer des codes au hasard. L'encodeur d'un VAE donne les paramètres d'une gaussienne par latent. Le code s'écrit $z = \mu + \sigma\varepsilon$.

</details>
<details><summary>Indice 3</summary>

a) La première exigence est que tous les codes soient rassemblés dans une région connue (A en fait partie) ; retrouve les deux autres dans la liste du §25.9.1. Pour b), relis ce que produit l'encodeur ; pour c), demande-toi si un nombre tiré au hasard est une fonction de $\mu$ ; pour d), calcule $1{,}5 + 0{,}5 \times (-2)$ ; pour e), demande-toi si $\varepsilon$ change d'un passage à l'autre.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 25.R1 — Quelles augmentations garder pour des chiffres manuscrits ?

<details><summary>Indice 1</summary>

Relis le §24.4.6 de la fiche du ch. 24 (augmentation de données).

</details>
<details><summary>Indice 2</summary>

Une augmentation garde le label et reste plausible ; elle ne touche que le train. Pour chaque transformation, demande-toi si un humain lirait encore le même chiffre.

</details>
<details><summary>Indice 3</summary>

a) Un décalage de 2 pixels laisse un chiffre lisible et inchangé (A est plausible). Fais le même test pour une petite rotation, une symétrie gauche-droite (pense à un 7), une rotation de 180° (pense à un 6) et un léger zoom. Pour b), demande-toi ce que le test doit représenter ; pour c), demande-toi si chaque epoch voit les mêmes images bruitées.

</details>

### 25.R2 — Résumer une séquence entière dans l'état final d'un LSTM

<details><summary>Indice 1</summary>

Relis les formes de `nn.LSTM` (fiche du ch. 22, et le §24.5.8 du ch. 24).

</details>
<details><summary>Indice 2</summary>

`h_n` a la forme $(L, N, H)$, quel que soit `batch_first`. Une séquence a $T \times D$ valeurs.

</details>
<details><summary>Indice 3</summary>

a) Une couche, 16 séquences, 32 valeurs d'état : écris la forme dans l'ordre $(L, N, H)$. Pour b), divise $50 \times 8$ par 32 ; pour c), pense à un encodeur et un décodeur récurrents qui se passent un état.

</details>

### 25.R3 — La divergence KL, ce que mesure le terme de régularisation du VAE

<details><summary>Indice 1</summary>

Relis la fiche du ch. 6 (divergence KL).

</details>
<details><summary>Indice 2</summary>

$\mathrm{KL}(P \,\|\, Q) = \sum_i p_i \log_2(p_i/q_i)$ en bits ; en nats, on prend le logarithme népérien, ce qui revient à multiplier la valeur en bits par $\ln 2$.

</details>
<details><summary>Indice 3</summary>

a) Le premier terme vaut $0{,}5 \log_2(0{,}5/0{,}25) = 0{,}5$ ; calcule le second, $0{,}5\log_2(0{,}5/0{,}75)$, et additionne. Pour b), calcule $\mathrm{KL}(Q \,\|\, P)$ et compare ; pour c), que vaut $\log 1$ ? ; pour d), relis l'inégalité de Gibbs ; pour e), multiplie a) par $\ln 2$.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 25.1 — Compter les paramètres d'un autoencodeur à goulot

<details><summary>Indice 1</summary>

Relis le comptage des paramètres d'une couche dense (ch. 16).

</details>
<details><summary>Indice 2</summary>

Une couche dense de $n_{in}$ vers $n_{out}$ a $(n_{in} + 1)\,n_{out}$ paramètres. L'encodeur et le décodeur ont les mêmes poids en nombre, mais pas les mêmes biais : un biais par neurone de **sortie** de chaque couche.

</details>
<details><summary>Indice 3</summary>

a) $(784 + 1) \times 20$. Pour b), $(20 + 1) \times 784$ ; pour c), additionne ; pour d), additionne les quatre couches 784 → 512 → 256 → 128 → 20 ; pour e), fais de même pour 20 → 128 → 256 → 512 → 784 et ajoute d) ; pour f), divise 784 par 20 ; pour g), $(10\,000 + 1) \times 20 + (20 + 1) \times 10\,000$.

</details>

### Ex 25.2 — Mélanger deux cercles : pixels contre paramètres

<details><summary>Indice 1</summary>

Relis le §25.2.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le mélange paramétrique applique $(1 - t)\cdot\text{A} + t\cdot\text{B}$ à chaque paramètre. Le mélange de contenu applique la même formule à chaque pixel : un pixel vaut l'intensité d'un disque s'il est dedans, 0 sinon. Pour savoir si un pixel est dans un disque, compare sa distance au centre au rayon.

</details>
<details><summary>Indice 3</summary>

a) $0{,}75 \times (2, 2) + 0{,}25 \times (10, 6) = (1{,}5 + 2{,}5\,;\ 1{,}5 + 1{,}5) = $ `"(4, 3)"`. Pour b) et c), applique la même formule au rayon et à l'intensité ; pour d) et e), détermine d'abord dans quel(s) disque(s) se trouve chaque pixel ; pour f), demande-toi si les deux disques se recouvrent ; pour g), calcule le rayon à $t = 0{,}5$, puis $\pi r^2$ ; pour h), la moyenne de $\pi \times 1^2$ et $\pi \times 4^2$.

</details>

### Ex 25.3 — Suivre les formes dans l'autoencodeur convolutif

<details><summary>Indice 1</summary>

Relis les formules de taille et de paramètres d'une convolution (ch. 21), et le §25.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une convolution $3 \times 3$ avec `padding=1` garde la hauteur et la largeur et donne autant de canaux qu'elle a de filtres ; un `MaxPool2d(2)` divise la hauteur et la largeur par 2 et garde les canaux ; un `Upsample(scale_factor=2)` les multiplie par 2. Une convolution a $C_{out}(C_{in} k^2 + 1)$ paramètres ; pooling et sur-échantillonnage n'en ont aucun.

</details>
<details><summary>Indice 3</summary>

a) Après la première convolution : 16 canaux de $28 \times 28$ ; le pooling donne `"(64, 16, 14, 14)"`. Pour b) et c), continue étage par étage ; pour d), additionne $16(1 \cdot 9 + 1)$, $8(16 \cdot 9 + 1)$ et $3(8 \cdot 9 + 1)$ ; pour e), les deux convolutions du décodeur ; pour f), additionne ; pour g), divise 784 par le nombre de valeurs du code ; pour h), suis les deux poolings du débruiteur et multiplie les trois dimensions.

</details>

### Ex 25.4 — Retrouver 28×28 : padding et output_padding d'une convolution transposée

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §25.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Applique $H_{out} = (H_{in} - 1)\,s - 2p + k + p_{out}$ à chaque convolution transposée, et $\lfloor (H_{in} + 2p - k)/s \rfloor + 1$ à la convolution de f). Pour h), les poids de `ConvTranspose2d(C_in, C_out, k)` ont la forme `(C_in, C_out, k, k)`, et il y a un biais par canal de sortie.

</details>
<details><summary>Indice 3</summary>

a) $(7 - 1) \times 2 - 2 \times 1 + 3 + 0 = 13$. Pour b) à e), change les réglages dans la même formule ; pour f), calcule $\lfloor (7 + 2 - 3)/2 \rfloor + 1$ ; pour g), applique la formule de la convolution transposée à la hauteur f) ; pour h), $32 \times 16 \times 3 \times 3$ plus les biais.

</details>

### Ex 25.5 — Dériver à travers un tirage : l'astuce de reparamétrisation

<details><summary>Indice 1</summary>

Relis le §25.9.2 de la fiche et la règle de la chaîne (ch. 18).

</details>
<details><summary>Indice 2</summary>

Dérive $z = \mu + \sigma\varepsilon$ comme une fonction ordinaire, avec $\varepsilon$ constant. Pour $\sigma = e^{s/2}$, la dérivée de $e^{s/2}$ par rapport à $s$ est $\tfrac{1}{2}e^{s/2}$. Pour 5, développe $(\mu + \sigma\varepsilon)^2$ et utilise la linéarité de l'espérance.

</details>
<details><summary>Indice 3</summary>

1. $\partial z/\partial \mu = 1$ ; fais de même pour $\sigma$, puis multiplie par $\partial\sigma/\partial s$. 2. Multiplie par $g$. 3. Calcule $\sigma = \sqrt{4}$, puis $z$, puis $L$, puis $g = z - y$, puis applique 2. 4. Demande-toi ce qu'est une dérivée quand deux appels avec le même $\mu$ donnent deux nombres différents. 5. $\mathbb{E}[z^2] = \mu^2 + 2\mu\sigma\,\mathbb{E}[\varepsilon] + \sigma^2\mathbb{E}[\varepsilon^2]$ ; puis dérive $(\mu + \sigma\varepsilon)^2$ par rapport à $\mu$ et prends l'espérance.

</details>

### Ex 25.6 — Où tombent les tirages d'une gaussienne en grande dimension

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la norme d'un tirage gaussien, au §25.5.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

$\mathbb{E}\lVert\mathbf{z}\rVert^2$ est la somme des $\mathbb{E}[z_j^2]$. En dimension 2, utilise $P(\lVert\mathbf{z}\rVert \le r) = 1 - e^{-r^2/2}$. Pour le milieu, la variance de $(a + b)/2$ pour $a$ et $b$ indépendants est $(\text{Var}\,a + \text{Var}\,b)/4$.

</details>
<details><summary>Indice 3</summary>

a) Chacune des 20 coordonnées a $\mathbb{E}[z_j^2] = 1$, donc $\mathbb{E}\lVert\mathbf{z}\rVert^2 = 20$. Pour b), prends la racine ; pour c) et d), remplace $r$ par 1 puis 3 dans la formule ; pour e), résous $1 - e^{-r^2/2} = 0{,}5$ ; pour f), calcule la variance d'une coordonnée du milieu, puis multiplie par 20 ; pour g) et h), prends les racines.

</details>

### Ex 25.7 — La divergence KL entre N(μ, σ²) et N(0, 1)

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la densité et l'espérance d'une gaussienne (§25.9.1 de la fiche).

</details>
<details><summary>Indice 2</summary>

Écris $\ln q(z) - \ln p(z)$, puis prends l'espérance terme à terme sous $q$ : les constantes restent, $(z - \mu)^2$ devient $\sigma^2$, et $z^2$ devient $\mu^2 + \sigma^2$. Pour les valeurs, applique la formule obtenue en faisant attention à ce qui est donné : $\sigma$ ou $\sigma^2$.

</details>
<details><summary>Indice 3</summary>

1–3. $\mathbb{E}_q[\ln q(z)] = -\tfrac{1}{2}\ln(2\pi\sigma^2) - \tfrac{1}{2}$ ; calcule de même $\mathbb{E}_q[\ln p(z)]$ et soustrais. 4. Remplace $\sigma^2$ par $e^s$ et dérive. a) $\tfrac{1}{2}(2^2 + 1 - 1 - \ln 1)$. Pour b) à d) et f), remplace $\mu$ et $\sigma^2$ dans la formule (en c), $\sigma^2 = 0{,}01$) ; pour e), additionne les trois latents ; pour g), évalue la dérivée de 4 en $e^s = 5$ ; pour h), divise d) par $\ln 2$.

</details>

### Ex 25.8 — Un autoencodeur linéaire retrouve l'ACP (cas 2D guidé)

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « projection orthogonale et ACP » du §25.6 de la fiche, et les vecteurs propres (ch. 12).

</details>
<details><summary>Indice 2</summary>

Pour un $\mathbf{w}$ unitaire, $\mathbf{x} - (\mathbf{w}^\top\mathbf{x})\mathbf{w}$ est orthogonal à $\mathbf{w}$ : applique Pythagore. Une moyenne de $(\mathbf{w}^\top\mathbf{x}_i)^2$ s'écrit $\mathbf{w}^\top\mathbf{C}\,\mathbf{w}$. Pour 2, écris $\mathbf{w}$ dans la base orthonormée des vecteurs propres. Pour 3, les valeurs propres d'une matrice $2 \times 2$ vérifient $\lambda^2 - \operatorname{tr}\,\lambda + \det = 0$.

</details>
<details><summary>Indice 3</summary>

1. $\lVert\mathbf{x} - \hat{\mathbf{x}}\rVert^2 = \lVert\mathbf{x}\rVert^2 - (\mathbf{w}^\top\mathbf{x})^2$ ; prends la moyenne. 2. $\mathbf{w}^\top\mathbf{C}\,\mathbf{w} = \lambda_1 c_1^2 + \lambda_2 c_2^2$ avec $c_1^2 + c_2^2 = 1$ ; majore. 3. $\operatorname{tr} = 7$, $\det = 6$ ; résous, puis cherche un vecteur propre de la plus grande. 4. Calcule $((c\mathbf{a})^\top\mathbf{x})(\mathbf{b}/c)$ ; avec $d$ latents, remplace $c$ par une matrice inversible. 5. Pense à la moyenne pour les biais ; pour le cercle, calcule $\mathbf{C}$, puis pense à un code qui serait l'angle du point.

</details>

### Ex 25.9 — Compression réelle : le décodeur n'est pas gratuit

<details><summary>Indice 1</summary>

Relis l'Ex 25.1 et le §25.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Compte en octets : un pixel MNIST en occupe un, un `float32` en occupe quatre. Le décodeur est la couche 20 → 784 de l'Ex 25.1. Pour 4, écris l'inégalité « images brutes > codes + décodeur » en fonction du nombre d'images $N$.

</details>
<details><summary>Indice 3</summary>

1. $10\,000 \times 784$ octets. 2. $10\,000 \times 20 \times 4$ octets pour les codes, et le nombre de paramètres du décodeur fois 4. 3. Divise 1 par la somme de 2. 4. Résous $784\,N > 80\,N + 4 \times 16\,464$. 5. Pense à la qualité, au temps de calcul, à la dépendance au modèle et à la durée de vie d'une archive.

</details>

### Ex 25.10 — Pourquoi diffuser dans un espace latent : facteur de compression et coût

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

### Ex 25.11 — Le VAE expliqué à un ami en cinq lignes

<details><summary>Indice 1</summary>

Relis « L'essentiel en 10 lignes » (points 2, 9 et 10) et le §25.9 de la fiche.

</details>
<details><summary>Indice 2</summary>

Trouve une image où quelqu'un résume une chose en quelques réglages et quelqu'un d'autre la refabrique à partir de ces réglages (un portrait-robot, une recette, un réglage d'égaliseur). Puis ajoute les deux règles du VAE : un peu de hasard autour des réglages, et tous les réglages rassemblés au même endroit.

</details>
<details><summary>Indice 3</summary>

Plan en cinq lignes : (1) l'image du quotidien et ce qu'elle résume ; (2) qui résume (l'encodeur) et qui refabrique (le décodeur) ; (3) le hasard autour du résumé, et pourquoi il oblige des résumés voisins à donner des résultats voisins ; (4) la règle qui rassemble les résumés ; (5) comment on invente : tirer un résumé au hasard dans cette zone et le refabriquer.

</details>

### Ex 25.12 — Deepfakes par autoencodeurs : un encodeur, deux décodeurs

<details><summary>Indice 1</summary>

Relis l'encadré ⚖️ de la section « au-delà du livre » de la fiche, et le rôle du code dans un autoencodeur (§25.3).

</details>
<details><summary>Indice 2</summary>

Un code partagé par deux décodeurs ne peut garder que ce qui est commun aux deux personnes ; l'identité est dans chaque décodeur. Pour les usages, demande-toi chaque fois qui a consenti et qui est lésé. Pour les lois, distingue ce qu'elles imposent (transparence, sanctions) de ce qu'elles ne peuvent pas garantir (identifier, détecter, retirer vite).

</details>
<details><summary>Indice 3</summary>

1. Liste ce qui est commun à A et B dans une image (pose, expression, lumière). 2. Décoder le code de A avec le décodeur de B : décris le résultat image par image. 3. Deux usages où les personnes filmées consentent, deux où elles ne consentent pas. 4. Ce que disent l'article 50 de l'AI Act et l'article 226-8 du Code pénal, puis ce qui leur échappe (auteurs anonymes, vitesse de diffusion, plateformes, détection). 5. Consentement, contrat, marquage, accès, données.

</details>

### Ex 25.13 — Lire Kingma et Welling (2013), « Auto-Encoding Variational Bayes »

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

Commence par « la diffusion fait des dizaines de passes d'un gros réseau », puis : un autoencodeur régularisé par KL comprime environ 48 fois, chaque passe coûte beaucoup moins (l'attention encore plus), le décodeur ne sert qu'une fois mais plafonne la qualité.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices du notebook (25.14 à 25.39) seront ajoutés à la prochaine session de génération, avec les parties A à C du notebook.
