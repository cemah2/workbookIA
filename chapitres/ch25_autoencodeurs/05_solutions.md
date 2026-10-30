# 25 · Autoencodeurs et VAE — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels, des exercices ✏️ 25.1 à 25.4 et 25.6 et de la dérivation ∂ 25.7 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 25.Q1 — Avec ou sans perte : le test de l'inversibilité
a) **« AC »** : le binaire et le ZIP se défont exactement ; arrondir (B), garder une lettre sur deux (D), le JPEG (E) et le passage en gris (F) perdent une information qu'aucun calcul ne peut retrouver. b) **Vrai** : c'est la définition, le test de l'inversibilité. c) **B** : l'erreur est petite mais pas nulle, l'image d'origine n'est pas retrouvée exactement. d) **8** $= 2\,400\,000 / 300\,000$. e) **Faux** : il y a moins de fichiers courts que de fichiers longs ; un compresseur sans perte qui raccourcit certains fichiers doit en allonger d'autres (ch. 6, Huffman).
**À retenir** : sans perte = inversible ; la taille n'y est pour rien. Un compresseur sans perte gagne sur les fichiers fréquents, jamais sur tous.

### 25.Q2 — Pourquoi JPEG et MP3 battent un autoencodeur généraliste
a) **C** : JPEG n'apprend rien ; ses règles viennent de l'étude des images et de la vision. b) **C** : le code de l'autoencodeur est adapté aux radios ; un paysage tombe hors de ce qu'il sait représenter. c) **Faux** : le livre dit l'inverse dès la première page ; JPEG et MP3 « n'ont rien à craindre ». d) **« AC »** : débruiter et réduire la dimension. e) **2,88** : $128\,000 \times 180 = 23\,040\,000$ bits, divisés par 8, soit $2\,880\,000$ octets.
**À retenir** : un algorithme conçu pour un domaine bat un autoencodeur généraliste sur ce domaine ; l'autoencodeur s'adapte aux données pour lesquelles personne n'a conçu d'algorithme.

### 25.Q3 — Mélange de contenu ou mélange de paramètres ?
a) **A** : on mélange les pixels eux-mêmes. b) **« ORKK »** : L (12) et S (19) donnent 15 ou 16, on prend la plus proche du début, O ; U (21) et O (15) donnent R (18) ; N (14) et I (9) donnent K (11) ou L, donc K ; E (5) et R (18) donnent K ou L, donc K. c) **Faux** : « ORKK » ne se prononce pas et ne veut rien dire ; l'ordre de l'alphabet n'a aucun rapport avec le sens des mots. d) **C** : il faut que des objets proches aient des codes proches, sans trous entre eux.
**À retenir** : le mélange paramétrique n'a de sens que si l'espace des paramètres est bien rangé ; c'est ce qu'on attendra d'un espace latent.

### 25.Q4 — Goulot, code, variables latentes : vrai ou faux
a) **Faux** : la cible est l'entrée elle-même. b) **Vrai**. c) **Vrai** : le goulot est la couche la plus étroite ; ses sorties sont le code. d) **C** : auto-supervisé ; « semi-supervisé », le mot du livre, désigne un mélange de données avec et sans labels. e) **Vrai** : il lui suffit d'apprendre l'identité, sans rien résumer ; c'est pour cela qu'on force un goulot (ou une autre contrainte : du bruit, de la parcimonie).
**À retenir** : un autoencodeur se définit par sa cible (l'entrée) et par sa contrainte (le goulot) ; sans contrainte, il n'apprend rien d'utile.

### 25.Q5 — Le réseau qui ne savait redessiner qu'une image
a) **81 920** $= 4\,096 \times 10 + 10 \times 4\,096$ (une image de $64 \times 64$ a $4\,096$ pixels). b) **4 096** : un biais par neurone de sortie, un neurone par pixel. c) **C** : avec une entrée nulle, le code ne vaut que les biais de l'encodeur, et la sortie vaut surtout ceux du décodeur, qui contiennent l'image. d) **Faux** : un réseau qui mémorise reconstruit parfaitement ses images d'entraînement. e) **C** : seule une image très différente montre si le réseau ressort toujours la même chose.
**À retenir** : une reconstruction parfaite ne prouve rien ; il faut évaluer sur des images jamais vues.

### 25.Q6 — Profondeur ou nombre de latents : qui améliore quoi ?
a) **C** : 3 nombres ne peuvent pas porter un chiffre, les reconstructions deviennent des taches. b) **B** : plus de couches, c'est plus de capacité pour calculer et décoder les mêmes 20 latents. c) **« AC »**. d) **Faux** : c'est une habitude commode, rien de plus. e) **B** : 2 latents se dessinent sur une page, au prix de reconstructions très floues.
**À retenir** : deux leviers indépendants, le nombre de latents (ce que le code peut porter) et la capacité (ce que le réseau sait en faire).

### 25.Q7 — Du bruit dans l'espace latent d'un autoencodeur classique
a) **B** : un bruit de ±1 est minuscule devant des latents de plusieurs centaines. b) **A** : ±100 est de l'ordre des latents eux-mêmes. c) **A** : chaque chiffre dépend plus ou moins de chaque latent ; certains sont presque intacts. d) **Faux** : la plupart des régions ne correspondent à rien (25.5.1, 25.5.2). e) **A** : des taches blanches (fig. 25.28).
**À retenir** : un bruit se juge en proportion de l'échelle des latents ; un autoencodeur classique ne dit pas où sont les codes valides.

### 25.Q8 — Lire une carte latente à deux dimensions
a) **A**. b) **B** : aucun exemple n'a de code dans cette zone ; le décodeur produit quelque chose, mais sans garantie de sens. c) **B** : c'est un mélange paramétrique, avec les latents pour paramètres. d) **Faux** : la ligne droite peut traverser des zones vides ou mêlées ; le livre trouve des formes sans signification. e) **« (5, 3) »** : $((2 + 8)/2,\ (6 + 0)/2)$.
**À retenir** : une carte latente montre des régions, des chevauchements et des trous ; ce sont les trous qui empêchent de générer.

### 25.Q9 — Autoencodeur linéaire et ACP
a) **B** : le sous-espace, pas les axes (∂ 25.8). b) **Faux** : seul l'autoencodeur linéaire à l'erreur quadratique retrouve l'ACP ; avec des ReLU, il peut suivre des surfaces courbes. c) **Vrai** : l'ACP centre les données ; un autoencodeur linéaire sans biais sur des données non centrées ne peut pas le faire. d) **A**. e) **3** $= 0{,}6 \times 5 + 0{,}8 \times 0$ : le code est la coordonnée de $\mathbf{x}$ sur la droite ; la reconstruction serait $3\,\mathbf{u} = (1{,}8\,;\ 2{,}4)$.
**À retenir** : autoencodeur linéaire + erreur quadratique = sous-espace de l'ACP ; la non-linéarité est ce qui le rend plus puissant.

### 25.Q10 — Autoencodeurs convolutifs et débruiteurs
a) **B**. b) **Faux** : le code du débruiteur du livre a plus de valeurs que l'image ; le but n'est pas de comprimer mais de retirer le bruit. c) **A** : une image a des pixels dans $[0, 1]$, et la sortie sigmoïde du réseau aussi ; un pixel bruité à 1,3 serait une cible impossible. d) **B**. e) **Faux** : il a appris ce qu'est un chiffre plausible, pas une photo.
**À retenir** : débruiteur = entrée bruitée, cible propre, bruit neuf à chaque mini-batch, écrêtage.

### 25.Q11 — Une entrée jamais vue : un tigre chez les chiffres
a) **C** : il cherche un chiffre là où il n'y en a pas. b) **Vrai** : c'est le principe de la détection d'anomalies. c) **Faux** : une image presque vide, simple, peut être bien reconstruite sans ressembler aux données. d) **A**. e) **« BC »**.
**À retenir** : l'erreur de reconstruction est un signal utile, pas une garantie dans les deux sens.

### 25.Q12 — Les trois exigences d'un VAE et l'astuce de reparamétrisation
a) **« ABC »**. b) **B** : une moyenne et une variance par latent ; on fait sortir $\log\sigma^2$, qui peut être négatif. c) **Faux** : un tirage écrit tel quel n'est pas une fonction de $\mu$ et $\sigma$ ; il faut le reformuler. d) **0,5** $= 1{,}5 + 0{,}5 \times (-2)$. e) **Faux** : $\boldsymbol{\varepsilon}$ est tiré à chaque passage, la sortie change un peu.
**À retenir** : le VAE remplace le code par une loi ; la reparamétrisation sort le hasard du chemin du gradient.

<a id="rappels"></a>

## 🔁 Rappels

### 25.R1 — Quelles augmentations garder pour des chiffres manuscrits ?
a) **« ABE »** : un léger décalage, une petite rotation et un léger zoom gardent le chiffre ; la symétrie gauche-droite (C) transforme un 2 ou un 7 en signe qui n'existe pas, et la rotation de 180° (D) change un 6 en 9. b) **Faux** : le test doit représenter les vraies données. c) **Vrai** : chaque epoch voit des versions bruitées différentes des mêmes images.
**À retenir** : une augmentation garde le label et le réalisme, et ne touche que le train (ch. 24).

### 25.R2 — Résumer une séquence entière dans l'état final d'un LSTM
a) **« (1, 16, 32) »** : $(L, N, H)$, une couche, 16 séquences, 32 valeurs d'état ; `batch_first` ne change pas `h_n` (ch. 22, ch. 24). b) **12,5** $= (50 \times 8)/32$. c) **Vrai** : c'est l'autoencodeur de séquences (*sequence-to-sequence*), où l'état final de l'encodeur sert de code et initialise le décodeur.
**À retenir** : l'état final d'un RNN est déjà un code de taille fixe pour une séquence de longueur quelconque.

### 25.R3 — La divergence KL, ce que mesure le terme de régularisation du VAE
a) **0,21** : $0{,}5\log_2(0{,}5/0{,}25) + 0{,}5\log_2(0{,}5/0{,}75) = 0{,}5 - 0{,}2925 = 0{,}2075$ bit. b) **Faux** : $\mathrm{KL}(Q \,\|\, P) = 0{,}25\log_2 0{,}5 + 0{,}75\log_2 1{,}5 = 0{,}1887$ bit, différente. c) **Vrai** : chaque terme vaut $p\log 1 = 0$. d) **Vrai** : l'inégalité de Gibbs (ch. 6). e) **0,14** : $0{,}2075 \times \ln 2 = 0{,}1438$ nat.
**À retenir** : la KL est positive, nulle seulement entre deux lois identiques, et pas symétrique ; dans un VAE, on la calcule entre $q(\mathbf{z} \mid \mathbf{x})$ et le prior.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 25.1 — Compter les paramètres d'un autoencodeur à goulot ✏️
**Réponses** : a) **15 700** ; b) **16 464** ; c) **32 164** ; d) **568 724** ; e) **1 138 212** ; f) **39,2** ; g) **410 020**.
**Démarche** : une couche dense de $n_{in}$ vers $n_{out}$ a $(n_{in} + 1)\,n_{out}$ paramètres (ch. 16). a) $785 \times 20 = 15\,700$. b) $21 \times 784 = 16\,464$. c) leur somme. d) $785 \times 512 + 513 \times 256 + 257 \times 128 + 129 \times 20 = 401\,920 + 131\,328 + 32\,896 + 2\,580 = 568\,724$. e) Le décodeur a les mêmes poids, mais d'autres biais (sur 128, 256, 512 et 784 neurones) : $21 \times 128 + 129 \times 256 + 257 \times 512 + 513 \times 784 = 2\,688 + 33\,024 + 131\,584 + 402\,192 = 569\,488$, et $568\,724 + 569\,488 = 1\,138\,212$. f) $784 / 20 = 39{,}2$. g) $10\,001 \times 20 + 21 \times 10\,000 = 200\,020 + 210\,000 = 410\,020$.
**Dans ta copie** : le livre compte les poids seuls, $2 \times 20 \times 10\,000 = 400\,000$ ; les biais en ajoutent $10\,020$. Avec $10\,000$ biais de sortie, un par pixel, le réseau peut ranger l'image entière dans ces biais : il la ressort quelle que soit l'entrée, sans rien comprimer.
**Erreurs fréquentes** : oublier les biais ; donner au décodeur les biais de l'encodeur (20 au lieu de 784) ; doubler l'encodeur pour l'autoencodeur profond (les biais diffèrent) ; diviser 28 par 20 en f).
**Variante** : avec un goulot de 32 au lieu de 20 dans l'autoencodeur profond, l'encodeur aurait $401\,920 + 131\,328 + 32\,896 + 129 \times 32 =$ **570 272** paramètres.

### Ex 25.2 — Mélanger deux cercles : pixels contre paramètres ✏️
**Réponses** : a) **« (4, 3) »** ; b) **1,75** ; c) **0,85** ; d) **0,75** ; e) **0,1** ; f) **2** ; g) **19,6** ; h) **26,7**.
**Démarche** : a) $0{,}75\,(2, 2) + 0{,}25\,(10, 6) = (1{,}5 + 2{,}5\,;\ 1{,}5 + 1{,}5) = (4, 3)$. b) $0{,}75 \times 1 + 0{,}25 \times 4 = 1{,}75$. c) $0{,}75 \times 1 + 0{,}25 \times 0{,}4 = 0{,}85$. d) Le pixel $(2, 2)$ est le centre de A, à $\sqrt{80} \approx 8{,}9$ du centre de B (hors de B) : $0{,}75 \times 1 + 0{,}25 \times 0 = 0{,}75$. e) Le pixel $(10, 6)$ est dans B seulement : $0{,}25 \times 0{,}4 = 0{,}1$. f) Deux disques, A à 75 % et B à 25 %, chacun à sa place. g) À $t = 0{,}5$, le rayon vaut $2{,}5$ : $\pi \times 6{,}25 \approx 19{,}63$. h) $(\pi + 16\pi)/2 = 8{,}5\pi \approx 26{,}70$.
**Dans ta copie** : le pixel $(4, 3)$ vaut **0** dans le mélange de contenu : il est à $\sqrt{5} \approx 2{,}24$ du centre de A (rayon 1) et à $\sqrt{45} \approx 6{,}71$ du centre de B (rayon 4). Le centre du disque intermédiaire est vide dans l'image mélangée par pixels. L'aire varie comme le **carré** du rayon : la moyenne des rayons n'est pas la racine de la moyenne des carrés. Pour que l'aire soit la moyenne des aires, il faudrait mélanger l'aire (ou $r^2$), soit un rayon $\sqrt{8{,}5} \approx 2{,}92$. Le résultat d'un mélange dépend des paramètres choisis pour décrire les objets : c'est vrai aussi des latents d'un autoencodeur, dont rien ne dit qu'ils sont les « bons » paramètres.
**Erreurs fréquentes** : échanger $t$ et $1 - t$ (le centre $(8, 5)$) ; prendre le milieu ($t = 0{,}5$) ; oublier de multiplier par l'intensité de B en e) ; confondre aire et périmètre.
**Variante** : à $t = 0{,}75$, le disque paramétrique a pour centre **(8, 5)**, pour rayon **3,25** et pour intensité **0,55**.

### Ex 25.3 — Suivre les formes dans l'autoencodeur convolutif ✏️
**Réponses** : a) **« (64, 16, 14, 14) »** ; b) **« (64, 8, 7, 7) »** ; c) **« (64, 3, 7, 7) »** ; d) **1 539** ; e) **593** ; f) **2 132** ; g) **5,33** ; h) **1 568**.
**Démarche** : une convolution $3 \times 3$ avec `padding=1` garde la taille ; un pooling $2 \times 2$ la divise par 2 et garde les canaux. a) 16 canaux, $28/2 = 14$. b) 8 canaux (ceux de la deuxième convolution), $14/2 = 7$. c) 3 canaux, pas de pooling. d) $16 \times (1 \times 9 + 1) + 8 \times (16 \times 9 + 1) + 3 \times (8 \times 9 + 1) = 160 + 1\,160 + 219$. e) Les sur-échantillonnages n'ont pas de paramètres : $16 \times (3 \times 9 + 1) + 1 \times (16 \times 9 + 1) = 448 + 145$. f) $1\,539 + 593$. g) $784 / (3 \times 7 \times 7) = 784/147 = 5{,}33$. h) Deux poolings : $32 \times 7 \times 7 = 1\,568$.
**Dans ta copie** : $1\,568 = 2 \times 784$, deux fois plus de valeurs que l'image. Pour un débruiteur, ce n'est pas un problème : on ne cherche pas à comprimer, on cherche une sortie propre ; un code plus grand garde plus de détails. L'autoencodeur à 20 latents de l'Ex 25.1 a $32\,164$ paramètres, quinze fois plus que le convolutif : le convolutif partage ses poids à toutes les positions (ch. 21), exploite la structure locale des images, et dispose d'un code plus grand (147 valeurs).
**Erreurs fréquentes** : mettre les canaux à la fin (format du livre) ; reprendre les « 7 par 7 par 7 » du livre ; compter des paramètres au pooling ou au sur-échantillonnage ; oublier les biais ; oublier l'axe des exemples.
**Variante** : avec 4 filtres au lieu de 3 dans la dernière convolution de l'encodeur, le code aurait **196** valeurs (facteur **4**), et l'encodeur $160 + 1\,160 + 292 =$ **1 612** paramètres.

### Ex 25.4 — Retrouver 28×28 : padding et output_padding d'une convolution transposée ✏️
**Réponses** : a) **13** ; b) **14** ; c) **28** ; d) **14** ; e) **14** ; f) **4** ; g) **7** ; h) **4 624**.
**Démarche** : a) $(7 - 1) \times 2 - 2 + 3 = 13$. b) $13 + 1 = 14$. c) $(14 - 1) \times 2 - 2 + 3 + 1 = 28$. d) $(7 - 1) \times 2 + 2 = 14$ : un noyau 2 de stride 2 ne fait pas se chevaucher les morceaux. e) $(7 - 1) \times 2 - 2 + 4 = 14$. f) $\lfloor (7 + 2 - 3)/2 \rfloor + 1 = 3 + 1 = 4$. g) $(4 - 1) \times 2 - 2 + 3 = 7$. h) Les poids de `ConvTranspose2d(32, 16, 3)` ont la forme $(32, 16, 3, 3)$ : $4\,608$, plus 16 biais.
**Dans ta copie** : `output_padding=1` donnerait 8. Une entrée de hauteur 8 donne aussi $\lfloor (8 + 2 - 3)/2 \rfloor + 1 = 4$ : l'arrondi vers le bas efface la différence, et seul `output_padding` dit, au retour, laquelle des deux tailles retrouver. Vérification sans rien entraîner : `nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)(torch.zeros(1, 1, 14, 14)).shape` donne `torch.Size([1, 1, 28, 28])`.
**Erreurs fréquentes** : écrire $H_{in}\,s$ au lieu de $(H_{in} - 1)\,s$ ; ajouter le padding au lieu de le retirer ; oublier le $+ 1$ de la formule de `Conv2d` ; compter en h) 32 biais (le premier axe des poids) au lieu de 16 (un par canal de sortie).
**Variante** : `ConvTranspose2d(k=5, s=2, p=2, output_padding=1)` sur une hauteur 14 donne $(14 - 1) \times 2 - 4 + 5 + 1 =$ **28**.

### Ex 25.5 — Dériver à travers un tirage : l'astuce de reparamétrisation ∂
**Réponse** : $\partial z/\partial \mu = 1$, $\partial z/\partial s = \tfrac{1}{2}\sigma\varepsilon$, $\partial L/\partial \mu = g$, $\partial L/\partial s = \tfrac{1}{2} g\,\sigma\varepsilon$ ; application : $z = 1{,}2$, $L = 0{,}02$, $\partial L/\partial \mu = 0{,}2$, $\partial L/\partial s = 0{,}1$.
**Démarche** :
1. $z = \mu + \sigma\varepsilon$ donne $\partial z/\partial \mu = 1$ et $\partial z/\partial \sigma = \varepsilon$. Avec $\sigma = e^{s/2}$, $\partial\sigma/\partial s = \tfrac{1}{2}e^{s/2} = \sigma/2$, donc $\partial z/\partial s = \varepsilon \cdot \sigma/2$.
2. Règle de la chaîne : $\partial L/\partial \mu = g \cdot 1 = g$ et $\partial L/\partial s = g \cdot \tfrac{1}{2}\sigma\varepsilon$. Avec $\varepsilon = 0$, ce gradient est nul : le code vaut $\mu$, il ne dépend pas de $\sigma$ pour ce tirage, et la loss ne peut rien dire sur $\sigma$.
3. $\sigma = 2$, $z = 0{,}2 + 2 \times 0{,}5 = 1{,}2$ ; $L = \tfrac{1}{2}(0{,}2)^2 = 0{,}02$ ; $g = z - y = 0{,}2$ ; $\partial L/\partial \mu = 0{,}2$ ; $\partial L/\partial s = 0{,}2 \times \tfrac{1}{2} \times 2 \times 0{,}5 = 0{,}1$.
4. Une fonction qui « tire au hasard » ne donne pas un nombre qui varie continûment avec $\mu$ : deux appels avec le même $\mu$ donnent deux nombres différents, et la rétropropagation n'a aucune dérivée à remonter. En écrivant $z = \mu + \sigma\varepsilon$ avec $\varepsilon$ tiré à part, on obtient une fonction ordinaire, dérivable, de $\mu$ et $\sigma$ ; le hasard devient une entrée, comme une image, qui ne reçoit pas de gradient.
5. $\mathbb{E}[z^2] = \mu^2 + 2\mu\sigma\mathbb{E}[\varepsilon] + \sigma^2\mathbb{E}[\varepsilon^2] = \mu^2 + \sigma^2$, de dérivée $2\mu$. En dérivant d'abord : $\partial(z^2)/\partial\mu = 2z = 2(\mu + \sigma\varepsilon)$, d'espérance $2\mu$. Les deux résultats sont égaux : le gradient calculé sur un seul tirage est un estimateur **sans biais** du gradient de l'espérance ; il est bruité, mais juste en moyenne, et la descente de gradient stochastique s'en accommode, comme du bruit d'un mini-batch.
**Erreurs fréquentes** : oublier le facteur $\tfrac{1}{2}$ de $\sigma = e^{s/2}$ ; dériver par rapport à $\sigma^2$ au lieu de $s = \log\sigma^2$ ; croire que $\varepsilon$ reçoit un gradient ; confondre $\sigma$ et $\sigma^2$ en 3.
**Variante** : avec $\sigma^2 = 1$ et $\varepsilon = -1$ (même $\mu$ et $y$) : $z = $ **−0,8**, $L = $ **1,62**, $\partial L/\partial \mu = $ **−1,8**, $\partial L/\partial s = -1{,}8 \times \tfrac{1}{2} \times 1 \times (-1) = $ **0,9**.

### Ex 25.6 — Où tombent les tirages d'une gaussienne en grande dimension ✏️
**Réponses** : a) **20** ; b) **4,47** ; c) **0,393** ; d) **0,989** ; e) **1,18** ; f) **10** ; g) **3,16** ; h) **22,63**.
**Démarche** : a) $\mathbb{E}\lVert\mathbf{z}\rVert^2 = \sum_{j=1}^{20} \mathbb{E}[z_j^2] = 20$. b) $\sqrt{20} \approx 4{,}472$. c) $1 - e^{-1/2} = 0{,}3935$. d) $1 - e^{-9/2} = 0{,}9889$. e) $1 - e^{-r^2/2} = \tfrac{1}{2}$ donne $r^2 = 2\ln 2$, $r = 1{,}177$. f) Chaque coordonnée de $\mathbf{m}$ est $(z_{0,j} + z_{1,j})/2$, de variance $(1 + 1)/4 = 1/2$ (variances additionnées pour des tirages indépendants) : $\mathbb{E}\lVert\mathbf{m}\rVert^2 = 20/2 = 10$. g) $\sqrt{10} \approx 3{,}162$. h) $\sqrt{512} \approx 22{,}627$.
**Dans ta copie** : le milieu est environ $1/\sqrt{2} \approx 0{,}71$ fois plus court qu'un tirage typique, et, en dimension 20, les tirages de norme proche de 3,16 sont rares (les normes se concentrent autour de 4,47). Un décodeur de VAE a été entraîné sur des codes qui ressemblent à des tirages du prior : le milieu tombe dans une région qu'il a peu vue, et sa sortie peut être plus floue ou plus « moyenne ». L'interpolation sphérique avance le long de l'arc : entre deux codes de même norme, tous ses points gardent cette norme typique. En dimension 2 (c), d)), au contraire, le centre n'est pas vide : la loi de la distance au centre n'est concentrée qu'en grande dimension.
**Erreurs fréquentes** : prendre $\sqrt{d}$ pour l'espérance de la norme (c'est la racine de l'espérance du carré, une très bonne approximation en grande dimension) ; oublier le « 1 − » ; diviser la variance par 2 au lieu de 4 en f) ; résoudre pour $r^2$ au lieu de $r$.
**Variante** : en dimension $d = 100$, un tirage typique a une norme de **10** et le milieu de deux tirages une norme de **7,07** (l'exemple de T. White, 2016).

### Ex 25.7 — La divergence KL entre N(μ, σ²) et N(0, 1) ∂
**Réponses** : a) **2** ; b) **0,359** ; c) **1,808** ; d) **1,307** ; e) **1** ; f) **0,417** ; g) **2** ; h) **1,885**.
**Démarche** :
1. $\ln q(z) = -\tfrac{1}{2}\ln(2\pi\sigma^2) - \frac{(z - \mu)^2}{2\sigma^2}$ et $\ln p(z) = -\tfrac{1}{2}\ln(2\pi) - \frac{z^2}{2}$.
2. $\mathbb{E}_q[\ln q(z)] = -\tfrac{1}{2}\ln(2\pi\sigma^2) - \frac{\sigma^2}{2\sigma^2} = -\tfrac{1}{2}\ln(2\pi) - \tfrac{1}{2}\ln\sigma^2 - \tfrac{1}{2}$ ; $\mathbb{E}_q[\ln p(z)] = -\tfrac{1}{2}\ln(2\pi) - \frac{\mu^2 + \sigma^2}{2}$.
3. La différence : $\mathrm{KL} = -\tfrac{1}{2}\ln\sigma^2 - \tfrac{1}{2} + \frac{\mu^2 + \sigma^2}{2} = \tfrac{1}{2}(\mu^2 + \sigma^2 - 1 - \ln\sigma^2)$ ; les termes en $\ln(2\pi)$ s'annulent.
4. Avec $s = \ln\sigma^2$ : $\mathrm{KL} = \tfrac{1}{2}(\mu^2 + e^s - 1 - s)$, d'où $\partial\mathrm{KL}/\partial\mu = \mu$ et $\partial\mathrm{KL}/\partial s = \tfrac{1}{2}(e^s - 1)$.
Valeurs : a) $\tfrac{1}{2}(4 + 1 - 1 - 0) = 2$. b) $\tfrac{1}{2}(e - 2) = 0{,}3591$. c) $\tfrac{1}{2}(0{,}01 - 1 + 4{,}6052) = 1{,}8076$. d) $\tfrac{1}{2}(1 + 4 - 1 - 1{,}3863) = 1{,}3069$. e) $0{,}5 + 0 + 0{,}5 = 1$ (somme sur les latents). f) $\tfrac{1}{2}(0{,}25 + 2{,}5 - 1 - 0{,}9163) = 0{,}4169$. g) $\tfrac{1}{2}(5 - 1) = 2$. h) $1{,}3069/\ln 2 = 1{,}8854$ bit.
**Dans ta copie** : pour $\mu$ fixé, la dérivée $\tfrac{1}{2}(\sigma^2 - 1)/\sigma^2$ par rapport à $\sigma^2$ s'annule en $\sigma^2 = 1$, le minimum. Le terme $-\ln\sigma^2$ explose quand $\sigma \to 0$ : un code presque certain (un autoencodeur classique déguisé) coûte cher. Le terme $\mu^2$ punit les codes loin de l'origine. Sans ce terme, l'encodeur réduirait $\sigma$ à presque rien pour reconstruire au mieux et étalerait les moyennes à sa guise : on retrouverait un autoencodeur classique, avec ses trous, incapable de générer à partir du prior.
**Erreurs fréquentes** : oublier le facteur $\tfrac{1}{2}$ ; mettre $\sigma$ au lieu de $\sigma^2$ dans la formule (c) ; prendre $\ln\sigma$ au lieu de $\ln\sigma^2$ ; moyenner au lieu de sommer sur les latents ; multiplier par $\ln 2$ pour passer aux bits.
**Variante** : un code à deux latents de moyennes nulles et de variances $0{,}5$ et $2$ a une KL de $\tfrac{1}{2}(0{,}5 - 1 + \ln 2) + \tfrac{1}{2}(2 - 1 - \ln 2) = $ **0,25** nat : des coûts du même ordre (0,097 et 0,153 nat) ; le trop large coûte un peu plus.

### Ex 25.8 — Un autoencodeur linéaire retrouve l'ACP (cas 2D guidé) ∂
**Réponse** : le meilleur autoencodeur lié projette sur la première composante principale, avec une erreur égale à la variance restante ; ici $\lambda = 6$ et $1$, $\mathbf{w} = (2, 1)/\sqrt{5}$, $L = 1$ (contre $2$ pour $\mathbf{w} = (1, 0)$) ; l'autoencodeur non lié ne détermine que la droite (le sous-espace).
**Démarche** :
1. $\hat{\mathbf{x}} = (\mathbf{w}^\top\mathbf{x})\,\mathbf{w}$ : c'est la projection sur la droite de $\mathbf{w}$ (rappel maths du §25.6), et $\mathbf{x} - \hat{\mathbf{x}}$ est orthogonal à $\mathbf{w}$. Par Pythagore, $\lVert\mathbf{x} - \hat{\mathbf{x}}\rVert^2 = \lVert\mathbf{x}\rVert^2 - (\mathbf{w}^\top\mathbf{x})^2$. En moyenne : $\frac{1}{n}\sum\lVert\mathbf{x}_i\rVert^2 = \operatorname{tr}(\mathbf{C})$ et $\frac{1}{n}\sum(\mathbf{w}^\top\mathbf{x}_i)^2 = \mathbf{w}^\top\mathbf{C}\,\mathbf{w}$, d'où $L = \operatorname{tr}(\mathbf{C}) - \mathbf{w}^\top\mathbf{C}\,\mathbf{w}$.
2. $\operatorname{tr}(\mathbf{C})$ ne dépend pas de $\mathbf{w}$ : minimiser $L$, c'est maximiser $\mathbf{w}^\top\mathbf{C}\,\mathbf{w}$. Dans la base orthonormée des vecteurs propres $\mathbf{u}_1, \mathbf{u}_2$ ($\lambda_1 \ge \lambda_2$), $\mathbf{w} = c_1\mathbf{u}_1 + c_2\mathbf{u}_2$ avec $c_1^2 + c_2^2 = 1$, et $\mathbf{w}^\top\mathbf{C}\,\mathbf{w} = \lambda_1c_1^2 + \lambda_2c_2^2 \le \lambda_1$, avec égalité pour $\mathbf{w} = \pm\mathbf{u}_1$.
3. $\operatorname{tr} = 7$, $\det = 6$ : $\lambda^2 - 7\lambda + 6 = 0$, $\lambda = 6$ et $1$. Pour $\lambda = 6$ : $-x + 2y = 0$, $\mathbf{u}_1 = (2, 1)/\sqrt{5}$. $L = 7 - 6 = 1$. Pour $\mathbf{w} = (1, 0)$ : $\mathbf{w}^\top\mathbf{C}\,\mathbf{w} = 5$, $L = 2$.
4. $\hat{\mathbf{x}} = ((c\,\mathbf{a})^\top\mathbf{x})\,(\mathbf{b}/c) = (\mathbf{a}^\top\mathbf{x})\,\mathbf{b}$ : même reconstruction. L'autoencodeur ne peut donc pas fixer l'échelle (ni le signe) de son code. Avec $d$ latents, n'importe quelle matrice inversible $\mathbf{M}$ appliquée au code ($\mathbf{z}' = \mathbf{M}\mathbf{z}$), compensée par $\mathbf{M}^{-1}$ dans le décodeur, donne la même sortie : seul le sous-espace engendré par le décodeur est déterminé, ses axes ne le sont pas (ni orthogonaux, ni rangés par variance).
5. Il faut des biais : l'encodeur peut alors retirer la moyenne et le décodeur la rajouter (ou bien on centre les données avant). Pour des points sur le cercle unité, $\mathbf{C} = \tfrac{1}{2}\mathbf{I}$ : toutes les directions se valent, la meilleure ACP à une composante garde un diamètre et perd la moitié de la variance ($L = \tfrac{1}{2}$). Un autoencodeur non linéaire peut prendre pour code l'**angle** $\theta$ du point et décoder $(\cos\theta, \sin\theta)$ : une seule valeur suffit, avec une erreur nulle. C'est l'exemple le plus simple d'une variété courbe qu'aucune projection linéaire ne suit.
**Erreurs fréquentes** : oublier $\lVert\mathbf{w}\rVert = 1$ (sinon $\mathbf{w}^\top\mathbf{C}\,\mathbf{w}$ grandit sans limite) ; prendre le vecteur propre de la plus petite valeur propre ; croire que l'autoencodeur non lié retrouve aussi les axes ; oublier le centrage.
**Variante** : avec $\mathbf{C} = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$, les valeurs propres sont **4** et **2**, $\mathbf{w} = (1, 1)/\sqrt{2}$, et l'erreur minimale vaut **2**.

### Ex 25.9 — Compression réelle : le décodeur n'est pas gratuit 🧮
**Réponse** : environ 7 840 ko d'images brutes contre environ 866 ko de codes et de décodeur, un facteur de 9 environ ; l'autoencodeur devient rentable à partir de 94 images.
**Démarche** :
1. $10\,000 \times 784 = 7\,840\,000$ octets, soit 7 840 ko.
2. Codes : $10\,000 \times 20 \times 4 = 800\,000$ octets (800 ko). Décodeur : $16\,464 \times 4 = 65\,856$ octets (environ 66 ko). Total : environ 866 ko.
3. $7\,840/866 \approx 9{,}1$, contre 39,2 en comptant les nombres : un latent `float32` occupe 4 octets quand un pixel en occupe un, et il faut stocker le décodeur.
4. $784\,N > 80\,N + 65\,856$ dès que $N > 93{,}5$ : à partir de 94 images.
5. Un format sans perte rend les images exactes ; l'autoencodeur les rend floues. Il faudrait comparer la qualité (l'erreur, le PSNR), le temps de calcul pour compresser et décompresser, la dépendance au modèle (le même décodeur, la même version de la bibliothèque, pour toujours) et la robustesse à des images d'un autre genre. Pour archiver, on veut des fichiers exacts, lisibles par n'importe quel outil dans vingt ans : c'est le rôle des formats standard. Pour mémoire, le fichier `data/mnist.npz` du workbook range les 70 000 images en environ 11,5 Mo, sans perte, soit environ 165 octets par image.
**Erreurs fréquentes** : oublier le décodeur ; compter un octet par latent ; comparer à la taille des fichiers compressés en oubliant que ceux-ci sont sans perte.
**Variante** : en arrondissant chaque latent sur un octet (une quantification), les codes ne pèsent plus que 200 ko, et le facteur passe à environ **29** ($7\,840/266$), avec une petite erreur de plus.

### Ex 25.10 — Pourquoi diffuser dans un espace latent : facteur de compression et coût 🧮
**Réponse** : 786 432 valeurs contre 16 384, un facteur 48 ; une passe convolutive environ 64 fois moins chère, une couche d'attention environ 4 096 fois moins chère ; le décodeur ne sert qu'une fois.
**Démarche** :
1. $512 \times 512 \times 3 = 786\,432$ ; $64 \times 64 \times 4 = 16\,384$ ; $786\,432/16\,384 = 48$.
2. $512 \times 512 = 262\,144$ positions contre $64 \times 64 = 4\,096$ : environ 64 fois moins cher par passe (les canaux changent aussi le coût, mais l'ordre de grandeur est là).
3. $64^2 = 4\,096$ fois moins de comparaisons, puisque le coût est quadratique en nombre de positions. C'est ce qui rend l'attention abordable dans les générateurs.
4. 50 passes de diffusion contre une seule passe du décodeur : même si le décodeur est un gros réseau, il pèse peu dans le total.
5. Des latents de 16 canaux gardent plus de détails (les visages, le texte dans les images), au prix d'une compression plus faible (12 au lieu de 48) et d'un réseau de diffusion qui a plus de valeurs à traiter et à apprendre. L'image finale ne peut pas être meilleure que ce que le décodeur sait reconstruire : son autoencodeur doit donc être presque sans perte visible. Le VAE du chapitre, lui, est jugé sur sa capacité à générer des chiffres plausibles, flous ou non.
**Erreurs fréquentes** : oublier les 3 canaux de l'image ; compter le facteur 48 pour l'attention, qui est quadratique ; croire que le latent est une petite image visible (ses 4 canaux n'ont pas de sens de couleur).
**Variante** : pour une image de $1\,024 \times 1\,024$ et le même facteur 8, le latent a $128 \times 128 \times 4 = $ **65 536** valeurs, et une couche d'attention sur l'image coûterait $64^2 = $ **4 096** fois plus que sur le latent.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 25.11 — Le VAE expliqué à un ami en cinq lignes 🗣️
**Réponse modèle** : « Imagine un robot dessinateur et son assistant. L'assistant regarde un visage et le résume en quelques réglages, comme les curseurs d'un logiciel de portrait-robot ; le dessinateur redessine le visage à partir de ces curseurs. On les entraîne ensemble à redessiner des milliers de visages. Le VAE ajoute deux règles : l'assistant ne donne pas des réglages exacts mais une petite zone autour, dans laquelle on pioche au hasard, et toutes ces zones doivent rester serrées autour du milieu des curseurs. Comme le dessinateur a appris à dessiner n'importe quel réglage de cette zone, on peut ensuite renvoyer l'assistant, tirer des réglages au hasard au milieu, et le dessinateur invente des visages qui n'existent pas mais ont l'air vrais. »
**Ce qui compte** : l'encodeur résume, le décodeur reconstruit ; le hasard autour du code (pour que des codes voisins donnent des images voisines) ; la contrainte qui rassemble les codes au même endroit (pour savoir où tirer) ; la génération par le décodeur seul.
**Erreurs fréquentes** : parler de « compresser des fichiers » ; oublier pourquoi on ajoute du hasard ; oublier la contrainte qui rassemble les codes (sans elle, le hasard ne suffit pas).
**Variante** : en une phrase pour un chef de projet : « Un VAE apprend à décrire des images par quelques nombres rangés de façon régulière, si bien qu'en tirant ces nombres au hasard, on obtient de nouvelles images plausibles. »

### Ex 25.12 — Deepfakes par autoencodeurs : un encodeur, deux décodeurs ⚖️
**Réponse** :
1. L'encodeur commun doit représenter les deux visages dans le **même** espace de codes. Comme il sert aux deux décodeurs, son code ne peut garder que ce qui est commun aux deux personnes : la pose de la tête, l'expression, la direction du regard, l'éclairage ; l'identité est dans les décodeurs.
2. On voit le visage de B, avec la pose, l'expression et l'éclairage de A, image après image : le décodeur de B dessine « B dans la situation décrite par le code ». La qualité dépend des données (beaucoup d'images de B dans des poses variées) et des retouches de raccord (bords, couleur de peau).
3. Usages légitimes : le doublage de films (la bouche suit la langue traduite), les effets spéciaux (rajeunir un acteur avec son accord), l'anonymisation de témoins, la recherche sur la détection. Usages nuisibles : les montages à caractère sexuel de personnes réelles (des victimes, très majoritairement des femmes, atteintes dans leur dignité et leur vie privée) ; la désinformation politique (les électeurs trompés, la personne visée) ; les escroqueries au faux dirigeant ou au faux proche (l'entreprise ou la famille volée).
4. Ces règles posent des principes utiles : un devoir de transparence pour celui qui diffuse, des sanctions pénales pour les montages sans consentement, plus lourdes pour les montages sexuels et en ligne. Elles ne règlent pas l'**identification** d'auteurs anonymes ou à l'étranger, la **vitesse** de diffusion (le mal est fait avant le retrait), la **détection** (un marquage peut être retiré, un détecteur trompé), la responsabilité des **plateformes** (qui relève d'autres textes, comme le règlement européen sur les services numériques), ni le soutien aux victimes. Une loi seule ne suffit pas : il faut aussi des outils techniques et des procédures de retrait rapides. (Le règlement européen a été modifié en 2026 sur certaines dates et a ajouté une interdiction des systèmes qui génèrent des images sexuelles non consenties de personnes identifiables : à vérifier au moment où tu lis.)
5. Le consentement écrit et révocable des acteurs, pour cet usage précis ; un contrat qui dit ce qui sera produit ; un marquage visible ou un filigrane dans chaque vidéo produite, et des métadonnées de provenance ; un accès au modèle limité et journalisé ; la suppression des données d'entraînement à la fin du projet ; et un refus de toute demande qui vise une personne qui n'a pas consenti.
**Erreurs fréquentes** : croire que l'encodeur apprend l'identité ; oublier les usages légitimes ; ne parler que de la technique de détection ; citer des lois sans dire ce qu'elles ne règlent pas.
**Variante** : avec une seule image de B, la méthode à deux décodeurs ne marche pas : chaque décodeur doit voir beaucoup d'images de son visage. Les méthodes récentes de génération d'un visage à partir d'une seule photo rendent cette protection naturelle caduque, d'où l'importance du marquage et du droit.

### Ex 25.13 — Lire Kingma et Welling (2013), « Auto-Encoding Variational Bayes » 📄
**Réponse** :
1. Le problème : apprendre efficacement un modèle probabiliste dont les variables latentes sont continues et dont le posterior n'est pas calculable, sur de grands datasets. Contributions : (1) une reformulation de la borne inférieure variationnelle (la reparamétrisation) qui donne un estimateur dérivable, optimisable par descente de gradient stochastique standard ; (2) pour des datasets d'exemples indépendants, un **modèle de reconnaissance** (l'encodeur) appris avec cet estimateur, qui donne une approximation du posterior pour chaque exemple sans calcul coûteux : l'algorithme AEVB, dont le VAE est le cas où les deux modèles sont des réseaux de neurones.
2. $p_\theta(\mathbf{z})$ est le prior des codes ; $p_\theta(\mathbf{x} \mid \mathbf{z})$ le modèle génératif, le **décodeur** ; $q_\phi(\mathbf{z} \mid \mathbf{x})$ le modèle de reconnaissance, l'**encodeur**. Le vrai posterior $p_\theta(\mathbf{z} \mid \mathbf{x}) = p_\theta(\mathbf{x} \mid \mathbf{z})\,p_\theta(\mathbf{z})/p_\theta(\mathbf{x})$ demande $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})\,p_\theta(\mathbf{z})\,d\mathbf{z}$, une intégrale sur tous les codes qu'on ne sait pas calculer quand le décodeur est un réseau.
3. $\mathcal{L} = -\mathrm{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p_\theta(\mathbf{z})) + \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$ : les auteurs la **maximisent** ; la loss de la fiche est $-\mathcal{L}$, la KL en régularisation et la reconstruction en moins log-vraisemblance.
4. Ils écrivent $\tilde{\mathbf{z}} = g_\phi(\boldsymbol{\epsilon}, \mathbf{x})$ avec $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$, et, pour une gaussienne, $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ : exactement la fiche et l'Ex 25.5, où le hasard est une entrée et le code une fonction dérivable de $\boldsymbol{\mu}$ et $\boldsymbol{\sigma}$.
5. L'annexe B donne $-\mathrm{KL} = \tfrac{1}{2}\sum_j(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$ : au signe près, la formule de l'Ex 25.7, sommée sur les latents.
6. Un seul tirage par exemple ($L = 1$) suffit si le mini-batch est assez grand (de l'ordre de 100 exemples). Ils comparent AEVB à l'algorithme *wake-sleep* sur MNIST et sur le dataset de visages Frey Face ; la figure 2 montre la borne inférieure en fonction du nombre d'exemples vus, pour plusieurs dimensions de l'espace latent : AEVB monte plus vite et plus haut. L'annexe A montre, pour des modèles à deux latents, la grille de chiffres et de visages décodés en parcourant l'espace latent (l'équivalent de la figure 25.68 du livre), et des tirages au hasard pour plusieurs dimensions (l'équivalent de la figure 25.70) : le livre refait ces figures, sans les maths.
**Erreurs fréquentes** : confondre le prior et le posterior ; prendre $q_\phi$ pour le décodeur ; oublier le signe de la KL ; croire que l'article est d'abord sur la génération d'images (il est d'abord sur l'inférence et l'apprentissage).
**Variante** : lis ensuite l'introduction de Kingma et Welling (2019, [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)), chapitre 2 : la même dérivation en plus détaillé, avec la démonstration de l'ELBO par l'inégalité de Jensen.

<a id="entretien"></a>

## 💼 Entretien

### 25.E1 — Autoencodeur ou ACP : quand préférer l'un ou l'autre ?
**Réponse modèle en 60 secondes** : « L'ACP est linéaire : elle cherche les directions de plus grande variance, et un autoencodeur linéaire entraîné à l'erreur quadratique retrouve d'ailleurs exactement son sous-espace. Je commence donc par l'ACP : elle est exacte, rapide, sans hyperparamètres, stable d'un lancement à l'autre, et ses composantes s'interprètent ; la variance expliquée me dit combien de dimensions garder. Je passe à un autoencodeur quand les données ont une structure non linéaire, comme des images ou des sons, où des couches non linéaires ou convolutives résument bien mieux avec peu de dimensions ; ou quand je veux aussi débruiter, détecter des anomalies ou générer. Le prix : un entraînement, des hyperparamètres, des résultats qui dépendent de la graine, et un code moins interprétable. Pour visualiser en 2D, j'utiliserais plutôt t-SNE ou UMAP. »
**Ce qui compte** : l'équivalence linéaire ; les avantages de l'ACP ; le cas non linéaire ; les usages en plus (débruiter, anomalies, générer) ; le coût de l'autoencodeur.
**Relances possibles** : « L'autoencodeur linéaire retrouve-t-il les composantes de l'ACP ? » (le sous-espace seulement, pas les axes ni leur ordre ; il faut une contrainte de plus, ou une ACP sur le code) · « Comment choisir la taille du goulot ? » (une courbe d'erreur de reconstruction sur la validation selon la taille, comme la variance expliquée de l'ACP, et l'usage aval : l'accuracy d'un classifieur sur le code) · « Sur des données tabulaires ? » (l'ACP, ou rien : les modèles d'arbres n'ont pas besoin de réduire la dimension).

### 25.E2 — Pourquoi un VAE sait-il générer et pas un autoencodeur classique ?
**Réponse modèle en 60 secondes** : « Un autoencodeur classique range ses codes comme il veut : il y a des zones vides, des zones où des classes se mélangent, et on ne sait pas où tirer. Un code pris au hasard tombe presque toujours là où le décodeur n'a rien appris, et sort une tache. Le VAE change deux choses. Son encodeur donne une petite gaussienne par exemple, une moyenne et une variance, et on tire le code dedans : le décodeur apprend à rendre une image plausible pour tout un voisinage de codes, ce qui remplit les trous. Et un terme KL dans la loss rapproche ces gaussiennes du prior, une gaussienne centrée réduite : tous les codes vivent dans la même région connue. Après l'entraînement, je tire des codes du prior, je les donne au décodeur seul, et j'obtiens des images nouvelles. Le compromis : les images sont plus floues qu'avec un GAN ou un modèle de diffusion. »
**Ce qui compte** : les trous et les mélanges de l'espace latent classique ; l'encodeur probabiliste ; le terme KL vers le prior ; tirer du prior et décoder ; la limite (flou).
**Relances possibles** : « Pourquoi les images d'un VAE sont-elles floues ? » (la reconstruction est une moyenne sur les codes possibles, et la loss pixel par pixel favorise la moyenne des images plausibles ; le compromis avec la KL en rajoute) · « Et si on augmente le poids de la KL ? » (β-VAE : un espace plus régulier, des images plus floues, jusqu'à l'effondrement du posterior) · « Comment évaluer un générateur ? » (regarder des tirages, mesurer la vraisemblance ou l'ELBO, des scores comme la FID, ch. 27).

### 25.E3 — Expliquer l'astuce de reparamétrisation
**Réponse modèle en 60 secondes** : « Dans un VAE, l'encodeur donne pour chaque latent une moyenne et une variance, et on doit tirer un code dans cette gaussienne. Or un tirage aléatoire ne se dérive pas : le nombre obtenu n'est pas une fonction de la moyenne, et la rétropropagation n'a rien à remonter vers l'encodeur. L'astuce consiste à tirer d'abord un bruit ε dans une gaussienne standard, qui ne dépend d'aucun poids, puis à calculer z égal à μ plus σ fois ε. Le code a la même loi qu'avant, mais c'est maintenant une fonction ordinaire, dérivable, de μ et σ ; ε est une simple entrée. La dérivée de z par rapport à μ vaut 1, par rapport à σ vaut ε. En pratique, l'encodeur sort log σ², pour pouvoir prendre n'importe quel réel, et on calcule σ comme l'exponentielle de la moitié. Le gradient obtenu avec un seul tirage est bruité mais sans biais. »
**Ce qui compte** : pourquoi un tirage bloque le gradient ; $z = \mu + \sigma\varepsilon$ ; ε comme entrée ; les dérivées ; log σ² ; l'estimateur sans biais.
**Relances possibles** : « Et pour une variable discrète ? » (la reparamétrisation ne s'applique pas directement ; on utilise Gumbel-softmax, ou un estimateur du gradient par la vraisemblance (REINFORCE), plus bruité) · « Combien de tirages par exemple ? » (un seul suffit en pratique avec des mini-batches de taille raisonnable, d'après Kingma et Welling) · « Pourquoi log σ² plutôt que σ ? » (une sortie linéaire peut être négative ; l'exponentielle garantit un σ positif, et le calcul est plus stable).

### 25.E4 — Détecter des anomalies avec un autoencodeur : principe et pièges
**Réponse modèle en 60 secondes** : « J'entraînerais un autoencodeur convolutif sur des photos de pièces **normales** uniquement, nombreuses. Il apprend à reconstruire ce qui est normal ; une pièce défectueuse est mal reconstruite, et l'erreur de reconstruction devient un score d'anomalie. Je fixe le seuil sur une validation de pièces normales, par exemple au 99ᵉ centile de leurs erreurs, selon le taux de fausses alertes acceptable, et je mesure la ROC-AUC sur les quelques défauts réels que j'ai. Les pièges : un autoencodeur trop capable reconstruit aussi les défauts, donc je limite le goulot et je vérifie ; une erreur moyenne sur toute l'image peut noyer un petit défaut, donc je regarde la carte d'erreur pixel par pixel ou le maximum local ; les variations normales, comme l'éclairage ou la position, doivent être dans l'entraînement, sinon tout est anomalie ; et je surveille la dérive dans le temps. Enfin, je compare à des méthodes plus simples, et aux modèles pré-entraînés qui font souvent mieux. »
**Ce qui compte** : entraîner sur le normal ; le score par erreur ; le seuil par validation et ROC-AUC ; la capacité ; l'erreur locale ; les variations normales ; la comparaison.
**Relances possibles** : « Pourquoi pas un classifieur ? » (presque pas d'exemples de défauts, et des défauts nouveaux qu'aucun exemple ne couvre ; on peut ajouter un classifieur quand on en a assez) · « Et un VAE ? » (même principe ; on peut utiliser l'ELBO comme score, mais les vraisemblances de modèles génératifs se trompent parfois sur des données d'un autre domaine (Nalisnick et coll., 2019)) · « Comment choisir le seuil sans défauts ? » (sur le taux de fausses alertes acceptable par les opérateurs, puis l'ajuster avec les défauts rencontrés).

### 25.E5 — Pourquoi Stable Diffusion travaille-t-il dans un espace latent ?
**Réponse modèle en 60 secondes** : « Un modèle de diffusion génère une image en débruitant un bruit pur, en des dizaines de passes d'un gros réseau. Faire ces passes sur des pixels de 512 par 512 coûte très cher, et une grande partie de l'effort irait à des détails imperceptibles. Stable Diffusion entraîne d'abord un autoencodeur, régularisé par une petite KL comme un VAE, qui comprime une image en un latent de 64 par 64 avec 4 canaux, environ 48 fois moins de valeurs, tout en gardant l'essentiel de ce qui se voit. La diffusion travaille dans cet espace, ce qui divise le coût de chaque passe, et surtout celui de l'attention, quadratique en nombre de positions. À la fin, le décodeur de l'autoencodeur transforme le latent en image, une seule fois. La KL garde l'espace latent régulier ; le décodeur doit être presque sans perte visible, puisqu'il plafonne la qualité finale. Les modèles récents prennent des latents de 16 canaux pour garder plus de détails. »
**Ce qui compte** : le coût des passes sur des pixels ; l'autoencodeur régularisé par KL ; le facteur de compression ; l'attention quadratique ; le décodeur à la fin ; la qualité plafonnée par le décodeur.
**Relances possibles** : « Pourquoi une KL si petite ? » (assez pour garder un espace régulier, pas trop pour que la reconstruction reste nette ; ce n'est pas un VAE fait pour générer seul) · « Que voit-on si l'on affiche le latent ? » (une image floue reconnaissable sur certains canaux, mais ses 4 canaux ne sont pas des couleurs) · « Qu'est-ce que le *scaling factor* ? » (un coefficient qui ramène les latents à une variance proche de 1 avant la diffusion, 0,18215 pour Stable Diffusion 1).

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb` ; voici, pour chaque exercice, la réponse, la démarche, les erreurs fréquentes et une variante. Les chiffres sont ceux du mode rapide, sur CPU (PyTorch 2.11, 2 cœurs) : un autre processeur ou un GPU change les dernières décimales des résultats d'entraînement, et les durées. Les entraînements longs de 📈 25.34 et les calibrages sur plusieurs graines (🔮 25.14, 25.17, 25.31, 🏆 25.39) viennent de `tools/chapters/history_ch25.py`, et leurs résultats sont dans `tools/chapters/history_ch25.json`.

### Ex 25.14 — Prédire : un autoencodeur entraîné sur une seule image 🔮
**Réponse** : a) **Faux** ; b) **Vrai** ; c) **Vrai**.

| | botte → botte | noir → botte | T-shirt → botte | T-shirt → T-shirt | code nul → botte |
|---|---|---|---|---|---|
| avec biais | 0,00000 | 0,170 | 0,015 | 0,086 | 0,179 |
| sans biais | 0,00000 | 0,192 | 0,016 | 0,085 | 0,192 |

**Démarche** : les deux réseaux reconstruisent la botte sans erreur visible : 20 latents pour une seule image, c'est beaucoup trop. Mais ils l'ont rangée **dans les poids du décodeur**, pas dans ses biais : la sortie vaut $\sigma(\mathbf{W}_2\mathbf{h} + \mathbf{b}_2)$, et l'entraînement a appris à envoyer **le** code de la botte, $\mathbf{h}_{\text{botte}}$, sur la botte. Une image noire donne un code presque nul ($\text{ReLU}(\mathbf{b}_1)$, de petits biais) : il ne reste que $\sigma(\mathbf{b}_2)$, une image grise sans rapport (colonne « code nul »). Un T-shirt, lui, a des pixels positifs comme la botte : son code est un multiple de $\mathbf{h}_{\text{botte}}$ dans les mêmes directions, et le décodeur rend… la botte (MSE de 0,015 avec elle, de 0,086 avec le T-shirt). Sans aucun biais, rien ne change pour la botte : la mémoire est dans $\mathbf{W}_2$. Le livre, lui, obtient le tigre à partir d'une image noire : après un entraînement beaucoup plus long sur 10 000 pixels, ses biais de sortie ont fini par porter l'image. Les deux chemins mènent à la même tricherie (le réseau n'a rien comprimé), et seule une évaluation sur d'autres images la révèle. Les trois réponses sont les mêmes sur cinq graines (`calib`, entrée `p14`).

**Erreurs fréquentes** : croire que « mémorisé » veut dire « dans les biais » (ici, les biais ne portent presque rien) ; lire la colonne « botte → botte » pour répondre à a) ; oublier que $\text{ReLU}(\mathbf{W}_1\mathbf{x})$ d'une image positive reste dans le même cône que celui de la botte.

**Variante** : entraîne le même réseau sur deux images, la botte et le T-shirt, et donne-lui l'image noire. Réponse : il reconstruit les deux sans erreur visible, et l'image noire donne encore un gris presque uniforme (des pixels autour de 0,48), ni la botte ni le T-shirt : deux images, c'est encore de la mémorisation, rangée dans les poids.

### Ex 25.15 — Premier autoencodeur dense 784-20-784 en PyTorch 📦
**Réponse** :
```python
def make_dense_ae_15(n_latent=20):
    return AutoEncoder(nn.Sequential(nn.Linear(784, n_latent), nn.ReLU()),
                       nn.Sequential(nn.Linear(n_latent, 784), nn.Sigmoid()))


def fit_ae_15(model, x, epochs, lr=1e-3, batch_size=128, seed=0):
    device = next(model.parameters()).device
    x = to_tensor(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=g)
        total = 0.0
        for i in range(0, len(x), batch_size):
            xb = x[order[i:i + batch_size]].to(device)
            loss = F.binary_cross_entropy(model(xb), xb)        # the target is the input itself
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        losses.append(total / len(x))
    return losses
```
**Démarche** : la boucle est celle du ch. 23 ; seule la cible change. Après 5 epochs sur 10 000 images, la MSE de test vaut 0,043 avec 20 latents, 0,061 avec 10 et 0,069 avec 3 : les chiffres à 20 latents sont flous mais lisibles, ceux à 10 se déforment, ceux à 3 sont des taches qui rappellent vaguement des chiffres, comme dans le livre. La sigmoïde ramène chaque sortie dans $[0, 1]$, comme un pixel ; la cross-entropy binaire accepte une cible qui n'est pas 0 ou 1 : elle est minimale quand la sortie égale la cible, et sa pente ne s'écrase pas comme celle d'une MSE après une sigmoïde (ch. 17).

**Erreurs fréquentes** : comparer la sortie à `y` (les labels) au lieu de `xb` ; créer le générateur dans la boucle des epochs (le même ordre à chaque epoch) ; moyenner les losses des batches sans les pondérer (le dernier batch est plus petit) ; appeler `torch.manual_seed` dans `fit_ae_15` (le contrat le laisse à l'appelant).

**Variante** : remplace la cross-entropy par l'erreur quadratique (`loss="mse"` dans `train_ae`). Réponse : ici, presque rien ne change (MSE de test de 0,043 après 5 epochs dans les deux cas) : les sorties restent loin de la saturation de la sigmoïde, où la MSE aurait une pente trop faible. Les deux losses ont le même minimum (la sortie égale à la cible) ; elles ne diffèrent nettement que lorsque des sorties saturent.

### Ex 25.16 — Autoencodeur profond 512-256-128-20 📦
**Réponse** :
```python
def make_deep_ae_16(n_latent=20, hidden=(512, 256, 128)):
    sizes = [784, *hidden, n_latent]
    encoder = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        encoder += [nn.Linear(n_in, n_out), nn.ReLU()]
    encoder.pop()                                       # no activation on the code
    back = sizes[::-1]                                  # n_latent, ..., 784
    decoder = []
    for n_in, n_out in zip(back[:-1], back[1:]):
        decoder += [nn.Linear(n_in, n_out), nn.ReLU()]
    decoder[-1] = nn.Sigmoid()                          # pixels in [0, 1]
    return AutoEncoder(nn.Sequential(*encoder), nn.Sequential(*decoder))
```
**Démarche** : avec le même goulot de 20 et 8 epochs, le dense (32 164 paramètres) atteint une MSE de test de 0,040, le profond (1 138 212 paramètres) 0,027. Le dense est limité par sa **capacité** : une seule couche pour calculer le code, une seule pour le décoder, donc un code presque linéaire. Le profond garde les mêmes 20 nombres mais les calcule et les décode mieux : c'est la leçon du §25.4, « assez de latents, et assez de capacité ». Il coûte 35 fois plus de paramètres et environ 4 fois plus de temps ici.

**Erreurs fréquentes** : mettre une ReLU sur le code (les latents négatifs sont perdus, et un latent qui reste négatif cesse d'apprendre) ; oublier la sigmoïde finale ; construire le décodeur avec les tailles dans le même ordre que l'encodeur.

**Variante** : un profond `hidden=(256, 64)`. Réponse : 438 180 paramètres (700 032 de moins ; la formule de l'Ex 25.1, couche par couche), et une erreur de test de 0,028 après 8 epochs, presque celle du grand profond (0,027) et bien meilleure que celle du dense (0,040) : la profondeur aide même avec des couches plus étroites.

### Ex 25.17 — Prédire : ce qu'un autoencodeur de chiffres fait d'un vêtement 🔮
**Réponse** : a) **B** (rapport 3,4) ; b) **B** (Sandal 0,055 et Sneaker 0,056) ; c) **Vrai** (0,092 contre 0,207).

**Démarche** : `AE_DEEP` se trompe en moyenne de 0,027 sur les chiffres et de 0,092 sur les vêtements, 3,4 fois plus. Les mieux reconstruits sont les sandales et les baskets : de petites formes claires, peu étendues, que le décodeur rend comme des traits épais, un peu comme un 2 couché ou un 7. Les pires sont les sacs et les bottines, de grandes surfaces claires qu'aucun chiffre ne ressemble à remplir. Pourtant, l'autoencodeur fait bien mieux qu'une image noire (0,207) : il place une forme claire au bon endroit, au centre de l'image, avec à peu près la bonne étendue. Il a appris des régularités des images 28 × 28 centrées, pas seulement des chiffres. L'écart d'erreur entre ce qu'il connaît et ce qu'il ne connaît pas est la base de la détection d'anomalies (25.37). Sur cinq graines, le rapport va de 3,1 à 3,4 et les deux meilleures classes sont toujours les sandales et les baskets (entrée `p17` de `calib`).

**Erreurs fréquentes** : comparer la MSE d'un vêtement à celle de **sa** classe de chiffres (il n'y en a pas) ; lire le tableau trié comme s'il allait du pire au meilleur ; croire qu'un autoencodeur « hors domaine » ne rend rien du tout.

**Variante** : donne à `AE_DEEP` des chiffres retournés (`XM_TEST.reshape(-1, 28, 28)[:, ::-1]`). Réponse : l'erreur moyenne passe de 0,027 à 0,050, moins que pour les vêtements (0,092) : les traits restent des traits, et certains chiffres (0, 1, 8) restent presque des chiffres une fois retournés.

### Ex 25.18 — Reproduire la carte latente 2D et sa grille décodée (fig. 25.29-25.30) 🎨
**Réponse** :
```python
def grid_codes_18(lims, n):
    xmin, xmax, ymin, ymax = lims
    xs, ys = np.linspace(xmin, xmax, n), np.linspace(ymax, ymin, n)       # y from the top to the bottom
    gx, gy = np.meshgrid(xs, ys)                                          # gx[i, j] = xs[j], gy[i, j] = ys[i]
    return np.column_stack([gx.ravel(), gy.ravel()])


def plot_latent_map_18(model, x, labels, n=15):
    codes = encode(model, x)
    lims = (codes[:, 0].min(), codes[:, 0].max(), codes[:, 1].min(), codes[:, 1].max())
    images = decode(model, grid_codes_18(lims, n).astype(np.float32)).reshape(n, n, 28, 28)
    mosaic = images.transpose(0, 2, 1, 3).reshape(28 * n, 28 * n)        # rows of images, then rows of pixels
    fig, (ax_map, ax_grid) = plt.subplots(1, 2, figsize=(12, 5.6))
    colors = wb.plot.class_colors(10)
    for digit in range(10):
        keep = np.asarray(labels) == digit
        ax_map.scatter(codes[keep, 0], codes[keep, 1], s=3, color=colors[digit], label=str(digit))
    ax_map.set_xlim(lims[0], lims[1])
    ax_map.set_ylim(lims[2], lims[3])
    ax_map.legend(markerscale=4, fontsize=8, ncol=2)
    ax_map.set_title("codes de 2 000 chiffres de test")
    ax_grid.imshow(mosaic, cmap="gray", extent=lims, aspect="auto")
    ax_grid.set_title(f"{n} × {n} codes de la grille, décodés")
    plt.tight_layout()
    plt.show()
    return mosaic
```
**Démarche** : la carte montre des régions par chiffre (les 0 et les 1 bien à part), des chevauchements (4, 7 et 9, et souvent 3, 5 et 8), et de grandes zones sans aucun point, surtout aux bords. La grille décodée change continûment d'une case à l'autre ; dans les zones vides, elle rend des formes qui ne sont aucun chiffre, ou des chiffres étirés. Tirer des codes au hasard tombe souvent dans ces zones : c'est le problème que le VAE résout (25.31, 25.33). La mosaïque demande le bon ordre des axes : `(ligne d'images, ligne de pixels, colonne d'images, colonne de pixels)`, d'où le `transpose(0, 2, 1, 3)`.

**Erreurs fréquentes** : `np.linspace(ymin, ymax, n)` (la carte est retournée de haut en bas par rapport aux points) ; un `reshape(28 * n, 28 * n)` sans `transpose` (des bandes horizontales au lieu d'images) ; oublier `.astype(np.float32)` (PyTorch refuse de mélanger des `float64` et des poids `float32`).

**Variante** : remplace `imshow(..., extent=lims)` par un `imshow` sans `extent`. Réponse : la mosaïque est juste mais ses axes comptent les pixels (0 à 420) au lieu des coordonnées latentes : on ne peut plus la poser à côté de la carte.

### Ex 25.19 — Secouer les variables latentes : bruit global, une seule composante, latents aléatoires 🔬
**Réponse** :
```python
def add_noise_19(codes, scale, rng, column=None):
    noisy = np.array(codes, dtype=np.float32)               # a copy
    if column is None:
        noisy += rng.uniform(-scale, scale, size=noisy.shape)
    else:
        noisy[:, column] += rng.uniform(-scale, scale, size=len(noisy))
    return noisy


def noise_mse_19(model, x, scales, seed=0, column=None):
    codes = encode(model, x)
    return [float(mse_per_image(decode(model, add_noise_19(codes, s, np.random.default_rng(seed), column)), x).mean())
            for s in scales]
```
**Démarche** : les latents de `AE_DEEP` ont un écart-type de 3,5 et vont de −12,6 à 11,3. Un bruit de ±0,1 écart-type ne change presque rien (MSE 0,0275 contre 0,0271) ; ±0,3 σ commence à abîmer les traits (0,031) ; ±1 σ double l'erreur (0,054), les chiffres changent parfois d'identité ; ±3 σ donne des formes illisibles (0,103), comme le bruit ±100 du livre. Bruités seuls, les latents ne comptent pas tout à fait autant : dans le diagramme en barres, le latent le plus sensible coûte environ 1,3 fois plus que le moins sensible ; aucun ne porte à lui seul l'identité du chiffre, l'information est répartie entre eux. Des codes tirés au hasard entre le min et le max de chaque latent se décodent en taches : chaque latent pris seul reste dans sa plage, mais leur **combinaison** tombe dans une zone vide de l'espace latent (25.18).

**Erreurs fréquentes** : bruiter `codes` sur place (la ligne suivante de l'expérience part de codes déjà bruités) ; réutiliser le même générateur d'une échelle à l'autre (les courbes ne se comparent plus, le bruit change en plus de l'échelle) ; un bruit en valeur absolue (±1) sans regarder l'échelle des latents.

**Variante** : bruite chaque latent en proportion de **son** écart-type (`scale` multiplié par `codes.std(axis=0)`, un bruit tiré par colonne). Réponse : le diagramme change peu (un rapport d'environ 1,3 entre le latent le plus sensible et le moins sensible, comme avec un bruit absolu) : ici, les latents ont des écarts-types voisins, et leurs sensibilités ne viennent pas de leur échelle.

### Ex 25.20 — Interpoler entre deux codes : linéaire et sphérique 🔨
**Réponse** (`solutions/mylearn_ref/vae.py` ; `_pair` valide les deux vecteurs et en fait des copies `float64`, comme en 25.28) :
```python
def interpolate_latents(z0, z1, n_steps, method="linear"):
    if method not in ("linear", "slerp"):
        raise ValueError(f"method must be 'linear' or 'slerp', got {method!r}")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 2:
        raise ValueError(f"n_steps must be an integer >= 2, got {n_steps!r}")
    a, b = _pair(z0, z1, ("z0", "z1"), 1)
    t = np.linspace(0.0, 1.0, int(n_steps))[:, None]
    linear = (1.0 - t) * a + t * b
    if method == "linear":
        return linear
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return linear
    cos = np.clip(np.dot(a / norm_a, b / norm_b), -1.0, 1.0)
    omega = np.arccos(cos)
    sin_omega = np.sin(omega)
    if abs(sin_omega) < 1e-6:                   # parallel or opposite vectors: the straight line
        return linear
    out = (np.sin((1.0 - t) * omega) / sin_omega) * a + (np.sin(t * omega) / sin_omega) * b
    out[0], out[-1] = a, b                      # exactly the ends
    return out
```
**Démarche** : la ligne droite moyenne les coordonnées ; le slerp suit l'arc entre les deux directions. Entre le code d'un 1 (norme 23,2) et celui d'un 0 (norme 14,5), la norme descend jusqu'à 11,2 au milieu de la ligne droite, alors que le slerp passe en douceur de 23,2 à 14,5 sans creux. Ici, les images des deux chemins se ressemblent : rien n'a imposé aux codes de `AE_DEEP` une norme typique. La différence compte dans un espace où les codes vivent sur une « coquille », comme ceux d'une gaussienne en grande dimension (✏️ 25.6) : celui d'un VAE ou d'un GAN, où le milieu d'une ligne droite tombe là où il n'y a presque jamais de code (25.Q9).

**Erreurs fréquentes** : diviser par les normes avant de les tester (`RuntimeWarning` et NaN pour un vecteur nul) ; un seuil de repli trop large (à 0,05 rad, $\sin\Omega \approx 0{,}05$ : c'est encore un vrai slerp, et les tests le vérifient) ; oublier `np.clip` avant `arccos` (un cosinus de 1,0000000002 donne NaN) ; ne pas forcer les extrémités (une erreur d'arrondi de $10^{-16}$ suffit à faire échouer l'égalité exacte).

**Variante** : entre $(3, 0)$ et $(0, 4)$, que donne le slerp à $t = 0{,}5$ ? Réponse : $\Omega = \pi/2$, donc $\frac{\sin(\pi/4)}{1}(3, 0) + \frac{\sin(\pi/4)}{1}(0, 4) \approx (2{,}12 ; 2{,}83)$, de norme 3,54 : entre les deux normes, sur une courbe qui n'est un arc de cercle que si les normes sont égales.

### Ex 25.21 — Mélanger dans l'espace des pixels ou dans l'espace latent 🔬
**Réponse** :
```python
def blend_pixels_21(x0, x1, n_steps):
    t = np.linspace(0, 1, n_steps)[:, None]
    return (1 - t) * np.asarray(x0, dtype=np.float32) + t * np.asarray(x1, dtype=np.float32)


def blend_codes_21(model, x0, x1, n_steps, method="linear"):
    z0, z1 = encode(model, np.stack([x0, x1]))
    path = mylearn.vae.interpolate_latents(z0, z1, n_steps, method=method)
    return decode(model, np.asarray(path, dtype=np.float32))


def off_manifold_21(model, images):
    images = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    return mse_per_image(reconstruct(model, images), images)
```
**Démarche** : le fondu des pixels superpose un 1 et un 0 : à mi-chemin, un 0 pâle traversé d'un trait, ce que l'autoencodeur ne sait pas redessiner (distance 0,016 à 0,024, cinq à vingt fois celle d'un vrai chiffre). L'interpolation des codes reste à une distance d'environ 0,001 à chaque étape : chaque image est, pour l'autoencodeur, un chiffre « possible » (forcément : elle sort de son décodeur). Mais « possible pour le décodeur » ne veut pas dire « un vrai chiffre » : entre un 1 et un 0, les étapes du milieu sont des formes ambiguës. Entre deux 5, tout se passe mieux : les deux codes sont voisins, et le chemin reste dans la région des 5. C'est le mélange paramétrique du ✏️ 25.2 : des paramètres bien rangés (ici, appris) donnent des intermédiaires plausibles, un mélange de contenu donne une superposition.

**Erreurs fréquentes** : interpoler les images reconstruites au lieu des codes ; décoder un chemin en `float64` ; conclure qu'un mélange latent est réaliste parce que l'autoencodeur le reconstruit bien (la mesure est biaisée en sa faveur : il faudrait un juge extérieur, comme le classifieur de 25.31).

**Variante** : mesure les étapes avec le classifieur `classify_31` de 25.31 plutôt qu'avec l'autoencodeur lui-même. Réponse : surprise : le classifieur est presque sûr de lui sur le fondu de pixels dès le milieu (0,87 puis 0,99 : il voit un 0 pâle, et le reconnaît), alors qu'il hésite au milieu du chemin latent (0,31 à 0,58) : les étapes latentes sont de vraies formes intermédiaires, ni un 1 ni un 0. « Être reconnu par un classifieur » et « ressembler à un chiffre bien écrit » sont deux questions différentes, et aucune des deux mesures ne suffit seule.

### Ex 25.22 — Taille du goulot et profondeur : la courbe d'erreur de reconstruction 🔬
**Réponse** :
```python
def bottleneck_sweep_22(sizes, makers, epochs, seed=2522, x=None):
    x = X_MNIST if x is None else x
    results = {}
    for name, maker in makers.items():
        for size in sizes:
            torch.manual_seed(seed)
            model = maker(size)
            train_ae(model, x, epochs, seed=seed)
            results[(name, size)] = float(mse_per_image(reconstruct(model, XM_TEST), XM_TEST).mean())
    return results
```
**Démarche** : en 4 epochs, le profond l'emporte nettement aux petits goulots (0,049 contre 0,117 à 2 latents, 0,039 contre 0,066 à 5) : avec peu de nombres, il faut beaucoup de calcul pour les choisir et les décoder. Mais sa courbe est presque plate de 5 à 50 latents (0,039 à 0,041) : en 4 epochs, il n'a pas eu le temps de se servir des latents en plus. Le dense, lui, progresse encore à 50 latents (0,036, mieux que le profond) : plus simple, il apprend vite, et un large goulot compense sa faible capacité. Sa courbe n'est pas lisse (5 et 10 latents presque à égalité) : avec une ReLU sur le code, des latents peuvent « mourir » tôt et rester inutilisés. Une comparaison à nombre d'epochs fixe désavantage le modèle le plus lent à apprendre : pour conclure, il faudrait entraîner chacun jusqu'à ce que sa loss de validation cesse de baisser (early stopping, ch. 23).

**Erreurs fréquentes** : appeler `torch.manual_seed` une seule fois avant les boucles (chaque modèle part alors d'un état aléatoire différent, et la comparaison mélange l'effet de la taille et celui du hasard) ; mesurer la loss d'entraînement au lieu de la MSE de test ; tracer en échelle linéaire (les petits goulots, où tout se joue, sont écrasés).

**Variante** : combien de paramètres ont le dense et le profond à 2 et à 50 latents ? Réponse : le dense en a $1\,569\,d + 784$, soit 3 922 et 79 234 ; le profond $1\,133\,072 + 257\,d$, soit 1 133 586 et 1 145 922. Le goulot change à peine la taille du profond (son coût est dans les couches cachées), alors qu'il multiplie par 20 celle du dense : à 50 latents, la courbe compare un dense de 79 000 paramètres à un profond de 1,1 million, entraînés aussi peu longtemps l'un que l'autre.

### Ex 25.23 — L'autoencodeur linéaire face à l'ACP de scikit-learn 🔬
**Réponse** :
```python
def fit_linear_ae_23(xc, k, epochs=40, lr=3e-3, batch_size=256, seed=2523):
    torch.manual_seed(seed)
    model = AutoEncoder(nn.Linear(784, k, bias=False), nn.Linear(k, 784, bias=False))
    train_ae(model, xc, epochs, lr=lr, batch_size=batch_size, seed=seed, loss="mse")
    return model.decoder.weight.detach().cpu().numpy().astype(np.float64)


def subspace_angles_deg_23(A, B):
    return np.degrees(subspace_angles(np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)))
```
**Démarche** : les erreurs de reconstruction sont presque égales (0,02727 pour l'autoencodeur, 0,02725 pour l'ACP, qui est l'optimum). Les angles principaux entre les deux sous-espaces sont petits : 1,6° en moyenne, 6,4° au plus. Les plus grands concernent les dernières directions : la 16ᵉ et la 17ᵉ valeur propre de la covariance sont proches, si bien qu'échanger un peu de l'une contre l'autre coûte très peu d'erreur, et 40 epochs de descente de gradient ne départagent pas tout à fait. Les colonnes du décodeur, elles, ne sont **pas** les axes de l'ACP : la première n'a qu'un cosinus de 0,02 avec le premier axe, et elles ne sont pas orthogonales (jusqu'à 0,10 de cosinus entre deux d'entre elles). Comme le ∂ 25.8 l'a montré, toute transformation inversible du code, compensée par le décodeur, donne la même reconstruction : l'erreur ne fixe que le sous-espace. Un test « mêmes axes » aurait échoué ; un test d'angles entre sous-espaces passe.

**Erreurs fréquentes** : oublier de centrer (sans biais, l'autoencodeur doit alors aussi représenter la moyenne, et son sous-espace n'est plus celui de l'ACP) ; prendre `model.encoder.weight` sans le transposer et le comparer à des colonnes ; comparer des axes un à un avec des cosinus et conclure que l'autoencodeur « n'a rien trouvé ».

**Variante** : orthonormalise les colonnes du décodeur (`np.linalg.qr`), projette les données dessus, et fais une ACP de ces 16 coordonnées. Réponse : ses axes, ramenés dans l'espace des pixels, retrouvent ceux de l'ACP à quelques degrés près : l'autoencodeur avait le bon sous-espace, il ne lui manquait qu'une rotation.

### Ex 25.24 — Autoencodeur convolutif à 147 latents 📦
**Réponse** :
```python
def make_conv_ae_24(relu_code=False):
    encoder = nn.Sequential(nn.Unflatten(1, (1, 28, 28)),
                            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 16 x 14 x 14
                            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8 x 7 x 7
                            nn.Conv2d(8, 3, 3, padding=1))                                  # 3 x 7 x 7: the code
    if relu_code:
        encoder.append(nn.ReLU())                                                            # the book's version
    decoder = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),   # 16 x 14 x 14
                            nn.Upsample(scale_factor=2), nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),  # 1 x 28 x 28
                            nn.Flatten())
    return AutoEncoder(encoder, decoder)
```
**Démarche** : les couches sont celles du ✏️ 25.3 ; `nn.Unflatten` et `nn.Flatten` font le lien avec les lignes de 784 pixels du notebook. Le modèle n'a que 2 132 paramètres (500 fois moins que `AE_DEEP`) pour un code de 147 valeurs (7 fois plus que ses 20 latents). Sans activation sur le code, il atteint en 3 epochs une MSE de test de 0,017, bien meilleure que les 0,027 de `AE_DEEP` : les filtres locaux et partagés conviennent aux images, et un code plus grand porte plus d'information. Avec la ReLU du livre sur le code, l'entraînement reste bloqué sur une image floue (0,100 après 3 epochs) : **toutes** les valeurs du code sont nulles. Les 147 sorties de la dernière convolution sont devenues négatives au début de l'entraînement, la ReLU les a mises à zéro, et sa pente nulle en dessous de 0 empêche tout gradient de les ramener : le décodeur ne reçoit plus rien et rend l'image moyenne. Le livre, avec d'autres poids de départ et 50 epochs, s'en est sorti ; ici, c'est le cas typique du neurone ReLU mort (ch. 17), en pire, puisque tout le goulot meurt. La ligne droite entre les codes d'un 1 et d'un 0 donne un fondu, le 1 s'efface pendant que le 0 apparaît : avec 147 valeurs qui gardent une organisation spatiale, le code ressemble encore à une petite image, et le mélanger revient presque à mélanger des pixels (fiche §25.7.1).

**Erreurs fréquentes** : oublier `nn.Unflatten` (une `Conv2d` refuse une entrée 2-D de 784 colonnes, ou la lit comme une image d'une seule ligne) ; oublier `nn.Flatten` à la fin (la sortie `(N, 1, 28, 28)` ne se compare plus aux lignes de pixels) ; `padding=0` (chaque convolution perd deux pixels et les tailles ne retombent plus sur 28).

**Variante** : remplace la ReLU du code par une `nn.LeakyReLU(0.1)`. Réponse : la pente n'est plus nulle sous 0, le gradient repasse, et l'entraînement ne se bloque plus (MSE de test de 0,028 après 3 epochs, au lieu de 0,100) ; un goulot entièrement mort n'est plus possible.

### Ex 25.25 — Le débruiteur qui apprenait le bruit 🐛
**Réponse** : `faulty_25 = "CFG"` ; la boucle corrigée :
```python
faulty_25 = "CFG"


def train_denoiser_fixed_25(model, x, epochs, sigma=0.5, lr=1e-3, batch_size=128, seed=0,
                            loss_fn=F.binary_cross_entropy):
    x = to_tensor(x)
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch_size):
            clean = x[order[i:i + batch_size]]
            inputs = torch.clamp(clean + sigma * torch.randn(clean.shape, generator=g), 0, 1)  # a new noise, clipped
            loss = loss_fn(model(inputs.to(DEVICE)), clean.to(DEVICE))                        # the target is clean
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
```
**Démarche** : (G) compare la sortie à l'entrée **bruitée** : le réseau apprend à recopier le bruit, l'identité est sa meilleure réponse, et ses images « débruitées » restent pleines de neige ; c'est la faute la plus grave, car rien d'autre ne peut la compenser. (C) tire le bruit **une seule fois**, avant les epochs : chaque image garde le même bruit d'une epoch à l'autre, et un réseau assez grand peut apprendre ce bruit par cœur au lieu de la règle « retirer du bruit » ; on le voit à une loss d'entraînement qui continue de baisser pendant que l'erreur de validation, sur un bruit neuf, remonte. (F) n'écrête pas : des pixels à −0,4 ou à 1,3 n'existent dans aucune image réelle, le réseau apprend sur une distribution d'entrées qu'il ne verra jamais, et la cross-entropy binaire n'est plus définie pour une cible hors de $[0, 1]$. Les trois diagnostics le repèrent sans entraînement, avec un faux modèle qui note ce qu'il reçoit : c'est plus rapide et plus sûr que de juger des images floues.

**Erreurs fréquentes** : marquer (E) (tirer l'ordre des mini-batches avec le même générateur que le bruit est permis) ; corriger (F) en écrêtant la cible au lieu de l'entrée ; tirer le nouveau bruit avec `torch.randn_like(clean)` sans le générateur `g` (le contrat demande `generator=g`, pour que deux appels de même graine donnent le même entraînement).

**Variante** : remplace l'écrêtage par `torch.sigmoid` sur l'entrée bruitée. Réponse : les valeurs restent dans $]0, 1[$, mais tous les pixels sont déformés, même ceux que le bruit n'a pas fait sortir de l'intervalle (un pixel noir devient 0,5) : le réseau doit en plus apprendre à défaire cette déformation. L'écrêtage ne touche que les pixels hors de $[0, 1]$.

### Ex 25.26 — Autoencodeur débruiteur (bruit gaussien, écrêtage) 📦
**Réponse** :
```python
def make_denoiser_26(channels=32):
    c = channels
    encoder = nn.Sequential(nn.Unflatten(1, (1, 28, 28)),
                            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
    decoder = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
                            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
                            nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid(), nn.Flatten())
    return AutoEncoder(encoder, decoder)


def psnr_26(clean, images):
    mse = ((np.asarray(images, dtype=np.float64) - np.asarray(clean, dtype=np.float64)) ** 2).mean(axis=1)
    return float(np.mean(10 * np.log10(1.0 / mse)))
```
**Démarche** : avec 32 filtres (28 353 paramètres) et 3 epochs sur 10 000 vêtements, le PSNR passe de 12,8 dB (images bruitées) à 18,6 dB : un gain de 5,8 dB, soit une MSE divisée par environ $10^{0{,}58} \approx 3{,}8$. Les images débruitées sont propres mais floues : les motifs fins (rayures, logos, coutures) disparaissent avec le bruit, car le réseau ne peut pas distinguer un détail fin d'un grain de bruit après deux réductions de résolution. On moyenne le PSNR image par image parce qu'il est logarithmique : le PSNR de la MSE moyenne serait dominé par les images les plus abîmées, et ne dirait pas la qualité typique d'une image.

**Erreurs fréquentes** : calculer la MSE sur tout le tableau au lieu d'image par image ; oublier le $10$ devant le $\log_{10}$ (des bels, pas des décibels) ; comparer le PSNR des images débruitées à celui des images **propres** contre elles-mêmes (une MSE nulle, un PSNR infini).

**Variante** : avec $\sigma = 0{,}5$, que devient le PSNR des images bruitées ? Réponse : sans écrêtage, il baisserait de $20\log_{10}(0{,}5/0{,}3) \approx 4{,}4$ dB ; avec l'écrêtage, qui limite les plus grosses erreurs, il ne baisse que de 3,5 dB, à 9,3 dB. Un débruiteur entraîné à ce bruit garde un gain, mais ses images perdent encore plus de détails.

### Ex 25.27 — Sur-échantillonnage explicite ou convolutions à pas : temps et qualité 🔬
**Réponse** :
```python
def make_strided_27(channels=32):
    c = channels
    encoder = nn.Sequential(nn.Unflatten(1, (1, 28, 28)),
                            nn.Conv2d(1, c, 3, stride=2, padding=1), nn.ReLU(),        # 28 -> 14
                            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.ReLU())        # 14 -> 7
    decoder = nn.Sequential(nn.ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 7 -> 14
                            nn.ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 14 -> 28
                            nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid(), nn.Flatten())
    return AutoEncoder(encoder, decoder)


def compare_27(makers, epochs, seed=2527):
    results = {}
    for name, maker in makers.items():
        torch.manual_seed(seed)
        model = maker()
        history = train_denoiser(model, X_FASHION[:N_DENOISE], epochs, seed=seed)
        results[name] = {"params": n_params(model), "sec_per_epoch": float(np.median(history["seconds"])),
                         "psnr": psnr(VAL_CLEAN_B, reconstruct(model, VAL_NOISY_B))}
    return results
```
**Démarche** : les deux modèles ont exactement le même nombre de paramètres (28 353 : un `ConvTranspose2d` de $c$ vers $c$ canaux a autant de poids qu'un `Conv2d`). La version à pas va plus vite (environ 6 s par epoch contre 8 à 9 s ici, selon les exécutions) : sa première convolution ne calcule que $14 \times 14$ positions au lieu de $28 \times 28$, et le décodeur n'a pas de convolution à pleine résolution avant le dernier étage. En 2 epochs, les deux débruitent presque aussi bien (17,6 dB contre 17,5 dB) : le livre, lui, attribue le temps le plus court au mauvais modèle (l'encadré ⚠️ de la fiche). Une convolution transposée de stride 2 et de noyau 3 recouvre certains pixels de sortie deux fois et d'autres une seule : elle peut laisser un motif en damier, visible surtout au début de l'entraînement ou sur des fonds unis ; ici, il est à peine perceptible.

**Erreurs fréquentes** : oublier `output_padding=1` (des sorties de 13 puis 25 pixels, ✏️ 25.4) ; mesurer la première epoch seule (elle compte des temps de démarrage) au lieu de la médiane ; conclure sur la qualité après 2 epochs (il faudrait entraîner chacun jusqu'au bout, ou à temps égal).

**Variante** : compare-les à **temps** égal plutôt qu'à epochs égales (par exemple 60 s chacun). Réponse : la version à pas fait environ 35 % d'epochs en plus dans le même temps (8,3 s contre 6,2 s par epoch lors d'une de nos exécutions : $8{,}3/6{,}2 \approx 1{,}35$) ; tant que les deux progressent encore, ces epochs en plus devraient lui donner l'avantage : c'est la comparaison qui compte quand le budget est un temps de calcul.

### Ex 25.28 — La reparamétrisation en NumPy : passe avant et passe arrière 🔨
**Réponse** (`solutions/mylearn_ref/vae.py` ; `_pair` vérifie deux tableaux 2-D sans NaN de même forme et en fait des copies `float64`) :
```python
def reparameterize(mu, log_var, rng=None):
    mu_a, lv = _pair(mu, log_var, ("mu", "log_var"), 2)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    eps = generator.standard_normal(mu_a.shape)
    z = mu_a + np.exp(0.5 * lv) * eps
    return z, eps


def reparameterize_backward(dz, eps, log_var):
    d, e = _pair(dz, eps, ("dz", "eps"), 2)
    _, lv = _pair(dz, log_var, ("dz", "log_var"), 2)
    return d.copy(), d * 0.5 * np.exp(0.5 * lv) * e
```
**Démarche** : $z = \mu + e^{s/2}\varepsilon$ avec $s = \log\sigma^2$ ; donc $\partial z/\partial\mu = 1$ et $\partial z/\partial s = \frac{1}{2}e^{s/2}\varepsilon = \frac{1}{2}\sigma\varepsilon$ (∂ 25.5), multipliés par $\partial L/\partial z$ (la règle de la chaîne). On rend `eps` parce que la passe arrière en a besoin : sans lui, impossible de calculer $\partial z/\partial s$. Ce gradient est nul quand $\varepsilon = 0$ (le tirage n'a pas bougé de la moyenne, la variance n'y a joué aucun rôle), et il s'écrase quand $s$ est très négatif ($\sigma \to 0$) : le code est alors presque déterministe. Sur 200 000 tirages de $\mathcal{N}(2, 9)$, la moyenne vaut 1,993 et la variance 8,978.

**Erreurs fréquentes** : `np.exp(log_var)` au lieu de `np.exp(0.5 * log_var)` (la variance au lieu de l'écart-type) ; recréer le générateur à partir d'un `Generator` reçu (les tirages se répètent d'un appel à l'autre) ; rendre `dz` lui-même comme `dmu` (une vue : modifier l'un modifie l'autre) ; laisser NumPy diffuser `(2, 3)` contre `(1, 3)`.

**Variante** : écris la passe arrière si l'encodeur rendait $\sigma$ directement (avec $z = \mu + \sigma\varepsilon$). Réponse : $\partial z/\partial\sigma = \varepsilon$ ; mais alors il faudrait garantir $\sigma > 0$ (une softplus en sortie), ce que la log-variance évite : elle peut prendre n'importe quelle valeur réelle.

### Ex 25.29 — Divergence KL gaussienne et son gradient 🔨
**Réponse** (`solutions/mylearn_ref/vae.py`) :
```python
def gaussian_kl(mu, log_var):
    mu_a, lv = _pair(mu, log_var, ("mu", "log_var"), 2)
    return 0.5 * np.sum(mu_a ** 2 + np.expm1(lv) - lv, axis=1)


def gaussian_kl_grad(mu, log_var):
    mu_a, lv = _pair(mu, log_var, ("mu", "log_var"), 2)
    return mu_a.copy(), 0.5 * np.expm1(lv)
```
**Démarche** : la KL d'un latent vaut $\frac{1}{2}(\mu^2 + e^{s} - 1 - s)$ (∂ 25.7) ; les latents sont indépendants, donc les KL s'**additionnent** (une moyenne changerait l'échelle de la loss avec le nombre de latents). Ses dérivées : $\mu$ par rapport à $\mu$, et $\frac{1}{2}(e^{s} - 1)$ par rapport à $s$. La KL vaut 0 exactement en $\mu = 0$ et $s = 0$ ($\sigma = 1$), là où les deux dérivées s'annulent : le code suit alors exactement le prior. `np.expm1(s)` calcule $e^{s} - 1$ sans perdre de chiffres quand $s$ est proche de 0, là où `np.exp(s) - 1` soustrait deux nombres presque égaux.

**Erreurs fréquentes** : moyenner sur les latents (`np.mean(..., axis=1)`) ; sommer sur l'axe 0 (une valeur par latent au lieu d'une par exemple) ; écrire $\ln\sigma$ au lieu de $\ln\sigma^2$ (la sortie de l'encodeur est déjà la log-variance).

**Variante** : la KL de $\mathcal{N}(\mu, \sigma^2)$ à $\mathcal{N}(0, \tau^2)$ (un prior plus large). Réponse : $\frac{1}{2}\left(\frac{\mu^2 + \sigma^2}{\tau^2} - 1 - \ln\frac{\sigma^2}{\tau^2}\right)$ ; avec $\tau = 1$, on retrouve la formule du chapitre.

### Ex 25.30 — Un VAE en PyTorch : reconstruction + KL 📦
**Réponse** :
```python
class VAE30(nn.Module):

    def __init__(self, n_latent=20, hidden=(512, 256)):
        super().__init__()
        self.n_latent, self.hidden = n_latent, tuple(hidden)
        layers, n_in = [], 784
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
            n_in = n_out
        self.encoder = nn.Sequential(*layers)
        self.to_mu = nn.Linear(n_in, n_latent)
        self.to_log_var = nn.Linear(n_in, n_latent)
        layers, n_in = [], n_latent
        for n_out in reversed(hidden):
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
            n_in = n_out
        self.decoder = nn.Sequential(*layers, nn.Linear(n_in, 784))

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_log_var(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)        # the reparameterization trick
        return self.decoder(z), mu, log_var


def vae_loss_30(x, logits, mu, log_var, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()
    kl = 0.5 * (mu ** 2 + torch.exp(log_var) - 1 - log_var).sum(dim=1).mean()
    return rec + beta * kl, rec, kl
```
**Démarche** : la classe reprend l'autoencodeur profond avec deux têtes ; `torch.randn_like(mu)` tire $\boldsymbol{\varepsilon}$ hors du graphe, et autograd dérive $\boldsymbol{\mu} + e^{s/2}\boldsymbol{\varepsilon}$ comme tu l'as fait en 25.28. La reconstruction est sommée sur les 784 pixels pour rester à la même échelle que la KL, sommée sur les latents : moyenner la reconstruction sur les pixels reviendrait à multiplier $\beta$ par 784 (fiche, pièges). Après 3 epochs, ton VAE a une reconstruction d'environ 171 nats et une KL d'environ 7 nats par image (environ 10 bits, $7/\ln 2$) : au début, la KL est petite (1,8 nat à la première epoch) parce que l'encodeur n'a encore rien de précis à dire de chaque image, puis elle monte à mesure que les codes deviennent informatifs. `VAE_C`, entraîné 10 epochs, atteint 104 nats de reconstruction et 15 de KL sur le test.

**Erreurs fréquentes** : une sigmoïde à la fin du décodeur **et** `binary_cross_entropy_with_logits` (la sigmoïde est appliquée deux fois) ; `reduction="mean"` sur la reconstruction ; `loss.item()` dans la loss (plus de gradient) ; tirer `eps` avec `torch.randn(mu.shape)` sans le mettre sur le device de `mu` (une erreur sur GPU).

**Variante** : remplace la cross-entropy binaire par une erreur quadratique sommée sur les pixels. Réponse : cela revient à supposer des pixels gaussiens de variance fixe ; l'échelle de la reconstruction change (une MSE sommée vaut quelques dizaines au lieu d'une centaine de nats), donc l'équilibre avec la KL aussi : à $\beta = 1$, la KL pèse relativement plus, et les images sont plus floues.

### Ex 25.31 — Prédire : décoder des tirages N(0, I) avec un AE puis avec un VAE 🔮
**Réponse** : a) **Vrai** ; b) **Faux** ; c) **Faux**.

**Démarche** : décodés par `AE_DEEP`, les 1 000 tirages donnent surtout des 2 (43 %) et des 4 (27 %), et seuls 11 % sont reconnus avec confiance. Les tirages de $\mathcal{N}(\mathbf{0}, \mathbf{I})$ ont une norme d'environ 4,4, alors que les codes de `AE_DEEP` ont une norme moyenne de 14,7 : les tirages tombent tous dans la même petite région près de l'origine, que l'autoencodeur n'a jamais utilisée, et le décodeur y rend quelques mêmes formes ambiguës. Décodés par `VAE_C`, les tirages se répartissent entre les dix chiffres (14 % au plus pour un seul) : les moyennes $\boldsymbol{\mu}$ du VAE ont une norme de 3,5, du même ordre que les tirages, parce que la KL les a rangées près du prior. Mais seuls 31 % sont reconnus avec confiance : ce VAE, entraîné 10 epochs sur 10 000 images, rend des chiffres flous, et un classifieur hésite sur un chiffre flou. Inversement, un classifieur peut être « sûr de lui » sur une tache : il n'a jamais appris à dire « aucun chiffre », sa softmax répartit toute la probabilité entre dix classes. Sur cinq graines, la part du chiffre le plus prédit va de 23 % à 45 % pour l'autoencodeur et de 15 % à 17 % pour le VAE, et la part reconnue avec confiance du VAE de 27 % à 33 % (entrée `p31` de `calib`).

**Erreurs fréquentes** : confondre « reconnu avec confiance » et « est un vrai chiffre » ; penser que le VAE donne des chiffres nets dès qu'il est entraîné (sa netteté dépend de l'entraînement et de $\beta$, 25.36).

**Variante** : décode avec `AE_DEEP` des codes tirés autour de la moyenne de **ses** codes, avec leur écart-type (`codes.mean(axis=0) + codes.std(axis=0) * Z_31`). Réponse : les tirages tombent enfin dans la bonne région, la répartition des chiffres s'étale ; mais les zones vides (25.18) restent, et une partie des images sont des formes hybrides : c'est ce que le VAE organise pour toi.

### Ex 25.32 — Un VAE est stochastique : sorties répétées, bruit relatif, interpolations 🔬
**Réponse** :
```python
def repeat_32(vae, x, k, seed=0):
    vae.eval()
    with torch.inference_mode():
        mu, log_var = vae.encode(to_tensor(x).reshape(1, 784).to(DEVICE))
        torch.manual_seed(seed)
        z = mu + torch.exp(0.5 * log_var) * torch.randn(k, mu.shape[1]).to(DEVICE)
        return torch.sigmoid(vae.decoder(z)).cpu().numpy()


def relative_noise_32(model, mu, frac, seed=0):
    mu = np.asarray(mu, dtype=np.float64)
    u = np.random.default_rng(seed).uniform(-1, 1, size=mu.shape)
    return (mu * (1 + frac * u)).astype(np.float32)
```
**Démarche** : les huit passages du même 2 sont huit codes différents, tirés autour de la même moyenne avec l'écart-type que l'encodeur a choisi pour cette image : les sorties se ressemblent mais varient (écart-type moyen d'un pixel de 0,025 d'un passage à l'autre), surtout le long des bords du trait. Face à un bruit relatif, les deux modèles tolèrent bien 10 % et 30 % (68 % et 67 % de chiffres reconnus avec confiance pour le VAE, 64 % et 63 % pour l'autoencodeur) ; à 100 %, le VAE tient mieux (55 % contre 44 %) : son décodeur a été entraîné sur des codes bruités (chaque passage tire un code), et la KL a resserré les codes, si bien que les codes voisins sont, eux aussi, des chiffres. L'interpolation sphérique d'un 4 à un 9 passe par des formes intermédiaires qui ressemblent à des chiffres à chaque étape.

**Erreurs fréquentes** : appeler `vae(x)` en mode entraînement pour répéter les passages (c'est bien un tirage, mais le contrat demande la graine et le mode évaluation) ; fixer la graine **avant** `encode` (sans effet ici, mais le contrat la fixe juste avant le tirage) ; un bruit additif de taille fixe au lieu d'un bruit proportionnel à la valeur.

**Variante** : décode $\boldsymbol{\mu}$ seul, sans tirage, et compare à la moyenne des huit sorties. Réponse : les deux images sont très proches mais pas égales, car le décodeur n'est pas linéaire : la moyenne des sorties n'est pas la sortie de la moyenne ; la sortie de $\boldsymbol{\mu}$ est un peu plus nette.

### Ex 25.33 — Reproduire le nuage latent d'un VAE 2D et la grille −3…3 (fig. 25.67-25.68) 🎨
**Réponse** :
```python
def plot_vae_map_33(vae, x, labels, lim=3.0, n=15):
    mu = encode_mu(vae, x)
    images = vae_images(vae, grid_codes((-lim, lim, -lim, lim), n)).reshape(n, n, 28, 28)
    mosaic = images.transpose(0, 2, 1, 3).reshape(28 * n, 28 * n)
    fig, (ax_map, ax_grid) = plt.subplots(1, 2, figsize=(12, 5.6))
    colors = wb.plot.class_colors(10)
    for digit in range(10):
        keep = np.asarray(labels) == digit
        ax_map.scatter(mu[keep, 0], mu[keep, 1], s=3, color=colors[digit], label=str(digit))
    for radius in (1, 2, 3):
        ax_map.add_patch(plt.Circle((0, 0), radius, fill=False, color="k", lw=0.8, ls="--"))
    ax_map.set_xlim(-lim, lim)
    ax_map.set_ylim(-lim, lim)
    ax_map.set_aspect("equal")
    ax_map.legend(markerscale=4, fontsize=8, ncol=2, loc="upper right")
    ax_map.set_title("moyennes mu de 2 000 chiffres de test")
    ax_grid.imshow(mosaic, cmap="gray", extent=(-lim, lim, -lim, lim))
    ax_grid.set_title(f"la grille de -{lim:g} à {lim:g}, décodée")
    plt.tight_layout()
    plt.show()
    return mosaic
```
**Démarche** : comparé à la carte de 25.18, le nuage du VAE est centré sur l'origine et à peu près rond, sans grandes zones vides : 34 % des moyennes $\boldsymbol{\mu}$ sont à moins de 1 de l'origine, 69 % à moins de 2, 90 % à moins de 3, au lieu de 39 %, 86 % et 99 % pour des tirages de $\mathcal{N}(\mathbf{0}, \mathbf{I})$. La grille de −3 à 3 se décode en chiffres presque partout, avec des transitions continues d'un chiffre à l'autre ; aux coins, plus loin que 3 de l'origine, les formes se dégradent. Le nuage ne remplit pas tout à fait la gaussienne : la KL pénalise chaque image séparément, en compromis avec la reconstruction, et rien n'oblige l'ensemble des codes à suivre exactement le prior ; ici, ils restent trop resserrés près du centre et laissent des bords peu habités. Avec deux latents seulement, la reconstruction reste médiocre (152 nats) : la KL et le goulot étroit se combinent.

**Erreurs fréquentes** : tracer des codes tirés au lieu des moyennes (un nuage plus épais, qui mélange l'incertitude et la position) ; des axes non égaux (`set_aspect("equal")`), qui rendent les cercles ovales ; une grille construite avec `np.linspace(-lim, lim, n)` pour les $y$ (la mosaïque à l'envers).

**Variante** : trace aussi les codes **tirés** (un passage de `VAE_2D` pour chaque image) et recompte la part à moins de 2 de l'origine. Réponse : elle ne bouge presque pas (69 %) : les $\sigma$ de ce VAE sont petits devant l'étalement des $\boldsymbol{\mu}$, et ajouter le tirage épaissit le nuage sans l'élargir ; ce qui sépare le nuage du prior, c'est sa forme d'ensemble, pas l'incertitude de chaque code.

### Ex 25.34 — Diagnostiquer des courbes reconstruction/KL (effondrement du posterior) 📈
**Réponse** :
```python
collapsed_34, active_34, peak_34, gap_34, mapping_34 = 'C', 1, 45.6, 106.6, 'BDCA'
```
**Démarche** : le run C a une KL nulle à toutes les epochs et aucun latent actif : son posterior s'est effondré, l'encodeur rend le prior pour toutes les images et le décodeur rend l'image moyenne (206 nats de reconstruction, sans progrès après la 2ᵉ epoch). Le run D garde un seul latent actif (KL de 0,4 nat) : un $\beta$ fort laisse passer le strict minimum, et sa reconstruction finale (188,9 nats) dépasse de 113,8 nats celle du run A (75,1), celui dont la KL finit la plus haute. Le run A part d'une KL de 68,0 nats et finit à 22,4 (−45,6) : à la première epoch, $\beta = 0$, rien ne retient la KL, l'encodeur utilise ses 20 latents ; puis $\beta$ monte et la KL baisse, mais 16 latents restent actifs. Le run B, à $\beta = 1$ dès le départ, finit avec une reconstruction moins bonne (82,3 nats) et seulement 12 latents actifs : dès les premières epochs, la KL a « éteint » des latents avant que le décodeur ait appris à s'en servir, et ils ne reviennent plus. C'est exactement ce contre quoi la montée progressive de $\beta$ protège. Le classement par KL finale donne la correspondance : $\beta = 1$ → B, $\beta = 20$ → D, $\beta = 100$ → C, $\beta$ montant → A, soit `"BDCA"`. En entreprise, on surveille la KL **par latent** (ou le nombre de latents actifs) dès les premières epochs : un effondrement se voit tôt, une KL totale qui tombe vers 0 aussi.

**Erreurs fréquentes** : prendre le run D pour l'effondrement (un seul latent actif, ce n'est pas zéro) ; lire `active[0]` au lieu de `active[-1]` ; soustraire dans le mauvais sens en c) ou d) ; associer le run A à $\beta = 1$ parce que les deux finissent avec des courbes proches.

**Variante** : que deviendrait le run B avec un décodeur beaucoup plus puissant (capable de dessiner seul des chiffres plausibles, comme un décodeur autorégressif) ? Réponse : il risquerait l'effondrement même à $\beta = 1$ : si le décodeur réussit sans le code, la KL est un coût sans contrepartie, et l'entraînement la pousse à 0.

### Ex 25.35 — Exporter le décodeur comme générateur autonome et le tester 🛠️
**Réponse** :
```python
def export_decoder_35(vae, path):
    torch.save({"n_latent": vae.n_latent, "hidden": list(vae.hidden), "state_dict": vae.decoder.state_dict()}, path)


def load_generator_35(path):
    saved = torch.load(path, weights_only=True)
    layers, n_in = [], saved["n_latent"]
    for n_out in reversed(saved["hidden"]):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    decoder = nn.Sequential(*layers, nn.Linear(n_in, 784))
    decoder.load_state_dict(saved["state_dict"])
    decoder.eval()

    def generate(n, seed=0):
        z = torch.randn(n, saved["n_latent"], generator=torch.Generator().manual_seed(seed))
        with torch.inference_mode():
            images = torch.round(torch.sigmoid(decoder(z)) * 255)
        return images.numpy().astype(np.uint8).reshape(n, 28, 28)

    return generate


def test_generator_shape_35():
    images = load_generator_35(GEN_PATH)(5)
    assert images.shape == (5, 28, 28), f"expected (5, 28, 28), got {images.shape}"
    assert images.dtype == np.uint8, f"expected uint8, got {images.dtype}"


def test_generator_seed_35():
    generate = load_generator_35(GEN_PATH)
    assert np.array_equal(generate(3, seed=7), generate(3, seed=7)), "the same seed must give the same images"
    assert not np.array_equal(generate(3, seed=7), generate(3, seed=8)), "two seeds must give different images"


GEN_PATH = str(WORK_DIR / "generator_35.pt")
with checked("25.35"):
    export_decoder_35(VAE_C.cpu(), GEN_PATH)
    VAE_C.to(DEVICE)
    generate_35 = load_generator_35(GEN_PATH)
    if returned("25.35", "load_generator_35", generate_35):
        images_35 = generate_35(6, seed=1)
        z_35 = torch.randn(6, 20, generator=torch.Generator().manual_seed(1))
        want_35 = np.round(vae_images(VAE_C, z_35.numpy()) * 255).astype(np.uint8).reshape(6, 28, 28)
        verdict("25.35", isinstance(images_35, np.ndarray) and images_35.dtype == np.uint8 and images_35.shape == (6, 28, 28)
                and np.abs(images_35.astype(int) - want_35.astype(int)).max() <= 1,
                "le générateur rechargé rend les images du décodeur de VAE_C, en uint8 (6, 28, 28).",
                f"attendu un tableau uint8 (6, 28, 28) égal aux images du décodeur de VAE_C pour la graine 1, reçu "
                f"{type(images_35).__name__} {shape_text(images_35)}.")
        show_rows([generate_35(10, seed=0) / 255.0], ["generate(10)"])
    PRE_35 = f"import torch\nimport torch.nn as nn\nGEN_PATH = {GEN_PATH!r}"
    TESTS_35 = [test_generator_shape_35, test_generator_seed_35]
    import inspect
    if all("NotImplementedError" not in inspect.getsource(test) for test in TESTS_35):
        print("Tes tests, sur ton générateur (ils doivent passer) :")
        result_35 = wb.run_pytest(TESTS_35, subject=load_generator_35, preamble=PRE_35)
        verdict("25.35", result_35.ok, "tes tests passent sur ton générateur.", "tes tests doivent passer sur ton générateur.")
        for broken_35, why_35 in [(broken_float_35, "un générateur qui rend des flottants dans [0, 1]"),
                                  (broken_seed_35, "un générateur qui ignore la graine")]:
            caught_35 = not wb.run_pytest(TESTS_35, subject=broken_35, name="load_generator_35", preamble=PRE_35, quiet=True).ok
            verdict("25.35", caught_35, f"tes tests attrapent {why_35}.", f"tes tests laissent passer {why_35}.")
    else:
        print("⏳ Ex 25.35 : écris tes deux tests.")
```
**Démarche** : le fichier contient tout ce qu'il faut pour reconstruire le décodeur : sa taille (`n_latent`, `hidden`) et ses poids. Le générateur rechargé rend exactement les images du décodeur de `VAE_C` pour la même graine (à un niveau de gris près au plus, l'arrondi). Les deux tests passent sur ce générateur, et chacun attrape un des deux générateurs cassés : le premier vérifie le `dtype` (les flottants dans $[0, 1]$ ne sont pas des `uint8`), le second l'effet de la graine (un générateur qui l'ignore rend les mêmes images pour deux graines). Un test qui ne vérifie que « ça ne plante pas » n'aurait attrapé ni l'un ni l'autre. `weights_only=True` interdit au chargement d'exécuter du code arbitraire caché dans le fichier (un fichier PyTorch ordinaire est un *pickle*) : c'est le défaut de `torch.load` depuis PyTorch 2.6, et une bonne habitude pour un fichier reçu. Pour une autre équipe, il faudrait livrer aussi une fiche modèle (ch. 23) : les données d'entraînement, ce que le générateur sait faire et ne sait pas faire, la version de PyTorch, et un exemple d'utilisation.

**Erreurs fréquentes** : sauvegarder tout le modèle avec `torch.save(vae)` (le fichier dépend alors de la classe `VAE` du notebook) ; oublier `eval()` (sans effet ici, mais un décodeur avec du dropout ou une batchnorm donnerait d'autres images) ; utiliser le générateur global (`torch.manual_seed`) au lieu d'un `torch.Generator` local, qui change l'état aléatoire de tout le programme ; un test qui compare deux appels avec des graines différentes sans vérifier qu'ils diffèrent.

**Variante** : ajoute un test qui vérifie que les images ne sont pas toutes noires ou toutes blanches (par exemple, une moyenne des pixels entre 10 et 200). Réponse : il attraperait un décodeur dont les poids n'ont pas été chargés (un décodeur neuf rend un gris uniforme proche de 128, que ce test seul ne verrait pas ; il faudrait aussi vérifier une variance minimale entre les images).

### Ex 25.36 — β-VAE : échanger la netteté contre la régularité 🔬
**Réponse** :
```python
def beta_sweep_36(betas, epochs, seed=2536, x=None, n_latent=16):
    x = X_FASHION if x is None else x
    results = {}
    for beta in betas:
        torch.manual_seed(seed)
        model = VAE(n_latent)
        train_vae(model, x, epochs, beta=beta, seed=seed)
        rec, kl, per_latent = vae_eval(model, XF_TEST)
        results[beta] = {"rec": rec, "kl": kl, "active": int(np.sum(per_latent > 0.05)), "model": model}
    return results
```
**Démarche** : quand $\beta$ passe de 0,5 à 16, la KL baisse de 17,6 à 2,2 nats et la reconstruction monte de 249,5 à 295,4 nats : on échange de l'information dans le code contre de la régularité. Les latents actifs passent de 16 à 4 : à $\beta = 16$, le VAE n'utilise plus que quelques directions, qui portent les facteurs les plus importants (la silhouette, la luminosité), et les reconstructions deviennent des formes moyennes floues ; les tirages de $\mathcal{N}(\mathbf{0}, \mathbf{I})$ sont plus homogènes et plus « typiques ». À $\beta$ petit, les reconstructions sont plus fidèles, mais les tirages au hasard sont moins réussis : les codes s'éloignent du prior. Pour un générateur, on choisit un $\beta$ autour de 1 (ou un peu moins avec plus de latents) en regardant les tirages ; pour un détecteur d'anomalies, un $\beta$ plus petit, car c'est la précision de la reconstruction qui fait le score. Les chiffres viennent de 4 epochs sur 10 000 vêtements : les tendances sont nettes, les valeurs ne sont pas finales.

**Erreurs fréquentes** : comparer les reconstructions à la loss totale (qui contient $\beta$ fois la KL : elle n'est pas comparable d'un $\beta$ à l'autre) ; oublier la graine avant `VAE(n_latent)` (les modèles diffèrent alors aussi par leurs poids de départ) ; compter comme actifs les latents à KL strictement positive (toute KL calculée est un peu positive).

**Variante** : entraîne $\beta = 4$ avec 32 latents au lieu de 16. Réponse : la KL totale reste du même ordre (4,5 nats contre 5,1), alors que les latents actifs passent de 6 à 10 : un $\beta$ fort limite d'abord l'information **totale** du code, que le VAE répartit alors sur un peu plus de latents.

### Ex 25.37 — Détecter des anomalies par l'erreur de reconstruction 🔬
**Réponse** :
```python
def fit_without_37(digit, epochs, seed=2537):
    torch.manual_seed(seed)
    model = make_deep_ae(20)
    train_ae(model, X_MNIST[Y_MNIST != digit], epochs, seed=seed)
    return model


def scores_37(model, x):
    return mse_per_image(reconstruct(model, x), x)


def auc_37(normal, anomalies):
    y_true = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomalies), dtype=int)])
    return float(mylearn.metrics.roc_auc(y_true, np.concatenate([normal, anomalies])))
```
**Démarche** : sans avoir vu de 7, l'autoencodeur les reconstruit souvent comme des 1 ou des 9 assez proches : ROC-AUC de 0,676 seulement, les 7 sont des anomalies « proches ». Les vêtements, très différents, sont bien repérés (0,906). Les chiffres très pâles (15 % de l'intensité) donnent une ROC-AUC de 0,285, **sous** 0,5 : ils sont faciles à reconstruire (presque tous les pixels sont proches de 0), et leur erreur est plus **petite** que celle des chiffres normaux. C'est la limite de toute détection par reconstruction (25.Q11) : un score faible veut dire « facile à reconstruire », pas « normal » ; une image presque vide, un fond uni, passent inaperçus. En pratique, on choisit le seuil sur une validation, à partir du taux de fausses alertes acceptable (un percentile des scores des données normales), et on mesure la precision et le recall des alertes (ch. 3), pas seulement la ROC-AUC.

**Erreurs fréquentes** : entraîner sur tous les chiffres (le 7 n'est plus une anomalie) ; inverser les labels (une ROC-AUC de $1 - x$) ; lire une ROC-AUC sous 0,5 comme « aléatoire » (elle dit que le score range systématiquement à l'envers).

**Variante** : normalise le score par l'intensité de l'image (la MSE divisée par la moyenne des pixels au carré). Réponse : les chiffres pâles sont tous repérés (ROC-AUC de 1,0), et les 7 mieux qu'avant (0,82 au lieu de 0,68) ; mais les vêtements le sont un peu moins (0,87 au lieu de 0,91), car leurs grandes surfaces claires gonflent le dénominateur. Chaque correction d'un score a ses propres angles morts, d'où l'intérêt de tester un détecteur sur plusieurs familles d'anomalies.

### Ex 25.38 — Un VAE entièrement en NumPy avec mylearn 🔨
**Réponse** :
```python
def _relu_38(z):
    return np.maximum(z, 0.0)


def _relu_derivative_38(z):
    return (z > 0).astype(float)


def init_numpy_vae_38(n_latent=8, hidden=128, seed=0):
    rng = np.random.default_rng(seed)
    return {"enc": mylearn.nn.layers.init_mlp([784, hidden, 2 * n_latent], rng=rng),
            "dec": mylearn.nn.layers.init_mlp([n_latent, hidden, 784], rng=rng)}


def numpy_vae_loss_38(params, x, eps):
    layers, backward, vae = mylearn.nn.layers, mylearn.nn.backward, mylearn.vae
    n, d = eps.shape
    h, cache_enc = layers.mlp_forward(x, params["enc"], activation=_relu_38, return_cache=True)
    mu, log_var = h[:, :d], h[:, d:]
    z = mu + np.exp(0.5 * log_var) * eps                                   # reparameterize, with eps given
    logits, cache_dec = layers.mlp_forward(z, params["dec"], activation=_relu_38, return_cache=True)
    mean_bce, dlogits = backward.binary_cross_entropy_with_logits(logits, x)
    rec, dlogits = 784 * mean_bce, 784 * dlogits                            # summed over pixels, mean over batch
    kl = float(np.mean(vae.gaussian_kl(mu, log_var)))
    grads_dec, dz = backward.mlp_backward(dlogits, cache_dec, params["dec"], activation_derivative=_relu_derivative_38,
                                          return_input_grad=True)
    dmu, dlog_var = vae.reparameterize_backward(dz, eps, log_var)           # through z
    gmu, glog_var = vae.gaussian_kl_grad(mu, log_var)                       # through the KL, averaged over the batch
    dh = np.concatenate([dmu + gmu / n, dlog_var + glog_var / n], axis=1)
    grads_enc = backward.mlp_backward(dh, cache_enc, params["enc"], activation_derivative=_relu_derivative_38)
    return rec + kl, rec, kl, {"enc": grads_enc, "dec": grads_dec}


def train_numpy_vae_38(params, x, epochs, lr=1e-3, batch_size=128, seed=0):
    flat = [p for part in ("enc", "dec") for layer in params[part] for p in layer]
    optimizer = mylearn.optim.Adam(flat, lr=lr)
    rng = np.random.default_rng(seed)
    d = params["dec"][0][0].shape[0]
    x = np.asarray(x, dtype=np.float64)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            xb = x[order[i:i + batch_size]]
            loss, _, _, grads = numpy_vae_loss_38(params, xb, rng.standard_normal((len(xb), d)))
            optimizer.step([g for part in ("enc", "dec") for layer in grads[part] for g in layer])
            total += loss * len(xb)
        losses.append(total / len(x))
    return losses
```
**Démarche** : le gradient arrive sur $\boldsymbol{\mu}$ et $\log\boldsymbol{\sigma}^2$ par deux chemins : la reconstruction (à travers le décodeur, puis $\mathbf{z}$, avec `reparameterize_backward`) et la KL (`gaussian_kl_grad`, divisé par $n$ parce que la KL est moyennée sur le batch) ; on additionne les deux avant de remonter dans l'encodeur. La vérification contre `torch.autograd` vient **avant** l'entraînement : une erreur de gradient (un facteur 784 oublié, un $1/n$ en trop) donne un VAE qui apprend quand même un peu, et rien ne la signalerait ; comparée à un oracle sur un petit cas, elle se voit tout de suite, et à la bonne couche. Entraîné 5 epochs sur 5 000 chiffres (3 s), ton VAE NumPy fait passer sa loss de 414 à 173 nats, et décode des taches qui commencent à ressembler à des chiffres. Le VAE PyTorch de 25.30, plus grand, prend environ 1,2 s par epoch sur 10 000 images, ton VAE NumPy environ 0,6 s sur 5 000 : pour de petites couches denses sur un CPU, NumPy n'est pas plus lent, mais on y écrit soi-même chaque gradient.

**Erreurs fréquentes** : oublier le facteur 784 sur le gradient de la reconstruction (la loss est sommée sur les pixels, la fonction de `mylearn` rend une moyenne) ; ne pas diviser le gradient de la KL par $n$ ; oublier `return_input_grad=True` sur le décodeur (pas de $\partial L/\partial \mathbf{z}$) ; donner à Adam les gradients dans un autre ordre que les paramètres.

**Variante** : remplace `eps` par des zéros dans `numpy_vae_loss_38`. Réponse : $\mathbf{z} = \boldsymbol{\mu}$, le gradient par rapport à $\log\boldsymbol{\sigma}^2$ ne vient plus que de la KL, qui pousse $\sigma$ vers 1 sans rien en face : c'est un autoencodeur déterministe régularisé, plus un VAE.

### Ex 25.39 — Défi débruitage Fashion-MNIST : le meilleur PSNR sous contrainte 🏆
**Réponse** (une solution parmi d'autres) :
```python
def make_denoiser_39():
    c = 32                                            # no pooling: every layer keeps the 28 x 28 resolution
    layers = [nn.Unflatten(1, (1, 28, 28)), nn.Conv2d(1, c, 3, padding=1), nn.ReLU()]
    for _ in range(3):
        layers += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU()]
    return AutoEncoder(nn.Sequential(*layers), nn.Sequential(nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid(), nn.Flatten()))


def train_39(model, x_clean):
    train_denoiser(model, x_clean, 3 if FAST_MODE else 2, lr=2e-3, seed=2539)
```
**Démarche** : le débruiteur de 25.26 plafonne vers 18,6 dB : ses deux poolings perdent la résolution, et le décodeur doit réinventer les détails. Un réseau qui garde la résolution $28 \times 28$ à chaque couche (quatre convolutions de 32 filtres, une convolution de sortie, 28 353 paramètres, autant que celui de 25.26) n'a rien à réinventer : chaque pixel de sortie voit un voisinage de $11 \times 11$ pixels d'entrée (cinq convolutions $3 \times 3$), assez pour estimer le bruit. Entraîné 3 epochs sur les 10 000 vêtements (51 s), il atteint 20,1 dB en validation et 20,2 dB sur le test caché. L'objectif, 19,3 dB, est la moyenne de cette recette sur trois graines (19,6 à 20,1 dB en validation) moins 0,5 dB (`tools/chapters/history_ch25.json`, entrée `p39`). Les leviers suivants, dans l'ordre de leur coût : plus d'epochs (le temps le permet), un learning rate un peu plus grand au début puis décroissant, un peu plus de largeur ; une connexion qui ajoute l'entrée à la sortie (le réseau prédit alors le bruit à retirer) aide aussi.

**Erreurs fréquentes** : régler l'architecture sur le test caché (il n'est révélé qu'une fois : la validation sert à choisir) ; entraîner avec un autre bruit que celui du défi ; dépasser le temps en oubliant que la première epoch inclut des temps de démarrage.

**Variante** : mesure le PSNR de validation de ta solution avec un bruit de $\sigma = 0{,}2$ et de $\sigma = 0{,}4$ sans la réentraîner. Réponse : à $\sigma = 0{,}2$, le gain reste positif mais le réseau lisse trop (il retire un bruit plus fort qu'il n'y en a) ; à $\sigma = 0{,}4$, il laisse du bruit : un débruiteur est spécialisé pour le bruit qu'il a vu, d'où l'intérêt d'entraîner sur une plage de $\sigma$.
