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
**Erreurs fréquentes** : écrire $H_{in}\,s$ au lieu de $(H_{in} - 1)\,s$ ; ajouter le padding au lieu de le retirer ; oublier le $+ 1$ de la formule de `Conv2d` ; prendre la forme des poids de `Conv2d` en h) (même total ici, mais pas en général : ce sont les biais qui changent, 16 et non 32).
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
**Variante** : un code à deux latents de moyennes nulles et de variances $0{,}5$ et $2$ a une KL de $\tfrac{1}{2}(0{,}5 - 1 + \ln 2) + \tfrac{1}{2}(2 - 1 - \ln 2) = $ **0,25** nat : trop étroit et trop large coûtent, ici, presque autant.

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

Les solutions des exercices du notebook (25.14 à 25.39) seront ajoutées à la prochaine session de génération, avec les parties A à C du notebook.
