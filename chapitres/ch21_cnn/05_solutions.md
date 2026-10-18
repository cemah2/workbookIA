# 21 · Réseaux convolutifs (CNN) — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels, des exercices ✏️ 21.1 à 21.6 et de l'estimation 🧮 21.8 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 21.Q1 — Les deux sens de « profondeur »
a) **3** : un canal par couleur primaire (rouge, vert, bleu) ; la taille, 224 × 224, n'est pas la profondeur. b) **64** : dans `(C, H, W)`, le premier nombre est le nombre de canaux, un par filtre. c) **B** : la profondeur d'un **réseau** est son nombre de couches de calcul (13 convolutions et 3 couches denses pour VGG16). d) **3 072** : $3 \times 32 \times 32$, trois valeurs par pixel. e) **Faux** : un CNN contient au moins une convolution, et presque toujours aussi des poolings, des activations et une ou plusieurs couches denses à la fin.
**À retenir** : « profondeur » d'un tenseur = nombre de canaux ; « profondeur » d'un réseau = nombre de couches. Le contexte dit lequel.

### 21.Q2 — Un seul neurone promené sur toute l'image
a) **Vrai** : c'est le partage de poids, un même neurone (les mêmes poids) évalue toutes les positions. b) **25** : $5 \times 5$ poids sur un seul canal. c) **Faux** : les 25 poids sont les mêmes pour une image de 100 × 100 ou de 1 000 × 1 000 ; seul le nombre de **positions** change. d) **B** : le pixel focal est sous l'ancre ; le champ récepteur est tout le carré lu. e) **B** : un côté impair a un centre, où l'on pose l'ancre, et le padding qui garde la taille est le même des deux côtés.
**À retenir** : le nombre de poids d'un filtre dépend de son côté et du nombre de canaux d'entrée, jamais de la taille de l'image.

### 21.Q3 — Convolution ou corrélation croisée ?
a) **Faux** : `nn.Conv2d` calcule une corrélation croisée, sans retournement ; comme les poids sont appris, le retournement ne changerait que leur ordre. b) **−2** : $1 \times 1 + 2 \times 0 + 3 \times (-1)$. c) **2** : le noyau retourné vaut $(-1, 0, 1)$, et $-1 + 0 + 3 = 2$. d) **Vrai** : un noyau symétrique est égal à son retourné. e) **16** : $(6 - 3 + 1)^2 = 4 \times 4$ morceaux. f) **27** : $3 \times 3$ valeurs par canal, sur 3 canaux.
**À retenir** : dans les bibliothèques, « convolution » veut dire corrélation croisée ; le retournement réapparaît dans la passe arrière.

### 21.Q4 — Détecteurs de motifs et hiérarchie de filtres
a) **3** : on gagne 1 par pixel blanc sous un $+1$ (3 au plus), on perd 1 par pixel blanc sous un $-1$ ; le maximum est atteint quand la colonne de gauche est blanche et tout le reste noir. b) **2** : $3 - 1$, le pixel blanc du centre étant sous un $-1$. c) **Faux** : les poids des filtres sont appris par la rétropropagation, comme ceux d'une couche dense. d) **B** : après un pooling ou un stride, chaque entrée de la couche haute résume déjà une région de l'image. e) **B** : une feature est ici un motif que cherche le filtre.
**À retenir** : un filtre est un détecteur de motif ; son score est grand quand le morceau d'image ressemble à ses poids.

### 21.Q5 — Padding et stride : que devient la taille ?
a) **8** : $12 - 5 + 1$. b) **3** : $(7 - 1)/2$ de chaque côté. c) **3** : $\lfloor (7 - 3)/2 \rfloor + 1$. d) **4** : $\lfloor (12 - 3)/3 \rfloor + 1$. e) **Vrai** : avec un stride égal au côté, les fenêtres se touchent sans se recouvrir, et $9 = 3 \times 3$ tombe juste.
**À retenir** : $H_{out} = \lfloor (H + 2p - k)/s \rfloor + 1$, direction par direction ; le padding « same » vaut $(k - 1)/2$ pour un stride de 1.

### 21.Q6 — Canaux d'entrée, nombre de filtres, canaux de sortie
a) **6** : un filtre a la profondeur de son entrée. b) **4** : un canal de sortie par filtre. c) **216** : $4 \times 6 \times 3 \times 3$. d) **220** : un biais par filtre. e) **54** : 6 filtres d'**un** canal, $6 \times 9$.
**À retenir** : poids d'une convolution standard $= C_{out} \times C_{in} \times k_h \times k_w$ ; avec `groups=g`, on divise par $g$.

### 21.Q7 — Convolution 1D ou convolution 1×1 ?
a) **150** : 3 mots de 50 nombres. b) **18** : $20 - 3 + 1$. c) **52 500** : $300 \times 175$, un poids par couple de canaux. d) **Vrai** : un filtre 1 × 1 ne lit que les canaux d'un pixel. e) **28** : la taille de l'image ne change pas.
**À retenir** : une convolution 1D glisse dans une seule direction ; une convolution 1 × 1 est une couche dense appliquée à chaque pixel, avec les mêmes poids partout.

### 21.Q8 — Couche de convolution et initialisation des filtres
a) **Vrai** : avec des poids et des biais tous égaux, les filtres reçoivent des gradients égaux et restent identiques ; il faut un tirage aléatoire pour briser la symétrie (PyTorch tire au hasard les poids **et** les biais). b) **576** : $64 \times 3 \times 3$. c) **0,059** : $\sqrt{2/576} = 1/\sqrt{288} = 1/(12\sqrt2) \approx 0{,}0589$. d) **B** : une carte de caractéristiques par filtre. e) **Faux** : `padding=0` par défaut, et la sortie perd 2 lignes et 2 colonnes.
**À retenir** : le fan-in d'une convolution est $C_{in}\,k_h\,k_w$ ; c'est lui qu'utilisent He et l'initialisation par défaut de PyTorch.

### 21.Q9 — Convolution transposée : les mots piégés
a) **Faux** : `dilation` espace les **poids** du noyau ; insérer des zéros entre les pixels de l'entrée, c'est la convolution transposée. b) **9** : $(4 - 1) \times 2 + 3$. c) **7** : $9 - 2 \times 1$. d) **2** : $s - 1$. e) **Vrai** : quand $k$ n'est pas un multiple de $s$, certaines cases reçoivent plus de tampons que leurs voisines. f) **7** : $d(k - 1) + 1 = 3 \times 2 + 1$.
**À retenir** : convolution transposée, $H_{out} = (H - 1)\,s - 2p + k$ ; dilatation, un noyau qui couvre $d(k - 1) + 1$ pixels.

### 21.Q10 — VGG16 et ce que voient ses filtres
a) **« (256, 28, 28) »** : trois poolings ($224/2^3 = 28$), 256 filtres dans le bloc 3. b) **5** : un par bloc. c) **14** : le bloc 5 vient après quatre poolings, $224/2^4$. d) **B** : on fixe les poids et l'on modifie les pixels pour augmenter la sortie du filtre. e) **Vrai** : bords et rayures en bas, textures et motifs plus complexes en haut.
**À retenir** : dans VGG16, le côté est divisé par 2 à chaque bloc et le nombre de canaux doublé (jusqu'à 512).

### 21.Q11 — Exemples adversariaux : vrai ou faux ?
a) **Faux** : elle est en général invisible à l'œil nu, c'est ce qui la rend inquiétante. b) **Vrai** : c'est la définition d'une perturbation universelle. c) **Faux** : les meilleurs réseaux, très bien entraînés, sont trompés aussi. d) **23,52** : chaque pixel bouge d'exactement $\varepsilon$ (le gradient n'est jamais nul et rien n'est ramené par le `clamp`), et $0{,}03 \times 784 = 23{,}52$. e) **B** : une attaque ciblée.
**À retenir** : FGSM déplace chaque pixel d'au plus $\varepsilon$ ; sur beaucoup de pixels, l'effet sur un score s'additionne.

### 21.Q12 — 🕰️ Après VGG16 : ResNet, ViT et modèles pré-entraînés
a) **B** : les connexions résiduelles. b) **196** : $(224/16)^2 = 14^2$. c) **512** : une moyenne par canal. d) **Vrai** : `DEFAULT` désigne les meilleurs poids publiés, pré-entraînés sur ImageNet.
**À retenir** : un CNN moderne finit par un pooling moyen global et une seule couche dense, et on part presque toujours de poids pré-entraînés.

<a id="rappels"></a>

## 🔁 Rappels

### 21.R1 — Ch. 20 : ce que changent model.train() et model.eval() pour le dropout
a) **Vrai** : en évaluation, le dropout inversé est l'identité. b) **1,3333** : $1/(1 - 0{,}25)$. c) **25** : en moyenne, une valeur sur quatre. d) **A** : avec le dropout actif, les prédictions sont bruitées, et l'accuracy en général un peu plus basse et variable d'une exécution à l'autre.
**À retenir** : `model.train()` et `model.eval()` changent le comportement du dropout (et de la batchnorm) ; on les appelle explicitement, chaque fois.

### 21.R2 — Ch. 18 : le gradient d'un poids utilisé à plusieurs endroits
a) **6** : $\partial L/\partial w = \sum_i x_i = 1 + 2 + 3$. b) **5** : $1 \times 1 + (-1) \times 2 + 2 \times 3$. c) **Vrai** : `backward` accumule avec `+=`.
**À retenir** : un poids partagé reçoit la **somme** des gradients de toutes ses utilisations ; c'est le gradient d'un filtre de convolution.

### 21.R3 — 0B : le produit scalaire, ce que calcule un filtre en chaque position
a) **1** : $3 + 0 - 2$. b) **5** : $0 + 2 + 3 + 0$. c) **Vrai** : $\mathbf{w} \cdot \mathbf{x} = \lVert\mathbf{w}\rVert\,\lVert\mathbf{x}\rVert \cos\theta$, maximal pour $\theta = 0$.
**À retenir** : en chaque position, un filtre calcule un produit scalaire entre ses poids et un morceau d'image : il mesure leur ressemblance.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 21.1 — Le détecteur de jaune à la main ✏️
**Réponses** : a) **1,65** ; b) **1,5** ; c) **1** ; d) **0** ; e) **A** ; f) **2** ; g) **−1** ; h) **24** ; i) **Vrai**.
**Démarche** : a) $0{,}9 + 0{,}85 - 0{,}1$. b) $1 + 0{,}5 - 0$. c) $1 + 1 - 1$. d) Tout est nul. e) 1,65 est le plus grand des quatre scores. f) Le score croît avec $R$ et $G$ et décroît avec $B$ : le maximum est en $(1, 1, 0)$, le jaune pur, $1 + 1 - 0 = 2$. g) Un détecteur de magenta veut du rouge et du bleu sans vert : $(+1, -1, +1)$. h) Un score par pixel : $4 \times 6$. i) Le même neurone lit les 3 canaux d'un pixel, sans voisin, en chaque position : une convolution 1 × 1 de 3 canaux vers 1.
**Dans ta copie** : le blanc a autant de rouge et de vert que le jaune pur, mais aussi du bleu, que le poids $-1$ retire : le neurone mesure « du rouge et du vert, sans bleu ». Un détecteur de gris clair aurait les poids $(1, 1, 1)$ (score $3v$ pour un gris $(v, v, v)$) : il distingue bien un gris clair (2,4) d'un blanc (3), mais il préfère le blanc ; **aucun** neurone linéaire, même avec un biais, ne peut donner à un gris clair $(0{,}8 ;\ 0{,}8 ;\ 0{,}8)$ un score plus grand qu'à la fois au blanc et au gris moyen : le long des gris, le score $v \sum_c w_c + b$ est une fonction affine de $v$, donc monotone, et son maximum est au noir ou au blanc. Pour reconnaître une **plage** de valeurs, il faut au moins deux neurones et une non-linéarité, par exemple $\mathrm{ReLU}(v - a) - 2\,\mathrm{ReLU}(v - c)$ (avec $a < c$), qui monte jusqu'à $c$ puis redescend.
**Erreurs fréquentes** : additionner les trois composantes (oublier le signe du bleu) ; croire que le blanc, « qui contient du jaune », a le score maximal ; compter $4 \times 6 \times 3$ scores.
**Variante** : avec un biais de $-1{,}5$ (le score devient $R + G - B - 1{,}5$, suivi d'une ReLU), le pixel de a) obtient **0,15** et l'orange **0** : seuls les jaunes assez purs passent le seuil.

### Ex 21.2 — Une corrélation 5×5 par 3×3 à la main ✏️
**Réponses** : a) **3** ; b) **9** ; c) **0** ; d) **5** ; e) **−4** ; f) **« (1, 2) »** ; g) **−2** ; h) **−8** ; i) **−9** ; j) **Vrai**.
**Démarche** : $\mathbf{K}$ ajoute la colonne de gauche du morceau, avec les poids $(1, 2, 1)$, et retire la colonne de droite avec les mêmes poids ; la colonne du milieu ne compte pas. a) $5 - 3 + 1$. b) Morceau des lignes 0 à 2 et colonnes 0 à 2 : gauche $(3, 2, 4)$, soit $3 + 4 + 4 = 11$ ; droite $(0, 1, 0)$, soit $2$ ; $11 - 2 = 9$. c) Lignes 1 à 3, colonnes 1 à 3 : gauche $(0, 1, 2)$ donne 4, droite $(2, 1, 0)$ donne 4 : 0. d) Lignes 2 à 4, colonnes 1 à 3 : gauche $(1, 2, 3)$ donne 8, droite $(1, 0, 2)$ donne 3 : 5. e) et f) La sortie complète est
$$\mathbf{Y} = \begin{pmatrix} 9 & -3 & -2 \\ 8 & 0 & -4 \\ 1 & 5 & -2 \end{pmatrix},$$
de minimum $-4$ en ligne 1, colonne 2. g) Avec le padding, la fenêtre de la position $(0, 0)$ est centrée sur $X_{0,0}$ : sa colonne de gauche est faite de zéros, sa colonne de droite vaut $(0, X_{0,1}, X_{1,1}) = (0, 1, 0)$, d'où $-2$. h) La fenêtre est centrée sur $X_{4,0}$ : colonne de gauche nulle, colonne de droite $(X_{3,1}, X_{4,1}, 0) = (2, 3, 0)$, d'où $-(2 + 2 \times 3) = -8$. i) Retourner $\mathbf{K}$ dans les deux directions donne $\begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} = -\mathbf{K}$ : $-9$. j) Le noyau retourné est $-\mathbf{K}$, donc la convolution vaut $-\mathbf{Y}$ partout.
**Dans ta copie** : $\mathbf{K}$ calcule « colonne de gauche moins colonne de droite » (avec plus de poids au centre) : de grandes valeurs positives pour un **bord vertical** clair à gauche et sombre à droite, de grandes valeurs négatives pour l'inverse. Dans $\mathbf{X}$, la première colonne $(3, 2, 4, 1, 0)$ est plus claire que la colonne 2 en haut, d'où $Y_{0,0} = 9$ et $Y_{1,0} = 8$ ; à droite, au milieu (lignes 1 à 3), la colonne 4 (avec le 3 de la ligne 2) est plus claire que la colonne 2, d'où $Y_{1,2} = -4$.
**Erreurs fréquentes** : retourner le noyau (on obtient $-9$ en b)) ; oublier les 2 de la ligne du milieu ; prendre le morceau centré en $(i, j)$ au lieu de celui qui commence en $(i, j)$ ; inverser ligne et colonne dans f).
**Variante** : le noyau transposé $\mathbf{K}^\top = \begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{pmatrix}$ cherche des bords **horizontaux** ; sans padding, il donne en $(0, 0)$ : $(3 + 2 + 0) - (4 + 2 + 0) =$ **−1**.

### Ex 21.3 — Tailles de sortie : padding, stride et la hiérarchie 600 → 300 → 100 ✏️
**Réponses** : a) **300** ; b) **101** ; c) **0** ; d) **1** ; e) **8** ; f) **24** ; g) **2** ; h) **300** ; i) **14** ; j) **7**.
**Démarche** : a) $\lfloor 599/2 \rfloor + 1 = 299 + 1$. b) $\lfloor 301/3 \rfloor + 1 = 100 + 1$ : avec un padding de 2, on obtient 101 et non 100 comme l'écrit le livre. c) $p = 0$ : $\lfloor 297/3 \rfloor + 1 = 100$. d) $p = 1$ : $\lfloor 299/3 \rfloor + 1 = 100$ ; $p = 2$ donne 101. e) Les filtres ont la profondeur de leur entrée, les 8 canaux du premier étage. f) Le noyau dilaté couvre $2 \times 2 + 1 = 5$ pixels : $28 - 5 + 1$. g) $(5 - 1)/2$. h) $99 \times 3 + 3$. i) $6 \times 2 - 2 + 4$. j) $\lfloor 12/2 \rfloor + 1$.
**Dans ta copie** : i) et j) se répondent : la convolution de noyau 4, stride 2, padding 1 passe de 14 à 7, et la convolution transposée de mêmes paramètres repasse de 7 à 14. Elle rend la **forme** de départ, pas les **valeurs** : c'est une multiplication par $\mathbf{C}^\top$, pas par $\mathbf{C}^{-1}$ (qui n'existe pas en général, puisque la convolution passe de $14^2$ à $7^2$ valeurs). Même la forme n'est pas toujours unique : une entrée de 15 donne aussi 7 à la convolution, d'où l'argument `output_padding` de `ConvTranspose2d` pour choisir entre 14 et 15.
**Erreurs fréquentes** : oublier le $+1$ ou appliquer la partie entière après lui ; croire le livre en b) sans refaire le calcul ; donner aux filtres du second étage la profondeur 4 (leur nombre) ; utiliser $k$ au lieu de $d(k - 1) + 1$ pour un noyau dilaté.
**Variante** : une image de 64, un noyau 4, un stride 2 et un padding 1 donnent **32** ; la convolution transposée de mêmes paramètres ramène 32 à **64**.

### Ex 21.4 — Compter les paramètres : convolution contre couche dense ✏️
**Réponses** : a) **320** ; b) **18 496** ; c) **1 179 776** ; d) **1 290** ; e) **1 199 882** ; f) **98,3** ; g) **16 981 120** ; h) **194 688**.
**Démarche** : a) $32 \times (1 \times 9 + 1)$. b) $64 \times (32 \times 9 + 1)$. c) Après deux convolutions sans padding ($28 \to 26 \to 24$) et le pooling ($24 \to 12$), on aplatit $64 \times 12 \times 12 = 9\,216$ valeurs : $9\,216 \times 128 + 128$. d) $128 \times 10 + 10$. e) La somme : pooling, dropout et aplatissement n'ont pas de paramètres. f) $1\,179\,776/1\,199\,882$. g) $784 \times 21\,632 + 21\,632$, avec $32 \times 26 \times 26 = 21\,632$ sorties. h) $26 \times 26$ positions, 32 filtres, 9 poids chacun.
**Dans ta copie** : la convolution a deux contraintes que la couche dense n'a pas. Chaque sortie ne lit que **9** pixels voisins (connexions locales) au lieu des 784, et toutes les positions **partagent** les mêmes 9 poids. La couche dense de g) est une convolution sans ces contraintes : elle pourrait apprendre un détecteur différent pour chaque position (un motif qui n'a de sens que dans un coin), ou relier des pixels éloignés dès la première couche ; mais elle a 53 000 fois plus de paramètres, il lui faudrait beaucoup plus de données pour les régler, et elle devrait apprendre séparément, pour chaque position, ce que la convolution apprend une fois. Pour des images, où un même motif peut apparaître partout, la contrainte est un avantage.
**Erreurs fréquentes** : oublier les biais ; prendre 28 × 28 au lieu de 26 × 26 positions ; aplatir $64 \times 24 \times 24$ (oublier le pooling) ou $32 \times 12 \times 12$ (le mauvais nombre de canaux) ; compter les paramètres du pooling.
**Variante** : avec des filtres 5 × 5 dans la deuxième convolution, celle-ci aurait $64 \times (32 \times 25 + 1) =$ **51 264** paramètres.

### Ex 21.5 — Champ récepteur effectif d'une pile convolution + pooling ✏️
**Réponses** : a) **5** ; b) **7** ; c) **110 592** ; d) **200 704** ; e) **11** ; f) **35** ; g) **9** ; h) **6**.
**Démarche** : on suit $r$ et le « saut » $J$ (le produit des strides déjà franchis), en partant de $r = 1$, $J = 1$ ; chaque couche fait $r \leftarrow r + (k - 1)J$, puis $J \leftarrow J s$. a) $1 + 2 + 2$. b) $1 + 2 + 2 + 2$. c) $3 \times 9 \times 64 \times 64$. d) $49 \times 64 \times 64$ : 81 % de poids en plus que les trois couches 3 × 3, pour le même champ récepteur et une seule non-linéarité au lieu de trois. e) Convolution : $r = 3$ ; pooling : $r = 3 + 2 = 5$, puis $J = 3$ ; convolution : $r = 5 + 2 \times 3 = 11$. f) Pooling : $r = 11 + 2 \times 3 = 17$, puis $J = 9$ ; convolution : $r = 17 + 2 \times 9 = 35$. g) $r = 5$, $J = 2$, puis $5 + 2 \times 2$. h) $3$, $5$, puis le pooling 2 × 2 : $5 + 1$.
**Dans ta copie** : le livre compte la région que recouvrent les 3 × 3 cases du niveau réduit, $3 \times 3 = 9$ pixels de côté, et oublie que chaque case du niveau 1 vient d'un filtre 3 × 3 qui déborde d'un pixel de chaque côté de son bloc : $9 + 2 = 11$. Le champ de f), 35, dépasse la taille de l'image (27 pixels) : la troisième convolution voit toute l'image. Ce n'est pas gênant pour un détecteur de visage entier, au contraire ; mais le champ récepteur **théorique** exagère ce qu'utilise vraiment le réseau : les pixels du centre du champ pèsent beaucoup plus que ceux du bord (le champ récepteur « effectif » est plus petit).
**Erreurs fréquentes** : multiplier le débord par le stride de la couche elle-même au lieu de celui des couches **précédentes** ; ajouter $k$ au lieu de $k - 1$ ; oublier que le pooling compte comme une couche.
**Variante** : l'entrée de ResNet, convolution 7 × 7 de stride 2 puis max pooling 3 × 3 de stride 2, donne un champ récepteur de $7 + 2 \times 2 =$ **11** pixels.

### Ex 21.6 — Convolution transposée à la main : insérer des zéros puis corréler ✏️
**Réponses** : a) **5** ; b) **1** ; c) **2** ; d) **2** ; e) **6** ; f) **3** ; g) **1** ; h) **35** ; i) **3** ; j) **1**.
**Démarche** : a) $(2 - 1) \times 2 + 3$. b) $s - 1$. c) $k - 1 - p = 2$. Les quatre tampons couvrent les carrés de coins $(0, 0)$, $(0, 2)$, $(2, 0)$ et $(2, 2)$. d) Seul le tampon de $x_{0,0}$ couvre $(0, 0)$ : $2 \times w_{0,0} = 2$. e) Seul celui de $x_{0,0}$ couvre $(1, 1)$ : $2 \times 3 = 6$. f) Deux tampons : $x_{0,0}\,w_{0,2} + x_{0,1}\,w_{0,0} = 4 - 1$. g) Les quatre : $x_{0,0}\,w_{2,2} + x_{0,1}\,w_{2,0} + x_{1,0}\,w_{0,2} + x_{1,1}\,w_{0,0} = -2 - 2 + 2 + 3$. h) Chaque tampon a pour somme $x_{i,j} \times 7$ (la somme de $\mathbf{w}$) : $5 \times 7$. La sortie complète est
$$\mathbf{y} = \begin{pmatrix} 2 & 0 & 3 & 0 & -2 \\ -2 & 6 & 1 & -3 & 0 \\ 5 & 2 & 1 & -1 & 7 \\ -1 & 3 & -3 & 9 & 0 \\ 2 & 1 & 5 & 3 & -3 \end{pmatrix}.$$
i) $5 - 2$. j) Le centre de la sortie 3 × 3 est la case $(2, 2)$ de $\mathbf{y}$.
**Dans ta copie** : on étire $\mathbf{x}$ en $\begin{pmatrix} 2 & 0 & -1 \\ 0 & 0 & 0 \\ 1 & 0 & 3 \end{pmatrix}$, on ajoute 2 zéros de chaque côté (une grille 7 × 7), et l'on corrèle avec le noyau retourné $w'_{u,v} = w_{2-u,\,2-v}$. Pour $y_{0,2}$, la fenêtre couvre les lignes 0 à 2 et les colonnes 2 à 4 de la grille ; seule sa dernière ligne touche $\mathbf{x}$ étiré, avec les valeurs $(2, 0, -1)$, face à $(w'_{2,0}, w'_{2,1}, w'_{2,2}) = (w_{0,2}, w_{0,1}, w_{0,0}) = (2, 0, 1)$ : $4 + 0 - 1 = 3$. La case $(2, 2)$ est recouverte par **4** tampons, la case $(1, 1)$ par **1** seul (et $(0, 2)$ par 2) : comme $k = 3$ n'est pas un multiple de $s = 2$, le nombre de contributions alterne d'une case à l'autre, et une image générée avec ce noyau et ce stride risque des **artefacts en damier**.
**Erreurs fréquentes** : poser le tampon de $x_{i,j}$ en $(i, j)$ au lieu de $(s\,i, s\,j)$ ; oublier un tampon dans les cases de chevauchement ; corréler avec le noyau non retourné dans la seconde méthode ; retirer le padding d'un seul côté.
**Variante** : avec un stride de 3 (égal au noyau), les tampons ne se recouvrent plus : la sortie fait **6** × 6, sa somme vaut toujours **35**, et $y_{0,2} = x_{0,0}\,w_{0,2} =$ **4**.

### Ex 21.7 — Rétropropager dans une convolution 1D : la somme sur les positions ∂
1. $m = n - k + 1$. Le poids $w_u$ intervient dans **toutes** les $m$ sorties, avec $\partial y_i/\partial w_u = x_{i+u}$ ; par la règle de la chaîne, ses contributions s'additionnent (comme pour un nœud utilisé plusieurs fois, ch. 18) : $\frac{\partial L}{\partial w_u} = \sum_{i=0}^{m-1} g_i\,x_{i+u}$. De même $\partial y_i/\partial b = 1$ donne $\sum_i g_i$. La première formule est la **corrélation croisée de $\mathbf{x}$ par $\mathbf{g}$** (le gradient de sortie joue le rôle du noyau) : elle a $n - m + 1 = k$ valeurs, une par poids.
2. $y_i$ dépend de $x_j$ si $j = i + u$ pour un $u$ entre $0$ et $k - 1$, c'est-à-dire si $0 \le j - i \le k - 1$, avec le coefficient $w_{j-i}$. Donc $\frac{\partial L}{\partial x_j} = \sum_i g_i\,w_{j-i}$ sur ces $i$ (qui doivent aussi vérifier $0 \le i \le m - 1$). Pour $x_0$, seul $i = 0$ convient : 1 terme, $g_0\,w_0$. Pour un pixel du milieu ($k - 1 \le j \le m - 1$), les $k$ valeurs de $i$ de $j - k + 1$ à $j$ conviennent : $k$ termes. Les pixels du bord reçoivent moins de gradient : ils sont lus par moins de fenêtres.
3. Notons $\tilde g_t = g_{t-(k-1)}$ (nul hors de $0 \le t - (k - 1) \le m - 1$) le gradient complété, et $w'_u = w_{k-1-u}$ le noyau retourné. Leur corrélation vaut $z_j = \sum_{u=0}^{k-1} \tilde g_{j+u}\,w'_u = \sum_u g_{j+u-k+1}\,w_{k-1-u}$ ; avec $i = j + u - k + 1$, on a $k - 1 - u = j - i$, et $z_j = \sum_i g_i\,w_{j-i}$ : c'est la formule de la question 2, les zéros du padding éliminant exactement les termes interdits. Taille : $(m + 2(k - 1)) - k + 1 = m + k - 1 = n$.
4. $\mathbf{y} = (2 - 2,\ 4 + 1,\ -2 - 3) = (0, 5, -5)$. $\partial L/\partial w_0 = 1 \times 1 + 0 \times 2 + 2 \times (-1) = -1$, $\partial L/\partial w_1 = 1 \times 2 + 0 \times (-1) + 2 \times 3 = 8$, $\partial L/\partial b = 3$. $\partial L/\partial \mathbf{x} = (g_0 w_0,\ g_0 w_1 + g_1 w_0,\ g_1 w_1 + g_2 w_0,\ g_2 w_1) = (2, -1, 4, -2)$. Différence finie sur $w_1$ avec $h = 0{,}1$ : pour $w_1 = -0{,}9$, $\mathbf{y} = (0{,}2 ;\ 4{,}9 ;\ -4{,}7)$ et $L = 0{,}2 - 9{,}4 = -9{,}2$ ; pour $w_1 = -1{,}1$, $\mathbf{y} = (-0{,}2 ;\ 5{,}1 ;\ -5{,}3)$ et $L = -10{,}8$ ; $(-9{,}2 + 10{,}8)/0{,}2 = 8$, exactement (la loss est linéaire en $w_1$).
5. Dans la vue « tampon », chaque $g_i$ dépose $g_i\,\mathbf{w}$ à partir de la position $i$ : la case $j$ reçoit $g_i\,w_{j-i}$ pour tous les $i$ tels que $0 \le j - i \le k - 1$, et la sortie a $(m - 1) + k = n$ cases. C'est la formule de la question 2 : la passe arrière d'une convolution (par rapport à son entrée) **est** une convolution transposée de même noyau. Avec un stride $s$, $y_i = b + \sum_u w_u\,x_{s i + u}$ et $\partial L/\partial x_j = \sum_i g_i\,w_{j - s i}$ : les tampons sont posés en $s\,i$, c'est une convolution transposée de stride $s$ ; de façon équivalente, on insère $s - 1$ zéros entre les valeurs de $g$ avant la convolution pleine (et l'on complète la fin par des zéros si $(n - k)$ n'est pas un multiple de $s$ : les derniers pixels, lus par aucune fenêtre, ont un gradient nul).
6. Passe avant : $m\,k$ multiplications. Poids : $k$ sommes de $m$ produits, $m\,k$. Biais : $m$ additions, aucune multiplication. Entrée : chaque $g_i$ multiplie $k$ poids, $m\,k$. La passe arrière coûte donc environ $2\,m\,k$ multiplications, deux passes avant, comme au ch. 18.
7. $\frac{\partial L}{\partial W_{o,c,u,v}} = \sum_{i,j} g_{o,i,j}\,x_{c,\,i+u,\,j+v}$, $\frac{\partial L}{\partial b_o} = \sum_{i,j} g_{o,i,j}$ et $\frac{\partial L}{\partial x_{c,p,q}} = \sum_o \sum_{u,v} g_{o,\,p-u,\,q-v}\,W_{o,c,u,v}$ (sur les indices valides) : le gradient d'une entrée additionne aussi les contributions de **tous les canaux de sortie**. Pour un mini-batch, les gradients des poids et des biais somment en plus sur les $N$ images (si la loss est une moyenne, le facteur $1/N$ est déjà dans $g$) ; le gradient de chaque image d'entrée reste séparé.
**Erreurs fréquentes** : oublier la somme sur les positions (un poids partagé a **un** gradient, somme de tous ses usages) ; écrire $w_{i-j}$ au lieu de $w_{j-i}$ ; oublier de retourner le noyau dans la convolution pleine ; oublier la somme sur les canaux de sortie en 2D ; en déduire que le gradient du bord est faux parce qu'il est plus petit.
**Variante** : avec $\mathbf{g} = (1, 1, 1)$ et les mêmes $\mathbf{x}$ et $\mathbf{w}$ (la loss $L = \sum_i y_i$), $\partial L/\partial \mathbf{w} =$ **(2, 4)** (les sommes des fenêtres $x_0..x_2$ et $x_1..x_3$) et $\partial L/\partial \mathbf{x} =$ **(2, 1, 1, −1)**.

### Ex 21.8 — Fermi : paramètres, mémoire et calcul de VGG16 🧮
**Réponses** : a) **86,7** ; b) **1,85** ; c) **15,3** ; d) **0,8** ; e) **54,2** ; f) **3,47** ; g) **Vrai**.
**Démarche** : une convolution coûte $H_{out}\,W_{out} \times C_{out} \times C_{in} \times 9$ multiplications-additions. a) $224^2 \times 64 \times 27 = 86\,704\,128$. b) $224^2 \times 64 \times 576 \approx 1{,}850 \times 10^9$. c) Par bloc : 1,94 ; 2,77 ; 4,62 ; 4,62 ; 1,39 milliard : environ $15{,}35 \times 10^9$. Chaque convolution de $C$ vers $C$ canaux des blocs 1 à 4 coûte 1,85 milliard (le côté est divisé par 2, les canaux d'entrée et de sortie multipliés par 2) ; la première de chaque bloc, qui double les canaux, coûte deux fois moins. d) Les couches denses font $25\,088 \times 4\,096 + 4\,096^2 + 4\,096 \times 1\,000 \approx 123{,}6$ millions de multiplications-additions (une par poids), soit $123{,}6/(15\,347 + 124) \approx 0{,}8\,\%$. e) Les sorties des convolutions : $2 \times 64 \times 224^2 + 2 \times 128 \times 112^2 + 3 \times 256 \times 56^2 + 3 \times 512 \times 28^2 + 3 \times 512 \times 14^2 = 13\,547\,520$ valeurs, fois 4 octets. f) $64 \times 54{,}19 \times 10^6$ octets. g) Elles ont 89 % des paramètres et moins de 1 % du calcul.
**Dans ta copie** : côté mémoire, pendant l'entraînement il faut garder aussi les sorties des ReLU et des poolings (et les positions des maximums), les gradients des activations pendant la passe arrière, et les poids avec leurs gradients et les états de l'optimiseur (Adam : environ 2,2 Go pour les 138 millions de paramètres) ; une implémentation par im2col matérialise en plus des matrices 9 fois plus grandes que l'entrée de chaque couche. Côté temps : la passe arrière coûte environ deux passes avant, le matériel n'atteint pas son débit de crête, le transfert des activations en mémoire limite souvent plus que le calcul, et il faut charger et augmenter les images. La b) montre qu'une seule convolution à pleine résolution avec 64 canaux coûte presque 2 milliards de multiplications-additions : le coût est proportionnel à l'aire de l'image. En divisant le côté par 4 dès le début (stride 2, puis pooling de stride 2), ResNet rend toutes les couches suivantes 16 fois moins chères.
**Erreurs fréquentes** : utiliser la taille d'entrée au lieu de celle de sortie ; oublier le facteur $C_{in}$ ; croire que le calcul suit les paramètres ; confondre Mo et Mio.
**Variante** : la première convolution de ResNet-18 (7 × 7, 3 → 64 canaux, stride 2, sortie 112 × 112) coûte $112^2 \times 64 \times 3 \times 49 \approx$ **118,0** millions de multiplications-additions, à peine plus que celle de VGG16, pour une sortie quatre fois plus petite.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 21.9 — Expliquer la convolution en cinq lignes 🗣️
**Réponse modèle** : « Imagine un petit tampon transparent, avec un motif dessiné dessus, par exemple un trait vertical. Tu le poses sur un coin de la photo et tu notes à quel point le dessous lui ressemble ; puis tu le décales d'un cran, et tu recommences, partout. Tu obtiens une carte qui dit où, sur la photo, se trouve ce motif. Une couche de convolution fait ça avec des dizaines de tampons, et ce sont les exemples qui lui apprennent quels motifs dessiner. Comme c'est le même tampon partout, elle reconnaît le motif où qu'il soit, avec très peu de réglages, alors qu'une couche dense devrait apprendre chaque position séparément. »
**Ce qui compte** : un même petit motif comparé à chaque endroit (les mêmes poids partout), une carte de ressemblance en sortie, des motifs appris et non dessinés à la main, l'avantage sur une couche dense (moins de réglages, un motif reconnu à n'importe quelle position) ; aucune formule. La limite de l'image : un vrai filtre ne compare pas « oui ou non », il calcule un score, et il peut lire plusieurs canaux à la fois ; et les couches suivantes combinent ces cartes en motifs plus grands.
**Erreurs fréquentes** : dire que le filtre « cherche une forme » qu'un humain a choisie ; oublier pourquoi c'est mieux qu'une couche dense ; parler de produit scalaire ou de matrice ; décrire le pooling à la place de la convolution.
**Variante** : explique en une phrase ce qu'ajoute une deuxième couche : « elle passe ses propres tampons sur les cartes de la première, pour trouver des combinaisons de motifs, des coins ou des yeux faits de traits ».

### Ex 21.10 — Deviner l'âge et le genre sur une photo : jusqu'où aller ? ⚖️
1. Le système apprend à reproduire les **annotations** du dataset : l'apparence que des annotateurs ont associée à un genre (souvent en deux catégories) et à un âge, à partir de la coiffure, du maquillage, de la pilosité, des vêtements, de l'éclairage. Il ne mesure ni l'identité de genre d'une personne ni son âge réel. Il risque de mal classer les personnes dont l'apparence s'écarte de la norme du dataset : personnes transgenres ou non binaires, personnes de groupes peu représentés dans les données (selon la couleur de peau, l'âge), personnes qui portent un voile, des lunettes, un masque. Les conséquences : des publicités qui leur assignent un genre qui n'est pas le leur (une forme de mégenrage public), des statistiques de clientèle faussées, et la répétition à grande échelle d'erreurs blessantes.
2. On constitue un jeu de test annoté avec le **genre déclaré** et l'**âge réel** des personnes (avec leur consentement), et des attributs pour former des groupes : tranche d'âge, couleur de peau (par exemple l'échelle de Fitzpatrick ou de Monk), lunettes, voile. Pour chaque groupe, on calcule la matrice de confusion du genre et l'erreur moyenne de l'âge, puis on compare les taux d'erreur entre groupes (l'étude *Gender Shades* de J. Buolamwini et T. Gebru, 2018, a mesuré ainsi des écarts de plus de 30 points entre femmes à peau foncée et hommes à peau claire, sur des systèmes commerciaux). Délicat : l'origine (que trahit la couleur de peau) et la religion (que trahit un voile) sont des données sensibles (RGPD, art. 9), dont la collecte demande un consentement explicite et des garanties ; le genre déclaré n'en fait pas partie en soi, mais reste une donnée délicate ; et des groupes petits donnent des taux d'erreur peu fiables.
3. Les clients passent devant une caméra sans l'avoir choisi. Le RGPD s'applique (une image de visage est une donnée personnelle) : il faut une base légale, une information claire **avant** l'entrée (art. 13), sans doute une analyse d'impact (art. 35), et un droit d'opposition ; or un client ne peut guère refuser d'être filmé s'il veut entrer. Le règlement sur l'IA impose aussi d'informer les personnes exposées à un système de catégorisation biométrique (art. 50, § 3). En pratique, un panneau à l'entrée ne suffit pas à rendre ce consentement libre.
4. Oui : comme au §21.10, une perturbation calculée (un motif imprimé sur un vêtement, un maquillage, un accessoire) peut faire changer la prédiction, et plus simplement un chapeau ou un masque. Une personne peut donc tromper la caméra volontairement, pour ne pas être profilée. Pour l'entreprise, c'est un problème de fiabilité : des prédictions qu'un client peut manipuler, et qui varient avec l'éclairage et l'angle, faussent les statistiques de clientèle sur lesquelles elle prend ses décisions.
5. Microsoft a retiré de son service Azure Face, en juin 2022, l'inférence de l'âge, du genre et des émotions, au nom de ses principes d'IA responsable (les clients existants l'ont gardée jusqu'en juin 2023). Le règlement européen sur l'IA (2024/1689) **interdit** la reconnaissance des émotions sur le lieu de travail et dans l'enseignement (art. 5, § 1, f)). Il interdit aussi la catégorisation biométrique qui déduit la race, les opinions politiques, l'appartenance syndicale, les convictions religieuses ou philosophiques, la vie sexuelle ou l'orientation sexuelle (art. 5, § 1, g)). Le sexe et l'âge ne sont pas dans cette liste, mais leur inférence reste une catégorisation biométrique (considérant 16) : elle est soumise à l'obligation d'information de l'article 50, et probablement classée **à haut risque** quand elle repose sur des attributs protégés (annexe III, point 1, b)). Les obligations des systèmes à haut risque de l'annexe III s'appliqueront à partir du 2 décembre 2027 (règlement omnibus 2026/1744). Recommandation raisonnable : ne pas déployer l'inférence de l'âge et du genre ; si l'objectif est de compter la fréquentation, un **comptage anonyme** (des passages, sans inférence sur les personnes ni conservation d'images, idéalement traité dans la caméra) répond au besoin avec beaucoup moins de risques ; pour adapter les publicités, s'appuyer sur des données que les clients choisissent de donner (programme de fidélité).
**Erreurs fréquentes** : croire que le système « voit » le genre ou l'âge réel ; mesurer seulement l'accuracy globale ; affirmer que l'AI Act interdit toute inférence du genre (il interdit certaines catégorisations et en encadre d'autres) ou qu'il ne dit rien ; oublier le RGPD, qui s'applique dès qu'il y a une image de visage ; confondre catégorisation (deviner des attributs) et identification (reconnaître une personne précise).
**Variante** : la même chaîne veut détecter les émotions des clients devant les rayons. Au travail ou à l'école, ce serait interdit (art. 5, § 1, f)) ; dans un magasin, c'est un système **à haut risque** (annexe III, point 1, c)), avec une obligation d'informer les personnes (art. 50, § 3), et sa base scientifique (lire une émotion sur un visage) est contestée.

### Ex 21.11 — Lire LeCun et coll. (1998) : LeNet-5 📄
1. Les **champs récepteurs locaux** (chaque unité ne lit qu'un petit voisinage : §21.2.4 et §21.3), les **poids partagés** (la même unité répliquée sur toute l'image : §21.2.3) et le **sous-échantillonnage** spatial (réduire la résolution : le pooling et le stride, §21.3.5 et §21.4.2).
2. D'abord la **taille** : une image de quelques centaines de pixels reliée à une première couche de 100 unités donne déjà des dizaines de milliers de poids, donc un besoin de données et de mémoire énorme ; et rien n'y rend le réseau **invariant** aux translations ou aux petites déformations : il devrait apprendre le même motif à chaque position. Ensuite, une couche dense **ignore la topologie** de l'entrée : on pourrait permuter les pixels sans rien changer à l'apprentissage, alors que des pixels voisins d'une image sont fortement corrélés, et que les motifs locaux (traits, coins) sont la structure qui compte.
3. Entrée 32 × 32 ; **C1** : 6 cartes 28 × 28 (convolution 5 × 5) ; **S2** : 6 cartes 14 × 14 ; **C3** : 16 cartes 10 × 10 (convolution 5 × 5) ; **S4** : 16 cartes 5 × 5 ; **C5** : 120 cartes 1 × 1 (convolution 5 × 5, qui équivaut ici à une couche dense) ; **F6** : 84 unités ; sortie : 10 unités à fonction de base radiale (RBF). Les chiffres (20 × 20 pixels, centrés dans une image 28 × 28) sont placés dans un cadre plus grand pour que les traits caractéristiques (extrémités, coins) puissent se trouver au **centre** du champ récepteur des détecteurs des couches hautes, y compris près du bord du chiffre : un padding avant l'heure.
4. $6 \times (5 \times 5 + 1) = 156$ paramètres, et des cartes de $32 - 5 + 1 = 28$ pixels de côté (avec $156 \times 28 \times 28 = 122\,304$ connexions, nombre que donne l'article).
5. Chaque unité additionne ses 4 entrées d'un bloc 2 × 2 (sans recouvrement), multiplie la somme par un **coefficient appris**, ajoute un **biais appris**, et passe le résultat dans une sigmoïde : un pooling moyen réglable. Deux paramètres par carte : 12 pour S2, 32 pour S4.
6. Les auteurs donnent deux raisons : garder un nombre de connexions raisonnable, et surtout **briser la symétrie** : des cartes qui reçoivent des combinaisons différentes de cartes d'entrée sont forcées d'extraire des features différentes (et complémentaires). C3 a ainsi 1 516 paramètres, au lieu de $16 \times (6 \times 25 + 1) = 2\,416$ si elle était complète. Aujourd'hui, `groups=` de `nn.Conv2d` découpe les canaux en groupes indépendants, de façon régulière (chaque groupe de sortie lit un groupe d'entrée) ; la convolution par canal en est le cas extrême.
7. **0,95 %** d'erreur sur le test de MNIST sans déformation, **0,8 %** avec des images d'entraînement déformées artificiellement (une augmentation de données). Le convnet du livre atteint un peu plus de 99 % d'accuracy, environ 1 % d'erreur : le même ordre de grandeur, avec un réseau plus gros et des outils plus récents, mais sans augmentation.
8. Ont disparu : les sorties RBF fixées à la main (remplacées par une couche dense et une softmax), les activations tanh et sigmoïde (remplacées par ReLU), le sous-échantillonnage moyen à coefficient appris (remplacé par le max pooling, puis par des convolutions à stride), la table de connexions irrégulière (remplacée par des convolutions complètes ou à groupes réguliers), et souvent les couches denses de la fin (remplacées par un pooling moyen global). Sont restés : l'alternance convolution, réduction de la résolution, la hiérarchie de features, les poids partagés et les champs récepteurs locaux, et l'apprentissage de bout en bout par descente de gradient.
**Erreurs fréquentes** : dire que S2 et S4 font un max pooling ; compter les cartes de C3 comme complètement connectées ; oublier que C5 est une convolution (qui ne vaut une couche dense que pour des entrées de 32 × 32) ; comparer des taux d'erreur obtenus avec et sans augmentation sans le dire.
**Variante** : compte les paramètres de C5 : chaque carte lit les 16 cartes de S4 avec des noyaux 5 × 5, $120 \times (16 \times 25 + 1) =$ **48 120**.

<a id="entretien"></a>

## 💼 Entretien

### 21.E1 — Pourquoi un CNN plutôt qu'un MLP pour des images ?
**Réponse modèle en 60 secondes** : « Pour deux propriétés des images. D'abord, l'information est **locale** : un bord, un coin, une texture se reconnaissent sur quelques pixels voisins ; une convolution ne relie chaque sortie qu'à un petit voisinage, au lieu de tous les pixels. Ensuite, un même motif peut apparaître **n'importe où** : la convolution utilise les mêmes poids à toutes les positions, et le détecteur appris une fois sert partout. Résultat : beaucoup moins de paramètres, une couche 3 × 3 de 64 vers 64 canaux a 37 000 poids quelle que soit la taille de l'image, alors qu'une couche dense de 1 000 neurones seulement, sur une image de 224 × 224 en couleur, en a déjà 150 millions ; donc moins de données nécessaires et moins d'overfitting. En empilant convolutions et réductions de résolution, on construit une hiérarchie : bords, puis textures, puis parties d'objets. Un MLP sur les pixels doit apprendre chaque position séparément, et ignore que des pixels sont voisins. »
**Ce qui compte** : connexions locales et partage de poids, l'ordre de grandeur des paramètres, l'équivariance aux translations, la hiérarchie de features.
**Relances possibles** : « Un CNN est-il invariant aux translations ? » (la convolution est **équivariante** : décaler l'image décale la carte ; le pooling et le pooling moyen global apportent une invariance partielle, mais le stride et le padding la cassent un peu, et un CNN n'est invariant ni aux rotations ni aux changements d'échelle sans augmentation de données) · « Et les Vision Transformers, qui n'ont pas ces a priori ? » (ils les apprennent à partir des données : moins efficaces sur de petits datasets, compétitifs ou meilleurs avec un grand pré-entraînement) · « Quand un MLP suffit-il ? » (pour de petites images bien centrées, comme MNIST, il atteint déjà 98 % ; pour des données tabulaires, sans structure spatiale, c'est la convolution qui n'a plus de sens).

### 21.E2 — Taille de sortie et paramètres d'une Conv2d(3, 64, 3, stride=2, padding=1) sur une image 224×224
**Réponse modèle en 60 secondes** : « La taille de sortie se calcule par direction : partie entière de (224 + 2 × 1 − 3)/2, plus 1, soit 111 + 1 = 112. La sortie a donc la forme (32, 64, 112, 112) : 32 images, un canal par filtre, et le côté divisé par deux. Les paramètres : chaque filtre a 3 canaux de 3 × 3 poids, 27 poids, plus un biais ; 64 filtres font 64 × 28 = 1 792 paramètres. Le poids a la forme (64, 3, 3, 3) dans PyTorch. Le nombre de paramètres ne dépend ni de la taille de l'image ni du mini-batch ; le calcul, lui, si : environ 112 × 112 × 64 × 27, soit 21,7 millions de multiplications-additions par image. »
**Ce qui compte** : la formule et sa partie entière, la forme NCHW, le décompte $C_{out}(C_{in}k^2 + 1)$, la distinction entre paramètres et calcul.
**Relances possibles** : « Et avec `padding="same"` ? » (PyTorch le refuse avec un stride différent de 1 ; avec un stride de 1, la sortie garderait 224) · « Et avec `bias=False` ? » (1 728 paramètres : on retire le biais quand une batchnorm suit, puisqu'elle a son propre $\beta$) · « Et une entrée de 225 ? » ($\lfloor 224/2 \rfloor + 1 = 113$ : la partie entière fait qu'une entrée impaire ne se divise pas exactement).

### 21.E3 — À quoi sert une convolution 1×1 ?
**Réponse modèle en 60 secondes** : « Elle ne regarde qu'un pixel, mais **tous ses canaux** : c'est une couche dense appliquée à chaque pixel, avec les mêmes poids partout. Elle apprend des combinaisons de features : si 256 canaux disent "bord vertical", "rouge", "texture fine", elle peut en faire 64 nouvelles features qui les combinent. On s'en sert surtout pour changer le nombre de canaux à faible coût : réduire 256 canaux à 64 avant une convolution 3 × 3 coûteuse, puis remonter à 256, c'est le bloc "bottleneck" de ResNet-50 ; ou ajuster le nombre de canaux d'une connexion résiduelle. Elle ajoute aussi une non-linéarité si on la fait suivre d'une ReLU. Dans MobileNet, une convolution par canal, qui regarde l'espace, est suivie d'une convolution 1 × 1, qui mélange les canaux : les deux ensemble coûtent bien moins qu'une convolution standard. Et en segmentation, une 1 × 1 finale donne un score par classe en chaque pixel. »
**Ce qui compte** : un mélange des canaux par pixel, la réduction ou l'augmentation du nombre de canaux, le coût ($C_{in} \times C_{out}$ poids), au moins un usage concret (bottleneck, séparable, tête de segmentation).
**Relances possibles** : « Combien coûte-t-elle comparée à une 3 × 3 ? » (neuf fois moins de poids et de calcul pour les mêmes canaux) · « D'où vient l'idée ? » (« Network in Network », M. Lin et coll., 2014, puis les modules Inception de GoogLeNet) · « Une 1 × 1 est-elle exactement une couche dense ? » (oui, appliquée pixel par pixel : `nn.Conv2d(C_in, C_out, 1)` et `nn.Linear(C_in, C_out)` sur le tenseur permuté en `(N, H, W, C)` donnent le même résultat avec les mêmes poids).

### 21.E4 — Exemples adversariaux : quels risques, quelles défenses ?
**Réponse modèle en 60 secondes** : « Un exemple adversarial est une entrée modifiée de façon presque invisible, mais calculée pour que le modèle se trompe. L'attaque la plus simple, FGSM, fait un pas dans le sens du signe du gradient de la loss par rapport à l'image : chaque pixel bouge d'un ou deux pour cent, et un très bon classifieur change d'avis. On évalue le risque en mesurant l'accuracy sous attaque, avec PGD, une version itérée de FGSM, pour plusieurs budgets ε, et en se demandant qui pourrait attaquer : un fraudeur face à un contrôle d'identité, quelqu'un qui colle un autocollant sur un panneau. Des attaques marchent même sans accès au modèle, par transfert depuis un autre réseau, et dans le monde physique. La défense la plus solide est l'entraînement adversarial, entraîner sur des exemples attaqués, qui coûte cher et fait baisser un peu l'accuracy propre ; on évalue avec des attaques fortes et variées, car beaucoup de défenses publiées masquaient seulement le gradient. Et l'on ne déploie pas un modèle seul là où une erreur provoquée a des conséquences graves : contrôle humain, plusieurs capteurs. »
**Ce qui compte** : la définition, FGSM ou PGD en une phrase, la mesure de robustesse (accuracy en fonction de ε), le modèle de menace, l'entraînement adversarial et son coût, la prudence sur les défenses qui masquent le gradient.
**Relances possibles** : « Pourquoi les réseaux sont-ils si fragiles ? » (une explication de Goodfellow et coll. : en grande dimension, même un modèle linéaire l'est, car de petites variations de nombreux pixels s'additionnent ; les réseaux de ReLU sont largement linéaires par morceaux) · « Une perturbation trompe-t-elle d'autres modèles ? » (souvent : c'est la transférabilité, qui permet des attaques en boîte noire) · « Comment comparer des défenses ? » (sur un classement comme RobustBench, avec une suite d'attaques standard comme AutoAttack, et non avec l'attaque choisie par les auteurs de la défense).

### 21.E5 — CNN ou Vision Transformer pour un nouveau projet de vision ?
**Réponse modèle en 60 secondes** : « Avec 5 000 images, je n'entraîne ni l'un ni l'autre de zéro : je pars d'un modèle pré-entraîné et je fais un fine-tuning. Mon premier essai serait un CNN pré-entraîné, un ResNet-50 ou un ConvNeXt, ou EfficientNet si l'appareil est modeste : c'est rapide, robuste avec peu de données grâce à ses a priori de localité, facile à déployer, et ça donne une référence en une journée. Ensuite j'essaierais un ViT pré-entraîné sur de très grandes données, par exemple un modèle auto-supervisé comme DINOv2, dont les features se transfèrent très bien ; à pré-entraînement égal, les deux familles sont proches. Je trancherais sur un jeu de validation représentatif, avec la même augmentation de données et les mêmes réglages, en regardant aussi la latence, la mémoire et le coût sur la cible de déploiement. Dans des images industrielles, des défauts minuscules ou des images très différentes d'ImageNet peuvent changer la donne : la résolution et la qualité des annotations comptent souvent plus que l'architecture. »
**Ce qui compte** : pas d'entraînement de zéro avec 5 000 images, le transfert, un CNN comme référence rapide et ses a priori, le ViT pré-entraîné comme candidat sérieux, une décision mesurée sur la validation et les contraintes de déploiement.
**Relances possibles** : « Pourquoi un ViT a-t-il besoin de plus de données ? » (il n'a pas l'a priori de localité ni le partage de poids des convolutions : il doit apprendre ces régularités des données ; l'article d'origine le trouvait en dessous des ResNet avec ImageNet seul, et au-dessus avec un pré-entraînement bien plus grand) · « Comment faire le fine-tuning avec si peu d'images ? » (geler d'abord le réseau et n'entraîner que la tête, puis dégeler les dernières couches avec un petit learning rate ; augmentation de données ; validation croisée ou un jeu de validation stratifié) · « Et si les classes sont déséquilibrées ? » (des poids dans la loss ou un échantillonnage équilibré, et des métriques par classe, recall et matrice de confusion, plutôt que l'accuracy, ch. 3).

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb` ; voici, pour chaque exercice, la réponse, la démarche, les erreurs fréquentes et une variante. Les chiffres sont ceux du mode rapide, sur CPU. Les fonctions de ta librairie sont données en entier ; elles partagent deux petits outils :

```python
import numbers

import numpy as np


def _as_pair(value, name, minimum):
    """An int or a pair of ints -> a pair of ints >= minimum (a bool is not an int)."""
    pair = (value, value) if isinstance(value, numbers.Integral) and not isinstance(value, bool) else value
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or not all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in pair)):
        raise ValueError(f"{name} must be an int or a tuple of 2 ints, got {value!r}")
    if min(pair) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(pair[0]), int(pair[1])


def _batch(x):
    arr = np.array(x, dtype=np.float64)              # a new float64 array: x is never modified
    if arr.ndim != 4:
        raise ValueError(f"x must have 4 axes (N, C, H, W), got shape {arr.shape}")
    return arr
```

### Ex 21.12 — conv_output_size et pad2d 🔨
**Réponses** : les six tailles de PyTorch, les quatre refus et les trois paddings passent ; les 41 tests de l'exercice passent.
**Démarche** :
```python
def conv_output_size(size, kernel_size, stride=1, padding=0, dilation=1):
    for name, value, minimum in [("size", size, 1), ("kernel_size", kernel_size, 1), ("stride", stride, 1),
                                 ("padding", padding, 0), ("dilation", dilation, 1)]:
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise ValueError(f"{name} must be an int, got {value!r}")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
    span = dilation * (kernel_size - 1) + 1          # the cells covered by a dilated kernel
    if span > size + 2 * padding:
        raise ValueError(f"the window ({span} cells) does not fit in {size} + 2 * {padding} cells")
    return int((size + 2 * padding - span) // stride + 1)


def pad2d(x, padding, value=0.0):
    arr = _batch(x)
    ph, pw = _as_pair(padding, "padding", 0)
    n, c, h, w = arr.shape
    out = np.full((n, c, h + 2 * ph, w + 2 * pw), value, dtype=np.float64)
    out[:, :, ph:ph + h, pw:pw + w] = arr
    return out
```
La formule compte les positions de départ d'une fenêtre de $d(k - 1) + 1$ cases dans $H + 2p$ cases, une toutes les $s$ cases. `_as_pair` sert à toutes les fonctions du module : un entier vaut pour les deux axes.
**Erreurs fréquentes** : `isinstance(True, int)` est vrai en Python (un booléen passerait pour 1) ; `np.floor(...)` sur un quotient réel au lieu de `//` (un `numpy.float64` si l'on oublie `int`) ; rendre un `numpy.int64` quand les arguments viennent d'un tableau ; oublier la dilatation dans l'étendue du noyau ; `np.pad` sur les quatre axes avec un seul nombre (il complète aussi les exemples et les canaux) ; rendre `x` lui-même quand le padding vaut 0.
**Variante** : la première couche d'un ResNet (un noyau 7 × 7, un stride de 2, un padding de 3) sur une image de 224 pixels : $\lfloor (224 + 6 - 7)/2 \rfloor + 1 =$ **112** ; `pad2d(np.ones((1, 1, 2, 2)), (0, 1))` a la forme **(1, 1, 2, 4)**.

### Ex 21.13 — Prédire où une image CIFAR-10 « jaunit » 🔮
**Réponses** : `weight_13 = np.array([1.0, 1.0, -1.0]).reshape(1, 3, 1, 1)` ; a) **C** (le chevreuil dans l'herbe) ; b) **True** ; c) **D** (la voiture).
**Démarche** : une convolution 1 × 1 à un filtre, c'est le neurone de ✏️ 21.1 appliqué à chaque pixel ; son poids a la forme `(C_out, C_in, 1, 1)`. a) L'herbe et le pelage ont beaucoup de rouge et de vert pour peu de bleu : la moyenne de C est la plus haute (1,15, contre 1,09 pour D, 0,91 pour A et −0,27 pour B). Le ciel blanc de A vaut à peu près $1 + 1 - 1 = 1$, sans guère dépasser cette valeur (1,02 au plus), et la voiture n'occupe qu'une partie de D. b) Un ciel bleu a $B$ grand, $R$ et $G$ plus petits : un score négatif. c) Le maximum possible vaut 2, pour $R = G = 1$ et $B = 0$, un jaune pur : seuls des pixels de la carrosserie de D s'en approchent (1,73, contre 1,50 au mieux pour C).
**Dans tes notes** : ce neurone ne détecte pas le jaune mais « beaucoup de rouge et de vert, peu de bleu » ; il répond au blanc et à l'herbe. Pour exiger aussi que $R$ et $G$ soient proches, il faut une non-linéarité et une seconde couche (✏️ 21.1).
**Erreurs fréquentes** : le poids rangé `(1, 1, 1, 3)` ou `(3, 1, 1, 1)` ; donner la photo dont la couleur **paraît** la plus jaune (la voiture) sans calculer le score de l'herbe ; oublier qu'un pixel blanc vaut 1, pas 2.
**Variante** : avec le poids $(1, 0, -1)$, « rouge moins bleu », un pixel blanc et un pixel vert pur valent tous deux **0**, un rouge pur **1** et un bleu pur **−1** : ce neurone sépare les couleurs chaudes des froides et ignore le vert.

### Ex 21.14 — im2col : la vue « œil de mouche » 🔨
**Réponses** : les trois cas contre `F.unfold`, le produit matriciel contre `F.conv2d` et le refus passent ; les 24 tests de l'exercice passent.
**Démarche** :
```python
def _slices(xp, kernel, stride, dilation, out):
    """For each kernel offset (i, j), the strided view of the padded input seen by that offset."""
    (kh, kw), (sh, sw), (dh, dw), (ho, wo) = kernel, stride, dilation, out
    return [[xp[:, :, i * dh:i * dh + sh * (ho - 1) + 1:sh, j * dw:j * dw + sw * (wo - 1) + 1:sw]
             for j in range(kw)] for i in range(kh)]


def im2col(x, kernel_size, stride=1, padding=0, dilation=1):
    arr = _batch(x)
    kernel = _as_pair(kernel_size, "kernel_size", 1)
    stride = _as_pair(stride, "stride", 1)
    padding = _as_pair(padding, "padding", 0)
    dilation = _as_pair(dilation, "dilation", 1)
    n, c, h, w = arr.shape
    out = tuple(conv_output_size(size, k, s, p, d) for size, k, s, p, d in zip((h, w), kernel, stride, padding, dilation))
    xp = pad2d(arr, padding)
    views = _slices(xp, kernel, stride, dilation, out)
    patches = np.stack([np.stack(row, axis=2) for row in views], axis=2)   # (N, C, kh, kw, Ho, Wo)
    return patches.reshape(n, c * kernel[0] * kernel[1], out[0] * out[1])
```
On ne boucle que sur les $k_h \times k_w$ décalages du noyau (9 pour un 3 × 3), pas sur les positions : chaque décalage lit une tranche à pas fixe de l'image complétée. Empilées dans l'ordre `(N, C, k_h, k_w, H_out, W_out)`, ces tranches donnent par un simple `reshape` l'indice `c·k_h·k_w + i·k_w + j` dans la colonne et `l = ligne·W_out + colonne` pour la position, l'ordre de `F.unfold` et de `weight.reshape(C_out, -1)`.
**Erreurs fréquentes** : empiler les décalages sur le dernier axe (`(N, C, H_out, W_out, k_h, k_w)`, puis `reshape` : les valeurs sont mélangées sans erreur de forme) ; une tranche sans borne de fin, `xp[:, :, i::s]` (quand le stride ne tombe pas juste, elle a une valeur de trop) ; oublier de multiplier le décalage par la dilatation ; une double boucle sur les positions (juste, mais des centaines de fois plus lente) ; rendre des vues de `x` (le contrat veut un nouveau tableau, que `reshape` après `np.stack` garantit).
**Variante** : `im2col` d'un mini-batch `(2, 3, 32, 32)` avec un noyau 5 × 5 et un padding de 2 a la forme **(2, 75, 1 024)** : chaque pixel est recopié dans 25 colonnes, si bien que la mémoire est multipliée par **25** ($k_h k_w$). C'est le prix de cette vue, et la raison pour laquelle les bibliothèques ne la construisent pas toujours.

### Ex 21.15 — conv2d : stride, padding, dilatation et groupes 🔨
**Réponses** : les six cas contre `F.conv2d` et les quatre refus passent ; les 47 tests de l'exercice passent.
**Démarche** :
```python
def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    arr = _batch(x)
    w = np.array(weight, dtype=np.float64)
    if w.ndim != 4:
        raise ValueError(f"weight must have 4 axes (C_out, C_in / groups, kh, kw), got shape {w.shape}")
    if not isinstance(groups, numbers.Integral) or isinstance(groups, bool) or groups < 1:
        raise ValueError(f"groups must be an int >= 1, got {groups!r}")
    n, c_in, h, wd = arr.shape
    c_out, c_group, kh, kw = w.shape
    if c_in % groups or c_out % groups:
        raise ValueError(f"groups={groups} must divide C_in={c_in} and C_out={c_out}")
    if c_group != c_in // groups:
        raise ValueError(f"weight expects {c_group} input channels per group, x gives {c_in // groups}")
    if bias is not None:
        bias = np.array(bias, dtype=np.float64)
        if bias.shape != (c_out,):
            raise ValueError(f"bias must have shape ({c_out},), got {bias.shape}")
    stride = _as_pair(stride, "stride", 1)
    dilation = _as_pair(dilation, "dilation", 1)
    if isinstance(padding, str):
        if padding == "valid":
            padding = (0, 0)
        elif padding == "same":
            if stride != (1, 1):
                raise ValueError(f'padding="same" needs a stride of 1, got {stride}')
            totals = (dilation[0] * (kh - 1), dilation[1] * (kw - 1))
            arr = np.pad(arr, ((0, 0), (0, 0), (totals[0] // 2, totals[0] - totals[0] // 2),
                               (totals[1] // 2, totals[1] - totals[1] // 2)))   # the extra zero at the bottom and right
            padding = (0, 0)
        else:
            raise ValueError(f'padding must be an int, a pair, "same" or "valid", got {padding!r}')
    padding = _as_pair(padding, "padding", 0)
    ho = conv_output_size(arr.shape[2], kh, stride[0], padding[0], dilation[0])
    wo = conv_output_size(arr.shape[3], kw, stride[1], padding[1], dilation[1])
    g_in, g_out = c_in // groups, c_out // groups
    parts = []
    for g in range(groups):                          # one matrix product per group
        cols = im2col(arr[:, g * g_in:(g + 1) * g_in], (kh, kw), stride, padding, dilation)    # (N, g_in kh kw, L)
        parts.append(w[g * g_out:(g + 1) * g_out].reshape(g_out, -1) @ cols)                  # (N, g_out, L)
    out = np.concatenate(parts, axis=1).reshape(n, c_out, ho, wo)
    if bias is not None:
        out += bias[None, :, None, None]
    return out
```
Toute la convolution tient dans le produit `poids aplatis @ colonnes` de 21.14, une fois par groupe ; `@` diffuse sur l'axe des images. `"same"` est le seul cas où la bordure peut être inégale : on complète soi-même avec `np.pad`, puis on continue sans padding.
**Erreurs fréquentes** : retourner le noyau (c'est la convolution des mathématiciens ; PyTorch, comme le livre, calcule une corrélation croisée) ; découper les groupes de sortie par `w[g::groups]` (des filtres entrelacés) au lieu de tranches consécutives ; mettre la ligne en trop de `"same"` en haut ; vérifier `bias.size == C_out` (un biais `(1, 2)` passerait) ; `out += bias` avec un biais de forme `(C_out,)` (diffusé sur le dernier axe, la largeur : une erreur de forme, ou un résultat faux quand $W_{out} = C_{out}$) ; calculer `ho` avec la taille d'avant le padding de `"same"`.
**Variante** : une convolution **séparable par canal** (`groups=32`, un filtre 3 × 3 par canal, puis une convolution 1 × 1 de 32 vers 64 canaux) remplace une convolution 3 × 3 ordinaire de 32 vers 64 canaux : $32 \times 9 + 32 + 32 \times 64 + 64 =$ **2 432** paramètres au lieu de $64 \times 32 \times 9 + 64 =$ **18 496**, presque 8 fois moins. C'est la brique de MobileNet.

### Ex 21.16 — Tests pytest paramétrés pour conv2d 🛠️
**Réponses** : les tests passent sur `conv_ok` (19 cas) et chacune des cinq versions fausses en fait échouer au moins un.
**Démarche** :
```python
import pytest


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("size", [(7, 7), (8, 5)])
def test_conv_matches_torch(stride, padding, size):
    rng = np.random.default_rng(stride * 100 + padding * 10 + size[1])
    x = rng.normal(size=(2, 3, *size))
    w = rng.normal(size=(4, 3, 3, 2))                      # an asymmetric, rectangular kernel
    b = rng.normal(size=4)
    want = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=stride, padding=padding).numpy()
    got = np.asarray(conv(x, w, b, stride=stride, padding=padding))
    assert got.shape == want.shape
    assert np.allclose(got, want)


def test_conv_without_bias_is_the_dot_product_of_one_window():
    x = np.arange(9.0).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    assert np.allclose(conv(x, w), [[[[0 - 4, 1 - 5], [3 - 7, 4 - 8]]]])


my_tests_16 = [test_conv_matches_torch, test_conv_without_bias_is_the_dot_product_of_one_window]
```
Le test paramétré essaie 18 combinaisons de stride, de padding et de taille ; avec un noyau rectangulaire **asymétrique** et un biais non nul, chaque réglage a un cas où il compte. Le second test, sans biais et calculé à la main, vérifie le cas `bias=None`.
**Dans tes notes** : `conv_bug_5` ne se trompe que quand le stride ne divise pas $H + 2p - k$ : une image 6 × 6 avec un noyau 2 et un stride de 2 le laisse passer. `conv_bug_3` passe tout test de stride 1, `conv_bug_1` tout test à noyau symétrique, `conv_bug_2` tout test sans biais, `conv_bug_4` tout test sans padding.
**Erreurs fréquentes** : un noyau symétrique (`np.ones`) : le retournement ne change rien ; un seul stride ou une seule taille ; comparer les valeurs avant les formes (`np.allclose` diffuse parfois deux formes différentes) ; mettre les tests eux-mêmes dans la liste avec des parenthèses (`test_conv()`) ; un test qui dépend d'un tirage sans graine ; tester `conv_ok` contre lui-même au lieu de `F.conv2d`.
**Variante** : un sixième bug, « le biais est ajouté deux fois », est attrapé par les **18** cas du test paramétré (son biais est non nul) et par **aucun** cas du test sans biais : chaque réglage a besoin d'au moins un test où il compte. (Le padding ajouté seulement en haut et à gauche, `conv_bug_4`, laisse même passer quelques cas avec padding : avec 5 colonnes, un padding de 1 et un stride de 2 ou 3, les fenêtres et leur nombre ne changent pas ; d'où plusieurs tailles d'image.)

### Ex 21.17 — Reproduire la chasse aux rayures verticales (Fig. 21.14) 🎨
**Réponses** : a) **3** ; b) **1** ; c) **18**.
**Démarche** : `scores_17 = mylearn.conv.conv2d(IMG_17[None, None], K_17[None, None])[0, 0]`, de forme `(22, 22)`. Le score parfait vaut la somme des $+1$ du filtre, **3** (les trois $+1$ sur du blanc, les six $-1$ sur du noir, qui comptent 0) ; le plus bas, **−6** (les six $-1$ sur du blanc, les trois $+1$ sur du noir). a) `np.sum(scores_17 == 3)`. b) `np.sum(scores_17 == -6)`. c) `np.sum(scores_17 == 2)`. La figure (voir le notebook de solutions) : cinq `imshow`, une barre de couleurs sur le panneau des scores, et pour chaque correspondance $(i, j)$ un `plt.Rectangle((j - 0.5, i - 0.5), 3, 3, fill=False)` : la sortie $(i, j)$ lit les pixels $i$ à $i + 2$ et $j$ à $j + 2$.
**Dans tes notes** : les deux correspondances voisines viennent d'une rayure de **quatre** pixels : le filtre ne contraint que trois lignes, et une rayure plus longue répond à deux positions successives. Une rayure parfaite dont un pixel du milieu est noir perd 1 : 2. Un score de 2 signale donc un motif presque complet.
**Erreurs fréquentes** : croire que le score parfait vaut 9 (les $-1$ sur du noir valent $-1 \times 0 = 0$, pas $+1$) : avec des pixels 0 et 1, un $-1$ ne peut que retirer ; compter les scores `>= 2` pour a) ; placer le cadre au **centre** de la fenêtre (décalé d'un pixel) ; oublier `[None, None]`, ou le `[0, 0]` final (un tableau `(1, 1, 22, 22)`).
**Variante** : avec des pixels $-1$ (noir) et $+1$ (blanc) au lieu de 0 et 1, le score parfait vaudrait **9** et le plus bas **−9** : chaque case compte, et le filtre devient un vrai produit scalaire entre deux motifs de $\pm 1$.

### Ex 21.18 — max_pool2d et avg_pool2d 🔨
**Réponses** : les quatre réglages contre `F.max_pool2d` et `F.avg_pool2d`, les valeurs négatives avec padding et le refus passent ; les 47 tests de l'exercice passent.
**Démarche** :
```python
def _pool_windows(x, kernel_size, stride, padding, value):
    arr = _batch(x)
    kernel = _as_pair(kernel_size, "kernel_size", 1)
    stride = kernel if stride is None else _as_pair(stride, "stride", 1)
    padding = _as_pair(padding, "padding", 0)
    if padding[0] > kernel[0] // 2 or padding[1] > kernel[1] // 2:
        raise ValueError(f"padding {padding} must be at most half the kernel {kernel}")
    out = tuple(conv_output_size(size, k, s, p) for size, k, s, p in zip(arr.shape[2:], kernel, stride, padding))
    xp = pad2d(arr, padding, value=value)
    return [view for row in _slices(xp, kernel, stride, (1, 1), out) for view in row], kernel


def max_pool2d(x, kernel_size, stride=None, padding=0):
    views, _ = _pool_windows(x, kernel_size, stride, padding, -np.inf)    # padding never wins the maximum
    return np.max(np.stack(views), axis=0)


def avg_pool2d(x, kernel_size, stride=None, padding=0):
    views, kernel = _pool_windows(x, kernel_size, stride, padding, 0.0)   # padding counted (count_include_pad)
    return np.sum(np.stack(views), axis=0) / (kernel[0] * kernel[1])
```
Les fenêtres d'un pooling sont celles d'une convolution sans dilatation : `_slices` de 21.14 les donne, une tranche par décalage, et le maximum (ou la moyenne) de ces tranches est le résultat.
**Erreurs fréquentes** : compléter par des zéros pour le max (sur des valeurs toutes négatives, le padding gagne) ; diviser la moyenne par le nombre de vraies cases de chaque fenêtre (c'est `count_include_pad=False`, pas le défaut) ; oublier `stride = kernel_size` par défaut (des fenêtres qui se recouvrent) ; réduire aussi sur l'axe des canaux.
**Variante** : `max_pool2d` d'une image 7 × 7 avec un noyau 2 (stride 2) a la taille **3 × 3** : la dernière ligne et la dernière colonne ne sont lues par aucune fenêtre. Avec `padding=1`, la taille devient **4 × 4**, et ces cases sont lues.

### Ex 21.19 — Flou, Sobel, netteté : des filtres faits main avec F.conv2d 📦
**Réponses** : les formes, les images uniformes, le bord vertical, la comparaison du flou à `F.avg_pool2d` et du Sobel aux deux étapes passent ; la figure montre la photo, son flou, $|$Sobel$|$ et sa version plus nette.
**Démarche** :
```python
def blur_19(x):
    weight = torch.full((3, 1, 3, 3), 1.0 / 9, dtype=x.dtype)            # (C_out, C_in / groups, 3, 3) = (3, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight, groups=3)


def sobel_19(x):
    sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=x.dtype)
    weight = torch.stack([c * sobel for c in LUMA_19])[None]             # (1, 3, 3, 3): grey THEN Sobel, in one layer
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight)


def sharpen_19(x):
    kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]], dtype=x.dtype)
    weight = kernel.expand(3, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight, groups=3).clamp(0.0, 1.0)
```
Avec `groups=3`, chaque filtre ne lit qu'un canal : le poids a la forme `(3, 1, 3, 3)`. Le Sobel de la luminance lit les trois canaux à la fois, avec un poids `(1, 3, 3, 3)` dont le canal $c$ vaut $a_c \times$ Sobel.
**Dans tes notes** : la luminance $\sum_c a_c x_c$ est linéaire, la convolution aussi : appliquer le Sobel à la luminance revient à appliquer $a_c \times$ Sobel à chaque canal et à sommer, ce que fait une convolution à trois canaux d'entrée. Avec des zéros autour, les fenêtres du bord mélangent l'image et du noir : un cadre sombre. Les sommes des poids : 1 pour le flou et la netteté (une image uniforme reste la même), 0 pour Sobel (elle donne 0).
**Erreurs fréquentes** : un poids `(3, 3, 3, 3)` sans `groups` (chaque sortie mélange les trois canaux : un flou gris) ; `F.pad(x, (1, 1))`, qui ne complète que la largeur ; des poids en `float32` pour une image `float64` (`RuntimeError: expected scalar type Double`) ; le Sobel retourné, de signe opposé (`[1, 0, -1]` : négatif là où l'image s'éclaire vers la droite) ; `torch.clamp_` sur `x` lui-même, qui modifie l'entrée.
**Variante** : le Sobel **vertical** (le noyau transposé, positif là où l'image s'éclaire vers le bas) et le module $\sqrt{G_x^2 + G_y^2}$ donnent tous les bords, quelle que soit leur direction ; sur l'image `EDGE_19`, dont le bord est vertical, le Sobel vertical vaut **0** partout.

### Ex 21.20 — Conv1d, noyau pleine largeur et 1×1 : prédire formes et paramètres 🔮
**Réponses** : a) **« (8, 16, 26) »** ; b) **5 136** ; c) **« (8, 16, 26, 1) »** ; d) **True** ; e) **1 040** ; f) **« (8, 16, 30, 30) »**.
**Démarche** : a) Une `Conv1d` glisse le long du dernier axe : $30 - 5 + 1 = 26$ positions, et les 64 canaux deviennent 16 filtres. b) Chaque filtre lit 64 canaux sur 5 positions, plus un biais : $16 \times (64 \times 5 + 1) = 5\,136$. c) Le noyau couvre les 64 colonnes : une seule position en largeur, l'axe reste, de taille 1. d) $16 \times (1 \times 5 \times 64 + 1) = 5\,136$ : les mêmes poids, rangés autrement. e) $16 \times (64 + 1) = 1\,040$. f) Un noyau 1 × 1 garde la taille de l'image.
**Dans tes notes** : la `Conv1d` et la `Conv2d` pleine largeur calculent la même chose : avec les poids copiés ($w_2[o, 0, u, c] = w_1[o, c, u]$), leurs sorties ne diffèrent que d'arrondis ($1{,}5 \times 10^{-6}$, en float32) ; seule la place des 64 nombres change (des canaux pour l'une, des colonnes pour l'autre). Une convolution 1 × 1 applique en chaque pixel la même couche dense `nn.Linear(64, 16)` aux 64 canaux de ce pixel.
**Erreurs fréquentes** : $30 - 5 = 25$ (une position oubliée) ; laisser 64 canaux en sortie ; croire qu'une `Conv1d` glisse sur les canaux ; oublier les biais ; pour f), appliquer $30 - 1$.
**Variante** : `nn.Conv1d(64, 16, kernel_size=5, padding=2, stride=2)(x1_20)` a la forme **(8, 16, 15)** ($\lfloor (30 + 4 - 5)/2 \rfloor + 1 = 15$) et toujours **5 136** paramètres : le stride et le padding ne changent pas les poids.

### Ex 21.21 — Le détecteur de visage hiérarchique (Fig. 21.18 à 21.23) 🎨
**Réponses** : a) **35** ; b) **4** ; c) **A** (un visage de face) ; d) **C** (aucun visage).
**Démarche** :
```python
def detect_21(image):
    C = mylearn.conv
    maps = np.asarray(image, dtype=float)[None, None]                  # (1, 1, 27, 27)
    for bank in (L1_21, L2_21):
        scores = C.conv2d(maps, bank, padding=1)
        wanted = (bank > 0).sum(axis=(1, 2, 3))                          # the number of +1 of each filter
        maps = C.max_pool2d((scores == wanted[None, :, None, None]).astype(float), 3)
        if bank is L1_21:
            level1 = maps[0]
    return level1, maps[0], C.conv2d(maps, L3_21)[0, :, 0, 0]
```
Les niveaux 1 et 2 répètent les mêmes trois étapes : une convolution qui garde la taille ($27 \to 27$, puis $9 \to 9$), une carte des positions où le score atteint le nombre de $+1$ du filtre, un max pooling 3 × 3 ($27 \to 9$, puis $9 \to 3$). Au niveau 3, une convolution 3 × 3 sans padding sur des cartes 3 × 3 rend un seul score par filtre : $(4, 0)$ pour `FACE_21`. Le filtre « face » atteint son nombre de $+1$ (4) : c) **A**. Pour `SHIFTED_21`, aucun des deux n'y arrive ($(1, 0)$) : d) **C**. La figure : `plt.subplots(1, 10)`, l'image, les huit cartes et un `ax.bar` des deux scores.
**Dans tes notes** : le max pooling regroupe les pixels par blocs **fixes** : décalé d'un pixel, un trait passe d'un bloc à son voisin et le motif du niveau 2 n'est plus aligné. La hiérarchie tolère les positions à l'intérieur d'un bloc, pas n'importe quel décalage. Un filtre du niveau 3 dépend de 35 × 35 pixels de l'image (✏️ 21.5 f). Un CNN entraîné apprend des poids réels à partir de milliers d'exemples, avec plus de filtres, des chevauchements et des poolings de stride 2 : il tolère décalages, déformations et bruit.
**Erreurs fréquentes** : une correspondance `scores > 0` au lieu de `== nombre de +1` (presque tout s'allume : 112 cases au niveau 1) ; compter les cases avant le pooling (49 au niveau 1) ; oublier `padding=1` (des cartes 25 × 25, puis 8 × 8 : les blocs ne tombent plus juste) ; un nombre de $+1$ unique pour tout le banc au lieu d'un par filtre.
**Variante** : sans le petit trait sous la bouche (`FACE_21[25, 12:15] = 0`), le score « face » tombe à **3** : la bouche du niveau 2 n'est plus trouvée, et le niveau 3, qui exige ses quatre parties, répond **C**. Retourné de gauche à droite (`FACE_21[:, ::-1]`), le dessin reste un visage de face (score **4**), alors que les traits ajoutés autour ne sont pas symétriques : les filtres de face sont eux-mêmes symétriques.

### Ex 21.22 — Le convnet MNIST du livre en PyTorch, sur Fashion-MNIST 📦
**Réponses** : la sortie `(5, 10)`, les onze couches et les 1 199 882 paramètres passent ; loss d'entraînement par epoch 0,92, 0,57 et 0,47 (une trentaine de secondes sur CPU pour 3 epochs) ; accuracy de validation **0,852** (au moins 0,80 attendu) ; a) **12 544** ; b) **1 625 866**.
**Démarche** :
```python
def make_convnet_22():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3), nn.ReLU(),
        nn.Conv2d(32, 64, 3), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(64 * 12 * 12, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 10))                       # logits: the softmax is inside nn.CrossEntropyLoss


def train_22(model, X, y, epochs=2, batch_size=128, lr=1e-3, seed=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        total = 0.0
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            xb, yb = X[batch].to(DEVICE), y[batch].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)              # weighted by the size of the (last, smaller) mini-batch
        history.append(total / len(X))
    return history
```
a) Avec des images 32 × 32 : $32 \to 30 \to 28$, puis le pooling, 14 : $64 \times 14 \times 14 = 12\,544$. b) Seule la première couche dense change : $320 + 18\,496 + (12\,544 \times 128 + 128) + 1\,290 = 1\,625\,866$.
**Dans tes notes** : le `lr` d'Adadelta multiplie un pas déjà adapté à l'échelle des gradients, d'où 1,0 par défaut ; avec Adam, `lr` **est** la taille du pas, et 1,0 ferait exploser l'entraînement (défaut $10^{-3}$). 98 % des paramètres sont dans la première couche dense ($9\,216 \times 128$) ; un pooling moyen global ramènerait les 64 cartes à 64 nombres et cette couche à $64 \times 128 + 128 = 8\,320$ paramètres, au prix de l'information de position.
**Erreurs fréquentes** : une softmax en dernière couche (la loss l'applique déjà : l'entraînement ralentit beaucoup) ; `nn.Linear(64 * 14 * 14, 128)`, les convolutions supposées sans effet sur la taille ($28 \to 14$) ; un générateur recréé à chaque epoch (le même ordre chaque fois) ; la moyenne des loss des mini-batches sans les pondérer (le dernier, plus petit, compte trop) ; `loss` au lieu de `loss.item()` dans la somme (le graphe de chaque mini-batch est gardé en mémoire) ; oublier `model.train()` après une mesure en mode évaluation.
**Variante** : sans ses deux dropouts, le réseau a **les mêmes** 1 199 882 paramètres (un dropout n'en a aucun) ; avec 16 puis 32 filtres au lieu de 32 puis 64, il en a $160 + 4\,640 + (4\,608 \times 128 + 128) + 1\,290 =$ **596 042**.

### Ex 21.23 — Diagnostiquer les courbes et la matrice de confusion du CNN 📈
**Réponses** : `per_class_23` passe sur `CM_23` et sur une matrice à ligne et colonne vides ; a) **B** ; b) **7** ; c) **0,075** ; d) **6** (Shirt) ; e) **0** (T-shirt/top) ; f) **0,528**.
**Démarche** :
```python
def per_class_23(cm):
    cm = np.asarray(cm, dtype=float)
    diag = np.diag(cm)
    rows, cols = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(diag, rows, out=np.zeros_like(diag), where=rows > 0)
    precision = np.divide(diag, cols, out=np.zeros_like(diag), where=cols > 0)
    return recall, precision


recall_23, precision_23 = per_class_23(CM_23)
errors_23 = CM_23 - np.diag(np.diag(CM_23))          # the off-diagonal cells: the errors only
dropout_23 = "B"
stop_23 = int(np.argmin(RUNS_23["A"]["val_loss"])) + 1
gap_23 = RUNS_23["A"]["train_acc"][-1] - RUNS_23["A"]["val_acc"][-1]
worst_23 = int(np.argmin(recall_23))
instead_23 = int(np.argmax(errors_23[worst_23]))
share_23 = (errors_23[worst_23].sum() + errors_23[:, worst_23].sum()) / errors_23.sum()
```
a) A s'écarte le plus : sa loss d'entraînement tombe à 0,014 quand celle de validation remonte jusqu'à 0,390 ; B, avec ses dropouts, garde les deux courbes proches (0,080 et 0,211 à la fin). b) La loss de validation de A est minimale à l'epoch 7 (0,228). c) $0{,}996 - 0{,}921 = 0{,}075$ (contre 0,043 pour B). d) Le recall de Shirt vaut 0,792, le plus bas (les autres sont au-dessus de 0,85). e) Sur sa ligne, hors de la diagonale, la case la plus grande est T-shirt/top (100 images sur ses 212 erreurs). f) $(212 + 165)/714 = 0{,}528$ : plus de la moitié des erreurs de tout le réseau touchent une seule classe.
**Dans tes notes** : de l'epoch 7 à l'epoch 15, la loss de validation de A monte de 70 % alors que son accuracy de validation reste vers 0,92 : le réseau devient **trop sûr de lui** sur les images qu'il rate (une probabilité de 0,99 sur la mauvaise classe coûte bien plus que 0,6), sans changer ses prédictions. Une loss qui remonte est donc le premier signe d'overfitting, avant l'accuracy. Shirt, T-shirt/top, Pullover et Coat ne se distinguent que par des détails (col, boutons, manches) qu'une image 28 × 28 en gris rend mal : il faudrait plus de filtres et de profondeur, de l'augmentation de données (ch. 24), ou accepter que certaines images sont ambiguës même pour un humain.
**Erreurs fréquentes** : diviser par la somme de la colonne pour le recall (c'est la precision) ; `np.argmin` sans le `+ 1` des epochs ; prendre l'epoch du minimum de la loss d'**entraînement**, ou du maximum de l'accuracy de validation (12) ; e) lu sur la colonne de Shirt (les images prédites Shirt à tort, surtout des Pullover) ; f) divisé par les 10 000 images au lieu des 714 erreurs ; un recall en `nan` pour une ligne vide (`0 / 0`).
**Variante** : la classe de plus faible **precision** de `CM_23` est aussi Shirt (**0,830**) : parmi les 971 images que le réseau appelle Shirt, 165 sont autre chose. La plus forte precision est celle de Sandal (**0,992**).

### Ex 21.24 — Le CNN dont les formes ne passent pas 🐛
**Réponses** : les quatre points du diagnostic passent, et le réseau réparé a 1 199 882 paramètres. Sur le code fautif, le premier « Tout exécuter » affiche le point 1 en échec (« conv1 reçoit un tenseur de forme (2, 28, 28, 1) pour 2 images », avec l'erreur de PyTorch) et les points 2 à 4 en attente.
**Démarche** :
```python
class Net24(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)        # bug 2: conv2 reads the 32 channels of conv1
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # bug 3: 28 -> 26 -> 24 -> pooling -> 12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):                        # x: (N, 28, 28, 1), the layout of the Keras notebook
        x = x.permute(0, 3, 1, 2)                # bug 1: PyTorch wants (N, C, H, W)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)        # bug 4: keep the axis of the examples
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```
Chaque bug cache le suivant : tant que `conv1` reçoit des images au mauvais format, `conv2` ne reçoit rien. On corrige dans l'ordre des diagnostics. Les erreurs de PyTorch : l'image au format Keras, `RuntimeError: Given groups=1, weight of size [32, 1, 3, 3], expected input[2, 28, 28, 1] to have 1 channels, but got 28 channels instead` ; `conv2`, le même message avec 64 et 32 ; `torch.flatten(x)` sur 2 images, un vecteur de $2 \times 9\,216$ valeurs que `fc1` refuse (`mat1 and mat2 shapes cannot be multiplied`) ; `fc1` de la mauvaise taille, le même message, avec $9\,216$ contre $10\,816$.
**Dans tes notes** : avec une seule image, `torch.flatten(x)` donne un vecteur de 9 216 valeurs, que `nn.Linear` accepte : la sortie a la forme `(10,)` au lieu de `(1, 10)`, sans erreur. Avec 2 images, le vecteur a 18 432 valeurs et `fc1` refuse. Pour éviter le calcul de tête : faire passer `torch.zeros(1, 1, 28, 28)` et afficher `x.shape` après chaque couche, ou `nn.LazyLinear(128)`, qui lit son nombre d'entrées au premier appel.
**Erreurs fréquentes** : `x.reshape(-1, 1, 28, 28)` au lieu de `permute` (les nombres sont dans le bon ordre seulement parce qu'il n'y a qu'un canal : avec trois canaux, l'image serait brouillée sans erreur) ; `x.view(x.size(0), -1)` après `permute` sans `contiguous` (refusé) ; changer `conv1` en `nn.Conv2d(28, 32, 3)` pour faire taire l'erreur ; corriger `fc1` en `64 * 13 * 13` → `64 * 14 * 14` (le pooling appliqué à 28).
**Variante** : le même réseau traduit avec `padding=1` dans les deux convolutions donne $28 \to 28 \to 28 \to 14$, et `fc1` doit lire $64 \times 14 \times 14 =$ **12 544** valeurs : le réseau a alors **1 625 866** paramètres.

### Ex 21.25 — Briser la symétrie : filtres identiques, aléatoires ou He 🔬
**Réponses** : `init_25` (poids à 0,05, biais à 0, refus d'un nom inconnu) et `distinct_filters_25` passent ; après une epoch sur 5 000 images : `"constant"`, **1** filtre distinct sur 8 et 0,682 d'accuracy de validation ; `"default"`, **8** filtres et 0,726 ; `"he"`, **8** filtres et 0,731.
**Démarche** :
```python
def init_25(model, kind):
    if kind not in ("constant", "default", "he"):
        raise ValueError(f"kind must be 'constant', 'default' or 'he', got {kind!r}")
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        if kind == "default":
            layer.reset_parameters()
            continue
        with torch.no_grad():
            if kind == "constant":
                layer.weight.fill_(0.05)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            layer.bias.zero_()


def distinct_filters_25(weight, tol=1e-6):
    filters = as_numpy(weight).reshape(len(weight), -1)
    kept = []
    for f in filters:
        if not any(np.max(np.abs(f - g)) <= tol for g in kept):
            kept.append(f)
    return len(kept)
```
Les modifications de poids se font sous `torch.no_grad()` : ce sont des opérations en place sur des feuilles qui demandent un gradient. Compter les filtres distincts revient à garder un représentant de chaque groupe.
**Dans tes notes** : avec des poids tous égaux, les huit cartes de la convolution sont identiques, la couche dense les lit avec les mêmes poids, donc les huit filtres reçoivent le même gradient et Adam leur applique le même pas : ils restent égaux pour toujours (ch. 16). Un seul filtre utile fait déjà bien mieux que le hasard (10 %), mais moins bien que huit. Entre `"default"` et `"he"`, l'écart après une epoch sur un réseau aussi peu profond est de l'ordre de ce que change une graine : He compte surtout pour les réseaux profonds sans normalisation.
**Erreurs fréquentes** : `layer.weight = 0.05` (refusé : `TypeError: cannot assign 'float' as parameter 'weight'`) ; `layer.weight.data = ...` sans les biais ; oublier la couche dense (le test de 0,05 partout échoue, et la couche dense aléatoire brise déjà la symétrie : les filtres se séparent, voir la variante) ; `np.unique` sur des filtres réels (deux filtres qui diffèrent de $10^{-9}$ sont comptés distincts) ; comparer chaque filtre au **précédent** seulement.
**Variante** : avec les poids de la **convolution seulement** à 0,05 (biais à 0) et l'initialisation par défaut pour la couche dense, on retrouve **8** filtres distincts après une epoch : la couche dense lit les huit cartes, identiques au départ, avec des poids différents, donc les huit filtres reçoivent des gradients différents dès le premier pas. Il suffit qu'**une** couche distingue les neurones pour briser la symétrie.

### Ex 21.26 — Stride ou pooling : accuracy et temps sur CIFAR-10 🔬
**Réponses** : les deux réseaux ont **60 362** paramètres ; **5 644 288** multiplications-additions par image avec le pooling, **1 441 792** avec le stride ; en mode rapide (2 epochs sur 5 000 photos), 3,8 s d'entraînement et 0,397 d'accuracy sur 1 000 photos de test pour le pooling, 1,0 s et 0,363 pour le stride.
**Démarche** :
```python
def make_pool_26():
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 8 * 8, 10))


def make_stride_26():
    return nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(64 * 8 * 8, 10))
```
Les deux réseaux ont exactement les mêmes poids ($896 + 18\,496 + 40\,970 = 60\,362$ paramètres) ; le pooling calcule sa première convolution sur 32 × 32 positions et la seconde sur 16 × 16, le stride sur 16 × 16 puis 8 × 8 : $32 \times 27 \times 1\,024 + 64 \times 288 \times 256 + 40\,960 = 5\,644\,288$ multiplications-additions contre $1\,441\,792$, quatre fois moins pour les convolutions.
**Dans tes notes** : l'écart de calcul vient de la taille des cartes, pas des poids. Ici, le pooling gagne 3 points d'accuracy pour un entraînement presque quatre fois plus long : il regarde chaque position avant de réduire, ce qui compte sur un réseau aussi petit (et l'écart, sur 1 000 photos, est à peine plus grand que l'incertitude de la mesure). Les architectures récentes réduisent par des convolutions de stride 2 : le réseau apprend comment réduire, pour un calcul bien moindre, et la profondeur compense.
**Erreurs fréquentes** : oublier `padding=1` (des cartes 30 × 30 puis 13 × 13, et une couche dense de la mauvaise taille) ; un pooling après la convolution de stride 2 (la taille tombe à 4 × 4) ; comparer des temps mesurés sur des machines ou des charges différentes ; conclure sur 1 000 photos de test sans regarder l'écart-type ($\sqrt{0{,}5 \times 0{,}5/1\,000} \approx 1{,}6$ point).
**Variante** : avec trois convolutions (32, 64, 128 filtres) au lieu de deux, le réseau à stride finit sur des cartes **4 × 4** et sa couche dense lit $128 \times 16 = $ **2 048** valeurs.

### Ex 21.27 — Filtres et cartes d'activation d'un ResNet-18 pré-entraîné 📦
**Réponses** : les filtres `(64, 7, 7, 3)` ramenés dans $[0, 1]$, l'ordre de leurs valeurs, les cartes de `layer1` `(1, 64, 56, 56)` contre les couches appelées une à une, le hook retiré, et celles de `layer4` `(1, 512, 7, 7)` passent ; le ResNet classe les quatre photos de CIFAR-10 (un oiseau, un chat, deux bateaux) en « toy terrier », « muzzle », « moving van » et « amphibian », avec 15 à 24 % de probabilité.
**Démarche** :
```python
def filters_27(model):
    w = model.conv1.weight.detach().cpu().double().permute(0, 2, 3, 1).numpy()    # (64, 3, 7, 7) -> (64, 7, 7, 3)
    low = w.min(axis=(1, 2, 3), keepdims=True)
    high = w.max(axis=(1, 2, 3), keepdims=True)
    return (w - low) / (high - low)


def maps_27(model, x, layer):
    captured = {}
    hook = layer.register_forward_hook(lambda module, inputs, output: captured.setdefault("out", output.detach()))
    try:
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        hook.remove()
    return captured["out"]
```
`imshow` veut des images `(H, W, 3)` dans $[0, 1]$ : on permute les axes et on ramène chaque filtre par **son** minimum et **son** maximum. Le hook garde la sortie de la couche pendant une passe avant ordinaire, puis est retiré dans un `finally`.
**Dans tes notes** : les filtres de `conv1` sont des détecteurs d'arêtes orientées, de rayures et de taches de couleur, ceux qu'on trouve à la première couche de presque tous les CNN entraînés sur des photos (le livre ne montre pas ces poids, mais les images qui excitent les filtres, §21.9.2). Les cartes de `layer1` (56 × 56) gardent la forme de la photo ; celles de `layer3` (14 × 14) sont des taches où il devient difficile de reconnaître l'image : elles répondent à des motifs plus grands et plus abstraits. Les photos 32 × 32 agrandies en 224 × 224 sont floues, très loin des photos d'ImageNet : les classes rendues (un chien, une muselière et deux véhicules, avec moins de 25 % de probabilité) en témoignent.
**Erreurs fréquentes** : `reshape(64, 7, 7, 3)` au lieu de `permute` (les canaux se mélangent aux pixels) ; un seul minimum et un seul maximum pour les 64 filtres (les filtres faibles deviennent gris uniforme) ; oublier `hook.remove()` (chaque appel ajoute un hook, et la mémoire grossit) ; oublier `model.eval()` (la batchnorm met à jour ses statistiques sur quatre photos) ; donner les photos en `uint8` sans le prétraitement des poids.
**Variante** : `maps_27(RESNET_27, x, RESNET_27.layer2)` a la forme **(1, 128, 28, 28)** : chaque bloc de ResNet-18 double les canaux et divise la taille par 2 (sauf `layer1`, qui garde 64 canaux de 56 × 56 après le max pooling d'entrée).

### Ex 21.28 — Ce qu'un filtre cherche : montée de gradient sur l'entrée 📦
**Réponses** : l'image `(1, 1, 28, 28)` dans $[0, 1]$, détachée, les poids et leurs gradients inchangés passent ; le canal 0 de la seconde convolution passe de −0,947 sur l'image de départ à **0,459** sur l'image trouvée, au-dessus de la meilleure des 500 images réelles (0,286) ; la figure montre les motifs des 8 premiers canaux et des 4 classes.
**Démarche** :
```python
def ascent_28(model, upto, channel, steps=60, lr=0.1, seed=0):
    part = model[:upto + 1]
    model.eval()
    frozen = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)                              # the weights are fixed: only the image is optimised
    x = 0.4 + 0.2 * torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(seed))
    x = x.to(DEVICE).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    try:
        for _ in range(steps):
            optimizer.zero_grad()
            loss = -part(x)[0, channel].mean()               # ascent: minimise the opposite
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)                           # stay a valid image
    finally:
        for p, flag in zip(model.parameters(), frozen):
            p.requires_grad_(flag)
    return x.detach()
```
C'est l'entraînement à l'envers : les poids sont fixes, l'image est le paramètre. `model[:3]` s'arrête juste après la seconde convolution, avant sa ReLU ; `model[:11]` est le réseau entier, et le canal $c$ y est le logit de la classe $c$.
**Dans tes notes** : certains canaux de la seconde convolution cherchent des textures répétées (rayures, grilles, grains : les canaux 1, 3 et 7) ; d'autres donnent une image presque uniforme, noire ou blanche (0, 2, 4, 6) : ils répondent au fond ou à l'encre, partout. Un filtre 3 × 3 répond partout où son petit motif se trouve, et la moyenne sur les $24 \times 24$ positions récompense une image qui le répète partout. Aucun motif ne ressemble à un vêtement. Les images des classes (pantalon, sandale, basket, sac) non plus : la montée de gradient trouve ce qui fait monter le **score**, pas ce qui ressemble aux exemples, et exploite des détails qu'aucune vraie image n'a, la même faiblesse que celle des exemples adversariaux (21.29). Sans le `clamp`, les pixels partiraient bien au-delà de $[0, 1]$, et l'image ne serait plus une image.
**Erreurs fréquentes** : `torch.optim.Adam(model.parameters())` (on entraîne le réseau, pas l'image) ; la loss sans le signe moins (on éteint le filtre) ; `x.clamp(0, 1)` sans le `_` ni l'affectation (rien ne change) ; un `clamp_` hors de `torch.no_grad()` (refusé sur une feuille qui demande un gradient) ; le mode entraînement (les dropouts brouillent chaque pas) ; oublier de remettre `requires_grad` à la fin.
**Variante** : maximiser la **somme** au lieu de la moyenne donne presque la même image (le gradient est multiplié par le nombre de positions, $24 \times 24 = 576$, et Adam divise chaque pas par la taille de ses gradients récents) ; maximiser une seule position, `part(x)[0, channel, 12, 12]`, ne change que les pixels de son champ récepteur, une zone de **5 × 5** pixels (deux convolutions 3 × 3 sans pooling : $3 + 2$).

### Ex 21.29 — Attaque FGSM contre ton CNN Fashion-MNIST 📦
**Réponses** : la formule contre la fiche, `x` inchangé, les gradients des poids inchangés et `eps = 0` passent ; sur 1 000 images de validation, l'accuracy passe de 0,861 ($\varepsilon = 0$) à 0,766, 0,587, **0,279** et 0,045 pour $\varepsilon$ = 0,02, 0,05, 0,1 et 0,2, contre 0,858, 0,853, **0,849** et 0,823 pour du bruit de signes aléatoires de même taille.
**Démarche** :
```python
def fgsm_29(model, x, y, eps):
    model.eval()
    x = x.clone().detach().to(DEVICE).requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x), y.to(DEVICE))
    (grad,) = torch.autograd.grad(loss, x)                   # only the gradient with respect to x: the weights' .grad stay
    return (x + eps * grad.sign()).clamp(0.0, 1.0).detach()


def attack_curve_29(model, X, y, eps_list, batch_size=500):
    curve = []
    for eps in eps_list:
        correct = 0
        for i in range(0, len(X), batch_size):
            xb, yb = X[i:i + batch_size], y[i:i + batch_size]
            adv = fgsm_29(model, xb, yb, eps)
            with torch.no_grad():
                correct += (model(adv).argmax(dim=1).cpu() == yb).sum().item()
        curve.append(correct / len(X))
    return curve
```
Un seul gradient, celui de la loss par rapport aux pixels, dont on garde le signe : chaque pixel bouge de $\pm \varepsilon$ dans la direction qui augmente le plus la loss **au premier ordre**. `torch.autograd.grad` rend ce gradient sans rien écrire dans le `.grad` des poids.
**Dans tes notes** : du bruit de même taille ne fait presque rien, parce que ses signes aléatoires se compensent dans le produit scalaire de chaque neurone ; FGSM aligne les $784$ changements avec le gradient, et leurs effets s'additionnent : $784 \times \varepsilon \times$ la taille moyenne d'une composante du gradient. C'est l'argument « linéaire » de Goodfellow et coll. (2015) : un réseau aux ReLU est presque linéaire par morceaux. Les images attaquées à $\varepsilon = 0{,}1$ restent reconnaissables pour un humain (un léger voile granuleux).
**Erreurs fréquentes** : `loss.backward()` (le `.grad` des poids est rempli, et la vérification le voit) ; `x.requires_grad_(True)` sur le tenseur de l'appelant (il est modifié) ; oublier le `clamp` ; le mode entraînement (les dropouts changent le gradient à chaque appel) ; `eps * grad` au lieu de `eps * grad.sign()` (une perturbation minuscule, très loin du budget $\varepsilon$) ; comparer les prédictions sur `DEVICE` à des labels sur le CPU.
**Variante** : la perturbation de FGSM a une norme $L_\infty$ d'au plus $\varepsilon$ (**0,1** à $\varepsilon = 0{,}1$) mais une norme $L_2$ qui peut atteindre $\varepsilon\sqrt{784} =$ **2,8** : invisible pixel par pixel, elle est grande au total.

### Ex 21.30 — conv_transpose2d par insertion de zéros 🔨
**Réponses** : les quatre cas contre `F.conv_transpose2d` et les trois refus passent ; les 29 tests de l'exercice passent.
**Démarche** :
```python
def conv_transpose2d(x, weight, bias=None, stride=1, padding=0):
    arr = _batch(x)
    w = np.array(weight, dtype=np.float64)
    if w.ndim != 4 or w.shape[0] != arr.shape[1]:
        raise ValueError(f"weight must have the shape (C_in={arr.shape[1]}, C_out, kh, kw), got {w.shape}")
    c_in, c_out, kh, kw = w.shape
    if bias is not None and np.shape(bias) != (c_out,):
        raise ValueError(f"bias must have shape ({c_out},), got {np.shape(bias)}")
    sh, sw = _as_pair(stride, "stride", 1)
    ph, pw = _as_pair(padding, "padding", 0)
    if ph > kh - 1 or pw > kw - 1:
        raise ValueError(f"padding ({ph}, {pw}) must be at most k - 1 = ({kh - 1}, {kw - 1})")
    n, _, h, wd = arr.shape
    if (h - 1) * sh - 2 * ph + kh < 1 or (wd - 1) * sw - 2 * pw + kw < 1:
        raise ValueError("the output would be empty")
    spread = np.zeros((n, c_in, (h - 1) * sh + 1, (wd - 1) * sw + 1))
    spread[:, :, ::sh, ::sw] = arr                   # s - 1 zeros between two input cells
    flipped = w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)          # flipped kernel, channel axes swapped: (C_out, C_in, kh, kw)
    return conv2d(spread, flipped, bias, padding=(kh - 1 - ph, kw - 1 - pw))
```
La recette du §21.8 : $s - 1$ zéros entre deux cases, une bordure de $k - 1 - p$ zéros, le noyau retourné, puis une convolution ordinaire ; la sortie mesure $(H - 1)s - 2p + k$.
**Erreurs fréquentes** : oublier de retourner le noyau (juste seulement pour des noyaux symétriques) ; ne pas échanger les deux axes de canaux (le poids d'une convolution transposée est rangé `(C_in, C_out, …)`) ; insérer des zéros **après** la dernière case (une ligne en trop) ; une bordure de $k - 1$ au lieu de $k - 1 - p$.
**Variante** : avec un noyau 3 × 3, un stride de 3 et aucun padding, les fenêtres de la sortie ne se recouvrent pas : chaque case de l'entrée devient un bloc 3 × 3 (son noyau multiplié par sa valeur), et une carte 4 × 4 devient $(4 - 1) \times 3 + 3 =$ **12 × 12**.

### Ex 21.31 — conv2d_backward : rétropropager à travers une convolution 🔨
**Réponses** : `dx`, `dW` et `db` des quatre cas contre l'autograd de PyTorch, et le refus, passent ; les 22 tests de l'exercice passent.
**Démarche** :
```python
def conv2d_backward(dout, x, weight, stride=1, padding=0):
    arr = _batch(x)
    w = np.array(weight, dtype=np.float64)
    g = np.array(dout, dtype=np.float64)
    n, c_in, h, wd = arr.shape
    c_out, _, kh, kw = w.shape
    stride = _as_pair(stride, "stride", 1)
    ph, pw = _as_pair(padding, "padding", 0)
    ho = conv_output_size(h, kh, stride[0], ph)
    wo = conv_output_size(wd, kw, stride[1], pw)
    if g.shape != (n, c_out, ho, wo):
        raise ValueError(f"dout must have the shape of the output {(n, c_out, ho, wo)}, got {g.shape}")
    cols = im2col(arr, (kh, kw), stride, (ph, pw))                  # (N, C_in kh kw, L)
    g2 = g.reshape(n, c_out, ho * wo)                                # (N, C_out, L)
    db = g.sum(axis=(0, 2, 3))
    dw = np.einsum("nol,nkl->ok", g2, cols).reshape(w.shape)        # sum over the images and the positions
    dcols = (w.reshape(c_out, -1).T @ g2).reshape(n, c_in, kh, kw, ho, wo)
    dxp = np.zeros((n, c_in, h + 2 * ph, wd + 2 * pw))
    for i in range(kh):                              # send each window's gradient back to its cells (col2im)
        for j in range(kw):
            dxp[:, :, i:i + stride[0] * (ho - 1) + 1:stride[0], j:j + stride[1] * (wo - 1) + 1:stride[1]] += dcols[:, :, i, j]
    return dxp[:, :, ph:ph + h, pw:pw + wd].copy(), dw, db
```
La passe avant est `W_aplati @ colonnes` : `dW` est la somme sur les images de `dout @ colonnesᵀ`, le gradient des colonnes `W_aplatiᵀ @ dout`, et chaque colonne renvoie le sien aux cases qu'elle a lues (« col2im », le chemin inverse d'`im2col`, avec des `+=` : une case lue par plusieurs fenêtres reçoit la somme). `db` somme `dout` sur les images et les positions (∂ 21.7).
**Erreurs fréquentes** : calculer `dx` par `conv_transpose2d(dout, W, stride, padding)` : juste tant que chaque ligne de l'image est lue, mais quand le stride laisse des lignes non lues en bas (6 pixels, un noyau 3, un stride 2), la sortie est trop courte d'une ligne ; `dxp[...] = ...` au lieu de `+=` (les recouvrements sont perdus) ; oublier de retirer la bordure ; une moyenne au lieu d'une somme pour `dW` et `db`.
**Variante** : avec un noyau 1 × 1 et un stride de 1, `conv2d_backward` se réduit à celle d'une couche dense appliquée en chaque pixel : `dW[o, c] = Σ_{n, i, j} dout[n, o, i, j] · x[n, c, i, j]`, et `dx = Wᵀ dout` en chaque pixel, la formule du ch. 18.

### Ex 21.32 — Défi Fashion-MNIST : au moins 92 % avec moins de 100 000 paramètres 🏆
**Réponses** : 80 698 paramètres, entraînement en 98 s (8 epochs sur les 60 000 images, au plus 150 s), 0,948 sur 10 000 images d'entraînement, et **0,923** sur le test caché : le défi (au moins 90 % en mode rapide) et le palier 🌟 (92 %) sont atteints.
**Démarche** :
```python
def train_32(X, y):
    torch.manual_seed(32)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),     # 28 -> 14
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 14 -> 7
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * 7 * 7, 48), nn.ReLU(), nn.Linear(48, 10)).to(DEVICE)
    epochs, batch = (8, 128) if FAST_MODE else (25, 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=epochs * math.ceil(len(X) / batch))
    generator = torch.Generator().manual_seed(32)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=generator).split(batch):
            loss = nn.functional.cross_entropy(model(X[idx].to(DEVICE)), y[idx].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model.eval()
```
Deux blocs « convolution, batchnorm, ReLU, pooling » ramènent l'image à 32 cartes de 7 × 7 pour seulement $160 + 32 + 4\,640 + 64 = 4\,896$ paramètres ; la couche dense de 48 neurones en prend $1\,568 \times 48 + 48 = 75\,312$, la dernière 490 : 80 698 en tout. Le planning `OneCycleLR` (ch. 19) monte puis redescend le learning rate, ce qui permet un `lr` élevé au milieu et une fin fine en peu d'epochs.
**Calibrage** (CPU à 2 cœurs, 60 000 images) : le même réseau avec AdamW à un learning rate constant de $3 \times 10^{-3}$ donnait 0,908 sur le test après 4 epochs (62 s) et 0,919 après 9 (146 s) ; avec `OneCycleLR`, 8 epochs donnent 0,923 et 9 epochs 0,923 : 8 epochs tiennent dans les 150 s avec de la marge. Le mode complet (25 epochs, 600 s permises) reste à mesurer sur Colab.
**Erreurs fréquentes** : partir du convnet de 21.22 (1,2 million de paramètres, presque tous dans la couche dense) ; oublier `model.eval()` avant la mesure (dropout et batchnorm en mode entraînement) ; régler les hyperparamètres en regardant le test caché ; trop d'epochs pour le temps permis (une epoch coûte environ 12 s ici).
**Variante** : en remplaçant `nn.Flatten()` et la couche dense par un pooling moyen global (`nn.AdaptiveAvgPool2d(1)`, puis `nn.Flatten()` et `nn.Linear(32, 10)`), le réseau n'a plus que **5 226** paramètres, mais, avec le même entraînement, ne fait plus que 0,854 sur le test (contre 0,923) : 32 nombres par image ne gardent pas la position des motifs.
