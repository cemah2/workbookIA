# 28 · Applications créatives — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels et des exercices ✏️ 28.1, 28.2 et 28.4 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 28.Q1 — Pourquoi partir d'un réseau pré-entraîné comme VGG16
a) **Faux** : le réseau est gelé ; seuls les pixels de l'image changent. b) **16** : 13 convolutions et 3 couches denses (les poolings n'ont pas de poids). c) **B** : ses filtres, appris sur plus d'un million de photos, détectent déjà des motifs très variés ; on s'en sert comme d'un instrument de mesure. d) **B** : 1,28 million d'images d'entraînement ; le « demi-million » du livre est faux, et 14 millions est la taille de toute la base ImageNet. e) **1 000**.
**À retenir** : on emprunte à un réseau pré-entraîné ses détecteurs de motifs ; on n'y touche pas.

### 28.Q2 — Montée de gradient sur l'entrée, poids gelés
a) **B** : les pixels. b) **Vrai** : $\nabla(-a) = -\nabla a$, donc un pas de descente sur $-a$ est un pas de montée sur $a$ ; c'est ainsi qu'on l'écrit avec un optimiseur PyTorch. c) **A** : un bruit uniforme entre −1 et 1. d) **B** : l'optimiseur ne reçoit que l'image. e) **Vrai** : geler les poids supprime leur gradient, pas celui des tenseurs qui demandent `requires_grad=True`, comme l'image.
**À retenir** : l'image est le seul paramètre ; `Adam([x])`, réseau gelé et en `eval()`.

### 28.Q3 — Visualiser une couche entière, pondérer ses filtres
a) **B**. b) **A** : rayures, couleurs, textures fines ; les parties d'objets viennent plus profond. c) **Vrai** : augmenter $-a_i$, c'est faire baisser $a_i$. d) **512**. e) **28** : trois poolings avant le bloc 4, $224 / 2^3$.
**À retenir** : la visualisation d'une couche additionne les activations de ses filtres, éventuellement pondérées ; la profondeur fait passer des traits aux formes.

### 28.Q4 — Deep dream : ce qui change par rapport à la visualisation d'une couche
a) **« AB »** : une photo au départ et plusieurs couches pondérées ; les poids du réseau ne changent jamais (C), et le rêve fait **monter** les activations, comme la visualisation (D). b) **Vrai**. c) **A** : *inceptionism*, du nom du réseau GoogLeNet (Inception) utilisé par Google. d) **C** : un masque sur le gradient (ou sur l'activation) limite le rêve à une zone. e) **Faux** : la visualisation et le rêve font **monter** une activation ; le transfert de style fait **baisser** des écarts à des cibles.
**À retenir** : deep dream = visualisation de plusieurs couches pondérées, à partir d'une photo (et, dans la version de Google, à plusieurs échelles).

### 28.Q5 — Des yeux partout : l'écho des données d'entraînement
a) **C**. b) **Faux** : plus d'une centaine des 1 000 classes d'ImageNet-1k sont des races de chiens, plus d'un dixième. c) **B** : un réseau ne rêve que de ce qu'il a appris. d) **B** : les couches profondes détectent des parties d'objets ; les couches précoces, des traits et des textures. e) **Vrai**.
**À retenir** : ce qu'un réseau « voit » dans une image renseigne autant sur ses données que sur l'image.

### 28.Q6 — Image de base, référence de style, image générée : que minimise-t-on ?
a) **C** : l'image générée ; la photo et le tableau ne servent qu'à calculer les cibles, et le réseau est gelé. b) **A**. c) **B**. d) **Faux** : contrairement au discriminateur d'un GAN, VGG16 est fixé à l'avance. e) **1** : seule l'image générée passe dans le réseau à chaque pas ; les activations de la photo et les matrices de Gram du tableau ont été calculées une fois.
**À retenir** : trois images, une seule variable ; les deux autres fournissent des cibles calculées une fois.

### 28.Q7 — Ce que la matrice de Gram retient et ce qu'elle oublie
a) **A** : $C \times C$, une ligne et une colonne par filtre. b) **B**. c) **Vrai** : chaque coefficient est une somme sur les positions, que l'on peut parcourir dans n'importe quel ordre (∂ 28.3). d) **B** : $G_{ii} = \frac{1}{CHW}\sum_k F_{ik}^2$, la moyenne des carrés divisée par $C$. e) **Faux** : Li et coll. (2017) montrent qu'égaler les matrices de Gram revient à minimiser une MMD à noyau polynomial du second degré entre les distributions de features ; le livre les cite à contresens.
**À retenir** : la matrice de Gram garde l'énergie de chaque filtre et leurs co-activations ; elle oublie où elles ont lieu.

### 28.Q8 — Loss de contenu : couches précoces ou profondes ?
a) **A** : une couche précoce garde presque toute l'information des pixels. b) **B** : la structure et les objets restent, les couleurs et les détails fins se perdent. c) **C** : le livre prend la couche bloc 1, conv 2 ; le code de Chollet prend `block5_conv2`. d) **Faux** : Gatys et coll. prennent `conv4_2`, une couche profonde. e) **B**.
**À retenir** : plus la couche de contenu est profonde, plus le style a de place ; le choix du livre (bloc 1) est inhabituel.

### 28.Q9 — Loss de style, poids relatifs et variation totale
a) **B** : les 13 convolutions de VGG16. b) **40** : $1 / 0{,}025$. c) **C**. d) **Vrai** : changer la normalisation de la matrice de Gram multiplie la loss de style par un facteur qui peut dépasser $10^{10}$ (✏️ 28.4 g) ; les poids se règlent pour **un** code donné. e) **Faux** : c'est l'inverse ; partir de la photo de contenu converge plus vite, car le contenu est déjà juste et il ne reste qu'à y faire passer le style.
**À retenir** : des poids ne se comparent qu'à normalisation égale ; on règle en regardant la valeur de chaque terme.

### 28.Q10 — Générer du texte mot à mot : ce qu'implique « le mot le plus probable »
a) **A**. b) **Faux** : le glouton est déterministe ; même début, même suite. c) **B** : des répétitions et des boucles ; un modèle mot à mot ne peut pas sortir un mot hors de son vocabulaire. d) **Faux** : la couche de sortie a exactement un neurone par mot du vocabulaire (les mots rares deviennent souvent `<unk>`). e) **A** : diviser par $T < 1$ agrandit les écarts entre logits, donc la softmax se concentre sur les plus grands.
**À retenir** : le glouton est déterministe et tourne en boucle ; on échantillonne, avec une température, un top-k ou un top-p.

<a id="rappels"></a>

## 🔁 Rappels

### 28.R1 — Un réseau qui sert de loss (le discriminateur)
a) **A** : le générateur ne voit jamais de vraie image ; il apprend par le gradient qui traverse le discriminateur. b) **B**. c) **Vrai**. d) **B** : chez Gatys, on optimise les pixels d'une image ; le transfert rapide de Johnson et coll. entraîne, lui, un réseau générateur. e) **B** : une loss perceptuelle ; une loss adversariale vient d'un discriminateur qui s'entraîne en même temps.
**À retenir** : dans un GAN comme dans le transfert de style, un réseau dérivable sert de critère ; dans le premier il apprend, dans le second il est gelé.

### 28.R2 — Mélange de contenu contre mélange de représentations
a) **B** : la moyenne de deux images superpose leurs traits, chacun à moitié d'intensité. b) **A** : le décodeur d'un VAE rend une image qui ressemble aux données, même entre deux codes. c) **Vrai**. d) **1** : $0{,}75 \times 0 + 0{,}25 \times 4$. e) **Faux** : $\frac{1}{4}(F_1 + F_2)(F_1 + F_2)^\top = \frac{1}{4}(F_1F_1^\top + F_2F_2^\top + F_1F_2^\top + F_2F_1^\top)$ ; les termes croisés font la différence avec la moyenne $\frac{1}{2}(F_1F_1^\top + F_2F_2^\top)$.
**À retenir** : mélanger des représentations n'est pas mélanger des pixels ; et une statistique quadratique ne se mélange pas linéairement.

### 28.R3 — Champ récepteur et hiérarchie des filtres de VGG16
a) **5**. b) **7** : chaque convolution 3 × 3 de pas 1 ajoute 2 pixels au côté du champ récepteur. c) **Vrai**. d) **1** : $32 / 2^5$. e) **36 928** : $64 \times 64 \times 3 \times 3 + 64$.
**À retenir** : les poolings divisent le côté par 2, les convolutions 3 × 3 élargissent le champ récepteur de 2 ; c'est pourquoi les couches profondes voient des motifs plus grands.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 28.1 — Tailles des activations et des matrices de Gram le long de VGG16 ✏️
**Réponse** : a) **3 211 264** ; b) **401 408** ; c) **262 144** ; d) **784** ; e) **131 328** ; f) **8** ; g) **Faux**.
**Démarche** : les convolutions gardent la taille, chaque pooling la divise par 2 : les cartes mesurent 224, 112, 56, 28 et 14 pixels de côté dans les blocs 1 à 5. a) $64 \times 224^2$. b) $512 \times 28^2$. c) $512^2$. d) $\frac{64 \times 224^2}{64^2} = \frac{224^2}{64} = 784$ : la matrice de Gram de la première couche est 784 fois plus petite que l'activation qu'elle résume. e) $\frac{512 \times 513}{2}$ : la diagonale et un seul des deux triangles. f) $128 / 2^4$. g) La matrice est $C \times C$ quelle que soit la taille des cartes. Dans ta copie : la loss de style compare deux matrices $C \times C$, qui ont la même taille même si les images diffèrent (la normalisation par $HW$ rend aussi leurs valeurs comparables) ; la loss de contenu compare les activations case par case, donc la photo de contenu et l'image générée doivent avoir la même taille.
**Erreurs fréquentes** : compter quatre poolings avant le bloc 4 (on trouve 14 au lieu de 28) ; oublier les canaux ; croire que la matrice de Gram dépend de la taille de l'image.
**Variante** : pour une entrée de 192 × 192, la couche bloc 3, conv 1 a des cartes de $192 / 4 = 48$ pixels de côté, soit $256 \times 48^2 = 589\,824$ valeurs ; sa matrice de Gram garde $256^2 = 65\,536$ coefficients.

### Ex 28.2 — Une matrice de Gram à la main ✏️
**Réponse** : a) **3** ; b) **11** ; c) **1** ; d) **0,917** ; e) **6,917** ; f) **0,369** ; g) **Vrai**.
$$F F^\top = \begin{pmatrix} 6 & 3 & 12 \\ 3 & 11 & 6 \\ 12 & 6 & 24 \end{pmatrix}, \qquad G = \frac{1}{12} F F^\top = \begin{pmatrix} 0{,}5 & 0{,}25 & 1 \\ 0{,}25 & 0{,}917 & 0{,}5 \\ 1 & 0{,}5 & 2 \end{pmatrix}$$
**Démarche** : a) $(1, 0, 2, 1) \cdot (0, 3, 1, 1) = 0 + 0 + 2 + 1$. b) $0 + 9 + 1 + 1$. c) $\frac{(1, 0, 2, 1) \cdot (2, 0, 4, 2)}{12} = \frac{12}{12}$. d) $\frac{11}{12}$. e) La somme des neuf coefficients de $G$ : $\frac{83}{12}$. f) $\frac{3}{\sqrt{6 \times 11}} = \frac{3}{8{,}124}$ (la normalisation se simplifie). g) Retourner chaque carte de gauche à droite permute les positions de la même façon pour les trois filtres. Dans ta copie : la similarité cosinus des filtres 1 et 3 vaut **1**, car la ligne 3 vaut exactement 2 fois la ligne 1 : les deux filtres s'allument aux mêmes endroits, dans les mêmes proportions. Les coefficients du livre dépassent 10 parce que sa matrice n'est pas normalisée : la somme porte sur les 64 cases de deux cartes de 8 × 8, et même des produits petits finissent par faire beaucoup.
**Erreurs fréquentes** : normaliser par $HW = 4$ au lieu de $CHW = 12$ (ou ne pas normaliser) ; faire la somme d'une ligne au lieu de la somme de ses carrés pour un coefficient diagonal ; oublier, dans e), que chaque coefficient hors diagonale apparaît deux fois.
**Variante** : avec deux filtres, $F = \begin{pmatrix} 2 & 0 & 1 & 1 \\ 1 & 1 & 0 & 2 \end{pmatrix}$, on trouve $F F^\top = \begin{pmatrix} 6 & 4 \\ 4 & 6 \end{pmatrix}$, $G = \begin{pmatrix} 0{,}75 & 0{,}5 \\ 0{,}5 & 0{,}75 \end{pmatrix}$ (division par 8), et une similarité cosinus de $\frac{4}{6} \approx 0{,}667$.

### Ex 28.3 — Mélanger les positions ne change pas la matrice de Gram ∂
**Réponse** : la matrice de Gram ne voit que des sommes sur les positions ; elle est invariante par toute permutation des positions commune à tous les canaux (et même par toute transformation orthogonale de l'espace des positions), mais pas par une permutation d'un seul canal ; sa diagonale vaut $\frac{1}{C}(\mu_i^2 + \sigma_i^2)$.
**Démarche** :
1. La colonne $k$ de $FP$ vaut $\sum_m F_{:, m} P_{mk}$ ; comme la colonne $k$ de $P$ a un seul 1, en ligne $\pi(k)$, c'est la colonne $\pi(k)$ de $F$ : $FP$ réordonne les colonnes de $F$, donc les positions. $(PP^\top)_{ij} = \sum_k P_{ik} P_{jk}$ : pour $i = j$, la ligne $i$ a un seul 1, d'où 1 ; pour $i \ne j$, les 1 des lignes $i$ et $j$ sont dans des colonnes différentes (une colonne n'a qu'un 1), d'où 0. Donc $PP^\top = I$.
2. $G(FP) = \frac{1}{CN} (FP)(FP)^\top = \frac{1}{CN} F P P^\top F^\top = \frac{1}{CN} F F^\top = G(F)$, avec $(AB)^\top = B^\top A^\top$.
3. Oui : le calcul du point 2 n'utilise que $QQ^\top = I$. Mais une matrice orthogonale quelconque **mélange** les valeurs (une rotation de l'espace des positions) : les nouvelles « activations » peuvent devenir négatives, ce qu'aucune ReLU ne produit. La matrice de Gram ne fait pas la différence : elle ne voit que les produits scalaires entre lignes.
4. $F = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix}$ : les deux filtres s'allument à la première position, $F F^\top = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$. En permutant le seul canal 2, $F' = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ et $F' F'^\top = I$ : le coefficient croisé passe de 1 à 0. Les co-activations dépendent de l'**alignement** des canaux, que la loss de style impose.
5. $C\,G_{ii} = \frac{1}{N}\sum_k F_{ik}^2$, la moyenne des carrés, qui vaut $\mu_i^2 + \sigma_i^2$ (la variance est la moyenne des carrés moins le carré de la moyenne, ch. 2). La matrice $G$ seule ne donne pas $\mu_i$ : le canal constant $(1, 1)$ ($\mu = 1$, $\sigma = 0$) et le canal $(0, \sqrt{2})$ ($\mu = \sigma = \frac{\sqrt{2}}{2}$) ont la même moyenne des carrés, 1. (Une matrice de Gram « centrée », calculée après avoir retranché les moyennes, donnerait les covariances ; AdaIN aligne directement $\mu_i$ et $\sigma_i$, 🔬 28.25.)
6. La loss de style impose des **statistiques d'ordre 2** des features : l'énergie de chaque filtre et leurs co-activations, moyennées sur toute l'image. Elle laisse libre la **disposition** : les touches de *La Nuit étoilée* peuvent se placer n'importe où, c'est la loss de contenu qui les organise selon la photo. Les tables à trois filtres du livre, $T_{ijl} = \frac{1}{N}\sum_k F_{ik} F_{jk} F_{lk}$, sont des statistiques d'ordre 3, elles aussi invariantes par permutation des positions (même raisonnement : une somme sur $k$) ; elles captureraient des asymétries que l'ordre 2 ignore, pour un coût en $C^3$.
**Erreurs fréquentes** : écrire $(FP)^\top = F^\top P^\top$ (l'ordre s'inverse) ; croire que la matrice de Gram est invariante par toute transformation des cartes (une permutation d'un seul canal la change) ; confondre $G_{ii}$ et la variance.
**Variante** : un retournement de l'image entière (miroir) ne permute pas exactement les positions des cartes d'activation, car les filtres ne sont pas symétriques : la matrice de Gram d'une photo et de son miroir est **proche**, mais pas égale ; seule une permutation des positions **des cartes elles-mêmes** la laisse exactement inchangée.

### Ex 28.4 — Losses de contenu, de style et de variation totale sur un mini-exemple ✏️
**Réponse** : a) **5** ; b) **1,25** ; c) **0,5** ; d) **0,75** ; e) **14** ; f) **0,8764** ; g) **12**.
**Démarche** : a) Écarts $(0, 2, -1, 0)$, carrés $0 + 4 + 1 + 0$. b) $5 / 4$. c) $F(\mathbf{x}) F(\mathbf{x})^\top = \begin{pmatrix} 2 & 2 \\ 2 & 4 \end{pmatrix}$, divisé par $CHW = 2 \times 2 = 4$ : $G(\mathbf{x}) = \begin{pmatrix} 0{,}5 & 0{,}5 \\ 0{,}5 & 1 \end{pmatrix}$. d) $G(\mathbf{s}) = \frac{1}{4}\begin{pmatrix} 4 & 0 \\ 0 & 4 \end{pmatrix} = I$ ; l'écart $\begin{pmatrix} -0{,}5 & 0{,}5 \\ 0{,}5 & 0 \end{pmatrix}$ a pour carré de norme $0{,}25 \times 3 = 0{,}75$. e) Écarts verticaux $(1, 0, 1)$ puis $(-1, 0, 2)$ : carrés $1 + 0 + 1 + 1 + 0 + 4 = 7$ ; horizontaux $(1, -1)$, $(0, 0)$, $(1, 2)$ : $1 + 1 + 0 + 0 + 1 + 4 = 7$ ; total 14 (avec des valeurs absolues, on trouverait 10 : l'écart de 2 fait la différence). f) $0{,}025 \times 5 + 0{,}75 + 10^{-4} \times 14 = 0{,}125 + 0{,}75 + 0{,}0014$. g) Sans la division par 4, l'écart est 4 fois plus grand, son carré 16 fois : $16 \times 0{,}75 = 12$. Dans ta copie : le **style** domine (0,75 sur 0,876, 86 %), alors même que l'écart de contenu (5) est bien plus grand que celui du style ; c'est le rôle de $\alpha$ et $\beta$. Pour une couche de 512 canaux de 28 × 28, $CHW = 401\,408$ : oublier la normalisation multiplie la loss de style par $401\,408^2 \approx 1{,}6 \times 10^{11}$, et le réglage $\alpha = 0{,}025$, $\beta = 1$ revient à ignorer complètement le contenu.
**Erreurs fréquentes** : additionner les écarts sans les élever au carré ; oublier une direction dans la variation totale ; prendre la racine de la somme (la norme, au lieu de son carré) ; normaliser la matrice de Gram de la référence et pas celle de l'image générée.
**Variante** : avec $F(\mathbf{x}) = (0, 1, 1, 1)$ pour le contenu, $L_c = 1 + 0 + 1 + 0 = 2$ (et `F.mse_loss` vaut 0,5) ; avec $F(\mathbf{x}) = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$ pour le style, $G(\mathbf{x}) = \begin{pmatrix} 0{,}25 & 0{,}25 \\ 0{,}25 & 0{,}5 \end{pmatrix}$ et $L_s = 0{,}5625 + 2 \times 0{,}0625 + 0{,}25 = 0{,}9375$.

### Ex 28.5 — Le gradient de la loss de style ∂
**Réponse** : $\frac{\partial}{\partial F} \lVert F F^\top - S \rVert_F^2 = 4\,(F F^\top - S)\,F$ ; avec la normalisation, $\frac{\partial L_s}{\partial F} = \frac{4}{N'}\bigl(G(\mathbf{x}) - G(\mathbf{s})\bigr) F$ ; sur l'Ex 28.4, $\begin{pmatrix} -0{,}5 & 0{,}5 \\ 0{,}5 & 0{,}5 \end{pmatrix}$.
**Démarche** :
1. $\frac{\partial F_{ik}}{\partial F_{ab}} = \delta_{ia}\delta_{kb}$, donc $\frac{\partial}{\partial F_{ab}} \sum_k F_{ik} F_{jk} = \sum_k (\delta_{ia}\delta_{kb} F_{jk} + F_{ik}\delta_{ja}\delta_{kb}) = \delta_{ia} F_{jb} + \delta_{ja} F_{ib}$.
2. $E = \sum_{i,j} D_{ij}^2$ donne $\frac{\partial E}{\partial F_{ab}} = \sum_{i,j} 2 D_{ij}\,(\delta_{ia} F_{jb} + \delta_{ja} F_{ib}) = 2 \sum_j D_{aj} F_{jb} + 2 \sum_i D_{ia} F_{ib} = 2(DF)_{ab} + 2(D^\top F)_{ab}$. $F F^\top$ est symétrique ; si $S$ l'est aussi, $D^\top = D$, et l'on obtient $4(DF)_{ab}$. C'est le seul endroit où la symétrie de $S$ sert (une matrice de Gram cible l'est toujours).
3. $L_s = \bigl\lVert \frac{1}{N'} F F^\top - G(\mathbf{s}) \bigr\rVert_F^2 = \frac{1}{N'^2} \bigl\lVert F F^\top - N' G(\mathbf{s}) \bigr\rVert_F^2$, donc $\frac{\partial L_s}{\partial F} = \frac{4}{N'^2}\bigl(F F^\top - N' G(\mathbf{s})\bigr) F = \frac{4}{N'}\bigl(G(\mathbf{x}) - G(\mathbf{s})\bigr) F$.
4. $N' = 4$ : $\frac{4}{4} \begin{pmatrix} -0{,}5 & 0{,}5 \\ 0{,}5 & 0 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} -0{,}5 & 0{,}5 \\ 0{,}5 & 0{,}5 \end{pmatrix}$ (autograd donne la même matrice).
5. Si $F = 0$, le gradient est nul : la loss de style ne peut rien faire naître d'une couche entièrement éteinte. Derrière une ReLU, une activation nulle vient d'une pré-activation négative ou nulle, où la dérivée de la ReLU vaut 0 : le gradient de ce neurone ne redescend pas vers les pixels. Une image uniforme donne des cartes constantes, et les détecteurs de bords et de textures, qui répondent à des variations, y restent éteints ou presque ; un bruit, ou la photo de contenu, allume dès le départ beaucoup de filtres, avec des valeurs variées. (La loss de contenu, elle, garde un gradient : l'argument porte sur le style.)
6. Par la règle de la chaîne à travers le réseau gelé : $\frac{\partial L_s}{\partial \mathbf{x}} = \sum_\ell \bigl(\frac{\partial F^\ell}{\partial \mathbf{x}}\bigr)^\top \frac{\partial L_s}{\partial F^\ell}$, que la rétropropagation calcule couche par couche ; dans le notebook, c'est `loss.backward()` qui le fait, et le résultat arrive dans `x.grad`.
**Erreurs fréquentes** : oublier l'un des deux termes de la dérivée du produit (on trouve $2DF$ au lieu de $4DF$) ; oublier le facteur $\frac{1}{N'}$ qui sort de la normalisation (le gradient est alors $N'$ fois trop grand) ; confondre $DF$ et $FD$ (les tailles ne collent pas : $D$ est $C \times C$, $F$ est $C \times N$).
**Variante** : avec $F(\mathbf{x}) = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$ et la même cible $G(\mathbf{s}) = I$, $D = \begin{pmatrix} -0{,}75 & 0{,}25 \\ 0{,}25 & -0{,}5 \end{pmatrix}$ et le gradient vaut $DF = \begin{pmatrix} -0{,}5 & 0{,}25 \\ -0{,}25 & -0{,}5 \end{pmatrix}$.

### Ex 28.6 — Combien coûte un transfert de style sur CPU ? 🧮
**Réponse** : environ **4,6 minutes** pour 300 pas de VGG16 à 224 × 224, **24 minutes** à 512 × 512, **1 seconde** de calcul pur pour 150 pas de SqueezeNet à 128 × 128 ; les convolutions de VGG16 n'utilisent que **11 %** de ses poids.
**Démarche** :
1. $300 \times 3 \times 15{,}5 \cdot 10^9 \approx 1{,}4 \cdot 10^{13}$ MAC ; à $5 \cdot 10^{10}$ MAC/s, 280 s, soit environ 4,6 minutes.
2. Chaque couche fait le même calcul en chaque pixel de ses cartes, et le nombre de pixels croît comme le carré du côté : $(512 / 224)^2 \approx 5{,}2$, d'où environ 1 450 s, soit 24 minutes.
3. $0{,}35 \cdot 10^9 \times (128 / 224)^2 \approx 1{,}1 \cdot 10^8$ MAC par passe ; $150 \times 3$ passes : $5 \cdot 10^{10}$ MAC, environ 1 s. Le temps mesuré sera plus long : chaque opération de PyTorch a un coût fixe (lancer un noyau, allouer de la mémoire), les petits tenseurs n'atteignent pas la vitesse de pointe du CPU, et les hooks, les matrices de Gram et les mises à jour d'Adam s'ajoutent ; 50 milliards de MAC par seconde est d'ailleurs optimiste pour deux cœurs.
4. $14{,}7 / 138{,}3 \approx 11\,\%$ ; les seuls poids des convolutions pèsent $14{,}7 \cdot 10^6 \times 4 \approx 59$ Mo (56 Mio). On télécharge 553 Mo pour en utiliser 59 : c'est une des raisons de préférer SqueezeNet en mode rapide.
5. $13{,}5 \cdot 10^6 \times 4 \approx 54$ Mo pour les seules sorties des convolutions (davantage avec celles des ReLU et des poolings). La rétropropagation en a besoin : pour dériver une ReLU, il faut savoir où l'entrée était positive ; pour un max-pooling, quel élément était le max ; pour une convolution, le gradient par rapport à son entrée utilise ses poids, et celui par rapport aux poids utilise son entrée. À 512 × 512, il faut multiplier par 5,2.
6. Un pas de Gatys coûte environ 3 passes avant, et il en faut 300 : environ 900 passes, contre une seule. Le réseau rapide est donc environ **900 fois** plus rapide, trois ordres de grandeur, à coût de réseau égal. Le prix : un entraînement préalable, par style, sur des dizaines de milliers de photos, avec les mêmes losses perceptuelles.
**Erreurs fréquentes** : oublier la rétropropagation (compter une passe par pas) ; faire croître le coût comme le côté au lieu de la surface ; oublier que les couches denses de VGG16 ne servent pas.
**Variante** : à 384 × 384, $(384/224)^2 \approx 2{,}94$ : environ 820 s, 14 minutes.

### Ex 28.7 — Vocabulaire de mots et taille de la couche de sortie d'un LSTM 🧮
**Réponse** : environ **2,6 millions** de paramètres, dont **51 %** dans la couche de sortie (1 328 700) ; environ 0,16 mot d'entraînement par paramètre ; un modèle uniforme a une perplexité de 10 300.
**Démarche** :
1. $(128 + 1) \times 10\,300 = 1\,328\,700$.
2. Embedding : $10\,300 \times 100 = 1\,030\,000$. Première couche de LSTM ($d = 100$) : $4 \times 128 \times 228 + 8 \times 128 = 117\,760$. Seconde ($d = 128$) : $4 \times 128 \times 256 + 1\,024 = 132\,096$.
3. Total : $2\,608\,556$ ; la couche de sortie en fait $\frac{1\,328\,700}{2\,608\,556} \approx 51\,\%$, l'embedding 39 %, les deux LSTM moins de 10 %. Le vocabulaire, pas la « mémoire » du réseau, fait la taille du modèle ; d'où l'intérêt de réduire le vocabulaire, ou de partager les poids de l'embedding et de la sortie (*weight tying*) quand leurs dimensions coïncident.
4. $\frac{427\,000}{2\,608\,556} \approx 0{,}16$ mot par paramètre : le modèle a largement de quoi apprendre le texte par cœur. Il faut une validation tenue à l'écart, du dropout, et se méfier des phrases « presque sensées », qui peuvent être des morceaux recopiés.
5. $(128 + 1) \times 5\,001 = 645\,129$ paramètres (5 001 sorties), deux fois moins. Couverture : $\frac{\ln 5\,000 + 0{,}577}{\ln 10\,300 + 0{,}577} = \frac{9{,}094}{9{,}817} \approx 0{,}93$ : la moitié des mots du vocabulaire couvre 93 % du texte ; les 7 % restants deviennent `<unk>`.
6. $\ln 10\,300 \approx 9{,}24$ nats ; perplexité $e^{9{,}24} = 10\,300$. Une perplexité de 150 : le modèle hésite, en moyenne, comme s'il choisissait uniformément entre 150 mots.
**Erreurs fréquentes** : oublier les biais de la couche de sortie ; compter quatre portes sans le facteur 4 du LSTM ; donner la perplexité en bits ou en logarithme décimal.
**Variante** : avec un vocabulaire de 20 000 mots, la couche de sortie aurait $129 \times 20\,000 = 2\,580\,000$ paramètres, plus que tout le reste du modèle.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 28.8 — Pourquoi la matrice de Gram capture le style, en cinq lignes 🗣️
**Réponse modèle** : (1) Imagine une centaine de petits détecteurs qui parcourent un tableau : l'un s'allume sur une touche jaune, l'autre sur une courbe, un autre sur un trait sombre. (2) Pour chaque paire de détecteurs, on compte combien de fois ils s'allument **au même endroit** : « touche jaune » et « courbe » ensemble, souvent ; « trait sombre » et « aplat lisse », jamais. (3) Ce tableau de comptes est la matrice de Gram. (4) Il oublie volontairement **où** se trouvent les touches : on ne sait plus s'il y a une église à gauche ou un arbre à droite. (5) C'est justement ce qu'on appelle un style : une manière de peindre qui se retrouve partout sur la toile, quelle que soit la scène.
**Ce qui compte** : des détecteurs de motifs (les filtres) ; des co-occurrences au même endroit, comptées sur toute l'image ; l'oubli de la position, présenté comme une qualité. Une explication qui parle de « couleurs moyennes » seulement rate les co-occurrences.

### Ex 28.9 — Imiter le style d'un artiste : droit d'auteur, domaine public, consentement ⚖️
**Réponse modèle** :
1. En 2026, dans l'Union européenne, sont dans le domaine public : Van Gogh (depuis 1961), Turner (depuis longtemps), Monet (depuis 1997), Munch et Kandinsky (depuis le 1ᵉʳ janvier 2015). Hopper y entrera le 1ᵉʳ janvier 2038, Picasso le 1ᵉʳ janvier 2044. Le tableau de l'auteur du livre, vivant, est protégé : il faut son accord. (Certains pays ont des règles particulières, et la durée diffère hors de l'Union : aux États-Unis, elle dépend aussi de la date de publication.)
2. Non : selon l'article 14 de la directive 2019/790, quand la durée de protection d'une œuvre d'art visuel a expiré, une reproduction de cette œuvre n'est pas protégée, sauf si elle est elle-même une création originale. Une photo fidèle de *La Nuit étoilée* est donc libre ; vérifier quand même la licence affichée et la créditer.
3. Non : le droit d'auteur protège des **œuvres**, une expression précise, pas un style, une technique ou une idée. Imiter un artiste vivant pose pourtant d'autres questions : la concurrence (des images « à la manière de » vendues moins cher, qui peuvent priver l'artiste de commandes), la confusion du public, l'usage de son nom, son droit moral, et le respect de sa volonté s'il refuse que l'on s'inspire de son travail par une machine. Et les œuvres elles-mêmes, utilisées comme références ou pour entraîner un modèle, sont copiées.
4. L'article 4 autorise les reproductions et extractions d'œuvres licitement accessibles pour la fouille de textes et de données, y compris à des fins commerciales, **sauf** si le titulaire s'y est opposé de manière appropriée, par exemple par des procédés lisibles par machine pour les contenus en ligne (l'*opt-out*). L'article 53 du règlement 2024/1689 impose aux fournisseurs de modèles d'IA à usage général (depuis le 2 août 2025) une politique pour respecter le droit d'auteur de l'Union, qui identifie et respecte ces *opt-out*, et la publication d'un résumé suffisamment détaillé des contenus utilisés pour l'entraînement.
5. Pour chaque image : la photo de contenu (titre, auteur, source, licence) ; l'œuvre de style (titre, artiste, date, statut dans le domaine public, source de la reproduction et sa licence) ; les réglages (réseau et version des poids, couches, $\alpha$, $\beta$, $\lambda$, nombre de pas, optimiseur, taille, graine, point de départ) ; la date et les versions des outils ; la licence choisie pour l'image produite ; et la mention qu'il s'agit d'une image générée.
**Ce qui compte** : la règle des 70 ans et son calcul au 1ᵉʳ janvier ; la distinction œuvre, style, reproduction ; le mécanisme de l'*opt-out* et les obligations des fournisseurs ; un réflexe de documentation. Dire « tout ce qui est sur Internet est libre » est l'erreur à ne pas faire.

### Ex 28.10 — Lire Gatys, Ecker et Bethge (2015), « A Neural Algorithm of Artistic Style » 📄
**Réponse modèle** :
1. VGG-19, dont ils gardent les 16 convolutions et les 5 poolings, sans les couches denses. Ils remplacent le max-pooling par un pooling **moyen**, qui améliore le passage du gradient et donne des images un peu plus agréables selon eux.
2. Le contenu : `conv4_2`. Le style : `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1` et `conv5_1`, chacune de poids $w_\ell = \frac{1}{5}$ (les autres couches ont un poids nul).
3. $L_c = \frac{1}{2}\sum_{i,j} \bigl(F^\ell_{ij} - P^\ell_{ij}\bigr)^2$, où $P^\ell$ est l'activation de la photo ; $G^\ell_{ij} = \sum_k F^\ell_{ik} F^\ell_{jk}$, **non normalisée** ; $E_\ell = \frac{1}{4 N_\ell^2 M_\ell^2}\sum_{i,j}\bigl(G^\ell_{ij} - A^\ell_{ij}\bigr)^2$, où $A^\ell$ est la matrice de Gram du tableau, et $L_s = \sum_\ell w_\ell E_\ell$. $N_\ell$ est le nombre de filtres ($C$ dans la fiche), $M_\ell$ le nombre de positions ($HW$). La fiche divise la matrice de Gram par $CHW = N_\ell M_\ell$ ; au carré, cela fait $\frac{1}{N_\ell^2 M_\ell^2}$ : la même normalisation, au facteur $\frac{1}{4}$ près.
4. $\alpha / \beta = 10^{-3}$ ou $10^{-4}$ pour leurs images. La figure 3 fait varier ce rapport (de $10^{-5}$ à $10^{-2}$, en colonnes) et le nombre de couches de style (en lignes) : avec un petit rapport, on ne voit presque que la texture du tableau ; avec un grand, la photo domine et le style n'est plus qu'une teinte.
5. D'un bruit blanc, par descente de gradient sur les pixels (la version CVPR 2016 précise L-BFGS). Le livre part aussi d'un bruit, mais il prend VGG16, une couche de contenu du premier bloc, toutes les couches de style, et réduit les images à 224 × 224.
6. Les reconstructions du contenu, couche par couche, sont presque parfaites dans les premières couches et ne gardent que l'organisation de la scène dans les dernières ; celles du style, avec un nombre croissant de couches, donnent des textures de plus en plus grandes et cohérentes. Ce sont les fig. 28.19 (contenu) et 28.24 (style cumulé) du livre.
7. L'affirmation que le contenu et le style sont **séparables** dans les représentations d'un CNN : les auteurs reconnaissent eux-mêmes qu'on ne peut pas les démêler complètement (on cherche un compromis par $\alpha / \beta$), la « séparation » dépend du choix des couches, et la matrice de Gram mesure une texture, pas tout ce qu'un historien de l'art appelle style (composition, sujet, période). On peut aussi discuter les rapprochements avec la vision biologique, que l'article suggère sans les tester.
**Ce qui compte** : retrouver les choix exacts (réseau, couches, normalisation, rapport) et voir que le livre en change presque tous ; relier les formules aux exercices ✏️ 28.4 et ∂ 28.5.

<a id="entretien"></a>

## 💼 Entretien

### 28.E1 — Comment visualiser ce qu'a appris un CNN ? 💼
**Réponse modèle en 60 secondes** : « Il y a trois familles. D'abord regarder directement : les poids de la première couche, qui ressemblent à des détecteurs de bords et de couleurs, et les cartes d'activation d'une image réelle, récupérées avec des hooks. Ensuite la maximisation d'activation : on gèle le réseau et on fait une montée de gradient sur l'image d'entrée pour trouver ce qui excite un filtre ou une classe ; sans régularisation, on obtient du bruit à haute fréquence, donc on ajoute des décalages aléatoires, du flou, plusieurs échelles. Enfin les cartes d'attribution, pour expliquer une décision : la saillance par le gradient, Grad-CAM, qui pondère les cartes d'une couche profonde par le gradient du score. Je compléterais par les images du dataset qui activent le plus un filtre. Les limites : une belle image n'est pas une explication, un neurone répond souvent à plusieurs concepts, et certaines cartes d'attribution changent à peine quand on randomise les poids : il faut les tester. »
**Ce qui compte** : distinguer comprendre un filtre (maximisation) et expliquer une prédiction (attribution) ; citer la régularisation ; être honnête sur les limites.
**Relances possibles** :
- « Pourquoi la montée de gradient brute donne-t-elle du bruit ? » → Le gradient trouve plus facilement des motifs de quelques pixels, de haute fréquence, qui excitent fortement le filtre ; rien ne pousse vers une image naturelle. On régularise (décalages, flou, octaves, variation totale) ou on paramètre l'image autrement.
- « Comment savoir si une carte d'attribution est fiable ? » → Des tests de cohérence : elle doit changer quand on randomise les poids ou les labels (Adebayo et coll., 2018), et masquer les zones désignées doit faire chuter le score plus que masquer des zones au hasard.
- « À quoi cela sert-il en pratique ? » → À déboguer : repérer qu'un classifieur regarde un filigrane, un fond ou un artefact du dataset plutôt que l'objet.

### 28.E2 — Expliquer le transfert de style neuronal et la matrice de Gram 💼
**Réponse modèle en 60 secondes** : « On part d'une photo, le contenu, et d'un tableau, le style, et on cherche une troisième image. On la passe dans un CNN pré-entraîné gelé, comme VGG. La loss de contenu compare ses activations à une couche profonde avec celles de la photo : elle impose la structure. La loss de style compare, sur plusieurs couches, ses matrices de Gram à celles du tableau. La matrice de Gram d'une couche, c'est le produit des cartes mises à plat par leur transposée : pour chaque paire de filtres, à quel point ils s'activent ensemble. Elle oublie les positions, donc elle décrit la texture, les touches, la palette, pas la scène. On ajoute une variation totale pour lisser, et on optimise les pixels avec L-BFGS ou Adam. On peut la voir comme une distance entre les distributions de features des deux images. »
**Ce qui compte** : les trois images, ce qui est optimisé (les pixels), le réseau gelé, la définition de la matrice de Gram et pourquoi elle oublie la position, la pondération des losses.
**Relances possibles** :
- « Pourquoi c'est lent, et comment l'accélérer ? » → Des centaines de passes avant et arrière par image ; on entraîne plutôt une fois un réseau qui transforme une photo en une passe (Johnson et coll., 2016), ou un réseau à style arbitraire comme AdaIN (2017).
- « Comment régler les poids ? » → En regardant la valeur de chaque terme pondéré au début, puis en ajustant le rapport contenu/style à l'œil ; les valeurs d'un autre code ne se recopient pas, car les normalisations diffèrent.
- « Pourquoi la matrice de Gram ? » → Elle mesure des statistiques d'ordre 2 des features, invariantes à la position ; Li et coll. (2017) montrent qu'égaler les matrices de Gram revient à minimiser une MMD à noyau polynomial entre les deux distributions de features ; aligner les moyennes et les écarts-types des canaux marche aussi.

### 28.E3 — Quelles stratégies de décodage pour générer du texte ? 💼
**Réponse modèle en 60 secondes** : « Le modèle donne une distribution sur le token suivant ; le décodage choisit. Le glouton prend le plus probable : déterministe, mais il tourne vite en boucle sur un texte libre. La recherche en faisceau garde les k meilleures suites : bien pour la traduction ou le résumé, où il y a une bonne réponse, mais fade et répétitive pour un texte libre. Pour un texte libre, on échantillonne : la température règle la prise de risque, en dessous de 1 plus sûr, au-dessus plus varié ; le top-k ne tire que parmi les k plus probables ; le top-p, parmi le plus petit ensemble dont la probabilité cumulée atteint p, ce qui s'adapte à la confiance du modèle. On ajoute parfois une pénalité de répétition. Pour une tâche à réponse unique, comme extraire un champ, je mets la température près de zéro. »
**Ce qui compte** : glouton et faisceau contre échantillonnage ; la température ; la différence entre top-k (taille fixe) et top-p (taille adaptative) ; le choix selon la tâche.
**Relances possibles** :
- « Pourquoi le texte le plus probable n'est-il pas le meilleur ? » → Le texte humain n'est presque jamais la suite la plus probable selon le modèle ; maximiser la probabilité mène à des phrases génériques et à des boucles, où chaque répétition rend la suivante plus probable (Holtzman et coll.).
- « Quel est le risque d'une température élevée ? » → Des tokens peu probables, donc des incohérences et des erreurs factuelles ; le top-p limite ce risque en coupant la queue de la distribution.
- « Comment évaluer un modèle de langue indépendamment du décodage ? » → Par sa perplexité sur un texte de validation, $e$ puissance la cross-entropy moyenne par token ; puis, pour la génération, par des évaluations humaines ou des mesures de diversité et de répétition.

### 28.E4 — Pourquoi la diffusion a-t-elle remplacé deep dream et le transfert de style de Gatys ? 💼
**Réponse modèle en 60 secondes** : « La méthode de Gatys optimise chaque image pendant des centaines de pas, ne connaît du style que des statistiques de texture, et ne comprend pas la scène : elle peut mettre des touches de ciel sur un visage. Aujourd'hui, je partirais d'un modèle de diffusion latente pré-entraîné : on décrit le style en texte, on garde la structure de la photo avec un ControlNet sur les contours ou la profondeur, ou on part de la photo bruitée en image-à-image ; pour un style précis, un IP-Adapter avec l'image de référence, ou un LoRA entraîné sur quelques œuvres. C'est plus riche, contrôlable et, après distillation, rapide. Si l'outil doit tourner en temps réel sur un téléphone, un petit réseau feed-forward de transfert rapide, comme AdaIN, reste une option légère. Et je vérifierais les droits sur les œuvres de référence. »
**Ce qui compte** : les limites de Gatys (coût par image, texture sans sémantique) ; les briques modernes (diffusion latente, ControlNet, IP-Adapter, LoRA) ; le compromis coût/qualité ; la question des droits.
**Relances possibles** :
- « Qu'apporte ControlNet ? » → Un conditionnement par une image structurelle (contours, profondeur, pose) ajouté à un modèle pré-entraîné gelé : la composition de la photo est gardée, le reste vient du texte ou du style.
- « Quelle différence entre LoRA et IP-Adapter ? » → LoRA ajuste le modèle (des matrices de rang faible ajoutées à ses poids) sur quelques images d'un style : un petit entraînement par style. IP-Adapter injecte une image de référence comme condition, sans réentraîner le modèle pour chaque style.
- « Que reste-t-il de Gatys aujourd'hui ? » → Les losses perceptuelles, qui servent à entraîner des autoencodeurs d'images et la super-résolution, et l'idée de résumer un style par des statistiques de canaux, qu'on retrouve dans les normalisations adaptatives (AdaIN, StyleGAN).

<a id="notebook"></a>

## Exercices du notebook

Les solutions des exercices du notebook (28.11 à 28.29) seront ajoutées à la prochaine session de génération, avec les parties A à D du notebook.
