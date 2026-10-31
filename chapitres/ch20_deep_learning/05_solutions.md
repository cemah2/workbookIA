# 20 · Deep learning et premiers pas en PyTorch — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels, des exercices ✏️ 20.1 à 20.4, 20.6 et 20.7 et de l'estimation 🧮 20.9 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 20.Q1 — Deep learning et machine learning : qui contient qui ?
a) **Vrai** : pour la plupart des auteurs, le machine learning regroupe toutes les méthodes qui apprennent à partir de données, réseaux de neurones compris. b) **Faux** : d'autres réservent « machine learning » aux méthodes sans réseaux (arbres, SVM, kNN), qu'ils opposent au deep learning. c) **C** : une forêt aléatoire est un ensemble d'arbres de décision (ch. 14), pas une pile de couches de neurones. d) **B** : chaque couche reçoit les sorties de la précédente, et peut donc combiner des motifs déjà détectés en motifs plus grands.
**À retenir** : le deep learning, c'est une pile de couches qui construit une hiérarchie de features ; vérifie toujours dans quel sens un texte emploie « machine learning ».

### 20.Q2 — Tenseurs : nombre de dimensions, taille, forme
a) **4**. b) **75 264** : $32 \times 3 \times 28 \times 28$. c) **« (3, 28, 28) »** : `x[0]` retire la dimension des exemples. d) **Vrai** : un tenseur est « plein », sans lignes de longueurs différentes. e) **B** : `torch.float32`, le type par défaut des réels dans PyTorch (NumPy utilise `float64`).
**À retenir** : forme = tailles, ndim = longueur de la forme, nombre d'éléments = produit des tailles ; dans PyTorch, la première dimension est celle des exemples, et le type par défaut est `float32`.

### 20.Q3 — Couche d'entrée, couche de sortie : laquelle contient des neurones ?
a) **Faux** : la couche d'entrée n'est qu'une mémoire où l'on place les données ; PyTorch n'en a même pas d'objet. b) **Vrai** : ce sont ses neurones qui produisent les scores. c) **2** : les couches de 256 et de 128 neurones ; l'entrée (784) n'est pas une couche de neurones, la sortie (10) n'est pas cachée. d) **B** : le livre dessine les entrées en bas, les sorties en haut.
**À retenir** : on compte les couches d'un réseau sans la couche d'entrée ; « plus haute » veut dire plus proche de la sortie.

### 20.Q4 — Couche dense et couche d'activation séparée
a) **12** : un poids par connexion, $4 \times 3$. b) **15** : plus un biais par neurone. c) **Vrai** : l'activation appliquée dans la couche ou dans une couche à part donne le même calcul. d) **B** : les logits ; `CrossEntropyLoss` applique elle-même `log_softmax`.
**À retenir** : `nn.Linear` n'a pas d'activation : on écrit `nn.Linear(...)`, puis `nn.ReLU()` ; la dernière couche rend des logits.

### 20.Q5 — Vrai ou faux : le dropout tire ses neurones une fois par epoch
a) **Faux** : PyTorch tire un nouveau masque à chaque passe avant, indépendamment pour chaque exemple ; c'est le livre qui décrit un tirage par epoch. b) **500** : un masque par mini-batch (de la taille du mini-batch entier). c) **1,25** : $1/(1 - 0{,}2)$. d) **C** : en évaluation, `nn.Dropout` est l'identité.
**À retenir** : dropout inversé = un nouveau masque à chaque passe, un facteur $1/(1-p)$ en entraînement, rien en évaluation.

### 20.Q6 — Vrai ou faux : la batchnorm n'a aucun paramètre
a) **Faux** : $\gamma$ et $\beta$ sont appris par la rétropropagation. b) **128** : 64 valeurs de $\gamma$ et 64 de $\beta$. c) **128** : 64 moyennes et 64 variances glissantes (PyTorch garde aussi un compteur, `num_batches_tracked`). d) **B** : les statistiques glissantes, pour qu'une prédiction ne dépende pas des autres exemples du mini-batch.
**À retenir** : une batchnorm a des paramètres appris ($\gamma$, $\beta$), des buffers (les statistiques glissantes) et deux réglages (`momentum`, `eps`).

### 20.Q7 — Convolution et pooling en un coup d'œil
a) **20** : $(3^2 \times 1 + 1) \times 2$. b) **256**. c) **10**. d) **Vrai** : le filtre déborde d'un pixel de chaque côté, et la bordure de zéros comble exactement ce débord.
**À retenir** : le nombre de paramètres d'une convolution ne dépend pas de la taille de l'image ; un pooling $k \times k$ divise la largeur et la hauteur par $k$.

### 20.Q8 — Récurrence n'est pas récursion
a) **Faux** : une cellule récurrente répète la même opération à chaque pas d'une séquence en gardant un état ; une fonction récursive s'appelle elle-même. b) **B** : le mot suivant dépend des mots précédents. c) **Vrai** : une couche dense n'a pas d'état : la même entrée donne toujours la même sortie.
**À retenir** : une couche récurrente a une mémoire (un état mis à jour à chaque élément de la séquence) ; une couche dense n'en a pas.

### 20.Q9 — Aplatir, remodeler, rogner, rembourrer, suréchantillonner, bruiter
a) **10**. b) **14**. c) **2**. d) **D** : aucune de ces trois couches n'apprend. e) **Faux** : comme le dropout, le bruit n'agit qu'en entraînement.
**À retenir** : les couches utilitaires changent la forme ou les valeurs sans rien apprendre ; elles ne changent pas le nombre de paramètres d'un réseau.

### 20.Q10 — VGG16 : ce qu'il reconnaît et ce qu'il ignore
a) **13**. b) **3**. c) **1 000**. d) **B** : 92,7 % est une accuracy top-5 ; le top-1 des poids de torchvision est d'environ 71,6 %. e) **Faux** : sa couche de sortie a 1 000 neurones, un par classe ; il ne peut répondre qu'avec l'un de ces 1 000 noms.
**À retenir** : un classifieur ne connaît que ses classes ; une accuracy annoncée n'a de sens qu'avec sa définition (top-1 ou top-5).

### 20.Q11 — Construire un réseau profond : par où commencer ?
a) **B**. b) **C** : le learning rate. c) **Faux** : on s'inspire d'architectures publiées, puis l'on essaie et mesure. d) **B** : un CNN (*convolutional neural network*).
**À retenir** : données d'abord, architecture ensuite, learning rate avant tout autre hyperparamètre ; puis essayer, mesurer, ajuster.

### 20.Q12 — Explication complète ou explication satisfaisante ?
a) **Faux** : une explication peut être complète et inutile, voire inacceptable (les chaussettes bleues du livre). b) **B** : un prêt refusé n'est jamais remboursé ni impayé : la banque n'a pas l'issue des dossiers qu'elle refuse. c) **A** : une boîte noire. d) **Faux** : une accuracy mesurée il y a trois ans ne dit rien du monde d'aujourd'hui ; il faut surveiller la dérive.
**À retenir** : une explication satisfaisante dit quels facteurs ont compté, s'ils sont légitimes, et ce qu'on pourrait changer ; une décision apprise sur le passé ne voit ni la dérive ni ce qu'elle n'observe pas.

<a id="rappels"></a>

## 🔁 Rappels

### 20.R1 — Ch. 19 : que fait weight_decay dans AdamW ?
a) **1,998** : $2 \times (1 - 0{,}001)$ ; au premier pas, $m = v = 0$ et le gradient nul donnent un pas d'Adam nul (plus tard, la moyenne $m$ des gradients passés ferait encore bouger le poids). b) **Faux** : AdamW découple la décroissance, qui n'entre ni dans $m$ ni dans $v$ ; c'est `Adam(weight_decay=...)` qui l'ajoute au gradient. c) **0,01**.
**À retenir** : AdamW rétrécit chaque poids d'un facteur $1 - \eta\lambda$ à chaque pas, même quand son gradient est nul.

### 20.R2 — Ch. 17 : pourquoi calcule-t-on la loss à partir des logits ?
a) **Faux**. b) **−1 000** : $0 - \log(e^{1000} + e^{0}) = -1000 - \log(1 + e^{-1000}) \approx -1000$. c) **1**. d) **B** : en passant par la softmax, $e^{-1000}$ vaut 0 en `float32`, et $\log 0 = -\infty$ ; `log_softmax` calcule directement $z_k - z_{\max} - \log\sum_j e^{z_j - z_{\max}}$, toujours fini.
**À retenir** : la loss prend les logits et fait la softmax elle-même, de façon stable ; une probabilité ne se calcule qu'au moment de l'afficher.

### 20.R3 — Ch. 13 : limiter l'overfitting d'un arbre, et l'équivalent pour un réseau
a) **A** : `max_depth` (ou `min_samples_leaf`, ou l'élagage). b) **B**. c) **Vrai** : il arrête l'entraînement avant que l'erreur de validation ne remonte.
**À retenir** : réduire la capacité ou ajouter du bruit, et arrêter à temps : les mêmes remèdes pour un arbre et pour un réseau, sous d'autres noms.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 20.1 — Suivre la forme d'un tenseur couche après couche ✏️
**Réponses** : a) **150** ; b) **« (10, 15) »** ; c) **« (32, 784) »** ; d) **« (32, 128) »** ; e) **100 480** ; f) **« (32, 3072) »** ; g) **« (25088,) »** ; h) **« (1, 25088) »**.
**Démarche** : a) $10 \times 3 \times 5$. b) On garde 10 et l'on répartit les $150$ éléments : $150/10 = 15$. c) `nn.Flatten()` garde 32 et met à plat $1 \times 28 \times 28 = 784$. d) La couche dense remplace la dernière dimension. e) $784 \times 128 + 128$. f) $3 \times 32 \times 32 = 3\,072$. g) `torch.flatten` met tout à plat : $1 \times 512 \times 7 \times 7 = 25\,088$ valeurs, une seule dimension. h) `nn.Flatten()` garde la dimension des exemples.
**Dans ta copie** : avec deux images, `torch.flatten` donne un vecteur de $50\,176$ valeurs, que la couche lit comme un seul exemple de $50\,176$ features : PyTorch lève `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x50176 and 25088x4096)`. Avec une seule image, l'erreur ne se voit pas : la sortie a la forme `(4096,)` au lieu de `(1, 4096)`, et le bug n'apparaît que plus loin (ou jamais, avant qu'on change la taille du mini-batch).
**Erreurs fréquentes** : écrire `"(25088)"` sans la virgule (en Python, c'est un nombre entre parenthèses, pas un tuple) ; oublier le canal dans le produit ; compter les biais par entrée au lieu de par sortie ; confondre `torch.flatten` et `nn.Flatten`.
**Variante** : un mini-batch d'ImageNet de forme `(64, 3, 224, 224)` donne **« (64, 150528) »** après `nn.Flatten()`.

### Ex 20.2 — Dimensionner la couche de sortie et choisir la loss ✏️
**Réponses** : a) **1** ; b) **A** ; c) **26** ; d) **C** ; e) **B** ; f) **5** ; g) **B** ; h) **C**.
**Démarche** : un prix est une valeur réelle : une sortie, `MSELoss`. Les 26 lettres sont exclusives : 26 logits, `CrossEntropyLoss`. Un seul logit pour spam ou non : `BCEWithLogitsLoss`, qui applique la sigmoïde elle-même. Cinq labels indépendants : cinq logits, une sigmoïde et une cross-entropy binaire par label, donc encore `BCEWithLogitsLoss`. Deux sorties dont une seule est vraie : c'est une classification à deux classes exclusives, `CrossEntropyLoss` (une softmax sur deux scores est équivalente à une sigmoïde sur leur différence).
**Dans ta copie** : des entiers de 0 à 25, de forme `(64,)` et de dtype `torch.int64` (`torch.long`), la forme la plus simple (des vecteurs one-hot `float` de forme `(64, 26)` marcheraient aussi : `CrossEntropyLoss` accepte des probabilités par classe, mais c'est plus lourd) ; les logits ont la forme `(64, 26)`.
**Erreurs fréquentes** : choisir `CrossEntropyLoss` pour des labels non exclusifs (elle force les probabilités à sommer à 1) ; mettre une sigmoïde ou une softmax dans le modèle avant ces losses ; donner à `CrossEntropyLoss` des indices écrits en réels, de forme `(64,)` (`RuntimeError: expected scalar type Long but found Float`), ou des entiers à `BCEWithLogitsLoss`.
**Variante** : prédire à la fois le prix et la surface d'une maison demande **2** sorties et `MSELoss` (**A**) ; reconnaître les 10 chiffres de MNIST, **10** sorties et `CrossEntropyLoss` (**C**).

### Ex 20.3 — Dropout inversé à la main ✏️
**Réponses** : a) **20** ; b) **3** ; c) **1,6667** ; d) **−1,6667** ; e) **10** ; f) **8,5** ; g) **5** ; h) **Vrai**.
**Démarche** : a) Chaque neurone est coupé avec la probabilité 0,2 : en moyenne $0{,}2 \times 100$ ; le nombre exact varie d'une passe à l'autre (loi binomiale, d'écart-type 4). b) $u_i \ge 0{,}4$ pour $u_2 = 0{,}7$, $u_4 = 0{,}9$ et $u_5 = 0{,}5$ : trois neurones gardés. c) $1/0{,}6$. d) Le neurone 2 est gardé : $-1 \times 1{,}6667$. e) $\tilde{\mathbf{a}} = (0 ;\ -1{,}6667 ;\ 0 ;\ 5 ;\ 6{,}6667)$, de somme $(-1 + 3 + 4)/0{,}6 = 10$. f) Par linéarité, $\mathbb{E}\big[\sum_i\tilde a_i\big] = \sum_i a_i = 8{,}5$ : le tirage de e) donne 10, un autre donnerait autre chose, la moyenne est 8,5. g) $\partial L/\partial a_i = (m_i/0{,}6) \times 1$ : $(0 ;\ 1{,}6667 ;\ 0 ;\ 1{,}6667 ;\ 1{,}6667)$, de somme 5. h) En évaluation, le dropout inversé est l'identité.
**Dans ta copie** : avec $p = 0{,}6$, on garde les neurones de $u_i \ge 0{,}6$ : le 2 ($u = 0{,}7$) et le 4 ($u = 0{,}9$), multipliés par $1/0{,}4 = 2{,}5$.
**Erreurs fréquentes** : garder les neurones de $u_i < p$ (la convention inverse : elle est aussi juste en loi, mais ce n'est pas celle de ta librairie) ; multiplier par $1 - p$ au lieu de diviser ; oublier le masque dans la passe arrière, ou l'appliquer sans le facteur.
**Variante** : avec $p = 0{,}5$ et les mêmes $\mathbf{u}$, on garde les neurones 2, 4 et 5, et $\tilde{\mathbf{a}} = (0, -2, 0, 6, 8)$, de somme **12**.

### Ex 20.4 — Batchnorm à la main sur un mini-batch de quatre exemples ✏️
**Réponses** : a) **4** ; b) **10** ; c) **5** ; d) **8** ; e) **−1,3416** ; f) **1,7071** ; g) **Vrai** ; h) **0,4** ; i) **1** ; j) **1,5667** ; k) **5,752**.
**Démarche** : c) Écarts $(-3, -1, 1, 3)$, carrés $(9, 1, 1, 9)$, moyenne $20/4 = 5$. d) Écarts $(0, 4, -4, 0)$, carrés de somme 32, moyenne 8. e) $-3/\sqrt5$. f) $\hat x = 4/\sqrt8 = \sqrt2 \approx 1{,}4142$, puis $0{,}5 \times 1{,}4142 + 1$. g) La moyenne des $\hat x$ d'une feature est nulle, donc celle des $y$ vaut $\gamma \times 0 + \beta$. h) $0{,}9 \times 0 + 0{,}1 \times 4$. i) $0{,}9 \times 0 + 0{,}1 \times 10$. j) Variance non biaisée $20/3 \approx 6{,}6667$, puis $0{,}9 \times 1 + 0{,}1 \times 6{,}6667$. k) $2 \times (4 - 0{,}4)/\sqrt{1{,}5667} + 0 = 7{,}2/1{,}2517$.
**Dans ta copie** : après un seul mini-batch, les statistiques glissantes n'ont pris que 10 % des statistiques du mini-batch : elles sont encore proches de leurs valeurs initiales (0 et 1). Un exemple égal à la moyenne du mini-batch (4) est donc vu, en évaluation, comme très au-dessus de la moyenne (0,4). Il faut de nombreux pas d'entraînement pour que les statistiques glissantes se rapprochent de celles des données : avec $\rho = 0{,}1$, la mémoire est d'environ 10 mini-batches (ch. 19), et le poids des valeurs initiales vaut $0{,}9^t$, moins de 1 % après 44 pas. C'est pourquoi un modèle évalué après très peu de pas, ou dont la distribution des entrées vient de changer, se comporte mal en `eval()`.
**Erreurs fréquentes** : normaliser par ligne (par exemple) au lieu de par colonne (par feature) ; utiliser la variance non biaisée pour normaliser, ou la biaisée pour la statistique glissante ; inverser $\rho$ et $1 - \rho$ (la convention de Keras) ; partir d'une variance glissante nulle au lieu de 1 ; utiliser en évaluation les statistiques du mini-batch.
**Variante** : pour une seule feature de valeurs $(2, 2, 8, 8)$, la moyenne vaut 5, la variance biaisée **9**, $\hat x$ de la dernière valeur $3/3 =$ **1**, la moyenne glissante après un pas **0,5** et la variance glissante $0{,}9 + 0{,}1 \times 12 =$ **2,1**.

### Ex 20.5 — Dropout : espérance, sous-réseaux et règle du test ∂
1. $m$ vaut 0 ou 1, donc $m^2 = m$ et $\mathbb{E}[m] = \mathbb{E}[m^2] = 1 - p$. Ainsi $\mathbb{E}[\tilde a] = \frac{a}{1 - p}\,(1 - p) = a$, et $\mathbb{E}[\tilde a^2] = \frac{a^2}{(1 - p)^2}\,(1 - p) = \frac{a^2}{1 - p}$, d'où $\mathrm{Var}[\tilde a] = \frac{a^2}{1 - p} - a^2 = a^2\,\frac{p}{1 - p}$. Quand $p \to 1$, la variance tend vers l'infini : on garde de moins en moins de neurones, avec un facteur de plus en plus grand. Pour $p = 0{,}5$, elle vaut $a^2$ : l'écart-type est de l'ordre de l'activation elle-même.
2. Chaque neurone est gardé ou coupé : $2^n$ masques, donc $2^n$ sous-réseaux, qui utilisent tous les mêmes poids (un neurone gardé a toujours les mêmes poids, quel que soit le masque). Chaque exemple de chaque mini-batch passe par un sous-réseau tiré au hasard, et le réseau complet du test se comporte comme une moyenne de tous. Pour $n = 100$, $2^{100} = (2^{10})^{10} \approx 10^{30}$ (exactement $1{,}27 \times 10^{30}$) : la plupart ne sont jamais tirés, mais ils partagent leurs poids avec ceux qui le sont.
3. $\mathbb{E}[z] = \sum_i w_i a_i\,\mathbb{E}[m_i] = (1 - p)\sum_i w_i a_i$, qui est exactement ce que calcule la couche au test avec les poids $(1 - p)\,w_i$ et tous les neurones gardés.
4. Dropout inversé avec des poids $w'_i$ : en entraînement, $z = \sum_i w'_i a_i m_i/(1 - p)$ ; au test, $z = \sum_i w'_i a_i$. Article d'origine avec des poids $w_i$ : en entraînement, $z = \sum_i w_i a_i m_i$ ; au test, $z = \sum_i (1 - p)\,w_i a_i$. Avec $w'_i = (1 - p)\,w_i$, les deux calculs coïncident pour chaque masque en entraînement **et** au test : les deux méthodes définissent la même famille de modèles, à un changement d'échelle des poids près. Les entraînements, eux, ne sont pas identiques avec le même learning rate : le gradient par rapport à $w'$ vaut celui par rapport à $w$ divisé par $1 - p$, et un pas sur $w'$ correspond à un pas sur $w$ de learning rate $\eta/(1 - p)^2$ ; l'initialisation et la régularisation L2 changent aussi d'échelle. Le dropout inversé a l'avantage pratique de ne rien changer au modèle de test.
5. Les quatre masques $(m_1, m_2)$, chacun de probabilité $\frac14$, donnent $z = m_1 - m_2 \in \{0, 1, -1, 0\}$, et $h(z) \in \{0, 1, 0, 0\}$ : $\mathbb{E}[h(z)] = \frac14$. Or $\mathbb{E}[z] = 0{,}5 \times (1 - 1) = 0$, et la règle du test calcule $h(0{,}5 \times 1 \times 1 + 0{,}5 \times 1 \times (-1)) = h(0) = 0$. L'écart vient de l'**inégalité de Jensen** : pour une fonction convexe, $\mathbb{E}[h(z)] \ge h(\mathbb{E}[z])$, avec égalité seulement si $h$ est affine (sur les valeurs prises). La règle du test est donc une approximation de la moyenne des sous-réseaux, exacte pour une couche linéaire, et qui marche bien en pratique pour les réseaux profonds.
**Erreurs fréquentes** : oublier que $m^2 = m$ et écrire $\mathbb{E}[m^2] = (1 - p)^2$ ; croire que le dropout inversé et l'article d'origine donnent des modèles différents au test ; appliquer Jensen dans le mauvais sens (pour une fonction concave, l'inégalité s'inverse).
**Variante** : avec $p = 0{,}2$, la variance de $\tilde a$ vaut $a^2 \times 0{,}2/0{,}8 = $ **0,25** $a^2$, soit un écart-type de **0,5** $|a|$.

### Ex 20.6 — Compter les 138 millions de paramètres de VGG16 ✏️
**Réponses** : a) **7** ; b) **1 792** ; c) **14 714 688** ; d) **102 764 544** ; e) **138 357 544** ; f) **89,4** ; g) **553,4** ; h) **B**.
**Démarche** : a) $224/2^5$. b) $(9 \times 3 + 1) \times 64$. c) Bloc 1 : $1\,792 + 36\,928 = 38\,720$ ; bloc 2 : $73\,856 + 147\,584 = 221\,440$ ; bloc 3 : $295\,168 + 2 \times 590\,080 = 1\,475\,328$ ; bloc 4 : $1\,180\,160 + 2 \times 2\,359\,808 = 5\,899\,776$ ; bloc 5 : $3 \times 2\,359\,808 = 7\,079\,424$ ; total $14\,714\,688$. d) $25\,088 \times 4\,096 + 4\,096$. e) Les deux autres couches denses : $4\,096 \times 4\,096 + 4\,096 = 16\,781\,312$ et $4\,096 \times 1\,000 + 1\,000 = 4\,097\,000$ ; les denses font $123\,642\,856$, et le total $138\,357\,544$ (le nombre de torchvision). f) $123\,642\,856/138\,357\,544 \approx 89{,}36\,\%$. g) $4 \times 138\,357\,544 = 553\,430\,176$ octets. h) $553\,430\,176/2^{20} \approx 527{,}8$ : torchvision écrit « MB » pour des mébioctets.
**Dans ta copie** : un filtre de convolution est **partagé** par toutes les positions de l'image : ses $9\,C_{in}$ poids servent des milliers de fois, alors qu'une couche dense a un poids par couple (entrée, sortie). Les convolutions font l'essentiel des calculs (quelque 15 milliards de multiplications-additions par image, 🧮 20.9) avec 11 % des paramètres. Dans un réseau moderne, un **pooling moyen global** résume chaque canal en une valeur, $512 \times 7 \times 7 \to 512$ ; une couche dense de 512 vers 4 096 aurait $2\,101\,248$ paramètres au lieu de $102\,764\,544$, soit **100 663 296** de moins (et ResNet va plus loin : une seule couche dense de 512 vers 1 000 classes).
**Erreurs fréquentes** : oublier les biais ; prendre $C_{in} = 3$ pour toutes les convolutions ; oublier que le padding garde la taille (et diviser à chaque convolution) ; confondre Mo et Mio.
**Variante** : VGG11 (8 convolutions : 64 | 128 | 256, 256 | 512, 512 | 512, 512, et les mêmes trois couches denses) a **9 220 480** paramètres de convolution et **132 863 336** en tout.

### Ex 20.7 — Connexion résiduelle et layer norm à la main ✏️
**Réponses** : a) **3** ; b) **3,5** ; c) **1,6036** ; d) **4** ; e) **Vrai** ; f) **0,9313** ; g) **44** ; h) **1,3478**.
**Démarche** : b) Écarts $(-2, -1, 0, 3)$, carrés de somme 14, divisée par 4. c) $3/\sqrt{3{,}5}$. d) Les sorties ont une moyenne nulle et une variance (biaisée) de 1 : la somme de leurs carrés vaut $4 \times 1$. e) La layer norm calcule ses statistiques sur les features de chaque exemple, seul. f) $0{,}5^{30} = 9{,}313 \times 10^{-10}$, soit $0{,}9313 \times 10^{-9}$. g) $\ln 0{,}01/\ln 0{,}9 \approx 43{,}7$ : $0{,}9^{43} \approx 0{,}0108$ et $0{,}9^{44} \approx 0{,}0097$. h) Chaque bloc a pour dérivée $1 + 0{,}01$ : $1{,}01^{30} \approx 1{,}3478$.
**Dans ta copie** : non : avec $F' = -0{,}5$, chaque bloc a pour dérivée $1 - 0{,}5 = 0{,}5$, et le produit sur 30 blocs s'effondre exactement comme sans connexion résiduelle. Une connexion résiduelle n'aide que si la branche $F$ reste **petite** : au début de l'entraînement, on l'initialise près de 0 (la dernière couche de la branche, ou le $\gamma$ de sa dernière normalisation, mis à 0 ou à une petite valeur), pour que chaque bloc parte de l'identité.
**Erreurs fréquentes** : utiliser la variance non biaisée (la layer norm de PyTorch divise par $D$) ; croire que la somme des carrés vaut 1 (c'est leur moyenne) ; additionner les dérivées au lieu de les multiplier ; prendre $F'$ pour la dérivée du bloc au lieu de $1 + F'$.
**Variante** : pour $\mathbf{x} = (2, 4, 6, 8)$, la moyenne vaut 5, la variance **5**, et la dernière composante de la layer norm $3/\sqrt5 \approx$ **1,3416**.

### Ex 20.8 — Le gradient de la batchnorm ∂
1. $y_i = \gamma\hat x_i + \beta$ : $\partial y_i/\partial\beta = 1$ et $\partial y_i/\partial\gamma = \hat x_i$, donc $\partial L/\partial\beta = \sum_i g_i$ et $\partial L/\partial\gamma = \sum_i g_i\hat x_i$ (une somme sur le mini-batch, comme le gradient d'un biais au ch. 18).
2. $\sum_i \hat x_i = s\sum_i(x_i - \mu) = s\,(m\mu - m\mu) = 0$, et $\sum_i \hat x_i^2 = s^2\sum_i (x_i - \mu)^2 = m\,\sigma^2 s^2 = m\,\frac{\sigma^2}{\sigma^2 + \epsilon}$, qui vaut $m$ pour $\epsilon = 0$ (si $\sigma^2 > 0$).
3. $\partial\mu/\partial x_j = 1/m$. $\partial\sigma^2/\partial x_j = \frac2m\sum_i (x_i - \mu)\,(\delta_{ij} - \frac1m) = \frac2m(x_j - \mu) - \frac{2}{m^2}\sum_i(x_i - \mu) = \frac2m(x_j - \mu)$ : le terme qui vient de $\mu$ est multiplié par $\sum_i(x_i - \mu) = 0$. Puis $\partial s/\partial x_j = -\frac12(\sigma^2 + \epsilon)^{-3/2}\cdot\frac2m(x_j - \mu) = -\frac{s^3}{m}(x_j - \mu)$. Enfin
$$\frac{\partial\hat x_i}{\partial x_j} = s\left(\delta_{ij} - \frac1m\right) + (x_i - \mu)\frac{\partial s}{\partial x_j} = s\left(\delta_{ij} - \frac1m - \frac1m\,\hat x_i\hat x_j\right),$$
car $(x_i - \mu)(x_j - \mu)\,s^3 = s\,\hat x_i\hat x_j$.
4. $\frac{\partial L}{\partial x_j} = \sum_i g_i\,\gamma\,\frac{\partial\hat x_i}{\partial x_j} = \gamma s\left(g_j - \frac1m\sum_i g_i - \frac{\hat x_j}{m}\sum_i g_i\hat x_i\right) = \frac{\gamma s}{m}\left(m\,g_j - \frac{\partial L}{\partial\beta} - \hat x_j\,\frac{\partial L}{\partial\gamma}\right)$, exacte pour tout $\epsilon \ge 0$ (avec $\sigma^2 > 0$ si $\epsilon = 0$). Une fois les deux sommes calculées (environ $2m$ opérations), chaque $\partial L/\partial x_j$ coûte un nombre fixe d'opérations : $O(m)$ en tout (quelque $5m$ opérations), comme la passe avant, au lieu des $O(m^2)$ de la matrice $m \times m$ des $\partial\hat x_i/\partial x_j$.
5. $\sum_j \partial L/\partial x_j = \frac{\gamma s}{m}\big(m\sum_j g_j - m\sum_i g_i - 0\big) = 0$ grâce à $\sum_j\hat x_j = 0$. Et $\sum_j \hat x_j\,\partial L/\partial x_j = \frac{\gamma s}{m}\big(m\sum_j g_j\hat x_j - 0 - \sum_j\hat x_j^2\sum_i g_i\hat x_i\big) = \gamma s\,(1 - \sigma^2 s^2)\sum_i g_i\hat x_i = \gamma s\,\frac{\epsilon}{\sigma^2 + \epsilon}\sum_i g_i\hat x_i$, nul pour $\epsilon = 0$ et minuscule quand $\epsilon \ll \sigma^2$. Interprétation : ajouter une même constante à toutes les entrées ne change pas $\hat x$ (on la retire avec la moyenne), et multiplier tous les écarts à la moyenne par un même facteur non plus (on divise par l'écart-type, exactement si $\epsilon = 0$). Ces deux directions de modification sont invisibles à la loss : le gradient leur est orthogonal.
6. En évaluation, $\hat x_j = (x_j - \mu^{\text{run}})/\sqrt{\sigma^{2,\text{run}} + \epsilon}$ avec des constantes : $\partial L/\partial x_j = \gamma\,g_j/\sqrt{\sigma^{2,\text{run}} + \epsilon}$, une simple mise à l'échelle, sans interaction entre les exemples.
7. Pour $m = 1$, $\mu = x_1$, $\sigma^2 = 0$ et $\hat x_1 = 0$ quelle que soit l'entrée : la sortie vaut $\beta$, et $\partial L/\partial x_1 = \gamma s\,(g_1 - g_1 - 0) = 0$. La couche ne transmet ni information ni gradient ; PyTorch refuse ce cas en entraînement (`BatchNorm1d` sur un seul exemple ; une `BatchNorm2d` sur une seule image a encore $H \times W$ valeurs par canal, et l'accepte).
8. Pour un canal $c$, $m = N \times H \times W$, et les sommes $\sum_i$ portent sur tous les exemples et toutes les positions $(n, h, w)$ du canal ; $\gamma_c$ et $\beta_c$ sont partagés par toutes ces positions, et leurs gradients sont aussi des sommes sur $(n, h, w)$.
**Erreurs fréquentes** : oublier que $\mu$ dépend de $x_j$ (on obtient $\gamma s\,g_j$, le gradient de l'évaluation) ; oublier le chemin par $\sigma^2$ ; garder le terme $\partial\mu/\partial x_j$ dans $\partial\sigma^2/\partial x_j$ sans voir qu'il s'annule ; former la matrice jacobienne $m \times m$ au lieu de la formule en $O(m)$.
**Variante** : pour $m = 2$, avec $\epsilon = 0$, $\hat x_1 = -1$ et $\hat x_2 = 1$ quelles que soient les deux valeurs (distinctes), et la formule donne $\partial L/\partial x_1 = \frac{\gamma s}{2}\big(2g_1 - (g_1 + g_2) + (g_2 - g_1)\big) =$ **0** : avec deux valeurs, la sortie ne dépend plus de l'entrée, sauf par l'ordre.

### Ex 20.9 — Fermi : entraîner VGG16 sur ImageNet chez soi ? 🧮
**Réponses** : a) **46,5** ; b) **4,40** ; c) **5,1** ; d) **510** ; e) **2,21**.
**Démarche** : a) $3 \times 15{,}5$. b) $46{,}5 \times 10^9 \times 1{,}28 \times 10^6 \times 74 \approx 4{,}40 \times 10^{18}$. c) $4{,}40 \times 10^{18}/10^{13} = 4{,}40 \times 10^5$ s, divisées par $86\,400$. d) Cent fois plus : $4{,}40 \times 10^7$ s, environ 509,8 jours, presque un an et demi. e) $16 \times 138\,357\,544 \approx 2{,}21 \times 10^9$ octets.
**Dans ta copie** : côté mémoire, ce calcul oublie les **activations** gardées pour la passe arrière : de l'ordre de 24 millions de valeurs par image pour VGG16 (environ 93 Mo en `float32` pour la seule passe avant, selon le cours CS231n de Stanford), soit plus de 20 Go pour un mini-batch de 256 images, bien plus que les 2,2 Go de e). Côté temps, il oublie que le matériel n'atteint jamais son débit de crête, le chargement et l'augmentation des images, et surtout les essais : on n'entraîne pas un réseau une seule fois. Simonyan et Zisserman indiquent deux à trois semaines par réseau, sur quatre GPU de 2014. Télécharger des poids pré-entraînés coûte quelques secondes et des centaines de mégaoctets, au lieu de jours de calcul (et de son énergie) : c'est l'idée du transfert, bonus B1.
**Erreurs fréquentes** : oublier le facteur 3 de la passe arrière ; confondre secondes, heures et jours ; compter seulement les paramètres en mémoire, sans les gradients ni les états de l'optimiseur.
**Variante** : ResNet-18 coûte 1,81 milliard d'opérations par image en passe avant ; entraîné 90 epochs, il demande $3 \times 1{,}81 \times 10^9 \times 1{,}28 \times 10^6 \times 90 \approx 6{,}26 \times 10^{17}$ opérations, soit environ **17,4 heures** à $10^{13}$ opérations par seconde.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 20.10 — Agents, superviseurs, directeur : expliquer l'apprentissage hiérarchique 🗣️
**Réponse modèle** : « Imagine une banque où cinq agents lisent chacun ton dossier et donnent une note, chacun selon ses propres critères. Des superviseurs ne lisent jamais le dossier : ils regardent seulement les notes des agents, et repèrent des combinaisons (“quand les agents 1 et 3 sont d'accord, c'est bon signe”). Le directeur, lui, ne voit que l'avis des superviseurs, et décide. Chaque niveau travaille sur ce que produit le niveau du dessous, et peut repérer des régularités de plus en plus générales : c'est ce qui rend le système précis. Mais pour dire pourquoi ton dossier a été refusé, il faudrait démêler toutes ces notes de notes : plus il y a de niveaux, plus c'est difficile. La limite de l'image : dans un réseau, personne ne choisit ses critères ; tous les “agents” sont réglés en même temps, automatiquement, pour faire moins d'erreurs sur des dossiers passés. »
**Ce qui compte** : les trois niveaux et ce que chacun voit, la hiérarchie, ce qui rend le système plus précis et le recul de l'explicabilité, la limite (des règles ajustées ensemble par l'entraînement, pas comprises ni choisies) ; aucune formule, et pas de niveau qui « comprend » le dossier.
**Erreurs fréquentes** : dire qu'un niveau « comprend » le dossier ou applique des règles qu'on lui a dictées (elles sont apprises toutes ensemble, à partir des erreurs passées) ; confondre profond (beaucoup de niveaux) et large (beaucoup d'agents par niveau) ; promettre qu'en montrant toutes les notes on explique la décision (une explication complète n'est pas forcément satisfaisante) ; glisser une formule.
**Variante** : explique en une phrase pourquoi l'on dit « profond » : « parce que le dossier passe par beaucoup de niveaux avant la décision, chacun ne voyant que le travail du précédent ».

### Ex 20.11 — Refus de prêt automatisé : explication, dérive et biais ⚖️
1. Les 40 valeurs et les poids sont une explication **complète** : en refaisant tous les calculs, on retrouve la décision. Elle n'est pas **satisfaisante** : Léa ne peut rien en tirer. Une explication satisfaisante ressemblerait à : « Les deux éléments qui ont le plus pesé sont la durée de votre contrat (14 mois de CDD) et l'absence d'historique de crédit. Votre code postal et votre âge ont aussi joué, et nous les réexaminons. Avec un CDI, ou un an d'historique bancaire de plus, votre demande aurait été acceptée. » Elle dit quels facteurs ont compté, si certains sont contestables, et ce qui changerait la décision : une **explication contrefactuelle**.
2. Le code postal est corrélé au revenu, à l'origine, parfois à l'âge : un modèle qui s'en sert peut reproduire une discrimination sans jamais voir la caractéristique sensible ; retirer celle-ci ne suffit donc pas. Pour le vérifier : séparer les dossiers par groupe (tranche d'âge, zone, et toute caractéristique sensible dont on dispose légalement pour l'audit), calculer pour chacun la matrice de confusion du ch. 3, puis comparer les taux de refus et surtout les taux d'erreur : la part de bons payeurs refusés (faux positifs du « défaut ») et la part de mauvais payeurs acceptés. Un écart important et inexpliqué est un signal à analyser.
3. Par exemple l'inflation de 2022-2023 et la hausse des taux d'intérêt, l'évolution du marché du travail (la part des CDD, du travail indépendant), ou un changement de clientèle (une campagne publicitaire qui attire des profils nouveaux). On détecte la dérive **sans attendre les défauts** en comparant la distribution des entrées et celle des probabilités prédites à celles des données d'entraînement (histogrammes, indice de stabilité de la population, tests statistiques), feature par feature, et en surveillant le taux de refus.
4. La banque ne connaît l'issue que des prêts qu'elle a accordés : si le modèle refuse à tort tous les jeunes en CDD, il n'y aura jamais de données pour le montrer, et le réentraînement confirmera le biais. Correction possible, mais coûteuse : accorder une petite part des dossiers refusés (au hasard, ou juste sous le seuil), pour observer leur issue ; on peut aussi utiliser des données externes (les remboursements de ces clients ailleurs) ou des méthodes statistiques de « réintégration des refusés », moins fiables.
5. **RGPD** : c'est une décision fondée **exclusivement** sur un traitement automatisé (aucun humain n'intervient), qui affecte Léa de façon significative (un refus de crédit) : l'article 22 s'applique. Elle n'est permise que si elle est nécessaire au contrat, autorisée par la loi ou fondée sur un consentement explicite, et Léa a alors droit à une intervention humaine, à exprimer son point de vue et à contester la décision ; elle peut aussi demander des informations utiles sur la logique du traitement (art. 15). L'arrêt SCHUFA (2023) confirme qu'un score qui détermine de façon décisive l'octroi du crédit relève de l'article 22. **AI Act** : un système qui évalue la solvabilité de personnes physiques est **à haut risque** (annexe III, point 5 b)) ; son fournisseur doit gérer les risques, garantir la qualité et la représentativité des données, documenter le système, en assurer la traçabilité et la transparence, prévoir un **contrôle humain** et en mesurer l'exactitude et la robustesse ; la banque qui l'utilise doit aussi évaluer son impact sur les droits fondamentaux (art. 27). **Article 86** : Léa pourra demander des explications claires et pertinentes sur le rôle du système dans la décision et sur ses principaux éléments ; les obligations des systèmes à haut risque de l'annexe III s'appliqueront à partir du 2 décembre 2027 (règlement omnibus 2026/1744) ; l'art. 86 vise les décisions fondées sur un tel système : vérifie sa date d'application dans le texte consolidé (EUR-Lex) avant de l'invoquer. S'y ajoute, dans toute l'Union, la nouvelle directive sur le crédit aux consommateurs (2023/2225, art. 18 § 8), dont les lois nationales s'appliquent à partir du 20 novembre 2026 : quand l'évaluation de la solvabilité passe par un traitement automatisé, le consommateur peut demander une intervention humaine, une explication de l'évaluation, faire valoir son point de vue et obtenir un réexamen.
6. Par exemple : (a) un audit d'équité par groupe (question 2) avant chaque mise en production, avec des seuils d'écart acceptables décidés à l'avance ; (b) la suppression ou la justification des features qui servent de variables de substitution, et une explication contrefactuelle générée pour chaque refus ; (c) un humain qui examine les refus proches du seuil et toute contestation ; (d) une surveillance de la dérive (question 3), un réentraînement planifié et une petite part de dossiers acceptés pour observer les refusés (question 4) ; et, pour tout cela, une documentation (une fiche du modèle) tenue à jour.
**Erreurs fréquentes** : croire que retirer l'âge ou l'origine du modèle suffit à le rendre équitable (les variables de substitution) ; confondre explication complète et satisfaisante ; oublier le biais de sélection ; affirmer que l'AI Act interdit le crédit automatisé (il l'encadre) ; donner des dates d'application sans vérifier le règlement omnibus.
**Variante** : un modèle de tri de CV pour un recrutement est aussi à haut risque (annexe III, point 4) ; le même raisonnement s'applique, avec un biais de sélection analogue (on ne voit les performances que des candidats embauchés).

### Ex 20.12 — Ioffe et Szegedy (2015), puis Santurkar et coll. (2018) : lecture critique 📄
1. L'*internal covariate shift* est le changement de la distribution des activations d'une couche pendant l'entraînement, parce que les paramètres des couches précédentes changent. Selon les auteurs, chaque couche doit sans cesse s'adapter à une nouvelle distribution d'entrées, ce qui oblige à de petits learning rates, à une initialisation soignée, et pousse les activations saturantes (sigmoïde) dans leurs zones plates.
2. Pour un mini-batch $\{x_1, \dots, x_m\}$ : $\mu_B = \frac1m\sum_i x_i$, $\sigma_B^2 = \frac1m\sum_i(x_i - \mu_B)^2$, $\hat x_i = (x_i - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$, $y_i = \gamma\hat x_i + \beta$ : la fiche §20.4.4, avec la variance biaisée. Les auteurs ajoutent $\gamma$ et $\beta$ pour que la normalisation ne réduise pas ce que la couche peut représenter (une sigmoïde dont les entrées sont toujours centrées réduites reste dans sa zone presque linéaire) : avec $\gamma = \sqrt{\mathrm{Var}[x]}$ et $\beta = \mathbb{E}[x]$, on retrouve exactement l'**identité**, donc la couche d'origine.
3. Pour l'inférence, ils remplacent les statistiques du mini-batch par celles de la population, estimées sur de nombreux mini-batches, avec la variance **non biaisée** : $\mathrm{Var}[x] = \frac{m}{m - 1}\,\mathbb{E}_B[\sigma_B^2]$ ; ils notent que des moyennes mobiles permettent de suivre le modèle pendant l'entraînement. PyTorch fait exactement cela, avec une moyenne mobile exponentielle (`momentum=0.1`) et la variance non biaisée du mini-batch.
4. Pour une couche de convolution, on normalise ensemble toutes les activations d'une même carte de features, pour tous les exemples et toutes les positions : un mini-batch de $m$ images et des cartes de $H \times W$ donnent $m' = m\,H\,W$ valeurs par canal, avec un seul $\gamma$ et un seul $\beta$ par canal. C'est la forme `(N, C, H, W)` de ta librairie : statistiques sur les axes 0, 2 et 3.
5. Sur ImageNet, une version d'Inception avec batchnorm atteint l'accuracy du modèle d'origine en moins de la moitié des pas ; avec un learning rate multiplié par 5 (et d'autres changements : dropout retiré, L2 réduite, décroissance du learning rate plus rapide), elle l'atteint en **14 fois moins de pas** ; avec un learning rate multiplié par 30, elle finit au-dessus ; un ensemble de ces réseaux atteint 4,9 % d'erreur top-5 en validation, mieux que les meilleurs résultats publiés alors. La vitesse et les grands learning rates relèvent de l'**optimisation** ; la possibilité de retirer le dropout et de réduire la L2 relève de la **régularisation**.
6. Santurkar et coll. ajoutent, après chaque batchnorm d'un réseau VGG entraîné sur CIFAR-10, un bruit aléatoire de moyenne non nulle et de variance non unitaire, qui change à chaque pas : ils recréent artificiellement un fort décalage des distributions. Le réseau « batchnorm bruitée » s'entraîne presque aussi bien que celui à batchnorm normale, et bien mieux que celui sans batchnorm. Ils mesurent aussi un *internal covariate shift* défini à partir des gradients, et montrent que la batchnorm ne le réduit pas, voire l'augmente. Conclusion : la stabilité des distributions n'explique pas le gain.
7. Selon eux, la batchnorm rend la loss et surtout son **gradient** plus **lisses** : la loss varie moins vite (une meilleure constante de Lipschitz) et le gradient change moins brusquement d'un point à un autre. Un gradient « plus prédictif » est un gradient qui reste à peu près valable un peu plus loin dans sa direction : un pas plus grand ne tombe pas sur une zone où la pente a changé du tout au tout, ce qui permet de prendre un learning rate plus grand sans diverger. Ils montrent aussi que d'autres normalisations (par des normes $\ell_p$) donnent des gains comparables.
8. Ce qui reste vrai : la méthode (l'algorithme, $\gamma$ et $\beta$, les statistiques de population en inférence) et les résultats mesurés (un entraînement plus rapide, des learning rates plus grands, de meilleures accuracies). Ce qui a été contesté : l'**explication** par l'*internal covariate shift*, qui a donné son titre à l'article. Leçon : l'explication qu'un article donne de ses propres résultats est une hypothèse, pas un résultat ; on la juge sur des expériences conçues pour la tester (comme celle du bruit), et une méthode peut être excellente pour une autre raison que celle de ses auteurs.
**Erreurs fréquentes** : conclure de Santurkar et coll. que la batchnorm ne sert à rien (ils expliquent au contraire pourquoi elle marche) ; confondre l'effet sur l'optimisation et l'effet régularisant ; oublier le facteur $\frac{m}{m-1}$ de l'inférence.
**Variante** : lis le résumé de J. Ba, J. Kiros et G. Hinton, [« Layer Normalization »](https://arxiv.org/abs/1607.06450) (2016) : quels deux défauts de la batchnorm motivent la layer norm ? **Réponse :** sa dépendance à la taille du mini-batch, et la difficulté de l'appliquer aux réseaux récurrents, où les statistiques changeraient à chaque pas de temps.

<a id="entretien"></a>

## 💼 Entretien

### 20.E1 — Dropout : fonctionnement, et ce que change model.eval()
**Réponse modèle en 60 secondes** : « Le dropout est une régularisation. Pendant l'entraînement, à chaque passe avant, chaque activation de la couche est mise à zéro avec une probabilité p, typiquement 0,1 à 0,5, avec un nouveau tirage à chaque mini-batch et pour chaque exemple ; les activations gardées sont multipliées par 1/(1 − p), pour que l'espérance ne change pas : c'est le dropout inversé. Un neurone ne pouvant pas compter sur ses voisins, l'information se répartit, et le réseau évite de se spécialiser sur des détails des données d'entraînement ; on peut aussi y voir une moyenne sur un nombre énorme de sous-réseaux qui partagent leurs poids. En évaluation, `model.eval()` éteint le dropout : la couche devient l'identité, et les prédictions sont déterministes. Si on l'oublie, la validation est mesurée avec des neurones coupés : des prédictions bruitées, une accuracy plus basse et instable d'une exécution à l'autre. Et il ne faut pas oublier `model.train()` en reprenant l'entraînement. »
**Ce qui compte** : le masque à chaque passe, le facteur $1/(1-p)$ et sa raison, l'effet régularisant, l'identité en `eval()`, les symptômes d'un oubli dans les deux sens.
**Relances possibles** : « Le dropout sert-il encore ? » (oui dans les couches denses et dans les Transformers de taille moyenne, souvent 0,1, et pour le fine-tuning ; beaucoup moins dans les CNN avec batchnorm, et souvent pas du tout dans le préentraînement des grands modèles de langage, qui voient chaque donnée une seule fois et surapprennent peu) · « Quelle valeur de p choisir ? » (0,1 à 0,3 le plus souvent, jusqu'à 0,5 dans de grandes couches denses comme celles de VGG16 ; plus le réseau surapprend, plus p est grand ; on le règle sur la validation) · « Pourquoi diviser par 1 − p pendant l'entraînement plutôt que multiplier les poids au test ? » (les deux donnent la même espérance, mais le dropout inversé laisse le modèle de test intact : rien à changer au moment de prédire).

### 20.E2 — Batchnorm ou layer norm : différences et usages
**Réponse modèle en 60 secondes** : « Les deux appliquent la même formule : on standardise, puis on multiplie par γ et on ajoute β, deux vecteurs appris. La différence est l'axe des statistiques. La batchnorm calcule une moyenne et une variance par feature, sur les exemples du mini-batch, et pour une image sur toutes les positions aussi ; en entraînement elle dépend donc des autres exemples, et elle tient à jour des statistiques glissantes qu'elle utilise en évaluation : elle se comporte différemment en `train()` et en `eval()`, et marche mal avec de petits mini-batches. La layer norm calcule ses statistiques par exemple, sur ses features : elle ne dépend pas du mini-batch, n'a pas de statistiques glissantes, et fait le même calcul en entraînement et en inférence, même sur un seul exemple. C'est pourquoi les Transformers et les RNN l'utilisent : séquences de longueurs variables, petits mini-batches, génération token par token. La batchnorm reste le standard des CNN comme ResNet. Les grands modèles de langage utilisent souvent RMSNorm, une layer norm sans centrage. »
**Ce qui compte** : la même formule, les axes, la conséquence train et eval, la dépendance au mini-batch, et les usages typiques.
**Relances possibles** : « Et la GroupNorm ? » (elle découpe les canaux d'une image en groupes et normalise chaque groupe par exemple : indépendante du mini-batch comme la layer norm, mais adaptée aux convolutions ; on l'utilise en détection et en segmentation, où les mini-batches sont petits) · « Que se passe-t-il avec un mini-batch d'un seul exemple ? » (une `BatchNorm1d` le refuse en entraînement : une seule valeur par feature, une variance nulle ; une `BatchNorm2d` l'accepte tant que l'image a plus d'un pixel, mais ses statistiques sont très bruitées ; la layer norm n'est pas gênée) · « Où place-t-on la layer norm dans un bloc de Transformer ? » (aujourd'hui, au début de chaque branche résiduelle, avant l'attention et avant la partie dense : le « pre-LN », plus stable à entraîner que le placement d'origine, après l'addition).

### 20.E3 — Décrire une boucle d'entraînement PyTorch, étape par étape
**Réponse modèle en 60 secondes** : « Avant la boucle : je crée le modèle et je l'envoie sur le device, je crée la loss, par exemple `CrossEntropyLoss` sur des logits, puis l'optimiseur sur `model.parameters()`, et un DataLoader qui mélange les données d'entraînement. Pour chaque epoch, j'appelle `model.train()`, puis, pour chaque mini-batch : j'envoie X et y sur le device ; je fais la passe avant, `logits = model(X)` ; je calcule la loss ; `optimizer.zero_grad()`, parce que PyTorch accumule les gradients ; `loss.backward()`, qui calcule les gradients ; `optimizer.step()`, qui met à jour les poids. L'ordre compte : `zero_grad` avant `backward`, `step` après. Après chaque epoch, je mesure la validation avec `model.eval()` et `torch.no_grad()`, j'enregistre les courbes, je sauvegarde le `state_dict` si c'est le meilleur modèle et je décide d'un early stopping ; et je n'oublie pas de revenir à `model.train()`. Si j'ai un planning du learning rate, `scheduler.step()` vient après `optimizer.step()`. »
**Ce qui compte** : la préparation, les cinq gestes dans le bon ordre et leur rôle, le device, la validation avec `eval()` et `no_grad()`, la sauvegarde.
**Relances possibles** : « Et si un mini-batch ne tient pas en mémoire ? » (l'accumulation de gradients : plusieurs passes avant et arrière sur des mini-batches plus petits sans `zero_grad()`, en divisant chaque loss par leur nombre, puis un seul `step()` ; on peut aussi utiliser la précision mixte) · « Pourquoi additionner `loss.item()` et pas `loss` ? » (un tenseur garde le graphe de calcul de chaque mini-batch : la mémoire grossit pendant l'epoch ; `.item()` rend un `float`) · « Où mettre le gradient clipping ? » (entre `loss.backward()` et `optimizer.step()` : `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`, qui agit sur les gradients déjà calculés, ch. 19).

### 20.E4 — Ton modèle surapprend : que fais-tu, et dans quel ordre ?
**Réponse modèle en 60 secondes** : « D'abord, je vérifie que c'est bien de l'overfitting : je regarde les courbes epoch par epoch. Si la loss de validation remonte pendant que celle d'entraînement baisse, oui ; je vérifie aussi qu'il n'y a pas de fuite ou de différence de distribution entre l'entraînement et la validation, et que la validation est mesurée en `eval()`. Ensuite, du moins cher au plus cher : l'early stopping, qui garde les poids de la meilleure epoch ; plus de données, ou de l'augmentation de données si ce sont des images ou du son ; de la régularisation, du weight decay avec AdamW et du dropout dans les couches denses ; puis un modèle plus petit, ou un modèle pré-entraîné qu'on ajuste au lieu d'entraîner de zéro. Je change une chose à la fois et je compare sur la validation, avec plusieurs graines, avant de toucher au test. Un écart de 14 points peut aussi venir d'une validation trop petite ou mal tirée : je vérifie l'intervalle de confiance. »
**Ce qui compte** : un diagnostic avant le remède, une liste ordonnée et justifiée, une méthode (un changement à la fois, la validation), et le soupçon d'une fuite ou d'un décalage.
**Relances possibles** : « Et si le modèle a 85 % en entraînement et en validation ? » (ce n'est pas de l'overfitting mais de l'underfitting, ou une limite des données : plus de capacité, un meilleur learning rate, plus d'epochs, de meilleures features, et l'on vérifie la qualité des labels) · « Comment savoir si plus de données aideraient ? » (une courbe d'apprentissage : entraîner sur 25, 50 puis 100 % des données ; si l'accuracy de validation monte encore à 100 %, plus de données aideront, ch. 9) · « Dropout ou weight decay : lequel en premier ? » (pas de règle : avec AdamW, un weight decay léger ne coûte presque rien ; le dropout va dans les couches denses ; une ablation sur la validation tranche).

### 20.E5 — Pourquoi les connexions résiduelles rendent-elles les réseaux très profonds entraînables ?
**Réponse modèle en 60 secondes** : « Un bloc résiduel calcule y = x + F(x) au lieu de F(x). Par la règle de la chaîne, sa dérivée est l'identité plus celle de F. Sans connexion résiduelle, le gradient qui remonte cent couches est un produit de cent facteurs, qui s'effondre ou explose très vite ; avec, chaque facteur reste proche de 1 tant que F est petite, et le gradient garde une voie directe de la loss jusqu'aux premières couches. Côté apprentissage, chaque bloc n'apprend qu'une correction de son entrée, et l'identité, qui n'est pas facile à représenter avec des couches ordinaires, devient gratuite : il suffit que F tende vers 0. C'est pour cela que ResNet a pu passer de vingt à plus de cent couches en 2015, là où un réseau ordinaire plus profond s'entraînait moins bien qu'un réseau moins profond. On les combine avec une normalisation, et elles sont dans chaque bloc des Transformers, donc dans tous les grands modèles de langage. »
**Ce qui compte** : la formule, la dérivée $1 + F'$ et le produit sur la profondeur, l'apprentissage d'une correction et l'identité facile, l'observation qui a motivé ResNet, et les usages actuels.
**Relances possibles** : « Que faire si les dimensions de x et de F(x) diffèrent ? » (une projection sur le raccourci : une convolution 1 × 1, ou une couche dense, qui adapte le nombre de canaux, avec un stride de 2 si la résolution change) · « Pourquoi initialiser la dernière couche de la branche près de 0 ? » (chaque bloc part alors de l'identité : le réseau profond commence comme un réseau peu profond, et le gradient traverse les blocs sans s'atténuer) · « Les connexions résiduelles suffisent-elles seules ? » (on les combine presque toujours avec une normalisation, batchnorm dans ResNet, layer norm dans les Transformers, qui garde les activations à une échelle raisonnable d'un bloc à l'autre).

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb` ; voici, pour chaque exercice, la réponse, la démarche, les erreurs fréquentes et une variante. Les chiffres sont ceux du mode rapide, sur CPU.

### Ex 20.13 — Premiers tenseurs PyTorch : créer, convertir, dtype, device 📦
**Réponses** : a) **0,1291** ; b) **« torch.float64 »** ; c) **True**.
**Démarche** : `torch.from_numpy(X_13).float().div(255).unsqueeze(1)` : la conversion garde le `uint8`, `.float()` passe en `float32`, la division ramène les pixels entre 0 et 1, et `unsqueeze(1)` ajoute l'axe des canaux (`(64, 28, 28)` devient `(64, 1, 28, 28)`). Les labels de `Y_13` sont déjà des `int64`. a) `images_13.mean().item()` : un pixel moyen de 0,129, donc des images surtout noires. b) `torch.from_numpy` garde le dtype de NumPy, `float64`, alors que `torch.tensor([0.0])` crée du `float32`. c) `from_numpy` **partage** la mémoire : changer `vec[0]` change aussi `t[0]`.
**Dans tes notes** : `torch.tensor(vec)` copie : `t[0]` resterait à 0. Le partage évite de recopier un grand tableau qu'on ne fait que lire ; mais un tableau modifié plus tard (une normalisation en place, un `+=`) change le tenseur sans prévenir.
**Erreurs fréquentes** : diviser des `uint8` avant la conversion en réels (`X_13 / 255` en NumPy donne du `float64`, puis un réseau en `float32` refuse ou convertit) ; `x.view(64, -1)` sur un tenseur non contigu (par exemple après un `transpose`) : `RuntimeError: view size is not compatible with input tensor's size and stride` (`reshape` copie si besoin) ; oublier `.item()` (un tenseur 0-D au lieu d'un `float`) ; convertir les labels en `int32` (`astype(np.int32)`, ou un `np.array` d'entiers sous Windows avec NumPy 1.x) : `CrossEntropyLoss` les refuse (`RuntimeError: expected scalar type Long but found Int`) et veut des `int64`.
**Variante** : `torch.tensor([1, 2]).dtype` vaut **torch.int64**, et `images_13.reshape(64, -1).shape` vaut **(64, 784)**, la forme qu'attend une couche dense.

### Ex 20.14 — Prédire les formes produites par les couches utilitaires 🔮
**Réponses** : a) **« (2, 3, 7, 7) »** ; b) **« (2, 3, 3, 3) »** ; c) **« (2, 3, 4, 4) »** ; d) **« (2, 3) »** ; e) **« (2, 3, 10) »** ; f) **« (2, 3, 4) »**.
**Démarche** : les couches 2-D n'agissent que sur les deux derniers axes. a) Une bordure de 1 de chaque côté : $5 + 2 = 7$. b) Une fenêtre de 2 et un pas de 2 sur 5 pixels : $5/2 = 2{,}5$, arrondi au-dessus avec `ceil_mode=True`, soit 3 : le bloc incomplet du bord est gardé. c) Avec un pas de 1, la fenêtre tient en $\lfloor (5 - 2)/1 \rfloor + 1 = 4$ positions : les blocs se recouvrent. d) `nn.Flatten()` garde l'axe 0 et aplatit le reste : il ne reste qu'un axe, rien ne change. e) `nn.Upsample` agrandit les axes qui suivent les canaux : sur `(N, C, L)`, seulement `L`. f) L'axe 1, de taille 12, devient deux axes, 3 et 4.
**Dans tes notes** : la section *Shape* de la page de chaque couche donne la formule de la taille de sortie (pour le pooling, avec `padding`, `dilation`, `stride` et `ceil_mode`).
**Erreurs fréquentes** : appliquer le padding aux canaux ; oublier que le padding est ajouté **des deux côtés** ; diviser par le pas sans compter la taille de la fenêtre ; croire que `nn.Flatten` rend toujours un vecteur.
**Variante** : `nn.AvgPool2d(3, stride=2)(x4_14)` a la forme **(2, 3, 2, 2)**, et `nn.ZeroPad2d((1, 0, 2, 0))(x4_14)`, qui ajoute 1 colonne à gauche et 2 lignes en haut, la forme **(2, 3, 7, 6)**.

### Ex 20.15 — Autograd : retrouver les gradients du réseau minuscule du ch. 18 📦
**Réponses** : a) $\begin{pmatrix} -2 & 0{,}25 \\ -4 & 0{,}5 \end{pmatrix}$ ; b) $\begin{pmatrix} 0{,}375 & -0{,}625 \\ -1{,}125 & 1{,}875 \end{pmatrix}$ ; c) la transposée de a), $\begin{pmatrix} -2 & -4 \\ 0{,}25 & 0{,}5 \end{pmatrix}$ ; d) **−4** ; e) **0,0531**.
**Démarche** : `tiny_loss_15` écrit la loss avec des opérations torch (`x @ w1 @ w2`, `** 2`, `.sum()`) ; la vérification appelle `backward()`, et les gradients sont ceux de 18.4 g et f, rangés comme les poids (ligne par ligne : XA, XB, puis YA, YB). c) `nn.Linear` range son poids en `(n_out, n_in)`, la transposée de `W1` : son gradient est la transposée de a). d) Deux `backward()` sans remise à zéro **additionnent** les gradients : $2 \times (-2) = -4$. e) Un pas de SGD, $w \leftarrow w - 0{,}1\,g$, puis la loss recalculée : 0,0531, la valeur de 18.4 i.
**Dans tes notes** : un gradient a toujours la forme de son paramètre. `optimizer.zero_grad()` se place avant `loss.backward()` (ou juste après `optimizer.step()`) ; sans lui, chaque pas utilise la somme de tous les gradients depuis le début.
**Erreurs fréquentes** : `torch.tensor(W1_15)` sans `requires_grad=True` (pas de `.grad`) ; passer par NumPy ou `.item()` dans la loss (le graphe est coupé, `grad_fn` vaut `None`) ; copier `W1_15` dans `nn.Linear` sans la transposer ; appeler `backward()` deux fois sur la même loss (PyTorch refuse : le graphe a été libéré ; il faut recalculer la loss).
**Variante** : avec `lr=0.05`, la loss après le pas vaut **0,1906** : la moitié du pas descend moins loin, mais la prévision linéaire de 18.4 h est déjà fausse pour ce pas.

### Ex 20.16 — dropout_forward et dropout_backward 🔨
**Réponses** : l'exemple de la docstring, la part de zéros et la moyenne, deux masques différents, l'évaluation, `p = 1`, le refus et la passe arrière contre l'autograd passent ; les 27 tests de l'exercice passent.
**Démarche** :
```python
def dropout_forward(x, p=0.5, training=True, rng=None):
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    x = np.array(x, dtype=np.float64)                 # a new float64 array
    if not training or p == 0.0:
        return x, None
    if p == 1.0:
        return np.zeros_like(x), np.zeros_like(x)
    rng = np.random.default_rng(rng)                  # None, an int or a Generator
    mask = (rng.random(x.shape) >= p) / (1.0 - p)    # kept with probability 1 - p, then scaled
    return x * mask, mask


def dropout_backward(dout, mask):
    dout = np.array(dout, dtype=np.float64)
    if mask is None:
        return dout
    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != dout.shape:
        raise ValueError(f"mask has shape {mask.shape} but dout has shape {dout.shape}")
    return dout * mask
```
**Dans tes notes** : le dropout est une multiplication par le masque, et sa dérivée est ce masque, celui de **cette** passe avant. Un nouveau masque ferait passer le gradient par des neurones qui n'ont pas participé au calcul : ce ne serait plus le gradient de la loss calculée.
**Erreurs fréquentes** : `0 <= p <= 1` écrit `p < 0 or p > 1` (un `nan` passe) ; garder les valeurs `< p` (une autre convention, que l'exemple de la docstring avec `rng=0` peut distinguer) ; rendre `x` lui-même en évaluation (pas une copie) ; diviser par $1 - p$ quand $p = 1$ ; tirer le masque avec `np.random` au lieu de `rng` ; un seul masque (`rng.random(x.shape[1:])`) partagé par tous les exemples du mini-batch.
**Variante** : `nn.Dropout2d(p)` de PyTorch coupe des **canaux entiers** d'une image (tous les pixels d'un canal ensemble) : sur `torch.ones(1, 4, 3, 3)` avec $p = 0{,}5$, chaque canal sort entièrement à 0 ou entièrement à 2. C'est le dropout qu'on utilise après des convolutions, où les pixels voisins sont trop corrélés pour qu'en couper quelques-uns serve à quelque chose.

### Ex 20.17 — layer_norm, l2_penalty et EarlyStopping 🔨
**Réponses** : `layer_norm` contre `torch.nn.functional.layer_norm`, l'exemple de `l2_penalty`, la suite d'`EarlyStopping` et la restauration passent ; les 43 tests de l'exercice passent.
**Démarche** : `layer_norm` : `mean = x.mean(axis=-1, keepdims=True)`, `var = x.var(axis=-1, keepdims=True)`, `(x - mean) / np.sqrt(var + eps)`, puis `gamma` et `beta` (vérifiés de forme `(D,)`). `l2_penalty` : `float(0.5 * lam * sum(np.sum(w ** 2) for w in weights))` et `[lam * w for w in weights]`, de nouveaux tableaux. `EarlyStopping.step` :
```python
def step(self, value, params=None):
    value = float(value)
    if self.mode == "min":
        better = value < self.best_ - self.min_delta      # a nan is never below (nor above) anything
    else:
        better = value > self.best_ + self.min_delta
    if self.n_checks_ == 0 and not math.isnan(value):
        better = True                                     # the first finite check is always an improvement
    if better:
        self.best_, self.best_epoch_, self.counter_, self.improved_ = value, self.n_checks_, 0, True
        if params is not None and self.restore_best:
            self.best_params_ = [np.array(p, dtype=np.float64, copy=True) for p in params]
    else:
        self.counter_ += 1
        self.improved_ = False
    self.n_checks_ += 1
    if self.counter_ >= self.patience:
        self.stopped_ = True
    return bool(self.counter_ >= self.patience)
```
 `restore` : `p[...] = saved` pour chaque couple, après avoir vérifié qu'un état est gardé et que les formes correspondent.
**Dans tes notes** : avec PyTorch, `step(val_loss)` sans `params`, puis, si `stopper.improved_`, `best_state = copy.deepcopy(model.state_dict())` ; à la fin, `model.load_state_dict(best_state)`. Le `deepcopy` est indispensable : un `state_dict` partage la mémoire des poids.
**Erreurs fréquentes** : normaliser sur l'axe 0 (c'est la batchnorm) ; la variance non biaisée ; oublier le ½ de la pénalité, ou rendre `np.float64` sans `float(...)` ; compter une valeur égale comme une amélioration ; `best_params_ = params` (une référence, qui suit l'entraînement) ; `params[i] = saved[i]` dans `restore` (la liste change, pas les tableaux du réseau) ; un `best_epoch_` compté à partir de 1.
**Variante** : la **RMSNorm** (fiche §20.4.4, encadré) divise par la racine de la moyenne des carrés, sans centrer : pour $(1, 2, 3, 6)$, $\sqrt{(1 + 4 + 9 + 36)/4} = \sqrt{12{,}5}$ et la dernière composante vaut $6/\sqrt{12{,}5} \approx$ **1,6971** (contre 1,6036 pour la layer norm de ✏️ 20.7).

### Ex 20.18 — Écrire sa classe nn.Module et compter ses paramètres 📦
**Réponses** : a) **12** ; b) **35** ; c) **109 386**.
**Démarche** : `__init__` appelle `super().__init__()`, puis range un `nn.Flatten()` et un `nn.ModuleList(nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:]))` ; `forward` aplatit, applique `torch.relu(layer(x))` pour toutes les couches sauf la dernière, et rend la sortie de la dernière. a) $(2 \times 2 + 2) \times 2 = 12$. b) $(2 \times 4 + 4) + (4 \times 3 + 3) + (3 \times 2 + 2) = 12 + 15 + 8 = 35$. c) $100\,480 + 8\,256 + 650 = 109\,386$.
**Dans tes notes** : avec une liste Python ordinaire, `len(list(model.parameters()))` vaut **0** : `nn.Module` n'enregistre que les sous-modules rangés en attributs (ou dans un `nn.ModuleList`, un `nn.Sequential`, un `nn.ModuleDict`). Si toutes les couches sont dans la liste, l'optimiseur refuse même d'être créé (`ValueError: optimizer got an empty parameter list`) ; si d'autres couches sont enregistrées, celles de la liste ne sont jamais mises à jour, ni déplacées par `model.to(device)`, ni sauvegardées par `state_dict()`, et là sans aucun message.
**Erreurs fréquentes** : oublier `super().__init__()` (« cannot assign module before Module.__init__() call ») ; une ReLU après la dernière couche (des logits toujours positifs) ; une softmax à la fin ; `torch.flatten(x)` au lieu de `nn.Flatten()` ; compter les paramètres sans les biais.
**Variante** : `MLP18([784, 256, 10])` a **203 530** paramètres, presque deux fois plus que c) avec une seule couche cachée : la première couche dense coûte $784 \times n_1$.

### Ex 20.19 — Dataset et DataLoader : MNIST en mini-batches 📦
**Réponses** : a) **16** ; b) **40** ; c) **15**.
**Démarche** : `__init__` garde les deux tableaux (convertis en tenseurs), `__len__` rend `len(self.labels)`, `__getitem__(i)` rend `(self.images[i].float().div(255).unsqueeze(0), self.labels[i])`. `make_loader_19` : `DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=drop_last)`. a) $\lceil 1\,000/64 \rceil = 16$ ; b) $1\,000 - 15 \times 64 = 40$ ; c) avec `drop_last=True`, le dernier mini-batch incomplet est jeté : 15.
**Dans tes notes** : convertir dans `__init__` est plus rapide quand tout tient en mémoire, mais coûte 4 fois plus de mémoire (`float32` au lieu de `uint8`). Pour des photos lues sur le disque, impossible de tout charger : `__getitem__` lit et transforme un exemple à la demande, éventuellement dans plusieurs processus (`num_workers`), et c'est là qu'on placera l'augmentation de données (ch. 24).
**Erreurs fréquentes** : rendre l'image `(28, 28)` sans l'axe des canaux ; oublier la division par 255 ; rendre un label `numpy.int64` au lieu d'un tenseur (le `DataLoader` le convertit, mais `dataset[i]` n'a pas le même type) ; créer le générateur une fois en dehors de la fonction (deux chargeurs partagent alors le même flux) ; `shuffle=True` sur le chargeur de validation (inutile, et l'ordre change d'une mesure à l'autre).
**Variante** : avec des mini-batches de 100, une epoch en compte **10**, le dernier est complet (**100** images), et `drop_last=True` n'en jette aucun (**10**).

### Ex 20.20 — Zoo des couches : convolution, pooling, récurrente, utilitaires, bruit 📦
**Réponses** : la convolution de la figure 20.10 rend `(1, 2, 5, 4)` (20 paramètres) ; le max pooling de `SMALL_20` vaut $\begin{pmatrix} 5 & 2 \\ 2 & 8 \end{pmatrix}$ et le pooling moyen $\begin{pmatrix} 3{,}25 & 1 \\ 1 & 4{,}75 \end{pmatrix}$ ; `Noise20` ajoute un bruit d'écart-type $\sigma$ en entraînement seulement ; `Zoo20` a 9 098 paramètres ; `nn.RNN` rend `(64, 28, 32)` et `(1, 64, 32)`.
**Démarche** :
```python
conv_20 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
max_20 = nn.MaxPool2d(2)(SMALL_20)
avg_20 = nn.AvgPool2d(2)(SMALL_20)


class Noise20(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            return x + self.sigma * torch.randn_like(x)
        return x


class Zoo20(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), Noise20(0.1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


rnn_20 = nn.RNN(input_size=28, hidden_size=32, batch_first=True)
```
`Zoo20` enchaîne deux convolutions à padding 1 (la taille reste 28, puis 14) et deux max poolings (28 → 14 → 7), puis un aplatissement de $16 \times 7 \times 7 = 784$ valeurs et une couche dense.
**Dans tes notes** : seules les convolutions (80 et 1 168 paramètres) et la couche dense (7 850) apprennent ; la couche dense porte 86 % des 9 098 paramètres, comme chez VGG16. `nn.RNN` a $28 \times 32$ poids pour l'entrée, $32 \times 32$ pour l'état et deux biais de 32 : $896 + 1\,024 + 64 = 1\,984$. À chaque pas, elle combine la ligne lue et son état précédent, et sa sortie empile les 28 états successifs ; le dernier est l'état final (ch. 22).
**Erreurs fréquentes** : `nn.Conv2d(2, 1, 3)` (les canaux d'entrée viennent en premier) ; oublier le padding (5 × 4 devient 3 × 2) ; un bruit ajouté aussi en évaluation ; un bruit tiré avec NumPy (hors du device, et hors du graphe) ; `nn.Linear(16 * 14 * 14, 10)` (un seul pooling compté) ; `nn.RNN(28, 32)` sans `batch_first=True` (il attend alors `(L, N, 28)`).
**Variante** : `nn.Conv2d(3, 16, 5)` a **1 216** paramètres ($(25 \times 3 + 1) \times 16$), et une couche `nn.LSTM(28, 32)`, qui a quatre jeux de poids au lieu d'un (ch. 22), en a **7 936**.

### Ex 20.21 — Reproduire la figure 20.17 : le plan de VGG16 couche par couche 🎨
**Réponses** : a) **« (1, 256, 28, 28) »** ; b) **74,3** ; le modèle a 138 357 544 paramètres, comme celui de torchvision ; le plan a 39 lignes.
**Démarche** :
```python
def make_vgg16_21():
    layers, c_in = [], 3
    for block in ([64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]):
        for c_out in block:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU()]
            c_in = c_out
        layers.append(nn.MaxPool2d(2))
    layers += [nn.Flatten(), nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
               nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 1000)]
    return nn.Sequential(*layers)


def plan_21(model, x):
    plan = []
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            plan.append((type(layer).__name__, tuple(x.shape)))
    return plan


vgg = make_vgg16_21()
plan = plan_21(vgg, torch.zeros(1, 3, 224, 224))
block3_21 = str([shape for name, shape in plan if name == "MaxPool2d"][2])
share_21 = 100 * n_params([m for m in vgg if isinstance(m, nn.Linear)][0]) / n_params(vgg)
```
`draw_plan_21` (dans le notebook de solutions) dessine un `plt.Rectangle` par couche, coloré par sorte de couche, avec la taille de sortie écrite verticalement. a) 224 divisé trois fois par 2, avec les 256 canaux du bloc 3. b) $102\,764\,544 / 138\,357\,544 = 74{,}3\,\%$. Une passe avant sur une image prend environ 0,2 s sur le CPU de génération.
**Dans tes notes** : la largeur est divisée par 2 à chaque pooling (224 → 112 → 56 → 28 → 14 → 7), jamais par une convolution (padding 1) ; le nombre de canaux double au début de chaque bloc, de 64 à 512. Le livre représente l'aplatissement par un symbole ; sur ta figure, c'est la boîte où la taille passe de $512 \times 7 \times 7$ à 25 088. La figure ne montre pas le coût : les convolutions font l'essentiel des calculs avec 11 % des paramètres, et une seule boîte, la première couche dense, en porte les trois quarts.
**Erreurs fréquentes** : un pooling après chaque convolution (l'image devient trop petite et la couche dense ne reçoit plus 25 088 valeurs) ; `kernel_size=3` sans `padding=1` ; oublier la ReLU après une convolution ; prendre la part des **trois** couches denses (89,4 %, ✏️ 20.6) ; garder plusieurs VGG16 en mémoire (553 Mo chacun).
**Variante** : VGG11 (8 convolutions : 64 | 128 | 256, 256 | 512, 512 | 512, 512) a **132 863 336** paramètres, et la même sortie après le 3ᵉ pooling, **(1, 256, 28, 28)** : le nombre de couches ne change que les calculs, pas les formes.

### Ex 20.22 — batchnorm_forward et batchnorm_backward 🔨
**Réponses** : l'exemple de la docstring, les passes avant, les statistiques glissantes, les gradients et l'évaluation contre `nn.BatchNorm1d` et `nn.BatchNorm2d`, en `(6, 3)` et en `(4, 3, 5, 5)`, et les deux refus passent ; les 32 tests de l'exercice passent.
**Démarche** : deux aides : les axes des statistiques (`(0,)` ou `(0, 2, 3)`), et la mise en forme d'un vecteur de taille `C` en `(1, C)` ou `(1, C, 1, 1)`. La passe avant vérifie toutes les formes, puis, en entraînement : `mean = x.mean(axis=axes)`, `var = x.var(axis=axes)` (biaisée), `inv_std = 1 / np.sqrt(var + eps)`, `x_hat = (x - shaped(mean)) * shaped(inv_std)` ; `running_mean *= 1 - momentum; running_mean += momentum * mean`, et de même pour `running_var` avec `var * m / (m - 1)`. En évaluation, les mêmes formules avec les statistiques glissantes. Le cache garde `x_hat`, `inv_std`, une copie de `gamma`, les axes, $m$ et le mode. La passe arrière (∂ 20.8) : `dbeta = dout.sum(axis=axes)`, `dgamma = (dout * x_hat).sum(axis=axes)`, puis, en entraînement, `dx = shaped(gamma * inv_std) / m * (m * dout - shaped(dbeta) - x_hat * shaped(dgamma))`.
**Dans tes notes** : $dx_j = \frac{\gamma s}{m}\big(m\,g_j - \sum_i g_i - \hat x_j\sum_i g_i\hat x_i\big)$, en $O(m)$ : deux sommes, puis un nombre fixe d'opérations par valeur. La statistique glissante sert à **estimer** la variance de toutes les données : l'estimateur non biaisé ($\frac{m}{m-1}$, ch. 2) ; la normalisation du mini-batch utilise la variance des valeurs qu'elle normalise, qui sortent alors exactement centrées réduites.
**Erreurs fréquentes** : `running_mean = (1 - momentum) * running_mean + ...` (un nouveau tableau : l'appelant ne voit rien) ; les axes `(0,)` pour une image (des statistiques par canal **et** par pixel) ; `var` non biaisée pour normaliser ; la passe arrière sans le chemin par la moyenne ou par la variance ; `dgamma` calculé avec `x` au lieu de `x_hat` ; diviser par $m$ = le nombre d'exemples au lieu du nombre de valeurs par canal ($N \times H \times W$).
**Variante** : `nn.BatchNorm1d(3, affine=False)` n'a **aucun** paramètre appris ($\gamma = 1$ et $\beta = 0$ fixés), mais garde ses deux statistiques glissantes : c'est la version « du livre », sans paramètres (fiche, encadré « Le livre, corrigé »).

### Ex 20.23 — La boucle d'entraînement PyTorch complète sur MNIST 📦
**Réponses** : a) **1,9155** ; b) **0,7755** ; après 3 epochs d'Adam sur 12 000 images, environ **93,5 %** en validation (97,8 % en mode complet, 10 epochs sur 50 000 images, en 16 s).
**Démarche** : `train_one_epoch_23` : `model.train()`, puis, pour chaque mini-batch, `X, y = X.to(device), y.to(device)`, `loss = loss_fn(model(X), y)`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, et les sommes `total += loss.item() * len(y)`, `count += len(y)` ; elle rend `total / count`. `evaluate_23` : `model.eval()` et `with torch.no_grad():`, la loss pondérée et les bonnes réponses `(logits.argmax(dim=1) == y).sum().item()`. Le contrôle (SGD, 32 mini-batches) part d'une loss de $\ln 10 \approx 2{,}3$ : sa moyenne sur l'epoch, 1,92, est encore haute, et l'accuracy de validation qui suit est de 77,55 %.
**Dans tes notes** : la loss d'entraînement d'une epoch est la moyenne des losses mesurées **pendant** l'epoch, avec des poids qui s'améliorent : les premiers mini-batches, mesurés avec des poids presque aléatoires, la tirent vers le haut ; la validation, mesurée après, voit les poids de la fin. Et l'entraînement est mesuré avec le dropout, la validation sans. `scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)`, créé avant la boucle, s'appelle après chaque évaluation : `scheduler.step(val_loss)`.
**Erreurs fréquentes** : la moyenne des losses de mini-batch sans pondération (le dernier mini-batch, plus petit, pèse autant que les autres) ; `total += loss` (un tenseur, qui garde le graphe de chaque mini-batch) ; oublier `model.train()` au début de l'epoch, après une évaluation ; une seconde passe avant pour calculer l'accuracy pendant l'entraînement (elle tire d'autres masques de dropout, et le contrôle reproductible change) ; les données laissées sur le CPU quand le modèle est sur le GPU.
**Variante** : la même expérience (mode rapide, 3 epochs, même modèle de départ) avec `torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)` au lieu d'Adam donne **95,1 %** en validation, contre 93,5 % pour Adam (le résultat ci-dessus) et 91,7 % pour SGD sans momentum : sur ce petit MLP, le réglage par défaut d'Adam n'est pas le meilleur, et seule ta boucle de 20.23 a servi aux trois.

### Ex 20.24 — Chasse aux bugs PyTorch : cinq erreurs classiques 🐛
**Réponses** : les cinq points passent, puis une epoch d'Adam donne environ 88 % sur les images d'entraînement.
**Démarche** : 1. `torch.flatten(x)` met le mini-batch entier à plat : `x.flatten(start_dim=1)` (ou `nn.Flatten()`). 2. La softmax à la fin : rendre `self.out(x)`, des logits. 3. `optimizer.zero_grad()` avant `loss.backward()`. 4. `total += loss.item() * len(y)`. 5. `model.eval()` au début d'`evaluate_24`, et le calcul sous `torch.no_grad()`.
**Dans tes notes** : 1. un plantage dès le premier mini-batch (`mat1 and mat2 shapes cannot be multiplied`) ; 2. une loss qui baisse à peine (une softmax de trop écrase les gradients), sans plantage ; 3. des gradients qui s'additionnent d'un mini-batch à l'autre : avec SGD, des pas qui grandissent, une loss qui oscille, puis explose ou devient `nan` ; avec Adam, qui normalise ses pas, une direction qui suit la somme de tous les gradients passés, et un apprentissage qui se dégrade ; 4. une mémoire qui grossit pendant l'epoch (le graphe de chaque mini-batch est gardé), et une fonction qui rend un tenseur ; 5. une accuracy plus basse, et différente à chaque mesure (le dropout reste actif) ; sans `torch.no_grad()`, de la mémoire perdue.
**Erreurs fréquentes** : corriger le point 1 avec `x.view(-1)` ou `reshape(-1)` (le même bug) ; placer `zero_grad()` entre `backward()` et `step()` (les gradients sont effacés avant d'être utilisés : les poids ne bougent plus, le point 3 le signale) ; ajouter `torch.no_grad()` sans `model.eval()` (le dropout reste actif) ; oublier de remettre `model.train()` dans l'epoch d'entraînement (ici, `train_epoch_24` le fait déjà).
**Variante** : avec la seule erreur 2 (la softmax à la fin), quelle est la loss **la plus basse** que peut atteindre le réseau ? `CrossEntropyLoss` prend les probabilités $p_j \in [0, 1]$ pour des logits : au mieux, $p_y = 1$ et les 9 autres valent 0, d'où $-1 + \ln(e + 9) = \ln(1 + 9/e) \approx$ **1,4612**. La loss ne peut jamais descendre sous 1,46, même avec un réseau parfait : le symptôme est une loss bloquée haut.

### Ex 20.25 — Courbes train/validation : overfitting, eval() oublié, fuite 📈
**Réponses** : a) **B** ; b) **7** ; c) **E** ; d) **A** ; e) **C** ; f) **D**.
**Démarche** : B : la loss de validation passe par un minimum net à l'epoch 7, puis remonte longtemps pendant que celle d'entraînement baisse vers 0 : un early stopping garde l'epoch 7. E : la validation colle à l'entraînement jusqu'à presque 0, avec 99,5 % d'accuracy, invraisemblable sur Fashion-MNIST. A : la validation est au niveau de l'entraînement (mesurée elle aussi avec le dropout), et elle saute d'une epoch à l'autre. C : deux losses hautes (vers 0,8) et collées, et une accuracy de 78 % : underfitting. D : la validation reste sous l'entraînement et ne remonte pas : sain.
**Dans tes notes** : f) la loss d'entraînement est mesurée **avec** le dropout (le réseau est handicapé) et en moyenne pendant l'epoch (avec des poids moins bons qu'à la fin) ; la validation, mesurée après l'epoch et sans dropout, peut donc être plus basse, sans fuite. c) Ce qui alerte : une accuracy au-dessus de l'état de l'art, et une validation qui suit l'entraînement à l'identique même quand celui-ci tend vers 0. Pour vérifier, on cherche des exemples en double entre les deux jeux (des empreintes des images, avec `hashlib` sur `x.tobytes()`, ou des doublons d'identifiants, ch. 12).
**Erreurs fréquentes** : prendre D pour une fuite (validation sous l'entraînement) ; compter les epochs à partir de 0 en b) ; choisir l'epoch du minimum de la loss d'**entraînement** ; prendre A pour de l'overfitting (sa validation ne remonte pas, elle est bruitée).
**Variante** : sur la courbe B, un early stopping de patience 3 s'arrête à l'epoch **10** (trois epochs sans amélioration après l'epoch 7) et recharge les poids de l'epoch 7.

### Ex 20.26 — Un modèle ImageNet face à des images qu'il n'a jamais vues 🔮
**Réponses** : a) **A** ; b) **A** ; c) **True** ; d) **D** ; e) **D**.
**Démarche** : a) seulement **7 %** des 100 photos de chats sont classées dans une race de chat domestique : à 32 × 32 pixels, agrandies à 256 × 256 puis recadrées au centre en 224 × 224 (les bords sont perdus), elles sont floues, et le modèle répond souvent un autre animal ou un objet ; les chiens, eux, sont reconnus comme un chien (une des 118 races d'ImageNet) environ 3 fois sur 4, parce qu'il y a beaucoup plus de classes de chiens. b) Sur le bruit, la probabilité la plus haute vaut **0,13** en moyenne : la softmax répartit sa masse entre de nombreuses classes. c) Sur les photos de CIFAR-10, elle vaut **0,38** : plus haute que sur le bruit, mais loin de 1. d) **68 %** des images de bruit reçoivent la même classe, « prayer rug » (un tapis de prière, une texture fine et colorée) : le bruit n'est pas « rien » pour le réseau, il ressemble à une texture. e) La règle « refuser sous 0,5 » refuse **75 %** des photos de CIFAR-10 (et tout le bruit).
**Dans tes notes** : la règle e) sépare le bruit des photos, mais elle refuse aussi la plupart des vraies images, qui sont « hors distribution » par leur taille : la probabilité la plus haute est un indice de confiance, imparfait (D. Hendrycks et K. Gimpel, 2017, en ont fait une méthode de référence pour détecter ces images, en notant ses limites). Un modèle moderne n'est pas forcément sûr de lui sur du bruit ; il peut l'être sur d'autres images hors de ses classes (le livre en montre). Pour reconnaître les chats de CIFAR-10 : adapter le modèle aux petites images et aux 10 classes, en réentraînant sa dernière couche (le transfert, bonus B1), ou entraîner un CNN sur CIFAR-10 (ch. 21).
**Erreurs fréquentes** : croire qu'un classifieur peut répondre « je ne sais pas » (il distribue toujours une probabilité de 1 entre ses classes) ; confondre la probabilité la plus haute avec l'accuracy ; oublier le prétraitement du modèle (sans normalisation, les prédictions n'ont plus de sens).
**Variante** : en comptant comme juste n'importe quelle classe de félin d'ImageNet (les indices 281 à 293 : les cinq chats domestiques, puis le puma, le lynx, le léopard, le lion, le tigre…), la part des 100 chats monte seulement à **11 %** : le modèle ne prend pas les chats pour d'autres félins, il y voit surtout autre chose.

### Ex 20.27 — Sauvegarder, recharger et reproduire un entraînement 🛠️
**Réponses** : le fichier contient les cinq clés et se relit en mode sûr ; chaque élément est rechargé ; la reprise donne exactement les mêmes poids que deux epochs d'affilée.
**Démarche** : `save_checkpoint_27` : `torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "generator": generator.get_state(), "torch_rng": torch.get_rng_state()}, path)`. `load_checkpoint_27` : `checkpoint = torch.load(path)`, puis `model.load_state_dict(...)`, `optimizer.load_state_dict(...)`, `generator.set_state(...)`, `torch.set_rng_state(...)`, et `return checkpoint["epoch"]`.
**Dans tes notes** : sans `"optimizer"`, Adam repart de moyennes nulles : ses pas suivants ont une autre taille ; sans `"generator"`, la seconde epoch voit les mini-batches dans un autre ordre ; sans `"torch_rng"`, le dropout tire d'autres masques. Chacun suffit à changer les poids. Sur GPU, il faut aussi l'état du générateur CUDA (`torch.cuda.get_rng_state_all()`) et des opérations déterministes (`torch.use_deterministic_algorithms(True)`).
**Erreurs fréquentes** : `torch.save(model, path)` (le modèle entier : le fichier dépend du code, et `torch.load` refuse de le relire en mode sûr) ; recharger le `state_dict` dans un modèle d'une autre architecture ; recréer l'optimiseur **après** avoir rechargé ses états ; écrire le checkpoint dans le dépôt (utilise un dossier ignoré par git).
**Variante** : reprends la seconde epoch en oubliant **un seul** des trois états (on part de la même sauvegarde après une epoch) : l'écart maximal entre les poids et ceux de deux epochs d'affilée vaut environ **0,020** sans l'état de l'optimiseur, **0,026** sans celui du générateur des données (un autre ordre des mini-batches) et **0,033** sans celui du générateur global (d'autres masques de dropout), alors qu'une reprise complète donne un écart de 0 : chaque oubli change visiblement le modèle.

### Ex 20.28 — Dropout, batchnorm, AdamW et early stopping sur Fashion-MNIST 📦
**Réponses** : en mode rapide (10 000 images), l'early stopping garde l'epoch **10** et s'arrête à la 13ᵉ (environ 6 s) ; le modèle restauré a une loss de validation de 0,389 et **86,3 %** d'accuracy.
**Démarche** : `make_model_28` : `torch.manual_seed(seed)`, puis `[nn.Flatten()]` et, pour chaque couche cachée, `nn.Linear`, `nn.BatchNorm1d`, `nn.ReLU()`, `nn.Dropout(0.3)`, puis la couche de sortie, dans un `nn.Sequential` envoyé sur `DEVICE`. `fit_28` crée l'optimiseur, l'early stopping et l'historique **avant** la boucle, et recharge les meilleurs poids **après** :
```python
def fit_28(model, max_epochs=MAX_EPOCHS_28, patience=PATIENCE_28):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    stopper = mylearn.nn.regularization.EarlyStopping(patience=patience)
    loss_fn = nn.CrossEntropyLoss()
    history, best_state = {"train_loss": [], "val_loss": [], "val_acc": []}, None
    for epoch in range(1, max_epochs + 1):
        history["train_loss"].append(train_one_epoch_23(model, train_loader_28(epoch), loss_fn, optimizer, DEVICE))
        val_loss, val_acc = evaluate_23(model, VAL_LOADER_28, loss_fn, DEVICE)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        stop = stopper.step(val_loss)
        if stopper.improved_:
            best_state = copy.deepcopy(model.state_dict())
        if stop:
            break
    model.load_state_dict(best_state)
    return history
```
**Dans tes notes** : sans `deepcopy`, `best_state` contient les **mêmes** tenseurs que le modèle (un `state_dict` partage la mémoire des poids) : ils continuent de changer, et `load_state_dict(best_state)` recharge les poids de la dernière epoch ; la vérification de la restauration le voit. La loss d'entraînement reste au-dessus de la validation au début, parce qu'elle est mesurée avec le dropout et en moyenne pendant l'epoch (la courbe D de 20.25).
**Erreurs fréquentes** : recréer l'early stopping ou l'optimiseur dans la boucle ; appeler `step` sur la loss d'entraînement ; sauver `model.state_dict()` sans copie ; oublier de recharger les meilleurs poids à la fin ; numéroter les epochs à partir de 0 (le chargeur n'a plus la graine prévue) ; une batchnorm placée après la ReLU (ça marche aussi, mais ce n'est pas l'architecture demandée).
**Variante** : en mode complet (50 000 images, 40 epochs au plus), l'early stopping garde l'epoch **7** (arrêt à la 10ᵉ, en 20 s) et le modèle fait **88,6 %** en validation, avec une loss de 0,305 au lieu de 0,389 : chaque epoch fait cinq fois plus de pas, si bien que le minimum arrive en moins d'epochs, mais beaucoup plus bas.

### Ex 20.29 — Ablation : qu'est-ce qui retarde l'overfitting ? 🔬
**Réponses** (mode rapide, 2 000 images, 25 epochs, moyennes sur 2 graines) :

| configuration | meilleure epoch | loss de validation minimale | loss de validation finale | loss d'entraînement finale | meilleure accuracy |
|---|---|---|---|---|---|
| rien | 11,5 | 0,538 | 0,617 | 0,127 | 0,830 |
| dropout 0,5 | 22,5 | 0,496 | 0,509 | 0,330 | 0,832 |
| weight decay 1 | 21,5 | 0,519 | 0,588 | 0,198 | 0,826 |
| batchnorm | 4,0 | 0,522 | 0,751 | 0,030 | 0,823 |
| les trois | 15,0 | 0,505 | 0,543 | 0,286 | 0,823 |

**Démarche** : la vérification moyenne les courbes de 2 graines par configuration.
```python
def run_29(dropout, weight_decay, batchnorm, seed, epochs=EPOCHS_29):
    torch.manual_seed(seed)
    layers = [nn.Flatten()]
    for n_in, n_out in [(784, 256), (256, 256)]:
        layers.append(nn.Linear(n_in, n_out))
        if batchnorm:
            layers.append(nn.BatchNorm1d(n_out))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    model = nn.Sequential(*layers, nn.Linear(256, 10)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loader = DataLoader(TRAIN_SET_29, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        history["train_loss"].append(train_one_epoch_23(model, loader, nn.CrossEntropyLoss(), optimizer, DEVICE))
        val_loss, val_acc = evaluate_23(model, VAL_LOADER_29, nn.CrossEntropyLoss(), DEVICE)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    return history
```
**Dans tes notes** : le **dropout** retarde le plus le minimum (epoch 22,5 au lieu de 11,5) et donne la meilleure loss, minimale comme finale ; le weight decay (fort, à 1) retarde aussi, un peu moins. La **batchnorm** n'est pas une régularisation ici : elle **accélère** l'apprentissage (son minimum arrive dès l'epoch 4) et surapprend plus vite et plus fort (une loss d'entraînement de 0,03, une validation finale de 0,75) : c'est l'encadré de la fiche. Les trois ensemble font moins bien que le dropout seul : les effets ne s'additionnent pas. Les écarts d'accuracy (0,823 à 0,832) sont du même ordre que les écarts entre deux graines (de 0,002 à 0,012 selon la configuration) : sur l'accuracy, on ne peut rien conclure ; sur la loss et l'epoch du minimum, les écarts sont nets. Pour conclure plus sûrement : plus de graines, avec leur écart-type, et un vrai jeu de test.
**Erreurs fréquentes** : comparer des configurations avec des graines différentes (l'écart vient alors aussi de l'initialisation) ; conclure sur une seule graine ; juger sur l'accuracy finale au lieu de la courbe de validation ; un weight decay de 0,01 (l'effet est invisible sur 25 epochs : AdamW rétrécit les poids de $\eta\lambda = 10^{-5}$ par pas).
**Variante** : la configuration « rien » entraînée sur **8 000** images au lieu de 2 000 (mêmes graines, 25 epochs) atteint une loss de validation minimale de **0,41** au lieu de 0,54, et **87 %** d'accuracy au lieu de 83 % : avoir plus de données fait mieux que toutes les régularisations du tableau.

### Ex 20.30 — Un réseau de 30 couches : connexions résiduelles et layer norm 🔬
**Réponses** (30 blocs de 64 neurones, 2 epochs sur 5 000 images) :

| réseau | rapport des gradients première/dernière couche à l'initialisation | loss après l'entraînement | accuracy |
|---|---|---|---|
| sans rien | $3{,}5 \times 10^{-11}$ | 2,301 | 0,111 |
| layer norm seule | $1{,}0 \times 10^{3}$ | 1,893 | 0,221 |
| connexions résiduelles | 0,63 | 0,484 | 0,842 |
| connexions résiduelles + layer norm | 0,60 | 0,232 | 0,927 |

**Démarche** :
```python
class Deep30(nn.Module):
    def __init__(self, depth=30, width=64, residual=False, norm=False):
        super().__init__()
        self.residual = residual
        self.inp = nn.Linear(784, width)
        self.norms = nn.ModuleList(nn.LayerNorm(width) if norm else nn.Identity() for _ in range(depth))
        self.linears = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.out = nn.Linear(width, 10)

    def forward(self, x):
        x = self.inp(x)
        for norm, linear in zip(self.norms, self.linears):
            h = torch.relu(linear(norm(x)))
            x = x + h if self.residual else h
        return self.out(x)


def grad_ratio_30(model, X, y):
    model.zero_grad()
    nn.functional.cross_entropy(model(X), y).backward()
    return (model.inp.weight.grad.norm() / model.out.weight.grad.norm()).item()
```
Sans raccourci, le gradient qui remonte 30 couches est un produit de 30 facteurs, qui s'effondre : $3 \times 10^{-11}$, comme les $0{,}5^{30} \approx 10^{-9}$ de ✏️ 20.7 f. La première couche ne reçoit rien, et le réseau reste à la loss de $\ln 10 \approx 2{,}3$ du hasard (11 % d'accuracy, la classe la plus fréquente). Avec les connexions résiduelles, chaque bloc a pour dérivée $I + F'$ : le gradient garde une voie directe, et le réseau apprend.
**Dans tes notes** : avec les seules connexions résiduelles, $x$ grossit à chaque bloc ($x + h$ avec $h \ge 0$ après la ReLU) : les activations et les gradients deviennent énormes (des normes de l'ordre de $10^2$ à $10^3$), même si leur rapport reste proche de 1 ; l'apprentissage marche, mais moins bien. La layer norm seule renormalise l'entrée de chaque couche dense, mais ne crée pas de voie directe : le gradient traverse toujours les 30 blocs, et cette fois il **explose** en remontant (une norme de 219 sur la première couche contre 0,215 sur la dernière, un rapport de $10^3$) ; l'apprentissage reste mauvais et instable. Les deux ensemble, la normalisation **dans** la branche et le raccourci **autour**, c'est l'architecture « pre-LN » de chaque bloc d'un Transformer (bonus B3).
**Erreurs fréquentes** : appliquer la ReLU après l'addition (`relu(x + h)`, une autre architecture, celle de ResNet, qui marche aussi) ; mettre la layer norm après la couche dense au lieu d'avant ; oublier `model.zero_grad()` dans `grad_ratio_30` (un second appel additionne les gradients) ; calculer le rapport sur les biais.
**Variante** : avec `torch.tanh` au lieu de la ReLU (mêmes graines), le rapport des gradients du réseau sans raccourci vaut environ $5 \times 10^{-7}$ au lieu de $3 \times 10^{-11}$ (la tanh écrase moins), et le réseau n'apprend toujours pas (10,5 % d'accuracy) ; avec les connexions résiduelles, il atteint **89,7 %** (0,842 avec la ReLU).

### Ex 20.31 — Défi Fashion-MNIST : au moins 90 % avec un MLP de 300 k paramètres au plus 🏆
**Réponses** : la référence, un MLP 784-256-256-10 (270 346 paramètres) avec batchnorm et dropout de 0,2, AdamW (weight decay $10^{-4}$) et un planning *one-cycle* (`OneCycleLR`, learning rate maximal $3 \times 10^{-3}$), 15 epochs de mini-batches de 128, s'entraîne en 25 à 30 s sur le CPU de génération (2 cœurs) et fait **90,45 %** sur le test caché (95,7 % sur 10 000 images d'entraînement).
**Démarche** :
```python
def train_31(X, y):
    torch.manual_seed(31)
    layers = []
    for n_in, n_out in [(784, 256), (256, 256)]:
        layers += [nn.Linear(n_in, n_out), nn.BatchNorm1d(n_out), nn.ReLU(), nn.Dropout(0.2)]
    model = nn.Sequential(*layers, nn.Linear(256, 10)).to(DEVICE)
    epochs, batch = 15, 128
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=epochs * math.ceil(len(X) / batch))
    generator = torch.Generator().manual_seed(31)
    X, y = X.to(DEVICE), y.to(DEVICE)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=generator).split(batch):
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()                          # OneCycleLR counts in steps, once per mini-batch
    return model.eval()
```
Calibrage (sur le test, une seule fois par réglage, après coup) : le MLP simple 784-256-10 avec Adam fait 87,4 % en 5 epochs, 88,1 % en 10 et 89,0 à 89,3 % en 20 (deux graines) ; la référence, 90,49 % et 90,45 % avec deux graines. Le défi demande donc plus que des epochs : la normalisation, le dropout et surtout le planning font la différence.
**Dans tes notes** : une démarche honnête garde une validation (les 10 000 dernières images d'entraînement), change une chose à la fois, et ne regarde le test qu'à la fin. Le calibrage ci-dessus montre seulement que le MLP simple, même entraîné plus longtemps, reste sous 90 % ; pour savoir lequel des ingrédients de la référence (planning, batchnorm, dropout, deuxième couche) compte le plus, il faut une ablation comme celle de 20.29, sur la validation.
**Erreurs fréquentes** : régler sur le test caché (le résultat est alors optimiste) ; dépasser 300 000 paramètres avec trois couches de 512 ; oublier `model.eval()` avant de rendre le modèle (le dropout fausse la mesure) ; appeler `scheduler.step()` une fois par epoch avec `OneCycleLR`, qui compte en **pas** ; un MLP qui reçoit `(N, 1, 28, 28)` sans aplatissement.
**Variante** : un essai honnête de lissage des labels (`label_smoothing=0.1` dans la cross-entropy), entraîné sur les 50 000 premières images et mesuré sur les 10 000 dernières (la validation), donne **90,24 %**, contre 90,31 % pour la référence entraînée de la même façon : aucun gain, l'idée est écartée sans toucher au test. Pour viser le palier 🌟 (91 %), il reste d'autres pistes à valider de même : plus d'epochs (le budget de temps le permet) ou une augmentation légère (décaler les images d'un ou deux pixels).
