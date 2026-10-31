# 24 · PyTorch en pratique 2 : améliorer, chercher, CNN et RNN — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels, des exercices ✏️ 24.1 à 24.5 et de l'estimation 🧮 24.7 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 24.Q1 — Compter les hyperparamètres et n'en changer qu'un
a) **« BCE »** : la taille de batch, le learning rate et le taux de dropout se choisissent avant l'entraînement ; les poids, les biais et les moyennes glissantes d'Adam sont calculés par lui. b) **729** $= 3^6$. c) **Faux** : garder la valeur par défaut, c'est choisir cette valeur ; elle n'est pas forcément la bonne pour ton problème (le livre compte ces défauts dans ses 27). d) **Faux** : si le meilleur réglage de A dépend de B (le learning rate et la taille de batch, par exemple), régler A puis B une seule fois peut s'arrêter loin de la meilleure combinaison ; il faut parfois revenir sur A. e) **C** : les objectifs (overfitting, accuracy, vitesse, mémoire) se contredisent souvent ; on en sert un ou deux, puis on passe aux suivants.
**À retenir** : un hyperparamètre se choisit, un paramètre s'apprend ; ne changer qu'une chose à la fois permet d'attribuer un effet, mais n'assure pas l'optimum quand les réglages interagissent.

### 24.Q2 — Taille de batch : bruit, overfitting et temps
a) **782** $= \lceil 50\,000/64 \rceil$ ($781{,}25$ : le dernier mini-batch, de 16 images, compte aussi). b) **196** $= \lceil 50\,000/256 \rceil$ ($195{,}3$). c) **5** : l'écart-type varie en $1/\sqrt{B}$, et $\sqrt{400}/\sqrt{16} = 20/4$. d) **Faux** : avec un batch de 8, l'epoch fait 64 fois plus de pas, chacun huit fois plus bruité ; à learning rate égal, la trajectoire n'est pas la même, et le livre observe plus d'overfitting. e) **A** : chaque pas a un coût fixe ; plus de pas, plus de temps (fig. 24.5).
**À retenir** : la taille de batch change à la fois le nombre de pas, leur bruit et le temps de calcul ; on la règle avec le learning rate.

### 24.Q3 — Plus de neurones, moins de neurones, dropout
a) **D** : le modèle d'une couche sur-apprenait déjà ; plus de neurones lui donnent encore plus de capacité, et la loss de validation remonte plus tôt (fig. 24.7). b) **0,3** : `p` est la probabilité de mettre une valeur à zéro. c) **C** : un masque neuf à chaque passe avant, valeur par valeur et exemple par exemple. d) **2,5** $= 2{,}0/(1 - 0{,}2)$ : le dropout inversé divise les valeurs gardées par $1 - p$. e) **Faux** : c'est le dropout d'origine qui multipliait par $1 - p$ en évaluation ; le dropout inversé de PyTorch ne fait rien en évaluation, puisqu'il a déjà divisé pendant l'entraînement.
**À retenir** : `nn.Dropout(p)` met une fraction $p$ des valeurs à zéro, avec un masque neuf à chaque passe avant, et divise les autres par $1 - p$ ; en évaluation, il laisse tout passer.

### 24.Q4 — Valider un réseau : coût, jeu non représentatif, normalisation dans chaque fold
a) **200** minutes : 5 folds, donc 5 entraînements de 40 minutes. b) **A** : les 2 000 dernières radios viennent toutes de l'hôpital B ; la validation mesure un seul hôpital, comme le dernier quart d'un paquet de cartes neuf du livre. c) **D** : le scaler apprend un minimum et un maximum ; dans le pipeline, il les apprend sur la partie d'entraînement de chaque fold, sans voir la validation. d) **Vrai** : pour un classifieur, un `cv` entier devient un `StratifiedKFold` (sans mélange) ; pour un régresseur, un `KFold`. e) **C** : les pixels étaient déjà dans $[0, 1]$ ; le scaler appris sur 90 % des images ne change presque rien.
**À retenir** : une validation croisée coûte $K$ entraînements ; tout ce qui apprend sur les données va dans le pipeline.

### 24.Q5 — 🕰️ KerasClassifier, grille, hasard ou zoom : chercher des hyperparamètres
a) **C** : `KerasClassifier` a été retiré de Keras et n'a jamais enveloppé de réseau PyTorch ; `MLPClassifier` est le réseau de scikit-learn, pas le tien ; skorch, ou ton propre estimateur (📦 24.16), le font. b) **5** (depuis scikit-learn 0.22 ; 3 auparavant, comme dans le livre). c) **48** $= 2 \times 3 \times 2 \times 4$. d) **49** : le réentraînement final, sur toutes les données, avec les meilleurs réglages. e) **D** : une grille large et grossière, puis des grilles plus fines autour des meilleures régions.
**À retenir** : coût d'une grille = produit des nombres de valeurs × nombre de folds (+ 1 pour le réentraînement final).

### 24.Q6 — 🕰️ Couches utilitaires de Keras et leurs équivalents PyTorch
a) **C** : `nn.Upsample(scale_factor=2)` répète chaque valeur (mode `"nearest"` par défaut). b) **A** : une tranche qui retire une ligne et une colonne de chaque bord. c) **B** : `nn.ZeroPad2d(1)`. d) **D** : `nn.AvgPool2d(2)`. e) **Faux** : `GaussianNoise` n'agit qu'à l'entraînement, comme le dropout. f) **C** : la batchnorm normalise les activations de chaque canal sur le mini-batch, puis applique $\gamma$ et $\beta$ appris ; elle ne touche ni aux poids ni aux gradients.
**À retenir** : presque chaque couche utilitaire de Keras a un module PyTorch du même nom ou presque ; celles qui n'en ont pas (le recadrage, le bruit) s'écrivent en une ligne.

### 24.Q7 — Préparer des images pour nn.Conv2d
a) **« (16, 3, 32, 32) »**. b) **D** : `permute(0, 3, 1, 2)` amène l'axe des canaux (3) en position 1. `reshape` mélangerait les valeurs ; `transpose(1, 3)` donne bien la forme `(16, 3, 32, 32)`, mais échange aussi la hauteur et la largeur : l'image serait couchée ; `permute(0, 2, 3, 1)` donnerait `(16, 32, 3, 32)`. c) **« (16, 12, 32, 32) »** : `padding="same"` garde la taille, seuls les canaux changent. d) **Faux** : PyTorch lève `ValueError: padding='same' is not supported for strided convolutions` ; on écrit `padding=2`. e) **−0,6** $= 51/127{,}5 - 1 = 0{,}4 - 1$. f) **Faux** : `nn.Conv2d` n'a pas d'activation ; on ajoute un `nn.ReLU()`.
**À retenir** : en PyTorch, des images en $(N, C, H, W)$ obtenues par `permute`, des convolutions sans activation, et `padding="same"` seulement avec un stride de 1.

### 24.Q8 — Augmentation et données synthétiques : quelles transformations sont plausibles ?
a) **B** : l'augmentation fabrique des exemples d'entraînement ; la validation et le test doivent ressembler aux vraies données. b) **Faux** : le `Dataset` tire une nouvelle transformation à chaque fois qu'il rend l'image, donc à chaque epoch. c) **C** : une transformation utile produit une image plausible de même label. d) **D** : un générateur fabrique des batches tant qu'on lui en demande ; il fallait lui dire combien de batches font une epoch (`steps_per_epoch` dans Keras 2 ; le livre écrit encore `samples_per_epoch`, le nom de Keras 1), pour savoir quand afficher les mesures et appeler les callbacks. e) **B** : un `IterableDataset` n'a pas de `__len__` ; c'est son code qui s'arrête après le nombre d'exemples choisi. f) **Vrai** : il ne revoit jamais deux fois le même exemple ; il peut toujours sur-apprendre les **particularités du générateur**, qui ne sont pas celles des vraies données.
**À retenir** : on augmente le train seulement, avec des transformations plausibles, tirées à nouveau à chaque epoch.

### 24.Q9 — Motifs conv-conv-pool, stride contre pooling, batchnorm contre dropout
a) **C** : convolution (sans activation), batchnorm, puis ReLU (listing 24.44). b) **Vrai** : la batchnorm soustrait la moyenne de chaque canal, qui contient le biais ; son $\beta$ joue son rôle (d'où `bias=False`). c) **2,25** : $\frac{(H/2)(W/2)(3C)(3C)}{H\,W\,C\,C} = \frac{9}{4}$ ; en **doublant** les canaux, l'usage, le rapport vaudrait 1 : c'est pourquoi on double les canaux quand on divise la taille par deux. d) **A** : 0,9912 contre 0,9939, en 36 s par epoch au lieu de 147 s. e) **C** : un pooling moyen global (`nn.AdaptiveAvgPool2d(1)`), puis une seule couche linéaire.
**À retenir** : conv → batchnorm → ReLU, sans biais ; doubler les canaux quand la taille est divisée par deux ; le stride remplace souvent le pooling.

### 24.Q10 — Séries pour un RNN : fenêtres, scaler, inverse_transform, RMSE
a) **D** : le scaler s'ajuste sur le train seul, puis transforme le train et le test. b) **« (500, 12, 1) »** : $(N, T, D)$. c) **A** : après `inverse_transform`, la RMSE se lit dans l'unité de la série, et se compare à d'autres modèles. d) **Faux** : c'est normal, et le réseau doit faire avec ; réajuster le scaler sur le test serait une fuite (et changerait l'échelle que le réseau a apprise). e) **C** : une couche dense d'un neurone, sans activation, pour ne pas écraser la valeur prédite. f) **B** : la persistance ; si le réseau ne la bat pas, il n'a rien appris d'utile.
**À retenir** : scaler appris sur le train, fenêtres $(N, w, 1)$, sortie linéaire, RMSE en unités d'origine, et la persistance comme référence.

### 24.Q11 — 🕰️ return_sequences, stateful, TimeDistributed : que devient chaque option en PyTorch ?
a) **D** : `output`, la sortie de la dernière couche à chaque pas, de forme $(N, T, H)$. b) **A** : `h_n[-1]`, l'état de la dernière couche au dernier pas ; `h_n[0]` est celui de la première couche. c) **B** : `nn.Linear` agit sur la dernière dimension, avec les mêmes poids à chaque pas. d) **C** : la ligne $i$ du batch suivant. e) **Vrai** : le graphe du batch précédent a été libéré par son `backward()` ; PyTorch lève « Trying to backward through the graph a second time ». f) **Faux** : chaque exemple d'un mini-batch est une séquence indépendante, qui part d'un état nul.
**À retenir** : `output` ↔ `return_sequences=True`, `h_n[-1]` ↔ la sortie par défaut de Keras, `nn.Linear` ↔ `TimeDistributed(Dense)`, l'état porté ↔ `(h, c)` passé au batch suivant avec `detach()`.

### 24.Q12 — 🕰️ API fonctionnelle de Keras 3 et forward() de PyTorch : branches, partage, gel
a) **C** : un seul objet, un seul jeu de poids ; les gradients des deux appels s'additionnent dans ses `.grad`. b) **B** : `a + b`. c) **« (4, 32, 8, 8) »** : les canaux s'additionnent, $16 + 16$. d) **Faux** : les gradients s'arrêtent, mais les statistiques glissantes changent encore à chaque passe avant en mode `train()` ; il faut aussi `module.eval()`. e) **A** : une convolution 2D attend des images à hauteur, largeur et canaux ; il faudrait d'abord remodeler le vecteur. f) **Vrai** : `keras.Input`, `keras.Model(inputs, outputs)`, `keras.layers.Add`.
**À retenir** : en PyTorch, `forward()` est l'API fonctionnelle : variables pour les branches, `+` et `torch.cat` pour les fusions, un module réutilisé pour partager ; geler, c'est `requires_grad_(False)` **et** `eval()` pour la batchnorm.

<a id="rappels"></a>

## 🔁 Rappels

### 24.R1 — Ch. 23 : ce qu'un historique d'entraînement doit contenir pour comparer deux runs
a) **« ABD »** : les losses d'entraînement et de validation, l'accuracy de validation et le learning rate de chaque epoch. Une copie des poids à chaque epoch coûterait trop de mémoire : on garde le meilleur checkpoint (ch. 23). La loss de test à chaque epoch inviterait à choisir l'epoch sur le test, une fuite. b) **« BCD »** : la graine, tous les hyperparamètres (défauts compris, qui changent d'une version à l'autre), la durée et la machine ; la couleur des courbes n'apprend rien. c) **Faux** : c'est la moyenne des losses des mini-batches **pendant** l'epoch, alors que les poids changent ; c'est pourquoi elle est un peu plus haute que la loss du modèle de fin d'epoch sur le même train. d) **5** : la plus petite loss, 0,44, est la cinquième valeur.
**À retenir** : un historique sert à lire les courbes et à choisir une epoch sans toucher au test ; pour comparer deux runs, on note aussi leur graine, leurs réglages, leur durée et leur machine.

### 24.R2 — Ch. 21 : taille de sortie d'une convolution à padding 'same' et stride 2
a) **14** $= \lceil 28/2 \rceil$. b) **8** $= \lceil 15/2 \rceil$. c) **2** $= \lfloor 5/2 \rfloor$. d) **16** $= \lfloor (32 + 4 - 5)/2 \rfloor + 1 = 15 + 1$. e) **7** $= \lfloor (15 - 3)/2 \rfloor + 1 = 6 + 1$. f) **Vrai** : PyTorch accepte un noyau pair avec `padding="same"` et un stride de 1 ; il met alors une ligne de zéros de plus en bas et une colonne de plus à droite (ch. 21).
**À retenir** : le « same » de Keras avec un stride $s$ donne $\lceil H/s \rceil$ ; en PyTorch, `padding=k // 2` le reproduit pour un noyau impair.

### 24.R3 — Ch. 17 : ce que change la division des logits avant le softmax
a) **0,665** $= e^2/(e^2 + e + 1) = 7{,}389/11{,}107$. b) **0,867** : logits $(4, 2, 0)$, $e^4/(e^4 + e^2 + 1) = 54{,}60/62{,}99$. c) **0,51** : logits $(1 ;\ 0{,}5 ;\ 0)$, $e/(e + e^{0{,}5} + 1) = 2{,}718/5{,}367 = 0{,}506$. d) **B** : les écarts entre logits divisés grandissent sans limite, et toute la probabilité va au plus grand. e) **Vrai** : diviser par un nombre positif garde l'ordre des logits, donc l'argmax.
**À retenir** : $T < 1$ rend la distribution plus pointue (texte plus sage, plus répétitif), $T > 1$ plus plate (plus inventif, plus d'erreurs) ; l'argmax ne change pas.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 24.1 — Le coût d'une grille (27 hyperparamètres, 54 entraînements) contre une recherche aléatoire : probabilité de tomber dans les 5 % meilleurs réglages en n essais ✏️
**Réponses** : a) **134 217 728** ; b) **4,3** ; c) **54** ; d) **63** ; e) **0,64** ; f) **59** ; g) **0,603**.
**Démarche** : a) $2^{27}$. b) $134\,217\,728 / (3\,600 \times 24 \times 365) = 134\,217\,728 / 31\,536\,000 = 4{,}26$ ans. c) $3 \times 3 \times 2 = 18$ candidats, chacun entraîné 3 fois : 54. d) $54 \times 70 = 3\,780$ s $= 63$ min. e) Un tirage rate les 5 % meilleurs avec la probabilité 0,95 ; 20 tirages indépendants les ratent tous avec la probabilité $0{,}95^{20} = 0{,}3585$ ; d'où $1 - 0{,}3585 = 0{,}6415$. f) Il faut $0{,}95^n \le 0{,}05$, soit $n \ge \ln 0{,}05 / \ln 0{,}95 = 58{,}4$ : $n = 59$ ($0{,}95^{58} = 0{,}0510$, $0{,}95^{59} = 0{,}0485$). g) $1 - 0{,}95^{18} = 1 - 0{,}3972 = 0{,}6028$.
**Dans ta copie** : la grille essaie **3** valeurs du nombre de neurones (20, 30, 40) ; 18 tirages d'un entier entre 20 et 40 en essaient jusqu'à 18 différentes (un peu moins s'il y a des doublons). Si le nombre de neurones est le seul hyperparamètre qui compte, la grille gaspille les deux autres dimensions : ses 18 candidats (54 entraînements) n'apprennent que 3 choses sur lui. C'est l'argument de Bergstra et Bengio (2012).
**Erreurs fréquentes** : calculer $0{,}05 \times 20 = 1$ (une « certitude ») au lieu de $1 - 0{,}95^{20}$ ; prendre $1 - 0{,}05^{20}$ ; arrondir f) au plus proche (58) au lieu du premier entier qui suffit ; oublier le nombre de folds en c).
**Variante** : pour viser les 1 % meilleurs avec une probabilité de 0,95, il faut $n \ge \ln 0{,}05 / \ln 0{,}99 = 298{,}1$, soit **299** tirages : viser cinq fois plus haut coûte cinq fois plus de tirages.

### Ex 24.2 — Formes et paramètres du CNN à pooling du livre ✏️
**Réponses** : a) **« (256, 30, 28, 28) »** ; b) **« (256, 30, 14, 14) »** ; c) **780** ; d) **4 336** ; e) **784** ; f) **100 480** ; g) **114 502** ; h) **0,955**.
**Démarche** : a) Noyau 5, padding 2, stride 1 : $H_{out} = 28 + 4 - 5 + 1 = 28$ ; 30 filtres. b) Le pooling divise par 2 : $14 \times 14$. c) $30 \times (1 \times 25 + 1) = 780$. d) $16 \times (30 \times 9 + 1) = 16 \times 271 = 4\,336$. e) La seconde convolution garde $14 \times 14$, le second pooling donne $7 \times 7$ : $16 \times 7 \times 7 = 784$. f) $(784 + 1) \times 128 = 100\,480$. g) $780 + 4\,336 + 100\,480 + 8\,256 + 650 = 114\,502$, avec $(128 + 1) \times 64 = 8\,256$ et $(64 + 1) \times 10 = 650$. h) $(100\,480 + 8\,256 + 650)/114\,502 = 109\,386/114\,502 = 0{,}9553$.
**Dans ta copie** : le livre oublie les **16 canaux** : il reste $7 \times 7$ positions, mais 16 valeurs par position, soit 784 valeurs, autant que les pixels de départ. Les deux convolutions n'ont que 5 116 paramètres ; la première couche dense en a 100 480. Un CNN moderne remplacerait `Flatten` par un pooling moyen global (`nn.AdaptiveAvgPool2d(1)`), qui réduit chaque canal à une valeur : la première couche dense ne recevrait que 16 valeurs, soit $(16 + 1) \times 128 = 2\,176$ paramètres.
**Erreurs fréquentes** : oublier les biais ($30 \times 25 = 750$) ; compter $C_{in} = 1$ dans la seconde convolution ; prendre $7 \times 7 = 49$ comme le livre ; donner la forme d'une seule image en a) ; mettre les canaux à la fin (format Keras).
**Variante** : avec un pooling moyen global à la place de `Flatten`, le modèle aurait $780 + 4\,336 + 2\,176 + 8\,256 + 650 =$ **16 198** paramètres, sept fois moins.

### Ex 24.3 — Stride ou pooling : compter les multiplications ✏️
**Réponses** : a) **588 000** ; b) **846 720** ; c) **109 184** ; d) **358 680** ; e) **3,30** ; f) **17 640** ; g) **4,1**.
**Démarche** : a) $28 \times 28 \times 30 \times 1 \times 25$. b) $14 \times 14 \times 16 \times 30 \times 9$ : la seconde convolution travaille sur la sortie du premier pooling. c) $784 \times 128 + 128 \times 64 + 64 \times 10 = 100\,352 + 8\,192 + 640$. d) Première convolution en $14 \times 14$ : $14 \times 14 \times 30 \times 25 = 147\,000$ ; seconde en $7 \times 7$ : $7 \times 7 \times 16 \times 30 \times 9 = 211\,680$ ; ensemble, 358 680, quatre fois moins que $588\,000 + 846\,720 = 1\,434\,720$. e) $(1\,434\,720 + 109\,184)/(358\,680 + 109\,184) = 1\,543\,904/467\,864 = 3{,}2999$. f) $30 \times 14 \times 14 = 5\,880$ valeurs de sortie, 3 comparaisons chacune. g) $147/36 = 4{,}08$.
**Dans ta copie** : le rapport des durées (4,1) dépasse celui des MAC (3,3), parce que le temps ne se résume pas aux multiplications : la version à pooling manipule des cartes quatre fois plus grandes **après** sa première convolution (ReLU, dropout et son masque, pooling, écriture et lecture en mémoire de tenseurs de $30 \times 28 \times 28$ valeurs), et la passe arrière refait le même trajet ; le pooling garde aussi la position de chaque maximum pour la passe arrière. Sur un CPU, déplacer des données coûte souvent autant que calculer.
**Erreurs fréquentes** : calculer la seconde convolution en $28 \times 28$ (elle vient après le pooling) ; oublier $C_{in}$ ; diviser par 4 tout le réseau, couches denses comprises ; arrondir e) à 3,29.
**Variante** : sur une image CIFAR-10 ($3 \times 32 \times 32$), la première convolution de 30 filtres $5 \times 5$ avec `padding=2` ferait $32 \times 32 \times 30 \times 3 \times 25 =$ **2 304 000** MAC en stride 1, et **576 000** en stride 2.

### Ex 24.4 — Ce que renvoie nn.LSTM : output, h_n et c_n ✏️
**Réponses** : a) **« (8, 5, 4) »** ; b) **« (2, 8, 4) »** ; c) **A** ; d) **« (8, 5, 8) »** ; e) **« (4, 8, 4) »** ; f) **304** ; g) **« (8, 5, 1) »** ; h) **« (5, 8, 3) »**.
**Démarche** : a) $(N, T, H)$ : la sortie de la dernière couche à chaque pas. b) $(n_{couches} \cdot n_{dir}, N, H)$ : l'état final de chaque couche. c) `h_n[0]` est l'état final de la première couche, de forme $(N, H) = (8, 4)$ ; `output[:, 0]`, la sortie du premier pas pour chaque exemple, a la même forme ; `output[0]` est la séquence du premier exemple, $(5, 4)$ ; `c_n` a la forme de `h_n` tout entier, $(2, 8, 4)$ ; `output[:, :, 0]` est $(8, 5)$. Même forme ne veut pas dire mêmes valeurs : c'est `output[:, -1]` qui est égal à `h_n[-1]` (le dernier pas de la dernière couche). d) Deux sens : leurs sorties côte à côte, $2H = 8$. e) $2 \times 2 = 4$ lignes : couche 0 aller, couche 0 retour, couche 1 aller, couche 1 retour. f) Première couche : $4H(D + H) + 2 \times 4H = 16 \times 7 + 32 = 144$ ; seconde couche, qui reçoit $H = 4$ valeurs : $16 \times 8 + 32 = 160$ ; total 304. g) `nn.Linear(4, 1)` remplace la dernière dimension. h) Sans `batch_first`, PyTorch attend $(T, N, D)$.
**Dans ta copie** : les 3 features n'apparaissent nulle part parce qu'elles ne servent qu'à l'intérieur de la cellule (elles sont multipliées par `weight_ih`) : la cellule rend toujours $H$ valeurs, quel que soit le nombre de features. En exécutant `lstm(torch.zeros(8, 5, 3))`, on retrouve les formes de a) et b), et `sum(p.numel() for p in lstm.parameters())` redonne f).
**Erreurs fréquentes** : écrire `h_n` en $(N, L, H)$ en croyant que `batch_first` le change ; confondre `output[0]` (le premier exemple) et `output[:, 0]` (le premier pas) ; compter un seul biais par couche, comme Keras (272 au lieu de 304) ; oublier que la seconde couche reçoit $H$ valeurs, pas $D$.
**Variante** : avec `nn.GRU(3, 4, num_layers=2, batch_first=True)`, trois blocs au lieu de quatre : $12 \times 7 + 24 = 108$ et $12 \times 8 + 24 = 120$, soit **228** paramètres ; les formes de `output` et de `h_n` sont les mêmes, et il n'y a pas de `c_n`.

### Ex 24.5 — Fenêtres, pas d'optimisation et durée : de 200 à 20 000 points ✏️
**Réponses** : a) **187** ; b) **19 987** ; c) **19 987** ; d) **625** ; e) **8,3** ; f) **1 008** ; g) **0,047**.
**Démarche** : a) Une fenêtre de 13 points plus sa cible occupe 14 points ; les débuts possibles vont de 0 à $200 - 14$, soit $200 - 13 = 187$ fenêtres (la formule de `make_windows` : $(T - w - h)/s + 1$ avec $h = s = 1$). b) $20\,000 - 13$. c) Un pas par exemple. d) $19\,987/32 = 624{,}6$, arrondi au-dessus. e) $100 \times 5 = 500$ s $= 8{,}3$ min. f) $28 \times 3\,600 = 100\,800$ s pour 100 epochs, soit 1 008 s par epoch. g) $5 \times 187/19\,987 = 0{,}0468$ s.
**Dans ta copie** : les deux affirmations sont incompatibles : 100 epochs de 5 s font 8 minutes, pas 28 heures ; 28 heures supposeraient 17 minutes par epoch. La phrase suivante du livre tranche : chaque epoch prenait environ dix fois plus de temps que sur 2 000 points, elle-même dix fois plus que sur 200 ; avec des batches d'un exemple, 5 s pour 20 000 pas d'un LSTM minuscule (quelques centaines de paramètres) reste vraisemblable ; 1 008 s, soit 50 ms par pas, ne l'est guère. La durée totale est donc probablement fausse (peut-être confondue avec une autre expérience).
**Erreurs fréquentes** : compter $T - w + 1$ fenêtres (en oubliant la place de la cible) ; arrondir d) au plus proche (624) ; convertir 500 s en 8,20 minutes (lire 8 min 20 s comme 8,20).
**Variante** : avec une fenêtre de 5 points et 2 000 points, **1 995** exemples, soit $\lceil 1\,995/32 \rceil =$ **63** pas par epoch avec des batches de 32.

### Ex 24.6 — Dropout inversé et max-norm : espérance et projection ∂
**Réponse** :
1. $\mathbb{E}[\tilde{h}] = \frac{h}{1-p}\,\mathbb{E}[m] = \frac{h}{1-p}\,(1 - p) = h$.
2. $m^2 = m$, donc $\mathbb{E}[\tilde{h}^2] = \frac{h^2}{(1-p)^2}(1 - p) = \frac{h^2}{1 - p}$, et $\text{Var}(\tilde{h}) = \frac{h^2}{1-p} - h^2 = \frac{p}{1-p}\,h^2$. Pour $p = 0{,}5$ : $h^2$ (un écart-type égal à $|h|$). Quand $p \to 1$, la variance tend vers l'infini : la valeur gardée, rare, est divisée par un nombre minuscule.
3. Version d'origine : à l'entraînement $m\,h$, d'espérance $(1 - p)\,h$ ; en évaluation $(1 - p)\,h$ : c'est la même valeur. PyTorch préfère diviser à l'entraînement : en évaluation, le module ne fait rien, le modèle exporté n'a pas à connaître $p$, et changer $p$ ne change pas l'échelle des activations en évaluation.
4. $\mathbb{E}[z] = w_1\mathbb{E}[\tilde{h}_1] + w_2\mathbb{E}[\tilde{h}_2] = w_1 h_1 + w_2 h_2$, la valeur d'évaluation, par linéarité. Avec $h_1 = 1$, $h_2 = -1$, $p = 0{,}5$, $z = 2(m_1 - m_2)$ prend les valeurs $0, -2, 2, 0$ avec la probabilité $1/4$ chacune ; $\mathbb{E}[\text{ReLU}(z)] = (0 + 0 + 2 + 0)/4 = 0{,}5$, alors que la valeur d'évaluation est $\text{ReLU}(0) = 0$. Après une non-linéarité, le réseau en évaluation ne calcule donc pas la moyenne exacte des sous-réseaux : c'est une **approximation** (Srivastava et coll. parlent de *weight scaling inference rule*), très bonne en pratique.
5. Si $\lVert\mathbf{w}\rVert \le c$, $\mathbf{w}' = \mathbf{w}$ est dans la boule, à distance 0. Sinon, pour tout $\mathbf{v}$ de la boule, l'inégalité triangulaire donne $\lVert\mathbf{w}\rVert \le \lVert\mathbf{w} - \mathbf{v}\rVert + \lVert\mathbf{v}\rVert \le \lVert\mathbf{w} - \mathbf{v}\rVert + c$, donc $\lVert\mathbf{w} - \mathbf{v}\rVert \ge \lVert\mathbf{w}\rVert - c$. Or $\mathbf{w}' = \frac{c}{\lVert\mathbf{w}\rVert}\mathbf{w}$ est dans la boule (de norme $c$) et $\lVert\mathbf{w} - \mathbf{w}'\rVert = (1 - \frac{c}{\lVert\mathbf{w}\rVert})\lVert\mathbf{w}\rVert = \lVert\mathbf{w}\rVert - c$ : il atteint la borne, c'est donc le point le plus proche.
6. $\lVert\mathbf{w}\rVert = \sqrt{9 + 16 + 144} = 13 > 3$, donc $\mathbf{w}' = \frac{3}{13}(3, 4, 12) \approx (0{,}692 ;\ 0{,}923 ;\ 2{,}769)$, et $\mathbf{w} - \mathbf{w}' = \frac{10}{13}\mathbf{w}$, de norme $10 = 13 - 3$, comme l'annonçait la question 5. Pour une convolution, `weight[o, c, u, v]` relie la position $(u, v)$ du canal d'entrée $c$ au canal de sortie $o$ : tous les poids du **filtre** `weight[o]` (ses $C_{in}\,k\,k$ valeurs) servent à calculer chaque valeur du canal $o$. On applique donc la max-norm à chaque filtre entier : `weight.copy_(torch.renorm(weight, p=2, dim=0, maxnorm=c))`, car `renorm` traite chaque sous-tenseur `weight[o]` le long de la dimension 0, quelle que soit sa forme (le `MaxNorm` de Keras fait de même avec `axis=[0, 1, 2]` sur son noyau `(k, k, C_in, C_out)`). Sous `torch.no_grad()`, parce que c'est une modification directe des poids, que l'autograd ne doit pas enregistrer (et qu'il refuserait sur une feuille qui demande un gradient) ; et après `optimizer.step()`, pour que la contrainte tienne après chaque mise à jour.

**Démarche** : tout repose sur deux propriétés de la variable de Bernoulli ($\mathbb{E}[m] = \mathbb{E}[m^2] = 1 - p$) et sur la linéarité de l'espérance, qui ne passe pas à travers une ReLU. Pour la projection, on cherche une borne inférieure de la distance, puis un point qui l'atteint.
**Erreurs fréquentes** : écrire $\mathbb{E}[m] = p$ (c'est la probabilité de **mettre à zéro**) ; oublier de retrancher $\mathbb{E}[\tilde{h}]^2$ ; croire que la linéarité de l'espérance traverse la ReLU ; appliquer `renorm` sur `dim=1` (les poids qui **partent** d'un canal d'entrée, vers tous les filtres).
**Variante** : pour $p = 0{,}2$, $\text{Var}(\tilde{h}) = \frac{0{,}2}{0{,}8}h^2 =$ **0,25** $h^2$, un écart-type de $0{,}5\,|h|$ ; dans l'exemple de la question 4 avec $p = 0{,}2$, $z = 1{,}25\,(m_1 - m_2)$ et $\mathbb{E}[\text{ReLU}(z)] = 1{,}25 \times 0{,}8 \times 0{,}2 =$ **0,2**.

### Ex 24.7 — Fermi : le budget d'une recherche d'hyperparamètres sur CPU, sur GPU et en précision mixte 🧮
**Réponses** : a) **150** ; b) **50** ; c) **108** ; d) **90** ; e) **0,45** ; f) **10,8** ; g) **25**.
**Démarche** : a) $25 \times 10^6 \times 2 \times 3 = 1{,}5 \times 10^8$ opérations par image ; $\times 50\,000 \times 20 = 10^6$ passages d'images : $1{,}5 \times 10^{14}$. b) $1{,}5 \times 10^{14} / 5 \times 10^{10} = 3\,000$ s $= 50$ min. c) $4 \times 3 \times 3 = 36$ candidats, $\times 3$ folds. d) $108 \times 50 = 5\,400$ min $= 90$ h. e) $1{,}5 \times 10^{14}/10^{13} = 15$ s par entraînement, $\times 108 = 1\,620$ s $= 0{,}45$ h. f) $1\,620/2{,}5 = 648$ s $= 10{,}8$ min. g) $30 \times 50$ min $= 1\,500$ min $= 25$ h.
**Dans ta copie** : avec un CPU et un week-end (une cinquantaine d'heures), la grille avec validation croisée (90 h) ne tient pas ; une recherche au hasard d'une trentaine de réglages sur un seul jeu de validation (25 h), voire sur un sous-ensemble des données et moins d'epochs pour éliminer d'abord les mauvais réglages, tient, avec de la marge pour réentraîner les deux ou trois meilleurs avec plusieurs graines. Un seul jeu de validation rend chaque score plus bruité (une seule découpe) et peut favoriser un réglage chanceux ; on compense en revalidant les finalistes. Les vraies durées seront plus longues : le débit soutenu est rarement atteint (lecture des données, augmentation, couches peu efficaces, mémoire), et l'évaluation de chaque epoch s'ajoute.
**Erreurs fréquentes** : oublier le facteur 2 (une MAC fait deux opérations) ou le facteur 3 (passe arrière) ; oublier les folds en c) ; confondre milliards ($10^9$) et milliers de milliards ($10^{12}$) ; convertir 648 s en 6,48 minutes.
**Variante** : avec un successive halving qui évalue les 36 candidats sur 1 epoch au lieu de 20, garde le tiers des meilleurs pour 5 epochs, puis les 4 meilleurs pour 20 epochs, sur un seul jeu de validation et sur le CPU : $(36 \times 1 + 12 \times 5 + 4 \times 20) \times 2{,}5$ min $= 176 \times 2{,}5 =$ **440** minutes, environ 7 h 20 (une epoch coûte $50/20 = 2{,}5$ min).

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 24.8 — L'augmentation de données en cinq lignes 🗣️
**Réponse modèle** : « Pour apprendre à un enfant à reconnaître un chat, on ne lui montre pas une seule photo prise de face : il en voit de près, de loin, tourné à gauche, à droite, au soleil, à l'ombre, et il comprend que c'est toujours un chat. L'augmentation de données fait la même chose avec les photos qu'on a déjà : à chaque passage, le programme les décale un peu, les retourne en miroir, change un peu la lumière, et le modèle apprend que ces détails ne changent pas la réponse. Le nom sur la photo reste le même. On ne touche pas aux photos qui servent à le noter, qui doivent rester réelles. Et on évite les transformations impossibles : un chat la tête en bas, ou un chiffre retourné en miroir, qui n'est plus le même chiffre. »
**Ce qui compte** : des vues différentes du même objet ; le label inchangé ; une transformation nouvelle à chaque passage ; seulement le train ; la plausibilité, avec un exemple à éviter.
**Erreurs fréquentes** : dire qu'on « ajoute des données » sans dire qu'elles sont fabriquées à partir des anciennes ; oublier que le label ne change pas ; augmenter aussi le test ; proposer des transformations qui changent la classe (un 6 qui devient un 9).
**Variante** : en une phrase pour un chef de projet : « On montre au modèle chaque photo sous plusieurs angles et éclairages, pour qu'il apprenne l'objet plutôt que la photo. »

### Ex 24.9 — Une seule graine suffit-elle ? Reproductibilité et honnêteté des comparaisons d'hyperparamètres ⚖️
1. A : moyenne **86,90 %**, écart-type **0,37** point ; B : moyenne **87,04 %**, écart-type **0,21** point. L'écart des moyennes (0,14 point, en faveur de B) est plus petit que la dispersion d'une graine à l'autre ; les deux intervalles se recouvrent largement. La conclusion du rapport ne tient pas : avec une graine chacun, l'écart de 0,5 point était du bruit, et rien ne permet de dire que A est meilleur (ni que B l'est).
2. L'initialisation des poids, l'ordre des mini-batches (le mélange du `DataLoader`), les masques du dropout, les tirages de l'augmentation de données, souvent le découpage train/validation, et parfois des opérations non déterministes du GPU (sommes dans un ordre variable).
3. C'est le **biais d'optimisme** du gagnant (ch. 8) : chaque score est la vraie valeur plus un bruit (ici de l'ordre de 0,3 point) ; le plus haut de 60 scores est presque toujours un réglage qui a eu de la chance. Le maximum de 60 bruits gaussiens d'écart-type 0,3 dépasse leur moyenne d'environ $2{,}3 \times 0{,}3 \approx 0{,}7$ point. Avant d'annoncer, on réentraîne le réglage retenu avec plusieurs nouvelles graines (et l'on compare aux deux ou trois suivants de la même façon).
4. Le test a servi à **choisir** : il fait partie de l'entraînement au sens large, et sa valeur est biaisée vers le haut, pour la même raison qu'en 3. Il faut un jeu de validation pour choisir, et un test regardé une seule fois, à la fin ; si le test a déjà servi, il faut le dire, ou en constituer un nouveau.
5. Strubell et coll. ont estimé, pour une recherche d'architecture d'un Transformer, des émissions de l'ordre de 284 tonnes d'équivalent CO₂, environ cinq fois celles d'une voiture sur toute sa vie. Ils recommandent de publier le temps d'entraînement et la sensibilité aux hyperparamètres. Un rapport honnête dit combien de runs ont été lancés, avec quel matériel et combien d'heures ; sans cela, un lecteur ne peut pas savoir si le gain annoncé vient d'une meilleure idée ou de cent essais de plus, ni juger s'il pourrait le reproduire.
6. Un protocole possible : (1) fixer avant de commencer la métrique, le jeu de validation et le budget ; (2) plusieurs graines (au moins 3 à 5) par réglage comparé, avec moyenne et écart-type ; (3) choisir sur la validation seulement, regarder le test une fois à la fin ; (4) une ligne par run dans le journal (date, commit, réglages complets, graine, métriques, durée, machine), runs ratés compris ; (5) annoncer le nombre total de runs et d'heures de calcul, et l'écart observé d'une graine à l'autre.

**Erreurs fréquentes** : comparer les moyennes sans regarder la dispersion ; croire qu'une graine fixe rend un résultat « vrai » (il est reproductible, pas représentatif) ; oublier le biais du maximum ; penser que regarder le test « juste une fois de temps en temps » est sans conséquence.
**Variante** : avec 3 graines seulement (les trois premières de chaque liste), A fait $86{,}97$ % et B $87{,}07$ % de moyenne : l'écart reste plus petit que la dispersion de chaque réglage (environ 0,4 et 0,2 point) ; plus de graines ne change pas la conclusion, elle la rend plus sûre.

### Ex 24.10 — Lire He et coll. (2016) : les réseaux résiduels 📄
1. Le problème : les réseaux plus profonds sont plus difficiles à entraîner. L'idée : faire apprendre à chaque groupe de couches une **fonction résiduelle** par rapport à son entrée. Sur ImageNet, les auteurs entraînent des réseaux résiduels jusqu'à **152 couches**, huit fois plus profonds que les VGG, mais moins coûteux ; un ensemble de ces réseaux atteint **3,57 %** d'erreur top-5 sur le test d'ImageNet et gagne la **1ʳᵉ place** de la tâche de classification d'ILSVRC 2015 (ainsi que des premières places en détection et en localisation, et sur COCO).
2. Sur CIFAR-10, le réseau ordinaire de 56 couches a une erreur plus haute que celui de 20 couches, **sur le test et sur l'entraînement**. L'overfitting serait une erreur d'entraînement basse et une erreur de test haute ; ici, le réseau profond n'arrive même pas à bien apprendre le train : c'est une difficulté d'**optimisation**, que les auteurs appellent la dégradation.
3. Si le bloc doit réaliser $H(\mathbf{x})$, on lui fait apprendre $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$ et l'on ajoute $\mathbf{x}$ à sa sortie. Si l'identité est la meilleure chose à faire, il suffit de pousser les poids de $F$ vers zéro ; faire rendre exactement son entrée à une pile de couches non linéaires (convolutions et ReLU) est bien plus difficile. Et un réseau plus profond peut alors au moins égaler un réseau moins profond : les couches en trop n'ont qu'à apprendre $F = 0$.
4. Quand les dimensions changent (plus de canaux, cartes plus petites), le raccourci ne peut pas être l'identité. Option (A) : l'identité, complétée par des canaux de zéros, avec un stride de 2 ; option (B) : une **projection** $W_s \mathbf{x}$, réalisée par une convolution $1 \times 1$ de stride 2. Le `ResidualBlock` de la fiche suit l'option (B) (convolution $1 \times 1$ et batchnorm sur le raccourci quand les formes diffèrent). Sur CIFAR-10, les auteurs utilisent l'option (A) partout : leurs réseaux résiduels ont exactement les paramètres des réseaux ordinaires.
5. Une première convolution $3 \times 3$ à 16 filtres, puis trois groupes de $2n$ couches : $32 \times 32$ avec 16 filtres, $16 \times 16$ avec 32, $8 \times 8$ avec 64 (le passage d'un groupe au suivant se fait par un stride de 2) ; puis un pooling moyen global et une couche dense à 10 sorties avec softmax : $6n + 2$ couches. Pour $n = 3$, **20** couches (environ 0,27 million de paramètres). Le réseau de 110 couches atteint 6,43 % d'erreur de test (le meilleur de 5 entraînements ; 6,61 % en moyenne). Le réseau de 1 202 couches s'entraîne sans difficulté (erreur d'entraînement presque nulle), mais son erreur de test est plus haute que celle du réseau de 110 couches : les auteurs l'attribuent à l'**overfitting**, un réseau de 19,4 millions de paramètres étant trop gros pour ce petit dataset.
6. La connexion résiduelle donne au gradient un chemin direct vers les premières couches, et à chaque bloc la possibilité de ne presque rien faire : on peut empiler beaucoup de blocs sans dégradation. Les Transformers en ont autour de chaque couche d'attention et de chaque couche dense ; sans elles, on n'entraînerait pas des modèles de dizaines de couches.

**Erreurs fréquentes** : dire que les réseaux profonds échouaient à cause de l'overfitting (la figure 1 montre le contraire) ; confondre le nombre de couches et le nombre de blocs ; croire que le raccourci contient des poids dans tous les cas ; attribuer l'échec du réseau de 1 202 couches à une difficulté d'optimisation.
**Variante** : pour $n = 9$, le réseau a $6 \times 9 + 2 =$ **56** couches, la profondeur du réseau ordinaire de la figure 1 ; dans leur tableau, la version résiduelle de 56 couches fait mieux que celle de 20, à l'inverse des réseaux ordinaires.

<a id="entretien"></a>

## 💼 Entretien

### 24.E1 — Comment organises-tu une recherche d'hyperparamètres avec un budget limité ?
**Réponse modèle en 60 secondes** : « D'abord, je chiffre le budget : combien dure un entraînement, donc combien d'essais tiennent dans deux jours. Je pars d'une référence simple qui marche, avec des réglages connus pour ce type de modèle, et je note chaque run dans un journal. Ensuite, je cherche d'abord ce qui compte le plus : le learning rate, sur une échelle logarithmique, puis la taille du modèle et la régularisation. Je préfère un tirage au hasard à une grille : à budget égal, il essaie plus de valeurs de chaque hyperparamètre, et en général seuls quelques-uns comptent. Pour économiser, je travaille d'abord sur un sous-ensemble des données et peu d'epochs, et j'élimine tôt les mauvais candidats, avec du successive halving ou Hyperband, ou Optuna qui le fait. Puis je zoome autour des meilleurs réglages. Enfin, je réentraîne les deux ou trois finalistes avec plusieurs graines, je choisis sur la validation, et je n'ouvre le test qu'une fois, à la fin. »
**Ce qui compte** : chiffrer le budget ; une référence ; l'ordre d'importance (learning rate d'abord) ; le hasard sur une échelle log ; sous-ensemble et élimination précoce ; le zoom ; plusieurs graines pour les finalistes ; le test à la fin ; un journal.
**Relances possibles** : « Pourquoi une échelle logarithmique pour le learning rate ? » (son effet dépend de son ordre de grandeur : passer de 0,001 à 0,002 change autant que de 0,01 à 0,02 ; tirer uniformément entre $10^{-4}$ et $10^{-1}$ passerait 90 % des essais au-dessus de 0,01) · « Qu'est-ce que le successive halving ? » (on lance beaucoup de candidats avec un petit budget, on garde la meilleure fraction, on lui donne plus de budget, et ainsi de suite ; `HalvingRandomSearchCV` dans scikit-learn, Hyperband et ASHA ailleurs) · « Et si deux réglages sont à égalité ? » (on regarde l'écart d'une graine à l'autre ; s'ils sont dans le bruit, on prend le plus simple, le plus rapide ou le plus stable).

### 24.E2 — Pourquoi l'augmentation de données améliore-t-elle la généralisation ? Donne des transformations à éviter
**Réponse modèle en 60 secondes** : « L'augmentation dit au modèle ce qui ne doit pas changer sa réponse : une voiture décalée de quelques pixels, retournée en miroir ou photographiée par temps gris reste une voiture. En transformant chaque image au hasard à chaque epoch, on lui montre beaucoup plus d'exemples différents que le dataset n'en contient, et il ne peut plus apprendre chaque image par cœur : c'est un régularisateur, souvent le plus efficace pour des images. Les transformations à éviter sont celles qui changent le label ou produisent des images impossibles : le miroir sur des chiffres ou du texte, un 6 tourné qui devient un 9, le haut et le bas inversés sur des photos, un changement de couleur quand la couleur est l'information, comme pour des feux de signalisation ou des maladies de la peau. On ne l'applique jamais à la validation ni au test, et on règle sa force comme un hyperparamètre ; aujourd'hui, des politiques comme RandAugment ou TrivialAugment, et des mélanges comme mixup et CutMix, sont très utilisés. »
**Ce qui compte** : l'invariance ; plus d'exemples différents, moins d'apprentissage par cœur ; la plausibilité avec des exemples concrets ; train seulement ; un hyperparamètre ; les politiques modernes.
**Relances possibles** : « Comment savoir si une augmentation est trop forte ? » (l'accuracy d'entraînement plafonne bas et celle de validation aussi : le modèle n'arrive plus à apprendre ; on la réduit, ou on entraîne plus longtemps) · « Qu'est-ce que mixup ? » (on mélange deux images avec un poids $\lambda$ et l'on mélange leurs labels avec le même poids : le modèle apprend des transitions douces entre classes) · « Et l'augmentation au moment du test ? » (*test-time augmentation* : on prédit sur plusieurs versions transformées d'une image de test et on moyenne les probabilités ; ce n'est pas de l'augmentation d'entraînement, et on le dit quand on publie un score).

### 24.E3 — Dropout ou batchnorm : lequel, où, et peut-on combiner les deux ?
**Réponse modèle en 60 secondes** : « Ils ne font pas la même chose. La batchnorm normalise les activations de chaque canal sur le mini-batch : elle a été conçue pour stabiliser et accélérer l'entraînement, et elle régularise un peu en plus, grâce au bruit des statistiques du mini-batch. Le dropout est un régularisateur pur : il éteint des valeurs au hasard pour que le réseau ne dépende d'aucune en particulier. Dans un CNN moderne, je mets une batchnorm après chaque convolution, avant l'activation, et du dropout surtout dans la tête, avant la dernière couche linéaire ; dans un Transformer, c'est plutôt la layer norm et du dropout. On peut combiner les deux, mais pas n'importe comment : un dropout placé juste avant une batchnorm fausse ses statistiques, apprises avec le dropout actif et utilisées sans lui en évaluation. Les deux changent de comportement entre `train()` et `eval()`. Je choisis en mesurant : l'écart entre train et validation dit s'il faut plus de régularisation. »
**Ce qui compte** : des buts différents ; les places habituelles ; le piège du dropout avant la batchnorm ; les deux modes ; choisir par la mesure.
**Relances possibles** : « Pourquoi la batchnorm régularise-t-elle un peu ? » (chaque exemple est normalisé avec les statistiques des autres exemples du mini-batch, qui changent d'un tirage à l'autre : un bruit qui joue le rôle d'un petit dropout) · « Que faire si les mini-batches sont minuscules ? » (les statistiques deviennent trop bruitées : on passe à la group norm ou à la layer norm, qui ne dépendent pas du batch) · « Le dropout sur une convolution ? » (`nn.Dropout2d` éteint des canaux entiers, plus efficace que d'éteindre des pixels voisins très corrélés).

### 24.E4 — Comment éviter la fuite de données dans une validation croisée avec prétraitement ?
**Réponse modèle en 60 secondes** : « Un score de validation croisée bien au-dessus du test, c'est le signe classique d'une fuite. Je cherche d'abord tout ce qui a été **appris sur l'ensemble des données avant le découpage** : la normalisation, l'imputation des valeurs manquantes, la sélection de features, un vocabulaire, un rééquilibrage des classes. La sélection de features est la plus traîtresse : choisir les features les plus corrélées au label sur tout le dataset, puis valider, donne des scores excellents même sur des données aléatoires. La règle : tout ce qui apprend va dans un `Pipeline`, et c'est le pipeline entier que la validation croisée réajuste dans chaque fold. Ensuite, je cherche les fuites qui ne viennent pas du prétraitement : des doublons entre folds, des groupes (plusieurs images d'un même patient, à garder dans le même fold avec `GroupKFold`), ou le temps (`TimeSeriesSplit` pour ne jamais prédire le passé avec le futur). Et le test reste à part jusqu'à la fin. »
**Ce qui compte** : le symptôme ; la liste de ce qui apprend ; la sélection de features ; le `Pipeline` réajusté dans chaque fold ; groupes, doublons et temps ; le test à part.
**Relances possibles** : « Comment prouver qu'il y a une fuite ? » (mélanger les labels : un pipeline honnête tombe au hasard ; si la sélection de features fuit, il garde un bon score ; ou comparer avec et sans l'étape suspecte dans le pipeline) · « Et la recherche d'hyperparamètres ? » (le score du meilleur candidat de `GridSearchCV` est optimiste, puisqu'il a servi à choisir ; une validation croisée imbriquée, ou un test à part, donne une estimation honnête) · « La normalisation d'images par 255 est-elle une fuite ? » (non : elle n'apprend rien sur les données ; seules les statistiques calculées sur les données comptent).

### 24.E5 — Batchnorm : que fait-elle, et pourquoi se comporte-t-elle différemment en train et en eval ?
**Réponse modèle en 60 secondes** : « Pour chaque canal, la batchnorm centre et réduit les activations avec la moyenne et l'écart-type du mini-batch, puis applique un facteur $\gamma$ et un décalage $\beta$ appris : le réseau garde la possibilité de choisir l'échelle qui lui convient, mais l'entraînement devient plus stable et accepte de plus grands learning rates. En entraînement, elle utilise les statistiques du mini-batch, et met à jour des moyennes glissantes. En évaluation, elle utilise ces moyennes glissantes : sinon, la prédiction d'une image dépendrait des autres images du batch, et une image seule, ou un petit batch, aurait des statistiques très différentes de celles du train. Les bugs classiques : oublier `model.eval()`, et les prédictions changent d'un batch à l'autre ; s'entraîner avec des batches de taille 1 ou 2, où les statistiques n'ont aucun sens ; geler une partie d'un réseau avec `requires_grad_(False)` mais la laisser en mode `train()`, ce qui modifie encore ses statistiques glissantes ; ou des données de test très différentes du train, que les statistiques glissantes normalisent mal. »
**Ce qui compte** : la normalisation par canal sur le mini-batch, $\gamma$ et $\beta$ ; les deux modes et leur raison ; au moins trois bugs concrets.
**Relances possibles** : « Pourquoi le biais de la convolution est-il inutile avant ? » (la batchnorm soustrait la moyenne du canal, biais compris, et $\beta$ le remplace) · « Que valent les statistiques glissantes ? » (une moyenne exponentielle : à chaque pas, $\mu_{\text{gliss}} \leftarrow (1 - \rho)\,\mu_{\text{gliss}} + \rho\,\mu_{\text{batch}}$, avec $\rho = 0{,}1$ par défaut dans PyTorch) · « Par quoi la remplace-t-on dans un Transformer ? » (par la layer norm, qui normalise chaque exemple sur ses features et se comporte de la même façon en entraînement et en évaluation).

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb` ; voici, pour chaque exercice, la réponse, la démarche, les erreurs fréquentes et une variante. Les chiffres sont ceux du mode rapide, sur CPU (PyTorch 2.11, 2 cœurs) : un autre processeur ou un GPU change les dernières décimales des résultats d'entraînement, et les durées. Les historiques longs (24.14, 24.21, 24.23, 24.30) et la recette du défi en mode complet viennent de `tools/chapters/history_ch24.py`.

### Ex 24.11 — Changer un seul hyperparamètre : la taille de batch 🔬
**Réponse** :
```python
EPOCHS_11 = 8 if FAST_MODE else 30
BATCH_SIZES_11 = (1024, 256, 64, 16)


def run_11(batch_size, lr=1e-3, epochs=EPOCHS_11, seed=2411):
    torch.manual_seed(seed)
    model = make_mlp((784,))
    loader = make_loader(XA_TR, YA_TR, batch_size=batch_size, seed=seed)
    return train_classifier(model, loader, XA_VAL, YA_VAL, epochs, lr=lr, seed=seed)


def summary_11(history):
    val = np.asarray(history["val_loss"])
    best = int(np.argmin(val))                            # an index, counted from 0
    return {"best_epoch": best + 1, "best_val_loss": float(val[best]),
            "final_gap": float(history["val_loss"][-1] - history["train_loss"][-1]),
            "final_val_acc": float(history["val_acc"][-1]), "sec_per_epoch": float(np.median(history["seconds"]))}
```
**Démarche** : le tableau de l'expérience (8 epochs sur 6 000 images, Adam à 0,001) :

| run | pas par epoch | meilleure epoch | écart final | val_acc finale | s par epoch (≈) |
|---|---|---|---|---|---|
| $B = 1\,024$ | 6 | 7 | 0,092 | 83,6 % | 0,2 |
| $B = 256$ | 24 | 8 | 0,144 | 85,0 % | 0,2 |
| $B = 64$ | 94 | 3 | 0,222 | 84,5 % | 0,35 |
| $B = 16$ | 375 | 3 | 0,250 | 84,0 % | 1,0 |
| $B = 1\,024$, lr × 4 | 6 | 7 | 0,115 | 84,3 % | 0,2 |

Les durées varient d'une exécution à l'autre ; leur ordre de grandeur, non.

**Dans tes notes** : les petits batches surapprennent le plus vite (meilleure epoch 3, écart final le plus grand) : à learning rate égal, ils font beaucoup plus de pas par epoch, chacun plus bruité, et vont donc plus loin en une epoch. Une epoch coûte d'autant plus cher que le batch est petit (chaque pas a un coût fixe) ; sur un GPU, l'écart serait encore plus grand, un gros batch remplissant la machine pour presque le même temps. Le cinquième run rattrape une partie du retard des batches de 1 024 sans atteindre ceux de 256 : la taille de batch et le learning rate sont couplés, et « changer un seul hyperparamètre » ne change pas qu'une seule chose.
**Erreurs fréquentes** : rendre `np.argmin` tel quel (l'epoch 1 devient 0) ; construire le modèle **avant** `torch.manual_seed(seed)` (deux appels donnent deux modèles différents) ; un loader sans graine, si bien que l'ordre des mini-batches change d'un run à l'autre ; la moyenne des durées au lieu de la médiane ; conclure sur la taille de batch en oubliant que le nombre de pas a changé aussi.
**Variante** : en mode complet (54 000 images), une epoch de mini-batches de 16 fait $\lceil 54\,000 / 16 \rceil =$ **3 375** pas, contre **53** pour des batches de 1 024.

### Ex 24.12 — Prédire le classement de quatre MLP (784, 784 + 784, 64, 32 + 32) 🔮
**Réponses** : a) **B** ; b) **False** ; c) **False** ; d) **True**. L'expérience (graine 2412) :

| modèle | paramètres | epoch du min | val_loss min | val_loss finale | val_acc finale |
|---|---|---|---|---|---|
| A (784) | 623 290 | 10 | 0,402 | 0,411 | 86,3 % |
| B (784, 784) | 1 238 730 | 9 | 0,409 | 0,475 | 86,3 % |
| C (64) | 50 890 | 14 | 0,428 | 0,441 | 84,3 % |
| D (32, 32) | 26 506 | 15 | 0,456 | 0,456 | 83,6 % |

**Démarche** : le modèle le plus large apprend 6 000 images le plus vite, atteint le premier le minimum de sa loss de validation, puis cette loss remonte (1,16 fois le minimum à la 15ᵉ epoch). Les petits modèles n'ont pas fini d'apprendre en 15 epochs : leur minimum tombe à la fin, et leur accuracy reste 2 à 3 points sous celle des grands. Les quatre réponses ont été vérifiées sur cinq graines (entrée `calib` de `tools/chapters/history_ch24.json`) : B est toujours le premier à son minimum (entre la 5ᵉ et la 9ᵉ epoch), C reste toujours sous A et D sous C, et le rapport de d) va de 1,16 à 1,28. Le meilleur des deux grands, lui, change d'une graine à l'autre : c'est pourquoi on ne le demande pas.
**Dans tes notes** : après son minimum, la loss de validation de B remonte, alors que son accuracy de validation bouge à peine : le modèle devient plus sûr de lui, y compris sur ses erreurs, ce que la loss punit et que l'accuracy ne voit pas. B a 615 440 paramètres de plus que A, et n'a rien gagné. Avec une seule graine, on peut écrire que les petits modèles sont derrière (2 à 3 points, bien plus que la variation d'une graine à l'autre), pas que A bat B.
**Erreurs fréquentes** : croire que plus de neurones donne toujours une meilleure accuracy ; juger l'overfitting sur l'accuracy seule ; oublier que D a moins de paramètres que C (deux fois moins : $784 \times 32$ contre $784 \times 64$ pour la première couche) ; comparer des modèles entraînés avec des réglages différents.
**Variante** : un MLP de deux couches de 128 neurones, `make_mlp((128, 128))`, a $785 \times 128 + 129 \times 128 + 129 \times 10 =$ **118 282** paramètres, cinq fois moins que A.

### Ex 24.13 — Dropout, max-norm et learning rate trop grand 📦
**Réponse** :
```python
EPOCHS_13 = 15 if FAST_MODE else 50


def max_norm_13(model, c=3.0):
    layers = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))][:-1]
    clipped = 0
    with torch.no_grad():                                 # a projection, not a learning step
        for layer in layers:
            norms = layer.weight.flatten(1).norm(dim=1)   # one norm per row (Linear) or per filter (Conv2d)
            clipped += int((norms > c).sum())
            layer.weight.copy_(torch.renorm(layer.weight, p=2, dim=0, maxnorm=c))
    return clipped


def run_13(p_drop=0.0, c=None, lr=1e-3, epochs=EPOCHS_13, seed=2413):
    torch.manual_seed(seed)
    model = make_mlp((32, 32), p_drop=p_drop)
    after = None if c is None else (lambda m: max_norm_13(m, c))
    history = train_classifier(model, make_loader(XA_TR, YA_TR, 256, seed), XA_VAL, YA_VAL, epochs, lr=lr, seed=seed,
                               after_step=after)
    return model, history
```
**Démarche** : sur le petit modèle de la vérification, la ligne $(3, 4, 12)$, de norme 13, devient $(3, 4, 12) \times 3/13$ ; la ligne $(0, 6)$ devient $(0, 3)$ ; les lignes plus courtes ne bougent pas ; la couche de sortie non plus. `flatten(1)` aligne un filtre de convolution entier sur une ligne, ce qui traite les deux types de couches d'un coup. L'expérience (15 epochs sur 6 000 images) :

| run | train_acc | val_acc | train_loss | val_loss | norme max |
|---|---|---|---|---|---|
| sans dropout | 86,9 % | 83,5 % | 0,359 | 0,454 | 1,615 |
| dropout 0,2 | 81,5 % | 82,6 % | 0,515 | 0,478 | 1,482 |
| dropout 0,2 + max-norm 3 | 81,5 % | 82,6 % | 0,515 | 0,478 | 1,482 |
| dropout 0,2 + max-norm 0,5 | 79,4 % | 81,1 % | 0,588 | 0,527 | 0,500 |
| dropout + max-norm 3, lr 0,01 | 82,5 % | 82,0 % | 0,467 | 0,483 | 3,000 |
| dropout + max-norm 3, lr 0,1 | 37,4 % | 48,9 % | 1,543 | 1,291 | 3,000 |

**Dans tes notes** : le dropout fait monter la loss d'entraînement plus que celle de validation et rapproche les deux. La max-norm de 3 ne change rien : aucune ligne n'atteint la norme 3 avec un learning rate de 0,001 (1,48 au plus), les deux runs sont identiques. Avec un learning rate de 0,1, Adam fait des pas de 0,1 sur des poids de l'ordre de 0,05 : l'entraînement s'effondre, et la max-norm, qui ramène sans cesse des lignes à la norme 3, ne suffit pas à le sauver. Le conseil « 10 à 100 fois plus grand » vaut pour SGD avec momentum, dont le pas est proportionnel au gradient (ch. 19).
**Erreurs fréquentes** : appliquer la max-norm à la couche de sortie ; `layer.weight = torch.renorm(...)` au lieu de `copy_` (PyTorch refuse d'affecter un tenseur ordinaire à un paramètre, et un `nn.Parameter` neuf ne serait plus celui que connaît l'optimiseur) ; oublier `torch.no_grad()` (PyTorch refuse de modifier sur place une feuille qui demande un gradient) ; normaliser les **colonnes** (`dim=1`) au lieu des lignes ; pour une convolution, normaliser chaque ligne de chaque noyau au lieu du filtre entier ; compter les lignes après la projection (il n'y en a plus aucune au-dessus de `c`).
**Variante** : la ligne $(6, 8, 0)$, de norme 10, avec $c = 5$ devient **$(3, 4, 0)$**, et `max_norm_13` compte **1** ligne ramenée.

### Ex 24.14 — Pourquoi la validation dépasse le train avec dropout 📈
**Réponses** : a) **30** ; b) **11,4** ; c) **0,08** ; d) **2,5** ; e) **2,0**.
**Démarche** :
```python
run_14 = np.asarray(HISTORY_14["running_acc"])
train_mode_14 = np.asarray(HISTORY_14["end_train_mode_acc"])
eval_mode_14 = np.asarray(HISTORY_14["end_eval_mode_acc"])
val_14 = np.asarray(HISTORY_14["val_acc"])
late_14 = slice(20, 30)                                   # epochs 21 to 30 (indices 20 to 29)
above_14 = int(np.sum(val_14 > run_14))
averaging_1_14 = round(100 * (train_mode_14[0] - run_14[0]), 1)
averaging_late_14 = round(100 * float(np.mean(train_mode_14[late_14] - run_14[late_14])), 2)
dropout_late_14 = round(100 * float(np.mean(eval_mode_14[late_14] - train_mode_14[late_14])), 1)
honest_gap_14 = round(100 * float(np.mean(eval_mode_14[late_14] - val_14[late_14])), 1)
print(above_14, averaging_1_14, averaging_late_14, dropout_late_14, honest_gap_14)
```
a) La validation est au-dessus de l'accuracy affichée pendant les 30 epochs. b) et c) L'écart dû à la moyenne pendant l'epoch : 11,4 points à la première epoch, quand le modèle change beaucoup ; 0,08 point à la fin. d) L'écart dû au dropout actif : 2,5 points. e) Mesurées toutes deux en mode évaluation, l'accuracy d'entraînement dépasse la validation de 2,0 points.
**Dans tes notes** : les deux raisons sont la moyenne prise pendant que les poids changent et le dropout actif pendant la mesure d'entraînement ; en fin d'entraînement, la seconde compte presque seule. Une fois mesurés de la même façon, le train dépasse la validation : le modèle ne généralise pas « mieux que parfait ». Pour juger l'overfitting d'un modèle à dropout, on trace le train évalué en mode évaluation contre la validation.
**Erreurs fréquentes** : prendre `end_eval_mode_acc` au lieu de `running_acc` en a) (la validation n'est **jamais** au-dessus du train mesuré dropout coupé) ; oublier de multiplier par 100 ; prendre la seule epoch 30 au lieu de la moyenne des epochs 21 à 30 (la tranche `[20:30]`) ; inverser l'ordre d'une différence.
**Variante** : à la 2ᵉ epoch, l'effet de la moyenne pendant l'epoch vaut **1,8** point : il fond dès que le modèle change moins vite.

### Ex 24.15 — Journal d'expériences : un run, une ligne 🛠️
**Réponse** :
```python
def git_commit_15():
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def _plain_15(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def log_run_15(path, params, metrics, seed):
    import datetime
    run = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "git": git_commit_15(),
           "seed": seed, "params": params, "metrics": metrics,
           "versions": {"python": platform.python_version(), "torch": torch.__version__}, "machine": platform.platform()}
    line = json.dumps(run, default=_plain_15, ensure_ascii=False)          # tuples are written as lists
    with open(path, "a", encoding="utf-8") as f:                           # "a": append, never overwrite
        f.write(line + "\n")
    return json.loads(line)


def read_runs_15(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def runs_table_15(runs):
    table = pd.json_normalize(runs, sep=".")
    keep = ["time", "git", "seed"] + [c for c in table.columns if c.startswith(("params.", "metrics."))]
    return table[[c for c in keep if c in table.columns]]


def best_run_15(runs, metric="val_acc", mode="max"):
    pick = max if mode == "max" else min
    return pick(runs, key=lambda run: run["metrics"][metric])
```
**Démarche** : le mode `"a"` ajoute à la fin du fichier ; `json.dumps(..., default=_plain_15)` appelle `_plain_15` pour chaque objet qu'il ne sait pas écrire (un `np.float32`, un `np.int64`) ; relire la ligne écrite (`json.loads`) rend exactement ce qu'on retrouvera plus tard, tuples devenus listes compris. `pd.json_normalize` aplatit les dictionnaires imbriqués en colonnes `params.lr`, `metrics.val_acc`. La cellule suivante note les cinq runs de 24.11 et les six de 24.13, avec tous leurs réglages.
**Dans tes notes** : manquent encore l'identité exacte des données (quelles images, quel découpage, une empreinte du fichier), les réglages laissés par défaut (les `betas` d'Adam, l'initialisation), la graine du mélange quand elle diffère de celle des poids, et le matériel (CPU ou GPU). Un fichier où l'on ajoute des lignes ne perd jamais un run, même quand deux entraînements écrivent l'un après l'autre ou qu'un run plante en route ; un fichier réécrit à chaque fois perd tout à la première erreur.
**Erreurs fréquentes** : ouvrir en mode `"w"` (chaque run efface les précédents) ; `json.dumps` sans `default` (`TypeError: Object of type float32 is not JSON serializable`) ; écrire le dictionnaire avec `str(run)` (des apostrophes, pas du JSON) ; l'heure locale sans fuseau ; oublier le `"\n"` (deux runs sur une ligne) ; ne noter que les réglages qui ont changé.
**Variante** : relancer deux fois la cellule d'expérience donne un journal de **11** lignes, pas 22 : elle efface le fichier avant de le remplir (`unlink`), comme un script qui repart de zéro ; dans un vrai projet, on ne l'efface jamais.

### Ex 24.16 — Un réseau PyTorch déguisé en estimateur scikit-learn 📦
**Réponse** :
```python
class TorchMLPClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, hidden=(32, 32), p_drop=0.2, lr=1e-3, epochs=5, batch_size=256, seed=0):
        self.hidden = hidden          # store, nothing else: clone() rebuilds from get_params()
        self.p_drop = p_drop
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def _network(self, n_in, n_out):
        layers = []
        for n in self.hidden:
            layers += [nn.Linear(n_in, n), nn.ReLU()]
            if self.p_drop > 0:
                layers.append(nn.Dropout(self.p_drop))
            n_in = n
        layers.append(nn.Linear(n_in, n_out))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)   # labels -> 0 .. K-1
        self.n_features_in_ = X.shape[1]
        torch.manual_seed(self.seed)                                           # the weights and the dropout masks
        self.model_ = self._network(X.shape[1], len(self.classes_))
        loader = make_loader(torch.from_numpy(X), torch.from_numpy(codes.astype(np.int64)), self.batch_size, self.seed)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.history_ = []
        for _ in range(self.epochs):
            self.model_.train()
            total = 0.0
            for xb, yb in loader:
                loss = loss_fn(self.model_(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            self.history_.append(total / len(X))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[-1]} features, the network was fitted with {self.n_features_in_}")
        self.model_.eval()
        with torch.inference_mode():
            logits = torch.cat([self.model_(torch.from_numpy(X[i:i + 4096])) for i in range(0, len(X), 4096)])
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
```
**Démarche** : `np.unique(y, return_inverse=True)` donne à la fois `classes_` (triées) et le code de chaque label, de 0 à 9, sur lequel apprend la cross-entropy ; `predict` revient aux labels par `self.classes_[...]`. Toute la construction est dans `fit` : `clone` et `set_params` n'ont rien à recalculer. La vérification entraîne une couche de 64 neurones 3 epochs sur des labels qui sont des noms de vêtements : 73,8 % sur la validation, des probabilités dont chaque ligne somme à 1, les mêmes à graine égale.
**Dans tes notes** : `clone` recrée l'estimateur en rappelant `__init__` avec `get_params()` ; un réseau construit dans `__init__` ne connaîtrait ni le nombre de features ni le nombre de classes, et un `set_params(hidden=...)` ne le reconstruirait pas. Si `fit` repartait du réseau précédent, le fold 2 commencerait avec des poids qui ont déjà vu ses exemples de validation (ils faisaient partie du train du fold 1) : une fuite, et des scores trop optimistes.
**Erreurs fréquentes** : `BaseEstimator` avant `ClassifierMixin` (`is_classifier` devient faux, sans message, et `cross_val_score` ne stratifie plus) ; calculer quoi que ce soit dans `__init__` ; `nn.CrossEntropyLoss` sur des labels qui sont des chaînes ; `predict` qui rend les codes 0 à 9 au lieu des noms ; oublier `return self` ; oublier `model_.eval()` dans `predict_proba` (le dropout reste actif) ; `torch.manual_seed` après la construction du réseau.
**Variante** : avec les réglages par défaut (`hidden=(32, 32)`) sur 784 features et 10 classes, `model_` a $785 \times 32 + 33 \times 32 + 33 \times 10 =$ **26 506** paramètres.

### Ex 24.17 — Validation croisée stratifiée avec normalisation dans un Pipeline 📦
**Réponse** :
```python
def fold_class_counts_17(cv, X, y):
    classes, codes = np.unique(np.asarray(y), return_inverse=True)
    return np.array([np.bincount(codes[val], minlength=len(classes)) for _, val in cv.split(X, y)])


def make_pipe_17(scaler, **net_params):
    return Pipeline([("scale", scaler), ("net", TorchMLPClassifier(**net_params))])


def max_after_scaling_17(scaler, X_fit, X_new):
    return float(np.max(clone(scaler).fit(X_fit).transform(X_new)))
```
**Démarche** : stratifiés, les cinq folds gardent chaque classe dans la même proportion (110 à 127 exemples de chaque classe par fold de validation, les classes du sous-ensemble n'ayant pas toutes 600 images) ; sur le fichier trié, `KFold` sans mélange donne des folds de validation de 2 à 4 classes. Le `MinMaxScaler` ajusté sur le train transforme certains pixels de validation jusqu'à 4,17. La validation croisée à 5 folds : pixels bruts 47,5 % ± 1,4 ; `MinMaxScaler` 65,5 % ± 0,9 ; division par 255 65,8 % ± 0,6 ; sur le fichier trié, sans mélange : 0 à 1,4 % par fold.
**Dans tes notes** : sur le fichier trié, chaque fold de validation contient des classes absentes, ou presque, de son entraînement : le réseau ne peut pas les prédire, d'où une accuracy presque nulle. Les valeurs au-dessus de 1 viennent des pixels du bord, dont le maximum appris sur le train est petit ; le score de validation croisée ne les révèle pas, ces pixels étant rares. Pour des images, une mise à l'échelle globale (diviser par 255) ; dans le pipeline, pour que le modèle livré applique exactement le prétraitement validé.
**Erreurs fréquentes** : compter les classes de la partie **entraînement** du fold ; `np.bincount` sans `minlength` (une classe absente raccourcit la ligne) ; ajuster le scaler sur toutes les données avant `cross_val_score` (une fuite) ; ajuster `scaler` lui-même au lieu d'une copie (`clone`), si bien que l'objet de l'appelant est modifié ; `make_pipeline`, dont les étapes ne s'appellent pas `scale` et `net`.
**Variante** : avec `KFold(n_splits=10)` sans mélange sur le fichier trié, chaque fold de validation contient **1 à 3** classes (`[1, 2, 2, 3, 1, 2, 2, 3, 2, 1]`) ; un fold d'une seule classe, absente de son entraînement, donne une accuracy nulle à coup sûr.

### Ex 24.18 — Grille, hasard, puis zoom : chercher les hyperparamètres 📦
**Réponse** :
```python
def count_fits_18(cv, param_grid=None, n_iter=None, refit=True):
    candidates = n_iter if param_grid is None else math.prod(len(values) for values in param_grid.values())
    return cv * candidates + (1 if refit else 0)


CV_18 = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def grid_search_18(pipe, X, y):
    grid = {"net__hidden": [(32,), (128,)], "net__lr": [1e-3, 1e-2]}
    return GridSearchCV(pipe, grid, cv=CV_18).fit(X, y)


def random_search_18(pipe, X, y, n_iter=6, seed=0):
    space = {"net__lr": loguniform(1e-4, 3e-2), "net__hidden": [(32,), (64,), (128,), (64, 64)],
             "net__p_drop": uniform(0, 0.5)}                  # uniform(loc, scale): from 0 to 0 + 0.5
    return RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=CV_18, random_state=seed).fit(X, y)


def zoom_values_18(best, factor=2.0, n=3):
    half = n // 2
    return [best * factor ** k for k in range(-half, half + 1)]


def results_18(search):
    table = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    return table.sort_values("rank_test_score").reset_index(drop=True)
```
**Démarche** : la vérification compte les `fit` réellement faits par un estimateur qui les compte : 13 pour une grille de 4 candidats à 3 folds (refit compris), 13 pour 4 tirages. Sur les 6 000 images, 3 epochs par entraînement : la grille trouve une couche de 128 avec un learning rate de 0,01 (80,3 %, écart-type entre folds 1,0 point), en 2 s ; le hasard trouve 80,7 % (une couche de 128, learning rate 0,0103, dropout 0,24) en 19 entraînements ; le zoom autour de 0,0103 (0,0051 ; 0,0103 ; 0,0206) garde 0,0103. Le modèle final fait 80,6 % sur les 6 000 images de validation.
**Dans tes notes** : le hasard fait un peu mieux que la grille pour 6 entraînements de plus, parce qu'il essaie six learning rates au lieu de deux ; l'écart entre les deux meilleurs candidats (0,4 point) est plus petit que l'écart-type entre folds (environ 1 point) : il faudrait plus de folds, plusieurs graines ou plus de données pour conclure. Le score de la recherche est le meilleur de plusieurs candidats, donc optimiste : seules des données que la recherche n'a jamais vues mesurent honnêtement le modèle choisi.
**Erreurs fréquentes** : `uniform(0.1, 0.5)` en croyant tirer entre 0,1 et 0,5 (c'est entre 0,1 et 0,6) ; `net_lr` avec un seul tiret bas (`ValueError: Invalid parameter`) ; oublier le refit dans le compte ; un `cv` différent pour la grille et le hasard (les scores ne se comparent plus) ; trier `results_18` par `mean_test_score` croissant ; un zoom linéaire ($\pm 0{,}005$) au lieu de logarithmique.
**Variante** : un `RandomizedSearchCV` de 20 tirages avec 5 folds et refit fait $5 \times 20 + 1 =$ **101** entraînements ; la grille de la fiche ($3 \times 2$ candidats, 3 folds), **19**.

### Ex 24.19 — Un CNN traduit du Keras qui refuse de tourner 🐛
**Réponse** :
```python
def make_cnn_19():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3),                              # 32 -> 30 ("valid")
        nn.ReLU(),                                        # bug 1: Keras's activation="relu" was lost
        nn.MaxPool2d(2),                                  # 30 -> 15
        nn.Conv2d(16, 32, 3, stride=2, padding=1),        # bug 2: "same" is refused with a stride of 2; 15 -> 8
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10),                        # bug 3: 8 x 8 positions, not 7 x 7
    )                                                     # bug 4: no Softmax, CrossEntropyLoss wants logits


def prepare_19(images):
    x = torch.from_numpy(images).float() / 255
    x = x.permute(0, 3, 1, 2)                             # bug 5: NHWC -> NCHW is a permute, never a reshape
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)            # bug 6: one value per CHANNEL (axis 1), not per column
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous()
```
**Démarche** : les trois exceptions arrivent l'une après l'autre : `ValueError: padding='same' is not supported for strided convolutions` à la construction ; puis, à la passe avant, `mat1 and mat2 shapes cannot be multiplied (2x2048 and 1568x10)` ($32 \to 30 \to 15 \to 8$ : $32 \times 8 \times 8 = 2\,048$ valeurs, pas $32 \times 7 \times 7 = 1\,568$) ; puis, dans `prepare_19`, `The size of tensor a (32) must match the size of tensor b (3) at non-singleton dimension 3` (le broadcasting aligne `MEAN` sur le dernier axe). Les trois erreurs silencieuses se voient à la liste des couches (la ReLU manquante, la softmax finale) et à la comparaison de `prepare_19` avec une préparation calculée autrement. Réparé, le modèle (25 578 paramètres) fait 40,1 % de validation après 2 epochs sur 5 000 images.
**Dans tes notes** : les erreurs bruyantes : le padding `"same"` avec un stride de 2, la taille de la couche dense, la forme de `MEAN`. Les silencieuses : la ReLU perdue (en imprimant le modèle, deux couches linéaires de suite), la softmax finale (des sorties qui somment à 1, et un entraînement qui stagne), le `reshape` (en affichant une image préparée : des bandes colorées au lieu d'une photo). La taille de la couche dense : $32$ (entrée) $\to 30$ (convolution $3 \times 3$ sans padding) $\to 15$ (pooling) $\to \lfloor (15 + 2 - 3)/2 \rfloor + 1 = 8$ (stride 2, padding 1).
**Erreurs fréquentes** : remplacer `"same"` par `padding=0` (la taille devient 7, et il faut alors $32 \times 7 \times 7$ : le modèle tourne, mais ce n'est plus celui de Keras) ; garder la softmax et passer à `nn.NLLLoss` sans `log` ; `x.transpose(1, 3)` (les axes H et W s'échangent : l'image est transposée) ; corriger `MEAN` par `torch.tensor(MEAN)[:, None, None]` sans vérifier qu'il s'aligne bien sur l'axe 1.
**Variante** : sur des images $64 \times 64$, le même modèle mène à $64 \to 62 \to 31 \to 16$, et la couche dense reçoit $32 \times 16 \times 16 =$ **8 192** valeurs.

### Ex 24.20 — Du CNN simple au CNN profond sur CIFAR-10 📦
**Réponses** : a) **330 122** ; b) **82 330** ; c) **82 554** ; d) **(1, 64, 4, 4)**.
**Démarche** :
```python
def make_simple_cnn_20():
    return nn.Sequential(nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 32 * 32, 10))


def make_deep_cnn_20(batchnorm=False, p_drop=0.0, widths=(16, 32, 64)):
    layers, c_in, side = [], 3, 32
    for width in widths:
        for c in (c_in, width):                           # two convolutions per stage
            layers.append(nn.Conv2d(c, width, 3, padding=1, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
        c_in, side = width, side // 2                     # each stage halves the height and the width
    layers += [nn.Flatten(), nn.Linear(c_in * side * side, 10)]
    return nn.Sequential(*layers)


def trace_shapes_20(model, x):
    model.eval()
    shapes = []
    with torch.inference_mode():
        for layer in model:
            x = layer(x)
            shapes.append((type(layer).__name__, tuple(x.shape)))
    return shapes
```
a) $32 \times (3 \times 25 + 1) = 2\,432$ pour la convolution, $32\,768 \times 10 + 10 = 327\,690$ pour la couche dense. b) Les six convolutions : $448 + 2\,320 + 4\,640 + 9\,248 + 18\,496 + 36\,928 = 72\,080$, plus $1\,024 \times 10 + 10 = 10\,250$. c) Sans les 224 biais des convolutions, avec deux paramètres par canal pour chacune des six batchnorms ($2 \times 224 = 448$) : $82\,330 - 224 + 448$ ; le dropout n'a pas de paramètres. d) Trois poolings : $32 / 2 / 2 / 2 = 4$. Après 2 epochs sur 5 000 images : simple 44,5 % de validation, profond 33,6 %, une epoch du profond durant environ deux fois plus longtemps.
**Dans tes notes** : 99 % des paramètres du CNN simple sont dans sa couche dense. Il est devant après deux epochs, parce que le profond, sans batchnorm, démarre lentement ; le profond fait trois fois et demie plus de multiplications-additions par image (9,9 millions contre 2,8). Doubler les canaux à chaque étage garde un coût de calcul du même ordre d'un étage à l'autre.
**Erreurs fréquentes** : `nn.Linear(64 * 8 * 8, 10)` (deux poolings comptés au lieu de trois) ; des convolutions avec biais devant la batchnorm ; un `nn.Dropout` après chaque convolution au lieu de chaque étage ; imbriquer des `nn.Sequential` par étage (le contrat veut une liste à plat, et `trace_shapes_20` ne verrait que trois blocs) ; oublier `model.eval()` dans `trace_shapes_20` (un dropout changerait les sorties, pas les formes, mais une batchnorm en mode entraînement sur une seule image modifierait ses statistiques glissantes).
**Variante** : `make_deep_cnn_20(widths=(8, 16, 32))` a **23 314** paramètres, trois fois et demie moins que les 82 330 du réglage par défaut : diviser les largeurs par 2 divise par 4 les poids des convolutions (72 080 → 18 184), mais seulement par 2 ceux de la couche dense (10 250 → 5 130).

### Ex 24.21 — Batchnorm, dropout ou les deux : accuracy et écart entre train et validation d'un CNN 🔬
**Réponses** : a) **D** ; b) **B** ; c) **22,4** ; d) **C**.
**Démarche** :
```python
VARIANTS_21 = {"plain": {}, "dropout": {"p_drop": 0.25}, "batchnorm": {"batchnorm": True},
               "both": {"batchnorm": True, "p_drop": 0.25}}


def run_21(variant, epochs=EPOCHS_C, seed=2421):
    torch.manual_seed(seed)
    model = make_deep_cnn_20(**VARIANTS_21[variant])
    return train_classifier(model, make_loader(XC_TR_T, YC_TR_T, 128, seed), XC_VAL_T, YC_VAL_T, epochs, seed=seed)


def gaps_21(history):
    return [100 * (t - v) for t, v in zip(history["train_acc"], history["val_acc"])]
```
```python
letters_21 = dict(zip("ABCD", ["plain", "dropout", "batchnorm", "both"]))
best_21 = max(letters_21, key=lambda k: HISTORY_21[letters_21[k]]["val_acc"][-1])
best_gap_21 = min(letters_21, key=lambda k: gaps_21(HISTORY_21[letters_21[k]])[-1])
plain_gap_21 = round(gaps_21(HISTORY_21["plain"])[-1], 1)
first_65_21 = {k: next((i for i, a in enumerate(HISTORY_21[v]["val_acc"]) if a >= 0.65), 99) for k, v in letters_21.items()}
fastest_21 = min(first_65_21, key=first_65_21.get)
print(best_21, best_gap_21, plain_gap_21, fastest_21)
```
Les deux epochs sur 5 000 images : validation 29,0 % (sans rien), 27,9 % (dropout), 42,2 % (batchnorm), 37,7 % (les deux). L'historique long, à la 20ᵉ epoch : sans rien, 88,8 % sur le train et 66,4 % en validation (22,4 points d'écart) ; dropout, 73,2 % et 71,0 % (2,2 points) ; batchnorm, 97,1 % et 69,1 % (28,0 points) ; les deux, 79,3 % et 76,7 % (2,6 points). La batchnorm atteint 65 % dès la 3ᵉ epoch, les deux ensemble à la 5ᵉ, le dropout à la 10ᵉ, le modèle sans rien à la 9ᵉ.
**Dans tes notes** : seule, la batchnorm accélère l'apprentissage mais ne régularise pas : elle surapprend même davantage ; le dropout réduit l'écart à 2 points au prix d'un départ plus lent ; les deux ensemble font mieux que chacun seul. Les deux premières epochs ne permettaient pas de deviner ce classement.
**Erreurs fréquentes** : comparer les variantes sur deux epochs ; confondre « meilleur écart » et « meilleure accuracy » ; oublier `* 100` dans `gaps_21` ; lire la 20ᵉ epoch à l'indice 20 (il n'existe pas : c'est l'indice 19, ou `-1`).
**Variante** : à la 10ᵉ epoch, l'écart de la variante `"both"` n'est que de **0,4** point : son overfitting arrive tard.

### Ex 24.22 — Prédire l'accuracy d'un CNN sur des images retournées, décalées, tournées ou en gris 🔮
**Réponses** : a) **False** ; b) **True** ; c) **True** ; d) **True** ; e) **True**. L'expérience (graine 2422, 2 000 images de test) : intactes 50,3 % ; miroir gauche-droite 50,0 % (−0,3 point) ; retournement haut-bas 33,3 % (−17,1) ; rotation de 90° 21,1 % (−29,2) ; décalage de 4 pixels 27,1 % (−23,2) ; niveaux de gris 43,3 % (−7,0).
**Démarche** : les réponses tiennent sur cinq graines (entrée `calib` de `tools/chapters/history_ch24.json`) : le miroir coûte de −0,2 à 0,9 point, le retournement 14 à 17, la rotation 25 à 29, le décalage 13 à 24, les niveaux de gris 4 à 10. Le train contient déjà des objets tournés à gauche et à droite, jamais le ciel en bas ; la couleur aide moins que l'orientation. La surprise est le décalage : un CNN entraîné sans augmentation n'est pas invariant par translation.
**Dans tes notes** : le miroir gauche-droite produit des images que le train contient déjà ; le retournement haut-bas et la rotation, jamais. Les poolings (une position sur deux gardée), les bords complétés de zéros et la bande noire, un bord net que le réseau n'a jamais vu, rendent le modèle sensible à un décalage de quelques pixels : c'est exactement ce que corrige le recadrage décalé de l'augmentation. Pour CIFAR-10 : le miroir gauche-droite et de petits décalages ; pas de retournement haut-bas ni de rotation de 90°.
**Erreurs fréquentes** : croire qu'un CNN est invariant par translation « par construction » (ses convolutions le sont, pas le réseau entier) ; juger une transformation sur une seule image ; confondre « plausible » (la question de l'augmentation) et « coûteux pour ce modèle » (ce qu'on mesure ici).
**Variante** : le décalage de 4 pixels laisse une bande noire de $32^2 - 28^2 =$ **240** pixels, soit 23 % de l'image : une part que le réseau n'a jamais vue noire.

### Ex 24.23 — Augmentation de données avec torchvision.transforms.v2 📦
**Réponses** : a) **4,8** ; b) **13** ; c) **18,0**.
**Démarche** :
```python
def make_transforms_23(augment=True):
    finish = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    random_ops = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()] if augment else []
    return v2.Compose([v2.ToImage(), *random_ops, *finish]), v2.Compose([v2.ToImage(), *finish])


class CifarImages_23(Dataset):

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


def channel_stats_23(images):
    x = np.asarray(images, dtype=np.float64) / 255.0
    return tuple(float(m) for m in x.mean(axis=(0, 1, 2))), tuple(float(s) for s in x.std(axis=(0, 1, 2)))


def make_train_loader_23(dataset, batch_size=128, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
```
```python
with_23, without_23 = HISTORY_23["with"], HISTORY_23["without"]
gain_23 = round(100 * (max(with_23["val_acc"]) - max(without_23["val_acc"])), 1)
above_23 = [w > wo for w, wo in zip(with_23["val_acc"], without_23["val_acc"])]
cross_23 = next(i + 1 for i in range(len(above_23)) if all(above_23[i:]))
train_gap_23 = round(100 * (without_23["train_acc"][-1] - with_23["train_acc"][-1]), 1)
print(gain_23, cross_23, train_gap_23)
```
La vérification compare chaque tirage aux 81 décalages possibles de l'image et de son miroir : tous sont l'un ou l'autre, jamais un retournement haut-bas. Les statistiques des 5 000 images d'entraînement (0,4907 ; 0,4813 ; 0,4447 pour les moyennes) sont proches de celles de la fiche. Après 2 epochs sur 5 000 images : 43,0 % de validation sans augmentation, 40,5 % avec. Sur l'historique long : meilleure validation 73,7 % sans augmentation, 78,6 % avec ; à la 25ᵉ epoch, 100,0 % d'entraînement sans augmentation, 82,0 % avec.
**Dans tes notes** : en deux epochs, le modèle n'a pas eu le temps de surapprendre : l'augmentation n'a rien à empêcher, elle rend seulement la tâche plus difficile. Avec augmentation, le train change à chaque epoch : le modèle ne peut pas l'apprendre par cœur, d'où une accuracy d'entraînement loin de 100 %. La graine du loader fixe l'ordre ; `torch.manual_seed` fixe les transformations tirées dans le processus principal ; avec des workers, il faut aussi le générateur du loader, dont chaque worker tire sa graine (et le même nombre de workers).
**Erreurs fréquentes** : appliquer l'augmentation à la validation ou au test ; placer `RandomCrop` après `Normalize` (le padding de zéros devient une valeur normalisée différente du noir) ; transformer toutes les images une fois dans `__init__` (chaque epoch voit alors les mêmes images) ; `v2.ToDtype(torch.float32)` sans `scale=True` (les pixels restent entre 0 et 255) ; `std` avec le diviseur $n - 1$ (accepté à cette précision près, mais pas celui du contrat).
**Variante** : `RandomCrop(32, padding=4)` puis `RandomHorizontalFlip()` peuvent produire $9 \times 9 \times 2 =$ **162** images différentes à partir d'une seule.

### Ex 24.24 — Des images synthétiques générées à la volée 📦
**Réponse** :
```python
_LINES_24 = {0: [[(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)]],
             1: [[(0.5, 0.2), (0.5, 0.8)], [(0.2, 0.5), (0.8, 0.5)]],
             2: [[(0.2, c), (0.8, c)] for c in (0.25, 0.5, 0.75)],
             3: [[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]],
             4: [[(0.5 + 0.3 * np.sin(t), 0.5 + 0.3 * np.cos(t)) for t in np.linspace(0, 2 * np.pi, 25)]]}


def draw_shape_24(rng, label=None, size=32):
    label = int(rng.integers(5)) if label is None else int(label)
    scale, shift = rng.uniform(0.8, 1.1), rng.uniform(-0.1, 0.1, size=2)
    image = np.zeros((size, size), dtype=np.float32)
    for line in _LINES_24[label]:
        points = np.asarray(line, dtype=float) + rng.normal(0.0, 0.03, size=(len(line), 2))
        if label == 4:
            points[-1] = points[0]                         # keep the circle closed
        points = ((points - 0.5) * scale + 0.5 + shift) * (size - 1)
        for p, q in zip(points[:-1], points[1:]):
            stroke_24(image, p, q, width=size / 32)
    return image, label


def worker_share_24(n, worker_id, num_workers):
    return n // num_workers + (1 if worker_id < n % num_workers else 0)


class ShapeStream_24(IterableDataset):

    def __init__(self, n_per_epoch, size=32):
        self.n_per_epoch, self.size = n_per_epoch, size

    def __iter__(self):
        rng = np.random.default_rng(int(torch.randint(2**31, ())))   # a new seed at each epoch, in each worker
        info = get_worker_info()                                      # None in the main process
        n = self.n_per_epoch if info is None else worker_share_24(self.n_per_epoch, info.id, info.num_workers)
        for _ in range(n):
            image, label = draw_shape_24(rng, size=self.size)
            yield torch.from_numpy(image).unsqueeze(0), label
```
**Démarche** : les cinq formes sont des lignes brisées en coordonnées unitaires ; le bruit déplace chaque point, puis l'agrandissement et le décalage déplacent toute la forme, et `stroke_24` trace chaque segment. La vérification : les cinq formes tirées à peu près autant (96 à 110 fois sur 500), la forme moyenne la plus proche donne le bon label 93 fois sur 100, et deux workers produisent 103 images toutes différentes. Le petit CNN (6 373 paramètres) reconnaît 97,4 % des dessins neufs après une epoch de 3 000 dessins, 100 % après trois.
**Dans tes notes** : le réseau ne voit jamais deux fois le même dessin : il ne peut pas apprendre le train par cœur. Avec `103 // 4`, une epoch ferait 100 exemples. Une graine fixée dans `__init__` redonnerait les mêmes images à chaque epoch et dans chaque worker.
**Erreurs fréquentes** : tirer la graine une fois pour toutes dans `__init__` ; `np.random.default_rng()` sans graine (le flux n'est plus reproductible avec `torch.manual_seed`) ; des coordonnées (x, y) au lieu de (ligne, colonne), qui transposent les formes (le Z devient un И) ; oublier de fermer le cercle ; rendre une image `float64` ou un label NumPy.
**Variante** : avec 1 000 exemples par epoch et 3 workers, `worker_share_24` donne **334, 333 et 333** ; la version de la fiche (`1000 // 3`) en ferait 999.

### Ex 24.25 — Sommes de sinus et LSTM minuscule : fenêtres, scaler, RMSE 📦
**Réponse** :
```python
def prepare_25(train_series, test_series, window):
    scaler = SeriesScaler().fit(np.asarray(train_series, dtype=float).reshape(-1, 1))   # min and max of the train only
    out = [scaler]
    for series in (train_series, test_series):
        scaled = scaler.transform(np.asarray(series, dtype=float).reshape(-1, 1)).ravel()
        X, y = make_windows(scaled, window=window)                                     # (N, w, 1) and (N,)
        out += [torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)]
    return tuple(out)


def fit_25(model, X, y, epochs=100, batch_size=16, lr=1e-2, seed=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = make_loader(X, y, batch_size, seed)
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        losses.append(total / len(X))
    return losses


def _to_units_25(scaler, values):
    return scaler.inverse_transform(np.asarray(values, dtype=float).reshape(-1, 1)).ravel()


def rmse_25(model, scaler, X, y):
    model.eval()
    with torch.inference_mode():
        pred = model(X).numpy()
    return float(np.sqrt(np.mean((_to_units_25(scaler, pred) - _to_units_25(scaler, y.numpy())) ** 2)))


def persistence_rmse_25(scaler, X, y):
    last = X[:, -1, 0].numpy()
    return float(np.sqrt(np.mean((_to_units_25(scaler, last) - _to_units_25(scaler, y.numpy())) ** 2)))
```
**Démarche** : 200 points et des fenêtres de 3 donnent 197 exemples. Un modèle qui prédirait toujours 0 (le minimum du train, dans l'échelle du scaler) aurait une RMSE de 3,137 sur le test calme ; la persistance, 0,171. L'expérience (100 epochs, mini-batches de 16, learning rate 0,01) :

| série | fenêtre | RMSE de test | persistance |
|---|---|---|---|
| calme | 1 | 0,188 | 0,170 |
| calme | 3 | 0,047 | 0,171 |
| calme | 5 | 0,038 | 0,171 |
| agitée | 1 | 0,893 | 0,886 |
| agitée | 3 | 0,129 | 0,884 |
| agitée | 5 | 0,129 | 0,882 |

**Dans tes notes** : le réseau bat nettement la persistance dès 3 points. Avec un seul point, il ne sait pas si la courbe monte ou descend, et fait un peu moins bien que la persistance. La RMSE dans les unités de la série se lit et se compare directement ; dans l'échelle du scaler, elle dépendrait du min et du max du train.
**Erreurs fréquentes** : ajuster le scaler sur le train **et** le test (une fuite) ; passer une série `(T,)` au scaler (il attend une colonne) ; `inverse_transform` sur les prédictions seulement, pas sur les cibles ; la RMSE dans l'échelle du scaler ; des cibles `(N, 1)` contre des prédictions `(N,)`, que `nn.MSELoss` diffuse en une matrice `(N, N)` (un avertissement, et une loss fausse) ; `y` en `float64` (la passe avant rend du `float64`, puis `backward()` lève `RuntimeError: Found dtype Double but expected Float`).
**Variante** : avec une fenêtre de 10 sur la série calme, le train de 200 points donne **190** fenêtres, et une epoch de mini-batches de 16 fait $\lceil 190/16 \rceil =$ **12** pas, contre 13 avec une fenêtre de 3.

### Ex 24.26 — Fenêtre, profondeur ou données : qu'est-ce qui aide vraiment ? 🔬
**Réponse** :
```python
STEPS_26 = 4_000 if FAST_MODE else 8_000           # optimizer steps of every run (the same compute for all)
TEST_START_26 = 30_000                             # the test: 200 points from point 30 000, far from every train


def run_26(n_train, window, sizes=(3,), steps=STEPS_26, seed=2426):
    train = upsloping(sum_of_sines(n_train, D_THETA, 0, **BUSY))
    test = upsloping(sum_of_sines(200, D_THETA, TEST_START_26, **BUSY))
    scaler, X_tr, y_tr, X_te, y_te = prepare_25(train, test, window)
    torch.manual_seed(seed)
    model = TinyForecaster(sizes[0]) if len(sizes) == 1 else Stacked(sizes)
    epochs = math.ceil(steps / math.ceil(len(X_tr) / 32))          # mini-batches per epoch: ceil(N / 32)
    fit_25(model, X_tr, y_tr, epochs=epochs, batch_size=32, lr=1e-2, seed=seed)
    return {"rmse": rmse_25(model, scaler, X_te, y_te), "persistence": persistence_rmse_25(scaler, X_te, y_te),
            "epochs": epochs}
```
**Démarche** : 4 000 pas par run, mini-batches de 32 :

| points | fenêtre | couches | epochs | RMSE de test | persistance |
|---|---|---|---|---|---|
| 200 | 5 | (3,) | 572 | 1,424 | 2,708 |
| 200 | 5 | (9, 6, 3) | 572 | 1,112 | 2,708 |
| 200 | 13 | (3,) | 667 | 1,287 | 2,668 |
| 200 | 13 | (9, 6, 3) | 667 | 1,074 | 2,668 |
| 2 000 | 5 | (3,) | 64 | 1,388 | 2,708 |
| 2 000 | 5 | (9, 6, 3) | 64 | 0,943 | 2,708 |
| 2 000 | 13 | (3,) | 64 | 1,096 | 2,668 |
| 2 000 | 13 | (9, 6, 3) | 64 | 0,848 | 2,668 |

Sur deux autres graines (entrée `calib` de `tools/chapters/history_ch24.json`), le meilleur réglage reste les trois couches avec 2 000 points (0,80 à 1,03), et la profondeur aide dans presque tous les cas ; à 1 500 pas, les écarts se noyaient dans la variation d'une graine à l'autre.
**Dans tes notes** : le levier qui fait le plus baisser la RMSE est la **profondeur** : de 0,21 à 0,45 de moins dans les quatre paires, contre environ 0,14 en moyenne pour la fenêtre et 0,16 pour les données. Aucun levier ne suffit seul : c'est le réseau profond qui profite le plus de données (fenêtre de 5 : les trois couches font passer de 1,42 à 1,11 avec 200 points, de 1,39 à 0,94 avec 2 000) ; le petit réseau de 3 valeurs d'état en profite peu. La fenêtre de 13 a besoin d'exemples variés pour que le réseau apprenne à se servir de ses premiers points au lieu de les retenir par cœur. Avec une graine, on n'interprète que les écarts de plusieurs dixièmes qui se répètent.
**Erreurs fréquentes** : `steps // ceil(N / 32)` au lieu d'un arrondi supérieur (moins de pas que demandé) ; compter les pas avec le nombre de **points** au lieu du nombre de fenêtres ; construire le modèle avant `torch.manual_seed` ; un test qui chevauche le train (avec 20 000 points, un test à partir du point 400 serait dans le train : d'où `TEST_START_26 = 30_000`).
**Variante** : en mode complet, 20 000 points et une fenêtre de 13 donnent 19 987 fenêtres, soit 625 mini-batches de 32 par epoch : les 8 000 pas demandent $\lceil 8\,000/625 \rceil =$ **13** epochs.

### Ex 24.27 — Séquences retournées, état porté entre batches, couche par pas de temps 📦
**Réponses** : a) **(4, 9, 6)** ; b) **(3, 4, 6)** ; c) **(4, 9, 12)** ; d) **(4, 4, 6)** ; e) **8**.
**Démarche** :
```python
def lstm_shapes_27(lstm, x):
    output, (h_n, c_n) = lstm(x)
    return {"output": tuple(output.shape), "h_n": tuple(h_n.shape), "c_n": tuple(c_n.shape)}


def last_steps_27(lstm, x):
    output, (h_n, _) = lstm(x)
    if lstm.bidirectional:                                 # h_n: (2 L, N, H), the two directions of each layer
        return output[:, -1, :lstm.hidden_size], h_n[-2]   # h_n[-1] is the backward direction of the last layer
    return output[:, -1], h_n[-1]


def to_streams_27(series, n_streams, seq_len):
    series = np.asarray(series, dtype=np.float32)
    L = (len(series) - 1) // n_streams
    n_chunks = L // seq_len
    streams = np.stack([series[b * L:b * L + L + 1] for b in range(n_streams)])      # (n_streams, L + 1)
    inputs = streams[:, :n_chunks * seq_len].reshape(n_streams, n_chunks, seq_len)
    targets = streams[:, 1:n_chunks * seq_len + 1].reshape(n_streams, n_chunks, seq_len)
    X = torch.from_numpy(np.ascontiguousarray(inputs.transpose(1, 0, 2))).unsqueeze(-1)
    Y = torch.from_numpy(np.ascontiguousarray(targets.transpose(1, 0, 2)))
    return X, Y


def run_carried_27(lstm, X, carry=True):
    outputs, state = [], None
    with torch.inference_mode():
        for chunk in X:
            output, new_state = lstm(chunk, state if carry else None)
            outputs.append(output)
            state = new_state
    return torch.stack(outputs)


def train_carried_27(lstm, head, X, Y, epochs, lr=1e-2):
    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        lstm.train()
        state, total = None, 0.0                             # zero state at the start of each epoch
        for xb, yb in zip(X, Y):                             # chunk k + 1 continues chunk k, row by row
            if state is not None:
                state = tuple(s.detach() for s in state)     # keep the values, cut the graph (truncated BPTT)
            output, state = lstm(xb, state)
            loss = loss_fn(head(output).squeeze(-1), yb)     # one shared Linear at every step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total / len(X))
    return losses
```
a) et b) `output` garde la dernière couche, à chaque pas, l'axe des exemples en premier ; `h_n` garde chaque couche, au dernier pas, les exemples au milieu. c) et d) Bidirectionnel : les deux sens côte à côte dans `output` ($2H = 12$), une ligne par couche et par sens dans `h_n` ($2 \times 2 = 4$). e) $L = \lfloor 3\,331 / 16 \rfloor = 208$, puis $\lfloor 208 / 24 \rfloor = 8$ morceaux. L'état porté donne exactement les sorties d'une seule passe sur chaque flux ; remis à zéro, seul le premier morceau est identique. Sur les taches solaires, après 60 epochs : une erreur de 0,0033 aux mois 1 à 3 de chaque morceau avec l'état porté, 0,0053 sans ; à partir du 13ᵉ mois, 0,0038 dans les deux cas.
**Dans tes notes** : un LSTM ne fait que transformer son état pas après pas : porter l'état d'un morceau au suivant, c'est faire le même calcul en plusieurs fois. Ranger la série en flux parallèles garantit que la ligne $b$ d'un batch continue la ligne $b$ du précédent. Sans état, le modèle se trompe surtout au début de chaque morceau, faute de contexte. Le `detach()` coupe les gradients : le modèle garde la mémoire au-delà de 24 mois, mais n'apprend pas à s'en servir.
**Erreurs fréquentes** : `h_n[-1]` pour un LSTM bidirectionnel (c'est le sens arrière, qui a lu la séquence à l'envers) ; `output[:, 0]` (le premier pas) ; découper la série en morceaux consécutifs rangés dans le même batch ; oublier `detach()` (`RuntimeError: Trying to backward through the graph a second time`) ; détacher **avant** le premier morceau un état `None` ; remettre l'état à zéro à chaque morceau.
**Variante** : `nn.LSTM(3, 7, num_layers=4, batch_first=True, bidirectional=True)` sur une entrée `(5, 11, 3)` rend `output` de forme **(5, 11, 14)** et `h_n` de forme **(8, 5, 7)**.

### Ex 24.28 — Au-delà de nn.Sequential : branches, addition, couches partagées, gel 📦
**Réponses** : a) **78 042** ; b) **308 650** ; c) **72 906**.
**Démarche** :
```python
class TwoBranch_28(nn.Module):

    def __init__(self, c_in, c_out, merge="add"):
        super().__init__()
        if merge not in ("add", "cat"):
            raise ValueError(f"merge must be 'add' or 'cat', got {merge!r}")
        self.wide = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.narrow = nn.Conv2d(c_in, c_out, 1)
        self.merge = merge

    def forward(self, x):
        a, b = self.wide(x), self.narrow(x)              # two branches, the same input
        merged = a + b if self.merge == "add" else torch.cat([a, b], dim=1)
        return torch.relu(merged)


class Siamese_28(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder                           # one module, called twice: its weights are shared

    def forward(self, a, b):
        return ((self.encoder(a) - self.encoder(b)) ** 2).flatten(1).sum(dim=1)


def make_resnet_28(widths=(16, 32, 64), n_classes=10):
    layers = [nn.Conv2d(3, widths[0], 3, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU()]
    c_in = widths[0]
    for i, width in enumerate(widths):
        layers.append(ResidualBlock(c_in, width, stride=1 if i == 0 else 2))
        c_in = width
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, n_classes)]
    return nn.Sequential(*layers)


def freeze_28(module):
    module.requires_grad_(False)                         # no gradient, no update of gamma, beta, weights
    module.eval()                                        # batchnorm: use (and keep) the running statistics
    return module


def train_mode_28(model, frozen):
    model.train()                                        # train() reaches every submodule...
    for module in frozen:
        module.eval()                                    # ... so the frozen ones go back to eval mode


def trainable_28(model):
    return [p for p in model.parameters() if p.requires_grad]
```
a) Le tronc (432 + 32), le premier bloc (deux convolutions $16 \to 16$ et deux batchnorms : 4 672, raccourci identité), le deuxième (13 824 + 128 de convolutions et de batchnorms, plus 512 + 64 pour le raccourci $1 \times 1$ : 14 528), le troisième (57 728), et la couche dense (650). b) Doubler les largeurs multiplie les poids des convolutions par 4 environ. c) On gèle 432 + 32 + 4 672 = 5 136 paramètres : il en reste 72 906 qui apprennent. Après trois pas, seuls les modules 4, 5 et 8 ont changé : le tronc et le premier bloc, statistiques glissantes comprises, sont intacts.
**Dans tes notes** : la somme garde $c_{out}$ canaux mais exige deux branches de même forme ; la concaténation garde les deux informations séparées, au prix d'une couche suivante deux fois plus large. `requires_grad_(False)` arrête les mises à jour de $\gamma$ et $\beta$, pas celles des statistiques glissantes, que chaque passe avant en mode entraînement modifie. Le premier bloc garde 16 canaux et la même taille : le raccourci identité suffit.
**Erreurs fréquentes** : deux encodeurs (une copie) dans le modèle siamois ; `torch.cat(..., dim=0)` (les exemples au lieu des canaux) ; geler avec `requires_grad_(False)` seulement ; appeler `model.train()` après le gel sans remettre les modules gelés en évaluation ; donner tous les paramètres à l'optimiseur (du calcul et de la mémoire gaspillés ; et si un gradient restait, avec `zero_grad(set_to_none=False)`, le momentum continuerait à déplacer les poids gelés) ; un premier bloc à stride 2.
**Variante** : `make_resnet_28(widths=(8, 16, 32))` a **19 954** paramètres, environ quatre fois moins que le ResNet du défi.

### Ex 24.29 — Précision mixte et torch.compile : mesurer avant de croire 📦
**Réponse** :
```python
def amp_dtype_29(device_type):
    types = {"cpu": torch.bfloat16, "cuda": torch.float16}
    if device_type not in types:
        raise ValueError(f"no mixed precision type for {device_type!r}")
    return types[device_type]


def time_steps_29(model, batches, amp=False, warmup=3, lr=1e-3):
    device_type = "cuda" if str(DEVICE).startswith("cuda") else "cpu"
    dtype = amp_dtype_29(device_type)
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device_type, enabled=(amp and dtype == torch.float16))
    durations, losses = [], []
    for i, (xb, yb) in enumerate(batches):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if device_type == "cuda":
            torch.cuda.synchronize()                       # the GPU computes asynchronously
        start = time.perf_counter()
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=amp):
            loss = nn.functional.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        scaler.scale(loss).backward()                      # identity when the scaler is disabled
        scaler.step(optimizer)
        scaler.update()
        if device_type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:                                    # the first steps warm up (and compile)
            durations.append(time.perf_counter() - start)
            losses.append(loss.item())
    return float(np.median(durations)), losses
```
**Démarche** : sous `torch.autocast` sur CPU, `nn.Linear` rend du `bfloat16`, ses poids restant en `float32`. Sur le CPU de génération (2 cœurs), un pas du CNN profond prend environ 71 ms en `float32` et 50 ms en précision mixte (un rapport de 0,69 à 0,76 selon l'exécution), avec les mêmes losses à 0,002 près. Sur un autre processeur, le rapport peut dépasser 1 ; sur un GPU récent, il descend souvent sous 0,5. Aucun temps n'est vérifié : il dépend de la machine.
**Dans tes notes** : ici, la précision mixte accélère d'environ 25 à 30 % ; elle gagne vraiment sur les GPU à cœurs tensoriels, et surtout pour les gros modèles. Les premiers pas paient l'allocation de la mémoire, le choix des algorithmes et la compilation ; la médiane ignore les pas ralentis par autre chose. En `float16` sans `GradScaler`, les petits gradients s'arrondissent à zéro : l'entraînement stagne, sans erreur.
**Erreurs fréquentes** : chronométrer le premier pas ; lire l'horloge sans `torch.cuda.synchronize()` sur un GPU (on mesure le temps de **lancer** les calculs) ; mettre `optimizer.step()` sous `autocast` ; un `GradScaler` en `bfloat16` (inutile) ; comparer deux modèles initialisés différemment ; enregistrer un temps comme résultat.
**Variante** : une sortie de convolution de forme `(128, 16, 32, 32)` occupe $128 \times 16 \times 1\,024 \times 4 = $ **8 388 608** octets en `float32`, et la moitié, **4 194 304**, en `bfloat16`.

### Ex 24.30 — Générer du texte : embeddings, séquences décalées, mots contre caractères 📦
**Réponses** : a) **33 948** ; b) **0,05785**.
**Démarche** :
```python
WORD_RE_30 = re.compile(r"[a-z]+(?:['’][a-z]+)*|[^\sa-z]")


def tokenize_words_30(text):
    return WORD_RE_30.findall(text.lower())


def make_vocab_30(tokens, max_size):
    return build_vocab(tokens, max_size=max_size, unk_token="<unk>")


def unk_rate_30(ids):
    return float(np.mean(np.asarray(ids) == 0))


def split_pairs_30(ids, seq_len, val_share=0.1):
    ids = np.asarray(ids)
    n_val = int(len(ids) * val_share)
    out = []
    for part in (ids[:len(ids) - n_val], ids[len(ids) - n_val:]):
        X, Y = make_sequence_pairs(part, seq_len)
        out += [torch.as_tensor(X, dtype=torch.int64), torch.as_tensor(Y, dtype=torch.int64)]
    return tuple(out)


def train_lm_30(model, X, Y, epochs, batch_size=64, lr=3e-3, seed=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(X, Y, batch_size, seed)
    losses = []
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            logits, _ = model(xb)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))   # T terms per row
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * yb.numel()
            count += yb.numel()
        losses.append(total / count)
    return losses


def mean_loss_30(model, X, Y, batch_size=256):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for i in range(0, len(X), batch_size):
            logits, _ = model(X[i:i + batch_size])
            total += nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y[i:i + batch_size].reshape(-1),
                                                 reduction="sum").item()
    return total / Y.numel()


def generate_30(model, prompt_ids, n_tokens, temperature=1.0, seed=0):
    rng = np.random.default_rng(seed)
    model.eval()
    new = []
    with torch.inference_mode():
        logits, state = model(torch.as_tensor([list(prompt_ids)], dtype=torch.int64))
        for _ in range(n_tokens):
            next_id = int(sample_from_logits(logits[0, -1].numpy(), temperature=temperature, rng=rng))
            new.append(next_id)
            logits, state = model(torch.tensor([[next_id]]), state)      # one token, the state carried along
    return new
```
a) Les 150 000 premiers caractères donnent 33 948 tokens (27 839 mots et 6 109 signes). b) Avec 2 001 entrées, 1 964 tokens sont inconnus, 5,785 %. Un modèle non entraîné a une perplexité de 2 001. En mode rapide (6 epochs) : le modèle de caractères (86 caractères) finit à 2,092 nats par caractère en validation, perplexité 8,1 ; le modèle de mots, à 5,365 nats par token, perplexité 214, soit 1,214 nat par caractère. Après 20 epochs sur tout le texte (`HISTORY_30`) : 1,549 nat par caractère pour le modèle de caractères, 4,187 nats par token pour celui de mots (8 % de `<unk>`), soit 0,953 nat par caractère.
**Dans tes notes** : non entraîné, le modèle donne à peu près $1/V$ à chaque token : sa loss vaut $\ln V$, sa perplexité $V$. Les perplexités ne se comparent qu'à vocabulaire égal ; la loss par caractère compare la même chose, mais le modèle de mots est avantagé par `<unk>`, un « mot » fréquent et facile à prédire qui cache tous les mots rares. La température règle l'audace : à 0,5 des tournures sûres et répétées, à 1,5 des mots inventés ou des suites de mots sans lien.
**Erreurs fréquentes** : la loss sur la seule dernière position (le *many-to-one* du livre) ; `cross_entropy(logits, y)` avec des logits `(N, T, V)` (PyTorch attend les classes sur l'axe 1 : il faut `reshape(-1, V)` ou `logits.transpose(1, 2)`) ; oublier de porter l'état pendant la génération (le modèle relit tout le prompt à chaque token, ou l'oublie) ; une moyenne par mini-batch au lieu d'une moyenne par token dans `mean_loss_30` ; `max_size=2000` en oubliant que `<unk>` compte dans les entrées.
**Variante** : avec un vocabulaire de 1 001 entrées, la part de tokens inconnus monte à **11,9 %**.

### Ex 24.31 — Défi CIFAR-10 frugal : au moins 80 % de test accuracy avec moins de 100 000 paramètres 🏆
**Réponse** :
```python
EPOCHS_31 = 8 if FAST_MODE else 30


def train_31(seed, images, labels, epochs=EPOCHS_31):
    torch.manual_seed(seed)
    model = make_resnet_28().to(DEVICE)                            # 78 042 parameters
    train_tf, _ = make_transforms_23(augment=True)                 # crops and left-right mirrors
    loader = make_train_loader_23(CifarImages_23(images, labels, train_tf), batch_size=128, seed=seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()                                        # OneCycleLR: one step per mini-batch
    return model
```
**Démarche** : en mode rapide (10 000 images, 8 epochs, 1 min 30 s sur 2 CPU), la recette atteint 63,5 % de validation et 63,1 % de test avec la graine 0, et 61,8 à 63,9 % de test avec les graines 1 et 2 (entrée `calib` de `tools/chapters/history_ch24.json`) : l'objectif de 60 % passe, de peu. En mode complet (45 000 images, 30 epochs), `tools/chapters/history_ch24.py` l'a entraînée sur CPU, une minute par epoch : 85,3 % de validation et **85,4 %** de test, avec 78 042 paramètres ; sur un GPU, quelques secondes par epoch. Le planning `OneCycleLR` monte jusqu'à 0,1 puis redescend presque à zéro, ce qui pose les poids au fond d'une vallée ; le label smoothing (0,1) et le weight decay retiennent le réseau de devenir trop sûr de lui ; l'augmentation empêche d'apprendre le train par cœur sur un entraînement long, ce que 24.23 a montré pour le CNN profond.
**Dans tes notes** : ce qui aide le plus dépend du budget. En mode complet, l'augmentation (sans elle, 30 epochs suffisent à apprendre le train par cœur, comme en 24.23) et le planning qui finit à un learning rate presque nul ; en mode rapide, 8 epochs ne laissent pas le temps de surapprendre, et la recette sans augmentation passe plus haut. Les essais qui échouent le plus souvent : un learning rate constant, des canaux ajoutés jusqu'à dépasser 100 000 paramètres, et une recette réglée sur le test au lieu de la validation.
**Erreurs fréquentes** : régler la recette en relançant la cellule jusqu'à ce que le test passe (le test est révélé une fois par architecture et par session pour cette raison) ; augmenter la validation ou le test ; un `OneCycleLR` avancé une fois par epoch (le planning ne fait qu'une petite partie de son parcours) ; ajouter des canaux jusqu'à dépasser 100 000 paramètres ; comparer deux recettes sur une seule graine, quand l'écart entre graines atteint deux points en mode rapide.
**Variante** : en mode rapide, la même recette **sans** augmentation fait mieux : **65,8 %** de test (graine 0), contre 63,1 % : en 8 epochs, le réseau n'a pas le temps de surapprendre, et l'augmentation ralentit plus qu'elle n'aide, comme dans le court entraînement de 24.23. Sur un entraînement long, l'historique de 24.23 montre l'inverse pour le CNN profond. Le bon réglage dépend du budget d'entraînement.
