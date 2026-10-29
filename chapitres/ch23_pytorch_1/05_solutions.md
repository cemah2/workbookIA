# 23 · PyTorch en pratique 1 : du dataset au modèle sauvegardé — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels, des exercices ✏️ 23.1 à 23.7 et de l'estimation 🧮 23.8 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 23.Q1 — Pourquoi utiliser une bibliothèque plutôt que tout coder soi-même ?
a) **Faux** : écrire `mylearn` t'a appris ce que calculent les fonctions que tu appelles maintenant ; c'est ce qui te permet de lire un message d'erreur, de choisir une loss ou de soupçonner un résultat trop beau. b) **B** : les formules sont les mêmes ; ce qu'apportent des années de travail d'une grande équipe, c'est la vitesse (GPU, calcul en parallèle), la stabilité numérique et des milliers de cas limites déjà traités. c) **C** : on écrit en Python pour penser au problème, pas aux registres du processeur ; une bibliothèque fait monter d'un niveau de plus. d) **Vrai** : les tests de `mylearn` comparent tes fonctions à NumPy, scikit-learn ou PyTorch, qui servent d'oracle. e) **Faux** : la bibliothèque calcule, mais c'est toi qui choisis les données, le prétraitement, l'architecture, la loss et la façon d'évaluer.
**À retenir** : on code une fois soi-même pour comprendre, puis on utilise la bibliothèque pour travailler ; les deux se complètent.

### 23.Q2 — Avertissement ou erreur ? Versions et dépréciations
a) **B** : un avertissement n'arrête rien ; un `FutureWarning` ou un `DeprecationWarning` annonce qu'une API ou un comportement va changer ou disparaître. b) **Vrai** : une exception que personne ne rattrape remonte jusqu'à Jupyter, qui arrête la cellule et affiche la pile d'appels. c) **Faux** : depuis Keras 2.0.6, `lr` a été renommé `learning_rate` et `period` `save_freq`, `predict_classes` a disparu (TensorFlow 2.6), et Keras 3 ne connaît plus Theano ni CNTK. d) **D** : les versions exactes (un fichier `requirements.txt` avec des `==`) et les graines figent ce qui peut changer. e) **Faux** : tu ne verrais plus l'annonce des changements à venir ; on filtre au cas par cas un avertissement compris, avec `warnings.filterwarnings("ignore", category=..., message=...)`, jamais tous.
**À retenir** : un avertissement est un message du futur ; on le lit, on corrige ou on le filtre précisément, mais on ne le fait pas taire en bloc.

### 23.Q3 — Déboguer méthodiquement : réduire, isoler, reconstruire
a) **C** : réduire l'appel jusqu'à l'exemple le plus petit qui plante, isoler l'élément fautif, puis reconstruire à partir de ce qui marche. b) **Vrai** : `breakpoint()` (Python 3.7) ouvre `pdb` à cet endroit ; `p x.shape` affiche une forme, `c` reprend l'exécution. c) **32** : `mat1`, l'entrée de la couche, a 32 lignes, une par image du mini-batch. d) **784** : la couche précédente sort 784 valeurs par image ; `748` est une faute de frappe (la même que celle du livre, p. 1153).
e) **Faux** : presque toujours, la bibliothèque a reçu des formes, des types ou des devices qui ne vont pas ensemble, et c'est ton code qui les a fournis ; le message nomme le problème, la pile d'appels dit où il est entré.
**À retenir** : lis la dernière ligne du message, puis remonte la pile jusqu'à la première ligne de **ton** code ; affiche les formes juste avant.

### 23.Q4 — Les trois sens du mot « modèle »
a) **A** : un plan de couches, sans poids appris. b) **B** : une architecture et ses poids, de quoi prédire. c) **C** : une façon de programmer. d) **Faux** : un `state_dict` ne contient que des tenseurs rangés par nom ; l'architecture, c'est le code (`make_mlp`) qu'on exécute avant de charger. e) **Vrai** : des formes compatibles garantissent seulement que le calcul s'exécute ; ce qu'apprend le réseau dépend des données, de la loss, de l'optimisation et de l'architecture.
**À retenir** : quand quelqu'un dit « le modèle », demande-toi s'il parle d'un plan, d'un réseau entraîné ou d'une API.

### 23.Q5 — Tenseurs, arrays et type float32
a) **4** : `(N, C, H, W)`, même avec un seul canal, `(N, 1, 28, 28)`. b) **B** : `float32`, le type par défaut des réels de PyTorch. c) **C** : `float64`, le type par défaut de NumPy, que `from_numpy` conserve ; d'où l'erreur « mat1 and mat2 must have the same dtype, but got Double and Float » quand on le passe à un modèle. d) **Vrai** : `from_numpy` ne copie pas ; `torch.tensor(a)` copie. e) **4** octets : 32 bits.
**À retenir** : vérifie toujours `dtype` et `shape` à la frontière NumPy/PyTorch ; `x.float()` règle la plupart des conflits de type.

### 23.Q6 — channels_first, channels_last et NCHW
a) **« (3, 48, 64) »** : canaux, hauteur (48 lignes), largeur (64 colonnes). b) **« (48, 64, 3) »**. c) **A** : `(N, C, H, W)`. d) **« (0, 3, 1, 2) »** : l'axe 0 du résultat est l'axe 0 (N) de l'entrée, puis l'axe 3 (C), puis 1 (H) et 2 (W). e) **Faux** : une image carrée de 3 × 3 pixels a la forme `(3, 3, 3)` dans les deux formats, et aucune erreur ne se produit ; plus généralement, un `reshape` maladroit donne la bonne forme avec de mauvaises valeurs (✏️ 23.2).
**À retenir** : on vérifie un format d'image en l'affichant, pas seulement en lisant sa forme.

### 23.Q7 — CPU, GPU, MPS, TPU : où calcule PyTorch ?
a) **C** : `"mps"`. b) **Vrai** : « Expected all tensors to be on the same device » ; on déplace chaque mini-batch avec `.to(device)`. c) **C** : des milliers d'unités de calcul simples font en même temps les multiplications-additions d'un produit de matrices. d) **Vrai** : dans un environnement CPU, il n'y a pas de GPU, et le code doit prévoir ce cas (§23.3.5). e) **Faux** : PyTorch ne se sert d'un TPU que par une bibliothèque dédiée (PyTorch/XLA, remplacé par TorchTPU en 2026).
**À retenir** : une variable `device`, choisie une fois en tête de notebook, puis `.to(device)` sur le modèle et sur chaque mini-batch.

### 23.Q8 — Iris, MNIST, Fashion-MNIST : les « hello world » du ML
a) **60 000**. b) **10**. c) **A** : selon ses auteurs, MNIST est trop facile, trop utilisé et ne représente pas les tâches modernes de vision. d) **Vrai** : 28 × 28 pixels en niveaux de gris, 60 000 images d'entraînement et 10 000 de test, les mêmes formats de fichiers. e) **4** : longueur et largeur du sépale et du pétale.
**À retenir** : un petit dataset qui se charge en une seconde est le bon terrain pour mettre au point un pipeline ; Fashion-MNIST est plus discriminant que MNIST.

### 23.Q9 — Labels entiers, one-hot et logits
a) **B** : des logits ; la loss applique elle-même une log-softmax stable. b) **B** : des indices de classe `int64` de forme `(N,)` ; elle accepte aussi des probabilités cibles de forme `(N, K)` depuis la version 1.10, mais ce n'est pas l'usage courant. c) **Faux** : la seconde softmax reçoit des valeurs entre 0 et 1, un calcul tout à fait valide ; l'entraînement est seulement plus lent et moins bon (✏️ 23.4). d) **« (3, 10) »**. e) **« (3, 9) »** : sans `num_classes`, le nombre de classes vaut le plus grand label plus 1, ici 9.
**À retenir** : en PyTorch, le modèle sort des logits, les labels sont des entiers, et la softmax ne sert qu'à lire des probabilités.

### 23.Q10 — fit() décortiqué : batch, epoch, validation, train() et eval()
a) **Faux** : `model.eval()` change le dropout et la batchnorm ; les gradients sont coupés par `torch.no_grad()` ou `torch.inference_mode()`. b) **B**. c) **D** : le test sert à choisir l'epoch, son score final est donc optimiste ; c'est la fuite du ch. 8, que le livre condamne lui-même p. 1126. d) **Vrai** : c'est la moyenne pondérée des loss des mini-batches, calculées pendant que les poids changent ; elle est un peu plus haute que la loss du modèle de fin d'epoch. e) **C** : créer `loss_fn` et `optimizer`.
**À retenir** : `train()`/`eval()` règlent les couches, `no_grad`/`inference_mode` règlent l'autograd ; la validation se prend dans le train, le test ne sert qu'une fois.

### 23.Q11 — Sauvegarder tout, les poids ou l'architecture ?
a) **C** : les poids, l'état de l'optimiseur (les moments d'Adam, le learning rate), celui du planning s'il y en a un, l'epoch et l'état des générateurs aléatoires. b) **B** : un dictionnaire ordonné de noms (`"1.weight"`, `"1.bias"`…) vers des tenseurs, paramètres et buffers (comme les statistiques glissantes d'une batchnorm). c) **Faux** : `weights_only=True` restreint le chargement aux tenseurs et aux types simples ; c'est justement ce qui empêche un fichier piégé d'exécuter du code. (Une faille de ce mécanisme, CVE-2025-32434, permettait encore de le contourner jusqu'à PyTorch 2.5.1 : d'où la précision « 2.6 ou plus récent », et la règle de garder PyTorch à jour.) d) **4** : le poids et le biais de chacune des deux couches `Linear` ; `Flatten` et `ReLU` n'ont pas de paramètres. e) **Vrai** : `load_state_dict` vérifie que les noms et les formes correspondent.
**À retenir** : pour prédire, un `state_dict` et le code qui reconstruit le modèle ; pour reprendre, un checkpoint complet ; pour charger un fichier venu d'ailleurs, `weights_only=True`.

### 23.Q12 — 🕰️ Que devient le code Keras du livre en 2026 ?
a) **B** : JAX, TensorFlow et PyTorch (plus OpenVINO pour la seule inférence). b) **A** : `np.argmax(model.predict(x), axis=-1)`. c) **Faux** : `torch.compile` accélère un modèle déjà défini ; le `compile()` de Keras choisit la loss, l'optimiseur et les mesures. d) **Faux** : Theano s'est arrêté à sa version 1.0 (2017, reprise ensuite sous d'autres noms par l'équipe de PyMC), CNTK à sa version 2.7 (2019). e) **C** : `.keras`, une archive qui contient la configuration et les poids ; le `.h5` reste lisible comme format ancien.
**À retenir** : le livre enseigne une démarche qui n'a pas vieilli ; ses appels de fonctions, eux, se traduisent.

<a id="rappels"></a>

## 🔁 Rappels

### 23.R1 — Ch. 22 : (N, T, D) et batch_first
a) **« (32, 20, 50) »**. b) **« (20, 32, 50) »** : le temps d'abord, la valeur par défaut des couches récurrentes de PyTorch. c) **« (2, 32, 64) »** : (couches × sens, N, H). d) **Faux** : `batch_first` ne concerne que l'entrée et `output`, pas `h_n` (ni `c_n`). e) **Vrai** : dans les deux formats courants du workbook, l'axe 0 est celui des exemples, ce qui permet à `DataLoader` de les empiler.
**À retenir** : avant d'entraîner, écris la forme attendue de chaque tenseur, axe par axe, et compare-la à `x.shape`.

### 23.R2 — Ch. 20 : Les cinq lignes d'une étape d'entraînement
a) **« BCEAD »** : remise à zéro, passe avant, loss, passe arrière, mise à jour. (L'ordre de la fiche, avec `zero_grad` juste avant `backward`, est aussi correct : ce qui compte, c'est que les gradients soient remis à zéro avant d'être recalculés.) b) **Vrai** : `backward` **ajoute** au `.grad` existant ; c'est utile pour accumuler des gradients sur plusieurs mini-batches, dangereux quand on l'oublie. c) **B**. d) **Vrai** : `loss` est relié à tout le graphe du mini-batch ; additionner les tenseurs garde ces graphes vivants et fait grossir la mémoire, alors que `loss.item()` est un simple nombre. e) **B** : une fois par epoch, après les `optimizer.step()` (PyTorch avertit si on l'appelle avant).
**À retenir** : zéro, avant, loss, arrière, pas ; et `.item()` pour garder la trace d'une loss.

### 23.R3 — Ch. 2 : Moyenne et écart-type du train pour normaliser les pixels
a) **0,4**. b) **0,335** : les carrés des écarts font $0{,}16 + 0{,}04 + 0 + 0 + 0{,}36 = 0{,}56$, leur moyenne $0{,}112$, sa racine $0{,}33466$. c) **1,494** : $0{,}5 / 0{,}33466$. d) **Faux** : on applique au test la moyenne et l'écart-type du train ; les recalculer sur le test, c'est donner au modèle des données transformées autrement que celles qu'il a apprises, et faire fuiter de l'information du test. e) **A**.
**À retenir** : les statistiques de normalisation font partie du modèle : apprises sur le train, sauvegardées avec les poids, appliquées telles quelles à tout le reste.

<a id="papier"></a>

## ✏️ 🧮 Papier-crayon

### Ex 23.1 — Remodeler 12 éléments : indices et ordre de lecture ✏️
**Réponses** : a) **V** ; b) **T** ; c) **W** ; d) **(0, 1, 1)** ; e) **AD** ; f) **[0, 4, 8, 1]** ; g) **11**.
**Démarche** : a) Position $2 \times 4 + 1 = 9$ : V. b) $2 \times 3 + 1 = 7$ : T. c) $1 \times 6 + 2 \times 2 + 0 = 10$ : W. d) Q est à la position 4 ; $4 = 0 \times 6 + 1 \times 3 + 1$. e) $5 \times 3 = 15$ et $4 \times 4 = 16$ ne font pas 12. f) `a.T` a la forme $(4, 3)$ et `a.T[i, j] = a[j, i]` ; lue ligne par ligne, sa première ligne est la première colonne de `a`, $0, 4, 8$, puis vient $a[0, 1] = 1$. g) $1 \times 6 + 2 \times 2 + 1 = 11$, le dernier élément.
**Dans ta copie** : `a.T` est une vue de la même mémoire, mais son ordre de lecture (ligne par ligne **de la transposée**) n'est plus l'ordre de la mémoire ; le tenseur est non contigu. `reshape(12)` rend les valeurs dans l'ordre de lecture de `a.T`, ce qui oblige à **copier** ; `view(12)` refuse, puisqu'une vue ne peut pas réordonner la mémoire.
**Erreurs fréquentes** : compter les positions à partir de 1 (N au lieu de M) ; inverser les rôles des tailles dans la formule ($i\,D_1$ au lieu de $i\,D_2$) ; croire que `a.T.reshape(12)` rend les valeurs dans l'ordre de la mémoire.
**Variante** : après `reshape(6, 2)`, la lettre à l'indice `(3, 0)` est à la position $3 \times 2 + 0 = 6$ : **S**.

### Ex 23.2 — De NHWC à NCHW : où atterrit chaque pixel ? ✏️
**Réponses** : a) **15** ; b) **(3, 2, 3)** ; c) **(1, 2, 0)** ; d) **5** ; e) **Faux** ; f) **B** ; g) **A** ; h) **786 432**.
**Démarche** : a) $9 \times 1 + 3 \times 2 + 0$. b) Les tailles de `x` (2, 3, 3) prises dans l'ordre des axes 2, 0, 1 : $(3, 2, 3)$. c) `y[c, h, w] = x[h, w, c]` : `y[0, 1, 2] = x[1, 2, 0]`, le rouge du pixel de la ligne 1, colonne 2. d) `z` est rangé dans son propre ordre C, de forme $(3, 2, 3)$ : position $6c + 3h + w = 0 + 3 + 2 = 5$ (dans `x`, cette valeur était à la position 15 : `contiguous` l'a déplacée). e) `y` n'est pas contigu (`permute` ne déplace rien), et `view` ne peut pas réordonner la mémoire : `RuntimeError`. f) `w = x.reshape(3, 2, 3)` relit la mémoire de `x` dans l'ordre : `w[0, 0, 1]` est la position 1 de la mémoire de `x`, soit `x[0, 0, 1]`, le **vert** du pixel (0, 0). g) `y[0, 0, 1] = x[0, 1, 0]`, le **rouge** du pixel (0, 1) : c'est ce que doit contenir le canal 0 à la ligne 0, colonne 1. h) $64 \times 3 \times 32 \times 32 = 196\,608$ valeurs, fois 4 octets.
**Dans ta copie** : `reshape` ne vérifie que le nombre d'éléments, 18 dans les deux cas ; la forme obtenue est celle qu'on attendait, mais chaque « canal » de `w` mélange des canaux et des pixels. Pour le détecter : afficher l'image convertie (elle apparaît rayée ou découpée en bandes), ou vérifier sur un pixel connu que `w[:, h, w]` vaut `x[h, w, :]`, ou construire un test avec une image dont un seul pixel est rouge.
**Erreurs fréquentes** : lire `permute(2, 0, 1)` à l'envers (« l'axe 0 va en position 2 ») ; croire que `permute` déplace les données ; prendre la bonne forme de `w` pour une preuve que la conversion est juste ; oublier les 4 octets d'un `float32`.
**Variante** : la position en mémoire de `z[2, 0, 1]` est $6 \times 2 + 0 + 1 =$ **13** ; c'est `x[0, 1, 2]`, qui était à la position 5 de `x`.

### Ex 23.3 — 623 290 paramètres : refaire le compte du livre ✏️
**Réponses** : a) **615 440** ; b) **7 850** ; c) **623 290** ; d) **0,987** ; e) **203 530** ; f) **535 818** ; g) **0**.
**Démarche** : a) $(784 + 1) \times 784$. b) $(784 + 1) \times 10$. c) $615\,440 + 7\,850$. d) $615\,440 / 623\,290 = 0{,}9874$. e) $(784 + 1) \times 256 + (256 + 1) \times 10 = 200\,960 + 2\,570$. f) $(784 + 1) \times 512 + (512 + 1) \times 256 + (256 + 1) \times 10 = 401\,920 + 131\,328 + 2\,570$. g) `Flatten` ne fait que changer la forme.
**Dans ta copie** : chacun des 784 pixels est relié à chacun des 784 neurones, soit $784^2$ poids, alors que la couche de sortie n'a que $784 \times 10$ poids. Une **convolution** (ch. 21) partage ses poids sur toute l'image : un noyau de $3 \times 3$ a 9 poids et un biais quelle que soit la taille de l'image, et `nn.Conv2d(1, 32, 3)` n'a que 320 paramètres.
**Erreurs fréquentes** : oublier les biais ($784 \times 784 = 614\,656$) ; compter des paramètres à `ReLU` ou `Flatten` ; arrondir d) à 0,98 au lieu de 3 décimales.
**Variante** : avec une couche cachée de 128 neurones, $(784 + 1) \times 128 + (128 + 1) \times 10 =$ **101 770** paramètres, six fois moins.

### Ex 23.4 — Une cross-entropy à la main : indice de classe ou one-hot ✏️
**Réponses** : a) **0,665** ; b) **2,408** ; c) **0,408** ; d) **Vrai** ; e) **0,797** ; f) **0,753** ; g) **2,303**.
**Démarche** : $e^1 = 2{,}7183$, $e^2 = 7{,}3891$, $e^0 = 1$, somme $11{,}1073$. a) $7{,}3891 / 11{,}1073 = 0{,}6652$. b) $\ln 11{,}1073 = 2{,}4076$. c) $-2 + 2{,}4076 = 0{,}4076$, soit aussi $-\ln 0{,}6652$. d) Avec $\mathbf{q} = (0, 1, 0)$, $-\sum_j q_j \ln p_j = -\ln p_1$ : la même loss. e) Les probabilités $(0{,}2447 ;\ 0{,}6652 ;\ 0{,}0900)$ prises comme logits : $-0{,}6652 + \ln(e^{0{,}2447} + e^{0{,}6652} + e^{0{,}0900}) = -0{,}6652 + \ln 4{,}3164 = 0{,}7972$. f) Seconde image : $-0 + \ln 3 = 1{,}0986$ ; moyenne $(0{,}4076 + 1{,}0986)/2 = 0{,}7531$. g) $-\ln(1/10) = \ln 10 = 2{,}3026$.
**Dans ta copie** : des logits entre 0 et 1 ne peuvent pas s'écarter de plus de 1 ; même parfaitement sûr de lui, le modèle donne au plus $e/(e + 9) \approx 0{,}23$ à la bonne classe sur 10, et la loss ne descend jamais sous $-\ln 0{,}23 \approx 1{,}46$. Les gradients passent par une softmax de plus et sont écrasés : l'entraînement se traîne, mais rien n'est invalide, donc rien ne plante. La valeur $\ln 10 \approx 2{,}30$ est la loss attendue au **début** d'un entraînement à 10 classes, quand le modèle à peine initialisé donne des probabilités presque égales : une première loss très différente signale un problème (labels, normalisation, initialisation).
**Erreurs fréquentes** : prendre le logarithme décimal ; oublier le signe moins ; faire la somme des loss du mini-batch au lieu de la moyenne ; arrondir les probabilités à 2 décimales avant e).
**Variante** : avec les logits $(2 ;\ 0 ;\ 0)$ et le label 0, la loss vaut $-2 + \ln(e^2 + 2) =$ **0,240**.

### Ex 23.5 — Batches par epoch et nombre de mises à jour ✏️
**Réponses** : a) **211** ; b) **210** ; c) **240** ; d) **4 220** ; e) **235** ; f) **188** ; g) **53 760**.
**Démarche** : le train fait $60\,000 \times 0{,}9 = 54\,000$ images. a) $\lceil 54\,000 / 256 \rceil = \lceil 210{,}94 \rceil$. b) $\lfloor 210{,}94 \rfloor$. c) $54\,000 - 210 \times 256 = 54\,000 - 53\,760$. d) $211 \times 20$. e) $\lceil 60\,000 / 256 \rceil = \lceil 234{,}4 \rceil$. f) $\lceil 12\,000 / 64 \rceil = \lceil 187{,}5 \rceil$. g) $210 \times 256$.
**Dans ta copie** : non ; avec `drop_last=True`, au plus $B - 1$ images (ici 240) sont laissées de côté à chaque epoch, et comme le `DataLoader` mélange à chaque epoch, ce ne sont pas les mêmes : chaque image est vue presque à chaque epoch. On s'en sert quand un mini-batch trop petit pose problème (une batchnorm sur un mini-batch d'une seule image, une taille fixe exigée par un modèle compilé).
**Erreurs fréquentes** : oublier de retirer la validation ; arrondir a) au-dessous ; croire qu'un mini-batch incomplet compte pour une demi-mise à jour.
**Variante** : avec des mini-batches de 128, $\lceil 54\,000 / 128 \rceil =$ **422** mini-batches, dont un dernier de **112** images.

### Ex 23.6 — Early stopping à la main : patience et min_delta ✏️
**Réponses** : a) **10** ; b) **7** ; c) **8** ; d) **5** ; e) **16** ; f) **11** ; g) **10**.
**Démarche** : a) Améliorations aux epochs 1, 2, 3 ; l'epoch 4 (0,41) fait monter le compteur à 1 ; l'epoch 5 (0,38) est une amélioration ; 6 (0,385) : compteur 1 ; 7 (0,379 < 0,38) : amélioration ; 8, 9, 10 : compteur 1, 2, 3, arrêt à la fin de l'epoch 10. b) La dernière amélioration est l'epoch 7. c) Avec $\delta = 0{,}005$ : l'epoch 5 est une amélioration ($0{,}38 < 0{,}40 - 0{,}005$) ; l'epoch 7 ne l'est pas ($0{,}379 \ge 0{,}38 - 0{,}005 = 0{,}375$), la meilleure loss reste 0,38 ; le compteur vaut 1, 2, 3 aux epochs 6, 7, 8 : arrêt à la fin de l'epoch 8. d) Epoch 5. e) Comme en a) jusqu'à l'epoch 10, où le compteur vaut 3 ; l'epoch 11 (0,37) est une amélioration ; les epochs 12 à 16 font monter le compteur à 5 : arrêt à la fin de l'epoch 16. f) Epoch 11. g) Une règle qui ne recharge rien garde les poids de l'epoch où elle s'arrête : 10. (L'`EarlyStopping` de Keras 2.0, lui, attendait une epoch de plus que sa patience : il se serait arrêté plus tard.)
**Dans ta copie** : trois epochs moins bonnes de suite (8, 9, 10) suffisent à arrêter, alors que la loss de validation est bruitée et que le modèle allait trouver mieux à l'epoch 11 ; l'early stopping ne voit pas l'avenir. Une patience plus grande coûte des epochs de calcul (ici 6 de plus), et, si l'on choisit sur la validation parmi plus d'epochs, son score de la meilleure epoch devient un peu plus optimiste ; d'où le test, à part.
**Erreurs fréquentes** : compter l'epoch d'amélioration dans la patience ; mettre à jour la meilleure loss avec une baisse plus petite que `min_delta` ; recharger les poids de l'epoch d'arrêt ; arrêter quand le compteur **dépasse** la patience.
**Variante** : avec `patience=4` et `min_delta=0`, l'entraînement s'arrête à la fin de l'epoch **15** et recharge l'epoch **11**.

### Ex 23.7 — Plannings de learning rate : le planning du livre avec LambdaLR ✏️
**Réponses** : a) **0,75** ; b) **90** ; c) **0,005** ; d) **20** ; e) **0,001** ; f) **0,0854** ; g) **10**.
**Démarche** : a) $1 \times \max(0{,}1 ;\ 0{,}75)$. b) $1 - 0{,}01\,e \le 0{,}1 \iff e \ge 90$. c) $0{,}01 \times \max(0{,}1 ;\ 0{,}5)$ : le collègue a divisé tout le planning par 100. d) Il faut $0{,}05 \cdot \lambda(e) = \max(0{,}1 ;\ 1 - 0{,}01\,e)$, donc $\lambda(e) = \max(0{,}1 ;\ 1 - 0{,}01\,e) / 0{,}05$ et $\lambda(0) = 1 / 0{,}05 = 20$. e) Pendant l'epoch 65, $\lfloor 65/30 \rfloor = 2$ paliers sont passés : $0{,}1 \times 0{,}1^2$. f) $0{,}1 \times (1 + \cos(\pi/4))/2 = 0{,}1 \times 1{,}70711 / 2 = 0{,}085355$. g) Les epochs 90 à 99.
**Dans ta copie** : chaque appel avance le planning d'un cran ; avec quelques centaines d'appels par epoch (211 avec les réglages de 23.5), $\lambda$ atteint son plancher après 90 mini-batches, avant la fin de la première epoch, et tout le reste de l'entraînement se fait à 0,1 : le planning est comprimé 211 fois. Pour un planning par mini-batch, il faudrait l'écrire en pas : $\lambda(s) = \max(0{,}1 ;\ 1 - 0{,}01\,s/211)$.
**Erreurs fréquentes** : lire la valeur de $\lambda$ comme un learning rate (c) ; compter les epochs à partir de 1 ; pour `StepLR`, arrondir $65/30$ au plus proche ; calculer le cosinus en degrés.
**Variante** : `ExponentialLR(optimizer, gamma=0.95)` avec $\eta_0 = 0{,}1$ donne pendant l'epoch 10 le learning rate $0{,}1 \times 0{,}95^{10} =$ **0,0599**.

### Ex 23.8 — Fermi : durée d'une epoch et poids d'un checkpoint 🧮
**Réponses** : a) **224,4** ; b) **4,5** ; c) **2,5** ; d) **7,5** ; e) **802**.
**Démarche** : a) $6 \times 623\,290 \times 60\,000 = 2{,}2438 \times 10^{11}$. b) $2{,}2438 \times 10^{11} / 5 \times 10^{10} = 4{,}49$ s. c) $623\,290 \times 4 = 2\,493\,160$ octets. d) Les poids et deux moments : $3 \times 2{,}49$ Mo $= 7{,}48$ Mo (plus quelques octets pour l'epoch et le learning rate). e) $3\,600 / 4{,}4877 = 802{,}2$, soit 802 epochs complètes.
**Dans ta copie** : oui, c'est le même ordre de grandeur que les 2 à 3 secondes du livre (un CPU de 2014 à plusieurs cœurs, avec des instructions vectorielles, fait quelques dizaines de milliards d'opérations par seconde). Une vraie epoch prend plus longtemps parce que : le chargement et la mise en forme des mini-batches prennent du temps ; la validation ajoute une passe avant sur ses images ; les activations (ReLU), la softmax, la loss et la mise à jour d'Adam ne sont pas comptées ; le CPU n'atteint pas sa vitesse maximale sur de petits produits de matrices (mémoire, surcoût de Python à chaque mini-batch).
**Erreurs fréquentes** : oublier le facteur 6 (passe avant et passe arrière) ; confondre mégaoctets et mébioctets ($2^{20}$) ; compter Adam comme un seul tenseur de plus ; arrondir e) au plus proche.
**Variante** : le MLP à 256 neurones cachés (203 530 paramètres) demande $6 \times 203\,530 \times 60\,000 =$ **73,3** milliards d'opérations par epoch, environ 1,5 seconde au même rythme.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 23.9 — Pourquoi le prétraitement doit voyager avec le modèle 🗣️
**Réponse modèle** : « Un modèle, c'est comme un traducteur qui n'a jamais lu que des textes tapés en minuscules et sans accents : donne-lui un texte écrit autrement, il traduira quand même, sans te prévenir, et fera des contresens. Les poids ont été appris sur des nombres préparés d'une façon précise : divisés par 255, centrés et réduits avec la moyenne et l'écart-type **du train**, rangés dans un certain ordre. Si on lui donne des pixels bruts, il répond quand même, avec assurance, et se trompe. Il faut donc livrer avec les poids la fonction de prétraitement et ses nombres (moyenne, écart-type, taille des images, ordre des classes), ou mieux, les mettre dans le modèle lui-même. Et le tout se décrit dans une fiche modèle, pour que personne n'ait à deviner. »
**Ce qui compte** : un modèle mal nourri ne plante pas, il se trompe en silence ; les nombres du prétraitement viennent du train et ne se recalculent pas ; trois façons d'éviter l'oubli (dans le modèle, à côté des poids, dans la fiche modèle) ; l'ordre des classes fait partie du contrat.
**Erreurs fréquentes** : dire que le modèle « plantera » ; proposer de recalculer la normalisation sur les nouvelles données ; oublier la liste des classes ; utiliser le mot « normaliser » sans dire ce qu'il recouvre.
**Variante** : en une phrase pour un chef de projet : « Le fichier de poids est un moteur réglé pour un carburant précis ; on le livre avec la pompe qui prépare ce carburant, sinon il tourne, mais mal. »

### Ex 23.10 — Réutiliser un modèle pré-entraîné : licence, données d'origine, biais hérités ⚖️
1. Le **code** de torchvision est sous licence BSD-3-Clause, mais cette licence ne dit rien des **poids**. La documentation de torchvision prévient que les poids pré-entraînés « peuvent avoir leur propre licence ou leurs propres conditions, issues du dataset d'entraînement », et qu'il revient à l'utilisateur de vérifier qu'il a le droit de s'en servir pour son usage. Les conditions d'ImageNet réservent la base à la recherche et à l'enseignement **non commerciaux**. Le statut juridique de poids entraînés sur une base soumise à de telles conditions n'est pas tranché (beaucoup d'entreprises s'en servent, ce qui ne vaut pas autorisation) : pour une application payante, c'est un risque à faire évaluer par un juriste, à documenter, et à réduire si possible (question 5).
2. L'audit de Yang, Qinami, Fei-Fei, Deng et Russakovsky (FAT* 2020) a examiné les 2 832 catégories de **personnes** de l'ImageNet complet : 1 593 ont été jugées « dangereuses », parce que leurs noms étaient offensants (insultes, termes racistes ou misogynes) ou sensibles (orientation sexuelle, religion, maladie…), et beaucoup d'autres non représentables par une image (on ne voit pas sur une photo qu'une personne est « philanthrope ») ; elles encouragent à associer une apparence à un jugement. La mise à jour d'ImageNet du 17 septembre 2019 a annoncé leur retrait, avec 600 040 images. Ces catégories n'étaient pas dans le sous-ensemble à 1 000 classes qui sert à pré-entraîner les modèles de torchvision ; mais le reproche le plus général le concerne : la collecte de photos de personnes sans leur consentement (une étude de 2021, Yang et coll., ICML 2022, a trouvé au moins un visage dans 17 % de ses images). Un modèle de **vêtements** est concerné parce que ses features ont été apprises sur ces images : ce que le réseau a appris à remarquer, et à ignorer, vient de leur contenu, et les vêtements y apparaissent souvent portés par des personnes photographiées sans le savoir ; le sous-ensemble contient d'ailleurs des classes de vêtements comme « kimono », « abaya » ou « sarong », chacune vue à travers les photos et les labels d'un web surtout occidental.
3. Par exemple : des vêtements traditionnels ou régionaux peu présents dans les données (un boubou, un sari) reconnus moins bien ou rangés dans une classe voisine ; des erreurs plus fréquentes sur des photos de personnes à la peau foncée, ou prises avec un éclairage faible, un téléphone d'entrée de gamme, des poses ou des arrière-plans différents de ceux des photos d'origine. La parade : mesurer l'accuracy par groupe (type de vêtement, conditions de prise de vue, et, avec précaution, couleur de peau) sur des données de test qui ressemblent à celles des utilisateurs.
4. La fiche modèle doit nommer le modèle de départ (architecture, version des poids, `ResNet18_Weights.IMAGENET1K_V1`), sa provenance (torchvision) et sa licence ou ses conditions, les données de pré-entraînement (ImageNet, avec ses limites connues et le lien vers l'audit), ce qui a été entraîné en plus (la tête, sur quelles photos, combien, avec quel consentement), le prétraitement, les résultats par groupe, les usages prévus et exclus (pas de déduction sur la personne qui porte le vêtement), et les limites.
5. Des poids dont la licence autorise explicitement l'usage commercial et dont les données sont documentées ; un pré-entraînement auto-supervisé sur ses propres photos, ou sur des données sous licence ; l'achat ou la collecte de données avec consentement ; à défaut, un modèle plus petit entraîné de zéro sur ses propres données, moins précis au départ mais sans dette juridique. Dans tous les cas, documenter le choix.
**Erreurs fréquentes** : confondre la licence du code et celle des poids ; affirmer qu'un usage commercial est clairement interdit ou clairement permis ; confondre l'ImageNet complet, dont les catégories de personnes ont été retirées, et le sous-ensemble à 1 000 classes des poids `IMAGENET1K_V1`, qui n'en avait pas mais contient beaucoup de personnes photographiées sans leur accord ; croire qu'un modèle de vêtements ne peut pas hériter de biais liés aux personnes.
**Variante** : si l'entreprise voulait aussi estimer l'âge ou le genre de la personne pour « personnaliser » les suggestions, le problème changerait de nature : déduire des caractéristiques d'une personne à partir d'une photo touche aux données personnelles (RGPD, consentement explicite, minimisation), et ces déductions sont souvent fausses et discriminantes ; la réponse raisonnable est de ne pas le faire.

### Ex 23.11 — Lire Krizhevsky et coll. (2012) : la recette d'entraînement d'AlexNet 📄
1. Sur deux cartes NVIDIA GTX 580 de 3 Go, pendant cinq à six jours. Un seul GPU n'avait pas assez de mémoire pour le réseau et ses activations : les auteurs ont mis la moitié des noyaux sur chaque carte, les deux moitiés ne communiquant qu'à certaines couches.
2. Un petit CNN à quatre couches atteint 25 % d'erreur d'entraînement sur CIFAR-10 **six fois plus vite** avec la ReLU qu'avec la tanh. C'est ce que dit le ch. 17 : la ReLU ne sature pas pour les entrées positives, sa dérivée y vaut 1, alors que la tanh sature et écrase ses gradients.
3. `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)`, avec un `DataLoader(..., batch_size=128, shuffle=True)`. (Leur règle de mise à jour multiplie le weight decay par le learning rate, ce qui revient au `weight_decay` de PyTorch.)
4. Le learning rate partait de 0,01 et était **divisé par 10** à la main quand l'erreur de validation cessait de s'améliorer ; il a été réduit **trois fois**, sur environ **90 passages** sur les 1,2 million d'images. `ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=...)`, appelé à chaque epoch avec l'erreur (ou la loss) de validation, fait la même chose automatiquement.
5. L'**augmentation de données** (des morceaux de 224 × 224 pris au hasard dans les images de 256 × 256, leurs symétriques horizontaux, et une variation des couleurs fondée sur une ACP des valeurs RVB) et le **dropout** (avec une probabilité de 0,5, dans les deux premières couches entièrement connectées). Le dropout est au ch. 20, l'augmentation de données au ch. 24.
6. 15,3 % d'erreur top-5 sur le test d'ILSVRC-2012, contre 26,2 % pour la deuxième équipe. Ce n'est pas un seul réseau : c'est la **moyenne des prédictions de sept CNN**, dont deux pré-entraînés sur une version plus grande d'ImageNet (automne 2011) ; un seul CNN faisait environ 18,2 % sur la validation. (Sur ILSVRC-2010 : 37,5 % d'erreur top-1 et 17,0 % top-5.)
7. Resté standard : la ReLU, le dropout, l'augmentation par recadrages et symétries, SGD avec momentum et weight decay, la baisse du learning rate par paliers, l'entraînement sur GPU. Fait autrement : la normalisation locale de la réponse (LRN) a disparu au profit de la batchnorm (2015) ; l'initialisation par une loi normale d'écart-type 0,01 a laissé la place à celle de He (ch. 16) ; les grands noyaux de 11 × 11 à des piles de 3 × 3 ; la division manuelle du learning rate à des plannings écrits d'avance (paliers, cosinus, warmup) ; Adam ou AdamW sont courants ; les énormes couches entièrement connectées ont été remplacées par un pooling global moyen ; et l'on répartit les données, plutôt que le réseau, entre les GPU.
**Erreurs fréquentes** : citer les chiffres de la page de résumé du site des actes (500 000 neurones, 39,7 % et 18,9 %), qui sont ceux d'une version antérieure ; attribuer les 15,3 % à un seul réseau ; croire que la baisse du learning rate suivait un calendrier fixé d'avance ; confondre l'erreur top-1 et l'erreur top-5.
**Variante** : avec la recette d'AlexNet (0,01 divisé par 10 trois fois), le learning rate final vaut $0{,}01 / 10^3 =$ **0,00001**.

<a id="entretien"></a>

## 💼 Entretien

### 23.E1 — Ton entraînement PyTorch est deux fois plus lent que prévu : où cherches-tu, et dans quel ordre ?
**Réponse modèle en 60 secondes** : « D'abord je mesure, au lieu de deviner : je chronomètre séparément le temps passé à attendre les données et le temps de calcul, sur quelques dizaines de pas, en écartant les premiers, plus lents. Si c'est le chargement, je regarde le `DataLoader` : le nombre de `num_workers`, `pin_memory` sur GPU, et des transformations trop coûteuses faites à chaque image, qu'on peut précalculer. Si c'est le calcul, je vérifie que le modèle **et** les mini-batches sont bien sur le GPU, puis je cherche les synchronisations cachées : un `.item()`, un `print` ou une conversion en NumPy à chaque pas obligent le CPU à attendre le GPU. Ensuite, la taille des mini-batches, souvent trop petite pour occuper le GPU, et le type des nombres : la précision mixte (`torch.autocast`) accélère beaucoup les GPU récents. Si ce n'est pas assez, je lance `torch.profiler` pour voir les opérations les plus coûteuses, et j'essaie `torch.compile`. Je compare enfin à une mesure de référence : deux fois plus lent que quoi, sur quelle machine ? »
**Ce qui compte** : mesurer avant d'optimiser ; séparer données et calcul ; le device du modèle et des données ; les synchronisations ; la taille des mini-batches et la précision mixte ; un profiler ; une référence de comparaison.
**Relances possibles** : « Pourquoi un `.item()` ralentit-il ? » (sur GPU, les opérations sont lancées de façon asynchrone ; lire une valeur sur le CPU oblige à attendre que tout ce qui précède soit fini, et casse le recouvrement entre le CPU et le GPU) · « Comment savoir si le GPU est bien utilisé ? » (`nvidia-smi` montre son taux d'utilisation et sa mémoire ; un taux faible et irrégulier indique qu'il attend les données) · « Que gagne-t-on avec la précision mixte ? » (les calculs en `float16` ou `bfloat16` vont souvent deux fois plus vite et prennent deux fois moins de mémoire sur les GPU qui ont des unités dédiées ; on garde une copie des poids en `float32` et, en `float16`, une mise à l'échelle de la loss).

### 23.E2 — Pourquoi appeler model.eval() et torch.no_grad() (ou inference_mode) à l'inférence ?
**Réponse modèle en 60 secondes** : « Ce sont deux interrupteurs indépendants. `model.eval()` change ce que **calculent** certaines couches : le dropout ne coupe plus rien, et la batchnorm utilise ses statistiques glissantes, apprises pendant l'entraînement, au lieu de celles du mini-batch. Si on l'oublie, les prédictions sont fausses ou changent d'un appel à l'autre, et une image seule dépend des autres images du mini-batch. `torch.no_grad()` change ce que PyTorch **garde en mémoire** : il n'enregistre plus le graphe de calcul pour la passe arrière. Si on l'oublie, les prédictions sont justes, mais on gaspille de la mémoire et du temps, parfois au point de saturer le GPU. `torch.inference_mode()` est une version plus stricte et un peu plus rapide de `no_grad`, dont les tenseurs ne peuvent plus servir à un calcul de gradient. À l'inférence, on fait les deux ; et on n'oublie pas de revenir à `model.train()` avant de reprendre l'entraînement. »
**Ce qui compte** : deux mécanismes distincts (les couches, l'autograd) ; la conséquence de chaque oubli (un résultat faux, un gaspillage) ; la différence entre `no_grad` et `inference_mode` ; le retour en mode entraînement.
**Relances possibles** : « Un modèle sans dropout ni batchnorm a-t-il besoin de `eval()` ? » (son calcul ne change pas, mais on l'appelle quand même : c'est une habitude qui évite les surprises quand l'architecture change, et certains modules ont d'autres comportements) · « Peut-on utiliser `inference_mode` pendant la validation d'un entraînement ? » (oui, à condition de ne pas réutiliser ses sorties dans un calcul qui demande un gradient) · « Comment vérifier qu'on n'a rien oublié ? » (deux prédictions successives sur la même image doivent être identiques, et la sortie ne doit pas avoir de `grad_fn`).

### 23.E3 — Comment sauvegarder un modèle PyTorch pour reprendre l'entraînement plus tard ou le déployer ?
**Réponse modèle en 60 secondes** : « Je ne sauvegarde pas l'objet modèle avec pickle, qui dépend du code exact des classes ; je sauvegarde des `state_dict`, et le contenu dépend du but. Pour **reprendre** demain, il me faut tout ce qui détermine la suite : les poids, l'état de l'optimiseur, puisque sans lui Adam repartirait de moments nuls, l'état du planning de learning rate, l'epoch, le meilleur score et l'état des générateurs aléatoires ; je l'écris régulièrement, et à chaque meilleure validation. Pour **déployer**, je livre les poids seuls, dans un format sûr, safetensors ou un fichier chargé avec `weights_only=True`, avec ce qu'il faut pour reconstruire l'architecture (le code et ses arguments), le prétraitement et ses paramètres, l'ordre des classes, les versions des bibliothèques et une fiche modèle. Si la production n'est pas en Python, j'exporte avec `torch.export` ou en ONNX. Et je teste le rechargement : mêmes prédictions avant et après. »
**Ce qui compte** : `state_dict` plutôt que l'objet ; le contenu d'un checkpoint de reprise ; le paquet de déploiement, prétraitement compris ; la sécurité du chargement ; un test de rechargement.
**Relances possibles** : « Pourquoi `weights_only=True` ? » (un fichier pickle peut exécuter du code à son chargement ; depuis PyTorch 2.6, `torch.load` n'accepte par défaut que des tenseurs et des types simples) · « Le modèle rechargé donne des résultats différents : pourquoi ? » (`model.eval()` oublié, un prétraitement différent, une autre architecture chargée avec `strict=False`, ou un ordre des classes différent) · « Où sauvegarder pendant un long entraînement sur un serveur partagé ? » (sur un stockage persistant, pas sur le disque temporaire de la machine, et en gardant les deux derniers checkpoints au cas où l'écriture du dernier serait interrompue).

### 23.E4 — La loss de validation remonte alors que l'accuracy de validation progresse encore : que se passe-t-il ?
**Réponse modèle en 60 secondes** : « C'est le début de l'overfitting, vu d'abord par la loss. L'accuracy ne regarde que la classe prédite ; la cross-entropy regarde la probabilité donnée à la bonne classe. Le modèle continue de corriger quelques erreurs, donc l'accuracy monte, mais il devient de plus en plus sûr de lui, y compris sur les images qu'il rate : une erreur commise avec une probabilité de 0,01 pour la bonne classe coûte $-\ln 0{,}01 \approx 4{,}6$ à elle seule, et quelques erreurs de ce genre suffisent à faire monter la moyenne. Le modèle devient donc moins bien **calibré** : ses probabilités ne sont plus fiables. Ce que je fais dépend de ce qui compte pour l'usage : si seule la classe compte, je peux choisir l'epoch sur l'accuracy de validation ; si l'on se sert des probabilités, pour un seuil ou une décision, je m'arrête sur la loss, ou je recalibre ensuite (par une température, ch. 3 et 17). Dans les deux cas, j'ajoute de la régularisation, weight decay, dropout, augmentation de données, et je garde le meilleur checkpoint. »
**Ce qui compte** : la différence entre ce que mesurent l'accuracy et la loss ; la sur-confiance sur les erreurs ; la calibration ; un choix d'arrêt lié à l'usage ; la régularisation et le meilleur checkpoint.
**Relances possibles** : « Comment vérifier que c'est bien la confiance qui augmente ? » (tracer l'histogramme des probabilités données à la bonne classe sur les images mal classées, ou un diagramme de fiabilité et le score de Brier, au fil des epochs) · « Le label smoothing aiderait-il ? » (oui : il interdit au modèle de viser une probabilité de 1, ce qui limite la sur-confiance ; `nn.CrossEntropyLoss(label_smoothing=0.1)`) · « Et si la loss de validation est bruitée ? » (lisser la courbe, utiliser une patience, et vérifier que la validation est assez grande : avec 6 000 images, une accuracy se mesure à environ un demi-point près).

### 23.E5 — Keras ou PyTorch : comment choisir un framework aujourd'hui ?
**Réponse modèle en 60 secondes** : « Je regarde d'abord le contexte : ce que l'équipe sait faire, le code et les modèles existants, et où le modèle sera déployé. PyTorch domine la recherche et les modèles publiés : la plupart des articles et des modèles de Hugging Face sont en PyTorch, et sa boucle explicite donne un contrôle total, au prix de quelques lignes à écrire, ou d'une bibliothèque comme Lightning. Keras offre une API de plus haut niveau, `compile` et `fit`, très rapide pour des modèles classiques, et depuis Keras 3 il tourne au choix sur JAX, TensorFlow ou PyTorch : le choix n'est plus exclusif. TensorFlow garde des atouts pour certains déploiements, sur mobile ou dans le navigateur. Par défaut, pour un nouveau projet de deep learning, je prendrais PyTorch, pour son écosystème et parce que c'est là que se trouvent les modèles pré-entraînés ; mais les concepts, tenseurs, autograd, boucle, sauvegarde, se transposent en une semaine d'un framework à l'autre. »
**Ce qui compte** : des critères (équipe, écosystème, déploiement, niveau d'abstraction) avant une préférence ; la place de PyTorch dans la recherche ; Keras 3 et ses backends ; un choix par défaut argumenté ; la transposabilité des concepts.
**Relances possibles** : « Et JAX ? » (une bibliothèque de transformations de fonctions, `grad`, `jit`, `vmap`, très utilisée par Google et pour la recherche sur TPU ; plus bas niveau, avec des bibliothèques comme Flax au-dessus) · « Qu'apporte Lightning ? » (une boucle d'entraînement standard, des callbacks, l'entraînement distribué et la précision mixte sans réécrire la boucle, tout en gardant des modules PyTorch) · « Comment passer un modèle de PyTorch en production sans Python ? » (l'exporter avec `torch.export` ou en ONNX, puis le servir avec un moteur d'inférence comme ONNX Runtime ou TensorRT).

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb` ; voici, pour chaque exercice, la réponse, la démarche, les erreurs fréquentes et une variante. Les chiffres sont ceux du mode rapide, sur CPU (PyTorch 2.11) : un autre processeur ou un GPU peut changer les dernières décimales des résultats d'entraînement.

### Ex 23.12 — Environnement et tenseurs : versions, device, reshape, view, permute 📦
**Réponses** : a) **215** ; b) **UserWarning**.
**Démarche** :
```python
def library_versions_12():
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "torchvision": torchvision.__version__}


def pick_device_12():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_nchw_12(images):
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"expected 4 axes (N, H, W, C), got the shape {images.shape}")
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().contiguous()   # permute, never reshape


def caught_warnings_12(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")        # every warning, even one already shown in this session
        fn()
    return [w.category.__name__ for w in caught]
```
a) `x[1, 2, 3, 4]` est le canal 2 du pixel (ligne 3, colonne 4) de l'image 1 : `IMAGES_12[1, 3, 4, 2]`, qui vaut 215. b) `nn.MSELoss` émet un `UserWarning` : la cible `(4,)` va être diffusée contre la prédiction `(4, 1)`.
**Dans tes notes** : `permute` rend une vue non contiguë ; `view` ne sait que redécouper la mémoire dans son ordre et refuse, `reshape` recopie. Diffusée, la cible `(4,)` contre la prédiction `(4, 1)` donne une matrice `(4, 4)` d'écarts : la loss compare chaque prédiction à toutes les cibles, une moyenne fausse sans erreur.
**Erreurs fréquentes** : `torch.from_numpy(images).reshape(N, C, H, W)` (la bonne forme, de mauvaises valeurs : 35 au lieu de 215) ; oublier `.contiguous()` ; `str(w.category)` (« `<class 'UserWarning'>` ») au lieu de `__name__` ; oublier `simplefilter("always")`, si bien qu'un avertissement déjà affiché dans la session n'est plus enregistré.
**Variante** : `to_nchw_12(IMAGES_12)[3, 0, 4, 5]`, le rouge du pixel (4, 5) de l'image 3, vaut `IMAGES_12[3, 4, 5, 0]` = **204**.

### Ex 23.13 — Charger et regarder Fashion-MNIST 📦
**Réponses** : a) **Ankle boot** ; b) **6 000** ; c) **Sandal** ; d) **Coat**.
**Démarche** :
```python
X_train_13, y_train_13 = wb.datasets.load_fashion_mnist("train", seed=None)
X_test_13, y_test_13 = wb.datasets.load_fashion_mnist("test", seed=None)
first_class_13 = CLASSES[y_train_13[0]]
per_class_13 = int(np.bincount(y_train_13)[0])                  # 6000 for each of the 10 classes
means_13 = [X_train_13[y_train_13 == k].mean() for k in range(10)]
darkest_13, brightest_13 = CLASSES[int(np.argmin(means_13))], CLASSES[int(np.argmax(means_13))]


def show_grid_13(images, labels, per_class=8):
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 0.9, 10 * 0.95))
    for k in range(10):
        chosen = np.flatnonzero(labels == k)[:per_class]
        for j, ax in enumerate(axes[k]):
            ax.imshow(images[chosen[j]], cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[k, 0].set_ylabel(CLASSES[k], rotation=0, ha="right", va="center", fontsize=8)
    plt.show()
```
Les moyennes des pixels vont de 35 (sandales) à 98 (manteaux), sur une échelle de 0 à 255.
**Dans tes notes** : les hauts (T-shirt, pull, manteau, chemise) se ressemblent le plus. Les sandales sont faites de fines lanières : peu de pixels allumés. Problème de **contenu** : des images floues, coupées, ambiguës ou mal étiquetées (on en verra en 23.20). Problème de **structure** : des images de tailles différentes, un type inattendu (des `float` de 0 à 1 au lieu d'entiers de 0 à 255), des labels décalés par rapport aux images, ou une classe absente.
**Erreurs fréquentes** : rendre le label (9) au lieu du nom ; compter le train et le test ensemble (7 000) ; `ax.imshow(image)` sans `cmap` (des fausses couleurs) ; `axes[k][j]` sur une grille d'une seule ligne (un tableau 1-D).
**Variante** : la classe dont la plus grande part des pixels est allumée (non nulle) est **Pullover** (65 % des pixels) ; la moins remplie, **Sandal** (32 %).

### Ex 23.14 — Oublier la conversion en float ou la normalisation : erreur ou entraînement lent ? 🔮
**Réponses** : a) **C** ; b) **False** ; c) **True** ; d) **False**. L'expérience : loss d'entraînement de la 1ʳᵉ epoch 0,69 (normalisés), 0,88 (divisés par 255), **25,4** (bruts) ; accuracy de validation après 3 epochs 83,3 %, 82,8 % et 77,8 % ; aucune loss `nan` ; les pixels `uint8` lèvent `RuntimeError: mat1 and mat2 must have the same dtype, but got Byte and Float`.
**Démarche** : des entrées 255 fois trop grandes donnent des premiers logits de l'ordre de plusieurs dizaines ; une erreur commise avec de tels logits coûte des dizaines de nats, d'où une première loss moyenne plus de trente fois plus grande. Mais Adam divise chaque gradient par la racine de la moyenne de ses carrés récents : ses pas restent de l'ordre du learning rate, rien ne déborde, et le réseau rattrape une bonne partie du retard en trois epochs. La normalisation complète devance la simple division par 255 de 3,5 points après une epoch, d'un demi-point après trois.
**Dans tes notes** : avec SGD, le pas est le learning rate fois le gradient ; des gradients 255 fois trop grands donnent des pas 255 fois trop grands (voir la variante). On normalise toujours parce que le réglage du learning rate, l'initialisation et les réseaux plus profonds supposent des entrées de taille 1 ; un MLP avec Adam est seulement indulgent.
**Erreurs fréquentes** : croire que l'oubli de la normalisation lève une erreur (seul l'oubli de la conversion en `float` en lève une) ; regarder l'accuracy de la 1ʳᵉ epoch au lieu de la 3ᵉ ; confondre « l'écart existe » et « l'écart dépasse 3 points ».
**Variante** : avec SGD (learning rate 0,1) au lieu d'Adam, et les pixels bruts, la loss moyenne de la 1ʳᵉ epoch dépasse **2 millions**, puis le réseau reste bloqué vers **2,3**, soit $\ln 10$ : les pas géants ont rendu les ReLU muettes, et il n'apprend plus rien. Avec les pixels divisés par 255, le même SGD descend à 0,70 en trois epochs.

### Ex 23.15 — Tout le prétraitement au même endroit : split stratifié, float32, normalisation, DataLoader 📦
**Réponses** : a) **600** ; b) **0,2858** ; c) **0,3529**.
**Démarche** :
```python
def split_15(X, y, val_size=0.1, seed=0):
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)


def fit_stats_15(images):
    x = np.asarray(images) / 255.0
    return float(x.mean()), float(x.std())


def preprocess_15(images, mean, std):
    x = (np.asarray(images, dtype=np.float32) / 255.0 - np.float32(mean)) / np.float32(std)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1)          # a new array: the input is untouched


def make_loaders_15(X_tr, y_tr, X_val, y_val, mean, std, batch_size=256, seed=0):
    train_ds = TensorDataset(preprocess_15(X_tr, mean, std), torch.as_tensor(np.asarray(y_tr), dtype=torch.int64))
    val_ds = TensorDataset(preprocess_15(X_val, mean, std), torch.as_tensor(np.asarray(y_val), dtype=torch.int64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    return train_loader, DataLoader(val_ds, batch_size=1024)
```
a) Stratifié, le découpage garde 10 % de chaque classe : 600 chemises sur 6 000. b) et c) La moyenne et l'écart-type des pixels divisés par 255, sur les 54 000 images d'entraînement.
**Dans tes notes** : la validation joue le rôle de données jamais vues : ses statistiques ne doivent pas entrer dans le prétraitement (fuite, ch. 12). Un générateur propre au loader rend l'ordre des mini-batches indépendant de tout ce qui a tiré des nombres avant (l'initialisation du modèle, un dropout) : changer l'architecture ne change pas l'ordre des données.
**Erreurs fréquentes** : oublier `stratify=y` (un autre nombre de chemises) ; calculer les statistiques sur les pixels de 0 à 255 (72,9 au lieu de 0,2858) ou sur toutes les images ; `(x - np.float64(mean)) / std`, qui rend du `float64` sous NumPy 2 ; `x -= mean` sur le tableau reçu, qui le modifie ; `shuffle=True` pour la validation.
**Variante** : sur les 60 000 images (validation comprise, à tort), la moyenne vaudrait **0,2860** et l'écart-type **0,3530** : l'écart est minuscule ici, mais c'est une fuite, et sur de petites données elle peut fausser l'évaluation.

### Ex 23.16 — Le MLP du livre avec nn.Sequential et nn.Flatten 📦
**Réponses** : a) **79 510** ; b) **(7, 10)**.
**Démarche** :
```python
def make_mlp_16(n_hidden=784, n_classes=10):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_classes))


def make_lazy_mlp_16(n_hidden=784, n_classes=10):
    return nn.Sequential(nn.Flatten(), nn.LazyLinear(n_hidden), nn.ReLU(), nn.LazyLinear(n_classes))


def count_params_16(model):
    return sum(p.numel() for p in model.parameters())
```
a) $(784 + 1) \times 100 + (100 + 1) \times 10 = 78\,500 + 1\,010$. b) `Flatten` garde l'axe des exemples : 7 lignes de 10 logits. La loss du modèle non entraîné vaut 2,295, proche de $\ln 10$.
**Dans tes notes** : à l'initialisation, les logits sont petits et presque égaux, la softmax donne environ 1/10 à chaque classe, et la loss vaut à peu près $-\ln(1/10)$. `nn.LazyLinear` dispense de calculer les tailles d'entrée, mais le modèle n'a pas de paramètres avant le premier passage : on ne peut ni les compter, ni créer l'optimiseur, ni charger un `state_dict`.
**Erreurs fréquentes** : une `nn.Softmax` en dernière couche ; oublier `nn.Flatten` (`mat1 and mat2 shapes cannot be multiplied (28x28 and 784x784)` : la couche reçoit les lignes de chaque image) ; `nn.Linear(n_hidden, n_classes)` écrit `nn.Linear(784, n_classes)` quand `n_hidden` change.
**Variante** : `make_mlp_16(n_hidden=50, n_classes=5)` a $(784 + 1) \times 50 + (50 + 1) \times 5 =$ **39 505** paramètres.

### Ex 23.17 — Trois bugs d'inférence : eval() oublié, gradients calculés pour rien (inference_mode), chargement non sûr (weights_only) 🐛
**Réponse** :
```python
def predict_17(model, images):
    model.eval()                                  # bug 1: dropout off (and batchnorm on its running statistics)
    with torch.inference_mode():                  # bug 2: no graph, no gradient
        return model(preprocess(images)).argmax(dim=1)


def load_weights_17(path):
    return torch.load(path, weights_only=True)   # bug 3: never unpickle arbitrary objects
```
**Démarche** : le symptôme : 137 prédictions sur 200 changent entre deux appels identiques, parce que `model_17`, en mode entraînement, éteint au hasard la moitié de ses neurones cachés à chaque passage. Une fois `model.eval()` ajouté, le deuxième diagnostic regarde, par un *hook*, dans quel mode se fait la passe avant : sans `torch.inference_mode()`, PyTorch enregistre le graphe de calcul de chaque prédiction. Le troisième charge le fichier piégé : avec `weights_only=False`, `pickle` appelle la fonction que le fichier lui désigne (ici `trap_17`, qui lève un drapeau) ; avec `weights_only=True`, il refuse (`UnpicklingError: Weights only load failed`), et les vrais poids, un dictionnaire de tenseurs, se chargent normalement.
**Dans tes notes** : le premier bug fausse les résultats, le deuxième coûte de la mémoire et du temps, le troisième est une faille de sécurité. Sans dropout ni batchnorm, le premier ne se verrait pas, jusqu'au jour où l'on en ajoute : on appelle `model.eval()` dans tous les cas.
**Erreurs fréquentes** : `torch.no_grad()` au lieu de `torch.inference_mode()` (les gradients sont coupés, mais pas avec l'outil demandé) ; `model.eval` sans parenthèses (rien ne se passe) ; mettre `model.eval()` dans la cellule d'appel au lieu de la fonction ; « réparer » le chargement avec un `try`/`except` autour de `weights_only=False`.
**Variante** : sous `torch.no_grad()`, la sortie a `requires_grad` **False** et `is_inference()` **False** ; sous `torch.inference_mode()`, `is_inference()` vaut **True**, et ce tenseur ne pourra plus servir à un calcul qui demande un gradient.

### Ex 23.18 — Ta fonction fit() : boucle, validation et historique 📦
**Réponse** :
```python
@torch.inference_mode()
def evaluate_18(model, loader, loss_fn, device="cpu"):
    model.eval()
    total, correct, count = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += loss_fn(logits, yb).item() * len(xb)           # loss_fn gives the MEAN of the mini-batch
        correct += (logits.argmax(dim=1) == yb).sum().item()
        count += len(xb)
    return total / count, correct / count


def fit_18(model, train_loader, val_loader, loss_fn, optimizer, epochs, callbacks=(), device="cpu"):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()                                            # evaluate_18 left the model in evaluation mode
        total, correct, count = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            count += len(xb)
        val_loss, val_acc = evaluate_18(model, val_loader, loss_fn, device)
        for key, value in (("train_loss", total / count), ("train_acc", correct / count),
                           ("val_loss", val_loss), ("val_acc", val_acc)):
            history[key].append(float(value))
        stop = [bool(callback(epoch, history, model, optimizer)) for callback in callbacks]   # call them ALL
        if any(stop):
            break
    return history
```
**Démarche** : trois epochs d'Adam sur 12 000 images : loss d'entraînement 0,687, 0,463, 0,391 ; accuracy de validation 82,0 %, 83,0 %, 85,5 %. La pondération par `len(xb)` donne la vraie moyenne sur toutes les images, même quand le dernier mini-batch est plus petit (le test à learning rate nul compare la loss d'entraînement à `evaluate_18` sur les mêmes 1 000 images). La liste en compréhension appelle **tous** les callbacks avant de décider : un `any(...)` sur un générateur s'arrêterait au premier `True`, et le meilleur checkpoint pourrait ne pas être écrit.
**Dans tes notes** : la loss d'entraînement d'une epoch est une moyenne prise pendant que les poids s'améliorent : ses premiers mini-batches ont été vus par un modèle moins bon, d'où une valeur un peu plus haute que celle du modèle final. `evaluate_18` met le modèle en mode évaluation ; sans `model.train()` au début de chaque epoch, le dropout et la batchnorm y resteraient pour toute la suite, sans erreur.
**Erreurs fréquentes** : moyenner les loss des mini-batches sans les pondérer (faux quand le dernier est incomplet) ; `total += loss` (le graphe de chaque mini-batch reste en mémoire) ; `model.train()` une seule fois avant la boucle ; appeler les callbacks avant d'ajouter l'epoch à l'historique ; compter les epochs à partir de 0 pour les callbacks ; `return` d'un tenseur au lieu d'un `float`.
**Variante** : ajoute à l'historique la durée de chaque epoch (`time.perf_counter()` avant et après) : environ **0,3 s** par epoch sur 12 000 images et 2 CPU, 1,5 s sur les 54 000.

### Ex 23.19 — Lire l'historique : 3 epochs, puis 50 📈
**Réponses** : a) **8** ; b) **29** ; c) **1,81** ; d) **−0,1** ; e) **18**.
**Démarche** :
```python
val_loss_19, val_acc_19 = np.array(HISTORY_19["val_loss"]), np.array(HISTORY_19["val_acc"])
best_loss_epoch_19 = int(np.argmin(val_loss_19)) + 1          # epochs are counted from 1, indices from 0
best_acc_epoch_19 = int(np.argmax(val_acc_19)) + 1
loss_ratio_19 = float(val_loss_19[-1] / val_loss_19.min())
acc_change_19 = float((val_acc_19[-1] - val_acc_19[best_loss_epoch_19 - 1]) * 100)
first_95_19 = int(np.argmax(np.array(HISTORY_19["train_acc"]) >= 0.95)) + 1
```
La loss de validation est au plus bas à l'epoch 8 (0,284), puis remonte jusqu'à 0,514 à l'epoch 50. L'accuracy de validation, elle, continue de monter un peu (90,5 % à l'epoch 29) et finit à 89,8 %, un dixième de point sous celle de l'epoch 8 : la loss a presque doublé sans que l'accuracy bouge.
**Dans tes notes** : la 3ᵉ epoch seule fait croire à une remontée (sa loss de validation passe de 0,342 à 0,367) : c'est un soubresaut, pas encore l'overfitting, que montre la suite. Au sens de la loss, l'overfitting commence après l'epoch 8 : la loss d'entraînement continue de baisser, celle de validation remonte. Le modèle devient de plus en plus sûr de lui, y compris quand il se trompe (💼 23.E4). On garderait l'epoch 8 si l'on se sert des probabilités ; si seule la classe compte, une epoch vers 20 à 30 donne un demi-point d'accuracy de plus, à condition d'avoir choisi ce critère **avant** de regarder.
**Erreurs fréquentes** : oublier le `+ 1` (l'indice 7 est l'epoch 8) ; chercher la meilleure epoch sur la loss d'entraînement ; inverser le rapport de c) ; donner d) en fraction (−0,001) au lieu de points.
**Variante** : la plus petite loss d'**entraînement** est à l'epoch **48** (0,041) : sur l'entraînement, tout s'améliore encore à la fin, ce qui est justement le signe de l'overfitting.

### Ex 23.20 — Analyser les erreurs : matrice de confusion et grille des confusions 📦
**Réponse** :
```python
@torch.inference_mode()
def predict_all_20(model, loader, device="cpu"):
    model.eval()
    truths, preds = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(dim=1).cpu())
        truths.append(yb)
    return torch.cat(truths).numpy().astype(np.int64), torch.cat(preds).numpy().astype(np.int64)


def top_confusions_20(C, k=3):
    C = np.asarray(C)
    cells = [(int(C[i, j]), i, j) for i in range(C.shape[0]) for j in range(C.shape[1]) if i != j]
    cells.sort(key=lambda cell: (-cell[0], cell[1], cell[2]))
    return [(i, j, count) for count, i, j in cells[:k]]


def show_confusions_20(images, y_true, y_pred, true_class, pred_class, n=8):
    chosen = np.flatnonzero((np.asarray(y_true) == true_class) & (np.asarray(y_pred) == pred_class))[:n]
    fig, axes = plt.subplots(1, n, figsize=(n * 1.1, 1.6))
    for ax, index in zip(np.atleast_1d(axes), list(chosen) + [None] * n):
        if index is not None:
            ax.imshow(images[index], cmap="gray_r")
        ax.axis("off")
    fig.suptitle(f"{CLASSES[true_class]} prédit {CLASSES[pred_class]} ({len(chosen)} montrées)", fontsize=9)
    plt.show()
```
**Démarche** : après 5 epochs sur 12 000 images, l'accuracy de validation est de 86,0 %. Les trois confusions les plus fréquentes : une chemise prise pour un T-shirt (105 fois), un manteau pour un pull (81), un pull pour un manteau (67).
**Dans tes notes** : les confusions restent entre les hauts du corps, souvent dans les deux sens (manteau et pull). Beaucoup d'images confondues sont ambiguës même à l'œil (une chemise sans boutons visibles, un pull à fermeture qui ressemble à un manteau), et quelques-unes semblent mal étiquetées : le modèle n'est pas « bête », la résolution et le dataset limitent ce qu'il peut faire.
**Erreurs fréquentes** : lire la matrice à l'envers (les lignes sont la vérité) ; garder la diagonale dans `top_confusions_20` ; trier sans règle d'ex aequo ; oublier `model.eval()` dans `predict_all_20` ; prendre des tenseurs pour l'indexation NumPy des images.
**Variante** : sur la matrice symétrisée `C + C.T` de l'exemple `[[50, 3, 9], [2, 40, 9], [9, 1, 30]]`, qui compte les confusions dans les deux sens, la paire la plus confondue est **(0, 2)**, avec **18** erreurs.

### Ex 23.21 — Prédire la réaction du MLP à des images « du monde réel » 🔮
**Réponses** : a) **C** ; b) **True** ; c) **True** ; d) **False**. L'expérience : originales 88,7 % (confiance moyenne 0,89) ; décalées de 2 pixels 56,1 % (0,70) ; inversées 3,1 % (**0,95**) ; miroir 63,7 % (0,84).
**Démarche** : une couche dense relie chaque poids à **une** position de pixel ; décalées de 2 pixels, les bordures d'un vêtement tombent sur des poids appris pour d'autres positions, et un tiers des réponses deviennent fausses. Inversées, les images ont un fond clair que le modèle n'a jamais vu : il ne « sait pas qu'il ne sait pas », et la softmax rend quand même des probabilités très tranchées, souvent vers une même classe, d'où une accuracy sous le hasard. Le miroir coûte 25 points : dans Fashion-MNIST, toutes les chaussures pointent du même côté, et les fermetures et les poches des vêtements sont placées de la même façon.
**Dans tes notes** : une confiance élevée ne garantit rien hors des données d'entraînement ; pour savoir qu'une image « ne ressemble à rien de connu », il faut des méthodes dédiées (détection d'exemples hors distribution). Pour qu'un décalage ne coûte presque rien : un CNN, dont les filtres sont partagés sur toutes les positions (ch. 21), et l'augmentation de données, qui montre des images décalées pendant l'entraînement (ch. 24).
**Erreurs fréquentes** : croire qu'un modèle perdu donne des probabilités proches de 0,1 partout ; penser qu'un décalage de 2 pixels est invisible pour un réseau parce qu'il l'est pour un humain.
**Variante** : décalées de **4** pixels, les images ne sont plus reconnues qu'à environ **25 %** : l'accuracy chute vite avec le décalage.

### Ex 23.22 — Prédire sur de nouvelles images : eval, inference_mode, probabilités 📦
**Réponse** :
```python
def predict_proba_22(model, images, mean=MEAN, std=STD, device="cpu"):
    images = np.asarray(images)
    if images.ndim == 2:                                  # one image: a batch of one
        images = images[None]
    model.eval()
    with torch.inference_mode():
        logits = model(preprocess(images, mean, std).to(device))
    return logits.softmax(dim=1).cpu().numpy()


def top_k_22(probs, k=2):
    probs = np.asarray(probs)
    order = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return [[(CLASSES[j], float(row[j])) for j in idx] for row, idx in zip(probs, order)]


def save_preprocessing_22(path, mean, std, classes):
    config = {"mean": float(mean), "std": float(std), "classes": list(classes), "input_shape": [1, 28, 28]}
    Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_preprocessing_22(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
```
**Démarche** : la fonction met elle-même le modèle en mode évaluation (la vérification le laisse exprès en mode entraînement avant l'appel), applique **le même** prétraitement qu'à l'entraînement et rend des probabilités NumPy ; une image seule devient un mini-batch d'une image. Le fichier JSON contient tout ce qu'une application doit savoir pour préparer une image et nommer la sortie.
**Dans tes notes** : sans ce fichier, l'application ne connaît ni $\mu$, ni $\sigma$, ni l'ordre des classes. Avec des pixels de 0 à 255, le modèle répondrait sans erreur, mais faux, et sûr de lui (23.14 et 23.21).
**Erreurs fréquentes** : oublier la softmax (des logits, pas des probabilités) ; `json.dumps` d'un `np.float32` (`TypeError: Object of type float32 is not JSON serializable`) ; rendre un tenseur au lieu d'un tableau NumPy ; `np.argsort(probs)[:, :k]` (les $k$ **moins** probables).
**Variante** : `top_k_22(probs, k=3)` sur `[0.1, 0.6, 0, 0, 0, 0, 0.3, 0, 0, 0]` rend **[("Trouser", 0.6), ("Shirt", 0.3), ("T-shirt/top", 0.1)]**.

### Ex 23.23 — Reprendre un entraînement interrompu : checkpoint complet (modèle, optimiseur, epoch, générateurs aléatoires) 📦
**Réponse** :
```python
def save_checkpoint_23(path, model, optimizer, epoch, generator, config):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": int(epoch),
                "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(), "config": dict(config)}, path)


def resume_23(path, lr=1e-3):
    checkpoint = torch.load(path, weights_only=True)
    model = make_mlp(**checkpoint["config"])               # the architecture is code: rebuilt from its arguments
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer"])     # Adam's moments and step counts
    generator = torch.Generator()
    generator.set_state(checkpoint["loader_rng"])          # the order of the next mini-batches
    torch.set_rng_state(checkpoint["torch_rng"])           # the next dropout masks
    return model, optimizer, generator, checkpoint["epoch"]
```
**Démarche** : le checkpoint (2,5 Mo) contient six entrées, toutes des tenseurs ou des types simples, ce que `weights_only=True` accepte. Après 2 + 2 epochs, les poids sont identiques au bit près à ceux de 4 epochs d'affilée : même état d'Adam, même ordre des mini-batches, mêmes masques de dropout. L'optimiseur doit être créé sur les paramètres du **nouveau** modèle avant de recevoir son état.
**Dans tes notes** : chaque oubli casse l'égalité au bit près. Sans l'état d'Adam, ses moments repartent de zéro et ses premiers pas sont de la taille du learning rate pour chaque poids : une secousse visible sur la loss, l'oubli le plus coûteux. Sans les générateurs, l'ordre des mini-batches ou les masques de dropout changent : l'entraînement reste bon, mais n'est plus reproductible.
**Erreurs fréquentes** : sauver l'objet `model` lui-même (refusé par `weights_only=True`, et fragile) ; créer l'Adam sur l'ancien modèle ; oublier `torch.set_rng_state` ; recréer le `DataLoader` avec un nouveau générateur à graine fixe au lieu du générateur restauré ; sauver `epoch` comme un tenseur ou un `np.int64`.
**Variante** : sans l'état d'Adam, le fichier ne ferait plus qu'environ **0,8 Mo** : les 203 530 poids en `float32`, au lieu de trois fois plus avec les deux moments d'Adam.

### Ex 23.24 — Un modèle pré-entraîné torchvision : transforms, gel et nouvelle tête 📦
**Réponse** :
```python
from torchvision.transforms import v2


def make_transfer_24(weights, n_classes=10):
    model = resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad_(False)                                  # freeze the backbone
    model.fc = nn.Linear(model.fc.in_features, n_classes)        # a new head: its parameters are trainable
    return model


def to_resnet_input_24(images, weights, size):
    reference = weights.transforms()
    x = torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    resize = v2.Resize((size, size), antialias=True)
    return v2.Normalize(reference.mean, reference.std)(resize(x)).contiguous()


def extract_features_24(model, x, batch_size=250):
    head, model.fc = model.fc, nn.Identity()                     # everything but the head
    model.eval()
    try:
        with torch.inference_mode():
            return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    finally:
        model.fc = head
```
**Démarche** : 5 130 paramètres entraînables (la tête, $512 \times 10 + 10$) contre 11 176 512 gelés. Les features de 3 000 images à 112 pixels prennent une trentaine de secondes ; la tête, entraînée 20 epochs sur ces features, atteint **86,0 %** sur 1 000 images de validation, contre **83,5 %** pour le MLP du livre entraîné sur les mêmes 2 000 images. La nouvelle couche est créée **après** le gel : ses paramètres restent entraînables.
**Dans tes notes** : l'avantage est modeste. Les features d'ImageNet ont été apprises sur des photos en couleur d'objets du monde, à 224 pixels ; des vêtements gris de 28 pixels agrandis en sont loin, un déplacement de domaine. Le transfert brille quand les nouvelles images ressemblent à celles du pré-entraînement et qu'on en a peu, et plus encore quand on dégèle les dernières couches (le fine-tuning, chapitre bonus B1).
**Erreurs fréquentes** : geler **après** avoir remplacé `fc` (la tête est gelée aussi : 0 paramètre entraînable) ; oublier la normalisation d'ImageNet ; donner une image à 1 canal au réseau ; `model.eval()` oublié (la batchnorm du ResNet utiliserait les statistiques de chaque mini-batch) ; laisser `model.fc = nn.Identity()` après l'extraction.
**Variante** : dégeler aussi le dernier bloc (`model.layer4`) ajoute **8 393 728** paramètres entraînables : on passe de l'extraction de features au fine-tuning partiel, beaucoup plus coûteux, puisqu'il faut alors refaire la passe avant à chaque epoch.

### Ex 23.25 — Les callbacks en PyTorch : meilleur checkpoint, planning de LR, early stopping 📦
**Réponse** :
```python
class BestCheckpoint_25:
    def __init__(self, path, monitor="val_loss", mode="min"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.path, self.monitor, self.mode = Path(path), monitor, mode
        self.best_ = math.inf if mode == "min" else -math.inf
        self.best_epoch_ = None

    def __call__(self, epoch, history, model, optimizer):
        value = history[self.monitor][-1]
        better = value < self.best_ if self.mode == "min" else value > self.best_
        if better:
            self.best_, self.best_epoch_ = value, epoch
            torch.save(model.state_dict(), self.path)
        return False


class SchedulerCallback_25:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, epoch, history, model, optimizer):
        history.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(history["val_loss"][-1])
        else:
            self.scheduler.step()
        return False


class EarlyStoppingCallback_25:
    def __init__(self, patience, min_delta=0.0, monitor="val_loss"):
        self.stopper = mylearn.nn.regularization.EarlyStopping(patience=patience, min_delta=min_delta)
        self.monitor = monitor

    def __call__(self, epoch, history, model, optimizer):
        return bool(self.stopper.step(history[self.monitor][-1]))
```
**Démarche** : sur les historiques écrits à la main, le meilleur checkpoint garde l'epoch 4 (et non l'ex aequo de l'epoch 5 : « strictement meilleur ») ; le planning `StepLR` enregistre les learning rates 1, 1, 0,5, 0,5, 0,25 ; l'early stopping de patience 2 demande l'arrêt à l'epoch 6. Le vrai entraînement s'arrête après 13 epochs ; la meilleure est la 9ᵉ (val_loss 0,352), et le modèle rechargé retrouve exactement cette val_loss (accuracy 87,5 %).
**Dans tes notes** : l'ordre compte peu ici, puisque `fit_18` appelle tous les callbacks avant de s'arrêter et que chacun ne fait que lire l'historique ; il compterait si l'un modifiait le modèle ou l'historique avant les autres. À l'arrêt, le modèle a les poids de la **dernière** epoch, `patience` epochs après la meilleure ; recharger le fichier, c'est le `restore_best_weights` de Keras.
**Erreurs fréquentes** : `<=` au lieu de `<` (le checkpoint suit les ex aequo) ; lire le learning rate **après** `scheduler.step()` (décalé d'une epoch) ; `scheduler.step()` sans la val_loss pour un `ReduceLROnPlateau` (`TypeError`) ; créer l'`EarlyStopping` dans `__call__` (le compteur repart de zéro à chaque epoch) ; oublier de recharger le meilleur checkpoint à la fin.
**Variante** : avec une patience de 3 au lieu de 2, sur les mêmes loss `[0.9, 0.8, 0.82, 0.79, 0.85, 0.86, 0.70]`, l'early stopping ne demande **aucun arrêt** : le compteur atteint 2 à l'epoch 6, puis l'epoch 7 est une amélioration, la nouvelle meilleure.

### Ex 23.26 — Planning de learning rate : constant, par paliers, cosinus ou plateau 🔬
**Réponse** :
```python
def make_scheduler_26(name, optimizer, epochs):
    schedules = torch.optim.lr_scheduler
    if name == "constant":
        return schedules.LambdaLR(optimizer, lambda e: 1.0)
    if name == "step":
        return schedules.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    if name == "cosine":
        return schedules.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return schedules.ReduceLROnPlateau(optimizer, factor=0.3, patience=1)
    raise ValueError(f"unknown schedule: {name!r}")
```
**Démarche** : en 12 epochs (mode rapide), l'accuracy de validation finale est de 86,3 % avec le learning rate constant, 87,3 % par paliers, 87,7 % en cosinus et 87,7 % sur plateau ; la val_loss finale passe de 0,397 (constant) à 0,345 (cosinus).
**Dans tes notes** : les trois plannings qui font baisser le learning rate finissent environ un point au-dessus du constant, et leur loss de validation baisse nettement au moment où le learning rate tombe : de petits pas laissent les poids se poser au fond de la vallée au lieu d'osciller autour. Le plateau ne réagit qu'au bout de quelques epochs sans progrès. Un point d'écart sur une seule graine ne prouve rien : il faudrait répéter avec plusieurs graines et comparer moyennes et écarts, comme dans le défi 23.29.
**Erreurs fréquentes** : rendre `None` pour le planning constant (le callback n'a rien à faire avancer) ; `step_size=epochs // 3` sans le `max(1, ...)` (0 pour moins de 3 epochs) ; `CosineAnnealingLR(T_max=epochs - 1)`.
**Variante** : en cosinus sur 12 epochs, à partir de 0,05, le learning rate de l'epoch 6 (comptée à partir de 0) vaut $0{,}05 \times (1 + \cos(\pi \cdot 6/12))/2 =$ **0,025**, la moitié du départ.

### Ex 23.27 — Patience de l'early stopping : 1, 3 ou 10 🔬
**Réponses** : a) **3** ; b) **2** ; c) **18** ; d) **4**.
**Démarche** :
```python
def simulate_27(val_losses, patience, min_delta=0.0):
    stopper = mylearn.nn.regularization.EarlyStopping(patience=patience, min_delta=min_delta)
    for epoch, loss in enumerate(val_losses, start=1):
        if stopper.step(loss):
            return epoch, stopper.best_epoch_ + 1                # best_epoch_ is a 0-based index
    return None, stopper.best_epoch_ + 1
```
a) et b) La loss de validation remonte à l'epoch 3 : une patience de 1 s'arrête aussitôt et recharge l'epoch 2 (87,5 %). c) Avec une patience de 10, on s'arrête 10 epochs après l'epoch 8, la meilleure : à l'epoch 18. d) Avec `min_delta=0.01`, après l'epoch 4 (0,315), les baisses suivantes (0,315, puis 0,306 et 0,305) sont plus petites que 0,01 et ne comptent pas : le compteur atteint 3 à l'epoch 7, et l'on recharge l'epoch 4 (88,7 %).
**Dans tes notes** : une patience trop petite s'arrête sur le bruit et perd plus de deux points ; dès 2, l'arrêt retrouve l'epoch 8 (89,9 %), et une patience plus grande ne change plus rien au modèle rechargé : elle coûte seulement des epochs. `min_delta` peut faire manquer des progrès réels quand la loss avance par petits pas. Ici, une patience de 3 à 5, sans `min_delta`, est un bon choix.
**Erreurs fréquentes** : rendre `best_epoch_` sans `+ 1` (une epoch de trop tôt) ; compter les epochs à partir de 0 ; confondre l'epoch d'arrêt et l'epoch rechargée ; reprendre la règle de Keras 2.0, qui attendait une epoch de plus que sa patience.
**Variante** : avec une patience de 4, l'arrêt a lieu à l'epoch **12**, et l'on recharge encore l'epoch **8**.

### Ex 23.28 — Rédiger la fiche du modèle (model card) 🛠️
**Réponse** : une fiche possible (le notebook de solutions la construit avec une f-string, qui insère les chiffres) :

> **Détails du modèle** : le MLP du livre, 623 290 paramètres, Adam (learning rate 0,001), 5 epochs, graine 2321, PyTorch 2.11. **Usage prévu** : classer des images de vêtements de 28 × 28 pixels, en niveaux de gris, centrées sur fond noir, comme celles de Fashion-MNIST ; usage pédagogique ; hors du champ : des photos prises au téléphone, en couleur, décentrées ou sur fond clair, et toute décision qui concerne une personne. **Facteurs** : la classe (les hauts se confondent), le cadrage (2 pixels de décalage coûtent plus de 30 points), le contraste (des images inversées tombent sous le hasard), l'orientation (le miroir coûte 25 points). **Métriques** : accuracy de validation et accuracy par classe, matrice de confusion. **Données d'évaluation** : 6 000 images de validation, 600 par classe ; le test n'a pas servi. **Données d'entraînement** : Fashion-MNIST (Zalando, licence MIT), 54 000 images, découpage stratifié ; prétraitement : pixels / 255, puis $(x - 0{,}2858)/0{,}3529$, sauvegardé avec le modèle. **Analyses quantitatives** : accuracy de validation 0,8867 ; les classes les moins bien reconnues sont la chemise, le pull et le manteau. **Considérations éthiques** : des vêtements de catalogue surtout occidentaux, photographiés en studio ; le modèle ne doit pas servir à décrire ou juger des personnes. **Mises en garde et recommandations** : une confiance élevée ne garantit rien hors des données d'entraînement (0,95 sur des images inversées, classées faux) ; pour un usage réel, un CNN, de l'augmentation de données, une évaluation sur des photos du terrain et une mesure finale sur le test, une seule fois.

**Démarche** : chaque section répond à une question de Mitchell et coll. : qui a fait le modèle et comment, pour quoi il est fait (et pour quoi il ne l'est pas), ce qui fait varier ses résultats, comment on les mesure, sur quelles données, avec quels chiffres, quels risques et quelles précautions. Les chiffres viennent des mesures du chapitre, pas d'une impression.
**Erreurs fréquentes** : une fiche réduite à l'accuracy globale ; recopier les nombres à la main (ils divergent au prochain entraînement) ; oublier le prétraitement, sans lequel le modèle est inutilisable ; une section « Usage prévu » sans rien d'exclu.
**Variante** : ajoute une section « Versions et reproductibilité » (versions de Python, de PyTorch et de NumPy, graine, commit du code, empreinte SHA-256 du fichier de poids) : c'est ce qu'une équipe vérifie en premier quand un modèle se comporte autrement en production.

### Ex 23.29 — Défi reproductibilité : un MLP Fashion-MNIST à au moins 88 % dont l'accuracy varie de moins de 0,5 point sur 5 graines, en moins de 3 min de CPU 🏆
**Réponse** :
```python
def train_29(seed, x, y, epochs=8, batch_size=256):
    torch.manual_seed(seed)                                   # the weights, and the dropout if any
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)        # targets 0.91 / 0.01: less overconfident, more stable
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()                                    # OneCycleLR is stepped after each mini-batch
    return model
```
**Démarche** : validation 90,4 à 90,6 % (écart 0,27 point), test 89,65 à 89,81 % (écart 0,16 point), une minute pour les cinq graines sur 2 CPU. Le point de départ (le MLP du livre, Adam constant, 8 epochs) fait 87,4 à 88,6 % sur le test, avec un écart de 1,2 point. Deux ingrédients font la différence : un planning qui finit à un learning rate presque nul (`OneCycleLR`, appelé à chaque mini-batch), qui pose chaque modèle au fond de sa vallée au lieu de le laisser osciller, et le label smoothing, qui vise 0,91 au lieu de 1 pour la bonne classe et empêche le modèle de devenir trop sûr de lui.
**Erreurs fréquentes** : régler la recette en relançant la cellule jusqu'à ce que le test passe (le test est révélé une fois par session pour cette raison) ; oublier le générateur du `DataLoader` (l'ordre des mini-batches dépend alors de ce qui a été tiré avant) ; appeler `OneCycleLR.step()` une fois par epoch (le planning ne fait qu'une petite partie de son parcours) ; ajouter des epochs au-delà du budget de 3 minutes.
**Variante** : la même recette **sans** label smoothing fait 89,2 à 89,7 % sur le test, avec un écart de **0,55 point** : elle échoue de peu au critère de stabilité.
