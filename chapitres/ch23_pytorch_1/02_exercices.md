# 23 · PyTorch en pratique 1 : du dataset au modèle sauvegardé — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch23_pytorch_1/06_mes_reponses.md` (créée par `python tools/start_chapter.py 23`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 23.1 à 23.7 et de l'estimation 🧮 23.8 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), plusieurs lettres entre guillemets pour un choix multiple ou un ordre (`"EF"`), une forme ou un triplet d'indices entre guillemets (`"(5, 7, 1)"`), une liste de nombres entre crochets (`[9, 9, 9]`).
> Conventions du chapitre : celles de PyTorch (fiche). Les positions et les indices se comptent à partir de 0 ; les epochs se comptent à partir de 1 dans les énoncés, sauf pour `LambdaLR`, dont l'énoncé précise le compte. Un tableau est rangé en mémoire dans l'ordre C (le dernier indice varie le plus vite).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 23.Q1 — Pourquoi utiliser une bibliothèque plutôt que tout coder soi-même ? 🧠 ⏱️ 2 min
*Fiche §23.1, §23.2 · livre §23.1, §23.2*

a) Vrai ou faux : quand une bibliothèque existe, écrire son propre code n'apprend rien.
b) Ce que PyTorch t'apporte et que ta librairie `mylearn` n'a pas : (A) des formules plus justes ; (B) la vitesse sur GPU, la stabilité numérique et le calcul en parallèle, déjà optimisés ; (C) une accuracy garantie plus haute ; (D) la possibilité de se passer de données.
c) L'analogie du livre : utiliser une bibliothèque, c'est comme (A) programmer en assembleur pour mieux comprendre ; (B) apprendre une langue étrangère ; (C) écrire en Python plutôt qu'en code machine, pour penser au problème plutôt qu'au processeur ; (D) construire une maison sans plan.
d) Vrai ou faux : dans ce workbook, les fonctions de `mylearn` ont été comparées à PyTorch ou à scikit-learn, qui servaient d'oracle.
e) Vrai ou faux : une bibliothèque dispense de comprendre ce que fait un modèle.

### 23.Q2 — Avertissement ou erreur ? Versions et dépréciations 🧠 ⏱️ 3 min
*Fiche §23.1.3, §23.2.1 · livre §23.1.3, §23.2.1*

a) Un `FutureWarning` s'affiche sous une cellule. Cela veut dire : (A) le programme s'est arrêté ; (B) le code tourne encore, mais un comportement ou une API va changer ; (C) le résultat est faux ; (D) la bibliothèque n'est pas installée.
b) Vrai ou faux : une exception non rattrapée (`RuntimeError`, `ValueError`) arrête l'exécution de la cellule.
c) Vrai ou faux : le code Keras 2.0.6 du livre tourne tel quel avec Keras 3.
d) Pour qu'un notebook donne les mêmes résultats dans un an : (A) installer la dernière version de chaque bibliothèque ; (B) faire taire les avertissements ; (C) éviter toute bibliothèque ; (D) noter les versions exactes des bibliothèques et fixer les graines.
e) Vrai ou faux : écrire `warnings.filterwarnings("ignore")` en tête d'un notebook est une bonne pratique.

### 23.Q3 — Déboguer méthodiquement : réduire, isoler, reconstruire 🧠 ⏱️ 3 min
*Fiche §23.2.2 · livre §23.2.2*

a) Un appel plante au fond d'une bibliothèque que tu ne connais pas. La méthode que conseille le livre : (A) lire tout le code source de la bibliothèque ; (B) relancer jusqu'à ce que ça passe ; (C) réduire l'appel fautif, isoler ce qui provoque l'erreur, puis reconstruire pas à pas à partir d'un exemple qui marche ; (D) changer de bibliothèque.
b) Vrai ou faux : l'instruction `breakpoint()` arrête l'exécution et ouvre le débogueur de Python à cet endroit.

Un modèle `nn.Sequential(nn.Flatten(), nn.Linear(784, 784), nn.ReLU(), nn.Linear(748, 10))` lève :

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x784 and 748x10)
```

c) La taille du mini-batch.
d) La valeur que doit avoir le premier argument de la dernière couche (`in_features`).
e) Vrai ou faux : une erreur levée au fond d'une bibliothèque vient forcément d'un bug de la bibliothèque.

### 23.Q4 — Les trois sens du mot « modèle » 🧠 ⏱️ 3 min
*Fiche §23.3, §23.3.1 · livre §23.3, §23.3.1*

Les trois sens : (A) l'architecture, le plan des couches ; (B) l'architecture avec ses poids appris ; (C) l'API, la façon d'écrire et d'entraîner un réseau. Donne la lettre du sens de chaque phrase.

a) « Le modèle `Flatten`, `Linear(784, 784)`, `ReLU`, `Linear(784, 10)` », dessiné au tableau avant tout entraînement.
b) « Je t'envoie le modèle » : le fichier de poids appris et la fonction qui reconstruit le réseau, pour prédire sans réentraîner.
c) « Le modèle de programmation de PyTorch » : `torch.nn`, `torch.optim`, `torch.utils.data`.
d) Vrai ou faux : `model.state_dict()` contient l'architecture du réseau.
e) Vrai ou faux : une suite de couches dont les formes s'enchaînent sans erreur peut ne rien apprendre d'utile.

### 23.Q5 — Tenseurs, arrays et type float32 🧠 ⏱️ 3 min
*Fiche §23.3.2, §23.3.3, §23.5.5 · livre §23.3.2, §23.3.3, §23.5.5*

a) Le nombre d'axes (`ndim`) d'un mini-batch d'images de Fashion-MNIST au format de PyTorch.
b) Le type de `torch.tensor([0.5, 1.0])` : (A) `float16` ; (B) `float32` ; (C) `float64` ; (D) `uint8`.
c) Le type de `torch.from_numpy(np.array([0.5, 1.0]))` (mêmes choix).
d) Vrai ou faux : `torch.from_numpy(a)` partage la mémoire du tableau `a` : modifier l'un modifie l'autre.
e) Le nombre d'octets d'un nombre `float32`.

### 23.Q6 — channels_first, channels_last et NCHW 🧠 ★★ ⏱️ 3 min
*Fiche §23.3.4, §23.5.1 · livre §23.3.4, §23.5.1 · parcours R*

Une photo en couleur (3 canaux) fait 64 pixels de large et 48 de haut.

a) Sa forme en `channels_first`, sans l'axe des exemples.
b) Sa forme en `channels_last`.
c) La forme d'entrée qu'attend `nn.Conv2d` : (A) `(N, C, H, W)` ; (B) `(N, H, W, C)` ; (C) `(C, N, H, W)` ; (D) `(N, W, H, C)`.
d) Les arguments de `x.permute(...)` qui passent un mini-batch `x` de NHWC à NCHW, écrits comme un tuple entre guillemets.
e) Vrai ou faux : une image `channels_last` prise par erreur pour une image `channels_first` provoque toujours une erreur.

### 23.Q7 — CPU, GPU, MPS, TPU : où calcule PyTorch ? 🧠 ⏱️ 2 min
*Fiche §23.3.5 · livre §23.3.5*

a) Le device du GPU d'un Mac récent : (A) `"cuda"` ; (B) `"metal"` ; (C) `"mps"` ; (D) `"tpu"`.
b) Vrai ou faux : un modèle sur `"cuda"` appliqué à un tenseur resté sur `"cpu"` lève une erreur.
c) Pourquoi un GPU accélère-t-il l'entraînement ? (A) son horloge est plus rapide ; (B) il a plus de mémoire que le processeur ; (C) il fait en parallèle un grand nombre de calculs identiques, comme ceux d'un produit de matrices ; (D) il calcule en entiers.
d) Vrai ou faux : sur Colab, `torch.cuda.is_available()` vaut `True` seulement si l'on a choisi un environnement d'exécution avec GPU.
e) Vrai ou faux : PyTorch se sert d'un TPU de Colab avec `device="cuda"`.

### 23.Q8 — Iris, MNIST, Fashion-MNIST : les « hello world » du ML 🧠 ⏱️ 3 min
*Fiche §23.4, §23.4.1, §23.5, §23.5.2 · livre §23.4, §23.4.1, §23.5, §23.5.2*

a) Le nombre d'images d'entraînement de Fashion-MNIST.
b) Son nombre de classes.
c) Pourquoi ses auteurs l'ont-ils créé ? (A) MNIST est trop facile, trop utilisé et ne ressemble pas aux tâches modernes de vision ; (B) MNIST est devenu payant ; (C) MNIST a été retiré pour des raisons éthiques ; (D) MNIST est en couleur.
d) Vrai ou faux : les images de Fashion-MNIST ont la même taille, et le dataset le même découpage train/test, que MNIST.
e) Le nombre de mesures (features) de chaque fleur du dataset Iris.

### 23.Q9 — Labels entiers, one-hot et logits 🧠 ★★ ⏱️ 3 min
*Fiche §23.5.7, §23.6.3 · livre §23.5.7, §23.6.3 · parcours R*

a) Le premier argument qu'attend `nn.CrossEntropyLoss` : (A) des probabilités, après une softmax ; (B) des logits, sans softmax ; (C) des vecteurs one-hot ; (D) des labels entiers.
b) Le second argument, dans l'usage courant du chapitre : (A) des one-hot `float32` de forme `(N, 10)`, obligatoirement ; (B) des indices de classe `int64` de forme `(N,)` ; (C) les noms des classes ; (D) des `float32` de forme `(N,)`.
c) Vrai ou faux : un modèle qui se termine par `nn.Softmax(dim=1)`, entraîné avec `nn.CrossEntropyLoss`, lève une erreur.
d) La forme de `F.one_hot(torch.tensor([3, 0, 9]), num_classes=10)`.
e) La forme de `F.one_hot(torch.tensor([0, 3, 8]))`, sans `num_classes`.

### 23.Q10 — fit() décortiqué : batch, epoch, validation, train() et eval() 🧠 ⏱️ 3 min
*Fiche §23.6, §23.6.4, §23.7, §23.8 · livre §23.6, §23.6.4, §23.7, §23.8 · parcours R*

a) Vrai ou faux : `model.eval()` coupe le calcul des gradients.
b) Ce que changent `model.train()` et `model.eval()` : (A) le learning rate ; (B) le comportement du dropout et de la batchnorm ; (C) le device ; (D) le nombre d'epochs.
c) Le livre donne à `fit()` le jeu de **test** de MNIST comme données de validation, et choisit sur lui le nombre d'epochs. Ce qui ne va pas : (A) rien, ce sont des données que le modèle n'a jamais vues ; (B) le test est trop petit ; (C) Keras l'interdit ; (D) le test sert à choisir, et sa mesure finale n'est plus une estimation honnête.
d) Vrai ou faux : la loss d'entraînement affichée pour une epoch est la moyenne des loss des mini-batches pendant l'epoch, alors que les poids changent.
e) L'équivalent PyTorch du `compile()` de Keras : (A) `torch.compile(model)` ; (B) `model.eval()` ; (C) créer l'objet loss et l'objet optimiseur ; (D) `model.to(device)`.

### 23.Q11 — Sauvegarder tout, les poids ou l'architecture ? 🧠 ★★ ⏱️ 3 min
*Fiche §23.9 à §23.9.3 · livre §23.9 à §23.9.3 · parcours R*

a) Ce qu'il faut sauvegarder pour **reprendre** un entraînement interrompu comme s'il ne l'avait pas été : (A) le `state_dict` du modèle seul ; (B) l'objet `model`, avec `torch.save(model)` ; (C) les poids, l'état de l'optimiseur, l'epoch et l'état des générateurs aléatoires ; (D) les prédictions sur le test.
b) Ce que contient un `state_dict` : (A) l'architecture et les poids ; (B) les tenseurs des paramètres et des buffers, rangés par nom ; (C) le code des classes ; (D) l'état de l'optimiseur.
c) Vrai ou faux : avec PyTorch 2.6 ou plus récent, `torch.load(path, weights_only=True)` exécute le code Python que peut contenir un fichier pickle.
d) Le nombre de tenseurs du `state_dict` du MLP du livre (`Flatten`, `Linear(784, 784)`, `ReLU`, `Linear(784, 10)`).
e) Vrai ou faux : pour charger un `state_dict`, il faut d'abord reconstruire un modèle de la même architecture.

### 23.Q12 — 🕰️ Que devient le code Keras du livre en 2026 ? 🧠 ⏱️ 3 min
*Fiche §23.3.3 et ses encadrés 🕰️ · livre §23.1, §23.3.3 · parcours R*

a) Les backends de Keras 3 : (A) Theano, TensorFlow et CNTK ; (B) JAX, TensorFlow et PyTorch ; (C) PyTorch seulement ; (D) TensorFlow seulement.
b) Ce qui remplace `model.predict_classes(x)` : (A) `predict`, puis l'argmax ; (B) `predict_proba` ; (C) `to_categorical` ; (D) `fit`.
c) Vrai ou faux : `torch.compile` est l'équivalent PyTorch du `compile()` de Keras.
d) Vrai ou faux : Theano et CNTK sont toujours développés.
e) Le format d'enregistrement natif de Keras 3 : (A) `.yaml` ; (B) `.pt` ; (C) `.keras` ; (D) `.h5`, seulement.

<a id="rappels"></a>

## 🔁 Rappels

### 23.R1 — Ch. 22 : (N, T, D) et batch_first 🔁 ★ ⏱️ 5 min
*Ch. 22 (§22.5, §22.8.2) · fiche §23.3.4 · parcours R*

Un mini-batch de 32 phrases de 20 tokens, chaque token représenté par un vecteur de 50 valeurs, entre dans une `nn.GRU`.

a) Sa forme avec `batch_first=True`.
b) Sa forme sans `batch_first` (la valeur par défaut).
c) La forme de `h_n` rendu par `nn.GRU(50, 64, num_layers=2)`, à un seul sens, pour ce mini-batch.
d) Vrai ou faux : `batch_first=True` change la forme de `h_n`.
e) Vrai ou faux : dans `(N, C, H, W)` comme dans `(N, T, D)` avec `batch_first=True`, l'axe 0 est celui des exemples.

### 23.R2 — Ch. 20 : les cinq lignes d'une étape d'entraînement 🔁 ★ ⏱️ 5 min
*Ch. 20 (§20.5) · fiche §23.7 · parcours R*

Les cinq lignes d'une étape d'entraînement, dans le désordre : (A) `loss.backward()` ; (B) `optimizer.zero_grad()` ; (C) `logits = model(xb)` ; (D) `optimizer.step()` ; (E) `loss = loss_fn(logits, yb)`.

a) Un ordre correct qui commence par `optimizer.zero_grad()`, écrit comme une suite de lettres.
b) Vrai ou faux : sans `optimizer.zero_grad()`, les gradients s'additionnent d'un mini-batch au suivant.
c) Ce que fait `loss.backward()` : (A) il met à jour les poids ; (B) il calcule les gradients et les range dans le `.grad` de chaque paramètre ; (C) il remet les gradients à zéro ; (D) il calcule la loss.
d) Vrai ou faux : dans la boucle, `total += loss` garde en mémoire les graphes de calcul de tous les mini-batches, alors que `total += loss.item()` ne les garde pas.
e) Pour un planning de learning rate par epoch, on appelle `scheduler.step()` : (A) après chaque `optimizer.step()` ; (B) une fois par epoch, après les `optimizer.step()` de l'epoch ; (C) avant le premier `optimizer.step()` ; (D) jamais : l'optimiseur s'en charge.

### 23.R3 — Ch. 2 : moyenne et écart-type du train pour normaliser les pixels 🔁 ★ ⏱️ 5 min
*Ch. 2 (moyenne, écart-type) · fiche §23.5.6 · parcours R, M*

Les pixels d'un minuscule train, déjà divisés par 255, valent $0 ;\ 0{,}2 ;\ 0{,}4 ;\ 0{,}4 ;\ 1$.

a) Leur moyenne $\mu$.
b) Leur écart-type $\sigma$, avec le diviseur $n$ (`ddof=0`, comme `np.std`), à 3 décimales.
c) La valeur normalisée $(x - \mu)/\sigma$ d'un nouveau pixel de valeur $0{,}9$, à 3 décimales.
d) Vrai ou faux : pour normaliser le jeu de test, on recalcule la moyenne et l'écart-type sur le test.
e) Ce que calcule `v2.Normalize(mean, std)` de torchvision sur chaque valeur $x$ : (A) $(x - \text{mean})/\text{std}$ ; (B) $(x - \min)/(\max - \min)$ ; (C) $x / 255$ ; (D) $x \cdot \text{std} + \text{mean}$.

<a id="papier"></a>

## ✏️ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 23.1 — Remodeler 12 éléments : indices et ordre de lecture ✏️ ★ ⏱️ 10 min
**Objectif :** trouver où atterrit un élément quand on remodèle un tenseur, et savoir quand l'ordre de lecture change.
**Prérequis :** 0A (tableaux NumPy, `reshape`) · fiche §23.5.1 · **Parcours :** R, M

Les douze lettres de M à X sont rangées dans un tenseur `t` à un seul axe : M est à la position 0, N à la position 1… et X à la position 11. On le remodèle (`t.reshape(...)`, ordre C).

a) Après `reshape(3, 4)`, la lettre à l'indice `(2, 1)`.
b) Après `reshape(4, 3)`, la lettre à l'indice `(2, 1)`.
c) Après `reshape(2, 3, 2)`, la lettre à l'indice `(1, 2, 0)`.
d) Après `reshape(2, 2, 3)`, l'indice de la lettre Q, écrit comme un triplet entre guillemets.
e) Parmi ces formes, lesquelles sont **impossibles** pour 12 éléments ? (A) `(5, 3)` ; (B) `(2, 6)` ; (C) `(3, 2, 2)` ; (D) `(4, 4)` ; (E) `(12,)`. Écris leurs lettres.
f) `a = np.arange(12).reshape(3, 4)` : les quatre premières valeurs de `a.T.reshape(12)`, en liste.
g) Pour un tenseur de forme $(2, 3, 2)$, la position en mémoire (l'indice plat) de l'élément d'indice `(1, 2, 1)`.

Dans ta copie : pourquoi `a.T.reshape(12)` ne rend-il pas `0, 1, 2, …` ? Que fait `reshape` dans ce cas, que `view` refuserait ?

### Ex 23.2 — De NHWC à NCHW : où atterrit chaque pixel ? ✏️ ★★ ⏱️ 15 min
**Objectif :** suivre les indices d'un pixel à travers `permute` et comprendre pourquoi `reshape` ne convertit pas un format d'image.
**Prérequis :** Ex 23.1 · fiche §23.3.4, §23.5.1 · **Parcours :** M

Une image en couleur de 2 lignes et 3 colonnes est rangée en `channels_last` dans un tenseur `x` de forme `(2, 3, 3)` : `x[h, w, c]` est la valeur du canal `c` (0 rouge, 1 vert, 2 bleu) du pixel de la ligne `h` et de la colonne `w`. Les 18 valeurs sont en mémoire dans l'ordre C.

a) La position en mémoire de `x[1, 2, 0]`, le rouge du pixel de la ligne 1 et de la colonne 2.
b) La forme de `y = x.permute(2, 0, 1)`.
c) `y[0, 1, 2]` est la même valeur que `x[?, ?, ?]` : écris le triplet d'indices de `x`, entre guillemets.
d) Après `z = y.contiguous()`, la position en mémoire de `z[0, 1, 2]`.
e) Vrai ou faux : `y.view(18)` fonctionne sans erreur.
f) Un collègue convertit avec `w = x.reshape(3, 2, 3)`. La valeur `w[0, 0, 1]` est : (A) le rouge du pixel (0, 1) ; (B) le vert du pixel (0, 0) ; (C) le bleu du pixel (0, 0) ; (D) aucune : `reshape` lève une erreur.
g) Et la valeur `y[0, 0, 1]`, avec la bonne conversion (mêmes choix) ?
h) Un mini-batch de 64 images en couleur de 32 × 32 pixels en `float32` : sa taille en octets.

Dans ta copie : pourquoi la conversion du collègue ne lève-t-elle aucune erreur ? Comment l'aurais-tu détectée ?

### Ex 23.3 — 623 290 paramètres : refaire le compte du livre ✏️ ★ ⏱️ 10 min
**Objectif :** compter les paramètres d'un MLP couche par couche, et voir où ils se concentrent.
**Prérequis :** ch. 16 (couche dense) · fiche §23.6.2 · **Fil rouge :** Fashion-MNIST · **Parcours :** R, M

Le MLP du livre : `Flatten`, `Linear(784, 784)`, `ReLU`, `Linear(784, 10)`.

a) Le nombre de paramètres de `Linear(784, 784)`.
b) Celui de `Linear(784, 10)`.
c) Le total du modèle.
d) La part du total qui est dans la première couche, en fraction (entre 0 et 1), à 3 décimales.
e) Le total si la couche cachée a 256 neurones au lieu de 784.
f) Le total d'un MLP `784 → 512 → 256 → 10` (deux couches cachées, avec biais).
g) Le nombre de paramètres de `nn.Flatten()`.

Dans ta copie : pourquoi la première couche porte-t-elle presque tous les paramètres ? Que propose le ch. 21 pour en avoir beaucoup moins sur des images ?

### Ex 23.4 — Une cross-entropy à la main : indice de classe ou one-hot ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer la cross-entropy que rend `nn.CrossEntropyLoss` à partir de logits, et mesurer l'effet d'une softmax en trop.
**Prérequis :** ch. 6 (cross-entropy) · ch. 17 (softmax) · fiche §23.5.7, §23.6.3 · **Parcours :** R, M

Un classifieur à 3 classes rend les logits $\mathbf{z} = (1 ;\ 2 ;\ 0)$ pour une image dont le label est la classe 1.

a) La probabilité de la classe 1 après une softmax, à 3 décimales.
b) $\ln \sum_j e^{z_j}$, à 3 décimales.
c) La loss que rend `nn.CrossEntropyLoss()(z, 1)`, à 3 décimales.
d) Vrai ou faux : avec la cible one-hot `[0, 1, 0]` (des probabilités cibles), la loss est la même.
e) Un modèle se termine par une `nn.Softmax` : `nn.CrossEntropyLoss` reçoit donc les **probabilités** de a) comme si c'étaient des logits. La loss qu'elle rend, à 3 décimales.
f) Un mini-batch de deux images : la première comme ci-dessus ; la seconde a les logits $(0 ;\ 0 ;\ 0)$ et le label 2. La loss du mini-batch (la moyenne, comme le fait `nn.CrossEntropyLoss` par défaut), à 3 décimales.
g) La loss d'un modèle qui donne la même probabilité à chacune des 10 classes de Fashion-MNIST, à 3 décimales.

Dans ta copie : pourquoi la softmax en trop freine-t-elle l'entraînement sans rien faire planter ? À quoi sert la valeur de g) quand tu lances un entraînement ?

### Ex 23.5 — Batches par epoch et nombre de mises à jour ✏️ ★ ⏱️ 10 min
**Objectif :** compter les mini-batches et les mises à jour d'un entraînement, avec et sans `drop_last`.
**Prérequis :** 0A (division entière) · fiche §23.7, §23.8.1 · **Fil rouge :** Fashion-MNIST

On garde 10 % des 60 000 images d'entraînement de Fashion-MNIST pour la validation, et l'on entraîne sur les autres avec des mini-batches de 256.

a) Le nombre de mini-batches par epoch (`drop_last=False`, la valeur par défaut).
b) Avec `drop_last=True`.
c) La taille du dernier mini-batch avec `drop_last=False`.
d) Le nombre de mises à jour des poids (`optimizer.step()`) en 20 epochs, avec `drop_last=False`.
e) Le nombre de mini-batches par epoch du livre : les 60 000 images, des mini-batches de 256.
f) Le mode rapide du notebook entraîne sur 12 000 images, par mini-batches de 64 : le nombre de mini-batches par epoch.
g) Avec les réglages de b), le nombre d'images vues par epoch.

Dans ta copie : `drop_last=True` fait-il perdre des images pour de bon ? Pourquoi pas, si le `DataLoader` mélange à chaque epoch ?

### Ex 23.6 — Early stopping à la main : patience et min_delta ✏️ ★★ ⏱️ 15 min
**Objectif :** appliquer la règle de l'early stopping à une suite de loss de validation, et voir l'effet de `patience` et de `min_delta`.
**Prérequis :** ch. 20 (`EarlyStopping` de `mylearn`) · fiche §23.8.3, §23.10.3

Règle (celle de la fiche et de ton `EarlyStopping` du ch. 20) : une epoch est une **amélioration** si sa loss est strictement inférieure à la meilleure loss, moins `min_delta` (la première epoch en est toujours une) ; seule une amélioration met à jour la meilleure loss et remet le compteur à 0 ; sinon le compteur augmente de 1 ; on s'arrête à la fin de l'epoch où le compteur atteint `patience`, et l'on recharge les poids de la **dernière amélioration**.

Les loss de validation des epochs 1 à 16 :

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 0,60 | 0,45 | 0,40 | 0,41 | 0,38 | 0,385 | 0,379 | 0,39 | 0,40 | 0,41 | 0,37 | 0,42 | 0,43 | 0,44 | 0,45 | 0,46 |

a) Avec `patience=3` et `min_delta=0` : l'epoch à la fin de laquelle l'entraînement s'arrête.
b) L'epoch dont on recharge les poids.
c) Avec `patience=3` et `min_delta=0.005` : l'epoch d'arrêt.
d) L'epoch dont on recharge les poids.
e) Avec `patience=5` et `min_delta=0` : l'epoch d'arrêt.
f) L'epoch dont on recharge les poids.
g) Dans le cas a), avec une règle d'arrêt qui ne recharge rien (comme l'`EarlyStopping` de Keras avec `restore_best_weights=False`, sa valeur par défaut) : l'epoch dont le modèle garde les poids.

Dans ta copie : pourquoi la patience de 3 rate-t-elle l'amélioration de l'epoch 11 ? Que coûte une patience plus grande ?

### Ex 23.7 — Plannings de learning rate : le planning du livre avec LambdaLR ✏️ ★★ ⏱️ 20 min
**Objectif :** traduire un planning de learning rate de Keras en PyTorch, et calculer le learning rate des plannings courants.
**Prérequis :** ch. 19 (plannings) · fiche §23.10.2 · **Parcours :** M

Dans cet exercice, les epochs sont comptées **à partir de 0** : pendant l'epoch $e$, le planning a reçu $e$ appels à `scheduler.step()`. `LambdaLR(optimizer, lr_lambda)` donne le learning rate $\eta_e = \eta_0 \cdot \lambda(e)$, où $\eta_0$ est le learning rate donné à la création de l'optimiseur. Le livre utilise $\lambda(e) = \max(0{,}1 ;\ 1 - 0{,}01\,e)$ avec $\eta_0 = 1$.

a) Le learning rate pendant l'epoch 25.
b) La première epoch où il vaut 0,1.
c) Un collègue crée l'optimiseur avec $\eta_0 = 0{,}01$ et garde la même fonction $\lambda$ : le learning rate pendant l'epoch 50.
d) Pour retrouver exactement le planning du livre avec un optimiseur créé avec $\eta_0 = 0{,}05$, il faut une autre fonction $\lambda$ : sa valeur $\lambda(0)$.
e) `StepLR(optimizer, step_size=30, gamma=0.1)` avec $\eta_0 = 0{,}1$ : le learning rate pendant l'epoch 65.
f) `CosineAnnealingLR(optimizer, T_max=100)` avec $\eta_0 = 0{,}1$ (et `eta_min=0`) : le learning rate pendant l'epoch 25, à 4 décimales. La formule : $\eta_e = \eta_0 \cdot \frac{1 + \cos(\pi e / T_{\max})}{2}$.
g) Avec le planning du livre sur 100 epochs (0 à 99) : le nombre d'epochs entraînées au learning rate 0,1.

Dans ta copie : que se passerait-il si l'on appelait `scheduler.step()` après **chaque mini-batch** avec le planning du livre et quelques centaines de mini-batches par epoch ?

### Ex 23.8 — Fermi : durée d'une epoch et poids d'un checkpoint 🧮 ★★ ⏱️ 20 min
**Objectif :** estimer le coût de calcul d'une epoch et la taille d'un checkpoint à partir du nombre de paramètres.
**Prérequis :** Ex 23.3 · fiche §23.3.5, §23.9.1, encadré « combien de calculs par epoch ? » · **Fil rouge :** Fashion-MNIST · **Parcours :** M

Le MLP du livre a $P = 623\,290$ paramètres et s'entraîne sur $N = 60\,000$ images. On compte $6 P N$ opérations par epoch (passe avant et passe arrière).

a) Le nombre d'opérations d'une epoch, en milliards, à 1 décimale.
b) Sa durée, en secondes, sur un CPU qui en fait 50 milliards par seconde, à 1 décimale.
c) La taille des poids en `float32`, en mégaoctets ($10^6$ octets), à 1 décimale.
d) La taille d'un checkpoint qui contient aussi l'état d'Adam (deux moments par paramètre, en `float32`), en mégaoctets, à 1 décimale.
e) Au même rythme, le nombre d'epochs **complètes** que l'on peut faire en une heure.

Dans ta copie : le livre annonce 2 à 3 secondes par epoch sur un iMac de 2014, sans GPU. Ta durée de b) est-elle plausible ? Cite deux raisons pour lesquelles une vraie epoch peut prendre plus longtemps que ce calcul (pense à ce que fait le programme en dehors des multiplications).

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 23.9 — Pourquoi le prétraitement doit voyager avec le modèle 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant pourquoi un modèle ne se livre jamais sans son prétraitement.
**Prérequis :** fiche §23.5.8, §23.9.5 · **Parcours :** R

Dans ta copie, en 5 lignes au plus et sans formule, explique à un collègue qui débute pourquoi un fichier de poids ne suffit pas pour utiliser un modèle, et ce qu'il faut lui joindre. Prends une image du quotidien (autre qu'une recette de cuisine). Puis relis-toi : as-tu dit ce qui se passe si l'on oublie le prétraitement (le modèle plante-t-il ?) ? Que les nombres du prétraitement viennent du train ? Comment on évite l'oubli ?

### Ex 23.10 — Réutiliser un modèle pré-entraîné : licence, données d'origine, biais hérités ⚖️ ★★ ⏱️ 20 min
**Objectif :** évaluer ce qu'on hérite en réutilisant un modèle pré-entraîné : droits, données, biais.
**Prérequis :** fiche §23.9.4, §23.9.5 · ch. 12 (données) · MP3 (fiche modèle)

Une jeune entreprise veut proposer une application payante qui reconnaît les vêtements sur les photos de ses utilisateurs. Pour aller vite, elle part d'un ResNet pré-entraîné sur ImageNet téléchargé avec torchvision, gèle ses couches et entraîne une nouvelle tête sur ses propres photos de vêtements. Réponds dans ta copie.

1. Le code de torchvision est sous licence BSD. Les **poids** le sont-ils aussi ? Que dit la documentation de torchvision des licences des poids, et que disent les conditions d'utilisation d'ImageNet ? Qu'en conclus-tu pour un usage commercial ? Vérifie tes sources.
2. ImageNet a été constitué en 2009 à partir d'images collectées sur le web et annotées par des travailleurs payés à la tâche. En 2019, une partie de ses catégories de personnes a été retirée après un audit. Que lui reprochait-on ? Pourquoi un modèle de **vêtements** est-il concerné par ce que contenait le dataset d'origine ?
3. Les photos d'ImageNet viennent surtout de certains pays et de certains milieux. Donne deux façons dont ce biais pourrait se voir dans l'application (types de vêtements, couleurs de peau, éclairages, poses).
4. Que doit dire la fiche modèle de l'application sur le modèle de départ (provenance, licence, données d'entraînement, limites connues) ?
5. Quelles alternatives aurait l'entreprise si la licence ou les données posent problème ?

### Ex 23.11 — Lire Krizhevsky et coll. (2012) : la recette d'entraînement d'AlexNet 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a lancé l'essor du deep learning en vision, et traduire sa recette d'entraînement en PyTorch.
**Prérequis :** ch. 21 (CNN) · fiche §23.3.5, §23.9.4, §23.10.2 · **Parcours :** M

Lis A. Krizhevsky, I. Sutskever et G. E. Hinton, « ImageNet Classification with Deep Convolutional Neural Networks », *Advances in Neural Information Processing Systems 25* (NIPS 2012), p. 1097-1105 (le PDF est en accès libre sur le site des actes ; prends la version PDF, pas la page de résumé, dont les chiffres diffèrent) : le résumé, les sections 1, 3.1, 3.2, 4 (réduire l'overfitting) et 5 (détails de l'apprentissage), et le début de la section 6. Réponds dans ta copie.

1. Sur quel matériel le réseau a-t-il été entraîné, et combien de temps ? Pourquoi les auteurs ont-ils dû répartir le réseau sur deux cartes ?
2. Section 3.1 : que montre la figure 1 sur la ReLU face à la tanh ? Relie-la au ch. 17.
3. Section 5 : écris la ligne PyTorch qui crée leur optimiseur (taille des mini-batches, momentum, weight decay, learning rate de départ).
4. Section 5 : comment faisaient-ils évoluer le learning rate, combien de fois, et sur combien de passages sur les données ? Quel planning de `torch.optim.lr_scheduler` fait la même chose automatiquement ?
5. Section 4 : quelles deux méthodes contre l'overfitting emploient-ils ? Lesquelles retrouves-tu aux ch. 20 et 24 ?
6. Section 6 : quelle erreur top-5 sur le test d'ILSVRC-2012, contre combien pour le deuxième ? Comment ce résultat a-t-il été obtenu (un seul réseau ?) ?
7. Bilan : dans la recette d'AlexNet, qu'est-ce qui est resté standard, et qu'est-ce qu'on fait autrement aujourd'hui (normalisation, optimiseur, planning, initialisation) ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 23.E1 — Ton entraînement PyTorch est deux fois plus lent que prévu : où cherches-tu, et dans quel ordre ? 💼 ★★ ⏱️ 10 min
*Fiche §23.3.5, §23.7, §23.8 · parcours R*

« Votre entraînement PyTorch prend deux fois plus de temps que prévu. Comment cherchez-vous la cause, et dans quel ordre ? »

### 23.E2 — Pourquoi appeler model.eval() et torch.no_grad() (ou inference_mode) à l'inférence ? 💼 ★★ ⏱️ 10 min
*Fiche §23.7, §23.8.2 · parcours R*

« Que font `model.eval()` et `torch.no_grad()` ? Pourquoi faut-il les deux pour prédire, et que se passe-t-il si l'on oublie l'un ou l'autre ? »

### 23.E3 — Comment sauvegarder un modèle PyTorch pour reprendre l'entraînement plus tard ou le déployer ? 💼 ★★ ⏱️ 10 min
*Fiche §23.9, §23.10.1 · parcours R*

« Comment sauvegardez-vous un modèle PyTorch ? Est-ce la même chose pour reprendre un entraînement demain et pour le mettre en production ? »

### 23.E4 — La loss de validation remonte alors que l'accuracy de validation progresse encore : que se passe-t-il ? 💼 ★★ ⏱️ 10 min
*Fiche §23.8.3, §23.10.3 · parcours R*

« Sur vos courbes, la loss de validation remonte depuis quelques epochs, mais l'accuracy de validation continue de monter. Que se passe-t-il, et que faites-vous ? »

### 23.E5 — Keras ou PyTorch : comment choisir un framework aujourd'hui ? 💼 ★★ ⏱️ 10 min
*Fiche §23.2, §23.3.3 (encadré 🕰️) · parcours R*

« Keras ou PyTorch : lequel choisiriez-vous pour un nouveau projet, et sur quels critères ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 23.12 | Environnement et tenseurs : versions, device, reshape, view, permute | 📦 | ★★ | 20 |
| 23.13 | Charger et regarder Fashion-MNIST | 📦 | ★★ | 20 |
| 23.14 | Oublier la conversion en float ou la normalisation : erreur ou entraînement lent ? | 🔮 | ★ | 10 |
| 23.15 | Tout le prétraitement au même endroit : split stratifié, float32, normalisation, DataLoader | 📦 | ★★ | 30 |
| 23.16 | Le MLP du livre avec nn.Sequential et nn.Flatten | 📦 | ★ | 15 |
| 23.17 | Trois bugs d'inférence : eval() oublié, gradients calculés pour rien (inference_mode), chargement non sûr (weights_only) | 🐛 | ★★ | 25 |
| 23.18 | Ta fonction fit() : boucle, validation et historique | 📦 | ★★★ | 45 |
| 23.19 | Lire l'historique : 3 epochs, puis 50 | 📈 | ★★ | 20 |
| 23.20 | Analyser les erreurs : matrice de confusion et grille des confusions | 📦 | ★★ | 25 |
| 23.21 | Prédire la réaction du MLP à des images « du monde réel » | 🔮 | ★ | 10 |
| 23.22 | Prédire sur de nouvelles images : eval, inference_mode, probabilités | 📦 | ★★ | 25 |
| 23.23 | Reprendre un entraînement interrompu : checkpoint complet (modèle, optimiseur, epoch, générateurs aléatoires) | 📦 | ★★ | 30 |
| 23.24 | Un modèle pré-entraîné torchvision : transforms, gel et nouvelle tête | 📦 | ★★★ | 40 |
| 23.25 | Les callbacks en PyTorch : meilleur checkpoint, planning de LR, early stopping | 📦 | ★★★ | 45 |
| 23.26 | Planning de learning rate : constant, par paliers, cosinus ou plateau | 🔬 | ★★ | 30 |
| 23.27 | Patience de l'early stopping : 1, 3 ou 10 | 🔬 | ★★ | 20 |
| 23.28 | Rédiger la fiche du modèle (model card) | 🛠️ | ★★ | 20 |
| 23.29 | Défi reproductibilité : un MLP Fashion-MNIST à au moins 88 % dont l'accuracy varie de moins de 0,5 point sur 5 graines, en moins de 3 min de CPU | 🏆 | ★★★★ | 90 |

Tout le notebook tourne sur un CPU en mode rapide ; aucun exercice ne demande de GPU.
