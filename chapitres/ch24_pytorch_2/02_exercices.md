# 24 · PyTorch en pratique 2 : améliorer, chercher, CNN et RNN — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch24_pytorch_2/06_mes_reponses.md` (créée par `python tools/start_chapter.py 24`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 24.1 à 24.5 et de l'estimation 🧮 24.7 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la dérivation ∂ 24.6, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"F"`), plusieurs lettres entre guillemets pour un choix multiple (`"AF"`), une forme entre guillemets (`"(3, 5, 1)"`).
> Conventions du chapitre : celles de PyTorch (fiche) ; les images en $(N, C, H, W)$, les séquences en $(N, T, D)$ avec `batch_first=True`. Les epochs se comptent à partir de 1.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 24.Q1 — Compter les hyperparamètres et n'en changer qu'un 🧠 ⏱️ 3 min
*Fiche §24.2.1 à §24.2.3 · livre §24.1 à §24.2.2 · parcours R*

a) Parmi ces grandeurs, lesquelles sont des hyperparamètres ? (A) les poids de la première couche ; (B) la taille de batch ; (C) le learning rate d'Adam ; (D) les biais de la couche de sortie ; (E) le taux de dropout ; (F) les moyennes glissantes qu'Adam range dans son état. Écris leurs lettres.
b) Un réseau a 6 hyperparamètres, et l'on essaie 3 valeurs pour chacun : le nombre de combinaisons possibles.
c) Vrai ou faux : laisser un argument à sa valeur par défaut, c'est ne pas choisir cet hyperparamètre.
d) Vrai ou faux : quand deux hyperparamètres interagissent, les régler une fois chacun, l'un après l'autre, donne toujours la meilleure combinaison.
e) Ce que conseille le livre pour améliorer un modèle : (A) viser tous les objectifs à la fois ; (B) trouver le réglage optimal exact avant de passer à la suite ; (C) un ou deux objectifs à la fois, puis passer aux suivants ; (D) changer plusieurs hyperparamètres à chaque essai, pour gagner du temps.

### 24.Q2 — Taille de batch : bruit, overfitting et temps 🧠 ★★ ⏱️ 3 min
*Fiche §24.2.2 · livre §24.2.2 · parcours R*

On entraîne sur $N = 50\,000$ images, sans `drop_last`.

a) Le nombre de mises à jour des poids par epoch avec des mini-batches de 64.
b) Le même nombre avec des mini-batches de 256.
c) En passant d'un batch de 400 à un batch de 16, l'écart-type du bruit du gradient est multiplié par combien ?
d) Vrai ou faux : à learning rate égal, passer d'un batch de 512 à un batch de 8 ne change que la durée de l'entraînement, pas son résultat.
e) Sur le CPU du livre, quand la taille de batch diminue, la durée d'une epoch : (A) augmente ; (B) diminue ; (C) ne change pas.

### 24.Q3 — Plus de neurones, moins de neurones, dropout 🧠 ⏱️ 3 min
*Fiche §24.2.4 à §24.2.6 · livre §24.2.4 à §24.2.6 · parcours R*

a) Un MLP sur-apprend déjà ; on lui ajoute une deuxième couche cachée aussi large que la première. Ce que le livre attend, et observe : (A) moins d'overfitting ; (B) aucune différence ; (C) une accuracy d'entraînement plus basse ; (D) un overfitting plus rapide.
b) La fraction des valeurs que `nn.Dropout(p=0.3)` met à zéro, en moyenne, pendant l'entraînement.
c) Quand `nn.Dropout` tire-t-il son masque ? (A) une fois par epoch ; (B) une fois par mini-batch, le même pour tous ses exemples ; (C) à chaque passe avant, indépendamment pour chaque exemple et chaque valeur ; (D) une fois pour toutes, à la création du module.
d) Avec `p = 0.2`, en entraînement, une activation de valeur 2,0 qui est gardée devient : sa valeur.
e) Vrai ou faux : en mode `model.eval()`, `nn.Dropout` multiplie ses entrées par $1 - p$.

### 24.Q4 — Valider un réseau : coût, jeu non représentatif, normalisation dans chaque fold 🧠 ★★ ⏱️ 3 min
*Fiche §24.3 à §24.3.3 · livre §24.3 à §24.3.3 · parcours R*

a) Un entraînement dure 40 minutes. La durée d'une validation croisée à 5 folds, en minutes.
b) Un fichier de 10 000 radios est trié par hôpital : les 6 000 premières viennent de l'hôpital A, les 4 000 suivantes de l'hôpital B. On garde les 2 000 dernières pour valider. Le problème : (A) la validation ne contient que l'hôpital B, absent d'une partie de l'entraînement : elle ne représente pas les données ; (B) la validation est trop petite ; (C) les labels fuient vers la validation ; (D) aucun.
c) Pourquoi place-t-on le `MinMaxScaler` dans le `Pipeline` passé à `cross_val_score` ? (A) pour aller plus vite ; (B) parce que `cross_val_score` refuse des données non normalisées ; (C) pour mettre aussi les labels à l'échelle ; (D) pour qu'il soit réappris sur la partie d'entraînement de chaque fold, sans voir la validation de ce fold.
d) Vrai ou faux : `cross_val_score(estimateur, X, y, cv=5)` fait des folds stratifiés quand l'estimateur est un classifieur.
e) Avec et sans normalisation dans le pipeline, le livre trouve : (A) un score bien meilleur avec ; (B) un score bien meilleur sans ; (C) presque le même score, les pixels étant déjà divisés par 255.

### 24.Q5 — 🕰️ KerasClassifier, grille, hasard ou zoom : chercher des hyperparamètres 🧠 ⏱️ 3 min
*Fiche §24.3.1, §24.3.4, §24.4.8 · livre §24.3.1, §24.3.4, §24.4.8*

a) Aujourd'hui, pour régler un `nn.Module` avec `GridSearchCV`, on utilise : (A) `keras.wrappers.scikit_learn.KerasClassifier` ; (B) `sklearn.neural_network.MLPClassifier` ; (C) skorch, ou son propre estimateur ; (D) rien : c'est impossible.
b) Le nombre de folds de `GridSearchCV` quand on ne précise pas `cv`, dans scikit-learn aujourd'hui.

La grille `{"net__lr": [1e-3, 1e-2], "net__hidden": [(32,), (64,), (128,)], "net__p_drop": [0.0, 0.2]}` est passée à `GridSearchCV(..., cv=4)`.

c) Le nombre d'entraînements de la validation croisée, sans le réentraînement final.
d) Le nombre total d'entraînements, réentraînement final compris (`refit=True`, le défaut).
e) La recherche multirésolution, c'est : (A) entraîner sur des images de plusieurs résolutions ; (B) tirer des réglages au hasard ; (C) répartir la recherche sur plusieurs GPU ; (D) une grille large et grossière, puis des grilles plus fines autour des meilleures régions.

### 24.Q6 — 🕰️ Couches utilitaires de Keras et leurs équivalents PyTorch 🧠 ⏱️ 3 min
*Fiche §24.4, §24.4.1 · livre §24.4, §24.4.1*

Les équivalents PyTorch : (A) une tranche `x[..., 1:-1, 1:-1]` ; (B) `nn.ZeroPad2d(1)` ; (C) `nn.Upsample(scale_factor=2)` ; (D) `nn.AvgPool2d(2)` ; (E) `nn.Flatten()`. Donne la lettre de l'équivalent de chaque couche Keras.

a) `UpSampling2D(size=2)`.
b) `Cropping2D(cropping=1)`.
c) `ZeroPadding2D(padding=1)`.
d) `AveragePooling2D(pool_size=2)`.
e) Vrai ou faux : la couche `GaussianNoise` de Keras ajoute aussi son bruit pendant l'évaluation.
f) Ce que normalise une batchnorm : (A) les poids de la couche ; (B) les gradients ; (C) les activations de chaque canal, sur le mini-batch, avant d'appliquer $\gamma$ et $\beta$ appris ; (D) seulement les images d'entrée.

### 24.Q7 — Préparer des images pour nn.Conv2d 🧠 ⏱️ 3 min
*Fiche §24.4.2, §24.4.3 · livre §24.4.2, §24.4.3 · parcours R*

a) La forme d'un mini-batch de 16 images CIFAR-10 (32 × 32, en couleur) attendue par `nn.Conv2d`.
b) Un tableau d'images a la forme `(16, 32, 32, 3)`. Pour le donner à `nn.Conv2d` : (A) `x.reshape(16, 3, 32, 32)` ; (B) `x.transpose(1, 3)` ; (C) `x.permute(0, 2, 3, 1)` ; (D) `x.permute(0, 3, 1, 2)`.
c) La forme de la sortie de `nn.Conv2d(3, 12, 5, padding="same")` pour ce mini-batch.
d) Vrai ou faux : `nn.Conv2d(3, 12, 5, stride=2, padding="same")` se construit sans erreur.
e) Un pixel de valeur 51 (sur 255) mis dans $[-1, 1]$ par $x/127{,}5 - 1$.
f) Vrai ou faux : `nn.Conv2d` applique une ReLU à sa sortie, sauf si on le lui interdit.

### 24.Q8 — Augmentation et données synthétiques : quelles transformations sont plausibles ? 🧠 ⏱️ 3 min
*Fiche §24.4.6, §24.4.7 · livre §24.4.6, §24.4.7 · parcours R*

a) L'augmentation de données s'applique : (A) au train et à la validation ; (B) au train seulement ; (C) à toutes les données ; (D) au test seulement.
b) Vrai ou faux : quand l'augmentation est faite par le `Dataset`, une image donnée est transformée de la même façon à chaque epoch.
c) Une transformation d'augmentation est utile quand : (A) elle est la plus forte possible ; (B) elle change le label de l'image ; (C) elle produit une image que le modèle pourrait rencontrer, de même label ; (D) on l'applique aussi au test.
d) Pourquoi `fit_generator` de Keras demandait-il la taille d'une epoch (un nombre de batches, ou d'exemples dans le livre) ? (A) pour réserver la mémoire ; (B) pour régler le learning rate ; (C) pour mélanger les données ; (D) un générateur ne s'arrête jamais : il faut lui dire ce qui fait une epoch.
e) Un `IterableDataset` qui dessine des formes à la volée : (A) sa longueur est le nombre de formes différentes ; (B) il n'a pas de longueur : c'est le code qui décide combien d'exemples font une epoch ; (C) sa longueur est la taille du batch.
f) Vrai ou faux : avec des exemples neufs à chaque tirage, un modèle ne peut pas apprendre ses exemples d'entraînement par cœur.

### 24.Q9 — Motifs conv-conv-pool, stride contre pooling, batchnorm contre dropout 🧠 ★★ ⏱️ 3 min
*Fiche §24.4.4, §24.4.5 · livre §24.4.4, §24.4.5*

a) L'ordre que suit le livre pour la batchnorm : (A) convolution, ReLU, batchnorm ; (B) batchnorm, convolution, ReLU ; (C) convolution, batchnorm, ReLU.
b) Vrai ou faux : juste avant une `nn.BatchNorm2d`, le biais de la convolution ne sert à rien.
c) Une convolution $3 \times 3$ a autant de canaux en entrée qu'en sortie. Dans le bloc suivant, la hauteur et la largeur sont divisées par 2, et les canaux d'entrée et de sortie multipliés par 3. Son nombre de multiplications-additions est multiplié par combien ? (2 décimales)
d) Le livre remplace le pooling par un stride de 2 dans les convolutions (fig. 24.33). Il obtient : (A) presque la même accuracy, en environ un quart du temps ; (B) une accuracy bien plus basse, dans le même temps ; (C) une meilleure accuracy, en deux fois plus de temps.
e) Les CNN modernes finissent le plus souvent par : (A) trois grosses couches denses ; (B) un `Flatten` de toute la dernière carte ; (C) un pooling moyen global, puis une couche linéaire.

### 24.Q10 — Séries pour un RNN : fenêtres, scaler, inverse_transform, RMSE 🧠 ⏱️ 3 min
*Fiche §24.5 à §24.5.4 · livre §24.5 à §24.5.4 · parcours R*

a) Le `MinMaxScaler` d'une série s'ajuste sur : (A) toute la série ; (B) chaque fenêtre ; (C) le test ; (D) le train seul.
b) La forme du tenseur `X` attendue par `nn.LSTM(batch_first=True)` pour 500 fenêtres de 12 pas d'une série à une seule variable.
c) La RMSE d'une prévision se calcule : (A) après `inverse_transform`, dans les unités de la série ; (B) sur les valeurs mises à l'échelle, entre 0 et 1.
d) Vrai ou faux : sur le test, une valeur hors de l'intervalle du train sort de $[0, 1]$ après le scaler ; il faut alors réajuster le scaler sur le test.
e) La dernière couche du prévisionniste du livre : (A) une couche dense suivie d'une sigmoïde ; (B) une softmax ; (C) une couche dense sans activation.
f) Pour savoir si un LSTM apprend vraiment quelque chose sur une série, on le compare d'abord à : (A) un réseau plus gros ; (B) la persistance (prédire la dernière valeur connue) ; (C) sa RMSE sur le train.

### 24.Q11 — 🕰️ return_sequences, stateful, TimeDistributed : que devient chaque option en PyTorch ? 🧠 ★★ ⏱️ 3 min
*Fiche §24.5.8 à §24.5.10 · livre §24.5.8 à §24.5.10 · parcours R*

`output, (h_n, c_n) = nn.LSTM(D, H, num_layers=2, batch_first=True)(x)`, à un seul sens.

a) L'équivalent de `LSTM(H, return_sequences=True)` de Keras : (A) `h_n` ; (B) `c_n` ; (C) `output[:, -1]` ; (D) `output`.
b) L'équivalent de la sortie par défaut de Keras (`return_sequences=False`) de la deuxième couche : (A) `h_n[-1]` ; (B) `h_n[0]` ; (C) `c_n[-1]` ; (D) `output[0]`.
c) L'équivalent de `TimeDistributed(Dense(5))` appliqué à `output` : (A) `nn.Flatten()` puis `nn.Linear(T * H, 5)` ; (B) `nn.Linear(H, 5)` appliqué directement à `output` ; (C) une boucle Python sur les exemples.
d) Avec l'état porté (*stateful*), l'état final de la ligne $i$ du batch $k$ sert d'état initial à : (A) la ligne $i + 1$ du batch $k$ ; (B) toutes les lignes du batch $k + 1$ ; (C) la ligne $i$ du batch $k + 1$.
e) Vrai ou faux : si l'on porte l'état d'un batch au suivant sans `detach()`, le `backward()` du deuxième batch lève une erreur.
f) Vrai ou faux : un LSTM sans état porté transmet son état d'un exemple au suivant à l'intérieur d'un mini-batch.

### 24.Q12 — 🕰️ API fonctionnelle de Keras 3 et forward() de PyTorch : branches, partage, gel 🧠 ⏱️ 3 min
*Fiche §24.6 à §24.6.2 · livre §24.6 à §24.6.2*

a) Une couche `self.fc` appelée deux fois dans `forward()` : (A) est copiée à chaque appel ; (B) lève une erreur ; (C) partage ses poids entre les deux appels.
b) L'équivalent PyTorch de la couche `Add` de Keras : (A) `torch.cat` ; (B) `+` ; (C) `torch.stack`.
c) La forme de `torch.cat([a, b], dim=1)` pour deux branches `a` et `b` de forme `(4, 16, 8, 8)`.
d) Vrai ou faux : `requires_grad_(False)` suffit à figer complètement une `nn.BatchNorm2d` dans un modèle entraîné en mode `train()`.
e) Le second modèle du livre branche une `Conv2D` sur la sortie d'une couche `Dense(1000)`. Ce modèle : (A) ne se construit pas : une convolution 2D attend une image ; (B) tourne mais apprend mal ; (C) tourne normalement.
f) Vrai ou faux : Keras 3 a gardé l'API fonctionnelle.

<a id="rappels"></a>

## 🔁 Rappels

### 24.R1 — Ch. 23 : ce qu'un historique d'entraînement doit contenir pour comparer deux runs 🔁 ★ ⏱️ 5 min
*Ch. 23 (§23.7, §23.8.3) · fiche §24.2.2, §24.2.7 · parcours R*

a) Ce que garde l'historique d'une fonction `fit()` honnête : (A) la loss d'entraînement de chaque epoch ; (B) la loss et l'accuracy de validation de chaque epoch ; (C) une copie des poids de chaque epoch ; (D) le learning rate de chaque epoch (utile dès qu'un planning le fait varier) ; (E) la loss de test de chaque epoch. Écris leurs lettres.
b) Ce qu'il faut noter en plus pour comparer deux runs : (A) la couleur des courbes ; (B) la graine ; (C) tous les hyperparamètres, valeurs par défaut comprises ; (D) la durée et la machine. Écris leurs lettres.
c) Vrai ou faux : la loss d'entraînement qu'affiche une boucle comme celle du ch. 23 est calculée avec les poids de fin d'epoch.
d) Les losses de validation d'un run, epoch par epoch à partir de l'epoch 1, valent $0{,}52 ;\ 0{,}47 ;\ 0{,}45 ;\ 0{,}46 ;\ 0{,}44 ;\ 0{,}45$. Le numéro de la meilleure epoch.

### 24.R2 — Ch. 21 : taille de sortie d'une convolution à padding 'same' et stride 2 🔁 ★ ⏱️ 5 min
*Ch. 21 (§21.3.4, §21.3.5) · fiche §24.4.3 · parcours R, M*

Rappel : avec `padding="same"`, Keras rend une sortie de $\lceil H/s \rceil$ lignes ; PyTorch suit $H_{out} = \lfloor (H + 2p - k)/s \rfloor + 1$.

a) `Conv2D(8, 5, strides=2, padding="same")` de Keras sur une image $28 \times 28$ : la hauteur de la sortie.
b) La même couche sur une image $15 \times 15$ : la hauteur de la sortie.
c) Le padding $p$ de `nn.Conv2d(·, ·, 5, stride=2, padding=p)` qui reproduit ce `"same"` de Keras.
d) Avec ce padding, la hauteur de la sortie de PyTorch pour une image $32 \times 32$.
e) La hauteur de la sortie de `nn.Conv2d(·, ·, 3, stride=2, padding=0)` pour une image $15 \times 15$.
f) Vrai ou faux : en PyTorch, `padding="same"` avec un stride de 1 garde la taille de l'image, même pour un noyau de taille paire.

### 24.R3 — Ch. 17 : ce que change la division des logits avant le softmax 🔁 ★ ⏱️ 5 min
*Ch. 17 (§17.7) et ch. 22 (température) · fiche §24.5.11 · parcours R*

Un modèle rend les logits $(2, 1, 0)$ pour trois tokens. On tire le suivant dans $\text{softmax}(z/T)$.

a) La plus grande probabilité pour $T = 1$, à 3 décimales.
b) La même pour $T = 0{,}5$, à 3 décimales.
c) La même pour $T = 2$, à 2 décimales.
d) Quand $T$ tend vers 0, la distribution : (A) devient uniforme ; (B) met toute la probabilité sur le plus grand logit ; (C) ne change pas.
e) Vrai ou faux : pour $T > 0$, diviser les logits par $T$ ne change pas le token le plus probable.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 24.1 — Le coût d'une grille (27 hyperparamètres, 54 entraînements) contre une recherche aléatoire : probabilité de tomber dans les 5 % meilleurs réglages en n essais ✏️ ★ ⏱️ 10 min
**Objectif :** chiffrer le coût d'une recherche en grille, et la chance qu'a une recherche au hasard de trouver un bon réglage.
**Prérequis :** ch. 15 (`GridSearchCV`) · fiche §24.2.1, §24.3.4 · **Parcours :** R, M

Le livre compte 27 hyperparamètres pour son petit MLP, puis lance une grille de 3 nombres de couches, 3 nombres de neurones et 2 optimiseurs, avec 3 folds.

a) Si chacun des 27 hyperparamètres ne prenait que 2 valeurs, le nombre de combinaisons.
b) À une seconde par entraînement, le temps qu'il faudrait pour toutes les essayer, en années de 365 jours, à 1 décimale.
c) Le nombre d'entraînements de la grille du livre (validation croisée seulement).
d) Si chaque entraînement dure 70 secondes, la durée de cette grille, en minutes.

On tire maintenant des réglages au hasard, indépendamment, dans un espace où 5 % des réglages sont « assez bons ».

e) La probabilité qu'au moins un de 20 tirages soit assez bon, à 2 décimales.
f) Le plus petit nombre de tirages pour que cette probabilité atteigne 0,95.
g) La même probabilité qu'en e) pour 18 tirages, autant que la grille du livre a de candidats, à 3 décimales.

Dans ta copie : la grille du livre essaie combien de valeurs différentes du nombre de neurones ? Et 18 tirages au hasard d'un entier entre 20 et 40 ? Pourquoi est-ce important quand un seul hyperparamètre compte vraiment ?

### Ex 24.2 — Formes et paramètres du CNN à pooling du livre ✏️ ★★ ⏱️ 20 min
**Objectif :** suivre les formes d'un tenseur d'images à travers un CNN et compter ses paramètres couche par couche.
**Prérequis :** ch. 21 (tailles de sortie, paramètres d'une convolution) · fiche §24.4.3, §24.4.4 · **Fil rouge :** MNIST · **Parcours :** R, M

Le CNN à pooling du livre (fig. 24.31), traduit en PyTorch, pour des images MNIST $1 \times 28 \times 28$ :
`Conv2d(1, 30, 5, padding=2)`, `ReLU`, `Dropout(0.2)`, `MaxPool2d(2)`, `Conv2d(30, 16, 3, padding=1)`, `ReLU`, `Dropout(0.2)`, `MaxPool2d(2)`, `Flatten`, `Linear(n, 128)`, `ReLU`, `Linear(128, 64)`, `ReLU`, `Linear(64, 10)`.

a) La forme de la sortie de la première convolution pour un mini-batch de 256 images.
b) La forme après le premier pooling, pour ce même mini-batch.
c) Le nombre de paramètres de la première convolution.
d) Celui de la seconde convolution.
e) La longueur $n$ du vecteur que reçoit la première couche dense.
f) Le nombre de paramètres de `Linear(n, 128)`.
g) Le nombre total de paramètres du modèle.
h) La part de ce total qui est dans les trois couches denses, en fraction (entre 0 et 1), à 3 décimales.

Dans ta copie : le livre justifie ses trois couches denses en disant qu'il ne reste que $7 \times 7 = 49$ valeurs après les deux poolings. Où est l'erreur ? Que proposerait un CNN moderne pour réduire le nombre de paramètres de la première couche dense ?

### Ex 24.3 — Stride ou pooling : compter les multiplications ✏️ ★★ ⏱️ 20 min
**Objectif :** comparer le coût de calcul d'un CNN à pooling et de sa version à stride, et le confronter aux durées mesurées par le livre.
**Prérequis :** Ex 24.2 · fiche §24.4.3, §24.4.4 · **Fil rouge :** MNIST · **Parcours :** M

On compte les multiplications-additions (MAC) d'**une** image, avec $\text{MAC}_{\text{conv}} = H_{out} W_{out} C_{out} C_{in} k_h k_w$ et $\text{MAC}_{\text{dense}} = n_{in}\, n_{out}$ (on néglige les biais, les ReLU et le dropout). La version à pooling est le CNN de l'Ex 24.2 ; la version à stride (fig. 24.33) n'a pas de pooling, et ses deux convolutions ont un stride de 2 : `Conv2d(1, 30, 5, stride=2, padding=2)` puis `Conv2d(30, 16, 3, stride=2, padding=1)`, suivies des mêmes couches denses.

a) Les MAC de la première convolution de la version à pooling.
b) Ceux de la seconde convolution de la version à pooling.
c) Ceux des trois couches denses (les mêmes dans les deux versions).
d) Ceux des deux convolutions de la version à stride, ensemble.
e) Le rapport des MAC totaux (convolutions et couches denses) de la version à pooling à ceux de la version à stride, à 2 décimales.
f) Le nombre de comparaisons du premier max pooling $2 \times 2$ (3 comparaisons pour chaque valeur de sortie).
g) Le rapport des durées d'epoch mesurées par le livre (147 s contre 36 s), à 1 décimale.

Dans ta copie : pourquoi le rapport des durées n'est-il pas égal au rapport des MAC ? Cite deux choses que fait le programme en dehors des multiplications.

### Ex 24.4 — Ce que renvoie nn.LSTM : output, h_n et c_n ✏️ ★★ ⏱️ 15 min
**Objectif :** prévoir la forme de chaque sortie de `nn.LSTM` et compter ses paramètres.
**Prérequis :** ch. 22 (LSTM, formes `(N, T, D)`) · fiche §24.5.6, §24.5.8, §24.5.10 · **Parcours :** R, M

`lstm = nn.LSTM(input_size=3, hidden_size=4, num_layers=2, batch_first=True)` reçoit un tenseur `x` de forme `(8, 5, 3)` : 8 séquences de 5 pas, 3 features par pas. On écrit `output, (h_n, c_n) = lstm(x)`.

a) La forme de `output`.
b) La forme de `h_n`.
c) Le tenseur qui a la même forme que `h_n[0]` : (A) `output[:, 0]` ; (B) `output[0]` ; (C) `c_n` ; (D) `output[:, :, 0]`.
d) La forme de `output` si le LSTM est construit avec `bidirectional=True`.
e) La forme de `h_n` dans ce cas.
f) Le nombre de paramètres du LSTM à un seul sens (PyTorch range deux vecteurs de biais par couche).
g) La forme de `nn.Linear(4, 1)(output)` pour le LSTM à un seul sens.
h) La forme que doit avoir `x` si le LSTM est construit sans `batch_first`.

Dans ta copie : pourquoi le nombre de features (3) n'apparaît-il dans aucune sortie ? Vérifie ensuite tes formes et ton compte de paramètres en exécutant ce LSTM sur `torch.zeros(8, 5, 3)`.

### Ex 24.5 — Fenêtres, pas d'optimisation et durée : de 200 à 20 000 points ✏️ ★ ⏱️ 10 min
**Objectif :** compter les exemples et les pas d'optimisation d'un entraînement sur des fenêtres, et vérifier une durée annoncée.
**Prérequis :** ch. 22 (`make_windows`) · fiche §24.5.2, §24.5.7 · **Fil rouge :** séries synthétiques · **Parcours :** M

Une série est découpée en fenêtres de $w = 13$ points qui commencent à chaque pas ; la cible de chaque fenêtre est la valeur qui la suit.

a) Le nombre d'exemples tirés d'une série de 200 points.
b) Le nombre d'exemples tirés d'une série de 20 000 points.
c) Avec des batches d'un seul exemple (comme le livre), le nombre de pas d'optimisation par epoch sur la série de 20 000 points.
d) Le même nombre avec des batches de 32.
e) Le livre annonce un peu plus de 5 secondes par epoch sur la série de 20 000 points. La durée de 100 epochs à 5 s, en minutes, à 1 décimale.
f) Le livre annonce pourtant « un peu plus de 28 heures » pour ces 100 epochs. La durée d'une epoch qui correspondrait à 28 heures exactement, en secondes.
g) Si la durée d'une epoch est proportionnelle au nombre d'exemples, celle d'une epoch sur la série de 200 points, à 5 s pour celle de 20 000 points, en secondes, à 3 décimales.

Dans ta copie : laquelle des deux durées du livre te semble la bonne, et pourquoi ? Quelle phrase du livre permet de trancher ?

### Ex 24.6 — Dropout inversé et max-norm : espérance et projection ∂ ★★ ⏱️ 25 min
**Objectif :** démontrer que le dropout inversé garde l'espérance des activations, mesurer le bruit qu'il ajoute, et justifier la formule de la max-norm.
**Prérequis :** ch. 20 (dropout inversé) · 0B (espérance, variance) · fiche §24.2.6 · **Parcours :** M

Une activation $h$ (un nombre fixé) passe par un dropout inversé de taux $p$ ($0 < p < 1$) : $\tilde{h} = m\,h/(1 - p)$, avec $m = 1$ avec la probabilité $1 - p$ et $m = 0$ sinon. Réponds dans ta copie, avec la démarche.

1. Montre que $\mathbb{E}[\tilde{h}] = h$.
2. Calcule $\text{Var}(\tilde{h})$ en fonction de $h$ et $p$. Que vaut-elle pour $p = 0{,}5$ ? Comment varie-t-elle quand $p$ approche de 1 ?
3. La version d'origine du dropout ne divisait pas par $1 - p$ à l'entraînement, et multipliait les activations par $1 - p$ en évaluation. Montre qu'elle donne, en évaluation, une activation égale à l'espérance de son activation d'entraînement. Pourquoi PyTorch préfère-t-il la version inversée ?
4. Un neurone de la couche suivante calcule $z = w_1 \tilde{h}_1 + w_2 \tilde{h}_2$, avec des masques indépendants. Montre que $\mathbb{E}[z]$ est la valeur de $z$ en évaluation. Puis prends $h_1 = 1$, $h_2 = -1$, $w_1 = w_2 = 1$ et $p = 0{,}5$ : calcule $\mathbb{E}[\text{ReLU}(z)]$ et compare-la à $\text{ReLU}$ de la valeur d'évaluation. Qu'en conclus-tu sur ce que calcule le réseau en évaluation ?
5. La max-norm remplace $\mathbf{w}$ par $\mathbf{w}' = \mathbf{w} \cdot \min(1, c/\lVert\mathbf{w}\rVert)$. Montre que $\mathbf{w}'$ est le point de la boule $\{\mathbf{v} : \lVert\mathbf{v}\rVert \le c\}$ le plus proche de $\mathbf{w}$ (indice : pour $\lVert\mathbf{w}\rVert > c$ et un $\mathbf{v}$ de la boule, minore $\lVert\mathbf{w} - \mathbf{v}\rVert$ par l'inégalité triangulaire).
6. Prends $\mathbf{w} = (3, 4, 12)$ et $c = 3$ : calcule $\mathbf{w}'$ et la distance $\lVert\mathbf{w} - \mathbf{w}'\rVert$, et vérifie qu'elle vaut $\lVert\mathbf{w}\rVert - c$. Puis : le poids d'un `nn.Conv2d(C_in, C_out, k)` a la forme `(C_out, C_in, k, k)`. Quels poids « arrivent » sur un même canal de sortie ? Sur quelle dimension `torch.renorm` doit-il travailler pour leur appliquer la max-norm, et pourquoi le fait-on sous `torch.no_grad()` ?

### Ex 24.7 — Fermi : le budget d'une recherche d'hyperparamètres sur CPU, sur GPU et en précision mixte 🧮 ★★ ⏱️ 20 min
**Objectif :** estimer, avant de la lancer, la durée d'une recherche d'hyperparamètres, et choisir une stratégie qui tient dans le budget.
**Prérequis :** Ex 24.1 · fiche §24.3.4, §24.4.8, « précision mixte et compilation » · **Fil rouge :** CIFAR-10 · **Parcours :** R, M

Un petit CNN fait 25 millions de MAC par image en passe avant. On compte 2 opérations par MAC, et un entraînement coûte 3 fois la passe avant (passe avant et passe arrière). On l'entraîne 20 epochs sur les 50 000 images de CIFAR-10.

a) Le nombre d'opérations d'un entraînement, en milliers de milliards ($10^{12}$).
b) Sa durée, en minutes, sur un CPU qui soutient 50 milliards d'opérations par seconde.
c) Une grille croise 4 learning rates, 3 largeurs de réseau et 3 taux de dropout, avec 3 folds : le nombre d'entraînements.
d) La durée de cette grille sur le CPU, en heures.
e) Sa durée sur un GPU qui soutient $10^{13}$ opérations par seconde en `float32`, en heures, à 2 décimales.
f) En précision mixte, le même GPU va 2,5 fois plus vite : la durée de la grille, en minutes, à 1 décimale.
g) Une recherche au hasard de 30 réglages, chacun évalué sur un seul jeu de validation (sans validation croisée), sur le CPU : sa durée, en heures.

Dans ta copie : quelle stratégie choisirais-tu avec un CPU seulement et un week-end de calcul ? Que perds-tu en remplaçant la validation croisée par un seul jeu de validation ? Pourquoi les vraies durées seront-elles plus longues que ces estimations ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 24.8 — L'augmentation de données en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce qu'est l'augmentation de données et pourquoi elle aide.
**Prérequis :** fiche §24.4.6 · **Parcours :** R

Dans ta copie, en 5 lignes au plus et sans formule, explique à un collègue qui débute ce qu'est l'augmentation de données d'images et pourquoi elle améliore un modèle. Prends une image du quotidien (autre que l'apprentissage d'une langue). Puis relis-toi : as-tu dit que le label ne change pas ? Qu'on ne transforme ni la validation ni le test ? Qu'une transformation doit rester plausible, avec un exemple de transformation à éviter ?

### Ex 24.9 — Une seule graine suffit-elle ? Reproductibilité et honnêteté des comparaisons d'hyperparamètres ⚖️ ★★ ⏱️ 20 min
**Objectif :** juger si une comparaison de réglages est honnête, et dire ce qu'un rapport d'expériences doit contenir.
**Prérequis :** Ex 24.1 · ch. 2 (moyenne, écart-type) · ch. 8 (validation, test) · fiche §24.2.7, §24.3.4 · **Parcours :** R

Une équipe compare deux réglages A et B d'un CNN sur CIFAR-10. Avec une graine chacun, A obtient 87,4 % d'accuracy de validation et B 86,9 % ; le rapport interne conclut : « A est meilleur de 0,5 point ». Un collègue relance chaque réglage avec 5 graines : A obtient $87{,}4 ;\ 86{,}6 ;\ 86{,}9 ;\ 87{,}1 ;\ 86{,}5$ et B $86{,}9 ;\ 87{,}3 ;\ 87{,}0 ;\ 86{,}8 ;\ 87{,}2$ (en %). Réponds dans ta copie.

1. Calcule la moyenne et l'écart-type (avec $n - 1$) de chaque réglage. La conclusion du rapport tient-elle ?
2. Une « graine » fixe quoi, dans un entraînement de CNN ? Cite au moins quatre sources de hasard.
3. L'équipe a ensuite essayé 60 réglages, une graine chacun, et retenu le meilleur, à 88,1 %. Pourquoi ce 88,1 % surestime-t-il probablement la performance du réglage retenu ? Que faut-il faire avant de l'annoncer ?
4. Le meilleur réglage a été choisi en regardant l'accuracy de **test**. Qu'est-ce que cela change à la valeur annoncée ?
5. Strubell, Ganesh et McCallum (2019) ont estimé le coût énergétique de recherches d'architectures en traitement du langage. Que doit dire un rapport sur le calcul dépensé, et pourquoi est-ce une question d'honnêteté autant que d'écologie ?
6. Propose un protocole de comparaison honnête en cinq points, que tu pourrais suivre dans ton propre journal d'expériences.

### Ex 24.10 — Lire He et coll. (2016) : les réseaux résiduels 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a introduit les connexions résiduelles, et relier ses expériences sur CIFAR-10 à ce que tu construis dans le notebook.
**Prérequis :** ch. 21 (CNN, ResNet) · fiche §24.4.5, §24.6.2 · **Fil rouge :** CIFAR-10 · **Parcours :** M

Lis K. He, X. Zhang, S. Ren et J. Sun, « Deep Residual Learning for Image Recognition », CVPR 2016 ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) : le résumé, l'introduction et sa figure 1, la section 3 (3.1 à 3.3), et la section 4.2 sur CIFAR-10. Réponds dans ta copie.

1. Résumé : quel problème les auteurs veulent-ils résoudre, quelle profondeur atteignent-ils sur ImageNet, et quel résultat obtiennent-ils à ILSVRC 2015 ?
2. Figure 1 : que montrent les courbes d'erreur d'entraînement et de test des réseaux « ordinaires » de 20 et 56 couches sur CIFAR-10 ? Pourquoi les auteurs disent-ils que ce n'est **pas** de l'overfitting ?
3. Section 3.1 : qu'est-ce que la fonction résiduelle $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$ ? Pourquoi serait-il plus facile, pour un bloc, d'apprendre $F = 0$ que d'apprendre $H$ égal à l'identité ?
4. Sections 3.2 et 3.3, et figure 2 : que fait le raccourci quand les formes de $\mathbf{x}$ et $F(\mathbf{x})$ diffèrent ? Relie les options (A) et (B) des auteurs au `ResidualBlock` de la fiche.
5. Section 4.2 : décris le réseau de $6n + 2$ couches utilisé sur CIFAR-10 (filtres, tailles des cartes, fin du réseau). Combien de couches pour $n = 3$ ? Que se passe-t-il pour le réseau de 1 202 couches, et pourquoi selon les auteurs ?
6. Bilan : pourquoi trouve-t-on aujourd'hui des connexions résiduelles dans presque tous les réseaux profonds, Transformers compris ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 24.E1 — Comment organises-tu une recherche d'hyperparamètres avec un budget limité ? 💼 ★★ ⏱️ 10 min
*Fiche §24.2, §24.3.4, §24.4.8 · parcours R*

« Vous avez deux jours de GPU pour régler un réseau. Comment organisez-vous la recherche d'hyperparamètres ? »

### 24.E2 — Pourquoi l'augmentation de données améliore-t-elle la généralisation ? Donne des transformations à éviter 💼 ★★ ⏱️ 10 min
*Fiche §24.4.6 · parcours R*

« Pourquoi l'augmentation de données améliore-t-elle la généralisation ? Donnez des transformations à éviter, et dites pourquoi. »

### 24.E3 — Dropout ou batchnorm : lequel, où, et peut-on combiner les deux ? 💼 ★★ ⏱️ 10 min
*Fiche §24.2.6, §24.4.1, §24.4.4 · parcours R*

« Dropout ou batchnorm : lequel utilisez-vous, à quel endroit du réseau, et peut-on combiner les deux ? »

### 24.E4 — Comment éviter la fuite de données dans une validation croisée avec prétraitement ? 💼 ★★ ⏱️ 10 min
*Fiche §24.3.3 · parcours R*

« Votre validation croisée donne un score bien meilleur que le test final. Votre pipeline normalise les données et choisit des features. Où cherchez-vous une fuite de données, et comment l'évitez-vous ? »

### 24.E5 — Batchnorm : que fait-elle, et pourquoi se comporte-t-elle différemment en train et en eval ? 💼 ★★ ⏱️ 10 min
*Fiche §24.4.1, §24.4.4, §24.6 · parcours R*

« Que fait une couche de batchnorm ? Pourquoi se comporte-t-elle différemment en entraînement et en évaluation, et quels bugs cela peut-il causer ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 24.11 | Changer un seul hyperparamètre : la taille de batch | 🔬 | ★★ | 30 |
| 24.12 | Prédire le classement de quatre MLP (784, 784 + 784, 64, 32 + 32) | 🔮 | ★★ | 25 |
| 24.13 | Dropout, max-norm et learning rate trop grand | 📦 | ★★ | 30 |
| 24.14 | Pourquoi la validation dépasse le train avec dropout | 📈 | ★ | 10 |
| 24.15 | Journal d'expériences : un run, une ligne | 🛠️ | ★★ | 20 |
| 24.16 | Un réseau PyTorch déguisé en estimateur scikit-learn | 📦 | ★★★ | 50 |
| 24.17 | Validation croisée stratifiée avec normalisation dans un Pipeline | 📦 | ★★ | 25 |
| 24.18 | Grille, hasard, puis zoom : chercher les hyperparamètres | 📦 | ★★★ | 45 |
| 24.19 | Un CNN traduit du Keras qui refuse de tourner | 🐛 | ★★ | 25 |
| 24.20 | Du CNN simple au CNN profond sur CIFAR-10 | 📦 | ★★★ | 45 |
| 24.21 | Batchnorm, dropout ou les deux : accuracy et écart entre train et validation d'un CNN | 🔬 | ★★★ | 40 |
| 24.22 | Prédire l'accuracy d'un CNN sur des images retournées, décalées, tournées ou en gris | 🔮 | ★ | 10 |
| 24.23 | Augmentation de données avec torchvision.transforms.v2 | 📦 | ★★★ | 45 |
| 24.24 | Des images synthétiques générées à la volée | 📦 | ★★ | 30 |
| 24.25 | Sommes de sinus et LSTM minuscule : fenêtres, scaler, RMSE | 📦 | ★★★ | 45 |
| 24.26 | Fenêtre, profondeur ou données : qu'est-ce qui aide vraiment ? | 🔬 | ★★★ | 45 |
| 24.27 | Séquences retournées, état porté entre batches, couche par pas de temps | 📦 | ★★★ | 40 |
| 24.28 | Au-delà de nn.Sequential : branches, addition, couches partagées, gel | 📦 | ★★★ | 45 |
| 24.29 | Précision mixte et torch.compile : mesurer avant de croire | 📦 | ★★ | 25 |
| 24.30 | Générer du texte : embeddings, séquences décalées, mots contre caractères | 📦 | ★★★★ | 100 |
| 24.31 | Défi CIFAR-10 frugal : au moins 80 % de test accuracy avec moins de 100 000 paramètres | 🏆 | ★★★★ | 180 |

🚀 *GPU Colab conseillé* pour le mode complet (`FAST_MODE = False`) de 24.18, 24.20, 24.21, 24.23, 24.30 et 24.31 ; le mode rapide tourne sur CPU.
