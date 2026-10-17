# 20 · Deep learning et premiers pas en PyTorch — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch20_deep_learning/06_mes_reponses.md` (créée par `python tools/start_chapter.py 20`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 20.1 à 20.4, 20.6 et 20.7 et de l'estimation 🧮 20.9 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les preuves ∂ 20.5 et 20.8, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), une forme entre guillemets (`"(3, 5)"`, ou `"(7,)"` pour une seule dimension).
> Conventions du chapitre : les formules sont celles de PyTorch (fiche) : dropout inversé, batchnorm avec la variance biaisée pour normaliser et la variance non biaisée pour la statistique glissante, `momentum` de la batchnorm $m = 0{,}1$. Sauf mention contraire, on prend $\epsilon = 0$ dans les calculs à la main.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 20.Q1 — Deep learning et machine learning : qui contient qui ? 🧠 ⏱️ 3 min
*Fiche §20.1, §20.2 · livre §20.1, §20.2 · parcours R*

a) Vrai ou faux : dans l'usage le plus courant, le deep learning est une partie du machine learning.
b) Vrai ou faux : tous les auteurs emploient « machine learning » dans ce sens.
c) Lequel de ces modèles n'est **pas** un réseau profond ? (A) un MLP 784-256-128-10 ; (B) un réseau de convolution à huit couches ; (C) une forêt aléatoire de 500 arbres ; (D) un réseau récurrent à deux couches.
d) Pourquoi parle-t-on d'apprentissage **hiérarchique** ? (A) les couches sont rangées par nombre de neurones décroissant ; (B) chaque couche travaille sur les résultats de la précédente, donc sur des motifs de plus en plus grands et abstraits ; (C) le réseau apprend d'abord les classes les plus fréquentes ; (D) les poids sont triés par ordre d'importance.

### 20.Q2 — Tenseurs : nombre de dimensions, taille, forme 🧠 ⏱️ 3 min
*Fiche §20.2.1 · livre §20.2.1 · parcours R*

Un mini-batch d'images a la forme `(32, 3, 28, 28)` dans PyTorch.

a) Son nombre de dimensions (`x.ndim`).
b) Son nombre d'éléments (`x.numel()`).
c) La forme d'une seule image, `x[0]` (entre guillemets).
d) Vrai ou faux : dans un tenseur, toutes les lignes d'une même dimension ont la même longueur.
e) Le type de `torch.tensor([1.0, 2.0])` : (A) `torch.float64` ; (B) `torch.float32` ; (C) `torch.int64` ; (D) `torch.float16`.

### 20.Q3 — Couche d'entrée, couche de sortie : laquelle contient des neurones ? 🧠 ⏱️ 3 min
*Fiche §20.3, §20.3.1, §20.3.2 · livre §20.3 · parcours R*

a) Vrai ou faux : la couche d'entrée contient des neurones.
b) Vrai ou faux : la couche de sortie contient des neurones.
c) Un MLP dense 784-256-128-10 (784 entrées, 10 sorties) : son nombre de couches cachées.
d) Dans le vocabulaire du livre, une couche « plus haute » qu'une autre est : (A) plus proche de l'entrée ; (B) plus proche de la sortie ; (C) plus large ; (D) entraînée plus tard.

### 20.Q4 — Couche dense et couche d'activation séparée 🧠 ⏱️ 3 min
*Fiche §20.4, §20.4.1, §20.4.2 · livre §20.4.1, §20.4.2 · parcours R*

a) Une couche dense de 3 neurones reçoit les 4 sorties de la couche précédente : son nombre de poids, sans les biais.
b) Son nombre de paramètres, biais compris.
c) Vrai ou faux : `nn.Sequential(nn.Linear(4, 3), nn.ReLU())` calcule la même chose qu'une couche dense de 3 neurones dont l'activation est ReLU.
d) Que doit recevoir `nn.CrossEntropyLoss` ? (A) les probabilités, après une softmax ; (B) les logits, scores bruts de la dernière couche ; (C) la classe prédite par le réseau ; (D) la sigmoïde de chaque score.

### 20.Q5 — Vrai ou faux : le dropout tire ses neurones une fois par epoch 🧠 ⏱️ 3 min
*Fiche §20.4.3 · livre §20.4.3 · parcours R*

a) Vrai ou faux : dans PyTorch, `nn.Dropout` tire les neurones à couper une seule fois par epoch.
b) Une epoch de 500 mini-batches, en mode entraînement : combien de masques de dropout une couche `nn.Dropout` tire-t-elle ?
c) Avec $p = 0{,}2$, par quel facteur `nn.Dropout` multiplie-t-il les activations qu'il garde ?
d) En mode évaluation (`model.eval()`), `nn.Dropout` : (A) coupe une proportion $p$ des neurones ; (B) multiplie toutes les activations par $1 - p$ ; (C) rend son entrée inchangée ; (D) coupe les mêmes neurones qu'au dernier mini-batch d'entraînement.

### 20.Q6 — Vrai ou faux : la batchnorm n'a aucun paramètre 🧠 ⏱️ 3 min
*Fiche §20.4.4 · livre §20.4.4 · parcours R*

a) Vrai ou faux : une couche de batchnorm n'a aucun paramètre appris.
b) Le nombre de paramètres appris de `nn.BatchNorm1d(64)`.
c) Le nombre de statistiques glissantes qu'elle tient à jour (moyennes et variances, sans compter le compteur de mini-batches).
d) En mode évaluation, elle normalise avec : (A) la moyenne et la variance du mini-batch courant ; (B) ses statistiques glissantes ; (C) rien, elle rend son entrée ; (D) les statistiques du premier mini-batch d'entraînement.

### 20.Q7 — Convolution et pooling en un coup d'œil 🧠 ⏱️ 3 min
*Fiche §20.4.5, §20.4.6 · livre §20.4.5, §20.4.6 · parcours R*

a) Le nombre de paramètres de `nn.Conv2d(1, 2, kernel_size=3)` : deux filtres 3 × 3 sur une image en niveaux de gris, avec un biais par filtre.
b) Un max pooling 2 × 2 (blocs disjoints) sur une image de 512 × 512 pixels : la largeur de l'image de sortie.
c) Un pooling 3 × 3 (blocs disjoints) sur une image de 30 × 30 : sa largeur de sortie.
d) Vrai ou faux : un filtre 3 × 3 avec un zero-padding de 1 pixel garde à une image de 5 × 4 pixels sa taille de 5 × 4.

### 20.Q8 — Récurrence n'est pas récursion 🧠 ⏱️ 3 min
*Fiche §20.4.7 · livre §20.4.7*

a) Vrai ou faux : « récurrent » et « récursif » sont deux mots pour la même idée.
b) Laquelle de ces tâches demande une mémoire de ce qui précède ? (A) classer une photo ; (B) prédire le mot suivant d'une phrase ; (C) estimer le prix d'une maison d'après ses features ; (D) détecter un chat dans une image.
c) Vrai ou faux : une couche dense traite chaque entrée sans rien garder de l'entrée précédente.

### 20.Q9 — Aplatir, remodeler, rogner, rembourrer, suréchantillonner, bruiter 🧠 ⏱️ 3 min
*Fiche §20.4.8, §20.5 · livre §20.4.8, §20.5*

a) `nn.ZeroPad2d(1)` sur une image de 8 × 8 : la largeur de sortie.
b) `nn.Upsample(scale_factor=2)` sur une image de 7 × 7 : la largeur de sortie.
c) Un recadrage centré fait passer une image de 28 × 28 à 24 × 24 : combien de pixels retire-t-il de chaque côté ?
d) Laquelle de ces couches a des paramètres appris ? (A) l'aplatissement ; (B) le suréchantillonnage ; (C) le zero-padding ; (D) aucune des trois.
e) Vrai ou faux : une couche de bruit ajoute aussi son bruit en évaluation.

### 20.Q10 — VGG16 : ce qu'il reconnaît et ce qu'il ignore 🧠 ⏱️ 3 min
*Fiche §20.6 · livre §20.6*

a) Le nombre de couches de convolution de VGG16.
b) Son nombre de couches denses.
c) Le nombre de classes qu'il distingue.
d) Les « 92,7 % de bonnes réponses » que le livre lui attribue sont : (A) une accuracy top-1 ; (B) une accuracy top-5 ; (C) une accuracy sur les données d'entraînement ; (D) une précision.
e) Vrai ou faux : devant un objet qui n'est dans aucune de ses classes, VGG16 peut répondre par un nom de catégorie nouveau.

### 20.Q11 — Construire un réseau profond : par où commencer ? 🧠 ⏱️ 3 min
*Fiche §20.7, §20.7.1 · livre §20.7 · parcours R*

a) La première étape : (A) choisir l'architecture ; (B) regarder et préparer les données ; (C) régler le learning rate ; (D) entraîner un grand réseau pour voir.
b) L'hyperparamètre que le livre juge le plus important : (A) le nombre d'epochs ; (B) la taille des mini-batches ; (C) le learning rate ; (D) la graine aléatoire.
c) Vrai ou faux : il existe une recette qui garantit la bonne architecture pour un problème donné.
d) Un réseau fait surtout de couches de convolution s'appelle : (A) un MLP ; (B) un CNN ; (C) un RNN ; (D) un GAN.

### 20.Q12 — Explication complète ou explication satisfaisante ? 🧠 ⏱️ 3 min
*Fiche §20.8, §20.8.1 · livre §20.8 · parcours R*

a) Vrai ou faux : montrer tous les calculs du réseau, neurone par neurone, donne toujours une explication satisfaisante.
b) Une banque refuse les dossiers que son modèle juge risqués. Pourquoi ne peut-elle pas vérifier, plus tard, que ces refus étaient justifiés ? (A) les données sont trop volumineuses ; (B) elle n'observe jamais si les dossiers refusés auraient remboursé ; (C) le réseau a trop de couches ; (D) la loi lui interdit de garder les dossiers.
c) Un système dont on ne peut pas expliquer les décisions est une : (A) boîte noire ; (B) boîte blanche ; (C) régression linéaire ; (D) dérive.
d) Vrai ou faux : un modèle de crédit qui avait 95 % d'accuracy sur son jeu de test il y a trois ans reste fiable aujourd'hui.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 20.R1 — Que fait weight_decay dans AdamW ? 🔁 ★ ⏱️ 5 min
*Ch. 19 (fiche, section AdamW) · parcours R*

`torch.optim.AdamW` commence chaque pas par $w \leftarrow w\,(1 - \eta\lambda)$, puis fait le pas d'Adam.

a) Un poids vaut $w = 2$, son gradient est nul, $\eta = 0{,}01$ et $\lambda = 0{,}1$ (`weight_decay=0.1`). La valeur de $w$ après un pas d'AdamW (le pas d'Adam est nul quand le gradient est nul).
b) Vrai ou faux : dans AdamW, le weight decay passe par les moyennes mobiles $m$ et $v$.
c) La valeur par défaut de `weight_decay` dans `torch.optim.AdamW`.

### 20.R2 — Pourquoi calcule-t-on la loss à partir des logits ? 🔁 ★ ⏱️ 5 min
*Ch. 17 (fiche, softmax stable) et ch. 18 (fiche, la loss d'un classifieur) · parcours R*

a) Vrai ou faux : `nn.CrossEntropyLoss` attend des probabilités.
b) Pour les logits $(1000, 0)$, la deuxième composante de `log_softmax` (arrondie à l'entier).
c) La somme des sorties d'une softmax, quels que soient les logits.
d) Pourquoi la loss fait-elle elle-même la softmax ? (A) c'est plus rapide à écrire ; (B) calculer `log_softmax` d'un coup évite les dépassements et le logarithme de 0 ; (C) la softmax n'a pas de dérivée ; (D) PyTorch n'a pas de fonction softmax.

### 20.R3 — Limiter l'overfitting d'un arbre, et l'équivalent pour un réseau 🔁 ★ ⏱️ 5 min
*Ch. 13 (fiche, overfitting des arbres) et ch. 9 · parcours R*

a) Un `DecisionTreeClassifier` sans limite apprend par cœur ses données d'entraînement. Quel hyperparamètre réduire en premier ? (A) `max_depth` ; (B) `random_state` ; (C) `n_jobs` ; (D) `criterion`.
b) Lequel de ces gestes régularise un réseau de neurones ? (A) doubler le nombre de neurones de chaque couche ; (B) ajouter du dropout ; (C) entraîner plus d'epochs ; (D) supprimer le jeu de validation.
c) Vrai ou faux : l'early stopping est une régularisation, au sens du livre (toute technique qui retarde l'overfitting).

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 20.1 — Suivre la forme d'un tenseur couche après couche ✏️ ★ ⏱️ 15 min
**Objectif :** prévoir la forme d'un tenseur après chaque couche, et compter les paramètres d'une couche dense.
**Prérequis :** 0A (formes NumPy, `reshape`) · ch. 16 (`count_parameters`) · fiche §20.2.1, §20.4.8 · **Parcours :** R, M

Les formes s'écrivent entre guillemets, comme PyTorch les affiche sans `torch.Size` : `"(32, 784)"`.

a) Le livre remodèle un tenseur de forme $10 \times 3 \times 5$ (§20.4.8). Son nombre d'éléments.
b) On le remodèle en deux dimensions en gardant la première : sa nouvelle forme.
c) Un mini-batch MNIST de forme `(32, 1, 28, 28)` traverse `nn.Flatten()` : la forme de sortie.
d) Puis `nn.Linear(784, 128)` : la forme de sortie.
e) Le nombre de paramètres de cette couche `nn.Linear(784, 128)`.
f) Un mini-batch CIFAR-10 de forme `(32, 3, 32, 32)` traverse `nn.Flatten()` : la forme de sortie.
g) À la sortie de ses convolutions, VGG16 produit, pour **une** image, un tenseur de forme `(1, 512, 7, 7)`. La forme de `torch.flatten(x)`.
h) La forme de `nn.Flatten()(x)` pour le même tenseur.

Dans ta copie : quelle erreur produirait la couche `nn.Linear(25088, 4096)` si l'on utilisait la forme de g) avec un mini-batch de deux images ?

### Ex 20.2 — Dimensionner la couche de sortie et choisir la loss ✏️ ★ ⏱️ 10 min
**Objectif :** choisir le nombre de sorties et la loss PyTorch selon le problème.
**Prérequis :** ch. 17 (softmax, sigmoïde) · ch. 18 (loss d'un classifieur) · fiche §20.3.2 · **Parcours :** R

Les losses : (A) `nn.MSELoss()` ; (B) `nn.BCEWithLogitsLoss()` ; (C) `nn.CrossEntropyLoss()`.

a) Prédire le prix d'une maison : le nombre de sorties du réseau.
b) La loss (une lettre).
c) Reconnaître les 26 lettres de l'alphabet (une lettre par image) : le nombre de sorties.
d) La loss.
e) Classer des e-mails en spam ou non, avec une seule sortie : la loss.
f) Attribuer à des photos 5 labels non exclusifs (plage, montagne, ville, nuit, personnes ; une photo peut en avoir plusieurs) : le nombre de sorties.
g) La loss.
h) Classer des e-mails en spam ou non, avec deux sorties (une par classe) : la loss.

Dans ta copie : pour la question c), quelle forme et quel dtype doivent avoir les labels d'un mini-batch de 64 images ?

### Ex 20.3 — Dropout inversé à la main ✏️ ★★ ⏱️ 15 min
**Objectif :** appliquer le dropout inversé à la main, en passe avant et en passe arrière, et vérifier qu'il garde l'espérance.
**Prérequis :** ch. 2 (variable de Bernoulli) · fiche §20.4.3 · **Parcours :** R, M

a) Le livre applique un dropout de 20 % à une couche de 100 neurones. Combien de neurones sont coupés, en moyenne, à chaque passe avant ?

On applique maintenant un dropout de $p = 0{,}4$ aux activations $\mathbf{a} = (2, -1, 0{,}5, 3, 4)$. Comme ta librairie, on tire cinq nombres uniformes $\mathbf{u} = (0{,}1 ;\ 0{,}7 ;\ 0{,}35 ;\ 0{,}9 ;\ 0{,}5)$ et l'on **garde** le neurone $i$ quand $u_i \ge p$.

b) Le nombre de neurones gardés.
c) Le facteur appliqué aux activations gardées (4 décimales).
d) La deuxième composante de la sortie $\tilde{\mathbf{a}}$ (4 décimales).
e) La somme des composantes de $\tilde{\mathbf{a}}$ (4 décimales).
f) La somme des composantes de $\tilde{\mathbf{a}}$, **en moyenne** sur tous les tirages possibles du masque.
g) La passe arrière reçoit $\partial L/\partial\tilde{\mathbf{a}} = (1, 1, 1, 1, 1)$. La somme des composantes de $\partial L/\partial\mathbf{a}$ (4 décimales).
h) Vrai ou faux : en mode évaluation, la sortie est $(2, -1, 0{,}5, 3, 4)$.

Dans ta copie : avec ces mêmes $\mathbf{u}$, quels neurones garderait-on avec $p = 0{,}6$ ?

### Ex 20.4 — Batchnorm à la main sur un mini-batch de quatre exemples ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer à la main une passe de batchnorm en entraînement, la mise à jour de ses statistiques glissantes, et une prédiction en évaluation.
**Prérequis :** ch. 12 (standardisation) · ch. 19 (moyenne mobile exponentielle) · fiche §20.4.4 · **Parcours :** M

Un mini-batch de 4 exemples à 2 features :
$$\mathbf{X} = \begin{pmatrix} 1 & 10 \\ 3 & 14 \\ 5 & 6 \\ 7 & 10 \end{pmatrix},\qquad \boldsymbol{\gamma} = (2 ;\ 0{,}5),\qquad \boldsymbol{\beta} = (0 ;\ 1).$$
La couche est neuve : ses statistiques glissantes valent $\mu^{\text{run}} = (0, 0)$ et $\sigma^{2,\text{run}} = (1, 1)$, comme dans PyTorch. On prend $\epsilon = 0$.

a) La moyenne de la feature 1 sur le mini-batch.
b) La moyenne de la feature 2.
c) La variance (biaisée) de la feature 1.
d) La variance (biaisée) de la feature 2.
e) La valeur normalisée $\hat x$ du premier exemple, feature 1 (4 décimales).
f) La sortie $y$ du deuxième exemple, feature 2 (4 décimales).
g) Vrai ou faux : la moyenne des sorties $y$ de la feature 1 sur le mini-batch vaut $\beta_1 = 0$.
h) Après ce mini-batch, la moyenne glissante de la feature 1.
i) La moyenne glissante de la feature 2.
j) La variance glissante de la feature 1 (4 décimales ; souviens-toi qu'elle est mise à jour avec la variance non biaisée).
k) En évaluation, la couche reçoit l'exemple $(4, 12)$. Sa sortie pour la feature 1 (3 décimales).

Dans ta copie : la sortie de k) est loin de ce qu'aurait donné la normalisation du mini-batch (un exemple égal à la moyenne aurait donné $\beta_1 = 0$). Pourquoi ? Que faudrait-il pour que les deux se rapprochent ?

### Ex 20.5 — Dropout : espérance, sous-réseaux et règle du test ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer les propriétés du dropout inversé et comparer la règle de l'article d'origine.
**Prérequis :** Ex 20.3 · ch. 2 (espérance, variance, Bernoulli) · fiche §20.4.3 · **Parcours :** M

Une activation $a$ fixée, un masque $m \sim \mathrm{Bernoulli}(1 - p)$ avec $0 \le p < 1$, et $\tilde a = a\,m/(1 - p)$. Réponds dans ta copie.

1. Montre que $\mathbb{E}[\tilde a] = a$ et que $\mathrm{Var}[\tilde a] = a^2\,\frac{p}{1 - p}$. Que devient la variance quand $p \to 1$ ? Quand $p = 0{,}5$ ?
2. Une couche de $n$ neurones avec dropout : combien de masques différents existe-t-il ? Explique pourquoi on dit que le dropout entraîne un très grand nombre de **sous-réseaux** qui partagent leurs poids. Ordre de grandeur pour $n = 100$ ?
3. L'article d'origine (Srivastava et coll., 2014) ne divise pas par $1 - p$ pendant l'entraînement, et multiplie les poids sortants par $1 - p$ au test. La couche suivante est linéaire : $z = \sum_i w_i\,a_i\,m_i$ en entraînement. Montre que $\mathbb{E}[z] = (1 - p)\sum_i w_i a_i$, donc que la règle du test donne exactement l'espérance de $z$.
4. Montre que les deux méthodes (l'article d'origine et le dropout inversé) produisent le même modèle au test, à un changement d'échelle des poids près.
5. Si une non-linéarité suit, la règle du test n'est plus exacte. Avec la convention de la question 3, deux activations $a_1 = 1$, $a_2 = -1$, des poids $w_1 = w_2 = 1$, $p = 0{,}5$ et $h(z) = \max(0, z)$ (ReLU), calcule l'espérance de $h(z)$ sur les quatre masques et compare-la à $h(\mathbb{E}[z])$. Quel résultat général (inégalité) explique l'écart ?

### Ex 20.6 — Compter les 138 millions de paramètres de VGG16 ✏️ ★★ ⏱️ 20 min
**Objectif :** compter les paramètres d'un vrai réseau, couche par couche, et voir où ils se concentrent.
**Prérequis :** ch. 16 (compter les paramètres d'une couche dense) · fiche §20.4.5, §20.6 · **Parcours :** M

VGG16 reçoit des images en couleur de 224 × 224 pixels. Ses 13 convolutions ont toutes des filtres 3 × 3 avec un biais par filtre et un zero-padding de 1 ; elles forment 5 blocs de 64, 64 | 128, 128 | 256, 256, 256 | 512, 512, 512 | 512, 512, 512 filtres, chaque bloc suivi d'un max pooling 2 × 2. Viennent ensuite un aplatissement et trois couches denses de 4 096, 4 096 et 1 000 neurones. Rappel : une convolution de $C_{out}$ filtres $k \times k$ sur $C_{in}$ canaux a $(k^2 C_{in} + 1)\,C_{out}$ paramètres.

a) La largeur de l'image après les 5 max poolings.
b) Le nombre de paramètres de la première convolution (3 canaux en entrée, 64 filtres).
c) Le nombre total de paramètres des 13 convolutions.
d) Le nombre de paramètres de la première couche dense (biais compris).
e) Le nombre total de paramètres de VGG16.
f) La part des paramètres dans les trois couches denses, en % (1 décimale).
g) La mémoire de ces paramètres en `float32` (4 octets), en Mo de $10^6$ octets (1 décimale).
h) La page de torchvision annonce un fichier de poids de 527,8 « MB ». Ce nombre est exprimé : (A) en Mo de $10^6$ octets ; (B) en Mio de $2^{20}$ octets ; (C) en mégabits ; (D) il compte seulement les convolutions.

Dans ta copie : pourquoi les convolutions, qui font l'essentiel des calculs, ont-elles si peu de paramètres ? Quelle couche remplacerait la première couche dense dans un réseau moderne, et combien de paramètres économiserait-elle ?

### Ex 20.7 — Connexion résiduelle et LayerNorm à la main ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer une layer norm à la main et chiffrer ce que change une connexion résiduelle pour le gradient.
**Prérequis :** ch. 18 (règle de la chaîne) · fiche §20.4.8, section sur les connexions résiduelles · **Parcours :** M

Un exemple a les features $\mathbf{x} = (1, 2, 3, 6)$. On prend $\epsilon = 0$, $\boldsymbol\gamma = \mathbf{1}$, $\boldsymbol\beta = \mathbf{0}$.

a) La moyenne de ses features.
b) Leur variance (biaisée).
c) La quatrième composante de `layer_norm(x)` (4 décimales).
d) La somme des carrés des quatre composantes de `layer_norm(x)`.
e) Vrai ou faux : la sortie de la layer norm pour cet exemple ne change pas s'il est placé dans un mini-batch avec d'autres exemples.

Un gradient remonte une chaîne de couches ; chaque couche le multiplie par sa dérivée (on raisonne en dimension 1).

f) 30 couches sans connexion résiduelle, de dérivée 0,5 chacune : le facteur total, en multiple de $10^{-9}$ (4 décimales).
g) Avec des couches de dérivée 0,9 : le nombre minimal de couches pour que le facteur passe sous 0,01.
h) 30 blocs résiduels $y = x + F(x)$, dont chaque branche $F$ a une dérivée de 0,01 : le facteur total (4 décimales).

Dans ta copie : un bloc résiduel dont la branche a une dérivée de $-0{,}5$ fait-il mieux qu'une couche ordinaire de dérivée 0,5 ? Qu'en conclus-tu sur ce qu'il faut à $F$ au début de l'entraînement ?

### Ex 20.8 — Le gradient de la batchnorm ∂ ★★★ ⏱️ 40 min
**Objectif :** dériver la passe arrière de la batchnorm, celle que tu programmeras en 🔨 20.22.
**Prérequis :** Ex 20.4 · ch. 18 (rétropropagation, gradient d'une somme) · fiche §20.4.4 · **Parcours :** M

Une seule feature, un mini-batch de $m$ valeurs $x_1, \dots, x_m$. On note $\mu = \frac1m\sum_i x_i$, $\sigma^2 = \frac1m\sum_i(x_i - \mu)^2$, $s = (\sigma^2 + \epsilon)^{-1/2}$, $\hat x_i = (x_i - \mu)\,s$ et $y_i = \gamma\hat x_i + \beta$. On connaît $g_i = \partial L/\partial y_i$. Réponds dans ta copie.

1. Montre que $\partial L/\partial\beta = \sum_i g_i$ et $\partial L/\partial\gamma = \sum_i g_i\,\hat x_i$.
2. Montre que $\sum_i \hat x_i = 0$ et que $\sum_i \hat x_i^2 = m\,\sigma^2 s^2$ (qui vaut $m$ quand $\epsilon = 0$).
3. Calcule $\partial\mu/\partial x_j$, $\partial\sigma^2/\partial x_j$ (montre que le terme qui vient de $\mu$ s'annule), puis $\partial\hat x_i/\partial x_j$. On note $\delta_{ij}$ le symbole de Kronecker.
4. Déduis-en la formule, exacte pour tout $\epsilon \ge 0$ :
$$\frac{\partial L}{\partial x_j} = \frac{\gamma\,s}{m}\left(m\,g_j - \sum_i g_i - \hat x_j\sum_i g_i\,\hat x_i\right) = \frac{\gamma\,s}{m}\left(m\,g_j - \frac{\partial L}{\partial\beta} - \hat x_j\,\frac{\partial L}{\partial\gamma}\right).$$
Combien d'opérations coûte-t-elle pour les $m$ valeurs, une fois $\partial L/\partial\beta$ et $\partial L/\partial\gamma$ calculés ?
5. Montre que $\sum_j \partial L/\partial x_j = 0$, et que, pour $\epsilon = 0$, $\sum_j \hat x_j\,\partial L/\partial x_j = 0$. Interprète : quelles modifications de l'entrée la batchnorm rend-elle invisibles à la loss ?
6. En évaluation, $\mu$ et $\sigma^2$ sont les statistiques glissantes, des constantes. Quel est alors $\partial L/\partial x_j$ ?
7. Pourquoi ce calcul n'a-t-il pas de sens pour $m = 1$ ?
8. Pour une image (forme $(N, C, H, W)$), la batchnorm calcule une moyenne et une variance par canal sur $N \times H \times W$ valeurs. Que deviennent $m$ et les sommes de la question 4 ?

### Ex 20.9 — Fermi : entraîner VGG16 sur ImageNet chez soi ? 🧮 ★★ ⏱️ 20 min
**Objectif :** estimer en ordre de grandeur le calcul et la mémoire qu'a demandés l'entraînement de VGG16.
**Prérequis :** Ex 20.6 · ch. 18 (coût d'une passe arrière) · fiche §20.6 · **Parcours :** M

Une passe avant de VGG16 coûte environ 15,5 milliards d'opérations (multiplications-additions) par image ; on compte qu'une passe avant et une passe arrière en coûtent ensemble 3 fois plus. ImageNet a 1,28 million d'images d'entraînement, et VGG16 a été entraîné pendant environ 74 epochs.

a) Le coût d'entraînement par image, en milliards d'opérations (1 décimale).
b) Le coût total de l'entraînement, en multiple de $10^{18}$ opérations (2 décimales).
c) La durée sur un GPU qui fait réellement $10^{13}$ opérations par seconde, en jours (1 décimale).
d) La durée sur un ordinateur portable qui en fait $10^{11}$ par seconde, en jours (arrondie à l'entier).
e) La mémoire des paramètres, des gradients et des deux états d'Adam (si tu l'entraînais avec Adam), en `float32`, pour les 138 357 544 paramètres, en Go de $10^9$ octets (2 décimales).

Dans ta copie : qu'est-ce que ce calcul oublie, côté mémoire et côté temps ? Pourquoi, en pratique, télécharge-t-on des poids pré-entraînés plutôt que d'entraîner soi-même ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 20.10 — Agents, superviseurs, directeur : expliquer l'apprentissage hiérarchique 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer ce qu'est un réseau profond à quelqu'un qui n'en a jamais vu.
**Prérequis :** fiche §20.2, §20.8 · **Parcours :** R

Dans ta copie, en 6 lignes au plus et sans formule, explique à un ami ce qu'est un réseau profond en reprenant l'analogie de la banque : les agents, les superviseurs, le directeur. Dis ce que chaque niveau regarde, ce qui rend le système plus précis, et pourquoi il devient plus difficile d'expliquer une décision. Puis relis-toi : as-tu évité de dire que chaque niveau « comprend » le dossier ? Où l'analogie s'arrête-t-elle (pense à la façon dont les poids sont réglés) ?

### Ex 20.11 — Refus de prêt automatisé : explication, dérive et biais ⚖️ ★★★ ⏱️ 40 min
**Objectif :** analyser un refus de prêt décidé par un réseau, du point de vue du demandeur, de la banque et du droit.
**Prérequis :** ch. 3 (matrice de confusion, taux d'erreur par groupe) · fiche §20.8, §20.8.1 · **Parcours :** R

Une banque en ligne accorde ses petits prêts avec un MLP entraîné sur ses dossiers de 2019 à 2022 : 40 features, dont le code postal, l'âge, le type de contrat de travail, l'historique bancaire. Le réseau rend une probabilité de défaut ; au-dessus de 15 %, le prêt est refusé, sans intervention humaine. Léa, 24 ans, en CDD depuis 14 mois, reçoit un refus et demande pourquoi. Réponds dans ta copie.

1. La banque lui envoie la liste des 40 valeurs de son dossier et les poids du réseau. Est-ce une explication complète ? Satisfaisante ? Rédige en trois phrases ce que serait, pour Léa, une explication **satisfaisante** au sens du livre.
2. Le code postal n'est pas une donnée « sensible », mais il peut en être un **substitut** (*proxy*). Explique comment un modèle peut discriminer selon une caractéristique qu'il ne voit pas. Comment le vérifierais-tu avec une matrice de confusion par groupe (ch. 3) ?
3. Les données vont de 2019 à 2022. Cite deux changements économiques qui pourraient rendre le modèle faux aujourd'hui. Comment détecterais-tu cette **dérive** sans attendre les défauts de paiement ?
4. La banque n'a observé que les remboursements des prêts **accordés**. Explique pourquoi ce biais de sélection l'empêche de savoir si ses refus sont justifiés. Propose une façon (coûteuse) de le corriger.
5. Le droit. D'après l'encadré 🕰️ de la fiche §20.8.1 : l'article 22 du RGPD s'applique-t-il à ce refus ? Dans quelle catégorie l'AI Act range-t-il ce système, et quelles obligations en découlent ? Que peut demander Léa, au titre de l'article 86 ?
6. Tu es le data scientist de la banque. Propose quatre mesures concrètes, techniques ou d'organisation, à mettre en place avant la prochaine version du modèle.

### Ex 20.12 — Ioffe & Szegedy (2015), puis Santurkar et coll. (2018) : lecture critique 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a proposé la batchnorm, puis celui qui a remis en cause son explication, et juger ce qui reste vrai.
**Prérequis :** Ex 20.4 · fiche §20.4.4 · **Parcours :** M

Lis S. Ioffe et C. Szegedy, « Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift », ICML 2015 ([arXiv:1502.03167](https://arxiv.org/abs/1502.03167) ; les sections 1 à 3 et 4.2 suffisent), puis S. Santurkar, D. Tsipras, A. Ilyas et A. Madry, « How Does Batch Normalization Help Optimization? », NeurIPS 2018 ([arXiv:1805.11604](https://arxiv.org/abs/1805.11604) ; les sections 1 à 3). Réponds dans ta copie.

1. Comment Ioffe et Szegedy définissent-ils l'*internal covariate shift* ? Pourquoi, selon eux, ralentit-il l'entraînement ?
2. Recopie l'algorithme 1 (la transformation sur un mini-batch) avec les notations de la fiche. Pourquoi les auteurs ajoutent-ils $\gamma$ et $\beta$ (fin du §3) ? Que permettent-ils de retrouver ?
3. L'algorithme 2 décrit l'inférence. Quelle variance utilisent-ils pour la population, biaisée ou non ? Compare avec ce que fait PyTorch (fiche).
4. Pour une couche de convolution (§3.2), sur quelles valeurs les statistiques sont-elles calculées ? Relie avec la forme `(N, C, H, W)` de ta librairie.
5. Quels résultats le §4.2 annonce-t-il sur ImageNet (vitesse d'entraînement, learning rate, dropout) ? Lesquels relèvent de l'optimisation, lesquels de la régularisation ?
6. Santurkar et coll., §2 : décris leur expérience du « bruit » ajouté après la batchnorm. Qu'est-ce qu'elle montre sur le lien entre l'*internal covariate shift* et les performances ?
7. Leur §3 : quelle propriété du paysage de la loss la batchnorm améliore-t-elle, selon eux ? Que veut dire « gradients plus prédictifs » ? En quoi cela explique-t-il qu'on puisse prendre un learning rate plus grand ?
8. Bilan : qu'est-ce qui, dans l'article de 2015, reste vrai (la méthode, les résultats) et qu'est-ce qui a été contesté (l'explication) ? Que retiens-tu sur la façon de lire l'explication qu'un article donne de ses propres résultats ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 20.E1 — Dropout : fonctionnement, et ce que change model.eval() 💼 ★★ ⏱️ 10 min
*Fiche §20.4.3, section PyTorch en pratique · parcours R*

« Expliquez le dropout. Que se passe-t-il quand on appelle `model.eval()`, et que se passe-t-il si on l'oublie ? »

### 20.E2 — Batchnorm ou LayerNorm : différences et usages 💼 ★★ ⏱️ 10 min
*Fiche §20.4.4, §20.4.8 · parcours R*

« Quelle est la différence entre la batchnorm et la layer norm ? Pourquoi les Transformers utilisent-ils la seconde ? »

### 20.E3 — Décrire une boucle d'entraînement PyTorch, étape par étape 💼 ★★ ⏱️ 10 min
*Fiche, section PyTorch en pratique · parcours R*

« Décrivez-moi une boucle d'entraînement PyTorch, étape par étape. Que fait chaque ligne, et dans quel ordre doivent-elles venir ? »

### 20.E4 — Ton modèle surapprend : que fais-tu, et dans quel ordre ? 💼 ★★ ⏱️ 10 min
*Fiche §20.7.1, section sur la régularisation · parcours R*

« Votre modèle a 99 % d'accuracy en entraînement et 85 % en validation. Que faites-vous, et dans quel ordre ? »

### 20.E5 — Pourquoi les connexions résiduelles rendent-elles les réseaux très profonds entraînables ? 💼 ★★ ⏱️ 10 min
*Fiche, section sur les connexions résiduelles · parcours R*

« Pourquoi les connexions résiduelles permettent-elles d'entraîner des réseaux de cent couches ou plus ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb` ; ils arriveront avec la prochaine session de génération du chapitre.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 20.13 | Premiers tenseurs PyTorch : créer, convertir, dtype, device | 📦 | ★ | 15 |
| 20.14 | Prédire les formes produites par les couches utilitaires | 🔮 | ★ | 15 |
| 20.15 | Autograd : retrouver les gradients du réseau minuscule du ch. 18 | 📦 | ★★ | 20 |
| 20.16 | dropout_forward et dropout_backward | 🔨 | ★★ | 25 |
| 20.17 | layer_norm, l2_penalty et EarlyStopping | 🔨 | ★★ | 30 |
| 20.18 | Écrire sa classe nn.Module et compter ses paramètres | 📦 | ★★ | 25 |
| 20.19 | Dataset et DataLoader : MNIST en mini-batches | 📦 | ★★ | 25 |
| 20.20 | Zoo des couches : convolution, pooling, récurrente, utilitaires, bruit | 📦 | ★★ | 30 |
| 20.21 | Reproduire la figure 20.17 : le plan de VGG16 couche par couche | 🎨 | ★★ | 30 |
| 20.22 | batchnorm_forward et batchnorm_backward | 🔨 | ★★★ | 75 |
| 20.23 | La boucle d'entraînement PyTorch complète sur MNIST | 📦 | ★★★ | 45 |
| 20.24 | Chasse aux bugs PyTorch : cinq erreurs classiques | 🐛 | ★★ | 30 |
| 20.25 | Courbes train/validation : overfitting, eval() oublié, fuite | 📈 | ★★ | 20 |
| 20.26 | Un modèle ImageNet face à des images qu'il n'a jamais vues | 🔮 | ★★ | 25 |
| 20.27 | Sauvegarder, recharger et reproduire un entraînement | 🛠️ | ★★ | 25 |
| 20.28 | Dropout, batchnorm, AdamW et early stopping sur Fashion-MNIST | 📦 | ★★★ | 60 |
| 20.29 | Ablation : qu'est-ce qui retarde l'overfitting ? | 🔬 | ★★★ | 60 |
| 20.30 | Un réseau de 30 couches : connexions résiduelles et LayerNorm | 🔬 | ★★★ | 45 |
| 20.31 | Défi Fashion-MNIST : au moins 90 % avec un MLP de 300 k paramètres au plus | 🏆 | ★★★★ | 120 |
