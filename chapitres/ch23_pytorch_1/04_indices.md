# 23 · PyTorch en pratique 1 : du dataset au modèle sauvegardé — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 23.Q1 — Pourquoi utiliser une bibliothèque plutôt que tout coder soi-même ?

<details><summary>Indice 1</summary>

Relis les §23.1 et §23.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Sépare deux questions : ce qu'on **apprend** en écrivant soi-même un algorithme (c'est ce que tu as fait avec `mylearn`), et ce qu'une bibliothèque mûre apporte **en plus** quand il faut entraîner de vrais modèles. Pour c), cherche l'analogie où l'on gagne en niveau d'abstraction.

</details>
<details><summary>Indice 3</summary>

a) Faux : écrire `mylearn` t'a appris ce que font les fonctions que tu vas appeler. Pour b), cherche ce qu'une équipe de spécialistes optimise pendant des années (la vitesse, les cas numériques limites) ; pour c), l'option qui parle de penser au problème plutôt qu'à la machine ; pour d), rappelle-toi comment les tests de `mylearn` décident si ta fonction est juste ; pour e), demande-toi qui choisit l'architecture, la loss et le prétraitement.

</details>

### 23.Q2 — Avertissement ou erreur ? Versions et dépréciations

<details><summary>Indice 1</summary>

Relis les §23.1.3 et §23.2.1 de la fiche, et l'encadré 🕰️ sur Keras.

</details>
<details><summary>Indice 2</summary>

Un avertissement s'affiche et le programme continue ; une exception l'arrête, sauf si un `try` la rattrape. Une dépréciation annonce un changement futur. Pour c), rappelle-toi ce qui a changé entre Keras 2 et Keras 3 (noms d'arguments, méthodes retirées).

</details>
<details><summary>Indice 3</summary>

a) B : un `FutureWarning` prévient qu'un comportement va changer, le code tourne encore. Pour b), demande-toi ce que fait Python d'une exception que personne ne rattrape ; pour c), cherche dans la fiche une méthode du livre qui a été retirée ; pour d), l'option qui fige ce qui peut changer d'une année à l'autre ; pour e), demande-toi ce que tu perds en ne voyant plus aucun avertissement.

</details>

### 23.Q3 — Déboguer méthodiquement : réduire, isoler, reconstruire

<details><summary>Indice 1</summary>

Relis le §23.2.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Dans le message `mat1 and mat2 shapes cannot be multiplied (A x B and C x D)`, `mat1` est l'entrée de la couche, un mini-batch de forme `(taille du mini-batch, features)`, et `mat2` est la matrice de poids de la couche, de forme `(in_features, out_features)`. Le produit n'est possible que si $B = C$.

</details>
<details><summary>Indice 3</summary>

a) C : on réduit le problème à l'appel le plus petit qui plante, puis on reconstruit à partir de ce qui marche. Pour b), relis la fin du §23.2.2 ; pour c), lis le premier nombre de `mat1` ; pour d), lis le second nombre de `mat1`, ce que sort la couche précédente ; pour e), demande-toi quel code a fourni les formes fautives.

</details>

### 23.Q4 — Les trois sens du mot « modèle »

<details><summary>Indice 1</summary>

Relis les §23.3 et §23.3.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour chaque phrase, demande-toi : y a-t-il des poids appris ? Parle-t-on d'un réseau précis ou d'une façon de programmer ? Pour d), rappelle-toi ce que l'on doit faire **avant** `load_state_dict`.

</details>
<details><summary>Indice 3</summary>

a) A : un plan de couches dessiné avant l'entraînement n'a pas de poids appris. Pour b), regarde ce que contient l'envoi ; pour c), demande-toi si l'on parle d'un réseau ; pour d), cherche dans le §23.9.2 pourquoi on reconstruit le modèle avant de charger ; pour e), demande-toi si « les formes s'enchaînent » dit quoi que ce soit de la qualité.

</details>

### 23.Q5 — Tenseurs, arrays et type float32

<details><summary>Indice 1</summary>

Relis les §23.3.2, §23.3.3 et §23.5.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le format d'images de PyTorch est `(N, C, H, W)`. `torch.tensor` crée un tenseur du type par défaut de PyTorch ; `torch.from_numpy` garde le type du tableau NumPy et sa mémoire. Un `float32` tient sur 32 bits.

</details>
<details><summary>Indice 3</summary>

a) 4 : les axes N, C, H et W (même avec un seul canal). Pour b), cherche le type par défaut des réels en PyTorch ; pour c), le type par défaut des réels en NumPy ; pour d), demande-toi si `from_numpy` copie ; pour e), divise 32 par le nombre de bits d'un octet.

</details>

### 23.Q6 — channels_first, channels_last et NCHW

<details><summary>Indice 1</summary>

Relis les §23.3.4 et §23.5.1 de la fiche, et la figure des deux rangements.

</details>
<details><summary>Indice 2</summary>

Une forme d'image se lit toujours (hauteur, largeur), le nombre de lignes d'abord ; channels_first met les canaux devant, channels_last derrière. Pour d), `permute` reçoit, pour chaque axe du **résultat**, le numéro de l'axe d'origine qui y va.

</details>
<details><summary>Indice 3</summary>

a) `"(3, 48, 64)"` : 3 canaux, 48 lignes, 64 colonnes. Pour b), déplace les canaux à la fin ; pour c), rappelle-toi le format du ch. 21 ; pour d), dans NHWC, N est l'axe 0, H l'axe 1, W l'axe 2 et C l'axe 3 : écris les numéros dans l'ordre N, C, H, W ; pour e), pense à une image carrée de 3 × 3 pixels en couleur.

</details>

### 23.Q7 — CPU, GPU, MPS, TPU : où calcule PyTorch ?

<details><summary>Indice 1</summary>

Relis le §23.3.5 de la fiche et son encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

Chaque type de processeur a son nom de device ; le GPU d'Apple passe par le framework Metal, mais son device porte un autre nom. Un calcul a besoin que tous ses tenseurs soient au même endroit. Sur Colab, le type de processeur se choisit dans le menu de l'environnement d'exécution.

</details>
<details><summary>Indice 3</summary>

a) C : `"mps"`, pour Metal Performance Shaders. Pour b), demande-toi où se ferait le produit de matrices ; pour c), cherche ce qu'un GPU fait de mieux qu'un CPU ; pour d), demande-toi s'il y a un GPU dans un environnement CPU ; pour e), relis ce que dit l'encadré sur PyTorch/XLA et TorchTPU.

</details>

### 23.Q8 — Iris, MNIST, Fashion-MNIST : les « hello world » du ML

<details><summary>Indice 1</summary>

Relis les §23.4.1 et §23.5.2 de la fiche, et l'encadré 🕰️ sur Fashion-MNIST.

</details>
<details><summary>Indice 2</summary>

Fashion-MNIST a été conçu comme un **remplaçant direct** de MNIST : même format, mêmes tailles. Iris décrit chaque fleur par des longueurs et des largeurs de deux de ses parties.

</details>
<details><summary>Indice 3</summary>

a) 60 000. Pour b), compte les types de vêtements ; pour c), relis les trois raisons données par ses auteurs ; pour d), relis ce que veut dire « remplaçant direct » ; pour e), multiplie le nombre de parties de la fleur mesurées par le nombre de mesures de chacune.

</details>

### 23.Q9 — Labels entiers, one-hot et logits

<details><summary>Indice 1</summary>

Relis les §23.5.7 et §23.6.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

`nn.CrossEntropyLoss` fait elle-même la softmax. `F.one_hot` rend un tenseur de forme `(nombre de labels, nombre de classes)` ; sans `num_classes`, le nombre de classes est deviné d'après le plus grand label.

</details>
<details><summary>Indice 3</summary>

a) B : des logits, puisque la loss applique sa propre softmax. Pour b), relis le type et la forme des labels dans le §23.5.5 ; pour c), demande-toi si une seconde softmax produit une valeur interdite ; pour d), compte les labels et lis `num_classes` ; pour e), applique la règle qui devine le nombre de classes à partir du plus grand label (les classes commencent à 0).

</details>

### 23.Q10 — fit() décortiqué : batch, epoch, validation, train() et eval()

<details><summary>Indice 1</summary>

Relis les §23.6.3, §23.7 et §23.8.1 de la fiche, et l'encadré « `model.eval()` ne coupe pas les gradients ».

</details>
<details><summary>Indice 2</summary>

Distingue ce qui change le **comportement des couches** (dropout, batchnorm) de ce qui coupe le **calcul des gradients**. Pour c), rappelle-toi la règle du ch. 8 : un jeu qui sert à choisir quelque chose ne mesure plus honnêtement ce choix. Pour e), demande-toi ce que reçoit le `compile()` de Keras.

</details>
<details><summary>Indice 3</summary>

a) Faux : `model.eval()` change le dropout et la batchnorm ; ce sont `torch.no_grad()` ou `torch.inference_mode()` qui coupent les gradients. Pour b), c'est la même distinction ; pour c), demande-toi sur quel jeu le livre choisit l'epoch d'arrêt ; pour d), relis le paragraphe sur `loss.item()` au §23.7 ; pour e), cherche la ligne PyTorch qui choisit la loss et l'optimiseur.

</details>

### 23.Q11 — Sauvegarder tout, les poids ou l'architecture ?

<details><summary>Indice 1</summary>

Relis les §23.9.1 à §23.9.3 de la fiche et l'encadré 🕰️ sur les formats.

</details>
<details><summary>Indice 2</summary>

Pour reprendre « comme si de rien n'était », liste tout ce qui détermine la suite de l'entraînement : les poids, mais aussi ce que l'optimiseur a accumulé, l'endroit où l'on s'est arrêté et les tirages aléatoires à venir. Pour d), une couche `Linear` a deux tenseurs ; `Flatten` et `ReLU` n'en ont aucun.

</details>
<details><summary>Indice 3</summary>

a) C : les poids, l'état de l'optimiseur, l'epoch et l'état des générateurs aléatoires. Pour b), regarde ce que rend `model.state_dict()` (des noms et des tenseurs) ; pour c), relis pourquoi `weights_only=True` est devenu le défaut ; pour d), compte deux tenseurs (poids et biais) par couche qui a des paramètres ; pour e), relis ce que vérifie `load_state_dict`.

</details>

### 23.Q12 — 🕰️ Que devient le code Keras du livre en 2026 ?

<details><summary>Indice 1</summary>

Relis les encadrés 🕰️ de la fiche, en particulier ceux du §23.3.3, du §23.6.3 et du §23.8.2.

</details>
<details><summary>Indice 2</summary>

Keras 3 (2023) est redevenu une bibliothèque à plusieurs backends, mais pas avec ceux de 2017. `torch.compile` transforme un modèle en code plus rapide ; le `compile()` de Keras choisit la loss et l'optimiseur.

</details>
<details><summary>Indice 3</summary>

a) B : JAX, TensorFlow et PyTorch. Pour b), relis l'encadré du §23.8.2 ; pour c), compare ce que font les deux fonctions ; pour d), relis ce que l'encadré du §23.3.3 dit de Theano (2017) et de CNTK (2019) ; pour e), relis l'encadré sur les formats de sauvegarde.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 23.R1 — Ch. 22 : (N, T, D) et batch_first

<details><summary>Indice 1</summary>

Relis, au ch. 22, le paragraphe sur la forme des entrées d'une couche récurrente et celui sur `output` et `h_n`.

</details>
<details><summary>Indice 2</summary>

Avec `batch_first=True`, l'entrée est `(N, T, D)` ; sans, `(T, N, D)`. `h_n` a toujours la forme `(nombre de couches × nombre de sens, N, H)`, quel que soit `batch_first`.

</details>
<details><summary>Indice 3</summary>

a) `"(32, 20, 50)"` : 32 phrases, 20 tokens, 50 valeurs par token. Pour b), échange les deux premiers axes ; pour c), applique la forme de `h_n` avec 2 couches, 1 sens et $H = 64$ ; pour d), relis ce que la documentation de `nn.GRU` dit de `batch_first` et des états ; pour e), compare la place de N dans les deux formes.

</details>

### 23.R2 — Ch. 20 : Les cinq lignes d'une étape d'entraînement

<details><summary>Indice 1</summary>

Relis, au ch. 20, la boucle d'entraînement, et le §23.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Il faut des logits avant de calculer une loss, une loss avant de calculer des gradients, et des gradients avant de mettre à jour les poids. Les gradients de PyTorch s'**accumulent** dans `.grad`. Pour d), demande-toi ce qu'un tenseur de loss garde avec lui.

</details>
<details><summary>Indice 3</summary>

a) La suite commence par B ; enchaîne la passe avant, la loss, la passe arrière, puis la mise à jour. Pour b), relis ce que fait `zero_grad` ; pour c), demande-toi laquelle des quatre options est le travail de l'autograd ; pour d), `loss` est relié à tout le graphe de calcul du mini-batch, `loss.item()` est un nombre Python ; pour e), relis le §23.10.2 sur la place de `scheduler.step()`.

</details>

### 23.R3 — Ch. 2 : Moyenne et écart-type du train pour normaliser les pixels

<details><summary>Indice 1</summary>

Relis, au ch. 2, la moyenne et l'écart-type, et le §23.5.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

$\sigma = \sqrt{\frac{1}{n}\sum_i (x_i - \mu)^2}$ avec $n = 5$. Écris les cinq écarts à la moyenne, élève-les au carré, fais leur moyenne, puis prends la racine. Pour c), garde $\sigma$ non arrondi.

</details>
<details><summary>Indice 3</summary>

a) $\mu = (0 + 0{,}2 + 0{,}4 + 0{,}4 + 1)/5 = 0{,}4$. Pour b), les écarts sont $-0{,}4$, $-0{,}2$, $0$, $0$ et $0{,}6$ : additionne leurs carrés, divise par 5, prends la racine ; pour c), applique $(0{,}9 - \mu)/\sigma$ ; pour d), relis la règle du ch. 12 ; pour e), cherche l'option qui centre puis réduit.

</details>

<a id="papier"></a>

## ✏️ 🧮 Papier-crayon

### Ex 23.1 — Remodeler 12 éléments : indices et ordre de lecture

<details><summary>Indice 1</summary>

Relis le §23.5.1 de la fiche, sa formule de l'indice plat et sa figure.

</details>
<details><summary>Indice 2</summary>

`reshape` ne déplace rien : l'élément d'indice $(i, j)$ d'une forme $(D_1, D_2)$ est celui de position $i\,D_2 + j$ dans la liste, et pour $(D_1, D_2, D_3)$, la position $i\,D_2 D_3 + j\,D_3 + k$. Pour d), fais le chemin inverse : divise la position de Q par $D_2 D_3$, puis le reste par $D_3$. Pour f), `a.T[i, j]` vaut `a[j, i]`.

</details>
<details><summary>Indice 3</summary>

a) Position $2 \times 4 + 1 = 9$, la lettre V (M est 0, N 1… V 9). Pour b) et c), calcule la position avec la formule de chaque forme, puis traduis-la en lettre ; pour d), Q est à la position 4 ; pour e), cherche les formes dont le produit des tailles n'est pas 12 ; pour f), lis la première ligne de `a.T`, qui est la première colonne de `a`, puis le début de la deuxième ; pour g), applique la formule à trois indices.

</details>

### Ex 23.2 — De NHWC à NCHW : où atterrit chaque pixel ?

<details><summary>Indice 1</summary>

Relis les §23.3.4 et §23.5.1 de la fiche, en particulier la différence entre `reshape` et `permute`.

</details>
<details><summary>Indice 2</summary>

Pour `x` de forme $(2, 3, 3)$, la position de `x[h, w, c]` est $9h + 3w + c$. `y = x.permute(2, 0, 1)` vérifie `y[c, h, w] = x[h, w, c]` : les mêmes valeurs, d'autres indices, la même mémoire. Après `.contiguous()`, `z` est rangé dans **son** ordre C : la position de `z[c, h, w]` est $6c + 3h + w$. `reshape`, lui, relit la mémoire de `x` dans l'ordre et la redécoupe.

</details>
<details><summary>Indice 3</summary>

a) $9 \times 1 + 3 \times 2 + 0 = 15$. Pour b), écris les tailles de `x` dans l'ordre (2, 0, 1) ; pour c), retourne la relation `y[c, h, w] = x[h, w, c]` ; pour d), applique $6c + 3h + w$ ; pour e), demande-toi si `y` est contigu ; pour f), `w = x.reshape(3, 2, 3)` a pour élément `[0, 0, 1]` la valeur de la position 1 de la mémoire de `x` : trouve ses indices `(h, w, c)` ; pour g), utilise la relation de c) ; pour h), multiplie le nombre de valeurs par 4 octets.

</details>

### Ex 23.3 — 623 290 paramètres : refaire le compte du livre

<details><summary>Indice 1</summary>

Relis le §23.6.2 de la fiche et son encadré 🧮 « les paramètres d'une couche dense ».

</details>
<details><summary>Indice 2</summary>

Une couche `Linear(n_in, n_out)` a $(n_{in} + 1)\,n_{out}$ paramètres. Additionne couche par couche ; `Flatten` et `ReLU` n'en ont pas.

</details>
<details><summary>Indice 3</summary>

a) $(784 + 1) \times 784 = 615\,440$. Pour b), applique la formule à 784 entrées et 10 neurones ; pour c), additionne ; pour d), divise a) par c) ; pour e) et f), refais le compte de chaque couche avec les nouvelles tailles ; pour g), demande-toi ce qu'il y aurait à apprendre dans un aplatissement.

</details>

### Ex 23.4 — Une cross-entropy à la main : indice de classe ou one-hot

<details><summary>Indice 1</summary>

Relis les §23.5.7 et §23.6.3 de la fiche, et la formule $L = -z_y + \ln \sum_j e^{z_j}$.

</details>
<details><summary>Indice 2</summary>

Calcule $e^{1}$, $e^{2}$ et $e^{0}$ une fois pour toutes, et leur somme. Avec une cible en probabilités $\mathbf{q}$, la loss est $-\sum_j q_j \ln \operatorname{softmax}(\mathbf{z})_j$. Pour e), recommence le calcul avec les trois probabilités de a) **à la place** des logits. Pour g), toutes les classes ont la même probabilité.

</details>
<details><summary>Indice 3</summary>

a) $e^2 / (e^1 + e^2 + e^0) = 7{,}389 / 11{,}107 \approx 0{,}665$. Pour b), prends le logarithme de la somme ; pour c), applique $-z_1 + \text{b)}$ ; pour d), développe la somme avec $\mathbf{q} = (0, 1, 0)$ : un seul terme reste ; pour e), calcule $-p_1 + \ln(e^{p_0} + e^{p_1} + e^{p_2})$ avec les probabilités de a) ; pour f), la loss de la seconde image est $\ln 3$ : fais la moyenne des deux ; pour g), $-\ln(1/10)$.

</details>

### Ex 23.5 — Batches par epoch et nombre de mises à jour

<details><summary>Indice 1</summary>

Relis le §23.7 de la fiche et sa formule du nombre de mini-batches.

</details>
<details><summary>Indice 2</summary>

Le train fait $60\,000 - 6\,000 = 54\,000$ images. Sans `drop_last`, on arrondit $N / B$ **au-dessus** ; avec, **en dessous**. Le dernier mini-batch contient le reste de la division. Chaque mini-batch donne une mise à jour.

</details>
<details><summary>Indice 3</summary>

a) $54\,000 / 256 = 210{,}9$, arrondi au-dessus : 211. Pour b), arrondis en dessous ; pour c), calcule $54\,000 - 210 \times 256$ ; pour d), multiplie a) par le nombre d'epochs ; pour e) et f), refais a) avec les nouveaux $N$ et $B$ ; pour g), multiplie b) par la taille d'un mini-batch.

</details>

### Ex 23.6 — Early stopping à la main : patience et min_delta

<details><summary>Indice 1</summary>

Relis le §23.10.3 de la fiche et sa figure ; la règle est rappelée dans l'énoncé.

</details>
<details><summary>Indice 2</summary>

Fais un tableau à trois colonnes : l'epoch, la meilleure loss **avant** cette epoch, le compteur **après** cette epoch. Une epoch est une amélioration si sa loss est $< \text{meilleure} - \delta$ ; sinon le compteur augmente. Avec $\delta = 0{,}005$, une baisse de moins de 0,005 n'est **pas** une amélioration et ne change **pas** la meilleure loss.

</details>
<details><summary>Indice 3</summary>

a) Améliorations aux epochs 1, 2, 3, 5 et 7 (0,379 < 0,38) ; le compteur vaut 1, 2, 3 aux epochs 8, 9, 10 : arrêt à la fin de l'epoch 10. Pour b), cherche la dernière amélioration ; pour c), reprends le tableau : l'epoch 7 est-elle sous $0{,}38 - 0{,}005$ ? Pour d), même critère qu'en b) ; pour e), le compteur atteint-il 5 avant l'epoch 11 ? Pour f), même critère qu'en b) ; pour g), relis l'encadré « quels poids garde l'early stopping ? ».

</details>

### Ex 23.7 — Plannings de learning rate : le planning du livre avec LambdaLR

<details><summary>Indice 1</summary>

Relis le §23.10.2 de la fiche, sa figure, et la formule $\eta_e = \eta_0 \cdot \lambda(e)$.

</details>
<details><summary>Indice 2</summary>

`LambdaLR` multiplie : le learning rate est toujours $\eta_0$ fois ta fonction. Pour d), cherche $\lambda$ telle que $0{,}05 \cdot \lambda(e)$ redonne $\max(0{,}1 ;\ 1 - 0{,}01\,e)$. `StepLR` multiplie par $\gamma$ une fois toutes les `step_size` epochs : pendant l'epoch $e$, il l'a fait $\lfloor e / 30 \rfloor$ fois.

</details>
<details><summary>Indice 3</summary>

a) $1 \times \max(0{,}1 ;\ 1 - 0{,}25) = 0{,}75$. Pour b), résous $1 - 0{,}01\,e \le 0{,}1$ ; pour c), multiplie $\lambda(50)$ par $0{,}01$ ; pour d), divise le learning rate voulu à l'epoch 0 par $0{,}05$ ; pour e), calcule $0{,}1 \times 0{,}1^{\lfloor 65/30 \rfloor}$ ; pour f), $\cos(\pi/4) = \sqrt{2}/2$ ; pour g), compte les epochs de b) à 99.

</details>

### Ex 23.8 — Fermi : durée d'une epoch et poids d'un checkpoint

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « combien de calculs par epoch ? » du §23.10.3, et le §23.9.1 pour le contenu d'un checkpoint.

</details>
<details><summary>Indice 2</summary>

$6PN$ avec $P \approx 6{,}2 \times 10^5$ et $N = 6 \times 10^4$ donne quelques $10^{11}$ opérations. Un `float32` fait 4 octets ; Adam garde deux moments par paramètre, en plus des poids. Pour e), divise 3 600 secondes par la durée d'une epoch, et arrondis **en dessous**.

</details>
<details><summary>Indice 3</summary>

a) $6 \times 623\,290 \times 60\,000 = 2{,}24 \times 10^{11}$, soit 224,4 milliards. Pour b), divise a) par $5 \times 10^{10}$ ; pour c), multiplie $P$ par 4 octets ; pour d), trois tenseurs de la taille des poids ; pour e), utilise la durée non arrondie de b). Dans ta copie, pense au chargement des mini-batches, à l'évaluation de la validation et aux opérations qui ne sont pas des produits de matrices.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 23.9 — Pourquoi le prétraitement doit voyager avec le modèle

<details><summary>Indice 1</summary>

Relis les §23.5.8 et §23.9.5 de la fiche, et l'encadré 💼 du §23.5.8.

</details>
<details><summary>Indice 2</summary>

Cherche une situation où un réglage ne sert à rien sans la convention qui va avec : une mesure sans son unité, une clé sans la serrure, des instructions écrites pour une autre machine. Le modèle a appris sur des nombres préparés d'une certaine façon ; que reçoit-il si on lui donne des pixels bruts ?

</details>
<details><summary>Indice 3</summary>

Commence par : « Un modèle, c'est comme un traducteur qui n'a appris qu'une seule orthographe… » (ou une image à toi). Puis dis ce qui arrive avec les données brutes (il répond quand même, sans prévenir, et se trompe), d'où viennent les nombres du prétraitement (le train), et ce qu'il faut joindre aux poids (la fonction ou ses paramètres, et une fiche modèle).

</details>

### Ex 23.10 — Réutiliser un modèle pré-entraîné : licence, données d'origine, biais hérités

<details><summary>Indice 1</summary>

Relis le §23.9.4 de la fiche et son encadré 🕰️ ; cherche la page « Models and pre-trained weights » de la documentation de torchvision et la page de téléchargement d'ImageNet (image-net.org).

</details>
<details><summary>Indice 2</summary>

1. Distingue la licence du **code**, celle des **poids** et les conditions des **données** qui ont servi à les produire ; lis ce que torchvision dit de la responsabilité de l'utilisateur. 2. Cherche l'article de Yang, Qinami, Fei-Fei, Deng et Russakovsky (FAT* 2020) et la mise à jour d'ImageNet du 17 septembre 2019. 3. Pense au déplacement de domaine (ch. 12) : photos de catalogue, de rue, de pays différents. 5. Pense aux modèles dont les poids ont une licence claire, et à l'entraînement sur tes propres données.

</details>
<details><summary>Indice 3</summary>

Pour 1., la documentation de torchvision dit que les poids « peuvent avoir leur propre licence ou leurs propres conditions, issues du dataset d'entraînement » et que c'est à l'utilisateur de vérifier ; ImageNet réserve sa base à la recherche et à l'enseignement non commerciaux : conclus prudemment (zone grise, avis juridique). Pour 2., l'audit portait sur les catégories de **personnes** (catégories offensantes, sensibles, ou non représentables par une image) ; demande-toi si les features apprises dépendent de toutes les images vues. Pour 4., reprends les rubriques de la fiche modèle de Mitchell et coll.

</details>

### Ex 23.11 — Lire Krizhevsky et coll. (2012) : la recette d'entraînement d'AlexNet

<details><summary>Indice 1</summary>

L'article est en accès libre dans les actes de NeurIPS : [proceedings.neurips.cc, NIPS 2012, « ImageNet Classification with Deep Convolutional Neural Networks »](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) ; ouvre le lien « Paper » (le PDF). Lis d'abord le résumé et la section 5, qui fait une demi-page.

</details>
<details><summary>Indice 2</summary>

1. La fin de la section 1 et la section 3.2 parlent du matériel. 3. La section 5 donne quatre nombres : la taille des mini-batches, le momentum, le weight decay et le learning rate de départ. 4. Cherche « divided the learning rate by 10 » et le nombre de cycles sur le training set. 5. Les sous-sections de la section 4. 6. Le tableau 2 et sa légende.

</details>
<details><summary>Indice 3</summary>

Pour 3., la ligne a la forme `torch.optim.SGD(model.parameters(), lr=..., momentum=..., weight_decay=...)`, la taille des mini-batches allant dans le `DataLoader`. Pour 4., le planning de PyTorch qui divise le learning rate quand la validation ne progresse plus est `ReduceLROnPlateau` (avec `factor=0.1`). Pour 6., lis la légende du tableau 2 : combien de CNN sont moyennés ? Pour 7., pense à la batchnorm (2015), à Adam (2014), au cosinus et au warmup (ch. 19), et à l'initialisation de He (ch. 16).

</details>

<a id="entretien"></a>

## 💼 Entretien

### 23.E1 — Ton entraînement PyTorch est deux fois plus lent que prévu : où cherches-tu, et dans quel ordre ?

<details><summary>Indice 1</summary>

Relis les §23.3.5 et §23.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Mesure avant de deviner : chronomètre séparément le chargement des données et le calcul. Puis passe en revue, du plus fréquent au plus rare : le device, le `DataLoader`, les synchronisations cachées dans la boucle, la taille des mini-batches, le type des nombres.

</details>
<details><summary>Indice 3</summary>

Commence par : « D'abord, je mesure où passe le temps, parce que les deux grands suspects sont le chargement des données et le calcul… » Puis : le modèle et les mini-batches sont-ils sur le GPU ? Le `DataLoader` a-t-il assez de `num_workers` ? Y a-t-il un `.item()` ou un `print` à chaque pas qui force une synchronisation ? Puis le profiler (`torch.profiler`), et seulement ensuite les optimisations (mixed precision, `torch.compile`).

</details>

### 23.E2 — Pourquoi appeler model.eval() et torch.no_grad() (ou inference_mode) à l'inférence ?

<details><summary>Indice 1</summary>

Relis le §23.7 de la fiche, l'encadré « `model.eval()` ne coupe pas les gradients », et le §23.8.2.

</details>
<details><summary>Indice 2</summary>

Les deux gestes agissent sur deux choses différentes : le comportement de certaines couches, et l'enregistrement du graphe de calcul. Pour chacun, décris ce qui se passe quand on l'oublie : un résultat faux, ou un gaspillage.

</details>
<details><summary>Indice 3</summary>

Commence par : « Ce sont deux interrupteurs indépendants : `eval()` change ce que **calculent** le dropout et la batchnorm, `no_grad` change ce que PyTorch **garde en mémoire** pour la passe arrière… » Puis l'oubli de chacun, la différence entre `no_grad` et `inference_mode`, et le retour à `model.train()`.

</details>

### 23.E3 — Comment sauvegarder un modèle PyTorch pour reprendre l'entraînement plus tard ou le déployer ?

<details><summary>Indice 1</summary>

Relis les §23.9.1 à §23.9.5 et le §23.10.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Deux besoins, deux contenus. Pour reprendre : tout ce qui détermine la suite. Pour déployer : les poids, l'architecture (le code et ses arguments), le prétraitement, et un format sûr. Pense aussi à la sécurité du chargement.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je ne sauvegarde jamais l'objet modèle avec pickle ; je sauvegarde des `state_dict`, et le contenu dépend du but… » Puis le checkpoint complet (modèle, optimiseur, planning, epoch, générateurs), puis le paquet de déploiement (poids, arguments de l'architecture, prétraitement, versions, fiche modèle), et `weights_only=True` ou safetensors.

</details>

### 23.E4 — La loss de validation remonte alors que l'accuracy de validation progresse encore : que se passe-t-il ?

<details><summary>Indice 1</summary>

Relis le §23.8.3 de la fiche et le §23.10.3.

</details>
<details><summary>Indice 2</summary>

L'accuracy ne regarde que l'argmax ; la cross-entropy regarde la probabilité donnée à la bonne classe. Imagine un modèle qui corrige quelques erreurs, mais devient très sûr de lui sur les images qu'il rate.

</details>
<details><summary>Indice 3</summary>

Commence par : « C'est le début de l'overfitting : le modèle devient trop confiant… » Puis explique avec un exemple chiffré (une erreur à $p = 0{,}01$ pour la bonne classe coûte $-\ln 0{,}01 \approx 4{,}6$), dis ce que tu surveilles et ce que tu fais (early stopping sur la loss ou sur la mesure qui compte vraiment, régularisation, calibration).

</details>

### 23.E5 — Keras ou PyTorch : comment choisir un framework aujourd'hui ?

<details><summary>Indice 1</summary>

Relis les §23.1, §23.2 et l'encadré 🕰️ du §23.3.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Les critères : l'équipe et son code existant, l'écosystème (modèles pré-entraînés, recherche), le déploiement visé, le niveau d'abstraction souhaité. Et le fait que Keras 3 tourne sur PyTorch : le choix n'est plus exclusif.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je regarde d'abord ce que l'équipe utilise et où le modèle sera déployé… » Puis : PyTorch domine la recherche et les modèles publiés (Hugging Face) ; Keras offre une API haut niveau et, depuis la version 3, plusieurs backends dont PyTorch ; les concepts (tenseurs, autograd, boucle, sauvegarde) se transposent. Termine par ce que tu choisirais par défaut, et pourquoi.

</details>

<a id="notebook"></a>

## Notebook

Les indices des exercices 23.12 à 23.29 seront écrits à la session suivante, avec les parties A à D du notebook.
