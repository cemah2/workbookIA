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

a) B : un `FutureWarning` prévient qu'un comportement va changer, le code tourne encore. Pour b), demande-toi ce que fait Python d'une exception que personne ne rattrape ; pour c), compare les appels du livre aux encadrés 🕰️ ; pour d), l'option qui fige ce qui peut changer d'une année à l'autre ; pour e), demande-toi ce que tu perds en ne voyant plus aucun avertissement.

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

Le format d'images de PyTorch est `(N, C, H, W)`. Pour b), cherche le type par défaut des réels de PyTorch ; pour c) et d), relis dans le §23.3.3 ce que `torch.from_numpy` fait d'un tableau NumPy. Un `float32` tient sur 32 bits.

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

Fashion-MNIST a été conçu comme un **remplaçant direct** de MNIST : relis ce que ses auteurs entendent par là. Iris décrit chaque fleur par des longueurs et des largeurs de deux de ses parties.

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

a) B : des logits, puisque la loss applique sa propre softmax. Pour b), relis le type et la forme des labels dans les §23.5.5 et §23.5.7 ; pour c), demande-toi si une seconde softmax produit une valeur interdite ; pour d), compte les labels et lis `num_classes` ; pour e), applique la règle qui devine le nombre de classes à partir du plus grand label (les classes commencent à 0).

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

Pour reprendre « comme si de rien n'était », liste tout ce qui détermine la suite de l'entraînement : les poids, mais aussi ce que l'optimiseur a accumulé, l'endroit où l'on s'est arrêté et les tirages aléatoires à venir. Pour d), compte les tenseurs de chaque couche qui a des paramètres.

</details>
<details><summary>Indice 3</summary>

a) C : les poids, l'état de l'optimiseur, l'epoch et l'état des générateurs aléatoires. Pour b), regarde ce que rend `model.state_dict()` (des noms et des tenseurs) ; pour c), relis à quoi sert l'option `weights_only` ; pour d), compte deux tenseurs (poids et biais) par couche qui a des paramètres ; pour e), relis ce que vérifie `load_state_dict`.

</details>

### 23.Q12 — 🕰️ Que devient le code Keras du livre en 2026 ?

<details><summary>Indice 1</summary>

Relis les encadrés 🕰️ de la fiche, en particulier ceux du §23.3.3, du §23.6.3 et du §23.8.2.

</details>
<details><summary>Indice 2</summary>

Keras 3 (2023) est redevenu une bibliothèque à plusieurs backends, mais pas avec ceux de 2017. Pour c), relis ce que fait chacune des deux fonctions dans l'encadré 🕰️ du §23.6.3.

</details>
<details><summary>Indice 3</summary>

a) B : JAX, TensorFlow et PyTorch. Pour b), relis l'encadré du §23.8.2 ; pour c), compare ce que font les deux fonctions ; pour d), relis ce que l'encadré du §23.3.3 dit de leur développement ; pour e), relis l'encadré sur les formats de sauvegarde.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 23.R1 — Ch. 22 : (N, T, D) et batch_first

<details><summary>Indice 1</summary>

Relis, au ch. 22, le paragraphe sur la forme des entrées d'une couche récurrente et celui sur `output` et `h_n`.

</details>
<details><summary>Indice 2</summary>

Avec `batch_first=True`, l'entrée est `(N, T, D)` ; sans, `(T, N, D)`. Pour c) et d), relis dans la documentation de `nn.GRU` la forme de `h_n`, et ce que dit `batch_first` des états.

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

a) Remplis le tableau epoch par epoch (attention : 0,379 < 0,38) : le compteur atteint 3 pour la première fois à la fin de l'epoch 10. Pour b), relis la règle de rechargement ; pour c), reprends le tableau : l'epoch 7 est-elle sous $0{,}38 - 0{,}005$ ? Pour d), même critère qu'en b) ; pour e), le compteur atteint-il 5 avant l'epoch 11 ? Pour f), même critère qu'en b) ; pour g), relis l'encadré « quels poids garde l'early stopping ? ».

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

1. La fin de la section 1 et la section 3.2 parlent du matériel. 3. La section 5 donne quatre nombres : la taille des mini-batches, le momentum, le weight decay et le learning rate de départ. 4. Cherche « divide the learning rate by 10 » et le nombre de cycles sur le training set. 5. Les sous-sections de la section 4. 6. Le tableau 2 et sa légende.

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

## Exercices du notebook

### Ex 23.12 — Environnement et tenseurs : versions, device, reshape, view, permute 📦

<details><summary>Indice 1</summary>

Relis les §23.1.3, §23.3.3, §23.3.5 et §23.5.1 de la fiche. Chaque bibliothèque donne sa version dans un attribut `__version__` ; pour Python, cherche dans le module `platform`. Pour les avertissements, le module `warnings` sait les **enregistrer** au lieu de les afficher.

</details>
<details><summary>Indice 2</summary>

`pick_device_12` : deux tests, `torch.cuda.is_available()` puis `torch.backends.mps.is_available()`. `to_nchw_12` : vérifie `ndim`, puis `torch.from_numpy`, `permute` pour réordonner les axes, `.float()` pour le type, `.contiguous()` pour recopier dans le nouvel ordre. `caught_warnings_12` : `warnings.catch_warnings(record=True)` rend une liste d'objets dont l'attribut `category` est la classe de l'avertissement ; ajoute un filtre qui laisse passer **tous** les avertissements, même déjà vus.

</details>
<details><summary>Indice 3</summary>

```python
def to_nchw_12(images):
    images = np.asarray(images)
    # refuse an array that does not have 4 axes (ValueError)
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().contiguous()


def caught_warnings_12(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # call fn
    # return the class names: w.category.__name__ for each w
```

</details>

### Ex 23.13 — Charger et regarder Fashion-MNIST 📦

<details><summary>Indice 1</summary>

Relis les §23.5.2 et §23.5.3 de la fiche : le chargement, puis les contrôles de structure (forme, type, minimum, maximum, effectifs par classe) et l'affichage d'images avec leur label.

</details>
<details><summary>Indice 2</summary>

`CLASSES[label]` donne le nom d'une classe ; `np.bincount(y)` compte les labels. Pour la moyenne par classe, sélectionne les images d'une classe avec un masque booléen (`X[y == k]`), puis `.mean()` ; `np.argmin` et `np.argmax` donnent la classe extrême. Pour la grille, `plt.subplots(10, per_class)` rend un tableau d'axes ; `np.flatnonzero(labels == k)` donne les indices des images de la classe $k$.

</details>
<details><summary>Indice 3</summary>

```python
X_train_13, y_train_13 = wb.datasets.load_fashion_mnist("train", seed=None)
first_class_13 = CLASSES[y_train_13[0]]
means_13 = [X_train_13[y_train_13 == k].mean() for k in range(10)]
# darkest_13, brightest_13: the names of the classes of np.argmin(means_13) and np.argmax(means_13)


def show_grid_13(images, labels, per_class=8):
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 0.9, 9.5))
    # for each class k: the first per_class indices of np.flatnonzero(labels == k), one imshow per axis,
    # axes[k, 0].set_ylabel(CLASSES[k], rotation=0, ...) ; then plt.show()
```

</details>

### Ex 23.14 — Oublier la conversion en float ou la normalisation : erreur ou entraînement lent ? 🔮

<details><summary>Indice 1</summary>

Relis les §23.5.5 et §23.5.6 de la fiche, et, au ch. 19, comment Adam choisit la taille de ses pas. Les pixels bruts sont 255 fois plus grands que les pixels divisés par 255 : que deviennent les premiers logits, et donc la première loss ?

</details>
<details><summary>Indice 2</summary>

Pour a) et b) : Adam divise chaque gradient par une moyenne de sa propre taille (la racine de son second moment) ; ses pas restent donc de l'ordre du learning rate, quelle que soit l'échelle des gradients. Une loss devient `nan` quand un calcul déborde. Pour c), pense à la loss d'un modèle dont les logits valent des centaines au lieu de quelques unités, quand il se trompe. Pour d), demande-toi si un MLP entraîné avec Adam est sensible à un décalage de moyenne de ses entrées (0,29 en moyenne, au lieu de 0).

</details>
<details><summary>Indice 3</summary>

Pose chaque prédiction comme une comparaison : a) une accuracy après 3 epochs (pense à ce que fait un modèle quand ses pas restent raisonnables) ; b) un débordement, ou pas ; c) un rapport entre deux premières loss moyennes ; d) un écart d'accuracy, à comparer à 3 points. L'expérience imprime les trois listes de loss et d'accuracy : lis-les **après** avoir écrit tes quatre réponses.

</details>

### Ex 23.15 — Tout le prétraitement au même endroit : split stratifié, float32, normalisation, DataLoader 📦

<details><summary>Indice 1</summary>

Relis les §23.5.4 à §23.5.8 de la fiche, et ✏️ 23.5. `train_test_split` de scikit-learn fait le découpage stratifié ; `TensorDataset` et `DataLoader` font le reste (ch. 20).

</details>
<details><summary>Indice 2</summary>

`split_15` : `train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)` rend déjà les quatre tableaux dans l'ordre demandé. `fit_stats_15` : divise par 255, puis `.mean()` et `.std()`, convertis en `float`. `preprocess_15` : calcule en `float32` d'un bout à l'autre (convertis aussi `mean` et `std` en `np.float32`, ou en `float` Python), puis `torch.from_numpy(...)` et `.unsqueeze(1)` pour l'axe des canaux. `make_loaders_15` : un `TensorDataset` par partie, avec des labels `int64` ; le loader d'entraînement reçoit `shuffle=True` et `generator=torch.Generator().manual_seed(seed)`.

</details>
<details><summary>Indice 3</summary>

```python
def preprocess_15(images, mean, std):
    x = (np.asarray(images, dtype=np.float32) / 255.0 - np.float32(mean)) / np.float32(std)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1)


def make_loaders_15(X_tr, y_tr, X_val, y_val, mean, std, batch_size=256, seed=0):
    train_ds = TensorDataset(preprocess_15(X_tr, mean, std), torch.as_tensor(np.asarray(y_tr), dtype=torch.int64))
    # val_ds: the same with X_val, y_val
    # train_loader: shuffle=True, generator=torch.Generator().manual_seed(seed) ; val_loader: batch_size=1024
```

</details>

### Ex 23.16 — Le MLP du livre avec nn.Sequential et nn.Flatten 📦

<details><summary>Indice 1</summary>

Relis le §23.6.2 de la fiche et ✏️ 23.3. `nn.Sequential` applique ses couches dans l'ordre ; `nn.LazyLinear(n_out)` ne demande que la taille de **sortie**.

</details>
<details><summary>Indice 2</summary>

`make_mlp_16` : quatre couches dans un `nn.Sequential`, la dernière sans softmax. `make_lazy_mlp_16` : les mêmes, avec `nn.LazyLinear(n_hidden)` et `nn.LazyLinear(n_classes)`. `count_params_16` : additionne `p.numel()` sur `model.parameters()`.

</details>
<details><summary>Indice 3</summary>

```python
def make_mlp_16(n_hidden=784, n_classes=10):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, n_hidden), ...)    # then the ReLU and the last Linear


def count_params_16(model):
    return sum(p.numel() for p in model.parameters())
```

</details>

### Ex 23.17 — Trois bugs d'inférence : eval() oublié, gradients calculés pour rien (inference_mode), chargement non sûr (weights_only) 🐛

<details><summary>Indice 1</summary>

Relis l'encadré « `model.eval()` ne coupe pas les gradients » du §23.7, le §23.8.2 et l'encadré 🕰️ sur les formats de sauvegarde du §23.9.2. Le titre de l'exercice nomme les trois bugs.

</details>
<details><summary>Indice 2</summary>

Point 1 : `model_17` contient un dropout ; dans quel mode est-il quand `predict_17` l'appelle ? Point 2 : la sortie doit être calculée **dans** un bloc `with torch.inference_mode():`. Point 3 : `torch.load` a un argument qui n'accepte que des tenseurs et des types simples ; c'est même sa valeur par défaut depuis PyTorch 2.6.

</details>
<details><summary>Indice 3</summary>

```python
def predict_17(model, images):
    model.eval()
    with torch.inference_mode():
        logits = ...                      # the model on the preprocessed images
    return logits.argmax(dim=1)
```
Pour `load_weights_17`, une seule valeur change dans la ligne existante.

</details>

### Ex 23.18 — Ta fonction fit() : boucle, validation et historique 📦

<details><summary>Indice 1</summary>

Relis le §23.7 de la fiche : `train_one_epoch` et `evaluate` y sont presque écrits. Ta `fit_18` les réunit, ajoute l'accuracy d'entraînement, l'historique et les callbacks.

</details>
<details><summary>Indice 2</summary>

`evaluate_18` : `model.eval()`, un bloc `torch.inference_mode()` (ou le décorateur `@torch.inference_mode()`), puis, pour chaque mini-batch, ajoute `loss.item() * len(xb)` à un total, le nombre de bonnes réponses à un autre, `len(xb)` au compte ; divise à la fin. `fit_18` : un dictionnaire de quatre listes vides ; pour `epoch` de 1 à `epochs` : `model.train()`, la boucle des cinq lignes (23.R2) avec les mêmes totaux pondérés, puis `evaluate_18`, puis ajoute les quatre `float` à l'historique ; appelle **tous** les callbacks et arrête si l'un a rendu `True`.

</details>
<details><summary>Indice 3</summary>

```python
def fit_18(model, train_loader, val_loader, loss_fn, optimizer, epochs, callbacks=(), device="cpu"):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, count = 0.0, 0, 0
        for xb, yb in train_loader:
            # forward, loss, zero_grad, backward, step ; then the three weighted totals
        # evaluate_18 on val_loader ; append the four floats to history
        stop = [bool(cb(epoch, history, model, optimizer)) for cb in callbacks]
        # break if any(stop)
    return history
```

</details>

### Ex 23.19 — Lire l'historique : 3 epochs, puis 50 📈

<details><summary>Indice 1</summary>

Relis les §23.8.1 et §23.8.3 de la fiche, et le ch. 9 sur l'overfitting. Toutes les réponses se calculent sur `HISTORY_19` avec NumPy.

</details>
<details><summary>Indice 2</summary>

`np.argmin` et `np.argmax` rendent un **indice**, compté à partir de 0 : l'epoch est l'indice plus 1. Pour e), `np.array(HISTORY_19["train_acc"]) >= 0.95` est un tableau de booléens ; `np.argmax` d'un tableau de booléens donne le premier `True`.

</details>
<details><summary>Indice 3</summary>

a) `int(np.argmin(HISTORY_19["val_loss"])) + 1`. Pour b), le même calcul sur l'accuracy de validation, avec le maximum ; pour c), un quotient de deux valeurs de `val_loss` ; pour d), une différence de deux valeurs de `val_acc`, multipliée par 100 ; pour e), le premier `True` du tableau de booléens, plus 1.

</details>

### Ex 23.20 — Analyser les erreurs : matrice de confusion et grille des confusions 📦

<details><summary>Indice 1</summary>

Relis le §23.8.1 de la fiche et, au ch. 3, ta `confusion_matrix` (lignes : classe réelle, colonnes : classe prédite).

</details>
<details><summary>Indice 2</summary>

`predict_all_20` : comme `evaluate_18`, mais garde les labels et les `argmax` de chaque mini-batch dans deux listes, puis `torch.cat` et `.numpy()`. `top_confusions_20` : liste les cases $(i, j)$ avec $i \neq j$, trie par la clé `(-effectif, i, j)`, garde les $k$ premières. `show_confusions_20` : un masque `(y_true == true_class) & (y_pred == pred_class)`, `np.flatnonzero`, puis une ligne de `plt.subplots(1, n)`.

</details>
<details><summary>Indice 3</summary>

```python
def top_confusions_20(C, k=3):
    C = np.asarray(C)
    cells = [(int(C[i, j]), i, j) for i in range(C.shape[0]) for j in range(C.shape[1]) if i != j]
    cells.sort(key=lambda cell: (-cell[0], cell[1], cell[2]))
    # return the first k as (true_class, predicted_class, count) tuples
```

</details>

### Ex 23.21 — Prédire la réaction du MLP à des images « du monde réel » 🔮

<details><summary>Indice 1</summary>

Relis le §23.6.1 de la fiche : une couche dense reçoit les 784 pixels **aplatis**, chacun avec ses propres poids. Que sait-elle du voisinage des pixels ?

</details>
<details><summary>Indice 2</summary>

Pour a), pense à ce que voit le premier neurone caché : chaque poids regarde **une** position ; après un décalage, chaque pixel arrive sur les poids appris pour son voisin. Pour b) et c), le modèle n'a jamais vu de fond clair : la softmax rend-elle des probabilités prudentes quand l'entrée ne ressemble à rien de connu, ou n'importe quelles probabilités ? Pour d), regarde de près des images de chaussures et de vestes : sont-elles vraiment symétriques ?

</details>
<details><summary>Indice 3</summary>

Pose chaque prédiction comme une comparaison : a) une accuracy, à placer dans l'une des quatre tranches ; b) une accuracy face au hasard (10 % pour 10 classes équilibrées) ; c) une confiance moyenne face à 0,8 ; d) une accuracy face à 80 %. Écris tes quatre réponses avant d'exécuter l'expérience.

</details>

### Ex 23.22 — Prédire sur de nouvelles images : eval, inference_mode, probabilités 📦

<details><summary>Indice 1</summary>

Relis les §23.8.2 et §23.9.5 de la fiche. La softmax transforme les logits en probabilités ; le module `json` écrit et relit un dictionnaire.

</details>
<details><summary>Indice 2</summary>

`predict_proba_22` : si l'image a 2 axes, ajoute l'axe des exemples (`images[None]`) ; puis `model.eval()`, un bloc `torch.inference_mode()`, `preprocess(images, mean, std)`, le modèle, `.softmax(dim=1)`, `.cpu().numpy()`. `top_k_22` : `np.argsort(-probs, axis=1)[:, :k]` donne les indices des $k$ plus grandes valeurs de chaque ligne. `save_preprocessing_22` : construis le dictionnaire avec `float(mean)`, `float(std)` et `list(classes)`, puis `json.dumps` et `Path(path).write_text`.

</details>
<details><summary>Indice 3</summary>

```python
def predict_proba_22(model, images, mean=MEAN, std=STD, device="cpu"):
    images = np.asarray(images)
    # one image (28, 28): images = images[None]
    model.eval()
    with torch.inference_mode():
        logits = model(preprocess(images, mean, std).to(device))
    return logits.softmax(dim=1).cpu().numpy()
```
Pour `top_k_22`, chaque ligne devient `[(CLASSES[j], float(row[j])) for j in idx]`.

</details>

### Ex 23.23 — Reprendre un entraînement interrompu : checkpoint complet (modèle, optimiseur, epoch, générateurs aléatoires) 📦

<details><summary>Indice 1</summary>

Relis le §23.9.1 de la fiche (le dictionnaire `checkpoint`) et la section « Au-delà du livre : la reproductibilité ». Tout ce qui tire au hasard pendant l'entraînement doit être sauvé.

</details>
<details><summary>Indice 2</summary>

`save_checkpoint_23` : un dictionnaire avec `model.state_dict()`, `optimizer.state_dict()`, `int(epoch)`, `torch.get_rng_state()`, `generator.get_state()` et `dict(config)`, puis `torch.save`. `resume_23` : `torch.load(path, weights_only=True)`, `make_mlp(**config)`, `load_state_dict` pour le modèle, un `Adam` **sur les paramètres du nouveau modèle** puis son `load_state_dict`, un `torch.Generator()` puis `set_state`, et `torch.set_rng_state` pour le générateur global.

</details>
<details><summary>Indice 3</summary>

```python
def resume_23(path, lr=1e-3):
    checkpoint = torch.load(path, weights_only=True)
    model = make_mlp(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer.load_state_dict ; a new torch.Generator() and set_state ; torch.set_rng_state
    return model, optimizer, generator, checkpoint["epoch"]
```

</details>

### Ex 23.24 — Un modèle pré-entraîné torchvision : transforms, gel et nouvelle tête 📦

<details><summary>Indice 1</summary>

Relis le §23.9.4 de la fiche et, au ch. 21, comment on charge un ResNet-18 et ses transformations. Geler un paramètre, c'est lui retirer son gradient : `requires_grad_(False)`.

</details>
<details><summary>Indice 2</summary>

`make_transfer_24` : `resnet18(weights=weights)`, une boucle `requires_grad_(False)` sur **tous** les paramètres, **puis** une nouvelle `nn.Linear(model.fc.in_features, n_classes)` (créée après le gel, elle reste entraînable). `to_resnet_input_24` : `torch.from_numpy(images / 255)`, `.unsqueeze(1)` puis `.expand(-1, 3, -1, -1)` pour les 3 canaux, `v2.Resize((size, size), antialias=True)`, puis `v2.Normalize(t.mean, t.std)` avec `t = weights.transforms()`. `extract_features_24` : remplace provisoirement `model.fc` par `nn.Identity()` (ou empile les couches sauf la dernière), puis remets la tête.

</details>
<details><summary>Indice 3</summary>

```python
def extract_features_24(model, x, batch_size=250):
    head, model.fc = model.fc, nn.Identity()           # everything but the head
    model.eval()
    try:
        with torch.inference_mode():
            # torch.cat of model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)
    finally:
        model.fc = head
```

</details>

### Ex 23.25 — Les callbacks en PyTorch : meilleur checkpoint, planning de LR, early stopping 📦

<details><summary>Indice 1</summary>

Relis les §23.10.1 à §23.10.3 de la fiche, et la docstring de ton `EarlyStopping` du ch. 20. Un objet qui a une méthode `__call__` s'appelle comme une fonction et garde son état dans ses attributs.

</details>
<details><summary>Indice 2</summary>

`BestCheckpoint_25` : dans `__init__`, range `path`, `monitor` et `mode`, et initialise `best_` à `math.inf` ou `-math.inf`, `best_epoch_` à `None` ; dans `__call__`, compare `history[monitor][-1]` à `best_` avec `<` ou `>` (strict), et sauve si c'est mieux. `SchedulerCallback_25` : lis `optimizer.param_groups[0]["lr"]` **avant** de faire avancer le planning ; teste `isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)`. `EarlyStoppingCallback_25` : crée `mylearn.nn.regularization.EarlyStopping(patience=..., min_delta=...)` dans `__init__`, et rends le résultat de son `step`.

</details>
<details><summary>Indice 3</summary>

```python
class SchedulerCallback_25:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, epoch, history, model, optimizer):
        history.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])
        # ReduceLROnPlateau: self.scheduler.step(history["val_loss"][-1]) ; otherwise: self.scheduler.step()
        return False
```

</details>

### Ex 23.26 — Planning de learning rate : constant, par paliers, cosinus ou plateau 🔬

<details><summary>Indice 1</summary>

Relis le §23.10.2 de la fiche et ✏️ 23.7. Les quatre plannings sont dans `torch.optim.lr_scheduler`.

</details>
<details><summary>Indice 2</summary>

Une suite de `if name == ...:` qui rendent chacun un planning construit sur `optimizer`, puis un `raise ValueError` à la fin. Le planning constant est un `LambdaLR` dont la fonction rend toujours 1.

</details>
<details><summary>Indice 3</summary>

```python
def make_scheduler_26(name, optimizer, epochs):
    schedules = torch.optim.lr_scheduler
    if name == "constant":
        return schedules.LambdaLR(optimizer, lambda e: 1.0)
    if name == "step":
        return schedules.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    # "cosine" and "plateau" ; then raise ValueError(...)
```

</details>

### Ex 23.27 — Patience de l'early stopping : 1, 3 ou 10 🔬

<details><summary>Indice 1</summary>

Relis le §23.10.3 de la fiche, ✏️ 23.6 et la docstring de ton `EarlyStopping` du ch. 20, en particulier ce qu'est `best_epoch_`.

</details>
<details><summary>Indice 2</summary>

Crée un `EarlyStopping(patience=patience, min_delta=min_delta)` ; parcours les loss avec `enumerate(val_losses, start=1)` ; à chaque loss, `step` ; dès qu'il rend `True`, rends l'epoch et la meilleure epoch. Si la boucle finit sans arrêt, rends `None` et la meilleure epoch. `best_epoch_` est un indice compté à partir de 0.

</details>
<details><summary>Indice 3</summary>

```python
def simulate_27(val_losses, patience, min_delta=0.0):
    stopper = mylearn.nn.regularization.EarlyStopping(patience=patience, min_delta=min_delta)
    for epoch, loss in enumerate(val_losses, start=1):
        if stopper.step(loss):
            return epoch, ...                 # best_epoch_ is 0-based
    return None, ...
```

</details>

### Ex 23.28 — Rédiger la fiche du modèle (model card) 🛠️

<details><summary>Indice 1</summary>

Relis le §23.9.5 de la fiche et la fiche modèle que tu as écrite au MP3. L'article de Mitchell et coll. (2019) donne, pour chacune des neuf sections, les questions auxquelles elle répond.

</details>
<details><summary>Indice 2</summary>

Une section par entrée de `REQUIRED_28`, dans l'ordre, chacune avec deux à quatre phrases. Les chiffres viennent de `CARD_FACTS_28` (une f-string les insère : `{MEAN:.4f}`, `{VAL_ACC_C:.4f}`). Pour « Facteurs » et « Mises en garde », sers-toi de ce que tu as mesuré en 23.20 et 23.21 ; pour « Analyses quantitatives », tu peux calculer l'accuracy par classe de `MODEL_C`.

</details>
<details><summary>Indice 3</summary>

```python
MODEL_CARD_28 = f"""# Fiche modèle : MLP Fashion-MNIST

## Détails du modèle
{CARD_FACTS_28["architecture"]}. Entraînement : {CARD_FACTS_28["training"]}.

## Usage prévu
...
"""   # the nine sections, with MEAN and STD (4 decimals), the validation accuracy, the dataset and its licence
```

</details>

### Ex 23.29 — Défi reproductibilité : un MLP Fashion-MNIST à au moins 88 % dont l'accuracy varie de moins de 0,5 point sur 5 graines, en moins de 3 min de CPU 🏆

<details><summary>Indice 1</summary>

Relis la section « Au-delà du livre : la reproductibilité » de la fiche, et regarde l'effet des plannings en 23.26. Deux choses à obtenir : une bonne accuracy, et des résultats qui ne dépendent presque pas de la graine. Mesure d'abord le point de départ (le MLP du livre, Adam constant) sur la validation.

</details>
<details><summary>Indice 2</summary>

L'écart entre graines vient surtout de la fin de l'entraînement : avec un learning rate constant, les poids oscillent encore, et l'accuracy finale dépend de l'endroit où l'oscillation s'arrête. Un planning qui finit à un learning rate très faible (cosinus, `OneCycleLR` appelé à chaque mini-batch) pose chaque modèle au fond de sa vallée. Un label smoothing (`nn.CrossEntropyLoss(label_smoothing=0.1)`) limite la sur-confiance et stabilise encore. Pour le budget : quelques epochs suffisent, et un réseau un peu plus étroit que celui du livre va plus vite.

</details>
<details><summary>Indice 3</summary>

```python
def train_29(seed, x, y, epochs=8, batch_size=256):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), ...)   # two hidden layers, then 10 logits
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=epochs * len(loader))
    # the training loop: schedule.step() after EACH optimizer.step()
```

</details>
