# 24 · PyTorch en pratique 2 : améliorer, chercher, CNN et RNN — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 24.Q1 — Compter les hyperparamètres et n'en changer qu'un

<details><summary>Indice 1</summary>

Relis les §24.2.1 à §24.2.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un hyperparamètre se **choisit** avant l'entraînement ; un paramètre (un poids, un biais, l'état d'un optimiseur) est **calculé** par l'entraînement. Pour b), chaque hyperparamètre multiplie le nombre de combinaisons par son nombre de valeurs. Pour d), pense à ce que veut dire « interagir ».

</details>
<details><summary>Indice 3</summary>

a) La taille de batch (B), le learning rate (C) et le taux de dropout (E) sont choisis avant l'entraînement ; les poids, les biais et l'état d'Adam sont calculés par lui : `"BCE"`. Pour b), calcule $3^6$ ; pour c), demande-toi si garder la valeur par défaut est un choix ; pour d), cherche un cas où le meilleur réglage de A dépend de B ; pour e), relis le conseil du §24.2.3 sur les objectifs.

</details>

### 24.Q2 — Taille de batch : bruit, overfitting et temps

<details><summary>Indice 1</summary>

Relis le §24.2.2 de la fiche et son encadré 🧮 sur le bruit d'une moyenne.

</details>
<details><summary>Indice 2</summary>

Une epoch fait $\lceil N/B \rceil$ mises à jour. Le bruit du gradient d'un mini-batch a un écart-type en $\sigma/\sqrt{B}$ : compare les deux racines. Pour d), demande-toi combien de pas, et de quelle taille, chaque réglage fait en une epoch.

</details>
<details><summary>Indice 3</summary>

a) $50\,000 / 64 = 781{,}25$, arrondi au-dessus : 782 mises à jour (le dernier mini-batch, incomplet, compte aussi). Pour b), le même calcul avec 256 ; pour c), le rapport $\sqrt{400}/\sqrt{16}$ ; pour d), compare le nombre de pas par epoch et leur bruit dans les deux cas ; pour e), relis ce que montre la fig. 24.5 du livre.

</details>

### 24.Q3 — Plus de neurones, moins de neurones, dropout

<details><summary>Indice 1</summary>

Relis les §24.2.4 à §24.2.6 de la fiche, et l'encadré « Le livre, corrigé » sur le masque.

</details>
<details><summary>Indice 2</summary>

Un réseau plus capable apprend plus vite les détails du train. Vérifie dans la fiche si le `p` de `nn.Dropout` est la probabilité de garder ou d'éteindre une valeur. Le dropout inversé corrige l'échelle pendant l'entraînement : demande-toi ce qu'il lui reste à faire en évaluation.

</details>
<details><summary>Indice 3</summary>

a) Le livre annonce que le modèle sur-apprendra plus vite, et l'observe : (D). Pour b), lis le sens de `p` dans la documentation ou la fiche ; pour c), relis à quel moment un nouveau masque est tiré ; pour d), calcule $2{,}0/(1 - 0{,}2)$ ; pour e), relis ce que fait le module en mode évaluation dans un dropout **inversé**.

</details>

### 24.Q4 — Valider un réseau : coût, jeu non représentatif, normalisation dans chaque fold

<details><summary>Indice 1</summary>

Relis les §24.3 à §24.3.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Chaque fold est un entraînement complet. Un jeu de validation doit ressembler aux données qu'on veut évaluer. Pour c), demande-toi ce que le scaler apprend, et sur quelles lignes il a le droit de l'apprendre. Pour d), rappelle-toi ce que fait scikit-learn d'un `cv` entier quand l'estimateur est un classifieur (ch. 15).

</details>
<details><summary>Indice 3</summary>

a) 5 folds, donc 5 entraînements de 40 minutes : $5 \times 40 = 200$ minutes. Pour b), regarde de quel hôpital viennent les 2 000 dernières radios ; pour c), demande-toi ce que verrait le scaler s'il était ajusté avant le découpage ; pour d), cherche le découpage qu'utilise `cross_val_score` pour un classifieur ; pour e), relis les scores du livre aux §24.3.2 et §24.3.3 de la fiche, et son explication.

</details>

### 24.Q5 — 🕰️ KerasClassifier, grille, hasard ou zoom : chercher des hyperparamètres

<details><summary>Indice 1</summary>

Relis les §24.3.1 et §24.3.4 de la fiche et leurs encadrés 🕰️.

</details>
<details><summary>Indice 2</summary>

Une grille essaie le produit des nombres de valeurs, et chaque candidat est entraîné une fois par fold ; `refit=True` ajoute un entraînement final sur toutes les données.

</details>
<details><summary>Indice 3</summary>

a) `KerasClassifier` a été retiré de Keras et n'a jamais enveloppé un `nn.Module` ; `MLPClassifier` est un autre réseau, celui de scikit-learn ; skorch, ou un estimateur écrit soi-même, enveloppe un réseau PyTorch : (C). Pour b), relis l'encadré 🕰️ sur la recherche ; pour c), multiplie $2 \times 3 \times 2$ par le nombre de folds ; pour d), ajoute l'entraînement final ; pour e), relis l'image de la bibliothèque.

</details>

### 24.Q6 — 🕰️ Couches utilitaires de Keras et leurs équivalents PyTorch

<details><summary>Indice 1</summary>

Relis le tableau du §24.4.1 de la fiche et l'encadré « Le livre, corrigé » sur la batchnorm.

</details>
<details><summary>Indice 2</summary>

Demande-toi, pour chaque couche, si elle agrandit l'image, la rétrécit en retirant des bords, l'entoure de zéros ou la résume par blocs.

</details>
<details><summary>Indice 3</summary>

a) `UpSampling2D` répète chaque valeur pour agrandir l'image : c'est (C), `nn.Upsample(scale_factor=2)`. Pour b), cherche ce qui retire une ligne et une colonne de chaque bord ; pour c), ce qui en ajoute ; pour d), ce qui remplace chaque bloc $2 \times 2$ par sa moyenne ; pour e), relis quand ce bruit est actif ; pour f), relis ce que normalise la batchnorm, d'après l'encadré.

</details>

### 24.Q7 — Préparer des images pour nn.Conv2d

<details><summary>Indice 1</summary>

Relis les §24.4.2 et §24.4.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

PyTorch attend $(N, C, H, W)$. Passer de NHWC à NCHW déplace un axe, sans réécrire les valeurs dans un autre ordre de lecture. `padding="same"` garde la taille ; relis au §24.4.3 avec quels strides PyTorch l'accepte.

</details>
<details><summary>Indice 3</summary>

a) 16 images, 3 canaux, 32 lignes, 32 colonnes : `"(16, 3, 32, 32)"`. Pour b), écris où va chaque axe de `(N, H, W, C)` (et méfie-toi d'une option qui échange $H$ et $W$) ; pour c), seul le nombre de canaux change ; pour d), relis la première des trois différences avec Keras au §24.4.3 ; pour e), calcule $51/127{,}5 - 1$ ; pour f), relis la deuxième différence.

</details>

### 24.Q8 — Augmentation et données synthétiques : quelles transformations sont plausibles ?

<details><summary>Indice 1</summary>

Relis les §24.4.6 et §24.4.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

L'augmentation fabrique des exemples d'**entraînement** ; la validation et le test mesurent ce que verra le modèle. Demande-toi à quel moment le `Dataset` tire sa transformation, et ce qui arrête un générateur Python.

</details>
<details><summary>Indice 3</summary>

a) La validation et le test doivent ressembler aux données réelles : l'augmentation est réservée au train, (B). Pour b), demande-toi quand le `Dataset` tire ses transformations ; pour c), relis la règle de plausibilité ; pour d), demande-toi quand un générateur s'arrête ; pour e), relis ce qu'est une epoch pour un `IterableDataset` ; pour f), demande-toi si le modèle revoit deux fois le même exemple.

</details>

### 24.Q9 — Motifs conv-conv-pool, stride contre pooling, batchnorm contre dropout

<details><summary>Indice 1</summary>

Relis les §24.4.4 et §24.4.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

La batchnorm normalise chaque canal ; regarde ce qu'elle fait de la moyenne. Les MAC d'une convolution valent $H_{out} W_{out} C_{out} C_{in} k^2$ : regarde ce que deviennent chacun des facteurs.

</details>
<details><summary>Indice 3</summary>

a) Le listing 24.44 place la batchnorm juste après la convolution, écrite sans activation, puis une couche ReLU : (C). Pour b), demande-toi ce que devient un biais constant après la soustraction de la moyenne ; pour c), écris le rapport $\frac{(H/2)(W/2)(3C)(3C)}{H\,W\,C\,C}$ ; pour d), relis le tableau du §24.4.4 (accuracy et temps par epoch) ; pour e), relis la fin du §24.4.5.

</details>

### 24.Q10 — Séries pour un RNN : fenêtres, scaler, inverse_transform, RMSE

<details><summary>Indice 1</summary>

Relis les §24.5.2 à §24.5.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le scaler apprend un minimum et un maximum : sur quelles données a-t-on le droit d'apprendre ? Un LSTM `batch_first` attend $(N, T, D)$. Pour c), demande-toi dans quelle unité un utilisateur veut lire l'erreur.

</details>
<details><summary>Indice 3</summary>

a) Le scaler apprend un minimum et un maximum : comme tout prétraitement, il ne doit voir que le train, (D). Pour b), place 500, 12 et le nombre de variables dans $(N, T, D)$ ; pour c), demande-toi dans quelle unité une RMSE se lit ; pour d), demande-toi ce que ferait un scaler réajusté sur le test ; pour e), relis pourquoi le livre ne met pas d'activation ; pour f), relis la fin du §24.5.4.

</details>

### 24.Q11 — 🕰️ return_sequences, stateful, TimeDistributed : que devient chaque option en PyTorch ?

<details><summary>Indice 1</summary>

Relis les §24.5.8 à §24.5.10 de la fiche et la figure des sorties de `nn.LSTM`.

</details>
<details><summary>Indice 2</summary>

`output` rassemble la sortie de la **dernière couche à chaque pas** ; `h_n` rassemble l'état de **chaque couche au dernier pas**. Pour c), regarde sur quelle dimension agit `nn.Linear` ; pour d), demande-toi quelle ligne du batch suivant continue la même séquence.

</details>
<details><summary>Indice 3</summary>

a) `return_sequences=True` rend la sortie de chaque pas, de forme $(N, T, H)$ : c'est (D), `output`. Pour b), cherche ce qui donne la sortie du dernier pas de la **dernière** couche ; pour c), relis ce que fait `nn.Linear` sur un tenseur à trois dimensions ; pour d), relis l'encadré « Le livre, corrigé » du §24.5.9 ; pour e), demande-toi ce que devient le graphe du batch précédent après son `backward()` ; pour f), relis le début du §24.5.9.

</details>

### 24.Q12 — 🕰️ API fonctionnelle de Keras 3 et forward() de PyTorch : branches, partage, gel

<details><summary>Indice 1</summary>

Relis le §24.6 de la fiche, son tableau et l'encadré sur la batchnorm gelée.

</details>
<details><summary>Indice 2</summary>

Un module est un objet qui porte ses poids : l'appeler deux fois réutilise les mêmes. `torch.cat` met des tenseurs bout à bout le long d'un axe.

</details>
<details><summary>Indice 3</summary>

a) `self.fc` est un seul objet, avec un seul jeu de poids : les deux appels utilisent les mêmes poids, (C). Pour b), cherche l'opération qui additionne deux tenseurs valeur par valeur ; pour c), ajoute les tailles de l'axe 1 ; pour d), relis ce que `requires_grad_(False)` n'arrête pas ; pour e), relis l'encadré « Le livre, corrigé » du §24.6.2 ; pour f), relis l'encadré 🕰️ sur l'API fonctionnelle.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 24.R1 — Ch. 23 : ce qu'un historique d'entraînement doit contenir pour comparer deux runs

<details><summary>Indice 1</summary>

Relis le §23.7 de la fiche du ch. 23 et l'encadré 💼 sur le journal d'expériences (§24.2.7).

</details>
<details><summary>Indice 2</summary>

Un historique sert à lire les courbes et à choisir une epoch, sans jamais toucher au test. Pour comparer deux runs, il faut pouvoir dire ce qui les distingue. Pour c), rappelle-toi comment `train_one_epoch` calcule sa loss.

</details>
<details><summary>Indice 3</summary>

a) Les losses d'entraînement et de validation, l'accuracy de validation et le learning rate de chaque epoch servent à lire les courbes ; une copie des poids à chaque epoch coûterait trop de mémoire (on garde le meilleur checkpoint), et le test ne se regarde qu'une fois, à la fin : `"ABD"`. Pour b), garde ce qui distingue deux runs ou permet de les refaire ; pour c), relis comment la loss d'une epoch est moyennée pendant que les poids changent ; pour d), cherche la plus petite des six valeurs et compte sa position à partir de 1.

</details>

### 24.R2 — Ch. 21 : taille de sortie d'une convolution à padding 'same' et stride 2

<details><summary>Indice 1</summary>

Relis le §24.4.3 de la fiche (le padding) et la formule de taille du ch. 21.

</details>
<details><summary>Indice 2</summary>

Keras « same » : $\lceil H/s \rceil$. PyTorch : $\lfloor (H + 2p - k)/s \rfloor + 1$. Pour reproduire le « same » de Keras avec un noyau impair $k$, on prend $p = \lfloor k/2 \rfloor$.

</details>
<details><summary>Indice 3</summary>

a) $\lceil 28/2 \rceil = 14$. Pour b), $\lceil 15/2 \rceil$ ; pour c), $\lfloor 5/2 \rfloor$ ; pour d), applique la formule de PyTorch avec ce padding à $H = 32$ ; pour e), la même formule avec $k = 3$, $p = 0$, $H = 15$ ; pour f), relis ce que PyTorch fait d'un noyau de taille paire avec `padding="same"`.

</details>

### 24.R3 — Ch. 17 : ce que change la division des logits avant le softmax

<details><summary>Indice 1</summary>

Relis l'encadré sur la température du ch. 22 et la fin du §24.5.11 de la fiche.

</details>
<details><summary>Indice 2</summary>

$\text{softmax}(z/T)_i = e^{z_i/T} / \sum_j e^{z_j/T}$. Divise d'abord les logits par $T$, puis applique la softmax.

</details>
<details><summary>Indice 3</summary>

a) Pour $T = 1$ : $e^2/(e^2 + e^1 + e^0) = 7{,}389/11{,}107 \approx 0{,}665$. Pour b), les logits deviennent $(4, 2, 0)$ ; pour c), $(1 ;\ 0{,}5 ;\ 0)$ ; pour d), regarde comment évolue l'écart entre les logits divisés quand $T$ diminue ; pour e), demande-toi si une division par un nombre positif change l'ordre des logits.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 24.1 — Le coût d'une grille (27 hyperparamètres, 54 entraînements) contre une recherche aléatoire : probabilité de tomber dans les 5 % meilleurs réglages en n essais ✏️

<details><summary>Indice 1</summary>

Relis le §24.3.4 de la fiche : le coût d'une grille et la probabilité d'un tirage au hasard.

</details>
<details><summary>Indice 2</summary>

Une grille coûte le produit des nombres de valeurs, fois le nombre de folds. Un tirage rate les 5 % meilleurs avec la probabilité 0,95 ; $n$ tirages indépendants les ratent tous avec la probabilité $0{,}95^n$.

</details>
<details><summary>Indice 3</summary>

a) $2^{27} = 134\,217\,728$ combinaisons. Pour b), divise ce nombre de secondes par $3\,600 \times 24 \times 365$ ; pour c), $3 \times (3 \times 3 \times 2)$ ; pour d), multiplie par 70 s, puis convertis en minutes ; pour e), calcule $1 - 0{,}95^{20}$ ; pour f), cherche le plus petit $n$ tel que $0{,}95^n \le 0{,}05$ (prends les logarithmes) ; pour g), refais e) avec $n = 18$.

</details>

### Ex 24.2 — Formes et paramètres du CNN à pooling du livre ✏️

<details><summary>Indice 1</summary>

Relis le §24.4.3 de la fiche (paramètres d'une convolution) et la formule de taille du ch. 21.

</details>
<details><summary>Indice 2</summary>

Avec `padding=2` pour un noyau 5 et `padding=1` pour un noyau 3, une convolution de stride 1 garde $H$ et $W$ ; un `MaxPool2d(2)` les divise par 2. Une convolution a $C_{out}(C_{in} k^2 + 1)$ paramètres, une couche dense $(n_{in} + 1)\,n_{out}$ ; dropout, ReLU, pooling et `Flatten` n'en ont aucun.

</details>
<details><summary>Indice 3</summary>

a) 256 images, 30 filtres, la taille gardée : `"(256, 30, 28, 28)"`. Pour b), divise $H$ et $W$ par 2 ; pour c), applique $C_{out}(C_{in} k^2 + 1)$ avec $C_{in} = 1$, $k = 5$ ; pour d), avec $C_{in} = 30$, $k = 3$, $C_{out} = 16$ ; pour e), multiplie les trois dimensions de la sortie du second pooling ; pour f), $(n + 1) \times 128$ ; pour g), additionne les cinq couches qui ont des paramètres (deux convolutions, trois couches denses) ; pour h), divise la somme des trois couches denses par g).

</details>

### Ex 24.3 — Stride ou pooling : compter les multiplications ✏️

<details><summary>Indice 1</summary>

Relis la formule des MAC au §24.4.3 de la fiche, et reprends les formes de l'Ex 24.2.

</details>
<details><summary>Indice 2</summary>

$H_{out}$ et $W_{out}$ sont ceux de la sortie de **la convolution elle-même**, avant le pooling. Avec un stride de 2 et le padding indiqué, chaque convolution divise la taille par 2 : la première sort en $14 \times 14$, la seconde en $7 \times 7$.

</details>
<details><summary>Indice 3</summary>

a) $28 \times 28 \times 30 \times 1 \times 5 \times 5 = 588\,000$. Pour b), la seconde convolution travaille sur des cartes $14 \times 14$, avec $C_{in} = 30$, $C_{out} = 16$, $k = 3$ ; pour c), $784 \times 128 + 128 \times 64 + 64 \times 10$ ; pour d), refais a) et b) avec des sorties $14 \times 14$ puis $7 \times 7$, et additionne ; pour e), ajoute c) à chaque version, puis divise ; pour f), compte les valeurs de sortie du pooling ($30 \times 14 \times 14$), fois 3 ; pour g), $147/36$.

</details>

### Ex 24.4 — Ce que renvoie nn.LSTM : output, h_n et c_n ✏️

<details><summary>Indice 1</summary>

Relis le §24.5.8 de la fiche et la figure des sorties de `nn.LSTM`.

</details>
<details><summary>Indice 2</summary>

`output` : $(N, T, n_{dir} H)$ ; `h_n` et `c_n` : $(n_{couches}\, n_{dir}, N, H)$, quel que soit `batch_first`. Une couche LSTM de PyTorch a $4H(D_{in} + H)$ poids et $2 \times 4H$ biais, où $D_{in}$ est la taille de son entrée ($D$ pour la première couche, $H$ pour les suivantes).

</details>
<details><summary>Indice 3</summary>

a) $N = 8$, $T = 5$, $H = 4$ : `"(8, 5, 4)"`. Pour b), 2 couches, un sens ; pour c), écris la forme de `h_n[0]`, puis celle de chaque option ; pour d) et e), double ce que dit la formule pour deux sens ; pour f), la première couche a $16 \times (3 + 4) + 32$ paramètres, calcule la seconde avec $D_{in} = 4$ et additionne ; pour g), `nn.Linear` remplace la dernière dimension ; pour h), sans `batch_first`, l'axe du temps passe en premier.

</details>

### Ex 24.5 — Fenêtres, pas d'optimisation et durée : de 200 à 20 000 points ✏️

<details><summary>Indice 1</summary>

Relis les §24.5.2 et §24.5.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une série de $T$ points donne $T - w$ fenêtres (la dernière fenêtre doit laisser une valeur pour sa cible). Le nombre de pas par epoch est $\lceil N/B \rceil$. Pour les durées, convertis tout en secondes.

</details>
<details><summary>Indice 3</summary>

a) $200 - 13 = 187$ exemples. Pour b), le même calcul avec 20 000 ; pour c), un pas par exemple ; pour d), divise b) par 32 et arrondis au-dessus ; pour e), $100 \times 5$ s, en minutes ; pour f), $28 \times 3\,600$ s divisé par 100 ; pour g), $5 \times$ le rapport des nombres d'exemples a) et b).

</details>

### Ex 24.6 — Dropout inversé et max-norm : espérance et projection ∂

<details><summary>Indice 1</summary>

Relis le §24.2.6 de la fiche, et le rappel sur l'espérance et la variance d'une variable de Bernoulli (0B, ch. 2).

</details>
<details><summary>Indice 2</summary>

$m$ ne prend que deux valeurs : $\mathbb{E}[m] = 1 - p$ et $\mathbb{E}[m^2] = 1 - p$. Utilise $\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$. Pour 4., énumère les quatre masques possibles et leurs probabilités. Pour 5., si $\lVert\mathbf{w}\rVert > c$, un point $\mathbf{v}$ de la boule vérifie $\lVert\mathbf{w}\rVert \le \lVert\mathbf{w} - \mathbf{v}\rVert + \lVert\mathbf{v}\rVert$.

</details>
<details><summary>Indice 3</summary>

1. $\mathbb{E}[\tilde{h}] = \frac{h}{1 - p}\,\mathbb{E}[m] = \frac{h}{1-p}(1 - p) = h$. Pour 2., calcule $\mathbb{E}[\tilde{h}^2] = \frac{h^2}{(1-p)^2}\mathbb{E}[m^2]$ et retranche $h^2$ ; pour 3., calcule l'espérance de $m\,h$ ; pour 4., la linéarité de l'espérance donne la première partie, puis chacun des quatre masques a la probabilité $1/4$ : calcule $z$ dans chaque cas, puis la moyenne des $\text{ReLU}(z)$ ; pour 5., minore $\lVert\mathbf{w} - \mathbf{v}\rVert$ par $\lVert\mathbf{w}\rVert - c$, puis calcule $\lVert\mathbf{w} - \mathbf{w}'\rVert$ ; pour 6., $\lVert\mathbf{w}\rVert = \sqrt{9 + 16 + 144}$, puis applique la formule ; pour la convolution, demande-toi quels poids servent à calculer une même valeur de sortie, et sur quel indice de `weight[o, c, u, v]` ils se repèrent.

</details>

### Ex 24.7 — Fermi : le budget d'une recherche d'hyperparamètres sur CPU, sur GPU et en précision mixte 🧮

<details><summary>Indice 1</summary>

Relis le §24.4.8 de la fiche, la partie « précision mixte et compilation », et l'Ex 24.1.

</details>
<details><summary>Indice 2</summary>

Opérations d'un entraînement = MAC par image × 2 × 3 × nombre d'images × nombre d'epochs. Durée = opérations / débit. Une grille coûte le produit des nombres de valeurs, fois le nombre de folds. Convertis les secondes au bon moment.

</details>
<details><summary>Indice 3</summary>

a) $25 \times 10^6 \times 2 \times 3 \times 50\,000 \times 20 = 1{,}5 \times 10^{14}$, soit 150 milliers de milliards. Pour b), divise a) par $5 \times 10^{10}$ et convertis en minutes ; pour c), $4 \times 3 \times 3$ fois le nombre de folds ; pour d), multiplie b) par c) et convertis en heures ; pour e), divise a) par $10^{13}$, multiplie par c), convertis ; pour f), divise e) par 2,5 et convertis en minutes ; pour g), 30 entraînements de la durée b).

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 24.8 — La data augmentation en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis le §24.4.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

Cherche une situation de la vie courante où l'on apprend à reconnaître quelque chose en le voyant sous plusieurs angles, éclairages ou distances, sans que ce soit une autre chose. Puis dis ce qu'on ne fait pas (changer la nature de la chose, ou s'entraîner avec des vues impossibles).

</details>
<details><summary>Indice 3</summary>

Commence par : « Pour apprendre à reconnaître un chat, il faut l'avoir vu de près, de loin, à gauche, à droite… » Puis : on fabrique ces vues à partir des photos qu'on a, à chaque passage différemment ; le label reste le même ; on ne transforme que les photos d'entraînement ; on évite ce qui n'arrive jamais (un chat la tête en bas, un « 6 » retourné qui devient un « 9 »).

</details>

### Ex 24.9 — Une seule graine suffit-elle ? Reproductibilité et honnêteté des comparaisons d'hyperparamètres ⚖️

<details><summary>Indice 1</summary>

Relis le §24.2.7 de la fiche (le journal d'expériences), le §24.3.4, et la partie « reproductibilité » de la fiche du ch. 23.

</details>
<details><summary>Indice 2</summary>

Un écart entre deux moyennes ne veut dire quelque chose que s'il est grand devant leur dispersion. Le maximum de plusieurs mesures bruitées est, en moyenne, plus haut que la vraie valeur de celle qui gagne. Un test qui a servi à choisir n'est plus un test.

</details>
<details><summary>Indice 3</summary>

1. La moyenne de A vaut $434{,}5/5 = 86{,}90$ ; calcule de même celle de B, puis les écarts-types avec $n - 1$, et compare l'écart des moyennes aux écarts-types. Pour 2., pense à tout ce qui est tiré au hasard entre le chargement des données et la fin de l'entraînement ; pour 3., pense au plus haut de 60 lancers d'un dé un peu bruité ; pour 4., demande-toi ce qu'il reste pour estimer la performance sans biais ; pour 5., pense à ce qu'un lecteur ne peut pas deviner (le nombre de runs, les heures de GPU) ; pour 6., reprends les réponses précédentes.

</details>

### Ex 24.10 — Lire He et coll. (2016) : les réseaux résiduels 📄

<details><summary>Indice 1</summary>

Relis les §24.4.5 et §24.6.2 de la fiche, puis lis le résumé et la figure 1 de l'article avant le reste.

</details>
<details><summary>Indice 2</summary>

Pour 2., compare les courbes d'**entraînement** des deux réseaux : l'overfitting, c'est une erreur d'entraînement basse et une erreur de test haute. Pour 4., cherche les équations (1) et (2) de la section 3.2, et les options (A) et (B) de la section 3.3. Pour 5., le tableau de la section 4.2 donne les tailles des cartes et le nombre de filtres de chaque groupe de couches.

</details>
<details><summary>Indice 3</summary>

1. Le problème annoncé dès la première phrase : les réseaux plus profonds sont plus difficiles à entraîner ; cherche ensuite la profondeur des réseaux testés sur ImageNet et la place obtenue à ILSVRC 2015. Pour 2., si le réseau de 56 couches a aussi une erreur d'entraînement plus haute, ce n'est pas un problème de généralisation ; pour 3., demande-toi ce qu'il faut faire aux poids d'une pile de couches non linéaires pour qu'elle rende exactement son entrée ; pour 4., l'option (A) ajoute des zéros, l'option (B) une projection ; pour 5., compte $6 \times 3 + 2$ ; pour 6., pense au chemin direct du gradient.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 24.E1 — Comment organises-tu une recherche d'hyperparamètres avec un budget limité ?

<details><summary>Indice 1</summary>

Relis les §24.2, §24.3.4 et §24.4.8 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pense en étapes : d'abord une référence qui marche, puis l'ordre des hyperparamètres (le learning rate d'abord), puis la méthode (hasard plutôt que grille, échelle logarithmique), puis les économies (sous-ensemble, peu d'epochs, élimination précoce), et enfin la vérification (plusieurs graines, test une seule fois).

</details>
<details><summary>Indice 3</summary>

Commence par : « D'abord, je chiffre le budget : combien dure un entraînement, donc combien d'essais tiennent dans deux jours. » Puis : une référence simple ; le learning rate en premier, sur une échelle logarithmique ; une recherche au hasard sur un sous-ensemble et peu d'epochs ; l'élimination des mauvais candidats (successive halving) ; un zoom autour des meilleurs ; plusieurs graines pour le finaliste ; un journal de chaque run ; le test à la toute fin.

</details>

### 24.E2 — Pourquoi la data augmentation améliore-t-elle la généralisation ? Donne des transformations à éviter

<details><summary>Indice 1</summary>

Relis le §24.4.6 de la fiche et son encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

Deux idées : l'augmentation dit au modèle ce qui **ne doit pas** changer sa réponse (une invariance), et elle multiplie les exemples différents qu'il voit, ce qui le freine dans l'apprentissage par cœur. Puis les pièges : un label qui change, une image irréaliste, une augmentation appliquée au test.

</details>
<details><summary>Indice 3</summary>

Commence par : « L'augmentation apprend au modèle ses invariances : une photo de voiture décalée ou en miroir reste une voiture… » Puis : chaque epoch voit des versions différentes, d'où moins d'overfitting ; à éviter : le miroir sur du texte ou des chiffres, le haut et le bas inversés sur des photos, des couleurs changées quand la couleur est l'information (feux de signalisation, fruits mûrs) ; jamais sur la validation ou le test ; à régler comme un hyperparamètre ; RandAugment, mixup.

</details>

### 24.E3 — Dropout ou batchnorm : lequel, où, et peut-on combiner les deux ?

<details><summary>Indice 1</summary>

Relis les §24.2.6, §24.4.1 et §24.4.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Distingue le but premier de chacun (régulariser ; stabiliser et accélérer l'entraînement), leur place habituelle (le dropout dans les couches denses ou avant la tête ; la batchnorm entre convolution et activation), et ce qui se passe quand on les met l'un après l'autre (les statistiques de la batchnorm apprises avec un dropout actif ne sont plus celles de l'évaluation).

</details>
<details><summary>Indice 3</summary>

Commence par : « Ils ne font pas la même chose : la batchnorm stabilise l'entraînement, le dropout est un régularisateur… » Puis : dans un CNN moderne, la batchnorm après chaque convolution, du dropout surtout dans la tête ; on peut combiner, mais pas un dropout juste avant une batchnorm ; les deux changent de comportement entre `train()` et `eval()` ; on choisit en mesurant l'écart entre train et validation.

</details>

### 24.E4 — Comment éviter la fuite de données dans une validation croisée avec prétraitement ?

<details><summary>Indice 1</summary>

Relis le §24.3.3 de la fiche, et le ch. 8 sur la fuite de données.

</details>
<details><summary>Indice 2</summary>

Fais la liste de tout ce qui **apprend** quelque chose sur les données dans le pipeline : la normalisation, mais aussi la sélection de features, l'imputation, le vocabulaire, le rééquilibrage des classes. Puis pense aux fuites qui ne viennent pas du prétraitement : des doublons, des groupes (un même patient), le temps.

</details>
<details><summary>Indice 3</summary>

Commence par : « Le signe classique : un score de validation croisée bien au-dessus du test. Je cherche d'abord tout ce qui a été ajusté sur l'ensemble des données avant le découpage… » Puis : tout dans un `Pipeline`, que la validation croisée réajuste dans chaque fold ; la sélection de features aussi ; `GroupKFold` pour des groupes, `TimeSeriesSplit` pour le temps ; les doublons ; le test gardé à part jusqu'à la fin.

</details>

### 24.E5 — Batchnorm : que fait-elle, et pourquoi se comporte-t-elle différemment en train et en eval ?

<details><summary>Indice 1</summary>

Relis l'encadré « Le livre, corrigé » du §24.4.1 de la fiche, l'encadré sur la batchnorm gelée du §24.6, et le ch. 20.

</details>
<details><summary>Indice 2</summary>

En entraînement, elle normalise avec la moyenne et la variance **du mini-batch**, et met à jour des moyennes glissantes ; en évaluation, elle utilise ces moyennes glissantes. Demande-toi pourquoi on ne peut pas utiliser les statistiques du mini-batch à l'inférence, et ce qui arrive si l'on oublie `model.eval()`, ou si les mini-batches sont minuscules.

</details>
<details><summary>Indice 3</summary>

Commence par : « Pour chaque canal, elle centre et réduit les activations avec la moyenne et l'écart-type du mini-batch, puis applique un facteur $\gamma$ et un décalage $\beta$ appris… » Puis : en évaluation, les statistiques glissantes, pour qu'une prédiction ne dépende pas des autres images du batch ; bugs : `model.eval()` oublié, des batches de taille 1 ou 2, des statistiques glissantes qui bougent sur une partie « gelée », une distribution de test différente ; alternatives : layer norm et group norm.

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex 24.11 — Changer un seul hyperparamètre : la taille de batch 🔬

<details><summary>Indice 1</summary>

Relis le §24.2.2 de la fiche (les trois mécanismes : le bruit, le nombre de pas, le temps) et la cellule d'outils : `make_mlp`, `make_loader` et `train_classifier` font déjà tout le travail ; `run_11` les enchaîne dans le bon ordre.

</details>
<details><summary>Indice 2</summary>

`run_11` : la graine d'abord (`torch.manual_seed(seed)`), puis le modèle (`make_mlp((784,))`), le loader (`make_loader(XA_TR, YA_TR, batch_size, seed)`) et `train_classifier(model, loader, XA_VAL, YA_VAL, epochs, lr=lr, seed=seed)`. `summary_11` : `np.argmin` sur les losses de validation donne un **indice**, compté à partir de 0 ; la dernière epoch est l'indice `-1` ; `np.median` pour la durée. Convertis en `int` et `float` Python.

</details>
<details><summary>Indice 3</summary>

```python
def summary_11(history):
    val = np.asarray(history["val_loss"])
    best = int(np.argmin(val))                       # an index: the epoch is best + 1
    return {"best_epoch": best + 1, "best_val_loss": float(val[best]),
            # "final_gap": val_loss - train_loss at index -1 ; "final_val_acc" ; "sec_per_epoch": np.median(...)
            }
```
Pour les notes : compte les pas d'une epoch, $\lceil 6\,000 / B \rceil$, pour chaque taille de batch.

</details>

### Ex 24.12 — Prédire le classement de quatre MLP (784, 784 + 784, 64, 32 + 32) 🔮

<details><summary>Indice 1</summary>

Relis les §24.2.4 et §24.2.5 de la fiche : ce que le livre a vu sur MNIST, et pourquoi. Ici, Fashion-MNIST est plus difficile, et l'entraînement ne dure que 15 epochs sur 6 000 images : qu'est-ce qu'un modèle de 600 000 paramètres, ou de 26 000, peut faire de 6 000 images en 15 epochs ?

</details>
<details><summary>Indice 2</summary>

Pour a) et d) : à quelle vitesse un modèle très large apprend-il 6 000 images, et que fait sa loss de validation une fois qu'il les connaît ? Pour b) et c) : l'accuracy de validation dépend à la fois de la capacité (assez de neurones pour séparer les classes) et de l'overfitting ; compte les paramètres de A et de C, puis, entre deux modèles de 64 neurones en tout, l'un en une couche, l'autre en deux, lequel a le plus de paramètres ?

</details>
<details><summary>Indice 3</summary>

Pose chaque prédiction comme une comparaison : a) quatre epochs, celles du minimum de la loss de validation de chaque modèle ; b) et c) deux accuracies de validation à la 15ᵉ epoch ; d) le rapport entre la loss de validation finale de B et sa plus petite valeur, à comparer à 1,1. Le tableau de l'expérience donne toutes ces colonnes : lis-le **après** avoir écrit tes quatre réponses.

</details>

### Ex 24.13 — Dropout, max-norm et learning rate trop grand 📦

<details><summary>Indice 1</summary>

Relis le §24.2.6 de la fiche (la formule de la max-norm, le code qui l'applique après `optimizer.step()`) et le ∂ 24.6 (pourquoi `renorm` sur la dimension 0 traite un filtre entier d'une convolution). Garde toutes les couches `nn.Linear` et `nn.Conv2d` dans l'ordre de `model.modules()`, puis retire la dernière.

</details>
<details><summary>Indice 2</summary>

`max_norm_13` : sous `torch.no_grad()`, pour chaque couche, `layer.weight.flatten(1).norm(dim=1)` donne une norme par ligne ou par filtre ; compte celles qui dépassent `c` ; puis `layer.weight.copy_(torch.renorm(layer.weight, p=2, dim=0, maxnorm=c))` modifie le poids sur place, sans le remplacer. `run_13` : comme `run_11`, avec `make_mlp((32, 32), p_drop=p_drop)` et `after_step=lambda m: max_norm_13(m, c)` quand `c` n'est pas `None`.

</details>
<details><summary>Indice 3</summary>

```python
def max_norm_13(model, c=3.0):
    layers = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))][:-1]   # not the output layer
    clipped = 0
    with torch.no_grad():
        for layer in layers:
            norms = layer.weight.flatten(1).norm(dim=1)
            # count the norms above c, then copy_ the renormed weight into layer.weight
    return clipped
```

</details>

### Ex 24.14 — Pourquoi la validation dépasse le train avec dropout 📈

<details><summary>Indice 1</summary>

Relis le §24.2.6 de la fiche (ce que fait le dropout en entraînement et en évaluation) et 24.R1 (ce qu'est une accuracy « moyenne pendant l'epoch »). Les quatre courbes de `HISTORY_14` séparent les deux effets : chaque différence entre deux d'entre elles n'en mesure qu'un.

</details>
<details><summary>Indice 2</summary>

Convertis les listes en tableaux NumPy : les différences se calculent alors d'un coup. L'epoch 1 est l'indice 0 ; les epochs 21 à 30 sont la tranche `[20:30]`. Multiplie par 100 pour passer en points, et arrondis seulement à la fin. a) compte les epochs où une condition est vraie : `np.sum` d'un tableau de booléens.

</details>
<details><summary>Indice 3</summary>

```python
run_14 = np.asarray(HISTORY_14["running_acc"])
val_14 = np.asarray(HISTORY_14["val_acc"])
above_14 = int(np.sum(val_14 > run_14))
```
Les autres réponses sont des différences de deux courbes, multipliées par 100 : b) fin d'epoch en mode entraînement moins moyenne pendant l'epoch, à l'indice 0 ; c) la même différence, en moyenne (`np.mean`) sur la tranche `[20:30]` ; d) mode évaluation moins mode entraînement, en fin d'epoch ; e) mode évaluation moins validation.

</details>

### Ex 24.15 — Journal d'expériences : un run, une ligne 🛠️

<details><summary>Indice 1</summary>

Relis l'encadré « En entreprise » du §24.2.7. Le format *JSON lines* : chaque ligne est un objet JSON complet, écrit par `json.dumps` puis un `"\n"` ; le fichier s'ouvre en mode `"a"` (*append*), qui ajoute à la fin sans effacer.

</details>
<details><summary>Indice 2</summary>

`git_commit_15` : `subprocess.run([...], cwd=ROOT, capture_output=True, text=True)`, puis `returncode` et `stdout.strip()` ; un `try` pour le cas où git n'est pas installé (`OSError`). `log_run_15` : `datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")` pour l'heure ; `json.dumps(run, default=...)` appelle la fonction `default` pour chaque objet qu'il ne sait pas écrire (un nombre NumPy : `.item()`) ; un tuple s'écrit déjà comme une liste. Rends `json.loads(line)` : ce qu'on relira. `runs_table_15` : `pd.json_normalize(runs, sep=".")` aplatit les dictionnaires imbriqués en colonnes `params.lr`. `best_run_15` : `max` ou `min` avec une `key`.

</details>
<details><summary>Indice 3</summary>

```python
def log_run_15(path, params, metrics, seed):
    run = {...}                                                         # the seven fields of the contract
    line = json.dumps(run, default=_plain_15, ensure_ascii=False)     # _plain_15: what json cannot write -> plain Python
    # append the line and a "\n" to the file (mode "a"), then return what json.loads gives back from the line
```

</details>

### Ex 24.16 — Un réseau PyTorch déguisé en estimateur scikit-learn 📦

<details><summary>Indice 1</summary>

Relis le §24.3.1 de la fiche (le contrat d'un estimateur) et, au ch. 15, ce que font `clone` et `get_params`. `fit` reprend ta boucle du ch. 23 (Ex 23.18) ; la seule nouveauté est le passage entre des labels quelconques et les entiers de 0 à $K - 1$ qu'attend la cross-entropy.

</details>
<details><summary>Indice 2</summary>

`np.unique(y, return_inverse=True)` rend les classes triées **et** le code de chaque label (son indice dans `classes_`) : la cross-entropy apprend sur les codes, `predict` revient aux labels par `self.classes_[indices]`. Dans `fit` : `torch.manual_seed(self.seed)` avant de construire le réseau, puis `make_loader(..., self.batch_size, self.seed)`, Adam, et une boucle sur `self.epochs` ; range ce qu'il faut dans des attributs finis par `_` et `return self`. `predict_proba` : vérifie le nombre de colonnes, `model_.eval()`, `torch.inference_mode()`, `torch.softmax(logits, dim=1).numpy()`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    X = np.asarray(X, dtype=np.float32)
    self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
    self.n_features_in_ = X.shape[1]
    torch.manual_seed(self.seed)
    self.model_ = ...            # Linear + ReLU (+ Dropout) per size of self.hidden, then Linear(.., len(self.classes_))
    # loader of (X, codes as int64), Adam(self.lr), self.epochs epochs; self.history_ = mean loss per epoch
    return self


def predict(self, X):
    # the index of the most probable column of predict_proba, then back to the labels through self.classes_
```

</details>

### Ex 24.17 — Validation croisée stratifiée avec normalisation dans un Pipeline 📦

<details><summary>Indice 1</summary>

Relis les §24.3.2 et §24.3.3 de la fiche, et, au ch. 8, ce que rend `cv.split(X, y)` : des couples d'indices `(train, validation)`, un par fold.

</details>
<details><summary>Indice 2</summary>

`fold_class_counts_17` : code les labels avec `np.unique(..., return_inverse=True)`, puis, pour chaque fold, `np.bincount(codes[val], minlength=K)` compte chaque classe de la partie validation ; empile les lignes en un tableau. `make_pipe_17` : un `Pipeline` de deux étapes nommées. `max_after_scaling_17` : `clone(scaler)` donne une copie neuve, qu'on ajuste sur `X_fit` puis qui transforme `X_new` ; `np.max` et `float`.

</details>
<details><summary>Indice 3</summary>

```python
def fold_class_counts_17(cv, X, y):
    classes, codes = np.unique(np.asarray(y), return_inverse=True)
    rows = []
    for _, val in cv.split(X, y):
        # count codes[val] for each class (np.bincount with minlength), append the row
    # return the rows as one array
```

</details>

### Ex 24.18 — Grille, hasard, puis zoom : chercher les hyperparamètres 📦

<details><summary>Indice 1</summary>

Relis le §24.3.4 de la fiche (le coût d'une recherche, `cv_results_`, `RandomizedSearchCV` et les distributions de `scipy.stats`) et ✏️ 24.1. Les clés des réglages d'une étape de pipeline s'écrivent `net__lr`, avec deux tirets bas.

</details>
<details><summary>Indice 2</summary>

`count_fits_18` : `math.prod` des longueurs des listes de la grille, ou `n_iter`, fois `cv`, plus 1 si `refit`. `grid_search_18` : `GridSearchCV(pipe, grille, cv=StratifiedKFold(3, shuffle=True, random_state=0)).fit(X, y)`. `random_search_18` : un dictionnaire de distributions et de listes, `RandomizedSearchCV(pipe, espace, n_iter=n_iter, cv=..., random_state=seed).fit(X, y)` ; `uniform(0, 0.5)` va de 0 à 0,5. `zoom_values_18` : les exposants vont de `-(n // 2)` à `n // 2`. `results_18` : `pd.DataFrame(search.cv_results_)`, quatre colonnes, `sort_values("rank_test_score")`.

</details>
<details><summary>Indice 3</summary>

```python
def count_fits_18(cv, param_grid=None, n_iter=None, refit=True):
    candidates = n_iter if param_grid is None else math.prod(len(v) for v in param_grid.values())
    # cv trainings per candidate, and one more when refit


def zoom_values_18(best, factor=2.0, n=3):
    half = n // 2
    # one value best * factor ** k for each k from -half to half
```

</details>

### Ex 24.19 — Un CNN traduit du Keras qui refuse de tourner 🐛

<details><summary>Indice 1</summary>

Relis les §24.4.1 à §24.4.3 de la fiche : les trois différences avec Keras (le padding, l'activation, la forme de l'entrée), puis le ✏️ 24.2 pour suivre une taille couche par couche. Lis les messages de la vérification dans l'ordre : chaque exception corrigée en révèle une autre.

</details>
<details><summary>Indice 2</summary>

Les erreurs bruyantes : un réglage de padding que PyTorch refuse avec un stride de 2 ; une taille d'entrée de couche dense mal calculée (fais le calcul $32 \to 30 \to 15 \to\ ?$) ; une soustraction qui aligne `MEAN` sur le mauvais axe (le broadcasting aligne les axes **par la fin**). Les erreurs silencieuses : compare la liste des couches à la docstring (activation par activation), regarde la dernière couche quand la loss est `nn.CrossEntropyLoss`, et demande-toi ce que fait `reshape` à des pixels rangés en NHWC.

</details>
<details><summary>Indice 3</summary>

```python
def prepare_19(images):
    x = torch.from_numpy(images).float() / 255
    x = ...                                           # NHWC -> NCHW: which method reorders the axes?
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)        # one value per channel, on axis 1
    # the same for std, then standardise
```
Dans `make_cnn_19`, quatre lignes à corriger ou à ajouter : relis chaque ligne de la docstring Keras à côté de la couche PyTorch qui la traduit, et calcule à la main la taille de l'entrée de la couche dense.

</details>

### Ex 24.20 — Du CNN simple au CNN profond sur CIFAR-10 📦

<details><summary>Indice 1</summary>

Relis les §24.4.3 à §24.4.5 de la fiche (les formules de taille et de paramètres, le motif conv-conv-pool, le biais inutile avant une batchnorm) et le ✏️ 24.2.

</details>
<details><summary>Indice 2</summary>

`make_deep_cnn_20` : une liste `layers` qu'on remplit étage par étage (deux convolutions, chacune suivie de sa batchnorm éventuelle et de sa ReLU, puis le pooling et le dropout éventuel), en gardant à jour le nombre de canaux d'entrée et la taille de l'image (divisée par 2 à chaque pooling) ; `bias=not batchnorm` ; puis `nn.Sequential(*layers)`. `trace_shapes_20` : itérer sur un `nn.Sequential` donne ses couches dans l'ordre ; mets le modèle en évaluation, et `torch.inference_mode()`.

</details>
<details><summary>Indice 3</summary>

```python
def make_deep_cnn_20(batchnorm=False, p_drop=0.0, widths=(16, 32, 64)):
    layers, c_in, side = [], 3, 32
    for width in widths:
        for c in (c_in, width):
            layers.append(nn.Conv2d(c, width, 3, padding=1, bias=not batchnorm))
            # the BatchNorm2d if batchnorm, then the ReLU
        # the MaxPool2d(2), the Dropout if p_drop > 0 ; then update c_in and side
    layers += [nn.Flatten(), nn.Linear(c_in * side * side, 10)]
    return nn.Sequential(*layers)
```

</details>

### Ex 24.21 — Batchnorm, dropout ou les deux : accuracy et écart entre train et validation d'un CNN 🔬

<details><summary>Indice 1</summary>

Relis les §24.4.4 et §24.4.5 de la fiche, et l'encadré « Le livre, corrigé — ce que fait la batchnorm » du §24.4.1 : à quoi sert-elle d'abord ? `run_21` ressemble beaucoup à `run_11`.

</details>
<details><summary>Indice 2</summary>

`run_21` : graine, `make_deep_cnn_20(**VARIANTS_21[variant])`, `make_loader(XC_TR_T, YC_TR_T, 128, seed)`, `train_classifier`. `gaps_21` : une liste en compréhension sur `zip(history["train_acc"], history["val_acc"])`. Pour les questions sur `HISTORY_21` : le dictionnaire `letters_21 = dict(zip("ABCD", ["plain", "dropout", "batchnorm", "both"]))` relie les lettres aux variantes ; `max` et `min` avec une `key` trouvent la meilleure.

</details>
<details><summary>Indice 3</summary>

```python
letters_21 = dict(zip("ABCD", ["plain", "dropout", "batchnorm", "both"]))
best_21 = max(letters_21, key=lambda k: HISTORY_21[letters_21[k]]["val_acc"][-1])
```
b) le même schéma avec `min` sur le dernier écart de `gaps_21` ; c) ce dernier écart pour `"plain"`, arrondi à 1 décimale ; d) pour chaque variante, l'indice de la première epoch où `val_acc >= 0.65`, puis la lettre du plus petit.

</details>

### Ex 24.22 — Prédire l'accuracy d'un CNN sur des images retournées, décalées, tournées ou en gris 🔮

<details><summary>Indice 1</summary>

Relis le §24.4.6 de la fiche (la règle de plausibilité) et 24.Q8. Un modèle entraîné sans augmentation ne connaît que ce que montrent les photos de son train : demande-toi, pour chaque transformation, si le train contenait déjà des images qui lui ressemblent.

</details>
<details><summary>Indice 2</summary>

Pour a) : dans des photos d'animaux et de véhicules, combien regardent vers la gauche, combien vers la droite ? Pour b) et e) : le ciel est-il parfois en bas dans CIFAR-10, et une photo couchée sur le côté ? Pour c) : que voit le réseau dans la bande noire, et a-t-il vu, à l'entraînement, des images bordées de noir ou des objets décalés ? Pour d) : la couleur aide-t-elle à reconnaître une grenouille, un avion dans le ciel, un bateau sur la mer, et est-ce autant que l'orientation ?

</details>
<details><summary>Indice 3</summary>

Pose chaque prédiction comme une comparaison entre deux lignes du tableau de l'expérience : la ligne de la transformation et la ligne « aucune », la différence étant à comparer au seuil de la question ; pour d) et e), deux baisses entre elles. Écris tes cinq réponses avant de lancer la cellule.

</details>

### Ex 24.23 — Augmentation de données avec torchvision.transforms.v2 📦

<details><summary>Indice 1</summary>

Relis les §24.4.2 et §24.4.6 de la fiche (les transformations `to_input` et `train_tf`, et les trois règles), et, au ch. 23, ce que doivent avoir un `Dataset` (`__len__`, `__getitem__`) et un `DataLoader` reproductible.

</details>
<details><summary>Indice 2</summary>

`make_transforms_23` : construis la liste de fin (`ToDtype`, `Normalize`) une fois, la liste des opérations aléatoires seulement si `augment`, et assemble deux `v2.Compose`. `CifarImages_23` : `__init__` range les trois arguments ; `__getitem__` applique la transformation **à chaque appel**. `channel_stats_23` : divise par 255 en `float64`, puis `mean` et `std` sur les axes `(0, 1, 2)` (tout sauf le canal). `make_train_loader_23` : `generator=torch.Generator().manual_seed(seed)`. Pour les questions sur `HISTORY_23` : un parcours des epochs, et `all` sur la fin d'une liste de booléens.

</details>
<details><summary>Indice 3</summary>

```python
def make_transforms_23(augment=True):
    finish = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    random_ops = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()] if augment else []
    # return two v2.Compose: ToImage, random_ops, finish ; and ToImage, finish


with_23, without_23 = HISTORY_23["with"], HISTORY_23["without"]
gain_23 = round(100 * (max(with_23["val_acc"]) - max(without_23["val_acc"])), 1)
```
b) `above_23 = [w > wo for w, wo in zip(with_23["val_acc"], without_23["val_acc"])]`, puis la première epoch `i + 1` telle que `all(above_23[i:])` ; c) une différence d'accuracies d'entraînement à l'indice `-1`, en points.

</details>

### Ex 24.24 — Des images synthétiques générées à la volée 📦

<details><summary>Indice 1</summary>

Relis le §24.4.7 de la fiche (le `ShapeStream`, ses deux pièges) et, en 0A, les générateurs. Décris chaque forme comme une ou plusieurs **lignes brisées** (des listes de points) en coordonnées unitaires ; le dessin relie les points consécutifs par `stroke_24`.

</details>
<details><summary>Indice 2</summary>

`draw_shape_24` : tire le label s'il manque (`rng.integers(5)`), le facteur (`rng.uniform(0.8, 1.1)`) et le décalage (`rng.uniform(-0.1, 0.1, size=2)`) ; pour chaque ligne brisée, ajoute le bruit (`rng.normal(0, 0.03, size=(n, 2))`), applique $(p - 0{,}5) \times$ facteur $+ 0{,}5 +$ décalage, multiplie par `size - 1`, puis trace chaque segment. Le cercle : 25 points sur `np.linspace(0, 2π, 25)`, le dernier égal au premier après le bruit. `worker_share_24` : une division entière et un reste. `ShapeStream_24.__iter__` : la graine de `np.random.default_rng` vient de `torch.randint(2**31, ())`, et `get_worker_info()` dit si l'on est dans un worker (`info.id`, `info.num_workers`).

</details>
<details><summary>Indice 3</summary>

```python
_LINES_24 = {0: [[(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)]],    # Z: top bar, diagonal, bottom bar
             1: [[(0.5, 0.2), (0.5, 0.8)], [(0.2, 0.5), (0.8, 0.5)]],  # plus: two lines
             # 2: three vertical bars ; 3: square U ; 4: circle of 25 points
             }


def worker_share_24(n, worker_id, num_workers):
    base, extra = divmod(n, num_workers)
    # the first `extra` workers take one more
```

</details>

### Ex 24.25 — Sommes de sinus et LSTM minuscule : fenêtres, scaler, RMSE 📦

<details><summary>Indice 1</summary>

Relis les §24.5.2 à §24.5.4 de la fiche : l'ordre des trois étapes de préparation, et la règle des colonnes (`reshape(-1, 1)` en entrée du scaler, `.ravel()` en sortie). La RMSE se calcule **après** `inverse_transform`, sur les prédictions comme sur les cibles.

</details>
<details><summary>Indice 2</summary>

`prepare_25` : `SeriesScaler().fit(train.reshape(-1, 1))`, puis, pour chaque série, `transform` en colonne, `.ravel()`, `make_windows(..., window=window)`, et `torch.as_tensor(..., dtype=torch.float32)`. `fit_25` : la boucle de `train_classifier`, avec `nn.MSELoss()`. `rmse_25` : prédictions en mode évaluation sans graphe, `.numpy()`, puis une petite fonction qui fait `inverse_transform` sur une colonne et `.ravel()`. `persistence_rmse_25` : la dernière valeur de chaque fenêtre est `X[:, -1, 0]`.

</details>
<details><summary>Indice 3</summary>

```python
def _to_units_25(scaler, values):
    return scaler.inverse_transform(np.asarray(values, dtype=float).reshape(-1, 1)).ravel()


def rmse_25(model, scaler, X, y):
    model.eval()
    with torch.inference_mode():
        pred = model(X).numpy()
    # the square root of the mean of (_to_units_25(pred) - _to_units_25(y)) ** 2, as a float
```

</details>

### Ex 24.26 — Fenêtre, profondeur ou données : qu'est-ce qui aide vraiment ? 🔬

<details><summary>Indice 1</summary>

Relis les §24.5.5 à §24.5.7 de la fiche et le ✏️ 24.5 (le nombre de fenêtres et de pas par epoch). `run_26` réutilise tes fonctions de 24.25 : seule la façon de compter les epochs est nouvelle.

</details>
<details><summary>Indice 2</summary>

Construis les deux séries avec `upsloping(sum_of_sines(...))`, puis `prepare_25`. Un mini-batch de 32 sur $N$ fenêtres : $\lceil N/32 \rceil$ pas par epoch ; il faut donc $\lceil \text{steps} / \lceil N/32 \rceil \rceil$ epochs (`math.ceil` deux fois). Le modèle se construit **après** `torch.manual_seed(seed)` : `TinyForecaster(sizes[0])` ou `Stacked(sizes)` selon `len(sizes)`.

</details>
<details><summary>Indice 3</summary>

```python
def run_26(n_train, window, sizes=(3,), steps=STEPS_26, seed=2426):
    train = upsloping(sum_of_sines(n_train, D_THETA, 0, **BUSY))
    test = upsloping(sum_of_sines(200, D_THETA, TEST_START_26, **BUSY))
    scaler, X_tr, y_tr, X_te, y_te = prepare_25(train, test, window)
    epochs = math.ceil(steps / math.ceil(len(X_tr) / 32))
    # the model after torch.manual_seed(seed), fit_25(..., epochs=epochs, batch_size=32, lr=1e-2, seed=seed), the dict
```

</details>

### Ex 24.27 — Séquences retournées, état porté entre batches, couche par pas de temps 📦

<details><summary>Indice 1</summary>

Relis les §24.5.8 à §24.5.10 de la fiche (les trois détails sur `output` et `h_n`, l'encadré sur l'état porté, le `detach()`) et le ✏️ 24.4. Pour `to_streams_27`, dessine sur papier `np.arange(101)` coupé en 4 flux de 26 valeurs qui se chevauchent d'une valeur.

</details>
<details><summary>Indice 2</summary>

`last_steps_27` : `lstm.bidirectional` et `lstm.hidden_size` disent quoi lire ; `h_n` range les couches et les sens dans son premier axe (couche 0 avant, couche 0 arrière, couche 1 avant…). `to_streams_27` : empile les flux en un tableau `(n_streams, L + 1)`, prends les entrées `[:, :n_chunks * seq_len]` et les cibles `[:, 1:n_chunks * seq_len + 1]`, redécoupe en `(n_streams, n_chunks, seq_len)`, puis échange les deux premiers axes. `run_carried_27` : une boucle sur les morceaux, `lstm(chunk, state)`, et l'état rendu gardé pour le suivant. `train_carried_27` : la boucle de la fiche, avec `head(output).squeeze(-1)`.

</details>
<details><summary>Indice 3</summary>

```python
def to_streams_27(series, n_streams, seq_len):
    series = np.asarray(series, dtype=np.float32)
    L = (len(series) - 1) // n_streams
    n_chunks = L // seq_len
    streams = np.stack([series[b * L:b * L + L + 1] for b in range(n_streams)])     # (n_streams, L + 1)
    inputs = streams[:, :n_chunks * seq_len].reshape(n_streams, n_chunks, seq_len)
    # targets: the same, shifted by one ; then transpose(1, 0, 2), to tensors, and unsqueeze(-1) for X
```

</details>

### Ex 24.28 — Au-delà de nn.Sequential : branches, addition, couches partagées, gel 📦

<details><summary>Indice 1</summary>

Relis le §24.6 de la fiche (le tableau Keras / PyTorch, l'encadré « geler une batchnorm ») et le §24.6.2 (le bloc résiduel, qu'on t'a donné). Dans `__init__`, on crée les couches ; dans `forward()`, on écrit le chemin des données en Python ordinaire.

</details>
<details><summary>Indice 2</summary>

`TwoBranch_28` : deux convolutions rangées comme attributs, la fusion par `a + b` ou `torch.cat([a, b], dim=1)`, puis `torch.relu`. `Siamese_28` : un seul attribut, l'encodeur, appelé deux fois dans `forward`. `make_resnet_28` : une liste qui commence par le tronc, puis un `ResidualBlock(c_in, width, stride)` par largeur. `freeze_28` : `requires_grad_(False)` **et** `eval()`. `train_mode_28` : `model.train()` atteint tous les sous-modules : remets ensuite les modules gelés en évaluation. `trainable_28` : une liste en compréhension sur `model.parameters()`.

</details>
<details><summary>Indice 3</summary>

```python
class Siamese_28(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder                    # ONE module, used twice

    def forward(self, a, b):
        # the squared difference of self.encoder(a) and self.encoder(b), flattened, summed over dim 1


def freeze_28(module):
    module.requires_grad_(False)
    # and the second half of the freezing (the batchnorm statistics), then return the module
```

</details>

### Ex 24.29 — Précision mixte et torch.compile : mesurer avant de croire 📦

<details><summary>Indice 1</summary>

Relis la section « Au-delà du livre : précision mixte et compilation » de la fiche : la boucle de précision mixte, et les règles d'une mesure honnête (chauffer, mesurer chaque pas, la médiane, synchroniser le GPU).

</details>
<details><summary>Indice 2</summary>

`amp_dtype_29` : un dictionnaire, et `ValueError` pour une clé absente. `time_steps_29` : le type de device vient de `DEVICE` (`"cuda"` ou `"cpu"`) ; `torch.autocast(..., enabled=amp)` laisse la même boucle servir aux deux modes ; un `GradScaler(device_type, enabled=(amp and dtype == torch.float16))` ne fait rien quand il est désactivé ; `time.perf_counter()` avant et après le pas, gardé seulement à partir de l'indice `warmup` ; `np.median` à la fin.

</details>
<details><summary>Indice 3</summary>

```python
for i, (xb, yb) in enumerate(batches):
    start = time.perf_counter()
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=amp):
        loss = nn.functional.cross_entropy(model(xb), yb)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    # scaler.step(optimizer), scaler.update(), synchronize on a GPU,
    # then keep the duration and loss.item() when i >= warmup
```

</details>

### Ex 24.30 — Générer du texte : embeddings, séquences décalées, mots contre caractères 📦

<details><summary>Indice 1</summary>

Relis le §24.5.11 de la fiche (les deux améliorations, l'encadré sur l'embedding, la perplexité) et, au ch. 22, tes fonctions `build_vocab`, `encode`, `make_sequence_pairs` et `sample_from_logits`. La loss « à chaque position » est une cross-entropy sur $N \times T$ prédictions à la fois.

</details>
<details><summary>Indice 2</summary>

`tokenize_words_30` : une expression régulière à deux branches, `re.findall` sur le texte en minuscules ; la première branche reconnaît un mot (des lettres, puis éventuellement une apostrophe suivie de lettres, plusieurs fois), la seconde tout caractère qui n'est ni une espace ni une lettre (`[^\sa-z]`). `split_pairs_30` : coupe les ids, puis `make_sequence_pairs` sur chaque partie. `train_lm_30` : `logits.reshape(-1, logits.shape[-1])` contre `yb.reshape(-1)`, et `nn.utils.clip_grad_norm_` entre `backward` et `step`. `mean_loss_30` : `reduction="sum"`, puis divise par le nombre total de tokens. `generate_30` : un premier appel sur tout le prompt (un tenseur `(1, T)`), puis une boucle d'appels sur un seul token, avec l'état.

</details>
<details><summary>Indice 3</summary>

```python
WORD_RE_30 = re.compile(r"[a-z]+...|[^\sa-z]")     # a word, then an optional apostrophe + letters (repeated) | any other non-space


def generate_30(model, prompt_ids, n_tokens, temperature=1.0, seed=0):
    rng = np.random.default_rng(seed)
    model.eval()
    new = []
    with torch.inference_mode():
        logits, state = model(torch.as_tensor([list(prompt_ids)], dtype=torch.int64))
        # n_tokens times: draw from logits[0, -1] with sample_from_logits, append, feed [[next_id]] with the state
    return new
```

</details>

### Ex 24.31 — Défi CIFAR-10 frugal : au moins 80 % de test accuracy avec moins de 100 000 paramètres 🏆

<details><summary>Indice 1</summary>

Pars de ton ResNet de 24.28 (78 042 paramètres) et de l'augmentation de 24.23 : les deux sont faits pour ce défi. Relis aussi, au ch. 23, le défi 23.29 : ce qui a stabilisé la recette (un planning qui finit à un learning rate presque nul, le label smoothing).

</details>
<details><summary>Indice 2</summary>

Une recette classique pour CIFAR-10 : `CifarImages_23` avec la transformation d'entraînement augmentée, des mini-batches de 128, SGD avec momentum de Nesterov et un peu de weight decay, un planning `OneCycleLR` avancé à chaque mini-batch, et `nn.CrossEntropyLoss(label_smoothing=0.1)`. En mode rapide, compte une dizaine de secondes par epoch sur 10 000 images ; en mode complet, sur un GPU, quelques secondes par epoch sur 45 000.

</details>
<details><summary>Indice 3</summary>

```python
def train_31(seed, images, labels, epochs=EPOCHS_31):
    torch.manual_seed(seed)
    model = make_resnet_28().to(DEVICE)
    train_tf, _ = make_transforms_23(augment=True)
    loader = make_train_loader_23(CifarImages_23(images, labels, train_tf), batch_size=128, seed=seed)
    # SGD(lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4) ; OneCycleLR(max_lr=0.1, total_steps=epochs * len(loader))
    # the loop: loss with label smoothing, step, schedule.step() after each mini-batch ; return model
```

</details>
