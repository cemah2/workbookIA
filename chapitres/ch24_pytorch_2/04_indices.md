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

### 24.R1 — Ch. 23 : Ce qu'un historique d'entraînement doit contenir pour comparer deux runs

<details><summary>Indice 1</summary>

Relis le §23.7 de la fiche du ch. 23 et l'encadré 💼 sur le journal d'expériences (§24.2.7).

</details>
<details><summary>Indice 2</summary>

Un historique sert à lire les courbes et à choisir une epoch, sans jamais toucher au test. Pour comparer deux runs, il faut pouvoir dire ce qui les distingue. Pour c), rappelle-toi comment `train_one_epoch` calcule sa loss.

</details>
<details><summary>Indice 3</summary>

a) Les losses d'entraînement et de validation, l'accuracy de validation et le learning rate de chaque epoch servent à lire les courbes ; une copie des poids à chaque epoch coûterait trop de mémoire (on garde le meilleur checkpoint), et le test ne se regarde qu'une fois, à la fin : `"ABD"`. Pour b), garde ce qui distingue deux runs ou permet de les refaire ; pour c), relis comment la loss d'une epoch est moyennée pendant que les poids changent ; pour d), cherche la plus petite des six valeurs et compte sa position à partir de 1.

</details>

### 24.R2 — Ch. 21 : Taille de sortie d'une convolution à padding 'same' et stride 2

<details><summary>Indice 1</summary>

Relis le §24.4.3 de la fiche (le padding) et la formule de taille du ch. 21.

</details>
<details><summary>Indice 2</summary>

Keras « same » : $\lceil H/s \rceil$. PyTorch : $\lfloor (H + 2p - k)/s \rfloor + 1$. Pour reproduire le « same » de Keras avec un noyau impair $k$, on prend $p = \lfloor k/2 \rfloor$.

</details>
<details><summary>Indice 3</summary>

a) $\lceil 28/2 \rceil = 14$. Pour b), $\lceil 15/2 \rceil$ ; pour c), $\lfloor 5/2 \rfloor$ ; pour d), applique la formule de PyTorch avec ce padding à $H = 32$ ; pour e), la même formule avec $k = 3$, $p = 0$, $H = 15$ ; pour f), relis ce que PyTorch fait d'un noyau de taille paire avec `padding="same"`.

</details>

### 24.R3 — Ch. 17 : Ce que change la division des logits avant le softmax

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

### Ex 24.1 — Le coût d'une grille (27 hyperparamètres, 54 entraînements) contre une recherche aléatoire : probabilité de tomber dans les 5 % meilleurs réglages en n essais

<details><summary>Indice 1</summary>

Relis le §24.3.4 de la fiche : le coût d'une grille et la probabilité d'un tirage au hasard.

</details>
<details><summary>Indice 2</summary>

Une grille coûte le produit des nombres de valeurs, fois le nombre de folds. Un tirage rate les 5 % meilleurs avec la probabilité 0,95 ; $n$ tirages indépendants les ratent tous avec la probabilité $0{,}95^n$.

</details>
<details><summary>Indice 3</summary>

a) $2^{27} = 134\,217\,728$ combinaisons. Pour b), divise ce nombre de secondes par $3\,600 \times 24 \times 365$ ; pour c), $3 \times (3 \times 3 \times 2)$ ; pour d), multiplie par 70 s, puis convertis en minutes ; pour e), calcule $1 - 0{,}95^{20}$ ; pour f), cherche le plus petit $n$ tel que $0{,}95^n \le 0{,}05$ (prends les logarithmes) ; pour g), refais e) avec $n = 18$.

</details>

### Ex 24.2 — Formes et paramètres du CNN à pooling du livre

<details><summary>Indice 1</summary>

Relis le §24.4.3 de la fiche (paramètres d'une convolution) et la formule de taille du ch. 21.

</details>
<details><summary>Indice 2</summary>

Avec `padding=2` pour un noyau 5 et `padding=1` pour un noyau 3, une convolution de stride 1 garde $H$ et $W$ ; un `MaxPool2d(2)` les divise par 2. Une convolution a $C_{out}(C_{in} k^2 + 1)$ paramètres, une couche dense $(n_{in} + 1)\,n_{out}$ ; dropout, ReLU, pooling et `Flatten` n'en ont aucun.

</details>
<details><summary>Indice 3</summary>

a) 256 images, 30 filtres, la taille gardée : `"(256, 30, 28, 28)"`. Pour b), divise $H$ et $W$ par 2 ; pour c), applique $C_{out}(C_{in} k^2 + 1)$ avec $C_{in} = 1$, $k = 5$ ; pour d), avec $C_{in} = 30$, $k = 3$, $C_{out} = 16$ ; pour e), multiplie les trois dimensions de la sortie du second pooling ; pour f), $(n + 1) \times 128$ ; pour g), additionne les six couches qui ont des paramètres ; pour h), divise la somme des trois couches denses par g).

</details>

### Ex 24.3 — Stride ou pooling : compter les multiplications

<details><summary>Indice 1</summary>

Relis la formule des MAC au §24.4.3 de la fiche, et reprends les formes de l'Ex 24.2.

</details>
<details><summary>Indice 2</summary>

$H_{out}$ et $W_{out}$ sont ceux de la sortie de **la convolution elle-même**, avant le pooling. Avec un stride de 2 et le padding indiqué, chaque convolution divise la taille par 2 : la première sort en $14 \times 14$, la seconde en $7 \times 7$.

</details>
<details><summary>Indice 3</summary>

a) $28 \times 28 \times 30 \times 1 \times 5 \times 5 = 588\,000$. Pour b), la seconde convolution travaille sur des cartes $14 \times 14$, avec $C_{in} = 30$, $C_{out} = 16$, $k = 3$ ; pour c), $784 \times 128 + 128 \times 64 + 64 \times 10$ ; pour d), refais a) et b) avec des sorties $14 \times 14$ puis $7 \times 7$, et additionne ; pour e), ajoute c) à chaque version, puis divise ; pour f), compte les valeurs de sortie du pooling ($30 \times 14 \times 14$), fois 3 ; pour g), $147/36$.

</details>

### Ex 24.4 — Ce que renvoie nn.LSTM : output, h_n et c_n

<details><summary>Indice 1</summary>

Relis le §24.5.8 de la fiche et la figure des sorties de `nn.LSTM`.

</details>
<details><summary>Indice 2</summary>

`output` : $(N, T, n_{dir} H)$ ; `h_n` et `c_n` : $(n_{couches}\, n_{dir}, N, H)$, quel que soit `batch_first`. Une couche LSTM de PyTorch a $4H(D_{in} + H)$ poids et $2 \times 4H$ biais, où $D_{in}$ est la taille de son entrée ($D$ pour la première couche, $H$ pour les suivantes).

</details>
<details><summary>Indice 3</summary>

a) $N = 8$, $T = 5$, $H = 4$ : `"(8, 5, 4)"`. Pour b), 2 couches, un sens ; pour c), écris la forme de `h_n[0]`, puis celle de chaque option ; pour d) et e), double ce que dit la formule pour deux sens ; pour f), la première couche a $16 \times (3 + 4) + 32$ paramètres, calcule la seconde avec $D_{in} = 4$ et additionne ; pour g), `nn.Linear` remplace la dernière dimension ; pour h), sans `batch_first`, l'axe du temps passe en premier.

</details>

### Ex 24.5 — Fenêtres, pas d'optimisation et durée : de 200 à 20 000 points

<details><summary>Indice 1</summary>

Relis les §24.5.2 et §24.5.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une série de $T$ points donne $T - w$ fenêtres (la dernière fenêtre doit laisser une valeur pour sa cible). Le nombre de pas par epoch est $\lceil N/B \rceil$. Pour les durées, convertis tout en secondes.

</details>
<details><summary>Indice 3</summary>

a) $200 - 13 = 187$ exemples. Pour b), le même calcul avec 20 000 ; pour c), un pas par exemple ; pour d), divise b) par 32 et arrondis au-dessus ; pour e), $100 \times 5$ s, en minutes ; pour f), $28 \times 3\,600$ s divisé par 100 ; pour g), $5 \times$ le rapport des nombres d'exemples a) et b).

</details>

### Ex 24.6 — Dropout inversé et max-norm : espérance et projection

<details><summary>Indice 1</summary>

Relis le §24.2.6 de la fiche, et le rappel sur l'espérance et la variance d'une variable de Bernoulli (0B, ch. 2).

</details>
<details><summary>Indice 2</summary>

$m$ ne prend que deux valeurs : $\mathbb{E}[m] = 1 - p$ et $\mathbb{E}[m^2] = 1 - p$. Utilise $\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$. Pour 4., énumère les quatre masques possibles et leurs probabilités. Pour 5., si $\lVert\mathbf{w}\rVert > c$, un point $\mathbf{v}$ de la boule vérifie $\lVert\mathbf{w}\rVert \le \lVert\mathbf{w} - \mathbf{v}\rVert + \lVert\mathbf{v}\rVert$.

</details>
<details><summary>Indice 3</summary>

1. $\mathbb{E}[\tilde{h}] = \frac{h}{1 - p}\,\mathbb{E}[m] = \frac{h}{1-p}(1 - p) = h$. Pour 2., calcule $\mathbb{E}[\tilde{h}^2] = \frac{h^2}{(1-p)^2}\mathbb{E}[m^2]$ et retranche $h^2$ ; pour 3., calcule l'espérance de $m\,h$ ; pour 4., la linéarité de l'espérance donne la première partie, puis chacun des quatre masques a la probabilité $1/4$ : calcule $z$ dans chaque cas, puis la moyenne des $\text{ReLU}(z)$ ; pour 5., minore $\lVert\mathbf{w} - \mathbf{v}\rVert$ par $\lVert\mathbf{w}\rVert - c$, puis calcule $\lVert\mathbf{w} - \mathbf{w}'\rVert$ ; pour 6., $\lVert\mathbf{w}\rVert = \sqrt{9 + 16 + 144}$, puis applique la formule ; pour la convolution, demande-toi quels poids servent à calculer une même valeur de sortie, et sur quel indice de `weight[o, c, u, v]` ils se repèrent.

</details>

### Ex 24.7 — Fermi : le budget d'une recherche d'hyperparamètres sur CPU, sur GPU et en précision mixte

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

### Ex 24.8 — La data augmentation en cinq lignes

<details><summary>Indice 1</summary>

Relis le §24.4.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

Cherche une situation de la vie courante où l'on apprend à reconnaître quelque chose en le voyant sous plusieurs angles, éclairages ou distances, sans que ce soit une autre chose. Puis dis ce qu'on ne fait pas (changer la nature de la chose, ou s'entraîner avec des vues impossibles).

</details>
<details><summary>Indice 3</summary>

Commence par : « Pour apprendre à reconnaître un chat, il faut l'avoir vu de près, de loin, à gauche, à droite… » Puis : on fabrique ces vues à partir des photos qu'on a, à chaque passage différemment ; le label reste le même ; on ne transforme que les photos d'entraînement ; on évite ce qui n'arrive jamais (un chat la tête en bas, un « 6 » retourné qui devient un « 9 »).

</details>

### Ex 24.9 — Une seule graine suffit-elle ? Reproductibilité et honnêteté des comparaisons d'hyperparamètres

<details><summary>Indice 1</summary>

Relis le §24.2.7 de la fiche (le journal d'expériences), le §24.3.4, et la partie « reproductibilité » de la fiche du ch. 23.

</details>
<details><summary>Indice 2</summary>

Un écart entre deux moyennes ne veut dire quelque chose que s'il est grand devant leur dispersion. Le maximum de plusieurs mesures bruitées est, en moyenne, plus haut que la vraie valeur de celle qui gagne. Un test qui a servi à choisir n'est plus un test.

</details>
<details><summary>Indice 3</summary>

1. La moyenne de A vaut $434{,}5/5 = 86{,}90$ ; calcule de même celle de B, puis les écarts-types avec $n - 1$, et compare l'écart des moyennes aux écarts-types. Pour 2., pense à tout ce qui est tiré au hasard entre le chargement des données et la fin de l'entraînement ; pour 3., pense au plus haut de 60 lancers d'un dé un peu bruité ; pour 4., demande-toi ce qu'il reste pour estimer la performance sans biais ; pour 5., pense à ce qu'un lecteur ne peut pas deviner (le nombre de runs, les heures de GPU) ; pour 6., reprends les réponses précédentes.

</details>

### Ex 24.10 — Lire He et coll. (2016) : les réseaux résiduels

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

*Les indices des exercices 24.11 à 24.31 arrivent avec le notebook, à la prochaine session de génération.*
