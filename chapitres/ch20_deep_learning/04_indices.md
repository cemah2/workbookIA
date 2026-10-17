# 20 · Deep learning et premiers pas en PyTorch — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 20.Q1 — Deep learning et machine learning : qui contient qui ?

<details><summary>Indice 1</summary>

Relis le premier paragraphe du §20.2 de la fiche : le livre y signale une ambiguïté de vocabulaire.

</details>
<details><summary>Indice 2</summary>

Pour c), demande-toi lequel des quatre modèles n'est pas fait d'une pile de couches de neurones. Pour d), regarde ce que chaque couche reçoit en entrée, et ce que cela change pour les motifs qu'elle peut détecter (figure 20.1 du livre).

</details>
<details><summary>Indice 3</summary>

a) Vrai : pour la plupart des auteurs, le machine learning englobe le deep learning et les autres méthodes. Pour b), relis la phrase de la fiche sur « d'autres » auteurs ; pour c), cherche le seul modèle du ch. 13 ou 14 de la liste ; pour d), retiens que chaque couche voit les sorties de la précédente, pas les données brutes.

</details>

### 20.Q2 — Tenseurs : nombre de dimensions, taille, forme

<details><summary>Indice 1</summary>

Relis le §20.2.1 de la fiche et son encadré « Rappel outil ».

</details>
<details><summary>Indice 2</summary>

Le nombre de dimensions est la longueur de la forme ; le nombre d'éléments est le produit des tailles ; indexer par `x[0]` retire la première dimension. Pour e), compare le type par défaut des réels de NumPy et de PyTorch.

</details>
<details><summary>Indice 3</summary>

a) La forme `(32, 3, 28, 28)` a 4 nombres : 4 dimensions. Pour b), multiplie les quatre tailles ; pour c), retire la dimension des exemples ; pour d), relis la définition du livre (« même longueur ») ; pour e), retiens que `1.0` est un réel et que PyTorch n'est pas NumPy.

</details>

### 20.Q3 — Couche d'entrée, couche de sortie : laquelle contient des neurones ?

<details><summary>Indice 1</summary>

Relis les §20.3.1 et §20.3.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

La couche d'entrée ne calcule rien : que contient-elle ? Pour c), compte les couches qui ne sont ni l'entrée (784) ni la sortie (10). Pour d), rappelle-toi où le livre dessine les entrées.

</details>
<details><summary>Indice 3</summary>

a) Faux : la couche d'entrée n'est qu'une mémoire où l'on dépose les données. Pour b), demande-toi ce qui produit les scores de sortie ; pour c), il reste les couches de 256 et de 128 neurones ; pour d), les entrées sont en bas, les sorties en haut.

</details>

### 20.Q4 — Couche dense et couche d'activation séparée

<details><summary>Indice 1</summary>

Relis les §20.4.1 et §20.4.2 de la fiche, puis l'encadré « Piège classique » du §20.3.2.

</details>
<details><summary>Indice 2</summary>

Une couche dense a un poids par connexion et un biais par neurone. Une activation appliquée dans la couche ou juste après donne le même nombre. Pour d), demande-toi qui fait la softmax.

</details>
<details><summary>Indice 3</summary>

a) Chacun des 3 neurones reçoit les 4 sorties : $4 \times 3 = 12$ poids. Pour b), ajoute un biais par neurone ; pour c), compare les deux calculs ; pour d), rappelle-toi que `CrossEntropyLoss` applique elle-même `log_softmax`.

</details>

### 20.Q5 — Vrai ou faux : le dropout tire ses neurones une fois par epoch

<details><summary>Indice 1</summary>

Relis le §20.4.3 de la fiche, ses formules et l'encadré « Le livre, corrigé ».

</details>
<details><summary>Indice 2</summary>

Un masque est tiré à chaque passe avant ; une epoch de $B$ mini-batches fait $B$ passes avant. Le facteur du dropout inversé garde l'espérance de chaque activation.

</details>
<details><summary>Indice 3</summary>

a) Faux : c'est la description du livre, pas ce que fait PyTorch. Pour b), compte les passes avant d'une epoch ; pour c), divise par la probabilité de garder un neurone, $1 - p$ ; pour d), relis ce que fait le dropout en évaluation.

</details>

### 20.Q6 — Vrai ou faux : la batchnorm n'a aucun paramètre

<details><summary>Indice 1</summary>

Relis le §20.4.4 de la fiche, ses formules et l'encadré « Le livre, corrigé ».

</details>
<details><summary>Indice 2</summary>

Distingue ce qui est **appris** par la rétropropagation ($\gamma$, $\beta$) de ce qui est **tenu à jour** par une moyenne mobile (la moyenne et la variance glissantes). Compte-les par feature.

</details>
<details><summary>Indice 3</summary>

a) Faux : $\gamma$ et $\beta$ sont appris. Pour b), deux paramètres appris par feature, et 64 features ; pour c), deux statistiques par feature ; pour d), demande-toi pourquoi une prédiction ne devrait pas dépendre des autres exemples du mini-batch.

</details>

### 20.Q7 — Convolution et pooling en un coup d'œil

<details><summary>Indice 1</summary>

Relis les §20.4.5 et §20.4.6 de la fiche, et la formule du nombre de paramètres d'une convolution.

</details>
<details><summary>Indice 2</summary>

$(k^2 C_{in} + 1)\,C_{out}$ paramètres. Un pooling $k \times k$ à blocs disjoints divise la largeur par $k$ (partie entière). Un filtre $3 \times 3$ déborde d'un pixel de chaque côté de l'image.

</details>
<details><summary>Indice 3</summary>

a) $(3^2 \times 1 + 1) \times 2 = 20$. Pour b) et c), divise la largeur par la taille du bloc ; pour d), demande-toi ce que comble une bordure de zéros d'un pixel.

</details>

### 20.Q8 — Récurrence n'est pas récursion

<details><summary>Indice 1</summary>

Relis le §20.4.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

La récurrence répète la même opération à chaque élément d'une séquence en gardant un état ; la récursion est une fonction qui s'appelle elle-même. Pour b), cherche la seule tâche où l'ordre des éléments, et ce qui précède, compte.

</details>
<details><summary>Indice 3</summary>

a) Faux : le livre insiste sur la différence. Pour b), parmi une photo, une maison et une phrase, laquelle est une séquence ? Pour c), demande-toi si une couche dense garde quoi que ce soit d'une entrée à la suivante.

</details>

### 20.Q9 — Aplatir, remodeler, rogner, rembourrer, suréchantillonner, bruiter

<details><summary>Indice 1</summary>

Relis le §20.4.8 de la fiche et son tableau des couches utilitaires.

</details>
<details><summary>Indice 2</summary>

Le zero-padding ajoute une bordure de chaque côté ; le suréchantillonnage multiplie la taille ; un recadrage centré retire autant de chaque côté. Les couches utilitaires n'apprennent rien. Une couche de bruit est une régularisation, comme le dropout.

</details>
<details><summary>Indice 3</summary>

a) Une bordure de 1 pixel à gauche et à droite : $8 + 2 = 10$. Pour b), multiplie par 2 ; pour c), répartis $28 - 24$ sur deux côtés ; pour d), relis la colonne « Paramètres appris » du tableau ; pour e), demande-toi à quoi sert une régularisation au moment de prédire.

</details>

### 20.Q10 — VGG16 : ce qu'il reconnaît et ce qu'il ignore

<details><summary>Indice 1</summary>

Relis le §20.6 de la fiche et son encadré « Le livre, corrigé ».

</details>
<details><summary>Indice 2</summary>

Le « 16 » de VGG16 compte les couches de calcul : convolutions et couches denses. Une accuracy top-5 compte une réponse juste quand la bonne classe est parmi les 5 plus probables. La couche de sortie fixe la liste des réponses possibles.

</details>
<details><summary>Indice 3</summary>

a) 13 convolutions, en 5 blocs. Pour b), retire 13 de 16 ; pour c), relis le nombre de classes du concours ; pour d), relis l'encadré ; pour e), compte les neurones de la couche de sortie : chacun correspond à un nom.

</details>

### 20.Q11 — Construire un réseau profond : par où commencer ?

<details><summary>Indice 1</summary>

Relis les §20.7 et §20.7.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

La démarche du livre a cinq étapes, dans l'ordre ; il nomme l'hyperparamètre le plus important à l'étape 4. Les noms MLP, CNN et RNN viennent du type de couche dominant.

</details>
<details><summary>Indice 3</summary>

a) B : on regarde et prépare les données avant tout. Pour b), relis l'étape 4 ; pour c), relis la phrase sur les recettes ; pour d), « convolution » donne la lettre C du sigle.

</details>

### 20.Q12 — Explication complète ou explication satisfaisante ?

<details><summary>Indice 1</summary>

Relis les §20.8 et §20.8.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une explication complète montre tous les calculs ; une explication satisfaisante dit quels facteurs ont compté et pourquoi ils sont légitimes. Pour b), pense à ce que la banque observe après un refus. Pour d), pense à la dérive.

</details>
<details><summary>Indice 3</summary>

a) Faux : l'exemple des chaussettes bleues est complet et inacceptable. Pour b), un prêt refusé n'est jamais remboursé ni impayé ; pour c), cherche le terme du livre pour un système qu'on ne peut pas expliquer ; pour d), demande-toi si le monde a pu changer depuis la mesure.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 20.R1 — Que fait weight_decay dans AdamW ?

<details><summary>Indice 1</summary>

Relis la section AdamW de la fiche du ch. 19.

</details>
<details><summary>Indice 2</summary>

AdamW rétrécit d'abord le poids d'un facteur $1 - \eta\lambda$, puis fait le pas d'Adam sur le gradient brut. Avec un gradient nul, le pas d'Adam est nul.

</details>
<details><summary>Indice 3</summary>

a) $2 \times (1 - 0{,}01 \times 0{,}1)$. Pour b), relis ce qui distingue AdamW d'`Adam(weight_decay=...)` ; pour c), ouvre la documentation de `torch.optim.AdamW` (ou la fiche du ch. 19).

</details>

### 20.R2 — Pourquoi calcule-t-on la loss à partir des logits ?

<details><summary>Indice 1</summary>

Relis, au ch. 17, la softmax stable, puis l'encadré « Piège classique » du §20.3.2 de cette fiche.

</details>
<details><summary>Indice 2</summary>

$\mathrm{log\_softmax}(\mathbf{z})_k = z_k - \log\sum_j e^{z_j}$, et $\log\sum_j e^{z_j} = z_{\max} + \log\sum_j e^{z_j - z_{\max}}$. Calculer d'abord la softmax puis son logarithme ferait apparaître $e^{-1000}$, qui vaut 0 en `float32`.

</details>
<details><summary>Indice 3</summary>

a) Faux : elle attend des logits. Pour b), $\log(e^{1000} + e^{0}) \approx 1000$, donc la composante vaut environ $0 - 1000$ ; pour c), une softmax donne des probabilités ; pour d), demande-toi ce que vaut $\log(0)$.

</details>

### 20.R3 — Limiter l'overfitting d'un arbre, et l'équivalent pour un réseau

<details><summary>Indice 1</summary>

Relis, au ch. 13, la section sur l'overfitting des arbres, puis la définition de la régularisation au §20.4.3.

</details>
<details><summary>Indice 2</summary>

Un arbre surapprend quand il est trop profond. Un réseau surapprend quand il a trop de capacité pour ses données : les gestes qui régularisent réduisent cette capacité ou ajoutent du bruit. Le livre appelle régularisation toute technique qui retarde l'overfitting.

</details>
<details><summary>Indice 3</summary>

a) A : limiter la profondeur. Pour b), écarte les gestes qui augmentent la capacité ou suppriment la mesure ; pour c), demande-toi ce que fait l'early stopping à l'overfitting.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 20.1 — Suivre la forme d'un tenseur couche après couche

<details><summary>Indice 1</summary>

Relis le §20.2.1 de la fiche (conventions de forme), le tableau du §20.4.8 et l'encadré « Piège classique » sur `torch.flatten`.

</details>
<details><summary>Indice 2</summary>

Un remodelage garde le nombre d'éléments. `nn.Flatten()` garde la première dimension et multiplie toutes les autres ; `torch.flatten(x)` multiplie tout. Une couche `nn.Linear(n_in, n_out)` remplace la dernière dimension $n_{in}$ par $n_{out}$, et a $n_{in} n_{out} + n_{out}$ paramètres.

</details>
<details><summary>Indice 3</summary>

a) $10 \times 3 \times 5 = 150$. Pour b), garde 10 et calcule $150/10$ ; pour c) et f), garde 32 et multiplie les trois autres tailles ; pour d), remplace 784 par 128 ; pour e), applique la formule ; pour g) et h), décide si la dimension des exemples est gardée, et souviens-toi qu'une forme à une dimension s'écrit avec une virgule.

</details>

### Ex 20.2 — Dimensionner la couche de sortie et choisir la loss

<details><summary>Indice 1</summary>

Relis le §20.3.2 de la fiche et son tableau.

</details>
<details><summary>Indice 2</summary>

Une valeur réelle : `MSELoss`. Une seule classe vraie parmi $K$ (même $K = 2$) : $K$ logits et `CrossEntropyLoss`. Une probabilité par sortie, chaque sortie oui ou non indépendamment : `BCEWithLogitsLoss`.

</details>
<details><summary>Indice 3</summary>

a) Un prix : une valeur, donc 1 sortie. Pour b), c'est une régression ; pour c) et d), les 26 lettres sont exclusives ; pour e), un seul logit ; pour f) et g), les labels ne sont pas exclusifs ; pour h), deux scores dont un seul est vrai.

</details>

### Ex 20.3 — Dropout inversé à la main

<details><summary>Indice 1</summary>

Relis les formules du §20.4.3 de la fiche, et l'espérance d'une variable de Bernoulli (ch. 2).

</details>
<details><summary>Indice 2</summary>

Le masque vaut $m_i/(1 - p)$, avec $m_i = 1$ si $u_i \ge p$ et 0 sinon ; la sortie est $a_i$ fois le masque, la passe arrière multiplie le gradient par le même masque. En moyenne sur les tirages, chaque activation vaut $a_i$.

</details>
<details><summary>Indice 3</summary>

a) Chaque neurone est coupé avec la probabilité 0,2 : en moyenne $0{,}2 \times 100 = 20$. Pour b), compte les $u_i \ge 0{,}4$ ; pour c), calcule $1/0{,}6$ ; pour d), le neurone 2 est-il gardé ? Pour e), additionne les sorties des neurones gardés ; pour f), utilise $\mathbb{E}[\tilde a_i] = a_i$ ; pour g), additionne les valeurs du masque ; pour h), relis ce que fait le dropout en évaluation.

</details>

### Ex 20.4 — Batchnorm à la main sur un mini-batch de quatre exemples

<details><summary>Indice 1</summary>

Relis le §20.4.4 de la fiche : ses formules, la mise à jour des statistiques glissantes et l'encadré sur les deux variances.

</details>
<details><summary>Indice 2</summary>

Une feature = une colonne. Pour normaliser, la variance divise par 4 ; pour la statistique glissante, la variance non biaisée divise par 3. La mise à jour : $(1 - 0{,}1) \times$ l'ancienne $+ 0{,}1 \times$ la nouvelle, en partant de 0 pour la moyenne et de 1 pour la variance. En évaluation, on remplace $\mu_B$ et $\sigma_B^2$ par les statistiques glissantes.

</details>
<details><summary>Indice 3</summary>

a) $(1 + 3 + 5 + 7)/4 = 4$. Pour c), les écarts à la moyenne sont $-3, -1, 1, 3$ ; pour d), fais de même avec la colonne 2 ; pour e) et f), applique $\hat x = (x - \mu)/\sigma$ puis $y = \gamma\hat x + \beta$ ; pour g), que vaut la moyenne des $\hat x$ ? Pour h) à j), applique la mise à jour ; pour k), $2 \times (4 - 0{,}4)/\sqrt{\sigma^{2,\text{run}}_1} + 0$.

</details>

### Ex 20.5 — Dropout : espérance, sous-réseaux et règle du test

<details><summary>Indice 1</summary>

Revois au ch. 2 l'espérance et la variance d'une variable de Bernoulli, puis relis le §20.4.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

1. $m^2 = m$ pour une variable qui vaut 0 ou 1. 2. Un masque est une suite de $n$ bits. 3. Linéarité de l'espérance. 4. Pose $w' = (1 - p)\,w$ et compare les deux calculs, en entraînement puis au test. 5. Énumère les quatre masques, chacun de probabilité $\frac14$ ; l'inégalité porte sur les fonctions convexes.

</details>
<details><summary>Indice 3</summary>

1. $\mathbb{E}[m] = 1 - p$, donc $\mathbb{E}[\tilde a] = a\,(1 - p)/(1 - p)$ ; pour la variance, calcule $\mathbb{E}[\tilde a^2] = a^2\,\mathbb{E}[m^2]/(1 - p)^2$ puis retire $a^2$. 2. $2^n$ masques ; pour $n = 100$, utilise $2^{10} \approx 10^3$. 3. $\mathbb{E}[z] = \sum_i w_i a_i\,\mathbb{E}[m_i]$. 4. En entraînement, $\sum_i w'_i a_i m_i/(1 - p) = \sum_i w_i a_i m_i$ : les deux réseaux calculent la même chose ; regarde ensuite ce que chacun utilise au test. 5. Les quatre valeurs de $z$ sont $0$, $1$, $-1$ et $0$ ; compare la moyenne de $h(z)$ à $h$ de la moyenne, et pense à l'inégalité de Jensen.

</details>

### Ex 20.6 — Compter les 138 millions de paramètres de VGG16

<details><summary>Indice 1</summary>

Relis la formule du nombre de paramètres d'une convolution (fiche §20.4.5) et celle d'une couche dense (§20.4.1).

</details>
<details><summary>Indice 2</summary>

Le zero-padding de 1 garde la taille à chaque convolution ; chaque pooling divise la largeur par 2. Dans un bloc, $C_{in}$ de la première convolution est le nombre de filtres du bloc précédent (3 pour la toute première). La première couche dense reçoit l'aplatissement de $512 \times 7 \times 7$ valeurs.

</details>
<details><summary>Indice 3</summary>

a) $224/2^5 = 7$. Pour b), $(9 \times 3 + 1) \times 64$ ; pour c), fais un tableau à 13 lignes $(9\,C_{in} + 1)\,C_{out}$ et additionne (la deuxième ligne est $(9 \times 64 + 1) \times 64 = 36\,928$) ; pour d), $25\,088 \times 4\,096 + 4\,096$ ; pour e), ajoute les deux autres couches denses ; pour f) et g), divise ou multiplie par 4 octets ; pour h), compare 553,4 et 527,8 : quelle unité donne le plus petit nombre ?

</details>

### Ex 20.7 — Connexion résiduelle et LayerNorm à la main

<details><summary>Indice 1</summary>

Relis la layer norm (fiche §20.4.8) et la section sur les connexions résiduelles.

</details>
<details><summary>Indice 2</summary>

La layer norm standardise les features d'un exemple avec sa moyenne et sa variance biaisée. Le facteur à travers une chaîne est le produit des dérivées ; un bloc résiduel a pour dérivée $1 + F'$. Pour g), cherche le plus petit $n$ tel que $0{,}9^n < 0{,}01$.

</details>
<details><summary>Indice 3</summary>

a) $(1 + 2 + 3 + 6)/4 = 3$. Pour b), les écarts sont $-2, -1, 0, 3$ ; pour c), $(6 - 3)/\sqrt{\sigma^2}$ ; pour d), une sortie de variance 1 et de moyenne 0 sur 4 valeurs ; pour e), relis sur quoi la layer norm calcule ses statistiques ; pour f), $0{,}5^{30}$ divisé par $10^{-9}$ ; pour g), $n > \ln 0{,}01/\ln 0{,}9$ ; pour h), $(1 + 0{,}01)^{30}$.

</details>

### Ex 20.8 — Le gradient de la batchnorm

<details><summary>Indice 1</summary>

Relis la rétropropagation à travers une somme et un produit (ch. 18), puis les formules du §20.4.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

1. $y_i$ dépend de $\beta$ et $\gamma$ directement. 2. $\sum_i(x_i - \mu) = 0$. 3. Chaque $x_j$ agit sur $\hat x_i$ par trois chemins : directement, par $\mu$ et par $\sigma^2$ (donc par $s$). 4. Règle de la chaîne : $\partial L/\partial x_j = \sum_i g_i\,\gamma\,\partial\hat x_i/\partial x_j$. 5. Utilise la question 2. 6. $\mu$ et $\sigma^2$ ne dépendent plus de $x$. 7. Que vaut $\sigma^2$ pour une seule valeur ? 8. Une seule feature par canal.

</details>
<details><summary>Indice 3</summary>

1. $\partial y_i/\partial\beta = 1$ et $\partial y_i/\partial\gamma = \hat x_i$. 3. $\partial\mu/\partial x_j = 1/m$ ; $\partial\sigma^2/\partial x_j = \frac2m(x_j - \mu)$, le terme en $\partial\mu/\partial x_j$ étant multiplié par $\sum_i(x_i - \mu) = 0$ ; avec $\partial s/\partial x_j = -\frac12(\sigma^2 + \epsilon)^{-3/2}\,\partial\sigma^2/\partial x_j$, montre que $\partial\hat x_i/\partial x_j = s\,(\delta_{ij} - \frac1m - \frac1m\hat x_i\hat x_j)$. 4. Remplace dans la somme et sépare les trois termes. 5. Somme sur $j$ en utilisant $\sum_j \hat x_j = 0$, puis $\sum_j \hat x_j^2$. 6. Il ne reste que le chemin direct. 7. $\hat x = 0$ quelle que soit l'entrée. 8. Les sommes portent sur $n$, $h$ et $w$.

</details>

### Ex 20.9 — Fermi : entraîner VGG16 sur ImageNet chez soi ?

<details><summary>Indice 1</summary>

Relis le §20.6 de la fiche et, au ch. 18, le coût d'une passe arrière comparé à la passe avant.

</details>
<details><summary>Indice 2</summary>

Coût total = coût par image × nombre d'images × nombre d'epochs. Durée = coût total / débit, puis divise par 86 400 secondes par jour. Mémoire : 4 nombres `float32` par paramètre (le poids, son gradient, $m$ et $v$).

</details>
<details><summary>Indice 3</summary>

a) $3 \times 15{,}5 = 46{,}5$ milliards par image. Pour b), multiplie par $1{,}28 \times 10^6$ et par 74, puis divise par $10^{18}$ ; pour c) et d), divise par le débit puis par 86 400 ; pour e), $4 \times 4 \times 138\,357\,544$ octets, divisés par $10^9$.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 20.10 — Agents, superviseurs, directeur : expliquer l'apprentissage hiérarchique

<details><summary>Indice 1</summary>

Relis le §20.8 de la fiche, puis le §20.2 sur l'apprentissage hiérarchique.

</details>
<details><summary>Indice 2</summary>

Une phrase par niveau (ce qu'il voit, ce qu'il produit), une phrase sur la précision, une sur l'explication, une sur la limite de l'analogie : dans un réseau, personne ne « décide » ses règles, elles sont toutes ajustées ensemble à partir des erreurs passées.

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Imagine une banque où cinq agents lisent chacun le dossier et donnent une note ; des superviseurs ne lisent pas le dossier, seulement les notes des agents… » Continue avec le directeur, puis la difficulté d'expliquer, puis la limite.

</details>

### Ex 20.11 — Refus de prêt automatisé : explication, dérive et biais

<details><summary>Indice 1</summary>

Relis le §20.8.1 de la fiche et ses deux encadrés (🔗 et 🕰️), puis la matrice de confusion du ch. 3.

</details>
<details><summary>Indice 2</summary>

1. Distingue « tous les calculs » de « les raisons qui ont compté, et ce qu'on pourrait changer ». 2. Une feature corrélée à une caractéristique sensible la transmet au modèle. 3. Compare la distribution des entrées d'aujourd'hui à celle de l'entraînement. 4. On n'observe l'issue que d'un côté de la décision. 5. Relis les conditions de l'article 22 (« exclusivement », « effets significatifs ») et la catégorie de l'annexe III. 6. Pense aux données, à l'évaluation, à l'humain et au suivi.

</details>
<details><summary>Indice 3</summary>

1. Les 40 valeurs et les poids sont une explication complète ; une explication satisfaisante ressemblerait à : « les deux facteurs qui ont le plus pesé sont… ; votre demande serait acceptée si… ». 2. Calcule le taux de refus et le taux de faux positifs par groupe (par tranche d'âge, par zone) et compare-les. 3. Pense à l'inflation, aux taux d'intérêt, au marché du travail ; surveille l'écart entre les distributions. 4. Une solution consiste à accorder une petite part de prêts au hasard. 5. Une décision automatique, sans humain, qui refuse un crédit : regarde l'arrêt SCHUFA ; l'AI Act classe la solvabilité dans l'annexe III. 6. Quatre mesures, une par domaine.

</details>

### Ex 20.12 — Ioffe & Szegedy (2015), puis Santurkar et coll. (2018) : lecture critique

<details><summary>Indice 1</summary>

Lis d'abord le résumé et l'introduction des deux articles, puis reviens aux questions dans l'ordre. Garde la fiche §20.4.4 à côté.

</details>
<details><summary>Indice 2</summary>

1 et 2. L'introduction et le §3 de l'article de 2015. 3. L'algorithme 2 et le paragraphe juste avant (le facteur $\frac{m}{m-1}$). 4. Le §3.2. 5. Le §4.2 et son tableau. 6. La figure 2 de Santurkar et coll. 7. Leur §3 (« smoothness », la constante de Lipschitz du gradient). 8. Sépare ce qui se mesure de ce qui s'interprète.

</details>
<details><summary>Indice 3</summary>

1. Le changement de la distribution des entrées d'une couche pendant l'entraînement, parce que les couches précédentes changent. 2. Avec $\gamma = \sqrt{\mathrm{Var}[x]}$ et $\beta = \mathbb{E}[x]$, on retrouve l'identité. 3. Cherche l'estimation sans biais de la variance. 4. Toutes les positions d'une carte de features partagent les statistiques. 5. Cherche le nombre de pas pour atteindre l'accuracy d'Inception. 6. Un bruit qui recrée artificiellement un décalage de distribution, et la vitesse d'entraînement qui reste bonne. 7. Le gradient change moins brusquement : un pas plus grand dans sa direction reste sûr. 8. La méthode marche ; c'est l'explication qui a changé.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 20.E1 — Dropout : fonctionnement, et ce que change model.eval()

<details><summary>Indice 1</summary>

Relis le §20.4.3 de la fiche et l'encadré « Piège classique — oublier `model.eval()` ».

</details>
<details><summary>Indice 2</summary>

Quatre points : ce que fait la couche en entraînement (masque à chaque passe, facteur $1/(1-p)$), pourquoi elle régularise, ce qu'elle fait en évaluation, les symptômes d'un `eval()` oublié.

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Le dropout est une régularisation : à chaque passe avant d'entraînement, chaque activation est mise à zéro avec une probabilité p… » Continue avec le facteur, la raison, `eval()` et l'oubli.

</details>

### 20.E2 — Batchnorm ou LayerNorm : différences et usages

<details><summary>Indice 1</summary>

Relis les §20.4.4 et §20.4.8 de la fiche, et la figure des deux grilles.

</details>
<details><summary>Indice 2</summary>

Sur quels axes chacune calcule ses statistiques ; ce que cela change entre l'entraînement et l'évaluation ; leur dépendance à la taille du mini-batch ; où chacune s'utilise.

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Les deux appliquent la même formule, une standardisation suivie d'une échelle γ et d'un décalage β appris ; la différence est l'axe des statistiques… » Puis : statistiques glissantes, petits mini-batches, séquences, Transformers.

</details>

### 20.E3 — Décrire une boucle d'entraînement PyTorch, étape par étape

<details><summary>Indice 1</summary>

Relis la section « PyTorch en pratique » de la fiche, et la figure de la boucle.

</details>
<details><summary>Indice 2</summary>

Avant la boucle (modèle, loss, optimiseur, device, `train()`), dans la boucle (les cinq gestes), après chaque epoch (la validation avec `eval()` et `no_grad()`, l'early stopping ou la sauvegarde).

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Je crée le modèle, je l'envoie sur le device, je crée la loss et l'optimiseur sur `model.parameters()`. Pour chaque epoch, `model.train()`, puis pour chaque mini-batch du DataLoader… »

</details>

### 20.E4 — Ton modèle surapprend : que fais-tu, et dans quel ordre ?

<details><summary>Indice 1</summary>

Relis les §20.7.1 de la fiche, la section sur la régularisation, et le ch. 9.

</details>
<details><summary>Indice 2</summary>

D'abord vérifier que c'est bien de l'overfitting (les courbes, une fuite, un écart de distribution entre entraînement et validation) ; ensuite les remèdes, du moins cher au plus cher : early stopping, plus de données ou d'augmentation, régularisation (weight decay, dropout), modèle plus petit.

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Avant de corriger, je regarde les courbes d'entraînement et de validation epoch par epoch, et je vérifie que la validation ressemble aux données d'entraînement… »

</details>

### 20.E5 — Pourquoi les connexions résiduelles rendent-elles les réseaux très profonds entraînables ?

<details><summary>Indice 1</summary>

Relis la section sur les connexions résiduelles de la fiche, et ✏️ 20.7.

</details>
<details><summary>Indice 2</summary>

Deux arguments : le gradient (la dérivée $1 + F'$ et le produit sur la profondeur) et l'apprentissage (une correction plutôt qu'une transformation entière, l'identité facile à représenter) ; puis où on les trouve.

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Un bloc résiduel calcule y = x + F(x) au lieu de F(x). Par la règle de la chaîne, sa dérivée est l'identité plus celle de F… »

</details>

<a id="notebook"></a>

## Notebook

Les indices des exercices du notebook (20.13 à 20.31) arrivent avec ses parties A à D, à la prochaine session de génération du chapitre.
