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

a) Vrai : pour la plupart des auteurs, le machine learning englobe le deep learning et les autres méthodes. Pour b), relis la phrase de la fiche sur « d'autres » auteurs ; pour c), cherche le seul modèle du ch. 13 ou 14 de la liste ; pour d), demande-toi ce que reçoit en entrée une couche du milieu de la pile.

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

a) Faux : la couche d'entrée n'est qu'une mémoire où l'on dépose les données. Pour b), demande-toi ce qui produit les scores de sortie ; pour c), il reste les couches de 256 et de 128 neurones ; pour d), rappelle-toi où le livre dessine les entrées, et dans quel sens se lit la pile.

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

a) Faux : le livre insiste sur la différence. Pour b), demande-toi pour laquelle des quatre tâches l'ordre des éléments, et ce qui les précède, change la réponse. Pour c), demande-toi si une couche dense garde quoi que ce soit d'une entrée à la suivante.

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

a) B : on regarde et prépare les données avant tout. Pour b), relis l'étape 4 ; pour c), relis la phrase sur les recettes ; pour d), souviens-toi que chaque sigle vient du type de couche qui domine le réseau.

</details>

### 20.Q12 — Explication complète ou explication satisfaisante ?

<details><summary>Indice 1</summary>

Relis les §20.8 et §20.8.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une explication complète montre tous les calculs ; une explication satisfaisante dit quels facteurs ont compté et pourquoi ils sont légitimes. Pour b), pense à ce que la banque observe après un refus. Pour d), pense à la dérive.

</details>
<details><summary>Indice 3</summary>

a) Faux : l'exemple des chaussettes bleues est complet et inacceptable. Pour b), demande-toi ce que la banque apprend, après coup, d'un dossier qu'elle a refusé ; pour c), cherche le terme du livre pour un système qu'on ne peut pas expliquer ; pour d), demande-toi si le monde a pu changer depuis la mesure.

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

Le facteur appliqué vaut $m_i/(1 - p)$, avec $m_i = 1$ si $u_i \ge p$ et 0 sinon ; la sortie est $a_i$ fois le masque, la passe arrière multiplie le gradient par le même masque. En moyenne sur les tirages, chaque activation vaut $a_i$.

</details>
<details><summary>Indice 3</summary>

a) Chaque neurone est coupé avec la probabilité 0,2 : en moyenne $0{,}2 \times 100 = 20$. Pour b), compte les $u_i \ge 0{,}4$ ; pour c), calcule $1/0{,}6$ ; pour d), le neurone 2 est-il gardé ? Pour e), additionne les sorties des neurones gardés ; pour f), utilise $\mathbb{E}[\tilde a_i] = a_i$ ; pour g), additionne les facteurs $m_i/(1 - p)$ ; pour h), relis ce que fait le dropout en évaluation.

</details>

### Ex 20.4 — Batchnorm à la main sur un mini-batch de quatre exemples

<details><summary>Indice 1</summary>

Relis le §20.4.4 de la fiche : ses formules, la mise à jour des statistiques glissantes et l'encadré sur les deux variances.

</details>
<details><summary>Indice 2</summary>

Une feature = une colonne. Pour normaliser, la variance divise par 4 ; pour la statistique glissante, la variance non biaisée divise par 3. La mise à jour : $(1 - 0{,}1) \times$ l'ancienne $+ 0{,}1 \times$ la nouvelle, en partant de 0 pour la moyenne et de 1 pour la variance. En évaluation, on remplace $\mu_B$ et $\sigma_B^2$ par les statistiques glissantes.

</details>
<details><summary>Indice 3</summary>

a) $(1 + 3 + 5 + 7)/4 = 4$. Pour c), les écarts à la moyenne sont $-3, -1, 1, 3$ ; pour d), fais de même avec la colonne 2 ; pour e) et f), applique $\hat x = (x - \mu)/\sigma$ puis $y = \gamma\hat x + \beta$ ; pour g), que vaut la moyenne des $\hat x$ ? Pour h) à j), applique la mise à jour ; pour k), $\gamma_1\,(4 - \mu^{\text{run}}_1)/\sqrt{\sigma^{2,\text{run}}_1} + \beta_1$.

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

a) $224/2^5 = 7$. Pour b), $(9 \times 3 + 1) \times 64$ ; pour c), fais un tableau à 13 lignes $(9\,C_{in} + 1)\,C_{out}$ et additionne (la deuxième ligne est $(9 \times 64 + 1) \times 64 = 36\,928$) ; pour d), $25\,088 \times 4\,096 + 4\,096$ ; pour e), ajoute les deux autres couches denses ; pour f) et g), divise ou multiplie par 4 octets ; pour h), compare ton résultat de g) et 527,8 : quelle unité donne le plus petit nombre ?

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

1 et 2. L'introduction et le §3 de l'article de 2015. 3. L'algorithme 2 et le paragraphe juste avant. 4. Le §3.2. 5. Le §4.2 et son tableau. 6. La figure 2 de Santurkar et coll. 7. Leur §3 (« smoothness », la constante de Lipschitz du gradient). 8. Sépare ce qui se mesure de ce qui s'interprète.

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

## Exercices du notebook

### Ex 20.13 — Premiers tenseurs PyTorch : créer, convertir, dtype, device 📦

<details><summary>Indice 1</summary>

Relis l'encadré « Rappel outil — les tenseurs PyTorch » du §20.2.1 de la fiche : la conversion depuis NumPy, le changement de dtype, l'envoi sur un appareil.

</details>
<details><summary>Indice 2</summary>

Trois étapes pour les images : convertir (`torch.from_numpy`), passer en réels (`.float()`) et diviser par 255, ajouter l'axe des canaux (`.unsqueeze(1)` ou `.reshape(64, 1, 28, 28)`). Les labels de `Y_13` sont déjà des `int64`. Un tenseur rend un nombre Python avec `.item()`. Pour c), fais l'essai dans une cellule.

</details>
<details><summary>Indice 3</summary>

```python
images_13 = torch.from_numpy(X_13).float().div(255).unsqueeze(1)
labels_13 = torch.from_numpy(Y_13)
images_dev_13 = images_13.to(DEVICE)
```
Pour a), la moyenne d'un tenseur, puis `.item()` ; pour b), `str(...)` du `.dtype` ; pour c), modifie `vec` puis regarde `t[0]`.

</details>

### Ex 20.14 — Prédire les formes produites par les couches utilitaires 🔮

<details><summary>Indice 1</summary>

Pour chaque couche, pose-toi deux questions : sur quels axes agit-elle, et que fait-elle sur chacun ? Les couches « 2d » de PyTorch sont faites pour des images `(N, C, H, W)`.

</details>
<details><summary>Indice 2</summary>

Un zero-padding ou un pooling 2-D ne touche ni à l'axe des exemples ni à celui des canaux. Pour un pooling, compte combien de positions de la fenêtre tiennent sur 5 pixels, selon le pas, et ce que fait l'arrondi au-dessus. `nn.Flatten()` garde l'axe 0 et met à plat **tout ce qui suit**. `nn.Upsample` agrandit les axes « spatiaux », ceux qui viennent après les canaux. `nn.Unflatten(dim, forme)` remplace un seul axe par plusieurs.

</details>
<details><summary>Indice 3</summary>

Pour a) : $(2, 3, 5, 5)$ devient $(2, 3, 5 + 2, 5 + 2)$, soit `"(2, 3, 7, 7)"`. Pour b) et c), calcule le nombre de positions d'une fenêtre de taille $k$ et de pas $s$ sur $n$ pixels, $\lfloor (n - k)/s \rfloor + 1$ (ou arrondi au-dessus avec `ceil_mode`) ; pour d), demande-toi ce qu'il reste à aplatir après l'axe 0 ; pour e), quels axes viennent après les canaux dans `(N, C, L)` ; pour f), remplace l'axe 1, de taille 12, par deux axes.

</details>

### Ex 20.15 — Autograd : retrouver les gradients du réseau minuscule du ch. 18 📦

<details><summary>Indice 1</summary>

Relis, dans la fiche, la section « PyTorch en pratique », point 1 (l'autograd), puis ta solution de ✏️ 18.4 : PyTorch calcule les mêmes nombres.

</details>
<details><summary>Indice 2</summary>

Dans `tiny_loss_15`, transforme `X_15` et `Y_15` en tenseurs `float64` (`torch.tensor`), puis écris la formule avec `@`, `**` et `.sum()`. Dans `Tiny15`, deux attributs `nn.Linear(2, 2, bias=False, dtype=torch.float64)`, et une passe avant qui les enchaîne. Pour d) et e), crée toi-même des feuilles avec `requires_grad=True` : deux `backward()` de suite pour d) ; pour e), un seul `backward()`, puis un `torch.optim.SGD([w1, w2], lr=0.1)` et son `step()`, et la loss recalculée.

</details>
<details><summary>Indice 3</summary>

```python
def tiny_loss_15(w1, w2):
    x = torch.tensor(X_15, dtype=torch.float64)
    y = torch.tensor(Y_15, dtype=torch.float64)
    return 0.5 * ((x @ w1 @ w2 - y) ** 2).sum()
```
Pour `Tiny15`, `forward` rend `self.out(self.hidden(x))`. Pour e) : `opt = torch.optim.SGD([w1, w2], lr=0.1)`, `tiny_loss_15(w1, w2).backward()`, `opt.step()`, puis `tiny_loss_15(w1, w2).item()`.

</details>

### Ex 20.16 — dropout_forward et dropout_backward 🔨

<details><summary>Indice 1</summary>

Relis les formules du dropout inversé (fiche §20.4.3) et ta solution de ✏️ 20.3 : le masque vaut 0 ou $1/(1 - p)$, et la passe arrière multiplie par le même masque.

</details>
<details><summary>Indice 2</summary>

Dans l'ordre : vérifie `p` ; convertis `x` en un **nouveau** tableau `float64` (`np.array(x, dtype=float)`, qui copie) ; traite les cas particuliers (évaluation ou `p = 0` : la copie et `None` ; `p = 1` : des zéros) ; sinon, tire `rng.random(x.shape)`, garde les valeurs `>= p`, et divise par `1 - p`. La passe arrière vérifie la forme du masque, puis multiplie.

</details>
<details><summary>Indice 3</summary>

```python
def dropout_forward(x, p=0.5, training=True, rng=None):
    if not 0.0 <= p <= 1.0:            # also refuses a nan
        raise ValueError(...)
    x = np.array(x, dtype=np.float64)  # a new array: the input is never modified
    # evaluation or p == 0: (x, None) ; p == 1: zeros for out and mask
    rng = np.random.default_rng() if rng is None else rng
    mask = (rng.random(x.shape) >= p) / (1.0 - p)
    return x * mask, mask
```
Dans `dropout_backward`, la copie de `dout` quand `mask` vaut `None`, sinon `dout * mask` après la vérification des formes.

</details>

### Ex 20.17 — layer_norm, l2_penalty et EarlyStopping 🔨

<details><summary>Indice 1</summary>

Relis la layer norm (fiche §20.4.8 et ✏️ 20.7), la pénalité L2 et l'early stopping (section « régulariser un réseau » ; ch. 9), puis la docstring d'`EarlyStopping` et ses attributs.

</details>
<details><summary>Indice 2</summary>

`layer_norm` : les statistiques sur `axis=-1` avec `keepdims=True`, la variance biaisée (`x.var(axis=-1, keepdims=True)`), puis `gamma` et `beta` après avoir vérifié leur forme `(x.shape[-1],)`. `l2_penalty` : une somme de carrés sur la liste, puis la liste des `lam * W`. `step` : décide d'abord si `value` est une amélioration (nan, première valeur finie, puis la comparaison stricte selon le mode et `min_delta`), met à jour les attributs dans les deux cas, puis incrémente `n_checks_` et décide de l'arrêt.

</details>
<details><summary>Indice 3</summary>

```python
def step(self, value, params=None):
    value = float(value)
    if math.isnan(value):
        better = False
    elif self.best_epoch_ == -1:       # no finite value seen yet
        better = True
    else:
        better = value < self.best_ - self.min_delta if self.mode == "min" else value > self.best_ + self.min_delta
    # better: best_, best_epoch_ (= n_checks_), counter_ = 0, improved_, and a deep copy of params if asked
    # otherwise: counter_ += 1, improved_ = False
    # then n_checks_ += 1, stopped_ = counter_ >= patience, and return bool(stopped_)
```
`restore` vérifie qu'il y a des paramètres gardés et que les formes correspondent, puis `p[...] = saved` pour chaque tableau.

</details>

### Ex 20.18 — Écrire sa classe nn.Module et compter ses paramètres 📦

<details><summary>Indice 1</summary>

Relis, dans la fiche, la section « PyTorch en pratique », point 2 (`nn.Module`), et l'exemple de la classe `MLP`.

</details>
<details><summary>Indice 2</summary>

Dans `__init__` : `super().__init__()` d'abord, puis un `nn.Flatten()` et un `nn.ModuleList` construit à partir des couples `(sizes[k], sizes[k + 1])`. Dans `forward` : aplatis, puis une boucle sur toutes les couches **sauf la dernière** avec une ReLU, et la dernière sans. Pour compter, `n_params(MLP18([...]))`.

</details>
<details><summary>Indice 3</summary>

```python
def __init__(self, sizes):
    super().__init__()
    self.flatten = nn.Flatten()
    self.layers = nn.ModuleList(nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:]))

def forward(self, x):
    x = self.flatten(x)
    # a ReLU after every layer of self.layers[:-1], then the last layer alone
```
Pour a), `params_222_18 = n_params(MLP18([2, 2, 2]))` ; de même pour b) et c).

</details>

### Ex 20.19 — Dataset et DataLoader : MNIST en mini-batches 📦

<details><summary>Indice 1</summary>

Relis, dans la fiche, la section « PyTorch en pratique », point 3 (`Dataset` et `DataLoader`), et les méthodes spéciales `__len__` et `__getitem__` (0A).

</details>
<details><summary>Indice 2</summary>

`__init__` garde les deux tableaux (en tenseurs, si tu veux) ; `__len__` rend le nombre d'images ; `__getitem__(i)` convertit **une** image : `float`, division par 255, ajout de l'axe des canaux, et rend le couple avec le label `i`. `make_loader_19` : un `DataLoader(dataset, batch_size=64, shuffle=True, generator=..., drop_last=drop_last)`. Pour a) à c), parcours le chargeur et compte, ou utilise `len(loader)`.

</details>
<details><summary>Indice 3</summary>

```python
def __getitem__(self, i):
    image = self.images[i].float().div(255).unsqueeze(0)    # (28, 28) -> (1, 28, 28)
    return image, self.labels[i]
```
Pour a) et b) : `sizes = [len(y) for _, y in make_loader_19(MNISTData19(X_19, Y_19), seed=0)]`, puis le nombre d'éléments de `sizes` et le dernier ; pour c), la même chose avec `drop_last=True`.

</details>

### Ex 20.20 — Zoo des couches : convolution, pooling, récurrente, utilitaires, bruit 📦

<details><summary>Indice 1</summary>

Relis les §20.4.5 à §20.4.8 de la fiche et leur tableau : chaque couche du livre a son nom dans PyTorch.

</details>
<details><summary>Indice 2</summary>

`nn.Conv2d(C_in, C_out, kernel_size, padding)`, `nn.MaxPool2d(2)` et `nn.AvgPool2d(2)` (appliqués directement à `SMALL_20`), `nn.RNN(input_size, hidden_size, batch_first=True)`. Pour `Noise20`, garde `sigma` dans `__init__` et teste `self.training` dans `forward` (`torch.randn_like(x)` tire un bruit de la forme de `x`). Pour `Zoo20`, un `nn.Sequential` pour les convolutions et un autre pour la tête, ou des attributs un par un.

</details>
<details><summary>Indice 3</summary>

```python
conv_20 = nn.Conv2d(1, 2, kernel_size=3, padding=1)

class Noise20(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        # the noise only when self.training is True
```
Dans `Zoo20`, la couche dense reçoit $16 \times 7 \times 7$ valeurs : 28 pixels divisés par 2, deux fois.

</details>

### Ex 20.21 — Reproduire la figure 20.17 : le plan de VGG16 couche par couche 🎨

<details><summary>Indice 1</summary>

Relis la description de VGG16 (fiche §20.6) et ✏️ 20.6, qui en fait le compte. La figure 20.17 du livre montre une boîte par couche, de gauche à droite.

</details>
<details><summary>Indice 2</summary>

Construis la liste des couches avec deux boucles (les blocs, puis les nombres de filtres de chaque bloc), en suivant le nombre de canaux d'entrée ; ajoute un `nn.MaxPool2d(2)` à la fin de chaque bloc, puis l'aplatissement et la tête. `plan_21` : une boucle sur les couches du `nn.Sequential`, qui applique chacune à `x` sous `torch.no_grad()` et note son nom et la forme obtenue. `draw_plan_21` : un `plt.Rectangle` par couche (`ax.add_patch`), une couleur choisie dans un dictionnaire selon le nom, et un `ax.text` pour la taille.

</details>
<details><summary>Indice 3</summary>

```python
layers, c_in = [], 3
for block in ([64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]):
    for c_out in block:
        layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU()]
        c_in = c_out
    # the pooling of the block
# then nn.Flatten(), the three dense layers with their ReLU and Dropout(0.5)
```
Pour a), cherche dans le plan la 3ᵉ ligne dont le nom est `"MaxPool2d"` ; pour b), divise les paramètres de la première `nn.Linear` par ceux du modèle entier.

</details>

### Ex 20.22 — batchnorm_forward et batchnorm_backward 🔨

<details><summary>Indice 1</summary>

Relis le §20.4.4 de la fiche, tes calculs de ✏️ 20.4 et ta preuve ∂ 20.8 : la passe avant, la mise à jour des statistiques glissantes et la formule de la passe arrière.

</details>
<details><summary>Indice 2</summary>

Une fonction privée choisit les axes des statistiques : `(0,)` pour un `x` 2-D, `(0, 2, 3)` pour un 4-D (sinon `ValueError`) ; une autre remet un vecteur de taille `C` à la forme diffusable `(1, C)` ou `(1, C, 1, 1)`. Vérifie toutes les formes avant de calculer. En entraînement : `mean` et `var` (biaisée) sur ces axes, `x_hat`, puis `running_mean *= 1 - momentum` et `running_mean += momentum * mean` (en place), de même pour la variance, avec le facteur $\frac{m}{m-1}$. Garde dans le cache `x_hat`, `1 / sqrt(var + eps)`, une copie de `gamma`, les axes, $m$ et le mode.

</details>
<details><summary>Indice 3</summary>

```python
def batchnorm_backward(dout, cache):
    axes, m = cache["axes"], cache["count"]
    dbeta = dout.sum(axis=axes)
    dgamma = (dout * cache["x_hat"]).sum(axis=axes)
    scale = shaped(cache["gamma"] * cache["inv_std"])          # (1, C) or (1, C, 1, 1)
    if not cache["training"]:
        return dout * scale, dgamma, dbeta
    # training: the formula of ∂ 20.8, with shaped(dbeta) and shaped(dgamma)
```

</details>

### Ex 20.23 — La boucle d'entraînement PyTorch complète sur MNIST 📦

<details><summary>Indice 1</summary>

Relis, dans la fiche, la section « PyTorch en pratique », point 4 (la boucle), et la figure de la boucle : cinq gestes par mini-batch, dans un ordre fixe.

</details>
<details><summary>Indice 2</summary>

`train_one_epoch_23` : `model.train()`, puis une boucle `for X, y in loader:` ; les données sur `device` ; la passe avant et la loss ; `zero_grad`, `backward`, `step` ; et deux compteurs, la somme de `loss.item() * len(y)` et le nombre d'exemples. `evaluate_23` : `model.eval()`, tout dans `with torch.no_grad():`, et trois compteurs (la loss pondérée, les bonnes réponses `(logits.argmax(dim=1) == y).sum().item()`, le nombre d'exemples).

</details>
<details><summary>Indice 3</summary>

```python
def train_one_epoch_23(model, loader, loss_fn, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        # the three steps of the update, in order
        total += loss.item() * len(y)
        count += len(y)
    return total / count
```

</details>

### Ex 20.24 — Chasse aux bugs PyTorch : cinq erreurs classiques 🐛

<details><summary>Indice 1</summary>

Relis le tableau des pièges de la fiche : les cinq erreurs y sont, parmi d'autres. Pour chaque point de la vérification, regarde le message, puis la ligne du code qui produit ce qu'il décrit.

</details>
<details><summary>Indice 2</summary>

Point 1 : que fait `torch.flatten` à l'axe des exemples ? Point 2 : que doit rendre un modèle dont la sortie va dans `cross_entropy` ? Point 3 : que fait `backward()` au `.grad` déjà présent, et quel appel manque ? Point 4 : quel est le type de `total` après `total += loss * len(y)` ? Point 5 : dans quel mode est le modèle pendant `evaluate_24` ?

</details>
<details><summary>Indice 3</summary>

La correction du point 1 : `x = torch.flatten(x, start_dim=1)`. Pour les autres : la dernière ligne de `forward` ; un appel à ajouter avant `loss.backward()` ; une méthode à appeler sur `loss` avant de l'additionner ; deux lignes au début d'`evaluate_24` (le mode du modèle, et un bloc qui désactive le calcul des gradients).

</details>

### Ex 20.25 — Courbes train/validation : overfitting, eval() oublié, fuite 📈

<details><summary>Indice 1</summary>

Relis, dans la fiche, l'encadré « Piège classique — oublier `model.eval()` » et la section sur la régularisation ; au ch. 9, la forme d'une courbe de validation qui surapprend.

</details>
<details><summary>Indice 2</summary>

Pour chaque entraînement, regarde trois choses : la loss de validation remonte-t-elle après un minimum ? est-elle au-dessus ou au-dessous de celle d'entraînement, et de combien ? est-elle régulière d'une epoch à l'autre ? Regarde aussi l'accuracy finale : est-elle réaliste pour des vêtements (les meilleurs modèles font environ 93 à 96 %) ?

</details>
<details><summary>Indice 3</summary>

Pour a) : cherche l'entraînement dont la loss de validation passe par un minimum net, puis remonte longtemps pendant que celle d'entraînement baisse ; pour b), l'epoch de ce minimum (`np.argmin(CURVES_25[...]["val_loss"]) + 1`). Pour les autres : une validation qui colle à l'entraînement jusqu'à une accuracy invraisemblable ; une validation au niveau de l'entraînement, mais qui saute d'une epoch à l'autre ; deux losses hautes et collées ; une validation qui reste sous l'entraînement sans remonter.

</details>

### Ex 20.26 — Un modèle ImageNet face à des images qu'il n'a jamais vues 🔮

<details><summary>Indice 1</summary>

Relis le §20.6 de la fiche : un classifieur ne peut répondre qu'avec ses classes, et sa softmax distribue toujours une probabilité de 1 entre elles.

</details>
<details><summary>Indice 2</summary>

Pour a), pense à ce que deviennent des photos de 32 × 32 pixels agrandies à 224 × 224, et au nombre de classes d'animaux voisines parmi les 1 000. Pour b) et c), pense à ce que la probabilité la plus haute mesure, et à ce qu'un modèle a vu pendant son entraînement. Pour d), demande-toi si du bruit tiré au hasard ressemble davantage à une classe qu'à une autre. Pour e), relie ta réponse à c).

</details>
<details><summary>Indice 3</summary>

Écris, pour chaque prédiction, une phrase de justification avant de choisir : a) « une photo de 32 × 32 pixels, agrandie à 224 × 224, garde-t-elle les détails qui distinguent un chat d'un autre animal ? » ; b) « une softmax sur 1 000 classes, sans aucune classe proche, donne… » ; c) « une photo réelle, même floue, ressemble plus ou moins qu'un bruit à une image d'ImageNet… » ; d) « deux bruits différents ont-ils des textures différentes ? » ; e) « une photo de CIFAR-10 a une probabilité la plus haute de l'ordre de… ».

</details>

### Ex 20.27 — Sauvegarder, recharger et reproduire un entraînement 🛠️

<details><summary>Indice 1</summary>

Relis, dans la fiche, la section « PyTorch en pratique », point 5 (sauvegarder), et l'encadré sur la reproductibilité : un entraînement dépend des poids, de l'optimiseur, de l'ordre des données et du hasard du dropout.

</details>
<details><summary>Indice 2</summary>

Chaque objet a sa paire de méthodes : `state_dict()` et `load_state_dict(...)` pour le modèle et l'optimiseur, `get_state()` et `set_state(...)` pour un `torch.Generator`, `torch.get_rng_state()` et `torch.set_rng_state(...)` pour le générateur global. Le dictionnaire ne contient que des tenseurs, des dictionnaires et des nombres : `torch.load` le relit en mode sûr.

</details>
<details><summary>Indice 3</summary>

```python
def save_checkpoint_27(path, model, optimizer, epoch, generator):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "generator": generator.get_state(), "torch_rng": torch.get_rng_state()}, path)
```
`load_checkpoint_27` fait l'inverse, clé par clé, après `checkpoint = torch.load(path)`, et rend `checkpoint["epoch"]`.

</details>

### Ex 20.28 — Dropout, batchnorm, AdamW et early stopping sur Fashion-MNIST 📦

<details><summary>Indice 1</summary>

Relis ta solution de 20.23 (les deux fonctions) et la note de 20.17 sur `EarlyStopping` avec un modèle PyTorch (`improved_` et `copy.deepcopy(model.state_dict())`).

</details>
<details><summary>Indice 2</summary>

`make_model_28` : `torch.manual_seed(seed)`, puis une liste de couches construite par une boucle sur les deux couches cachées, un `nn.Sequential`, et `.to(DEVICE)`. `fit_28` : crée l'optimiseur, l'early stopping, la loss et l'historique **avant** la boucle ; à chaque epoch, entraîne, évalue, note, puis `stop = stopper.step(val_loss)` ; si `stopper.improved_`, copie les poids ; si `stop`, sors de la boucle ; après la boucle, recharge les meilleurs poids.

</details>
<details><summary>Indice 3</summary>

```python
for epoch in range(1, max_epochs + 1):
    history["train_loss"].append(train_one_epoch_23(model, train_loader_28(epoch), loss_fn, optimizer, DEVICE))
    val_loss, val_acc = evaluate_23(model, VAL_LOADER_28, loss_fn, DEVICE)
    # note them, then the early stopping:
    stop = stopper.step(val_loss)
    if stopper.improved_:
        best_state = copy.deepcopy(model.state_dict())
    # ...
model.load_state_dict(best_state)
```

</details>

### Ex 20.29 — Ablation : qu'est-ce qui retarde l'overfitting ? 🔬

<details><summary>Indice 1</summary>

Relis, dans la fiche, le paragraphe sur l'ablation : on change une seule chose à la fois, tout le reste égal, et l'on compare les courbes de validation.

</details>
<details><summary>Indice 2</summary>

La structure de `run_29` est celle de `fit_28` sans early stopping : une graine, un modèle construit couche par couche (les couches optionnelles ajoutées par des `if`), un optimiseur, un chargeur avec son générateur, puis `epochs` tours de `train_one_epoch_23` et `evaluate_23`. Pour lire le tableau, compare chaque ligne à la ligne « rien », colonne par colonne.

</details>
<details><summary>Indice 3</summary>

```python
torch.manual_seed(seed)
layers = [nn.Flatten()]
for n_in, n_out in [(784, 256), (256, 256)]:
    layers.append(nn.Linear(n_in, n_out))
    if batchnorm:
        layers.append(nn.BatchNorm1d(n_out))
    # the ReLU, then the dropout if dropout > 0
model = nn.Sequential(*layers, nn.Linear(256, 10)).to(DEVICE)
```

</details>

### Ex 20.30 — Un réseau de 30 couches : connexions résiduelles et LayerNorm 🔬

<details><summary>Indice 1</summary>

Relis la section de la fiche sur les connexions résiduelles et ✏️ 20.7 : le gradient qui traverse $L$ couches est un produit de $L$ facteurs.

</details>
<details><summary>Indice 2</summary>

Range les blocs dans deux `nn.ModuleList` (les normalisations, avec `nn.Identity()` quand il n'y en a pas, et les couches denses), ou dans une liste de petits modules. Dans `forward`, une boucle calcule `h`, puis choisit entre `x + h` et `h`. `grad_ratio_30` : remets les gradients à zéro (`model.zero_grad()`), une passe avant, la cross-entropy, `backward()`, puis le rapport des normes `.grad.norm()`.

</details>
<details><summary>Indice 3</summary>

```python
def forward(self, x):
    x = self.inp(x)
    for norm, linear in zip(self.norms, self.linears):
        h = torch.relu(linear(norm(x)))
        x = x + h if self.residual else h
    return self.out(x)
```
Pour le rapport : `(model.inp.weight.grad.norm() / model.out.weight.grad.norm()).item()`.

</details>

### Ex 20.31 — Défi Fashion-MNIST : au moins 90 % avec un MLP de 300 k paramètres au plus 🏆

<details><summary>Indice 1</summary>

Pars de ton modèle de 20.28 : il fait déjà environ 86 % avec 10 000 images. Que changent les 60 000 images, et un réglage plus soigné ? Garde une validation (les 10 000 dernières images d'entraînement) pour tous tes essais.

</details>
<details><summary>Indice 2</summary>

Les leviers, du plus simple au plus fin : toutes les données, plus d'epochs (le temps le permet : une epoch prend quelques secondes), la batchnorm, un dropout léger, AdamW et un **planning** du learning rate (ch. 19 : un cosinus, ou `torch.optim.lr_scheduler.OneCycleLR`, qui monte puis descend), des mini-batches de 128 à 256. Mesure chaque changement sur la validation, un à la fois (20.29), et regarde le temps.

</details>
<details><summary>Indice 3</summary>

Un squelette :
```python
def train_31(X, y):
    torch.manual_seed(...)
    model = ...                                   # 784 -> ... -> 10, at most 300 000 parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=..., weight_decay=...)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=..., total_steps=epochs * batches_per_epoch)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(batch):
            # forward, loss, zero_grad, backward, step, then scheduler.step() after EACH update
    return model.eval()
```

</details>
