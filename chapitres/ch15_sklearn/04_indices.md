# 15 · scikit-learn — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ Papier-crayon](#papier) · [🧮 🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 15.Q1 — Coder soi-même ou utiliser une bibliothèque ?

<details><summary>Indice 1</summary>

Relis le §15.1 de la fiche (le choix du livre) et le §15.2 (ce qu'est scikit-learn, et ce qu'il n'est pas).

</details>
<details><summary>Indice 2</summary>

Pour b), sépare ce qu'une bibliothèque **garantit** (la qualité de son code) de ce qu'elle ne peut pas savoir à ta place (tes données, tes choix). Pour d), demande-toi quel outil le workbook réserve aux réseaux de neurones à partir du ch. 20.

</details>
<details><summary>Indice 3</summary>

a) Le livre explique pourquoi il prend une bibliothèque plutôt que d'écrire tout lui-même : l'affirmation est donc **fausse**. b) Pour chaque option, demande-toi si du code peut la garantir sans connaître tes données. c) Relis la première phrase du §15.2. d) Quel framework de deep learning le workbook utilise-t-il ?

</details>

### 15.Q2 — Importer proprement et lire une dépréciation

<details><summary>Indice 1</summary>

Relis le §15.3 de la fiche : les deux façons d'importer, puis le paragraphe « Suivre les dépréciations ».

</details>
<details><summary>Indice 2</summary>

Pour a), demande-toi quel **nom** l'instruction `from sklearn import linear_model` crée dans ton programme. Pour b) à d), relis ce que le §15.3 dit du sort d'un argument déprécié. Pour e), lis ce que la fiche dit de l'adresse « stable ».

</details>
<details><summary>Indice 3</summary>

a) L'instruction crée le nom `linear_model`, un module : la classe s'atteint donc par `linear_model.Ridge()`, l'option **B**. b) Lis le message mot à mot : « *is deprecated in 1.6 and will be removed in 1.8* ». c) Que devient un argument déprécié deux versions plus tard ? d) Quel réflexe évite que le code casse le jour du retrait ? e) Quelle version décrit l'adresse « stable » en 2026, et laquelle est installée sur Colab ?

</details>

### 15.Q3 — Hyperparamètres au constructeur, attributs appris avec « _ »

<details><summary>Indice 1</summary>

Relis le §15.4.1 (ce que fait le constructeur) et le §15.4.2 (ce que fait `fit`, et le tiret bas).

</details>
<details><summary>Indice 2</summary>

Un hyperparamètre, c'est toi qui le fixes avant l'apprentissage ; un attribut qui finit par `_` est appris par `fit`. Pour d), relis l'encadré sur `clone` (§15.9.3).

</details>
<details><summary>Indice 3</summary>

a) `alpha` est donné au constructeur, avant toute donnée : c'est donc un réglage que tu choisis, un hyperparamètre. b) `coef_` finit par un tiret bas : qu'est-ce que cela dit sur le moment où il existe ? c) Que renvoie `get_params` ? d) La copie que fait `clone` garde-t-elle ce que l'objet a appris ? e) Le constructeur voit-il des données ?

</details>

### 15.Q4 — Ce que renvoient fit et predict

<details><summary>Indice 1</summary>

Relis les §15.4.2 et §15.4.3 : ce que renvoie `fit`, la forme de `X`, ce que renvoie `predict`.

</details>
<details><summary>Indice 2</summary>

Pour a), pense à l'écriture enchaînée `Ridge().fit(X, y).predict(X_new)`. Pour b) et c), relis ce que le §15.4.2 dit de la forme de `X` et le §15.4.3 de ce que renvoie `predict`. Pour d), relis l'encadré 🕰️ du §15.4.3.

</details>
<details><summary>Indice 3</summary>

a) Pour qu'on puisse écrire `.predict(...)` juste après `.fit(...)`, `fit` doit renvoyer un objet qui a une méthode `predict` : l'objet `ridge` lui-même, l'option **C**. b) Une ligne par exemple, une colonne par feature : combien de lignes, combien de colonnes ? c) Une valeur par exemple, dans un tableau à une dimension : combien de valeurs ? d) Quelle forme `predict` exige-t-il pour `X` (encadré 🕰️ du §15.4.3) ? e) Relis ce que fait un second appel à `fit`.

</details>

### 15.Q5 — decision_function ou predict_proba ?

<details><summary>Indice 1</summary>

Relis le tableau du §15.4.4 et l'encadré ⚠️ qui le suit.

</details>
<details><summary>Indice 2</summary>

Pour a) à c), relis la ligne de chaque méthode dans le tableau du §15.4.4 : de quoi dépend le nombre de colonnes ? Pour d), applique la sigmoïde de l'encadré 🧮. Pour e) et f), relis le paragraphe « Tous les classifieurs n'ont pas ces méthodes ».

</details>
<details><summary>Indice 3</summary>

a) Deux classes, quatre exemples : un seul score par exemple, dans un tableau à une dimension, de forme `(4,)` : l'option **A**. b) Combien de probabilités faut-il par exemple pour que chaque ligne somme à 1 ? c) Que dit la dernière colonne du tableau, pour $K > 2$ ? d) Calcule $\sigma(0) = 1/(1 + e^{0})$. e) Ce classifieur a-t-il une lecture probabiliste de son score ? f) Que dit la documentation de l'accord entre `predict` et `predict_proba` d'un SVC ?

</details>

### 15.Q6 — KMeans : n_clusters, labels_ et choix de k

<details><summary>Indice 1</summary>

Relis le §15.5 de la fiche et son encadré sur l'inertie et la silhouette.

</details>
<details><summary>Indice 2</summary>

La fiche donne la valeur par défaut de `n_clusters` et dit où `KMeans` trouve le nombre de features. Pour d), quelle valeur de $k$ donne la plus petite inertie possible, et est-ce un bon découpage ?

</details>
<details><summary>Indice 3</summary>

a) La parenthèse qui suit `n_clusters` au §15.5 donne le défaut : **8**. b) Le constructeur reçoit-il des données ? c) Que contient l'attribut `labels_` ? d) Si $k = n$, chaque point a son propre centre : que vaut l'inertie, et ce découpage a-t-il un intérêt ? e) Lequel des trois algorithmes regroupe par densité ?

</details>

### 15.Q7 — fit, transform, fit_transform, inverse_transform

<details><summary>Indice 1</summary>

Relis la règle d'or du §15.6 et l'encadré 🧮 sur la mise à l'échelle min-max.

</details>
<details><summary>Indice 2</summary>

Le scaler a appris $m = 20$ et $M = 60$ : toute valeur se transforme par $x' = (x - m)/(M - m)$, et s'inverse par $x = m + (M - m)\,x'$, même si elle sort de l'étendue de l'entraînement.

</details>
<details><summary>Indice 3</summary>

a) On n'apprend rien sur le test : on applique ce qui a été appris, avec `transform`, l'option **B**. b) $(70 - 20)/40$. c) $(10 - 20)/40$ : attention au signe. d) $20 + 40 \times 0{,}75$. e) Relis l'encadré ⚠️ du §15.6.1.

</details>

### 15.Q8 — Ce que la PCA fait vraiment aux features

<details><summary>Indice 1</summary>

Relis le §15.7 de la fiche et son encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

Pour a) et b), relis ce que contient `components_` ; pour c) et d), ce que fait `whiten=True` à chaque composante ; pour e), si la PCA est supervisée.

</details>
<details><summary>Indice 3</summary>

a) Une composante est une direction dans l'espace des trois features : la valeur d'un exemple sur cette direction mélange ses trois coordonnées centrées, c'est une combinaison linéaire, l'option **B**. b) La PCA garde-t-elle une feature d'origine ? c) Que vaut la variance d'une composante divisée par son écart-type ? d) Diviser un vecteur par un nombre positif change-t-il sa direction ? e) La PCA est-elle une méthode supervisée ?

</details>

### 15.Q9 — Un ensemble est un estimateur comme les autres

<details><summary>Indice 1</summary>

Relis le §15.8 de la fiche, son code et ses deux encadrés.

</details>
<details><summary>Indice 2</summary>

Pour b) et c), relis l'encadré 🕰️ et le commentaire de la dernière ligne du code. Pour d) et e), relis l'encadré ⚠️ : `n_estimators` est-il un nombre exact, et comment AdaBoost pondère-t-il ses membres ?

</details>
<details><summary>Indice 3</summary>

a) Le §15.8 commence par la réponse : un ensemble se manipule comme un modèle seul, avec `fit` et `predict` ; c'est **vrai**. b) Quel nom remplace `base_estimator` depuis la version 1.2 ? c) Que crée `AdaBoostClassifier()` sans argument ? d) `n_estimators` est-il un nombre exact ou une borne ? Relis l'encadré ⚠️. e) Que fait le poids $\alpha_m$ dans le vote d'AdaBoost (ch. 14) ?

</details>

### 15.Q10 — Grille ou recherche aléatoire ?

<details><summary>Indice 1</summary>

Relis les §15.9, §15.9.2 et §15.9.4, et la figure de la grille et du hasard.

</details>
<details><summary>Indice 2</summary>

Un paramètre est appris, un hyperparamètre choisi. Une grille essaie toutes les combinaisons : leur nombre est un **produit**. Pour e), compte les valeurs de l'hyperparamètre important que voit chaque méthode sur la figure. Pour f), relis l'encadré sur les distributions.

</details>
<details><summary>Indice 3</summary>

a) C'est toi qui fixes le nombre de clusters d'un k-means, avant `fit` : c'est un hyperparamètre, l'option **B**. b) La recherche change-t-elle le temps d'un entraînement ? c) Multiplie les trois nombres de valeurs. d) Relis la valeur par défaut de `n_iter`. e) Une grille 3 × 3 répète chaque valeur de l'hyperparamètre important sur l'autre axe : combien de valeurs distinctes en essaie-t-elle, contre neuf tirages au hasard ? f) Entre 0,001 et 1 000, quelle part d'un tirage uniforme tombe sous 1 ?

</details>

### 15.Q11 — Pipeline : step__param et absence de fuite

<details><summary>Indice 1</summary>

Relis le paragraphe « Nommer les hyperparamètres d'un pipeline » (§15.9.5) et le §15.9.7.

</details>
<details><summary>Indice 2</summary>

Le nom d'un hyperparamètre dans une grille sur un pipeline s'écrit : le nom de l'**étape** que tu as choisi, **deux** tirets bas, le nom de l'hyperparamètre. Pour b) et c), relis ce que fait `cross_val_score` de l'estimateur qu'il reçoit (§15.9.7).

</details>
<details><summary>Indice 3</summary>

a) L'étape du SVC s'appelle `"clf"` dans ce pipeline : le nom de l'étape, deux tirets bas, `C`, soit `"clf__C"`, l'option **D**. b) Que fait `cross_val_score` d'un pipeline à chaque tour ? c) Le fold de validation doit-il servir à apprendre la transformation ? d) Relis comment `make_pipeline` nomme ses étapes. e) Le pipeline du livre finit-il par un transformateur ?

</details>

### 15.Q12 — Datasets, train_test_split et documentation

<details><summary>Indice 1</summary>

Relis les §15.10 et §15.11 de la fiche, et l'encadré 🕰️ sur Boston Housing.

</details>
<details><summary>Indice 2</summary>

La fiche donne le défaut de `test_size`, ce que fait `shuffle`, le rôle de `stratify` et la règle d'arrondi de la taille du test. Pour f), relis la liste des générateurs `make_*`.

</details>
<details><summary>Indice 3</summary>

a) La phrase sur `test_size` au §15.11 donne le défaut, une proportion : **0,25**. b) Que montre l'exemple du cercle généré dans l'ordre ? c) Quel argument garde les proportions des classes ? d) Que devient `load_boston` en 1.2 ? e) Calcule $0{,}3 \times 125$, puis applique la règle d'arrondi de l'énoncé. f) Relis la liste des générateurs `make_*` du §15.10 et ce que dessine chacun.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 15.R1 — Ch. 14 : forêt aléatoire ou gradient boosting ?

<details><summary>Indice 1</summary>

Relis le tableau « Bilan : bagging ou boosting ? » de la fiche du ch. 14.

</details>
<details><summary>Indice 2</summary>

Pour chaque méthode, demande-toi si un arbre a besoin des précédents, et quel réglage fixe la taille de chaque pas.

</details>
<details><summary>Indice 3</summary>

a) La forêt peut entraîner tous ses arbres en même temps, puisqu'aucun n'a besoin des autres : l'option **A**. b) Relis la ligne « plus de membres » du tableau. c) Le boosting part d'apprenants faibles : leur défaut est-il le biais ou la variance ? d) Quel réglage multiplie la contribution de chaque arbre ? e) Quelle méthode donne un score out-of-bag ?

</details>

### 15.R2 — Ch. 12 : où ajuster le scaler dans une validation croisée ?

<details><summary>Indice 1</summary>

Relis le §12.10 de la fiche du ch. 12 (transformations et validation croisée).

</details>
<details><summary>Indice 2</summary>

Demande-toi quelles lignes ont servi à calculer la transformation, et lesquelles servent ensuite à noter le modèle. Pour c), un `StandardScaler` calcule $z = (x - \mu)/\sigma$ avec la moyenne et l'écart-type de l'entraînement.

</details>
<details><summary>Indice 3</summary>

a) Une standardisation de tout le dataset calcule la moyenne avec les lignes des folds de validation : ces lignes ont servi à préparer le modèle qu'on note sur elles, c'est une fuite ; **vrai**. b) Quelles lignes a-t-on le droit d'utiliser pour apprendre, à chaque tour ? c) $(65 - 50)/10$. d) Relis ce qu'a montré l'expérience du ch. 8 (sélection sur du bruit). e) Laquelle des deux étapes a besoin d'une statistique calculée sur les données ?

</details>

### 15.R3 — Ch. 8 : combien de modèles pour une validation croisée à 5 folds ?

<details><summary>Indice 1</summary>

Relis le §8.5.1 de la fiche du ch. 8 (la validation croisée à $k$ folds).

</details>
<details><summary>Indice 2</summary>

Un tour par fold ; à chaque tour, un fold est mis de côté pour la validation et les autres servent à l'entraînement. Pour d), `KFold` donne un exemple de plus aux premiers folds quand $k$ ne divise pas $n$.

</details>
<details><summary>Indice 3</summary>

a) Un modèle par tour, et un tour par fold : 5 folds donnent donc 5 modèles. b) Un exemple appartient à un seul fold : dans combien de tours ce fold est-il mis de côté ? c) Dans combien de tours ne l'est-il pas ? d) Divise 1 003 par 5 : le quotient et le reste. e) Leave-one-out : combien de folds ? f) À quoi sert la stratification ?

</details>

<a id="papier"></a>

## ✏️ Papier-crayon

### Ex 15.1 — Compter les modèles entraînés par une recherche en grille ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « combien de modèles entraîne une recherche ? » (§15.9.2) : le nombre de candidats, multiplié par le nombre de folds, plus le réentraînement final.

</details>
<details><summary>Indice 2</summary>

Compte d'abord les candidats (le produit des nombres de valeurs ; pour une liste de grilles, la somme des produits de chaque grille), puis multiplie par $k$, puis ajoute 1 si `refit` est vrai. Pour h), chaque fold extérieur lance une recherche **complète**.

</details>
<details><summary>Indice 3</summary>

a) $4 \times 3 = 12$ candidats. b) Chaque candidat est noté sur 5 folds : multiplie a) par 5. c) Ajoute le réentraînement final. d) Même calcul avec 3 folds. e) Trois hyperparamètres, à 5, 2 et 3 valeurs, puis 4 folds et le réentraînement final. f) Deux grilles : $1 \times 3$ et $1 \times 3 \times 2$ candidats ; additionne, multiplie par 5, sans réentraînement final. g) $n_{\text{iter}} \times k + 1$. h) Trois recherches complètes, comme en c).

</details>

### Ex 15.2 — De decision_function à predict_proba ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la sigmoïde et le softmax (§15.4.4), et la règle de `predict` à deux classes.

</details>
<details><summary>Indice 2</summary>

$\hat p = 1/(1 + e^{-s})$ ; le score qui correspond à une probabilité $\hat p$ est le logit $\ln\big(\hat p / (1 - \hat p)\big)$. À deux classes, `predict` renvoie la classe 1 quand le score est **strictement** positif. Pour h), $\hat p_k = e^{s_k} / \sum_j e^{s_j}$.

</details>
<details><summary>Indice 3</summary>

a) $\sigma(2) = 1/(1 + e^{-2}) = 1/(1 + 0{,}1353\ldots) \approx 0{,}8808$. b) Même calcul avec $s = -1$. c) $s = 0$. d) Applique la règle du signe aux cinq scores, en faisant attention au score nul. e) $\ln(0{,}9/0{,}1)$. f) $\ln(0{,}8/0{,}2)$. g) Compare chaque score au seuil de f). h) $e^2 / (e^2 + e^1 + e^{-1})$. i) Que devient la fraction si l'on multiplie chaque exponentielle par $e^5$ ? j) Relis pourquoi `RidgeClassifier` n'a pas de `predict_proba`.

</details>

### Ex 15.3 — Ramener une droite ajustée dans l'espace d'origine ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la mise à l'échelle min-max et son inverse (§15.6), puis le §15.6.1 : on ramène une droite en ramenant **deux** de ses points.

</details>
<details><summary>Indice 2</summary>

Ici $x' = (x + 3)/8$ et $y = 10 + 20\,y'$. Pour une valeur de $x$ : transforme-la en $x'$, applique la droite pour obtenir $y'$, puis ramène $y'$ en $y$. La pente d'origine se lit entre les deux extrémités ramenées.

</details>
<details><summary>Indice 3</summary>

a) $x' = (1 - (-3))/(5 - (-3)) = 4/8 = 0{,}5$. b) $y' = 0{,}6 \times x' + 0{,}25$. c) $y = 10 + 20\,y'$. d) En $x' = 0$, $y' = 0{,}25$ ; ramène-le. e) En $x' = 1$, $y' = 0{,}6 + 0{,}25$ ; ramène-le. f) La différence des $y$ de d) et e), divisée par l'étendue de $x$. g) Repars de l'extrémité gauche ($x = -3$) avec la pente de f), ou transforme $x = 0$. h) $(11 + 3)/8$ : la valeur sort de $[0, 1]$, et c'est normal. i) Applique la droite à h), puis ramène $y'$. j) Fais le calcul : `inverse_transform` sur le point $(0{,}6 ;\ 0{,}25)$ donne $(-3 + 8 \times 0{,}6 ;\ 10 + 20 \times 0{,}25)$ ; compare à f) et g).

</details>

### Ex 15.4 — Combien de features polynomiales ? ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « combien de features polynomiales ? » (§15.9.5) : $\binom{p + d}{d}$, constante comprise.

</details>
<details><summary>Indice 2</summary>

Pour un degré **exactement** $d$, compte les monômes de degré au plus $d$, moins ceux de degré au plus $d - 1$ (ou applique la même idée des étoiles et des barres sans la variable fictive : $\binom{p + d - 1}{d}$). Avec `interaction_only=True`, il n'y a pas de carré : un monôme de degré $j$ est un choix de $j$ variables **différentes**, soit $\binom{p}{j}$.

</details>
<details><summary>Indice 3</summary>

a) $\binom{2 + 2}{2} = \binom{4}{2} = 6$ : $1$, $x$, $y$, $x^2$, $xy$, $y^2$. b) $\binom{7}{5}$. c) $\binom{2 + 4 - 1}{4}$, ou énumère $x^4$, $x^3 y$… d) $\binom{10}{2}$. e) $\binom{11}{3}$. f) $\binom{23}{3} - 1$. g) $\binom{5}{0} + \binom{5}{1} + \binom{5}{2}$. h) Nombre de lignes × nombre de colonnes × 8 octets, divisé par $10^9$. i) Calcule $\binom{10 + d}{d}$ pour $d = 1, 2, 3, \ldots$ jusqu'à dépasser 1 000.

</details>

<a id="reflexion"></a>

## 🧮 🗣️ ⚖️ 📄 Réflexion

### Ex 15.5 — Budget d'une recherche d'hyperparamètres 🧮

<details><summary>Indice 1</summary>

Relis les deux encadrés 🧮 du §15.9.2 (le nombre d'entraînements, la probabilité qu'au moins un essai soit bon) et l'encadré 🕰️ du §15.9.4 sur le successive halving.

</details>
<details><summary>Indice 2</summary>

Une grille de 5 hyperparamètres à 5 valeurs a $5^5$ combinaisons ; chacune coûte 5 entraînements de 4 minutes. Pour le successive halving, écris pour chaque tour le nombre de candidats et la part des données qu'ils reçoivent, puis additionne les produits.

</details>
<details><summary>Indice 3</summary>

a) $5^5 \times 5 = 3\,125 \times 5 = 15\,625$ entraînements. b) Multiplie par 4 minutes, puis divise par $60 \times 24$. c) Divise b) par 8. d) $60 \times 5 \times 4$ minutes, divisées par 8 cœurs, puis par 60. e) $1 - 0{,}95^{60}$. f) Le plus petit entier $n$ tel que $n \ge \ln 0{,}01 / \ln 0{,}95$. g) Tours : $81 \times \frac{1}{81}$, $27 \times \frac{3}{81}$, $9 \times \frac{9}{81}$, $3 \times \frac{27}{81}$, $1 \times 1$ ; additionne. h) $81$ divisé par g).

</details>

### Ex 15.6 — Le Pipeline expliqué en 5 lignes 🗣️

<details><summary>Indice 1</summary>

Relis les §15.9.5 et §15.9.7 de la fiche, et la figure des folds avec et sans pipeline.

</details>
<details><summary>Indice 2</summary>

Une bonne image : une recette ou une chaîne de montage, dont chaque poste est « réglé » sur les seuls ingrédients d'entraînement. L'essentiel à dire : à l'entraînement, chaque étape apprend puis transforme ; à la prédiction, elle ne fait que transformer ; dans une validation croisée, tout est réappris à chaque tour, sur les seuls folds d'entraînement.

</details>
<details><summary>Indice 3</summary>

Plan en cinq lignes : (1) l'image ; (2) ce que fait `fit` du pipeline (`fit_transform` sur chaque transformateur, `fit` sur le modèle) et ce que fait `predict` (`transform` partout, puis `predict`) ; (3) ce qui se passe dans `cross_val_score(pipe, X, y)`, tour par tour ; (4) pourquoi standardiser avant la validation croisée laisse le fold de validation influencer la préparation ; (5) quand l'effet est petit (une mise à l'échelle sur beaucoup d'exemples) et quand il est énorme (une étape qui regarde la cible).

</details>

### Ex 15.7 — Boston Housing : un dataset retiré pour raisons éthiques ⚖️

<details><summary>Indice 1</summary>

Lis l'avertissement de la documentation de `load_boston` (version 1.1) et l'article de Carlisle : ils répondent aux questions 1 et 2.

</details>
<details><summary>Indice 2</summary>

Pour 1, regarde la forme de la fonction $1\,000\,(B_k - 0{,}63)^2$ : où est son minimum, et que vaut-elle pour deux valeurs de $B_k$ symétriques autour de 0,63 ? Pour 2, demande-toi ce que d'autres colonnes, la cible et l'usage même du dataset en cours transmettent. Pour 4, ouvre `data/cards/california_housing.md`.

</details>
<details><summary>Indice 3</summary>

Structure possible : (1) une parabole de minimum en 0,63, qui encode l'hypothèse d'un effet de la ségrégation sur les prix, et qui envoie deux villes différentes sur la même valeur ; (2) les autres colonnes, la cible et les résultats publiés restent construits dans le même cadre, et l'habitude d'utiliser ce dataset sans contexte est elle-même le problème ; (3) le cycle de dépréciation : un avertissement, des alternatives, du temps pour migrer ; (4) la date du recensement, l'unité (le groupe d'îlots), le plafond de la cible ; (5) qui a collecté, pourquoi, avec quelles variables construites et quelles limites documentées.

</details>

### Ex 15.8 — Lire l'article sur la conception de l'API de scikit-learn 📄

<details><summary>Indice 1</summary>

La section 2.1 de l'article énonce les principes ; les sections 2.3 à 2.5 décrivent les trois interfaces ; la section 3 les méta-estimateurs, les pipelines et la sélection de modèles.

</details>
<details><summary>Indice 2</summary>

Pour chaque principe, cherche un exemple dans la fiche : l'interface commune (`fit`, `predict`, `transform`), les attributs publics (`alpha`, `coef_`), les données en tableaux NumPy, les pipelines, les valeurs par défaut. Pour la question 6, cherche les limites dans les sections 4 et 6 de l'article (implémentation, directions futures), puis compare-les aux encadrés 🕰️ de la fiche.

</details>
<details><summary>Indice 3</summary>

Plan : (1) les cinq principes, chacun avec un exemple du chapitre ; (2) estimateur (`fit`), prédicteur (`predict`, `score`), transformateur (`transform`), et pourquoi le constructeur ne fait que ranger ; (3) des tableaux NumPy ou des matrices creuses SciPy, et ce que cela apporte (pas de conversion, l'écosystème NumPy) ; (4) les méta-estimateurs qui reçoivent un estimateur, et ce que `get_params`, `set_params` et `clone` leur permettent ; (5) le *duck typing*, et ce qui manque à une classe de `mylearn` pour entrer dans `cross_val_score` ; (6) les limites citées par l'article (sauvegarde, parallélisme, algorithmes absents, représentation des données), ce que la bibliothèque a ajouté depuis, et la limite qu'il ne cite pas (des données hétérogènes, avec des noms de colonnes).

</details>

<a id="entretien"></a>

## 💼 Entretien

### 15.E1 — fit, transform, predict : pourquoi jamais fit sur le test ?

<details><summary>Indice 1</summary>

Relis la règle d'or du §15.6 et le §15.9.7.

</details>
<details><summary>Indice 2</summary>

Définis chaque méthode en une phrase, puis explique ce qu'un `fit` sur le test change : la transformation n'est plus la même que celle de l'entraînement, et le test influence le modèle.

</details>
<details><summary>Indice 3</summary>

Plan : les quatre méthodes ; l'exemple du scaler (des statistiques de l'entraînement, appliquées partout) ; deux conséquences d'un `fit` sur le test (un modèle qui ne reçoit plus les nombres qu'il a appris à lire, une évaluation optimiste) ; le pipeline qui l'empêche, en validation croisée comme en production.

</details>

### 15.E2 — Grille, hasard ou optimisation bayésienne ?

<details><summary>Indice 1</summary>

Relis les §15.9.2 à §15.9.4, leurs encadrés et le piège « test veut dire validation ».

</details>
<details><summary>Indice 2</summary>

Avec une dizaine d'hyperparamètres, une grille est hors de prix ; compare la recherche au hasard (et son argument de Bergstra et Bengio) aux méthodes qui apprennent des essais précédents. N'oublie pas comment évaluer honnêtement le modèle retenu.

</details>
<details><summary>Indice 3</summary>

Plan : le coût d'une grille (un produit) ; la recherche au hasard comme point de départ, avec des distributions log ; le successive halving ou Optuna pour aller plus loin ; un jeu de test mis de côté avant la recherche, ou une validation croisée imbriquée, pour l'évaluation finale.

</details>

### 15.E3 — Pipeline et ColumnTransformer sur des données hétérogènes

<details><summary>Indice 1</summary>

Relis la section « Au-delà du livre : les données hétérogènes ».

</details>
<details><summary>Indice 2</summary>

Une branche par type de colonne (imputation, puis mise à l'échelle ou encodage), réunies par un `ColumnTransformer`, le tout suivi du modèle dans un pipeline.

</details>
<details><summary>Indice 3</summary>

Plan : la branche numérique, la branche catégorielle et leurs réglages (`handle_unknown`), le `ColumnTransformer` et ses sélecteurs, le pipeline final, puis les deux raisons (pas de fuite en validation croisée, un seul objet à régler et à déployer).

</details>

### 15.E4 — predict_proba donne-t-il une vraie probabilité ?

<details><summary>Indice 1</summary>

Relis le §15.4.4 et la section « Au-delà du livre : seuil de décision, calibration et classes déséquilibrées ».

</details>
<details><summary>Indice 2</summary>

Distingue un score qui **classe** bien les exemples d'une probabilité **calibrée** ; dis comment on vérifie la calibration, comment on la corrige, et sur quoi se fonde le choix du seuil.

</details>
<details><summary>Indice 3</summary>

Plan : « pas forcément » ; la courbe de calibration sur un jeu de validation ; `CalibratedClassifierCV` ; le seuil choisi d'après le coût des erreurs, sur une validation, avec `TunedThresholdClassifierCV` ou à la main.

</details>

### 15.E5 — Mettre un modèle scikit-learn en production

<details><summary>Indice 1</summary>

Relis la section « Au-delà du livre : sauvegarder un modèle et ses versions ».

</details>
<details><summary>Indice 2</summary>

Ce qu'on sauvegarde (le pipeline entier), avec quoi (`joblib`, ou un format plus sûr), ce qu'on note à côté (versions, données, score), et comment on le sert.

</details>
<details><summary>Indice 3</summary>

Plan : réentraîner le pipeline retenu sur toutes les données d'entraînement ; le sauvegarder avec `joblib` et un fichier de versions et de métadonnées ; épingler l'environnement (`requirements.txt`, conteneur) ; le servir derrière une API qui charge le modèle une fois ; surveiller ses entrées et ses performances.

</details>

<a id="notebook"></a>

## Notebook

### Partie A · Un estimateur et ses sorties

### Ex 15.9 — Anatomie d'un estimateur : Ridge sur un sinus bruité 📦

<details><summary>Indice 1</summary>

Les trois temps de la fiche (§15.4) : créer l'objet avec ses hyperparamètres, `fit` sur une matrice `X` à deux dimensions, puis lire les attributs appris (ceux qui finissent par `_`) et appeler `predict` ou `score`, toujours avec des données en deux dimensions.

</details>
<details><summary>Indice 2</summary>

a) `reshape(-1, 1)` fait une colonne. b) et c) `fit` renvoie le modèle, `coef_` est un tableau d'un élément par feature, `intercept_` un nombre. d) Fabrique une matrice de deux exemples d'une feature avec `np.array([[...], [...]])`, et passe-la à `predict` en un appel ; `tolist()` donne une liste. e) `score(X, y)` d'un régresseur. f) `clone` (dans `sklearn.base`) copie les hyperparamètres sans rien d'appris ; `set_params` renvoie l'objet, ce qui permet d'enchaîner `.fit(...)`. g) Un masque booléen `x_9 < 2.5` choisit les lignes de `X_9` et de `y_9`.

</details>
<details><summary>Indice 3</summary>

Le squelette : `X_9 = x_9.reshape(-1, 1)` ; `model_9 = Ridge(alpha=1.0).fit(X_9, y_9)` ; `slope_9 = float(model_9.coef_[0])`. Pour les autres : c) l'attribut appris de l'ordonnée à l'origine ; d) `model_9.predict(` une matrice `(2, 1)` des deux extrémités `)` ; e) la méthode `score` ; f) `clone(model_9).set_params(alpha=100.0).fit(...)`, puis sa pente comme b) ; g) un `Ridge` neuf, entraîné sur les lignes du masque.

</details>

### Ex 15.10 — predict_proba : prédire ce que valent les probabilités de trois classifieurs 🔮

<details><summary>Indice 1</summary>

Pour chaque modèle, demande-toi **d'où viennent** ses probabilités : d'une calibration ajoutée après coup (le `SVC`), des proportions d'espèces dans une feuille (l'arbre), ou d'un produit de vraisemblances (Naïve Bayes, ch. 13). La fiche (§15.4.4) dit pour le SVC ce que prévient la documentation.

</details>
<details><summary>Indice 2</summary>

a) `predict` d'un SVC suit le signe de ses scores ; `predict_proba` vient d'une régression logistique sur ces scores, apprise par une validation croisée interne : deux calculs différents ne peuvent se contredire que là où ils hésitent entre deux espèces. Quelle part d'une carte des mesures du bec est proche d'une frontière ? b) Jusqu'où pousse un arbre sans limite de profondeur, et que vaut la proportion de l'espèce majoritaire dans une feuille où il n'y a plus qu'une espèce ? Les manchots ont-ils souvent exactement les mêmes 4 mesures ? c) Que fait Naïve Bayes quand deux mesures corrélées disent la même chose (ch. 13) ?

</details>
<details><summary>Indice 3</summary>

a) Le critère : les désaccords ne peuvent exister qu'au voisinage des frontières ; estime la part de la carte qu'occupe une bande étroite autour des frontières entre les trois espèces, puis choisis la tranche. b) La question à te poser : où un arbre sans limite de profondeur arrête-t-il ses coupures, et que vaut alors la plus grande probabilité d'une feuille ? c) Le critère : compter deux fois une même preuve rend-il plus ou moins sûr de soi ? N'exécute l'expérience qu'après avoir écrit tes trois prédictions.

</details>

### Ex 15.11 — Datasets synthétiques, fetch_* et train_test_split stratifié 📦

<details><summary>Indice 1</summary>

Un loader rend un `Bunch` (un dictionnaire dont les clés se lisent comme des attributs : `data`, `target`, `feature_names`) ou, avec `return_X_y=True`, le couple `(X, y)`. `train_test_split` rend les morceaux dans l'ordre `X_train, X_test, y_train, y_test`.

</details>
<details><summary>Indice 2</summary>

a) La forme de `X` donne deux des trois nombres ; le nombre de classes est celui des valeurs distinctes de `y` (`np.unique`). b) `feature_names[0]`. c) `X.shape`, en liste. d) à f) Garde la quatrième sortie de `train_test_split` (`y_test`) et prends sa moyenne : avec des labels 0 et 1, c'est la part de la classe 1. g) et h) Sa somme compte les exemples de la classe 1.

</details>
<details><summary>Indice 3</summary>

a) `[*load_wine(return_X_y=True)[0].shape, ...]`, complété par le nombre de classes. Pour les autres, ce qu'il faut lire : b) le premier nom de `feature_names` ; c) la forme de la matrice de California ; d) à f) la moyenne de la quatrième sortie de `train_test_split`, avec les arguments que demande chaque question ; g) et h) la somme de cette même sortie sur les données rares, en entier.

</details>

### Ex 15.12 — Faire tourner le code de 2018 avec scikit-learn 1.6 🐛

<details><summary>Indice 1</summary>

Chaque panne a laissé une trace : le type de l'exception et sa dernière ligne disent quel argument, quel attribut ou quelle forme ne va plus. Pour chacune, ouvre la page de la classe ou de la fonction dans la documentation **1.6**, et cherche les mentions « *Changed in version* », « *Deprecated since version* » ou « *Removed* ».

</details>
<details><summary>Indice 2</summary>

Les cinq pannes qui plantent se lisent dans les messages : une forme de données, un argument inconnu d'un encodeur, un attribut qui n'existe pas, une clé absente d'un dictionnaire, un argument inconnu d'une fonction de mesure. La sixième ne plante pas : le programme tourne, mais un résultat diffère du rapport. Compare alors chaque appel aux valeurs par défaut de 0.19 (la documentation de la version 0.19 existe encore en ligne, et la fiche liste plusieurs défauts qui ont changé).

</details>
<details><summary>Indice 3</summary>

Le premier point : `line.predict(np.array([[x.min()]]))` (un exemple d'une feature, forme `(1, 1)`). Pour les autres, ce qu'il faut chercher : 2) le nouveau nom de l'argument de sortie dense de `OneHotEncoder` ; 3) le vrai nom de l'attribut du meilleur réglage ; 4) l'argument de `GridSearchCV` qui fait calculer les scores d'entraînement ; 5) la fonction qui rend directement la RMSE (ou une racine) ; 6) le nombre de folds que `GridSearchCV` faisait par défaut en 0.19, à écrire explicitement.

</details>

### Partie B · Clustering, transformateurs, PCA et ensembles

### Ex 15.13 — Clustering avec scikit-learn : KMeans (le coude du livre), DBSCAN et clustering hiérarchique sur deux lunes et sept blobs 📦

<details><summary>Indice 1</summary>

Tous les clusterings ont `fit_predict(X)`, qui rend un label par point ; `KMeans` garde aussi son `inertia_`. La silhouette moyenne se calcule avec `silhouette_score(X, labels)`, l'accord avec les vrais groupes avec `adjusted_rand_score(vrais, trouvés)`.

</details>
<details><summary>Indice 2</summary>

a) et b) Une boucle (ou une liste en compréhension) sur `KS_13`, qui entraîne un `KMeans(n_clusters=k, n_init=10, random_state=0)` par $k$ et garde son inertie et sa silhouette. c) La position du maximum de b) (`np.argmax`) désigne un élément de `KS_13`. d) à h) `fit_predict` sur les lunes, puis l'indice de Rand ajusté. f) Les labels de DBSCAN : −1 pour le bruit, 0, 1, 2… pour les clusters ; compte les labels distincts autres que −1, et les points de label −1.

</details>
<details><summary>Indice 3</summary>

a) `models_13 = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_blobs_13) for k in KS_13]`, puis `[m.inertia_ for m in models_13]`. Pour les autres, le calcul à faire : b) la silhouette des labels de chaque modèle ; c) l'élément de `KS_13` à la position du maximum de b) ; d), e), g) et h) l'indice de Rand ajusté entre `y_moons_13` et les labels rendus par `fit_predict` ; f) deux comptes sur les labels de DBSCAN, les valeurs distinctes hors −1 et les −1.

</details>

### Ex 15.14 — Transformateurs : test hors de [0, 1] et inverse_transform 📦

<details><summary>Indice 1</summary>

La règle d'or du ch. 12 : `fit` sur l'entraînement, `transform` partout. Pour ramener une droite de l'espace transformé, il suffit d'en ramener **deux points** (✏️ 15.3) : `inverse_transform` attend des points complets, avec toutes leurs coordonnées.

</details>
<details><summary>Indice 2</summary>

a) et b) `scaler_14.transform(TEST_14)`, puis `min(axis=0)` et `max(axis=0)`. c) Un point sort du carré si **l'une** de ses coordonnées est sous 0 ou au-dessus de 1 : combine deux masques avec `|`, puis `.any(axis=1)` ligne par ligne. d) Le scaler des points transforme les deux colonnes ; la régression apprend la colonne 1 transformée à partir de la colonne 0 transformée ; ses deux extrémités ont pour abscisses 0 et 1 et pour ordonnées ses prédictions en 0 et en 1 ; empile ces deux points en `(2, 2)` et passe-les à `inverse_transform`. e) Une pente : différence des ordonnées sur différence des abscisses. f) Une régression sur les données brutes, `coef_[0]`.

</details>
<details><summary>Indice 3</summary>

a) `scaler_14.transform(TEST_14).min(axis=0).tolist()`. Pour les autres, le calcul à faire : b) le même avec le maximum ; c) la moyenne, sur les lignes, du masque « une coordonnée hors de $[0, 1]$ » ; d) les deux points `(0, ŷ₀)` et `(1, ŷ₁)` de l'espace transformé, empilés en `(2, 2)`, passés à `inverse_transform` ; e) la pente entre les deux lignes de `ends_14` ; f) le `coef_[0]` d'une régression sur les heures en colonne et les voitures.

</details>

### Ex 15.15 — PCA de 3D vers 2D et blanchiment 📦

<details><summary>Indice 1</summary>

Une `PCA` entraînée garde ses directions (`components_`) et la part de variance de chacune (`explained_variance_ratio_`) ; `transform` projette, `inverse_transform` revient dans l'espace d'origine, sur le plan gardé. Le blanchiment (`whiten=True`) divise chaque composante par son écart-type.

</details>
<details><summary>Indice 2</summary>

a) et b) Une liste, et sa somme. c) `inverse_transform(transform(BLOB_15))` a la forme de `BLOB_15` : la moyenne des carrés de la différence, sur toutes les valeurs, est un seul nombre. d) `fit_transform` d'une PCA blanchie, puis `np.var(..., axis=0, ddof=1)`. e) et f) Un `KMeans(n_clusters=2, n_init=10, random_state=0)` sur chacune des deux projections, puis `adjusted_rand_score(group_15, labels)`.

</details>
<details><summary>Indice 3</summary>

a) `pca_15 = PCA(n_components=2).fit(BLOB_15)` et `ratio_15 = pca_15.explained_variance_ratio_.tolist()`. Pour les autres, le calcul à faire : b) la somme de a) ; c) l'aller-retour `transform` puis `inverse_transform`, comparé à `BLOB_15` par la moyenne des carrés des écarts ; d) une PCA blanchie à deux composantes, puis la variance de chaque colonne avec `ddof=1` ; e) et f) le même k-means sur chacune des deux projections, noté par l'indice de Rand ajusté face à `group_15`.

</details>

### Ex 15.16 — Cinq spirales : quel ensemble les démêle ? 🔬

<details><summary>Indice 1</summary>

Un AdaBoost est un estimateur comme un autre (§15.8) ; après `fit`, `estimators_` contient les membres qu'il a gardés, et `n_estimators` n'est qu'un maximum. Relis la règle d'arrêt de SAMME au ch. 14 : un membre parfait, ou pas meilleur que le hasard.

</details>
<details><summary>Indice 2</summary>

Écris une petite fonction qui entraîne un AdaBoost (50 tours au plus, `random_state=0`) sur la partie entraînement à partir d'un modèle de base, et une autre qui rend `[model.score(X_TEST_16, Y_TEST_16), len(model.estimators_)]`. Les membres par défaut d'AdaBoost s'obtiennent en ne passant pas `estimator`. e) Une liste en compréhension sur `DEPTHS_16`, avec `DecisionTreeClassifier(max_depth=depth, random_state=0)` (`max_depth=None` : sans limite). c) Le `score` sur la partie entraînement.

</details>
<details><summary>Indice 3</summary>

Le squelette : deux petites fonctions, `ada_16(estimator=None)`, qui rend `AdaBoostClassifier(estimator=estimator, n_estimators=50, random_state=0)` entraîné sur la partie entraînement, et `summary_16(model)`, qui rend la liste `[accuracy de test, nombre de membres]` ; la ligne clé de la seconde : `len(model.estimators_)`. Ensuite, chaque réponse est un appel : a) `summary_16(ada_16(RidgeClassifier()))`, et de même pour les autres modèles de base.

</details>

### Partie C · Évaluer, régler et enchaîner

### Ex 15.17 — cross_validate, KFold, StratifiedKFold et RidgeClassifierCV 📦

<details><summary>Indice 1</summary>

`cross_val_score` rend un tableau de scores, un par fold ; `cross_validate` rend un dictionnaire de tableaux, dont les clés suivent les noms de `scoring` (`test_accuracy`, `train_accuracy`, `test_f1_macro`…). Relis dans la fiche (§15.9.1) ce que `cv` fait d'un entier pour un classifieur, et ce que fait vraiment un estimateur `...CV`.

</details>
<details><summary>Indice 2</summary>

a) et b) Deux appels à `cross_val_score`, l'un avec `cv=KFold(5)`, l'autre avec `cv=5`. c) à e) Un appel à `cross_validate`, puis la moyenne de trois de ses tableaux. f) et g) Le pipeline entraîné : son dernier élément (`ridge_cv_17[-1]`) garde l'`alpha_` choisi ; `ridge_cv_17.score(...)` note le modèle entraîné. h) Un pipeline neuf, identique, passé à `cross_val_score` avec `FOLDS_17`.

</details>
<details><summary>Indice 3</summary>

a) `cross_val_score(MODEL_17, X_17, Y_17, cv=KFold(5)).tolist()`. Pour les autres, ce qu'il faut lire : b) la moyenne du même appel avec `cv=5` ; c) à e) les clés `"test_accuracy"`, `"train_accuracy"` et `"test_f1_macro"` de `results_17` ; f) l'attribut appris de la dernière étape ; g) la méthode `score` du pipeline entraîné ; h) la moyenne de `cross_val_score` d'un pipeline construit à neuf, avec `FOLDS_17`.

</details>

### Ex 15.18 — GridSearchCV : best_params_, cv_results_ et score d'entraînement 📦

<details><summary>Indice 1</summary>

Après `fit`, une recherche garde son meilleur réglage (`best_params_`), son score (`best_score_`), sa position (`best_index_`) et le tableau complet (`cv_results_`), qu'on lit mieux en `DataFrame`. Relis l'encadré de la fiche sur l'ordre des candidats et les égalités (§15.9.3).

</details>
<details><summary>Indice 2</summary>

a) et b) `best_params_` est un dictionnaire : ses clés sont les noms de la grille. d) Une ligne de `cv_results_` par candidat. e) La colonne `rank_test_score` : compte les 1. f) `cv_results_["params"]` est la liste des candidats dans l'ordre où ils ont été évalués ; le deuxième est d'indice 1. g) Les colonnes `mean_train_score` et `mean_test_score`, à la ligne `best_index_`. h) `search_18.score(...)` utilise le meilleur modèle, réentraîné.

</details>
<details><summary>Indice 3</summary>

a) `search_18 = GridSearchCV(RidgeClassifier(), GRID_18, cv=FOLDS_18, return_train_score=True).fit(X_TRAIN_18, Y_TRAIN_18)`, puis `search_18.best_params_["alpha"]`. Pour les autres, ce qu'il faut lire : b) l'autre clé de `best_params_` ; c) `best_score_` ; d) le nombre de lignes de `cv_results_` ; e) le nombre de rangs égaux à 1 ; f) le `solver` de l'élément d'indice 1 de `cv_results_["params"]` ; g) la différence des deux moyennes à la ligne `best_index_` ; h) le score sur `X_TEST_18`, `Y_TEST_18`.

</details>

### Ex 15.19 — RandomizedSearchCV avec des distributions 📦

<details><summary>Indice 1</summary>

`RandomizedSearchCV` s'emploie comme `GridSearchCV`, avec `param_distributions` et `n_iter`. Relis l'encadré de la fiche sur les distributions de SciPy (§15.9.4) : ce que tire `loguniform`, ce que signifient les deux arguments de `uniform`, et quand la recherche tire avec ou sans remise.

</details>
<details><summary>Indice 2</summary>

a) à c) `best_params_` et `best_score_`, comme en 15.18 (convertis les valeurs avec `float`). d) `cv_results_["param_C"]` contient les 20 valeurs essayées : un ensemble (`set`) les rend distinctes. e) et f) Transforme chaque dictionnaire de `cv_results_["params"]` en un objet comparable et hachable (un tuple de ses paires triées), puis compare le nombre de candidats au nombre de candidats distincts. g) et h) `rvs(10_000, random_state=1519)` donne un tableau ; la moyenne d'un masque booléen est une part.

</details>
<details><summary>Indice 3</summary>

a) `random_19 = RandomizedSearchCV(SVC(), {...}, n_iter=20, cv=FOLDS_19, random_state=1519).fit(X_19, Y_19)`, puis `float(random_19.best_params_["C"])`. Pour les autres, le calcul à faire : b) et c) la même lecture pour `gamma`, puis `best_score_` ; d) la taille de l'ensemble des valeurs de `cv_results_["param_C"]` ; e) et f) pour chaque recherche, le nombre de candidats moins le nombre de candidats distincts (la ligne clé : `seen = [tuple(sorted(p.items())) for p in search.cv_results_["params"]]`) ; g) et h) la moyenne du masque `tirages < 1`.

</details>

### Ex 15.20 — Grille ou hasard : qui gagne avec 16 essais ? 🔮

<details><summary>Indice 1</summary>

Les surfaces décroissent avec la distance au pic : un ensemble d'essais gagne s'il contient un point **plus près du pic** que le meilleur point de l'autre. Regarde, pour chaque surface, ce que la grille essaie sur l'axe qui compte, et ce que 16 tirages y essaient.

</details>
<details><summary>Indice 2</summary>

a) Seul $h_1$ compte : la grille essaie 4 valeurs de $h_1$, espacées de 0,25 (de 0,125 à 0,875) ; à quelle distance du pic tombe la plus proche, et quelle est la chance qu'un tirage uniforme de $h_1$ tombe plus près ? Puis la formule $1 - (1 - q)^{16}$ de la fiche. b) Les deux comptent : la zone gagnante est un disque autour du pic, de rayon égal à la distance du meilleur point de la grille ; quelle part du carré occupe-t-il ? c) Un tirage peut-il battre **strictement** un score que la grille atteint déjà au sommet ?

</details>
<details><summary>Indice 3</summary>

a) Le calcul à poser : $d$, la distance de 0,53 à la valeur de la grille la plus proche ; un tirage uniforme de $h_1$ tombe plus près du pic avec la probabilité $2d$ ; au moins un des 16 avec la probabilité $1 - (1 - 2d)^{16}$ : calcule-la et situe-la dans les tranches. b) Le calcul à poser : la distance $r$ du pic au point de la grille le plus proche, l'aire $\pi r^2$ du disque, puis la même formule. c) Le critère : le pic est-il un point de la grille ? d) se fait avec le calcul de b). N'exécute l'expérience qu'après avoir écrit tes prédictions.

</details>

### Ex 15.21 — Frontières de décision : DecisionBoundaryDisplay et surfaces 3D 📦

<details><summary>Indice 1</summary>

`DecisionBoundaryDisplay.from_estimator(modèle, X, response_method=...)` calcule la réponse du modèle sur une grille qui couvre `X`, la trace, et garde la grille (`xx0`, `xx1`), la réponse (`response`) et les axes (`ax_`).

</details>
<details><summary>Indice 2</summary>

a) `response_method="predict"`, puis `classes_21.ax_.scatter(X_21[:, 0], X_21[:, 1], c=Y_21)`. b) `response_method="decision_function"` avec une palette divergente (`cmap="RdBu_r"`), puis la ligne de niveau 0 tracée par `ax_.contour` sur la même grille. c) `np.clip(scores_21.response, -1, 1)` et `plot_surface(scores_21.xx0, scores_21.xx1, ...)` sur des axes 3D.

</details>
<details><summary>Indice 3</summary>

Le squelette : `classes_21 = DecisionBoundaryDisplay.from_estimator(MODEL_21, X_21, response_method="predict", alpha=0.4)` ; `scores_21 = DecisionBoundaryDisplay.from_estimator(MODEL_21, X_21, response_method="decision_function", cmap="RdBu_r")` ; `scores_21.ax_.contour(scores_21.xx0, scores_21.xx1, scores_21.response, levels=[0], colors="black")`. Pour c) : `ax = plt.figure().add_subplot(projection="3d")`, puis `ax.plot_surface(...)` avec la réponse écrêtée.

</details>

### Ex 15.22 — Pipeline polynomial + ridge : degré et alpha ensemble 📦

<details><summary>Indice 1</summary>

Un pipeline est un estimateur : il entre tel quel dans `GridSearchCV`. Ses hyperparamètres se nomment `étape__paramètre`, avec le nom que tu as donné à l'étape (fiche §15.9.5) ; `pipe.get_params().keys()` les liste.

</details>
<details><summary>Indice 2</summary>

La grille : un dictionnaire de deux clés, `"poly__degree"` et `"ridge__alpha"`. a) `best_params_` avec ces deux clés. c) `best_estimator_.named_steps["poly"]` est l'étape entraînée ; son attribut `n_output_features_` compte ses colonnes. e) et f) `pd.DataFrame(search_22.cv_results_)` a une colonne `param_poly__degree` et une colonne `param_ridge__alpha` : filtre les lignes qui t'intéressent.

</details>
<details><summary>Indice 3</summary>

Le squelette : `grid_22 = {"poly__degree": DEGREES_22, "ridge__alpha": ALPHAS_22}` et `search_22 = GridSearchCV(pipe_22, grid_22, cv=FOLDS_22, return_train_score=True).fit(X_TRAIN_22, Y_TRAIN_22)`. Pour les autres, ce qu'il faut lire : a) les deux valeurs de `best_params_`, en liste ; b) `best_score_` ; c) `n_output_features_` de l'étape `"poly"` du meilleur pipeline ; d) le score sur le test ; e) le score moyen d'une ligne de degré 0 ; f) la différence des deux moyennes sur la ligne du degré 9 et de l'alpha 0,001.

</details>

### Ex 15.23 — Lire cv_results_ : carte de chaleur et écarts-types 📈

<details><summary>Indice 1</summary>

`cv_results_` en `DataFrame` a une ligne par candidat ; `pivot(index=..., columns=..., values=...)` en fait un tableau degré × alpha. La règle d'une erreur type (MP2) : on garde les candidats dont le score atteint celui du meilleur moins son erreur type, l'écart-type divisé par $\sqrt{k}$.

</details>
<details><summary>Indice 2</summary>

a) `results.pivot(index="param_poly__degree", columns="param_ridge__alpha", values="mean_test_score")`, et de même avec `std_test_score` ; `plt.imshow(table.to_numpy())`, puis `ax.text` dans chaque case. b) Pour chaque colonne de la table, la ligne du maximum (`idxmax`) est un degré. c) Un masque sur la colonne `mean_test_score` du `DataFrame`. d) Le seuil : `best_score_` moins le `std_test_score` de la ligne `best_index_`, divisé par $\sqrt 5$ ; puis compte les candidats au-dessus.

</details>
<details><summary>Indice 3</summary>

a) `table_23 = results_23.pivot(index="param_poly__degree", columns="param_ridge__alpha", values="mean_test_score")`. Pour les autres, le critère : b) pour chaque colonne de `table_23`, le degré de son maximum, en entier ; c) les candidats dont `mean_test_score` est strictement sous 0,5 ; d) le seuil « meilleur score moins l'écart-type du meilleur divisé par $\sqrt 5$ », puis les candidats qui l'atteignent. La lecture des cartes se fait ensuite dans tes notes.

</details>

### Ex 15.24 — Refaire l'expérience de fuite du ch. 8 (sélection de features sur du bruit) avec un Pipeline 🔬

<details><summary>Indice 1</summary>

Une sélection de features regarde la cible : faite sur toutes les lignes, elle a vu les labels des folds de validation. Dans un pipeline, elle est réapprise sur l'entraînement de chaque fold (fiche §15.9.7) ; dans une validation croisée imbriquée, la recherche entière l'est aussi.

</details>
<details><summary>Indice 2</summary>

a) `SelectKBest(f_classif, k=20).fit_transform(X_24, Y_24)`, puis une validation croisée de la régression logistique sur les colonnes gardées. b) Le pipeline `make_pipeline(SelectKBest(...), LogisticRegression(...))` passé à `cross_val_score` avec `X_24` entier. c) Une `GridSearchCV` de ce pipeline sur `{"selectkbest__k": KS_24}`. d) Une recherche neuve passée à `cross_val_score` avec un autre découpage extérieur.

</details>
<details><summary>Indice 3</summary>

a) `selected = SelectKBest(f_classif, k=20).fit_transform(X_24, Y_24)`, puis `cross_val_score(LogisticRegression(max_iter=1000), selected, Y_24, cv=FOLDS_24).mean()`. Pour les autres, la différence avec a) : b) la sélection entre dans le pipeline, qui reçoit `X_24` entier ; c) le pipeline entre dans une recherche sur `selectkbest__k`, dont on lit le `best_score_` ; d) une recherche neuve, identique, entre elle-même dans `cross_val_score`, avec un découpage extérieur à toi.

</details>

### Partie D · Décider, préparer, brancher, sauvegarder

### Ex 15.25 — Régler le seuil de décision d'un classifieur 🔬

<details><summary>Indice 1</summary>

Un seuil se choisit comme un hyperparamètre : sur une validation, jamais sur le test, d'après une mesure qui reflète le coût réel (fiche, « seuil de décision »). Pour la théorie, compare le coût **moyen** des deux décisions possibles pour un exemple dont la probabilité d'être positif vaut $p$.

</details>
<details><summary>Indice 2</summary>

b) `MODEL_25.predict_proba(X_VAL_25)[:, 1]` une fois, puis, pour chaque seuil, la règle `(proba >= t).astype(int)` et `cost_25`. c) La position du plus petit coût (`np.argmin`, qui prend le premier en cas d'égalité) désigne un seuil de `THRESHOLDS_25`. e) Décider « positif » coûte en moyenne $(1 - p) \times 1$ (une fausse alerte si l'exemple est négatif), décider « négatif » coûte $p \times 10$ : où les deux s'égalent-ils ? f) `recall_score` pour chacune des deux règles. g) et h) `TunedThresholdClassifierCV` s'emploie comme un classifieur ; son seuil est dans `best_threshold_`.

</details>
<details><summary>Indice 3</summary>

b) `val_proba = MODEL_25.predict_proba(X_VAL_25)[:, 1]` et `val_costs_25 = [cost_25(Y_VAL_25, (val_proba >= t).astype(int)) for t in THRESHOLDS_25]`. Pour les autres, le calcul à faire : c) le seuil de `THRESHOLDS_25` à la position du plus petit coût ; d) la même règle sur les probabilités du test ; e) résous $10\,p = 1 - p$ ; f) deux `recall_score` sur le test ; g) et h) le `fit` de `TunedThresholdClassifierCV` sur `X_REST_25`, puis son `best_threshold_` et le coût de son `predict` sur le test.

</details>

### Ex 15.26 — Classes déséquilibrées : class_weight, rééchantillonnage et la bonne métrique (recall, PR-AUC) 📦

<details><summary>Indice 1</summary>

Avec 2 % de positifs, l'accuracy d'un modèle qui répond toujours « négatif » est déjà de 98 %. Les trois leviers de la fiche (repondérer, rééchantillonner l'entraînement seul, déplacer le seuil) se comparent sur des mesures qui regardent la classe rare : recall, precision, balanced accuracy, average precision.

</details>
<details><summary>Indice 2</summary>

`report_26` fait tout le travail : il suffit de lui donner chaque modèle et ses données d'entraînement. d) Les indices des positifs : `np.flatnonzero(Y_TRAIN_26 == 1)` ; tire autant d'indices de négatifs, sans remise, avec le générateur imposé ; concatène les deux ; entraîne sur ces lignes seulement. e) `FixedThresholdClassifier` entoure un classifieur. f) `compute_class_weight("balanced", classes=..., y=...)`.

</details>
<details><summary>Indice 3</summary>

a) `dummy_26 = report_26(DummyClassifier(strategy="most_frequent"), X_TRAIN_26, Y_TRAIN_26)`. Pour les autres, ce qui change dans l'appel : b) et c) le pipeline, sans puis avec `class_weight="balanced"` ; d) les lignes d'entraînement, réduites aux positifs et à autant de négatifs tirés par `np.random.default_rng(1526).choice(..., replace=False)` ; e) le pipeline entouré d'un `FixedThresholdClassifier` au seuil 0,1 ; f) pas de modèle : les poids « balanced » des deux classes, en liste.

</details>

### Ex 15.27 — La même validation croisée pour six familles de modèles scikit-learn : score moyen, écart-type et temps 📦

<details><summary>Indice 1</summary>

Une comparaison honnête : les mêmes folds pour tous (un objet de découpage, pas un entier mélangé différemment à chaque appel), plusieurs mesures par modèle (la moyenne **et** la dispersion), et le temps. `cross_validate` rend tout cela d'un coup.

</details>
<details><summary>Indice 2</summary>

Dans `compare_27`, une boucle sur `models.items()` : `cross_validate(model, X, y, cv=cv)` rend `test_score` et `fit_time` ; range pour chaque nom un petit dictionnaire `{"mean": ..., "std": ..., "fit_ms": ...}`, puis fais un `DataFrame` indexé par les noms (`pd.DataFrame.from_dict(rows, orient="index")`) et trie-le (`sort_values`). c) et d) Deux `cross_val_score` sur `FOLDS_27`.

</details>
<details><summary>Indice 3</summary>

Le squelette de `compare_27` : une boucle `for name, model in models.items():`, un `cross_validate` par modèle, un dictionnaire de trois nombres rangé sous `name`, puis un `DataFrame` trié. La ligne clé : `rows[name] = {"mean": ..., "std": result["test_score"].std(ddof=1), "fit_ms": ...}` (les durées de `fit_time` sont en secondes). Pour c) et d) : `cross_val_score(modèle, X_27, Y_27, cv=FOLDS_27).mean()`.

</details>

### Ex 15.28 — ColumnTransformer sur Penguins brut 📦

<details><summary>Indice 1</summary>

Chaque type de colonne a sa préparation, réunie dans un `ColumnTransformer` à deux branches, lui-même première étape d'un pipeline (fiche, « données hétérogènes »). Les colonnes se choisissent par type avec `make_column_selector`, et le tout se règle comme un seul estimateur, avec des noms enchaînés `étape__branche__étape__paramètre`.

</details>
<details><summary>Indice 2</summary>

La préparation : deux `make_pipeline` (imputer puis scaler, imputer puis encodeur), puis `ColumnTransformer([("num", ..., make_column_selector(dtype_include="number")), ("cat", ..., make_column_selector(dtype_include=object))])`. a) Après `fit`, `model_28[0]` est le `ColumnTransformer` entraîné ; `get_feature_names_out()` rend un tableau de noms. b) `clone` fait un pipeline neuf. c) `predict` sur `NEW_28` tel quel. d) et e) `model_28.get_params().keys()` donne les noms longs : celui de la stratégie de l'imputeur des nombres passe par `columntransformer`, `num` et `simpleimputer` ; `ties_28` et `weak_28` se lisent dans `cv_results_`.

</details>
<details><summary>Indice 3</summary>

La ligne clé de la préparation : `prep_28 = ColumnTransformer([("num", numeric_28, make_column_selector(dtype_include="number")), ("cat", categorical_28, make_column_selector(dtype_include=object))])`. a) La longueur de `model_28[0].get_feature_names_out()`. Pour les autres, ce qu'il faut faire : b) la moyenne d'une validation croisée d'un clone ; c) une prédiction sur `NEW_28`, en liste ; d) et e) une grille à deux clés, dont `"columntransformer__num__simpleimputer__strategy"`, puis, dans `pd.DataFrame(search_28.cv_results_)`, les lignes de rang 1 et la ligne (médiane, 0,01).

</details>

### Ex 15.29 — Quand scikit-learn refuse un estimateur mylearn 🐛

<details><summary>Indice 1</summary>

Relis la section « brancher ton propre estimateur » de la fiche et l'encadré sur `get_params`, `set_params` et `clone` (§15.9.3) : ce que `BaseEstimator` lit pour fabriquer `get_params`, ce que `ClassifierMixin` déclare, dans quel ordre on les hérite, et ce qu'un constructeur a le droit de faire.

</details>
<details><summary>Indice 2</summary>

a) Lance l'appel dans une cellule à toi : le nom de l'exception est au début de la dernière ligne du message. L'adaptateur : trois règles, trois bugs : 1) `get_params` cherche, pour chaque argument du constructeur, un attribut **du même nom** ; 2) l'ordre d'héritage décide si l'objet est reconnu comme classifieur (`is_classifier`), donc si `cv=5` stratifie ; 3) `set_params` ne fait que changer des attributs : un modèle construit dans `__init__` ne voit pas ce changement. Corrige-les un par un, en relançant la vérification.

</details>
<details><summary>Indice 3</summary>

a) Le critère : lis le type de l'exception levée par `clone`. L'adaptateur corrigé : la signature `class MylearnKNN_29(ClassifierMixin, BaseEstimator):` et `def __init__(self, n_neighbors=5):`, dont le corps ne fait que ranger son argument sous le même nom ; la ligne clé de `fit` : `self.model_ = mylearn.neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors)`. Le reste de `fit` et `predict` change peu.

</details>

### Ex 15.30 — Sauvegarder, recharger et versionner un pipeline 🛠️

<details><summary>Indice 1</summary>

Trois règles de la fiche : sauvegarder le pipeline **entier**, noter les versions à côté du fichier, ne jamais charger un fichier inconnu. `joblib.dump(objet, chemin)` et `joblib.load(chemin)` écrivent et relisent un objet Python ; `json.dumps` et `json.loads` font de même pour un dictionnaire de textes et de nombres.

</details>
<details><summary>Indice 2</summary>

`save_30` : un `joblib.dump` vers `folder / "model.joblib"`, un dictionnaire des sept informations (les versions par `platform.python_version()`, `np.__version__`, `sklearn.__version__`, `joblib.__version__`), écrit en JSON dans `folder / "model.json"` ; rends le dictionnaire. `load_30` : relis le JSON, compare sa version de scikit-learn à `sklearn.__version__`, affiche la ligne `⚠️` si elles diffèrent, puis rends `joblib.load(...)`.

</details>
<details><summary>Indice 3</summary>

Le squelette de `save_30` : un `joblib.dump`, un dictionnaire, une écriture JSON, `return` du dictionnaire. Deux lignes clés : `meta = {"python": platform.python_version(), ..., "saved_on": datetime.date.today().isoformat()}` et `(folder / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")`. `load_30` fait le chemin inverse, avec la comparaison des versions entre la lecture du JSON et le `joblib.load`.

</details>

### Ex 15.31 — Défi : prédire le sexe des manchots depuis le fichier brut 🏆

<details><summary>Indice 1</summary>

Commence par la baseline de l'énoncé (le `ColumnTransformer` de 15.28 et une régression logistique, dans une `GridSearchCV` notée à la log loss), et regarde son score public. La log loss punit les probabilités trop sûres d'elles quand elles se trompent : un bon modèle est juste **et** modeste. Et pense à la biologie : à quoi reconnaît-on un mâle, quand les trois espèces n'ont pas la même taille ?

</details>
<details><summary>Indice 2</summary>

Deux pistes, à mesurer chacune par le score public ou ta propre validation croisée imbriquée. 1) Des mesures relatives à l'espèce : un mâle Adélie est grand **pour un Adélie** ; un transformateur à toi (`TransformerMixin`, `BaseEstimator`) peut apprendre, dans `fit`, la moyenne et l'écart-type de chaque mesure dans chaque espèce, et rendre des z-scores dans `transform` (une valeur manquante devient 0, la moyenne de son espèce). 2) Des probabilités plus fiables : une recherche notée `scoring="neg_log_loss"`, des modèles qui donnent des probabilités raisonnables (une régression logistique, un SVC avec `probability=True`), et un vote souple de plusieurs modèles, qui adoucit les probabilités extrêmes.

</details>
<details><summary>Indice 3</summary>

Le squelette : une branche à toi pour les mesures relatives à l'espèce et un one-hot pour le texte, dans un `ColumnTransformer` ; un modèle qui donne des probabilités (ou un vote souple de deux) ; une recherche notée à la log loss autour du tout. Deux lignes clés : `prep = ColumnTransformer([("within", WithinSpecies(), ["Species"] + NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Species", "Island"])])` et `return GridSearchCV(Pipeline([("prep", prep), ("model", ...)]), grille, scoring="neg_log_loss", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0))`. Le transformateur : `fit` fait un `groupby` sur l'espèce, `transform` aligne les moyennes et les écarts-types sur l'espèce de chaque ligne avec `reindex`.

</details>
