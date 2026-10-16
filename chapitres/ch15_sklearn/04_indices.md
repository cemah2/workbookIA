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

### 15.R1 — Ch. 14 : Forêt aléatoire ou gradient boosting ?

<details><summary>Indice 1</summary>

Relis le tableau « Bilan : bagging ou boosting ? » de la fiche du ch. 14.

</details>
<details><summary>Indice 2</summary>

Pour chaque méthode, demande-toi si un arbre a besoin des précédents, et quel réglage fixe la taille de chaque pas.

</details>
<details><summary>Indice 3</summary>

a) La forêt peut entraîner tous ses arbres en même temps, puisqu'aucun n'a besoin des autres : l'option **A**. b) Relis la ligne « plus de membres » du tableau. c) Le boosting part d'apprenants faibles : leur défaut est-il le biais ou la variance ? d) Quel réglage multiplie la contribution de chaque arbre ? e) Quelle méthode donne un score out-of-bag ?

</details>

### 15.R2 — Ch. 12 : Où ajuster le scaler dans une validation croisée ?

<details><summary>Indice 1</summary>

Relis le §12.10 de la fiche du ch. 12 (transformations et validation croisée).

</details>
<details><summary>Indice 2</summary>

Demande-toi quelles lignes ont servi à calculer la transformation, et lesquelles servent ensuite à noter le modèle. Pour c), un `StandardScaler` calcule $z = (x - \mu)/\sigma$ avec la moyenne et l'écart-type de l'entraînement.

</details>
<details><summary>Indice 3</summary>

a) Une standardisation de tout le dataset calcule la moyenne avec les lignes des folds de validation : ces lignes ont servi à préparer le modèle qu'on note sur elles, c'est une fuite ; **vrai**. b) Quelles lignes a-t-on le droit d'utiliser pour apprendre, à chaque tour ? c) $(65 - 50)/10$. d) Relis ce qu'a montré l'expérience du ch. 8 (sélection sur du bruit). e) Laquelle des deux étapes a besoin d'une statistique calculée sur les données ?

</details>

### 15.R3 — Ch. 8 : Combien de modèles pour une validation croisée à 5 folds ?

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

### Ex 15.1 — Compter les modèles entraînés par une recherche en grille

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « combien de modèles entraîne une recherche ? » (§15.9.2) : le nombre de candidats, multiplié par le nombre de folds, plus le réentraînement final.

</details>
<details><summary>Indice 2</summary>

Compte d'abord les candidats (le produit des nombres de valeurs ; pour une liste de grilles, la somme des produits de chaque grille), puis multiplie par $k$, puis ajoute 1 si `refit` est vrai. Pour h), chaque fold extérieur lance une recherche **complète**.

</details>
<details><summary>Indice 3</summary>

a) $4 \times 3 = 12$ candidats. b) Chaque candidat est noté sur 5 folds : multiplie a) par 5. c) Ajoute le réentraînement final. d) Même calcul avec 3 folds. e) Trois hyperparamètres, à 5, 2 et 3 valeurs, puis 4 folds et le réentraînement final. f) Deux grilles : $1 \times 3$ et $1 \times 3 \times 2$ candidats ; additionne, multiplie par 5, sans réentraînement final. g) $n_{\text{iter}} \times k + 1$. h) Trois recherches complètes, comme en c).

</details>

### Ex 15.2 — De decision_function à predict_proba

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la sigmoïde et le softmax (§15.4.4), et la règle de `predict` à deux classes.

</details>
<details><summary>Indice 2</summary>

$\hat p = 1/(1 + e^{-s})$ ; le score qui correspond à une probabilité $\hat p$ est le logit $\ln\big(\hat p / (1 - \hat p)\big)$. À deux classes, `predict` renvoie la classe 1 quand le score est **strictement** positif. Pour h), $\hat p_k = e^{s_k} / \sum_j e^{s_j}$.

</details>
<details><summary>Indice 3</summary>

a) $\sigma(2) = 1/(1 + e^{-2}) = 1/(1 + 0{,}1353\ldots) \approx 0{,}8808$. b) Même calcul avec $s = -1$. c) $s = 0$. d) Applique la règle du signe aux cinq scores, en faisant attention au score nul. e) $\ln(0{,}9/0{,}1)$. f) $\ln(0{,}8/0{,}2)$. g) Compare chaque score au seuil de f). h) $e^2 / (e^2 + e^1 + e^{-1})$. i) Que devient la fraction si l'on multiplie chaque exponentielle par $e^5$ ? j) Relis pourquoi `RidgeClassifier` n'a pas de `predict_proba`.

</details>

### Ex 15.3 — Ramener une droite ajustée dans l'espace d'origine

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la mise à l'échelle min-max et son inverse (§15.6), puis le §15.6.1 : on ramène une droite en ramenant **deux** de ses points.

</details>
<details><summary>Indice 2</summary>

Ici $x' = (x + 3)/8$ et $y = 10 + 20\,y'$. Pour une valeur de $x$ : transforme-la en $x'$, applique la droite pour obtenir $y'$, puis ramène $y'$ en $y$. La pente d'origine se lit entre les deux extrémités ramenées.

</details>
<details><summary>Indice 3</summary>

a) $x' = (1 - (-3))/(5 - (-3)) = 4/8 = 0{,}5$. b) $y' = 0{,}6 \times x' + 0{,}25$. c) $y = 10 + 20\,y'$. d) En $x' = 0$, $y' = 0{,}25$ ; ramène-le. e) En $x' = 1$, $y' = 0{,}6 + 0{,}25$ ; ramène-le. f) La différence des $y$ de d) et e), divisée par l'étendue de $x$. g) Repars de l'extrémité gauche ($x = -3$) avec la pente de f), ou transforme $x = 0$. h) $(11 + 3)/8$ : la valeur sort de $[0, 1]$, et c'est normal. i) Applique la droite à h), puis ramène $y'$. j) Fais le calcul : `inverse_transform` sur le point $(0{,}6 ;\ 0{,}25)$ donne $(-3 + 8 \times 0{,}6 ;\ 10 + 20 \times 0{,}25)$ ; compare à f) et g).

</details>

### Ex 15.4 — Combien de features polynomiales ?

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

### Ex 15.5 — Budget d'une recherche d'hyperparamètres

<details><summary>Indice 1</summary>

Relis les deux encadrés 🧮 du §15.9.2 (le nombre d'entraînements, la probabilité qu'au moins un essai soit bon) et l'encadré 🕰️ du §15.9.4 sur le successive halving.

</details>
<details><summary>Indice 2</summary>

Une grille de 5 hyperparamètres à 5 valeurs a $5^5$ combinaisons ; chacune coûte 5 entraînements de 4 minutes. Pour le successive halving, écris pour chaque tour le nombre de candidats et la part des données qu'ils reçoivent, puis additionne les produits.

</details>
<details><summary>Indice 3</summary>

a) $5^5 \times 5 = 3\,125 \times 5 = 15\,625$ entraînements. b) Multiplie par 4 minutes, puis divise par $60 \times 24$. c) Divise b) par 8. d) $60 \times 5 \times 4$ minutes, divisées par 8 cœurs, puis par 60. e) $1 - 0{,}95^{60}$. f) Le plus petit entier $n$ tel que $n \ge \ln 0{,}01 / \ln 0{,}95$. g) Tours : $81 \times \frac{1}{81}$, $27 \times \frac{3}{81}$, $9 \times \frac{9}{81}$, $3 \times \frac{27}{81}$, $1 \times 1$ ; additionne. h) $81$ divisé par g).

</details>

### Ex 15.6 — Le Pipeline expliqué en 5 lignes

<details><summary>Indice 1</summary>

Relis les §15.9.5 et §15.9.7 de la fiche, et la figure des folds avec et sans pipeline.

</details>
<details><summary>Indice 2</summary>

Une bonne image : une recette ou une chaîne de montage, dont chaque poste est « réglé » sur les seuls ingrédients d'entraînement. L'essentiel à dire : à l'entraînement, chaque étape apprend puis transforme ; à la prédiction, elle ne fait que transformer ; dans une validation croisée, tout est réappris à chaque tour, sur les seuls folds d'entraînement.

</details>
<details><summary>Indice 3</summary>

Plan en cinq lignes : (1) l'image ; (2) ce que fait `fit` du pipeline (`fit_transform` sur chaque transformateur, `fit` sur le modèle) et ce que fait `predict` (`transform` partout, puis `predict`) ; (3) ce qui se passe dans `cross_val_score(pipe, X, y)`, tour par tour ; (4) pourquoi standardiser avant la validation croisée laisse le fold de validation influencer la préparation ; (5) quand l'effet est petit (une mise à l'échelle sur beaucoup d'exemples) et quand il est énorme (une étape qui regarde la cible).

</details>

### Ex 15.7 — Boston Housing : un dataset retiré pour raisons éthiques

<details><summary>Indice 1</summary>

Lis l'avertissement de la documentation de `load_boston` (version 1.1) et l'article de Carlisle : ils répondent aux questions 1 et 2.

</details>
<details><summary>Indice 2</summary>

Pour 1, regarde la forme de la fonction $1\,000\,(B_k - 0{,}63)^2$ : où est son minimum, et que vaut-elle pour deux valeurs de $B_k$ symétriques autour de 0,63 ? Pour 2, demande-toi ce que d'autres colonnes, la cible et l'usage même du dataset en cours transmettent. Pour 4, ouvre `data/cards/california_housing.md`.

</details>
<details><summary>Indice 3</summary>

Structure possible : (1) une parabole de minimum en 0,63, qui encode l'hypothèse d'un effet de la ségrégation sur les prix, et qui envoie deux villes différentes sur la même valeur ; (2) les autres colonnes, la cible et les résultats publiés restent construits dans le même cadre, et l'habitude d'utiliser ce dataset sans contexte est elle-même le problème ; (3) le cycle de dépréciation : un avertissement, des alternatives, du temps pour migrer ; (4) la date du recensement, l'unité (le groupe d'îlots), le plafond de la cible ; (5) qui a collecté, pourquoi, avec quelles variables construites et quelles limites documentées.

</details>

### Ex 15.8 — Lire l'article sur la conception de l'API de scikit-learn

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

Les indices des exercices du notebook (15.9 à 15.31) arriveront avec eux, à la prochaine session de génération.
