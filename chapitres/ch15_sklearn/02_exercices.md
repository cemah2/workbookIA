# 15 · scikit-learn — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch15_sklearn/06_mes_reponses.md` (créée par `python tools/start_chapter.py 15`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 15.1 à 15.4 et de l'estimation 🧮 15.5 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.7` ou `"0,7"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), une liste pour plusieurs nombres (`[2, 5]`).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ Papier-crayon](#papier) · [🧮 🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 ou 4 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 15.Q1 — Coder soi-même ou utiliser une bibliothèque ? 🧠 ⏱️ 3 min
*Fiche §15.1, §15.2 · livre §15.1, §15.2 · parcours R*

a) Vrai ou faux : dans ce chapitre, le livre écrit lui-même le code de chaque algorithme qu'il utilise.
b) Ce qu'apporte d'abord une bibliothèque largement utilisée par rapport à du code maison : (A) du code relu, testé et rendu rapide par une communauté, utilisé par beaucoup de projets ; (B) la garantie que le modèle convient à tes données ; (C) la dispense de comprendre les algorithmes ; (D) l'absence d'hyperparamètres à régler.
c) Vrai ou faux : les calculs de scikit-learn s'appuient sur NumPy et SciPy.
d) Pour entraîner un grand réseau de neurones profond sur GPU, l'outil du workbook est : (A) scikit-learn ; (B) PyTorch ; (C) SciPy ; (D) seaborn.

### 15.Q2 — Importer proprement et lire une dépréciation 🧠 ⏱️ 3 min
*Fiche §15.3 · livre §15.3 · parcours R, C*

a) Après `from sklearn import linear_model`, la ligne qui crée un modèle Ridge : (A) `Ridge()` ; (B) `linear_model.Ridge()` ; (C) `sklearn.Ridge()` ; (D) `linear_model()`.
b) Un avertissement « *'algorithm' is deprecated in 1.6 and will be removed in 1.8* » signifie : (A) ton code plante déjà ; (B) il marche encore, mais l'argument disparaîtra dans la version 1.8 ; (C) l'argument n'a jamais existé ; (D) il faut réinstaller scikit-learn.
c) Vrai ou faux : un programme qui tournait sans erreur avec scikit-learn 0.19 tourne forcément avec la version 1.6.
d) Le meilleur réflexe devant un `FutureWarning` : (A) masquer les avertissements ; (B) lire la documentation de la version installée et modifier l'appel ; (C) revenir à l'ancienne version ; (D) l'ignorer tant que le code tourne.
e) Vrai ou faux : l'adresse `scikit-learn.org/stable/` montre toujours la documentation de la version installée sur ta machine.

### 15.Q3 — Hyperparamètres au constructeur, attributs appris avec « _ » 🧠 ⏱️ 3 min
*Fiche §15.4, §15.4.1 · livre §15.4, §15.4.1 · parcours R, C*

a) Dans `Ridge(alpha=2.0)`, `alpha` est : (A) un paramètre appris par `fit` ; (B) un hyperparamètre ; (C) une feature ; (D) un label.
b) Vrai ou faux : juste après `model = Ridge(alpha=2.0)`, l'attribut `model.coef_` existe.
c) Vrai ou faux : `model.get_params()` renvoie un dictionnaire qui contient la clé `"alpha"`.
d) `clone(model)` renvoie : (A) une copie de l'objet avec tout ce qu'il a appris ; (B) un objet neuf du même type, aux mêmes hyperparamètres, non entraîné ; (C) une copie entraînée sur de nouvelles données ; (D) la liste des hyperparamètres.
e) Vrai ou faux : le constructeur d'un estimateur de scikit-learn vérifie les données et commence l'apprentissage.

### 15.Q4 — Ce que renvoient fit et predict 🧠 ⏱️ 3 min
*Fiche §15.4.2, §15.4.3 · livre §15.4.2, §15.4.3 · parcours R, C*

a) `ridge.fit(X, y)` renvoie : (A) `None` ; (B) les prédictions sur `X` ; (C) l'objet `ridge` lui-même ; (D) le score sur `X`.
b) `X` contient 100 exemples d'une seule feature. La forme qu'attend `fit` pour `X` : (A) `(100,)` ; (B) `(100, 1)` ; (C) `(1, 100)`.
c) La forme de `ridge.predict(X_new)` quand `X_new` a la forme `(5, 1)` : (A) `(5,)` ; (B) `(5, 1)` ; (C) `(1, 5)` ; (D) `(1,)`.
d) Vrai ou faux : avec scikit-learn 1.6, `ridge.predict(3.0)` renvoie la prédiction en $x = 3$.
e) Vrai ou faux : appeler `fit` une seconde fois sur d'autres données reprend l'apprentissage là où il s'était arrêté.

### 15.Q5 — decision_function ou predict_proba ? 🧠 ⏱️ 3 min
*Fiche §15.4.4 · livre §15.4.4 · parcours R*

Une régression logistique est entraînée sur un problème à **deux** classes ; `X_new` contient 4 exemples.

a) La forme de `decision_function(X_new)` : (A) `(4,)` ; (B) `(4, 1)` ; (C) `(4, 2)` ; (D) `(2, 4)`.
b) La forme de `predict_proba(X_new)`, avec les mêmes choix.
c) Le même modèle entraîné sur un problème à **trois** classes : la forme de `decision_function(X_new)` : (A) `(4,)` ; (B) `(4, 2)` ; (C) `(4, 3)` ; (D) `(3, 4)`.
d) À deux classes, la probabilité de la classe `classes_[1]` donnée par `predict_proba` pour un exemple dont le score vaut 0.
e) Vrai ou faux : `RidgeClassifier` a une méthode `predict_proba`.
f) Vrai ou faux : à deux classes, un `SVC(probability=True)` peut prédire avec `predict` une classe dont la probabilité donnée par `predict_proba` est inférieure à 0,5.

### 15.Q6 — KMeans : n_clusters, labels_ et choix de k 🧠 ⏱️ 3 min
*Fiche §15.5 · livre §15.5 · parcours R*

a) La valeur par défaut de `n_clusters` de `KMeans`.
b) Vrai ou faux : à la création d'un `KMeans`, il faut lui dire combien de features auront les données.
c) Après `km.fit(X)`, `km.labels_` contient : (A) les vraies classes des exemples ; (B) le numéro du cluster de chaque exemple de `X` ; (C) les coordonnées des centres ; (D) l'inertie.
d) Vrai ou faux : comme l'inertie baisse quand $k$ augmente, il suffit de prendre le $k$ qui donne la plus petite inertie.
e) L'algorithme qui laisse de côté les points isolés (label −1) et trouve seul le nombre de groupes : (A) `KMeans` ; (B) `DBSCAN` ; (C) `AgglomerativeClustering(n_clusters=3)`.

### 15.Q7 — fit, transform, fit_transform, inverse_transform 🧠 ⏱️ 3 min
*Fiche §15.6, §15.6.1 · livre §15.6, §15.6.1 · parcours R, C*

Un `MinMaxScaler` est appris sur une feature qui va de 20 à 60 à l'entraînement.

a) Pour transformer le test, on appelle sur ce scaler : (A) `fit(X_test)` ; (B) `transform(X_test)` ; (C) `fit_transform(X_test)` ; (D) `inverse_transform(X_test)`.
b) La valeur transformée d'une valeur de test de 70.
c) La valeur transformée de 10.
d) Le résultat d'`inverse_transform` pour la valeur transformée 0,75.
e) Vrai ou faux : tout transformateur de scikit-learn a une méthode `inverse_transform`.

### 15.Q8 — Ce que la PCA fait vraiment aux features 🧠 ⏱️ 3 min
*Fiche §15.7 · livre §15.7 · parcours R, M*

a) Après `PCA(n_components=2)` sur 3 features, chaque nouvelle feature est : (A) l'une des trois features d'origine ; (B) une combinaison linéaire des trois features centrées ; (C) le produit de deux features ; (D) une feature d'origine divisée par son écart-type.
b) Vrai ou faux : la PCA décide quelle feature d'origine supprimer.
c) Vrai ou faux : avec `whiten=True`, chaque composante produite sur l'entraînement a une variance de 1.
d) Vrai ou faux : `whiten=True` change les directions (`components_`) trouvées par la PCA.
e) Vrai ou faux : la PCA se sert des labels `y` pour choisir ses directions.

### 15.Q9 — Un ensemble est un estimateur comme les autres 🧠 ⏱️ 3 min
*Fiche §15.8 · livre §15.8 · parcours R*

a) Vrai ou faux : un `AdaBoostClassifier` s'entraîne avec `fit` et prédit avec `predict`, comme un seul arbre.
b) Le nom de l'argument qui donne le modèle de base d'`AdaBoostClassifier` dans scikit-learn 1.6 : (A) `base_estimator` ; (B) `estimator` ; (C) `model` ; (D) `algorithm`.
c) Son modèle de base par défaut : (A) un arbre sans limite de profondeur ; (B) une souche, un arbre de profondeur 1 ; (C) une régression logistique ; (D) un `RidgeClassifier`.
d) Vrai ou faux : après `fit`, `AdaBoostClassifier(n_estimators=50)` contient toujours exactement 50 membres.
e) AdaBoost combine ses membres par : (A) une voix par membre, la classe la plus citée gagnant ; (B) un vote pondéré par le poids $\alpha_m$ de chaque membre ; (C) la moyenne de leurs probabilités, à poids égaux.

### 15.Q10 — Grille ou recherche aléatoire ? 🧠 ⏱️ 4 min
*Fiche §15.9 à §15.9.4 · livre §15.9 à §15.9.4 · parcours R*

a) Le nombre de clusters d'un k-means est : (A) un paramètre appris ; (B) un hyperparamètre.
b) Vrai ou faux : sur un seul cœur, `GridSearchCV` entraîne chaque modèle plus vite que des boucles écrites à la main.
c) Une grille de trois hyperparamètres, avec 3, 4 et 2 valeurs : le nombre de combinaisons.
d) Le nombre de combinaisons qu'essaie `RandomizedSearchCV` par défaut.
e) Vrai ou faux : quand un seul de deux hyperparamètres compte vraiment, neuf essais au hasard essaient en général plus de valeurs différentes de cet hyperparamètre qu'une grille 3 × 3.
f) Vrai ou faux : pour une régularisation à explorer entre 0,001 et 1 000, une distribution `loguniform` convient mieux qu'une distribution uniforme.

### 15.Q11 — Pipeline : step__param et absence de fuite 🧠 ⏱️ 4 min
*Fiche §15.9.5 à §15.9.7 · livre §15.9.5 à §15.9.7 · parcours R*

On travaille avec `pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC())])`.

a) Le nom à mettre dans une grille sur `pipe` pour l'hyperparamètre `C` du SVC : (A) `"C"` ; (B) `"clf_C"` ; (C) `"svc__C"` ; (D) `"clf__C"`.
b) Vrai ou faux : dans `cross_val_score(pipe, X, y, cv=5)`, le scaler est réappris sur la partie entraînement de chaque tour.
c) À chaque tour, les données du fold de validation passent par : (A) `fit_transform` du scaler ; (B) `transform` du scaler appris sur les autres folds ; (C) aucune transformation ; (D) un scaler appris une fois sur toutes les lignes.
d) Le nom que `make_pipeline(StandardScaler(), SVC())` donne à l'étape `SVC()` : (A) `"SVC"` ; (B) `"svc"` ; (C) `"clf"` ; (D) `"step_1"`.
e) Vrai ou faux : la dernière étape d'un `Pipeline` doit être un transformateur.

### 15.Q12 — Datasets, train_test_split et documentation 🧠 ⏱️ 3 min
*Fiche §15.10 à §15.12 · livre §15.10 à §15.12 · parcours R*

a) La part du test de `train_test_split` quand on ne donne ni `test_size` ni `train_size`.
b) Vrai ou faux : `train_test_split` mélange les exemples par défaut.
c) Pour garder les proportions des classes dans l'entraînement et le test : (A) `shuffle=False` ; (B) `stratify=y` ; (C) `random_state=0` ; (D) `test_size=0.5`.
d) Vrai ou faux : `load_boston` existe encore dans scikit-learn 1.6.
e) 125 exemples et `test_size=0.3` : le nombre d'exemples du test (scikit-learn arrondit la taille du test au-dessus).
f) Le générateur de deux cercles emboîtés : (A) `make_moons` ; (B) `make_circles` ; (C) `make_blobs` ; (D) `make_classification`.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 15.R1 — Ch. 14 : Forêt aléatoire ou gradient boosting ? 🔁 ★ ⏱️ 5 min
*Ch. 14 (fiche, bilan : bagging ou boosting ?) · parcours R, M, C*

a) Celui qui entraîne ses arbres indépendamment les uns des autres : (A) la forêt aléatoire ; (B) le gradient boosting.
b) Vrai ou faux : ajouter des arbres à une forêt aléatoire finit par la faire surapprendre.
c) Le gradient boosting réduit surtout : (A) le biais ; (B) la variance.
d) L'hyperparamètre qui règle la taille de chaque pas du gradient boosting : (A) `max_features` ; (B) `learning_rate` ; (C) `n_jobs` ; (D) `bootstrap`.
e) Vrai ou faux : le score out-of-bag est une validation gratuite que donne un gradient boosting avec ses réglages par défaut.

### 15.R2 — Ch. 12 : Où ajuster le scaler dans une validation croisée ? 🔁 ★ ⏱️ 5 min
*Ch. 12 (fiche §12.10, transformations et validation croisée) · parcours R, M, C*

a) Vrai ou faux : standardiser tout le dataset avant une validation croisée est une fuite de données.
b) À chaque tour, le scaler s'apprend sur : (A) tout le dataset ; (B) les folds d'entraînement du tour ; (C) le fold de validation du tour ; (D) le jeu de test final.
c) Un `StandardScaler` a appris une moyenne de 50 et un écart-type de 10 sur l'entraînement. La valeur transformée de 65.
d) Vrai ou faux : une sélection de features faite sur tout le dataset avant la validation croisée peut donner un bon score même sur des features de pur bruit.
e) Laquelle de ces deux étapes apprend quelque chose des données, et doit donc s'apprendre sur l'entraînement seul ? (A) remplacer les valeurs manquantes par la médiane de la colonne ; (B) prendre le logarithme d'une feature positive.

### 15.R3 — Ch. 8 : Combien de modèles pour une validation croisée à 5 folds ? 🔁 ★ ⏱️ 5 min
*Ch. 8 (fiche §8.5, la validation croisée) · parcours R, M, C*

a) Le nombre de modèles entraînés par une validation croisée à 5 folds (sans réentraînement final).
b) Combien de fois chaque exemple sert-il à la validation ?
c) Et à l'entraînement ?
d) $n = 1\,003$ exemples et 5 folds sans mélange (`KFold`, qui donne un exemple de plus aux premiers folds quand $k$ ne divise pas $n$) : la taille du plus grand fold.
e) Le nombre de modèles entraînés par un leave-one-out sur 40 exemples.
f) Vrai ou faux : `StratifiedKFold` garde dans chaque fold à peu près les proportions des classes.

<a id="papier"></a>

## ✏️ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 15.1 — Compter les modèles entraînés par une recherche en grille ✏️ ★ ⏱️ 10 min
**Objectif :** compter les entraînements d'une recherche en grille ou au hasard, avec le réentraînement final, une grille en liste et une validation croisée imbriquée.
**Prérequis :** ch. 8 (validation croisée à k folds) · fiche §15.9.2, §15.9.3, §15.9.5 · **Parcours :** R, M

On lance

```python
search = GridSearchCV(RidgeClassifier(), param_grid={"alpha": [0.01, 0.1, 1, 10],
                                                     "solver": ["svd", "cholesky", "lsqr"]}, cv=5)
search.fit(X_train, y_train)          # refit=True par défaut
```

a) Le nombre de candidats (combinaisons) de la grille.
b) Le nombre d'entraînements faits pendant les validations croisées de la recherche, sans le réentraînement final.
c) Le nombre total d'entraînements que fait `search.fit`.
d) La même recherche, sans l'argument `cv`, avec scikit-learn 0.19, dont le défaut était de 3 folds : le nombre total d'entraînements.
e) Un pipeline `make_pipeline(PolynomialFeatures(), RidgeClassifier())` et la grille `{"polynomialfeatures__degree": [1, 2, 3, 4, 5], "polynomialfeatures__interaction_only": [True, False], "ridgeclassifier__alpha": [0.1, 1, 10]}`, avec `cv=4` : le nombre total d'entraînements du pipeline, réentraînement final compris.
f) Une grille donnée comme une liste de deux dictionnaires, `[{"kernel": ["linear"], "C": [0.1, 1, 10]}, {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": [0.01, 0.1]}]`, avec `cv=5` et `refit=False` : le nombre d'entraînements.
g) `RandomizedSearchCV(..., n_iter=20, cv=5)`, avec le réentraînement final par défaut : le nombre total d'entraînements.
h) On évalue la recherche de l'énoncé par une validation croisée imbriquée : `cross_val_score(search, X, y, cv=3)`, où `search` est l'objet `GridSearchCV` ci-dessus. Le nombre total d'entraînements de `RidgeClassifier`.

Dans ta copie : à quoi sert le réentraînement final ? Pourquoi `search.best_score_` n'est-il pas une estimation honnête de la performance sur des données nouvelles, et que mesure h) à sa place ?

### Ex 15.2 — De decision_function à predict_proba ✏️ ★ ⏱️ 10 min
**Objectif :** passer du score d'un classifieur linéaire à sa probabilité et à sa classe, à deux classes puis à trois, et traduire un seuil de probabilité en seuil de score.
**Prérequis :** ch. 13 (régression logistique, sigmoïde) · fiche §15.4.4 · **Parcours :** R, M

Une régression logistique à deux classes (0 et 1) donne, pour cinq exemples, les scores `decision_function` suivants : $2{,}0$ ; $-1{,}0$ ; $0{,}0$ ; $0{,}5$ ; $-3{,}2$.

a) La probabilité de la classe 1 donnée par `predict_proba` pour le premier exemple (4 décimales).
b) La même probabilité pour le deuxième exemple (4 décimales).
c) Pour le troisième exemple.
d) Les classes que renvoie `predict` pour les cinq exemples, dans l'ordre (une liste).
e) Le score qui correspond à une probabilité de 0,9 (4 décimales).
f) On décide de ne classer 1 qu'un exemple de probabilité au moins 0,8. Le seuil de score équivalent (4 décimales).
g) Avec ce seuil, combien des cinq exemples sont classés 1 ?
h) Une régression logistique multinomiale à trois classes donne à un exemple les scores $(2 ;\ 1 ;\ -1)$. La probabilité de la première classe (4 décimales).
i) Vrai ou faux : ajouter 5 aux trois scores de h) change les probabilités des trois classes.
j) Vrai ou faux : la sigmoïde du `decision_function` d'un `RidgeClassifier` donne des probabilités calibrées.

Dans ta copie : pourquoi la sigmoïde convient-elle à la régression logistique et pas au `RidgeClassifier` ? Que faudrait-il faire pour obtenir des probabilités utilisables à partir de ce dernier ?

### Ex 15.3 — Ramener une droite ajustée dans l'espace d'origine ✏️ ★★ ⏱️ 15 min
**Objectif :** utiliser une mise à l'échelle min-max et son inverse pour ramener dans les unités d'origine une droite ajustée sur des données transformées, comme le livre au §15.6.1.
**Prérequis :** ch. 12 (normalisation min-max) · fiche §15.6, §15.6.1 · **Fil rouge :** données synthétiques · **Parcours :** M

Un `MinMaxScaler` est appris sur des points d'entraînement à deux features, $x$ et $y$ : à l'entraînement, $x$ va de $-3$ à $5$ et $y$ de $10$ à $30$. On ajuste ensuite une droite aux points **transformés** et l'on trouve $y' = 0{,}6\,x' + 0{,}25$.

a) La valeur transformée $x'$ de $x = 1$.
b) La prédiction transformée $y'$ en $x = 1$.
c) La prédiction dans l'unité d'origine de $y$, en $x = 1$.
d) L'extrémité gauche de la droite (en $x' = 0$), ramenée dans l'espace d'origine : sa coordonnée $y$.
e) L'extrémité droite (en $x' = 1$), ramenée dans l'espace d'origine : sa coordonnée $y$.
f) La pente de la droite dans l'espace d'origine.
g) Son ordonnée à l'origine (la valeur de $y$ en $x = 0$).
h) Un point de test a $x = 11$ : sa valeur transformée $x'$.
i) La prédiction dans l'unité d'origine en $x = 11$.
j) Vrai ou faux : on obtient la droite d'origine en appliquant `inverse_transform` au couple $(0{,}6 ;\ 0{,}25)$ (la pente et l'ordonnée), comme si c'était un point.

Dans ta copie : écris la pente d'origine en fonction de la pente $a$ dans l'espace transformé et des étendues de $x$ et de $y$. Pourquoi une droite reste-t-elle une droite quand on la ramène dans l'espace d'origine ? Le resterait-elle après une transformation logarithmique de $y$ ?

### Ex 15.4 — Combien de features polynomiales ? ✏️ ★★ ⏱️ 20 min
**Objectif :** compter les features que produit `PolynomialFeatures` avec le coefficient binomial, et en estimer le coût en mémoire.
**Prérequis :** ch. 9 (features polynomiales) · Ex 14.5 (coefficient binomial) · fiche §15.9.5 · **Parcours :** M

Sauf mention contraire, `PolynomialFeatures(degree=d)` garde la constante 1 (`include_bias=True`) et tous les monômes de degré au plus $d$.

a) Avec $p = 2$ features et $d = 2$ : le nombre de features produites.
b) Avec $p = 2$ et $d = 5$.
c) Avec $p = 2$ : le nombre de monômes de degré **exactement** 4.
d) Avec $p = 8$ et $d = 2$.
e) Avec $p = 8$ et $d = 3$.
f) Avec $p = 20$, $d = 3$ et `include_bias=False`.
g) Avec $p = 5$, $d = 2$ et `interaction_only=True` (constante comprise).
h) $n = 100\,000$ exemples, $p = 20$ et $d = 3$ (constante comprise), en `float64` (8 octets par nombre) : la taille du tableau produit, en Go ($10^9$ octets, 2 décimales).
i) Le plus petit degré $d$ pour lequel $p = 10$ features donnent plus de 1 000 features (constante comprise).

Dans ta copie : démontre la formule $\binom{p + d}{d}$ par l'argument des étoiles et des barres de la fiche. Que vaut le nombre de monômes de degré **exactement** $d$ ? Avec $p = 2$ et un degré qui monte de 1 à 6, comment croît le nombre de features : comme $d$, comme $d^2$, comme $2^d$ ?

<a id="reflexion"></a>

## 🧮 🗣️ ⚖️ 📄 Réflexion

### Ex 15.5 — Budget d'une recherche d'hyperparamètres 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer à la main le coût d'une recherche en grille, au hasard et par successive halving, et le nombre d'essais au hasard nécessaires.
**Prérequis :** Ex 15.1 · fiche §15.9.2, §15.9.4 · **Parcours :** M

Un gradient boosting met 4 minutes à s'entraîner sur ton ordinateur, avec un seul cœur. Tu veux régler 5 hyperparamètres, chaque réglage étant noté par une validation croisée à 5 folds. On néglige le temps de prédiction et le réentraînement final.

a) Une grille de 5 valeurs par hyperparamètre : le nombre d'entraînements.
b) La durée de cette recherche sur un cœur, en jours (1 décimale).
c) La même durée avec 8 cœurs, si le calcul se répartit parfaitement (1 décimale).
d) Une recherche au hasard de 60 essais, toujours avec 5 folds et 8 cœurs : sa durée en heures (1 décimale).
e) La probabilité qu'au moins un des 60 essais tombe dans les 5 % meilleurs réglages de l'espace exploré (4 décimales).
f) Le plus petit nombre d'essais au hasard pour que cette probabilité atteigne au moins 0,99.
g) Un successive halving part de 81 candidats entraînés sur $1/81$ des données, garde le meilleur tiers à chaque tour et triple les données, jusqu'à un seul candidat sur toutes les données (5 tours). On suppose le temps d'entraînement proportionnel à la taille des données. Le coût total, par fold, en « entraînements sur toutes les données ».
h) Le rapport entre le coût d'une grille qui entraînerait ces 81 candidats sur toutes les données et celui de g) (1 décimale).

Dans ta copie : vérifie le calcul du livre (quatre hyperparamètres à six valeurs, une heure par modèle : « 54 jours »). Quelles hypothèses de ces estimations sont les plus fragiles (temps constant d'un réglage à l'autre, parallélisme parfait, « 5 % meilleurs » de l'espace exploré) ?

### Ex 15.6 — Le Pipeline expliqué en 5 lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce qu'est un pipeline et pourquoi il empêche la fuite de données.
**Prérequis :** ch. 12 (fuite en validation croisée) · fiche §15.9.5, §15.9.7 · **Parcours :** R

Un collègue qui débute standardise ses données, puis lance une validation croisée ; il ne voit pas l'intérêt du `Pipeline` que tu lui conseilles. Explique-lui, en **cinq lignes** au plus :
- une image de la vie courante ;
- ce que fait un pipeline à l'entraînement et à la prédiction ;
- ce qui se passe, fold par fold, quand on le passe à `cross_val_score` ;
- pourquoi son code actuel donne un score trop beau, et quand l'écart devient énorme.

Enregistre-toi ou écris ta réponse, puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 15.7 — Boston Housing : un dataset retiré pour raisons éthiques ⚖️ ★★ ⏱️ 20 min
**Objectif :** comprendre pourquoi un dataset « classique » a été retiré, et en tirer des règles pour choisir et documenter des données.
**Prérequis :** fiche §15.10 · `data/cards/california_housing.md` · **Fil rouge :** California · **Parcours :** R

Le livre charge Boston Housing avec `load_boston()`. Ce dataset (Harrison et Rubinfeld, 1978 : 506 districts de la région de Boston, 13 features, le prix médian des logements comme cible) a été retiré de scikit-learn en 2022, à la version 1.2. Lis l'avertissement de la documentation de la version 1.1 ([`load_boston`](https://scikit-learn.org/1.1/modules/generated/sklearn.datasets.load_boston.html)) et l'article de M. Carlisle qu'il cite ([« Racist data destruction? »](https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8)). Réponds dans ta copie.

1. La feature `B` vaut $1\,000\,(B_k - 0{,}63)^2$, où $B_k$ est la part d'habitants noirs de la ville. Pourquoi ses auteurs l'ont-ils construite ainsi, et quelle hypothèse cette forme impose-t-elle ? Pourquoi dit-on qu'elle n'est pas « inversible » ?
2. Pourquoi supprimer la colonne `B` ne suffit-il pas à rendre ce dataset sans problème pour un exercice sur la prédiction des prix ?
3. Les mainteneurs ne l'ont pas supprimé sans prévenir : il a d'abord été déprécié en 1.0. Qu'apporte cette procédure aux utilisateurs ? Que doit faire une équipe dont un vieux projet en dépend ?
4. Le workbook utilise California Housing. Lis sa data card : quelles limites a-t-il, lui aussi (date, unité géographique, valeurs plafonnées) ?
5. Rédige trois questions à se poser avant d'adopter un dataset « classique » pour un projet ou un cours.

### Ex 15.8 — Lire l'article sur la conception de l'API de scikit-learn 📄 ★★ ⏱️ 30 min
**Objectif :** lire un article de conception logicielle et relier ses principes à ce que tu as fait dans `mylearn`.
**Prérequis :** Ex 15.6 · fiche §15.2, §15.4, §15.9.3 · **Parcours :** C

Lis L. Buitinck et coll., [« API design for machine learning software: experiences from the scikit-learn project »](https://arxiv.org/abs/1309.0238) (atelier d'ECML PKDD 2013) : l'introduction, la section 2 (*Core API*) et la section 3 (*Advanced API*) ; survole les sections 4 à 6. Réponds dans ta copie.

1. Quels principes de conception l'article énonce-t-il ? Pour chacun, donne un exemple pris dans ce chapitre.
2. Quelles sont les trois interfaces de base (estimateur, prédicteur, transformateur) ? Que fait le constructeur, que fait `fit`, et pourquoi les hyperparamètres sont-ils rangés dans des attributs publics ?
3. Comment l'article représente-t-il les données ? Pourquoi ce choix plutôt que des classes de données propres à la bibliothèque ?
4. Qu'est-ce qu'un méta-estimateur ? Comment les pipelines et la sélection de modèles (`GridSearchCV`) s'appuient-ils sur l'interface commune ?
5. Pourquoi l'article préfère-t-il le *duck typing* à l'héritage obligatoire ? Quelle limite pratique en vois-tu pour une classe de ta librairie `mylearn` ?
6. Quelles limites ou difficultés l'article reconnaît-il (pense à la sauvegarde des modèles, au parallélisme, aux algorithmes absents) ? Lesquelles ont trouvé une réponse depuis 2013 ? Et quelle limite de la représentation des données n'évoque-t-il pas, que `ColumnTransformer` et `set_output` ont comblée ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 15.E1 — fit, transform, predict : pourquoi jamais fit sur le test ? 💼 ★★ ⏱️ 10 min
*Fiche §15.6, §15.9.7 · parcours R*

« Expliquez la différence entre `fit`, `transform`, `fit_transform` et `predict`. Pourquoi ne faut-il jamais appeler `fit` sur le jeu de test ? »

### 15.E2 — Grille, hasard ou optimisation bayésienne ? 💼 ★★ ⏱️ 10 min
*Fiche §15.9.2 à §15.9.4 · parcours R*

« Vous devez régler une dizaine d'hyperparamètres d'un gradient boosting avec un budget de calcul limité. Recherche en grille, au hasard, ou autre chose ? Comment évaluez-vous le modèle final ? »

### 15.E3 — Pipeline et ColumnTransformer sur des données hétérogènes 💼 ★★ ⏱️ 10 min
*Fiche §15.9.5, §15.9.7, au-delà du livre : les données hétérogènes · parcours R*

« Votre dataset a des colonnes numériques avec des valeurs manquantes et des colonnes catégorielles. Comment construisez-vous la préparation dans scikit-learn, et pourquoi ? »

### 15.E4 — predict_proba donne-t-il une vraie probabilité ? 💼 ★★ ⏱️ 10 min
*Fiche §15.4.4, au-delà du livre : seuil et calibration · parcours R*

« Votre modèle renvoie 0,8 pour un client. Peut-on dire qu'il a 80 % de chances de résilier ? Comment le vérifier, et comment choisir le seuil de décision ? »

### 15.E5 — Mettre un modèle scikit-learn en production 💼 ★★ ⏱️ 10 min
*Fiche §15.12, au-delà du livre : sauvegarder un modèle · parcours R*

« Votre modèle est prêt dans un notebook. Comment le livrez-vous pour qu'il serve en production et qu'on puisse reproduire ses résultats dans six mois ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch15_sklearn/03_notebook.ipynb`). Ce chapitre n'ajoute aucun module à ta librairie : il te fait pratiquer scikit-learn, et comparer ta librairie à la bibliothèque. La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels, aux exercices ✏️ et à l'estimation 🧮 ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 15.9 | Anatomie d'un estimateur : Ridge sur un sinus bruité | 📦 | ★ | 15 |
| 15.10 | predict_proba : prédire ce que valent les probabilités de trois classifieurs | 🔮 | ★ | 10 |
| 15.11 | Datasets synthétiques, fetch_* et train_test_split stratifié | 📦 | ★ | 15 |
| 15.12 | Faire tourner le code de 2018 avec scikit-learn 1.6 | 🐛 | ★★ | 25 |
| 15.13 | Clustering avec scikit-learn : KMeans (le coude du livre), DBSCAN et clustering hiérarchique sur deux lunes et sept blobs | 📦 | ★★ | 25 |
| 15.14 | Transformateurs : test hors de [0, 1] et inverse_transform | 📦 | ★★ | 20 |
| 15.15 | PCA de 3D vers 2D et blanchiment | 📦 | ★★ | 20 |
| 15.16 | Cinq spirales : quel ensemble les démêle ? | 🔬 | ★★ | 30 |
| 15.17 | cross_validate, KFold, StratifiedKFold et RidgeClassifierCV | 📦 | ★★ | 25 |
| 15.18 | GridSearchCV : best_params_, cv_results_ et score d'entraînement | 📦 | ★★ | 25 |
| 15.19 | RandomizedSearchCV avec des distributions | 📦 | ★★ | 25 |
| 15.20 | Grille ou hasard : qui gagne avec 16 essais ? | 🔮 | ★★ | 20 |
| 15.21 | Frontières de décision : DecisionBoundaryDisplay et surfaces 3D | 📦 | ★★ | 20 |
| 15.22 | Pipeline polynomial + ridge : degré et alpha ensemble | 📦 | ★★★ | 35 |
| 15.23 | Lire cv_results_ : carte de chaleur et écarts-types | 📈 | ★★ | 20 |
| 15.24 | Refaire l'expérience de fuite du ch. 8 (sélection de features sur du bruit) avec un Pipeline | 🔬 | ★★ | 25 |
| 15.25 | Régler le seuil de décision d'un classifieur | 🔬 | ★★★ | 35 |
| 15.26 | Classes déséquilibrées : class_weight, rééchantillonnage et la bonne métrique (recall, PR-AUC) | 📦 | ★★★ | 35 |
| 15.27 | La même validation croisée pour six familles de modèles scikit-learn : score moyen, écart-type et temps | 📦 | ★★★ | 40 |
| 15.28 | ColumnTransformer sur Penguins brut | 📦 | ★★★ | 45 |
| 15.29 | Quand scikit-learn refuse un estimateur mylearn | 🐛 | ★★★ | 35 |
| 15.30 | Sauvegarder, recharger et versionner un pipeline | 🛠️ | ★★ | 20 |
| 15.31 | Défi : prédire le sexe des manchots depuis le fichier brut | 🏆 | ★★★★ | 100 |
