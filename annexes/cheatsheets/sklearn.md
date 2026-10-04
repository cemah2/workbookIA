# Cheatsheet scikit-learn

> Aide-mémoire rempli au fil des chapitres (1, 3, 6 à 10, puis 12 à 15). Une ligne = une commande utile + ce qu'elle fait.

## L'API commune (fit, predict, transform)

| Code | Effet |
|---|---|
| `model = DecisionTreeClassifier(max_depth=2, random_state=0)` | crée le modèle : on fixe les **hyperparamètres** (ch. 1) |
| `model.fit(X_train, y_train)` | entraîne : apprend les **paramètres** ; `X` de forme `(n_samples, n_features)`, `y` de forme `(n_samples,)` ; renvoie le modèle |
| `model.predict(X_new)` | prédit un label par ligne de `X_new` |
| `model.score(X_test, y_test)` | accuracy pour un classifieur (R² pour une régression, ch. 9) |
| `model.predict_proba(X_new)` | une « probabilité » par classe, dans l'ordre de `model.classes_` ; à lire avec prudence : pour un arbre, ce sont les proportions de classes de la feuille atteinte (1.19) |
| `groups = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)` | non supervisé : `fit` ne reçoit que `X`, et `fit_predict` renvoie un groupe par ligne |
| `model.tree_`, `model.coefs_`, `model.classes_`, `model.n_features_in_` | ce qui a été appris : les attributs appris finissent par un tiret bas |

## Découper les données

| Code | Effet |
|---|---|
| `from sklearn.model_selection import train_test_split` | le hold-out (ch. 8) |
| `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)` | 20 % de test, $\lceil 0{,}2\,n \rceil$ exemples (arrondi vers le haut) ; `random_state` rend le découpage reproductible ; l'ordre de sortie est train, test pour chaque tableau |
| `train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)` | découpage stratifié : chaque classe garde sa proportion (refusé avec `shuffle=False`) |
| `train_test_split(X, y, test_size=0.2, shuffle=False)` | les **dernières** lignes en test, sans mélange : pour une série temporelle seulement (un fichier trié par classe donnerait un test d'une seule classe) |
| `KFold(n_splits=5, shuffle=True, random_state=0)` | k-fold, mélangée une fois ; `for train_idx, val_idx in cv.split(X):` donne les indices de chaque tour ; sans `shuffle`, des folds consécutifs, les premiers ayant un exemple de plus |
| `StratifiedKFold(n_splits=5)` ; `cv.split(X, y)` | k-fold stratifiée : chaque fold garde les proportions des classes (avertissement si une classe a moins de 5 membres) |
| `GroupKFold(n_splits=5)` ; `cv.split(X, y, groups)` | un groupe (patient, client, décennie) entier par fold ; `shuffle=True` depuis la version 1.6 ; `StratifiedGroupKFold` stratifie en plus |
| `TimeSeriesSplit(n_splits=5, gap=0)` | découpages chronologiques : l'entraînement toujours **avant** la validation, et de plus en plus long ; `gap` retire des exemples entre les deux |
| `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)` | k-fold répétée, en remélangeant à chaque répétition (`RepeatedKFold` sans stratification) |
| `LeaveOneOut()` | un exemple par fold : $n$ entraînements |

## Prétraitement

| Code | Effet |
|---|---|
| `PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)` | toutes les puissances et tous les produits des features jusqu'au degré 3 (ch. 9) ; `get_feature_names_out()` nomme les colonnes (`x0`, `x1`, `x0^2`, `x0 x1`, `x1^2`…) ; à standardiser ensuite si un modèle pénalisé suit |

## Modèles courants

| Code | Effet |
|---|---|
| `DecisionTreeClassifier(max_depth=2, random_state=0)` | arbre de décision (ch. 1 en boîte noire, ch. 13) ; `export_text(model, feature_names=...)` affiche ses règles |
| `KMeans(n_clusters=3, n_init=10, random_state=0)` | clustering en $k$ groupes (ch. 1, ch. 7) ; mettre les features à la même échelle avant |
| `MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, random_state=0)` | réseau de neurones à une couche cachée de 128 neurones (ch. 1, ch. 16) ; poids dans `coefs_`, biais dans `intercepts_` |
| `KMeans(n_clusters=3, random_state=0).fit(X)` | depuis la version 1.4, `n_init="auto"` : un seul départ avec k-means++ (la règle complète, selon `init` : 7.9) ; attributs `cluster_centers_`, `labels_`, `inertia_`, `n_iter_` (ch. 7) |
| `KMeans(..., init=C0, n_init=1, algorithm="lloyd", tol=1e-4)` | centres de départ imposés (un tableau `(k, n_features)`) : un seul départ, résultat reproductible (7.26) |
| `from sklearn.cluster import kmeans_plusplus` ; `centers, indices = kmeans_plusplus(X, n_clusters=3, random_state=0)` | seulement l'initialisation k-means++ (version « gloutonne ») |
| `DBSCAN(eps=0.5, min_samples=5).fit_predict(X)` | clustering par densité : pas de $k$, bruit noté −1, `min_samples` compte le point lui-même ; pas de `predict` ; standardiser d'abord (ch. 7) |
| `HDBSCAN(min_cluster_size=5).fit_predict(X)` | DBSCAN pour toutes les valeurs de `eps` à la fois : clusters de densités différentes ; −1 bruit, −2 valeur infinie, −3 valeur manquante ; depuis la version 1.3 |
| `NearestCentroid().fit(X, y)` | centroïde le plus proche (`sklearn.neighbors`) : `centroids_`, `classes_` ; ses `decision_function` et `predict_proba` (1.6) normalisent par l'écart-type intra-classe |
| `OneVsRestClassifier(LinearSVC())`, `OneVsOneClassifier(SVC())` | imposer une stratégie multi-classe (`sklearn.multiclass`) ; inutile en général : tous les classifieurs de scikit-learn gèrent plusieurs classes d'office (`SVC` fait de l'OvO, `LinearSVC` de l'OvR) |
| `MultiOutputClassifier(model)`, `ClassifierChain(model)` | multi-étiquette : `y` est une matrice binaire `(n_samples, n_labels)` ; un modèle par label, ou une chaîne où chaque modèle reçoit en plus les labels des précédents à l'entraînement, leurs prédictions au moment de prédire |
| `LinearRegression().fit(X, y)` | moindres carrés (ch. 9) : `coef_` (un poids par feature), `intercept_` ; `score` donne le R² |
| `Ridge(alpha=1.0)` | pénalité L2 : minimise $\lVert y - Xw - b \rVert^2 + \alpha \lVert w \rVert^2$ (une **somme** de carrés), l'ordonnée à l'origine n'est pas pénalisée ; standardiser les features avant ; `RidgeCV(alphas=...)` choisit `alpha` par validation croisée |
| `Lasso(alpha=0.1)`, `ElasticNet(alpha=0.1, l1_ratio=0.5)` | pénalité L1 (ou mélange L1 + L2) : minimise $\frac{1}{2n}\lVert y - Xw - b \rVert^2 + \alpha \lVert w \rVert_1$, donc un même `alpha` n'a pas la force qu'il a dans `Ridge` ; beaucoup de `coef_` exactement nuls ; `ElasticNet` remplace la pénalité par $\alpha\rho\lVert w \rVert_1 + \frac{\alpha(1 - \rho)}{2}\lVert w \rVert^2$, avec $\rho$ = `l1_ratio` ; `LassoCV` ; un `ConvergenceWarning` se règle en standardisant ou en augmentant `max_iter` |
| `alphas, coefs, _ = lasso_path(X, y)` | chemin de régularisation : les poids pour 100 valeurs décroissantes d'`alpha`, `coefs` de forme `(n_features, n_alphas)` ; pas d'ordonnée à l'origine : centrer `X` et `y` avant |
| `BayesianRidge().fit(X, y)` ; `model.predict(X_new, return_std=True)` | régression linéaire bayésienne (ch. 9) : prior gaussien sur les poids, précisions du bruit (`alpha_`, ce n'est pas une force de pénalité) et du prior (`lambda_`) estimées sur les données ; `return_std` donne l'incertitude de chaque prédiction |
| `MLPRegressor(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)` | early stopping intégré : 10 % de l'entraînement mis de côté, arrêt après 10 epochs sans gain de plus de `tol`, meilleurs poids repris ; même idée dans `SGDRegressor` et `HistGradientBoostingRegressor` (`early_stopping="auto"` : actif au-delà de 10 000 exemples) |
| `Perceptron(max_iter=100, shuffle=False, tol=None).fit(X, y)` | le perceptron de Rosenblatt (ch. 10), règle classique, exemples dans l'ordre ; les valeurs par défaut (`max_iter=1000`, `tol=1e-3`, `shuffle=True`) mélangent et s'arrêtent tôt ; `coef_` de forme `(1, n_features)` pour deux classes, `intercept_` de forme `(1,)` |
| `Perceptron(...).n_iter_` | le nombre d'epochs faites : avec `tol=None`, toujours `max_iter`, même si le perceptron a convergé avant (ch. 10) |
| `Perceptron(...).partial_fit(X, y, classes=[0, 1])` | une epoch de plus à chaque appel (`classes` obligatoire au premier) : pour suivre l'entraînement epoch par epoch (10.13, 10.19) |
| `Perceptron(...).fit(X, y, coef_init=c0, intercept_init=b0)` | partir de poids donnés au lieu de zéro (10.17) |
| `Perceptron()` sur trois classes ou plus | un-contre-tous automatique : `coef_` de forme `(n_classes, n_features)`, les mêmes scores qu'un `OneVsRestClassifier` de perceptrons (10.24) ; `Perceptron()` équivaut à `SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)` |

## Pipeline et ColumnTransformer

| Code | Effet |
|---|---|
| `make_pipeline(StandardScaler(), NearestCentroid())` | enchaîne prétraitement et modèle ; dans `cross_val_score`, chaque étape est réajustée sur la seule partie d'entraînement de chaque tour : pas de fuite (ch. 8, ch. 15) |
| `make_pipeline(PolynomialFeatures(2), LinearRegression())` | régression polynomiale (8.16) |

## Évaluation et validation croisée

| Code | Effet |
|---|---|
| `cross_val_score(model, X, y, cv=5)` | un score de validation par tour (le `score` du modèle, ou `scoring="f1_macro"`…) ; avec un entier, `StratifiedKFold` pour un classifieur et `KFold` sinon, **sans mélange** (ch. 8) |
| `cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0))`, `cv=list_of_pairs` | imposer les folds : un découpeur, ou une liste de paires `(train_idx, val_idx)` ; `groups=` pour `GroupKFold` |
| `cross_validate(model, X, y, cv=5, return_train_score=True)` | un dictionnaire : scores de test et d'entraînement, temps d'entraînement et de prédiction |
| `from sklearn.base import clone` ; `clone(model)` | un modèle neuf, mêmes hyperparamètres, rien d'appris (ce que fait `cross_val_score` à chaque tour) |
| `model.get_params()`, `model.set_params(max_depth=3)` | lire et changer les hyperparamètres (ceux de `__init__`) |
| `validation_curve(model, X, y, param_name="ridge__alpha", param_range=alphas, cv=5)` | courbe de validation (ch. 9) : scores d'entraînement et de validation pour chaque valeur d'un hyperparamètre, deux tableaux `(n_valeurs, n_folds)` ; `ridge__alpha` désigne l'`alpha` de l'étape `ridge` d'un pipeline |
| `sizes, train_scores, val_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1, 8), cv=5)` | courbe d'apprentissage (ch. 9) : les mêmes scores pour des jeux d'entraînement de plus en plus grands ; `sizes` donne les tailles en nombre d'exemples |
| `scoring="neg_mean_squared_error"`, `"neg_root_mean_squared_error"`, `"neg_mean_absolute_error"` | scikit-learn maximise toujours un score : les erreurs sont rendues **négatives** ; prendre `-scores` pour les lire |
| `permutation_test_score(model, X, y, cv=5, n_permutations=1000)` | le modèle fait-il mieux que le hasard ? permute les **labels** ; renvoie le score, les scores permutés et la p-valeur $(C + 1)/(n_{\text{perm}} + 1)$ |
| `scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue` | test exact de McNemar : deux classifieurs sur les mêmes exemples, $b$ et $c$ désaccords gagnés par chacun (ch. 8) |
| `from sklearn import metrics` | toutes les mesures d'évaluation (ch. 3) |
| `metrics.mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score` | mesures d'une régression (ch. 9) ; `root_mean_squared_error` existe depuis la version 1.4, et l'ancien `mean_squared_error(..., squared=False)` a disparu en 1.6 |
| `metrics.confusion_matrix(y_true, y_pred)` | matrice de confusion : vérité en lignes, prédiction en colonnes, labels triés (`[[TN, FP], [FN, TP]]` pour 0/1) ; `labels=[...]` impose l'ordre |
| `metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)` | dessine la matrice de confusion (axes « True label » et « Predicted label ») |
| `metrics.accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `fbeta_score(..., beta=2)` | les mesures de base ; `pos_label=` choisit la classe positive, `zero_division=` la valeur d'un 0/0 |
| `precision_score(..., average="macro")` | plusieurs classes : `"macro"`, `"weighted"`, `"micro"` ou `None` (une valeur par classe) |
| `metrics.classification_report(y_true, y_pred, digits=3)` | precision, recall, F1 et support de chaque classe, accuracy, moyennes macro et pondérée |
| `metrics.classification_report(..., output_dict=True)` | le même rapport en dictionnaire : `report["Gentoo"]["recall"]`, `report["macro avg"]["f1-score"]` |
| `metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true")` | chaque ligne divisée par son total : le recall de chaque classe sur la diagonale (`"pred"` : les precisions ; `"all"` : les probabilités jointes) |
| `tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[neg, pos]).ravel()` | les quatre cases d'une matrice binaire dans un ordre sûr, quels que soient les labels |
| `metrics.balanced_accuracy_score`, `matthews_corrcoef` | mesures robustes au déséquilibre des classes |
| `metrics.roc_curve(y_true, scores)` | FPR, TPR et seuils (le premier seuil vaut `np.inf`) ; `drop_intermediate=False` garde tous les points |
| `metrics.roc_auc_score(y_true, scores)` | aire sous la courbe ROC |
| `metrics.precision_recall_curve(y_true, scores)` | precision, recall (un point de plus que de seuils) et seuils croissants |
| `metrics.average_precision_score(y_true, scores)` | average precision (aire en escalier sous la courbe PR) |
| `metrics.RocCurveDisplay.from_predictions`, `PrecisionRecallDisplay.from_predictions` | tracer les courbes ROC et precision-recall |
| `from sklearn.calibration import calibration_curve, CalibrationDisplay` | diagramme de fiabilité : `calibration_curve(y_true, proba, n_bins=10)` ; par défaut `n_bins=5` (10 dans `mylearn.metrics.calibration_curve`) et `strategy="uniform"` |
| `metrics.brier_score_loss(y_true, proba)` | score de Brier |
| `metrics.log_loss(y_true, proba, labels=range(k))` | log loss, en nats : la cross-entropy moyenne ; probabilités coupées à $[\varepsilon ; 1 - \varepsilon]$ ; `labels` si une classe manque dans `y_true` (ch. 6) |
| `CalibratedClassifierCV(model, method="sigmoid")` | recalibrer un modèle (Platt ; `"isotonic"` avec beaucoup de données) |
| `metrics.silhouette_score(X, labels)`, `silhouette_samples(X, labels)` | coefficient de silhouette (moyen, ou par point) d'un clustering, entre −1 et 1 ; demande entre 2 et $n - 1$ clusters (ch. 7) |
| `metrics.davies_bouldin_score`, `calinski_harabasz_score` | autres critères internes d'un clustering (Davies-Bouldin : plus petit = mieux) |
| `metrics.adjusted_rand_score(y_true, labels)`, `homogeneity_score` | comparer un clustering à des labels de référence, quand on en a pour contrôler |

