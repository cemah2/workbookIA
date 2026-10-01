# Cheatsheet scikit-learn

> Aide-mémoire rempli au fil des chapitres (8, 12-15). Une ligne = une commande utile + ce qu'elle fait.

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
| | |

## Prétraitement

| Code | Effet |
|---|---|
| | |

## Modèles courants

| Code | Effet |
|---|---|
| `DecisionTreeClassifier(max_depth=2, random_state=0)` | arbre de décision (ch. 1 en boîte noire, ch. 13) ; `export_text(model, feature_names=...)` affiche ses règles |
| `KMeans(n_clusters=3, n_init=10, random_state=0)` | clustering en $k$ groupes (ch. 1, ch. 7) ; mettre les features à la même échelle avant |
| `MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, random_state=0)` | réseau de neurones à une couche cachée de 128 neurones (ch. 1, ch. 16) ; poids dans `coefs_`, biais dans `intercepts_` |

## Pipeline et ColumnTransformer

| Code | Effet |
|---|---|
| | |

## Évaluation et validation croisée

| Code | Effet |
|---|---|
| `from sklearn import metrics` | toutes les mesures d'évaluation (ch. 3) |
| `metrics.confusion_matrix(y_true, y_pred)` | matrice de confusion : vérité en lignes, prédiction en colonnes, étiquettes triées (`[[TN, FP], [FN, TP]]` pour 0/1) ; `labels=[...]` impose l'ordre |
| `metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)` | dessine la matrice de confusion (axes « True label » et « Predicted label ») |
| `metrics.accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `fbeta_score(..., beta=2)` | les mesures de base ; `pos_label=` choisit la classe positive, `zero_division=` la valeur d'un 0/0 |
| `precision_score(..., average="macro")` | plusieurs classes : `"macro"`, `"weighted"`, `"micro"` ou `None` (une valeur par classe) |
| `metrics.classification_report(y_true, y_pred, digits=3)` | precision, recall, F1 et support de chaque classe, accuracy, moyennes macro et pondérée |
| `metrics.classification_report(..., output_dict=True)` | le même rapport en dictionnaire : `report["Gentoo"]["recall"]`, `report["macro avg"]["f1-score"]` |
| `metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true")` | chaque ligne divisée par son total : le recall de chaque classe sur la diagonale (`"pred"` : les precisions ; `"all"` : les probabilités jointes) |
| `tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[neg, pos]).ravel()` | les quatre cases d'une matrice binaire dans un ordre sûr, quelles que soient les étiquettes |
| `metrics.balanced_accuracy_score`, `matthews_corrcoef` | mesures robustes au déséquilibre des classes |
| `metrics.roc_curve(y_true, scores)` | FPR, TPR et seuils (le premier seuil vaut `np.inf`) ; `drop_intermediate=False` garde tous les points |
| `metrics.roc_auc_score(y_true, scores)` | aire sous la courbe ROC |
| `metrics.precision_recall_curve(y_true, scores)` | precision, recall (un point de plus que de seuils) et seuils croissants |
| `metrics.average_precision_score(y_true, scores)` | average precision (aire en escalier sous la courbe PR) |
| `metrics.RocCurveDisplay.from_predictions`, `PrecisionRecallDisplay.from_predictions` | tracer les courbes ROC et precision-recall |
| `from sklearn.calibration import calibration_curve, CalibrationDisplay` | diagramme de fiabilité (`n_bins=10`, `strategy="uniform"`) |
| `metrics.brier_score_loss(y_true, proba)` | score de Brier |
| `metrics.log_loss(y_true, proba, labels=range(k))` | log loss, en nats : la cross-entropy moyenne ; probabilités coupées à $[\varepsilon ; 1 - \varepsilon]$ ; `labels` si une classe manque dans `y_true` (ch. 6) |
| `CalibratedClassifierCV(model, method="sigmoid")` | recalibrer un modèle (Platt ; `"isotonic"` avec beaucoup de données) |

