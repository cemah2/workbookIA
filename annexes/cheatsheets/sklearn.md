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
| | |

