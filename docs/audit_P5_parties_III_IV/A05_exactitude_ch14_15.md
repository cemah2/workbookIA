# Rapport A5 — exactitude des ch. 14 et 15 (audit P5, parties III et IV)

Périmètre : `chapitres/ch14_ensembles`, `chapitres/ch15_sklearn`, leurs lignes dans `annexes/formulaire.md`, `glossaire.md`, `erreurs_frequentes.md`, `cheatsheets/sklearn.md`, `data/cards/california_housing.md` et `penguins.md`, renvois des autres chapitres vers 14 et 15. Lecture seule ; `git status --short` vide à la fin. Scripts : `.../A05_exact_ch14_15/scripts/` (`ch14_paper.py`, `ch14_ada_gb.py`, `ch14_var.py`, `sk_checks1.py`, `sk_checks2.py`, `nbdump.py`, `nbout.py`, `grepdoc.py`) ; pages web dans `.../A05_exact_ch14_15/web/`.

## Constats

### 1. MINEUR · `chapitres/ch14_ensembles/01_fiche.md:102` (encadré 🕰️ « voter avec des probabilités »)
**Extrait :** « **Le livre :** fait voter les membres à la pluralité, une voix par membre. »
**Problème :** c'est injuste envers le livre, et la fiche se contredit. Le livre écrit : « In this chapter, we'll use weighted plurality voting » (p. 544), et la fiche le dit elle-même ligne 92 : « Le vote pondéré. C'est la règle que suit le livre dans tout le chapitre (p. 544) ». Ce que le livre ne fait jamais, c'est moyenner des probabilités : chez lui, chaque membre donne une classe.
**Correction :** « **Le livre :** fait voter chaque membre pour une seule classe (un vote dur), à la pluralité ou avec des poids (p. 543-546). »

### 2. MINEUR · `chapitres/ch14_ensembles/01_fiche.md:154` (🕰️ « les forêts dans scikit-learn »)
**Extrait :** « **Le livre :** demande de choisir la taille des bootstraps, le nombre d'arbres et la fraction de features, sans donner de valeurs. »
**Problème :** le livre donne une valeur pour le nombre d'arbres : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session) (p. 546). La fiche la corrige d'ailleurs ligne 134.
**Correction :** « … et la fraction de features ; il ne donne qu'une règle pour le nombre d'arbres, autant que de classes (p. 546), corrigée au §14.4. »

### 3. MINEUR · `chapitres/ch14_ensembles/01_fiche.md:250` (🕰️ « le gradient boosting, référence des données tabulaires »)
**Extrait :** « **Le livre :** ne présente le boosting qu'à travers AdaBoost. »
**Problème :** le livre présente le boosting avec des droites qu'il pondère à la main (p. 551-559), et ne fait que nommer AdaBoost (p. 560). La fiche le dit elle-même ligne 3 (« l'algorithme AdaBoost que le livre ne fait qu'évoquer ») et ligne 184 (« que le livre cite sans le détailler »).
**Correction :** « **Le livre :** présente le boosting avec des droites pondérées à la main, ne fait que citer AdaBoost, et ne parle pas du gradient boosting. »

### 4. MINEUR · `annexes/glossaire.md:538` (contredit le ch. 15)
**Extrait :** « | transformateur (scikit-learn) | transformer | objet à `fit`, `transform` et `inverse_transform` : il apprend des paramètres, puis les réapplique (…) | 12 | »
**Problème :** c'est justement l'erreur du livre que corrige l'encadré ⚠️ du §15.6.1 (fiche 15:201), et que reprennent 15.Q7 e et une flashcard. Vérifié avec scikit-learn 1.6.1 : `all_estimators(type_filter="transformer")` compte 84 transformateurs, dont 41 seulement ont un `inverse_transform` ; `PolynomialFeatures`, `Normalizer`, `SplineTransformer` et `KMeans` n'en ont pas (`sk_checks1.py`). La fiche du ch. 12 (l. 194) écrivait déjà « quand il existe ».
**Correction :** « objet à `fit` et `transform` (et `inverse_transform` quand il existe, soit pour environ la moitié des transformateurs, ch. 15) : il apprend des paramètres, puis les réapplique… »

### 5. MINEUR · `chapitres/ch09_overfitting/01_fiche.md:328` (renvoi vers le ch. 14, promesse non tenue)
**Extrait :** « **Ch. 14** : le bagging réduit la variance ; tu le mesureras avec ta `bias_variance_decomposition`. »
**Problème :** aucun fichier de `chapitres/ch14_ensembles/` ne contient `bias_variance` (vérifié par grep, notebooks compris), ni CP3 ni MP3. Le ch. 14 mesure l'effet du nombre d'arbres et de la corrélation (encadré 🧮 du §14.4, 🔬 14.13, 🔬 14.16), jamais une décomposition biais-variance.
**Correction (ch. 9) :** « **Ch. 14** : le bagging réduit la variance (la variance d'une moyenne de modèles corrélés, §14.4 ; 🔬 14.16). » Autre possibilité : une variante de 14.17 qui appelle `bias_variance_decomposition` sur un arbre seul puis sur un bagging.

### 6. MINEUR · `chapitres/ch17_activations/01_fiche.md:234` (renvoi vers le ch. 15, non tenu)
**Extrait :** « La température sert à **calibrer** les probabilités d'un réseau trop sûr de lui (*temperature scaling*, ch. 15 et 20) »
**Problème :** ni le ch. 15 (fiche, exercices, solutions, notebooks) ni la fiche du ch. 20 ne parlent de température (grep « temperature|température » sans résultat). Le ch. 15 ne présente que Platt et la régression isotone (`CalibratedClassifierCV`).
**Correction :** « (*temperature scaling* : la calibration des probabilités vue au ch. 3 et au ch. 15, appliquée aux logits d'un réseau) », ou bien ajouter une phrase sur la température dans la section calibration de la fiche du ch. 15.

### 7. MINEUR · durées citées dans `chapitres/ch14_ensembles/05_solutions.md`, différentes des sorties conservées
- l. 225 (14.15) : « `n_jobs=-1` divise le temps par 1,5 à 2 (environ 3,5 s contre 2 s dans les exécutions du corrigé) ». La sortie conservée de `05_solutions.ipynb` (cellule 71) affiche : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session).
- l. 300 (14.26) : « 0,5 à 0,6 s (XGBoost) et 0,5 s (LightGBM) ». La cellule 124 affiche : « fit times (s): {'hgb': 2.28, 'xgb': 0.39, 'lgbm': 0.42} ».
**Problème :** l'apprenant compare le corrigé au notebook de solutions, et ni ce rapport ni ces fourchettes n'y figurent. (Aucune durée n'a été mesurée pendant l'audit : ce constat compare deux textes du dépôt.)
**Correction :** l. 225, « divise le temps par 1,3 à 2 selon la charge de la machine (4,4 s contre 3,3 s dans l'exécution conservée) » ; l. 300, « 0,4 à 0,6 s (XGBoost) et 0,4 à 0,5 s (LightGBM) ».

### 8. MINEUR · `chapitres/ch15_sklearn/05_solutions.md:252` (15.19, « Dans tes notes »)
**Extrait :** « `uniform(1e-3, 1e3)` tire entre 0,001 et 1 000,001 (…) : 0,05 % de ses tirages sont sous 1, contre 50,2 % pour `loguniform`. »
**Problème :** 0,05 % (5 tirages sur 10 000, `random_state=1519`) est la valeur d'un échantillon, alors que la phrase la présente comme la part de la loi. La part attendue vaut $(1 - 0{,}001)/1000 \approx 0{,}1\,\%$ ; la solution de 15.Q10 f (l. 48) dit d'ailleurs « environ une fois sur mille ». La réponse h) = 0,0005 est juste (recalculée). Seul le commentaire est en cause.
**Correction :** « … : 0,05 % de tes 10 000 tirages sont sous 1 (5 tirages ; la part attendue est 0,999/1 000, environ 0,1 %), contre 50,2 % pour `loguniform` (50 % attendus). »

### 9. MINEUR · `chapitres/ch15_sklearn/05_solutions.md:208` (15.13, « Dans tes notes »)
**Extrait :** « un coude flou, entre 3 et 5, comme le jugement « à l'œil » du livre. »
**Problème :** la phrase laisse croire que le livre situait le coude entre 3 et 5. Le livre juge à l'œil que « somewhere between 5 and 8 seems like a good choice » (p. 586-587), et la fiche du ch. 15 l'écrit ainsi (l. 172 : « 5 à 8 clusters »).
**Correction :** « un coude flou, entre 3 et 5, aussi peu tranché que le jugement « à l'œil » du livre (5 à 8 clusters). »

### 10. SUGGESTION · `chapitres/ch14_ensembles/01_fiche.md:132`
**Extrait :** « $n$ tirages parmi $n$ (`max_samples=1.0`, le défaut de scikit-learn et de ta librairie) ».
**Problème :** c'est le défaut de `BaggingClassifier`. Pour les forêts, le défaut est `max_samples=None`, comme le dit l'encadré 🕰️ de la l. 154 (vérifié : `RandomForestClassifier().get_params()["max_samples"]` vaut `None`). Les deux écritures donnent $n$ tirages.
**Correction :** « (`max_samples=1.0` dans `BaggingClassifier` et ta librairie, `None` dans les forêts : $n$ tirages dans les deux cas) ».

### 11. SUGGESTION · `chapitres/ch14_ensembles/figures/importances.png` (`tools/chapters/figures_ch14.py:231-258`)
**Problème :** dans le panneau (b), la colonne « bruit à 3 valeurs », du pur bruit (`rng.integers(0, 3)`), affiche une importance par permutation d'environ 0,012, avec une barre d'erreur qui ne touche pas 0, soit environ 4 manchots sur 320 de validation. Le texte voisin (l. 156) dit qu'« une feature inutile ne fait rien perdre ». Un débutant peut y lire une contradiction.
**Correction :** ajouter à la légende « (les petites valeurs des deux colonnes de bruit, 0,01 au plus, sont des fluctuations de la mesure) », ou choisir une graine ou un jeu de validation plus grand où le bruit tombe sous 0,005.

### 12. SUGGESTION · paraphrases attribuées mais proches (§3)
Ces passages sont courts, attribués au livre, et ce sont des définitions banales ; on peut les reformuler par prudence.
- Ch. 14, fiche l. 86 : « la **prévision d'ensemble** en météorologie, qui fait tourner le même modèle avec des conditions initiales légèrement différentes (p. 543) ». Livre, p. 543 : « Weather forecasts often come from running the same model many times but with slightly different starting conditions ». Proposition : « la prévision d'ensemble, qui relance un même modèle météo à partir d'états de départ un peu perturbés ».
- Ch. 15, fiche l. 88 : « scikit-learn sert d'outil d'exploration et de préparation autour d'un réseau, et beaucoup de projets de deep learning commencent par lui ». Livre, p. 569 : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session) Proposition : « pour lui, scikit-learn est l'atelier où l'on examine et prépare les données avant de passer au réseau ».

## Vérifications faites sans écart (résumé)

**Ch. 14, calculs.**
- Toutes les réponses courtes ont été recalculées en Python : quiz Q1 à Q11, rappels R1 à R3, ✏️ 14.1 (8 items), 14.2 (8), 14.5 (9), 14.6 (13), 14.7 (11), ∂ 14.3 (8). Elles sont identiques à `05_solutions.md`.
- Démarches, erreurs fréquentes et variantes recalculées : 14.1 (variante [2, 0, 0, 1]), 14.2 (3 régions), 14.3 (0,599 ; 8,03), 14.5 (0,800), 14.6 (variante avec `learning_rate=0.5` : 0,973, 0,274, 0,104, 0,207, 0,671), 14.7 (sommes des carrés, variante 6,167 et seuil 2,5).
- 14.7 vérifié aussi avec `sklearn.ensemble.GradientBoostingRegressor` : `train_score_` vaut [3,075 ; 1,3944], la MSE de f) et j).
- Mini-exemples de la fiche : 0,967 ; 2,3 contre 3,6 ; 0,583 ; 0,328, 0,366 et 3,36 ; 0,55 ; 1,735 et 5,67 ; 0,767 ; 2,59.
- Nombres des solutions du notebook comparés aux sorties de `05_solutions.ipynb` (14.11 à 14.28) : concordants, durées mises à part (constat 7). Aussi recalculés : 14.13 (101 votants, 163 pour q = 0,55, limite c·q + 1 − c, corrélation c) et 14.16 (29 % et 2 % de nœuds avec une ou deux features utiles).

**Ch. 15, calculs.**
- Toutes les réponses courtes recalculées : Q1 à Q12, R1 à R3, ✏️ 15.1 à 15.4, 🧮 15.5. Elles sont identiques.
- Variantes recalculées : 91 entraînements ; 0,8176, 0,3775 et 0,8473 ; −2, 2 et 0,4 ; 35 et 15 ; 3,6 jours et 149 essais.
- Erreurs fréquentes chiffrées vérifiées : 72, 90, 60, 1,32 Go.
- 15.20 : 1 − 0,81¹⁶ ≈ 0,966 et π r² = 0,0567, d'où 0,607 ; variante 47,8 %.
- 15.12, variante : inerties 1 353,3 et 1 219,3, rejouées.
- 15.25 : seuils 1/11 et 1/6. 15.26 : poids 0,510 et 25. 15.23 : seuil de la règle d'une erreur type 0,9620.

**Comportements de scikit-learn 1.6.1 rejoués** (`sk_checks1.py`, `sk_checks2.py`) : tous conformes à la fiche, aux solutions, à `erreurs_frequentes.md` et à la cheatsheet.
- AdaBoost : l'appel du livre `AdaBoostClassifier(RidgeClassifier(), algorithm='SAMME')` tourne avec un seul `FutureWarning` (« deprecated in 1.6 and has no effect … removed in 1.8 ») ; `algorithm='SAMME.R'` est refusé ; `base_estimator` lève `TypeError` ; un AdaBoost d'arbres sans limite garde un seul membre.
- Validation et recherche : `RidgeClassifierCV().score` lève `NotFittedError` (`alphas=(0.1, 1.0, 10.0)`, `cv=None`) ; `ParameterGrid` range les clés par ordre alphabétique, la dernière variant le plus vite ; `ParameterSampler` tire avec remise dès qu'il y a un `randint`, sans remise sur des listes, avec le `UserWarning` « total space … smaller than n_iter ».
- Ordre des mixins : `is_classifier` vaut `False` pour `(BaseEstimator, ClassifierMixin)`, sans avertissement, et `check_cv` rend alors un `KFold`.
- Messages et retraits : `predict(3.0)` donne « Expected 2D array, got scalar array » ; `mean_squared_error(squared=False)` lève `TypeError` ; `set_output(transform="pandas")` sur un one-hot creux donne « Pandas output does not support sparse data » ; `load_boston` lève une `ImportError` explicative.
- Sorties et transformateurs : `decision_function` d'un SVC binaire est de forme `(n,)` ; 41 transformateurs sur 84 ont un `inverse_transform`.
- Défauts : `TunedThresholdClassifierCV` vise `balanced_accuracy`, `calibration_curve` prend `n_bins=5`, `train_test_split` arrondit le test au-dessus (`ceil`). Défauts des forêts, d'ExtraTrees, d'AdaBoost, de GB, de HGB (255 bins, 31 feuilles, `early_stopping="auto"`), de Stacking (`cv=None`, donc 5 folds) et de Bagging tous conformes.
- `HistGradientBoostingRegressor.fit(X, y, sample_weight=None)` n'accepte pas de jeu de validation.
- XGBoost 3.4.1 : `eta` vaut 0,3, `max_depth` 6, croissance en profondeur (`depthwise`), et `predict` utilise d'office `best_iteration`. LightGBM 4.6.0 : `num_leaves` vaut 31.
- `rng.choice(n, n)` et `rng.integers(0, n, n)` donnent les mêmes tirages en NumPy 2.1.3.

**Encadrés « Le livre : » et « Le livre, corrigé ».** Tous ont été comparés au texte de `vol1.txt` (p. 541-562 et 563-654). Ils sont fidèles et justes, sauf les constats 1 à 3. Points contrôlés :
- Apollo : « Neither spacecraft had a backup computer », « every board was duplicated twice ».
- Bonab16 (p. 546 et 560) ; la théorie du boosting limitée au binaire (p. 560) ; les poids (1.0, 1.5, −2) et (−8, 2, 3, 4).
- Tigre à 49 % et léopard à 48 % (p. 581) ; `decision_function` « for every class » (p. 582 et 641) ; « Every scikit-learn transformer provides … inverse_transform() » (p. 594).
- PCA : « determine which features to remove », et `whiten` qui « can help PCA produce its best results » (p. 599).
- AdaBoost : le renvoi au ch. 13 (p. 601-602), le vote (p. 604), les « 50 decision trees » (p. 605).
- `RidgeClassifierCV.score` et ses 8 folds (p. 607) ; le défaut de 3 folds (p. 615) ; « the routine that makes a RidgeClassifier » (p. 620-621) ; `best_parameters_` et `normalize=False` (p. 622-624) ; « first one encountered » (p. 625) ; la légende de la figure 15.27.
- Les 54 jours (p. 612) ; le degré 0 (figure 15.35, p. 640) ; « probability » sur une surface écrêtée à [−1 ; 1] (p. 643) ; le `MinMaxScaler` après `PolynomialFeatures` (p. 646) ; le cercle parcouru dans le sens des aiguilles d'une montre (p. 651).

**Autres vérifications.**
- Flashcards : les 25 du ch. 14 et les 25 du ch. 15 sont relues, sans erreur de fait.
- Annexes : formulaire des ch. 14 et 15 et cheatsheet relus, conformes ; dans `erreurs_frequentes.md`, les lignes sur scikit-learn qui concernent les ch. 14 et 15 sont conformes (rejouées).
- Data cards : California (plafonds 5, 52 et 15 ; recensement de 1990 ; 20 640 lignes) et Penguins (146, 119 et 68 manchots sur 333) concordent avec 15.7, 15.11 et 15.17.
- Cohérence avec les ch. 8, 9, 12 et 13 : pas de contradiction sur les conventions (`cv` par défaut, ex aequo du vote, `inverse_transform` « quand il existe », `predict_proba` des arbres), hors constats 4 à 6.

## Affirmations 🕰️ revérifiées sur le web (le 2026-10-17)

| # | Affirmation (fiche) | Verdict | Source |
|---|---|---|---|
| 1 | Dernière version de scikit-learn : 1.9.1, septembre 2026 (15 §15.1) | ✅ 1.9.1 publiée le 2026-09-10 | [PyPI, scikit-learn](https://pypi.org/project/scikit-learn/#history) (JSON de PyPI) |
| 2 | En 0.19 : `cv` à 3 folds par défaut ; `return_train_score` en `'warn'`, passé à `False` en 0.21 ; `best_parameters_` écrit par erreur dans la description de `refit` (15 §15.9.3) | ✅ texte exact dans la doc 0.19 | [GridSearchCV 0.19](https://scikit-learn.org/0.19/modules/generated/sklearn.model_selection.GridSearchCV.html) |
| 3 | AdaBoost : `algorithm` déprécié en 1.6, retiré en 1.8 ; `n_estimators` est un maximum (arrêt si ajustement parfait) ; `base_estimator` renommé en 1.2 (14 §14.7, 15 §15.8) | ✅ | [AdaBoostClassifier 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) |
| 4 | RandomForest : `n_estimators` passé de 10 à 100 en 0.22 ; `max_features` passé de `"auto"` à `"sqrt"` en 1.1 ; avertissement sur la MDI et la forte cardinalité (14 §14.5) | ✅ | [RandomForestClassifier 1.3](https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| 5 | Guide utilisateur : forêts qui moyennent les probabilités, « contrairement à la publication originale » ; « nombre critique » d'arbres ; HGB plus rapide de plusieurs ordres de grandeur au-delà de quelques dizaines de milliers d'exemples (14 §14.3, §14.5, GB) | ✅ | [User guide 1.11, Ensembles (1.6)](https://scikit-learn.org/1.6/modules/ensemble.html) |
| 6 | HGB : early stopping automatique au-delà de 10 000 exemples ; 255 bins (14, GB) | ✅ | [HistGradientBoostingRegressor 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) |
| 7 | Exemple Titanic avec `random_num` (une valeur par ligne) et `random_cat` (3 valeurs) ; permutation proche de 0 sur le test (14 §14.5) | ✅ | [Permutation Importance vs MDI (1.6)](https://scikit-learn.org/1.6/auto_examples/inspection/plot_permutation_importance.html) |
| 8 | `KMeans` : `n_init="auto"` par défaut depuis 1.4 (15 §15.3) | ✅ | [KMeans 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.KMeans.html) |
| 9 | `load_boston` : déprécié en 1.0, problème éthique, variable B, alternatives California et Ames `fetch_openml(name="house_prices")` (15 §15.10) | ✅ | [load_boston 1.1](https://scikit-learn.org/1.1/modules/generated/sklearn.datasets.load_boston.html) |
| 10 | `SVC(probability=True)` : validation croisée à 5 folds en interne, `predict_proba` qui peut contredire `predict` (15 §15.4.4 et seuil) | ✅ | [SVC 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.SVC.html) |
| 11 | Persistance : pickle et joblib, cloudpickle, skops.io et ONNX comparés ; ONNX « most secure option » (15, sauvegarde) | ✅ | [Model persistence 1.6](https://scikit-learn.org/1.6/model_persistence.html) |
| 12 | `sns.set()` : alias de `set_theme()`, « may be removed in the future » (15 §15.3) | ✅ | [seaborn.set](https://seaborn.pydata.org/generated/seaborn.set.html) |
| 13 | XGBoost : `learning_rate` (eta) à 0,3 et `max_depth` à 6 par défaut (14, GB) | ✅ | [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) |
| 14 | LightGBM : `num_leaves` à 31, croissance feuille par feuille (14, GB) | ✅ | [LightGBM Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [Features](https://lightgbm.readthedocs.io/en/stable/Features.html) |
| 15 | TabPFN : datasets d'au plus 10 000 exemples, bat en 2,8 s des baselines réglées pendant 4 h (14, GB) | ✅ | [Hollmann et coll., Nature 637, 2025](https://www.nature.com/articles/s41586-024-08328-6) |
| 16 | Grinsztajn et coll. : 45 datasets, environ 10 000 exemples ; titre arXiv sans « typical » (14, GB, 📄 14.10) | ✅ | [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) |
| 17 | CatBoost : Prokhorenkova et coll., features catégorielles (14, GB) | ✅ | [arXiv:1706.09516](https://arxiv.org/abs/1706.09516) |

Non vérifiable ici : l'article de M. Carlisle sur Medium (réponse 403 du proxy), cité en ⚖️ 15.7 (« 36 lignes ambiguës »). Il avait été vérifié à la session 31.

## Bilan

**Ce qui est solide.** L'exactitude des deux chapitres est très bonne. Aucune réponse vérifiée n'est fausse : environ 210 réponses courtes recalculées, et les nombres du notebook concordent avec les sorties conservées. Une cinquantaine de comportements de scikit-learn, XGBoost et LightGBM ont été rejoués : tous sont conformes. Toutes les affirmations 🕰️ échantillonnées tiennent. Les encadrés « Le livre, corrigé » sont justes et bien sourcés. Aucun constat MAJEUR.

**Trois risques principaux.**
1. Les encadrés 🕰️ « Le livre : » résument parfois le livre trop vite, et contredisent la fiche elle-même (constats 1 à 3) : c'est l'image du livre qu'en garde l'apprenant.
2. Des renvois entre chapitres annoncent ce qui n'a pas été écrit (ch. 9 vers 14, ch. 17 vers 15 et 20), et une ligne du glossaire reprend l'erreur du livre que corrige le ch. 15 (constats 4 à 6).
3. Les commentaires écrits à la main dans `05_solutions.md` (durées, part d'un échantillon, résumé du livre) peuvent diverger des sorties exécutées (constats 7 à 9). Les réponses, elles, sont générées et sûres.
