# Mini-projet MP3 — Prédire un revenu à partir du recensement : pipeline tabulaire complet et audit d'équité

**Cahier des charges** du mini-projet du checkpoint III. Compte environ **8 heures** : huit étapes (7 h 10), plus la lecture du cahier des charges et, à la fin, celle de la solution. C'est un projet de portfolio : à la fin, un module testé, un notebook propre, cinq figures, un modèle sauvegardé, un fichier de résultats, un README, une model card et une data card qu'un recruteur peut lire, vérifier et relancer.

| | |
|---|---|
| **Problème** | prédire si le revenu annuel d'une personne dépasse 50 000 dollars, à partir de ses réponses au recensement, et dire pour qui le modèle se trompe |
| **Données** | *Adult* (*Census Income*), extrait du recensement américain de 1994 : 48 842 personnes, 14 colonnes mixtes (6 numériques, 8 catégorielles), des valeurs manquantes, 24 % de revenus élevés ; licence CC BY 4.0 |
| **Méthode imposée** | un unique `Pipeline` scikit-learn par modèle (`ColumnTransformer` pour le prétraitement), les classifieurs des ch. 13 et 14, le gradient boosting, la validation croisée du ch. 15 ; ta librairie `mylearn` pour l'audit et les intervalles |
| **Prérequis** | chapitres 12 à 15 ; de `mylearn` : `metrics.confusion_matrix`, `metrics.roc_auc` (ch. 3) et `stats.bootstrap_ci` (ch. 2), listées plus bas dans « Ta librairie » |
| **Évaluation** | sur 20, avec la grille ci-dessous |

## Le contexte

Un institut de statistique publie chaque année la part des personnes dont le revenu dépasse un seuil, par région, par métier, par sexe et par âge. Une partie des personnes interrogées ne répond pas à la question du revenu, mais répond aux autres. L'institut veut **compléter leur tranche de revenu** (au-dessus ou au-dessous de 50 000 dollars) à partir de leurs autres réponses : c'est une imputation, une pratique courante des enquêtes. La prédiction ne décide rien pour une personne ; elle entre dans des statistiques publiées. D'où deux exigences : un score honnête, et des erreurs qui ne faussent pas les statistiques d'un groupe. Une erreur qui pèse plus sur les femmes que sur les hommes fausse justement la statistique « part des femmes au-dessus du seuil ».

*Adult* sert depuis 1996 de dataset d'exemple, et l'usage ci-dessus est un cadre d'exercice : un modèle entraîné sur des données de 1994 ne dit rien des revenus d'aujourd'hui (voir la data card du dépôt, [`data/cards/adult.md`](../../data/cards/adult.md)).

## Le pipeline

Tout ce qui apprend des données vit dans un `Pipeline` : la validation croisée le clone et le réentraîne dans chaque fold, et c'est le même objet, prétraitement compris, que l'on sauvegarde à la fin.

1. Un `ColumnTransformer` qui écarte deux colonnes (`fnlwgt`, un poids d'échantillonnage du recensement, et `education`, une copie de `education-num`) et prépare les autres de deux façons :
   - **« onehot »**, pour les modèles des ch. 13 et 14 : les colonnes numériques complétées par leur médiane puis standardisées ; les catégorielles avec une catégorie `missing` pour les valeurs manquantes, puis encodées en one-hot (les catégories rares regroupées en une colonne, où vont aussi les catégories inconnues à la prédiction) ;
   - **« ordinal »**, pour le gradient boosting et ses catégories natives : les catégorielles codées 0, 1, 2… (les manquantes restent NaN), les numériques telles quelles.
2. Le modèle : `DummyClassifier`, la régression logistique, au moins quatre classifieurs des ch. 13 et 14 (d'au moins trois familles : les voisins, Naïve Bayes, les arbres, les ensembles, les SVM), puis `HistGradientBoostingClassifier`.

## Le protocole

1. Le **jeu de test** (20 % des personnes, **stratifié** sur la cible, graine 2026) est tiré **avant toute exploration**, enregistré dans `test_indices.npy` et commité aussitôt, seul (une cellule du notebook fait ce commit, sur Colab comme ailleurs). Le **coffre** (`data.TestVault`) ne rend que les personnes d'entraînement, et chaque ouverture est inscrite dans `vault.json`.
2. Toutes les **comparaisons** se font en validation croisée stratifiée à 5 folds, **les mêmes folds pour tous les modèles**, chaque score avec son écart-type entre folds, et les écarts lus **fold par fold**.
3. Le **score de décision** est annoncé avant les comparaisons : la PR-AUC (`average_precision`), adaptée au déséquilibre des classes, avec la ROC-AUC en second avis (tu peux en choisir un autre si tu le justifies, avant de comparer).
4. Les **hyperparamètres** du gradient boosting se règlent par une recherche au hasard sur ces folds, avec une règle annoncée à l'avance pour décider si le meilleur candidat remplace les valeurs par défaut.
5. Le **seuil de décision** se choisit sur les **prédictions hors fold**, jamais sur le test.
6. Le coffre s'ouvre **une fois**, en MP3.6, pour le modèle et le seuil retenus. L'audit par sous-groupe et l'importance par permutation de MP3.7 se font sur ce test, sans plus rien changer au modèle ; l'expérience qui compare des variantes du modèle (sans `sex`, sans `relationship`) se fait sur les folds.

## Les contraintes

- **scikit-learn pour les modèles**, dans des `Pipeline` ; NumPy et pandas pour le reste ; ta librairie `mylearn` pour la matrice de confusion et la ROC-AUC de l'audit et pour les intervalles bootstrap.
- **Pas de fuite** : le protocole ci-dessus, sans exception ; une entorse se signale dans le README.
- **Reproductible** : graines fixées et inscrites dans `results.json` ; le notebook s'exécute d'un bout à l'autre en moins de 10 minutes sur un CPU (un peu plus de 2 minutes pour la solution de référence en mode rapide).
- **Le code durable va dans un module**, `income.py`, testé avec pytest ; le notebook l'appelle, le vérifie, prend les décisions et trace les figures.

## Les livrables

1. `income.py` : `make_test_indices`, `build_preprocessor`, `build_pipeline`, `cross_validate_all`, `f1_threshold`, `age_groups`, `subgroup_report`, `library_versions`, `save_model` et `load_model`.
2. `test_income.py` : au moins **six** tests pytest, dont un sur l'absence de fuite (le prétraitement appris sur les seules lignes reçues par `fit`, ou réappris dans chaque fold) et un sur la reproductibilité.
3. `mp3_revenus.ipynb` : propre, exécuté de bout en bout, avec tes commentaires ; le mode rapide (`FAST_MODE`) est documenté dans sa première cellule.
4. `test_indices.npy`, commité seul et avant toute modélisation, et `vault.json` (une seule ouverture du coffre).
5. `model.joblib` : le pipeline final entraîné, avec son seuil, ses scores et les versions des bibliothèques (`save_model`) ; `results.json` : les choix, les scores par fold, les scores du test et leurs intervalles, l'audit, les graines et les versions.
6. Cinq figures : `figures/exploration.png`, `figures/comparaison_modeles.png`, `figures/courbes_pr_roc.png`, `figures/audit_sous_groupes.png`, `figures/importance_permutation.png`.
7. Trois documents, en français ou en anglais : le **README de portfolio** (`README.md` : le problème, les données, la démarche, les résultats, les limites), la **model card** (`MODEL_CARD.md` : l'usage prévu et les usages exclus, les données, les métriques globales et par sous-groupe, les risques et les limites) et la **data card** (`DATA_CARD.md` : la provenance, la licence, la collecte, la composition, les biais), un modèle en avance de celle du ch. 29.

## Les étapes

| Étape | Ce que tu fais | Ce que tu produis | ⏱️ |
|---|---|---|---|
| MP3.1 | Cadrer l'usage et les erreurs qui coûtent ; tirer un jeu de test stratifié, le commiter ; commencer la data card | `make_test_indices`, `test_indices.npy` commité, « Le problème » du README, la provenance de la data card | 45 min |
| MP3.2 | Explorer les seules personnes d'entraînement : types, valeurs manquantes, doublons, déséquilibre, taux de revenus élevés par groupe | `missing_table`, la figure d'exploration, la composition de la data card | 60 min |
| MP3.3 | Le `ColumnTransformer` et le `Pipeline` ; les baselines (`DummyClassifier`, régression logistique) sur 5 scores ; la fuite mesurée | `build_preprocessor`, `build_pipeline`, `cross_validate_all`, le premier tableau des scores | 45 min |
| MP3.4 | Au moins quatre classifieurs des ch. 13 et 14, de trois familles au moins, sur les mêmes folds, comparés fold par fold, avec leur coût | `CANDIDATES`, le tableau complet | 60 min |
| MP3.5 | Le gradient boosting : catégories natives ou one-hot, recherche au hasard, choix final par une règle annoncée | `SEARCH_SPACE`, `FINAL`, la figure de comparaison | 60 min |
| MP3.6 | Les prédictions hors fold, les courbes PR et ROC, le seuil ; l'ouverture unique du coffre, avec des intervalles bootstrap | `f1_threshold`, `THRESHOLD`, les scores du test, la figure des courbes | 40 min |
| MP3.7 | L'audit par sexe, âge et catégorie raciale ; le retrait de `sex` et de `relationship` (sur les folds) ; l'importance par permutation | `age_groups`, `subgroup_report`, deux figures, `results.json` | 60 min |
| MP3.8 | Six tests au moins ; le modèle sauvegardé et rechargé ; README, model card, data card ; le notebook relancé de zéro ; commit | `library_versions`, `save_model`, `load_model`, le projet complet | 60 min |

Chaque étape du notebook se termine par des **garde-fous** (✅ ou ❌) : des propriétés qu'un travail juste doit vérifier (un test stratifié et reproductible, des médianes apprises sur les seules lignes d'entraînement, un score de fold égal à celui d'un pipeline entraîné à la main, un audit qui redonne les taux de scikit-learn, un modèle rechargé qui prédit la même chose…). Ce ne sont pas des notes : deux projets justes peuvent choisir d'autres modèles et obtenir d'autres chiffres.

## La grille d'évaluation (sur 20)

| Critère | Points | Tous les points | La moitié des points |
|---|---|---|---|
| **Pipeline unique sans fuite**, vérifié par un test | 4 | tout le prétraitement dans un `Pipeline` réentraîné dans chaque fold, test stratifié commité avant toute modélisation, coffre ouvert une fois, un test pytest qui vérifie l'absence de fuite | un prétraitement appris hors du pipeline, ou pas de test de fuite |
| **Baseline et comparaison honnête** des modèles | 3 | `DummyClassifier` et régression logistique dans le tableau, au moins quatre modèles des ch. 13 et 14, de trois familles au moins, sur les mêmes folds, scores avec écart-type, écarts lus fold par fold, coûts comparés | pas de baseline, ou des modèles mesurés sur des folds différents |
| **Gradient boosting réglé et justifié** | 3 | catégories natives et one-hot comparées, recherche sur les mêmes folds, règle de choix annoncée et appliquée, optimisme du meilleur candidat discuté | le meilleur candidat retenu sans règle ni discussion |
| **Métriques adaptées au déséquilibre** et seuil argumenté | 3 | PR-AUC et ROC-AUC lues et comparées, seuil choisi sur les prédictions hors fold et justifié par l'usage (taux de sélection, precision et recall), scores du test avec leurs intervalles | l'accuracy seule, ou un seuil choisi sur le test |
| **Audit par sous-groupe** et discussion éthique | 3 | taux par sexe et par âge (TPR, FPR, taux de sélection face à la prévalence), effectifs lus, retrait de `sex` mesuré, importance par permutation interprétée, limites d'*Adult* discutées | des taux sans interprétation, ou un attribut sensible retiré sans mesurer l'effet |
| **Reproductibilité** : graines, versions, tests | 2 | graines et versions dans `results.json` et `model.joblib`, six tests verts, notebook qui s'exécute de zéro, historique git propre | l'un de ces éléments manque |
| **README, model card et data card** clairs | 2 | les trois documents complets, chiffrés, avec les figures, les usages exclus et les limites | des documents sans chiffres ou sans limites |

## Les extensions (pour aller plus loin)

- **CatBoost et ses catégories natives** : `pip install catboost` (hors de l'environnement figé du workbook) ; compare-le à `HistGradientBoostingClassifier` sur les mêmes folds, et compte le temps.
- **La calibration des probabilités** : `CalibratedClassifierCV` et la courbe de calibration (`mylearn.metrics.calibration_curve`, ch. 3) ; une imputation qui tire la tranche au hasard selon la probabilité prédite demande des probabilités calibrées.
- **Les explications SHAP** des prédictions (chapitre B6) : pourquoi le modèle prédit-il un revenu élevé pour cette personne ?
- **Des données récentes** : le paquet `folktables` (Ding et coll., 2021) donne la tâche `ACSIncome`, tirée de l'enquête ACS du recensement américain, pour remplacer *Adult* : refais l'audit sur des données d'aujourd'hui, État par État.
- **Une API de prédiction minimale** (chapitre B7) qui charge `model.joblib` et vérifie les versions.

## Ta librairie : les fonctions que le projet appelle

Le projet appelle trois fonctions de ta librairie `mylearn`. Les quatre parcours les écrivent toutes, dans les exercices ci-dessous. Avant chaque étape, vérifie que celles dont elle se sert passent leurs tests : le repli sur la référence ne remplace qu'un module **absent** de `mon_travail/mylearn/`. Dans un module que tu as copié, une fonction que tu n'as pas encore écrite lève `NotImplementedError`, et l'étape s'arrête.

| Étape | Fonctions de `mylearn` | Exercices qui les écrivent |
|---|---|---|
| MP3.6 | `stats.bootstrap_ci` | 2.22 |
| MP3.7 | `metrics.confusion_matrix`, `metrics.roc_auc` (qui s'appuie sur `roc_curve` et `auc`) | 3.15, 3.24 |

Pour vérifier une fonction, lance depuis la racine du dépôt les tests de son chapitre, filtrés sur son nom :

```bash
python -m pytest tests/test_ch03_metrics.py -q -k "confusion_matrix or roc_auc"
```

## Démarrer

```bash
python tools/start_chapter.py CP3
```

copie le kit de départ dans `mon_travail/projets/partie_3_revenus/` : `mp3_revenus.ipynb` (les huit étapes), `income.py` (à écrire), `test_income.py` (deux tests d'exemple, à compléter), `data.py` (fourni : les données et le coffre), `conftest.py` (fourni : pytest trouve ta librairie), et les modèles de `README.md`, `MODEL_CARD.md` et `DATA_CARD.md`. Travaille **dans ce dossier-là** ; les tests se lancent depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_3_revenus -q
```

Tant que `income.py` n'est pas écrit, les tests affichent « ⏳ pas encore écrit : … » avec le nom de la fonction qui manque : c'est normal.

Les données se chargent avec `data.load_census()` : depuis OpenML la première fois (puis depuis le cache), ou depuis la copie versionnée du dépôt (`data/adult.csv.gz`) si le téléchargement échoue. Les deux sources donnent la même table.

Tu peux faire corriger ton projet par Claude avec le prompt P9 : il lit ton dossier, sans jamais y écrire, et le note avec cette grille.

## La solution de référence

Le dossier [`solution/`](solution/) contient un projet complet : le module, 26 tests, le notebook exécuté, `test_indices.npy` (commité seul, avant toute modélisation), `vault.json`, `results.json`, les cinq figures, un README de portfolio, une model card et une data card (`model.joblib` n'y est pas versionné : le notebook le réécrit). **Ne l'ouvre qu'après avoir fini le tien** : c'est une façon de faire parmi d'autres, et tes chiffres n'ont pas à être les mêmes.
