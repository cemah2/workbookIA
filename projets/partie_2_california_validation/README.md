# Mini-projet MP2 — Prix des logements californiens : un protocole d'évaluation honnête

**Cahier des charges** du mini-projet du checkpoint II. Compte environ **10 heures**, en sept étapes. C'est un projet de portfolio : à la fin, un module testé, un notebook propre, quatre figures, un fichier de résultats et un README qu'un recruteur peut lire, vérifier et relancer.

| | |
|---|---|
| **Problème** | prédire la valeur médiane des logements d'un district californien, et dire honnêtement avec quelle erreur |
| **Données** | *California Housing* (recensement américain de 1990) : 20 640 districts, 8 mesures, une cible plafonnée à 500 000 dollars |
| **Méthode imposée** | des modèles linéaires de ta librairie `mylearn` (moindres carrés, Ridge, Lasso), des features polynomiales et des zones géographiques par k-means, sans bibliothèque de machine learning |
| **Prérequis** | chapitres 7 à 9 : `mylearn.cluster` (k-means, silhouette), `mylearn.model_selection` (folds, `clone`), `mylearn.linear` (moindres carrés, Ridge, Lasso), et `mylearn.stats` (ch. 2) pour le bootstrap ; si tu n'as pas écrit l'un de ces modules, celui de la référence est utilisé à sa place |
| **Évaluation** | sur 20, avec la grille ci-dessous |

## Le contexte

Un bureau d'études urbaines veut une première estimation de la valeur médiane des logements de chaque district, à partir des données du recensement, pour repérer les quartiers dont les prix s'écartent de ce qu'annoncent leurs caractéristiques. Ce qu'on te demande avant tout, c'est **un chiffre d'erreur auquel on peut se fier**. Un score trop beau parce que le jeu de test a servi à régler le modèle, ou parce que le prétraitement a vu les districts de validation, ne vaut rien : c'est la leçon centrale des chapitres 8 et 9, et ce projet la met en pratique de bout en bout.

## Le modèle

Une chaîne de six étapes, toutes apprises dans `fit`, sur les seules lignes d'entraînement :

1. des **bornes** pour chaque mesure, les percentiles 1 et 99 de l'entraînement : quelques districts atypiques (jusqu'à 600 personnes par ménage en moyenne) ne tirent plus la régression ;
2. les **z-scores** des mesures bornées ;
3. les **features polynomiales** de degré `degree` ;
4. au besoin, des **zones géographiques** : un k-means sur la latitude et la longitude, une colonne 0/1 par zone ;
5. un second z-score, de toutes les colonnes, pour que la pénalité les traite de la même façon ;
6. un modèle linéaire : **moindres carrés, Ridge ou Lasso**.

## Le protocole, le cœur du projet

1. Le **jeu de test** (20 % des districts, graine 2026) est tiré **avant toute exploration**, enregistré dans `test_indices.npy` et **commité aussitôt, seul** : la date de ce commit montre que le découpage a été fixé avant tes résultats (sur Colab, sans terminal, une cellule du notebook fait ce commit).
2. Le **coffre** (`data.TestVault`) ne rend que les districts d'entraînement. Il ne s'ouvre qu'à l'étape MP2.6, une seule fois, et chaque ouverture est inscrite dans `vault.json`, qui documente ainsi l'usage du test.
3. Toutes les **décisions** (les bornes, le degré, α, Ridge ou Lasso, le nombre de zones) se prennent par **validation croisée à 5 folds**, les mêmes folds pour tous les modèles. Tout ce qui apprend des données apprend dans `fit`, que la validation croisée refait dans chaque fold.
4. Chaque choix est donné **avec son incertitude** (l'écart-type entre les folds), et suit une règle annoncée à l'avance : la **règle d'une erreur type** (le modèle le plus simple dont le score ne dépasse pas le meilleur de plus d'une erreur type, estimée par $\sigma/\sqrt{k}$ : une approximation optimiste, puisque les folds partagent leurs données).
5. Le **score final** est mesuré une fois sur le test, avec un intervalle bootstrap.

## Les contraintes

- **Pas de bibliothèque de machine learning pour les modèles** : NumPy et ta librairie `mylearn`. scikit-learn ne sert qu'à **vérifier** tes chiffres (étapes MP2.2, MP2.3 et MP2.6).
- **Pas de fuite** : le protocole ci-dessus, sans exception ; une entorse se signale dans le README.
- **Reproductible** : graines fixées et inscrites dans `results.json` ; le notebook s'exécute d'un bout à l'autre en moins de 10 minutes sur un CPU (environ 2 minutes pour la solution de référence).
- **Le code durable va dans un module**, `housing.py`, testé avec pytest ; le notebook l'appelle, le vérifie, prend les décisions et trace les figures.

## Les livrables

1. `housing.py` : `make_test_indices`, `rmse`, `HousingModel` (`fit`, `transform`, `zones`, `predict`), `cross_validate`, `validation_curve`, `learning_curve` et `out_of_fold_predictions`, qui réutilisent `mylearn.linear`, `mylearn.cluster` et `mylearn.model_selection`.
2. `test_housing.py` : au moins **six** tests pytest (des oracles NumPy ou scikit-learn, des propriétés, des entrées invalides).
3. `test_indices.npy`, commité **seul et avant toute modélisation**, et `vault.json` (une seule ouverture du coffre).
4. `mp2_california.ipynb` : propre, exécuté de bout en bout (« Run all »), avec tes commentaires ; le mode rapide (`FAST_MODE`) est documenté dans sa première cellule.
5. `results.json` : les hyperparamètres retenus, les scores par fold, le score du test et son intervalle, les graines, les versions des bibliothèques.
6. Quatre figures : `figures/courbes_validation.png` (α et degré), `figures/zones.png` (silhouette et validation croisée selon le nombre de zones), `figures/courbes_apprentissage.png`, `figures/carte_zones_residus.png` (les zones et les résidus sur la carte).
7. Le **README de portfolio** (`README.md` de ton dossier) : le problème, les données, le protocole, le tableau des scores (moyenne ± écart-type en validation croisée, score final du test), les figures, les limites et les questions d'équité, la façon de reproduire.

## Les étapes

| Étape | Ce que tu fais | Ce que tu produis | ⏱️ |
|---|---|---|---|
| MP2.1 | Cadrer le problème ; tirer et geler le jeu de test, le commiter ; explorer les seuls districts d'entraînement | `make_test_indices`, `test_indices.npy` commité, la section « Le problème » du README | 60 min |
| MP2.2 | Une référence naïve (prédire la moyenne), puis les moindres carrés en 5 folds, sans bornes et avec | `rmse`, `HousingModel` (degré 1), `cross_validate`, `mean_baseline`, le premier tableau des scores | 60 min |
| MP2.3 | Features polynomiales, Ridge et Lasso : courbes de validation, degré et α choisis par la règle d'une erreur type | `validation_curve`, `pick_simplest`, la figure des courbes de validation | 120 min |
| MP2.4 | Des zones par k-means : la silhouette, puis le nombre de zones et α choisis par validation croisée | `HousingModel` complet (les zones), la figure des zones | 90 min |
| MP2.5 | Diagnostiquer : courbes d'apprentissage, variance, résidus hors fold par région et pour les districts plafonnés | `learning_curve`, `out_of_fold_predictions`, `residuals_by_group`, deux figures | 90 min |
| MP2.6 | Ouvrir le coffre une seule fois ; comparer ta librairie à scikit-learn | le score du test et son intervalle, `results.json` | 45 min |
| MP2.7 | Emballer : six tests au moins, README complet, notebook relancé de zéro, commit | le projet complet | 75 min |

Chaque étape du notebook se termine par des **garde-fous** (✅ ou ❌) : des propriétés qu'un travail juste doit vérifier (des indices de test triés et reproductibles, des prédictions identiques à celles de scikit-learn sur la même chaîne, des prédictions hors fold qui redonnent les scores des folds…). Ce ne sont pas des notes : deux projets justes peuvent faire d'autres choix et obtenir d'autres chiffres.

Les scores de ce projet ne se comparent pas à ceux du défi 🏆 9.31 : il écartait les districts plafonnés et les ménages atypiques ; ici, tous les districts restent, comme dans la réalité.

## La grille d'évaluation (sur 20)

| Critère | Points | Tous les points | La moitié des points |
|---|---|---|---|
| **Protocole sans fuite** | 4 | test tiré et commité avant toute modélisation (son commit précède les résultats), coffre ouvert une seule fois, bornes, z-scores et zones appris dans `fit` et refaits dans chaque fold, les mêmes folds pour toutes les comparaisons | test gelé mais ouvert plusieurs fois sans le dire, ou un prétraitement appris sur tous les districts d'entraînement avant la validation croisée |
| **Modèles `mylearn` corrects et vérifiés**, référence naïve | 3 | moindres carrés, Ridge et Lasso (et toute la chaîne) donnent les prédictions de scikit-learn, la référence naïve est dans le tableau | un modèle non vérifié, ou pas de référence |
| **Hyperparamètres choisis par validation croisée**, avec leur incertitude | 3 | degré et α choisis sur les folds, chaque score avec son écart-type, une règle de choix annoncée et appliquée, et la différence avec le meilleur score discutée | le meilleur score retenu sans incertitude, ou α choisi sur un seul découpage |
| **Zones k-means justifiées**, apport mesuré | 2 | silhouette calculée et discutée, nombre de zones choisi par validation croisée, gain mesuré face au modèle sans zone, limite de la dépendance spatiale signalée | nombre de zones choisi par la silhouette seule, ou gain non mesuré |
| **Diagnostic biais-variance** argumenté par des courbes | 3 | courbes d'apprentissage de deux modèles bien lues (biais ou variance), une mesure de la variance, résidus par région et des districts plafonnés commentés | des courbes sans interprétation |
| **README, figures, limites et équité** | 3 | tableau des scores avec incertitudes et score du test, quatre figures commentées, limites (données de 1990, cible plafonnée, districts voisins répartis dans les folds) et équité (qui le modèle dessert mal, l'emplacement comme indicateur indirect) discutées | des résultats sans limites, ou sans chiffres |
| **Reproductibilité** | 2 | graines, versions et choix dans `results.json`, notebook qui s'exécute de zéro, historique git propre (`test_indices.npy` commité en premier) | l'un de ces éléments manque |

## Les extensions (pour aller plus loin)

- **Une validation croisée spatiale** : des folds faits de blocs géographiques (les cellules d'une grille de 0,5°, ou les groupes d'un k-means grossier), pour mesurer ce que vaut le modèle dans un quartier qu'il n'a jamais vu ; compare le score et le nombre de zones choisi avec ceux des folds tirés au hasard.
- **HDBSCAN pour les zones** (`sklearn.cluster.HDBSCAN`, dans scikit-learn depuis la version 1.3) : des zones de densité, de tailles variées, et un groupe « bruit » pour les districts isolés ; compare à k-means.
- **La double descente** avec des features aléatoires (des ReLU de projections aléatoires des mesures, la solution de norme minimale `np.linalg.pinv`, comme en 🔬 9.30) : l'erreur de validation selon le nombre de features, autour de $p = n$.
- **Choisir α comme un bandit** (ch. 11) : chaque valeur de α est un bras, chaque tirage entraîne le modèle sur un fold et rapporte −RMSE ; `ucb_action` de ta librairie `bandit` choisit le bras suivant. Ou bien le *successive halving* : beaucoup de valeurs évaluées sur peu de districts, les meilleures gardées sur de plus en plus de districts. Compare le coût (le nombre d'entraînements, les districts vus) à celui de la grille.
- **Aperçu du ch. 14** : `HistGradientBoostingRegressor` de scikit-learn, sur les mêmes folds, l'état de l'art des données tabulaires ; combien gagne-t-il sur ton meilleur modèle linéaire ?
- **Une carte interactive des résidus** 🕰️ (vérifié le 3 octobre 2026) : plotly (5.24.1) et folium (0.20.0) sont préinstallés sur Colab (image du 2 octobre 2026) ; en local, `pip install plotly` ou `pip install folium`. *Source :* [googlecolab/backend-info, `pip-freeze.txt`](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.txt).

## Démarrer

```bash
python tools/start_chapter.py CP2
```

copie le kit de départ dans `mon_travail/projets/partie_2_california_validation/` : `mp2_california.ipynb` (les sept étapes), `housing.py` (à écrire), `test_housing.py` (deux tests d'exemple, à compléter), `data.py` (fourni : les données et le coffre), `conftest.py` (fourni : pytest trouve ta librairie) et `README.md` (le modèle de ton README de portfolio). Travaille **dans ce dossier-là** ; les tests se lancent depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_2_california_validation -q
```

Tu peux faire corriger ton projet par Claude avec le prompt P9 : il lit ton dossier, sans jamais y écrire, et le note avec cette grille.

## La solution de référence

Le dossier [`solution/`](solution/) contient un projet complet : le module, 25 tests, le notebook exécuté, `test_indices.npy` (commité seul, avant toute modélisation), `vault.json`, `results.json`, les quatre figures et un README de portfolio rédigé. **Ne l'ouvre qu'après avoir fini le tien** : c'est une façon de faire parmi d'autres, et tes chiffres n'ont pas à être les mêmes.
