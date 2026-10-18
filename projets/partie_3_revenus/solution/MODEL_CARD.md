# Model card · Classifieur de revenu (> 50 000 $) sur *Adult*

Le format suit Mitchell et coll. (2019), « Model Cards for Model Reporting » : à quoi sert le modèle, à quoi il ne doit pas servir, et comment il se comporte, globalement et groupe par groupe.

## Le modèle

- **Type** : `HistGradientBoostingClassifier(random_state=0)` de scikit-learn 1.6.1, avec ses valeurs par défaut (100 itérations au plus, 31 feuilles par arbre, taux 0,1), dans un `Pipeline` dont le `ColumnTransformer` code les 7 colonnes catégorielles en entiers (catégories natives du modèle ; les catégories vues moins de 20 fois regroupées, les manquantes en NaN) et laisse les 5 colonnes numériques telles quelles ; `fnlwgt` et `education` sont écartées.
- **Seuil de décision** : 0,3607, celui du meilleur F1 sur les prédictions hors fold de l'entraînement, fixé avant l'évaluation sur le test. Pour une imputation, utiliser plutôt le seuil 0,416 (voir « Les risques et limites »).
- **Date** : le 16 octobre 2026 ; Python 3.13, NumPy 2.1.3, pandas 2.2.3, joblib 1.6.0.
- **Le fichier** : `model.joblib`, écrit par `income.save_model` avec le seuil, les scores et les versions ; `income.load_model` le recharge et avertit si la version de scikit-learn diffère (un pipeline sérialisé n'est garanti qu'avec la même version). joblib exécute du code au chargement : ne charge qu'un fichier de confiance.

## Usage prévu

- **L'usage** : compléter, dans une enquête, la tranche de revenu (plus ou moins de 50 000 dollars) des personnes qui n'ont pas répondu à cette question, pour calculer des **statistiques agrégées** (des parts par groupe). C'est un cadre d'exercice : le modèle sert à apprendre la démarche, sur des données de 1994.
- **Les utilisateurs** : des statisticiens qui savent que la prédiction est une estimation et la traitent comme telle (des marges d'erreur, des contrôles par groupe).
- **Les usages exclus** : toute décision qui concerne une personne (crédit, embauche, assurance, logement, aides sociales, contrôle fiscal). Le modèle n'a pas été évalué pour cela, il s'appuie sur des attributs sensibles et sur leurs variables de substitution (*proxy*), et ses erreurs ne pèsent pas de la même façon sur tous les groupes. Tout usage sur des données d'aujourd'hui est exclu aussi : les revenus et le seuil de 1994 ne s'y transposent pas.

## Les données

*Adult*, recensement américain de 1994 (UCI, CC BY 4.0) : 48 842 personnes, dont 39 073 pour l'entraînement et la validation croisée et 9 769 pour le test, tirées avec une stratification sur la cible (graine 2026) et commitées avant toute modélisation. 12 colonnes utilisées sur 14. La [data card](DATA_CARD.md) détaille la provenance, la composition et les biais.

## Les métriques

| Score | Validation croisée (5 folds) | Test (intervalle bootstrap à 95 %) | Baseline (test) |
|---|---|---|---|
| PR-AUC | 0,829 ± 0,007 | 0,832 [0,821 ; 0,845] | 0,239 |
| ROC-AUC | 0,928 ± 0,003 | 0,931 [0,927 ; 0,937] | 0,500 |
| F1, seuil 0,361 | 0,730 (hors fold) | 0,728 [0,716 ; 0,742] | 0 |
| Accuracy, seuil 0,361 | | 0,860 [0,853 ; 0,866] | 0,761 |
| Taux de sélection, seuil 0,361 | 0,269 (hors fold) | 0,275 | |
| Prévalence | 0,239 | 0,239 | |

Le jeu de test n'a été utilisé qu'une fois, pour ce modèle et ce seuil (`vault.json`).

## Les sous-groupes

Sur le test, au seuil 0,361 :

| Groupe | Effectif | Prévalence | Taux de sélection | TPR | FPR | Precision | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Femmes | 3 287 | 0,109 | 0,103 | 0,695 | 0,030 | 0,736 | 0,957 |
| Hommes | 6 482 | 0,306 | 0,363 | 0,798 | 0,171 | 0,672 | 0,911 |
| Moins de 25 ans | 1 702 | 0,010 | 0,002 | 0,235 | 0,000 | 1,000 | 0,954 |
| 25-34 ans | 2 466 | 0,168 | 0,171 | 0,686 | 0,067 | 0,675 | 0,927 |
| 35-44 ans | 2 460 | 0,334 | 0,386 | 0,790 | 0,184 | 0,683 | 0,902 |
| 45-54 ans | 1 730 | 0,393 | 0,482 | 0,841 | 0,249 | 0,687 | 0,895 |
| 55-64 ans | 983 | 0,331 | 0,407 | 0,825 | 0,201 | 0,670 | 0,905 |
| 65 ans et plus | 428 | 0,189 | 0,189 | 0,642 | 0,084 | 0,642 | 0,900 |

- Le modèle **sur-prédit les revenus élevés des hommes** (36,3 % prédits pour 30,6 %) et ceux des 35-64 ans, et **sous-prédit légèrement ceux des femmes** (10,3 % pour 10,9 %) et des moins de 25 ans. Imputées par ce modèle, les statistiques exagéreraient l'écart entre les sexes (26,0 points prédits pour 19,7 réels).
- Le TPR est plus bas chez les femmes (0,695 contre 0,798) et le FPR bien plus bas (0,030 contre 0,171) : une partie de ces écarts vient de la différence de prévalence. Le classement lui-même est meilleur chez les femmes (ROC-AUC 0,957 contre 0,911).
- Les catégories raciales « Amer-Indian-Eskimo » (90 personnes) et « Other » (75) sont trop petites pour un audit fiable ; les taux par catégorie sont dans `results.json`.
- **Retirer `sex`, puis `sex` et `relationship`**, ne change ni la PR-AUC hors fold (0,829, 0,829, 0,828) ni l'écart de sélection entre hommes et femmes (0,244, 0,234, 0,247) : d'autres colonnes portent la même information.

## Les risques et limites

- **Le seuil déployé gonfle la part des revenus élevés** (27,5 % prédits pour 23,9 % sur le test) : pour une imputation, utiliser le seuil 0,416, choisi lui aussi sur les prédictions hors fold, qui en prédit autant qu'il y en a (F1 hors fold 0,726), ou mieux, des probabilités calibrées et un tirage selon elles.
- **Les écarts entre groupes** : même avec le seuil de la prévalence, les parts par groupe restent à contrôler, groupe par groupe, avant toute publication.
- **Des données de 1994**, un seuil de 50 000 dollars de 1994, des catégories du recensement de l'époque (deux sexes, cinq catégories raciales) : les scores ne valent que pour cette population.
- **Un benchmark usé** : *Adult* a servi dans des milliers d'articles ; Ding et coll. (2021) recommandent des données ACS récentes (`folktables`) pour tout travail sérieux sur l'équité.
- **Avant tout usage réel** : réentraîner sur des données actuelles, calibrer, auditer chaque groupe publié avec des intervalles, et documenter l'imputation dans la méthodologie de l'enquête.
