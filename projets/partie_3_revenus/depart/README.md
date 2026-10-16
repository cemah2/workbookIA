# Prédire un revenu à partir du recensement de 1994 : un pipeline tabulaire complet et un audit d'équité

> TODO : une phrase qui résume le projet et son résultat principal (le modèle retenu, sa PR-AUC et sa ROC-AUC sur le test avec leurs intervalles, comparées à la baseline), et le principal constat de l'audit.

Mini-projet du checkpoint III du workbook *Deep Learning* (d'après A. Glassner) : un `Pipeline` scikit-learn sans fuite, des classifieurs comparés en validation croisée sur les mêmes folds, un gradient boosting réglé, un seuil de décision choisi hors fold, un jeu de test ouvert une seule fois, et un audit des erreurs par sous-groupe. Voir aussi la [model card](MODEL_CARD.md) et la [data card](DATA_CARD.md).

<!-- Ce fichier est le modèle de ton README de portfolio : remplace chaque paragraphe qui commence par TODO (et la ligne TODO du tableau) par ton texte, avec tes propres chiffres. Tu peux l'écrire en anglais si tu vises des postes à l'international, avec le même plan (## The problem, ## The data, ## The approach, ## Results, ## Limitations). -->

## Le problème

TODO : l'usage (qui utilise la prédiction, pour quoi faire) ; ce que coûtent un faux positif et un faux négatif dans cet usage ; pourquoi une erreur qui pèse plus sur un groupe fausse ses statistiques ; le score de décision et pourquoi.

## Les données

TODO : la source (*Adult*, recensement américain de 1994, UCI, licence CC BY 4.0), la taille, les colonnes et la cible, les valeurs manquantes et ce qu'elles veulent dire, le déséquilibre, les colonnes écartées ; le jeu de test : 20 %, stratifié, tiré avec une graine fixe et commité avant toute modélisation (donne la date du commit).

## La démarche

TODO : le pipeline (le `ColumnTransformer`, les deux encodages) ; les 5 folds stratifiés communs à tous les modèles ; la règle de choix des hyperparamètres ; le seuil choisi hors fold ; le coffre ouvert une seule fois (`vault.json`).

## Les résultats

TODO : le tableau des scores en validation croisée (moyenne ± écart-type) et du test (avec ses intervalles) ; le choix du modèle et du seuil ; ce que montrent les figures.

| Modèle | PR-AUC (5 folds) | ROC-AUC (5 folds) | Test |
|---|---|---|---|
| TODO | | | |

![La part des revenus élevés selon le sexe, l'âge, le niveau d'études et la situation matrimoniale](figures/exploration.png)

![Tous les modèles en validation croisée, sur les mêmes folds](figures/comparaison_modeles.png)

![Courbes precision-recall et ROC, prédictions hors fold](figures/courbes_pr_roc.png)

![L'audit du test par sexe et par âge](figures/audit_sous_groupes.png)

![L'importance par permutation des colonnes d'origine](figures/importance_permutation.png)

## L'audit

TODO : les écarts entre groupes (TPR, FPR, taux de sélection face à la prévalence), les effectifs, ce que montre le retrait de `sex` et de `relationship`, les colonnes qui comptent.

## Les limites

TODO : par exemple des données de 1994 et un seuil de 50 000 $ de 1994, un benchmark « mis à la retraite » (Ding et coll., 2021), des catégories datées, des indicateurs indirects des attributs sensibles, des petits groupes, et les usages à exclure.

## Pistes

TODO : ce que je ferais ensuite (calibration, seuil adapté à l'usage, données ACS récentes avec `folktables`, SHAP…).

## Reproduire

Depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_3_revenus -q   # les tests du module
```

puis ouvrir `mp3_revenus.ipynb` et tout exécuter (quelques minutes sur un CPU). Les graines, les choix, les scores par fold, les scores du test et les versions des bibliothèques sont dans `results.json` ; le pipeline entraîné est dans `model.joblib` (`income.load_model`).
