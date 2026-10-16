# Model card · Classifieur de revenu (> 50 000 $) sur *Adult*

<!-- Le modèle d'une model card (Mitchell et coll., 2019, « Model Cards for Model Reporting ») : remplace chaque paragraphe qui commence par TODO. Une model card dit à quoi sert un modèle, à quoi il ne doit pas servir, et comment il se comporte, globalement et groupe par groupe. -->

## Le modèle

TODO : le type de modèle, ses hyperparamètres, le prétraitement, le seuil de décision, la date, la version de scikit-learn ; le fichier (`model.joblib`) et comment le charger (`income.load_model`, et pourquoi ne charger qu'un fichier de confiance).

## Usage prévu

TODO : l'usage pour lequel il a été construit (et le cadre d'exercice) ; les utilisateurs ; les **usages exclus** (toute décision sur une personne : crédit, embauche, assurance, aides…) et pourquoi.

## Les données

TODO : les données d'entraînement et de test (renvoie à la data card), leur date, les colonnes utilisées et écartées.

## Les métriques

TODO : les scores du test avec leurs intervalles (ROC-AUC, PR-AUC, F1, accuracy au seuil retenu), la baseline, le taux de sélection face à la prévalence ; les scores de validation croisée.

## Les sous-groupes

TODO : le tableau de l'audit par sexe et par âge (effectif, prévalence, taux de sélection, TPR, FPR), ce qu'il montre, et l'effet du retrait de `sex`.

## Les risques et limites

TODO : les biais connus, ce que le modèle ne sait pas faire, les conditions dans lesquelles ses scores ne valent plus, et ce qu'il faudrait faire avant tout usage réel.
