# Protocole d'évaluation · Prévoir la demande horaire de vélos

> **Exemple de calibrage du workbook (PF.4)**, sur un dataset qui n'est pas le tien. Écrit après l'audit de PF.3 et avant toute expérience, puis figé par le tag `protocole`.

## Le découpage

**Temporel**, parce que le modèle servira à prévoir le lendemain : l'audit a montré qu'un découpage aléatoire mettrait des heures voisines (et une même journée) des deux côtés, et que la demande croît d'une année sur l'autre (un modèle validé au hasard serait jugé sur des jours qu'il a presque vus).

- entraînement : du 1ᵉʳ janvier 2011 au 30 juin 2012 (13 003 heures) · validation : juillet à septembre 2012 (2 208 heures) · test : octobre à décembre 2012 (2 168 heures, 92 jours) ;
- les retards (*lags*) se calculent sur une grille horaire complète, avec des valeurs manquantes pour les 165 heures absentes (pas des zéros), et ne remontent pas à moins de 48 h : à 18 h la veille, l'heure de 23 h du lendemain moins 24 h n'est pas encore connue ;
- fichier : `data/splits.json` · SHA-256 du découpage : `<les 64 caractères rendus par save_splits>`.

Ce que l'audit a changé : `casual` et `registered` sont retirées (leur somme vaut `cnt` sur chaque ligne : une fuite de la cible) ; 22 heures d'humidité nulle sont traitées comme manquantes ; les 2 180 heures de vent nul sont gardées et signalées (vent calme ou mesure absente : la source ne le dit pas) ; la description des saisons du `Readme.txt` de l'archive (1 = printemps) contredit les données (janvier est en saison 1 ; la page de l'UCI dit aujourd'hui 1 = hiver) : la saison est recalculée depuis la date ; la page annonce 17 389 instances, le fichier `hour.csv` en a 17 379 : les chiffres du protocole sont ceux du fichier.

## Le test scellé

Ouvert une fois, à la fin de la semaine 4, dans `notebooks/03_evaluation.ipynb`, pour les trois modèles. Aucun choix ne se fait sur lui : ni features, ni hyperparamètres, ni epoch d'arrêt. Les journées de l'ouragan Sandy (29 et 30 octobre 2012, 12 heures enregistrées) restent dans le test et sont rapportées aussi à part. En cas de réouverture : la raison, la date et les deux résultats dans le rapport.

## Les métriques

Principale : **MAE** (`sklearn.metrics.mean_absolute_error`), en locations par heure. Secondaires : la MAE des heures de pointe des jours ouvrés, la RMSE, la part des heures sous-estimées de plus de 100 vélos. **Intervalles** : bootstrap percentile à 95 %, 2 000 tirages, **par blocs d'une journée** (on tire des jours, puis on garde leurs 24 heures) : les erreurs de deux heures voisines ne sont pas indépendantes, un bootstrap heure par heure donnerait un intervalle trop étroit.

## Les baselines

**Naïve** : la persistance hebdomadaire, la même heure sept jours plus tôt. **Classique** : `HistGradientBoostingRegressor` sur le calendrier, la météo et les retards de 48 h et de 168 h, réglages par défaut, dans un `Pipeline`. La validation croisée de PF.5 utilise `TimeSeriesSplit` (cinq folds) sur l'entraînement seulement.

## Le critère de succès

MAE du modèle retenu au moins 10 % sous celle du gradient boosting, intervalle apparié à 95 % de la différence entièrement sous 0 ; prédiction des 24 heures du lendemain en moins de 1 s sur CPU.

## Les sous-groupes et la robustesse

Sous-groupes (au moins 200 heures chacun, sinon signalés sans conclusion) : jours ouvrés et non ouvrés ; tranches horaires (nuit, pointes, journée) ; météo 1, 2 et 3 (la météo 4 n'a que 3 heures en tout : signalée). Robustesse : la température et la température ressentie bruitées de ± 2 °C (une erreur de prévision météo), et la MAE des journées de Sandy à part.

## Le budget

Mode rapide : les 4 000 dernières heures de l'entraînement, 5 epochs, 500 tirages bootstrap, moins de 10 minutes sur CPU. Mode complet : tout l'entraînement, 50 epochs au plus avec early stopping ; le dataset est petit, un CPU suffit (pas de GPU). Une vingtaine de runs pour la recherche d'hyperparamètres, trois graines pour le modèle retenu et ses deux ablations (sans les retards ; un GRU sur les 168 dernières heures à la place du MLP à embeddings).

## Modifications

Aucune.
