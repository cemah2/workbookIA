# Protocole d'évaluation · TODO : le titre du projet

<!-- PF.4 : écrit AVANT toute expérience (après l'exploration de PF.3), puis figé : `git add docs/protocole.md data/splits.json && git commit -m "Protocole d'évaluation" && git tag protocole && git push --tags`. Une modification ultérieure n'est pas interdite, mais elle se fait dans un nouveau commit, s'ajoute à la section « Modifications » ci-dessous et se signale dans le rapport (une modification non signalée coûte 3 points). Remplace chaque paragraphe qui commence par TODO. Un exemple rempli : projets/projet_final/exemple/protocole.md du workbook. -->

## Le découpage

TODO : la règle (aléatoire stratifié, par groupes, temporel) et pourquoi (ce que l'audit de PF.3 a montré : doublons, groupes, dérive dans le temps) ; les proportions (entraînement, validation, test) ; la graine ; le fichier.

- Fichier du découpage : `data/splits.json` (écrit une fois par `projet split`, jamais refait)
- SHA-256 du découpage : `TODO-les-64-caractères-que-rend-save_splits`
- Tailles : entraînement TODO · validation TODO · test TODO

## Le test scellé

Le test ne sert qu'une fois, à la fin, pour le modèle retenu et les deux baselines, dans `notebooks/03_evaluation.ipynb` (`evaluate.open_sealed_test`, qui note l'ouverture dans `experiments/test_access.jsonl`). Aucun choix (features, architecture, hyperparamètres, seuil, epoch d'arrêt) ne se fait sur lui. TODO : ce que tu feras si tu dois le rouvrir (le dire, pourquoi, et rapporter les deux résultats).

## Les métriques

TODO : la métrique principale (une seule, celle de `config.metric`) et les secondaires ; leur définition (`sklearn.metrics`, moyenne macro ou pondérée, classe positive) ; l'intervalle (bootstrap percentile à 95 %, `n_bootstrap` tirages, par groupes s'il y en a).

## Les baselines

TODO : la baseline naïve (classe majoritaire, moyenne, persistance) et la baseline classique (le modèle et ses réglages par défaut) ; la validation croisée (le nombre de folds, la règle des folds : groupes, temps).

## Le critère de succès

TODO : repris de docs/cadrage.md, chiffré : la différence avec la baseline classique sur le test, son intervalle bootstrap apparié, la contrainte (latence, taille).

## Les sous-groupes et la robustesse

TODO : les sous-groupes que tu rapporteras (classes, sources, périodes, une variable sensible si tu as le droit de t'en servir) et la taille minimale d'un sous-groupe ; les perturbations de robustesse (bruit, décalage, autre période) décidées maintenant.

## Le budget

TODO : le temps et le matériel par run (FAST_MODE sur CPU en moins de 10 minutes ; le mode complet sur GPU Colab), le nombre de runs de la recherche d'hyperparamètres (une vingtaine au plus), trois graines pour le modèle retenu et ses ablations.

## Modifications

<!-- Une ligne par modification après le tag : date, commit, ce qui change, pourquoi, et si le test avait déjà été ouvert (il ne doit pas l'avoir été). -->

Aucune.
