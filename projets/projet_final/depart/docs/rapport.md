# Rapport · TODO : le titre du projet

<!-- Le rapport technique du projet : il se remplit au fil des étapes, section par section (la colonne « Étape » dit quand). Chaque chiffre vient d'un fichier du dépôt (un notebook exécuté, experiments/runs.jsonl, experiments/test_results.json) : cite-le. Le README en reprend l'essentiel pour un lecteur pressé ; ce rapport est pour celui qui veut vérifier. Remplace chaque paragraphe qui commence par TODO. -->

| Section | Étape |
|---|---|
| Le problème | PF.1 |
| Les données | PF.2, PF.3 |
| Le protocole | PF.4 |
| Les baselines | PF.5 |
| Le modèle | PF.6 |
| Les expériences | PF.7 |
| Les résultats | PF.8 |
| Explicabilité et équité | PF.9 |
| Industrialisation et démonstration | PF.10, PF.11 |
| Limites et suite | PF.12 |

## Le problème

TODO : trois lignes reprises de docs/cadrage.md, et le critère de succès chiffré.

## Les données

TODO (PF.3) : la source et la licence (renvoi à data/card.md) ; les chiffres de l'audit (taille, doublons exacts et quasi-doublons, valeurs manquantes, déséquilibre, fuites trouvées et corrigées) ; le décalage entre les découpages (proportions de la cible, distributions des entrées) ; une ou deux figures de `figures/`. Ce que l'audit a changé dans le découpage.

## Le protocole

TODO (PF.4) : le résumé de docs/protocole.md, le lien vers le commit tagué `protocole`, et la liste des modifications après le tag (ou « aucune »).

## Les baselines

TODO (PF.5) : la baseline naïve et la baseline classique, leurs scores de validation croisée (moyenne ± écart-type sur les folds), le run de chacune dans le journal.

## Le modèle

TODO (PF.6) : le modèle deep learning choisi et **pourquoi celui-là** pour ces données (un modèle pré-entraîné, un encodeur, un réseau séquentiel, un réseau tabulaire, un agent) ; son architecture, son nombre de paramètres, son entraînement (optimiseur, planning, epochs, early stopping), sa reproductibilité (graine, commande).

## Les expériences

TODO (PF.7) : la recherche d'hyperparamètres (l'espace, le nombre de runs, sur la validation seulement), les ablations (une variable à la fois, trois graines), le tableau généré par `scripts/summarize_runs.py` (copié ici, avec la commande qui le régénère), et ce que tu en conclus.

## Les résultats

TODO (PF.8) : le test scellé, ouvert une fois (la date et le commit, repris de experiments/test_access.jsonl). Le tableau : la métrique principale et les secondaires des trois modèles, chacune avec son intervalle bootstrap à 95 % ; la différence avec la baseline classique et son intervalle apparié ; le critère de succès atteint ou non. L'analyse d'erreurs (les erreurs les plus sûres d'elles, lues une par une, rangées par cause), les sous-groupes (avec leurs effectifs), la robustesse.

| Modèle | Métrique principale [IC 95 %] | Secondaire 1 | Secondaire 2 |
|---|---|---|---|
| Baseline naïve | TODO | | |
| Baseline classique | TODO | | |
| Modèle retenu | TODO | | |

## Explicabilité et équité

TODO (PF.9) : les attributions adaptées au modèle (SHAP ou permutation pour un modèle tabulaire, Grad-CAM ou gradients intégrés pour des images, attributions par token pour du texte) et ce qu'elles montrent, avec leur test de santé ; l'audit par sous-groupe (indicateurs, intervalles, effectifs) ; les limites de ces explications ; les usages à proscrire.

## Industrialisation et démonstration

TODO (PF.10, PF.11) : comment installer et lancer (la commande unique), ce que testent les tests, le badge de l'intégration continue, l'API ou la démo, la latence mesurée (p50 et p95, la machine).

## Limites et suite

TODO (PF.12) : ce que le projet ne montre pas, ce qui pourrait le contredire, ce que tu ferais avec un mois de plus.
