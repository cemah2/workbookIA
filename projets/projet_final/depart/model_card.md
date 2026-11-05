# Model card · TODO : le nom du modèle

<!-- Le modèle d'une model card (Mitchell et coll., 2019, « Model Cards for Model Reporting » ; 🛠️ 23.28, B6.28) : remplace chaque paragraphe qui commence par TODO. Une model card dit à quoi sert un modèle, à quoi il ne doit pas servir, comment on l'a mesuré et ce qu'il peut faire de travers. Les chiffres sont ceux du test scellé (experiments/test_results.json), avec leurs intervalles. -->

## Le modèle

TODO : l'architecture, le nombre de paramètres, le modèle pré-entraîné de départ s'il y en a un (son nom, sa révision, sa licence), l'entraînement (données, epochs, optimiseur, graine), la date, la version du paquet et les versions des bibliothèques (experiments/runs.jsonl).

## Usage prévu

TODO : la tâche, les utilisateurs visés, l'entrée et la sortie ; les **usages exclus** (et pourquoi) : une décision automatique sur une personne, une autre population, une autre période, un autre capteur.

## Les données

TODO : renvoi à data/card.md (source, licence, période, découpage, SHA-256 du découpage).

## Les métriques

TODO : la métrique principale et les secondaires sur le test scellé, avec leurs intervalles bootstrap à 95 %, à côté des deux baselines ; la latence (p50, p95, la machine).

## Explicabilité et équité

TODO (PF.9) : ce que montrent les attributions, les indicateurs par sous-groupe avec leurs effectifs et leurs intervalles, les écarts trouvés et ce qui a été fait (ou pas) pour les réduire.

## Les limites et risques

TODO : les cas où le modèle se trompe (l'analyse d'erreurs), le décalage de distribution attendu en usage réel, les risques d'un mauvais usage, ce qu'il faudrait faire avant tout usage réel (une validation sur de nouvelles données, une surveillance de la dérive, un humain dans la boucle).
