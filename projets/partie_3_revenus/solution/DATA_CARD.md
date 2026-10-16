# Data card · *Adult* (*Census Income*), recensement américain de 1994

Le format suit Gebru et coll., « Datasheets for Datasets » (2018 ; *Communications of the ACM*, 2021) (le ch. 29 y reviendra). La fiche du dépôt, [`data/cards/adult.md`](../../../data/cards/adult.md), en est la version courte ; celle-ci ajoute ce que le projet a mesuré.

## La provenance

- **L'extraction** : Barry Becker, à partir de la « 1994 Census database » du US Census Bureau (le supplément annuel ASEC de la *Current Population Survey*, d'après Ding et coll., 2021 : la documentation de l'UCI ne nomme pas l'enquête), avec les conditions âge > 16 ans, revenu > 100 dollars, poids d'échantillonnage > 1 et au moins une heure de travail par semaine ; publiée en 1996 sur le dépôt de l'UCI.
- **La référence** : Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. [doi:10.24432/C5XW20](https://doi.org/10.24432/C5XW20).
- **Le chargement** : `sklearn.datasets.fetch_openml(data_id=1590)` (OpenML réunit en une table les fichiers d'entraînement, 32 561 lignes, et de test, 16 281 lignes, d'origine), ou la copie versionnée `data/adult.csv.gz` (0,6 Mo), identique valeur par valeur, vérifiée par un test du dépôt. `data.load_census()` rend les 14 colonnes et la cible en 0/1.

## La licence

Creative Commons Attribution 4.0 (CC BY 4.0) : réutilisation, modification et redistribution permises, y compris commerciales, à condition de citer la source.

## La collecte

Une enquête par sondage du Bureau du recensement : chaque ligne est une personne interrogée, et `fnlwgt` dit combien de personnes de la population elle représente. Le fichier n'est donc pas un échantillon simple de la population américaine : ses proportions (par exemple 67 % d'hommes) sont celles des personnes retenues par l'extraction, non pondérées. Les « ? » sont des réponses non données. Le revenu est celui que la personne a déclaré à l'enquête.

## La composition

- **48 842 personnes**, 14 colonnes : 6 numériques (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) et 8 catégorielles (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).
- **La cible** : 11 687 revenus au-dessus de 50 000 dollars (23,9 %), 37 155 au-dessous.
- **Le découpage** : 39 073 personnes d'entraînement et 9 769 de test (20 %), stratifié (23,93 % de revenus élevés des deux côtés), graine 2026.
- **Les valeurs manquantes** (les « ? » des fichiers d'origine, en NaN), sur l'entraînement : `occupation` 2 244 (5,7 %), `workclass` 2 235 (5,7 %), `native-country` 691 (1,8 %). Les 2 235 personnes sans `workclass` n'ont pas non plus d'`occupation` ; les 9 autres sans métier n'ont jamais travaillé. Ces personnes n'ont que 10,0 % de revenus élevés : le manque est une information, pas un bruit.
- **Les doublons** : 36 lignes en double exact dans l'entraînement (52 dans la table entière) ; 4 475 sans `fnlwgt`. Gardés : des réponses identiques de personnes différentes sont plausibles.
- **Des colonnes particulières** : `fnlwgt` est un poids d'échantillonnage (le nombre de personnes que la ligne représente), pas une caractéristique de la personne ; `education` est une copie exacte de `education-num` (16 niveaux). `capital-gain` est nul pour 91,7 % des personnes et plafonné à 99 999 (193 personnes) ; `hours-per-week` va de 1 à 99 (108 personnes à 99, une valeur plafond). `relationship` mêle la situation familiale et le sexe : « Husband » compte 15 745 hommes et une femme, « Wife » 1 853 femmes et 3 hommes.

## Les biais et limites

- **Une photographie de 1994** : les revenus, les métiers et le seuil de 50 000 dollars (environ le double en dollars d'aujourd'hui) ont changé.
- **Un seuil haut** : le 76ᵉ centile des revenus individuels de 1994, et les 88ᵉ et 89ᵉ pour les personnes noires et pour les femmes (Ding et coll., 2021) : dans ces groupes, la plupart des personnes sont sous le seuil.
- **Des attributs sensibles** (`sex`, `race`, `native-country`, `marital-status`, `relationship`) et leurs **indicateurs indirects** : retirer `sex` ne retire pas l'information du sexe (MP3.7 le mesure).
- **Des catégories datées ou réductrices** : deux sexes, cinq catégories raciales du recensement de 1994 ; des groupes petits (« Amer-Indian-Eskimo » : 1 % des personnes, « Other » : 0,8 %).
- **Un échantillon d'enquête** : les poids `fnlwgt` ne sont pas utilisés ; les proportions du fichier ne sont pas celles de la population américaine.
- **Un benchmark « mis à la retraite »** par Ding, Hardt, Miller et Schmidt (2021), [*Retiring Adult: New Datasets for Fair Machine Learning*](https://arxiv.org/abs/2108.04884), qui proposent les tâches ACS du paquet `folktables`.

## Les usages

- **Appropriés** : apprendre la démarche d'un projet tabulaire (prétraitement, validation croisée, déséquilibre, audit), comparer des méthodes, illustrer les limites de l'équité « par aveuglement ».
- **À éviter** : toute conclusion sur les revenus d'aujourd'hui, toute décision sur des personnes, et tout travail de recherche sur l'équité qui se contenterait d'*Adult*.
