# Cadrage · TODO : le titre du projet

<!-- PF.1 : une page, pas plus. Écris-la AVANT de toucher aux données, puis commite-la. Le pitch reprend ton Ex 29.4 (cinq lignes pour quelqu'un qui ne fait pas de deep learning). Remplace chaque paragraphe qui commence par TODO. Un exemple rempli, sur un autre dataset : projets/projet_final/exemple/cadrage.md du workbook. -->

## Le pitch en cinq lignes

TODO : (1) le problème et pour qui il compte ; (2) les données (source, taille, licence) ; (3) ce que le modèle reçoit et ce qu'il rend ; (4) comment tu sauras si c'est réussi (une métrique et une baseline à battre) ; (5) un risque ou une limite connue.

## Le problème et ses utilisateurs

TODO : qui utiliserait le modèle, pour quelle décision, à quelle fréquence, et ce qu'on fait aujourd'hui sans lui (la règle simple, l'humain, rien). Ce qu'une erreur coûte, dans chaque sens (un faux positif, un faux négatif, une surestimation).

## La métrique métier et la métrique technique

TODO : la **métrique métier** (ce que l'utilisateur compte : des heures gagnées, des vélos manquants, des alertes manquées) et la **métrique technique** qui la suit (F1 macro, recall à precision fixée, MAE, perplexité, retour moyen) ; pourquoi celle-là et pas l'accuracy (déséquilibre, coûts asymétriques).

## Le critère de succès chiffré

TODO : un nombre fixé **maintenant**, avant les expériences : « le modèle bat la baseline classique d'au moins X points de <métrique> sur le test scellé, intervalle bootstrap à 95 % au-dessus de 0 », et une contrainte (latence, taille, CPU). Ce qui se passera si le critère n'est pas atteint (c'est un résultat, il se rapporte).

## Ce que le projet ne fait pas

TODO : le périmètre (pas de déploiement réel, pas d'état de l'art, une seule langue ou une seule ville…).

## Les risques

TODO : trois risques au plus, chacun avec ce que tu feras : sur les données (fuite, labels faux, données personnelles), sur le modèle (raccourci appris, décalage de distribution), sur l'usage (décision automatique sur des personnes, mauvaise interprétation d'un score).
