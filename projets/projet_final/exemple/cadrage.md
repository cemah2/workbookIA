# Cadrage · Prévoir la demande horaire de vélos pour planifier le rééquilibrage

> **Exemple de calibrage du workbook (PF.1)**, sur un dataset qui n'est pas le tien. Le niveau de détail attendu, pas un modèle à recopier.

## Le pitch en cinq lignes

1. Une station de vélos vide ou pleine fait perdre des clients ; l'équipe de rééquilibrage de l'opérateur planifie ses camions **la veille**, pour chaque heure du lendemain.
2. Les données : 17 379 heures de locations de Capital Bikeshare (Washington, 2011-2012), avec la météo et le calendrier ; Bike Sharing Dataset de l'UCI, licence CC BY 4.0.
3. Le modèle reçoit le calendrier, la météo prévue et l'historique connu jusqu'à 48 h avant ; il rend le nombre de locations de chaque heure du lendemain.
4. Réussi si l'erreur absolue moyenne (MAE) sur le dernier trimestre 2012 est au moins 10 % plus basse que celle d'un gradient boosting, et toujours sous la persistance hebdomadaire (la même heure, une semaine plus tôt).
5. Risque principal : la météo des données est la météo observée, pas la météo prévue la veille ; le modèle sera plus optimiste qu'en usage réel.

## Le problème et ses utilisateurs

Les planificateurs de l'opérateur, une fois par jour vers 18 h. Aujourd'hui, ils reprennent les chiffres de la semaine précédente, corrigés à la main pour les jours fériés : c'est la baseline naïve. Une **sous-estimation** aux heures de pointe laisse des stations vides (clients perdus, plaintes) ; une **surestimation** mobilise un camion pour rien (coût, sans perte de clients). Les deux erreurs coûtent, la première plus que la seconde.

## La métrique métier et la métrique technique

**Métier** : les vélos manquants ou en trop par heure, sur le réseau entier. **Technique** : la **MAE** en locations par heure, qui se lit dans la même unité. Secondaires : la MAE des heures de pointe (7 h à 9 h et 17 h à 19 h, les jours ouvrés), où les erreurs coûtent le plus ; la RMSE, qui punit les grosses erreurs ; la part des heures sous-estimées de plus de 100 vélos. Pas de R² : il ne parle pas aux planificateurs.

## Le critère de succès chiffré

Sur le test scellé (octobre à décembre 2012) : MAE du modèle retenu au moins **10 % plus basse** que celle du gradient boosting, avec un intervalle bootstrap apparié à 95 % de la différence entièrement sous 0 ; et une prédiction des 24 heures du lendemain en moins de 1 s sur un CPU. Si le critère n'est pas atteint, c'est le résultat du projet, et le gradient boosting est la recommandation.

## Ce que le projet ne fait pas

Pas de prévision par station (les données sont agrégées sur le réseau), pas de déploiement réel, pas d'optimisation des tournées des camions.

## Les risques

| Risque | Parade |
|---|---|
| Météo observée au lieu de la météo prévue : le modèle profite d'une information qu'il n'aura pas | le dire dans la model card ; une perturbation de robustesse (température bruitée de ± 2 °C, comme une erreur de prévision) |
| Deux années seulement, une demande qui croît d'une année à l'autre : le test ressemble peu à l'entraînement | un découpage temporel, une validation sur l'été 2012, la croissance discutée dans le rapport |
| Des événements rares (l'ouragan Sandy, fin octobre 2012) dans le test | les garder, les rapporter à part, ne rien exclure après avoir vu les erreurs |
