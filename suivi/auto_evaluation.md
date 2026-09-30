# Auto-évaluation des compétences

> **Modèle tenu par Claude : ne le remplis pas ici.** Ta copie personnelle est `mon_travail/suivi/auto_evaluation.md`, créée par `python tools/start_chapter.py --init`. Les sections des nouveaux chapitres y sont ajoutées automatiquement à la fin, sans toucher à ce que tu as déjà écrit.

À remplir à la fin de chaque chapitre, puis à revoir avant chaque checkpoint. Sois honnête : c'est un outil pour savoir quoi réviser, pas une note.

| Niveau | Signification |
|:-:|---|
| **0** | Je ne sais pas encore de quoi il s'agit. |
| **1** | Je reconnais la notion et je peux la décrire avec le cours sous les yeux. |
| **2** | Je sais l'appliquer seul sur un exercice standard. |
| **3** | Je sais l'expliquer à quelqu'un, l'implémenter from scratch et répondre en entretien. |

Les compétences de chaque chapitre (issues des objectifs de sa fiche) sont ajoutées par Claude à la génération du chapitre. Note la date à chaque évaluation pour voir ta progression : `2 (10/10) → 3 (02/11)`.

<!-- wb:section setup -->
## Mise en place

| Compétence | Niveau |
|---|:-:|
| Ouvrir un notebook dans Colab et exécuter toutes les cellules | |
| Activer un environnement virtuel et lancer Jupyter en local | |
| Utiliser `git pull` pour récupérer les nouveaux chapitres | |
| Lancer les tests `pytest` et lire leur résultat | |
<!-- wb:end setup -->
<!-- wb:section 0A -->
## 0A — Python, notebooks et outils

| Compétence | Niveau |
|---|:-:|
| Utiliser un notebook (Colab ou Jupyter) et le terminal : ordre d'exécution, « Run all », `wb.check`, pytest, git | |
| Écrire du Python lisible : types, structures de données, boucles, compréhensions, fonctions (lambda, `*args`, fermetures) | |
| Lire et écrire des classes (méthodes spéciales, héritage, générateurs), comme celles de scikit-learn et PyTorch | |
| Manipuler des arrays NumPy : formes, masques, broadcasting, réductions par axe, `reshape`, aléatoire reproductible | |
| Explorer un dataset avec pandas (sélection, valeurs manquantes, `groupby`) et le visualiser avec matplotlib | |
| Lire, écrire et sérialiser des fichiers (`pathlib`, JSON, pickle) ; extraire de l'information avec une regex | |
| Documenter, tester et versionner : docstring NumPy, doctest, pytest, commits propres | |
| Écrire des fonctions `mylearn` (`utils`) et les valider par des tests à oracle | |
<!-- wb:end 0A -->
<!-- wb:section 0B -->
## 0B — Maths du lycée au ML

| Compétence | Niveau |
|---|:-:|
| Calculer avec Σ, Π, puissances, valeur absolue, partie entière, coefficients binomiaux, suites géométriques | |
| Reconnaître et manipuler les fonctions usuelles : affine, polynôme, exp, logarithmes, sigmoïde, tanh, cosinus | |
| Calculer normes, distances, produits scalaires, cosinus, produits matrice-vecteur et matriciels, en vérifiant les formes | |
| Dériver (règles usuelles, règle de la chaîne) et trouver un minimum en annulant la dérivée | |
| Calculer dérivées partielles et gradient, lire des lignes de niveau, sommer sur les chemins d'un graphe de calcul | |
| Calculer probabilités, espérance et variance d'une variable discrète, et les confirmer par simulation | |
| Implémenter l'algèbre linéaire en Python pur (`linalg_basics`) et la valider contre NumPy | |
<!-- wb:end 0B -->
<!-- wb:section 1 -->
## 1 — Introduction au machine learning et au deep learning

| Compétence | Niveau |
|---|:-:|
| Expliquer la différence entre un système expert (règles écrites) et un modèle appris à partir d'exemples | |
| Employer le vocabulaire de base : échantillon, feature, label, paramètre, hyperparamètre, loss, learning rate, généralisation | |
| Classer une tâche : classification, régression, clustering, débruitage, réduction de dimension, génération, renforcement | |
| Charger et décrire les quatre fils rouges (Penguins, MNIST, Holmes et Verne, taches solaires) et lire leur data card | |
| Coder une boucle d'entraînement minimale et expliquer l'effet du learning rate | |
| Diagnostiquer une évaluation faussée : mémorisation, test vu à l'entraînement, fuite du label | |
| Situer les LLM, les modèles de diffusion et les foundation models sur la carte du ML (réponse d'entretien en une minute) | |
<!-- wb:end 1 -->
<!-- wb:section 2 -->
## 2 — Hasard et statistiques de base

| Compétence | Niveau |
|---|:-:|
| Calculer à la main et en NumPy moyenne, médiane, mode, variance (ddof 0 ou 1), écart-type, percentiles et z-scores, et choisir entre moyenne et médiane | |
| Distinguer loi discrète (pmf) et densité (aire = probabilité), et reconnaître les lois uniforme, normale, de Bernoulli et catégorielle | |
| Utiliser la règle 68-95-99,7 et dire quand elle ne s'applique pas | |
| Tirer au hasard de façon reproductible (graine, `default_rng`), dans une loi discrète, avec ou sans remise | |
| Expliquer l'hypothèse i.i.d. et repérer ce qui la casse (séries temporelles, doublons, dérive) | |
| Construire un intervalle de confiance par bootstrap, l'expliquer et dire ce qu'il ne corrige pas | |
| Calculer et interpréter covariance, corrélation et leurs matrices, sans confondre corrélation et causalité | |
| Justifier par un graphique (Anscombe) qu'il faut toujours regarder ses données | |
| Écrire les fonctions de `mylearn.stats` et les valider par des tests à oracle | |
<!-- wb:end 2 -->

<!-- wb:section 3 -->
## 3 — Probabilités et mesure de la qualité

| Compétence | Niveau |
|---|:-:|
| Calculer des probabilités simples, conditionnelles, jointes et marginales à partir d'aires, de comptages ou d'une table (`pd.crosstab`) | |
| Ne pas confondre $P(A \mid B)$ et $P(B \mid A)$, et savoir quand deux événements sont indépendants | |
| Construire une matrice de confusion (et la lire dans la disposition de scikit-learn) | |
| Calculer et expliquer accuracy, precision, recall, spécificité, NPV, F1 et les autres mesures | |
| Choisir la mesure et le seuil selon le coût des faux positifs et des faux négatifs | |
| Expliquer pourquoi, quand la maladie est rare, la plupart des résultats positifs d'un très bon test peuvent être faux | |
| Distinguer les moyennes macro, micro et pondérée sur plusieurs classes | |
| Tracer et interpréter une courbe ROC (AUC) et une courbe precision-recall (average precision) | |
| Vérifier la calibration de probabilités (diagramme de fiabilité, score de Brier) | |
| Écrire les fonctions de `mylearn.metrics` et les valider contre scikit-learn | |
<!-- wb:end 3 -->
