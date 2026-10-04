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

<!-- wb:section 4 -->
## 4 — Règle de Bayes

| Compétence | Niveau |
|---|:-:|
| Expliquer la différence entre les points de vue fréquentiste et bayésien, sans caricature | |
| Démontrer la règle de Bayes à partir de la règle du produit et nommer ses quatre termes (prior, vraisemblance, évidence, posterior) | |
| Calculer un posterior à la main pour deux hypothèses, puis pour quelques hypothèses, et avec la forme « cotes » | |
| Relier la règle de Bayes à la matrice de confusion (precision, NPV) et à la prévalence, sans inverser la condition | |
| Implémenter la mise à jour séquentielle et vérifier qu'elle ne dépend pas de l'ordre des observations | |
| Diagnostiquer et corriger un underflow en passant aux log-probabilités (astuce log-sum-exp) | |
| Reconnaître la loi Beta dans le posterior du biais d'une pièce et le résumer (MAP, moyenne, $P(\theta > x)$) | |
| Calculer et interpréter un intervalle de crédibilité, et le comparer à un intervalle de confiance bootstrap | |
| Écrire les fonctions de `mylearn.bayes` et les valider par des tests à oracle (SciPy) | |
<!-- wb:end 4 -->

<!-- wb:section 5 -->
## 5 — Courbes et surfaces

| Compétence | Niveau |
|---|:-:|
| Reconnaître une courbe continue, lisse et univoque, et dire pourquoi on l'exige (et ce qui se passe pour ReLU) | |
| Distinguer extrema locaux et globaux, et les repérer sur une courbe échantillonnée | |
| Approcher une dérivée par une différence finie et choisir le pas $h$ (erreur de troncature contre erreur d'arrondi) | |
| Calculer un gradient à la main et numériquement, et interpréter sa direction, sa norme et les lignes de niveau | |
| Implémenter la descente (et la montée) de gradient, et diagnostiquer un learning rate trop petit ou trop grand | |
| Classer un point critique (minimum, maximum, point selle, plat) et expliquer pourquoi les points selles comptent en grande dimension | |
| Vérifier un gradient numérique avec la différentiation automatique de PyTorch | |
| Écrire les fonctions de `mylearn.calculus` et les valider par des tests à oracle (PyTorch, SciPy) | |
<!-- wb:end 5 -->

<!-- wb:section 6 -->
## 6 — Théorie de l'information

| Compétence | Niveau |
|---|:-:|
| Calculer la surprise d'un événement en bits et en nats, et le nombre de bits d'un code de longueur fixe | |
| Calculer l'entropie d'une distribution, et dire quand elle est nulle ou maximale | |
| Construire un code de Huffman à la main et en Python, encoder et décoder, et le comparer au Morse et à l'entropie | |
| Calculer une cross-entropy et une divergence KL, démontrer $H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)$ et expliquer l'asymétrie de la KL | |
| Estimer une distribution de lettres ou de mots avec lissage, et réparer une cross-entropy infinie | |
| Relier cross-entropy, log loss et perplexité à l'entraînement des classifieurs et des modèles de langage | |
| Mesurer ce que le contexte fait gagner (bigrammes, compresseurs, blocs de lettres) | |
| Écrire les fonctions de `mylearn.info` et les valider par des tests à oracle (SciPy, scikit-learn, PyTorch) | |
<!-- wb:end 6 -->
<!-- wb:section CP1 -->
## CP1 — Checkpoint de la partie I

| Bilan | Résultat |
|---|:-:|
| Note de l'examen blanc (sur 20), et date | |
| Questions où j'ai eu moins de la moitié des points (remédiation du corrigé faite ?) | |
| Reprise de ces questions une semaine plus tard (date, points) | |
| Note du mini-projet MP1 avec la grille (sur 20) | |
| Je sais redessiner la carte mentale de la partie I de mémoire (0 à 3) | |
| Je sais écrire les 20 formules clés et dire ce que chacune mesure (0 à 3) | |
| Je sais définir à l'oral les 11 mots du vocabulaire de la synthèse (0 à 3) | |
<!-- wb:end CP1 -->
<!-- wb:section 7 -->
## 7 — Classification

| Compétence | Niveau |
|---|:-:|
| Distinguer classification binaire, multi-classe et multi-étiquette, et lire des régions et des frontières de décision | |
| Choisir un seuil de décision d'après le coût des faux positifs et des faux négatifs, et dire qui doit fixer ces coûts | |
| Compter les classifieurs d'un un-contre-tous et d'un un-contre-un, dépouiller des votes, et choisir entre les deux stratégies | |
| Programmer le centroïde le plus proche et les méta-estimateurs `OneVsRestClassifier` et `OneVsOneClassifier` | |
| Dérouler k-means à la main, et programmer Lloyd et k-means++ en NumPy vectorisé (`KMeans`) | |
| Choisir un nombre de clusters avec l'inertie et la silhouette, et reconnaître quand préférer DBSCAN ou HDBSCAN | |
| Calculer une densité d'échantillons, le volume d'une boule et le rayon de l'hyper-orange en dimension $d$ | |
| Expliquer la malédiction de la dimension, le phénomène de Hughes, la concentration des distances et leurs parades | |
<!-- wb:end 7 -->
<!-- wb:section 8 -->
## 8 — Entraînement et test

| Compétence | Niveau |
|---|:-:|
| Décrire la boucle d'entraînement (prédiction, comparaison, mise à jour, epochs) et dire pourquoi le score d'entraînement trompe (mémorisation, raccourcis appris) | |
| Découper des données en entraînement, validation et test, avec ou sans stratification, et calculer les tailles obtenues (hold-out, folds) | |
| Programmer `train_test_split`, la k-fold simple et stratifiée, `clone` et `cross_val_score`, et les vérifier contre scikit-learn | |
| Choisir un hyperparamètre sur un jeu de validation, réentraîner, puis tester une seule fois, et expliquer pourquoi le score de validation du gagnant est optimiste | |
| Repérer une fuite de données dans un protocole (prétraitement, sélection de features, choix sur le test, doublons, features illégitimes, données dépendantes) et la corriger | |
| Choisir le schéma de validation adapté aux données : hold-out, k-fold, groupes (`GroupKFold`), séries temporelles (`TimeSeriesSplit`), validation croisée imbriquée | |
| Chiffrer l'incertitude d'un score (erreur type, taille de test nécessaire) et comparer deux modèles avec un test apparié (permutation, McNemar, bootstrap apparié) | |
| Expliquer en entretien les trois jeux (entraînement, validation, test), la p-valeur, et les causes d'un modèle qui déçoit en production | |
<!-- wb:end 8 -->
<!-- wb:section 9 -->
## 9 — Overfitting et underfitting

| Compétence | Niveau |
|---|:-:|
| Diagnostiquer l'underfitting et l'overfitting sur des courbes d'entraînement, de validation et d'apprentissage, et choisir le remède qui convient | |
| Calculer et interpréter MSE, RMSE, MAE et R², et dire laquelle choisir selon le coût des grosses erreurs | |
| Dériver à la main les moindres carrés, Ridge et Lasso en dimension 1, et expliquer pourquoi la pénalité L1 met des poids exactement à zéro | |
| Programmer `polynomial_features`, `LinearRegression`, `Ridge` et `Lasso` (descente de coordonnées) et les vérifier contre scikit-learn | |
| Appliquer l'early stopping avec patience et `min_delta`, et choisir la force de la régularisation par validation croisée | |
| Mesurer le biais² et la variance d'une famille de modèles par simulation, et vérifier la décomposition biais² + variance + bruit | |
| Ajuster une droite par mises à jour bayésiennes sur une grille pente-ordonnée, et relier le MAP à Ridge | |
| Expliquer en entretien le compromis biais-variance, L1 contre L2, et ce que la double descente change (ou non) | |
<!-- wb:end 9 -->
<!-- wb:section 10 -->
## 10 — Neurones

| Compétence | Niveau |
|---|:-:|
| Décrire le fonctionnement simplifié d'un neurone biologique (neurotransmetteurs, somme, seuil, décharge) et ce que le neurone artificiel en garde | |
| Calculer à la main la sortie d'un perceptron et d'un neurone moderne (poids, biais, activation), avec et sans l'astuce du biais | |
| Trouver des poids pour les portes logiques, démontrer qu'aucun perceptron ne calcule XOR, et câbler XOR avec deux couches | |
| Appliquer à la main la règle d'apprentissage du perceptron, et démontrer le théorème de convergence $(R/\gamma)^2$ | |
| Programmer `sign_step`, `add_bias_column`, `neuron_forward` et la classe `Perceptron`, et les vérifier contre NumPy, PyTorch et scikit-learn | |
| Traduire les poids nommés d'un schéma (AD, DA) en matrice, dans la convention de mylearn et dans celle de PyTorch | |
| Reconnaître sur des courbes d'erreurs si des données sont séparables, et stabiliser un perceptron qui ne converge pas (moyenne des poids, pocket) | |
| Raconter l'histoire du perceptron avec ses nuances, et expliquer en entretien le rôle du biais et des activations dérivables | |
<!-- wb:end 10 -->
<!-- wb:section 11 -->
## 11 — Apprentissage et raisonnement

| Compétence | Niveau |
|---|:-:|
| Décomposer un algorithme en représentation, évaluation et optimisation, et dire ce que le théorème No Free Lunch dit et ne dit pas | |
| Distinguer loss, métrique et objectif, et precision et recall, sur un cas concret | |
| Distinguer déduction, induction et abduction, et juger si un raisonnement est valide et s'il est solide | |
| Vérifier un syllogisme par les règles de distribution, par un diagramme de Venn et par force brute (256 mondes), et nommer les sophismes classiques | |
| Chiffrer une généralisation (erreur type, règle de succession) et repérer un échantillon trop petit ou biaisé | |
| Classer une rétroaction dans les quatre cases du conditionnement opérant | |
| Programmer un bandit, ε-greedy, UCB et Thompson dans `mylearn.bandit`, et les vérifier contre NumPy et SciPy | |
| Comparer équitablement des stratégies par leur regret, et journaliser une expérience pour pouvoir la refaire | |
<!-- wb:end 11 -->
<!-- wb:section CP2 -->
## CP2 — Checkpoint de la partie II

| Bilan | Résultat |
|---|:-:|
| Note de l'examen blanc (sur 20), et date | |
| Questions où j'ai eu moins de la moitié des points (remédiation du corrigé faite ?) | |
| Reprise de ces questions une semaine plus tard (date, points) | |
| Note du mini-projet MP2 avec la grille (sur 20) | |
| Je sais redessiner la carte mentale de la partie II de mémoire (0 à 3) | |
| Je sais écrire les 22 formules clés et dire ce que chacune mesure (0 à 3) | |
| Je sais définir à l'oral les 14 mots du vocabulaire de la synthèse (0 à 3) | |
| Je sais décrire, sans notes, un protocole d'évaluation sans fuite (test gelé, prétraitement dans chaque fold, règle de choix, ouverture unique du test) (0 à 3) | |
<!-- wb:end CP2 -->
