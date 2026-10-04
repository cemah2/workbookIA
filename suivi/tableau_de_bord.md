# Tableau de bord

> **Modèle tenu par Claude : ne coche pas ici.** Ta copie personnelle est `mon_travail/suivi/tableau_de_bord.md`, créée par `python tools/start_chapter.py --init`. À chaque nouveau chapitre, `start_chapter.py` ajoute à la fin de ta copie les sections publiées depuis, sans toucher à tes cases déjà cochées ; si une section déjà copiée change ici (titre, ★ ou durée), il te le signale (ℹ️) sans modifier ta copie.

Coche les cases au fur et à mesure (`- [x]`).

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · 🚀 GPU Colab conseillé

## Progression globale

| Partie | Chapitres | Exercices faits | Temps passé |
|---|---|---|---|
| Mise en place | setup | / 6 | |
| 0 · Prérequis | 0A, 0B | | |
| I · Fondations | 1 à 6 | | |
| II · Concepts | 7 à 11 | | |
| III · ML classique | 12 à 15 | | |
| IV · Réseaux | 16 à 20 | | |
| V · Architectures | 21 à 24 | | |
| VI · Génératif et RL | 25 à 29 | | |
| VII · Bonus | B1 à B8 | | |

<!-- wb:section setup -->
## Mise en place ⏱️ 1 h 30 environ

- [ ] Ouvrir `00_setup/demo.ipynb` dans Colab et tout exécuter (⏱️ 15 min)
- [ ] demo.3 et demo.4 jusqu'à obtenir ✅ : facultatif, à refaire après les premiers pas pandas de 0A (partie E) (⏱️ 5 min)
- [ ] Installer l'environnement local avec `00_setup/INSTALL_LOCAL.md` (⏱️ 45 min, optionnel)
- [ ] `python 00_setup/check_env.py` tout vert (⏱️ 2 min)
- [ ] `python tools/start_chapter.py --init` : crée ta librairie `mon_travail/mylearn` et tes fichiers de suivi (⏱️ 2 min) ; tu écriras `mean` en 0A.26
- [ ] Installer Anki (⏱️ 10 min)
<!-- wb:end setup -->

<!-- wb:section 0A -->
## 0A — Python, notebooks et outils ⏱️ 28 h

- [ ] 0A.Q1–Q12 🧠 Quiz (12 questions, 37 min)
- [ ] 0A.1 ✏️ Évaluer des expressions à la main : //, %, **, conversions ★ 10 min
- [ ] 0A.2 ✏️ Indices et tranches à la main : listes, tuples, chaînes ★ 10 min
- [ ] 0A.3 ✏️ Dérouler une boucle et une compréhension pas à pas ★ 15 min
- [ ] 0A.4 ✏️ Un groupby à la main sur huit manchots ★ 10 min
- [ ] 0A.5 ✏️ Mini-batches : combien de batches, de quelle taille, combien de mises à jour ? ★ 10 min
- [ ] 0A.6 ✏️ Portée, valeurs par défaut et arguments nommés : qui vaut quoi ? ★★ 15 min
- [ ] 0A.7 ✏️ Formes NumPy à la main : indexation, réductions, reshape ★★ 15 min
- [ ] 0A.8 ✏️ Broadcasting : compatibles ou non, et quelle forme ? ★★ 15 min
- [ ] 0A.9 🗣️ Liste Python ou array NumPy : l'expliquer en cinq lignes ★ 10 min
- [ ] 0A.10 🛠️ Premier commit propre depuis le terminal ★★ 20 min
- [ ] 0A.11 🛠️ .gitignore : ce qui ne doit jamais entrer dans le dépôt ★★ 15 min
- [ ] 0A.12 🛠️ Une branche pour essayer sans risque (aperçu) ★★ 15 min
- [ ] 0A.13 🔮 Ordre d'exécution des cellules : que vaut x ? ★ 10 min
- [ ] 0A.14 🔨 Nombres et f-strings : la fiche d'un manchot ★ 13 min
- [ ] 0A.15 🔨 Chaînes : nettoyer les noms d'espèces de Penguins brut ★ 13 min
- [ ] 0A.16 🔨 Listes : les nageoires de dix manchots ★ 13 min
- [ ] 0A.17 🔨 Tuples et déballage : renvoyer et échanger plusieurs valeurs ★ 13 min
- [ ] 0A.18 🔨 Dictionnaires : une fiche par espèce ★ 15 min
- [ ] 0A.19 🔨 Ensembles : quelles espèces sur quelles îles ? ★ 13 min
- [ ] 0A.20 🔨 Conditions : classer un manchot selon sa masse ★ 13 min
- [ ] 0A.21 🔨 Boucles : for, range, enumerate, zip et while ★ 15 min
- [ ] 0A.22 🔨 Compréhensions : filtrer et transformer en une ligne ★ 15 min
- [ ] 0A.23 🔨 Tes premières fonctions : paramètres, valeurs par défaut, return ★ 15 min
- [ ] 0A.24 🔨 Importer des modules : math, random, statistics et Counter ★ 13 min
- [ ] 0A.25 🔨 Exceptions : lever une ValueError et la rattraper ★ 15 min
- [ ] 0A.26 🔨 Ton premier module mylearn : mean et son test ★ 15 min
- [ ] 0A.27 📦 Premiers arrays NumPy : dtype, shape, ndim ★ 13 min
- [ ] 0A.28 📦 Indexation, tranches et masques booléens ★ 15 min
- [ ] 0A.29 🔮 Vue ou copie : qui est modifié ? ★ 10 min
- [ ] 0A.30 📦 Calcul vectorisé : unités, normalisation, fonctions universelles ★ 15 min
- [ ] 0A.31 📦 Aléatoire reproductible : default_rng, graine, permutation, choice ★ 15 min
- [ ] 0A.32 📦 Premier contact avec Penguins : read_csv, head, info, describe ★ 13 min
- [ ] 0A.33 📦 Sélectionner : colonnes, loc, iloc et filtres ★ 15 min
- [ ] 0A.34 📦 Valeurs manquantes et doublons : isna, dropna, fillna, duplicated ★ 15 min
- [ ] 0A.35 📦 De pandas à NumPy : construire X et y ★ 13 min
- [ ] 0A.36 📦 Premiers graphiques : plot, scatter, hist ★ 15 min
- [ ] 0A.37 🐛 Lire un traceback : cinq bugs de débutant ★★ 15 min
- [ ] 0A.38 🔨 Lire penguins.csv comme un simple fichier texte (pathlib, with) ★★ 20 min
- [ ] 0A.39 🔨 Sauvegarder et recharger des résultats : json et pickle ★★ 20 min
- [ ] 0A.40 🔨 Arguments variables : *args, **kwargs et keyword-only ★★ 26 min
- [ ] 0A.41 🔨 Fonctions en argument : lambda, key= et Callable ★★ 20 min
- [ ] 0A.42 🔨 Fermetures : une fabrique de fonctions ★★ 26 min
- [ ] 0A.43 🔮 Le piège des lambdas créées dans une boucle ★★ 15 min
- [ ] 0A.44 🔨 Fonctions récursives : parcourir un arbre de dictionnaires (profondeur, nombre de feuilles) ★★ 25 min
- [ ] 0A.45 🔨 Expressions régulières : identifiants et dates de Penguins brut ★★ 26 min
- [ ] 0A.46 🔨 itertools et heapq : paires de features, grille, top-k ★★ 26 min
- [ ] 0A.47 🔨 Une classe RunningStats : __init__, attributs, méthodes, puis la même en @dataclass ★★ 26 min
- [ ] 0A.48 🔨 Méthodes spéciales : une classe Vector2D qui s'additionne ★★ 30 min
- [ ] 0A.49 🔨 Héritage et super() : un mini-estimateur fit/predict appelable ★★ 30 min
- [ ] 0A.50 🔨 Générateurs et itérables : un mini-Dataset de manchots ★★ 30 min
- [ ] 0A.51 📦 Réductions par axe, tri, argmax et unique ★★ 26 min
- [ ] 0A.52 📦 Broadcasting : standardiser toutes les colonnes d'un coup ★★ 26 min
- [ ] 0A.53 📦 reshape, transposée et empilement ★★ 20 min
- [ ] 0A.54 📦 Images MNIST : un tableau (N, 28, 28) ★★ 26 min
- [ ] 0A.55 🔬 Boucle Python contre NumPy : mesurer le gain ★★ 20 min
- [ ] 0A.56 🐛 Bugs NumPy : axis oublié, formes (n,) et (n, 1), vue modifiée ★★ 20 min
- [ ] 0A.57 📦 Compter et regrouper : value_counts, groupby, agg, sort_values ★★ 26 min
- [ ] 0A.58 🐛 Le filtre qui ne filtre pas : and, &, parenthèses et copies ★★ 15 min
- [ ] 0A.59 📦 Figures à plusieurs panneaux : subplots, imshow, show_images ★★ 26 min
- [ ] 0A.60 📈 Quelle mesure sépare le mieux les espèces ? ★★ 15 min
- [ ] 0A.61 🛠️ Une docstring au format NumPy, vérifiée par doctest ★★ 20 min
- [ ] 0A.62 🛠️ Écrire tes propres tests : assert, approx, raises, parametrize ★★ 25 min
- [ ] 0A.63 🔨 utils.count_values : compter sans pandas ★★ 26 min
- [ ] 0A.64 🔨 utils.argmax : le premier maximum, avec des boucles ★★★ 35 min
- [ ] 0A.65 🔨 utils.one_hot : des labels aux vecteurs ★★ 26 min
- [ ] 0A.66 🔨 utils.iterate_minibatches : découper un dataset en mini-batches ★★★ 39 min
- [ ] 0A.67 🏆 Enquête : dix questions sur les manchots, dix réponses vérifiées ★★★ 45 min
- [ ] 0A.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (30 cartes)
<!-- wb:end 0A -->

<!-- wb:section 0B -->
## 0B — Maths du lycée au ML ⏱️ 21 h

- [ ] 0B.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] 0B.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 0B.1 ✏️ Puissances, racines et notation scientifique sans calculatrice ★ 10 min
- [ ] 0B.2 ✏️ Valeur absolue, partie entière et signe : tableau de valeurs ★ 10 min
- [ ] 0B.3 ✏️ Lire et calculer des Σ et des Π ★ 15 min
- [ ] 0B.4 ✏️ Moyenne pondérée, somme pondérée et moyenne mobile à la main ★ 15 min
- [ ] 0B.5 ✏️ Ensembles : union, intersection, complémentaire et cardinal ★ 10 min
- [ ] 0B.6 ✏️ Droites et paraboles : pente, ordonnée à l'origine, racines, sommet ★ 15 min
- [ ] 0B.7 ✏️ Cosinus : cercle, période et planning en cosinus ★ 15 min
- [ ] 0B.8 ✏️ Vecteurs : somme, multiple, norme et distance entre deux manchots ★ 10 min
- [ ] 0B.9 ✏️ Transposée et produit matrice-vecteur : deux lectures ★ 15 min
- [ ] 0B.10 ✏️ Taux d'accroissement : de la sécante à la tangente ★ 15 min
- [ ] 0B.11 ✏️ Probabilités : issues, complémentaire, union ★ 10 min
- [ ] 0B.12 ✏️ Dénombrer : choix successifs, factorielle et C(n, k) ★★ 20 min
- [ ] 0B.13 ✏️ Suites géométriques : ce qui fond, ce qui explose ★★ 15 min
- [ ] 0B.14 ∂ La somme géométrique démontrée pas à pas ★★ 15 min
- [ ] 0B.15 ✏️ Exponentielles et logarithmes : règles de calcul en bases 2, e et 10 ★★ 20 min
- [ ] 0B.16 ∂ Changer de base : log₂ x = ln x / ln 2, bits et nats ★★ 15 min
- [ ] 0B.17 ✏️ Sigmoïde et tanh : valeurs, limites, symétries ★★ 15 min
- [ ] 0B.18 ✏️ Produit scalaire, similarité cosinus et produit de Hadamard ★★ 15 min
- [ ] 0B.19 ∂ Développer ‖a − b‖² avec le produit scalaire ★★ 15 min
- [ ] 0B.20 ✏️ Produit matriciel : calculer et vérifier les formes ★★ 20 min
- [ ] 0B.21 ✏️ Identité, inverse 2 × 2 et système de deux équations ★★ 20 min
- [ ] 0B.22 ✏️ Dériver avec les règles : somme, produit, quotient, exp, ln ★★ 25 min
- [ ] 0B.23 ✏️ Règle de la chaîne : décomposer, puis dériver ★★ 25 min
- [ ] 0B.24 ∂ La moyenne minimise la somme des carrés des écarts ★★ 20 min
- [ ] 0B.25 ✏️ Lignes de niveau, dérivées partielles, gradient et un pas de descente ★★ 25 min
- [ ] 0B.26 ✏️ Indépendance : tester P(A ∩ B) = P(A) P(B) avec deux dés ★★ 15 min
- [ ] 0B.27 ✏️ Espérance et variance d'une variable discrète ★★ 20 min
- [ ] 0B.28 ∂ Variance : deux formules, et l'espérance est linéaire ★★ 20 min
- [ ] 0B.29 ∂ Règle de la chaîne à deux variables : la somme sur les chemins ★★★ 30 min
- [ ] 0B.30 🧮 Fermi : combien de multiplications dans un produit matriciel ? ★★ 15 min
- [ ] 0B.31 🗣️ Le gradient expliqué à un randonneur dans le brouillard ★ 10 min
- [ ] 0B.32 🛠️ Écrire des maths en LaTeX dans Markdown ★ 10 min
- [ ] 0B.33 📦 Calculer avec Python : puissances, arrondis, abs, signe et C(n, k) ★ 10 min
- [ ] 0B.34 🔮 0,99 puissance 1000 : presque 1 ou presque 0 ? ★ 10 min
- [ ] 0B.35 📦 Σ, Π et moyennes en code : sum, math.prod, np.average, moyenne mobile ★★ 15 min
- [ ] 0B.36 📦 Galerie des fonctions usuelles : de l'affine au cosinus ★★ 20 min
- [ ] 0B.37 🐛 exp et log en NumPy : -inf, nan et dépassements ★★ 15 min
- [ ] 0B.38 🔨 linalg_basics (1) : additionner, soustraire, multiplier des vecteurs ★★ 15 min
- [ ] 0B.39 🔨 linalg_basics (2) : produit scalaire, norme, distance, cosinus ★★ 20 min
- [ ] 0B.40 📦 Tes fonctions contre NumPy : mêmes résultats, autre vitesse ★★ 15 min
- [ ] 0B.41 🔬 Distance ou similarité cosinus : l'effet de la longueur ★★ 20 min
- [ ] 0B.42 🔨 linalg_basics (3) : forme, transposée, identité, matrice × vecteur ★★ 20 min
- [ ] 0B.43 🔨 linalg_basics (4) : matmul et vérification des formes ★★ 25 min
- [ ] 0B.44 🔮 AB = BA ? (AB)ᵀ = BᵀAᵀ ? Prédire, puis tester ★★ 15 min
- [ ] 0B.45 🐛 Le produit qui n'en est pas un : *, @, (n,) et (n, 1) ★★ 20 min
- [ ] 0B.46 📦 Inverse et systèmes : np.linalg.inv et np.linalg.solve ★★ 15 min
- [ ] 0B.47 🔨 Pentes numériques : vérifier tes dérivées à la main ★★ 20 min
- [ ] 0B.48 📈 Lire les variations : f, f′ et les points où f′ s'annule ★★ 15 min
- [ ] 0B.49 📈 Carte de lignes de niveau et flèches du gradient ★★ 20 min
- [ ] 0B.50 🔮 Contre le gradient, avec lui ou le long d'une ligne de niveau : où va f ? ★★ 15 min
- [ ] 0B.51 🔨 Dérivées partielles numériques et somme sur les chemins ★★ 25 min
- [ ] 0B.52 🔬 Simuler des dés : fréquences, indépendance, loi des grands nombres ★★ 25 min
- [ ] 0B.53 🔨 Espérance et variance : le calcul exact contre la simulation ★★ 20 min
- [ ] 0B.54 🏆 L'ordre des produits : calculer A·B·C·v des dizaines de fois plus vite ★★★ 40 min
- [ ] 0B.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (30 cartes)
<!-- wb:end 0B -->

<!-- wb:section 1 -->
## 1 — Introduction au machine learning et au deep learning ⏱️ 14 h

- [ ] 1.Q1–Q11 🧠 Quiz (11 questions, 36 min)
- [ ] 1.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 1.1 ✏️ Accuracy et erreurs à l'échelle d'un centre de tri ★ 10 min
- [ ] 1.2 ✏️ Concerts : la valeur manquante et celle de demain ★ 15 min
- [ ] 1.3 ✏️ Compter les connexions d'un réseau en couches ★ 10 min
- [ ] 1.4 ✏️ Moins de nombres pour dire la même chose ★★ 15 min
- [ ] 1.5 🧮 Fermi : combien coûte l'étiquetage de MNIST ? ★★ 15 min
- [ ] 1.6 🗣️ Le machine learning en cinq lignes ★ 10 min
- [ ] 1.7 ⚖️ Reconnaissance faciale : utile, risquée, encadrée ★★ 20 min
- [ ] 1.8 📄 Galton (1886) : l'origine du mot « régression » ★★ 25 min
- [ ] 1.9 📦 Penguins : échantillons, features et labels ★ 10 min
- [ ] 1.10 📦 MNIST : une image, 784 nombres ★ 10 min
- [ ] 1.11 📦 Holmes et Verne : le texte devient des nombres ★★ 15 min
- [ ] 1.12 📦 Taches solaires : tracer, lisser, repérer le cycle ★★ 20 min
- [ ] 1.13 🛠️ Lire les data cards des quatre fils rouges ★★ 15 min
- [ ] 1.14 🔮 Mémoriser n'est pas apprendre ★★ 15 min
- [ ] 1.15 🔨 Un système expert pour les manchots ★★ 20 min
- [ ] 1.16 🔨 La boucle d'entraînement à la main ★★ 30 min
- [ ] 1.17 🔬 Learning rate : trop prudent, trop pressé ★★ 20 min
- [ ] 1.18 📦 Un arbre de décision apprend les règles à ta place ★★ 20 min
- [ ] 1.19 🔮 Un manchot d'une espèce jamais vue ★★ 15 min
- [ ] 1.20 🐛 Le score trop beau pour être vrai ★★ 20 min
- [ ] 1.21 📦 Regrouper les manchots sans leurs labels ★★ 20 min
- [ ] 1.22 🔬 L'agent cuisinier : apprendre par la récompense ★★ 30 min
- [ ] 1.23 📦 Un réseau de neurones en boîte noire sur MNIST ★★ 25 min
- [ ] 1.24 🔨 Fabriquer du faux Holmes et du faux Verne ★★★ 35 min
- [ ] 1.25 🏆 Battre l'expert : 95 % avec tes propres règles ★★★ 40 min
- [ ] 1.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (20 cartes)
<!-- wb:end 1 -->

<!-- wb:section 2 -->
## 2 — Hasard et statistiques de base ⏱️ 19 h

- [ ] 2.Q1–Q12 🧠 Quiz (12 questions, 37 min)
- [ ] 2.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 2.1 ✏️ Moyenne, médiane et mode d'une liste de salaires ★ 10 min
- [ ] 2.2 ✏️ De la casse de voitures à la distribution de probabilité ★ 10 min
- [ ] 2.3 ✏️ La règle 68-95-99,7 sur les nageoires des manchots ★ 10 min
- [ ] 2.4 ✏️ Variance : diviser par N ou par N − 1 ? ★★ 15 min
- [ ] 2.5 ✏️ Espérances : Bernoulli, multinoulli et jeu de hasard ★★ 15 min
- [ ] 2.6 ✏️ Compter les tirages avec et sans remise ★★ 20 min
- [ ] 2.7 ✏️ Covariance et corrélation de cinq points à la main ★★ 20 min
- [ ] 2.8 ∂ Changer d'unité : la covariance bouge, pas la corrélation ★★ 25 min
- [ ] 2.9 🗣️ Le bootstrap en cinq lignes ★ 10 min
- [ ] 2.10 ⚖️ Corrélation, causalité et échantillon biaisé ★★ 20 min
- [ ] 2.11 🧮 Fermi : la taille de l'espace des images ★★ 15 min
- [ ] 2.12 📄 Anscombe (1973) : regarder avant de calculer ★★ 25 min
- [ ] 2.13 🔨 Tendances centrales : mean, median, mode ★★ 30 min
- [ ] 2.14 🔮 Graine fixée ou graine libre ? ★ 10 min
- [ ] 2.15 🔨 Dispersion : variance, std, percentile, zscore ★★★ 40 min
- [ ] 2.16 🔨 Un histogramme fait maison ★★ 20 min
- [ ] 2.17 🎨 Galerie des lois usuelles ★★ 20 min
- [ ] 2.18 🔬 68-95-99,7 : la théorie face aux tirages et aux manchots ★★ 20 min
- [ ] 2.19 🔨 La roue de la fortune : tirer dans une distribution discrète ★★ 25 min
- [ ] 2.20 🔮 Le pelage des animaux : une variable qui dépend d'une autre ★★ 15 min
- [ ] 2.21 🔨 Tirer avec ou sans remise ★★ 20 min
- [ ] 2.22 🔨 Bootstrap : distribution et intervalle de confiance ★★ 30 min
- [ ] 2.23 🔬 Bootstraps de 20 (livre) ou de n (aujourd'hui) ? ★★ 30 min
- [ ] 2.24 📦 Comparer avec scipy.stats.bootstrap ★★ 15 min
- [ ] 2.25 📦 Distances entre chiffres dans l'espace à 784 dimensions ★★ 25 min
- [ ] 2.26 🔨 Covariance et corrélation ★★ 25 min
- [ ] 2.27 📈 Deviner la corrélation d'un nuage de points ★★ 20 min
- [ ] 2.28 🔨 Matrices de covariance et de corrélation des manchots ★★ 25 min
- [ ] 2.29 🐛 Le piège de ddof : NumPy, pandas et toi ★★ 20 min
- [ ] 2.30 🎨 Le quartet d'Anscombe ★★ 25 min
- [ ] 2.31 🛠️ Docstring et test pytest pour zscore ★★ 30 min
- [ ] 2.32 🏆 Mêmes statistiques, autre dessin : fabrique ton quartet ★★★ 60 min
- [ ] 2.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (28 cartes)
<!-- wb:end 2 -->

<!-- wb:section 3 -->
## 3 — Probabilités et mesure de la qualité ⏱️ 19 h

- [ ] 3.Q1–Q12 🧠 Quiz (12 questions, 38 min)
- [ ] 3.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 3.1 ✏️ Fléchettes et aires : probabilités simples et conditionnelles ★ 15 min
- [ ] 3.2 ✏️ Les 20 points : matrice de confusion et quatre mesures ★ 15 min
- [ ] 3.3 ✏️ Le glacier : jointes, marginales et conditionnelles ★★ 20 min
- [ ] 3.4 ∂ Règle du produit et formule des probabilités totales ★★ 20 min
- [ ] 3.5 ✏️ Toutes les mesures du tableau récapitulatif ★★ 20 min
- [ ] 3.6 ✏️ Trois espèces : moyennes macro, micro et pondérée ★★ 25 min
- [ ] 3.7 ✏️ Le test « fiable à 99 % » dans une ville à 1 % de malades ★★ 25 min
- [ ] 3.8 ∂ F1, moyenne harmonique : pourquoi elle punit le maillon faible ★★ 25 min
- [ ] 3.9 🗣️ Precision et recall expliqués à une médecin ★ 10 min
- [ ] 3.10 ⚖️ Dépistage de masse : que dire à une personne testée positive ? ★★ 20 min
- [ ] 3.11 📄 Fawcett (2006) : une introduction à l'analyse ROC ★★ 30 min
- [ ] 3.12 🔬 Dix mille fléchettes : estimer des aires (et π) ★ 15 min
- [ ] 3.13 🔮 Deux disques : P(A|B) = P(B|A) ? ★ 10 min
- [ ] 3.14 📦 Penguins : espèce × île avec pd.crosstab ★★ 20 min
- [ ] 3.15 🔨 confusion_matrix à la manière de scikit-learn ★★ 20 min
- [ ] 3.16 🔨 accuracy, precision, recall, F-beta et F1 (cas binaire) ★★★ 40 min
- [ ] 3.17 🐛 La matrice à l'envers ★★ 15 min
- [ ] 3.18 🔮 Tout positif, un seul positif : prédire les scores ★★ 15 min
- [ ] 3.19 🔨 Le tableau de bord complet : classification_rates ★★ 20 min
- [ ] 3.20 🔬 Un seuil sur la nageoire : precision et recall en balance ★★ 25 min
- [ ] 3.21 🔬 Simuler le dépistage : la prévalence fait la precision ★★ 30 min
- [ ] 3.22 📦 Vérifier avec scikit-learn : classification_report et affichages ★★ 20 min
- [ ] 3.23 🛠️ Lire la documentation de sklearn.metrics ★★ 25 min
- [ ] 3.24 🔨 Courbe ROC et AUC ★★★ 40 min
- [ ] 3.25 🔨 Moyennes macro, micro et pondérée ★★★ 35 min
- [ ] 3.26 🔨 Courbe precision-recall et average precision ★★★ 40 min
- [ ] 3.27 📈 ROC ou PR ? Lire les courbes d'un problème déséquilibré ★★ 25 min
- [ ] 3.28 🔨 Calibration : quand la météo annonce 70 % ★★★ 35 min
- [ ] 3.29 🏆 Recall ≥ 0,99 au meilleur prix ★★★ 45 min
- [ ] 3.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (30 cartes)
<!-- wb:end 3 -->

<!-- wb:section 4 -->
## 4 — Règle de Bayes ⏱️ 16 h

- [ ] 4.Q1–Q10 🧠 Quiz (10 questions, 30 min)
- [ ] 4.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 4.1 ✏️ Une face : la pièce est-elle équilibrée ? ★ 10 min
- [ ] 4.2 ✏️ Une pile : le verdict s'inverse ★ 10 min
- [ ] 4.3 ∂ Retrouver la règle de Bayes en trois lignes ★★ 15 min
- [ ] 4.4 ✏️ Vie extraterrestre : lire la sonde avec Bayes ★★ 20 min
- [ ] 4.5 ✏️ Deux faces : une mise à jour double ou deux simples ? ★★ 20 min
- [ ] 4.6 ✏️ Cinq hypothèses de biais après face, pile, face ★★ 20 min
- [ ] 4.7 ∂ Le posterior reste une distribution, un prior nul reste nul ★★ 20 min
- [ ] 4.8 ✏️ Combien de sondes pour descendre sous un sur un million ? ★★ 25 min
- [ ] 4.9 🗣️ La règle de Bayes sans formule, en cinq lignes ★ 10 min
- [ ] 4.10 ⚖️ Le prior est un choix : erreur du procureur et priors partiaux ★★ 20 min
- [ ] 4.11 📄 VanderPlas (2014) : fréquentisme et bayésianisme ★★ 30 min
- [ ] 4.12 🔮 Combien de lancers pour démasquer la pièce truquée ? ★ 10 min
- [ ] 4.13 🔬 L'estimation fréquentiste : la moyenne courante des faces ★ 15 min
- [ ] 4.14 🔨 evidence et bayes_posterior ★★ 20 min
- [ ] 4.15 📦 Bayes chez les manchots : l'espèce sachant l'île ★★ 20 min
- [ ] 4.16 🔨 La boucle posterior-prior : update_discrete ★★ 30 min
- [ ] 4.17 🎨 Reproduire les trente lancers de la figure 4.24 ★★ 25 min
- [ ] 4.18 🐛 Le posterior qui s'évanouit : underflow ★★ 20 min
- [ ] 4.19 🔬 La grille biais × proportion de faces ★★ 30 min
- [ ] 4.20 🔬 Envoyer des sondes jusqu'à la décision ★★ 25 min
- [ ] 4.21 🔮 Un prior trompeur centré sur 0,8 ★★ 20 min
- [ ] 4.22 📦 Le posterior continu : vérifier avec scipy.stats.beta ★★ 20 min
- [ ] 4.23 🛠️ Refactoriser : du copier-coller à une fonction testée ★★ 20 min
- [ ] 4.24 🔨 coin_bias_posterior : 500 hypothèses en log-probabilités ★★★ 35 min
- [ ] 4.25 🔨 Intervalle de crédibilité contre intervalle bootstrap ★★★ 35 min
- [ ] 4.26 🏆 Le détective de pièces : vingt pièces, le moins de lancers possible ★★★ 60 min
- [ ] 4.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (22 cartes)
<!-- wb:end 4 -->

<!-- wb:section 5 -->
## 5 — Courbes et surfaces ⏱️ 14 h

- [ ] 5.Q1–Q10 🧠 Quiz (10 questions, 30 min)
- [ ] 5.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 5.1 ✏️ La sécante qui se resserre sur la tangente ★ 10 min
- [ ] 5.2 ✏️ Gradient à la main et direction de plus grande pente ★ 15 min
- [ ] 5.3 ✏️ Trois pas de descente de gradient à la main ★★ 15 min
- [ ] 5.4 ✏️ Tableau de variations : extrema locaux et globaux de x³ − 3x ★★ 20 min
- [ ] 5.5 ✏️ Point selle : x² − y² vu dans deux directions ★★ 20 min
- [ ] 5.6 ∂ Pourquoi la différence centrée est plus précise (calcul exact sur x³) ★★ 25 min
- [ ] 5.7 ∂ Rosenbrock : gradient et minimum à la main ★★★ 30 min
- [ ] 5.8 🗣️ Le gradient expliqué avec de l'eau sur un drap ★ 10 min
- [ ] 5.9 🧮 Fermi : le prix d'un gradient numérique pour un million de paramètres ★★ 15 min
- [ ] 5.10 📄 Dauphin et coll. (2014) : les points selles en grande dimension ★★ 30 min
- [ ] 5.11 🔨 Dérivées numériques : première et seconde ★★ 25 min
- [ ] 5.12 🔮 Quel pas h choisir ? Prédire la courbe d'erreur ★★ 15 min
- [ ] 5.13 🔬 Erreur de troncature contre erreur d'arrondi ★★ 25 min
- [ ] 5.14 🔨 Les maxima des cycles solaires ★★ 30 min
- [ ] 5.15 🔨 numerical_gradient sur la vallée de Rosenbrock ★★ 25 min
- [ ] 5.16 🐛 Le gradient qui abîme son entrée ★★ 20 min
- [ ] 5.17 📈 Lire des lignes de niveau : où pointe le gradient ? ★★ 20 min
- [ ] 5.18 🔨 gradient_descent, et sa version qui monte ★★ 30 min
- [ ] 5.19 🔬 Learning rate sur un bol : trop petit, juste, trop grand ★★ 25 min
- [ ] 5.20 🔮 Démarrer pile sur un point selle ★★ 15 min
- [ ] 5.21 📦 Le même gradient avec torch.autograd ★★ 20 min
- [ ] 5.22 🎨 L'eau qui descend la surface (figure 5.18) ★★ 30 min
- [ ] 5.23 🛠️ Tests de propriétés paramétrés avec pytest ★★ 20 min
- [ ] 5.24 🔨 Minimum, maximum, selle ou plat : classify_critical_point ★★★ 35 min
- [ ] 5.25 🏆 Atteindre le fond de la vallée de Rosenbrock ★★★ 60 min
- [ ] 5.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (20 cartes)
<!-- wb:end 5 -->

<!-- wb:section 6 -->
## 6 — Théorie de l'information ⏱️ 17 h

- [ ] 6.Q1–Q12 🧠 Quiz (12 questions, 37 min)
- [ ] 6.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 6.1 ✏️ Combien de bits pour une pièce, un dé, une lettre E ? ★ 10 min
- [ ] 6.2 ✏️ Bits par mot : Seuss, Holmes et l'alphabet ★ 10 min
- [ ] 6.3 ✏️ Entropie de quelques distributions ★ 15 min
- [ ] 6.4 ✏️ Morse contre code fixe : SHERLOCK HOLMES ★★ 20 min
- [ ] 6.5 ✏️ Cross-entropy et KL dans les deux sens ★★ 20 min
- [ ] 6.6 ✏️ Un code de Huffman à la main ★★ 25 min
- [ ] 6.7 ∂ H(p, q) = H(p) + KL(p‖q), et KL(p‖p) = 0 ★★ 20 min
- [ ] 6.8 ∂ L'entropie d'une pièce est maximale à p = 1/2 ★★★ 30 min
- [ ] 6.9 🗣️ L'entropie expliquée avec un jeu de devinettes ★ 10 min
- [ ] 6.10 🧮 Fermi : combien de bits pour envoyer tout Holmes ? ★★ 20 min
- [ ] 6.11 📄 Shannon (1948) : l'introduction et le schéma de communication ★★ 30 min
- [ ] 6.12 🔨 self_information et entropy ★ 15 min
- [ ] 6.13 🔨 Distributions de caractères et de mots ★★ 20 min
- [ ] 6.14 🔮 Qui a l'entropie par lettre la plus haute : Holmes ou Verne ? ★★ 15 min
- [ ] 6.15 🎨 Fréquences des lettres en anglais et en français ★★ 20 min
- [ ] 6.16 🔨 cross_entropy, kl_divergence et js_divergence ★★ 25 min
- [ ] 6.17 🐛 La cross-entropy infinie : la lettre qui manque ★★ 20 min
- [ ] 6.18 🔬 Coder le français avec le code de l'anglais, et l'inverse ★★ 25 min
- [ ] 6.19 📦 scipy.stats.entropy et un vrai compresseur (zlib) ★★ 20 min
- [ ] 6.20 📈 Lire une courbe de loss : nats, bits et perplexité ★★ 20 min
- [ ] 6.21 🛠️ Mesurer avant d'optimiser : compter des caractères vite ★★ 20 min
- [ ] 6.22 🔨 perplexity et log_loss ★★ 30 min
- [ ] 6.23 🔨 Huffman : construire, encoder, décoder ★★★ 70 min
- [ ] 6.24 🔬 Compresser Holmes : code fixe, Morse, Huffman et entropie ★★★ 30 min
- [ ] 6.25 🔮 Le code de Huffman de Holmes pour envoyer Verne ★★★ 30 min
- [ ] 6.26 🔬 Le contexte local réduit la surprise : les bigrammes ★★★ 40 min
- [ ] 6.27 🏆 Passer sous la barre de Huffman lettre à lettre ★★★ 60 min
- [ ] 6.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 6 -->

<!-- wb:section CP1 -->
## CP1 — Checkpoint I — Fondations ⏱️ 11 h 30

- [ ] CP1.1 🧠 Questions flash sur toute la partie (vrai ou faux, justifié) ★ 10 min
- [ ] CP1.2 ✏️ Partie 0 : norme, produit scalaire et log₂ ★ 5 min
- [ ] CP1.3 🐛 Partie 0 : la moyenne qui oublie axis ★ 5 min
- [ ] CP1.4 ✏️ Statistiques de cinq points à la main ★★ 10 min
- [ ] CP1.5 🔨 Coder un intervalle de confiance bootstrap ★★ 10 min
- [ ] CP1.6 ✏️ Dépistage : matrice de confusion, precision, NPV et règle de Bayes ★★ 12 min
- [ ] CP1.7 📈 Lire une courbe ROC et une courbe PR déséquilibrées ★★ 8 min
- [ ] CP1.8 ✏️ Trois hypothèses, deux lancers ★★ 10 min
- [ ] CP1.9 ∂ Gradient, point selle et deux pas de descente ★★ 12 min
- [ ] CP1.10 🔮 Prédire l'effet du learning rate ★ 5 min
- [ ] CP1.11 ✏️ Entropie, cross-entropy, KL et Huffman ★★ 12 min
- [ ] CP1.12 🗣️ Un LLM expliqué en cinq lignes : données, loss, perplexité ★ 5 min
- [ ] CP1.13 ⚖️ Corrélation, causalité et échantillon : juger une affirmation ★ 5 min
- [ ] CP1.14 💼 Entretien express : 98 % d'accuracy sur la fraude ★★ 8 min
- [ ] Mini-projet MP1 — Détecteur de langue anglais / français from scratch (≈ 8 h)
<!-- wb:end CP1 -->

<!-- wb:section 7 -->
## 7 — Classification ⏱️ 21 h

- [ ] 7.Q1–Q11 🧠 Quiz (11 questions, 34 min)
- [ ] 7.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 7.1 ✏️ Compter les classifieurs OvR et OvO ★ 10 min
- [ ] 7.2 ✏️ Dépouiller les votes d'un un-contre-un à quatre classes ★★ 15 min
- [ ] 7.3 ✏️ Une itération de k-means à la main ★★ 20 min
- [ ] 7.4 ✏️ Densité d'échantillons et nombre d'œufs nécessaires ★★ 15 min
- [ ] 7.5 ∂ Le rayon de l'hyper-orange : r(d) = √d − 1 ★★ 25 min
- [ ] 7.6 ∂ Boule dans un cube : rapport des volumes par récurrence ★★★ 35 min
- [ ] 7.7 ∂ La frontière du centroïde le plus proche est une droite ★★★ 30 min
- [ ] 7.8 🗣️ La malédiction de la dimension en cinq lignes ★ 10 min
- [ ] 7.9 🛠️ Lire la documentation officielle de KMeans (scikit-learn) ★ 15 min
- [ ] 7.10 ⚖️ Qui fixe le seuil ? Œufs, dépistage et coût des erreurs ★★ 20 min
- [ ] 7.11 📦 Des œufs en 2D : données, régions et frontière de décision ★ 15 min
- [ ] 7.12 🔮 k-means sur deux lunes : où tombera la coupure ? ★ 10 min
- [ ] 7.13 🔨 Distances au carré vectorisées : pairwise_sq_distances ★★ 20 min
- [ ] 7.14 🔨 Le classifieur du centroïde le plus proche ★★ 30 min
- [ ] 7.15 📈 Carte de probabilité et politique de seuil pour les œufs ★★ 25 min
- [ ] 7.16 🔮 Un-contre-tous avec des centroïdes : quelle classe sera sacrifiée ? ★★ 15 min
- [ ] 7.17 📦 Manchots sans labels : k-means face aux espèces ★★ 25 min
- [ ] 7.18 📦 Formes arbitraires et bruit : DBSCAN et HDBSCAN ★★ 25 min
- [ ] 7.19 🔮 Distance au plus proche voisin quand la dimension grimpe ★★ 15 min
- [ ] 7.20 🔬 Densité, plus proche voisin et concentration des distances ★★ 30 min
- [ ] 7.21 🎨 Reproduire les figures 7.27 et 7.30 (boule/cube, hyper-orange) ★★ 25 min
- [ ] 7.22 🔨 Un-contre-tous générique : OneVsRestClassifier ★★★ 40 min
- [ ] 7.23 🔨 Un-contre-un générique : OneVsOneClassifier ★★★ 45 min
- [ ] 7.24 🔬 OvR, OvO ou multi-classe natif : accuracy, nombre de modèles, temps ★★★ 35 min
- [ ] 7.25 🔨 Initialisation k-means++ ★★★ 35 min
- [ ] 7.26 🔨 k-means de Lloyd : la classe KMeans ★★★★ 120 min
- [ ] 7.27 🐛 k-means piégé : quatre bugs à débusquer ★★★ 30 min
- [ ] 7.28 🔨 Coefficient de silhouette ★★★ 40 min
- [ ] 7.29 🔬 Choisir k : coude de l'inertie et silhouette, de k = 2 à 7 ★★★ 35 min
- [ ] 7.30 🔬 Phénomène de Hughes : des features de bruit qui font chuter l'accuracy ★★★ 40 min
- [ ] 7.31 🏆 Défi : retrouver les espèces de manchots sans labels ★★★ 60 min
- [ ] 7.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 7 -->

<!-- wb:section 8 -->
## 8 — Entraînement et test ⏱️ 17 h

- [ ] 8.Q1–Q11 🧠 Quiz (11 questions, 34 min)
- [ ] 8.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 8.1 ✏️ Découper 344 manchots : hold-out, validation et folds ★ 15 min
- [ ] 8.2 ✏️ Compter les entraînements d'une recherche d'hyperparamètres ★ 10 min
- [ ] 8.3 ✏️ Fuite ou pas ? Six protocoles à auditer ★★ 20 min
- [ ] 8.4 ✏️ Quelle confiance accorder à une accuracy de test ? Erreur type et taille du test ★★ 20 min
- [ ] 8.5 ✏️ Moyenne et écart-type de scores de validation croisée ★★ 15 min
- [ ] 8.6 ∂ Le biais d'optimisme du meilleur de K modèles ★★★ 30 min
- [ ] 8.7 🗣️ Pourquoi le jeu de test reste sous clé : l'analogie de l'examen ★ 10 min
- [ ] 8.8 📈 Comparer des modèles à partir de boîtes à moustaches de scores ★★ 15 min
- [ ] 8.9 ⚖️ Raccourcis appris : radiographies, chars d'assaut et responsabilité ★★ 25 min
- [ ] 8.10 📄 Kapoor & Narayanan (2023) : une taxonomie des fuites ★★ 30 min
- [ ] 8.11 📦 train_test_split de scikit-learn : tailles, stratify, random_state ★ 10 min
- [ ] 8.12 🔮 Le modèle qui apprend par cœur : accuracy d'entraînement et de test ★ 15 min
- [ ] 8.13 🔨 train_test_split from scratch ★★ 30 min
- [ ] 8.14 🔨 Les indices de la k-fold : kfold_indices ★★ 25 min
- [ ] 8.15 🎨 Reproduire la figure 8.13 : la rotation des folds ★★ 20 min
- [ ] 8.16 🔨 Un estimateur maison à la scikit-learn : PolyFit(degree) ★★ 25 min
- [ ] 8.17 🔮 Validation ou test : lequel sera le plus optimiste ? ★★ 15 min
- [ ] 8.18 🔨 Boucle de sélection sur un jeu de validation : le degré du polynôme ★★ 30 min
- [ ] 8.19 📦 Données dépendantes : GroupKFold et TimeSeriesSplit ★★ 25 min
- [ ] 8.20 🛠️ Écrire tes propres tests pytest pour train_test_split ★★ 25 min
- [ ] 8.21 🔨 k-fold stratifiée : stratified_kfold_indices ★★★ 40 min
- [ ] 8.22 🔨 clone et cross_val_score ★★★ 45 min
- [ ] 8.23 🔬 Variabilité de l'évaluation : hold-out répétés contre k-fold ★★★ 40 min
- [ ] 8.24 🐛 Un notebook trop beau pour être vrai : quatre fuites à corriger ★★★ 35 min
- [ ] 8.25 🔬 Sélectionner des features avant la validation croisée : 90 % sur du bruit ★★★ 40 min
- [ ] 8.26 🔬 Comparer deux modèles honnêtement : test par permutation, p-valeur et bootstrap apparié ★★★ 40 min
- [ ] 8.27 🏆 Défi : la meilleure paire de features, choisie sans toucher au test ★★★ 60 min
- [ ] 8.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (22 cartes)
<!-- wb:end 8 -->

<!-- wb:section 9 -->
## 9 — Overfitting et underfitting ⏱️ 22 h

- [ ] 9.Q1–Q11 🧠 Quiz (11 questions, 34 min)
- [ ] 9.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 9.1 ✏️ MSE et R² à la main sur cinq points ★ 10 min
- [ ] 9.2 ∂ Moindres carrés : la meilleure droite par dérivées partielles ★★ 25 min
- [ ] 9.3 ∂ Ridge en dimension 1 : w* = Σxy / (Σx² + λ) ★★ 20 min
- [ ] 9.4 ✏️ Biais² et variance à partir d'un tableau de prédictions ★★ 20 min
- [ ] 9.5 ✏️ Early stopping avec patience sur une courbe de loss ★★ 15 min
- [ ] 9.6 ∂ Lasso en dimension 1 : le seuillage doux et les zéros exacts ★★★ 35 min
- [ ] 9.7 ✏️ Mise à jour bayésienne d'une droite sur une grille 3 × 3 ★★★ 30 min
- [ ] 9.8 🗣️ Le compromis biais-variance raconté avec le tempo de la boutique ★ 10 min
- [ ] 9.9 📈 Diagnostiquer quatre paires de courbes d'entraînement et de validation ★★ 20 min
- [ ] 9.10 ⚖️ Écarter un point aberrant : nettoyage ou manipulation ? ★★ 20 min
- [ ] 9.11 📄 Belkin et coll. (2019) : la double descente ★★ 30 min
- [ ] 9.12 📦 Le tempo de la boutique : polynômes de degré 1, 4 et 15 ★ 15 min
- [ ] 9.13 🔮 Erreurs d'entraînement et de test selon le degré : ta courbe d'abord ★ 15 min
- [ ] 9.14 🔨 mean_squared_error, mean_absolute_error et r2_score ★★ 20 min
- [ ] 9.15 🔨 polynomial_features, interactions comprises ★★ 25 min
- [ ] 9.16 🔨 LinearRegression par moindres carrés ★★ 30 min
- [ ] 9.17 🔨 Ridge en forme fermée, intercept non pénalisé ★★ 30 min
- [ ] 9.18 🔬 Courbes de validation : le degré, puis λ ★★ 30 min
- [ ] 9.19 🔮 Que deviennent les coefficients quand λ grandit ? ★★ 15 min
- [ ] 9.20 🔨 Early stopping d'une descente de gradient sur un polynôme de degré 12 ★★ 30 min
- [ ] 9.21 📦 Courbes d'apprentissage sur California avec learning_curve ★★ 30 min
- [ ] 9.22 📦 Ridge contre Lasso sur California : chemins de régularisation ★★ 30 min
- [ ] 9.23 🔨 Lasso par descente de coordonnées et soft_threshold ★★★ 90 min
- [ ] 9.24 🔨 Biais et variance mesurés : 50 sous-échantillons de 30 points ★★★ 45 min
- [ ] 9.25 🎨 Reproduire les figures 9.13 et 9.15, puis la courbe en U ★★★ 40 min
- [ ] 9.26 🔨 Le posterior des droites sur une grille pente-ordonnée ★★★ 45 min
- [ ] 9.27 🎨 Reproduire la figure 9.20 : prior, vraisemblances, posteriors et droites tirées ★★★ 40 min
- [ ] 9.28 🐛 Régularisation piégée : quatre erreurs qui faussent Ridge ★★★ 30 min
- [ ] 9.29 🛠️ Refactoriser l'expérience biais-variance en fonction testée ★★★ 30 min
- [ ] 9.30 🔬 Double descente avec des features aléatoires ★★★ 45 min
- [ ] 9.31 🏆 Défi California : le meilleur modèle linéaire régularisé ★★★ 90 min
- [ ] 9.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 9 -->

<!-- wb:section 10 -->
## 10 — Neurones ⏱️ 14 h

- [ ] 10.Q1–Q9 🧠 Quiz (9 questions, 28 min)
- [ ] 10.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 10.1 ✏️ Sortie d'un perceptron à quatre entrées, avec et sans biais ★ 10 min
- [ ] 10.2 ✏️ L'astuce du biais : même neurone, une entrée de plus ★ 10 min
- [ ] 10.3 ✏️ Portes logiques à la main : AND, OR, NOT, NAND ★★ 20 min
- [ ] 10.4 ∂ Pourquoi un seul perceptron ne peut pas calculer XOR ★★ 25 min
- [ ] 10.5 ✏️ XOR en deux couches : câbler et nommer les poids ★★ 25 min
- [ ] 10.6 ✏️ Une epoch de la règle du perceptron à la main ★★ 25 min
- [ ] 10.7 ∂ Le théorème de convergence du perceptron, guidé pas à pas ★★★ 45 min
- [ ] 10.8 🗣️ Pourquoi un neurone artificiel n'est pas un neurone ★ 10 min
- [ ] 10.9 🧮 Fermi : cerveau humain contre grands modèles ★★ 20 min
- [ ] 10.10 ⚖️ « Cerveaux électroniques » : hype, hivers de l'IA et responsabilité ★★ 20 min
- [ ] 10.11 📄 Rosenblatt (1958) : le perceptron dans le texte ★★ 30 min
- [ ] 10.12 🔨 sign_step et add_bias_column ★ 10 min
- [ ] 10.13 🔮 AND, OR, XOR : le perceptron va-t-il converger ? ★ 10 min
- [ ] 10.14 🔨 neuron_forward : un neurone appliqué à tout un batch ★★ 15 min
- [ ] 10.15 🔨 Des noms de poids (AD, BE…) à la matrice W ★★ 20 min
- [ ] 10.16 🔨 XOR avec trois neurones câblés à la main ★★ 25 min
- [ ] 10.17 🔮 Le learning rate change-t-il un perceptron qui part de zéro ? ★★ 15 min
- [ ] 10.18 📦 Le perceptron de scikit-learn sur portes logiques et manchots ★★ 20 min
- [ ] 10.19 📈 Erreurs par epoch : séparable ou pas ? ★★ 20 min
- [ ] 10.20 🛠️ Docstring NumPy et doctest pour neuron_forward ★★ 20 min
- [ ] 10.21 🔨 La classe Perceptron et sa règle d'apprentissage ★★★ 45 min
- [ ] 10.22 🐛 Perceptron piégé : quatre bugs classiques ★★★ 30 min
- [ ] 10.23 🔬 Marge et vitesse de convergence : la borne (R/γ)² à l'épreuve ★★★ 40 min
- [ ] 10.24 🔬 Trois espèces de manchots avec des perceptrons en un-contre-tous ★★★ 35 min
- [ ] 10.25 🏆 Défi « Mark I » : un perceptron sur des chiffres de 20 × 20 pixels ★★★ 60 min
- [ ] 10.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (18 cartes)
<!-- wb:end 10 -->

<!-- wb:section 11 -->
## 11 — Apprentissage et raisonnement ⏱️ 18 h

- [ ] 11.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] 11.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 11.1 ✏️ Représentable sur n bits : compter, puis conclure ★ 10 min
- [ ] 11.2 ∂ Moyenne incrémentale : Qₙ₊₁ = Qₙ + (Rₙ − Qₙ)/n ★ 15 min
- [ ] 11.3 ✏️ Syllogismes : valides ? solides ? ★★ 20 min
- [ ] 11.4 ✏️ Six raisonnements fautifs à diagnostiquer et à réfuter ★★ 25 min
- [ ] 11.5 ✏️ Enquête au phare : réduire le domaine du discours ★★ 20 min
- [ ] 11.6 ✏️ Syllogisme statistique et prédiction : 15 % de pommes mûres ★★ 20 min
- [ ] 11.7 ✏️ Renforcement ou punition, positif ou négatif : classer huit situations ★★ 15 min
- [ ] 11.8 ✏️ Un bandit à la main : ε-greedy, moyennes et regret ★★ 25 min
- [ ] 11.9 🗣️ Déduction et induction dans un projet de ML, en cinq lignes ★ 10 min
- [ ] 11.10 📈 Lire les courbes d'un bandit : ε = 0, 0,01 et 0,1 ★★ 15 min
- [ ] 11.11 ⚖️ Explorer sur des humains : essais adaptatifs, recommandation, A/B tests ★★ 25 min
- [ ] 11.12 📄 Domingos (2012) : représentation, évaluation, optimisation et autres leçons ★★ 30 min
- [ ] 11.13 🔮 Glouton pur sur trois bras : que va-t-il se passer ? ★ 10 min
- [ ] 11.14 🔨 Holmes déduit-il ? Compter et citer le vocabulaire du raisonnement ★★ 25 min
- [ ] 11.15 🔨 Valider un syllogisme par force brute : 256 mondes de Venn ★★ 30 min
- [ ] 11.16 🎨 Reproduire la figure 11.5 : les cinq sophismes en diagrammes ★★ 25 min
- [ ] 11.17 🔬 Généralisation hâtive et échantillon biaisé chez les manchots ★★ 25 min
- [ ] 11.18 🔬 Des points sur un cercle : quand le modèle trahit l'induction ★★ 25 min
- [ ] 11.19 🔨 BernoulliBandit et GaussianBandit ★★ 20 min
- [ ] 11.20 🔨 argmax_random_tie, epsilon_greedy_action et incremental_update ★★ 25 min
- [ ] 11.21 🔨 run_bandit : la boucle d'interaction et ses courbes ★★★ 40 min
- [ ] 11.22 🔮 Initialisation optimiste sans ε : prédire, puis mesurer ★★★ 30 min
- [ ] 11.23 🔨 ucb_action et thompson_action ★★★ 40 min
- [ ] 11.24 🐛 Bandit piégé : l'agent qui n'explore jamais ★★★ 30 min
- [ ] 11.25 🛠️ Un journal d'expériences reproductible (JSON) ★★★ 30 min
- [ ] 11.26 🔬 Tournoi : ε-greedy, optimiste, UCB et Thompson ★★★ 45 min
- [ ] 11.27 🏆 Défi : battre UCB1 sur un banc de bandits de Bernoulli ★★★ 60 min
- [ ] 11.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 11 -->

<!-- wb:section CP2 -->
## CP2 — Checkpoint II — Concepts ⏱️ 13 h 30

- [ ] CP2.1 🧠 Vrai ou faux justifiés : huit affirmations sur la partie II ★ 10 min
- [ ] CP2.2 ✏️ Un-contre-tous, un-contre-un : compter les modèles et dépouiller un vote ★ 5 min
- [ ] CP2.3 ✏️ Une itération de k-means et l'inertie obtenue ★★ 15 min
- [ ] CP2.4 ✏️ Densité d'échantillons et hyper-orange en dimension d ★ 6 min
- [ ] CP2.5 ✏️ Plan d'évaluation : tailles des jeux, nombre d'entraînements, fuites ★ 8 min
- [ ] CP2.6 ∂ Ridge en dimension 1 : dériver w* et interpréter λ ★★ 15 min
- [ ] CP2.7 📈 Diagnostiquer trois paires de courbes d'apprentissage ★ 6 min
- [ ] CP2.8 ✏️ Perceptron : une epoch sur NAND, puis pourquoi pas XOR ★★ 15 min
- [ ] CP2.9 ✏️ Syllogismes et sophismes : valide, solide, nommer l'erreur ★ 5 min
- [ ] CP2.10 🐛 La fuite cachée d'une validation croisée ★ 9 min
- [ ] CP2.11 🔨 Coder epsilon_greedy_action et une moyenne incrémentale ★ 9 min
- [ ] CP2.12 💼 Entretien : « comment savez-vous que votre modèle ne surapprend pas ? » ★ 5 min
- [ ] CP2.13 ✏️ Parties antérieures : matrice de confusion, Bayes et entropie ★ 9 min
- [ ] Mini-projet MP2 — Prix des logements californiens : un protocole d'évaluation honnête (≈ 10 h)
<!-- wb:end CP2 -->
