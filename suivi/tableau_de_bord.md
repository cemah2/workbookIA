# Tableau de bord

> **Modèle tenu par Claude : ne coche pas ici.** Ta copie personnelle est `mon_travail/suivi/tableau_de_bord.md`, créée par `python tools/start_chapter.py --init`. À chaque nouveau chapitre, `start_chapter.py` ajoute à la fin de ta copie les sections publiées depuis, sans toucher à tes cases déjà cochées.

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
- [ ] Refaire demo.3 et demo.4 jusqu'à obtenir ✅ (⏱️ 5 min)
- [ ] Installer l'environnement local avec `00_setup/INSTALL_LOCAL.md` (⏱️ 45 min, optionnel)
- [ ] `python 00_setup/check_env.py` tout vert (⏱️ 2 min)
- [ ] `python tools/start_chapter.py --init`, puis implémenter `mean` dans `mon_travail/mylearn/_example.py` jusqu'à 5 tests verts (⏱️ 15 min)
- [ ] Installer Anki (⏱️ 10 min)
<!-- wb:end setup -->

<!-- wb:section 0A -->
## 0A — Python, notebooks et outils ⏱️ 28 h

- [ ] 0A.Q1–Q12 🧠 Quiz (12 questions, 37 min)
- [ ] 0A.1 ✏️ Évaluer des expressions à la main : //, %, **, conversions ★ 10 min
- [ ] 0A.2 ✏️ Indices et tranches à la main : listes, tuples, chaînes ★ 10 min
- [ ] 0A.3 ✏️ Dérouler une boucle et une compréhension pas à pas ★ 15 min
- [ ] 0A.4 ✏️ Un groupby à la main sur huit manchots ★ 10 min
- [ ] 0A.5 ✏️ Mini-lots : combien de lots, de quelle taille, combien de mises à jour ? ★ 10 min
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
- [ ] 0A.53 📦 reshape, transpose et empilement ★★ 20 min
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
- [ ] 0A.65 🔨 utils.one_hot : des étiquettes aux vecteurs ★★ 26 min
- [ ] 0A.66 🔨 utils.iterate_minibatches : découper un dataset en mini-lots ★★★ 39 min
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
- [ ] 0B.33 📦 Calculer avec Python : puissances, arrondis, |x|, signe et C(n, k) ★ 10 min
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
- [ ] 1.5 🧮 Fermi : combien coûtent les étiquettes de MNIST ? ★★ 15 min
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
- [ ] 1.21 📦 Regrouper les manchots sans leurs étiquettes ★★ 20 min
- [ ] 1.22 🔬 L'agent cuisinier : apprendre par la récompense ★★ 30 min
- [ ] 1.23 📦 Un réseau de neurones en boîte noire sur MNIST ★★ 25 min
- [ ] 1.24 🔨 Fabriquer du faux Holmes et du faux Verne ★★★ 35 min
- [ ] 1.25 🏆 Battre l'expert : 95 % avec tes propres règles ★★★ 40 min
- [ ] 1.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (20 cartes)
<!-- wb:end 1 -->

<!-- wb:section 2 -->
## 2 — Hasard et statistiques de base ⏱️ 18 h

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
## 3 — Probabilités et mesure de la qualité ⏱️ 18 h

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
- [ ] 3.16 🔨 accuracy, precision, recall, F-beta et F1 (cas binaire) ★★ 25 min
- [ ] 3.17 🐛 La matrice à l'envers ★★ 15 min
- [ ] 3.18 🔮 Tout positif, un seul positif : prédire les scores ★★ 15 min
- [ ] 3.19 🔨 Le tableau de bord complet : classification_rates ★★ 20 min
- [ ] 3.20 🔬 Un seuil sur la nageoire : precision et recall en balance ★★ 25 min
- [ ] 3.21 🔬 Simuler le dépistage : la prévalence fait la precision ★★ 25 min
- [ ] 3.22 📦 Vérifier avec scikit-learn : classification_report et affichages ★★ 20 min
- [ ] 3.23 🛠️ Lire la documentation de sklearn.metrics ★★ 15 min
- [ ] 3.24 🔨 Courbe ROC et AUC ★★★ 40 min
- [ ] 3.25 🔨 Moyennes macro, micro et pondérée ★★★ 35 min
- [ ] 3.26 🔨 Courbe precision-recall et average precision ★★★ 40 min
- [ ] 3.27 📈 ROC ou PR ? Lire les courbes d'un problème déséquilibré ★★★ 30 min
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
- [ ] 5.10 📄 Dauphin et al. (2014) : les points selles en grande dimension ★★ 30 min
- [ ] 5.11 🔨 Dérivées numériques : première et seconde ★ 15 min
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
- [ ] 5.22 🎨 L'eau qui descend la surface (figure 5.18) ★★ 25 min
- [ ] 5.23 🛠️ Tests de propriétés paramétrés avec pytest ★★ 20 min
- [ ] 5.24 🔨 Minimum, maximum, selle ou plat : classify_critical_point ★★★ 35 min
- [ ] 5.25 🏆 Atteindre le fond de la vallée de Rosenbrock ★★★ 60 min
- [ ] 5.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (20 cartes)
<!-- wb:end 5 -->

<!-- wb:section 6 -->
## 6 — Théorie de l'information ⏱️ 16 h

- [ ] 6.Q1–Q12 🧠 Quiz (12 questions, 37 min)
- [ ] 6.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 6.1 ✏️ Combien de bits pour une pièce, un dé, une lettre E ? ★ 10 min
- [ ] 6.2 ✏️ Bits par mot : Seuss, Stevenson et l'alphabet ★ 10 min
- [ ] 6.3 ✏️ Entropie de trois distributions ★ 15 min
- [ ] 6.4 ✏️ Morse contre code fixe : SQUIRE TRELAWNEY ★★ 20 min
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
- [ ] 6.23 🔨 Huffman : construire, encoder, décoder ★★★ 45 min
- [ ] 6.24 🔬 Compresser Holmes : code fixe, Morse, Huffman et entropie ★★★ 30 min
- [ ] 6.25 🔮 Le code de Huffman de Holmes pour envoyer Verne ★★★ 30 min
- [ ] 6.26 🔬 Le contexte local réduit la surprise : les bigrammes ★★★ 40 min
- [ ] 6.27 🏆 Passer sous la barre de Huffman lettre à lettre ★★★ 60 min
- [ ] 6.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 6 -->

<!-- wb:section CP1 -->
## CP1 — Checkpoint I — Fondations ⏱️ 11 h

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
## 7 — Classification ⏱️ 20 h

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
- [ ] 7.17 📦 Manchots sans étiquettes : k-means face aux espèces ★★ 25 min
- [ ] 7.18 📦 Formes arbitraires et bruit : DBSCAN et HDBSCAN ★★ 25 min
- [ ] 7.19 🔮 Distance au plus proche voisin quand la dimension grimpe ★★ 15 min
- [ ] 7.20 🔬 Densité, plus proche voisin et concentration des distances ★★ 30 min
- [ ] 7.21 🎨 Reproduire les figures 7.27 et 7.30 (boule/cube, hyper-orange) ★★ 25 min
- [ ] 7.22 🔨 Un-contre-tous générique : OneVsRestClassifier ★★★ 40 min
- [ ] 7.23 🔨 Un-contre-un générique : OneVsOneClassifier ★★★ 45 min
- [ ] 7.24 🔬 OvR, OvO ou multi-classe natif : accuracy, nombre de modèles, temps ★★★ 35 min
- [ ] 7.25 🔨 Initialisation k-means++ ★★★ 35 min
- [ ] 7.26 🔨 k-means de Lloyd : la classe KMeans ★★★ 60 min
- [ ] 7.27 🐛 k-means piégé : quatre bugs à débusquer ★★★ 30 min
- [ ] 7.28 🔨 Coefficient de silhouette ★★★ 40 min
- [ ] 7.29 🔬 Choisir k : coude de l'inertie et silhouette, de k = 2 à 7 ★★★ 35 min
- [ ] 7.30 🔬 Phénomène de Hughes : des features de bruit qui font chuter l'accuracy ★★★ 40 min
- [ ] 7.31 🏆 Défi : retrouver les espèces de manchots sans étiquettes ★★★ 60 min
- [ ] 7.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 7 -->

<!-- wb:section 8 -->
## 8 — Entraînement et test ⏱️ 16 h

- [ ] 8.Q1–Q11 🧠 Quiz (11 questions, 34 min)
- [ ] 8.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 8.1 ✏️ Découper 344 manchots : hold-out, validation et folds ★ 10 min
- [ ] 8.2 ✏️ Compter les entraînements d'une recherche d'hyperparamètres ★ 10 min
- [ ] 8.3 ✏️ Fuite ou pas ? Six protocoles à auditer ★★ 20 min
- [ ] 8.4 ✏️ Quelle confiance accorder à une accuracy de test ? Erreur-type et taille du test ★★ 20 min
- [ ] 8.5 ✏️ Moyenne et écart-type de scores de validation croisée ★★ 15 min
- [ ] 8.6 ∂ Le biais d'optimisme du meilleur de K modèles ★★★ 30 min
- [ ] 8.7 🗣️ Pourquoi le jeu de test reste sous clé : l'analogie de l'examen ★ 10 min
- [ ] 8.8 📈 Comparer deux modèles à partir de boîtes à moustaches de scores ★★ 15 min
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
## 9 — Surapprentissage et sous-apprentissage ⏱️ 21 h

- [ ] 9.Q1–Q11 🧠 Quiz (11 questions, 34 min)
- [ ] 9.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 9.1 ✏️ MSE et R² à la main sur cinq points ★ 10 min
- [ ] 9.2 ∂ Moindres carrés : la meilleure droite par dérivées partielles ★★ 25 min
- [ ] 9.3 ∂ Ridge en dimension 1 : w* = Σxy / (Σx² + λ) ★★ 20 min
- [ ] 9.4 ✏️ Biais² et variance à partir d'un tableau de prédictions ★★ 20 min
- [ ] 9.5 ✏️ Arrêt anticipé avec patience sur une suite de pertes ★★ 15 min
- [ ] 9.6 ∂ Lasso en dimension 1 : le seuillage doux et les zéros exacts ★★★ 35 min
- [ ] 9.7 ✏️ Mise à jour bayésienne d'une droite sur une grille 3 × 3 ★★★ 30 min
- [ ] 9.8 🗣️ Le compromis biais-variance raconté avec le tempo de la boutique ★ 10 min
- [ ] 9.9 📈 Diagnostiquer quatre paires de courbes d'entraînement et de validation ★★ 20 min
- [ ] 9.10 ⚖️ Écarter un point aberrant : nettoyage ou manipulation ? ★★ 20 min
- [ ] 9.11 📄 Belkin et al. (2019) : la double descente ★★ 30 min
- [ ] 9.12 📦 Le tempo de la boutique : polynômes de degré 1, 4 et 15 ★ 15 min
- [ ] 9.13 🔮 Erreurs d'entraînement et de test selon le degré : ta courbe d'abord ★ 15 min
- [ ] 9.14 🔨 mean_squared_error, mean_absolute_error et r2_score ★★ 20 min
- [ ] 9.15 🔨 polynomial_features, interactions comprises ★★ 25 min
- [ ] 9.16 🔨 LinearRegression par moindres carrés ★★ 30 min
- [ ] 9.17 🔨 Ridge en forme fermée, intercept non pénalisé ★★ 30 min
- [ ] 9.18 🔬 Courbes de validation : le degré, puis λ ★★ 30 min
- [ ] 9.19 🔮 Que deviennent les coefficients quand λ grandit ? ★★ 15 min
- [ ] 9.20 🔨 Arrêt anticipé d'une descente de gradient sur un polynôme de degré 12 ★★ 30 min
- [ ] 9.21 📦 Courbes d'apprentissage sur California avec learning_curve ★★ 30 min
- [ ] 9.22 📦 Ridge contre Lasso sur California : chemins de régularisation ★★ 30 min
- [ ] 9.23 🔨 Lasso par descente de coordonnées et soft_threshold ★★★ 60 min
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
- [ ] 10.6 ✏️ Une époque de la règle du perceptron à la main ★★ 25 min
- [ ] 10.7 ∂ Le théorème de convergence du perceptron, guidé pas à pas ★★★ 45 min
- [ ] 10.8 🗣️ Pourquoi un neurone artificiel n'est pas un neurone ★ 10 min
- [ ] 10.9 🧮 Fermi : cerveau humain contre grands modèles ★★ 20 min
- [ ] 10.10 ⚖️ « Cerveaux électroniques » : hype, hivers de l'IA et responsabilité ★★ 20 min
- [ ] 10.11 📄 Rosenblatt (1958) : le perceptron dans le texte ★★ 30 min
- [ ] 10.12 🔨 sign_step et add_bias_column ★ 10 min
- [ ] 10.13 🔮 AND, OR, XOR : le perceptron va-t-il converger ? ★ 10 min
- [ ] 10.14 🔨 neuron_forward : un neurone appliqué à tout un lot ★★ 15 min
- [ ] 10.15 🔨 Des noms de poids (AD, BE…) à la matrice W ★★ 20 min
- [ ] 10.16 🔨 XOR avec trois neurones câblés à la main ★★ 25 min
- [ ] 10.17 🔮 Le learning rate change-t-il un perceptron qui part de zéro ? ★★ 15 min
- [ ] 10.18 📦 Le perceptron de scikit-learn sur portes logiques et manchots ★★ 20 min
- [ ] 10.19 📈 Erreurs par époque : séparable ou pas ? ★★ 20 min
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
## 11 — Apprentissage et raisonnement ⏱️ 17 h

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
## CP2 — Checkpoint II — Concepts ⏱️ 13 h

- [ ] CP2.1 🧠 Vrai ou faux justifiés : huit affirmations sur la partie II ★ 10 min
- [ ] CP2.2 ✏️ Un-contre-tous, un-contre-un : compter les modèles et dépouiller un vote ★ 5 min
- [ ] CP2.3 ✏️ Une itération de k-means et l'inertie obtenue ★★ 15 min
- [ ] CP2.4 ✏️ Densité d'échantillons et hyper-orange en dimension d ★ 6 min
- [ ] CP2.5 ✏️ Plan d'évaluation : tailles des jeux, nombre d'entraînements, fuites ★ 8 min
- [ ] CP2.6 ∂ Ridge en dimension 1 : dériver w* et interpréter λ ★★ 15 min
- [ ] CP2.7 📈 Diagnostiquer trois paires de courbes d'apprentissage ★ 6 min
- [ ] CP2.8 ✏️ Perceptron : une époque sur NAND, puis pourquoi pas XOR ★★ 15 min
- [ ] CP2.9 ✏️ Syllogismes et sophismes : valide, solide, nommer l'erreur ★ 5 min
- [ ] CP2.10 🐛 La fuite cachée d'une validation croisée ★ 9 min
- [ ] CP2.11 🔨 Coder epsilon_greedy_action et une moyenne incrémentale ★ 9 min
- [ ] CP2.12 💼 Entretien : « comment savez-vous que votre modèle ne surapprend pas ? » ★ 5 min
- [ ] CP2.13 ✏️ Parties antérieures : matrice de confusion, Bayes et entropie ★ 9 min
- [ ] Mini-projet MP2 — Prix des logements californiens : un protocole d'évaluation honnête (≈ 10 h)
<!-- wb:end CP2 -->

<!-- wb:section 12 -->
## 12 — Préparation des données ⏱️ 22 h

- [ ] 12.Q1–Q11 🧠 Quiz (11 questions, 35 min)
- [ ] 12.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 12.1 ✏️ One-hot à la main sur Penguins ★ 10 min
- [ ] 12.2 ✏️ Min-max et z-score de cinq valeurs ★ 10 min
- [ ] 12.3 ✏️ Mise à l'échelle univariée ou multivariée ★ 10 min
- [ ] 12.4 ✏️ Réappliquer la transformation : −10 °C, −50 °C et retour aux voitures ★★ 15 min
- [ ] 12.5 ✏️ Trois découpes d'un même tableau : échantillon, feature, élément ★★ 15 min
- [ ] 12.6 ∂ Montrer que la standardisation donne moyenne 0 et variance 1 ★★ 20 min
- [ ] 12.7 ✏️ PCA à la main en 2D : covariance, axe principal, projection ★★★ 35 min
- [ ] 12.8 ∂ Variance d'une projection et axe de variance maximale ★★★ 30 min
- [ ] 12.9 🗣️ La fuite de données expliquée en 5 lignes ★ 10 min
- [ ] 12.10 ⚖️ Supprimer ou imputer : qui disparaît des données ? ★★ 20 min
- [ ] 12.11 📦 Diagnostic de Penguins brut avec pandas ★ 15 min
- [ ] 12.12 🐛 CSV à la française : virgules décimales et « 7e-3 » ★ 15 min
- [ ] 12.13 🔮 Où tombent les données de test après un MinMaxScaler ? ★ 10 min
- [ ] 12.14 📦 pandas pour préparer : dates (« Date Egg »), jointure merge avec une table des îles, pivot_table ★★ 30 min
- [ ] 12.15 📦 Lire des données dans une base SQL : sqlite3, SELECT, JOIN, GROUP BY et pd.read_sql ★★ 30 min
- [ ] 12.16 🔨 Coder StandardScaler (fit, transform, inverse) ★★ 25 min
- [ ] 12.17 🔨 Coder MinMaxScaler avec feature_range et clip ★★ 20 min
- [ ] 12.18 🔨 Coder SimpleImputer (moyenne, médiane, mode, constante) ★★ 25 min
- [ ] 12.19 📦 Scalers de scikit-learn face aux outliers de California ★★ 25 min
- [ ] 12.20 🐛 Le scaler réentraîné sur le test et les get_dummies désalignés ★★ 20 min
- [ ] 12.21 📦 Trafic et température : transformer la cible et revenir aux voitures ★★ 25 min
- [ ] 12.22 🎨 Projeter un nuage 2D : axe horizontal contre axe de variance maximale ★★ 25 min
- [ ] 12.23 📈 Lire une courbe de variance expliquée cumulée ★★ 15 min
- [ ] 12.24 🔮 Combien de composantes pour 90 % de la variance de MNIST ? ★★ 20 min
- [ ] 12.25 📦 Sélection de features : colonnes constantes, VarianceThreshold, SelectKBest ★★ 25 min
- [ ] 12.26 🔨 Coder OrdinalEncoder et OneHotEncoder ★★★ 45 min
- [ ] 12.27 🔨 Coder la PCA (SVD, variance expliquée, whitening, reconstruction) ★★★ 60 min
- [ ] 12.28 📦 Chiffres propres : PCA sur MNIST et reconstructions ★★★ 40 min
- [ ] 12.29 🔬 Échelle des features et descente de gradient ★★★ 40 min
- [ ] 12.30 📦 PCA, t-SNE ou UMAP pour voir MNIST en 2D ★★★ 40 min
- [ ] 12.31 🔬 Fuite de données par le prétraitement : scaler, imputation et TargetEncoder ajustés avant le split ★★★ 45 min
- [ ] 12.32 🛠️ Une fonction de préparation documentée et son test anti-fuite ★★★ 30 min
- [ ] 12.33 🏆 Défi : Penguins brut prêt pour l'entraînement, sans fuite ★★★★ 100 min
- [ ] 12.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 12 -->

<!-- wb:section 13 -->
## 13 — Classifieurs ⏱️ 26 h

- [ ] 13.Q1–Q12 🧠 Quiz (12 questions, 39 min)
- [ ] 13.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 13.1 ✏️ Vote des k plus proches voisins à la main ★ 15 min
- [ ] 13.2 ✏️ Gini et entropie d'une feuille ★ 10 min
- [ ] 13.3 ✏️ Choisir le meilleur seuil sur la longueur des nageoires ★★ 20 min
- [ ] 13.4 ✏️ Distance à une droite et marge d'un SVM ★★ 20 min
- [ ] 13.5 ✏️ L'astuce du noyau vérifiée à la main ★★ 20 min
- [ ] 13.6 ✏️ Anglais ou français ? Naïve Bayes multinomial sur des lettres ★★ 25 min
- [ ] 13.7 ✏️ Postérieur d'un Naïve Bayes gaussien ★★★ 30 min
- [ ] 13.8 ∂ Gradient de la log-loss de la régression logistique ★★★ 40 min
- [ ] 13.9 🧮 Mémoire et temps d'un kNN sur MNIST complet ★★ 15 min
- [ ] 13.10 🗣️ La marge d'un SVM expliquée en 5 lignes ★ 10 min
- [ ] 13.11 ⚖️ Refus de crédit : arbre lisible ou SVM opaque ? ★★ 20 min
- [ ] 13.12 🔮 Prédire la frontière de cinq classifieurs avant de l'afficher ★ 15 min
- [ ] 13.13 🔬 Choisir k : de la frontière en dents de scie à l'érosion ★★ 30 min
- [ ] 13.14 🔨 Coder pairwise_distances et KNeighborsClassifier ★★★ 45 min
- [ ] 13.15 🔬 kNN et malédiction de la dimension ★★★ 35 min
- [ ] 13.16 📦 SVM linéaire : vecteurs de support et paramètre C ★★ 25 min
- [ ] 13.17 🎨 Relever les cercles en 3D pour les séparer par un plan ★★ 30 min
- [ ] 13.18 📦 Noyau RBF : régler C et gamma par validation croisée ★★★ 35 min
- [ ] 13.19 🔬 Instabilité et profondeur d'un arbre ★★ 30 min
- [ ] 13.20 🔨 Coder les impuretés et le gain d'un split ★★ 25 min
- [ ] 13.21 🐛 L'arbre parfait qui triche : une colonne identifiant ★★ 20 min
- [ ] 13.22 🔨 Coder la recherche du meilleur split ★★★ 45 min
- [ ] 13.23 📦 Arbres scikit-learn : plot_tree, export_text et élagage ccp_alpha ★★★ 35 min
- [ ] 13.24 🔨 Coder un arbre de décision récursif ★★★★ 100 min
- [ ] 13.25 🔨 Préparer l'arbre pour les ensembles : poids, features tirées, importances ★★★ 50 min
- [ ] 13.26 🔨 Arbre de régression : même squelette, autre critère ★★ 30 min
- [ ] 13.27 🔮 Naïve Bayes sur des blobs gaussiens puis sur des lunes ★ 10 min
- [ ] 13.28 🔨 Coder GaussianNB avec log-densités et logsumexp ★★★ 40 min
- [ ] 13.29 🔨 Coder MultinomialNB avec lissage ★★ 30 min
- [ ] 13.30 📦 Dialogue ou narration ? Classer des phrases de Holmes : sac de mots, TF-IDF et Naive Bayes ★★★ 40 min
- [ ] 13.31 📦 Régression logistique scikit-learn : coefficients et probabilités ★★ 25 min
- [ ] 13.32 🔨 Coder la régression logistique par descente de gradient ★★★ 50 min
- [ ] 13.33 📈 Associer quatre frontières à leurs classifieurs ★ 15 min
- [ ] 13.34 🛠️ Mesurer et vectoriser : accélérer son kNN ★★ 25 min
- [ ] 13.35 🔬 Comparatif mylearn contre scikit-learn : accuracy et temps ★★★ 45 min
- [ ] 13.36 🏆 Défi : 96 % sur MNIST avec un classifieur classique ★★★★ 100 min
- [ ] 13.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (30 cartes)
<!-- wb:end 13 -->

<!-- wb:section 14 -->
## 14 — Ensembles ⏱️ 19 h

- [ ] 14.Q1–Q11 🧠 Quiz (11 questions, 35 min)
- [ ] 14.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 14.1 ✏️ Dépouiller un vote pondéré ★ 10 min
- [ ] 14.2 ✏️ Additionner les voix de trois droites, région par région ★★ 15 min
- [ ] 14.3 ∂ Probabilité d'être oublié par un bootstrap : vers 36,8 % ★★ 20 min
- [ ] 14.4 ∂ Ajuster les résidus, c'est descendre le gradient de la MSE ★★ 20 min
- [ ] 14.5 ✏️ Quand la majorité de votants indépendants se trompe ★★ 25 min
- [ ] 14.6 ✏️ Une itération d'AdaBoost à la main ★★★ 30 min
- [ ] 14.7 ✏️ Deux itérations de gradient boosting sur cinq points ★★★ 35 min
- [ ] 14.8 🗣️ Bagging et boosting expliqués en 5 lignes ★ 10 min
- [ ] 14.9 ⚖️ Un jury de modèles biaisés reste biaisé ★★ 20 min
- [ ] 14.10 📄 Pourquoi les arbres battent encore le deep learning sur le tabulaire ★★ 30 min
- [ ] 14.11 🔮 Trois classifieurs du ch. 13 votent : mieux que le meilleur ? ★ 10 min
- [ ] 14.12 🔨 Coder le vote pondéré et le tirage bootstrap ★★ 25 min
- [ ] 14.13 🔬 Jury de votants : indépendants puis corrélés ★★ 30 min
- [ ] 14.14 📦 Voting et stacking avec les classifieurs du ch. 13 ★★ 25 min
- [ ] 14.15 📦 Forêts scikit-learn sur California : OOB et parallélisme ★★ 25 min
- [ ] 14.16 🔬 Combien d'arbres ? Bagging, forêt et ExtraTrees face au bruit ★★ 30 min
- [ ] 14.17 🔨 Coder BaggingClassifier avec score out-of-bag ★★★ 50 min
- [ ] 14.18 🔨 Coder RandomForest et ExtraTrees ★★★ 45 min
- [ ] 14.19 📦 Importance des features : impureté contre permutation ★★★ 35 min
- [ ] 14.20 📈 Lire les courbes train/validation d'un boosting ★ 15 min
- [ ] 14.21 🔮 Learning rate et nombre d'arbres : prédire les courbes ★ 15 min
- [ ] 14.22 🎨 Frontières d'AdaBoost après 1, 3, 10 et 50 souches ★★ 25 min
- [ ] 14.23 🐛 L'AdaBoost qui n'apprend rien ★★ 25 min
- [ ] 14.24 🔨 Coder AdaBoost (SAMME) avec des souches pondérées ★★★ 50 min
- [ ] 14.25 🔨 Coder un gradient boosting de régression ★★★ 50 min
- [ ] 14.26 📦 HistGradientBoosting, XGBoost et LightGBM sur California ★★★ 45 min
- [ ] 14.27 🛠️ Un benchmark reproductible : graines, versions et tableau de résultats ★★ 25 min
- [ ] 14.28 🏆 Défi : RMSE ≤ 0,45 sur California ★★★★ 100 min
- [ ] 14.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 14 -->

<!-- wb:section 15 -->
## 15 — scikit-learn ⏱️ 21 h

- [ ] 15.Q1–Q12 🧠 Quiz (12 questions, 38 min)
- [ ] 15.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 15.1 ✏️ Compter les modèles entraînés par une recherche en grille ★ 10 min
- [ ] 15.2 ✏️ De decision_function à predict_proba ★ 10 min
- [ ] 15.3 ✏️ Ramener une droite ajustée dans l'espace d'origine ★★ 15 min
- [ ] 15.4 ✏️ Combien de features polynomiales ? ★★ 20 min
- [ ] 15.5 🧮 Budget d'une recherche d'hyperparamètres ★★ 15 min
- [ ] 15.6 🗣️ Le Pipeline expliqué en 5 lignes ★ 10 min
- [ ] 15.7 ⚖️ Boston Housing : un dataset retiré pour raisons éthiques ★★ 20 min
- [ ] 15.8 📄 Lire l'article sur la conception de l'API de scikit-learn ★★ 30 min
- [ ] 15.9 📦 Anatomie d'un estimateur : Ridge sur un sinus bruité ★ 15 min
- [ ] 15.10 🔮 decision_function ou predict_proba : prédire formes et valeurs ★ 10 min
- [ ] 15.11 📦 Datasets synthétiques, fetch_* et train_test_split stratifié ★ 15 min
- [ ] 15.12 🐛 Faire tourner le code de 2018 avec scikit-learn 1.6 ★★ 25 min
- [ ] 15.13 📦 Clustering avec scikit-learn : KMeans (le coude du livre), DBSCAN et clustering hiérarchique sur deux lunes et sept blobs ★★ 25 min
- [ ] 15.14 📦 Transformers : test hors de [0, 1] et inverse_transform ★★ 20 min
- [ ] 15.15 📦 PCA de 3D vers 2D et blanchiment ★★ 20 min
- [ ] 15.16 🔬 Cinq spirales : quel ensemble les démêle ? ★★ 30 min
- [ ] 15.17 📦 cross_validate, KFold, StratifiedKFold et RidgeClassifierCV ★★ 25 min
- [ ] 15.18 📦 GridSearchCV : best_params_, cv_results_ et score d'entraînement ★★ 25 min
- [ ] 15.19 📦 RandomizedSearchCV avec des distributions ★★ 25 min
- [ ] 15.20 🔮 Grille ou hasard : qui gagne avec 16 essais ? ★★ 20 min
- [ ] 15.21 📦 Frontières de décision : DecisionBoundaryDisplay et surfaces 3D ★★ 20 min
- [ ] 15.22 📦 Pipeline polynomial + ridge : degré et alpha ensemble ★★★ 35 min
- [ ] 15.23 📈 Lire cv_results_ : carte de chaleur et écarts-types ★★ 20 min
- [ ] 15.24 🔬 Refaire l'expérience de fuite du ch. 8 (sélection de features sur du bruit) avec un Pipeline ★★ 25 min
- [ ] 15.25 🔬 Régler le seuil de décision d'un classifieur ★★★ 35 min
- [ ] 15.26 📦 Classes déséquilibrées : class_weight, rééchantillonnage et la bonne métrique (recall, PR-AUC) ★★★ 35 min
- [ ] 15.27 📦 La même validation croisée pour six familles de modèles scikit-learn : score moyen, écart-type et temps ★★★ 40 min
- [ ] 15.28 📦 ColumnTransformer sur Penguins brut ★★★ 45 min
- [ ] 15.29 🐛 Quand scikit-learn refuse un estimateur mylearn ★★★ 35 min
- [ ] 15.30 🛠️ Sauvegarder, recharger et versionner un pipeline ★★ 20 min
- [ ] 15.31 🏆 Défi : prédire le sexe des manchots depuis le fichier brut ★★★★ 100 min
- [ ] 15.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 15 -->

<!-- wb:section CP3 -->
## CP3 — Checkpoint III — ML classique ⏱️ 11 h

- [ ] CP3.1 🔮 Prédire la sortie de six extraits scikit-learn (formes, scores, avertissements) ★ 8 min
- [ ] CP3.2 ✏️ Standardiser et encoder un mini-tableau, puis transformer une ligne de test ★ 8 min
- [ ] CP3.3 ✏️ Gini et meilleur seuil sur six points ★ 8 min
- [ ] CP3.4 ✏️ kNN à la main avant et après standardisation ★ 6 min
- [ ] CP3.5 ✏️ Une itération d'AdaBoost et une de gradient boosting ★ 10 min
- [ ] CP3.6 🐛 Trouver trois fuites dans un script d'entraînement ★ 12 min
- [ ] CP3.7 📦 Pipeline ColumnTransformer + HistGradientBoosting + recherche en grille ★★ 20 min
- [ ] CP3.8 📈 Lire une courbe de validation (max_depth) et une courbe de boosting ★ 5 min
- [ ] CP3.9 🗣️ Expliquer l'astuce du noyau en 5 lignes à un débutant ★ 5 min
- [ ] CP3.10 💼 Comme en entretien : predict_proba vaut 0,9, est-ce fiable, et comment déployer ce pipeline ? ★ 5 min
- [ ] CP3.11 ✏️ Parties I et II : precision/recall/F1 d'une matrice de confusion ; modèles d'un 5-fold stratifié ★ 8 min
- [ ] Mini-projet MP3 — Prédire un revenu à partir du recensement : pipeline tabulaire complet et audit d'équité (≈ 8 h)
<!-- wb:end CP3 -->

<!-- wb:section 16 -->
## 16 — Réseaux feed-forward ⏱️ 13 h

- [ ] 16.Q1–Q10 🧠 Quiz (10 questions, 30 min)
- [ ] 16.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 16.1 ✏️ Compter les poids et les biais d'un MLP ★ 10 min
- [ ] 16.2 ✏️ Passe avant à la main dans un réseau 2-2-1 ★ 15 min
- [ ] 16.3 ✏️ Formes et broadcasting d'un mini-lot à travers les couches ★★ 15 min
- [ ] 16.4 ✏️ Ordre d'évaluation d'un graphe et détection de boucle ★★ 15 min
- [ ] 16.5 ✏️ Simuler un réseau asynchrone avec horodatage ★★ 20 min
- [ ] 16.6 ∂ Symétrie : pourquoi une initialisation constante ne se brise jamais ★★ 20 min
- [ ] 16.7 ∂ Conserver la variance : retrouver les règles de LeCun, Glorot et He ★★★ 40 min
- [ ] 16.8 🧮 Fermi : taille, mémoire et coût d'un MLP pour MNIST ★★ 15 min
- [ ] 16.9 🗣️ Expliquer un réseau feed-forward avec une chaîne d'usine ★ 10 min
- [ ] 16.10 📄 Glorot & Bengio (2010) : pourquoi les réseaux profonds apprenaient mal ★★★ 45 min
- [ ] 16.11 🔮 Prédire les formes des sorties avant d'exécuter ★ 10 min
- [ ] 16.12 🔨 Compter les paramètres par programme : count_parameters ★ 10 min
- [ ] 16.13 🔨 La couche dense : dense_forward ★ 15 min
- [ ] 16.14 🔨 Dix façons d'initialiser : init_weights ★★ 25 min
- [ ] 16.15 🎨 Reproduire les figures 16.5 et 16.6, puis comparer Glorot et He ★★ 20 min
- [ ] 16.16 🔨 Ordre topologique d'un graphe : topological_order ★★ 25 min
- [ ] 16.17 🔨 Empiler les couches : init_mlp et mlp_forward ★★ 30 min
- [ ] 16.18 📦 Relire un MLPClassifier de scikit-learn avec ton mlp_forward ★★ 25 min
- [ ] 16.19 🔮 Initialisation constante : les neurones restent-ils jumeaux ? ★★ 20 min
- [ ] 16.20 🐛 Chasse au bug : transposée, biais mal diffusé, variance prise pour un écart-type ★★ 25 min
- [ ] 16.21 🛠️ Rendre l'aléatoire reproductible et le prouver avec pytest ★★ 20 min
- [ ] 16.22 📈 Lire des histogrammes d'activations couche par couche ★★ 20 min
- [ ] 16.23 🔬 Activations à travers 20 couches selon l'initialisation ★★★ 45 min
- [ ] 16.24 🏆 Défi : garder vivant un réseau ReLU de 50 couches ★★★ 60 min
- [ ] 16.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (18 cartes)
<!-- wb:end 16 -->

<!-- wb:section 17 -->
## 17 — Fonctions d'activation ⏱️ 14 h

- [ ] 17.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] 17.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 17.1 ✏️ Effondrer un réseau linéaire en un seul neurone ★ 15 min
- [ ] 17.2 ✏️ Tableau de valeurs de neuf activations ★ 15 min
- [ ] 17.3 ∂ Une composition d'affines reste affine ; une ReLU casse tout ★★ 20 min
- [ ] 17.4 ∂ Dériver la sigmoïde, tanh et softplus ★★ 25 min
- [ ] 17.5 ✏️ Maxout et ReLU : construire |x|, une cuvette et une bosse ★★ 20 min
- [ ] 17.6 ✏️ Softmax à la main : translation, température, rapports ★★ 20 min
- [ ] 17.7 ∂ La Jacobienne de la softmax ★★★ 30 min
- [ ] 17.8 🧮 Fermi : une activation coûte-t-elle cher face au produit matriciel ? ★★ 15 min
- [ ] 17.9 🗣️ Expliquer la non-linéarité avec un train articulé ★ 10 min
- [ ] 17.10 📄 Hendrycks & Gimpel (2016) : les GELU ★★★ 45 min
- [ ] 17.11 🔮 Un MLP sans activation sur les lunes : quelle frontière ? ★ 15 min
- [ ] 17.12 🔨 identity, step, relu, leaky_relu et leurs dérivées ★ 15 min
- [ ] 17.13 🔨 Sigmoïde, tanh et softplus sans débordement ★★ 25 min
- [ ] 17.14 🔨 ELU, SiLU, GELU et le registre get_activation ★★ 25 min
- [ ] 17.15 🔨 Softmax et log_softmax stables (astuce log-sum-exp) ★★ 25 min
- [ ] 17.16 🔨 La Jacobienne de la softmax en NumPy ★★★ 30 min
- [ ] 17.17 🐛 Chasse au bug : nan, inf et mauvais axe ★★ 25 min
- [ ] 17.18 🎨 Reproduire la galerie de la figure 17.22, dérivées comprises ★★ 25 min
- [ ] 17.19 🔮 Softmax et température : prédire la distribution ★★ 15 min
- [ ] 17.20 📦 Quatre activations de scikit-learn sur les lunes et les spirales ★★ 25 min
- [ ] 17.21 📈 Diagnostiquer la saturation sur des histogrammes de pré-activations ★★ 20 min
- [ ] 17.22 🛠️ Une batterie de tests paramétrés pour toutes les activations ★★ 25 min
- [ ] 17.23 🔬 Produit des dérivées sur 10 couches : sigmoïde, tanh, ReLU, GELU ★★★ 45 min
- [ ] 17.24 🏆 Défi spirales : au moins 97 % avec au plus 2 000 paramètres ★★★ 60 min
- [ ] 17.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (22 cartes)
<!-- wb:end 17 -->

<!-- wb:section 18 -->
## 18 — Rétropropagation ⏱️ 26 h

- [ ] 18.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] 18.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 18.1 ✏️ Le delta, amplificateur de changement ★ 10 min
- [ ] 18.2 ✏️ Deltas de sortie avec l'erreur quadratique : attention au signe ★★ 15 min
- [ ] 18.3 ✏️ Descente sur une parabole : quel learning rate ? ★★ 15 min
- [ ] 18.4 ✏️ Rétropropagation complète du réseau minuscule, sans activation ★★ 30 min
- [ ] 18.5 ✏️ Le même réseau avec une sigmoïde cachée : le facteur σ'(z) ★★ 25 min
- [ ] 18.6 ∂ La règle de la chaîne derrière les deltas ★★ 25 min
- [ ] 18.7 ∂ Forme matricielle : XᵀΔ, ΔWᵀ et somme des lignes ★★★ 35 min
- [ ] 18.8 ∂ Softmax et entropie croisée : le gradient vaut p − y ★★★ 35 min
- [ ] 18.9 🧮 Fermi : rétropropagation contre force brute ★★ 20 min
- [ ] 18.10 🗣️ Expliquer la rétropropagation sans équation ★ 10 min
- [ ] 18.11 📄 Rumelhart, Hinton & Williams (1986) : l'article qui a relancé les réseaux ★★★ 45 min
- [ ] 18.12 🔨 mse_loss : la perte et son gradient ★ 10 min
- [ ] 18.13 🔮 La méthode très lente : combien d'essais pour progresser ? ★★ 20 min
- [ ] 18.14 🔨 Entropies croisées calculées sur les logits (binaire et softmax) ★★ 30 min
- [ ] 18.15 🔨 dense_backward : les trois gradients d'une couche ★★ 25 min
- [ ] 18.16 🔨 gradient_check : vérifier un gradient par différences finies ★★ 25 min
- [ ] 18.17 🐛 Chasse au bug : trois gradients faux démasqués ★★ 25 min
- [ ] 18.18 🔨 mlp_backward : rétropropager à travers tout le réseau ★★★ 45 min
- [ ] 18.19 🔨 Entraîner le réseau 2-4-4-1 à 37 poids sur les lunes ★★★ 45 min
- [ ] 18.20 📦 La même expérience avec MLPClassifier(solver='sgd') ★★ 20 min
- [ ] 18.21 🎨 Reproduire les figures 18.49 à 18.57 (η = 0,5 ; 0,05 ; 0,01) ★★ 30 min
- [ ] 18.22 📈 Lire des courbes de loss : divergence, lenteur, plateau, bug ★★ 20 min
- [ ] 18.23 🔬 Balayage du learning rate sur cinq graines ★★★ 40 min
- [ ] 18.24 🔬 Neurones saturés et ReLU mortes pendant l'entraînement ★★★ 45 min
- [ ] 18.25 🛠️ Écrire un test pytest de gradient pour une couche ★★ 25 min
- [ ] 18.26 🔨 Micro-autograd : la classe Value ★★★ 90 min
- [ ] 18.27 🔮 À la main, Value, mlp_backward, différences finies : quatre gradients identiques ? ★★ 20 min
- [ ] 18.28 🔨 MiniMLP : entraîner un petit réseau fait de Value ★★★ 45 min
- [ ] 18.29 🏆 Défi MNIST : au moins 97 % avec ton MLP NumPy ★★★★ 120 min
- [ ] 18.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (28 cartes)
<!-- wb:end 18 -->

<!-- wb:section 19 -->
## 19 — Optimiseurs ⏱️ 20 h

- [ ] 19.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] 19.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 19.1 ✏️ Plannings exponentiel et par paliers à la main ★ 10 min
- [ ] 19.2 ✏️ Trois pas de momentum, deux conventions ★★ 15 min
- [ ] 19.3 ✏️ Nesterov : gradient anticipé et reformulation de PyTorch ★★ 20 min
- [ ] 19.4 ✏️ Adagrad et RMSprop sur une suite de gradients ★★ 15 min
- [ ] 19.5 ✏️ Un pas d'Adam et la correction de biais ★★ 20 min
- [ ] 19.6 ✏️ Adam + L2 ou AdamW : la même pénalité ? Convertir alpha, lambda et weight_decay d'une convention à l'autre ★★ 25 min
- [ ] 19.7 ∂ Sur une parabole, η doit rester sous 2/a ★★★ 30 min
- [ ] 19.8 ∂ Moyenne mobile exponentielle : mémoire effective et biais initial ★★★ 30 min
- [ ] 19.9 🧮 Fermi : la mémoire des optimiseurs, du MLP au LLM ★★ 15 min
- [ ] 19.10 🗣️ Expliquer le momentum avec une bille qui roule ★ 10 min
- [ ] 19.11 ⚖️ Le coût énergétique d'une recherche d'hyperparamètres ★★ 20 min
- [ ] 19.12 📄 Kingma & Ba (2014) : l'algorithme Adam ligne par ligne ★★★ 45 min
- [ ] 19.13 🛠️ Lire la doc de torch.optim : correspondance livre ↔ PyTorch ★★ 25 min
- [ ] 19.14 🔮 Pas constant sur une vallée : combien de pas pour le fond ? ★ 15 min
- [ ] 19.15 🔨 Plannings : exponential_decay_lr, step_decay_lr, bold_driver_lr ★ 15 min
- [ ] 19.16 🔨 SGD, Momentum et Nesterov à la manière de torch.optim ★★ 30 min
- [ ] 19.17 🔨 Adagrad, RMSprop et Adadelta ★★ 30 min
- [ ] 19.18 🔨 Adam et AdamW ★★ 30 min
- [ ] 19.19 🔨 Warmup + cosinus et gradient clipping ★★ 25 min
- [ ] 19.20 🔮 Trop de momentum : la bille quitte-t-elle la vallée ? ★★ 15 min
- [ ] 19.21 🎨 Reproduire les figures du chapitre : pas constant, décroissant, momentum, Nesterov ★★ 30 min
- [ ] 19.22 🐛 Chasse au bug : un Adam qui diverge ★★ 25 min
- [ ] 19.23 📈 Diagnostiquer des courbes de loss d'optimiseurs ★★ 20 min
- [ ] 19.24 📦 Les solveurs de MLPClassifier : sgd, momentum, adaptive, adam ★★ 25 min
- [ ] 19.25 🔬 Batch, stochastique, mini-lot sur les lunes ★★★ 45 min
- [ ] 19.26 🔬 Course d'optimiseurs sur la fonction de Rosenbrock ★★★ 45 min
- [ ] 19.27 🔬 Huit optimiseurs sur MNIST, avec warmup, cosinus et clipping ★★★ 60 min
- [ ] 19.28 🏆 Défi : 98 % sur MNIST en cinq epochs ★★★ 90 min
- [ ] 19.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (26 cartes)
<!-- wb:end 19 -->

<!-- wb:section 20 -->
## 20 — Deep learning et premiers pas en PyTorch ⏱️ 24 h

- [ ] 20.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] 20.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 20.1 ✏️ Suivre la forme d'un tenseur couche après couche ★ 15 min
- [ ] 20.2 ✏️ Dimensionner la couche de sortie et choisir la perte ★ 10 min
- [ ] 20.3 ✏️ Dropout inversé à la main ★★ 15 min
- [ ] 20.4 ✏️ Batchnorm à la main sur un mini-lot de quatre exemples ★★ 20 min
- [ ] 20.5 ∂ Dropout : espérance, sous-réseaux et règle du test ★★ 20 min
- [ ] 20.6 ✏️ Compter les 138 millions de paramètres de VGG16 ★★ 20 min
- [ ] 20.7 ✏️ Connexion résiduelle et LayerNorm à la main ★★ 20 min
- [ ] 20.8 ∂ Le gradient de la batchnorm ★★★ 40 min
- [ ] 20.9 🧮 Fermi : entraîner VGG16 sur ImageNet chez soi ? ★★ 20 min
- [ ] 20.10 🗣️ Agents, superviseurs, directeur : expliquer l'apprentissage hiérarchique ★ 10 min
- [ ] 20.11 ⚖️ Refus de prêt automatisé : explication, dérive et biais ★★★ 40 min
- [ ] 20.12 📄 Ioffe & Szegedy (2015), puis Santurkar et al. (2018) : lecture critique ★★★ 45 min
- [ ] 20.13 📦 Premiers tenseurs PyTorch : créer, convertir, dtype, device ★ 15 min
- [ ] 20.14 🔮 Prédire les formes produites par les couches utilitaires ★ 15 min
- [ ] 20.15 📦 Autograd : retrouver les gradients du réseau minuscule du ch. 18 ★★ 20 min
- [ ] 20.16 🔨 dropout_forward et dropout_backward ★★ 25 min
- [ ] 20.17 🔨 layer_norm, l2_penalty et EarlyStopping ★★ 30 min
- [ ] 20.18 📦 Écrire sa classe nn.Module et compter ses paramètres ★★ 25 min
- [ ] 20.19 📦 Dataset et DataLoader : MNIST en mini-lots ★★ 25 min
- [ ] 20.20 📦 Zoo des couches : convolution, pooling, récurrente, utilitaires, bruit ★★ 30 min
- [ ] 20.21 🎨 Reproduire la figure 20.17 : le plan de VGG16 couche par couche ★★ 30 min
- [ ] 20.22 🔨 batchnorm_forward et batchnorm_backward ★★★ 75 min
- [ ] 20.23 📦 La boucle d'entraînement PyTorch complète sur MNIST ★★★ 45 min
- [ ] 20.24 🐛 Chasse aux bugs PyTorch : cinq erreurs classiques ★★ 30 min
- [ ] 20.25 📈 Courbes train/validation : surapprentissage, eval() oublié, fuite ★★ 20 min
- [ ] 20.26 🔮 Un modèle ImageNet face à des images qu'il n'a jamais vues ★★ 25 min
- [ ] 20.27 🛠️ Sauvegarder, recharger et reproduire un entraînement ★★ 25 min
- [ ] 20.28 📦 Dropout, BatchNorm, AdamW et early stopping sur Fashion-MNIST ★★★ 60 min
- [ ] 20.29 🔬 Ablation : qu'est-ce qui retarde le surapprentissage ? ★★★ 60 min
- [ ] 20.30 🔬 Un réseau de 30 couches : connexions résiduelles et LayerNorm ★★★ 45 min
- [ ] 20.31 🏆 Défi Fashion-MNIST : au moins 90 % avec un MLP de 300 k paramètres au plus ★★★★ 120 min
- [ ] 20.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (28 cartes)
<!-- wb:end 20 -->

<!-- wb:section CP4 -->
## CP4 — Checkpoint IV — Réseaux de neurones ⏱️ 11 h

- [ ] CP4.1 🧠 Questions flash : formes, nombres de paramètres, valeurs d'activations ★ 8 min
- [ ] CP4.2 ✏️ Passe avant, rétropropagation et un pas de SGD sur un réseau 2-2-1 à ReLU ★★ 18 min
- [ ] CP4.3 ✏️ Un pas de momentum et deux pas d'Adam à la main ★ 10 min
- [ ] CP4.4 ✏️ Dropout et batchnorm : mode entraînement contre mode évaluation ★ 10 min
- [ ] CP4.5 ∂ Pourquoi le gradient s'évanouit à travers des sigmoïdes empilées ★ 8 min
- [ ] CP4.6 🧮 Fermi : paramètres et mémoire d'entraînement d'un MLP avec AdamW ★ 6 min
- [ ] CP4.7 🐛 Quatre bugs dans une boucle d'entraînement PyTorch (sur papier) ★ 10 min
- [ ] CP4.8 📈 Quatre courbes de loss, quatre diagnostics ★ 6 min
- [ ] CP4.9 📦 Écrire de mémoire une epoch d'entraînement PyTorch avec AdamW et clipping ★★ 15 min
- [ ] CP4.10 💼 Expliquer la rétropropagation comme en entretien ★ 5 min
- [ ] CP4.11 ⚖️ Refus de crédit par un réseau : quelle explication fournir ? ★ 5 min
- [ ] CP4.12 ✏️ Retour sur les parties I à III : entropie croisée, régularisation, standardisation ★ 10 min
- [ ] Mini-projet MP4 — Un mini-framework de deep learning en NumPy, validé contre PyTorch (≈ 8 h)
<!-- wb:end CP4 -->

<!-- wb:section 21 -->
## 21 — Réseaux convolutifs (CNN) ⏱️ 27 h

- [ ] 21.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] 21.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 21.1 ✏️ Le détecteur de jaune à la main ★ 10 min
- [ ] 21.2 ✏️ Une corrélation 5×5 par 3×3 à la main ★★ 20 min
- [ ] 21.3 ✏️ Tailles de sortie : padding, stride et la hiérarchie 600 → 300 → 100 ★★ 20 min
- [ ] 21.4 ✏️ Compter les paramètres : convolution contre couche dense ★★ 15 min
- [ ] 21.5 ✏️ Champ récepteur effectif d'une pile convolution + pooling ★★ 20 min
- [ ] 21.6 ✏️ Convolution transposée à la main : insérer des zéros puis corréler ★★★ 30 min
- [ ] 21.7 ∂ Rétropropager dans une convolution 1D : la somme sur les positions ★★★ 35 min
- [ ] 21.8 🧮 Fermi : paramètres, mémoire et calcul de VGG16 ★★ 25 min
- [ ] 21.9 🗣️ Expliquer la convolution en cinq lignes ★ 10 min
- [ ] 21.10 ⚖️ Deviner l'âge et le genre sur une photo : jusqu'où aller ? ★★ 20 min
- [ ] 21.11 📄 Lire LeCun et al. (1998) : LeNet-5 ★★★ 45 min
- [ ] 21.12 🔨 conv_output_size et pad2d ★ 15 min
- [ ] 21.13 🔮 Prédire où une image CIFAR-10 « jaunit » ★ 10 min
- [ ] 21.14 🔨 im2col : la vue « œil de mouche » ★★ 30 min
- [ ] 21.15 🔨 conv2d : stride, padding, dilatation et groupes ★★★ 90 min
- [ ] 21.16 🛠️ Tests pytest paramétrés pour conv2d ★★ 20 min
- [ ] 21.17 🎨 Reproduire la chasse aux rayures verticales (Fig. 21.14) ★★ 20 min
- [ ] 21.18 🔨 max_pool2d et avg_pool2d ★★ 25 min
- [ ] 21.19 📦 Flou, Sobel, netteté : des filtres faits main avec F.conv2d ★★ 20 min
- [ ] 21.20 🔮 Conv1d, noyau pleine largeur et 1×1 : prédire formes et paramètres ★★ 15 min
- [ ] 21.21 🎨 Le détecteur de visage hiérarchique (Fig. 21.18 à 21.23) ★★★ 45 min
- [ ] 21.22 📦 Le convnet MNIST du livre en PyTorch, sur Fashion-MNIST ★★★ 45 min
- [ ] 21.23 📈 Diagnostiquer les courbes et la matrice de confusion du CNN ★★ 15 min
- [ ] 21.24 🐛 Le CNN dont les formes ne passent pas ★★ 20 min
- [ ] 21.25 🔬 Briser la symétrie : filtres identiques, aléatoires ou He ★★ 25 min
- [ ] 21.26 🔬 Stride ou pooling : précision et temps sur CIFAR-10 ★★★ 40 min 🚀
- [ ] 21.27 📦 Filtres et cartes d'activation d'un ResNet-18 pré-entraîné ★★ 30 min
- [ ] 21.28 📦 Ce qu'un filtre cherche : montée de gradient sur l'entrée ★★★ 45 min
- [ ] 21.29 📦 Attaque FGSM contre ton CNN Fashion-MNIST ★★★ 40 min
- [ ] 21.30 🔨 conv_transpose2d par insertion de zéros ★★★ 50 min
- [ ] 21.31 🔨 conv2d_backward : rétropropager à travers une convolution ★★★★ 100 min
- [ ] 21.32 🏆 Défi Fashion-MNIST : au moins 92 % avec moins de 100 000 paramètres ★★★★ 120 min
- [ ] 21.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 21 -->

<!-- wb:section 22 -->
## 22 — Réseaux récurrents (RNN, LSTM, GRU) ⏱️ 22 h

- [ ] 22.Q1–Q12 🧠 Quiz (12 questions, 35 min)
- [ ] 22.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 22.1 ✏️ Combien de fenêtres ? Températures, taches solaires et Holmes ★ 10 min
- [ ] 22.2 ✏️ Dérouler à la main une cellule RNN scalaire sur trois pas ★★ 15 min
- [ ] 22.3 ✏️ Compter les poids : la cellule à 69 poids du livre, puis nn.RNN et nn.LSTM ★★ 20 min
- [ ] 22.4 ✏️ Un pas de LSTM à la main ★★ 25 min
- [ ] 22.5 ✏️ Softmax avec température à la main ★ 10 min
- [ ] 22.6 ∂ Pourquoi le gradient s'évanouit : un produit de dérivées de tanh ★★★ 35 min
- [ ] 22.7 ∂ Le chemin additif de la cellule LSTM ★★★ 30 min
- [ ] 22.8 🧮 Fermi : les 17 millions de poids du CNN-LSTM et le prix d'une époque ★★ 20 min
- [ ] 22.9 🗣️ Expliquer l'état caché d'un RNN en cinq lignes ★ 10 min
- [ ] 22.10 ⚖️ Écrire « à la manière de », imiter une voix : usages et abus ★★ 20 min
- [ ] 22.11 📄 Lire Hochreiter et Schmidhuber (1997) : Long Short-Term Memory ★★★ 45 min
- [ ] 22.12 🔨 build_vocab, encode et decode sur Holmes et Verne ★ 15 min
- [ ] 22.13 🔨 make_windows et make_sequence_pairs ★★ 20 min
- [ ] 22.14 🔮 Un MLP sur des fenêtres : l'ordre compte-t-il ? ★★ 20 min
- [ ] 22.15 🔨 rnn_cell_forward et rnn_forward ★★ 30 min
- [ ] 22.16 🔨 lstm_cell_forward et lstm_forward (portes i, f, g, o) ★★★ 45 min
- [ ] 22.17 🔨 gru_cell_forward (portes r, z, n) ★★ 25 min
- [ ] 22.18 🛠️ Benchmark : ta cellule NumPy contre nn.LSTM ★★ 20 min
- [ ] 22.19 🔬 Mesurer la disparition du gradient : RNN tanh contre LSTM ★★ 30 min
- [ ] 22.20 🔨 temperature_softmax et sample_from_logits ★★ 25 min
- [ ] 22.21 🐛 Série temporelle piégée : fuite, mélange et batch_first ★★ 25 min
- [ ] 22.22 📦 RNN profonds et bidirectionnels : formes de output et h_n ★★ 25 min
- [ ] 22.23 📦 Prévoir les taches solaires avec nn.LSTM, face à la persistance ★★★ 45 min
- [ ] 22.24 🔮 Température : prédire l'allure du texte généré ★ 10 min
- [ ] 22.25 📦 Générer du Holmes caractère par caractère (deux LSTM, many-to-one) ★★★ 60 min 🚀
- [ ] 22.26 📈 Lire la courbe de loss et la perplexité du générateur ★★ 15 min
- [ ] 22.27 🔬 Holmes contre Verne : même réseau, deux langues ★★★ 40 min 🚀
- [ ] 22.28 📦 CNN-LSTM : la balle monte-t-elle ou descend-elle ? ★★★ 50 min
- [ ] 22.29 🏆 Défi taches solaires : battre la persistance de 20 % à six mois ★★★★ 120 min
- [ ] 22.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 22 -->

<!-- wb:section 23 -->
## 23 — PyTorch en pratique 1 : du jeu de données au modèle sauvegardé ⏱️ 19 h

- [ ] 23.Q1–Q12 🧠 Quiz (12 questions, 34 min)
- [ ] 23.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 23.1 ✏️ Remodeler 12 éléments : indices et ordre de lecture ★ 10 min
- [ ] 23.2 ✏️ De NHWC à NCHW : où atterrit chaque pixel ? ★★ 15 min
- [ ] 23.3 ✏️ 623 290 paramètres : refaire le compte du livre ★ 10 min
- [ ] 23.4 ✏️ Une cross-entropy à la main : indice de classe ou one-hot ★★ 20 min
- [ ] 23.5 ✏️ Batches par époque et nombre de mises à jour ★ 10 min
- [ ] 23.6 ✏️ Early stopping à la main : patience et min_delta ★★ 15 min
- [ ] 23.7 ✏️ Plannings de learning rate : le planning du livre avec LambdaLR ★★ 20 min
- [ ] 23.8 🧮 Fermi : durée d'une époque et poids d'un checkpoint ★★ 20 min
- [ ] 23.9 🗣️ Pourquoi le prétraitement doit voyager avec le modèle ★ 10 min
- [ ] 23.10 ⚖️ Réutiliser un modèle pré-entraîné : licence, données d'origine, biais hérités ★★ 20 min
- [ ] 23.11 📄 Lire Krizhevsky et al. (2012) : la recette d'entraînement d'AlexNet ★★★ 45 min
- [ ] 23.12 📦 Environnement et tenseurs : versions, device, reshape, view, permute ★★ 20 min
- [ ] 23.13 📦 Charger et regarder Fashion-MNIST ★★ 20 min
- [ ] 23.14 🔮 Oublier la conversion en float ou la normalisation : erreur ou entraînement lent ? ★ 10 min
- [ ] 23.15 📦 Tout le prétraitement au même endroit : split stratifié, float32, normalisation, DataLoader ★★ 30 min
- [ ] 23.16 📦 Le MLP du livre avec nn.Sequential et nn.Flatten ★ 15 min
- [ ] 23.17 🐛 Trois bugs d'inférence : eval() oublié, gradients calculés pour rien (inference_mode), chargement non sûr (weights_only) ★★ 25 min
- [ ] 23.18 📦 Ta fonction fit() : boucle, validation et historique ★★★ 45 min
- [ ] 23.19 📈 Lire l'historique : 3 époques, puis 50 ★★ 20 min
- [ ] 23.20 📦 Analyser les erreurs : matrice de confusion et grille des confusions ★★ 25 min
- [ ] 23.21 🔮 Prédire la réaction du MLP à des images « du monde réel » ★ 10 min
- [ ] 23.22 📦 Prédire sur de nouvelles images : eval, inference_mode, probabilités ★★ 25 min
- [ ] 23.23 📦 Reprendre un entraînement interrompu : checkpoint complet (modèle, optimiseur, époque, générateurs aléatoires) ★★ 30 min
- [ ] 23.24 📦 Un modèle pré-entraîné torchvision : transforms, gel et nouvelle tête ★★★ 40 min
- [ ] 23.25 📦 Les callbacks en PyTorch : meilleur checkpoint, planning de LR, early stopping ★★★ 45 min
- [ ] 23.26 🔬 Planning de learning rate : constant, par paliers, cosinus ou plateau ★★ 30 min
- [ ] 23.27 🔬 Patience de l'early stopping : 1, 3 ou 10 ★★ 20 min
- [ ] 23.28 🛠️ Rédiger la fiche du modèle (model card) ★★ 20 min
- [ ] 23.29 🏆 Défi reproductibilité : un MLP Fashion-MNIST à au moins 88 % dont l'accuracy varie de moins de 0,5 point sur 5 graines, en moins de 3 min de CPU ★★★★ 90 min
- [ ] 23.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (22 cartes)
<!-- wb:end 23 -->

<!-- wb:section 24 -->
## 24 — PyTorch en pratique 2 : améliorer, chercher, CNN et RNN ⏱️ 28 h

- [ ] 24.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] 24.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 24.1 ✏️ Le coût d'une grille (27 hyperparamètres, 54 entraînements) contre une recherche aléatoire : probabilité de tomber dans les 5 % meilleurs réglages en n essais ★ 10 min
- [ ] 24.2 ✏️ Formes et paramètres du CNN à pooling du livre ★★ 20 min
- [ ] 24.3 ✏️ Stride ou pooling : compter les multiplications ★★ 20 min
- [ ] 24.4 ✏️ Ce que renvoie nn.LSTM : output, h_n et c_n ★★ 15 min
- [ ] 24.5 ✏️ Fenêtres, pas d'optimisation et durée : de 200 à 20 000 points ★ 10 min
- [ ] 24.6 ∂ Dropout inversé et max-norm : espérance et projection ★★ 25 min
- [ ] 24.7 🧮 Fermi : le budget d'une recherche d'hyperparamètres sur CPU, sur GPU et en précision mixte ★★ 20 min
- [ ] 24.8 🗣️ La data augmentation en cinq lignes ★ 10 min
- [ ] 24.9 ⚖️ Une seule graine suffit-elle ? Reproductibilité et honnêteté des comparaisons d'hyperparamètres ★★ 20 min
- [ ] 24.10 📄 Lire He et al. (2016) : les réseaux résiduels ★★★ 45 min
- [ ] 24.11 🔬 Changer un seul hyperparamètre : la taille de batch ★★ 30 min
- [ ] 24.12 🔮 Prédire le classement de quatre MLP (784, 784 + 784, 64, 32 + 32) ★★ 25 min
- [ ] 24.13 📦 Dropout, max-norm et learning rate trop grand ★★ 30 min
- [ ] 24.14 📈 Pourquoi la validation dépasse le train avec dropout ★ 10 min
- [ ] 24.15 🛠️ Journal d'expériences : un run, une ligne ★★ 20 min
- [ ] 24.16 📦 Un réseau PyTorch déguisé en estimateur scikit-learn ★★★ 50 min
- [ ] 24.17 📦 Validation croisée stratifiée avec normalisation dans un Pipeline ★★ 25 min
- [ ] 24.18 📦 Grille, hasard, puis zoom : chercher les hyperparamètres ★★★ 45 min 🚀
- [ ] 24.19 🐛 Un CNN traduit du Keras qui refuse de tourner ★★ 25 min
- [ ] 24.20 📦 Du CNN simple au CNN profond sur CIFAR-10 ★★★ 45 min 🚀
- [ ] 24.21 🔬 Batch norm, dropout ou les deux : précision et écart entre train et validation d'un CNN ★★★ 40 min 🚀
- [ ] 24.22 🔮 Quelles augmentations sont plausibles ? Chiffres, vêtements, animaux ★ 10 min
- [ ] 24.23 📦 Augmentation de données avec torchvision.transforms.v2 ★★★ 45 min 🚀
- [ ] 24.24 📦 Des images synthétiques générées à la volée ★★ 30 min
- [ ] 24.25 📦 Sommes de sinus et LSTM minuscule : fenêtres, scaler, RMSE ★★★ 45 min
- [ ] 24.26 🔬 Fenêtre, profondeur ou données : qu'est-ce qui aide vraiment ? ★★★ 45 min
- [ ] 24.27 📦 Séquences retournées, état porté entre batches, couche par pas de temps ★★★ 40 min
- [ ] 24.28 📦 Au-delà de nn.Sequential : branches, addition, couches partagées, gel ★★★ 45 min
- [ ] 24.29 📦 Précision mixte et torch.compile : mesurer avant de croire ★★ 25 min
- [ ] 24.30 📦 Générer du texte : embeddings, séquences décalées, mots contre caractères ★★★★ 100 min 🚀
- [ ] 24.31 🏆 Défi CIFAR-10 frugal : au moins 80 % de test accuracy avec moins de 100 000 paramètres ★★★★ 180 min 🚀
- [ ] 24.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 24 -->

<!-- wb:section CP5 -->
## CP5 — Checkpoint V — Architectures (CNN, RNN, PyTorch en pratique) ⏱️ 13 h

- [ ] CP5.1 🔮 Forme de sortie ou erreur ? Six extraits PyTorch à prédire sans les exécuter ★ 10 min
- [ ] CP5.2 ✏️ Formes, paramètres et champ récepteur d'un petit CNN pour CIFAR-10 ★★ 15 min
- [ ] CP5.3 ✏️ Un pas de LSTM à la main ★ 10 min
- [ ] CP5.4 🐛 Formes et batch_first : quatre bugs de CNN et de RNN (sur papier) ★ 10 min
- [ ] CP5.5 🔨 rnn_forward à partir d'une cellule fournie ★ 10 min
- [ ] CP5.6 📦 Petit CNN Fashion-MNIST avec early stopping et sauvegarde du meilleur modèle ★★ 20 min
- [ ] CP5.7 📈 Diagnostiquer quatre courbes d'entraînement ★ 10 min
- [ ] CP5.8 🧮 Fermi : paramètres et durée d'une époque d'un générateur de texte LSTM ★ 5 min
- [ ] CP5.9 🗣️ Choisir et justifier une architecture pour trois projets ★ 5 min
- [ ] CP5.10 ✏️ Retour sur les parties III et IV : Pipeline sans fuite et gradient d'une couche dense ★ 10 min
- [ ] Mini-projet MP5 — Classifieur d'images CIFAR-10 de bout en bout (≈ 10 h)
<!-- wb:end CP5 -->

<!-- wb:section 25 -->
## 25 — Autoencodeurs et VAE ⏱️ 28 h

- [ ] 25.Q1–Q12 🧠 Quiz (12 questions, 37 min)
- [ ] 25.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 25.1 ✏️ Compter les paramètres d'un autoencodeur à goulot ★ 10 min
- [ ] 25.2 ✏️ Mélanger deux cercles : pixels contre paramètres ★ 10 min
- [ ] 25.3 ✏️ Suivre les formes dans l'autoencodeur convolutif ★★ 15 min
- [ ] 25.4 ✏️ Retrouver 28×28 : padding et output_padding d'une convolution transposée ★★ 15 min
- [ ] 25.5 ∂ Dériver à travers un tirage : l'astuce de reparamétrisation ★★ 20 min
- [ ] 25.6 ✏️ Où tombent les tirages d'une gaussienne en grande dimension ★★ 25 min
- [ ] 25.7 ∂ La divergence KL entre N(μ, σ²) et N(0, 1) ★★★ 35 min
- [ ] 25.8 ∂ Un autoencodeur linéaire retrouve l'ACP (cas 2D guidé) ★★★ 40 min
- [ ] 25.9 🧮 Compression réelle : le décodeur n'est pas gratuit ★★ 15 min
- [ ] 25.10 🧮 Pourquoi diffuser dans un espace latent : facteur de compression et coût ★★ 15 min
- [ ] 25.11 🗣️ Le VAE expliqué à un ami en cinq lignes ★ 10 min
- [ ] 25.12 ⚖️ Deepfakes par autoencodeurs : un encodeur, deux décodeurs ★★ 20 min
- [ ] 25.13 📄 Lire Kingma et Welling (2013), « Auto-Encoding Variational Bayes » ★★★ 45 min
- [ ] 25.14 🔮 Prédire : un autoencodeur entraîné sur une seule image ★ 15 min
- [ ] 25.15 📦 Premier autoencodeur dense 784-20-784 en PyTorch ★★ 25 min
- [ ] 25.16 📦 Autoencodeur profond 512-256-128-20 ★★ 25 min
- [ ] 25.17 🔮 Prédire : ce qu'un autoencodeur de chiffres fait d'un vêtement ★★ 15 min
- [ ] 25.18 🎨 Reproduire la carte latente 2D et sa grille décodée (fig. 25.29-25.30) ★★ 30 min
- [ ] 25.19 🔬 Secouer les variables latentes : bruit global, une seule composante, latents aléatoires ★★ 25 min
- [ ] 25.20 🔨 Interpoler entre deux codes : linéaire et sphérique ★★ 20 min
- [ ] 25.21 🔬 Mélanger dans l'espace des pixels ou dans l'espace latent ★★ 25 min
- [ ] 25.22 🔬 Taille du goulot et profondeur : la courbe d'erreur de reconstruction ★★★ 40 min
- [ ] 25.23 🔬 L'autoencodeur linéaire face à l'ACP de scikit-learn ★★★ 40 min
- [ ] 25.24 📦 Autoencodeur convolutif à 147 latents ★★ 30 min
- [ ] 25.25 🐛 Le débruiteur qui apprenait le bruit ★★ 25 min
- [ ] 25.26 📦 Autoencodeur débruiteur (bruit gaussien, écrêtage) ★★ 30 min
- [ ] 25.27 🔬 Sur-échantillonnage explicite ou convolutions à pas : temps et qualité ★★★ 35 min 🚀
- [ ] 25.28 🔨 La reparamétrisation en NumPy : passe avant et passe arrière ★★ 25 min
- [ ] 25.29 🔨 Divergence KL gaussienne et son gradient ★★ 20 min
- [ ] 25.30 📦 Un VAE en PyTorch : reconstruction + KL ★★ 30 min
- [ ] 25.31 🔮 Prédire : décoder des tirages N(0, I) avec un AE puis avec un VAE ★★ 15 min
- [ ] 25.32 🔬 Un VAE est stochastique : sorties répétées, bruit relatif, interpolations ★★ 30 min
- [ ] 25.33 🎨 Reproduire le nuage latent d'un VAE 2D et la grille −3…3 (fig. 25.67-25.68) ★★ 30 min
- [ ] 25.34 📈 Diagnostiquer des courbes reconstruction/KL (effondrement du postérieur) ★★ 15 min
- [ ] 25.35 🛠️ Exporter le décodeur comme générateur autonome et le tester ★★ 20 min
- [ ] 25.36 🔬 β-VAE : échanger la netteté contre la régularité ★★★ 40 min
- [ ] 25.37 🔬 Détecter des anomalies par l'erreur de reconstruction ★★★ 45 min
- [ ] 25.38 🔨 Un VAE entièrement en NumPy avec mylearn ★★★★ 150 min
- [ ] 25.39 🏆 Défi débruitage Fashion-MNIST : le meilleur PSNR sous contrainte ★★★★ 120 min 🚀
- [ ] 25.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (25 cartes)
<!-- wb:end 25 -->

<!-- wb:section 26 -->
## 26 — Apprentissage par renforcement ⏱️ 29 h

- [ ] 26.Q1–Q12 🧠 Quiz (12 questions, 37 min)
- [ ] 26.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 26.1 ✏️ Retours total et actualisé d'une partie de six coups ★ 10 min
- [ ] 26.2 ✏️ Taille des tables : de Flippers au Go ★ 10 min
- [ ] 26.3 ✏️ Une mise à jour de Q-learning à la main ★ 10 min
- [ ] 26.4 ✏️ La même transition vue par SARSA ★ 10 min
- [ ] 26.5 ✏️ Probabilités d'action : ε-greedy (standard et version du livre) et softmax ★★ 20 min
- [ ] 26.6 ✏️ Regarder la valeur remonter une chaîne d'états ★★ 20 min
- [ ] 26.7 ∂ Somme géométrique et horizon effectif de γ ★★ 20 min
- [ ] 26.8 ∂ Équation de Bellman : le point fixe que vise le Q-learning ★★★ 35 min
- [ ] 26.9 🧮 Mémoire d'une table Q : morpion 4×4, 5×5 et Go ★★ 15 min
- [ ] 26.10 🗣️ Attribution du crédit et exploration expliquées en cinq lignes ★ 10 min
- [ ] 26.11 ⚖️ Quand l'agent optimise la mauvaise récompense ★★ 20 min
- [ ] 26.12 📄 Lire Mnih et al. (2015), « Human-level control through deep reinforcement learning » ★★★ 45 min
- [ ] 26.13 📦 Prendre en main gymnasium avec FrozenLake ★ 15 min
- [ ] 26.14 🔨 Retours actualisés et retours à venir ★ 15 min
- [ ] 26.15 🎨 Reproduire les courbes d'actualisation (fig. 26.13-26.14) ★ 15 min
- [ ] 26.16 🔮 Prédire l'effet du camion sur la table L ★ 10 min
- [ ] 26.17 🔬 Flippers : la longueur optimale de chaque partie par recherche exhaustive ★★ 25 min
- [ ] 26.18 🔬 L-learning sur Flippers, sans puis avec camion ★★★ 40 min
- [ ] 26.19 🔨 Politiques gloutonne et softmax (ε-greedy importée du ch. 11) ★★ 20 min
- [ ] 26.20 🔨 Mises à jour de Q-learning et de SARSA, états terminaux compris ★★ 25 min
- [ ] 26.21 🔨 La boucle d'entraînement tabulaire et l'évaluation d'une politique ★★★ 45 min
- [ ] 26.22 📦 Q-learning sur FrozenLake, glissant ou non ★★ 25 min
- [ ] 26.23 📈 Lire des courbes d'apprentissage RL bruitées ★★ 15 min
- [ ] 26.24 🐛 L'agent qui n'apprenait rien ★★ 25 min
- [ ] 26.25 🔬 Q-learning sur Flippers avec camion : 300, 3 000 puis 6 000 parties ★★★ 40 min
- [ ] 26.26 🔨 Itération sur les valeurs : la référence quand le modèle est connu ★★★ 40 min
- [ ] 26.27 🔬 α, γ, ε : balayage d'hyperparamètres sur Flippers bruité ★★★ 45 min
- [ ] 26.28 🔮 Prédire le chemin appris par SARSA et par Q-learning au bord de la falaise ★ 10 min
- [ ] 26.29 🔬 Q-learning contre SARSA sur Flippers et CliffWalking ★★★ 45 min
- [ ] 26.30 🎨 Reproduire les comparaisons au fil de l'entraînement (fig. 26.58-26.60) ★★★ 35 min
- [ ] 26.31 🔨 Une mémoire de rejeu (replay buffer) ★★ 20 min
- [ ] 26.32 🔬 Concevoir la récompense : finale, pénalité par coup ou façonnée ★★ 30 min
- [ ] 26.33 🔬 Rejouer l'expérience pour apprendre avec moins d'épisodes ★★★ 35 min
- [ ] 26.34 🛠️ Rapporter un résultat RL honnête : plusieurs graines, configuration, intervalle ★★ 25 min
- [ ] 26.35 📦 Morpion contre un adversaire aléatoire : masque d'actions ou pénalité ★★★ 45 min
- [ ] 26.36 📦 CartPole : discrétiser un état continu pour un Q-learning tabulaire ★★★ 45 min
- [ ] 26.37 🔬 Auto-apprentissage au morpion : l'agent joue les deux camps ★★★★ 90 min
- [ ] 26.38 🏆 Défi CartPole tabulaire : tenir 195 pas en moyenne ★★★★ 120 min
- [ ] 26.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (30 cartes)
<!-- wb:end 26 -->

<!-- wb:section 27 -->
## 27 — Réseaux antagonistes génératifs (GAN) ⏱️ 20 h

- [ ] 27.Q1–Q10 🧠 Quiz (10 questions, 30 min)
- [ ] 27.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 27.1 ✏️ Dimensions d'un DCGAN pour MNIST : 6 272 neurones, tailles 7 → 14 → 28 et retour ★ 15 min
- [ ] 27.2 ✏️ Valeur minimax d'une petite matrice de gains ★ 10 min
- [ ] 27.3 ✏️ Pertes BCE du discriminateur et du générateur à partir de sorties données ★★ 15 min
- [ ] 27.4 ✏️ À l'équilibre, la perte vaut ln 2 et non 0,5 ★★ 15 min
- [ ] 27.5 ∂ Le discriminateur optimal D* = p_data / (p_data + p_g) ★★★ 30 min
- [ ] 27.6 ∂ Pourquoi log(1 − D(G(z))) sature et pas −log D(G(z)) ★★★ 30 min
- [ ] 27.7 🧮 Croissance progressive : où passe le temps de calcul ? ★★ 15 min
- [ ] 27.8 🗣️ Un GAN en cinq lignes, sans faux billets ★ 10 min
- [ ] 27.9 ⚖️ Visages qui n'existent pas, deepfakes et obligation de transparence ★★ 20 min
- [ ] 27.10 📄 Lire Goodfellow et al. (2014), « Generative Adversarial Nets » ★★★ 45 min
- [ ] 27.11 🔨 Distance de Wasserstein 1D entre deux échantillons ★ 15 min
- [ ] 27.12 📦 Un GAN minuscule pour une gaussienne 1D ★★ 30 min
- [ ] 27.13 🔮 Prédire la moyenne et l'écart-type générés au fil des époques ★★ 15 min
- [ ] 27.14 📈 Lire les courbes d'un GAN : pertes, D(x) et D(G(z)) ★★ 15 min
- [ ] 27.15 🎨 Reproduire l'apprentissage du nuage centré en (5, 5) (fig. 27.20-27.21) ★★ 30 min
- [ ] 27.16 🐛 Le GAN qui s'entraînait à l'envers ★★ 25 min
- [ ] 27.17 🔮 Prédire l'effet d'un discriminateur trop fort ★★ 15 min
- [ ] 27.18 🔬 Perte saturante ou non saturante : la dynamique du début ★★ 30 min
- [ ] 27.19 🔬 Le round en quatre temps du livre contre la boucle standard ★★★ 40 min
- [ ] 27.20 📦 Un DCGAN sur MNIST selon les règles de Radford ★★★ 45 min 🚀
- [ ] 27.21 🔬 Règles empiriques à l'épreuve : une architecture de 2016 contre celle de Radford ★★★ 45 min 🚀
- [ ] 27.22 🔨 Distance de Fréchet entre deux gaussiennes (la formule du FID) ★★★ 35 min
- [ ] 27.23 📦 Évaluer un générateur : distance de Fréchet sur les features d'un classifieur ★★★ 40 min
- [ ] 27.24 🔬 Provoquer puis soigner un effondrement de mode (anneau de 8 gaussiennes) ★★★ 45 min
- [ ] 27.25 📦 Un GAN conditionnel : choisir le chiffre généré ★★★ 45 min 🚀
- [ ] 27.26 🔬 Se promener dans l'espace latent du générateur ★★★ 30 min
- [ ] 27.27 🛠️ Checkpoint d'un GAN : deux réseaux, deux optimiseurs, états aléatoires, reprise à l'identique ★★★ 30 min
- [ ] 27.28 🏆 Défi DCGAN Fashion-MNIST : passer sous un seuil de distance de Fréchet ★★★★ 120 min 🚀
- [ ] 27.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (22 cartes)
<!-- wb:end 27 -->

<!-- wb:section 28 -->
## 28 — Applications créatives ⏱️ 19 h

- [ ] 28.Q1–Q10 🧠 Quiz (10 questions, 30 min)
- [ ] 28.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 28.1 ✏️ Tailles des activations et des matrices de Gram le long de VGG16 ★ 10 min
- [ ] 28.2 ✏️ Une matrice de Gram à la main ★★ 15 min
- [ ] 28.3 ∂ Mélanger les positions ne change pas la matrice de Gram ★★ 20 min
- [ ] 28.4 ✏️ Pertes de contenu, de style et de variation totale sur un mini-exemple ★★ 20 min
- [ ] 28.5 ∂ Le gradient de la perte de style ★★★ 35 min
- [ ] 28.6 🧮 Combien coûte un transfert de style sur CPU ? ★★ 15 min
- [ ] 28.7 🧮 Vocabulaire de mots et taille de la couche de sortie d'un LSTM ★ 10 min
- [ ] 28.8 🗣️ Pourquoi la matrice de Gram capture le style, en cinq lignes ★ 10 min
- [ ] 28.9 ⚖️ Imiter le style d'un artiste : droit d'auteur, domaine public, consentement ★★ 25 min
- [ ] 28.10 📄 Lire Gatys, Ecker et Bethge (2015), « A Neural Algorithm of Artistic Style » ★★★ 45 min
- [ ] 28.11 📦 Un extracteur multi-couches par hooks sur un réseau pré-entraîné gelé ★★ 25 min
- [ ] 28.12 🔮 Prédire l'image qui excite un filtre précoce, puis un filtre profond ★★ 15 min
- [ ] 28.13 📦 Visualiser un filtre d'un réseau ImageNet : la montée de gradient du ch. 21 passe à l'échelle ★★ 30 min
- [ ] 28.14 🎨 Une galerie couche par couche (fig. 28.7) ★★ 25 min
- [ ] 28.15 🔬 Régulariser la visualisation : décalages, flou, octaves ★★★ 40 min
- [ ] 28.16 📦 Deep dream sur une photo libre ★★★ 40 min 🚀
- [ ] 28.17 🔬 Choisir couches et poids, ne rêver que dans une zone ★★★ 30 min
- [ ] 28.18 📦 La matrice de Gram en PyTorch, vérifiée contre NumPy ★ 15 min
- [ ] 28.19 🔮 Prédire une reconstruction depuis le bruit : contenu profond seul, style seul ★ 10 min
- [ ] 28.20 📈 Lire les courbes de pertes d'un transfert de style ★★ 15 min
- [ ] 28.21 🐛 Le transfert de style qui ne produisait que du gris ★★ 25 min
- [ ] 28.22 🎨 Reconstructions de contenu et de style couche par couche (fig. 28.19, 28.23, 28.24) ★★★ 45 min 🚀
- [ ] 28.23 📦 Transfert de style complet sous budget CPU ★★★ 45 min 🚀
- [ ] 28.24 🔬 Poids contenu/style, variation totale et image de départ ★★★ 35 min 🚀
- [ ] 28.25 🔬 Au-delà de Gram : moyennes et écarts-types des canaux ★★★ 40 min
- [ ] 28.26 🛠️ Crédits, licences et paramètres : documenter une galerie générée ★★★ 30 min
- [ ] 28.27 📦 Plus de Verne : réutiliser le code du LSTM mot à mot de 24.30 sur Verne et l'entraîner plus longtemps, du charabia aux phrases ★★★ 45 min 🚀
- [ ] 28.28 🔬 Glouton, température, top-k ou top-p : répétitions et diversité ★★★ 30 min
- [ ] 28.29 🏆 Défi : un transfert de style en moins de 2 minutes de CPU, perte de style divisée par 10 et contenu reconnaissable ★★★★ 90 min
- [ ] 28.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (20 cartes)
<!-- wb:end 28 -->

<!-- wb:section 29 -->
## 29 — Datasets et préparation du projet final ⏱️ 9,4 h

- [ ] 29.Q1–Q8 🧠 Quiz (8 questions, 25 min)
- [ ] 29.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] 29.1 ✏️ Poids en mémoire de MNIST, Fashion-MNIST et CIFAR-10 ★ 10 min
- [ ] 29.2 ✏️ Découpage stratifié et référence « classe majoritaire » ★ 10 min
- [ ] 29.3 🧮 Télécharger Open Images ou LAION : stockage et temps ★★ 15 min
- [ ] 29.4 🗣️ Présenter son projet final en cinq lignes ★ 10 min
- [ ] 29.5 ⚖️ Retirés, restreints ou biaisés : que vérifier avant d'utiliser un dataset ? ★★ 25 min
- [ ] 29.6 📄 Lire Gebru et al., « Datasheets for Datasets » ★★★ 40 min
- [ ] 29.7 📦 Les datasets intégrés de scikit-learn, et une affirmation du livre à vérifier ★ 15 min
- [ ] 29.8 🔮 Prédire : le même modèle sur MNIST puis sur Fashion-MNIST ★ 10 min
- [ ] 29.9 🔮 Prédire la chute d'accuracy en changeant de source de chiffres ★ 10 min
- [ ] 29.10 📦 torchvision.datasets et Hugging Face datasets : mêmes données, deux API ★★ 25 min
- [ ] 29.11 📦 Explorer le Hugging Face Hub et OpenML : chercher, filtrer, lire la licence ★★ 25 min
- [ ] 29.12 📈 Profil express d'un dataset inconnu ★★ 20 min
- [ ] 29.13 🐛 Un score trop beau pour être vrai : doublons entre entraînement et test ★★ 25 min
- [ ] 29.14 🔬 Décalage de distribution : un modèle MNIST face aux chiffres de scikit-learn ★★★ 35 min
- [ ] 29.15 🛠️ Data card et script de téléchargement vérifié pour le projet final ★★★ 60 min
- [ ] 29.16 🏆 Le dataset piégé : trouver les huit défauts ★★★ 60 min
- [ ] 29.E1–E4 💼 Entretien (4 questions, 40 min)
- [ ] Flashcards importées dans Anki (15 cartes)
<!-- wb:end 29 -->

<!-- wb:section CP6 -->
## CP6 — Checkpoint VI — Génératif et apprentissage par renforcement ⏱️ 13 h

- [ ] CP6.1 🧠 Questions flash : dix réponses courtes sur la partie VI (valeurs, formes, vrai/faux) ★ 12 min
- [ ] CP6.2 ✏️ VAE : un tirage reparamétrisé et sa divergence KL ★ 10 min
- [ ] CP6.3 ✏️ Deux transitions : mise à jour Q-learning puis SARSA ★ 10 min
- [ ] CP6.4 ∂ Discriminateur optimal et pertes à l'équilibre ★ 12 min
- [ ] CP6.5 ✏️ Matrice de Gram de trois cartes d'activation ★ 6 min
- [ ] CP6.6 🔮 Prédire : γ → 0 puis ε = 0 pour un agent Q-learning sur Flippers ★ 5 min
- [ ] CP6.7 📈 Trois courbes, trois diagnostics : VAE, GAN, apprentissage par renforcement ★ 8 min
- [ ] CP6.8 📦 Écrire la perte d'un VAE en PyTorch et la tester sur un batch fourni ★★ 15 min
- [ ] CP6.9 🐛 Réparer une boucle d'entraînement de GAN ★ 10 min
- [ ] CP6.10 🧮 Table Q ou réseau : estimer la mémoire nécessaire ★ 5 min
- [ ] CP6.11 ⚖️ Un dataset de visages pour un produit commercial ★ 5 min
- [ ] CP6.12 💼 Comme en entretien : GAN, VAE ou diffusion, la réponse en 60 secondes ★ 6 min
- [ ] CP6.13 ✏️ Retour sur les parties IV et V : taille de sortie d'une convolution et un pas d'Adam ★ 10 min
- [ ] Mini-projet MP6 — Banc d'essai génératif : VAE contre GAN sur Fashion-MNIST (≈ 10 h)
<!-- wb:end CP6 -->

<!-- wb:section B1 -->
## B1 — Transfer learning et modèles pré-entraînés ⏱️ 19 h

- [ ] B1.Q1–Q11 🧠 Quiz (11 questions, 33 min)
- [ ] B1.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] B1.1 ✏️ Nouvelle tête : formes des tenseurs de l'image 32×32 redimensionnée jusqu'aux logits ★ 15 min
- [ ] B1.2 ✏️ k plus proches voisins cosinus sur six embeddings donnés ★ 10 min
- [ ] B1.3 ✏️ Learning rates discriminatifs et warmup : le learning rate de chaque groupe, pas à pas ★ 15 min
- [ ] B1.4 ✏️ Compter les paramètres entraînables : tête seule, dernier bloc, réseau entier ★★ 20 min
- [ ] B1.5 ✏️ Zéro-shot CLIP à la main : cosinus, logit_scale et softmax ★★ 20 min
- [ ] B1.6 ∂ Rester près des poids pré-entraînés : le gradient de la pénalité L2-SP ★★ 25 min
- [ ] B1.7 🧮 Fermi : pré-entraîner sur ImageNet ou fine-tuner sur 5 000 images (GPU-heures, énergie, coût) ★★ 20 min
- [ ] B1.8 🗣️ Le transfer learning expliqué en cinq lignes ★ 10 min
- [ ] B1.9 ⚖️ Biais et licences hérités d'un modèle pré-entraîné : un projet d'imagerie médicale ★★ 25 min
- [ ] B1.10 📄 Yosinski et al. (2014) : jusqu'où les features se transfèrent-elles ? ★★★ 45 min
- [ ] B1.11 📦 Trois modèles pré-entraînés en main : ResNet-18, MobileNetV3-small, DINOv2-small ★★ 20 min
- [ ] B1.12 🔮 Prédire : un ResNet ImageNet face à des images CIFAR-10 de 32×32 pixels ★ 10 min
- [ ] B1.13 📦 Extraire des embeddings figés, puis linear probe et k plus proches voisins ★★ 30 min
- [ ] B1.14 🔬 Résolution d'entrée : accuracy du linear probe contre temps de calcul (32 à 224 pixels) ★★★ 40 min
- [ ] B1.15 📈 Lire des projections PCA/UMAP : pixels bruts, ResNet et DINOv2 ★★ 20 min
- [ ] B1.16 🔬 Sonder chaque bloc : linear probe couche par couche, CIFAR-10 contre Fashion-MNIST ★★★ 45 min
- [ ] B1.17 📦 Fine-tuning partiel : dégeler le dernier bloc avec des learning rates par groupe ★★★ 45 min
- [ ] B1.18 🐛 Le fine-tuning qui détruit le modèle : quatre erreurs à trouver ★★ 25 min
- [ ] B1.19 📦 Fine-tuning complet : warmup + cosinus, early stopping, meilleur checkpoint ★★★ 50 min 🚀
- [ ] B1.20 🔮 Prédire : où le pré-entraînement rapporte-t-il le plus, 10 ou 1 000 images par classe ? ★ 10 min
- [ ] B1.21 🔬 Courbes d'apprentissage : from scratch, linear probe et fine-tuning de 10 à 1 000 images par classe ★★★ 60 min 🚀
- [ ] B1.22 🔬 Domaine lointain : transférer vers Fashion-MNIST, comparé au CNN from scratch ★★★ 45 min
- [ ] B1.23 🔬 Oubli catastrophique : que reste-t-il des prédictions ImageNet après le fine-tuning ? ★★ 25 min
- [ ] B1.24 📦 Zéro-shot avec CLIP sur CIFAR-10 : gabarits de prompts et ensembles de prompts ★★ 30 min
- [ ] B1.25 📦 Few-shot par prototypes de classe avec DINOv2 et CLIP (1 à 16 exemples par classe) ★★★ 40 min
- [ ] B1.26 🔬 Taille, latence et accuracy sur CPU : MobileNetV3-small, ResNet-18, DINOv2-small ★★ 30 min
- [ ] B1.27 🛠️ Publier un modèle fine-tuné : safetensors, configuration et model card qui cite le modèle de base ★★ 25 min
- [ ] B1.28 🏆 Défi : au moins 90 % sur le test CIFAR-10 avec 100 images par classe et 10 minutes de CPU ★★★★ 120 min
- [ ] B1.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (24 cartes)
<!-- wb:end B1 -->

<!-- wb:section B2 -->
## B2 — Tokenisation et embeddings ⏱️ 18 h

- [ ] B2.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] B2.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] B2.1 ✏️ « Noël à Paris » en UTF-8 : octets, points de code, longueur ★ 10 min
- [ ] B2.2 ✏️ Analogie à la main : roi − homme + femme sur des vecteurs de dimension 3 ★ 10 min
- [ ] B2.3 ✏️ Trois fusions BPE à la main sur un mini-corpus ★★ 20 min
- [ ] B2.4 ✏️ Encoder un mot jamais vu avec des fusions déjà apprises ★ 10 min
- [ ] B2.5 ✏️ One-hot fois matrice : nn.Embedding à la main, et quelles lignes reçoivent un gradient ★★ 15 min
- [ ] B2.6 ✏️ Co-occurrences et PPMI à la main (fenêtre de 1, quatre mots) ★★ 25 min
- [ ] B2.7 ∂ Skip-gram avec échantillonnage négatif : gradients par rapport au mot et au contexte ★★★ 35 min
- [ ] B2.8 🧮 Fermi : fertilité française et anglaise, pages dans 8 192 tokens, prix d'un roman traduit par API ★★ 15 min
- [ ] B2.9 🧮 Fermi : poids de la table d'embedding de GPT-2 et d'un LLM à 128 000 tokens ★ 10 min
- [ ] B2.10 🗣️ BPE en cinq lignes, sans jargon ★ 10 min
- [ ] B2.11 ⚖️ Une taxe cachée sur certaines langues, des stéréotypes dans les vecteurs : tokenisation et biais ★★ 25 min
- [ ] B2.12 📄 Sennrich, Haddow et Birch (2016) : les sous-mots pour traduire les mots rares ★★★ 45 min
- [ ] B2.13 📦 Tokenizers réels : GPT-2, BERT et un tokenizer multilingue sur Holmes et Verne ★★ 20 min
- [ ] B2.14 🔮 Prédire : quel texte coûte le plus de tokens, Holmes, Verne, des nombres ou des emoji ? ★ 10 min
- [ ] B2.15 🔨 get_pair_counts et merge_pair ★★ 20 min
- [ ] B2.16 🔨 BPETokenizer.fit : apprendre les fusions, au niveau caractère puis octet ★★★ 50 min
- [ ] B2.17 🔨 encode et decode : fusions appliquées par rang, aller-retour sans perte ★★★ 40 min
- [ ] B2.18 🐛 Le tokenizer qui abîme les accents : trois bugs d'Unicode et de fusions ★★ 25 min
- [ ] B2.19 📦 Ton BPE contre la bibliothèque tokenizers : mêmes fusions, autre vitesse ★★ 25 min
- [ ] B2.20 🔬 Taille du vocabulaire : compression et entropie par token, de 300 à 8 000 tokens ★★★ 40 min
- [ ] B2.21 🔬 Tokenizer entraîné sur Holmes, appliqué à Verne (et l'inverse) : le coût de la langue ★★ 25 min
- [ ] B2.22 🔨 cooccurrence_matrix et ppmi ★★ 30 min
- [ ] B2.23 🔨 cosine_similarity_matrix et most_similar ★★ 20 min
- [ ] B2.24 📦 Embeddings PPMI + SVD sur Holmes : les voisins de « Holmes », « Watson » et « letter » ★★ 30 min
- [ ] B2.25 📦 word2vec (skip-gram, échantillonnage négatif) en PyTorch sur Verne ★★★ 60 min
- [ ] B2.26 🔮 Prédire les voisins de « Fogg » et de « Passepartout » avant de les calculer ★ 10 min
- [ ] B2.27 📈 Lire une carte PCA/UMAP d'embeddings : groupes, axes et effet de la fréquence ★★ 20 min
- [ ] B2.28 📦 Embeddings de phrases (MiniLM, mean pooling) et recherche sémantique dans Holmes ★★ 30 min
- [ ] B2.29 🔬 Statique contre contextuel : le même mot dans deux phrases ★★ 25 min
- [ ] B2.30 🛠️ Sauvegarder un tokenizer en JSON et écrire son test de non-régression ★★ 20 min
- [ ] B2.31 🏆 Défi : la meilleure compression sans perte de Verne avec 2 000 tokens, en moins de 2 min de CPU ★★★★ 90 min
- [ ] B2.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (26 cartes)
<!-- wb:end B2 -->

<!-- wb:section B3 -->
## B3 — Attention et Transformers : un mini-GPT from scratch ⏱️ 22 h

- [ ] B3.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] B3.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] B3.1 ✏️ L'attention à la main : trois tokens, d_k = 2, scores, softmax et sortie ★★ 20 min
- [ ] B3.2 ✏️ Le même calcul avec un masque causal ★ 10 min
- [ ] B3.3 ✏️ Suivre les formes dans l'attention multi-têtes et compter ses paramètres ★★ 15 min
- [ ] B3.4 ✏️ Encodage sinusoïdal : quelques valeurs et l'effet d'un décalage de position ★★ 20 min
- [ ] B3.5 ✏️ Compter les paramètres d'un GPT et retrouver les 124 millions de GPT-2 small ★★ 25 min
- [ ] B3.6 ∂ Pourquoi √d_k : variance du produit scalaire de deux vecteurs aléatoires ★★ 25 min
- [ ] B3.7 ∂ RoPE : la rotation rend q_m · k_n fonction de m − n seulement ★★★ 35 min
- [ ] B3.8 🧮 Fermi : FLOPs d'entraînement (≈ 6 N D), ton mini-GPT sur CPU, GPT-3, et la mémoire du KV cache ★★ 20 min
- [ ] B3.9 🗣️ L'attention expliquée en cinq lignes ★ 10 min
- [ ] B3.10 ⚖️ Mémorisation : quand un modèle de langue recrache ses données d'entraînement ★★ 20 min
- [ ] B3.11 📄 Vaswani et al. (2017) : « Attention Is All You Need » ★★★ 45 min
- [ ] B3.12 🔨 causal_mask, padding_mask et scaled_dot_product_attention ★★ 30 min
- [ ] B3.13 🔮 Prédire la carte d'attention quand une clé copie la requête, puis quand on retire la division par √d_k ★ 10 min
- [ ] B3.14 🔨 split_heads, merge_heads et multi_head_attention ★★★ 45 min
- [ ] B3.15 🔨 positional_encoding et apply_rope ★★ 30 min
- [ ] B3.16 🔨 transformer_block en NumPy, validé contre nn.TransformerEncoderLayer ★★★★ 100 min
- [ ] B3.17 🐛 Trois masques qui mentent : convention inversée, masque transposé, padding compté dans la perte ★★ 25 min
- [ ] B3.18 📦 De NumPy à PyTorch : un module CausalSelfAttention avec F.scaled_dot_product_attention ★★ 30 min
- [ ] B3.19 📦 Mini-GPT caractère par caractère : embeddings de tokens et de positions, blocs pré-LN, tête liée ★★★ 60 min
- [ ] B3.20 📦 Entraîner le mini-GPT sur Holmes : AdamW, warmup + cosinus, clipping, perplexité, face au LSTM du ch. 22 ★★★ 60 min 🚀
- [ ] B3.21 📈 Lire les courbes d'entraînement du mini-GPT et poser un diagnostic ★★ 15 min
- [ ] B3.22 🔮 Prédire : le mini-GPT de Holmes amorcé avec une phrase de Verne ★ 10 min
- [ ] B3.23 📦 Générer vite : le KV cache, et le gain de vitesse mesuré selon la longueur ★★★ 45 min
- [ ] B3.24 📈 Les cartes d'attention de chaque tête sur une phrase de Holmes ★★ 20 min
- [ ] B3.25 🔬 Ablations : sans position, sans résiduel, post-LN, une seule tête ★★★ 60 min 🚀
- [ ] B3.26 🔬 Tokens BPE du B2 contre caractères : perplexité par caractère à budget égal ★★★ 45 min 🚀
- [ ] B3.27 🔬 Temps et mémoire de l'attention selon la longueur du contexte (64 à 2 048) ★★ 25 min
- [ ] B3.28 📦 Un Vision Transformer minuscule : Fashion-MNIST découpé en patchs ★★★ 45 min 🚀
- [ ] B3.29 🛠️ Un test pytest qui prouve la causalité : modifier le futur ne change pas les logits passés ★★ 20 min
- [ ] B3.30 🏆 Défi : la meilleure perplexité caractère sur Holmes avec au plus 1 M de paramètres et 10 min de CPU ★★★★ 120 min
- [ ] B3.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (28 cartes)
<!-- wb:end B3 -->

<!-- wb:section B4 -->
## B4 — LLM en pratique : Hugging Face, prompting, RAG et LoRA ⏱️ 21 h

- [ ] B4.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] B4.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] B4.1 ✏️ Mémoire d'un modèle : paramètres, dtype et KV cache — tient-il sur ton ordinateur ? ★ 15 min
- [ ] B4.2 ✏️ Recall@k et MRR pour cinq requêtes ★ 10 min
- [ ] B4.3 ✏️ LoRA : paramètres entraînables d'une couche 896×896 de rang 8, puis de tout un modèle ★ 15 min
- [ ] B4.4 ✏️ Un chat template à la main : rôles, jetons spéciaux et invite de génération ★★ 15 min
- [ ] B4.5 ✏️ BM25 à la main : trois documents, une requête de deux mots ★★ 20 min
- [ ] B4.6 ∂ LoRA : ΔW = BA de rang au plus r, B nulle au départ, gradients de A et de B, facteur α/r ★★★ 30 min
- [ ] B4.7 🧮 Fermi : coût et latence d'un assistant RAG, API facturée au token contre petit modèle local ★★ 20 min
- [ ] B4.8 🗣️ Le RAG expliqué en cinq lignes ★ 10 min
- [ ] B4.9 ⚖️ Un assistant RH qui répond sur les salaires : données personnelles, hallucinations, injection, transparence ★★★ 35 min
- [ ] B4.10 ⚖️ Fine-tuner sur des romans : domaine public, œuvres protégées et droit de retrait (opt-out) ★★ 20 min
- [ ] B4.11 📄 Hu et al. (2021) : « LoRA: Low-Rank Adaptation of Large Language Models » ★★★ 45 min
- [ ] B4.12 📦 Premier contact : AutoTokenizer, AutoModelForCausalLM, generate() et pipeline sur un petit modèle ★★ 25 min
- [ ] B4.13 🔮 Prédire : dix générations du même prompt à température 0, 0,7 et 1,5 ★ 10 min
- [ ] B4.14 📦 Chat template, message système, arrêt et streaming avec un modèle instruct ★★ 25 min
- [ ] B4.15 📦 Prompting pour extraire des informations de Holmes : zero-shot, few-shot, JSON validé, vote majoritaire ★★★ 45 min
- [ ] B4.16 🔬 Sensibilité au prompt : cinq formulations, trois graines, ordre des exemples ★★★ 40 min
- [ ] B4.17 🔨 chunk_text et BM25 : découper un roman et classer ses passages ★★★ 45 min
- [ ] B4.18 🔨 cosine_top_k, fusion de classements (RRF), recall@k et MRR ★★ 30 min
- [ ] B4.19 🔮 Prédire : sur des questions paraphrasées, BM25 ou recherche dense gagne-t-elle ? ★ 10 min
- [ ] B4.20 📦 Recherche dense (MiniLM, e5) contre BM25 contre hybride sur 30 questions annotées ★★★ 45 min
- [ ] B4.21 📦 Un RAG complet sur Holmes : récupération hybride, prompt avec citations, réponse d'un petit modèle ★★★ 60 min
- [ ] B4.22 🐛 Le RAG qui répond à côté : quatre défauts de découpage, d'encodage et de contexte ★★ 30 min
- [ ] B4.23 📈 Évaluer le RAG : recall@k, exactitude des réponses, « je ne sais pas » et analyse des échecs ★★ 20 min
- [ ] B4.24 🔬 Injection de prompt cachée dans un passage : attaquer, puis atténuer et mesurer ★★ 30 min
- [ ] B4.25 📦 LoRA from scratch : une couche LoRALinear en PyTorch, vérifiée contre peft ★★★ 40 min
- [ ] B4.26 📦 LoRA avec peft : adapter un petit modèle au style de Holmes, perplexité avant et après, fusion ★★★ 60 min 🚀
- [ ] B4.27 🔬 Rang, alpha et modules ciblés : perplexité contre paramètres entraînables ★★★ 45 min 🚀
- [ ] B4.28 📦 Classer des textes : encodeur fine-tuné par LoRA, LLM en zero-shot, TF-IDF + régression logistique ★★★ 60 min 🚀
- [ ] B4.29 🛠️ Rendre un pipeline LLM reproductible : révision du modèle figée, prompts versionnés, cache des réponses ★★ 25 min
- [ ] B4.30 🏆 Défi RAG : au moins 80 % de bonnes réponses citées sur 30 questions Holmes, modèle ≤ 0,6 B sur CPU ★★★★ 120 min
- [ ] B4.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (28 cartes)
<!-- wb:end B4 -->

<!-- wb:section B5 -->
## B5 — Modèles de diffusion : un DDPM minimal ⏱️ 21 h

- [ ] B5.Q1–Q11 🧠 Quiz (11 questions, 33 min)
- [ ] B5.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] B5.1 ✏️ Guidance sans classifieur : combiner les deux prédictions pour w = 0, 1 et 3 ★ 10 min
- [ ] B5.2 ✏️ Planning linéaire : β_t, α_t et ᾱ_t pour T = 4, puis ᾱ_T pour T = 1 000 ★★ 20 min
- [ ] B5.3 ✏️ Rapport signal sur bruit : à quel pas un chiffre devient-il illisible, planning linéaire ou cosinus ? ★★ 20 min
- [ ] B5.4 ✏️ Un pas DDPM à la main, en dimension 1, avec un bruit prédit donné ★★ 15 min
- [ ] B5.5 ✏️ Un pas DDIM à la main et l'image estimée x̂_0 ★★ 15 min
- [ ] B5.6 ∂ Forme fermée : deux pas de bruit gaussien n'en font qu'un, d'où q(x_t | x_0) ★★★ 35 min
- [ ] B5.7 ∂ Moyenne et variance du posterior q(x_{t−1} | x_t, x_0) en dimension 1 ★★★ 45 min
- [ ] B5.8 🧮 Fermi : coût d'échantillonnage, de MNIST sur CPU à une image 1024×1024, et ce que gagnent DDIM et l'espace latent ★★ 20 min
- [ ] B5.9 🗣️ La diffusion expliquée en cinq lignes ★ 10 min
- [ ] B5.10 ⚖️ Un générateur peut-il recopier ses images d'entraînement ? Mémorisation, droits et traçabilité ★★ 20 min
- [ ] B5.11 📄 Ho, Jain et Abbeel (2020) : « Denoising Diffusion Probabilistic Models » ★★★ 45 min
- [ ] B5.12 🔨 linear_beta_schedule, cosine_beta_schedule et noise_schedule ★★ 20 min
- [ ] B5.13 🔮 Prédire l'allure d'un chiffre bruité aux pas 100, 300 et 600, planning linéaire puis cosinus ★ 10 min
- [ ] B5.14 🔨 q_sample : bruiter un lot d'images à des pas différents ★★ 20 min
- [ ] B5.15 📈 Lire ᾱ_t, le rapport signal sur bruit et la grille de bruitage des deux plannings ★ 15 min
- [ ] B5.16 🔨 q_posterior_mean_variance et predict_x0_from_eps ★★ 30 min
- [ ] B5.17 🔨 ddpm_step et ddim_step ★★★ 40 min
- [ ] B5.18 🔬 Avec le vrai bruit, DDPM et DDIM retrouvent-ils x_0 ? Contrôle sur des données 2D ★★ 25 min
- [ ] B5.19 📦 Diffusion en 2D : un MLP débruiteur conditionné par le pas de temps, entraîné puis échantillonné ★★ 30 min
- [ ] B5.20 🔬 Flow matching en 2D : même MLP, cible « vitesse » et intégration d'Euler ★★★ 40 min
- [ ] B5.21 📦 Un U-Net minuscule en PyTorch : blocs convolutifs, connexions de saut, encodage du pas de temps ★★★ 50 min
- [ ] B5.22 🐛 Le DDPM qui ne produit que du gris : quatre bugs d'indices, de racines et d'échelle ★★ 25 min
- [ ] B5.23 📦 Entraîner un DDPM sur MNIST : boucle, moyenne exponentielle des poids (EMA), courbe de perte ★★★ 60 min 🚀
- [ ] B5.24 📦 Échantillonner : DDPM à 1 000 pas contre DDIM à 50 pas, qualité et temps ★★ 30 min
- [ ] B5.25 🔬 Nombre de pas DDIM, planning linéaire ou cosinus : distance de Fréchet contre temps de calcul ★★★ 45 min 🚀
- [ ] B5.26 🔮 Prédire l'effet de w = 0, 1, 3 et 7 sur la netteté et la diversité ★ 10 min
- [ ] B5.27 📦 Diffusion conditionnelle et guidance sans classifieur sur Fashion-MNIST ★★★ 60 min 🚀
- [ ] B5.28 🔬 Mémorisation : plus proches voisins des échantillons, entraînement sur 500 images contre 60 000 ★★★ 40 min 🚀
- [ ] B5.29 📦 Distance de Fréchet : situer le DDPM face au VAE (ch. 25) et au DCGAN (ch. 27) sur MNIST, avec le code d'évaluation du mini-projet MP6 ★★★ 30 min
- [ ] B5.30 🛠️ Un échantillonneur autonome : sample.py avec --steps, --seed et --guidance, images et paramètres sauvegardés ★★ 25 min
- [ ] B5.31 🏆 Défi : la plus petite distance de Fréchet sur Fashion-MNIST en au plus 50 pas d'échantillonnage ★★★★ 120 min 🚀
- [ ] B5.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (26 cartes)
<!-- wb:end B5 -->

<!-- wb:section B6 -->
## B6 — Explicabilité, équité et éthique ⏱️ 19 h

- [ ] B6.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] B6.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] B6.1 ✏️ Valeurs de Shapley exactes à la main : trois features, huit coalitions ★★ 25 min
- [ ] B6.2 ∂ Modèle linéaire et fond indépendant : φ_i = w_i (x_i − E[x_i]) ★★ 20 min
- [ ] B6.3 ✏️ Gradients intégrés à la main sur f(x) = x₁² + 3x₂ et vérification de la complétude ★★ 20 min
- [ ] B6.4 ✏️ Grad-CAM à la main : deux cartes 3×3 et leurs gradients ★★ 15 min
- [ ] B6.5 ✏️ Audit à la main : deux groupes, deux matrices de confusion, cinq indicateurs ★★ 20 min
- [ ] B6.6 ∂ Impossibilité : taux de base différents, calibration et égalité des erreurs incompatibles ★★★ 40 min
- [ ] B6.7 🧮 Fermi : coût de Shapley exact, de KernelSHAP et de TreeSHAP pour 10, 20 et 50 features ★★ 15 min
- [ ] B6.8 🗣️ Expliquer une valeur SHAP à un client dont le prêt est refusé, en cinq lignes ★ 10 min
- [ ] B6.9 ⚖️ Scoring de crédit : le code postal est-il un proxy ? Droits du client, obligations de l'entreprise ★★★ 35 min
- [ ] B6.10 ⚖️ Reconnaissance faciale : erreurs par sous-groupe et décision de déploiement ★★ 25 min
- [ ] B6.11 📄 Lundberg et Lee (2017) : « A Unified Approach to Interpreting Model Predictions » ★★★ 45 min
- [ ] B6.12 📦 Explications globales sur California : coefficients, permutation, dépendance partielle et ICE ★★ 30 min
- [ ] B6.13 🔮 Prédire la forme des courbes ICE du revenu médian et de la latitude ★ 10 min
- [ ] B6.14 🔨 shapley_values : Shapley exact par énumération des coalitions ★★★ 45 min
- [ ] B6.15 📦 SHAP sur un gradient boosting California : TreeExplainer, waterfall, beeswarm, dépendance ★★ 30 min
- [ ] B6.16 🔬 Le fond change tout : trois fonds différents, puis une feature dupliquée ★★★ 40 min
- [ ] B6.17 📦 Penguins : SHAP par classe pour une régression logistique et une forêt aléatoire ★★ 25 min
- [ ] B6.18 🔨 integrated_gradients et grad_cam ★★ 30 min
- [ ] B6.19 📦 Saillance, gradient × entrée et gradients intégrés sur un CNN CIFAR-10 ★★ 30 min
- [ ] B6.20 📦 Grad-CAM par hooks sur le ResNet fine-tuné du B1 : bonnes prédictions et erreurs ★★★ 45 min
- [ ] B6.21 🐛 Grad-CAM à l'envers : mauvaise couche, mauvaise classe, ReLU oubliée, carte mal alignée ★★ 25 min
- [ ] B6.22 🔬 Test de santé d'Adebayo : randomiser les poids couche par couche, la carte change-t-elle ? ★★★ 40 min
- [ ] B6.23 🔨 group_rates, demographic_parity_difference et equalized_odds_difference ★★ 25 min
- [ ] B6.24 📦 Audit d'équité sur Adult : indicateurs par sexe et par âge, intervalles de confiance bootstrap ★★★ 45 min
- [ ] B6.25 📈 Lire un tableau d'audit : où est l'écart, est-il significatif, que conclure ? ★★ 15 min
- [ ] B6.26 🔮 Prédire : retirer la variable « sexe » rend-il le modèle équitable ? ★ 10 min
- [ ] B6.27 🔬 Atténuer : retrait des proxys, repondération, seuils par groupe, et ce que cela coûte ★★★ 50 min
- [ ] B6.28 🛠️ Model card avec sections explicabilité et équité : indicateurs par groupe, limites, usage prévu ★★ 25 min
- [ ] B6.29 🏆 Défi : écart d'equalized odds sous 0,05 sur Adult, en perdant moins de 2 points d'accuracy ★★★★ 90 min
- [ ] B6.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (28 cartes)
<!-- wb:end B6 -->

<!-- wb:section B7 -->
## B7 — Du notebook à la production ⏱️ 18 h

- [ ] B7.Q1–Q11 🧠 Quiz (11 questions, 33 min)
- [ ] B7.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] B7.1 ✏️ Taille d'un artefact : float32, bfloat16, int8, et la mémoire au chargement ★ 10 min
- [ ] B7.2 ✏️ Cinq graines, un intervalle : la dispersion d'une accuracy rapportée ★★ 20 min
- [ ] B7.3 ✏️ Latence et débit : p50 et p95 sur vingt mesures, batch 1 contre batch 32 ★★ 15 min
- [ ] B7.4 ✏️ Indice de stabilité de population (PSI) à la main sur quatre intervalles ★★ 20 min
- [ ] B7.5 🧮 Fermi : servir 100 requêtes par seconde — cœurs CPU, regroupement en lots, coût mensuel ★★ 20 min
- [ ] B7.6 🗣️ « Ça marche dans mon notebook » : convaincre un manager d'industrialiser, en cinq lignes ★ 10 min
- [ ] B7.7 ⚖️ Journaliser les prédictions en production : minimisation, durée de conservation, droits des personnes ★★ 20 min
- [ ] B7.8 📄 Sculley et al. (2015) : « Hidden Technical Debt in Machine Learning Systems » ★★★ 45 min
- [ ] B7.9 📦 Du notebook au paquet : data.py, model.py, train.py, pyproject.toml et installation éditable ★★★ 60 min
- [ ] B7.10 📦 Une interface en ligne de commande : argparse, logging et codes de sortie ★★ 30 min
- [ ] B7.11 📦 Configuration : dataclass + YAML, surcharges en ligne de commande, validation ★★ 30 min
- [ ] B7.12 🔮 Prédire : même graine, puis num_workers = 4, puis GPU — résultats identiques ? ★ 10 min
- [ ] B7.13 🔬 Mesurer le non-déterminisme : graines fixées ou non, algorithmes déterministes, cinq exécutions ★★ 30 min
- [ ] B7.14 📦 Suivi d'expériences maison : identifiant de run, configuration, métriques et artefacts en JSONL ★★ 30 min
- [ ] B7.15 📦 Tests pytest pour le ML : formes, surapprentissage d'un mini-lot, prétraitement, schéma des données ★★★ 45 min
- [ ] B7.16 🐛 Le modèle qui marche dans le notebook et pas en production : cinq écarts à retrouver ★★ 30 min
- [ ] B7.17 📦 L'artefact du modèle : safetensors, configuration, normalisation, classes, version, empreinte des données ★★ 30 min
- [ ] B7.18 📦 Une API de prédiction avec FastAPI (repli : http.server) : /health, /predict, validation des entrées ★★★ 45 min
- [ ] B7.19 🔮 Prédire la latence p95 de l'API, requête par requête puis par lots de 32 ★ 10 min
- [ ] B7.20 🔬 Accélérer l'inférence sur CPU : lots, threads, inference_mode, bfloat16, quantification ★★ 30 min
- [ ] B7.21 📦 Exporter avec torch.export et vérifier l'équivalence numérique (ONNX en option) ★★ 25 min
- [ ] B7.22 📈 Surveiller la dérive : PSI et test KS sur des images assombries ou bruitées, quand alerter ? ★★ 25 min
- [ ] B7.23 🛠️ Intégration continue : un workflow GitHub Actions qui installe, teste et lance un entraînement fumée ★★ 25 min
- [ ] B7.24 📦 Lire et compléter un Dockerfile : image de base, dépendances CPU, utilisateur non root, variables ★★ 20 min
- [ ] B7.25 🏆 Défi : un dépôt prêt pour la production, de git clone à une prédiction servie en moins de 10 min de CPU ★★★★ 180 min
- [ ] B7.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (24 cartes)
<!-- wb:end B7 -->

<!-- wb:section B8 -->
## B8 — RL moderne : DQN, gradient de politique, PPO et RLHF ⏱️ 21 h

- [ ] B8.Q1–Q12 🧠 Quiz (12 questions, 36 min)
- [ ] B8.R1–R3 🔁 Rappels (3 questions, 15 min)
- [ ] B8.1 ✏️ Cible DQN à la main : état terminal, puis cible Double DQN ★ 15 min
- [ ] B8.2 ✏️ Perte de Huber et son gradient pour trois erreurs TD ★ 10 min
- [ ] B8.3 ✏️ Gradient du log d'une politique softmax à trois actions ★★ 15 min
- [ ] B8.4 ✏️ GAE à la main sur quatre pas, pour λ = 0, 0,95 et 1 ★★ 20 min
- [ ] B8.5 ✏️ Objectif clippé de PPO : quatre cas, valeur et gradient nul ou non ★★ 20 min
- [ ] B8.6 ✏️ Préférences : perte de Bradley-Terry puis perte DPO sur une paire de réponses ★★ 20 min
- [ ] B8.7 ∂ Le gradient de politique en une étape : l'astuce de la log-dérivée, puis la baseline sans biais ★★★ 40 min
- [ ] B8.8 🧮 Fermi : combien d'interactions ? CartPole, Atari (DQN 2015), AlphaGo Zero et le temps CPU correspondant ★★ 15 min
- [ ] B8.9 🗣️ PPO expliqué en cinq lignes ★ 10 min
- [ ] B8.10 ⚖️ RLHF : qui choisit les préférences ? Annotateurs, valeurs et flagornerie des modèles ★★ 25 min
- [ ] B8.11 📄 Schulman et al. (2017) : « Proximal Policy Optimization Algorithms » ★★★ 45 min
- [ ] B8.12 📦 CartPole pour le deep RL : espaces, politique aléatoire, evaluate_policy et la référence tabulaire du ch. 26 ★ 15 min
- [ ] B8.13 📦 DQN sur CartPole : réseau Q, ReplayBuffer de rl.py, réseau cible, Huber, ε décroissant ★★★ 60 min
- [ ] B8.14 🐛 Le DQN qui diverge : quatre erreurs de cible, de fin d'épisode et d'exploration ★★ 30 min
- [ ] B8.15 🔬 Ablations du DQN : sans réseau cible, sans rejeu, avec Double DQN — trois graines ★★★ 50 min 🚀
- [ ] B8.16 🔮 Prédire : REINFORCE sans baseline, avec baseline, avec retours normalisés — lequel apprend le plus vite ? ★ 10 min
- [ ] B8.17 📦 REINFORCE avec et sans baseline : variance du gradient et vitesse d'apprentissage ★★★ 45 min
- [ ] B8.18 🔨 compute_gae et explained_variance ★★ 30 min
- [ ] B8.19 🔨 clipped_surrogate : valeur et gradient par rapport au ratio ★★ 20 min
- [ ] B8.20 📦 PPO from scratch sur CartPole : environnements vectorisés, acteur-critique, GAE, clip, entropie ★★★★ 150 min
- [ ] B8.21 📈 Lire les journaux de PPO : KL approchée, fraction clippée, entropie, variance expliquée ★★ 20 min
- [ ] B8.22 🔬 Hyperparamètres de PPO : ε du clip, λ, nombre d'époques — robustesse sur trois graines ★★★ 45 min 🚀
- [ ] B8.23 🔮 Prédire : l'agent PPO face à une gravité doublée ou à un mât deux fois plus long ★ 10 min
- [ ] B8.24 🔬 Robustesse : modifier la physique de CartPole et mesurer la chute du retour ★★ 25 min
- [ ] B8.25 🔨 bradley_terry_loss et dpo_loss ★★ 20 min
- [ ] B8.26 📦 RLHF miniature : un modèle de récompense appris sur des préférences simulées, puis PPO — et le reward hacking ★★★★ 90 min
- [ ] B8.27 🛠️ Revue de code : comparer ton PPO à une implémentation de référence et documenter chaque écart ★★ 30 min
- [ ] B8.28 🏆 Défi : CartPole-v1 résolu (≥ 475 sur 100 épisodes) en au plus 100 000 pas, trois graines sur trois ★★★★ 120 min
- [ ] B8.E1–E5 💼 Entretien (5 questions, 50 min)
- [ ] Flashcards importées dans Anki (26 cartes)
<!-- wb:end B8 -->

<!-- wb:section PF -->
## PF — Projet final : un projet de bout en bout sur ton propre dataset ⏱️ 41 h

- [ ] PF.1 🗣️ Cadrage : problème, utilisateurs, métrique métier et métrique technique, critère de succès chiffré, risques ★★★ 90 min
- [ ] PF.2 🛠️ Données : data card complète, licence vérifiée, script de téléchargement reproductible (révision figée, SHA-256) ★★★ 90 min
- [ ] PF.3 📈 Exploration et audit de qualité : doublons, fuites, déséquilibre, valeurs manquantes, décalage entre découpages ★★★★ 180 min
- [ ] PF.4 ✏️ Protocole d'évaluation écrit avant toute expérience : découpage, métriques, test scellé, baseline naïve, budget ★★★ 60 min
- [ ] PF.5 📦 Baselines : prédicteur naïf et modèle classique dans un Pipeline sans fuite, en validation croisée ★★★★ 180 min
- [ ] PF.6 📦 Modèle deep learning adapté aux données : transfer learning, Transformer ou LLM avec LoRA, réseau séquentiel ou tabulaire ★★★★ 480 min 🚀
- [ ] PF.7 🔬 Expériences : journal, recherche d'hyperparamètres ciblée, ablations, trois graines ★★★★ 360 min 🚀
- [ ] PF.8 📈 Évaluation honnête sur le test scellé : une seule passe, intervalles bootstrap, analyse d'erreurs, sous-groupes, robustesse ★★★★ 180 min
- [ ] PF.9 ⚖️ Explicabilité et équité : attributions adaptées au modèle, audit par sous-groupe, limites et usages à proscrire ★★★★ 150 min
- [ ] PF.10 🛠️ Industrialisation : paquet installable, configuration, ligne de commande, tests pytest, intégration continue ★★★★ 240 min
- [ ] PF.11 📦 Démonstration : prédiction en ligne de commande et API HTTP (ou démo Gradio), latence mesurée ★★★★ 150 min
- [ ] PF.12 🛠️ Rapport et portfolio : README, model card, data card, figures, dépôt GitHub propre et reproductible ★★★★ 180 min
- [ ] PF.13 🗣️ Soutenance : dix minutes de présentation, cinq minutes de questions façon entretien, démonstration en direct ★★★★ 120 min
<!-- wb:end PF -->
