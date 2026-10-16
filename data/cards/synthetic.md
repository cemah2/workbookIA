# Données synthétiques (`wb.synth`)

Données générées à la demande, avec une graine (`seed`) pour être reproductibles. Aucune licence ni biais « du monde réel » : c'est justement l'intérêt, on contrôle tout (bruit, nombre de points, forme de la frontière).

| Fonction | Renvoie | Usage typique |
|---|---|---|
| `make_moons(n, noise, seed)` | `X (n, 2)`, `y` 2 classes | frontières non linéaires (ch. 7, 13, 16) ; ensembles avec des colonnes de bruit ajoutées, frontières d'AdaBoost, AdaBoost à réparer (ch. 14) |
| `make_circles(n, noise, factor, seed)` | `X`, `y` 2 classes | séparation impossible en ligne droite, relevée en 3D (ch. 13) |
| `make_blobs(n, centers, std, n_features, seed)` | `X`, `y` k classes | k-means, classification simple (ch. 7) ; AdaBoost à trois classes qui se recouvrent (ch. 14) |
| `make_spirals(n, n_classes, noise, turns, seed)` | `X`, `y` | noyau RBF réglé par validation croisée (ch. 13), défi pour les réseaux (ch. 16-17) |
| `make_xor(n, noise, seed)` | `X`, `y` | le problème du XOR (ch. 10, 13, 16) |
| `logic_gate(name)` | table de vérité `X (4, 2)`, `y` | perceptron, portes AND/OR/XOR/NAND/NOR (ch. 10) |
| `make_linear(n, w, b, noise, seed)` | `X`, `y` réels | boucle d'entraînement à la main (ch. 1) |
| `make_polynomial(n, coefs, noise, seed)` | `x`, `y`, fonction vraie `f` | overfitting et underfitting (figures du ch. 1) |
| `noisy_sine(n, freq, noise, t_max, trend, seed)` | `t`, `y` | moyenne mobile (0B.35), séries temporelles contrôlées (ch. 22) |
| `rosenbrock(x, y)`, `rosenbrock_grad(x, y)` | valeur, gradient | surfaces et optimiseurs (ch. 5, 19) |
| `coin_flips(n, p, seed)` | tableau de 0/1 | probabilités, pièce truquée (ch. 2-4) |
| `gaussian_1d(n, mu, sigma, seed)` | échantillon 1D | GAN minimal (ch. 27) |

Le ch. 9 définit ses propres courbes dans le notebook (le tempo de la boutique, le vent au sommet d'une montagne, la fonction de 9.30) : leurs formules y sont lisibles. Le ch. 10 tire dans son notebook des points sur la sphère unité, à une distance contrôlée d'un hyperplan (`sphere_data_23`, la marge de 10.23). Le ch. 11 tire des points d'un arc du cercle unité, bruités (`arc_points_18`), joue des bandits de Bernoulli et gaussiens (`mylearn.bandit`), et compare des stratégies sur des bancs de bandits « à bande » dont les récompenses sont écrites d'avance (`TapeBandit` et `make_bench` de 11.26 : moyennes uniformes dans [0, 1], 10 bras, 1 000 pas).

Le ch. 14 simule dans son notebook des jurys de votants indépendants ou corrélés (`jury_13`).

Le ch. 15 se sert de `make_moons` (le code de 2018 à réparer en 15.12, les recherches en grille et au hasard de 15.18, 15.19 et 15.22, les frontières de 15.21) et de `make_spirals` (AdaBoost sur cinq spirales, 15.16), ainsi que des générateurs de scikit-learn, avec leurs graines dans le notebook : `make_blobs` (sept nuages pour le coude de k-means, 15.13), `make_moons` peu bruité (DBSCAN et clustering hiérarchique, 15.13), `make_circles` (découpage sans mélange en 15.11, carte de chaleur de `cv_results_` en 15.23) et `make_classification` (classe rare en 15.11, seuil de décision en 15.25, 2 % de positifs en 15.26). Il invente aussi un sinus bruité (15.9), un nuage aplati en 3D pour la PCA (15.15), 2 000 features de bruit pur (15.24) et trois surfaces de validation connues d'avance (🔮 15.20).

Conventions : `X` en `float64` de forme `(n, n_features)`, `y` en `int64` de forme `(n,)`. Les signatures sont stables : les chapitres s'appuient dessus.
