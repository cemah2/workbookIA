# Données synthétiques (`wb.synth`)

Données générées à la demande, avec une graine (`seed`) pour être reproductibles. Aucune licence ni biais « du monde réel » : c'est justement l'intérêt, on contrôle tout (bruit, nombre de points, forme de la frontière).

| Fonction | Renvoie | Usage typique |
|---|---|---|
| `make_moons(n, noise, seed)` | `X (n, 2)`, `y` 2 classes | frontières non linéaires (ch. 7, 13, 16) ; MLP sans activation contre MLP ReLU, quatre activations de scikit-learn (ch. 17) ; les 1 500 lunes standardisées du réseau 2-4-4-1 du livre : méthode très lente, trois learning rates, balayage sur cinq graines, `MLPClassifier` en descente de gradient stochastique, et 100 à 200 lunes pour le `MiniMLP` (ch. 18) ; 300 lunes pour batch, stochastique et mini-batch (ch. 19) ; ensembles avec des colonnes de bruit ajoutées, frontières d'AdaBoost, AdaBoost à réparer (ch. 14) |
| `make_circles(n, noise, factor, seed)` | `X`, `y` 2 classes | séparation impossible en ligne droite, relevée en 3D (ch. 13) |
| `make_blobs(n, centers, std, n_features, seed)` | `X`, `y` k classes | k-means, classification simple (ch. 7) ; AdaBoost à trois classes qui se recouvrent (ch. 14) |
| `make_spirals(n, n_classes, noise, turns, seed)` | `X`, `y` | noyau RBF réglé par validation croisée (ch. 13), défi pour les réseaux (ch. 16-17) : quatre activations de scikit-learn sur 1,5 tour, défi à 2,5 tours et 1 600 points (ch. 17) |
| `make_xor(n, noise, seed)` | `X`, `y` | le problème du XOR (ch. 10, 13) ; un réseau 2-4-1 aux neurones jumeaux, entraîné par différences finies (ch. 16) |
| `logic_gate(name)` | table de vérité `X (4, 2)`, `y` | perceptron, portes AND/OR/XOR/NAND/NOR (ch. 10) |
| `make_linear(n, w, b, noise, seed)` | `X`, `y` réels | boucle d'entraînement à la main (ch. 1) |
| `make_polynomial(n, coefs, noise, seed)` | `x`, `y`, fonction vraie `f` | overfitting et underfitting (figures du ch. 1) |
| `noisy_sine(n, freq, noise, t_max, trend, seed)` | `t`, `y` | moyenne mobile (0B.35), séries temporelles contrôlées (ch. 22) |
| `rosenbrock(x, y)`, `rosenbrock_grad(x, y)` | valeur, gradient | surfaces et optimiseurs (ch. 5) ; 100 pas de chaque optimiseur de ta librairie comparés à `torch.optim`, un Adam à réparer, une course de huit optimiseurs sur 13 learning rates (ch. 19) |
| `coin_flips(n, p, seed)` | tableau de 0/1 | probabilités, pièce truquée (ch. 2-4) |
| `gaussian_1d(n, mu, sigma, seed)` | échantillon 1D | GAN minimal (ch. 27) |
| `optimizer_curves(seed, n_updates)` | dictionnaire de six courbes `A` à `F` | courbes de loss par mise à jour, typiques d'optimiseurs et de plannings, à reconnaître (ch. 19, 19.23) |
| `training_curves(seed, n_epochs)` | dictionnaire de cinq historiques `A` à `E` (`train_loss`, `val_loss`, `val_acc`) | courbes d'entraînement et de validation par epoch : overfitting, underfitting, dropout, fuite de données, `model.eval()` oublié, à diagnostiquer (ch. 20, 20.25) |

Le ch. 9 définit ses propres courbes dans le notebook (le tempo de la boutique, le vent au sommet d'une montagne, la fonction de 9.30) : leurs formules y sont lisibles. Le ch. 10 tire dans son notebook des points sur la sphère unité, à une distance contrôlée d'un hyperplan (`sphere_data_23`, la marge de 10.23). Le ch. 11 tire des points d'un arc du cercle unité, bruités (`arc_points_18`), joue des bandits de Bernoulli et gaussiens (`mylearn.bandit`), et compare des stratégies sur des bancs de bandits « à bande » dont les récompenses sont écrites d'avance (`TapeBandit` et `make_bench` de 11.26 : moyennes uniformes dans [0, 1], 10 bras, 1 000 pas).

Le ch. 14 simule dans son notebook des jurys de votants indépendants ou corrélés (`jury_13`). Le ch. 16 tire ses poids avec `np.random.default_rng` (lois uniforme et normale, tableaux de 1 000 × 1 000 poids) et ses graphes orientés acycliques au hasard dans le notebook. Le ch. 17 tire dans son notebook des pré-activations gaussiennes (quatre couches de moyennes et d'écarts-types différents pour les histogrammes de saturation, 17.21), des logits de l'ordre de 1 000 pour les tests de stabilité (17.17) et un réseau de 10 couches de 64 neurones aux poids normaux (17.23).

Le ch. 18 fabrique dans son notebook six courbes de loss typiques (divergence, lenteur, plateau, erreur de signe, poids figés, entraînement sain : 18.22), un petit réseau tanh aux poids normaux pour une chasse aux gradients faux (18.17) et le réseau de quatre neurones de ✏️ 18.4, aux poids multiples de 1/4 (18.27).

Le ch. 19 définit dans son notebook la vallée $L(w) = 1 - e^{-8w^2}$, une gaussienne renversée (pas constant, momentum et figures du livre : 19.14, 19.20, 19.21) ; il reprend `make_moons` pour le banc d'essai du livre, 300 points standardisés et un réseau 2-12-13-13-2 (batch, stochastique, mini-batch : 19.25), et lit les courbes de `optimizer_curves` (19.23).

Le ch. 20 fabrique dans son notebook le tenseur `torch.arange(150.0)` de forme (2, 3, 5, 5) dont il faut prédire les formes (🔮 20.14), les poids exacts du petit réseau à dériver à la main (20.15), 200 images de bruit uniforme en 32 × 32 (`np.random.default_rng(0)`, 🔮 20.26) et le bruit gaussien de la couche `Noise20` (20.20) ; il lit les historiques de `training_curves` (20.25).

Le ch. 15 se sert de `make_moons` (le code de 2018 à réparer en 15.12, les recherches en grille et au hasard de 15.18, 15.19 et 15.22, les frontières de 15.21) et de `make_spirals` (AdaBoost sur cinq spirales, 15.16), ainsi que des générateurs de scikit-learn, avec leurs graines dans le notebook : `make_blobs` (sept nuages pour le coude de k-means, 15.13), `make_moons` peu bruité (DBSCAN et clustering hiérarchique, 15.13), `make_circles` (découpage sans mélange en 15.11, carte de chaleur de `cv_results_` en 15.23) et `make_classification` (classe rare en 15.11, seuil de décision en 15.25, 2 % de positifs en 15.26). Il invente aussi un sinus bruité (15.9), un nuage aplati en 3D pour la PCA (15.15), 2 000 features de bruit pur (15.24) et trois surfaces de validation connues d'avance (🔮 15.20).

Conventions : `X` en `float64` de forme `(n, n_features)`, `y` en `int64` de forme `(n,)`. Les signatures sont stables : les chapitres s'appuient dessus.
