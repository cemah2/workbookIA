# Données synthétiques (`wb.synth`)

Données générées à la demande, avec une graine (`seed`) pour être reproductibles. Aucune licence ni biais « du monde réel » : c'est justement l'intérêt, on contrôle tout (bruit, nombre de points, forme de la frontière).

| Fonction | Renvoie | Usage typique |
|---|---|---|
| `make_moons(n, noise, seed)` | `X (n, 2)`, `y` 2 classes | frontières non linéaires (ch. 7, 13, 16) |
| `make_circles(n, noise, factor, seed)` | `X`, `y` 2 classes | séparation impossible en ligne droite |
| `make_blobs(n, centers, std, n_features, seed)` | `X`, `y` k classes | k-means, classification simple (ch. 7) |
| `make_spirals(n, n_classes, noise, turns, seed)` | `X`, `y` | défi pour les réseaux (ch. 16-17) |
| `make_xor(n, noise, seed)` | `X`, `y` | le problème du XOR (ch. 10, 16) |
| `logic_gate(name)` | table de vérité `X (4, 2)`, `y` | perceptron, portes AND/OR/XOR/NAND/NOR (ch. 10) |
| `make_linear(n, w, b, noise, seed)` | `X`, `y` réels | boucle d'entraînement à la main (ch. 1) |
| `make_polynomial(n, coefs, noise, seed)` | `x`, `y`, fonction vraie `f` | overfitting et underfitting (figures du ch. 1) |
| `noisy_sine(n, freq, noise, t_max, trend, seed)` | `t`, `y` | séries temporelles contrôlées (ch. 22) |
| `rosenbrock(x, y)`, `rosenbrock_grad(x, y)` | valeur, gradient | surfaces et optimiseurs (ch. 5, 19) |
| `coin_flips(n, p, seed)` | tableau de 0/1 | probabilités, pièce truquée (ch. 2-4) |
| `gaussian_1d(n, mu, sigma, seed)` | échantillon 1D | GAN minimal (ch. 27) |

Le ch. 9 définit ses propres courbes dans le notebook (le tempo de la boutique, le vent au sommet d'une montagne, la fonction de 9.30) : leurs formules y sont lisibles. Le ch. 10 tire dans son notebook des points sur la sphère unité, à une distance contrôlée d'un hyperplan (`sphere_data_23`, la marge de 10.23).

Conventions : `X` en `float64` de forme `(n, n_features)`, `y` en `int64` de forme `(n,)`. Les signatures sont stables : les chapitres s'appuient dessus.
