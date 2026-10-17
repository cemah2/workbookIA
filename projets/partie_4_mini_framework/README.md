# Mini-projet MP4 — Un mini-framework de deep learning en NumPy, validé contre PyTorch

> Le mini-projet du checkpoint IV, à faire après l'examen blanc (environ 8 h). Il assemble ce que tu as écrit dans ta librairie aux ch. 16 à 20 en un petit framework orienté objet, `minifw`, prouve par des tests qu'il calcule ce que calcule PyTorch, à l'arrondi près, puis le met à l'épreuve sur Fashion-MNIST. Un projet de portfolio : un package propre, des tests qui attrapent des erreurs, des chiffres honnêtes.

| | |
|---|---|
| **Problème** | construire, en NumPy, les briques d'un framework de deep learning (couches, conteneur, losses, boucle d'entraînement) et montrer qu'elles sont justes, en les comparant à PyTorch à l'arrondi près |
| **Données** | deux demi-lunes synthétiques (1 000 points, pour la mise au point) ; Fashion-MNIST (Zalando, 60 000 images d'entraînement d'articles de mode (vêtements, chaussures, sacs) en 28 × 28, 10 classes, et 10 000 images de test ; licence MIT) |
| **Méthode imposée** | NumPy pour tout `minifw`, à partir de ta librairie `mylearn` (ch. 16 à 20) ; PyTorch comme oracle des tests et comme modèle de comparaison, jamais à l'intérieur de `minifw` |
| **Prérequis** | chapitres 16 à 20 ; les fonctions de `mylearn` listées plus bas dans « Ta librairie » |
| **Évaluation** | sur 20, avec la grille ci-dessous |

## Le contexte

PyTorch fait en une ligne ce que tu as écrit en plusieurs chapitres : `nn.Linear`, `loss.backward()`, `torch.optim.AdamW`. Un entretien d'ingénieur ML demande souvent de savoir ce qui se passe dessous, et un dépôt qui le montre vaut mieux qu'un discours. Ce projet range ton travail des ch. 16 à 20 dans l'architecture de PyTorch (des couches qui ont chacune leur passe avant et leur passe arrière, un conteneur qui les enchaîne, un optimiseur qui modifie les poids sur place) et la valide **contre PyTorch lui-même** : mêmes poids, mêmes entrées, mêmes sorties et mêmes gradients, à $10^{-6}$ près en valeur relative (et $10^{-8}$ en valeur absolue) en `float64`, des écarts réels de l'ordre de $10^{-16}$.

## L'architecture

Le package `minifw/` a quatre modules ; leurs docstrings sont le contrat (le kit de départ les donne).

| Module | Contenu | Ce qu'il appelle dans ta librairie |
|---|---|---|
| `layers.py` | `Layer` (la classe de base, donnée), `Dense`, `Activation` (`ReLU`, `Tanh`), `Dropout`, `BatchNorm1d` | `dense_forward`, `init_weights`, `dense_backward`, `get_activation`, `dropout_forward`, `dropout_backward`, `batchnorm_forward`, `batchnorm_backward` |
| `losses.py` | `SoftmaxCrossEntropy`, `MSELoss` : la loss moyenne et son gradient, sur les logits | `softmax_cross_entropy`, `mse_loss` |
| `model.py` | `Sequential` : passes avant et arrière enchaînées, `parameters`, `gradients`, `train`, `eval`, `n_parameters`, `state_dict`, `load_state_dict` | |
| `train.py` | `iterate_minibatches`, `evaluate`, `fit` (AdamW de `mylearn.optim`, warmup + cosinus, clipping, early stopping), `library_versions` | `cosine_lr`, `clip_grad_norm`, `EarlyStopping` ; l'optimiseur `AdamW` est passé à `fit` |

Le contrat de chaque couche : `forward(x)` calcule la sortie d'un mini-batch et garde ce que la passe arrière demandera ; `backward(dout)` reçoit $\partial L / \partial(\text{sortie})$, range les gradients des paramètres **dans** la couche et rend $\partial L / \partial x$ ; `parameters()` et `gradients()` rendent les tableaux **eux-mêmes**, dans le même ordre, si bien que `mylearn.optim.AdamW(model.parameters()).step(model.gradients())` met le modèle à jour sur place ; l'attribut `training` règle `Dropout` et `BatchNorm1d`.

## Le protocole

1. **Les tests d'abord** : chaque couche, la loss, le réseau entier et un pas d'AdamW sont comparés à PyTorch en `float64`, avec les mêmes poids (`nn.Linear` range $\mathbf{W}$ transposée). Tes tests doivent aussi **attraper des erreurs** : le notebook les relance contre six versions fautives de `minifw` (des mutants), et en attend au moins cinq attrapées.
2. **Un entraînement identique à PyTorch** : sans dropout (la seule source de hasard que les deux ne partagent pas), deux epochs de `minifw` et de son jumeau PyTorch, mêmes poids de départ, mêmes mini-batches, même planning, même clipping, donnent les mêmes poids à $10^{-6}$ près.
3. **Fashion-MNIST** : les 50 000 premières images d'entraînement pour apprendre, les 10 000 dernières pour la **validation** ; tous les choix (l'architecture, la régularisation, l'arrêt) se font sur la validation, avec une règle écrite avant de lire les chiffres.
4. **Le test** (les 10 000 images de test de Fashion-MNIST) s'ouvre **une seule fois**, en mode complet, pour la configuration retenue ; chaque ouverture est inscrite dans `test_log.json`.
5. **Les graines** sont écrites dans le notebook : tout tirage passe par un `np.random.default_rng(graine)`, jamais par l'état global de NumPy.

## Les contraintes

- **NumPy seulement dans `minifw`**, qui s'appuie sur ta librairie : il ne recode pas ce que `mylearn` fait déjà, il l'emballe. PyTorch n'apparaît que dans les tests et dans le notebook.
- **`float64` partout** dans `minifw` : c'est ce qui rend la comparaison à PyTorch exacte.
- **Reproductible** : graines et versions dans `results.json` ; le notebook s'exécute d'un bout à l'autre en une à deux minutes en mode rapide, environ cinq minutes en mode complet, sur un CPU à deux cœurs.
- **Lisible** : des docstrings, des annotations de type, pas de duplication entre les couches.

## Les livrables

1. Le package `minifw/` (`layers.py`, `losses.py`, `model.py`, `train.py`), avec docstrings et annotations de type.
2. `test_minifw.py` : au moins **dix** tests pytest : l'équivalence avec PyTorch (les sorties et les gradients de chaque couche, en entraînement et en évaluation, le réseau entier, un pas d'AdamW), et les propriétés que PyTorch ne vérifie pas pour toi (le masque du dropout, l'ordre des paramètres, `state_dict`, le planning de `fit`) ; ensemble, ils attrapent au moins cinq des six mutants.
3. `mp4_mini_framework.ipynb` : propre, exécuté de bout en bout en mode complet, avec tes observations : les lunes (la frontière de décision), Fashion-MNIST (les courbes, la matrice de confusion).
4. Un tableau de résultats, dans le README et dans `results.json` : minifw contre PyTorch (accuracy de validation et de test, temps par epoch, nombre de paramètres), l'ablation, les graines et les versions ; `test_log.json` (une seule ouverture du test).
5. Le **README de portfolio** (`README.md`, le modèle est dans le kit) : l'objectif, l'architecture du code, les tests, les résultats chiffrés, les figures légendées (`figures/moons_frontiere.png`, `figures/fashion_courbes.png`, `figures/fashion_confusion.png`), les limites, ce que tu as appris.

## Les étapes

| Étape | Ce que tu fais | Ce que tu produis | ⏱️ |
|---|---|---|---|
| MP4.1 | Lire ce cahier des charges ; la structure du package ; l'environnement et les graines | « L'objectif » du README, `library_versions` | 30 min |
| MP4.2 | Les couches `Dense`, `Activation`, `Dropout` et `BatchNorm1d`, sur ta librairie, comparées à PyTorch | `layers.py` | 90 min |
| MP4.3 | Les losses sur les logits et le conteneur `Sequential` ; un réseau entier comparé à PyTorch, un gradient vérifié par différences finies | `losses.py`, `model.py` | 60 min |
| MP4.4 | Tes tests d'équivalence avec PyTorch ; les six mutants | `test_minifw.py` | 90 min |
| MP4.5 | La boucle d'entraînement : mini-batches, AdamW, warmup + cosinus, clipping, early stopping ; les lunes | `train.py`, `figures/moons_frontiere.png` | 60 min |
| MP4.6 | Fashion-MNIST : un entraînement identique à PyTorch, minifw contre PyTorch, l'ablation, le choix, le test ouvert une fois | `build_mlp`, `FINAL`, `results.json`, deux figures | 105 min |
| MP4.7 | Le README, les figures, les limites ; le notebook relancé de zéro en mode complet ; commit | le projet complet | 45 min |

## La grille d'évaluation (sur 20)

| Critère | Points | Tous les points | La moitié des points |
|---|---|---|---|
| **Justesse du code** : tests d'équivalence avec PyTorch | 5 | toutes les couches (en entraînement et en évaluation), la loss, le réseau entier et un pas d'AdamW comparés à PyTorch, verts ; l'entraînement de deux epochs identique à PyTorch ; au moins cinq mutants sur six attrapés | des tests sur les sorties seulement, ou une couche sans test |
| **Qualité du code** | 4 | le contrat respecté, des docstrings et des annotations, aucune duplication entre les couches, des tests lisibles (un nom clair, un message qui dit l'attendu) | du code dupliqué ou des tests sans message |
| **Rigueur expérimentale** | 4 | graines écrites, validation séparée, courbes d'entraînement et de validation, comparaison à PyTorch à configuration égale, règle de choix écrite avant l'ablation, test ouvert une fois | un choix fait sur le test, ou une comparaison à réglages différents |
| **Résultats** | 3 | au moins **88 %** d'accuracy de test sur Fashion-MNIST avec `minifw` (en mode complet), et l'écart à PyTorch (accuracy, temps) expliqué | moins de 88 %, ou un écart à PyTorch non commenté |
| **Communication** | 4 | un README qu'on comprend en deux minutes, les figures légendées, des limites honnêtes, ce que tu as appris | un README sans chiffres ou sans limites |

## Les extensions (pour aller plus loin)

- **Une couche `Conv2d` par im2col** (un aperçu du ch. 21) : la convolution devient un produit matriciel ; compare-la à `nn.Conv2d`, sorties et gradients.
- **Un autograd vectoriel** : généralise la classe `Value` du ch. 18 à des tableaux NumPy, et retrouve les gradients de `Sequential` sans écrire de passe arrière.
- **Des connexions résiduelles et la layer norm** (`mylearn.nn.regularization.layer_norm`) dans `minifw`, pour entraîner un MLP de 20 couches (le ch. 20, 20.30).
- **La sauvegarde au format `.npz`** de `state_dict()`, avec l'état de l'optimiseur et des générateurs, et une reprise d'entraînement identique au bit près.
- **Le profilage** (`cProfile`) des passes avant et arrière, puis leur accélération (`float32`, moins de copies).

## Ta librairie : les fonctions que le projet appelle

Le projet appelle seize fonctions de ta librairie `mylearn`. Les quatre parcours les écrivent toutes, dans les exercices ci-dessous. Avant chaque étape, vérifie que celles dont elle se sert passent leurs tests : le repli sur la référence ne remplace qu'un module **absent** de `mon_travail/mylearn/`. Dans un module que tu as copié, une fonction que tu n'as pas encore écrite lève `NotImplementedError`, et l'étape s'arrête.

| Étape | Fonctions de `mylearn` | Exercices qui les écrivent |
|---|---|---|
| MP4.2 | `nn.layers.init_weights`, `nn.layers.dense_forward` | 16.14, 16.13 |
| MP4.2 | `nn.activations.get_activation` (qui rend `relu`, `tanh` et leurs dérivées) | 17.14 (17.12, 17.13) |
| MP4.2 | `nn.backward.dense_backward` | 18.15 |
| MP4.2 | `nn.regularization.dropout_forward`, `dropout_backward`, `batchnorm_forward`, `batchnorm_backward` | 20.16, 20.22 |
| MP4.3 | `nn.backward.softmax_cross_entropy`, `nn.backward.mse_loss`, `nn.backward.gradient_check` | 18.14 (et `log_softmax`, 17.15), 18.12, 18.16 |
| MP4.5 | `optim.AdamW`, `optim.cosine_lr`, `optim.clip_grad_norm`, `nn.regularization.EarlyStopping` | 19.18, 19.19, 20.17 |
| MP4.6 | `metrics.confusion_matrix` | 3.15 |

Pour vérifier une fonction, lance depuis la racine du dépôt les tests de son chapitre, filtrés sur son nom :

```bash
python -m pytest tests/test_ch20_regularization.py -q -k "test_batchnorm_ or test_dropout_"
```

## Démarrer

```bash
python tools/start_chapter.py CP4
```

copie le kit de départ dans `mon_travail/projets/partie_4_mini_framework/` : `mp4_mini_framework.ipynb` (les sept étapes), le package `minifw/` (les classes, les signatures et les docstrings ; les fonctions à écrire lèvent `NotImplementedError`), `test_minifw.py` (deux tests d'exemple, à compléter), `data.py` (fourni : les données et le test qui s'ouvre une fois), `conftest.py` (fourni : pytest trouve ta librairie), et le modèle de `README.md`. Travaille **dans ce dossier-là** ; les tests se lancent depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_4_mini_framework -q
```

Tant que `minifw` n'est pas écrit, les tests affichent « ⏳ pas encore écrit : … » avec le nom de la méthode qui manque : c'est normal. PyTorch est préinstallé sur Colab ; sur ton ordinateur, il est installé depuis la mise en place (`00_setup/INSTALL_LOCAL.md`, étape 5) ; sur un Mac Intel, où PyTorch ne s'installe plus, fais ce projet sur Colab.

Fashion-MNIST se charge avec `data.load_fashion()` (le téléchargement se fait une fois, puis depuis le cache du workbook) ; le test, avec `data.open_test(dossier, description)`, à la toute fin.

Tu peux faire corriger ton projet par Claude avec le prompt P9 : il lit ton dossier, sans jamais y écrire, et le note avec cette grille.

## La solution de référence

Le dossier [`solution/`](solution/) contient un projet complet : le package, 23 tests, le notebook exécuté (en mode rapide), `results.json` et les figures du mode complet, `test_log.json` (une seule ouverture du test) et un README de portfolio. **Ne l'ouvre qu'après avoir fini le tien** : c'est une façon de faire parmi d'autres, et tes chiffres n'ont pas à être les mêmes.
