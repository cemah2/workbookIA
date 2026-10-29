# Mini-projet MP5 — Un classifieur d'images CIFAR-10 de bout en bout

> Le mini-projet du checkpoint V, à faire après l'examen blanc (environ 10 h). Il réunit les chapitres 21 à 24 dans un vrai projet de vision : un petit package PyTorch testé, des baselines, un pipeline de données sans fuite, une boucle d'entraînement réutilisable, des architectures comparées une variable à la fois, une recherche d'hyperparamètres, un modèle livré avec son prétraitement, une analyse d'erreurs honnête et une fiche modèle. Un projet de portfolio : la démarche compte autant que le score.

| | |
|---|---|
| **Problème** | classer des photos 32 × 32 en couleur dans dix classes (avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion) |
| **Données** | CIFAR-10 (Krizhevsky, 2009 : 50 000 images d'entraînement, 10 000 de test, 6 000 par classe ; pas de licence formelle, les auteurs demandent de citer leur rapport technique) ; la validation (5 000 images) se coupe dans les 50 000 |
| **Méthode imposée** | PyTorch et torchvision dans le package `cifarkit/` ; trois fonctions de ta librairie `mylearn` (« Ta librairie », plus bas) ; le test ouvert une seule fois, en mode complet |
| **Prérequis** | chapitres 21 à 24 ; le split stratifié (ch. 8 et 15) et la matrice de confusion (ch. 3) |
| **Évaluation** | sur 20, avec la grille ci-dessous |

## Le contexte

Un recruteur qui ouvre ton dépôt ne cherche pas le meilleur score sur CIFAR-10 (les publications dépassent 99 %) : il cherche quelqu'un qui sait mener un projet. Partir d'une baseline, ne changer qu'une chose à la fois, garder une trace de chaque essai, savoir pourquoi un modèle se trompe et livrer un modèle qu'un autre peut recharger sans le notebook. Ce projet t'y entraîne sur un problème classique, assez petit pour tourner sur un CPU en mode rapide, assez riche pour que les choix comptent : l'augmentation, la normalisation, l'architecture, le learning rate.

## L'architecture du code

Le package `cifarkit/` a quatre modules ; leurs docstrings sont le contrat (le kit de départ les donne).

| Module | Contenu | Ce qu'il appelle dans ta librairie |
|---|---|---|
| `pipeline.py` | `stratified_split`, `channel_stats`, `make_transform` (avec ou sans augmentation `transforms.v2`), `CifarDataset`, `make_loader` | `model_selection.train_test_split` |
| `models.py` | `n_parameters`, les baselines `mlp` et `small_cnn` (avec `flat_size`), `vgg` (pooling ou stride, batchnorm, dropout), `ResidualBlock`, `resnet`, `build_model` (un modèle d'après sa configuration) | `conv.conv_output_size` |
| `train.py` | `evaluate`, `train_one_epoch`, `make_optimizer`, `make_scheduler`, `fit` (meilleur checkpoint, early stopping, planning, reprise exacte), `save_model` et `load_model` (le modèle livré), `predict_proba`, `fgsm`, `log_run` (le journal) ; fournies : `quick_fit`, `read_runs`, `library_versions` | |
| `predict.py` | `load_image`, `classify` ; fournie : la ligne de commande `python -m cifarkit.predict IMAGE --model DOSSIER` | |

Le notebook appelle aussi `metrics.confusion_matrix` de ta librairie (MP5.6). Les fonctions qui se servent de `mylearn` l'importent à l'intérieur de leur corps : le modèle livré se recharge et classe des images sans ta librairie. Le fichier `data.py` (fourni) charge CIFAR-10 et n'ouvre le test qu'en le notant dans `test_log.json`.

## Le protocole

1. **La validation** : 5 000 images tirées des 50 000 par un split **stratifié** (500 par classe), une fois pour toutes, avec une graine écrite. Tous les choix se font sur elle.
2. **Le prétraitement** : la moyenne et l'écart-type de chaque canal s'apprennent sur les images d'entraînement seules, et sont enregistrés avec le modèle ; l'augmentation ne touche que l'entraînement.
3. **Les baselines d'abord** : prédire toujours la même classe (10 %), un MLP, un petit CNN. Un modèle qui ne bat pas le petit CNN n'apporte rien.
4. **Une variable à la fois** : chaque comparaison change une seule chose (le pooling contre le stride, la batchnorm, le dropout, les raccourcis), à budget égal ; **chaque run** (même raté) entre dans le journal `runs.csv`.
5. **Le test** (les 10 000 images de test) s'ouvre **une seule fois**, en mode complet, pour le modèle retenu ; chaque ouverture est notée dans `test_log.json`.
6. **Les graines** sont écrites : `torch.manual_seed` avant de construire chaque modèle, un `torch.Generator` par `DataLoader`.

## Les contraintes

- **Deux modes** : le mode rapide (`FAST_MODE = True`) tient en une dizaine de minutes sur un CPU à deux cœurs, sur des sous-ensembles ; le mode complet (45 000 images, 30 epochs pour le modèle final) se lance sur un GPU de Colab (🚀 : quelques minutes ; environ une heure et demie sur un CPU à deux cœurs).
- **Un package réutilisable** : le notebook n'a pas de boucle d'entraînement à lui ; il appelle `cifarkit`. Le modèle livré se recharge sans le notebook (`load_model`, `python -m cifarkit.predict`).
- **Reproductible** : graines et versions dans `results.json` et `model.json` ; la reprise d'un entraînement interrompu donne les mêmes poids, au bit près.
- **Lisible** : des docstrings, des annotations de type, des tests dont le message dit ce qui était attendu.

## Les livrables

1. Le notebook `mp5_classifieur_cifar10.ipynb` : propre, « Tout exécuter » sans erreur (moins de 10 minutes en mode rapide sur CPU), exécuté en mode complet pour les résultats, avec tes observations.
2. Le package `cifarkit/` (`pipeline.py`, `models.py`, `train.py` avec `fit()`, le meilleur checkpoint, l'early stopping et le planning de learning rate, `predict.py`) et `test_cifarkit.py`, au moins **trois** tests, qui attrapent au moins deux des cinq mutants de l'étape MP5.8.
3. Le journal `runs.csv` (au moins **8** runs, le notebook en écrit une vingtaine) et le tableau de synthèse du README.
4. Le modèle livré : `model.pt` (le `state_dict`) et `model.json` (la configuration, la moyenne et l'écart-type de normalisation, les classes, les métriques, les versions), et `python -m cifarkit.predict` qui classe une image.
5. Le **README de portfolio** (`README.md`) et la **fiche modèle** (`MODEL_CARD.md`) : leurs modèles sont dans le kit.
6. L'analyse d'erreurs : la matrice de confusion, les 20 erreurs les plus sûres d'elles commentées, l'accuracy sous l'attaque FGSM (`figures/`).

## Les étapes

| Étape | Ce que tu fais | Ce que tu produis | ⏱️ |
|---|---|---|---|
| MP5.1 | Cadrage, protocole de validation (split stratifié) et baselines (MLP puis petit CNN) | « L'objectif » et « Le protocole » du README, `stratified_split`, `mlp`, `small_cnn` | 60 min |
| MP5.2 | Pipeline de données : normalisation par canal apprise sur l'entraînement, augmentation `transforms.v2`, `Dataset` et `DataLoader` | `pipeline.py`, `figures/augmentation.png` | 60 min |
| MP5.3 | Boucle d'entraînement réutilisable : `fit()`, meilleur checkpoint, early stopping, planning de LR, reprise exacte, journal | `train.py` (première partie), les premiers runs du journal | 90 min |
| MP5.4 | Architectures : CNN type VGG puis blocs résiduels ; comparer stride et pooling, batchnorm et dropout | `vgg`, `ResidualBlock`, `resnet`, `build_model`, `figures/architectures.png` | 90 min |
| MP5.5 | Recherche d'hyperparamètres ciblée (aléatoire puis zoom) sur un sous-ensemble, puis l'entraînement complet (GPU Colab 🚀) ; le modèle livré | `save_model`, `load_model`, `model.pt`, `model.json`, `results.json`, `figures/courbes_final.png` | 90 min |
| MP5.6 | Analyse d'erreurs et robustesse : confusions, exemples mal classés, FGSM | `predict_proba`, `fgsm`, trois figures, tes commentaires | 60 min |
| MP5.7 | Comparaison avec un ResNet-18 pré-entraîné de torchvision (nouvelle tête seule) : coût, gain, limites | `resnet18_features`, la comparaison dans le README | 60 min |
| MP5.8 | Livraison : README, fiche modèle, `predict.py`, tests et mutants, nettoyage | `predict.py`, `test_cifarkit.py`, `README.md`, `MODEL_CARD.md` | 60 min |

## La grille d'évaluation (sur 20)

| Critère | Points | Tous les points | La moitié des points |
|---|---|---|---|
| **Pipeline de données** | 3 | split stratifié à graine écrite, statistiques de normalisation apprises sur l'entraînement seul et sauvegardées avec le modèle, augmentation sur l'entraînement seulement | une fuite (statistiques sur tout le dataset, augmentation de la validation) ou un prétraitement oublié dans le modèle livré |
| **Boucle d'entraînement** | 4 | graines, meilleur checkpoint, early stopping, planning compté en pas, reprise exacte vérifiée, `evaluate` en moyenne par exemple | un `fit()` sans reprise ou sans meilleur checkpoint |
| **Démarche expérimentale** | 4 | baselines, journal d'au moins 8 runs, une variable à la fois, règle de choix écrite avant de lire les chiffres, conclusions limitées à ce que les mesures permettent | des comparaisons à budgets différents, ou un journal incomplet |
| **Performance** | 3 | au moins **80 %** d'accuracy de test en mode complet (ou, sans GPU, au moins **65 %** de validation en mode rapide sur CPU), le test ouvert une seule fois | entre 70 et 80 % en mode complet (55 à 65 % en mode rapide) |
| **Analyse d'erreurs et robustesse** | 3 | matrice de confusion lue, 20 erreurs commentées (excusables ou non), FGSM mesuré et interprété, limites décrites honnêtement | des figures sans commentaire |
| **Qualité de livraison** | 3 | README qu'on comprend en deux minutes, fiche modèle, `predict.py` qui marche, code lisible, tests verts qui attrapent des mutants | un README sans chiffres ou un modèle qu'on ne sait pas recharger |

## Les extensions (pour aller plus loin)

- **Un Vision Transformer minuscule** entraîné depuis zéro, comparé au CNN à budget égal (lien avec le chapitre bonus B3).
- **Grad-CAM** sur les erreurs du modèle : où regarde-t-il quand il se trompe ? (lien avec B6).
- **Précision mixte et `torch.compile`** : mesurer le gain réel sur GPU, avant de le croire (24.29).
- **L'export** avec `torch.export` ou ONNX, et une démo Gradio (lien avec B7, qui part de ce projet).
- **La généralisation** sur CIFAR-10.1 ou sur tes propres photos réduites en 32 × 32 : combien de points perd le modèle hors de sa distribution ?
- **La même démarche pour une série temporelle** : prévoir les taches solaires à six mois avec un LSTM, face à la persistance (22.23).

## Ta librairie : les fonctions que le projet appelle

Le projet appelle trois fonctions de ta librairie `mylearn`. Les quatre parcours les écrivent toutes, dans les exercices ci-dessous. Avant chaque étape, vérifie que celles dont elle se sert passent leurs tests : le repli sur la référence ne remplace qu'un module **absent** de `mon_travail/mylearn/`. Dans un module que tu as copié, une fonction que tu n'as pas encore écrite lève `NotImplementedError`, et l'étape s'arrête.

| Étape | Fonction de `mylearn` | Exercice qui l'écrit |
|---|---|---|
| MP5.1 | `model_selection.train_test_split` (avec `stratify`) | 8.13 |
| MP5.1 | `conv.conv_output_size` | 21.12 |
| MP5.6 | `metrics.confusion_matrix` | 3.15 |

Pour vérifier une fonction, lance depuis la racine du dépôt les tests de son chapitre, filtrés sur son nom :

```bash
python -m pytest tests/test_ch21_conv.py -q -k "test_conv_output_size"
```

## Démarrer

```bash
python tools/start_chapter.py CP5
```

copie le kit de départ dans `mon_travail/projets/partie_5_classifieur_cifar10/` : `mp5_classifieur_cifar10.ipynb` (les huit étapes), le package `cifarkit/` (les classes, les signatures et les docstrings ; les fonctions à écrire lèvent `NotImplementedError`), `test_cifarkit.py` (un test d'exemple, à compléter), `data.py` (fourni : CIFAR-10 et le test qui s'ouvre une fois), `conftest.py` (fourni : pytest trouve ta librairie), et les modèles de `README.md` et de `MODEL_CARD.md`. Travaille **dans ce dossier-là** ; les tests se lancent depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_5_classifieur_cifar10 -q
```

CIFAR-10 se charge avec `data.load_cifar10()` (environ 170 Mo la première fois, puis le cache du workbook) ; le test, avec `data.open_test(dossier, description)`, à la toute fin. Pour le mode complet sur Colab : *Exécution → Modifier le type d'exécution → GPU*, puis `FAST_MODE = False` dans la cellule de setup ; enregistre ensuite ta copie et les fichiers produits (`results.json`, `runs.csv`, `model.pt`, `model.json`, `figures/`, `test_log.json`) dans ton dépôt.

Tu peux faire corriger ton projet par Claude avec le prompt P9 : il lit ton dossier, sans jamais y écrire, et le note avec cette grille.

## La solution de référence

Le dossier [`solution/`](solution/) contient un projet complet : le package, ses tests, le notebook exécuté (en mode rapide), et les résultats du mode complet (`results.json`, `runs.csv`, `model.pt`, `model.json`, les figures, `test_log.json`), avec un README de portfolio et une fiche modèle. **Ne l'ouvre qu'après avoir fini le tien** : c'est une façon de faire parmi d'autres, et tes chiffres n'ont pas à être les mêmes.
