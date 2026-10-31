"""Parts A and B of the chapter B1 notebook: statements and cells (used by build_chB1.py).

Parts C and D (B1.20 to B1.28) come next (build_chB1.NEXT_SESSION).
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part, guarded

import chB1_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB1.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}


def _cat(value, low, high):
    return "A" if value < low else ("B" if value <= high else "C")


def answers() -> dict:
    """The answers of the predictions, from the history (several seeds; checked by history_chB1.py)."""
    p12 = HISTORY.get("p12", {}).get("runs", [])
    probe32 = [r["probe32"] for r in p12] or [0.53]
    top5 = [r["top5_32"] for r in p12] or [0.16]
    cats_a = {_cat(v, 0.30, 0.70) for v in probe32}
    cats_b = {_cat(v, 0.30, 0.70) for v in top5}
    if len(cats_a) != 1 or len(cats_b) != 1:
        raise ValueError(f"the categories of 🔮 B1.12 change from one seed to another: {probe32}, {top5}")
    u15 = HISTORY.get("u15", {}).get("answers", {})
    return {"probe32": cats_a.pop(), "top5": cats_b.pop(), "separated": u15.get("separated", "C"),
            "pixel_classes": u15.get("pixel_classes", False), "vehicles": u15.get("vehicles", True)}


A = answers()
for _row in HISTORY.get("bug18", {}).get("versions", []):      # 🐛 B1.18 replayed on its 16 versions (history_chB1.py)
    if _row["first_failure"] != _row["expected"]:
        raise ValueError(f"🐛 B1.18: the version with the fixes {_row['fixed']!r} stops at the wrong point: {_row}")

RECORD_12 = (f'wb.record("B1.12a", "{A["probe32"]}", mistakes={{'
             '"regarde la ligne des 32 pixels : l\'accuracy du linear probe est-elle sous 30 % ?": "A", '
             '"regarde la ligne des 32 pixels : l\'accuracy du linear probe dépasse-t-elle 70 % ?": "C"})\n'
             f'wb.record("B1.12b", "{A["top5"]}", mistakes={{'
             '"regarde la ligne des 32 pixels : quelle part des images couvrent les 5 classes ImageNet les plus fréquentes ?": "B", '
             '"regarde la ligne des 32 pixels : combien de classes ImageNet différentes le réseau prédit-il ?": "C"})')

_SEP_HINTS = {"A": "compare les trois panneaux du bas : où les nuages d'une seule couleur sont-ils les plus nets ?",
              "B": "compare les panneaux UMAP de (B) et de (C) : lesquels ont les groupes les plus nets et les moins mêlés ?",
              "C": "compare les panneaux UMAP de (B) et de (C) : où les groupes d'une seule couleur sont-ils les plus nets ?"}
RECORD_15 = (
    f'wb.record("B1.15a", "{A["separated"]}", mistakes={{'
    + ", ".join(f'"{_SEP_HINTS[k]}": "{k}"' for k in "ABC" if k != A["separated"]) + "})\n"
    f'wb.record("B1.15b", {A["pixel_classes"]}, mistakes={{"regarde les couleurs de chaque groupe dans l\'UMAP de (A) : '
    f'un groupe y est-il fait d\'une seule couleur ?": {not A["pixel_classes"]}}})\n'
    f'wb.record("B1.15c", {A["vehicles"]}, mistakes={{"regarde où tombent les quatre couleurs des véhicules dans '
    f'l\'UMAP de (B) : à part des animaux, ou mêlées à eux ?": {not A["vehicles"]}}})\n'
    'wb.record("B1.15d", False, mistakes={"une projection en 2 dimensions garde peu de l\'information : compare au '
    'linear probe de B1.13, entraîné sur les 512 dimensions": True})')

SOLUTION_12 = f'probe32_12, top5_12 = "{A["probe32"]}", "{A["top5"]}"'
SOLUTION_15 = (f'separated_15, pixel_classes_15, vehicles_15, pca_says_15 = "{A["separated"]}", '
               f'{A["pixel_classes"]}, {A["vehicles"]}, False')

# ---------------------------------------------------------------------------------------------------------------------
PART_A = Part("A", "Charger et sonder : modèles pré-entraînés, embeddings figés, linear probe",
              "Trois modèles pré-entraînés et leur prétraitement, puis l'extraction de features : un backbone gelé "
              "transforme chaque image de CIFAR-10 en un embedding, calculé une fois, sur lequel on entraîne un linear "
              "probe ou un k plus proches voisins. Les cellules de vérification comparent ton code à un backbone de "
              "référence, `RESNET` (dans les outils : ResNet-18 gelé, `wb.transfer`), dont les embeddings sont "
              "calculés une seule fois puis rangés dans un cache sur le disque (`data/downloads/embeddings`, hors du "
              "dépôt) : relancer une cellule ne refait pas le calcul. Les images viennent de sous-ensembles **emboîtés "
              "et stratifiés** de CIFAR-10, à graine fixée : `X_TRAIN` (`PER_CLASS` images par classe), `X_VAL` "
              "(des images d'entraînement qui ne sont dans aucun sous-ensemble d'entraînement), `X_TEST` ; "
              "`nested(k)` donne la position, dans `X_TRAIN`, des $k$ images par classe d'un sous-ensemble plus petit. "
              "En mode rapide, ResNet-18 reçoit des images agrandies à `SIZE` = 96 pixels (224 en mode complet).",
              exercises=[
    Ex("B1.11", "📦", 2, 20, "Trois modèles pré-entraînés en main : ResNet-18, MobileNetV3-small, DINOv2-small",
       "charger trois modèles pré-entraînés avec leur prétraitement, et lire sur les objets ce que la documentation "
       "ne dit pas.",
       "ch. 23 (modèles de torchvision) · fiche §201.2 (encadrés 🧮 sur `weights.transforms()` et l'API transformers), "
       "§201.5 (la nouvelle tête)", tracks="R, M, C",
       body=r"""**Contrat** de `load_backbones_11()` : un dictionnaire de trois couples, chacun en mode `eval()` :
- `"resnet18"` et `"mobilenet_v3_small"` : `(modèle, prétraitement)`, le modèle de torchvision avec ses poids `IMAGENET1K_V1` (tête ImageNet comprise), le prétraitement `weights.transforms()` de ces poids ;
- `"dinov2_small"` : `(modèle, processeur)`, le backbone `AutoModel` du dépôt `DINOV2_NAME` (`facebook/dinov2-small`) à la révision `DINOV2_REVISION` (un commit fixé, dans les outils), et son `AutoImageProcessor` à la même révision.

Le premier appel télécharge les poids (≈ 45, 10 et 90 Mo), les suivants les relisent dans le cache.

Ensuite, **lis sur les objets** (dans une cellule à toi : `print(model)`, `print(preprocess)`, `model.config`, une passe avant) et reporte les **valeurs** :
a) la taille du vecteur à l'**entrée** de `mobilenet.classifier` ;
b) la taille du vecteur **entre** ses deux couches denses ;
c) `hidden_size` de DINOv2-small ;
d) `True` ou `False` : `pooler_output` est égal à `last_hidden_state[:, 0]` (essaie sur deux images de `X_TEST`, passées par le processeur) ;
e) `resize_size` du prétraitement de ResNet-18 ;
f) les lettres des modèles dont le prétraitement interpole en **bicubique** : (A) ResNet-18, (B) MobileNetV3-small, (C) DINOv2-small.

La cellule de vérification contrôle chaque valeur, puis compare tes trois modèles aux modèles de référence (poids, mode, prétraitement).

Dans tes notes : pour MobileNetV3-small, lequel des deux vecteurs de a) et b) prendrais-tu comme embedding, et pourquoi (pense à ce que la première couche dense a appris) ? Que contient la fiche modèle de `facebook/dinov2-small` sur le Hub (licence, données, usages) ?""",
       todo=C.TODO_11, check=C.CHECK_11, solution=C.SOLUTION_11 + "\n\n\n" + C.CHECK_11, record=C.RECORD_11),

    Ex("B1.12", "🔮", 1, 10, "Prédire : un ResNet ImageNet face à des images CIFAR-10 de 32×32 pixels",
       "prédire, en chiffres, ce que garde un ResNet entraîné sur des images de 224 pixels quand on lui donne des images "
       "de 32 pixels.",
       "Ex B1.11 · ✏️ B1.1 f) · fiche §201.5 (la résolution)", tracks="C",
       body=r"""ResNet-18 a appris sur des images de 224 × 224 ; celles de CIFAR-10 mesurent 32 × 32. Le ✏️ B1.1 f) t'a fait suivre la forme : à 32 pixels, la carte de `layer4` n'a plus qu'**une** position. La cellule d'expérience donne les images **telles quelles** (32 × 32, normalisées) au ResNet-18 de référence, puis agrandies à `SIZE` pixels, et mesure deux choses :
- l'accuracy d'un linear probe entraîné sur les embeddings de `X_TRAIN`, mesurée sur `X_TEST` ;
- les classes **ImageNet** que prédit la tête d'origine du réseau (ses 1 000 sorties) pour les images de test.

**Avant d'exécuter**, écris tes deux prédictions, **pour les images de 32 pixels** :
a) l'accuracy du linear probe : `"A"` moins de 30 %, `"B"` de 30 à 70 %, `"C"` plus de 70 % ;
b) la part des images de test dont la classe ImageNet prédite est l'une des **5 classes les plus fréquentes** parmi les prédictions : `"A"` moins de 30 %, `"B"` de 30 à 70 %, `"C"` plus de 70 %.

La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites (≈ 30 s la première fois : les embeddings sont calculés puis mis en cache).""",
       todo=C.TODO_12, solution=SOLUTION_12, hypothesis=True, record=RECORD_12,
       after=[("code", guarded(C.EXPERIMENT_12, ["probe32_12", "top5_12"],
                               "⏳ Ex B1.12 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : tes prédictions étaient-elles justes ? Compare les deux lignes (32 pixels et "
                     "`SIZE` pixels). Une seule position dans `layer4` : que reste-t-il pour le linear probe ? Les classes "
                     "ImageNet prédites ressemblent-elles aux classes de CIFAR-10 ?")]),

    Ex("B1.13", "📦", 2, 30, "Extraire des embeddings figés, puis linear probe et k plus proches voisins",
       "extraire les embeddings figés d'un backbone pré-entraîné, puis les classer par un linear probe et par les "
       "k plus proches voisins.",
       "Ex B1.11 · ch. 15 (Pipeline) · ch. 13 (k-NN) · ✏️ B1.2 · fiche §201.3", tracks="R, M, C",
       body=r"""**Contrat**
- `make_backbone_13()` : ton ResNet-18 de B1.11 (poids `IMAGENET1K_V1`), sa tête `fc` remplacée par `nn.Identity()`, en mode `eval()` : sa sortie est l'embedding $\phi(\mathbf{x})$, de taille 512 ;
- `embed_13(backbone, images, size, batch_size=64)` : pour des images `uint8` de forme `(N, 32, 32, 3)`, laissées intactes, un tableau NumPy `float32` de forme `(N, d)`, calculé par mini-batches de `batch_size`, sans gradient. Chaque image passe en `float` dans $[0, 1]$, canaux en premier, agrandie à `size` × `size` (interpolation bilinéaire, sans recadrage), puis normalisée par les moyennes et écarts-types d'ImageNet (`wb.datasets.IMAGENET_MEAN`, `IMAGENET_STD`, un par canal) ;
- `probe_13(E_train, y_train, E_test, y_test, C=1.0)` : l'accuracy de test (un `float`) d'un linear probe, une `StandardScaler` puis une `LogisticRegression(C=C, max_iter=2000)` ;
- `knn_13(E_train, y_train, E_test, y_test, k=10)` : l'accuracy de test des `k` plus proches voisins pour la similarité cosinus.

La cellule de vérification compare tes embeddings de 16 images à ceux de `RESNET`, vérifie qu'ils ne dépendent pas de la taille des mini-batches, puis lance ton probe et ton k-NN sur les embeddings en cache de `X_TRAIN` et de `X_TEST`, avec plusieurs réglages. La cellule suivante choisit `C` et `k` sur la **validation** et mesure le couple choisi une fois sur le test.

Dans tes notes : le linear probe et le k-NN se valent-ils ? Lequel préférerais-tu si des classes nouvelles arrivaient chaque semaine ? Pourquoi choisir `C` et `k` sur `X_VAL` et pas sur `X_TEST` ?""",
       todo=C.TODO_13, check=C.CHECK_13, solution=C.SOLUTION_13 + "\n\n\n" + C.CHECK_13,
       after=[("code", C.EXPERIMENT_13)]),

    Ex("B1.14", "🔬", 3, 40, "Résolution d'entrée : accuracy du linear probe contre temps de calcul (32 à 224 pixels)",
       "mesurer ce que la résolution d'entrée apporte au linear probe et ce qu'elle coûte en temps de calcul, pour en "
       "choisir une.",
       "Ex B1.13, B1.12 · fiche §201.5 (la résolution), §201.9 (mesurer une latence)", tracks="C",
       body=r"""**Contrat** de `resolution_sweep_14(backbone, sizes, train, test, repeats=3)` : `train` et `test` sont des couples `(images, labels)` ; pour chaque taille de `sizes`, dans l'ordre reçu, un dictionnaire `{"size": taille, "accuracy": ..., "ms_per_image": ...}` :
- `accuracy` : ton `probe_13` sur les embeddings de ton `embed_13` à cette taille ;
- `ms_per_image` : le temps de `embed_13` sur les 64 premières images de test à cette taille, la **plus petite** de `repeats` mesures (`time.perf_counter()`), divisée par 64, en millisecondes.

La cellule de vérification essaie ta fonction sur un petit jeu, avec un backbone dont le premier appel à chaque taille est artificiellement lent (un démarrage à froid). La cellule suivante balaie six tailles, de 32 à 224 pixels, avec `SWEEP_PER_CLASS` images par classe en entraînement et en test (≈ 45 s en mode rapide), puis trace l'accuracy selon la taille et selon le temps par image.

Dans tes notes : décris les deux courbes. Où l'accuracy cesse-t-elle de beaucoup monter ? Le temps suit-il le nombre de pixels ($(224/96)^2 \approx 5{,}4$) ? Quelle résolution choisirais-tu pour un service qui doit répondre en moins de 10 ms par image sur ce CPU, et laquelle pour la meilleure accuracy ? Pourquoi garder la plus petite des mesures, et pas la moyenne ?""",
       todo=C.TODO_14, check=C.CHECK_14, solution=C.SOLUTION_14 + "\n\n\n" + C.CHECK_14,
       after=[("code", C.EXPERIMENT_14)]),

    Ex("B1.15", "📈", 2, 20, "Lire des projections PCA/UMAP : pixels bruts, ResNet et DINOv2",
       "lire des projections en deux dimensions de trois représentations des mêmes images, et dire ce qu'elles "
       "montrent et ce qu'elles ne prouvent pas.",
       "Ex B1.13 · ch. 12 (PCA, UMAP) · fiche §201.3 (regarder et sonder), §201.8 (DINOv2)", tracks="C",
       body=r"""La cellule suivante projette en deux dimensions les mêmes 600 images de test (60 par classe), sous trois représentations : (A) les **pixels bruts** (3 072 valeurs), (B) les embeddings de **ResNet-18** (512), (C) ceux de **DINOv2-small** (384, images à `DINO_SIZE` pixels), chacune standardisée, puis réduite par une **PCA** (en haut) et par **UMAP** (en bas, `n_neighbors=15`, graine fixée). Une couleur par classe (≈ 1 min 30 s la première fois : les embeddings DINOv2 de tout `X_TEST` sont calculés, puis mis en cache pour la suite du notebook ; ≈ 25 s ensuite).

Lis la figure et écris :
a) la lettre de la représentation dont l'UMAP montre les groupes **les plus nets** : `"A"`, `"B"` ou `"C"` ;
b) `True` ou `False` : les groupes que l'on voit dans l'UMAP des pixels bruts (A) sont des classes ;
c) `True` ou `False` : dans l'UMAP de (B), les quatre véhicules (avion, automobile, bateau, camion) occupent une région à part des six animaux ;
d) `True` ou `False` : la PCA de (B) mêle les classes, donc un classifieur linéaire sur ces embeddings ne peut pas faire beaucoup mieux que le hasard.

Une fois tes quatre réponses écrites, la cellule de vérification affiche des mesures faites sur les projections : l'accuracy des 10 plus proches voisins **dans le plan** de chaque projection, et la part des images classées dans la bonne famille (véhicules ou animaux).

Dans tes notes : pourquoi la PCA des pixels ne sépare-t-elle presque rien (que varie-t-il le plus d'une photo à l'autre) ? Pourquoi UMAP sépare-t-il mieux que la PCA, et pourquoi ne faut-il pas lire les **distances entre** les groupes d'un UMAP (ch. 12) ?""",
       given=C.GIVEN_15, todo=C.TODO_15, check=C.CHECK_15, solution=SOLUTION_15, record=RECORD_15,
       after=[("code", C.MEASURES_15)]),

    Ex("B1.16", "🔬", 3, 45, "Sonder chaque bloc : linear probe couche par couche, CIFAR-10 contre Fashion-MNIST",
       "mesurer, bloc par bloc, ce qu'un backbone ImageNet garde d'utile pour un domaine proche et pour un domaine "
       "lointain.",
       "Ex B1.13, 📄 B1.10 · fiche §201.1 (hiérarchie des features), §201.3 (regarder et sonder), §201.5 (les canaux), "
       "§201.7 (distance de domaine)", tracks="M, C",
       body=r"""Le 📄 B1.10 a montré, sur ImageNet coupé en deux, jusqu'où les features se transfèrent. Ici, deux tâches : CIFAR-10 (des photos, proches d'ImageNet) et Fashion-MNIST (des vêtements en niveaux de gris sur fond noir, loin d'ImageNet). La cellule suivante charge les deux et donne `to_input(images, size)` (de `wb.transfer`) : la préparation de ton `embed_13`, qui accepte aussi des images en niveaux de gris `(N, H, W)`, dont le canal est répété trois fois après l'agrandissement (fiche §201.5).

**Contrat**
- `block_embeddings_16(model, images, size, blocks=RESNET_BLOCKS, batch_size=64)` : pour un ResNet-18 de torchvision en mode `eval()` et des images `uint8` (en couleur ou en niveaux de gris), un dictionnaire `{bloc: tableau float32 (N, C)}` : le **pooling moyen global** de la sortie de chaque bloc demandé, parmi `"stem"` (la sortie de `model.maxpool`), `"layer1"` … `"layer4"`. Chaque mini-batch est préparé par `to_input(batch, size)` et ne passe qu'**une fois** dans le réseau pour tous les blocs, sans gradient ;
- `probe_by_block_16(model, train, test, size, blocks=RESNET_BLOCKS)` : `{bloc: accuracy de test de ton probe_13 sur les embeddings de ce bloc}`.

La cellule de vérification compare tes embeddings à ceux de `RESNET` sur des images des deux datasets et compte les passes dans le réseau. La suivante sonde les cinq blocs sur les deux datasets, avec `BLOCK_PER_CLASS` images par classe (≈ 15 s en mode rapide), et trace les deux courbes.

Dans tes notes : sur CIFAR-10, quel bloc donne le meilleur probe ? Et sur Fashion-MNIST ? Compare l'écart entre le stem et le meilleur bloc sur chaque dataset : que dit-il de la distance de domaine ? Relie tes courbes au « général → spécifique » de Yosinski et coll. (📄 B1.10). Quelle couche choisirais-tu comme embedding pour une tâche loin d'ImageNet, avant tout fine-tuning ?""",
       given=C.GIVEN_16, todo=C.TODO_16, check=C.CHECK_16, solution=C.SOLUTION_16 + "\n\n\n" + C.CHECK_16,
       after=[("code", C.EXPERIMENT_16)]),
])

PART_B = Part("B", "Fine-tuner : le dernier bloc, puis tout le réseau",
              "On ne gèle plus tout le backbone : on dégèle le dernier bloc, puis tout le réseau, avec des learning "
              "rates par groupe, un planning et des batchnorms maîtrisées. En mode rapide : `FT_PER_CLASS` = 50 images "
              "par classe (500 images), 96 pixels, mini-batches de 32, `FT_EPOCHS` = 3 epochs ; un pas de fine-tuning "
              "complet de ResNet-18 coûte alors ≈ 0,6 s sur deux cœurs. 🚀 En mode complet (`FAST_MODE = False`) : "
              "1 000 images par classe à 224 pixels et 10 epochs, à lancer sur un GPU de Colab. La cellule suivante donne "
              "`batches(X, y, batch_size, seed)` (des mini-batches mélangés, préparés et normalisés), "
              "`evaluate(model, X, y)` (l'accuracy, en mode `eval()`) et `IMAGENET_RESNET18` (les poids pré-entraînés, "
              "pour les vérifications).", given=C.GIVEN_B, exercises=[
    Ex("B1.17", "📦", 3, 45, "Fine-tuning partiel : dégeler le dernier bloc avec des learning rates par groupe",
       "fine-tuner le dernier bloc d'un ResNet pré-entraîné avec des learning rates par groupe, en gardant les "
       "batchnorms sur leurs statistiques d'ImageNet.",
       "Ex B1.13 · ✏️ B1.3, B1.4 · ch. 23 (gel, nouvelle tête) · fiche §201.4", tracks="R, C",
       body=r"""**Contrat**
- `param_groups_17(modules, head_lr, gamma, weight_decay=0.01)` : les groupes de paramètres d'AdamW. `modules` est une liste de modules **de la tête vers l'entrée** (par exemple `[model.fc, model.layer4]`) ; le $k$-ième (à partir de 0) reçoit le learning rate `head_lr * gamma**k`. Chaque module donne au plus deux dictionnaires `{"params", "lr", "weight_decay"}` : ses paramètres de dimension 2 ou plus avec `weight_decay`, les autres (biais, paramètres des batchnorms) avec 0. Seulement les paramètres qui demandent un gradient, et aucun groupe vide ;
- `make_partial_17(num_classes=10)` : ResNet-18 `IMAGENET1K_V1`, une nouvelle tête `nn.Linear(512, num_classes)`, tous les paramètres gelés sauf ceux de `layer4` et de la tête ; sur `DEVICE` ;
- `freeze_batchnorm_17(model)` : met chaque batchnorm de `model` en mode `eval()`, sans changer le mode des autres modules ;
- `train_partial_17(model, X, y, epochs, head_lr, gamma, batch_size=32, seed=0)` : crée **un** `torch.optim.AdamW` sur `param_groups_17([model.fc, model.layer4], head_lr, gamma)`, puis, pour chaque epoch : `model.train()`, puis `freeze_batchnorm_17(model)`, puis les mini-batches de `batches(X, y, batch_size, seed + epoch)` avec la cross-entropy. Rend la liste des losses d'entraînement moyennes de chaque epoch (des `float`).

La cellule de vérification essaie tes groupes sur un petit réseau, contrôle ton modèle (ce qui s'entraîne, les poids de départ, les modes), puis fait deux pas d'entraînement : elle observe les learning rates que reçoit AdamW, les blocs gelés, les statistiques glissantes des batchnorms, la tête et `layer4`. La cellule suivante fine-tune `FT_EPOCHS` epochs sur `XT_B` (≈ 30 s en mode rapide) avec `head_lr = 1e-3` et `gamma = 0,1`, et compare au linear probe sur les **mêmes** images.

Dans tes notes : combien de paramètres entraînes-tu (✏️ B1.4) ? Comment évoluent la loss d'entraînement et l'accuracy de test : le dégel de `layer4` rapporte-t-il beaucoup par rapport à la tête seule, avec 50 images par classe ? Pourquoi aucun weight decay sur les biais et les paramètres des batchnorms ?""",
       todo=C.TODO_17, check=C.CHECK_17, solution=C.SOLUTION_17 + "\n\n\n" + C.CHECK_17,
       after=[("code", C.EXPERIMENT_17)]),

    Ex("B1.18", "🐛", 2, 25, "Le fine-tuning qui détruit le modèle : quatre erreurs à trouver",
       "trouver et corriger les quatre erreurs d'un fine-tuning, guidé par des diagnostics enchaînés.",
       "Ex B1.17 · fiche §201.4, §201.5 · tableau des pièges ⚠️", tracks="R, C",
       body=r"""Un collègue a écrit `finetune_18`, un fine-tuning court de **tout** ResNet-18 par mini-batches de 4 images. Le code tourne sans erreur, mais le modèle n'apprend pas ce qu'il devrait, et son réseau pré-entraîné en sort abîmé. La fonction contient **quatre** erreurs. Corrige-la **sur place**, dans sa cellule, sans changer sa signature ni ce qu'elle rend.

La cellule de vérification lance `diagnose_18` : un fine-tuning de 12 pas sur 50 images, observé de l'intérieur, et quatre diagnostics enchaînés, chacun ne s'exécutant que si le précédent passe (une erreur peut en cacher une autre) : la tête apprend-elle ? que reçoit la première convolution ? que deviennent les statistiques glissantes des batchnorms ? de combien bougent les poids de la première couche ? Elle tourne dès le premier « Tout exécuter » : ses messages sont ton point de départ. Une fois les quatre points passés, la cellule suivante entraîne ta version `STEPS_18` pas et mesure son accuracy de test.

Dans tes notes : les quatre erreurs, et pour chacune le symptôme qu'elle produit seule. Lesquelles auraient laissé une loss d'entraînement qui baisse quand même ? Pourquoi l'erreur de la batchnorm ne se voit-elle qu'à l'évaluation ?""",
       given=C.GIVEN_18, todo=C.TODO_18, check=C.CHECK_18, solution=C.SOLUTION_18 + "\n\n\n" + C.CHECK_18,
       after=[("code", C.EXPERIMENT_18)]),

    Ex("B1.19", "📦", 3, 50, "Fine-tuning complet : warmup + cosinus, early stopping, meilleur checkpoint",
       "écrire un fine-tuning complet avec learning rates par groupe, warmup puis cosinus pas à pas, early stopping et "
       "retour au meilleur checkpoint.",
       "Ex B1.17 · ✏️ B1.3 · ch. 23 (`fit()`, early stopping, meilleur checkpoint) · ch. 19 (warmup + cosinus) · "
       "fiche §201.4", tracks="R, C",
       body=r"""**Contrat**
- `warmup_cosine_19(step, t_warmup, t_total)` : le facteur du learning rate au pas `step` (0, 1, 2…) : `step / t_warmup` pendant le warmup, puis un cosinus de 1 à 0 au pas `t_total`, et 0 au-delà ; 0 au pas 0, comme dans la fiche ; avec `t_warmup = 0`, pas de warmup ;
- `make_full_19(num_classes=10)` : ResNet-18 `IMAGENET1K_V1`, une nouvelle tête `nn.Linear(512, num_classes)`, **tous** les paramètres entraînables ; sur `DEVICE` ;
- `fine_tune_19(model, train, val, epochs, head_lr, gamma, t_warmup, patience, batch_size=32, seed=0)` (`train` et `val` sont des couples `(images, labels)`) :
  - un `AdamW` sur `param_groups_17` (B1.17) avec cinq modules, de la tête vers l'entrée : `model.fc`, `model.layer4`, `model.layer3`, `model.layer2`, puis `nn.ModuleList([model.conv1, model.bn1, model.layer1])` ;
  - un `LambdaLR` de facteur `warmup_cosine_19(t, t_warmup, t_total)`, avec `t_total = epochs × ceil(len(images) / batch_size)`, qui avance d'un pas après **chaque** `optimizer.step()` ;
  - chaque epoch : `model.train()`, `freeze_batchnorm_17(model)`, les mini-batches de `batches(X, y, batch_size, seed + epoch)` ; à la fin de l'epoch, **un** appel `evaluate(model, X_val, y_val)` ;
  - le meilleur checkpoint : une **copie** des poids de l'epoch à la meilleure accuracy de validation (une égalité ne compte pas comme un progrès) ; arrêt dès que `patience` epochs de suite n'ont pas battu ce record ; à la fin, ces poids sont rechargés dans `model`, laissé en mode `eval()` ;
  - rend `{"train_loss": [...], "val_acc": [...], "lr": [...], "best_epoch": ...}` : un `float` par epoch faite pour les deux premières, la liste des learning rates des groupes **utilisés** à chaque pas (lus avant `optimizer.step()`), et l'indice (à partir de 0) de la meilleure epoch.

La cellule de vérification contrôle ton planning sur quelques pas, puis lance ton fine-tuning sur 20 images avec une validation **scriptée** : `evaluate` rend tour à tour 0,30, 0,50, 0,40, 0,45, 0,60 et retient les poids qu'il voit, ce qui vérifie l'arrêt, `best_epoch`, les learning rates de chaque pas et les poids rendus. La cellule suivante fine-tune tout le réseau sur `XT_B` (`head_lr = 1e-3`, `gamma = 0,2`, une epoch de warmup ; ≈ 50 s en mode rapide), puis trace le planning de chaque groupe et la validation. 🚀 Mode complet : 10 000 images à 224 pixels, 10 epochs, une patience de 2 ; plusieurs heures sur CPU, quelques minutes par epoch sur un GPU de Colab.

Dans tes notes : compare l'accuracy de test à celles de B1.17 et du linear probe, sur les mêmes images. Pourquoi le meilleur checkpoint n'est-il pas forcément le dernier ? Pourquoi le planning avance-t-il à chaque mini-batch et pas à chaque epoch (✏️ B1.3) ? Que se passerait-il si l'on gardait `model.state_dict()` sans le copier ?""",
       todo=C.TODO_19, check=C.CHECK_19, solution=C.SOLUTION_19 + "\n\n\n" + C.CHECK_19,
       after=[("code", C.EXPERIMENT_19)]),
])

PARTS = [PART_A, PART_B]
