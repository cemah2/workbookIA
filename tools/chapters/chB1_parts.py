"""Parts A to D of the chapter B1 notebook: statements and cells (used by build_chB1.py)."""

from __future__ import annotations

import base64
import json
import zlib
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

# ---------------------------------------------------------------------------------------------------------------------
def _packed(data) -> str:
    """JSON data packed (zlib + base64), so that a code cell does not show the numbers before the prediction."""
    return base64.b64encode(zlib.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")


def _gap_cat(points):
    return "A" if points < 10 else ("B" if points <= 25 else "C")


def answers_cd() -> dict:
    """The answers of parts C and D, from the history (several seeds; the categories must not change from one to
    another, with a margin of 3 points to the bounds)."""
    lc = HISTORY.get("lc21", {})
    p20 = lc.get("p20", {})
    p20 = {k: list(v) for k, v in p20.items()}
    p20.setdefault("1000", []).extend(HISTORY.get("p20", {}).get("runs1000", []))   # two more draws at 1 000
    out = {}
    for size, key in (("10", "gap10"), ("1000", "gap1000")):
        gaps = [100 * (r["probe"] - r["scratch"]) for r in p20.get(size, [])] or ([35.0] if size == "10" else [18.0])
        cats = {_gap_cat(g) for g in gaps} | {_gap_cat(g - 3) for g in gaps} | {_gap_cat(g + 3) for g in gaps}
        if len(cats) != 1:
            raise ValueError(f"🔮 B1.20: the gap at {size} images per class is too close to a bound: {gaps}")
        out[key] = cats.pop()
    runs23 = HISTORY.get("o23", {}).get("runs", [])
    a19 = [r["agree19"] for r in runs23] or [0.5]
    cats19 = {("A" if a < 0.30 else "B" if a <= 0.70 else "C") for a in a19}
    more = {r["agree17"] > r["agree19"] for r in runs23} or {True}
    if len(cats19) != 1 or len(more) != 1:
        raise ValueError(f"🔬 B1.23: the answers change from one seed to another: {runs23}")
    out["kept19"], out["partial_more"] = cats19.pop(), more.pop()
    out["runs20"] = {k: [{"seed": r["seed"], "probe": r["probe"], "scratch": r["scratch"]} for r in v] for k, v in p20.items()}
    out["curves21"] = lc.get("summary", {})
    out["curves22"] = {k: v["summary"] for k, v in HISTORY.get("fm22", {}).items()}
    out["c28"] = HISTORY.get("c28", {}).get("C", 0.1)
    return out


B = answers_cd()
_GAP_HINTS = {"gap10": "regarde les lignes à 10 images par classe", "gap1000": "regarde les lignes à 1 000 images par classe"}


def _gap_mistakes(key):
    want = B[key]
    msgs = {"A": f"{_GAP_HINTS[key]} : l'écart est-il sous 10 points ?",
            "B": f"{_GAP_HINTS[key]} : l'écart est-il entre 10 et 25 points ?",
            "C": f"{_GAP_HINTS[key]} : l'écart dépasse-t-il 25 points ?"}
    return "{" + ", ".join(f'"{msgs[k]}": "{k}"' for k in "ABC" if k != want) + "}"


RECORD_20 = (f'wb.record("B1.20a", "{B["gap10"]}", mistakes={_gap_mistakes("gap10")})\n'
             f'wb.record("B1.20b", "{B["gap1000"]}", mistakes={_gap_mistakes("gap1000")})')
SOLUTION_20 = f'gap10_20, gap1000_20 = "{B["gap10"]}", "{B["gap1000"]}"'
_KEPT_HINTS = {"A": "relis la ligne de MODEL_19 : l'accord est-il sous 30 % ?",
               "B": "relis la ligne de MODEL_19 : l'accord est-il entre 30 et 70 % ?",
               "C": "relis la ligne de MODEL_19 : l'accord dépasse-t-il 70 % ?"}
RECORD_23 = (f'wb.record("B1.23a", "{B["kept19"]}", mistakes={{'
             + ", ".join(f'"{_KEPT_HINTS[k]}": "{k}"' for k in "ABC" if k != B["kept19"]) + "})\n"
             f'wb.record("B1.23b", {B["partial_more"]}, mistakes={{"compare les lignes de MODEL_17 et de MODEL_19 : '
             f'lequel garde la plus grande part des prédictions ImageNet ?": {not B["partial_more"]}}})')
SOLUTION_ANSWERS_23 = f'kept19_23, partial_more_23 = "{B["kept19"]}", {B["partial_more"]}'

PART_C = Part("C", "Quand le transfert aide, quand il échoue",
              "On mesure maintenant ce que rapporte le pré-entraînement et où il cesse de rapporter : selon la quantité "
              "de données (CIFAR-10, de 10 à 1 000 images par classe), sur un domaine lointain (Fashion-MNIST), et ce "
              "qu'il coûte au modèle d'origine (l'oubli). La référence **from scratch** est un petit CNN comme au ch. 21 "
              "(`small_cnn`, ≈ 24 000 paramètres), entraîné par `train_scratch` avec le même soin à chaque taille : "
              "augmentation (miroirs, recadrages décalés), AdamW, un planning one-cycle, 15 epochs et au moins 200 pas "
              "(≈ 3 s pour 100 images sur deux cœurs) ; la cellule suivante le donne. Les protocoles complets (jusqu'à "
              "1 000 images par classe, plusieurs tirages, fine-tunings compris) prennent plus d'une heure sur CPU : "
              "leurs résultats ont été précalculés par `tools/chapters/history_chB1.py` avec les fonctions mêmes des "
              "solutions, à 96 pixels, et s'affichent à côté de ce que ton code mesure en mode rapide. 🚀 En mode complet, "
              "ton code refait tout, à 224 pixels.", given=C.GIVEN_C, exercises=[
    Ex("B1.20", "🔮", 1, 10, "Prédire : où le pré-entraînement rapporte-t-il le plus, 10 ou 1 000 images par classe ?",
       "prédire, en points d'accuracy, ce que rapporte un backbone pré-entraîné face à un CNN from scratch avec très peu "
       "et avec beaucoup d'images.",
       "Ex B1.13 · fiche §201.6 (la courbe d'apprentissage selon la taille du train)", tracks="R, C",
       body=r"""La fiche (§201.6) dit le **sens** : le pré-entraînement compte davantage quand les données sont rares. Ici, tu prédis **de combien**. Deux stratégies, sur les mêmes images d'entraînement et le même test (`X_TEST`) :
- le **linear probe** sur les embeddings figés du ResNet-18 pré-entraîné (B1.13), avec `C = 0,01` ;
- le **CNN from scratch** de `train_scratch`, sur les images de 32 pixels.

L'écart se compte **en points d'accuracy** : l'accuracy du linear probe moins celle du CNN from scratch.

**Avant d'exécuter**, écris tes deux prédictions :
a) l'écart avec **10** images par classe (100 images) : `"A"` moins de 10 points, `"B"` de 10 à 25 points, `"C"` plus de 25 points ;
b) l'écart avec **1 000** images par classe (10 000 images), mêmes catégories.

La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites. Elle mesure le tirage du notebook à 10 images par classe (≈ 5 s), puis affiche le même protocole précalculé, à 96 pixels, sur trois tirages à 10 et à 1 000 images par classe ; tes prédictions sont vérifiées sur ce tableau.""",
       todo=C.TODO_20, solution=SOLUTION_20, hypothesis=True, record=RECORD_20,
       after=[("code", guarded(C.EXPERIMENT_20.replace("{RUNS_20}", _packed(B["runs20"])), ["gap10_20", "gap1000_20"],
                               "⏳ Ex B1.20 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : tes prédictions étaient-elles justes ? Compare, à 10 et à 1 000 images par classe, "
                     "l'écart en points et le rapport des deux accuracies : que dit chacun ? Que faudrait-il changer au CNN "
                     "from scratch pour qu'il ait sa meilleure chance (He et coll., 2019, fiche §201.6) ?")]),

    Ex("B1.21", "🔬", 3, 60, "Courbes d'apprentissage : from scratch, linear probe et fine-tuning de 10 à 1 000 images par classe",
       "tracer, selon un protocole honnête, les courbes d'apprentissage de trois stratégies, et lire où chacune l'emporte.",
       "Ex B1.20, B1.19, B1.13 · fiche §201.6 (le protocole des courbes d'apprentissage) · ch. 2 (variabilité d'un "
       "échantillon)", tracks="C",
       body=r"""Le protocole est celui de la fiche (§201.6) : des sous-ensembles **emboîtés et stratifiés** du train (`wb.datasets.load_cifar10_subset(k, "train", seed)` : pour une même graine, les 10 images par classe sont parmi les 30, qui sont parmi les 100…), le **même** test pour tous les points, **plusieurs graines** aux petites tailles, une recette fixée d'avance pour chaque stratégie.

**Contrat** (les docstrings de la cellule suivante) : trois fonctions `fit_eval(X, y, X_test, y_test, seed=0)` qui rendent une accuracy de test (un `float`) :
- `probe_21` : ton `probe_13` avec `C = 0,01` sur les embeddings figés du ResNet-18 de référence, lus par `reference_embeddings(images, SIZE, subset)` (calculés une fois, puis relus dans le cache) ;
- `scratch_21` : `train_scratch(X, y, seed=seed)`, puis son accuracy ;
- `finetune_21` : un fine-tuning complet par tes `make_full_19` et `fine_tune_19`, avec `FT_RECIPE_21`, validé sur `(X_VAL, Y_VAL)`, une epoch de warmup, au moins 40 pas, puis `evaluate` sur le test ;

et deux fonctions du protocole :
- `learning_curve_21(fit_eval, plan, test, load=wb.datasets.load_cifar10_subset)` : `plan` associe une taille (images par classe) à ses graines ; un point `{"size", "seed", "accuracy"}` par taille, dans l'ordre **croissant**, et par graine, dans l'ordre donné ; chaque point reçoit `load(size, "train", seed)` et le test ;
- `summarize_21(rows)` : `{taille: (moyenne, écart-type, nombre de points)}`, écart-type de `np.std` (`ddof = 0`).

La cellule de vérification essaie `learning_curve_21` avec un faux `fit_eval` qui note ce qu'il reçoit (tailles, graines, sous-ensembles, test), puis tes trois stratégies sur de tout petits jeux (pour `finetune_21`, sans entraînement : seuls les arguments passés à `fine_tune_19` sont lus). La cellule suivante trace les courbes du linear probe et du CNN from scratch mesurées par ton code (10 et 30 images par classe sur trois tirages, 100 sur un ; ≈ 40 s en mode rapide), à côté des trois courbes complètes précalculées (jusqu'à 1 000 images par classe, fine-tuning compris). 🚀 Mode complet : tout le protocole avec les trois stratégies, à 224 pixels ; plusieurs heures sur CPU (le seul fine-tuning à 1 000 images par classe compte ≈ 940 pas de 3 s), quelques dizaines de minutes sur un GPU de Colab.

Dans tes notes : où le fine-tuning complet dépasse-t-il le linear probe, et où reste-t-il dessous ? À quelle taille l'écart-type d'un tirage à l'autre est-il le plus grand, et que vaut une conclusion tirée d'un seul tirage à cette taille ? Pourquoi le même test et des sous-ensembles emboîtés ? Que rapporte le passage de 100 à 1 000 images par classe à chaque stratégie ?""",
       todo=C.TODO_21, check=C.CHECK_21, solution=C.SOLUTION_21 + "\n\n\n" + C.CHECK_21,
       after=[("code", C.EXPERIMENT_21.replace("{CURVES_21}", _packed(B["curves21"])))]),

    Ex("B1.22", "🔬", 3, 45, "Domaine lointain : transférer vers Fashion-MNIST, comparé au CNN from scratch",
       "mesurer si un backbone ImageNet gelé bat encore un CNN entraîné de zéro sur un domaine lointain, et à partir de "
       "combien d'images il cesse de le faire.",
       "Ex B1.21, B1.16, B1.17, B1.13 · ch. 21 (un CNN sur Fashion-MNIST) · fiche §201.7 (distance de domaine, transfert "
       "négatif)", tracks="C",
       body=r"""Fashion-MNIST est loin d'ImageNet (B1.16) : des vêtements en niveaux de gris, centrés sur fond noir. Le pré-entraînement y rapporte-t-il encore, face à un CNN entraîné de zéro **sur les mêmes images** ? La fiche (§201.7) le dit : « le transfert fait 85 % » ne veut rien dire sans cette baseline. La cellule suivante charge un test fixe de Fashion-MNIST (`XF_TE_22`) et donne `fm_embeddings(images)` : les embeddings de `layer3` et de `layer4` du ResNet-18 de référence pour des images agrandies à `FM_SIZE` pixels (64 en mode rapide, deux fois moins cher qu'à 96 pour les mêmes conclusions ; `SIZE` en mode complet), calculés en une passe puis mis en cache.

**Contrat**
- `fm_probe_22(block)` : rend une fonction `fit_eval(X, y, X_test, y_test, seed=0)` (comme en B1.21) qui donne l'accuracy de test de ton `probe_13` avec `C = 0,01` sur `fm_embeddings(images)[block]`, `block` valant `"layer3"` ou `"layer4"` ;
- `fm_scratch_22(X, y, X_test, y_test, seed=0)` : l'accuracy de test de `train_scratch(X, y, seed=seed)`, sur des images en niveaux de gris.

La cellule de vérification compare tes fonctions à la référence sur de petits jeux. La cellule suivante trace, avec ton `learning_curve_21` (`load=wb.datasets.load_fashion_mnist_subset`), les courbes du CNN from scratch et des deux probes gelés (10 images par classe sur deux tirages, 100 et 300 sur un ; ≈ 45 s en mode rapide), à côté des courbes précalculées jusqu'à 1 000 images par classe, avec en plus le fine-tuning partiel de B1.17 (`layer4` et la tête, mêmes réglages qu'en B1.17). 🚀 En mode complet, ton code refait tout, fine-tuning partiel compris.

Dans tes notes : à quelles tailles le probe gelé l'emporte-t-il sur le CNN from scratch, et l'inverse ? Est-ce du transfert négatif au sens de la fiche ? `layer3` ou `layer4` : lequel garde son avance le plus longtemps, et pourquoi (B1.16) ? Que change le fine-tuning partiel ? Compare à CIFAR-10 (B1.21) : sur quel domaine le pré-entraînement garde-t-il son avance à 1 000 images par classe ?""",
       given=C.GIVEN_22, todo=C.TODO_22, check=C.CHECK_22, solution=C.SOLUTION_22 + "\n\n\n" + C.CHECK_22,
       after=[("code", C.EXPERIMENT_22.replace("{CURVES_22}", _packed(B["curves22"])))]),

    Ex("B1.23", "🔬", 2, 25, "Oubli catastrophique : que reste-t-il des prédictions ImageNet après le fine-tuning ?",
       "mesurer ce qu'un fine-tuning fait oublier au modèle de base, en remettant son ancienne tête sur le backbone "
       "fine-tuné.",
       "Ex B1.19, B1.17 · fiche §201.7 (l'oubli catastrophique)", tracks="C",
       body=r"""Après B1.17 et B1.19, tu as deux réseaux fine-tunés sur CIFAR-10 : `MODEL_17` (seuls `layer4` et la tête ont appris) et `MODEL_19` (tout le réseau). Que reste-t-il de ce que savait le ResNet-18 d'origine ? La fiche (§201.7) donne la mesure : remettre l'**ancienne tête** (les 1 000 sorties d'ImageNet) sur le backbone fine-tuné, et compter les images pour lesquelles sa classe ImageNet prédite est **la même** que celle du réseau d'origine : l'**accord top-1**.

**Contrat** de `imagenet_agreement_23(model, reference, images, size=SIZE, batch_size=128)` (la docstring) : l'accord top-1, un `float` entre 0 et 1, entre le backbone de `model` coiffé de la tête ImageNet `reference.fc` et `reference` lui-même (le ResNet-18 pré-entraîné, `RESNET.full`), en mode `eval()`, sans gradient, les images préparées par `to_input(batch, size)` ; `model` reste intact (sa tête à 10 classes, ses poids, son mode).

La cellule de vérification essaie ta fonction sur un réseau connu : une copie du ResNet pré-entraîné dont les poids de `layer4` ont été un peu déplacés. La cellule suivante mesure l'accord sur 500 images de test de CIFAR-10 pour le réseau pré-entraîné lui-même (un contrôle), `MODEL_17` et `MODEL_19` (⏳ tant que ces modèles n'existent pas). Lis le tableau, puis écris dans la cellule d'après :
a) l'accord de `MODEL_19` : `"A"` moins de 30 %, `"B"` de 30 à 70 %, `"C"` plus de 70 % ;
b) `True` ou `False` : `MODEL_17` garde une plus grande part des prédictions ImageNet que `MODEL_19`.

Dans tes notes : pourquoi l'accord du contrôle vaut-il 100 % ? Les batchnorms sont restées gelées pendant les deux fine-tunings : pourquoi l'oubli a-t-il lieu quand même ? Si ton modèle devait garder ses deux compétences, que ferais-tu (fiche §201.7) ?""",
       todo=C.TODO_23, check=C.CHECK_23, solution=C.SOLUTION_23 + "\n\n\n" + C.CHECK_23,
       after=[("code", C.EXPERIMENT_23), ("todo", C.ANSWERS_23), ("check", C.CHECK_23B),
              ("solution", SOLUTION_ANSWERS_23), ("record", RECORD_23)]),
])

PART_D = Part("D", "Modèles de fondation, coût et livraison",
              "CLIP classe des images sans un seul exemple (zéro-shot) ; DINOv2 et CLIP les classent avec quelques "
              "exemples (few-shot par prototypes) ; puis le coût d'usage de trois backbones sur CPU, la publication du "
              "modèle fine-tuné et le défi. CLIP (≈ 600 Mo) se télécharge à la première exécution de B1.24 ; s'il ne "
              "peut pas l'être, B1.24 affiche ⏳ (à valider sur Colab) et B1.25 se rabat sur les prototypes de DINOv2. "
              "Zéro-shot et few-shot se mesurent sur les mêmes `D_PER_CLASS` = 50 images de test par classe en mode "
              "rapide (500 images ; tout le test en mode complet) : CLIP coûte ≈ 50 ms par image sur deux cœurs.",
              exercises=[
    Ex("B1.24", "📦", 2, 30, "Zéro-shot avec CLIP sur CIFAR-10 : gabarits de prompts et ensembles de prompts",
       "classer CIFAR-10 sans aucun exemple étiqueté avec CLIP, et mesurer ce que rapportent un gabarit et un ensemble "
       "de gabarits.",
       "Ex B1.11 · ✏️ B1.5 · ch. 17 (softmax) · fiche §201.8 (le zéro-shot, les gabarits)", tracks="R, C",
       body=r"""La cellule suivante charge CLIP ViT-B/32 par `wb.transfer` (`CLIP.full` : le `CLIPModel` ; `CLIP.processor` : son `CLIPProcessor` ; révision figée), calcule une fois les embeddings d'image des 500 images de test `X_D` (`CLIP_D`, 224 pixels, ≈ 25 s la première fois), lit l'échelle apprise `CLIP_SCALE` (= `exp(logit_scale)`, ✏️ B1.5) et donne `CLIP_TEMPLATES`, les 18 gabarits que les auteurs de CLIP ont utilisés pour CIFAR-10.

**Contrat** (les docstrings)
- `text_embeddings_24(model, processor, classnames, templates)` : un tableau NumPy `float32` `(K, 512)`, une ligne de norme 1 par classe : la moyenne des vecteurs de texte **normalisés** de ses phrases `template.format(classe)`, normalisée à son tour ; sans gradient ;
- `zero_shot_24(image_emb, text_emb, scale)` : les probabilités `(N, K)`, la softmax sur les classes de `scale` × cosinus, pour des embeddings que la fonction normalise elle-même.

La cellule de vérification compare `zero_shot_24` à la formule sur des embeddings de normes quelconques (avec $s = 100$ et $s = 1$), puis `text_embeddings_24` à une référence sur deux classes et deux gabarits. La cellule suivante classe les 500 images de test avec trois jeux de phrases : le nom seul, « a photo of a {}. », puis les 18 gabarits ; elle refait le dernier avec $s = 1$ et donne l'accuracy par classe. Elle garde les vecteurs des 18 gabarits dans `TEXT_24`, pour B1.25.

Dans tes notes : que rapportent un gabarit, puis l'ensemble, sur ces 500 images (combien d'images cela fait-il) ? Que change $s = 1$, et que ne change-t-il pas ? Compare au linear probe de B1.13, qui avait 2 000 images étiquetées. Quelles classes CLIP réussit-il le moins, et pourquoi, d'après sa fiche modèle (§201.8, « Les limites ») ?""",
       given=C.GIVEN_24, todo=C.TODO_24, check=C.CHECK_24, solution=C.SOLUTION_24 + "\n\n\n" + C.CHECK_24,
       after=[("code", C.EXPERIMENT_24)]),

    Ex("B1.25", "📦", 3, 40, "Few-shot par prototypes de classe avec DINOv2 et CLIP (1 à 16 exemples par classe)",
       "classer avec quelques exemples par classe par les prototypes de deux modèles de fondation, et comparer au "
       "zéro-shot.",
       "Ex B1.24, B1.13 · fiche §201.6 (le few-shot et les prototypes de classe), §201.8", tracks="C",
       body=r"""La cellule suivante prépare le **support** : 16 images d'entraînement par classe (`XS_25`, `YS_25`), passées une fois par DINOv2-small (`DINO_S_25`, `DINO_SIZE` pixels) et par CLIP (`CLIP_S_25`) ; les **requêtes** sont les 500 images de test de B1.24 (`DINO_D_25`, `CLIP_D`). Elle donne aussi `draw_support(labels, k, draw, seed=0)` : les positions de $k$ exemples par classe pour le tirage numéro `draw`, toujours les mêmes pour les mêmes arguments.

**Contrat** (les docstrings)
- `prototypes_25(emb, labels, n_classes=10)` : `(n_classes, d)`, la moyenne des embeddings **normalisés** (norme 1) de chaque classe ;
- `predict_25(emb, protos)` : pour chaque embedding, la classe du prototype au plus grand **cosinus** ;
- `few_shot_25(support, query, shots, draws, seed=0)` : `support` et `query` sont des couples `(embeddings, labels)` ; pour chaque $k$ de `shots`, dans l'ordre, `{"shots": k, "mean": ..., "std": ...}` : la moyenne et l'écart-type (`np.std`, `ddof = 0`) des accuracies sur les requêtes des prototypes de `draws` tirages `draw_support(labels du support, k, d, seed)`, `d` de 0 à `draws - 1`.

La cellule de vérification compare tes fonctions à la formule sur de petits tableaux aléatoires (embeddings et prototypes de normes très différentes). La cellule suivante trace l'accuracy des prototypes de DINOv2 et de CLIP pour 1, 2, 4, 8 et 16 exemples par classe (`DRAWS_25` tirages chacun), avec le zéro-shot de CLIP de B1.24 et une combinaison : le vecteur de texte de chaque classe plus son prototype normalisé (≈ 15 s la première fois). Sans CLIP, seuls les prototypes de DINOv2 sont mesurés.

Dans tes notes : combien d'exemples par classe faut-il aux prototypes de CLIP pour égaler son zéro-shot ? DINOv2 ou CLIP : lequel donne les meilleurs prototypes, et qu'est-ce qui, dans leur entraînement (§201.8), peut l'expliquer ? Que rapporte la combinaison texte + prototype, et à quel nombre d'exemples ? Comment varie l'écart-type avec le nombre d'exemples ? Ajouter une 11ᵉ classe : que faudrait-il recalculer ?""",
       given=C.GIVEN_25, todo=C.TODO_25, check=C.CHECK_25, solution=C.SOLUTION_25 + "\n\n\n" + C.CHECK_25,
       after=[("code", C.EXPERIMENT_25)]),

    Ex("B1.26", "🔬", 2, 30, "Taille, latence et accuracy sur CPU : MobileNetV3-small, ResNet-18, DINOv2-small",
       "mesurer correctement la taille, la latence, le débit et l'accuracy de trois backbones, pour en choisir un selon "
       "un budget.",
       "Ex B1.11 · fiche §201.9 (quatre mesures à ne pas confondre, mesurer une latence)", tracks="R, C",
       body=r"""La cellule suivante donne les trois backbones gelés de `wb.transfer` : MobileNetV3-small (`MOBILE`, embedding de 576 valeurs), ResNet-18 (`RESNET`) et DINOv2-small (`DINO`) ; `bb.model` rend l'embedding.

**Contrat** (les docstrings)
- `latency_ms_26(model, x, reps=20, warmup=3)` : le temps d'un appel `model(x)`, en millisecondes : la **médiane** de `reps` appels chronométrés (`time.perf_counter()`), après `warmup` appels non chronométrés ; le modèle en `eval()`, les appels sous `torch.inference_mode()` ;
- `size_26(model)` : `(nombre de paramètres, leur taille en Mo)`, la taille étant la somme de `numel() × element_size()` des paramètres, divisée par $10^6$ ;
- `profile_26(model, size, batch=32)` : `{"params", "mb", "latency_ms", "images_per_s"}` : `size_26`, la latence d'**une** image (`torch.randn(1, 3, size, size)`), et le débit en images par seconde de batches de `batch` images (`batch` divisé par la latence d'un tel batch, en secondes ; `reps=5` et `warmup=1` pour les batches).

La cellule de vérification essaie tes fonctions sur un faux backbone dont on connaît le temps (des appels de chauffe lents, un appel lent au milieu, comme une tâche de fond) et sur un petit réseau en `float32` et `float16`. La cellule suivante mesure les trois backbones à 224 pixels (latence d'une image, débit par batches de 16), la latence au côté de leur linear probe (`SIZE` ou `DINO_SIZE` pixels), et l'accuracy de test d'un linear probe appris sur 100 images par classe (≈ 45 s la première fois, surtout les embeddings DINOv2, mis en cache pour B1.28). Une mesure de temps dépend de la machine : elle ne se vérifie que par un seuil, jamais par sa valeur.

Dans tes notes : MobileNetV3-small fait 0,06 GFLOPS par image de 224 pixels, ResNet-18 1,8 (`weights.meta["_ops"]`) : le rapport des latences suit-il celui des FLOPS (fiche §201.9) ? Quel backbone choisirais-tu pour répondre en moins de 20 ms par image sur ce CPU, et lequel pour classer un million d'images en une nuit ? Que coûte chaque point d'accuracy gagné ? Que changerait la demi-précision (`float16`) à la taille ?""",
       given=C.GIVEN_26, todo=C.TODO_26, check=C.CHECK_26, solution=C.SOLUTION_26 + "\n\n\n" + C.CHECK_26,
       after=[("code", C.EXPERIMENT_26)]),

    Ex("B1.27", "🛠️", 2, 25, "Publier un modèle fine-tuné : safetensors, configuration et fiche modèle qui cite le modèle de base",
       "publier un modèle dérivé en trois fichiers, poids, configuration et fiche modèle, et le recharger sans pickle.",
       "Ex B1.19 · ch. 23 (fiche modèle, `weights_only=True`) · fiche §201.2 (licences), §201.9 (publier un modèle dérivé) · "
       "`data/cards/pretrained.md`, `data/cards/cifar10.md`", tracks="R, C",
       body=r"""Tu publies `MODEL_19`, le ResNet-18 fine-tuné de B1.19, comme un **modèle dérivé** (fiche §201.9). La cellule suivante fixe le dossier `PUBLISH_DIR_27` (à côté de ton notebook dans `mon_travail`, sinon un dossier temporaire), le nom des poids de départ `BASE_MODEL_27` et les sections de la fiche modèle, `CARD_SECTIONS_27`.

**Contrat** (les docstrings)
- `publish_27(model, folder, test_accuracy)` écrit trois fichiers dans `folder` et rend ce dossier :
  - `model.safetensors` : le `state_dict` (tenseurs sur le CPU, contigus), avec des métadonnées qui contiennent au moins `base_model` (= `BASE_MODEL_27`) et `num_classes` (`"10"`) : `save_file` n'accepte que des **chaînes** ;
  - `config.json` : `architecture` (`"resnet18"`), `base_model`, `num_classes` (l'entier 10), `class_names` (les 10 noms de `CLASSES`, dans l'ordre des sorties), `image_size` (`SIZE`), `interpolation` (`"bilinear"`), `mean` et `std` (ceux d'ImageNet), `test_accuracy`, `versions` (`torch` et `torchvision`) ;
  - `README.md` : la fiche modèle, avec les sections `## ` de `CARD_SECTIONS_27`, dans l'ordre, au moins 150 mots, qui cite le modèle de base (ResNet-18, `IMAGENET1K_V1`) et sa licence (BSD-3-Clause pour torchvision, et les conditions d'ImageNet, recherche et enseignement, non commercial : `data/cards/pretrained.md`), le dataset et sa source (CIFAR-10, A. Krizhevsky, 2009 : `data/cards/cifar10.md`), et l'accuracy de test écrite en pourcentage avec une décimale et une virgule (comme « 81,3 % »).
- `load_published_27(folder)` : `(model, config)`, le réseau reconstruit d'après `config.json` (`resnet18(weights=None)`, une nouvelle tête), ses poids lus dans `model.safetensors` **sans pickle**, en `eval()`, sur le CPU.

La cellule de vérification publie `MODEL_19` avec son accuracy de test de B1.19, relit les trois fichiers (tenseurs, métadonnées, champs, sections), puis recharge le modèle par ta fonction, `torch.load` désactivé le temps de l'appel, et compare ses logits à ceux de `MODEL_19` sur 8 images. ⏳ tant que B1.19 n'est pas fait.

Dans tes notes : qu'est-ce que ton modèle hérite de son modèle de base, et que doit en dire sa fiche ? Pourquoi les classes **dans l'ordre des sorties** ? Pourquoi ne pas publier un fichier `.pt` ? Qu'est-ce qu'un utilisateur ne pourrait pas refaire avec ces trois fichiers seulement ?""",
       given=C.GIVEN_27, todo=C.TODO_27, check=C.CHECK_27, solution=C.SOLUTION_27 + "\n\n\n" + C.CHECK_27),

    Ex("B1.28", "🏆", 4, 120, "Défi : au moins 90 % sur le test CIFAR-10 avec 100 images par classe et 10 minutes de CPU",
       "atteindre un objectif d'accuracy avec très peu d'images étiquetées et un budget de calcul, en choisissant parmi "
       "tout ce que le chapitre a mesuré.",
       "Ex B1.19, B1.25 · les mesures de B1.21 à B1.26 · tout le chapitre", tracks="C",
       body=r"""Le défi : **au moins 90 %** sur le test de CIFAR-10, avec **100 images étiquetées par classe** (`X_TR_28`, `Y_TR_28` : 1 000 images) et **10 minutes de CPU**, entraînement et prédiction du test compris. En mode rapide, le test a 2 000 images ; en mode complet, les 10 000.

Écris `method_28(X, y)` : elle s'entraîne sur ces images et ces labels **seulement**, et rend une fonction `predict(images)` qui donne les labels (des entiers). Tout ce que le chapitre a montré est permis : un backbone pré-entraîné de `wb.transfer`, gelé ou fine-tuné, des embeddings, des prototypes, le zéro-shot, l'augmentation. Interdit : d'autres images étiquetées, et toute variable du test (`X_TEST`, `Y_TEST`) ou d'autres labels globaux.

`evaluate_28(method_28)` (cellule suivante) :
1. un **contrôle anti-fuite** : ta méthode, entraînée sur des labels **mélangés** (les variables globales de labels mélangées elles aussi, le temps du contrôle), doit rester au hasard : ses prédictions ne doivent retrouver ni les vrais labels de la validation, ni les labels globaux mélangés de la validation et du test ;
2. l'accuracy sur la **validation** `X_VAL` (500 images, jamais dans `X_TR_28`) : le test n'est révélé que si elle atteint `GATE_28` = 80 % ;
3. le test, révélé **une fois par session** : le premier résultat est figé (`REVEALED_28`), une nouvelle exécution affiche le même.

Règles d'honneur : règle ta méthode sur `X_VAL`, jamais sur le test ; le temps compte le premier calcul de tes embeddings, même si le cache les relit ensuite (dis-le dans tes notes si tu relances).

Dans tes notes : ta démarche, tes essais sur la validation et leur temps ; ce que tu gardes, et pourquoi ; ce que coûterait ta méthode en production (B1.26) ; et ce que dit la licence du backbone choisi d'un usage commercial (§201.2).""",
       note="Ici, les embeddings DINOv2 des images d'entraînement, de validation et de test viennent du cache (calculés "
            "par B1.15, B1.26 et le contrôle anti-fuite) : le temps affiché est presque nul. À froid, la même méthode coûte "
            "≈ 75 s sur deux cœurs (≈ 27 s pour les 1 000 images d'entraînement, ≈ 48 s pour les 2 000 images de test, "
            "`tools/chapters/history_chB1.py`) : loin des 10 minutes.",
       given=C.GIVEN_28, todo=C.TODO_28, check=C.CHECK_28,
       solution=C.SOLUTION_28.replace("{C28}", repr(B["c28"])) + "\n\n\n" + C.CHECK_28),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]
