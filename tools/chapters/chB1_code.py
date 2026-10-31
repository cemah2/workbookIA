"""Code cells of the chapter B1 notebook (parts A and B), shared by build_chB1.py and history_chB1.py (used by Claude).

Each constant is the source of one cell: TOOLS (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
"""

TOOLS = r'''# Tools for the notebook exercises (parts A and B)
import copy
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from wb.nbtools import StopCheck, checked, filled, fr, returned, same, verdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # a computer without PyTorch (Mac Intel): this chapter is done on Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : ce chapitre se fait sur Colab (00_setup/INSTALL_LOCAL.md).")

warnings.filterwarnings("ignore", message=".*IProgress not found.*")    # tqdm without ipywidgets (transformers)
from wb import transfer
from wb.transfer import DINOV2_NAME, DINOV2_REVISION

DEVICE = cfg.device                       # "cuda" on a Colab GPU, else "cpu"
SIZE = 96 if FAST_MODE else 224           # the side of the images given to ResNet-18 (CIFAR-10 images are 32 x 32)
DINO_SIZE = 140 if FAST_MODE else 224     # the side of the images given to DINOv2 (a multiple of its 14-pixel patches)
PER_CLASS = 200 if FAST_MODE else 1000    # training images per class (nested, stratified subsets of CIFAR-10)
TEST_PER_CLASS = 200 if FAST_MODE else 1000   # test images per class (1 000 = the whole test split)
VAL_PER_CLASS = 50 if FAST_MODE else 100      # validation images per class, never in a training subset
CLASSES = wb.datasets.CIFAR10_CLASSES
X_TRAIN, Y_TRAIN = wb.datasets.load_cifar10_subset(PER_CLASS)          # uint8 (N, 32, 32, 3), int64 labels
X_TEST, Y_TEST = wb.datasets.load_cifar10_subset(TEST_PER_CLASS, "test")
X_VAL, Y_VAL = wb.datasets.load_cifar10_subset(VAL_PER_CLASS, "val")
# The frozen REFERENCE backbone of the checks and of the given experiments (wb.transfer), not the one you write:
# RESNET.model(x) gives the 512 values of the embedding, RESNET.full is the whole network (ImageNet head included)
RESNET = transfer.load_backbone("resnet18")


def pct(x, digits=1):
    """A share written the French way: 0.734 -> '73,4 %'."""
    return f"{100 * x:.{digits}f} %".replace(".", ",")


def nested(per_class, part="train"):
    """The positions, in X_TRAIN (part="train") or X_TEST ("test"), of the smaller nested subset of `per_class`
    images per class (the subsets are nested: the 30 per class are among the 200)."""
    big = wb.datasets.cifar10_subset_indices(PER_CLASS if part == "train" else TEST_PER_CLASS, part)
    return np.searchsorted(big, wb.datasets.cifar10_subset_indices(per_class, part))


def reference_embeddings(images, size, subset, block=None):
    """The embeddings of the reference ResNet-18, computed once and then read from the cache (data/downloads)."""
    return transfer.cached_embeddings(RESNET, images, size=size, block=block, subset=subset, verbose=False)


def reference_probe(E_train, y_train, E_test, y_test, C=1.0):
    """Test accuracy of the reference linear probe (StandardScaler, then LogisticRegression)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return float(probe.fit(E_train, y_train).score(E_test, y_test))


def n_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def need(ex_id, *names):
    """Stop a check cell with ⏳ while a function of an earlier exercise is not written yet."""
    for name in names:
        fn = globals().get(name)
        if fn is None:
            print(f"⏳ Ex {ex_id} : définis d'abord {name}.")
            raise StopCheck


torch.manual_seed(0)
print(f"CIFAR-10 : {len(X_TRAIN)} images d'entraînement, {len(X_VAL)} de validation, {len(X_TEST)} de test ; "
      f"ResNet-18 à {SIZE} pixels, DINOv2 à {DINO_SIZE} pixels ; calcul sur {DEVICE}.")'''

# --- Part A -----------------------------------------------------------------------------------------------------------
TODO_11 = r'''def load_backbones_11():
    """The three pre-trained models of the chapter, each with its preprocessing, in eval() mode.

    Returns {"resnet18": (model, preprocess), "mobilenet_v3_small": (model, preprocess),
             "dinov2_small": (model, processor)}:
    - the two torchvision models with their IMAGENET1K_V1 weights (ImageNet head kept), preprocess = weights.transforms();
    - DINOv2-small from the Hugging Face Hub (DINOV2_NAME, revision DINOV2_REVISION): AutoModel and AutoImageProcessor."""
    raise NotImplementedError("load_backbones_11")


# What you read on the objects, in a cell of your own (the values, e.g. 42 or "AB")
mobilenet_in_11 = ...     # a) the size of the vector at the ENTRANCE of mobilenet.classifier
mobilenet_mid_11 = ...    # b) the size of the vector BETWEEN the two dense layers of mobilenet.classifier
dino_d_11 = ...           # c) hidden_size of DINOv2-small
pooler_is_cls_11 = ...    # d) True or False: pooler_output equals last_hidden_state[:, 0] (try two images)
resize_11 = ...           # e) resize_size of the preprocessing of ResNet-18 (a whole number)
bicubic_11 = ...          # f) the letters of the models whose preprocessing interpolates in BICUBIC:
                          #    (A) ResNet-18, (B) MobileNetV3-small, (C) DINOv2-small, e.g. "AB"'''

SOLUTION_11 = r'''from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18
from transformers import AutoImageProcessor, AutoModel


def load_backbones_11():
    """The three pre-trained models of the chapter, each with its preprocessing, in eval() mode.

    Returns {"resnet18": (model, preprocess), "mobilenet_v3_small": (model, preprocess),
             "dinov2_small": (model, processor)}:
    - the two torchvision models with their IMAGENET1K_V1 weights (ImageNet head kept), preprocess = weights.transforms();
    - DINOv2-small from the Hugging Face Hub (DINOV2_NAME, revision DINOV2_REVISION): AutoModel and AutoImageProcessor."""
    out = {}
    for name, builder, weights in (("resnet18", resnet18, ResNet18_Weights.IMAGENET1K_V1),
                                   ("mobilenet_v3_small", mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1)):
        out[name] = (builder(weights=weights).eval(), weights.transforms())
    with transfer.quiet_hub():                                   # no progress bar in the notebook
        processor = AutoImageProcessor.from_pretrained(DINOV2_NAME, revision=DINOV2_REVISION)
        model = AutoModel.from_pretrained(DINOV2_NAME, revision=DINOV2_REVISION).eval()
    out["dinov2_small"] = (model, processor)
    return out


models_11 = load_backbones_11()
mobilenet_11, mobilenet_prep_11 = models_11["mobilenet_v3_small"]
print(mobilenet_11.classifier)                                   # Linear(576 -> 1024), Hardswish, Dropout, Linear(1024 -> 1000)
dino_11, processor_11 = models_11["dinov2_small"]
with torch.inference_mode():
    out_11 = dino_11(pixel_values=processor_11(images=list(X_TEST[:2]), return_tensors="pt")["pixel_values"])
print(models_11["resnet18"][1])
print(type(processor_11).__name__, processor_11.resample, processor_11.crop_size)

mobilenet_in_11 = mobilenet_11.classifier[0].in_features
mobilenet_mid_11 = mobilenet_11.classifier[3].in_features
dino_d_11 = dino_11.config.hidden_size
pooler_is_cls_11 = bool(torch.allclose(out_11.pooler_output, out_11.last_hidden_state[:, 0]))
resize_11 = models_11["resnet18"][1].resize_size[0]
bicubic_11 = "C"                                                 # BILINEAR for both CNNs, resample=3 (BICUBIC) for DINOv2'''

CHECK_11 = r'''# a) to f): the values you read (each one is checked on its own)
for id_11, value_11 in (("B1.11a", mobilenet_in_11), ("B1.11b", mobilenet_mid_11), ("B1.11c", dino_d_11),
                        ("B1.11d", pooler_is_cls_11), ("B1.11e", resize_11), ("B1.11f", bicubic_11)):
    with wb.attempt(id_11):
        wb.check(id_11, value_11)

# load_backbones_11: compared with the reference models of wb.transfer
with checked("B1.11"):
    from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights
    loaded_11 = load_backbones_11()
    if not returned("B1.11", "load_backbones_11", loaded_11):
        raise StopCheck
    keys_11 = {"resnet18", "mobilenet_v3_small", "dinov2_small"}
    if not isinstance(loaded_11, dict) or set(loaded_11) != keys_11:
        print(f"❌ Ex B1.11 : attendu un dictionnaire aux clés {sorted(keys_11)}, reçu {type(loaded_11).__name__} "
              f"{sorted(loaded_11) if isinstance(loaded_11, dict) else ''}.")
        raise StopCheck
    for name_11 in sorted(keys_11):
        if not (isinstance(loaded_11[name_11], (tuple, list)) and len(loaded_11[name_11]) == 2):
            print(f"❌ Ex B1.11 : loaded['{name_11}'] doit être un couple (modèle, prétraitement).")
            raise StopCheck
    for name_11, weights_11 in (("resnet18", ResNet18_Weights.IMAGENET1K_V1),
                                ("mobilenet_v3_small", MobileNet_V3_Small_Weights.IMAGENET1K_V1)):
        model_11, prep_11 = loaded_11[name_11]
        ref_11 = transfer.load_backbone(name_11).full.state_dict()
        mine_11 = model_11.state_dict()
        same_11 = set(mine_11) == set(ref_11) and all(torch.equal(mine_11[k], ref_11[k]) for k in ref_11)
        verdict("B1.11", same_11, f"{name_11} : les poids IMAGENET1K_V1, tête ImageNet comprise.",
                f"{name_11} : ce ne sont pas les poids IMAGENET1K_V1 avec leur tête ImageNet (autres poids, tête "
                "remplacée, ou poids modifiés ?).")
        verdict("B1.11", not model_11.training, f"{name_11} est en mode eval().",
                f"{name_11} est en mode train() : un modèle torchvision se crée en mode entraînement.")
        verdict("B1.11", str(prep_11) == str(weights_11.transforms()), f"{name_11} : le prétraitement de ses poids.",
                f"{name_11} : attendu le prétraitement de ses poids, reçu {str(prep_11)[:120]!r}.")
    dino_ref_11 = transfer.try_backbone("dinov2_small", "B1.11")
    if dino_ref_11 is not None:
        model_11, processor_11 = loaded_11["dinov2_small"]
        mine_11, ref_11 = model_11.state_dict(), dino_ref_11.full.state_dict()
        same_11 = set(mine_11) == set(ref_11) and all(torch.equal(mine_11[k], ref_11[k]) for k in ref_11)
        verdict("B1.11", same_11 and getattr(getattr(model_11, "config", None), "hidden_size", None) == 384,
                "dinov2_small : le backbone de facebook/dinov2-small, sans tête.",
                "dinov2_small : ce n'est pas le backbone de facebook/dinov2-small chargé par AutoModel.")
        verdict("B1.11", not model_11.training, "dinov2_small est en mode eval().", "dinov2_small est en mode train().")
        crop_11 = getattr(processor_11, "crop_size", None)
        verdict("B1.11", getattr(processor_11, "resample", None) == 3 and getattr(crop_11, "height", None) == 224
                and tuple(getattr(processor_11, "image_mean", ())) == wb.datasets.IMAGENET_MEAN,
                "dinov2_small : son processeur d'images (recadrage 224, interpolation bicubique, statistiques d'ImageNet).",
                f"dinov2_small : attendu le processeur d'images du dépôt (AutoImageProcessor), reçu {type(processor_11).__name__}.")'''

RECORD_11 = r'''wb.record("B1.11a", mobilenet_in_11, mistakes={
    "c'est la taille entre les deux couches denses : on demande l'entrée de classifier": 1024,
    "c'est le nombre de sorties de la tête ImageNet": 1000,
    "960, c'est MobileNetV3-large : relis le nom du modèle": 960})
wb.record("B1.11b", mobilenet_mid_11, mistakes={
    "c'est la taille à l'entrée de classifier : on demande celle entre ses deux couches denses": 576,
    "c'est le nombre de sorties de la tête ImageNet": 1000,
    "1 280, c'est MobileNetV3-large : relis le nom du modèle": 1280})
wb.record("B1.11c", dino_d_11, mistakes={
    "768, c'est DINOv2-base : relis le nom du dépôt": 768,
    "257 est un nombre de tokens, pas la taille de chacun": 257,
    "c'est la taille cachée du MLP de chaque bloc : lis hidden_size dans la configuration": 1536})
wb.record("B1.11d", pooler_is_cls_11, mistakes={
    "compare les deux tenseurs avec torch.allclose, sur une même passe avant de DINOv2": False})
wb.record("B1.11e", resize_11, mistakes={
    "224 est la taille du recadrage (crop_size)": 224,
    "232 est le redimensionnement d'autres poids (encadré 🧮 de la fiche) : lis ceux de ResNet-18": 232})
wb.record("B1.11f", bicubic_11, mistakes={
    "print(preprocess) affiche l'interpolation des deux CNN : relis-la": "ABC",
    "les deux CNN ont-ils la même interpolation que DINOv2 ? relis print(preprocess)": "AC",
    "MobileNetV3-small a-t-il la même interpolation que DINOv2 ? relis print(preprocess)": "BC",
    "relis l'interpolation des deux CNN, puis le champ resample du processeur de DINOv2": "AB",
    "relis l'interpolation de ResNet-18, puis le champ resample du processeur de DINOv2": "A",
    "relis l'interpolation de MobileNetV3-small, puis le champ resample du processeur de DINOv2": "B"})'''

TODO_12 = r'''probe32_12 = ...   # a) the linear probe accuracy on the embeddings of the images at 32 x 32 (NOT resized):
                   #    "A" below 30 %, "B" from 30 to 70 %, "C" above 70 %
top5_12 = ...      # b) at 32 x 32, the share of the test images whose predicted ImageNet class is one of the 5 most
                   #    frequent predicted classes: "A" below 30 %, "B" from 30 to 70 %, "C" above 70 %'''

EXPERIMENT_12 = r'''with checked("B1.12"):
    rows_12 = []
    for size_12 in (32, SIZE):
        E_train_12 = reference_embeddings(X_TRAIN, size_12, "train")
        E_test_12 = reference_embeddings(X_TEST, size_12, "test")
        acc_12 = reference_probe(E_train_12, Y_TRAIN, E_test_12, Y_TEST)
        with torch.inference_mode():
            imagenet_12 = RESNET.full.fc(torch.from_numpy(E_test_12)).argmax(1).numpy()   # the ImageNet head
        names_12, counts_12 = np.unique(imagenet_12, return_counts=True)
        top_12 = np.argsort(-counts_12, kind="stable")[:5]
        share_12 = counts_12[top_12].sum() / len(imagenet_12)
        print(f"{size_12:>3} pixels : linear probe {pct(acc_12)} ; {len(names_12)} classes ImageNet différentes prédites "
              f"pour {len(imagenet_12)} images ; les 5 plus fréquentes ({', '.join(RESNET.categories[names_12[k]] for k in top_12)}) "
              f"couvrent {pct(share_12)} des images.")
    print("(le hasard : 10 % ; linear probe entraîné sur", len(X_TRAIN), "images, mesuré sur", len(X_TEST), ")")
    wb.check("B1.12a", probe32_12)
    wb.check("B1.12b", top5_12)'''

TODO_13 = r'''def make_backbone_13():
    """Your ResNet-18 of B1.11 turned into a backbone: IMAGENET1K_V1 weights, fc replaced by nn.Identity(), eval()."""
    raise NotImplementedError("make_backbone_13")


def embed_13(backbone, images, size, batch_size=64):
    """Frozen embeddings of CIFAR-10 images: a float32 NumPy array (N, d).

    backbone: a network whose output is the embedding, in eval() mode; images: uint8 (N, 32, 32, 3), left unchanged;
    size: the side of the resized images. By mini-batches of batch_size, without gradient: pixels in [0, 1],
    channels first, resized to size x size (bilinear), normalised with the ImageNet statistics."""
    raise NotImplementedError("embed_13")


def probe_13(E_train, y_train, E_test, y_test, C=1.0):
    """Test accuracy (a float) of a linear probe: StandardScaler, then LogisticRegression(C=C, max_iter=2000)."""
    raise NotImplementedError("probe_13")


def knn_13(E_train, y_train, E_test, y_test, k=10):
    """Test accuracy (a float) of the k nearest neighbours for the cosine similarity."""
    raise NotImplementedError("knn_13")'''

SOLUTION_13 = r'''from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

MEAN_13 = torch.tensor(wb.datasets.IMAGENET_MEAN).view(1, 3, 1, 1)
STD_13 = torch.tensor(wb.datasets.IMAGENET_STD).view(1, 3, 1, 1)


def make_backbone_13():
    """Your ResNet-18 of B1.11 turned into a backbone: IMAGENET1K_V1 weights, fc replaced by nn.Identity(), eval()."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()                                  # phi(x): the 512 values after the global average pooling
    return model.eval()


def embed_13(backbone, images, size, batch_size=64):
    """Frozen embeddings of CIFAR-10 images: a float32 NumPy array (N, d).

    backbone: a network whose output is the embedding, in eval() mode; images: uint8 (N, 32, 32, 3), left unchanged;
    size: the side of the resized images. By mini-batches of batch_size, without gradient: pixels in [0, 1],
    channels first, resized to size x size (bilinear), normalised with the ImageNet statistics."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i:i + batch_size]).permute(0, 3, 1, 2).float() / 255   # NHWC -> NCHW, [0, 1]
            if size != x.shape[-1]:
                x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            out.append(backbone((x - MEAN_13) / STD_13))
    return torch.cat(out).numpy()


def probe_13(E_train, y_train, E_test, y_test, C=1.0):
    """Test accuracy (a float) of a linear probe: StandardScaler, then LogisticRegression(C=C, max_iter=2000)."""
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return float(probe.fit(E_train, y_train).score(E_test, y_test))


def knn_13(E_train, y_train, E_test, y_test, k=10):
    """Test accuracy (a float) of the k nearest neighbours for the cosine similarity."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    return float(knn.fit(E_train, y_train).score(E_test, y_test))'''

CHECK_13 = r'''with checked("B1.13"):
    backbone_13 = make_backbone_13()
    if not returned("B1.13", "make_backbone_13", backbone_13):
        raise StopCheck
    verdict("B1.13", not backbone_13.training, "le backbone est en mode eval().",
            "le backbone est en mode train() : ses batchnorms normaliseraient avec les statistiques de chaque mini-batch.")
    images_13 = X_TEST[:16].copy()
    E_13 = embed_13(backbone_13, images_13, SIZE)
    if not returned("B1.13", "embed_13", E_13):
        raise StopCheck
    verdict("B1.13", np.array_equal(images_13, X_TEST[:16]), "embed_13 laisse les images intactes.",
            "embed_13 a modifié les images reçues : travaille sur une copie convertie.")
    if not isinstance(E_13, np.ndarray):
        print(f"❌ Ex B1.13 : embed_13 doit rendre un tableau NumPy, reçu {type(E_13).__name__}"
              + (" (convertis avec .numpy())." if torch.is_tensor(E_13) else "."))
        raise StopCheck
    want_13 = RESNET.embed(X_TEST[:16], SIZE)
    if E_13.shape != want_13.shape:
        print(f"❌ Ex B1.13 : embed_13(backbone, X_TEST[:16], SIZE) : attendu la forme {want_13.shape}, reçu {E_13.shape}.")
        raise StopCheck
    if not np.allclose(E_13, want_13, rtol=1e-3, atol=1e-3):
        variants_13 = {   # what a frequent slip gives, to say where to look
            "sans la normalisation d'ImageNet": transfer.to_input(X_TEST[:16], SIZE, (0, 0, 0), (1, 1, 1)),
            "avec une interpolation bicubique": transfer.to_input(X_TEST[:16], SIZE, resample="bicubic"),
            "sans agrandir les images": transfer.to_input(X_TEST[:16], None),
        }
        with torch.inference_mode():
            hit_13 = [k for k, x in variants_13.items() if np.allclose(E_13, RESNET.model(x).numpy(), atol=1e-3)]
        print("❌ Ex B1.13 : embed_13 ne rend pas les embeddings du contrat"
              + (f" : tes valeurs sont celles d'images préparées {hit_13[0]}." if hit_13 else
                 " : vérifie l'ordre des axes, la division par 255, l'agrandissement à size × size et la normalisation."))
        raise StopCheck
    verdict("B1.13", E_13.dtype == np.float32, "embed_13 rend les embeddings attendus, en float32.",
            f"embed_13 rend du {E_13.dtype} au lieu de float32.")
    E_small_13 = embed_13(backbone_13, X_TEST[:16].copy(), SIZE, batch_size=5)
    verdict("B1.13", isinstance(E_small_13, np.ndarray) and np.allclose(E_small_13, E_13, atol=1e-4),
            "l'embedding d'une image ne dépend pas des images de son mini-batch.",
            "avec batch_size=5, les embeddings changent : le backbone est-il en mode eval(), toutes les images "
            "sont-elles rendues ?")

with checked("B1.13"):
    # the probes on the cached embeddings of the reference backbone (the same as yours, computed once)
    E_train_13 = reference_embeddings(X_TRAIN, SIZE, "train")
    E_test_13 = reference_embeddings(X_TEST, SIZE, "test")
    for name_13, fn_13, kwargs_13, want_13 in (
            ("probe_13", probe_13, {}, reference_probe(E_train_13, Y_TRAIN, E_test_13, Y_TEST)),
            ("probe_13 avec C=0,01", probe_13, {"C": 0.01}, reference_probe(E_train_13, Y_TRAIN, E_test_13, Y_TEST, C=0.01)),
            ("knn_13", knn_13, {}, None), ("knn_13 avec k=1", knn_13, {"k": 1}, None)):
        if want_13 is None:
            from sklearn.neighbors import KNeighborsClassifier
            want_13 = float(KNeighborsClassifier(n_neighbors=kwargs_13.get("k", 10), metric="cosine")
                            .fit(E_train_13, Y_TRAIN).score(E_test_13, Y_TEST))
        got_13 = fn_13(E_train_13.copy(), Y_TRAIN.copy(), E_test_13.copy(), Y_TEST.copy(), **kwargs_13)
        if not returned("B1.13", name_13, got_13):
            continue
        if isinstance(got_13, bool) or not isinstance(got_13, (float, np.floating)) or not 0 <= got_13 <= 1:
            print(f"❌ Ex B1.13 : {name_13} doit rendre une accuracy (un float entre 0 et 1), reçu {got_13!r}.")
            continue
        verdict("B1.13", abs(got_13 - want_13) < 0.003, f"{name_13} : {pct(got_13)} sur les {len(Y_TEST)} images de test.",
                f"{name_13} : {pct(got_13)}, attendu {pct(want_13)} : relis le contrat (standardisation, C, "
                "similarité cosinus, k).")'''

EXPERIMENT_13 = r'''# Choose C and k on the VALIDATION images (never on the test), then measure the chosen pair once on the test
with checked("B1.13"):
    E_val_13 = reference_embeddings(X_VAL, SIZE, "val")
    E_train_13 = reference_embeddings(X_TRAIN, SIZE, "train")
    E_test_13 = reference_embeddings(X_TEST, SIZE, "test")
    val_probe_13 = {C: probe_13(E_train_13, Y_TRAIN, E_val_13, Y_VAL, C=C) for C in (0.01, 0.1, 1.0)}
    val_knn_13 = {k: knn_13(E_train_13, Y_TRAIN, E_val_13, Y_VAL, k=k) for k in (1, 5, 10, 25)}
    print("validation, linear probe :", " ; ".join(f"C = {fr(C, 2)} : {pct(a)}" for C, a in val_probe_13.items()))
    print("validation, k-NN cosinus :", " ; ".join(f"k = {k} : {pct(a)}" for k, a in val_knn_13.items()))
    best_C_13 = max(val_probe_13, key=val_probe_13.get)
    best_k_13 = max(val_knn_13, key=val_knn_13.get)
    print(f"test : linear probe (C = {fr(best_C_13, 2)}) {pct(probe_13(E_train_13, Y_TRAIN, E_test_13, Y_TEST, C=best_C_13))} ; "
          f"k-NN (k = {best_k_13}) {pct(knn_13(E_train_13, Y_TRAIN, E_test_13, Y_TEST, k=best_k_13))}")'''

TODO_14 = r'''def resolution_sweep_14(backbone, sizes, train, test, repeats=3):
    """For each size of `sizes`, in that order, a dict {"size": size, "accuracy": float, "ms_per_image": float}.

    train and test are pairs (images uint8 (N, 32, 32, 3), labels). accuracy: probe_13 on the embeddings of embed_13
    at that size; ms_per_image: the time of embed_13 on the first 64 test images, the SMALLEST of `repeats`
    measures, divided by 64, in milliseconds."""
    raise NotImplementedError("resolution_sweep_14")'''

SOLUTION_14 = r'''def resolution_sweep_14(backbone, sizes, train, test, repeats=3):
    """For each size of `sizes`, in that order, a dict {"size": size, "accuracy": float, "ms_per_image": float}.

    train and test are pairs (images uint8 (N, 32, 32, 3), labels). accuracy: probe_13 on the embeddings of embed_13
    at that size; ms_per_image: the time of embed_13 on the first 64 test images, the SMALLEST of `repeats`
    measures, divided by 64, in milliseconds."""
    (X_tr, y_tr), (X_te, y_te) = train, test
    rows = []
    for size in sizes:
        accuracy = probe_13(embed_13(backbone, X_tr, size), y_tr, embed_13(backbone, X_te, size), y_te)
        batch = X_te[:64]
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            embed_13(backbone, batch, size)
            times.append(time.perf_counter() - start)
        rows.append({"size": size, "accuracy": accuracy, "ms_per_image": 1000 * min(times) / len(batch)})
    return rows'''

CHECK_14 = r'''class SlowFirstCall_14(nn.Module):
    """A probe around a backbone: the first call at each image size waits 0.5 s more (a cold start); it also times
    its other calls, image by image."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.seen, self.fast_ms = inner, set(), {}

    def forward(self, x):
        if x.shape[-1] not in self.seen:
            self.seen.add(x.shape[-1])
            time.sleep(0.5)
            return self.inner(x)
        start = time.perf_counter()
        out = self.inner(x)
        ms = 1000 * (time.perf_counter() - start) / len(x)
        self.fast_ms[x.shape[-1]] = min(ms, self.fast_ms.get(x.shape[-1], ms))
        return out


with checked("B1.14"):
    small_tr_14 = (X_TRAIN[nested(10)], Y_TRAIN[nested(10)])
    small_te_14 = (X_TEST[nested(10, "test")], Y_TEST[nested(10, "test")])
    probe_net_14 = SlowFirstCall_14(RESNET.model)
    rows_14 = resolution_sweep_14(probe_net_14, [48, 32], small_tr_14, small_te_14, repeats=3)
    if not returned("B1.14", "resolution_sweep_14", rows_14):
        raise StopCheck
    ok_14 = (isinstance(rows_14, list) and len(rows_14) == 2 and all(isinstance(r, dict) for r in rows_14)
             and all({"size", "accuracy", "ms_per_image"} <= set(r) for r in rows_14))
    if not ok_14:
        print('❌ Ex B1.14 : attendu une liste de deux dictionnaires {"size", "accuracy", "ms_per_image"} pour deux '
              f"tailles, reçu {type(rows_14).__name__} {rows_14!r:.200}.")
        raise StopCheck
    verdict("B1.14", [r["size"] for r in rows_14] == [48, 32], "une ligne par taille, dans l'ordre reçu.",
            f"tailles rendues {[r['size'] for r in rows_14]}, attendu [48, 32] (l'ordre reçu).")
    for r_14 in rows_14:
        want_14 = reference_probe(RESNET.embed(small_tr_14[0], r_14["size"]), small_tr_14[1],
                                  RESNET.embed(small_te_14[0], r_14["size"]), small_te_14[1])
        acc_ok_14 = isinstance(r_14["accuracy"], (float, np.floating)) and abs(float(r_14["accuracy"]) - want_14) < 0.011
        verdict("B1.14", acc_ok_14,
                f"{r_14['size']} pixels : accuracy {pct(float(r_14['accuracy']))}, comme le linear probe de référence.",
                f"{r_14['size']} pixels : accuracy {r_14['accuracy']!r:.40}, attendu {pct(want_14)} (probe_13 sur les "
                "embeddings de embed_13 à cette taille).")
    for r_14 in rows_14:
        ms_14, fast_14 = float(r_14["ms_per_image"]), probe_net_14.fast_ms.get(r_14["size"])
        if fast_14 is None:
            print(f"❌ Ex B1.14 : {r_14['size']} pixels : le backbone n'a été appelé qu'une fois à cette taille : "
                  "mesure plusieurs fois.")
            continue
        verdict("B1.14", 0.5 * fast_14 <= ms_14 <= 1.5 * fast_14 + 1.0,
                f"{r_14['size']} pixels : {fr(ms_14, 2)} ms par image ; le démarrage lent du premier appel n'est pas compté.",
                f"{r_14['size']} pixels : {fr(ms_14, 4)} ms par image, alors qu'une passe chaude de ce backbone prend "
                f"{fr(fast_14, 2)} ms par image et que le premier appel à chaque taille dure ici 0,5 s de plus : garde "
                "la plus petite de plusieurs mesures, en millisecondes par image.")'''


EXPERIMENT_14 = r'''SWEEP_SIZES = (32, 64, 96, 128, 160, 224)
SWEEP_PER_CLASS = 30 if FAST_MODE else 200     # train and test images per class of the sweep
with checked("B1.14"):
    need("B1.14", "resolution_sweep_14", "make_backbone_13")
    tr_14, te_14 = nested(SWEEP_PER_CLASS), nested(SWEEP_PER_CLASS, "test")
    sweep_14 = resolution_sweep_14(make_backbone_13(), SWEEP_SIZES, (X_TRAIN[tr_14], Y_TRAIN[tr_14]),
                                   (X_TEST[te_14], Y_TEST[te_14]))
    print(f"{'pixels':>6} {'accuracy':>9} {'ms par image':>13}")
    for r_14 in sweep_14:
        print(f"{r_14['size']:>6} {pct(r_14['accuracy']):>9} {fr(r_14['ms_per_image'], 2):>13}")
    acc_14 = np.array([r["accuracy"] for r in sweep_14])
    ms_14 = np.array([r["ms_per_image"] for r in sweep_14])
    fig_14, axes_14 = plt.subplots(1, 2, figsize=(10, 3.6))
    axes_14[0].plot(SWEEP_SIZES, acc_14, "o-")
    axes_14[0].set(xlabel="côté des images (pixels)", ylabel="accuracy du linear probe", title="Accuracy selon la résolution")
    axes_14[1].plot(ms_14, acc_14, "o-")
    for s_14, m_14, a_14 in zip(SWEEP_SIZES, ms_14, acc_14):
        axes_14[1].annotate(str(s_14), (m_14, a_14), textcoords="offset points", xytext=(4, -10), fontsize=8)
    axes_14[1].set(xscale="log", xlabel="temps par image (ms, échelle log)", ylabel="accuracy du linear probe",
                   title="Accuracy contre coût")
    plt.tight_layout()
    plt.show()
    verdict("B1.14", acc_14[-1] - acc_14[0] > 0.1 and ms_14[-1] > ms_14[1],
            "l'accuracy et le temps montent avec la résolution.",
            "ni l'accuracy ni le temps ne montent avec la résolution : relis ton code avant de conclure.")'''

GIVEN_15 = r'''# Six projections of the same 600 test images (60 per class): PCA and UMAP of (A) the raw pixels, (B) the embeddings
# of ResNet-18 (SIZE pixels) and (C) those of DINOv2-small (DINO_SIZE pixels); one colour per class
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

wb.ensure("umap-learn")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")        # umap imports tqdm, which warns about Jupyter widgets
    import umap

DINO = transfer.try_backbone("dinov2_small", "B1.15")
idx_15 = nested(60, "test")
y_15 = Y_TEST[idx_15]
REPS_15 = {"(A) pixels bruts": X_TEST[idx_15].reshape(len(idx_15), -1) / 255.0,
           "(B) ResNet-18": reference_embeddings(X_TEST, SIZE, "test")[idx_15]}
if DINO is not None:   # the embeddings of the whole X_TEST, computed once (≈ 1 min) and reused by parts C and D
    DINO_TEST = transfer.cached_embeddings(DINO, X_TEST, size=DINO_SIZE, subset="test", verbose=False)
    REPS_15["(C) DINOv2-small"] = DINO_TEST[idx_15]
PROJ_15 = {}
for name_15, E_15 in REPS_15.items():
    Z_15 = StandardScaler().fit_transform(E_15)
    PROJ_15[name_15] = {"PCA": PCA(2, svd_solver="full").fit_transform(Z_15),
                        "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0, n_jobs=1).fit_transform(Z_15)}
fig_15, axes_15 = plt.subplots(2, len(PROJ_15), figsize=(4.2 * len(PROJ_15), 8.2), squeeze=False)
colors_15 = plt.cm.tab10(np.arange(10))
for j_15, (name_15, proj_15) in enumerate(PROJ_15.items()):
    for i_15, method_15 in enumerate(("PCA", "UMAP")):
        ax_15 = axes_15[i_15, j_15]
        for c_15 in range(10):
            pts_15 = proj_15[method_15][y_15 == c_15]
            ax_15.scatter(pts_15[:, 0], pts_15[:, 1], s=7, color=colors_15[c_15], label=CLASSES[c_15])
        ax_15.set_title(f"{name_15} — {method_15}", fontsize=10)
        ax_15.set_xticks([])
        ax_15.set_yticks([])
axes_15[0, -1].legend(fontsize=7, markerscale=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()'''

TODO_15 = r'''separated_15 = ...       # a) the letter of the representation whose UMAP shows the best-separated groups: "A", "B" or "C"
pixel_classes_15 = ...   # b) True or False: the groups seen in the UMAP of the raw pixels (A) are classes
vehicles_15 = ...        # c) True or False: in the UMAP of (B), the four vehicles (airplane, automobile, ship, truck)
                         #    form a region apart from the six animals
pca_says_15 = ...        # d) True or False: the PCA of (B) mixes the classes, so a linear classifier on these
                         #    embeddings cannot do much better than chance'''

CHECK_15 = r'''for id_15, value_15 in (("B1.15a", separated_15), ("B1.15b", pixel_classes_15), ("B1.15c", vehicles_15), ("B1.15d", pca_says_15)):
    with wb.attempt(id_15):
        wb.check(id_15, value_15)'''

MEASURES_15 = r'''if not filled(separated_15, pixel_classes_15, vehicles_15, pca_says_15):
    print("⏳ Ex B1.15 : les mesures s'affichent une fois tes quatre réponses écrites.")
else:
    # what the figure was showing, measured: the accuracy of the 10 nearest neighbours IN the plane of each projection
    # (5-fold cross-validation), and the share of images put in the right family (vehicles or animals)
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    vehicle_15 = np.isin(y_15, [0, 1, 8, 9])
    for name_15, proj_15 in PROJ_15.items():
        line_15 = []
        for method_15 in ("PCA", "UMAP"):
            pred_15 = cross_val_predict(KNeighborsClassifier(10), proj_15[method_15], y_15, cv=5)
            family_15 = np.mean(np.isin(pred_15, [0, 1, 8, 9]) == vehicle_15)
            line_15.append(f"{method_15} : 10-NN dans le plan {pct(np.mean(pred_15 == y_15), 0)}, bonne famille {pct(family_15, 0)}")
        print(f"{name_15} — " + " ; ".join(line_15))
    print("(le hasard : 10 % pour la classe ; le linear probe de B1.13, sur les 512 dimensions de (B) : plus de 70 %)")'''

GIVEN_16 = r'''# Two datasets for the probes, block by block: CIFAR-10 (close to ImageNet) and Fashion-MNIST (far: grayscale clothes)
from wb.transfer import RESNET_BLOCKS, to_input   # to_input(images, size): what embed_13 does, grayscale accepted

BLOCK_PER_CLASS = 50 if FAST_MODE else 300          # train and test images per class
XC_TR_16, YC_TR_16 = X_TRAIN[nested(BLOCK_PER_CLASS)], Y_TRAIN[nested(BLOCK_PER_CLASS)]
XC_TE_16, YC_TE_16 = X_TEST[nested(BLOCK_PER_CLASS, "test")], Y_TEST[nested(BLOCK_PER_CLASS, "test")]
XF_TR_16, YF_TR_16 = wb.datasets.load_fashion_mnist_subset(BLOCK_PER_CLASS, "train")   # uint8 (N, 28, 28)
XF_TE_16, YF_TE_16 = wb.datasets.load_fashion_mnist_subset(BLOCK_PER_CLASS, "test")
print("blocs :", RESNET_BLOCKS, "; to_input(XF_TR_16[:2], SIZE).shape =", tuple(to_input(XF_TR_16[:2], SIZE).shape))'''

TODO_16 = r'''def block_embeddings_16(model, images, size, blocks=RESNET_BLOCKS, batch_size=64):
    """{block: float32 NumPy array (N, C)}: the global average pooling of the output of each block of `blocks`.

    model: a torchvision ResNet-18 in eval() mode (its head is not used); images: uint8, in colour (N, H, W, 3) or
    grayscale (N, H, W), each mini-batch prepared by to_input(batch, size); "stem" is the output of model.maxpool.
    ONE forward pass per mini-batch for all the blocks, without gradient."""
    raise NotImplementedError("block_embeddings_16")


def probe_by_block_16(model, train, test, size, blocks=RESNET_BLOCKS):
    """{block: test accuracy of probe_13 on the embeddings of that block}; train and test are pairs (images, labels)."""
    raise NotImplementedError("probe_by_block_16")'''

SOLUTION_16 = r'''from torchvision.models.feature_extraction import create_feature_extractor

STEM_NODE_16 = "maxpool"


def block_embeddings_16(model, images, size, blocks=RESNET_BLOCKS, batch_size=64):
    """{block: float32 NumPy array (N, C)}: the global average pooling of the output of each block of `blocks`.

    model: a torchvision ResNet-18 in eval() mode (its head is not used); images: uint8, in colour (N, H, W, 3) or
    grayscale (N, H, W), each mini-batch prepared by to_input(batch, size); "stem" is the output of model.maxpool.
    ONE forward pass per mini-batch for all the blocks, without gradient."""
    nodes = {(STEM_NODE_16 if b == "stem" else b): b for b in blocks}
    extractor = create_feature_extractor(model, return_nodes=nodes)
    parts = {b: [] for b in blocks}
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            maps = extractor(to_input(images[i:i + batch_size], size))
            for b in blocks:
                parts[b].append(maps[b].mean(dim=(2, 3)))       # global average pooling: one value per channel
    return {b: torch.cat(parts[b]).numpy() for b in blocks}


def probe_by_block_16(model, train, test, size, blocks=RESNET_BLOCKS):
    """{block: test accuracy of probe_13 on the embeddings of that block}; train and test are pairs (images, labels)."""
    E_tr = block_embeddings_16(model, train[0], size, blocks)
    E_te = block_embeddings_16(model, test[0], size, blocks)
    return {b: probe_13(E_tr[b], train[1], E_te[b], test[1]) for b in blocks}'''

CHECK_16 = r'''with checked("B1.16"):
    calls_16 = []
    hook_16 = RESNET.full.conv1.register_forward_hook(lambda m, i, o: calls_16.append(o.shape[0]))
    try:
        for name_16, sample_16 in (("CIFAR-10", X_TEST[:6].copy()), ("Fashion-MNIST", XF_TE_16[:6].copy())):
            want_16 = RESNET.embed_blocks(sample_16, 64, batch_size=4)
            calls_16.clear()                             # count the passes of YOUR function only
            got_16 = block_embeddings_16(RESNET.full, sample_16, 64, batch_size=4)
            passes_16 = list(calls_16)
            if not returned("B1.16", "block_embeddings_16", got_16):
                raise StopCheck
            if not isinstance(got_16, dict) or set(got_16) != set(RESNET_BLOCKS):
                print(f"❌ Ex B1.16 : attendu un dictionnaire aux clés {list(RESNET_BLOCKS)}, reçu "
                      f"{sorted(got_16) if isinstance(got_16, dict) else type(got_16).__name__}.")
                raise StopCheck
            ok_16 = all(same("B1.16", f"{name_16}, bloc {b}", got_16[b], want_16[b], rtol=1e-4, atol=1e-4)
                        for b in RESNET_BLOCKS)
            verdict("B1.16", passes_16 == [4, 2], f"{name_16} : une seule passe avant par mini-batch pour les cinq blocs.",
                    f"{name_16} : {len(passes_16)} passes dans conv1 pour 6 images en mini-batches de 4 ({passes_16}) : "
                    "attendu une passe par mini-batch, tous les blocs à la fois.")
    finally:
        hook_16.remove()
    tr_16, te_16 = nested(10), nested(10, "test")
    probes_16 = probe_by_block_16(RESNET.full, (X_TRAIN[tr_16], Y_TRAIN[tr_16]), (X_TEST[te_16], Y_TEST[te_16]), 64,
                                  blocks=("stem", "layer4"))
    if not returned("B1.16", "probe_by_block_16", probes_16):
        raise StopCheck
    if not isinstance(probes_16, dict) or set(probes_16) != {"stem", "layer4"}:
        print(f"❌ Ex B1.16 : probe_by_block_16(..., blocks=('stem', 'layer4')) doit rendre un dictionnaire à ces deux "
              f"clés, reçu {probes_16!r:.120}.")
        raise StopCheck
    for b_16 in ("stem", "layer4"):
        want_16 = reference_probe(RESNET.embed(X_TRAIN[tr_16], 64, b_16), Y_TRAIN[tr_16],
                                  RESNET.embed(X_TEST[te_16], 64, b_16), Y_TEST[te_16])
        verdict("B1.16", abs(float(probes_16[b_16]) - want_16) < 0.011,
                f"probe_by_block_16, bloc {b_16} : {pct(float(probes_16[b_16]))}, comme la référence.",
                f"probe_by_block_16, bloc {b_16} : "
                + (pct(float(probes_16[b_16])) if isinstance(probes_16[b_16], (float, np.floating)) else repr(probes_16[b_16]))
                + f", attendu {pct(want_16)}.")'''

EXPERIMENT_16 = r'''with checked("B1.16"):
    need("B1.16", "probe_by_block_16")
    model_16 = make_backbone_13()
    results_16 = {"CIFAR-10": probe_by_block_16(model_16, (XC_TR_16, YC_TR_16), (XC_TE_16, YC_TE_16), SIZE),
                  "Fashion-MNIST": probe_by_block_16(model_16, (XF_TR_16, YF_TR_16), (XF_TE_16, YF_TE_16), SIZE)}
    print(f"{'bloc':>8} " + " ".join(f"{name:>14}" for name in results_16))
    for b_16 in RESNET_BLOCKS:
        print(f"{b_16:>8} " + " ".join(f"{pct(results_16[name][b_16]):>14}" for name in results_16))
    plt.figure(figsize=(7, 3.8))
    for name_16, accs_16 in results_16.items():
        plt.plot(list(RESNET_BLOCKS), [accs_16[b] for b in RESNET_BLOCKS], "o-", label=name_16)
    plt.ylabel("accuracy du linear probe (test)")
    plt.xlabel("bloc sondé (sortie, pooling moyen global)")
    plt.title(f"Linear probe bloc par bloc (ResNet-18 gelé, {BLOCK_PER_CLASS} images par classe)", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.show()'''

# --- Part B -----------------------------------------------------------------------------------------------------------
GIVEN_B = r'''# The data and the helpers of part B: fine-tuning ResNet-18 on CIFAR-10
from wb.transfer import to_input

FT_PER_CLASS = 50 if FAST_MODE else 1000     # training images per class of the fine-tunings (nested in X_TRAIN)
FT_EPOCHS = 3 if FAST_MODE else 10
XT_B, YT_B = X_TRAIN[nested(FT_PER_CLASS)], Y_TRAIN[nested(FT_PER_CLASS)]
IMAGENET_RESNET18 = RESNET.full.state_dict()       # the pre-trained weights, ImageNet head included (for the checks)


def batches(X, y, batch_size, seed):
    """Shuffled mini-batches (x, y) of uint8 images X: x prepared by to_input(..., SIZE) (normalised), both on DEVICE.
    The order depends on `seed` only."""
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)).numpy()
    for i in range(0, len(X), batch_size):
        idx = order[i:i + batch_size]
        yield to_input(X[idx], SIZE).to(DEVICE), torch.from_numpy(y[idx]).to(DEVICE)


def evaluate(model, X, y, batch_size=128):
    """Accuracy of `model` on uint8 images X at SIZE pixels, in eval() mode, without gradient (the model stays in
    eval() mode afterwards)."""
    model.eval()
    correct = 0
    with torch.inference_mode():
        for i in range(0, len(X), batch_size):
            pred = model(to_input(X[i:i + batch_size], SIZE).to(DEVICE)).argmax(1).cpu().numpy()
            correct += int((pred == y[i:i + batch_size]).sum())
    return correct / len(X)


def changed(a, b):
    """True when two tensors differ."""
    return not torch.equal(a.detach().cpu(), b.detach().cpu())


print(f"fine-tuning : {len(XT_B)} images d'entraînement ({FT_PER_CLASS} par classe), {len(X_VAL)} de validation, "
      f"{FT_EPOCHS} epochs au plus, {SIZE} pixels")'''

TODO_17 = r'''def param_groups_17(modules, head_lr, gamma, weight_decay=0.01):
    """Parameter groups for AdamW. modules: a list of modules from the head DOWN (e.g. [model.fc, model.layer4]); the
    k-th one gets the learning rate head_lr * gamma**k. Each module gives (at most) two dicts {"params", "lr",
    "weight_decay"}: its parameters of dimension 2 or more with weight_decay, the others (biases, batchnorm parameters)
    with 0. Only the parameters that require a gradient; no empty group."""
    raise NotImplementedError("param_groups_17")


def make_partial_17(num_classes=10):
    """ResNet-18 IMAGENET1K_V1 with a new head nn.Linear(512, num_classes); every parameter frozen except those of
    layer4 and of the head. On DEVICE."""
    raise NotImplementedError("make_partial_17")


def freeze_batchnorm_17(model):
    """Put every batchnorm of `model` in eval() mode (the other modules keep their mode)."""
    raise NotImplementedError("freeze_batchnorm_17")


def train_partial_17(model, X, y, epochs, head_lr, gamma, batch_size=32, seed=0):
    """Fine-tunes the model of make_partial_17 on (X, y) and returns the mean training loss of each epoch (floats)."""
    raise NotImplementedError("train_partial_17")'''

SOLUTION_17 = r'''from torchvision.models import ResNet18_Weights, resnet18


def param_groups_17(modules, head_lr, gamma, weight_decay=0.01):
    """Parameter groups for AdamW. modules: a list of modules from the head DOWN (e.g. [model.fc, model.layer4]); the
    k-th one gets the learning rate head_lr * gamma**k. Each module gives (at most) two dicts {"params", "lr",
    "weight_decay"}: its parameters of dimension 2 or more with weight_decay, the others (biases, batchnorm parameters)
    with 0. Only the parameters that require a gradient; no empty group."""
    groups = []
    for k, module in enumerate(modules):
        lr = head_lr * gamma ** k
        params = [p for p in module.parameters() if p.requires_grad]
        decay = [p for p in params if p.ndim >= 2]                 # weights of the convolutions and dense layers
        no_decay = [p for p in params if p.ndim < 2]               # biases, gamma and beta of the batchnorms
        groups += [{"params": ps, "lr": lr, "weight_decay": wd}
                   for ps, wd in ((decay, weight_decay), (no_decay, 0.0)) if ps]
    return groups


def make_partial_17(num_classes=10):
    """ResNet-18 IMAGENET1K_V1 with a new head nn.Linear(512, num_classes); every parameter frozen except those of
    layer4 and of the head. On DEVICE."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)       # the new head, BEFORE the optimizer
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.layer4, model.fc):
        for p in module.parameters():
            p.requires_grad_(True)
    return model.to(DEVICE)


def freeze_batchnorm_17(model):
    """Put every batchnorm of `model` in eval() mode (the other modules keep their mode)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_partial_17(model, X, y, epochs, head_lr, gamma, batch_size=32, seed=0):
    """Fine-tunes the model of make_partial_17 on (X, y) and returns the mean training loss of each epoch (floats)."""
    optimizer = torch.optim.AdamW(param_groups_17([model.fc, model.layer4], head_lr, gamma))
    losses = []
    for epoch in range(epochs):
        model.train()                       # train() puts EVERY module in training mode...
        freeze_batchnorm_17(model)          # ...so the batchnorms go back to eval() right after, at each epoch
        total, count = 0.0, 0
        for xb, yb in batches(X, y, batch_size, seed + epoch):
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            count += len(yb)
        losses.append(total / count)
    return losses'''

CHECK_17 = r'''class SpyAdamW_17(torch.optim.AdamW):
    """torch.optim.AdamW, which also remembers the learning rate of each of its groups (for the check)."""
    seen = []

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        SpyAdamW_17.seen.append([(g["lr"], g["weight_decay"], [id(p) for p in g["params"]]) for g in self.param_groups])


def group_report_17(groups, modules, head_lr, gamma, weight_decay):
    """What is wrong with a list of parameter groups ("" when it follows the contract)."""
    if not isinstance(groups, list) or not all(isinstance(g, dict) and {"params", "lr", "weight_decay"} <= set(g)
                                               for g in groups):
        return 'attendu une liste de dictionnaires {"params", "lr", "weight_decay"}'
    owner = {}
    for k, module in enumerate(modules):
        for p in module.parameters():
            if p.requires_grad:
                owner[id(p)] = (k, p.ndim)
    seen = []
    for g in groups:
        params = list(g["params"])
        if not params:
            return "un groupe est vide"
        for p in params:
            if id(p) not in owner:
                return "un groupe contient un paramètre gelé ou d'un autre module"
            k, ndim = owner[id(p)]
            if not math.isclose(g["lr"], head_lr * gamma ** k, rel_tol=1e-9):
                return f"le module n° {k} (en partant de la tête) a le learning rate {g['lr']:.3g}, attendu head_lr × gamma^{k}"
            want = weight_decay if ndim >= 2 else 0.0
            if not math.isclose(g["weight_decay"], want, abs_tol=1e-12):
                return (f"un paramètre de dimension {ndim} a weight_decay={g['weight_decay']}, attendu {want} "
                        "(les biais et les paramètres des batchnorms n'en ont pas)")
            seen.append(id(p))
    if len(seen) != len(set(seen)):
        return "un paramètre est dans deux groupes"
    if set(seen) != set(owner):
        return f"{len(set(owner) - set(seen))} paramètre(s) entraînable(s) dans aucun groupe"
    return ""


with checked("B1.17"):
    toy_17 = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
    toy_17[0].bias.requires_grad_(False)
    modules_17 = [toy_17[2], nn.ModuleList([toy_17[0], toy_17[1]])]
    groups_17 = param_groups_17(modules_17, 1e-3, 0.5, weight_decay=0.05)
    if not returned("B1.17", "param_groups_17", groups_17):
        raise StopCheck
    problem_17 = group_report_17(groups_17, modules_17, 1e-3, 0.5, 0.05)
    verdict("B1.17", not problem_17, "param_groups_17 : learning rates par module, weight decay sur les poids seulement, "
            "paramètres gelés exclus.", f"param_groups_17 sur un petit réseau : {problem_17}.")
    if problem_17:
        raise StopCheck

with checked("B1.17"):
    model_17 = make_partial_17()
    if not returned("B1.17", "make_partial_17", model_17):
        raise StopCheck
    head_ok_17 = isinstance(getattr(model_17, "fc", None), nn.Linear) and model_17.fc.out_features == 10
    verdict("B1.17", head_ok_17, "une nouvelle tête à 10 sorties.", "attendu model.fc = nn.Linear(512, 10).")
    trainable_17 = {n for n, p in model_17.named_parameters() if p.requires_grad}
    want_17 = {n for n, _ in model_17.named_parameters() if n.startswith(("layer4.", "fc."))}
    verdict("B1.17", trainable_17 == want_17, "seuls layer4 et la tête sont entraînables.",
            f"{len(trainable_17 - want_17)} paramètre(s) entraînable(s) en trop, {len(want_17 - trainable_17)} gelé(s) "
            "à tort : seuls layer4 et la tête s'entraînent.")
    sd_17 = model_17.state_dict()
    verdict("B1.17", all(torch.equal(sd_17[k].cpu(), v) for k, v in IMAGENET_RESNET18.items() if not k.startswith("fc.")),
            "le backbone a les poids IMAGENET1K_V1.", "le backbone n'a pas les poids IMAGENET1K_V1.")
    model_17.train()
    freeze_batchnorm_17(model_17)
    bns_17 = [m for m in model_17.modules() if isinstance(m, nn.BatchNorm2d)]
    verdict("B1.17", bns_17 and not any(m.training for m in bns_17) and model_17.training and model_17.layer1.training,
            "freeze_batchnorm_17 : les 20 batchnorms en eval(), le reste en train().",
            "après model.train() puis freeze_batchnorm_17(model), attendu les batchnorms en eval() et le reste en train().")

with checked("B1.17"), torch.random.fork_rng():
    model_17 = make_partial_17()                       # a fresh model for two training steps
    if not returned("B1.17", "make_partial_17", model_17):
        raise StopCheck
    head_before_17 = model_17.fc.weight.detach().clone()
    SpyAdamW_17.seen.clear()
    adamw_17, torch.optim.AdamW = torch.optim.AdamW, SpyAdamW_17
    try:
        losses_17 = train_partial_17(model_17, X_TRAIN[nested(2)], Y_TRAIN[nested(2)], 1, 1e-3, 0.1, batch_size=10)
    finally:
        torch.optim.AdamW = adamw_17
    if not returned("B1.17", "train_partial_17", losses_17):
        raise StopCheck
    verdict("B1.17", isinstance(losses_17, list) and len(losses_17) == 1 and all(isinstance(v, float) and math.isfinite(v) for v in losses_17),
            f"train_partial_17 rend une loss par epoch ({fr(losses_17[0], 3) if isinstance(losses_17, list) and losses_17 else '?'}).",
            f"attendu une liste d'un float par epoch, reçu {losses_17!r:.100}.")
    if SpyAdamW_17.seen:
        lrs_17 = sorted({round(lr, 12) for lr, _, _ in SpyAdamW_17.seen[-1]})
        verdict("B1.17", lrs_17 == [1e-4, 1e-3], "AdamW reçoit 1e-3 pour la tête et 1e-4 pour layer4 (gamma = 0,1).",
                f"learning rates reçus par AdamW : {lrs_17}, attendu [1e-4, 1e-3] (param_groups_17 avec [model.fc, model.layer4]).")
    else:
        print("⚠️ Ex B1.17 : l'optimiseur n'a pas pu être observé (crée-le par torch.optim.AdamW dans train_partial_17).")
    sd_17 = model_17.state_dict()
    frozen_17 = [k for k, v in IMAGENET_RESNET18.items() if not k.startswith(("layer4.", "fc.")) and "running" not in k
                 and "num_batches" not in k and changed(sd_17[k], v)]
    stats_17 = [k for k, v in IMAGENET_RESNET18.items() if ("running" in k or "num_batches" in k) and changed(sd_17[k], v)]
    verdict("B1.17", not frozen_17, "les blocs gelés n'ont pas bougé.", f"{len(frozen_17)} tenseurs gelés ont changé (par ex. {frozen_17[:1]}).")
    verdict("B1.17", not stats_17, "les statistiques glissantes des batchnorms sont restées celles d'ImageNet.",
            f"{len(stats_17)} statistiques glissantes ont changé (par ex. {stats_17[:1]}) : dans quel mode étaient les "
            "batchnorms pendant l'entraînement ?")
    verdict("B1.17", changed(model_17.fc.weight, head_before_17) and any(changed(sd_17[k], IMAGENET_RESNET18[k]) for k in sd_17 if k.startswith("layer4.") and "running" not in k),
            "la tête et layer4 ont appris.", "la tête ou layer4 n'ont pas bougé : l'optimiseur les reçoit-il ?")'''

EXPERIMENT_17 = r'''with checked("B1.17"), torch.random.fork_rng():
    torch.manual_seed(17)
    MODEL_17 = make_partial_17()
    start_17 = time.perf_counter()
    losses_17 = train_partial_17(MODEL_17, XT_B, YT_B, FT_EPOCHS, head_lr=1e-3, gamma=0.1)
    seconds_17 = time.perf_counter() - start_17
    val_17, test_17 = evaluate(MODEL_17, X_VAL, Y_VAL), evaluate(MODEL_17, X_TEST, Y_TEST)
    probe_17 = reference_probe(reference_embeddings(X_TRAIN, SIZE, "train")[nested(FT_PER_CLASS)], YT_B,
                               reference_embeddings(X_TEST, SIZE, "test"), Y_TEST)
    print("loss d'entraînement par epoch :", ", ".join(fr(v, 3) for v in losses_17), f"({seconds_17:.0f} s)")
    print(f"layer4 + tête : validation {pct(val_17)}, test {pct(test_17)} ; tête seule (linear probe sur les mêmes "
          f"{len(XT_B)} images) : test {pct(probe_17)}")'''

TODO_18 = r'''# The fine-tuning to repair IN PLACE (same signature, same returns): the diagnostics of the next cell start from it
from torchvision.models import ResNet18_Weights, resnet18


def finetune_18(images, labels, steps=12, batch_size=4, seed=0):
    """A short fine-tuning of the whole ResNet-18 on CIFAR-10 images (uint8 (N, 32, 32, 3)), by mini-batches of
    batch_size images at SIZE pixels. Returns (model, losses): the network in eval() mode, the loss of each step."""
    torch.manual_seed(seed)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.train()
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed)).numpy()
    losses = []
    for step in range(steps):
        idx = order[(np.arange(batch_size) + step * batch_size) % len(images)]
        x = torch.from_numpy(images[idx]).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        loss = F.cross_entropy(model(x), torch.from_numpy(labels[idx]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model.eval(), losses'''

SOLUTION_18 = r'''from torchvision.models import ResNet18_Weights, resnet18

MEAN_18 = torch.tensor(wb.datasets.IMAGENET_MEAN).view(1, 3, 1, 1)
STD_18 = torch.tensor(wb.datasets.IMAGENET_STD).view(1, 3, 1, 1)


def finetune_18(images, labels, steps=12, batch_size=4, seed=0):
    """A short fine-tuning of the whole ResNet-18 on CIFAR-10 images (uint8 (N, 32, 32, 3)), by mini-batches of
    batch_size images at SIZE pixels. Returns (model, losses): the network in eval() mode, the loss of each step."""
    torch.manual_seed(seed)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)                 # 1. the new head first...
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    optimizer = torch.optim.AdamW([{"params": model.fc.parameters(), "lr": 1e-3},    # ...then the optimizer, which sees it
                                   {"params": backbone, "lr": 1e-5}])               # 4. small steps for the pre-trained layers
    model.train()
    freeze_batchnorm_17(model)                                     # 3. batchnorms in eval(): batches of 4 images
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed)).numpy()
    losses = []
    for step in range(steps):
        idx = order[(np.arange(batch_size) + step * batch_size) % len(images)]
        x = torch.from_numpy(images[idx]).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        x = (x - MEAN_18) / STD_18                                 # 2. the ImageNet normalisation
        loss = F.cross_entropy(model(x), torch.from_numpy(labels[idx]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model.eval(), losses'''

GIVEN_18 = r'''def diagnose_18(fn, steps=12):
    """Four diagnoses of a short fine-tuning, in a row: the first one that fails stops the chain (one error can hide
    another). Returns True when the four pass."""
    images, labels = X_TRAIN[nested(5)], Y_TRAIN[nested(5)]
    record = {"head": None, "inputs": []}

    def spy(module, inputs):                    # sees every module call: the first 3-channel convolution, the new head
        if isinstance(module, nn.Conv2d) and module.in_channels == 3 and module.out_channels == 64:
            record["inputs"].append((float(inputs[0].mean()), float(inputs[0].std())))
        if isinstance(module, nn.Linear) and module.out_features == 10 and record["head"] is None:
            record["head"] = module.weight.detach().clone()

    handle = torch.nn.modules.module.register_module_forward_pre_hook(spy)
    try:
        with torch.random.fork_rng():
            result = fn(images, labels, steps=steps)
    finally:
        handle.remove()
    if not (isinstance(result, tuple) and len(result) == 2):
        print(f"❌ Ex B1.18 : finetune_18 doit rendre le couple (model, losses), reçu {type(result).__name__}.")
        return False
    model, losses = result
    ref = IMAGENET_RESNET18
    head = getattr(model, "fc", None)
    moved_head = (float(np.nan_to_num((head.weight.detach() - record["head"]).abs().max().item(), nan=np.inf))
                  if record["head"] is not None and isinstance(head, nn.Linear)
                  and head.weight.shape == record["head"].shape else 0.0)                 # NaN weights have moved
    mean_in = float(np.mean([m for m, _ in record["inputs"]])) if record["inputs"] else float("nan")
    std_in = float(np.mean([s for _, s in record["inputs"]])) if record["inputs"] else float("nan")
    drift = float((model.bn1.running_mean.detach() - ref["bn1.running_mean"]).abs().max())
    w0 = ref["conv1.weight"]
    rel = float((model.conv1.weight.detach() - w0).norm() / w0.norm())
    last = float(np.mean(losses[-4:])) if len(losses) else float("nan")
    points = [  # (what passes, test, what was measured, a question that points where to look)
        ("la nouvelle tête apprend", moved_head > 1e-6,
         f"plus grand changement d'un poids de la tête à 10 classes en {steps} pas : {fr(moved_head, 6)}",
         "les poids de la tête n'ont pas bougé : quels paramètres l'optimiseur met-il à jour, et depuis quand existe la tête ?"),
        ("la première convolution reçoit des entrées à l'échelle de son pré-entraînement", std_in > 0.6,
         f"entrées de conv1 : moyenne {fr(mean_in, 2)}, écart-type {fr(std_in, 2)}",
         "ce sont des pixels entre 0 et 1 : est-ce ainsi qu'étaient préparées les images sur lesquelles ResNet-18 a appris ?"),
        ("les statistiques glissantes des batchnorms restent celles d'ImageNet", drift < 1e-6,
         f"plus grand écart de bn1.running_mean à sa valeur ImageNet : {fr(drift, 4)}",
         f"bn1.running_mean a dérivé en {steps} pas de 4 images : avec quelles statistiques les batchnorms normalisent-elles "
         "pendant l'entraînement, et que mettent-elles à jour ?"),
        ("les couches pré-entraînées bougent peu et la loss reste finie", rel < 0.02 and math.isfinite(last),
         f"changement relatif des poids de conv1 : {pct(rel, 2)} ; loss moyenne des 4 derniers pas : "
         + (fr(last, 3) if abs(last) < 1e4 else f"{last:.2e}".replace(".", ",")),
         "les premières couches changent beaucoup en quelques pas : quelle taille de pas reçoivent-elles ?"),
    ]
    for k, (name, ok, measured, why) in enumerate(points, start=1):
        if ok:
            print(f"✅ Ex B1.18 : point {k}, {name} ({measured}).")
        else:
            print(f"❌ Ex B1.18 : point {k} : {measured} ; {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex B1.18 : point {later} : corrige d'abord le point {k}.")
            return False
    print("✅ Ex B1.18 : les quatre diagnostics passent.")
    return True'''

CHECK_18 = r'''STEPS_18 = 60 if FAST_MODE else 300     # the steps of the training that follows the diagnoses
with checked("B1.18"):
    ok_18 = diagnose_18(finetune_18)'''

EXPERIMENT_18 = r'''if not globals().get("ok_18"):
    print("⏳ Ex B1.18 : l'entraînement attend que les quatre points de la cellule précédente passent.")
else:
    with checked("B1.18"), torch.random.fork_rng():
        model_18, losses_18 = finetune_18(XT_B, YT_B, steps=STEPS_18)
        acc_18 = evaluate(model_18, X_TEST[nested(100, "test")], Y_TEST[nested(100, "test")])
        print(f"{STEPS_18} pas de 4 images : loss {fr(np.mean(losses_18[:10]), 2)} sur les 10 premiers pas, "
              f"{fr(np.mean(losses_18[-10:]), 2)} sur les 10 derniers ; accuracy de test {pct(acc_18)} (le hasard : 10 %).")
        verdict("B1.18", acc_18 > 0.3, "le fine-tuning réparé apprend.",
                "l'accuracy reste proche du hasard : une erreur silencieuse reste-t-elle ?")'''

TODO_19 = r'''def warmup_cosine_19(step, t_warmup, t_total):
    """The factor of the learning rate at step `step` (0, 1, 2...): step / t_warmup during the warmup, then a cosine
    from 1 down to 0 at step t_total (0 at step 0, as in the fiche)."""
    raise NotImplementedError("warmup_cosine_19")


def make_full_19(num_classes=10):
    """ResNet-18 IMAGENET1K_V1 with a new head nn.Linear(512, num_classes), every parameter trainable. On DEVICE."""
    raise NotImplementedError("make_full_19")


def fine_tune_19(model, train, val, epochs, head_lr, gamma, t_warmup, patience, batch_size=32, seed=0):
    """Full fine-tuning with warmup + cosine and early stopping; leaves in `model` the weights of its best epoch.

    Returns {"train_loss": [one float per epoch], "val_acc": [one float per epoch],
             "lr": [[the learning rate of each group] for each step], "best_epoch": index of the best epoch}."""
    raise NotImplementedError("fine_tune_19")'''

SOLUTION_19 = r'''def warmup_cosine_19(step, t_warmup, t_total):
    """The factor of the learning rate at step `step` (0, 1, 2...): step / t_warmup during the warmup, then a cosine
    from 1 down to 0 at step t_total (0 at step 0, as in the fiche)."""
    if step < t_warmup:
        return step / t_warmup
    progress = (step - t_warmup) / max(1, t_total - t_warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_full_19(num_classes=10):
    """ResNet-18 IMAGENET1K_V1 with a new head nn.Linear(512, num_classes), every parameter trainable. On DEVICE."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def fine_tune_19(model, train, val, epochs, head_lr, gamma, t_warmup, patience, batch_size=32, seed=0):
    """Full fine-tuning with warmup + cosine and early stopping; leaves in `model` the weights of its best epoch.

    Returns {"train_loss": [one float per epoch], "val_acc": [one float per epoch],
             "lr": [[the learning rate of each group] for each step], "best_epoch": index of the best epoch}."""
    (X, y), (X_val, y_val) = train, val
    depth = [model.fc, model.layer4, model.layer3, model.layer2, nn.ModuleList([model.conv1, model.bn1, model.layer1])]
    optimizer = torch.optim.AdamW(param_groups_17(depth, head_lr, gamma))
    t_total = epochs * math.ceil(len(X) / batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: warmup_cosine_19(t, t_warmup, t_total))
    history = {"train_loss": [], "val_acc": [], "lr": [], "best_epoch": None}
    best_acc, best_state, waited = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        freeze_batchnorm_17(model)
        total, count = 0.0, 0
        for xb, yb in batches(X, y, batch_size, seed + epoch):
            history["lr"].append([g["lr"] for g in optimizer.param_groups])   # the rates used by this step
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()                                                  # one step of the planning per mini-batch
            total += loss.item() * len(yb)
            count += len(yb)
        history["train_loss"].append(total / count)
        acc = evaluate(model, X_val, y_val)
        history["val_acc"].append(acc)
        if acc > best_acc:                                                    # a better epoch: keep a COPY of its weights
            best_acc, best_state, waited = acc, copy.deepcopy(model.state_dict()), 0
            history["best_epoch"] = epoch
        else:
            waited += 1
            if waited >= patience:                                            # `patience` epochs without progress: stop
                break
    model.load_state_dict(best_state)                                         # the best checkpoint, not the last one
    model.eval()
    return history'''

CHECK_19 = r'''with checked("B1.19"):
    for args_19, want_19 in (((0, 4, 20), 0.0), ((2, 4, 20), 0.5), ((4, 4, 20), 1.0), ((12, 4, 20), 0.5),
                             ((20, 4, 20), 0.0), ((25, 4, 20), 0.0), ((3, 0, 10), 0.5 * (1 + math.cos(math.pi * 0.3)))):
        got_19 = warmup_cosine_19(*args_19)
        if not returned("B1.19", "warmup_cosine_19", got_19):
            raise StopCheck
        if not math.isclose(float(got_19), want_19, abs_tol=1e-9):
            print(f"❌ Ex B1.19 : warmup_cosine_19{args_19} = {got_19!r}, attendu {want_19:.4f}.")
            raise StopCheck
    print("✅ Ex B1.19 : warmup_cosine_19 suit le warmup puis le cosinus (0 au pas 0, 1 à la fin du warmup, 0 à t_total).")

with checked("B1.19"):
    model_19 = make_full_19()
    if not returned("B1.19", "make_full_19", model_19):
        raise StopCheck
    verdict("B1.19", isinstance(getattr(model_19, "fc", None), nn.Linear) and model_19.fc.out_features == 10
            and all(p.requires_grad for p in model_19.parameters())
            and all(torch.equal(model_19.state_dict()[k].cpu(), v) for k, v in IMAGENET_RESNET18.items() if not k.startswith("fc.")),
            "make_full_19 : les poids IMAGENET1K_V1, une tête à 10 sorties, tout est entraînable.",
            "make_full_19 : attendu les poids IMAGENET1K_V1, model.fc = nn.Linear(512, 10) et tous les paramètres entraînables.")

with checked("B1.19"), torch.random.fork_rng():
    # a scripted validation: evaluate() returns these accuracies in turn and remembers the weights it saw
    script_19, seen_19 = [0.30, 0.50, 0.40, 0.45, 0.60], []
    evaluate_19 = evaluate

    def scripted_evaluate_19(model, X, y, batch_size=128):
        seen_19.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return script_19[len(seen_19) - 1] if len(seen_19) <= len(script_19) else 0.0

    tiny_19 = (X_TRAIN[nested(2)], Y_TRAIN[nested(2)])
    evaluate = scripted_evaluate_19
    try:
        hist_19 = fine_tune_19(make_full_19(), tiny_19, (X_VAL[:10], Y_VAL[:10]), epochs=5, head_lr=1e-3, gamma=0.5,
                               t_warmup=3, patience=2, batch_size=10, seed=0)
    finally:
        evaluate = evaluate_19
    if not returned("B1.19", "fine_tune_19", hist_19):
        raise StopCheck
    if not isinstance(hist_19, dict) or not {"train_loss", "val_acc", "lr", "best_epoch"} <= set(hist_19):
        print('❌ Ex B1.19 : attendu un dictionnaire {"train_loss", "val_acc", "lr", "best_epoch"}, reçu '
              f"{type(hist_19).__name__} {sorted(hist_19) if isinstance(hist_19, dict) else ''}.")
        raise StopCheck
    verdict("B1.19", len(seen_19) == 4 and list(hist_19["val_acc"]) == script_19[:4],
            "early stopping : avec une patience de 2, l'entraînement s'arrête après la 4ᵉ epoch (validation 0,30, 0,50, 0,40, 0,45).",
            f"avec les accuracies de validation 0,30, 0,50, 0,40, 0,45, 0,60 et une patience de 2, attendu 4 epochs "
            f"(et evaluate appelée une fois par epoch) ; reçu {len(seen_19)} appel(s) d'evaluate, val_acc = {hist_19['val_acc']!r:.80}.")
    verdict("B1.19", hist_19["best_epoch"] == 1, "best_epoch = 1 (l'epoch à 0,50).",
            f"best_epoch = {hist_19['best_epoch']!r}, attendu l'indice (à partir de 0) de l'epoch à 0,50.")
    lrs_19 = hist_19["lr"]
    n_steps_19 = 2 * min(len(seen_19), 5)           # 20 images in mini-batches of 10: 2 steps per epoch
    t_total_19 = 5 * 2                               # the planned total: epochs x steps per epoch
    lr_ok_19 = isinstance(lrs_19, list) and len(lrs_19) == n_steps_19 and all(
        sorted(round(v, 15) for v in row) == sorted(round(1e-3 * 0.5 ** k * warmup_cosine_19(t, 3, t_total_19), 15)
                                                     for k in range(5) for _ in range(2))
        for t, row in enumerate(lrs_19))
    verdict("B1.19", lr_ok_19, "learning rates : 5 modules × 2 groupes, head_lr × gamma^k × warmup_cosine_19(t), pas par pas "
            "(t_total = epochs × pas par epoch).",
            f"history['lr'] : attendu {n_steps_19} lignes (une par pas), chacune avec les 10 learning rates des groupes au pas t, "
            "lus avant optimizer.step() (relis le contrat : modules, gamma, t_warmup, t_total).")

with checked("B1.19"), torch.random.fork_rng():
    # the best checkpoint: the weights left in the model must be those seen at the best epoch, not the last ones
    seen_19 = []
    model_19 = make_full_19()
    evaluate = scripted_evaluate_19
    try:
        fine_tune_19(model_19, tiny_19, (X_VAL[:10], Y_VAL[:10]), epochs=5, head_lr=1e-3, gamma=0.5, t_warmup=3,
                     patience=2, batch_size=10, seed=0)
    finally:
        evaluate = evaluate_19
    if len(seen_19) >= 4:
        final_19 = model_19.state_dict()
        best_19 = all(torch.equal(final_19[k], seen_19[1][k]) for k in final_19)
        last_19 = all(torch.equal(final_19[k], seen_19[-1][k]) for k in final_19)
        verdict("B1.19", best_19, "le modèle rendu a les poids de la meilleure epoch (une copie, rechargée à la fin).",
                "le modèle rendu a les poids de la dernière epoch : as-tu gardé une COPIE des poids de la meilleure, et l'as-tu "
                "rechargée ?" if last_19 else "le modèle rendu n'a les poids d'aucune epoch évaluée : que recharges-tu à la fin ?")'''

EXPERIMENT_19 = r'''with checked("B1.19"), torch.random.fork_rng():
    torch.manual_seed(19)
    MODEL_19 = make_full_19()
    steps_per_epoch_19 = math.ceil(len(XT_B) / 32)
    start_19 = time.perf_counter()
    history_19 = fine_tune_19(MODEL_19, (XT_B, YT_B), (X_VAL, Y_VAL), epochs=FT_EPOCHS, head_lr=1e-3, gamma=0.2,
                              t_warmup=steps_per_epoch_19, patience=1 if FAST_MODE else 2)
    seconds_19 = time.perf_counter() - start_19
    test_19 = evaluate(MODEL_19, X_TEST, Y_TEST)
    print("validation par epoch :", ", ".join(pct(a) for a in history_19["val_acc"]),
          f"; meilleure epoch : {history_19['best_epoch']} ; test du meilleur modèle : {pct(test_19)} ({seconds_19:.0f} s)")
    lr_19 = np.array(history_19["lr"])
    fig_19, axes_19 = plt.subplots(1, 2, figsize=(10, 3.4))
    steps_19 = np.arange(1, len(lr_19))                 # step 0 has a learning rate of 0: no place on a log scale
    for k_19 in range(lr_19.shape[1]):
        axes_19[0].plot(steps_19, lr_19[1:, k_19], lw=1)
    axes_19[0].set(yscale="log", xlabel="pas", ylabel="learning rate (échelle log)",
                   title="Un planning par groupe (deux groupes par module)")
    axes_19[1].plot(range(len(history_19["val_acc"])), history_19["val_acc"], "o-")
    axes_19[1].axvline(history_19["best_epoch"], color="gray", ls="--", lw=1)
    axes_19[1].set_xticks(range(len(history_19["val_acc"])))
    axes_19[1].set(xlabel="epoch", ylabel="accuracy de validation", title="Validation et meilleur checkpoint")
    plt.tight_layout()
    plt.show()'''
