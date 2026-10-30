"""Code cells of the chapter 29 notebook (parts A and B), used by build_ch29.py (used by Claude).

Each constant is the source of one cell: TOOLS (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
"""

TOOLS = r'''# Tools for the notebook exercises (parts A and B)
import copy
import hashlib
import re
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wb.nbtools import StopCheck, checked, filled, fr, returned, same, verdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # a computer without PyTorch (Mac Intel): parts of this chapter need Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : 29.8, 29.9 et 29.14 se font sur Colab (00_setup/INSTALL_LOCAL.md).")

WORK_DIR = Path(tempfile.mkdtemp(prefix="wb_ch29_"))   # a temporary folder for the files the exercises write


def pct(x, digits=1):
    """A share written the French way: 0.734 -> '73,4 %'."""
    return f"{100 * x:.{digits}f}".replace(".", ",") + " %"


def online(host="huggingface.co", timeout=5.0):
    """True if `host` answers on port 443 (the live parts of 29.10 and 29.11 are skipped without a network)."""
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def train_mlp(X, y, epochs=2, seed=0, batch_size=64, lr=1e-3):
    """The small network of 29.8, 29.9 and 29.14: 784-128-10 (ReLU), `epochs` epochs of Adam on uint8 images
    (N, 28, 28) divided by 255. Deterministic for a given seed."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32) / 255.0, torch.as_tensor(y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(Xt), generator=generator)
        for i in range(0, len(Xt), batch_size):
            idx = order[i:i + batch_size]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def evaluate(model, X, y):
    """(accuracy, mean confidence) of `model` on uint8 images X (N, 28, 28): the confidence of one image is its
    largest softmax probability."""
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32) / 255.0)
    probs = F.softmax(logits, dim=1)
    return float((logits.argmax(1).numpy() == np.asarray(y)).mean()), float(probs.max(1).values.mean())


def fine_tune(model, X, y, steps=100, lr=1e-3):
    """A copy of `model` trained `steps` more steps of Adam on all of (X, y) at once (a few labelled examples)."""
    model = copy.deepcopy(model).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32) / 255.0, torch.as_tensor(y, dtype=torch.long)
    for _ in range(steps):
        loss = F.cross_entropy(model(Xt), yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def digits_naive(images):
    """The 8 x 8 digits of scikit-learn (values 0 to 16) enlarged as they are to 28 x 28, uint8 0 to 255 (29.9)."""
    t = torch.as_tensor(np.asarray(images), dtype=torch.float32)[:, None] / 16.0
    out = F.interpolate(t, size=(28, 28), mode="bilinear", align_corners=False)[:, 0] * 255.0
    return np.clip(np.rint(out.numpy()), 0, 255).astype(np.uint8)


def show_digits(rows, titles, n=8):
    """One row of n images (28 x 28) per array of `rows`, its title on the left."""
    fig, axes = plt.subplots(len(rows), n, figsize=(1.1 * n, 1.25 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (images, title) in enumerate(zip(rows, titles)):
        for c in range(n):
            axes[r, c].imshow(images[c], cmap="gray", vmin=0, vmax=255)
            axes[r, c].axis("off")
        axes[r, 0].set_title(title, fontsize=8, loc="left")
    plt.tight_layout()
    plt.show()


# 29.11: stand-ins for the Hugging Face Hub and OpenML, so that the checks never depend on the network
FAKE_HUB = [SimpleNamespace(id=i, downloads=d, tags=t) for i, d, t in [
    ("demo/penguins-clean", 950, ["task_categories:tabular-classification", "license:cc0-1.0"]),
    ("demo/penguins-raw", 120, ["license:cc-by-4.0"]),
    ("someone/penguins-images", 3400, ["license:cc-by-nc-4.0", "size_categories:1K<n<10K"]),
    ("other/penguin-sounds", 40, []),
    ("lab/penguins-gps", 610, ["license:odbl"]),
    ("demo/fruits", 5000, ["license:mit"]),
]]


class FakeHubApi:
    """Answers list_datasets(search=..., limit=..., sort="downloads") like HfApi, from FAKE_HUB (no network)."""
    def __init__(self):
        self.calls = []

    def list_datasets(self, search=None, limit=None, sort=None, **kwargs):
        self.calls.append({"search": search, "limit": limit, "sort": sort, **kwargs})
        rows = [r for r in FAKE_HUB if search is None or search.lower() in r.id.lower()]
        if sort == "downloads":
            rows = sorted(rows, key=lambda r: -r.downloads)
        return iter(rows[:limit] if limit else rows)


def fake_fetch_openml(data_id=None, as_frame=True, **kwargs):
    """Answers like sklearn.datasets.fetch_openml(data_id=42585, as_frame=True), from the workbook copy of Penguins."""
    from sklearn.utils import Bunch
    frame = wb.datasets.load_penguins().drop(columns=["year"], errors="ignore")
    details = {"id": str(data_id), "name": "penguins", "version": "1", "licence": "CC0",
               "default_target_attribute": "species", "original_data_url": "https://github.com/allisonhorst/palmerpenguins",
               "md5_checksum": "0d4936a104bb85ef5127a5a63dce81f9"}
    return Bunch(frame=frame, details=details, target_names=["species"])


def watch_leaks(fn, X, y, groups):
    """Run fn(X, y, groups) with spies on PCA and KNeighborsClassifier (looked up in the notebook's globals, as the
    learner's function does) and record which rows of X each fit and each score saw (29.13). Returns
    (result, events): events is a list of (kind, row indices or None), kind in "pca_fit", "knn_fit", "knn_score"."""
    g = globals()
    flat = np.asarray(X).reshape(len(X), -1).astype(np.uint8)
    index = {row.tobytes(): i for i, row in enumerate(flat)}

    def rows_of(A):
        A = np.asarray(A, dtype=float)
        if A.ndim < 2 or A.size // len(A) != flat.shape[1]:
            return None                                          # not raw pixels (already transformed)
        A = A.reshape(len(A), -1)
        scale = 255.0 if A.max() <= 1.0 + 1e-9 else 1.0
        found = [index.get(r.tobytes()) for r in np.clip(np.rint(A * scale), 0, 255).astype(np.uint8)]
        return None if any(f is None for f in found) else np.array(found)

    events, made = [], {}
    real_pca, real_knn = g["PCA"], g["KNeighborsClassifier"]

    def seen(A):
        return made[id(A)][1] if id(A) in made else rows_of(A)

    class SpyPCA(real_pca):
        def fit(self, A, y=None):
            events.append(("pca_fit", rows_of(A)))
            return super().fit(A, y)

        def fit_transform(self, A, y=None):
            events.append(("pca_fit", rows_of(A)))
            out = super().fit_transform(A, y)
            made[id(out)] = (out, rows_of(A))
            return out

        def transform(self, A):
            out = super().transform(A)
            made[id(out)] = (out, rows_of(A))
            return out

    class SpyKNN(real_knn):
        def fit(self, A, y):
            events.append(("knn_fit", seen(A)))
            return super().fit(A, y)

        def score(self, A, y, sample_weight=None):
            events.append(("knn_score", seen(A)))
            return super().score(A, y, sample_weight=sample_weight)

    g["PCA"], g["KNeighborsClassifier"] = SpyPCA, SpyKNN
    try:
        result = fn(X, y, groups)
    finally:
        g["PCA"], g["KNeighborsClassifier"] = real_pca, real_knn
    return result, events


# 29.15: the sections a data card must have (the sheet, « Au-delà du livre : documenter et télécharger »)
CARD_SECTIONS = ["Provenance", "Contenu", "Licence", "Usage prévu", "Limites et biais", "Téléchargement"]


def check_card(text):
    """The list of problems of a data card written in Markdown (empty when it is complete)."""
    problems = [f"section « ## {s} » absente" for s in CARD_SECTIONS
                if not re.search(rf"^##\s+{re.escape(s)}\s*$", text, flags=re.M)]
    if "..." in text:
        problems.append("il reste des « ... » à remplacer")
    if not re.search(r"\b[0-9a-f]{64}\b", text):
        problems.append("aucune somme de contrôle SHA-256 (64 caractères hexadécimaux) dans la carte")
    if not re.search(r"https?://", text):
        problems.append("aucune adresse (URL) de la source")
    return problems'''

# ---------------------------------------------------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------------------------------------------------
GIVEN_7 = r'''from sklearn import datasets as skd

# The loaders to inventory (name -> function of sklearn.datasets); "diabetes" is a regression
SKLEARN_LOADERS_7 = {"iris": "load_iris", "digits": "load_digits", "wine": "load_wine",
                     "breast_cancer": "load_breast_cancer", "diabetes": "load_diabetes"}
REGRESSION_7 = {"diabetes"}'''

TODO_7 = r'''def inventory_7():
    """A DataFrame, one row per dataset of SKLEARN_LOADERS_7 (in this order), with the columns "dataset" (its name),
    "exemples", "variables", "tache" ("classification" or "régression", from REGRESSION_7) and "classes" (the number
    of distinct labels, 0 for a regression), measured on the loaded data (getattr(skd, loader)(return_X_y=True))."""
    raise NotImplementedError("inventory_7")


def has_loader_7(name):
    """True if sklearn.datasets still provides the function `name` (e.g. "load_iris"), False otherwise. Careful: for
    a loader that was removed, scikit-learn raises an ImportError (with an explanation), not an AttributeError."""
    raise NotImplementedError("has_loader_7")


minority_7 = ...   # b) the share of the minority class of breast_cancer, 3 decimals'''

SOLUTION_7 = r'''def inventory_7():
    """A DataFrame, one row per dataset of SKLEARN_LOADERS_7 (in this order), with the columns "dataset" (its name),
    "exemples", "variables", "tache" ("classification" or "régression", from REGRESSION_7) and "classes" (the number
    of distinct labels, 0 for a regression), measured on the loaded data (getattr(skd, loader)(return_X_y=True))."""
    rows = []
    for name, loader in SKLEARN_LOADERS_7.items():
        X, y = getattr(skd, loader)(return_X_y=True)
        regression = name in REGRESSION_7
        rows.append({"dataset": name, "exemples": X.shape[0], "variables": X.shape[1],
                     "tache": "régression" if regression else "classification",
                     "classes": 0 if regression else len(np.unique(y))})
    return pd.DataFrame(rows)


def has_loader_7(name):
    """True if sklearn.datasets still provides the function `name` (e.g. "load_iris"), False otherwise. Careful: for
    a loader that was removed, scikit-learn raises an ImportError (with an explanation), not an AttributeError."""
    try:
        return callable(getattr(skd, name))
    except (AttributeError, ImportError):
        return False


_, y_bc_7 = skd.load_breast_cancer(return_X_y=True)
minority_7 = round(float(np.bincount(y_bc_7).min() / len(y_bc_7)), 3)'''

CHECK_7 = r'''with checked("29.7"):
    table_7 = inventory_7()
    if not returned("29.7", "inventory_7", table_7):
        raise StopCheck
    columns_7 = ["dataset", "exemples", "variables", "tache", "classes"]
    verdict("29.7", isinstance(table_7, pd.DataFrame) and list(table_7.columns) == columns_7,
            "un DataFrame aux cinq colonnes demandées.",
            f"attendu un DataFrame aux colonnes {columns_7}, reçu {list(getattr(table_7, 'columns', []))}.")
    want_7 = []
    for name_7, loader_7 in SKLEARN_LOADERS_7.items():
        X_w7, y_w7 = getattr(skd, loader_7)(return_X_y=True)
        reg_7 = name_7 in REGRESSION_7
        want_7.append([name_7, X_w7.shape[0], X_w7.shape[1], "régression" if reg_7 else "classification",
                       0 if reg_7 else len(np.unique(y_w7))])
    got_7 = table_7[columns_7].values.tolist() if list(table_7.columns) == columns_7 else []
    bad_7 = [w[0] for w, g in zip(want_7, got_7) if [str(v) for v in w] != [str(v) for v in g]]
    verdict("29.7", len(got_7) == len(want_7) and not bad_7, "les tailles, les tâches et les classes sont justes.",
            f"lignes fausses ou manquantes : {bad_7 or 'nombre de lignes'} (mesure sur X.shape et np.unique(y)).")
    verdict("29.7", has_loader_7("load_iris") is True and has_loader_7("fetch_openml") is True,
            "has_loader_7 reconnaît les fonctions présentes.", "has_loader_7('load_iris') devrait rendre True.")
    verdict("29.7", has_loader_7("load_nothing_7") is False, "has_loader_7 rend False pour un nom inconnu.",
            "has_loader_7('load_nothing_7') devrait rendre False, sans lever d'erreur.")
    wb.check("29.7a", int(table_7.set_index("dataset").loc["digits", "exemples"]), computed=True)
    try:
        boston_7 = has_loader_7("load_boston")
    except ImportError:
        print("❌ Ex 29.7 : has_loader_7('load_boston') laisse passer l'ImportError de scikit-learn : hasattr, ou un "
              "except AttributeError seul, ne suffit pas ; attrape aussi ImportError.")
        raise StopCheck
    wb.check("29.7c", boston_7, computed=True)
wb.check("29.7b", minority_7)'''

RECORD_7 = r'''wb.record("29.7a", 1797, mistakes={"c'est le nombre du livre : mesure X.shape[0] sur les données chargées": 5620})
wb.record("29.7b", 0.373, decimals=3, mistakes={"c'est la part de la classe majoritaire : prends la plus petite": 0.627,
                                                "une part entre 0 et 1, pas un effectif": 212.0})
wb.record("29.7c", False, mistakes={"load_boston a été retiré en 1.2 : un ImportError doit donner False": True})'''

GIVEN_8 = r'''# The same small network (784-128-10, two epochs of Adam on 10 000 training images, train_mlp in the tools) trained
# twice: on MNIST, then on Fashion-MNIST. Nothing is printed here: the accuracies come after your predictions.
XM_TRAIN, YM_TRAIN = wb.datasets.load_mnist("train", 10_000)
XM_TEST, YM_TEST = wb.datasets.load_mnist("test")
XF_TRAIN, YF_TRAIN = wb.datasets.load_fashion_mnist("train", 10_000)
XF_TEST, YF_TEST = wb.datasets.load_fashion_mnist("test")
MODEL_M = train_mlp(XM_TRAIN, YM_TRAIN) if torch is not None else None
MODEL_F = train_mlp(XF_TRAIN, YF_TRAIN) if torch is not None else None'''

TODO_8 = r'''mnist_8 = ...     # a) the MNIST test accuracy of MODEL_M: "A" below 0.90, "B" from 0.90 to 0.97, "C" above 0.97
drop_8 = ...      # b) the points lost by MODEL_F on Fashion-MNIST: "A" fewer than 5, "B" from 5 to 15, "C" more than 15
crossed_8 = ...   # c) MODEL_M (trained on digits) on the Fashion-MNIST test: "A" below 0.2, "B" 0.2 to 0.5, "C" above 0.5'''

SOLUTION_8 = r'''mnist_8, drop_8, crossed_8 = "B", "B", "A"'''

EXPERIMENT_8 = r'''acc_m8, conf_m8 = evaluate(MODEL_M, XM_TEST, YM_TEST)
acc_f8, conf_f8 = evaluate(MODEL_F, XF_TEST, YF_TEST)
acc_x8, conf_x8 = evaluate(MODEL_M, XF_TEST, YF_TEST)
print(f"MNIST : accuracy {fr(acc_m8)} ; Fashion-MNIST : accuracy {fr(acc_f8)}, soit {fr(100 * (acc_m8 - acc_f8), 1)} points de moins")
print(f"Le modèle MNIST sur Fashion-MNIST : accuracy {fr(acc_x8)} (le hasard : 0,1), confiance moyenne {fr(conf_x8)}")
with torch.no_grad():
    pred_f8 = MODEL_F(torch.as_tensor(XF_TEST, dtype=torch.float32) / 255.0).argmax(1).numpy()
recall_f8 = [float((pred_f8[YF_TEST == c] == c).mean()) for c in range(10)]
plt.figure(figsize=(8, 3))
plt.bar(wb.datasets.FASHION_CLASSES, recall_f8, color="tab:orange")
plt.xticks(rotation=45, ha="right")
plt.ylabel("recall")
plt.title("Fashion-MNIST : recall de chaque classe")
plt.tight_layout()
plt.show()
wb.check("29.8a", mnist_8)
wb.check("29.8b", drop_8)
wb.check("29.8c", crossed_8)'''

RECORD_8 = r'''wb.record("29.8a", "B", mistakes={"deux epochs sur 10 000 images, un petit réseau dense : on n'atteint pas les 98 % habituels": "C",
                                   "même un petit réseau dense dépasse 90 % sur MNIST": "A"})
wb.record("29.8b", "B", mistakes={"regarde les deux accuracies : l'écart dépasse 5 points": "A",
                                  "regarde les deux accuracies : l'écart reste sous 15 points": "C"})
wb.record("29.8c", "A", mistakes={"un modèle de chiffres ne sait rien des vêtements : il fait à peu près le hasard": "B",
                                  "un modèle de chiffres ne sait rien des vêtements : son accuracy est celle du hasard": "C"})'''

GIVEN_9 = r'''from sklearn.datasets import load_digits

DIGITS_9 = load_digits()                       # 1 797 digits of 8 x 8 pixels, values 0 to 16 (an UCI collection)
XD_9 = digits_naive(DIGITS_9.images)           # enlarged as they are to 28 x 28, 0 to 255: the MNIST format?
YD_9 = DIGITS_9.target'''

TODO_9 = r'''digits_9 = ...      # a) the accuracy of MODEL_M on XD_9: "A" above 0.7, "B" from 0.4 to 0.7, "C" below 0.4
confident_9 = ...   # b) True if the mean confidence of MODEL_M on XD_9 stays above 0.6, else False'''

SOLUTION_9 = r'''digits_9, confident_9 = "C", True'''

EXPERIMENT_9 = r'''show_digits([XM_TEST[:8], XD_9[:8]], ["MNIST (test)", "scikit-learn, agrandis tels quels"])
acc_m9, conf_m9 = evaluate(MODEL_M, XM_TEST, YM_TEST)
acc_d9, conf_d9 = evaluate(MODEL_M, XD_9, YD_9)
print(f"MNIST (test)          : accuracy {fr(acc_m9)}, confiance moyenne {fr(conf_m9)}")
print(f"chiffres scikit-learn : accuracy {fr(acc_d9)}, confiance moyenne {fr(conf_d9)}")
print(f"pixels moyens : {fr(float(XM_TEST.mean()), 1)} sur MNIST, {fr(float(XD_9.mean()), 1)} ici (0 à 255)")
wb.check("29.9a", digits_9)
wb.check("29.9b", confident_9)'''

RECORD_9 = r'''wb.record("29.9a", "C", mistakes={"ce sont bien des chiffres, mais ni cadrés ni dessinés comme ceux de MNIST : l'accuracy chute beaucoup plus": "A",
                                  "regarde l'accuracy mesurée : elle tombe sous 0,4": "B"})
wb.record("29.9b", True, mistakes={"un modèle ne sait pas qu'il sort de sa distribution : sa confiance reste élevée quand il se trompe": False})'''

GIVEN_10 = r'''# One version of Fashion-MNIST on the Hugging Face Hub, pinned to an exact revision (a commit of the dataset repository)
FASHION_REPO_10 = "zalando-datasets/fashion_mnist"
FASHION_REV_10 = "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2"
import warnings
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # tqdm's notice about ipywidgets, printed once at import
        from datasets import Dataset, load_dataset
    try:
        with wb.datasets.quiet_huggingface():
            HF_10 = load_dataset(FASHION_REPO_10, split="test", revision=FASHION_REV_10,
                                 cache_dir=str(wb.datasets.cache_dir() / "hf"))
        SOURCE_10 = "le Hugging Face Hub"
    except Exception as exc:              # no network: the same format, built from the workbook copy
        from PIL import Image
        X_off10, y_off10 = wb.datasets.load_fashion_mnist("test")
        HF_10 = Dataset.from_dict({"image": [Image.fromarray(img) for img in X_off10], "label": y_off10.tolist()})
        SOURCE_10 = f"une copie locale au même format (Hub injoignable : {type(exc).__name__})"
except ImportError:
    HF_10, SOURCE_10 = None, "rien : la bibliothèque datasets est absente (pip install datasets)"
print(f"Fashion-MNIST (test) chargé depuis {SOURCE_10} : {HF_10}")'''

TODO_10 = r'''def hf_to_numpy_10(split):
    """(X, y) from a split of Hugging Face `datasets` with the columns "image" (PIL images) and "label": X uint8
    (N, 28, 28) and y int64 (N,), in the order of the split."""
    raise NotImplementedError("hf_to_numpy_10")


def compare_10(Xa, ya, Xb, yb):
    """Compare two versions of the same images: a dict with "same_shape" (bool), "same_labels" (bool: as many labels,
    equal in the same order), "same_pixels" (bool, False when the shapes differ) and "max_diff" (int: the largest
    absolute difference between two pixels, 0 when identical, -1 when the shapes differ)."""
    raise NotImplementedError("compare_10")'''

SOLUTION_10 = r'''def hf_to_numpy_10(split):
    """(X, y) from a split of Hugging Face `datasets` with the columns "image" (PIL images) and "label": X uint8
    (N, 28, 28) and y int64 (N,), in the order of the split."""
    X = np.stack([np.asarray(img, dtype=np.uint8) for img in split["image"]])
    y = np.asarray(split["label"], dtype=np.int64)
    return X, y


def compare_10(Xa, ya, Xb, yb):
    """Compare two versions of the same images: a dict with "same_shape" (bool), "same_labels" (bool: as many labels,
    equal in the same order), "same_pixels" (bool, False when the shapes differ) and "max_diff" (int: the largest
    absolute difference between two pixels, 0 when identical, -1 when the shapes differ)."""
    Xa, Xb, ya, yb = np.asarray(Xa), np.asarray(Xb), np.asarray(ya), np.asarray(yb)
    same_shape = Xa.shape == Xb.shape
    same_labels = ya.shape == yb.shape and bool((ya == yb).all())
    max_diff = int(np.abs(Xa.astype(np.int64) - Xb.astype(np.int64)).max()) if same_shape else -1
    return {"same_shape": same_shape, "same_labels": same_labels, "same_pixels": max_diff == 0, "max_diff": max_diff}'''

CHECK_10 = r'''with checked("29.10"):
    if HF_10 is None:
        print("⏳ Ex 29.10 : la bibliothèque datasets est absente : pip install datasets (déjà là sur Colab).")
        raise StopCheck
    X_hf10, y_hf10 = hf_to_numpy_10(HF_10)
    if not returned("29.10", "hf_to_numpy_10", X_hf10):
        raise StopCheck
    verdict("29.10", isinstance(X_hf10, np.ndarray) and X_hf10.shape == (len(HF_10), 28, 28) and X_hf10.dtype == np.uint8,
            "X : uint8 de forme (N, 28, 28).", f"X attendu uint8 (N, 28, 28), reçu {getattr(X_hf10, 'dtype', type(X_hf10))} {getattr(X_hf10, 'shape', '')}.")
    verdict("29.10", np.asarray(y_hf10).dtype == np.int64 and np.asarray(y_hf10).shape == (len(HF_10),),
            "y : int64 de forme (N,).", f"y attendu int64 (N,), reçu {np.asarray(y_hf10).dtype} {np.asarray(y_hf10).shape}.")
    if X_hf10.dtype == np.uint8:
        verdict("29.10", int(X_hf10[0].sum()) == int(np.asarray(HF_10[0]["image"], dtype=np.int64).sum()) and int(y_hf10[0]) == HF_10[0]["label"],
                "le premier exemple est bien le premier du split.", "le premier exemple ne correspond pas à HF_10[0] : garde l'ordre du split.")
    X_wb10, y_wb10 = wb.datasets.load_fashion_mnist("test")
    rng_10 = np.random.default_rng(10)
    X_a10 = X_wb10[:50].copy()
    r_10, c_10 = np.unravel_index(int(X_a10[3].argmax()), (28, 28))
    X_a10[3, r_10, c_10] -= 7                                           # its brightest pixel, darker by 7
    probe_10 = compare_10(X_a10, y_wb10[:50], X_wb10[:50], y_wb10[:50])   # the darker images first: 3 - 10 in uint8
    if not returned("29.10", "compare_10", probe_10):
        raise StopCheck
    keys_10 = {"same_shape", "same_labels", "same_pixels", "max_diff"}
    verdict("29.10", isinstance(probe_10, dict) and set(probe_10) == keys_10, "compare_10 rend un dict aux quatre clés.",
            f"attendu un dict aux clés {sorted(keys_10)}, reçu {probe_10!r:.120}.")
    verdict("29.10", probe_10.get("max_diff") == 7 and not bool(probe_10.get("same_pixels")),
            "un pixel plus sombre de 7 est vu (max_diff = 7, same_pixels False).",
            f"un pixel plus sombre de 7 : attendu max_diff = 7 et same_pixels False, reçu {probe_10.get('max_diff')} et "
            f"{probe_10.get('same_pixels')} (en uint8, 3 − 10 vaut 249 : convertis en entiers assez grands, int64, avant de soustraire).")
    shuffled_10 = compare_10(X_wb10[:50], y_wb10[:50], X_wb10[:50], rng_10.permutation(y_wb10[:50]))
    verdict("29.10", not bool(shuffled_10.get("same_labels")) and bool(shuffled_10.get("same_pixels")),
            "des labels mélangés sont vus.", "des labels mélangés (images identiques) : attendu same_labels False, same_pixels True.")
    other_10 = compare_10(X_wb10[:50], y_wb10[:50], X_wb10[:40], y_wb10[:40])
    verdict("29.10", not bool(other_10.get("same_shape")) and not bool(other_10.get("same_pixels")) and other_10.get("max_diff") == -1,
            "des formes différentes donnent max_diff = -1, sans erreur.",
            f"formes différentes : attendu same_shape et same_pixels False, max_diff -1, reçu {other_10}.")
    result_10 = compare_10(X_hf10, y_hf10, X_wb10, y_wb10)
    print(f"Hub (révision {FASHION_REV_10[:8]}) contre wb.datasets.load_fashion_mnist('test') : {result_10}")
    print("Noms des classes sur le Hub :", HF_10.features["label"].names if hasattr(HF_10.features["label"], "names") else "(pas de noms)")
    print("Noms des classes du workbook :", wb.datasets.FASHION_CLASSES)'''

TODO_11 = r'''def search_hub_11(query, limit=5, api=None):
    """A DataFrame of the `limit` most downloaded datasets of the Hugging Face Hub found by `query`, the most downloaded
    first, with the columns "id", "downloads" and "licence" (xxx of a tag "license:xxx", as it is; "inconnue" without one).
    `api` is an HfApi (a new one when None): use api.list_datasets(search=..., limit=..., sort="downloads")."""
    raise NotImplementedError("search_hub_11")


def openml_summary_11(data_id, fetch=None):
    """A dict about a dataset of OpenML, from fetch(data_id=data_id, as_frame=True) (fetch: by default
    sklearn.datasets.fetch_openml): "name", "version", "licence", "rows" and "columns" (of bunch.frame, target
    included), "target" (details["default_target_attribute"]) and "source" (details["original_data_url"], None when
    absent)."""
    raise NotImplementedError("openml_summary_11")'''

SOLUTION_11 = r'''def search_hub_11(query, limit=5, api=None):
    """A DataFrame of the `limit` most downloaded datasets of the Hugging Face Hub found by `query`, the most downloaded
    first, with the columns "id", "downloads" and "licence" (xxx of a tag "license:xxx", as it is; "inconnue" without one).
    `api` is an HfApi (a new one when None): use api.list_datasets(search=..., limit=..., sort="downloads")."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    rows = []
    for info in api.list_datasets(search=query, limit=limit, sort="downloads"):
        licences = [t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")]
        rows.append({"id": info.id, "downloads": info.downloads, "licence": licences[0] if licences else "inconnue"})
    return pd.DataFrame(rows, columns=["id", "downloads", "licence"])


def openml_summary_11(data_id, fetch=None):
    """A dict about a dataset of OpenML, from fetch(data_id=data_id, as_frame=True) (fetch: by default
    sklearn.datasets.fetch_openml): "name", "version", "licence", "rows" and "columns" (of bunch.frame, target
    included), "target" (details["default_target_attribute"]) and "source" (details["original_data_url"], None when
    absent)."""
    if fetch is None:
        from sklearn.datasets import fetch_openml as fetch
    bunch = fetch(data_id=data_id, as_frame=True)
    details = bunch.details
    rows, columns = bunch.frame.shape
    return {"name": details["name"], "version": details["version"], "licence": details.get("licence"),
            "rows": rows, "columns": columns, "target": details.get("default_target_attribute"),
            "source": details.get("original_data_url")}'''

CHECK_11 = r'''with checked("29.11"):
    api_11 = FakeHubApi()
    found_11 = search_hub_11("penguin", limit=3, api=api_11)
    if not returned("29.11", "search_hub_11", found_11):
        raise StopCheck
    verdict("29.11", isinstance(found_11, pd.DataFrame) and list(found_11.columns) == ["id", "downloads", "licence"],
            "search_hub_11 rend un DataFrame aux colonnes id, downloads, licence.",
            f"attendu les colonnes ['id', 'downloads', 'licence'], reçu {list(getattr(found_11, 'columns', []))}.")
    want_11 = [("someone/penguins-images", 3400, "cc-by-nc-4.0"), ("demo/penguins-clean", 950, "cc0-1.0"),
               ("lab/penguins-gps", 610, "odbl")]
    got_11 = [tuple(r) for r in found_11[["id", "downloads", "licence"]].itertuples(index=False)] if len(found_11.columns) == 3 else []
    verdict("29.11", got_11 == want_11, "les 3 plus téléchargés, dans l'ordre, avec leur licence (sur un Hub simulé).",
            f"sur le Hub simulé, attendu {want_11}, reçu {got_11} (demande sort=\"downloads\" et limit à list_datasets).")
    none_11 = search_hub_11("sounds", limit=5, api=FakeHubApi())
    verdict("29.11", list(none_11["licence"]) == ["inconnue"], "un dataset sans tag de licence : « inconnue ».",
            f"pour un dataset sans tag license:, attendu « inconnue », reçu {list(none_11['licence'])}.")
    summary_11 = openml_summary_11(42585, fetch=fake_fetch_openml)
    if not returned("29.11", "openml_summary_11", summary_11):
        raise StopCheck
    frame_11 = fake_fetch_openml(42585).frame
    want_s11 = {"name": "penguins", "version": "1", "licence": "CC0", "rows": frame_11.shape[0],
                "columns": frame_11.shape[1], "target": "species", "source": "https://github.com/allisonhorst/palmerpenguins"}
    verdict("29.11", summary_11 == want_s11, "openml_summary_11 lit le nom, la version, la licence, la taille, la cible et la source.",
            f"sur un OpenML simulé, attendu {want_s11}, reçu {summary_11}.")'''

EXPERIMENT_11 = r'''# Live, if the network answers (a few seconds; OpenML caches its files in the workbook cache)
if online("huggingface.co"):
    with checked("29.11"):
        print(search_hub_11("penguins", limit=5).to_string(index=False))
else:
    print("⏳ Hub injoignable : la recherche en direct est sautée (les vérifications ci-dessus suffisent).")
if online("www.openml.org"):
    with checked("29.11"):
        from sklearn.datasets import fetch_openml
        def fetch_cached_11(**kwargs):
            return fetch_openml(data_home=str(wb.datasets.cache_dir() / "openml"), **kwargs)
        print(openml_summary_11(42585, fetch=fetch_cached_11))
else:
    print("⏳ OpenML injoignable : le résumé en direct est sauté.")'''

# ---------------------------------------------------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------------------------------------------------
GIVEN_12 = r'''ADULT_12 = wb.datasets.load_adult()     # the 1994 US census (48 842 people), data/cards/adult.md
ADULT_12.head(3)'''

TODO_12 = r'''def profile_12(df):
    """One row per column of df (same order, index = the column names), with the columns "type" (str of the dtype),
    "manquants" (the number of missing values), "part_manquants" (their share, 0 to 1), "distinctes" (the number of
    distinct values, missing ones excluded), "mode" (the most frequent value) and "part_mode" (its share among the
    values that are not missing)."""
    raise NotImplementedError("profile_12")


most_missing_12 = ...   # a) the column of ADULT_12 with the most missing values (its name, between quotes)
duplicates_12 = ...     # b) the number of rows of ADULT_12 that repeat an earlier row
majority_12 = ...       # c) the share of the majority class of the column "class", 2 decimals'''

SOLUTION_12 = r'''def profile_12(df):
    """One row per column of df (same order, index = the column names), with the columns "type" (str of the dtype),
    "manquants" (the number of missing values), "part_manquants" (their share, 0 to 1), "distinctes" (the number of
    distinct values, missing ones excluded), "mode" (the most frequent value) and "part_mode" (its share among the
    values that are not missing)."""
    rows = {}
    for col in df.columns:
        s = df[col]
        counts = s.value_counts(dropna=True)
        rows[col] = {"type": str(s.dtype), "manquants": int(s.isna().sum()), "part_manquants": float(s.isna().mean()),
                     "distinctes": int(s.nunique(dropna=True)), "mode": counts.index[0],
                     "part_mode": float(counts.iloc[0] / counts.sum())}
    return pd.DataFrame.from_dict(rows, orient="index")


prof_12 = profile_12(ADULT_12)
most_missing_12 = prof_12["manquants"].idxmax()
duplicates_12 = int(ADULT_12.duplicated().sum())
majority_12 = round(float(ADULT_12["class"].value_counts(normalize=True).iloc[0]), 2)'''

CHECK_12 = r'''with checked("29.12"):
    prof_c12 = profile_12(ADULT_12)
    if not returned("29.12", "profile_12", prof_c12):
        raise StopCheck
    cols_12 = ["type", "manquants", "part_manquants", "distinctes", "mode", "part_mode"]
    verdict("29.12", isinstance(prof_c12, pd.DataFrame) and list(prof_c12.columns) == cols_12
            and list(prof_c12.index) == list(ADULT_12.columns),
            "une ligne par colonne, les six colonnes demandées.",
            f"attendu l'index {list(ADULT_12.columns)[:3]}… et les colonnes {cols_12}, reçu {list(getattr(prof_c12, 'columns', []))}.")
    want_12 = pd.DataFrame({"type": ADULT_12.dtypes.astype(str), "manquants": ADULT_12.isna().sum(),
                            "part_manquants": ADULT_12.isna().mean(), "distinctes": ADULT_12.nunique(),
                            "part_mode": [float(ADULT_12[c].value_counts(normalize=True).iloc[0]) for c in ADULT_12.columns]})
    for field_12 in ["type", "manquants", "part_manquants", "distinctes", "part_mode"]:
        if field_12 == "type":
            bad_12 = [c for c in ADULT_12.columns if str(prof_c12.loc[c, field_12]) != want_12.loc[c, field_12]]
        else:
            bad_12 = [c for c in ADULT_12.columns
                      if not np.isclose(float(prof_c12.loc[c, field_12]), float(want_12.loc[c, field_12]), rtol=1e-9, atol=1e-12)]
        verdict("29.12", not bad_12, f"« {field_12} » est juste pour les {len(ADULT_12.columns)} colonnes.",
                f"« {field_12} » faux pour {', '.join(bad_12[:4])}{'…' if len(bad_12) > 4 else ''} "
                f"(par exemple {bad_12[0] if bad_12 else ''} : attendu {want_12.loc[bad_12[0], field_12] if bad_12 else ''}, "
                f"reçu {prof_c12.loc[bad_12[0], field_12] if bad_12 else ''}).")
    verdict("29.12", str(prof_c12.loc["sex", "mode"]) == str(ADULT_12["sex"].mode().iloc[0]), "le mode est la valeur la plus fréquente.",
            f"mode de « sex » : attendu {ADULT_12['sex'].mode().iloc[0]!r}, reçu {prof_c12.loc['sex', 'mode']!r}.")
    fig_12, axes_12 = plt.subplots(1, 2, figsize=(11, 3.5))
    prof_c12["part_manquants"].plot.bar(ax=axes_12[0], color="tab:red", title="part de valeurs manquantes")
    prof_c12["part_mode"].plot.bar(ax=axes_12[1], color="tab:blue", title="part de la valeur la plus fréquente")
    for ax_12 in axes_12:
        ax_12.set_ylim(0, 1.05 if ax_12 is axes_12[1] else None)
    plt.tight_layout()
    plt.show()
wb.check("29.12a", most_missing_12)
wb.check("29.12b", duplicates_12)
wb.check("29.12c", majority_12)'''

RECORD_12 = r'''wb.record("29.12a", "occupation", mistakes={"workclass est juste derrière : compare les deux nombres de manquants": "workclass",
                                            "native-country en a beaucoup moins : regarde la colonne manquants": "native-country"})
wb.record("29.12b", 52, mistakes={"duplicated() marque chaque répétition, pas les deux lignes d'une paire": 104,
                                  "ADULT_12.duplicated() compte les lignes répétées : il y en a": 0})
wb.record("29.12c", 0.76, decimals=2, mistakes={"c'est la classe minoritaire (>50K)": 0.24})'''

GIVEN_13 = r'''from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier


def make_near_duplicates_13(n_sources=500, copies=4, seed=13):
    """Photos taken several times: each of n_sources Fashion-MNIST images appears `copies` times, each time with a
    slightly different brightness and a small noise. X uint8 (N, 28, 28), y (N,), groups (N,): the index of the
    source image of each row (the rows of a group are near-duplicates)."""
    Xs, ys = wb.datasets.load_fashion_mnist("train", n_sources, seed=seed)
    rng = np.random.default_rng(seed)
    n = n_sources * copies
    X = np.repeat(Xs.astype(float), copies, axis=0) * rng.uniform(0.9, 1.1, (n, 1, 1)) + rng.normal(0, 6, (n, 28, 28))
    return np.clip(np.rint(X), 0, 255).astype(np.uint8), np.repeat(ys, copies), np.repeat(np.arange(n_sources), copies)


X_13, Y_13, GROUPS_13 = make_near_duplicates_13()


# The evaluation to debug (do not change this cell: copy the function into the next one to fix it)
def evaluate_buggy_13(X, y, groups):
    X = X.reshape(len(X), -1) / 255.0                                                                    # A
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)         # B
    pca = PCA(n_components=40, random_state=0).fit(X)                                                    # C
    X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)                                                # D
    best = max((1, 3, 5, 9), key=lambda k: KNeighborsClassifier(k).fit(X_tr, y_tr).score(X_te, y_te))   # E
    model = KNeighborsClassifier(best).fit(X_tr, y_tr)                                                   # F
    return best, model.score(X_te, y_te)                                                                 # G


best_b13, score_b13 = evaluate_buggy_13(X_13, Y_13, GROUPS_13)
print(f"{len(X_13)} images, {len(np.unique(GROUPS_13))} sources ; évaluation à corriger : k = {best_b13}, accuracy de test {fr(score_b13)}")'''

TODO_13 = r'''faulty_13 = ...   # a) the letters of the faulty lines of evaluate_buggy_13, e.g. "AB"
shared_13 = ...   # b) the number of test rows of the split of line B whose source image (GROUPS_13) is also in the training rows


def evaluate_fixed_13(X, y, groups):
    """evaluate_buggy_13 without its leaks (same returns: the chosen k, the test accuracy):
    - test split by groups: GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0);
    - k chosen on a validation part carved out of the training rows, also by groups (test_size=0.25, random_state=1);
    - the PCA fitted on the rows that train the model, never on the rows it is scored on;
    - at the end, k-NN with the chosen k refitted on all the training rows, scored once on the test rows."""
    raise NotImplementedError("evaluate_fixed_13")'''

SOLUTION_13 = r'''faulty_13 = "BCE"
g_tr13, g_te13 = train_test_split(GROUPS_13, test_size=0.25, random_state=0, stratify=Y_13)   # the split of line B
shared_13 = int(np.isin(g_te13, g_tr13).sum())


def evaluate_fixed_13(X, y, groups):
    """evaluate_buggy_13 without its leaks (same returns: the chosen k, the test accuracy):
    - test split by groups: GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0);
    - k chosen on a validation part carved out of the training rows, also by groups (test_size=0.25, random_state=1);
    - the PCA fitted on the rows that train the model, never on the rows it is scored on;
    - at the end, k-NN with the chosen k refitted on all the training rows, scored once on the test rows."""
    X = X.reshape(len(X), -1) / 255.0
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups))
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1).split(X[train], y[train], groups[train]))
    fit, val = train[fit], train[val]
    pca = PCA(n_components=40, random_state=0).fit(X[fit])
    scores = {k: KNeighborsClassifier(k).fit(pca.transform(X[fit]), y[fit]).score(pca.transform(X[val]), y[val])
              for k in (1, 3, 5, 9)}
    best = max(scores, key=scores.get)
    pca = PCA(n_components=40, random_state=0).fit(X[train])
    model = KNeighborsClassifier(best).fit(pca.transform(X[train]), y[train])
    return best, model.score(pca.transform(X[test]), y[test])'''

CHECK_13 = r'''wb.check("29.13a", faulty_13)
wb.check("29.13b", shared_13)
with checked("29.13"):
    out_13 = evaluate_fixed_13(X_13, Y_13, GROUPS_13)
    if not returned("29.13", "evaluate_fixed_13", out_13):
        raise StopCheck
    best_13, score_13 = out_13
    verdict("29.13", best_13 in (1, 3, 5, 9) and 0 <= score_13 <= 1, f"k = {best_13}, accuracy de test {fr(score_13)}.",
            f"attendu (k parmi 1, 3, 5, 9 ; une accuracy entre 0 et 1), reçu {out_13}.")
    # A trap: one random label per source image. Nothing can be learned, so an honest evaluation gives about 0.1;
    # a split that leaves near-duplicates on both sides recognises the copies and scores near 1.
    y_rand13 = np.random.default_rng(1).integers(0, 10, GROUPS_13.max() + 1)[GROUPS_13]
    _, score_rand13 = evaluate_fixed_13(X_13, y_rand13, GROUPS_13)
    _, score_rand_b13 = evaluate_buggy_13(X_13, y_rand13, GROUPS_13)
    verdict("29.13", score_rand13 < 0.3,
            f"labels tirés au hasard par source : {fr(score_rand13)} (le hasard), contre {fr(score_rand_b13)} pour la version fautive.",
            f"labels tirés au hasard par source : ta version obtient {fr(score_rand13)}, loin du hasard (0,1) : des copies "
            "d'une même source sont encore des deux côtés du découpage.")
    verdict("29.13", 0.5 < score_13 < score_b13 - 0.1, "sur les vraies données, l'accuracy honnête est nettement plus basse.",
            f"accuracy {fr(score_13)} : attendu nettement moins que les {fr(score_b13)} de la version fautive, et plus de 0,5.")
    # Spies on PCA and KNeighborsClassifier: which rows does each fit and each score see?
    _, events_13 = watch_leaks(evaluate_fixed_13, X_13, Y_13, GROUPS_13)
    scores_13 = [r for kind, r in events_13 if kind == "knn_score"]
    if not scores_13 or any(r is None for _, r in events_13):
        print("ℹ️ Ex 29.13 : ta fonction n'appelle pas PCA et KNeighborsClassifier du notebook sur des lignes reconnaissables : "
              "les vérifications ligne à ligne sont sautées.")
    else:
        test_13 = set(scores_13[-1].tolist())                   # the rows of the last score: the test
        groups_test13 = set(GROUPS_13[sorted(test_13)].tolist())
        verdict("29.13", not any(test_13 & set(r.tolist()) for kind, r in events_13 if kind == "pca_fit"),
                "la PCA n'est jamais ajustée sur des lignes de test.", "la PCA a vu des lignes de test pendant un fit (ligne C).")
        uses_13 = sum(1 for r in scores_13 if test_13 & set(r.tolist()))
        verdict("29.13", uses_13 == 1, "le test ne sert qu'une fois, à la fin.",
                f"le test a servi à {uses_13} scores : choisis k sur une validation, pas sur le test (ligne E).")
        verdict("29.13", not any(groups_test13 & set(GROUPS_13[r].tolist()) for kind, r in events_13 if kind == "knn_fit"),
                "aucun vêtement du test n'a servi à entraîner.", "des vêtements du test ont aussi servi à entraîner : découpe par groupes (ligne B).")
        overlap_13, last_fit_13 = False, None
        for kind, r in events_13:
            if kind == "knn_fit":
                last_fit_13 = r
            elif kind == "knn_score" and last_fit_13 is not None and not test_13 & set(r.tolist()):
                overlap_13 |= bool(set(GROUPS_13[r].tolist()) & set(GROUPS_13[last_fit_13].tolist()))
        verdict("29.13", not overlap_13, "la validation ne partage aucun vêtement avec les lignes qui entraînent.",
                "la validation partage des vêtements avec les lignes qui entraînent : découpe-la aussi par groupes.")'''

RECORD_13 = None   # built by ch29_parts (bug_letters_mistakes)

GIVEN_14 = r'''from scipy import ndimage'''

TODO_14 = r'''def to_mnist_14(images, method="box"):
    """Digits of scikit-learn (N, 8, 8), values 0 to 16, converted to the MNIST format: uint8 (N, 28, 28), 0 to 255
    (values * 255 / 16, rounded with np.rint, clipped to [0, 255] at the very end):
    - "naive": the whole 8 x 8 image enlarged to 28 x 28 (F.interpolate, mode="bilinear", align_corners=False);
    - "box": enlarged to 20 x 20 the same way, then 4 black pixels added on each side (MNIST digits fit in 20 x 20);
    - "center": "box", then each image shifted (ndimage.shift, order=1, float values) so that its centre of mass
      (ndimage.center_of_mass) lands on the centre (13.5, 13.5)."""
    raise NotImplementedError("to_mnist_14")'''

SOLUTION_14 = r'''def to_mnist_14(images, method="box"):
    """Digits of scikit-learn (N, 8, 8), values 0 to 16, converted to the MNIST format: uint8 (N, 28, 28), 0 to 255
    (values * 255 / 16, rounded with np.rint, clipped to [0, 255] at the very end):
    - "naive": the whole 8 x 8 image enlarged to 28 x 28 (F.interpolate, mode="bilinear", align_corners=False);
    - "box": enlarged to 20 x 20 the same way, then 4 black pixels added on each side (MNIST digits fit in 20 x 20);
    - "center": "box", then each image shifted (ndimage.shift, order=1, float values) so that its centre of mass
      (ndimage.center_of_mass) lands on the centre (13.5, 13.5)."""
    t = torch.as_tensor(np.asarray(images), dtype=torch.float32)[:, None] * 255.0 / 16.0
    if method == "naive":
        out = F.interpolate(t, size=(28, 28), mode="bilinear", align_corners=False)[:, 0].numpy()
    else:
        small = F.interpolate(t, size=(20, 20), mode="bilinear", align_corners=False)
        out = F.pad(small, (4, 4, 4, 4))[:, 0].numpy()
        if method == "center":
            shifted = []
            for img in out:
                cy, cx = ndimage.center_of_mass(img)
                shifted.append(ndimage.shift(img, (13.5 - cy, 13.5 - cx), order=1))
            out = np.array(shifted)
        elif method != "box":
            raise ValueError(f"unknown method {method!r}")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)'''

CHECK_14 = r'''with checked("29.14"):
    few_14 = DIGITS_9.images[:20]
    naive_14 = to_mnist_14(few_14, "naive")
    if not returned("29.14", "to_mnist_14", naive_14):
        raise StopCheck
    verdict("29.14", isinstance(naive_14, np.ndarray) and naive_14.shape == (20, 28, 28) and naive_14.dtype == np.uint8,
            "uint8 de forme (N, 28, 28).", f"attendu uint8 (20, 28, 28), reçu {getattr(naive_14, 'dtype', type(naive_14))} {getattr(naive_14, 'shape', '')}.")
    same("29.14", "naive (comparé à digits_naive des outils)", naive_14.astype(float), digits_naive(few_14).astype(float), atol=1.0)
    box_14 = to_mnist_14(few_14, "box")
    border_14 = np.concatenate([box_14[:, :4].ravel(), box_14[:, -4:].ravel(), box_14[:, :, :4].ravel(), box_14[:, :, -4:].ravel()])
    verdict("29.14", border_14.max() == 0 and box_14[:, 4:24, 4:24].max() > 200,
            "box : un cadre noir de 4 pixels autour d'un chiffre de 20 × 20.",
            "box : attendu 4 pixels noirs de chaque côté et le chiffre agrandi au centre, en 20 × 20.")
    t_14 = torch.as_tensor(few_14, dtype=torch.float32)[:, None] * 255.0 / 16.0
    want_box_14 = np.clip(np.rint(F.interpolate(t_14, size=(20, 20), mode="bilinear", align_corners=False)[:, 0].numpy()), 0, 255)
    same("29.14", "box (le carré central de 20 × 20)", box_14[:, 4:24, 4:24].astype(float), want_box_14, atol=1.0)
    center_14 = to_mnist_14(few_14, "center")
    mass_14 = np.array([ndimage.center_of_mass(img.astype(float)) for img in center_14])
    verdict("29.14", np.abs(mass_14 - 13.5).max() < 0.5, "center : le centre de masse de chaque chiffre est au centre de l'image.",
            f"center : un centre de masse s'écarte de {fr(float(np.abs(mass_14 - 13.5).max()), 2)} pixel du centre (13,5 ; 13,5).")'''

EXPERIMENT_14 = r'''with checked("29.14"):
    # 1. The three conversions, the same network trained on MNIST (MODEL_M, 29.8): what does each step bring?
    methods_14 = ["naive", "box", "center"]
    converted_14 = {m: to_mnist_14(DIGITS_9.images, m) for m in methods_14}
    show_digits([XM_TEST[:8]] + [converted_14[m][:8] for m in methods_14], ["MNIST"] + methods_14)
    for m in methods_14:
        acc_14, conf_14 = evaluate(MODEL_M, converted_14[m], YD_9)
        print(f"{m:7s} : accuracy {fr(acc_14)}, confiance moyenne {fr(conf_14)}")

    # 2. Adapt with a few labelled digits of the new source (k per class), scored on the other digits
    rng_14 = np.random.default_rng(0)
    X_best14 = converted_14["center"]
    for k_14 in (0, 1, 5, 20):
        idx_14 = np.concatenate([rng_14.permutation(np.flatnonzero(YD_9 == c))[:k_14] for c in range(10)]).astype(int)
        rest_14 = np.setdiff1d(np.arange(len(YD_9)), idx_14)
        model_14 = fine_tune(MODEL_M, X_best14[idx_14], YD_9[idx_14]) if k_14 else MODEL_M
        print(f"{k_14:2d} chiffres étiquetés par classe : accuracy {fr(evaluate(model_14, X_best14[rest_14], YD_9[rest_14])[0])}")

    # 3. The reference: a model trained on the new source itself (logistic regression on the 8 x 8 pixels)
    from sklearn.linear_model import LogisticRegression
    Xa_14, Xb_14, ya_14, yb_14 = train_test_split(DIGITS_9.data, YD_9, test_size=0.25, random_state=0, stratify=YD_9)
    print(f"référence entraînée sur les chiffres de scikit-learn : accuracy {fr(LogisticRegression(max_iter=3000).fit(Xa_14, ya_14).score(Xb_14, yb_14))}")'''

TODO_15 = r'''def sha256_15(path, chunk_size=1 << 20):
    """The SHA-256 of a file, 64 hexadecimal characters, read by chunks of chunk_size bytes (never all at once)."""
    raise NotImplementedError("sha256_15")


def download_verified_15(url, dest, sha256):
    """Download `url` to `dest`, unless dest already exists with the right SHA-256 (then nothing is downloaded).
    Download into dest + ".part" (for example with urllib.request.urlopen), check its SHA-256, then rename it to dest; on a wrong
    SHA-256, delete the .part file and raise ValueError (dest is left as it was). Return Path(dest)."""
    raise NotImplementedError("download_verified_15")


# The data card of YOUR final project dataset: replace every "..." (the check lists what is missing)
CARD_15 = """# ...

## Provenance
...

## Contenu
...

## Licence
...

## Usage prévu
...

## Limites et biais
...

## Téléchargement
...
"""'''

SOLUTION_15 = r'''def sha256_15(path, chunk_size=1 << 20):
    """The SHA-256 of a file, 64 hexadecimal characters, read by chunks of chunk_size bytes (never all at once)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_15(url, dest, sha256):
    """Download `url` to `dest`, unless dest already exists with the right SHA-256 (then nothing is downloaded).
    Download into dest + ".part" (for example with urllib.request.urlopen), check its SHA-256, then rename it to dest; on a wrong
    SHA-256, delete the .part file and raise ValueError (dest is left as it was). Return Path(dest)."""
    import shutil
    import urllib.request
    dest = Path(dest)
    if dest.exists() and sha256_15(dest) == sha256:
        return dest
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out)
    got = sha256_15(part)
    if got != sha256:
        part.unlink()
        raise ValueError(f"SHA-256 inattendu pour {url} : {got} au lieu de {sha256}")
    part.replace(dest)
    return dest


PENGUINS_SHA_15 = sha256_15(ROOT / "data" / "penguins.csv")
CARD_15 = f"""# Palmer Penguins (exemple de data card)

## Provenance
Mesures de 344 manchots de trois espèces (Adélie, Chinstrap, Gentoo) sur trois îles de l'archipel Palmer (Antarctique),
de 2007 à 2009, par K. Gorman et la station Palmer (LTER). Paquet R palmerpenguins (A. Horst, A. Hill, K. Gorman) :
https://allisonhorst.github.io/palmerpenguins/ ; copie versionnée du workbook : data/penguins.csv.

## Contenu
344 lignes, 8 colonnes : espèce (le label), île, longueur et profondeur du bec (mm), longueur des nageoires (mm),
masse (g), sexe, année. 11 lignes ont des valeurs manquantes (dont 2 sans aucune mesure). Classes : 152 Adélie,
68 Chinstrap, 124 Gentoo.

## Licence
CC0 1.0 (domaine public) : tout usage, commercial compris ; citer les auteurs reste l'usage.

## Usage prévu
Apprendre la classification et l'exploration de données tabulaires. Pas d'usage scientifique sur les manchots en
général : trois îles et trois saisons seulement.

## Limites et biais
Petit (344 lignes), trois îles d'une seule région, trois années : un modèle ne vaut que pour cette population
(décalage de distribution ailleurs). Chinstrap est la classe rare (20 %) et n'a été mesurée que sur une île :
l'île et l'espèce sont confondues.

## Téléchargement
Copie versionnée du workbook, data/penguins.csv, SHA-256 {PENGUINS_SHA_15} :
download_verified_15((ROOT / "data" / "penguins.csv").as_uri(), "penguins.csv", sha256=PENGUINS_SHA_15)
"""'''

CHECK_15 = r'''with checked("29.15"):
    src_15 = WORK_DIR / "source_15.bin"
    src_15.write_bytes(np.random.default_rng(15).bytes(3_000_000))           # a 3 MB file to "download"
    good_15 = hashlib.sha256(src_15.read_bytes()).hexdigest()
    got_15 = sha256_15(src_15)
    if not returned("29.15", "sha256_15", got_15):
        raise StopCheck
    verdict("29.15", got_15 == good_15, "sha256_15 rend la bonne somme de contrôle.",
            f"sha256_15 : attendu {good_15[:16]}…, reçu {str(got_15)[:16]}… (hexdigest(), sur tout le fichier).")
    dest_15 = WORK_DIR / "copy_15.bin"
    path_15 = download_verified_15(src_15.as_uri(), dest_15, good_15)
    verdict("29.15", Path(path_15) == dest_15 and dest_15.exists() and sha256_15(dest_15) == good_15,
            "un téléchargement vérifié arrive à destination.", "le fichier n'est pas arrivé intact dans dest, ou le chemin rendu est faux.")
    verdict("29.15", not dest_15.with_name(dest_15.name + ".part").exists(), "aucun fichier .part ne reste.",
            "un fichier .part reste après un téléchargement réussi : renomme-le en dest.")
    src_15.write_bytes(b"un autre contenu")                                   # the source changes...
    try:
        download_verified_15(src_15.as_uri(), dest_15, good_15)              # ...but dest is already there, intact
        skipped_15 = sha256_15(dest_15) == good_15
    except ValueError:
        skipped_15 = False
    verdict("29.15", skipped_15, "un fichier déjà présent et intact n'est pas téléchargé de nouveau.",
            "dest était déjà là avec la bonne somme : il ne fallait rien télécharger.")
    bad_15 = WORK_DIR / "bad_15.bin"
    try:
        download_verified_15(src_15.as_uri(), bad_15, good_15)
        raised_15 = False
    except ValueError:
        raised_15 = True
    verdict("29.15", raised_15 and not bad_15.exists() and not bad_15.with_name(bad_15.name + ".part").exists(),
            "une somme fausse lève ValueError et ne laisse aucun fichier.",
            "une somme fausse doit lever ValueError, sans laisser dest ni le .part.")
    real_sha15, sizes_15 = hashlib.sha256, []

    class SpySha15:
        """hashlib.sha256 that records the size of every piece of data it receives."""
        def __init__(self, data=b"", **kwargs):
            self.h = real_sha15(data, **kwargs)
            if data:
                sizes_15.append(len(data))

        def update(self, data):
            sizes_15.append(len(data))
            self.h.update(data)

        def __getattr__(self, name):
            return getattr(self.h, name)

    hashlib.sha256 = SpySha15
    try:
        small_15 = sha256_15(dest_15, chunk_size=1000)
    finally:
        hashlib.sha256 = real_sha15
    verdict("29.15", small_15 == good_15 and (not sizes_15 or max(sizes_15) <= 1000),
            "sha256_15 lit le fichier par morceaux de chunk_size octets.",
            f"avec chunk_size=1000, sha256_15 a reçu un morceau de {max(sizes_15, default=0)} octets : lis f.read(chunk_size) en boucle.")
    src2_15 = WORK_DIR / "source2_15.bin"
    src2_15.write_bytes(b"second fichier, " * 1000)
    good2_15 = hashlib.sha256(src2_15.read_bytes()).hexdigest()
    dest2_15 = WORK_DIR / "copy2_15.bin"
    dest2_15.write_bytes(b"un fichier abime")                          # dest exists but is corrupted
    download_verified_15(src2_15.as_uri(), dest2_15, good2_15)
    verdict("29.15", sha256_15(dest2_15) == good2_15, "un fichier déjà présent mais abîmé est téléchargé de nouveau.",
            "dest existait avec une mauvaise somme : il fallait le télécharger de nouveau (vérifie la somme, pas seulement l'existence).")
    dest3_15 = WORK_DIR / "copy3_15.bin"
    dest3_15.write_bytes(b"ancienne version")
    try:
        download_verified_15(src2_15.as_uri(), dest3_15, good_15)       # the source does not match this sum
    except ValueError:
        pass
    verdict("29.15", dest3_15.exists() and dest3_15.read_bytes() == b"ancienne version" and not dest3_15.with_name(dest3_15.name + ".part").exists(),
            "après un échec, dest reste tel quel.", "un téléchargement refusé a écrasé ou supprimé dest : télécharge dans le .part, et ne renomme qu'après vérification.")
    problems_15 = check_card(CARD_15)
    if "..." in CARD_15:
        print("⏳ Ex 29.15 : la data card de ton projet n'est pas encore remplie (il reste des « ... »).")
    else:
        verdict("29.15", not problems_15, "la data card a ses six sections, une source et une somme de contrôle.",
                "data card incomplète : " + " ; ".join(problems_15) + ".")'''

GIVEN_16 = r'''FRUITS_16 = wb.datasets.make_trap_dataset()   # 1 230 rows, 11 columns; data/cards/trap.md
FRUITS_16.head()'''

TODO_16 = r'''# Explore FRUITS_16 in as many cells as you like (describe, value_counts, duplicated, groupby...), then answer:
exact_16 = ...     # a) the number of rows that repeat an earlier row exactly
near_16 = ...      # b) the number of "test" rows that have a near-duplicate among the "train" rows: identical in every
                   #    column except id_lot, split and poids_g, and poids_g at most 1 g apart
missing_16 = ...   # c) the column whose missing values are coded by a number (its name)
constant_16 = ...  # d) the column that carries no information at all
leak_16 = ...      # e) the column that gives the label away
units_16 = ...     # f) the column where some values are in another unit
labels_16 = ...    # g) the number of rows whose label is not one of "pomme", "poire", "coing"
rare_16 = ...      # h) the class (correctly written) with the fewest rows


def clean_16(df):
    """A cleaned copy of df: exact duplicates and the test near-duplicates of b) removed, missing values back to NaN,
    the columns of d) and e) dropped, the values of f) in a single unit, the labels of g) corrected (or the rows
    removed). df itself is not modified."""
    raise NotImplementedError("clean_16")'''

SOLUTION_16 = r'''df_16 = FRUITS_16
exact_16 = int(df_16.duplicated().sum())
train_16, test_16 = df_16[df_16["split"] == "train"], df_16[df_16["split"] == "test"]
key_16 = [c for c in df_16.columns if c not in ("id_lot", "split", "poids_g")]
pairs_16 = test_16.reset_index().merge(train_16[key_16 + ["poids_g"]], on=key_16, suffixes=("", "_train"))
pairs_16 = pairs_16[(pairs_16["poids_g"] - pairs_16["poids_g_train"]).abs() <= 1]
near_16 = int(pairs_16["index"].nunique())
print(df_16.describe().T[["min", "max"]])
missing_16 = "acidite"          # min = -999
constant_16 = "pays"            # a single distinct value
leak_16 = "code_qualite"        # its letter (M, R, C) is always the same for a given fruit
units_16 = "diametre_cm"        # a few values near 100: millimetres
valid_16 = {"pomme", "poire", "coing"}
labels_16 = int((~df_16["fruit"].isin(valid_16)).sum())
rare_16 = df_16.loc[df_16["fruit"].isin(valid_16), "fruit"].value_counts().idxmin()
print(pd.crosstab(df_16["code_qualite"].str[1], df_16["fruit"]))


def clean_16(df):
    """A cleaned copy of df: exact duplicates and the test near-duplicates of b) removed, missing values back to NaN,
    the columns of d) and e) dropped, the values of f) in a single unit, the labels of g) corrected (or the rows
    removed). df itself is not modified."""
    out = df.drop_duplicates().copy()
    train, test = out[out["split"] == "train"], out[out["split"] == "test"]
    key = [c for c in out.columns if c not in ("id_lot", "split", "poids_g")]
    pairs = test.reset_index().merge(train[key + ["poids_g"]], on=key, suffixes=("", "_train"))
    near = pairs.loc[(pairs["poids_g"] - pairs["poids_g_train"]).abs() <= 1, "index"].unique()
    out = out.drop(index=near)
    out["acidite"] = out["acidite"].replace(-999.0, np.nan)
    out = out.drop(columns=["pays", "code_qualite"])
    out.loc[out["diametre_cm"] > 30, "diametre_cm"] /= 10
    fixed = out["fruit"].str.strip().str.lower().replace({"pome": "pomme", "pommes": "pomme", "poir": "poire"})
    out["fruit"] = fixed
    return out[out["fruit"].isin(["pomme", "poire", "coing"])].reset_index(drop=True)'''

CHECK_16 = r'''wb.check("29.16a", exact_16)
wb.check("29.16b", near_16)
wb.check("29.16c", missing_16)
wb.check("29.16d", constant_16)
wb.check("29.16e", leak_16)
wb.check("29.16f", units_16)
wb.check("29.16g", labels_16)
wb.check("29.16h", rare_16)
with checked("29.16"):
    before_16 = FRUITS_16.copy()
    clean_c16 = clean_16(FRUITS_16)
    if not returned("29.16", "clean_16", clean_c16):
        raise StopCheck
    verdict("29.16", FRUITS_16.equals(before_16), "le tableau d'origine n'a pas été modifié.",
            "clean_16 a modifié FRUITS_16 : travaille sur une copie (df.copy()).")
    answers_16 = wb.datasets.trap_dataset_answers()
    gone_16 = [c for c in (answers_16["constant_column"], answers_16["leaking_column"]) if c in clean_c16.columns]
    verdict("29.16", not gone_16, "les colonnes inutile et fuyante ont disparu.", f"colonnes encore présentes : {gone_16}.")
    cols_c16 = [c for c in clean_c16.columns if c != "id_lot"]
    verdict("29.16", not clean_c16[cols_c16].duplicated().any(), "plus aucun doublon exact.",
            f"{int(clean_c16[cols_c16].duplicated().sum())} ligne(s) encore en double.")
    verdict("29.16", not (clean_c16["acidite"] == -999).any(), "plus de −999 dans acidite.",
            "il reste des −999 dans acidite : remplace-les par NaN (ou retire ces lignes).")
    verdict("29.16", clean_c16["diametre_cm"].max() < 30, "les diamètres sont tous en centimètres.",
            f"un diamètre vaut encore {fr(float(clean_c16['diametre_cm'].max()), 1)} : convertis les millimètres.")
    verdict("29.16", set(clean_c16["fruit"]) <= {"pomme", "poire", "coing"}, "tous les labels sont valides.",
            f"labels encore invalides : {sorted(set(clean_c16['fruit']) - {'pomme', 'poire', 'coing'})}.")
    tr_c16, te_c16 = clean_c16[clean_c16["split"] == "train"], clean_c16[clean_c16["split"] == "test"]
    key_c16 = [c for c in clean_c16.columns if c not in ("id_lot", "split", "poids_g")]
    left_c16 = te_c16.merge(tr_c16[key_c16], on=key_c16)
    verdict("29.16", len(left_c16) == 0, "aucun quasi-doublon de l'entraînement ne reste dans le test.",
            f"{len(left_c16)} ligne(s) de test ont encore leur quasi-doublon dans l'entraînement.")
    verdict("29.16", len(clean_c16) >= 1150, f"{len(clean_c16)} lignes gardées sur {len(FRUITS_16)}.",
            f"seulement {len(clean_c16)} lignes gardées : as-tu jeté plus que les défauts ?")'''

EXPERIMENT_16 = r'''# What the defects change: the same forest, trained on "train" and scored on "test", before and after cleaning
from sklearn.ensemble import RandomForestClassifier


def score_16(df, columns):
    """Accuracy on the "test" rows of a random forest trained on the "train" rows (missing values set to -1)."""
    data = df.copy()
    X = pd.get_dummies(data[columns], dtype=float).fillna(-1)
    train, test = data["split"] == "train", data["split"] == "test"
    forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X[train], data.loc[train, "fruit"])
    return forest.score(X[test], data.loc[test, "fruit"])


with checked("29.16"):
    if (clean_16(FRUITS_16)["split"] == "test").sum() == 0:
        print("❌ Ex 29.16 : clean_16 ne garde aucune ligne de test : l'expérience est impossible.")
        raise StopCheck
    dirty_cols_16 = [c for c in FRUITS_16.columns if c not in ("id_lot", "split", "fruit")]
    clean_e16 = clean_16(FRUITS_16)
    clean_cols_16 = [c for c in clean_e16.columns if c not in ("id_lot", "split", "fruit")]
    print(f"données brutes, toutes les colonnes : accuracy {fr(score_16(FRUITS_16, dirty_cols_16))}")
    print(f"données nettoyées                   : accuracy {fr(score_16(clean_e16, clean_cols_16))}")'''
