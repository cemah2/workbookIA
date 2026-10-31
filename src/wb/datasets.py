"""Loaders for the workbook's common-thread datasets (BIBLE §9).

Every loader has fallbacks: versioned copy in ``data/`` → official download
→ mirror. Heavy downloads go to a cache directory (``data/downloads/`` locally,
``/content/wb_cache`` on Colab: much faster than Google Drive), never to git.

Quick reference
---------------
>>> df = load_penguins()                     # pandas DataFrame, 344 rows
>>> df = load_california()                   # 20 640 rows, target MedHouseVal
>>> X, y = load_mnist("train", n=5000)       # uint8 (5000, 28, 28), labels 0-9
>>> X, y = load_fashion_mnist("test")        # same format, clothes
>>> X, y = load_cifar10("train", n=1000)     # uint8 (1000, 32, 32, 3)
>>> text = load_holmes()                     # str, English
>>> texte = load_verne()                     # str, French
>>> df = load_sunspots()                     # monthly sunspot numbers since 1749
>>> env = make_env("CartPole-v1", seed=0)    # gymnasium environment
>>> img = load_sample_image("china", 192)    # float32 (3, H, W) in [0, 1], free photo (CC BY 2.0)
>>> art = load_style_image("starry_night", 192)   # public-domain painting, same format
>>> net, mean, std = load_pretrained("squeezenet1_1")   # frozen ImageNet features
Each dataset has a card: ``dataset_card("mnist")``.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib
import io
import os
import shutil
import tempfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np

from wb.core import is_colab, repo_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MNIST_CLASSES = [str(i) for i in range(10)]
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

URLS = {
    "penguins": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
    "penguins_raw": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins_raw.csv",
    "holmes": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "verne": "https://www.gutenberg.org/cache/epub/800/pg800.txt",
    "sunspots": "https://www.sidc.be/SILSO/INFO/snmtotcsv.php",
    "fashion_mnist_github": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}

FILES = {
    "penguins": "penguins.csv",
    "penguins_raw": "penguins_raw.csv",
    "california": "california_housing.csv",
    "mnist": "mnist.npz",
    "holmes": "text/holmes_adventures_pg1661.txt",
    "verne": "text/verne_tour_du_monde_pg800.txt",
    "sunspots": "sunspots_monthly.csv",
    "adult": "adult.csv.gz",
}

_SUBSTITUTE_WARNING = (
    "⚠️ SUBSTITUT SYNTHÉTIQUE : le vrai dataset {name} est inaccessible ici. Les données "
    "renvoyées ont la bonne forme mais sont aléatoires : le code tourne, les résultats "
    "n'ont aucun sens. Cellule « à valider sur Colab »."
)


# ---------------------------------------------------------------------------
# Paths and downloads
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    """The versioned ``data/`` directory of the repository."""
    return repo_root() / "data"


def cache_dir() -> Path:
    """Directory for downloaded (non-versioned) data.

    ``WB_CACHE_DIR`` if set; ``/content/wb_cache`` on Colab; else ``data/downloads``.
    """
    env = os.environ.get("WB_CACHE_DIR")
    if env:
        path = Path(env)
    elif is_colab():
        path = Path("/content/wb_cache")
    else:
        path = data_dir() / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: str | Path, *, timeout: float = 60.0, retries: int = 2) -> Path:
    """Download ``url`` to ``dest`` (atomic write). Returns the path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "dl-workbook/0.1 (educational)"})
    last_error = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                    shutil.copyfileobj(response, tmp)
            os.replace(tmp.name, dest)
            os.chmod(dest, 0o644)  # temporary files are created private (0600)
            return dest
        except Exception as exc:  # network errors are many and varied
            last_error = exc
    raise ConnectionError(f"Téléchargement impossible : {url} ({last_error})")


def _subsample(X, y, n: int | None, seed: int | None):
    if n is None or n >= len(y):
        return X, y
    idx = np.sort(np.random.default_rng(seed).choice(len(y), size=n, replace=False))
    return X[idx], y[idx]


def _finish_images(X, y, n, seed, flatten, normalize):
    X, y = _subsample(X, y, n, seed)
    if flatten:
        X = X.reshape(len(X), -1)
    if normalize:
        X = X.astype(np.float32) / 255.0
    return X, y.astype(np.int64)


def _check_split(split: str) -> bool:
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    return split == "train"


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------
def _read_csv_with_fallback(key: str, **kwargs):
    import pandas as pd

    local = data_dir() / FILES[key]
    if local.exists():
        return pd.read_csv(local, **kwargs)
    cached = cache_dir() / FILES[key]
    if not cached.exists():
        warnings.warn(f"{local} introuvable : téléchargement depuis {URLS[key]}")
        download(URLS[key], cached)
    return pd.read_csv(cached, **kwargs)


def load_penguins(dropna: bool = False):
    """Palmer Penguins: 344 penguins, 3 species, 8 columns, some missing values.

    ``dropna=True`` removes the rows with a missing value (333 rows remain).
    """
    df = _read_csv_with_fallback("penguins")
    return df.dropna().reset_index(drop=True) if dropna else df


def load_penguins_raw():
    """The raw, messier version of Palmer Penguins (17 columns), for data cleaning (ch. 12)."""
    return _read_csv_with_fallback("penguins_raw")


def load_california(as_frame: bool = True, return_X_y: bool = False, source: str = "auto"):
    """California Housing (1990 census): 20 640 districts, 8 features, target ``MedHouseVal``.

    ``source``: ``"auto"`` (scikit-learn download, else the versioned CSV),
    ``"sklearn"`` or ``"csv"``. Returns a DataFrame, or ``(X, y)`` with
    ``return_X_y=True``.
    """
    import pandas as pd

    import socket

    df = None
    if source in ("auto", "sklearn"):
        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)  # never hang behind a blocking proxy
        try:
            from sklearn.datasets import fetch_california_housing

            bunch = fetch_california_housing(data_home=str(cache_dir() / "sklearn"), as_frame=True)
            df = bunch.frame
        except Exception as exc:
            if source == "sklearn":
                raise
            warnings.warn(f"fetch_california_housing a échoué ({exc}) : j'utilise la copie CSV.")
        finally:
            socket.setdefaulttimeout(previous_timeout)
    if df is None:  # round_trip: exactly the same floats as scikit-learn
        df = pd.read_csv(data_dir() / FILES["california"], float_precision="round_trip")
    if return_X_y:
        X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
        return (X, y) if as_frame else (X.to_numpy(), y.to_numpy())
    return df if as_frame else df.to_numpy()


ADULT_NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
ADULT_CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                     "native-country"]
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                 "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                 "class"]


def _normalize_adult(df):
    """The same DataFrame whatever the source: columns in order, int64 numbers, text as object with NaN."""
    import numpy as np

    out = df[ADULT_COLUMNS].copy()
    for column in ADULT_NUMERIC:
        out[column] = out[column].astype("int64")
    for column in ADULT_CATEGORICAL + ["class"]:
        values = out[column].astype(object)
        out[column] = values.where(values.notna(), np.nan)
    return out.reset_index(drop=True)


def load_adult(source: str = "auto"):
    """Adult / Census Income (UCI, 1994 US census): 48 842 people, 14 features and the target ``class``.

    6 numeric columns (``ADULT_NUMERIC``, int64) and 8 text columns (``ADULT_CATEGORICAL``, object), with
    missing values as NaN (the "?" of the original files) in ``workclass``, ``occupation`` and
    ``native-country``; ``class`` is ``"<=50K"`` or ``">50K"`` (income over 50 000 $ a year).
    ``source``: ``"auto"`` (OpenML, ``fetch_openml(data_id=1590)``, else the versioned copy
    ``data/adult.csv.gz``), ``"openml"`` or ``"csv"``. Both sources give the same DataFrame.
    """
    import socket

    import pandas as pd

    if source not in ("auto", "openml", "csv"):
        raise ValueError(f"source must be 'auto', 'openml' or 'csv', not {source!r}")
    if source in ("auto", "openml"):
        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)  # never hang behind a blocking proxy
        try:
            from sklearn.datasets import fetch_openml

            bunch = fetch_openml(data_id=1590, as_frame=True, data_home=str(cache_dir() / "sklearn"))
            return _normalize_adult(bunch.frame)
        except Exception as exc:
            if source == "openml":
                raise
            warnings.warn(f"fetch_openml a échoué ({exc}) : j'utilise la copie versionnée.")
        finally:
            socket.setdefaulttimeout(previous_timeout)
    df = pd.read_csv(data_dir() / FILES["adult"], keep_default_na=False, na_values=[""])
    return _normalize_adult(df)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def _read_idx(raw: bytes) -> np.ndarray:
    """Parse an IDX file (MNIST format), gzipped or not."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _mnist_like_from_torchvision(name: str):
    import torchvision

    root = cache_dir() / "torchvision"
    cls = getattr(torchvision.datasets, name)
    train = cls(str(root), train=True, download=True)
    test = cls(str(root), train=False, download=True)
    return (train.data.numpy(), train.targets.numpy(), test.data.numpy(), test.targets.numpy())


def _mnist_like_from_openml(openml_name: str):
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(openml_name, version=1, return_X_y=True, as_frame=False,
                        data_home=str(cache_dir() / "sklearn"))
    X = X.astype(np.uint8).reshape(-1, 28, 28)
    y = y.astype(np.int64)
    return X[:60000], y[:60000], X[60000:], y[60000:]


def _fashion_from_github():
    base = URLS["fashion_mnist_github"]
    parts = []
    for fname in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                  "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"):
        path = cache_dir() / "fashion_github" / fname
        if not path.exists():
            download(base + fname, path)
        parts.append(_read_idx(path.read_bytes()))
    return tuple(parts)


def _load_cached_npz(path: Path, train: bool):
    with np.load(path) as data:
        key = "train" if train else "test"
        return data[f"x_{key}"], data[f"y_{key}"]


def _save_npz(path: Path, x_train, y_train, x_test, y_test) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x_train=x_train, y_train=np.asarray(y_train, dtype=np.uint8),
                        x_test=x_test, y_test=np.asarray(y_test, dtype=np.uint8))


def _load_with_sources(name: str, cache_file: Path, sources, train: bool,
                       allow_substitute: bool, shape):
    """Try each source in order; cache the first success as a compressed .npz."""
    if cache_file.exists():
        return _load_cached_npz(cache_file, train)
    errors = []
    for label, fetch in sources:
        try:
            x_tr, y_tr, x_te, y_te = fetch()
            _save_npz(cache_file, x_tr, y_tr, x_te, y_te)
            return (x_tr, y_tr) if train else (x_te, y_te)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    if allow_substitute:
        print(_SUBSTITUTE_WARNING.format(name=name))
        n = 1000 if train else 200
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, (n, *shape), dtype=np.uint8), rng.integers(0, 10, n)
    raise ConnectionError(
        f"{name} impossible à charger. Sources essayées :\n  - " + "\n  - ".join(errors)
        + "\nCauses possibles : pas de connexion, ou bibliothèque absente (torchvision, datasets). "
        "Sur Colab tout est installé ; en local, voir 00_setup/INSTALL_LOCAL.md."
    )


def load_mnist(split: str = "train", n: int | None = None, *, flatten: bool = False,
               normalize: bool = False, seed: int | None = 0, allow_substitute: bool = False):
    """MNIST handwritten digits: ``X`` uint8 (N, 28, 28), ``y`` int64 in 0-9.

    ``split``: "train" (60 000) or "test" (10 000). ``n``: random subset of
    size n (reproducible with ``seed``). ``flatten=True`` → (N, 784).
    ``normalize=True`` → float32 in [0, 1]. Source: ``data/mnist.npz``
    (versioned), else torchvision, else OpenML.
    """
    train = _check_split(split)
    local = data_dir() / FILES["mnist"]
    if local.exists():
        X, y = _load_cached_npz(local, train)
    else:
        X, y = _load_with_sources(
            "MNIST", cache_dir() / "mnist.npz",
            [("torchvision", lambda: _mnist_like_from_torchvision("MNIST")),
             ("OpenML", lambda: _mnist_like_from_openml("mnist_784"))],
            train, allow_substitute, (28, 28))
    return _finish_images(X, y, n, seed, flatten, normalize)


def load_fashion_mnist(split: str = "train", n: int | None = None, *, flatten: bool = False,
                       normalize: bool = False, seed: int | None = 0,
                       allow_substitute: bool = False):
    """Fashion-MNIST (Zalando): 10 clothing classes, same format as :func:`load_mnist`.

    Class names: ``FASHION_CLASSES``. Sources: torchvision, else the GitHub
    repository of Zalando Research, else OpenML. Cached after the first load.
    """
    train = _check_split(split)
    X, y = _load_with_sources(
        "Fashion-MNIST", cache_dir() / "fashion_mnist.npz",
        [("torchvision", lambda: _mnist_like_from_torchvision("FashionMNIST")),
         ("GitHub zalandoresearch", _fashion_from_github),
         ("OpenML", lambda: _mnist_like_from_openml("Fashion-MNIST"))],
        train, allow_substitute, (28, 28))
    return _finish_images(X, y, n, seed, flatten, normalize)


def _cifar_from_torchvision():
    import torchvision

    root = str(cache_dir() / "torchvision")
    train = torchvision.datasets.CIFAR10(root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True)
    return (train.data, np.asarray(train.targets), test.data, np.asarray(test.targets))


@contextlib.contextmanager
def _quiet_huggingface():
    """No progress bars, warnings or info logs from `datasets` and `huggingface_hub` (restored afterwards).

    A first download otherwise fills a beginner's notebook with red stderr lines (tqdm without ipywidgets,
    the Hub's notice about unauthenticated requests, one bar per split).
    """
    restore = []
    for package in ("datasets", "huggingface_hub"):
        try:
            utils = importlib.import_module(f"{package}.utils")
            logs = importlib.import_module(f"{package}.utils.logging")
            bars, level = utils.are_progress_bars_disabled(), logs.get_verbosity()
            utils.disable_progress_bars()
            logs.set_verbosity_error()
        except Exception:  # noqa: BLE001 - another version of the package: keep its output
            continue
        restore.append((utils, logs, bars, level))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*IProgress not found.*")
            warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
            yield
    finally:
        for utils, logs, bars, level in restore:
            try:
                logs.set_verbosity(level)
                if not bars:
                    utils.enable_progress_bars()
            except Exception:  # noqa: BLE001
                pass


# Public name (chapter 29 loads a dataset of the Hub itself and keeps its notebook quiet in the same way)
quiet_huggingface = _quiet_huggingface

# The commit of the Hub repository that the workbook reads (checked on 2026-10-31: still the latest one), so that
# a later change of the dataset on the Hub cannot change the images (chapter 29: a reproducible download)
CIFAR10_HF_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"


def _cifar_from_huggingface():
    from datasets import load_dataset

    print("ℹ️ Premier chargement de CIFAR-10 (≈ 145 Mo depuis Hugging Face) : une seule fois, le cache sert ensuite.")
    with _quiet_huggingface():
        ds = load_dataset("uoft-cs/cifar10", revision=CIFAR10_HF_REVISION, cache_dir=str(cache_dir() / "hf"))
    out = []
    for split in ("train", "test"):
        part = ds[split]
        out.append(np.stack([np.asarray(img) for img in part["img"]]).astype(np.uint8))
        out.append(np.asarray(part["label"]))
    return tuple(out)


def load_cifar10(split: str = "train", n: int | None = None, *, normalize: bool = False,
                 channels_first: bool = False, seed: int | None = 0,
                 allow_substitute: bool = False):
    """CIFAR-10: 60 000 colour images 32×32, 10 classes (``CIFAR10_CLASSES``).

    ``X`` uint8 (N, 32, 32, 3); ``channels_first=True`` → (N, 3, 32, 32) as
    PyTorch expects. First call downloads ~145 MB from Hugging Face (~170 MB from Toronto), then keeps a
    ~173 MB cache ``cifar10.npz`` (never versioned).
    """
    train = _check_split(split)
    X, y = _load_with_sources(
        "CIFAR-10", cache_dir() / "cifar10.npz",
        # Hugging Face first: served by a CDN, usually much faster than the Toronto server.
        [("Hugging Face", _cifar_from_huggingface), ("torchvision", _cifar_from_torchvision)],
        train, allow_substitute, (32, 32, 3))
    X, y = _finish_images(X, y, n, seed, False, normalize)
    if channels_first:
        X = X.transpose(0, 3, 1, 2)
    return X, y


def torchvision_dataset(name: str, train: bool = True, transform=None, download: bool = True):
    """A torchvision dataset object (MNIST, FashionMNIST, CIFAR10...) stored in the cache.

    Use it with a ``torch.utils.data.DataLoader`` from chapter 20 onwards.
    """
    import torchvision

    cls = getattr(torchvision.datasets, name)
    return cls(str(cache_dir() / "torchvision"), train=train, transform=transform, download=download)


# ---------------------------------------------------------------------------
# Texts
# ---------------------------------------------------------------------------
_GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg header, licence footer and credit lines."""
    start = text.find(_GUTENBERG_START)
    if start != -1:
        text = text[text.find("\n", start) + 1:]
    end = text.find(_GUTENBERG_END)
    if end != -1:
        text = text[:end]
    lines = text.strip().split("\n")
    while lines and (lines[0].startswith("Produced by") or not lines[0].strip()):
        lines.pop(0)  # transcriber credit
    while lines and (lines[-1].lower().startswith("end of") and "gutenberg" in lines[-1].lower()
                     or not lines[-1].strip()):
        lines.pop()  # "End of Project Gutenberg's ..."
    return "\n".join(lines) + "\n"


def _load_text(key: str, strip: bool) -> str:
    path = data_dir() / FILES[key]
    if not path.exists():
        path = cache_dir() / Path(FILES[key]).name
        if not path.exists():
            warnings.warn(f"copie versionnée absente : téléchargement depuis {URLS[key]}")
            download(URLS[key], path)
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return strip_gutenberg(text) if strip else text


def load_holmes(strip_header: bool = True) -> str:
    """*The Adventures of Sherlock Holmes* (A. Conan Doyle, Gutenberg #1661), English."""
    return _load_text("holmes", strip_header)


def load_verne(strip_header: bool = True) -> str:
    """*Le Tour du monde en quatre-vingts jours* (J. Verne, Gutenberg #800), French."""
    return _load_text("verne", strip_header)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def parse_silso(raw: str):
    """Parse the SILSO monthly CSV (``;`` separated) into a clean DataFrame."""
    import pandas as pd

    df = pd.read_csv(io.StringIO(raw), sep=";", header=None,
                     names=["year", "month", "decimal_year", "sunspots", "std", "n_obs", "definitive"])
    df["std"] = df["std"].where(df["std"] >= 0)
    df["n_obs"] = df["n_obs"].where(df["n_obs"] >= 0).astype("Int64")
    df["sunspots"] = df["sunspots"].where(df["sunspots"] >= 0)
    df["definitive"] = df["definitive"].astype(bool)
    return df


def load_sunspots(definitive_only: bool = False):
    """Monthly mean total sunspot number since January 1749 (SILSO, Brussels).

    Columns: ``date``, ``year``, ``month``, ``decimal_year``, ``sunspots``,
    ``std``, ``n_obs``, ``definitive``. The last months are provisional
    (``definitive == False``); ``definitive_only=True`` drops them.
    """
    import pandas as pd

    path = data_dir() / FILES["sunspots"]
    if path.exists():
        df = pd.read_csv(path)
        df["n_obs"] = df["n_obs"].astype("Int64")
        df["definitive"] = df["definitive"].astype(bool)
    else:
        warnings.warn("copie versionnée absente : téléchargement depuis SILSO")
        raw_path = download(URLS["sunspots"], cache_dir() / "SN_m_tot_V2.0.csv")
        df = parse_silso(raw_path.read_text(encoding="utf-8"))
    df.insert(0, "date", pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)))
    if definitive_only:
        df = df[df["definitive"]].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Images and pre-trained networks (chapter 28)
# ---------------------------------------------------------------------------
CONTENT_IMAGES = {   # photos shipped with scikit-learn (sklearn/datasets/images/README.txt): CC BY 2.0, credit needed
    "china": {"file": "china.jpg", "title": "Temple en Chine (photo)", "author": "danielbuechele (Flickr)",
              "source": "https://www.flickr.com/photos/danielbuechele/6061409035/",
              "license": "CC BY 2.0 (https://creativecommons.org/licenses/by/2.0/)"},
    "flower": {"file": "flower.jpg", "title": "Fleur (photo)", "author": "vultilion (Flickr ; le README de scikit-learn "
               "crédite danielbuechele)", "source": "https://www.flickr.com/photos/vultilion/6056698931/",
               "license": "CC BY 2.0 (https://creativecommons.org/licenses/by/2.0/)"},
}

_WIKIMEDIA_THUMB = "https://upload.wikimedia.org/wikipedia/commons/thumb/"
STYLE_IMAGES = {     # public-domain paintings, 500 px wide reproductions from Wikimedia Commons, versioned in data/images
    "starry_night": {
        "title": "La Nuit étoilée", "artist": "Vincent van Gogh", "date": "1889", "artist_died": 1890,
        "collection": "Museum of Modern Art, New York",
        "source": _WIKIMEDIA_THUMB + "e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
                  "500px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "sha256": "9aab486cee8e3a778a260cbd00591684ccc6cf1b204a12b0bf7e62258b746404"},
    "great_wave": {
        "title": "La Grande Vague de Kanagawa", "artist": "Katsushika Hokusai", "date": "vers 1831",
        "artist_died": 1849, "collection": "estampe (nombreux tirages ; reproduction de Wikimedia Commons)",
        "source": _WIKIMEDIA_THUMB + "a/a5/Tsunami_by_hokusai_19th_century.jpg/"
                  "500px-Tsunami_by_hokusai_19th_century.jpg",
        "sha256": "343166b78ac071e82603925f06f9892197bdccd410356fc65650b3346b12a177"},
    "composition_vii": {
        "title": "Composition VII", "artist": "Vassily Kandinsky", "date": "1913", "artist_died": 1944,
        "collection": "Galerie Tretiakov, Moscou",
        "source": _WIKIMEDIA_THUMB + "b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/"
                  "500px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
        "sha256": "d2641f057c9d71d8a156492cf9711b6f74019f084deba6e38e829890bcbe3176"},
}
for _info in STYLE_IMAGES.values():
    _info["license"] = (f"domaine public (l'artiste est mort en {_info['artist_died']} : plus de 70 ans) ; "
                        "reproduction fidèle non protégée (directive (UE) 2019/790, art. 14)")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_array(img, size: int | None) -> np.ndarray:
    """A PIL image as a float32 array (3, H, W) in [0, 1]; ``size`` = the longer side (proportions kept)."""
    from PIL import Image

    img = img.convert("RGB")
    if size is not None:
        if int(size) < 8:
            raise ValueError("size must be at least 8 pixels")
        w, h = img.size
        scale = int(size) / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)


def load_sample_image(name: str = "china", size: int | None = None) -> np.ndarray:
    """A free photo shipped with scikit-learn ("china" or "flower", CC BY 2.0: credit with ``image_credit``).

    Returns a float32 array (3, H, W) in [0, 1]; ``size`` is the longer side after resizing (proportions kept).
    No download.
    """
    from PIL import Image
    from sklearn.datasets import load_sample_image as _sk_load

    if name not in CONTENT_IMAGES:
        raise ValueError(f"unknown photo {name!r}; choose among {sorted(CONTENT_IMAGES)}")
    return _image_array(Image.fromarray(_sk_load(CONTENT_IMAGES[name]["file"])), size)


def procedural_style(size: int = 256, seed: int = 0) -> np.ndarray:
    """A synthetic "painting" (3, H, W) in [0, 1]: swirling coloured strokes, deterministic. The fallback of
    ``load_style_image`` when a painting cannot be read; it is not a work of art and has no author."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    field = np.zeros((size, size))
    for _ in range(6):
        cx, cy, k = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(6, 14)
        r = np.hypot(xx - cx, yy - cy)
        field += np.sin(k * r + 3 * np.arctan2(yy - cy, xx - cx) + rng.uniform(0, 2 * np.pi))
    palette = rng.uniform(0.1, 0.9, size=(3, 3))
    t = (np.tanh(field) + 1) / 2
    img = np.stack([palette[c, 0] * (1 - t) + palette[c, 1] * t + 0.15 * np.sin(20 * t + c) for c in range(3)])
    return np.clip(img, 0, 1).astype(np.float32)


def load_style_image(name: str = "starry_night", size: int | None = None, *, allow_download: bool = True) -> np.ndarray:
    """A public-domain painting (see ``STYLE_IMAGES``), float32 (3, H, W) in [0, 1], longer side ``size``.

    Order: the versioned copy ``data/images/<name>.jpg`` → the cache → a download from Wikimedia Commons (checked
    by SHA-256) → ``procedural_style`` with a warning, so that a notebook always runs.
    """
    import hashlib

    from PIL import Image

    if name not in STYLE_IMAGES:
        raise ValueError(f"unknown painting {name!r}; choose among {sorted(STYLE_IMAGES)}")
    info = STYLE_IMAGES[name]
    for path in (data_dir() / "images" / f"{name}.jpg", cache_dir() / "images" / f"{name}.jpg"):
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]:
            return _image_array(Image.open(path), size)
    if allow_download:
        path = cache_dir() / "images" / f"{name}.jpg"
        try:
            download(info["source"], path, retries=1)
            if hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]:
                return _image_array(Image.open(path), size)
            path.unlink()
        except ConnectionError:
            pass
    warnings.warn(f"⚠️ SUBSTITUT : le tableau {name!r} est introuvable ici ; une texture procédurale le remplace "
                  "(le code tourne, le style n'est pas celui du tableau).")
    return procedural_style(size or 256, seed=sorted(STYLE_IMAGES).index(name))


def image_credit(name: str) -> str:
    """The credit line of a photo (``CONTENT_IMAGES``) or of a painting (``STYLE_IMAGES``), in French."""
    if name in CONTENT_IMAGES:
        i = CONTENT_IMAGES[name]
        return f"{i['title']}, {i['author']}, {i['source']}, licence {i['license']}"
    if name in STYLE_IMAGES:
        i = STYLE_IMAGES[name]
        return (f"{i['artist']}, « {i['title']} » ({i['date']}), {i['collection']} ; reproduction : {i['source']} ; "
                f"{i['license']}")
    raise ValueError(f"unknown image {name!r}")


def load_pretrained(name: str = "squeezenet1_1"):
    """A frozen ImageNet network for chapter 28: ``(features, mean, std)``.

    ``features`` is the convolutional part (``model.features``) of ``squeezenet1_1`` (4.7 MiB of weights, the fast
    mode) or ``vgg16`` (528 MiB, the full mode), in ``eval()`` mode, every parameter with ``requires_grad=False``,
    every ReLU rebuilt with ``inplace=False`` (so that a hook keeps what it records); ``mean`` and ``std`` are the
    ImageNet normalisation, as tensors (1, 3, 1, 1). The weights are downloaded once into the torch cache.
    """
    import torch
    import torchvision

    builders = {"squeezenet1_1": torchvision.models.squeezenet1_1, "vgg16": torchvision.models.vgg16}
    if name not in builders:
        raise ValueError(f"unknown network {name!r}; choose among {sorted(builders)}")
    features = builders[name](weights="DEFAULT").features
    for module in features.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    features.eval()
    for p in features.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    return features, mean, std


# ---------------------------------------------------------------------------
# A dataset with planted defects (chapter 29)
# ---------------------------------------------------------------------------
TRAP_CLASSES = ("pomme", "poire", "coing")


def make_trap_dataset(seed: int = 2916, n: int = 1200):
    """A fruit dataset (pandas DataFrame) with EIGHT planted defects, for the data audit of chapter 29.

    Columns: ``id_lot``, ``poids_g``, ``diametre_cm``, ``sucre``, ``acidite``, ``couleur_r``, ``couleur_v``, ``pays``,
    ``code_qualite``, ``fruit`` (the label) and ``split`` ("train" or "test"). The defects (what an audit must find)
    are described by ``trap_dataset_answers(seed, n)``, which the notebook never shows before the check.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    probs = np.array([0.6, 0.38, 0.02])                                   # defect 8: a very rare class
    fruit = rng.choice(len(TRAP_CLASSES), size=n, p=probs)
    centre = {0: (150, 7.5, 12.0, 0.55), 1: (180, 6.8, 10.0, 0.35), 2: (300, 9.0, 8.0, 0.90)}
    rows = []
    for k, f in enumerate(fruit):
        w, d, su, ac = centre[int(f)]
        rows.append({"id_lot": 10_000 + k,
                     "poids_g": round(float(rng.normal(w, w * 0.12)), 1),
                     "diametre_cm": round(float(rng.normal(d, 0.6)), 2),
                     "sucre": round(float(rng.normal(su, 1.2)), 2),
                     "acidite": round(float(rng.normal(ac, 0.08)), 3),
                     "couleur_r": int(np.clip(rng.normal(170 if f == 0 else 120, 30), 0, 255)),
                     "couleur_v": int(np.clip(rng.normal(120 if f == 0 else 170, 30), 0, 255)),
                     "pays": "France",                                   # defect 4: a constant column
                     "fruit": TRAP_CLASSES[int(f)]})
    df = pd.DataFrame(rows)
    letters = {"pomme": "M", "poire": "R", "coing": "C"}
    df["code_qualite"] = [f"Q{letters[t]}-{rng.integers(1, 6)}" for t in df["fruit"]]   # defect 5: leaks the label (QM-3...)
    df["split"] = np.where(rng.random(n) < 0.8, "train", "test")
    train_idx = df.index[df["split"] == "train"].to_numpy()
    test_idx = df.index[df["split"] == "test"].to_numpy()
    # defect 6: inconsistent units, some diameters in millimetres
    mm = rng.choice(df.index.to_numpy(), size=37, replace=False)
    df.loc[mm, "diametre_cm"] = (df.loc[mm, "diametre_cm"] * 10).round(1)
    # defect 3: missing values coded -999 in acidite
    miss = rng.choice(df.index.to_numpy(), size=29, replace=False)
    df.loc[miss, "acidite"] = -999.0
    # defect 7: invalid labels, misspellings of the TRUE fruit (a typo never turns a pear into an apple, so that the
    # labels can be repaired by normalising them); no random draw here, the stream of rng is unchanged
    typos = rng.choice(df.index.to_numpy(), size=11, replace=False)
    variants = {"pomme": ["pome", "pommes", "POMME"], "poire": ["poire ", "Poire", "poir"], "coing": ["Coing", "coings"]}
    used = {t: 0 for t in variants}
    misspelt = []
    for t in df.loc[typos, "fruit"]:
        misspelt.append(variants[t][used[t] % len(variants[t])])
        used[t] += 1
    df.loc[typos, "fruit"] = misspelt
    # defect 2: near-duplicates of training rows placed in the test split (tiny noise on the measures)
    src = rng.choice(train_idx, size=17, replace=False)
    near = df.loc[src].copy()
    near["poids_g"] = (near["poids_g"] + rng.normal(0, 0.05, len(near))).round(2)
    near["split"] = "test"
    near["id_lot"] = 20_000 + np.arange(len(near))
    # defect 1: exact duplicate rows (in the training split)
    dup = df.loc[rng.choice(train_idx, size=13, replace=False)].copy()
    df = pd.concat([df, near, dup], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    del test_idx
    return df[["id_lot", "poids_g", "diametre_cm", "sucre", "acidite", "couleur_r", "couleur_v", "pays",
               "code_qualite", "fruit", "split"]]


def trap_dataset_answers(seed: int = 2916, n: int = 1200) -> dict:
    """The eight defects of ``make_trap_dataset(seed, n)``, measured on the table itself (used by the solutions)."""
    df = make_trap_dataset(seed, n)
    measures = ["poids_g", "diametre_cm", "sucre", "acidite", "couleur_r", "couleur_v"]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    key = ["diametre_cm", "sucre", "acidite", "couleur_r", "couleur_v", "fruit"]
    near = test.merge(train[key].drop_duplicates(), on=key, how="inner")
    valid = df["fruit"].isin(TRAP_CLASSES)
    counts = df.loc[valid, "fruit"].value_counts()
    return {"exact_duplicates": int(df.duplicated().sum()),
            "near_duplicates": int(len(near)),
            "missing_code_column": "acidite", "missing_code_count": int((df["acidite"] == -999).sum()),
            "constant_column": "pays", "leaking_column": "code_qualite",
            "units_column": "diametre_cm", "units_rows": int((df["diametre_cm"] > 30).sum()),
            "invalid_labels": int((~valid).sum()), "rare_class": str(counts.idxmin()),
            "measures": measures}


# ---------------------------------------------------------------------------
# Reinforcement learning environments
# ---------------------------------------------------------------------------
def make_env(name: str = "CartPole-v1", seed: int | None = 0, **kwargs):
    """Create a gymnasium environment (FrozenLake-v1, CartPole-v1...) and seed it.

    Returns the environment; ``env.reset(seed=seed)`` has already been called
    so ``env.action_space`` is seeded too.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium n'est pas installé : pip install gymnasium==1.4.0") from exc
    env = gym.make(name, **kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
CATALOGUE = [  # (name, loader, versioned file, kind, card)
    ("penguins", "load_penguins()", "penguins.csv", "tabulaire, classification", "penguins"),
    ("penguins_raw", "load_penguins_raw()", "penguins_raw.csv", "tabulaire brut, nettoyage", "penguins"),
    ("california_housing", "load_california()", "california_housing.csv", "tabulaire, régression",
     "california_housing"),
    ("mnist", "load_mnist()", "mnist.npz", "images 28×28, 10 chiffres", "mnist"),
    ("fashion_mnist", "load_fashion_mnist()", None, "images 28×28, 10 vêtements", "fashion_mnist"),
    ("cifar10", "load_cifar10()", None, "images couleur 32×32, 10 classes", "cifar10"),
    ("holmes", "load_holmes()", "text/holmes_adventures_pg1661.txt", "texte anglais", "holmes"),
    ("verne", "load_verne()", "text/verne_tour_du_monde_pg800.txt", "texte français", "verne"),
    ("sunspots", "load_sunspots()", "sunspots_monthly.csv", "série temporelle mensuelle", "sunspots"),
    ("adult", "load_adult()", "adult.csv.gz", "tabulaire mixte, classification binaire (revenus)", "adult"),
    ("synthetic", "wb.synth.*", None, "données générées", "synthetic"),
    ("rl_envs", "make_env()", None, "environnements d'apprentissage par renforcement", "rl_envs"),
    ("images_ch28", "load_sample_image(), load_style_image()", "images/starry_night.jpg",
     "photos libres et tableaux du domaine public", "images"),
    ("pretrained_cnn", "load_pretrained()", None, "réseaux pré-entraînés sur ImageNet (poids torchvision)", "pretrained"),
    ("trap_fruits", "make_trap_dataset()", None, "tabulaire généré, huit défauts plantés (audit du ch. 29)", "trap"),
]


def list_datasets():
    """Table of the workbook datasets: loader, versioned file, card."""
    import pandas as pd

    rows = []
    for name, loader, file, kind, card in CATALOGUE:
        versioned = bool(file) and (data_dir() / file).exists()
        full_loader = loader if loader.startswith("wb.") else f"wb.datasets.{loader}"
        rows.append({"dataset": name, "loader": full_loader, "type": kind,
                     "versionné": "oui" if versioned else "non (téléchargé)",
                     "fiche": f"data/cards/{card}.md"})
    return pd.DataFrame(rows)


def dataset_card(name: str, show: bool = True) -> str | None:
    """Print the data card of a dataset (``show=False`` returns the text instead)."""
    cards = {entry[0]: entry[4] for entry in CATALOGUE}
    path = data_dir() / "cards" / f"{cards.get(name, name)}.md"
    if not path.exists():
        raise FileNotFoundError(f"pas de fiche pour {name!r} ; voir list_datasets()")
    text = path.read_text(encoding="utf-8")
    if show:
        print(text)
        return None
    return text
