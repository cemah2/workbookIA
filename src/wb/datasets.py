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
Each dataset has a card: ``dataset_card("mnist")``.
"""

from __future__ import annotations

import gzip
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


def _cifar_from_huggingface():
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10", cache_dir=str(cache_dir() / "hf"))
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
    PyTorch expects. First call downloads ~170 MB (never versioned).
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
        raise ImportError("gymnasium n'est pas installé : pip install gymnasium==1.3.0") from exc
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
    ("synthetic", "wb.synth.*", None, "données générées", "synthetic"),
    ("rl_envs", "make_env()", None, "environnements d'apprentissage par renforcement", "rl_envs"),
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
