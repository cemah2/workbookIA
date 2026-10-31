"""Tests of wb.datasets. The versioned datasets are always tested; the ones
that need a download are marked ``network`` (run them with --run-network)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wb import datasets as D

ROOT = Path(__file__).resolve().parents[2]


def test_penguins():
    df = D.load_penguins()
    assert df.shape == (344, 8)
    assert set(df["species"].unique()) == {"Adelie", "Chinstrap", "Gentoo"}
    assert df.isna().any().any()
    assert D.load_penguins(dropna=True).shape[0] == 333
    raw = D.load_penguins_raw()
    assert raw.shape[0] == 344 and raw.shape[1] == 17


def test_california_csv():
    df = D.load_california(source="csv")
    assert df.shape == (20640, 9)
    assert list(df.columns)[-1] == "MedHouseVal"
    X, y = D.load_california(source="csv", return_X_y=True)
    assert X.shape == (20640, 8) and y.shape == (20640,)
    Xn, yn = D.load_california(source="csv", return_X_y=True, as_frame=False)
    assert isinstance(Xn, np.ndarray)


@pytest.mark.network
def test_california_csv_identical_to_sklearn():
    online = D.load_california(source="sklearn")
    local = D.load_california(source="csv")
    pd.testing.assert_frame_equal(online.reset_index(drop=True), local, check_exact=True)


def test_adult_csv():
    df = D.load_adult(source="csv")
    assert df.shape == (48842, 15) and list(df.columns) == D.ADULT_COLUMNS
    assert all(df[c].dtype == np.int64 for c in D.ADULT_NUMERIC)
    assert all(df[c].dtype == object for c in D.ADULT_CATEGORICAL + ["class"])
    assert df.isna().sum().to_dict() == {**{c: 0 for c in D.ADULT_COLUMNS}, "workclass": 2799, "occupation": 2809,
                                         "native-country": 857}
    assert df["class"].value_counts().to_dict() == {"<=50K": 37155, ">50K": 11687}
    assert not (df[D.ADULT_CATEGORICAL] == "?").any().any(), "the « ? » of the original files must be NaN"
    with pytest.raises(ValueError):
        D.load_adult(source="internet")


def test_adult_copy_is_small_enough():
    assert (ROOT / "data" / "adult.csv.gz").stat().st_size < 1_000_000


@pytest.mark.network
def test_adult_csv_identical_to_openml():
    pd.testing.assert_frame_equal(D.load_adult(source="openml"), D.load_adult(source="csv"), check_exact=True)


def test_mnist_versioned():
    X, y = D.load_mnist("train")
    assert X.shape == (60000, 28, 28) and X.dtype == np.uint8
    assert y.shape == (60000,) and y.dtype == np.int64 and set(np.unique(y)) == set(range(10))
    Xt, yt = D.load_mnist("test", n=500, flatten=True, normalize=True, seed=1)
    assert Xt.shape == (500, 784) and Xt.dtype == np.float32 and 0 <= Xt.min() and Xt.max() <= 1
    Xt2, _ = D.load_mnist("test", n=500, flatten=True, normalize=True, seed=1)
    np.testing.assert_array_equal(Xt, Xt2)
    with pytest.raises(ValueError):
        D.load_mnist("valid")


def test_mnist_npz_is_small_enough():
    assert (ROOT / "data" / "mnist.npz").stat().st_size < 20e6


def test_texts():
    holmes, verne = D.load_holmes(), D.load_verne()
    assert "Sherlock Holmes" in holmes and "Phileas Fogg" in verne
    for text in (holmes, verne):
        assert "GUTENBERG" not in text.upper()[:2000] and "GUTENBERG" not in text.upper()[-2000:]
        assert "\r" not in text and len(text) > 300_000
    assert "é" in verne and verne.startswith("Jules Verne") and verne.rstrip().endswith("FIN")
    assert "START OF THE PROJECT GUTENBERG" in D.load_holmes(strip_header=False)


def test_strip_gutenberg_without_markers():
    assert D.strip_gutenberg("hello") == "hello\n"


def test_sunspots():
    df = D.load_sunspots()
    assert list(df.columns[:5]) == ["date", "year", "month", "decimal_year", "sunspots"]
    assert df["date"].iloc[0] == pd.Timestamp("1749-01-01")
    assert len(df) > 3300 and df["sunspots"].notna().all()
    assert df["date"].is_monotonic_increasing and df["date"].diff().dt.days.max() <= 31
    definitive = D.load_sunspots(definitive_only=True)
    assert definitive["definitive"].all() and len(definitive) <= len(df)


def test_parse_silso():
    raw = "1749;01;1749.042;  96.7; -1.0;   -1;1\n2026;08;2026.623;  76.0; 12.3; 1248;0\n"
    df = D.parse_silso(raw)
    assert np.isnan(df.loc[0, "std"]) and df.loc[1, "n_obs"] == 1248
    assert df["definitive"].tolist() == [True, False]


def test_read_idx_roundtrip():
    import gzip

    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    header = bytes([0, 0, 8, 3]) + b"".join(int(d).to_bytes(4, "big") for d in arr.shape)
    raw = header + arr.tobytes()
    np.testing.assert_array_equal(D._read_idx(raw), arr)
    np.testing.assert_array_equal(D._read_idx(gzip.compress(raw)), arr)


def test_catalogue_and_cards():
    table = D.list_datasets()
    assert len(table) == len(D.CATALOGUE)
    for name, _, file, _, card in D.CATALOGUE:
        assert (ROOT / "data" / "cards" / f"{card}.md").exists(), f"missing card for {name}"
        if file:
            assert (ROOT / "data" / file).exists(), f"missing versioned file for {name}"
    text = D.dataset_card("penguins", show=False)
    assert "Licence" in text or "licence" in text


def test_card_fingerprints_match_the_versioned_files():
    """Every « SHA-256 de `data/...` : `hex` » of a data card is the hash of that file (chapter 29: a card gives the
    fingerprint of what it describes; a rebuilt file must update its card)."""
    import hashlib

    pattern = re.compile(r"SHA-256 (?:de|du fichier)\s*(?:`(data/[^`]+)`)?\s*:\s*`([0-9a-f]{64})`")
    seen = 0
    for card in sorted((ROOT / "data" / "cards").glob("*.md")):
        text = card.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            path = match.group(1)
            if path is None:   # « SHA-256 du fichier » : the file of the « **Fichier** » row of the same card
                path = re.search(r"\*\*Fichier\*\* \| `(data/[^`]+)`", text).group(1)
            digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            assert digest == match.group(2), f"expected the SHA-256 of {path} in {card.name}, got {match.group(2)}"
            seen += 1
    assert seen >= 7, f"expected at least 7 fingerprints in data/cards, found {seen}"


def test_cifar10_huggingface_revision_is_pinned():
    import inspect

    assert re.fullmatch(r"[0-9a-f]{40}", D.CIFAR10_HF_REVISION)
    assert "revision=CIFAR10_HF_REVISION" in inspect.getsource(D._cifar_from_huggingface)


def test_substitute_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("WB_CACHE_DIR", str(tmp_path))
    failing = [("nowhere", lambda: (_ for _ in ()).throw(OSError("offline")))]
    with pytest.raises(ConnectionError):
        D._load_with_sources("X", tmp_path / "x.npz", failing, True, False, (4, 4))
    X, y = D._load_with_sources("X", tmp_path / "x.npz", failing, True, True, (4, 4))
    assert X.shape[1:] == (4, 4)


def test_quiet_huggingface_silences_then_restores_the_output():
    pytest.importorskip("datasets")
    import datasets.utils as utils
    from datasets.utils import logging as logs

    before = (utils.are_progress_bars_disabled(), logs.get_verbosity())
    with D._quiet_huggingface():
        assert utils.are_progress_bars_disabled() and logs.get_verbosity() == logs.ERROR
    assert (utils.are_progress_bars_disabled(), logs.get_verbosity()) == before


def test_make_env():
    pytest.importorskip("gymnasium")
    env = D.make_env("CartPole-v1", seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    lake = D.make_env("FrozenLake-v1", seed=0, is_slippery=False)
    assert lake.observation_space.n == 16


@pytest.mark.network
def test_fashion_mnist_download():
    X, y = D.load_fashion_mnist("test", n=100)
    assert X.shape == (100, 28, 28) and len(D.FASHION_CLASSES) == 10
    X, y = D.load_fashion_mnist("train")
    assert X.shape == (60000, 28, 28)


@pytest.mark.network
def test_cifar10_download():
    X, y = D.load_cifar10("test", n=50, channels_first=True)
    assert X.shape == (50, 3, 32, 32) and X.dtype == np.uint8
    X, y = D.load_cifar10("train")
    assert X.shape == (50000, 32, 32, 3) and set(np.unique(y)) == set(range(10))
