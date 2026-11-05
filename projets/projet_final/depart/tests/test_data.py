"""Tests of data.py (PF.2 to PF.4), marked « todo »: skipped with ⏳ until the functions exist, then they must pass.

They run on a toy table built here, not on your dataset: what they check (a split without overlap, reproducible,
that keeps groups together and the order of time, a frozen file that cannot change silently) holds for every
project. Add the tests of YOUR data: the schema of load_examples, the number of rows of the data card.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from conftest import PACKAGES

pytestmark = pytest.mark.todo
PACKAGE = sorted(PACKAGES)[0]
data = pytest.importorskip(f"{PACKAGE}.data")
Config = pytest.importorskip(f"{PACKAGE}.config").Config


@pytest.fixture
def toy():
    """200 examples: an id, a feature, an imbalanced label (25 % of « b »), 50 groups of 4, a time stamp."""
    rng = np.random.default_rng(0)
    ids = np.arange(200)
    return pd.DataFrame({"id": ids, "x": rng.normal(size=200), "label": np.where(ids % 4 == 0, "b", "a"),
                         "group": ids // 4, "t": ids})


def test_sha256_file_is_the_one_of_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(256)) * 5000)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert data.sha256_file(path, chunk_size=1000) == expected, f"expected the SHA-256 of hashlib, {expected}"


def test_the_splits_are_disjoint_and_complete(toy):
    splits = data.make_splits(toy, Config(target="label"))
    assert sorted(splits) == ["test", "train", "val"], f"expected the keys train, val, test, got {sorted(splits)}"
    every = [str(i) for name in ("train", "val", "test") for i in splits[name]]
    assert len(every) == len(set(every)) == len(toy), \
        f"expected {len(toy)} distinct ids in the three splits, got {len(every)} ids, {len(set(every))} distinct"


def test_the_split_is_reproducible_and_follows_the_seed(toy):
    a, b = data.make_splits(toy, Config(seed=0)), data.make_splits(toy, Config(seed=0))
    c = data.make_splits(toy, Config(seed=1))
    assert a == b, "expected the same split for the same data and the same seed"
    assert a != c, "expected another split for another seed"


def test_a_group_never_crosses_two_splits(toy):
    splits = data.make_splits(toy, Config(group_column="group"))
    where = {}
    for name, ids in splits.items():
        for g in toy.set_index(toy["id"].astype(str)).loc[[str(i) for i in ids], "group"]:
            where.setdefault(g, set()).add(name)
    crossing = sorted(g for g, names in where.items() if len(names) > 1)
    assert not crossing, f"expected each group in one split only, groups in two splits: {crossing[:10]}"


def test_the_test_comes_after_the_training_in_time(toy):
    splits = data.make_splits(toy, Config(time_column="t"))
    t = toy.set_index(toy["id"].astype(str))["t"]
    last_train = t.loc[[str(i) for i in splits["train"]]].max()
    first_test = t.loc[[str(i) for i in splits["test"]]].min()
    assert last_train < first_test, f"expected every test time after the training, got {last_train} >= {first_test}"


def test_a_saved_split_reloads_with_its_digest(toy, tmp_path):
    splits = data.make_splits(toy, Config())
    digest = data.save_splits(splits, tmp_path / "splits.json")
    assert digest == hashlib.sha256((tmp_path / "splits.json").read_bytes()).hexdigest(), \
        "expected save_splits to return the SHA-256 of the file it wrote"
    again = data.load_splits(tmp_path / "splits.json", digest)
    assert {k: sorted(map(str, v)) for k, v in again.items()} == {k: sorted(map(str, v)) for k, v in splits.items()}, \
        "expected load_splits to give back the split that save_splits wrote"


def test_a_changed_split_is_refused(toy, tmp_path):
    path = tmp_path / "splits.json"
    digest = data.save_splits(data.make_splits(toy, Config()), path)
    path.write_text(path.read_text(encoding="utf-8").replace('"1"', '"1 "', 1), encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_splits(path, digest)


def test_select_refuses_the_test_split(toy):
    splits = data.make_splits(toy, Config())
    train = data.select(toy, splits, "train")
    assert len(train) == len(splits["train"]), f"expected {len(splits['train'])} training rows, got {len(train)}"
    with pytest.raises(PermissionError):
        data.select(toy, splits, "test")


# TODO PF.2: test_load_examples_has_the_schema_of_the_data_card (an "id" column, unique; the target; the number of
# rows written in data/card.md), marked @pytest.mark.data and skipped when the data is not downloaded (in the CI):
# pytest.skip if the files are not found. PF_STRICT=1 accepts this skip, and only this one.
