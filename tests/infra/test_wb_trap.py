"""The dataset with planted defects of chapter 29 (wb.datasets.make_trap_dataset)."""

import pandas as pd

from wb import datasets


def test_trap_dataset_is_deterministic_and_shaped():
    a, b = datasets.make_trap_dataset(), datasets.make_trap_dataset()
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["id_lot", "poids_g", "diametre_cm", "sucre", "acidite", "couleur_r", "couleur_v", "pays",
                               "code_qualite", "fruit", "split"]
    assert set(a["split"]) == {"train", "test"}


def test_trap_dataset_has_its_eight_defects():
    df, ans = datasets.make_trap_dataset(), datasets.trap_dataset_answers()
    assert ans["exact_duplicates"] == 13 and df.duplicated().sum() == 13
    assert ans["near_duplicates"] == 17
    assert (df["acidite"] == -999).sum() == ans["missing_code_count"] > 0
    assert df["pays"].nunique() == 1
    clean = df[df["fruit"].isin(datasets.TRAP_CLASSES)]
    assert pd.crosstab(clean["code_qualite"].str[:2], clean["fruit"]).gt(0).sum(axis=1).eq(1).all(), "the code must leak the label"
    assert (df["diametre_cm"] > 30).sum() == ans["units_rows"] > 0
    assert (~df["fruit"].isin(datasets.TRAP_CLASSES)).sum() == ans["invalid_labels"] == 11
    assert clean["fruit"].value_counts(normalize=True)["coing"] < 0.03


def test_other_seeds_also_work():
    df = datasets.make_trap_dataset(seed=1, n=400)
    assert len(df) == 400 + 17 + 13
