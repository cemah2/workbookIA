"""Data of the mini-project MP5: CIFAR-10 (the 50 000 training images, and a test set opened once).

Provided (do not change it): your notebook and your tests use it.

    X, y = data.load_cifar10()                     # 50 000 uint8 images (N, 32, 32, 3) and their labels 0-9
    X_test, y_test = data.open_test(PROJECT, "resnet 16-32-64, ...")   # once, at the very end (test_log.json)

The validation set is yours to cut from the 50 000 training images (cifarkit.stratified_split, step MP5.1).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import wb

CLASSES = list(wb.datasets.CIFAR10_CLASSES)


def load_cifar10() -> tuple[np.ndarray, np.ndarray]:
    """(X, y): the 50 000 training images of CIFAR-10, uint8 (N, 32, 32, 3) in their original order, and their labels
    (int64, 0-9). The first call downloads about 145 MB from the Hugging Face Hub (then: the cache of the workbook)."""
    X, y = wb.datasets.load_cifar10("train", seed=None)
    return np.ascontiguousarray(X, dtype=np.uint8), np.asarray(y, dtype=np.int64)


def test_openings(folder) -> list[dict]:
    """The openings of the test set logged so far in folder / "test_log.json", oldest first."""
    path = Path(folder) / "test_log.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


test_openings.__test__ = False   # not a test for pytest, despite its name


def open_test(folder, description: str) -> tuple[np.ndarray, np.ndarray]:
    """(X_test, y_test): the 10 000 test images of CIFAR-10, uint8 (N, 32, 32, 3), and their labels.

    Every opening is logged in folder / "test_log.json" with its date and the description of the model evaluated.
    Opening again for the same description is allowed (the notebook is run again); for another description, it
    still opens, but warns that the test score is no longer an honest measure.
    """
    folder = Path(folder)
    log = test_openings(folder)
    description = str(description)
    same = [entry for entry in log if entry["description"] == description]
    if same:
        print(f"🔓 Le test a déjà été ouvert le {same[0]['date']} pour ce même modèle : ce n'est pas une nouvelle décision.")
    else:
        if log:
            print(f"⚠️ Le test a déjà été ouvert le {log[0]['date']} pour un autre modèle : ce nouveau score n'est plus "
                  "une mesure honnête, car tu as vu le test avant de faire ce choix. Dis-le dans ton README.")
        log.append({"date": dt.date.today().isoformat(), "description": description})
        (folder / "test_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"🔓 Test ouvert ({len(log)} ouverture(s) enregistrée(s) dans test_log.json).")
    X, y = wb.datasets.load_cifar10("test", seed=None)
    return np.ascontiguousarray(X, dtype=np.uint8), np.asarray(y, dtype=np.int64)


open_test.__test__ = False
