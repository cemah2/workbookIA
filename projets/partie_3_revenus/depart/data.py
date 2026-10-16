"""Data of the mini-project MP3: the 1994 census (Adult), and the vault that keeps the test people aside.

Provided (do not change it): your notebook and your module use it.

    X, y = data.load_census()                        # 48 842 people: 14 columns, y = 1 above 50 000 $ a year
    vault = data.TestVault(PROJECT, X, y)            # reads PROJECT / "test_indices.npy"
    X_train, y_train = vault.train_part()            # everything you may look at
    X_test, y_test = vault.open("final model: ...")  # once, at the very end (logged in vault.json)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import wb

FEATURES = [column for column in wb.datasets.ADULT_COLUMNS if column != "class"]   # the 14 columns, in order
TARGET = "class"                # "<=50K" or ">50K": the yearly income
POSITIVE = ">50K"               # y = 1 for an income over 50 000 $ a year (in 1994 dollars)


def load_census(source: str = "auto") -> tuple[pd.DataFrame, np.ndarray]:
    """The 48 842 people of the census: X, a DataFrame of the 14 columns FEATURES (missing values as NaN), and y.

    ``y`` is an int array: 1 when the income is over 50 000 $ a year, 0 otherwise. ``source`` is passed to
    ``wb.datasets.load_adult``: ``"auto"`` (OpenML, else the copy of the repository), ``"openml"`` or ``"csv"``;
    both sources give the same table.
    """
    frame = wb.datasets.load_adult(source=source)
    return frame[FEATURES].copy(), (frame[TARGET] == POSITIVE).to_numpy(dtype=int)


class TestVault:
    """Keeps the test people aside until the final evaluation, and logs every opening.

    The test indices (row positions in X) are read from ``folder / "test_indices.npy"``
    (written once by your notebook, step MP3.1, then committed). ``train_part`` gives the
    other rows. ``open`` gives the test rows and writes the opening in ``folder /
    "vault.json"``: the date and a description of the model evaluated. Opening the vault
    again for the same description is allowed (the notebook is run again); for another
    description, it still opens but warns that the test score is no longer an honest measure.
    The rows keep their index of X, so that a row can be traced back to the full table.
    """

    __test__ = False            # not a test class for pytest, despite its name

    def __init__(self, folder, X, y):
        self.folder = Path(folder)
        self.X, self.y = pd.DataFrame(X), np.asarray(y)
        if len(self.X) != len(self.y):
            raise ValueError(f"X and y have different lengths: {len(self.X)} and {len(self.y)}")
        self.indices_path = self.folder / "test_indices.npy"
        self.log_path = self.folder / "vault.json"
        if not self.indices_path.exists():
            raise FileNotFoundError(f"{self.indices_path} does not exist: write it first (step MP3.1)")
        test = np.load(self.indices_path)
        if test.ndim != 1 or len(np.unique(test)) != len(test) or test.min() < 0 or test.max() >= len(self.y):
            raise ValueError("test_indices.npy must hold distinct indices between 0 and n - 1")
        self.test_indices = np.sort(test)
        self.train_indices = np.setdiff1d(np.arange(len(self.y)), self.test_indices)

    def train_part(self) -> tuple[pd.DataFrame, np.ndarray]:
        """X and y of the people who are not in the test set (copies)."""
        return self.X.iloc[self.train_indices].copy(), self.y[self.train_indices].copy()

    def openings(self) -> list[dict]:
        """The openings logged so far, oldest first."""
        if not self.log_path.exists():
            return []
        return json.loads(self.log_path.read_text(encoding="utf-8"))

    def open(self, description: str) -> tuple[pd.DataFrame, np.ndarray]:
        """X and y of the test people (copies); the opening is logged with its description."""
        description = str(description)
        log = self.openings()
        same = [entry for entry in log if entry["description"] == description]
        if same:
            print(f"🔓 Le coffre a déjà été ouvert le {same[0]['date']} pour ce même modèle : "
                  "ce n'est pas une nouvelle décision.")
        else:
            if log:
                print(f"⚠️ Le coffre a déjà été ouvert le {log[0]['date']} pour un autre modèle. Ce nouveau score de "
                      "test n'est plus une mesure honnête : tu as vu le test avant de faire ce choix. Dis-le dans "
                      "ton README.")
            log.append({"date": dt.date.today().isoformat(), "description": description})
            self.log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"🔓 Coffre ouvert ({len(log)} ouverture(s) enregistrée(s) dans vault.json).")
        return self.X.iloc[self.test_indices].copy(), self.y[self.test_indices].copy()
