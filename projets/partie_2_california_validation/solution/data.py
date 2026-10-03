"""Data of the mini-project MP2: the California districts, and the vault that keeps the test districts aside.

Provided (do not change it): your notebook and your module use it.

    X, y = data.load_housing()                       # the 20 640 districts
    vault = data.TestVault(PROJECT, X, y)            # reads PROJECT / "test_indices.npy"
    X_train, y_train = vault.train_part()            # everything you may look at
    X_test, y_test = vault.open("final model: ...")  # once, at the very end (logged in vault.json)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import wb

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"          # median house value of the district, in hundreds of thousands of dollars
CAP = 5.0                       # the target is capped: 5.00001 means "500 000 $ or more"


def load_housing() -> tuple[np.ndarray, np.ndarray]:
    """The 20 640 districts of the 1990 census: X of shape (20640, 8) (columns FEATURES) and y."""
    frame = wb.datasets.load_california()
    return frame[FEATURES].to_numpy(dtype=float), frame[TARGET].to_numpy(dtype=float)


class TestVault:
    """Keeps the test districts aside until the final evaluation, and logs every opening.

    The test indices are read from ``folder / "test_indices.npy"`` (written once by your
    notebook, step MP2.1, then committed). ``train_part`` gives the other districts.
    ``open`` gives the test districts and writes the opening in ``folder / "vault.json"``:
    the date and a description of the model evaluated. Opening the vault again for the
    same description is allowed (the notebook is run again); for another description, it
    still opens but warns that the test score is no longer an honest measure.
    """

    __test__ = False            # not a test class for pytest, despite its name

    def __init__(self, folder, X, y):
        self.folder = Path(folder)
        self.X, self.y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.indices_path = self.folder / "test_indices.npy"
        self.log_path = self.folder / "vault.json"
        if not self.indices_path.exists():
            raise FileNotFoundError(f"{self.indices_path} does not exist: write it first (step MP2.1)")
        test = np.load(self.indices_path)
        if test.ndim != 1 or len(np.unique(test)) != len(test) or test.min() < 0 or test.max() >= len(self.y):
            raise ValueError("test_indices.npy must hold distinct indices between 0 and n - 1")
        self.test_indices = np.sort(test)
        self.train_indices = np.setdiff1d(np.arange(len(self.y)), self.test_indices)

    def train_part(self) -> tuple[np.ndarray, np.ndarray]:
        """X and y of the districts that are not in the test set (copies)."""
        return self.X[self.train_indices].copy(), self.y[self.train_indices].copy()

    def openings(self) -> list[dict]:
        """The openings logged so far, oldest first."""
        if not self.log_path.exists():
            return []
        return json.loads(self.log_path.read_text(encoding="utf-8"))

    def open(self, description: str) -> tuple[np.ndarray, np.ndarray]:
        """X and y of the test districts (copies); the opening is logged with its description."""
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
        return self.X[self.test_indices].copy(), self.y[self.test_indices].copy()
