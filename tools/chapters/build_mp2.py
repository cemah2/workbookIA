#!/usr/bin/env python
"""Build the mini-project MP2 from a single source (used by Claude).

    python tools/chapters/build_mp2.py
    python tools/run_all_notebooks.py projets/partie_2_california_validation/solution/mp2_california.ipynb --inplace
    python tools/run_all_notebooks.py projets/partie_2_california_validation/depart/mp2_california.ipynb

Writes the two notebooks (depart/ and solution/mp2_california.ipynb) and the starter
module depart/housing.py, generated from the reference solution/housing.py: same
constants, signatures and docstrings, private helpers removed, every public function
raising NotImplementedError with the step of the notebook where it is written.
``python tools/start_chapter.py CP2`` copies depart/ into mon_travail/projets/.
Steps MP2.1 to MP2.7 follow the brief (docs/syllabus/data/cp2.json, miniproject).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import ROOT, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "projets/partie_2_california_validation"
STARTER = f"{FOLDER}/depart/mp2_california.ipynb"
SOLUTION = f"{FOLDER}/solution/mp2_california.ipynb"
MODULE_SOLUTION = ROOT / FOLDER / "solution" / "housing.py"
MODULE_STARTER = ROOT / FOLDER / "depart" / "housing.py"

STEPS = [("MP2.1", "Cadrer le projet, geler le jeu de test, explorer l'entraînement", 60),
         ("MP2.2", "Une référence naïve, puis les moindres carrés en validation croisée", 60),
         ("MP2.3", "Features polynomiales, Ridge et Lasso : le degré et α choisis par validation croisée", 120),
         ("MP2.4", "Des zones géographiques par k-means : la silhouette, puis la validation croisée", 90),
         ("MP2.5", "Diagnostiquer : courbes d'apprentissage, biais et variance, résidus par région", 90),
         ("MP2.6", "Ouvrir le coffre une seule fois : le test, et la comparaison avec scikit-learn", 45),
         ("MP2.7", "Emballer : tests, README de portfolio, notebook relancé, commit", 75)]

# Step of the notebook where each public function of housing.py is written (starter module).
TODO_STEPS = {
    "make_test_indices": "MP2.1",
    "rmse": "MP2.2",
    "HousingModel.fit": 'MP2.2 (degree=1, penalty="none"), then MP2.3 (degree, Ridge, Lasso) and MP2.4 (zones)',
    "HousingModel.transform": "MP2.2, then MP2.4 (the zones)",
    "HousingModel.zones": "MP2.4",
    "HousingModel.predict": "MP2.2",
    "cross_validate": "MP2.2",
    "out_of_fold_predictions": "MP2.5",
    "validation_curve": "MP2.3",
    "learning_curve": "MP2.5",
}

STARTER_DOC_FIRST = ("Median house value of California districts with linear models (mini-project of part II): "
                     "YOUR module.")
STARTER_DOC_END = """Write every function that raises NotImplementedError (the step of the notebook is given
in the TODO). Keep the signatures and the docstrings: the notebook and the tests rely on
them. You may add private helpers (names starting with _).
"""

# ---------------------------------------------------------------------------
# The starter module, generated from the reference
# ---------------------------------------------------------------------------


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    """At most 2 blank lines before a top-level line, at most 1 before an indented one."""
    out: list[str] = []
    blanks = 0
    for line in lines:
        if not line.strip():
            blanks += 1
            continue
        if out:
            out.extend([""] * min(blanks, 1 if line.startswith(" ") else 2))
        blanks = 0
        out.append(line)
    return out


def starter_module(source: str) -> str:
    """The starter housing.py: the reference without its bodies (docstrings kept) and without private helpers."""
    tree = ast.parse(source)
    lines = source.splitlines()
    drop: set[int] = set()           # 0-based line numbers removed
    stubs: dict[int, str] = {}       # 0-based line of the docstring end -> the line inserted after it

    def first_line(node) -> int:
        return min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1

    def handle(node, qualname: str) -> None:
        name = node.name
        if name.startswith("_") and name != "__init__":
            drop.update(range(first_line(node), node.end_lineno))
            return
        if name == "__init__":
            return
        doc = node.body[0]
        if not (isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str)):
            raise ValueError(f"{qualname} has no docstring")
        drop.update(range(doc.end_lineno, node.end_lineno))
        indent = " " * doc.col_offset
        stubs[doc.end_lineno - 1] = f"{indent}raise NotImplementedError  # TODO {TODO_STEPS[qualname]}"

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            handle(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    handle(item, f"{node.name}.{item.name}")
    out = []
    for i, line in enumerate(lines):
        if i not in drop:
            out.append(line)
        if i in stubs:
            out.append(stubs[i])
    text = "\n".join(_collapse_blank_lines(out)) + "\n"
    # The module docstring: the learner's version.
    first = source.splitlines()[0]
    if not first.startswith('"""'):
        raise ValueError("housing.py must start with its docstring")
    text = text.replace(first, '"""' + STARTER_DOC_FIRST, 1)
    doc_end = text.index('"""', 3)
    text = text[:doc_end] + "\n" + STARTER_DOC_END + text[doc_end:]
    defined = set(re.findall(r"(?m)^\s*def (\w+)", text))
    missing = [name for name in TODO_STEPS if name.split(".")[-1] not in defined]
    if missing:
        raise ValueError(f"functions missing from the starter module: {sorted(missing)}")
    return text


# ---------------------------------------------------------------------------
# Notebook cells common to both versions
# ---------------------------------------------------------------------------
PROJECT_STARTER = r'''# The project folder: your copy (mon_travail/projets/...), where data.py, housing.py and test_housing.py live
PROJECT_NAME = "partie_2_california_validation"
STARTER_KIT = ROOT / "projets" / PROJECT_NAME / "depart"
candidates = [Path.cwd(), ROOT / "mon_travail" / "projets" / PROJECT_NAME, STARTER_KIT]
PROJECT = next(folder for folder in candidates if (folder / "housing.py").exists() and (folder / "data.py").exists())
FIGURES = PROJECT / "figures"
MYLEARN_SOURCE = "learner (reference modules for the missing ones)"
if PROJECT.resolve() == STARTER_KIT.resolve():
    print("⚠️ Ce dossier est le kit de départ du dépôt, pas ta copie : lance python tools/start_chapter.py CP2, puis "
          "ouvre mon_travail/projets/partie_2_california_validation/mp2_california.ipynb.")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import data  # noqa: E402

try:
    import housing  # noqa: E402   (your module: it imports your mylearn library)
except ModuleNotFoundError as exc:
    housing = None
    print(f"⚠️ {exc} : lance d'abord python tools/start_chapter.py CP2, qui crée ta librairie mylearn.")
except Exception as exc:                       # a syntax error in housing.py must not stop the notebook
    housing = None
    print(f"⚠️ housing.py ne s'importe pas : {type(exc).__name__}: {exc}. Corrige-le ; l'étape MP2.1 le rechargera.")
print("Dossier du projet :", PROJECT)'''

PROJECT_SOLUTION = r'''# The project folder of the reference solution
PROJECT_NAME = "partie_2_california_validation"
PROJECT = ROOT / "projets" / PROJECT_NAME / "solution"
FIGURES = PROJECT / "figures"
MYLEARN_SOURCE = "reference"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import data  # noqa: E402
import housing  # noqa: E402

print("Dossier du projet :", PROJECT.relative_to(ROOT))'''

TOOLS = r'''# Tools of the notebook (run this cell)
import importlib
import json
import platform
import re
from collections import Counter
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_FOLDS = 5                                    # folds of every decision (MP2.2 to MP2.4) and of the residuals
N_FOLDS_DIAGNOSTIC = 3 if FAST_MODE else 5     # folds of the learning curves and of the variance (MP2.5)
TEST_SEED, FOLD_SEED, ZONE_SEED = 2026, 0, 0   # seeds of the test set, of the folds, of the k-means of the zones
DEGREES = [1, 2, 3, 4]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]    # Ridge (MP2.3)
LASSO_ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1]            # Lasso, degree 2 (MP2.3)
ZONES = [0, 2, 4, 8, 16, 32, 64, 128, 256]                # numbers of zones (MP2.4)

FAILED = {}         # check cell -> its guardrails that failed at its last run (the vault stays closed while any remain)
_CURRENT = [None]


def checks_of(cell):
    """Start the guardrails of one check cell: its failures of a previous run are forgotten."""
    _CURRENT[0] = cell
    FAILED[cell] = []


def verdict(step, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message (remembered until the cell runs again)."""
    print(f"✅ {step} : {success}" if ok else f"❌ {step} : {failure}")
    if not ok and _CURRENT[0] is not None:
        FAILED[_CURRENT[0]].append(failure)
    return bool(ok)


def filled(*values):
    """True when none of the values is still `...` (or None)."""
    return all(value is not ... and value is not None for value in values)


def ready(*names):
    """True when the variables of the earlier steps exist; otherwise print which ones are missing."""
    missing = [name for name in names if name not in globals()]
    if missing:
        print(f"⏳ il manque {', '.join(missing)} : fais d'abord les étapes précédentes.")
    return not missing


def fmt_int(n):
    """1234567 -> "1 234 567" (the French thousands separator, a narrow space)."""
    return f"{n:,}".replace(",", " ")


def reload_housing():
    """Import your latest saved housing.py again (after each change of the file)."""
    global housing
    current = globals().get("housing")
    housing = importlib.reload(current) if current is not None else importlib.import_module("housing")
    return housing


def housing_ok():
    """Reload your housing.py; False, with the reason, when it cannot be imported yet (a syntax error...)."""
    try:
        reload_housing()
    except Exception as exc:
        print(f"⚠️ housing.py ne s'importe pas : {type(exc).__name__}: {exc}")
        return False
    return True


def as_result(train_rmse, val_rmse):
    """The scores of one model, fold by fold, as stored in cv_results (and in results.json)."""
    return {"train_rmse": np.asarray(train_rmse, dtype=float), "val_rmse": np.asarray(val_rmse, dtype=float)}


def score_table(results):
    """One row per model: mean and standard deviation (ddof=1) of the validation RMSE, mean training RMSE."""
    rows = {name: {"RMSE validation": np.mean(r["val_rmse"]), "écart-type (folds)": np.std(r["val_rmse"], ddof=1),
                   "RMSE entraînement": np.mean(r["train_rmse"])} for name, r in results.items()}
    return pd.DataFrame(rows).T.round(4)


def canonical(params):
    """The hyperparameters, sorted and typed the same way (alpha=300 and alpha=300.0 are the same model); the
    default bounds and the seed of the zones are left out (FINAL may give them or not)."""
    kinds = {"degree": int, "n_zones": int, "alpha": float, "penalty": str}
    defaults = {"random_state": ZONE_SEED, "clip": (1.0, 99.0)}

    def is_default(key, value):
        try:
            return key in defaults and np.array_equal(np.asarray(value, dtype=float), np.asarray(defaults[key], dtype=float))
        except (TypeError, ValueError):
            return False

    kept = {key: value for key, value in params.items() if not is_default(key, value)}
    return {key: kinds[key](value) if key in kinds else value for key, value in sorted(kept.items())}


def column_names(degree):
    """Names of the polynomial features of data.FEATURES, in the order of mylearn.linear.polynomial_features."""
    powers = {1: "", 2: "²", 3: "³", 4: "⁴"}
    names = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(len(data.FEATURES)), d):
            names.append("·".join(data.FEATURES[i] + powers[c] for i, c in sorted(Counter(combo).items())))
    return names


def git(path, *args):
    """Run a git command in the folder of `path` (None when git cannot be found)."""
    try:
        return subprocess.run(["git", *args], cwd=path.parent, capture_output=True, text=True)
    except OSError:
        return None


def committed(path):
    """(True, "commit abc1234 du 2026-10-03") when the file is committed and unchanged; (False or None, why) otherwise."""
    log = git(path, "log", "-1", "--format=%h du %as", "--", path.name)
    status = git(path, "status", "--porcelain", "--", path.name)
    if log is None:
        return None, "ne peut pas être vérifié : git est introuvable"
    if log.returncode != 0:
        return None, "ne peut pas être vérifié : le dossier n'est pas dans un dépôt git"
    if not log.stdout.strip():
        return False, "n'est pas encore commité"
    if status.stdout.strip():
        return False, f"a changé depuis son commit ({log.stdout.strip()})"
    return True, f"commit {log.stdout.strip()}"


def committed_alone(path):
    """(True, why) when the first commit of the file holds that file only; (False or None, why) otherwise."""
    added = git(path, "log", "--diff-filter=A", "--format=%h", "--", path.name)
    if added is None or added.returncode != 0 or not added.stdout.split():
        return None, "n'a pas de commit"
    first = added.stdout.split()[-1]                       # the oldest commit that added the file
    files = [line for line in git(path, "show", "--name-only", "--format=", first).stdout.splitlines() if line.strip()]
    return len(files) == 1, f"commit {first}, {len(files)} fichier(s)"


def commit_file(path, message):
    """Commit this one file with git, from the notebook (with a default identity when git has none)."""
    top = git(path, "rev-parse", "--show-toplevel")
    if top is None or top.returncode != 0:
        print("⚠️ Ce dossier n'est pas dans un dépôt git : rien n'est commité.")
        return False
    who = [] if git(path, "config", "user.email").returncode == 0 else ["-c", "user.name=Workbook",
                                                                       "-c", "user.email=workbook@localhost"]
    git(path, "add", "--", path.name)
    result = git(path, *who, "commit", "-m", message, "--", path.name)
    print((result.stdout or result.stderr).strip())
    return result.returncode == 0


def save(fig, name):
    """Save a figure in the figures/ folder of the project (the README shows it)."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=110, bbox_inches="tight")


def run_project_tests():
    """Run the tests of the project folder (test_housing.py) and print the failures; return pytest's summary line."""
    command = [sys.executable, "-m", "pytest", str(PROJECT / "test_housing.py"), "-q", "-p", "no:cacheprovider",
               "--color=no", "-rfE", "--tb=no"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "1000"})        # long lines: the reason of each failure
    lines = result.stdout.strip().splitlines()
    for line in [line for line in lines if line.startswith(("FAILED ", "ERROR "))][:8]:
        kind, _, rest = line.partition(" ")
        name, _, reason = rest.partition(" - ")
        where = " (erreur hors du test : import, fixture…)" if kind == "ERROR" else ""
        print(f"❌ {name.split('::')[-1]}{where}" + (f"\n   {reason[:600]}" if reason else ""))
    if any(line.startswith("ERROR ") and " - " not in line for line in lines):     # a collection error: why?
        detail = subprocess.run(command[:-1] + ["--tb=short"], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "COLUMNS": "1000"}).stdout.splitlines()
        for line in [line for line in detail if line.startswith("E ")][:6]:
            print("  ", line[1:].strip()[:300])
    summary = lines[-1] if lines else result.stderr.strip()[-300:]
    print("pytest:", summary)
    return summary'''

# ---------------------------------------------------------------------------
# MP2.1
# ---------------------------------------------------------------------------
MP21_MD = r'''## MP2.1 · Cadrer le projet, geler le jeu de test, explorer l'entraînement ⏱️ 60 min

**Objectif :** mettre les districts de test hors de portée avant de regarder les données, puis explorer les autres.

a) **Cadre le problème** dans ton `README.md` (section « Le problème ») : qui utiliserait ces prédictions et pour quelle décision, ce que vaut une erreur de 0,5 (la cible est en centaines de milliers de dollars), et la mesure que tu suivras : la RMSE, dans l'unité de la cible.
b) Écris `make_test_indices` dans `housing.py` (lis sa docstring). La vérification enregistre les indices dans `test_indices.npy`, **une seule fois** : si le fichier existe, elle le relit et ne l'écrase jamais.
c) **Commite ce fichier tout de suite, seul**, avant toute modélisation : la date de ce commit montre que le découpage a été fixé avant tes résultats (le coffre et `vault.json` garderont ensuite la trace de l'ouverture du test). Depuis la racine du dépôt :

```bash
git add mon_travail/projets/partie_2_california_validation/test_indices.npy
git commit -m "MP2: freeze the test set" -- mon_travail/projets/partie_2_california_validation/test_indices.npy
```

Sans terminal (sur Colab, par exemple), passe `COMMIT` à `True` dans la cellule qui suit la vérification, exécute-la une fois, puis remets `False`.

d) Le coffre (`data.TestVault`) garde les districts de test ; `vault.train_part()` rend les autres, les seuls que tu regardes jusqu'à MP2.6. La cellule d'exploration les décrit et trace leur carte. Dans la cellule ✍️ : quelle part des valeurs est plafonnée, et que fait ce plafond aux erreurs ? Quelles mesures ont des valeurs extrêmes ? Où sont les logements chers ?'''

MP21_CHECK = r'''X_all, y_all = data.load_housing()
print(f"{fmt_int(len(y_all))} districts, {X_all.shape[1]} mesures : {', '.join(data.FEATURES)}")
INDICES = PROJECT / "test_indices.npy"
checks_of("MP2.1")
if housing_ok():
    with wb.attempt("MP2.1"):
        test_indices = np.asarray(housing.make_test_indices(len(y_all), test_size=0.2, seed=TEST_SEED))
        n_test = int(np.ceil(0.2 * len(y_all)))
        checks = [
            verdict("MP2.1", test_indices.shape == (n_test,), f"{fmt_int(n_test)} districts de test (20 %).",
                    f"il faut ceil(0,2 × n) = {n_test} indices de test, pas un tableau de forme {test_indices.shape}."),
            verdict("MP2.1", test_indices.size > 0 and bool(np.all(np.diff(test_indices) > 0)) and test_indices.min() >= 0
                    and test_indices.max() < len(y_all), "des indices triés, sans répétition, entre 0 et n − 1.",
                    "les indices doivent être triés dans l'ordre croissant, sans répétition, entre 0 et n − 1."),
            verdict("MP2.1", np.array_equal(np.asarray(housing.make_test_indices(len(y_all), 0.2, TEST_SEED)), test_indices),
                    "même graine, mêmes indices.", "deux appels avec la même graine donnent des indices différents : "
                    "crée le générateur dans la fonction, à partir de la graine."),
            verdict("MP2.1", not np.array_equal(np.asarray(housing.make_test_indices(len(y_all), 0.2, TEST_SEED + 1)),
                                                test_indices),
                    "une autre graine donne un autre tirage.", "une autre graine donne les mêmes indices : la graine "
                    "n'est pas utilisée."),
            verdict("MP2.1", np.array_equal(test_indices,
                                            np.sort(np.random.default_rng(TEST_SEED).permutation(len(y_all))[:n_test])),
                    "le tirage est celui de la docstring (le même pour tout le monde, et pour les tests).",
                    "les indices doivent être les ceil(0,2 × n) premiers de np.random.default_rng(seed).permutation(n), "
                    "triés (la docstring) : les tests de test_housing.py s'appuient sur ce découpage."),
        ]
        if INDICES.exists():
            print("✅ MP2.1 : test_indices.npy existe déjà et n'a pas changé : il n'est jamais réécrit."
                  if np.array_equal(np.load(INDICES), test_indices) else
                  "⚠️ MP2.1 : test_indices.npy existe déjà et diffère de make_test_indices ; il n'est pas réécrit. "
                  "C'est lui qui fait foi (le coffre le lit) : explique l'écart dans ton README.")
        elif all(checks):
            np.save(INDICES, test_indices)
            print(f"💾 {INDICES.name} est écrit dans le dossier du projet : commite-le maintenant (consigne c).")
        else:
            print("⏳ MP2.1 : test_indices.npy n'est pas écrit tant qu'une vérification échoue.")
if INDICES.exists():
    state, detail = committed(INDICES)
    print(f"✅ MP2.1 : test_indices.npy est commité ({detail})." if state else
          f"⏳ MP2.1 : test_indices.npy {detail} : commite-le avant d'aller plus loin (consigne c).")
    vault = data.TestVault(PROJECT, X_all, y_all)
    X_train, y_train = vault.train_part()
    print(f"Le coffre garde {fmt_int(len(vault.test_indices))} districts de test ; "
          f"il reste {fmt_int(len(y_train))} districts d'entraînement.")'''

MP21_COMMIT = r'''# Without a terminal (on Colab, for example): set COMMIT to True, run this cell once, then set it back to False
COMMIT = False
if COMMIT and not INDICES.exists():
    print("⏳ MP2.1 : pas encore de test_indices.npy à commiter (consigne b).")
elif COMMIT:
    commit_file(INDICES, "MP2: freeze the test set")
    state, detail = committed(INDICES)
    print(f"✅ MP2.1 : test_indices.npy est commité ({detail})." if state else f"⏳ MP2.1 : test_indices.npy {detail}.")'''

MP21_EXPLORE = r'''if ready("X_train", "y_train"):
    frame = pd.DataFrame(X_train, columns=data.FEATURES).assign(**{data.TARGET: y_train})
    print(frame.describe(percentiles=[0.01, 0.5, 0.99]).T.round(2).to_string())
    capped = y_train >= data.CAP
    print(f"\nValeurs plafonnées (cible ≥ {data.CAP}) : {fmt_int(capped.sum())} districts sur "
          f"{fmt_int(len(y_train))} ({capped.mean():.1%}).")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    points = axes[0].scatter(X_train[:, 7], X_train[:, 6], c=y_train, s=2, cmap="viridis")
    fig.colorbar(points, ax=axes[0], label="median house value (100 000 $)")
    axes[0].set_aspect(1 / np.cos(np.radians(36)))       # one degree of longitude is shorter than one of latitude
    axes[0].set(xlabel="longitude", ylabel="latitude", title="Training districts")
    axes[1].hist(y_train, bins=50)
    axes[1].set(xlabel="median house value (100 000 $)", ylabel="districts", title="The target (capped at 5)")
    plt.show()'''

# ---------------------------------------------------------------------------
# MP2.2
# ---------------------------------------------------------------------------
MP22_MD = r'''## MP2.2 · Une référence naïve, puis les moindres carrés en validation croisée ⏱️ 60 min

**Objectif :** un premier score honnête, avec son incertitude, et le chiffre à battre.

a) Dans `housing.py`, écris `rmse`, puis `HousingModel` pour `degree=1` et `penalty="none"` : `fit` (les bornes, les z-scores, puis `mylearn.linear.LinearRegression`), `transform` et `predict` ; enfin `cross_validate`. Tout ce qui s'apprend des données s'apprend dans `fit`, sur les lignes qu'il reçoit : en validation croisée, chaque fold réapprend ses bornes et ses z-scores sur ses propres lignes d'entraînement.
b) Les folds sont donnés : 5 folds tirés une fois pour toutes (graine `FOLD_SEED`), **les mêmes pour toutes les comparaisons**. Deux modèles mesurés sur les mêmes folds se comparent fold par fold (une comparaison appariée), ce qui rend leurs écarts plus fiables.
c) Écris `mean_baseline(y, folds)` ci-dessous : la RMSE de validation de chaque fold quand on prédit la moyenne des valeurs d'entraînement du fold.
d) La vérification compare tes moindres carrés à ceux de scikit-learn sur un fold, puis affiche les scores de la référence et des moindres carrés, sans bornes (`clip=None`) et avec. Dans la cellule ✍️ : combien gagne-t-on sur la référence ? Que changent les bornes, et pourquoi autant ? Que dit l'écart-type entre les folds ?'''

MP22_FOLDS = r'''if ready("X_train", "y_train"):
    folds = mylearn.model_selection.kfold_indices(len(y_train), N_FOLDS, shuffle=True,
                                                  rng=np.random.default_rng(FOLD_SEED))
    print("districts de validation par fold :", [len(val) for _, val in folds])'''

MP22_TODO = r'''def mean_baseline(y, folds):
    """Validation RMSE of each fold (an array, one value per fold) when every district of the validation part is
    predicted by the mean of the training values of the fold."""
    raise NotImplementedError  # TODO MP2.2 c)'''

MP22_SOLUTION = r'''def mean_baseline(y, folds):
    """Validation RMSE of each fold (an array, one value per fold) when every district of the validation part is
    predicted by the mean of the training values of the fold."""
    y = np.asarray(y, dtype=float)
    return np.array([housing.rmse(y[val], np.full(len(val), y[train].mean())) for train, val in folds])'''

MP22_CHECK = r'''if ready("X_train", "y_train", "folds") and housing_ok():
    checks_of("MP2.2")
    cv_results = globals().get("cv_results") or {}      # kept when this cell runs again: the later steps add to it
    with wb.attempt("MP2.2"):
        baseline = np.asarray(mean_baseline(y_train, folds), dtype=float)
        expected = [np.sqrt(np.mean((y_train[val] - y_train[train].mean()) ** 2)) for train, val in folds]
        verdict("MP2.2", baseline.shape == (N_FOLDS,) and np.allclose(baseline, expected),
                f"la référence naïve : une RMSE par fold, {baseline.mean():.4f} en moyenne.",
                "mean_baseline doit rendre une RMSE par fold, en prédisant la moyenne des valeurs d'entraînement du "
                "fold (pas celle de tout y_train, qui contient les districts de validation).")
        cv_results["référence : la moyenne"] = as_result([np.std(y_train[train]) for train, _ in folds], baseline)
    with wb.attempt("MP2.2"):
        from sklearn.linear_model import LinearRegression as SkLinearRegression    # only to check, never to build
        train, val = folds[0]
        raw = housing.HousingModel(clip=None).fit(X_train[train], y_train[train])
        oracle = SkLinearRegression().fit(X_train[train], y_train[train]).predict(X_train[val])
        gap = float(np.abs(np.asarray(raw.predict(X_train[val])) - oracle).max())
        verdict("MP2.2", gap < 1e-6, "sans bornes, tes moindres carrés donnent les prédictions de scikit-learn.",
                f"sans bornes, tes prédictions s'écartent de celles de LinearRegression de scikit-learn (écart maximal "
                f"{gap:.3g}) : standardiser ne change pas les moindres carrés, l'écart vient d'ailleurs.")
        bounded = housing.HousingModel().fit(X_train[train], y_train[train])
        low, high = np.percentile(X_train[train], [1.0, 99.0], axis=0)
        verdict("MP2.2", np.allclose(bounded.low_, low) and np.allclose(bounded.high_, high),
                "les bornes sont les percentiles 1 et 99 des lignes d'entraînement du fold.",
                "low_ et high_ doivent être les percentiles 1 et 99 de chaque colonne des lignes reçues par fit.")
        some = val[:6]
        alone = [np.asarray(bounded.predict(X_train[i:i + 1]))[0] for i in some]
        verdict("MP2.2", np.allclose(bounded.predict(X_train[some]), alone),
                "la prédiction d'un district ne dépend pas des autres districts prédits avec lui.",
                "la prédiction d'un district change selon les districts prédits avec lui : transform doit réutiliser "
                "les statistiques apprises par fit, jamais les recalculer.")
        fresh = housing.HousingModel()
        before = dict(vars(fresh))
        scores = housing.cross_validate(fresh, X_train, y_train, folds)
        after = vars(fresh)
        untouched = after.keys() == before.keys() and all(after[key] is before[key] for key in before)
        verdict("MP2.2", len(scores["val_rmse"]) == len(scores["train_rmse"]) == N_FOLDS and untouched,
                "cross_validate rend deux RMSE par fold et laisse intact le modèle reçu.",
                "cross_validate doit rendre une RMSE d'entraînement et une de validation par fold, en entraînant un "
                "clone du modèle reçu dans chaque fold (mylearn.model_selection.clone), jamais le modèle lui-même.")
        oracle_scores = []
        for train, val in folds:                       # the same model, learnt by scikit-learn inside each fold
            low, high = np.percentile(X_train[train], [1.0, 99.0], axis=0)
            twin = SkLinearRegression().fit(np.clip(X_train[train], low, high), y_train[train])
            residuals = y_train[val] - twin.predict(np.clip(X_train[val], low, high))
            oracle_scores.append(float(np.sqrt(np.mean(residuals ** 2))))
        verdict("MP2.2", np.allclose(scores["val_rmse"], oracle_scores, atol=1e-6),
                "fold par fold, ce sont les RMSE d'un modèle appris sur les seules lignes d'entraînement du fold.",
                f"un modèle appris sur les seules lignes d'entraînement de chaque fold, bornes comprises, donne les RMSE "
                f"de validation {np.round(oracle_scores, 4).tolist()} : cross_validate doit tout réapprendre dans "
                f"chaque fold, le prétraitement compris.")
        no_bounds = housing.cross_validate(housing.HousingModel(clip=None), X_train, y_train, folds)
        cv_results["moindres carrés, sans bornes"] = as_result(no_bounds["train_rmse"], no_bounds["val_rmse"])
        cv_results["moindres carrés, bornes 1-99 %"] = as_result(scores["train_rmse"], scores["val_rmse"])
        print(score_table(cv_results).to_string())
        print("\nRMSE de validation, fold par fold :")
        print(pd.DataFrame({name: r["val_rmse"] for name, r in cv_results.items()},
                           index=pd.Index(range(N_FOLDS), name="fold")).T.round(4).to_string())'''

# ---------------------------------------------------------------------------
# MP2.3
# ---------------------------------------------------------------------------
MP23_MD = r'''## MP2.3 · Features polynomiales, Ridge et Lasso : le degré et α choisis par validation croisée ⏱️ 120 min

**Objectif :** régler la capacité du modèle (le degré) et sa régularisation (α) sur les folds, et dire avec quelle incertitude.

a) Complète `HousingModel.fit` pour `degree > 1` (`mylearn.linear.polynomial_features` des mesures bornées et standardisées, puis un **second z-score**, de toutes les colonnes, appris lui aussi sur les lignes d'entraînement) et pour `penalty="ridge"` ou `"lasso"` (α passé tel quel) ; puis écris `validation_curve` (docstring).
b) La cellule suivante vérifie ta chaîne contre scikit-learn, puis trace les **courbes de validation** de Ridge : la RMSE d'entraînement et de validation selon α, pour les degrés 1 à 4 (≈ 30 s). Elle les enregistre dans `figures/courbes_validation.png`.
c) Le meilleur score n'est pas forcément le bon choix : son avance peut n'être que du bruit. Écris `pick_simplest` (docstring) : la **règle d'une erreur type**, qui prend le modèle le plus simple dont le score ne dépasse pas le meilleur de plus d'une erreur type. L'erreur type du score moyen se prend ici égale à $\sigma/\sqrt{k}$, l'écart-type des $k$ folds divisé par $\sqrt{k}$ : une approximation optimiste, puisque les folds partagent leurs données (CP2.1 d), mais un ordre de grandeur utile. Les candidats Ridge sont rangés du plus simple au plus complexe : le degré croissant, puis α décroissant (plus α est grand, plus le modèle est contraint).
d) La dernière cellule mesure le Lasso au degré 2 (≈ 20 s) : son score selon α, et le nombre de colonnes qu'il garde sur les 44.

Dans la cellule ✍️ : où le modèle sous-apprend-il, où surapprend-il ? Quel modèle la règle retient-elle ? Est-ce celui du meilleur score, et pourquoi ? Pourquoi $\sigma/\sqrt{k}$ est-elle optimiste ? Le Lasso fait-il mieux que Ridge ici ? À quoi d'autre peut-il servir ?'''

MP23_CURVES = r'''if ready("X_train", "y_train", "folds") and housing_ok():
    checks_of("MP2.3 courbes")
    with wb.attempt("MP2.3"):
        from sklearn.linear_model import Ridge as SkRidge                       # only to check, never to build
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        def sklearn_chain(train, val, degree, alpha):
            """The same chain learnt by scikit-learn on the training rows of a fold: (validation RMSE, predictions)."""
            low, high = np.percentile(X_train[train], [1.0, 99.0], axis=0)
            first = StandardScaler().fit(np.clip(X_train[train], low, high))
            poly = PolynomialFeatures(degree, include_bias=False)
            Z = poly.fit_transform(first.transform(np.clip(X_train[train], low, high)))
            second = StandardScaler().fit(Z)
            twin = SkRidge(alpha=alpha).fit(second.transform(Z), y_train[train])
            predicted = twin.predict(second.transform(poly.transform(first.transform(np.clip(X_train[val], low, high)))))
            return float(np.sqrt(np.mean((y_train[val] - predicted) ** 2))), predicted, Z

        train, val = folds[0]
        model = housing.HousingModel(degree=2, penalty="ridge", alpha=10.0).fit(X_train[train], y_train[train])
        _, oracle, Z = sklearn_chain(train, val, 2, 10.0)
        gap = float(np.abs(np.asarray(model.predict(X_train[val])) - oracle).max())
        verdict("MP2.3", gap < 1e-6, "degré 2 + Ridge : les prédictions de la même chaîne construite avec scikit-learn.",
                f"degré 2 + Ridge : écart maximal {gap:.3g} avec la chaîne de scikit-learn (bornes, StandardScaler, "
                f"PolynomialFeatures, StandardScaler, Ridge) : vérifie l'ordre des étapes, le second z-score, et les "
                f"écarts-types (ddof=0, comme StandardScaler).")
        verdict("MP2.3", np.allclose(model.design_mean_, Z.mean(axis=0)),
                "le second z-score est appris sur les lignes d'entraînement du fold.",
                "design_mean_ doit être la moyenne des colonnes polynomiales des lignes reçues par fit (calculées avec "
                "les bornes et le premier z-score de ces lignes, écarts-types avec ddof=0).")
        ridge_curves = {}
        for degree in DEGREES:
            curve = housing.validation_curve(housing.HousingModel(degree=degree, penalty="ridge"), "alpha", ALPHAS,
                                             X_train, y_train, folds)
            ridge_curves[degree] = tuple(np.asarray(part, dtype=float) for part in curve)
        verdict("MP2.3", all(part.shape == (len(ALPHAS), N_FOLDS) for pair in ridge_curves.values() for part in pair),
                f"validation_curve rend deux tableaux ({len(ALPHAS)}, {N_FOLDS}) : une ligne par valeur de alpha.",
                f"validation_curve doit rendre deux tableaux de forme ({len(ALPHAS)}, {N_FOLDS}).")
        oracle_scores = [sklearn_chain(train, val, 2, 10.0)[0] for train, val in folds]
        verdict("MP2.3", np.allclose(ridge_curves[2][1][ALPHAS.index(10.0)], oracle_scores, atol=1e-6),
                "au degré 2 et pour alpha = 10, les cinq RMSE de validation sont celles de la chaîne de scikit-learn "
                "apprise dans chaque fold.",
                f"au degré 2 et pour alpha = 10, la chaîne de scikit-learn apprise dans chaque fold donne "
                f"{np.round(oracle_scores, 4).tolist()} : validation_curve doit mesurer chaque valeur avec "
                f"cross_validate, sur les mêmes folds.")
        print("RMSE de validation (moyenne des 5 folds) :")
        print(pd.DataFrame({f"degré {d}": val.mean(axis=1) for d, (_, val) in ridge_curves.items()},
                           index=pd.Index(ALPHAS, name="alpha")).round(4).to_string())
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for degree, (train_rmse, val_rmse) in ridge_curves.items():
            mean, std = val_rmse.mean(axis=1), val_rmse.std(axis=1, ddof=1)
            line, = ax.plot(ALPHAS, mean, marker="o", label=f"degree {degree}, validation")
            ax.fill_between(ALPHAS, mean - std, mean + std, color=line.get_color(), alpha=0.15)
            ax.plot(ALPHAS, train_rmse.mean(axis=1), ls="--", color=line.get_color(), label=f"degree {degree}, training")
        ax.set_xscale("log")
        ax.set(xlabel="alpha (Ridge penalty)", ylabel="RMSE (100 000 $)", ylim=(0.5, 0.85),
               title="Ridge: validation curves (5 folds, band: ± 1 standard deviation)")
        ax.grid(True, alpha=0.4)
        ax.legend(ncol=2, fontsize=8)
        save(fig, "courbes_validation.png")
        plt.show()'''

MP23_TODO = r'''def pick_simplest(mean_rmse, std_rmse, n_folds):
    """Index of the candidate chosen by the one-standard-error rule.

    The candidates are sorted from the simplest to the most complex; mean_rmse and std_rmse are the mean and the
    standard deviation (ddof=1) of their validation RMSE over n_folds folds. The best candidate has the lowest mean;
    its standard error is its standard deviation divided by sqrt(n_folds). The rule returns the index of the first
    (simplest) candidate whose mean is at most the best mean plus that standard error.
    """
    raise NotImplementedError  # TODO MP2.3 c)'''

MP23_SOLUTION = r'''def pick_simplest(mean_rmse, std_rmse, n_folds):
    """Index of the candidate chosen by the one-standard-error rule.

    The candidates are sorted from the simplest to the most complex; mean_rmse and std_rmse are the mean and the
    standard deviation (ddof=1) of their validation RMSE over n_folds folds. The best candidate has the lowest mean;
    its standard error is its standard deviation divided by sqrt(n_folds). The rule returns the index of the first
    (simplest) candidate whose mean is at most the best mean plus that standard error.
    """
    mean_rmse, std_rmse = np.asarray(mean_rmse, dtype=float), np.asarray(std_rmse, dtype=float)
    best = int(np.argmin(mean_rmse))
    threshold = mean_rmse[best] + std_rmse[best] / np.sqrt(n_folds)
    return int(np.flatnonzero(mean_rmse <= threshold)[0])'''

MP23_CHOICE = r'''if ready("ridge_curves"):
    checks_of("MP2.3 choix")
    with wb.attempt("MP2.3"):
        verdict("MP2.3", pick_simplest([0.70, 0.61, 0.60, 0.62], [0.05, 0.05, 0.04, 0.05], 4) == 1,
                "pick_simplest prend le premier candidat assez proche du meilleur.",
                "pick_simplest([0.70, 0.61, 0.60, 0.62], [0.05, 0.05, 0.04, 0.05], 4) doit rendre 1 : le meilleur "
                "fait 0.60 avec une erreur type de 0.04 / √4 = 0.02, et 0.61 est le premier score ≤ 0.62.")
        verdict("MP2.3", pick_simplest([0.70, 0.635, 0.60], [0.05, 0.05, 0.04], 4) == 2,
                "l'erreur type est l'écart-type du meilleur divisé par √k.",
                "pick_simplest([0.70, 0.635, 0.60], [0.05, 0.05, 0.04], 4) doit rendre 2 : la marge est l'erreur type "
                "0.04 / √4 = 0.02, pas l'écart-type 0.04.")
        candidates = [(d, a) for d in DEGREES for a in sorted(ALPHAS, reverse=True)]     # the simplest first
        means = np.array([ridge_curves[d][1][ALPHAS.index(a)].mean() for d, a in candidates])
        stds = np.array([ridge_curves[d][1][ALPHAS.index(a)].std(ddof=1) for d, a in candidates])
        best, pick = int(np.argmin(means)), pick_simplest(means, stds, N_FOLDS)
        chosen_ridge = {"degree": candidates[pick][0], "alpha": candidates[pick][1]}
        print(f"le meilleur score : degré {candidates[best][0]}, alpha = {candidates[best][1]:g}, "
              f"RMSE {means[best]:.4f} ± {stds[best]:.4f} (erreur type {stds[best] / np.sqrt(N_FOLDS):.4f})")
        print(f"la règle d'une erreur type retient : degré {chosen_ridge['degree']}, alpha = {chosen_ridge['alpha']:g}, "
              f"RMSE {means[pick]:.4f} ± {stds[pick]:.4f}")
        d, a = chosen_ridge["degree"], chosen_ridge["alpha"]
        cv_results[f"Ridge, degré {d}, alpha = {a:g}"] = as_result(*(part[ALPHAS.index(a)] for part in ridge_curves[d]))
        print()
        print(score_table(cv_results).to_string())'''

MP23_LASSO = r'''if ready("X_train", "y_train", "folds", "ridge_curves") and housing_ok():
    checks_of("MP2.3 lasso")
    with wb.attempt("MP2.3"):
        from sklearn.linear_model import Lasso as SkLasso                       # only to check, never to build
        train, val = folds[0]
        model = housing.HousingModel(degree=2, penalty="lasso", alpha=0.01).fit(X_train[train], y_train[train])
        design = np.asarray(model.transform(X_train[train]))
        oracle = SkLasso(alpha=0.01, tol=1e-10, max_iter=100_000).fit(design, y_train[train])
        gap = float(np.abs(np.asarray(model.predict(X_train[val])) - oracle.predict(model.transform(X_train[val]))).max())
        verdict("MP2.3", gap < 1e-3, "le Lasso donne les prédictions de celui de scikit-learn sur la même matrice.",
                f"le Lasso s'écarte de celui de scikit-learn sur la même matrice (écart maximal {gap:.3g}) : "
                + ("alpha est-il passé tel quel à Lasso, sans être multiplié ni divisé ?" if gap > 0.1 else
                   "la tolérance d'arrêt de la descente de coordonnées est-elle assez petite (tol=1e-6) ?"))
        lasso_train, lasso_val = (np.asarray(part, dtype=float) for part in housing.validation_curve(
            housing.HousingModel(degree=2, penalty="lasso"), "alpha", LASSO_ALPHAS, X_train, y_train, folds))
        names = column_names(2)
        kept = {}
        for alpha in LASSO_ALPHAS:
            weights = np.asarray(housing.HousingModel(degree=2, penalty="lasso", alpha=alpha)
                                 .fit(X_train, y_train).regressor_.coef_)
            kept[alpha] = [name for name, w in zip(names, weights) if w != 0]
        ridge2 = ridge_curves[2][1].mean(axis=1)
        print(f"Ridge, degré 2 : la meilleure RMSE de validation est {ridge2.min():.4f}")
        print(pd.DataFrame({"RMSE validation": lasso_val.mean(axis=1), "écart-type": lasso_val.std(axis=1, ddof=1),
                            "colonnes gardées (sur 44)": [len(kept[a]) for a in LASSO_ALPHAS]},
                           index=pd.Index(LASSO_ALPHAS, name="alpha (Lasso)")).round(4).to_string())
        strongest = LASSO_ALPHAS[-1]
        print(f"\nles {len(kept[strongest])} colonnes gardées avec alpha = {strongest:g} :", ", ".join(kept[strongest]))'''

# ---------------------------------------------------------------------------
# MP2.4
# ---------------------------------------------------------------------------
MP24_MD = r'''## MP2.4 · Des zones géographiques par k-means : la silhouette, puis la validation croisée ⏱️ 90 min

**Objectif :** donner au modèle linéaire une notion de quartier, et mesurer ce qu'elle rapporte.

Un modèle linéaire en latitude et longitude ne sait dire que « plus au nord, plus à l'ouest » ; même au degré 3, il ne voit pas qu'un district de la baie de San Francisco vaut plus que son voisin de l'intérieur. Des **zones** apprises par k-means sur les coordonnées, codées en colonnes 0/1 (une par zone), lui donnent un niveau par quartier.

a) Complète `HousingModel` pour `n_zones > 0` : dans `fit`, un k-means (`mylearn.cluster.KMeans`, un seul essai, graine `random_state`) sur la latitude et la longitude **brutes** (non bornées), standardisées avec leurs propres moyennes et écarts-types d'entraînement ; une colonne 0/1 par zone, ajoutée aux features polynomiales avant le second z-score ; puis la méthode `zones`.
b) **La silhouette** : la première cellule calcule `mylearn.cluster.silhouette_score` pour k = 2 à 12 (sur 2 000 districts tirés au hasard, pour le temps de calcul). Quel k préfère-t-elle ?
c) **La validation croisée** : la seconde cellule trace la courbe de validation selon `n_zones` ∈ {0, 2, 4, …, 256}, avec le degré et α retenus en MP2.3, et choisit le nombre de zones par ta règle d'une erreur type (le plus simple : le moins de zones) ; puis elle réajuste α pour ce nombre de zones, car des colonnes en plus peuvent changer la bonne pénalité (≈ 40 s). La figure va dans `figures/zones.png`.

Dans la cellule ✍️ : la silhouette et la validation croisée choisissent-elles le même k, et pourquoi ? Que représentent les zones pour le modèle ? Pourquoi le k-means doit-il être refait dans chaque fold ? Jusqu'où pousser le nombre de zones, et qu'est-ce qui limite ce choix ?'''

MP24_SILHOUETTE = r'''if ready("X_train"):
    geo = X_train[:, [6, 7]]                                    # latitude, longitude
    geo_z = (geo - geo.mean(axis=0)) / geo.std(axis=0)
    sample = np.random.default_rng(1).choice(len(geo_z), 2000, replace=False)
    silhouettes = {}
    for k in range(2, 13):
        labels = np.asarray(mylearn.cluster.KMeans(n_clusters=k, n_init=1, random_state=ZONE_SEED).fit(geo_z).labels_)
        silhouettes[k] = float(mylearn.cluster.silhouette_score(geo_z[sample], labels[sample]))
    print("silhouette (2 000 districts) :", {k: round(s, 3) for k, s in silhouettes.items()})
    print(f"la silhouette préfère k = {max(silhouettes, key=silhouettes.get)}")'''

MP24_ZONES = r'''if ready("X_train", "y_train", "folds", "chosen_ridge", "silhouettes") and housing_ok():
    checks_of("MP2.4")
    with wb.attempt("MP2.4"):
        train, val = folds[0]
        model = housing.HousingModel(n_zones=5, random_state=ZONE_SEED).fit(X_train[train], y_train[train])
        geo_train = X_train[train][:, [6, 7]]                    # raw coordinates, not bounded
        z_val = (X_train[val][:, [6, 7]] - geo_train.mean(axis=0)) / geo_train.std(axis=0)
        centres = np.asarray(model.kmeans_.cluster_centers_)
        nearest = ((z_val[:, None, :] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
        verdict("MP2.4", np.array_equal(np.asarray(model.zones(X_train[val])), nearest),
                "la zone d'un district est son centre k-means le plus proche, en coordonnées standardisées avec les "
                "statistiques d'entraînement.",
                "zones doit rendre le centre le plus proche de chaque district, la latitude et la longitude BRUTES "
                "(non bornées) standardisées avec leurs moyennes et écarts-types (ddof=0) sur les lignes reçues par fit.")
        verdict("MP2.4", np.asarray(model.transform(X_train[val[:3]])).shape == (3, 8 + 5),
                "transform ajoute une colonne par zone.", "avec 5 zones et le degré 1, transform doit rendre 8 + 5 colonnes.")
        zone_train, zone_val = (np.asarray(part, dtype=float) for part in housing.validation_curve(
            housing.HousingModel(penalty="ridge", random_state=ZONE_SEED, **chosen_ridge), "n_zones", ZONES,
            X_train, y_train, folds))
        means, stds = zone_val.mean(axis=1), zone_val.std(axis=1, ddof=1)
        chosen_zones = ZONES[pick_simplest(means, stds, N_FOLDS)]
        print(pd.DataFrame({"RMSE validation": means, "écart-type": stds, "RMSE entraînement": zone_train.mean(axis=1)},
                           index=pd.Index(ZONES, name="zones")).round(4).to_string())
        print(f"la règle d'une erreur type retient {chosen_zones} zones")
        grid = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
        alphas_zones = [a for a in grid if chosen_ridge["alpha"] / 10 <= a <= chosen_ridge["alpha"] * 10]
        alphas_zones = alphas_zones or [chosen_ridge["alpha"]]
        alpha_train, alpha_val = (np.asarray(part, dtype=float) for part in housing.validation_curve(
            housing.HousingModel(degree=chosen_ridge["degree"], penalty="ridge", n_zones=chosen_zones,
                                 random_state=ZONE_SEED), "alpha", alphas_zones, X_train, y_train, folds))
        order = np.argsort(alphas_zones)[::-1]                    # the simplest first: the largest alpha
        a_means, a_stds = alpha_val.mean(axis=1), alpha_val.std(axis=1, ddof=1)
        best_alpha = alphas_zones[order[pick_simplest(a_means[order], a_stds[order], N_FOLDS)]]
        print(pd.DataFrame({"RMSE validation": a_means, "écart-type": a_stds},
                           index=pd.Index(alphas_zones, name=f"alpha ({chosen_zones} zones)")).round(4).to_string())
        chosen = {"degree": chosen_ridge["degree"], "penalty": "ridge", "alpha": float(best_alpha), "n_zones": chosen_zones}
        i = alphas_zones.index(best_alpha)
        chosen_name = f"Ridge, degré {chosen['degree']}, alpha = {best_alpha:g}, {chosen_zones} zones"
        cv_results[chosen_name] = as_result(alpha_train[i], alpha_val[i])
        print(f"\nle modèle retenu : {chosen}")
        print(score_table(cv_results).to_string())
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
        axes[0].plot(list(silhouettes), list(silhouettes.values()), marker="o")
        axes[0].set(xlabel="number of clusters k", ylabel="silhouette", title="k-means on latitude and longitude")
        positions = np.arange(len(ZONES))
        axes[1].errorbar(positions, means, yerr=stds, marker="o", capsize=4, label="validation (± 1 std)")
        axes[1].plot(positions, zone_train.mean(axis=1), ls="--", marker=".", label="training")
        axes[1].set_xticks(positions, [str(k) for k in ZONES])
        axes[1].set(xlabel="number of zones (one-hot columns)", ylabel="RMSE (100 000 $)",
                    title=f"Ridge, degree {chosen_ridge['degree']}, alpha = {chosen_ridge['alpha']:g}: adding zones")
        for ax in axes:
            ax.grid(True, alpha=0.4)
        axes[1].legend()
        save(fig, "zones.png")
        plt.show()'''

# ---------------------------------------------------------------------------
# MP2.5
# ---------------------------------------------------------------------------
MP25_MD = r'''## MP2.5 · Diagnostiquer : courbes d'apprentissage, biais et variance, résidus par région ⏱️ 90 min

**Objectif :** comprendre ce qui limite le modèle retenu, et qui il dessert mal.

a) Écris `learning_curve` et `out_of_fold_predictions` dans `housing.py` (docstrings).
b) La première cellule trace les **courbes d'apprentissage** des moindres carrés (MP2.2) et du modèle retenu, de 500 districts d'entraînement à tous (`figures/courbes_apprentissage.png`), puis mesure la **variance** des prédictions des modèles appris sur les différents folds, sur les mêmes 2 000 districts (une mesure relative : ces modèles partagent une partie de leurs données). En mode rapide (`FAST_MODE`), cette cellule utilise 3 folds au lieu de 5 : les tailles sont plus petites, les conclusions les mêmes (≈ 10 s au lieu de 20 s).
c) Écris `residuals_by_group` ci-dessous (docstring).
d) La dernière cellule calcule les **prédictions hors fold** du modèle retenu (chaque district prédit par le modèle qui ne l'a pas vu), puis les résidus des districts plafonnés et ceux de 8 grandes régions (un k-means à 8 groupes, seulement pour le rapport). La carte des zones du modèle et des résidus va dans `figures/carte_zones_residus.png`.

Dans la cellule ✍️ : le modèle retenu souffre-t-il surtout de biais ou de variance ? Plus de districts l'aideraient-ils ? Qui est mal servi, et que faudrait-il dire à un utilisateur ?'''

MP25_CURVES = r'''if ready("X_train", "y_train", "chosen", "chosen_ridge") and housing_ok():
    checks_of("MP2.5 courbes")
    with wb.attempt("MP2.5"):
        diag_folds = folds if N_FOLDS_DIAGNOSTIC == N_FOLDS else mylearn.model_selection.kfold_indices(
            len(y_train), N_FOLDS_DIAGNOSTIC, shuffle=True, rng=np.random.default_rng(FOLD_SEED))
        smallest = min(len(train) for train, _ in diag_folds)
        sizes = [500, 1000, 2000, 4000, 8000, smallest]
        learners = {"least squares (degree 1)": housing.HousingModel(),
                    "chosen model": housing.HousingModel(random_state=ZONE_SEED, **chosen)}
        curves, intact = {}, True
        for name, model in learners.items():      # copies of the folds: a learning_curve that shuffles them in place
            copies = [(np.array(train), np.array(val)) for train, val in diag_folds]      # cannot change them here
            curves[name] = tuple(np.asarray(part, dtype=float)
                                 for part in housing.learning_curve(model, X_train, y_train, sizes, copies, seed=0))
            intact = intact and all(np.array_equal(mine, given) for pair, fold in zip(copies, diag_folds)
                                    for mine, given in zip(pair, fold))
        verdict("MP2.5", all(part.shape == (len(sizes), len(diag_folds)) for pair in curves.values() for part in pair),
                f"learning_curve rend deux tableaux ({len(sizes)}, {len(diag_folds)}) : une ligne par taille.",
                f"learning_curve doit rendre deux tableaux de forme ({len(sizes)}, {len(diag_folds)}).")
        verdict("MP2.5", intact, "learning_curve laisse intacts les folds qu'elle reçoit.",
                "learning_curve a modifié les folds qu'elle reçoit (un mélange en place ?) : la docstring demande de "
                "ne jamais les modifier, car tous les modèles du projet partagent les mêmes folds.")
        train0, val0 = diag_folds[0]
        rows = np.asarray(train0)[np.random.default_rng(0).permutation(len(train0))][:1000]
        alone = housing.rmse(y_train[val0], housing.HousingModel().fit(X_train[rows], y_train[rows]).predict(X_train[val0]))
        verdict("MP2.5", np.isclose(curves["least squares (degree 1)"][1][1, 0], alone),
                "chaque taille apprend sur les premières lignes d'entraînement du fold, mélangées par la graine.",
                "à la taille 1000, le fold 0 doit apprendre sur les 1000 premiers indices de "
                "train_idx[np.random.default_rng(seed).permutation(len(train_idx))], et être mesuré sur toute sa "
                "validation.")
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
        for ax, (name, (train_rmse, val_rmse)) in zip(axes, curves.items()):
            for scores, label, style in [(train_rmse, "training", "--"), (val_rmse, "validation", "-")]:
                mean, std = scores.mean(axis=1), scores.std(axis=1, ddof=1)
                ax.plot(sizes, mean, ls=style, marker="o", label=label)
                ax.fill_between(sizes, mean - std, mean + std, alpha=0.15)
            ax.set_xscale("log")
            ax.set(xlabel="training districts", title=f"Learning curves: {name}")
            ax.grid(True, alpha=0.4)
            ax.legend()
        axes[0].set_ylabel("RMSE (100 000 $)")
        save(fig, "courbes_apprentissage.png")
        plt.show()
        print(pd.DataFrame({f"{name}, {kind}": part.mean(axis=1) for name, pair in curves.items()
                            for kind, part in zip(("entraînement", "validation"), pair)},
                           index=pd.Index(sizes, name="districts")).round(4).to_string())
        probe = np.random.default_rng(5).choice(len(y_train), 2000, replace=False)
        spread = {}
        for name, params in [("degré 1", {}), ("degré 4, alpha = 0.01", {"degree": 4, "penalty": "ridge", "alpha": 0.01}),
                             (f"degré {chosen['degree']}, alpha = {chosen_ridge['alpha']:g}",
                              {"degree": chosen["degree"], "penalty": "ridge", "alpha": chosen_ridge["alpha"]}),
                             ("modèle retenu", {**chosen, "random_state": ZONE_SEED})]:
            predictions = np.array([np.asarray(housing.HousingModel(**params).fit(X_train[train], y_train[train])
                                               .predict(X_train[probe])) for train, _ in diag_folds])
            spread[name] = float(predictions.var(axis=0).mean())
        print("\nvariance des prédictions entre les modèles des folds (moyenne sur 2 000 districts) :")
        print({name: round(value, 4) for name, value in spread.items()})'''

MP25_TODO = r'''def residuals_by_group(y, predictions, groups):
    """One row per group, the groups sorted in the index, and four columns: "n" (number of districts),
    "mean_value" (mean of y), "rmse" and "mean_residual" (mean of y - prediction: > 0 when the model underestimates)."""
    raise NotImplementedError  # TODO MP2.5 c)'''

MP25_SOLUTION = r'''def residuals_by_group(y, predictions, groups):
    """One row per group, the groups sorted in the index, and four columns: "n" (number of districts),
    "mean_value" (mean of y), "rmse" and "mean_residual" (mean of y - prediction: > 0 when the model underestimates)."""
    y = np.asarray(y, dtype=float)
    frame = pd.DataFrame({"group": np.asarray(groups), "y": y, "residual": y - np.asarray(predictions, dtype=float)})
    grouped = frame.groupby("group")
    return pd.DataFrame({"n": grouped.size(), "mean_value": grouped["y"].mean(),
                         "rmse": grouped["residual"].apply(lambda r: float(np.sqrt(np.mean(r ** 2)))),
                         "mean_residual": grouped["residual"].mean()})'''

MP25_RESIDUALS = r'''if ready("X_train", "y_train", "folds", "chosen", "chosen_name", "cv_results") and housing_ok():
    checks_of("MP2.5 résidus")
    with wb.attempt("MP2.5"):
        oof = np.asarray(housing.out_of_fold_predictions(housing.HousingModel(random_state=ZONE_SEED, **chosen),
                                                         X_train, y_train, folds), dtype=float)
        verdict("MP2.5", np.allclose([housing.rmse(y_train[val], oof[val]) for _, val in folds],
                                     cv_results[chosen_name]["val_rmse"]),
                "fold par fold, les prédictions hors fold redonnent les RMSE de validation du modèle retenu.",
                "fold par fold, les prédictions hors fold devraient redonner les RMSE de validation du modèle retenu "
                "(MP2.4). Les causes possibles : out_of_fold_predictions n'utilise pas le modèle appris sur "
                "l'entraînement de chaque fold ; cross_validate ou validation_curve apprennent quelque chose hors du "
                "fold ; le modèle n'est pas reproductible (la graine du k-means) ; les folds ont été modifiés en place.")
        print(f"RMSE hors fold sur les {fmt_int(len(y_train))} districts : {housing.rmse(y_train, oof):.4f}")
        geo = X_train[:, [6, 7]]
        geo_z = (geo - geo.mean(axis=0)) / geo.std(axis=0)
        regions = np.asarray(mylearn.cluster.KMeans(n_clusters=8, random_state=ZONE_SEED).fit(geo_z).labels_)
        table = residuals_by_group(y_train, oof, regions)
        residual = y_train - oof
        expected = pd.DataFrame({"r": residual, "g": regions}).groupby("g")["r"]
        verdict("MP2.5", list(table.index) == sorted(set(regions)) and np.allclose(table["mean_residual"], expected.mean())
                and np.allclose(table["rmse"], expected.apply(lambda r: np.sqrt(np.mean(r ** 2))))
                and np.array_equal(table["n"], expected.size()),
                "residuals_by_group : une ligne par région, les bons n, RMSE et résidus moyens.",
                "residuals_by_group doit rendre, pour chaque groupe (trié), n, la valeur moyenne, la RMSE et le résidu "
                "moyen y − prédiction.")
        capped = y_train >= data.CAP
        table = table.assign(latitude=[X_train[regions == g, 6].mean() for g in table.index],
                             longitude=[X_train[regions == g, 7].mean() for g in table.index],
                             relative_rmse=table["rmse"] / table["mean_value"],
                             capped=[capped[regions == g].mean() for g in table.index])
        print("\nrésidus hors fold par région (résidu = vraie valeur − prédiction) :")
        print(table.sort_values("rmse", ascending=False).round(3).to_string())
        by_cap = residuals_by_group(y_train, oof, np.where(capped, "plafonnés", "autres"))
        print("\nles districts plafonnés et les autres :")
        print(by_cap.round(3).to_string())
        final_model = housing.HousingModel(random_state=ZONE_SEED, **chosen).fit(X_train, y_train)
        zones_all = np.asarray(final_model.zones(X_train))
        aspect = 1 / np.cos(np.radians(36))
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
        axes[0].scatter(X_train[:, 7], X_train[:, 6], c=zones_all % 20, cmap="tab20", s=2)
        axes[0].set(title=f"The {chosen['n_zones']} zones of the chosen model")
        points = axes[1].scatter(X_train[:, 7], X_train[:, 6], c=np.clip(residual, -1.5, 1.5), cmap="RdBu", s=2,
                                 vmin=-1.5, vmax=1.5)
        fig.colorbar(points, ax=axes[1], label="true − predicted (100 000 $)\nred: overestimated · blue: underestimated")
        axes[1].set(title="Out-of-fold residuals")
        for ax in axes:
            ax.set_aspect(aspect)
            ax.set(xlabel="longitude", ylabel="latitude")
        save(fig, "carte_zones_residus.png")
        plt.show()'''

# ---------------------------------------------------------------------------
# MP2.6
# ---------------------------------------------------------------------------
MP26_MD = r'''## MP2.6 · Ouvrir le coffre une seule fois : le test, et la comparaison avec scikit-learn ⏱️ 45 min

**Objectif :** le score honnête du modèle retenu, et une vérification indépendante de ta librairie.

a) Remplis `FINAL` : les hyperparamètres retenus en MP2.3 et MP2.4 (le dictionnaire `chosen`, sans `random_state` ni `clip`), **et rien d'autre** : c'est la décision que tu as prise sur les folds, avant de voir le test. Le coffre reste fermé tant que `FINAL` diffère de `chosen`, ou qu'un garde-fou des étapes MP2.1 à MP2.5 est encore en échec.
b) La vérification compare d'abord ta librairie à scikit-learn, sans toucher au test : le `Ridge` de scikit-learn sur ta matrice de design doit redonner tes prédictions ; une chaîne entière construite avec scikit-learn (`PolynomialFeatures`, `StandardScaler`, `KMeans`, `Ridge`) doit obtenir un score voisin en validation croisée (pas identique : son k-means ne tombe pas sur les mêmes zones).
c) Puis elle entraîne `FINAL` sur les 16 512 districts d'entraînement et **ouvre le coffre** (`vault.open`, inscrit dans `vault.json` avec la date) : la RMSE du test, son intervalle bootstrap à 95 % (`mylearn.stats.bootstrap_ci`), le $R^2$ et la référence naïve. Enfin, elle écrit `results.json` : hyperparamètres, scores par fold, score du test, graines et versions.

Dans la cellule ✍️ : le score du test tombe-t-il dans ce qu'annonçait la validation croisée ? Que ferais-tu s'il était nettement moins bon ? (Pas de nouvel essai sur le test : ce serait le transformer en jeu de validation.)'''

MP26_TODO = r'''FINAL = ...   # the hyperparameters chosen in MP2.3 and MP2.4 (the dict chosen, without random_state nor clip)'''

MP26_SOLUTION = r'''FINAL = {"degree": 3, "penalty": "ridge", "alpha": 300.0, "n_zones": 256}   # chosen in MP2.3 and MP2.4'''

MP26_CHECK = r'''if not filled(FINAL):
    print("⏳ MP2.6 : remplis FINAL (consigne a).")
elif ready("X_train", "y_train", "folds", "vault", "cv_results", "chosen", "chosen_name", "oof") and housing_ok():
    checks_of("MP2.6")
    with wb.attempt("MP2.6"):
        import sklearn
        from sklearn.cluster import KMeans as SkKMeans                          # only to check, never to build
        from sklearn.linear_model import Lasso as SkLasso, LinearRegression as SkLinearRegression, Ridge as SkRidge
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        same = isinstance(FINAL, dict) and canonical(FINAL) == canonical(chosen)
        decided = verdict("MP2.6", same, "FINAL est le modèle retenu par tes folds.",
                          f"FINAL n'est pas le modèle retenu par tes folds ({chosen}) : le coffre reste fermé. Le test ne "
                          f"mesure que la décision prise sur les folds ; si tu as une raison d'en choisir un autre, "
                          f"écris-la dans ton README et donne aussi ce modèle à chosen.")
        blocking = sorted(cell for cell, failures in FAILED.items() if failures and cell < "MP2.6")
        if blocking:
            print(f"⏳ MP2.6 : le coffre reste fermé tant que des garde-fous échouent ({', '.join(blocking)}) : "
                  f"corrige-les, relance ces cellules, puis celle-ci.")
        params = {"random_state": ZONE_SEED, **canonical(FINAL)}

        def sklearn_regressor(params):
            penalty, alpha = params.get("penalty", "none"), params.get("alpha", 1.0)
            if penalty == "ridge":
                return SkRidge(alpha=alpha)
            return SkLasso(alpha=alpha, tol=1e-10, max_iter=100_000) if penalty == "lasso" else SkLinearRegression()

        def sklearn_chain(X_fit, y_fit, X_new, params):
            """The same chain built with scikit-learn (and NumPy for the bounds), its own k-means for the zones."""
            low, high = np.percentile(X_fit, [1.0, 99.0], axis=0)
            first = StandardScaler().fit(np.clip(X_fit, low, high))
            poly = PolynomialFeatures(params.get("degree", 1), include_bias=False).fit(first.transform(np.clip(X_fit, low, high)))
            k = params.get("n_zones", 0)
            geo_scaler = StandardScaler().fit(X_fit[:, [6, 7]])
            kmeans = SkKMeans(n_clusters=k, n_init=1, random_state=ZONE_SEED).fit(geo_scaler.transform(X_fit[:, [6, 7]])) if k else None

            def design(X):
                columns = poly.transform(first.transform(np.clip(X, low, high)))
                if kmeans is None:
                    return columns
                return np.hstack([columns, np.eye(k)[kmeans.predict(geo_scaler.transform(X[:, [6, 7]]))]])

            second = StandardScaler().fit(design(X_fit))
            regressor = sklearn_regressor(params).fit(second.transform(design(X_fit)), y_fit)
            return regressor.predict(second.transform(design(X_new)))

        final_model = housing.HousingModel(**params).fit(X_train, y_train)
        design = np.asarray(final_model.transform(X_train))
        twin = sklearn_regressor(FINAL).fit(design, y_train)
        gap = float(np.abs(twin.predict(design) - np.asarray(final_model.predict(X_train))).max())
        verdict("MP2.6", gap < 1e-4, f"scikit-learn, sur ta matrice de design : les mêmes prédictions que ta librairie "
                f"(écart maximal {gap:.1e}).",
                f"sur ta matrice de design, le modèle de scikit-learn s'écarte de tes prédictions (écart maximal {gap:.3g}).")
        sk_scores = np.array([housing.rmse(y_train[val], sklearn_chain(X_train[train], y_train[train], X_train[val], FINAL))
                              for train, val in folds])
        mine = (np.asarray(cv_results[chosen_name]["val_rmse"], dtype=float) if same else
                np.asarray(housing.cross_validate(housing.HousingModel(**params), X_train, y_train, folds)["val_rmse"],
                           dtype=float))
        print(f"validation croisée, 5 folds : ta librairie {np.mean(mine):.4f} ± {np.std(mine, ddof=1):.4f} ; "
              f"la chaîne scikit-learn {sk_scores.mean():.4f} ± {sk_scores.std(ddof=1):.4f}")
        verdict("MP2.6", abs(sk_scores.mean() - np.mean(mine)) < 0.01,
                "la chaîne de scikit-learn obtient un score voisin (à 0.01 près).",
                "la chaîne de scikit-learn obtient un score assez différent du tien (plus de 0.01) : compare les étapes.")

        if decided and not blocking:
            description = "HousingModel(" + ", ".join(f"{key}={value!r}" for key, value in sorted(params.items())) + ")"
            X_test, y_test = vault.open(description)                       # the only opening of the vault
            predictions = np.asarray(final_model.predict(X_test), dtype=float)
            test_rmse = housing.rmse(y_test, predictions)
            squared = (y_test - predictions) ** 2
            ci = mylearn.stats.bootstrap_ci(squared, statistic=lambda e: float(np.sqrt(np.mean(e))), n_boot=1000,
                                            rng=np.random.default_rng(0))
            baseline_test = housing.rmse(y_test, np.full(len(y_test), y_train.mean()))
            r2 = float(mylearn.linear.r2_score(y_test, predictions))
            print(f"\nTEST ({fmt_int(len(y_test))} districts) : RMSE {test_rmse:.4f}, intervalle bootstrap à 95 % "
                  f"[{ci[0]:.4f} ; {ci[1]:.4f}], R² = {r2:.3f}")
            print(f"référence naïve (la moyenne de l'entraînement) sur le test : RMSE {baseline_test:.4f}")
            results = {
                "project": "MP2 - California housing: an honest evaluation protocol",
                "data": {"districts": int(len(y_all)), "train": int(len(y_train)), "test": int(len(y_test)), "test_size": 0.2,
                         "target": data.TARGET, "unit": "100 000 $"},
                "seeds": {"test": TEST_SEED, "folds": FOLD_SEED, "zones": ZONE_SEED, "bootstrap": 0},
                "n_folds": N_FOLDS,
                "chosen": {**dict(sorted(params.items())), "clip": [1.0, 99.0]},
                "cross_validation": {name: {"val_rmse": np.round(r["val_rmse"], 4).tolist(),
                                            "train_rmse": np.round(r["train_rmse"], 4).tolist(),
                                            "mean": round(float(np.mean(r["val_rmse"])), 4),
                                            "std": round(float(np.std(r["val_rmse"], ddof=1)), 4)}
                                     for name, r in cv_results.items()},
                "scikit_learn_chain_cv": {"val_rmse": np.round(sk_scores, 4).tolist(), "mean": round(float(sk_scores.mean()), 4)},
                "test": {"rmse": round(test_rmse, 4), "rmse_ci95": [round(ci[0], 4), round(ci[1], 4)], "r2": round(r2, 4),
                         "baseline_rmse": round(baseline_test, 4)},
                "vault_openings": vault.openings(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit-learn": sklearn.__version__, "mylearn": MYLEARN_SOURCE},
                "fast_mode": bool(FAST_MODE),
            }
            (PROJECT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print("💾 results.json écrit.")'''

# ---------------------------------------------------------------------------
# MP2.7
# ---------------------------------------------------------------------------
MP27_MD = r'''## MP2.7 · Emballer : tests, README de portfolio, notebook relancé, commit ⏱️ 75 min

**Objectif :** un projet qu'un recruteur peut lire, vérifier et relancer.

a) **Les tests** : complète `test_housing.py` jusqu'à au moins **six** tests, tous verts (les idées sont dans l'en-tête du fichier) ; chaque message d'échec dit ce qui était attendu.
b) **Le README** (`README.md` de ton dossier) : remplace chaque paragraphe « TODO » par ton texte et tes chiffres (le tableau des scores, avec leur incertitude, le score du test, les choix et les figures), puis les limites et les questions d'équité. Tu peux l'écrire en anglais, avec le même plan.
c) **La propreté** : redémarre le noyau et exécute tout le notebook (quelques minutes), relis les sorties : le coffre doit annoncer « déjà ouvert … pour ce même modèle », pas une nouvelle ouverture. Puis enregistre ton travail avec git, par exemple :

```bash
git add mon_travail/projets/partie_2_california_validation
git commit -m "MP2: honest evaluation of linear models on California housing"
```

La vérification ci-dessous lance tes tests et relit ton dossier. Ensuite : la grille d'évaluation du cahier des charges, puis, seulement après, la solution de référence (`projets/partie_2_california_validation/solution/`).'''

MP27_CHECK = r'''if ready("results", "vault"):
    checks_of("MP2.7")
    summary = run_project_tests()
    passed = int(re.search(r"(\d+) passed", summary).group(1)) if re.search(r"(\d+) passed", summary) else 0
    verdict("MP2.7", not re.search(r"failed|error", summary) and passed >= 6, f"{passed} tests, tous verts.",
            f"il faut au moins 6 tests, tous verts ({passed} vert(s) pour l'instant).")
    readme_path = PROJECT / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    headers = [h.lower() for h in re.findall(r"(?m)^#{2,}\s+(.*)$", readme)]
    SECTIONS = {"Le problème": ("problème", "probleme", "problem"),          # a README in English is welcome
                "Les données": ("données", "donnees", "data"),
                "Le protocole": ("protocole", "protocol"),
                "Les résultats": ("résultat", "resultat", "result"),
                "Les limites": ("limite", "limit")}
    for section, keywords in SECTIONS.items():
        found = any(word in header for header in headers for word in keywords)
        verdict("MP2.7", found, f"README : section « {section} ».",
                f"README : il manque une section « ## {section} » (ou en anglais, « ## {keywords[-1].capitalize()}… »).")
    todo = re.findall(r"(?m)^\W*TODO\b", readme)
    verdict("MP2.7", bool(readme) and not todo, "README : plus aucun paragraphe « TODO ».",
            f"README : {len(todo)} paragraphe(s) commencent encore par « TODO » : remplace-les par ton texte." if readme
            else "il manque README.md dans le dossier du projet.")
    for figure in ("courbes_validation.png", "zones.png", "courbes_apprentissage.png", "carte_zones_residus.png"):
        verdict("MP2.7", f"figures/{figure}" in readme and (FIGURES / figure).exists(), f"README : figures/{figure}.",
                f"README : affiche figures/{figure} (le notebook la crée).")
    state, detail = committed(PROJECT / "test_indices.npy")
    verdict("MP2.7", bool(state), f"test_indices.npy est commité ({detail}).", f"test_indices.npy {detail}.")
    alone, detail = committed_alone(PROJECT / "test_indices.npy")
    if alone is not None:
        verdict("MP2.7", alone, f"le premier commit de test_indices.npy ne contient que lui ({detail}).",
                f"le premier commit de test_indices.npy contient d'autres fichiers ({detail}) : la grille demande un "
                f"commit du seul jeu de test, avant toute modélisation ; dis-le dans ton README.")
    descriptions = {opening["description"] for opening in vault.openings()}
    verdict("MP2.7", len(descriptions) == 1, "le coffre n'a été ouvert que pour un seul modèle.",
            f"le coffre a été ouvert pour {len(descriptions)} modèles (vault.json) : le score du test n'est plus une "
            f"mesure honnête, dis-le dans ton README.")
    verdict("MP2.7", (PROJECT / "results.json").exists(), "results.json est écrit.", "il manque results.json (MP2.6).")'''

NOTES = {   # step -> (questions for the learner, observations of the reference solution)
    "MP2.1": ("""- La part des valeurs plafonnées, et ce que ce plafond fait aux erreurs :
- Les mesures qui ont des valeurs extrêmes (et ce qu'elles représentent) :
- Où sont les logements chers :""",
              """- **Le plafond** : 809 districts d'entraînement sur 16 512 (4,9 %) sont à 5 : « 500 000 dollars ou plus ». Au-dessus, la vraie valeur est inconnue ; un modèle linéaire, qui ignore le plafond, sous-estime ces districts, et leur erreur mesure autant le plafond que le modèle (MP2.5 le chiffre). Deux mesures sont plafonnées elles aussi : 1 018 districts ont un `HouseAge` de 52 ans exactement (37 seulement à 51 ans), et 44 un `MedInc` de 15.
- **Les valeurs extrêmes** : `AveOccup` monte à 599,7 personnes par ménage (99 % des districts sont sous 5,4), `AveRooms` à 141,9 pièces, `AveBedrms` à 34,1, `Population` à 35 682 habitants. Ce sont des districts avec très peu de ménages et beaucoup de logements vides ou collectifs (résidences de vacances, foyers…), où une moyenne par ménage n'a plus grand sens. Ces points ont un fort effet de levier sur une régression : d'où les bornes de MP2.2.
- **Les logements chers** sont sur la côte : la baie de San Francisco, Los Angeles et sa côte, San Diego. L'intérieur (la Vallée centrale, le nord, le désert) est bien moins cher. L'emplacement compte autant que le revenu, et une fonction linéaire de la latitude et de la longitude ne peut pas le décrire : c'est l'idée des zones de MP2.4."""),
    "MP2.2": ("""- Le gain des moindres carrés sur la référence naïve :
- Ce que changent les bornes, et pourquoi :
- Ce que dit l'écart-type entre les folds :""",
              """- **Le gain** : la référence naïve fait 1,159 en validation (l'écart-type des valeurs), les moindres carrés bornés 0,662 : 43 % d'erreur en moins, soit un $R^2$ d'environ $1 - (0{,}662/1{,}159)^2 \\approx 0{,}67$.
- **Les bornes** font passer la RMSE de 0,735 à 0,662, et l'erreur d'entraînement elle-même baisse (de 0,731 à 0,662) : sans bornes, quelques districts aux valeurs extrêmes, à fort effet de levier, tirent la droite vers eux et faussent les prédictions de tous les autres. Les bornes sont apprises dans chaque fold (les percentiles 1 et 99 de ses lignes d'entraînement) : elles ne voient jamais la validation.
- **L'écart-type entre les folds** vaut 0,015 : les scores des folds vont de 0,643 à 0,682. L'erreur type de la moyenne vaut à peu près $0{,}015/\\sqrt{5} \\approx 0{,}007$ (une approximation optimiste, puisque les folds partagent leurs données) : un écart de quelques millièmes entre deux modèles ne se voit pas à l'œil nu. Mais les folds sont communs, et la comparaison fold par fold est nette : les bornes gagnent sur les 5 folds, de 0,06 à 0,08."""),
    "MP2.3": ("""- Où le modèle sous-apprend, où il surapprend (lis les courbes) :
- Le modèle retenu par la règle d'une erreur type, et s'il diffère du meilleur score, pourquoi :
- Pourquoi l'erreur type σ/√k est optimiste :
- Le Lasso face à Ridge, et à quoi d'autre il peut servir :""",
              """- **Les courbes** : aux degrés 1 et 2, l'entraînement et la validation restent collés (0,662 ; 0,602 et 0,605) : ces modèles sous-apprennent, et une pénalité ne peut que les dégrader, nettement au-delà de α = 100 (0,83 au degré 1 avec α = 10 000). Au degré 4 peu pénalisé, l'écart s'ouvre : 0,519 à l'entraînement, 0,586 en validation, et des folds qui s'éparpillent (écart-type 0,028 contre 0,015) : c'est du surapprentissage, que α = 100 corrige (0,5755, le meilleur score de la grille).
- **La règle d'une erreur type** retient le degré 3 avec α = 100 (0,5815 ± 0,0154), et non le meilleur score, le degré 4 avec α = 100 (0,5755 ± 0,0177) : leur écart, 0,006, est plus petit que l'erreur type du meilleur (0,0079), et le degré 3 n'a que 164 colonnes au lieu de 494.
- **$\\sigma/\\sqrt{k}$ est optimiste** : elle suppose cinq mesures indépendantes. Or les modèles des folds partagent les trois quarts de leurs données d'entraînement, et leurs erreurs sont corrélées : la vraie incertitude du score moyen est plus grande, et il n'existe même pas d'estimateur sans biais de cette variance qui vaille pour toutes les distributions (Bengio et Grandvalet, 2004). La règle est un garde-fou pratique, pas un test.
- **Le Lasso** ne fait pas mieux que Ridge au degré 2 (0,6055 au mieux, contre 0,6048) : à ce degré, le modèle sous-apprend et aucune pénalité n'aide. Son intérêt est ailleurs : avec α = 0,03, il ne garde que 16 colonnes sur 44 pour 0,665, à peu près le score des moindres carrés sur les 8 mesures (0,662) ; avec α = 0,1, 7 colonnes (`MedInc`, `HouseAge`, `AveOccup`, `Latitude`, `MedInc·AveRooms`, `AveOccup²`, `Latitude²`) pour 0,762 : un modèle lisible, au prix de l'erreur."""),
    "MP2.4": ("""- Le k que préfère la silhouette, celui que retient la validation croisée, et pourquoi ils diffèrent (ou non) :
- Ce que les zones représentent pour le modèle :
- Pourquoi le k-means est refait dans chaque fold :
- Jusqu'où pousser le nombre de zones, et ce qui limite ce choix :""",
              """- **La silhouette préfère k = 2** (0,755) : le nord et le sud de la Californie, deux nuages de districts bien séparés. Mais en validation croisée, 2 zones n'apportent rien (0,5816 contre 0,5815 sans zone). La silhouette mesure si les groupes sont **compacts et séparés**, pas s'ils **aident à prédire** les prix ; ici, les zones sont des features, et le bon critère est la validation croisée. Elle retient 256 zones (0,4970), le haut de la grille ; α, réajusté pour 256 zones, passe à 300 (0,4984 ± 0,0101 ; le meilleur, α = 10, fait 0,4964, à moins d'une erreur type).
- **Ce que représentent les zones** : un niveau de prix par quartier, appris sur les districts d'entraînement de la zone (une cinquantaine par zone avec 256 zones). C'est une moyenne locale des prix, sous forme de colonnes 0/1 : le modèle, qui ne savait dire que « plus au nord, plus à l'ouest », sait dire « dans ce quartier de la baie ».
- **Refaire le k-means dans chaque fold** : les zones sont apprises des données. Apprises sur tous les districts, elles placeraient leurs frontières en tenant compte de ceux de validation : la fuite serait légère ici (le k-means ne voit pas la cible), mais le principe ne change pas : tout ce qui apprend des données apprend dans `fit`, que la validation croisée refait dans chaque fold.
- **Jusqu'où aller** : la courbe descend encore (pendant la préparation du projet, hors notebook : 0,483 avec 512 zones et 0,471 avec 1 024, pour α = 100). Mais avec 1 024 zones, une zone ne contient plus qu'une douzaine de districts d'entraînement : le modèle fait la moyenne des prix de quelques voisins. Surtout, les folds sont tirés au hasard : chaque district de validation a des voisins immédiats à l'entraînement, au prix très proche, et plus les zones sont petites, plus le modèle exploite ce voisinage, qui n'existera pas pour un quartier nouveau. Une validation croisée **spatiale** (des folds faits de blocs géographiques) mesurerait ce que vaut le modèle loin de ses données (voir les extensions). Le temps de calcul compte aussi : la grille s'arrête à 256."""),
    "MP2.5": ("""- Biais ou variance (lis les courbes d'apprentissage et la variance des prédictions) :
- Ce qu'apporteraient plus de districts :
- Qui est mal servi, et ce qu'il faudrait dire à un utilisateur :""",
              """- **Biais ou variance** : les courbes des moindres carrés sont collées dès 1 000 districts (0,662 à l'entraînement comme en validation) : un biais pur, que plus de données ne corrigeraient pas. Le modèle retenu garde un écart de 0,02 avec tous les districts (0,480 à l'entraînement, 0,500 en validation, avec 3 folds) : un peu de variance, mais surtout du biais, et du bruit (la valeur médiane d'un district n'est pas une fonction exacte de ces 8 mesures). La variance des prédictions entre les modèles des folds va dans le même sens : 0,0001 pour les moindres carrés, 0,0019 pour le degré 3 sans zone, 0,0113 pour le modèle retenu, 0,0139 pour le degré 4 peu pénalisé, à comparer à une erreur quadratique moyenne d'environ 0,25. C'est une mesure relative, et plutôt basse (ces modèles partagent une partie de leurs données), mais la variance n'est qu'une petite part de l'erreur. Les zones ont échangé beaucoup de biais contre un peu de variance.
- **Plus de districts** : la courbe de validation du modèle retenu descend encore (0,505 avec 8 000 districts, 0,500 avec 11 008) : quelques milliers de districts de plus feraient gagner un peu, sans changer l'ordre de grandeur.
- **Qui est mal servi** : d'abord les 809 districts plafonnés (RMSE 1,06, résidu moyen +0,64) : le modèle les sous-estime de 64 000 dollars en moyenne, et davantage en réalité, puisque leur vraie valeur dépasse 500 000 dollars. En valeur absolue, les régions chères ont les plus grosses erreurs (Los Angeles : 0,557 ; la baie de San Francisco : 0,541). En valeur relative, c'est l'inverse : l'erreur vaut 21 % de la valeur moyenne dans la baie et 23 % à Los Angeles, mais de 24 à 29 % ailleurs (29 % autour de Sacramento, 28,5 % à San Diego, 28 % autour de Fresno). À un utilisateur, il faut dire : l'erreur typique est d'environ 50 000 dollars (un quart de la valeur d'un district moyen), plus lourde en proportion hors des deux grandes métropoles, et le modèle ne dit rien au-delà de 500 000 dollars."""),
    "MP2.6": ("""- Le score du test face à la validation croisée :
- Ce que tu ferais si le test était nettement moins bon :""",
              """- **Le test** : RMSE 0,478, intervalle bootstrap à 95 % [0,459 ; 0,498], $R^2 = 0{,}822$, contre 1,134 pour la référence naïve. La validation croisée annonçait 0,498 ± 0,010 : le test fait un peu mieux, et le score de validation croisée est juste au-dessus de son intervalle. L'écart de 0,02 tient surtout au hasard de l'échantillon de test (la demi-largeur de son intervalle est de 0,019) ; les districts d'entraînement en plus (16 512 au lieu d'environ 13 200 dans chaque fold) n'en expliquent que quelques millièmes, d'après la courbe d'apprentissage. Rien ne suggère que la sélection ait surajusté les folds : le test n'est pas moins bon que la validation croisée.
- **scikit-learn** : sur la même matrice de design, son `Ridge` redonne les prédictions de `mylearn` (écart maximal de l'ordre de $10^{-14}$, affiché par la vérification). La chaîne entière, avec son propre k-means, fait 0,4973 ± 0,0123 en validation croisée, contre 0,4984 ± 0,0101 : ses zones sont différentes (un autre optimum local du k-means), le score est le même à l'incertitude près.
- **Si le test était nettement moins bon** : on ne retouche pas le modèle en regardant le test. On le dit dans le README, on cherche la cause sans le test (une fuite dans le protocole, des districts de test différents de ceux d'entraînement, un fold atypique), et si l'on change le modèle, il faut un nouveau jeu de test, ou annoncer que le score du test est devenu optimiste."""),
}


def notes_cell(step: str, exercise: bool):
    questions, observations = NOTES[step]
    if exercise:
        return md(f"✍️ **Tes observations ({step})** : double-clique sur cette cellule pour écrire tes réponses ; "
                  f"elles nourriront ton README.\n\n{questions}")
    return md(f"✍️ **Observations ({step})**\n\n{observations}")


FOOTER = r'''## ✅ Bilan

**Grille d'évaluation** (sur 20, détaillée dans le cahier des charges) : protocole sans fuite (4) · modèles `mylearn` vérifiés contre scikit-learn, référence naïve (3) · hyperparamètres choisis par validation croisée, avec leur incertitude (3) · zones k-means justifiées et apport mesuré (2) · diagnostic biais-variance (3) · README, figures, limites et équité (3) · reproductibilité (2).

**Pour aller plus loin** : les extensions du cahier des charges (une validation croisée spatiale, HDBSCAN pour les zones, la double descente avec des features aléatoires, un réglage de α par *successive halving* avec ta librairie `bandit`, le gradient boosting du ch. 14, une carte interactive). La suite du workbook : la partie III, où l'on prépare les données (ch. 12) et où l'on entraîne les grands classifieurs classiques.'''


def header_cells(kind: str) -> list:
    rel = STARTER if kind == "exercise" else SOLUTION
    rows = ["| Étape | Contenu | ⏱️ |", "|---|---|---|"] + [f"| {sid} | {title} | {minutes} min |"
                                                            for sid, title, minutes in STEPS]
    fast = ("**Mode rapide.** Avec `FAST_MODE = True` (la cellule de setup), seules les courbes d'apprentissage et la "
            "mesure de la variance (MP2.5) passent de 5 à 3 folds ; toutes les décisions et les résidus gardent leurs "
            "5 folds. Le notebook complet tourne en deux minutes environ sur un CPU, une dizaine de secondes de plus "
            "en mode complet.")
    if kind == "exercise":
        title = "# Mini-projet MP2 · Prix des logements californiens : un protocole d'évaluation honnête"
        how = ("Le cahier des charges est dans `projets/partie_2_california_validation/README.md` : lis-le d'abord. "
               "Ce notebook te guide en sept étapes. Le code durable va dans **ton module** `housing.py` (et ses tests, "
               "`test_housing.py`) ; le notebook l'appelle, le vérifie, prend les décisions sur les folds et trace les "
               "figures. Les cellules à compléter lèvent `NotImplementedError` ou contiennent des `...` ; les "
               "vérifications affichent ✅ ou ❌ (des garde-fous, pas une note), et ⏳ tant qu'une étape n'est pas faite. "
               "Pas de bibliothèque de machine learning pour les modèles : NumPy et ta librairie `mylearn` "
               "(scikit-learn ne sert qu'à vérifier tes chiffres).\n\n"
               "> Travaille dans **ta copie** (`mon_travail/projets/partie_2_california_validation/mp2_california.ipynb`, "
               "créée par `python tools/start_chapter.py CP2`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = ("# Mini-projet MP2 · Prix des logements californiens : un protocole d'évaluation honnête — "
                 "solution de référence (exécutée)")
        how = ("La solution de référence, exécutée avec `housing.py` de ce dossier et la librairie de référence. Son "
               "jeu de test (`test_indices.npy`) a été commité avant toute modélisation, et le coffre n'a été ouvert "
               "qu'une fois (`vault.json`). Tes chiffres n'ont pas à être identiques : d'autres grilles ou d'autres "
               "choix donnent d'autres valeurs, tout aussi justes s'ils sont bien mesurés. Le README de portfolio "
               "rédigé à partir de ces résultats est dans ce dossier (`README.md`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n{fast}\n\n" + "\n".join(rows))]


def build(kind: str) -> list:
    exercise = kind == "exercise"
    cells = header_cells(kind)
    cells.append(setup_cell("exercise", chapter="CP2") if exercise else setup_cell("solution"))
    cells.append(code(PROJECT_STARTER if exercise else PROJECT_SOLUTION))
    cells.append(code(TOOLS))
    cells += [md(MP21_MD), code(MP21_CHECK), code(MP21_COMMIT), code(MP21_EXPLORE), notes_cell("MP2.1", exercise)]
    cells += [md(MP22_MD), code(MP22_FOLDS), code(MP22_TODO if exercise else MP22_SOLUTION), code(MP22_CHECK),
              notes_cell("MP2.2", exercise)]
    cells += [md(MP23_MD), code(MP23_CURVES), code(MP23_TODO if exercise else MP23_SOLUTION), code(MP23_CHOICE),
              code(MP23_LASSO), notes_cell("MP2.3", exercise)]
    cells += [md(MP24_MD), code(MP24_SILHOUETTE), code(MP24_ZONES), notes_cell("MP2.4", exercise)]
    cells += [md(MP25_MD), code(MP25_CURVES), code(MP25_TODO if exercise else MP25_SOLUTION), code(MP25_RESIDUALS),
              notes_cell("MP2.5", exercise)]
    cells += [md(MP26_MD), code(MP26_TODO if exercise else MP26_SOLUTION), code(MP26_CHECK), notes_cell("MP2.6", exercise)]
    cells += [md(MP27_MD), code(MP27_CHECK), md(FOOTER)]
    return cells


def main() -> int:
    MODULE_STARTER.write_text(starter_module(MODULE_SOLUTION.read_text(encoding="utf-8")), encoding="utf-8")
    write_notebook(STARTER, build("exercise"))
    write_notebook(SOLUTION, build("solution"))
    print(f"✅ {MODULE_STARTER.relative_to(ROOT)}, {STARTER} and {SOLUTION} written ({len(STEPS)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
