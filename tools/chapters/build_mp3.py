#!/usr/bin/env python
"""Build the mini-project MP3 from a single source (used by Claude).

    python tools/chapters/build_mp3.py
    python tools/run_all_notebooks.py projets/partie_3_revenus/solution/mp3_revenus.ipynb --inplace
    python tools/run_all_notebooks.py projets/partie_3_revenus/depart/mp3_revenus.ipynb

Writes the two notebooks (depart/ and solution/mp3_revenus.ipynb) and the starter module
depart/income.py, generated from the reference solution/income.py: same constants,
signatures and docstrings, private helpers removed, every public function raising
NotImplementedError with the step of the notebook where it is written.
``python tools/start_chapter.py CP3`` copies depart/ into mon_travail/projets/.
Steps MP3.1 to MP3.8 follow the brief (docs/syllabus/data/cp3.json, miniproject).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import ROOT, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "projets/partie_3_revenus"
NOTEBOOK = "mp3_revenus.ipynb"
STARTER = f"{FOLDER}/depart/{NOTEBOOK}"
SOLUTION = f"{FOLDER}/solution/{NOTEBOOK}"
MODULE_SOLUTION = ROOT / FOLDER / "solution" / "income.py"
MODULE_STARTER = ROOT / FOLDER / "depart" / "income.py"

STEPS = [("MP3.1", "Cadrer le projet, commencer la data card, geler un jeu de test stratifié", 45),
         ("MP3.2", "Explorer et nettoyer : types, valeurs manquantes, doublons, déséquilibre", 60),
         ("MP3.3", "Les baselines : DummyClassifier, puis un Pipeline avec la régression logistique", 45),
         ("MP3.4", "Les classifieurs des ch. 13 et 14 dans le même ColumnTransformer, sur les mêmes folds", 60),
         ("MP3.5", "Le gradient boosting : catégories natives, recherche d'hyperparamètres, choix justifié", 60),
         ("MP3.6", "Des métriques adaptées au déséquilibre, le seuil de décision, le test ouvert une fois", 40),
         ("MP3.7", "L'audit : les erreurs par sous-groupe, le retrait de sex, l'importance par permutation", 60),
         ("MP3.8", "Emballer : tests, modèle sauvegardé, README, model card, data card, commit", 60)]

# Step of the notebook where each public function of income.py is written (starter module).
TODO_STEPS = {
    "make_test_indices": "MP3.1",
    "build_preprocessor": "MP3.3",
    "build_pipeline": "MP3.3",
    "cross_validate_all": "MP3.3",
    "f1_threshold": "MP3.6",
    "age_groups": "MP3.7",
    "subgroup_report": "MP3.7",
    "library_versions": "MP3.8",
    "save_model": "MP3.8",
    "load_model": "MP3.8",
}

STARTER_DOC_FIRST = ("Income over 50 000 $ from the 1994 census, with scikit-learn pipelines (mini-project of "
                     "part III): YOUR module.")
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
    """The starter income.py: the reference without its bodies (docstrings kept) and without private helpers."""
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
        stubs[doc.end_lineno - 1] = f"{indent}raise NotImplementedError(\"{qualname}\")  # TODO {TODO_STEPS[qualname]}"

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
    first = source.splitlines()[0]
    if not first.startswith('"""'):
        raise ValueError("income.py must start with its docstring")
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
PROJECT_STARTER = r'''# The project folder: your copy (mon_travail/projets/...), where data.py, income.py and test_income.py live
PROJECT_NAME = "partie_3_revenus"
STARTER_KIT = ROOT / "projets" / PROJECT_NAME / "depart"
candidates = [Path.cwd(), ROOT / "mon_travail" / "projets" / PROJECT_NAME, STARTER_KIT]
PROJECT = next(folder for folder in candidates if (folder / "income.py").exists() and (folder / "data.py").exists())
FIGURES = PROJECT / "figures"
MYLEARN_SOURCE = "learner (reference modules for the missing ones)"
if PROJECT.resolve() == STARTER_KIT.resolve():
    print("⚠️ Ce dossier est le kit de départ du dépôt, pas ta copie : lance python tools/start_chapter.py CP3, puis "
          "ouvre mon_travail/projets/partie_3_revenus/mp3_revenus.ipynb.")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import data  # noqa: E402

try:
    import income  # noqa: E402   (your module: it imports your mylearn library)
except ModuleNotFoundError as exc:
    income = None
    print(f"⚠️ {exc} : lance d'abord python tools/start_chapter.py CP3, qui crée ta librairie mylearn.")
except Exception as exc:                       # a syntax error in income.py must not stop the notebook
    income = None
    print(f"⚠️ income.py ne s'importe pas : {type(exc).__name__}: {exc}. Corrige-le ; l'étape MP3.1 le rechargera.")
print("Dossier du projet :", PROJECT)'''

PROJECT_SOLUTION = r'''# The project folder of the reference solution
PROJECT_NAME = "partie_3_revenus"
PROJECT = ROOT / "projets" / PROJECT_NAME / "solution"
FIGURES = PROJECT / "figures"
MYLEARN_SOURCE = "reference"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import data  # noqa: E402
import income  # noqa: E402

print("Dossier du projet :", PROJECT.relative_to(ROOT))'''

TOOLS = r'''# Tools of the notebook (run this cell)
import importlib
import json
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5                                    # folds of every comparison and of the out-of-fold predictions
TEST_SEED, FOLD_SEED, SEARCH_SEED = 2026, 0, 0 # seeds of the test set, of the folds, of the random search
N_SEARCH = 12 if FAST_MODE else 30             # candidates of the random search (MP3.5)
N_BOOT = 300 if FAST_MODE else 1000            # bootstrap resamples of the test scores (MP3.6)
N_REPEATS = 5 if FAST_MODE else 10             # repeats of the permutation importance (MP3.7)
MAIN = "average_precision"                     # the score that decides (MP3.3): the PR-AUC

FAILED = {}         # check cell -> its guardrails that failed at its last run (the vault stays closed while any remain)
_CURRENT = [None]
UNFINISHED = "la cellule s'est arrêtée avant la fin de ses vérifications (une erreur, ou une étape pas encore faite)"


def checks_of(cell):
    """Start the guardrails of one check cell: its failures of a previous run are forgotten. Until checks_done() runs,
    the cell counts as failed: a cell stopped by an error keeps the vault closed."""
    _CURRENT[0] = cell
    FAILED[cell] = [UNFINISHED]


def checks_done():
    """The current check cell ran to its end."""
    if _CURRENT[0] in FAILED and UNFINISHED in FAILED[_CURRENT[0]]:
        FAILED[_CURRENT[0]].remove(UNFINISHED)


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
        print(f"⏳ il manque {', '.join(missing)} : exécute d'abord les cellules des étapes précédentes, vérifications comprises (ce sont souvent elles qui créent ces variables).")
    return not missing


def fmt_int(n):
    """1234567 -> "1 234 567" (the French thousands separator, a narrow space)."""
    return f"{n:,}".replace(",", " ")


def income_ok():
    """Import your latest saved income.py again; False, with the reason, when it cannot be imported yet."""
    global income
    try:
        current = globals().get("income")
        income = importlib.reload(current) if current is not None else importlib.import_module("income")
    except Exception as exc:
        print(f"⚠️ income.py ne s'importe pas : {type(exc).__name__}: {exc}")
        return False
    return True


METRIC_NAMES = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "f1": "F1 (seuil 0,5)",
                "roc_auc": "ROC-AUC", "average_precision": "PR-AUC"}


def score_table(results, metrics=("accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision")):
    """One row per model: « mean ± standard deviation (ddof=1) » of each score over the folds, and the fit time."""
    rows = {}
    for name, scores in results.items():
        row = {METRIC_NAMES[m]: f"{np.mean(scores[m]):.4f} ± {np.std(scores[m], ddof=1):.4f}" for m in metrics if m in scores}
        if "fit_time" in scores:
            row["entraînement (s/fold)"] = f"{np.mean(scores['fit_time']):.1f}"
        if "score_time" in scores:
            row["prédiction (s/fold)"] = f"{np.mean(scores['score_time']):.1f}"
        rows[name] = row
    return pd.DataFrame(rows).T


def paired(results, name, reference, metric=MAIN):
    """Fold-by-fold difference of `metric` between two models measured on the same folds: (mean, std error, wins)."""
    diff = np.asarray(results[name][metric]) - np.asarray(results[reference][metric])
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff))), int((diff > 0).sum())


def git(path, *args):
    """Run a git command in the folder of `path` (None when git cannot be found)."""
    try:
        return subprocess.run(["git", *args], cwd=path.parent, capture_output=True, text=True)
    except OSError:
        return None


def committed(path):
    """(True, "commit abc1234 du 2026-10-16") when the file is committed and unchanged; (False or None, why) otherwise."""
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
    """Run the tests of the project folder (test_income.py) and print the failures; return pytest's summary line."""
    command = [sys.executable, "-m", "pytest", str(PROJECT / "test_income.py"), "-q", "-p", "no:cacheprovider",
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
    return summary


def sections_of(path):
    """The level-2+ headers of a Markdown file, lowercased ([] when the file does not exist)."""
    if not path.exists():
        return []
    return [h.lower() for h in re.findall(r"(?m)^#{2,}\s+(.*)$", path.read_text(encoding="utf-8"))]


def check_document(step, path, sections):
    """Guardrails of a Markdown deliverable: it exists, has its sections (French or English), and no TODO left."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not verdict(step, bool(text), f"{path.name} existe.", f"il manque {path.name} dans le dossier du projet."):
        return ""
    headers = sections_of(path)
    for section, keywords in sections.items():
        verdict(step, any(word in header for header in headers for word in keywords), f"{path.name} : section « {section} ».",
                f"{path.name} : il manque une section « ## {section} » (ou en anglais, « ## {keywords[-1].capitalize()}… »).")
    todo = re.findall(r"(?m)^\W*TODO\b", text)
    verdict(step, not todo, f"{path.name} : plus aucun paragraphe « TODO ».",
            f"{path.name} : {len(todo)} paragraphe(s) commencent encore par « TODO » : remplace-les par ton texte.")
    return text'''

# ---------------------------------------------------------------------------
# MP3.1
# ---------------------------------------------------------------------------
MP31_MD = r'''## MP3.1 · Cadrer le projet, commencer la data card, geler un jeu de test stratifié ⏱️ 45 min

**Objectif :** dire à quoi sert le modèle et quelles erreurs coûtent, puis mettre le jeu de test hors de portée avant de regarder les données.

a) **Cadre le problème** dans ton `README.md` (section « Le problème »). Le cahier des charges propose un usage : un institut de statistique complète la tranche de revenu des personnes qui n'ont pas répondu à cette question de l'enquête. Qui utilise la prédiction, que coûte un faux positif, un faux négatif, et pourquoi une erreur qui pèse plus sur un groupe que sur un autre fausse les statistiques de ce groupe ?
b) **Commence la data card** (`DATA_CARD.md`) : la provenance, la licence et la collecte (le cahier des charges donne les sources) ; la composition viendra en MP3.2.
c) Écris `make_test_indices` dans `income.py` (lis sa docstring) : un découpage **stratifié**, qui garde la même part de revenus élevés dans le test et dans le reste. La vérification enregistre les positions dans `test_indices.npy`, **une seule fois** : si le fichier existe, elle le relit et ne l'écrase jamais.
d) **Commite ce fichier tout de suite, seul**, avant toute modélisation. Depuis la racine du dépôt :

```bash
git add mon_travail/projets/partie_3_revenus/test_indices.npy
git commit -m "MP3: freeze the test set" -- mon_travail/projets/partie_3_revenus/test_indices.npy
```

Sans terminal (sur Colab, par exemple), passe `COMMIT` à `True` dans la cellule qui suit la vérification, exécute-la une fois, puis remets `False`. Le coffre (`data.TestVault`) garde ensuite les personnes de test : `vault.train_part()` rend les autres, les seules que tu regardes jusqu'à MP3.6.'''

MP31_CHECK = r'''X_all, y_all = data.load_census()
print(f"{fmt_int(len(y_all))} personnes, {X_all.shape[1]} colonnes ; revenus de plus de 50 000 $ : "
      f"{fmt_int(int(y_all.sum()))} ({y_all.mean():.1%})")
INDICES = PROJECT / "test_indices.npy"
checks_of("MP3.1")
if income_ok():
    with wb.attempt("MP3.1"):
        from sklearn.model_selection import train_test_split            # the oracle of the docstring
        test_indices = np.asarray(income.make_test_indices(y_all, test_size=0.2, seed=TEST_SEED))
        n_test = int(np.ceil(0.2 * len(y_all)))
        rest = np.setdiff1d(np.arange(len(y_all)), test_indices)
        checks = [
            verdict("MP3.1", test_indices.shape == (n_test,), f"{fmt_int(n_test)} personnes de test (20 %).",
                    f"il faut ceil(0,2 × n) = {n_test} positions de test, pas un tableau de forme {test_indices.shape}."),
            verdict("MP3.1", test_indices.size > 0 and bool(np.all(np.diff(test_indices) > 0)) and test_indices.min() >= 0
                    and test_indices.max() < len(y_all), "des positions triées, sans répétition, entre 0 et n − 1.",
                    "les positions doivent être triées dans l'ordre croissant, sans répétition, entre 0 et n − 1."),
            verdict("MP3.1", test_indices.size > 0 and abs(y_all[test_indices].mean() - y_all[rest].mean()) < 1e-3,
                    f"stratifié : {y_all[test_indices].mean():.2%} de revenus élevés dans le test, "
                    f"{y_all[rest].mean():.2%} dans le reste." if test_indices.size else "",
                    "le test et le reste n'ont pas la même part de revenus élevés : passe stratify=y à train_test_split."),
            verdict("MP3.1", np.array_equal(np.asarray(income.make_test_indices(y_all, 0.2, TEST_SEED)), test_indices),
                    "même graine, mêmes positions.", "deux appels avec la même graine donnent des positions différentes."),
            verdict("MP3.1", np.array_equal(test_indices, np.sort(train_test_split(np.arange(len(y_all)), test_size=0.2,
                                                                                    stratify=y_all,
                                                                                    random_state=TEST_SEED)[1])),
                    "le tirage est celui de la docstring (le même pour tout le monde, et pour les tests).",
                    "les positions doivent être la partie test de train_test_split(np.arange(n), test_size, stratify=y, "
                    "random_state=seed), triées (la docstring) : les tests de test_income.py s'appuient sur ce découpage."),
        ]
        if INDICES.exists():
            print("✅ MP3.1 : test_indices.npy existe déjà et n'a pas changé : il n'est jamais réécrit."
                  if np.array_equal(np.load(INDICES), test_indices) else
                  "⚠️ MP3.1 : test_indices.npy existe déjà et diffère de make_test_indices ; il n'est pas réécrit. "
                  "C'est lui qui fait foi (le coffre le lit) : explique l'écart dans ton README.")
        elif all(checks):
            np.save(INDICES, test_indices)
            print(f"💾 {INDICES.name} est écrit dans le dossier du projet : commite-le maintenant (consigne d).")
        else:
            print("⏳ MP3.1 : test_indices.npy n'est pas écrit tant qu'une vérification échoue.")
        checks_done()
if INDICES.exists():
    state, detail = committed(INDICES)
    print(f"✅ MP3.1 : test_indices.npy est commité ({detail})." if state else
          f"⏳ MP3.1 : test_indices.npy {detail} : commite-le avant d'aller plus loin (consigne d).")
    vault = data.TestVault(PROJECT, X_all, y_all)
    X_train, y_train = vault.train_part()
    print(f"Le coffre garde {fmt_int(len(vault.test_indices))} personnes de test ; "
          f"il reste {fmt_int(len(y_train))} personnes d'entraînement.")'''

MP31_COMMIT = r'''# Without a terminal (on Colab, for example): set COMMIT to True, run this cell once, then set it back to False
COMMIT = False
if COMMIT and not INDICES.exists():
    print("⏳ MP3.1 : pas encore de test_indices.npy à commiter (consigne c).")
elif COMMIT:
    commit_file(INDICES, "MP3: freeze the test set")
    state, detail = committed(INDICES)
    print(f"✅ MP3.1 : test_indices.npy est commité ({detail})." if state else f"⏳ MP3.1 : test_indices.npy {detail}.")'''

# ---------------------------------------------------------------------------
# MP3.2
# ---------------------------------------------------------------------------
MP32_MD = r'''## MP3.2 · Explorer et nettoyer : types, valeurs manquantes, doublons, déséquilibre ⏱️ 60 min

**Objectif :** connaître les colonnes avant de les donner à un modèle, et décider de ce que le pipeline en fera.

a) Écris `missing_table(X)` ci-dessous : pour chaque colonne qui a des valeurs manquantes, leur nombre et leur part, de la plus touchée à la moins touchée.
b) La cellule d'exploration décrit les personnes d'entraînement : les types, les doublons, le déséquilibre des classes, les gains en capital, la correspondance entre `education` et `education-num`, et la part des revenus élevés selon le sexe, l'âge, le niveau d'études et la situation matrimoniale (`figures/exploration.png`).
c) **Complète la data card** (section « Composition ») avec ces chiffres. Dans la cellule ✍️ : que veut dire « manquant » pour chaque colonne touchée (les trois sont des réponses non données, pas des erreurs de saisie) ? Quelles colonnes écartes-tu, et pourquoi ? Que fait le déséquilibre à l'accuracy ? Qu'est-ce qui frappe dans les taux de revenus élevés par groupe ?

Le nettoyage reste minimal, et rien n'est appris ici : les colonnes écartées (`income.DROPPED`) le sont par le `ColumnTransformer`, et les valeurs manquantes sont remplacées **dans le pipeline**, en MP3.3, avec des statistiques apprises sur les lignes d'entraînement de chaque fold.'''

MP32_TODO = r'''def missing_table(X):
    """For each column of the DataFrame X that has missing values: their number and their share.

    A DataFrame indexed by the column names, with the columns "manquants" (int) and "part" (float, between 0 and 1),
    sorted by decreasing number of missing values; the columns without missing value are left out.
    """
    raise NotImplementedError("missing_table")  # TODO MP3.2 a)'''

MP32_SOLUTION = r'''def missing_table(X):
    """For each column of the DataFrame X that has missing values: their number and their share.

    A DataFrame indexed by the column names, with the columns "manquants" (int) and "part" (float, between 0 and 1),
    sorted by decreasing number of missing values; the columns without missing value are left out.
    """
    counts = X.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    return pd.DataFrame({"manquants": counts.astype(int), "part": counts / len(X)})'''

MP32_CHECK = r'''if ready("X_train", "y_train"):
    checks_of("MP3.2")
    with wb.attempt("MP3.2"):
        table = missing_table(X_train)
        expected = X_train.isna().sum()
        expected = expected[expected > 0].sort_values(ascending=False)
        ok = (isinstance(table, pd.DataFrame) and list(table.index) == list(expected.index)
              and {"manquants", "part"} <= set(table.columns)
              and np.array_equal(np.asarray(table["manquants"], dtype=int), expected.to_numpy())
              and np.allclose(np.asarray(table["part"], dtype=float), expected.to_numpy() / len(X_train)))
        verdict("MP3.2", ok, "missing_table donne les bons nombres et les bonnes parts, dans le bon ordre.",
                f"missing_table doit rendre, pour les seules colonnes touchées et de la plus touchée à la moins touchée, "
                f"le nombre et la part des valeurs manquantes : {expected.to_dict()} sur {fmt_int(len(X_train))} lignes.")
        print(table.round(4).to_string() if isinstance(table, pd.DataFrame) else table)
        both = X_train[["workclass", "occupation"]].isna()
        print(f"\nworkclass et occupation manquants ensemble : {fmt_int(int(both.all(axis=1).sum()))} lignes ; "
              f"occupation seule : {fmt_int(int((both['occupation'] & ~both['workclass']).sum()))} "
              f"(workclass = {X_train.loc[both['occupation'] & ~both['workclass'], 'workclass'].unique().tolist()})")
        checks_done()'''

MP32_EXPLORE = r'''if ready("X_train", "y_train"):
    print(X_train.dtypes.value_counts().rename("colonnes").to_string(), "\n")
    print(X_train.describe().T.round(1).to_string(), "\n")
    full = X_train.assign(target=y_train)
    print(f"doublons exacts (les 14 colonnes et la cible) : {fmt_int(int(full.duplicated().sum()))} lignes ; "
          f"sans fnlwgt : {fmt_int(int(full.drop(columns='fnlwgt').duplicated().sum()))}")
    print(f"revenus de plus de 50 000 $ : {y_train.mean():.2%} des personnes d'entraînement "
          f"(un classifieur qui répond toujours « non » a donc une accuracy de {1 - y_train.mean():.2%})")
    for column in ("capital-gain", "capital-loss"):
        values = X_train[column]
        print(f"{column} : nul pour {(values == 0).mean():.1%} des personnes, maximum {fmt_int(int(values.max()))} "
              f"({fmt_int(int((values == values.max()).sum()))} personnes au maximum)")
    pairs = X_train.groupby("education")["education-num"].nunique()
    print(f"education → education-num : {len(pairs)} niveaux, un seul code chacun : {bool((pairs == 1).all())}")
    print(f"hours-per-week : de {X_train['hours-per-week'].min()} à {X_train['hours-per-week'].max()} ; "
          f"exactement 99 : {fmt_int(int((X_train['hours-per-week'] == 99).sum()))} personnes")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    bands = pd.cut(X_train["age"], [16, 24, 34, 44, 54, 64, 100], labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
    views = [("sex", X_train["sex"]), ("age band", bands), ("education-num", X_train["education-num"]),
             ("marital-status", X_train["marital-status"])]
    for ax, (title, groups) in zip(axes.ravel(), views):
        rates = full.groupby(groups, observed=True)["target"].agg(["mean", "size"])     # the order of the bands kept
        ax.bar(rates.index.astype(str), rates["mean"])
        ax.axhline(y_train.mean(), color="black", ls="--", lw=1, label=f"all ({y_train.mean():.1%})")
        ax.set(title=f"Share of incomes over 50K by {title}", ylabel="share > 50K", ylim=(0, 0.75))
        ax.tick_params(axis="x", rotation=30 if title == "marital-status" else 0, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right" if title == "marital-status" else "center")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "exploration.png")
    plt.show()'''

# ---------------------------------------------------------------------------
# MP3.3
# ---------------------------------------------------------------------------
MP33_MD = r'''## MP3.3 · Les baselines : DummyClassifier, puis un Pipeline avec la régression logistique ⏱️ 45 min

**Objectif :** un premier score honnête, avec son incertitude, et le chiffre à battre.

a) Dans `income.py`, écris `build_preprocessor`, `build_pipeline` et `cross_validate_all` (lis leurs docstrings) : tout le prétraitement dans un `ColumnTransformer`, le tout dans un `Pipeline` que la validation croisée clone et réentraîne dans chaque fold.
b) Les folds sont donnés : un `StratifiedKFold` à 5 folds, graine `FOLD_SEED`, **les mêmes pour toutes les comparaisons** (MP3.3 à MP3.6). Deux modèles mesurés sur les mêmes folds se comparent fold par fold.
c) La vérification contrôle ta chaîne, puis mesure deux baselines : `DummyClassifier()` (il prédit toujours la classe la plus fréquente) et la régression logistique (`LogisticRegression(max_iter=2000)`), sur cinq scores. Elle mesure aussi une version **qui fuit** : le prétraitement appris une fois sur toutes les personnes d'entraînement, avant la validation croisée.

Dans la cellule ✍️ : que vaut l'accuracy de la baseline, et quel score suivras-tu pour comparer les modèles ? Que gagne la régression logistique ? Que change la fuite ici, et pourquoi un `Pipeline` reste indispensable ? Le notebook prend la PR-AUC (`average_precision`, la variable `MAIN`) comme score de décision ; tu peux en choisir un autre si tu le justifies.'''

MP33_FOLDS = r'''if ready("X_train", "y_train"):
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds = list(cv.split(X_train, y_train))
    print("personnes de validation par fold :", [len(val) for _, val in folds],
          "; part de revenus élevés :", [f"{y_train[val].mean():.3f}" for _, val in folds])'''

MP33_CHECK = r'''if ready("X_train", "y_train", "folds") and income_ok():
    checks_of("MP3.3")
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    with wb.attempt("MP3.3"):
        train, val = folds[0]
        prep = income.build_preprocessor("onehot")
        verdict("MP3.3", isinstance(prep, ColumnTransformer), "build_preprocessor rend un ColumnTransformer.",
                f"build_preprocessor doit rendre un ColumnTransformer, pas un {type(prep).__name__}.")
        Z = prep.fit_transform(X_train.iloc[train])
        names = list(prep.get_feature_names_out())
        verdict("MP3.3", isinstance(Z, np.ndarray) and not np.isnan(Z).any(),
                f"« onehot » : une matrice dense sans NaN, {Z.shape[1]} colonnes.",
                "le prétraitement « onehot » doit rendre un tableau NumPy dense sans valeur manquante.")
        verdict("MP3.3", not {"fnlwgt", "education"} & set(names) and not any(n.startswith("education_") for n in names),
                "fnlwgt et education sont écartés.", "fnlwgt et education ne doivent pas entrer dans le modèle (DROPPED).")
        medians = prep.named_transformers_["numeric"].named_steps["impute"].statistics_
        verdict("MP3.3", np.allclose(medians, X_train.iloc[train][income.NUMERIC].median()),
                "les médianes sont apprises sur les lignes reçues par fit.",
                "les médianes de l'imputation doivent être celles des lignes reçues par fit.")
        pipeline = income.build_pipeline(LogisticRegression(max_iter=2000))
        verdict("MP3.3", isinstance(pipeline, Pipeline) and list(pipeline.named_steps) == ["prep", "model"],
                "build_pipeline rend un Pipeline à deux étapes, « prep » et « model ».",
                "build_pipeline doit rendre Pipeline([(\"prep\", …), (\"model\", model)]).")
        ordinal = income.build_preprocessor("ordinal").fit(X_train.iloc[train])
        Zo = ordinal.transform(X_train.iloc[val])
        verdict("MP3.3", Zo.shape[1] == len(income.CATEGORICAL) + len(income.NUMERIC)
                and np.isnan(Zo[:, :len(income.CATEGORICAL)]).sum() == X_train.iloc[val][income.CATEGORICAL].isna().sum().sum(),
                "« ordinal » : une colonne par variable, les catégorielles d'abord, les manquants restés NaN.",
                "le prétraitement « ordinal » doit garder une colonne par variable (les catégorielles d'abord) et "
                "laisser les valeurs manquantes en NaN.")
        baselines = {"baseline : la classe majoritaire": income.build_pipeline(DummyClassifier()),
                     "régression logistique": pipeline}
        cv_results = globals().get("cv_results") or {}      # kept when this cell runs again: the later steps add to it
        cv_results.update(income.cross_validate_all(baselines, X_train, y_train, folds))
        by_hand = []
        for fold_train, fold_val in folds:      # by hand: the whole pipeline fitted on the training rows of each fold
            fitted = income.build_pipeline(LogisticRegression(max_iter=2000)).fit(X_train.iloc[fold_train],
                                                                                   y_train[fold_train])
            by_hand.append(roc_auc_score(y_train[fold_val], fitted.predict_proba(X_train.iloc[fold_val])[:, 1]))
        gap = float(np.max(np.abs(np.asarray(cv_results["régression logistique"]["roc_auc"]) - by_hand)))
        verdict("MP3.3", gap < 1e-9,
                "fold par fold, le score de cross_validate_all est celui d'un pipeline entraîné sur les seules lignes "
                "d'entraînement du fold.",
                f"un pipeline entraîné à la main sur les lignes d'entraînement de chaque fold donne les ROC-AUC "
                f"{np.round(by_hand, 6).tolist()} (écart maximal avec les tiennes : {gap:.2g}) : cross_validate_all doit "
                f"passer à cross_validate les folds reçus (cv) et tout le pipeline, prétraitement compris.")
        leaky_prep = income.build_preprocessor("onehot").fit(X_train)                 # the leak, on purpose
        leaky = income.cross_validate_all({"x": Pipeline([("model", LogisticRegression(max_iter=2000))])},
                                          pd.DataFrame(leaky_prep.transform(X_train)), y_train, folds)["x"]
        print(score_table(cv_results).to_string())
        honest = cv_results["régression logistique"]
        print(f"\nla fuite (prétraitement appris sur tout l'entraînement) : PR-AUC "
              f"{np.mean(leaky[MAIN]):.5f} contre {np.mean(honest[MAIN]):.5f} dans le pipeline "
              f"(écart {np.mean(leaky[MAIN]) - np.mean(honest[MAIN]):+.5f})")
        checks_done()'''

# ---------------------------------------------------------------------------
# MP3.4
# ---------------------------------------------------------------------------
MP34_MD = r'''## MP3.4 · Les classifieurs des ch. 13 et 14 dans le même ColumnTransformer, sur les mêmes folds ⏱️ 60 min

**Objectif :** une comparaison honnête : le même prétraitement, les mêmes folds, l'incertitude de chaque score et le coût de chaque modèle.

a) Remplis `CANDIDATES` : au moins **quatre** modèles des ch. 13 et 14, d'au moins **trois familles** (les voisins, Naïve Bayes, les arbres, les ensembles, les SVM : k plus proches voisins, arbre de décision, forêt aléatoire, bagging, AdaBoost, `LinearSVC`…), chacun dans `income.build_pipeline` avec l'encodage « onehot ». Choisis leurs hyperparamètres avec ce que tu sais d'eux (la vérification n'en règle aucun) ; un SVM à noyau sur 31 000 lignes demande plusieurs minutes par fold : préfère `LinearSVC`, ou laisse-le de côté. Le notebook doit rester sous les 10 minutes.
b) La vérification les mesure sur les mêmes folds que les baselines (≈ 30 s pour la solution), puis compare chacun à la régression logistique **fold par fold**, sur le score de décision.

Dans la cellule ✍️ : quel classement ? Quels écarts dépassent le bruit des folds ? Pourquoi le Naïve Bayes gaussien est-il loin derrière sur ces colonnes ? Que coûte chaque modèle, à l'entraînement et à la prédiction ?'''

MP34_IMPORTS = r'''from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier'''

MP34_TODO = r'''CANDIDATES = ...   # name -> income.build_pipeline(model): at least four models of chapters 13 and 14'''

MP34_SOLUTION = r'''CANDIDATES = {
    "k plus proches voisins (k = 25)": income.build_pipeline(KNeighborsClassifier(n_neighbors=25)),
    "Naïve Bayes gaussien": income.build_pipeline(GaussianNB()),
    "arbre de décision (profondeur 8)": income.build_pipeline(DecisionTreeClassifier(max_depth=8, random_state=0)),
    "forêt aléatoire (200 arbres)": income.build_pipeline(
        RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0)),
}'''

MP34_CHECK = r'''if not filled(CANDIDATES):
    print("⏳ MP3.4 : remplis CANDIDATES (consigne a).")
elif ready("X_train", "y_train", "folds", "cv_results") and income_ok():
    checks_of("MP3.4")
    from sklearn.pipeline import Pipeline
    with wb.attempt("MP3.4"):
        FAMILIES = {"neighbors": "voisins", "naive_bayes": "Naïve Bayes", "tree": "arbres", "ensemble": "ensembles",
                    "svm": "SVM"}

        def family(model):
            """The family of a model, from its sklearn sub-package, looking inside a wrapper (CalibratedClassifierCV...)."""
            parts = type(model).__module__.split(".")
            kind = parts[1] if len(parts) > 1 and parts[0] == "sklearn" else type(model).__name__
            if kind in ("calibration", "multiclass"):
                inner = getattr(model, "estimator", None)
                return "svm" if inner is None and kind == "calibration" else family(inner) if inner is not None else kind
            return kind

        found = [family(p.named_steps["model"]) for p in CANDIDATES.values()
                 if isinstance(p, Pipeline) and "model" in p.named_steps]
        families = {FAMILIES[kind] for kind in found if kind in FAMILIES}
        others = sorted({kind for kind in found if kind not in FAMILIES})
        ok = isinstance(CANDIDATES, dict) and all(isinstance(p, Pipeline) and list(p.named_steps) == ["prep", "model"]
                                                   for p in CANDIDATES.values())
        verdict("MP3.4", ok, "chaque candidat est un Pipeline de build_pipeline.",
                "CANDIDATES doit associer à chaque nom un income.build_pipeline(model).")
        verdict("MP3.4", ok and len(CANDIDATES) >= 4 and len(families) >= 3,
                f"{len(CANDIDATES)} modèles, de {len(families)} familles ({', '.join(sorted(families))}).",
                f"il faut au moins quatre modèles, d'au moins trois familles des ch. 13 et 14 (voisins, Naïve Bayes, "
                f"arbres, ensembles, SVM) ; familles trouvées : {', '.join(sorted(families)) or 'aucune'}"
                + (f" (ne comptent pas : {', '.join(others)})." if others else "."))
        start = time.perf_counter()
        cv_results.update(income.cross_validate_all(CANDIDATES, X_train, y_train, folds))
        print(f"validation croisée des {len(CANDIDATES)} candidats : {time.perf_counter() - start:.0f} s\n")
        print(score_table(cv_results).to_string())
        print(f"\n{METRIC_NAMES[MAIN]}, différence fold par fold avec la régression logistique "
              "(moyenne ± erreur type, folds gagnés sur 5) :")
        for name in CANDIDATES:
            mean, se, wins = paired(cv_results, name, "régression logistique")
            print(f"  {name:<40} {mean:+.4f} ± {se:.4f}   {wins}/5")
        checks_done()'''

# ---------------------------------------------------------------------------
# MP3.5
# ---------------------------------------------------------------------------
MP35_MD = r'''## MP3.5 · Le gradient boosting : catégories natives, recherche d'hyperparamètres, choix justifié ⏱️ 60 min

**Objectif :** l'état de l'art des données tabulaires, réglé sans tricher, et un choix final qu'on peut défendre.

a) La première cellule mesure `HistGradientBoostingClassifier(random_state=0)` avec ses valeurs par défaut, de deux façons : l'encodage « ordinal » et ses catégories natives (`build_pipeline` lui dit quelles colonnes sont catégorielles), puis l'encodage « onehot ».
b) Remplis `SEARCH_SPACE` : les hyperparamètres à explorer, préfixés par `model__` (le nom de l'étape dans le pipeline), avec des listes ou des lois de `scipy.stats` (`loguniform`, `randint`). La recherche au hasard (`RandomizedSearchCV`, `N_SEARCH` candidats, ≈ 1 min) les mesure **sur les mêmes folds** que les autres modèles, avec le score de décision.
c) Remplis `FINAL` : les hyperparamètres retenus, `{}` pour les valeurs par défaut ou `SEARCHED[k]` pour le candidat de la ligne k du tableau (copier les valeurs arrondies du tableau marche aussi), et justifie ton choix. Le meilleur des `N_SEARCH` candidats est optimiste : il a gagné une compétition sur ces mêmes folds, et une partie de son avance est due au hasard (le ch. 15 le corrige par une validation croisée imbriquée). La vérification te donne une règle : le meilleur candidat ne remplace les valeurs par défaut que s'il les bat sur au moins 4 folds sur 5 **et** de plus d'une erreur type de la différence appariée.

Dans la cellule ✍️ : catégories natives ou one-hot ? Qu'apporte la recherche ? Quel modèle retiens-tu, et sur quelle règle ?'''

MP35_IMPORTS = r'''from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV'''

MP35_DEFAULTS = r'''if ready("X_train", "y_train", "folds", "cv_results") and income_ok():
    checks_of("MP3.5 défauts")
    with wb.attempt("MP3.5"):
        boosting = {"gradient boosting, catégories natives": income.build_pipeline(
                        HistGradientBoostingClassifier(random_state=0), "ordinal"),
                    "gradient boosting, one-hot": income.build_pipeline(HistGradientBoostingClassifier(random_state=0))}
        verdict("MP3.5", boosting["gradient boosting, catégories natives"].named_steps["model"].categorical_features
                == list(range(len(income.CATEGORICAL))),
                "« ordinal » : le modèle sait quelles colonnes sont catégorielles.",
                f"avec l'encodage « ordinal », build_pipeline doit donner au modèle categorical_features = "
                f"{list(range(len(income.CATEGORICAL)))} (la docstring).")
        cv_results.update(income.cross_validate_all(boosting, X_train, y_train, folds))
        print(score_table({name: cv_results[name] for name in ["régression logistique", *boosting]}).to_string())
        mean, se, wins = paired(cv_results, "gradient boosting, catégories natives", "régression logistique")
        print(f"\nPR-AUC, gradient boosting (catégories natives) contre régression logistique : {mean:+.4f} ± {se:.4f}, "
              f"{wins}/5 folds gagnés")
        checks_done()'''

MP35_TODO_SPACE = r'''SEARCH_SPACE = ...   # "model__<hyperparameter>": a list or a scipy.stats distribution'''

MP35_SOLUTION_SPACE = r'''SEARCH_SPACE = {
    "model__learning_rate": loguniform(0.02, 0.3),
    "model__max_iter": randint(100, 500),
    "model__max_leaf_nodes": randint(8, 64),
    "model__min_samples_leaf": randint(10, 200),
    "model__l2_regularization": loguniform(1e-3, 10),
}'''

MP35_SEARCH = r'''if not filled(SEARCH_SPACE):
    print("⏳ MP3.5 : remplis SEARCH_SPACE (consigne b).")
elif ready("X_train", "y_train", "folds", "cv_results", "boosting") and income_ok():
    checks_of("MP3.5 recherche")
    with wb.attempt("MP3.5"):
        base = income.build_pipeline(HistGradientBoostingClassifier(random_state=0), "ordinal")
        known = set(base.get_params())
        unknown = sorted(set(SEARCH_SPACE) - known) if isinstance(SEARCH_SPACE, dict) else ["?"]
        if verdict("MP3.5", isinstance(SEARCH_SPACE, dict) and SEARCH_SPACE and not unknown,
                   f"{len(SEARCH_SPACE)} hyperparamètres du pipeline à explorer.",
                   f"SEARCH_SPACE doit être un dict de paramètres du pipeline (préfixe model__) ; inconnus : {unknown}."):
            start = time.perf_counter()
            search = RandomizedSearchCV(base, SEARCH_SPACE, n_iter=N_SEARCH, cv=folds, scoring=MAIN,
                                        random_state=SEARCH_SEED, refit=False).fit(X_train, y_train)
            print(f"recherche au hasard, {N_SEARCH} candidats × {N_FOLDS} folds : {time.perf_counter() - start:.0f} s\n")
            table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
            splits = [f"split{k}_test_score" for k in range(N_FOLDS)]
            shown = table[["params", "mean_test_score", "std_test_score", "mean_fit_time"]].head(5).copy()
            shown.insert(0, "SEARCHED[k]", range(len(shown)))
            shown["params"] = shown["params"].map(lambda p: {k.removeprefix("model__"): (round(v, 4) if isinstance(v, float) else v)
                                                             for k, v in p.items()})
            print(shown.round(4).to_string(index=False))
            best = table.iloc[0]
            cv_results["gradient boosting, meilleur candidat de la recherche"] = {MAIN: best[splits].to_numpy(dtype=float)}
            mean, se, wins = paired(cv_results, "gradient boosting, meilleur candidat de la recherche",
                                    "gradient boosting, catégories natives")
            beats = wins >= 4 and mean > se
            print(f"\nle meilleur candidat contre les valeurs par défaut (PR-AUC, mêmes folds) : {mean:+.4f} ± {se:.4f}, "
                  f"{wins}/5 folds gagnés → la règle {'le retient' if beats else 'garde les valeurs par défaut'}")
            SEARCHED = [{k.removeprefix("model__"): v for k, v in p.items()} for p in table["params"]]
            print("pour retenir un candidat, copie-le depuis SEARCHED : FINAL = SEARCHED[k] (k = 0 pour le meilleur).")
        checks_done()'''

MP35_TODO_FINAL = r'''FINAL = ...   # the hyperparameters of HistGradientBoostingClassifier you keep ({} = the default values)'''

MP35_SOLUTION_FINAL = r'''FINAL = {}   # the best candidate does not beat the default values by the rule (MP3.5 c)'''

MP35_FINAL = r'''if not filled(FINAL):
    print("⏳ MP3.5 : remplis FINAL (consigne c).")
elif ready("X_train", "y_train", "folds", "cv_results", "SEARCHED") and income_ok():
    checks_of("MP3.5 choix")
    with wb.attempt("MP3.5"):
        def same_value(mine, measured):
            """Equal, or equal once rounded like the table (4 decimals) for a float."""
            if isinstance(measured, float):
                return (isinstance(mine, (int, float, np.number)) and not isinstance(mine, bool)
                        and (np.isclose(mine, measured, rtol=1e-6) or round(float(mine), 4) == round(measured, 4)))
            return mine == measured

        def same_params(a, b):
            """The same hyperparameters (values copied from the table, rounded to 4 decimals, are accepted)."""
            return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)

        rank = next((i for i, p in enumerate(SEARCHED) if isinstance(FINAL, dict) and same_params(FINAL, p)), None)
        known = isinstance(FINAL, dict) and (FINAL == {} or rank is not None)
        verdict("MP3.5", known, "FINAL a été mesuré sur les folds : "
                + ("les valeurs par défaut." if FINAL == {} else f"le candidat SEARCHED[{rank}] de la recherche."
                   if rank is not None else ""),
                "FINAL doit être {} (les valeurs par défaut) ou les paramètres d'un candidat de la recherche "
                "(sans le préfixe model__) : un modèle jamais mesuré sur les folds ne peut pas être retenu.")
        if known:
            final_pipeline = income.build_pipeline(HistGradientBoostingClassifier(random_state=0, **(FINAL or {})),
                                                   "ordinal")
        else:
            globals().pop("final_pipeline", None)      # no final model until FINAL is a measured one
        final_name = "gradient boosting retenu"
        if known and FINAL == {}:
            cv_results[final_name] = cv_results["gradient boosting, catégories natives"]
        elif known:
            cv_results[final_name] = income.cross_validate_all({final_name: final_pipeline}, X_train, y_train, folds)[final_name]
        names = [name for name in cv_results if name != "gradient boosting, meilleur candidat de la recherche"
                 and (name != final_name or FINAL != {}) and "roc_auc" in cv_results[name]]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for ax, metric in zip(axes, ("average_precision", "roc_auc")):
            means = [np.mean(cv_results[n][metric]) for n in names]
            stds = [np.std(cv_results[n][metric], ddof=1) for n in names]
            ax.barh(names, means, xerr=stds, color=["C1" if "gradient" in n else "C0" for n in names])
            low = min(m for m, n in zip(means, names) if "baseline" not in n)
            ax.set(xlabel=f"{METRIC_NAMES[metric]} (5 folds, bar: ± 1 standard deviation)", xlim=(low - 0.05, 1.0) if
                   metric == "roc_auc" else (0.2, 0.9), title=METRIC_NAMES[metric])
            ax.grid(True, axis="x", alpha=0.4)
        axes[0].invert_yaxis()
        fig.suptitle("Cross-validation of every model, on the same 5 folds")
        fig.tight_layout()
        save(fig, "comparaison_modeles.png")
        plt.show()
        checks_done()'''

# ---------------------------------------------------------------------------
# MP3.6
# ---------------------------------------------------------------------------
MP36_MD = r'''## MP3.6 · Des métriques adaptées au déséquilibre, le seuil de décision, le test ouvert une fois ⏱️ 40 min

**Objectif :** un seuil choisi sur les folds, puis le score honnête du modèle retenu.

a) La première cellule calcule les **prédictions hors fold** du modèle retenu et de la régression logistique (`cross_val_predict` sur les mêmes folds : chaque personne est prédite par le modèle du fold qui ne l'a pas vue) et trace leurs courbes precision-recall et ROC (`figures/courbes_pr_roc.png`).
b) Écris `f1_threshold` dans `income.py`. La cellule compare trois seuils sur les prédictions hors fold : 0,5, celui du meilleur F1 (`best_threshold`) et celui qui prédit autant de revenus élevés qu'il y en a (`share_threshold`, le quantile $1 - \text{prévalence}$ des probabilités). Remplis `THRESHOLD`, le seuil que tu déploies, et justifie-le par l'usage : une imputation qui gonfle la part des revenus élevés fausse les statistiques qu'on en tire.
c) La dernière cellule entraîne le modèle retenu sur toutes les personnes d'entraînement et **ouvre le coffre**, une fois : ROC-AUC, PR-AUC, F1 et accuracy au seuil choisi, chacun avec son intervalle bootstrap à 95 % (`mylearn.stats.bootstrap_ci` de ta librairie), à côté de la baseline. Le coffre reste fermé tant qu'un garde-fou des étapes MP3.1 à MP3.5 échoue.

Dans la cellule ✍️ : laquelle de la PR-AUC ou de la ROC-AUC sépare le plus les modèles, et pourquoi ? Que change le seuil retenu face à 0,5 ? Le test tombe-t-il dans ce qu'annonçait la validation croisée ?'''

MP36_OOF = r'''if ready("X_train", "y_train", "folds", "final_pipeline") and income_ok():
    checks_of("MP3.6 hors fold")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    from sklearn.model_selection import cross_val_predict
    with wb.attempt("MP3.6"):
        oof = cross_val_predict(final_pipeline, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
        oof_logreg = cross_val_predict(income.build_pipeline(LogisticRegression(max_iter=2000)), X_train, y_train,
                                       cv=folds, method="predict_proba")[:, 1]
        for name, proba in (("retenu", oof), ("régression logistique", oof_logreg)):
            print(f"{name:<22} hors fold : PR-AUC {average_precision_score(y_train, proba):.4f}, "
                  f"ROC-AUC {roc_auc_score(y_train, proba):.4f}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, proba, color in (("selected gradient boosting", oof, "C1"), ("logistic regression", oof_logreg, "C0")):
            precision, recall, _ = precision_recall_curve(y_train, proba)
            axes[0].plot(recall, precision, color=color, label=f"{name} (AP {average_precision_score(y_train, proba):.3f})")
            fpr, tpr, _ = roc_curve(y_train, proba)
            axes[1].plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(y_train, proba):.3f})")
            pred = proba >= 0.5
            axes[0].plot(pred[y_train == 1].mean(), y_train[pred].mean(), "o", color=color, ms=7)
            axes[1].plot(pred[y_train == 0].mean(), pred[y_train == 1].mean(), "o", color=color, ms=7)
        axes[0].axhline(y_train.mean(), color="gray", ls="--", label=f"constant classifier ({y_train.mean():.3f})")
        axes[0].set(xlabel="recall", ylabel="precision", title="Precision-recall (out-of-fold; dot: threshold 0.5)",
                    xlim=(0, 1), ylim=(0, 1.02))
        axes[1].plot([0, 1], [0, 1], color="gray", ls="--", label="constant classifier")
        axes[1].set(xlabel="false positive rate", ylabel="true positive rate (recall)",
                    title="ROC (out-of-fold; dot: threshold 0.5)")
        for ax in axes:
            ax.grid(True, alpha=0.4)
            ax.legend(fontsize=8, loc="lower left" if ax is axes[0] else "lower right")
        fig.tight_layout()
        save(fig, "courbes_pr_roc.png")
        plt.show()
        checks_done()'''

MP36_THRESHOLD = r'''if ready("oof", "y_train") and income_ok():
    checks_of("MP3.6 seuil")
    from sklearn.metrics import f1_score, precision_score, recall_score
    with wb.attempt("MP3.6"):
        best_threshold, best_f1 = income.f1_threshold(y_train, oof)
        candidates = np.unique(oof)
        grid = candidates[:: max(1, len(candidates) // 400)]              # a brute force on 400 thresholds
        brute = max(f1_score(y_train, oof >= t) for t in grid)
        verdict("MP3.6", best_f1 >= brute - 1e-12 and np.isclose(best_f1, f1_score(y_train, oof >= best_threshold)),
                f"f1_threshold : seuil {best_threshold:.4f}, F1 hors fold {best_f1:.4f} (le meilleur d'une recherche "
                f"sur {len(grid)} seuils : {brute:.4f}).",
                f"f1_threshold doit rendre le seuil du meilleur F1 et ce F1 : une recherche sur {len(grid)} seuils trouve "
                f"déjà F1 = {brute:.4f}, et le F1 rendu doit être celui de proba >= seuil.")
        tie = income.f1_threshold(np.array([0, 0, 1, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        verdict("MP3.6", np.allclose([float(v) for v in tie], [0.3, 2 / 3]),
                "à égalité de F1, f1_threshold rend le plus petit seuil.",
                f"sur y = [0, 0, 1, 0, 0, 1] et proba = [0,1 ; 0,2 ; … ; 0,6], les seuils 0,3 et 0,6 donnent tous deux "
                f"F1 = 2/3 : on attend (0.3, 0.6667), le plus petit des deux (la docstring) ; ta fonction rend {tie!r}.")
        share_threshold = float(np.quantile(oof, 1 - y_train.mean()))   # as many predicted positives as true ones
        rows = {}
        for label, t in (("0,5", 0.5), (f"meilleur F1 ({best_threshold:.3f})", best_threshold),
                         (f"même part que la prévalence ({share_threshold:.3f})", share_threshold)):
            pred = oof >= t
            rows[label] = {"precision": precision_score(y_train, pred), "recall": recall_score(y_train, pred),
                           "F1": f1_score(y_train, pred), "taux de sélection": pred.mean(), "prévalence": y_train.mean()}
        print(pd.DataFrame(rows).T.round(4).to_string())
        checks_done()'''

MP36_TODO = r'''THRESHOLD = ...   # the threshold you deploy: best_threshold, share_threshold or another one, justified by the use'''

MP36_SOLUTION = r'''THRESHOLD = best_threshold   # the best F1 out of fold, chosen here before the vault opens (see the observations)'''

MP36_OPEN = r'''if not filled(THRESHOLD):
    print("⏳ MP3.6 : remplis THRESHOLD (consigne b).")
elif ready("X_train", "y_train", "vault", "final_pipeline", "oof", "cv_results") and income_ok():
    checks_of("MP3.6 test")
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
    with wb.attempt("MP3.6"):
        verdict("MP3.6", isinstance(THRESHOLD, (float, int, np.floating)) and 0 < float(THRESHOLD) < 1,
                f"le seuil retenu : {float(THRESHOLD):.4f}." if isinstance(THRESHOLD, (float, int, np.floating)) else "",
                "THRESHOLD doit être un nombre strictement entre 0 et 1.")
        blocking = sorted(cell for cell, failures in FAILED.items() if failures and cell < "MP3.6 test")
        if blocking:
            print(f"⏳ MP3.6 : le coffre reste fermé tant que des garde-fous échouent ({', '.join(blocking)}) : "
                  f"corrige-les, relance ces cellules, puis celle-ci.")
        else:
            final_pipeline.fit(X_train, y_train)
            params = final_pipeline.named_steps["model"].get_params()
            def plain(value):
                """3.0, np.float64(3.0) and 3 written the same way, so that one model has one description."""
                return round(float(value), 6) if isinstance(value, (float, int, np.number)) and not isinstance(value, bool) else value

            description = (f"HistGradientBoostingClassifier({', '.join(f'{k}={plain(v)!r}' for k, v in sorted((FINAL or {}).items()))})"
                           f", encoding ordinal, threshold {float(THRESHOLD):.4f}")
            X_test, y_test = vault.open(description)                       # the only opening of the vault
            proba_test = final_pipeline.predict_proba(X_test)[:, 1]
            pred_test = (proba_test >= THRESHOLD).astype(int)
            pairs = np.column_stack([y_test, proba_test])

            def with_ci(score):
                value = float(score(y_test, proba_test))
                low, high = mylearn.stats.bootstrap_ci(pairs, statistic=lambda a: float(score(a[:, 0], a[:, 1])),
                                                       n_boot=N_BOOT, rng=np.random.default_rng(0))
                return {"value": round(value, 4), "ci95": [round(low, 4), round(high, 4)]}

            test_scores = {
                "roc_auc": with_ci(roc_auc_score),
                "average_precision": with_ci(average_precision_score),
                "f1": with_ci(lambda y, p: f1_score(y, p >= THRESHOLD)),
                "accuracy": with_ci(lambda y, p: accuracy_score(y, p >= THRESHOLD)),
                "selection_rate": round(float(pred_test.mean()), 4), "prevalence": round(float(y_test.mean()), 4),
            }
            dummy = DummyClassifier().fit(X_train, y_train)
            baseline_test = {"accuracy": round(float(accuracy_score(y_test, dummy.predict(X_test))), 4),
                             "average_precision": round(float(y_test.mean()), 4), "roc_auc": 0.5, "f1": 0.0}
            TEST_NAMES = {"roc_auc": "ROC-AUC", "average_precision": "PR-AUC", "f1": "F1 (seuil retenu)",
                          "accuracy": "accuracy (seuil retenu)"}
            rows = {TEST_NAMES[m]: {"validation croisée (5 folds)": f"{np.mean(cv_results['gradient boosting retenu'][m]):.4f} "
                                      f"± {np.std(cv_results['gradient boosting retenu'][m], ddof=1):.4f}"
                                      if m in ("roc_auc", "average_precision") else "",
                                      "test": f"{test_scores[m]['value']:.4f}",
                                      "intervalle bootstrap à 95 %": f"[{test_scores[m]['ci95'][0]:.4f} ; {test_scores[m]['ci95'][1]:.4f}]",
                                      "baseline sur le test": f"{baseline_test[m]:.4f}"}
                    for m in ("roc_auc", "average_precision", "f1", "accuracy")}
            print(f"\nTEST ({fmt_int(len(y_test))} personnes), seuil {float(THRESHOLD):.4f} :")
            print(pd.DataFrame(rows).T.to_string())
            print(f"taux de sélection {test_scores['selection_rate']:.4f} pour une prévalence de {test_scores['prevalence']:.4f}")
        checks_done()'''

# ---------------------------------------------------------------------------
# MP3.7
# ---------------------------------------------------------------------------
MP37_MD = r'''## MP3.7 · L'audit : les erreurs par sous-groupe, le retrait de sex, l'importance par permutation ⏱️ 60 min

**Objectif :** savoir pour qui le modèle se trompe, et de quelle façon, avant d'écrire la model card.

a) Écris `age_groups` et `subgroup_report` dans `income.py` (docstrings) : pour chaque groupe, l'effectif, la prévalence, le taux de sélection, l'accuracy, le TPR, le FPR, la precision et la ROC-AUC, avec `confusion_matrix` et `roc_auc` de **ta** librairie `mylearn.metrics`.
b) La vérification audite les prédictions du test (le modèle ne change plus : le coffre est ouvert) selon le sexe, l'âge et la catégorie raciale du recensement, et trace `figures/audit_sous_groupes.png`. Les effectifs comptent : un taux mesuré sur 80 personnes a un intervalle large.
c) **Prédis d'abord** : sans les colonnes `sex` et `relationship` (qui code « Husband » et « Wife »), l'écart de taux de sélection entre les hommes et les femmes se réduit-il de plus de moitié ? Remplis `PREDICTION` (`True` ou `False`), puis exécute l'expérience. Elle se fait **sur les folds**, avec des prédictions hors fold, jamais sur le test : comparer des modèles sur le test serait l'ouvrir une deuxième fois.
d) La dernière cellule mesure l'**importance par permutation** de chaque colonne d'origine sur le test (`sklearn.inspection.permutation_importance`, la PR-AUC) : la baisse du score quand on mélange cette colonne. `fnlwgt` et `education`, écartées par le pipeline, servent de contrôle. Elle écrit ensuite `results.json`.

Dans la cellule ✍️ : les écarts entre femmes et hommes (TPR, FPR, taux de sélection), et la part qui vient de la prévalence ; ce que montre le retrait de `sex` ; les colonnes qui comptent, et pourquoi des colonnes corrélées se partagent l'importance ; ce que tu écriras dans la model card.'''

MP37_AUDIT = r'''if ready("y_test", "proba_test", "pred_test", "X_test") and income_ok():
    checks_of("MP3.7")
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix, roc_auc_score
    with wb.attempt("MP3.7"):
        bands = np.asarray(income.age_groups(X_test["age"]))
        expected = np.asarray(pd.cut(X_test["age"], [-np.inf, 25, 35, 45, 55, 65, np.inf], right=False,
                                     labels=income.AGE_LABELS).astype(str))
        verdict("MP3.7", np.array_equal(bands.astype(str), expected), "age_groups : les bonnes tranches d'âge.",
                f"age_groups doit couper aux âges {income.AGE_BINS} (25 ans est dans « 25-34 »).")
        audits = {column: income.subgroup_report(y_test, pred_test, proba_test, groups)
                  for column, groups in (("sex", X_test["sex"]), ("age", bands), ("race", X_test["race"]))}
        audits["age"] = audits["age"].reindex([g for g in income.AGE_LABELS if g in audits["age"].index])
        report = audits["sex"]
        ok = isinstance(report, pd.DataFrame) and list(report.columns) == income.REPORT_COLUMNS
        for group in (report.index if ok else []):
            mask = (X_test["sex"] == group).to_numpy()
            (tn, fp), (fn, tp) = sk_confusion_matrix(y_test[mask], pred_test[mask], labels=[0, 1])
            truth = [mask.sum(), (tp + fn) / mask.sum(), (tp + fp) / mask.sum(), (tp + tn) / mask.sum(), tp / (tp + fn),
                     fp / (fp + tn), tp / (tp + fp), roc_auc_score(y_test[mask], proba_test[mask])]
            ok = ok and np.allclose(report.loc[group].to_numpy(dtype=float), truth)
        verdict("MP3.7", ok, "subgroup_report redonne, groupe par groupe, les taux de scikit-learn.",
                f"subgroup_report doit rendre les colonnes {income.REPORT_COLUMNS}, avec les taux de chaque groupe "
                f"entre 0 et 1 (compare ta ligne « Female » à confusion_matrix de scikit-learn sur les seules femmes).")
        verdict("MP3.7", all(list(audits[c].index) == sorted(audits[c].index) for c in ("sex", "race")),
                "les groupes sont rangés par nom.", "subgroup_report doit ranger les groupes par nom (la docstring).")
        overall = income.subgroup_report(y_test, pred_test, proba_test, np.full(len(y_test), "toutes"))
        for column, table in audits.items():
            print(f"\n— selon {column} —")
            print(pd.concat([table, overall]).round(3).to_string())
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})
        rates = [("prevalence", "prevalence"), ("selection_rate", "selection rate"), ("tpr", "TPR (recall)"),
                 ("fpr", "FPR")]
        for ax, column in zip(axes, ("sex", "age")):
            table = audits[column]
            x = np.arange(len(table))
            for k, (key, label) in enumerate(rates):
                ax.bar(x + (k - 1.5) * 0.2, table[key], width=0.2, label=label)
            ax.set_xticks(x, [f"{g}\n(n = {fmt_int(int(n))})" for g, n in zip(table.index, table["n"])], fontsize=8)
            ax.set(title=f"Test set, by {column} (threshold {float(THRESHOLD):.3f})", ylim=(0, 1))
            ax.grid(True, axis="y", alpha=0.4)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        save(fig, "audit_sous_groupes.png")
        plt.show()
        checks_done()'''

MP37_TODO = r'''PREDICTION = ...   # True or False: without sex and relationship, does the men-women gap in selection rate shrink by more than half?'''

MP37_SOLUTION = r'''PREDICTION = False   # the other columns (marital-status, occupation, hours-per-week...) carry most of the gap'''

MP37_WITHOUT = r'''if not filled(PREDICTION):
    print("⏳ MP3.7 : note d'abord ta prédiction dans PREDICTION (consigne c) : l'expérience attend.")
elif ready("X_train", "y_train", "folds", "oof") and income_ok():
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import cross_val_predict
    with wb.attempt("MP3.7"):
        women = (X_train["sex"] == "Female").to_numpy()
        rows = {}
        variants = {"toutes les colonnes": income.CATEGORICAL,
                    "sans sex": [c for c in income.CATEGORICAL if c != "sex"],
                    "sans sex ni relationship": [c for c in income.CATEGORICAL if c not in ("sex", "relationship")]}
        for name, categorical in variants.items():
            proba = oof if name == "toutes les colonnes" else cross_val_predict(
                income.build_pipeline(HistGradientBoostingClassifier(random_state=0, **(FINAL or {})), "ordinal",
                                      categorical=categorical), X_train, y_train, cv=folds, method="predict_proba")[:, 1]
            threshold, _ = income.f1_threshold(y_train, proba)
            pred = proba >= threshold
            rows[name] = {"PR-AUC hors fold": average_precision_score(y_train, proba), "seuil (meilleur F1)": threshold,
                          "sélection hommes": pred[~women].mean(), "sélection femmes": pred[women].mean(),
                          "écart de sélection": pred[~women].mean() - pred[women].mean(),
                          "écart de TPR": pred[~women & (y_train == 1)].mean() - pred[women & (y_train == 1)].mean()}
        table = pd.DataFrame(rows).T
        print(table.round(4).to_string())
        print(f"\nprévalence : hommes {y_train[~women].mean():.4f}, femmes {y_train[women].mean():.4f} "
              f"(écart {y_train[~women].mean() - y_train[women].mean():.4f})")
        before, after = table.loc["toutes les colonnes", "écart de sélection"], table.loc["sans sex ni relationship", "écart de sélection"]
        happened = bool(after < before / 2)
        print(f"sans sex ni relationship, l'écart de sélection passe de {before:.4f} à {after:.4f} ({after / before - 1:+.0%}) : "
              f"ta prédiction ({PREDICTION}) est {'juste' if bool(PREDICTION) == happened else 'fausse'}.")'''

MP37_IMPORTANCE = r'''if ready("X_test", "y_test", "final_pipeline", "test_scores", "audits") and income_ok():
    from sklearn.inspection import permutation_importance
    with wb.attempt("MP3.7"):
        start = time.perf_counter()
        importance = permutation_importance(final_pipeline, X_test, y_test, scoring=MAIN, n_repeats=N_REPEATS,
                                            random_state=0)
        order = np.argsort(importance.importances_mean)
        print(f"importance par permutation, {N_REPEATS} répétitions : {time.perf_counter() - start:.0f} s")
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.boxplot(importance.importances[order].T, vert=False, tick_labels=np.asarray(X_test.columns)[order])
        ax.axvline(0, color="gray", lw=1)
        ax.set(xlabel=f"drop of the PR-AUC on the test set when the column is shuffled ({N_REPEATS} repeats)",
               title="Permutation importance of the original columns")
        ax.grid(True, axis="x", alpha=0.4)
        fig.tight_layout()
        save(fig, "importance_permutation.png")
        plt.show()
        importances = {column: round(float(importance.importances_mean[i]), 4) for i, column in enumerate(X_test.columns)}
        print(pd.Series(importances).sort_values(ascending=False).to_string())
        results = {
            "project": "MP3 - Income over 50K from the 1994 census: a complete tabular pipeline and a fairness audit",
            "data": {"people": int(len(y_all)), "train": int(len(y_train)), "test": int(len(y_test)), "test_size": 0.2,
                     "positive": data.POSITIVE, "prevalence_train": round(float(y_train.mean()), 4)},
            "seeds": {"test": TEST_SEED, "folds": FOLD_SEED, "search": SEARCH_SEED, "bootstrap": 0, "permutation": 0},
            "n_folds": N_FOLDS, "decision_score": MAIN, "n_search": N_SEARCH,
            "chosen": {"model": "HistGradientBoostingClassifier", "encoding": "ordinal", "params": FINAL or {},
                       "threshold": round(float(THRESHOLD), 4), "dropped_columns": income.DROPPED},
            "cross_validation": {name: {metric: {"folds": np.round(values, 4).tolist(), "mean": round(float(np.mean(values)), 4),
                                                 "std": round(float(np.std(values, ddof=1)), 4)}
                                        for metric, values in scores.items()} for name, scores in cv_results.items()},
            "test": {**test_scores, "baseline": baseline_test, "n_boot": N_BOOT},
            "audit_test": {column: table.round(4).reset_index().to_dict(orient="records") for column, table in audits.items()},
            "permutation_importance_test": importances,
            "vault_openings": vault.openings(),
            "fast_mode": bool(FAST_MODE),
        }
        try:
            results["versions"] = {**income.library_versions(), "mylearn": MYLEARN_SOURCE}
        except NotImplementedError:                    # written in MP3.8
            results["versions"] = {"scikit-learn": sklearn.__version__, "mylearn": MYLEARN_SOURCE}
        (PROJECT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("💾 results.json écrit.")'''

# ---------------------------------------------------------------------------
# MP3.8
# ---------------------------------------------------------------------------
MP38_MD = r'''## MP3.8 · Emballer : tests, modèle sauvegardé, README, model card, data card, commit ⏱️ 60 min

**Objectif :** un projet qu'un recruteur peut lire, vérifier, relancer, et un modèle qu'on peut recharger.

a) **Les tests** : complète `test_income.py` jusqu'à au moins **six** tests, tous verts (les idées sont dans l'en-tête du fichier), dont au moins un sur l'absence de fuite et un sur la reproductibilité.
b) **Le modèle** : écris `library_versions`, `save_model` et `load_model` dans `income.py`. La vérification sauvegarde le pipeline retenu dans `model.joblib`, avec le seuil, les versions et les scores, puis le recharge et vérifie qu'il prédit la même chose.
c) **Les documents** : remplace chaque paragraphe « TODO » de `README.md`, de `MODEL_CARD.md` et de `DATA_CARD.md` par ton texte et tes chiffres. Tu peux les écrire en anglais, avec le même plan.
d) **La propreté** : redémarre le noyau, exécute tout le notebook (quelques minutes), relis les sorties (le coffre doit annoncer « déjà ouvert … pour ce même modèle »), puis enregistre ton travail avec git :

```bash
git add mon_travail/projets/partie_3_revenus
git commit -m "MP3: income classifier with a scikit-learn pipeline and a fairness audit"
```

`model.joblib` fait moins de 2 Mo : tu peux le commiter dans ton dépôt de portfolio (le dépôt du workbook l'ignore). La vérification ci-dessous lance tes tests et relit ton dossier. Ensuite : la grille du cahier des charges, puis, seulement après, la solution de référence (`projets/partie_3_revenus/solution/`).'''

MP38_CHECK = r'''if ready("final_pipeline", "vault", "test_scores", "results") and income_ok():
    checks_of("MP3.8")
    with wb.attempt("MP3.8"):
        versions = income.library_versions()
        verdict("MP3.8", isinstance(versions, dict) and versions.get("scikit-learn") == sklearn.__version__
                and {"python", "numpy", "pandas", "scikit-learn", "joblib"} <= set(versions),
                f"library_versions : scikit-learn {versions.get('scikit-learn')}, Python {versions.get('python')}.",
                "library_versions doit rendre les versions de python, numpy, pandas, scikit-learn et joblib.")
        metadata = {"threshold": round(float(THRESHOLD), 4), "params": FINAL or {}, "encoding": "ordinal",
                    "columns": list(X_train.columns), "test": {k: test_scores[k] for k in ("roc_auc", "average_precision")}}
        path = income.save_model(final_pipeline, PROJECT / "model.joblib", metadata)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded, info = income.load_model(path)
        same = np.allclose(loaded.predict_proba(X_train.iloc[:500])[:, 1], final_pipeline.predict_proba(X_train.iloc[:500])[:, 1])
        verdict("MP3.8", same and info.get("metadata", {}).get("threshold") == metadata["threshold"],
                f"model.joblib ({(PROJECT / 'model.joblib').stat().st_size / 1e6:.1f} Mo) : rechargé, il prédit la même "
                f"chose et garde son seuil.",
                "rechargé avec load_model, le pipeline doit prédire les mêmes probabilités et garder ses métadonnées.")
        results["versions"] = {**versions, "mylearn": MYLEARN_SOURCE}
        results["model_file"] = {"name": "model.joblib", "saved_with": info.get("versions")}
        (PROJECT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        checks_done()
    summary = run_project_tests()
    passed = int(re.search(r"(\d+) passed", summary).group(1)) if re.search(r"(\d+) passed", summary) else 0
    verdict("MP3.8", not re.search(r"failed|error", summary) and passed >= 6, f"{passed} tests, tous verts.",
            f"il faut au moins 6 tests, tous verts ({passed} vert(s) pour l'instant).")
    readme = check_document("MP3.8", PROJECT / "README.md", {
        "Le problème": ("problème", "probleme", "problem"), "Les données": ("données", "donnees", "data"),
        "La démarche": ("démarche", "demarche", "méthode", "approach"), "Les résultats": ("résultat", "resultat", "result"),
        "Les limites": ("limite", "limit")})
    for figure in ("exploration.png", "comparaison_modeles.png", "courbes_pr_roc.png", "audit_sous_groupes.png",
                   "importance_permutation.png"):
        verdict("MP3.8", f"figures/{figure}" in readme and (FIGURES / figure).exists(), f"README : figures/{figure}.",
                f"README : affiche figures/{figure} (le notebook la crée).")
    check_document("MP3.8", PROJECT / "MODEL_CARD.md", {
        "Usage prévu": ("usage", "intended"), "Les données": ("données", "donnees", "data"),
        "Les métriques": ("métrique", "metrique", "metric"), "Les sous-groupes": ("sous-groupe", "subgroup"),
        "Les risques et limites": ("risque", "limite", "limit")})
    check_document("MP3.8", PROJECT / "DATA_CARD.md", {
        "La provenance": ("provenance",), "La licence": ("licence", "license"), "La composition": ("composition",),
        "Les biais": ("biais", "bias")})
    state, detail = committed(PROJECT / "test_indices.npy")
    verdict("MP3.8", bool(state), f"test_indices.npy est commité ({detail}).", f"test_indices.npy {detail}.")
    descriptions = {opening["description"] for opening in vault.openings()}
    verdict("MP3.8", len(descriptions) == 1, "le coffre n'a été ouvert que pour un seul modèle.",
            f"le coffre a été ouvert pour {len(descriptions)} modèles (vault.json) : le score du test n'est plus une "
            f"mesure honnête, dis-le dans ton README.")'''

NOTES = {   # step -> (questions for the learner, observations of the reference solution)
    "MP3.1": ("""- L'usage du modèle, qui l'utilise, et ce que coûtent un faux positif et un faux négatif :
- Pourquoi une erreur qui pèse plus sur un groupe fausse les statistiques de ce groupe :
- Pourquoi un découpage stratifié :""", r"""- **L'usage** : un institut de statistique complète la tranche de revenu (plus ou moins de 50 000 $) des personnes qui n'ont pas répondu à cette question. La prédiction ne décide rien pour une personne : elle entre dans des statistiques (la part des revenus élevés par région, par métier, par sexe). Un faux positif gonfle la part des hauts revenus, un faux négatif la réduit : dans une imputation, il faut surtout que les erreurs ne penchent pas d'un côté, globalement et dans chaque groupe.
- **Une erreur qui pèse plus sur un groupe** : si le modèle sous-estime les revenus élevés des femmes plus que ceux des hommes, la part des femmes au-dessus de 50 000 $ calculée après imputation est biaisée vers le bas, et l'écart entre les sexes paraît plus grand qu'il n'est. L'audit de MP3.7 mesure ce risque.
- **Stratifié** : 23,9 % de revenus élevés. Un tirage au hasard de 9 769 personnes donnerait une part qui varie d'environ 0,4 point (écart-type $\sqrt{0{,}239 \times 0{,}761 / 9\,769} \approx 0{,}0043$) ; la stratification la fixe à 23,93 % dans les deux parties : le score du test est un peu moins bruité, et comparable à la validation croisée, stratifiée elle aussi."""),
    "MP3.2": ("""- Les colonnes à valeurs manquantes, et ce que « manquant » veut dire pour chacune :
- Les colonnes écartées, et pourquoi :
- Le déséquilibre, et ce qu'il fait à l'accuracy :
- Ce qui frappe dans les taux de revenus élevés par groupe :""", r"""- **Les valeurs manquantes** (sur les 39 073 personnes d'entraînement) : `occupation` 2 244 (5,7 %), `workclass` 2 235 (5,7 %), `native-country` 691 (1,8 %). Les deux premières vont ensemble : les 2 235 personnes sans `workclass` n'ont pas non plus d'`occupation`, et les 9 autres sans métier ont `workclass = Never-worked`. « Manquant » veut souvent dire « sans emploi déclaré » : ces personnes n'ont que 10,0 % de revenus élevés, contre 23,9 %. Ce n'est pas un trou au hasard : la catégorie `missing` de l'encodage « onehot » garde cette information, et le gradient boosting traite les NaN comme une valeur à part. Pour `native-country`, les manquants ont la même part de revenus élevés que l'ensemble (24,5 %).
- **Les colonnes écartées** : `education` est une copie exacte de `education-num` (16 niveaux, un code chacun) ; `fnlwgt` est un poids d'échantillonnage, le nombre de personnes que la ligne représente, pas une caractéristique de la personne. Il rend presque chaque ligne unique : 36 doublons exacts avec lui, 4 475 sans lui. Les doublons restent : des réponses identiques de personnes différentes sont plausibles, et les retirer changerait la distribution.
- **Le déséquilibre** : 23,9 % de revenus élevés, donc une accuracy de 76,1 % pour qui répond toujours « non ». L'accuracy seule ne dit presque rien : il faut un score centré sur la classe rare (la PR-AUC, dont la baseline vaut la prévalence, 0,239) ou insensible au déséquilibre (la ROC-AUC, la balanced accuracy).
- **Les taux par groupe** : 30,3 % de revenus élevés chez les hommes, 10,9 % chez les femmes (un tiers des personnes) ; 1,1 % avant 25 ans, 39,4 % entre 45 et 54 ans ; de 1,5 % à 7 % sans diplôme de fin d'études secondaires (`education-num` ≤ 8), 73 % pour les docteurs et les diplômés des écoles professionnelles (15 et 16) ; 44,7 % chez les personnes mariées (`Married-civ-spouse`, 46 % des personnes), 4,5 % chez les célibataires. `capital-gain` est nul pour 91,7 % des personnes, mais 61,8 % de celles qui en déclarent ont un revenu élevé : une colonne rare et très informative. Les écarts entre les sexes et le rôle du mariage annoncent l'audit : `relationship` code « Husband » (15 745 hommes, une femme) et « Wife » (1 853 femmes, 3 hommes)."""),
    "MP3.3": ("""- L'accuracy de la baseline, et le score que tu suis :
- Ce que gagne la régression logistique :
- Ce que change la fuite ici, et pourquoi un Pipeline reste indispensable :""", r"""- **La baseline** : accuracy 0,7607 (la part des « non »), balanced accuracy 0,5, F1 0, ROC-AUC 0,5, PR-AUC 0,239 (la prévalence). Le score de décision est la PR-AUC : il ne dépend pas d'un seuil (choisi en MP3.6), il se concentre sur la classe rare, et sa baseline, la prévalence, rend visible l'apport d'un modèle. La ROC-AUC sert de second avis.
- **La régression logistique** : PR-AUC 0,7647 ± 0,0061, ROC-AUC 0,9057 ± 0,0030, accuracy 0,852, soit 9 points au-dessus de la baseline, mais un F1 de 0,660 au seuil 0,5.
- **La fuite** : apprendre l'imputation et la standardisation sur tout l'entraînement, avant la validation croisée, change la PR-AUC moyenne de moins de $10^{-5}$ : des médianes et des moyennes apprises sur 39 073 lignes ou sur 31 258 sont presque les mêmes, et la cible n'y entre pas. La fuite existe mais ne se voit pas ici ; elle se verrait avec un prétraitement qui regarde la cible (un encodage par la moyenne de la cible, une sélection de variables) ou avec peu de données. Le `Pipeline` reste indispensable : il rend la validation croisée honnête par construction, quel que soit le prétraitement, et c'est le même objet, prétraitement compris, qu'on sauvegarde et qu'on déploie (MP3.8)."""),
    "MP3.4": ("""- Le classement, et les écarts qui dépassent le bruit des folds :
- Pourquoi le Naïve Bayes gaussien est loin derrière :
- Le coût de chaque modèle :""", r"""- **Le classement** (PR-AUC, mêmes folds) : forêt aléatoire 0,806, régression logistique 0,765, arbre de profondeur 8 0,752, kNN 0,736, Naïve Bayes 0,636. Fold par fold, les écarts sont nets : la forêt bat la régression logistique sur les 5 folds (+0,042 ± 0,002), les trois autres perdent sur les 5. Seul l'écart de l'arbre (−0,013 ± 0,005) est du même ordre que l'écart-type de ses scores entre folds (0,015) : sur un autre découpage, son rang pourrait changer, pas le constat d'ensemble.
- **Le Naïve Bayes gaussien** suppose les colonnes indépendantes sachant la classe, et chacune gaussienne. Ici, 77 des 82 colonnes sont les 0/1 du one-hot, loin d'une gaussienne, et beaucoup disent la même chose (`relationship`, `marital-status` et `sex`) : le modèle compte plusieurs fois la même information et donne des probabilités extrêmes. Sa balanced accuracy reste correcte (0,768), mais son accuracy tombe à 0,705 : il prédit trop de revenus élevés.
- **Le coût** : à l'entraînement, la forêt est la plus chère (environ 2,5 s par fold, avec deux cœurs), la régression logistique prend moins d'une seconde, les autres moins de 0,3 s ; à la prédiction, kNN est de loin le plus lent (près de 2 s pour 7 815 personnes, contre 0,3 s pour la forêt et moins de 0,1 s pour les autres) : il compare chaque personne aux 31 258 personnes d'entraînement."""),
    "MP3.5": ("""- Catégories natives ou one-hot :
- Ce qu'apporte la recherche, et pourquoi le meilleur candidat est optimiste :
- Le modèle retenu, et la règle qui l'a choisi :""", r"""- **Catégories natives ou one-hot** : le même score (PR-AUC 0,8288 ± 0,0071 contre 0,8293 ± 0,0077), mais un entraînement deux fois plus rapide avec les catégories natives (environ 0,6 s contre 1,3 s par fold), sur 12 colonnes au lieu de 82. Le gradient boosting bat la régression logistique sur les 5 folds (+0,064 ± 0,002), et la forêt aléatoire de 0,023.
- **La recherche** : 12 candidats tirés au hasard, chacun sur les 5 folds (moins d'une minute). Les cinq meilleurs sont à égalité (de 0,8277 à 0,8280), et aucun ne bat les valeurs par défaut (0,8288) : le meilleur perd 0,0008 ± 0,0007, sur 3 folds sur 5. Les valeurs par défaut de `HistGradientBoostingClassifier` sont déjà bonnes sur un dataset de cette taille. Le meilleur candidat est de plus optimiste : choisi parmi 12 sur ces mêmes folds, il profite de leurs fluctuations, et une validation croisée imbriquée (ch. 15) mesurerait son score sans ce biais.
- **Le modèle retenu** : `HistGradientBoostingClassifier(random_state=0)`, les valeurs par défaut, avec les catégories natives. La règle annoncée avant la recherche (battre les valeurs par défaut sur 4 folds sur 5, et de plus d'une erreur type) les garde ; à score égal, on garde aussi le modèle le plus simple à décrire."""),
    "MP3.6": ("""- PR-AUC ou ROC-AUC : laquelle sépare le plus les modèles, et pourquoi :
- Le seuil retenu, et ce qu'il change face à 0,5 :
- Le test face à la validation croisée :""", r"""- **PR-AUC ou ROC-AUC** : de la régression logistique au gradient boosting, la ROC-AUC passe de 0,906 à 0,928 (+0,022), la PR-AUC de 0,765 à 0,829 (+0,064). La ROC-AUC rapporte les faux positifs aux 76 % de revenus modestes : un faux positif de plus y pèse peu. La PR-AUC compte chaque faux positif face aux vrais positifs, qui sont rares : elle sépare mieux les modèles quand la classe d'intérêt est minoritaire.
- **Le seuil** : à 0,5, precision 0,778 et recall 0,654 (F1 0,711) ; le seuil du meilleur F1, 0,361, échange de la precision (0,690) contre du recall (0,774), pour un F1 de 0,730. Mais il prédit 26,9 % de revenus élevés pour une prévalence de 23,9 % : en imputation, il gonflerait la part des hauts revenus de 3 points. Le seuil 0,416, qui en prédit autant qu'il y en a, garde presque le même F1 (0,726). La solution a pourtant déployé le seuil du meilleur F1, fixé dans la cellule `THRESHOLD` avant l'ouverture du coffre : c'est une erreur de cadrage, que ce tableau montrait déjà, et la leçon de cette étape. Le seuil de la prévalence, choisi lui aussi sur les seules prédictions hors fold, est noté dans la model card comme le réglage à utiliser pour une imputation. Le coffre n'est pas rouvert pour autant : annoncer un nouveau seuil après avoir vu le test serait une nouvelle décision, dont le score de test ne serait plus une mesure honnête ; on la mesurerait sur un nouveau jeu de test.
- **Le test** : ROC-AUC 0,9315 [0,9267 ; 0,9366] et PR-AUC 0,8324 [0,8205 ; 0,8454], en accord avec la validation croisée (0,9278 ± 0,0029 et 0,8288 ± 0,0071, dans les intervalles du test) ; F1 0,728 et accuracy 0,860 au seuil retenu, contre 0,761 pour la baseline. Le taux de sélection du test, 27,5 % pour une prévalence de 23,9 %, confirme l'excès de revenus élevés prédits par ce seuil."""),
    "MP3.7": ("""- Les écarts entre femmes et hommes (TPR, FPR, taux de sélection), et la part qui vient de la prévalence :
- Ce que montre le retrait de sex et de relationship :
- Les colonnes qui comptent, et pourquoi des colonnes corrélées se partagent l'importance :
- Ce que tu écris dans la model card :""", r"""- **Femmes et hommes** (test, seuil 0,361) : TPR 0,695 chez les femmes, 0,798 chez les hommes ; FPR 0,030 et 0,171 ; taux de sélection 10,3 % et 36,3 % pour des prévalences de 10,9 % et 30,6 %. Le modèle sous-prédit un peu les revenus élevés des femmes et sur-prédit nettement ceux des hommes. Une partie des écarts de TPR et de FPR vient de la prévalence : là où les positifs sont rares, un même seuil donne moins de faux positifs et rate plus de vrais positifs. La ROC-AUC est même meilleure chez les femmes (0,957 contre 0,911) : le classement y est plus facile, mais les erreurs ne penchent pas du même côté. Pour une imputation, ce sont les taux de sélection face aux prévalences qui comptent : l'écart entre les sexes prédit (26,0 points) dépasse le vrai (19,7 points).
- **L'âge et les petits groupes** : avant 25 ans, 1 % de revenus élevés, et le modèle n'en retrouve que 4 sur 17 (TPR 0,235) ; entre 45 et 54 ans, il prédit 48,2 % de revenus élevés pour 39,3 %. Les groupes « Amer-Indian-Eskimo » (90 personnes, 10 revenus élevés) et « Other » (75) sont trop petits pour conclure : un TPR mesuré sur 10 personnes a un intervalle de plusieurs dizaines de points.
- **Le retrait de `sex` et de `relationship`** (hors fold, chaque variante avec son seuil du meilleur F1) : la PR-AUC ne bouge pas (0,8286, 0,8288, 0,8275), ni l'écart de sélection entre hommes et femmes (0,244, 0,234, 0,247, pour un écart de prévalence de 0,194). Les autres colonnes (la situation matrimoniale, le métier, les heures travaillées, les gains en capital) portent la même information : un modèle « aveugle » au sexe reproduit les mêmes écarts. Retirer un attribut sensible ne rend pas un modèle équitable ; le garder permet au moins de mesurer les écarts, et d'en corriger certains (par exemple avec un seuil par groupe, qui pose ses propres questions : chapitre B6).
- **L'importance par permutation** (baisse de la PR-AUC du test) : `capital-gain` domine (0,261), loin devant `age` (0,062), `marital-status` (0,059), `education-num` (0,059), `occupation` et `capital-loss` (0,041) ; `sex` (0,0015) et `race` (0,0009) ne comptent presque pas, et les contrôles `fnlwgt` et `education` valent exactement 0, comme il se doit pour des colonnes que le pipeline écarte. `sex` paraît inutile parce que `relationship` et `marital-status` portent déjà son information : mélanger une colonne dont l'information existe ailleurs ne coûte presque rien au modèle. L'importance par permutation mesure ce que le modèle perd sans la colonne, pas ce que la colonne « veut dire ».
- **La model card** : l'usage prévu (des statistiques agrégées, jamais une décision sur une personne), les scores et leurs intervalles, l'audit par sexe et par âge, l'excès de revenus élevés prédits au seuil du meilleur F1 et le seuil à préférer pour une imputation, les limites (des données de 1994, un seuil de 50 000 $ fixé il y a trente ans), et les usages exclus (crédit, embauche, assurance, toute décision individuelle)."""),
}


def notes_cell(step: str, exercise: bool):
    questions, observations = NOTES[step]
    if exercise:
        return md(f"✍️ **Tes observations ({step})** : double-clique sur cette cellule pour écrire tes réponses ; "
                  f"elles nourriront ton README et ta model card.\n\n{questions}")
    return md(f"✍️ **Observations ({step})**\n\n{observations}")


FOOTER = r'''## ✅ Bilan

**Grille d'évaluation** (sur 20, détaillée dans le cahier des charges) : pipeline unique sans fuite, vérifié par un test (4) · baseline et comparaison honnête des modèles en validation croisée (3) · gradient boosting réglé et justifié (3) · métriques adaptées au déséquilibre et seuil argumenté (3) · audit par sous-groupe et discussion éthique (3) · reproductibilité : graines, versions, tests (2) · README, model card et data card clairs (2).

**Pour aller plus loin** : les extensions du cahier des charges (CatBoost et ses catégories natives, la calibration des probabilités avec `CalibratedClassifierCV`, les explications SHAP du chapitre B6, les données récentes de l'enquête ACS avec `folktables`, une petite API de prédiction au chapitre B7). La suite du workbook : la partie IV, où commencent les réseaux de neurones (ch. 16).'''


def header_cells(kind: str) -> list:
    rel = STARTER if kind == "exercise" else SOLUTION
    rows = ["| Étape | Contenu | ⏱️ |", "|---|---|---|"] + [f"| {sid} | {title} | {minutes} min |"
                                                            for sid, title, minutes in STEPS]
    fast = ("**Mode rapide.** Avec `FAST_MODE = True` (la cellule de setup), la recherche d'hyperparamètres (MP3.5) "
            "essaie 12 candidats au lieu de 30, les intervalles bootstrap du test (MP3.6) reposent sur 300 "
            "rééchantillons au lieu de 1 000, et l'importance par permutation (MP3.7) sur 5 répétitions au lieu de 10 ; "
            "toutes les comparaisons gardent leurs 5 folds. Le notebook complet tourne en un peu plus de deux minutes "
            "sur un CPU à deux cœurs en mode rapide, environ quatre en mode complet ; ferme les autres notebooks pendant "
            "qu'il tourne : le gradient boosting ralentit beaucoup quand les cœurs sont partagés.")
    if kind == "exercise":
        title = "# Mini-projet MP3 · Prédire un revenu à partir du recensement : pipeline tabulaire complet et audit d'équité"
        how = ("Le cahier des charges est dans `projets/partie_3_revenus/README.md` : lis-le d'abord. Ce notebook te "
               "guide en huit étapes. Le code durable va dans **ton module** `income.py` (et ses tests, "
               "`test_income.py`) ; le notebook l'appelle, le vérifie, prend les décisions sur les folds et trace les "
               "figures. Les cellules à compléter lèvent `NotImplementedError` ou contiennent des `...` ; les "
               "vérifications affichent ✅ ou ❌ (des garde-fous, pas une note), et ⏳ tant qu'une étape n'est pas "
               "faite. Les modèles viennent de scikit-learn ; ta librairie `mylearn` sert à l'audit "
               "(`metrics.confusion_matrix`, `metrics.roc_auc`) et aux intervalles bootstrap (`stats.bootstrap_ci`).\n\n"
               "> Travaille dans **ta copie** (`mon_travail/projets/partie_3_revenus/mp3_revenus.ipynb`, créée par "
               "`python tools/start_chapter.py CP3`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = ("# Mini-projet MP3 · Prédire un revenu à partir du recensement : pipeline tabulaire complet et audit "
                 "d'équité — solution de référence (exécutée)")
        how = ("La solution de référence, exécutée avec `income.py` de ce dossier et la librairie de référence. Son "
               "jeu de test (`test_indices.npy`) a été commité seul, avant toute modélisation, et le coffre n'a été "
               "ouvert qu'une fois (`vault.json`). Tes chiffres n'ont pas à être identiques : d'autres modèles ou "
               "d'autres choix donnent d'autres valeurs, tout aussi justes s'ils sont bien mesurés. Le README de "
               "portfolio, la model card et la data card rédigés à partir de ces résultats sont dans ce dossier.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n{fast}\n\n" + "\n".join(rows))]


def build(kind: str) -> list:
    exercise = kind == "exercise"
    cells = header_cells(kind)
    cells.append(setup_cell("exercise", chapter="CP3") if exercise else setup_cell("solution"))
    cells.append(code(PROJECT_STARTER if exercise else PROJECT_SOLUTION))
    cells.append(code(TOOLS))
    cells += [md(MP31_MD), code(MP31_CHECK), code(MP31_COMMIT), notes_cell("MP3.1", exercise)]
    cells += [md(MP32_MD), code(MP32_TODO if exercise else MP32_SOLUTION), code(MP32_CHECK), code(MP32_EXPLORE),
              notes_cell("MP3.2", exercise)]
    cells += [md(MP33_MD), code(MP33_FOLDS), code(MP33_CHECK), notes_cell("MP3.3", exercise)]
    cells += [md(MP34_MD), code(MP34_IMPORTS), code(MP34_TODO if exercise else MP34_SOLUTION), code(MP34_CHECK),
              notes_cell("MP3.4", exercise)]
    cells += [md(MP35_MD), code(MP35_IMPORTS), code(MP35_DEFAULTS), code(MP35_TODO_SPACE if exercise else MP35_SOLUTION_SPACE),
              code(MP35_SEARCH), code(MP35_TODO_FINAL if exercise else MP35_SOLUTION_FINAL), code(MP35_FINAL),
              notes_cell("MP3.5", exercise)]
    cells += [md(MP36_MD), code(MP36_OOF), code(MP36_THRESHOLD), code(MP36_TODO if exercise else MP36_SOLUTION),
              code(MP36_OPEN), notes_cell("MP3.6", exercise)]
    cells += [md(MP37_MD), code(MP37_AUDIT), code(MP37_TODO if exercise else MP37_SOLUTION), code(MP37_WITHOUT),
              code(MP37_IMPORTANCE), notes_cell("MP3.7", exercise)]
    cells += [md(MP38_MD), code(MP38_CHECK), md(FOOTER)]
    return cells


def main() -> int:
    MODULE_STARTER.write_text(starter_module(MODULE_SOLUTION.read_text(encoding="utf-8")), encoding="utf-8")
    write_notebook(STARTER, build("exercise"))
    write_notebook(SOLUTION, build("solution"))
    print(f"✅ {MODULE_STARTER.relative_to(ROOT)}, {STARTER} and {SOLUTION} written ({len(STEPS)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
