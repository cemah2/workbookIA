#!/usr/bin/env python
"""Build the mini-project MP5 from a single source (used by Claude).

    python tools/chapters/build_mp5.py
    python tools/run_all_notebooks.py projets/partie_5_classifieur_cifar10/solution/mp5_classifieur_cifar10.ipynb --inplace
    python tools/run_all_notebooks.py projets/partie_5_classifieur_cifar10/depart/mp5_classifieur_cifar10.ipynb

Writes the two notebooks (depart/ and solution/mp5_classifieur_cifar10.ipynb) and the starter package
depart/cifarkit/, generated from the reference solution/cifarkit/: same constants, classes, signatures and
docstrings; ``__init__`` methods and the functions whose docstring starts with "Provided" are kept; private helpers
are removed; every other function raises NotImplementedError with the step of the notebook where it is written.
``python tools/start_chapter.py CP5`` copies depart/ into mon_travail/projets/. Steps MP5.1 to MP5.8 follow the brief
(docs/syllabus/data/cp5.json).

The reference results of the full mode (results.json, runs.csv, model.pt, model.json, the figures shown by the README,
test_log.json) come from one run of the solution notebook with FAST_MODE = False (on a CPU with 2 cores, about an hour
and twenty minutes; on a Colab GPU, a few minutes); the committed notebook is then executed in fast mode, which writes none of
them.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import ROOT, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "projets/partie_5_classifieur_cifar10"
NOTEBOOK = "mp5_classifieur_cifar10.ipynb"
STARTER = f"{FOLDER}/depart/{NOTEBOOK}"
SOLUTION = f"{FOLDER}/solution/{NOTEBOOK}"
PACKAGE_SOLUTION = ROOT / FOLDER / "solution" / "cifarkit"
PACKAGE_STARTER = ROOT / FOLDER / "depart" / "cifarkit"

STEPS = [("MP5.1", "Cadrage, protocole de validation (découpage stratifié) et baselines (MLP puis petit CNN)", 60),
         ("MP5.2", "Pipeline de données : normalisation par canal apprise sur l'entraînement, augmentation transforms.v2, Dataset et DataLoader", 60),
         ("MP5.3", "Boucle d'entraînement réutilisable : fit(), meilleur checkpoint, early stopping, planning de LR, reprise exacte, journal", 90),
         ("MP5.4", "Architectures : CNN type VGG puis blocs résiduels ; comparer stride et pooling, batchnorm et dropout", 90),
         ("MP5.5", "Recherche d'hyperparamètres ciblée (aléatoire puis zoom) sur un sous-ensemble, puis entraînement complet sur GPU Colab ; le modèle livré", 90),
         ("MP5.6", "Analyse d'erreurs et robustesse : confusions, exemples mal classés, FGSM", 60),
         ("MP5.7", "Comparaison avec un ResNet-18 pré-entraîné de torchvision (nouvelle tête seule) : coût, gain, limites", 60),
         ("MP5.8", "Livraison : README, fiche modèle, predict.py, tests et mutants, nettoyage", 90)]

# Step of the notebook where each public function of the package is written (starter package).
TODO_STEPS = {
    "stratified_split": "MP5.1", "n_parameters": "MP5.1", "mlp": "MP5.1", "flat_size": "MP5.1", "small_cnn": "MP5.1",
    "channel_stats": "MP5.2", "make_transform": "MP5.2", "CifarDataset.__len__": "MP5.2",
    "CifarDataset.__getitem__": "MP5.2", "make_loader": "MP5.2",
    "evaluate": "MP5.3", "train_one_epoch": "MP5.3", "make_optimizer": "MP5.3", "make_scheduler": "MP5.3",
    "fit": "MP5.3", "log_run": "MP5.3",
    "vgg": "MP5.4", "ResidualBlock.forward": "MP5.4", "resnet": "MP5.4", "build_model": "MP5.4",
    "save_model": "MP5.5", "load_model": "MP5.5",
    "predict_proba": "MP5.6", "fgsm": "MP5.6",
    "load_image": "MP5.8", "classify": "MP5.8",
}

STARTER_NOTE = """
Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

# ---------------------------------------------------------------------------
# The starter package, generated from the reference
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


def _docstring(node) -> str | None:
    doc = node.body[0] if node.body else None
    if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str):
        return doc.value.value
    return None


def starter_file(source: str, filename: str) -> str:
    """One starter file of cifarkit: bodies replaced by NotImplementedError, except __init__ and "Provided"
    functions."""
    tree = ast.parse(source)
    lines = source.splitlines()
    drop: set[int] = set()
    stubs: dict[int, str] = {}

    def first_line(node) -> int:
        return min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1

    def handle(node, qualname: str) -> None:
        name = node.name
        doc = _docstring(node)
        if name == "__init__" or (doc and doc.startswith("Provided")):
            return
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            drop.update(range(first_line(node), node.end_lineno))
            return
        if doc is None:
            raise ValueError(f"{filename}: {qualname} has no docstring")
        if qualname not in TODO_STEPS:
            raise ValueError(f"{filename}: {qualname} has no step in TODO_STEPS")
        doc_node = node.body[0]
        drop.update(range(doc_node.end_lineno, node.end_lineno))
        indent = " " * doc_node.col_offset
        stubs[doc_node.end_lineno - 1] = f"{indent}raise NotImplementedError(\"{qualname}\")  # TODO {TODO_STEPS[qualname]}"

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
    if filename != "__init__.py" and stubs:
        doc_end = text.index('"""', 3)
        text = text[:doc_end].rstrip("\n") + "\n" + STARTER_NOTE + text[doc_end:]
    return text


def write_starter_package() -> list[str]:
    """depart/cifarkit/*.py from solution/cifarkit/*.py; returns the names written."""
    PACKAGE_STARTER.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(PACKAGE_SOLUTION.glob("*.py")):
        (PACKAGE_STARTER / path.name).write_text(starter_file(path.read_text(encoding="utf-8"), path.name),
                                                 encoding="utf-8")
        written.append(path.name)
    defined = set()
    for name in written:
        defined |= set(re.findall(r"(?m)^\s*def (\w+)", (PACKAGE_STARTER / name).read_text(encoding="utf-8")))
    missing = [q for q in TODO_STEPS if q.split(".")[-1] not in defined]
    if missing:
        raise ValueError(f"functions missing from the starter package: {sorted(missing)}")
    return written


# ---------------------------------------------------------------------------
# Notebook cells common to both versions
# ---------------------------------------------------------------------------
PROJECT_STARTER = r'''# The project folder: your copy (mon_travail/projets/...), where cifarkit/, data.py and test_cifarkit.py live
PROJECT_NAME = "partie_5_classifieur_cifar10"
STARTER_KIT = ROOT / "projets" / PROJECT_NAME / "depart"
candidates = [Path.cwd(), ROOT / "mon_travail" / "projets" / PROJECT_NAME, STARTER_KIT]
PROJECT = next(folder for folder in candidates if (folder / "cifarkit" / "__init__.py").exists() and (folder / "data.py").exists())
FIGURES = PROJECT / "figures"
COMMIT_TEST_LOG = True    # MP5.5: each opening of the test set is committed at once (test_log.json)
if PROJECT.resolve() == STARTER_KIT.resolve():
    print("⚠️ Ce dossier est le kit de départ du dépôt, pas ta copie : lance python tools/start_chapter.py CP5, puis "
          "ouvre mon_travail/projets/partie_5_classifieur_cifar10/mp5_classifieur_cifar10.ipynb.")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import data  # noqa: E402

try:
    import cifarkit  # noqa: E402   (your package: some of its functions call your mylearn library)
except Exception as exc:                       # a syntax error in cifarkit must not stop the notebook
    cifarkit = None
    print(f"⚠️ cifarkit ne s'importe pas : {type(exc).__name__}: {exc}. Corrige-le ; l'étape MP5.1 le rechargera.")
print("Dossier du projet :", PROJECT)'''

PROJECT_SOLUTION = r'''# The project folder of the reference solution
PROJECT_NAME = "partie_5_classifieur_cifar10"
PROJECT = ROOT / "projets" / PROJECT_NAME / "solution"
FIGURES = PROJECT / "figures"
COMMIT_TEST_LOG = False   # the reference solution: its test_log.json is committed with the workbook
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import data  # noqa: E402
import cifarkit  # noqa: E402

print("Dossier du projet :", PROJECT.relative_to(ROOT))'''

TOOLS = r'''# Tools of the notebook (run this cell)
import copy
import importlib
import json
import math
import re
import tempfile
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wb.nbtools import filled

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # 🚀 on a Colab GPU, the full mode takes minutes

SEED = 0                                        # the seed of every draw of the notebook
N_VAL = 5_000                                   # the validation set, cut by stratified_split from the 50 000 images
N_VAL_USED = 2_000 if FAST_MODE else N_VAL      # in fast mode, the measures use 2 000 of them (400 for the search)
N_BASE = 10_000 if FAST_MODE else 45_000        # MP5.1 and MP5.3: the baselines
EPOCHS_BASE = 3 if FAST_MODE else 10
N_ARCH = 6_000 if FAST_MODE else 20_000         # MP5.4: the comparison of architectures
EPOCHS_ARCH = 4 if FAST_MODE else 10
N_SEARCH = 4_000 if FAST_MODE else 10_000       # MP5.5: the search of hyperparameters
EPOCHS_SEARCH = 3 if FAST_MODE else 4
N_FINAL = 12_000 if FAST_MODE else 45_000       # MP5.5: the final training
EPOCHS_FINAL = 10 if FAST_MODE else 30
N_PRETRAINED = 2_000 if FAST_MODE else 10_000   # MP5.7: images given to the pretrained ResNet-18
SIZE_PRETRAINED = 112 if FAST_MODE else 224     # their size (224 is the size of ImageNet; 112 is 4 times cheaper)
TARGET = 0.65 if FAST_MODE else 0.80            # the brief: 65 % of validation in fast mode, 80 % of test in full mode
RUNS = PROJECT / ("runs_rapide.csv" if FAST_MODE else "runs.csv")   # the experiment journal of this mode
MODEL_DIR = Path(tempfile.mkdtemp(prefix="cifarkit_rapide_")) if FAST_MODE else PROJECT   # where the final model goes
# (fast mode: a new folder for this session, so that no model of another run, or of the solution, is ever checked)

FAILED = {}         # check cell -> its guardrails that failed at its last run (the test set stays closed while any remain)
_CURRENT = [None]
UNFINISHED = "la cellule s'est arrêtée avant la fin de ses vérifications (une erreur, ou une étape pas encore faite)"


def checks_of(cell):
    """Start the guardrails of one check cell: its failures of a previous run are forgotten. Until checks_done() runs,
    the cell counts as failed."""
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


def pending(step, message):
    """Print ⏳ with the message: a step not done yet. Like a failed guardrail, it keeps the test set closed."""
    print(f"⏳ {step} : {message}")
    if _CURRENT[0] is not None:
        FAILED[_CURRENT[0]].append(message)
    return False


PENDING = "⏳ pas encore écrit"   # the first words of a test failure on a function not written yet (conftest.py)


@contextmanager
def attempt(step):
    """wb.attempt (⏳ for what is not written yet), and any other error of your code as one ❌ line, so that « Run all »
    goes on (the guardrails of the cell stay failed)."""
    try:
        with wb.attempt(step):
            yield
    except Exception as exc:
        verdict(step, False, "", f"ton code lève {type(exc).__name__}: {str(exc)[:300]} (la vérification de la cellule "
                "s'arrête là ; corrige, puis relance-la)")


def ready(*names):
    """True when the variables of the earlier steps exist; otherwise print which ones are missing."""
    missing = [name for name in names if name not in globals()]
    if missing:
        print(f"⏳ il manque {', '.join(missing)} : exécute d'abord les cellules des étapes précédentes, vérifications "
              "comprises (ce sont souvent elles qui créent ces variables).")
    return not missing


def kit_ok():
    """Import your latest saved cifarkit again (its modules, in order); False, with the reason, when it cannot be
    imported yet."""
    global cifarkit
    try:
        for name in ("cifarkit.pipeline", "cifarkit.models", "cifarkit.train", "cifarkit.predict"):
            module = sys.modules.get(name)
            importlib.reload(module) if module is not None else importlib.import_module(name)
        cifarkit = importlib.reload(sys.modules["cifarkit"]) if "cifarkit" in sys.modules else importlib.import_module("cifarkit")
    except Exception as exc:
        print(f"⚠️ cifarkit ne s'importe pas : {type(exc).__name__}: {exc}")
        return False
    return True


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
        top = git(path, "rev-parse", "--show-toplevel")
        if top is not None and top.returncode == 0:      # a repository without any commit yet
            return False, "n'est pas encore commité"
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
    """Save a figure in the figures/ folder of the project (the README shows those of the full mode)."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=110, bbox_inches="tight")


def figure_name(name):
    """The name of a figure of the README: « …_rapide.png » in fast mode, so that the full-mode figure stays."""
    return name.replace(".png", "_rapide.png") if FAST_MODE else name


def thousands(n):
    """An integer with a space between the thousands (12 345)."""
    return f"{n:,}".replace(",", " ")


def train_run(name, config, n_train, epochs, *, lr=0.1, weight_decay=5e-4, optimizer="sgd", schedule="onecycle",
              augment=True, label_smoothing=0.1, n_val=None, note="", folder=None, log=True):
    """One experiment: a new model build_model(config) (torch.manual_seed(SEED) first), the first n_train training
    images (with or without augmentation), the first n_val validation images (N_VAL_USED by default), fit(), then one
    line in the journal RUNS. Returns (model, history, seconds)."""
    n_val = n_val or N_VAL_USED
    torch.manual_seed(SEED)
    model = cifarkit.build_model(config)
    train_loader = cifarkit.make_loader(X_train[:n_train], y_train[:n_train], cifarkit.make_transform(MEAN, STD, augment),
                                        batch_size=128, shuffle=True, seed=SEED)
    val_loader = cifarkit.make_loader(X_val[:n_val], y_val[:n_val], cifarkit.make_transform(MEAN, STD), batch_size=500,
                                      shuffle=False)
    started = time.perf_counter()
    history = cifarkit.fit(model, train_loader, val_loader, epochs=epochs, lr=lr, optimizer=optimizer,
                           weight_decay=weight_decay, schedule=schedule, label_smoothing=label_smoothing,
                           folder=folder, device=DEVICE)
    seconds = time.perf_counter() - started
    best = history["best_epoch"]
    if log:
        cifarkit.log_run(RUNS, {
            "run": name, "date": time.strftime("%Y-%m-%d %H:%M"), "arch": config["arch"], "config": config,
            "n_train": n_train, "epochs": epochs, "optimizer": optimizer, "lr": lr, "weight_decay": weight_decay,
            "schedule": schedule, "augment": augment, "label_smoothing": label_smoothing, "seed": SEED,
            "params": cifarkit.n_parameters(model), "best_epoch": best, "val_acc": round(history["val_acc"][best - 1], 4),
            "val_loss": round(history["val_loss"][best - 1], 4), "seconds": round(seconds, 1), "note": note})
    return model, history, seconds


def run_project_tests(extra=(), env=None, quiet=False):
    """Run the tests of the project folder (test_cifarkit.py) and print the failures (⏳ for a function not written
    yet); return (passed, failed, reasons), reasons being the first line of each failure."""
    command = [sys.executable, "-m", "pytest", str(PROJECT / "test_cifarkit.py"), "-q", "-p", "no:cacheprovider",
               "--color=no", "-rfE", "--tb=no", *extra]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "1000", **(env or {})})
    lines = result.stdout.strip().splitlines()
    failures = [line for line in lines if line.startswith(("FAILED ", "ERROR "))]
    if not quiet:
        for line in failures[:8]:
            kind, _, rest = line.partition(" ")
            name, _, reason = rest.partition(" - ")
            where = " (erreur hors du test : import, fixture…)" if kind == "ERROR" else ""
            mark = "⏳" if reason.startswith(PENDING) else "❌"
            print(f"{mark} {name.split('::')[-1]}{where}" + (f"\n   {reason[:600]}" if reason else ""))
    summary = lines[-1] if lines else result.stderr.strip()[-300:]
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", summary))
    failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|error)", summary))
    if not quiet:
        print("pytest:", summary)
    return passed, failed, [line.partition(" - ")[2] for line in failures]


MUTANTS = {   # name -> (what the mutant breaks, its module, its function, the code of the wrong version, given `orig`)
    "evaluate": ("evaluate fait la moyenne des moyennes des mini-batches", "cifarkit.train", "evaluate",
                 "def bad(model, loader, device='cpu'):\n"
                 "    model.eval()\n"
                 "    with torch.no_grad():\n"
                 "        out = [(F.cross_entropy(model(X), y).item(), (model(X).argmax(1) == y).float().mean().item())"
                 " for X, y in loader]\n"
                 "    model.train()\n"
                 "    return float(np.mean([a for a, _ in out])), float(np.mean([b for _, b in out]))"),
    "channel_stats": ("channel_stats rend la moyenne des trois canaux pour chacun", "cifarkit.pipeline", "channel_stats",
                      "def bad(images):\n"
                      "    mean, std = orig(images)\n"
                      "    return [float(np.mean(mean))] * 3, [float(np.mean(std))] * 3"),
    "fit": ("fit rend le modèle de la dernière epoch, pas celui de la meilleure", "cifarkit.train", "fit",
            "def bad(model, *args, **kwargs):\n"
            "    history = orig(model, *args, **kwargs)\n"
            "    with torch.no_grad():\n"
            "        for p in model.parameters():\n"
            "            p.add_(1e-3)\n"
            "    return history"),
    "fgsm": ("fgsm fait un pas de 2ε au lieu de ε", "cifarkit.train", "fgsm",
             "def bad(model, images, labels, epsilon, mean, std):\n"
             "    return (images + 2 * epsilon * torch.sign(orig(model, images, labels, epsilon, mean, std) - images)).clamp(0, 1)"),
    "split": ("stratified_split tire la validation au hasard, sans stratifier", "cifarkit.pipeline", "stratified_split",
              "def bad(y, n_val, seed):\n"
              "    order = np.random.default_rng(seed).permutation(len(y))\n"
              "    return np.sort(order[n_val:]), np.sort(order[:n_val])"),
}


def mutant_plugin(folder, name):
    """A pytest plugin that swaps one function of cifarkit for its mutant once the tests are collected: in its module,
    and wherever a module (the package, the test file) holds the original function under a name."""
    _, module, function, source = MUTANTS[name]
    body = source.replace("\n", "\n    ")
    (folder / f"mutant_{name}.py").write_text(
        "import sys\n\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\n\n\n"
        "def pytest_collection_modifyitems(session, config, items):\n"
        f"    import {module}\n"
        f"    orig = sys.modules['{module}'].{function}\n"
        f"    {body}\n"
        "    for mod in list(sys.modules.values()):\n"
        "        for key, value in list(vars(mod).items()) if hasattr(mod, '__dict__') else []:\n"
        "            if value is orig:\n"
        "                setattr(mod, key, bad)\n", encoding="utf-8")


def check_document(step, path, sections):
    """Guardrails of a Markdown deliverable: it exists, has its sections (French or English), and no TODO left.
    What is still the template (a section or a paragraph that starts with TODO) is ⏳, not ❌."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not verdict(step, bool(text), f"{path.name} existe.", f"il manque {path.name} dans le dossier du projet."):
        return ""
    todo_line = r"(?m)^\W*TODO\b"
    parts = re.split(r"(?m)^(#{2,}\s+.*)$", text)                # text, header, body, header, body…
    bodies = [(parts[i].lstrip("#").strip().lower(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    for section, keywords in sections.items():
        found = [body for header, body in bodies if any(word in header for word in keywords)]
        if not found:
            verdict(step, False, "", f"{path.name} : il manque une section « ## {section} » (ou en anglais, "
                    f"« ## {keywords[-1].capitalize()}… »).")
        elif any(re.search(todo_line, body) for body in found):
            pending(step, f"{path.name} : la section « {section} » est encore celle du modèle (TODO).")
        else:
            verdict(step, True, f"{path.name} : section « {section} ».", "")
    todo = re.findall(todo_line, text)
    if todo:
        pending(step, f"{path.name} : {len(todo)} paragraphe(s) commencent encore par « TODO » : remplace-les par ton texte.")
    else:
        verdict(step, True, f"{path.name} : plus aucun paragraphe « TODO ».", "")
    return text'''

# ---------------------------------------------------------------------------
# MP5.1
# ---------------------------------------------------------------------------
MP51_MD = r'''## MP5.1 · Cadrage, protocole de validation (découpage stratifié) et baselines (MLP puis petit CNN) ⏱️ 60 min

**Objectif :** savoir ce que tu construis et comment tu le mesureras, puis poser les deux baselines que tout modèle devra battre.

a) **Lis le cahier des charges** (`projets/partie_5_classifieur_cifar10/README.md`), puis écris les sections « L'objectif » et « Le protocole » de ton `README.md` : le problème (dix classes, des images 32 × 32), la métrique (l'accuracy, des classes équilibrées), le découpage (45 000 images pour l'entraînement, 5 000 pour la validation, le test de 10 000 ouvert une seule fois, en mode complet) et la règle « une variable à la fois, chaque run dans le journal ».
b) Dans `cifarkit/pipeline.py`, écris `stratified_split` (sa docstring) : la validation, tirée une fois pour toutes avec la graine `SEED`, garde 500 images de chaque classe. Importe `train_test_split` de ta librairie **dans** la fonction : le modèle livré doit pouvoir servir sans `mylearn`.
c) Dans `cifarkit/models.py`, écris `n_parameters`, `mlp`, `flat_size` (avec `conv_output_size` de ta librairie, ch. 21, importée dans la fonction) et `small_cnn`.
d) La cellule suivante entraîne les deux baselines avec `quick_fit` (fournie : Adam, sans augmentation, des images ramenées dans $[-1, 1]$), sur `N_BASE` images, pendant `EPOCHS_BASE` epochs. Une baseline encore plus simple sert de plancher : prédire toujours la même classe.'''

MP51_CHECK = r'''checks_of("MP5.1")
if kit_ok():
    files = ["__init__.py", "pipeline.py", "models.py", "train.py", "predict.py"]
    missing = [name for name in files if not (PROJECT / "cifarkit" / name).exists()]
    verdict("MP5.1", not missing, "le package cifarkit a ses cinq fichiers.", f"il manque cifarkit/{', cifarkit/'.join(missing)}.")
    versions = cifarkit.library_versions()
    print("versions :", versions, "· device :", DEVICE)
    X_all, y_all = data.load_cifar10()
    with attempt("MP5.1"):
        train_idx, val_idx = cifarkit.stratified_split(y_all, N_VAL, seed=SEED)
        counts = np.bincount(y_all[val_idx], minlength=10)
        ok_split = (len(val_idx) == N_VAL and len(train_idx) == len(y_all) - N_VAL and not set(train_idx) & set(val_idx)
                    and counts.min() == counts.max() == N_VAL // 10)
        verdict("MP5.1", ok_split, f"stratified_split : {len(train_idx)} images d'entraînement, {len(val_idx)} de validation, "
                f"{counts[0]} par classe, sans recouvrement.",
                f"stratified_split(y, {N_VAL}, seed) doit rendre deux tableaux d'indices disjoints, {len(y_all) - N_VAL} et "
                f"{N_VAL}, avec 500 images de chaque classe en validation (reçu {len(train_idx)} et {len(val_idx)}, "
                f"{counts.tolist()} par classe).")
        again = cifarkit.stratified_split(y_all, N_VAL, seed=SEED)[1]
        verdict("MP5.1", np.array_equal(np.sort(again), np.sort(val_idx)), "la même graine donne la même validation.",
                "deux appels avec la même graine doivent donner la même validation (np.random.default_rng(seed)).")
        X_train, y_train, X_val, y_val = X_all[train_idx], y_all[train_idx], X_all[val_idx], y_all[val_idx]
    with attempt("MP5.1"):
        verdict("MP5.1", cifarkit.n_parameters(nn.Linear(3, 2)) == 8, "n_parameters(nn.Linear(3, 2)) = 8.",
                "n_parameters doit compter les poids ET les biais qui demandent un gradient : 8 pour nn.Linear(3, 2).")
        base_mlp, base_cnn = cifarkit.mlp(), cifarkit.small_cnn()
        want = {"mlp": 3072 * 512 + 512 + 512 * 10 + 10,
                "small_cnn": (3 * 9 * 32 + 32) + (32 * 9 * 64 + 64) + 64 * 8 * 8 * 10 + 10}
        for label, model_b in (("mlp", base_mlp), ("small_cnn", base_cnn)):
            got = cifarkit.n_parameters(model_b)
            out = model_b(torch.zeros(2, 3, 32, 32))
            verdict("MP5.1", got == want[label] and tuple(out.shape) == (2, 10),
                    f"{label}() : {thousands(got)} paramètres, des logits (2, 10).",
                    f"{label}() : {got} paramètres et une sortie {tuple(out.shape)}, attendu {want[label]} et (2, 10) : "
                    "relis sa docstring.")
        verdict("MP5.1", cifarkit.flat_size(32, [8, 16, 32]) == 32 * 4 * 4, "flat_size(32, [8, 16, 32]) = 32 × 4 × 4.",
                f"flat_size(32, [8, 16, 32]) vaut {cifarkit.flat_size(32, [8, 16, 32])}, attendu 512 : trois blocs "
                "convolution (padding 1) puis pooling 2 × 2 divisent le côté par 8.")
    checks_done()'''

MP51_BASELINES = r'''BASELINES = {}       # name -> (history of quick_fit, number of parameters)
if ready("X_train", "base_mlp", "base_cnn"):
    majority = np.bincount(y_train).argmax()
    floor = float((y_val[:N_VAL_USED] == majority).mean())
    print(f"plancher : toujours prédire la classe {data.CLASSES[majority]} donne {floor:.1%} de validation")
    with attempt("MP5.1"):
        for name, maker in (("mlp", cifarkit.mlp), ("small_cnn", cifarkit.small_cnn)):
            torch.manual_seed(SEED)
            model_b = maker()
            history_b = cifarkit.quick_fit(model_b, X_train[:N_BASE], y_train[:N_BASE], X_val[:N_VAL_USED],
                                           y_val[:N_VAL_USED], epochs=EPOCHS_BASE, seed=SEED)
            BASELINES[name] = (history_b, cifarkit.n_parameters(model_b))
            print(f"{name:10} {thousands(cifarkit.n_parameters(model_b)):>9} paramètres · accuracy de validation par epoch "
                  f"{[round(a, 3) for a in history_b['val_acc']]} · {sum(history_b['seconds']):.0f} s")'''

# ---------------------------------------------------------------------------
# MP5.2
# ---------------------------------------------------------------------------
MP52_MD = r'''## MP5.2 · Pipeline de données : normalisation par canal apprise sur l'entraînement, augmentation transforms.v2, Dataset et DataLoader ⏱️ 60 min

**Objectif :** un pipeline qui donne au réseau des images normalisées comme il faut, sans fuite, avec une augmentation qui ne s'applique qu'à l'entraînement.

Dans `cifarkit/pipeline.py`, écris `channel_stats`, `make_transform`, les deux méthodes de `CifarDataset` et `make_loader` (leurs docstrings). La moyenne et l'écart-type de chaque canal s'apprennent sur les **seules** images d'entraînement (la cellule ci-dessous les calcule, `MEAN` et `STD`) et voyageront avec le modèle (MP5.5). L'augmentation (un recadrage aléatoire après un padding de 4 pixels, une symétrie horizontale) ne touche ni la validation ni le test. La vérification compare tes transformations à un calcul à la main et trace quelques images augmentées (`figures/augmentation.png`).'''

MP52_CHECK = r'''checks_of("MP5.2")
if kit_ok() and ready("X_train"):
    with attempt("MP5.2"):
        MEAN, STD = cifarkit.channel_stats(X_train)
        x64 = X_train.astype(np.float64) / 255
        want_mean = [float(x64[..., c].mean()) for c in range(3)]
        want_std = [float(x64[..., c].std()) for c in range(3)]
        verdict("MP5.2", np.allclose(MEAN, want_mean, atol=1e-6) and np.allclose(STD, want_std, atol=1e-6),
                f"channel_stats sur les {len(X_train)} images d'entraînement : moyenne {np.round(MEAN, 4).tolist()}, "
                f"écart-type {np.round(STD, 4).tolist()}.",
                f"channel_stats : attendu la moyenne {np.round(want_mean, 4).tolist()} et l'écart-type "
                f"{np.round(want_std, 4).tolist()} (par canal, sur [0, 1], divisé par n), reçu {MEAN} et {STD}.")
        del x64
        plain = cifarkit.make_transform(MEAN, STD, augment=False)(X_val[0])
        want = (torch.from_numpy(X_val[0].astype(np.float32) / 255).permute(2, 0, 1)
                - torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)) / torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
        verdict("MP5.2", tuple(plain.shape) == (3, 32, 32) and plain.dtype == torch.float32 and torch.allclose(plain, want, atol=1e-5),
                "make_transform sans augmentation : un tenseur float32 (3, 32, 32), (x / 255 − moyenne) / écart-type par canal.",
                f"make_transform(augment=False) : attendu un float32 (3, 32, 32) égal à (x / 255 − mean) / std, canal par canal, "
                f"sans augmentation ; reçu {plain.dtype} {tuple(plain.shape)}"
                + (f", à {(plain - want).abs().max().item():.2g} au plus de la valeur attendue." if plain.shape == want.shape else "."))
        augmented = cifarkit.make_transform(MEAN, STD, augment=True)
        torch.manual_seed(SEED)
        draws = [augmented(X_val[0]) for _ in range(12)]
        verdict("MP5.2", all(tuple(d.shape) == (3, 32, 32) for d in draws) and len({d.sum().item() for d in draws}) > 1,
                "make_transform avec augmentation : 12 tirages de la même image, de la bonne forme, tous différents ou presque.",
                "make_transform(augment=True) doit garder la forme (3, 32, 32) et varier d'un tirage à l'autre "
                "(RandomCrop(32, padding=4), RandomHorizontalFlip()).")
        as_tensor = lambda image: torch.from_numpy(image.astype(np.float32))      # a transform without randomness
        dataset = cifarkit.CifarDataset(X_val[:10], y_val[:10], as_tensor)
        item = dataset[3]
        verdict("MP5.2", len(dataset) == 10 and isinstance(item[1], int) and torch.equal(item[0], as_tensor(X_val[3])),
                "CifarDataset : 10 éléments, (image transformée, label entier).",
                "CifarDataset : len doit rendre le nombre d'images, et [i] le couple (transform(images[i]), int(labels[i])).")
        loader_a = cifarkit.make_loader(X_val[:100], y_val[:100], cifarkit.make_transform(MEAN, STD), batch_size=32,
                                        shuffle=True, seed=SEED)
        batches = [yb for _, yb in loader_a]
        loader_b = cifarkit.make_loader(X_val[:100], y_val[:100], cifarkit.make_transform(MEAN, STD), batch_size=32,
                                        shuffle=True, seed=SEED)
        xb0, yb0 = next(iter(loader_b))
        verdict("MP5.2", [len(b) for b in batches] == [32, 32, 32, 4] and torch.equal(batches[0], yb0)
                and tuple(xb0.shape) == (32, 3, 32, 32) and yb0.dtype == torch.int64,
                "make_loader : des mini-batches (32, 3, 32, 32) et des labels int64 ; le même ordre pour la même graine.",
                "make_loader(..., batch_size=32, shuffle=True, seed) sur 100 images : 4 mini-batches (32, 32, 32, 4), des "
                "labels int64, et le même ordre à chaque nouveau chargeur de même graine (son propre torch.Generator).")
        ordered = torch.cat([yb for _, yb in cifarkit.make_loader(X_val[:100], y_val[:100], cifarkit.make_transform(MEAN, STD),
                                                                   batch_size=32, shuffle=False)])
        verdict("MP5.2", torch.equal(ordered, torch.from_numpy(y_val[:100])), "sans shuffle, l'ordre des images est gardé.",
                "make_loader(shuffle=False) doit garder l'ordre des images (la validation se compare image par image).")
        fig, axes = plt.subplots(2, 6, figsize=(10, 3.8))
        mean_t, std_t = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
        for k, ax in enumerate(axes.ravel()):
            image = X_val[0] if k == 0 else (draws[k - 1] * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(image)
            ax.set_title("originale" if k == 0 else f"tirage {k}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Augmentation (recadrage après padding de 4 pixels, symétrie) : un {data.CLASSES[y_val[0]]} de la validation")
        fig.tight_layout()
        save(fig, figure_name("augmentation.png"))
        plt.show()
    checks_done()'''

# ---------------------------------------------------------------------------
# MP5.3
# ---------------------------------------------------------------------------
MP53_MD = r'''## MP5.3 · Boucle d'entraînement réutilisable : fit(), meilleur checkpoint, early stopping, planning de LR, reprise exacte, journal ⏱️ 90 min

**Objectif :** une boucle d'entraînement écrite une fois, testée, qui garde le meilleur modèle, sait reprendre là où elle s'est arrêtée et note chaque run.

Dans `cifarkit/train.py`, écris `evaluate`, `train_one_epoch`, `make_optimizer`, `make_scheduler`, `fit` et `log_run` (leurs docstrings). `fit` garde les poids de l'epoch de meilleure accuracy de validation ; avec un dossier, il écrit `best.pt` et `last.pt` après chaque epoch, de quoi reprendre exactement (l'optimiseur, le planning, les générateurs aléatoires). La vérification contrôle chaque pièce sur de petites données, puis la reprise au bit près (2 + 2 epochs contre 4 d'affilée), avant les premiers vrais runs : les deux baselines de MP5.1 entrent dans le journal, puis le petit CNN entraîné par `fit`, sans puis avec augmentation.'''

MP53_CHECK = r'''checks_of("MP5.3")
if kit_ok():
    g = torch.Generator().manual_seed(SEED)
    x_t, y_t = torch.randn(100, 3, 32, 32, generator=g), torch.randint(0, 10, (100,), generator=g)

    def tiny_loaders():
        """100 training images (mini-batches of 32, 32, 32 and 4) and 50 validation images (32 and 18)."""
        return (DataLoader(TensorDataset(x_t, y_t), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(SEED)),
                DataLoader(TensorDataset(x_t[:50], y_t[:50]), batch_size=32))

    def weight_gap(model_1, model_2):
        return max((a - b).abs().max().item() for a, b in zip(model_1.state_dict().values(), model_2.state_dict().values())
                   if a.is_floating_point())

    with attempt("MP5.3"):
        torch.manual_seed(SEED)
        probe = cifarkit.vgg([4, 8], dropout=0.3)          # batchnorm and dropout: the evaluation mode changes the result
        with torch.no_grad():
            probe.train()(x_t)                                # running statistics that are no longer the initial ones
        reference_e = copy.deepcopy(probe).eval()
        with torch.no_grad():
            logits_e = reference_e(x_t[:50])
        buffers_e = [b.clone() for b in probe.buffers()]
        loss_e, acc_e = cifarkit.evaluate(probe, tiny_loaders()[1])
        kept_e = all(torch.equal(a, b) for a, b in zip(buffers_e, probe.buffers()))
        back_to_train = probe.training
        probe.eval()
        cifarkit.evaluate(probe, tiny_loaders()[1])
        back_to_eval = not probe.training
        verdict("MP5.3", abs(loss_e - F.cross_entropy(logits_e, y_t[:50]).item()) < 1e-5
                and abs(acc_e - (logits_e.argmax(1) == y_t[:50]).sum().item() / 50) < 1e-12 and kept_e,
                "evaluate : la moyenne par exemple sur 32 + 18 images, en mode évaluation (batchnorm et dropout).",
                "evaluate : attendu la loss et l'accuracy moyennes sur les 50 exemples (des mini-batches de 32 et 18 : "
                "somme, puis division par 50), mesurées en mode évaluation : avec model.eval(), la batchnorm se sert de "
                "ses statistiques sans les mettre à jour, et le dropout ne coupe plus rien.")
        verdict("MP5.3", back_to_train and back_to_eval,
                "evaluate : le modèle revient dans son mode d'avant (entraînement ou évaluation).",
                "evaluate doit remettre le modèle dans le mode où il était : en entraînement s'il y était, en évaluation "
                "sinon (model.training, lu avant de passer en évaluation).")
        verdict("MP5.3", isinstance(cifarkit.make_optimizer(probe, "sgd", 0.1, 5e-4), torch.optim.SGD)
                and isinstance(cifarkit.make_optimizer(probe, "adamw", 1e-3, 1e-2), torch.optim.AdamW)
                and cifarkit.make_optimizer(probe, "sgd", 0.1, 5e-4).param_groups[0]["nesterov"],
                "make_optimizer : SGD avec momentum de Nesterov, ou AdamW.",
                "make_optimizer(model, \"sgd\", ...) doit rendre torch.optim.SGD(momentum=0.9, nesterov=True), et "
                "\"adamw\" un torch.optim.AdamW.")
        opt_s = cifarkit.make_optimizer(probe, "sgd", 0.1, 0.0)
        sched_s = cifarkit.make_scheduler(opt_s, "onecycle", 0.1, 10)
        verdict("MP5.3", isinstance(sched_s, torch.optim.lr_scheduler.OneCycleLR) and sched_s.total_steps == 10
                and cifarkit.make_scheduler(opt_s, "constant", 0.1, 10) is None,
                "make_scheduler : OneCycleLR sur le nombre total de pas, rien pour un learning rate constant.",
                "make_scheduler(opt, \"onecycle\", lr, total_steps) doit rendre OneCycleLR(max_lr=lr, total_steps), et "
                "\"constant\", None.")
        torch.manual_seed(SEED)
        model_a = cifarkit.vgg([4, 8])                        # batchnorm: the training mode changes the result
        model_b = copy.deepcopy(model_a)
        model_a.eval()                                        # train_one_epoch must switch to training mode itself
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.05)
        sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=1, gamma=0.5)
        loss_a, acc_a = cifarkit.train_one_epoch(model_a, tiny_loaders()[0], opt_a, sched_a, label_smoothing=0.1)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.05)
        sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=1, gamma=0.5)
        model_b.train()
        total_b, correct_b = 0.0, 0
        for xb, yb in tiny_loaders()[0]:
            logits_b = model_b(xb)
            loss_b = F.cross_entropy(logits_b, yb, label_smoothing=0.1)
            opt_b.zero_grad()
            loss_b.backward()
            opt_b.step()
            sched_b.step()
            total_b += loss_b.item() * len(yb)
            correct_b += (logits_b.argmax(1) == yb).sum().item()
        gap_a = weight_gap(model_a, model_b)
        verdict("MP5.3", gap_a < 1e-6 and sched_a.last_epoch == 4,
                "train_one_epoch : les poids d'une epoch écrite à la main (4 pas, label smoothing, planning à chaque pas).",
                f"train_one_epoch : écart de {gap_a:.1e} aux poids d'une epoch écrite à la main"
                + (f", et {sched_a.last_epoch} pas du planning au lieu de 4" if sched_a.last_epoch != 4 else "")
                + " : model.train(), puis à chaque mini-batch la cross-entropy avec label_smoothing, zero_grad, "
                "backward, step, et scheduler.step().")
        verdict("MP5.3", abs(loss_a - total_b / 100) < 1e-5 and abs(acc_a - correct_b / 100) < 1e-12,
                "train_one_epoch : la loss et l'accuracy moyennes par exemple (32 + 32 + 32 + 4 images).",
                "train_one_epoch : attendu la loss moyenne PAR EXEMPLE (la loss de chaque mini-batch multipliée par sa "
                "taille, sommée, puis divisée par 100 : le dernier mini-batch n'a que 4 images) et l'accuracy mesurée "
                "pendant l'epoch.")
        x_v = torch.randn(50, 3, 32, 32, generator=g)        # labels = the predictions of the untrained model:
        torch.manual_seed(SEED)                               # the accuracy can only fall, the best epoch is not the last
        with torch.no_grad():
            y_v = cifarkit.small_cnn([4, 8]).eval()(x_v).argmax(1)
        falling = lambda: (tiny_loaders()[0], DataLoader(TensorDataset(x_v, y_v), batch_size=32))
        folder_f = Path(tempfile.mkdtemp())
        torch.manual_seed(SEED)
        model_f = cifarkit.small_cnn([4, 8])
        history_f = cifarkit.fit(model_f, *falling(), epochs=5, lr=0.05, schedule="constant", folder=folder_f)
        best_f = history_f["best_epoch"]
        verdict("MP5.3", len(history_f["val_acc"]) == 5 and best_f == int(np.argmax(history_f["val_acc"])) + 1,
                f"fit : 5 epochs, best_epoch = {best_f}, la première epoch de meilleure accuracy de validation.",
                f"fit : best_epoch doit être la PREMIÈRE epoch de meilleure accuracy de validation, comptée à partir de 1 "
                f"(reçu {best_f} pour {[round(a, 3) for a in history_f['val_acc']]}).")
        torch.manual_seed(SEED)
        model_g = cifarkit.small_cnn([4, 8])                  # the same run, stopped after the best epoch
        cifarkit.fit(model_g, *falling(), epochs=5, lr=0.05, schedule="constant", stop_after=best_f)
        gap_f = weight_gap(model_f, model_g)
        verdict("MP5.3", gap_f == 0.0 and (folder_f / "best.pt").exists() and (folder_f / "last.pt").exists(),
                f"fit : le modèle final a les poids de l'epoch {best_f} (la meilleure), pas ceux de la dernière ; best.pt "
                "et last.pt sont écrits.",
                f"fit : à la fin, le modèle doit avoir les poids de la meilleure epoch ({best_f}, écart {gap_f:.1e} au même "
                "run arrêté par stop_after après cette epoch), et le dossier contenir best.pt et last.pt.")
        torch.manual_seed(SEED)
        model_p = cifarkit.small_cnn([4, 8])
        history_p = cifarkit.fit(model_p, *tiny_loaders(), epochs=30, lr=0.0, schedule="constant", patience=2)
        verdict("MP5.3", len(history_p["val_acc"]) == 3 and history_p["best_epoch"] == 1,
                "fit avec patience = 2 et un learning rate nul : arrêt après 1 + 2 epochs, best_epoch = 1.",
                f"avec lr = 0, l'accuracy de validation ne bouge plus : attendu 3 epochs (patience = 2, un progrès "
                f"STRICT) et best_epoch = 1, reçu {len(history_p['val_acc'])} epochs et {history_p['best_epoch']}.")

        def resumable(stop, folder, resume=False):
            if not resume:
                torch.manual_seed(SEED)
            model_r = cifarkit.vgg([4, 8], dropout=0.3)
            history_r = cifarkit.fit(model_r, *tiny_loaders(), epochs=4, lr=0.05, schedule="onecycle", folder=folder,
                                     resume=resume, stop_after=stop)
            return model_r, history_r

        straight, history_s = resumable(None, Path(tempfile.mkdtemp()))
        opt_x = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.05, momentum=0.9, nesterov=True)
        sched_x, want_lr = torch.optim.lr_scheduler.OneCycleLR(opt_x, max_lr=0.05, total_steps=4 * 4), []
        for _ in range(4 * 4):
            opt_x.step()
            sched_x.step()
            want_lr.append(opt_x.param_groups[0]["lr"])
        verdict("MP5.3", np.allclose(history_s["lr"], want_lr[3::4], rtol=1e-9, atol=0),
                "fit : OneCycleLR sur 4 epochs × 4 mini-batches, un pas du planning par mini-batch.",
                f"fit : le learning rate en fin d'epoch devrait être {np.round(want_lr[3::4], 5).tolist()} "
                f"(OneCycleLR sur epochs × len(train_loader) pas, scheduler.step() après chaque mini-batch), reçu "
                f"{np.round(history_s['lr'], 5).tolist()}.")
        folder_r = Path(tempfile.mkdtemp())
        resumable(2, folder_r)
        torch.manual_seed(12345)                              # another state: the resumption must restore its own
        resumed, history_r = resumable(None, folder_r, resume=True)
        gap_r = weight_gap(straight, resumed)
        same_history = all(list(history_s[key]) == list(history_r[key]) for key in ("train_loss", "val_loss", "lr"))
        verdict("MP5.3", gap_r == 0.0 and same_history,
                "reprise exacte : 2 epochs, une interruption, puis 2 epochs = 4 epochs d'affilée, au bit près (poids et "
                "historique).",
                "après la reprise, "
                + (f"les poids diffèrent de ceux de 4 epochs d'affilée (écart {gap_r:.2e})" if gap_r else
                   "les losses ou les learning rates des epochs 3 et 4 diffèrent de ceux de 4 epochs d'affilée")
                + " : last.pt doit garder le modèle, l'optimiseur, le planning, l'epoch, l'historique, l'état "
                "aléatoire de PyTorch (dropout, augmentation) et celui du générateur du DataLoader, et fit doit tout "
                "restaurer.")
        journal = Path(tempfile.mkdtemp()) / "runs.csv"
        cifarkit.log_run(journal, {"run": "a", "val_acc": 0.5, "config": {"arch": "mlp"}})
        cifarkit.log_run(journal, {"run": "b", "val_acc": 0.6})
        rows = cifarkit.read_runs(journal)
        try:
            cifarkit.log_run(journal, {"accuracy": 0.7})
            refused = False
        except ValueError:
            refused = True
        verdict("MP5.3", [r["run"] for r in rows] == ["a", "b"] and list(rows[0]) == cifarkit.RUN_COLUMNS and refused,
                "log_run : un en-tête, une ligne par run, une colonne inconnue refusée.",
                "log_run : attendu l'en-tête RUN_COLUMNS une seule fois, une ligne par appel, et ValueError pour une clé "
                "qui n'est pas une colonne.")
    checks_done()'''

MP53_RUNS = r'''FIRST_RUNS = {}   # name -> history
if ready("MEAN", "BASELINES") and kit_ok():
    with attempt("MP5.3"):
        if RUNS.exists():
            RUNS.unlink()                                     # the journal of this mode starts again with the notebook
        for name, (history_b, params_b) in BASELINES.items():
            best_b = int(np.argmax(history_b["val_acc"])) + 1
            cifarkit.log_run(RUNS, {"run": f"baseline_{name}", "date": time.strftime("%Y-%m-%d %H:%M"),
                                    "arch": "mlp" if name == "mlp" else "cnn", "n_train": N_BASE, "epochs": EPOCHS_BASE,
                                    "optimizer": "adam", "lr": 1e-3, "weight_decay": 0, "schedule": "constant",
                                    "augment": False, "seed": SEED, "params": params_b, "best_epoch": best_b,
                                    "val_acc": round(history_b["val_acc"][best_b - 1], 4),
                                    "val_loss": round(history_b["val_loss"][best_b - 1], 4),
                                    "seconds": round(sum(history_b["seconds"]), 1), "note": "quick_fit (MP5.1)"})
        for augment in (False, True):
            name = f"cnn_fit_{'aug' if augment else 'sans_aug'}"
            _, FIRST_RUNS[name], seconds = train_run(name, {"arch": "cnn"}, N_BASE, EPOCHS_BASE, lr=0.05, augment=augment,
                                                     label_smoothing=0.0, note="le petit CNN de MP5.1, avec fit (MP5.3)")
            print(f"{name:18} accuracy de validation par epoch {[round(a, 3) for a in FIRST_RUNS[name]['val_acc']]} · {seconds:.0f} s")
        print(f"\njournal : {len(cifarkit.read_runs(RUNS))} runs dans {RUNS.name}")'''

# ---------------------------------------------------------------------------
# MP5.4
# ---------------------------------------------------------------------------
MP54_MD = r'''## MP5.4 · Architectures : CNN type VGG puis blocs résiduels ; comparer stride et pooling, batchnorm et dropout ⏱️ 90 min

**Objectif :** comparer des architectures à budget égal, une variable à la fois, et savoir ce que chacune apporte.

Dans `cifarkit/models.py`, écris `vgg`, la méthode `forward` de `ResidualBlock` (son `__init__` est donné), `resnet` et `build_model` (leurs docstrings). La vérification compte leurs paramètres et vérifie les formes, puis compare cinq architectures, toutes entraînées de la même façon (`N_ARCH` images augmentées, `EPOCHS_ARCH` epochs, SGD et `OneCycleLR` jusqu'à un learning rate de 0,05) : un VGG à trois étages (16, 32, 64 canaux) avec pooling et batchnorm, le même avec des convolutions de stride 2 à la place du pooling, le même sans batchnorm, le même avec du dropout, et un petit ResNet de mêmes largeurs. Chaque run entre dans le journal. Un run peut ne rien apprendre (une accuracy au niveau du hasard, sous le plancher de MP5.1) : à ce learning rate, c'est ce qui arrive souvent au réseau sans batchnorm ; ce n'est pas forcément un bug, la vérification de `vgg` ci-dessus le dit.'''

MP54_CHECK = r'''checks_of("MP5.4")
if kit_ok():
    with attempt("MP5.4"):
        def conv_params(c_in, c_out, k=3, bias=True):
            return c_out * c_in * k * k + (c_out if bias else 0)

        want_vgg = (conv_params(3, 16, bias=False) + 2 * 16 + conv_params(16, 16, bias=False) + 2 * 16
                    + conv_params(16, 32, bias=False) + 2 * 32 + conv_params(32, 32, bias=False) + 2 * 32
                    + conv_params(32, 64, bias=False) + 2 * 64 + conv_params(64, 64, bias=False) + 2 * 64 + 64 * 10 + 10)
        model_v = cifarkit.vgg([16, 32, 64])
        verdict("MP5.4", cifarkit.n_parameters(model_v) == want_vgg and tuple(model_v(torch.zeros(2, 3, 32, 32)).shape) == (2, 10),
                f"vgg([16, 32, 64]) : {thousands(want_vgg)} paramètres (convolutions sans biais, suivies d'une batchnorm), des logits (2, 10).",
                f"vgg([16, 32, 64]) a {cifarkit.n_parameters(model_v)} paramètres, attendu {want_vgg} : deux convolutions "
                "3 × 3 par étage, sans biais quand une batchnorm suit, puis le pooling global et une couche dense.")
        no_bn = cifarkit.vgg([16, 32, 64], batchnorm=False)
        verdict("MP5.4", not any(isinstance(m, nn.BatchNorm2d) for m in no_bn.modules())
                and sum(isinstance(m, nn.MaxPool2d) for m in model_v.modules()) == 2
                and not any(isinstance(m, nn.MaxPool2d) for m in cifarkit.vgg([16, 32, 64], downsample="stride").modules())
                and [m.stride[0] for m in cifarkit.vgg([16, 32, 64], downsample="stride").modules()
                     if isinstance(m, nn.Conv2d)] == [1, 2, 1, 2, 1, 1],
                "vgg : deux poolings (pas après le dernier étage) ; en version stride, la deuxième convolution des deux "
                "premiers étages a stride 2 ; aucune batchnorm sans batchnorm.",
                "vgg : un MaxPool2d après chaque étage sauf le dernier (downsample=\"pool\"), aucun avec downsample=\"stride\" "
                "(la deuxième convolution de l'étage a stride 2), et aucune BatchNorm2d quand batchnorm=False.")
        torch.manual_seed(SEED)
        block = cifarkit.ResidualBlock(8, 8).eval()
        x_r = torch.randn(2, 8, 8, 8)
        with torch.no_grad():
            inner = block.bn2(block.conv2(torch.relu(block.bn1(block.conv1(x_r)))))
            ok_block = torch.allclose(block(x_r), torch.relu(inner + x_r), atol=1e-6)
            down = cifarkit.ResidualBlock(8, 16, stride=2).eval()
            shape_down = tuple(down(x_r).shape)
        verdict("MP5.4", ok_block and shape_down == (2, 16, 4, 4),
                "ResidualBlock : relu(F(x) + x) ; avec stride 2 et 16 canaux, la sortie (2, 16, 4, 4) passe par le raccourci 1 × 1.",
                f"ResidualBlock.forward doit rendre relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x)) : la somme AVANT la "
                f"dernière ReLU ; avec stride 2, attendu (2, 16, 4, 4), reçu {shape_down}.")
        model_r = cifarkit.resnet([16, 32, 64])
        verdict("MP5.4", cifarkit.n_parameters(model_r) == 78_042 and tuple(model_r(torch.zeros(2, 3, 32, 32)).shape) == (2, 10),
                "resnet([16, 32, 64]) : 78 042 paramètres, des logits (2, 10).",
                f"resnet([16, 32, 64]) a {cifarkit.n_parameters(model_r)} paramètres, attendu 78 042 : un stem, un bloc "
                "résiduel par largeur (stride 2 à partir de la deuxième), le pooling global et une couche dense.")
        try:
            cifarkit.build_model({"arch": "transformer"})
            refused = False
        except ValueError:
            refused = True
        same_r = cifarkit.n_parameters(cifarkit.build_model({"arch": "resnet", "widths": [16, 32, 64]})) == 78_042
        verdict("MP5.4", refused and same_r, "build_model reconstruit un modèle d'après sa configuration et refuse une arch inconnue.",
                "build_model({\"arch\": \"resnet\", \"widths\": [16, 32, 64]}) doit rendre le même resnet, et lever ValueError "
                "pour une arch inconnue.")
    checks_done()'''

MP54_COMPARE = r'''ARCHITECTURES = {   # name -> configuration: one variable at a time, from vgg_pool
    "vgg_pool": {"arch": "vgg", "widths": [16, 32, 64], "downsample": "pool"},
    "vgg_stride": {"arch": "vgg", "widths": [16, 32, 64], "downsample": "stride"},
    "vgg_sans_bn": {"arch": "vgg", "widths": [16, 32, 64], "downsample": "pool", "batchnorm": False},
    "vgg_dropout": {"arch": "vgg", "widths": [16, 32, 64], "downsample": "pool", "dropout": 0.3},
    "resnet": {"arch": "resnet", "widths": [16, 32, 64]},
}
ARCH_RESULTS = {}
if ready("MEAN") and kit_ok():
    with attempt("MP5.4"):
        for name, config in ARCHITECTURES.items():
            model_c, history_c, seconds_c = train_run(name, config, N_ARCH, EPOCHS_ARCH, lr=0.05, note="comparaison des architectures (MP5.4)")
            ARCH_RESULTS[name] = {"history": history_c, "params": cifarkit.n_parameters(model_c),
                                  "seconds_per_epoch": seconds_c / len(history_c["val_acc"])}
        print(f"{len(ARCHITECTURES)} architectures, {N_ARCH} images, {EPOCHS_ARCH} epochs\n")
        print("architecture   paramètres   best val acc   s par epoch")
        for name, result in ARCH_RESULTS.items():
            print(f"{name:13} {result['params']:11d} {max(result['history']['val_acc']):14.4f} {result['seconds_per_epoch']:13.1f}")
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, result in ARCH_RESULTS.items():
            ax.plot(np.arange(1, len(result["history"]["val_acc"]) + 1), result["history"]["val_acc"], marker="o", label=name)
        ax.set(title=f"Accuracy de validation ({N_ARCH} images d'entraînement augmentées)", xlabel="epoch", ylabel="accuracy")
        ax.legend()
        fig.tight_layout()
        save(fig, figure_name("architectures.png"))
        plt.show()'''

# ---------------------------------------------------------------------------
# MP5.5
# ---------------------------------------------------------------------------
MP55_MD = r'''## MP5.5 · Recherche d'hyperparamètres ciblée (aléatoire puis zoom) sur un sous-ensemble, puis entraînement complet sur GPU Colab ; le modèle livré ⏱️ 90 min

**Objectif :** régler le learning rate et le weight decay à petit coût, puis entraîner une seule fois le modèle retenu, en entier, et le livrer avec son prétraitement.

a) Écris `sample_config(rng, lr_range, wd_range)` ci-dessous : un learning rate et un weight decay tirés **sur une échelle logarithmique** dans leurs intervalles. La recherche porte sur la meilleure architecture de MP5.4 : quatre tirages au hasard sur `N_SEARCH` images (`EPOCHS_SEARCH` epochs, 400 images de validation en mode rapide), puis deux tirages de **zoom**, autour du meilleur learning rate (un intervalle de ×½ à ×2).
b) Dans `cifarkit/train.py`, écris `save_model` et `load_model` (leurs docstrings) : le modèle livré, ce sont ses poids, sa configuration et sa normalisation.
c) **Le choix** : écris `RULE`, la règle, **avant** de lire le tableau de la recherche, puis `FINAL`, la configuration du modèle, et `FINAL_TRAINING`, ses réglages d'entraînement. La cellule suivante entraîne ce modèle une fois, sur `N_FINAL` images pendant `EPOCHS_FINAL` epochs : sur CPU, c'est le mode rapide ; en mode complet (`FAST_MODE = False`), 45 000 images et 30 epochs, à lancer sur un GPU de Colab (🚀, quelques minutes ; cet entraînement final prend environ une demi-heure sur un CPU à deux cœurs, et tout le notebook en mode complet environ une heure vingt). Elle enregistre le modèle (`model.pt` et `model.json`) et n'ouvre le test que si toutes les vérifications des étapes MP5.1 à MP5.4 passent, et seulement en mode complet ; chaque ouverture est notée dans `test_log.json`, que la cellule commite aussitôt (le test ne se rouvre pas tant que ce fichier a changé sans commit).'''

MP55_TODO = r'''def sample_config(rng, lr_range, wd_range):
    """{"lr": ..., "weight_decay": ...}: two floats drawn by rng on a log scale, lr in lr_range = (low, high) and
    weight_decay in wd_range (log10 drawn uniformly between log10(low) and log10(high))."""
    raise NotImplementedError("sample_config")  # TODO MP5.5 a)'''

MP55_SOLUTION = r'''def sample_config(rng, lr_range, wd_range):
    """{"lr": ..., "weight_decay": ...}: two floats drawn by rng on a log scale, lr in lr_range = (low, high) and
    weight_decay in wd_range (log10 drawn uniformly between log10(low) and log10(high))."""
    def log_uniform(low, high):
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))
    return {"lr": log_uniform(*lr_range), "weight_decay": log_uniform(*wd_range)}'''

MP55_SEARCH = r'''checks_of("MP5.5 a")
SEARCH = []          # (name, lr, weight_decay, best validation accuracy)
LR_RANGE, WD_RANGE = (0.01, 0.4), (1e-4, 3e-3)
if ready("MEAN", "sample_config", "ARCH_RESULTS") and ARCH_RESULTS and kit_ok():
    SEARCH_ARCH = max(ARCH_RESULTS, key=lambda name: max(ARCH_RESULTS[name]["history"]["val_acc"]))   # the best of MP5.4
    with attempt("MP5.5"):
        draws = [sample_config(np.random.default_rng(k), LR_RANGE, WD_RANGE) for k in range(400)]
        lrs = np.array([d["lr"] for d in draws])
        verdict("MP5.5", lrs.min() >= 0.01 and lrs.max() <= 0.4 and 0.4 < np.mean(lrs < np.sqrt(0.01 * 0.4)) < 0.6,
                f"sample_config : 400 tirages dans l'intervalle, la moitié environ sous la moyenne géométrique ({np.sqrt(0.004):.3f}).",
                "sample_config : un tirage LOGARITHMIQUE (10 ** uniform(log10(low), log10(high))) met la moitié des "
                "learning rates sous la moyenne géométrique des bornes, pas sous leur moyenne arithmétique.")
        rng_s = np.random.default_rng(SEED)
        trials = [("hasard", sample_config(rng_s, LR_RANGE, WD_RANGE)) for _ in range(4)]
        for k, (stage, hp) in enumerate(trials, start=1):
            _, history_s, _ = train_run(f"recherche_{k}", ARCHITECTURES[SEARCH_ARCH], N_SEARCH, EPOCHS_SEARCH, lr=hp["lr"],
                                        weight_decay=hp["weight_decay"], n_val=400 if FAST_MODE else None,
                                        note=f"recherche, {stage} (MP5.5)")
            SEARCH.append((f"recherche_{k}", hp["lr"], hp["weight_decay"], max(history_s["val_acc"])))
        best_lr = max(SEARCH, key=lambda row: row[3])[1]
        for k in (5, 6):
            hp = sample_config(rng_s, (best_lr / 2, best_lr * 2), WD_RANGE)
            _, history_s, _ = train_run(f"recherche_{k}", ARCHITECTURES[SEARCH_ARCH], N_SEARCH, EPOCHS_SEARCH, lr=hp["lr"],
                                        weight_decay=hp["weight_decay"], n_val=400 if FAST_MODE else None,
                                        note="recherche, zoom (MP5.5)")
            SEARCH.append((f"recherche_{k}", hp["lr"], hp["weight_decay"], max(history_s["val_acc"])))
        print(f"{SEARCH_ARCH}, {N_SEARCH} images, {EPOCHS_SEARCH} epochs\n")
        print("run           learning rate   weight decay   best val acc")
        for name, lr_s, wd_s, acc_s in SEARCH:
            print(f"{name:13} {lr_s:13.4f} {wd_s:14.5f} {acc_s:14.4f}")
    checks_done()'''

MP55_TODO_FINAL = r'''RULE = ...             # MP5.5 c) the rule of your choice, in one sentence (a string), written BEFORE the table
FINAL = ...            # MP5.5 c) the configuration of the delivered model, a dict for build_model
FINAL_TRAINING = ...   # MP5.5 c) its training settings: a dict with lr, weight_decay, augment, label_smoothing'''

MP55_SOLUTION_FINAL = r'''RULE = ("l'architecture de meilleure accuracy de validation en MP5.4, puis le learning rate et le weight decay du "
        "meilleur run de la recherche ; augmentation et label smoothing de 0,1 gardés (l'entraînement final est long)")
FINAL = ARCHITECTURES[SEARCH_ARCH] if "SEARCH_ARCH" in globals() else ...
_best_search = max(SEARCH, key=lambda row: row[3]) if SEARCH else None
FINAL_TRAINING = ({"lr": round(_best_search[1], 3), "weight_decay": float(f"{_best_search[2]:.1e}"), "augment": True,
                   "label_smoothing": 0.1} if _best_search else ...)
print("FINAL =", FINAL, "\nFINAL_TRAINING =", FINAL_TRAINING)'''

MP55_TRAIN = r'''checks_of("MP5.5 c")
final_ok = False
if ready("MEAN", "ARCH_RESULTS", "SEARCH") and filled(RULE, FINAL, FINAL_TRAINING) and kit_ok():
    with attempt("MP5.5"):
        keys = {"lr", "weight_decay", "augment", "label_smoothing"}
        verdict("MP5.5", isinstance(FINAL_TRAINING, dict) and keys <= set(FINAL_TRAINING),
                f"réglages retenus : {FINAL_TRAINING} ({RULE}).",
                f"FINAL_TRAINING doit être un dict avec les clés {sorted(keys)}.")
        best_arch = max(ARCH_RESULTS, key=lambda name: max(ARCH_RESULTS[name]["history"]["val_acc"]))
        if FINAL != ARCHITECTURES[best_arch]:
            print(f"ℹ️ MP5.5 : la meilleure architecture de MP5.4 est « {best_arch} » ; explique dans ton README "
                  "pourquoi tu en retiens une autre.")
        final_model, final_history, final_seconds = train_run(
            "final", FINAL, N_FINAL, EPOCHS_FINAL, lr=FINAL_TRAINING["lr"], weight_decay=FINAL_TRAINING["weight_decay"],
            augment=FINAL_TRAINING["augment"], label_smoothing=FINAL_TRAINING["label_smoothing"],
            folder=Path(tempfile.mkdtemp()), note="le modèle livré (MP5.5)")
        final_best = final_history["best_epoch"]
        final_val = final_history["val_acc"][final_best - 1]
        print(f"{len(final_history['val_acc'])} epochs en {final_seconds / 60:.1f} min ; meilleure accuracy de validation "
              f"{final_val:.4f} (epoch {final_best}), sur {N_VAL_USED} images")
        cifarkit.save_model(MODEL_DIR, final_model, FINAL, MEAN, STD,
                            {"val_acc": round(final_val, 4), "best_epoch": final_best, "n_train": N_FINAL,
                             "epochs": EPOCHS_FINAL, "training": FINAL_TRAINING})
        reloaded, info = cifarkit.load_model(MODEL_DIR)
        val_loader = cifarkit.make_loader(X_val[:N_VAL_USED], y_val[:N_VAL_USED], cifarkit.make_transform(info["mean"], info["std"]),
                                          batch_size=500, shuffle=False)
        verdict("MP5.5", abs(cifarkit.evaluate(reloaded, val_loader)[1] - final_val) < 1e-9 and not reloaded.training,
                f"save_model puis load_model : le modèle rechargé (en évaluation) retrouve {final_val:.4f} de validation.",
                "le modèle rechargé par load_model ne retrouve pas l'accuracy de validation de la meilleure epoch : "
                "save_model doit écrire les poids et model.json (config, mean, std) ; load_model reconstruit, charge, eval() ; "
                "et evaluate doit mesurer en mode évaluation (relis la vérification de MP5.3).")
        if FAST_MODE:
            verdict("MP5.5", final_val >= TARGET, f"objectif du mode rapide : {final_val:.1%} de validation, au moins 65 %.",
                    f"{final_val:.1%} de validation : le cahier des charges demande au moins 65 % en mode rapide.")
        epochs_f = np.arange(1, len(final_history["val_acc"]) + 1)
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
        ax_l.plot(epochs_f, final_history["train_loss"], ls="--", label="entraînement (pendant l'epoch)")
        ax_l.plot(epochs_f, final_history["val_loss"], label="validation")
        ax_a.plot(epochs_f, final_history["train_acc"], ls="--", label="entraînement (pendant l'epoch)")
        ax_a.plot(epochs_f, final_history["val_acc"], label="validation")
        for ax in (ax_l, ax_a):
            ax.axvline(final_best, color="0.6", ls=":", label="poids gardés")
            ax.set(xlabel="epoch")
            ax.legend()
        ax_l.set(title="loss (cross-entropy, label smoothing à l'entraînement)")
        ax_a.set(title="accuracy")
        fig.tight_layout()
        save(fig, figure_name("courbes_final.png"))
        plt.show()
        final_ok = True
    checks_done()'''

MP55_TEST = r'''checks_of("MP5.5 test")
if final_ok:
    blocking = {cell: [f for f in fails] for cell, fails in FAILED.items() if fails and not cell.startswith("MP5.5 test")}
    verdict("MP5.5", not blocking, "toutes les vérifications des étapes précédentes passent : "
            + ("en mode complet, le test pourrait s'ouvrir." if FAST_MODE else "le test peut s'ouvrir."),
            "le test reste fermé tant qu'une vérification échoue : " + " · ".join(f"{cell} : {fails[0].rstrip('.')}" for cell, fails in blocking.items())[:900] + ".")
    test_log = PROJECT / "test_log.json"
    logged, logged_detail = committed(test_log)          # a log erased since its commit counts as changed
    if not test_log.exists() and logged_detail == "n'est pas encore commité":
        logged = True                                     # no opening yet: the first one may come
    if FAST_MODE:
        print("⏸️ MP5.5 : en mode rapide, le test reste fermé : il ne s'ouvre qu'une fois, pour la mesure finale, en mode "
              "complet (FAST_MODE = False dans la cellule de setup), quand tes choix sont faits sur la validation.")
    elif not blocking and logged is False:
        print(f"⏳ MP5.5 : le test reste fermé : test_log.json {logged_detail}. Commite-le (ou restaure-le : git checkout -- "
              "test_log.json) avant de rouvrir le test : une ouverture qui n'est pas commitée ne laisse aucune trace si le "
              "fichier est effacé. Relance ensuite cette cellule.")
    elif not blocking:
        if logged is None:
            print(f"⚠️ MP5.5 : test_log.json {logged_detail} : tes ouvertures du test ne sont gardées que par ce fichier.")
        description = (f"{FINAL}, {N_FINAL} images, {EPOCHS_FINAL} epochs, SGD Nesterov + OneCycleLR, {FINAL_TRAINING}, "
                       f"graine {SEED}")
        X_test, y_test = data.open_test(PROJECT, description)
        if COMMIT_TEST_LOG and committed(test_log)[0] is False:      # a new opening: commit it at once
            commit_file(test_log, "MP5: open the test set")
        test_loader = cifarkit.make_loader(X_test, y_test, cifarkit.make_transform(MEAN, STD), batch_size=500, shuffle=False)
        test_loss, test_acc = cifarkit.evaluate(final_model, test_loader)
        print(f"test (ouvert une fois) : accuracy {test_acc:.4f}, loss {test_loss:.4f} · objectif du cahier des charges : "
              "au moins 80 % en mode complet")
        print("ouvertures du test :", len(data.test_openings(PROJECT)))
        verdict("MP5.5", test_acc >= TARGET, f"{test_acc:.1%} de test, au moins 80 %.",
                f"{test_acc:.1%} de test : le cahier des charges demande au moins 80 % en mode complet.")
        RESULTS = {
            "fast_mode": FAST_MODE, "device": DEVICE, "seed": SEED, "final": FINAL, "training": FINAL_TRAINING, "rule": RULE,
            "n_train": N_FINAL, "epochs": EPOCHS_FINAL, "best_epoch": final_history["best_epoch"],
            "params": cifarkit.n_parameters(final_model),
            "val_accuracy": final_history["val_acc"][final_history["best_epoch"] - 1], "test_accuracy": test_acc,
            "test_loss": test_loss, "minutes": round(final_seconds / 60, 1),
            "baselines": {name: max(h["val_acc"]) for name, (h, _) in BASELINES.items()},
            "architectures": {name: {"best_val_accuracy": max(r["history"]["val_acc"]), "params": r["params"],
                                     "seconds_per_epoch": round(r["seconds_per_epoch"], 1)} for name, r in ARCH_RESULTS.items()},
            "search": [{"run": n, "lr": lr_s, "weight_decay": wd_s, "best_val_accuracy": a} for n, lr_s, wd_s, a in SEARCH],
            "versions": cifarkit.library_versions(),
        }
        (PROJECT / "results.json").write_text(json.dumps(RESULTS, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("💾 results.json écrit.")
    checks_done()'''

# ---------------------------------------------------------------------------
# MP5.6
# ---------------------------------------------------------------------------
MP56_MD = r'''## MP5.6 · Analyse d'erreurs et robustesse : confusions, exemples mal classés, FGSM ⏱️ 60 min

**Objectif :** savoir **où** et **pourquoi** le modèle se trompe, et ce qu'il devient face à une perturbation invisible.

Dans `cifarkit/train.py`, écris `predict_proba` et `fgsm` (leurs docstrings). La cellule suivante calcule la matrice de confusion du modèle final avec `confusion_matrix` de ta librairie (sur le test en mode complet, une fois ouvert ; sur la validation en mode rapide), montre les 20 erreurs les plus « sûres d'elles » et mesure l'accuracy sous l'attaque FGSM (ch. 21) pour plusieurs tailles $\varepsilon$ de perturbation. Commente ensuite les 20 erreurs dans tes observations : lesquelles sont excusables (une image ambiguë, un label discutable), lesquelles ne le sont pas ?'''

MP56_CHECK = r'''checks_of("MP5.6")
if final_ok and kit_ok():
    from mylearn.metrics import confusion_matrix
    with attempt("MP5.6"):
        on_test = not FAST_MODE and "X_test" in globals()
        X_e, y_e = (X_test, y_test) if on_test else (X_val[:N_VAL_USED], y_val[:N_VAL_USED])
        where = "test" if on_test else "validation"
        proba = cifarkit.predict_proba(final_model, X_e, MEAN, STD)
        with torch.no_grad():
            x0 = (torch.from_numpy(X_e[:8].astype(np.float32) / 255).permute(0, 3, 1, 2)
                  - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
            want_p = torch.softmax(final_model.eval()(x0), dim=1).numpy()
        verdict("MP5.6", proba.shape == (len(X_e), 10) and np.allclose(proba[:8], want_p, atol=1e-5),
                "predict_proba : les probabilités (N, 10) du modèle en évaluation, avec sa normalisation.",
                f"predict_proba : attendu un tableau ({len(X_e)}, 10), la softmax des logits d'images ramenées dans [0, 1], "
                "mises en (N, 3, 32, 32) et normalisées par mean et std.")
        pred = proba.argmax(1)
        matrix = np.asarray(confusion_matrix(y_e, pred, labels=list(range(10))))
        accuracy_e = float((pred == y_e).mean())
        per_class = matrix.diagonal() / matrix.sum(axis=1)
        pairs = sorted(((matrix[i, j] + matrix[j, i], data.CLASSES[i], data.CLASSES[j]) for i in range(10) for j in range(i + 1, 10)), reverse=True)[:3]
        print(f"{where} : accuracy {accuracy_e:.4f} ; classe la plus faible : {data.CLASSES[per_class.argmin()]} "
              f"({per_class.min():.1%}) ; paires les plus confondues : " + ", ".join(f"{a}/{b} ({n})" for n, a, b in pairs))
        print("accuracy par classe :", {data.CLASSES[k]: round(float(per_class[k]), 3) for k in range(10)})
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.imshow(matrix, cmap="Blues")
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(10), yticks=range(10), xlabel="classe prédite", ylabel="vraie classe",
               title=f"Matrice de confusion ({where}, accuracy {accuracy_e:.1%})")
        ax.set_xticklabels(data.CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(data.CLASSES, fontsize=8)
        fig.tight_layout()
        save(fig, figure_name("confusion.png"))
        plt.show()
        wrong = np.flatnonzero(pred != y_e)
        worst = wrong[np.argsort(-proba[wrong, pred[wrong]])][:20]
        fig, axes = plt.subplots(4, 5, figsize=(11, 9.5))
        for ax, k in zip(axes.ravel(), worst):
            ax.imshow(X_e[k])
            ax.set_title(f"#{k} vrai : {data.CLASSES[y_e[k]]}\nprédit : {data.CLASSES[pred[k]]} ({proba[k, pred[k]]:.0%})", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Les 20 erreurs les plus sûres d'elles ({where})")
        fig.tight_layout()
        save(fig, figure_name("erreurs.png"))
        plt.show()
        n_attack = 1000
        x_att = torch.from_numpy(X_e[:n_attack].astype(np.float32) / 255).permute(0, 3, 1, 2).contiguous()
        y_att = torch.from_numpy(y_e[:n_attack])
        buffers_f = [b.clone() for b in final_model.buffers()]
        adv = cifarkit.fgsm(final_model, x_att[:16], y_att[:16], 4 / 255, MEAN, STD)
        verdict("MP5.6", tuple(adv.shape) == (16, 3, 32, 32) and (adv - x_att[:16]).abs().max() <= 4 / 255 + 1e-6
                and adv.min() >= 0 and adv.max() <= 1 and not adv.requires_grad,
                "fgsm : chaque pixel bouge d'au plus ε = 4/255 et reste dans [0, 1].",
                "fgsm : attendu x + ε·signe(gradient), ramené dans [0, 1] (clamp), détaché du graphe ; chaque pixel bouge "
                "d'au plus ε.")
        verdict("MP5.6", all(torch.equal(a, b) for a, b in zip(buffers_f, final_model.buffers())),
                "fgsm : le modèle attaqué est en mode évaluation (ses statistiques de batchnorm n'ont pas bougé).",
                "fgsm : les statistiques de batchnorm du modèle ont changé pendant l'attaque : la loss se calcule en mode "
                "évaluation (model.eval()), sinon chaque attaque modifie le modèle.")
        EPSILONS = [0, 1 / 255, 2 / 255, 4 / 255, 8 / 255]
        FGSM_ACC = []
        m_t, s_t = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
        for eps in EPSILONS:
            correct = 0
            for i in range(0, n_attack, 250):
                xb = x_att[i:i + 250] if eps == 0 else cifarkit.fgsm(final_model, x_att[i:i + 250], y_att[i:i + 250], eps, MEAN, STD)
                with torch.no_grad():
                    correct += (final_model((xb - m_t) / s_t).argmax(1) == y_att[i:i + 250]).sum().item()
            FGSM_ACC.append(correct / n_attack)
        print("FGSM, accuracy sur", n_attack, "images :", {f"{round(e * 255)}/255": round(a, 3) for e, a in zip(EPSILONS, FGSM_ACC)})
        verdict("MP5.6", FGSM_ACC[-1] < FGSM_ACC[0], "l'attaque fait baisser l'accuracy.",
                "l'accuracy sous l'attaque ne baisse pas : fgsm doit pousser chaque pixel dans le sens qui fait MONTER la "
                "loss, x + ε·signe(∂L/∂x), et non dans le sens qui la fait baisser.")
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot([round(e * 255) for e in EPSILONS], FGSM_ACC, marker="o")
        ax.set(xlabel="ε (en niveaux de gris sur 255)", ylabel="accuracy", ylim=(0, 1),
               title=f"Accuracy sous l'attaque FGSM ({where}, {n_attack} images)")
        fig.tight_layout()
        save(fig, figure_name("fgsm.png"))
        plt.show()
    checks_done()'''

# ---------------------------------------------------------------------------
# MP5.7
# ---------------------------------------------------------------------------
MP57_MD = r'''## MP5.7 · Comparaison avec un ResNet-18 pré-entraîné de torchvision (nouvelle tête seule) : coût, gain, limites ⏱️ 60 min

**Objectif :** savoir ce que vaut ton modèle face à un réseau pré-entraîné sur ImageNet, à quel coût, et ce que cette comparaison ne dit pas.

`torchvision.models.resnet18(weights="DEFAULT")` télécharge (une fois, environ 45 Mo) un ResNet-18 entraîné sur ImageNet (ch. 23, 23.24). Son corps est gelé ; seule une nouvelle tête, `nn.Linear(512, 10)`, apprend. Pour aller vite, on calcule une fois les 512 features de chaque image, puis on entraîne la tête sur ces features.

Écris `resnet18_features(model, images, size, batch_size=100)` ci-dessous : des images uint8 `(N, 32, 32, 3)` ramenées dans $[0, 1]$, agrandies à `size` × `size` (`antialias=True`), normalisées avec la moyenne et l'écart-type d'ImageNet (`IMAGENET_MEAN`, `IMAGENET_STD`), puis la sortie du réseau **avant** `model.fc`, en mode évaluation et sans graphe. La cellule suivante entraîne la tête sur `N_PRETRAINED` images d'entraînement, la mesure sur la validation, et compare : l'accuracy, le nombre de paramètres entraînés et le temps de calcul par image.'''

MP57_TODO = r'''IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def resnet18_features(model, images, size, batch_size=100):
    """(N, 512) float32 tensor: uint8 images (N, 32, 32, 3) scaled to [0, 1], put in (N, 3, 32, 32), resized to
    size x size (antialias=True), normalized with IMAGENET_MEAN and IMAGENET_STD, then the output of model BEFORE
    model.fc, in evaluation mode, without a graph, by mini-batches of batch_size; model.fc is unchanged afterwards."""
    raise NotImplementedError("resnet18_features")  # TODO MP5.7'''

MP57_SOLUTION = r'''IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def resnet18_features(model, images, size, batch_size=100):
    """(N, 512) float32 tensor: uint8 images (N, 32, 32, 3) scaled to [0, 1], put in (N, 3, 32, 32), resized to
    size x size (antialias=True), normalized with IMAGENET_MEAN and IMAGENET_STD, then the output of model BEFORE
    model.fc, in evaluation mode, without a graph, by mini-batches of batch_size; model.fc is unchanged afterwards."""
    from torchvision.transforms import v2
    prepare = v2.Compose([v2.Resize((size, size), antialias=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    head, model.fc = model.fc, nn.Identity()                        # everything but the head
    model.eval()
    try:
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                x = torch.from_numpy(np.asarray(images[i:i + batch_size], dtype=np.float32) / 255).permute(0, 3, 1, 2)
                out.append(model(prepare(x)))
        return torch.cat(out)
    finally:
        model.fc = head'''

MP57_RUN = r'''checks_of("MP5.7")
PRETRAINED = {}
try:
    from torchvision.models import ResNet18_Weights, resnet18
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)           # about 45 MB the first time, then the local cache
except Exception as exc:                                            # no network: the step is done on Colab
    backbone = None
    pending("MP5.7", f"poids de ResNet-18 indisponibles ({type(exc).__name__}) : fais cette étape sur Colab.")
if backbone is not None and ready("X_train", "resnet18_features") and final_ok:
    with attempt("MP5.7"):
        for p in backbone.parameters():
            p.requires_grad_(False)
        n_val_p = min(N_VAL_USED, 2_000) if FAST_MODE else N_VAL
        started = time.perf_counter()
        feats_tr = resnet18_features(backbone, X_train[:N_PRETRAINED], SIZE_PRETRAINED)
        feats_va = resnet18_features(backbone, X_val[:n_val_p], SIZE_PRETRAINED)
        extract_seconds = time.perf_counter() - started
        verdict("MP5.7", tuple(feats_tr.shape) == (N_PRETRAINED, 512) and isinstance(backbone.fc, nn.Linear)
                and not feats_tr.requires_grad,
                f"resnet18_features : ({N_PRETRAINED}, 512) features, sans graphe ; model.fc remis en place.",
                f"resnet18_features doit rendre un tenseur ({N_PRETRAINED}, 512) sans graphe (la sortie avant fc), et "
                "laisser model.fc tel quel.")
        torch.manual_seed(SEED)
        head = nn.Linear(512, 10)
        opt_h = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        t_tr, t_va = torch.from_numpy(y_train[:N_PRETRAINED]), torch.from_numpy(y_val[:n_val_p])
        generator = torch.Generator().manual_seed(SEED)
        for _ in range(30):
            for idx in torch.randperm(N_PRETRAINED, generator=generator).split(128):
                loss_h = F.cross_entropy(head(feats_tr[idx]), t_tr[idx])
                opt_h.zero_grad()
                loss_h.backward()
                opt_h.step()
        with torch.no_grad():
            acc_head = (head(feats_va).argmax(1) == t_va).float().mean().item()
        x_one = torch.zeros(64, 3, 32, 32)
        started = time.perf_counter()
        with torch.inference_mode():
            final_model.eval()(x_one)
        ours_ms = (time.perf_counter() - started) / 64 * 1000
        PRETRAINED = {"val_accuracy": acc_head, "n_train": N_PRETRAINED, "size": SIZE_PRETRAINED,
                      "trained_params": sum(p.numel() for p in head.parameters()),
                      "frozen_params": sum(p.numel() for p in backbone.parameters()),
                      "ms_per_image": extract_seconds / (N_PRETRAINED + n_val_p) * 1000}
        print(f"ResNet-18 gelé ({thousands(PRETRAINED['frozen_params'])} paramètres) + tête entraînée "
              f"({thousands(PRETRAINED['trained_params'])}) : {acc_head:.1%} de validation avec {N_PRETRAINED} images en "
              f"{SIZE_PRETRAINED} × {SIZE_PRETRAINED}, {PRETRAINED['ms_per_image']:.1f} ms par image pour les features")
        print(f"ton modèle ({thousands(cifarkit.n_parameters(final_model))} paramètres, tous entraînés) : "
              f"{final_history['val_acc'][final_history['best_epoch'] - 1]:.1%} de validation avec {N_FINAL} images, "
              f"{ours_ms:.2f} ms par image")
        if (PROJECT / "results.json").exists() and not FAST_MODE:
            results = json.loads((PROJECT / "results.json").read_text(encoding="utf-8"))
            results["pretrained_resnet18"] = PRETRAINED
            results["fgsm"] = {f"{round(e * 255)}/255": a for e, a in zip(EPSILONS, FGSM_ACC)} if "FGSM_ACC" in globals() else None
            (PROJECT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print("💾 results.json complété.")
    checks_done()'''

# ---------------------------------------------------------------------------
# MP5.8
# ---------------------------------------------------------------------------
MP58_MD = r'''## MP5.8 · Livraison : README, fiche modèle, predict.py, tests et mutants, nettoyage ⏱️ 90 min

**Objectif :** un dépôt qu'un recruteur comprend en deux minutes, un modèle qu'on peut réutiliser sans le notebook, et des tests qui attrapent des erreurs.

a) Dans `cifarkit/predict.py`, écris `load_image` et `classify` (la ligne de commande `main` est fournie) : `python -m cifarkit.predict photo.jpg --model .` classe n'importe quelle image avec le modèle livré.
b) Complète `test_cifarkit.py` : **au moins trois tests** (un est donné), chacun avec un nom clair et un message dont la première ligne dit ce qui était attendu. La vérification les relance contre cinq **mutants** de `cifarkit` (une erreur glissée dans `evaluate`, `channel_stats`, `fit`, `fgsm` ou `stratified_split`) : de bons tests en attrapent au moins deux.
c) Termine le `README.md` de portfolio et la fiche modèle `MODEL_CARD.md` (leurs modèles sont dans ton dossier) : les résultats chiffrés (`results.json`), le journal (`runs.csv`, au moins 8 runs), les figures légendées, les limites, ce que tu as appris. Relance tout le notebook, de zéro, en mode complet sur Colab (*Exécution → Tout exécuter*), puis commite le projet.'''

MP58_CHECK = r'''checks_of("MP5.8")
if kit_ok():
    with attempt("MP5.8"):
        if final_ok and (MODEL_DIR / "model.json").exists():
            folder_png = Path(tempfile.mkdtemp())
            from PIL import Image
            paths = []
            for k in range(3):
                path = folder_png / f"val_{k}_{data.CLASSES[y_val[k]]}.png"
                Image.fromarray(X_val[k]).resize((96, 96)).save(path)       # a larger image: classify resizes it
                paths.append(path)
            guesses = cifarkit.predict.classify(paths, MODEL_DIR, top=3)
            verdict("MP5.8", len(guesses) == 3 and all(len(g[1]) == 3 and g[1][0][1] >= g[1][1][1] for g in guesses),
                    "classify : trois images, leurs trois classes les plus probables, la plus probable d'abord.",
                    "classify doit rendre, pour chaque image, (chemin, [(classe, probabilité)] × top), triées par "
                    "probabilité décroissante.")
            command = [sys.executable, "-m", "cifarkit.predict", *map(str, paths), "--model", str(MODEL_DIR)]
            result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": os.pathsep.join([str(PROJECT), os.environ.get("PYTHONPATH", "")])})
            print("$ python -m cifarkit.predict val_0.png val_1.png val_2.png --model …\n" + (result.stdout or result.stderr)[-800:])
            verdict("MP5.8", result.returncode == 0 and result.stdout.count("\n") == 3,
                    "la ligne de commande classe les trois images, sans le notebook ni ta librairie.",
                    "python -m cifarkit.predict … --model … doit afficher une ligne par image (voir l'erreur ci-dessus ; "
                    "une fonction qui importe mylearn en tête de module empêche le modèle livré de servir seul).")
        else:
            pending("MP5.8", "pas encore de modèle livré : l'étape MP5.5 l'écrit.")
    passed, failed, reasons = run_project_tests()
    names_found = set(re.findall(r"def (test_\w+)", (PROJECT / "test_cifarkit.py").read_text(encoding="utf-8")))
    waiting = [reason for reason in reasons if reason.startswith(PENDING)]
    if waiting:
        pending("MP5.8", f"{len(waiting)} test(s) attendent une fonction de cifarkit pas encore écrite ; les mutants "
                "attendront que tes tests passent.")
    elif len(names_found) < 3:
        pending("MP5.8", f"test_cifarkit.py n'a que {len(names_found)} test(s) : écris-en au moins trois.")
    else:
        verdict("MP5.8", failed == 0 and passed >= 3, f"{passed} tests passent.",
                f"{passed} test(s) passe(nt) et {failed} échoue(nt) : il en faut au moins trois, tous verts.")
        if failed == 0 and passed:
            caught = []
            with tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                for name, (what, *_) in MUTANTS.items():
                    mutant_plugin(folder, name)
                    _, m_failed, _ = run_project_tests(["-p", f"mutant_{name}"], quiet=True,
                                                       env={"PYTHONPATH": os.pathsep.join([str(folder), os.environ.get("PYTHONPATH", "")])})
                    caught.append(m_failed > 0)
                    print(f"   {'attrapé' if m_failed else 'PAS attrapé'} : {what} ({m_failed} test(s) en échec)")
            verdict("MP5.8", sum(caught) >= 2, f"tes tests attrapent {sum(caught)} mutants sur {len(caught)}.",
                    f"tes tests n'attrapent que {sum(caught)} mutant(s) sur {len(caught)} : teste les propriétés que "
                    "l'entraînement ne montre pas (une moyenne par exemple, des statistiques par canal, les poids gardés, "
                    "la taille de la perturbation, la stratification).")
    runs = cifarkit.read_runs(PROJECT / "runs.csv")
    if runs:
        verdict("MP5.8", len(runs) >= 8, f"runs.csv : {len(runs)} runs dans le journal du mode complet.",
                f"runs.csv n'a que {len(runs)} runs : le cahier des charges en demande au moins 8.")
    else:
        pending("MP5.8", "runs.csv n'existe pas encore : le journal du mode complet s'écrit quand le notebook tourne avec "
                f"FAST_MODE = False (en mode rapide : {RUNS.name}, {len(cifarkit.read_runs(RUNS))} runs).")
    readme = check_document("MP5.8", PROJECT / "README.md",
                            {"L'objectif": ["objectif", "goal"], "Le protocole": ["protocole", "protocol"],
                             "Les résultats": ["résultat", "result"], "Les limites": ["limite", "limitation"],
                             "Reproduire": ["reprodui", "reproduc"], "Ce que j'ai appris": ["appris", "learn"]})
    check_document("MP5.8", PROJECT / "MODEL_CARD.md",
                   {"Usage prévu": ["usage", "intended"], "Données": ["donnée", "data"],
                    "Performances": ["performance"], "Limites et risques": ["limite", "risk"]})
    for figure, made_by in (("augmentation.png", "l'étape MP5.2 la crée, en mode complet"),
                            ("architectures.png", "l'étape MP5.4 la crée, en mode complet"),
                            ("courbes_final.png", "l'étape MP5.5 la crée, en mode complet"),
                            ("confusion.png", "l'étape MP5.6 la crée, en mode complet"),
                            ("erreurs.png", "l'étape MP5.6 la crée, en mode complet"),
                            ("fgsm.png", "l'étape MP5.6 la crée, en mode complet")):
        if not (FIGURES / figure).exists():
            pending("MP5.8", f"figures/{figure} n'existe pas encore : {made_by}.")
        elif readme:
            verdict("MP5.8", f"figures/{figure}" in readme, f"le README montre figures/{figure}.",
                    f"le README doit montrer figures/{figure}.")
    if (PROJECT / "results.json").exists():
        verdict("MP5.8", (PROJECT / "model.pt").exists() and (PROJECT / "model.json").exists(),
                "results.json, model.pt et model.json sont livrés.", "il manque model.pt ou model.json à côté de results.json.")
    else:
        pending("MP5.8", "results.json n'existe pas encore : l'étape MP5.5 l'écrit en mode complet, quand le test s'ouvre.")
    openings = data.test_openings(PROJECT)
    if not openings:
        pending("MP5.8", "le test n'est pas encore ouvert : il s'ouvre une fois, à l'étape MP5.5, en mode complet.")
    else:
        verdict("MP5.8", len({entry["description"] for entry in openings}) == 1,
                f"le test n'a été ouvert que pour un seul modèle ({len(openings)} entrée(s) dans test_log.json).",
                f"le test a été ouvert pour {len({e['description'] for e in openings})} modèles différents (test_log.json) : "
                "son score n'est plus une mesure honnête, dis-le dans ton README.")
    checks_done()'''

NOTES = {   # step -> questions for the learner
    "MP5.1": """- Le problème, la métrique, le découpage et la règle des expériences, en quatre phrases :
- Ce que font le MLP et le petit CNN, et pourquoi le CNN a moins de paramètres :""",
    "MP5.2": """- Pourquoi la moyenne et l'écart-type viennent de l'entraînement seul, et pourquoi ils voyagent avec le modèle :
- Les transformations d'augmentation qui conviennent à CIFAR-10, et une qui ne conviendrait pas :""",
    "MP5.3": """- Ce que garde last.pt, et pourquoi chaque pièce est nécessaire à une reprise exacte :
- Ce que montrent les premiers runs (baselines, petit CNN avec et sans augmentation) :""",
    "MP5.4": """- Ce que change chaque variable (stride ou pooling, batchnorm, dropout, raccourcis), à budget égal :
- Ce que tu retiens, et ce que ce petit budget ne permet pas de conclure :""",
    "MP5.5": """- Pourquoi un tirage logarithmique, et ce que montre la recherche :
- La règle de ton choix, et le score de validation du modèle final :
- (mode complet) Le score du test face à la validation :""",
    "MP5.6": """- Les classes les plus confondues, et pourquoi :
- Les 20 erreurs : lesquelles sont excusables, lesquelles ne le sont pas ?
- Ce que montre FGSM, et ce qu'il faudrait pour s'en protéger :""",
    "MP5.7": """- Le ResNet-18 pré-entraîné contre ton modèle : l'accuracy, le coût, les données nécessaires :
- Les limites de cette comparaison (taille des images, données d'ImageNet, licence) :""",
}

OBSERVATIONS: dict[str, str] = {   # the reference solution's answers (fast mode on CPU, and the full mode of results.json)
    "MP5.1": r"""- **Le cadrage** : dix classes équilibrées de photos 32 × 32, l'accuracy comme métrique (le hasard en donne 10 %), 45 000 images d'entraînement et 5 000 de validation tirées par un découpage stratifié (500 par classe), le test de 10 000 images ouvert une fois, en mode complet ; chaque run, même raté, entre dans le journal.
- **Les baselines** : en mode rapide (10 000 images, 3 epochs), le MLP plafonne à 40,9 % et le petit CNN atteint 51,3 % ; en mode complet (45 000 images, 10 epochs), 52,8 % et 72,1 %. Le CNN a 26 fois moins de paramètres (60 362 contre 1 578 506) : chaque filtre 3 × 3 est partagé par toutes les positions de l'image, là où la première couche du MLP relie chacun des 3 072 pixels à chacun de ses 512 neurones. Toujours prédire la même classe donne 10 à 10,5 %.""",
    "MP5.2": r"""- **Les statistiques de l'entraînement seul** : la validation et le test doivent être traités comme des images nouvelles, que le modèle n'a jamais vues ; calculer la moyenne et l'écart-type avec elles, c'est laisser fuir de l'information (une petite fuite ici, mais la règle vaut partout). Ces six nombres (0,4916 ; 0,4823 ; 0,4466 et 0,2471 ; 0,2435 ; 0,2615) font partie du modèle : sans eux, une image normalisée autrement donnerait d'autres logits ; ils voyagent donc dans `model.json`.
- **L'augmentation** : un recadrage de quelques pixels et une symétrie horizontale ne changent pas la classe d'un avion ou d'un chat. Une rotation de 180° ou une symétrie verticale ne conviendrait pas (un bateau à l'envers n'existe pas dans les données), ni une inversion des couleurs, qui ferait d'une scène de jour une image qu'on ne rencontrera jamais.""",
    "MP5.3": r"""- **Ce que garde last.pt** : les poids, l'état de l'optimiseur (les vitesses du momentum), celui du planning (le pas où en est `OneCycleLR`), l'epoch, l'historique, les meilleurs poids et le compteur de patience, l'état aléatoire de PyTorch (le dropout et l'augmentation tirent dedans) et celui du générateur du `DataLoader` (l'ordre des mini-batches). Sans l'un d'eux, la reprise donne d'autres poids : la vérification « 2 + 2 epochs = 4 epochs au bit près » échoue dès qu'une pièce manque.
- **Les premiers runs** : en mode rapide (3 epochs), l'augmentation fait **moins** bien (55,7 % contre 58,9 %) : en si peu d'epochs, l'overfitting n'a pas commencé, et l'augmentation ne fait que rendre l'apprentissage plus difficile. En mode complet (10 epochs sur 45 000 images), elle fait mieux (76,2 % contre 74,5 %). `fit` (SGD de Nesterov, `OneCycleLR`) bat aussi `quick_fit` (Adam constant) sur le même petit CNN : 74,5 % contre 72,1 %.""",
    "MP5.4": r"""- **Une variable à la fois** (mode complet : 20 000 images, 10 epochs ; mode rapide : 6 000 images, 4 epochs) : le VGG avec pooling et batchnorm gagne dans les deux modes (74,8 % ; 51,1 %). Le stride à la place du pooling coûte 4 à 7 points, pour un quart de temps en moins. Sans batchnorm, le même réseau, au même learning rate (0,05), n'apprend presque rien (50,8 % ; 9,3 % en mode rapide, le hasard) : la batchnorm décide du learning rate qu'on peut se permettre. Le dropout coûte 1,5 à 3 points sur un entraînement aussi court, où l'overfitting reste faible. Le petit ResNet (71,1 % ; 46,1 %), un tiers plus lent par epoch, reste derrière sur ce budget.
- **Ce que ce budget ne dit pas** : une seule graine par run, et un entraînement court qui favorise les réseaux qui apprennent vite ; sur 30 epochs, le ResNet pourrait rattraper le VGG, et le dropout servir. La règle choisie (la meilleure validation) prend le VGG avec pooling, en sachant que l'écart avec le ResNet n'est pas établi.""",
    "MP5.5": r"""- **Le tirage logarithmique** : un learning rate de 0,01 et un de 0,02 diffèrent autant qu'un de 0,1 et un de 0,2 ; tirer uniformément entre 0,01 et 0,4 mettrait trois tirages sur quatre au-dessus de 0,1. La recherche (le VGG retenu ; mode complet : 10 000 images, 4 epochs) écarte clairement le learning rate trop petit (0,012 : 48,3 %) ; les cinq autres tiennent en 1,1 point (60,5 à 61,5 %), à peine plus que le bruit de la mesure. En mode rapide (4 000 images, 3 epochs, 400 images de validation), l'ordre est différent (le meilleur run est à 0,094), ce qui montre la même chose : au-delà du learning rate trop petit, la recherche ne départage pas.
- **Le modèle final** : la règle prend le VGG avec pooling et le meilleur run de la recherche. Mode rapide : 71,7 % de validation (12 000 images, 10 epochs, 2,5 min) ; mode complet : 85,8 % (45 000 images, 30 epochs, 28 min sur CPU), la meilleure epoch étant la dernière, car `OneCycleLR` finit à un learning rate presque nul.
- **Le test** (mode complet, ouvert une fois) : 85,0 %, 0,8 point sous la validation, environ un écart-type et demi de la mesure : la validation, qui a servi à tous les choix, est un peu optimiste.""",
    "MP5.6": r"""- **Les confusions** (le test, en mode complet) : chat et chien (213 images dans les deux sens), avion et bateau (88), automobile et camion (85) ; la classe la plus faible est le chat (71,2 %), les meilleures le bateau et l'automobile (93 %). En mode rapide, sur la validation, chat et chien restent en tête (57), suivis d'avion et oiseau (37), puis de chat et grenouille (36) ; les classes les plus faibles sont le chat (50,8 %) et l'oiseau (54,3 %).
- **Les 20 erreurs les plus sûres d'elles** (94 à 98 %) : certaines sont excusables, un bateau vu de très près pris pour un avion, un camion de pompiers dont le label dit « automobile », et au moins deux images dont le label semble faux (une grenouille étiquetée chat, un « cheval » qui ressemble à un avion) ; d'autres non, des chiens debout dans l'herbe pris pour des cerfs : le modèle s'appuie sur le fond autant que sur l'animal. Et il est trop sûr de lui : ses probabilités ne sont pas calibrées.
- **FGSM** : avec ε = 1/255, un niveau de gris par pixel, l'accuracy passe de 86,5 % à 38,9 % (mode complet ; de 71,6 % à 39,8 % en mode rapide), et à 4/255 il ne reste que 7 %. Pour s'en protéger : un entraînement adversarial (des exemples FGSM ou PGD dans les mini-batches), plus coûteux, qui coûte aussi un peu d'accuracy sur les images normales.""",
    "MP5.7": r"""- **Le ResNet-18 pré-entraîné** (gelé, une tête de 5 130 paramètres) : 86,1 % de validation avec 10 000 images agrandies en 224 × 224 (mode complet), contre 85,8 % pour le VGG entraîné sur 45 000 images ; en mode rapide, 78,6 % avec 2 000 images en 112 × 112, contre 71,7 % pour le VGG avec 12 000 images. Il lui faut beaucoup moins de données étiquetées, mais 25 ms de calcul par image en 224 × 224 (6 ms en 112 × 112), contre 0,3 à 0,5 ms pour le VGG, et 11,7 millions de paramètres à stocker.
- **Les limites de la comparaison** : les images de CIFAR-10, agrandies de 32 à 224 pixels, sont floues pour un réseau entraîné sur des photos nettes ; un fine-tuning de tout le réseau (et pas seulement de sa tête) ferait bien mieux, mais coûterait un GPU. Ses poids viennent d'ImageNet, avec sa licence (non commerciale pour les images) et ses biais ; et les deux modèles ne sont pas mesurés sur les mêmes quantités de données.""",
}


def notes_cell(step: str, exercise: bool):
    questions = NOTES[step]
    if exercise or step not in OBSERVATIONS:
        return md(f"✍️ **Tes observations ({step})** : double-clique sur cette cellule pour écrire tes réponses ; "
                  f"elles nourriront ton README.\n\n{questions}")
    return md(f"✍️ **Observations ({step})**\n\n{OBSERVATIONS[step]}")


FOOTER = r'''## ✅ Bilan

**Grille d'évaluation** (sur 20, détaillée dans le cahier des charges) : pipeline de données sans fuite, prétraitement sauvegardé avec le modèle (3) · boucle d'entraînement correcte et reproductible : graines, checkpoints, early stopping, reprise (4) · démarche expérimentale : baselines, journal d'au moins 8 runs, une variable à la fois, conclusions justifiées (4) · performance : au moins 80 % de test en mode complet, ou 65 % de validation en mode rapide (3) · analyse d'erreurs et de robustesse, limites honnêtes (3) · qualité de livraison : README, fiche modèle, code lisible et testé, `predict.py` qui marche (3).

**Pour aller plus loin** : les extensions du cahier des charges (un petit Vision Transformer, Grad-CAM, précision mixte et `torch.compile` sur GPU, l'export ONNX et une démo, CIFAR-10.1 ou tes propres photos, la même démarche pour une série temporelle). La suite du workbook : la partie VI ; ce projet sert aussi de point de départ au chapitre bonus B7 (du notebook à la production).'''


def header_cells(kind: str) -> list:
    rel = STARTER if kind == "exercise" else SOLUTION
    rows = ["| Étape | Contenu | ⏱️ |", "|---|---|---|"] + [f"| {sid} | {title} | {minutes} min |"
                                                            for sid, title, minutes in STEPS]
    fast = ("**Mode rapide et mode complet.** Avec `FAST_MODE = True` (la cellule de setup), chaque expérience tourne sur "
            "un sous-ensemble (10 000 images pour les baselines, 6 000 pour les architectures, 4 000 pour la recherche, "
            "12 000 pour le modèle final, et 2 000 images de validation) : le notebook tient en une dizaine de minutes "
            "sur un CPU à deux cœurs, et vise 65 % de validation. Le mode complet (45 000 images, 30 epochs pour le "
            "modèle final) se lance sur un GPU de Colab (🚀, *Exécution → Modifier le type d'exécution → GPU*) : "
            "quelques minutes ; environ une heure vingt sur un CPU à deux cœurs. Le test de CIFAR-10 ne s'ouvre qu'en "
            "mode complet, une fois, pour la mesure finale (au moins 80 %). Ferme les autres notebooks pendant qu'il "
            "tourne : les convolutions ralentissent beaucoup quand les cœurs sont partagés.")
    if kind == "exercise":
        title = "# Mini-projet MP5 · Un classifieur d'images CIFAR-10 de bout en bout"
        how = ("Le cahier des charges est dans `projets/partie_5_classifieur_cifar10/README.md` : lis-le d'abord. Ce "
               "notebook te guide en huit étapes. Le code durable va dans **ton package** `cifarkit/` (et ses tests, "
               "`test_cifarkit.py`) ; le notebook le vérifie, l'entraîne, compare, note chaque run dans le journal et "
               "trace les figures. Les fonctions à écrire lèvent `NotImplementedError` ; les vérifications affichent ✅ "
               "ou ❌ (des garde-fous, pas une note), et ⏳ tant qu'une étape n'est pas faite. `cifarkit` s'appuie sur "
               "**ta** librairie `mylearn` pour trois fonctions : la liste est dans le cahier des charges, section "
               "« Ta librairie ».\n\n"
               "> Travaille dans **ta copie** (`mon_travail/projets/partie_5_classifieur_cifar10/mp5_classifieur_cifar10.ipynb`, "
               "créée par `python tools/start_chapter.py CP5`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le "
               "badge ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme "
               "l'explique `00_setup/COLAB.md` §2.")
    else:
        title = "# Mini-projet MP5 · Un classifieur d'images CIFAR-10 de bout en bout — solution de référence (exécutée)"
        how = ("La solution de référence, exécutée en mode rapide avec `cifarkit/` de ce dossier et la librairie de "
               "référence. Les résultats du mode complet (`results.json`, `runs.csv`, `model.pt` et `model.json`, les "
               "figures du README, `test_log.json`) viennent d'une exécution de ce notebook avec `FAST_MODE = False`, "
               "sur un CPU à deux cœurs ; le test de CIFAR-10 n'a été ouvert que pour un seul modèle. Tes chiffres n'ont "
               "pas à être identiques : d'autres choix donnent d'autres valeurs, tout aussi justes s'ils sont bien "
               "mesurés. Le README de portfolio et la fiche modèle rédigés à partir de ces résultats sont dans ce dossier.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n{fast}\n\n" + "\n".join(rows))]


def build(kind: str) -> list:
    exercise = kind == "exercise"
    cells = header_cells(kind)
    cells.append(setup_cell("exercise", chapter="CP5") if exercise else setup_cell("solution"))
    cells.append(code(PROJECT_STARTER if exercise else PROJECT_SOLUTION))
    cells.append(code(TOOLS))
    cells += [md(MP51_MD), code(MP51_CHECK), code(MP51_BASELINES), notes_cell("MP5.1", exercise)]
    cells += [md(MP52_MD), code(MP52_CHECK), notes_cell("MP5.2", exercise)]
    cells += [md(MP53_MD), code(MP53_CHECK), code(MP53_RUNS), notes_cell("MP5.3", exercise)]
    cells += [md(MP54_MD), code(MP54_CHECK), code(MP54_COMPARE), notes_cell("MP5.4", exercise)]
    cells += [md(MP55_MD), code(MP55_TODO if exercise else MP55_SOLUTION), code(MP55_SEARCH),
              code(MP55_TODO_FINAL if exercise else MP55_SOLUTION_FINAL), code(MP55_TRAIN), code(MP55_TEST),
              notes_cell("MP5.5", exercise)]
    cells += [md(MP56_MD), code(MP56_CHECK), notes_cell("MP5.6", exercise)]
    cells += [md(MP57_MD), code(MP57_TODO if exercise else MP57_SOLUTION), code(MP57_RUN), notes_cell("MP5.7", exercise)]
    cells += [md(MP58_MD), code(MP58_CHECK), md(FOOTER)]
    return cells


def main() -> int:
    written = write_starter_package()
    write_notebook(STARTER, build("exercise"))
    write_notebook(SOLUTION, build("solution"))
    print(f"✅ depart/cifarkit/ ({', '.join(written)}), {STARTER} and {SOLUTION} written ({len(STEPS)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
