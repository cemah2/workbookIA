#!/usr/bin/env python
"""Build the mini-project MP4 from a single source (used by Claude).

    python tools/chapters/build_mp4.py
    python tools/run_all_notebooks.py projets/partie_4_mini_framework/solution/mp4_mini_framework.ipynb --inplace
    python tools/run_all_notebooks.py projets/partie_4_mini_framework/depart/mp4_mini_framework.ipynb

Writes the two notebooks (depart/ and solution/mp4_mini_framework.ipynb) and the starter package
depart/minifw/, generated from the reference solution/minifw/: same constants, classes, signatures
and docstrings; ``__init__`` methods and the functions whose docstring starts with "Provided" are
kept; private helpers are removed; every other function raises NotImplementedError with the step
of the notebook where it is written. ``python tools/start_chapter.py CP4`` copies depart/ into
mon_travail/projets/. Steps MP4.1 to MP4.7 follow the brief (docs/syllabus/data/cp4.json).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import ROOT, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "projets/partie_4_mini_framework"
NOTEBOOK = "mp4_mini_framework.ipynb"
STARTER = f"{FOLDER}/depart/{NOTEBOOK}"
SOLUTION = f"{FOLDER}/solution/{NOTEBOOK}"
PACKAGE_SOLUTION = ROOT / FOLDER / "solution" / "minifw"
PACKAGE_STARTER = ROOT / FOLDER / "depart" / "minifw"

STEPS = [("MP4.1", "Cahier des charges, structure du package, environnement et graines", 30),
         ("MP4.2", "Les couches Dense, activations, Dropout et BatchNorm1d (forward/backward) sur mylearn, comparées à PyTorch", 90),
         ("MP4.3", "Les losses sur logits et le conteneur Sequential : un réseau entier comparé à PyTorch et aux différences finies", 60),
         ("MP4.4", "Tes tests d'équivalence avec PyTorch, mis à l'épreuve par six mutants", 90),
         ("MP4.5", "La boucle d'entraînement : mini-batches, AdamW, warmup + cosinus, clipping, early stopping ; les lunes", 60),
         ("MP4.6", "Fashion-MNIST : un entraînement identique à PyTorch, minifw contre PyTorch, l'ablation dropout/batchnorm, le test ouvert une fois", 105),
         ("MP4.7", "README, figures, limites ; le notebook relancé de zéro", 45)]

# Step of the notebook where each public function of the package is written (starter package).
TODO_STEPS = {
    "Dense.forward": "MP4.2", "Dense.backward": "MP4.2", "Dense.parameters": "MP4.2", "Dense.gradients": "MP4.2",
    "Activation.forward": "MP4.2", "Activation.backward": "MP4.2",
    "Dropout.forward": "MP4.2", "Dropout.backward": "MP4.2",
    "BatchNorm1d.forward": "MP4.2", "BatchNorm1d.backward": "MP4.2", "BatchNorm1d.parameters": "MP4.2",
    "BatchNorm1d.gradients": "MP4.2",
    "SoftmaxCrossEntropy.__call__": "MP4.3", "MSELoss.__call__": "MP4.3",
    "Sequential.forward": "MP4.3", "Sequential.backward": "MP4.3", "Sequential.parameters": "MP4.3",
    "Sequential.gradients": "MP4.3", "Sequential.train": "MP4.3", "Sequential.eval": "MP4.3",
    "Sequential.n_parameters": "MP4.3", "Sequential.state_dict": "MP4.3", "Sequential.load_state_dict": "MP4.3",
    "iterate_minibatches": "MP4.5", "evaluate": "MP4.5", "fit": "MP4.5", "library_versions": "MP4.1",
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
    """One starter file of minifw: bodies replaced by NotImplementedError, except __init__ and "Provided" functions."""
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
    """depart/minifw/*.py from solution/minifw/*.py; returns the names written."""
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
PROJECT_STARTER = r'''# The project folder: your copy (mon_travail/projets/...), where minifw/, data.py and test_minifw.py live
PROJECT_NAME = "partie_4_mini_framework"
STARTER_KIT = ROOT / "projets" / PROJECT_NAME / "depart"
candidates = [Path.cwd(), ROOT / "mon_travail" / "projets" / PROJECT_NAME, STARTER_KIT]
PROJECT = next(folder for folder in candidates if (folder / "minifw" / "__init__.py").exists() and (folder / "data.py").exists())
FIGURES = PROJECT / "figures"
if PROJECT.resolve() == STARTER_KIT.resolve():
    print("⚠️ Ce dossier est le kit de départ du dépôt, pas ta copie : lance python tools/start_chapter.py CP4, puis "
          "ouvre mon_travail/projets/partie_4_mini_framework/mp4_mini_framework.ipynb.")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import data  # noqa: E402

try:
    import minifw  # noqa: E402   (your package: it imports your mylearn library)
except ModuleNotFoundError as exc:
    minifw = None
    print(f"⚠️ {exc} : lance d'abord python tools/start_chapter.py CP4, qui crée ta librairie mylearn.")
except Exception as exc:                       # a syntax error in minifw must not stop the notebook
    minifw = None
    print(f"⚠️ minifw ne s'importe pas : {type(exc).__name__}: {exc}. Corrige-le ; l'étape MP4.1 le rechargera.")
print("Dossier du projet :", PROJECT)'''

PROJECT_SOLUTION = r'''# The project folder of the reference solution
PROJECT_NAME = "partie_4_mini_framework"
PROJECT = ROOT / "projets" / PROJECT_NAME / "solution"
FIGURES = PROJECT / "figures"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import data  # noqa: E402
import minifw  # noqa: E402

print("Dossier du projet :", PROJECT.relative_to(ROOT))'''

TOOLS = r'''# Tools of the notebook (run this cell)
import importlib
import json
import math
import re
import tempfile
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
except ImportError:
    torch = nn = None
    print("⏳ PyTorch n'est pas installé : les comparaisons avec PyTorch attendront (sur Colab, il l'est).")

N_FASHION = 20_000 if FAST_MODE else 50_000    # training images of Fashion-MNIST in MP4.6 (the 10 000 last: validation)
EPOCHS = 6 if FAST_MODE else 15                # epochs of the main training of MP4.6
N_ABLATION = 10_000 if FAST_MODE else 50_000   # training images of the ablation (MP4.6 c)
EPOCHS_ABLATION = 5 if FAST_MODE else 12
SEED = 0                                       # the seed of every draw of the notebook

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


def filled(*values):
    """True when none of the values is still `...` (or None)."""
    return all(value is not ... and value is not None for value in values)


def ready(*names):
    """True when the variables of the earlier steps exist; otherwise print which ones are missing."""
    missing = [name for name in names if name not in globals()]
    if missing:
        print(f"⏳ il manque {', '.join(missing)} : exécute d'abord les cellules des étapes précédentes, vérifications "
              "comprises (ce sont souvent elles qui créent ces variables).")
    return not missing


def minifw_ok():
    """Import your latest saved minifw again (its four modules, in order); False, with the reason, when it cannot be
    imported yet."""
    global minifw
    try:
        for name in ("minifw.layers", "minifw.losses", "minifw.model", "minifw.train"):
            module = sys.modules.get(name)
            importlib.reload(module) if module is not None else importlib.import_module(name)
        minifw = importlib.reload(sys.modules["minifw"]) if "minifw" in sys.modules else importlib.import_module("minifw")
    except Exception as exc:
        print(f"⚠️ minifw ne s'importe pas : {type(exc).__name__}: {exc}")
        return False
    return True


def gap(got, want):
    """The largest absolute gap between two arrays (inf when the shapes differ or got is not an array)."""
    try:
        got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return float("inf")
    return float(np.abs(got - want).max()) if got.shape == want.shape and got.size else float("inf")


def same(step, label, got, want, tol=1e-8):
    """✅ when got and want agree to tol (absolute), with the gap; ❌ with the gap and the shapes otherwise."""
    g = gap(got, want)
    def shape(a):
        try:
            return tuple(np.shape(np.asarray(a, dtype=float)))
        except (TypeError, ValueError):
            return None
    shape_got, shape_want = shape(got), shape(want)
    if shape_got is not None and shape_got != shape_want:
        return verdict(step, False, "", f"{label} : forme {shape_got}, celle de PyTorch est {shape_want}.")
    return verdict(step, g <= tol, f"{label} : comme PyTorch (écart {g:.1e}).", f"{label} : écart à PyTorch {g:.2e}.")


def tensor(a, grad=False):
    """A float64 tensor of a NumPy array."""
    return torch.tensor(np.asarray(a, dtype=np.float64), requires_grad=grad)


def torch_twin(model, dtype=None):
    """An nn.Sequential with the same layers and the SAME weights as a minifw Sequential (float64 by default)."""
    dtype = dtype or torch.float64
    modules = []
    for layer in model.layers:
        name = type(layer).__name__
        if name == "Dense":
            module = nn.Linear(*layer.W.shape, dtype=torch.float64)       # float64 BEFORE the copy: no rounding
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(layer.W.T.copy()))
                module.bias.copy_(torch.from_numpy(layer.b.copy()))
        elif name == "BatchNorm1d":
            module = nn.BatchNorm1d(len(layer.gamma), momentum=layer.momentum, eps=layer.eps, dtype=torch.float64)
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(layer.gamma.copy()))
                module.bias.copy_(torch.from_numpy(layer.beta.copy()))
                module.running_mean.copy_(torch.from_numpy(layer.running_mean.copy()))
                module.running_var.copy_(torch.from_numpy(layer.running_var.copy()))
        elif name == "Dropout":
            module = nn.Dropout(layer.p)
        elif name in ("ReLU", "Tanh") or getattr(layer, "name", None) in ("relu", "tanh"):
            module = nn.ReLU() if (name == "ReLU" or getattr(layer, "name", None) == "relu") else nn.Tanh()
        else:
            raise ValueError(f"no PyTorch twin for the layer {name}")
        modules.append(module)
    return nn.Sequential(*modules).to(dtype)


def save(fig, name):
    """Save a figure in the figures/ folder of the project (the README shows it)."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=110, bbox_inches="tight")


def run_project_tests(extra=(), env=None, quiet=False):
    """Run the tests of the project folder (test_minifw.py) and print the failures (⏳ for a function not written yet);
    return (passed, failed, reasons), reasons being the first line of each failure."""
    command = [sys.executable, "-m", "pytest", str(PROJECT / "test_minifw.py"), "-q", "-p", "no:cacheprovider",
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


MUTANTS = {   # name -> (what the mutant breaks, the patch applied to the classes of minifw before the tests run)
    "dense_backward": ("Dense.backward rend dL/dx multiplié par 1,01",
                       "orig = minifw.layers.Dense.backward\nminifw.layers.Dense.backward = lambda self, dout: 1.01 * orig(self, dout)"),
    "dense_gradients": ("Dense.gradients rend dW faux de 1 %",
                        "orig = minifw.layers.Dense.gradients\nminifw.layers.Dense.gradients = lambda self: [g * 1.01 for g in orig(self)]"),
    "batchnorm_forward": ("BatchNorm1d.forward ajoute 0,001 à sa sortie",
                          "orig = minifw.layers.BatchNorm1d.forward\nminifw.layers.BatchNorm1d.forward = lambda self, x: orig(self, x) + 1e-3"),
    "dropout_eval": ("Dropout.forward multiplie par 1 − p même en évaluation",
                     "orig = minifw.layers.Dropout.forward\nminifw.layers.Dropout.forward = lambda self, x: orig(self, x) if self.training else (1 - self.p) * np.asarray(x, dtype=float)"),
    "loss_gradient": ("SoftmaxCrossEntropy rend un gradient deux fois trop grand",
                      "orig = minifw.losses.SoftmaxCrossEntropy.__call__\nminifw.losses.SoftmaxCrossEntropy.__call__ = lambda self, z, y: (orig(self, z, y)[0], 2 * orig(self, z, y)[1])"),
    "sequential_backward": ("Sequential.backward oublie la dernière couche",
                            "def bad(self, dout):\n    for layer in reversed(self.layers[:-1]):\n        dout = layer.backward(dout)\n"
                            "    return dout\nminifw.model.Sequential.backward = bad"),
}


def mutant_plugin(folder, name):
    """A pytest plugin that patches minifw with one mutant once the tests are collected (minifw imported by then)."""
    patch = MUTANTS[name][1].replace("\n", "\n    ")
    (folder / f"mutant_{name}.py").write_text(
        "import numpy as np\n\n\ndef pytest_collection_modifyitems(session, config, items):\n    import minifw\n"
        "    import minifw.layers\n    import minifw.losses\n    import minifw.model\n    " + patch + "\n", encoding="utf-8")


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
# MP4.1
# ---------------------------------------------------------------------------
MP41_MD = r'''## MP4.1 · Cahier des charges, structure du package, environnement et graines ⏱️ 30 min

**Objectif :** savoir ce que tu construis et comment tu le prouveras, avant d'écrire une ligne de code.

a) **Lis le cahier des charges** (`projets/partie_4_mini_framework/README.md`), puis écris la section « L'objectif » de ton `README.md` : ce que fait `minifw`, ce qu'il ne fait pas (pas de GPU, pas d'autograd, pas de convolution), et comment tu prouves qu'il est juste (des tests d'équivalence avec PyTorch).
b) **La structure** : le package `minifw/` a quatre modules, `layers.py` (les couches), `losses.py` (les losses), `model.py` (le conteneur `Sequential`), `train.py` (la boucle d'entraînement) ; leurs docstrings sont le contrat. `minifw` ne recode rien de ce que ta librairie `mylearn` sait déjà faire : il l'emballe dans des objets.
c) Écris `library_versions` dans `train.py` (sa docstring). La vérification contrôle l'environnement et les graines : toutes les expériences du notebook tirent leurs nombres de générateurs `np.random.default_rng(...)` créés avec des graines écrites, jamais de l'état global de NumPy.'''

MP41_CHECK = r'''checks_of("MP4.1")
if minifw_ok():
    files = ["__init__.py", "layers.py", "losses.py", "model.py", "train.py"]
    missing = [name for name in files if not (PROJECT / "minifw" / name).exists()]
    verdict("MP4.1", not missing, "le package minifw a ses cinq fichiers.", f"il manque minifw/{', minifw/'.join(missing)}.")
    with attempt("MP4.1"):
        versions = minifw.library_versions()
        keys = {"python", "numpy"} | ({"torch"} if torch is not None else set())
        ok = isinstance(versions, dict) and keys <= set(versions)
        verdict("MP4.1", ok, f"library_versions() : {versions}." if ok else "",
                f"library_versions() doit rendre un dict avec au moins les clés {', '.join(sorted(keys))} ; reçu {versions!r}.")
        verdict("MP4.1", ok and versions["numpy"] == np.__version__, "la version de NumPy est celle qui tourne.",
                "la version de NumPy rendue n'est pas celle du noyau (np.__version__).")
    checks_done()'''

# ---------------------------------------------------------------------------
# MP4.2
# ---------------------------------------------------------------------------
MP42_MD = r'''## MP4.2 · Les couches Dense, activations, Dropout et BatchNorm1d (forward/backward) sur mylearn, comparées à PyTorch ⏱️ 90 min

**Objectif :** des couches qui gardent ce qu'il faut de la passe avant pour calculer, à la demande, les gradients de la passe arrière.

Dans `minifw/layers.py`, écris les méthodes de `Dense`, `Activation`, `Dropout` et `BatchNorm1d` (les `__init__` sont donnés : ils fixent les noms des attributs). Le contrat commun est dans la docstring du module : `forward(x)` calcule la sortie et garde le cache, `backward(dout)` range les gradients des paramètres **dans** la couche et rend $\partial L / \partial x$, `parameters()` et `gradients()` rendent les tableaux eux-mêmes, dans le même ordre, et l'attribut `training` change le comportement de `Dropout` et de `BatchNorm1d`. Chaque méthode appelle la fonction de ta librairie qui fait le calcul :

| Couche | `forward` | `backward` | ta librairie |
|---|---|---|---|
| `Dense` | `dense_forward` | `dense_backward` | ch. 16 (16.13), ch. 18 (18.15) |
| `Activation`, `ReLU`, `Tanh` | la fonction de `get_activation` | sa dérivée, en $z$ | ch. 17 (17.12 à 17.14) |
| `Dropout` | `dropout_forward` | `dropout_backward` | ch. 20 (20.16) |
| `BatchNorm1d` | `batchnorm_forward` | `batchnorm_backward` | ch. 20 (20.22) |

La vérification compare chaque couche à son équivalent PyTorch (`nn.Linear`, `torch.relu`, `nn.Dropout`, `nn.BatchNorm1d`), avec les mêmes poids, en `float64` : les sorties et les gradients doivent coïncider à $10^{-8}$ près.'''

MP42_CHECK = r'''checks_of("MP4.2")
if minifw_ok() and torch is not None:
    from minifw import BatchNorm1d, Dense, Dropout, ReLU, Tanh
    rng = np.random.default_rng(SEED)
    with attempt("MP4.2"):                       # Dense
        layer = Dense(5, 3, rng=np.random.default_rng(42))
        verdict("MP4.2", layer.W.shape == (5, 3) and np.array_equal(layer.W, Dense(5, 3, rng=np.random.default_rng(42)).W),
                "Dense : W de forme (n_in, n_out), le même avec la même graine.",
                "Dense(5, 3, rng) doit créer W de forme (5, 3), le même pour deux générateurs de même graine.")
        x, dout = rng.normal(size=(4, 5)), rng.normal(size=(4, 3))
        lin = nn.Linear(5, 3).double()
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(layer.W.T.copy()))
            lin.bias.copy_(torch.from_numpy(layer.b.copy()))
        xt = tensor(x, grad=True)
        out_t = lin(xt)
        out_t.backward(tensor(dout))
        same("MP4.2", "Dense.forward", layer.forward(x), out_t.detach())
        same("MP4.2", "Dense.backward, dL/dx", layer.backward(dout), xt.grad)
        same("MP4.2", "Dense : dW après backward", layer.dW, lin.weight.grad.T)
        same("MP4.2", "Dense : db après backward", layer.db, lin.bias.grad)
        params, grads = layer.parameters(), layer.gradients()
        verdict("MP4.2", len(params) == 2 and params[0] is layer.W and params[1] is layer.b and grads[0] is layer.dW,
                "Dense.parameters() rend [W, b] eux-mêmes, et gradients() [dW, db].",
                "parameters() doit rendre [W, b], les tableaux eux-mêmes (l'optimiseur les modifie sur place), et "
                "gradients() [dW, db] dans le même ordre.")
    for cls, fn in ((ReLU, torch.relu), (Tanh, torch.tanh)):
        with attempt("MP4.2"):                   # the activations
            x, dout = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
            x[0, 0] = 0.0
            xt = tensor(x, grad=True)
            out_t = fn(xt)
            out_t.backward(tensor(dout))
            act = cls()
            same("MP4.2", f"{cls.__name__}.forward", act.forward(x), out_t.detach())
            same("MP4.2", f"{cls.__name__}.backward", act.backward(dout), xt.grad)
    with attempt("MP4.2"):                       # Dropout
        drop = Dropout(0.25, rng=np.random.default_rng(1))
        out = drop.forward(np.ones((400, 50)))
        values = set(np.round(np.unique(out), 10))
        verdict("MP4.2", values <= {0.0, round(1 / 0.75, 10)} and abs((out == 0).mean() - 0.25) < 0.01,
                f"Dropout(0,25) en entraînement : {(out == 0).mean():.1%} de zéros, les autres valeurs à 1/(1 − p).",
                f"Dropout(0,25) en entraînement : environ 25 % de zéros et les autres valeurs à 1/(1 − p) = 1,3333 ; "
                f"reçu les valeurs {[float(v) for v in sorted(values)[:5]]} et {(out == 0).mean():.1%} de zéros.")
        again = Dropout(0.25, rng=np.random.default_rng(1)).forward(np.ones((400, 50)))
        verdict("MP4.2", np.array_equal(again, out),
                "Dropout tire ses masques du générateur reçu (même graine, mêmes masques).",
                "deux Dropout(0,25, rng=np.random.default_rng(1)) doivent tirer les mêmes masques : tire le masque avec "
                "self.rng (le générateur reçu), jamais avec l'état global de NumPy.")
        back = drop.backward(np.ones((400, 50)))
        verdict("MP4.2", np.array_equal(back == 0, out == 0), "Dropout.backward : le masque de la dernière passe avant.",
                "Dropout.backward doit multiplier dout par le masque de la DERNIÈRE passe avant (des zéros aux mêmes places).")
        drop.training = False
        x = rng.normal(size=(5, 3))
        same("MP4.2", "Dropout en évaluation (l'identité)", drop.forward(x), x)
    with attempt("MP4.2"):                       # BatchNorm1d
        bnorm = BatchNorm1d(4)
        bnorm.gamma[...] = rng.normal(size=4)
        bnorm.beta[...] = rng.normal(size=4)
        bn = nn.BatchNorm1d(4).double()
        with torch.no_grad():
            bn.weight.copy_(torch.from_numpy(bnorm.gamma.copy()))
            bn.bias.copy_(torch.from_numpy(bnorm.beta.copy()))
        x, dout = rng.normal(2.0, 3.0, size=(16, 4)), rng.normal(size=(16, 4))
        xt = tensor(x, grad=True)
        out_t = bn(xt)
        out_t.backward(tensor(dout))
        same("MP4.2", "BatchNorm1d.forward en entraînement", bnorm.forward(x), out_t.detach())
        same("MP4.2", "BatchNorm1d.backward, dL/dx", bnorm.backward(dout), xt.grad)
        same("MP4.2", "BatchNorm1d : dgamma et dbeta", np.r_[bnorm.dgamma, bnorm.dbeta], np.r_[bn.weight.grad, bn.bias.grad])
        same("MP4.2", "BatchNorm1d : statistiques glissantes", np.r_[bnorm.running_mean, bnorm.running_var],
             np.r_[bn.running_mean, bn.running_var])
        bnorm.training = False
        bn.eval()
        x = rng.normal(size=(3, 4))
        running = np.r_[bnorm.running_mean, bnorm.running_var].copy()
        same("MP4.2", "BatchNorm1d en évaluation", bnorm.forward(x), bn(tensor(x)).detach())
        verdict("MP4.2", np.array_equal(np.r_[bnorm.running_mean, bnorm.running_var], running),
                "en évaluation, les statistiques glissantes ne bougent pas.",
                "une passe avant en évaluation a modifié running_mean ou running_var : elles ne se mettent à jour qu'en "
                "entraînement (sinon, chaque évaluation fuit dans le modèle).")
        verdict("MP4.2", len(bnorm.parameters()) == 2 and bnorm.parameters()[0] is bnorm.gamma,
                "BatchNorm1d.parameters() rend [gamma, beta] (les statistiques glissantes sont des buffers).",
                "BatchNorm1d.parameters() doit rendre [gamma, beta], eux-mêmes ; running_mean et running_var n'y sont pas.")
    checks_done()
elif torch is None:
    print("⏳ MP4.2 : la vérification compare tes couches à PyTorch, qui n'est pas installé.")'''

# ---------------------------------------------------------------------------
# MP4.3
# ---------------------------------------------------------------------------
MP43_MD = r'''## MP4.3 · Les losses sur logits et le conteneur Sequential : un réseau entier comparé à PyTorch et aux différences finies ⏱️ 60 min

**Objectif :** un conteneur qui enchaîne les passes avant et arrière, et des losses calculées sur les logits.

a) Dans `minifw/losses.py`, écris `SoftmaxCrossEntropy.__call__` et `MSELoss.__call__` : la loss **moyenne** (un `float`) et son gradient par rapport à la sortie du modèle, avec `softmax_cross_entropy` et `mse_loss` de ta librairie (ch. 18).
b) Dans `minifw/model.py`, écris les méthodes de `Sequential` : `forward` enchaîne les couches, `backward` les parcourt **à l'envers** (c'est la règle de la chaîne, couche par couche), `parameters` et `gradients` concatènent ceux des couches dans le même ordre, `train` et `eval` changent le mode de chaque couche, `n_parameters` compte les valeurs apprises, `state_dict` copie paramètres **et** buffers, `load_state_dict` les recopie **sur place** (l'optimiseur garde des références vers ces tableaux).

La vérification compare un réseau entier (6-8-8-3, avec une batchnorm, ReLU et tanh) à son jumeau PyTorch : la loss et les huit gradients. Elle vérifie aussi un gradient par différences finies avec `gradient_check` de ta librairie (ch. 18), sans PyTorch.'''

MP43_CHECK = r'''checks_of("MP4.3")
if minifw_ok():
    from minifw import BatchNorm1d, Dense, ReLU, Sequential, SoftmaxCrossEntropy, Tanh
    from mylearn.nn.backward import gradient_check
    rng = np.random.default_rng(SEED)
    with attempt("MP4.3"):
        layers = [Dense(6, 8, rng=rng), BatchNorm1d(8), ReLU(), Dense(8, 8, rng=rng), Tanh(), Dense(8, 3, rng=rng)]
        model = Sequential(*layers)
        x, y = rng.normal(size=(12, 6)), rng.integers(0, 3, size=12)
        loss, dout = SoftmaxCrossEntropy()(model(x), y)
        verdict("MP4.3", isinstance(loss, float) and np.shape(dout) == (12, 3),
                "la loss est un float, son gradient a la forme des logits (12, 3).",
                f"SoftmaxCrossEntropy doit rendre (un float, un tableau (12, 3)) ; reçu {type(loss).__name__} et {np.shape(dout)}.")
        model.backward(dout)
        grads = [g.copy() for g in model.gradients()]
        shapes = [tuple(p.shape) for p in model.parameters()]
        verdict("MP4.3", shapes == [(6, 8), (8,), (8,), (8,), (8, 8), (8,), (8, 3), (3,)],
                "parameters() : W, b de chaque Dense et gamma, beta de la batchnorm, dans l'ordre des couches.",
                f"parameters() doit rendre W, b, gamma, beta… dans l'ordre des couches ; reçu les formes {shapes}.")
        verdict("MP4.3", model.parameters()[0] is layers[0].W and model.gradients()[0] is layers[0].dW,
                "parameters() et gradients() rendent les tableaux des couches eux-mêmes.",
                "model.parameters()[0] doit être layers[0].W lui-même (et gradients()[0], layers[0].dW), pas une copie : "
                "l'optimiseur modifie ces tableaux sur place.")
        verdict("MP4.3", model.n_parameters() == 6 * 8 + 8 + 16 + 8 * 8 + 8 + 8 * 3 + 3,
                f"n_parameters() = {model.n_parameters()}.", f"n_parameters() rend {model.n_parameters()!r} : attendu 171.")
        W = layers[0].W                              # finite differences on the first weights, with your gradient_check
        error = gradient_check(lambda: SoftmaxCrossEntropy()(model(x), y)[0], W, grads[0], n_checks=10,
                               rng=np.random.default_rng(1))
        verdict("MP4.3", error < 1e-6, f"différences finies sur W de la première couche : erreur relative {error:.1e}.",
                f"différences finies sur W de la première couche : erreur relative {error:.1e} (attendu < 1e-6) : la "
                "passe arrière ne donne pas la dérivée de la loss calculée par la passe avant.")
        if torch is not None:
            twin = torch_twin(Sequential(*layers))
            loss_t = nn.functional.cross_entropy(twin(tensor(x)), torch.from_numpy(y))
            loss_t.backward()
            same("MP4.3", "la loss du réseau entier", loss, loss_t.item())
            want = []
            for module in twin:
                if isinstance(module, nn.Linear):
                    want += [module.weight.grad.T, module.bias.grad]
                elif isinstance(module, nn.BatchNorm1d):
                    want += [module.weight.grad, module.bias.grad]
            worst = max(gap(g, w) for g, w in zip(grads, want)) if len(grads) == len(want) else float("inf")
            verdict("MP4.3", worst < 1e-8, f"les {len(want)} gradients du réseau entier, comme PyTorch (écart {worst:.1e}).",
                    f"gradients du réseau entier : écart à PyTorch {worst:.2e} (ou un nombre de gradients différent).")
        model.eval()
        verdict("MP4.3", all(not layer.training for layer in model.layers) and not model.training,
                "eval() passe chaque couche en mode évaluation.", "eval() doit mettre training = False dans chaque couche.")
        model.train()
        state = model.state_dict()
        verdict("MP4.3", "1.running_mean" in state and state["0.W"] is not layers[0].W,
                "state_dict() copie les paramètres et les buffers.",
                "state_dict() doit contenir des COPIES des paramètres et des buffers (« 1.running_mean »…).")
        layers[0].W += 1.0
        model.load_state_dict(state)
        verdict("MP4.3", model.layers[0].W is W and np.array_equal(W, state["0.W"]),
                "load_state_dict() recopie sur place.", "load_state_dict() doit recopier les valeurs DANS les tableaux existants.")
    checks_done()'''

# ---------------------------------------------------------------------------
# MP4.4
# ---------------------------------------------------------------------------
MP44_MD = r'''## MP4.4 · Tes tests d'équivalence avec PyTorch, mis à l'épreuve par six mutants ⏱️ 90 min

**Objectif :** des tests qui prouvent que `minifw` calcule ce que calcule PyTorch, et qui échouent quand il se trompe.

Complète `test_minifw.py` (deux tests d'exemple sont donnés) : **au moins dix tests**, chacun avec un nom clair et un message dont la première ligne dit ce qui était attendu, qui couvrent `Dense`, une activation, `Dropout`, `BatchNorm1d` (entraînement **et** évaluation), `SoftmaxCrossEntropy`, `Sequential` (le réseau entier, ses gradients) et un pas de `mylearn.optim.AdamW` sur les paramètres du modèle contre `torch.optim.AdamW` ; un test du clipping de `fit` contre la même boucle écrite en PyTorch est un bon onzième. PyTorch est l'oracle : il vérifie, `minifw` calcule.

La vérification lance tes tests, puis les relance contre six **mutants** de `minifw` (une erreur glissée dans une couche, la loss ou le conteneur) : un bon jeu de tests en attrape au moins cinq sur six.'''

MP44_CHECK = r'''checks_of("MP4.4")
EXAMPLE_TESTS = {"test_dense_forward_matches_nn_linear", "test_relu_backward_matches_autograd"}   # given in the kit
if minifw_ok():
    passed, failed, reasons = run_project_tests()
    names_found = set(re.findall(r"(?:def|class) ((?:test_|Test)\w+)", (PROJECT / "test_minifw.py").read_text(encoding="utf-8")))
    waiting = [reason for reason in reasons if reason.startswith(PENDING)]
    if waiting:
        pending("MP4.4", f"pas encore fait : {len(waiting)} test(s) attendent une fonction de minifw pas encore écrite "
                "(les étapes MP4.2 et MP4.3) ; les mutants attendront que tes tests passent.")
    elif names_found <= EXAMPLE_TESTS:
        pending("MP4.4", "pas encore fait : test_minifw.py n'a encore que les deux tests d'exemple ; écris les tiens "
                "(au moins dix en tout).")
    else:
        verdict("MP4.4", failed == 0 and passed >= 10, f"{passed} tests passent.",
                f"{passed} test(s) passe(nt) et {failed} échoue(nt) : il en faut au moins 10, tous verts.")
    names = " ".join(names_found).lower()
    for topic, words in {"Dense": ["dense"], "une activation": ["relu", "tanh", "activation"], "Dropout": ["dropout"],
                         "BatchNorm1d": ["batchnorm", "batch_norm"], "la loss": ["cross_entropy", "loss"],
                         "Sequential": ["sequential", "model", "network", "reseau"], "AdamW": ["adamw", "optim"]}.items():
        if waiting or names_found <= EXAMPLE_TESTS:
            break                                        # the step is not done yet: the topics will be checked then
        verdict("MP4.4", any(word in names for word in words), f"un test au moins porte sur {topic}.",
                f"aucun nom de test ne parle de {topic} (un nom qui contient {' ou '.join(words)}).")
    if failed == 0 and passed and not names_found <= EXAMPLE_TESTS:
        caught = []
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name, (what, _) in MUTANTS.items():
                mutant_plugin(folder, name)
                m_passed, m_failed, _ = run_project_tests(["-p", f"mutant_{name}"], quiet=True,
                                                          env={"PYTHONPATH": os.pathsep.join([str(folder), os.environ.get("PYTHONPATH", "")])})
                caught.append(m_failed > 0)
                print(f"   {'attrapé' if m_failed else 'PAS attrapé'} : {what} ({m_failed} test(s) en échec)")
        verdict("MP4.4", sum(caught) >= 5, f"tes tests attrapent {sum(caught)} mutants sur {len(caught)}.",
                f"tes tests n'attrapent que {sum(caught)} mutants sur {len(caught)} : compare aussi les gradients "
                "(pas seulement les sorties), l'évaluation (pas seulement l'entraînement) et la passe arrière du réseau entier.")
    checks_done()'''

# ---------------------------------------------------------------------------
# MP4.5
# ---------------------------------------------------------------------------
MP45_MD = r'''## MP4.5 · La boucle d'entraînement : mini-batches, AdamW, warmup + cosinus, clipping, early stopping ; les lunes ⏱️ 60 min

**Objectif :** la boucle d'entraînement complète, qui fait apprendre un réseau, et chacune de ses pièces vérifiée.

Dans `minifw/train.py`, écris `iterate_minibatches`, `evaluate` et `fit` (leurs docstrings). `fit` reçoit un optimiseur de ta librairie (`mylearn.optim.AdamW(model.parameters(), ...)`) : à chaque pas, il règle `optimizer.lr` avec `cosine_lr` (ch. 19), rogne les gradients avec `clip_grad_norm` (ch. 19), puis appelle `optimizer.step(model.gradients())` ; après chaque epoch, il mesure la validation, et `EarlyStopping` (ch. 20) décide de l'arrêt et des poids à recharger.

La vérification contrôle l'ordre des mini-batches, le planning du learning rate, le clipping et l'early stopping, puis entraîne un réseau 2-32-32-2 sur 1 000 points de deux demi-lunes (800 pour l'entraînement, 200 pour la validation) et trace sa frontière (`figures/moons_frontiere.png`).'''

MP45_CHECK = r'''checks_of("MP4.5")
if minifw_ok():
    from minifw import Dense, ReLU, Sequential, SoftmaxCrossEntropy, evaluate, fit, iterate_minibatches
    from mylearn.optim import AdamW, cosine_lr
    with attempt("MP4.5"):
        batches = list(iterate_minibatches(10, 4, np.random.default_rng(0)))
        verdict("MP4.5", [len(b) for b in batches] == [4, 4, 2]
                and np.array_equal(np.concatenate(batches), np.random.default_rng(0).permutation(10)),
                "iterate_minibatches : l'ordre rng.permutation(n), coupé en tranches de batch_size.",
                "iterate_minibatches(10, 4, rng) doit rendre rng.permutation(10) coupé en tranches de 4, 4 et 2.")
        X_moons, y_moons = data.load_moons(1000, noise=0.2, seed=SEED)
        X_m, y_m, X_mv, y_mv = X_moons[:800], y_moons[:800], X_moons[800:], y_moons[800:]
        rng = np.random.default_rng(SEED)
        probe = Sequential(Dense(2, 16, rng=rng), ReLU(), Dense(16, 2, rng=rng))
        history = fit(probe, SoftmaxCrossEntropy(), AdamW(probe.parameters(), lr=1e-2), X_m[:200], y_m[:200], X_mv, y_mv,
                      epochs=3, batch_size=50, base_lr=1e-2, warmup_steps=4, max_norm=0.5, rng=np.random.default_rng(1))
        verdict("MP4.5", isinstance(history, dict) and gap(history.get("lr"), [cosine_lr(t, 12, 1e-2, 4) for t in range(12)]) < 1e-12,
                "fit : le learning rate de chaque pas suit cosine_lr (warmup de 4 pas, 12 pas en tout).",
                "fit doit régler optimizer.lr = cosine_lr(t, total_steps, base_lr, warmup_steps, min_lr) AVANT chaque pas, "
                "t compté de 0 sur tout l'entraînement, et le noter dans history[\"lr\"].")
        norms = history.get("grad_norm", []) if isinstance(history, dict) else []
        verdict("MP4.5", len(norms) == 12 and all(isinstance(n, float) and n > 0 for n in norms),
                f"fit : 12 normes de gradient notées avant le clipping (la plus grande : {max(norms):.2f})." if norms else "",
                "fit doit noter dans history[\"grad_norm\"] la norme de chaque pas, rendue par clip_grad_norm (avant le clipping).")
        if torch is not None:                       # the same 12 steps with PyTorch: AdamW, LambdaLR, clip_grad_norm_
            rng = np.random.default_rng(SEED)
            probe_b = Sequential(Dense(2, 16, rng=rng), ReLU(), Dense(16, 2, rng=rng))
            twin_p = torch_twin(probe_b)
            opt_p = torch.optim.AdamW(twin_p.parameters(), lr=1e-2)
            sched_p = torch.optim.lr_scheduler.LambdaLR(opt_p, lambda t: cosine_lr(t, 12, 1.0, 4))
            X_p, y_p = torch.from_numpy(X_m[:200]), torch.from_numpy(y_m[:200])
            rng_p, clipped_p = np.random.default_rng(1), 0
            for _ in range(3):
                for idx in iterate_minibatches(200, 50, rng_p):
                    idx = torch.from_numpy(np.asarray(idx))
                    loss_t = nn.functional.cross_entropy(twin_p(X_p[idx]), y_p[idx])
                    opt_p.zero_grad()
                    loss_t.backward()
                    clipped_p += float(nn.utils.clip_grad_norm_(twin_p.parameters(), 0.5)) > 0.5
                    opt_p.step()
                    sched_p.step()
            worst_p = max(gap(layer.W, module.weight.detach().T) for layer, module in zip(probe.layers, twin_p)
                          if type(layer).__name__ == "Dense")
            verdict("MP4.5", worst_p < 1e-10,
                    f"fit : les 12 pas sont ceux de PyTorch (écart {worst_p:.1e}), dont {clipped_p} rognés par le clipping "
                    "de la norme globale.",
                    f"après les 12 pas, l'écart des poids à PyTorch (AdamW, le même planning, clip_grad_norm_ à 0,5) vaut "
                    f"{worst_p:.2e} : {clipped_p} pas sur 12 dépassent max_norm = 0,5. Coupe-t-on bien les gradients par "
                    "leur norme GLOBALE (clip_grad_norm de ta librairie), après backward et avant step, et dans l'ordre "
                    "rng.permutation des mini-batches ?")
        loss_p, _ = evaluate(probe, SoftmaxCrossEntropy(), X_mv, y_mv)
        verdict("MP4.5", probe.training and abs(loss_p - history["val_loss"][-1]) < 1e-12,
                "evaluate rend la loss de validation de la dernière epoch, et le modèle revient en mode entraînement.",
                "evaluate doit passer le modèle en évaluation, puis le remettre dans son mode d'avant ; la loss de la "
                "dernière epoch doit être celle de evaluate.")
        rng = np.random.default_rng(SEED)            # the training of the moons, with early stopping
        moons_model = Sequential(Dense(2, 32, rng=rng), ReLU(), Dense(32, 32, rng=rng), ReLU(), Dense(32, 2, rng=rng))
        moons_history = fit(moons_model, SoftmaxCrossEntropy(), AdamW(moons_model.parameters(), lr=1e-2, weight_decay=1e-4),
                            X_m, y_m, X_mv, y_mv, epochs=60, batch_size=32, base_lr=1e-2, warmup_steps=20, max_norm=1.0,
                            patience=8, rng=np.random.default_rng(2))
        best = moons_history["best_epoch"]
        norms_m = np.asarray(moons_history["grad_norm"], dtype=float)
        print(f"   lunes : la norme du gradient dépasse max_norm = 1 sur {int((norms_m > 1.0).sum())} pas sur {len(norms_m)} "
              f"(au plus {norms_m.max():.2f}) ; le clipping n'agit que sur ces pas-là.")
        val_loss_m, val_acc_m = evaluate(moons_model, SoftmaxCrossEntropy(), X_mv, y_mv)
        n_done = len(moons_history["val_loss"])
        verdict("MP4.5", n_done == 60 or n_done - best == 8,
                f"{n_done} epochs ; meilleure à l'epoch {best}" + (", arrêt 8 epochs plus tard." if n_done < 60 else "."),
                f"{n_done} epochs, meilleure à l'epoch {best} : avec une patience de 8, l'arrêt vient 8 epochs après la meilleure.")
        verdict("MP4.5", abs(val_loss_m - moons_history["val_loss"][best - 1]) < 1e-12,
                f"le modèle final a les poids de la meilleure epoch (loss de validation {val_loss_m:.4f}).",
                "le modèle final doit avoir les poids de la meilleure epoch : garde un state_dict() à chaque amélioration "
                "et recharge-le à la fin (best_epoch compte à partir de 1, comme history[\"val_loss\"][best_epoch - 1]).")
        verdict("MP4.5", val_acc_m >= 0.95, f"accuracy de validation sur les lunes : {val_acc_m:.1%}.",
                f"accuracy de validation sur les lunes : {val_acc_m:.1%}, attendu au moins 95 %.")
        grid = np.stack(np.meshgrid(np.linspace(-1.8, 2.8, 300), np.linspace(-1.4, 1.9, 240)), axis=-1).reshape(-1, 2)
        moons_model.eval()
        proba = np.exp(moons_model(grid) - moons_model(grid).max(axis=1, keepdims=True))
        proba = (proba / proba.sum(axis=1, keepdims=True))[:, 1].reshape(240, 300)
        fig, (ax_f, ax_c) = plt.subplots(1, 2, figsize=(12, 4.2))
        ax_f.contourf(np.linspace(-1.8, 2.8, 300), np.linspace(-1.4, 1.9, 240), proba, levels=20, cmap="RdBu_r", alpha=0.6)
        ax_f.contour(np.linspace(-1.8, 2.8, 300), np.linspace(-1.4, 1.9, 240), proba, levels=[0.5], colors="black")
        ax_f.scatter(X_mv[:, 0], X_mv[:, 1], c=y_mv, cmap="RdBu_r", edgecolor="white", s=20)
        ax_f.set(title=f"minifw 2-32-32-2 : frontière de décision (validation {val_acc_m:.1%})", xlabel="$x_1$", ylabel="$x_2$")
        epochs_m = np.arange(1, n_done + 1)
        ax_c.plot(epochs_m, moons_history["train_loss"], ls="--", label="entraînement")
        ax_c.plot(epochs_m, moons_history["val_loss"], label="validation")
        ax_c.axvline(best, color="0.6", ls=":", label="poids gardés")
        ax_c.set(title="loss (cross-entropy) par epoch", xlabel="epoch", ylabel="loss")
        ax_c.legend()
        fig.tight_layout()
        save(fig, "moons_frontiere.png")
        plt.show()
        moons_model.train()
    checks_done()'''

# ---------------------------------------------------------------------------
# MP4.6
# ---------------------------------------------------------------------------
MP46_MD = r'''## MP4.6 · Fashion-MNIST : un entraînement identique à PyTorch, minifw contre PyTorch, l'ablation dropout/batchnorm, le test ouvert une fois ⏱️ 105 min

**Objectif :** montrer que `minifw` apprend comme PyTorch, mesurer ce qu'il coûte, et choisir une configuration sur la validation avant d'ouvrir le test une seule fois.

a) Écris `build_mlp(dropout, batchnorm, rng)` ci-dessous : un MLP 784-256-128-10, chaque couche cachée faite d'une `Dense`, d'une `BatchNorm1d` si `batchnorm`, d'une `ReLU` et d'un `Dropout(dropout, rng)` si `dropout > 0` ; une dernière `Dense` vers les 10 logits ; tous les poids tirés de `rng`.
b) **Un entraînement identique à PyTorch** : sans dropout (seule source de hasard que les deux ne partagent pas), deux epochs de `minifw` et de son jumeau PyTorch, mêmes poids de départ, mêmes mini-batches, même planning, même clipping : les poids finaux doivent coïncider à $10^{-6}$ près.
c) **Le modèle principal**, avec batchnorm et dropout de 0,2 : `minifw` contre un modèle PyTorch de même architecture (en `float32`, avec son initialisation par défaut, son propre dropout et sans early stopping), mêmes réglages sinon ; leurs courbes, leurs temps par epoch, leurs nombres de paramètres.
d) **L'ablation** : les quatre combinaisons de dropout (0 ou 0,2) et de batchnorm, sur la validation.
e) **Le choix, puis le test** : écris `RULE`, la règle de choix, **avant** de lire le tableau de l'ablation, puis `FINAL`, la configuration qu'elle désigne, d'après la validation seulement. La cellule réentraîne cette configuration (si ce n'est pas le modèle principal) et n'ouvre le test que si toutes les vérifications des étapes MP4.2 à MP4.5 passent, et seulement **en mode complet** (`FAST_MODE = False`) : le test ne s'ouvre qu'une fois, pour la mesure finale. Elle enregistre alors les résultats dans `results.json` et la matrice de confusion du test dans `figures/`.'''

MP46_TODO = r'''def build_mlp(dropout, batchnorm, rng):
    """A minifw MLP 784-256-128-10: each hidden layer is Dense, BatchNorm1d (if batchnorm), ReLU, Dropout(dropout, rng)
    (if dropout > 0); then Dense(128, 10). All the weights are drawn from rng."""
    raise NotImplementedError("build_mlp")  # TODO MP4.6 a)'''

MP46_SOLUTION = r'''def build_mlp(dropout, batchnorm, rng):
    """A minifw MLP 784-256-128-10: each hidden layer is Dense, BatchNorm1d (if batchnorm), ReLU, Dropout(dropout, rng)
    (if dropout > 0); then Dense(128, 10). All the weights are drawn from rng."""
    from minifw import BatchNorm1d, Dense, Dropout, ReLU, Sequential
    layers = []
    for n_in, n_out in [(784, 256), (256, 128)]:
        layers.append(Dense(n_in, n_out, rng=rng))
        if batchnorm:
            layers.append(BatchNorm1d(n_out))
        layers.append(ReLU())
        if dropout > 0:
            layers.append(Dropout(dropout, rng=rng))
    layers.append(Dense(128, 10, rng=rng))
    return Sequential(*layers)'''

MP46_DATA = r'''X_tr, y_tr, X_va, y_va = data.load_fashion(N_FASHION)
SETTINGS = dict(batch_size=128, base_lr=2e-3, warmup_steps=100, max_norm=5.0)    # the same for every training
WEIGHT_DECAY, PATIENCE = 1e-4, 3
print(f"Fashion-MNIST : {len(y_tr)} images d'entraînement, {len(y_va)} de validation ; {EPOCHS} epochs au plus")'''

MP46_TWIN = r'''checks_of("MP4.6 b")
if minifw_ok() and torch is not None and ready("X_tr", "build_mlp"):
    from minifw import SoftmaxCrossEntropy, fit, iterate_minibatches
    from mylearn.optim import AdamW, cosine_lr
    with attempt("MP4.6"):
        model_b = build_mlp(0.0, True, np.random.default_rng(SEED))
        names_b = [type(layer).__name__ for layer in model_b.layers]
        verdict("MP4.6", names_b == ["Dense", "BatchNorm1d", "ReLU", "Dense", "BatchNorm1d", "ReLU", "Dense"]
                and model_b.n_parameters() == 235_146 + 2 * (256 + 128),
                "build_mlp(0, True, rng) : Dense, BatchNorm1d, ReLU (deux fois), puis Dense ; 235 914 paramètres.",
                f"build_mlp(0, True, rng) rend les couches {names_b} et {model_b.n_parameters()} paramètres : relis sa docstring.")
        twin_b = torch_twin(model_b)
        n_b, epochs_b = 10_000, 2
        steps_b = epochs_b * math.ceil(n_b / SETTINGS["batch_size"])
        opt_t = torch.optim.AdamW(twin_b.parameters(), lr=SETTINGS["base_lr"], weight_decay=WEIGHT_DECAY)
        sched_t = torch.optim.lr_scheduler.LambdaLR(opt_t, lambda t: cosine_lr(t, steps_b, 1.0, SETTINGS["warmup_steps"]))
        started = time.perf_counter()
        fit(model_b, SoftmaxCrossEntropy(), AdamW(model_b.parameters(), lr=SETTINGS["base_lr"], weight_decay=WEIGHT_DECAY),
            X_tr[:n_b], y_tr[:n_b], X_va, y_va, epochs=epochs_b, rng=np.random.default_rng(SEED + 1), **SETTINGS)
        time_minifw_b = time.perf_counter() - started
        X_t, y_t = torch.from_numpy(X_tr[:n_b]), torch.from_numpy(y_tr[:n_b])
        rng_t = np.random.default_rng(SEED + 1)
        started = time.perf_counter()
        for _ in range(epochs_b):
            twin_b.train()
            for idx in iterate_minibatches(n_b, SETTINGS["batch_size"], rng_t):
                idx = torch.from_numpy(idx)
                loss_t = nn.functional.cross_entropy(twin_b(X_t[idx]), y_t[idx])
                opt_t.zero_grad()
                loss_t.backward()
                nn.utils.clip_grad_norm_(twin_b.parameters(), SETTINGS["max_norm"])
                opt_t.step()
                sched_t.step()
        time_torch_b = time.perf_counter() - started
        worst_b = max(gap(layer.W, module.weight.detach().T) for layer, module in zip(model_b.layers, twin_b)
                      if type(layer).__name__ == "Dense")
        verdict("MP4.6", worst_b < 1e-6,
                f"après {steps_b} pas, les poids de minifw et de PyTorch coïncident (écart {worst_b:.1e}) : minifw "
                f"{time_minifw_b:.1f} s, PyTorch (float64) {time_torch_b:.1f} s.",
                f"après {steps_b} pas, l'écart des poids à PyTorch vaut {worst_b:.2e} : une couche, le planning, le "
                "clipping ou l'ordre des mini-batches diffère (les étapes MP4.2 à MP4.5 disent laquelle).")
    checks_done()'''

MP46_MAIN = r'''main_ok = False   # True once the minifw training of this cell has run to its end
if ready("X_tr", "build_mlp") and minifw is not None:
    from minifw import SoftmaxCrossEntropy, evaluate, fit
    from mylearn.optim import AdamW
    with attempt("MP4.6"):
        main_model = build_mlp(0.2, True, np.random.default_rng(SEED))
        started = time.perf_counter()
        main_history = fit(main_model, SoftmaxCrossEntropy(),
                           AdamW(main_model.parameters(), lr=SETTINGS["base_lr"], weight_decay=WEIGHT_DECAY),
                           X_tr, y_tr, X_va, y_va, epochs=EPOCHS, patience=PATIENCE, rng=np.random.default_rng(SEED + 1),
                           **SETTINGS)
        main_seconds = (time.perf_counter() - started) / len(main_history["val_loss"])
        print(f"minifw : {len(main_history['val_loss'])} epochs, {main_seconds:.1f} s par epoch, meilleure accuracy de "
              f"validation {max(main_history['val_acc']):.4f} ; epoch gardée par l'early stopping : "
              f"{main_history['best_epoch']} (accuracy {main_history['val_acc'][main_history['best_epoch'] - 1]:.4f})")
        main_ok = True
    if torch is not None and main_ok:
        with attempt("MP4.6"):                        # cosine_lr comes from your library: it may not be written yet
            torch.manual_seed(SEED)                   # the PyTorch model: same architecture, float32, its own dropout
            torch_model = nn.Sequential(nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(128, 10))
            steps = EPOCHS * math.ceil(len(X_tr) / SETTINGS["batch_size"])
            opt_t = torch.optim.AdamW(torch_model.parameters(), lr=SETTINGS["base_lr"], weight_decay=WEIGHT_DECAY)
            from mylearn.optim import cosine_lr
            sched_t = torch.optim.lr_scheduler.LambdaLR(opt_t,
                                                        lambda t: cosine_lr(t, steps, 1.0, SETTINGS["warmup_steps"]))
            X_t, y_t = torch.from_numpy(X_tr.astype(np.float32)), torch.from_numpy(y_tr)
            X_vt, y_vt = torch.from_numpy(X_va.astype(np.float32)), torch.from_numpy(y_va)
            torch_history = {"train_loss": [], "val_loss": [], "val_acc": []}
            generator = torch.Generator().manual_seed(SEED + 1)
            started = time.perf_counter()
            for epoch in range(EPOCHS):
                torch_model.train()
                total = 0.0
                for idx in torch.randperm(len(X_t), generator=generator).split(SETTINGS["batch_size"]):
                    loss_t = nn.functional.cross_entropy(torch_model(X_t[idx]), y_t[idx])
                    opt_t.zero_grad()
                    loss_t.backward()
                    nn.utils.clip_grad_norm_(torch_model.parameters(), SETTINGS["max_norm"])
                    opt_t.step()
                    sched_t.step()
                    total += loss_t.item() * len(idx)
                torch_model.eval()
                with torch.no_grad():
                    logits_v = torch_model(X_vt)
                    torch_history["val_loss"].append(nn.functional.cross_entropy(logits_v, y_vt).item())
                    torch_history["val_acc"].append((logits_v.argmax(1) == y_vt).float().mean().item())
                torch_history["train_loss"].append(total / len(X_t))
            torch_seconds = (time.perf_counter() - started) / EPOCHS
            print(f"PyTorch (float32) : {EPOCHS} epochs, {torch_seconds:.1f} s par epoch, meilleure accuracy de validation "
                  f"{max(torch_history['val_acc']):.4f}")
    if main_ok:
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
        for history, label in [(main_history, "minifw"), (globals().get("torch_history"), "PyTorch")]:
            if history:
                epochs_h = np.arange(1, len(history["val_loss"]) + 1)
                line, = ax_l.plot(epochs_h, history["val_loss"], label=f"{label}, validation")
                ax_l.plot(epochs_h, history["train_loss"], ls="--", color=line.get_color(), label=f"{label}, entraînement")
                ax_a.plot(epochs_h, history["val_acc"], color=line.get_color(), label=label)
        ax_l.set(title="loss (cross-entropy)", xlabel="epoch")
        ax_a.set(title="accuracy de validation", xlabel="epoch")
        ax_l.legend()
        ax_a.legend()
        fig.tight_layout()
        save(fig, "fashion_courbes_rapide.png" if FAST_MODE else "fashion_courbes.png")   # the README shows the full run
        plt.show()'''

MP46_ABLATION = r'''CONFIGS = {"rien": (0.0, False), "dropout": (0.2, False), "batchnorm": (0.0, True), "les deux": (0.2, True)}
ABLATION = {}
if ready("X_tr", "build_mlp") and minifw is not None:
    from minifw import SoftmaxCrossEntropy, fit
    from mylearn.optim import AdamW
    started = time.perf_counter()
    with attempt("MP4.6"):
        for name, (dropout, batchnorm) in CONFIGS.items():
            model_c = build_mlp(dropout, batchnorm, np.random.default_rng(SEED))
            history_c = fit(model_c, SoftmaxCrossEntropy(),
                            AdamW(model_c.parameters(), lr=SETTINGS["base_lr"], weight_decay=WEIGHT_DECAY),
                            X_tr[:N_ABLATION], y_tr[:N_ABLATION], X_va, y_va, epochs=EPOCHS_ABLATION,
                            rng=np.random.default_rng(SEED + 1), **SETTINGS)
            ABLATION[name] = history_c
        print(f"{len(CONFIGS)} entraînements de {EPOCHS_ABLATION} epochs sur {N_ABLATION} images en "
              f"{time.perf_counter() - started:.0f} s\n")
        print("configuration    best val acc   best epoch   min val loss   final train loss")
        for name, history_c in ABLATION.items():
            print(f"{name:14} {max(history_c['val_acc']):14.4f} {int(np.argmax(history_c['val_acc'])) + 1:12d} "
                  f"{min(history_c['val_loss']):14.4f} {history_c['train_loss'][-1]:18.4f}")'''

MP46_TODO_FINAL = r'''FINAL = ...   # MP4.6 e) the configuration you keep, a name of CONFIGS, chosen on the validation only
RULE = ...    # MP4.6 e) the rule that chose it, in one sentence (a string)'''

MP46_SOLUTION_FINAL = r'''RULE = "la meilleure accuracy de validation de l'ablation (la règle est écrite avant de lire le tableau)"
FINAL = max(ABLATION, key=lambda name: max(ABLATION[name]["val_acc"])) if ABLATION else ...   # MP4.6 e)
print("FINAL =", FINAL)'''

MP46_TEST = r'''checks_of("MP4.6 e")
if ready("X_tr", "main_model", "ABLATION", "build_mlp") and filled(FINAL, RULE):
    verdict("MP4.6", FINAL in CONFIGS, f"configuration retenue : {FINAL} ({RULE}).",
            f"FINAL doit être l'un des noms de CONFIGS ({', '.join(CONFIGS)}), pas {FINAL!r}.")
    if FINAL in CONFIGS and ABLATION:
        best_name = max(ABLATION, key=lambda name: max(ABLATION[name]["val_acc"]))
        if best_name != FINAL:
            print(f"ℹ️ MP4.6 : la meilleure accuracy de validation de l'ablation est celle de « {best_name} » ; "
                  f"explique dans ton README pourquoi tu retiens « {FINAL} ».")
    blocking = {cell: [f for f in fails] for cell, fails in FAILED.items() if fails and cell != "MP4.6 e"}
    verdict("MP4.6", not blocking, "toutes les vérifications des étapes précédentes passent : le test peut s'ouvrir.",
            "le test reste fermé tant qu'une vérification échoue : " + " · ".join(f"{cell} : {fails[0].rstrip('.')}" for cell, fails in blocking.items())[:900] + ".")
    if FAST_MODE:
        print("⏸️ MP4.6 : en mode rapide, le test reste fermé : il ne s'ouvre qu'une fois, pour la mesure finale, en mode "
              "complet (FAST_MODE = False dans la cellule de setup), quand tes choix sont faits sur la validation.")
    elif FINAL in CONFIGS and not blocking:
        from minifw import SoftmaxCrossEntropy, evaluate, fit
        from mylearn.metrics import confusion_matrix
        from mylearn.optim import AdamW
        if CONFIGS[FINAL] == (0.2, True):
            final_model, final_history = main_model, main_history
        else:
            final_model = build_mlp(*CONFIGS[FINAL], np.random.default_rng(SEED))
            final_history = fit(final_model, SoftmaxCrossEntropy(),
                                AdamW(final_model.parameters(), lr=SETTINGS["base_lr"], weight_decay=WEIGHT_DECAY),
                                X_tr, y_tr, X_va, y_va, epochs=EPOCHS, patience=PATIENCE,
                                rng=np.random.default_rng(SEED + 1), **SETTINGS)
        description = (f"minifw 784-256-128-10, {FINAL}, {len(y_tr)} images, {EPOCHS} epochs au plus, AdamW "
                       f"{SETTINGS['base_lr']}, warmup {SETTINGS['warmup_steps']} pas + cosinus, clipping {SETTINGS['max_norm']}")
        X_te, y_te = data.open_test(PROJECT, description)
        test_loss, test_acc = evaluate(final_model, SoftmaxCrossEntropy(), X_te, y_te)
        final_model.eval()
        predictions = np.concatenate([final_model(X_te[i:i + 1000]).argmax(axis=1) for i in range(0, len(X_te), 1000)])
        final_model.train()
        matrix = np.asarray(confusion_matrix(y_te, predictions, labels=list(range(10))))
        print(f"test (ouvert une fois) : accuracy {test_acc:.4f}, loss {test_loss:.4f} · objectif du cahier des charges : "
              f"au moins 88 % en mode complet")
        print("ouvertures du test :", len(data.test_openings(PROJECT)))
        classes = wb.datasets.FASHION_CLASSES
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.imshow(matrix, cmap="Blues")
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(10), yticks=range(10), xlabel="classe prédite", ylabel="vraie classe",
               title=f"minifw sur le test de Fashion-MNIST (accuracy {test_acc:.1%})")
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        fig.tight_layout()
        save(fig, "fashion_confusion.png")
        plt.show()
        results = {
            "fast_mode": FAST_MODE, "n_train": int(len(y_tr)), "epochs_max": EPOCHS, "settings": SETTINGS,
            "weight_decay": WEIGHT_DECAY, "patience": PATIENCE, "seed": SEED, "final": FINAL, "rule": RULE,
            "n_parameters": final_model.n_parameters(), "best_epoch": final_history["best_epoch"],
            "val_accuracy": final_history["val_acc"][final_history["best_epoch"] - 1],
            "test_accuracy": test_acc, "test_loss": test_loss,
            "main_config": "les deux", "minifw_best_val_accuracy": max(main_history["val_acc"]),
            "minifw_seconds_per_epoch": round(main_seconds, 2),
            "torch_seconds_per_epoch": round(torch_seconds, 2) if "torch_seconds" in globals() else None,
            "torch_best_val_accuracy": max(torch_history["val_acc"]) if "torch_history" in globals() else None,
            "twin_gap": worst_b if "worst_b" in globals() else None,
            "ablation": {name: {"best_val_accuracy": max(h["val_acc"]), "min_val_loss": min(h["val_loss"])}
                         for name, h in ABLATION.items()},
            "versions": minifw.library_versions(),
        }
        (PROJECT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("💾 results.json écrit.")
    checks_done()'''

# ---------------------------------------------------------------------------
# MP4.7
# ---------------------------------------------------------------------------
MP47_MD = r'''## MP4.7 · README, figures, limites ; le notebook relancé de zéro ⏱️ 45 min

**Objectif :** un dépôt qu'un recruteur comprend en deux minutes : ce que fait le code, comment il est prouvé, ce qu'il donne, et ce qu'il ne sait pas faire.

Termine ton `README.md` de portfolio (le modèle est dans ton dossier) : l'objectif, l'architecture du code (un schéma des modules et du contrat `forward`/`backward`), les résultats chiffrés (le tableau de `results.json` : minifw contre PyTorch, l'ablation, le test), les trois figures légendées, les limites (honnêtes : vitesse, pas de GPU ni d'autograd, `float64`, une seule graine…) et ce que tu as appris. Relance ensuite tout le notebook, de zéro (*Kernel → Restart & Run all*), puis commite le projet.'''

MP47_CHECK = r'''checks_of("MP4.7")
readme = check_document("MP4.7", PROJECT / "README.md",
                        {"L'objectif": ["objectif", "goal"], "L'architecture du code": ["architecture"],
                         "Les résultats": ["résultat", "result"], "Les limites": ["limite", "limitation"],
                         "Ce que j'ai appris": ["appris", "learn"]})
for figure, made_by in (("moons_frontiere.png", "l'étape MP4.5 la crée"),
                        ("fashion_courbes.png", "l'étape MP4.6 la crée, en mode complet"),
                        ("fashion_confusion.png", "l'étape MP4.6 la crée en mode complet, quand le test s'ouvre")):
    if not (FIGURES / figure).exists():
        pending("MP4.7", f"figures/{figure} n'existe pas encore : {made_by}.")
    elif readme:
        verdict("MP4.7", f"figures/{figure}" in readme, f"le README montre figures/{figure}.",
                f"le README doit montrer figures/{figure}.")
results_path = PROJECT / "results.json"
if results_path.exists():
    verdict("MP4.7", True, "results.json existe.", "")
else:
    pending("MP4.7", "results.json n'existe pas encore : il est écrit par l'étape MP4.6 en mode complet "
            "(FAST_MODE = False), quand le test s'ouvre.")
openings = data.test_openings(PROJECT)
if not openings:
    pending("MP4.7", "le test n'est pas encore ouvert : il s'ouvre une fois, à l'étape MP4.6, en mode complet.")
else:
    verdict("MP4.7", len({entry["description"] for entry in openings}) == 1,
            f"le test n'a été ouvert que pour un seul modèle ({len(openings)} entrée(s) dans test_log.json).",
            f"le test a été ouvert pour {len({e['description'] for e in openings})} modèles différents (test_log.json) : "
            "son score n'est plus une mesure honnête, dis-le dans ton README.")
if results_path.exists():
    results = json.loads(results_path.read_text(encoding="utf-8"))
    verdict("MP4.7", results["test_accuracy"] >= 0.88, f"accuracy de test {results['test_accuracy']:.4f} : au moins 88 %.",
            f"accuracy de test {results['test_accuracy']:.4f} : le cahier des charges demande au moins 88 %.")
passed, failed, reasons = run_project_tests(quiet=True)
waiting = [reason for reason in reasons if reason.startswith(PENDING)]
if waiting:
    pending("MP4.7", f"{len(waiting)} test(s) attendent une fonction de minifw pas encore écrite.")
elif passed + failed == 0:
    verdict("MP4.7", False, "", "aucun test n'a tourné : la cellule de vérification de l'étape MP4.4 affiche "
            "pourquoi (conftest.py, import de minifw ou de mylearn).")
else:
    verdict("MP4.7", failed == 0 and passed >= 10, f"les {passed} tests passent.",
            f"{passed} test(s) passe(nt) et {failed} échoue(nt) : il en faut au moins 10, tous verts.")
checks_done()'''

NOTES = {   # step -> (questions for the learner, observations of the reference solution)
    "MP4.1": ("""- Ce que fait minifw, et ce qu'il ne fait pas :
- Comment tu prouveras qu'il est juste :""", r"""OBS_MP41"""),
    "MP4.2": ("""- Ce que garde chaque couche pendant la passe avant, et pourquoi :
- Pourquoi `parameters()` rend les tableaux eux-mêmes, et pas des copies :
- Ce que change `training` pour Dropout et BatchNorm1d :""", r"""OBS_MP42"""),
    "MP4.3": ("""- Pourquoi `backward` parcourt les couches à l'envers :
- Ce que vérifie `gradient_check` que la comparaison à PyTorch ne vérifie pas (et l'inverse) :
- Pourquoi `load_state_dict` doit recopier sur place :""", r"""OBS_MP43"""),
    "MP4.4": ("""- Les mutants que tes tests attrapent, et celui qui leur échapperait le plus facilement :
- Pourquoi comparer les gradients, et pas seulement les sorties :""", r"""OBS_MP44"""),
    "MP4.5": ("""- Le rôle du warmup, du cosinus et du clipping ici :
- L'epoch gardée par l'early stopping, et pourquoi on recharge ses poids :""", r"""OBS_MP45"""),
    "MP4.6": ("""- Ce que prouve l'entraînement identique à PyTorch, et pourquoi il faut retirer le dropout pour le faire :
- minifw contre PyTorch : l'accuracy, le temps par epoch, et d'où vient l'écart de temps :
- Ce que montre l'ablation, et la règle de ton choix :
- Le score du test face à la validation :""", r"""OBS_MP46"""),
}


def notes_cell(step: str, exercise: bool):
    questions, observations = NOTES[step]
    if exercise:
        return md(f"✍️ **Tes observations ({step})** : double-clique sur cette cellule pour écrire tes réponses ; "
                  f"elles nourriront ton README.\n\n{questions}")
    return md(f"✍️ **Observations ({step})**\n\n{OBSERVATIONS.get(step, observations)}")


OBSERVATIONS: dict[str, str] = {   # the reference solution's answers to the questions of NOTES (fast and full mode)
    "MP4.1": r"""- **Ce que fait minifw** : les briques d'un MLP (couches denses, ReLU et tanh, dropout, batchnorm), deux losses calculées sur les logits, un conteneur `Sequential` et une boucle d'entraînement complète (mini-batches, AdamW, warmup + cosinus, clipping, early stopping). **Ce qu'il ne fait pas** : ni GPU, ni autograd (chaque couche a sa passe arrière écrite à la main), ni convolution, ni `float32`. Il n'a rien d'original : sa valeur est d'être prouvé juste, ligne à ligne, contre PyTorch.
- **La preuve**, en trois niveaux : chaque couche comparée à PyTorch en `float64` (sorties et gradients, en entraînement et en évaluation) ; le réseau entier, comparé à PyTorch et vérifié par différences finies ; un entraînement de 158 pas identique à celui de PyTorch (MP4.6 b). Et des tests qui échouent quand on casse le code : les six mutants de MP4.4.""",
    "MP4.2": r"""- **Ce que garde chaque couche** : `Dense` garde son entrée $\mathbf{x}$, car $\partial L/\partial \mathbf{W} = \mathbf{x}^\top \boldsymbol{\delta}$ ; `ReLU` et `Tanh` gardent leur entrée (ou leur sortie : $\tanh' = 1 - y^2$) pour la dérivée ; `Dropout` garde son masque, sans quoi la passe arrière ne saurait plus quels neurones ont été éteints ; `BatchNorm1d` garde le cache de `batchnorm_forward` ($\hat{\mathbf{x}}$, l'inverse de l'écart-type du mini-batch). Rien n'est recalculé : la passe arrière ne fait que lire.
- **Des tableaux, pas des copies** : `AdamW(model.parameters())` garde des références et modifie les tableaux sur place (`param -= ...`). Avec des copies, l'optimiseur mettrait à jour des tableaux que plus personne ne lit, et le modèle n'apprendrait jamais, sans aucune erreur.
- **`training`** : le dropout inversé divise déjà par $1 - p$ en entraînement, si bien qu'en évaluation il ne reste rien à faire (l'identité) ; la batchnorm normalise par les statistiques du mini-batch en entraînement, et met à jour ses statistiques glissantes (avec la variance **non biaisée**, comme PyTorch), puis s'en sert en évaluation. Toutes les comparaisons à PyTorch tombent entre 0 et $1{,}3 \times 10^{-15}$ : l'écart de l'arrondi, pas d'une formule.""",
    "MP4.3": r"""- **À l'envers** : la règle de la chaîne. Le gradient d'une couche demande $\partial L / \partial(\text{sa sortie})$, qui vient de la couche suivante : on part de la loss et on remonte. Chaque `backward` rend le $\partial L / \partial x$ que la couche précédente attend.
- **`gradient_check` contre PyTorch** : les différences finies (erreur relative $4{,}5 \times 10^{-9}$ ici) vérifient que la passe arrière est la dérivée de **ta** passe avant, sans oracle extérieur ; elles ne voient pas une passe avant fausse mais dérivée de façon cohérente (des poids rangés à l'envers par rapport à `nn.Linear`), ni une erreur qui ne touche que l'évaluation (les statistiques glissantes). PyTorch vérifie les **conventions** (la variance, la loss moyenne, la forme de $\mathbf{W}$), mais seulement là où il a un jumeau. Les deux ensemble : le réseau de 171 paramètres a ses 8 gradients à $1{,}1 \times 10^{-16}$ de PyTorch.
- **Sur place** : l'optimiseur garde les références des tableaux et ses moments $m$ et $v$, rangés dans le même ordre. Si `load_state_dict` remplaçait les tableaux, le modèle et l'optimiseur ne parleraient plus des mêmes poids : les pas suivants modifieraient les anciens tableaux, et le modèle chargé ne bougerait plus. C'est pour cette raison que PyTorch fait `param.copy_(...)`.""",
    "MP4.4": r"""- **Les mutants** : les 23 tests de la solution les attrapent tous les six, mais pas avec la même marge : quatre tests échouent pour la batchnorm décalée de 0,001, cinq pour le `Sequential` qui oublie une couche, **un seul** pour le dropout qui multiplie par $1 - p$ en évaluation. C'est le mutant qui échappe le plus facilement : les tests d'évaluation sont rares, et un réseau sans dropout ne le voit pas. Un facteur $1{,}000001$ échapperait à tous : la tolérance ($10^{-6}$) fixe ce que les tests peuvent voir.
- **Les gradients, pas seulement les sorties** : sans autograd, la passe avant et la passe arrière sont deux codes séparés. Le mutant qui fausse `dW` de 1 % passe tous les tests de sorties, et seul un test de gradient le voit. Un modèle à la passe arrière fausse apprend souvent quand même, moins bien : c'est exactement l'erreur qu'un entraînement ne révèle pas.""",
    "MP4.5": r"""- **Warmup, cosinus, clipping** : au début, les poids sont tirés au hasard et les moments d'Adam reposent sur peu de pas : le warmup garde des pas petits le temps qu'ils se stabilisent ; le cosinus fait décroître le learning rate jusqu'à `min_lr`, pour finir l'entraînement à petits pas près d'un minimum ; le clipping borne la norme globale du gradient, une assurance contre un mini-batch extrême. Dans la vérification de `fit` (max_norm = 0,5), la norme va jusqu'à 0,64, plusieurs des 12 pas sont rognés, et les poids restent ceux de la même boucle écrite en PyTorch (`clip_grad_norm_`) : c'est ce qui prouve que le clipping porte sur la norme **globale** ; dans l'entraînement des lunes (max_norm = 1), elle ne dépasse 1 que sur 24 pas sur 600 (au plus 4,7) : le clipping n'agit que sur quelques mini-batches extrêmes, ce qu'on attend de lui.
- **L'early stopping** (patience 8) : 24 epochs, la meilleure à l'epoch 16 (loss de validation 0,0380, accuracy 99,5 %), puis huit epochs sans progrès. On recharge les poids de l'epoch 16 parce que le modèle qu'on rend doit être celui qu'on a mesuré : les poids de l'epoch 24 sont ceux d'un modèle qui a continué à s'ajuster à l'entraînement, et dont la loss de validation est moins bonne.""",
    "MP4.6": r"""- **L'entraînement identique** : après 158 pas (deux epochs sur 10 000 images), les poids de minifw et de PyTorch diffèrent de $1{,}7 \times 10^{-16}$, l'écart de l'arrondi. Cela prouve plus que les tests d'une couche : l'ordre des mini-batches, la loss moyenne, AdamW et son weight decay découplé et le planning (pas par pas) sont les mêmes que ceux de PyTorch. Le clipping, lui, n'agit presque pas ici (max_norm = 5, une norme qui reste en dessous) : c'est la vérification de MP4.5 qui le compare à PyTorch, sur des pas rognés. Le dropout est retiré, car minifw tire ses masques avec NumPy et PyTorch avec son propre générateur : les deux réseaux n'éteindraient pas les mêmes neurones, et l'écart serait dû au hasard, pas à une erreur.
- **minifw contre PyTorch** (784-256-128-10, batchnorm et dropout 0,2, 235 914 paramètres) : en mode complet, 0,8994 d'accuracy de validation contre 0,8997, en 3,2 s par epoch contre 1,4 s (en mode rapide, 0,8804 contre 0,8808). Même architecture et mêmes réglages, mais PyTorch garde son initialisation par défaut (uniforme, et non celle de He), son propre ordre des mini-batches, et calcule en `float32` : à ces différences près, les accuracies sont les mêmes. Le temps : un facteur 2,3. À précision égale, dans l'entraînement jumeau du mode complet, il reste un facteur proche de 2 (1,3 s contre 0,7 s) : l'essentiel de l'écart vient des appels Python et des caches copiés à chaque couche, le reste du `float64` (deux fois plus d'octets à lire pour chaque produit matriciel). Les temps varient d'une exécution à l'autre, surtout quand le processeur est partagé.
- **L'ablation** (mode complet, 12 epochs sur 50 000 images ; accuracy de validation) : rien 0,8963, dropout 0,8931, batchnorm 0,8992, les deux 0,8988 (en mode rapide : 0,8599, 0,8510, 0,8665, 0,8602). La batchnorm aide dans les deux modes ; le dropout seul coûte un peu d'accuracy, car en 12 epochs ce réseau surapprend peu (la loss d'entraînement finale reste de l'ordre de 0,17 à 0,24, sauf avec la batchnorm seule), mais « les deux » a la meilleure loss de validation (0,2765). Les écarts entre batchnorm et les deux sont plus petits que le bruit d'une mesure sur 10 000 images (l'écart-type d'une accuracy de 0,9 y vaut $\sqrt{0{,}9 \times 0{,}1 / 10\,000} \approx 0{,}3$ point) : la règle, écrite avant, tranche pour la batchnorm.
- **Le test** (mode complet, ouvert une fois) : 0,8932, au-dessus des 88 % du cahier des charges, et 0,4 point sous la validation du modèle final (0,8976 à l'epoch 9). Un écart attendu, d'environ un écart-type et demi : la configuration et l'epoch ont été choisies sur la validation, qui est donc un peu optimiste, et le reste est du bruit de mesure. Le modèle final réentraîné ne retrouve pas tout à fait le score de l'ablation (0,8992) : son cosinus s'étale sur 15 epochs et non 12, la patience de 3 l'arrête à l'epoch 12, et 0,8976 est l'accuracy de l'epoch que garde l'early stopping (la meilleure loss de validation), pas la meilleure accuracy. Les erreurs du test sont celles qu'on attend d'un MLP sur des images : chemises, T-shirts, pulls et manteaux se confondent (la classe « Shirt » n'atteint que 65 %).""",
}

FOOTER = r'''## ✅ Bilan

**Grille d'évaluation** (sur 20, détaillée dans le cahier des charges) : justesse du code, des tests d'équivalence avec PyTorch verts pour toutes les couches et l'optimiseur (5) · qualité du code : structure, docstrings, annotations, pas de duplication, tests lisibles (4) · rigueur expérimentale : graines, validation séparée, courbes, comparaison à configuration égale (4) · résultats : au moins 88 % d'accuracy de test sur Fashion-MNIST avec minifw, écart à PyTorch expliqué (3) · communication : README clair, figures légendées, limites honnêtes (4).

**Pour aller plus loin** : les extensions du cahier des charges (une couche `Conv2d` par im2col, aperçu du ch. 21 ; un autograd vectoriel qui généralise la classe `Value` du ch. 18 ; des connexions résiduelles et la layer norm pour un MLP de 20 couches ; la sauvegarde au format `.npz` et la reprise ; le profilage). La suite du workbook : la partie V, où viennent les réseaux convolutifs (ch. 21), récurrents (ch. 22) et PyTorch en pratique (ch. 23 et 24).'''


def header_cells(kind: str) -> list:
    rel = STARTER if kind == "exercise" else SOLUTION
    rows = ["| Étape | Contenu | ⏱️ |", "|---|---|---|"] + [f"| {sid} | {title} | {minutes} min |"
                                                            for sid, title, minutes in STEPS]
    fast = ("**Mode rapide.** Avec `FAST_MODE = True` (la cellule de setup), le modèle principal s'entraîne sur 20 000 "
            "images pendant 6 epochs au plus, et l'ablation sur 10 000 images pendant 5 epochs ; en mode complet, sur "
            "50 000 images, pendant 15 et 12 epochs. Le test de Fashion-MNIST ne s'ouvre qu'en mode complet, une fois, "
            "pour la mesure finale. Le notebook tourne en une à deux minutes sur un CPU à deux cœurs en mode rapide, "
            "cinq minutes environ en mode complet. Ferme les autres notebooks pendant qu'il "
            "tourne : les produits matriciels ralentissent beaucoup quand les cœurs sont partagés.")
    if kind == "exercise":
        title = "# Mini-projet MP4 · Un mini-framework de deep learning en NumPy, validé contre PyTorch"
        how = ("Le cahier des charges est dans `projets/partie_4_mini_framework/README.md` : lis-le d'abord. Ce notebook "
               "te guide en sept étapes. Le code durable va dans **ton package** `minifw/` (et ses tests, "
               "`test_minifw.py`) ; le notebook le vérifie contre PyTorch, l'entraîne, compare et trace les figures. "
               "Les fonctions à écrire lèvent `NotImplementedError` ; les vérifications affichent ✅ ou ❌ (des "
               "garde-fous, pas une note), et ⏳ tant qu'une étape n'est pas faite. `minifw` s'appuie sur **ta** "
               "librairie `mylearn` (ch. 16 à 20) : la liste des fonctions est dans le cahier des charges, section "
               "« Ta librairie ».\n\n"
               "> Travaille dans **ta copie** (`mon_travail/projets/partie_4_mini_framework/mp4_mini_framework.ipynb`, "
               "créée par `python tools/start_chapter.py CP4`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le "
               "badge ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme "
               "l'explique `00_setup/COLAB.md` §2.")
    else:
        title = ("# Mini-projet MP4 · Un mini-framework de deep learning en NumPy, validé contre PyTorch — solution de "
                 "référence (exécutée)")
        how = ("La solution de référence, exécutée avec `minifw/` de ce dossier et la librairie de référence. Le test "
               "de Fashion-MNIST n'a été ouvert que pour un seul modèle (`test_log.json`). Tes chiffres n'ont pas à être "
               "identiques : d'autres choix donnent d'autres valeurs, tout aussi justes s'ils sont bien mesurés. Le "
               "README de portfolio rédigé à partir de ces résultats est dans ce dossier.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n{fast}\n\n" + "\n".join(rows))]


def build(kind: str) -> list:
    exercise = kind == "exercise"
    cells = header_cells(kind)
    cells.append(setup_cell("exercise", chapter="CP4") if exercise else setup_cell("solution"))
    cells.append(code(PROJECT_STARTER if exercise else PROJECT_SOLUTION))
    cells.append(code(TOOLS))
    cells += [md(MP41_MD), code(MP41_CHECK), notes_cell("MP4.1", exercise)]
    cells += [md(MP42_MD), code(MP42_CHECK), notes_cell("MP4.2", exercise)]
    cells += [md(MP43_MD), code(MP43_CHECK), notes_cell("MP4.3", exercise)]
    cells += [md(MP44_MD), code(MP44_CHECK), notes_cell("MP4.4", exercise)]
    cells += [md(MP45_MD), code(MP45_CHECK), notes_cell("MP4.5", exercise)]
    cells += [md(MP46_MD), code(MP46_TODO if exercise else MP46_SOLUTION), code(MP46_DATA), code(MP46_TWIN),
              code(MP46_MAIN), code(MP46_ABLATION), code(MP46_TODO_FINAL if exercise else MP46_SOLUTION_FINAL),
              code(MP46_TEST), notes_cell("MP4.6", exercise)]
    cells += [md(MP47_MD), code(MP47_CHECK), md(FOOTER)]
    return cells


def main() -> int:
    written = write_starter_package()
    write_notebook(STARTER, build("exercise"))
    write_notebook(SOLUTION, build("solution"))
    print(f"✅ depart/minifw/ ({', '.join(written)}), {STARTER} and {SOLUTION} written ({len(STEPS)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
