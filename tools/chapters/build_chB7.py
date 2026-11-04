#!/usr/bin/env python
"""Build the two notebooks of bonus chapter B7 from a single source (used by Claude).

    python tools/chapters/build_chB7.py
    python tools/run_all_notebooks.py chapitres/b7_production/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/b7_production/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q11, of the recalls R1 to R3 and of the ✏️ paper exercises
B7.1 to B7.4 (02_exercices.md). The estimate 🧮 B7.5, the reflection (🗣️ B7.6, ⚖️ B7.7, 📄 B7.8) and the interview
are corrected with 05_solutions.md. The answers are computed by hand formulas in the context cell of the solutions
notebook, then checked against the libraries: NumPy (np.percentile, np.std), SciPy (stats.t, ttest_1samp and
ttest_rel confidence intervals, stats.entropy), torchvision (ConvNeXt-Tiny built on the `meta` device, without
download), safetensors (the size of the MP5 VGG file and its header), PyYAML, logging, argparse, subprocess,
torch.utils.data.DataLoader, torch.export (a static batch and a data-dependent `if`), torch.load(weights_only=True)
and scikit-learn (InconsistentVersionWarning): part 0 downloads nothing. B7 has no mylearn module.
The notebook itself (exercises B7.9 to B7.25), in three phases: the project (part A, B7.9 to B7.15), the service
(part B, B7.16 to B7.21), the operations (part C, B7.22 to B7.25); chB7_parts*.py and chB7_code*.py hold the cells,
history_chB7.py what is precomputed (history_chB7.json).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (Paper, badge, check_unique_keys, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "B7"
FOLDER = "chapitres/b7_production"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = r'''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np

MB, MIB = 1e6, 2 ** 20                  # 1 Mo = 10**6 bytes, 1 Mio = 2**20 bytes (fiche, 207.8)

# quiz Q1: the cells x = 2 ; x = x * 10 ; print(x + 1), run in the order 1, 2, 3, 2, 3
Q1A = 2 * 10 * 10 + 1
# quiz Q3: basicConfig(level=WARNING), then 1 debug, 3 info, 1 warning and 1 error messages
LEVELS_Q3 = ["DEBUG"] + ["INFO"] * 3 + ["WARNING", "ERROR"]
Q3C = sum(level in ("WARNING", "ERROR", "CRITICAL") for level in LEVELS_Q3)
# quiz Q7: the ML Test Score is the minimum of the four section sums
SECTIONS_Q7 = {"data": 3.0, "model": 2.5, "infrastructure": 4.0, "monitoring": 1.0}
Q7E = min(SECTIONS_Q7.values())
# quiz Q9: 900 requests of 10 ms and 100 of 200 ms ; Little's law ; a throughput
LAT_Q9 = np.array([10.0] * 900 + [200.0] * 100)
Q9A, Q9B = float(LAT_Q9.mean()), float(np.percentile(LAT_Q9, 95))
Q9D = 60 * 0.150                         # L = lambda W, W in seconds
Q9E = 64 / 0.4
# recall R1: a JSONL log of 7 runs, 2 runs added (append, never rewrite)
R1D = 7 + 2

# ✏️ B7.1: ConvNeXt-Tiny (torchvision) and the VGG of the MP5 solution
N_B71, LIN_W_B71, LIN_B_B71 = 28_589_128, 26_646_528, 34_120
F32_B71 = 4 * N_B71 / MB
F32_MIB_B71 = 4 * N_B71 / MIB
BF16_B71 = 2 * N_B71 / MB
INT8_B71 = N_B71 / MB
DYNQ_B71 = (LIN_W_B71 + 4 * (N_B71 - LIN_W_B71)) / MB          # dense matrices in int8, everything else float32
RATIO_B71 = F32_B71 / DYNQ_B71
PEAK_B71 = 2 * F32_MIB_B71                                     # random weights + the copy read by torch.load
VGG_PARAMS_B71, VGG_LINEAR_B71, VGG_FILE_B71 = 72_954, 650, 296_464
VGG_BN_CHANNELS_B71 = (16, 16, 32, 32, 64, 64)
VGG_FLOAT_BUFFERS_B71 = 2 * sum(VGG_BN_CHANNELS_B71)            # running_mean and running_var
VGG_BYTES_B71 = 4 * (VGG_PARAMS_B71 + VGG_FLOAT_BUFFERS_B71) + 8 * len(VGG_BN_CHANNELS_B71)
HEADER_B71 = VGG_FILE_B71 - VGG_BYTES_B71
LIN_SHARE_B71 = 100 * VGG_LINEAR_B71 / VGG_PARAMS_B71

# ✏️ B7.2: five seeds, Student t with n - 1 = 4 degrees of freedom (the table of 207.4)
A_B72 = np.array([64.2, 65.0, 63.6, 64.8, 64.4])
B_B72 = np.array([64.9, 65.3, 64.5, 65.6, 64.7])
T4_B72 = 2.776


def interval_b72(values, t=T4_B72, ddof=1):
    """Mean, standard deviation, standard error and half-width of the Student interval."""
    m, s = float(np.mean(values)), float(np.std(values, ddof=ddof))
    se = s / math.sqrt(len(values))
    return m, s, se, t * se


MEAN_A_B72, S_A_B72, SE_A_B72, HW_A_B72 = interval_b72(A_B72)
LOW_A_B72 = MEAN_A_B72 - HW_A_B72
MEAN_B_B72, _, _, HW_B_B72 = interval_b72(B_B72)
OVERLAP_B72 = bool(MEAN_A_B72 + HW_A_B72 >= MEAN_B_B72 - HW_B_B72)
MEAN_D_B72, S_D_B72, SE_D_B72, HW_D_B72 = interval_b72(B_B72 - A_B72)
ZERO_IN_D_B72 = bool(MEAN_D_B72 - HW_D_B72 <= 0 <= MEAN_D_B72 + HW_D_B72)

# ✏️ B7.3: twenty latencies of batch 1 (ms), np.percentile's default (method="linear")
LAT_B73 = np.array([21.3, 19.8, 20.4, 22.1, 19.6, 20.9, 20.1, 35.7, 20.6, 19.9,
                    21.0, 20.2, 23.4, 19.7, 20.5, 47.9, 20.8, 20.3, 21.6, 20.0])
MEAN_B73 = float(LAT_B73.mean())
P50_B73, P95_B73 = (float(v) for v in np.percentile(LAT_B73, [50, 95]))
ABOVE_B73 = int((LAT_B73 > MEAN_B73).sum())
P50_32_B73, P95_32_B73 = 118.0, 131.0                          # batch 32, given by the statement
THR1_B73 = 1000 / P50_B73
THR32_B73 = 32 * 1000 / P50_32_B73
RATIO_B73 = THR32_B73 / THR1_B73
PER_IMAGE_B73 = P50_32_B73 / 32
SLO_B73 = bool(P95_32_B73 <= 100)
PAGE_B73 = 1 - 0.95 ** 8

# ✏️ B7.4: PSI on four intervals (the quartiles of the reference), natural log, epsilon added to every share
Q_B74 = np.full(4, 0.25)
COUNTS_B74 = {1: [38, 52, 54, 56], 2: [96, 62, 30, 12], 3: [120, 56, 24, 0]}
P_B74 = {week: np.array(c) / 200 for week, c in COUNTS_B74.items()}


def psi_terms_b74(p, q=Q_B74, eps=0.0, log=np.log):
    """The four terms (p_k - q_k) log(p_k / q_k), eps added to every share of p and q (without renormalizing)."""
    p, q = np.asarray(p, float) + eps, np.asarray(q, float) + eps
    return (p - q) * log(p / q)


C1_W1_B74 = float(psi_terms_b74(P_B74[1])[0])
PSI_W1_B74 = float(psi_terms_b74(P_B74[1]).sum())
PSI_W2_B74 = float(psi_terms_b74(P_B74[2]).sum())
TOP_W2_B74 = "ABCD"[int(np.argmax(psi_terms_b74(P_B74[2])))]
KL_PQ_W2_B74 = float(np.sum(P_B74[2] * np.log(P_B74[2] / Q_B74)))
KL_QP_W2_B74 = float(np.sum(Q_B74 * np.log(Q_B74 / P_B74[2])))
PSI_W3_B74 = float(psi_terms_b74(P_B74[3], eps=1e-4).sum())
PSI_W3_EPS3_B74 = float(psi_terms_b74(P_B74[3], eps=1e-3).sum())
# a classic mistake for i): epsilon added to the production shares only (or to the empty share only: same at 3 decimals)
PSI_W3_EPS3_PART_B74 = float(((P_B74[3] + 1e-3 - Q_B74) * np.log((P_B74[3] + 1e-3) / Q_B74)).sum())
'''

LIBRARY_CHECKS = r'''
# The same answers from the libraries (no download): NumPy, SciPy, torchvision on the `meta` device, safetensors,
# PyYAML, logging, argparse, subprocess, DataLoader, torch.export, torch.load and scikit-learn
import io
import json
import logging
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from scipy import stats

# quiz Q1: run the three cells in the order 1, 2, 3, 2, 3 and keep what the last one prints
cells, namespace, printed = {1: "x = 2", 2: "x = x * 10", 3: "shown = x + 1"}, {}, None
for k in (1, 2, 3, 2, 3):
    exec(cells[k], namespace)
assert namespace["shown"] == Q1A

# quiz Q3: a WARNING level lets through WARNING and ERROR only ; an exception swallowed by print ends with code 0
records = []
logger = logging.getLogger("wb_chB7_q3")
logger.propagate = False
handler = logging.Handler(level=logging.WARNING)
handler.emit = records.append
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
for level in LEVELS_Q3:
    logger.log(getattr(logging, level), "message")
logger.removeHandler(handler)
assert len(records) == Q3C
swallow = "try:\n    raise RuntimeError('epoch 2: loss is nan')\nexcept Exception as err:\n    print(err)\n"
assert subprocess.run([sys.executable, "-c", swallow], capture_output=True).returncode == 0
unknown = "import argparse; p = argparse.ArgumentParser(); p.add_argument('--lr'); p.parse_args(['--lerning-rate', '1'])"
assert subprocess.run([sys.executable, "-c", unknown], capture_output=True).returncode == 2

# quiz Q4: PyYAML 6.0 reads 5.0e-4 as a float, 5e-4 and "5.0e-4" as strings ; a dataclass does not check types
assert yaml.safe_load("lr: 5.0e-4")["lr"] == 0.0005
assert yaml.safe_load("lr: 5e-4")["lr"] == "5e-4" and yaml.safe_load('lr: "5.0e-4"')["lr"] == "5.0e-4"


@dataclass(frozen=True)
class FrozenOnly:
    lr: float = 0.05


assert FrozenOnly(lr="1e-3").lr == "1e-3"                       # accepted: no check of the annotation

# quiz Q5: float addition is not associative ; without a generator, an extra draw moves the DataLoader's order
assert ((0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3)) is False
import torch
from torch.utils.data import DataLoader


def first_order(extra_draw):
    torch.manual_seed(0)
    if extra_draw:
        torch.rand(1)
    return next(iter(DataLoader(list(range(10)), batch_size=10, shuffle=True))).tolist()


assert first_order(False) != first_order(True)

# quiz Q7, Q9 and recall R1
assert Q7E == 1.0 and Q9A == 29.0 and Q9B == 200.0 and math.isclose(Q9D, 9.0) and math.isclose(Q9E, 160.0)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "runs.jsonl"
    for seed in list(range(7)) + [10, 11]:                     # 7 runs, then 2 runs again with other seeds
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"seed": seed}) + "\n")
    assert len(path.read_text(encoding="utf-8").splitlines()) == R1D
assert datetime.strptime("20261104-161731", "%Y%m%d-%H%M%S").strftime("%H:%M:%S") == "16:17:31"

# quiz Q8: load_state_dict refuses tensors of another shape ; safetensors metadata must be strings
narrow, wide = torch.nn.Linear(12, 3), torch.nn.Linear(16, 3)
try:
    wide.load_state_dict(narrow.state_dict())
except RuntimeError as err:
    assert "size mismatch" in str(err)
else:
    raise AssertionError("load_state_dict accepted tensors of another shape")
from safetensors.torch import save as save_safetensors

try:
    save_safetensors({"w": torch.zeros(2)}, metadata={"version": 1})
except TypeError:
    pass
else:
    raise AssertionError("safetensors accepted a metadata value that is not a string")

# quiz Q10: an export without dynamic_shapes refuses another batch size ; a data-dependent if does not export
small = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3)).eval()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    exported = torch.export.export(small, (torch.randn(8, 3, 2, 2),)).module()
    try:
        exported(torch.randn(3, 3, 2, 2))
    except Exception as err:                                    # « Guard failed: input.size()[0] == 8 »
        assert "8" in str(err)
    else:
        raise AssertionError("the exported model accepted a batch of 3")


class DataDependent(torch.nn.Module):
    def forward(self, x):
        if x.mean() > 0:
            return 2 * x
        return x - 1


import contextlib

torch_loggers = ("torch", "torch.fx.experimental.symbolic_shapes", "torch._dynamo", "torch.export")
saved_levels = {name: logging.getLogger(name).level for name in torch_loggers}
try:                                                            # torch.export prints the trace that failed
    for name in torch_loggers:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.export.export(DataDependent(), (torch.randn(4, 3),))
except Exception as err:
    assert "data-dependent" in str(err), str(err)[:200]
else:
    raise AssertionError("a data-dependent if was exported")
finally:
    for name, level in saved_levels.items():
        logging.getLogger(name).setLevel(level)

# recall R2: weights_only=True refuses an arbitrary Python object


class Trainer:
    pass


buffer = io.BytesIO()
torch.save({"trainer": Trainer()}, buffer)
buffer.seek(0)
try:
    torch.load(buffer, weights_only=True)
except Exception:
    pass
else:
    raise AssertionError("weights_only=True loaded a Python object")

# recall R3: the warning of scikit-learn when a model was written by another version
from sklearn.exceptions import InconsistentVersionWarning

assert issubclass(InconsistentVersionWarning, UserWarning)

# ✏️ B7.1: ConvNeXt-Tiny counted on the `meta` device (no memory, no download) ; the MP5 VGG and its safetensors
from torchvision.models import convnext_tiny

with torch.device("meta"):
    convnext = convnext_tiny(weights=None)
linear = [m for m in convnext.modules() if isinstance(m, torch.nn.Linear)]
assert sum(p.numel() for p in convnext.parameters()) == N_B71 and not list(convnext.buffers())
assert sum(m.weight.numel() for m in linear) == LIN_W_B71 and sum(m.bias.numel() for m in linear) == LIN_B_B71


def vgg_mp5(widths=(16, 32, 64)):
    """The VGG of the MP5 solution (cifarkit.models.vgg with downsample="pool" and batchnorm)."""
    layers, c = [], 3
    for i, w in enumerate(widths):
        for _ in range(2):
            layers += [torch.nn.Conv2d(c, w, 3, padding=1, bias=False), torch.nn.BatchNorm2d(w), torch.nn.ReLU()]
            c = w
        if i < len(widths) - 1:
            layers.append(torch.nn.MaxPool2d(2))
    layers += [torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(c, 10)]
    return torch.nn.Sequential(*layers)


vgg = vgg_mp5()
state = vgg.state_dict()
assert sum(p.numel() for p in vgg.parameters()) == VGG_PARAMS_B71
assert sum(t.numel() * t.element_size() for t in state.values()) == VGG_BYTES_B71
assert len(save_safetensors(state)) == VGG_FILE_B71
assert sum(p.numel() for p in vgg[-1].parameters()) == VGG_LINEAR_B71

# ✏️ B7.2: the Student intervals of SciPy (exact quantile 2.7764 instead of the 2.776 of the table)
assert math.isclose(stats.t.ppf(0.975, 4), T4_B72, abs_tol=5e-4)
ci_a = stats.ttest_1samp(A_B72, 0).confidence_interval()
ci_d = stats.ttest_rel(B_B72, A_B72).confidence_interval()
assert round(float(ci_a.low), 2) == round(LOW_A_B72, 2)
assert round(float(ci_d.high - ci_d.low) / 2, 2) == round(HW_D_B72, 2) and (float(ci_d.low) > 0) == (not ZERO_IN_D_B72)

# ✏️ B7.3: the other conventions of np.percentile, for the record
assert np.percentile(LAT_B73, 95, method="inverted_cdf") == 35.7 and np.percentile(LAT_B73, 50, method="inverted_cdf") == 20.5

# ✏️ B7.4: PSI = KL(p || q) + KL(q || p), with scipy.stats.entropy
for week in (1, 2):
    kl = stats.entropy(P_B74[week], Q_B74) + stats.entropy(Q_B74, P_B74[week])
    assert math.isclose(kl, float(psi_terms_b74(P_B74[week]).sum()))
assert math.isclose(stats.entropy(Q_B74, P_B74[2]), KL_QP_W2_B74)
with np.errstate(divide="ignore", invalid="ignore"):
    assert np.isinf(psi_terms_b74(P_B74[3]).sum())
print("✅ the paper answers agree with NumPy, SciPy, torchvision, safetensors, PyYAML, logging, subprocess and torch")
'''

exec(PAPER_CONTEXT)  # the builder computes a few mistakes with the same constants

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
LETTERS = 'the letters, e.g. "EF"'
WHOLE = "a whole number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
DEC4 = "a number, 4 decimals"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


def several(answer: str, options: str, where: str) -> str:
    """Mistakes for a question answered by several letters: every other non-empty set of the options, grouped into
    three messages that name neither the letter in excess nor the missing one (« il manque », « en trop », both)."""
    groups = {"missing": [], "extra": [], "both": []}
    for size in range(1, len(options) + 1):
        for combo in combinations(options, size):
            chosen = "".join(combo)
            if chosen == answer:
                continue
            missing, extra = set(answer) - set(chosen), set(chosen) - set(answer)
            groups["both" if missing and extra else "missing" if missing else "extra"].append(chosen)
    messages = {"missing": f"il manque au moins une lettre : relis {where}",
                "extra": f"au moins une des lettres choisies n'a pas sa place : relis {where}",
                "both": f"il manque au moins une lettre, et une autre n'a pas sa place : relis {where}"}
    return "mistakes={" + ", ".join(f'"{messages[k]}": {set(v)!r}' for k, v in groups.items() if v) + "}"


PAPER = [
    Paper("B7.Q1", "Les pièges du notebook : état caché et ordre d'exécution", [
        ("a", WHOLE, "Q1A", r'''mistakes={"le noyau garde l'ordre où tu as exécuté les cellules, pas celui où elles sont écrites : refais la suite 1, 2, 3, 2, 3": 21,
          "compte les exécutions de la cellule 2 dans la suite 1, 2, 3, 2, 3": 2001,
          "relis la cellule 3 : elle affiche x + 1": 200}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "« Redémarrer » vide le noyau : d'où viendrait encore scale ?",
                                           "C": "Jupyter n'exécute que les cellules du fichier : relis « Ce que cache un notebook » (§207.1)"})),
        ("c", TF, "False", r'''mistakes={"un .ipynb range code, sorties et images en JSON : relis « Ce que cache un notebook » (§207.1)": True}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "un module déjà importé est-il relu par le noyau ? Relis la fin de « pyproject.toml et installation éditable »",
                                           "B": "le lien éditable existe déjà : est-ce l'installation ou le noyau qui garde l'ancienne version ?"})),
        ("e", TF, "True", r'''mistakes={"supprimer une cellule touche-t-il au noyau ? Relis « Ce que cache un notebook » (§207.1)": False}'''),
    ]),
    Paper("B7.Q2", "src/, tests/, configs/, scripts/ : où va quoi ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "un fichier de tests vérifie le code ; l'API, elle, importe cette fonction",
                                           "C": "configs/ ne contient que des réglages, pas du code",
                                           "D": "scripts/ range ce qu'on lance à la main ; cette fonction est importée par l'entraînement et l'API"})),
        ("b", LETTER, '"D"', letters("D", {"A": "src/cifarprod/ ne contient que le code qu'importent l'entraînement et le service : ce script, on le lance",
                                           "B": "ce script ne vérifie pas le code, il prépare les données",
                                           "C": "configs/ ne contient que des réglages, pas du code"})),
        ("c", LETTER, '"C"', letters("C", {"A": "un fichier YAML de réglages n'est pas du code importable",
                                           "B": "ce fichier décrit un entraînement ; il ne teste rien lui-même",
                                           "D": "scripts/ range du code à lancer ; un fichier YAML est un réglage"})),
        ("d", LETTER, '"B"', letters("B", {"A": "src/cifarprod/ ne contient que le code qui sert en production",
                                           "C": "configs/ ne contient que des réglages",
                                           "D": "où pytest cherche-t-il les tests d'un dépôt ? Relis l'arborescence de 207.1"})),
        ("e", TF, "False", r'''mistakes={"relis le paragraphe qui suit l'arborescence (§207.1) : les données et les sorties des runs vont-elles dans git ?": True}'''),
        ("f", LETTER, '"B"', letters("B", {"A": "relis l'avantage du layout src/ (§207.1) : le dossier courant suffit-il pour importer cifarprod ?",
                                           "C": "pytest installe-t-il quoi que ce soit ? Relis le layout src/ (§207.1)"})),
        ("g", TF, "True", r'''mistakes={"relis [project.scripts] (§207.1) : que crée cette ligne à l'installation ?": False}'''),
    ]),
    Paper("B7.Q3", "print ou logging : niveaux, destinations et messages utiles", [
        ("a", LETTER, '"B"', letters("B", {"A": "DEBUG sert au détail pour chercher un bug : l'écriture de l'artefact en est-elle un ?",
                                           "C": "WARNING signale l'inattendu : écrire l'artefact à la fin d'un run réussi l'est-il ?",
                                           "D": "ERROR dit qu'une opération a échoué : relis les niveaux (§207.2)"})),
        ("b", LETTER, '"C"', letters("C", {"A": "un entraînement dix fois plus lent que prévu : un simple détail de débogage ?",
                                           "B": "est-ce une étape normale, ou quelque chose d'inattendu qui n'empêche pas de continuer ?",
                                           "D": "le programme continue : une opération a-t-elle échoué ?"})),
        ("c", WHOLE, "Q3C", r'''mistakes={"un niveau WARNING laisse-t-il passer les messages INFO ? Relis les niveaux (§207.2)": 5,
          "un message ERROR est plus grave qu'un WARNING : passe-t-il le niveau choisi ?": 1,
          "un niveau WARNING laisse-t-il passer DEBUG et INFO ?": 6}'''),
        ("d", TF, "False", r'''mistakes={"relis « Le logging plutôt que print » (§207.2) : où configure-t-on les handlers, et combien de fois ?": True}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "quel fait vérifiable ce message donne-t-il ? Relis la fin de « Le logging plutôt que print »",
                                           "B": "un message utile donne des valeurs : lequel en donne ?"})),
        ("f", WHOLE, "0", r'''mistakes={"le except affiche l'erreur, puis le script va au bout : comment finit-il ? Relis « Les codes de sortie » (§207.2)": 1,
          "2 est le code d'argparse devant une mauvaise option : relis « Les codes de sortie »": 2}'''),
    ]),
    Paper("B7.Q4", "Ce qui va dans la configuration, et ce qui ne doit jamais y aller", [
        ("a", LETTERS, '"ACD"', several("ACD", "ABCD", "207.3 et « Variables d'environnement et secrets » (§207.9)")),
        ("b", LETTER, '"A"', letters("A", {"B": "relis le rappel outil sur PyYAML (§207.3) : que faut-il pour qu'un nombre en notation scientifique soit lu comme un float ?",
                                           "C": "des guillemets donnent-ils un nombre ou une chaîne en YAML ?"})),
        ("c", WHOLE, "2", r'''mistakes={"relis main de la fiche (§207.2) : que rend-il quand load_config lève une ValueError ?": 0,
          "1 est le code d'une erreur générale : relis le except de main (§207.2)": 1}'''),
        ("d", TF, "True", r'''mistakes={"relis « Variables d'environnement et secrets » (§207.9) : que garde une image de chacune de ses couches ?": False}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "un commit suivant efface-t-il le précédent de l'historique ?",
                                           "B": "un dépôt privé se clone, se partage et peut devenir public : relis 207.9"})),
        ("f", TF, "False", r'''mistakes={"relis le commentaire de __post_init__ dans TrainConfig (§207.3) : une dataclass vérifie-t-elle les types ?": True}'''),
    ]),
    Paper("B7.Q5", "Même graine, résultats différents : les sources de non-déterminisme", [
        ("a", LETTER, '"C"', letters("C", {"A": "l'initialisation de nn.Linear tire dans le générateur de PyTorch : torch.manual_seed le fixe-t-il ?",
                                           "B": "le dropout tire dans le générateur de PyTorch : torch.manual_seed le fixe-t-il ?"})),
        ("b", TF, "True", r'''mistakes={"relis le paragraphe qui suit seed_everything (§207.4) : l'addition des flottants est-elle associative ?": False}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "c'est le comportement sans warn_only : relis l'encadré 🕰️ de 207.4",
                                           "B": "PyTorch ne remplace pas une opération par une autre : relis l'encadré 🕰️ de 207.4"})),
        ("d", TF, "False", r'''mistakes={"calcule les deux sommes en Python : l'addition des flottants est-elle associative ?": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "sans generator, où le DataLoader tire-t-il son ordre ? Relis le paragraphe qui suit seed_everything (§207.4)",
                                           "C": "un tirage de plus n'est pas une erreur : que change-t-il au générateur global ?"})),
    ]),
    Paper("B7.Q6", "requirements.txt, fichier de verrouillage, pyproject.toml : qui fige quoi ?", [
        ("a", LETTER, '"C"', letters("C", {"A": "une borne >= dit-elle une version exacte ? Relis « pyproject.toml et installation éditable » (§207.1)",
                                           "B": "Colab installe ses propres versions : que dit une borne inférieure ?"})),
        ("b", LETTER, '"A"', letters("A", {"B": "pyproject.toml donne des bornes : relis « Les versions figées et l'empreinte des données » (§207.4)",
                                           "C": "le cache de pip garde des fichiers téléchargés, pas ce qu'un projet a installé : relis « Les versions figées et l'empreinte des données » (§207.4)"})),
        ("c", TF, "False", r'''mistakes={"relis « Les versions figées » (§207.4) : une bibliothèque qu'on publie garde-t-elle des versions exactes ou des bornes ?": True}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "le run a tourné avec du code modifié : ce code est-il tout entier dans a1b2c3d ?",
                                           "C": "un commit enregistré reste valide : que manque-t-il au run ?"})),
        ("e", TF, "True", r'''mistakes={"un fichier de verrouillage fige-t-il des versions exactes ou des bornes ? Relis « Les versions figées » (§207.4)": False}'''),
    ]),
    Paper("B7.Q7", "Que tester dans un projet de machine learning ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "la forme de la sortie dépend-elle de zero_grad ? Relis « Quatre familles de tests » (§207.6)",
                                           "C": "le schéma des données ne voit pas la boucle d'entraînement : relis « Quatre familles de tests »"})),
        ("b", LETTER, '"C"', letters("C", {"A": "l'entraînement fumée vérifie que tout va au bout, pas le contenu des données",
                                           "B": "apprendre 16 images par cœur dit-il quelque chose du recouvrement des jeux ?"})),
        ("c", LETTER, '"A"', letters("A", {"B": "relis la puce « bout en bout » (§207.6) : l'entraînement fumée juge-t-il l'accuracy ?",
                                           "C": "l'entraînement fumée lance un entraînement, pas l'API"})),
        ("d", TF, "False", r'''mistakes={"relis la puce « bout en bout » (§207.6) : que vérifie l'entraînement fumée, et que ne juge-t-il pas ?": True}'''),
        ("e", DEC1, "Q7E", r'''decimals=1, mistakes={"relis la grille de Breck et coll. (§207.6) : le score résume-t-il les quatre sommes par leur moyenne ?": float(np.mean(list(SECTIONS_Q7.values()))),
          "relis la grille de Breck et coll. (§207.6) : le score résume-t-il les quatre sommes par leur total ?": float(sum(SECTIONS_Q7.values())),
          "relis la grille de Breck et coll. (§207.6) : le score résume-t-il les quatre sommes par la plus grande ?": 4.0}'''),
    ]),
    Paper("B7.Q8", "Artefact de modèle : les poids seuls ne suffisent pas", [
        ("a", LETTERS, '"AB"', several("AB", "ABCD", "« L'artefact du modèle » (§207.7)")),
        ("b", LETTER, '"B"', letters("B", {"A": "load_state_dict vérifie-t-il les noms et les formes des tenseurs ? Relis §23.9.2 du ch. 23",
                                           "C": "PyTorch modifie-t-il les tenseurs qu'on lui donne ? Relis §23.9.2 du ch. 23"})),
        ("c", LETTER, '"C"', letters("C", {"A": "relis le paragraphe qui suit le code de « L'artefact du modèle » : quels types prennent les métadonnées de safetensors ?",
                                           "B": "safetensors ignore-t-il ce qu'il ne sait pas écrire ? Relis « L'artefact du modèle » (§207.7)"})),
        ("d", TF, "True", r'''mistakes={"relis « Les versions figées et l'empreinte des données » (§207.4) : que devient l'empreinte quand un seul pixel change ?": False}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "la version change-t-elle le calcul ? Relis la traçabilité dans « L'artefact du modèle » (§207.7)",
                                           "B": "le format des pixels est une information à part de l'artefact : relis la liste de « L'artefact du modèle »"})),
    ]),
    Paper("B7.Q9", "Latence p50, p95 et débit : ce que ressent l'utilisateur", [
        ("a", WHOLE, "round(Q9A)", r'''mistakes={"relis la question : la moyenne pèse chaque requête, et 900 requêtes prennent 10 ms": 10,
          "c'est la moyenne des deux durées, sans tenir compte des effectifs": 105,
          "relis la question : on demande la moyenne": 200}'''),
        ("b", WHOLE, "round(Q9B)", r'''mistakes={"relis la question : on demande le p95, pas la moyenne": 29,
          "compte les requêtes de 10 ms : laissent-elles 95 % des mesures en dessous de cette valeur ?": 10,
          "c'est la moyenne des deux durées : le p95 se lit dans les mesures triées": 105}'''),
        ("c", TF, "True", r'''mistakes={"imagine 1 000 requêtes de 29 ms chacune : compare leur p95 à celui de l'énoncé, à moyenne égale": False}'''),
        ("d", WHOLE, "round(Q9D)", r'''mistakes={"convertis 150 ms en secondes avant de multiplier": 9000,
          "relis la loi de Little (§207.7) : L multiplie le débit par le temps de réponse": 400}'''),
        ("e", WHOLE, "round(Q9E)", r'''mistakes={"le débit divise la taille du batch par sa durée": 26}'''),
        ("f", LETTER, '"A"', letters("A", {"B": "une image d'un batch attend-elle les autres ? Relis la fin de « Latence, percentiles et débit » (§207.7)",
                                           "C": "relis la fin de « Latence, percentiles et débit » : que gagne-t-on à grouper, même sur CPU ?"})),
    ]),
    Paper("B7.Q10", "torch.export, ONNX, TorchScript : lequel choisir aujourd'hui ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis l'encadré 🕰️ de 207.8 : que dit PyTorch 2.11 de torch.jit.script ?",
                                           "C": "un objet sauvé par pickle se recharge-t-il sans le code de sa classe ?"})),
        ("b", LETTER, '"A"', letters("A", {"B": "torch.compile accélère un modèle dans un programme Python : produit-il un fichier ?",
                                           "C": "un state_dict se recharge-t-il sans PyTorch ni le code du modèle ?"})),
        ("c", TF, "False", r'''mistakes={"relis l'encadré 🕰️ de 207.8 : que produit torch.compile ?": True}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "relis le paragraphe sur dynamic_shapes (§207.8) : que devient la taille de batch d'un export sans dynamic_shapes ?",
                                           "C": "un modèle exporté complète-t-il une entrée ? Relis le paragraphe sur les gardes (§207.8)"})),
        ("e", TF, "False", r'''mistakes={"relis l'encadré 🕰️ de 207.8 : sur quoi s'appuie torch.onnx.export par défaut depuis la 2.9 ?": True}'''),
        ("f", LETTER, '"A"', letters("A", {"B": "un graphe figé peut-il contenir un if Python sur une valeur des données ? Relis le début de « torch.export et ONNX »",
                                           "C": "garder la seule branche de l'exemple, c'était le défaut de torch.jit.trace : relis le début de « torch.export et ONNX »"})),
    ]),
    Paper("B7.Q11", "Écart entraînement-production et dérive des données : vrai ou faux", [
        ("a", TF, "True", r'''mistakes={"relis « Pourquoi un modèle se dégrade sans changer » (§207.10)": False}'''),
        ("b", TF, "False", r'''mistakes={"relis la définition du training-serving skew (§207.7) : d'où vient l'écart ?": True}'''),
        ("c", TF, "True", r'''mistakes={"relis la fin de « Le test de Kolmogorov-Smirnov » (§207.10) : que devient une p-valeur avec beaucoup de données ?": False}'''),
        ("d", TF, "True", r'''mistakes={"relis ce qu'on surveille sur les sorties (§207.10) : faut-il attendre les labels ?": False}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "les deux chaînes de préparation ont-elles changé, ou les images reçues ? Relis 207.7 et 207.10",
                                           "C": "la relation entre une image et sa classe a-t-elle changé ? Relis « Pourquoi un modèle se dégrade sans changer » (§207.10)"})),
        ("f", TF, "False", r'''mistakes={"relis la fin de « Les seuils » (§207.10) : quand fixe-t-on les bornes des intervalles ?": True}'''),
    ]),
    Paper("B7.R1", "Ce qu'une ligne de journal d'expériences doit contenir", [
        ("a", LETTERS, '"ABCD"', several("ABCD", "ABCD", "le journal d'expériences du ch. 24 (🛠️ 24.15) et 207.4")),
        ("b", LETTER, '"A"', letters("A", {"B": "relis la fin du paragraphe qui suit log_run (§207.5) : que fausse l'effacement d'un run raté ?",
                                           "C": "remplacer la ligne, c'est réécrire le passé : relis la docstring de log_run (§207.5)"})),
        ("c", TF, "False", r'''mistakes={"relis le paragraphe qui suit log_run (§207.5) : un champ ajouté gêne-t-il les anciennes lignes ?": True}'''),
        ("d", WHOLE, "R1D", r'''mistakes={"un journal JSONL s'écrit en ajoutant des lignes : relance-t-on un run en réécrivant la sienne ?": 7,
          "le journal garde-t-il les sept premiers runs ? Relis la docstring de log_run (§207.5)": 2}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "relis new_run_id (§207.5) : de quoi l'identifiant est-il fait ?",
                                           "B": "relis new_run_id (§207.5) : le commit y figure-t-il ?"})),
    ]),
    Paper("B7.R2", "Checkpoint complet, state_dict et weights_only", [
        ("a", LETTERS, '"ABD"', several("ABD", "ABCD", "§23.9.1 du ch. 23 (« Tout dans un fichier »)")),
        ("b", LETTER, '"B"', letters("B", {"A": "l'optimiseur sert-il à prédire ? Relis §23.9.2 du ch. 23 (« Les poids seuls »)",
                                           "C": "l'état de l'optimiseur contient-il les poids ? Relis §23.9.2 du ch. 23"})),
        ("c", TF, "True", r'''mistakes={"relis l'encadré 🕰️ du ch. 23 sur les formats de sauvegarde : que refuse weights_only=True ?": False}'''),
        ("d", TF, "False", r'''mistakes={"relis §23.9.2 du ch. 23 : que faut-il faire avant load_state_dict ?": True}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "relis §23.9.1 du ch. 23 : que perd-on sans l'état de l'optimiseur ?",
                                           "C": "le load_state_dict du modèle se soucie-t-il de l'optimiseur ?"})),
    ]),
    Paper("B7.R3", "Sauvegarder un Pipeline et figer les versions", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis la première règle de « sauvegarder un modèle et ses versions » (ch. 15)",
                                           "C": "des coefficients sans leur prétraitement servent-ils à prédire ? Relis la première règle du ch. 15"})),
        ("b", TF, "False", r'''mistakes={"relis la deuxième règle du ch. 15 : que garantit la documentation d'une version à l'autre ?": True}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "un FutureWarning annonce un changement d'API : relis la deuxième règle du ch. 15",
                                           "C": "relis la deuxième règle du ch. 15 : scikit-learn prévient-il, depuis la 1.3 ?"})),
        ("d", TF, "True", r'''mistakes={"relis la troisième règle du ch. 15 : que fait le chargement d'un pickle ?": False}'''),
        ("e", LETTERS, '"ABC"', several("ABC", "ABCD", "la deuxième règle du ch. 15 : que note-t-on à côté du fichier, et sous quelle forme ?")),
        ("f", LETTER, '"C"', letters("C", {"A": "skops.io vérifie les types avant de charger, mais en Python : relis la troisième règle du ch. 15",
                                           "B": "joblib demande Python et scikit-learn : relis la troisième règle du ch. 15"})),
    ]),
    Paper("B7.1", "Taille d'un artefact : float32, bfloat16, int8, et la mémoire au chargement", [
        ("a", DEC1, "F32_B71", r'''decimals=1, mistakes={"un float32 occupe 4 octets : tu as compté 4 bits": N_B71 * 4 / 8 / MB,
          "relis la convention de l'énoncé : 1 Mo = 10⁶ octets": F32_MIB_B71,
          "un float32 occupe 4 octets par paramètre, pas 1": N_B71 / MB,
          "tu as compté en bits : un octet en fait 8": 32 * N_B71 / MB}'''),
        ("b", DEC1, "F32_MIB_B71", r'''decimals=1, mistakes={"relis la question : en Mio, avec 1 Mio = 2²⁰ octets": F32_B71,
          "1 Mio = 2²⁰ = 1 048 576 octets : relis la convention de l'énoncé": 4 * N_B71 / 1_024_000}'''),
        ("c", DEC1, "BF16_B71", r'''decimals=1, mistakes={"un bfloat16 occupe 2 octets par paramètre, pas 4": F32_B71,
          "relis l'unité : en Mo, 1 Mo = 10⁶ octets": 2 * N_B71 / MIB,
          "le bfloat16 a 16 bits : combien d'octets ?": N_B71 / MB}'''),
        ("d", DEC1, "INT8_B71", r'''decimals=1, mistakes={"un int8 occupe 1 octet, pas 2": BF16_B71,
          "relis l'unité : en Mo, 1 Mo = 10⁶ octets": N_B71 / MIB}'''),
        ("e", DEC1, "DYNQ_B71", r'''decimals=1, mistakes={"seules les matrices des couches denses passent en int8 : relis l'énoncé": INT8_B71,
          "les biais des couches denses restent en float32 : relis l'énoncé": (LIN_W_B71 + LIN_B_B71 + 4 * (N_B71 - LIN_W_B71 - LIN_B_B71)) / MB,
          "n'oublie pas les paramètres hors des matrices denses, en float32": LIN_W_B71 / MB,
          "relis l'unité : en Mo, 1 Mo = 10⁶ octets": DYNQ_B71 * MB / MIB}'''),
        ("f", DEC1, "RATIO_B71", r'''decimals=1, mistakes={"relis la question : e) ne passe pas tous les paramètres en int8": 4.0,
          "vérifie le sens du rapport : a) divisé par e)": 1 / RATIO_B71}'''),
        ("g", DEC1, "PEAK_B71", r'''decimals=1, mistakes={"relis l'énoncé : combien de copies des poids sont en mémoire au moment de load_state_dict ?": F32_MIB_B71,
          "relis l'unité : en Mio": 2 * F32_B71,
          "relis l'énoncé : deux copies des poids, pas trois": 3 * F32_MIB_B71}'''),
        ("h", WHOLE, "VGG_BYTES_B71", r'''mistakes={"le state_dict contient aussi les buffers des batchnorms": 4 * VGG_PARAMS_B71,
          "relis l'énoncé : num_batches_tracked est un int64, de 8 octets": 4 * (VGG_PARAMS_B71 + VGG_FLOAT_BUFFERS_B71 + 6),
          "n'oublie pas les six compteurs num_batches_tracked": 4 * (VGG_PARAMS_B71 + VGG_FLOAT_BUFFERS_B71),
          "relis l'énoncé : chaque batchnorm a deux buffers d'une valeur par canal": 4 * (VGG_PARAMS_B71 + VGG_FLOAT_BUFFERS_B71 // 2) + 48}'''),
        ("i", WHOLE, "HEADER_B71", r'''mistakes={"les buffers des batchnorms sont aussi dans le fichier : retire-les avec les paramètres": VGG_FILE_B71 - 4 * VGG_PARAMS_B71,
          "relis la question : on demande l'en-tête, ce qui reste une fois les tenseurs retirés": VGG_BYTES_B71}'''),
        ("j", DEC1, "LIN_SHARE_B71", r'''decimals=1, mistakes={"relis la question : en pourcentage": VGG_LINEAR_B71 / VGG_PARAMS_B71,
          "vérifie le sens du rapport : les paramètres de la couche dense divisés par tous les paramètres": 100 * VGG_PARAMS_B71 / VGG_LINEAR_B71}'''),
    ]),
    Paper("B7.2", "Cinq graines, un intervalle : la dispersion d'une accuracy rapportée", [
        ("a", DEC1, "MEAN_A_B72", r'''decimals=1, mistakes={"divise la somme par le nombre de graines": float(A_B72.sum() / 4)}'''),
        ("b", DEC3, "S_A_B72", r'''decimals=3, mistakes={"l'écart-type corrigé divise par n − 1 : relis le rappel maths de 207.4": float(np.std(A_B72)),
          "c'est la variance : prends sa racine": float(np.var(A_B72, ddof=1)),
          "c'est la somme des carrés des écarts : divise-la, puis prends la racine": float(np.sum((A_B72 - A_B72.mean()) ** 2))}'''),
        ("c", DEC3, "SE_A_B72", r'''decimals=3, mistakes={"l'erreur type divise s par √n, avec n = 5 graines": S_A_B72 / 2,
          "l'erreur type divise s par √n, pas par n": S_A_B72 / 5,
          "pars de l'écart-type corrigé, divisé par n − 1 : relis b)": interval_b72(A_B72, ddof=0)[2]}'''),
        ("d", DEC2, "HW_A_B72", r'''decimals=2, mistakes={"avec cinq valeurs, on prend le quantile de Student : relis le rappel maths de 207.4": 1.96 * SE_A_B72,
          "le quantile de Student se lit à n − 1 = 4 degrés de liberté : relis la table de 207.4": 2.571 * SE_A_B72,
          "la demi-largeur multiplie t par l'erreur type, pas par l'écart-type": T4_B72 * S_A_B72,
          "pars de l'écart-type corrigé, divisé par n − 1": interval_b72(A_B72, ddof=0)[3]}'''),
        ("e", DEC2, "LOW_A_B72", r'''decimals=2, mistakes={"la borne basse retire la demi-largeur à la moyenne": MEAN_A_B72 + HW_A_B72,
          "avec cinq valeurs, on prend le quantile de Student : relis le rappel maths de 207.4": MEAN_A_B72 - 1.96 * SE_A_B72,
          "le quantile de Student se lit à n − 1 = 4 degrés de liberté : relis la table de 207.4": MEAN_A_B72 - 2.571 * SE_A_B72}'''),
        ("f", TF, "OVERLAP_B72", r'''mistakes={"calcule l'intervalle de B comme celui de A, puis compare leurs bornes": not OVERLAP_B72}'''),
        ("g", DEC1, "MEAN_D_B72", r'''decimals=1, mistakes={"vérifie le sens de la différence : B − A": -MEAN_D_B72}'''),
        ("h", DEC2, "HW_D_B72", r'''decimals=2, mistakes={"avec cinq différences, on prend le quantile de Student : relis le rappel maths de 207.4": 1.96 * SE_D_B72,
          "la demi-largeur multiplie t par l'erreur type des différences, pas par leur écart-type": T4_B72 * S_D_B72,
          "le quantile de Student se lit à n − 1 = 4 degrés de liberté : relis la table de 207.4": 2.571 * SE_D_B72}'''),
        ("i", TF, "ZERO_IN_D_B72", r'''mistakes={"calcule les deux bornes de l'intervalle de la différence avec g) et h)": not ZERO_IN_D_B72}'''),
    ]),
    Paper("B7.3", "Latence et débit : p50 et p95 sur vingt mesures, batch 1 contre batch 32", [
        ("a", DEC2, "MEAN_B73", r'''decimals=2, mistakes={"relis la question : la moyenne des vingt mesures": P50_B73,
          "divise la somme par le nombre de mesures, 20": float(LAT_B73.sum() / 19)}'''),
        ("b", DEC2, "P50_B73", r'''decimals=2, mistakes={"c'est la plus petite valeur qui laisse la moitié des mesures en dessous (method=\"inverted_cdf\") : relis la convention de l'énoncé": 20.5,
          "relis la position : h = 0,5 × (n − 1), avec n = 20 mesures": 20.6,
          "relis la question : on demande le p50, la médiane": MEAN_B73}'''),
        ("c", DEC2, "P95_B73", r'''decimals=2, mistakes={"c'est la convention « la plus petite valeur qui laisse 95 % des mesures en dessous » (inverted_cdf) : relis celle de l'énoncé": 35.7,
          "c'est la moyenne des deux plus grandes mesures : relis la position h de l'énoncé": 41.8,
          "c'est la plus grande mesure : relis la position h de l'énoncé": 47.9,
          "c'est une autre convention (position q × (n + 1)) : relis celle de l'énoncé": float(np.percentile(LAT_B73, 95, method="weibull"))}'''),
        ("d", WHOLE, "ABOVE_B73", r'''mistakes={"compte les mesures au-dessus de la moyenne, pas de la médiane": 10,
          "relis la question : au-dessus de la moyenne": 20 - ABOVE_B73}'''),
        ("e", WHOLE, "round(THR1_B73)", r'''mistakes={"relis la question : le débit se calcule ici avec le p50, pas la moyenne": round(1000 / MEAN_B73),
          "relis la question : le débit se calcule ici avec le p50, pas le p95": round(1000 / P95_B73)},
          fractional="le débit se donne ici en images par seconde, arrondi à l'entier"'''),
        ("f", WHOLE, "round(THR32_B73)", r'''mistakes={"un batch de 32 images : le débit divise sa taille par sa durée": round(1000 / P50_32_B73),
          "relis la question : le débit se calcule avec le p50 du batch, pas son p95": round(32_000 / P95_32_B73)},
          fractional="le débit se donne ici en images par seconde, arrondi à l'entier"'''),
        ("g", DEC1, "RATIO_B73", r'''decimals=1, mistakes={"vérifie le sens du rapport : f) divisé par e)": 1 / RATIO_B73,
          "tu as divisé les débits arrondis à l'entier : garde les valeurs non arrondies de e) et f)": round(THR32_B73) / round(THR1_B73),
          "le débit ne croît pas comme la taille du batch : calcule-le avec les mesures": 32.0}'''),
        ("h", DEC2, "PER_IMAGE_B73", r'''decimals=2, mistakes={"relis la question : le p50 du batch divisé par 32": P50_32_B73,
          "relis la question : on part du p50 du batch, pas de son p95": P95_32_B73 / 32}'''),
        ("i", TF, "SLO_B73", r'''mistakes={"chaque image attend la fin de son batch : compare le p95 d'un batch de 32 à l'objectif": not SLO_B73}'''),
        ("j", DEC2, "PAGE_B73", r'''decimals=2, mistakes={"relis la question : au moins une des huit dépasse le p95": 0.95 ** 8,
          "les probabilités ne s'additionnent pas ici : passe par l'événement contraire": 8 * 0.05,
          "relis la question : huit prédictions, pas une seule": 0.05}'''),
    ]),
    Paper("B7.4", "Indice de stabilité de population (PSI) à la main sur quatre intervalles", [
        ("a", DEC4, "C1_W1_B74", r'''decimals=4, mistakes={"le PSI utilise le logarithme naturel ln": float(psi_terms_b74(P_B74[1], log=np.log10)[0]),
          "vérifie le signe : p − q et ln(p/q) ont toujours le même signe": -C1_W1_B74,
          "relis la formule : la contribution multiplie ln(p/q) par p − q": float(np.log(P_B74[1][0] / 0.25))}'''),
        ("b", DEC3, "PSI_W1_B74", r'''decimals=3, mistakes={"le PSI utilise le logarithme naturel ln": float(psi_terms_b74(P_B74[1], log=np.log10).sum()),
          "il manque une partie du PSI : relis le lien avec la divergence KL (§207.10)": float(np.sum(P_B74[1] * np.log(P_B74[1] / Q_B74)))}'''),
        ("c", DEC3, "PSI_W2_B74", r'''decimals=3, mistakes={"le PSI utilise le logarithme naturel ln": float(psi_terms_b74(P_B74[2], log=np.log10).sum()),
          "il manque une partie du PSI : relis le lien avec la divergence KL (§207.10)": KL_PQ_W2_B74,
          "relis la question : on demande le PSI, la somme des quatre contributions (p − q) ln(p/q)": KL_QP_W2_B74}'''),
        ("d", LETTER, f'"{TOP_W2_B74}"', letters(TOP_W2_B74, {"A": "calcule la contribution de I₁ en semaine 2 et compare-la aux trois autres",
                                                               "B": "calcule la contribution de I₂ en semaine 2 et compare-la aux trois autres",
                                                               "C": "calcule la contribution de I₃ en semaine 2 et compare-la aux trois autres",
                                                               "D": "calcule la contribution de I₄ en semaine 2 et compare-la aux trois autres"})),
        ("e", DEC3, "KL_QP_W2_B74", r'''decimals=3, mistakes={"relis la question : q en premier, Σ q ln(q/p)": KL_PQ_W2_B74,
          "relis la question : on demande une seule des deux divergences": PSI_W2_B74,
          "la divergence KL utilise le logarithme naturel ln": float(np.sum(Q_B74 * np.log10(Q_B74 / P_B74[2])))}'''),
        ("f", LETTER, '"B"', letters("B", {"A": "compare le PSI de la semaine 2 au second seuil, 0,25",
                                           "C": "compare le PSI de la semaine 1 au premier seuil, 0,1"})),
        ("g", TF, "True", r'''mistakes={"calcule ln(0 / 0,25) : que vaut-il pour NumPy, et que devient son produit par (0 − 0,25) ?": False}'''),
        ("h", DEC2, "PSI_W3_B74", r'''decimals=2, mistakes={"le PSI utilise le logarithme naturel ln": float(psi_terms_b74(P_B74[3], eps=1e-4, log=np.log10).sum()),
          "relis l'énoncé : ε vaut 10⁻⁴ ici": PSI_W3_EPS3_B74}'''),
        ("i", DEC3, "PSI_W3_EPS3_B74", r'''decimals=3, mistakes={"relis la question : ε = 10⁻³": PSI_W3_B74,
          "relis l'énoncé : ε s'ajoute à toutes les parts, celles de la production et celles de la référence": PSI_W3_EPS3_PART_B74,
          "le PSI utilise le logarithme naturel ln": float(psi_terms_b74(P_B74[3], eps=1e-3, log=np.log10).sum())}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ B7.1 à B7.4)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "plusieurs choix : les lettres ensemble, entre guillemets, dans n'importe quel ordre (`\"EF\"`). "
               "Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore "
               "remplies affichent ⏳. Les questions « dans ta copie », l'estimation 🧮 B7.5, la réflexion "
               "(🗣️ B7.6, ⚖️ B7.7, 📄 B7.8) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to C (exercises B7.9 to B7.25). Session 76, first half: part A (B7.9 to B7.15) and the start of part B
# (🐛 B7.16), chB7_parts.py and chB7_code.py; second half: B7.17 to B7.21 appended to PART_B and part C (B7.22 to
# B7.25), chB7_parts2.py, chB7_code2.py and chB7_code3.py.
# ---------------------------------------------------------------------------
import dataclasses  # noqa: E402

import chB7_code  # noqa: E402,F401  (the cells; history_chB7.py reuses them)
import chB7_parts2  # noqa: E402,F401  (B7.17 to B7.21 into PART_B, part C into PARTS)
from chB7_parts import PART_B, PARTS  # noqa: E402,F401

for _part in PARTS:
    for _ex in _part.exercises:
        check_unique_keys(_ex.record or "x = 1", _ex.id)

# What the next half of the session adds (announced in the table and the footer until it exists)
NEXT_SESSION = [
    ("B", "B7.17 à B7.21", "le service (suite) : l'artefact, l'API, la latence, l'accélération sur CPU et l'export"),
    ("C", "B7.22 à B7.25", "l'exploitation : la dérive, l'intégration continue, le Dockerfile et le défi"),
]
_DONE = {ex.id for part in PARTS for ex in part.exercises}
NEXT_SESSION = [row for row in NEXT_SESSION if not ({f"B7.{n}" for n in range(int(row[1].split()[0][3:]),
                                                                              int(row[1].split()[2][3:]) + 1)} <= _DONE)]


def kind_part(part, kind: str):
    """The part with its tools for this notebook: «KIND» is "learner" (exercise notebook) or "solution"."""
    return dataclasses.replace(part, given=part.given.replace("«KIND»", "learner" if kind == "exercise" else "solution"))


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | B7.Q1 à B7.Q11, B7.R1 à B7.R3, B7.1 à B7.4 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(ex.table_row(part.key if i == 0 else ""))
    rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |" for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# B7 · Du notebook à la production — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/b7_production/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py B7`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# B7 · Du notebook à la production — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Restructurer un notebook en paquet Python installable, avec une ligne de commande, du logging et "
               "une configuration validée.\n"
               "- Rendre un entraînement reproductible et traçable (graines, déterminisme, versions, empreinte des "
               "données, journal des runs), et rapporter un score sur plusieurs graines avec son intervalle.\n"
               "- Écrire des tests utiles à un projet de ML et les automatiser dans une intégration continue.\n"
               "- Empaqueter un modèle avec son prétraitement, le servir par une API HTTP et mesurer sa latence.\n"
               "- Exporter et accélérer l'inférence sur CPU en vérifiant l'équivalence numérique ; détecter une "
               "dérive des données et décider d'une action.\n\n"
               "**Rappel express.** Intervalle sur $n$ graines : $\\bar a \\pm t_{0{,}975;\\,n-1}\\,\\frac{s}{\\sqrt n}$ "
               "($s$ divisé par $n - 1$). Percentile de `np.percentile` : position $h = \\frac{q}{100}(n - 1)$ dans "
               "les mesures triées, interpolation linéaire. Débit $= \\frac{\\text{taille du batch}}{\\text{latence "
               "du batch}}$ ; loi de Little $L = \\lambda W$. Taille des poids $= N \\times$ octets par paramètre "
               "(4, 2, 1). $\\operatorname{PSI} = \\sum_k (p_k - q_k)\\ln\\frac{p_k}{q_k} = D_{\\text{KL}}(p\\,\\|\\,q) "
               "+ D_{\\text{KL}}(q\\,\\|\\,p)$ ; $D = \\sup_x |F_{\\text{réf}}(x) - F_{\\text{prod}}(x)|$.")]


def footer_cells(kind: str) -> list:
    later = ("" if not NEXT_SESSION else
             "\n\n*Les exercices " + " et ".join(ids for _, ids, _ in NEXT_SESSION) + " seront ajoutés à la "
             "prochaine session de génération ; relance alors `python tools/start_chapter.py B7` pour obtenir le "
             "notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu transformer un notebook en paquet installable, avec une ligne de commande, une "
               "configuration validée, des runs journalisés et des tests qui attrapent de vraies erreurs ?\n"
               "2. Sais-tu rapporter une accuracy sur plusieurs graines avec son intervalle, et dire ce qui rend deux "
               "runs identiques au bit près (ou non) ?\n"
               "3. Sais-tu livrer un modèle dans un artefact qui se suffit, le servir par une API qui valide ses "
               "entrées, mesurer son p95 et vérifier un export ?\n"
               "4. Sais-tu calculer un PSI et lire un test KS, dire ce qu'une alerte mérite, et écrire le workflow "
               "et le Dockerfile qui rendent ton dépôt reproductible par n'importe qui ?\n\n"
               "**Pour aller plus loin** : l'article de Sculley et coll. (📄 B7.8) et le ML Test Score de Breck et "
               "coll., cités dans la fiche. La suite : le bonus B8, l'apprentissage par renforcement moderne."
               + later)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT + LIBRARY_CHECKS, PAPER_INTRO)
    for part in PARTS:
        cells += part_cells(kind_part(part, kind), kind)
    cells += footer_cells(kind)
    return cells


def main() -> int:
    write_notebook(f"{FOLDER}/03_notebook.ipynb", build("exercise"))
    write_notebook(f"{FOLDER}/05_solutions.ipynb", build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER)
    n_ex = sum(len(part.exercises) for part in PARTS)
    print(f"✅ {FOLDER}/03_notebook.ipynb and 05_solutions.ipynb written ({len(PAPER)} paper exercises, "
          f"{n_checks} paper checks, {n_ex} notebook exercises)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
