"""wb: utilities of the Deep Learning workbook.

>>> import wb
>>> cfg = wb.setup(seed=42, fast=True)   # seeds, device, plot style, environment report
>>> wb.check("2.3", my_answer)           # checks an answer without revealing it
>>> X, y = wb.synth.make_moons(200)      # synthetic data
>>> df = wb.datasets.load_penguins()     # common-thread datasets
>>> wb.plot.plot_decision_boundary(model, X, y)
>>> mylearn = wb.load_mylearn("learner") # your own library (or "ref" in solutions)

Importing ``wb`` is light: PyTorch is only imported by ``setup()`` (if
installed) and by the functions that need it.
"""

from wb import datasets, plot, synth
from wb._versions import __version__
from wb.checker import attempt, check, record
from wb.core import Config, by_mode, ensure, environment_report, get_device, is_colab, repo_root, seed_everything, setup, timer
from wb.impl import load_mylearn
from wb.testing import run_pytest

__all__ = [
    "__version__",
    "setup",
    "Config",
    "by_mode",
    "ensure",
    "timer",
    "get_device",
    "is_colab",
    "repo_root",
    "seed_everything",
    "environment_report",
    "check",
    "attempt",
    "record",
    "load_mylearn",
    "run_pytest",
    "datasets",
    "synth",
    "plot",
]
