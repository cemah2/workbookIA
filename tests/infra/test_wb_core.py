"""Tests of wb.setup and the environment helpers."""

import random
import re
from pathlib import Path

import numpy as np
import pytest

import wb
from wb import core
from wb._versions import DIST_NAMES, EXPECTED_VERSIONS

ROOT = Path(__file__).resolve().parents[2]


def test_repo_root_found_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert wb.repo_root() == ROOT
    monkeypatch.setenv("WB_ROOT", str(ROOT))
    assert wb.repo_root() == ROOT


def test_setup_seeds_are_reproducible(capsys):
    wb.setup(seed=123, verbose=False, style=False)
    a = (random.random(), np.random.rand())
    wb.setup(seed=123, verbose=False, style=False)
    b = (random.random(), np.random.rand())
    assert a == b


def test_setup_seeds_torch():
    torch = pytest.importorskip("torch")
    wb.setup(seed=7, verbose=False, style=False)
    a = torch.rand(3)
    wb.setup(seed=7, verbose=False, style=False)
    assert torch.equal(a, torch.rand(3))


def test_setup_report_and_config(capsys):
    cfg = wb.setup(seed=1, fast=True)
    out = capsys.readouterr().out
    assert "FAST_MODE : True" in out and "Python" in out
    assert cfg.device in ("cpu", "cuda") and cfg.seed == 1 and cfg.fast is True
    assert cfg.root == ROOT


def test_fast_mode_env_override(monkeypatch):
    monkeypatch.setenv("WB_FAST_MODE", "0")
    cfg = wb.setup(fast=True, verbose=False, style=False)
    assert cfg.fast is False and wb.by_mode(1, 10) == 10
    monkeypatch.setenv("WB_FAST_MODE", "1")
    cfg = wb.setup(fast=False, verbose=False, style=False)
    assert cfg.fast is True and wb.by_mode(1, 10) == 1


def test_timer(capsys):
    with wb.timer("bloc") as t:
        sum(range(1000))
    assert t["seconds"] >= 0 and "bloc" in capsys.readouterr().out


def test_expected_versions_match_requirements():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==")
            pins[name.strip().lower()] = version.strip()
    for module, version in EXPECTED_VERSIONS.items():
        dist = DIST_NAMES.get(module, module)
        assert pins.get(dist) == version, f"{dist}: requirements.txt and _versions.py disagree"


def test_expected_versions_match_bible():
    bible = (ROOT / "docs" / "BIBLE.md").read_text(encoding="utf-8")
    section = bible.split("## 21.")[1].split("## 22.")[0]
    for module, version in EXPECTED_VERSIONS.items():
        assert re.search(re.escape(version), section), f"{module} {version} missing from BIBLE §21"


def test_ensure_does_nothing_when_installed():
    assert wb.ensure("numpy", "scikit-learn") == []


def test_is_colab_false_here():
    assert core.is_colab() is False


def test_setup_survives_a_broken_torch(monkeypatch, capsys):
    import importlib.util
    import sys as _sys

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name == "torch" else real_find_spec(name, *a))
    monkeypatch.setitem(_sys.modules, "torch", None)  # "import torch" now raises ImportError
    monkeypatch.setattr(core, "_TORCH_ERROR", [])
    cfg = wb.setup(seed=1, style=False)
    out = capsys.readouterr().out
    assert cfg.device == "cpu" and cfg.torch_available is False
    assert "ne se charge pas" in out
