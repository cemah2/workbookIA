"""Tests of tools/chapters/chapter_kit.py: the blocks shared by the chapter builders (audit P5 V-VI, A10 20).

The 🔮 of every chapter rely on ``guarded``, the 🐛 answered by letters on ``bug_letters_mistakes``, and every
notebook header on ``Ex.header``: a regression would change notebooks or recorded answers without a clear signal.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "chapters"))

import chapter_kit  # noqa: E402
from chapter_kit import ROCKET, Ex, bug_letters_mistakes, check_unique_keys, contract_entry, guarded  # noqa: E402


def _ex(ex_id, **kwargs):
    base = dict(id=ex_id, type="🔬", stars=3, minutes=40, title="Un titre", goal="un but", prereq="Ex 1.1",
                body="")
    return Ex(**{**base, **kwargs})


# ---------------------------------------------------------------------------
# guarded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["...", "None"])
def test_guarded_does_not_run_before_the_predictions(capsys, value):
    code = guarded("print('résultat')\nprint(a + 1)", ["a", "b"], "⏳ écris d'abord tes prédictions")
    exec(code, {"a": 1, "b": eval(value)})
    assert capsys.readouterr().out == "⏳ écris d'abord tes prédictions\n"


def test_guarded_runs_once_the_predictions_are_written(capsys):
    code = guarded("x = a + 1\n\nprint(x)", ["a", "b"], "⏳ attends")
    exec(code, {"a": 1, "b": [0, 1]})
    assert capsys.readouterr().out == "2\n"
    ast.parse(code)                                       # blank lines of the body stay valid Python


# ---------------------------------------------------------------------------
# bug_letters_mistakes and check_unique_keys
# ---------------------------------------------------------------------------
HINTS = {"C": "relis la ligne du learning rate", "F": "relis l'ordre des appels", "G": "relis la loss"}


def test_bug_letters_mistakes_has_one_entry_per_near_miss():
    mistakes = eval(bug_letters_mistakes("CFG", "ABDEH", HINTS))
    answers = list(mistakes.values())
    # 3 sets missing one faulty line, 5 with one line too many, 3 x 5 swaps
    assert len(mistakes) == 3 + 5 + 15 and len(set(answers)) == len(answers)
    assert all(answer == "".join(sorted(answer)) for answer in answers)          # letters in alphabetical order
    assert "CFG" not in answers
    assert mistakes["il manque une ligne fautive : relis la loss"] == "CF"
    assert mistakes["une ligne de trop : la ligne (A) est correcte ; relis ce qu'elle fait"] == "ACFG"


def test_bug_letters_mistakes_special_message_and_duplicates():
    special = {("C", "A"): "la ligne (A) ne change rien au learning rate"}
    mistakes = eval(bug_letters_mistakes("CFG", "AB", HINTS, special=special))
    assert mistakes["la ligne (A) ne change rien au learning rate"] == "AFG"
    with pytest.raises(ValueError, match="duplicate"):
        bug_letters_mistakes("CF", "A", {"C": "même message", "F": "même message"})


def test_check_unique_keys():
    check_unique_keys('wb.record("1.1", 2, mistakes={"a": 1, "b": 2})', "ok")
    with pytest.raises(ValueError, match="duplicate keys"):
        check_unique_keys('wb.record("1.1", 2, mistakes={"a": 1, "a": 2})', "Ex 1.1")


# ---------------------------------------------------------------------------
# Ex.header and Ex.table_row: 🚀 and « Fil rouge » from the contract
# ---------------------------------------------------------------------------
def test_contract_entry_reads_the_syllabus():
    assert contract_entry("21.26")["gpu"] is True
    assert contract_entry("21.Q1")["gpu"] is False
    assert contract_entry("99.99") == {}


def test_header_and_table_row_mark_a_gpu_exercise():
    gpu, cpu = _ex("21.26"), _ex("21.12")
    assert gpu.header().splitlines()[0] == f"### Ex 21.26 — Un titre 🔬 ★★★ ⏱️ 40 min {ROCKET}"
    assert ROCKET not in cpu.header()
    assert gpu.table_row("D") == f"| D | 21.26 | Un titre {ROCKET} | 🔬 | ★★★ | 40 |"
    assert _ex("21.12", title="a | b").table_row() == "|  | 21.12 | a \\| b | 🔬 | ★★★ | 40 |"


def test_header_shows_the_thread_of_the_contract_in_words(monkeypatch):
    monkeypatch.setattr(chapter_kit, "_CONTRACT", {"9.1": {"thread": "synth"}, "9.2": {"thread": "MNIST"},
                                                   "9.3": {"thread": "synth/MNIST"}, "9.4": {"thread": "—"}})
    assert "**Fil rouge :** synthétique" in _ex("9.1").header()
    assert "**Fil rouge :** MNIST" in _ex("9.2").header()
    assert "**Fil rouge :** synthétique/MNIST" in _ex("9.3").header()
    assert "Fil rouge" not in _ex("9.4").header()
    assert "**Fil rouge :** Penguins" in _ex("9.2", thread="Penguins").header()      # the builder's choice wins
    assert "**Fil rouge :** synthétique" in _ex("9.4", thread="synth").header()
