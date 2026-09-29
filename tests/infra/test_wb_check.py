"""Tests of wb.check / wb.record: right values pass, wrong values fail, no leak."""

import json

import numpy as np
import pytest

import wb
from wb import checker as C


def entry(ex_id, value, **kwargs):
    return C.make_entry(ex_id, value, **kwargs)


def passes(ex_id, e, value):
    return C.check_entry(ex_id, e, value)[0]


def message(ex_id, e, value):
    return C.check_entry(ex_id, e, value)[2]


# ---------------------------------------------------------------- floats
def test_float_right_value_and_rounding_variants_pass():
    e = entry("x.1", 3.14159, decimals=2)
    for v in (3.14159, 3.14, 3.1449, "3.14", "3,14", np.float32(3.14159)):
        assert passes("x.1", e, v), v


def test_float_wrong_values_fail():
    e = entry("x.1", 3.14159, decimals=2)
    for v in (3.16, 3.0, -3.14, 31.4, 0.0):
        assert not passes("x.1", e, v), v


def test_float_boundary_tolerance():
    e = entry("x.2", 0.125000001, decimals=2)  # stored as "0.13"
    assert passes("x.2", e, 0.124999999)


def test_hash_never_contains_the_value():
    e = entry("x.3", 42.4242, decimals=4)
    dumped = json.dumps(e)
    assert "42.42" not in dumped and "42.4242" not in dumped


def test_decimals_required_for_floats():
    with pytest.raises(ValueError):
        entry("x.4", 1.5)


def test_same_value_different_ids_give_different_hashes():
    assert entry("a.1", 7)["hash"] != entry("a.2", 7)["hash"]


# ---------------------------------------------------------------- diagnostics
@pytest.mark.parametrize(
    "value, expected_fragment",
    [
        (-3.14159, "signe"),
        (314.159, "proportion"),
        (0.0314159, "pourcentage"),
        (3.2, "ordre de grandeur est bon"),
        (31.0, "trop grande"),
        (0.2, "trop petite"),
    ],
)
def test_float_diagnostics(value, expected_fragment):
    e = entry("d.1", 3.14159, decimals=2)
    assert expected_fragment in message("d.1", e, value)


def test_complement_and_coarse_diagnostics():
    e = entry("d.2", 0.25, decimals=2)
    assert "complément" in message("d.2", e, 0.75)
    e = entry("d.3", 0.123, decimals=3)
    assert "presque" in message("d.3", e, 0.12)


def test_int_off_by_one_and_type():
    e = entry("d.4", 344)
    assert passes("d.4", e, 344.0)
    assert "à 1 près" in message("d.4", e, 345)
    assert "entier" in message("d.4", e, 12.5)


def test_common_mistakes_are_reported():
    e = entry("d.5", 10.0, decimals=1, mistakes={"somme au lieu de moyenne": 50.0})
    assert "somme au lieu de moyenne" in message("d.5", e, 50.0)
    with pytest.raises(ValueError):
        entry("d.6", 10.0, decimals=1, mistakes={"identique": 10.0})


# ---------------------------------------------------------------- other kinds
def test_strings_ignore_case_accents_and_spaces():
    e = entry("s.1", "Rétropropagation du gradient")
    for v in ("retropropagation du gradient", "RÉTROPROPAGATIONDUGRADIENT", " rétro propagation du gradient "):
        assert passes("s.1", e, v)
    assert not passes("s.1", e, "backprop")
    assert "chaîne" in message("s.1", e, 3)


def test_booleans_accept_french_words():
    e = entry("b.1", True)
    assert passes("b.1", e, "vrai") and passes("b.1", e, "Oui") and passes("b.1", e, 1)
    assert not passes("b.1", e, False)


def test_arrays_shape_values_and_diagnostics():
    value = np.array([[1.5, 2.25], [3.0, 4.125]])
    e = entry("r.1", value, decimals=3)
    assert passes("r.1", e, value.tolist())
    assert "transposé" in message("r.1", e, value.T + 0) or "élément" in message("r.1", e, value.T)
    assert "reshape" in message("r.1", e, value.ravel())
    assert "3 élément(s) sur 4" in message("r.1", e, [[1.5, 2.25], [3.0, 9.0]])
    assert "Forme attendue" in message("r.1", e, np.zeros((3, 2)))


def test_transposed_rectangular_array():
    value = np.arange(6.0).reshape(2, 3)
    e = entry("r.2", value, decimals=1)
    assert "transposé" in message("r.2", e, value.T)


def test_integer_arrays_need_no_decimals():
    e = entry("r.3", [1, 2, 3])
    assert passes("r.3", e, np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        entry("r.4", [0.5, 1.5])


def test_sets_ignore_order():
    e = entry("t.1", ["b", "a", "c"], ordered=False)
    assert passes("t.1", e, {"a", "b", "c"})
    assert not passes("t.1", e, ["a", "b"])


def test_torch_tensor_accepted():
    torch = pytest.importorskip("torch")
    e = entry("r.5", [0.5, 0.25], decimals=2)
    assert passes("r.5", e, torch.tensor([0.5, 0.25]))
    e = entry("r.6", 2.5, decimals=1)
    assert passes("r.6", e, torch.tensor(2.5))


# ---------------------------------------------------------------- public API
def test_record_then_check_roundtrip(capsys):
    wb.record("rt.1", 0.8123, decimals=3)
    out = capsys.readouterr().out
    assert out.startswith(C.ANSWER_PREFIX)
    assert "0.812" not in out
    assert wb.check("rt.1", 0.8123)
    assert not wb.check("rt.1", 0.7)
    assert "✅" in capsys.readouterr().out


def test_record_with_one_decimal():
    # 1 decimal on a value with digits: fine; a wrong value must be rejected
    wb.record("rt.2", 12.34, decimals=1)
    assert not wb.check("rt.2", 12.6, quiet=True)


def test_pending_and_unknown(capsys):
    assert wb.check("rt.3", ...).status == "pending"
    assert wb.check("rt.3", None).status == "pending"
    assert wb.check("does.not.exist", 1).status == "unknown"
    out = capsys.readouterr().out
    assert "⏳" in out and "❓" in out


def test_check_reads_answers_file(tmp_path):
    e = entry("f.1", 17)
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"_meta": {}, "answers": {"f.1": e}}))
    assert wb.check("f.1", 17, answers_path=path, quiet=True)
    assert not wb.check("f.1", 18, answers_path=path, quiet=True)


def test_decimals_argument_mismatch_is_harmless(capsys):
    wb.record("rt.4", 2.71828, decimals=3)
    assert wb.check("rt.4", 2.71828, decimals=2)
    assert "3 décimale(s)" in capsys.readouterr().out


def test_check_result_displays_nothing():
    result = wb.check("rt.3", ..., quiet=True)
    assert result._ipython_display_() is None
    assert "pending" in repr(result)


def test_attempt_catches_only_todo(capsys):
    with wb.attempt("z.1"):
        raise NotImplementedError("todo")
    assert "⏳ Ex z.1" in capsys.readouterr().out
    with pytest.raises(ZeroDivisionError):
        with wb.attempt("z.2"):
            1 / 0


def test_attempt_catches_missing_mylearn(capsys, tmp_path):
    placeholder = wb.load_mylearn("learner", path=tmp_path / "nothing", missing_ok=True)
    with wb.attempt("z.3"):
        placeholder.metrics.f1([1], [1])
    assert "⏳ Ex z.3" in capsys.readouterr().out


# ---------------------------------------------------------------- regressions (independent review)
def test_integer_arrays_reject_probabilities():
    e = entry("g.1", [0, 1, 1, 0, 1])
    assert passes("g.1", e, np.array([0, 1, 1, 0, 1]))
    assert passes("g.1", e, [True, False, False, True, False]) is False
    assert not passes("g.1", e, [0.21, 0.74, 0.93, 0.08, 0.66])
    assert "entières" in message("g.1", e, [0.21, 0.74, 0.93, 0.08, 0.66])


def test_sets_of_floats_need_decimals():
    with pytest.raises(ValueError):
        entry("g.2", [0.1, 0.2], ordered=False)
    e = entry("g.3", [0.1, 0.2], decimals=1, ordered=False)
    assert passes("g.3", e, {0.2, 0.1}) and not passes("g.3", e, [0.3, 0.4])
    assert "ensemble" in message("g.3", e, 5)


def test_torch_non_tensors_and_special_tensors():
    torch = pytest.importorskip("torch")
    e = entry("g.4", [3, 4])
    assert passes("g.4", e, torch.zeros(3, 4).shape)  # torch.Size is a tuple
    e = entry("g.5", 2.5, decimals=1)
    assert passes("g.5", e, torch.tensor(2.5, dtype=torch.bfloat16))
    e = entry("g.6", [1.0, 2.0], decimals=1)
    grad = [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)]
    assert passes("g.6", e, grad)
    ok, status, _ = C.check_entry("g.4", entry("g.4", [3, 4]), torch.float32)
    assert not ok and status == "type"


def test_float32_array_and_rounding_boundary():
    e = entry("g.7", [0.125, 1.0], decimals=2)
    assert passes("g.7", e, np.array([0.125, 1.0], dtype=np.float32))
    assert passes("g.7", e, [0.12500001, 1.0])
    e = entry("g.8", 0.125, decimals=2)  # exactly on a boundary: both roundings accepted
    assert passes("g.8", e, 0.13) and passes("g.8", e, 0.12) and passes("g.8", e, 0.125)
    assert not passes("g.8", e, 0.14)


def test_value_rounding_to_zero_is_refused():
    with pytest.raises(ValueError):
        entry("g.9", 0.00023, decimals=3)


def test_zero_nan_and_inf_hints_do_not_leak():
    e = entry("g.10", 0.0, decimals=1)
    assert "signe" not in message("g.10", e, 0.5) and "signe" not in message("g.10", e, -0.5)
    e = entry("g.11", 2.5, decimals=1)
    assert "NaN" in message("g.11", e, float("nan"))
    assert "infinie" in message("g.11", e, float("inf"))
    assert "signe" not in message("g.11", e, 0.0)


def test_unusual_inputs_do_not_crash():
    import pandas as pd

    e = entry("g.12", 3.5, decimals=1)
    assert "trop grand" in message("g.12", e, 2**1100)
    assert not passes("g.12", e, pd.NA)
    assert passes("g.12", e, pd.Series([3.5]))  # e.g. the result of .mode()
    e = entry("g.13", 2**60)
    assert passes("g.13", e, 2**60) and not passes("g.13", e, 2**60 + 1)
    e = entry("g.14", 1000.5, decimals=1)
    assert passes("g.14", e, "1 000,5") and passes("g.14", e, "1,000.5")


def test_string_normalisation_extensions():
    e = entry("g.15", "sur-apprentissage")
    for v in ("surapprentissage", "Sur apprentissage", "sur‑apprentissage"):
        assert passes("g.15", e, v), v
    e = entry("g.16", "l'œuvre d'art")
    assert passes("g.16", e, "l’oeuvre d’art")
    e = entry("g.17", "Gentoo")
    import pandas as pd

    assert passes("g.17", e, pd.Series(["Gentoo"]))


def test_type_message_names_the_original_type():
    import pandas as pd

    e = entry("g.18", 3)
    assert "`Series`" in message("g.18", e, pd.Series([1.5, 2.5]))


def test_attempt_catches_from_import(capsys, tmp_path):
    pkg = tmp_path / "mylearn"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    wb.load_mylearn("learner", path=pkg)
    with wb.attempt("z.4"):
        from mylearn import metrics  # noqa: F401
    assert "⏳ Ex z.4" in capsys.readouterr().out
    wb.load_mylearn("ref")
