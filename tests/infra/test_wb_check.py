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



def test_int_non_integer_number_is_a_wrong_value_not_a_wrong_type():
    e = entry("d.4b", 0)
    assert passes("d.4b", e, 0.0)
    ok, status, text = C.check_entry("d.4b", e, 0.5)
    assert not ok and status == "wrong" and "entier" in text
    assert C.check_entry("d.4b", e, "abc")[1] == "type"


@pytest.mark.parametrize(
    "value, fragment, factor_claimed",
    [
        (11, "trop grande", False),     # 9 -> 11: one power of ten apart, but only 1.2 times too big
        (100, "trop grande", True),     # two powers of ten apart: at least a factor 10
        (1, "ordre de grandeur est bon", False),
    ],
)
def test_magnitude_messages_claim_a_factor_10_only_when_true(value, fragment, factor_claimed):
    e = entry("d.4c", 9)
    text = message("d.4c", e, value)
    assert fragment in text
    assert ("facteur 10" in text) is factor_claimed


def test_magnitude_messages_for_too_small_values():
    e = entry("d.4d", 12.0, decimals=1)
    assert "facteur 10" not in message("d.4d", e, 9.0) and "trop petite" in message("d.4d", e, 9.0)
    assert "facteur 10" in message("d.4d", e, 0.5)


def test_integer_arrays_accept_integers_computed_in_floating_point():
    e = entry("d.4e", [-1, 3])
    assert passes("d.4e", e, [-0.9999999999999998, 3.0000000000000004])   # np.roots, a grid read-out...
    ok, status, text = C.check_entry("d.4e", e, [-0.9, 3.2])
    assert not ok and "entières" in text and "round" in text

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


def test_record_returns_a_dict_that_displays_nothing(capsys):
    # the last line of an `answer` cell must not echo the entry (hash, mistakes) in the notebook
    recorded = wb.record("rt.5", 3.14159, decimals=2)
    capsys.readouterr()
    assert isinstance(recorded, dict) and "hash" in json.dumps(recorded)
    assert recorded._ipython_display_() is None
    assert capsys.readouterr().out == ""


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
    e = entry("g.16", -0.25, decimals=2)
    assert passes("g.16", e, "\u22120,25") and passes("g.16", e, "-0,25")  # typographic minus sign


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


def test_factor_100_messages_do_not_claim_which_one_is_the_percentage():
    e = entry("d.8", 0.95, decimals=2)          # e.g. an error rate of 0.95 %
    big = message("d.8", e, 95)                 # 95 errors, not a percentage at all
    assert "100 fois trop grande" in big and "pourcentage" in big and "proportion" in big
    e = entry("d.9", 4000)                      # 4000 euros: not a proportion
    assert "100 fois trop grande" in message("d.9", e, 400_000)
    assert "100 fois trop petite" in message("d.9", e, 40)


def test_integer_answer_given_unrounded_is_told_to_round_or_gets_its_mistake():
    e = entry("d.10", 11185, mistakes={"arrondi trop tôt": 11280, "7 intervalles, pas 8": 44})
    text = message("d.10", e, 11185.448)
    assert "arrondis" in text and "11185" not in text
    assert "7 intervalles" in C.check_entry("d.10", e, 43.75)[2]       # round(43.75) = 44, a declared mistake
    assert "arrondi trop tôt" in C.check_entry("d.10", e, 11279.6)[2]
    assert "entier" in message("d.10", e, 12.5)


def test_a_date_is_a_wrong_type_with_a_useful_message():
    import pandas as pd

    e = entry("d.11", 1778)
    ok, status, text = C.check_entry("d.11", e, pd.Timestamp("1778-05-01"))
    assert not ok and status == "type" and ".year" in text



def test_factor_100_is_not_claimed_when_the_answer_has_one_significant_digit():
    e = entry("d.12", 0.00926, decimals=2)      # stored as "0.01": 0.6 / 100 also rounds to 0.01
    for wrong in (0.6, 0.99, 1.0):
        assert "100 fois" not in message("d.12", e, wrong), wrong
    e = entry("d.13", 0.684, decimals=3)        # 3 significant digits: 68.4 is a percentage
    assert "100 fois trop grande" in message("d.13", e, 68.4)


def test_array_of_percentages_gets_the_factor_100_message():
    e = entry("d.14", [0.68391, 0.9546, 0.99708], decimals=3)
    text = message("d.14", e, [68.391, 95.46, 99.708])
    assert "100 fois trop grandes" in text and "0,684" not in text
    assert "100 fois trop petites" in message("d.14", e, [0.0068391, 0.009546, 0.0099708])
    small = entry("d.15", [0.01, 0.02], decimals=2)   # one significant digit: no claim
    assert "100 fois" not in message("d.15", small, [1.3, 1.7])
    counts = entry("d.16", [3, 5, 9, 0])            # integers: no percentage message
    assert "100 fois" not in message("d.16", counts, [300, 500, 900, 0])


def test_almost_only_one_unit_away_or_too_few_decimals():
    e = entry("d.17", 0.3, decimals=1)                 # one significant digit: 0.2 or 0.1 are far off
    for wrong in (0.1, 0.0, 0.2, 0.4, 0.45, -0.2):
        assert "presque" not in message("d.17", e, wrong), wrong
    e = entry("d.18", 0.86840, decimals=3)             # stored as "0.868"
    for early in (0.869, 0.867, 0.86865):              # a mean of rounded values, a truncation
        assert "presque" in message("d.18", e, early), early
    assert "presque" not in message("d.18", e, 0.865)  # three units away: a wrong value
    too_short = message("d.18", e, 0.87)               # right at 2 decimals, but 3 are asked
    assert "presque" in too_short and "3" in too_short
    e = entry("d.19", 0.65, decimals=2)                # 2 significant digits: 0.64 may well be another measure
    for wrong in (0.60, 0.64, 0.66):
        assert "presque" not in message("d.19", e, wrong), wrong
    e = entry("d.20", 3.2, decimals=1)
    assert "presque" not in message("d.20", e, 3.3) and "presque" not in message("d.20", e, 3.0)
    e = entry("d.25", 3.25, decimals=2)
    assert "presque" in message("d.25", e, 3.26)
    e = entry("d.26", 0.849, decimals=3)               # too few decimals is said first, even one unit away
    assert "juste à 2 décimale" in message("d.26", e, 0.85)


def test_fractions_written_as_text_are_numbers_and_percent_signs_get_a_message():
    e = entry("d.21", 41 / 76, decimals=3)
    assert passes("d.21", e, "41/76") and passes("d.21", e, "41 / 76")
    assert not passes("d.21", e, "76/41")
    e = entry("d.22", 3)
    assert passes("d.22", e, "6/2")
    ok, status, text = C.check_entry("d.23", entry("d.23", 0.3, decimals=2), "30 %")
    assert not ok and status == "type" and "sans le signe %" in text and "0.3" not in text
    assert C.check_entry("d.23", entry("d.23", 0.3, decimals=2), "1/0")[1] == "type"


def test_arrays_accept_numbers_written_as_text_and_hint_at_rounding():
    e = entry("d.24", [41 / 48, 13 / 18, 33 / 34], decimals=3)
    assert passes("d.24", e, ["0,854", "0,722", "0,971"])
    assert passes("d.24", e, np.array(["0.854", "0.722", "0.971"]))
    assert passes("d.24", e, ["41/48", "13/18", "33/34"])
    assert "presque" in message("d.24", e, [0.85, 0.72, 0.97])        # too few decimals
    assert "presque" in message("d.24", e, [0.854, 0.723, 0.971])     # one unit away
    far = message("d.24", e, [0.854, 0.65, 0.971])
    assert "presque" not in far and "2 élément(s) sur 3" in far
    assert "n'est pas un nombre" in message("d.24", e, ["0,854", "abc", "0,971"])


# ---------------------------------------------------------------- session 11 (chapter 3, notebook)
def test_mistake_keys_must_be_messages_not_values():
    """{wrong value: message} (inverted) would store the wrong choice in clear in answers.json."""
    with pytest.raises(ValueError, match="inverted"):
        entry("m.1", "B", mistakes={"A": "compare les precisions parmi 300 alertes"})
    e = entry("m.1", "B", mistakes={"compare les precisions parmi 300 alertes": "A"})
    assert "compare les precisions" in message("m.1", e, "A")
    assert "A" not in e["mistakes"].values()          # the wrong choice itself is never stored in clear


def test_transposed_square_array_is_diagnosed():
    matrix = [[146, 3, 2], [6, 61, 1], [14, 19, 90]]
    e = entry("m.2", matrix)
    assert "transposé" in message("m.2", e, np.array(matrix).T)
    symmetric = [[1, 2], [2, 5]]
    e_sym = entry("m.3", symmetric)
    assert passes("m.3", e_sym, np.array(symmetric).T)   # a symmetric table equals its transpose: right answer


def test_computed_values_get_no_rounding_advice():
    """A value computed by the learner's function is never a rounding slip."""
    e = entry("m.4", 0.9956, decimals=3)
    typed = message("m.4", e, 0.995)
    computed = C.check_entry("m.4", e, 0.995, computed=True)[2]
    assert "presque" in typed.lower()
    assert "trop tôt" not in computed and "décimale" not in computed and "ta fonction" in computed.lower()
    arr_e = entry("m.5", [0.794, 0.0122, 0.9956, 0.9803], decimals=3)
    wrong = [0.794, 0.012, 0.995, 0.980]
    assert "presque" in message("m.5", arr_e, wrong).lower()
    assert "ta fonction" in C.check_entry("m.5", arr_e, wrong, computed=True)[2].lower()


def test_record_warns_when_an_array_element_is_near_a_rounding_boundary(capsys):
    C.record("m.6", [0.8269, 0.914499], decimals=3)
    assert "limite d'arrondi" in capsys.readouterr().out
