"""Tests of wb.nbtools: the tools of the check cells, on the scenarios met in chapters 12 to 20.

Each scenario is what a learner's code does wrong (a function that returns None, a tuple instead of an
array, a wrong shape, an exception, a cell run twice, code not written yet, an endless loop) and what the
check cell must print.
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

import wb
from wb.errors import MylearnMissingError
from wb.nbtools import (StopCheck, as_numpy, checked, filled, fitted, fr, letters_text, pair, print_answer, refuses,
                        returned, run_pytest_group, same, shape_text, verdict, written)

ROOT = Path(__file__).resolve().parents[2]


def out(capsys) -> str:
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# verdict, filled, fr, returned
# ---------------------------------------------------------------------------
def test_verdict_prints_one_line_and_returns_nothing(capsys):
    assert verdict("21.4", True, "bien", "mal") is None
    assert verdict("21.4", np.bool_(False), "bien", "mal") is None
    assert out(capsys) == "✅ Ex 21.4 : bien\n❌ Ex 21.4 : mal\n"


def test_filled_fr_and_returned(capsys):
    assert filled(1, "a", 0.0) and not filled(1, ...) and not filled(None)
    assert fr(0.12345) == "0,123" and fr(2, digits=1) == "2,0"
    assert returned("21.4", "f", 0) is True
    assert returned("21.4", "f", None) is False
    assert out(capsys) == "❌ Ex 21.4 : f renvoie None : as-tu oublié le return ?\n"


def test_the_module_is_reachable_from_wb():
    assert wb.nbtools.checked is checked


# ---------------------------------------------------------------------------
# checked and written
# ---------------------------------------------------------------------------
def run_checked(body, ex_id="21.4"):
    """Run body() inside checked(ex_id), then mark that the cell went on."""
    with checked(ex_id):
        body()
    print("suite")


def raise_(exc):
    def body():
        raise exc
    return body


def test_checked_code_not_written_yet_is_pending_and_the_cell_goes_on(capsys):
    run_checked(raise_(NotImplementedError("softmax_rows")))
    run_checked(raise_(MylearnMissingError("mylearn.nn.attention")))
    text = out(capsys)
    assert text.count("⏳ Ex 21.4 : pas encore fait") == 2 and text.count("suite") == 2
    assert "❌" not in text


def test_checked_an_exception_of_the_learner_becomes_one_line(capsys):
    run_checked(raise_(IndexError("index 3 is out of bounds for axis 0 with size 3")))
    assert out(capsys) == ("❌ Ex 21.4 : ton code lève IndexError : index 3 is out of bounds for axis 0 with size 3\n"
                           "suite\n")


def test_checked_an_attribute_read_on_none_hints_at_the_return(capsys):
    """Scenario 18.26: `__add__` without return, then `.backward()` on its result."""
    run_checked(raise_(AttributeError("'NoneType' object has no attribute 'backward'")))
    text = out(capsys)
    assert text.startswith("❌ Ex 21.4 : ton code lève AttributeError : 'NoneType' object has no attribute 'backward'")
    assert "oublie-t-elle son return ?" in text and text.endswith("suite\n")


def test_checked_a_type_error_on_none_gets_the_message_of_wb_attempt(capsys):
    def body():
        a, b = None                                          # a function without return, then unpacked
    run_checked(body)
    text = out(capsys)
    assert "❌ Ex 21.4 : une valeur vaut None" in text and "oublié son `return`" in text and "suite" in text


def test_checked_a_tensor_given_to_numpy_is_not_a_forgotten_return(capsys):
    """Scenario 25.28: reparameterize returns a tensor, then np.mean(...) on it (audit P5 V-VI, A10 14)."""
    torch = pytest.importorskip("torch")
    run_checked(lambda: np.mean(torch.ones(3)))
    text = out(capsys)
    assert text.startswith("❌ Ex 21.4 : ton code lève TypeError : mean() received an invalid combination")
    assert "un tenseur PyTorch" in text and "return" not in text and text.endswith("suite\n")


def test_checked_never_prints_an_empty_message(capsys):
    """Scenario 24.18: scikit-learn's message starts with a new line (audit P5 V-VI, A10 18)."""
    run_checked(raise_(ValueError("\nAll the 12 fits failed.\nIt is very likely that your model is misconfigured.")))
    run_checked(raise_(RuntimeError()))
    lines = out(capsys).splitlines()
    assert lines[0] == "❌ Ex 21.4 : ton code lève ValueError : All the 12 fits failed."
    assert "❌ Ex 21.4 : ton code lève RuntimeError : (sans message)" in lines


def test_checked_stops_quietly_on_stopcheck(capsys):
    def body():
        print("❌ Ex 21.4 : la raison")
        raise StopCheck
    run_checked(body)
    assert out(capsys) == "❌ Ex 21.4 : la raison\nsuite\n"


def test_checked_lets_an_unrelated_import_error_through():
    with pytest.raises(ImportError):
        with checked("21.4"):
            import a_module_that_does_not_exist_anywhere  # noqa: F401


def test_a_check_cell_run_twice_prints_the_same_lines(capsys):
    """Nothing is remembered between two runs of a cell."""
    def cell():
        with checked("21.4"):
            same("21.4", "f(x)", np.array([1.0, 2.0]), [1.0, 3.0])
            raise ValueError("boom")
        print("fin")
    cell()
    first = out(capsys)
    cell()
    assert out(capsys) == first and first.count("❌") == 2


def test_written():
    def ok():
        return 1
    assert written(ok)
    assert not written(raise_(NotImplementedError()))
    assert not written(raise_(MylearnMissingError("mylearn.x")))
    assert not written(raise_(ImportError("no module mylearn.x")))
    assert written(raise_(ValueError("a bug: the tests will say which")))


# ---------------------------------------------------------------------------
# same, as_numpy
# ---------------------------------------------------------------------------
def test_same_agreeing_values(capsys):
    assert same("21.4", "f(x)", [1.0, 2.0 + 1e-13], np.array([1.0, 2.0])) is True
    assert same("21.4", "g(x)", np.full((4, 6), np.nan), np.full((4, 6), np.nan)) is True
    text = out(capsys)
    assert "✅ Ex 21.4 : f(x) : [1.0, 2.0], comme attendu." in text
    assert "✅ Ex 21.4 : g(x) : un tableau (4, 6), comme attendu." in text


def test_same_a_function_that_returns_none(capsys):
    assert same("21.4", "softmax_rows", None, np.ones(3)) is False
    assert out(capsys) == "❌ Ex 21.4 : softmax_rows renvoie None : as-tu oublié le return ?\n"


def test_same_not_an_array_of_numbers(capsys):
    assert same("21.4", "f", {"a": 1}, np.ones(3)) is False
    assert same("21.4", "f", [np.ones(2), np.ones(3)], np.ones(3)) is False     # ragged
    text = out(capsys)
    assert "attendu un tableau de nombres, reçu un objet dict." in text
    assert "reçu un objet list." in text


def test_same_a_tuple_instead_of_an_array(capsys):
    """`return out, cache` where only out was expected: the message names the tuple."""
    got = (np.ones((2, 3)), np.zeros((2, 3)))
    assert same("21.4", "f(x)", got, np.ones((2, 3))) is False
    assert out(capsys) == ("❌ Ex 21.4 : f(x) : attendu la forme (2, 3), reçu un tuple de 2 valeurs "
                           "(de forme (2, 2, 3) une fois converti).\n")


def test_same_a_wrong_shape(capsys):
    assert same("21.4", "f(x)", np.ones((3, 2)), np.ones((2, 3))) is False
    assert out(capsys) == "❌ Ex 21.4 : f(x) : attendu la forme (2, 3), reçu la forme (3, 2).\n"


def test_same_small_arrays_show_both_values(capsys):
    assert same("21.4", "f(x)", [0.5, 0.25], [0.5, 0.5]) is False
    assert out(capsys) == "❌ Ex 21.4 : f(x) : attendu [0.5, 0.5], reçu [0.5, 0.25].\n"


def test_same_large_arrays_show_the_first_difference():
    """Scenario 17.15: a softmax over axis 0 instead of axis 1."""
    z = np.arange(24.0).reshape(4, 6) / 10
    want = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    got = np.exp(z) / np.exp(z).sum(axis=0, keepdims=True)
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert same("17.15", "softmax(z)", got, want) is False
    text = buffer.getvalue()
    assert text.startswith("❌ Ex 17.15 : softmax(z) : première différence à l'indice (0, 0) : attendu ")
    assert f"attendu {want[0, 0]:.8g}, reçu {got[0, 0]:.8g}" in text
    assert f"plus grand écart {np.abs(got - want).max():.3g}" in text


def test_same_large_vector_with_infinite_values(capsys):
    got = np.zeros(10)
    got[3] = np.inf
    assert same("21.4", "f", got, np.zeros(10)) is False
    text = out(capsys)
    assert "première différence à l'indice 3 : attendu 0, reçu inf (plus grand écart inf)." in text
    got[3] = np.nan
    assert same("21.4", "f", got, np.zeros(10)) is False
    assert out(capsys).endswith("première différence à l'indice 3 : attendu 0, reçu nan.\n")


def test_same_and_as_numpy_read_a_tensor():
    torch = pytest.importorskip("torch")
    t = torch.tensor([[1.0, 2.0]], requires_grad=True) * 2
    assert as_numpy(t).dtype == np.float64 and as_numpy(t).tolist() == [[2.0, 4.0]]
    assert same("20.4", "t", t, np.array([[2.0, 4.0]]))
    assert as_numpy(None) is None and as_numpy("abc") is None


def test_same_and_as_numpy_read_a_list_of_tensors(capsys):
    """Scenario: `return [h, c]` with tensors that require grad (audit P5 V-VI, A10 10)."""
    torch = pytest.importorskip("torch")
    t = torch.ones(2, requires_grad=True)
    assert as_numpy([t * 1, t * 2]).tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert as_numpy((t * 3, [4.0, 5.0])).tolist() == [[3.0, 3.0], [4.0, 5.0]]
    run_checked(lambda: same("21.4", "f", [t * 1, t * 2], np.ones((2, 2))))
    text = out(capsys)
    assert text.startswith("❌ Ex 21.4 : f : attendu [[1.0, 1.0], [1.0, 1.0]], reçu [[1.0, 1.0], [2.0, 2.0]].")
    assert "lève" not in text


def test_as_numpy_refuses_complex_numbers(capsys):
    assert as_numpy(np.array([1 + 2j, 3.0])) is None and as_numpy([1j]) is None
    assert same("21.4", "fft", np.array([1.0 + 0j, 2.0 + 1j]), [1.0, 2.0]) is False
    assert "attendu un tableau de nombres, reçu un objet ndarray" in out(capsys)


@pytest.mark.parametrize("make", ["numpy", "torch"])
def test_same_says_when_a_float32_result_is_right_to_single_precision(capsys, make):
    """Scenario 21.15: conv2d computed in float32, values that print like the expected ones (A10 11)."""
    want = np.array([-11.850715, 0.25, 3.0])
    if make == "torch":
        torch = pytest.importorskip("torch")
        got = torch.tensor(want, dtype=torch.float32)
    else:
        got = want.astype(np.float32)
    assert same("21.15", "conv2d", got, want) is False
    text = out(capsys)
    assert "ton résultat est en float32, la précision attendue est celle du float64" in text
    assert same("21.15", "conv2d", np.float32([1.0, 2.0]), [1.0, 2.5]) is False      # really wrong: no hint
    assert "float32" not in out(capsys)


# ---------------------------------------------------------------------------
# pair, refuses, fitted, shape_text, print_answer
# ---------------------------------------------------------------------------
def test_pair_unpacks_a_good_result():
    out_, mask = pair("20.16", "dropout_forward", (np.ones(2), np.zeros(2)))
    assert out_.tolist() == [1.0, 1.0] and mask.tolist() == [0.0, 0.0]
    loss, grad = pair("18.12", "mse_loss", [np.float32(0.5), np.ones(2)], names=("loss", "grad"), floats=("loss",))
    assert type(loss) is float and loss == 0.5


def test_pair_an_array_of_two_rows_is_not_unpacked(capsys):
    """Scenario 20.16: dropout_forward returns only out, of shape (2, 4)."""
    run_checked(lambda: pair("20.16", "dropout_forward", np.ones((2, 4))), ex_id="20.16")
    assert out(capsys) == ("❌ Ex 20.16 : dropout_forward doit rendre le couple (out, mask), reçu un ndarray de "
                           "forme (2, 4).\nsuite\n")


@pytest.mark.parametrize("result, names, expected", [
    (None, ("out", "mask"), "dropout_forward renvoie None : as-tu oublié le return ?"),
    ((1, 2), ("dx", "dgamma", "dbeta"), "doit rendre le triplet (dx, dgamma, dbeta), reçu un tuple de 2 valeurs."),
    (0.5, ("loss", "grad"), "doit rendre le couple (loss, grad), reçu un float."),
], ids=["none", "two-values-for-a-triplet", "a-float-for-a-couple"])
def test_pair_wrong_results_stop_the_check(capsys, result, names, expected):
    with pytest.raises(StopCheck):
        pair("20.16", "dropout_forward", result, names=names)
    assert expected in out(capsys)


def test_pair_a_loss_that_is_not_a_float(capsys):
    with pytest.raises(StopCheck):
        pair("18.12", "mse_loss", (np.array([0.5]), np.ones(2)), names=("loss", "grad"), floats=("loss",))
    assert out(capsys) == ("❌ Ex 18.12 : mse_loss : loss doit être un float Python, reçu ndarray "
                           "(convertis avec float(...)).\n")


def test_refuses(capsys):
    def bad_p():
        raise ValueError("p must be in [0, 1), got 1.5")
    def other():
        raise TypeError("unsupported operand")
    assert refuses("20.16", "p = 1,5", bad_p) is True
    assert refuses("20.16", "p = 1,5", lambda: None) is False
    assert refuses("20.16", "p = 1,5", other) is False
    text = out(capsys)
    assert "✅ Ex 20.16 : p = 1,5 : ValueError levée (p must be in [0, 1), got 1.5)." in text
    assert "❌ Ex 20.16 : p = 1,5 : attendu ValueError, mais rien n'a été levé." in text
    assert "❌ Ex 20.16 : p = 1,5 : attendu ValueError, reçu TypeError : unsupported operand" in text
    with pytest.raises(NotImplementedError):
        refuses("20.16", "p", raise_(NotImplementedError()))
    with pytest.raises(NotImplementedError):         # a stub is not "the RuntimeError we expected" (A10 19)
        refuses("20.16", "p", raise_(NotImplementedError()), error=RuntimeError)
    with pytest.raises(MylearnMissingError):         # nor "the AttributeError we expected"
        refuses("20.16", "p", raise_(MylearnMissingError("mylearn.x")), error=AttributeError)
    run_checked(lambda: refuses("20.16", "p", raise_(NotImplementedError()), error=RuntimeError), ex_id="20.16")
    assert out(capsys) == "⏳ Ex 20.16 : pas encore fait.\nsuite\n"


def test_fitted_survives_a_fit_without_return_self():
    class NoReturn:
        def fit(self, X, y=None, sample_weight=None):
            self.seen_ = (len(X), y is not None, sample_weight)
    model = NoReturn()
    assert fitted(model, [1, 2, 3]) is model and model.seen_ == (3, False, None)
    assert fitted(model, [1, 2], [0, 1], sample_weight=[1, 1]).seen_ == (2, True, [1, 1])


@pytest.mark.parametrize("value, expected", [
    ("[2, 3]", "(2, 3)"), ("2, 3", "(2, 3)"), ("torch.Size([2, 3])", "(2, 3)"), ((2, 3), "(2, 3)"),
    ([np.int64(4)], "(4,)"), ("(5,)", "(5,)"), ("deux", "deux"), ((2.0, 3), (2.0, 3)), ((True,), (True,)),
    ("8 x 16 x 28", "(8, 16, 28)"), ("8×16", "(8, 16)"), ("(2; 3)", "(2, 3)"), ("2 3", "(2, 3)"),
    (np.array([4, 7]), "(4, 7)"), ("erreur", "erreur"), ("2.5, 3", "2.5, 3"),
], ids=["list-text", "bare-text", "torch-size-text", "tuple", "numpy-int", "one-dim", "word", "float", "bool",
        "times-x", "times-sign", "semicolon", "spaces", "numpy-array", "error-word", "decimal-text"])
def test_shape_text(value, expected):
    assert shape_text(value) == expected


@pytest.mark.parametrize("value, sort, expected", [
    ("ACE", False, "ACE"), ("a, c et e", False, "ACE"), ("(D) (B)", False, "DB"), ("(D) (B)", True, "BD"),
    (["F", "b"], True, "BF"), ("I ou F", False, "IF"), (..., False, ...), (None, True, None),
], ids=["plain", "words", "parentheses", "sorted", "list", "ou", "ellipsis", "none"])
def test_letters_text(value, sort, expected):
    assert letters_text(value, sort=sort) == expected


def test_shape_text_reads_a_torch_size():
    torch = pytest.importorskip("torch")
    assert shape_text(torch.zeros(2, 3).shape) == "(2, 3)"


def test_print_answer(capsys):
    print_answer("21.4", np.array([1 / 3, 2.0]))
    print_answer("21.5", ["a", "b"])
    print_answer("21.6", 0.5)
    assert out(capsys) == "21.4: [0.3333, 2.0]\n21.5: ['a', 'b']\n21.6: 0.5\n"


# ---------------------------------------------------------------------------
# run_pytest_group
# ---------------------------------------------------------------------------
GROUP_TESTS = '''
import os
import time
import pytest


@pytest.fixture
def broken():
    raise RuntimeError("fixture down")


def test_passes():
    assert os.environ["COLUMNS"] == "1000"


def test_value_is_wrong():
    assert 3 == 4, "expected 3 rows, got 4: " + "x" * 300 + " END"


def test_reads_an_attribute_on_none():
    None.transform


def test_fit_returns_self():
    assert False, "fit must return self"


def test_f_does_not_modify_its_input():
    assert False, "expected x unchanged by f"


def test_not_written():
    raise NotImplementedError("Ex 21.4 : softmax_rows")


def test_uses_a_broken_fixture(broken):
    pass


def test_slow():
    time.sleep(60)
'''

GROUP_CONFTEST = '''
import pytest


class _Pending:
    def __init__(self, text):
        self.text = text
        self.reprcrash = self

    @property
    def message(self):
        return self.text

    def toterminal(self, tw):
        tw.line(self.text)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.longrepr = _Pending(f"⏳ pas encore implémenté : {call.excinfo.value}")
'''


@pytest.fixture
def group_root(tmp_path):
    """A tiny repository: tests/test_group.py and a conftest that marks NotImplementedError with ⏳ (as ours)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_group.py").write_text(textwrap.dedent(GROUP_TESTS), encoding="utf-8")
    (tmp_path / "tests" / "conftest.py").write_text(textwrap.dedent(GROUP_CONFTEST), encoding="utf-8")
    return tmp_path


def test_run_pytest_group_prints_each_failure_on_two_lines(group_root, capsys):
    result = run_pytest_group("tests/test_group.py", "not slow", root=group_root)
    text = out(capsys)
    lines = text.splitlines()
    assert (result.passed, result.failed, result.errors, result.pending) == (1, 5, 1, 1)
    assert not result.ok
    assert "💡 un attribut est lu sur None" in lines[0]
    # the clearest failures first (an attribute on None, return self, an input modified), then the others,
    # the ⏳ at the end
    order = [line.split()[1] for line in lines if line[:1] in "❌⏳"]
    assert order == ["test_reads_an_attribute_on_none", "test_fit_returns_self", "test_f_does_not_modify_its_input",
                     "test_value_is_wrong", "test_uses_a_broken_fixture", "test_not_written"]
    i = lines.index("❌ test_value_is_wrong")
    assert lines[i + 1].startswith("   AssertionError: expected 3 rows, got 4: xxx") and lines[i + 1].endswith(" END")
    assert "❌ test_uses_a_broken_fixture (erreur hors du test : import, fixture…)" in lines
    i = lines.index("⏳ test_not_written")
    assert lines[i + 1] == "   pas encore implémenté : Ex 21.4 : softmax_rows"
    assert lines[-1].startswith("pytest: 5 failed, 1 passed, 1 deselected, 1 error")


def test_run_pytest_group_hides_what_exceeds_max_shown(group_root, capsys):
    run_pytest_group("tests/test_group.py", "not slow and not passes", root=group_root, max_shown=2)
    text = out(capsys)
    assert sum(line[:1] in "❌⏳" for line in text.splitlines()) == 2
    assert "   … et 4 autre(s) test(s) en échec, dont 1 pas encore écrit(s)" in text


def test_run_pytest_group_all_green(group_root, capsys):
    result = run_pytest_group(group_root / "tests" / "test_group.py", "passes", root=group_root)
    assert result.ok and result.passed == 1
    assert out(capsys).startswith("pytest: 1 passed, 7 deselected")


def test_run_pytest_group_stops_an_endless_loop(group_root, capsys):
    result = run_pytest_group("tests/test_group.py", "slow", root=group_root, timeout=3)
    assert result.timed_out and not result.ok
    assert out(capsys) == ("❌ les tests n'ont pas fini en 3 s : une boucle de ton code ne s'arrête-t-elle pas ?\n")


def test_run_pytest_group_shows_a_module_that_does_not_import(tmp_path, capsys):
    """The conftest stops pytest with one line (pytest.exit): that line, then a readable summary."""
    (tmp_path / "test_import.py").write_text(textwrap.dedent('''
        import pytest

        def test_a():
            pytest.exit("❌ mylearn/x.py ne s'importe pas : SyntaxError ligne 3.", returncode=1)
    '''), encoding="utf-8")
    run_pytest_group("test_import.py", root=tmp_path)
    lines = out(capsys).splitlines()
    assert lines[0] == "❌ mylearn/x.py ne s'importe pas : SyntaxError ligne 3."
    assert len(lines) == 2 and lines[1].startswith("pytest: ") and "!!!" not in lines[1]


@pytest.mark.parametrize("test_file, error", [
    ("tests", ValueError), ("tests/", ValueError), ("tests/conftest.py", ValueError),
    ("tests/test_missing.py", FileNotFoundError),
], ids=["folder", "folder-slash", "not-a-test-file", "missing-file"])
def test_run_pytest_group_needs_one_explicit_test_file(group_root, test_file, error):
    with pytest.raises(error):
        run_pytest_group(test_file, root=group_root)


def test_run_pytest_group_refuses_an_unknown_impl(group_root):
    with pytest.raises(ValueError, match="impl"):
        run_pytest_group("tests/test_group.py", impl="mine", root=group_root)


def test_run_pytest_group_on_the_stubs_and_the_reference(capsys):
    """The real conftest: the empty skeletons are ⏳ (never ❌), the reference is green."""
    keyword = "test_dropout_forward_docstring_example"
    stubs = run_pytest_group("tests/test_ch20_regularization.py", keyword, impl="stubs", root=ROOT)
    text = out(capsys)
    assert stubs.failed == stubs.pending == 1 and "❌" not in text
    assert text.splitlines()[0] == "⏳ test_dropout_forward_docstring_example"
    assert text.splitlines()[1].startswith("   pas encore implémenté : ")
    ref = run_pytest_group("tests/test_ch20_regularization.py", keyword, impl="ref", root=ROOT)
    assert ref.ok and out(capsys).startswith("pytest: 1 passed")
