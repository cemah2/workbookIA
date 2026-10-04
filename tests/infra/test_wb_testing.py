"""Tests of wb.run_pytest: notebook-style test functions run by a real pytest process."""

import numpy as np
import pandas as pd
import pytest

import wb
from wb.testing import source_of


def scale(values):
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        raise ValueError("all values are equal")
    return (values - values.min()) / span


def scale_bug(values):
    values = np.asarray(values, dtype=float)
    return values / values.max()


def test_three_values():
    assert list(scale([2, 4, 6])) == [0.0, 0.5, 1.0]


def test_floats():
    assert scale([0.1, 0.2, 0.3])[1] == pytest.approx(0.5)


def test_constant_raises():
    with pytest.raises(ValueError):
        scale([5, 5])


@pytest.mark.parametrize("values", [[1, 2], [3, -1, 7], [0.5, 0.25]])
def test_bounds(values):
    result = scale(values)
    assert result.min() == 0 and result.max() == 1


ALL = [test_three_values, test_floats, test_constant_raises, test_bounds]


def test_run_pytest_counts_parametrized_cases(capsys):
    result = wb.run_pytest(ALL, subject=scale)
    assert (result.passed, result.failed, result.errors) == (6, 0, 0)
    assert result.ok
    assert "6 passed" in capsys.readouterr().out


def test_run_pytest_renames_the_subject_and_catches_a_bug():
    result = wb.run_pytest(ALL, subject=scale_bug, name="scale", quiet=True)
    assert result.failed >= 2 and not result.ok
    assert "scale_bug" not in source_of(scale_bug, "scale")


def test_run_pytest_reports_errors_of_missing_names():
    result = wb.run_pytest([test_three_values], quiet=True)   # `scale` is not copied: NameError
    assert result.failed == 1 and not result.ok


def share_of_a(df):
    return float((df["kind"] == "a").mean())


def test_share_with_pandas():
    df = pd.DataFrame({"kind": ["a", "b", "a", "a"]})
    assert share_of_a(df) == 0.75


def test_run_pytest_preamble_adds_module_level_imports():
    without = wb.run_pytest([test_share_with_pandas], subject=share_of_a, quiet=True)
    assert without.failed == 1                                    # NameError: pd is not imported
    result = wb.run_pytest([test_share_with_pandas], subject=share_of_a, preamble="import pandas as pd", quiet=True)
    assert result.ok and result.passed == 1


def test_source_of_keeps_decorators_and_refuses_lambdas():
    assert source_of(test_bounds).startswith("@pytest.mark.parametrize")
    with pytest.raises(ValueError):
        source_of(eval("lambda x: x"))  # noqa: S307 - a lambda without readable source


def loop_forever():
    while True:
        pass


def test_endless():
    loop_forever()


test_endless.__test__ = False   # only for run_pytest: collected here, it would never end


def test_run_pytest_stops_a_test_that_never_ends(capsys):
    result = wb.run_pytest([test_endless], subject=loop_forever, timeout=3)
    assert not result.ok and result.errors == 1 and result.summary == "timeout"
    assert "boucle infinie" in capsys.readouterr().out
