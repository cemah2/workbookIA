"""Tools of the check cells of the exercise notebooks (chapter 21 on).

Until chapter 20, each chapter copied these helpers into its "Tools" cell, and the
copies drifted apart (BIBLE §22, audit P5 R3). One import replaces them::

    from wb.nbtools import checked, written, same, verdict, returned, pair, StopCheck, run_pytest_group

    with checked("21.4"):                                   # ⏳ / one ❌ line, the cell goes on
        out = mylearn.nn.attention.softmax_rows(scores)
        if returned("21.4", "softmax_rows", out):
            same("21.4", "softmax_rows(scores)", out, expected)
    if written(lambda: mylearn.nn.attention.softmax_rows(np.ones((1, 2)))):
        run_pytest_group("tests/test_ch21_attention.py", "test_softmax_rows_")
    else:
        print("⏳ Ex 21.4 : les tests attendent ta fonction.")

Every message for the learner is in French and starts with ✅, ❌ or ⏳ followed by
``Ex <id>``. Nothing here imports PyTorch: a tensor is recognised only when the
notebook has already imported ``torch``.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wb.checker import attempt
from wb.errors import MylearnMissingError

__all__ = [
    "verdict",
    "filled",
    "fr",
    "returned",
    "StopCheck",
    "checked",
    "written",
    "as_numpy",
    "same",
    "pair",
    "refuses",
    "fitted",
    "shape_text",
    "print_answer",
    "GroupResult",
    "run_pytest_group",
]

# Errors that mean "the learner's code is not written yet": ⏳, never ❌.
_NOT_WRITTEN = (NotImplementedError, ImportError, MylearnMissingError)
_NONE_HINT = " ; une fonction ou une méthode oublie-t-elle son return ?"


def _torch_tensor_type():
    """``torch.Tensor`` if the notebook has imported torch, else None (never imports it)."""
    torch = sys.modules.get("torch")
    return getattr(torch, "Tensor", None) if torch is not None else None


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def verdict(ex_id: str, ok, success: str, failure: str) -> None:
    """Print ``✅ Ex <id> : success`` or ``❌ Ex <id> : failure``.

    It returns None (as the copies of chapters 12 to 20), so that a call on the last line of a cell displays
    nothing more; :func:`same` returns its own verdict.
    """
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")


def filled(*values) -> bool:
    """True when none of the values is still ``...`` (or None): the answer has been written."""
    return all(value is not ... and value is not None for value in values)


def fr(x, digits: int = 3) -> str:
    """A number written the French way, with a decimal comma (for the messages in French)."""
    return f"{x:.{digits}f}".replace(".", ",")


def returned(ex_id: str, name: str, value) -> bool:
    """False, with a ❌ line, when the function ``name`` returned None (a forgotten return); True otherwise."""
    if value is None:
        print(f"❌ Ex {ex_id} : {name} renvoie None : as-tu oublié le return ?")
        return False
    return True


# ---------------------------------------------------------------------------
# Protected blocks
# ---------------------------------------------------------------------------
class StopCheck(Exception):
    """Raised once a ❌ line is printed (by :func:`pair`) when the rest of the check would make no sense.

    :func:`checked` stops the block quietly on it.
    """


@contextlib.contextmanager
def checked(ex_id: str):
    """Like ``wb.attempt`` (⏳ while the code is not written yet), but any other error of the learner's code
    becomes one ❌ line instead of stopping the cell, so that what follows (the tests) can say more.

    - ``NotImplementedError``, a missing mylearn module: ⏳ (``wb.attempt``);
    - ``TypeError`` about ``NoneType``: the forgotten-return message of ``wb.attempt``;
    - :class:`StopCheck`: the reason is already printed, the block stops silently;
    - any other exception: ``❌ Ex <id> : ton code lève <type> : <message>``, with a hint about a
      forgotten ``return`` when the message mentions ``NoneType`` (an ``AttributeError`` on None).
    """
    with attempt(ex_id):
        try:
            yield
        except _NOT_WRITTEN:
            raise
        except StopCheck:
            pass
        except Exception as exc:                      # a bug in the learner's code: report it, go on
            if isinstance(exc, TypeError) and "NoneType" in str(exc):
                raise                                 # wb.attempt explains the forgotten return
            hint = _NONE_HINT if "NoneType" in str(exc) else ""
            print(f"❌ Ex {ex_id} : ton code lève {type(exc).__name__} : {str(exc)[:300]}{hint}")


def written(probe) -> bool:
    """False while ``probe()`` stops on code not written yet (``NotImplementedError``, no mylearn module);
    True otherwise, even when it raises another error (the tests then say which)."""
    try:
        probe()
    except _NOT_WRITTEN:
        return False
    except Exception:
        return True
    return True


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------
def as_numpy(value):
    """A float64 NumPy array from a tensor (detached, on the CPU), an array, a list or a number; None if
    impossible (None itself, a dict, a ragged list...)."""
    if value is None:
        return None
    tensor = _torch_tensor_type()
    if tensor is not None and isinstance(value, tensor):
        return value.detach().cpu().double().numpy()
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _describe(value, arr) -> str:
    """How a value of the wrong shape is described: a tuple is named as such (a `return a, b`)."""
    if isinstance(value, tuple):
        return f"un tuple de {len(value)} valeurs (de forme {arr.shape} une fois converti)"
    return f"la forme {arr.shape}"


def same(ex_id: str, label: str, got, want, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    """✅ if ``got`` (array, list or tensor) equals ``want`` up to rounding (``np.allclose``, NaN equal to NaN),
    ❌ otherwise with what differs; returns the verdict.

    The ❌ line says, in this order: None (a forgotten return), not an array of numbers, a wrong shape
    (a tuple is named), then the values: both of them when ``want`` has at most 6 values, otherwise the
    first index that differs and the largest gap.
    """
    if got is None:
        print(f"❌ Ex {ex_id} : {label} renvoie None : as-tu oublié le return ?")
        return False
    arr = as_numpy(got)
    if arr is None:
        print(f"❌ Ex {ex_id} : {label} : attendu un tableau de nombres, reçu un objet {type(got).__name__}.")
        return False
    want = as_numpy(want)
    if arr.shape != want.shape:
        print(f"❌ Ex {ex_id} : {label} : attendu la forme {want.shape}, reçu {_describe(got, arr)}.")
        return False
    close = np.isclose(arr, want, rtol=rtol, atol=atol, equal_nan=True)
    ok = bool(close.all())
    shown = np.round(arr, 8).tolist() if arr.size <= 6 else f"un tableau {arr.shape}"
    if ok or want.size <= 6:
        failure = f"{label} : attendu {np.round(want, 8).tolist()}, reçu {shown}."
    else:
        bad = tuple(int(i) for i in np.argwhere(~close)[0])
        at = bad if arr.ndim > 1 else bad[0]
        with np.errstate(invalid="ignore", over="ignore"):
            gaps = np.abs(arr - want)[~close]
        gaps = gaps[~np.isnan(gaps)]                  # a NaN against a number has no size of gap
        largest = f" (plus grand écart {gaps.max():.3g})" if gaps.size else ""
        failure = (f"{label} : première différence à l'indice {at} : attendu {float(want[bad]):.8g}, reçu "
                   f"{float(arr[bad]):.8g}{largest}.")
    verdict(ex_id, ok, f"{label} : {shown}, comme attendu.", failure)
    return ok


def _count_word(n: int) -> str:
    return {2: "couple", 3: "triplet"}.get(n, f"tuple de {n} valeurs")


def pair(ex_id: str, label: str, result, names=("out", "mask"), floats=()):
    """The values of a result that must be a tuple ``(out, mask)``, ``(loss, grad)``, ``(dx, dgamma, dbeta)``...

    Otherwise one ❌ line, then :class:`StopCheck` (caught by :func:`checked`), instead of unpacking an array
    of two rows by mistake. ``floats`` lists the names that must be real numbers (a Python float or a NumPy
    floating scalar): they are returned as Python floats.

    >>> with checked("20.16"):
    ...     out, mask = pair("20.16", "dropout_forward", R.dropout_forward(x, p=0.5))
    """
    names = tuple(names)
    if result is None:
        print(f"❌ Ex {ex_id} : {label} renvoie None : as-tu oublié le return ?")
        raise StopCheck
    if not isinstance(result, (tuple, list)) or len(result) != len(names):
        if isinstance(result, (tuple, list)):
            what = f"un {type(result).__name__} de {len(result)} valeurs"
        else:
            what = f"un {type(result).__name__}"
            if hasattr(result, "shape"):
                what += f" de forme {tuple(np.shape(result))}"
        print(f"❌ Ex {ex_id} : {label} doit rendre le {_count_word(len(names))} ({', '.join(names)}), "
              f"reçu {what}.")
        raise StopCheck
    values = list(result)
    for i, name in enumerate(names):
        if name not in floats:
            continue
        value = values[i]
        if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
            print(f"❌ Ex {ex_id} : {label} : {name} doit être un float Python, reçu {type(value).__name__} "
                  "(convertis avec float(...)).")
            raise StopCheck
        values[i] = float(value)
    return tuple(values)


def refuses(ex_id: str, label: str, call, error=ValueError) -> bool:
    """✅ if ``call()`` raises ``error``, ❌ otherwise (another error, or nothing); code not written yet
    (``NotImplementedError``, no mylearn module) goes up to :func:`checked`, which prints ⏳."""
    try:
        call()
    except error as exc:
        print(f"✅ Ex {ex_id} : {label} : {error.__name__} levée ({str(exc)[:120]}).")
        return True
    except _NOT_WRITTEN:
        raise
    except Exception as exc:
        print(f"❌ Ex {ex_id} : {label} : attendu {error.__name__}, reçu {type(exc).__name__} : {str(exc)[:160]}")
        return False
    print(f"❌ Ex {ex_id} : {label} : attendu {error.__name__}, mais rien n'a été levé.")
    return False


def fitted(model, X, y=None, **fit_params):
    """The model after ``model.fit(X)`` or ``model.fit(X, y)``: no chained call, so that a ``fit`` that forgets
    ``return self`` (the tests report it) does not stop the check cells."""
    if y is None:
        model.fit(X, **fit_params)
    else:
        model.fit(X, y, **fit_params)
    return model


_SHAPE_TEXT = re.compile(r"\s*(torch\.Size)?[\s(\[]*\d+(\s*,\s*\d+)*\s*,?[\s)\]]*\s*")


def shape_text(value):
    """A shape written as ``"[2, 3]"``, ``"2, 3"``, ``"torch.Size([2, 3])"``, a tuple, a list or a
    ``torch.Size``, rewritten as the string ``"(2, 3)"``; any other value is returned unchanged."""
    if isinstance(value, (tuple, list)) and value and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                                                          for v in value):
        return str(tuple(int(v) for v in value))
    if isinstance(value, str) and _SHAPE_TEXT.fullmatch(value):
        return str(tuple(int(n) for n in re.findall(r"\d+", value)))
    return value


def print_answer(ex_id: str, value, **_) -> None:
    """Solutions notebook: show the value that the exercise notebook checks with ``wb.check``."""
    shown = value
    if isinstance(value, (np.ndarray, list, tuple)):
        try:
            shown = np.round(np.asarray(value, dtype=float), 4).tolist()
        except (TypeError, ValueError):   # a list of names
            shown = value
    print(f"{ex_id}:", shown)


# ---------------------------------------------------------------------------
# Running a group of tests of a mylearn module
# ---------------------------------------------------------------------------
@dataclass
class GroupResult:
    """Counts of :func:`run_pytest_group` (``pending``: failures on code not written yet, among ``failed``)."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    pending: int = 0
    summary: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True when at least one test ran and none failed."""
        return self.passed > 0 and self.failed == 0 and self.errors == 0 and not self.timed_out

    def _ipython_display_(self) -> None:   # the lines are already printed: a last-line call shows nothing more
        pass


_PENDING_MARK = "⏳"
_NONE_ATTRIBUTE = "'NoneType' object has no attribute"
_SUMMARY = re.compile(r"\d+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b|no tests ran")


def _minutes(seconds: float) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def _count(summary: str, word: str) -> int:
    return sum(int(n) for n in re.findall(rf"(\d+) {word}", summary))


def _test_path(test_file, root: Path) -> str:
    """The test file, relative to ``root`` when possible; a directory (``tests/``) is refused."""
    path = Path(test_file)
    full = path if path.is_absolute() else root / path
    if full.is_dir() or not path.name.startswith("test_") or path.suffix != ".py":
        raise ValueError(f"run_pytest_group needs one explicit test file (tests/test_chXX_<module>.py), "
                         f"not {str(test_file)!r}")
    if not full.is_file():
        raise FileNotFoundError(f"no test file {full}")
    try:
        return full.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(full)


def run_pytest_group(test_file, keyword: str | None = None, impl: str = "learner", timeout: float = 300,
                     *, max_shown: int = 8, root=None) -> GroupResult:
    """Run the tests of ``test_file`` selected by ``keyword`` (``pytest -k``), on the learner's code by default.

    Parameters
    ----------
    test_file : str or Path
        One explicit test file, relative to the repository (``"tests/test_ch21_attention.py"``); never the
        ``tests/`` folder.
    keyword : str, optional
        The ``-k`` expression (``"test_softmax_rows_ or test_mask_"``); None runs the whole file.
    impl : {"learner", "ref", "stubs"}
        Passed as ``--impl`` (``"ref"`` in the solutions notebook).
    timeout : float
        Seconds before the run is stopped (an endless loop in the learner's code) with one ❌ line.
    max_shown : int
        At most this many failures are printed.
    root : path, optional
        The repository (default: ``wb.repo_root()``, which reads ``WB_ROOT``).

    Each failure is printed on two lines: ``❌ <test name>`` (``⏳`` when the function is not written yet),
    then its reason (the first line of pytest's message, read with ``COLUMNS=1000``). Real failures come
    first, the tests of ``return self`` and of copies before the others, then the ⏳ ones; a 💡 line points
    to a forgotten ``return`` when an attribute is read on None; the last line is pytest's summary.
    """
    if impl not in ("learner", "ref", "stubs"):
        raise ValueError(f"impl must be 'learner', 'ref' or 'stubs', got {impl!r}")
    if root is None:
        from wb.core import repo_root
        root = repo_root()
    root = Path(root)
    rel = _test_path(test_file, root)
    command = [sys.executable, "-m", "pytest", rel, "-q", "-p", "no:cacheprovider", "--color=no", "-rfE",
               "--tb=no"]
    if keyword:
        command += ["-k", keyword]
    if impl != "learner":
        command.append(f"--impl={impl}")
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout,
                                env={**os.environ, "COLUMNS": "1000"})   # long lines: the reason of each failure
    except subprocess.TimeoutExpired:
        print(f"❌ les tests n'ont pas fini en {_minutes(timeout)} : une boucle de ton code ne s'arrête-t-elle pas ?")
        return GroupResult(summary="timeout", timed_out=True)

    lines = result.stdout.strip().splitlines()
    failures = []                                        # (rank, pending, kind, name, reason)
    for line in lines:
        if not line.startswith(("FAILED ", "ERROR ")):
            continue
        kind, _, rest = line.partition(" ")
        node, _, reason = rest.partition(" - ")
        name = node.split("::", 1)[1] if "::" in node else node
        pending = reason.startswith(_PENDING_MARK)
        clear_first = "returns_self" in name or "copies" in name or _NONE_ATTRIBUTE in reason
        rank = 2 if pending else (0 if clear_first else 1)
        failures.append((rank, pending, kind, name, reason))
    failures.sort(key=lambda f: f[0])                    # stable: pytest's order within each rank

    for line in lines:                                   # conftest notes: modules not created yet, import errors
        text = line.strip()
        if text.startswith("ℹ️"):
            print(text)
        elif "Exit: " in text:
            print(text.split("Exit: ", 1)[1].strip(" !"))
    if any(_NONE_ATTRIBUTE in f[4] for f in failures):
        print("💡 un attribut est lu sur None : une fonction ou une méthode (fit doit rendre self) "
              "oublie-t-elle son return ?")
    for _, pending, kind, name, reason in failures[:max_shown]:
        mark = _PENDING_MARK if pending else "❌"
        reason = reason.removeprefix(_PENDING_MARK).strip()
        where = " (erreur hors du test : import, fixture…)" if kind == "ERROR" else ""
        print(f"{mark} {name}{where}" + (f"\n   {reason[:800]}" if reason else ""))
    hidden = failures[max_shown:]
    if hidden:
        n_pending = sum(f[1] for f in hidden)
        detail = f", dont {n_pending} pas encore écrit(s)" if n_pending else ""
        print(f"   … et {len(hidden)} autre(s) test(s) en échec{detail}")
    counts = [line for line in lines if "Exit: " not in line and _SUMMARY.search(line)]
    summary = (counts[-1].strip("= ") if counts else lines[-1].strip("=! ")[:300] if lines
               else result.stderr.strip()[-300:])
    print("pytest:", summary)
    return GroupResult(passed=_count(summary, "passed"), failed=_count(summary, "failed"),
                       errors=_count(summary, "errors?"), pending=sum(f[1] for f in failures), summary=summary)
