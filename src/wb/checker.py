"""Automatic answer checking without revealing the solution (BIBLE §13).

How it works
------------
* In a **solutions** notebook, a code cell tagged ``answer`` calls
  ``wb.record("18.4", value, decimals=3)``. It prints a line
  ``WB_ANSWER {...json...}`` containing only a salted SHA-256 hash of the
  normalised value (plus harmless metadata: kind, number of decimals, shape,
  sign, order of magnitude, hashes of common mistakes).
* ``tools/build_answers.py`` harvests these lines from the executed notebook
  and updates ``src/wb/answers.json``. The file is never edited by hand.
* In an **exercise** notebook, ``wb.check("18.4", my_value)`` normalises the
  learner's value the same way, hashes it and compares. On failure it tries
  a few classic mistakes (sign, percentage, off-by-one, rounding...) to give
  a helpful message, without ever printing the expected value.

The hash is a pedagogical safeguard, not a safe: a simple value (a small
integer, a short word) can be recovered by brute force. Discipline stays with
the learner.

Normalisation
-------------
* floats: rounded to ``decimals`` decimals (``f"{x:.{d}f}"``, ``-0.0`` -> ``0.0``);
  a tolerance of a tenth of the last decimal absorbs rounding-boundary noise,
  and both roundings are accepted when the answer sits on a boundary (0.125);
  "3,14" and "1 000,5" are read as numbers;
* ints: exact (``344.0`` is accepted for ``344``);
* one-element containers (a Series from ``.mode()``, a 1-element tensor) are
  accepted for scalar answers;
* strings: case, accents, whitespace and hyphens ignored; typographic
  apostrophes unified; ligatures (œ, æ) unfolded;
* booleans: ``True``/``False`` (``"vrai"``, ``"oui"``... accepted);
* arrays (lists, NumPy, PyTorch, pandas): shape + rounded values; arrays of
  integers (classes, counts) reject non-integer values;
* sets (``ordered=False``): sorted normalised elements.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import sys
import unicodedata
from contextlib import contextmanager
from pathlib import Path

ANSWERS_PATH = Path(__file__).with_name("answers.json")
ANSWER_PREFIX = "WB_ANSWER "
FORMAT_VERSION = 1
_MAX_ELEMENT_HASHES = 100  # per-element hashes are stored for small arrays only

_PENDING_VALUES = (None, Ellipsis, NotImplemented)
_TRUE_WORDS = {"true", "vrai", "oui", "yes", "1"}
_FALSE_WORDS = {"false", "faux", "non", "no", "0"}

# Answers recorded in this Python process (solutions notebooks, tests).
RECORDED: dict[str, dict] = {}

_cache: dict = {"mtime": None, "path": None, "answers": {}}


class NormalizationError(ValueError):
    """The value cannot be converted to the expected kind."""


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
class CheckResult:
    """Outcome of :func:`check`. Truthy when the answer is correct.

    The message is printed by ``check``; the object itself displays nothing
    in a notebook, so ``wb.check(...)`` as the last line of a cell does not
    print ``True``/``False``.
    """

    def __init__(self, ex_id: str, passed: bool, status: str, message: str):
        self.ex_id = ex_id
        self.passed = passed
        self.status = status  # "correct", "wrong", "pending", "unknown", "type"
        self.message = message

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        return f"CheckResult(ex_id={self.ex_id!r}, status={self.status!r})"

    def _ipython_display_(self) -> None:  # the message was already printed
        return None


# ---------------------------------------------------------------------------
# Normalisation and hashing
# ---------------------------------------------------------------------------
def _to_python(value):
    """Convert torch / pandas / NumPy objects to NumPy arrays or Python scalars."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.float()
        value = tensor.numpy()
    elif type(value).__name__ == "NAType":
        raise NormalizationError("une vraie valeur (pas `pd.NA`, qui signifie « manquant »)")
    elif type(value).__module__.startswith("pandas") and hasattr(value, "to_numpy"):
        value = value.to_numpy()
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _deep_to_python(value):
    """Like :func:`_to_python`, recursively inside lists and tuples."""
    value = _to_python(value)
    if isinstance(value, (list, tuple)):
        return [_deep_to_python(item) for item in value]
    return value


def _unwrap_single(value):
    """A container with exactly one element (Series, 1-element tensor...) -> that element."""
    value = _to_python(value)
    if isinstance(value, (str, bytes)):
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray) and value.size == 1:
            item = value.reshape(-1)[0]
            return item.item() if hasattr(item, "item") else item
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_single(value[0])
    return value


def infer_kind(value) -> str:
    """Guess the kind of an answer: bool, int, float, str, set or array."""
    value = _to_python(value)
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (set, frozenset)):
        return "set"
    return "array"


def _fmt_float(x: float, decimals: int) -> str:
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    text = f"{x:.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"  # turn "-0.00" into "0.00"
    return text


_CHAR_MAP = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'",
                           "œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae"})
_IGNORED_CHARS = set("-‐‑‒–—−_   ")


def _norm_str(value) -> str:
    """Case, accents, spaces and hyphens are ignored; typographic apostrophes unified."""
    text = unicodedata.normalize("NFKD", str(value).translate(_CHAR_MAP))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return "".join(ch for ch in "".join(text.split()) if ch not in _IGNORED_CHARS)


def _norm_bool(value) -> str:
    value = _unwrap_single(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Number) and value in (0, 1):
        return "true" if value else "false"
    if isinstance(value, str):
        word = _norm_str(value)
        if word in _TRUE_WORDS:
            return "true"
        if word in _FALSE_WORDS:
            return "false"
    raise NormalizationError("un booléen (True ou False)")


def _parse_number(text: str) -> float:
    cleaned = "".join(text.split()).replace("_", "")
    for space in (" ", " ", " "):
        cleaned = cleaned.replace(space, "")
    if "," in cleaned and "." in cleaned:  # 1,000.5 (English thousands separator)
        cleaned = cleaned.replace(",", "")
    else:  # 3,14 (French decimal comma)
        cleaned = cleaned.replace(",", ".")
    return float(cleaned)


def _as_number(value) -> float:
    value = _unwrap_single(value)
    try:
        if isinstance(value, (bool, numbers.Real)):
            return float(value)
        if isinstance(value, str):
            return _parse_number(value)
    except OverflowError as exc:
        raise NormalizationError("un nombre de taille raisonnable (celui-ci est trop grand)") from exc
    except ValueError:
        pass
    raise NormalizationError("un nombre")


def _norm_int(value) -> str:
    raw = _unwrap_single(value)
    if isinstance(raw, numbers.Integral) and not isinstance(raw, bool):
        return str(int(raw))  # exact, even for very large integers
    x = _as_number(raw)
    if not math.isfinite(x) or abs(x - round(x)) > 1e-9:
        raise NormalizationError("un nombre entier")
    return str(int(round(x)))


def _as_array(value):
    import numpy as np

    value = _deep_to_python(value)
    if isinstance(value, (str, bytes)) or isinstance(value, numbers.Number):
        raise NormalizationError("un tableau (liste, array NumPy ou tenseur)")
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:  # ragged lists, objects, tensors needing grad...
        raise NormalizationError("un tableau de nombres (lignes de même longueur)") from exc
    return arr


def _norm_array(arr, decimals: int) -> str:
    shape = ",".join(str(s) for s in arr.shape)
    body = ",".join(_fmt_float(x, decimals) for x in arr.ravel())
    return f"shape=({shape});{body}"


def _norm_set(value, decimals: int | None) -> str:
    value = _deep_to_python(value)
    if isinstance(value, (str, bytes)) or isinstance(value, numbers.Number):
        raise NormalizationError("un ensemble d'éléments (set ou liste)")
    try:
        elements = list(value)
    except TypeError as exc:
        raise NormalizationError("un ensemble d'éléments (set ou liste)") from exc
    items = []
    for item in elements:
        item = _to_python(item)
        if isinstance(item, str):
            items.append("s:" + _norm_str(item))
        elif isinstance(item, numbers.Real):
            items.append("n:" + _fmt_float(item, decimals or 0))
        else:
            raise NormalizationError("un ensemble de nombres ou de chaînes")
    return "{" + ",".join(sorted(set(items))) + "}"


def normalize(value, kind: str, decimals: int | None = None) -> str:
    """Return the canonical string of ``value`` for the given ``kind``."""
    if kind == "bool":
        return _norm_bool(value)
    if kind == "int":
        return _norm_int(value)
    if kind == "float":
        return _fmt_float(_as_number(value), decimals if decimals is not None else 0)
    if kind == "str":
        raw = _unwrap_single(value)
        if not isinstance(raw, str):
            raise NormalizationError("une chaîne de caractères (str)")
        return _norm_str(raw)
    if kind == "array":
        return _norm_array(_as_array(value), decimals or 0)
    if kind == "set":
        return _norm_set(value, decimals)
    raise ValueError(f"unknown answer kind: {kind!r}")


def hash_answer(ex_id: str, kind: str, normalized: str) -> str:
    """Salted SHA-256 of a normalised answer."""
    payload = f"dlwb|{ex_id}|{kind}|{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _element_hash(ex_id: str, index: int, x: float, decimals: int) -> str:
    return hash_answer(ex_id, f"element:{index}", _fmt_float(x, decimals))


def _hash_value(ex_id: str, entry: dict, value, decimals: int | None = None) -> str | None:
    try:
        norm = normalize(value, entry["kind"], entry.get("decimals") if decimals is None else decimals)
    except (NormalizationError, TypeError, ValueError):
        return None
    return hash_answer(ex_id, entry["kind"], norm)


# ---------------------------------------------------------------------------
# answers.json
# ---------------------------------------------------------------------------
def load_answers(path: str | Path | None = None, *, reload: bool = False) -> dict:
    """Load ``answers.json`` (cached; reloaded automatically after a ``git pull``)."""
    path = Path(path) if path else ANSWERS_PATH
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if reload or _cache["path"] != path or _cache["mtime"] != mtime:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        _cache.update(path=path, mtime=mtime, answers=data.get("answers", {}))
    return _cache["answers"]


def _get_entry(ex_id: str, answers_path=None) -> dict | None:
    answers = load_answers(answers_path)
    if ex_id in answers:
        return answers[ex_id]
    return RECORDED.get(ex_id)


# ---------------------------------------------------------------------------
# Building an answer entry (solutions side)
# ---------------------------------------------------------------------------
def _near_rounding_boundary(x: float, decimals: int) -> bool:
    scaled = abs(x) * 10**decimals
    frac = scaled - math.floor(scaled)
    return abs(frac - 0.5) < 0.02


def _other_rounding(x: float, decimals: int, norm: str) -> str | None:
    """At a rounding boundary (e.g. 0.125 at 2 decimals), the other plausible rounding."""
    if not math.isfinite(x) or not _near_rounding_boundary(x, decimals):
        return None
    scale = 10**decimals
    for candidate in (math.floor(x * scale) / scale, math.ceil(x * scale) / scale):
        text = _fmt_float(candidate, decimals)
        if text != norm:
            return text
    return None


def make_entry(
    ex_id: str,
    value,
    decimals: int | None = None,
    *,
    kind: str | None = None,
    ordered: bool = True,
    mistakes: dict | None = None,
    source: str | None = None,
) -> dict:
    """Build the (hash-only) answer entry for ``value``. Used by :func:`record`."""
    import numpy as np

    ex_id = str(ex_id)
    value = _to_python(value)
    kind = kind or infer_kind(value)
    if kind == "array" and not ordered:
        kind = "set"
    if kind == "float" and decimals is None:
        raise ValueError(
            f"Ex {ex_id}: give `decimals` for a float answer (the statement must say how to round)."
        )
    integer_array = False
    if kind == "array":
        arr = _as_array(value)
        integer_array = bool(arr.size == 0 or np.all(arr == np.round(arr)))
        if decimals is None:
            if not integer_array:
                raise ValueError(f"Ex {ex_id}: give `decimals` for an array of floats.")
            decimals = 0
    if kind == "set" and decimals is None:
        reals = [x for x in _deep_to_python(list(value)) if isinstance(x, numbers.Real)]
        if any(float(x) != round(float(x)) for x in reals):
            raise ValueError(f"Ex {ex_id}: give `decimals` for a set containing non-integer numbers.")
    if kind in ("int", "bool", "str"):
        decimals = None

    norm = normalize(value, kind, decimals)
    entry: dict = {"kind": kind, "hash": hash_answer(ex_id, kind, norm)}
    if decimals is not None:
        entry["decimals"] = int(decimals)

    if kind in ("int", "float"):
        x = _as_number(value)
        if kind == "float" and x != 0 and float(norm) == 0:
            raise ValueError(f"Ex {ex_id}: the answer rounds to 0 with {decimals} decimals; use more decimals.")
        entry["sign"] = 0 if x == 0 else (1 if x > 0 else -1)
        entry["magnitude"] = None if x == 0 or not math.isfinite(x) else int(math.floor(math.log10(abs(x))))
        if kind == "float" and decimals and decimals > 0:
            entry["hash_coarse"] = hash_answer(ex_id, kind, _fmt_float(x, decimals - 1))
        if kind == "float":
            other = _other_rounding(x, decimals, norm)
            if other is not None:
                entry["alt_hashes"] = [hash_answer(ex_id, kind, other)]
    if kind == "array":
        arr = _as_array(value)
        entry["shape"] = list(arr.shape)
        if integer_array and decimals == 0:
            entry["integer"] = True
        if 1 < arr.size <= _MAX_ELEMENT_HASHES:
            entry["element_hashes"] = [_element_hash(ex_id, i, x, decimals) for i, x in enumerate(arr.ravel())]
    if mistakes:
        entry["mistakes"] = {}
        for message, wrong_value in mistakes.items():
            try:
                wrong_norm = normalize(wrong_value, kind, decimals)
            except NormalizationError as exc:
                raise ValueError(f"Ex {ex_id}: mistake {message!r} has the wrong kind") from exc
            if wrong_norm == norm:
                raise ValueError(f"Ex {ex_id}: mistake {message!r} equals the right answer")
            entry["mistakes"][hash_answer(ex_id, kind, wrong_norm)] = str(message)
    if source:
        entry["source"] = source
    return entry


# ---------------------------------------------------------------------------
# Checking (learner side)
# ---------------------------------------------------------------------------
_PRAISE = [
    "Bravo !",
    "Exact !",
    "Bien joué !",
    "C'est ça !",
    "Parfait !",
]


def _praise(ex_id: str) -> str:
    return _PRAISE[sum(map(ord, ex_id)) % len(_PRAISE)]


def _numeric_diagnosis(ex_id: str, entry: dict, x: float) -> str:
    """Explain a wrong scalar answer without revealing the expected value."""
    decimals = entry.get("decimals")
    if math.isnan(x):
        return ("Ta valeur est NaN (« pas un nombre ») : division 0/0, logarithme d'un nombre "
                "négatif ou valeur manquante dans le calcul ?")
    if math.isinf(x):
        return "Ta valeur est infinie : division par zéro ou exponentielle trop grande ?"

    def matches(candidate: float) -> bool:
        return _hash_value(ex_id, entry, candidate) == entry["hash"]

    if x != 0 and matches(-x):
        return "Le signe est inversé : vérifie l'ordre d'une soustraction ou un signe « moins »."
    if entry["kind"] == "int" and (matches(x + 1) or matches(x - 1)):
        return "Tu es à 1 près : erreur de bornes (off-by-one) ? Vérifie si les bornes sont incluses."
    if x != 0 and matches(x / 100):
        return "On attend une proportion entre 0 et 1, pas un pourcentage."
    if x != 0 and matches(x * 100):
        return "On attend un pourcentage, pas une proportion entre 0 et 1."
    if 0 <= x <= 1 and matches(1 - x):
        return "Tu as calculé le complément (1 − p) : relis bien ce qui est demandé."
    if entry["kind"] == "float" and decimals and entry.get("hash_coarse"):
        coarse = hash_answer(ex_id, "float", _fmt_float(x, decimals - 1))
        if coarse == entry["hash_coarse"]:
            return (
                f"Tu y es presque : c'est juste à {decimals - 1} décimale(s), mais pas à {decimals}. "
                "Arrondi trop tôt dans le calcul ?"
            )
    sign = entry.get("sign")
    if not sign or x == 0:  # expected 0, or the learner gave 0: no hint that could leak the answer
        return "Ce n'est pas la bonne valeur."
    if (x > 0) != (sign > 0):
        return "Le signe de ta réponse n'est pas le bon."
    magnitude = entry.get("magnitude")
    if magnitude is not None:
        x_mag = int(math.floor(math.log10(abs(x))))
        if x_mag == magnitude:
            return "L'ordre de grandeur est bon : c'est le détail du calcul qui cloche."
        if x_mag > magnitude:
            return "Ta valeur est trop grande d'au moins un facteur 10 : revois la méthode (unités, somme au lieu d'une moyenne ?)."
        return "Ta valeur est trop petite d'au moins un facteur 10 : revois la méthode (division en trop, unités ?)."
    return "Ce n'est pas la bonne valeur."


def _element_ok(ex_id: str, index: int, x: float, expected: str, decimals: int) -> bool:
    step = 10**-decimals / 10
    return any(_element_hash(ex_id, index, c, decimals) == expected for c in (x, x + step, x - step))


def _array_diagnosis(ex_id: str, entry: dict, arr) -> str:
    import numpy as np

    expected_shape = tuple(entry.get("shape", ()))
    if arr.shape != expected_shape:
        if arr.T.shape == expected_shape and _hash_value(ex_id, entry, arr.T) == entry["hash"]:
            return "Les valeurs sont bonnes mais le tableau est transposé (.T)."
        if arr.size == int(np.prod(expected_shape)) and (
            _hash_value(ex_id, entry, arr.reshape(expected_shape)) == entry["hash"]
        ):
            return f"Les valeurs sont bonnes, mais la forme doit être {expected_shape} (reshape)."
        return f"Forme attendue {expected_shape}, reçue {arr.shape}."
    element_hashes = entry.get("element_hashes")
    if element_hashes:
        decimals = entry.get("decimals") or 0
        wrong = [np.unravel_index(i, arr.shape) for i, (x, h) in enumerate(zip(arr.ravel(), element_hashes))
                 if not _element_ok(ex_id, i, x, h, decimals)]
        if wrong:
            first = tuple(int(i) for i in wrong[0])
            return (f"{arr.size - len(wrong)} élément(s) sur {arr.size} sont justes. "
                    f"Premier élément faux à l'indice {first}.")
    if _hash_value(ex_id, entry, -arr) == entry["hash"]:
        return "Tous les signes sont inversés."
    return "Les valeurs ne sont pas les bonnes."


def _type_name(value) -> str:
    return type(value).__name__


def check_entry(ex_id: str, entry: dict, value) -> tuple[bool, str, str]:
    """Compare ``value`` with a stored entry. Return (passed, status, message)."""
    import numpy as np

    kind = entry["kind"]
    decimals = entry.get("decimals")
    try:
        norm = normalize(value, kind, decimals)
    except NormalizationError as exc:
        return False, "type", f"J'attends {exc} ; j'ai reçu un objet de type `{_type_name(value)}`."

    if kind == "array":
        arr = _as_array(value)
        if entry.get("integer") and arr.size and not np.all(arr == np.round(arr)):
            return False, "wrong", ("On attend des valeurs entières (des classes ou des comptes ?), "
                                    "pas des nombres à virgule (des probabilités ?).")
        if hash_answer(ex_id, kind, norm) == entry["hash"]:
            return True, "correct", _praise(ex_id)
        element_hashes = entry.get("element_hashes")
        if element_hashes and list(arr.shape) == entry.get("shape") and all(
            _element_ok(ex_id, i, x, h, decimals or 0) for i, (x, h) in enumerate(zip(arr.ravel(), element_hashes))
        ):
            return True, "correct", _praise(ex_id)  # right up to rounding noise (e.g. float32)
        mistake = entry.get("mistakes", {}).get(hash_answer(ex_id, kind, norm))
        if mistake:
            return False, "wrong", f"Erreur classique. Piste : {mistake}"
        return False, "wrong", _array_diagnosis(ex_id, entry, arr)

    candidates = [norm]
    if kind == "float":  # absorb noise at a rounding boundary
        x = _as_number(value)
        step = 10 ** -(decimals or 0) / 10
        candidates += [_fmt_float(x + step, decimals or 0), _fmt_float(x - step, decimals or 0)]
    accepted = {entry["hash"], *entry.get("alt_hashes", [])}
    if any(hash_answer(ex_id, kind, c) in accepted for c in candidates):
        return True, "correct", _praise(ex_id)

    mistake = entry.get("mistakes", {}).get(hash_answer(ex_id, kind, norm))
    if mistake:
        return False, "wrong", f"Erreur classique. Piste : {mistake}"
    if kind in ("int", "float"):
        return False, "wrong", _numeric_diagnosis(ex_id, entry, _as_number(value))
    if kind == "bool":
        return False, "wrong", "Ce n'est pas la bonne réponse : relis l'énoncé et justifie ton choix."
    if kind == "set":
        return False, "wrong", "L'ensemble n'est pas le bon (l'ordre ne compte pas) : il manque ou il y a des éléments en trop."
    return False, "wrong", (
        "Ce n'est pas la réponse attendue (majuscules, accents, espaces et tirets sont ignorés). "
        "Vérifie l'orthographe et le terme exact demandé."
    )


def check(ex_id, value, decimals: int | None = None, *, quiet: bool = False,
          answers_path: str | Path | None = None) -> CheckResult:
    """Check the learner's answer to exercise ``ex_id``.

    ``decimals`` is optional: the number of decimals is stored with the answer
    (it is the one given in the statement). Prints a message and returns a
    :class:`CheckResult`, which is truthy when the answer is correct.
    """
    ex_id = str(ex_id)
    label = f"Ex {ex_id}"

    def done(passed: bool, status: str, message: str, icon: str) -> CheckResult:
        if not quiet:
            print(f"{icon} {label} : {message}")
        return CheckResult(ex_id, passed, status, message)

    if any(value is pending for pending in _PENDING_VALUES):
        return done(False, "pending", "pas encore fait (remplace `...` ou `None` par ta réponse).", "⏳")

    entry = _get_entry(ex_id, answers_path)
    if entry is None:
        return done(
            False,
            "unknown",
            "aucune réponse enregistrée pour cet exercice. Vérifie l'ID, puis fais un `git pull` "
            "(le fichier src/wb/answers.json est peut-être plus récent sur GitHub).",
            "❓",
        )
    note = ""
    stored = entry.get("decimals")
    if decimals is not None and stored is not None and int(decimals) != stored:
        note = f" (j'arrondis à {stored} décimale(s), comme dans l'énoncé)"

    passed, status, message = check_entry(ex_id, entry, value)
    icon = "✅" if passed else "❌"
    return done(passed, status, message + note, icon)


@contextmanager
def attempt(ex_id: str | None = None):
    """Run exercise code; if it is not written yet, print ⏳ instead of crashing.

    Usage in an exercise notebook::

        with wb.attempt("3.4"):
            wb.check("3.4", mylearn.metrics.f1(y_true, y_pred))

    Caught: ``NotImplementedError`` (a TODO not done yet) and a missing mylearn
    module or package. Real bugs still show their traceback.
    """
    from wb.errors import MylearnMissingError

    where = f"Ex {ex_id} : " if ex_id else ""
    try:
        yield
    except NotImplementedError as exc:
        detail = f" ({exc})" if str(exc) else ""
        print(f"⏳ {where}pas encore fait{detail}.")
    except MylearnMissingError as exc:
        print(f"⏳ {where}pas encore fait ({exc}).")
    except ImportError as exc:  # includes ModuleNotFoundError
        if not (exc.name or "").startswith("mylearn"):
            raise
        print(f"⏳ {where}pas encore fait (module {exc.name} absent ou incomplet : lance tools/start_chapter.py).")


# ---------------------------------------------------------------------------
# Recording (solutions side)
# ---------------------------------------------------------------------------
def _perturbed(value, entry: dict):
    """A value that must be rejected: used to prove the check can fail."""
    kind = entry["kind"]
    decimals = entry.get("decimals") or 0
    if kind == "bool":
        return not (_norm_bool(value) == "true")
    if kind == "int":
        return int(_norm_int(value)) + 1
    if kind == "float":
        x = _as_number(value)
        return x + 3 * 10 ** -decimals if math.isfinite(x) else 0.0
    if kind == "str":
        return str(value) + "zz"
    if kind == "set":
        return list(value) + ["__not_an_answer__"]
    arr = _as_array(value).copy()
    if arr.size == 0:
        return [[0.0]]
    arr.ravel()[0] += 3 * 10 ** -decimals
    return arr


def record(
    ex_id,
    value,
    decimals: int | None = None,
    *,
    kind: str | None = None,
    ordered: bool = True,
    mistakes: dict | None = None,
    source: str | None = None,
) -> dict:
    """Record the right answer of an exercise (solutions notebooks only).

    Call it in a code cell tagged ``answer``. ``mistakes`` maps a hint message
    to a classic wrong value, e.g. ``{"tu as oublié le biais": 0.42}``.
    The function checks that the right value passes and a wrong value fails,
    then prints the ``WB_ANSWER`` line harvested by ``tools/build_answers.py``.
    """
    ex_id = str(ex_id)
    entry = make_entry(ex_id, value, decimals, kind=kind, ordered=ordered,
                       mistakes=mistakes, source=source)
    ok, _, message = check_entry(ex_id, entry, value)
    if not ok:
        raise AssertionError(f"Ex {ex_id}: the right value does not pass its own check ({message})")
    bad_ok, _, _ = check_entry(ex_id, entry, _perturbed(value, entry))
    if bad_ok:
        raise AssertionError(f"Ex {ex_id}: a wrong value passes the check; use more decimals")
    if entry.get("alt_hashes"):
        print(f"ℹ️ Ex {ex_id} : valeur à une limite d'arrondi ; les deux arrondis sont acceptés.")
    if entry["kind"] == "array" and "element_hashes" not in entry and entry.get("decimals"):
        arr = _as_array(value)
        if any(_near_rounding_boundary(float(x), entry["decimals"]) for x in arr.ravel()):
            print(f"⚠️ Ex {ex_id} : un élément est proche d'une limite d'arrondi ; envisage un autre nombre de décimales.")
    RECORDED[ex_id] = entry
    print(ANSWER_PREFIX + json.dumps({"id": ex_id, **entry}, ensure_ascii=False, sort_keys=True))
    print(f"📝 Ex {ex_id} : réponse enregistrée ({entry['kind']}"
          + (f", {entry['decimals']} décimale(s)" if "decimals" in entry else "") + ").")
    return entry
