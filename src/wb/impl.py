"""Load one implementation of the ``mylearn`` library under the name ``mylearn``.

Three implementations live in the repository:

* ``learner`` -> ``mon_travail/mylearn/``  (the learner's own code, default)
* ``ref``     -> ``solutions/mylearn_ref/`` (reference implementation)
* ``stubs``   -> ``templates/mylearn_stubs/`` (empty skeletons; used by Claude
  to prove that the tests fail before anything is implemented)

Whichever is chosen is registered in ``sys.modules`` as ``mylearn``, so
``from mylearn.metrics import f1`` works the same way everywhere (notebooks,
tests). Submodules are also imported on first attribute access, so
``mylearn.metrics.f1`` works without an explicit import. Inside the packages,
imports between modules must be *relative* (``from .utils import x``).

Fallback (``fallback="ref"``, learner only): a module that is missing from the
learner's package is taken from the reference, so that skipping a chapter does
not break the later ones (e.g. ``ensemble.py`` of chapter 14 imports ``tree.py``
of chapter 13). With ``chapter="14"`` only the modules of chapters *before* 14
may come from the reference: the modules you are writing never do, so a check
can never pass with the reference's code instead of yours.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path

from wb.core import repo_root
from wb.errors import MylearnMissingError, MylearnNotFoundError

IMPLEMENTATIONS = {
    "learner": Path("mon_travail") / "mylearn",
    "ref": Path("solutions") / "mylearn_ref",
    "stubs": Path("templates") / "mylearn_stubs",
}

# Study order of the chapters (same as tools/syllabus.py ORDER; checked by a test).
CHAPTER_ORDER = (
    "0A", "0B", "1", "2", "3", "4", "5", "6", "CP1", "7", "8", "9", "10", "11", "CP2",
    "12", "13", "14", "15", "CP3", "16", "17", "18", "19", "20", "CP4",
    "21", "22", "23", "24", "CP5", "25", "26", "27", "28", "29", "CP6",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "PF",
)

__all__ = ["CHAPTER_ORDER", "IMPLEMENTATIONS", "load_mylearn", "modules_before", "mylearn_path",
           "MylearnNotFoundError"]


class _MissingPackage:
    """Placeholder returned by ``load_mylearn(..., missing_ok=True)``.

    Any attribute access raises :class:`MylearnMissingError`, which
    ``wb.attempt`` reports as "⏳ pas encore fait" instead of crashing.
    """

    def __init__(self, message: str):
        self._message = message

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise MylearnMissingError(self._message)

    def __repr__(self) -> str:
        return "<mylearn absent : lance tools/start_chapter.py>"


def mylearn_path(impl: str = "learner", root: str | Path | None = None) -> Path:
    """Directory of the requested implementation."""
    if impl not in IMPLEMENTATIONS:
        raise ValueError(f"impl must be one of {sorted(IMPLEMENTATIONS)}, not {impl!r}")
    root = Path(root) if root else repo_root()
    return root / IMPLEMENTATIONS[impl]


def _attach_lazy_getattr(module) -> None:
    """Let ``module.sub`` import the submodule ``sub`` on first access (packages only)."""
    if hasattr(module, "__path__") and "__getattr__" not in module.__dict__:
        module.__getattr__ = _make_lazy_getattr(module.__name__)


def _make_lazy_getattr(prefix: str):
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        full = f"{prefix}.{name}"
        try:
            submodule = importlib.import_module(full)
        except ModuleNotFoundError as exc:
            if exc.name and (full == exc.name or full.startswith(exc.name + ".")):
                raise MylearnMissingError(
                    f"le module {full} n'existe pas encore dans ta librairie : "
                    "lance tools/start_chapter.py pour ce chapitre"
                ) from None
            raise
        _attach_lazy_getattr(submodule)  # nested packages: mylearn.nn.activations
        return submodule

    return __getattr__


def _manifest(root: str | Path | None = None) -> dict:
    root = Path(root) if root else repo_root()
    path = root / IMPLEMENTATIONS["stubs"] / "MANIFEST.json"
    if not path.exists():
        return {"base": [], "chapters": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_chapter(chapter: str) -> str:
    text = str(chapter).strip().upper().removeprefix("CH")
    if text[:1].isdigit():
        digits = "".join(ch for ch in text if ch.isdigit())
        text = str(int(digits)) + text[len(digits):]
    if text not in CHAPTER_ORDER:
        raise ValueError(f"unknown chapter {chapter!r} (examples: 18, 0A, B3)")
    return text


def modules_before(chapter: str | None, root: str | Path | None = None) -> set[str] | None:
    """Stub files (e.g. ``"tree.py"``, ``"nn/layers.py"``) published before ``chapter``.

    ``None`` for ``chapter=None`` (no restriction). The base files are included.
    """
    if chapter is None:
        return None
    rank = CHAPTER_ORDER.index(_canonical_chapter(chapter))
    manifest = _manifest(root)
    allowed = set(manifest.get("base", []))
    for cid, files in manifest.get("chapters", {}).items():
        if cid in CHAPTER_ORDER and CHAPTER_ORDER.index(cid) < rank:
            allowed.update(files)
    return allowed


class _FallbackFinder(importlib.abc.MetaPathFinder):
    """Import ``alias.x`` from ``fallback`` when the learner's package has no module ``x``."""

    def __init__(self, alias: str, primary: Path, fallback: Path, allowed: set[str] | None,
                 notify: bool, owners: dict[str, str]):
        self.alias, self.primary, self.fallback = alias, primary, fallback
        self.allowed, self.notify, self.owners = allowed, notify, owners
        self.used: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.alias + "."):
            return None
        rel = "/".join(fullname.split(".")[1:])
        if (self.primary / f"{rel}.py").exists() or (self.primary / rel / "__init__.py").exists():
            return None  # the learner's own module wins
        package = self.fallback / rel / "__init__.py"
        module = self.fallback / f"{rel}.py"
        if package.exists():
            # its sub-modules are searched in the learner's folder only: each one comes
            # back through this finder, which applies the same rule to it
            candidate, rel_file = package, f"{rel}/__init__.py"
            locations = [str(self.primary / rel)]
        elif module.exists():
            candidate, rel_file, locations = module, f"{rel}.py", None
        else:
            return None
        if self.allowed is not None and rel_file not in self.allowed:
            return None  # a module of this chapter or a later one: never replaced
        self.used.append(fullname)
        if self.notify and locations is None:
            owner = self.owners.get(rel_file)
            where = f" (ch. {owner})" if owner else ""
            print(f"ℹ️ {fullname} : tu n'as pas ce module{where}, "
                  "la référence est utilisée.")
        return importlib.util.spec_from_file_location(
            fullname, candidate, submodule_search_locations=locations)


def _remove_finders(alias: str) -> None:
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not (isinstance(f, _FallbackFinder) and f.alias == alias)]


def _forget(alias: str) -> None:
    for name in list(sys.modules):
        if name == alias or name.startswith(alias + "."):
            del sys.modules[name]
    _remove_finders(alias)
    importlib.invalidate_caches()


def load_mylearn(
    impl: str | None = None,
    *,
    path: str | Path | None = None,
    root: str | Path | None = None,
    alias: str = "mylearn",
    missing_ok: bool = False,
    verbose: bool = False,
    fallback: str | None = None,
    chapter: str | None = None,
    notify: bool = True,
):
    """Import an implementation of mylearn and register it as ``alias``.

    ``impl`` defaults to the ``WB_IMPL`` environment variable, else ``"learner"``.
    ``path`` points to another directory (used by the tests of the mechanism).
    With ``missing_ok=True`` a missing package returns a placeholder whose use
    prints "⏳ pas encore fait" inside ``wb.attempt`` (exercise notebooks).
    ``fallback="ref"`` (learner only): modules missing from the learner's package
    come from the reference; with ``chapter`` given, only the modules of earlier
    chapters (see ``modules_before``). ``notify`` prints a line for each of them.
    """
    impl = impl or os.environ.get("WB_IMPL", "learner")
    pkg_dir = Path(path) if path else mylearn_path(impl, root)
    init = pkg_dir / "__init__.py"
    if not init.exists():
        if impl == "learner" and not path:
            message = (
                "ta librairie mylearn n'existe pas encore dans mon_travail/mylearn/ ; "
                "lance d'abord : python tools/start_chapter.py <chapitre>"
            )
        else:
            message = f"aucun package mylearn trouvé dans {pkg_dir}"
        _forget(alias)  # never leave a previously loaded implementation (e.g. ref) in place
        if missing_ok:
            print("ℹ️ " + message[0].upper() + message[1:] + ".")
            return _MissingPackage(message)
        raise MylearnNotFoundError(message)

    _forget(alias)  # forget any previously loaded implementation (and its submodules)

    spec = importlib.util.spec_from_file_location(
        alias, init, submodule_search_locations=[str(pkg_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[alias]
        raise
    _attach_lazy_getattr(module)
    module.__wb_impl__ = impl if not path else f"{impl}:{pkg_dir}"
    module.__wb_fallback__ = None
    if fallback is not None and impl == "learner":
        fb_dir = mylearn_path(fallback, root)
        manifest = _manifest(root)
        owners = {}
        for cid, files in manifest.get("chapters", {}).items():
            for rel in files:
                owners.setdefault(rel, cid)
        finder = _FallbackFinder(alias, pkg_dir, fb_dir, modules_before(chapter, root), notify,
                                 owners)
        sys.meta_path.insert(0, finder)
        module.__wb_fallback__ = finder
    if verbose:
        print(f"📦 mylearn chargé depuis {pkg_dir} (implémentation : {impl})")
    return module
