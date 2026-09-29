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
"""

from __future__ import annotations

import importlib
import importlib.util
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

__all__ = ["IMPLEMENTATIONS", "load_mylearn", "mylearn_path", "MylearnNotFoundError"]


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


def _forget(alias: str) -> None:
    for name in list(sys.modules):
        if name == alias or name.startswith(alias + "."):
            del sys.modules[name]
    importlib.invalidate_caches()


def load_mylearn(
    impl: str | None = None,
    *,
    path: str | Path | None = None,
    root: str | Path | None = None,
    alias: str = "mylearn",
    missing_ok: bool = False,
    verbose: bool = False,
):
    """Import an implementation of mylearn and register it as ``alias``.

    ``impl`` defaults to the ``WB_IMPL`` environment variable, else ``"learner"``.
    ``path`` points to another directory (used by the tests of the mechanism).
    With ``missing_ok=True`` a missing package returns a placeholder whose use
    prints "⏳ pas encore fait" inside ``wb.attempt`` (exercise notebooks).
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
    if verbose:
        print(f"📦 mylearn chargé depuis {pkg_dir} (implémentation : {impl})")
    return module
