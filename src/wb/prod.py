"""Infrastructure of bonus chapter B7, « Du notebook à la production » (``import wb.prod as wbp``).

The learner's code in B7 is the package of their own project, ``cifarprod``: the cells of the notebook WRITE its
files (pathlib) in a project folder, then RUN them in a separate process (subprocess), with absolute paths and the
``PYTHONPATH`` of the project and of ``wb`` (lesson of B5.30). This module gives:

- the project folder: ``project_dir("learner")`` is ``<repo>/mon_travail/b7_production/cifar-prod`` (the learner's
  space, wherever the notebook runs, Colab included); ``project_dir("solution")`` is a temporary folder, the same for
  the whole session: the solutions notebook never writes in the repository;
- :func:`write_file` / :func:`write_files`: write a file of the project from a cell, never outside the project nor
  in the repository outside ``mon_travail/``; a file changed outside the notebook (in an editor) is kept, not
  overwritten (``.wb_written.json`` remembers what the notebook wrote);
- :func:`run` and :func:`cli`: a command with a timeout, its exit code, stdout, stderr and duration
  (:class:`RunResult`); :func:`import_package`: the package of a project imported afresh in the notebook;
- :func:`install_editable`: ``pip install -e`` (or ``uv pip install -e``) of a project in a temporary virtual
  environment that sees the current packages: the learner's environment is never modified;
- the data: :func:`cifar_subset` (2 000 training and 500 validation images of CIFAR-10, stratified and fixed, with
  their fingerprint :data:`SUBSET_FINGERPRINT`), :func:`write_subset` (the ``.npz`` file of the project),
  :func:`test_images` (images of the test split), :func:`array_fingerprint`;
- the base model: :func:`reference_model` (the VGG delivered by the MP5 solution, versioned with the repository),
  :func:`base_model` (the learner's MP5 model if it exists, read only, else the reference: key ``source``),
  :func:`build_reference` (the oracle of the architectures);
- the images of a service: :func:`write_png`, :func:`imread_bgr` (reads like ``cv2.imread``: BGR, uint8);
- 🔮 B7.12: :func:`workers_run` (a short training whose data loader may draw random flips in its workers);
- 📦 B7.15: :data:`MUTANTS`, :func:`mutant_package` and :func:`run_tests` (the learner's tests on faulty versions of
  the reference package ``solutions/b7_cifarprod``).

Nothing here imports PyTorch at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PACKAGE = "cifarprod"
PROJECT_NAME = "cifar-prod"
WORK_FOLDER = "b7_production"
MANIFEST = ".wb_written.json"
SUBSET_FILE = "cifar10_2000.npz"
SUBSET_PER_CLASS = {"train": 200, "val": 50}          # 2 000 training and 500 validation images
SUBSET_FINGERPRINT = "286106e2dece6e60837434f3a343b7e33546b6a807ed5bb28b78b32e51558deb"   # x_train, y_train, x_val, y_val
MP5_FOLDER = "projets/partie_5_classifieur_cifar10"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

_SOLUTION_PROJECT: Path | None = None


def repo_root() -> Path:
    from wb.core import repo_root as _root
    return Path(_root())


def reference_package() -> Path:
    """The reference package of the chapter, ``solutions/b7_cifarprod`` (read only)."""
    return repo_root() / "solutions" / "b7_cifarprod"


# ---------------------------------------------------------------------------------------------------------------------
# The project folder and the files written by the cells
# ---------------------------------------------------------------------------------------------------------------------
def _temporary_project() -> Path:
    global _SOLUTION_PROJECT
    if _SOLUTION_PROJECT is None or not _SOLUTION_PROJECT.is_dir():
        _SOLUTION_PROJECT = Path(tempfile.mkdtemp(prefix="wb_b7_project_")) / PROJECT_NAME
    return _SOLUTION_PROJECT


def _on_colab() -> bool:
    import importlib.util
    try:
        return "google.colab" in sys.modules or importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):            # no "google" package at all
        return False


def project_dir(kind: str = "learner") -> Path:
    """The folder of the project (created):

    - ``"learner"``: ``<repo>/mon_travail/b7_production/cifar-prod``, the learner's space, when the notebook runs from
      ``mon_travail/``, on Colab (whose current folder is ``/content``) or outside the repository; but a temporary
      folder when it runs from the repository itself outside ``mon_travail/`` (the version of ``chapitres/``, which
      Claude runs to check it: it never writes in ``mon_travail/``); the variable ``WB_B7_PROJECT`` forces a folder;
    - ``"solution"``: a temporary folder ``wb_b7_project_*``, the same for the whole Python session.
    """
    if kind == "learner":
        forced = os.environ.get("WB_B7_PROJECT")
        root = repo_root()
        cwd = Path.cwd().resolve()
        if forced:
            folder = Path(forced)
        elif _inside(cwd, root) and not _inside(cwd, root / "mon_travail") and not _on_colab():
            folder = _temporary_project()
        else:
            folder = root / "mon_travail" / WORK_FOLDER / PROJECT_NAME
    elif kind == "solution":
        folder = _temporary_project()
    else:
        raise ValueError(f"kind must be 'learner' or 'solution', got {kind!r}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve()


def project_line(project: str | Path, kind: str = "learner") -> str:
    """The ℹ️ line that says where the project of the notebook is."""
    if not is_temporary(project):
        return f"ℹ️ ton projet : {project}"
    if kind == "solution":
        return "ℹ️ le projet : un dossier temporaire (le notebook de solutions n'écrit jamais dans le dépôt)."
    return ("ℹ️ ce notebook n'est pas ta copie de mon_travail/ : ton projet va dans un dossier temporaire, effacé à la "
            "fin de la session. Travaille dans ta copie (python tools/start_chapter.py B7).")


def is_temporary(project: str | Path) -> bool:
    """True when the project folder is a temporary one (not the learner's space)."""
    return _inside(Path(project), Path(tempfile.gettempdir())) and not _inside(Path(project), repo_root())


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def check_writable(project: str | Path) -> Path:
    """The project folder, resolved; ValueError if it lies in the repository outside ``mon_travail/`` (the notebooks
    never write in the files of the workbook)."""
    project = Path(project).resolve()
    try:
        root = repo_root()
    except FileNotFoundError:
        return project
    if _inside(project, root) and not _inside(project, root / "mon_travail"):
        raise ValueError(f"{project} is in the repository outside mon_travail/: the notebooks only write in "
                         "mon_travail/ (or in a temporary folder)")
    return project


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest(project: Path) -> dict:
    path = project / MANIFEST
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_file(project: str | Path, rel: str, text: str, *, if_absent: bool = False, overwrite: bool = False,
               quiet: bool = False) -> Path:
    """Write ``text`` into ``project / rel`` (folders created, UTF-8, ``\\n`` line ends) and return the path.

    - ``rel`` is a relative path inside the project (ValueError otherwise);
    - ``if_absent``: an existing file is left as it is (a starting version that a later exercise replaces);
    - a file whose content is not the last one this notebook wrote (changed in an editor) is kept, with one ℹ️ line,
      unless ``overwrite``: what is tested is then the file on the disk.
    """
    project = check_writable(project)
    target = (project / rel)
    if Path(rel).is_absolute() or not _inside(target, project) or target.resolve() == project:
        raise ValueError(f"{rel!r} must be a relative path inside the project {project}")
    if not isinstance(text, str):
        raise TypeError(f"the content of {rel} must be a string, got {type(text).__name__}")
    text = text.replace("\r\n", "\n")
    manifest = _manifest(project)
    if target.exists():
        current = target.read_text(encoding="utf-8", errors="replace")
        if if_absent or current == text:
            return target
        if not overwrite and manifest.get(rel) != _sha(current):
            if not quiet:
                print(f"ℹ️ {rel} a été modifié hors du notebook : je garde le fichier du disque (c'est lui qui est "
                      f"testé). Pour repartir du contenu de la cellule, supprime {target} ou passe overwrite=True.")
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".wb_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, target)
    manifest[rel] = _sha(text)
    (project / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return target


def write_files(project: str | Path, files: dict, **kwargs) -> list[Path]:
    """:func:`write_file` for each ``{relative path: text}`` of ``files``; a value ``...`` (not written yet) is
    skipped."""
    return [write_file(project, rel, text, **kwargs) for rel, text in files.items() if text is not ...]


# ---------------------------------------------------------------------------------------------------------------------
# Running a command
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class RunResult:
    """What a command did: its arguments, exit code (None after a timeout), stdout, stderr, duration in seconds."""

    args: list
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def last_json(self):
        """The last line of stdout read as JSON (None if it is not JSON)."""
        lines = [line for line in self.stdout.strip().splitlines() if line.strip()]
        if not lines:
            return None
        try:
            return json.loads(lines[-1])
        except ValueError:
            return None

    def stderr_lines(self, level: str | None = None) -> list[str]:
        """The lines of stderr (those that contain the word ``level`` if it is given, e.g. "ERROR")."""
        lines = self.stderr.splitlines()
        return [line for line in lines if re.search(rf"\b{level}\b", line)] if level else lines

    def command(self) -> str:
        """The command, shortened: the interpreter becomes ``python`` and absolute paths keep their last part."""
        shown = []
        for a in self.args:
            a = str(a)
            if a == sys.executable:
                shown.append("python")
            elif os.path.isabs(a):
                shown.append(Path(a).name)
            else:
                shown.append(a)
        return " ".join(shown)

    def tail(self, n: int = 3) -> str:
        """The last n lines of stderr, joined by « ⏎ » (for a one-line message)."""
        lines = [line.strip() for line in self.stderr.strip().splitlines() if line.strip()]
        return " ⏎ ".join(lines[-n:]) if lines else "(rien sur stderr)"

    def _ipython_display_(self) -> None:   # a RunResult on the last line of a cell: a short summary
        state = "délai dépassé" if self.timed_out else f"code de sortie {self.returncode}"
        print(f"$ {self.command()}  →  {state} en {self.seconds:.1f} s")


def _pythonpath(src: list) -> str:
    wb_src = str(Path(__file__).resolve().parents[1])
    parts = [str(Path(p).resolve()) for p in src] + [wb_src]
    parts += [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p not in parts]
    return os.pathsep.join(parts)


def run(args, cwd, *, timeout: float = 120, src=(), env: dict | None = None, input: str | None = None) -> RunResult:
    """Run ``args`` (a list; items converted to str) in the folder ``cwd`` (made absolute) and wait at most
    ``timeout`` seconds. The ``PYTHONPATH`` of the process starts with the folders ``src`` (absolute), then the
    ``src`` folder of ``wb``, then the current ``PYTHONPATH``; ``env`` adds or replaces variables. Never raises for a
    failing command: read ``returncode`` (None after a timeout, ``timed_out`` True)."""
    cwd = Path(cwd).resolve()
    args = [str(a) for a in args]
    full_env = {**os.environ, "PYTHONPATH": _pythonpath(list(src)), "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_COLORS": "0", "NO_COLOR": "1", **(env or {})}
    full_env.pop("FORCE_COLOR", None)
    start = time.perf_counter()
    try:
        done = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=full_env, input=input)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return RunResult(args, None, out, err, time.perf_counter() - start, True)
    except OSError as exc:                       # the program does not exist
        return RunResult(args, None, "", f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
    return RunResult(args, done.returncode, done.stdout, done.stderr, time.perf_counter() - start)


def cli(project: str | Path, *args, timeout: float = 120, env: dict | None = None) -> RunResult:
    """``python -m cifarprod <args>`` in the project folder, the package read from ``<project>/src``."""
    project = Path(project).resolve()
    return run([sys.executable, "-m", PACKAGE, *args], project, timeout=timeout, src=[project / "src"], env=env)


class PackageImportError(RuntimeError):
    """A module of the learner's package that does not import (its message says which, and why, in French)."""


def import_package(project: str | Path, name: str = PACKAGE):
    """The package ``name`` of ``<project>/src``, imported afresh (its modules already imported are forgotten, so that
    the files just written are read). Raises what the import raises."""
    import importlib

    src = str((Path(project) / "src").resolve())
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]
    sys.path[:] = [src] + [p for p in sys.path if p != src and not (p and Path(p, name, "__init__.py").is_file())]
    for cache in Path(src, name).rglob("__pycache__"):     # a file rewritten in the same second, with the same size,
        shutil.rmtree(cache, ignore_errors=True)            # would otherwise be read from a stale .pyc
    importlib.invalidate_caches()
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise PackageImportError(f"{name} ne s'importe pas ({type(error).__name__} : {error})") from error
    finally:
        sys.dont_write_bytecode = keep


def import_module(project: str | Path, module: str):
    """``cifarprod.<module>`` of the project, imported afresh with its package (:func:`import_package`). An
    ``ImportError`` (a module of the package not written yet, a name that does not exist) becomes a
    :class:`PackageImportError`, which the check cells report as one ❌ line."""
    import importlib

    import_package(project)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        return importlib.import_module(f"{PACKAGE}.{module}")
    except ImportError as error:
        raise PackageImportError(f"{PACKAGE}.{module} ne s'importe pas ({type(error).__name__} : {error})") from error
    finally:
        sys.dont_write_bytecode = keep


import contextlib  # noqa: E402


@contextlib.contextmanager
def torch_state_kept():
    """Run the learner's code in the notebook without changing the notebook's PyTorch: the deterministic mode, the
    random state (CPU) and the number of threads are put back afterwards, even after an error."""
    import torch

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)


@contextlib.contextmanager
def isolated_logging():
    """Call the learner's ``main()`` in the notebook: the handlers and the level of the root logger are put back
    afterwards (``logging.basicConfig`` would otherwise show every INFO line of every library for the rest of the
    session), and what it writes on stderr is kept in the yielded buffer instead of being shown."""
    import io
    import logging

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            yield buffer
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
        for handler in handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(level)


# ---------------------------------------------------------------------------------------------------------------------
# Editable install in a temporary virtual environment
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Install:
    """An editable install of a project in a temporary virtual environment (:func:`install_editable`)."""

    ok: bool | None                     # None: no installer (neither pip nor uv)
    method: str
    venv: Path | None = None
    python: Path | None = None
    scripts: Path | None = None
    seconds: float = 0.0
    log: str = ""

    def run(self, args, cwd, timeout: float = 120) -> RunResult:
        """A command run with the interpreter of the environment (``args[0] == "python"`` is replaced by it)."""
        args = [str(self.python) if a == "python" else a for a in args]
        env = {"PYTHONPATH": "", "VIRTUAL_ENV": str(self.venv)}
        return run(args, cwd, timeout=timeout, env=env)

    def cleanup(self) -> None:
        if self.venv is not None:
            shutil.rmtree(self.venv, ignore_errors=True)


def _site_dirs() -> list[str]:
    import site
    import sysconfig

    dirs = []
    for d in [*site.getsitepackages(), sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]]:
        if d not in dirs and Path(d).is_dir():
            dirs.append(d)
    user = site.getusersitepackages() if site.ENABLE_USER_SITE else None
    if user and Path(user).is_dir() and user not in dirs:
        dirs.append(user)
    return dirs


def install_editable(project: str | Path, *, timeout: float = 300) -> Install:
    """``pip install --no-deps --no-build-isolation -e <project>`` in a NEW virtual environment (``python -m venv
    --without-pip`` in a temporary folder), which sees the packages of the current one (torch, numpy, setuptools,
    pip) through a ``.pth`` file; uv is used when the current environment has no pip. The current environment is never
    modified. Returns an :class:`Install` (``ok`` False with the log of the installer when it fails)."""
    import importlib.util

    start = time.perf_counter()
    project = Path(project).resolve()
    venv = Path(tempfile.mkdtemp(prefix="wb_b7_venv_"))
    made = run([sys.executable, "-m", "venv", "--without-pip", str(venv)], venv, timeout=120, env={"PYTHONPATH": ""})
    if not made.ok:
        return Install(False, "venv", venv, seconds=time.perf_counter() - start, log=made.stderr)
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    purelib = run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], venv,
                  env={"PYTHONPATH": ""}).stdout.strip()
    lines = [f"import site; site.addsitedir({d!r})" for d in _site_dirs()]
    Path(purelib).mkdir(parents=True, exist_ok=True)
    (Path(purelib) / "zz_wb_parent_env.pth").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if importlib.util.find_spec("pip") is not None:
        method, args = "pip", [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-index",
                               "--disable-pip-version-check", "-q", "-e", project]
    elif shutil.which("uv"):
        method, args = "uv", [shutil.which("uv"), "pip", "install", "--python", python, "--no-deps",
                              "--no-build-isolation", "--offline", "-q", "-e", project]
    else:
        return Install(None, "none", venv, python, scripts, time.perf_counter() - start,
                       "neither pip nor uv in this environment")
    done = run(args, project, timeout=timeout, env={"PYTHONPATH": "", "VIRTUAL_ENV": str(venv)})
    log = (done.stdout + done.stderr).strip() or ("timeout" if done.timed_out else "")
    return Install(done.ok, method, venv, python, scripts, time.perf_counter() - start, log)


# ---------------------------------------------------------------------------------------------------------------------
# Data: the fixed CIFAR-10 subset of the chapter
# ---------------------------------------------------------------------------------------------------------------------
def array_fingerprint(*arrays) -> str:
    """SHA-256 of the dtype, the shape and the bytes of each array, in this order (the function of 207.4)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


_SUBSET: dict | None = None


def cifar_subset() -> dict:
    """The data of the chapter: ``{"x_train", "y_train", "x_val", "y_val"}``, 200 training and 50 validation images
    per class of CIFAR-10 (``wb.datasets.load_cifar10_subset``, seed 2011, disjoint), uint8 (N, 32, 32, 3) and int64
    labels, in the order of the dataset (classes mixed). Read once per session (copies are returned)."""
    global _SUBSET
    if _SUBSET is None:
        cached = _cache_dir() / SUBSET_FILE
        _SUBSET = _read_npz(cached, SUBSET_FINGERPRINT, subset_fingerprint)
        if _SUBSET is None:
            from wb import datasets
            x_train, y_train = datasets.load_cifar10_subset(SUBSET_PER_CLASS["train"], "train")
            x_val, y_val = datasets.load_cifar10_subset(SUBSET_PER_CLASS["val"], "val")
            _SUBSET = {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val}
            if subset_fingerprint(_SUBSET) != SUBSET_FINGERPRINT:
                raise RuntimeError("the CIFAR-10 subset of B7 does not have its recorded fingerprint "
                                   f"{SUBSET_FINGERPRINT[:12]}…: has wb.datasets changed?")
            _write_npz(cached, _SUBSET)
    return {k: v.copy() for k, v in _SUBSET.items()}


def _cache_dir() -> Path:
    from wb import datasets
    return Path(datasets.cache_dir()) / "b7"


def _read_npz(path: Path, want: str, fingerprint) -> dict | None:
    """The arrays of a cached .npz if its fingerprint is ``want``, else None (absent or damaged file)."""
    try:
        with np.load(path) as f:
            arrays = {k: f[k] for k in f.files}
    except Exception:  # noqa: BLE001 - no cache, or a damaged one: computed again
        return None
    return arrays if fingerprint(arrays) == want else None


def _write_npz(path: Path, arrays: dict) -> None:
    """An atomic, uncompressed .npz (a failure to write a cache is not an error)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except OSError:
        pass


def subset_fingerprint(arrays: dict) -> str:
    """The fingerprint of a subset: :func:`array_fingerprint` of x_train, y_train, x_val, y_val."""
    return array_fingerprint(*(arrays[k] for k in ("x_train", "y_train", "x_val", "y_val")))


def write_subset(folder: str | Path) -> Path:
    """``<folder>/cifar10_2000.npz`` (arrays x_train, y_train, x_val, y_val, uncompressed), written if it is absent or
    holds other arrays (atomic write); returns its path."""
    folder = Path(folder)
    check_writable(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SUBSET_FILE
    arrays = cifar_subset()
    want = subset_fingerprint(arrays)
    if path.is_file():
        try:
            with np.load(path) as f:
                if subset_fingerprint({k: f[k] for k in arrays}) == want:
                    return path
        except Exception:  # noqa: BLE001 - a damaged file is rewritten
            pass
    tmp = folder / (SUBSET_FILE + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


def test_images(per_class: int = 100, seed: int = 2011) -> tuple[np.ndarray, np.ndarray]:
    """``(X, y)``: ``per_class`` images of each class from the TEST split of CIFAR-10 (never seen by the models of the
    chapter, nor by the MP5 model), uint8 (N, 32, 32, 3) and int64 labels."""
    key = f"{per_class}-{seed}"
    if key not in _TEST:
        cached = _cache_dir() / f"cifar10_test_{per_class}_{seed}.npz"
        arrays = _read_npz(cached, _TEST_FINGERPRINTS.get(key, ""), lambda a: array_fingerprint(a["x"], a["y"]))
        if arrays is None:
            from wb import datasets
            x, y = datasets.load_cifar10_subset(per_class, "test", seed)
            arrays = {"x": x, "y": y}
            if key in _TEST_FINGERPRINTS and array_fingerprint(x, y) != _TEST_FINGERPRINTS[key]:
                raise RuntimeError(f"the CIFAR-10 test images ({key}) do not have their recorded fingerprint")
            if key in _TEST_FINGERPRINTS:
                _write_npz(cached, arrays)
        _TEST[key] = arrays
    return _TEST[key]["x"].copy(), _TEST[key]["y"].copy()


_TEST: dict = {}
_TEST_FINGERPRINTS = {"100-2011": "ddb4740418bb7c3e82994bcd3c68fedb578596a6ffca88719856ed5aa9360e66",    # (x, y)
                      "20-2011": "97b06a4da7b0821f0727c0dce53e4d7c765ab9a6d390bcc567f7d6342fe2538e"}


# ---------------------------------------------------------------------------------------------------------------------
# Models: the oracle of the architectures, the reference model of the MP5, the learner's MP5 model
# ---------------------------------------------------------------------------------------------------------------------
def build_reference(arch: str = "cnn", widths=(12, 24), n_classes: int = 10):
    """The architectures of the contract, written independently of the learner's package (the oracle of the checks):
    "cnn" (per width: Conv2d 3 x 3 padding 1, ReLU, MaxPool2d(2); Flatten, Linear) and "vgg" (per width: twice Conv2d
    3 x 3 padding 1 without bias, BatchNorm2d, ReLU; MaxPool2d(2) between the stages; AdaptiveAvgPool2d(1), Flatten,
    Linear), as cifarkit.small_cnn and cifarkit.vgg of the MP5."""
    import torch.nn as nn

    layers, c, side = [], 3, 32
    widths = [int(w) for w in widths]
    if arch == "cnn":
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c, side = w, side // 2
        return nn.Sequential(*layers, nn.Flatten(), nn.Linear(c * side * side, n_classes))
    if arch == "vgg":
        for i, w in enumerate(widths):
            for _ in range(2):
                layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU()]
                c = w
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, n_classes))
    raise ValueError(f"unknown arch {arch!r}")


_REFERENCE_STATE: dict | None = None


def reference_folder() -> Path:
    return repo_root() / MP5_FOLDER / "solution"


def reference_model():
    """``(model, info)``: the VGG 16-32-64 delivered by the MP5 solution (``model.pt``, ``model.json``, versioned with
    the repository), a NEW copy at each call, in eval() mode; ``info``: ``source`` ("reference"), ``folder``,
    ``config``, ``mean``, ``std``, ``classes``, ``val_acc``."""
    import torch

    global _REFERENCE_STATE
    folder = reference_folder()
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if _REFERENCE_STATE is None:
        _REFERENCE_STATE = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    config = meta["config"]
    model = build_reference(config["arch"], config["widths"])
    model.load_state_dict(_REFERENCE_STATE)
    model.eval()
    info = {"source": "reference", "folder": folder, "config": config, "mean": tuple(meta["mean"]),
            "std": tuple(meta["std"]), "classes": tuple(meta["classes"]),
            "val_acc": meta.get("metrics", {}).get("val_acc")}
    return model, info


def learner_mp5_folder() -> Path | None:
    """The folder of the learner's delivered MP5 model (``mon_travail/projets/partie_5_classifieur_cifar10`` with
    ``model.pt`` and ``model.json``), or None. Only read, never written."""
    try:
        folder = repo_root() / "mon_travail" / MP5_FOLDER
    except FileNotFoundError:
        return None
    if (folder / "model.pt").is_file() and (folder / "model.json").is_file():
        return folder
    return None


def _learner_mp5_model(folder: Path):
    """The learner's MP5 model, rebuilt by THEIR cifarkit.build_model (imported from their folder, read only)."""
    import importlib

    import torch

    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    saved = list(sys.path)
    for key in [k for k in sys.modules if k == "cifarkit" or k.startswith("cifarkit.")]:
        del sys.modules[key]
    try:
        sys.path.insert(0, str(folder))
        cifarkit = importlib.import_module("cifarkit")
        model = cifarkit.build_model(dict(meta["config"]))
    finally:
        sys.path[:] = saved
        for key in [k for k in sys.modules if k == "cifarkit" or k.startswith("cifarkit.")]:
            del sys.modules[key]
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    info = {"source": "learner", "folder": folder, "config": meta["config"], "mean": tuple(meta["mean"]),
            "std": tuple(meta["std"]), "classes": tuple(meta.get("classes", CLASSES)),
            "val_acc": meta.get("metrics", {}).get("val_acc")}
    return model, info


def base_model(source: str = "auto", *, quiet: bool = False):
    """``(model, info)``, the classifier the chapter serves: ``"learner"`` (the learner's MP5 model, if it exists and
    loads), ``"reference"`` (:func:`reference_model`), ``"auto"`` (the learner's, else the reference, with one ℹ️
    line that says which). The solutions notebook passes ``"reference"``. ``info["source"]`` tells which one came."""
    if source not in ("auto", "learner", "reference"):
        raise ValueError(f"source must be 'auto', 'learner' or 'reference', got {source!r}")
    if source in ("auto", "learner"):
        folder = learner_mp5_folder()
        if folder is not None:
            try:
                model, info = _learner_mp5_model(folder)
                if not quiet:
                    print(f"ℹ️ modèle de base : ton modèle du MP5 ({folder}, config {info['config']}).")
                return model, info
            except Exception as exc:  # noqa: BLE001 - the learner's MP5 code: say it, then fall back
                if source == "learner":
                    raise
                if not quiet:
                    print(f"ℹ️ ton modèle du MP5 ({folder}) ne se recharge pas ({type(exc).__name__}: "
                          f"{str(exc)[:120]}) : le modèle de référence le remplace.")
        elif source == "learner":
            raise FileNotFoundError(f"no delivered MP5 model in mon_travail/{MP5_FOLDER}")
    model, info = reference_model()
    if not quiet:
        print(f"ℹ️ modèle de base : le modèle de référence (le VGG 16-32-64 livré par la solution du MP5, "
              f"accuracy de validation {info['val_acc']}).")
    return model, info


# ---------------------------------------------------------------------------------------------------------------------
# Images of a service
# ---------------------------------------------------------------------------------------------------------------------
def write_png(image_rgb: np.ndarray, path: str | Path) -> Path:
    """Save a uint8 RGB image (H, W, 3) as a PNG file (lossless)."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").save(path)
    return path


def imread_bgr(path: str | Path) -> np.ndarray:
    """An image file read as ``cv2.imread`` reads it: a uint8 array (H, W, 3) with the channels in the order BLUE,
    GREEN, RED (OpenCV is not needed)."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


# ---------------------------------------------------------------------------------------------------------------------
# 🔮 B7.12: a short training whose loader may draw random flips in its workers
# ---------------------------------------------------------------------------------------------------------------------
class FlipDataset:
    """``n`` fixed images (float32 (3, 32, 32), ``np.random.default_rng(0)``) and the labels ``i % 10``; with
    ``augment``, each image is flipped left-right when ``torch.rand(1) < 0.5``, drawn in ``__getitem__`` (in the
    process that reads the item: a worker of the DataLoader, or the notebook itself with ``num_workers=0``)."""

    def __init__(self, n: int = 512, augment: bool = True) -> None:
        import torch

        rng = np.random.default_rng(0)
        self.x = torch.from_numpy(rng.standard_normal((n, 3, 32, 32)).astype(np.float32))
        self.y = torch.arange(n) % 10
        self.augment = augment

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        import torch

        x = self.x[i]
        if self.augment and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        return x, self.y[i]


def workers_run(num_workers: int = 0, seed: int = 0, *, augment: bool = True, n: int = 512, epochs: int = 2,
                batch_size: int = 64, device: str = "cpu") -> dict:
    """The weights (a dict of tensors, on the CPU) after a short training: ``torch.manual_seed(seed)``, a "cnn" (8, 16)
    of :func:`build_reference`, SGD (lr 0.05, momentum 0.9), ``epochs`` epochs over :class:`FlipDataset`, read by a
    ``DataLoader(batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=num_workers)``
    (no ``worker_init_fn``, no persistent workers). The warning of PyTorch about more workers than cores is muted."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    model = build_reference("cnn", (8, 16)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*worker processes in total.*")
        warnings.filterwarnings("ignore", message=r".*use of fork\(\) may lead to deadlocks.*")
        loader = DataLoader(FlipDataset(n, augment), batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=num_workers)
        for _ in range(epochs):
            for x, t in loader:
                optimizer.zero_grad()
                F.cross_entropy(model(x.to(device)), t.to(device)).backward()
                optimizer.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def same_weights(a: dict, b: dict) -> bool:
    """True when two state_dicts have the same names and exactly the same tensors (``torch.equal``)."""
    import torch
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def max_gap(a: dict, b: dict) -> float:
    """The largest absolute difference between two state_dicts of the same names (float tensors)."""
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a if a[k].is_floating_point())


# ---------------------------------------------------------------------------------------------------------------------
# 📦 B7.15: the learner's tests on faulty versions of the reference package
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Mutant:
    key: str
    file: str                          # relative to src/cifarprod
    edits: list                        # [(old, new)]: each old text must be found exactly once
    what: str                          # what is wrong, in French (the family of tests that should see it)
    slow: bool = False                 # seen only by an end-to-end test (marked slow)


MUTANTS = [
    Mutant("scale", "data.py", [(".to(torch.float32) / 255.0", ".to(torch.float32)")],
           "preprocess ne ramène plus les pixels dans [0, 1] avant de les normaliser"),
    Mutant("schema", "data.py", [("    check_images(X, y)\n    return np.ascontiguousarray(X), y.astype(np.int64)",
                                  "    return np.ascontiguousarray(X, dtype=np.float32) / 255, y.astype(np.int64)")],
           "load_npz rend des images en float dans [0, 1] au lieu de uint8, sans vérifier leur schéma"),
    Mutant("overlap", "data.py", [('        X, y = archive[f"x_{part}"], archive[f"y_{part}"]',
                                   '        X, y = archive["x_train"][:len(archive[f"x_{part}"])], archive[f"y_{part}"]')],
           "load_npz(path, \"val\") rend des images d'entraînement : la validation recouvre l'entraînement"),
    Mutant("validation", "config.py", [("        if not 0 < self.lr < 1:", "        if False:")],
           "TrainConfig accepte un learning rate hors de ]0, 1[ (lr = 0 ou lr = 5)"),
    Mutant("exitcode", "cli.py", [('        log.error("invalid configuration: %s", err)\n        return 2',
                                   '        log.error("invalid configuration: %s", err)\n        return 0')],
           "main rend 0 (un succès) quand la configuration est invalide"),
    Mutant("weights", "cli.py", [('        save_file(result["model"].state_dict(), str(weights))\n', ""),
                                 ("artifacts=[artifact(weights)])", "artifacts=[])")],
           "la commande train rend 0 sans écrire model.safetensors", slow=True),
]


def mutant_package(key: str | None, dest: str | Path, base: str | Path | None = None) -> Path:
    """A copy of the reference package (``src/``, ``pyproject.toml``) in ``dest`` with the fault ``key`` of
    :data:`MUTANTS`; ``key=None`` copies it unchanged. Returns ``dest``. ValueError if the fault no longer applies
    (the reference changed)."""
    base = Path(base) if base is not None else reference_package()
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(base / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copy2(base / "pyproject.toml", dest / "pyproject.toml")
    if key is not None:
        mutant = {m.key: m for m in MUTANTS}[key]
        path = dest / "src" / PACKAGE / mutant.file
        text = path.read_text(encoding="utf-8")
        for old, new in mutant.edits:
            if text.count(old) != 1:
                raise ValueError(f"mutant {key!r}: {old!r} is not found once in {mutant.file}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    return dest


_FAILED = re.compile(r"^(FAILED|ERROR) (\S+?)(?: - (.*))?$")


@dataclass
class TestRun:
    """A run of pytest by :func:`run_tests`: the counts, the failures as (name, reason), the RunResult."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failures: list = field(default_factory=list)
    summary: str = ""
    result: RunResult | None = None

    @property
    def ok(self) -> bool:
        return self.passed > 0 and self.failed == 0 and self.errors == 0


TestRun.__test__ = False   # not a test class for pytest, despite its name


def run_tests(package: str | Path, tests: str | Path, *, data: str | Path | None = None, marker: str | None = None,
              maxfail: int | None = None, timeout: float = 300) -> TestRun:
    """Run the test folder ``tests`` (copied into ``<package>/tests``, with the data file ``data`` copied into
    ``<package>/data``) on the package ``<package>/src``: ``python -m pytest tests -q`` in ``package``, with its own
    pyproject.toml (``-c``, ``--rootdir``, ``--strict-markers``), ``-m marker``, ``--maxfail``, ``COLUMNS=1000``."""
    package, tests = Path(package).resolve(), Path(tests).resolve()
    if tests != package / "tests":
        if (package / "tests").exists():
            shutil.rmtree(package / "tests")
        shutil.copytree(tests, package / "tests", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if data is not None:
        (package / "data").mkdir(exist_ok=True)
        target = package / "data" / Path(data).name
        if not target.exists():
            try:
                os.link(data, target)
            except OSError:
                shutil.copy2(data, target)
    args = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--color=no", "-rfE",
            "--tb=no", "--strict-markers", "-c", str(package / "pyproject.toml"), "--rootdir", str(package)]
    if marker:
        args += ["-m", marker]
    if maxfail:
        args += [f"--maxfail={maxfail}"]
    res = run(args, package, timeout=timeout, src=[package / "src"], env={"COLUMNS": "1000"})
    out = TestRun(result=res)
    for line in res.stdout.splitlines():
        m = _FAILED.match(line.strip())
        if m:
            name = m.group(2).split("::", 1)[-1]
            out.failures.append((name, (m.group(3) or "").strip()))
    lines = [line for line in res.stdout.strip().splitlines() if re.search(r"\d+ (passed|failed|error|skipped|"
                                                                           r"deselected)|no tests ran", line)]
    out.summary = lines[-1].strip("= ") if lines else (res.stdout.strip().splitlines() or [res.tail()])[-1][:300]
    for word, attr in (("passed", "passed"), ("failed", "failed"), ("errors?", "errors"), ("skipped", "skipped")):
        setattr(out, attr, sum(int(n) for n in re.findall(rf"(\d+) {word}\b", out.summary)))
    if res.timed_out:
        out.summary = f"timeout after {timeout:.0f} s"
    return out


EXPECTED_WORDS = ("expected", "attendu", "did not raise")


def says_expected(reason: str) -> bool:
    """True when the first line of the reason of a failure says what was expected (« expected », « attendu », or
    pytest.raises' « DID NOT RAISE »)."""
    first = reason.splitlines()[0].lower() if reason else ""
    return any(word in first for word in EXPECTED_WORDS)
