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

import atexit
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
# the training images of the MP5 reference model: the 45 000 images of cifarkit.stratified_split(y, 5000, seed=0) on the
# 50 000 training images of CIFAR-10, and their labels (array_fingerprint; history_chB7.py "mp5data" recomputes it and
# checks that their channel means are those of model.json)
MP5_DATA = {"path": "CIFAR-10 train, MP5 split (stratified, 5 000 for validation, seed 0)", "n_train": 45000,
            "sha256": "b76c6f2c4d0363823f47b8c3ce5216534ffd21bfd08e019a4a92c205c964af7b"}
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
        folder = Path(tempfile.mkdtemp(prefix="wb_b7_project_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)      # removed when the Python session ends
        _SOLUTION_PROJECT = folder / PROJECT_NAME
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
                      f"testé). Pour repartir du contenu de la cellule, supprime {target} : la cellule le réécrira.")
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
    leftover: bool = False              # processes it started were still alive after it ended (they were stopped)

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
        lines = [line.strip() for line in self.stderr.strip().splitlines()
                 if line.strip() and not set(line.strip()) <= set("^~ ")]   # not the markers under a traceback line
        return " ⏎ ".join(lines[-n:]) if lines else "(rien sur stderr)"

    def _ipython_display_(self) -> None:   # a RunResult on the last line of a cell: a short summary
        state = "délai dépassé" if self.timed_out else f"code de sortie {self.returncode}"
        print(f"$ {self.command()}  →  {state} en {self.seconds:.1f} s")


def _pythonpath(src: list) -> str:
    wb_src = str(Path(__file__).resolve().parents[1])
    parts = [str(Path(p).resolve()) for p in src] + [wb_src]
    parts += [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p not in parts]
    return os.pathsep.join(parts)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _stop_group(process: subprocess.Popen) -> bool:
    """Kill the process group of ``process`` (POSIX: its own session); True if something was still alive."""
    if os.name == "nt":                          # no process groups here: the process itself
        alive = process.poll() is None
        process.kill()
        return alive
    if not _group_alive(process.pid):
        return False
    if process.poll() is not None:                # orphans that are ending (zombies) leave the group at once
        time.sleep(0.2)
        if not _group_alive(process.pid):
            return False
    try:
        os.killpg(process.pid, 9)
    except (ProcessLookupError, PermissionError):
        pass
    return True


def run(args, cwd, *, timeout: float = 120, src=(), env: dict | None = None, input: str | None = None,
        grace: float = 2.0) -> RunResult:
    """Run ``args`` (a list; items converted to str) in the folder ``cwd`` (made absolute) and wait at most
    ``timeout`` seconds. The ``PYTHONPATH`` of the process starts with the folders ``src`` (absolute), then the
    ``src`` folder of ``wb``, then the current ``PYTHONPATH``; ``env`` adds or replaces variables. Never raises for a
    failing command: read ``returncode`` (None after a timeout, ``timed_out`` True).

    The command runs in its own session (a new process group): at the timeout, the whole group is killed, not only
    the command; and a command that ends while processes it started are still alive (a server never stopped, which
    may also keep stdout open) does not make the caller wait: after ``grace`` seconds the group is killed and
    ``leftover`` is True."""
    cwd = Path(cwd).resolve()
    args = [str(a) for a in args]
    full_env = {**os.environ, "PYTHONPATH": _pythonpath(list(src)), "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_COLORS": "0", "NO_COLOR": "1", **(env or {})}
    full_env.pop("FORCE_COLOR", None)
    start = time.perf_counter()
    try:
        process = subprocess.Popen(args, cwd=cwd, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                   errors="replace", env=full_env, start_new_session=os.name != "nt")
    except OSError as exc:                       # the program does not exist
        return RunResult(args, None, "", f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
    deadline, ended_at, timed_out, leftover = start + timeout, None, False, False
    pending_input = input
    while True:
        now = time.perf_counter()
        try:
            out, err = process.communicate(input=pending_input, timeout=max(0.01, min(0.5, deadline - now)))
            break
        except subprocess.TimeoutExpired:
            pending_input = None                 # sent once: communicate() goes on from where it was
            now = time.perf_counter()
            if process.poll() is not None:       # the command has ended, but its pipes are still open
                ended_at = ended_at if ended_at is not None else now
                stop = now - ended_at >= grace
                leftover = leftover or stop
            else:
                stop = now >= deadline
                timed_out = timed_out or stop
            if stop:
                _stop_group(process)
                try:
                    out, err = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:   # a process outside the group holds the pipes: give up on them
                    out, err = "", ""
                break
    if _stop_group(process):                     # ended with its pipes closed, but something it started lives on
        leftover = True
    process.wait()
    code = None if timed_out else process.returncode
    return RunResult(args, code, out or "", err or "", time.perf_counter() - start, timed_out, leftover)


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
            "val_acc": meta.get("metrics", {}).get("val_acc"), "data": dict(MP5_DATA)}
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
    Mutant("weights", "cli.py", [('        save_artifact(out, result["model"], arch=config.arch, widths=config.widths, '
                                  'mean=result["mean"],\n', '        dict(out=out, arch=config.arch, widths=config.widths, '
                                  'mean=result["mean"],\n'),
                                 ("artifacts=[artifact(weights), artifact(meta)])", "artifacts=[])")],
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


# ---------------------------------------------------------------------------------------------------------------------
# Part B: a server in another process, the reference artifact, latencies
# ---------------------------------------------------------------------------------------------------------------------
def free_port(host: str = "127.0.0.1") -> int:
    """A TCP port free right now on ``host`` (the system picks it)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    """True when something accepts TCP connections on ``host:port``."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@dataclass
class ServerProcess:
    """A server started by :func:`serve_process`: its base URL, its port and its process."""

    url: str
    port: int
    process: subprocess.Popen
    log: Path

    def stderr(self) -> str:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


@contextlib.contextmanager
def serve_process(src, artifact, *, backend: str = "auto", timeout: float = 60, env: dict | None = None,
                  module: str = f"{PACKAGE}.serve"):
    """``python -m cifarprod.serve --artifact <artifact> --port <free port> --backend <backend>`` in ANOTHER process,
    the package read from the folder ``src``; waits until ``GET /health`` answers 200 (RuntimeError with the end of
    its stderr if it stops or does not answer within ``timeout`` seconds), yields a :class:`ServerProcess`, and
    ALWAYS stops the process at the end (terminate, then kill after 10 s), even after an error."""
    import urllib.request

    port = free_port()
    log = Path(tempfile.mkdtemp(prefix="wb_b7_server_")) / "stderr.txt"
    full_env = {**os.environ, "PYTHONPATH": _pythonpath([src]), "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1",
                "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    args = [sys.executable, "-m", module, "--artifact", str(Path(artifact).resolve()), "--port", str(port),
            "--backend", backend]
    with log.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(args, cwd=str(Path(artifact).resolve().parent), env=full_env,
                                   stdout=subprocess.DEVNULL, stderr=err)
    server = ServerProcess(f"http://127.0.0.1:{port}", port, process, log)
    try:
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"the server stopped with the exit code {process.returncode}: "
                                   f"{server.stderr().strip()[-600:]}")
            try:
                with urllib.request.urlopen(f"{server.url}/health", timeout=2) as answer:
                    if answer.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"the server did not answer /health within {timeout:.0f} s: "
                                   f"{server.stderr().strip()[-600:]}")
            time.sleep(0.1)
        yield server
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)
        shutil.rmtree(log.parent, ignore_errors=True)


def reference_artifact(folder) -> Path:
    """The artifact of the reference model (the VGG of the MP5, :func:`reference_model`) written in ``folder`` by the
    ``save_artifact`` of the reference package, in another process (the notebook's own ``cifarprod`` is untouched):
    model.safetensors and meta.json, with the data fingerprint :data:`MP5_DATA` and its validation accuracy."""
    folder = Path(folder).resolve()
    code = ("import sys\nfrom wb import prod as wbp\nfrom cifarprod.artifact import save_artifact\n"
            "model, info = wbp.reference_model()\n"
            "save_artifact(sys.argv[1], model, arch=info['config']['arch'], widths=info['config']['widths'], "
            "mean=info['mean'], std=info['std'], classes=info['classes'], data=info['data'], "
            "metrics={'val_acc': info['val_acc']})\n")
    done = run([sys.executable, "-c", code, folder], folder.parent if folder.parent.exists() else Path.cwd(),
               src=[reference_package() / "src"], timeout=300)
    if not done.ok:
        raise RuntimeError(f"the reference artifact was not written: {done.tail(5)}")
    return folder


def latencies(call, n: int = 200, warmup: int = 20) -> np.ndarray:
    """The durations (ms) of ``n`` calls of ``call()``, after ``warmup`` calls that are not measured (207.7)."""
    for _ in range(warmup):
        call()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return 1000 * np.array(times)


# ---------------------------------------------------------------------------------------------------------------------
# Part C: reading a workflow of GitHub Actions and a Dockerfile (never run), the chain of the challenge
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Review:
    """One point of a review: its key, whether it holds, and the message for each case (in French)."""

    key: str
    ok: bool
    success: str
    failure: str


SECRET_WORDS = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY)", re.IGNORECASE)


def _bash_syntax(script: str) -> str | None:
    """None if ``bash -n`` accepts the script (or if bash is not there), else its message. Nothing is run."""
    bash = shutil.which("bash")
    if bash is None:
        return None
    done = subprocess.run([bash, "-n"], input=script, capture_output=True, text=True, timeout=30)
    return None if done.returncode == 0 else (done.stderr.strip() or f"code {done.returncode}")


def _workflow_steps(doc: dict) -> list[dict]:
    steps = []
    for job in (doc.get("jobs") or {}).values():
        if isinstance(job, dict):
            steps += [s for s in job.get("steps") or [] if isinstance(s, dict)]
    return steps


def review_workflow(text: str, project: str | Path | None = None) -> list[Review]:
    """The checks of a GitHub Actions workflow (📦 B7.23), read as YAML and never run: triggers, an Ubuntu job, the
    checkout and setup-python actions (a version written as a string, the pip cache), torch from the CPU index, the
    package installed with its extra dev, pytest, a smoke training on synthetic data after the tests, no failure
    swallowed, no secret in clear, ``bash -n`` on each ``run``, the files it names present in ``project``."""
    import yaml

    out: list[Review] = []

    def add(key, ok, success, failure):
        out.append(Review(key, bool(ok), success, failure))

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as err:
        add("yaml", False, "", f"le fichier n'est pas du YAML valide : {str(err).splitlines()[0]}")
        return out
    if not isinstance(doc, dict):
        add("yaml", False, "", f"le fichier doit décrire un dictionnaire (name, on, jobs), reçu {type(doc).__name__}")
        return out
    add("yaml", True, "YAML valide.", "")
    trigger = doc.get("on", doc.get(True))            # PyYAML 6 reads the key `on:` as the boolean True (YAML 1.1)
    if isinstance(trigger, str):
        events = {trigger}
    elif isinstance(trigger, (list, dict)):
        events = set(trigger)
    else:
        events = set()
    add("on", {"push", "pull_request"} <= events, "déclenché à chaque push et pull request (PyYAML lit la clé on: "
        "comme True : le vérificateur le sait).", f"attendu les déclencheurs push et pull_request, reçu {sorted(map(str, events))}")
    jobs = doc.get("jobs") if isinstance(doc.get("jobs"), dict) else {}
    ubuntu = [name for name, job in jobs.items() if isinstance(job, dict) and "ubuntu" in str(job.get("runs-on", ""))]
    add("job", bool(ubuntu), f"un job sur Ubuntu ({', '.join(ubuntu)}).", "attendu au moins un job avec runs-on: "
        "ubuntu-latest")
    steps = _workflow_steps(doc)
    runs = [(k, str(s.get("run", ""))) for k, s in enumerate(steps) if s.get("run")]
    uses = [(k, str(s.get("uses", "")), s.get("with") or {}) for k, s in enumerate(steps) if s.get("uses")]

    def find(pattern):
        """(step, offset) of the first match of ``pattern`` in the run blocks, in their order; None if none."""
        for k, text in runs:
            m = re.search(pattern, text)
            if m:
                return (k, m.start())
        return None

    checkout = [u for _, u, _ in uses if u.startswith("actions/checkout@")]
    major = re.match(r"actions/checkout@v(\d+)", checkout[0]) if checkout else None
    add("checkout", major and int(major.group(1)) >= 4, f"{checkout[0] if checkout else ''} récupère le code.",
        "attendu une étape uses: actions/checkout@v7 (la version de la fiche ; v4 au moins)")
    setup = [(u, w) for _, u, w in uses if u.startswith("actions/setup-python@")]
    version = setup[0][1].get("python-version") if setup else None
    add("python", setup and isinstance(version, str) and re.fullmatch(r"3\.1[1-9]", version),
        f"{setup[0][0] if setup else ''} avec python-version \"{version}\" (une chaîne).",
        "attendu une étape uses: actions/setup-python@v7 avec python-version: \"3.13\" entre guillemets"
        + (f" ; reçu {version!r} ({type(version).__name__}) : sans guillemets, YAML lit 3.10 comme le nombre 3.1"
           if setup and not isinstance(version, str) else ""))
    add("cache", setup and setup[0][1].get("cache") == "pip", "cache: pip (les paquets téléchargés sont gardés d'un "
        "run à l'autre).", "attendu cache: pip dans l'étape setup-python")
    torch_at = find(r"pip install[^\n]*\btorch\b[^\n]*download\.pytorch\.org/whl/cpu|pip install[^\n]*download\.pytorch"
                    r"\.org/whl/cpu[^\n]*\btorch\b")
    add("torch", torch_at is not None, "torch installé depuis l'index CPU de PyTorch.",
        "attendu une commande pip install torch ... --index-url https://download.pytorch.org/whl/cpu (sans l'index "
        "CPU, pip télécharge la version CUDA, plus de 2 Go)")
    install_at = find(r"pip install[^\n]*-e[^\n]*\[[^\]]*dev")
    add("install", install_at is not None and torch_at is not None and install_at > torch_at,
        "le paquet installé en mode éditable avec son extra dev, après torch.",
        "attendu pip install -e \".[dev]\" (l'extra dev apporte pytest), APRÈS l'installation de torch depuis l'index "
        "CPU (sinon pip prend torch sur PyPI)")
    test_at = find(r"\bpytest\b")
    test_text = dict(runs).get(test_at[0], "") if test_at else ""
    add("pytest", test_at is not None and install_at is not None and test_at > install_at and "not slow" not in test_text,
        "pytest après l'installation, les tests slow compris.",
        "attendu une étape run: pytest -q après l'installation, sans -m \"not slow\" (l'intégration continue lance "
        "aussi l'entraînement fumée marqué slow)")
    smoke_at = find(r"(?:\bcifarprod|-m cifarprod) train\b[^\n]*configs/smoke\.yaml|(?:\bcifarprod|-m cifarprod) train\b"
                    r"[^\n]*\n?[^\n]*configs/smoke\.yaml")
    synth_at = find(r"synthetic")
    smoke_text = dict(runs).get(smoke_at[0], "") if smoke_at else ""
    data_set = re.search(r"--set\s+[\"']?data=", smoke_text) is not None      # --set data="$X" or --set "data=$X"
    add("smoke", smoke_at is not None and test_at is not None and smoke_at > test_at and synth_at is not None
        and synth_at < smoke_at and data_set,
        "un entraînement fumée (configs/smoke.yaml) sur des données synthétiques, après les tests.",
        "attendu, après pytest, une étape qui écrit des données synthétiques (scripts/make_synthetic.py) puis "
        "cifarprod train --config configs/smoke.yaml --set data=<ce fichier> (les données du projet ne sont pas dans "
        "git) ; " + ("aucune étape run: ne lance cifarprod train avec configs/smoke.yaml" if smoke_at is None else
                     "aucune étape run: ne lance pytest avant lui" if test_at is None or smoke_at < test_at else
                     "aucune commande avant lui n'écrit de données synthétiques" if synth_at is None or synth_at > smoke_at
                     else f"son --set data=... est introuvable dans « {' '.join(smoke_text.split())[:160]} »"))
    swallowed = [k for k, t in runs if re.search(r"\|\|\s*(true|:|exit 0)\b", t)] + \
        [k for k, s in enumerate(steps) if s.get("continue-on-error") is True]
    add("failures", not swallowed, "aucune erreur avalée (pas de || true ni de continue-on-error).",
        f"l'étape {swallowed[0] + 1 if swallowed else ''} avale son échec (|| true ou continue-on-error) : une "
        "intégration continue qui ne peut pas échouer ne vérifie rien")
    clear = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (str, int, float)) and SECRET_WORDS.search(str(key)) and "${{" not in str(value):
                    clear.append(f"{key}: {value}")
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            clear.extend(m.group(0) for m in re.finditer(r"\b\w*(?:TOKEN|SECRET|PASSWORD|API_?KEY)\w*=(?!\$)\S+", node))

    walk(doc)
    add("secrets", not clear, "aucun secret écrit en clair (un secret passe par ${{ secrets.NOM }}).",
        f"un secret semble écrit en clair ({clear[0][:40] if clear else ''}) : il passe par ${{{{ secrets.NOM }}}}")
    bad = [(k, msg) for k, t in runs if (msg := _bash_syntax(t))]
    add("bash", not bad, "chaque bloc run: est du bash valide (bash -n, rien n'est lancé).",
        f"l'étape {bad[0][0] + 1 if bad else ''} n'est pas du bash valide : {bad[0][1][:160] if bad else ''}")
    if project is not None:
        named = sorted({m for _, t in runs for m in re.findall(r"(?:configs|scripts)/[\w./-]+", t)})
        missing = [m for m in named if not (Path(project) / m).exists()]
        add("files", not missing, f"les fichiers cités existent dans le projet ({', '.join(named) or 'aucun'}).",
            f"le workflow cite des fichiers absents du projet : {', '.join(missing)}")
    return out


def dockerfile_instructions(text: str) -> list[tuple[str, str]]:
    """The instructions of a Dockerfile as (KEYWORD, arguments): comments dropped, lines ending with a backslash
    joined to the next one."""
    out, current = [], ""
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not current and (not line or line.startswith("#")):
            continue
        if current and line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1].rstrip() + " "
            continue
        current += line
        if current.strip():
            keyword, _, args = current.strip().partition(" ")
            out.append((keyword.upper(), args.strip()))
        current = ""
    if current.strip():
        keyword, _, args = current.strip().partition(" ")
        out.append((keyword.upper(), args.strip()))
    return out


def _requirements(text: str) -> dict:
    """{name (lower case): the version pinned with ==, or None} of a requirements file."""
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(==\s*([^\s;]+))?", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(3)
    return out


def _exec_form(args: str):
    try:
        value = json.loads(args)
    except ValueError:
        return None
    return value if isinstance(value, list) and all(isinstance(v, str) for v in value) else None


def review_dockerfile(dockerfile: str, dockerignore: str, requirements: str,
                      project: str | Path | None = None) -> list[Review]:
    """The checks of a Dockerfile (📦 B7.24), read as text and never built: a pinned slim base, settings in ENV and no
    secret, a WORKDIR, the dependencies copied and installed BEFORE the code (cache of the layers), torch from the CPU
    index, a requirements file pinned to the versions of this environment (without torch), the package and the
    artifact copied, MODEL_DIR on the artifact, a user without privileges, EXPOSE and a CMD in exec form on 0.0.0.0
    and the same port, a .dockerignore; with ``project``, the command of CMD is tried (``--help``) in the package."""
    from importlib import metadata

    out: list[Review] = []

    def add(key, ok, success, failure):
        out.append(Review(key, bool(ok), success, failure))

    ins = dockerfile_instructions(dockerfile)
    words = [k for k, _ in ins]

    def where(keyword, predicate=lambda a: True):
        return [i for i, (k, a) in enumerate(ins) if k == keyword and predicate(a)]

    base = ins[0][1] if ins and ins[0][0] == "FROM" else ""
    pinned = re.fullmatch(r"python:3\.1[1-9]\.\d+-slim(-[a-z]+)?(@sha256:[0-9a-f]{64})?|python:[\w.-]+@sha256:[0-9a-f]{64}",
                          base.split(" AS ")[0].split(" as ")[0].strip())
    add("from", pinned, f"FROM {base} : une image slim, à une version précise.",
        f"FROM {base or '(absent en tête)'} : attendu une image slim de Python à une version précise "
        "(python:3.13.18-slim-trixie, ou un digest @sha256:...) ; un tag comme 3.13-slim ou latest bouge")
    envs = {}
    for i in where("ENV"):
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\"[^\"]*\"|\S+)", ins[i][1]):
            envs[m.group(1)] = m.group(2).strip('"')
        if "=" not in ins[i][1] and ins[i][1].split():          # the old form ENV NAME value
            envs[ins[i][1].split()[0]] = " ".join(ins[i][1].split()[1:])
    add("unbuffered", envs.get("PYTHONUNBUFFERED") == "1", "ENV PYTHONUNBUFFERED=1 (les logs sortent tout de suite).",
        "attendu ENV PYTHONUNBUFFERED=1 : sans lui, Python garde ses messages en mémoire et docker logs reste muet")
    secrets = [name for name in envs if SECRET_WORDS.search(name)]
    secrets += [f"ARG {a.split('=')[0]}" for i, (k, a) in enumerate(ins) if k == "ARG" and SECRET_WORDS.search(a)]
    add("secrets", not secrets, "aucun secret dans l'image (ni ENV ni ARG).",
        f"{', '.join(secrets)} : un secret écrit dans le Dockerfile reste lisible dans l'image (docker history) ; il "
        "arrive au lancement par une variable d'environnement (docker run -e) ou un gestionnaire de secrets")
    workdir = [ins[i][1] for i in where("WORKDIR")]
    add("workdir", workdir and workdir[-1].rstrip("/") not in ("",), f"WORKDIR {workdir[-1] if workdir else ''}.",
        "attendu un WORKDIR à soi (/app), pas la racine")
    copies = [(i, ins[i][1].split()) for i in where("COPY")]
    req_copy = next((i for i, a in copies if any("requirements" in x for x in a[:-1])), None)
    code_copy = next((i for i, a in copies if any(x.rstrip("/") in ("src", ".") for x in a[:-1])), None)
    req_install = next((i for i in where("RUN", lambda a: re.search(r"pip install[^&|;]*-r\s+\S*requirements", a))), None)
    add("layers", req_copy is not None and req_install is not None and code_copy is not None
        and req_copy < req_install < code_copy,
        "les dépendances sont copiées et installées AVANT le code : un changement du code ne réinstalle rien.",
        "attendu COPY du fichier requirements, puis RUN pip install -r ..., puis seulement COPY src/ (Docker garde une "
        "couche en cache tant qu'elle et les précédentes ne changent pas)")
    torch_run = next((ins[i][1] for i in where("RUN", lambda a: re.search(r"\btorch\b", a) is not None)), "")
    add("torch", "download.pytorch.org/whl/cpu" in torch_run and re.search(r"torch==\d", torch_run),
        "torch, à une version figée, depuis l'index CPU de PyTorch.",
        "attendu RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu (la version CUDA et "
        "ses bibliothèques NVIDIA pèsent plusieurs gigaoctets)")
    reqs = _requirements(requirements if isinstance(requirements, str) else "")
    loose = [name for name, version in reqs.items() if not version]
    wrong = []
    for name, version in reqs.items():
        if version:
            try:
                have = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
            if have.split("+")[0] != version:
                wrong.append(f"{name}=={version} (ici {have})")
    needed = {"fastapi", "uvicorn", "pydantic", "numpy", "safetensors", "pyyaml"}
    add("requirements", reqs and not loose and not wrong and needed <= set(reqs) and "torch" not in reqs,
        f"requirements : {len(reqs)} paquets figés aux versions de cet environnement, torch à part.",
        "attendu dans le fichier requirements les dépendances de l'API, chacune figée avec == à la version de cet "
        f"environnement, sans torch (installé à part depuis l'index CPU) ; "
        + "; ".join(part for part in [f"sans version : {', '.join(loose)}" if loose else "",
                                      f"autres versions : {', '.join(wrong)}" if wrong else "",
                                      f"manquent : {', '.join(sorted(needed - set(reqs)))}" if needed - set(reqs) else "",
                                      "torch y est" if "torch" in reqs else ""] if part))
    package = next((i for i in where("RUN", lambda a: re.search(r"pip install[^&|;]*(\s\.|\s-e\s+\.)(\s|$)", a + " "))), None)
    add("package", package is not None and code_copy is not None and package > code_copy,
        "le paquet installé après la copie de son code.", "attendu, après COPY src/, RUN pip install --no-deps . "
        "(le paquet lui-même ; ses dépendances sont déjà là)")
    artifact_copy = next(((i, a) for i, a in copies if any("artifact" in x for x in a[:-1])), None)
    model_dir = envs.get("MODEL_DIR", "")
    destination = ""
    if artifact_copy:
        destination = artifact_copy[1][-1].rstrip("/")
        if not destination.startswith("/") and workdir:
            destination = workdir[-1].rstrip("/") + "/" + destination.lstrip("./")
    add("artifact", artifact_copy is not None and model_dir.rstrip("/") == destination,
        f"l'artefact copié dans {destination}, et ENV MODEL_DIR={model_dir} le désigne.",
        f"attendu COPY artifact/ artifact/ et ENV MODEL_DIR sur ce dossier (ici MODEL_DIR={model_dir or 'absent'}, "
        f"artefact copié dans {destination or 'nulle part'}) : le service lit son modèle par cette variable")
    users = [(i, ins[i][1].split(":")[0].strip()) for i in where("USER")]
    created = any(re.search(r"\b(useradd|adduser)\b", ins[i][1]) for i in where("RUN") if not users or i < users[-1][0])
    cmd = where("CMD") + where("ENTRYPOINT")
    add("user", users and users[-1][1] not in ("root", "0") and created and (not cmd or users[-1][0] < cmd[-1]),
        f"USER {users[-1][1] if users else ''} : le serveur tourne sans privilèges.",
        "attendu un utilisateur créé (RUN useradd ... ou adduser) puis USER <cet utilisateur> avant CMD : un "
        "conteneur lancé en root donne ses droits à qui exploite une faille de l'API")
    exposed = [ins[i][1].split("/")[0].strip() for i in where("EXPOSE")]
    command = _exec_form(ins[cmd[-1]][1]) if cmd else None
    import shlex
    try:                                                   # the words of CMD, exec form or shell form
        words = command if command is not None else (shlex.split(ins[cmd[-1]][1]) if cmd else [])
    except ValueError:
        words = []
    port = None
    if "--port" in words and words.index("--port") + 1 < len(words):
        port = words[words.index("--port") + 1]
    add("cmd", command is not None, f"CMD en forme exec : {command}.",
        f"attendu CMD en forme exec, une liste JSON (CMD [\"python\", \"-m\", ...]) : en forme shell, le serveur ne "
        f"reçoit pas les signaux d'arrêt ; reçu {ins[cmd[-1]][1] if cmd else 'aucun CMD'}")
    add("host", "0.0.0.0" in words, "le serveur écoute 0.0.0.0 (joignable hors du conteneur).",
        "attendu --host 0.0.0.0 dans CMD : un serveur qui n'écoute que 127.0.0.1 est injoignable hors du conteneur")
    add("port", exposed and port is not None and port in exposed, f"EXPOSE {exposed[0] if exposed else ''} et --port "
        f"{port} : le même port.", f"attendu EXPOSE <port> et --port <le même> dans CMD (EXPOSE {exposed or 'absent'}, "
        f"--port {port or 'absent'})")
    ignore = [line.strip().rstrip("/") for line in (dockerignore or "").splitlines() if line.strip() and not line.startswith("#")]
    def ignored(name):
        return any(p.lstrip("*/").rstrip("*") in (name, name + "/") or p == name or p.endswith(name) for p in ignore)
    lacking = [name for name in (".git", "data", "runs", "__pycache__", ".env") if not ignored(name)]
    add("dockerignore", not lacking, ".dockerignore : .git, data, runs, __pycache__ et .env restent hors de l'image.",
        f".dockerignore : il manque {', '.join(lacking)} (sinon COPY envoie au build les données, l'historique, les "
        "caches, et peut-être un .env avec des secrets)")
    if project is not None and command:
        args = None
        if command[:2] == ["python", "-m"] and len(command) > 2:
            args = [sys.executable, "-m", command[2], "--help"]
        elif command[0] == "uvicorn":
            target = next((c for c in command[1:] if ":" in c and not c.startswith("-")), None)
            if target:
                module, _, attr = target.partition(":")
                args = [sys.executable, "-c", f"import {module} as m; assert callable(getattr(m, {attr!r}, None)) or "
                        f"hasattr(m, {attr!r})"]
        if args:
            done = run(args, project, src=[Path(project) / "src"], timeout=120)
            add("runs", done.ok, f"la commande de CMD existe dans ton paquet ({' '.join(command[:3])} ... --help : code 0).",
                f"la commande de CMD ne répond pas dans ton paquet ({done.tail(2)[:200]})")
    return out


# imports that need a distribution which the package that provides them does not require
_IMPORT_NEEDS = {"fastapi.testclient": "httpx", "starlette.testclient": "httpx"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def undeclared_imports(project: str | Path, extras=("dev", "serve"), folders=("tests", "scripts")) -> list[str]:
    """The imports of ``tests/`` and ``scripts/`` (``"requests (scripts/demo.py)"``) that ``pip install -e
    ".[dev,serve]"`` of ``project`` would not bring on a new machine: their distribution is not in the closure of
    the dependencies and of the ``extras`` declared in ``pyproject.toml`` (read through the metadata of this
    environment, without network). The standard library and the package itself are left out."""
    import ast
    import tomllib
    from importlib import metadata

    from packaging.requirements import Requirement

    project = Path(project)
    try:
        table = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
    except (OSError, ValueError):
        return []
    own = _norm(str(table.get("name", "")))
    todo = [Requirement(r).name for r in table.get("dependencies", [])]
    todo += [Requirement(r).name for e in extras for r in table.get("optional-dependencies", {}).get(e, [])]
    seen: set[str] = set()
    while todo:
        name = _norm(todo.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            needs = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        for text in needs:
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                todo.append(req.name)
    owners = metadata.packages_distributions()
    missing = set()
    for folder in folders:
        for path in sorted((project / folder).rglob("*.py")) if (project / folder).is_dir() else []:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    top = module.split(".")[0]
                    if top in sys.stdlib_module_names or _norm(top) == own or top == PACKAGE:
                        continue
                    dists = {_norm(d) for d in owners.get(top, [])}
                    rel = path.relative_to(project).as_posix()
                    if dists and not dists & seen:
                        missing.add(f"{top} ({rel})")
                    extra = next((need for prefix, need in _IMPORT_NEEDS.items()
                                  if module == prefix or module.startswith(prefix + ".")), None)
                    if extra and _norm(extra) not in seen:
                        missing.add(f"{extra}, pour {module} ({rel})")
    return sorted(missing)


def is_git_repository(project: str | Path) -> bool:
    """True when ``project`` is the top folder of a git repository (git must be installed)."""
    if shutil.which("git") is None:
        return False
    done = run(["git", "rev-parse", "--show-toplevel"], project, timeout=30)
    return done.ok and Path(done.stdout.strip()).resolve() == Path(project).resolve()


COPY_IGNORED = ("data", "runs", "__pycache__", "*.egg-info", ".pytest_cache", ".git", ".venv", MANIFEST, "*.pyc")


def clone_project(project: str | Path, dest: str | Path) -> tuple[Path, str, list[str]]:
    """A fresh copy of the project in ``dest``, as a new machine would get it: ``git clone`` when the project is a git
    repository (only what is committed), else a copy without data/, runs/, caches and .git. Returns (dest, how, notes):
    ``how`` is "git clone" or "copie", ``notes`` the files changed but not committed (git status --porcelain)."""
    project, dest = Path(project).resolve(), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    if is_git_repository(project):
        done = run(["git", "clone", "--quiet", str(project), str(dest)], project.parent, timeout=120)
        if not done.ok:
            raise RuntimeError(f"git clone failed: {done.tail(3)}")
        status = run(["git", "status", "--porcelain"], project, timeout=30).stdout.splitlines()
        return dest, "git clone", [line[3:] for line in status if line.strip()]
    shutil.copytree(project, dest, ignore=shutil.ignore_patterns(*COPY_IGNORED))
    return dest, "copie", []


def machine_speed(repeats: int = 5) -> float:
    """The speed of this machine, now: the median duration (s) of a fixed piece of work (8 SGD steps of the CNN
    (16, 32) on a batch of 128 random images, a new model each time), after one round not measured. PyTorch's global
    random state is not touched."""
    import torch
    import torch.nn.functional as F

    times = []
    with torch.random.fork_rng():
        for k in range(repeats + 1):
            torch.manual_seed(k)
            start = time.perf_counter()
            model = build_reference("cnn", (16, 32))
            opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            x, y = torch.randn(128, 3, 32, 32), torch.arange(128) % 10
            for _ in range(8):
                opt.zero_grad()
                F.cross_entropy(model(x), y).backward()
                opt.step()
            times.append(time.perf_counter() - start)
    return float(np.median(times[1:]))
