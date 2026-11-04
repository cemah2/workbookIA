"""Tests of wb.prod (the infrastructure of bonus chapter B7) and of its reference package solutions/b7_cifarprod."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from wb import prod as wbp

ROOT = Path(__file__).resolve().parents[2]
CACHED_SUBSET = (wbp._cache_dir() / wbp.SUBSET_FILE).is_file()
needs_subset = pytest.mark.skipif(not CACHED_SUBSET, reason="the CIFAR-10 subset of B7 is not cached here")


# ---------------------------------------------------------------------------------------------------------------------
# The project folder and the files written by the cells
# ---------------------------------------------------------------------------------------------------------------------
def test_solution_project_is_temporary_and_the_same_for_the_session():
    first, second = wbp.project_dir("solution"), wbp.project_dir("solution")
    assert first == second and first.is_dir() and wbp.is_temporary(first)
    assert first.name == wbp.PROJECT_NAME
    assert "dossier temporaire" in wbp.project_line(first, "solution")


def test_learner_project_run_from_the_repository_never_goes_into_mon_travail(monkeypatch):
    monkeypatch.delenv("WB_B7_PROJECT", raising=False)
    monkeypatch.chdir(ROOT / "chapitres")
    folder = wbp.project_dir("learner")
    assert wbp.is_temporary(folder), folder
    assert "n'est pas ta copie" in wbp.project_line(folder, "learner")


def test_learner_project_can_be_forced(tmp_path, monkeypatch):
    monkeypatch.setenv("WB_B7_PROJECT", str(tmp_path / "p"))
    assert wbp.project_dir("learner") == (tmp_path / "p").resolve()


def test_project_dir_refuses_another_kind():
    with pytest.raises(ValueError):
        wbp.project_dir("teacher")


def test_write_file_writes_inside_the_project_only(tmp_path):
    path = wbp.write_file(tmp_path, "src/pkg/a.py", "x = 1\r\n")
    assert path.read_text(encoding="utf-8") == "x = 1\n"
    for rel in ("../escape.py", "/etc/passwd", "."):
        with pytest.raises(ValueError):
            wbp.write_file(tmp_path, rel, "x")
    with pytest.raises(TypeError):
        wbp.write_file(tmp_path, "b.py", ...)


def test_write_file_never_writes_in_the_repository_outside_mon_travail():
    with pytest.raises(ValueError):
        wbp.check_writable(ROOT / "chapitres" / "b7_production" / "cifar-prod")
    assert wbp.check_writable(ROOT / "mon_travail" / "b7_production") == (ROOT / "mon_travail" / "b7_production").resolve()


def test_write_file_keeps_a_file_changed_outside_the_notebook(tmp_path, capsys):
    wbp.write_file(tmp_path, "a.py", "v = 1\n")
    wbp.write_file(tmp_path, "a.py", "v = 2\n")                 # written by the notebook: replaced
    assert (tmp_path / "a.py").read_text() == "v = 2\n"
    (tmp_path / "a.py").write_text("v = 'mine'\n")             # changed in an editor
    wbp.write_file(tmp_path, "a.py", "v = 3\n")
    assert (tmp_path / "a.py").read_text() == "v = 'mine'\n"
    assert "modifié hors du notebook" in capsys.readouterr().out
    wbp.write_file(tmp_path, "a.py", "v = 3\n", overwrite=True)
    assert (tmp_path / "a.py").read_text() == "v = 3\n"
    wbp.write_file(tmp_path, "a.py", "v = 4\n", if_absent=True)
    assert (tmp_path / "a.py").read_text() == "v = 3\n"


def test_write_files_skips_what_is_not_written_yet(tmp_path):
    written = wbp.write_files(tmp_path, {"a.txt": "a", "b.txt": ...})
    assert [p.name for p in written] == ["a.txt"] and not (tmp_path / "b.txt").exists()


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------
def test_run_captures_the_exit_code_the_streams_and_puts_src_first(tmp_path):
    (tmp_path / "src" / "pkg_b7_run").mkdir(parents=True)
    (tmp_path / "src" / "pkg_b7_run" / "__init__.py").write_text("NAME = 'mine'\n")
    code = ("import sys, os, pkg_b7_run, wb; print(pkg_b7_run.NAME); print(os.environ['PYTHONPATH'].split(os.pathsep)[0]);"
            " print('oops', file=sys.stderr); print('{\"a\": 1}'); sys.exit(3)")
    result = wbp.run([sys.executable, "-c", code], tmp_path, src=[tmp_path / "src"])
    assert result.returncode == 3 and not result.ok and not result.timed_out
    lines = result.stdout.splitlines()
    assert lines[0] == "mine" and lines[1] == str((tmp_path / "src").resolve())
    assert result.last_json() == {"a": 1} and result.stderr_lines() == ["oops"] and result.tail() == "oops"
    assert result.command().startswith("python -c")


def test_run_stops_at_the_timeout_and_survives_a_missing_program(tmp_path):
    result = wbp.run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, timeout=1)
    assert result.timed_out and result.returncode is None
    missing = wbp.run(["/no/such/program"], tmp_path)
    assert missing.returncode is None and "Error" in missing.stderr


def test_stderr_lines_by_level():
    result = wbp.RunResult(["x"], 0, "", "2026 INFO a: one\n[DEBUG] b - two\nERRORS are not ERROR lines\n")
    assert len(result.stderr_lines("INFO")) == 1 and len(result.stderr_lines("DEBUG")) == 1
    assert result.stderr_lines("ERROR") == ["ERRORS are not ERROR lines"]


def test_tail_leaves_out_the_markers_under_a_traceback_line():
    """Session 76 (V5): Python 3.13 underlines the failing expression with ~ and ^; a one-line message keeps the
    lines that say something."""
    err = "Traceback (most recent call last):\n    loss = model(x)\n           ~~~~~^^^\nRuntimeError: too small\n"
    assert wbp.RunResult(["x"], 1, "", err).tail(2) == "loss = model(x) ⏎ RuntimeError: too small"
    assert wbp.RunResult(["x"], 1, "", "").tail() == "(rien sur stderr)"


def test_temporary_project_is_removed_at_the_end_of_the_session():
    """Session 76 (V5): the temporary project of a Python session is removed when it ends (it used to pile up in
    /tmp, one per run of the notebook)."""
    import subprocess
    import sys
    code = "from wb import prod; p = prod.project_dir('solution'); (p / 'f.txt').write_text('x'); print(p.parent)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, f"expected the child to run, got {done.stderr[-300:]}"
    folder = Path(done.stdout.strip())
    assert folder.name.startswith("wb_b7_project_") and not folder.exists(), f"expected {folder} removed at exit"


def test_import_package_reads_the_files_again(tmp_path):
    pkg = tmp_path / "src" / wbp.PACKAGE
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "data.py").write_text("VALUE = 1\n")
    assert wbp.import_module(tmp_path, "data").VALUE == 1
    (pkg / "data.py").write_text("VALUE = 2\n")
    assert wbp.import_module(tmp_path, "data").VALUE == 2
    (pkg / "cli.py").write_text("from . import tracking\n")
    with pytest.raises(wbp.PackageImportError, match="ne s'importe pas"):
        wbp.import_module(tmp_path, "cli")
    for key in [k for k in sys.modules if k == wbp.PACKAGE or k.startswith(wbp.PACKAGE + ".")]:
        del sys.modules[key]
    sys.path.remove(str((tmp_path / "src").resolve()))


def test_isolated_logging_puts_the_root_logger_back(monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])                    # as in a notebook: no handler yet
    level = root.level
    with wbp.isolated_logging() as buffer:
        logging.basicConfig(level=logging.DEBUG)                 # what the learner's main() does
        logging.getLogger("b7").error("hidden")
    assert root.handlers == [] and root.level == level
    assert "hidden" in buffer.getvalue()


def test_torch_state_kept_restores_the_deterministic_mode():
    torch = pytest.importorskip("torch")
    before = torch.are_deterministic_algorithms_enabled()
    with wbp.torch_state_kept():
        torch.use_deterministic_algorithms(not before)
        torch.manual_seed(123)
    assert torch.are_deterministic_algorithms_enabled() == before


# ---------------------------------------------------------------------------------------------------------------------
# Data, models, images
# ---------------------------------------------------------------------------------------------------------------------
def test_array_fingerprint_hashes_dtype_shape_and_bytes():
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)
    h = hashlib.sha256()
    h.update(b"uint8(2, 3)")
    h.update(a.tobytes())
    assert wbp.array_fingerprint(a) == h.hexdigest()
    assert wbp.array_fingerprint(a) != wbp.array_fingerprint(a.astype(np.int64))
    assert wbp.array_fingerprint(a, a) != wbp.array_fingerprint(a)


@needs_subset
def test_cifar_subset_has_its_fingerprint_and_is_written_once(tmp_path):
    arrays = wbp.cifar_subset()
    assert wbp.subset_fingerprint(arrays) == wbp.SUBSET_FINGERPRINT
    assert arrays["x_train"].shape == (2000, 32, 32, 3) and arrays["x_val"].shape == (500, 32, 32, 3)
    assert np.bincount(arrays["y_train"]).tolist() == [200] * 10 and np.bincount(arrays["y_val"]).tolist() == [50] * 10
    train = {wbp.array_fingerprint(x) for x in arrays["x_train"]}
    assert not any(wbp.array_fingerprint(x) in train for x in arrays["x_val"])
    arrays["x_train"][0] = 0                                      # a copy: the session's arrays are untouched
    assert wbp.subset_fingerprint(wbp.cifar_subset()) == wbp.SUBSET_FINGERPRINT
    path = wbp.write_subset(tmp_path)
    stamp = path.stat().st_mtime_ns
    assert wbp.write_subset(tmp_path) == path and path.stat().st_mtime_ns == stamp


def test_build_reference_matches_the_mp5_architectures():
    torch = pytest.importorskip("torch")
    cnn = wbp.build_reference("cnn", (12, 24))
    assert sum(p.numel() for p in cnn.parameters()) == 18_322            # the CNN of the fiche
    vgg = wbp.build_reference("vgg", (16, 32, 64))
    state = torch.load(wbp.reference_folder() / "model.pt", map_location="cpu", weights_only=True)
    vgg.load_state_dict(state)                                           # the delivered model of the MP5 solution
    with pytest.raises(ValueError):
        wbp.build_reference("resnet")


def test_reference_model_is_a_fresh_copy_in_eval_mode():
    pytest.importorskip("torch")
    (a, info), (b, _) = wbp.reference_model(), wbp.reference_model()
    assert a is not b and not a.training and info["source"] == "reference"
    assert info["classes"] == wbp.CLASSES and len(info["mean"]) == 3
    next(a.parameters()).data.zero_()
    assert next(b.parameters()).abs().sum() > 0


def test_base_model_falls_back_on_the_reference(monkeypatch, capsys):
    pytest.importorskip("torch")
    monkeypatch.setattr(wbp, "learner_mp5_folder", lambda: None)
    model, info = wbp.base_model("auto")
    assert info["source"] == "reference" and "référence" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        wbp.base_model("learner")
    with pytest.raises(ValueError):
        wbp.base_model("mine")


def test_imread_bgr_reads_like_opencv(tmp_path):
    pytest.importorskip("PIL")
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 0] = 200                                           # red
    path = wbp.write_png(image, tmp_path / "red.png")
    bgr = wbp.imread_bgr(path)
    assert bgr.dtype == np.uint8 and bgr.shape == (4, 5, 3)
    assert (bgr[..., 2] == 200).all() and (bgr[..., 0] == 0).all()


def test_workers_run_gives_the_outcomes_of_b7_12():
    pytest.importorskip("torch")
    zero, zero_again = wbp.workers_run(0, seed=0, n=128, epochs=1), wbp.workers_run(0, seed=0, n=128, epochs=1)
    two, two_again = wbp.workers_run(2, seed=0, n=128, epochs=1), wbp.workers_run(2, seed=0, n=128, epochs=1)
    plain0 = wbp.workers_run(0, seed=0, n=128, epochs=1, augment=False)
    plain2 = wbp.workers_run(2, seed=0, n=128, epochs=1, augment=False)
    assert wbp.same_weights(zero, zero_again) and wbp.same_weights(two, two_again)
    assert not wbp.same_weights(zero, two) and wbp.max_gap(zero, two) > 0
    assert wbp.same_weights(plain0, plain2)


# ---------------------------------------------------------------------------------------------------------------------
# The reference package: installed in a new environment, its tests, and the faulty versions of B7.15
# ---------------------------------------------------------------------------------------------------------------------
def synthetic_data(folder: Path) -> Path:
    """A data file with the schema of the subset (2 000 + 500 random images): no CIFAR-10 needed."""
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / wbp.SUBSET_FILE
    np.savez(path, x_train=rng.integers(0, 256, (2000, 32, 32, 3), dtype=np.uint8), y_train=np.arange(2000) % 10,
             x_val=rng.integers(0, 256, (500, 32, 32, 3), dtype=np.uint8), y_val=np.arange(500) % 10)
    return path


def test_reference_package_installs_in_a_new_environment_and_passes_its_tests(tmp_path):
    pytest.importorskip("torch")
    project = tmp_path / "cifar-prod"
    shutil.copytree(wbp.reference_package(), project, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    synthetic_data(project / "data")
    install = wbp.install_editable(project)
    try:
        if install.ok is None:
            pytest.skip("neither pip nor uv here")
        assert install.ok, install.log
        where = install.run(["python", "-c", "import cifarprod; print(cifarprod.__file__)"], tmp_path)
        assert where.ok and Path(where.stdout.strip()).resolve().is_relative_to(project / "src"), where.stderr
        version = install.run([str(install.scripts / ("cifarprod.exe" if os.name == "nt" else "cifarprod")),
                               "--version"], tmp_path)
        assert version.ok and "0.1.0" in version.stdout, version.stderr
        tests = install.run(["python", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--strict-markers",
                             "-c", str(project / "pyproject.toml"), "--rootdir", str(project)], project, timeout=300)
        assert tests.ok, tests.stdout[-2000:] + tests.stderr[-2000:]
        assert " passed" in tests.stdout and "skipped" not in tests.stdout, tests.stdout[-500:]
    finally:
        install.cleanup()


def test_every_mutant_applies_and_fails_the_reference_tests(tmp_path):
    pytest.importorskip("torch")
    data = synthetic_data(tmp_path / "data")
    reference = wbp.run_tests(wbp.mutant_package(None, tmp_path / "reference"), wbp.reference_package() / "tests",
                              data=data)
    assert reference.ok and reference.failures == [], reference.summary
    for mutant in wbp.MUTANTS:
        run = wbp.run_tests(wbp.mutant_package(mutant.key, tmp_path / mutant.key), wbp.reference_package() / "tests",
                            data=data, marker="slow" if mutant.slow else "not slow", maxfail=1)
        assert run.failed + run.errors >= 1, f"{mutant.key}: {run.summary}"
        name, reason = run.failures[0]
        assert wbp.says_expected(reason), f"{mutant.key}: {name} - {reason}"


def test_a_mutant_that_no_longer_applies_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(wbp, "MUTANTS", [wbp.Mutant("gone", "data.py", [("this text is not there", "")], "x")])
    with pytest.raises(ValueError, match="not found once"):
        wbp.mutant_package("gone", tmp_path / "m")


def test_says_expected_reads_the_first_line_only():
    assert wbp.says_expected("AssertionError: expected 2, got 0")
    assert wbp.says_expected("Failed: DID NOT RAISE <class 'ValueError'>")
    assert wbp.says_expected("AssertionError: attendu 3")
    assert not wbp.says_expected("AssertionError: assert 1 == 2\nexpected later")
    assert not wbp.says_expected("")


def test_the_reference_package_files_fit_in_a_cell():
    """The solutions notebook puts each file of the reference package in a cell as r'''...'''."""
    for path in wbp.reference_package().rglob("*"):
        if path.is_file() and path.suffix in {".py", ".toml", ".yaml"} and "__pycache__" not in path.parts:
            text = path.read_text(encoding="utf-8")
            assert "'''" not in text and not text.endswith("\\"), path
    assert json.loads(json.dumps({"ok": True}))


# ---------------------------------------------------------------------------------------------------------------------
# Part B and C: a server in another process, the reference artifact, the reviews of a workflow and of a Dockerfile,
# the clone of the challenge
# ---------------------------------------------------------------------------------------------------------------------
def test_mp5_data_is_in_the_info_of_the_reference_model():
    pytest.importorskip("torch")
    _, info = wbp.reference_model()
    assert info["data"] == wbp.MP5_DATA and len(info["data"]["sha256"]) == 64 and info["data"]["n_train"] == 45000


def test_latencies_leave_out_the_calls_not_measured():
    calls = []
    times = wbp.latencies(lambda: calls.append(1), n=7, warmup=3)
    assert len(calls) == 10 and times.shape == (7,) and (times >= 0).all()


def test_free_port_then_port_open():
    import socket

    port = wbp.free_port()
    assert not wbp.port_open(port)
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))
        s.listen()
        assert wbp.port_open(port)


def test_reference_artifact_is_served_in_another_process_and_always_stopped(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pydantic")
    import requests

    folder = wbp.reference_artifact(tmp_path / "artifact")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["data"] == wbp.MP5_DATA and meta["arch"] == "vgg" and (folder / "model.safetensors").is_file()
    with wbp.serve_process(wbp.reference_package() / "src", folder, backend="stdlib") as server:
        assert requests.get(f"{server.url}/health", timeout=10).json()["model"]["weights_sha256"] == meta["weights_sha256"]
        process, port = server.process, server.port
    assert process.poll() is not None and not wbp.port_open(port)
    with pytest.raises(RuntimeError, match="stopped"):
        with wbp.serve_process(wbp.reference_package() / "src", tmp_path / "nothing", backend="stdlib", timeout=60):
            pass


REF = wbp.reference_package()


def review(items):
    return {item.key: item.ok for item in items}


def test_the_reference_workflow_passes_every_point():
    got = review(wbp.review_workflow((REF / ".github/workflows/ci.yml").read_text(encoding="utf-8"), REF))
    assert all(got.values()), got
    assert {"on", "python", "torch", "install", "pytest", "smoke", "failures", "secrets", "bash", "files"} <= set(got)


@pytest.mark.parametrize("old, new, key", [
    ('python-version: "3.13"', "python-version: 3.10", "python"),
    (" --index-url https://download.pytorch.org/whl/cpu", "", "torch"),
    ("run: pytest -q", 'run: pytest -q -m "not slow"', "pytest"),
    ("run: pytest -q", "run: pytest -q || true", "failures"),
    ("on:\n  push:\n  pull_request:", "on: [push]", "on"),
    ("          cifarprod train --config configs/smoke.yaml --set data=data/synthetic.npz --out runs/ci\n", "", "smoke"),
    ("    timeout-minutes: 20", "    timeout-minutes: 20\n    env:\n      HF_TOKEN: hf_abcdef123456", "secrets"),
    ("pip install -e \".[dev,serve]\"", "pip install -e \".[dev,serve]\" && fi", "bash"),
    ("configs/smoke.yaml", "configs/tiny.yaml", "files"),
], ids=["unquoted version", "no CPU index", "slow tests skipped", "failure swallowed", "push only", "no smoke training",
        "secret in clear", "bash syntax", "missing file"])
def test_a_faulty_workflow_fails_its_point(old, new, key):
    text = (REF / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert text.count(old) >= 1, old
    got = review(wbp.review_workflow(text.replace(old, new), REF))
    assert got[key] is False, (key, got)


def test_a_workflow_that_is_not_yaml_is_refused():
    assert review(wbp.review_workflow("jobs: [unclosed", None)) == {"yaml": False}


def test_dockerfile_instructions_join_the_continuation_lines():
    text = "# comment\nFROM python:3.13.18-slim\nRUN a \\\n    && b\n\nCMD [\"x\"]\n"
    assert wbp.dockerfile_instructions(text) == [("FROM", "python:3.13.18-slim"), ("RUN", "a && b"), ("CMD", '["x"]')]


def reference_docker():
    return [(REF / name).read_text(encoding="utf-8") for name in ("Dockerfile", ".dockerignore", "requirements-serve.txt")]


def test_the_reference_dockerfile_passes_every_point():
    got = review(wbp.review_dockerfile(*reference_docker(), REF))
    assert all(got.values()) and "runs" in got, got


def test_the_starting_dockerfile_of_b7_24_fails_where_it_should():
    sys.path.insert(0, str(ROOT / "tools" / "chapters"))
    import chB7_code3

    _, ignore, requirements = reference_docker()
    got = review(wbp.review_dockerfile(chB7_code3.DOCKERFILE_START, ignore, requirements))
    for key in ("from", "unbuffered", "secrets", "layers", "torch", "package", "artifact", "user", "cmd", "host", "port"):
        assert got[key] is False, (key, got)


@pytest.mark.parametrize("old, new, key", [
    ("FROM python:3.13.18-slim-trixie", "FROM python:3.13-slim", "from"),
    ("USER app", "USER root", "user"),
    ('CMD ["python", "-m", "cifarprod.serve", "--host", "0.0.0.0", "--port", "8000"]',
     "CMD python -m cifarprod.serve --host 0.0.0.0 --port 8000", "cmd"),
    ('"--host", "0.0.0.0", ', "", "host"),
    ("EXPOSE 8000", "EXPOSE 9000", "port"),
    ("    MODEL_DIR=/app/artifact", "    MODEL_DIR=/app/models", "artifact"),
    ("    PIP_NO_CACHE_DIR=1 \\\n", "    PIP_NO_CACHE_DIR=1 \\\n    API_KEY=abc \\\n", "secrets"),
    ("COPY requirements-serve.txt .\n", "COPY . .\n", "layers"),
], ids=["moving tag", "root", "shell form", "localhost", "other port", "MODEL_DIR elsewhere", "secret", "code first"])
def test_a_faulty_dockerfile_fails_its_point(old, new, key):
    dockerfile, ignore, requirements = reference_docker()
    assert dockerfile.count(old) == 1, old
    got = review(wbp.review_dockerfile(dockerfile.replace(old, new), ignore, requirements))
    assert got[key] is False, (key, got)


def test_unpinned_requirements_and_a_thin_dockerignore_are_refused():
    dockerfile, ignore, requirements = reference_docker()
    got = review(wbp.review_dockerfile(dockerfile, ignore, requirements.replace("fastapi==0.141.1", "fastapi")))
    assert got["requirements"] is False
    got = review(wbp.review_dockerfile(dockerfile, ignore, requirements + "torch==2.11.0\n"))
    assert got["requirements"] is False
    got = review(wbp.review_dockerfile(dockerfile, ignore.replace(".env\n", ""), requirements))
    assert got["dockerignore"] is False


def test_clone_project_clones_what_is_committed_or_copies(tmp_path):
    project = tmp_path / "p"
    (project / "src").mkdir(parents=True)
    (project / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (project / "data").mkdir()
    (project / "data" / "big.npz").write_bytes(b"0")
    copy, how, notes = wbp.clone_project(project, tmp_path / "copy")
    assert how == "copie" and (copy / "src" / "a.py").is_file() and not (copy / "data").exists() and notes == []
    if shutil.which("git") is None:
        pytest.skip("no git here")
    for args in (["git", "init", "-q"], ["git", "add", "src"],
                 ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-m", "t"]):
        assert wbp.run(args, project).ok
    (project / "src" / "b.py").write_text("y = 2\n", encoding="utf-8")
    clone, how, notes = wbp.clone_project(project, tmp_path / "clone")
    assert how == "git clone" and (clone / "src" / "a.py").is_file() and not (clone / "src" / "b.py").exists()
    assert any("b.py" in n or "src/" in n for n in notes) and any("data" in n for n in notes)


def test_machine_speed_is_a_positive_duration():
    pytest.importorskip("torch")
    speed = wbp.machine_speed(repeats=2)
    assert 0 < speed < 30
