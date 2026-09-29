#!/usr/bin/env python
"""Execute notebooks and report durations and errors.

    python tools/run_all_notebooks.py                      # every notebook, FAST_MODE
    python tools/run_all_notebooks.py chapitres/ch03_*     # one chapter (folder or files)
    python tools/run_all_notebooks.py --inplace            # save the outputs in the files
    python tools/run_all_notebooks.py --full               # FAST_MODE = False (long!)
    python tools/run_all_notebooks.py --report report.md   # also write a Markdown report

Budgets (BIBLE §4, FAST_MODE on CPU): a notebook must finish in < 10 min and
no cell may take > 3 min. Overruns are reported. Exit code 1 if any notebook
fails.

Engines: ``nbclient`` (a real Jupyter kernel, default when installed) or
``inprocess`` (a minimal fallback running cells with ``exec`` in this Python
process: no ``!shell`` or ``%magic``, which the workbook never uses).
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOBS = (
    "00_setup/*.ipynb",
    "chapitres/*/03_notebook.ipynb",
    "chapitres/*/05_solutions.ipynb",
)
NOTEBOOK_BUDGET = 600.0  # seconds
CELL_BUDGET = 180.0


@dataclass
class Result:
    path: Path
    ok: bool = True
    seconds: float = 0.0
    cell_times: list = field(default_factory=list)  # (cell index, seconds)
    error: str | None = None
    error_cell: int | None = None
    engine: str = ""

    @property
    def slowest(self):
        return max(self.cell_times, key=lambda t: t[1]) if self.cell_times else (None, 0.0)


# ---------------------------------------------------------------------------
# Engine 1: nbclient (real kernel)
# ---------------------------------------------------------------------------
def _parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def run_nbclient(path: Path, timeout: int, kernel: str | None) -> tuple[dict, Result]:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    result = Result(path, engine="nbclient")
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel or nb.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
        record_timing=True,
    )
    start = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as exc:
        result.ok = False
        text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).strip()  # drop ANSI colours
        result.error = text.splitlines()[-1] if text else repr(exc)
    except Exception as exc:  # timeouts, dead kernels...
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.perf_counter() - start
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        timing = cell.metadata.get("execution", {})
        begin, end = timing.get("iopub.execute_input"), timing.get("shell.execute_reply")
        if begin and end:
            result.cell_times.append((index, (_parse_ts(end) - _parse_ts(begin)).total_seconds()))
        if not result.ok and result.error_cell is None:
            if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                result.error_cell = index
    return nb, result


# ---------------------------------------------------------------------------
# Engine 2: in-process fallback
# ---------------------------------------------------------------------------
class _Collector:
    def __init__(self):
        self.outputs = []

    def display(self, *objs, **kwargs):
        for obj in objs:
            self.outputs.append(_rich_output(obj, "display_data"))


def _rich_output(obj, output_type: str, count: int | None = None) -> dict:
    data = {"text/plain": repr(obj)}
    html = getattr(obj, "_repr_html_", None)
    if callable(html):
        try:
            rendered = html()
            if rendered:
                data["text/html"] = rendered
        except Exception:
            pass
    markdown = getattr(obj, "_repr_markdown_", None)
    if callable(markdown):
        try:
            rendered = markdown()
            if rendered:
                data["text/markdown"] = rendered
        except Exception:
            pass
    output = {"output_type": output_type, "data": data, "metadata": {}}
    if output_type == "execute_result":
        output["execution_count"] = count
    return output


def _figures_to_outputs() -> list[dict]:
    if "matplotlib.pyplot" not in sys.modules:
        return []
    import matplotlib.pyplot as plt

    outputs = []
    for number in plt.get_fignums():
        fig = plt.figure(number)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        outputs.append({
            "output_type": "display_data",
            "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                     "text/plain": f"<Figure {number}>"},
            "metadata": {},
        })
    plt.close("all")
    return outputs


def run_inprocess(path: Path, timeout: int, kernel: str | None) -> tuple[dict, Result]:
    old_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    nb = json.loads(path.read_text(encoding="utf-8"))
    result = Result(path, engine="inprocess")
    namespace = {"__name__": "__main__"}
    collector = _Collector()
    namespace["display"] = collector.display
    try:  # route IPython.display.display to the collector as well
        import IPython.display as ipd

        ipd.display = collector.display
    except ImportError:
        pass
    try:  # plt.show() is a no-op here: figures are collected after each cell
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.show = lambda *args, **kwargs: None
    except ImportError:
        pass
    old_cwd, old_path = Path.cwd(), list(sys.path)
    os.chdir(path.parent)
    sys.path.insert(0, str(path.parent))
    start = time.perf_counter()
    count = 0
    try:
        for index, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            count += 1
            cell["execution_count"] = count
            collector.outputs = []
            stdout, stderr = io.StringIO(), io.StringIO()
            t0 = time.perf_counter()
            error = None
            try:
                if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
                    raise RuntimeError("magie IPython (!/%) non supportée par le moteur inprocess")
                tree = ast.parse(source)
                last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                body = ast.Module(body=tree.body[:-1] if last else tree.body, type_ignores=[])
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(body, f"<cell {index}>", "exec"), namespace)
                    value = None
                    if last is not None:
                        value = eval(compile(ast.Expression(last.value), f"<cell {index}>", "eval"), namespace)
                if value is not None and not hasattr(value, "_ipython_display_") and not (
                    type(value).__module__.startswith("matplotlib")
                ):
                    collector.outputs.append(_rich_output(value, "execute_result", count))
            except Exception as exc:
                error = exc
            outputs = []
            if stdout.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
            if stderr.getvalue():
                outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.getvalue()})
            outputs += collector.outputs + _figures_to_outputs()
            if error is not None:
                outputs.append({
                    "output_type": "error", "ename": type(error).__name__, "evalue": str(error),
                    "traceback": traceback.format_exception(type(error), error, error.__traceback__),
                })
            cell["outputs"] = outputs
            elapsed = time.perf_counter() - t0
            result.cell_times.append((index, elapsed))
            if error is not None:
                result.ok = False
                result.error = f"{type(error).__name__}: {error}"
                result.error_cell = index
                break
            if elapsed > timeout:
                result.ok = False
                result.error = f"cellule {index} : {elapsed:.0f} s > timeout {timeout} s"
                result.error_cell = index
                break
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        if old_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = old_backend
    result.seconds = time.perf_counter() - start
    return nb, result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def collect(paths: list[str], root: Path = ROOT) -> list[Path]:
    found: list[Path] = []
    if not paths:
        for pattern in DEFAULT_GLOBS:
            found += sorted(root.glob(pattern))
    def notebooks_in(folder: Path) -> list[Path]:
        return sorted(p for p in folder.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)

    for raw in paths:
        path = Path(raw)
        matches = sorted(Path().glob(raw)) if any(ch in raw for ch in "*?[") else [path]
        for match in matches:  # PowerShell does not expand globs: a pattern may match folders
            if match.is_dir():
                found += notebooks_in(match)
            elif match.suffix == ".ipynb":
                found.append(match)
    unique = []
    for p in found:
        if "mon_travail" in p.resolve().parts:  # never touch the learner's space
            continue
        if p.resolve() not in {u.resolve() for u in unique}:
            unique.append(p)
    return unique


def _save(nb, path: Path, engine: str) -> None:
    if engine == "nbclient":
        import nbformat

        nbformat.write(nb, path)
    else:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def pick_engine(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ipykernel  # noqa: F401
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401

        return "nbclient"
    except ImportError:
        return "inprocess"


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # emoji on Windows consoles and pipes
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="notebooks, folders or globs (default: all)")
    parser.add_argument("--inplace", action="store_true", help="save the executed notebooks (outputs kept)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="FAST_MODE = False (WB_FAST_MODE=0)")
    mode.add_argument("--fast", action="store_true", help="force FAST_MODE = True (default)")
    parser.add_argument("--engine", choices=["auto", "nbclient", "inprocess"], default="auto")
    parser.add_argument("--kernel", default=None, help="Jupyter kernel name (default: from the notebook)")
    parser.add_argument("--timeout", type=int, default=900, help="max seconds per cell before abort")
    parser.add_argument("--report", type=Path, default=None, help="write a Markdown report here")
    args = parser.parse_args(argv)

    os.environ["WB_FAST_MODE"] = "0" if args.full else "1"
    os.environ.setdefault("WB_ROOT", str(ROOT))
    engine = pick_engine(args.engine)
    runner = run_nbclient if engine == "nbclient" else run_inprocess
    notebooks = collect(args.paths)
    if not notebooks:
        print("Aucun notebook à exécuter.")
        return 0

    print(f"▶️ {len(notebooks)} notebook(s), moteur {engine}, FAST_MODE={'False' if args.full else 'True'}")
    results = []
    for path in notebooks:
        rel = path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path
        print(f"  … {rel}", flush=True)
        nb, result = runner(path.resolve(), args.timeout, args.kernel)
        results.append(result)
        if args.inplace:
            _save(nb, path, engine)
        status = "✅" if result.ok else "❌"
        cell, slow = result.slowest
        print(f"  {status} {rel} : {result.seconds:.1f} s (cellule la plus lente : n°{cell}, {slow:.1f} s)")
        if not result.ok:
            print(f"     erreur cellule {result.error_cell} : {result.error}")

    lines = ["| Notebook | Statut | Durée (s) | Cellule la plus lente | Budget |", "|---|---|---|---|---|"]
    over = 0
    for r in results:
        cell, slow = r.slowest
        flags = []
        if not args.full and r.seconds > NOTEBOOK_BUDGET:
            flags.append("> 10 min")
        if not args.full and slow > CELL_BUDGET:
            flags.append(f"cellule {cell} > 3 min")
        over += bool(flags)
        rel = r.path.relative_to(ROOT) if r.path.is_relative_to(ROOT) else r.path
        status = "✅" if r.ok else f"❌ cellule {r.error_cell} : {r.error}"
        lines.append(f"| {rel} | {status} | {r.seconds:.1f} | n°{cell} ({slow:.1f} s) | "
                     f"{'⚠️ ' + ', '.join(flags) if flags else 'OK'} |")
    failed = sum(not r.ok for r in results)
    summary = (f"{len(results) - failed}/{len(results)} notebook(s) OK, "
               f"{failed} en erreur, {over} hors budget.")
    print("\n" + "\n".join(lines) + "\n\n" + summary)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        header = f"# Exécution des notebooks ({datetime.now():%Y-%m-%d %H:%M}, moteur {engine})\n\n"
        args.report.write_text(header + "\n".join(lines) + "\n\n" + summary + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
