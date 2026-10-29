"""The `mistakes` dicts of the solution notebooks never repeat a message.

A dict literal with a repeated key keeps only its last value: the other wrong answers silently lose their message
(found in CP4.7, CP5.4 and ch. 22, session 52). Each answer cell (tag "answer") is parsed, and every dict literal
must have distinct constant keys.
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted([*ROOT.glob("chapitres/*/05_solutions.ipynb"), *ROOT.glob("checkpoints/*/03_examen_solutions.ipynb"),
                    *ROOT.glob("projets/*/solution/*.ipynb")])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_answer_cells_have_no_repeated_mistakes(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code" or "answer" not in cell.get("metadata", {}).get("tags", []):
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                repeated = sorted({key for key in keys if keys.count(key) > 1}, key=str)
                assert not repeated, f"cell {index}: a dict repeats the key(s) {repeated[:3]}: give each its own message"
