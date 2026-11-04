"""Export a model with torch.export, and check that the export computes what the model computes.

    path = export_artifact("artifact")                 # artifact/model.pt2, a batch of 1 to 1024 images
    exported = torch.export.load(path).module()        # a callable, without the code of cifarprod
    gap = max_gap(model, exported, x)                  # of the order of the float32 rounding, or 0

An ExportedProgram holds the graph of the operations, the weights and the guards (the shapes it accepts): it is
served without the Python code that built the model (another program, C++, a phone) and refuses an input of another
shape instead of computing in silence.
"""

from __future__ import annotations

from pathlib import Path

from .artifact import load_artifact


def export_model(model, path, max_batch: int = 1024) -> Path:
    """Export ``model`` in eval() mode (it is switched to it) with torch.export, the batch size free from 1 to
    ``max_batch`` (dynamic_shapes on the dimension 0 of the argument "input"), on an example batch of 2 images of
    3 x 32 x 32, and save the ExportedProgram in ``path`` (.pt2). Returns the path."""
    import torch

    model.eval()
    batch = torch.export.Dim("batch", min=1, max=int(max_batch))
    example = (torch.zeros(2, 3, 32, 32),)
    program = torch.export.export(model, example, dynamic_shapes={"input": {0: batch}})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, str(path))
    return path


def export_artifact(folder, max_batch: int = 1024) -> Path:
    """The model of the artifact ``folder`` (load_artifact) exported into ``folder / "model.pt2"``."""
    model, _ = load_artifact(folder)
    return export_model(model, Path(folder) / "model.pt2", max_batch)


def max_gap(model, exported, x) -> float:
    """The largest absolute difference between the logits of ``model`` and of ``exported`` on the batch ``x``
    (float32 (N, 3, 32, 32)), computed without a graph, as a Python float."""
    import torch

    with torch.inference_mode():
        return float((model(x).double() - exported(x).double()).abs().max())
