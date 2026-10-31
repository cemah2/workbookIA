"""Pre-trained backbones and frozen embeddings for bonus chapter B1 (transfer learning).

Quick reference
---------------
>>> bb = load_backbone("resnet18")            # frozen ImageNet ResNet-18 in eval(), d = 512, its preprocessing
>>> E = bb.embed(X, size=128)                 # X: uint8 (N, 32, 32, 3) -> float32 (N, 512), the frozen embeddings
>>> E2 = bb.embed(X, size=128, block="layer2")   # a block of the ResNet, pooled: (N, 128)
>>> E = cached_embeddings(bb, X, size=128, subset="cifar10-train200")   # computed once, then read from disk
>>> clip = try_backbone("clip_b32", "B1.24")  # None, with a ⏳ line, when the weights cannot be downloaded

The four backbones (``BACKBONES``): ``"resnet18"`` and ``"mobilenet_v3_small"`` (torchvision, ``IMAGENET1K_V1``),
``"dinov2_small"`` (``facebook/dinov2-small`` at ``DINOV2_REVISION``) and ``"clip_b32"`` (the image tower of
``openai/clip-vit-base-patch32`` at ``CLIP_REVISION``; ``bb.full`` is the whole ``CLIPModel``, ``bb.processor`` its
``CLIPProcessor``). Each one is frozen (``requires_grad=False``), in ``eval()`` mode, with the mean, the standard
deviation and the interpolation of its pre-training.

The embeddings cache (``cached_embeddings``, ``cached_arrays``) writes ``.npz`` files into
``wb.datasets.cache_dir() / "embeddings"`` (never versioned), under a key made of the model, its revision, the
resolution, the block, the subset, the seed and a fingerprint of the images: a file written for another key, or
left incomplete, is computed again, never read in part.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from wb import datasets

# ---------------------------------------------------------------------------
# The models of the chapter (revisions checked on 2026-10-31)
# ---------------------------------------------------------------------------
DINOV2_NAME = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"      # Apache 2.0, model.safetensors (88.2 MB)
CLIP_NAME = "openai/clip-vit-base-patch32"
# The main branch (3d74acf9…) only has a pytorch_model.bin (605 MB); transformers 5 then looks for the conversion of
# the Hub (refs/pr/66) and downloads its model.safetensors. The workbook reads that conversion directly: the same
# weights, in the safetensors format, at a fixed commit (data/cards/pretrained.md).
CLIP_REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"
CLIP_MAIN_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

BACKBONES = {
    "resnet18": {"source": "torchvision", "builder": "resnet18", "weights": "ResNet18_Weights",
                 "revision": "IMAGENET1K_V1", "dim": 512, "resample": "bilinear", "size": 224},
    "mobilenet_v3_small": {"source": "torchvision", "builder": "mobilenet_v3_small",
                           "weights": "MobileNet_V3_Small_Weights", "revision": "IMAGENET1K_V1", "dim": 576,
                           "resample": "bilinear", "size": 224},
    "dinov2_small": {"source": "huggingface", "repo": DINOV2_NAME, "revision": DINOV2_REVISION, "dim": 384,
                     "resample": "bicubic", "size": 224},
    "clip_b32": {"source": "huggingface", "repo": CLIP_NAME, "revision": CLIP_REVISION, "dim": 512,
                 "resample": "bicubic", "size": 224},
}
RESNET_BLOCKS = ("stem", "layer1", "layer2", "layer3", "layer4")
EMBED_CACHE_VERSION = 1


class BackboneUnavailable(RuntimeError):
    """The weights of a backbone cannot be loaded here (no network and no cache, a missing library)."""


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def to_input(images, size: int | None, mean=datasets.IMAGENET_MEAN, std=datasets.IMAGENET_STD,
             resample: str = "bilinear"):
    """Images -> the float32 tensor (N, 3, size, size) a pre-trained network expects.

    ``images``: uint8 NumPy images (N, H, W, 3) as ``load_cifar10`` gives them, grayscale (N, H, W) or (N, H, W, 1)
    as ``load_fashion_mnist``, or a float tensor or array (N, C, H, W) in [0, 1] with C = 1 or 3. Steps: values in
    [0, 1], channels first; resize to ``size`` x ``size`` (``resample``: "bilinear" or "bicubic"; antialiasing when
    the image shrinks; ``None`` keeps the size); a single channel is repeated three times
    (``x.expand(-1, 3, -1, -1)``, after the resizing: three times cheaper); normalisation by ``mean`` and ``std``
    (those of the pre-training, one value per channel).
    """
    import torch
    import torch.nn.functional as F

    if resample not in ("bilinear", "bicubic"):
        raise ValueError(f"resample must be 'bilinear' or 'bicubic', got {resample!r}")
    if isinstance(images, np.ndarray) and images.dtype == np.uint8:
        x = torch.from_numpy(np.ascontiguousarray(images))
        if x.ndim == 3:                                   # (N, H, W): grayscale
            x = x.unsqueeze(-1)
        if x.ndim != 4 or x.shape[-1] not in (1, 3):
            raise ValueError(f"uint8 images must be (N, H, W), (N, H, W, 1) or (N, H, W, 3), got {tuple(images.shape)}")
        x = x.permute(0, 3, 1, 2).float() / 255.0
    else:
        x = torch.as_tensor(images, dtype=torch.float32)
        if x.ndim != 4 or x.shape[1] not in (1, 3):
            raise ValueError(f"float images must be (N, C, H, W) with C = 1 or 3, got {tuple(x.shape)}")
    if size is not None and tuple(x.shape[-2:]) != (int(size), int(size)):
        shrink = int(size) < min(x.shape[-2:])
        x = F.interpolate(x, size=(int(size), int(size)), mode=resample, align_corners=False, antialias=shrink)
        if resample == "bicubic":
            x = x.clamp(0.0, 1.0)                         # bicubic overshoots a little at sharp edges
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous()


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------
@dataclass
class Backbone:
    """A frozen pre-trained backbone: ``model(x)`` returns the embedding (N, dim) of a preprocessed batch.

    ``full`` is the whole network (with its ImageNet head for torchvision, ``CLIPModel`` for CLIP), ``categories`` the
    names of the 1 000 ImageNet classes (torchvision only), ``processor`` the Hugging Face processor (Hub only).
    """

    name: str
    model: Any
    dim: int
    mean: tuple
    std: tuple
    resample: str
    revision: str
    source: str
    default_size: int = 224
    blocks: tuple = ()
    full: Any = None
    processor: Any = None
    categories: list = field(default_factory=list)

    def preprocess(self, images, size: int | None = None):
        """``to_input`` with the statistics and the interpolation of this backbone (``size`` defaults to 224)."""
        return to_input(images, size or self.default_size, self.mean, self.std, self.resample)

    def forward_blocks(self, x, blocks):
        """For a ResNet: ``{block: global average pooling of its output}`` for the blocks of ``RESNET_BLOCKS`` asked,
        in ONE pass of the preprocessed batch ``x`` (it stops after the deepest block asked)."""
        blocks = tuple(blocks)
        unknown = [b for b in blocks if b not in self.blocks]
        if unknown or not blocks:
            raise ValueError(f"{self.name} has no block {unknown or blocks!r}; choose among {self.blocks or 'none'}")
        net, out = self.full, {}
        h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        deepest = max(RESNET_BLOCKS.index(b) for b in blocks)
        for k, name in enumerate(RESNET_BLOCKS):
            if k > 0:
                h = getattr(net, name)(h)
            if name in blocks:
                out[name] = h.mean(dim=(2, 3))
            if k == deepest:
                break
        return out

    def forward_block(self, x, block: str | None = None):
        """The embedding of a preprocessed batch ``x``; for a ResNet, ``block`` in ``RESNET_BLOCKS`` gives the global
        average pooling of that block's output instead (``"layer4"`` is the usual embedding)."""
        if block is None:
            return self.model(x)
        return self.forward_blocks(x, (block,))[block]

    def _batches(self, images, size, batch_size, device):
        import torch

        if len(images) == 0:
            raise ValueError("no image to embed")
        if device is not None:
            self.model.to(device)
            if self.full is not None:
                self.full.to(device)
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                x = self.preprocess(images[i:i + batch_size], size)
                yield x if device is None else x.to(device)

    def embed(self, images, size: int | None = None, block: str | None = None, batch_size: int = 64,
              device: str | None = None) -> np.ndarray:
        """float32 embeddings (N, d) of ``images`` (any format of ``to_input``), batch by batch, without gradient
        (``device``: "cuda" moves the backbone there first)."""
        import torch

        out = [self.forward_block(x, block).float().cpu() for x in self._batches(images, size, batch_size, device)]
        return torch.cat(out).numpy()

    def embed_blocks(self, images, size: int | None = None, blocks=RESNET_BLOCKS, batch_size: int = 64,
                     device: str | None = None) -> dict:
        """``{block: float32 (N, C)}``: the pooled outputs of several blocks of a ResNet, in one pass per batch."""
        import torch

        parts: dict = {}
        for x in self._batches(images, size, batch_size, device):
            for name, value in self.forward_blocks(x, blocks).items():
                parts.setdefault(name, []).append(value.float().cpu())
        return {name: torch.cat(values).numpy() for name, values in parts.items()}


class _HubImageEmbedding:
    """``forward(x)``: the image embedding of a Hugging Face model (``pooler_output`` of DINOv2, the projected image
    embedding of CLIP)."""

    def __new__(cls, hf_model, kind: str):
        import torch.nn as nn

        class HubImageEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.hf = hf_model

            def forward(self, x):
                if kind == "clip":
                    out = self.hf.get_image_features(pixel_values=x)
                    return out if hasattr(out, "shape") else out.pooler_output    # transformers 5: an output object
                return self.hf(pixel_values=x).pooler_output

        return HubImageEmbedding()


@contextlib.contextmanager
def quiet_hub():
    """No progress bar and no info log of transformers or huggingface_hub while loading (restored afterwards)."""
    restore = []
    try:
        tf_logging = importlib.import_module("transformers.utils.logging")
        level = tf_logging.get_verbosity()
        tf_logging.set_verbosity_error()
        tf_logging.disable_progress_bar()
        restore.append(lambda: (tf_logging.set_verbosity(level), tf_logging.enable_progress_bar()))
    except Exception:  # noqa: BLE001 - transformers absent or another version
        pass
    try:
        with datasets.quiet_huggingface():
            yield
    finally:
        for undo in restore:
            with contextlib.suppress(Exception):
                undo()


def _freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _load_torchvision(name: str, spec: dict) -> Backbone:
    import torch.nn as nn
    import torchvision

    weights = getattr(getattr(torchvision.models, spec["weights"]), spec["revision"])
    full = _freeze(getattr(torchvision.models, spec["builder"])(weights=weights))
    model = copy.deepcopy(full)
    if name == "resnet18":
        model.fc = nn.Identity()                           # phi(x): the 512 values after the global average pooling
        blocks = RESNET_BLOCKS
    else:
        model.classifier = nn.Identity()                   # the 576 values at the entrance of classifier
        blocks = ()
    t = weights.transforms()
    return Backbone(name, _freeze(model), spec["dim"], tuple(t.mean), tuple(t.std), spec["resample"],
                    f"{spec['weights']}.{spec['revision']}", "torchvision", spec["size"], blocks, full=full,
                    categories=list(weights.meta["categories"]))


def _load_hub(name: str, spec: dict) -> Backbone:
    with quiet_hub():
        if name == "clip_b32":
            from transformers import CLIPModel, CLIPProcessor

            full = _freeze(CLIPModel.from_pretrained(spec["repo"], revision=spec["revision"]))
            processor = CLIPProcessor.from_pretrained(spec["repo"], revision=spec["revision"])
            model = _freeze(_HubImageEmbedding(full, "clip"))
            mean, std = CLIP_MEAN, CLIP_STD
        else:
            from transformers import AutoImageProcessor, AutoModel

            full = _freeze(AutoModel.from_pretrained(spec["repo"], revision=spec["revision"]))
            processor = AutoImageProcessor.from_pretrained(spec["repo"], revision=spec["revision"])
            model = _freeze(_HubImageEmbedding(full, "dinov2"))
            mean, std = tuple(processor.image_mean), tuple(processor.image_std)
    return Backbone(name, model, spec["dim"], mean, std, spec["resample"], spec["revision"], "huggingface",
                    spec["size"], (), full=full, processor=processor)


_LOADED: dict = {}


def load_backbone(name: str, *, reuse: bool = True) -> Backbone:
    """The frozen backbone ``name`` (a key of ``BACKBONES``), loaded once per session (``reuse=False`` loads it
    again). Raises :class:`BackboneUnavailable` with the reason when its weights cannot be read or downloaded."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose among {sorted(BACKBONES)}")
    if reuse and name in _LOADED:
        return _LOADED[name]
    spec = BACKBONES[name]
    try:
        backbone = _load_torchvision(name, spec) if spec["source"] == "torchvision" else _load_hub(name, spec)
    except Exception as exc:  # noqa: BLE001 - no network, a proxy, a missing package: the caller falls back
        where = spec.get("repo", spec.get("builder"))
        raise BackboneUnavailable(f"{name} ({where}) impossible à charger : {type(exc).__name__} : "
                                  f"{str(exc).strip()[:300]}") from exc
    _LOADED[name] = backbone
    return backbone


def try_backbone(name: str, ex_id: str | None = None) -> Backbone | None:
    """``load_backbone(name)``, or None with a ⏳ line when the weights cannot be loaded (the exercises that need
    this backbone then say so instead of crashing; to be validated on Colab)."""
    try:
        return load_backbone(name)
    except BackboneUnavailable as exc:
        where = f"Ex {ex_id} : " if ex_id else ""
        print(f"⏳ {where}{exc}. Sans ce modèle, cette partie est « à valider sur Colab » ; le reste du notebook "
              "continue.")
        return None


# ---------------------------------------------------------------------------
# The embeddings cache
# ---------------------------------------------------------------------------
def embeddings_dir() -> Path:
    """Where the cached embeddings go: ``wb.datasets.cache_dir() / "embeddings"`` (never versioned)."""
    path = datasets.cache_dir() / "embeddings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical(key: dict) -> str:
    return json.dumps(key, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def cache_file(key: dict, folder: str | Path | None = None) -> Path:
    """The ``.npz`` file of a key: a readable prefix and 16 hexadecimal digits of the SHA-256 of the key."""
    digest = hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()[:16]
    prefix = "-".join(str(key[k]) for k in ("model", "size", "block", "subset") if key.get(k) is not None)
    prefix = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in prefix)[:80] or "arrays"
    base = Path(folder) if folder is not None else embeddings_dir()
    return base / f"{prefix}-{digest}.npz"


def _cache_path(key: dict, folder) -> Path:
    path = cache_file(key, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_cache(path: Path, key: dict) -> dict | None:
    """The arrays of ``path`` if it holds exactly ``key``, complete; None otherwise (any doubt: recompute)."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["__key__"]) != _canonical(key) or int(data["__complete__"]) != 1:
                return None
            names = [str(n) for n in data["__names__"]]
            arrays = {n: data[n] for n in names}
    except Exception:  # noqa: BLE001 - truncated, corrupt, another format: recompute
        return None
    return arrays


def cached_arrays(key: dict, compute: Callable[[], dict], folder: str | Path | None = None,
                  verbose: bool = True) -> dict:
    """``compute()`` (a dict of NumPy arrays) computed once for ``key``, then read from its ``.npz`` file.

    The file stores the key, the names of the arrays and a completeness mark, and is written atomically (a temporary
    file, then ``os.replace``): a file of another key, an incomplete or unreadable file is computed again.
    """
    path = _cache_path(key, folder)
    arrays = _read_cache(path, key)
    if arrays is not None:
        if verbose:
            print(f"ℹ️ embeddings relus dans le cache ({path.name})")
        return arrays
    arrays = compute()
    if not isinstance(arrays, dict) or not arrays or any(str(n).startswith("__") for n in arrays):
        raise ValueError("compute() must return a non-empty dict of arrays whose names do not start with '__'")
    arrays = {str(n): np.asarray(a) for n, a in arrays.items()}
    if any(a.dtype == object for a in arrays.values()):
        raise ValueError("cached arrays cannot hold Python objects")
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, __key__=np.array(_canonical(key)), __names__=np.array(list(arrays)),
                     __complete__=np.array(1), **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return arrays


def images_fingerprint(images) -> str:
    """16 hexadecimal digits of the SHA-256 of the images (their bytes, shape and type)."""
    arr = np.ascontiguousarray(np.asarray(images))
    h = hashlib.sha256(f"{arr.shape}|{arr.dtype}".encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def embedding_key(backbone: Backbone, images, *, size: int | None, block: str | None, subset: str,
                  seed: int | None = None) -> dict:
    """The cache key of the embeddings of ``images`` by ``backbone``: model, revision, resolution, block, subset,
    seed, number of images and fingerprint of their pixels."""
    return {"version": EMBED_CACHE_VERSION, "model": backbone.name, "revision": backbone.revision,
            "size": int(size or backbone.default_size), "block": block or "final", "subset": str(subset),
            "seed": seed, "n": int(len(images)), "images": images_fingerprint(images)}


def cached_embeddings(backbone: Backbone, images, *, size: int | None = None, block: str | None = None,
                      subset: str, seed: int | None = None, batch_size: int = 64, folder: str | Path | None = None,
                      verbose: bool = True) -> np.ndarray:
    """``backbone.embed(images, size, block)``, computed once and read from the cache afterwards (float32 (N, d))."""
    key = embedding_key(backbone, images, size=size, block=block, subset=subset, seed=seed)
    arrays = cached_arrays(key, lambda: {"embeddings": backbone.embed(images, size, block, batch_size)},
                           folder=folder, verbose=verbose)
    emb = arrays["embeddings"]
    if emb.shape[0] != len(images):     # cannot happen with a matching key; never hand back a partial matrix
        raise RuntimeError(f"cached embeddings have {emb.shape[0]} rows for {len(images)} images")
    return emb


def cached_block_embeddings(backbone: Backbone, images, *, size: int | None = None, blocks=RESNET_BLOCKS,
                            subset: str, seed: int | None = None, batch_size: int = 64,
                            folder: str | Path | None = None, verbose: bool = True) -> dict:
    """``backbone.embed_blocks(images, size, blocks)`` (``{block: (N, C)}``), computed once, then read from the cache."""
    blocks = tuple(blocks)
    key = embedding_key(backbone, images, size=size, block="+".join(blocks), subset=subset, seed=seed)
    arrays = cached_arrays(key, lambda: backbone.embed_blocks(images, size, blocks, batch_size), folder=folder,
                           verbose=verbose)
    if set(arrays) != set(blocks) or any(a.shape[0] != len(images) for a in arrays.values()):
        raise RuntimeError("cached block embeddings do not match the images and blocks asked")
    return {b: arrays[b] for b in blocks}


def clear_embeddings_cache(folder: str | Path | None = None) -> int:
    """Remove the cached ``.npz`` files (all of them); returns how many were removed."""
    base = Path(folder) if folder is not None else embeddings_dir()
    removed = 0
    for path in base.glob("*.npz"):
        path.unlink()
        removed += 1
    return removed


__all__ = ["BACKBONES", "Backbone", "BackboneUnavailable", "CLIP_MAIN_REVISION", "CLIP_NAME", "CLIP_REVISION",
           "DINOV2_NAME", "DINOV2_REVISION", "RESNET_BLOCKS", "cache_file", "cached_arrays", "cached_block_embeddings", "cached_embeddings",
           "clear_embeddings_cache", "embedding_key", "embeddings_dir", "images_fingerprint", "load_backbone",
           "quiet_hub", "to_input", "try_backbone"]
