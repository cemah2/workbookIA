"""The models of cifarprod: the CNNs of the MP5 (cifarkit), rebuilt from a few settings, so that an artifact can
rebuild them.

    model = build_model("vgg", widths=(16, 32))       # float32 (N, 3, 32, 32) -> logits (N, 10)
    model = build_model(widths=(12, 24))              # the defaults: the CNN of the fiche (18 322 parameters)

- "cnn": for each width w, Conv2d(c, w, 3, padding=1), ReLU, MaxPool2d(2); then Flatten and Linear (no global pooling);
- "vgg": for each width w, two Conv2d(c, w, 3, padding=1, bias=False), each followed by BatchNorm2d(w) and ReLU; a
  MaxPool2d(2) after every stage but the last; then AdaptiveAvgPool2d(1), Flatten, Linear(widths[-1], n_classes).
"""

from __future__ import annotations

import torch.nn as nn

ARCHS = ("cnn", "vgg")


def build_model(arch: str = "cnn", widths=(12, 24), n_classes: int = 10) -> nn.Sequential:
    """The network ``arch`` ("cnn" or "vgg") with one stage per width; ValueError for another arch or empty widths."""
    widths = tuple(int(w) for w in widths)
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}: expected one of {ARCHS}")
    if not widths or min(widths) < 1:
        raise ValueError(f"widths must be positive whole numbers, got {widths}")
    layers, c = [], 3
    if arch == "cnn":
        side = 32
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c, side = w, side // 2
        return nn.Sequential(*layers, nn.Flatten(), nn.Linear(c * side * side, n_classes))
    for i, w in enumerate(widths):
        for _ in range(2):
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU()]
            c = w
        if i < len(widths) - 1:
            layers.append(nn.MaxPool2d(2))
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, n_classes)]
    return nn.Sequential(*layers)


def n_parameters(model: nn.Module) -> int:
    """The number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
