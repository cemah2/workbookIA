"""The models of cifarkit: two baselines (an MLP, a small CNN), a VGG-style CNN and a small ResNet for 32 x 32 colour
images (chapters 21, 23 and 24), all built from a configuration dictionary, so that a checkpoint can rebuild them.

    model = build_model({"arch": "resnet", "widths": [16, 32, 64], "dropout": 0.0})

Every model takes float32 tensors (N, 3, 32, 32) and returns the 10 logits (N, 10). The mylearn library is imported
inside the function that uses it (flat_size), so that a delivered model is rebuilt without it (predict.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def n_parameters(model: nn.Module) -> int:
    """The number of trainable parameters (``requires_grad``) of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mlp(hidden: int = 512, n_classes: int = 10) -> nn.Sequential:
    """The first baseline: Flatten, Linear(3 * 32 * 32, hidden), ReLU, Linear(hidden, n_classes)."""
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))


def flat_size(size: int, widths: list[int]) -> int:
    """The number of values that small_cnn(widths) flattens: after each block (a 3 x 3 convolution with padding 1,
    then a 2 x 2 max pooling), the side is given by ``mylearn.conv.conv_output_size``; times the last width."""
    from mylearn.conv import conv_output_size          # here: cifarkit imports without mylearn (predict.py)
    for _ in widths:
        size = conv_output_size(size, 3, stride=1, padding=1)
        size = conv_output_size(size, 2, stride=2)
    return widths[-1] * size * size


def small_cnn(widths: list[int] = (32, 64), n_classes: int = 10) -> nn.Sequential:
    """The second baseline: for each width w, Conv2d(c, w, 3, padding=1), ReLU, MaxPool2d(2); then Flatten and
    Linear(flat_size(32, widths), n_classes), without global pooling."""
    layers, c = [], 3
    for w in widths:
        layers += [nn.Conv2d(c, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        c = w
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(flat_size(32, list(widths)), n_classes))


def vgg(widths: list[int] = (32, 64, 128), downsample: str = "pool", batchnorm: bool = True, dropout: float = 0.0,
        n_classes: int = 10) -> nn.Sequential:
    """A VGG-style CNN: one stage per width w, made of two 3 x 3 convolutions (padding 1) to w channels, each followed
    by BatchNorm2d(w) if batchnorm (the convolution then has no bias) and by ReLU. Every stage but the last halves the
    side: with downsample="pool", a MaxPool2d(2) after the stage; with downsample="stride", its second convolution has
    stride 2 instead. Then AdaptiveAvgPool2d(1), Flatten, Dropout(dropout) if dropout > 0, and Linear(widths[-1],
    n_classes)."""
    if downsample not in ("pool", "stride"):
        raise ValueError(f"downsample must be 'pool' or 'stride', got {downsample!r}")
    layers, c = [], 3
    for i, w in enumerate(widths):
        last = i == len(widths) - 1
        for j in range(2):
            stride = 2 if (downsample == "stride" and j == 1 and not last) else 1
            layers.append(nn.Conv2d(c, w, 3, stride=stride, padding=1, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(w))
            layers.append(nn.ReLU())
            c = w
        if downsample == "pool" and not last:
            layers.append(nn.MaxPool2d(2))
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(c, n_classes))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Two 3 x 3 convolutions (the first with the given stride), each followed by a BatchNorm2d, and a shortcut:
    ``relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))``. The shortcut is the identity when the shape does not
    change, otherwise a 1 x 1 convolution with the same stride followed by a BatchNorm2d. The convolutions have no
    bias (the batchnorm that follows has one)."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The output of the block, of shape (N, c_out, H / stride, W / stride)."""
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


def resnet(widths: list[int] = (16, 32, 64), blocks: int = 1, dropout: float = 0.0,
           n_classes: int = 10) -> nn.Sequential:
    """A small ResNet for 32 x 32 images: a stem (Conv2d(3, widths[0], 3, padding=1, bias=False), BatchNorm2d, ReLU),
    then ``blocks`` ResidualBlocks per width (the first block of every width but the first has stride 2), then
    AdaptiveAvgPool2d(1), Flatten, Dropout(dropout) if dropout > 0, and Linear(widths[-1], n_classes)."""
    layers = [nn.Conv2d(3, widths[0], 3, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU()]
    c = widths[0]
    for i, w in enumerate(widths):
        for j in range(blocks):
            layers.append(ResidualBlock(c, w, stride=2 if (i > 0 and j == 0) else 1))
            c = w
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers, nn.Linear(c, n_classes))


def build_model(config: dict) -> nn.Module:
    """The model described by config: config["arch"] is "mlp", "cnn", "vgg" or "resnet", and the other keys are the
    arguments of mlp, small_cnn, vgg or resnet (for example {"arch": "vgg", "widths": [32, 64], "downsample":
    "stride"}). Raises ValueError for another arch."""
    builders = {"mlp": mlp, "cnn": small_cnn, "vgg": vgg, "resnet": resnet}
    config = dict(config)
    arch = config.pop("arch", None)
    if arch not in builders:
        raise ValueError(f"unknown arch {arch!r}: expected one of {sorted(builders)}")
    return builders[arch](**config)
