"""Tests of the model: the shape of the output, and a mini-batch learned by heart (fiche, 207.6)."""

import pytest
import torch
import torch.nn.functional as F

from cifarprod.model import build_model


@pytest.mark.parametrize("batch", [1, 7])
def test_output_shape(model, batch):
    out = model(torch.zeros(batch, 3, 32, 32))
    assert out.shape == (batch, 10), f"expected ({batch}, 10), got {tuple(out.shape)}"


def test_overfits_one_batch(model):
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(16, 3, 32, 32, generator=g), torch.arange(16) % 10
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"expected a loss < 0.05 on 16 memorised images, got {loss.item():.3f}"


@pytest.mark.parametrize("arch", ["cnn", "vgg"])
def test_both_archs_give_ten_logits(arch):
    out = build_model(arch, widths=(8, 16))(torch.zeros(3, 3, 32, 32))
    assert out.shape == (3, 10), f"expected (3, 10) for {arch}, got {tuple(out.shape)}"
