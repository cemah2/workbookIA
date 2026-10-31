"""Your tests of genbench (step MP6.7): at least three, each with a clear name and a message whose first line says
what was expected (the notebook only prints that line).

    python -m pytest mon_travail/projets/partie_6_generatif -q

Run them on small random images and tiny networks: what they check must not depend on Fashion-MNIST, and they must
take a few seconds. The oracles are PyTorch, NumPy, SciPy or a property written by hand, never another function of
genbench. One example is given; the notebook runs your tests again against five mutants of genbench (MP6.7), and
good tests catch at least two.
"""

import torch
import torch.nn as nn

from genbench import class_report


class FixedLogits(nn.Module):
    """A fake classifier that returns given logits, whatever the images."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.dummy = nn.Parameter(torch.zeros(1))      # a parameter, as a real model has

    def forward(self, x):
        return self.logits[: len(x)]


def test_class_report_sees_a_collapse():
    collapsed = class_report(FixedLogits(torch.eye(10)[[4] * 8] * 5), torch.zeros(8, 1, 28, 28))
    assert collapsed["counts"][4] == 8 and collapsed["entropy"] == 0, \
        f"expected 8 images in class 4 and an entropy of 0, got {collapsed['counts']} and {collapsed['entropy']}"


# TODO MP6.7: your tests (vae_loss, gan_step, classifier_features, nearest_neighbors, the resumption of train_gan...)
