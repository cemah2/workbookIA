"""The training of genbench: seeds and configuration, the reference classifier, the VAE (its loss with beta) and the
GAN (one step, a journaled training with a complete checkpoint and an exact resumption) (chapters 23, 25 and 27).

    set_seed(0)
    history = train_vae(vae, images, epochs=5, beta=1.0, val_images=val)
    history = train_gan(G, D, images, epochs=5, folder=Path("runs/gan_0"))            # last.pt and log.csv
    history = train_gan(G, D, images, epochs=8, folder=Path("runs/gan_0"), resume=True)   # 3 more epochs

Images are float tensors (N, 1, 28, 28) in [0, 1] (images_to_tensor); train_gan brings them to [-1, 1] itself.

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_COLUMNS = ["epoch", "loss_d", "loss_g", "d_real", "d_fake", "seconds"]


def library_versions() -> dict:
    """Provided: the versions of Python, NumPy and PyTorch that run the project."""
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    """Provided: uint8 images (N, 28, 28) -> a float32 tensor (N, 1, 28, 28) in [0, 1]."""
    return torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).unsqueeze(1).contiguous()


# ---------------------------------------------------------------------------------------------- seeds and configuration
def set_seed(seed: int) -> None:
    """Seed the three random generators the project uses: Python's ``random``, NumPy's global generator
    (``np.random.seed``) and PyTorch's (``torch.manual_seed``)."""
    raise NotImplementedError("set_seed")  # TODO MP6.1


def save_config(path: Path, config: dict) -> dict:
    """Write the configuration of the experiments to the JSON file path, BEFORE they run: the keys of config, plus
    "date" (today, ISO format) and "versions" (library_versions()). Returns what was written. The config must be
    JSON-serializable (TypeError otherwise, and nothing is written) and must not already hold "date" or "versions"
    (ValueError)."""
    raise NotImplementedError("save_config")  # TODO MP6.1


# ---------------------------------------------------------------------------------------------- the reference classifier
def train_classifier(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, *, epochs: int, lr: float = 1e-3,
                     batch_size: int = 128, seed: int = 0, val: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict:
    """Train the classifier with Adam(lr) and the cross-entropy, in training mode, by mini-batches of batch_size in the
    order of ``torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))``, drawn again at each epoch
    from that same generator (the last mini-batch may be smaller). Returns {"train_loss": the mean loss per example of
    each epoch, "val_acc": the accuracy on val = (images, labels) after each epoch, measured in evaluation mode (an
    empty list without val), "seconds": per epoch}. The model ends in evaluation mode."""
    raise NotImplementedError("train_classifier")  # TODO MP6.2


# ---------------------------------------------------------------------------------------------- the VAE
def vae_loss(logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(loss, rec, kl), three scalar tensors, all MEAN PER EXAMPLE of the mini-batch (chapter 25):
    rec = the binary cross-entropy with logits SUMMED over the 784 pixels, divided by N;
    kl = 0.5 * sum over the latents of (mu^2 + exp(log_var) - 1 - log_var), divided by N;
    loss = rec + beta * kl."""
    raise NotImplementedError("vae_loss")  # TODO MP6.3


def train_vae(model: nn.Module, images: torch.Tensor, *, epochs: int, beta: float = 1.0, lr: float = 1e-3,
              batch_size: int = 128, seed: int = 0, val_images: torch.Tensor | None = None) -> dict:
    """Train the VAE with Adam(lr) on vae_loss(..., beta), in training mode, by mini-batches of batch_size in the
    order of a ``torch.Generator().manual_seed(seed)`` (as train_classifier); the noise of the reparameterization
    comes from PyTorch's global random state (seed it before building the model). Returns the history, one value per
    epoch: "loss", "rec", "kl" (means per example over the epoch, each mini-batch weighted by its size), "val_rec" and
    "val_kl" (on val_images, in evaluation mode, so z = mu; empty lists without them) and "seconds". The model ends in
    evaluation mode."""
    raise NotImplementedError("train_vae")  # TODO MP6.3


# ---------------------------------------------------------------------------------------------- the GAN
def gan_step(G: nn.Module, D: nn.Module, opt_g: torch.optim.Optimizer, opt_d: torch.optim.Optimizer,
             real: torch.Tensor, z: torch.Tensor, labels: torch.Tensor | None = None,
             fake_labels: torch.Tensor | None = None) -> dict:
    """One step of the GAN (chapter 27), on real images in [-1, 1] and noise z:
    1) the fakes G(z) (with fake_labels for a conditional GAN); the discriminator's step: the binary cross-entropy
       with logits of D(real) against 1 PLUS that of D(fake.detach()) against 0 (reals with their labels, fakes with
       fake_labels), zero_grad, backward, opt_d.step();
    2) the generator's step, with the NON-SATURATING loss: the binary cross-entropy of D(fake), through the updated
       D, against 1; zero_grad, backward, opt_g.step() only.
    Returns {"loss_d", "loss_g", "d_real", "d_fake"}: the two losses and the mean probabilities sigmoid(D(real)) and
    sigmoid(D(fake)) of the discriminator's step, as floats."""
    raise NotImplementedError("gan_step")  # TODO MP6.4


def save_checkpoint(path: Path, G: nn.Module, D: nn.Module, opt_g: torch.optim.Optimizer,
                    opt_d: torch.optim.Optimizer, epoch: int, history: dict, generator: torch.Generator) -> None:
    """Save in ONE file (torch.save) everything an exact resumption needs: the state_dicts of G, D, opt_g and opt_d,
    the epoch, the history, the state of the torch.Generator that draws the mini-batches and the noise
    (generator.get_state()) and PyTorch's global random state (torch.get_rng_state()). Only tensors, numbers,
    strings, and dicts or lists of them: the file reloads with torch.load(path, weights_only=True)."""
    raise NotImplementedError("save_checkpoint")  # TODO MP6.4


def load_checkpoint(path: Path, G: nn.Module, D: nn.Module, opt_g: torch.optim.Optimizer,
                    opt_d: torch.optim.Optimizer, generator: torch.Generator) -> tuple[int, dict]:
    """Restore into the given objects what save_checkpoint saved (with ``torch.load(path, weights_only=True)``):
    the two networks, the two optimizers, the state of generator and PyTorch's global random state. Returns
    (epoch, history)."""
    raise NotImplementedError("load_checkpoint")  # TODO MP6.4


def train_gan(G: nn.Module, D: nn.Module, images: torch.Tensor, *, epochs: int, labels: torch.Tensor | None = None,
              lr: float = 2e-4, batch_size: int = 64, seed: int = 0, folder: Path | None = None, resume: bool = False,
              stop_after: int | None = None, on_epoch=None) -> dict:
    """Train the GAN for ``epochs`` epochs with gan_step, and journal it.

    - Two optimizers Adam(lr, betas=(0.5, 0.999)), one per network (Radford). Images in [0, 1] are brought to
      [-1, 1] (2x - 1). One ``generator = torch.Generator().manual_seed(seed)`` draws everything: at each epoch the
      order ``torch.randperm(len(images), generator=generator)``, cut in mini-batches of batch_size (the last
      incomplete one is dropped: a batchnorm needs equal batches), then for each mini-batch the noise
      ``torch.randn(m, G.latent_dim, generator=generator)`` and, for a conditional GAN (labels given), the fake labels
      ``torch.randint(0, G.n_classes, (m,), generator=generator)``, in this order; both are drawn on the CPU, then
      moved to ``images.device`` (a GPU on Colab). G and D are in training mode.
    - The history: "loss_d", "loss_g", "d_real", "d_fake" (the means of gan_step's values over each epoch),
      "seconds" (per epoch) and "steps", a dict with the lists "loss_d" and "loss_g" of every step.
    - With a folder: after each epoch, ``folder / "last.pt"`` (save_checkpoint) and one row appended to
      ``folder / "log.csv"`` (columns LOG_COLUMNS, header when the file is new). With ``resume=True``, train_gan
      first restores last.pt (load_checkpoint: networks, optimizers, history, random states) and goes on from the
      next epoch up to ``epochs``. ``stop_after=k`` stops after epoch k, as an interrupted run would: 2 epochs, then a
      resumption up to 3, give the same weights, bit for bit, as 3 epochs in a row.
    - ``on_epoch(epoch, G)``, when given, is called after each epoch (a grid of samples for a fixed noise...).

    Returns the history. G and D end in evaluation mode.
    """
    raise NotImplementedError("train_gan")  # TODO MP6.4
