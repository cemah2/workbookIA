"""The training of genbench: seeds and configuration, the reference classifier, the VAE (its loss with beta) and the
GAN (one step, a journaled training with a complete checkpoint and an exact resumption) (chapters 23, 25 and 27).

    set_seed(0)
    history = train_vae(vae, images, epochs=5, beta=1.0, val_images=val)
    history = train_gan(G, D, images, epochs=5, folder=Path("runs/gan_0"))            # last.pt and log.csv
    history = train_gan(G, D, images, epochs=8, folder=Path("runs/gan_0"), resume=True)   # 3 more epochs

Images are float tensors (N, 1, 28, 28) in [0, 1] (images_to_tensor); train_gan brings them to [-1, 1] itself.
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_config(path: Path, config: dict) -> dict:
    """Write the configuration of the experiments to the JSON file path, BEFORE they run: the keys of config, plus
    "date" (today, as "YYYY-MM-DD": ``datetime.date.today().isoformat()``) and "versions" (library_versions()). Returns what was written. The config must be
    JSON-serializable (TypeError otherwise, and nothing is written) and must not already hold "date" or "versions"
    (ValueError)."""
    clash = {"date", "versions"} & set(config)
    if clash:
        raise ValueError(f"config must not hold {sorted(clash)}: save_config adds them")
    content = {**config, "date": dt.date.today().isoformat(), "versions": library_versions()}
    text = json.dumps(content, indent=1, ensure_ascii=False)          # TypeError before anything is written
    Path(path).write_text(text + "\n", encoding="utf-8")
    return content


# ---------------------------------------------------------------------------------------------- the reference classifier
def train_classifier(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, *, epochs: int, lr: float = 1e-3,
                     batch_size: int = 128, seed: int = 0, val: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict:
    """Train the classifier with Adam(lr) and the cross-entropy, in training mode, by mini-batches of batch_size in the
    order of ``torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))``, drawn again at each epoch
    from that same generator (the last mini-batch may be smaller). Returns {"train_loss": the mean loss per example of
    each epoch, "val_acc": the accuracy on val = (images, labels) after each epoch, measured in evaluation mode (an
    empty list without val), "seconds": per epoch}. The model ends in evaluation mode."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history = {"train_loss": [], "val_acc": [], "seconds": []}
    for _ in range(epochs):
        started, total = time.perf_counter(), 0.0
        model.train()
        for idx in torch.randperm(len(images), generator=generator).split(batch_size):
            loss = F.cross_entropy(model(images[idx]), labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train_loss"].append(total / len(images))
        if val is not None:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(val[0][i:i + 1000]).argmax(1) for i in range(0, len(val[0]), 1000)])
            history["val_acc"].append((pred == val[1]).float().mean().item())
        history["seconds"].append(time.perf_counter() - started)
    model.eval()
    return history


# ---------------------------------------------------------------------------------------------- the VAE
def vae_loss(logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(loss, rec, kl), three scalar tensors, all MEAN PER EXAMPLE of the mini-batch (chapter 25):
    rec = the binary cross-entropy with logits SUMMED over the 784 pixels, divided by N;
    kl = 0.5 * sum over the latents of (mu^2 + exp(log_var) - 1 - log_var), divided by N;
    loss = rec + beta * kl."""
    n = len(x)
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / n
    kl = 0.5 * (mu.pow(2) + log_var.exp() - 1.0 - log_var).sum() / n
    return rec + beta * kl, rec, kl


def train_vae(model: nn.Module, images: torch.Tensor, *, epochs: int, beta: float = 1.0, lr: float = 1e-3,
              batch_size: int = 128, seed: int = 0, val_images: torch.Tensor | None = None) -> dict:
    """Train the VAE with Adam(lr) on vae_loss(..., beta), in training mode, by mini-batches of batch_size in the
    order of a ``torch.Generator().manual_seed(seed)`` (as train_classifier); the noise of the reparameterization
    comes from PyTorch's global random state (seed it before building the model). Returns the history, one value per
    epoch: "loss", "rec", "kl" (means per example over the epoch, each mini-batch weighted by its size), "val_rec" and
    "val_kl" (on val_images, in evaluation mode, so z = mu; empty lists without them) and "seconds". The model ends in
    evaluation mode."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history = {"loss": [], "rec": [], "kl": [], "val_rec": [], "val_kl": [], "seconds": []}
    for _ in range(epochs):
        started, sums = time.perf_counter(), np.zeros(3)
        model.train()
        for idx in torch.randperm(len(images), generator=generator).split(batch_size):
            x = images[idx]
            logits, mu, log_var = model(x)
            loss, rec, kl = vae_loss(logits, x, mu, log_var, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += np.array([loss.item(), rec.item(), kl.item()]) * len(idx)
        for key, value in zip(("loss", "rec", "kl"), sums / len(images)):
            history[key].append(float(value))
        if val_images is not None:
            model.eval()
            totals = np.zeros(2)
            with torch.no_grad():
                for i in range(0, len(val_images), 1000):
                    x = val_images[i:i + 1000]
                    logits, mu, log_var = model(x)
                    _, rec, kl = vae_loss(logits, x, mu, log_var, beta)
                    totals += np.array([rec.item(), kl.item()]) * len(x)
            history["val_rec"].append(float(totals[0] / len(val_images)))
            history["val_kl"].append(float(totals[1] / len(val_images)))
        history["seconds"].append(time.perf_counter() - started)
    model.eval()
    return history


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
    fake = G(z) if fake_labels is None else G(z, fake_labels)
    logits_real = D(real) if labels is None else D(real, labels)
    logits_fake = D(fake.detach()) if fake_labels is None else D(fake.detach(), fake_labels)
    loss_d = (F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
              + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake)))
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()
    logits = D(fake) if fake_labels is None else D(fake, fake_labels)
    loss_g = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return {"loss_d": loss_d.item(), "loss_g": loss_g.item(),
            "d_real": torch.sigmoid(logits_real).mean().item(), "d_fake": torch.sigmoid(logits_fake).mean().item()}


def save_checkpoint(path: Path, G: nn.Module, D: nn.Module, opt_g: torch.optim.Optimizer,
                    opt_d: torch.optim.Optimizer, epoch: int, history: dict, generator: torch.Generator) -> None:
    """Save in ONE file (torch.save) everything an exact resumption needs: the state_dicts of G, D, opt_g and opt_d,
    the epoch, the history, the state of the torch.Generator that draws the mini-batches and the noise
    (generator.get_state()) and PyTorch's global random state (torch.get_rng_state(); train_gan never draws from
    it, it is saved for safety, should a dropout or another draw be added). Only tensors, numbers,
    strings, and dicts or lists of them: the file reloads with torch.load(path, weights_only=True)."""
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "epoch": int(epoch), "history": history, "generator": generator.get_state(),
                "torch_rng": torch.get_rng_state()}, path)


def load_checkpoint(path: Path, G: nn.Module, D: nn.Module, opt_g: torch.optim.Optimizer,
                    opt_d: torch.optim.Optimizer, generator: torch.Generator) -> tuple[int, dict]:
    """Restore into the given objects what save_checkpoint saved (with ``torch.load(path, weights_only=True)``):
    the two networks, the two optimizers, the state of generator and PyTorch's global random state. Returns
    (epoch, history)."""
    state = torch.load(path, weights_only=True)
    G.load_state_dict(state["G"])
    D.load_state_dict(state["D"])
    opt_g.load_state_dict(state["opt_g"])
    opt_d.load_state_dict(state["opt_d"])
    generator.set_state(state["generator"])
    torch.set_rng_state(state["torch_rng"])
    return state["epoch"], state["history"]


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
      ``folder / "log.csv"`` (columns LOG_COLUMNS, header when the file is new). Without resume, a new run starts
      a new journal: an existing log.csv of the folder is deleted first. With ``resume=True``, train_gan first
      restores last.pt (load_checkpoint: networks, optimizers, history, random states) and goes on from the next
      epoch up to ``epochs``, appending to log.csv; without folder / "last.pt", it raises FileNotFoundError (it
      never silently starts again from scratch). The draws all come from ``generator``: PyTorch's global random
      state is not used, only saved. ``stop_after=k`` stops after epoch k, as an interrupted run would: 2 epochs, then a
      resumption up to 3, give the same weights, bit for bit, as 3 epochs in a row.
    - ``on_epoch(epoch, G)``, when given, is called after each epoch (a grid of samples for a fixed noise...).

    Returns the history. G and D end in evaluation mode.
    """
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    generator = torch.Generator().manual_seed(seed)
    history = {"loss_d": [], "loss_g": [], "d_real": [], "d_fake": [], "seconds": [],
               "steps": {"loss_d": [], "loss_g": []}}
    start = 1
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if not resume and (folder / "log.csv").exists():
            (folder / "log.csv").unlink()                                 # a new run, a new journal
    if resume:
        if folder is None or not (folder / "last.pt").exists():
            raise FileNotFoundError("resume=True needs folder / 'last.pt', written by an earlier train_gan")
        epoch_done, history = load_checkpoint(folder / "last.pt", G, D, opt_g, opt_d, generator)
        start = epoch_done + 1
    conditional = labels is not None
    for epoch in range(start, epochs + 1):
        if stop_after is not None and epoch > stop_after:
            break
        started = time.perf_counter()
        G.train()
        D.train()
        values = {"loss_d": [], "loss_g": [], "d_real": [], "d_fake": []}
        order = torch.randperm(len(images), generator=generator)
        for i in range(0, len(images) - batch_size + 1, batch_size):
            idx = order[i:i + batch_size]
            real = images[idx] * 2 - 1
            z = torch.randn(len(idx), G.latent_dim, generator=generator).to(images.device)
            fake_labels = (torch.randint(0, G.n_classes, (len(idx),), generator=generator).to(images.device)
                           if conditional else None)
            out = gan_step(G, D, opt_g, opt_d, real, z, labels[idx] if conditional else None, fake_labels)
            for key in values:
                values[key].append(out[key])
        for key, series in values.items():
            history[key].append(float(np.mean(series)))
        history["steps"]["loss_d"] += values["loss_d"]
        history["steps"]["loss_g"] += values["loss_g"]
        history["seconds"].append(time.perf_counter() - started)
        if folder is not None:
            save_checkpoint(folder / "last.pt", G, D, opt_g, opt_d, epoch, history, generator)
            new = not (folder / "log.csv").exists()
            with (folder / "log.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                if new:
                    writer.writerow(LOG_COLUMNS)
                writer.writerow([epoch] + [round(history[key][-1], 5) for key in LOG_COLUMNS[1:]])
        if on_epoch is not None:
            on_epoch(epoch, G)
    G.eval()
    D.eval()
    return history
