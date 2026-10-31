"""Tests of genbench: the networks, the VAE loss, the GAN step and its exact resumption, the measures of the protocol.

    python -m pytest projets/partie_6_generatif/solution -q

They run on small random images and tiny networks (a few seconds on a CPU): what they check does not depend on
Fashion-MNIST. The oracles are PyTorch (F.binary_cross_entropy_with_logits, a GAN step written by hand), NumPy,
SciPy and properties written by hand; never another function of genbench.
"""

import copy
import json
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

from genbench import (LOG_COLUMNS, Classifier, ConvVAE, Discriminator, Generator, class_report, classifier_features,
                      discriminator_body, frechet_distance, gan_step, generator_body, init_dcgan_weights, interpolate,
                      kl_per_latent, load_checkpoint, n_parameters, nearest_neighbors, sample_images, save_checkpoint,
                      save_config, set_seed, train_classifier, train_gan, vae_loss)


@pytest.fixture
def images():
    """64 random images (64, 1, 28, 28) in [0, 1] and their labels (int64, 0-9)."""
    g = torch.Generator().manual_seed(5)
    return torch.rand(64, 1, 28, 28, generator=g), torch.randint(0, 10, (64,), generator=g)


def tiny_gan(conditional=False, seed=0):
    """A tiny DCGAN (noise of 8 values, width 4), initialized with Radford's rule."""
    torch.manual_seed(seed)
    G = Generator(8, 4, n_classes=10 if conditional else 0)
    D = Discriminator(4, n_classes=10 if conditional else 0)
    G.apply(init_dcgan_weights)
    D.apply(init_dcgan_weights)
    return G, D


def largest_gap(model_1, model_2):
    """The largest absolute gap between the floating-point tensors of two state_dicts."""
    return max((a - b).abs().max().item() for a, b in zip(model_1.state_dict().values(), model_2.state_dict().values())
               if a.is_floating_point())


# ---------------------------------------------------------------- seeds and configuration
def test_set_seed_replays_the_three_generators():
    set_seed(3)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(3)
    again = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == again, f"expected the same draws of random, NumPy and PyTorch after set_seed(3), got {first} and {again}"


def test_save_config_adds_date_and_versions(tmp_path):
    written = save_config(tmp_path / "config.json", {"seeds": [0, 1, 2]})
    reread = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert reread == written and reread["seeds"] == [0, 1, 2], "expected the config, reread from the JSON file"
    assert {"date", "versions"} <= set(reread), f"expected the keys date and versions, got {sorted(reread)}"
    with pytest.raises(TypeError):
        save_config(tmp_path / "bad.json", {"model": object()})
    assert not (tmp_path / "bad.json").exists(), "expected no file when the config is not JSON-serializable"


# ---------------------------------------------------------------- the networks
def test_networks_have_the_documented_sizes():
    sizes = {"Classifier": (n_parameters(Classifier()), 218_586), "Generator": (n_parameters(Generator(64, 32)), 240_321),
             "Discriminator": (n_parameters(Discriminator(32)), 36_577)}
    wrong = {name: got for name, (got, want) in sizes.items() if got != want}
    assert not wrong, f"expected {({k: w for k, (_, w) in sizes.items()})} parameters, got {wrong}"


def test_dcgan_bodies_follow_radford():
    want_g = [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Unflatten, nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU,
              nn.ConvTranspose2d, nn.Tanh]
    want_d = [nn.Conv2d, nn.LeakyReLU, nn.Conv2d, nn.BatchNorm2d, nn.LeakyReLU, nn.Flatten, nn.Linear]
    got_g, got_d = [type(m) for m in generator_body(8, 4)], [type(m) for m in discriminator_body(1, 4)]
    assert got_g == want_g, f"expected the generator layers {[t.__name__ for t in want_g]}, got {[t.__name__ for t in got_g]}"
    assert got_d == want_d, f"expected the discriminator layers {[t.__name__ for t in want_d]}, got {[t.__name__ for t in got_d]}"
    G, D = Generator(8, 4).eval(), Discriminator(4)
    out = G(torch.randn(3, 8))
    assert out.shape == (3, 1, 28, 28) and out.abs().max() <= 1, f"expected G(z) (3, 1, 28, 28) in [-1, 1], got {tuple(out.shape)}"
    assert D(out).shape == (3, 1), f"expected D(x) of shape (3, 1), got {tuple(D(out).shape)}"


def test_conditional_networks_need_labels():
    G, D = tiny_gan(conditional=True)
    z, labels = torch.randn(4, 8), torch.tensor([0, 3, 7, 9])
    G.eval()
    with torch.no_grad():
        want = G.net(torch.cat([z, G.embed(labels)], dim=1))
        assert torch.allclose(G(z, labels), want), "expected G(z, y) = net(concatenation of z, then embed(y))"
    with pytest.raises(ValueError):
        G(z)
    with pytest.raises(ValueError):
        Discriminator(4)(torch.zeros(2, 1, 28, 28), torch.tensor([1, 2]))


def test_init_follows_radford():
    torch.manual_seed(0)
    G = Generator(64, 32)
    G.apply(init_dcgan_weights)
    linear, bn1d = G.net[0], G.net[1]
    assert abs(linear.weight.std().item() - 0.02) < 0.002, f"expected weights ~ N(0, 0.02^2), got std {linear.weight.std():.4f}"
    assert abs(bn1d.weight.mean().item() - 1) < 0.005 and bn1d.bias.abs().max() == 0, \
        "expected batchnorm weights ~ N(1, 0.02^2) and biases at 0"


def test_classifier_features_are_the_penultimate_layer_in_eval_mode(images):
    x, _ = images
    torch.manual_seed(0)
    model = Classifier(width=4, n_features=8, dropout=0.5)
    with torch.no_grad():
        model.train()(x)                                   # running statistics that are no longer the initial ones
    model.eval()
    with torch.no_grad():
        want = model.to_features(model.body(x)).double().numpy()
    model.train()
    got = classifier_features(model, x, batch_size=24)    # 24 + 24 + 16
    assert got.shape == (64, 8) and got.dtype == np.float64, f"expected float64 features (64, 8), got {got.dtype} {got.shape}"
    assert np.allclose(got, want, atol=1e-6), "expected the features of the model in evaluation mode (batchnorm, dropout)"
    assert model.training, "expected the model back in its previous mode (training)"


def test_train_classifier_ends_in_eval_mode(images):
    x, y = images
    torch.manual_seed(0)
    model = Classifier(width=4, n_features=8)
    history = train_classifier(model, x, y, epochs=2, batch_size=16, val=(x[:20], y[:20]))
    assert len(history["train_loss"]) == 2 and len(history["val_acc"]) == 2, f"expected 2 values per list, got {history}"
    assert not model.training, "expected the classifier in evaluation mode after training"


# ---------------------------------------------------------------- the VAE
def test_vae_loss_matches_the_formula():
    g = torch.Generator().manual_seed(1)
    logits, x = torch.randn(5, 1, 28, 28, generator=g), torch.rand(5, 1, 28, 28, generator=g)
    mu, log_var = torch.randn(5, 4, generator=g), torch.randn(5, 4, generator=g)
    loss, rec, kl = vae_loss(logits, x, mu, log_var, beta=2.5)
    want_rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum").item() / 5
    want_kl = 0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum().item() / 5
    assert abs(rec.item() - want_rec) < 1e-3, f"expected the BCE summed over the pixels / N = {want_rec:.3f}, got {rec.item():.3f}"
    assert abs(kl.item() - want_kl) < 1e-4, f"expected the KL summed over the latents / N = {want_kl:.4f}, got {kl.item():.4f}"
    assert abs(loss.item() - (want_rec + 2.5 * want_kl)) < 1e-3, "expected loss = rec + beta * kl"


def test_vae_reparameterizes_in_training_and_uses_mu_in_eval():
    vae = ConvVAE(latent_dim=2, width=4)
    mu = torch.tensor([[1.0, -2.0]]).repeat(40_000, 1)
    log_var = torch.log(torch.tensor([[4.0, 0.25]])).repeat(40_000, 1)
    torch.manual_seed(0)
    z = vae.train().reparameterize(mu, log_var)
    assert torch.allclose(z.mean(0), torch.tensor([1.0, -2.0]), atol=0.03) and \
        torch.allclose(z.std(0), torch.tensor([2.0, 0.5]), atol=0.03), \
        f"expected z with means (1, -2) and standard deviations (2, 0.5), got {z.mean(0)} and {z.std(0)}"
    assert torch.equal(vae.eval().reparameterize(mu, log_var), mu), "expected z = mu in evaluation mode"


def test_kl_per_latent_is_the_mean_kl(images):
    x, _ = images
    torch.manual_seed(0)
    vae = ConvVAE(latent_dim=3, width=4).eval()
    with torch.no_grad():
        mu, log_var = vae.encode(x)
    want = (0.5 * (mu ** 2 + log_var.exp() - 1 - log_var)).double().mean(0).numpy()
    assert np.allclose(kl_per_latent(vae, x, batch_size=10), want, atol=1e-6), f"expected the mean KL per latent {want}"


def test_sample_images_are_reproducible_and_in_unit_range():
    torch.manual_seed(0)
    vae, (G, _) = ConvVAE(latent_dim=3, width=4), tiny_gan()
    state = torch.get_rng_state()
    for model in (vae, G):
        first, again = sample_images(model, 6, seed=4), sample_images(model, 6, seed=4)
        assert first.shape == (6, 1, 28, 28) and torch.equal(first, again), "expected the same 6 images for the same seed"
        assert first.min() >= 0 and first.max() <= 1, f"expected images in [0, 1], got [{first.min():.2f}, {first.max():.2f}]"
    assert torch.equal(state, torch.get_rng_state()), "expected PyTorch's global random state untouched"


def test_interpolate_starts_and_ends_on_the_two_codes():
    torch.manual_seed(0)
    vae = ConvVAE(latent_dim=3, width=4).eval()
    z0, z1 = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 2.0, 0.0])
    path = interpolate(vae, z0, z1, 5)
    with torch.no_grad():
        ends = torch.sigmoid(vae.decode(torch.stack([z0, z1])))
    assert path.shape == (5, 1, 28, 28), f"expected (5, 1, 28, 28), got {tuple(path.shape)}"
    assert torch.allclose(path[0], ends[0], atol=1e-6) and torch.allclose(path[-1], ends[1], atol=1e-6), \
        "expected the first image decoded from z0 and the last one from z1"


# ---------------------------------------------------------------- the GAN
def hand_step(G, D, opt_g, opt_d, real, z):
    """The GAN step written by hand: D on reals (1) and detached fakes (0), then G with the label 1."""
    fake = G(z)
    a, b = D(real), D(fake.detach())
    loss_d = F.binary_cross_entropy_with_logits(a, torch.ones_like(a)) + F.binary_cross_entropy_with_logits(b, torch.zeros_like(b))
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()
    c = D(fake)
    loss_g = F.binary_cross_entropy_with_logits(c, torch.ones_like(c))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return loss_d.item(), loss_g.item()


def test_gan_step_matches_a_step_written_by_hand(images):
    x, _ = images
    G1, D1 = tiny_gan()
    G2, D2 = copy.deepcopy(G1), copy.deepcopy(D1)
    adam = lambda model: torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.5, 0.999))
    real, z = x[:16] * 2 - 1, torch.randn(16, 8)
    out = gan_step(G1, D1, adam(G1), adam(D1), real, z)
    want_d, want_g = hand_step(G2, D2, adam(G2), adam(D2), real, z)
    assert abs(out["loss_d"] - want_d) < 1e-5 and abs(out["loss_g"] - want_g) < 1e-5, \
        f"expected loss_d {want_d:.5f} and loss_g {want_g:.5f}, got {out['loss_d']:.5f} and {out['loss_g']:.5f}"
    assert largest_gap(G1, G2) < 1e-6 and largest_gap(D1, D2) < 1e-6, \
        "expected the weights of a step written by hand (G trained with the label 1 through the updated D)"


def test_checkpoint_reloads_with_weights_only(tmp_path):
    G, D = tiny_gan()
    opt_g, opt_d = torch.optim.Adam(G.parameters()), torch.optim.Adam(D.parameters())
    gan_step(G, D, opt_g, opt_d, torch.rand(8, 1, 28, 28) * 2 - 1, torch.randn(8, 8))
    gen = torch.Generator().manual_seed(9)
    save_checkpoint(tmp_path / "last.pt", G, D, opt_g, opt_d, 3, {"loss_d": [1.0]}, gen)
    torch.load(tmp_path / "last.pt", weights_only=True)
    G2, D2 = tiny_gan(seed=1)
    opt_g2, opt_d2 = torch.optim.Adam(G2.parameters()), torch.optim.Adam(D2.parameters())
    gen2 = torch.Generator().manual_seed(1)
    epoch, history = load_checkpoint(tmp_path / "last.pt", G2, D2, opt_g2, opt_d2, gen2)
    assert epoch == 3 and history == {"loss_d": [1.0]}, f"expected epoch 3 and the history, got {epoch} and {history}"
    assert largest_gap(G, G2) == 0 and largest_gap(D, D2) == 0, "expected the weights of G and D restored"
    assert torch.equal(gen.get_state(), gen2.get_state()), "expected the state of the generator restored"


def test_train_gan_resumes_bit_for_bit(tmp_path, images):
    x, y = images
    def run(folder, stop=None, resume=False):
        G, D = tiny_gan(conditional=True)
        history = train_gan(G, D, x, labels=y, epochs=3, batch_size=16, seed=2, folder=folder, stop_after=stop, resume=resume)
        return G, history
    straight, history_s = run(tmp_path / "a")
    run(tmp_path / "b", stop=2)
    torch.manual_seed(1234)                                         # another state: the resumption must restore its own
    resumed, history_r = run(tmp_path / "b", resume=True)
    assert largest_gap(straight, resumed) == 0.0, \
        f"expected 2 + 1 epochs = 3 epochs in a row bit for bit, largest gap {largest_gap(straight, resumed):.2e}"
    assert history_r["loss_d"] == history_s["loss_d"] and len(history_r["steps"]["loss_g"]) == 3 * 4, \
        "expected the same history, 4 steps per epoch (64 images, mini-batches of 16)"
    rows = (tmp_path / "b" / "log.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",") == LOG_COLUMNS and len(rows) == 4, f"expected a header and 3 rows in log.csv, got {rows}"


# ---------------------------------------------------------------- the measures
class FixedLogits(nn.Module):
    """A fake classifier that returns given logits, whatever the images."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.logits[: len(x)]


def test_class_report_counts_and_entropy():
    uniform = class_report(FixedLogits(torch.eye(10).repeat(3, 1) * 5), torch.zeros(30, 1, 28, 28))
    assert uniform["counts"] == [3] * 10 and abs(uniform["entropy"] - 1) < 1e-9, \
        f"expected 3 images per class and an entropy of 1, got {uniform['counts']} and {uniform['entropy']}"
    collapsed = class_report(FixedLogits(torch.eye(10)[[4] * 8] * 5), torch.zeros(8, 1, 28, 28))
    assert collapsed["entropy"] == 0 and collapsed["max_share"] == 1, "expected an entropy of 0 when every image is in one class"
    want = torch.softmax(torch.eye(10)[[4] * 8] * 5, dim=1).max(1).values.mean().item()
    assert abs(collapsed["confidence"] - want) < 1e-6, f"expected a mean confidence of {want:.4f}"


def test_nearest_neighbors_match_numpy(images):
    x, _ = images
    queries, references = x[:7], x[10:60]
    distances, indices = nearest_neighbors(queries, references, batch_size=3)
    q, r = queries.reshape(7, -1).double().numpy(), references.reshape(50, -1).double().numpy()
    full = np.sqrt(((q[:, None, :] - r[None, :, :]) ** 2).sum(-1))
    assert np.array_equal(indices, full.argmin(1)), f"expected the indices {full.argmin(1)}, got {indices}"
    assert np.allclose(distances, full.min(1)), "expected the Euclidean distances (not squared) to the nearest reference"


def test_frechet_distance_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 4)), rng.normal(size=(200, 4)) * 1.5 + 0.5
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    want = ((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s1 + s2 - 2 * linalg.sqrtm(s1 @ s2).real)
    assert abs(frechet_distance(a, b) - want) < 1e-6, f"expected {want:.6f}, got {frechet_distance(a, b):.6f}"
