"""The networks of genbench: the reference classifier whose features measure the generators, a convolutional VAE and
a DCGAN (generator and discriminator, optionally conditioned on the class) for 28 x 28 grayscale images (chapters 21,
25 and 27).

    clf = Classifier()                       # clf(x): 10 logits; clf.features(x): its penultimate layer, 128 values
    vae = ConvVAE(latent_dim=16)             # vae(x) -> (logits, mu, log_var); vae.decode(z) -> logits
    G, D = Generator(latent_dim=64), Discriminator()
    G.apply(init_dcgan_weights)
    D.apply(init_dcgan_weights)

Images are float tensors (N, 1, 28, 28): in [0, 1] for the classifier and the VAE, in [-1, 1] for the GAN (the tanh
output of the generator); evaluate.sample_images brings the samples of both generators to [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 10


def n_parameters(model: nn.Module) -> int:
    """Provided: the number of parameters of the model that require a gradient (weights and biases)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------------- the reference classifier
class Classifier(nn.Module):
    """The reference classifier, a small CNN: two stages of two 3 x 3 convolutions (``width`` then ``2 * width``
    channels, padding 1, no bias, each followed by a batchnorm and a ReLU) and a 2 x 2 max pooling (28 -> 14 -> 7);
    then a dropout, a dense layer to ``n_features`` values and a ReLU (the features), and a dense layer to the 10
    logits. With width 16 and 128 features: 218 586 parameters."""

    def __init__(self, width: int = 16, n_features: int = 128, dropout: float = 0.3):
        super().__init__()
        layers = []
        for c_in, c_out in ((1, width), (width, 2 * width)):
            for a, b in ((c_in, c_out), (c_out, c_out)):
                layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU()]
            layers.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*layers)
        self.to_features = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * width * 7 * 7, n_features),
                                         nn.ReLU())
        self.head = nn.Linear(n_features, N_CLASSES)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The features of images x (N, 1, 28, 28) in [0, 1]: the ``n_features`` values of the penultimate layer,
        after its ReLU (body, then to_features), shape (N, n_features)."""
        return self.to_features(self.body(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The 10 logits (N, 10) of images x (N, 1, 28, 28) in [0, 1]: the head applied to the features."""
        return self.head(self.features(x))


# ---------------------------------------------------------------------------------------------- the VAE
class ConvVAE(nn.Module):
    """A convolutional VAE for 28 x 28 images in [0, 1] (chapter 25).

    - Encoder: two convolutions of kernel 4, stride 2, padding 1 (28 -> 14 -> 7; ``width`` then ``2 * width``
      channels), each followed by a ReLU, then a flatten, and two dense heads to ``latent_dim`` values: the means mu
      and the log-variances log_var of q(z | x).
    - Decoder: a dense layer from the code to ``2 * width`` channels of 7 x 7 and a ReLU, then two transposed
      convolutions of kernel 4, stride 2, padding 1 (7 -> 14 -> 28), a ReLU between them, and NO activation at the
      end: the decoder returns LOGITS (the sigmoid is in the loss, chapter 25).
    """

    def __init__(self, latent_dim: int = 16, width: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Conv2d(1, width, 4, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, 2 * width, 4, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        self.to_mu = nn.Linear(2 * width * 7 * 7, latent_dim)
        self.to_log_var = nn.Linear(2 * width * 7 * 7, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 2 * width * 7 * 7), nn.ReLU(), nn.Unflatten(1, (2 * width, 7, 7)),
                                     nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(width, 1, 4, stride=2, padding=1))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(mu, log_var), two tensors (N, latent_dim): the two heads applied to the output of the encoder."""
        h = self.encoder(x)
        return self.to_mu(h), self.to_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """The code z (N, latent_dim). In training mode (``self.training``), z = mu + exp(log_var / 2) * eps with eps
        drawn by ``torch.randn_like(mu)`` (PyTorch's global random state); in evaluation mode, z = mu (no noise: the
        reconstruction and the validation measures are deterministic)."""
        if not self.training:
            return mu
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """The logits (N, 1, 28, 28) of the images decoded from codes z (N, latent_dim)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(logits, mu, log_var): encode, reparameterize, decode."""
        mu, log_var = self.encode(x)
        return self.decode(self.reparameterize(mu, log_var)), mu, log_var


# ---------------------------------------------------------------------------------------------- the DCGAN
def generator_body(in_dim: int, width: int) -> nn.Sequential:
    """The body of a DCGAN generator (Radford's rules, adapted to 28 x 28): an nn.Sequential (N, in_dim) ->
    (N, 1, 28, 28) in [-1, 1], whose modules are, in this order: nn.Linear(in_dim, 2 * width * 7 * 7, bias=False),
    nn.BatchNorm1d(2 * width * 7 * 7), nn.ReLU(), nn.Unflatten(1, (2 * width, 7, 7)), nn.ConvTranspose2d(2 * width,
    width, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(), nn.ConvTranspose2d(width, 1, 4,
    stride=2, padding=1), nn.Tanh(). With in_dim = 64 and width = 32: 240 321 parameters."""
    return nn.Sequential(
        nn.Linear(in_dim, 2 * width * 7 * 7, bias=False), nn.BatchNorm1d(2 * width * 7 * 7), nn.ReLU(),
        nn.Unflatten(1, (2 * width, 7, 7)),
        nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
        nn.ConvTranspose2d(width, 1, 4, stride=2, padding=1), nn.Tanh())


def discriminator_body(in_channels: int, width: int) -> nn.Sequential:
    """The body of a DCGAN discriminator: an nn.Sequential (N, in_channels, 28, 28) -> (N, 1) LOGITS, whose modules
    are, in this order: nn.Conv2d(in_channels, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2) (no batchnorm on the
    input layer), nn.Conv2d(width, 2 * width, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(2 * width),
    nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(2 * width * 7 * 7, 1). No pooling, no sigmoid. With in_channels = 1
    and width = 32: 36 577 parameters."""
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        nn.Conv2d(width, 2 * width, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(2 * width), nn.LeakyReLU(0.2),
        nn.Flatten(), nn.Linear(2 * width * 7 * 7, 1))


class Generator(nn.Module):
    """A DCGAN generator: noise z (N, latent_dim) -> images (N, 1, 28, 28) in [-1, 1]. With ``n_classes > 0`` it is
    conditional (a cGAN, chapter 27): the label goes through an nn.Embedding(n_classes, n_classes) whose output is
    concatenated AFTER z, before the body."""

    def __init__(self, latent_dim: int = 64, width: int = 32, n_classes: int = 0):
        super().__init__()
        self.latent_dim, self.n_classes = latent_dim, n_classes
        self.embed = nn.Embedding(n_classes, n_classes) if n_classes else None
        self.net = generator_body(latent_dim + n_classes, width)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """The images G(z) (N, 1, 28, 28). A conditional generator needs labels (N,) int64 (ValueError without them);
        an unconditional one refuses them (ValueError)."""
        if (labels is None) != (self.embed is None):
            raise ValueError("a conditional generator needs labels, an unconditional one takes none")
        if self.embed is not None:
            z = torch.cat([z, self.embed(labels)], dim=1)
        return self.net(z)


class Discriminator(nn.Module):
    """A DCGAN discriminator: images (N, 1, 28, 28) in [-1, 1] -> logits (N, 1). With ``n_classes > 0`` it is
    conditional: the label goes through an nn.Embedding(n_classes, 28 * 28), reshaped into a second channel of
    28 x 28 put AFTER the image, before the body."""

    def __init__(self, width: int = 32, n_classes: int = 0):
        super().__init__()
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, 28 * 28) if n_classes else None
        self.net = discriminator_body(2 if n_classes else 1, width)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """The logits D(x) (N, 1). Labels as for the Generator: required when conditional, refused otherwise
        (ValueError)."""
        if (labels is None) != (self.embed is None):
            raise ValueError("a conditional discriminator needs labels, an unconditional one takes none")
        if self.embed is not None:
            x = torch.cat([x, self.embed(labels).view(-1, 1, 28, 28)], dim=1)
        return self.net(x)


def init_dcgan_weights(module: nn.Module) -> None:
    """For ``G.apply(init_dcgan_weights)`` (Radford): the weights of nn.Linear, nn.Conv2d and nn.ConvTranspose2d
    drawn from N(0, 0.02^2), their biases untouched; the weights of nn.BatchNorm1d and nn.BatchNorm2d drawn from
    N(1, 0.02^2) and their biases set to 0; every other module (embeddings included) is left as it is."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
