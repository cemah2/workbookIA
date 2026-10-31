"""The evaluation of genbench: the features of the reference classifier, the Fréchet distance between real and
generated images, the classes of the samples (mode collapse), their nearest neighbours in the training set
(memorization), the samples themselves and the interpolations of the gallery (chapters 25 and 27).

    real = classifier_features(clf, val_images)
    fake = classifier_features(clf, sample_images(G, 2000, seed=1))
    frechet_distance(real, fake), class_report(clf, sample_images(G, 2000, seed=1))
    distances, indices = nearest_neighbors(samples, train_images)

Every image is a float tensor (N, 1, 28, 28) in [0, 1].

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .models import ConvVAE


# ---------------------------------------------------------------------------------------------- the reference classifier
def classifier_features(model: nn.Module, images: torch.Tensor, batch_size: int = 500) -> np.ndarray:
    """The features ``model.features(x)`` of the images, computed in evaluation mode and without a graph, by
    mini-batches of batch_size: a float64 NumPy array (N, n_features). The model goes back to its previous mode."""
    raise NotImplementedError("classifier_features")  # TODO MP6.2


def predict_proba(model: nn.Module, images: torch.Tensor, batch_size: int = 500) -> np.ndarray:
    """The probabilities (N, 10) of the classes (softmax of the logits), in evaluation mode, without a graph, by
    mini-batches; a float64 NumPy array. The model goes back to its previous mode."""
    raise NotImplementedError("predict_proba")  # TODO MP6.2


def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    """The share of images whose most probable class (predict_proba) is the label."""
    raise NotImplementedError("accuracy")  # TODO MP6.2


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Provided: the squared Fréchet distance between two Gaussians fitted to two sets of features (n, d), the
    formula of the FID (chapter 27). It calls mylearn.gen_metrics.frechet_distance_from_features, the function of
    exercise 27.22, when your library has it; otherwise (the fast track does not write 27.22, or the delivered
    project runs without mylearn) the same formula computed here with scipy.linalg.sqrtm, as pytorch-fid does."""
    try:
        from mylearn.gen_metrics import frechet_distance_from_features
    except Exception:                      # no mylearn (a delivered project) or no gen_metrics module
        frechet_distance_from_features = None
    if frechet_distance_from_features is not None:
        try:
            return float(frechet_distance_from_features(real_features, fake_features))
        except NotImplementedError:        # gen_metrics.py copied, 27.22 not written (fast track)
            pass
    from scipy import linalg
    a, b = np.asarray(real_features, dtype=np.float64), np.asarray(fake_features, dtype=np.float64)
    mu1, mu2 = a.mean(axis=0), b.mean(axis=0)
    s1, s2 = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    root = linalg.sqrtm(s1 @ s2).real
    return float(max(np.sum((mu1 - mu2) ** 2) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(root), 0.0))


# ---------------------------------------------------------------------------------------------- the generators
def decode_codes(model: nn.Module, codes: torch.Tensor, labels: torch.Tensor | None = None,
                 batch_size: int = 500) -> torch.Tensor:
    """The images in [0, 1], float tensor (N, 1, 28, 28) on the CPU, made from the codes (N, latent_dim) by a
    generator, in evaluation mode, without a graph, by mini-batches: ``sigmoid(model.decode(z))`` for a VAE (a model
    with a decode method), ``(model(z, labels) + 1) / 2`` for a GAN generator (labels only for a conditional one).
    Codes and labels are moved to the device of the model's parameters. The model goes back to its previous mode."""
    raise NotImplementedError("decode_codes")  # TODO MP6.3


def sample_images(model: nn.Module, n: int, seed: int, labels: torch.Tensor | None = None) -> torch.Tensor:
    """n new images in [0, 1] (n, 1, 28, 28) from a generator (a ConvVAE or a Generator): the codes
    ``torch.randn(n, model.latent_dim, generator=torch.Generator().manual_seed(seed))`` (the prior N(0, I)), given to
    decode_codes. Same seed, same images; PyTorch's global random state is not touched."""
    raise NotImplementedError("sample_images")  # TODO MP6.3


def kl_per_latent(model: ConvVAE, images: torch.Tensor, batch_size: int = 500) -> np.ndarray:
    """The KL divergence of each latent, 0.5 * (mu^2 + exp(log_var) - 1 - log_var), averaged over the images, from
    ``model.encode`` in evaluation mode and without a graph: a float64 NumPy array (latent_dim,). A latent whose
    mean KL is below 0.01 nat is inactive (its q(z|x) is the prior whatever x: posterior collapse). The model goes
    back to its previous mode."""
    raise NotImplementedError("kl_per_latent")  # TODO MP6.3


def interpolate(model: nn.Module, z0: torch.Tensor, z1: torch.Tensor, n_steps: int, method: str = "slerp",
                label: int | None = None) -> torch.Tensor:
    """The images (n_steps, 1, 28, 28) in [0, 1] decoded along a path from the code z0 to the code z1 (latent_dim,):
    the codes of ``mylearn.vae.interpolate_latents(z0, z1, n_steps, method)`` (your function of exercise 25.20,
    imported INSIDE this function, with NumPy arrays), as float32, given to decode_codes; for a conditional
    generator, the same label for every step."""
    raise NotImplementedError("interpolate")  # TODO MP6.6


# ---------------------------------------------------------------------------------------------- the protocol
def class_report(classifier: nn.Module, images: torch.Tensor, n_classes: int = 10) -> dict:
    """What the reference classifier sees in a set of images (mode collapse): {"counts": the number of images
    predicted in each class (a list of n_classes ints), "entropy": the entropy of the shares of these counts, in
    nats, divided by ln(n_classes) (1 when every class has the same share, 0 when all are in one class),
    "max_share": the largest share, "confidence": the mean, over the images, of the probability of their predicted
    class}. Probabilities from predict_proba."""
    raise NotImplementedError("class_report")  # TODO MP6.5


def nearest_neighbors(queries: torch.Tensor, references: torch.Tensor,
                      batch_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """For each query image, its nearest reference image in pixel space (memorization): (distances, indices), two
    NumPy arrays (N,), the Euclidean distance between the flattened images (float64, NOT squared) and the index of
    the nearest reference (int64; the first one in case of a tie). Computed by blocks of batch_size queries
    (torch.cdist), so that the memory stays bounded; each block goes to the device of the references."""
    raise NotImplementedError("nearest_neighbors")  # TODO MP6.5


def image_grid(images: torch.Tensor, nrow: int, pad: int = 2) -> np.ndarray:
    """Provided: a 2-D NumPy array that puts images (N, 1, H, W) in [0, 1] on a grid of nrow images per row,
    separated by pad pixels of value 1 (white), for plt.imshow(grid, cmap="gray_r" or "gray")."""
    images = images.detach().cpu().numpy()[:, 0]
    n, h, w = images.shape
    rows = (n + nrow - 1) // nrow
    grid = np.ones((rows * (h + pad) + pad, nrow * (w + pad) + pad))
    for k, image in enumerate(images):
        r, c = divmod(k, nrow)
        grid[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = image
    return grid
