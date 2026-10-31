"""genbench: a benchmark of generative models on Fashion-MNIST, a convolutional VAE against a DCGAN, measured by a
protocol fixed before the experiments.

    from genbench import Classifier, ConvVAE, Generator, Discriminator, train_vae, train_gan
    from genbench import classifier_features, frechet_distance, class_report, nearest_neighbors, sample_images

The networks (models.py), their training (train.py) and their evaluation (evaluate.py) are checked by
test_genbench.py.
"""

from .evaluate import (accuracy, class_report, classifier_features, decode_codes, frechet_distance, image_grid,
                       interpolate, kl_per_latent, nearest_neighbors, predict_proba, sample_images)
from .models import (N_CLASSES, Classifier, ConvVAE, Discriminator, Generator, discriminator_body, generator_body,
                     init_dcgan_weights, n_parameters)
from .train import (LOG_COLUMNS, gan_step, images_to_tensor, library_versions, load_checkpoint, save_checkpoint,
                    save_config, set_seed, train_classifier, train_gan, train_vae, vae_loss)

__all__ = ["Classifier", "ConvVAE", "Discriminator", "Generator", "LOG_COLUMNS", "N_CLASSES", "accuracy",
           "class_report", "classifier_features", "decode_codes", "discriminator_body", "frechet_distance",
           "gan_step", "generator_body", "image_grid", "images_to_tensor", "init_dcgan_weights", "interpolate",
           "kl_per_latent", "library_versions", "load_checkpoint", "n_parameters", "nearest_neighbors",
           "predict_proba", "sample_images", "save_checkpoint", "save_config", "set_seed", "train_classifier",
           "train_gan", "train_vae", "vae_loss"]
