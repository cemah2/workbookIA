"""cifarkit: a CIFAR-10 image classifier in PyTorch, from the data pipeline to the delivered model.

    from cifarkit import stratified_split, channel_stats, make_transform, make_loader, build_model, fit, evaluate

The pipeline (pipeline.py), the models (models.py), the training (train.py) and the command line that classifies new
images (predict.py) are checked by test_cifarkit.py.
"""

from .models import ResidualBlock, build_model, flat_size, mlp, n_parameters, resnet, small_cnn, vgg
from .pipeline import CLASSES, CifarDataset, channel_stats, make_loader, make_transform, stratified_split
from .train import (RUN_COLUMNS, evaluate, fgsm, fit, library_versions, load_model, log_run, make_optimizer,
                    make_scheduler, predict_proba, quick_fit, read_runs, save_model, train_one_epoch)

__all__ = ["CLASSES", "CifarDataset", "RUN_COLUMNS", "ResidualBlock", "build_model", "channel_stats", "evaluate",
           "fgsm", "fit", "flat_size", "library_versions", "load_model", "log_run", "make_loader", "make_optimizer",
           "make_scheduler", "make_transform", "mlp", "n_parameters", "predict_proba", "quick_fit", "read_runs",
           "resnet", "save_model", "small_cnn", "stratified_split", "train_one_epoch", "vgg"]
