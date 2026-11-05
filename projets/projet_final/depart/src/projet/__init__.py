"""The package of your final project (rename it: see the README, « Renommer le paquet »).

Modules, in the order of the project:

- ``config``     the settings: a typed, frozen dataclass read from ``configs/*.yaml`` (provided);
- ``data``       download, load, split and seal the dataset (PF.2 to PF.4);
- ``preprocess`` everything learned from the data before the model: scalers, encoders, tokenizers, transforms;
- ``models``     the naive predictor, the classical baseline and the deep learning model (PF.5, PF.6);
- ``train``      seeds, cross-validation, the training loop, checkpoints and the run journal (PF.5 to PF.7);
- ``evaluate``   metrics, bootstrap intervals, subgroups, errors and the one opening of the sealed test (PF.8, PF.9);
- ``predict``    the artifact of the delivered model and its predictor (PF.11);
- ``serve``      the HTTP API and the latency measurement (PF.11);
- ``cli``        the command line ``projet ...`` (provided wiring, to adapt).

A function still to write raises ``NotImplementedError("TODO PF.n: name")``: the notebooks show it as ⏳, the tests
skip it (``PF_STRICT=1`` turns those skips into failures, for the end of the project).
"""

__version__ = "0.1.0"
