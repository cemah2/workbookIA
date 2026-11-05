"""Everything learned from the data before the model (steps PF.5 and PF.6).

Scalers, imputers and encoders (tabular data), a tokenizer or a TF-IDF (text), the normalization statistics and the
augmentation of images, the windows and the scaling of a time series: all of it is **fitted on the training split
only**, then applied as is to validation, test and new inputs. The fitted preprocessor travels with the model, in the
artifact (``predict.save_artifact``): a model without its preprocessing is not a model (ch. 23, 23.9).

    pre = build_preprocessor(config)                       # unfitted
    pipe = sklearn.pipeline.make_pipeline(pre, model)      # the classical baseline: no leak in cross-validation
    train_ds = make_dataset(train_df, pre.fit(train_df), config, train=True)   # the deep model
"""

from __future__ import annotations

import pandas as pd

from .config import Config


def build_preprocessor(config: Config):
    """An UNFITTED preprocessor with the scikit-learn API (``fit(df)``, ``transform(df)``, ``get_params``), built from
    the settings of ``config`` alone.

    Tabular: a ``ColumnTransformer`` (12.28, 15.28); text: a vectorizer or a wrapper around a tokenizer (B2, B4);
    images: a small class that stores the mean and std of the channels in ``fit`` (MP5) or the transform of the
    pre-trained model (B1.11); a series: the scaling of the windows (22.23, 24.25). It receives rows of
    ``load_examples`` as they are (``fit(df)`` or ``fit(df, y)``) and ignores the column ``id`` and the target (a model
    that reads its target or an identifier is the leak of 13.21); the group and time columns enter only as features
    chosen on purpose (the hour of the day, not the row number). ``fit`` reads only
    the frame it receives. It must survive ``sklearn.base.clone`` (no fitted attribute set in ``__init__``).
    """
    # TODO PF.5: the preprocessor of your data
    raise NotImplementedError("TODO PF.5: build_preprocessor")


def make_dataset(df: pd.DataFrame, preprocessor, config: Config, *, train: bool = False):
    """A ``torch.utils.data.Dataset`` of ``(inputs, target)`` pairs for the deep model, from the rows of ``df`` and a
    FITTED preprocessor.

    The augmentation (if any) is drawn in ``__getitem__`` and only when ``train=True`` (24.23, B7.12); validation and
    test inputs are transformed exactly as new inputs will be at prediction time. Without a target column (prediction),
    the target is ``-1`` (or NaN for a regression).
    """
    # TODO PF.6: a Dataset class (or a TensorDataset if everything fits in memory)
    raise NotImplementedError("TODO PF.6: make_dataset")
