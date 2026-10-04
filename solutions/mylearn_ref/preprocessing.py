"""Scalers, imputer, encoders and PCA — mylearn, chapter 12 (Data Preparation).

Preprocessing transformers "à la scikit-learn", in NumPy only: ``fit`` learns
statistics on the training set, ``transform`` applies exactly the same statistics to
any new data (validation, test, production), ``fit_transform`` does both on the same
data and ``inverse_transform`` goes back to the original units. Fitting a transformer
on the test set is a data leak (chapter 8).

Reference implementation: read it only after trying (``mon_travail/mylearn/preprocessing.py``).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike


def _check_numeric(X: ArrayLike, name: str = "X", allow_nan: bool = False) -> np.ndarray:
    """X as a 2-D float64 array (a copy), with finite values unless NaN are allowed."""
    try:
        X = np.array(X, dtype=float)               # np.array copies: the caller's data are never modified
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold numbers only ({exc})") from None
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got an array with {X.ndim} "
                         "dimension(s): use X.reshape(-1, 1) for a single feature")
    if np.isinf(X).any():
        raise ValueError(f"{name} contains infinite values")
    if not allow_nan and np.isnan(X).any():
        raise ValueError(f"{name} contains NaN values: remove the missing values first (SimpleImputer)")
    return X


def _check_categorical(X: ArrayLike, name: str = "X") -> np.ndarray:
    """X as a 2-D object array of categories, without missing values."""
    X = np.array(X, dtype=object)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got an array with {X.ndim} "
                         "dimension(s): use a list of lists, or X.reshape(-1, 1) for a single column")
    for value in X.ravel():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError(f"{name} contains missing values (None or NaN): impute them first")
    return X


def _check_fitted(estimator, attribute: str) -> None:
    """RuntimeError if `estimator` has not been fitted (its learnt attribute does not exist yet)."""
    if not hasattr(estimator, attribute):
        raise RuntimeError(f"this {type(estimator).__name__} is not fitted yet: call fit(X) first")


def _check_n_features(estimator, X: np.ndarray, name: str = "X") -> None:
    if X.shape[1] != estimator.n_features_in_:
        raise ValueError(f"{name} has {X.shape[1]} feature(s), but {type(estimator).__name__} was fitted "
                         f"with {estimator.n_features_in_}")


def _learn_categories(estimator, X: np.ndarray) -> None:
    """categories_ (one array per column: sorted unique values, or the given lists) and n_features_in_."""
    if isinstance(estimator.categories, str):
        if estimator.categories != "auto":
            raise ValueError(f"categories must be 'auto' or one list per column, got {estimator.categories!r}")
        try:
            estimator.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        except TypeError:
            raise ValueError("a column mixes strings and numbers: its categories cannot be sorted") from None
    else:
        given = list(estimator.categories)
        if len(given) != X.shape[1]:
            raise ValueError(f"categories gives {len(given)} list(s) for {X.shape[1]} column(s)")
        categories = []
        for j, cats in enumerate(given):
            cats = np.array(list(cats), dtype=object)
            if len(set(cats.tolist())) != len(cats):
                raise ValueError(f"the categories given for column {j} contain duplicates")
            unknown = [value for value in X[:, j] if value not in set(cats.tolist())]
            if unknown:
                raise ValueError(f"column {j} holds {unknown[0]!r}, absent from the categories given for it")
            categories.append(cats)
        estimator.categories_ = categories
    estimator.n_features_in_ = X.shape[1]


def _codes(categories: np.ndarray, column: np.ndarray, j: int, unknown: str = "error") -> np.ndarray:
    """Position of every value of `column` in `categories` (-1 for an unknown value with unknown='ignore')."""
    index = {value: i for i, value in enumerate(categories.tolist())}
    codes = np.empty(len(column), dtype=int)
    for i, value in enumerate(column):
        code = index.get(value, -1)
        if code < 0 and unknown == "error":
            raise ValueError(f"column {j} holds {value!r}, a category not seen in fit")
        codes[i] = code
    return codes


class StandardScaler:
    """Standardise every feature: ``z = (x - mean_) / scale_``.

    The statistics are learnt in ``fit`` on the training set: per-feature mean and
    population variance (ddof=0). A feature with zero variance gets ``scale_ = 1`` (it
    is centred but left unscaled, instead of dividing by 0).

    Parameters
    ----------
    with_mean : bool, default=True
        Subtract the per-feature mean.
    with_std : bool, default=True
        Divide by the per-feature standard deviation.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean of the training set (always computed).
    var_ : np.ndarray of shape (n_features,)
        Per-feature population variance (always computed).
    scale_ : np.ndarray of shape (n_features,)
        ``sqrt(var_)`` with zeros replaced by 1; all ones if ``with_std=False``.
    n_features_in_ : int
        Number of features seen in ``fit``.
    n_samples_seen_ : int
        Number of samples seen in ``fit``.

    Notes
    -----
    Tested against ``sklearn.preprocessing.StandardScaler`` (``mean_``, ``var_``,
    ``scale_``, ``transform``, ``inverse_transform``) with the default options, plus the
    property "the transformed training set has mean 0 and standard deviation 1 per
    feature". Difference: scikit-learn stores None instead of arrays for ``var_`` and
    ``scale_`` when ``with_std=False``.

    Examples
    --------
    >>> X = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    >>> scaler = StandardScaler().fit(X)
    >>> scaler.mean_
    array([ 3., 10.])
    >>> scaler.scale_.round(4)  # the constant feature keeps scale 1
    array([1.633, 1.   ])
    >>> scaler.transform(X).round(4)
    array([[-1.2247,  0.    ],
           [ 0.    ,  0.    ],
           [ 1.2247,  0.    ]])
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Learn the per-feature mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, finite values.
        y : None
            Ignored; accepted for compatibility with pipelines (scikit-learn convention).

        Returns
        -------
        Self
            The fitted scaler (``self``).

        Raises
        ------
        ValueError
            If ``X`` is not 2-D or contains NaN or infinite values.
        """
        X = _check_numeric(X)
        if len(X) == 0:
            raise ValueError("X has no sample: fit needs at least one row")
        self.mean_ = X.mean(axis=0)
        # population variance (ddof=0); a constant column is detected exactly (max == min): computed in floats, the
        # variance of a column of 0.1 is about 1e-34, not 0, and dividing by its root would blow the column up
        constant = X.max(axis=0) == X.min(axis=0)
        self.var_ = np.where(constant, 0.0, X.var(axis=0))
        if self.with_std:
            self.scale_ = np.where(constant, 1.0, np.sqrt(self.var_))    # a constant feature is centred, not scaled
        else:
            self.scale_ = np.ones(X.shape[1])
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Standardise X with the statistics learnt in ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform (train, validation, test or new data).

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Float64 array.

        Raises
        ------
        RuntimeError
            If the scaler is not fitted.
        ValueError
            If ``X`` is not 2-D, contains NaN or infinite values, or has a number of
            features different from ``n_features_in_``.
        """
        _check_fitted(self, "mean_")
        X = _check_numeric(X)
        _check_n_features(self, X)
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """Fit on X, then return the transformed X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Standardised training data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> np.ndarray:
        """Undo the standardisation: back to the original units.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Standardised data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            ``Z * scale_ + mean_`` (without ``mean_`` if ``with_mean=False``).

        Raises
        ------
        RuntimeError
            If the scaler is not fitted.
        ValueError
            If ``Z`` is not 2-D or has a wrong number of features.
        """
        _check_fitted(self, "mean_")
        Z = _check_numeric(Z, "Z", allow_nan=True)
        _check_n_features(self, Z, "Z")
        X = Z * self.scale_
        return X + self.mean_ if self.with_mean else X


class MinMaxScaler:
    """Rescale every feature linearly so that its training range maps to feature_range.

    ``X_t = X * scale_ + min_`` with ``scale_ = (b - a) / data_range_`` and
    ``min_ = a - data_min_ * scale_``, where ``(a, b) = feature_range``: the training
    minimum becomes a and the training maximum becomes b. New data can fall outside
    [a, b] unless ``clip=True``.

    Parameters
    ----------
    feature_range : tuple of (float, float), default=(0.0, 1.0)
        Target range (a, b) with a < b, e.g. (0, 1) or (-1, 1).
    clip : bool, default=False
        Clip the transformed values to ``feature_range``.

    Attributes
    ----------
    data_min_ : np.ndarray of shape (n_features,)
        Per-feature minimum of the training set.
    data_max_ : np.ndarray of shape (n_features,)
        Per-feature maximum of the training set.
    data_range_ : np.ndarray of shape (n_features,)
        ``data_max_ - data_min_``; a constant feature (range 0) is treated as having
        range 1 when computing ``scale_`` (no division by 0).
    scale_ : np.ndarray of shape (n_features,)
        Per-feature multiplicative factor.
    min_ : np.ndarray of shape (n_features,)
        Per-feature additive term.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Notes
    -----
    Tested against ``sklearn.preprocessing.MinMaxScaler`` (``data_min_``, ``data_max_``,
    ``scale_``, ``min_``, ``transform`` and ``inverse_transform``, with ``clip`` True
    and False).

    Examples
    --------
    >>> X = np.array([[-1.0, 2.0], [-0.5, 6.0], [0.0, 10.0], [1.0, 18.0]])
    >>> scaler = MinMaxScaler().fit(X)
    >>> scaler.transform(X)
    array([[0.  , 0.  ],
           [0.25, 0.25],
           [0.5 , 0.5 ],
           [1.  , 1.  ]])
    >>> scaler.transform(np.array([[2.0, 2.0]]))
    array([[1.5, 0. ]])
    >>> MinMaxScaler(clip=True).fit(X).transform(np.array([[2.0, 2.0]]))
    array([[1., 0.]])
    """

    def __init__(
        self, feature_range: tuple[float, float] = (0.0, 1.0), clip: bool = False
    ) -> None:
        self.feature_range = feature_range
        self.clip = clip

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Learn the per-feature minimum and maximum.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, finite values.
        y : None
            Ignored.

        Returns
        -------
        Self
            The fitted scaler (``self``).

        Raises
        ------
        ValueError
            If ``feature_range[0] >= feature_range[1]``, or ``X`` is not 2-D or
            contains NaN or infinite values.
        """
        a, b = self._range()
        X = _check_numeric(X)
        if len(X) == 0:
            raise ValueError("X has no sample: fit needs at least one row")
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (b - a) / np.where(self.data_range_ == 0.0, 1.0, self.data_range_)
        self.min_ = a - self.data_min_ * self.scale_
        self.n_features_in_ = X.shape[1]
        return self

    def _range(self) -> tuple[float, float]:
        """The validated feature_range (a, b), with a < b."""
        try:
            a, b = (float(v) for v in self.feature_range)
        except (TypeError, ValueError):
            raise ValueError(f"feature_range must be a pair of numbers (a, b), got {self.feature_range!r}") from None
        if not a < b:
            raise ValueError(f"feature_range must satisfy a < b, got {self.feature_range!r}")
        return a, b

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Rescale X with the statistics learnt in ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Float64 array (clipped to ``feature_range`` if ``clip=True``).

        Raises
        ------
        RuntimeError
            If the scaler is not fitted.
        ValueError
            If ``X`` is invalid or has a number of features different from
            ``n_features_in_``.
        """
        _check_fitted(self, "scale_")
        X = _check_numeric(X)
        _check_n_features(self, X)
        X_t = X * self.scale_ + self.min_
        if self.clip:
            a, b = self._range()
            X_t = np.clip(X_t, a, b)
        return X_t

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """Fit on X, then return the transformed X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Rescaled training data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> np.ndarray:
        """Undo the rescaling: back to the original units.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Rescaled data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            ``(Z - min_) / scale_``.

        Raises
        ------
        RuntimeError
            If the scaler is not fitted.
        ValueError
            If ``Z`` is not 2-D or has a wrong number of features.
        """
        _check_fitted(self, "scale_")
        Z = _check_numeric(Z, "Z", allow_nan=True)
        _check_n_features(self, Z, "Z")
        return (Z - self.min_) / self.scale_


class SimpleImputer:
    """Replace the missing values (NaN) of a numeric array by a per-feature statistic.

    The statistic of every column is learnt in ``fit`` from its observed (non-NaN)
    values: "mean", "median", "most_frequent" (the smallest value among ties) or
    "constant" (``fill_value``). ``transform`` replaces every NaN of column j by
    ``statistics_[j]``. Categorical columns are imputed with scikit-learn in the notebook.

    Parameters
    ----------
    strategy : str, default="mean"
        One of "mean", "median", "most_frequent", "constant".
    fill_value : float or None, default=None
        Value used by the "constant" strategy (None means 0.0).

    Attributes
    ----------
    statistics_ : np.ndarray of shape (n_features,)
        Replacement value of every column.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Notes
    -----
    Tested against ``sklearn.impute.SimpleImputer`` on float arrays whose columns contain
    at least one observed value (``statistics_`` and ``transform``). Documented
    difference: a column entirely NaN raises ``ValueError`` here, while scikit-learn
    silently drops it.

    Examples
    --------
    >>> X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
    >>> imputer = SimpleImputer(strategy="mean").fit(X)
    >>> imputer.statistics_
    array([2., 6.])
    >>> imputer.transform(X)
    array([[1., 6.],
           [3., 4.],
           [2., 8.]])
    """

    def __init__(self, strategy: str = "mean", fill_value: float | None = None) -> None:
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Learn the replacement value of every column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Float data, missing values encoded as ``np.nan``.
        y : None
            Ignored.

        Returns
        -------
        Self
            The fitted imputer (``self``).

        Raises
        ------
        ValueError
            If ``strategy`` is unknown, ``X`` is not 2-D, or a column is entirely NaN
            with the "mean", "median" or "most_frequent" strategy.
        """
        if self.strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError("strategy must be 'mean', 'median', 'most_frequent' or 'constant', "
                             f"got {self.strategy!r}")
        X = _check_numeric(X, allow_nan=True)
        if self.strategy == "constant":
            fill = 0.0 if self.fill_value is None else float(self.fill_value)
            self.statistics_ = np.full(X.shape[1], fill)
        else:
            statistics = []
            for j in range(X.shape[1]):
                observed = X[~np.isnan(X[:, j]), j]
                if observed.size == 0:
                    raise ValueError(f"column {j} is entirely NaN: the {self.strategy!r} strategy has no value "
                                     "to learn (drop the column, or use strategy='constant')")
                if self.strategy == "mean":
                    statistics.append(observed.mean())
                elif self.strategy == "median":
                    statistics.append(np.median(observed))
                else:                                # the most frequent value, the smallest one among ties
                    values, counts = np.unique(observed, return_counts=True)
                    statistics.append(values[np.argmax(counts)])
            self.statistics_ = np.array(statistics, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Replace every NaN by the statistic of its column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Float data with NaN.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Float64 array without NaN (X itself is not modified).

        Raises
        ------
        RuntimeError
            If the imputer is not fitted.
        ValueError
            If ``X`` is not 2-D or has a number of features different from
            ``n_features_in_``.
        """
        _check_fitted(self, "statistics_")
        X = _check_numeric(X, allow_nan=True)
        _check_n_features(self, X)
        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """Fit on X, then return the imputed X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Float data with NaN.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Imputed data.
        """
        return self.fit(X).transform(X)


class OrdinalEncoder:
    """Encode the categories of every column as integer codes 0..K-1.

    With ``categories="auto"`` the codes follow the sorted order of the categories seen
    in ``fit`` (alphabetical for strings). A list of categories per column imposes the
    order instead, which is what ordinal data need (S < M < L, rainbow colours...).

    Parameters
    ----------
    categories : "auto" or list of lists, default="auto"
        "auto", or one list of categories per column, in the desired order.

    Attributes
    ----------
    categories_ : list of np.ndarray
        ``categories_[j]`` holds the categories of column j; the code of a category is
        its position in this array.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Notes
    -----
    Tested against ``sklearn.preprocessing.OrdinalEncoder`` (``categories_``,
    ``transform``, ``inverse_transform``).

    Examples
    --------
    >>> X = [["red", "S"], ["blue", "M"], ["red", "L"]]
    >>> encoder = OrdinalEncoder().fit(X)
    >>> [cats.tolist() for cats in encoder.categories_]
    [['blue', 'red'], ['L', 'M', 'S']]
    >>> encoder.transform(X)
    array([[1., 2.],
           [0., 1.],
           [1., 0.]])
    >>> OrdinalEncoder(categories=[["S", "M", "L"]]).fit_transform([["S"], ["L"], ["M"]])
    array([[0.],
           [2.],
           [1.]])
    """

    def __init__(self, categories: str | list[list] = "auto") -> None:
        self.categories = categories

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Learn (or check) the categories of every column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Strings or numbers (object arrays and lists of lists are fine).
        y : None
            Ignored.

        Returns
        -------
        Self
            The fitted encoder (``self``).

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, if ``categories`` does not give one list per column,
            or if a value of X is absent from the given list of its column.
        """
        X = _check_categorical(X)
        _learn_categories(self, X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Replace every category by its code.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with the same columns as in ``fit``.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Float64 codes.

        Raises
        ------
        RuntimeError
            If the encoder is not fitted.
        ValueError
            If a category was not seen in ``fit`` or X has a wrong number of features.
        """
        _check_fitted(self, "categories_")
        X = _check_categorical(X)
        _check_n_features(self, X)
        codes = [_codes(self.categories_[j], X[:, j], j) for j in range(X.shape[1])]
        return np.column_stack(codes).astype(float) if codes else np.empty((len(X), 0))

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """Fit on X, then return the codes of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Categorical data.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Float64 codes.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Replace every code by its category.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Integer-valued codes.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Object array of the original categories.

        Raises
        ------
        RuntimeError
            If the encoder is not fitted.
        ValueError
            If a code is not a valid position or X has a wrong number of features.
        """
        _check_fitted(self, "categories_")
        X = _check_numeric(X)
        _check_n_features(self, X)
        out = np.empty(X.shape, dtype=object)
        for j, cats in enumerate(self.categories_):
            codes = X[:, j]
            if np.any(codes != np.round(codes)) or np.any((codes < 0) | (codes >= len(cats))):
                raise ValueError(f"column {j} holds codes that are not integers between 0 and {len(cats) - 1}")
            out[:, j] = cats[codes.astype(int)]
        return out


class OneHotEncoder:
    """One-hot encode every categorical column (dense output).

    Column j with K_j categories becomes a block of K_j columns of 0/1, with a single 1
    at the position of the category. The blocks of all the columns are concatenated.
    ``drop`` removes one column per block (the first category), which avoids a
    perfectly redundant column for linear models.

    Parameters
    ----------
    categories : "auto" or list of lists, default="auto"
        "auto" (sorted categories seen in ``fit``) or one list of categories per column.
    drop : {None, "first", "if_binary"}, default=None
        None: keep every category. "first": drop the first category of every column.
        "if_binary": drop the first category only for the columns with exactly 2
        categories.
    handle_unknown : {"error", "ignore"}, default="error"
        What ``transform`` does with a category not seen in ``fit``: raise a
        ``ValueError``, or encode it as an all-zero block.

    Attributes
    ----------
    categories_ : list of np.ndarray
        ``categories_[j]`` holds the categories of column j (dropped ones included).
    drop_idx_ : np.ndarray or None
        None if ``drop`` is None; else an object array of shape (n_features,) holding
        the index of the dropped category of every column, or None for a column where
        nothing is dropped (as scikit-learn).
    n_features_in_ : int
        Number of features seen in ``fit``.

    Notes
    -----
    Tested against ``sklearn.preprocessing.OneHotEncoder(sparse_output=False)`` with the
    same options: ``categories_``, ``transform``, ``inverse_transform`` and
    ``get_feature_names_out``.

    Examples
    --------
    >>> X = [["Biscoe", "male"], ["Dream", "female"], ["Biscoe", "female"]]
    >>> encoder = OneHotEncoder().fit(X)
    >>> encoder.transform(X)
    array([[1., 0., 0., 1.],
           [0., 1., 1., 0.],
           [1., 0., 1., 0.]])
    >>> encoder.get_feature_names_out(["island", "sex"]).tolist()
    ['island_Biscoe', 'island_Dream', 'sex_female', 'sex_male']
    >>> OneHotEncoder(drop="if_binary").fit_transform(X)
    array([[0., 1.],
           [1., 0.],
           [0., 0.]])
    """

    def __init__(
        self,
        categories: str | list[list] = "auto",
        drop: str | None = None,
        handle_unknown: str = "error",
    ) -> None:
        self.categories = categories
        self.drop = drop
        self.handle_unknown = handle_unknown

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Learn the categories of every column and which ones are dropped.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Strings or numbers (object arrays and lists of lists are fine).
        y : None
            Ignored.

        Returns
        -------
        Self
            The fitted encoder (``self``).

        Raises
        ------
        ValueError
            If ``drop`` or ``handle_unknown`` is not a valid option, ``X`` is not 2-D,
            or a value of X is absent from the given list of its column.
        """
        if self.drop not in (None, "first", "if_binary"):
            raise ValueError(f"drop must be None, 'first' or 'if_binary', got {self.drop!r}")
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {self.handle_unknown!r}")
        X = _check_categorical(X)
        _learn_categories(self, X)
        if self.drop is None:
            self.drop_idx_ = None
        else:
            self.drop_idx_ = np.array([0 if self.drop == "first" or len(cats) == 2 else None
                                       for cats in self.categories_], dtype=object)
        return self

    def _kept(self, j: int) -> list[int]:
        """Indices of the categories of column j that get an output column."""
        dropped = None if self.drop_idx_ is None else self.drop_idx_[j]
        return [i for i in range(len(self.categories_[j])) if i != dropped]

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Return the one-hot encoding of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with the same columns as in ``fit``.

        Returns
        -------
        np.ndarray of shape (n_samples, n_output_features)
            Float64 0/1 array; ``n_output_features`` is the total number of kept
            categories.

        Raises
        ------
        RuntimeError
            If the encoder is not fitted.
        ValueError
            If a category was not seen in ``fit`` and ``handle_unknown="error"``, or X
            has a wrong number of features.
        """
        _check_fitted(self, "categories_")
        X = _check_categorical(X)
        _check_n_features(self, X)
        blocks = []
        for j, cats in enumerate(self.categories_):
            codes = _codes(cats, X[:, j], j, self.handle_unknown)
            block = np.zeros((len(X), len(cats)))
            known = codes >= 0
            block[np.flatnonzero(known), codes[known]] = 1.0     # an unknown category stays all zeros
            blocks.append(block[:, self._kept(j)])
        return np.hstack(blocks) if blocks else np.empty((len(X), 0))

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """Fit on X, then return the one-hot encoding of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Categorical data.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples, n_output_features)
            Float64 0/1 array.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Convert a one-hot encoding back to the categories.

        In a block, the position of the 1 gives the category. An all-zero block gives
        the dropped category of that column if one was dropped, else None (a category
        that was unknown with ``handle_unknown="ignore"``), as scikit-learn.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_output_features)
            One-hot encoded data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Object array of categories (or None).

        Raises
        ------
        RuntimeError
            If the encoder is not fitted.
        ValueError
            If X does not have ``n_output_features`` columns.
        """
        _check_fitted(self, "categories_")
        X = _check_numeric(X)
        widths = [len(self._kept(j)) for j in range(self.n_features_in_)]
        if X.shape[1] != sum(widths):
            raise ValueError(f"X has {X.shape[1]} column(s), but the encoding has {sum(widths)}")
        out = np.empty((len(X), self.n_features_in_), dtype=object)
        start = 0
        for j, width in enumerate(widths):
            block = X[:, start:start + width]
            start += width
            kept = self._kept(j)
            dropped = None if self.drop_idx_ is None else self.drop_idx_[j]
            for i in range(len(X)):
                if width == 0 or not block[i].any():          # all zeros: the dropped (or an unknown) category
                    out[i, j] = None if dropped is None else self.categories_[j][dropped]
                else:
                    out[i, j] = self.categories_[j][kept[int(np.argmax(block[i]))]]
        return out

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return the names of the output columns.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Names of the input columns; None uses "x0", "x1", ...

        Returns
        -------
        np.ndarray of shape (n_output_features,)
            Names "<column>_<category>", e.g. "x0_Biscoe" or "island_Biscoe", in the
            order of the output columns (dropped categories excluded).

        Raises
        ------
        RuntimeError
            If the encoder is not fitted.
        ValueError
            If ``input_features`` does not have ``n_features_in_`` names.
        """
        _check_fitted(self, "categories_")
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        input_features = list(input_features)
        if len(input_features) != self.n_features_in_:
            raise ValueError(f"input_features has {len(input_features)} name(s), but the encoder was fitted "
                             f"with {self.n_features_in_} column(s)")
        names = [f"{input_features[j]}_{self.categories_[j][i]}"
                 for j in range(self.n_features_in_) for i in self._kept(j)]
        return np.array(names, dtype=object)


class PCA:
    """Principal component analysis by singular value decomposition.

    The data are centred (``mean_``), then ``np.linalg.svd(Xc, full_matrices=False)``
    gives ``Xc = U S Vᵀ``: the rows of Vᵀ are the principal directions, sorted by
    decreasing variance ``S² / (n_samples - 1)``. ``transform`` projects on the kept
    directions: ``Z = (X - mean_) @ components_.T``, divided by
    ``sqrt(explained_variance_)`` if ``whiten``. The sign of a direction is arbitrary:
    as scikit-learn, each component is flipped so that its largest-magnitude
    coordinate is positive.

    Parameters
    ----------
    n_components : int, float or None, default=None
        Int k >= 1: keep k components. Float in (0, 1): keep the smallest k whose
        cumulative explained variance ratio is strictly greater than it (as
        scikit-learn). None: keep ``min(n_samples, n_features)`` components.
    whiten : bool, default=False
        Rescale the projected coordinates to unit variance.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean of the training set.
    components_ : np.ndarray of shape (n_components_, n_features)
        Principal directions, one unit vector per row.
    explained_variance_ : np.ndarray of shape (n_components_,)
        Variance along each direction (ddof=1).
    explained_variance_ratio_ : np.ndarray of shape (n_components_,)
        Fraction of the total variance explained by each kept direction.
    singular_values_ : np.ndarray of shape (n_components_,)
        Singular values S of the kept directions.
    n_components_ : int
        Number of kept components.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Notes
    -----
    Tested against ``sklearn.decomposition.PCA(svd_solver="full")``:
    ``explained_variance_``, ``explained_variance_ratio_`` and ``singular_values_``
    allclose; ``components_`` and ``transform`` compared up to the sign of each
    component; ``inverse_transform(transform(X))`` equals X when all components are kept.

    Examples
    --------
    >>> X = np.array([[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0],
    ...               [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
    >>> pca = PCA(n_components=2).fit(X)
    >>> pca.explained_variance_ratio_.round(4)
    array([0.9924, 0.0076])
    >>> pca.components_.round(4)
    array([[ 0.8385,  0.5449],
           [-0.5449,  0.8385]])
    >>> PCA(n_components=0.95).fit(X).n_components_
    1
    """

    def __init__(self, n_components: int | float | None = None, whiten: bool = False) -> None:
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Compute the principal components of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        Self
            The fitted PCA (``self``).

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, or ``n_components`` is not None, an int in
            1..min(n_samples, n_features) or a float in (0, 1).
        """
        X = _check_numeric(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA needs at least 2 samples (the variances use n_samples - 1)")
        k_max = min(n_samples, n_features)
        nc = self.n_components
        if nc is not None:
            if isinstance(nc, (bool, np.bool_)):
                raise ValueError(f"n_components must be None, an int or a float in (0, 1), got {nc!r}")
            if isinstance(nc, (int, np.integer)):
                if not 1 <= nc <= k_max:
                    raise ValueError(f"n_components={nc} must be between 1 and min(n_samples, n_features) = {k_max}")
            elif isinstance(nc, (float, np.floating)):
                if not 0.0 < nc < 1.0:
                    raise ValueError(f"a float n_components must be in (0, 1), got {nc!r}")
            else:
                raise ValueError(f"n_components must be None, an int or a float in (0, 1), got {nc!r}")
        self.mean_ = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        signs = np.sign(Vt[np.arange(len(Vt)), np.argmax(np.abs(Vt), axis=1)])
        Vt = Vt * np.where(signs == 0, 1.0, signs)[:, None]    # largest-magnitude coordinate positive
        variance = S**2 / (n_samples - 1)
        total = variance.sum()
        ratio = variance / total if total > 0 else np.zeros_like(variance)
        if nc is None:
            k = k_max
        elif isinstance(nc, (float, np.floating)):
            k = min(int(np.searchsorted(np.cumsum(ratio), nc, side="right")) + 1, len(ratio))
        else:
            k = int(nc)
        self.components_ = Vt[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.singular_values_ = S[:k]
        self.n_components_ = k
        self.n_features_in_ = n_features
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Project X on the principal components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components_)
            Coordinates in the principal basis (unit variance per column if ``whiten``).

        Raises
        ------
        RuntimeError
            If the PCA is not fitted.
        ValueError
            If X has a number of features different from ``n_features_in_``.
        """
        _check_fitted(self, "components_")
        X = _check_numeric(X)
        _check_n_features(self, X)
        Z = (X - self.mean_) @ self.components_.T
        return Z / np.sqrt(self.explained_variance_) if self.whiten else Z

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """Fit on X, then return its projection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components_)
            Projected training data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> np.ndarray:
        """Map projected coordinates back to the original space (reconstruction).

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components_)
            Coordinates in the principal basis.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            ``Z @ components_ + mean_`` (whitening undone first); equal to the original
            data when all components are kept.

        Raises
        ------
        RuntimeError
            If the PCA is not fitted.
        ValueError
            If Z does not have ``n_components_`` columns.
        """
        _check_fitted(self, "components_")
        Z = _check_numeric(Z, "Z")
        if Z.shape[1] != self.n_components_:
            raise ValueError(f"Z has {Z.shape[1]} column(s), but the PCA keeps {self.n_components_} component(s)")
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_
