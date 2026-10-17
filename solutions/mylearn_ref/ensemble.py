"""Voting, bagging, forests and boosting — mylearn, chapter 14 (Ensembles).

Many weak or unstable models combined into a strong one: weighted plurality vote,
bootstrap samples, bagging, random forests, extremely randomized trees, AdaBoost (SAMME)
and gradient boosting for regression. The members are the trees of chapter 13
(``tree.py``); each member is a fresh ``clone`` (chapter 8) of an unfitted template.

Reference implementation: read it only after trying (``mon_travail/mylearn/ensemble.py``).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .model_selection import clone  # chapter 8
from .tree import DecisionTreeClassifier, DecisionTreeRegressor  # chapter 13


def _is_int(value) -> bool:
    """An int (Python or NumPy), but not a bool."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_X(X: ArrayLike, model=None) -> np.ndarray:
    """X as a 2-D float64 array (a copy), without NaN, with at least one sample.

    With a fitted ``model``, X must also have as many columns as the data its members were fitted on.
    """
    try:
        X = np.array(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"X must hold numbers only ({exc})") from None
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got an array with {X.ndim} dimension(s): "
                         "use X.reshape(-1, 1) for a single feature")
    if np.isnan(X).any():
        raise ValueError("X contains NaN values: remove the missing values first")
    if X.shape[0] == 0:
        raise ValueError("X has no sample")
    if model is not None:
        expected = getattr(model.estimators_[0], "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(f"X has {X.shape[1]} feature(s), but {type(model).__name__} was fitted with {expected}")
    return X


def _check_y(y: ArrayLike, n: int, regression: bool = False) -> np.ndarray:
    """y as a 1-D array (float64 for a regression) of length n, without NaN."""
    try:
        y = np.array(y, dtype=float) if regression else np.array(y)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"y must hold numbers only ({exc})") from None
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if y.dtype.kind in "fc" and np.isnan(y).any():
        raise ValueError("y contains NaN values: remove the missing values first")
    if len(y) != n:
        raise ValueError(f"X has {n} samples but y has {len(y)}")
    return y


def _check_n_estimators(n_estimators) -> None:
    if not (_is_int(n_estimators) and n_estimators >= 1):
        raise ValueError(f"n_estimators must be an int >= 1, got {n_estimators!r}")


def _check_fitted(model) -> None:
    if not hasattr(model, "estimators_"):
        raise RuntimeError(f"this {type(model).__name__} is not fitted yet: call fit(X, y) first")


def _accuracy(model, X: ArrayLike, y: ArrayLike) -> float:
    predictions = model.predict(X)
    y = np.asarray(y)
    if y.shape != predictions.shape:
        raise ValueError(f"X has {len(predictions)} samples but y has shape {y.shape}")
    return float(np.mean(predictions == y))


def _aligned_proba(member, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """member.predict_proba(X) with its columns placed on `classes` (a member may lack some classes)."""
    proba = np.asarray(member.predict_proba(X), dtype=float)
    out = np.zeros((X.shape[0], len(classes)))
    out[:, np.searchsorted(classes, member.classes_)] = proba
    return out


def _mean_importances(trees: list, X: ArrayLike) -> np.ndarray:
    """Mean importances of the trees that split at least once, renormalised to sum 1 (as scikit-learn).

    A tree whose root is a leaf (a pure bootstrap sample) has all-zero importances: averaging it in would make the
    importances of the forest sum to less than 1.
    """
    split = [tree.feature_importances_ for tree in trees if np.sum(tree.feature_importances_) > 0]
    if not split:
        return np.zeros(np.shape(X)[1])
    mean = np.mean(split, axis=0)
    return mean / mean.sum()


def _soft_vote(model, X: ArrayLike) -> np.ndarray:
    _check_fitted(model)
    X = _check_X(X, model)
    total = np.zeros((X.shape[0], len(model.classes_)))
    for member in model.estimators_:
        total += _aligned_proba(member, X, model.classes_)
    return total / len(model.estimators_)


def plurality_vote(predictions: ArrayLike, weights: ArrayLike | None = None) -> np.ndarray:
    """Combine the labels predicted by several voters with a weighted plurality vote.

    For every sample, each label receives the sum of the weights of the voters that
    predicted it; the label with the largest total wins (ties: the smallest label).
    Weights may be negative, as in the book (a voter that is usually wrong).

    Parameters
    ----------
    predictions : array-like of shape (n_voters, n_samples)
        Row v holds the labels predicted by voter v.
    weights : array-like of shape (n_voters,) or None, default=None
        Weight of every voter; None means all 1 (simple majority).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Winning label of every sample.

    Raises
    ------
    ValueError
        If ``predictions`` is not 2-D or ``weights`` does not have one value per voter.

    Notes
    -----
    Tested against ``scipy.stats.mode`` for unweighted votes and a brute-force loop for
    weighted votes.

    Examples
    --------
    >>> predictions = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 1]])  # 3 voters, 3 samples
    >>> plurality_vote(predictions)
    array([0, 0, 1])
    >>> plurality_vote(predictions, weights=[1.0, 1.0, 3.0])
    array([1, 0, 1])
    """
    predictions = np.asarray(predictions)
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2-D (n_voters, n_samples), got shape {predictions.shape}")
    if predictions.shape[0] == 0:
        raise ValueError("predictions has no voter")
    if predictions.dtype.kind in "fc" and np.isnan(predictions).any():
        raise ValueError("predictions contains NaN values: remove the missing values first")
    n_voters, n_samples = predictions.shape
    if weights is None:
        w = np.ones(n_voters)
    else:
        try:
            w = np.array(weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weights must hold numbers only ({exc})") from None
        if w.shape != (n_voters,):
            raise ValueError(f"weights must have one value per voter, shape ({n_voters},), got {w.shape}")
        if np.isnan(w).any():
            raise ValueError("weights contains NaN values: remove the missing values first")
    labels, codes = np.unique(predictions, return_inverse=True)   # every label starts with a score of 0
    codes = codes.reshape(n_voters, n_samples)
    table = np.zeros((n_samples, len(labels)))
    for v in range(n_voters):
        table[np.arange(n_samples), codes[v]] += w[v]
    return labels[np.argmax(table, axis=1)]                        # argmax: the smallest of tied labels


def bootstrap_indices(
    n_samples: int, n_draws: int | None = None, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Draw indices uniformly with replacement (a bootstrap sample).

    On average a bootstrap sample of size n contains about 1 - (1 - 1/n)^n ≈ 63.2 % of
    the distinct indices; the others are "out-of-bag".

    Parameters
    ----------
    n_samples : int
        Size n of the population (indices 0..n-1).
    n_draws : int or None, default=None
        Number of draws; None means ``n_samples``.
    rng : np.random.Generator or None, default=None
        Random generator; ``np.random.default_rng()`` if None.

    Returns
    -------
    np.ndarray of shape (n_draws,)
        Int indices in [0, n_samples), possibly repeated.

    Raises
    ------
    ValueError
        If ``n_samples < 1`` or ``n_draws < 1``.

    Notes
    -----
    Tested by properties: range and shape, same indices with the same generator, mean
    fraction of distinct indices close to 1 - (1 - 1/n)^n.

    Examples
    --------
    >>> idx = bootstrap_indices(5, rng=np.random.default_rng(0))
    >>> idx.shape
    (5,)
    >>> bool(idx.min() >= 0 and idx.max() < 5)
    True
    """
    if not (_is_int(n_samples) and n_samples >= 1):
        raise ValueError(f"n_samples must be an int >= 1, got {n_samples!r}")
    if n_draws is None:
        n_draws = n_samples
    if not (_is_int(n_draws) and n_draws >= 1):
        raise ValueError(f"n_draws must be None or an int >= 1, got {n_draws!r}")
    rng = np.random.default_rng(rng)
    return rng.integers(0, n_samples, size=n_draws)


class BaggingClassifier:
    """Bootstrap aggregating (bagging) of classifiers.

    ``fit`` creates the generator ``np.random.default_rng(random_state)``; each member is
    ``clone(template)`` (``template = estimator`` or ``DecisionTreeClassifier()``),
    trained on its own sample of the training set: ``bootstrap_indices`` (with
    replacement) or distinct indices (``bootstrap=False``). If the member has a
    ``random_state`` attribute, it is set to an int drawn from the ensemble generator,
    so the whole ensemble is reproducible. Prediction is a soft vote, as scikit-learn:
    the members' ``predict_proba`` (aligned on ``classes_``, a member may have seen only
    some classes) are averaged and ``predict`` takes the argmax. The book's hard vote is
    ``plurality_vote``.

    Parameters
    ----------
    estimator : object or None, default=None
        Unfitted classifier with ``fit``, ``predict``, ``predict_proba`` and
        ``classes_``; None means ``DecisionTreeClassifier()``. Never fitted itself.
    n_estimators : int, default=10
        Number of members (>= 1).
    max_samples : int or float, default=1.0
        Size of each member's sample: an int in 1..n, or a float f in (0, 1] giving
        ``int(f * n)`` (at least 1).
    bootstrap : bool, default=True
        Draw with replacement; if False, distinct indices.
    oob_score : bool, default=False
        Compute ``oob_score_``, the out-of-bag accuracy (requires ``bootstrap=True``).
    random_state : int or None, default=None
        Seed of the ensemble generator.

    Attributes
    ----------
    estimators_ : list
        The fitted members.
    estimators_samples_ : list of np.ndarray
        Training indices of every member.
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    oob_score_ : float
        Only if ``oob_score=True``: accuracy of the out-of-bag predictions. Each
        training sample is predicted by averaging the ``predict_proba`` of the members
        that did not see it; samples seen by every member are left out.

    Notes
    -----
    Statistical comparison with ``sklearn.ensemble.BaggingClassifier(
    DecisionTreeClassifier())`` (the random streams differ): test accuracy within 0.03,
    ``oob_score_`` within 0.05. Properties: number of members, reproducibility with
    ``random_state``, distinct bootstrap samples.

    Examples
    --------
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> bag = BaggingClassifier(n_estimators=5, random_state=0).fit(X, y)
    >>> len(bag.estimators_)
    5
    >>> bag.predict(np.array([[1.5], [11.5]]))
    array([0, 1])
    """

    def __init__(
        self,
        estimator: object | None = None,
        n_estimators: int = 10,
        max_samples: int | float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit every member on its own sample of the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        Self
            The fitted ensemble (``self``).

        Raises
        ------
        ValueError
            If ``n_estimators < 1``, ``max_samples`` is invalid, ``oob_score=True``
            with ``bootstrap=False``, or ``len(X) != len(y)``.
        """
        _check_n_estimators(self.n_estimators)
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True: without replacement, no sample is out-of-bag "
                             "for sure")
        X = _check_X(X)
        y = _check_y(y, len(X))
        n = len(X)
        if _is_int(self.max_samples):
            if not 1 <= self.max_samples <= n:
                raise ValueError(f"max_samples must be an int in 1..{n} or a float in (0, 1], "
                                 f"got {self.max_samples!r}")
            n_draws = int(self.max_samples)
        elif isinstance(self.max_samples, (float, np.floating)) and 0.0 < self.max_samples <= 1.0:
            n_draws = max(1, int(self.max_samples * n))
        else:
            raise ValueError(f"max_samples must be an int in 1..{n} or a float in (0, 1], got {self.max_samples!r}")
        rng = np.random.default_rng(self.random_state)
        template = DecisionTreeClassifier() if self.estimator is None else self.estimator
        self.classes_ = np.unique(y)
        members, samples = [], []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = bootstrap_indices(n, n_draws, rng)
            else:
                indices = rng.choice(n, size=n_draws, replace=False)
            member = clone(template)
            if hasattr(member, "random_state"):
                member.random_state = int(rng.integers(np.iinfo(np.int32).max))
            member.fit(X[indices], y[indices])
            members.append(member)
            samples.append(indices)
        self.estimators_ = members
        self.estimators_samples_ = samples
        if self.oob_score:
            total = np.zeros((n, len(self.classes_)))
            counts = np.zeros(n)
            for member, indices in zip(members, samples):
                out = np.ones(n, dtype=bool)
                out[indices] = False
                if out.any():
                    total[out] += _aligned_proba(member, X[out], self.classes_)
                    counts[out] += 1
            kept = counts > 0                                     # samples seen by every member are left out
            if not kept.any():
                raise ValueError("no sample is out-of-bag: increase n_estimators or decrease max_samples")
            predicted = self.classes_[np.argmax(total[kept] / counts[kept, None], axis=1)]
            self.oob_score_ = float(np.mean(predicted == y[kept]))
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the average of the members' class probabilities (soft vote).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Rows summing to 1; column k is ``classes_[k]``.

        Raises
        ------
        RuntimeError
            If the ensemble is not fitted.
        """
        return _soft_vote(self, X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the largest average probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_``.
        """
        proba = self.predict_proba(X)                  # first: raises RuntimeError before fit
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction of correct predictions, in [0, 1].
        """
        return _accuracy(self, X, y)


class RandomForestClassifier:
    """Random forest: bagging of decision trees with feature bagging.

    Every member is a ``DecisionTreeClassifier`` with the forest's ``max_depth``,
    ``min_samples_split``, ``min_samples_leaf`` and ``max_features``, trained on a
    bootstrap sample: at every node, a new random subset of ``max_features`` features
    is examined, which decorrelates the trees. Soft vote as ``BaggingClassifier``.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees (>= 1).
    max_depth : int or None, default=None
        Maximum depth of every tree.
    min_samples_split : int, default=2
        Minimum number of samples to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples in each child.
    max_features : int, float, str or None, default="sqrt"
        Features examined at each node (see ``DecisionTreeClassifier``); "sqrt" as
        scikit-learn.
    bootstrap : bool, default=True
        Train every tree on a bootstrap sample (else on the whole training set).
    oob_score : bool, default=False
        Compute ``oob_score_`` (requires ``bootstrap=True``).
    random_state : int or None, default=None
        Seed of the forest generator (bootstrap samples and trees' seeds).

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The fitted trees.
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    feature_importances_ : np.ndarray of shape (n_features,)
        Mean of the trees' ``feature_importances_``.
    oob_score_ : float
        Only if ``oob_score=True``: out-of-bag accuracy (see ``BaggingClassifier``).

    Notes
    -----
    Statistical comparison with ``sklearn.ensemble.RandomForestClassifier``: accuracy
    within 0.03 on moons and Penguins, same ranking of the top feature importances.

    Examples
    --------
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    >>> forest.predict(np.array([[1.5], [11.5]]))
    array([0, 1])
    >>> forest.feature_importances_
    array([1.])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the trees of the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        Self
            The fitted forest (``self``).

        Raises
        ------
        ValueError
            If a hyperparameter is invalid or ``len(X) != len(y)``.
        """
        _check_n_estimators(self.n_estimators)
        template = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        bagging = BaggingClassifier(template, n_estimators=self.n_estimators, max_samples=1.0,
                                    bootstrap=self.bootstrap, oob_score=self.oob_score,
                                    random_state=self.random_state).fit(X, y)
        self._bagging = bagging
        self.estimators_ = bagging.estimators_
        self.classes_ = bagging.classes_
        self.feature_importances_ = _mean_importances(self.estimators_, X)
        if self.oob_score:
            self.oob_score_ = bagging.oob_score_
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the average of the trees' class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Rows summing to 1.

        Raises
        ------
        RuntimeError
            If the forest is not fitted.
        """
        return _soft_vote(self, X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the largest average probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_``.
        """
        proba = self.predict_proba(X)                  # first: raises RuntimeError before fit
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction of correct predictions, in [0, 1].
        """
        return _accuracy(self, X, y)


class ExtraTreesClassifier:
    """Extremely randomized trees (ExtraTrees).

    Like a random forest, but every tree uses ``splitter="random"``: for each candidate
    feature a single threshold is drawn at random instead of searching the best one.
    By default (``bootstrap=False``) every tree sees the whole training set, as
    scikit-learn: the randomness comes from the features and thresholds only.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees (>= 1).
    max_depth : int or None, default=None
        Maximum depth of every tree.
    min_samples_split : int, default=2
        Minimum number of samples to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples in each child.
    max_features : int, float, str or None, default="sqrt"
        Features examined at each node.
    bootstrap : bool, default=False
        Train every tree on a bootstrap sample instead of the whole training set.
    random_state : int or None, default=None
        Seed of the ensemble generator.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The fitted trees.
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    feature_importances_ : np.ndarray of shape (n_features,)
        Mean of the trees' ``feature_importances_``.

    Notes
    -----
    Statistical comparison with ``sklearn.ensemble.ExtraTreesClassifier``: accuracy
    within 0.03.

    Examples
    --------
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> extra = ExtraTreesClassifier(n_estimators=5, random_state=0).fit(X, y)
    >>> extra.predict(np.array([[1.5], [11.5]]))
    array([0, 1])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the randomized trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        Self
            The fitted ensemble (``self``).

        Raises
        ------
        ValueError
            If a hyperparameter is invalid or ``len(X) != len(y)``.
        """
        _check_n_estimators(self.n_estimators)
        template = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                          splitter="random")
        bagging = BaggingClassifier(template, n_estimators=self.n_estimators, max_samples=1.0,
                                    bootstrap=self.bootstrap, random_state=self.random_state).fit(X, y)
        self._bagging = bagging
        self.estimators_ = bagging.estimators_
        self.classes_ = bagging.classes_
        self.feature_importances_ = _mean_importances(self.estimators_, X)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the average of the trees' class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Rows summing to 1.

        Raises
        ------
        RuntimeError
            If the ensemble is not fitted.
        """
        return _soft_vote(self, X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the largest average probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_``.
        """
        proba = self.predict_proba(X)                  # first: raises RuntimeError before fit
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction of correct predictions, in [0, 1].
        """
        return _accuracy(self, X, y)


class AdaBoostClassifier:
    """Multi-class AdaBoost (SAMME) with sample weights.

    The sample weights start uniform (1/n). At each round m: normalise the weights to
    sum 1, fit ``clone(template)`` with ``sample_weight``, compute its weighted error
    ``err`` (sum of the weights of the misclassified samples), then:

    - ``err == 0``: keep the member with weight 1.0 (error 0.0) and stop;
    - ``err >= 1 - 1/K``: discard the member (no better than chance) and stop;
    - otherwise ``alpha_m = learning_rate * (ln((1 - err) / err) + ln(K - 1))``, keep
      the member and multiply the weights of the misclassified samples by
      ``exp(alpha_m)``.

    ``decision_function`` sums the weights ``alpha_m`` of the members that vote for
    each class, divided by the sum of all the ``alpha_m``; ``predict`` takes the argmax.

    Parameters
    ----------
    estimator : object or None, default=None
        Unfitted template whose ``fit`` accepts ``sample_weight``; None means
        ``DecisionTreeClassifier(max_depth=1)`` (a decision stump).
    n_estimators : int, default=50
        Maximum number of rounds (>= 1).
    learning_rate : float, default=1.0
        Shrinkage of every ``alpha_m`` (> 0).
    random_state : int or None, default=None
        Seed of a generator whose ints are given to the members' ``random_state``
        (when they have one).

    Attributes
    ----------
    estimators_ : list
        The kept members, in order.
    estimator_weights_ : np.ndarray of shape (n_fitted,)
        ``alpha_m`` of every kept member.
    estimator_errors_ : np.ndarray of shape (n_fitted,)
        Weighted error ``err`` of every kept member.
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.

    Notes
    -----
    Tested against ``sklearn.ensemble.AdaBoostClassifier(estimator=
    DecisionTreeClassifier(max_depth=1))`` with the same ``n_estimators`` and
    ``learning_rate``, on continuous data without ties: ``estimator_weights_`` and
    ``estimator_errors_`` allclose (scikit-learn pads them with zeros up to
    ``n_estimators``), identical ``predict`` and ``staged_predict``. scikit-learn's
    ``decision_function`` is 1-D in the binary case; ours is always (n_samples, K).

    Examples
    --------
    >>> X = np.arange(1.0, 8.0).reshape(-1, 1)
    >>> y = np.array([0, 0, 0, 1, 1, 0, 1])
    >>> ada = AdaBoostClassifier(n_estimators=3).fit(X, y)
    >>> ada.estimator_errors_.round(4)
    array([0.1429, 0.1667, 0.2   ])
    >>> ada.estimator_weights_.round(4)
    array([1.7918, 1.6094, 1.3863])
    >>> [pred.tolist() for pred in ada.staged_predict(X)]
    [[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 1]]
    """

    def __init__(
        self,
        estimator: object | None = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Run the boosting rounds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels (at least 2 classes).

        Returns
        -------
        Self
            The fitted ensemble (``self``).

        Raises
        ------
        ValueError
            If ``n_estimators < 1``, ``learning_rate <= 0``, the template's ``fit`` has
            no ``sample_weight`` parameter (``inspect.signature``), y has fewer than 2
            classes, ``len(X) != len(y)``, or the very first member is no better than
            chance (empty ensemble).
        """
        _check_n_estimators(self.n_estimators)
        if not (isinstance(self.learning_rate, (int, float, np.integer, np.floating))
                and not isinstance(self.learning_rate, bool) and self.learning_rate > 0):
            raise ValueError(f"learning_rate must be a number > 0, got {self.learning_rate!r}")
        template = DecisionTreeClassifier(max_depth=1) if self.estimator is None else self.estimator
        if "sample_weight" not in inspect.signature(template.fit).parameters:
            raise ValueError(f"{type(template).__name__}.fit has no sample_weight parameter: AdaBoost needs "
                             "weighted members")
        X = _check_X(X)
        y = _check_y(y, len(X))
        classes = np.unique(y)
        K = len(classes)
        if K < 2:
            raise ValueError(f"y must have at least 2 classes, got {K}")
        rng = np.random.default_rng(self.random_state)
        w = np.full(len(X), 1.0 / len(X))
        members, alphas, errors = [], [], []
        for _ in range(self.n_estimators):
            w = w / w.sum()
            member = clone(template)
            if hasattr(member, "random_state"):
                member.random_state = int(rng.integers(np.iinfo(np.int32).max))
            member.fit(X, y, sample_weight=w)
            wrong = member.predict(X) != y
            err = float(np.sum(w[wrong]))
            if err <= 0.0:
                members.append(member)
                alphas.append(1.0)
                errors.append(0.0)
                break
            if err >= 1.0 - 1.0 / K:
                if not members:
                    raise ValueError(f"the first member is no better than chance (weighted error {err:.4g} >= "
                                     f"1 - 1/K = {1.0 - 1.0 / K:.4g}): the ensemble is empty")
                break
            alpha = self.learning_rate * (np.log((1.0 - err) / err) + np.log(K - 1.0))
            members.append(member)
            alphas.append(float(alpha))
            errors.append(err)
            w = np.where(wrong, w * np.exp(alpha), w)
        self.classes_ = classes
        self.estimators_ = members
        self.estimator_weights_ = np.array(alphas)
        self.estimator_errors_ = np.array(errors)
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return the normalised weighted votes of every class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            ``sum_m alpha_m * [member m predicts class k] / sum_m alpha_m``.

        Raises
        ------
        RuntimeError
            If the ensemble is not fitted.
        """
        _check_fitted(self)
        X = _check_X(X, self)
        votes = np.zeros((X.shape[0], len(self.classes_)))
        rows = np.arange(X.shape[0])
        for member, alpha in zip(self.estimators_, self.estimator_weights_):
            votes[rows, np.searchsorted(self.classes_, member.predict(X))] += alpha
        return votes / self.estimator_weights_.sum()

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the largest weighted vote.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_``.
        """
        votes = self.decision_function(X)              # first: raises RuntimeError before fit
        return self.classes_[np.argmax(votes, axis=1)]

    def staged_predict(self, X: ArrayLike) -> Iterator[np.ndarray]:
        """Yield the predictions of the ensemble after each round.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Yields
        ------
        np.ndarray of shape (n_samples,)
            Predictions using the first m members, for m = 1..len(estimators_).
        """
        _check_fitted(self)
        X = _check_X(X, self)
        votes = np.zeros((X.shape[0], len(self.classes_)))
        rows = np.arange(X.shape[0])
        for member, alpha in zip(self.estimators_, self.estimator_weights_):
            votes[rows, np.searchsorted(self.classes_, member.predict(X))] += alpha
            yield self.classes_[np.argmax(votes, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction of correct predictions, in [0, 1].
        """
        return _accuracy(self, X, y)


class GradientBoostingRegressor:
    """Gradient boosting for regression with the squared error.

    ``init_ = mean(y)`` and ``F(x) = init_`` at first. At each stage, a
    ``DecisionTreeRegressor(max_depth, min_samples_leaf)`` (a ``clone`` of a template)
    is fitted to the residuals ``y - F(X)`` (the negative gradient of the squared error),
    on a random subsample of ``max(1, int(subsample * n))`` samples drawn without
    replacement if ``subsample < 1``; then ``F += learning_rate * tree.predict``.
    ``train_score_[m]`` is the MSE on the stage's (in-bag) samples after stage m.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of stages (>= 1).
    learning_rate : float, default=0.1
        Shrinkage of every tree's contribution (> 0; usually in (0, 1]).
    max_depth : int, default=3
        Depth of every regression tree.
    min_samples_leaf : int, default=1
        Minimum number of samples in each leaf of the trees.
    subsample : float, default=1.0
        Fraction of the samples used at each stage, in (0, 1] (stochastic gradient
        boosting when < 1).
    random_state : int or None, default=None
        Seed of the generator used for the subsamples.

    Attributes
    ----------
    init_ : float
        Initial constant prediction (mean of y).
    estimators_ : list of DecisionTreeRegressor
        The fitted trees, one per stage.
    train_score_ : np.ndarray of shape (n_estimators,)
        In-bag MSE after each stage (the full training MSE when ``subsample=1``).

    Notes
    -----
    Tested against ``sklearn.ensemble.GradientBoostingRegressor(
    criterion="squared_error")`` with the same hyperparameters and ``subsample=1.0``, on
    continuous data without ties: ``predict``, ``staged_predict`` and ``train_score_``
    allclose.

    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y = np.array([1.0, 1.2, 3.0, 3.4])
    >>> gb = GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, max_depth=1).fit(X, y)
    >>> round(gb.init_, 4)
    2.15
    >>> [pred.round(4).tolist() for pred in gb.staged_predict(np.array([[0.0], [5.0]]))]
    [[1.625, 2.675], [1.3625, 2.9375]]
    >>> gb.train_score_.round(4)
    array([0.3006, 0.0939])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the stages one after the other on the residuals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Real targets.

        Returns
        -------
        Self
            The fitted model (``self``).

        Raises
        ------
        ValueError
            If ``learning_rate <= 0``, ``n_estimators < 1``, ``subsample`` is not in
            (0, 1], or ``len(X) != len(y)``.
        """
        _check_n_estimators(self.n_estimators)
        if not (isinstance(self.learning_rate, (int, float, np.integer, np.floating))
                and not isinstance(self.learning_rate, bool) and self.learning_rate > 0):
            raise ValueError(f"learning_rate must be a number > 0, got {self.learning_rate!r}")
        if not (isinstance(self.subsample, (int, float, np.integer, np.floating))
                and not isinstance(self.subsample, bool) and 0.0 < self.subsample <= 1.0):
            raise ValueError(f"subsample must be in (0, 1], got {self.subsample!r}")
        X = _check_X(X)
        y = _check_y(y, len(X), regression=True)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        template = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.init_ = float(np.mean(y))
        F = np.full(n, self.init_)
        n_inbag = max(1, int(self.subsample * n))
        trees, scores = [], []
        for _ in range(self.n_estimators):
            rows = np.arange(n) if self.subsample >= 1.0 else rng.choice(n, size=n_inbag, replace=False)
            residuals = y - F                                     # the negative gradient of (y - F)^2 / 2
            tree = clone(template)
            tree.fit(X[rows], residuals[rows])
            F = F + self.learning_rate * tree.predict(X)
            trees.append(tree)
            scores.append(float(np.mean((y[rows] - F[rows]) ** 2)))
        self.estimators_ = trees
        self.train_score_ = np.array(scores)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict ``init_ + learning_rate * sum of the trees' predictions``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Float predictions.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        prediction = None
        for prediction in self.staged_predict(X):
            pass
        return prediction

    def staged_predict(self, X: ArrayLike) -> Iterator[np.ndarray]:
        """Yield the predictions after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Yields
        ------
        np.ndarray of shape (n_samples,)
            Predictions using the first m trees, for m = 1..n_estimators.
        """
        _check_fitted(self)
        X = _check_X(X, self)
        F = np.full(X.shape[0], self.init_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            True targets.

        Returns
        -------
        float
            Coefficient of determination R² (as ``linear.r2_score``, chapter 9).
        """
        predictions = self.predict(X)
        y = _check_y(y, len(predictions), regression=True)
        ss_res = float(np.sum((y - predictions) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:                              # constant target (as scikit-learn's r2_score)
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
