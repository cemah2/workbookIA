"""Voting, bagging, forests and boosting — mylearn, chapter 14 (Ensembles).

Many weak or unstable models combined into a strong one: weighted plurality vote,
bootstrap samples, bagging, random forests, extremely randomized trees, AdaBoost (SAMME)
and gradient boosting for regression. The members are the trees of chapter 13
(``tree.py``); each member is a fresh ``clone`` (chapter 8) of an unfitted template.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .model_selection import clone  # chapter 8
from .tree import DecisionTreeClassifier, DecisionTreeRegressor  # chapter 13


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
    # TODO: sorted distinct labels, then a (n_samples, n_labels) table of weighted
    # votes and its argmax.
    raise NotImplementedError("plurality_vote() is not implemented yet")


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
    # TODO: one call to the generator.
    raise NotImplementedError("bootstrap_indices() is not implemented yet")


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
        # TODO: generator, template, then for each member: indices, clone, seed, fit;
        # finally the out-of-bag score if requested.
        raise NotImplementedError("fit() is not implemented yet")

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
        # TODO: align each member's columns (member.classes_) on classes_, then average.
        raise NotImplementedError("predict_proba() is not implemented yet")

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
        # TODO: argmax of predict_proba.
        raise NotImplementedError("predict() is not implemented yet")

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
        # TODO: compare the predictions with y and average.
        raise NotImplementedError("score() is not implemented yet")


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
        # TODO: a BaggingClassifier of DecisionTreeClassifier(max_features=...) does
        # the job; keep it in a private attribute (name starting with "_").
        raise NotImplementedError("fit() is not implemented yet")

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
        # TODO: soft vote over estimators_.
        raise NotImplementedError("predict_proba() is not implemented yet")

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
        # TODO: argmax of predict_proba.
        raise NotImplementedError("predict() is not implemented yet")

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
        # TODO: compare the predictions with y and average.
        raise NotImplementedError("score() is not implemented yet")


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
        # TODO: as RandomForestClassifier, with splitter="random" trees.
        raise NotImplementedError("fit() is not implemented yet")

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
        # TODO: soft vote over estimators_.
        raise NotImplementedError("predict_proba() is not implemented yet")

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
        # TODO: argmax of predict_proba.
        raise NotImplementedError("predict() is not implemented yet")

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
        # TODO: compare the predictions with y and average.
        raise NotImplementedError("score() is not implemented yet")


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
        # TODO: the loop described in the class docstring.
        raise NotImplementedError("fit() is not implemented yet")

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
        # TODO: one weighted vote table over the members.
        raise NotImplementedError("decision_function() is not implemented yet")

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
        # TODO: argmax of decision_function.
        raise NotImplementedError("predict() is not implemented yet")

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
        # TODO: accumulate the weighted votes member after member and yield (a
        # generator function uses the keyword yield).
        raise NotImplementedError("staged_predict() is not implemented yet")

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
        # TODO: compare the predictions with y and average.
        raise NotImplementedError("score() is not implemented yet")


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
        # TODO: init_, then for each stage: residuals, (subsample), clone + fit a tree,
        # update F, record the in-bag MSE.
        raise NotImplementedError("fit() is not implemented yet")

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
        # TODO: start from init_ and add every tree's contribution.
        raise NotImplementedError("predict() is not implemented yet")

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
        # TODO: same as predict, yielding after every tree.
        raise NotImplementedError("staged_predict() is not implemented yet")

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
        # TODO: 1 - SS_res / SS_tot (you may import r2_score from .linear here).
        raise NotImplementedError("score() is not implemented yet")
