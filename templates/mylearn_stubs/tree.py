"""Decision trees (CART) — mylearn, chapter 13 (Classifiers).

Impurity measures (Gini, entropy, variance), the gain of a split, the search for the
best split and the greedy recursive trees for classification and regression. The
trees are built for chapter 14: ``sample_weight`` (AdaBoost), ``max_features`` and
``random_state`` (random forest), ``splitter="random"`` (ExtraTrees) and
``DecisionTreeRegressor`` (gradient boosting) reuse them without any change.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike


def gini_impurity(y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Compute the Gini impurity ``1 - sum_k p_k^2`` of a set of labels.

    ``p_k`` is the (weighted) proportion of class k: the sum of the weights of the
    samples of class k divided by the total weight. 0 for a pure node, maximal
    (1 - 1/K) when the K classes are equally represented.

    Parameters
    ----------
    y : array-like of shape (n,)
        Class labels (any values).
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.

    Returns
    -------
    float
        Impurity in [0, 1 - 1/K].

    Raises
    ------
    ValueError
        If ``y`` is empty, or the weights are negative, sum to 0 or do not have the
        shape of ``y``.

    Notes
    -----
    Tested on hand-computed values and against the root impurity
    ``tree_.impurity[0]`` of ``sklearn.tree.DecisionTreeClassifier(max_depth=1)``.

    Examples
    --------
    >>> gini_impurity([0, 0, 1, 1])
    0.5
    >>> gini_impurity(["a", "a", "a", "b"])
    0.375
    >>> gini_impurity([0, 1], sample_weight=[3.0, 1.0])
    0.375
    """
    # TODO: weighted class proportions (np.unique with return_inverse and
    # np.bincount with weights help), then the formula.
    raise NotImplementedError("gini_impurity() is not implemented yet")


def entropy_impurity(y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Compute the Shannon entropy in bits ``-sum_k p_k log2 p_k`` of a set of labels.

    ``p_k`` is the (weighted) proportion of class k; by convention 0 log 0 = 0 (only the
    classes present are summed). 0 for a pure node, 1 bit for two balanced classes.

    Parameters
    ----------
    y : array-like of shape (n,)
        Class labels (any values).
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.

    Returns
    -------
    float
        Entropy >= 0, in bits.

    Raises
    ------
    ValueError
        If ``y`` is empty or the weights are invalid (see ``gini_impurity``).

    Notes
    -----
    Tested against ``scipy.stats.entropy(p, base=2)`` and the root impurity of
    ``sklearn.tree.DecisionTreeClassifier(criterion="entropy")`` (also in base 2).

    Examples
    --------
    >>> entropy_impurity([0, 0, 1, 1])
    1.0
    >>> round(entropy_impurity([0, 0, 0, 1]), 4)
    0.8113
    """
    # TODO: weighted class proportions, then the formula with np.log2.
    raise NotImplementedError("entropy_impurity() is not implemented yet")


def variance_impurity(y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Compute the (weighted) mean squared deviation from the (weighted) mean.

    Impurity of a regression node: ``sum_i w_i (y_i - ybar)^2 / sum_i w_i`` with
    ``ybar = sum_i w_i y_i / sum_i w_i`` (population variance when all weights are 1).

    Parameters
    ----------
    y : array-like of shape (n,)
        Real targets.
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.

    Returns
    -------
    float
        Variance >= 0.

    Raises
    ------
    ValueError
        If ``y`` is empty or the weights are invalid (see ``gini_impurity``).

    Notes
    -----
    Tested against ``np.average`` and the root impurity ``tree_.impurity[0]`` of
    ``sklearn.tree.DecisionTreeRegressor(max_depth=1)``.

    Examples
    --------
    >>> variance_impurity([1.0, 2.0, 3.0, 4.0])
    1.25
    """
    # TODO: weighted mean, then weighted mean of the squared deviations.
    raise NotImplementedError("variance_impurity() is not implemented yet")


def split_gain(
    y: ArrayLike,
    left_mask: ArrayLike,
    criterion: str = "gini",
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute the impurity decrease of a split.

    ``gain = I(parent) - (w_L / w) I(left) - (w_R / w) I(right)`` where I is the
    impurity of the criterion and w_L, w_R, w the total weights of the left child, the
    right child and the parent (the numbers of samples when all weights are 1).

    Parameters
    ----------
    y : array-like of shape (n,)
        Labels (classification) or targets (regression) of the parent node.
    left_mask : array-like of shape (n,)
        Booleans: True sends the sample to the left child, False to the right child.
    criterion : str, default="gini"
        "gini", "entropy" or "squared_error" (variance).
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.

    Returns
    -------
    float
        The gain (>= 0 up to rounding); 0.0 if one child is empty.

    Raises
    ------
    ValueError
        If ``criterion`` is unknown or the shapes of ``y``, ``left_mask`` and
        ``sample_weight`` differ.

    Notes
    -----
    Tested on hand-computed values and for consistency with the impurities of the two
    children in ``tree_.impurity`` of scikit-learn trees.

    Examples
    --------
    >>> split_gain([0, 0, 1, 1], [True, True, False, False])
    0.5
    >>> split_gain([0, 0, 1, 1], [True, True, False, False], criterion="entropy")
    1.0
    """
    # TODO: choose the impurity function, then the formula above.
    raise NotImplementedError("split_gain() is not implemented yet")


def best_split(
    X: ArrayLike,
    y: ArrayLike,
    criterion: str = "gini",
    sample_weight: ArrayLike | None = None,
    feature_indices: ArrayLike | None = None,
    min_samples_leaf: int = 1,
    splitter: str = "best",
    rng: np.random.Generator | None = None,
) -> tuple[int, float, float] | None:
    """Find the best (feature, threshold) split of a node.

    Samples with ``X[:, feature] <= threshold`` go left, the others right. Candidate
    thresholds, for every candidate feature:

    - ``splitter="best"``: the midpoints between consecutive distinct sorted values of
      the feature (e.g. 3.5 between 3 and 4), as scikit-learn;
    - ``splitter="random"`` (ExtraTrees): a single threshold drawn uniformly in
      (min, max) of the feature with ``rng.uniform`` (a constant feature is skipped).

    A candidate is valid only if each child gets at least ``min_samples_leaf`` samples.
    The valid candidate with the largest gain (``split_gain``) wins; ties go to the
    lowest feature index, then to the lowest threshold.

    Parameters
    ----------
    X : array-like of shape (n, n_features)
        Samples of the node (no NaN).
    y : array-like of shape (n,)
        Labels or targets of the node.
    criterion : str, default="gini"
        "gini", "entropy" or "squared_error".
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.
    feature_indices : array-like of int or None, default=None
        Features to consider (the random subset of a forest); None means all.
    min_samples_leaf : int, default=1
        Minimum number of samples in each child.
    splitter : str, default="best"
        "best" or "random".
    rng : np.random.Generator or None, default=None
        Generator used by ``splitter="random"``; ``np.random.default_rng()`` if None.

    Returns
    -------
    tuple of (int, float, float) or None
        ``(feature, threshold, gain)`` as Python int and floats, or None when no valid
        split has a strictly positive gain (the node becomes a leaf; scikit-learn may
        still split with a zero gain).

    Raises
    ------
    ValueError
        If ``criterion`` or ``splitter`` is unknown, or the shapes are inconsistent.

    Notes
    -----
    Tested against a brute-force search and against the root feature and threshold of
    ``sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor(max_depth=1)`` on
    continuous data without ties.

    Examples
    --------
    >>> X = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> best_split(X, y)
    (0, 2.5, 0.5)
    """
    # TODO: loop over the candidate features; for each one, the candidate thresholds,
    # the left mask, the validity check and the gain; keep the best candidate.
    raise NotImplementedError("best_split() is not implemented yet")


@dataclass(eq=False)
class Node:
    """One node of a decision tree (a leaf when ``left`` is None).

    Provided: already implemented, no exercise. A dataclass: ``Node(...)`` stores the
    fields below, nothing more. An internal node sends a sample to ``left`` when
    ``x[feature] <= threshold``, else to ``right``.

    Parameters
    ----------
    feature : int or None
        Index of the split feature (None for a leaf).
    threshold : float or None
        Split threshold (None for a leaf).
    left : Node or None
        Left child (None for a leaf).
    right : Node or None
        Right child (None for a leaf).
    value : np.ndarray
        Classifier: shape (n_classes,), weighted class proportions of the node, aligned
        on the ``classes_`` of the tree. Regressor: shape (1,), weighted mean target.
    n_samples : int
        Number of training samples that reached the node.
    impurity : float
        Impurity of the node for the criterion of the tree.

    Examples
    --------
    >>> leaf = Node(feature=None, threshold=None, left=None, right=None,
    ...             value=np.array([1.0, 0.0]), n_samples=2, impurity=0.0)
    >>> leaf.left is None
    True
    """

    feature: int | None
    threshold: float | None
    left: Node | None
    right: Node | None
    value: np.ndarray
    n_samples: int
    impurity: float


class DecisionTreeClassifier:
    """Decision tree classifier grown greedily and recursively (CART).

    ``fit`` builds ``root_``: a node becomes a leaf when it is pure, when
    ``max_depth`` is reached, when it has fewer than ``min_samples_split`` samples, or
    when ``best_split`` finds no valid split with a positive gain. Otherwise it is split
    by ``best_split`` and both children are built the same way. A sample is predicted
    by the ``value`` of the leaf it reaches.

    At every node, the candidate features are ``max_features`` features drawn without
    replacement (``rng.choice``) with the generator ``np.random.default_rng(random_state)``
    created in ``fit``; with ``max_features=None`` all features are candidates.

    Parameters
    ----------
    criterion : str, default="gini"
        "gini" or "entropy".
    max_depth : int or None, default=None
        Maximum depth (>= 1); None grows the tree until the leaves are pure.
    min_samples_split : int, default=2
        A node with fewer samples is not split (>= 2).
    min_samples_leaf : int, default=1
        Minimum number of samples in each child (>= 1).
    max_features : int, float, str or None, default=None
        Features examined at each node: None (all d features), an int k in 1..d, a
        float f in (0, 1] (``max(1, int(f * d))``), "sqrt" (``max(1, int(sqrt(d)))``)
        or "log2" (``max(1, int(log2(d)))``).
    splitter : str, default="best"
        "best" or "random" (see ``best_split``).
    random_state : int or None, default=None
        Seed of the generator used for ``max_features`` and ``splitter="random"``.

    Attributes
    ----------
    root_ : Node
        Root of the fitted tree.
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_importances_ : np.ndarray of shape (n_features,)
        For each feature, the sum over the nodes split on it of
        ``(w_node / w_root) * gain``, normalised to sum to 1 (all zeros if the root is
        a leaf).

    Notes
    -----
    Tested against ``sklearn.tree.DecisionTreeClassifier`` with the same
    hyperparameters (``max_features=None``) on continuous data without ties: identical
    ``predict``, ``predict_proba``, ``get_depth`` and ``get_n_leaves``, allclose
    ``feature_importances_``; ``sample_weight=2`` equals duplicating the samples.
    ``max_features`` and ``splitter`` are tested by properties (same tree with the same
    ``random_state``, different trees with different seeds).

    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> tree = DecisionTreeClassifier().fit(X, y)
    >>> tree.root_.feature, tree.root_.threshold
    (0, 2.5)
    >>> tree.get_depth(), tree.get_n_leaves()
    (1, 2)
    >>> tree.predict(np.array([[1.5], [3.5]]))
    array([0, 1])
    >>> tree.predict_proba(np.array([[1.5], [3.5]]))
    array([[1., 0.],
           [0., 1.]])
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        splitter: str = "best",
        random_state: int | None = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self:
        """Grow the tree on (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples (no NaN).
        y : array-like of shape (n_samples,)
            Labels (any sortable values).
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Non-negative weights (AdaBoost); None means all 1.

        Returns
        -------
        Self
            The fitted tree (``self``), with ``root_``, ``classes_``, ``n_classes_``,
            ``n_features_in_`` and ``feature_importances_``.

        Raises
        ------
        ValueError
            If a hyperparameter is invalid, X is not 2-D or contains NaN,
            ``len(X) != len(y)``, or ``sample_weight`` is negative or misshaped.
        """
        # TODO: validate, encode the labels, create the generator, then build the nodes
        # recursively (a private helper method is welcome) and the importances.
        raise NotImplementedError("fit() is not implemented yet")

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the class proportions of the leaf reached by every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Weighted class proportions of the leaves; column k is ``classes_[k]``.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        ValueError
            If X has a number of features different from ``n_features_in_``.
        """
        # TODO: walk down from root_ for every sample.
        raise NotImplementedError("predict_proba() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the most probable class of the leaf reached by every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_`` (ties: the first class).

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        # TODO: argmax of predict_proba.
        raise NotImplementedError("predict() is not implemented yet")

    def get_depth(self) -> int:
        """Return the depth of the tree (0 when the root is a leaf).

        Returns
        -------
        int
            Largest number of edges between the root and a leaf.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        # TODO: recursion on the children.
        raise NotImplementedError("get_depth() is not implemented yet")

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the tree.

        Returns
        -------
        int
            Number of nodes without children.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        # TODO: recursion on the children.
        raise NotImplementedError("get_n_leaves() is not implemented yet")

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


class DecisionTreeRegressor:
    """Decision tree regressor: the same greedy tree with the variance criterion.

    Grown exactly like ``DecisionTreeClassifier`` (same stopping rules, same
    ``max_features`` and ``splitter`` options), with ``variance_impurity`` as impurity;
    every leaf predicts the (weighted) mean target of its training samples. The
    gradient boosting of chapter 14 is made of such trees.

    Parameters
    ----------
    criterion : str, default="squared_error"
        Only "squared_error" (variance) is supported.
    max_depth : int or None, default=None
        Maximum depth (>= 1); None grows the tree until the leaves are pure.
    min_samples_split : int, default=2
        A node with fewer samples is not split (>= 2).
    min_samples_leaf : int, default=1
        Minimum number of samples in each child (>= 1).
    max_features : int, float, str or None, default=None
        Features examined at each node, as in ``DecisionTreeClassifier``.
    splitter : str, default="best"
        "best" or "random".
    random_state : int or None, default=None
        Seed of the generator used for ``max_features`` and ``splitter="random"``.

    Attributes
    ----------
    root_ : Node
        Root of the fitted tree (``value`` of shape (1,) in every node).
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_importances_ : np.ndarray of shape (n_features,)
        Normalised weighted variance decrease per feature (see
        ``DecisionTreeClassifier``).

    Notes
    -----
    Tested against ``sklearn.tree.DecisionTreeRegressor`` with the same hyperparameters
    on continuous data without ties: allclose predictions, same depth and number of
    leaves.

    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y = np.array([1.0, 1.2, 3.0, 3.4])
    >>> reg = DecisionTreeRegressor(max_depth=1).fit(X, y)
    >>> reg.predict(np.array([[0.0], [5.0]])).round(4)
    array([1.1, 3.2])
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        splitter: str = "best",
        random_state: int | None = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self:
        """Grow the regression tree on (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples (no NaN).
        y : array-like of shape (n_samples,)
            Real targets.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Non-negative weights; None means all 1.

        Returns
        -------
        Self
            The fitted tree (``self``), with ``root_``, ``n_features_in_`` and
            ``feature_importances_``.

        Raises
        ------
        ValueError
            If a hyperparameter is invalid (including a criterion other than
            "squared_error"), X is not 2-D or contains NaN, ``len(X) != len(y)``, or
            ``sample_weight`` is negative or misshaped.
        """
        # TODO: same construction as the classifier, with the variance criterion and
        # the weighted mean as leaf value.
        raise NotImplementedError("fit() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the mean target of the leaf reached by every sample.

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
            If the tree is not fitted.
        ValueError
            If X has a number of features different from ``n_features_in_``.
        """
        # TODO: walk down from root_ for every sample.
        raise NotImplementedError("predict() is not implemented yet")

    def get_depth(self) -> int:
        """Return the depth of the tree (0 when the root is a leaf).

        Returns
        -------
        int
            Largest number of edges between the root and a leaf.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        # TODO: recursion on the children.
        raise NotImplementedError("get_depth() is not implemented yet")

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the tree.

        Returns
        -------
        int
            Number of nodes without children.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        # TODO: recursion on the children.
        raise NotImplementedError("get_n_leaves() is not implemented yet")

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
