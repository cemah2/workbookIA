"""Distances, nearest centroid and k-means — mylearn, chapter 7 (Classification).

This module holds your first estimators "à la scikit-learn": ``fit`` learns from the
data and stores the learnt attributes (their names end with ``_``), ``predict`` uses them.
You code vectorised squared distances, the nearest centroid classifier, the k-means++
seeding, Lloyd's k-means algorithm and the silhouette coefficient used to choose k.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike


def pairwise_sq_distances(A: ArrayLike, B: ArrayLike) -> np.ndarray:
    """Compute the squared Euclidean distances between every row of A and every row of B.

    Vectorised with broadcasting or with the identity
    ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2 (no Python loop over the samples).
    That identity can give tiny negative values (rounding errors): they are clipped to 0.

    Parameters
    ----------
    A : array-like of shape (n_a, n_features)
        First set of points, one per row.
    B : array-like of shape (n_b, n_features)
        Second set of points, one per row.

    Returns
    -------
    np.ndarray of shape (n_a, n_b)
        ``D[i, j] = ||A[i] - B[j]||^2``, float values >= 0.

    Raises
    ------
    ValueError
        If ``A`` or ``B`` is not 2-D, or if they have different numbers of features.

    Notes
    -----
    Tested against ``scipy.spatial.distance.cdist(A, B, "sqeuclidean")``.

    Examples
    --------
    >>> A = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> B = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    >>> pairwise_sq_distances(A, B)
    array([[ 1.,  4., 25.],
           [ 1.,  2., 13.]])
    """
    # TODO: check the shapes, then compute all the squared distances at once.
    raise NotImplementedError("pairwise_sq_distances() is not implemented yet")


class NearestCentroid:
    """Classifier that predicts the class whose centroid is the closest.

    The centroid of a class is the mean of its training samples. A new sample gets the
    label of the nearest centroid (squared Euclidean distance): the decision regions are
    the Voronoi cells of the centroids, and the boundary between two classes is a line
    (a hyperplane in higher dimension).

    This classifier has no hyperparameter: ``__init__`` takes no argument.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit`` (any sortable labels: ints, strings...).
    centroids_ : np.ndarray of shape (n_classes, n_features)
        ``centroids_[k]`` is the mean of the training samples of class ``classes_[k]``.

    Notes
    -----
    ``centroids_`` and ``predict`` are tested against ``sklearn.neighbors.NearestCentroid``.
    Its ``decision_function`` (scikit-learn >= 1.6) divides by a within-class standard
    deviation: ours is simpler and is only tested by properties (its sign or its argmax
    agrees with ``predict``).

    Examples
    --------
    >>> X = np.array([[0.0, 0.0], [0.0, 2.0], [4.0, 0.0], [4.0, 2.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = NearestCentroid().fit(X, y)
    >>> clf.centroids_
    array([[0., 1.],
           [4., 1.]])
    >>> X_new = np.array([[1.0, 1.0], [3.5, 0.0]])
    >>> clf.predict(X_new)
    array([0, 1])
    >>> clf.decision_function(X_new)
    array([-8., 12.])
    >>> clf.score(X, y)
    1.0
    """

    def __init__(self) -> None:
        # No hyperparameter to store: everything is learnt in fit.
        pass

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Compute the centroid of every class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels (any sortable values).

        Returns
        -------
        Self
            The fitted classifier (``self``), with ``classes_`` and ``centroids_``.

        Raises
        ------
        ValueError
            If ``y`` has fewer than 2 distinct classes or if ``len(X) != len(y)``.
        """
        # TODO: find the sorted classes, then average the rows of X of each class.
        raise NotImplementedError("fit() is not implemented yet")

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Score every sample using its squared distances to the centroids.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        np.ndarray
            With 2 classes: shape (n_samples,), ``d²(x, c0) - d²(x, c1)``; a positive
            score means ``classes_[1]``. With K > 2 classes: shape (n_samples, K),
            column k is ``-d²(x, c_k)`` (the largest score is the nearest centroid).
        """
        # TODO: reuse pairwise_sq_distances, then build the scores described above.
        raise NotImplementedError("decision_function() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the label of the nearest centroid for every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_``. Ties (equal distances) go to the
            first class of ``classes_``.
        """
        # TODO: index classes_ with the position of the closest centroid.
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


def kmeans_plusplus(
    X: ArrayLike, n_clusters: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Choose initial k-means centres with the k-means++ seeding.

    Original version of Arthur & Vassilvitskii (2007), without local trials: the first
    centre is a row of X drawn uniformly; each next centre is a row drawn with
    probability ``D(x)^2 / sum(D^2)``, where ``D(x)`` is the distance from x to the
    nearest centre already chosen. Far-away points are therefore likely to be picked,
    and a row that is already a centre (D = 0) can never be picked again.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data points.
    n_clusters : int
        Number of centres k to choose.
    rng : np.random.Generator or None, default=None
        Random generator; ``np.random.default_rng()`` if None.

    Returns
    -------
    np.ndarray of shape (n_clusters, n_features)
        The chosen centres: distinct rows of ``X``.

    Raises
    ------
    ValueError
        If ``n_clusters < 1`` or ``n_clusters`` is larger than the number of distinct
        rows of ``X``.

    Notes
    -----
    Tested by properties: the centres are distinct rows of X; on a toy set the
    selection frequencies match ``D^2 / sum(D^2)`` (chi-square test); after Lloyd's
    algorithm (``sklearn.cluster.KMeans(init=centres, n_init=1)``) the mean inertia is
    not worse than with random rows as initial centres. scikit-learn's own
    ``kmeans_plusplus`` adds local trials, so its draws differ from ours.

    Examples
    --------
    >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
    >>> centres = kmeans_plusplus(X, 2, rng=np.random.default_rng(0))
    >>> centres.shape
    (2, 2)
    """
    # TODO: draw the first centre uniformly, then loop: distances to the nearest
    # chosen centre, probabilities proportional to D^2, draw with rng.choice(p=...).
    raise NotImplementedError("kmeans_plusplus() is not implemented yet")


class KMeans:
    """k-means clustering with Lloyd's algorithm.

    One run starts from k initial centres and repeats two steps: (1) assign every
    sample to its nearest centre, (2) move every centre to the mean of its samples.
    It stops when the assignment no longer changes, when the sum of the squared centre
    shifts is <= ``tol * mean(np.var(X, axis=0))`` (scikit-learn's rule), or after
    ``max_iter`` iterations. ``labels_`` and ``inertia_`` are then computed with the
    final centres. Lloyd's algorithm only finds a local minimum of the inertia
    ``sum_i ||x_i - mu_{c_i}||^2``: ``fit`` does ``n_init`` runs from different
    initialisations and keeps the one with the lowest inertia (the first one on ties).

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters k.
    init : str or array-like of shape (n_clusters, n_features), default="k-means++"
        Initial centres: "k-means++" (``kmeans_plusplus``), "random" (k distinct rows
        of X drawn uniformly) or given centres (then a single run is done, whatever
        ``n_init``).
    n_init : int, default=10
        Number of runs with different initialisations. scikit-learn >= 1.4 uses
        ``n_init="auto"`` (a single run with k-means++).
    max_iter : int, default=300
        Maximum number of Lloyd iterations per run.
    tol : float, default=1e-4
        Relative tolerance on the squared centre shifts (see above).
    random_state : int or None, default=None
        Seed of the generator ``np.random.default_rng(random_state)`` created in
        ``fit`` and used by every initialisation.

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Centres of the best run.
    labels_ : np.ndarray of shape (n_samples,)
        Index (int) of the nearest centre of every training sample.
    inertia_ : float
        Sum of the squared distances of the samples to their nearest centre.
    n_iter_ : int
        Number of Lloyd iterations of the best run.

    Notes
    -----
    Tested against ``sklearn.cluster.KMeans(n_clusters=k, init=C0, n_init=1,
    algorithm="lloyd", tol=tol)``: same ``cluster_centers_``, ``labels_``, ``inertia_``
    and ``n_iter_``. An empty cluster keeps its previous centre (scikit-learn moves it
    to a far-away point instead, so results differ in that rare case).

    Examples
    --------
    >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
    >>> km = KMeans(n_clusters=2, init=np.array([[0.0, 0.0], [5.0, 5.0]])).fit(X)
    >>> km.cluster_centers_
    array([[0. , 0.5],
           [5. , 5.5]])
    >>> km.labels_
    array([0, 0, 1, 1])
    >>> km.inertia_
    1.0
    >>> km.predict(np.array([[1.0, 1.0], [4.0, 4.0]]))
    array([0, 1])
    >>> km.score(X)
    -1.0
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | ArrayLike = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Run k-means ``n_init`` times and keep the run with the lowest inertia.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to cluster.
        y : None
            Ignored (unsupervised); accepted so that ``fit(X, y)`` works everywhere,
            e.g. in ``cross_val_score``.

        Returns
        -------
        Self
            The fitted estimator (``self``), with ``cluster_centers_``, ``labels_``,
            ``inertia_`` and ``n_iter_``.

        Raises
        ------
        ValueError
            If ``n_clusters < 1`` or ``n_clusters > n_samples``, ``n_init < 1``,
            ``max_iter < 1``, ``init`` is an unknown string, or an ``init`` array does
            not have the shape (n_clusters, n_features).
        """
        # TODO: validate, create the generator, then for each run: initial centres,
        # Lloyd iterations, inertia; keep the best run.
        raise NotImplementedError("fit() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the index of the nearest learnt centre for every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Cluster indices (ints in 0..n_clusters-1).
        """
        # TODO: nearest centre with pairwise_sq_distances.
        raise NotImplementedError("predict() is not implemented yet")

    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """Fit on X and return ``labels_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to cluster.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Cluster index of every sample.
        """
        # TODO: one line with fit.
        raise NotImplementedError("fit_predict() is not implemented yet")

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Return the Euclidean distances (not squared) of every sample to every centre.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_clusters)
            ``D[i, k] = ||X[i] - cluster_centers_[k]||``.
        """
        # TODO: square root of the squared distances.
        raise NotImplementedError("transform() is not implemented yet")

    def score(self, X: ArrayLike, y: ArrayLike | None = None) -> float:
        """Return minus the inertia of X with respect to the learnt centres.

        scikit-learn convention: a score is "higher is better", hence the minus sign.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : None
            Ignored.

        Returns
        -------
        float
            ``-sum_i min_k ||X[i] - cluster_centers_[k]||^2`` (<= 0).
        """
        # TODO: squared distance of each sample to its nearest centre, summed.
        raise NotImplementedError("score() is not implemented yet")


def silhouette_samples(X: ArrayLike, labels: ArrayLike) -> np.ndarray:
    """Compute the silhouette coefficient of every sample.

    ``s(i) = (b(i) - a(i)) / max(a(i), b(i))`` where ``a(i)`` is the mean Euclidean
    distance from sample i to the other samples of its own cluster and ``b(i)`` the
    mean distance to the samples of the nearest other cluster. Close to 1: well inside
    its cluster; close to 0: on a border; negative: probably in the wrong cluster.
    A sample alone in its cluster gets 0.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data points.
    labels : array-like of shape (n_samples,)
        Cluster label of every sample.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Silhouette values in [-1, 1].

    Raises
    ------
    ValueError
        If the number of distinct labels is < 2 or > n_samples - 1.

    Notes
    -----
    Tested against ``sklearn.metrics.silhouette_samples``.

    Examples
    --------
    >>> X = np.array([[0.0], [1.0], [4.0], [6.0]])
    >>> silhouette_samples(X, [0, 0, 1, 1]).round(4)
    array([0.8   , 0.75  , 0.4286, 0.6364])
    """
    # TODO: distance matrix (square root of pairwise_sq_distances), then a(i) and b(i)
    # with one boolean mask per cluster.
    raise NotImplementedError("silhouette_samples() is not implemented yet")


def silhouette_score(X: ArrayLike, labels: ArrayLike) -> float:
    """Compute the mean silhouette coefficient over all samples (higher is better).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data points.
    labels : array-like of shape (n_samples,)
        Cluster label of every sample.

    Returns
    -------
    float
        Mean of ``silhouette_samples(X, labels)``, in [-1, 1].

    Raises
    ------
    ValueError
        Same conditions as ``silhouette_samples``.

    Notes
    -----
    Tested against ``sklearn.metrics.silhouette_score``.

    Examples
    --------
    >>> X = np.array([[0.0], [1.0], [4.0], [6.0]])
    >>> round(silhouette_score(X, [0, 0, 1, 1]), 4)
    0.6537
    """
    # TODO: one line with silhouette_samples.
    raise NotImplementedError("silhouette_score() is not implemented yet")
