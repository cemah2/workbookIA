"""Distances, nearest centroid and k-means — mylearn, chapter 7 (Classification).

This module holds your first estimators "à la scikit-learn": ``fit`` learns from the
data and stores the learnt attributes (their names end with ``_``), ``predict`` uses them.
You code vectorised squared distances, the nearest centroid classifier, the k-means++
seeding, Lloyd's k-means algorithm and the silhouette coefficient used to choose k.

Reference implementation: read it only after trying (``mon_travail/mylearn/cluster.py``).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike


def _as_2d(X: ArrayLike, name: str) -> np.ndarray:
    """A 2-D float array (n_samples, n_features), or a ValueError that says why not."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (n_samples, n_features), got shape {arr.shape}")
    return arr


def _nearest(X: ArrayLike, centres: np.ndarray) -> np.ndarray:
    """Index of the nearest centre of every row of X (ties: the first centre)."""
    return pairwise_sq_distances(X, centres).argmin(axis=1)


def _lloyd(X: np.ndarray, centres: np.ndarray, max_iter: int, tol: float):
    """One run of Lloyd's algorithm from the given centres, with scikit-learn's stopping rules.

    Returns (centres, labels, inertia, n_iter). An empty cluster keeps its previous centre.
    """
    labels_old = np.full(len(X), -1)
    strict = False
    for i in range(max_iter):
        labels = _nearest(X, centres)                       # (1) assignment step
        new = centres.copy()
        for j in range(len(centres)):                        # (2) update step
            members = labels == j
            if members.any():
                new[j] = X[members].mean(axis=0)
        shift = float(((new - centres) ** 2).sum())
        centres = new
        if np.array_equal(labels, labels_old):               # same assignment: converged
            strict = True
            break
        if shift <= tol:                                     # the centres hardly moved
            break
        labels_old = labels
    if not strict:                                           # labels of the final centres
        labels = _nearest(X, centres)
    inertia = float(((X - centres[labels]) ** 2).sum())
    return centres, labels, inertia, i + 1


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
    A = _as_2d(A, "A")
    B = _as_2d(B, "B")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have the same number of features, got {A.shape[1]} and {B.shape[1]}")
    sq = (A * A).sum(axis=1)[:, None] - 2.0 * (A @ B.T) + (B * B).sum(axis=1)[None, :]
    return np.maximum(sq, 0.0)                               # rounding errors can give tiny negatives


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
        X = _as_2d(X, "X")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"y must be a 1-D array with one label per row of X, got shape {y.shape} for X of shape {X.shape}")
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError(f"at least 2 classes are needed, got {len(classes)}")
        self.classes_ = classes
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in classes])
        return self

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
        D = pairwise_sq_distances(X, self.centroids_)
        if len(self.classes_) == 2:
            return D[:, 0] - D[:, 1]
        return -D

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
        return self.classes_[_nearest(X, self.centroids_)]

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
        return float(np.mean(self.predict(X) == np.asarray(y)))


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
    X = _as_2d(X, "X")
    n_distinct = len(np.unique(X, axis=0))
    if not 1 <= n_clusters <= n_distinct:
        raise ValueError(f"n_clusters must be between 1 and the number of distinct rows of X ({n_distinct}), "
                         f"got {n_clusters}")
    rng = np.random.default_rng() if rng is None else rng
    chosen = [int(rng.integers(len(X)))]                     # the first centre: uniform
    d2 = ((X - X[chosen[0]]) ** 2).sum(axis=1)              # D(x)^2: squared distance to the nearest centre
    for _ in range(1, n_clusters):
        nxt = int(rng.choice(len(X), p=d2 / d2.sum()))      # far-away points are more likely
        chosen.append(nxt)
        d2 = np.minimum(d2, ((X - X[nxt]) ** 2).sum(axis=1))
    return X[chosen].copy()


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
        X = _as_2d(X, "X")
        n_samples, n_features = X.shape
        k = self.n_clusters
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= n_samples:
            raise ValueError(f"n_clusters must be an integer between 1 and n_samples = {n_samples}, got {k!r}")
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError(f"n_init must be an integer >= 1, got {self.n_init!r}")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError(f"max_iter must be an integer >= 1, got {self.max_iter!r}")
        given = None
        if isinstance(self.init, str):
            if self.init not in ("k-means++", "random"):
                raise ValueError(f"init must be 'k-means++', 'random' or an array of centres, got {self.init!r}")
        else:
            given = np.array(self.init, dtype=float)             # a copy: init itself is never modified
            if given.shape != (k, n_features):
                raise ValueError(f"an init array must have the shape (n_clusters, n_features) = {(k, n_features)}, "
                                 f"got {given.shape}")
        rng = np.random.default_rng(self.random_state)
        tol = self.tol * float(np.mean(np.var(X, axis=0)))      # scikit-learn's relative tolerance
        best = None
        for _ in range(1 if given is not None else self.n_init):
            if given is not None:
                centres = given.copy()
            elif self.init == "k-means++":
                centres = kmeans_plusplus(X, k, rng=rng)
            else:
                centres = X[rng.choice(n_samples, size=k, replace=False)].copy()
            run = _lloyd(X, centres, self.max_iter, tol)
            if best is None or run[2] < best[2]:                 # strict <: the first run wins on ties
                best = run
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

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
        return _nearest(X, self.cluster_centers_)

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
        return self.fit(X).labels_

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
        return np.sqrt(pairwise_sq_distances(X, self.cluster_centers_))

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
        return -float(pairwise_sq_distances(X, self.cluster_centers_).min(axis=1).sum())


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
    X = _as_2d(X, "X")
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(X):
        raise ValueError(f"labels must be a 1-D array with one label per row of X, got shape {labels.shape} "
                         f"for X of shape {X.shape}")
    _, codes = np.unique(labels, return_inverse=True)
    n, n_labels = len(X), int(codes.max()) + 1
    if not 2 <= n_labels <= n - 1:
        raise ValueError(f"the number of distinct labels must be between 2 and n_samples - 1 = {n - 1}, got {n_labels}")
    dist = np.sqrt(pairwise_sq_distances(X, X))
    np.fill_diagonal(dist, 0.0)
    counts = np.bincount(codes, minlength=n_labels)
    sums = np.stack([dist[:, codes == c].sum(axis=1) for c in range(n_labels)], axis=1)   # (n, n_labels)
    rows = np.arange(n)
    own = counts[codes]
    a = sums[rows, codes] / np.maximum(own - 1, 1)          # mean distance to the others of its cluster
    means = sums / counts
    means[rows, codes] = np.inf
    b = means.min(axis=1)                                    # mean distance to the nearest other cluster
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (b - a) / np.maximum(a, b)
    s[own == 1] = 0.0                                        # a sample alone in its cluster
    return np.nan_to_num(s)                                  # a == b == 0 (duplicate points): 0


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
    return float(np.mean(silhouette_samples(X, labels)))
