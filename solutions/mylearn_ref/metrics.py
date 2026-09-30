"""Classification metrics — mylearn, chapter 3 (Probability and measuring quality).

Measures of the quality of a classifier: confusion matrix, accuracy, precision,
recall and F-scores (binary case and multiclass averages), every rate of a binary
confusion matrix, ROC and precision-recall curves with their areas, and calibration.
The conventions are scikit-learn's (rows of the confusion matrix = truth), so that
your results can be compared with the tools used at work. Reused in every later
chapter (7 to 29) and in the final project.

Reference implementation: read it only after trying (``mon_travail/mylearn/metrics.py``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


_AVERAGES = ("binary", "macro", "micro", "weighted", None)


def _labels(y: ArrayLike, name: str) -> np.ndarray:
    """A 1-D array of labels."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _kind(arr: np.ndarray) -> str:
    """'text' for strings, 'number' for booleans, integers and floats, 'object' otherwise."""
    if arr.dtype.kind in "US":
        return "text"
    if arr.dtype.kind in "biuf":
        return "number"
    return "object"


def _check_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Two non-empty 1-D label arrays of the same length and of the same kind."""
    a, b = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if len(a) != len(b):
        raise ValueError(f"y_true and y_pred have different lengths: {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("y_true and y_pred are empty")
    if {_kind(a), _kind(b)} == {"text", "number"}:
        raise ValueError("y_true and y_pred mix text labels and numbers")
    return a, b


def _sorted_labels(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The sorted union of the labels found in a and b."""
    try:
        return np.unique(np.concatenate([a, b]))
    except TypeError as exc:  # object arrays mixing types that cannot be sorted
        raise ValueError(f"the labels cannot be sorted: {exc}") from None


def _check_binary_labels(present: np.ndarray, pos_label) -> None:
    """At most two labels, and pos_label among them when there are two."""
    if len(present) > 2:
        raise ValueError(
            f"{len(present)} labels found ({present.tolist()}): the binary case allows at most two; "
            "choose average='macro', 'micro', 'weighted' or None"
        )
    if len(present) == 2 and pos_label not in present.tolist():
        raise ValueError(f"pos_label={pos_label!r} is not one of the labels {present.tolist()}")


def _divide(num, den, zero_division: float) -> np.ndarray:
    """num / den element by element, with zero_division where den == 0."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, float(zero_division))
    np.divide(num, den, out=out, where=den != 0)
    return out


def _class_counts(a: np.ndarray, b: np.ndarray, labels: np.ndarray):
    """TP, FP, FN and support (number of true samples) of each label, one against the rest."""
    C = confusion_matrix(a, b, labels=labels)
    tp = np.diag(C)
    return tp, C.sum(axis=0) - tp, C.sum(axis=1) - tp, C.sum(axis=1)


def _score(y_true, y_pred, pos_label, average, zero_division, kind: str, beta: float = 1.0):
    """Precision, recall or F-beta for the binary case or combined over the classes."""
    a, b = _check_pair(y_true, y_pred)
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
    present = _sorted_labels(a, b)
    if average == "binary":
        _check_binary_labels(present, pos_label)
        t, p = a == pos_label, b == pos_label
        tp = np.array([np.sum(t & p)])
        fp, fn, support = np.array([np.sum(~t & p)]), np.array([np.sum(t & ~p)]), np.array([np.sum(t)])
    else:
        tp, fp, fn, support = _class_counts(a, b, present)
    if average == "micro":
        tp, fp, fn = np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()])
    if kind == "precision":
        values = _divide(tp, tp + fp, zero_division)
    elif kind == "recall":
        values = _divide(tp, tp + fn, zero_division)
    else:
        b2 = beta**2
        values = _divide((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp, zero_division)
    if average is None:
        return values
    if average == "macro":
        return float(np.mean(values))
    if average == "weighted":
        return float(np.average(values, weights=support))
    return float(values[0])  # binary or micro


def _binary_scores(y_true: ArrayLike, y_score: ArrayLike, pos_label) -> tuple[np.ndarray, np.ndarray]:
    """Boolean truth (positive class) and float scores, checked."""
    t = _labels(y_true, "y_true")
    s = np.asarray(y_score, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"y_score must be 1-D, got shape {s.shape}")
    if len(t) != len(s):
        raise ValueError(f"y_true and y_score have different lengths: {len(t)} and {len(s)}")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN")
    classes = np.unique(t)
    if len(classes) != 2:
        raise ValueError(f"y_true must contain exactly two classes, found {len(classes)}: {classes.tolist()}")
    if pos_label not in classes.tolist():
        raise ValueError(f"pos_label={pos_label!r} is not one of the classes {classes.tolist()}")
    return t == pos_label, s


def _counts_by_threshold(positive: np.ndarray, scores: np.ndarray):
    """TP and FP when predicting positive for score >= t, for every distinct score t (decreasing)."""
    order = np.argsort(-scores, kind="mergesort")
    s, pos = scores[order], positive[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last sample of each group of equal scores
    return np.cumsum(pos)[last], np.cumsum(~pos)[last], s[last]


def _check_outcomes(y_true: ArrayLike, y_prob: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """0/1 outcomes and probabilities in [0, 1] of the same length."""
    t = _labels(y_true, "y_true")
    p = np.asarray(y_prob, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"y_prob must be 1-D, got shape {p.shape}")
    if len(t) != len(p):
        raise ValueError(f"y_true and y_prob have different lengths: {len(t)} and {len(p)}")
    if len(t) == 0:
        raise ValueError("y_true and y_prob are empty")
    if _kind(t) != "number" or not np.isin(t, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("y_prob must contain probabilities in [0, 1]")
    return t.astype(float), p


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, labels: ArrayLike | None = None
) -> np.ndarray:
    """Count the (true label, predicted label) pairs in a confusion matrix.

    ``C[i, j]`` is the number of samples whose true label is ``labels[i]`` and
    whose predicted label is ``labels[j]``: rows are the truth, columns are the
    predictions (scikit-learn layout). For 0/1 labels this gives
    ``[[TN, FP], [FN, TP]]``; the book puts TP in the top-left corner instead.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels (ints or strings).
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : array-like of shape (n_labels,) or None, default=None
        Order of the rows and columns. If None, the sorted union of the labels
        found in ``y_true`` and ``y_pred``.

    Returns
    -------
    np.ndarray of shape (n_labels, n_labels)
        Int counts; they sum to ``n_samples``.

    Raises
    ------
    ValueError
        If the lengths differ, if the inputs are empty, or if a label of ``y_true``
        or ``y_pred`` is missing from ``labels`` (scikit-learn would silently drop
        those samples).

    Notes
    -----
    Tested against ``sklearn.metrics.confusion_matrix``.

    Examples
    --------
    >>> confusion_matrix([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])
    array([[2, 0],
           [1, 2]])
    >>> y_true = ["spam", "ham", "spam", "ham", "ham"]
    >>> y_pred = ["spam", "spam", "ham", "ham", "ham"]
    >>> confusion_matrix(y_true, y_pred, labels=["spam", "ham"])
    array([[1, 1],
           [1, 2]])
    """
    a, b = _check_pair(y_true, y_pred)
    if labels is None:
        labs = _sorted_labels(a, b)
    else:
        labs = np.asarray(labels)
        if labs.ndim != 1 or len(labs) == 0:
            raise ValueError("labels must be a non-empty 1-D list")
    index = {label: i for i, label in enumerate(labs.tolist())}
    if len(index) != len(labs):
        raise ValueError("labels contains duplicates")
    try:
        rows = [index[label] for label in a.tolist()]
        cols = [index[label] for label in b.tolist()]
    except KeyError as exc:
        raise ValueError(f"the label {exc.args[0]!r} of the data is missing from labels") from None
    C = np.zeros((len(labs), len(labs)), dtype=np.int64)
    np.add.at(C, (rows, cols), 1)
    return C


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the accuracy: the share of predictions equal to the truth.

    Works for any number of classes.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        The accuracy, in [0, 1].

    Raises
    ------
    ValueError
        If the lengths differ or the inputs are empty.

    Notes
    -----
    Tested against ``sklearn.metrics.accuracy_score``.

    Examples
    --------
    >>> accuracy([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    a, b = _check_pair(y_true, y_pred)
    return float(np.mean(a == b))


def precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: int | str = 1,
    average: str | None = "binary",
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """Compute the precision TP / (TP + FP): the share of positive predictions that are right.

    Also called the positive predictive value. In the multiclass case, each class
    in turn plays the positive class ("one against the rest"), then the per-class
    values are combined according to ``average``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int or str, default=1
        The positive class, used only when ``average='binary'``.
    average : {'binary', 'macro', 'micro', 'weighted'} or None, default='binary'
        ``'binary'``: only the class ``pos_label`` (at most two labels allowed).
        ``'macro'``: unweighted mean of the per-class values.
        ``'micro'``: one value computed from the TP and FP pooled over the classes.
        ``'weighted'``: mean of the per-class values weighted by their support
        (number of true samples of each class).
        ``None``: one value per class, in the sorted order of the labels found in
        ``y_true`` and ``y_pred``.
    zero_division : float, default=0.0
        Value used when ``TP + FP = 0`` (no positive prediction).

    Returns
    -------
    float or np.ndarray
        A Python float, or a float array of shape (n_labels,) if ``average`` is None.

    Raises
    ------
    ValueError
        If the lengths differ or the inputs are empty, if ``average`` is unknown,
        or, with ``average='binary'``, if more than two labels are present or if
        two labels are present and ``pos_label`` is not one of them.

    Notes
    -----
    Tested against ``sklearn.metrics.precision_score(y_true, y_pred,
    pos_label=pos_label, average=average, zero_division=zero_division)``.

    Examples
    --------
    >>> precision([0, 1, 1, 0, 1], [0, 1, 0, 1, 1])
    0.6666666666666666
    >>> precision([0, 1, 2, 2], [0, 2, 2, 1], average=None)
    array([1. , 0. , 0.5])
    >>> precision([0, 1, 2, 2], [0, 2, 2, 1], average="macro")
    0.5
    """
    return _score(y_true, y_pred, pos_label, average, zero_division, "precision")


def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: int | str = 1,
    average: str | None = "binary",
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """Compute the recall TP / (TP + FN): the share of actual positives that are found.

    Also called sensitivity or true positive rate (TPR). In the multiclass case,
    each class in turn plays the positive class, then the per-class values are
    combined according to ``average``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int or str, default=1
        The positive class, used only when ``average='binary'``.
    average : {'binary', 'macro', 'micro', 'weighted'} or None, default='binary'
        As in ``precision`` (``'micro'`` pools TP and FN over the classes).
    zero_division : float, default=0.0
        Value used when ``TP + FN = 0`` (no actual positive).

    Returns
    -------
    float or np.ndarray
        A Python float, or a float array of shape (n_labels,) if ``average`` is None.

    Raises
    ------
    ValueError
        Same cases as ``precision``.

    Notes
    -----
    Tested against ``sklearn.metrics.recall_score`` (same arguments).

    Examples
    --------
    >>> recall([0, 1, 1, 0, 1], [0, 1, 0, 1, 1])
    0.6666666666666666
    >>> y_true = ["spam", "ham", "spam", "ham", "ham"]
    >>> y_pred = ["spam", "spam", "ham", "ham", "ham"]
    >>> recall(y_true, y_pred, pos_label="spam")
    0.5
    """
    return _score(y_true, y_pred, pos_label, average, zero_division, "recall")


def fbeta(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    beta: float = 1.0,
    pos_label: int | str = 1,
    average: str | None = "binary",
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """Compute the F-beta score, a weighted harmonic mean of precision and recall.

    ``F_beta = (1 + beta²) TP / ((1 + beta²) TP + beta² FN + FP)``. ``beta > 1``
    gives more weight to recall, ``beta < 1`` to precision; ``beta = 1`` is the F1
    score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    beta : float, default=1.0
        Weight of recall, ``> 0``.
    pos_label : int or str, default=1
        As in ``precision``.
    average : {'binary', 'macro', 'micro', 'weighted'} or None, default='binary'
        As in ``precision`` (``'macro'`` and ``'weighted'`` average the per-class
        F-scores; ``'micro'`` computes one F-score from the pooled counts).
    zero_division : float, default=0.0
        Value used when ``TP + FN + FP = 0``.

    Returns
    -------
    float or np.ndarray
        A Python float, or a float array of shape (n_labels,) if ``average`` is None.

    Raises
    ------
    ValueError
        If ``beta <= 0``, plus the cases of ``precision``.

    Notes
    -----
    Tested against ``sklearn.metrics.fbeta_score`` (same arguments).

    Examples
    --------
    >>> round(fbeta([1, 1, 1, 0, 0], [1, 0, 0, 1, 0], beta=2), 4)
    0.3571
    >>> round(fbeta([1, 1, 1, 0, 0], [1, 0, 0, 1, 0], beta=0.5), 4)
    0.4545
    """
    if not beta > 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    return _score(y_true, y_pred, pos_label, average, zero_division, "fbeta", beta=beta)


def f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: int | str = 1,
    average: str | None = "binary",
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """Compute the F1 score, the harmonic mean of precision and recall.

    ``F1 = 2 P R / (P + R) = 2 TP / (2 TP + FN + FP)``: ``fbeta`` with ``beta = 1``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int or str, default=1
        As in ``precision``.
    average : {'binary', 'macro', 'micro', 'weighted'} or None, default='binary'
        As in ``fbeta``.
    zero_division : float, default=0.0
        As in ``fbeta``.

    Returns
    -------
    float or np.ndarray
        A Python float, or a float array of shape (n_labels,) if ``average`` is None.

    Raises
    ------
    ValueError
        Same cases as ``fbeta``.

    Notes
    -----
    Tested against ``sklearn.metrics.f1_score`` (same arguments).

    Examples
    --------
    >>> f1([0, 1, 1, 0], [0, 1, 0, 0])
    0.6666666666666666
    >>> f1([0, 1, 2, 2], [0, 2, 2, 1], average="weighted")
    0.5
    """
    return fbeta(y_true, y_pred, beta=1.0, pos_label=pos_label, average=average,
                 zero_division=zero_division)


def classification_rates(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: int | str = 1,
    zero_division: float = 0.0,
) -> dict[str, float]:
    """Compute every rate of a binary confusion matrix in one dictionary.

    The book's summary table (Fig. 3.32). With TP, FN, FP, TN the four counts and
    n their sum, the keys are, in this order:

    - ``'accuracy'``: (TP + TN) / n
    - ``'balanced_accuracy'``: (recall + specificity) / 2
    - ``'precision'``: TP / (TP + FP)
    - ``'recall'``: TP / (TP + FN) (sensitivity, TPR)
    - ``'specificity'``: TN / (TN + FP) (TNR)
    - ``'npv'``: TN / (TN + FN) (negative predictive value)
    - ``'fpr'``: FP / (FP + TN) (false positive rate)
    - ``'fnr'``: FN / (FN + TP) (false negative rate)
    - ``'fdr'``: FP / (FP + TP) (false discovery rate)
    - ``'false_omission_rate'``: FN / (FN + TN)
    - ``'prevalence'``: (TP + FN) / n
    - ``'f1'``: 2 TP / (2 TP + FP + FN)
    - ``'mcc'``: Matthews correlation coefficient,
      (TP TN - FP FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels, at most two distinct values.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int or str, default=1
        The positive class; the other label is the negative class.
    zero_division : float, default=0.0
        Value used for any 0/0 ratio.

    Returns
    -------
    dict of str to float
        The 13 rates above, as Python floats, with the keys in the order listed.

    Raises
    ------
    ValueError
        If more than two labels are present in ``y_true`` and ``y_pred``
        together, if two labels are present and ``pos_label`` is not one of them,
        if the lengths differ or if the inputs are empty.

    Notes
    -----
    Tested against scikit-learn: ``precision_score`` and ``recall_score`` (with
    ``pos_label`` swapped for ``npv`` and ``specificity``),
    ``balanced_accuracy_score``, ``f1_score``, ``matthews_corrcoef``, and the
    identities ``fpr = 1 - specificity``, ``fnr = 1 - recall``,
    ``fdr = 1 - precision``, ``false_omission_rate = 1 - npv``.

    Examples
    --------
    >>> rates = classification_rates([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0])
    >>> rates["specificity"], rates["npv"], rates["prevalence"]
    (0.6, 0.75, 0.375)
    >>> round(rates["mcc"], 4)
    0.2582
    """
    a, b = _check_pair(y_true, y_pred)
    _check_binary_labels(_sorted_labels(a, b), pos_label)
    t, p = a == pos_label, b == pos_label
    tp, fn = int(np.sum(t & p)), int(np.sum(t & ~p))
    fp, tn = int(np.sum(~t & p)), int(np.sum(~t & ~p))
    n = tp + fn + fp + tn

    def ratio(num: float, den: float) -> float:
        return float(num / den) if den != 0 else float(zero_division)

    recall_ = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "accuracy": ratio(tp + tn, n),
        "balanced_accuracy": (recall_ + specificity) / 2,
        "precision": ratio(tp, tp + fp),
        "recall": recall_,
        "specificity": specificity,
        "npv": ratio(tn, tn + fn),
        "fpr": ratio(fp, fp + tn),
        "fnr": ratio(fn, fn + tp),
        "fdr": ratio(fp, fp + tp),
        "false_omission_rate": ratio(fn, fn + tn),
        "prevalence": ratio(tp + fn, n),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": ratio(tp * tn - fp * fn, np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))),
    }


def roc_curve(
    y_true: ArrayLike, y_score: ArrayLike, pos_label: int | str = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ROC curve: false and true positive rates for every threshold.

    For a threshold t, the classifier predicts positive when ``score >= t``. The
    thresholds are ``+inf`` (nothing predicted positive: the point (0, 0)) followed
    by every distinct score, in decreasing order.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels, exactly two distinct values.
    y_score : array-like of shape (n_samples,)
        Scores, higher = more likely positive (probabilities or any real numbers).
    pos_label : int or str, default=1
        The positive class.

    Returns
    -------
    fpr : np.ndarray of shape (n_thresholds,)
        False positive rate FP / (FP + TN) at each threshold (non-decreasing).
    tpr : np.ndarray of shape (n_thresholds,)
        True positive rate TP / (TP + FN) at each threshold (non-decreasing).
    thresholds : np.ndarray of shape (n_thresholds,)
        Decreasing thresholds, ``thresholds[0] = np.inf``;
        ``n_thresholds = 1 + number of distinct scores``.

    Raises
    ------
    ValueError
        If ``y_true`` does not contain exactly two classes or ``pos_label`` is not
        one of them, if the lengths differ, or if the scores contain NaN.

    Notes
    -----
    Tested against ``sklearn.metrics.roc_curve(y_true, y_score, pos_label=pos_label,
    drop_intermediate=False)``.

    Examples
    --------
    >>> fpr, tpr, thresholds = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> thresholds
    array([ inf, 0.8 , 0.4 , 0.35, 0.1 ])
    """
    positive, scores = _binary_scores(y_true, y_score, pos_label)
    tps, fps, thresholds = _counts_by_threshold(positive, scores)
    tps, fps = np.r_[0, tps], np.r_[0, fps]  # threshold +inf: nothing predicted positive
    return fps / fps[-1], tps / tps[-1], np.r_[np.inf, thresholds]


def auc(x: ArrayLike, y: ArrayLike) -> float:
    """Compute the area under a curve given by its points, with the trapezoidal rule.

    Between two consecutive points, the curve is replaced by a straight segment;
    the area is the sum of the areas of the trapezoids below the segments.

    Parameters
    ----------
    x : array-like of shape (n_points,)
        Coordinates, monotonic: all non-decreasing or all non-increasing (repeated
        values are allowed, as in a ROC curve).
    y : array-like of shape (n_points,)
        Values of the curve at ``x``.

    Returns
    -------
    float
        The area. As in scikit-learn, it does not depend on the direction of
        ``x``: listing the points from right to left gives the same area.

    Raises
    ------
    ValueError
        If there are fewer than 2 points, if the lengths differ, or if ``x`` is
        neither non-decreasing nor non-increasing.

    Notes
    -----
    Tested against ``sklearn.metrics.auc`` (and ``np.trapezoid`` for an
    increasing ``x``).

    Examples
    --------
    >>> auc([0, 0, 0.5, 0.5, 1], [0, 0.5, 0.5, 1, 1])
    0.75
    >>> auc([1, 0.5, 0], [1, 1, 0])
    0.75
    """
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError(f"x and y must be 1-D, got shapes {xa.shape} and {ya.shape}")
    if len(xa) != len(ya):
        raise ValueError(f"x and y have different lengths: {len(xa)} and {len(ya)}")
    if len(xa) < 2:
        raise ValueError(f"at least 2 points are needed to compute an area, got {len(xa)}")
    dx = np.diff(xa)
    if np.all(dx >= 0):
        direction = 1.0
    elif np.all(dx <= 0):
        direction = -1.0
    else:
        raise ValueError("x is neither non-decreasing nor non-increasing")
    return float(direction * np.sum(dx * (ya[1:] + ya[:-1]) / 2))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike, pos_label: int | str = 1) -> float:
    """Compute the area under the ROC curve (ROC-AUC).

    It equals the probability that a randomly chosen positive gets a higher score
    than a randomly chosen negative, a tie counting 1/2. 1 is a perfect ranking,
    0.5 is no better than chance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels, exactly two distinct values.
    y_score : array-like of shape (n_samples,)
        Scores, higher = more likely positive.
    pos_label : int or str, default=1
        The positive class.

    Returns
    -------
    float
        The area, in [0, 1].

    Raises
    ------
    ValueError
        Same cases as ``roc_curve``.

    Notes
    -----
    Tested against ``sklearn.metrics.roc_auc_score`` and against a direct count
    over all (positive, negative) pairs on small inputs.

    Examples
    --------
    >>> roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    0.75
    >>> roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.8])
    0.875
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    return auc(fpr, tpr)


def precision_recall_curve(
    y_true: ArrayLike, y_score: ArrayLike, pos_label: int | str = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision and recall for every threshold.

    For a threshold t, the classifier predicts positive when ``score >= t``. The
    thresholds are the distinct scores in increasing order (scikit-learn's
    convention); a last point (precision 1, recall 0) with no threshold closes the
    curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels, exactly two distinct values.
    y_score : array-like of shape (n_samples,)
        Scores, higher = more likely positive.
    pos_label : int or str, default=1
        The positive class.

    Returns
    -------
    precision : np.ndarray of shape (n_thresholds + 1,)
        ``precision[i]`` at ``thresholds[i]``, then 1.0.
    recall : np.ndarray of shape (n_thresholds + 1,)
        ``recall[i]`` at ``thresholds[i]`` (non-increasing), then 0.0.
    thresholds : np.ndarray of shape (n_thresholds,)
        The distinct scores, increasing.

    Raises
    ------
    ValueError
        If ``y_true`` does not contain exactly two classes or ``pos_label`` is not
        one of them, or if the lengths differ.

    Notes
    -----
    Tested against ``sklearn.metrics.precision_recall_curve(y_true, y_score,
    pos_label=pos_label, drop_intermediate=False)`` (scikit-learn 1.6: every
    threshold is kept, no truncation at full recall).

    Examples
    --------
    >>> y_true, y_score = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    >>> precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    >>> precision
    array([0.5       , 0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.1 , 0.35, 0.4 , 0.8 ])
    """
    positive, scores = _binary_scores(y_true, y_score, pos_label)
    tps, fps, thresholds = _counts_by_threshold(positive, scores)
    precision_ = tps / (tps + fps)
    recall_ = tps / tps[-1]
    # increasing thresholds (scikit-learn), then the closing point (precision 1, recall 0)
    return np.r_[precision_[::-1], 1.0], np.r_[recall_[::-1], 0.0], thresholds[::-1]


def average_precision(
    y_true: ArrayLike, y_score: ArrayLike, pos_label: int | str = 1
) -> float:
    """Compute the average precision (AP), the area under the precision-recall curve.

    Step sum over the thresholds: ``AP = Σ_n (R_n - R_(n-1)) P_n``, where ``P_n``
    and ``R_n`` are the precision and recall at the n-th threshold (from the
    highest to the lowest), with ``R_0 = 0``. There is no trapezoidal
    interpolation: that would be too optimistic.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels, exactly two distinct values.
    y_score : array-like of shape (n_samples,)
        Scores, higher = more likely positive.
    pos_label : int or str, default=1
        The positive class.

    Returns
    -------
    float
        The average precision, in [0, 1].

    Raises
    ------
    ValueError
        Same cases as ``precision_recall_curve``.

    Notes
    -----
    Tested against ``sklearn.metrics.average_precision_score``.

    Examples
    --------
    >>> round(average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 4)
    0.8333
    """
    precision_, recall_, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    p, r = precision_[::-1], recall_[::-1]  # from the highest threshold: r[0] = 0 (the closing point)
    return float(np.sum(np.diff(r) * p[1:]))


def calibration_curve(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the data of a reliability diagram (calibration curve).

    The interval [0, 1] is cut into ``n_bins`` bins of equal width, with edges
    ``np.linspace(0, 1, n_bins + 1)``; a probability that falls exactly on an inner
    edge goes to the lower bin. For each non-empty bin, the function returns the
    observed share of positives and the mean predicted probability. A well
    calibrated model has both values close to each other.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Outcomes, 0 or 1.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of class 1, in [0, 1].
    n_bins : int, default=10
        Number of bins, ``>= 1``.

    Returns
    -------
    prob_true : np.ndarray of shape (n_nonempty_bins,)
        Share of positives in each non-empty bin, in increasing bin order.
    prob_pred : np.ndarray of shape (n_nonempty_bins,)
        Mean predicted probability in each non-empty bin.

    Raises
    ------
    ValueError
        If ``y_true`` contains values other than 0 and 1, if a probability is
        outside [0, 1], if the lengths differ or if ``n_bins < 1``.

    Notes
    -----
    Tested against ``sklearn.calibration.calibration_curve(y_true, y_prob,
    n_bins=n_bins, strategy='uniform')``.

    Examples
    --------
    >>> y_true, y_prob = [0, 0, 1, 1, 1], [0.1, 0.3, 0.6, 0.8, 0.9]
    >>> prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=2)
    >>> prob_true
    array([0., 1.])
    >>> prob_pred
    array([0.2       , 0.76666667])
    """
    t, p = _check_outcomes(y_true, y_prob)
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be an integer >= 1, got {n_bins!r}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.searchsorted(edges[1:-1], p, side="left")  # a value on an inner edge goes to the lower bin
    counts = np.bincount(bins, minlength=n_bins)
    positives = np.bincount(bins, weights=t, minlength=n_bins)
    prob_sums = np.bincount(bins, weights=p, minlength=n_bins)
    nonempty = counts > 0
    return positives[nonempty] / counts[nonempty], prob_sums[nonempty] / counts[nonempty]


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Compute the Brier score: mean squared gap between probability and outcome.

    Mean of ``(y_prob - y_true)²``: 0 for perfect, confident predictions; always
    predicting 0.5 gives 0.25.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Outcomes, 0 or 1.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of class 1, in [0, 1].

    Returns
    -------
    float
        The Brier score, in [0, 1] (lower is better).

    Raises
    ------
    ValueError
        If ``y_true`` contains values other than 0 and 1, if a probability is
        outside [0, 1] or if the lengths differ.

    Notes
    -----
    Tested against ``sklearn.metrics.brier_score_loss``.

    Examples
    --------
    >>> brier_score([0, 1, 1], [0.25, 0.75, 0.5])
    0.125
    """
    t, p = _check_outcomes(y_true, y_prob)
    return float(np.mean((p - t) ** 2))
