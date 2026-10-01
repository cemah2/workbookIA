"""Language identification from letter frequencies (mini-project of part I, reference solution).

A naive Bayes model of characters. Each language is a smoothed probability distribution
of letters (``mylearn.info.char_distribution``); a text is scored by the sum of the
log-probabilities of its letters (the letters are assumed independent: the i.i.d.
hypothesis); Bayes' rule turns the scores and a prior into probabilities
(``mylearn.bayes.bayes_posterior``). A temperature, fitted by gradient descent on the
log loss (``mylearn.calculus``, ``mylearn.info.log_loss``), can correct the confidence of
the probabilities; ``evaluate`` measures a binary detector with ``mylearn.metrics``.
No machine learning library: NumPy and the learner's own ``mylearn``.

    model = LanguageDetector().fit(train_texts, train_labels)
    model.predict(["The weather is lovely today.", "Il fait beau aujourd'hui."])
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mylearn import bayes, calculus, info, metrics

LETTERS = "abcdefghijklmnopqrstuvwxyz"
ACCENTED = "àâæçéèêëîïôœùûüÿ"          # the accented letters and ligatures of French
ALPHABET = LETTERS + ACCENTED            # 42 letters


def posterior_from_loglik(log_lik, prior, temperature: float = 1.0) -> np.ndarray:
    """Turn log-likelihoods into posterior probabilities with Bayes' rule.

    Each row is divided by ``temperature``, then shifted so that its largest value is 0
    (the log-sum-exp trick: multiplying all the likelihoods of a row by the same number does
    not change the posterior, and the largest likelihood becomes 1 instead of underflowing to
    0); ``mylearn.bayes.bayes_posterior`` then combines the likelihoods with the prior.

    Parameters
    ----------
    log_lik : array-like of shape (n_texts, n_classes)
        Natural logarithms of P(text | class).
    prior : array-like of shape (n_classes,)
        Prior probabilities of the classes, summing to 1.
    temperature : float, default=1.0
        Divides the log-likelihoods, ``> 0``: above 1, the probabilities move toward the prior.

    Returns
    -------
    np.ndarray of shape (n_texts, n_classes)
        Posterior probabilities; every row sums to 1.
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    scaled = np.atleast_2d(np.asarray(log_lik, dtype=float)) / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    return np.array([bayes.bayes_posterior(prior, np.exp(row)) for row in shifted]).reshape(scaled.shape)


class LanguageDetector:
    """Naive Bayes language detector on letters.

    Parameters
    ----------
    alphabet : str or sequence of str, default=ALPHABET
        The letters the model looks at, after lower-casing; other characters (spaces,
        punctuation, digits) are ignored.
    smoothing : float, default=1.0
        Laplace pseudo-count added to the count of every letter of every language, ``> 0``:
        no letter gets a probability of 0, so no text gets an infinite surprise.
    prior : array-like or None, default=None
        Prior probabilities of the languages, in the order of ``classes_`` (sorted labels);
        None means a uniform prior. Read by ``predict_proba`` at every call: it can be
        changed after ``fit`` (``model.prior = [0.1, 0.9]``) without fitting again.
    temperature : float, default=1.0
        Divides the log-likelihoods before Bayes' rule, ``> 0`` (see ``fit_temperature``).
        Read by ``predict_proba`` at every call too (``model.temperature = T``).

    Attributes
    ----------
    classes_ : list of str
        The languages seen by ``fit``, sorted (``["en", "fr"]``).
    log_probs_ : np.ndarray of shape (n_classes, len(alphabet))
        Natural log of the probability of each letter in each language.
    """

    def __init__(self, alphabet: str | Sequence[str] = ALPHABET, smoothing: float = 1.0, prior=None,
                 temperature: float = 1.0):
        self.alphabet = alphabet
        self.smoothing = smoothing
        self.prior = prior
        self.temperature = temperature

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> LanguageDetector:
        """Estimate one smoothed letter distribution per language.

        Parameters
        ----------
        texts : sequence of str
            Training texts (for example one string per chapter).
        labels : sequence of str
            The language of each text, at least two different languages.

        Returns
        -------
        LanguageDetector
            The fitted model itself.
        """
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels must have the same length, got {len(texts)} and {len(labels)}")
        if not self.smoothing > 0:
            raise ValueError(f"smoothing must be > 0 (a letter of probability 0 has an infinite surprise), "
                             f"got {self.smoothing!r}")
        self.classes_ = sorted(set(labels))
        if len(self.classes_) < 2:
            raise ValueError(f"at least two languages are needed, got {self.classes_}")
        letters = list(self.alphabet)
        log_probs = []
        for language in self.classes_:
            text = " ".join(t for t, label in zip(texts, labels) if label == language)
            _, probs = info.char_distribution(text, alphabet=letters, smoothing=self.smoothing)
            log_probs.append(np.log(probs))
        self.log_probs_ = np.array(log_probs)
        self._index = {letter: j for j, letter in enumerate(letters)}
        return self

    def _counts(self, texts: Sequence[str]) -> np.ndarray:
        """Count the letters of the alphabet in each text (after lower-casing): shape (n_texts, n_letters)."""
        if isinstance(texts, str):
            raise TypeError("texts must be a list of strings, not one string (write [text])")
        counts = np.zeros((len(texts), len(self._index)))
        for i, text in enumerate(texts):
            for character in text.lower():
                j = self._index.get(character)
                if j is not None:
                    counts[i, j] += 1
        return counts

    def log_likelihood(self, texts: Sequence[str]) -> np.ndarray:
        """Natural log of P(text | language): the sum of the log-probabilities of the letters of the text.

        Parameters
        ----------
        texts : sequence of str
            The texts to score. A text without any letter of the alphabet scores 0 everywhere.

        Returns
        -------
        np.ndarray of shape (n_texts, n_classes)
            One column per language, in the order of ``classes_``.
        """
        if not hasattr(self, "log_probs_"):
            raise RuntimeError("call fit before log_likelihood")
        return self._counts(texts) @ self.log_probs_.T

    def _prior(self) -> np.ndarray:
        k = len(self.classes_)
        prior = np.full(k, 1.0 / k) if self.prior is None else np.asarray(self.prior, dtype=float)
        if prior.shape != (k,):
            raise ValueError(f"prior must have one value per language ({self.classes_}), got shape {prior.shape}")
        return prior

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Posterior probability of each language for each text: Bayes' rule with the current ``prior``
        and ``temperature`` attributes.

        Returns
        -------
        np.ndarray of shape (n_texts, n_classes)
            Rows summing to 1, columns in the order of ``classes_``.
        """
        return posterior_from_loglik(self.log_likelihood(texts), self._prior(), self.temperature)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """The most probable language of each text (the first one, in ``classes_`` order, in case of a tie)."""
        return np.array(self.classes_, dtype=object)[self.predict_proba(texts).argmax(axis=1)]


def fit_temperature(model: LanguageDetector, texts: Sequence[str], labels: Sequence[str], *, lr: float = 1.0,
                    n_steps: int = 200, tol: float = 1e-6) -> float:
    """Fit the temperature that minimizes the log loss of the model on (texts, labels).

    The log-likelihoods are computed once. The parameter is ``log T`` (so that ``T`` stays
    positive), moved by ``mylearn.calculus.gradient_descent`` with the slope given by
    ``mylearn.calculus.numerical_derivative``; the loss is ``mylearn.info.log_loss`` (nats).
    The model itself is not modified: set ``model.temperature`` to the result.

    Parameters
    ----------
    model : LanguageDetector
        A fitted model.
    texts, labels : sequences
        Validation texts and their languages (never the test set).
    lr, n_steps, tol : float, int, float
        Learning rate, maximum number of steps and stopping tolerance of the descent.

    Returns
    -------
    float
        The fitted temperature ``T > 0``.
    """
    log_lik = model.log_likelihood(texts)
    y = np.array([model.classes_.index(label) for label in labels])
    prior = model._prior()

    def loss(log_t: float) -> float:
        probs = posterior_from_loglik(log_lik, prior, float(np.exp(log_t)))
        return info.log_loss(y, probs if probs.shape[1] > 2 else probs[:, 1])

    def grad(point: np.ndarray) -> np.ndarray:
        return np.array([calculus.numerical_derivative(loss, float(point[0]), h=1e-4)])

    final, _ = calculus.gradient_descent(grad, [0.0], lr=lr, n_steps=n_steps, tol=tol)
    return float(np.exp(final[0]))


def evaluate(model: LanguageDetector, texts: Sequence[str], labels: Sequence[str], positive: str = "fr") -> dict:
    """Measure a binary detector: accuracy, precision, recall, F1, ROC-AUC, Brier score and log loss.

    ``positive`` is the language counted as the positive class; the score of the ROC curve and
    of the probabilistic measures is the probability of that language.

    Returns
    -------
    dict
        ``{"accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"}`` (log loss in nats).
    """
    labels = np.asarray(labels, dtype=object)
    predicted = model.predict(texts)
    score = model.predict_proba(texts)[:, model.classes_.index(positive)]
    y = (labels == positive).astype(int)
    return {
        "accuracy": metrics.accuracy(labels, predicted),
        "precision": metrics.precision(labels, predicted, pos_label=positive),
        "recall": metrics.recall(labels, predicted, pos_label=positive),
        "f1": metrics.f1(labels, predicted, pos_label=positive),
        "roc_auc": metrics.roc_auc(labels, score, pos_label=positive),
        "brier": metrics.brier_score(y, score),
        "log_loss": info.log_loss(y, score),
    }
