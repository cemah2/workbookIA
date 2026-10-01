"""Tests of the language detector (reference solution of the mini-project of part I).

    python -m pytest projets/partie_1_detecteur_langue/solution -q

The oracles are NumPy computations written here (sums of log-probabilities, the softmax
form of Bayes' rule) and properties (shapes, sums, reproducibility, effect of the prior and
of the temperature).
"""

import math

import numpy as np
import pytest

import data
import langid


@pytest.fixture(scope="module")
def chapters():
    return data.load_chapters()


@pytest.fixture(scope="module")
def model(chapters):
    texts = chapters["en"][:8] + chapters["fr"][:25]
    labels = ["en"] * 8 + ["fr"] * 25
    return langid.LanguageDetector().fit(texts, labels)


def test_predict_proba_has_one_row_per_text_summing_to_one(model):
    texts = ["Hello there", "Bonjour à tous", "", "1234 !!"]
    proba = model.predict_proba(texts)
    assert proba.shape == (4, 2), f"expected one row per text and one column per language, got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0), f"every row must sum to 1, got {proba.sum(axis=1)}"
    assert ((proba >= 0) & (proba <= 1)).all()


def test_obvious_sentences_are_recognized(model):
    english = ["The detective looked at me with a smile.", "I have never seen such a strange case."]
    french = ["Le voyageur arriva à la gare avant midi.", "Il fallait traverser l'océan en quatre-vingts jours."]
    predicted = list(model.predict(english + french))
    assert predicted == ["en", "en", "fr", "fr"], f"expected ['en', 'en', 'fr', 'fr'], got {predicted}"


def test_log_likelihood_is_the_sum_of_the_letter_log_probabilities(model):
    text = "Été, 2 x!"                               # letters é, t, é, x; the rest is ignored
    letters = list(model.alphabet)
    expected = [sum(model.log_probs_[k][letters.index(c)] for c in "étéx") for k in range(2)]
    assert np.allclose(model.log_likelihood([text])[0], expected), "expected the sum of log p(letter) of é, t, é, x"


def test_smoothing_gives_every_letter_a_probability(chapters):
    tiny = langid.LanguageDetector(smoothing=1.0).fit(["aaa", "bbb"], ["en", "fr"])
    assert np.isfinite(tiny.log_probs_).all(), "with smoothing > 0, no letter may have a probability of 0"
    probs = np.exp(tiny.log_probs_)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, tiny.alphabet.index("z")] == pytest.approx(1 / (3 + 42)), "z: (0 + 1) / (3 + 42 letters)"


def test_predict_proba_matches_the_softmax_form_of_bayes_rule(model):
    texts = ["the cat", "le chat", "abc"]
    log_lik = model.log_likelihood(texts)
    for prior in ([0.5, 0.5], [0.2, 0.8]):
        model.prior = prior
        z = log_lik + np.log(prior)
        expected = np.exp(z - z.max(axis=1, keepdims=True))
        expected /= expected.sum(axis=1, keepdims=True)
        assert np.allclose(model.predict_proba(texts), expected), f"prior {prior}: expected softmax(log-likelihood + log prior)"
    model.prior = None


def test_long_texts_do_not_underflow(model, chapters):
    text = chapters["fr"][30][:5000]                 # 5 000 characters: the likelihood itself is far below 1e-308
    proba = model.predict_proba([text])
    assert np.isfinite(proba).all() and proba[0, 1] > 0.999


def test_prior_decides_a_text_without_letters(model):
    model.prior = [0.1, 0.9]
    try:
        assert list(model.predict(["", "42 !"])) == ["fr", "fr"], "with no letter, the posterior is the prior"
        assert np.allclose(model.predict_proba([""]), [[0.1, 0.9]])
    finally:
        model.prior = None


def test_temperature_moves_the_probabilities_toward_the_prior(model):
    texts = ["the old house", "la vieille maison"]
    sharp = model.predict_proba(texts)
    model.temperature = 4.0
    try:
        soft = model.predict_proba(texts)
    finally:
        model.temperature = 1.0
    assert (np.abs(soft - 0.5) < np.abs(sharp - 0.5)).all(), "with the uniform prior, T > 1 must move every probability toward 0.5"
    assert (np.argmax(soft, axis=1) == np.argmax(sharp, axis=1)).all(), "with the uniform prior, the temperature never changes the decision"


def test_fit_and_excerpts_are_reproducible(chapters, model):
    again = langid.LanguageDetector().fit(chapters["en"][:8] + chapters["fr"][:25], ["en"] * 8 + ["fr"] * 25)
    assert np.array_equal(again.log_probs_, model.log_probs_)
    first = data.sample_excerpts(chapters["fr"][31:], 20, 5, np.random.default_rng(0))
    second = data.sample_excerpts(chapters["fr"][31:], 20, 5, np.random.default_rng(0))
    assert first == second and all(len(e) == 20 for e in first)


def test_fit_temperature_finds_the_minimum_of_the_log_loss(model, chapters):
    rng = np.random.default_rng(1)
    texts = data.sample_excerpts(chapters["en"][8:10], 10, 60, rng) + data.sample_excerpts(chapters["fr"][25:31], 10, 60, rng)
    labels = ["en"] * 60 + ["fr"] * 60
    t = langid.fit_temperature(model, texts, labels)
    log_lik = model.log_likelihood(texts)
    y = np.array([0] * 60 + [1] * 60)

    def nll(temperature):
        z = log_lik / temperature
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p = np.clip(p[:, 1] / p.sum(axis=1), np.finfo(float).eps, 1 - np.finfo(float).eps)   # as log_loss does
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    grid = np.exp(np.linspace(-2, 2, 401))
    best = grid[np.argmin([nll(g) for g in grid])]
    assert t > 0 and abs(math.log(t) - math.log(best)) < 0.02, f"expected T close to {best:.3f}, got {t:.3f}"


@pytest.mark.parametrize("kwargs, texts, labels, error", [
    ({"smoothing": 0.0}, ["aa", "bb"], ["en", "fr"], ValueError),
    ({}, ["aa", "bb"], ["en", "en"], ValueError),
    ({}, ["aa"], ["en", "fr"], ValueError),
], ids=["no-smoothing", "one-language", "lengths-differ"])
def test_fit_rejects_invalid_inputs(kwargs, texts, labels, error):
    with pytest.raises(error):
        langid.LanguageDetector(**kwargs).fit(texts, labels)


def test_a_single_string_is_refused(model):
    with pytest.raises(TypeError):
        model.predict("Bonjour")


def test_evaluate_matches_direct_counts(model, chapters):
    rng = np.random.default_rng(2)
    texts = data.sample_excerpts(chapters["en"][10:], 20, 50, rng) + data.sample_excerpts(chapters["fr"][31:], 20, 50, rng)
    labels = np.array(["en"] * 50 + ["fr"] * 50, dtype=object)
    scores = langid.evaluate(model, texts, labels)
    predicted = model.predict(texts)
    tp = np.sum((predicted == "fr") & (labels == "fr"))
    fp = np.sum((predicted == "fr") & (labels == "en"))
    assert scores["accuracy"] == pytest.approx(np.mean(predicted == labels))
    assert scores["precision"] == pytest.approx(tp / (tp + fp))
    assert 0.5 < scores["roc_auc"] <= 1.0 and 0 <= scores["brier"] <= 1
