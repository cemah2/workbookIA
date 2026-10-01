"""Tests of my language detector (mini-project of part I).

    python -m pytest mon_travail/projets/partie_1_detecteur_langue -q

Two tests are given as examples. Write at least four more (step MP1.7 of the notebook
checks that there are six or more), each with a clear name and a message that says what
was expected. Ideas, from the deliverables of the brief:
- the log-likelihood of a short text is the sum of the log-probabilities of its letters
  (compute it by hand with model.log_probs_), and the other characters are ignored;
- smoothing: a letter that never occurs in a language still gets a probability > 0;
- reproducibility: two fits on the same texts give the same log_probs_, and the same
  generator seed gives the same excerpts (data.sample_excerpts);
- the prior: with prior [0.1, 0.9], a text without any letter gets the probabilities of the prior;
- the temperature: with the default (uniform) prior, T > 1 moves the probabilities toward 0.5
  without changing the decisions (with another prior, toward the prior);
- a long text (a whole chapter) gives finite probabilities: no underflow;
- invalid inputs: smoothing = 0, a single language, texts and labels of different lengths.
"""

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


def test_obvious_sentences_are_recognized(model):
    english = ["The detective looked at me with a smile.", "I have never seen such a strange case."]
    french = ["Le voyageur arriva à la gare avant midi.", "Il fallait traverser l'océan en quatre-vingts jours."]
    predicted = list(model.predict(english + french))
    assert predicted == ["en", "en", "fr", "fr"], f"expected ['en', 'en', 'fr', 'fr'], got {predicted}"


# TODO MP1.7: your tests (at least four more)
