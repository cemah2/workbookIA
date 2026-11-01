"""Code cells of the chapter B2 notebook, phase « embeddings » (parts D and E, B2.22 to B2.31), shared by build_chB2.py
and history_chB2.py (used by Claude).

Same conventions as chB2_code.py: TOOLS_D (both notebooks), GIVEN_* (both), TODO_* (exercise notebook), SOLUTION_*
(solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
"""

TOOLS_D = r'''# Tools of the phase « embeddings » (parts D and E)
from wb.nbtools import pair

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # a computer without PyTorch (Mac Intel): parts D and E are done on Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : B2.25 à B2.29 se font sur Colab (00_setup/INSTALL_LOCAL.md) ; les "
          "tests de ta librairie tournent sans lui : python -m pytest tests/test_chb2_embeddings.py -q")

    def _skip_torch_cells(lines):
        """IPython input transformer: without PyTorch, a cell that uses it prints ⏳ instead of stopping "Run all"."""
        if re.search(r"\b(torch|torchvision)\b|(?<![\w.])(nn|F)\.", "".join(lines)):
            return ['print("⏳ PyTorch absent sur cet ordinateur : cette cellule se fait sur Colab.")\n']
        return lines

    try:
        _shell = get_ipython()
    except NameError:                     # not in a notebook
        _shell = None
    if _shell is not None and all(getattr(f, "__name__", "") != "_skip_torch_cells" for f in _shell.input_transformers_cleanup):
        _shell.input_transformers_cleanup.append(_skip_torch_cells)

TEST_EMB = "tests/test_chb2_embeddings.py"     # the tests of your library embeddings.py
WORDS_H = wbtext.words("holmes")               # the 105 863 words of Holmes, [a-z]+ in lower case (the map of the fiche)
WORDS_V = wbtext.words("verne")                # the 72 367 words of Verne, accents kept


def nearest_words(E, itos, word, k=10, stoi=None):
    """The k nearest other words of `word` by cosine (rows of E of norm 1), ties to the smaller id: [(word, cos)]."""
    stoi = stoi or {w: i for i, w in enumerate(itos)}
    sims = E @ E[stoi[word]]
    order = [i for i in np.argsort(-sims, kind="stable") if i != stoi[word]][:k]
    return [(itos[i], float(sims[i])) for i in order]


print(f"Holmes : {len(WORDS_H):_} mots ({len(set(WORDS_H)):_} formes) ; Verne : {len(WORDS_V):_} mots "
      f"({len(set(WORDS_V)):_} formes).".replace("_", " "))'''

# --- Part D -----------------------------------------------------------------------------------------------------------
CHECK_22 = r'''with checked("B2.22"):
    Em = mylearn.embeddings
    ids_22 = np.array([0, 0, 0, 1], dtype=np.int32)              # int32, as the tests
    arg_22 = ids_22.copy()
    C_22 = Em.cooccurrence_matrix(arg_22, 2, window=1)
    if returned("B2.22", "cooccurrence_matrix", C_22):
        ok_22 = (isinstance(C_22, np.ndarray) and C_22.dtype.kind == "f" and C_22.shape == (2, 2)
                 and np.array_equal(C_22, [[4.0, 1.0], [1.0, 0.0]]))
        verdict("B2.22", ok_22, "[0, 0, 0, 1], fenêtre 1 : C[0, 0] = 4, chaque répétition d'une paire compte ; une matrice "
                "float (2, 2).",
                f"cooccurrence_matrix([0, 0, 0, 1], 2, window=1) doit rendre la matrice float [[4, 1], [1, 0]] (une paire "
                f"répétée s'ajoute à chaque fois) ; reçu {np.asarray(C_22).tolist()} de type {getattr(C_22, 'dtype', type(C_22).__name__)}.")
        verdict("B2.22", np.array_equal(arg_22, ids_22), "les ids reçus n'ont pas changé.", "cooccurrence_matrix a modifié les ids reçus.")
    right_22 = Em.cooccurrence_matrix([2, 0, 1], 3, window=2, weighting="harmonic", symmetric=False)
    same("B2.22", "[2, 0, 1], fenêtre 2, harmonique, contextes de droite seulement", right_22,
         [[0, 1, 0], [0, 0, 0], [1, 0.5, 0]], rtol=1e-6, atol=1e-6)
    # Holmes: the 2 000 most frequent words (the others removed here), against a count by scipy.sparse
    import scipy.sparse as sp
    stoi_22, _ = wbtext.top_vocab(WORDS_H, 2000)
    h_22 = wbtext.word_ids(WORDS_H[:30000], stoi_22)
    h_22 = h_22[h_22 >= 0]
    pairs_22 = [(h_22[:-d], h_22[d:], np.full(len(h_22) - d, 1.0 / d)) for d in (1, 2, 3)]
    rows_22 = np.concatenate([a for a, b, w in pairs_22] + [b for a, b, w in pairs_22])
    cols_22 = np.concatenate([b for a, b, w in pairs_22] + [a for a, b, w in pairs_22])
    want_22 = sp.coo_matrix((np.concatenate([w for *_, w in pairs_22] * 2), (rows_22, cols_22)), shape=(2000, 2000)).toarray()
    same("B2.22", f"Holmes ({len(h_22):_} mots, 2 000 formes), fenêtre 3, harmonique : la matrice d'un comptage "
         "par scipy.sparse".replace("_", " "), Em.cooccurrence_matrix(h_22, 2000, window=3, weighting="harmonic"), want_22,
         rtol=1e-6, atol=1e-6)
    for label_22, call_22 in (("des ids 2-D", lambda: Em.cooccurrence_matrix([[0, 1]], 2)),
                              ("window = 0", lambda: Em.cooccurrence_matrix([0, 1], 2, window=0)),
                              ("weighting = 'gaussian'", lambda: Em.cooccurrence_matrix([0, 1], 2, weighting="gaussian")),
                              ("un id égal à vocab_size", lambda: Em.cooccurrence_matrix([0, 2], 2)),
                              ("un id de -1", lambda: Em.cooccurrence_matrix([0, -1], 2))):
        refuses("B2.22", label_22, call_22)

    # ppmi: the matrix chien / chat / os of the fiche (§202.8), a word never seen, an « independent » matrix
    C3_22 = np.array([[0, 4, 1], [4, 0, 1], [1, 1, 2]])          # integers: ppmi must accept them
    arg3_22 = C3_22.copy()
    P_22 = Em.ppmi(arg3_22)
    if returned("B2.22", "ppmi", P_22):
        verdict("B2.22", isinstance(P_22, np.ndarray) and P_22.shape == (3, 3) and np.isclose(P_22[0, 1], math.log(56 / 25))
                and P_22[0, 2] == 0 and np.isclose(P_22[2, 2], math.log(28 / 16)),
                "PPMI(chien, chat) = log 2,24 ≈ 0,807, PPMI(chien, os) = 0 (PMI négative), PPMI(os, os) ≈ 0,560, comme la fiche.",
                f"sur la matrice chien / chat / os de la fiche, attendu PPMI(chien, chat) ≈ 0,807, PPMI(chien, os) = 0 et "
                f"PPMI(os, os) ≈ 0,560 (logarithme naturel) ; reçu {np.round(np.asarray(P_22, dtype=float), 3).tolist()}.")
        verdict("B2.22", np.array_equal(arg3_22, C3_22), "la matrice reçue n'a pas changé.", "ppmi a modifié la matrice reçue.")
    P75_22 = Em.ppmi(C3_22, alpha=0.75)
    verdict("B2.22", np.isclose(P75_22[0, 1], 0.8230, atol=5e-4), f"alpha = 0,75 : PPMI(chien, chat) = {fr(P75_22[0, 1])}, "
            "comme la fiche.", f"alpha = 0,75 : attendu PPMI(chien, chat) ≈ 0,823 (fiche §202.8) ; reçu {fr(P75_22[0, 1])} : "
            "le lissage ne porte que sur les colonnes, puis se renormalise.")
    with warnings.catch_warnings(record=True) as caught_22, np.errstate(all="warn"):
        warnings.simplefilter("always")
        Z_22 = Em.ppmi(np.array([[0.0, 2, 1, 0], [2, 0, 3, 0], [1, 3, 0, 0], [0, 0, 0, 0]]))
    verdict("B2.22", np.all(np.isfinite(Z_22)) and np.all(Z_22[3] == 0) and np.all(Z_22[:, 3] == 0),
            "un mot jamais vu (ligne et colonne nulles) : des 0, sans NaN.",
            "une ligne et une colonne nulles doivent donner des 0, jamais NaN : mets 0 là où C == 0.")
    if any(issubclass(w.category, RuntimeWarning) for w in caught_22):
        print("💡 Ex B2.22 : ppmi émet un avertissement de NumPy (log(0) ou 0 / 0) : calcule le logarithme sur les seules "
              "cases non nulles, ou dans np.errstate(divide='ignore', invalid='ignore').")
    indep_22 = Em.ppmi(np.outer([1.0, 2, 3], [4.0, 1, 5]))
    verdict("B2.22", np.allclose(indep_22, 0, atol=1e-9), "une matrice « indépendante » np.outer(u, v) : PPMI nulle partout.",
            f"np.outer(u, v) donne p(w, c) = p(w) p(c) : attendu une PPMI nulle ; reçu le maximum {float(np.max(indep_22)):.3g}.")
    for label_22, call_22 in (("une matrice 1-D", lambda: Em.ppmi([1.0, 2.0])),
                              ("une entrée négative", lambda: Em.ppmi([[1.0, -1.0], [0.0, 2.0]])),
                              ("une somme nulle", lambda: Em.ppmi(np.zeros((2, 2))))):
        refuses("B2.22", label_22, call_22)'''

CHECK_23 = r'''with checked("B2.23"):
    Em = mylearn.embeddings
    rng_23 = np.random.default_rng(23)
    A_23, B_23 = rng_23.normal(size=(5, 3)), rng_23.normal(size=(4, 3))
    A_23[2] = 0.0                                                   # a row of zeros
    argA_23, argB_23 = A_23.copy(), B_23.copy()
    S_23 = Em.cosine_similarity_matrix(argA_23, argB_23)
    from sklearn.metrics.pairwise import cosine_similarity
    if returned("B2.23", "cosine_similarity_matrix", S_23):
        same("B2.23", "cosine_similarity_matrix(A, B), A (5, 3) avec une ligne nulle, B (4, 3), contre scikit-learn",
             S_23, cosine_similarity(A_23, B_23), rtol=1e-6, atol=1e-9)
        verdict("B2.23", np.array_equal(argA_23, A_23) and np.array_equal(argB_23, B_23), "A et B n'ont pas changé.",
                "cosine_similarity_matrix a modifié A ou B.")
    same("B2.23", "cosine_similarity_matrix(A) (B = None : B = A)", Em.cosine_similarity_matrix(A_23), cosine_similarity(A_23),
         rtol=1e-6, atol=1e-9)
    same("B2.23", "[[3, 4]] contre [[6, 8]] avec eps = 0,5 (ajouté à chaque norme)",
         Em.cosine_similarity_matrix([[3, 4]], [[6, 8]], eps=0.5), [[50 / (5.5 * 10.5)]])
    refuses("B2.23", "A 1-D", lambda: Em.cosine_similarity_matrix([1.0, 2.0]))
    refuses("B2.23", "A et B de 3 et 2 colonnes", lambda: Em.cosine_similarity_matrix(A_23, B_23[:, :2]))

    E_23 = rng_23.normal(size=(120, 3))                             # 120 rows, 40 of them equal to the query
    copies_23 = np.sort(rng_23.choice(120, 40, replace=False))
    E_23[copies_23] = [1.0, 0.0, 0.0]
    out_23 = Em.most_similar(E_23, [1.0, 0.0, 0.0], k=5)
    if returned("B2.23", "most_similar", out_23):
        idx_23, sims_23 = pair("B2.23", "most_similar", out_23, names=("indices", "similarities"))
        got_23 = np.asarray(idx_23).tolist()
        verdict("B2.23", isinstance(idx_23, np.ndarray) and isinstance(sims_23, np.ndarray)
                and np.issubdtype(idx_23.dtype, np.integer) and got_23 == copies_23[:5].tolist()
                and np.allclose(sims_23, 1),
                f"40 lignes à égalité parmi 120 : les 5 plus petits indices, {copies_23[:5].tolist()} (un tri stable), en "
                "tableaux NumPy.",
                f"40 lignes égales à la requête parmi 120 : attendu les indices {copies_23[:5].tolist()} (égalités : le plus "
                f"petit indice, un tri stable), deux tableaux NumPy ; reçu {got_23}.")
    idx_23, _ = Em.most_similar(E_23, [1.0, 0.0, 0.0], k=3, exclude=tuple(copies_23[:2].tolist()))
    want_23 = copies_23[2:5].tolist()
    verdict("B2.23", np.asarray(idx_23).tolist() == want_23, f"exclude retiré avant le top-k : {want_23}.",
            f"avec exclude={tuple(copies_23[:2].tolist())}, attendu {want_23} (on retire avant de garder les k premiers, "
            f"égalités au plus petit indice) ; reçu {np.asarray(idx_23).tolist()}.")
    few_23 = Em.most_similar(np.eye(3), [1.0, 1.0, 0.0], k=10, exclude=(1,))
    verdict("B2.23", np.asarray(few_23[0]).tolist() == [0, 2], "moins de k lignes restantes : toutes, ici [0, 2].",
            f"3 lignes, une exclue, k = 10 : attendu les deux restantes [0, 2] ; reçu {np.asarray(few_23[0]).tolist()}.")
    none_23 = Em.most_similar(np.eye(2), [1.0, 0.0], k=2, exclude=(0, 1))
    verdict("B2.23", none_23[0].shape == (0,) and none_23[1].shape == (0,), "tout exclu : deux tableaux de forme (0,).",
            f"tout exclu : attendu deux tableaux vides de forme (0,) ; reçu {none_23!r}.")
    zero_23 = Em.most_similar(np.eye(3), [0.0, 0.0, 0.0], k=2)
    verdict("B2.23", np.asarray(zero_23[0]).tolist() == [0, 1] and np.allclose(zero_23[1], 0),
            "une requête nulle : des similarités nulles et les k premiers indices.",
            f"une requête nulle doit donner des similarités nulles et les indices [0, 1] ; reçu {np.asarray(zero_23[0]).tolist()}.")
    refuses("B2.23", "k = 0", lambda: Em.most_similar(np.eye(3), [1.0, 0, 0], k=0))
    refuses("B2.23", "une requête de longueur 2 pour des lignes de 3", lambda: Em.most_similar(np.eye(3), [1.0, 0], k=1))
    refuses("B2.23", "une requête de forme (1, 3)", lambda: Em.most_similar(np.eye(3), [[1.0, 0, 0]], k=1))'''

TODO_24 = r'''from sklearn.decomposition import TruncatedSVD


def ppmi_svd_24(words, n_words=4000, window=2, dim=100, alpha=0.75):
    """(E, stoi, itos): PPMI + SVD word vectors (rows of norm 1) of the n_words most frequent words of `words`, made
    with your cooccurrence_matrix and ppmi; a word outside the vocabulary keeps its place, but its pairs do not count."""
    raise NotImplementedError("ppmi_svd_24")


def neighbours_24(E, stoi, itos, word, k=8):
    """The k nearest other words of `word` by cosine, with your most_similar: [(word, cosine), ...]."""
    raise NotImplementedError("neighbours_24")'''

SOLUTION_24 = r'''from sklearn.decomposition import TruncatedSVD


def ppmi_svd_24(words, n_words=4000, window=2, dim=100, alpha=0.75):
    """(E, stoi, itos): PPMI + SVD word vectors (rows of norm 1) of the n_words most frequent words of `words`, made
    with your cooccurrence_matrix and ppmi; a word outside the vocabulary keeps its place, but its pairs do not count."""
    stoi, itos = wbtext.top_vocab(words, n_words)
    v = len(itos)
    ids = wbtext.word_ids(words, stoi)                        # -1 for a word outside the vocabulary
    ids = np.where(ids < 0, v, ids)                           # one extra id for all of them: they keep their place...
    C = mylearn.embeddings.cooccurrence_matrix(ids, v + 1, window=window)[:v, :v]   # ...and their row and column go
    M = mylearn.embeddings.ppmi(C, alpha=alpha)
    svd = TruncatedSVD(n_components=dim, algorithm="arpack", random_state=0)
    E = svd.fit_transform(M) / np.sqrt(svd.singular_values_)  # U_d S_d / S_d^(1/2) = U_d S_d^(1/2)
    return E / np.linalg.norm(E, axis=1, keepdims=True), stoi, itos


def neighbours_24(E, stoi, itos, word, k=8):
    """The k nearest other words of `word` by cosine, with your most_similar: [(word, cosine), ...]."""
    idx, sims = mylearn.embeddings.most_similar(E, E[stoi[word]], k=k, exclude=(stoi[word],))
    return [(itos[i], float(s)) for i, s in zip(idx, sims)]'''

CHECK_24 = r'''WORDS_24 = ("holmes", "watson", "letter")
ok_24 = False
with checked("B2.24"):
    start_24 = time.perf_counter()
    out_24 = ppmi_svd_24(WORDS_H)
    seconds_24 = time.perf_counter() - start_24
    if not returned("B2.24", "ppmi_svd_24", out_24):
        raise StopCheck
    E_24, stoi_24, itos_24 = pair("B2.24", "ppmi_svd_24", out_24, names=("E", "stoi", "itos"))
    E_24 = np.asarray(E_24, dtype=float)
    verdict("B2.24", E_24.shape == (4000, 100) and np.allclose(np.linalg.norm(E_24, axis=1), 1, atol=1e-6)
            and list(itos_24) == wbtext.top_vocab(WORDS_H, 4000)[1],
            f"E : 4 000 mots × 100 dimensions, lignes de norme 1, le vocabulaire de top_vocab ({fr(seconds_24, 1)} s).",
            f"attendu E de forme (4000, 100) aux lignes de norme 1 et itos = top_vocab(words, 4000)[1] ; reçu E {E_24.shape}.")
    R_24, rstoi_24, _ = wbtext.reference_ppmi_svd(WORDS_H)     # the same pipeline, counted by scipy.sparse
    rows_24 = [stoi_24[w] for w in WORDS_24]
    gap_24 = float(np.abs(E_24[rows_24] @ E_24.T - R_24[rows_24] @ R_24.T).max())
    verdict("B2.24", gap_24 < 1e-3, "les cosinus de « holmes », « watson » et « letter » avec les 4 000 mots sont ceux de "
            f"la référence (écart maximal {gap_24:.1e}).",
            f"les cosinus de « holmes », « watson » et « letter » s'écartent de la référence jusqu'à {gap_24:.2f} : as-tu "
            "gardé la place des mots hors vocabulaire (sans les retirer du texte), lissé les seules colonnes (alpha), pris "
            "U S^(1/2) (pas U S), normalisé les lignes, et algorithm='arpack' ?")
    nb_24 = {w: neighbours_24(E_24, stoi_24, itos_24, w) for w in WORDS_24}
    if not returned("B2.24", "neighbours_24", nb_24["holmes"]):
        raise StopCheck
    want_24 = {w: nearest_words(E_24, list(itos_24), w, 8, dict(stoi_24)) for w in WORDS_24}
    verdict("B2.24", all(len(nb_24[w]) == 8 and [x for x, _ in nb_24[w]] == [x for x, _ in want_24[w]] for w in WORDS_24),
            "neighbours_24 : les 8 voisins de chaque mot, sans le mot lui-même, par cosinus décroissant.",
            "neighbours_24 doit rendre les 8 plus proches autres mots, (mot, cosinus) par cosinus décroissant, sans le mot.")
    for w in WORDS_24:
        print(f"{w:>7} : " + ", ".join(f"{x} {fr(s, 2)}" for x, s in nb_24[w]))
    ok_24 = True'''

GIVEN_25 = r'''# The data of word2vec: the words of Verne seen at least 5 times (the rarer ones are removed from the text, as word2vec
# does), their ids by decreasing frequency, and their counts
MIN_COUNT_25 = 5
N_VOCAB_25 = sum(1 for n in Counter(WORDS_V).values() if n >= MIN_COUNT_25)
STOI_25, ITOS_25 = wbtext.top_vocab(WORDS_V, N_VOCAB_25)       # id 0 = « de », the most frequent
IDS_25 = wbtext.word_ids(WORDS_V, STOI_25)
IDS_25 = IDS_25[IDS_25 >= 0]                                   # the text without its rare words (int64)
COUNTS_25 = np.bincount(IDS_25, minlength=N_VOCAB_25).astype(float)
EPOCHS_25 = 10 if FAST_MODE else 30
# pairs of related words, to measure the quality of the vectors: the rank of the second among the neighbours of the first
PAIRS_25 = [("deux", "trois"), ("quatre", "cinq"), ("vingt", "cent"), ("heures", "minutes"), ("jours", "heures"),
            ("bombay", "calcutta"), ("yokohama", "hong"), ("train", "wagon"), ("paquebot", "steamer"), ("main", "mains"),
            ("tête", "yeux"), ("dit", "répondit")]
print(f"{N_VOCAB_25:_} mots de vocabulaire, {len(IDS_25):_} mots dans le texte d'entraînement, {EPOCHS_25} epochs.".replace("_", " "))'''

TODO_25 = r'''def keep_probability_25(counts, t=1e-3):
    """For each word of the vocabulary, the probability of KEEPING one occurrence: min(1, sqrt(t / f)), f = count / total
    (a float array of the shape of counts)."""
    raise NotImplementedError("keep_probability_25")


def noise_distribution_25(counts, power=0.75):
    """The law of the negatives: counts ** power, normalised to a sum of 1 (a float array)."""
    raise NotImplementedError("noise_distribution_25")


def skipgram_pairs_25(ids, window=2):
    """(centers, contexts): two int64 arrays of the same length, one entry per (position i, position j) of the sequence
    with 1 <= |i - j| <= window (the word ids[i] and its context ids[j])."""
    raise NotImplementedError("skipgram_pairs_25")


class SkipGram25(nn.Module):
    """Skip-gram with negative sampling: the table `center` (the vectors v_w) and the table `context` (the u_c)."""

    def __init__(self, vocab_size, dim):
        super().__init__()
        raise NotImplementedError("SkipGram25.__init__")

    def forward(self, centers, contexts, negatives):
        """The mean over the batch of the loss of the fiche: centers and contexts (B,), negatives (B, K) -> a scalar."""
        raise NotImplementedError("SkipGram25.forward")


def train_word2vec_25(ids, counts, dim=100, window=2, negatives=5, t=1e-3, epochs=EPOCHS_25, batch_size=512, lr=0.01,
                      seed=0):
    """(model, losses): a SkipGram25 trained by Adam on the text `ids`, and the mean loss of each epoch."""
    raise NotImplementedError("train_word2vec_25")'''

SOLUTION_25 = r'''def keep_probability_25(counts, t=1e-3):
    """For each word of the vocabulary, the probability of KEEPING one occurrence: min(1, sqrt(t / f)), f = count / total
    (a float array of the shape of counts)."""
    counts = np.asarray(counts, dtype=float)
    f = counts / counts.sum()
    with np.errstate(divide="ignore"):                       # a count of 0: f = 0, kept with probability 1
        return np.minimum(1.0, np.sqrt(t / f))


def noise_distribution_25(counts, power=0.75):
    """The law of the negatives: counts ** power, normalised to a sum of 1 (a float array)."""
    p = np.asarray(counts, dtype=float) ** power
    return p / p.sum()


def skipgram_pairs_25(ids, window=2):
    """(centers, contexts): two int64 arrays of the same length, one entry per (position i, position j) of the sequence
    with 1 <= |i - j| <= window (the word ids[i] and its context ids[j])."""
    ids = np.asarray(ids, dtype=np.int64)
    centers, contexts = [], []
    for d in range(1, window + 1):                           # the context d positions to the right, then to the left
        centers += [ids[:-d], ids[d:]]
        contexts += [ids[d:], ids[:-d]]
    return np.concatenate(centers), np.concatenate(contexts)


class SkipGram25(nn.Module):
    """Skip-gram with negative sampling: the table `center` (the vectors v_w) and the table `context` (the u_c)."""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)   # as word2vec: small v_w,
        nn.init.zeros_(self.context.weight)                            # u_c = 0 at the start

    def forward(self, centers, contexts, negatives):
        """The mean over the batch of the loss of the fiche: centers and contexts (B,), negatives (B, K) -> a scalar."""
        v = self.center(centers)                                       # (B, d)
        pos = (self.context(contexts) * v).sum(dim=-1)                 # (B,): u_c . v_w
        neg = torch.bmm(self.context(negatives), v.unsqueeze(-1)).squeeze(-1)   # (B, K): u_n . v_w
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(dim=-1)).mean()


def train_word2vec_25(ids, counts, dim=100, window=2, negatives=5, t=1e-3, epochs=EPOCHS_25, batch_size=512, lr=0.01,
                      seed=0):
    """(model, losses): a SkipGram25 trained by Adam on the text `ids`, and the mean loss of each epoch."""
    ids = np.asarray(ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = SkipGram25(len(counts), dim)
    keep = keep_probability_25(counts, t)
    noise = torch.as_tensor(noise_distribution_25(counts), dtype=torch.float)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        kept = ids[rng.random(len(ids)) < keep[ids]]                   # a new subsampling at each epoch
        centers, contexts = skipgram_pairs_25(kept, window)
        order = rng.permutation(len(centers))
        centers, contexts = torch.from_numpy(centers[order]), torch.from_numpy(contexts[order])
        total = 0.0
        for start in range(0, len(centers), batch_size):
            c, x = centers[start:start + batch_size], contexts[start:start + batch_size]
            neg = torch.multinomial(noise, len(c) * negatives, replacement=True).view(len(c), negatives)
            loss = model(c, x, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(c)
        losses.append(total / len(centers))
    return model, losses'''

CHECK_25 = r'''ok_25 = False
with checked("B2.25"):
    keep_25 = keep_probability_25(np.array([990.0, 10.0]), t=1e-3)
    if returned("B2.25", "keep_probability_25", keep_25):
        same("B2.25", "keep_probability_25([990, 10], t=1e-3) (f = 0,99 et 0,01)", keep_25,
             [math.sqrt(1e-3 / 0.99), math.sqrt(1e-3 / 0.01)], rtol=1e-6)
    verdict("B2.25", np.allclose(keep_probability_25(np.array([1.0, 1e6]), t=1e-3), [1.0, math.sqrt(1e-3 / (1e6 / (1e6 + 1)))]),
            "un mot très rare est toujours gardé (probabilité 1, pas plus).",
            "la probabilité de garder un mot rare doit être bornée à 1 : min(1, sqrt(t / f)).")
    same("B2.25", "noise_distribution_25([100, 1]) (fiche §202.8)", noise_distribution_25(np.array([100.0, 1.0])),
         np.array([100 ** 0.75, 1.0]) / (100 ** 0.75 + 1), rtol=1e-6)
    pairs_25 = skipgram_pairs_25(np.array([5, 6, 7, 8]), window=2)
    centers_25, contexts_25 = pair("B2.25", "skipgram_pairs_25", pairs_25, names=("centers", "contexts"))
    want_pairs_25 = sorted((a, b) for i, a in enumerate([5, 6, 7, 8]) for j, b in enumerate([5, 6, 7, 8]) if 1 <= abs(i - j) <= 2)
    got_pairs_25 = sorted(zip(np.asarray(centers_25).tolist(), np.asarray(contexts_25).tolist()))
    verdict("B2.25", got_pairs_25 == want_pairs_25 and np.asarray(centers_25).dtype == np.int64,
            f"skipgram_pairs_25([5, 6, 7, 8], window=2) : les {len(want_pairs_25)} paires (mot, contexte), en int64.",
            f"skipgram_pairs_25([5, 6, 7, 8], window=2) : attendu les {len(want_pairs_25)} paires {want_pairs_25} (dans "
            f"n'importe quel ordre, en int64) ; reçu {got_pairs_25}.")
    toy_25 = SkipGram25(10, 4)
    verdict("B2.25", isinstance(getattr(toy_25, "center", None), nn.Embedding) and isinstance(getattr(toy_25, "context", None), nn.Embedding)
            and toy_25.center.weight.shape == (10, 4) and toy_25.context.weight.shape == (10, 4)
            and toy_25.context.weight.detach().abs().max().item() == 0 and toy_25.center.weight.detach().abs().max().item() <= 0.5 / 4,
            "SkipGram25(10, 4) : deux nn.Embedding (10, 4), center dans [-0,125 ; 0,125] et context à 0 au départ.",
            "SkipGram25 doit avoir deux nn.Embedding(vocab_size, dim), center (uniforme dans [-0,5/d ; 0,5/d]) et "
            "context (des zéros), comme word2vec.")
    with torch.no_grad():                                        # fixed weights: the loss of the fiche, by hand
        g_25 = torch.Generator().manual_seed(25)
        toy_25.center.weight.copy_(torch.randn(10, 4, generator=g_25))
        toy_25.context.weight.copy_(torch.randn(10, 4, generator=g_25))
    c_25, x_25, n_25 = torch.tensor([1, 4, 4]), torch.tensor([2, 0, 7]), torch.tensor([[3, 5], [9, 9], [1, 2]])
    loss_25 = toy_25(c_25, x_25, n_25)
    V_25, U_25 = toy_25.center.weight.detach(), toy_25.context.weight.detach()
    want_25 = float(np.mean([-F.logsigmoid(U_25[x] @ V_25[c]).item() - sum(F.logsigmoid(-(U_25[n] @ V_25[c])).item() for n in ns)
                             for c, x, ns in zip(c_25.tolist(), x_25.tolist(), n_25.tolist())]))
    verdict("B2.25", isinstance(loss_25, torch.Tensor) and loss_25.ndim == 0 and math.isclose(loss_25.item(), want_25, rel_tol=1e-5),
            f"forward : la moyenne sur le batch de la loss de la fiche ({fr(want_25)} sur trois paires et K = 2 négatifs).",
            f"forward doit rendre un tenseur scalaire, la moyenne sur le batch de -log σ(u_c·v_w) - Σ log σ(-u_n·v_w) : "
            f"attendu {fr(want_25)} ; reçu {loss_25!r}.")
    received_25, real_pairs_25 = [], skipgram_pairs_25          # a spy: the length of the text that gives the pairs

    def spy_pairs_25(ids, window=2):
        received_25.append(len(ids))
        return real_pairs_25(ids, window)

    try:
        globals()["skipgram_pairs_25"] = spy_pairs_25
        small_25 = train_word2vec_25(IDS_25[:3000], COUNTS_25, dim=8, epochs=2, seed=1)
    finally:
        globals()["skipgram_pairs_25"] = real_pairs_25
    model_s25, losses_s25 = pair("B2.25", "train_word2vec_25", small_25, names=("model", "losses"))
    verdict("B2.25", isinstance(model_s25, SkipGram25) and len(losses_s25) == 2 and all(isinstance(v, float) for v in losses_s25),
            "train_word2vec_25 rend (le modèle SkipGram25, la loss moyenne de chaque epoch, des float).",
            "train_word2vec_25 doit rendre le couple (un SkipGram25, une liste d'une loss float par epoch).")
    kept_25 = float(np.minimum(1, np.sqrt(1e-3 / (COUNTS_25 / COUNTS_25.sum())))[IDS_25[:3000]].sum())
    if len(received_25) == 2:
        verdict("B2.25", all(0.8 * kept_25 < n < 1.2 * kept_25 for n in received_25) and received_25[0] != received_25[1],
                f"sous-échantillonnage : sur 3 000 mots, tes paires viennent de {received_25[0]} puis {received_25[1]} mots "
                f"gardés (≈ {kept_25:.0f} attendus), un nouveau tirage à chaque epoch.",
                f"sur 3 000 mots, tes paires viennent de {received_25} mots, alors que le sous-échantillonnage en garde ≈ "
                f"{kept_25:.0f} par epoch, tirés à nouveau à chaque epoch : relis le contrat de train_word2vec_25.")
    else:
        print("💡 Ex B2.25 : train_word2vec_25 n'appelle pas skipgram_pairs_25 une fois par epoch : la cellule ne peut pas "
              "vérifier ton sous-échantillonnage.")
    ok_25 = True'''

EXPERIMENT_25 = r'''if not globals().get("ok_25"):
    print("⏳ Ex B2.25 : l'entraînement attend que tes fonctions passent la cellule précédente.")
else:
    with checked("B2.25"):
        ok_25 = False
        start_25 = time.perf_counter()
        MODEL_25, LOSSES_25 = train_word2vec_25(IDS_25, COUNTS_25, seed=0)
        seconds_25 = time.perf_counter() - start_25
        E_25 = MODEL_25.center.weight.detach().double().numpy()
        E_25 = E_25 / np.linalg.norm(E_25, axis=1, keepdims=True)      # the word vectors: rows of norm 1
        print(f"{EPOCHS_25} epochs en {fr(seconds_25, 1)} s ; loss moyenne par epoch : "
              + ", ".join(fr(v, 3) for v in LOSSES_25))
        verdict("B2.25", len(LOSSES_25) == EPOCHS_25 and LOSSES_25[-1] < LOSSES_25[0] - 0.3,
                f"la loss baisse de {fr(LOSSES_25[0], 2)} à {fr(LOSSES_25[-1], 2)}.",
                f"la loss devrait baisser nettement d'une epoch à l'autre ({fr(LOSSES_25[0], 2)} → {fr(LOSSES_25[-1], 2)}) : "
                "relis les signes de forward, l'ordre zero_grad / backward / step, et le learning rate.")
        ranks_25 = []
        for a, b in PAIRS_25:
            order = [w for w, _ in nearest_words(E_25, ITOS_25, a, len(ITOS_25) - 1, STOI_25)]
            ranks_25.append(order.index(b) + 1)
        print("Rang du second mot parmi les voisins du premier : "
              + ", ".join(f"{a} → {b} {r}" for (a, b), r in zip(PAIRS_25, ranks_25)))
        median_25 = float(np.median(ranks_25))
        verdict("B2.25", median_25 <= 100, f"rang médian {median_25:g} sur {len(ITOS_25) - 1} voisins possibles (au hasard : "
                f"≈ {len(ITOS_25) // 2}) : tes vecteurs rapprochent des mots de même sens.",
                f"rang médian {median_25:g} (au hasard : ≈ {len(ITOS_25) // 2}) : tes vecteurs n'ont pas appris grand-chose. "
                "Relis le sous-échantillonnage, la loi des négatifs et forward.")
        ok_25 = median_25 <= 100'''

TODO_26 = r'''fogg_26 = ...         # a) the TWO nearest neighbours of « fogg »: "A" two other characters (passepartout, fix, aouda…),
                      #    "B" « phileas » and « mr », "C" two verbs of speech (dit, répondit…), "D" two other words
characters_26 = ...   # b) True or False: at least 5 of the 10 nearest neighbours of « passepartout » are names of
                      #    characters of the novel (CHARACTERS_26)'''

GIVEN_26 = r'''# The names of the characters of the novel (titles such as « mr » are not names)
CHARACTERS_26 = {"fogg", "phileas", "passepartout", "fix", "aouda", "proctor", "stamp", "cromarty", "francis", "obadiah",
                 "speedy", "batulcar", "stuart", "flanagan", "sullivan", "fallentin", "ralph", "gauthier", "andrew", "john",
                 "mudge", "bunsby"}'''

EXPERIMENT_26 = r'''if not globals().get("ok_25"):
    print("⏳ Ex B2.26 : l'expérience attend ton word2vec entraîné en B2.25 (les vecteurs E_25) ; tes prédictions seront "
          "vérifiées ensuite.")
else:
    neighbours_26 = {w: nearest_words(E_25, ITOS_25, w, 10, STOI_25) for w in ("fogg", "passepartout")}
    for w, nb in neighbours_26.items():
        print(f"{w} ({int(COUNTS_25[STOI_25[w]])} fois) : " + ", ".join(f"{x} {fr(s, 2)}" for x, s in nb))
    top2_26 = {x for x, _ in neighbours_26["fogg"][:2]}
    names_26 = sum(x in CHARACTERS_26 for x, _ in neighbours_26["passepartout"])
    print(f"Les deux premiers voisins de « fogg » : {sorted(top2_26)} ; noms de personnages parmi les 10 voisins de "
          f"« passepartout » : {names_26}.")
    print("Fréquence (rang) des voisins de « passepartout » : "
          + ", ".join(f"{x} {STOI_25[x]}" for x, _ in neighbours_26["passepartout"]))
    wb.check("B2.26a", fogg_26)
    wb.check("B2.26b", characters_26)'''

GIVEN_27 = r'''# Five groups of words of Verne, for the third map
GROUPS_27 = {"nombres": "deux trois quatre cinq six sept huit neuf dix vingt cent mille".split(),
             "temps": "heures jours minutes heure jour".split(),
             "lieux": "bombay calcutta londres yokohama hong kong san francisco suez liverpool".split(),
             "transports": "mer paquebot train wagon locomotive traîneau steamer".split(),
             "corps": "mains main tête yeux bras pied pieds".split()}


def maps_27(E, counts, groups, seed=0):
    """The three maps of the figure: PCA of all the words, UMAP of all the words (None without umap-learn), PCA of the
    words of the groups."""
    from sklearn.decomposition import PCA
    pca_all = PCA(2, svd_solver="full").fit_transform(E)
    try:
        import umap
        umap_all = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed, n_jobs=1).fit_transform(E)
    except ImportError:
        umap_all = None
        print("⏳ Ex B2.27 : umap-learn n'est pas installé ici : la carte UMAP est « à valider sur Colab ».")
    words = [w for ws in groups.values() for w in ws]
    pca_groups = PCA(2, svd_solver="full").fit_transform(E[[STOI_25[w] for w in words]])
    return pca_all, umap_all, pca_groups, words'''

EXPERIMENT_27 = r'''if not globals().get("ok_25"):
    print("⏳ Ex B2.27 : les cartes attendent ton word2vec entraîné en B2.25 (les vecteurs E_25).")
else:
    start_27 = time.perf_counter()
    PCA_ALL_27, UMAP_ALL_27, PCA_GROUPS_27, GROUP_WORDS_27 = maps_27(E_25, COUNTS_25, GROUPS_27)
    logf_27 = np.log10(COUNTS_25)
    fig_27, axes_27 = plt.subplots(1, 3, figsize=(16, 5))
    for ax, P, title in ((axes_27[0], PCA_ALL_27, "(1) PCA des 1 590 mots"), (axes_27[1], UMAP_ALL_27, "(2) UMAP des 1 590 mots")):
        if P is None:
            ax.set_axis_off()
            continue
        sc = ax.scatter(P[:, 0], P[:, 1], c=logf_27, cmap="viridis", s=8, alpha=0.8)
        ax.set_title(title)
    fig_27.colorbar(sc, ax=axes_27[:2].tolist(), label="log10 du nombre d'occurrences (jaune : les plus fréquents)")
    colors_27 = plt.get_cmap("tab10")
    k_27 = 0
    for g, (name, ws) in enumerate(GROUPS_27.items()):
        pts = PCA_GROUPS_27[k_27:k_27 + len(ws)]
        axes_27[2].scatter(pts[:, 0], pts[:, 1], color=colors_27(g), label=name, s=30)
        for (x, y), w in zip(pts, ws):
            axes_27[2].annotate(w, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        k_27 += len(ws)
    axes_27[2].legend(fontsize=8)
    axes_27[2].set_title("(3) PCA des mots de cinq groupes")
    print(f"Cartes calculées en {fr(time.perf_counter() - start_27, 1)} s.")
    plt.show()'''

TODO_27 = r'''pca_axis_27 = ...         # a) True or False: on map (1), the colour (the frequency) changes steadily along the
                          #    horizontal axis, the frequent words on one side and the rare ones on the other
umap_frequent_27 = ...    # b) True or False: on map (2), the most frequent words (yellow) gather in one region (or a
                          #    few), instead of being spread all over the map'''

CHECK_27 = r'''with wb.attempt("B2.27"):
    wb.check("B2.27a", pca_axis_27)
    wb.check("B2.27b", umap_frequent_27)
if globals().get("ok_25") and filled(pca_axis_27, umap_frequent_27) and "PCA_ALL_27" in globals():
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors
    rho_27 = [spearmanr(PCA_ALL_27[:, j], logf_27)[0] for j in (0, 1)]
    print(f"(1) corrélation de rang entre la fréquence et l'axe horizontal : {fr(rho_27[0], 2)} ; l'axe vertical : {fr(rho_27[1], 2)}.")
    if UMAP_ALL_27 is not None:
        _, near_27 = NearestNeighbors(n_neighbors=11).fit(UMAP_ALL_27).kneighbors(UMAP_ALL_27)
        top_27 = np.arange(len(ITOS_25)) < 100
        share_27 = float(top_27[near_27[top_27, 1:]].mean())
        print(f"(2) les 10 voisins sur la carte des 100 mots les plus fréquents sont aussi parmi les 100 plus fréquents "
              f"{fr(100 * share_27, 0)} fois sur 100 (au hasard : {fr(100 * 100 / len(ITOS_25), 0)}).")
    labels_27 = np.array([g for g, ws in GROUPS_27.items() for _ in ws])
    _, near3_27 = NearestNeighbors(n_neighbors=4).fit(PCA_GROUPS_27).kneighbors(PCA_GROUPS_27)
    print("(3) part des 3 voisins sur la carte qui sont du même groupe : "
          + ", ".join(f"{g} {fr(float((labels_27[near3_27[labels_27 == g, 1:]] == g).mean()), 2)}" for g in GROUPS_27))'''

# --- Part E -----------------------------------------------------------------------------------------------------------
GIVEN_28 = r'''# all-MiniLM-L6-v2 at its frozen revision (eval(), no gradient), the passages of Holmes and their index
SM_28 = wbtext.try_sentence_model("minilm", "B2.28")
PASSAGES_28, STORY_28 = wbtext.holmes_passages()               # 1 055 passages of at most 120 words, and their story
TITLES_28 = [title for title, _ in wbtext.holmes_stories()]
QUERIES_28 = ["a woman whose fiancé vanished on the morning of the wedding",
              "a deadly snake that comes through a ventilator",
              "men with red hair paid to copy out the encyclopaedia",
              "a precious blue stone found in the crop of a Christmas goose",
              "a beggar in the City who is really a respectable gentleman",
              "a governess asked to cut her hair short"]
INDEX_28 = None
if SM_28 is not None:
    start_28 = time.perf_counter()
    INDEX_28 = wbtext.passage_embeddings(SM_28, PASSAGES_28, "holmes-120")   # (1 055, 384), the reference index
    print(f"{SM_28} ; index de {INDEX_28.shape[0]} passages en {fr(time.perf_counter() - start_28, 1)} s.")'''

TODO_28 = r'''def embed_28(texts, tok, model, batch_size=32, max_length=256):
    """float32 array (N, 384): the sentence embeddings of the texts, mean pooling with the attention mask, norm 1."""
    raise NotImplementedError("embed_28")


def search_28(query, index, k=5):
    """(indices, cosines): the k rows of `index` (passages embedded, norm 1) closest to the query (a str), embedded by
    embed_28 with SM_28, found by your most_similar (B2.23)."""
    raise NotImplementedError("search_28")'''

SOLUTION_28 = r'''def embed_28(texts, tok, model, batch_size=32, max_length=256):
    """float32 array (N, 384): the sentence embeddings of the texts, mean pooling with the attention mask, norm 1."""
    out = []
    for start in range(0, len(texts), batch_size):
        batch = tok(list(texts[start:start + batch_size]), padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")
        with torch.inference_mode():
            h = model(**batch).last_hidden_state                    # (B, T, 384)
        m = batch["attention_mask"].unsqueeze(-1).float()           # (B, T, 1)
        out.append(F.normalize((h * m).sum(dim=1) / m.sum(dim=1), dim=1))
    return torch.cat(out).numpy()


def search_28(query, index, k=5):
    """(indices, cosines): the k rows of `index` (passages embedded, norm 1) closest to the query (a str), embedded by
    embed_28 with SM_28, found by your most_similar (B2.23)."""
    q = embed_28([query], SM_28.tokenizer, SM_28.model)[0]
    return mylearn.embeddings.most_similar(index, q, k=k)'''

CHECK_28 = r'''if SM_28 is None:
    print("⏳ Ex B2.28 : la vérification attend all-MiniLM-L6-v2 (connexion au Hub, ou Colab).")
else:
    with checked("B2.28"):
        tok_28, model_28 = SM_28.tokenizer, SM_28.model
        texts_28 = [PASSAGES_28[3], "Holmes smiled.", PASSAGES_28[400], " ".join(PASSAGES_28[10:15])]   # the last: > 256 tokens
        got_28 = embed_28(texts_28, tok_28, model_28)
        if not returned("B2.28", "embed_28", got_28):
            raise StopCheck
        want_28 = SM_28.embed(texts_28)
        same("B2.28", "embed_28 de quatre textes de longueurs différentes, dont un de plus de 256 tokens", got_28, want_28,
             rtol=1e-4, atol=1e-5)
        alone_28 = embed_28(["Holmes smiled."], tok_28, model_28)
        verdict("B2.28", np.allclose(np.asarray(alone_28)[0], np.asarray(got_28)[1], atol=1e-5),
                "une phrase courte a le même vecteur seule ou dans un batch avec des textes longs : le masque retire le padding.",
                "« Holmes smiled. » n'a pas le même vecteur seule et dans un batch : le padding entre-t-il dans ta moyenne ?")
        if not np.allclose(got_28, want_28, rtol=1e-4, atol=1e-5) and np.allclose(np.asarray(got_28)[:3], want_28[:3], atol=1e-5):
            print("💡 seul le texte long diffère : max_length = 256, comme la fiche modèle (pas 512).")
        found_28 = search_28(QUERIES_28[0], INDEX_28, k=5)
        if not returned("B2.28", "search_28", found_28):
            raise StopCheck
        idx_28, sims_28 = pair("B2.28", "search_28", found_28, names=("indices", "cosines"))
        ref_28 = np.argsort(-(INDEX_28 @ SM_28.embed(QUERIES_28[0])[0]), kind="stable")[:5]
        verdict("B2.28", np.asarray(idx_28).tolist() == ref_28.tolist(), "search_28 : les 5 passages de la référence, dans l'ordre.",
                f"search_28 : attendu les passages {ref_28.tolist()}, reçu {np.asarray(idx_28).tolist()}.")
        for q in QUERIES_28:
            idx, sims = search_28(q, INDEX_28, k=3)
            print(f"\n« {q} »")
            for i, s in zip(idx, sims):
                print(f"   {fr(float(s), 3)}  {TITLES_28[STORY_28[i]]:<28} {PASSAGES_28[i][:90]}…")'''

GIVEN_29 = r'''# Three words with two senses each, six sentences each (taken from Holmes or written in its style)
SENTENCES_29 = {
    "bank": [("argent", "This fellow Merryweather is a bank director, and personally interested in the matter."),
             ("argent", "I was seated in my office at the bank when a card was brought in to me."),
             ("argent", "He had some fourteen thousand pounds, which lay to his credit at the bank."),
             ("rive", "As of the swimmer who leaves the bank, she hurried across the road."),
             ("rive", "He was lying on the grassy bank of the pool, close to the water."),
             ("rive", "The boat was tied to a tree on the bank of the river.")],
    "match": [("allumette", "Holmes sprang from the bed, struck a match, and lashed at the bell-pull."),
              ("allumette", "In her right hand was found the charred stump of a match."),
              ("allumette", "By the light of her match she caught a hurried glimpse of him."),
              ("mariage", "My stepfather has offered no opposition to the match, and we are to be married."),
              ("mariage", "Her father thought it a very good match for her, and gave his consent."),
              ("mariage", "It was a love match, and the wedding was fixed for the spring.")],
    "ring": [("bijou", "He slipped an emerald snake ring from his finger and held it out."),
             ("bijou", "She put a new wedding ring upon the top of the pile."),
             ("bijou", "He wore a heavy gold ring with a large stone on his little finger."),
             ("sonnerie", "There came a ring to my bell, about the hour when a man gives his first yawn."),
             ("sonnerie", "I think that I heard a ring at the door, said Holmes."),
             ("sonnerie", "There was a sharp ring at the bell, and the maid went to open.")]}
SM_29 = SM_28 if "SM_28" in globals() else wbtext.try_sentence_model("minilm", "B2.29")'''

TODO_29 = r'''def contextual_vector_29(sentence, word, tok, model):
    """The output of the last block (last_hidden_state) for `word` in `sentence`: the mean of the vectors of the tokens
    of its first occurrence (found with the offsets of the tokenizer), a NumPy array (384,)."""
    raise NotImplementedError("contextual_vector_29")


def static_vector_29(word, tok, model):
    """The row of the input table (model.get_input_embeddings()) of `word`, averaged over its tokens: (384,)."""
    raise NotImplementedError("static_vector_29")


def sense_means_29(vectors, senses):
    """(mean cosine of the pairs of uses of the SAME sense, mean cosine of the pairs of DIFFERENT senses), two floats."""
    raise NotImplementedError("sense_means_29")'''

SOLUTION_29 = r'''def contextual_vector_29(sentence, word, tok, model):
    """The output of the last block (last_hidden_state) for `word` in `sentence`: the mean of the vectors of the tokens
    of its first occurrence (found with the offsets of the tokenizer), a NumPy array (384,)."""
    start = re.search(rf"\b{re.escape(word)}\b", sentence).start()
    end = start + len(word)
    batch = tok(sentence, return_tensors="pt", return_offsets_mapping=True)
    offsets = batch.pop("offset_mapping")[0].tolist()                 # (start, end) of each token in the sentence
    keep = [k for k, (a, b) in enumerate(offsets) if b > a and a < end and b > start]   # special tokens: (0, 0)
    with torch.inference_mode():
        h = model(**batch).last_hidden_state[0]                       # (T, 384)
    return h[keep].mean(dim=0).numpy()


def static_vector_29(word, tok, model):
    """The row of the input table (model.get_input_embeddings()) of `word`, averaged over its tokens: (384,)."""
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return model.get_input_embeddings().weight[ids].mean(dim=0).detach().numpy()


def sense_means_29(vectors, senses):
    """(mean cosine of the pairs of uses of the SAME sense, mean cosine of the pairs of DIFFERENT senses), two floats."""
    V = np.asarray(vectors, dtype=float)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    i, j = np.triu_indices(len(senses), k=1)                          # each pair of uses once
    same_sense = np.array(senses)[i] == np.array(senses)[j]
    return float(S[i, j][same_sense].mean()), float(S[i, j][~same_sense].mean())'''

CHECK_29 = r'''if SM_29 is None:
    print("⏳ Ex B2.29 : l'expérience attend all-MiniLM-L6-v2 (connexion au Hub, ou Colab).")
else:
    with checked("B2.29"):
        tok_29, model_29 = SM_29.tokenizer, SM_29.model
        probe_29 = "The bank-note was found on the bank, near the river."      # two « bank »: the first one counts
        v_29 = contextual_vector_29(probe_29, "bank", tok_29, model_29)
        if not returned("B2.29", "contextual_vector_29", v_29):
            raise StopCheck
        same("B2.29", "contextual_vector_29 de « bank » (la première occurrence)", v_29, SM_29.word_vector(probe_29, "bank"),
             rtol=1e-4, atol=1e-5)
        same("B2.29", "contextual_vector_29 d'un mot de plusieurs tokens (« Merryweather »)",
             contextual_vector_29(SENTENCES_29["bank"][0][1], "Merryweather", tok_29, model_29),
             SM_29.word_vector(SENTENCES_29["bank"][0][1], "Merryweather"), rtol=1e-4, atol=1e-5)
        same("B2.29", "static_vector_29 de « bank » et de « Merryweather »",
             np.stack([static_vector_29(w, tok_29, model_29) for w in ("bank", "Merryweather")]),
             np.stack([SM_29.static_vector(w) for w in ("bank", "Merryweather")]), rtol=1e-5, atol=1e-6)
        means_29 = sense_means_29(np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]]), ["a", "a", "b", "b"])
        if returned("B2.29", "sense_means_29", means_29):
            want_29 = (np.mean([math.sqrt(0.5), 0.0]), np.mean([math.sqrt(0.5), -math.sqrt(0.5), 0.0, -1.0]))
            same("B2.29", "sense_means_29 sur quatre vecteurs du plan (sens a, a, b, b)", means_29, want_29)

if SM_29 is not None and written(lambda: contextual_vector_29("a bank", "bank", SM_29.tokenizer, SM_29.model)) \
        and written(lambda: sense_means_29(np.eye(3), ["a", "a", "b"])):
    with checked("B2.29"):
        rows_29 = []
        fig_29, axes_29 = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, (word, uses) in zip(axes_29, SENTENCES_29.items()):
            senses = [s for s, _ in uses]
            ctx = np.stack([contextual_vector_29(s, word, SM_29.tokenizer, SM_29.model) for _, s in uses])
            stat = np.stack([static_vector_29(word, SM_29.tokenizer, SM_29.model) for _ in uses])
            same_ctx, cross_ctx = sense_means_29(ctx, senses)
            same_st, cross_st = sense_means_29(stat, senses)
            rows_29.append({"mot": word, "contextuel, même sens": same_ctx, "contextuel, sens différents": cross_ctx,
                            "statique, même sens": same_st, "statique, sens différents": cross_st})
            U = ctx / np.linalg.norm(ctx, axis=1, keepdims=True)
            im = ax.imshow(U @ U.T, vmin=0.6, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(6), [f"{s[:4]}{k % 3 + 1}" for k, s in enumerate(senses)], fontsize=8)
            ax.set_yticks(range(6), [f"{s[:4]}{k % 3 + 1}" for k, s in enumerate(senses)], fontsize=8)
            ax.set_title(f"« {word} »")
            ax.grid(False)
        fig_29.colorbar(im, ax=axes_29.tolist(), label="cosinus des vecteurs contextuels")
        table_29 = pd.DataFrame(rows_29).set_index("mot")
        print(table_29.to_string(float_format=lambda x: f"{x:.3f}"))
        verdict("B2.29", np.allclose(table_29[["statique, même sens", "statique, sens différents"]], 1.0),
                "le vecteur statique d'un mot est le même dans toutes les phrases : cosinus 1, quel que soit le sens.",
                "le vecteur statique doit être la même ligne de la table dans toutes les phrases (cosinus 1).")
        plt.show()'''

GIVEN_30 = r'''# The tokenizer to deliver: a byte-level BPE of 1 000 tokens of your library, learned on VERNE_TRAIN, and the frozen
# reference text of its non-regression test (French, emoji, ideographs, a « € »)
import tempfile
from pathlib import Path

REFERENCE_TEXT_30 = VERNE_VAL[:1500] + "\n« Ça coûte 15 € », dit-il 🙂 漢字.\n"
_here_30 = Path.cwd().resolve()
DELIVERY_30 = (_here_30 / "livraison_b2_30") if "mon_travail" in _here_30.parts else Path(tempfile.mkdtemp(prefix="wb_b2_30_"))
TOK_30 = fast_byte_bpe(VERNE_TRAIN, 1000) if bpe_written() else None
LIBRARY_30 = None
if TOK_30 is not None:                       # where the test file imports BPETokenizer from: your library
    _package_30 = Path(mylearn.tokenizer.__file__).resolve().parent    # mon_travail/mylearn (or the reference)
    LIBRARY_30 = (str(_package_30.parent), f"{_package_30.name}.tokenizer")
print(f"Dossier de livraison : {DELIVERY_30}")


def broken_copies_30(folder):
    """Three broken copies of the delivered tokenizer.json (for the check): two merges swapped, the last merge removed,
    the pre-tokenizer changed. Returns {description: path}."""
    state = json.loads((Path(folder) / "tokenizer.json").read_text(encoding="utf-8"))
    out = {}
    for name, change in (("deux fusions échangées", lambda s: s["merges"].__setitem__(slice(10, 12), s["merges"][10:12][::-1])),
                         ("la dernière fusion retirée", lambda s: s["merges"].pop()),
                         ("le pré-tokenizer 'whitespace' au lieu de 'gpt2'", lambda s: s["params"].__setitem__("pre_tokenizer", "whitespace"))):
        broken = json.loads(json.dumps(state))
        change(broken)
        target = Path(tempfile.mkdtemp(prefix="wb_b2_30_broken_")) / "tokenizer.json"
        target.write_text(json.dumps(broken), encoding="utf-8")
        out[name] = target
    return out'''

TODO_30 = r'''def deliver_30(tok, folder, reference_text):
    """Write folder/tokenizer.json (tok.save) and folder/golden.json ({"text": reference_text, "ids": its ids}); return
    (path of tokenizer.json, path of golden.json)."""
    raise NotImplementedError("deliver_30")


def test_tokenizer_regression_30():
    """The non-regression test: BPETokenizer.load(TOKENIZER_PATH) must give the frozen ids of GOLDEN_PATH, decode them back
    to the text, and keep its vocabulary size (len(vocab_))."""
    raise NotImplementedError("test_tokenizer_regression_30")'''

SOLUTION_30 = r'''def deliver_30(tok, folder, reference_text):
    """Write folder/tokenizer.json (tok.save) and folder/golden.json ({"text": reference_text, "ids": its ids}); return
    (path of tokenizer.json, path of golden.json)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tok_path, golden_path = folder / "tokenizer.json", folder / "golden.json"
    tok.save(tok_path)
    golden = {"text": reference_text, "ids": tok.encode(reference_text), "vocab_size": len(tok.vocab_)}
    golden_path.write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
    return tok_path, golden_path


def test_tokenizer_regression_30():
    """The non-regression test: BPETokenizer.load(TOKENIZER_PATH) must give the frozen ids of GOLDEN_PATH, decode them back
    to the text, and keep its vocabulary size (len(vocab_))."""
    tok = BPETokenizer.load(TOKENIZER_PATH)
    golden = json.loads(Path(GOLDEN_PATH).read_text(encoding="utf-8"))
    ids = tok.encode(golden["text"])
    assert len(tok.vocab_) == golden["vocab_size"], f"expected {golden['vocab_size']} tokens, got {len(tok.vocab_)}"
    first = next((k for k, (a, b) in enumerate(zip(ids, golden["ids"])) if a != b), None)
    assert ids == golden["ids"], f"expected the frozen ids; first difference at position {first}"
    assert tok.decode(ids) == golden["text"], "expected decode(encode(text)) == text"'''

CHECK_30 = r'''import inspect

if TOK_30 is None:
    print("⏳ Ex B2.30 : la livraison attend ton BPETokenizer (B2.16 et B2.17).")
else:
    with checked("B2.30"):
        paths_30 = deliver_30(TOK_30, DELIVERY_30, REFERENCE_TEXT_30)
        if not returned("B2.30", "deliver_30", paths_30):
            raise StopCheck
        tok_path_30, golden_path_30 = (Path(p) for p in pair("B2.30", "deliver_30", paths_30, names=("tokenizer.json", "golden.json")))
        reloaded_30 = type(TOK_30).load(tok_path_30)
        golden_30 = json.loads(golden_path_30.read_text(encoding="utf-8"))
        verdict("B2.30", reloaded_30.merges_ == TOK_30.merges_ and golden_30.get("text") == REFERENCE_TEXT_30
                and golden_30.get("ids") == TOK_30.encode(REFERENCE_TEXT_30),
                f"livraison : {tok_path_30.name} se recharge à l'identique, {golden_path_30.name} contient le texte et ses "
                f"{len(golden_30.get('ids', []))} ids figés.",
                "tokenizer.json doit se recharger avec BPETokenizer.load, et golden.json contenir le texte de référence "
                "(clé 'text') et ses ids (clé 'ids').")
    if "NotImplementedError" in inspect.getsource(test_tokenizer_regression_30):
        print("⏳ Ex B2.30 : écris ton test de non-régression.")
    elif "tok_path_30" in globals():
        def preamble_30(tok_path):
            return (f"import json\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, {LIBRARY_30[0]!r})\n"
                    f"from {LIBRARY_30[1]} import BPETokenizer\nTOKENIZER_PATH = {str(tok_path)!r}\n"
                    f"GOLDEN_PATH = {str(golden_path_30)!r}")
        print("Ton test, sur le tokenizer livré (il doit passer) :")
        good_30 = wb.run_pytest([test_tokenizer_regression_30], preamble=preamble_30(tok_path_30))
        verdict("B2.30", good_30.ok, "ton test passe sur le tokenizer livré.", "ton test doit passer sur le tokenizer livré "
                "(la sortie de pytest est au-dessus).")
        for why_30, path_30 in (broken_copies_30(DELIVERY_30).items() if good_30.ok else ()):
            caught_30 = not wb.run_pytest([test_tokenizer_regression_30], preamble=preamble_30(path_30), quiet=True).ok
            verdict("B2.30", caught_30, f"ton test attrape un tokenizer.json abîmé : {why_30}.",
                    f"ton test laisse passer un tokenizer.json abîmé : {why_30}.")'''

GIVEN_31 = r'''# The challenge: the best lossless compression of VERNE_VAL with at most 2 000 tokens, in less than 2 minutes of CPU
GOAL_31, LIMIT_31, MAX_TOKENS_31 = 3.20, 120.0, 2000     # bytes per token to reach; seconds; tokens (special included)
BASELINE_31 = 3.001        # B2.20: byte-level BPE 'gpt2' of 2 000 tokens, by rank, learned on VERNE_TRAIN, on VERNE_VAL
BASELINE_CONTROL_31 = 2.955   # the same BPE learned on the first 8/9 of VERNE_TRAIN, on its last ninth (the control)


def _make_evaluate_31(validation, train):
    """evaluate_31, with the validation text and the control text kept here only (the global variables that hold them
    are replaced by a decoy while your method runs)."""
    cut = wbtext.split_point(train, 8 / 9)
    texts = {"validation": (validation, train), "contrôle": (train[cut:], train[:cut])}   # name: (measured, learned on)
    decoy = " ".join(sorted(set(train.split()))[:6000])       # words of the training text, in alphabetical order

    def marker(text):
        at = next(i for i in range(1000, len(text)) if not set(text[i:i + 60]) & set("\n'\"\\"))
        return text[at:at + 60]                                 # a piece of the text, the same in a repr

    def holds(value, markers):
        try:
            if isinstance(value, (bytes, bytearray)):
                return any(m.encode("utf-8") in value for m in markers)
            shown = value if isinstance(value, str) else str(value)[:5_000_000]
        except Exception:  # noqa: BLE001 - an object without a readable str
            return False
        return any(m in shown for m in markers)

    def foreign(vocab, measured, learned):
        """The tokens of 4 bytes or more found in the measured text but nowhere in the text the method received."""
        measured, learned = measured.encode("utf-8"), learned.encode("utf-8")
        return sorted({b for b in vocab.values() if isinstance(b, bytes) and len(b) >= 4 and b in measured
                       and b not in learned}, key=lambda b: (-len(b), b))

    def run(method, name):
        """Train the method on the text that goes with `name` and encode the text measured, the global variables that
        hold a measured text being replaced by the decoy, and wbtext giving only the text learned on. Returns
        (tokenizer, ids, seconds), or None."""
        g = globals()
        measured, learned = texts[name]
        markers = [marker(validation)] + ([marker(measured)] if name == "contrôle" else [])
        hidden = {n: v for n, v in list(g.items()) if not n.startswith("_")
                  and isinstance(v, (str, bytes, bytearray, dict, list, tuple)) and holds(v, markers)}
        load_text, load_verne, cached = wbtext.load_text, wbtext.datasets.load_verne, wbtext._TEXTS.get("verne")
        try:
            for n, v in hidden.items():
                g[n] = decoy.encode("utf-8") if isinstance(v, (bytes, bytearray)) else decoy
            # the other paths to Verne (wbtext, wb.datasets) give only the text learned on during the run
            wbtext.load_text = lambda text_name, _load=load_text: learned if text_name == "verne" else _load(text_name)
            wbtext.datasets.load_verne = lambda *args, **kwargs: learned
            wbtext._TEXTS["verne"] = learned
            start = time.perf_counter()
            tok = method(learned)
            fit_seconds = time.perf_counter() - start
            vocab = getattr(tok, "vocab_", None)
            if not isinstance(vocab, dict) or not vocab or not all(isinstance(b, bytes) for b in vocab.values()):
                print("❌ Ex B2.31 : le tokenizer rendu doit avoir un vocab_, un dict {id: bytes} (comme ton BPETokenizer).")
                return None
            start = time.perf_counter()
            ids = tok.encode(measured)
            return tok, ids, fit_seconds + time.perf_counter() - start
        finally:
            for n, v in hidden.items():
                g[n] = v
            wbtext.load_text, wbtext.datasets.load_verne = load_text, load_verne
            if cached is None:
                wbtext._TEXTS.pop("verne", None)
            else:
                wbtext._TEXTS["verne"] = cached

    def evaluate_31(method):
        """method(text) -> tokenizer: the four rules of the statement, then the bytes per token on the validation."""
        out = run(method, "validation")
        if out is None:
            return None
        tok, ids, seconds = out
        vocab, val_bytes = tok.vocab_, validation.encode("utf-8")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, (int, np.integer)) and i in vocab for i in ids):
            print("❌ Ex B2.31 : encode doit rendre une liste d'ids entiers, tous des clés de vocab_.")
            return None
        lossless = b"".join(vocab[i] for i in ids) == val_bytes
        decoded = tok.decode(ids) == validation if hasattr(tok, "decode") else False
        score = len(val_bytes) / len(ids)
        print(f"{len(vocab)} tokens ; validation : {len(ids):_} tokens pour {len(val_bytes):_} octets, soit {fr(score, 3)} "
              f"octets par token ; entraînement + encodage : {fr(seconds, 1)} s.".replace("_", " "))
        verdict("B2.31", len(vocab) <= MAX_TOKENS_31, f"{len(vocab)} tokens, au plus {MAX_TOKENS_31}.",
                f"{len(vocab)} tokens dans vocab_ : le défi en permet {MAX_TOKENS_31}, tokens spéciaux compris.")
        verdict("B2.31", lossless and decoded, "sans perte : les octets des tokens redonnent la validation, et decode aussi.",
                "pas sans perte : " + ("les octets des tokens, mis bout à bout, ne redonnent pas la validation" if not lossless
                                       else "decode(ids) ne redonne pas la validation") + ".")
        # rule 3: tokens seen only in the validation, then the control (the same method, learned without the last
        # ninth of VERNE_TRAIN, measured on it)
        unseen = foreign(vocab, validation, train)
        verdict("B2.31", not unseen, "aucun token de 4 octets ou plus n'est dans la validation sans être dans VERNE_TRAIN.",
                f"{len(unseen)} token(s) de 4 octets ou plus sont dans la validation mais nulle part dans VERNE_TRAIN, le "
                f"texte que reçoit ta méthode (le plus long : {unseen[0].decode('utf-8', errors='replace')!r}) : d'où "
                "viennent-ils ?" if unseen else "")
        honest, c_seconds, control = False, None, run(method, "contrôle")
        if control is not None:
            (measured, learned), (c_tok, c_ids, c_seconds) = texts["contrôle"], control
            if isinstance(c_ids, list) and c_ids and b"".join(c_tok.vocab_.get(i, b"?") for i in c_ids) == measured.encode("utf-8"):
                gain, c_gain = score / BASELINE_31, len(measured.encode("utf-8")) / len(c_ids) / BASELINE_CONTROL_31
                c_unseen = foreign(c_tok.vocab_, measured, learned)
                honest = gain - c_gain <= 0.02 and not c_unseen
                if c_unseen:
                    print(f"❌ Ex B2.31 : contrôle : {len(c_unseen)} token(s) de 4 octets ou plus sont dans le texte tenu à "
                          "l'écart mais nulle part dans le texte que ta méthode a reçu : apprend-elle sur autre chose que "
                          "son argument (VERNE_TRAIN lu par une variable globale, un fichier) ?")
                else:
                    verdict("B2.31", honest, f"contrôle : sur un texte tenu à l'écart, ta méthode gagne autant sur la "
                            f"référence de B2.20 ({fr(100 * (c_gain - 1), 1)} %) que sur la validation "
                            f"({fr(100 * (gain - 1), 1)} %).",
                            f"contrôle : ta méthode gagne {fr(100 * (gain - 1), 1)} % sur la validation, mais seulement "
                            f"{fr(100 * (c_gain - 1), 1)} % sur un texte tenu à l'écart : apprend-elle autre chose que le "
                            "texte qu'elle reçoit (la validation, une variable globale) ?")
            else:
                print("❌ Ex B2.31 : contrôle : sur le texte tenu à l'écart, ta méthode n'est pas sans perte.")
        # rule 4, measured cold: the control is a text the method has never seen, so it trains there for real
        slowest = max(seconds, c_seconds if c_seconds is not None else 0.0)
        verdict("B2.31", slowest <= LIMIT_31,
                f"en {fr(seconds, 1)} s sur la validation et {fr(c_seconds or 0.0, 1)} s sur le contrôle, chacun sous les "
                f"{LIMIT_31:.0f} s.",
                f"{fr(seconds, 1)} s sur la validation, {fr(c_seconds or 0.0, 1)} s sur le contrôle : le défi en permet "
                f"{LIMIT_31:.0f} à chacun (entraînement + encodage).")
        ok = len(vocab) <= MAX_TOKENS_31 and lossless and decoded and not unseen and slowest <= LIMIT_31 and honest
        verdict("B2.31", ok and score >= GOAL_31, f"défi réussi : {fr(score, 3)} octets par token (objectif {fr(GOAL_31, 2)}, "
                f"référence de B2.20 {fr(BASELINE_31, 3)}).",
                f"pas encore : {fr(score, 3)} octets par token, il en faut {fr(GOAL_31, 2)} en respectant les quatre règles.")
        return {"bytes_per_token": score, "seconds": seconds, "control_seconds": c_seconds, "tokens": len(vocab),
                "honest": honest and not unseen, "ok": ok and score >= GOAL_31}

    return evaluate_31


evaluate_31 = _make_evaluate_31(VERNE_VAL, VERNE_TRAIN)'''

TODO_31 = r'''def method_31(text):
    """Your method: learn on `text` (VERNE_TRAIN) ONLY a tokenizer with vocab_ ({id: bytes}, at most 2 000 entries),
    encode(text) -> list of ids and decode(ids) -> str, lossless."""
    raise NotImplementedError("method_31")'''

SOLUTION_31 = r'''from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def gpt2_byte_table():
    """GPT-2's table byte -> printable character (fiche §202.4), used by the ByteLevel pre-tokenizer of tokenizers."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = {}, 0
    for b in range(256):
        if b in keep:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + extra)
            extra += 1
    return chars


class LinesTokenizer31:
    """A byte-level vocabulary, applied by the shortest segmentation (dynamic programming)."""

    def __init__(self, vocab):
        self.vocab_ = dict(vocab)
        self._index = {b: i for i, b in sorted(self.vocab_.items(), reverse=True)}   # the smallest id for a duplicate
        self._longest = max(len(b) for b in self.vocab_.values())

    def encode(self, text):
        data = text.encode("utf-8")
        n, best, back = len(data), [0] + [math.inf] * len(data), [None] * (len(data) + 1)
        for i in range(n):                                    # best[j]: fewest tokens for the first j bytes
            step = best[i] + 1
            for j in range(i + 1, min(n, i + self._longest) + 1):
                token = self._index.get(data[i:j])
                if token is not None and step < best[j]:
                    best[j], back[j] = step, (i, token)
        ids, j = [], n
        while j:
            i, token = back[j]
            ids.append(token)
            j = i
        return ids[::-1]

    def decode(self, ids):
        return b"".join(self.vocab_[i] for i in ids).decode("utf-8", errors="replace")


def method_31(text):
    """Your method: learn on `text` (VERNE_TRAIN) ONLY a tokenizer with vocab_ ({id: bytes}, at most 2 000 entries),
    encode(text) -> list of ids and decode(ids) -> str, lossless."""
    # 1. no pre-tokenization inside a line: merges may cross the spaces (« de la », « Phileas Fogg »)
    hf = Tokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    trainer = trainers.BpeTrainer(vocab_size=MAX_TOKENS_31, min_frequency=2, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    hf.train_from_iterator(text.splitlines(keepends=True), trainer)
    to_byte = {c: b for b, c in gpt2_byte_table().items()}
    vocab = {i: bytes(to_byte[c] for c in s) for s, i in hf.get_vocab().items()}
    # 2. the shortest segmentation with this vocabulary, instead of the merges applied by rank
    return LinesTokenizer31(vocab)'''

CHECK_31 = r'''with checked("B2.31"):
    need("B2.31", "method_31")
    result_31 = evaluate_31(method_31)'''
