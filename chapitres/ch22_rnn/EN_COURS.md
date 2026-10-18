# Chapitre 22 : génération en cours

Ce chapitre est généré en deux sessions. Tant que ce fichier existe, `python tools/start_chapter.py 22` copie la fiche, `06_mes_reponses.md` et les squelettes `mylearn/rnn.py` et `mylearn/sequences.py`, mais pas encore le notebook.

| Fichier | État |
|---|---|
| `01_fiche.md` et `figures/` | complets (session 46) |
| `02_exercices.md`, `06_mes_reponses.md` | complets (session 46) |
| `04_indices.md`, `05_solutions.md` | quiz, rappels, papier, réflexion et entretien (session 46) ; notebook 22.12 à 22.29 à la session 47 |
| `flashcards.csv` | complet (session 46) |
| `03_notebook.ipynb`, `05_solutions.ipynb` | partie 0 (session 46) ; parties A à D (22.12 à 22.29) à la session 47 |
| `mylearn.rnn`, `mylearn.sequences` | références et tests complets (session 46) |

## Pour la session 47 : groupes de tests et précisions de contrat à écrire dans les énoncés des 🔨

Groupes `-k` : 22.12 `test_build_vocab_ or test_encode_ or test_decode_` ; 22.13 `test_make_windows_ or test_make_sequence_pairs_` (`tests/test_ch22_sequences.py`) ; 22.15 `test_rnn_cell_forward_ or test_rnn_forward_` ; 22.16 `test_lstm_cell_forward_ or test_lstm_forward_` ; 22.17 `test_gru_cell_forward_` (`tests/test_ch22_rnn.py`) ; 22.20 `test_temperature_softmax_ or test_sample_from_logits_` (`tests/test_ch22_sequences.py`).

Les tests (110 dans `test_ch22_rnn.py`, 100 dans `test_ch22_sequences.py`) exigent, au-delà des docstrings :

1. **Commun** : les entrées ne sont jamais modifiées et peuvent être des listes ; `rnn.py` rend toujours des tableaux **neufs** en float64 (jamais des vues, même `h_T` de `rnn_forward`).
2. **Poids** : la disposition de `mylearn`, `W_ih` de forme `(D, kH)` et `W_hh` de forme `(H, kH)` (la transposée de PyTorch) ; la disposition de PyTorch est refusée (`ValueError`), comme un biais diffusable ((1,) ou (kH, 1)) : un biais a la forme `(kH,)` exactement, `None` vaut zéro.
3. **Formes** : `x_t` et les états sont 2-D, avec le même N ; `x` est 3-D `(N, T, D)` avec T ≥ 1 ; un `h0` (ou `c0`) de mauvaise forme est refusé.
4. **Activations** : `nonlinearity` vaut `"tanh"` ou `"relu"`, rien d'autre ; aucun `RuntimeWarning` avec des pré-activations de ±1 000 (une sigmoïde stable).
5. **LSTM** : ordre des portes i, f, g, o ; `lstm_cell_forward` rend `(h_t, c_t)` dans cet ordre, `lstm_forward` rend `(outputs, (h_T, c_T))`, la paire étant un **tuple**. **GRU** : ordre r, z, n ; la porte r multiplie `(h W_hn + b_hn)` ; `h = (1 − z) n + z h_prev`.
6. **Fenêtres** (`make_windows`, `make_sequence_pairs`) : gardent le dtype de la série, rendent des tableaux neufs, acceptent une série `(T,)` ou `(T, D)` seulement ; `window`, `horizon`, `seq_len` et `stride` sont des `int` ≥ 1 (un `bool` n'en est pas un) ; une série trop courte pour un seul exemple : `ValueError`.
7. **`build_vocab`** : prend les caractères d'une `str` ou n'importe quel itérable de `str` (un générateur aussi) ; classe par (effectif décroissant, puis ordre de tri) ; met `unk` en premier (id 0), même s'il apparaît dans le texte, et le compte dans `max_size` ; refuse les tokens vides, et un `max_size` ou un `min_freq` qui n'est pas un `int` ≥ 1.
8. **`encode`** : des ids int64 (forme `(0,)` pour un texte vide) ; un token inconnu sans `unk` : `ValueError`, pas `KeyError` ; un `unk` absent de `stoi` : `ValueError`. **`decode`** : refuse les ids < 0 ou ≥ `len(itos)`, et des ids qui ne sont pas 1-D.
9. **`temperature_softmax`** : le long du dernier axe, forme gardée, stable (logits de ±1 000), T ≤ 0 : `ValueError`.
10. **`sample_from_logits`** : un `int` Python pour des logits `(V,)`, un tableau int64 `(N,)` pour `(N, V)` ; T = 0 : le premier argmax ; `top_k` garde les ex aequo du k-ième logit ; T < 0 ou `top_k` hors de [1, V] (ou pas un `int`) : `ValueError` ; la même graine donne les mêmes ids ; `rng=None` fonctionne. Les fréquences ne sont testées que statistiquement (20 000 tirages) : la méthode de tirage n'est pas imposée.

À reprendre aussi à la session 47 : `PARTS` et `NEXT_SESSION` de `tools/chapters/build_ch22.py` (les quatre parties annoncées), les sections « Notebook » de `04_indices.md` et `05_solutions.md`, puis supprimer ce fichier.
