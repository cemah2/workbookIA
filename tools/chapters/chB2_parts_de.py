"""Parts D and E of the chapter B2 notebook (phase « embeddings », B2.22 to B2.31): statements and cells (used by
chB2_parts.py, which appends them to PARTS)."""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part, guarded

import chB2_code as C0
import chB2_code_de as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB2.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}
W2V = HISTORY.get("w2v", {}).get("answers", {"fogg": "B", "characters": False, "pca_axis": False, "umap_frequent": True})

RELOAD = C0.RELOAD


def mylearn_howto(file: str) -> str:
    return (f"> **Mode d'emploi mylearn** : ouvre `mon_travail/mylearn/{file}` (créé par `python tools/start_chapter.py "
            "B2`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de l'exercice par ton code et "
            "**enregistre**. NumPy seulement. La première cellule de vérification recharge ta librairie et l'essaie sur "
            "des cas visibles ; la seconde lance les tests de l'exercice. "
            f"`python -m pytest tests/test_chb2_{file[:-3]}.py -q` les lance tous dans un terminal.")


def tests_after(ex_id: str, probe: str, keyword: str, n_tests: int, what: str = "tes fonctions") -> list:
    """The second check cell of a 🔨 of embeddings.py: the tests of the exercise (explicit test file, -k group)."""
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_EMB, "{keyword}")   # {n_tests} tests\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')
    ref = cell.replace('run_pytest_group(TEST_EMB, "', 'run_pytest_group(TEST_EMB, impl="ref", keyword="')
    return [("check", cell), ("solution", ref)]


def _letters(answer: str, hints: dict) -> str:
    return "{" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items() if letter != answer) + "}"


SOLUTION_26 = f'fogg_26, characters_26 = "{W2V["fogg"]}", {W2V["characters"]}'
RECORD_26 = (
    f'wb.record("B2.26a", "{W2V["fogg"]}", mistakes=' + _letters(W2V["fogg"], {
        "A": "regarde les deux premiers voisins affichés pour « fogg » : sont-ce d'autres personnages ?",
        "B": "regarde les deux premiers voisins affichés pour « fogg » : sont-ce les mots qui l'entourent dans le texte ?",
        "C": "regarde les deux premiers voisins affichés pour « fogg » : sont-ce des verbes ?",
        "D": "regarde les deux premiers voisins affichés pour « fogg » : relis les trois autres choix"}) + ")\n"
    f'wb.record("B2.26b", {W2V["characters"]}, mistakes={{"compte les noms de CHARACTERS_26 parmi les 10 voisins de '
    f'« passepartout » (la ligne affichée sous les voisins)": {not W2V["characters"]}}})')
SOLUTION_27 = f'pca_axis_27, umap_frequent_27 = {W2V["pca_axis"]}, {W2V["umap_frequent"]}'
RECORD_27 = (
    f'wb.record("B2.27a", {W2V["pca_axis"]}, mistakes={{"suis la couleur de la carte (1) de gauche à droite : passe-t-elle '
    f'régulièrement du violet au jaune ?": {not W2V["pca_axis"]}}})\n'
    f'wb.record("B2.27b", {W2V["umap_frequent"]}, mistakes={{"regarde où sont les points jaunes de la carte (2) : '
    f'partout, ou surtout à quelques endroits ?": {not W2V["umap_frequent"]}}})')

COMMON_EMB = (
    "**Contrat commun de `embeddings.py`** (les docstrings, plus ces précisions que vérifient les tests) :\n"
    "- tes fonctions acceptent des listes et des tableaux d'entiers ou de flottants, et ne modifient jamais leurs "
    "entrées ;\n"
    "- une entrée invalide lève `ValueError`, jamais `IndexError` ni un résultat silencieusement faux ;\n"
    "- NumPy seulement ; les tests comparent à une double boucle, à la formule en `float64`, à scikit-learn et à "
    "PyTorch, à `rtol = 1e-6` et `atol = 1e-6` près (un calcul en `float32` passe ; la tolérance s'affiche dans le "
    "message).")

# ---------------------------------------------------------------------------------------------------------------------
PART_D = Part("D", "Des vecteurs de mots : compter, factoriser, prédire",
              "Deuxième phase : des tokens aux vecteurs. Tu écris dans `mylearn/embeddings.py` la matrice de "
              "co-occurrences et la PPMI (B2.22), puis le cosinus et les plus proches voisins (B2.23). Tu construis "
              "des vecteurs de mots de Holmes par PPMI + SVD (B2.24), puis tu entraînes un word2vec en PyTorch sur "
              "Verne (B2.25), dont tu prédis les voisins de deux personnages (B2.26) avant de lire ses cartes (B2.27). "
              "`cosine_similarity_matrix` et `most_similar` resserviront en B4 (recherche dense).\n\n"
              "La cellule suivante importe PyTorch et prépare les mots des deux romans, comme la carte de la fiche "
              "(§202.9) : `WORDS_H`, en minuscules et en lettres `a` à `z` ; `WORDS_V`, en minuscules, accents gardés "
              "(« été », « où »). Elle donne aussi `nearest_words(E, itos, mot, k)`, les $k$ plus proches voisins "
              "d'un mot dans une matrice de vecteurs de norme 1, pour les cellules de vérification. Au ch. 22, tu "
              "avais déjà découpé ces deux textes en mots pour en faire un vocabulaire ; ici, chaque mot devient un "
              "vecteur.\n\n" + COMMON_EMB,
              given=C.TOOLS_D, exercises=[
    Ex("B2.22", "🔨", 2, 30, "cooccurrence_matrix et ppmi",
       "compter les co-occurrences d'une suite de mots dans une fenêtre, puis les transformer en information mutuelle "
       "ponctuelle positive.",
       "✏️ B2.6 · ch. 6 (information, divergence KL) · fiche §202.8 (« La matrice de co-occurrences », « PMI et PPMI »)",
       tracks="M, C", mylearn="embeddings.py",
       body=mylearn_howto("embeddings.py") + r"""

**Contrat**
- `cooccurrence_matrix(ids, vocab_size, window=2, weighting="uniform", symmetric=True)` :
  - rend une matrice **float** `(vocab_size, vocab_size)`, même en `'uniform'` ;
  - pour chaque position `i` et chaque distance `d` de 1 à `window`, ajoute le poids (1, ou `1 / d` en `'harmonic'`) à `C[ids[i], ids[i + d]]` et, si `symmetric`, à `C[ids[i], ids[i - d]]` ; avec `symmetric=False`, seulement le contexte de droite ;
  - les répétitions d'une même paire s'**additionnent** : `[0, 0, 0, 1]` avec une fenêtre de 1 donne `C[0, 0] = 4` ;
  - les ids sont des entiers (une liste, des `int32` ou des `uint8` conviennent) ; une suite plus courte que la fenêtre est permise ;
  - `ValueError` pour des ids qui ne sont pas 1-D, `window < 1`, une pondération inconnue, et un id hors de `[0, vocab_size)`, −1 compris (pas d'`IndexError`, pas d'indice négatif à la Python).
- `ppmi(C, alpha=1.0)` :
  - $\max\big(0, \log \frac{p(w, c)}{p(w)\,p_\alpha(c)}\big)$ avec le **logarithme naturel**, les lignes pour les mots, les colonnes pour les contextes ;
  - le lissage `alpha` ne porte que sur les **colonnes** : $p_\alpha(c) = \#(c)^\alpha / \sum_{c'} \#(c')^\alpha$ (renormalisé) ;
  - 0 là où `C == 0`, donc une ligne ou une colonne nulle donne des 0, **jamais NaN** ;
  - accepte des entiers et des listes ; `ValueError` pour `C` qui n'est pas 2-D, une entrée négative ou une somme nulle.

La première cellule de vérification essaie tes deux fonctions sur des cas visibles (la matrice chien / chat / os de la fiche, un mot jamais vu, une matrice « indépendante »), compare tes comptes sur 30 000 mots de Holmes à ceux de `scipy.sparse`, puis essaie huit refus. La seconde lance les 36 tests de l'exercice (`-k "test_cooccurrence_matrix_ or test_ppmi_"`).

Dans tes notes : pourquoi la ligne d'un mot très fréquent (« the ») domine-t-elle la matrice de comptes, et que fait la PPMI de ses cases ? Que vaut la PPMI de deux mots qui ne se rencontrent jamais, et pourquoi la PMI, elle, ne conviendrait-elle pas à une matrice creuse ?""",
       check=RELOAD + C.CHECK_22, solution=C.CHECK_22,
       after=tests_after("B2.22", "mylearn.embeddings.cooccurrence_matrix([0, 1], 2, window=1)",
                         "test_cooccurrence_matrix_ or test_ppmi_", 36)),

    Ex("B2.23", "🔨", 2, 20, "cosine_similarity_matrix et most_similar",
       "calculer tous les cosinus entre deux ensembles de vecteurs, puis chercher les plus proches voisins d'une "
       "requête, égalités et exclusions comprises.",
       "✏️ B2.2 · 0B (produit scalaire, norme) · fiche §202.9 (« Le cosinus », « Les plus proches voisins », « Les "
       "analogies »)", tracks="R, C", mylearn="embeddings.py",
       body=mylearn_howto("embeddings.py") + r"""

**Contrat**
- `cosine_similarity_matrix(A, B=None, eps=1e-12)` :
  - rend `S` de forme `(n, m)`, `S[i, j] = A[i] · B[j] / ((‖A[i]‖ + eps)(‖B[j]‖ + eps))` : `eps` s'ajoute à **chaque** norme, pas sous la racine ni au produit (testé avec `eps = 0,5` et 2) ; une ligne nulle a un cosinus de 0 avec tout ;
  - `B = None` vaut `B = A` ; accepte des listes et des entiers, sans les modifier ;
  - `ValueError` pour `A` ou `B` qui n'est pas 2-D, et pour des nombres de colonnes différents.
- `most_similar(E, query, k=10, exclude=())` :
  - rend un **tuple de deux tableaux NumPy** 1-D : les indices (entiers) et les cosinus (flottants), par cosinus décroissant ;
  - les égalités vont au **plus petit indice** (testé sur 120 lignes, dont 40 copies de trois lignes : le tri par défaut de `np.argsort` et `np.argpartition` ne le garantissent pas) ;
  - les lignes de `exclude` (des indices valides de `E`) sont retirées **avant** de garder les `k` premières ; s'il en reste moins de `k`, toutes ; si tout est exclu, deux tableaux de forme `(0,)` ;
  - une requête nulle donne des cosinus nuls, donc les `k` premiers indices ;
  - `ValueError` pour `k < 1`, une requête qui n'a pas la longueur `E.shape[1]`, ou de forme `(1, d)`.

La première cellule de vérification compare ta matrice à `sklearn.metrics.pairwise.cosine_similarity`, essaie `eps`, puis `most_similar` sur 120 lignes dont 40 égales à la requête (des égalités), avec des exclusions, trop peu de lignes, une requête nulle et trois refus. La seconde lance les 26 tests de l'exercice (`-k "test_cosine_similarity_matrix_ or test_most_similar_"`).

Dans tes notes : pourquoi normaliser les lignes **une fois**, puis faire un seul produit matriciel, plutôt que de calculer chaque cosinus séparément ? Pour l'analogie du ✏️ B2.2, quels indices passes-tu à `exclude`, et que se passerait-il sans eux ?""",
       check=RELOAD + C.CHECK_23, solution=C.CHECK_23,
       after=tests_after("B2.23", "mylearn.embeddings.cosine_similarity_matrix([[1.0, 0.0]])",
                         "test_cosine_similarity_matrix_ or test_most_similar_", 26)),

    Ex("B2.24", "📦", 2, 30, "Embeddings PPMI + SVD sur Holmes : les voisins de « Holmes », « Watson » et « letter »",
       "construire des vecteurs de mots par comptage, PPMI et SVD tronquée, avec ta librairie et scikit-learn, puis "
       "les interroger.",
       "Ex B2.22, B2.23 · ch. 12 (PCA) · fiche §202.8 (« Réduire : la SVD tronquée »), §202.9", tracks="C",
       body=r"""La carte de la fiche (§202.9) a été faite avec des vecteurs de Holmes construits par comptage. Tu refais le calcul avec **ta** librairie, et tu interroges trois mots dont la fiche ne dit rien.

**Contrat**
- `ppmi_svd_24(words, n_words=4000, window=2, dim=100, alpha=0.75)` rend `(E, stoi, itos)` :
  - le vocabulaire est `wbtext.top_vocab(words, n_words)`, les `n_words` mots les plus fréquents (`itos[0]` est le plus fréquent) ;
  - les co-occurrences viennent de **ton** `cooccurrence_matrix` (fenêtre symétrique, pondération uniforme), la PPMI lissée de **ton** `ppmi(C, alpha)` ;
  - un mot **hors du vocabulaire garde sa place** dans le texte, mais ses paires ne comptent pas : on ne le retire pas, sinon deux mots qui l'entourent deviendraient voisins. `wbtext.word_ids(words, stoi)` lui donne l'id −1, que ton `cooccurrence_matrix` refuse : à toi de trouver comment ;
  - la SVD tronquée est `TruncatedSVD(n_components=dim, algorithm="arpack", random_state=0)` de scikit-learn, sur la matrice PPMI ;
  - `E` est `(n_words, dim)` : les lignes de $\mathbf{U}_d\,\mathbf{S}_d^{1/2}$ (pas de `fit_transform` seul, qui rend $\mathbf{U}_d\,\mathbf{S}_d$), chacune ramenée à la **norme 1**.
- `neighbours_24(E, stoi, itos, word, k=8)` : la liste des `k` couples `(mot, cosinus)` les plus proches de `word`, par cosinus décroissant, sans `word` lui-même, trouvés par **ton** `most_similar`.

La cellule de vérification lance ta fonction sur les 105 863 mots de Holmes (5 à 15 s, la SVD comprise), puis compare les cosinus de « holmes », « watson » et « letter » avec les 4 000 mots à ceux d'une référence (le même calcul, compté par `scipy.sparse`, `wbtext.reference_ppmi_svd`). Elle affiche enfin les 8 voisins des trois mots.

Dans tes notes :
- les voisins de « holmes » : de quelle nature sont-ils (des noms, des mots qui l'accompagnent dans le texte, des verbes) ? Pourquoi « sherlock » est-il si proche ?
- ceux de « watson » et de « letter » : lesquels ont un sens voisin, lesquels partagent seulement des contextes (« a … of », « the … was ») ?
- « watson » n'apparaît que 81 fois, « letter » 30 fois : que dit la qualité de leurs voisins de la quantité de texte qu'il faut ?""",
       todo=C.TODO_24, check=C.CHECK_24, solution=C.SOLUTION_24 + "\n\n\n" + C.CHECK_24),

    Ex("B2.25", "📦", 3, 60, "word2vec (skip-gram, échantillonnage négatif) en PyTorch sur Verne",
       "entraîner des embeddings word2vec en PyTorch, avec les trois réglages de l'article : la loi des négatifs, leur "
       "nombre et le sous-échantillonnage.",
       "∂ B2.7 · Ex B2.24 · ch. 20 (boucle d'entraînement PyTorch) · fiche §202.7, §202.8 (« word2vec : prédire plutôt "
       "que compter »)", tracks="C",
       body=r"""Tu entraînes un skip-gram avec échantillonnage négatif (fiche §202.8) sur *Le Tour du monde en quatre-vingts jours*. La cellule suivante prépare les données : les 1 590 mots vus au moins 5 fois (`STOI_25`, `ITOS_25`, ids par fréquence décroissante), le texte `IDS_25` **sans** ses mots rares (word2vec les retire avant l'entraînement : contrairement à B2.24, ses voisins se rapprochent), et leurs comptes `COUNTS_25`.

**Contrat**
- `keep_probability_25(counts, t=1e-3)` : pour chaque mot, la probabilité de **garder** une de ses occurrences, $\min\big(1, \sqrt{t / f(w)}\big)$ avec $f(w) = \#(w) / \sum \#$ (la fiche donne la probabilité de la jeter, $1 - \sqrt{t/f}$, avec $t \approx 10^{-5}$ pour un milliard de mots ; sur un roman de 72 000 mots, $t = 10^{-3}$) ;
- `noise_distribution_25(counts, power=0.75)` : la loi des négatifs, les comptes à la puissance `power`, de somme 1 ;
- `skipgram_pairs_25(ids, window=2)` : `(centers, contexts)`, deux tableaux `int64` de même longueur, une entrée par couple de positions $(i, j)$ avec $1 \le |i - j| \le$ `window` (le mot `ids[i]` et son contexte `ids[j]`), dans l'ordre que tu veux ;
- `SkipGram25(vocab_size, dim)`, un `nn.Module` :
  - deux tables `nn.Embedding(vocab_size, dim)` : `self.center` (les $\mathbf{v}_w$), initialisée uniforme dans $[-0{,}5/d ;\ 0{,}5/d]$, et `self.context` (les $\mathbf{u}_c$), initialisée à **zéro**, comme word2vec ;
  - `forward(centers, contexts, negatives)`, avec des ids `(B,)`, `(B,)` et `(B, K)`, rend la **moyenne sur le batch** de la loss de la fiche, $-\log \sigma(\mathbf{u}_c^\top \mathbf{v}_w) - \sum_k \log \sigma(-\mathbf{u}_{n_k}^\top \mathbf{v}_w)$, un tenseur scalaire, calculée avec `F.logsigmoid` ;
- `train_word2vec_25(ids, counts, dim=100, window=2, negatives=5, t=1e-3, epochs=EPOCHS_25, batch_size=512, lr=0.01, seed=0)` rend `(model, losses)` : un `SkipGram25` entraîné par `torch.optim.Adam(lr)`, et la loss moyenne de chaque epoch (une liste de `float`). À chaque epoch : un **nouveau** sous-échantillonnage (`np.random.default_rng(seed)`, une fois au départ, avec `torch.manual_seed(seed)`), les paires du texte gardé, mélangées, des mini-batches de `batch_size` paires, et pour chaque paire `negatives` négatifs tirés selon `noise_distribution_25` (`torch.multinomial`, avec remise).

La première cellule de vérification essaie tes fonctions sur des cas visibles, compare `forward` à la formule sur des poids fixés, et lance un mini-entraînement sur 3 000 mots, dont elle compte les mots gardés à chaque epoch. La suivante entraîne ton word2vec sur tout le texte (`EPOCHS_25` = 10 epochs en mode rapide, ≈ 10 s sur deux cœurs ; 30 en mode complet), garde les vecteurs `E_25` (les lignes de `center`, de norme 1) et mesure leur qualité : pour douze paires de mots liés (« deux » et « trois », « bombay » et « calcutta »…), le rang du second parmi les voisins du premier.

Dans tes notes : au premier pas, `context` est nul : que valent alors tous les scores, et la loss d'une paire avec 5 négatifs (calcule-la, puis compare à ta loss de la première epoch) ? Que changerait un sous-échantillonnage avec $t = 10^{-5}$ sur ce petit roman (calcule la probabilité de garder « fogg ») ? Pourquoi garde-t-on la table `center`, et pas `context` ?""",
       given=C.GIVEN_25, todo=C.TODO_25, check=C.CHECK_25, solution=C.SOLUTION_25 + "\n\n\n" + C.CHECK_25,
       after=[("code", C.EXPERIMENT_25)]),

    Ex("B2.26", "🔮", 1, 10, "Prédire les voisins de « Fogg » et de « Passepartout » avant de les calculer",
       "prédire ce que capturent les voisins de deux personnages dans des vecteurs word2vec, avant de les calculer.",
       "Ex B2.25 · fiche §202.8 (l'hypothèse distributionnelle), §202.9 (« Les plus proches voisins »)", tracks="C",
       hypothesis=True,
       body=r"""Phileas Fogg (« fogg » : 685 occurrences, le 14ᵉ mot du roman) et Passepartout (451) sont les deux héros du roman. Tes vecteurs `E_25` de B2.25 vont donner leurs plus proches voisins par cosinus. **Avant d'exécuter**, écris tes deux prédictions :
- a) `fogg_26` : les **deux** plus proches voisins de « fogg » sont… `"A"` deux autres personnages (passepartout, fix, aouda…) ; `"B"` « phileas » et « mr », dans un ordre ou l'autre ; `"C"` deux verbes de parole (« dit », « répondit »…) ; `"D"` deux autres mots ;
- b) `characters_26` : `True` ou `False` : au moins 5 des 10 plus proches voisins de « passepartout » sont des noms de personnages du roman (la liste `CHARACTERS_26` de la cellule suivante).

L'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites, et attend ton word2vec de B2.25 ; elle vérifie tes prédictions à la fin.""",
       given=C.GIVEN_26, todo=C.TODO_26, solution=SOLUTION_26, record=RECORD_26,
       after=[("code", guarded(C.EXPERIMENT_26, ["fogg_26", "characters_26"],
                               "⏳ Ex B2.26 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Deux mots sont proches "
                     "quand ils ont des **contextes** semblables (§202.8) : quels contextes « fogg » partage-t-il avec "
                     "ses deux premiers voisins ? Regarde la fréquence (le rang) des voisins de « passepartout » : que "
                     "remarques-tu, et qu'est-ce que cela dit de ce que mesure le cosinus de word2vec sur un petit "
                     "texte ?")]),

    Ex("B2.27", "📈", 2, 20, "Lire une carte PCA/UMAP d'embeddings : groupes, axes et effet de la fréquence",
       "lire trois cartes en deux dimensions de vecteurs word2vec, et distinguer ce qu'elles montrent du sens et de la "
       "fréquence des mots.",
       "Ex B2.25 · ch. 12 (PCA, UMAP) · fiche §202.9 (« Visualiser »)", tracks="C",
       body=r"""La cellule suivante dessine trois cartes de tes vecteurs `E_25` de B2.25 (≈ 20 s la première fois, le temps qu'UMAP se compile) :
1. la **PCA** des 1 590 mots, chaque point coloré par le logarithme de sa fréquence (violet : rare, jaune : fréquent) ;
2. l'**UMAP** des mêmes mots (`n_neighbors=15`, distance cosinus, graine fixée), avec les mêmes couleurs ;
3. la **PCA** des seuls mots de cinq groupes (nombres, temps, lieux, transports, parties du corps), chacun avec son nom, comme la carte de la fiche.

Lis les cartes et écris :
- a) `pca_axis_27` : `True` ou `False` : sur la carte (1), la couleur change régulièrement le long de l'axe **horizontal**, les mots fréquents d'un côté et les rares de l'autre ;
- b) `umap_frequent_27` : `True` ou `False` : sur la carte (2), les mots les plus fréquents (en jaune) se rassemblent dans une région (ou quelques-unes), au lieu d'être répartis sur toute la carte.

Une fois tes deux réponses écrites, la cellule de vérification affiche des mesures faites sur les coordonnées : la corrélation de rang entre la fréquence et chaque axe de la carte (1), la part des voisins des mots fréquents qui sont eux aussi fréquents sur la carte (2), et, pour chaque groupe de la carte (3), la part de ses voisins qui sont du même groupe.

Dans tes notes : sur la carte (3), quels groupes se séparent, lesquels se mêlent, et quels mots sont « mal placés » (pense aux contextes qu'ils partagent, comme « minutes » et les nombres dans la fiche) ? Ce que tu as lu de la fréquence sur les cartes (1) et (2) : comment l'expliques-tu par ce que garde chaque méthode, les directions de plus grande variance pour la PCA, les voisinages pour UMAP ? Pourquoi ne faut-il pas lire les **distances entre** les groupes d'un UMAP (ch. 12, B1) ?""",
       given=C.GIVEN_27 + "\n\n\n" + C.EXPERIMENT_27, todo=C.TODO_27, check=C.CHECK_27,
       solution=SOLUTION_27, record=RECORD_27, after=[("solution", C.CHECK_27)]),
])

PART_E = Part("E", "Des embeddings de phrases à la livraison, et le défi",
              "Trois exercices sur ce que font les modèles d'aujourd'hui, puis le défi. Tu calcules des embeddings de "
              "phrases avec un modèle pré-entraîné pour chercher des passages de Holmes par le sens (B2.28), tu "
              "compares un vecteur statique et un vecteur contextuel du même mot (B2.29), tu livres un tokenizer avec "
              "son test de non-régression (B2.30), et tu cherches la meilleure compression de Verne avec 2 000 tokens "
              "(B2.31).\n\n"
              "B2.28 et B2.29 utilisent `all-MiniLM-L6-v2` (`sentence-transformers/all-MiniLM-L6-v2`, ≈ 90 Mo, "
              "Apache 2.0, `data/cards/pretrained.md`), chargé une fois à une révision figée, en `eval()` et sans "
              "gradient, par `wbtext.try_sentence_model` : `SM_28.tokenizer` et `SM_28.model` sont son tokenizer et son "
              "`AutoModel`. Sans connexion au Hub, ces deux exercices affichent ⏳ « à valider sur Colab ».",
              exercises=[
    Ex("B2.28", "📦", 2, 30, "Embeddings de phrases (MiniLM, mean pooling) et recherche sémantique dans Holmes",
       "calculer des embeddings de phrases par mean pooling avec un modèle pré-entraîné, et chercher des passages par "
       "le sens.",
       "Ex B2.23 · B1 (embeddings figés, révision fixée) · fiche §202.10 (« D'une suite de vecteurs à un vecteur de "
       "phrase », « Une recherche sémantique en vingt lignes »)", tracks="R, C",
       body=r"""La cellule suivante charge le modèle et découpe Holmes en 1 055 passages d'au plus 120 mots (`PASSAGES_28`, chacun avec l'indice de son conte dans `STORY_28` et `TITLES_28`), tous sous les 256 tokens que lit MiniLM. Elle calcule une fois l'**index** `INDEX_28` : l'embedding de référence de chaque passage, `(1 055, 384)`, rangé dans le cache `data/downloads/embeddings` (≈ 20 s la première fois, une lecture de fichier ensuite).

**Contrat**
- `embed_28(texts, tok, model, batch_size=32, max_length=256)` : un tableau `float32` `(N, 384)`, un embedding par texte, dans l'ordre :
  - les textes passent par mini-batches de `batch_size` dans `tok(..., padding=True, truncation=True, max_length=max_length, return_tensors="pt")`, puis dans `model`, sans gradient ;
  - le **mean pooling avec le masque d'attention** de la fiche, sur `last_hidden_state` : le padding ne compte pas ;
  - chaque embedding ramené à la **norme 1** ;
  - `max_length=256`, la longueur de la fiche modèle (`tok.model_max_length` dit 512 : piège classique de la fiche).
- `search_28(query, index, k=5)` : `(indices, cosinus)` des `k` lignes de `index` les plus proches de la requête (une `str`), que tu embeddes par `embed_28` avec `SM_28.tokenizer` et `SM_28.model`, et que **ton** `most_similar` (B2.23) cherche.

La cellule de vérification compare `embed_28` à la référence sur quatre textes de longueurs différentes (dont un de plus de 256 tokens), vérifie qu'une phrase courte a le même vecteur seule et au milieu de textes longs, puis cherche six requêtes et affiche, pour chacune, les trois passages trouvés, leur cosinus et leur conte.

Dans tes notes : pour chaque requête, le premier passage est-il du bon conte, et contient-il les mots de la requête ? Essaie une requête à toi, puis la même avec d'autres mots : le classement change-t-il ? Les cosinus vont de 0,4 à 0,65 environ : pourquoi un seuil fixe (« pertinent au-dessus de 0,5 ») serait-il fragile, d'une requête à l'autre et d'un modèle à l'autre ? Pour Verne, il faudrait `multilingual-e5-small` et ses préfixes (`wbtext.try_sentence_model("e5")`, `kind="query"` et `"passage"` : ≈ 470 Mo, à essayer sur Colab).""",
       given=C.GIVEN_28, todo=C.TODO_28, check=C.CHECK_28, solution=C.SOLUTION_28 + "\n\n\n" + C.CHECK_28),

    Ex("B2.29", "🔬", 2, 25, "Statique contre contextuel : le même mot dans deux phrases",
       "mesurer qu'un embedding statique ignore la phrase, alors qu'un embedding contextuel dépend du sens du mot dans "
       "sa phrase.",
       "Ex B2.28 · 🧠 Q11 · fiche §202.10 (« Un mot, plusieurs sens »)", tracks="C",
       body=r"""Trois mots anglais à deux sens, chacun dans six phrases (`SENTENCES_29`, prises dans Holmes ou écrites dans son style) : « bank » (la banque, la rive), « match » (l'allumette, le mariage), « ring » (la bague, la sonnerie). Pour chaque emploi, tu prends deux vecteurs de MiniLM : la ligne de sa **table d'entrée** (statique, la même dans toutes les phrases) et la sortie de son **dernier bloc** (contextuelle).

**Contrat**
- `contextual_vector_29(sentence, word, tok, model)` : un tableau NumPy `(384,)`, la moyenne des vecteurs de `last_hidden_state` sur les tokens de la **première occurrence** du mot entier `word` dans `sentence` (un mot entier n'est ni précédé ni suivi d'une lettre, d'un chiffre ou de « _ » : un trait d'union sépare, et « bank » de « bank-note » compte) (un mot de plusieurs tokens, comme « Merryweather », en a plusieurs) ; trouve-les avec les positions que rend `tok(sentence, return_offsets_mapping=True)`, sans les tokens spéciaux ; sans gradient.
- `static_vector_29(word, tok, model)` : `(384,)`, la moyenne des lignes de `model.get_input_embeddings().weight` pour les tokens de `word` (sans tokens spéciaux).
- `sense_means_29(vectors, senses)` : le couple `(moyenne des cosinus des paires d'emplois de même sens, moyenne des cosinus des paires de sens différents)`, deux `float` ; chaque paire d'emplois compte une fois.

La première cellule de vérification compare tes vecteurs à ceux de référence (`SM_29.word_vector`, `SM_29.static_vector`) et `sense_means_29` à un petit calcul visible. La suivante calcule, pour chaque mot, les cosinus des six emplois (une carte de chaleur par mot) et le tableau des moyennes, statiques et contextuelles.

Dans tes notes : que valent les cosinus statiques, et pourquoi n'est-ce pas une mesure ? Pour chaque mot, l'écart entre « même sens » et « sens différents » : lequel des trois sépare le mieux ses deux sens, lequel le moins ? Pourquoi tous les cosinus contextuels sont-ils élevés (au-dessus de 0,6), même entre deux sens ? Que te dit ce chapitre de l'entraînement de MiniLM (pour des phrases entières, §202.10) ?""",
       given=C.GIVEN_29, todo=C.TODO_29, check=C.CHECK_29, solution=C.SOLUTION_29 + "\n\n\n" + C.CHECK_29),

    Ex("B2.30", "🛠️", 2, 20, "Sauvegarder un tokenizer en JSON et écrire son test de non-régression",
       "livrer un tokenizer comme un artefact versionné, et le protéger par un test de non-régression qui échoue dès "
       "qu'il change.",
       "Ex B2.17 · fiche §202.4, §202.5 (« Les tokenizers des modèles, lus dans leur `tokenizer.json` ») · ch. 0A (tests "
       "pytest)", tracks="R, C",
       body=r"""Un modèle est entraîné avec **un** tokenizer : si ses fusions changent, ses identifiants changent, et le modèle lit n'importe quoi sans le moindre message. En entreprise, le tokenizer est donc livré comme un fichier, à côté des poids, et un **test de non-régression** vérifie qu'il redonne toujours les mêmes identifiants sur un texte de référence figé.

La cellule suivante prépare `TOK_30`, un `BPETokenizer` octet de **ta** librairie, 1 000 tokens appris sur `VERNE_TRAIN` (par `fast_byte_bpe`), le texte de référence `REFERENCE_TEXT_30` (un passage de la validation de Verne, puis « € », un emoji et des idéogrammes), et le dossier de livraison `DELIVERY_30` (à côté de ton notebook dans `mon_travail`, sinon un dossier temporaire).

**Contrat**
- `deliver_30(tok, folder, reference_text)` : écrit `folder/tokenizer.json` (par `tok.save`, fourni) et `folder/golden.json`, un dictionnaire JSON qui contient au moins `"text"` (le texte de référence) et `"ids"` (ses identifiants figés), et ce que ton test veut vérifier d'autre ; rend le couple de leurs chemins `(tokenizer.json, golden.json)`.
- `test_tokenizer_regression_30()` : un test pytest qui recharge le tokenizer par `BPETokenizer.load(TOKENIZER_PATH)`, puis vérifie avec des `assert` que l'encodage du texte de `GOLDEN_PATH` redonne les ids figés, que leur décodage redonne le texte, et que la taille du vocabulaire (`len(tok.vocab_)`, le nombre de tokens du tokenizer) n'a pas changé. `BPETokenizer`, `TOKENIZER_PATH`, `GOLDEN_PATH`, `json` et `Path` sont fournis au test, qui tourne dans un fichier à part (`wb.run_pytest`) : il ne doit pas lire d'autre variable du notebook.

La cellule de vérification livre `TOK_30`, relit les deux fichiers, puis lance ton test sur le tokenizer livré (il doit passer) et sur trois copies abîmées de `tokenizer.json` (il doit échouer sur chacune).

Dans tes notes : quels fichiers committerais-tu à côté des poids d'un modèle, et pourquoi le texte de référence contient-il des accents, un emoji et des idéogrammes ? Une des trois copies abîmées donne les mêmes ids pour presque tous les textes : laquelle, et qu'est-ce qui, dans ton test, la trahit ? Quand faut-il **régénérer** `golden.json`, et qui doit le décider ?""",
       given=C.GIVEN_30, todo=C.TODO_30, check=C.CHECK_30, solution=C.SOLUTION_30 + "\n\n\n" + C.CHECK_30),

    Ex("B2.31", "🏆", 4, 90, "Défi : la meilleure compression sans perte de Verne avec 2 000 tokens, en moins de 2 min de CPU",
       "trouver, avec ce que le chapitre a montré, un tokenizer de 2 000 tokens qui compresse Verne nettement mieux que "
       "le BPE de B2.20.",
       "Ex B2.20, B2.21 · fiche §202.3 à §202.6 · tout le chapitre", tracks="C",
       body=r"""En B2.20, un BPE octet de 2 000 tokens avec le pré-tokenizer `'gpt2'`, appliqué par rang, a fait **3,001 octets par token** sur `VERNE_VAL`. Le défi : **au moins 3,20**, toujours **sans perte**, avec **2 000 tokens au plus** et en **moins de 2 minutes** de CPU.

Écris `method_31(text)` : elle reçoit `VERNE_TRAIN` et rend un tokenizer, un objet qui a :
- `vocab_` : un dictionnaire `{id: bytes}` d'au plus 2 000 entrées, tokens spéciaux compris (celui de ton `BPETokenizer` convient) ;
- `encode(text)` : la liste des ids du texte, tous des clés de `vocab_` ;
- `decode(ids)` : le texte.

Tout est permis : ton `BPETokenizer`, la bibliothèque `tokenizers`, d'autres pré-tokenizers, une autre façon d'appliquer un vocabulaire, une idée à toi. `evaluate_31(method_31)` (cellule suivante) vérifie quatre règles :
1. **2 000 tokens au plus** dans `vocab_` ;
2. **sans perte** : les octets des tokens, mis bout à bout, redonnent exactement la validation, et `decode` aussi : un id vaut toujours les mêmes octets, quel que soit son voisinage ;
3. **rien de la validation** : ta méthode n'apprend que sur le texte qu'elle reçoit. Pendant la course, toutes les variables globales qui contiennent la validation (`VERNE_VAL` et celles des exercices précédents) valent un leurre, `wbtext` ne rend que le texte d'entraînement de Verne, et un token de 4 octets ou plus qui serait dans la validation mais nulle part dans `VERNE_TRAIN` est signalé. Puis un **contrôle** : la même méthode, apprise sur les 8/9 de `VERNE_TRAIN`, est mesurée sur le dernier neuvième ; elle doit y gagner autant sur le BPE de B2.20 que sur la validation (2 points d'écart au plus), et aucun de ses tokens de 4 octets ou plus ne doit venir du neuvième tenu à l'écart ;
4. **moins de 2 minutes** pour l'entraînement et l'encodage, mesurés ensemble et **à froid** : sur la validation, et aussi sur le contrôle, un texte que ta méthode n'a jamais vu (un entraînement fait d'avance, dans une autre cellule, ne compte pas).

Puis elle affiche ton score : les octets UTF-8 de la validation divisés par son nombre de tokens. Règle d'honneur : la validation sert à mesurer, jamais à apprendre.

Dans tes notes : tes idées, leur gain sur la validation et leur temps ; ce que coûterait ta méthode à un modèle de langue (la longueur des séquences, la table d'embedding, les tokens rares, B2.20) ; et pourquoi un tokenizer qui compresse mieux n'est pas toujours un meilleur tokenizer.""",
       given=C.GIVEN_31, todo=C.TODO_31, check=C.CHECK_31, solution=C.SOLUTION_31 + "\n\n\n" + C.CHECK_31),
])
