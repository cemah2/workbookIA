"""Parts A to C of the chapter B2 notebook (tokenization phase): statements and cells (used by build_chB2.py).

Parts D and E (embeddings, B2.22 to B2.31) come at the next generation step (build_chB2.NEXT_SESSION).
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part, guarded

import chB2_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB2.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}

for _row in HISTORY.get("bug18", {}).get("versions", []):      # 🐛 B2.18 replayed on its 8 versions (history_chB2.py)
    if _row["first_failure"] != _row["expected"]:
        raise ValueError(f"🐛 B2.18: the version with the fixes {_row['fixed']!r} stops at the wrong point: {_row}")

P14 = HISTORY.get("p14", {}).get("answers", {"costliest": "D", "ratio": "B", "bert_emoji": "A"})


def _letters(answer: str, hints: dict) -> str:
    """The mistakes dict (as source code) of a multiple choice: one message for each other letter."""
    return "{" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items() if letter != answer) + "}"


RECORD_14 = (
    f'wb.record("B2.14a", "{P14["costliest"]}", mistakes=' + _letters(P14["costliest"], {
        "A": "regarde la colonne de Holmes dans le tableau des octets par token de GPT-2 : est-ce la plus petite valeur ?",
        "B": "regarde la colonne de Verne dans le tableau des octets par token de GPT-2 : est-ce la plus petite valeur ?",
        "C": "regarde la colonne des nombres dans le tableau des octets par token de GPT-2 : est-ce la plus petite valeur ?",
        "D": "regarde la colonne des emoji dans le tableau des octets par token de GPT-2 : est-ce la plus petite valeur ?"})
    + ")\n"
    f'wb.record("B2.14b", "{P14["ratio"]}", mistakes=' + _letters(P14["ratio"], {
        "A": "relis le rapport de CamemBERT affiché sous les tableaux : est-il sous 0,9 ?",
        "B": "relis le rapport de CamemBERT affiché sous les tableaux : est-il entre 0,9 et 1,1 ?",
        "C": "relis le rapport de CamemBERT affiché sous les tableaux : dépasse-t-il 1,1 ? Sur quels textes CamemBERT "
             "a-t-il appris ses tokens ?"})
    + ")\n"
    f'wb.record("B2.14c", "{P14["bert_emoji"]}", mistakes=' + _letters(P14["bert_emoji"], {
        "A": "regarde la ligne de BERT dans le tableau des tokens, colonne des emoji",
        "B": "regarde la ligne de BERT, colonne des emoji : combien de tokens, et lesquels (la dernière ligne affichée) ?",
        "C": "regarde la ligne de BERT, colonne des emoji : combien de tokens, et lesquels (la dernière ligne affichée) ? "
             "Relis aussi ce que fait WordPiece d'un mot qu'il ne sait pas découper (§202.5)"})
    + ")")
SOLUTION_14 = (f'costliest_14, ratio_14, bert_emoji_14 = "{P14["costliest"]}", "{P14["ratio"]}", "{P14["bert_emoji"]}"')

RELOAD = C.RELOAD


def mylearn_howto(file: str) -> str:
    return (f"> **Mode d'emploi mylearn** : ouvre `mon_travail/mylearn/{file}` (créé par `python tools/start_chapter.py "
            "B2`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de l'exercice par ton code et "
            "**enregistre**. La bibliothèque standard seulement. La première cellule de vérification recharge ta "
            "librairie et l'essaie sur des cas visibles ; la seconde lance les tests de l'exercice. "
            f"`python -m pytest tests/test_chb2_{file[:-3]}.py -q` les lance tous dans un terminal.")


def ref_check(check: str) -> str:
    """The solutions version of a check cell: the tests run on the reference."""
    return check.replace('run_pytest_group(TEST_TOK, "', 'run_pytest_group(TEST_TOK, impl="ref", keyword="')


def tests_after(ex_id: str, probe: str, keyword: str, n_tests: int, what: str = "tes fonctions") -> list:
    """The second check cell of a 🔨: the tests of the exercise (explicit test file, -k group)."""
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_TOK, "{keyword}")   # {n_tests} tests\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')
    return [("check", cell), ("solution", ref_check(cell))]


COMMON_TOK = (
    "**Contrat commun de `tokenizer.py`** (les docstrings, plus ces précisions que vérifient les tests) :\n"
    "- les identifiants : d'abord l'alphabet, `0 … n_base_ − 1` (les 256 octets au niveau octet, ou les caractères "
    "des pré-tokens triés par point de code au niveau caractère) ; puis une fusion par identifiant, dans l'ordre "
    "d'apprentissage (la fusion de rang $r$ crée l'identifiant `n_base_ + r`) ; puis les tokens spéciaux ;\n"
    "- tes fonctions ne modifient jamais leurs entrées ; une entrée invalide lève `ValueError`, jamais `KeyError` ni "
    "`AttributeError` ;\n"
    "- la bibliothèque standard seulement ; `pre_tokenize`, `save` et `load` sont fournis, déjà écrits.")

# ---------------------------------------------------------------------------------------------------------------------
PART_A = Part("A", "Des tokenizers réels : ce que coûte un texte",
              "Avant d'écrire ton propre BPE, tu prends en main trois tokenizers de modèles réels et tu mesures ce que "
              "leur coûtent Holmes et Verne : leur nombre de tokens, leur fertilité (tokens par mot) et leur "
              "compression (octets par token, fiche §202.6). La cellule suivante charge les outils de la phase "
              "« tokenisation » (parties A à C) :\n"
              "- `EXCERPT_H` et `EXCERPT_V` : les 50 000 caractères `[100000:150000]` de chaque roman, figés ;\n"
              "- `HOLMES_TRAIN`, `HOLMES_VAL`, `VERNE_TRAIN`, `VERNE_VAL` : chaque roman coupé à 90 % / 10 %, au "
              "début d'une ligne et sans mélange (un tokenizer appris sur l'entraînement ne voit jamais la "
              "validation) ;\n"
              "- `TOKENIZER_REPOS` : le dépôt du Hugging Face Hub et la révision (un commit fixé) de six tokenizers "
              "(`data/cards/pretrained.md`) ; seuls leurs fichiers de tokenizer se téléchargent, quelques Mo ;\n"
              "- `n_bytes`, `reference_char_bpe` (le BPE de la bibliothèque `tokenizers`, pour les vérifications), "
              "`truncated_bpe` (partie C).\n\n"
              "**Données** : *The Adventures of Sherlock Holmes* (A. Conan Doyle, Project Gutenberg n° 1661, "
              "`data/cards/holmes.md`) et *Le Tour du monde en quatre-vingts jours* (J. Verne, n° 800, "
              "`data/cards/verne.md`), dans le domaine public ; tu les as déjà découpés en caractères et en mots au "
              "ch. 22. Sans connexion au Hub, les cellules qui ont besoin d'un tokenizer réel affichent ⏳ « à valider "
              "sur Colab » et le reste du notebook continue.",
              given=C.TOOLS, exercises=[
    Ex("B2.13", "📦", 2, 20, "Tokenizers réels : GPT-2, BERT et un tokenizer multilingue sur Holmes et Verne",
       "charger trois tokenizers réels à une révision fixée, et mesurer ce que leur coûte un texte anglais et un "
       "texte français.",
       "ch. 22 (vocabulaire et encodage) · fiche §202.1, §202.5 (« Utiliser un tokenizer avec `transformers` »), "
       "§202.6 (« Deux rapports »)", tracks="R, C",
       body=r"""**Contrat**
- `load_tokenizers_13()` : un dictionnaire de trois tokenizers de `transformers`, chacun chargé par `AutoTokenizer.from_pretrained(dépôt, revision=révision)` avec le dépôt et la révision de `TOKENIZER_REPOS` :
  - `"gpt2"`, le BPE octet de GPT-2 ;
  - `"bert"`, le WordPiece de `bert-base-uncased` ;
  - `"e5"`, le tokenizer multilingue de `multilingual-e5-small` (celui de XLM-R).
- `measure_13(tok, text)` : un dictionnaire de quatre mesures :
  - `"tokens"` : le nombre de tokens du texte **seul**, sans les tokens spéciaux que certains tokenizers ajoutent autour d'un texte ;
  - `"words"` : le nombre de mots, `len(text.split())` ;
  - `"fertility"` : tokens / mots ;
  - `"bytes_per_token"` : octets UTF-8 du texte / tokens, la compression.

La cellule de vérification compare tes trois tokenizers à ceux de référence (`wbtext.load_tokenizer`) : leur type, leur taille et les identifiants d'une phrase d'essai. Elle mesure ensuite chacun sur `EXCERPT_H` et `EXCERPT_V` et affiche le tableau des six lignes. Quatre valeurs sont vérifiées, calculées par ta fonction (rien à taper) :
- a) le nombre de tokens de GPT-2 sur l'extrait de Holmes ;
- b) celui de BERT sur l'extrait de Verne ;
- c) la fertilité de e5 sur Holmes ;
- d) la compression de e5 sur Verne.

Dans tes notes : quel tokenizer est le plus économe sur Holmes, et sur Verne ? BERT *uncased* paraît économe sur Verne : `bert.decode(bert(phrase, add_special_tokens=False)["input_ids"])` redonne-t-il une phrase accentuée de Verne ? Que perd-il ? Affiche les tokens d'une même phrase avec les trois tokenizers (`tok.tokenize(phrase)`) : que signalent « Ġ », « ## » et « ▁ » ?""",
       todo=C.TODO_13, check=C.CHECK_13, solution=C.SOLUTION_13 + "\n\n\n" + C.CHECK_13, record=C.RECORD_13),

    Ex("B2.14", "🔮", 1, 10, "Prédire : quel texte coûte le plus de tokens, Holmes, Verne, des nombres ou des emoji ?",
       "prédire, en ordres de grandeur, ce que coûtent en tokens un texte anglais, un texte français, des nombres et "
       "des emoji.",
       "Ex B2.13 · fiche §202.2 (octets UTF-8), §202.4, §202.5 (WordPiece), §202.6 (« Les chiffres et les lettres »)",
       tracks="R, C", hypothesis=True,
       body=r"""La cellule suivante prépare deux textes de plus, à côté des deux extraits de B2.13 :
- `NUMBERS_14` : 2 000 nombres de 1 à 5 chiffres, séparés par des espaces (11 768 octets) ;
- `EMOJI_14` : 1 200 emoji collés, sans espace (6 000 octets ; « 👍🏽 » et « ❤️ » font deux points de code chacun).

L'expérience compte les tokens des quatre textes, tokens spéciaux exclus, avec six tokenizers : GPT-2, BERT, e5, CamemBERT, Qwen3 et SmolLM2. **Avant d'exécuter**, écris tes trois prédictions :
- a) `costliest_14` : avec GPT-2, le texte qui a le **moins** d'octets par token (celui qui coûte le plus cher pour sa taille) : `"A"` Holmes, `"B"` Verne, `"C"` les nombres, `"D"` les emoji ;
- b) `ratio_14` : avec CamemBERT, le rapport entre les tokens de `EXCERPT_V` et ceux de `EXCERPT_H`, deux textes de 50 000 caractères : `"A"` moins de 0,9, `"B"` de 0,9 à 1,1, `"C"` plus de 1,1 ;
- c) `bert_emoji_14` : le nombre de tokens de BERT (`bert-base-uncased`) pour `EMOJI_14` : `"A"` de 1 à 10, `"B"` de 11 à 2 000, `"C"` plus de 2 000.

La cellule d'expérience n'affiche rien tant que tes trois prédictions ne sont pas écrites ; elle les vérifie à la fin.""",
       given=C.GIVEN_14, todo=C.TODO_14, solution=SOLUTION_14, record=RECORD_14,
       after=[("code", guarded(C.EXPERIMENT_14, ["costliest_14", "ratio_14", "bert_emoji_14"],
                               "⏳ Ex B2.14 : écris d'abord tes trois prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Compare les six "
                     "tokenizers texte par texte : lesquels traitent les chiffres un par un (fiche §202.6), et que "
                     "change ce choix pour les nombres ? Sur quel texte les six tokenizers sont-ils le plus "
                     "inégaux, et pourquoi ?")]),
])

PART_B = Part("B", "Ton BPE : tokenizer.py, puis un tokenizer à réparer et la comparaison à tokenizers",
              "Tu écris dans `mylearn/tokenizer.py` un BPE complet, au niveau caractère et au niveau octet : compter "
              "et fusionner les paires (B2.15), apprendre les fusions (B2.16), encoder par rang et décoder sans perte "
              "(B2.17). Tu répares ensuite un petit tokenizer qui abîme les accents (B2.18), puis tu compares ton BPE "
              "à la bibliothèque `tokenizers`, écrite en Rust : mêmes fusions, autre vitesse (B2.19). Ton "
              "`BPETokenizer` resservira en B3 (un mini-GPT sur des tokens BPE).\n\n" + COMMON_TOK,
              exercises=[
    Ex("B2.15", "🔨", 2, 20, "get_pair_counts et merge_pair",
       "compter les paires de tokens voisins d'un corpus pondéré, puis fusionner une paire dans une suite "
       "d'identifiants.",
       "✏️ B2.3 · fiche §202.3 (« L'entraînement », « Ta librairie : `tokenizer.py` »)", tracks="R, M, C",
       mylearn="tokenizer.py",
       body=mylearn_howto("tokenizer.py") + r"""

**Contrat**
- `get_pair_counts(words)` :
  - `words` est un corpus pré-tokenisé, `{mot en tuple d'identifiants: fréquence}` ; tout mapping convient, en lecture seule (les tests en passent un `MappingProxyType` et un `Counter`), et il n'est pas modifié ;
  - rend un `dict` (un `Counter` convient) `{(gauche, droite): compte}` des paires d'identifiants voisins, chaque occurrence comptant la fréquence de son mot ;
  - les paires qui se chevauchent comptent : `(1, 1, 1)` contient deux fois `(1, 1)` ; aucune paire ne franchit la frontière de deux mots ;
  - une fréquence négative lève `ValueError` ; un mot de fréquence 0 n'est pas testé (le plus naturel : il ne produit aucune paire).
- `merge_pair(ids, pair, new_id)` :
  - remplace chaque occurrence de `pair` par `new_id`, de gauche à droite et sans chevauchement (comme `str.replace`) ;
  - rend toujours une **liste neuve** : jamais l'entrée elle-même, même sans aucune occurrence, et jamais un tuple ;
  - accepte un tuple, sans le modifier ; une paire qui n'a pas exactement deux identifiants lève `ValueError`.

La première cellule de vérification essaie tes deux fonctions sur des cas visibles, puis compte les paires des mots de l'extrait de Verne (chaque caractère écrit par son point de code) et les compare à un comptage direct ; elle essaie enfin deux refus. La seconde lance les 42 tests de l'exercice (`-k "test_get_pair_counts_ or test_merge_pair_"`).

Dans tes notes : l'extrait de Verne a 7 948 mots, mais combien de mots distincts (la cellule l'affiche) ? Pourquoi compter les paires sur les mots distincts, avec leur fréquence, plutôt que sur le texte entier ?""",
       check=RELOAD + C.CHECK_15, solution=C.CHECK_15,
       after=tests_after("B2.15", "mylearn.tokenizer.get_pair_counts({(1, 2): 1})",
                         "test_get_pair_counts_ or test_merge_pair_", 42)),

    Ex("B2.16", "🔨", 3, 50, "BPETokenizer.fit : apprendre les fusions, au niveau caractère puis octet",
       "apprendre les fusions d'un BPE au niveau caractère et au niveau octet, avec le départage et les conditions "
       "d'arrêt de `tokenizers`.",
       "Ex B2.15 · ✏️ B2.3 · fiche §202.3 (encadré 🕰️ sur le départage), §202.4 (tokens spéciaux)", tracks="R, M, C",
       mylearn="tokenizer.py",
       body=mylearn_howto("tokenizer.py") + r"""

**Contrat** de `BPETokenizer.fit(texts)` :
- **Entrée et sortie** : il rend `self` et pose les quatre attributs `n_base_`, `merges_`, `vocab_` et `special_ids_`. Une seule `str` est **un** texte ; tout itérable de `str` convient, un générateur aussi, et la liste reçue n'est pas modifiée.
- **Corpus** : chaque texte est découpé par `pre_tokenize(texte, self.pre_tokenizer, tokens spéciaux)`. Les tokens spéciaux, `unk_token` compris, sont **retirés avant tout comptage** : leurs caractères n'entrent pas dans l'alphabet, et les fusions sont apprises sur le reste. Des pré-tokens identiques sont comptés une fois, avec leur fréquence.
- **Alphabet** : au niveau octet, `n_base_ = 256` et `vocab_[i] = bytes([i])`. Au niveau caractère, les caractères distincts des pré-tokens (sans l'espace avec `'whitespace'`), triés par point de code ; `vocab_[i]` contient leurs octets UTF-8.
- **Fusions** : chaque étape fusionne la paire la plus fréquente, et à égalité la plus petite paire `(gauche, droite)`. La fusion de rang $r$ crée l'identifiant `n_base_ + r`, avec `vocab_[n_base_ + r] = vocab_[a] + vocab_[b]`. `merges_` est une liste de **tuples** `(a, b)`, dans l'ordre d'apprentissage.
- **Arrêt** : l'entraînement s'arrête quand `n_base_ + fusions + spéciaux` atteint `vocab_size`, quand il ne reste aucune paire, ou quand la meilleure paire apparaît moins de `min_frequency` fois (`min_frequency = 0` est permis). `vocab_size` compte **tous** les tokens spéciaux, `unk_token` compris quand il est ajouté : avec `vocab_size = n_base_ + spéciaux`, aucune fusion.
- **Tokens spéciaux** : `special_ids_` les place après les fusions, dans l'ordre donné, puis `unk_token` s'il n'y est pas déjà (une seule fois). `vocab_` contient aussi leurs octets UTF-8, donc `len(vocab_) = n_base_ + fusions + spéciaux`. Un token spécial donné deux fois n'est pas testé.
- **Erreurs** : `ValueError` pour un `level` ou un `pre_tokenizer` inconnu, un `unk_token` au niveau octet, et `vocab_size < n_base_ +` spéciaux (au niveau caractère, une fois le corpus lu).

Les tests comparent tes fusions à celles de `tokenizers` aux deux niveaux, sur des extraits de Holmes et de Verne. Une version qui recompte toutes les paires à chaque fusion les passe en quelques secondes ; sur un roman entier, elle met de 30 s à 2 min pour 2 000 tokens, que tu sentiras en B2.19 (l'indice 2 dit comment aller plus vite).

La première cellule de vérification essaie `fit` sur trois petites phrases, puis compare tes 400 tokens de l'extrait de Verne à ceux de `BpeTrainer`, essaie le niveau octet avec deux tokens spéciaux, et trois refus. La seconde lance les 49 tests de l'exercice (`-k "test_pre_tokenize_ or test_bpe_fit_"`, dont les 5 de `pre_tokenize`, fourni).

Dans tes notes : pourquoi l'alphabet est-il trié par point de code, et que deviendrait la comparaison à `tokenizers` s'il suivait l'ordre d'apparition des caractères ? Pourquoi `vocab_size` compte-t-il les tokens spéciaux (pense à la table d'embedding d'un modèle, §202.7) ?""",
       check=RELOAD + C.CHECK_16, solution=C.CHECK_16,
       after=tests_after("B2.16", 'mylearn.tokenizer.BPETokenizer(vocab_size=258).fit("aa aa")',
                         "test_pre_tokenize_ or test_bpe_fit_", 49, "ta méthode fit")),

    Ex("B2.17", "🔨", 3, 40, "encode et decode : fusions appliquées par rang, aller-retour sans perte",
       "encoder un texte en appliquant les fusions par rang, puis le décoder sans perte au niveau octet.",
       "Ex B2.16 · ✏️ B2.4 · fiche §202.3 (« Encoder un mot nouveau : les fusions par rang »), §202.4",
       tracks="R, M, C", mylearn="tokenizer.py",
       body=mylearn_howto("tokenizer.py") + r"""

**Contrat**
- `encode(text)` rend une `list` d'`int` (`[]` pour un texte vide) :
  - toute clé de `special_ids_` écrite dans le texte devient son identifiant, `unk_token` compris : « <unk> » tapé tel quel donne l'identifiant de `<unk>` ;
  - dans chaque autre pré-token, jamais entre deux, on part des identifiants de base (octets UTF-8, ou caractères), puis on fusionne la paire voisine de **plus petit rang** parmi celles de `merges_`, et ainsi de suite jusqu'à ce qu'aucune paire voisine ne soit une fusion apprise ;
  - au niveau caractère, chaque caractère inconnu donne l'identifiant de `unk_token`, un par caractère.
- `decode(ids)` recolle les octets de tous les tokens, **puis** décode l'UTF-8 avec `errors="replace"` : `[0xC3, 0xA9]` donne « é », et `[0xC3]` seul donne « � ». Il accepte tout entier, Python ou NumPy, qui est une clé de `vocab_`. Avec `'whitespace'`, les espaces disparaissent.
- `tokenize(text)` rend les tokens de `encode(text)` en `str`, « � » pour un morceau de caractère.
- `ValueError`, jamais `AttributeError` ni `KeyError`, pour :
  - un tokenizer non entraîné, dans `encode` comme dans `decode` ;
  - un caractère inconnu sans `unk_token` ;
  - un identifiant inconnu passé à `decode`, −1 compris (ce n'est pas un indice Python).
- `encode` et `decode` ne lisent que les hyperparamètres et les quatre attributs appris : un cache privé posé par `fit` casse le tokenizer rechargé par `load` (testé).

Les tests comparent tes identifiants à ceux de `tokenizers`, vérifient l'aller-retour sans perte sur environ 300 chaînes Unicode tirées au hasard (emoji, idéogrammes, accents combinants, espaces multiples), et que ton nombre de tokens reste à moins de 2 % de celui d'un BPE `ByteLevel` de même taille.

La première cellule de vérification compare ton encodage d'un passage de Verne à celui de `tokenizers`, essaie `<unk>`, l'aller-retour d'un texte accentué avec un emoji et des idéogrammes, un octet seul, `tokenize`, et quatre refus. La seconde lance les 36 tests de l'exercice (`-k "test_bpe_encode_ or test_bpe_decode_ or test_bpe_tokenize_ or test_bpe_save_load_"`).

Dans tes notes : avec `'whitespace'`, que redonne `decode(encode("le chat dort"))`, et pourquoi le pré-tokenizer `'gpt2'` règle-t-il ce problème ? Que rend `tokenize("été")` au niveau octet avec seulement 256 tokens, et pourquoi ?""",
       check=RELOAD + C.CHECK_17, solution=C.CHECK_17,
       after=tests_after("B2.17", 'mylearn.tokenizer.BPETokenizer(vocab_size=258).fit("aa aa").encode("aa")',
                         "test_bpe_encode_ or test_bpe_decode_ or test_bpe_tokenize_ or test_bpe_save_load_", 36,
                         "tes méthodes fit et encode")),

    Ex("B2.18", "🐛", 2, 25, "Le tokenizer qui abîme les accents : trois bugs d'Unicode et de fusions",
       "trouver et corriger les trois erreurs d'un petit tokenizer BPE, guidé par des diagnostics enchaînés.",
       "Ex B2.17 · fiche §202.2 (normalisation), §202.3, §202.4 · tableau des pièges ⚠️", thread="Verne",
       tracks="R, C",
       body=r"""Un collègue a écrit un petit tokenizer BPE au niveau octet, avec les pré-tokens de GPT-2, en cinq fonctions : `clean_18`, `merge_18`, `train_18`, `encode_18` et `decode_18`. Il tourne, mais il abîme le français. Le code contient **trois** erreurs. Corrige-le **sur place**, dans sa cellule, sans changer les noms, les signatures ni ce que rendent les fonctions.

Ce qu'il devrait faire, comme ton `BPETokenizer` de B2.17 :
- `train_18(text, vocab_size)` rend `(merges, words)` : les fusions dans l'ordre d'apprentissage, et chaque pré-token du texte avec ses identifiants à la fin de l'entraînement ;
- `encode_18(text, merges)` encode par rang ;
- `decode_18(ids, merges)` redonne le texte exact, et un octet seul devient « � ».

La cellule de vérification lance `diagnose_18` : un entraînement de 300 tokens sur les 120 000 premiers caractères de `VERNE_TRAIN` (environ 1 s), puis trois diagnostics enchaînés, chacun ne s'exécutant que si le précédent passe, car une erreur peut en cacher une autre :
- `decode` redonne-t-il un texte d'essai ?
- comment « é » est-il encodé ?
- `encode` découpe-t-il les mots du texte d'entraînement comme l'entraînement les a laissés ?

Elle tourne dès le premier « Tout exécuter » : ses messages sont ton point de départ. Une fois les trois points passés, la cellule suivante mesure ta version réparée sur la validation de Verne.

Dans tes notes : les trois erreurs, et pour chacune le symptôme qu'elle produit seule. Laquelle aurait laissé passer un test d'aller-retour sur un texte anglais sans accent ni emoji ? Laquelle ne change jamais le texte rendu par `decode`, seulement le nombre de tokens ?""",
       given=C.GIVEN_18, todo=C.TODO_18, check=C.CHECK_18, solution=C.SOLUTION_18 + "\n\n\n" + C.CHECK_18,
       after=[("code", C.EXPERIMENT_18)]),

    Ex("B2.19", "📦", 2, 25, "Ton BPE contre la bibliothèque tokenizers : mêmes fusions, autre vitesse",
       "entraîner un BPE avec la bibliothèque `tokenizers`, lire ses fusions dans sa description JSON, et vérifier "
       "que ton BPE apprend exactement les mêmes.",
       "Ex B2.17 · fiche §202.3 (encadré 🕰️ sur le départage), §202.5 (« Les tokenizers des modèles, lus dans leur "
       "`tokenizer.json` », « Entraîner son propre tokenizer avec `tokenizers` »)", tracks="M, C",
       body=r"""**Contrat**
- `train_hf_19(text, vocab_size, min_frequency=2)` : un `Tokenizer` de `tokenizers`, de modèle `models.BPE()` et de pré-tokenizer `pre_tokenizers.Whitespace()`, entraîné sur `text` par un `BpeTrainer` de ces `vocab_size` et `min_frequency`, sans token spécial ni barre de progression ;
- `hf_merges_19(hf_tok)` : ses fusions, en liste de tuples `(id gauche, id droite)` d'`int`, dans l'ordre d'apprentissage. Lis-les dans sa description JSON, `hf_tok.to_str()`, qui est le contenu de son `tokenizer.json`, avec son vocabulaire `hf_tok.get_vocab()`.

La cellule de vérification entraîne ta fonction sur `VERNE_TRAIN` avec 2 000 tokens. Elle vérifie trois valeurs, calculées par tes fonctions (rien à taper) :
- a) la taille de l'alphabet de base, la taille du vocabulaire moins le nombre de fusions ;
- b) le rang de la fusion qui crée le token « Passepartout » ;
- c) le nombre de tokens de `VERNE_VAL`.

Elle entraîne ensuite **ton** `BPETokenizer` de la même façon (niveau caractère, `'whitespace'`, 2 000 tokens), compare ses fusions et ses identifiants à ceux de `tokenizers`, et affiche les deux temps d'entraînement. Ce temps est une mesure, pas une réponse : il dépend de la machine, et de ta façon de compter les paires (de 1 s à deux minutes environ).

Dans tes notes : combien de fois ton entraînement est-il plus lent que celui de `tokenizers`, et d'où vient l'écart (le langage, l'algorithme) ? Que dit le rang de « Passepartout » de sa fréquence dans le roman ? Pourquoi la comparaison exacte n'est-elle possible qu'avec la même règle de départage et le même ordre de l'alphabet ?""",
       todo=C.TODO_19, check=C.CHECK_19, solution=C.SOLUTION_19 + "\n\n\n" + C.CHECK_19, record=C.RECORD_19),
])

PART_C = Part("C", "Expériences : la taille du vocabulaire et la langue d'entraînement",
              "Deux expériences avec ton `BPETokenizer` au niveau octet et le pré-tokenizer `'gpt2'` (sans perte). "
              "D'abord la taille du vocabulaire, de 300 à 8 000 tokens (B2.20) ; puis ce que coûte un tokenizer "
              "appris sur une langue quand on l'applique à l'autre (B2.21). Les mesures se font toujours sur la "
              "**validation**, que les tokenizers n'ont jamais vue.\n\n"
              "Deux outils (dans la cellule des outils) :\n"
              "- `fast_byte_bpe(texts, vocab_size)` rend un `BPETokenizer` de **ta** librairie dont les fusions sont "
              "apprises par `tokenizers`, en Rust : exactement celles de ton `fit` (B2.19 l'a montré au niveau "
              "caractère, B2.20 le vérifie au niveau octet), en moins d'une seconde pour tout Verne, là où une "
              "version Python qui recompte toutes les paires à chaque fusion met une à quatre minutes. Ton "
              "`encode` fait tout le reste ;\n"
              "- `truncated_bpe(tok, taille)` garde les `taille − 256` premières fusions d'un tokenizer : les "
              "fusions s'apprennent une à une, et les premières sont les mêmes quelle que soit la taille finale.\n",
              exercises=[
    Ex("B2.20", "🔬", 3, 40, "Taille du vocabulaire : compression et entropie par token, de 300 à 8 000 tokens",
       "mesurer ce qu'apporte un vocabulaire plus grand, en compression et en information par token, pour choisir "
       "une taille.",
       "Ex B2.19 · ch. 6 (entropie, code de Huffman) · fiche §202.3, §202.6", tracks="M, C",
       body=r"""Quelle taille de vocabulaire choisir ? Plus de tokens raccourcissent les textes, mais agrandissent la table d'embedding et rendent chaque token plus rare. Tu mesures sur `VERNE_VAL`, pour six tailles de 300 à 8 000 :
- la **compression**, en octets par token ;
- l'**entropie** de la distribution des tokens, en bits par token : $H = -\sum_t p_t \log_2 p_t$, où $p_t$ est la fréquence de l'identifiant $t$ dans le texte encodé (ch. 6) ;
- les **bits par octet**, $H$ divisé par la compression : ce que coûterait chaque octet du texte avec un code de Huffman idéal sur les tokens, si on les supposait indépendants (ch. 6).

**Contrat**
- `compression_20(tok, text)` : le nombre d'octets UTF-8 de `text` divisé par le nombre d'identifiants de `tok.encode(text)` ;
- `token_entropy_20(tok, text)` : l'entropie en bits (logarithme de base 2) des fréquences des identifiants de `tok.encode(text)` ;
- `sweep_20(big_tok, text, sizes)` : une ligne par taille de `sizes`, dans leur ordre ; chaque ligne est un dictionnaire de clés `"vocab_size"`, `"bytes_per_token"`, `"bits_per_token"`, `"bits_per_byte"` et `"used"` (le nombre d'identifiants distincts du texte encodé), mesurées sur `text` avec `truncated_bpe(big_tok, size)`.

La cellule de vérification essaie tes deux mesures sur un faux tokenizer, qui rend des identifiants fixés à l'avance. La cellule d'expérience apprend un BPE de 8 000 tokens sur `VERNE_TRAIN` avec `fast_byte_bpe`. Elle vérifie à 400 tokens que ton `fit` apprend les mêmes fusions (sur 100 000 caractères, de 1 à 10 s selon ta façon de compter les paires) et que `truncated_bpe` redonne bien le tokenizer plus petit. Elle lance enfin `sweep_20` et trace les trois courbes.

Dans tes notes :
- la compression gagne-t-elle autant à chaque doublement de la taille ?
- l'entropie par token et les bits par octet : lequel monte, lequel baisse, et comment les deux vont-ils ensemble ?
- combien des 8 000 tokens servent dans la validation (colonne `used`) ?
- quelle taille prendrais-tu pour un petit modèle de langue sur Verne, et que ces courbes ne disent-elles pas (la taille de la table d'embedding, 🧮 B2.9 ; les tokens rares, vus trop peu pour être bien appris) ?""",
       todo=C.TODO_20, check=C.CHECK_20, solution=C.SOLUTION_20 + "\n\n\n" + C.CHECK_20,
       after=[("code", C.EXPERIMENT_20)]),

    Ex("B2.21", "🔬", 2, 25, "Tokenizer entraîné sur Holmes, appliqué à Verne (et l'inverse) : le coût de la langue",
       "mesurer ce que coûte un tokenizer appris sur une autre langue, et ce que coûte un tokenizer partagé.",
       "Ex B2.20 · Ex B2.13 · fiche §202.6 (« Le même texte, un autre prix selon la langue »)", tracks="C",
       body=r"""En B2.13, GPT-2 compresse moins bien Verne que Holmes. Cela vient-il du français, ou des textes sur lesquels son tokenizer a appris ? Tu entraînes trois BPE octet de 2 000 tokens et tu les mesures sur la validation des deux romans.

**Contrat**
- `train_tokenizers_21(vocab_size=2000)` : trois `BPETokenizer` de ta librairie, au niveau octet, de pré-tokenizer `'gpt2'` et de `vocab_size` tokens, appris par `fast_byte_bpe` (ou par ton `fit`, plus lent, qui donne les mêmes fusions), dans un dictionnaire :
  - `"holmes"`, appris sur `HOLMES_TRAIN` ;
  - `"verne"`, appris sur `VERNE_TRAIN` ;
  - `"mixte"`, appris sur les deux textes ensemble.
- `cross_table_21(tokenizers, texts)` : un `DataFrame` de pandas des octets par token (`compression_20`, B2.20), une ligne par tokenizer et une colonne par texte ; `tokenizers` et `texts` sont des dictionnaires nom → objet.

La cellule de vérification entraîne tes trois tokenizers (quelques secondes avec `fast_byte_bpe`) et contrôle leurs réglages. Elle vérifie aussi que celui de Verne commence par les fusions de B2.20. Elle calcule ensuite ton tableau, y ajoute GPT-2 et le trace.

Dans tes notes :
- de combien la compression baisse-t-elle quand on applique un tokenizer à l'autre roman ?
- le tokenizer mixte : que coûte-t-il à chaque langue, par rapport à son tokenizer « maison » ?
- où se place GPT-2, avec ses 50 257 tokens ?
- qu'en conclus-tu sur la « taxe » de certaines langues (fiche §202.6), et sur le choix des données d'entraînement du tokenizer d'un modèle multilingue ?""",
       todo=C.TODO_21, check=C.CHECK_21, solution=C.SOLUTION_21 + "\n\n\n" + C.CHECK_21),
])

PARTS = [PART_A, PART_B, PART_C]
