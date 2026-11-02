"""Parts A and B of the chapter B4 notebook (B4.12 to B4.21): statements and cells (used by build_chB4.py).

Part A: usage and prompting with transformers (B4.12 to B4.16). Part B, first half: your retrieval library and the RAG
(B4.17 to B4.21). The second half of the chapter (B4.22 to B4.30) reuses the tools of these parts (CACHE, LLM, QWEN,
QA, REF_CHUNKS, INDEX_20, search_20, rag_21…), see chapitres/b4_llm/EN_COURS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part, guarded

import chB4_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB4.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}

P13 = HISTORY.get("p13", {}).get("answers", {"identical": False, "watson": False})
P19 = HISTORY.get("p19", {}).get("answers", {"bm25_half": False, "dense_half": False, "hybrid_best": True})
C12 = HISTORY.get("c12", {"n_prompt": 22, "n_linear": 211, "n_block": 3_540_096, "words": 20, "chars": 102,
                          "chat_prompt": 47, "attention": 884_736, "mlp": 2_654_208, "norms": 1_152})
E5_HOLMES = HISTORY.get("e5_holmes", {}).get("rows", [])

TEST = "tests/test_chb4_retrieval.py"
GROUP_17 = "test_chunk_text_ or test_bm25_"
GROUP_18 = "test_cosine_top_k_ or test_reciprocal_rank_fusion_ or test_recall_at_k_ or test_mean_reciprocal_rank_"


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** : ouvre `mon_travail/mylearn/retrieval.py` (créé par `python tools/start_chapter.py "
            "B4`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de l'exercice par ton code et "
            "**enregistre**. NumPy et la bibliothèque standard seulement. La première cellule de vérification recharge ta "
            "librairie et l'essaie sur des cas visibles ; la seconde lance les tests de l'exercice. "
            "`python -m pytest tests/test_chb4_retrieval.py -q` les lance tous dans un terminal.")


def tests_after(ex_id: str, probe: str, keyword: str, n_tests: int, what: str = "tes fonctions") -> list:
    """The second check cell of a 🔨: the tests of the exercise (explicit test file, -k group)."""
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_RET, "{keyword}")   # {n_tests} tests\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')
    ref = cell.replace('run_pytest_group(TEST_RET, "', 'run_pytest_group(TEST_RET, impl="ref", keyword="')
    return [("check", cell), ("solution", ref)]


def _mistakes(entries: dict) -> str:
    return "{" + ", ".join(f"{json.dumps(msg, ensure_ascii=False)}: {value!r}" for msg, value in entries.items()) + "}"


RECORD_12 = (
    f'wb.record("B4.12a", {C12["n_prompt"]}, mistakes={_mistakes({"c’est le nombre de mots : un token n’est pas un mot (B2)": C12["words"], "c’est le nombre de caractères : compte les identifiants que rend le tokenizer": C12["chars"], "tu comptes le prompt après le chat template : on demande les tokens de tok(PROMPT_12), le texte brut": C12["chat_prompt"]})}, '
    'fractional="un nombre de tokens est un entier : len(tok(PROMPT_12)[\\"input_ids\\"])")\n'
    f'wb.record("B4.12b", {C12["n_linear"]}, mistakes={_mistakes({"la tête de sortie (lm_head) est aussi un nn.Linear, même si son poids est lié à la table des tokens": C12["n_linear"] - 1, "c’est le nombre de couches denses d’un seul bloc : compte dans tout le modèle": 7, "c’est le nombre de blocs : compte les nn.Linear de tout le modèle": 30})}, '
    'fractional="un nombre de modules est un entier")\n'
    f'wb.record("B4.12c", {C12["n_block"]}, mistakes={_mistakes({"les deux couches de normalisation du bloc (RMSNorm) ont aussi des poids": C12["n_block"] - C12["norms"], "tu ne comptes que l’attention : le MLP du bloc compte aussi": C12["attention"], "tu ne comptes que le MLP : l’attention du bloc compte aussi": C12["mlp"]})}, '
    'fractional="un nombre de paramètres est un entier : additionne p.numel() sur model.model.layers[0].parameters()")')

RECORD_13 = (
    f'wb.record("B4.13a", {P13["identical"]}, mistakes={{"relis, à la température 0,7, la colonne « identiques à la '
    f'réponse gloutonne »": {not P13["identical"]}}})\n'
    f'wb.record("B4.13b", {P13["watson"]}, mistakes={{"relis, à la température 0,7, combien des 10 réponses nomment '
    f'Watson": {not P13["watson"]}}})')

RECORD_19 = (
    f'wb.record("B4.19a", {P19["bm25_half"]}, mistakes={{"relis la dernière ligne : combien de questions BM25 '
    f'retrouve-t-il dans ses 5 premiers résultats ?": {not P19["bm25_half"]}}})\n'
    f'wb.record("B4.19b", {P19["dense_half"]}, mistakes={{"relis la dernière ligne : combien de questions la recherche '
    f'dense retrouve-t-elle dans ses 5 premiers résultats ?": {not P19["dense_half"]}}})\n'
    f'wb.record("B4.19c", {P19["hybrid_best"]}, mistakes={{"compare, sur la dernière ligne, le compte de l’hybride à '
    f'celui de la meilleure des deux autres": {not P19["hybrid_best"]}}})')

# ---------------------------------------------------------------------------------------------------------------------
PART_A = Part("A", "Usage et prompting",
              "Tu fais tourner un vrai petit LLM sur ton CPU : SmolLM2-135M-Instruct (135 millions de paramètres, Apache "
              "2.0), à une révision figée, en float32. Tu le charges et le mesures (B4.12), tu prédis ce que fait la "
              "température (B4.13), tu écris une conversation avec son chat template, un message système, une chaîne "
              "d'arrêt et le streaming (B4.14), puis tu lui fais extraire des informations de Holmes en JSON validé "
              "(B4.15) et tu mesures la fragilité d'un prompt (B4.16).\n\n"
              "**Les réponses générées sont mises en cache.** Une génération coûte d'une à quelques secondes ; les "
              "expériences en demandent des dizaines. `LLM.generate(prompt, ..., cache=CACHE)` (la cellule suivante) "
              "relit une réponse déjà générée **pour exactement la même entrée** : le même modèle à la même révision, "
              "les mêmes tokens de prompt, les mêmes réglages, la même graine, les mêmes versions de transformers et de "
              "torch. Changer un seul caractère du prompt donne une autre clé, donc une nouvelle génération : le cache ne "
              "remplace jamais ton code, il évite seulement de refaire un calcul identique. Il est rangé hors du dépôt "
              "(`data/downloads/llm/`, sur Colab `/content/wb_cache/llm/`) ; `CACHE.clear()` le vide.\n\n"
              "La cellule suivante charge les outils de la partie : `SMOL` (le dépôt, la révision et la licence du "
              "modèle), `CACHE`, `LLM` (le même SmolLM2, chargé par `wb.llm` : les vérifications et les expériences s'en "
              "servent ; ta version vient en B4.12), et `spy_from_pretrained`, qui note les appels à `from_pretrained` "
              "pendant une vérification. Premier lancement : environ 270 Mo à télécharger, une fois.",
              given=C.TOOLS_A, exercises=[
    Ex("B4.12", "📦", 2, 25, "Premier contact : AutoTokenizer, AutoModelForCausalLM, generate() et pipeline sur un petit "
                            "modèle",
       "charger un petit LLM à une révision figée, le compter, le faire générer, mesurer sa vitesse et la comparer à ton "
       "estimation.",
       "✏️ B4.1 · fiche §204.2 (« Trois objets », « Vitesse sur CPU », « Le pipeline : une ligne pour essayer »), §204.3 "
       "(« `generate()` et ses réglages »)", tracks="R, C",
       body=r"""**Contrat**
- `load_12()` rend `(tok, model)` : `AutoTokenizer.from_pretrained` et `AutoModelForCausalLM.from_pretrained` sur `SMOL["repo"]`, **avec** `revision=SMOL["revision"]`, le modèle en float32 (`dtype=torch.float32` : sur un CPU sans instructions bfloat16, le float32 va plus vite, fiche §204.2) et en mode eval.
- `complete_12(tok, model, prompt, max_new_tokens)` continue le **texte brut** `prompt` (sans chat template, `tok(prompt, return_tensors="pt")`), de façon gloutonne (`do_sample=False`), et rend `(text, n_new)` : le texte des seuls tokens générés, décodé sans les tokens spéciaux, et leur nombre.
- `counts_12(tok, model)` rend le triplet `(n_prompt, n_linear, n_block)` :
  - a) `n_prompt` : le nombre de tokens de `tok(PROMPT_12)` ;
  - b) `n_linear` : le nombre de modules `nn.Linear` du modèle, la tête de sortie comprise ;
  - c) `n_block` : le nombre de paramètres du premier bloc décodeur, `model.model.layers[0]`.
- `make_pipe_12(tok, model)` rend un `pipeline("text-generation", ...)` construit sur ces deux objets.
- `estimate_12` : ta vitesse estimée en ✏️ B4.1 j), en tokens par seconde (rien à recalculer : recopie ton résultat).

La cellule de vérification appelle `load_12` en notant ses appels à `from_pretrained` (révision, dtype), contrôle le float32 et le mode eval, vérifie tes trois comptes par `wb.check` (la cellule les calcule avec ta fonction : ne les tape pas), puis génère 64 tokens après une passe de chauffe, compare ta continuation à celle de la référence et **mesure** ta vitesse. Une durée n'est jamais une réponse vérifiée : elle dépend de la machine et de sa charge. Enfin, le pipeline répond à `QUESTION_12` en 24 tokens gloutons ; sa réponse est comparée à celle de la référence.

Dans tes notes : ta mesure contre ton estimation de ✏️ B4.1 j) : dans quel sens se trompe le calcul « par proportion des mémoires », et d'où vient l'écart (taille de la table des tokens, petites matrices, surcoût de Python à chaque token) ? Que contient la sortie de `generate()` avant que tu gardes les seuls tokens générés ? Que fait le pipeline à ta place, et que perds-tu à t'en servir pour une expérience ?""",
       given=C.GIVEN_12, todo=C.TODO_12, check=C.CHECK_12, solution=C.SOLUTION_12 + "\n\n\n" + C.CHECK_12,
       record=RECORD_12),

    Ex("B4.13", "🔮", 1, 10, "Prédire : dix générations du même prompt à température 0, 0,7 et 1,5",
       "prédire ce que donnent dix tirages à température 0,7 face à la réponse gloutonne d'un petit modèle.",
       "Ex B4.12 · B3 (décodage) · ch. 22 (température) · fiche §204.3 (« `generate()` et ses réglages », piège des "
       "réglages cachés)", tracks="C", hypothesis=True,
       body=r"""Le prompt `PROMPT_13` demande à SmolLM2 le nom du médecin qui partage l'appartement de Sherlock Holmes. La cellule suivante donne `draws_13(temperature)` : dix réponses de 24 tokens au plus, gloutonnes pour une température de 0 (`do_sample=False`), tirées au hasard sinon, **sans filtre** top-k ni top-p, après `torch.manual_seed(seed)` pour les graines 0 à 9 (et par le cache).

**Avant d'exécuter**, écris tes deux prédictions :
- a) `identical_13` : à 0,7, au moins 3 des 10 réponses sont-elles identiques, mot pour mot, à la réponse à température 0 ? `True` ou `False` ;
- b) `watson_13` : à 0,7, au moins 7 des 10 réponses nomment-elles Watson ? `True` ou `False`.

La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites. Elle affiche ensuite les trente réponses, et pour chaque température le nombre de réponses distinctes, de réponses qui nomment Watson et de réponses identiques à la gloutonne ; elle vérifie tes prédictions à la fin. Elle ne dépend pas de ton code.""",
       given=C.GIVEN_13, todo=C.TODO_13,
       solution=f'identical_13, watson_13 = {P13["identical"]}, {P13["watson"]}', record=RECORD_13,
       after=[("code", guarded(C.EXPERIMENT_13, ["identical_13", "watson_13"],
                               "⏳ Ex B4.13 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? À quelle température "
                     "écrirais-tu un résumé, une extraction de date, un slogan ? Le modèle « sait »-il que le médecin "
                     "s'appelle Watson, si 0,7 le dit rarement ? Refais l'expérience avec `top_k=20, top_p=0.8` (les "
                     "réglages de Qwen2.5, fiche §204.3) : que change le filtre ?")]),

    Ex("B4.14", "📦", 2, 25, "Chat template, message système, arrêt et streaming avec un modèle instruct",
       "écrire une conversation avec le chat template, puis arrêter et diffuser la génération.",
       "Ex B4.12 · ✏️ B4.4 · fiche §204.3 (« S'arrêter », « Le streaming », « Rôles et chat template »)", tracks="R, C",
       body=r"""**Contrat**
- `chat_14(tok, model, messages, max_new_tokens=48, stop_strings=None, streamer=None)` rend la réponse gloutonne du modèle à la conversation `messages` (une liste de `{"role": ..., "content": ...}`) :
  - le prompt : `tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)` ;
  - `generate()` avec `do_sample=False`, `max_new_tokens`, et, s'ils sont donnés, `stop_strings` (avec `tokenizer=tok`, que demandent les chaînes d'arrêt) et `streamer` ;
  - les seuls tokens générés, décodés sans les tokens spéciaux ; une chaîne d'arrêt qui **termine** la réponse est retirée (`generate()` la laisse, fiche §204.3).
- `stream_14(tok, model, messages, max_new_tokens=48)` rend la liste des morceaux de la même réponse, dans l'ordre où un `TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)` les donne pendant que `generate()` tourne dans un autre fil d'exécution (`threading.Thread`) : leur concaténation est la réponse.

La cellule de vérification prend ton modèle de B4.12 (ou, s'il manque, le même SmolLM2 chargé par la référence) et compare tes réponses à celles de la référence : une question seule (`ONE_14`, dont elle affiche le prompt après le chat template), deux messages système (`SYSTEM_14` : une réponse courte ; une consigne en français pour une question sur Verne), un second tour (ta première réponse renvoyée comme message `assistant`, puis une nouvelle question), une liste (`LIST_14`) coupée au premier retour à la ligne par `stop_strings=["\n"]`, puis le streaming : un `TextStreamer` qui affiche la réponse pendant la génération, et tes morceaux de `stream_14`.

Dans tes notes : quel message système le gabarit de SmolLM2 ajoute-t-il quand tu n'en donnes pas, et qu'en penses-tu pour un produit (✏️ B4.4) ? La consigne en français est-elle suivie par ce modèle de 135 M de paramètres, entraîné surtout sur de l'anglais ? Au second tour, que renvoies-tu au modèle, et pourquoi le temps de réponse grandit-il avec la conversation ? À quoi sert le streaming si la réponse complète prend le même temps ?""",
       given=C.GIVEN_14, todo=C.TODO_14, check=C.CHECK_14, solution=C.SOLUTION_14 + "\n\n\n" + C.CHECK_14),

    Ex("B4.15", "📦", 3, 45, "Prompting pour extraire des informations de Holmes : zéro-shot, few-shot, JSON validé, vote "
                            "majoritaire",
       "faire extraire des informations en JSON, valider la sortie, et comparer zéro-shot, few-shot et vote majoritaire.",
       "Ex B4.14 · 0A.39 et 0A.47 (JSON, validation) · fiche §204.4 (« La sortie structurée », « Voter : "
       "l'auto-cohérence »), §204.5 (« Les métriques »)", tracks="R, C",
       body=r"""La tâche : pour un court passage de Holmes, rendre un JSON `{"name": ..., "place": ..., "year": ...}`, la personne nommée, le lieu, et l'année écrite dans le texte, chaque valeur `null` si le texte ne la donne pas. La cellule suivante donne `SYSTEM_15` (la consigne), `EXTRACT_15` (8 passages et leur réponse attendue, écrits pour le workbook), `EXAMPLES_15` (trois autres passages et leur réponse, pour les exemples d'un prompt few-shot), `field_ok_15` et `score_15` (la part de JSON valides et la précision de chaque champ ; un texte qui contient les mots attendus compte juste : « Lebanon, Pennsylvania » pour « Lebanon »).

**Contrat**
- `parse_json_15(reply)` rend le dictionnaire `{"name", "place", "year"}` d'une réponse, ou `None` :
  - la réponse sans les blancs autour, puis sans une balise Markdown qui l'entoure (`` ```json `` … `` ``` ``, ou `` ``` `` … `` ``` ``), lue par `json.loads` ; un texte autour du JSON (« Here is the JSON: … ») le rend invalide ;
  - `None` si ce n'est pas un objet JSON avec **exactement** les trois clés de `KEYS_15`, ou si une valeur a le mauvais type : `name` et `place` une chaîne ou `null`, `year` un entier (pas un booléen, pas une chaîne comme `"1888"`) ou `null` ;
  - ne lève jamais d'exception.
- `messages_15(text, examples=(), system=SYSTEM_15)` rend les messages : le message système `system`, puis, pour chaque `(texte, réponse)` de `examples`, un message `user` (le texte) suivi d'un message `assistant` (`json.dumps(réponse)`), enfin le message `user` `text`. Sans exemple, c'est un prompt zéro-shot.
- `vote_15(replies)` rend le dictionnaire le plus fréquent parmi les réponses que `parse_json_15` accepte (deux dictionnaires sont égaux quand leurs trois valeurs le sont), le premier vu en cas d'égalité, `None` si aucune n'est valide ; la liste reçue n'est pas modifiée.

La cellule de vérification essaie `parse_json_15` sur treize réponses d'essai, la structure de `messages_15` et `vote_15` sur quatre listes. L'expérience compare ensuite, sur les 8 passages : le **zéro-shot** (glouton), le **few-shot** avec les deux premiers exemples (glouton), et le **vote** sur 5 tirages few-shot à $T = 0{,}7$ (graines 0 à 4) ; 56 générations de 40 tokens au plus, environ 1 min 30 sur CPU la première fois (puis le cache).

Dans tes notes : quelles erreurs de format fait le modèle en zéro-shot, et lesquelles disparaissent avec deux exemples ? Le vote corrige-t-il des erreurs de contenu, ou seulement de format ? Sur 8 passages, un écart d'une réponse vaut combien de points, et que peux-tu conclure (fiche §204.5) ? Quelle consigne ou quel exemple ajouterais-tu pour le champ qui échoue le plus ?""",
       given=C.GIVEN_15, todo=C.TODO_15, check=C.CHECK_15, solution=C.SOLUTION_15 + "\n\n\n" + C.CHECK_15,
       after=[("code", C.EXPERIMENT_15)]),

    Ex("B4.16", "🔬", 3, 40, "Sensibilité au prompt : cinq formulations, trois graines, ordre des exemples",
       "mesurer la dispersion d'une extraction entre formulations du prompt, et la comparer à celle due aux graines.",
       "Ex B4.15 · ch. 2 (écart-type) · fiche §204.4 (« Un prompt est fragile »), §204.5 (« La variance »)", tracks="C",
       body=r"""Les ampleurs publiées (fiche §204.4) portent sur d'autres modèles et d'autres tâches : tu mesures ici sur la tâche de B4.15. La cellule suivante donne `FORMULATIONS_16` (cinq messages système qui demandent la même chose, avec d'autres mots et une autre mise en forme ; `F0` est `SYSTEM_15`), `ORDERS_16` (trois ordres des trois exemples de `EXAMPLES_15`), `SEEDS_16` (0, 1, 2), `SNIPPETS_16` (les 6 premiers passages de `EXTRACT_15` en mode rapide, les 8 en mode complet) et `MODEL_16` (SmolLM2 en mode rapide ; Qwen2.5-0.5B-Instruct en mode complet).

**Contrat**
- `accuracy_16(system, examples, seed)` rend la précision moyenne des champs, `score_15(...)["all"]`, des réponses de `MODEL_16` aux passages de `SNIPPETS_16` : pour chaque passage, les messages `messages_15(texte, examples, system)` et **un** tirage, `MODEL_16.generate(messages, max_new_tokens=40, do_sample=True, temperature=0.7, seed=seed, cache=CACHE)`, lu par `parse_json_15` (tes fonctions de B4.15).
- `spread_16(table)` reçoit un tableau `(n_variantes, n_graines)` de précisions, une ligne par variante du prompt, et rend `(mean, between, within)` : la moyenne du tableau, l'écart-type (`ddof=0`) des moyennes des lignes (la dispersion **entre** variantes), et la moyenne des écarts-types (`ddof=0`) de chaque ligne (la dispersion due aux **graines**).

La cellule de vérification essaie `spread_16` sur un petit tableau, puis `accuracy_16` avec un modèle factice qui note ses appels (les messages, les réglages, le nombre d'appels) et rend un JSON fixe. L'expérience remplit deux tableaux, cinq formulations × trois graines (trois exemples dans l'ordre de `EXAMPLES_15`), puis trois ordres des exemples × trois graines (la formulation `F0`), affiche leurs dispersions et les trace : 108 générations en mode rapide, environ 3 min sur CPU la première fois ; en mode complet, Qwen2.5 sur les 8 passages, environ 10 min.

Dans tes notes : l'écart entre formulations dépasse-t-il la dispersion due aux graines, et de combien ? Laquelle des deux conclusions est fragile avec 6 passages et 3 graines (combien vaut un passage sur la précision moyenne) ? Que changerais-tu au protocole pour départager deux formulations (fiche §204.5) ? Garderais-tu la meilleure formulation, ou la plus stable ?""",
       given=C.GIVEN_16, todo=C.TODO_16, check=C.CHECK_16, solution=C.SOLUTION_16 + "\n\n\n" + C.CHECK_16,
       after=[("code", C.EXPERIMENT_16)]),
])

# ---------------------------------------------------------------------------------------------------------------------
COMMON_RET = (
    "**Contrat commun de `retrieval.py`** (les docstrings, plus ces précisions que vérifient les tests) :\n"
    "- tes fonctions ne modifient jamais leurs entrées ; les listes sont acceptées partout où un tableau l'est ;\n"
    "- les valeurs sont comparées à rtol = atol = $10^{-6}$, les identifiants, les indices et les chunks exactement ;\n"
    "- `cosine_top_k` réutilise ta `cosine_similarity_matrix` de `mylearn/embeddings.py` (B2, 🔨 B2.23) : le stub "
    "l'importe. Si ce fichier manque dans ta librairie, la référence le remplace (une ligne ℹ️ le dit) ; s'il existe "
    "mais que la fonction n'est pas écrite, les tests qui en dépendent affichent ⏳ en nommant son fichier et son "
    "chapitre.")

PART_B = Part("B", "Recherche et RAG",
              "Un RAG ne répond bien que si la recherche lui montre le bon passage. Tu écris dans `mylearn/retrieval.py` "
              "le découpage et BM25 (B4.17), la recherche dense, la fusion RRF et les métriques (B4.18), tu prédis qui "
              "de BM25 ou de la recherche dense retrouve des questions paraphrasées (B4.19), tu compares les méthodes "
              "sur 30 questions annotées (B4.20), puis tu assembles un RAG complet avec citations vérifiées (B4.21).\n\n"
              "**Les questions** (`data/qa/`, data card `data/cards/qa.md`, écrites pour le workbook) : 30 questions "
              "factuelles sur les douze nouvelles de Holmes, chacune avec sa réponse courte, des variantes acceptées et "
              "son **extrait source** (au plus 21 mots du texte, situé par ses positions de caractères) ; 10 d'entre "
              "elles sont **paraphrasées** : aucun de leurs mots-clés n'apparaît dans les 140 mots qui entourent leur "
              "extrait. Un chunk est **pertinent** quand il contient l'extrait entier (`wbrag.relevant_ids`) : la "
              "vérité ne dépend pas du découpage, et un recouvrement de 20 mots garde chaque extrait entier dans au moins "
              "un chunk. S'y ajoutent 10 questions en français sur les chapitres I à VII du *Tour du monde en "
              "quatre-vingts jours*, et 5 questions sans réponse dans les nouvelles (la partie suivante s'en sert).\n\n"
              + COMMON_RET + "\n\n"
              "La cellule suivante charge les outils de la partie : `TEST_RET` (le fichier des tests), `QA`, `QA_FR`, "
              "`DOCS` et `DOCS_FR` (les nouvelles et les chapitres, `[(titre, texte)]`), `REF_CHUNKS` et `REF_DOC` (le "
              "découpage de référence, 120 mots, recouvrement de 20, nouvelle par nouvelle : 1 045 chunks, et la "
              "nouvelle de chaque chunk), `tokenize` (les tokens de BM25 : minuscules, suites de lettres et de chiffres ; "
              "la **même** fonction pour le corpus et les requêtes).",
              given=C.TOOLS_B, exercises=[
    Ex("B4.17", "🔨", 3, 45, "chunk_text et BM25 : découper un roman et classer ses passages",
       "écrire le découpage avec recouvrement et BM25 de ta librairie, puis classer les passages de Holmes.",
       "✏️ B4.5 · ch. 13 (TF-IDF, 13.30) · fiche §204.6 (« Découper », « De TF-IDF à BM25 »)", tracks="R, C",
       mylearn="retrieval.py",
       body=mylearn_howto() + r"""

**Contrat**
- `chunk_text(text, chunk_size=200, overlap=50, unit="word")` :
  - les fenêtres de `chunk_size` unités commencent toutes les `chunk_size - overlap` unités, jusqu'à la première qui atteint la fin du texte (la dernière peut être plus courte) ; un texte d'exactement `chunk_size` unités donne **un seul** chunk ;
  - `unit="word"` : le texte coupé aux blancs (`split()`), chaque chunk recollé avec une seule espace ; `unit="char"` : **chaque caractère** est une unité, espaces et retours à la ligne compris, sans aucune normalisation (le « whitespace-normalised » de la docstring ne concerne que l'unité `"word"`) ;
  - les arguments sont vérifiés **avant** tout calcul : `ValueError` si `chunk_size < 1`, `overlap < 0` ou `overlap >= chunk_size` (la fenêtre n'avancerait pas : les tests passent par une garde qui arrête une boucle sans fin), ou pour une unité inconnue.
- `BM25(k1=1.5, b=0.75)`, `fit(corpus)`, `get_scores(query)` : la formule de la fiche, avec l'idf toujours positif $\ln\big(1 + \frac{N - n_t + 0{,}5}{n_t + 0{,}5}\big)$ :
  - `fit` reçoit des documents déjà tokenisés (des listes de chaînes), apprend `idf_` (un dictionnaire, une entrée par terme du corpus), `doc_len_` (un tableau NumPy), `avgdl_`, `n_docs_` et `term_freqs_` (une liste de dictionnaires, un par document), et rend `self` ; un second `fit` remplace toutes les statistiques ;
  - le corpus peut contenir des documents vides : ils comptent dans $N$ et dans $\overline{|d|}$, et leur score est 0 ;
  - `get_scores` rend un tableau NumPy de flottants de forme `(n_docs,)` ; un document qui ne contient pas un terme de la requête reçoit **0** de ce terme, y compris avec `k1 = 0` ou pour un document vide avec `b = 1` : un dénominateur nul donne 0, jamais NaN ; les tokens sont comparés tels quels (casse comprise) ;
  - `ValueError` pour `get_scores` avant `fit`, un corpus vide, `k1 < 0` ou `b` hors de $[0, 1]$ (ces deux-là peuvent être refusés dès le constructeur ou par `fit`).

La première cellule de vérification essaie `chunk_text` sur l'exemple de la fiche, un texte d'exactement `chunk_size` mots, l'unité `"char"` et un refus ; BM25 sur les quatre documents de la fiche (contre un calcul de référence), le cas `k1 = 0` avec un document vide et un refus ; puis elle découpe les douze nouvelles de Holmes (120 mots, recouvrement de 20 : les chunks doivent être ceux de `REF_CHUNKS`) et affiche les trois passages que ton BM25 classe en tête pour « violin music concert ». La seconde lance les 62 tests de l'exercice (`-k "test_chunk_text_ or test_bm25_"`).

Dans tes notes : combien de chunks donne la formule de la fiche pour une nouvelle de $n$ mots, et pourquoi le total (1 045) dépasse-t-il le nombre de mots divisé par 100 ? Pourquoi un recouvrement de 20 mots garde-t-il entier tout extrait de 21 mots au plus ? Les trois passages de « violin music concert » parlent-ils tous de musique, et lequel des quatre morceaux de la formule de BM25 explique le premier ?""",
       check=C.RELOAD + C.CHECK_17, solution=C.CHECK_17,
       after=tests_after("B4.17", "mylearn.retrieval.chunk_text('a b', 1, 0)", GROUP_17, 62)),

    Ex("B4.18", "🔨", 2, 30, "cosine_top_k, fusion de classements (RRF), recall@k et MRR",
       "écrire la recherche dense exhaustive, la fusion RRF et les deux métriques de recherche de ta librairie.",
       "✏️ B4.2 · B2 (`cosine_similarity_matrix`, 🔨 B2.23) · fiche §204.6 (« La recherche dense », « La recherche "
       "hybride et la fusion RRF », « Mesurer une recherche : recall@k et MRR »)", tracks="R, C", mylearn="retrieval.py",
       body=mylearn_howto() + r"""

**Contrat**
- `cosine_top_k(query_emb, doc_embs, k=5)` rend `(indices, similarities)`, deux tableaux NumPy (des entiers, des flottants), de forme `(k,)` pour une requête `(d,)` et `(q, k)` pour des requêtes `(q, d)` : les $k$ documents de plus grand cosinus, triés par similarité décroissante, les ex aequo départagés par le **plus petit** indice ; un vecteur nul a une similarité de **0** avec tout, comme dans ta `cosine_similarity_matrix` (la réutiliser évite le NaN : un cosinus maison sans `eps` échoue) ; `ValueError` si `k < 1`, si `k > n` (contrairement à `most_similar` de B2, qui tronquait) ou si les dimensions diffèrent.
- `reciprocal_rank_fusion(rankings, k=60)` : le score de chaque identifiant est $\sum_r 1/(k + \operatorname{rang}_r)$, les rangs à partir de 1 ; rend un tableau NumPy d'identifiants **entiers**, du meilleur score au moins bon, à égalité le plus petit identifiant d'abord ; les classements peuvent avoir des longueurs différentes, être vides, ou être des tableaux NumPy (les lignes rendues par `cosine_top_k`) ; une fusion vide (`[[]]`, `[[], []]`) rend un tableau d'entiers de forme `(0,)` (`np.array([])` serait un tableau de flottants) ; `ValueError` si `rankings` est vide ou si `k < 0`.
- `recall_at_k(retrieved, relevant, k)` et `mean_reciprocal_rank(retrieved, relevant)` :
  - `retrieved` : une liste d'identifiants par requête, ou le tableau 2-D d'indices que rend `cosine_top_k` (teste `len(retrieved) == 0`, pas `not retrieved`) ; dans chaque liste, les identifiants sont distincts ; une liste peut être plus courte que `k` ;
  - `relevant` : un ensemble, une liste ou un tuple d'identifiants par requête ;
  - recall@k est la moyenne de $|\text{top-}k \cap \text{pertinents}| / |\text{pertinents}|$ (une intersection d'ensembles) ; `ValueError` si les longueurs diffèrent, si `k < 1` ou si un ensemble pertinent est vide ;
  - le MRR est la moyenne de $1/\text{rang}$ du premier pertinent, 0 quand aucun n'est retrouvé ; une requête dont l'ensemble pertinent est vide compte 0, **sans** erreur (contrairement à `recall_at_k`) ; `ValueError` si les longueurs diffèrent ;
  - les deux rendent un nombre.

La première cellule de vérification essaie `cosine_top_k` sur l'exemple de la fiche, un vecteur nul et un refus, la fusion de la fiche et une fusion vide, recall@1, recall@3 et MRR de l'exemple de la fiche, puis les métriques sur un tableau rendu par `cosine_top_k` et avec un ensemble pertinent vide. La seconde lance les 67 tests de l'exercice (`-k "test_cosine_top_k_ or test_reciprocal_rank_fusion_ or test_recall_at_k_ or test_mean_reciprocal_rank_"`).

Dans tes notes : pourquoi le plus grand cosinus est-il aussi le plus proche voisin euclidien pour des vecteurs de norme 1 (fiche §204.6) ? Que deviendrait la fusion de la fiche avec $k = 0$, et pourquoi Cormack et ses coauteurs prennent-ils $k = 60$ ? Pourquoi le recall@k refuse-t-il un ensemble pertinent vide, alors que le MRR l'accepte ?""",
       check=C.RELOAD + C.CHECK_18, solution=C.CHECK_18,
       after=tests_after("B4.18", "mylearn.retrieval.recall_at_k([[0]], [{0}], 1)", GROUP_18, 67)),

    Ex("B4.19", "🔮", 1, 10, "Prédire : sur des questions paraphrasées, BM25 ou recherche dense gagne-t-elle ?",
       "prédire combien de questions paraphrasées BM25, la recherche dense et l'hybride retrouvent sur Holmes.",
       "Ex B4.17 · Ex B4.18 · 🧠 Q9 · fiche §204.6 (« La recherche dense », « La recherche hybride et la fusion RRF »)",
       tracks="R, C", hypothesis=True,
       body=r"""Les 10 questions paraphrasées de `QA` ne partagent aucun mot-clé avec le texte qui entoure leur réponse : « How many big gems decorate the royal headpiece? » cherche un passage qui parle de « beryls » et de « coronet ». L'expérience découpe les douze nouvelles avec ton `chunk_text` (120 mots, recouvrement de 20), indexe les chunks avec ton `BM25` (sur `tokenize`) et avec all-MiniLM-L6-v2 (la recherche dense de B2 : environ 40 s d'embeddings la première fois, puis le cache), et cherche chaque question de trois façons : BM25, la recherche dense (ton `cosine_top_k`), et l'**hybride**, ta `reciprocal_rank_fusion` ($k = 60$) des 20 premiers résultats des deux. Pour chaque méthode, elle compte les questions dont le chunk pertinent est dans les **5 premiers** résultats.

**Avant d'exécuter**, écris tes trois prédictions :
- a) `bm25_half_19` : BM25 retrouve-t-il au moins 5 des 10 questions dans ses 5 premiers résultats ? `True` ou `False` ;
- b) `dense_half_19` : et la recherche dense ? `True` ou `False` ;
- c) `hybrid_best_19` : l'hybride en retrouve-t-il **au moins autant** que la meilleure des deux ? `True` ou `False`.

La cellule d'expérience n'affiche rien tant que tes prédictions ne sont pas écrites ; elle appelle ton code de B4.17 et B4.18, et ne vérifie tes prédictions que si elle a pu tourner. Elle affiche, pour chaque question, le rang du chunk pertinent dans chaque méthode, puis les trois comptes.""",
       given=C.GIVEN_19, todo=C.TODO_19,
       solution=(f'bm25_half_19, dense_half_19, hybrid_best_19 = {P19["bm25_half"]}, {P19["dense_half"]}, '
                 f'{P19["hybrid_best"]}'),
       record=RECORD_19,
       after=[("code", guarded(C.EXPERIMENT_19, ["bm25_half_19", "dense_half_19", "hybrid_best_19"],
                               "⏳ Ex B4.19 : écris d'abord tes trois prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Regarde une question que "
                     "la recherche dense rate : à quel rang est son chunk, et que contient le premier chunk trouvé à sa "
                     "place ? Pourquoi un modèle d'embedding entraîné sur des phrases modernes peut-il ne pas "
                     "rapprocher « headpiece » de « coronet » dans un passage de 120 mots du XIXᵉ siècle ? Que dit la "
                     "fusion d'un document classé loin par les deux méthodes ?")]),

    Ex("B4.20", "📦", 3, 45, "Recherche dense (MiniLM, e5) contre BM25 contre hybride sur 30 questions annotées",
       "écrire un index et une recherche à trois méthodes avec ta librairie, et les évaluer sur Holmes et Verne.",
       "Ex B4.19 · B2 (MiniLM, e5 et ses préfixes, 📦 B2.28) · fiche §204.6 (« La recherche dense », « La recherche "
       "hybride et la fusion RRF », « Mesurer une recherche : recall@k et MRR »)", tracks="R, C",
       body=r"""**Contrat** (avec ta librairie `mylearn.retrieval`, tes fonctions de B4.17 et B4.18)
- `build_index_20(chunks, sentence_model, subset)` rend le dictionnaire `{"chunks", "bm25", "model", "embeddings"}` : les chunks, ton `BM25` ajusté sur `[tokenize(c) for c in chunks]`, le modèle reçu, et les embeddings des chunks, `wb.text.passage_embeddings(sentence_model, chunks, subset)` (B2 : le préfixe « passage: » pour e5, la moyenne masquée puis la norme 1, calculés une fois puis relus du cache sous le nom `subset`).
- `search_20(index, query, method="hybrid", k=5)` rend la liste des $k$ meilleurs identifiants de chunks (des `int` Python, le meilleur d'abord) :
  - `"bm25"` : les scores de ton BM25 pour `tokenize(query)`, triés par score décroissant, à égalité le plus petit identifiant (un tri stable) ;
  - `"dense"` : ton `cosine_top_k` de `index["model"].embed([query], kind="query")[0]` (le préfixe « query: » pour e5) contre les embeddings ;
  - `"hybrid"` : ta `reciprocal_rank_fusion` ($k = 60$) des `DEPTH_20` (20) premiers résultats de BM25 et de la recherche dense, coupée aux $k$ premiers ;
  - `ValueError` pour une autre méthode.
- `evaluate_20(index, items, method)` rend `{"recall@1", "recall@5", "mrr"}`, trois `float` Python : pour chaque question de `items`, les `DEPTH_20` premiers résultats de `search_20` et ses chunks pertinents, `wbrag.relevant_ids(index["chunks"], item)` ; puis ton `recall_at_k` (à 1 et à 5) et ton `mean_reciprocal_rank`.

La cellule de vérification construit ton index des chunks de référence avec MiniLM (les embeddings de B4.19, relus du cache), compare `search_20` à la recherche de référence (`wbrag.ReferenceIndex`) sur deux questions et trois méthodes, essaie un refus, et compare `evaluate_20` à un calcul de référence sur six questions. L'expérience évalue ensuite les trois méthodes sur Holmes avec MiniLM (les 30 questions, les 20 directes, les 10 paraphrasées), puis sur Verne avec e5 et avec MiniLM (les 10 questions en français ; environ 15 s d'embeddings e5 la première fois). Elle affiche aussi, précalculés, les résultats d'e5 sur Holmes (`E5_HOLMES_20` : environ 1 min d'embeddings sur CPU).

Dans tes notes : sur Holmes, quelle méthode préfères-tu, et sur quel groupe de questions la différence se joue-t-elle ? Le recall@1 et le recall@5 classent-ils les méthodes dans le même ordre ? Sur Verne, que gagne un modèle multilingue, et pourquoi MiniLM, entraîné sur de l'anglais, s'en sort-il si mal ? Avec 10 ou 30 questions, quel écart de recall@5 jugerais-tu significatif (fiche §204.5) ?""",
       given=None, todo=C.TODO_20, check=C.CHECK_20, solution=C.SOLUTION_20 + "\n\n\n" + C.CHECK_20,
       after=[("code", C.EXPERIMENT_20)]),

    Ex("B4.21", "📦", 3, 60, "Un RAG complet sur Holmes : récupération hybride, prompt avec citations, réponse d'un petit "
                            "modèle",
       "assembler un RAG qui cite ses passages, et vérifier ses citations contre les passages.",
       "Ex B4.20 · Ex B4.14 · fiche §204.7 (« Les quatre étapes », « Le contexte : combien, et dans quel ordre », "
       "« Évaluer et diagnostiquer »)", tracks="R, C",
       body=r"""Le générateur est Qwen2.5-0.5B-Instruct (494 M de paramètres, Apache 2.0, environ 1 Go à télécharger une fois) : SmolLM2-135M ne suit pas la consigne de citer. La cellule suivante donne `QWEN` (ses poids ne sont chargés qu'à la première génération que le cache ne connaît pas), `SYSTEM_21` (les règles : seulement les passages, une réponse courte suivie du numéro du passage entre crochets, « I don't know » sinon), `EXAMPLE_21` (un échange d'exemple qui montre le format de la réponse) et `QUESTIONS_21` (en mode rapide, 10 des 30 questions, dont 3 paraphrasées ; les 30 en mode complet).

**Contrat**
- `rag_messages_21(question, passages)` rend les messages : `{"role": "system", "content": SYSTEM_21}`, les deux messages de `EXAMPLE_21`, puis un message `user` au contenu exact `"Passages:\n[1] <premier passage>\n[2] <deuxième>…\n\nQuestion: <question>"` : les passages numérotés à partir de 1, dans l'ordre reçu, un par ligne, la question en dernier.
- `parse_reply_21(reply)` rend `(answer, cited)` : la réponse sans ses citations « [n] » (les blancs réduits à une espace, les blancs et les points retirés aux deux bouts), et la liste des numéros cités, dans l'ordre de leur première apparition, chacun une fois.
- `verify_21(answer, cited, passages)` rend `True` quand les citations tiennent : `cited` n'est pas vide, chaque numéro est un passage (de 1 à `len(passages)`), et au moins un passage cité contient **tous** les mots de contenu de la réponse : `wbrag.content_words(answer)`, non vide, inclus dans `set(tokenize(passage))`.
- `rag_21(question, index, llm, k=3)` enchaîne `search_20(index, question, "hybrid", k)`, les passages (`index["chunks"]`), `rag_messages_21`, une réponse gloutonne de 24 tokens au plus par le cache (`llm.generate(messages, max_new_tokens=24, cache=CACHE)`), `parse_reply_21` et `verify_21`, et rend `{"ids", "passages", "reply", "answer", "cited", "verified"}`.

La cellule de vérification compare `rag_messages_21` au prompt de référence, essaie `parse_reply_21` sur cinq réponses et `verify_21` sur huit cas. L'expérience fait tourner ton RAG sur `QUESTIONS_21` avec ton index de B4.20 (environ 40 s pour 10 questions la première fois, puis le cache) : pour chaque question, le passage pertinent est-il parmi les 3, la réponse est-elle juste (`wbrag.answer_in` : la réponse attendue ou une variante, en mots entiers), la citation est-elle vérifiée ? Elle compare enfin ta chaîne à la référence (`wbrag.ReferenceRAG`) sur la première question.

Dans tes notes : combien de réponses justes ne sont pas vérifiées, et pourquoi (le numéro cité, ou les mots de la réponse) ? Une réponse juste sans le bon passage dans les 3 : d'où vient-elle (fiche §204.5, contamination) ? Que vaut une citation que l'on ne vérifie pas ? La règle de `verify_21` accepte-t-elle une réponse fausse, ou refuse-t-elle une réponse juste ? Donne un exemple de chaque.""",
       given=C.GIVEN_21, todo=C.TODO_21, check=C.CHECK_21, solution=C.SOLUTION_21 + "\n\n\n" + C.CHECK_21,
       after=[("code", C.EXPERIMENT_21)]),
])


def _e5_rows() -> str:
    return "[\n" + "".join(f"    {row!r},\n" for row in (tuple(r) for r in E5_HOLMES)) + "]"


# B4.16 shows the full protocol (Qwen2.5, 8 passages), precomputed by history_chB4.py
PART_A.exercises[4].given = (C.GIVEN_16 + "\n# The full protocol, precomputed by tools/chapters/history_chB4.py with the "
                             "solution of this exercise\nFULL_16 = " + repr(HISTORY.get("p16_full", {})))

# B4.20 shows the e5 rows on Holmes, precomputed by history_chB4.py (≈ 1 min of embeddings on CPU)
PART_B.exercises[3].given = (C.GIVEN_20 + "\n# e5 on Holmes (precomputed by tools/chapters/history_chB4.py with the solution "
                             "of this exercise): (method, questions, scores)\nE5_HOLMES_20 = " + _e5_rows())

PARTS = [PART_A, PART_B]
