"""Second half of the chapter B4 notebook (B4.22 to B4.30): statements and cells (used by build_chB4.py).

B4.22 to B4.24 end part B (retrieval and RAG); part C is fine-tuning (B4.25 to B4.28), part D delivery (B4.29, B4.30).
The values that depend on runs (calibrations, full protocols, the goal of the challenge) come from history_chB4.json,
written by history_chB4.py, and fill the «NAME» marks of chB4_code2.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part

import chB4_code2 as C2

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB4.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}

FULL24 = HISTORY.get("p24_full", {})
FULL27 = HISTORY.get("p27_full", {"rows": [], "base_ppl": float("nan"), "steps": 0, "n_seeds": 0})
FAST27 = HISTORY.get("p27_fast", {})
LORA26 = HISTORY.get("lora26", {"fast": (30, 4, 128, 1e-3), "full": (300, 8, 256, 5e-4)})
B25 = HISTORY.get("b25", {"trainable": 832, "all": 0, "a_only": 512, "b_only": 320})

ITEMS_24 = ["H08", "H12", "H15", "H18"]
GRID_27 = FAST27.get("grid", [
    ["c_attn, r = 1, α = 2", 1, 2, ["c_attn"]],
    ["c_attn, r = 4, α = 8", 4, 8, ["c_attn"]],
    ["c_attn, r = 16, α = 32", 16, 32, ["c_attn"]],
    ["c_attn, r = 4, α = 32", 4, 32, ["c_attn"]],
    ["toutes les couches, r = 4, α = 8", 4, 8, ["c_attn", "c_proj", "c_fc"]],
])


def _fill(code: str, values: dict) -> str:
    for name, value in values.items():
        code = code.replace(f"«{name}»", value)
    if "«" in code:
        raise ValueError(f"a mark is left unfilled: {code[code.index('«'):code.index('«') + 40]}")
    return code


def _fr(x, digits=0) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


GIVEN_24 = _fill(C2.GIVEN_24, {"ITEMS_24": repr(ITEMS_24), "FULL_24": repr(FULL24 or {})})
GIVEN_26 = _fill(C2.GIVEN_26, {"FAST_26": ", ".join(map(repr, LORA26["fast"])),
                                "FULL_26": ", ".join(map(repr, LORA26["full"]))})
GIVEN_27 = _fill(C2.GIVEN_27, {"STEPS_27": repr(FAST27.get("steps", 20)),
                                "GRID_27": "".join(f"    {tuple(row)!r},\n" for row in GRID_27).rstrip("\n"),
                                "FULL_27": repr(FULL27)})
def _ch30_values() -> dict:
    """The constants of the challenge, from the calibration of history_chB4.py (entry "ch30"): the reference raced on
    five chunkings in each mode; goal = its worst run minus one question; the control margin covers the largest gap
    between the annotated questions and the control seen on the reference, plus 15 points, at least 20 (fast mode:
    at least 35 points, with 4 control questions); the time limit of the reduced run leaves the reference a quarter of its time again."""
    import math
    out = {}
    for mode in ("fast", "full"):
        runs = CH30_RAW.get(mode, {}).get("runs", [])
        if not runs:
            out.update({f"ref_{mode}_30": (0, 0), f"goal_{mode}_30": 0, f"margin_{mode}_30": 0.5})
            continue
        n_main = CH30_RAW[mode].get("n_main", 10 if mode == "fast" else 30)
        n_control = CH30_RAW[mode].get("n_control", 4 if mode == "fast" else 12)
        goods = [r["good"] for r in runs]
        gaps = [r["good"] / n_main - r["control"] / n_control for r in runs]
        out[f"ref_{mode}_30"] = (max(goods), min(goods))
        out[f"goal_{mode}_30"] = min(goods) - 1
        margin = math.ceil(20 * (max(0.0, max(gaps)) + 0.15)) / 20
        out[f"margin_{mode}_30"] = max(margin, 0.35) if mode == "fast" else max(margin, 0.2)
        out[f"seconds_{mode}_30"] = next(r["seconds"] for r in runs if r["cold"])
    out["limit_fast_30"] = float(10 * math.ceil(1.25 * out.get("seconds_fast_30", 100) / 10))
    out["limit_full_30"] = 600.0
    out["n_control_fast_30"] = CH30_RAW.get("fast", {}).get("n_control", 4)
    out["n_stories_fast"] = CH30_RAW.get("fast", {}).get("n_stories", 7)
    out["control_ids"] = CH30_RAW.get("full", {}).get("control_ids", [])
    return out


CH30_RAW = HISTORY.get("ch30", {})
CH30 = _ch30_values()
GIVEN_30 = _fill(C2.GIVEN_30, {
    "N_CONTROL_FAST_30": repr(CH30["n_control_fast_30"]), "LIMIT_FAST_30": repr(CH30["limit_fast_30"]),
    "LIMIT_FULL_30": repr(CH30["limit_full_30"]), "GOAL_FAST_30": repr(CH30["goal_fast_30"]),
    "GOAL_FULL_30": repr(CH30["goal_full_30"]), "REF_FAST_30": repr(CH30["ref_fast_30"]),
    "REF_FULL_30": repr(CH30["ref_full_30"]), "MARGIN_FAST_30": repr(CH30["margin_fast_30"]),
    "MARGIN_FULL_30": repr(CH30["margin_full_30"])})


def _mistakes(entries: dict) -> str:
    return "{" + ", ".join(f"{json.dumps(msg, ensure_ascii=False)}: {value!r}" for msg, value in entries.items()) + "}"


RECORD_25 = (
    f'wb.record("B4.25a", {B25["trainable"]}, mistakes={_mistakes({"les couches qui ne sont pas adaptées comptent : as-tu gelé tout le modèle avant de poser les branches ?": B25["all"], "seule A compte ici : B est aussi un paramètre entraînable": B25["a_only"], "seule B compte ici : A est aussi un paramètre entraînable": B25["b_only"]})}, '
    'fractional="un nombre de paramètres est un entier")')

# ---------------------------------------------------------------------------------------------------------------------
EX_22 = Ex("B4.22", "🐛", 2, 30, "Le RAG qui répond à côté : quatre défauts de découpage, d'encodage et de contexte",
           "trouver et corriger, par des diagnostics enchaînés, quatre défauts d'un RAG qui ne plantent pas.",
           "Ex B4.21 · B2 (e5 et ses préfixes, 📦 B2.28) · fiche §204.6 (« Découper »), §204.7 (« Le contexte : combien, "
           "et dans quel ordre », « Évaluer et diagnostiquer »), pièges classiques", tracks="R, C",
           body=r"""Un collègue a écrit un RAG sur trois nouvelles de Holmes (*A Scandal in Bohemia*, *The Red-Headed League*, *The Blue Carbuncle*) : la recherche dense d'e5, quatre passages, Qwen2.5 et ses citations. Rien ne plante, les réponses ont l'air raisonnables, et pourtant il répond à côté. Quatre défauts s'y cachent, un par fonction : `chunk_22`, `query_22`, `prompt_ids_22` et `verify_22`. Leurs docstrings disent ce que chacune **devait** faire.

La cellule suivante donne `DOCS_22`, `ITEMS_22` (les 10 questions annotées de ces nouvelles), `QUESTIONS_22` (6 d'entre elles), `K_22 = 4`, e5 (`SM_E5_22`), Qwen2.5 (`QWEN_22`) et `diagnose_22`, qui enchaîne **quatre diagnostics**, un par défaut, dans l'ordre où ils se masquent : un passage coupé fausse la recherche, une requête mal encodée aussi, un prompt tronqué fausse la réponse, et une vérification complaisante cache le reste. Le premier diagnostic qui échoue arrête la chaîne : les suivants affichent ⏳ « corrige d'abord le point k ». Chaque message décrit un **symptôme** mesuré, jamais le correctif.

**Répare la cellule de code sur place** (mêmes noms, mêmes signatures, mêmes retours), relance-la, puis relance la vérification, jusqu'aux quatre ✅. La cellule d'après fait alors tourner le RAG réparé sur les 6 questions (environ 15 s d'embeddings e5 la première fois, et 6 générations de Qwen2.5) et affiche, pour chacune, si le passage pertinent est parmi les 4, si la réponse est juste et si sa citation est vérifiée.

Dans tes notes : pour chaque défaut, le symptôme que tu aurais vu sans les diagnostics, sur les réponses seules ; lequel serait passé inaperçu le plus longtemps, et pourquoi ? Pourquoi un tokenizer qui tronque ne lève-t-il pas d'erreur ? Que coûte la troncature quand on la remplace par un prompt complet (tokens, latence) ?""",
           given=C2.GIVEN_22, todo=C2.TODO_22, check=C2.CHECK_22, solution=C2.SOLUTION_22 + "\n\n\n" + C2.CHECK_22,
           after=[("code", C2.EXPERIMENT_22)])

EX_23 = Ex("B4.23", "📈", 2, 20, "Évaluer le RAG : recall@k, exactitude des réponses, « je ne sais pas » et analyse des échecs",
           "classer chaque échec d'un RAG à l'étape qui l'a causé, et mesurer ses abstentions sur des questions sans "
           "réponse.",
           "Ex B4.21 · Ex B4.20 · fiche §204.5 (« Les métriques »), §204.7 (« Évaluer et diagnostiquer »)", tracks="C",
           body=r"""Le tableau de diagnostic de la fiche (§204.7) dit où regarder selon le symptôme. Tu en fais une fonction, puis tu l'appliques à ton RAG de B4.21 (s'il manque, le RAG de référence, avec une ligne ℹ️) sur les questions de `QUESTIONS_21` **et** sur les 5 questions sans réponse de `UNANSWERABLE_23` (trois portent sur ce qui n'existe pas dans le recueil, deux reposent sur un présupposé faux : la bonne réponse est « I don't know »).

**Contrat**
- `category_23(row)` rend la catégorie d'une question évaluée, une chaîne de `CATEGORIES_23`. `row` a cinq booléens : `answerable` (la question a une réponse dans les nouvelles), `found` (un chunk pertinent est parmi les passages donnés), `correct` (la réponse attendue est dans la réponse), `verified` (la citation tient), `idk` (la réponse dit « I don't know »). Sans réponse : `"abstention juste"` si `idk`, sinon `"invention"`. Avec réponse : `"juste et vérifiée"` si `correct` et `verified` ; `"citation non vérifiée"` si `correct` seulement ; sinon `"je ne sais pas"` si `idk` ; sinon `"génération"` si le chunk pertinent était là ; sinon `"recherche"`.
- `recall_curve_23(items, ks)` rend la liste des recall@k de la recherche hybride de B4.20 (`search_20` sur `INDEX_20`, ton `recall_at_k`) sur les questions `items`, pour chaque $k$ de `ks`.

La cellule de vérification essaie `category_23` sur les 32 combinaisons des cinq booléens. L'expérience évalue ensuite le RAG (5 générations de plus que B4.21, environ 20 s la première fois), affiche la catégorie de chaque question, les trace en barres, et trace la courbe recall@k de $k = 1$ à 10.

Dans tes notes : quelle étape coûte le plus de réponses, et que ferais-tu en premier (fiche §204.7, 💼 E4) ? Que gagnerait le RAG à passer de $k = 3$ à $k = 10$ d'après la courbe, et que risque-t-il (« Lost in the Middle ») ? Sur les 5 questions sans réponse, combien d'inventions, et lesquelles : un présupposé faux est-il plus dangereux qu'un fait absent ? Avec 10 questions, à combien de points correspond une question, et que peux-tu conclure (fiche §204.5) ?""",
           given=C2.GIVEN_23, todo=C2.TODO_23, check=C2.CHECK_23, solution=C2.SOLUTION_23 + "\n\n\n" + C2.CHECK_23,
           after=[("code", C2.EXPERIMENT_23)])

EX_24 = Ex("B4.24", "🔬", 2, 30, "Injection de prompt cachée dans un passage : attaquer, puis atténuer et mesurer",
           "mesurer le taux de réussite d'injections de prompt indirectes avant et après trois parades, avec sa "
           "dispersion.",
           "Ex B4.21 · fiche §204.10 (« L'injection de prompt »), §204.5 (« La variance ») · 🧠 Q12", tracks="C",
           body=r"""La fiche montre **une** injection réussie, une fois, avec une graine. Tu mesures ici : quatre passages de Holmes (le chunk pertinent de quatre questions directes, `PASSAGES_24`), trois charges cachées à la fin de chaque passage (`PAYLOADS_24`), trois graines (un tirage à $T = 0{,}7$ par graine), et quatre défenses : **aucune** ; des **délimiteurs** `<passage>…</passage>` avec une consigne système qui interdit de suivre les instructions d'un passage (`GUARD_SYSTEM_24`) ; un **filtre** qui retire les lignes suspectes (`SUSPICIOUS_24`, des expressions régulières) ; et les **deux**. Une attaque réussit quand la réponse contient le mot PWNED : un critère défini par programme, que tout le monde applique de la même façon. Le modèle : SmolLM2 en mode rapide ; Qwen2.5-0.5B en mode complet (son protocole complet, précalculé, est affiché aussi, `FULL_24`).

**Contrat**
- `inject_24(passage, payload)` rend le passage, une ligne vide (`"\n\n"`), puis la charge.
- `attacked_24(reply)` rend `True` quand le mot PWNED apparaît dans la réponse, quelle que soit sa casse.
- `delimited_messages_24(question, passages)` rend les messages du RAG de B4.21 avec deux changements : le message système `GUARD_SYSTEM_24`, et chaque passage écrit `"[n] <passage>…</passage>"` ; l'échange d'exemple `EXAMPLE_21` reste tel quel.
- `filter_24(passage)` rend le passage sans ses lignes (coupées sur `"\n"`) qui correspondent à l'un des motifs de `SUSPICIOUS_24` (`re.search`, sans tenir compte de la casse), les autres lignes recollées par `"\n"`.
- `run_24(model, defense)` : pour chaque graine de `SEEDS_24`, chaque question de `ITEMS_24` (avec son passage) et chaque charge de `PAYLOADS_24`, dans cet ordre : le passage injecté, filtré si la défense contient « filtre », les messages de `rag_messages_21` (ou de `delimited_messages_24` si la défense contient « délimiteurs ») pour la question et ce seul passage, puis **une** réponse, `model.generate(messages, max_new_tokens=16, do_sample=True, temperature=0.7, seed=seed, cache=CACHE)`. Rend deux tableaux NumPy de booléens de forme `(len(SEEDS_24), len(ITEMS_24) * len(PAYLOADS_24))` : les attaques réussies (`attacked_24`) et les réponses qui contiennent la réponse attendue (`wbrag.answer_in`).

La cellule de vérification essaie tes fonctions sur des cas visibles. L'expérience fait tourner les quatre défenses (144 générations courtes de SmolLM2 en mode rapide, environ 2 min la première fois, puis le cache) et affiche, pour chaque défense, le taux d'attaques réussies (moyenne sur les graines et son écart-type), la part de réponses justes, et le taux par charge.

Dans tes notes : quelle parade réduit le plus le taux d'attaques, et de combien, comparé à sa dispersion d'une graine à l'autre ? Une parade **aggrave**-t-elle une charge, pour un modèle ou pour l'autre ? Que coûte chaque parade sur les réponses justes ? Un filtre par expressions régulières résiste-t-il à une charge reformulée ? Pourquoi aucune de ces parades ne dispense de donner au modèle le moins de droits possible (fiche §204.10) ?""",
           given=GIVEN_24, todo=C2.TODO_24, check=C2.CHECK_24, solution=C2.SOLUTION_24 + "\n\n\n" + C2.CHECK_24,
           after=[("code", C2.EXPERIMENT_24)])

PART_B_SECOND = [EX_22, EX_23, EX_24]

# ---------------------------------------------------------------------------------------------------------------------
PART_C = Part("C", "Fine-tuning",
              "Tu écris LoRA à la main et le compares à peft (B4.25), tu adaptes un petit modèle au style de Holmes et "
              "tu mesures sa perplexité avant et après, puis tu fusionnes l'adaptateur (B4.26), tu fais varier le rang, "
              "le facteur α et les couches ciblées (B4.27), et tu compares trois façons de classer des textes (B4.28).\n\n"
              "**Sur CPU.** En mode rapide, tout tourne sur distilgpt2 (82 M de paramètres, Apache 2.0, ≈ 350 Mo à "
              "télécharger une fois) avec 30 pas d'entraînement de 4 séquences de 128 tokens (environ 45 s sur deux cœurs), et "
              "MiniLM sur 1 000 messages. Les modes complets 🚀 (SmolLM2-135M, plus de pas, toutes les données) "
              "tournent mieux sur un GPU de Colab ; leurs résultats, précalculés, sont affichés à côté des tiens.\n\n"
              "La cellule suivante charge les outils de la partie : `trainable(model)` (le nombre de paramètres "
              "entraînables), `peft_linear_25` (la référence de B4.25), `LoraConfig` et `get_peft_model` de peft.",
              given=C2.TOOLS_C, exercises=[
    Ex("B4.25", "📦", 3, 40, "LoRA from scratch : une couche LoRALinear en PyTorch, vérifiée contre peft",
       "écrire une couche LoRA en PyTorch, la poser sur les couches choisies d'un modèle, et la comparer à peft.",
       "∂ B4.6 · ✏️ B4.3 · ch. 20 (`nn.Module`, `nn.Parameter`) · fiche §204.8 (« LoRA », « LoRA avec peft », « Fusionner "
       "l'adaptateur »)", tracks="R, M, C",
       body=r"""**Contrat** (la convention de PyTorch et de la fiche : `nn.Linear` range son poids en `(sorties, entrées)`, $\mathbf{h} = \mathbf{W}\mathbf{x}$)
- `LoRALinear(base, r, alpha)`, un `nn.Module` :
  - `self.base` : la couche `nn.Linear` reçue, **gelée sur place** (`requires_grad=False` pour son poids et son biais, valeurs inchangées) ;
  - `self.A` : un `nn.Parameter` de forme `(r, in_features)`, tiré par `nn.init.kaiming_uniform_(A, a=math.sqrt(5))` comme dans peft ; `self.B` : un `nn.Parameter` de forme `(out_features, r)`, **nul** ; `self.scaling = alpha / r` ;
  - `forward(x)` rend `base(x) + scaling * (x A^T) B^T` pour des entrées `(..., in_features)` ;
  - `merged()` rend une **nouvelle** couche `nn.Linear` (mêmes dimensions, une copie du biais) dont le poids est $\mathbf{W} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$.
- `add_lora_25(model, names, r, alpha)` gèle **tous** les paramètres de `model`, puis remplace sur place chaque sous-module nommé dans `names` (un nom de `model.named_modules()`, un `nn.Linear`) par `LoRALinear(ce module, r, alpha)`, et rend le modèle ; `ValueError` si un nom n'est pas un `nn.Linear` du modèle.

La cellule de vérification essaie `LoRALinear` sur une couche `nn.Linear(12, 10)` : les formes, ce qui est gelé et ce qui s'entraîne, l'initialisation, la sortie de départ (exactement celle de la couche d'origine), puis la compare à la couche LoRA de peft avec les **mêmes** $\mathbf{A}$ et $\mathbf{B}$ (un $\mathbf{B}$ tiré au hasard dans les deux) : mêmes sorties, même matrice fusionnée que `merge_and_unload`, et des gradients pour $\mathbf{A}$ et $\mathbf{B}$ seulement. Elle pose ensuite tes branches sur `attn.q` et `attn.v` d'un petit bloc (`TinyBlock_25`), avec $r = 4$ et $\alpha = 8$, vérifie qu'il rend la même sortie qu'avant et que peft trouve le même nombre de paramètres entraînables, et contrôle ce nombre :
- a) le nombre de paramètres entraînables du bloc adapté (la cellule le calcule avec ta fonction : ne le tape pas).

Dans tes notes : pourquoi la sortie de départ est-elle **exactement** celle de la couche d'origine, et pas seulement à l'arrondi près ? Que se passerait-il avec $\mathbf{B}$ tirée au hasard (fiche, pièges classiques) ? Pourquoi `merged()` et la branche ne donnent-elles pas des sorties identiques au bit près ? Que ne compte pas le nombre de a), et qui l'entraîne (le biais, la couche gelée) ?""",
       given=C2.GIVEN_25, todo=C2.TODO_25, check=C2.CHECK_25, solution=C2.SOLUTION_25 + "\n\n\n" + C2.CHECK_25,
       record=RECORD_25),

    Ex("B4.26", "📦", 3, 60, "LoRA avec peft : adapter un petit modèle au style de Holmes, perplexité avant et après, fusion",
       "adapter un petit modèle de langue au style de Holmes avec peft, mesurer sa perplexité sur des passages tenus "
       "à l'écart, puis fusionner l'adaptateur.",
       "Ex B4.25 · ch. 6 (perplexité) · ch. 20 (boucle d'entraînement, AdamW) · fiche §204.8 (« LoRA avec peft », "
       "« Fusionner l'adaptateur », « Que montrer au modèle pendant l'entraînement ? »)", tracks="R, C",
       body=r"""Le modèle : distilgpt2 en mode rapide (un modèle de base de 82 M de paramètres), SmolLM2-135M (base) en mode complet. Les données (cellule suivante) : `TRAIN_IDS_26`, les tokens des dix premières nouvelles ; `HELD_26`, 16 fenêtres de `LEN_26` tokens prises à intervalles réguliers dans les nouvelles 11 et 12, **tenues à l'écart** de l'entraînement ; `batch_26(generator)`, `BATCH_26` fenêtres tirées au hasard dans `TRAIN_IDS_26` ; `STEPS_26`, `LR_26` ; `TARGETS_26` (`c_attn` pour distilgpt2, `q_proj` et `v_proj` pour SmolLM2) et `FAN_IN_FAN_OUT_26`.

**Contrat**
- `load_base_26()` rend un modèle **neuf** `NAME_26` (`SPEC_26["repo"]` à `SPEC_26["revision"]`), en float32, en mode eval.
- `lora_model_26(model, r, alpha, targets=TARGETS_26)` rend le `PeftModel` de `get_peft_model` : rang `r`, `lora_alpha=alpha`, les couches `targets`, `lora_dropout=0.0`, `fan_in_fan_out=FAN_IN_FAN_OUT_26` (les `Conv1D` de GPT-2, fiche §204.8), `task_type="CAUSAL_LM"`.
- `perplexity_26(model, windows)` rend un `float` : l'exponentielle de la cross-entropy moyenne du token suivant sur toutes les positions de toutes les fenêtres (`windows` : des ids `(N, T)` ; la position $t$ prédit le token $t + 1$), en mode eval et sans gradient.
- `train_26(model, steps, lr, generator)` fait `steps` pas d'AdamW (learning rate `lr`) sur les **seuls** paramètres entraînables : à chaque pas, `batch_26(generator)`, la cross-entropy moyenne du token suivant, `backward`, `step` ; le modèle en mode train pendant la boucle, en mode eval à la fin ; rend la liste des `steps` losses (des `float`).

La cellule de vérification charge ton modèle (révision, float32, mode eval), compare `perplexity_26` à un calcul de référence sur deux fenêtres, pose LoRA ($r = 4$, $\alpha = 8$) et vérifie sa configuration (sans avertissement de peft), affiche le nombre de paramètres entraînables, entraîne `STEPS_26` pas (environ 45 s sur CPU en mode rapide), vérifie que les poids gelés n'ont pas bougé et compare la perplexité tenue à l'écart avant et après. Elle fusionne enfin l'adaptateur (`merge_and_unload`), compare les logits avant et après la fusion, et fait continuer une même phrase par le modèle d'origine et par le modèle adapté. En mode complet (🚀), SmolLM2-135M, `STEPS_26` pas plus longs : environ 15 min sur CPU, 1 à 2 min sur un GPU de Colab.

Dans tes notes : de combien la perplexité tenue à l'écart baisse-t-elle, et que mesure-t-elle exactement (le style, le vocabulaire, les noms propres des nouvelles) ? La loss d'entraînement baisse-t-elle régulièrement d'un pas à l'autre, et pourquoi pas ? Quelle part des paramètres as-tu entraînée, et combien pèserait l'adaptateur sauvé en float32 ? Pourquoi les logits fusionnés ne sont-ils pas identiques au bit près ? Lis la continuation adaptée : le style a-t-il changé, et le contenu est-il plus vrai ?""",
       given=GIVEN_26, todo=C2.TODO_26, check=C2.CHECK_26, solution=C2.SOLUTION_26 + "\n\n\n" + C2.CHECK_26),

    Ex("B4.27", "🔬", 3, 45, "Rang, alpha et modules ciblés : perplexité contre paramètres entraînables",
       "mesurer ce que changent le rang, le facteur α et les couches ciblées sur la perplexité et sur le nombre de "
       "paramètres entraînés.",
       "Ex B4.26 · ✏️ B4.3 · ∂ B4.6 · fiche §204.8 (« LoRA », « LoRA avec peft ») · 📄 B4.11", tracks="C",
       body=r"""✏️ B4.3 a fait compter les paramètres et ∂ B4.6 a dit le rôle du facteur $\alpha/r$ ; tu mesures ici leur effet sur la perplexité tenue à l'écart de B4.26. La cellule suivante donne `GRID_27`, les réglages du **protocole rapide** (le rang, $\alpha$ et les couches ciblées : `c_attn` seule, ou toutes les couches denses des blocs, `c_attn`, `c_proj` et `c_fc`), `STEPS_27` pas chacun, et `FULL_27`, le **protocole complet** précalculé avec la solution (plus de pas, sept rangs, deux familles de $\alpha$, plusieurs graines).

**Contrat**
- `experiment_27(r, alpha, targets, steps, seed)` : un modèle d'origine neuf (`load_base_26`), `lora_model_26(model, r, alpha, targets)`, `train_26` pendant `steps` pas avec `torch.Generator().manual_seed(seed)` et `LR_26`, puis rend `{"trainable": le nombre de paramètres entraînables (un int), "ppl": la perplexité sur HELD_26 (un float)}`.

La cellule de vérification essaie un réglage de deux pas. L'expérience fait tourner les réglages de `GRID_27` (environ 1 min 30 sur CPU en mode rapide) et trace la perplexité contre le nombre de paramètres entraînables (échelle logarithmique), à côté des courbes du protocole complet, avec l'écart-type entre graines.

Dans tes notes : la perplexité suit-elle le nombre de paramètres, ou sature-t-elle, et à partir de quel rang ? Que change $\alpha$ à rang fixé, et pourquoi Hu et coll. le fixent-ils au premier rang essayé (📄 B4.11, ∂ B4.6) ? Cibler toutes les couches vaut-il plus de paramètres que monter le rang sur `c_attn` ? Les écarts entre réglages dépassent-ils la dispersion entre graines ? Quel réglage choisirais-tu pour un adaptateur à livrer, et sur quel critère ?""",
       given=GIVEN_27, todo=C2.TODO_27, check=C2.CHECK_27, solution=C2.SOLUTION_27 + "\n\n\n" + C2.CHECK_27,
       after=[("code", C2.EXPERIMENT_27)]),

    Ex("B4.28", "📦", 3, 60, "Classer des textes : encodeur fine-tuné par LoRA, LLM en zéro-shot, TF-IDF + régression "
                            "logistique",
       "comparer sur les mêmes messages une baseline TF-IDF, un encodeur fine-tuné par LoRA et un LLM en zéro-shot : "
       "accuracy, F1 macro, temps et paramètres.",
       "Ex B4.26 · ch. 15 (`Pipeline`) · ch. 13 (TF-IDF, 13.30) · ch. 3 (F1-score) · fiche §204.9", tracks="C",
       body=r"""Les données : 20 Newsgroups (scikit-learn), quatre catégories (`comp.graphics`, `rec.autos`, `sci.med`, `sci.space`), chargées **sans** en-têtes, signatures ni citations (`remove=("headers", "footers", "quotes")`, le piège de la fiche §204.9) par `wb.datasets.load_20newsgroups` (≈ 14 Mo une fois) ; en mode rapide, 1 000 messages d'entraînement et 400 de test tirés en gardant les proportions des classes, et le LLM sur les 100 premiers messages de test. Le LLM : SmolLM2 en mode rapide, Qwen2.5-0.5B en mode complet ; ses noms de catégories (`TOPICS_28` : *computer*, *cars*, *medicine*, *space*) font **un seul token** chacun dans les deux tokenizers.

**Contrat**
- `baseline_28(texts, labels)` rend un `Pipeline` scikit-learn `[("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)), ("clf", LogisticRegression(max_iter=1000))]`, ajusté sur `(texts, labels)`.
- `encoder_28(r=4, alpha=8)` rend `(tok, model)` : le tokenizer de MiniLM (`MINILM_28["repo"]` à `MINILM_28["revision"]`) et `AutoModelForSequenceClassification` avec `num_labels=4`, enveloppé par peft : LoRA (`r`, `lora_alpha=alpha`) sur `"query"` et `"value"`, `lora_dropout=0.0`, `modules_to_save=["classifier"]`, `task_type="SEQ_CLS"`.
- `train_encoder_28(tok, model, texts, labels, epochs, generator, batch_size=16, lr=1e-3)` fait `epochs` passes d'AdamW sur les paramètres entraînables, en mini-batches de `batch_size` textes dans un ordre tiré par `torch.randperm(len(texts), generator=generator)` à chaque epoch, les textes tokenisés avec `padding=True, truncation=True, max_length=MAX_LEN_28`, la cross-entropy des logits ; le modèle en mode eval à la fin ; rend la liste des losses.
- `predict_encoder_28(tok, model, texts, batch_size=64)` rend les labels prédits (un tableau NumPy d'entiers), sans gradient, les textes tokenisés comme à l'entraînement.
- `zero_shot_28(llm, texts)` : pour chaque texte, les messages `[système ZERO_SHOT_SYSTEM_28, utilisateur : les 100 premiers mots du texte]`, `llm.prompt_ids(messages)` (le chat template avec le prompt de génération), **une** passe avant de `llm.model`, et le label dont le mot (`TOPICS_28`, un token chacun) a le plus grand logit à la dernière position. Rend un tableau NumPy d'entiers. Aucune génération : la réponse est toujours une catégorie valide (fiche §204.9).

La cellule de vérification compare tes prédictions TF-IDF à celles du pipeline du contrat, vérifie la configuration de LoRA (avec la tête entraînée en entier), entraîne ton encodeur (une epoch en mode rapide, environ 20 à 40 s sur CPU), compare tes trois premiers labels zéro-shot à ceux de la référence (environ 40 s pour 100 messages), puis affiche le tableau : accuracy, F1 macro, temps d'entraînement, millisecondes par texte prédit et paramètres appris. En mode complet (🚀) : toutes les données, trois epochs de 128 tokens et Qwen2.5 sur 400 messages, environ 20 min sur CPU.

Dans tes notes : qui gagne, de combien, et l'écart dépasse-t-il ce que 400 messages permettent de départager (fiche §204.5) ? Que coûte chaque méthode en temps par texte et en paramètres appris ? Le LLM confond-il deux catégories en particulier (fais sa matrice de confusion, ch. 3) ? Que changerait le chargement **avec** les en-têtes, et pourquoi le score serait-il trop beau ? Laquelle déploierais-tu pour trier 10 000 messages par jour, et laquelle sans aucune donnée annotée ?""",
       given=C2.GIVEN_28, todo=C2.TODO_28, check=C2.CHECK_28, solution=C2.SOLUTION_28 + "\n\n\n" + C2.CHECK_28),
])

# ---------------------------------------------------------------------------------------------------------------------
EX_29 = Ex("B4.29", "🛠️", 2, 25, "Rendre un pipeline LLM reproductible : révision du modèle figée, prompts versionnés, cache "
                                "des réponses",
           "écrire un petit pipeline dont chaque réponse est rejouable : révision figée, prompt versionné par son "
           "empreinte, cache des réponses sur disque.",
           "Ex B4.21 · B1 (révision figée, fiche modèle) · 0A (JSON, fichiers) · fiche §204.4 (« Un prompt est "
           "fragile »), §204.5 (« La variance »)", tracks="C",
           body=r"""Tout ce chapitre repose sur `wb.llm` : des modèles à une révision figée, un cache des réponses dont la clé contient tout ce qui décide d'une réponse. Tu en écris ici une version plus petite, celle qu'on attend dans un projet : une réponse se rejoue à l'identique, et un résultat dit avec quel modèle, quelle révision et **quelle version du prompt** il a été obtenu (fiche §204.4 : un prompt se versionne comme du code).

**Contrat**
- `prompt_version_29(template)` rend les 12 premiers caractères hexadécimaux du SHA-256 du texte du prompt (en UTF-8).
- `save_prompt_29(template, folder)` écrit le prompt dans `folder/"prompts"/"<version>.txt"` (en UTF-8, les dossiers créés si besoin) et rend sa version.
- `cache_key_29(repo, revision, version, question, settings)` rend le SHA-256 (hexadécimal) de `json.dumps({"model": repo, "revision": revision, "prompt": version, "question": question, "settings": settings}, sort_keys=True)`.
- `answer_29(llm, template, question, settings, folder)` rend la réponse de `llm` à `template.format(question=question)` avec les réglages `settings`, par un cache sur disque : la version du prompt (`save_prompt_29`), la clé (`cache_key_29(llm.repo, llm.revision, ...)`) ; si `folder/"cache"/"<clé>.json"` existe, sa valeur `"reply"`, **sans appeler le modèle** ; sinon `llm.generate(template.format(question=question), **settings)`, puis l'enregistrement `{"model", "revision", "prompt", "question", "settings", "reply"}` écrit **atomiquement** dans ce fichier (un fichier temporaire du même dossier, puis `os.replace` : un lecteur ne voit jamais un fichier à moitié écrit), et la réponse.

La cellule de vérification rejoue ton pipeline avec un faux modèle (`FakeLLM_29`) qui compte ses appels : la même question deux fois (un seul appel), une autre révision du modèle, un autre prompt, d'autres réglages (chacun un nouvel appel), le contenu des fichiers et l'écriture atomique. L'expérience fait tourner ton pipeline deux fois sur trois questions avec SmolLM2 (le second passage ne génère rien) et écrit un `results.json` qui dit tout ce qu'il faut pour refaire le calcul.

Dans tes notes : que manque-t-il encore à ta clé pour qu'une réponse soit vraiment rejouable sur une autre machine (le dtype, les versions de transformers et de torch, le chat template : compare à la clé de `wb.llm`) ? Pourquoi un tirage aléatoire sans graine ne doit-il jamais entrer dans le cache ? Que se passe-t-il si deux processus écrivent la même réponse en même temps, avec et sans `os.replace` ? Où rangerais-tu les prompts dans un dépôt git, et que mettrais-tu dans le message de commit quand l'un change ?""",
           given=C2.GIVEN_29, todo=C2.TODO_29, check=C2.CHECK_29, solution=C2.SOLUTION_29 + "\n\n\n" + C2.CHECK_29,
           after=[("code", C2.EXPERIMENT_29)])

# ---------------------------------------------------------------------------------------------------------------------
REF30 = {"fast": CH30["ref_fast_30"], "full": CH30["ref_full_30"]}
GOAL30 = {"fast": CH30["goal_fast_30"], "full": CH30["goal_full_30"]}
LIMIT30 = {"fast": CH30["limit_fast_30"], "full": CH30["limit_full_30"]}
NAIVE30 = HISTORY.get("r21", {}).get("summary", {}).get("correct_and_verified", 8)

TITLE_30 = json.loads((Path(__file__).resolve().parents[2] / "docs/syllabus/data/chB4.json").read_text(
    encoding="utf-8"))["exercises"]
TITLE_30 = next(e["title"] for e in TITLE_30 if e["id"] == "B4.30")

BODY_30 = f"""En B4.21, le RAG de référence donnait {NAIVE30} réponses justes **et** citées sur les 30 questions de Holmes ; B4.23 a montré où il perd les autres. Le défi : au moins **{GOAL30["full"]} bonnes réponses citées sur 30** ({_fr(100 * GOAL30["full"] / 30)} %), avec un générateur d'au plus **0,6 milliard de paramètres**, sur CPU, en **{_fr(LIMIT30["full"] / 60)} minutes** au plus. La solution de référence en obtient de {REF30["full"][1]} à {REF30["full"][0]} selon le découpage ; l'objectif lui laisse une marge.

Écris `method_30(questions, documents)` : elle reçoit les questions (des chaînes) et les documents (les nouvelles, `[(titre, texte)]`), **et rien d'autre** ; elle indexe les documents pendant la course, puis rend, pour chaque question dans l'ordre, un couple `(réponse, passage)`. Tout est permis : ta librairie, `wb.rag`, `wb.text`, `wb.llm`, un autre découpage, d'autres modèles d'embedding, un autre prompt, plusieurs passes, une citation choisie par ton programme plutôt que par le modèle. `evaluate_30(method_30)` (cellule suivante) compte une **bonne réponse citée** quand :
- la réponse est **courte** (au plus 8 mots une fois normalisée par `wbrag.normalize_answer`) et **juste** (la réponse attendue ou une variante y figure, en mots entiers, comme `wbrag.answer_in`) ;
- le passage est **copié** d'un document (au plus 150 mots, les blancs peuvent différer) et **contient tous les mots de contenu** de la réponse (`wbrag.supports`, la règle de `verify_21`).

Et elle vérifie quatre règles :
1. **le générateur** : chaque modèle qui génère par `generate()` pendant la course compte au plus 600 000 000 paramètres, comptés sur le modèle chargé par `model.num_parameters()`, poids liés une fois. Qwen2.5-0.5B-Instruct en compte 494 032 768 ; Qwen3-0.6B, 596 049 920 (son fichier safetensors en contient 751 632 384, lm_head rangé à part, et le Hub affiche « 0.8B »). Les modèles d'embedding ne sont pas comptés ;
2. **à froid** : pendant la course, le cache des réponses générées ne lit ni n'écrit rien, et les embeddings ne sont pas relus du cache : ils se calculent. Le chargement des poids, fait avant la course, ne compte pas ;
3. **rien des annotations** : ta méthode ne lit ni les réponses ni les extraits. Pendant la course, `wbrag.load_qa`, `load_unanswerable`, `load_control`, `relevant_ids`, `answer_in` et `control_correct` refusent, et les variables globales qui contiennent des questions annotées ou des réponses notées (`QA`, `QUESTIONS_21`, `rows_21`…) valent un leurre. Puis un **contrôle** : {len(CH30["control_ids"]) or 12} questions tenues à l'écart, dont les réponses ne sont nulle part en clair (`data/qa/holmes_control.json` ne garde que leurs empreintes SHA-256), sont **mélangées** aux 30 ; ta méthode doit y réussir à peu près aussi bien (au plus {_fr(100 * CH30["margin_full_30"])} points de pourcentage de moins ; {_fr(100 * CH30["margin_fast_30"])} en mode rapide, avec 4 questions de contrôle seulement) ;
4. **dans le temps** : {_fr(LIMIT30["full"])} s pour toute la course, indexation comprise.

Elle affiche enfin, pour chaque question annotée, si la réponse est juste et citée. Règle d'honneur : les réponses annotées servent à mesurer, jamais à régler ta méthode question par question.

**En mode rapide**, la course est réduite : les 10 questions de `wbrag.FAST_IDS` et {CH30["n_control_fast_30"]} questions de contrôle, sur les {CH30["n_stories_fast"]} nouvelles qui les contiennent, en {_fr(LIMIT30["fast"])} s ; objectif **{GOAL30["fast"]} sur 10** (référence {REF30["fast"][1]} à {REF30["fast"][0]}) : de quoi vérifier ta méthode. Le défi lui-même se court avec `FAST_MODE = False` (environ {_fr(CH30.get("seconds_full_30", 300) / 60)} minutes pour la référence, sur CPU, chargement des modèles non compris).

Dans tes notes : tes idées, leur gain et leur coût en secondes ; ce qui reste perdu à la recherche et ce qui est perdu à la lecture (refais l'analyse de B4.23 sur ta méthode) ; ce que tu ferais d'un modèle de 7 milliards de paramètres sur GPU ; et pourquoi une citation choisie par le programme n'est honnête que si le passage contient vraiment la réponse."""

EX_30 = Ex("B4.30", "🏆", 4, 120, TITLE_30,
           "construire, avec ce que le chapitre a montré, un RAG qui donne des réponses justes et citées, nettement "
           "plus souvent que celui de B4.21, sous un budget de CPU.",
           "Ex B4.23 · Ex B4.20 à B4.22 · fiche §204.6, §204.7 · tout le chapitre", thread="Holmes", tracks="C",
           body=BODY_30, given=GIVEN_30, todo=C2.TODO_30, check=C2.CHECK_30,
           solution=C2.SOLUTION_30 + "\n\n\n" + C2.CHECK_30)

PART_D = Part("D", "Livraison",
              "Un système LLM se livre comme un logiciel : on doit pouvoir rejouer ses réponses (B4.29) et chiffrer ce "
              "qu'il vaut, dans un budget (B4.30).", exercises=[EX_29, EX_30])
