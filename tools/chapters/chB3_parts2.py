"""Parts C to E of the chapter B3 notebook, second half (B3.21 to B3.30): statements and cells (used by build_chB3.py,
which appends them to the PARTS of chB3_parts.py).

Part C reloads the mini-GPT saved by B3.20 (curves, French prompt, KV cache, attention maps); part D, the experiments
(ablations, BPE against characters, long context); part E, the ViT, the causality test and the challenge. The long
protocols are precomputed by history_chB3.py with the solutions (history_chB3.json); the placeholders «NAME» of the
cells of chB3_code2.py are filled here.
"""

from __future__ import annotations

import json
import re

import numpy as np
from chapter_kit import Ex, Part, guarded

import chB3_code2 as C
from chB3_parts import HISTORY

LETTERS_25 = {"no_position": "P", "no_residual": "R", "post_ln": "L", "one_head": "H"}


def _letters(answer: str, hints: dict) -> str:
    return "{" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items() if letter != answer) + "}"


def _fill(src: str, values: dict) -> str:
    for key, value in values.items():
        src = src.replace(f"«{key}»", value)
    left = re.findall(r"«[A-Z0-9_]+»", src)
    if left:
        raise ValueError(f"unfilled placeholders {left}")
    return src


# --- 📈 B3.21: the two full runs, smoothed -------------------------------------------------------------------------
def _full_21() -> str:
    out = {}
    for name, key in (("sans dropout", "gpt_full_nodropout"), ("dropout 0,2", "gpt_full")):
        h = HISTORY.get(key, {}).get("history")
        if not h:
            out[name] = {"train_iter": [25], "train": [4.5], "val_iter": [0], "val": [4.5]}
            continue
        tl = np.asarray(h["train_loss"])
        n = len(tl) // 50
        out[name] = {"train_iter": [int(50 * i + 25) for i in range(n)],
                     "train": [round(float(v), 4) for v in tl[:50 * n].reshape(n, 50).mean(axis=1)],
                     "val_iter": list(h["val_iter"]), "val": [round(float(v), 4) for v in h["val_loss"]]}
    return json.dumps(out, ensure_ascii=False)


def _answers_21() -> dict:
    nd = HISTORY.get("gpt_full_nodropout", {}).get("history")
    dr = HISTORY.get("gpt_full", {}).get("history")
    if not (nd and dr):
        return {"best": 2500, "early": False, "close": True}
    best = nd["val_iter"][int(np.argmin(nd["val_loss"]))]
    choices = (1000, 2500, 4000, 5000)
    nearest = min(choices, key=lambda c: abs(c - best))
    if abs(nearest - best) > 500:
        raise ValueError(f"B3.21 c): the best iteration {best} is far from the choices {choices}")
    early = min(nd["val_loss"]) < dr["val_loss"][-1]
    if abs(min(nd["val_loss"]) - dr["val_loss"][-1]) < 0.015:
        raise ValueError("B3.21 d): the two validation losses are too close to be read on the graph")
    gap = abs(float(np.mean(dr["train_loss"][-250:])) - dr["val_loss"][-1])
    gap_nd = abs(float(np.mean(nd["train_loss"][-250:])) - nd["val_loss"][-1])
    if not (gap < 0.05 and gap_nd > 0.3):
        raise ValueError(f"B3.21 e): the gaps {gap:.3f} and {gap_nd:.3f} do not separate the two runs")
    return {"best": nearest, "early": bool(early), "close": True}


A21 = _answers_21()
GIVEN_21 = _fill(C.GIVEN_21, {"FULL_21": _full_21()})
SOLUTION_21 = (f'vocab_21, fast_21, best_21, early_21, close_21 = 88, "A", {A21["best"]}, {A21["early"]}, {A21["close"]}')
RECORD_21 = (
    'wb.record("B3.21a", 88, mistakes={"c\'est le vocabulaire du quiz Q10 : combien de caractères distincts a Holmes '
    '(la cellule des outils de la partie A le dit) ?": 65, "c\'est le nombre d\'ids de B3.17, padding compris : combien '
    'de caractères distincts le modèle prédit-il ?": 89, "c\'est le vocabulaire de GPT-2 : ici, le modèle prédit des '
    'caractères": 50257, "la loss de départ n\'est qu\'à peu près log V : combien de caractères distincts le modèle '
    'prédit-il ?": 90, "on demande V, pas la loss": 4}, fractional="on demande un nombre de caractères, un entier")\n'
    'wb.record("B3.21b", "A", mistakes=' + _letters("A", {
        "B": "regarde la courbe de validation du graphique 1 sur ses dernières itérations : remonte-t-elle ?",
        "C": "regarde la courbe d'entraînement du graphique 1 : saute-t-elle, ou diverge-t-elle ?"}) + ')\n'
    f'wb.record("B3.21c", {A21["best"]}, mistakes={{'
    + ", ".join(f'"{msg}": {v}' for v, msg in {
        1000: "graphique 3 : la courbe rouge (sans dropout) est-elle à son plus bas vers 1 000 itérations ?",
        2500: "graphique 3 : la courbe rouge (sans dropout) est-elle à son plus bas vers 2 500 itérations ?",
        4000: "graphique 3 : où la courbe rouge (sans dropout) atteint-elle son plus bas ?",
        5000: "graphique 3 : la courbe rouge (sans dropout) finit-elle à son plus bas ?"}.items() if v != A21["best"])
    + '}, fractional="choisis l\'une des quatre itérations proposées", choices=[1000, 2500, 4000, 5000])\n'
    f'wb.record("B3.21d", {A21["early"]}, mistakes={{"graphique 3 : compare le point le plus bas de la courbe rouge à la '
    f'fin de la courbe verte": {not A21["early"]}}})\n'
    f'wb.record("B3.21e", {A21["close"]}, mistakes={{"graphique 2 : à 5 000 itérations, mesure l\'écart entre les deux '
    f'courbes vertes (entraînement et validation avec dropout)": {not A21["close"]}}})')

# --- 🔮 B3.22 ------------------------------------------------------------------------------------------------------
P22 = HISTORY.get("p22", {}).get("answers", {"french": False, "beats_uniform": True})
SOLUTION_22 = f'french_22, uniform_22 = {P22["french"]}, {P22["beats_uniform"]}'
RECORD_22 = (f'wb.record("B3.22a", {P22["french"]}, mistakes={{"compte les continuations où les mots français '
             f'l\'emportent, dans la ligne affichée pour le prompt français": {not P22["french"]}}})\n'
             f'wb.record("B3.22b", {P22["beats_uniform"]}, mistakes={{"compare la loss sur Verne à celle d\'une '
             f'prédiction uniforme, affichées l\'une à côté de l\'autre": {not P22["beats_uniform"]}}})')


# --- 📈 B3.24: the heads of the model of the solution (the fast model of seed 2320, read on its maps) ---------------
def _heads_24() -> str:
    scores = HISTORY.get("p24", {}).get("fast 2320")
    if not scores or "spaces" not in scores:
        return "previous_24, spaces_24, spread_24 = (0, 0), (0, 1), (0, 2)"
    picks = []
    for key in ("previous", "spaces", "spread"):
        table = np.array(scores[key])
        picks.append(tuple(int(i) for i in np.unravel_index(table.argmax(), table.shape)))
    return (f"previous_24, spaces_24, spread_24 = {picks[0]}, {picks[1]}, {picks[2]}   # read on the maps of the model "
            "of the solution")


SOLUTION_24 = _heads_24()


# --- 🔬 B3.25: the full protocol and its answers -------------------------------------------------------------------
def _ablations():
    abl = HISTORY.get("ablations", {})
    full = abl.get("full", {}).get("losses")
    if not full:
        return ({n: [1.6, 1.6, 1.6] for n in ("reference", "no_position", "no_residual", "post_ln", "one_head")},
                {"stuck": "LR", "mildest": "H", "clear": 4})
    ref = np.array(full["reference"])
    gaps = {n: np.array(full[n]) - ref for n in LETTERS_25}
    means = {n: g.mean() for n, g in gaps.items()}
    stds = {n: g.std() for n, g in gaps.items()}
    losses = {n: np.mean(full[n]) for n in LETTERS_25}
    if any(abs(v - 3.0) < 0.1 for v in losses.values()):
        raise ValueError(f"B3.25 a): a mean loss is too close to 3 nats: {losses}")
    order = sorted(means, key=means.get)
    if means[order[1]] - means[order[0]] < 2 * max(stds[order[1]], stds[order[0]]):
        raise ValueError(f"B3.25 b): the two mildest ablations are not separated: {means}, {stds}")
    for n in LETTERS_25:
        if abs(means[n] - 2 * stds[n]) < 0.01:
            raise ValueError(f"B3.25 c): the gap of {n} is too close to twice its dispersion")
    clear = sum(means[n] > 2 * stds[n] for n in LETTERS_25)
    stuck = "".join(sorted(LETTERS_25[n] for n in LETTERS_25 if losses[n] > 3.0))
    return full, {"stuck": stuck, "mildest": LETTERS_25[order[0]], "clear": int(clear)}


FULL_25, A25 = _ablations()
GIVEN_25 = _fill(C.GIVEN_25, {"PROTOCOL_FAST_25": repr(C.PROTOCOLS["25"]["fast"]),
                              "PROTOCOL_FULL_25": repr(C.PROTOCOLS["25"]["full"]),
                              "FULL_25": repr({n: [round(v, 4) for v in vs] for n, vs in FULL_25.items()})})
_HINTS_25 = {"P": "relis la colonne « protocole complet » : est-ce bien sans position que l'écart est",
             "R": "relis la colonne « protocole complet » : est-ce bien sans résiduel que l'écart est",
             "L": "relis la colonne « protocole complet » : est-ce bien en post-LN que l'écart est",
             "H": "relis la colonne « protocole complet » : est-ce bien avec une seule tête que l'écart est"}
SOLUTION_25B = f'stuck_25, mildest_25, clear_25 = "{A25["stuck"]}", "{A25["mildest"]}", {A25["clear"]}'


def _stuck_mistakes(answer: str) -> str:
    """One message for each set of letters that misses one ablation or has one too many (no letter named)."""
    entries = {}
    for k, letter in enumerate(sorted(set(answer))):
        entries[f"il en manque {'une' if k == 0 else 'au moins une'} : compare chaque loss moyenne du protocole complet "
                f"à 3 nats{' (relis toutes les lignes)' if k else ''}"] = answer.replace(letter, "")
    for k, letter in enumerate(sorted(set("HLPR") - set(answer))):
        entries[f"une de trop{' encore' if k else ''} : une ablation ne compte que si sa loss moyenne du protocole complet "
                f"dépasse 3 nats{' (relis chaque ligne)' if k else ''}"] = "".join(sorted(answer + letter))
    return "{" + ", ".join(f"{json.dumps(m, ensure_ascii=False)}: {json.dumps(v)}" for m, v in entries.items() if v) + "}"


RECORD_25 = (
    f'wb.record("B3.25a", "{A25["stuck"]}", mistakes=' + _stuck_mistakes(A25["stuck"]) + ')\n'
    f'wb.record("B3.25b", "{A25["mildest"]}", mistakes='
    + _letters(A25["mildest"], {k: v + " le plus petit ?" for k, v in _HINTS_25.items()}) + ')\n'
    f'wb.record("B3.25c", {A25["clear"]}, mistakes={{'
    + ", ".join(f'"compare, ligne par ligne, l\'écart moyen du protocole complet au double de sa dispersion (tu en '
                f'comptes {k})": {k}' for k in range(5) if k != A25["clear"])
    + '}, fractional="on demande un nombre d\'ablations, un entier", choices=[0, 1, 2, 3, 4])')


# --- 🔬 B3.26 ------------------------------------------------------------------------------------------------------
def _bpe():
    b = HISTORY.get("bpe26", {})
    full = b.get("full", {}).get("losses")
    if not full:
        return {"caractères": [1.6, 1.6, 1.6], "BPE 512": [1.7, 1.7, 1.7]}, "C", 2.118
    c, p = np.array(full["caractères"]), np.array(full["BPE 512"])
    spread = 2 * max(c.std(), p.std())
    if abs(abs(c.mean() - p.mean()) - spread) < 0.01:
        raise ValueError("B3.26 b): the gap is too close to the dispersion")
    letter = "E" if abs(c.mean() - p.mean()) <= spread else ("C" if c.mean() < p.mean() else "B")
    return full, letter, b["chars_per_token"]["val"]


FULL_26, A26, CPT_VAL_26 = _bpe()
GIVEN_26 = _fill(C.GIVEN_26, {"PROTOCOL_FAST_26": repr(C.PROTOCOLS["26"]["fast"]),
                              "PROTOCOL_FULL_26": repr(C.PROTOCOLS["26"]["full"]),
                              "FULL_26": repr({n: [round(v, 4) for v in vs] for n, vs in FULL_26.items()})})
SOLUTION_26B = f'better_26 = "{A26}"'
RECORD_26 = (f'wb.record("B3.26a", {CPT_VAL_26!r}, decimals=4)\n'
             f'wb.record("B3.26b", "{A26}", mistakes=' + _letters(A26, {
                 "C": "relis les deux lignes du protocole complet : le modèle de caractères est-il le plus bas, au-delà de "
                      "la dispersion ?",
                 "B": "relis les deux lignes du protocole complet : le modèle BPE est-il le plus bas, au-delà de la "
                      "dispersion ?",
                 "E": "relis les deux lignes du protocole complet : l'écart des moyennes est-il plus petit que la "
                      "dispersion des graines ?"}) + ')')

# --- 📦 B3.28 ------------------------------------------------------------------------------------------------------
V28 = HISTORY.get("vit28", {})
GIVEN_28 = _fill(C.GIVEN_28, {"EPOCHS_FULL_28": "12", "TARGET_28": str(V28.get("target_fast", 0.8)),
                              "TARGET_FULL_28": str(V28.get("target_full", 0.85))})

# --- 🏆 B3.30 ------------------------------------------------------------------------------------------------------
CH30 = HISTORY.get("challenge30", {})


def _goal(mode: str, default: float) -> str:
    return repr(CH30.get(mode, {}).get("goal", default))


GIVEN_30 = _fill(C.GIVEN_30, {
    "GOAL_FAST_30": _goal("fast", 1.75), "GOAL_FULL_30": _goal("full", 1.45),
    "REF_FAST_30": repr(tuple(round(v, 4) for v in CH30.get("fast", {}).get("reference", (1.66, 1.68)))),
    "REF_FULL_30": repr(tuple(round(v, 4) for v in CH30.get("full", {}).get("reference", (1.37, 1.38)))),
    "CONTROL_GAP_FAST_30": repr(tuple(round(v, 3) for v in CH30.get("fast", {}).get("control_gap", (0.0, 0.1)))),
    "CONTROL_GAP_FULL_30": repr(tuple(round(v, 3) for v in CH30.get("full", {}).get("control_gap", (0.0, 0.1))))})


def _fr(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


GOAL_TEXT = {mode: _fr(CH30.get(mode, {}).get("goal", d)) for mode, d in (("fast", 1.75), ("full", 1.45))}
REF_TEXT = {mode: tuple(_fr(v, 3) for v in CH30.get(mode, {}).get("reference", d))
            for mode, d in (("fast", (1.66, 1.68)), ("full", (1.37, 1.38)))}

# ---------------------------------------------------------------------------------------------------------------------
PART_C = Part("C", "Ton mini-GPT entraîné : courbes, génération, KV cache, cartes d'attention",
              "La partie C ne réentraîne rien : elle recharge le mini-GPT que B3.20 a sauvegardé. Tu lis ses courbes "
              "d'entraînement (B3.21), tu prédis ce qu'il fait d'un prompt en français (B3.22), tu le fais générer vite "
              "avec un KV cache (B3.23) et tu lis les cartes d'attention de ses têtes (B3.24).\n\n"
              "La cellule suivante recharge le modèle : `GPT` (ton `MiniGPT_19`, en mode eval), `CKPT` (sa sauvegarde ; "
              "`CKPT[\"history\"]` est l'historique de l'entraînement) et `N_HEAD` (4). Si la sauvegarde manque (une "
              "session Colab qui se termine l'efface), relance les cellules de B3.19 et B3.20, ou mets "
              "`REFERENCE_FALLBACK = True` dans la cellule : le mini-GPT de référence de `wb.gpt`, entraîné une fois "
              "avec la recette de B3.20, remplace alors le tien.",
              given=C.TOOLS_C, exercises=[
    Ex("B3.21", "📈", 2, 15, "Lire les courbes d'entraînement du mini-GPT et poser un diagnostic",
       "lire les courbes d'entraînement de trois mini-GPT et dire ce qu'elles montrent : le départ, l'apprentissage, "
       "l'overfitting et l'effet du dropout.",
       "Ex B3.20 · fiche §203.8 (« La loss et la perplexité », « La recette d'entraînement ») · ch. 20 (courbes, "
       "overfitting, dropout)", thread="Holmes", tracks="C",
       body=r"""La cellule suivante trace trois graphiques :
1. **ton entraînement** de B3.20, relu dans sa sauvegarde : la loss d'entraînement (moyennée par blocs de 10 itérations), la loss de validation estimée toutes les 50 itérations et le learning rate (en pointillés, axe de droite) ;
2. deux entraînements du mini-GPT de référence **en mode complet** (5 000 itérations, environ 40 passages sur le texte d'entraînement), précalculés : l'un **sans dropout** (la configuration du mode rapide), l'autre avec un **dropout de 0,2** (celle de `CONFIG_20` en mode complet) ; la loss d'entraînement est moyennée par blocs de 50 itérations, la validation estimée toutes les 250 ;
3. leurs deux courbes de validation à partir de 1 000 itérations, sur une échelle resserrée.

Les loss sont en nats par caractère. Écris tes cinq lectures :
- a) `vocab_21` : à l'itération 0, la loss de validation est proche de $\log V$ (le contrôle de départ, fiche §203.8) : que vaut $V$ ? Un entier.
- b) `fast_21` : ton entraînement (graphique 1) : `"A"` la validation baisse encore à la fin, le modèle pourrait apprendre davantage ; `"B"` overfitting, la validation remonte ; `"C"` instable, la loss saute ou diverge.
- c) `best_21` : sans dropout, à quelle itération la loss de validation est-elle la plus basse : 1 000, 2 500, 4 000 ou 5 000 ?
- d) `early_21` : `True` ou `False` : arrêté à sa meilleure itération (early stopping, ch. 20), l'entraînement **sans** dropout aurait-il une loss de validation plus basse que celui **avec** dropout à 5 000 itérations ?
- e) `close_21` : `True` ou `False` : avec dropout, les courbes d'entraînement et de validation finissent-elles à moins de 0,1 nat l'une de l'autre ?

Dans tes notes : ton diagnostic de chacun des trois entraînements, en une phrase. Que change le dropout à la courbe d'entraînement, et pourquoi ? Où vois-tu le warmup et la décroissance du learning rate sur le graphique 1, et que font-ils à la loss ? Sans dropout, que ferais-tu de l'entraînement complet si tu ne pouvais pas en changer la configuration ?""",
       given=GIVEN_21, todo=C.TODO_21, check=C.CHECK_21, solution=SOLUTION_21, record=RECORD_21),

    Ex("B3.22", "🔮", 1, 10, "Prédire : le mini-GPT de Holmes amorcé avec une phrase de Verne",
       "prédire ce que fait un modèle de langue entraîné sur de l'anglais quand on lui donne un prompt en français.",
       "Ex B3.20 · fiche §203.8 (« La loss et la perplexité »), §203.9 (« La boucle de génération ») · ch. 22 (Holmes et "
       "Verne)", thread="Holmes/Verne", tracks="C", hypothesis=True,
       body=r"""Ton mini-GPT n'a lu que Holmes, en anglais. La cellule suivante prépare `PROMPT_FR_22`, une phrase de Verne (*Le Tour du monde en quatre-vingts jours*, ch. 22), et `FRENCH_22`, toute la validation de Verne, écrites avec les caractères de Holmes (`to_holmes_22` : l'apostrophe et les guillemets deviennent ceux de Holmes, un accent que Holmes n'emploie jamais disparaît, ê → e, ç → c). `languages_22(texte)` compte les mots anglais (ceux que seul le texte d'entraînement de Holmes contient) et les mots français (ceux que seul celui de Verne contient) d'un texte.

L'expérience tire 10 continuations de 120 caractères après le prompt français (température 1, `wbgpt.generate`), et 10 après un prompt anglais de même style, pour comparer ; puis elle mesure la loss par caractère de ton modèle sur toute la validation de Verne. **Avant d'exécuter**, écris tes deux prédictions :
- a) `french_22` : `True` ou `False` : sur les 10 continuations du prompt français, **plus de la moitié** ont-elles plus de mots français que de mots anglais ?
- b) `uniform_22` : `True` ou `False` : sur le texte français de Verne, la loss par caractère de ton modèle est-elle **plus basse** que celle d'une prédiction uniforme sur les 88 caractères ($\log 88$ nats par caractère) ?

La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites ; elle les vérifie à la fin.""",
       given=C.GIVEN_22, todo=C.TODO_22, solution=SOLUTION_22, record=RECORD_22,
       after=[("code", guarded(C.EXPERIMENT_22, ["french_22", "uniform_22"],
                               "⏳ Ex B3.22 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Que reste-t-il de "
                     "l'anglais dans ce que le modèle sait encore prédire du français (lettres, espaces, ponctuation, "
                     "mots communs) ? Compare, dans le tableau des positions, la loss du début et de la fin d'une "
                     "fenêtre sur Holmes, puis sur Verne : que te dit cet écart sur ce que le modèle fait de son "
                     "contexte ? Que faudrait-il pour qu'il continue en français ?")]),

    Ex("B3.23", "📦", 3, 45, "Générer vite : le KV cache, et le gain de vitesse mesuré selon la longueur",
       "écrire la génération gloutonne du mini-GPT sans cache puis avec un KV cache, vérifier qu'elles donnent les mêmes "
       "tokens, et mesurer le gain.",
       "Ex B3.20 · fiche §203.9 (« Pourquoi c'est lent sans cache », « Le KV cache ») · 🧠 Q11 · 🧮 B3.8",
       thread="Holmes", tracks="R, C",
       body=r"""**Contrat**
- `generate_23(model, idx, n_new)` : la génération **gloutonne** sans cache (fiche §203.9) : `idx` (N, T0) int64 → un nouveau tenseur (N, T0 + n_new), `idx` suivi de `n_new` tokens ; à chaque pas, les logits de la **dernière** position, calculés sur les `model.block_size` derniers tokens de la séquence, et le token le plus probable (l'argmax) ; sans gradient, le modèle en mode eval.
- `generate_cached_23(model, idx, n_new, n_head)` : **les mêmes tokens**, avec un KV cache :
  - un **prefill** : le prompt passe une fois, en entier, dans les blocs, et remplit pour chaque bloc les clés et les valeurs de ses tokens, de forme (N, n_head, T0, d_h) ;
  - puis un token par pas : **seul** le nouveau token passe dans le modèle, avec l'embedding de **sa** position ; sa clé et sa valeur s'ajoutent au cache de chaque bloc, et sa requête regarde tout le cache ;
  - écrit avec les sous-modules du modèle, aux noms de B3.19 : `tok_emb`, `pos_emb`, puis pour chaque bloc `ln1`, `attn.qkv`, `attn.proj`, `ln2`, `mlp`, enfin `ln_f` et `head` ; jamais `model(...)`, un bloc ou un module d'attention appelé en entier : ils recalculeraient toute la séquence ;
  - `n_head` donne le nombre de têtes (ton module n'a pas à l'exposer) ; `ValueError` si `T0 + n_new > model.block_size` : au-delà, il n'y a plus de position apprise.

La cellule de vérification essaie tes deux fonctions sur un petit mini-GPT de `wb.gpt` (2 blocs, 2 têtes, un contexte de 32 tokens) : `generate_23` contre `wbgpt.generate`, la génération sans cache de référence (40 tokens après 5 : le contexte doit être coupé), puis `generate_cached_23` jusqu'au contexte plein, avec des *hooks* de PyTorch (`register_forward_hook`) qui comptent les tokens que voit chaque `attn.qkv` et vérifient que ni le modèle, ni un bloc, ni un module d'attention ne sont appelés ; enfin un refus. La cellule suivante mesure, sur **ton** mini-GPT et 4 prompts de 32 caractères, le temps des deux générations pour 16 à 96 nouveaux tokens (le meilleur de trois essais), vérifie que les tokens sont les mêmes et trace les temps. Un temps est une mesure, pas une réponse : il dépend de la machine.

Dans tes notes : le gain grandit-il avec le nombre de tokens générés, et pourquoi reste-t-il modeste ici (pense au contexte de 128 et à ce que coûte un appel de module sur CPU) ? Combien de mémoire ton cache occupe-t-il à 128 tokens pour les 4 prompts (🧮 B3.8) ? Pourquoi le token **tiré au dernier pas** n'a-t-il pas besoin de passer dans le modèle ? Avec un tirage aléatoire au lieu de l'argmax, que faudrait-il pour que les deux générations restent identiques ?""",
       todo=C.TODO_23, check=C.CHECK_23, solution=C.SOLUTION_23 + "\n\n\n" + C.CHECK_23,
       after=[("code", C.EXPERIMENT_23)]),

    Ex("B3.24", "📈", 2, 20, "Les cartes d'attention de chaque tête sur une phrase de Holmes",
       "lire les cartes d'attention des 16 têtes du mini-GPT et vérifier sur leurs poids ce que l'on croit y voir.",
       "Ex B3.20 · fiche §203.3 (« Le masque causal »), §203.4 (« Plusieurs questions à la fois ») · Ex B3.17 (une carte "
       "d'attention)", thread="Holmes", tracks="C",
       body=r"""La cellule suivante trace les cartes d'attention des 16 têtes de ton mini-GPT (4 blocs × 4 têtes) sur une phrase de Holmes : une ligne par requête, une colonne par clé, du violet (poids 0) au jaune (poids 1), les caractères sur les axes (␣ pour un espace). Le module d'attention ne rend pas ses poids (il appelle SDPA) : `attention_maps_24(model, ids)` les recalcule à partir de l'entrée de chaque attention, de `attn.qkv`, de l'échelle $1/\sqrt{d_h}$ et du masque causal.

Écris, pour chaque affirmation, la tête qui la vérifie le mieux, un couple `(bloc, tête)` d'entiers comptés à partir de 0 :
- a) `previous_24` : la tête qui regarde le plus le **caractère précédent** ;
- b) `spaces_24` : la tête qui regarde le plus les **espaces** (des colonnes claires sous chaque espace) ;
- c) `spread_24` : la tête dont l'attention est la plus **étalée** sur les clés (la moins concentrée).

La cellule de vérification mesure chaque affirmation sur les poids de 16 fenêtres de 128 caractères de la validation : a) le poids moyen de la clé $i - 1$ pour la requête $i$ ; b) le poids total des clés qui sont des espaces, sur les requêtes 32 à 127 ; c) l'entropie moyenne d'une ligne de poids, $-\sum_j \alpha_{ij} \ln \alpha_{ij}$, sur les requêtes 32 à 127. Ta tête est acceptée si sa mesure atteint au moins 90 % de la plus forte des 16. Ton modèle n'est pas celui d'un autre : les têtes qui portent ces motifs changent d'un entraînement à l'autre, c'est pourquoi la cellule les mesure sur tes poids au lieu de comparer à une réponse.

Dans tes notes : décris deux autres motifs que tu vois (deux caractères en arrière, la ponctuation, les majuscules…) et vérifie-les sur les poids, avec un calcul du même genre. Les têtes du bloc 0 et celles du bloc 3 se ressemblent-elles ? Pourquoi une tête « caractère précédent » est-elle utile à un modèle de caractères, et pourquoi aucune carte n'a-t-elle de poids au-dessus de la diagonale ?""",
       given=C.GIVEN_24, todo=C.TODO_24, check=C.CHECK_24, solution=SOLUTION_24 + "\n\n\n" + C.CHECK_24),
])

PART_D = Part("D", "Expériences : ablations, tokens, contexte long",
              "Trois expériences : ce que coûte chaque pièce du bloc (B3.25), des tokens BPE au lieu de caractères "
              "(B3.26), et le prix d'un long contexte pour l'attention (B3.27). Les deux premières entraînent plusieurs "
              "modèles : en mode rapide, un petit modèle (2 blocs, $d = 64$, des contextes de 64 caractères, 200 "
              "itérations, environ une minute par exercice), à côté des résultats du protocole complet (le mini-GPT de "
              "B3.20, 600 itérations, trois graines), précalculés avec la solution par "
              "`tools/chapters/history_chB3.py` ; le mode complet (🚀) refait ce protocole avec ton code, environ 35 "
              "minutes sur CPU pour B3.25. Elles entraînent avec ton `train_20` (B3.20) s'il a passé ses contrôles "
              "dans la session, sinon avec la recette de `wb.gpt` (une ligne ℹ️ le dit).",
              exercises=[
    Ex("B3.25", "🔬", 3, 60, "Ablations : sans position, sans résiduel, post-LN, une seule tête",
       "mesurer ce que coûte chacune de quatre ablations du mini-GPT, avec sa dispersion d'une graine à l'autre, et les "
       "classer.",
       "Ex B3.19, B3.20 · fiche §203.5 (« L'attention ne voit pas l'ordre »), §203.6 (« Deux sous-couches », « Pré-LN ou "
       "post-LN »), §203.4 · ch. 20 (connexions résiduelles)", thread="Holmes", tracks="C",
       body=r"""La fiche dit déjà dans quel sens vont ces ablations : sans encodage de position, un modèle causal retrouve une partie de l'ordre (§203.5) ; le post-LN réclame un warmup soigné (§203.6) ; les connexions résiduelles rendent un réseau profond entraînable (ch. 20). La question est **de combien** : l'écart de loss de validation de chaque ablation, en nats par caractère, comparé à sa dispersion d'une graine à l'autre.

**Contrat** de `make_variant_25(name, config)`, qui rend un **nouveau** modèle :
- `name` dans `VARIANTS_25` : `"reference"`, le mini-GPT de B3.19 ; `"no_position"`, sans positions apprises (pas de `pos_emb` du tout : son `state_dict` n'a pas de `pos_emb.weight`) ; `"no_residual"`, chaque sous-couche remplace son entrée au lieu de s'y ajouter : $\mathbf{h} = \operatorname{attn}(\operatorname{LN}_1(\mathbf{x}))$, puis $\mathbf{y} = \operatorname{mlp}(\operatorname{LN}_2(\mathbf{h}))$ ; `"post_ln"`, le bloc de 2017 : $\mathbf{h} = \operatorname{LN}_1(\mathbf{x} + \operatorname{attn}(\mathbf{x}))$, puis $\mathbf{y} = \operatorname{LN}_2(\mathbf{h} + \operatorname{mlp}(\mathbf{h}))$ ; `"one_head"`, une attention à **une** tête de `d_model` features au lieu de `config["n_head"]` ; `ValueError` pour un autre nom ;
- `config` : un dictionnaire comme `MINI_CONFIG` (`vocab_size`, `block_size`, `n_layer`, `n_head`, `d_model`, `dropout`, ici 0) ;
- pour le reste, le modèle de B3.19 : les mêmes noms (`tok_emb`, `pos_emb`, `blocks[i]` avec `ln1`, `attn` et ses `qkv` et `proj`, `ln2`, `mlp`, puis `ln_f` et `head`), la tête liée, l'initialisation de nanoGPT, `forward(idx)` → logits `(N, T, vocab_size)`, la causalité. Tu peux réutiliser tes classes `Block_19`, `CausalSelfAttention_18` et `MiniGPT_19`.

La première cellule de vérification construit les cinq variantes sur un petit modèle (2 blocs, $d = 16$) et contrôle pour chacune les noms et les formes du `state_dict`, le calcul d'un bloc à partir de ses propres sous-modules, le nombre de têtes de l'attention (recalculée à partir de `qkv`), l'usage des positions (un même caractère répété), la causalité, la tête liée et l'initialisation. La cellule suivante entraîne les cinq variantes avec les réglages de `PROTOCOL_25` et chaque graine, mesure leur loss sur la validation (`wbgpt.char_loss`), puis affiche pour chacune la loss moyenne, son écart à la référence graine par graine (moyenne ± écart-type) et les mêmes chiffres du protocole complet (`FULL_25`), avec un graphique des écarts. Puis réponds, d'après le **protocole complet** :
- a) `stuck_25` : les lettres des ablations dont la loss reste au-dessus de 3 nats par caractère, à peu près celle d'un modèle qui n'aurait appris que la fréquence de chaque caractère : `"P"` sans position, `"R"` sans résiduel, `"L"` post-LN, `"H"` une seule tête (par exemple `"PH"`) ;
- b) `mildest_25` : l'ablation qui perd le moins (mêmes lettres) ;
- c) `clear_25` : combien des quatre ablations ont un écart moyen supérieur au double de sa dispersion (l'écart-type de l'écart sur les trois graines) ? Un entier de 0 à 4.

Dans tes notes : ton classement, avec les écarts et leurs dispersions. Ton protocole rapide donne-t-il le même classement que le protocole complet, et qu'est-ce qui peut les séparer (taille du modèle, nombre d'itérations, nombre de graines) ? Pourquoi le modèle sans position ne s'effondre-t-il pas ? Que ferais-tu pour savoir si le post-LN souffre du warmup trop court de la recette (50 itérations) ? Pourquoi trois graines suffisent-elles à peine à conclure ?""",
       given=GIVEN_25, todo=C.TODO_25, check=C.CHECK_25, solution=C.SOLUTION_25 + "\n\n\n" + C.CHECK_25,
       after=[("code", C.EXPERIMENT_25), ("todo", C.TODO_25B), ("check", C.CHECK_25B), ("solution", SOLUTION_25B),
              ("record", RECORD_25)]),

    Ex("B3.26", "🔬", 3, 45, "Tokens BPE du B2 contre caractères : perplexité par caractère à budget égal",
       "comparer, à budget égal et en nats par caractère, un mini-GPT de caractères et un mini-GPT de tokens BPE.",
       "Ex B3.20 · B2 (BPE au niveau octet, B2.16, B2.17, B2.20) · fiche §203.8 (« La loss et la perplexité ») · 🧠 R1",
       thread="Holmes", tracks="M, C",
       body=r"""Une loss par token ne se compare pas d'un tokenizer à l'autre : un token BPE porte plusieurs caractères. La fiche (§203.8) ramène tout au **caractère** : la loss par caractère vaut la loss par token multipliée par le nombre de tokens et divisée par le nombre de caractères du même texte.

La cellule suivante prépare `TOK_26` : **ton** `BPETokenizer` de B2, au niveau octet, avec le pré-tokenizer `'gpt2'` et 512 tokens, appris sur le texte d'entraînement de Holmes (ses fusions viennent de la bibliothèque `tokenizers`, qui suit la règle de ton `fit`, comme en B2.20 ; ton `encode` fait le reste ; si ta classe n'est pas écrite, ou si ton `encode` ne donne pas les ids de la référence pour les mêmes fusions, la référence la remplace et une ligne le dit).

**Budget égal**, ici : **le même texte vu**. Les deux modèles ont les mêmes blocs, les mêmes têtes et le même $d$, le même nombre d'itérations et de séquences par batch, et un contexte qui couvre le même nombre de caractères : `char_block` caractères (128 dans le protocole complet, 64 en mode rapide) pour le modèle de caractères, `budget_block_26(caractères par token, char_block)` tokens pour le modèle BPE. Ils voient donc à peu près les mêmes caractères d'entraînement, mais ils ne font pas le même calcul : le modèle BPE a moins de positions par séquence et une table de 512 tokens au lieu de 88.

**Contrat**
- `chars_per_token_26(tok, text)` : le nombre moyen de caractères par token de `text` encodé par `tok`, un float ;
- `budget_block_26(chars_per_token, char_block)` : l'entier le plus proche de `char_block / chars_per_token` ;
- `nats_per_char_26(model, ids, tok, block_size)` : la loss d'un modèle BPE en nats par **caractère** sur les ids `ids` (un tenseur 1-D) : `wbgpt.char_loss(model, ids, block_size=block_size, stride=block_size // 4)` rend une moyenne **par token** sur les cibles `ids[block_size:]` ; convertis-la avec le nombre de ces tokens et le nombre de caractères qu'ils écrivent (`tok.decode`).

La première cellule de vérification essaie tes trois fonctions (dont `nats_per_char_26` sur un modèle uniforme, dont on connaît la loss par token : $\log 512$), puis vérifie le nombre de caractères par token de `TOK_26` sur toute la validation de Holmes (B3.26a, rien à taper). La cellule suivante entraîne, pour chaque graine de `PROTOCOL_26`, un mini-GPT de caractères et un mini-GPT BPE (ta classe `MiniGPT_19` si elle a passé ses contrôles), et affiche pour chacun le contexte, le nombre de paramètres, le temps d'entraînement et la loss par caractère sur la validation, avec les mêmes chiffres du protocole complet (`FULL_26`). Puis réponds :
- b) `better_26` : dans le protocole complet, lequel a la loss par caractère la plus basse : `"C"` les caractères, `"B"` le BPE, ou `"E"` à égalité (l'écart des moyennes est plus petit que le double de la plus grande des deux dispersions) ?

Dans tes notes : pourquoi la loss par token du modèle BPE ne se compare-t-elle à rien ? Combien de caractères chaque modèle a-t-il vus, et combien de positions a-t-il calculées par itération ? Le résultat changerait-il à **calcul** égal (le même nombre de positions par batch), ou avec 2 000 tokens (B2.20) ? Pourquoi les grands modèles de langue prennent-ils des tokens BPE alors qu'un modèle de caractères n'a ni token inconnu ni table énorme ?""",
       given=GIVEN_26, todo=C.TODO_26, check=C.CHECK_26, solution=C.SOLUTION_26 + "\n\n\n" + C.CHECK_26,
       after=[("code", C.EXPERIMENT_26), ("todo", C.TODO_26B), ("check", C.CHECK_26B), ("solution", SOLUTION_26B),
              ("record", RECORD_26)]),

    Ex("B3.27", "🔬", 2, 25, "Temps et mémoire de l'attention selon la longueur du contexte (64 à 2 048)",
       "mesurer le temps et la mémoire d'une attention de 64 à 2 048 tokens avec les deux noyaux de SDPA sur CPU, et "
       "relier les mesures à la formule $T^2$.",
       "Ex B3.18 · fiche §203.10 (« La mémoire : la matrice $T \\times T$ », « FlashAttention ») · 🧠 Q12",
       tracks="C",
       body=r"""`F.scaled_dot_product_attention` choisit un **noyau** (*kernel*) selon le matériel et les arguments. Sur CPU, torch 2.11 en a deux : `SDPBackend.MATH`, qui écrit la matrice des poids $T \times T$ en entier, et `SDPBackend.FLASH_ATTENTION`, un noyau de type FlashAttention écrit pour CPU (fiche §203.10), qui parcourt les clés par blocs ; `torch.nn.attention.sdpa_kernel(backend)` impose l'un ou l'autre dans un bloc `with`.

**Contrat**
- `attention_time_27(T, backend, n_head=4, d_head=32, repeats=5)` : des requêtes, clés et valeurs aléatoires de forme `(1, n_head, T, d_head)` en float32, puis, dans `with sdpa_kernel(backend):`, un premier appel causal de `F.scaled_dot_product_attention` non mesuré, puis `repeats` appels mesurés (`time.perf_counter`) ; rend **le meilleur** des `repeats` temps, en millisecondes (un float) ;
- `weights_bytes_27(T, n_head=4, bytes_per_value=4)` : la mémoire, en octets (un int), des poids d'attention `(n_head, T, T)` d'une séquence.

La cellule de vérification essaie ta formule, puis ta mesure avec une sonde : les deux premiers appels de SDPA y sont ralentis de 50 ms, et seul le meilleur de plusieurs essais les ignore. La cellule suivante affiche le noyau que PyTorch choisit par défaut sur ce CPU, essaie le troisième noyau, `EFFICIENT_ATTENTION`, puis mesure pour $T$ = 64, 128, …, 2 048 et chaque noyau le temps (ta fonction) et la plus grosse allocation d'un opérateur (`profiled_bytes_27`, le profileur de PyTorch : `tracemalloc` ne voit pas la mémoire de PyTorch), à côté de ta formule ; elle calcule la pente du temps en échelles logarithmiques au-delà de 512 tokens et trace les deux graphiques. Les temps dépendent de la machine : donne-les en fourchettes.

Dans tes notes : le noyau choisi par défaut, et ce que dit l'essai d'`EFFICIENT_ATTENTION`. Tes temps à 2 048 tokens pour les deux noyaux, en fourchette, et leur rapport. La pente du noyau MATH vaut-elle 2, et pourquoi pas tout à fait ? Que mesure le profileur pour FLASH, et pourquoi si peu ? D'après ta formule, combien de mémoire faudrait-il au noyau MATH pour 32 têtes et 131 072 tokens (fiche §203.10) ? Le noyau FLASH change-t-il le nombre de FLOPs ?""",
       given=C.GIVEN_27, todo=C.TODO_27, check=C.CHECK_27, solution=C.SOLUTION_27 + "\n\n\n" + C.CHECK_27,
       after=[("code", C.EXPERIMENT_27)]),
])

PART_E = Part("E", "Un ViT, un test de causalité et le défi",
              "Le Transformer hors du texte : un Vision Transformer minuscule sur Fashion-MNIST (B3.28) ; une compétence "
              "pro : un test pytest qui garantit la causalité du mini-GPT (B3.29) ; et le défi du chapitre (B3.30).",
              exercises=[
    Ex("B3.28", "📦", 3, 45, "Un Vision Transformer minuscule : Fashion-MNIST découpé en patchs",
       "écrire un Vision Transformer minuscule, des patchs de $7 \\times 7$ à la classe, et l'entraîner sur Fashion-MNIST.",
       "Ex B3.19 · ch. 21 (convolutions, Fashion-MNIST) · fiche §203.10 (« Le Vision Transformer »), §203.7 (l'encodeur)",
       thread="Fashion-MNIST", tracks="C",
       body=r"""Une image de Fashion-MNIST ($28 \times 28$, un canal) se découpe en $4 \times 4 = 16$ patchs de $7 \times 7$ pixels : 16 tokens, plus un token `[CLS]` dont la sortie classe l'image (fiche §203.10). L'exemple de la fiche prend CIFAR-10 et des patchs de 4 pixels ; ici, des vêtements, et un **encodeur** : chaque token voit tous les autres.

**Contrat**
- `EncoderBlock_28(d_model, n_head, dropout=0.0)` : ton bloc pré-LN de B3.19, **sans masque causal** : $\mathbf{h} = \mathbf{x} + \operatorname{attn}(\operatorname{LN}_1(\mathbf{x}))$, puis $\mathbf{h} + \operatorname{mlp}(\operatorname{LN}_2(\mathbf{h}))$, l'attention multi-têtes par `F.scaled_dot_product_attention` (avec ses projections `qkv` et `proj`, comme `CausalSelfAttention_18`, et un dropout des poids en mode train seulement), le MLP GELU de largeur $4d$ ; `x` (N, T, d) → (N, T, d).
- `ViT_28(image_size=28, patch_size=7, n_classes=10, d_model=64, n_layer=4, n_head=4, dropout=0.0)` :
  - `patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)` : une couche dense par patch, écrite comme une convolution (fiche §203.10) ; ses 16 sorties, dans l'ordre de lecture (ligne par ligne, ce que donne `flatten(2)`), sont les tokens des patchs ;
  - `cls_token`, un `nn.Parameter` de forme `(1, 1, d_model)`, placé **en tête** de la séquence ; `pos_emb`, des positions apprises, un `nn.Parameter` de forme `(1, 1 + 16, d_model)` (initialisées petites, $\mathcal{N}(0;\ 0{,}02^2)$ par exemple), ajoutées aux 17 tokens ;
  - `blocks`, un `nn.ModuleList` de `n_layer` `EncoderBlock_28`, puis `ln_f` (`nn.LayerNorm`) et `head = nn.Linear(d_model, n_classes)` appliquée à la sortie du `[CLS]` ;
  - `forward(x)` : des images `(N, 1, 28, 28)` → des logits `(N, n_classes)` ; `ValueError` si `patch_size` ne divise pas `image_size` ;
  - ni `nn.Transformer…` ni `nn.MultiheadAttention`.

La cellule suivante charge Fashion-MNIST (normalisé comme aux ch. 20 et 21), et donne `train_vit_28` (AdamW, warmup puis cosinus, clipping) et `accuracy_28`. La cellule de vérification contrôle les formes des paramètres, celle des logits, que la sortie dépende du premier **et** du dernier patch (le `[CLS]` voit toute l'image), que les positions servent (échanger deux patchs change la sortie) et un refus. La suivante entraîne ton ViT : 12 000 images et 3 epochs en mode rapide (moins d'une minute) ; 🚀 60 000 images et 12 epochs en mode complet (environ 6 min sur CPU) ; puis elle mesure l'accuracy sur les 10 000 images de test, comparée au seuil `TARGET_28`.

Dans tes notes : ton nombre de paramètres et ton accuracy, face au CNN du ch. 21 (le défi y demandait 92 % avec moins de 100 000 paramètres). Pourquoi un ViT manque-t-il de biais inductifs (§203.10), et qu'est-ce que cela change sur 12 000 images ? Que donnerait ton ViT si le masque causal restait dans les blocs ? Le `[CLS]` est-il indispensable, ou la moyenne des 16 tokens ferait-elle l'affaire ?""",
       given=GIVEN_28, todo=C.TODO_28, check=C.CHECK_28, solution=C.SOLUTION_28 + "\n\n\n" + C.CHECK_28,
       after=[("code", C.EXPERIMENT_28)]),

    Ex("B3.29", "🛠️", 2, 20, "Un test pytest qui prouve la causalité : modifier le futur ne change pas les logits passés",
       "écrire le test pytest qui garantit qu'un modèle de langue causal ne voit pas le futur, et vérifier qu'il attrape "
       "des modèles qui trichent.",
       "Ex B3.19 · fiche §203.3 (« Le masque causal »), §203.8 (💼 En entreprise : les trois contrôles) · ✏️ B3.2 · ch. 0A "
       "(tests pytest)", thread="Holmes", tracks="C",
       body=r"""Un modèle causal qui voit le futur s'entraîne très bien : sa loss d'entraînement baisse vite et il ne sait rien générer (🐛 B3.17). En entreprise, l'invariant de la causalité se garde par un **test** qui tourne à chaque modification du code : changer le token d'une position $t$ ne change **aucun** logit des positions avant $t$.

Écris `test_causality_29()`, une fonction de test pytest :
- elle tourne dans un fichier à part, que `wb.run_pytest` écrit et lance dans un autre processus : elle ne lit **aucune** variable du notebook ; ce fichier lui fournit `MODEL` (un mini-GPT en mode eval, sans gradient), `VOCAB_SIZE`, `BLOCK_SIZE` et `torch` ;
- elle vérifie l'invariant par des `assert` dont le message dit ce qui était attendu ;
- elle doit **échouer** sur un modèle qui voit le futur, ne serait-ce qu'à une seule position, et **passer** sur le bon ; elle tourne en quelques secondes.

La cellule suivante prépare les modèles testés : ton mini-GPT entraîné (sa sauvegarde de B3.20, chargée dans la classe `ReferenceGPT` de `wb.gpt`, aux mêmes noms : une classe définie dans un notebook ne se copie pas dans un fichier de test), ou un mini-GPT non entraîné s'il n'y a pas de sauvegarde (la causalité ne dépend pas de l'entraînement), et trois copies dont l'attention voit le futur. La cellule de vérification lance ton test sur le bon modèle (il doit passer), puis sur les trois copies : un masque causal transposé, un masque qui laisse voir le token suivant, et un masque qui ne le laisse voir qu'à partir de la position 100. Pour chacune, ton test doit échouer : la cellule affiche alors le nom du test et la raison de l'échec, sur deux lignes.

Dans tes notes : pourquoi comparer les logits **exactement** est-il possible ici, et quand faudrait-il une tolérance ? Que coûte ton test, en passes du modèle ? Quel autre invariant du mini-GPT mériterait un test (la forme des logits, la loss de départ, les poids liés) ? Où placerais-tu ces tests dans un dépôt, et quand les lancerais-tu ?""",
       given=C.GIVEN_29, todo=C.TODO_29, check=C.CHECK_29, solution=C.SOLUTION_29 + "\n\n\n" + C.CHECK_29),

    Ex("B3.30", "🏆", 4, 120, "Défi : la meilleure perplexité caractère sur Holmes avec au plus 1 M de paramètres et 10 min "
                              "de CPU",
       "trouver, avec ce que le chapitre a montré, un modèle de caractères qui prédit la validation de Holmes nettement "
       "mieux que le mini-GPT de B3.20, en 10 minutes de CPU.",
       "Ex B3.20, B3.21, B3.25 · fiche §203.8 · tout le chapitre", thread="Holmes", tracks="C",
       body=f"""En B3.20, 600 itérations ont donné environ 1,59 nat par caractère sur la validation de Holmes. Le défi : au plus **{GOAL_TEXT["full"]}** nats par caractère, avec **au plus 1 000 000 de valeurs apprises** et **10 minutes** de CPU. La solution de référence y arrive en {REF_TEXT["full"][0]} à {REF_TEXT["full"][1]} selon la graine ; l'objectif lui laisse 5 % de marge.

Écris `method_30(train_ids, vocab_size, seconds)` : elle reçoit les ids du texte d'entraînement de Holmes (le découpage de B3.20, un tenseur 1-D), la taille du vocabulaire (88) et son budget de temps en secondes ; elle construit et entraîne **dans ce budget** un modèle neuf, puis le rend. Le modèle rendu : un `nn.Module`, `model(idx)` → des logits `(N, T, vocab_size)` pour des ids `(N, T)` avec $T$ au plus `model.block_size`, un attribut entier `block_size` de 1 à 512. Tout est permis : ton mini-GPT, une autre forme ($d$, blocs, têtes, contexte), d'autres réglages, un autre modèle. `evaluate_30(method_30)` (cellule suivante) vérifie quatre règles :
1. **1 000 000 de valeurs apprises au plus**, paramètres et buffers du modèle rendu, chaque tenseur partagé compté une fois ;
2. **un modèle neuf, à froid** : il n'existait pas avant l'appel et n'a les poids d'aucun modèle du notebook ; pendant la course, le dossier des sauvegardes de `wb.gpt` est vide et le modèle de référence indisponible ;
3. **rien de la validation** : ta méthode n'apprend que sur les ids qu'elle reçoit. Pendant la course, les variables globales qui contiennent la validation (`DATA` et les tableaux d'ids des exercices précédents) valent un leurre, et `wb.text`, `wb.datasets` et `wb.gpt` ne rendent que le texte d'entraînement ou refusent. Puis un **contrôle** : la même méthode, avec le quart du temps et les 8/9 du texte d'entraînement seulement, est mesurée sur la validation et sur le dernier neuvième, tenu à l'écart ; les deux loss doivent différer comme pour la solution de référence ;
4. **dans le temps** : 10 minutes pour la course (et 2 min 30 pour le contrôle), mesurées à froid, entraînement compris : ta méthode doit respecter son argument `seconds`.

Puis elle affiche ton score : la loss par caractère sur la validation, chaque caractère à partir de l'indice 512 prédit avec le contexte de ton modèle. Règle d'honneur : la validation sert à mesurer, jamais à apprendre ni à choisir à la main des poids.

**En mode rapide**, la course est réduite à 2 minutes (30 s pour le contrôle) et l'objectif est **{GOAL_TEXT["fast"]}** (référence {REF_TEXT["fast"][0]} à {REF_TEXT["fast"][1]}) : de quoi vérifier ta méthode. Le défi lui-même se court avec `FAST_MODE = False` (environ 13 minutes avec le contrôle, sur CPU).

Dans tes notes : tes idées, leur gain sur la validation et leur temps ; ce qu'apportent les courbes de B3.21 et les ablations de B3.25 ; ce que tu ferais d'une heure de CPU au lieu de 10 minutes ; et pourquoi une meilleure perplexité ne garantit pas un texte généré plus agréable à lire.""",
       given=GIVEN_30, todo=C.TODO_30, check=C.CHECK_30, solution=C.SOLUTION_30 + "\n\n\n" + C.CHECK_30),
])

PARTS2 = [PART_C, PART_D, PART_E]
